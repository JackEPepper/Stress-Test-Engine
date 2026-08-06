"""Engine orchestration."""

from __future__ import annotations

import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import openpyxl
import pandas as pd

from .borrower import (
    build_borrowers,
    build_source_reconciliation,
    enrich_borrowers,
    record_best_available_fallbacks,
    record_identity_data_issues,
    split_identity_balance_scope,
)
from .cecl import attach_cecl_reserve_basis, cecl_history_frame
from .comparison import build_comparison_report
from .config import output_dir_for, validate_scenario
from .exceptions import exception_frame, record_exception
from .io import load_inputs, metadata_for_inputs, write_csv, write_json
from .modules.base import initialize_results
from .modules.ci import run_ci
from .modules.consumer import run_consumer
from .modules.cre import run_cre
from .progress import ProgressReporter, ProgressStep
from .reporting import build_reports
from .tagging import (
    add_cecl_selection_summary,
    apply_tags,
    assign_primary_modules,
    resolve_cecl_level_tags,
)
from .targeted import run_targeted_stress, targeted_enabled
from .utils import hash_json
from .version import VERSION


OUTPUT_MANIFEST_KIND = "stress_engine_outputs"


class StressEngine:
    """Run the full scenario pipeline from raw inputs through final reports.

    Called by the CLI (`stress_engine.cli.main`) and batch runner. This class is
    the only orchestrator; stress modules and reporting functions are
    intentionally pure transformations of DataFrames.
    """

    def __init__(self, scenario: Mapping[str, Any], base_dir: str | Path):
        """Validate and retain one merged scenario and its path anchor."""
        self.scenario = dict(scenario)
        self.base_dir = Path(base_dir).resolve()
        validate_scenario(self.scenario)

    def run(
        self,
        output_dir: str | Path | None = None,
        write_outputs: bool = True,
        run_comparison: bool = True,
        progress: ProgressReporter | None = None,
    ) -> Dict[str, Any]:
        """Execute one standard or targeted scenario and optionally write outputs."""
        reporter = progress if progress is not None else ProgressReporter()
        previous = (
            _previous_scenarios(self.scenario, self.base_dir)
            if run_comparison
            else []
        )
        targeted_mode = targeted_enabled(self.scenario)
        reporter.start(
            self._progress_title(targeted_mode),
            self._progress_steps(
                targeted_mode=targeted_mode,
                previous_count=len(previous),
                write_outputs=write_outputs,
            ),
        )
        exceptions = []

        # 1. Load every configured CSV/XLSX once. `load_inputs` also profiles
        # each table so input statistics can be reported without re-reading.
        with reporter.step("inputs"):
            raw_loaded = load_inputs(self.scenario, self.base_dir)
            input_summary = pd.concat(
                [item.profile for item in raw_loaded.values()], ignore_index=True
            )
            source_rows = sum(len(item.frame) for item in raw_loaded.values())
            reporter.update(
                f"Loaded {len(raw_loaded)} input tables with {source_rows:,} source rows."
            )

        # 2. Build the borrower universe from the identity file, then use tags
        # to derive model populations, CECL portfolios, and reconciliation tags.
        with reporter.step("population"):
            identity = raw_loaded["identity"].frame
            record_identity_data_issues(identity, self.scenario, exceptions)
            identity, input_balance_out_of_scope = split_identity_balance_scope(
                identity, self.scenario
            )
            loaded = dict(raw_loaded)
            loaded["identity"] = replace(raw_loaded["identity"], frame=identity)
            borrowers = build_borrowers(identity, self.scenario, exceptions)
            borrowers, tag_summary = apply_tags(
                borrowers, self.scenario, loaded, exceptions
            )
            borrowers = assign_primary_modules(
                borrowers, self.scenario, exceptions
            )
            borrowers = resolve_cecl_level_tags(
                borrowers,
                self.scenario,
                exceptions,
                emit_priority_warnings=not targeted_mode,
            )
            tag_summary = add_cecl_selection_summary(
                tag_summary, borrowers, self.scenario
            )
            excluded_count = int(
                borrowers.get("model_excluded", pd.Series(False, index=borrowers.index))
                .fillna(False)
                .astype(bool)
                .sum()
            )
            reporter.update(
                f"Built {len(borrowers):,} borrowers; {excluded_count:,} model-excluded."
            )

        # 3. Enrichment runs after tagging so sources can be restricted by tag
        # if the scenario requests it. The audit copy is the post-tag/enriched
        # borrower state before any stress formulas mutate results.
        with reporter.step("enrichment"):
            borrowers = enrich_borrowers(borrowers, loaded, self.scenario)
            record_best_available_fallbacks(
                borrowers, self.scenario, exceptions
            )
            source_reconciliation = build_source_reconciliation(
                borrowers, loaded, self.scenario, exceptions
            )
            audit_borrowers = borrowers.copy()
            reporter.update(
                f"Reconciled {len(source_reconciliation):,} configured sources."
            )

        # Targeted mode changes the result grain to loan/variant and builds its
        # own per-variant module reports. Shared input and reconciliation controls
        # are attached afterward so both execution modes expose the same audits.
        if targeted_mode:
            with reporter.step("stress"):
                targeted = run_targeted_stress(
                    self.scenario,
                    loaded,
                    exceptions,
                    input_out_of_scope=input_balance_out_of_scope,
                    progress=reporter,
                )
            with reporter.step("reports"):
                reports = targeted["reports"]
                reports["input_summary"] = input_summary
                reports["tag_summary"] = targeted["tag_summary"]
                reports["source_reconciliation"] = source_reconciliation
                reports["out_of_scope_detail"] = targeted["out_of_scope"]
                reports["exception_log"] = exception_frame(
                    exceptions, ["scenario_variant", "loan_id"]
                )
                reporter.update(
                    f"Assembled {len(reports):,} report tables and control files."
                )
            if previous:
                with reporter.step("comparison"):
                    reports["scenario_diff"] = build_comparison_report(
                        self.scenario,
                        reports,
                        previous,
                        max_variable_reruns=self.scenario.get("comparison", {}).get(
                            "max_variable_reruns"
                        ),
                    )
                    comparison_errors = _record_comparison_errors(
                        reports["scenario_diff"], exceptions
                    )
                    reports["exception_log"] = exception_frame(
                        exceptions, ["scenario_variant", "loan_id"]
                    )
                    reporter.update(
                        _comparison_progress_message(
                            len(previous), comparison_errors
                        )
                    )
            with reporter.step("metadata"):
                metadata = self._metadata(raw_loaded, reports)
                metadata["targeted_stress"] = {
                    "enabled": True,
                    "primary_variant": targeted["primary_variant"],
                    "variants": targeted["variant_names"],
                    "result_grain": "loan",
                }
                metadata["output_hashes"]["stressed_loan_results"] = hash_json(
                    targeted["variant_results"].fillna("").to_dict(orient="records")
                )
            if write_outputs:
                with reporter.step("outputs"):
                    destination = output_dir_for(
                        self.scenario, self.base_dir, output_dir
                    )
                    self._write_targeted_outputs(
                        destination,
                        audit_borrowers,
                        targeted["variant_results"],
                        reports,
                        metadata,
                    )
                    reporter.update(
                        f"Wrote {len(reports) + 5:,} audit artifacts."
                    )
            reporter.finish("Targeted stress run complete")
            return {
                "borrowers": audit_borrowers,
                "loan_context": targeted["context"],
                "results": targeted["results"],
                "variant_results": targeted["variant_results"],
                "reports": reports,
                "metadata": metadata,
            }

        # 4. Stress modules share a results frame initialized with base buckets.
        # `module_population` inside each module enforces the resolved
        # `primary_module`, preventing double-stress when tags overlap.
        with reporter.step("stress"):
            results = initialize_results(borrowers, self.scenario, exceptions)
            results, reserve_basis = attach_cecl_reserve_basis(
                results,
                self.scenario,
                exceptions,
                history=cecl_history_frame(self.scenario, loaded),
            )
            out_of_scope_frames = (
                [input_balance_out_of_scope]
                if not input_balance_out_of_scope.empty
                else []
            )
            module_order = self.scenario.get(
                "module_order", ["CRE", "C&I", "Consumer"]
            )
            for index, module_name in enumerate(module_order, start=1):
                if module_name == "CRE":
                    results, out = run_cre(results, self.scenario, exceptions)
                elif module_name == "C&I":
                    results, out = run_ci(results, self.scenario, exceptions)
                elif module_name == "Consumer":
                    results, out = run_consumer(
                        results, self.scenario, loaded, exceptions
                    )
                else:
                    raise ValueError(
                        f"Unsupported module in module_order: {module_name}"
                    )
                if out is not None and not out.empty:
                    out_of_scope_frames.append(out)
                reporter.update(
                    f"Completed {module_name} module ({index}/{len(module_order)}).",
                    completed=index,
                    total=len(module_order),
                )
            out_of_scope = (
                pd.concat(out_of_scope_frames, ignore_index=True)
                if out_of_scope_frames
                else pd.DataFrame()
            )

        # 5. Reporting consumes stressed results plus audit borrowers. CECL and
        # overlay warnings append to the shared exception list for one final log.
        with reporter.step("reports"):
            reports = build_reports(
                results,
                audit_borrowers,
                self.scenario,
                out_of_scope,
                exceptions,
                reserve_basis,
            )
            reports["input_summary"] = input_summary
            reports["tag_summary"] = tag_summary
            reports["source_reconciliation"] = source_reconciliation
            reports["out_of_scope_detail"] = out_of_scope
            reports["exception_log"] = exception_frame(exceptions)
            reporter.update(
                f"Built {len(reports):,} report tables and control files."
            )

        # 6. Optional scenario comparison reruns prior scenarios and variable
        # perturbations. It is deliberately last so it can compare final reports.
        if previous:
            with reporter.step("comparison"):
                reports["scenario_diff"] = build_comparison_report(
                    self.scenario,
                    reports,
                    previous,
                    max_variable_reruns=self.scenario.get("comparison", {}).get(
                        "max_variable_reruns"
                    ),
                )
                comparison_errors = _record_comparison_errors(
                    reports["scenario_diff"], exceptions
                )
                reports["exception_log"] = exception_frame(exceptions)
                reporter.update(
                    _comparison_progress_message(
                        len(previous), comparison_errors
                    )
                )

        with reporter.step("metadata"):
            # Metadata and hashes are deliberately last: comparison failures may
            # add control rows that must be reflected in deterministic report hashes.
            metadata = self._metadata(raw_loaded, reports)
        if write_outputs:
            with reporter.step("outputs"):
                destination = output_dir_for(
                    self.scenario, self.base_dir, output_dir
                )
                self._write_outputs(
                    destination, audit_borrowers, results, reports, metadata
                )
                reporter.update(
                    f"Wrote {len(reports) + 5:,} audit artifacts."
                )

        reporter.finish("Stress run complete")
        return {
            "borrowers": audit_borrowers,
            "results": results,
            "reports": reports,
            "metadata": metadata,
        }

    def _progress_title(self, targeted_mode: bool) -> str:
        """Describe the current scenario and execution mode for the terminal."""
        scenario_id = str(self.scenario.get("scenario_id", "unnamed scenario"))
        if targeted_mode:
            variants = self.scenario.get("targeted_stress", {}).get("variants", {})
            variant_count = len(variants) + 1 if isinstance(variants, Mapping) else 1
            return f"{scenario_id} | targeted run | {variant_count} variants"
        return f"{scenario_id} | standard run"

    def _progress_steps(
        self,
        *,
        targeted_mode: bool,
        previous_count: int,
        write_outputs: bool,
    ) -> list[ProgressStep]:
        """Build the visible run plan without changing calculation behavior."""
        module_count = len(
            self.scenario.get("module_order", ["CRE", "C&I", "Consumer"])
        )
        variant_count = 1
        stress_label = "Apply configured stress modules"
        stress_weight = max(module_count, 1) * 2.0
        # Weights express relative work rather than promised wall-clock time;
        # observed durations continuously recalibrate the displayed ETA.
        if targeted_mode:
            variants = self.scenario.get("targeted_stress", {}).get("variants", {})
            variant_count += len(variants) if isinstance(variants, Mapping) else 0
            stress_label = f"Run baseline and {variant_count - 1} targeted variants"
            stress_weight = variant_count * 3.0

        steps = [
            ProgressStep("inputs", "Load and profile input tables", 3.0),
            ProgressStep(
                "population", "Build population and resolve model tags", 3.0
            ),
            ProgressStep(
                "enrichment", "Enrich borrowers and reconcile sources", 2.5
            ),
            ProgressStep("stress", stress_label, stress_weight),
            ProgressStep("reports", "Build reports and control files", 3.0),
        ]
        if previous_count:
            scenario_file_label = (
                "prior scenario file"
                if previous_count == 1
                else "prior scenario files"
            )
            steps.append(
                ProgressStep(
                    "comparison",
                    f"Compare with {previous_count} {scenario_file_label}",
                    previous_count * 4.0,
                )
            )
        steps.append(ProgressStep("metadata", "Finalize audit metadata", 1.0))
        if write_outputs:
            steps.append(ProgressStep("outputs", "Write output artifacts", 3.0))
        return steps

    def _metadata(
        self,
        loaded: Mapping[str, Any],
        reports: Mapping[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Build audit metadata after all reports exist.

        Called once by `run`. Output hashes are calculated from report rows so
        users can detect changes in reports even when filenames are identical.
        """
        output_hashes = {}
        for name, frame in reports.items():
            if isinstance(frame, pd.DataFrame):
                output_hashes[name] = hash_json(frame.fillna("").to_dict(orient="records"))
        exception_log = reports.get("exception_log", pd.DataFrame())
        severity_counts = (
            exception_log["severity"].value_counts().sort_index().to_dict()
            if isinstance(exception_log, pd.DataFrame) and "severity" in exception_log.columns
            else {}
        )
        return {
            "engine_version": VERSION,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "openpyxl_version": openpyxl.__version__,
            "scenario_files": self.scenario.get("_metadata", {}).get("scenario_files", []),
            "scenario_hash": self.scenario.get("_metadata", {}).get(
                "scenario_hash", hash_json(self.scenario)
            ),
            "stress_levels": self.scenario.get("stress_levels", ["S1", "S2"]),
            "input_files": metadata_for_inputs(loaded),
            "exception_count": int(len(exception_log)),
            "exception_counts_by_severity": {
                str(key): int(value) for key, value in severity_counts.items()
            },
            "output_hashes": output_hashes,
        }

    def _write_outputs(
        self,
        output_dir: Path,
        audit_borrowers: pd.DataFrame,
        results: pd.DataFrame,
        reports: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any],
    ) -> None:
        """Write audit, result, report, metadata, and scenario-used artifacts.

        Called only when `run(write_outputs=True)`. Sorting is centralized here
        to make repeated runs stable even when DataFrame group order varies.
        """
        borrower_id = self.scenario["borrower"]["borrower_id_field"]
        output_dir.mkdir(parents=True, exist_ok=True)
        # Targeted output preserves variant/loan grain, while its report files
        # use the same manifest ownership and stale-file cleanup as legacy mode.
        current_files = {
            "borrower_audit_raw.csv",
            "stressed_borrower_results.csv",
            "metadata.json",
            "scenario_used.json",
        }
        write_csv(audit_borrowers, output_dir / "borrower_audit_raw.csv", [borrower_id])
        write_csv(results, output_dir / "stressed_borrower_results.csv", [borrower_id])
        for name, frame in reports.items():
            if isinstance(frame, pd.DataFrame):
                sort_cols = _sort_columns(name, frame)
                filename = f"{name}.csv"
                current_files.add(filename)
                write_csv(frame, output_dir / filename, sort_cols)
        write_json(metadata, output_dir / "metadata.json")
        write_json(_scenario_for_audit(self.scenario), output_dir / "scenario_used.json")
        _remove_stale_outputs(output_dir, current_files)
        write_json(
            {
                "engine_version": VERSION,
                "kind": OUTPUT_MANIFEST_KIND,
                "files": sorted(current_files | {"output_manifest.json"}),
            },
            output_dir / "output_manifest.json",
        )

    def _write_targeted_outputs(
        self,
        output_dir: Path,
        audit_borrowers: pd.DataFrame,
        variant_results: pd.DataFrame,
        reports: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any],
    ) -> None:
        """Write loan-grain targeted variants without changing legacy mode files."""
        borrower_id = self.scenario["borrower"]["borrower_id_field"]
        loan_id = self.scenario["borrower"].get("loan_id_field", "loan_id")
        output_dir.mkdir(parents=True, exist_ok=True)
        current_files = {
            "borrower_audit_raw.csv",
            "stressed_loan_results.csv",
            "metadata.json",
            "scenario_used.json",
        }
        write_csv(audit_borrowers, output_dir / "borrower_audit_raw.csv", [borrower_id])
        write_csv(
            variant_results,
            output_dir / "stressed_loan_results.csv",
            ["scenario_variant", borrower_id, loan_id, "_exposure_id"],
        )
        for name, frame in reports.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            filename = f"{name}.csv"
            current_files.add(filename)
            # Sort by whichever audit dimensions each report exposes so output
            # hashes remain stable without imposing one schema on every report.
            sort_cols = [
                column
                for column in [
                    "scenario_variant",
                    "shock_order",
                    "operation_sequence",
                    "shock",
                    "tier",
                    "portfolio",
                    "stress_level",
                    "bucket",
                    "period",
                    borrower_id,
                    loan_id,
                    "_exposure_id",
                ]
                if column in frame.columns
            ]
            write_csv(frame, output_dir / filename, sort_cols)
        write_json(metadata, output_dir / "metadata.json")
        write_json(_scenario_for_audit(self.scenario), output_dir / "scenario_used.json")
        _remove_stale_outputs(output_dir, current_files)
        write_json(
            {
                "engine_version": VERSION,
                "kind": OUTPUT_MANIFEST_KIND,
                "files": sorted(current_files | {"output_manifest.json"}),
            },
            output_dir / "output_manifest.json",
        )


def _previous_scenarios(
    scenario: Mapping[str, Any], base_dir: Path
) -> list[str]:
    """Resolve configured comparison paths relative to the scenario folder."""
    comparison = scenario.get("comparison") or {}
    if not isinstance(comparison, Mapping):
        raise ValueError("Scenario comparison must be a JSON object.")
    previous = comparison.get("previous_scenarios", [])
    if previous is None:
        return []
    if isinstance(previous, (str, Path)):
        previous = [previous]
    elif not isinstance(previous, list):
        raise ValueError(
            "Scenario comparison.previous_scenarios must be a string or list."
        )
    resolved = []
    for item in previous:
        if not isinstance(item, (str, Path)) or not str(item).strip():
            raise ValueError(
                "Scenario comparison.previous_scenarios entries must be "
                "nonblank paths."
            )
        path = Path(str(item))
        resolved.append(
            str(path.resolve() if path.is_absolute() else (base_dir / path).resolve())
        )
    return resolved


def _record_comparison_errors(
    report: pd.DataFrame,
    exceptions: list[Dict[str, Any]],
) -> int:
    """Promote comparison rerun failures into the primary control log."""
    if report.empty or "metric" not in report.columns:
        return 0
    failures = report[
        report["metric"].isin({"load_or_run_error", "rerun_error"})
    ]
    for row in failures.to_dict(orient="records"):
        record_exception(
            exceptions,
            "ERROR",
            "comparison",
            "SCENARIO_COMPARISON_FAILED",
            "A prior-scenario or marginal comparison rerun could not complete.",
            source=row.get("previous_scenario", ""),
            field=row.get("change_path", ""),
            details=row.get("notes", ""),
        )
    return int(len(failures))


def _comparison_progress_message(
    previous_count: int,
    error_count: int,
) -> str:
    """Summarize comparison completion without hiding captured control errors."""
    if error_count:
        suffix = "error" if error_count == 1 else "errors"
        return (
            f"Comparison finished with {error_count} {suffix}; review "
            "scenario_diff and exception_log."
        )
    scenario_file_label = (
        "prior scenario file" if previous_count == 1 else "prior scenario files"
    )
    return f"Compared against {previous_count:,} {scenario_file_label}."


def _sort_columns(name: str, frame: pd.DataFrame) -> list[str]:
    """Return stable output sort columns for each report type."""
    candidates = {
        "migration_summary": ["portfolio", "stress_level", "bucket"],
        "cecl_summary": ["portfolio", "stress_level", "bucket"],
        "cecl_basis_summary": ["portfolio", "bucket", "period"],
        "tag_summary": ["tag", "tie_out_name"],
        "out_of_scope_detail": ["module", "stress_level", "borrower_id", "field"],
        "out_of_scope_summary": ["module", "stress_level", "test", "field"],
        "input_summary": ["dataset", "field"],
        "source_reconciliation": ["source"],
        "scenario_diff": [
            "previous_scenario",
            "change_kind",
            "change_path",
            "portfolio",
            "stress_level",
        ],
        "exception_log": ["severity", "stage", "code", "portfolio", "stress_level", "bucket"],
    }
    return [column for column in candidates.get(name, []) if column in frame.columns]


def _scenario_for_audit(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop runtime-only metadata before writing the scenario audit copy."""
    return {key: value for key, value in scenario.items() if key != "_metadata"}


def _remove_stale_outputs(output_dir: Path, current_files: set[str]) -> None:
    """Remove only prior engine-owned files that are absent from the current run."""
    prior_files: set[str] = set()
    manifest = output_dir / "output_manifest.json"
    if manifest.exists():
        try:
            with manifest.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            files = (
                payload.get("files", [])
                if isinstance(payload, Mapping)
                and payload.get("kind") == OUTPUT_MANIFEST_KIND
                else []
            )
            if isinstance(files, list):
                prior_files.update(
                    filename
                    for filename in files
                    if isinstance(filename, str) and Path(filename).name == filename
                )
        except (OSError, ValueError, TypeError):
            pass
    for filename in sorted(prior_files - current_files - {"output_manifest.json"}):
        path = output_dir / filename
        if path.is_file():
            path.unlink()
