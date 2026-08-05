"""Engine orchestration."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from .borrower import (
    build_borrowers,
    build_source_reconciliation,
    enrich_borrowers,
    record_best_available_fallbacks,
    record_identity_data_issues,
)
from .cecl import attach_cecl_reserve_basis, cecl_history_frame
from .comparison import build_comparison_report
from .config import output_dir_for, validate_scenario
from .exceptions import exception_frame
from .io import load_inputs, metadata_for_inputs, write_csv, write_json
from .modules.base import initialize_results
from .modules.ci import run_ci
from .modules.consumer import run_consumer
from .modules.cre import run_cre
from .reporting import build_reports
from .tagging import (
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
        self.scenario = dict(scenario)
        self.base_dir = Path(base_dir).resolve()
        validate_scenario(self.scenario)

    def run(
        self,
        output_dir: str | Path | None = None,
        write_outputs: bool = True,
        run_comparison: bool = True,
    ) -> Dict[str, Any]:
        exceptions = []

        # 1. Load every configured CSV/XLSX once. `load_inputs` also profiles
        # each table so input statistics can be reported without re-reading.
        loaded = load_inputs(self.scenario, self.base_dir)
        input_summary = pd.concat([item.profile for item in loaded.values()], ignore_index=True)

        # 2. Build the borrower universe from the identity file, then use tags
        # to derive model populations, CECL portfolios, and reconciliation tags.
        identity = loaded["identity"].frame
        record_identity_data_issues(identity, self.scenario, exceptions)
        borrowers = build_borrowers(identity, self.scenario, exceptions)
        borrowers, tag_summary = apply_tags(borrowers, self.scenario, loaded, exceptions)
        borrowers = assign_primary_modules(borrowers, self.scenario, exceptions)
        borrowers = resolve_cecl_level_tags(borrowers, self.scenario)

        # 3. Enrichment runs after tagging so sources can be restricted by tag
        # if the scenario requests it. The audit copy is the post-tag/enriched
        # borrower state before any stress formulas mutate results.
        borrowers = enrich_borrowers(borrowers, loaded, self.scenario)
        record_best_available_fallbacks(borrowers, self.scenario, exceptions)
        source_reconciliation = build_source_reconciliation(borrowers, loaded, self.scenario, exceptions)
        audit_borrowers = borrowers.copy()

        if targeted_enabled(self.scenario):
            targeted = run_targeted_stress(self.scenario, loaded, exceptions)
            reports = targeted["reports"]
            reports["input_summary"] = input_summary
            reports["tag_summary"] = targeted["tag_summary"]
            reports["source_reconciliation"] = source_reconciliation
            reports["out_of_scope_detail"] = targeted["out_of_scope"]
            reports["exception_log"] = exception_frame(
                exceptions, ["scenario_variant", "loan_id"]
            )
            if run_comparison:
                previous = _previous_scenarios(self.scenario)
                if previous:
                    reports["scenario_diff"] = build_comparison_report(
                        self.scenario,
                        reports,
                        previous,
                        max_variable_reruns=self.scenario.get("comparison", {}).get(
                            "max_variable_reruns"
                        ),
                    )
            metadata = self._metadata(loaded, reports)
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
        results = initialize_results(borrowers, self.scenario, exceptions)
        results, reserve_basis = attach_cecl_reserve_basis(
            results,
            self.scenario,
            exceptions,
            history=cecl_history_frame(self.scenario, loaded),
        )
        out_of_scope_frames = []
        module_order = self.scenario.get("module_order", ["CRE", "C&I", "Consumer"])
        for module_name in module_order:
            if module_name == "CRE":
                results, out = run_cre(results, self.scenario, exceptions)
            elif module_name == "C&I":
                results, out = run_ci(results, self.scenario, exceptions)
            elif module_name == "Consumer":
                results, out = run_consumer(results, self.scenario, loaded, exceptions)
            else:
                raise ValueError(f"Unsupported module in module_order: {module_name}")
            if out is not None and not out.empty:
                out_of_scope_frames.append(out)
        out_of_scope = pd.concat(out_of_scope_frames, ignore_index=True) if out_of_scope_frames else pd.DataFrame()

        # 5. Reporting consumes stressed results plus audit borrowers. CECL and
        # overlay warnings append to the shared exception list for one final log.
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

        # 6. Optional scenario comparison reruns prior scenarios and variable
        # perturbations. It is deliberately last so it can compare final reports.
        if run_comparison:
            previous = _previous_scenarios(self.scenario)
            if previous:
                reports["scenario_diff"] = build_comparison_report(
                    self.scenario,
                    reports,
                    previous,
                    max_variable_reruns=self.scenario.get("comparison", {}).get("max_variable_reruns"),
                )

        metadata = self._metadata(loaded, reports)
        if write_outputs:
            destination = output_dir_for(self.scenario, self.base_dir, output_dir)
            self._write_outputs(destination, audit_borrowers, results, reports, metadata)

        return {
            "borrowers": audit_borrowers,
            "results": results,
            "reports": reports,
            "metadata": metadata,
        }

    def _metadata(self, loaded: Mapping[str, Any], reports: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
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
            "scenario_files": self.scenario.get("_metadata", {}).get("scenario_files", []),
            "scenario_hash": self.scenario.get("_metadata", {}).get("scenario_hash", hash_json(self.scenario)),
            "stress_levels": self.scenario.get("stress_levels", ["S1", "S2"]),
            "input_files": metadata_for_inputs(loaded),
            "exception_count": int(len(exception_log)),
            "exception_counts_by_severity": {str(key): int(value) for key, value in severity_counts.items()},
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
        current_files = {"borrower_audit_raw.csv", "stressed_borrower_results.csv", "metadata.json", "scenario_used.json"}
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


def _previous_scenarios(scenario: Mapping[str, Any]) -> list[str]:
    """Normalize optional comparison scenario config to a list of paths."""
    comparison = scenario.get("comparison", {})
    previous = comparison.get("previous_scenarios", [])
    if previous is None:
        return []
    if isinstance(previous, (str, Path)):
        return [str(previous)]
    return [str(item) for item in previous]


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
        "scenario_diff": ["previous_scenario", "change_kind", "change_path", "portfolio", "stress_level"],
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
