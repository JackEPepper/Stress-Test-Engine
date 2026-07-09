"""Engine orchestration."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from .borrower import build_borrowers, enrich_borrowers
from .comparison import build_comparison_report
from .config import load_scenario, output_dir_for
from .io import load_inputs, metadata_for_inputs, write_csv, write_json
from .modules.base import initialize_results
from .modules.ci import run_ci
from .modules.consumer import run_consumer
from .modules.cre import run_cre
from .reporting import build_reports
from .tagging import apply_tags
from .utils import hash_json


ENGINE_VERSION = "0.1.0"


class StressEngine:
    """Run a JSON-defined stress scenario against configured input tables."""

    def __init__(self, scenario: Mapping[str, Any], base_dir: str | Path):
        self.scenario = dict(scenario)
        self.base_dir = Path(base_dir).resolve()

    def run(
        self,
        output_dir: str | Path | None = None,
        write_outputs: bool = True,
        run_comparison: bool = True,
    ) -> Dict[str, Any]:
        loaded = load_inputs(self.scenario, self.base_dir)
        input_summary = pd.concat([item.profile for item in loaded.values()], ignore_index=True)

        identity = loaded["identity"].frame
        borrowers = build_borrowers(identity, self.scenario)
        borrowers, tag_summary = apply_tags(borrowers, self.scenario, loaded)
        borrowers = enrich_borrowers(borrowers, loaded, self.scenario)
        audit_borrowers = borrowers.copy()

        results = initialize_results(borrowers, self.scenario)
        out_of_scope_frames = []
        module_order = self.scenario.get("module_order", ["CRE", "C&I", "Consumer"])
        for module_name in module_order:
            raw_name = str(module_name).lower()
            normalized = raw_name.replace("&", "and").replace(" ", "_")
            if normalized in {"cre", "commercial real estate", "commercial_real_estate"}:
                results, out = run_cre(results, self.scenario)
            elif raw_name in {"c&i", "ci"} or normalized in {
                "candi",
                "commercial_and_industrial",
                "commercial_industrial",
            }:
                results, out = run_ci(results, self.scenario)
            elif normalized == "consumer":
                results, out = run_consumer(results, self.scenario, loaded)
            else:
                continue
            if out is not None and not out.empty:
                out_of_scope_frames.append(out)
        out_of_scope = pd.concat(out_of_scope_frames, ignore_index=True) if out_of_scope_frames else pd.DataFrame()

        reports = build_reports(results, audit_borrowers, self.scenario, out_of_scope)
        reports["input_summary"] = input_summary
        reports["tag_summary"] = tag_summary
        reports["out_of_scope_detail"] = out_of_scope

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
        output_hashes = {}
        for name, frame in reports.items():
            if isinstance(frame, pd.DataFrame):
                output_hashes[name] = hash_json(frame.fillna("").to_dict(orient="records"))
        return {
            "engine_version": ENGINE_VERSION,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "scenario_files": self.scenario.get("_metadata", {}).get("scenario_files", []),
            "scenario_hash": self.scenario.get("_metadata", {}).get("scenario_hash", hash_json(self.scenario)),
            "stress_levels": self.scenario.get("stress_levels", ["S1", "S2"]),
            "input_files": metadata_for_inputs(loaded),
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
        borrower_id = self.scenario["borrower"]["borrower_id_field"]
        write_csv(audit_borrowers, output_dir / "borrower_audit_raw.csv", [borrower_id])
        write_csv(results, output_dir / "stressed_borrower_results.csv", [borrower_id])
        for name, frame in reports.items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                sort_cols = _sort_columns(name, frame)
                write_csv(frame, output_dir / f"{name}.csv", sort_cols)
        write_json(metadata, output_dir / "metadata.json")
        write_json(_scenario_for_audit(self.scenario), output_dir / "scenario_used.json")


def run_scenario(
    scenario_paths: str | Path | Iterable[str | Path],
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    scenario, base_dir = load_scenario(scenario_paths)
    return StressEngine(scenario, base_dir).run(output_dir=output_dir, write_outputs=write_outputs)


def _previous_scenarios(scenario: Mapping[str, Any]) -> list[str]:
    comparison = scenario.get("comparison", {})
    previous = comparison.get("previous_scenarios", comparison.get("previous_scenario", []))
    if previous is None:
        return []
    if isinstance(previous, (str, Path)):
        return [str(previous)]
    return [str(item) for item in previous]


def _sort_columns(name: str, frame: pd.DataFrame) -> list[str]:
    candidates = {
        "migration_summary": ["portfolio", "stress_level", "bucket"],
        "cecl_summary": ["portfolio", "stress_level", "bucket"],
        "tag_summary": ["tag", "tie_out_name"],
        "out_of_scope_detail": ["module", "stress_level", "borrower_id", "field"],
        "out_of_scope_summary": ["module", "stress_level", "test", "field"],
        "input_summary": ["dataset", "field"],
        "scenario_diff": ["previous_scenario", "change_kind", "change_path", "portfolio", "stress_level"],
    }
    return [column for column in candidates.get(name, []) if column in frame.columns]


def _scenario_for_audit(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in scenario.items() if key != "_metadata"}
