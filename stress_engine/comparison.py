"""Previous-scenario comparison and marginal-impact reporting."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

from .config import load_scenario
from .utils import flatten_json, set_json_path


IGNORED_SCENARIO_PREFIXES = ("_metadata", "outputs.directory", "comparison")
DATA_PREFIXES = ("inputs",)


def build_comparison_report(
    current_scenario: Mapping[str, Any],
    current_reports: Mapping[str, pd.DataFrame],
    previous_scenarios: Iterable[str | Path],
    max_variable_reruns: int | None = None,
) -> pd.DataFrame:
    """Compare current results to one or more previous scenarios.

    Called from `StressEngine.run` after current reports are complete. It first
    reruns the previous scenario, then reruns one changed variable at a time to
    approximate marginal CECL impact.
    """
    previous_paths = [Path(path) for path in previous_scenarios]
    if not previous_paths:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    from .engine import StressEngine

    for previous_path in previous_paths:
        try:
            previous_scenario, previous_base_dir = load_scenario(previous_path)
            previous_scenario = _without_comparison(previous_scenario)
            previous_result = StressEngine(previous_scenario, previous_base_dir).run(write_outputs=False, run_comparison=False)
            previous_reports = previous_result["reports"]
            rows.extend(
                _data_change_rows(
                    str(previous_path),
                    previous_reports,
                    current_reports,
                    previous_reports.get("cecl_summary", pd.DataFrame()),
                    current_reports.get("cecl_summary", pd.DataFrame()),
                )
            )

            diffs = _scenario_diffs(previous_scenario, current_scenario)
            variable_diffs = [item for item in diffs if not item["path"].startswith(DATA_PREFIXES)]
            if max_variable_reruns is not None:
                skipped = variable_diffs[max_variable_reruns:]
                variable_diffs = variable_diffs[:max_variable_reruns]
                for diff in skipped:
                    rows.append(
                        {
                            "previous_scenario": str(previous_path),
                            "change_kind": "scenario_variable_skipped",
                            "change_path": diff["path"],
                            "old_value": _stringify(diff["old_value"]),
                            "new_value": _stringify(diff["new_value"]),
                            "portfolio": "",
                            "stress_level": "",
                            "metric": "not_rerun",
                            "previous_value": "",
                            "changed_value": "",
                            "marginal_impact": "",
                            "notes": f"Skipped because max_variable_reruns={max_variable_reruns}.",
                        }
                    )
            for diff in variable_diffs:
                try:
                    mutated = set_json_path(previous_scenario, diff["path"], diff["new_value"])
                    mutated = _without_comparison(mutated)
                    mutated_result = StressEngine(mutated, previous_base_dir).run(write_outputs=False, run_comparison=False)
                    impact_rows = _cecl_impact_rows(
                        str(previous_path),
                        "scenario_variable",
                        diff["path"],
                        diff["old_value"],
                        diff["new_value"],
                        previous_reports.get("cecl_summary", pd.DataFrame()),
                        mutated_result["reports"].get("cecl_summary", pd.DataFrame()),
                    )
                    rows.extend(impact_rows)
                except Exception as exc:  # pragma: no cover - reportable audit artifact
                    rows.append(
                        {
                            "previous_scenario": str(previous_path),
                            "change_kind": "scenario_variable",
                            "change_path": diff["path"],
                            "old_value": _stringify(diff["old_value"]),
                            "new_value": _stringify(diff["new_value"]),
                            "portfolio": "ERROR",
                            "stress_level": "",
                            "metric": "rerun_error",
                            "previous_value": "",
                            "changed_value": "",
                            "marginal_impact": "",
                            "notes": str(exc),
                        }
                    )
        except Exception as exc:  # pragma: no cover - reportable audit artifact
            rows.append(
                {
                    "previous_scenario": str(previous_path),
                    "change_kind": "previous_scenario",
                    "change_path": "",
                    "old_value": "",
                    "new_value": "",
                    "portfolio": "ERROR",
                    "stress_level": "",
                    "metric": "load_or_run_error",
                    "previous_value": "",
                    "changed_value": "",
                    "marginal_impact": "",
                    "notes": str(exc),
                }
            )
    return pd.DataFrame(rows)


def _scenario_diffs(previous: Mapping[str, Any], current: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return flattened JSON paths where scenario values changed."""
    prev_flat = flatten_json(previous)
    curr_flat = flatten_json(current)
    paths = sorted(set(prev_flat) | set(curr_flat))
    diffs = []
    for path in paths:
        if path.startswith(IGNORED_SCENARIO_PREFIXES):
            continue
        old_value = prev_flat.get(path)
        new_value = curr_flat.get(path)
        if _stringify(old_value) != _stringify(new_value):
            diffs.append({"path": path, "old_value": old_value, "new_value": new_value})
    return diffs


def _data_change_rows(
    previous_path: str,
    previous_reports: Mapping[str, pd.DataFrame],
    current_reports: Mapping[str, pd.DataFrame],
    previous_cecl: pd.DataFrame,
    current_cecl: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Report aggregate input-profile changes between two completed runs."""
    prev = previous_reports.get("input_summary", pd.DataFrame())
    curr = current_reports.get("input_summary", pd.DataFrame())
    if prev.empty or curr.empty:
        return []
    merged = prev.merge(
        curr,
        on=["dataset", "field"],
        how="outer",
        suffixes=("_previous", "_current"),
    )
    changes = []
    for _, row in merged.iterrows():
        for metric in ("row_count", "non_null_count", "missing_count", "unique_count", "numeric_sum"):
            old = row.get(f"{metric}_previous")
            new = row.get(f"{metric}_current")
            if _stringify(old) != _stringify(new):
                changes.append((f"data.{row.get('dataset')}.{row.get('field')}.{metric}", old, new))
    impact = _cecl_impact_rows(
        previous_path,
        "data_aggregate",
        "all_loaded_data",
        "",
        "",
        previous_cecl,
        current_cecl,
    )
    rows: List[Dict[str, Any]] = []
    for item in impact:
        cloned = dict(item)
        cloned["notes"] = f"Aggregate effect of all data changes; changed_profile_metric_count={len(changes)}."
        rows.append(cloned)
    for path, old, new in changes:
        rows.append(
            {
                "previous_scenario": previous_path,
                "change_kind": "data_profile",
                "change_path": path,
                "old_value": _stringify(old),
                "new_value": _stringify(new),
                "portfolio": "",
                "stress_level": "",
                "metric": "input_profile_change",
                "previous_value": old,
                "changed_value": new,
                "marginal_impact": "",
                "notes": "Descriptive input change only; no standalone CECL impact is attributed to this statistic.",
            }
        )
    return rows


def _cecl_impact_rows(
    previous_path: str,
    change_kind: str,
    change_path: str,
    old_value: Any,
    new_value: Any,
    previous_cecl: pd.DataFrame,
    changed_cecl: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Calculate CECL reserve/ratio deltas by portfolio and stress level."""
    if previous_cecl.empty or changed_cecl.empty:
        return []
    prev = previous_cecl[previous_cecl["bucket"] == "Total"].copy()
    curr = changed_cecl[changed_cecl["bucket"] == "Total"].copy()
    variant_aware = "scenario_variant" in prev.columns or "scenario_variant" in curr.columns
    if variant_aware:
        if "scenario_variant" not in prev.columns:
            prev["scenario_variant"] = "baseline"
        if "scenario_variant" not in curr.columns:
            curr["scenario_variant"] = "baseline"
    keys = ["portfolio", "stress_level", "bucket"]
    if variant_aware:
        keys.insert(0, "scenario_variant")
    merged = prev.merge(curr, on=keys, how="outer", suffixes=("_previous", "_changed"))
    rows = []
    for _, row in merged.iterrows():
        previous_status = row.get("cecl_reserve_status_previous", "")
        changed_status = row.get("cecl_reserve_status_changed", "")
        if _stringify(previous_status) != _stringify(changed_status):
            rows.append(
                {
                    "previous_scenario": previous_path,
                    "change_kind": change_kind,
                    "change_path": change_path,
                    "old_value": _stringify(old_value),
                    "new_value": _stringify(new_value),
                    "portfolio": row.get("portfolio"),
                    **(
                        {"scenario_variant": row.get("scenario_variant")}
                        if variant_aware
                        else {}
                    ),
                    "stress_level": row.get("stress_level"),
                    "metric": "cecl_reserve_status",
                    "previous_value": previous_status,
                    "changed_value": changed_status,
                    "marginal_impact": "",
                    "notes": "CECL availability status changed; numeric deltas may be unavailable.",
                }
            )
        for metric in ("proforma_cecl_reserve", "proforma_cecl_ratio"):
            old_metric = row.get(f"{metric}_previous")
            new_metric = row.get(f"{metric}_changed")
            if pd.isna(old_metric) and pd.isna(new_metric):
                continue
            if pd.isna(old_metric) or pd.isna(new_metric):
                impact = ""
                notes = "Numeric impact unavailable because one side of the comparison is unavailable."
            else:
                try:
                    impact = float(new_metric) - float(old_metric)
                    notes = ""
                except (TypeError, ValueError):
                    impact = ""
                    notes = "Numeric impact could not be calculated from the reported values."
            if impact == 0:
                continue
            rows.append(
                {
                    "previous_scenario": previous_path,
                    "change_kind": change_kind,
                    "change_path": change_path,
                    "old_value": _stringify(old_value),
                    "new_value": _stringify(new_value),
                    "portfolio": row.get("portfolio"),
                    **(
                        {"scenario_variant": row.get("scenario_variant")}
                        if variant_aware
                        else {}
                    ),
                    "stress_level": row.get("stress_level"),
                    "metric": metric,
                    "previous_value": old_metric,
                    "changed_value": new_metric,
                    "marginal_impact": impact,
                    "notes": notes,
                }
            )
    return rows


def _without_comparison(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    """Disable recursive comparison before comparison reruns."""
    result = copy.deepcopy(dict(scenario))
    result["comparison"] = {}
    return result


def _stringify(value: Any) -> str:
    """Normalize values before equality checks and report display."""
    return json.dumps(value, sort_keys=True, default=str)
