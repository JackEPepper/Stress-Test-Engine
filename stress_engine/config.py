"""Scenario loading and light validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .utils import deep_merge, hash_json, resolve_path


def load_scenario(paths: str | Path | Iterable[str | Path]) -> Tuple[Dict[str, Any], Path]:
    """Load one or more JSON files and deep-merge them in order.

    Later files override earlier files. Relative input paths are resolved from
    the directory of the first scenario file, which makes layered scenario JSONs
    portable.
    """
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("At least one scenario JSON path is required.")

    merged: Dict[str, Any] = {}
    resolved_paths: List[str] = []
    for path in path_list:
        actual = path.resolve()
        with actual.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Scenario file must contain a JSON object: {actual}")
        merged = deep_merge(merged, payload)
        resolved_paths.append(str(actual))

    base_dir = path_list[0].resolve().parent
    merged.setdefault("_metadata", {})
    merged["_metadata"].update(
        {
            "scenario_files": resolved_paths,
            "scenario_hash": hash_json({k: v for k, v in merged.items() if k != "_metadata"}),
        }
    )
    validate_scenario(merged)
    return merged, base_dir


def validate_scenario(scenario: Dict[str, Any]) -> None:
    """Perform lightweight required-section validation after JSON merge."""
    required = ["inputs", "borrower", "tags", "modules"]
    missing = [key for key in required if key not in scenario]
    if missing:
        raise ValueError(f"Scenario is missing required sections: {', '.join(missing)}")
    if "identity" not in scenario["inputs"]:
        raise ValueError("Scenario inputs must include an 'identity' source.")
    borrower = scenario.get("borrower", {})
    for field in ("borrower_id_field", "balance_field"):
        if field not in borrower:
            raise ValueError(f"Scenario borrower section must define '{field}'.")
    levels = [str(level) for level in scenario.get("stress_levels", ["S1", "S2"])]
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("Scenario stress_levels must contain unique, nonblank levels.")
    if any(not level.strip() for level in levels):
        raise ValueError("Scenario stress_levels cannot contain blank names.")
    cre = scenario.get("modules", {}).get("CRE", scenario.get("modules", {}).get("cre", {}))
    if cre and cre.get("enabled", True):
        cutoff = scenario.get("run", {}).get("cutoff_date", scenario.get("cutoff_date"))
        if cutoff is None or pd.isna(pd.to_datetime(cutoff, errors="coerce")):
            raise ValueError("An enabled CRE module requires a valid run.cutoff_date.")


def output_dir_for(scenario: Dict[str, Any], base_dir: Path, override: str | Path | None = None) -> Path:
    """Resolve the scenario or CLI output directory relative to scenario JSON."""
    if override:
        return resolve_path(override, base_dir)
    outputs = scenario.get("outputs", {})
    directory = outputs.get("directory", "outputs/latest")
    return resolve_path(directory, base_dir)
