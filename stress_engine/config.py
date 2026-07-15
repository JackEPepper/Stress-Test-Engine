"""Scenario loading and light validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .utils import deep_merge, hash_json, resolve_path


_INCLUDE_KEY = "$include"


def load_scenario(paths: str | Path | Iterable[str | Path]) -> Tuple[Dict[str, Any], Path]:
    """Load one or more JSON files, including manifests, and deep-merge them.

    A file can declare ``"$include": ["relative/file.json", ...]``. Includes
    are merged in listed order, followed by the declaring file, so local values
    override included defaults. Explicitly supplied files are then merged in
    command-line order. Relative data input paths remain anchored to the first
    explicitly supplied scenario file.
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
        payload, loaded_paths = _load_scenario_file(path.resolve(), stack=())
        merged = deep_merge(merged, payload)
        resolved_paths.extend(str(item) for item in loaded_paths)

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


def _load_scenario_file(path: Path, stack: Tuple[Path, ...]) -> Tuple[Dict[str, Any], List[Path]]:
    """Load one scenario fragment and recursively expand its relative includes."""
    actual = path.resolve()
    if actual in stack:
        cycle = " -> ".join(str(item) for item in (*stack, actual))
        raise ValueError(f"Scenario include cycle detected: {cycle}")

    with actual.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario file must contain a JSON object: {actual}")

    local_payload = dict(payload)
    raw_includes = local_payload.pop(_INCLUDE_KEY, [])
    if isinstance(raw_includes, (str, Path)):
        includes = [raw_includes]
    elif isinstance(raw_includes, list):
        includes = raw_includes
    else:
        raise ValueError(f"Scenario {_INCLUDE_KEY} must be a string or list: {actual}")

    merged: Dict[str, Any] = {}
    resolved_paths: List[Path] = []
    next_stack = (*stack, actual)
    for include in includes:
        if not isinstance(include, str) or not include.strip():
            raise ValueError(f"Scenario {_INCLUDE_KEY} entries must be nonblank strings: {actual}")
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = actual.parent / include_path
        included, included_paths = _load_scenario_file(include_path, next_stack)
        merged = deep_merge(merged, included)
        resolved_paths.extend(included_paths)

    merged = deep_merge(merged, local_payload)
    resolved_paths.append(actual)
    return merged, resolved_paths


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
    cre = scenario.get("modules", {}).get("CRE", {})
    if cre and cre.get("enabled", True):
        cutoff = scenario.get("run", {}).get("cutoff_date")
        if cutoff is None or pd.isna(pd.to_datetime(cutoff, errors="coerce")):
            raise ValueError("An enabled CRE module requires a valid run.cutoff_date.")


def output_dir_for(scenario: Dict[str, Any], base_dir: Path, override: str | Path | None = None) -> Path:
    """Resolve the scenario or CLI output directory relative to scenario JSON."""
    if override:
        return resolve_path(override, base_dir)
    outputs = scenario.get("outputs", {})
    directory = outputs.get("directory", "outputs/latest")
    return resolve_path(directory, base_dir)
