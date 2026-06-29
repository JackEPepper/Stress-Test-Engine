"""Run metadata helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from stress_engine.io.loaders import hash_file, hash_payload


def build_run_metadata(
    config: Mapping[str, object],
    scenario: Mapping[str, object],
    config_path: Path,
    scenario_path: Path,
    input_paths: Mapping[str, Path],
    external_paths: Mapping[str, Path],
    tables: Mapping[str, pd.DataFrame],
    results: pd.DataFrame,
    validation_issue_count: int,
    tie_out_status: str,
) -> dict:
    input_hashes = {name: hash_file(path) for name, path in input_paths.items() if path.exists()}
    external_hashes = {name: hash_file(path) for name, path in external_paths.items() if path.exists()}
    run_seed = {
        "engine_version": config.get("engine_version"),
        "scenario_hash": hash_file(scenario_path),
        "input_hashes": input_hashes,
        "external_hashes": external_hashes,
        "as_of_date": scenario.get("as_of_date"),
    }
    run_id = hash_payload(run_seed)[:16]
    out_of_scope = results[results["scope_status"] == "out_of_scope"] if "scope_status" in results else pd.DataFrame()
    return {
        "run_id": run_id,
        "scenario_id": scenario.get("scenario_id"),
        "as_of_date": scenario.get("as_of_date"),
        "engine_version": config.get("engine_version"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "scenario_file": str(scenario_path),
        "config_hash": hash_file(config_path),
        "scenario_hash": hash_file(scenario_path),
        "input_file_hashes": input_hashes,
        "external_source_hashes": external_hashes,
        "row_counts": {name: len(frame) for name, frame in tables.items()},
        "validation_issue_count": validation_issue_count,
        "out_of_scope_loan_count": int(len(out_of_scope)),
        "out_of_scope_balance": float(out_of_scope["balance"].sum()) if "balance" in out_of_scope else 0.0,
        "external_source_tie_out_status": tie_out_status,
    }
