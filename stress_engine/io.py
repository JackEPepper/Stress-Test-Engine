"""Input loading, profiling, and output writing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .utils import coerce_numeric_frame, hash_file, json_safe, resolve_path, sort_frame


@dataclass
class LoadedTable:
    name: str
    frame: pd.DataFrame
    path: Path
    file_hash: str
    profile: pd.DataFrame


def read_table(name: str, spec: Mapping[str, Any], base_dir: Path) -> LoadedTable:
    if "path" not in spec:
        raise ValueError(f"Input source '{name}' must define a path.")
    path = resolve_path(spec["path"], base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input source '{name}' not found: {path}")
    file_type = str(spec.get("type") or path.suffix.lstrip(".")).lower()
    if file_type in {"csv", "txt"}:
        df = pd.read_csv(path, **spec.get("read_options", {}))
    elif file_type in {"xlsx", "xlsm", "xls"}:
        sheet_name = spec.get("sheet_name", spec.get("sheet", 0))
        df = pd.read_excel(path, sheet_name=sheet_name, **spec.get("read_options", {}))
    else:
        raise ValueError(f"Unsupported input type for '{name}': {file_type}")

    rename = spec.get("rename", {})
    if rename:
        df = df.rename(columns=rename)

    for field in spec.get("date_columns", []):
        if field in df.columns:
            df[field] = pd.to_datetime(df[field], errors="coerce")

    numeric_fields = set(spec.get("numeric_columns", []))
    numeric_fields.update(spec.get("balance_fields", []))
    numeric_fields.update(spec.get("sum_fields", []))
    numeric_fields.update(spec.get("dollar_fields", []))
    df = coerce_numeric_frame(df, numeric_fields)
    df = df.reset_index(drop=True)
    df["_source_row"] = np.arange(1, len(df) + 1)

    required = spec.get("required_columns", [])
    missing = [field for field in required if field not in df.columns]
    if missing:
        raise ValueError(f"Input source '{name}' missing required columns: {', '.join(missing)}")

    profile = profile_frame(name, df, str(path))
    return LoadedTable(name=name, frame=df, path=path, file_hash=hash_file(path), profile=profile)


def load_inputs(scenario: Mapping[str, Any], base_dir: Path) -> Dict[str, LoadedTable]:
    inputs = scenario.get("inputs", {})
    loaded: Dict[str, LoadedTable] = {}
    loaded["identity"] = read_table("identity", inputs["identity"], base_dir)
    for name, spec in inputs.get("sources", {}).items():
        loaded[name] = read_table(str(name), spec, base_dir)
    return loaded


def profile_frame(name: str, df: pd.DataFrame, path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for column in sorted(df.columns):
        if column == "_source_row":
            continue
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        row: Dict[str, Any] = {
            "dataset": name,
            "path": path,
            "field": column,
            "row_count": int(len(df)),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "numeric_sum": np.nan,
            "numeric_mean": np.nan,
            "numeric_min": np.nan,
            "numeric_max": np.nan,
        }
        if numeric.notna().any():
            row.update(
                {
                    "numeric_sum": float(numeric.sum()),
                    "numeric_mean": float(numeric.mean()),
                    "numeric_min": float(numeric.min()),
                    "numeric_max": float(numeric.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_for_inputs(loaded: Mapping[str, LoadedTable]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in sorted(loaded):
        item = loaded[name]
        rows.append(
            {
                "name": name,
                "path": str(item.path),
                "sha256": item.file_hash,
                "rows": int(len(item.frame)),
                "columns": int(len(item.frame.columns) - (1 if "_source_row" in item.frame.columns else 0)),
            }
        )
    return rows


def write_csv(df: pd.DataFrame, path: Path, sort_by: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = sort_frame(df, sort_by or [])
    frame.to_csv(path, index=False)


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(data)), handle, indent=2, sort_keys=True, default=str)
