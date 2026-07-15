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
    """Container for one loaded input plus audit metadata."""
    name: str
    frame: pd.DataFrame
    path: Path
    file_hash: str
    profile: pd.DataFrame
    coercion_issues: List[Dict[str, Any]]


def read_table(name: str, spec: Mapping[str, Any], base_dir: Path) -> LoadedTable:
    """Load one CSV/XLSX input according to its scenario spec.

    Called by `load_inputs`; the returned profile is later written to
    `input_summary.csv`.
    """
    if "path" not in spec:
        raise ValueError(f"Input source '{name}' must define a path.")
    path = resolve_path(spec["path"], base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input source '{name}' not found: {path}")
    file_type = str(spec.get("type") or path.suffix.lstrip(".")).lower()
    read_options = dict(spec.get("read_options", {}))
    rename = _column_rename_map(name, spec)
    source_for_canonical = {canonical: source for source, canonical in rename.items()}
    string_columns = [str(field) for field in spec.get("string_columns", [])]
    if string_columns:
        source_string_columns = [source_for_canonical.get(field, field) for field in string_columns]
        configured_dtype = read_options.get("dtype")
        if configured_dtype is None:
            read_options["dtype"] = {field: "string" for field in source_string_columns}
        elif isinstance(configured_dtype, Mapping):
            merged_dtype = dict(configured_dtype)
            for field in source_string_columns:
                merged_dtype.setdefault(field, "string")
            read_options["dtype"] = merged_dtype
    if file_type == "csv":
        df = pd.read_csv(path, **read_options)
    elif file_type in {"xlsx", "xlsm"}:
        sheet_name = spec.get("sheet_name", 0)
        df = pd.read_excel(path, sheet_name=sheet_name, **read_options)
    else:
        raise ValueError(f"Unsupported input type for '{name}': {file_type}")

    if not rename:
        raise ValueError(f"Input source '{name}' must define column_aliases for every source column.")
    missing_aliases = [source for source in rename if source not in df.columns]
    if missing_aliases:
        raise ValueError(
            f"Input source '{name}' is missing configured aliased columns: {', '.join(missing_aliases)}"
        )
    unmapped = [str(column) for column in df.columns if column not in rename]
    if unmapped:
        raise ValueError(f"Input source '{name}' has columns missing from column_aliases: {', '.join(unmapped)}")
    collisions = [
        (source, canonical)
        for source, canonical in rename.items()
        if source != canonical and source in df.columns and canonical in df.columns
    ]
    if collisions:
        details = ", ".join(f"{source} -> {canonical}" for source, canonical in collisions)
        raise ValueError(f"Input source '{name}' alias collides with an existing canonical column: {details}")
    df = df.rename(columns=rename)
    duplicates = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Input source '{name}' aliases create duplicate columns: {', '.join(duplicates)}")

    coercion_issues: List[Dict[str, Any]] = []
    for field in spec.get("date_columns", []):
        if field in df.columns:
            original = df[field]
            converted = pd.to_datetime(original, errors="coerce")
            invalid = original.notna() & original.astype(str).str.strip().ne("") & converted.isna()
            if invalid.any():
                coercion_issues.append(
                    {
                        "source": name,
                        "field": field,
                        "kind": "invalid_date_coerced_to_missing",
                        "count": int(invalid.sum()),
                    }
                )
            df[field] = converted

    numeric_fields = set(spec.get("numeric_columns", []))
    for field in sorted(numeric_fields):
        if field not in df.columns:
            continue
        original = df[field]
        converted = coerce_numeric_frame(df[[field]], [field])[field]
        invalid = original.notna() & original.astype(str).str.strip().ne("") & converted.isna()
        if invalid.any():
            coercion_issues.append(
                {
                    "source": name,
                    "field": field,
                    "kind": "invalid_numeric_coerced_to_missing",
                    "count": int(invalid.sum()),
                }
            )
    df = coerce_numeric_frame(df, numeric_fields)
    df = df.reset_index(drop=True)
    df["_source_row"] = np.arange(1, len(df) + 1)

    required = spec.get("required_columns", [])
    missing = [field for field in required if field not in df.columns]
    if missing:
        raise ValueError(f"Input source '{name}' missing required columns: {', '.join(missing)}")

    profile = profile_frame(name, df, str(path))
    return LoadedTable(
        name=name,
        frame=df,
        path=path,
        file_hash=hash_file(path),
        profile=profile,
        coercion_issues=coercion_issues,
    )


def _column_rename_map(name: str, spec: Mapping[str, Any]) -> Dict[str, str]:
    """Return source-to-canonical renames from ``column_aliases``."""
    aliases = spec.get("column_aliases", {}) or {}
    if not isinstance(aliases, Mapping):
        raise ValueError(f"Input source '{name}' column_aliases must be a JSON object.")

    rename: Dict[str, str] = {}
    for canonical, source in aliases.items():
        _add_column_rename(name, rename, source, canonical)
    return rename


def _add_column_rename(name: str, rename: Dict[str, str], source: Any, canonical: Any) -> None:
    source_name = str(source).strip()
    canonical_name = str(canonical).strip()
    if not source_name or not canonical_name:
        raise ValueError(f"Input source '{name}' column aliases must use nonblank names.")
    existing = rename.get(source_name)
    if existing is not None and existing != canonical_name:
        raise ValueError(
            f"Input source '{name}' maps source column '{source_name}' to both '{existing}' and '{canonical_name}'."
        )
    rename[source_name] = canonical_name


def load_inputs(scenario: Mapping[str, Any], base_dir: Path) -> Dict[str, LoadedTable]:
    """Load the identity file plus arbitrary configured source tables."""
    inputs = scenario.get("inputs", {})
    loaded: Dict[str, LoadedTable] = {}
    borrower = scenario.get("borrower", {})
    identity_spec = dict(inputs["identity"])
    identity_strings = list(identity_spec.get("string_columns", []))
    identity_strings.extend(
        field
        for field in (borrower.get("borrower_id_field"), borrower.get("loan_id_field"))
        if field
    )
    identity_spec["string_columns"] = list(dict.fromkeys(identity_strings))
    loaded["identity"] = read_table("identity", identity_spec, base_dir)
    for name, spec in inputs.get("sources", {}).items():
        source_spec = dict(spec)
        key = source_spec.get("key")
        if key:
            source_spec["string_columns"] = list(dict.fromkeys([*source_spec.get("string_columns", []), key]))
        loaded[name] = read_table(str(name), source_spec, base_dir)
    return loaded


def profile_frame(name: str, df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Build field-level counts and numeric stats for input reporting."""
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
    """Summarize input files for `metadata.json`."""
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
    """Write one CSV output with optional stable sorting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = sort_frame(df, sort_by or [])
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(data: Mapping[str, Any], path: Path) -> None:
    """Write JSON output after converting pandas/numpy values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(data)), handle, indent=2, sort_keys=True, default=str)
    temporary.replace(path)
