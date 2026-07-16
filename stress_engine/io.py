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
    paths: List[Path]
    file_hashes: List[str]
    file_row_counts: List[int]
    profile: pd.DataFrame
    coercion_issues: List[Dict[str, Any]]


def read_table(name: str, spec: Mapping[str, Any], base_dir: Path) -> LoadedTable:
    """Load and concatenate one or more CSV/XLSX files for an input source.

    Called by `load_inputs`; the returned profile is later written to
    `input_summary.csv`.
    """
    paths = _input_paths(name, spec, base_dir)
    read_options = dict(spec.get("read_options", {}))
    rename = _column_rename_map(name, spec)
    if not rename:
        raise ValueError(f"Input source '{name}' must define column_aliases for every source column.")
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
    frames: List[pd.DataFrame] = []
    file_row_counts: List[int] = []
    for path in paths:
        frame = _read_input_file(name, path, spec, read_options)
        _validate_source_columns(name, path, frame, rename)
        frame = frame.rename(columns=rename)
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"Input source '{name}' aliases create duplicate columns in {path.name}: "
                f"{', '.join(duplicates)}"
            )
        frame["_source_file"] = str(path)
        frame["_source_file_row"] = np.arange(1, len(frame) + 1)
        frames.append(frame)
        file_row_counts.append(len(frame))
    df = pd.concat(frames, ignore_index=True)

    coercion_issues: List[Dict[str, Any]] = []
    for field in spec.get("date_columns", []):
        if field in df.columns:
            original = df[field]
            converted = pd.to_datetime(original, errors="coerce")
            invalid = original.notna() & original.astype(str).str.strip().ne("") & converted.isna()
            if invalid.any():
                coercion_issues.extend(
                    _coercion_issue_rows(
                        name,
                        df,
                        invalid,
                        field,
                        "invalid_date_coerced_to_missing",
                    )
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
            coercion_issues.extend(
                _coercion_issue_rows(
                    name,
                    df,
                    invalid,
                    field,
                    "invalid_numeric_coerced_to_missing",
                )
            )
    df = coerce_numeric_frame(df, numeric_fields)
    df = df.reset_index(drop=True)
    df["_source_row"] = np.arange(1, len(df) + 1)

    required = spec.get("required_columns", [])
    missing = [field for field in required if field not in df.columns]
    if missing:
        raise ValueError(f"Input source '{name}' missing required columns: {', '.join(missing)}")

    path_text = ";".join(str(path) for path in paths)
    profile = profile_frame(name, df, path_text)
    return LoadedTable(
        name=name,
        frame=df,
        paths=paths,
        file_hashes=[hash_file(path) for path in paths],
        file_row_counts=file_row_counts,
        profile=profile,
        coercion_issues=coercion_issues,
    )


def _input_paths(name: str, spec: Mapping[str, Any], base_dir: Path) -> List[Path]:
    """Resolve the exclusive ``path`` or ``paths`` setting for one source."""
    has_path = "path" in spec
    has_paths = "paths" in spec
    if has_path == has_paths:
        raise ValueError(f"Input source '{name}' must define exactly one of path or paths.")
    raw_paths = [spec["path"]] if has_path else spec["paths"]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"Input source '{name}' paths must be a nonempty JSON list.")
    paths: List[Path] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
            raise ValueError(f"Input source '{name}' paths must contain nonblank file names.")
        path = resolve_path(raw_path, base_dir).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input source '{name}' not found: {path}")
        path_key = str(path).casefold()
        if path_key in seen:
            raise ValueError(f"Input source '{name}' paths contains the same file more than once: {path}")
        seen.add(path_key)
        paths.append(path)
    return paths


def _read_input_file(
    name: str,
    path: Path,
    spec: Mapping[str, Any],
    read_options: Mapping[str, Any],
) -> pd.DataFrame:
    """Read one physical file belonging to a logical input source."""
    file_type = str(spec.get("type") or path.suffix.lstrip(".")).lower()
    if file_type == "csv":
        return pd.read_csv(path, **dict(read_options))
    if file_type in {"xlsx", "xlsm"}:
        return pd.read_excel(
            path,
            sheet_name=spec.get("sheet_name", 0),
            **dict(read_options),
        )
    raise ValueError(f"Unsupported input type for '{name}' file '{path.name}': {file_type}")


def _validate_source_columns(
    name: str,
    path: Path,
    frame: pd.DataFrame,
    rename: Mapping[str, str],
) -> None:
    """Require every physical file in a logical source to use the same aliases."""
    missing_aliases = [source for source in rename if source not in frame.columns]
    if missing_aliases:
        raise ValueError(
            f"Input source '{name}' file '{path.name}' is missing configured aliased columns: "
            f"{', '.join(missing_aliases)}"
        )
    unmapped = [str(column) for column in frame.columns if column not in rename]
    if unmapped:
        raise ValueError(
            f"Input source '{name}' file '{path.name}' has columns missing from column_aliases: "
            f"{', '.join(unmapped)}"
        )
    collisions = [
        (source, canonical)
        for source, canonical in rename.items()
        if source != canonical and source in frame.columns and canonical in frame.columns
    ]
    if collisions:
        details = ", ".join(f"{source} -> {canonical}" for source, canonical in collisions)
        raise ValueError(
            f"Input source '{name}' file '{path.name}' alias collides with an existing canonical column: "
            f"{details}"
        )


def _coercion_issue_rows(
    name: str,
    frame: pd.DataFrame,
    invalid: pd.Series,
    field: str,
    kind: str,
) -> List[Dict[str, Any]]:
    """Describe coercion failures by physical source file."""
    rows: List[Dict[str, Any]] = []
    counts = frame.loc[invalid, "_source_file"].value_counts(sort=False)
    for path, count in counts.items():
        rows.append(
            {
                "source": name,
                "path": str(path),
                "field": field,
                "kind": kind,
                "count": int(count),
            }
        )
    return rows


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
        if str(column).startswith("_source_"):
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
        column_count = sum(not str(column).startswith("_source_") for column in item.frame.columns)
        for path, file_hash, row_count in zip(item.paths, item.file_hashes, item.file_row_counts):
            rows.append(
                {
                    "name": name,
                    "path": str(path),
                    "sha256": file_hash,
                    "rows": int(row_count),
                    "columns": column_count,
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
