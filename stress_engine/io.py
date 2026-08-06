"""Input loading, profiling, and output writing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .utils import (
    coerce_numeric_frame,
    hash_file,
    json_safe,
    parse_date_series,
    resolve_path,
    sort_frame,
)


_PRESERVED_NUMERIC_TOKEN = object()
RAW_INVALID_NUMERIC_PREFIX = "_raw_invalid_numeric__"
# pandas has changed its built-in NA vocabulary across releases (notably by
# adding ``None`` in 2.0). Own the default set so identical extracts load the
# same way in legacy Anaconda and current environments.
DEFAULT_NA_VALUES = (
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
)


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
        raise ValueError(f"Input source '{name}' must define column_aliases for the columns it imports.")
    if read_options.get("keep_default_na", True) is not False:
        configured_na = read_options.get("na_values")
        read_options["keep_default_na"] = False
        if isinstance(configured_na, Mapping):
            # A pandas per-column mapping replaces global custom tokens. Expand
            # it to every imported source field so the stable defaults remain
            # global while retaining each field's configured additions.
            read_options["na_values"] = {
                field: _merged_na_values(configured_na.get(field))
                for field in rename
            }
        else:
            read_options["na_values"] = _merged_na_values(configured_na)
    source_for_canonical = {canonical: source for source, canonical in rename.items()}
    raw_preserved_numeric_tokens = spec.get(
        "_preserve_numeric_tokens", {}
    )
    preserved_numeric_tokens = (
        {
            str(field): {
                str(token).strip().casefold() for token in tokens
            }
            for field, tokens in raw_preserved_numeric_tokens.items()
        }
        if isinstance(raw_preserved_numeric_tokens, Mapping)
        else {}
    )
    if preserved_numeric_tokens:
        if str(read_options.get("engine", "")).strip().casefold() == "pyarrow":
            raise ValueError(
                f"Input source '{name}' cannot use the pyarrow CSV engine "
                "when preserved numeric tokens are configured."
            )
        date_fields = {str(field) for field in spec.get("date_columns", [])}
        date_conflicts = sorted(set(preserved_numeric_tokens) & date_fields)
        if date_conflicts:
            raise ValueError(
                f"Input source '{name}' cannot parse preserved numeric token "
                "fields as dates: "
                f"{', '.join(date_conflicts)}"
            )
        configured_converters = read_options.get("converters")
        if configured_converters is None:
            converters: Dict[Any, Any] = {}
        elif isinstance(configured_converters, Mapping):
            converters = dict(configured_converters)
        else:
            raise ValueError(
                f"Input source '{name}' read_options.converters must be an object."
            )
        for field, tokens in preserved_numeric_tokens.items():
            source_field = source_for_canonical.get(field, field)
            downstream = converters.get(source_field)
            if downstream is not None and not callable(downstream):
                raise ValueError(
                    f"Input source '{name}' converter for '{source_field}' must be callable."
                )
            # A converter sees the raw cell before pandas' default or custom
            # NA recognition. Preserve only the configured skip tokens while
            # retaining the ordinary missing-value behavior of every other
            # column in the file.
            converters[source_field] = _preserved_numeric_token_converter(
                tokens, downstream
            )
        read_options["converters"] = converters
    string_columns = [str(field) for field in spec.get("string_columns", [])]
    preserved_source_fields = {
        source_for_canonical.get(field, field)
        for field in preserved_numeric_tokens
    }
    source_string_columns = [
        source_for_canonical.get(field, field)
        for field in string_columns
        if source_for_canonical.get(field, field)
        not in preserved_source_fields
    ]
    configured_dtype = read_options.get("dtype")
    if configured_dtype is None:
        if source_string_columns:
            read_options["dtype"] = {field: "string" for field in source_string_columns}
    elif isinstance(configured_dtype, Mapping):
        merged_dtype = dict(configured_dtype)
        for field in source_string_columns:
            merged_dtype.setdefault(field, "string")
        for field in preserved_source_fields:
            merged_dtype.pop(field, None)
        read_options["dtype"] = merged_dtype
    elif preserved_source_fields:
        # A scalar dtype cannot exclude the converter-owned ratio column.
        # Expand it over imported source fields so pandas does not apply both
        # a dtype and a converter to that column.
        read_options["dtype"] = {
            field: configured_dtype
            for field in rename
            if field not in preserved_source_fields
        }
    # Normalize every physical file before concatenation so schema failures
    # remain attributable to a specific file and provenance stays row-exact.
    frames: List[pd.DataFrame] = []
    file_row_counts: List[int] = []
    for path in paths:
        frame = _read_input_file(name, path, spec, read_options)
        _validate_source_columns(name, path, frame, rename)
        # column_aliases is an import allowlist. Unmapped vendor fields are
        # intentionally discarded before they can affect engine behavior.
        mapped_columns = [column for column in frame.columns if column in rename]
        frame = frame.loc[:, mapped_columns]
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

    # Audit coercion against the original cells before replacing them. This
    # distinguishes source data-quality failures from genuinely blank values.
    coercion_issues: List[Dict[str, Any]] = []
    for field in spec.get("date_columns", []):
        if field in df.columns:
            original = df[field]
            converted = parse_date_series(original)
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
    preserved_invalid_fields = set(
        spec.get("_preserve_invalid_numeric_values", [])
    )
    preserved_masks: Dict[str, pd.Series] = {}
    for field in sorted(numeric_fields):
        if field not in df.columns:
            continue
        original = df[field]
        preserve_mask = pd.Series(False, index=df.index)
        if field in preserved_numeric_tokens:
            preserve_mask = original.map(
                lambda value: value is _PRESERVED_NUMERIC_TOKEN
            )
            preserved_masks[field] = preserve_mask
        converted = coerce_numeric_frame(df[[field]], [field])[field]
        invalid = (
            original.notna()
            & original.astype(str).str.strip().ne("")
            & converted.isna()
            & ~preserve_mask
        )
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
            if field in preserved_invalid_fields:
                raw_field = f"{RAW_INVALID_NUMERIC_PREFIX}{field}"
                df[raw_field] = pd.Series(
                    pd.NA, index=df.index, dtype=object
                )
                df.loc[invalid, raw_field] = original.loc[invalid]
    df = coerce_numeric_frame(df, numeric_fields)
    for field, preserve_mask in preserved_masks.items():
        if not preserve_mask.any():
            continue
        df[field] = df[field].astype(object)
        df.loc[preserve_mask, field] = "N/A"
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


def _preserved_numeric_token_converter(
    tokens: set[str], downstream: Any = None
):
    """Keep configured raw numeric sentinels ahead of pandas NA parsing."""

    def convert(value: Any) -> Any:
        """Preserve recognized sentinels and delegate every other raw cell."""
        text = str(value).strip()
        if text.casefold() in tokens:
            return _PRESERVED_NUMERIC_TOKEN
        return downstream(value) if downstream is not None else value

    return convert


def _merged_na_values(configured: Any) -> List[Any]:
    """Merge custom NA tokens into the engine's version-stable defaults."""
    if configured is None:
        additions: List[Any] = []
    elif isinstance(configured, (list, tuple, set)):
        additions = list(configured)
    else:
        additions = [configured]
    return list(dict.fromkeys([*DEFAULT_NA_VALUES, *additions]))


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
    """Require configured source columns while allowing unrelated extra columns."""
    missing_aliases = [source for source in rename if source not in frame.columns]
    if missing_aliases:
        raise ValueError(
            f"Input source '{name}' file '{path.name}' is missing configured aliased columns: "
            f"{', '.join(missing_aliases)}"
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
    """Add one unambiguous, nonblank source-to-canonical column alias."""
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
    identity_strings.extend(
        spec.get("identity_key")
        for spec in inputs.get("sources", {}).values()
        if spec.get("identity_key")
    )
    identity_spec["string_columns"] = list(dict.fromkeys(identity_strings))
    balance_field = borrower.get("balance_field")
    if balance_field:
        identity_spec["numeric_columns"] = list(
            dict.fromkeys(
                [*identity_spec.get("numeric_columns", []), balance_field]
            )
        )
        identity_spec["_preserve_invalid_numeric_values"] = [
            balance_field
        ]
    loaded["identity"] = read_table("identity", identity_spec, base_dir)
    cecl = scenario.get("cecl", {})
    basis = cecl.get("reserve_basis", {}) if isinstance(cecl, Mapping) else {}
    historical = (
        basis.get("historical", {}) if isinstance(basis, Mapping) else {}
    )
    optional_history_source = (
        str(historical.get("source", "")).strip()
        if isinstance(historical, Mapping)
        and historical.get("enabled", False) is True
        else ""
    )
    for name, spec in inputs.get("sources", {}).items():
        source_spec = dict(spec)
        if str(name) == optional_history_source:
            source_spec["_preserve_numeric_tokens"] = {
                str(
                    historical.get(
                        "ratio_field", "historical_cecl_ratio"
                    )
                ).strip(): ["N/A", "#N/A"]
            }
        key = source_spec.get("key")
        if key:
            source_spec["string_columns"] = list(dict.fromkeys([*source_spec.get("string_columns", []), key]))
        try:
            loaded[name] = read_table(str(name), source_spec, base_dir)
        except FileNotFoundError:
            if str(name) != optional_history_source:
                raise
            # CECL tag history is commercial-only. Deferring a missing file
            # lets Consumer-only runs proceed and lets mixed runs report the
            # commercial history component as unavailable in the CECL audit.
            continue
    return loaded


def profile_frame(name: str, df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Build field-level counts and numeric stats for input reporting."""
    rows: List[Dict[str, Any]] = []
    for column in sorted(df.columns):
        if str(column).startswith(("_source_", RAW_INVALID_NUMERIC_PREFIX)):
            continue
        series = df[column]
        # Dates are audit dimensions, not numeric measures. Coercing parsed
        # datetimes produces version-dependent epoch units and turns ``NaT``
        # into the minimum int64 sentinel on older pandas releases.
        numeric = (
            None
            if pd.api.types.is_datetime64_any_dtype(series.dtype)
            else pd.to_numeric(series, errors="coerce")
        )
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
        if numeric is not None and numeric.notna().any():
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
        column_count = sum(
            not str(column).startswith(("_source_", RAW_INVALID_NUMERIC_PREFIX))
            for column in item.frame.columns
        )
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
    frame = _spreadsheet_safe_frame(sort_frame(df, sort_by or []))
    # A sibling temporary keeps readers from observing a partially written
    # report while preserving an atomic replace on the destination volume.
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _spreadsheet_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return an export-only copy with spreadsheet formulas neutralized."""
    columns: List[pd.Series] = []
    for position in range(df.shape[1]):
        series = df.iloc[:, position].copy()
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            series = series.map(_neutralize_spreadsheet_formula)
        columns.append(series)
    frame = pd.concat(columns, axis=1) if columns else df.copy()
    frame.columns = [_neutralize_spreadsheet_formula(column) for column in df.columns]
    return frame


def _neutralize_spreadsheet_formula(value: Any) -> Any:
    """Prefix formula-like strings so spreadsheet programs treat them as text."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def write_json(data: Mapping[str, Any], path: Path) -> None:
    """Write JSON output after converting pandas/numpy values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Match CSV's atomic-publication contract for audit metadata as well.
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(data)), handle, indent=2, sort_keys=True, default=str)
    temporary.replace(path)
