"""Shared deterministic utility helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd


BUCKET_ORDER = {
    "Pass": 0,
    "Special Mention": 1,
    "Substandard": 2,
    "Unknown": -1,
}


def stable_name(value: Any) -> str:
    """Return a predictable, column-safe name."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unnamed"


def as_list(value: Any) -> List[Any]:
    """Normalize scalar-or-list scenario values to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def condition_fields(conditions: Any) -> set[str]:
    """Collect canonical field names from a nested tag-condition block."""
    if isinstance(conditions, Mapping):
        fields = {str(conditions["field"])} if conditions.get("field") else set()
        for key in ("all", "any"):
            for item in as_list(conditions.get(key)):
                fields.update(condition_fields(item))
        return fields
    fields: set[str] = set()
    for item in as_list(conditions):
        fields.update(condition_fields(item))
    return fields


def is_missing(value: Any) -> bool:
    """Return True for Python/pandas missing values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_number(value: Any, default: float = np.nan) -> float:
    """Convert formatted numbers/currency/percent text to float."""
    if is_missing(value):
        return default
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    is_percent = "%" in text
    is_parenthetical = text.startswith("(") and text.endswith(")")
    text = text.replace("$", "").replace(",", "").replace("%", "")
    if is_parenthetical:
        text = f"-{text[1:-1]}"
    try:
        number = float(text)
        return number / 100.0 if is_percent else number
    except ValueError:
        return default


def coerce_numeric_frame(df: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    """Coerce configured numeric columns after stripping common formatting."""
    out = df.copy()
    for field in fields:
        if field in out.columns:
            text = out[field].astype(str).str.strip()
            is_percent = text.str.contains("%", regex=False, na=False)
            is_parenthetical = text.str.startswith("(") & text.str.endswith(")")
            cleaned = text.str.replace(r"[$,%]", "", regex=True).str.replace(",", "")
            cleaned = cleaned.where(~is_parenthetical, "-" + cleaned.str.slice(1, -1))
            numeric = pd.to_numeric(cleaned, errors="coerce")
            out[field] = numeric.where(~is_percent, numeric / 100.0)
    return out


def parse_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def risk_bucket_from_rating(rating: Any) -> str:
    """Map in-place rating to Pass/Special Mention/Substandard."""
    numeric = to_number(rating)
    if math.isnan(numeric):
        return "Unknown"
    if numeric < 7:
        return "Pass"
    if numeric == 7:
        return "Special Mention"
    return "Substandard"


def worse_bucket(left: str, right: str) -> str:
    """Return the higher-risk bucket based on BUCKET_ORDER."""
    return left if BUCKET_ORDER.get(left, -1) >= BUCKET_ORDER.get(right, -1) else right


def lower_metric_bucket(value: float, cutoffs: Mapping[str, Any]) -> str:
    """For ratios where lower is worse, such as DSCR and FCCR."""
    if is_missing(value):
        return "Unknown"
    sub = to_number(cutoffs.get("substandard"))
    sm = to_number(cutoffs.get("special_mention"))
    if not math.isnan(sub) and value <= sub:
        return "Substandard"
    if not math.isnan(sm) and value <= sm:
        return "Special Mention"
    return "Pass"


def higher_metric_bucket(value: float, cutoffs: Mapping[str, Any]) -> str:
    """For ratios where higher is worse, such as LTV."""
    if is_missing(value):
        return "Unknown"
    sub = to_number(cutoffs.get("substandard"))
    sm = to_number(cutoffs.get("special_mention"))
    if not math.isnan(sub) and value >= sub:
        return "Substandard"
    if not math.isnan(sm) and value >= sm:
        return "Special Mention"
    return "Pass"


def annual_debt_payment(balance: float, annual_rate: float, amortization_years: float) -> float:
    """Annual amortizing debt payment used by CRE refinance DSCR."""
    balance = to_number(balance, 0.0)
    annual_rate = to_number(annual_rate, 0.0)
    amortization_years = to_number(amortization_years, 0.0)
    if balance <= 0 or amortization_years <= 0:
        return np.nan
    if abs(annual_rate) < 1e-12:
        return balance / amortization_years
    return balance * annual_rate / (1 - (1 + annual_rate) ** (-amortization_years))


def get_levels(scenario: Mapping[str, Any]) -> List[str]:
    """Return scenario stress levels, defaulting to S1/S2."""
    return [str(level) for level in scenario.get("stress_levels", ["S1", "S2"])]


def lookup_parameter_with_source(
    table: Any,
    subsector: Any,
    level: str,
    default: Any = np.nan,
) -> Tuple[Any, str]:
    """Return a parameter plus whether it came from sector, default, or fallback."""
    if isinstance(table, Mapping):
        sector_key = str(subsector) if not is_missing(subsector) else None
        if sector_key and sector_key in table:
            candidate = table[sector_key]
            if isinstance(candidate, Mapping):
                if level in candidate:
                    return candidate[level], "sector"
                for key in ("default", "all", "*"):
                    if key in candidate:
                        return candidate[key], "sector_default"
            elif candidate is not None:
                return candidate, "sector"
        if level in table:
            return table[level], "level"
        for key in ("default", "all", "*"):
            if key not in table:
                continue
            candidate = table[key]
            if isinstance(candidate, Mapping):
                if level in candidate:
                    return candidate[level], "default"
                for nested in ("default", "all", "*"):
                    if nested in candidate:
                        return candidate[nested], "default"
            elif candidate is not None:
                return candidate, "default"
        return default, "missing"
    if table is None:
        return default, "missing"
    return table, "scalar"


def ensure_columns(df: pd.DataFrame, fields: Sequence[str], context: str) -> None:
    """Raise a readable error when required fields are missing."""
    missing = [field for field in fields if field and field not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def first_non_null(values: pd.Series) -> Any:
    not_null = values.dropna()
    if not_null.empty:
        return np.nan
    return not_null.iloc[0]


def join_unique(values: pd.Series) -> str:
    items = [str(item) for item in values.dropna().astype(str).unique()]
    return ";".join(sorted(items))


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def flatten_json(data: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(data, Mapping):
        for key in sorted(data):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(data[key], path))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            path = f"{prefix}[{index}]"
            out.update(flatten_json(value, path))
        if not data:
            out[prefix] = []
    else:
        out[prefix] = data
    return out


_PATH_TOKEN_RE = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


def json_path_tokens(path: str) -> List[Any]:
    """Tokenize dotted JSON paths such as ``modules.CRE.x[0]``."""
    tokens = [
        match.group(1) if match.group(1) is not None else int(match.group(2))
        for match in _PATH_TOKEN_RE.finditer(path)
    ]
    if not tokens:
        raise ValueError(f"Invalid JSON path: {path}")
    return tokens


def get_json_path(data: Mapping[str, Any], path: str) -> Any:
    """Return a value from a dotted/list-index JSON path."""
    cursor: Any = data
    for token in json_path_tokens(path):
        cursor = cursor[token]
    return cursor


def set_json_path_in_place(data: Dict[str, Any], path: str, value: Any, allow_create: bool = False) -> None:
    """Set a JSON path, optionally creating missing dictionary keys."""
    tokens = json_path_tokens(path)
    cursor: Any = data
    for token, next_token in zip(tokens[:-1], tokens[1:]):
        if isinstance(token, int):
            cursor = cursor[token]
            continue
        if token not in cursor:
            if not allow_create:
                raise KeyError(path)
            cursor[token] = [] if isinstance(next_token, int) else {}
        cursor = cursor[token]
    final = tokens[-1]
    if isinstance(final, int):
        cursor[final] = copy.deepcopy(value)
    else:
        if final not in cursor and not allow_create:
            raise KeyError(path)
        cursor[final] = copy.deepcopy(value)


def set_json_path(data: Any, path: str, value: Any) -> Any:
    """Return a deep copy with one flattened-json path replaced."""
    result = copy.deepcopy(data)
    set_json_path_in_place(result, path, value)
    return result


def resolve_path(path_value: Any, base_dir: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def sort_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    sort_cols = [column for column in columns if column in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def compare_values(left: Any, right: Any) -> bool:
    if is_missing(left) and is_missing(right):
        return True
    return left == right


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted average while ignoring missing values and zero weights."""
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = values.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def pct(numerator: float, denominator: float) -> float:
    """Safe division for report ratios."""
    denominator = to_number(denominator, 0.0)
    if denominator == 0:
        return np.nan
    return to_number(numerator, 0.0) / denominator


def json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    return value
