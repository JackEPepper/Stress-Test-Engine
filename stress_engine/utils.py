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
ORDER_BUCKET = {value: key for key, value in BUCKET_ORDER.items()}


def stable_name(value: Any) -> str:
    """Return a predictable, column-safe name."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unnamed"


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_number(value: Any, default: float = np.nan) -> float:
    if is_missing(value):
        return default
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    text = text.replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return default


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def coerce_numeric_frame(df: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for field in fields:
        if field in out.columns:
            out[field] = pd.to_numeric(
                out[field].astype(str).str.replace(r"[$,%]", "", regex=True).str.replace(",", ""),
                errors="coerce",
            )
    return out


def parse_date(value: Any) -> pd.Timestamp:
    if is_missing(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def parse_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def risk_bucket_from_rating(rating: Any) -> str:
    numeric = to_number(rating)
    if math.isnan(numeric):
        return "Unknown"
    if numeric < 7:
        return "Pass"
    if numeric == 7:
        return "Special Mention"
    return "Substandard"


def worse_bucket(left: str, right: str) -> str:
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
    """Annual amortizing debt payment."""
    balance = to_number(balance, 0.0)
    annual_rate = to_number(annual_rate, 0.0)
    amortization_years = to_number(amortization_years, 0.0)
    if balance <= 0 or amortization_years <= 0:
        return np.nan
    if abs(annual_rate) < 1e-12:
        return balance / amortization_years
    return balance * annual_rate / (1 - (1 + annual_rate) ** (-amortization_years))


def get_levels(scenario: Mapping[str, Any]) -> List[str]:
    return [str(level) for level in scenario.get("stress_levels", ["S1", "S2"])]


def lookup_parameter(table: Any, subsector: Any, level: str, default: Any = np.nan) -> Any:
    """Lookup a scenario assumption supporting default/all and level maps.

    Supported forms:
      {"default": {"S1": 0.1}, "Office": {"S1": 0.2}}
      {"S1": 0.1, "S2": 0.2}
      0.1
    """
    if isinstance(table, Mapping):
        sector_key = str(subsector) if not is_missing(subsector) else None
        choices: List[Any] = []
        if sector_key and sector_key in table:
            choices.append(table[sector_key])
        for key in ("default", "all", "*"):
            if key in table:
                choices.append(table[key])
        if level in table:
            return table[level]
        for candidate in choices:
            if isinstance(candidate, Mapping):
                if level in candidate:
                    return candidate[level]
                for key in ("default", "all", "*"):
                    if key in candidate:
                        return candidate[key]
            elif candidate is not None:
                return candidate
        return default
    if table is None:
        return default
    return table


def ensure_columns(df: pd.DataFrame, fields: Sequence[str], context: str) -> None:
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


def set_json_path(data: Any, path: str, value: Any) -> Any:
    """Return a deep copy with one flattened-json path replaced."""
    result = copy.deepcopy(data)
    tokens: List[Any] = []
    for match in _PATH_TOKEN_RE.finditer(path):
        token = match.group(1) if match.group(1) is not None else int(match.group(2))
        tokens.append(token)
    cursor = result
    for token in tokens[:-1]:
        cursor = cursor[token]
    cursor[tokens[-1]] = copy.deepcopy(value)
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
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = values.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def pct(numerator: float, denominator: float) -> float:
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
