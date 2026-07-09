"""Borrower universe construction and deterministic enrichment."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from .io import LoadedTable
from .utils import as_list, ensure_columns, first_non_null, join_unique, parse_date_series, stable_name


def build_borrowers(identity: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    config = scenario.get("borrower", {})
    borrower_id = config["borrower_id_field"]
    balance_fields = set(config.get("sum_fields", config.get("balance_fields", [config["balance_field"]])))
    aggregations: Dict[str, Any] = dict(config.get("aggregation", {}))
    loan_id_field = config.get("loan_id_field")

    ensure_columns(identity, [borrower_id], "Identity input")
    for field in balance_fields:
        if field in identity.columns:
            aggregations.setdefault(field, "sum")
    if loan_id_field and loan_id_field in identity.columns:
        aggregations.setdefault(loan_id_field, "list_unique")

    agg_map: Dict[str, Any] = {}
    for column in identity.columns:
        if column in {borrower_id, "_source_row"}:
            continue
        method = aggregations.get(column, config.get("default_aggregation", "first"))
        agg_map[column] = _aggregation_callable(method)

    borrowers = (
        identity.sort_values([borrower_id, "_source_row"], kind="mergesort")
        .groupby(borrower_id, dropna=False)
        .agg(agg_map)
        .reset_index()
    )
    loan_counts = identity.groupby(borrower_id, dropna=False).size().rename("loan_count").reset_index()
    borrowers = borrowers.merge(loan_counts, on=borrower_id, how="left")
    borrowers = borrowers.sort_values([borrower_id], kind="mergesort").reset_index(drop=True)
    return borrowers


def enrich_borrowers(
    borrowers: pd.DataFrame,
    loaded: Mapping[str, LoadedTable],
    scenario: Mapping[str, Any],
) -> pd.DataFrame:
    out = borrowers.copy()
    borrower_id = scenario["borrower"]["borrower_id_field"]
    source_specs = scenario.get("inputs", {}).get("sources", {})
    for name in sorted(source_specs):
        spec = source_specs[name]
        if spec.get("merge", True) is False:
            continue
        if name not in loaded:
            continue
        aggregated = aggregate_source(name, loaded[name].frame, spec, borrower_id)
        if aggregated.empty:
            continue
        out = _merge_source(out, aggregated, borrower_id, spec)
    return out


def aggregate_source(name: str, df: pd.DataFrame, spec: Mapping[str, Any], borrower_id: str) -> pd.DataFrame:
    key = spec.get("key", borrower_id)
    merge_key = spec.get("merge_key", borrower_id)
    if key not in df.columns:
        raise ValueError(f"Source '{name}' key column '{key}' is missing.")
    frame = df.copy()
    for field in spec.get("date_columns", []):
        if field in frame.columns:
            frame[field] = parse_date_series(frame[field])
    aggregation = spec.get("aggregation", {})
    if not aggregation:
        aggregation = {
            column: "first"
            for column in frame.columns
            if column not in {key, "_source_row"}
        }

    pieces: List[pd.DataFrame] = []
    for output_field, method_spec in aggregation.items():
        piece = _aggregate_field(frame, key, output_field, method_spec)
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=[merge_key])
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=key, how="outer")
    if merge_key != key:
        result = result.rename(columns={key: merge_key})
    return result.sort_values([merge_key], kind="mergesort").reset_index(drop=True)


def _aggregate_field(df: pd.DataFrame, key: str, output_field: str, method_spec: Any) -> pd.DataFrame:
    if isinstance(method_spec, str):
        method = method_spec
        field = output_field
        options: Mapping[str, Any] = {}
    else:
        options = method_spec
        method = options.get("method", "first")
        field = options.get("field", output_field)

    if method == "latest":
        date_field = options.get("date_field")
        if not date_field:
            raise ValueError(f"Latest aggregation for '{output_field}' requires date_field.")
        required = list(dict.fromkeys([key, field, date_field]))
        ensure_columns(df, required, f"Latest aggregation for '{output_field}'")
        work = df[required + (["_source_row"] if "_source_row" in df.columns else [])].copy()
        work[date_field] = parse_date_series(work[date_field])
        sort_cols = [key, date_field]
        if "_source_row" in work.columns:
            sort_cols.append("_source_row")
        work = work.sort_values(sort_cols, kind="mergesort")
        values = work.dropna(subset=[field]).groupby(key, dropna=False).tail(1)
        return values[[key, field]].rename(columns={field: output_field})

    if method == "unique_sum":
        unique_fields = as_list(options.get("unique_fields", [field]))
        required = [key, field] + [col for col in unique_fields if col != field]
        ensure_columns(df, required, f"Unique sum aggregation for '{output_field}'")
        work = df[required].drop_duplicates([key] + unique_fields)
        work[field] = pd.to_numeric(work[field], errors="coerce")
        return work.groupby(key, dropna=False)[field].sum(min_count=1).reset_index(name=output_field)

    ensure_columns(df, [key, field], f"Aggregation for '{output_field}'")
    func = _aggregation_callable(method)
    return df.groupby(key, dropna=False)[field].agg(func).reset_index(name=output_field)


def _aggregation_callable(method: Any) -> Any:
    if callable(method):
        return method
    if isinstance(method, MutableMapping):
        method = method.get("method", "first")
    method = str(method)
    if method == "first":
        return first_non_null
    if method == "last":
        return lambda values: values.dropna().iloc[-1] if not values.dropna().empty else np.nan
    if method == "sum":
        return lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)
    if method == "mean":
        return lambda values: pd.to_numeric(values, errors="coerce").mean()
    if method == "max":
        return _max_value
    if method == "min":
        return _min_value
    if method == "list_unique":
        return join_unique
    raise ValueError(f"Unsupported aggregation method: {method}")


def _max_value(values: pd.Series) -> Any:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    if pd.api.types.is_datetime64_any_dtype(clean):
        return clean.max()
    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.notna().all():
        return numeric.max()
    dates = pd.to_datetime(clean, errors="coerce")
    if dates.notna().all():
        return dates.max()
    if numeric.notna().any():
        return numeric.max()
    return clean.max()


def _min_value(values: pd.Series) -> Any:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    if pd.api.types.is_datetime64_any_dtype(clean):
        return clean.min()
    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.notna().all():
        return numeric.min()
    dates = pd.to_datetime(clean, errors="coerce")
    if dates.notna().all():
        return dates.min()
    if numeric.notna().any():
        return numeric.min()
    return clean.min()


def _merge_source(
    borrowers: pd.DataFrame,
    source: pd.DataFrame,
    borrower_id: str,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    required_tags = [f"tag_{stable_name(tag)}" for tag in as_list(spec.get("required_tags"))]
    on_conflict = spec.get("on_conflict", "fill_missing")
    result = borrowers.copy()
    mask = pd.Series(True, index=result.index)
    for tag_col in required_tags:
        if tag_col in result.columns:
            mask &= result[tag_col].fillna(False).astype(bool)
        else:
            mask &= False
    merged = result[[borrower_id]].merge(source, on=borrower_id, how="left")
    for column in source.columns:
        if column == borrower_id:
            continue
        values = merged[column]
        values = values.where(mask, np.nan)
        if column not in result.columns:
            result[column] = values
        elif on_conflict == "overwrite":
            result[column] = values.combine_first(result[column])
        elif on_conflict == "fill_missing":
            result[column] = result[column].combine_first(values)
        elif on_conflict == "suffix":
            result[f"{column}_{stable_name(spec.get('name', 'source'))}"] = values
        elif on_conflict == "error":
            raise ValueError(f"Enrichment column conflict for '{column}'.")
        else:
            raise ValueError(f"Unsupported on_conflict mode: {on_conflict}")
    return result
