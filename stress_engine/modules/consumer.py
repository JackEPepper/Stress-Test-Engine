"""Consumer stress module."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .base import module_population, record_out_of_scope
from ..utils import get_levels, is_missing, lookup_parameter, to_number


def run_consumer(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = scenario.get("modules", {}).get("Consumer", scenario.get("modules", {}).get("consumer", {}))
    if not config or not config.get("enabled", True):
        return results, pd.DataFrame()
    out = results.copy()
    levels = get_levels(scenario)
    mask = module_population(out, scenario, config)
    borrower_cfg = scenario["borrower"]
    balance_field = config.get("balance_field", borrower_cfg["balance_field"])
    pd_table = _pd_lookup_table(config, inputs)
    out_scope: List[Dict[str, Any]] = []

    for idx, row in out.loc[mask].iterrows():
        out.at[idx, "module_applied"] = _append_module(out.at[idx, "module_applied"], "Consumer")
        fico, fico_field = _latest_candidate(row, config.get("fico_candidates", [{"score_field": "fico_score", "date_field": "fico_date"}]), "score_field")
        appraisal, appraisal_field = _latest_candidate(
            row,
            config.get("appraisal_candidates", [{"value_field": "appraised_value", "date_field": "appraisal_date"}]),
            "value_field",
        )
        balance = to_number(row.get(balance_field))
        missing = []
        if is_missing(fico):
            missing.append(fico_field or "fico")
        if is_missing(appraisal):
            missing.append(appraisal_field or "appraised_value")
        if is_missing(balance):
            missing.append(balance_field)
        if missing:
            for level in levels:
                out.at[idx, f"out_of_scope_{level}"] = True
                record_out_of_scope(out_scope, row, scenario, "Consumer", level, "PD/LGD", missing, "missing_required_field")
            continue

        base_pd = _lookup_pd(to_number(fico), pd_table)
        if is_missing(base_pd):
            for level in levels:
                out.at[idx, f"out_of_scope_{level}"] = True
                record_out_of_scope(out_scope, row, scenario, "Consumer", level, "PD", ["fico_pd_lookup"], "missing_pd_lookup")
            continue

        unstressed_lgd = max(balance - to_number(appraisal), 0.0)
        out.at[idx, "consumer_fico"] = to_number(fico)
        out.at[idx, "consumer_appraised_value"] = to_number(appraisal)
        out.at[idx, "consumer_pd_unstressed"] = base_pd
        out.at[idx, "consumer_lgd_unstressed"] = unstressed_lgd
        out.at[idx, "consumer_lgd_ratio_unstressed"] = unstressed_lgd / balance if balance else np.nan
        out.at[idx, "consumer_el_unstressed"] = base_pd * unstressed_lgd

        for level in levels:
            factor = to_number(lookup_parameter(config.get("pd_increase_factor"), row.get(config.get("segment_field", "")), level), 1.0)
            pd_value = min(base_pd * factor, float(config.get("pd_cap", 1.0)))
            collateral_factor = to_number(lookup_parameter(config.get("collateral_value_factor"), row.get(config.get("segment_field", "")), level), 1.0)
            rushed_sale_discount = to_number(config.get("rushed_sale_discount", 0.0), 0.0)
            closing_costs = to_number(config.get("closing_costs", 0.0), 0.0)
            stressed_value = to_number(appraisal) * collateral_factor * (1 - rushed_sale_discount) * (1 - closing_costs)
            lgd = max(balance - stressed_value, 0.0)
            out.at[idx, f"consumer_pd_{level}"] = pd_value
            out.at[idx, f"consumer_stressed_collateral_value_{level}"] = stressed_value
            out.at[idx, f"consumer_lgd_{level}"] = lgd
            out.at[idx, f"consumer_lgd_ratio_{level}"] = lgd / balance if balance else np.nan
            out.at[idx, f"consumer_el_{level}"] = pd_value * lgd
    return out, pd.DataFrame(out_scope)


def _pd_lookup_table(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> pd.DataFrame:
    if "pd_lookup" in config:
        return pd.DataFrame(config["pd_lookup"])
    source = config.get("pd_lookup_source")
    if source:
        if source not in inputs:
            raise ValueError(f"Consumer PD lookup source '{source}' was not loaded.")
        return inputs[source].frame.copy()
    raise ValueError("Consumer module requires pd_lookup or pd_lookup_source.")


def _lookup_pd(score: float, table: pd.DataFrame) -> float:
    if table.empty:
        return np.nan
    if "fico" in table.columns and "pd" in table.columns:
        exact = table[pd.to_numeric(table["fico"], errors="coerce") == score]
        if not exact.empty:
            return to_number(exact.iloc[0]["pd"])
    min_cols = [col for col in ("min_score", "fico_min", "min_fico") if col in table.columns]
    max_cols = [col for col in ("max_score", "fico_max", "max_fico") if col in table.columns]
    pd_col = "pd" if "pd" in table.columns else "probability_of_default"
    if min_cols and max_cols and pd_col in table.columns:
        lo = pd.to_numeric(table[min_cols[0]], errors="coerce")
        hi = pd.to_numeric(table[max_cols[0]], errors="coerce")
        match = table[(score >= lo) & (score <= hi)]
        if not match.empty:
            return to_number(match.iloc[0][pd_col])
    return np.nan


def _latest_candidate(row: Mapping[str, Any], candidates: List[Mapping[str, Any]], value_key: str) -> tuple[float, str | None]:
    available: List[tuple[pd.Timestamp, int, float, str]] = []
    for order, candidate in enumerate(candidates):
        field = candidate.get(value_key)
        if not field:
            continue
        value = row.get(field)
        if pd.isna(value):
            continue
        date_field = candidate.get("date_field")
        date = pd.to_datetime(row.get(date_field), errors="coerce") if date_field else pd.Timestamp.min
        if pd.isna(date):
            date = pd.Timestamp.min
        available.append((date, -order, to_number(value), field))
    if not available:
        first_field = candidates[0].get(value_key) if candidates else None
        return np.nan, first_field
    available.sort()
    _, _, value, field = available[-1]
    return value, field


def _append_module(existing: Any, module: str) -> str:
    if not existing:
        return module
    pieces = [item for item in str(existing).split(";") if item]
    if module not in pieces:
        pieces.append(module)
    return ";".join(pieces)
