"""Consumer stress module."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .base import append_module, module_population, record_out_of_scope, targeted_parameter
from ..exceptions import record_exception
from ..utils import get_levels, is_missing, lookup_parameter_with_source, to_number


def run_consumer(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    inputs: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run consumer PD/LGD/EL stress for borrowers whose primary module is Consumer.

    Called from `StressEngine.run`. Consumer does not migrate commercial risk
    buckets; it writes probability of default, loss given default, and expected
    loss metrics for base and stress levels.
    """
    exceptions = exceptions if exceptions is not None else []
    config = scenario.get("modules", {}).get("Consumer", {})
    if not config or not config.get("enabled", True):
        return results, pd.DataFrame()
    config = dict(config)
    config["_module_name"] = "Consumer"
    out = results.copy()
    levels = get_levels(scenario)
    mask = module_population(out, scenario, config)
    borrower_cfg = scenario["borrower"]
    balance_field = config.get("balance_field", borrower_cfg["balance_field"])
    pd_table = _pd_lookup_table(config, inputs)
    _validate_pd_lookup(pd_table, exceptions)
    out_scope: List[Dict[str, Any]] = []
    reserve_field = scenario.get("cecl", {}).get("reserve_field", "cecl_reserve")
    qualitative_floor = to_number(config.get("qualitative_reserve_floor"), np.nan)
    reserve_field_available = reserve_field in out.columns
    rushed_sale_discount = to_number(config.get("rushed_sale_discount"), np.nan)
    closing_costs = to_number(config.get("closing_costs"), np.nan)
    liquidation_assumption_errors = []
    if is_missing(rushed_sale_discount) or not 0 <= rushed_sale_discount <= 1:
        liquidation_assumption_errors.append("rushed_sale_discount")
    if is_missing(closing_costs) or not 0 <= closing_costs <= 1:
        liquidation_assumption_errors.append("closing_costs")
    liquidation_factor = (
        (1 - rushed_sale_discount) * (1 - closing_costs)
        if not liquidation_assumption_errors
        else np.nan
    )
    fallback_events: set[tuple[str, str, str]] = set()

    for idx, row in out.loc[mask].iterrows():
        out.at[idx, "module_applied"] = append_module(out.at[idx, "module_applied"], "Consumer")
        fico_field = config["fico_field"]
        appraisal_field = config["appraisal_field"]
        fico = row.get(fico_field)
        appraisal = row.get(appraisal_field)
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

        unstressed_collateral_value = to_number(appraisal) * liquidation_factor
        unstressed_lgd = (
            max(balance - unstressed_collateral_value, 0.0)
            if not is_missing(unstressed_collateral_value)
            else np.nan
        )
        unstressed_el = (
            base_pd * unstressed_lgd
            if not is_missing(unstressed_lgd)
            else np.nan
        )
        out.at[idx, "consumer_fico"] = to_number(fico)
        out.at[idx, "consumer_appraised_value"] = to_number(appraisal)
        out.at[idx, "consumer_collateral_value_unstressed"] = unstressed_collateral_value
        out.at[idx, "consumer_pd_unstressed"] = base_pd
        out.at[idx, "consumer_lgd_unstressed"] = unstressed_lgd
        out.at[idx, "consumer_lgd_ratio_unstressed"] = unstressed_lgd / balance if balance else np.nan
        out.at[idx, "consumer_el_unstressed"] = unstressed_el
        base_reserve = to_number(row.get(reserve_field), 0.0) if reserve_field_available else np.nan
        qualitative_reserve = (
            base_reserve - unstressed_el
            if not is_missing(base_reserve) and not is_missing(unstressed_el)
            else np.nan
        )
        if not is_missing(qualitative_reserve) and not is_missing(qualitative_floor):
            qualitative_reserve = max(qualitative_reserve, qualitative_floor)
        out.at[idx, "consumer_cecl_reserve_base"] = base_reserve
        out.at[idx, "consumer_qualitative_reserve"] = qualitative_reserve

        for level in levels:
            segment = row.get(config.get("segment_field", ""))
            factor_raw, factor_source = lookup_parameter_with_source(
                config.get("pd_increase_factor"), segment, level, np.nan
            )
            collateral_raw, collateral_source = lookup_parameter_with_source(
                config.get("collateral_value_factor"), segment, level, np.nan
            )
            factor = to_number(factor_raw, np.nan)
            collateral_factor = to_number(collateral_raw, np.nan)
            factor = targeted_parameter(
                row, scenario, "Consumer", "pd_increase_factor", level, factor
            )
            collateral_factor = targeted_parameter(
                row,
                scenario,
                "Consumer",
                "collateral_value_factor",
                level,
                collateral_factor,
            )
            level_rushed_sale_discount = targeted_parameter(
                row,
                scenario,
                "Consumer",
                "rushed_sale_discount",
                level,
                rushed_sale_discount,
            )
            level_closing_costs = targeted_parameter(
                row, scenario, "Consumer", "closing_costs", level, closing_costs
            )
            _log_consumer_default(
                exceptions, fallback_events, "pd_increase_factor", segment, level, factor_source
            )
            _log_consumer_default(
                exceptions, fallback_events, "collateral_value_factor", segment, level, collateral_source
            )
            invalid_assumptions = []
            if is_missing(level_rushed_sale_discount) or not 0 <= level_rushed_sale_discount <= 1:
                invalid_assumptions.append("rushed_sale_discount")
            if is_missing(level_closing_costs) or not 0 <= level_closing_costs <= 1:
                invalid_assumptions.append("closing_costs")
            if is_missing(factor) or factor < 0:
                invalid_assumptions.append("pd_increase_factor")
            if is_missing(collateral_factor) or collateral_factor < 0:
                invalid_assumptions.append("collateral_value_factor")
            if invalid_assumptions:
                out.at[idx, f"out_of_scope_{level}"] = True
                record_out_of_scope(
                    out_scope,
                    row,
                    scenario,
                    "Consumer",
                    level,
                    "PD/LGD",
                    invalid_assumptions,
                    "missing_or_invalid_scenario_assumption",
                )
                for field in invalid_assumptions:
                    record_exception(
                        exceptions,
                        "ERROR",
                        "Consumer",
                        "CONSUMER_SCENARIO_ASSUMPTION_INVALID",
                        "Consumer stress assumption was missing or outside its acceptable range; this level was not calculated.",
                        borrower_id=row.get(borrower_cfg.get("borrower_id_field", "borrower_id")),
                        stress_level=level,
                        module="Consumer",
                        field=field,
                    )
                continue
            pd_value = max(0.0, min(base_pd * factor, float(config.get("pd_cap", 1.0))))
            # Stressed collateral value applies the scenario collateral factor
            # and global liquidation impacts before LGD is capped at zero.
            level_liquidation_factor = (
                (1 - level_rushed_sale_discount) * (1 - level_closing_costs)
            )
            stressed_value = (
                to_number(appraisal) * collateral_factor * level_liquidation_factor
            )
            lgd = max(balance - stressed_value, 0.0)
            out.at[idx, f"consumer_pd_{level}"] = pd_value
            out.at[idx, f"consumer_stressed_collateral_value_{level}"] = stressed_value
            out.at[idx, f"consumer_lgd_{level}"] = lgd
            out.at[idx, f"consumer_lgd_ratio_{level}"] = lgd / balance if balance else np.nan
            out.at[idx, f"consumer_el_{level}"] = pd_value * lgd
            if reserve_field_available:
                out.at[idx, f"consumer_proforma_cecl_{level}"] = pd_value * lgd + qualitative_reserve
    return out, pd.DataFrame(out_scope)


def _pd_lookup_table(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> pd.DataFrame:
    """Return the configured FICO-to-PD input table."""
    source = config["pd_lookup_source"]
    if source not in inputs:
        raise ValueError(f"Consumer PD lookup source '{source}' was not loaded.")
    return inputs[source].frame.copy()


def _lookup_pd(score: float, table: pd.DataFrame) -> float:
    """Map a FICO score to one canonical min/max PD band."""
    required = {"min_score", "max_score", "pd"}
    if table.empty or not required <= set(table.columns):
        return np.nan
    lo = pd.to_numeric(table["min_score"], errors="coerce")
    hi = pd.to_numeric(table["max_score"], errors="coerce")
    match = table[(score >= lo) & (score <= hi)]
    if len(match) == 1:
        return to_number(match.iloc[0]["pd"])
    return np.nan


def _validate_pd_lookup(table: pd.DataFrame, exceptions: List[Dict[str, Any]]) -> None:
    """Log malformed, overlapping, and gapped FICO-to-PD lookup bands."""
    if table.empty:
        record_exception(
            exceptions,
            "ERROR",
            "Consumer",
            "CONSUMER_PD_LOOKUP_EMPTY",
            "Consumer PD lookup table was empty.",
            field="fico_pd_lookup",
        )
        return
    required = {"min_score", "max_score", "pd"}
    if not required <= set(table.columns):
        record_exception(
            exceptions,
            "ERROR",
            "Consumer",
            "CONSUMER_PD_LOOKUP_COLUMNS_INVALID",
            "Consumer PD lookup table did not contain min_score, max_score, and pd.",
            field="fico_pd_lookup",
        )
        return
    work = pd.DataFrame(
        {
            "lo": pd.to_numeric(table["min_score"], errors="coerce"),
            "hi": pd.to_numeric(table["max_score"], errors="coerce"),
            "pd": pd.to_numeric(table["pd"], errors="coerce"),
        }
    ).sort_values(["lo", "hi"], kind="mergesort")
    invalid = work[work[["lo", "hi", "pd"]].isna().any(axis=1) | (work["lo"] > work["hi"]) | ~work["pd"].between(0, 1)]
    if not invalid.empty:
        record_exception(
            exceptions,
            "ERROR",
            "Consumer",
            "CONSUMER_PD_LOOKUP_VALUES_INVALID",
            "Consumer PD lookup contained missing values, reversed bands, or PD values outside 0 to 1.",
            field="fico_pd_lookup",
            details=f"invalid_row_count={len(invalid)}",
        )
    previous_hi = None
    overlaps = 0
    gaps = 0
    for _, band in work.dropna().iterrows():
        if previous_hi is not None:
            if band["lo"] <= previous_hi:
                overlaps += 1
            elif band["lo"] > previous_hi + 1:
                gaps += 1
        previous_hi = max(previous_hi, band["hi"]) if previous_hi is not None else band["hi"]
    if overlaps:
        record_exception(
            exceptions,
            "ERROR",
            "Consumer",
            "CONSUMER_PD_LOOKUP_BANDS_OVERLAP",
            "Consumer PD lookup bands overlapped; scores with multiple matches are out of scope.",
            field="fico_pd_lookup",
            details=f"overlap_count={overlaps}",
        )
    if gaps:
        record_exception(
            exceptions,
            "WARNING",
            "Consumer",
            "CONSUMER_PD_LOOKUP_BANDS_GAPPED",
            "Consumer PD lookup bands contained score gaps; unmatched scores are out of scope.",
            field="fico_pd_lookup",
            details=f"gap_count={gaps}",
        )


def _log_consumer_default(
    exceptions: List[Dict[str, Any]],
    events: set[tuple[str, str, str]],
    parameter: str,
    segment: Any,
    level: str,
    source: str,
) -> None:
    """Log configured default-table use for segmented Consumer assumptions."""
    event = (parameter, str(segment), level)
    if source not in {"default", "sector_default"} or event in events:
        return
    events.add(event)
    record_exception(
        exceptions,
        "INFO",
        "Consumer",
        "SCENARIO_DEFAULT_PARAMETER_USED",
        "Consumer assumption used a configured default value rather than a segment-specific value.",
        stress_level=level,
        module="Consumer",
        field=parameter,
        details=f"segment={segment}; source={source}",
    )
