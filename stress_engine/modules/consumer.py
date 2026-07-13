"""Consumer stress module."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .base import module_population, record_out_of_scope
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
    config = scenario.get("modules", {}).get("Consumer", scenario.get("modules", {}).get("consumer", {}))
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
    fallback_events: set[tuple[str, str, str]] = set()

    for idx, row in out.loc[mask].iterrows():
        out.at[idx, "module_applied"] = _append_module(out.at[idx, "module_applied"], "Consumer")
        # The candidates lists let scenarios choose current/origination fields.
        # `_latest_candidate` uses date and configured order for deterministic
        # selection.
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
        base_reserve = to_number(row.get(reserve_field), 0.0) if reserve_field_available else np.nan
        qualitative_reserve = base_reserve - (base_pd * unstressed_lgd)
        if not is_missing(qualitative_floor):
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
            _log_consumer_default(
                exceptions, fallback_events, "pd_increase_factor", segment, level, factor_source
            )
            _log_consumer_default(
                exceptions, fallback_events, "collateral_value_factor", segment, level, collateral_source
            )
            rushed_sale_discount = to_number(config.get("rushed_sale_discount"), np.nan)
            closing_costs = to_number(config.get("closing_costs"), np.nan)
            invalid_assumptions = []
            if is_missing(factor) or factor < 0:
                invalid_assumptions.append("pd_increase_factor")
            if is_missing(collateral_factor) or collateral_factor < 0:
                invalid_assumptions.append("collateral_value_factor")
            if is_missing(rushed_sale_discount) or not 0 <= rushed_sale_discount <= 1:
                invalid_assumptions.append("rushed_sale_discount")
            if is_missing(closing_costs) or not 0 <= closing_costs <= 1:
                invalid_assumptions.append("closing_costs")
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
            stressed_value = to_number(appraisal) * collateral_factor * (1 - rushed_sale_discount) * (1 - closing_costs)
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
    """Return the FICO-to-PD lookup table from JSON or a loaded source."""
    if "pd_lookup" in config:
        return pd.DataFrame(config["pd_lookup"])
    source = config.get("pd_lookup_source")
    if source:
        if source not in inputs:
            raise ValueError(f"Consumer PD lookup source '{source}' was not loaded.")
        return inputs[source].frame.copy()
    raise ValueError("Consumer module requires pd_lookup or pd_lookup_source.")


def _lookup_pd(score: float, table: pd.DataFrame) -> float:
    """Map FICO score to PD using exact score or min/max band rows."""
    if table.empty:
        return np.nan
    if "fico" in table.columns and "pd" in table.columns:
        exact = table[pd.to_numeric(table["fico"], errors="coerce") == score]
        if len(exact) == 1:
            return to_number(exact.iloc[0]["pd"])
        if len(exact) > 1:
            return np.nan
    min_cols = [col for col in ("min_score", "fico_min", "min_fico") if col in table.columns]
    max_cols = [col for col in ("max_score", "fico_max", "max_fico") if col in table.columns]
    pd_col = "pd" if "pd" in table.columns else "probability_of_default"
    if min_cols and max_cols and pd_col in table.columns:
        lo = pd.to_numeric(table[min_cols[0]], errors="coerce")
        hi = pd.to_numeric(table[max_cols[0]], errors="coerce")
        match = table[(score >= lo) & (score <= hi)]
        if len(match) == 1:
            return to_number(match.iloc[0][pd_col])
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
    min_cols = [col for col in ("min_score", "fico_min", "min_fico") if col in table.columns]
    max_cols = [col for col in ("max_score", "fico_max", "max_fico") if col in table.columns]
    pd_col = "pd" if "pd" in table.columns else "probability_of_default"
    if not min_cols or not max_cols or pd_col not in table.columns:
        if not ({"fico", "pd"} <= set(table.columns)):
            record_exception(
                exceptions,
                "ERROR",
                "Consumer",
                "CONSUMER_PD_LOOKUP_COLUMNS_INVALID",
                "Consumer PD lookup table did not contain an exact-score or min/max band structure.",
                field="fico_pd_lookup",
            )
        return
    work = pd.DataFrame(
        {
            "lo": pd.to_numeric(table[min_cols[0]], errors="coerce"),
            "hi": pd.to_numeric(table[max_cols[0]], errors="coerce"),
            "pd": pd.to_numeric(table[pd_col], errors="coerce"),
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


def _latest_candidate(row: Mapping[str, Any], candidates: List[Mapping[str, Any]], value_key: str) -> tuple[float, str | None]:
    """Pick the latest usable candidate value by date.

    Called for both FICO scores and appraisals. Blank values are skipped, and
    zeros are skipped by default because neither a FICO score nor an appraisal
    of zero should block fallback to a lower-priority candidate.
    """
    available: List[tuple[pd.Timestamp, int, float, str]] = []
    for order, candidate in enumerate(candidates):
        field = candidate.get(value_key)
        if not field:
            continue
        value = row.get(field)
        if pd.isna(value):
            continue
        numeric_value = to_number(value)
        if candidate.get("treat_zero_as_missing", True) and numeric_value == 0:
            continue
        date_field = candidate.get("date_field")
        date = pd.to_datetime(row.get(date_field), errors="coerce") if date_field else pd.Timestamp.min
        if pd.isna(date):
            date = pd.Timestamp.min
        available.append((date, -order, numeric_value, field))
    if not available:
        first_field = candidates[0].get(value_key) if candidates else None
        return np.nan, first_field
    available.sort()
    _, _, value, field = available[-1]
    return value, field


def _append_module(existing: Any, module: str) -> str:
    """Append the module name to the borrower-level module audit field."""
    if not existing:
        return module
    pieces = [item for item in str(existing).split(";") if item]
    if module not in pieces:
        pieces.append(module)
    return ";".join(pieces)


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
