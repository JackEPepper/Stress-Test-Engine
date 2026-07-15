"""Commercial Real Estate stress module."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .base import append_module, missing_fields, module_population, record_out_of_scope
from ..exceptions import record_exception
from ..utils import (
    annual_debt_payment,
    get_levels,
    higher_metric_bucket,
    is_missing,
    lookup_parameter_with_source,
    lower_metric_bucket,
    to_number,
    worse_bucket,
)


def run_cre(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run CRE stresses for borrowers whose `primary_module` is CRE.

    Called from `StressEngine.run`. The shared `module_population` helper
    applies eligible tags plus module-order resolution before this function
    loops through stress levels and borrower rows.
    """
    exceptions = exceptions if exceptions is not None else []
    config = scenario.get("modules", {}).get("CRE", {})
    if not config or not config.get("enabled", True):
        return results, pd.DataFrame()
    config = dict(config)
    config["_module_name"] = "CRE"

    out = results.copy()
    levels = get_levels(scenario)
    mask = module_population(out, scenario, config)
    borrower_cfg = scenario["borrower"]
    balance_field = config.get("balance_field", borrower_cfg["balance_field"])
    maturity_field = config.get("maturity_date_field", borrower_cfg.get("maturity_date_field", "maturity_date"))
    subsector_field = config.get("subsector_field", "cre_subsector")
    cutoff_date = pd.to_datetime(scenario["run"]["cutoff_date"])
    threshold_months = int(config.get("maturity_threshold_months", 12))
    threshold_date = cutoff_date + pd.DateOffset(months=threshold_months)
    out_scope: List[Dict[str, Any]] = []
    fallback_events: set[tuple[str, str, str]] = set()

    for level in levels:
        for idx, row in out.loc[mask].iterrows():
            out.at[idx, "module_applied"] = append_module(out.at[idx, "module_applied"], "CRE")
            base_bucket = row.get("base_bucket", "Unknown")
            if base_bucket == "Substandard":
                # Substandard is already the highest modeled commercial bucket,
                # so the stress cannot migrate it further.
                out.at[idx, f"stressed_bucket_{level}"] = "Substandard"
                continue
            maturity = pd.to_datetime(row.get(maturity_field), errors="coerce")
            if pd.isna(maturity):
                _mark_out(out, idx, level, row, scenario, out_scope, "maturity_split", [maturity_field], "missing_maturity")
                continue
            if maturity > threshold_date:
                # Longer-dated CRE loans are tested on DSCR only.
                stressed_bucket = _run_dscr(
                    out, idx, row, scenario, config, level, subsector_field, out_scope, exceptions, fallback_events
                )
            else:
                # Near-maturity CRE loans are tested on refinance DSCR and LTV.
                stressed_bucket = _run_refi_ltv(
                    out,
                    idx,
                    row,
                    scenario,
                    config,
                    level,
                    subsector_field,
                    balance_field,
                    out_scope,
                    exceptions,
                    fallback_events,
                )
            if stressed_bucket:
                # An unknown in-place rating cannot be represented as a valid
                # migration origin, so retain Unknown while still saving metrics.
                out.at[idx, f"stressed_bucket_{level}"] = (
                    "Unknown" if base_bucket == "Unknown" else worse_bucket(base_bucket, stressed_bucket)
                )
    return out, pd.DataFrame(out_scope)


def _run_dscr(
    out: pd.DataFrame,
    idx: int,
    row: Mapping[str, Any],
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
    level: str,
    subsector_field: str,
    out_scope: List[Dict[str, Any]],
    exceptions: List[Dict[str, Any]],
    fallback_events: set[tuple[str, str, str]],
) -> str | None:
    """Apply the CRE DSCR test and return its migrated bucket.

    Called only from `run_cre` for loans maturing after the configured
    threshold date. Formula: stressed DSCR = current DSCR * (1 - decline).
    """
    dscr_cfg = config.get("tests", {}).get("dscr", {})
    if not dscr_cfg.get("enabled", True):
        return row.get("base_bucket", "Pass")
    dscr_field = dscr_cfg.get("field", "dscr")
    required = [dscr_field, subsector_field]
    missing = missing_fields(row, required)
    if missing:
        _mark_out(out, idx, level, row, scenario, out_scope, "DSCR", missing, "missing_required_field")
        return None
    dscr = to_number(row.get(dscr_field))
    if not _in_range(dscr, dscr_cfg.get("acceptable_range")):
        _mark_out(out, idx, level, row, scenario, out_scope, "DSCR", [dscr_field], "outside_acceptable_range")
        return None
    decline = _resolve_assumption(
        dscr_cfg.get("decline"),
        row.get(subsector_field),
        level,
        "dscr_decline",
        exceptions,
        fallback_events,
    )
    if is_missing(decline) or not 0 <= decline <= 1:
        _mark_out(out, idx, level, row, scenario, out_scope, "DSCR", ["dscr_decline"], "missing_or_invalid_scenario_assumption")
        return None
    stressed_dscr = dscr * (1 - decline)
    out.at[idx, f"cre_dscr_{level}"] = stressed_dscr
    return lower_metric_bucket(stressed_dscr, dscr_cfg.get("cutoffs", {}))


def _run_refi_ltv(
    out: pd.DataFrame,
    idx: int,
    row: Mapping[str, Any],
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
    level: str,
    subsector_field: str,
    balance_field: str,
    out_scope: List[Dict[str, Any]],
    exceptions: List[Dict[str, Any]],
    fallback_events: set[tuple[str, str, str]],
) -> str | None:
    """Apply near-maturity refinance DSCR and LTV tests.

    Called from `run_cre` for loans inside the maturity threshold. It returns
    the worst bucket from base rating, refinance DSCR, and LTV.
    """
    tests = config.get("tests", {})
    refi_cfg = tests.get("refinance", {})
    ltv_cfg = tests.get("ltv", {})
    noi_field = refi_cfg.get("noi_field", ltv_cfg.get("noi_field", "noi"))
    required = [subsector_field, balance_field, noi_field]
    missing = missing_fields(row, required)
    if missing:
        _mark_out(out, idx, level, row, scenario, out_scope, "Refinance/LTV", missing, "missing_required_field")
        return None
    balance = to_number(row.get(balance_field))
    noi = to_number(row.get(noi_field))
    if balance <= 0:
        _mark_out(out, idx, level, row, scenario, out_scope, "Refinance/LTV", [balance_field], "nonpositive_balance")
        return None
    ratio = noi / balance if balance else np.nan
    if not _in_range(ratio, config.get("noi_balance_ratio_acceptable_range")):
        _mark_out(out, idx, level, row, scenario, out_scope, "Refinance/LTV", [noi_field, balance_field], "outside_noi_balance_ratio_range")
        return None

    best = row.get("base_bucket", "Pass")
    if refi_cfg.get("enabled", True):
        dscr_cfg = tests.get("dscr", {})
        decline_table = refi_cfg.get("noi_decline", dscr_cfg.get("decline"))
        sector = row.get(subsector_field)
        noi_decline = _resolve_assumption(
            decline_table, sector, level, "refinance_noi_decline", exceptions, fallback_events
        )
        spread = _resolve_assumption(
            refi_cfg.get("credit_spreads"), sector, level, "credit_spread", exceptions, fallback_events
        )
        treasury = to_number(refi_cfg.get("treasury_rate"), np.nan)
        amort_years = _resolve_assumption(
            refi_cfg.get("amortization_years"), sector, level, "amortization_years", exceptions, fallback_events
        )
        invalid_assumptions = []
        if is_missing(noi_decline) or not 0 <= noi_decline <= 1:
            invalid_assumptions.append("refinance_noi_decline")
        if is_missing(spread) or spread < 0:
            invalid_assumptions.append("credit_spread")
        if is_missing(treasury) or treasury < 0:
            invalid_assumptions.append("treasury_rate")
        if is_missing(amort_years) or amort_years <= 0:
            invalid_assumptions.append("amortization_years")
        if invalid_assumptions:
            _mark_out(
                out,
                idx,
                level,
                row,
                scenario,
                out_scope,
                "Refinance",
                invalid_assumptions,
                "missing_or_invalid_scenario_assumption",
            )
            for field in invalid_assumptions:
                _record_invalid_assumption(exceptions, field, sector, level)
            return None
        # Annual debt payment uses a standard amortizing loan formula from
        # `utils.annual_debt_payment`: balance, stressed rate, amortization.
        payment = annual_debt_payment(balance, treasury + spread, amort_years)
        if is_missing(payment) or payment <= 0:
            _mark_out(out, idx, level, row, scenario, out_scope, "Refinance", ["amortization_years"], "invalid_debt_payment")
            return None
        stressed_noi = noi * (1 - noi_decline)
        stressed_dscr = stressed_noi / payment
        out.at[idx, f"cre_refi_dscr_{level}"] = stressed_dscr
        out.at[idx, f"cre_dscr_{level}"] = stressed_dscr
        best = worse_bucket(best, lower_metric_bucket(stressed_dscr, refi_cfg.get("cutoffs", {})))

    if ltv_cfg.get("enabled", True):
        cap_rate = _resolve_assumption(
            ltv_cfg.get("cap_rates"),
            row.get(subsector_field),
            level,
            "cap_rate",
            exceptions,
            fallback_events,
        )
        if is_missing(cap_rate) or cap_rate <= 0 or noi <= 0:
            _mark_out(out, idx, level, row, scenario, out_scope, "LTV", ["cap_rates", noi_field], "invalid_ltv_inputs")
            return None
        # Per requirement, stressed LTV is balance times stressed cap rate
        # divided by NOI. Higher values map to worse buckets.
        stressed_ltv = balance * cap_rate / noi
        out.at[idx, f"cre_ltv_{level}"] = stressed_ltv
        best = worse_bucket(best, higher_metric_bucket(stressed_ltv, ltv_cfg.get("cutoffs", {})))
    return best


def _resolve_assumption(
    table: Any,
    sector: Any,
    level: str,
    parameter: str,
    exceptions: List[Dict[str, Any]],
    fallback_events: set[tuple[str, str, str]],
) -> float:
    """Resolve a CRE assumption and log intentional default-table use once."""
    raw, source = lookup_parameter_with_source(table, sector, level, np.nan)
    event = (parameter, str(sector), level)
    if source in {"default", "sector_default"} and event not in fallback_events:
        fallback_events.add(event)
        record_exception(
            exceptions,
            "INFO",
            "CRE",
            "SCENARIO_DEFAULT_PARAMETER_USED",
            "CRE assumption used a configured default value rather than a subsector-specific value.",
            stress_level=level,
            module="CRE",
            field=parameter,
            details=f"subsector={sector}; source={source}",
        )
    if source == "missing" and event not in fallback_events:
        fallback_events.add(event)
        _record_invalid_assumption(exceptions, parameter, sector, level)
    return to_number(raw, np.nan)


def _record_invalid_assumption(
    exceptions: List[Dict[str, Any]],
    parameter: str,
    sector: Any,
    level: str,
) -> None:
    """Log a missing/invalid CRE scenario assumption."""
    record_exception(
        exceptions,
        "ERROR",
        "CRE",
        "CRE_SCENARIO_ASSUMPTION_INVALID",
        "CRE scenario assumption was missing or invalid; applicable calculations were not produced.",
        stress_level=level,
        module="CRE",
        field=parameter,
        details=f"subsector={sector}",
    )


def _in_range(value: float, range_spec: Any) -> bool:
    """Validate a numeric input against optional scenario sanity bounds."""
    if not range_spec:
        return True
    lower, upper = range_spec
    lower = to_number(lower, -np.inf)
    upper = to_number(upper, np.inf)
    return lower <= value <= upper


def _mark_out(
    out: pd.DataFrame,
    idx: int,
    level: str,
    row: Mapping[str, Any],
    scenario: Mapping[str, Any],
    out_scope: List[Dict[str, Any]],
    test: str,
    fields: List[str],
    reason: str,
) -> None:
    """Mark a CRE borrower/level out of scope and add detail rows."""
    out.at[idx, f"out_of_scope_{level}"] = True
    record_out_of_scope(out_scope, row, scenario, "CRE", level, test, fields, reason)
