"""Commercial Real Estate stress module."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .base import missing_fields, module_population, record_out_of_scope
from ..utils import (
    annual_debt_payment,
    get_levels,
    higher_metric_bucket,
    is_missing,
    lookup_parameter,
    lower_metric_bucket,
    to_number,
    worse_bucket,
)


def run_cre(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = scenario.get("modules", {}).get("CRE", scenario.get("modules", {}).get("cre", {}))
    if not config or not config.get("enabled", True):
        return results, pd.DataFrame()

    out = results.copy()
    levels = get_levels(scenario)
    mask = module_population(out, scenario, config)
    borrower_cfg = scenario["borrower"]
    balance_field = config.get("balance_field", borrower_cfg["balance_field"])
    maturity_field = config.get("maturity_date_field", borrower_cfg.get("maturity_date_field", "maturity_date"))
    subsector_field = config.get("subsector_field", "cre_subsector")
    cutoff_date = pd.to_datetime(scenario.get("run", {}).get("cutoff_date", scenario.get("cutoff_date")))
    threshold_months = int(config.get("maturity_months", config.get("maturity_threshold_months", 12)))
    threshold_date = cutoff_date + pd.DateOffset(months=threshold_months)
    out_scope: List[Dict[str, Any]] = []

    for level in levels:
        for idx, row in out.loc[mask].iterrows():
            out.at[idx, "module_applied"] = _append_module(out.at[idx, "module_applied"], "CRE")
            base_bucket = row.get("base_bucket", "Unknown")
            if base_bucket == "Substandard":
                out.at[idx, f"stressed_bucket_{level}"] = "Substandard"
                continue
            maturity = pd.to_datetime(row.get(maturity_field), errors="coerce")
            if pd.isna(maturity):
                _mark_out(out, idx, level, row, scenario, out_scope, "maturity_split", [maturity_field], "missing_maturity")
                continue
            if maturity > threshold_date:
                stressed_bucket = _run_dscr(out, idx, row, scenario, config, level, subsector_field, out_scope)
            else:
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
                )
            if stressed_bucket:
                out.at[idx, f"stressed_bucket_{level}"] = worse_bucket(base_bucket, stressed_bucket)
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
) -> str | None:
    dscr_cfg = config.get("tests", {}).get("dscr", config.get("dscr", {}))
    if not dscr_cfg.get("enabled", True):
        return row.get("base_bucket", "Pass")
    dscr_field = dscr_cfg.get("field", "dscr")
    required = [dscr_field, subsector_field]
    missing = missing_fields(row, required)
    if missing:
        _mark_out(out, idx, level, row, scenario, out_scope, "DSCR", missing, "missing_required_field")
        return None
    dscr = to_number(row.get(dscr_field))
    if not _in_range(dscr, dscr_cfg.get("acceptable_range", config.get("dscr_acceptable_range"))):
        _mark_out(out, idx, level, row, scenario, out_scope, "DSCR", [dscr_field], "outside_acceptable_range")
        return None
    decline = to_number(lookup_parameter(dscr_cfg.get("decline", config.get("dscr_decline")), row.get(subsector_field), level), 0.0)
    stressed_dscr = dscr * (1 - decline)
    out.at[idx, f"cre_dscr_{level}"] = stressed_dscr
    return lower_metric_bucket(stressed_dscr, dscr_cfg.get("cutoffs", config.get("dscr_cutoffs", {})))


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
) -> str | None:
    tests = config.get("tests", {})
    refi_cfg = tests.get("refinance", config.get("refinance", {}))
    ltv_cfg = tests.get("ltv", config.get("ltv", {}))
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
    if not _in_range(ratio, config.get("noi_balance_ratio_acceptable_range", ltv_cfg.get("noi_balance_ratio_acceptable_range"))):
        _mark_out(out, idx, level, row, scenario, out_scope, "Refinance/LTV", [noi_field, balance_field], "outside_noi_balance_ratio_range")
        return None

    best = row.get("base_bucket", "Pass")
    if refi_cfg.get("enabled", True):
        dscr_cfg = tests.get("dscr", config.get("dscr", {}))
        decline_table = refi_cfg.get("noi_decline", dscr_cfg.get("decline", config.get("dscr_decline")))
        noi_decline = to_number(lookup_parameter(decline_table, row.get(subsector_field), level), 0.0)
        spread = to_number(lookup_parameter(refi_cfg.get("credit_spreads"), row.get(subsector_field), level), 0.0)
        treasury = to_number(refi_cfg.get("treasury_rate", config.get("treasury_rate", 0.0)), 0.0)
        amort_years = to_number(lookup_parameter(refi_cfg.get("amortization_years"), row.get(subsector_field), level), np.nan)
        payment = annual_debt_payment(balance, treasury + spread, amort_years)
        if is_missing(payment) or payment <= 0:
            _mark_out(out, idx, level, row, scenario, out_scope, "Refinance", ["amortization_years"], "invalid_debt_payment")
            return None
        stressed_noi = noi * (1 - noi_decline)
        stressed_dscr = stressed_noi / payment
        out.at[idx, f"cre_refi_dscr_{level}"] = stressed_dscr
        out.at[idx, f"cre_dscr_{level}"] = stressed_dscr
        best = worse_bucket(best, lower_metric_bucket(stressed_dscr, refi_cfg.get("cutoffs", config.get("dscr_cutoffs", {}))))

    if ltv_cfg.get("enabled", True):
        cap_rate = to_number(lookup_parameter(ltv_cfg.get("cap_rates", config.get("ltv_cap_rates")), row.get(subsector_field), level), np.nan)
        if is_missing(cap_rate) or cap_rate <= 0 or noi <= 0:
            _mark_out(out, idx, level, row, scenario, out_scope, "LTV", ["cap_rates", noi_field], "invalid_ltv_inputs")
            return None
        stressed_ltv = balance * cap_rate / noi
        out.at[idx, f"cre_ltv_{level}"] = stressed_ltv
        best = worse_bucket(best, higher_metric_bucket(stressed_ltv, ltv_cfg.get("cutoffs", config.get("ltv_cutoffs", {}))))
    return best


def _in_range(value: float, range_spec: Any) -> bool:
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
    out.at[idx, f"out_of_scope_{level}"] = True
    record_out_of_scope(out_scope, row, scenario, "CRE", level, test, fields, reason)


def _append_module(existing: Any, module: str) -> str:
    if not existing:
        return module
    pieces = [item for item in str(existing).split(";") if item]
    if module not in pieces:
        pieces.append(module)
    return ";".join(pieces)
