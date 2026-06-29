"""CRE stress module."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from stress_engine.stress.base import amortized_payment, base_pd_lgd, clamp, expected_loss


def apply_cre(row: pd.Series, scenario: Mapping[str, object], config: Mapping[str, object]) -> dict:
    sector = str(row.get("cre_sector") or row.get("sector") or "").lower()
    sector_scenario = scenario.get("cre", {}).get(sector, {})
    balance = float(row.get("balance", 0.0))
    base_pd, base_lgd = base_pd_lgd(row, "cre", config)
    noi = float(row.get("noi", 0.0))
    cap_rate = float(sector_scenario.get("base_cap_rate", 0.07)) + float(sector_scenario.get("cap_rate_shock", 0.0))
    noi_shock = float(sector_scenario.get("noi_shock", 0.0))
    stressed_noi = noi * (1 + noi_shock)
    stressed_value = stressed_noi / cap_rate if cap_rate > 0 else 0.0
    stressed_ltv = balance / stressed_value if stressed_value > 0 else 9.99

    if row.get("maturity_formula") == "near_term":
        stressed_dscr = stressed_noi / float(row.get("debt_service_cre") or row.get("debt_service") or 1.0)
        pd_multiplier = 1.0 + max(stressed_ltv - 0.75, 0.0) + 0.25
    else:
        stressed_rate = float(scenario.get("treasury_5y_rate", 0.0)) + float(sector_scenario.get("interest_spread", 0.0))
        amortization_years = float(sector_scenario.get("amortization_years", 25))
        stressed_debt_service = amortized_payment(balance, stressed_rate, amortization_years)
        stressed_dscr = stressed_noi / stressed_debt_service if stressed_debt_service > 0 else 9.99
        pd_multiplier = 1.0 + max(stressed_ltv - 0.80, 0.0) + max(1.20 - stressed_dscr, 0.0) * 0.20

    value_decline = max(1 - (stressed_value / float(row.get("appraised_value", stressed_value) or stressed_value)), 0.0)
    stressed_pd = clamp(base_pd * pd_multiplier, config.get("pd_floor", 0.0001), config.get("pd_cap", 1.0))
    stressed_lgd = clamp(base_lgd + value_decline * 0.50, config.get("lgd_floor", 0.0), config.get("lgd_cap", 1.0))
    return {
        "base_pd": base_pd,
        "base_lgd": base_lgd,
        "base_expected_loss": expected_loss(balance, base_pd, base_lgd),
        "stressed_noi": stressed_noi,
        "stressed_value": stressed_value,
        "stressed_ltv": stressed_ltv,
        "stressed_dscr": stressed_dscr,
        "stressed_pd": stressed_pd,
        "stressed_lgd": stressed_lgd,
        "stressed_expected_loss": expected_loss(balance, stressed_pd, stressed_lgd),
        "stress_flag": "cre_stressed",
    }
