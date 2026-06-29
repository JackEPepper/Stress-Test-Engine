"""CRE stress module."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from stress_engine.stress.base import amortized_payment


def apply_cre(row: pd.Series, scenario: Mapping[str, object], config: Mapping[str, object]) -> dict:
    sector = str(row.get("cre_sector") or row.get("sector") or "").lower()
    sector_scenario = scenario.get("cre", {}).get(sector, {})
    balance = float(row.get("balance", 0.0))
    noi = float(row.get("noi", 0.0))
    base_debt_service = float(row.get("debt_service_cre") or row.get("debt_service") or 0.0)
    base_dscr = noi / base_debt_service if base_debt_service > 0 else float("nan")
    cap_rate = float(sector_scenario.get("base_cap_rate", 0.07)) + float(sector_scenario.get("cap_rate_shock", 0.0))
    noi_shock = float(sector_scenario.get("noi_shock", 0.0))
    stressed_noi = noi * (1 + noi_shock)
    stressed_value = stressed_noi / cap_rate if cap_rate > 0 else 0.0
    stressed_ltv = balance / stressed_value if stressed_value > 0 else 9.99

    if row.get("maturity_formula") == "near_term":
        stressed_debt_service = base_debt_service
        stressed_dscr = stressed_noi / stressed_debt_service if stressed_debt_service > 0 else float("nan")
    else:
        stressed_rate = float(scenario.get("treasury_5y_rate", 0.0)) + float(sector_scenario.get("interest_spread", 0.0))
        amortization_years = float(sector_scenario.get("amortization_years", 25))
        stressed_debt_service = amortized_payment(balance, stressed_rate, amortization_years)
        stressed_dscr = stressed_noi / stressed_debt_service if stressed_debt_service > 0 else float("nan")

    return {
        "base_dscr": base_dscr,
        "stressed_noi": stressed_noi,
        "stressed_debt_service": stressed_debt_service,
        "stressed_value": stressed_value,
        "stressed_ltv": stressed_ltv,
        "stressed_dscr": stressed_dscr,
        "dscr_change": stressed_dscr - base_dscr if pd.notna(stressed_dscr) and pd.notna(base_dscr) else float("nan"),
        "stress_flag": "cre_stressed",
    }
