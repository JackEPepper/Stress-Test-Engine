"""C&I stress module."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from stress_engine.stress.base import base_pd_lgd, clamp, expected_loss


def apply_ci(row: pd.Series, scenario: Mapping[str, object], config: Mapping[str, object]) -> dict:
    formula = str(row.get("selected_formula", "formula_1"))
    formula_scenario = scenario.get("ci", {}).get(formula, {})
    balance = float(row.get("balance", 0.0))
    base_pd, base_lgd = base_pd_lgd(row, "ci", config)
    ebitda = float(row.get("ebitda", 0.0))
    revenue = float(row.get("revenue", 0.0)) if pd.notna(row.get("revenue")) else 0.0
    total_debt = float(row.get("total_debt", 0.0)) if pd.notna(row.get("total_debt")) else 0.0
    debt_service = float(row.get("debt_service_ci", 0.0) or row.get("debt_service", 0.0))
    cash = float(row.get("cash", 0.0)) if pd.notna(row.get("cash")) else 0.0
    interest_expense = float(row.get("interest_expense", 0.0)) if pd.notna(row.get("interest_expense")) else 0.0

    ebitda_shock = float(formula_scenario.get("ebitda_shock", 0.0))
    interest_rate_shock = float(formula_scenario.get("interest_rate_shock", 0.0))
    stressed_ebitda = ebitda * (1 + ebitda_shock)

    if formula == "formula_2":
        revenue_shock = float(formula_scenario.get("revenue_shock", 0.0))
        stressed_revenue = revenue * (1 + revenue_shock)
        stressed_ebitda = stressed_ebitda + (stressed_revenue - revenue) * 0.10
    else:
        stressed_revenue = revenue

    if formula == "formula_3":
        liquidity_haircut = float(formula_scenario.get("liquidity_haircut", 0.0))
        stressed_cash = cash * (1 + liquidity_haircut)
    else:
        stressed_cash = cash

    maturity_add_on = 0.20 if row.get("maturity_formula") == "near_term" else 0.0
    stressed_interest_expense = interest_expense + total_debt * interest_rate_shock
    stressed_debt_service = debt_service + total_debt * interest_rate_shock
    stressed_debt_to_ebitda = total_debt / stressed_ebitda if stressed_ebitda > 0 else 9.99
    stressed_fixed_charge_coverage = stressed_ebitda / stressed_debt_service if stressed_debt_service > 0 else 9.99
    liquidity_gap = max((stressed_debt_service - stressed_cash) / balance, 0.0) if balance > 0 else 0.0
    pd_multiplier = 1.0 + max(stressed_debt_to_ebitda - 3.0, 0.0) * 0.08 + max(1.25 - stressed_fixed_charge_coverage, 0.0) * 0.25
    pd_multiplier += liquidity_gap * 0.50 + maturity_add_on
    stressed_pd = clamp(base_pd * pd_multiplier, config.get("pd_floor", 0.0001), config.get("pd_cap", 1.0))
    stressed_lgd = clamp(base_lgd + liquidity_gap * 0.10, config.get("lgd_floor", 0.0), config.get("lgd_cap", 1.0))

    return {
        "base_pd": base_pd,
        "base_lgd": base_lgd,
        "base_expected_loss": expected_loss(balance, base_pd, base_lgd),
        "stressed_ebitda": stressed_ebitda,
        "stressed_revenue": stressed_revenue,
        "stressed_cash": stressed_cash,
        "stressed_interest_expense": stressed_interest_expense,
        "stressed_debt_to_ebitda": stressed_debt_to_ebitda,
        "stressed_fixed_charge_coverage": stressed_fixed_charge_coverage,
        "stressed_pd": stressed_pd,
        "stressed_lgd": stressed_lgd,
        "stressed_expected_loss": expected_loss(balance, stressed_pd, stressed_lgd),
        "stress_flag": "ci_stressed",
    }
