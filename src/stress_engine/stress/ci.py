"""C&I stress module."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def apply_ci(row: pd.Series, scenario: Mapping[str, object], config: Mapping[str, object]) -> dict:
    formula = str(row.get("selected_formula", "formula_1"))
    formula_scenario = scenario.get("ci", {}).get(formula, {})
    balance = float(row.get("balance", 0.0))
    ebitda = float(row.get("ebitda", 0.0))
    revenue = float(row.get("revenue", 0.0)) if pd.notna(row.get("revenue")) else 0.0
    total_debt = float(row.get("total_debt", 0.0)) if pd.notna(row.get("total_debt")) else 0.0
    debt_service = float(row.get("debt_service_ci", 0.0) or row.get("debt_service", 0.0))
    cash = float(row.get("cash", 0.0)) if pd.notna(row.get("cash")) else 0.0
    interest_expense = float(row.get("interest_expense", 0.0)) if pd.notna(row.get("interest_expense")) else 0.0
    base_fixed_charge_coverage = ebitda / debt_service if debt_service > 0 else float("nan")

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

    stressed_interest_expense = interest_expense + total_debt * interest_rate_shock
    stressed_debt_service = debt_service + total_debt * interest_rate_shock
    stressed_debt_to_ebitda = total_debt / stressed_ebitda if stressed_ebitda > 0 else 9.99
    stressed_fixed_charge_coverage = stressed_ebitda / stressed_debt_service if stressed_debt_service > 0 else 9.99
    liquidity_gap = max((stressed_debt_service - stressed_cash) / balance, 0.0) if balance > 0 else 0.0

    return {
        "base_fixed_charge_coverage": base_fixed_charge_coverage,
        "stressed_ebitda": stressed_ebitda,
        "stressed_revenue": stressed_revenue,
        "stressed_cash": stressed_cash,
        "stressed_interest_expense": stressed_interest_expense,
        "stressed_debt_service": stressed_debt_service,
        "stressed_debt_to_ebitda": stressed_debt_to_ebitda,
        "stressed_fixed_charge_coverage": stressed_fixed_charge_coverage,
        "fixed_charge_coverage_change": (
            stressed_fixed_charge_coverage - base_fixed_charge_coverage
            if pd.notna(stressed_fixed_charge_coverage) and pd.notna(base_fixed_charge_coverage)
            else float("nan")
        ),
        "liquidity_gap": liquidity_gap,
        "stress_flag": "ci_stressed",
    }
