"""Consumer stress module."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from stress_engine.stress.base import clamp, expected_loss_from_rate


def apply_consumer(row: pd.Series, scenario: Mapping[str, object], config: Mapping[str, object], fico_pd_table: pd.DataFrame) -> dict:
    balance = float(row.get("balance", 0.0))
    base_el_rate = row.get("current_el_rate")
    base_pd = _pd_from_fico(row.get("fico"), fico_pd_table)
    if pd.isna(base_el_rate):
        if pd.notna(base_pd):
            base_el_rate = base_pd * float(config.get("default_consumer_lgd", 0.55))
    if pd.isna(base_el_rate):
        base_el_rate = config.get("default_consumer_el_rate", 0.025)

    consumer_scenario = scenario.get("consumer", {})
    el_multiplier = float(consumer_scenario.get("el_multiplier", 1.0))
    value_loss_shock = abs(float(consumer_scenario.get("value_loss_shock", 0.0)))
    maturity_add_on = 0.10 if row.get("maturity_formula") == "near_term" else 0.0
    stressed_el_rate = clamp(
        base_el_rate * (el_multiplier + maturity_add_on) + value_loss_shock * 0.10,
        config.get("el_rate_floor", 0.0),
        config.get("el_rate_cap", 1.0),
    )
    return {
        "base_pd_from_fico": base_pd,
        "base_el_rate": base_el_rate,
        "base_expected_loss": expected_loss_from_rate(balance, base_el_rate),
        "stressed_el_rate": stressed_el_rate,
        "stressed_expected_loss": expected_loss_from_rate(balance, stressed_el_rate),
        "expected_loss_change": expected_loss_from_rate(balance, stressed_el_rate)
        - expected_loss_from_rate(balance, base_el_rate),
        "stress_flag": "consumer_stressed",
    }


def _pd_from_fico(fico: object, fico_pd_table: pd.DataFrame) -> float:
    if pd.isna(fico):
        return float("nan")
    matches = fico_pd_table[(fico_pd_table["fico_min"] <= fico) & (fico_pd_table["fico_max"] >= fico)]
    if matches.empty:
        return float("nan")
    return float(matches.iloc[0]["base_pd"])
