"""Consumer stress module."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from stress_engine.stress.base import base_pd_lgd, clamp, expected_loss


def apply_consumer(
    row: pd.Series, scenario: Mapping[str, object], config: Mapping[str, object], fico_pd_table: pd.DataFrame
) -> dict:
    balance = float(row.get("balance", 0.0))
    _, base_lgd = base_pd_lgd(row, "consumer", config)
    base_pd = _pd_from_fico(row.get("fico"), fico_pd_table)
    if pd.isna(base_pd):
        base_pd = config.get("default_pd", {}).get("consumer", 0.04)

    consumer_scenario = scenario.get("consumer", {})
    pd_multiplier = float(consumer_scenario.get("pd_multiplier", 1.0))
    value_loss_shock = abs(float(consumer_scenario.get("value_loss_shock", 0.0)))
    maturity_add_on = 0.10 if row.get("maturity_formula") == "near_term" else 0.0
    stressed_pd = clamp(base_pd * (pd_multiplier + maturity_add_on), config.get("pd_floor", 0.0001), config.get("pd_cap", 1.0))
    stressed_lgd = clamp(base_lgd + value_loss_shock, config.get("lgd_floor", 0.0), config.get("lgd_cap", 1.0))
    return {
        "base_pd": base_pd,
        "base_lgd": base_lgd,
        "base_expected_loss": expected_loss(balance, base_pd, base_lgd),
        "base_pd_from_fico": base_pd,
        "stressed_pd": stressed_pd,
        "stressed_lgd": stressed_lgd,
        "stressed_expected_loss": expected_loss(balance, stressed_pd, stressed_lgd),
        "stress_flag": "consumer_stressed",
    }


def _pd_from_fico(fico: object, fico_pd_table: pd.DataFrame) -> float:
    if pd.isna(fico):
        return float("nan")
    matches = fico_pd_table[(fico_pd_table["fico_min"] <= fico) & (fico_pd_table["fico_max"] >= fico)]
    if matches.empty:
        return float("nan")
    return float(matches.iloc[0]["base_pd"])
