"""Shared stress calculation helpers."""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd


def clamp(value: float, floor: float, cap: float) -> float:
    if pd.isna(value):
        return value
    return max(floor, min(cap, float(value)))


def expected_loss(balance: float, pd_value: float, lgd_value: float) -> float:
    if pd.isna(balance) or pd.isna(pd_value) or pd.isna(lgd_value):
        return 0.0
    return float(balance) * float(pd_value) * float(lgd_value)


def amortized_payment(principal: float, annual_rate: float, years: float) -> float:
    if principal <= 0 or years <= 0:
        return 0.0
    periods = years * 12
    monthly_rate = annual_rate / 12
    if math.isclose(monthly_rate, 0.0):
        return principal / years
    monthly_payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-periods))
    return monthly_payment * 12


def base_pd_lgd(row: pd.Series, module: str, config: Mapping[str, object]) -> tuple[float, float]:
    current_pd = row.get("current_pd")
    current_lgd = row.get("current_lgd")
    pd_value = current_pd if pd.notna(current_pd) else config.get("default_pd", {}).get(module, 0.03)
    lgd_value = current_lgd if pd.notna(current_lgd) else config.get("default_lgd", {}).get(module, 0.45)
    return (
        clamp(float(pd_value), config.get("pd_floor", 0.0001), config.get("pd_cap", 1.0)),
        clamp(float(lgd_value), config.get("lgd_floor", 0.0), config.get("lgd_cap", 1.0)),
    )
