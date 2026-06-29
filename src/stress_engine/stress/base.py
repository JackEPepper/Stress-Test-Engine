"""Shared stress calculation helpers."""

from __future__ import annotations

import math

import pandas as pd


def clamp(value: float, floor: float, cap: float) -> float:
    if pd.isna(value):
        return value
    return max(floor, min(cap, float(value)))


def expected_loss_from_rate(balance: float, el_rate: float) -> float:
    if pd.isna(balance) or pd.isna(el_rate):
        return 0.0
    return float(balance) * float(el_rate)


def amortized_payment(principal: float, annual_rate: float, years: float) -> float:
    if principal <= 0 or years <= 0:
        return 0.0
    periods = years * 12
    monthly_rate = annual_rate / 12
    if math.isclose(monthly_rate, 0.0):
        return principal / years
    monthly_payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-periods))
    return monthly_payment * 12
