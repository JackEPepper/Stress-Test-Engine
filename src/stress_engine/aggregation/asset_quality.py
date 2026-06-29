"""Asset quality aggregations."""

from __future__ import annotations

import pandas as pd


def summarize(frame: pd.DataFrame, group_fields: list[str]) -> pd.DataFrame:
    data = frame[frame["scope_status"] == "in_scope"].copy()
    if data.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in data.groupby(group_fields, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {field: value for field, value in zip(group_fields, keys)}
        row.update(_summary_row(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_row(group: pd.DataFrame) -> pd.Series:
    balance = group["balance"].sum()
    return pd.Series(
        {
            "exposure_count": len(group),
            "total_balance": balance,
            "weighted_average_base_pd": _weighted_average(group, "base_pd", balance),
            "weighted_average_stressed_pd": _weighted_average(group, "stressed_pd", balance),
            "weighted_average_base_lgd": _weighted_average(group, "base_lgd", balance),
            "weighted_average_stressed_lgd": _weighted_average(group, "stressed_lgd", balance),
            "base_expected_loss": group["base_expected_loss"].sum(),
            "stressed_expected_loss": group["stressed_expected_loss"].sum(),
            "expected_loss_change": group["expected_loss_change"].sum(),
            "expected_loss_change_pct_balance": group["expected_loss_change"].sum() / balance if balance else 0.0,
        }
    )


def _weighted_average(group: pd.DataFrame, column: str, balance: float) -> float:
    if balance == 0 or column not in group:
        return 0.0
    return float((group[column].fillna(0) * group["balance"]).sum() / balance)


def out_of_scope_summary(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = frame[frame["scope_status"] == "out_of_scope"].copy()
    rows = []
    for _, row in excluded.iterrows():
        reasons = str(row.get("out_of_scope_reasons", "")).split("|")
        for reason in reasons:
            if reason:
                rows.append({"reason": reason, "balance": row.get("balance", 0.0), "loan_id": row.get("loan_id")})
    if not rows:
        return pd.DataFrame(columns=["reason", "loan_count", "total_balance"])
    return pd.DataFrame(rows).groupby("reason").agg(loan_count=("loan_id", "count"), total_balance=("balance", "sum")).reset_index()
