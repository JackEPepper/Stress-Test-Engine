"""Portfolio-level migration overlays for portfolios without financials."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from ..utils import BUCKET_ORDER, as_list, get_levels, pct, to_number


BUCKETS = ["Pass", "Special Mention", "Substandard"]


def apply_overlays(
    bucket_summary: pd.DataFrame,
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlays = scenario.get("overlays", {})
    if not overlays:
        return bucket_summary, pd.DataFrame()
    if isinstance(overlays, list):
        overlay_items = {item["portfolio"]: item for item in overlays}
    else:
        overlay_items = overlays
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    levels = get_levels(scenario)
    summary = bucket_summary.copy()
    overlay_rows: List[Dict[str, Any]] = []

    for portfolio, config in overlay_items.items():
        if not config or not config.get("enabled", True):
            continue
        source_portfolios = as_list(config.get("source_portfolios", config.get("basis_portfolios")))
        if not source_portfolios:
            continue
        base_rows = borrowers[borrowers[portfolio_field] == portfolio]
        total_balance = float(pd.to_numeric(base_rows[balance_field], errors="coerce").sum())
        if total_balance <= 0:
            continue
        base_ratios = _portfolio_base_ratios(base_rows, balance_field)
        source_base = _source_ratios(summary, source_portfolios, "Base")
        replacement: List[Dict[str, Any]] = []
        for level in levels:
            source_level = _source_ratios(summary, source_portfolios, level)
            sm_ratio = _grown_ratio(base_ratios["Special Mention"], source_base["Special Mention"], source_level["Special Mention"], config)
            sub_ratio = _grown_ratio(base_ratios["Substandard"], source_base["Substandard"], source_level["Substandard"], config)
            sm_ratio, sub_ratio = _cap_ratios(sm_ratio, sub_ratio)
            ratios = {
                "Pass": max(1 - sm_ratio - sub_ratio, 0.0),
                "Special Mention": sm_ratio,
                "Substandard": sub_ratio,
            }
            for bucket in BUCKETS:
                replacement.append(
                    {
                        "portfolio": portfolio,
                        "stress_level": level,
                        "bucket": bucket,
                        "balance": total_balance * ratios[bucket],
                        "borrower_count": np.nan,
                        "source": "overlay",
                    }
                )
            overlay_rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": level,
                    "source_portfolios": ";".join(map(str, source_portfolios)),
                    "base_special_mention_ratio": base_ratios["Special Mention"],
                    "base_substandard_ratio": base_ratios["Substandard"],
                    "stressed_special_mention_ratio": sm_ratio,
                    "stressed_substandard_ratio": sub_ratio,
                }
            )

        summary = summary[summary["portfolio"] != portfolio]
        base_replacement = []
        for bucket in BUCKETS:
            base_replacement.append(
                {
                    "portfolio": portfolio,
                    "stress_level": "Base",
                    "bucket": bucket,
                    "balance": total_balance * base_ratios[bucket],
                    "borrower_count": int((base_rows["base_bucket"] == bucket).sum()) if "base_bucket" in base_rows.columns else np.nan,
                    "source": "overlay_base",
                }
            )
        summary = pd.concat([summary, pd.DataFrame(base_replacement + replacement)], ignore_index=True)
    return summary, pd.DataFrame(overlay_rows)


def _portfolio_base_ratios(rows: pd.DataFrame, balance_field: str) -> Dict[str, float]:
    total = float(pd.to_numeric(rows[balance_field], errors="coerce").sum())
    ratios = {bucket: 0.0 for bucket in BUCKETS}
    if total <= 0 or "base_bucket" not in rows.columns:
        ratios["Pass"] = 1.0
        return ratios
    for bucket in BUCKETS:
        ratios[bucket] = pct(pd.to_numeric(rows.loc[rows["base_bucket"] == bucket, balance_field], errors="coerce").sum(), total)
    return ratios


def _source_ratios(summary: pd.DataFrame, portfolios: List[Any], level: str) -> Dict[str, float]:
    rows = summary[(summary["portfolio"].isin(portfolios)) & (summary["stress_level"] == level)]
    total = float(pd.to_numeric(rows["balance"], errors="coerce").sum())
    return {
        bucket: pct(pd.to_numeric(rows.loc[rows["bucket"] == bucket, "balance"], errors="coerce").sum(), total)
        for bucket in BUCKETS
    }


def _grown_ratio(base_ratio: float, source_base_ratio: float, source_level_ratio: float, config: Mapping[str, Any]) -> float:
    if source_base_ratio and not pd.isna(source_base_ratio):
        return to_number(base_ratio, 0.0) * (source_level_ratio / source_base_ratio)
    if config.get("zero_base_behavior", "absolute_delta") == "absolute_delta":
        return to_number(base_ratio, 0.0) + max(to_number(source_level_ratio, 0.0) - to_number(source_base_ratio, 0.0), 0.0)
    return to_number(base_ratio, 0.0)


def _cap_ratios(sm_ratio: float, sub_ratio: float) -> Tuple[float, float]:
    sm_ratio = min(max(to_number(sm_ratio, 0.0), 0.0), 1.0)
    sub_ratio = min(max(to_number(sub_ratio, 0.0), 0.0), 1.0)
    if sm_ratio + sub_ratio <= 1:
        return sm_ratio, sub_ratio
    total = sm_ratio + sub_ratio
    return sm_ratio / total, sub_ratio / total
