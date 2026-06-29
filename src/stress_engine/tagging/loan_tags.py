"""Rules-based loan tagging."""

from __future__ import annotations

from typing import List

import pandas as pd


def add_tags(frame: pd.DataFrame, as_of_date: pd.Timestamp, near_term_days: int) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["tags"] = tagged.apply(lambda row: "|".join(_tags_for_row(row, as_of_date, near_term_days)), axis=1)
    return tagged


def tags_as_list(tags: object) -> List[str]:
    if not isinstance(tags, str) or not tags:
        return []
    return [tag for tag in tags.split("|") if tag]


def _tags_for_row(row: pd.Series, as_of_date: pd.Timestamp, near_term_days: int) -> List[str]:
    tags: List[str] = []
    portfolio = str(row.get("portfolio", "")).lower()
    product_type = str(row.get("product_type", "")).lower()
    sector = str(row.get("sector", "")).lower()

    if portfolio:
        tags.append(f"portfolio_{portfolio.replace('&', 'and').replace(' ', '_')}")
    if product_type:
        tags.append(f"product_{product_type.replace(' ', '_')}")
    if sector:
        tags.append(f"sector_{sector.replace(' ', '_')}")

    if pd.notna(row.get("cre_sector")) and str(row.get("cre_sector", "")).strip():
        tags.append("has_cre_collateral")
    if pd.notna(row.get("ebitda")):
        tags.append("has_ci_financials")
    if pd.notna(row.get("fico")):
        tags.append("has_fico_pd")

    if portfolio == "cre" or "has_cre_collateral" in tags:
        tags.append("eligible_cre")
    if portfolio in {"c&i", "ci"} or "has_ci_financials" in tags:
        tags.append("eligible_ci")
    if portfolio == "consumer" or "has_fico_pd" in tags:
        tags.append("eligible_consumer")

    maturity_date = row.get("maturity_date")
    if pd.notna(maturity_date):
        days_to_maturity = int((maturity_date - as_of_date).days)
        if days_to_maturity <= near_term_days:
            tags.append("near_term_maturity")
        else:
            tags.append("longer_term_maturity")

    if product_type == "revolver":
        tags.append("ci_formula_3")
    elif "construction" in sector:
        tags.append("ci_formula_2")
    else:
        tags.append("ci_formula_1")

    return sorted(set(tags))
