"""Shared stress-module helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from ..tagging import model_eligible_tag_names
from ..utils import as_list, risk_bucket_from_rating, stable_name


def initialize_results(borrowers: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    config = scenario.get("borrower", {})
    risk_rating_field = config.get("risk_rating_field", "risk_rating")
    levels = [str(level) for level in scenario.get("stress_levels", ["S1", "S2"])]
    result = borrowers.copy()
    if risk_rating_field in result.columns:
        result["base_bucket"] = result[risk_rating_field].apply(risk_bucket_from_rating)
    else:
        result["base_bucket"] = "Unknown"
    result["module_applied"] = ""
    for level in levels:
        result[f"stressed_bucket_{level}"] = result["base_bucket"]
        result[f"out_of_scope_{level}"] = False
    return result


def module_population(df: pd.DataFrame, scenario: Mapping[str, Any], module_config: Mapping[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    eligible_tags = as_list(module_config.get("eligible_tags"))
    if eligible_tags:
        allowed = model_eligible_tag_names(scenario)
        tag_mask = pd.Series(False, index=df.index)
        for tag in eligible_tags:
            if tag not in allowed and not module_config.get("allow_non_model_tags", False):
                continue
            column = f"tag_{stable_name(tag)}"
            if column in df.columns:
                tag_mask |= df[column].fillna(False).astype(bool)
        mask &= tag_mask

    portfolio_field = module_config.get("portfolio_field", scenario.get("borrower", {}).get("portfolio_field"))
    portfolio_values = as_list(module_config.get("portfolio_values"))
    if portfolio_field and portfolio_values and portfolio_field in df.columns:
        mask &= df[portfolio_field].isin(portfolio_values)

    module_field = module_config.get("module_field", scenario.get("borrower", {}).get("module_field"))
    module_values = as_list(module_config.get("module_values"))
    if module_field and module_values and module_field in df.columns:
        mask &= df[module_field].isin(module_values)
    return mask.fillna(False)


def record_out_of_scope(
    rows: List[Dict[str, Any]],
    borrower: Mapping[str, Any],
    scenario: Mapping[str, Any],
    module_name: str,
    level: str,
    test: str,
    fields: Iterable[str],
    reason: str,
) -> None:
    borrower_config = scenario.get("borrower", {})
    borrower_id = borrower_config.get("borrower_id_field", "borrower_id")
    portfolio_field = borrower_config.get("portfolio_field", "portfolio")
    for field in fields:
        rows.append(
            {
                "borrower_id": borrower.get(borrower_id, np.nan),
                "portfolio": borrower.get(portfolio_field, np.nan),
                "module": module_name,
                "stress_level": level,
                "test": test,
                "field": field,
                "reason": reason,
            }
        )


def missing_fields(row: Mapping[str, Any], fields: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for field in fields:
        if field not in row or pd.isna(row[field]):
            missing.append(field)
    return missing
