"""Shared stress-module helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from ..exceptions import record_exception
from ..tagging import model_eligible_tag_names
from ..utils import as_list, is_missing, risk_bucket_from_rating, stable_name, to_number


def initialize_results(
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Create the borrower-level stress result frame.

    Called by `StressEngine.run` before any module executes. It derives the
    in-place commercial risk bucket from risk rating and pre-populates each
    stress level with the base bucket.
    """
    exceptions = exceptions if exceptions is not None else []
    config = scenario.get("borrower", {})
    risk_rating_field = config.get("risk_rating_field", "risk_rating")
    levels = [str(level) for level in scenario.get("stress_levels", ["S1", "S2"])]
    result = borrowers.copy()
    if risk_rating_field in result.columns:
        result["base_bucket"] = result[risk_rating_field].apply(risk_bucket_from_rating)
    else:
        result["base_bucket"] = "Unknown"
    borrower_id = config.get("borrower_id_field", "borrower_id")
    module_field = "primary_module" if "primary_module" in result.columns else config.get("module_field", "model_module")
    affected_modules = {"CRE", "C&I", "Overlay"}
    for _, row in result[result["base_bucket"] == "Unknown"].iterrows():
        module = str(row.get(module_field, ""))
        if module not in affected_modules:
            continue
        record_exception(
            exceptions,
            "WARNING",
            "risk_rating",
            "RISK_RATING_MISSING",
            "Commercial or overlay exposure had no usable in-place risk rating; its balance remains in the Unknown bucket.",
            borrower_id=row.get(borrower_id),
            portfolio=row.get(config.get("portfolio_field", "model_portfolio")),
            module=row.get(module_field),
            field=risk_rating_field,
        )
    result["module_applied"] = ""
    for level in levels:
        result[f"stressed_bucket_{level}"] = result["base_bucket"]
        result[f"out_of_scope_{level}"] = False
    return result


def module_population(df: pd.DataFrame, scenario: Mapping[str, Any], module_config: Mapping[str, Any]) -> pd.Series:
    """Return rows eligible for one stress module.

    Called inside each module. It checks model-eligible tags and the resolved
    ``primary_module`` field.
    """
    mask = pd.Series(True, index=df.index)
    if "model_excluded" in df.columns:
        mask &= ~df["model_excluded"].fillna(False).astype(bool)
    eligible_tags = as_list(module_config.get("eligible_tags"))
    if eligible_tags:
        allowed = model_eligible_tag_names(scenario)
        tag_mask = pd.Series(False, index=df.index)
        for tag in eligible_tags:
            if tag not in allowed:
                continue
            column = f"tag_{stable_name(tag)}"
            if column in df.columns:
                tag_mask |= df[column].fillna(False).astype(bool)
        mask &= tag_mask
    module_name = module_config.get("_module_name")
    if module_name and "primary_module" in df.columns:
        mask &= df["primary_module"].astype(str) == str(module_name)
    if scenario.get("_targeted_mode") and "_targeted_active" in df.columns:
        mask &= df["_targeted_active"].fillna(False).astype(bool)
    return mask.fillna(False)


def targeted_override_column(module: str, parameter: str, level: str) -> str:
    """Return the internal loan-level effective-assumption column name."""
    return f"_targeted_{stable_name(module)}_{stable_name(parameter)}_{stable_name(level)}"


def targeted_parameter(
    row: Mapping[str, Any],
    scenario: Mapping[str, Any],
    module: str,
    parameter: str,
    level: str,
    baseline: Any,
) -> float:
    """Use a resolved targeted assumption when present, otherwise the baseline."""
    if not scenario.get("_targeted_mode"):
        return to_number(baseline)
    column = targeted_override_column(module, parameter, level)
    value = row.get(column)
    return to_number(baseline) if is_missing(value) else to_number(value)


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
    """Append loan-level out-of-scope detail rows for final reporting."""
    borrower_config = scenario.get("borrower", {})
    borrower_id = borrower_config.get("borrower_id_field", "borrower_id")
    portfolio_field = borrower_config.get("portfolio_field", "portfolio")
    for field in fields:
        detail = {
                "borrower_id": borrower.get(borrower_id, np.nan),
                "portfolio": borrower.get(portfolio_field, np.nan),
                "module": module_name,
                "stress_level": level,
                "test": test,
                "field": field,
                "reason": reason,
            }
        if scenario.get("_targeted_mode"):
            loan_id = borrower_config.get("loan_id_field", "loan_id")
            detail["loan_id"] = borrower.get(loan_id, np.nan)
            detail["scenario_variant"] = scenario.get("_scenario_variant", "")
        rows.append(detail)


def missing_fields(row: Mapping[str, Any], fields: Iterable[str]) -> List[str]:
    """Return applicable fields that are missing from a row."""
    missing: List[str] = []
    for field in fields:
        if field not in row or pd.isna(row[field]):
            missing.append(field)
    return missing


def append_module(existing: Any, module: str) -> str:
    """Append a module name to the borrower-level audit field once."""
    if not existing:
        return module
    pieces = [item for item in str(existing).split(";") if item]
    if module not in pieces:
        pieces.append(module)
    return ";".join(pieces)
