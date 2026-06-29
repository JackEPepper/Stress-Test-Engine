"""Dynamic loan-level validation."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def validate_dynamic_rules(frame: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    validated = frame.copy()
    validated["scope_status"] = "in_scope"
    validated["out_of_scope_reasons"] = ""

    for index, row in validated.iterrows():
        reasons = _row_reasons(row, config)
        if reasons:
            validated.at[index, "scope_status"] = "out_of_scope"
            validated.at[index, "out_of_scope_reasons"] = "|".join(reasons)

    return validated


def _row_reasons(row: pd.Series, config: Mapping[str, object]) -> list[str]:
    module = str(row.get("selected_stress_module", ""))
    formula = str(row.get("selected_formula", ""))
    maturity = str(row.get("maturity_formula", ""))
    if not module:
        return ["no_eligible_stress_module"]

    required = (
        config.get("dynamic_required_fields", {})
        .get(module, {})
        .get(formula, {})
        .get(maturity, [])
    )
    reasons: list[str] = []
    for field in required:
        if field not in row.index or pd.isna(row.get(field)) or row.get(field) == "":
            reasons.append(f"missing_required_field:{field}")

    for field in config.get("positive_fields", {}).get(module, []):
        value = row.get(field)
        if pd.isna(value) or value <= 0:
            reasons.append(f"non_positive_required_field:{field}")

    return reasons
