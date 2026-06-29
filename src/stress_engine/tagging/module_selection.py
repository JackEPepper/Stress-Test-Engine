"""Primary module and formula selection."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from stress_engine.tagging.loan_tags import tags_as_list


def select_modules_and_formulas(frame: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    selected = frame.copy()
    priority = list(config.get("primary_module_priority", ["cre", "ci", "consumer"]))
    selected["selected_stress_module"] = selected["tags"].apply(lambda tags: _select_module(tags, priority))
    selected["selected_formula"] = selected.apply(lambda row: _select_formula(row, config), axis=1)
    selected["maturity_formula"] = selected["tags"].apply(
        lambda tags: "near_term" if "near_term_maturity" in tags_as_list(tags) else "longer_term"
    )
    return selected


def _select_module(tags: object, priority: list[str]) -> str:
    tag_list = tags_as_list(tags)
    for module in priority:
        if f"eligible_{module}" in tag_list:
            return module
    return ""


def _select_formula(row: pd.Series, config: Mapping[str, object]) -> str:
    module = row.get("selected_stress_module", "")
    if module in {"cre", "consumer"}:
        return "standard"
    if module != "ci":
        return ""

    sector = str(row.get("sector", "")).lower()
    product_type = str(row.get("product_type", "")).lower()
    for rule in config.get("ci_formula_rules", []):
        formula = rule.get("formula", "")
        if sector in set(rule.get("sectors", [])):
            return formula
        if product_type in set(rule.get("product_types", [])):
            return formula
    return "formula_1"
