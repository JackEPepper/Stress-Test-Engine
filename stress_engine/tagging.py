"""JSON-defined borrower tagging and tie-out checks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .io import LoadedTable
from .utils import as_list, compare_values, stable_name, to_number


def apply_tags(
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
    loaded: Mapping[str, LoadedTable],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = borrowers.copy()
    tag_rows: List[Dict[str, Any]] = []
    tag_defs = normalize_tag_defs(scenario.get("tags", {}))
    balance_field = scenario["borrower"]["balance_field"]

    for tag in tag_defs:
        name = tag["name"]
        tag_col = f"tag_{stable_name(name)}"
        include = tag.get("include", tag.get("conditions", []))
        exclude = tag.get("exclude", [])
        mask = evaluate_conditions(result, include)
        if exclude:
            mask &= ~evaluate_conditions(result, exclude)
        result[tag_col] = mask.fillna(False).astype(bool)
        result.loc[result[tag_col], "all_tags"] = (
            result.loc[result[tag_col], "all_tags"].fillna("")
            if "all_tags" in result.columns
            else ""
        )

        tagged = result[result[tag_col]]
        row = {
            "tag": name,
            "tag_column": tag_col,
            "model_eligible": bool(tag.get("model_eligible", True)),
            "internal_only": bool(tag.get("internal_only", False)),
            "borrower_count": int(len(tagged)),
            "balance_field": balance_field,
            "balance": float(pd.to_numeric(tagged.get(balance_field, pd.Series(dtype=float)), errors="coerce").sum()),
            "tie_out_name": np.nan,
            "expected": np.nan,
            "actual": np.nan,
            "difference": np.nan,
            "tolerance": np.nan,
            "passed": np.nan,
        }
        tag_rows.append(row)
        for tieout in as_list(tag.get("tie_out")):
            tag_rows.append(_evaluate_tieout(name, tag_col, result, tieout, loaded, balance_field))

    tag_summary = pd.DataFrame(tag_rows)
    result["all_tags"] = _tag_list(result, tag_defs, include_internal=True)
    result["model_tags"] = _tag_list(result, tag_defs, include_internal=False)
    return result, tag_summary


def normalize_tag_defs(tags: Any) -> List[Dict[str, Any]]:
    if isinstance(tags, Mapping):
        out = []
        for name, spec in tags.items():
            item = dict(spec)
            item.setdefault("name", name)
            out.append(item)
        return out
    if isinstance(tags, list):
        return [dict(item) for item in tags]
    raise ValueError("Scenario tags must be an object or a list.")


def evaluate_conditions(df: pd.DataFrame, conditions: Any) -> pd.Series:
    if not conditions:
        return pd.Series(True, index=df.index)
    if isinstance(conditions, Mapping):
        if "all" in conditions:
            mask = pd.Series(True, index=df.index)
            for item in conditions["all"]:
                mask &= evaluate_conditions(df, [item])
            return mask
        if "any" in conditions:
            mask = pd.Series(False, index=df.index)
            for item in conditions["any"]:
                mask |= evaluate_conditions(df, [item])
            return mask
        conditions = [conditions]

    mask = pd.Series(True, index=df.index)
    for condition in conditions:
        mask &= _evaluate_condition(df, condition)
    return mask.fillna(False)


def _evaluate_condition(df: pd.DataFrame, condition: Mapping[str, Any]) -> pd.Series:
    field = condition.get("field")
    op = str(condition.get("op", "eq")).lower()
    value = condition.get("value")
    if field not in df.columns:
        return pd.Series(False, index=df.index)
    series = df[field]

    if op in {"eq", "=="}:
        return series.apply(lambda item: compare_values(item, value))
    if op in {"ne", "!="}:
        return ~series.apply(lambda item: compare_values(item, value))
    if op == "in":
        values = set(as_list(value))
        return series.isin(values)
    if op in {"not_in", "notin"}:
        values = set(as_list(value))
        return ~series.isin(values)
    if op == "gt":
        return pd.to_numeric(series, errors="coerce") > to_number(value)
    if op == "gte":
        return pd.to_numeric(series, errors="coerce") >= to_number(value)
    if op == "lt":
        return pd.to_numeric(series, errors="coerce") < to_number(value)
    if op == "lte":
        return pd.to_numeric(series, errors="coerce") <= to_number(value)
    if op == "between":
        lower, upper = as_list(value)
        numeric = pd.to_numeric(series, errors="coerce")
        return (numeric >= to_number(lower)) & (numeric <= to_number(upper))
    if op == "contains":
        return series.astype(str).str.contains(str(value), case=bool(condition.get("case", False)), na=False)
    if op == "startswith":
        return series.astype(str).str.startswith(str(value), na=False)
    if op == "endswith":
        return series.astype(str).str.endswith(str(value), na=False)
    if op in {"is_null", "null"}:
        return series.isna()
    if op in {"not_null", "notnull"}:
        return series.notna()
    if op == "regex":
        return series.astype(str).str.contains(str(value), regex=True, na=False)
    raise ValueError(f"Unsupported tag condition operator: {op}")


def _evaluate_tieout(
    tag_name: str,
    tag_col: str,
    borrowers: pd.DataFrame,
    tieout: Mapping[str, Any],
    loaded: Mapping[str, LoadedTable],
    default_balance_field: str,
) -> Dict[str, Any]:
    source_name = tieout.get("source")
    actual_field = tieout.get("actual_field", tieout.get("balance_field", default_balance_field))
    tolerance = float(tieout.get("tolerance", 0.0))
    actual = float(pd.to_numeric(borrowers.loc[borrowers[tag_col], actual_field], errors="coerce").sum())
    expected = tieout.get("expected")
    if expected is None:
        if source_name not in loaded:
            raise ValueError(f"Tie-out for tag '{tag_name}' references unknown source '{source_name}'.")
        frame = loaded[source_name].frame
        amount_field = tieout.get("amount_field", "expected")
        if amount_field not in frame.columns:
            raise ValueError(f"Tie-out source '{source_name}' missing amount field '{amount_field}'.")
        filtered = frame
        for condition in as_list(tieout.get("lookup")):
            filtered = filtered[evaluate_conditions(filtered, [condition])]
        if "key_field" in tieout:
            key_field = tieout["key_field"]
            match_value = tieout.get("match_value", tag_name)
            filtered = filtered[filtered[key_field] == match_value]
        expected = float(pd.to_numeric(filtered[amount_field], errors="coerce").sum())
    expected = float(expected)
    difference = actual - expected
    return {
        "tag": tag_name,
        "tag_column": tag_col,
        "model_eligible": np.nan,
        "internal_only": np.nan,
        "borrower_count": int(borrowers[tag_col].sum()),
        "balance_field": actual_field,
        "balance": actual,
        "tie_out_name": tieout.get("name", tag_name),
        "expected": expected,
        "actual": actual,
        "difference": difference,
        "tolerance": tolerance,
        "passed": bool(abs(difference) <= tolerance),
    }


def _tag_list(df: pd.DataFrame, tag_defs: List[Mapping[str, Any]], include_internal: bool) -> pd.Series:
    values: List[str] = []
    for _, row in df.iterrows():
        active: List[str] = []
        for tag in tag_defs:
            if not include_internal and not tag.get("model_eligible", True):
                continue
            col = f"tag_{stable_name(tag['name'])}"
            if col in df.columns and bool(row.get(col, False)):
                active.append(str(tag["name"]))
        values.append(";".join(active))
    return pd.Series(values, index=df.index)


def model_eligible_tag_names(scenario: Mapping[str, Any]) -> set[str]:
    tags = normalize_tag_defs(scenario.get("tags", {}))
    return {str(tag["name"]) for tag in tags if tag.get("model_eligible", True)}
