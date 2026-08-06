"""JSON-defined borrower tagging and tie-out checks."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .exceptions import record_exception
from .io import LoadedTable
from .utils import (
    as_list,
    compare_values,
    condition_fields,
    is_missing,
    risk_bucket_from_rating,
    stable_name,
    to_number,
)


CECL_LEVEL_TAG_FIELD = "cecl_level_tag"
CECL_LEVEL_MODULES = {"CRE", "C&I", "Overlay"}


TAG_CONDITION_OPERATORS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
    "contains",
    "has_token",
    "has_any_token",
    "has_all_tokens",
    "startswith",
    "endswith",
    "is_null",
    "not_null",
    "regex",
}


def apply_tags(
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
    loaded: Mapping[str, LoadedTable],
    exceptions: List[Dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate scenario tags, assignments, and tie-outs against borrowers.

    Called by `StressEngine.run` after borrower aggregation. Tag columns drive
    model eligibility, reconciliation-only populations, and derived fields such
    as `cre_subsector`, `ci_sector`, and `model_portfolio`.
    """
    exceptions = exceptions if exceptions is not None else []
    result = borrowers.copy()
    tag_rows: List[Dict[str, Any]] = []
    tag_defs = normalize_tag_defs(scenario.get("tags", {}))
    balance_field = scenario["borrower"]["balance_field"]
    borrower_config = scenario.get("borrower", {})
    cecl_config = scenario.get("cecl", {})
    routing_fields = {
        borrower_config.get("module_field", "model_module"),
        borrower_config.get("portfolio_field", "model_portfolio"),
        (
            cecl_config.get("portfolio_field", "cecl_portfolio")
            if isinstance(cecl_config, Mapping)
            else "cecl_portfolio"
        ),
    }

    for tag in tag_defs:
        name = tag["name"]
        tag_col = f"tag_{stable_name(name)}"
        include = tag.get("include", [])
        exclude = tag.get("exclude", [])
        missing_condition_fields = sorted(
            field for field in (condition_fields(include) | condition_fields(exclude)) if field not in result.columns
        )
        for field in missing_condition_fields:
            record_exception(
                exceptions,
                "ERROR",
                "tagging",
                "TAG_CONDITION_FIELD_MISSING",
                "A tag referenced a field that was not present in the borrower identity data; the condition matched no borrowers.",
                field=field,
                source=name,
            )
        if missing_condition_fields and tag.get("exclude_from_model", False):
            raise ValueError(
                f"Model-exclusion tag '{name}' references missing condition "
                f"fields: {', '.join(missing_condition_fields)}."
            )
        mask = evaluate_conditions(result, include)
        if exclude:
            mask &= ~evaluate_conditions(result, exclude)
        result[tag_col] = mask.fillna(False).astype(bool)
        # Assignments let input files stay narrow. For example, a single
        # `subsector` token can derive both model routing and CECL grouping.
        _apply_assignments(
            result,
            result[tag_col],
            tag.get("assign", {}),
            exceptions,
            name,
            (
                routing_fields
                if tag.get("cecl_level", False)
                and tag.get("cecl_module") == "Overlay"
                else set()
            ),
        )

        tagged = result[result[tag_col]]
        row = {
            "tag": name,
            "tag_column": tag_col,
            "model_eligible": bool(tag.get("model_eligible", True)),
            "exclude_from_model": bool(tag.get("exclude_from_model", False)),
            "cecl_level": bool(tag.get("cecl_level", False)),
            "cecl_module": tag.get("cecl_module", np.nan),
            "cecl_priority": tag.get("cecl_priority", np.nan),
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
            tieout_row = _evaluate_tieout(name, tag_col, result, tieout, loaded, balance_field)
            tieout_row["cecl_level"] = bool(tag.get("cecl_level", False))
            tieout_row["cecl_module"] = tag.get("cecl_module", np.nan)
            tieout_row["cecl_priority"] = tag.get(
                "cecl_priority", np.nan
            )
            tag_rows.append(tieout_row)
            match_count = tieout_row.get("expected_match_count")
            if pd.notna(match_count) and int(match_count) == 0:
                record_exception(
                    exceptions,
                    "WARNING",
                    "tagging",
                    "TAG_TIEOUT_EXPECTED_ROW_MISSING",
                    "No expected tie-out row matched this tag; expected amount was treated as zero for display only.",
                    source=str(tieout.get("source", "scenario")),
                    field=name,
                )
            elif pd.notna(match_count) and int(match_count) > 1:
                record_exception(
                    exceptions,
                    "WARNING",
                    "tagging",
                    "TAG_TIEOUT_EXPECTED_ROWS_DUPLICATE",
                    "Multiple expected tie-out rows matched this tag and were summed.",
                    source=str(tieout.get("source", "scenario")),
                    field=name,
                    details=f"match_count={int(match_count)}",
                )
            if tieout_row.get("passed") is False:
                record_exception(
                    exceptions,
                    "WARNING",
                    "tagging",
                    "TAG_TIEOUT_DIFFERENCE",
                    "Tag population did not reconcile to the configured expected total within tolerance.",
                    source=str(tieout.get("source", "scenario")),
                    field=name,
                    details=(
                        f"expected={tieout_row['expected']}; actual={tieout_row['actual']}; "
                        f"difference={tieout_row['difference']}; tolerance={tieout_row['tolerance']}"
                    ),
                )

    tag_summary = pd.DataFrame(tag_rows)
    exclusion_defs = [
        tag for tag in tag_defs if tag.get("exclude_from_model", False)
    ]
    exclusion_columns = [
        f"tag_{stable_name(tag['name'])}" for tag in exclusion_defs
    ]
    if exclusion_columns:
        result["model_excluded"] = (
            result[exclusion_columns].fillna(False).astype(bool).any(axis=1)
        )
        result["model_exclusion_tags"] = _tag_list(
            result, exclusion_defs, include_internal=True
        )
    else:
        result["model_excluded"] = False
        result["model_exclusion_tags"] = ""
    tag_summary = _add_model_exclusion_breakdown(tag_summary, result)
    result["all_tags"] = _tag_list(result, tag_defs, include_internal=True)
    result["model_tags"] = _tag_list(result, tag_defs, include_internal=False)
    result.loc[result["model_excluded"], "model_tags"] = ""
    return result, tag_summary


def normalize_tag_defs(tags: Any) -> List[Dict[str, Any]]:
    """Convert the required object-style tag definitions to a list."""
    if not isinstance(tags, Mapping):
        raise ValueError("Scenario tags must be a JSON object keyed by tag name.")
    out = []
    for name, spec in tags.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Scenario tag names must be nonblank strings.")
        if name != name.strip():
            raise ValueError(
                f"Scenario tag name '{name}' cannot contain surrounding whitespace."
            )
        if not isinstance(spec, Mapping):
            raise ValueError(f"Tag '{name}' must be a JSON object.")
        if "name" in spec:
            raise ValueError(f"Tag '{name}' must use its object key as the name; remove the nested name field.")
        item = dict(spec)
        item["name"] = str(name)
        if "assign" in item and not isinstance(item["assign"], Mapping):
            raise ValueError(f"Tag '{name}' assign must be a JSON object.")
        for flag in ("model_eligible", "exclude_from_model", "cecl_level"):
            if flag in item and not isinstance(item[flag], bool):
                raise ValueError(f"Tag '{name}' {flag} must be a JSON boolean.")
        include_has_conditions = _validate_condition_block(
            item.get("include", []), f"Tag '{name}' include"
        )
        _validate_condition_block(
            item.get("exclude", []), f"Tag '{name}' exclude"
        )
        if item.get("cecl_level", False):
            cecl_module = item.get("cecl_module")
            if not isinstance(cecl_module, str) or not cecl_module.strip():
                raise ValueError(
                    f"CECL-level tag '{name}' must define a nonblank cecl_module."
                )
            cecl_module = cecl_module.strip()
            if cecl_module == "Consumer":
                raise ValueError(
                    f"CECL-level tag '{name}' cannot target Consumer; Consumer "
                    "always uses its current in-place CECL reserve."
                )
            if item.get("exclude_from_model", False):
                raise ValueError(
                    f"Tag '{name}' cannot be both cecl_level and exclude_from_model."
                )
            item["cecl_module"] = cecl_module
            cecl_priority = item.get("cecl_priority", 0)
            if (
                isinstance(cecl_priority, bool)
                or not isinstance(cecl_priority, int)
                or cecl_priority < 0
            ):
                raise ValueError(
                    f"CECL-level tag '{name}' cecl_priority must be a "
                    "nonnegative JSON integer."
                )
            item["cecl_priority"] = cecl_priority
        elif "cecl_module" in item:
            raise ValueError(
                f"Tag '{name}' cecl_module requires cecl_level to be true."
            )
        elif "cecl_priority" in item:
            raise ValueError(
                f"Tag '{name}' cecl_priority requires cecl_level to be true."
            )
        if item.get("exclude_from_model", False):
            if item.get("model_eligible", False):
                raise ValueError(
                    f"Tag '{name}' cannot be both model_eligible and exclude_from_model."
                )
            if not include_has_conditions:
                raise ValueError(
                    f"Tag '{name}' with exclude_from_model must define a "
                    "nonempty include condition."
                )
            item["model_eligible"] = False
        out.append(item)
    cecl_columns: Dict[str, str] = {}
    for tag in out:
        if not tag.get("cecl_level", False):
            continue
        column = stable_name(tag["name"])
        previous = cecl_columns.get(column)
        if previous is not None:
            raise ValueError(
                "CECL-level tag names must produce unique tag columns; "
                f"'{previous}' and '{tag['name']}' both normalize to '{column}'."
            )
        cecl_columns[column] = str(tag["name"])
    return out


def _validate_condition_block(conditions: Any, path: str) -> bool:
    """Validate tag-condition structure and report whether it has an atom."""
    if conditions is None or conditions == [] or conditions == {}:
        return False
    if isinstance(conditions, Mapping):
        logical_keys = [key for key in ("all", "any") if key in conditions]
        if logical_keys:
            if len(logical_keys) != 1 or set(conditions) != {logical_keys[0]}:
                raise ValueError(
                    f"{path} logical conditions must contain exactly one of "
                    "'all' or 'any' and no atomic fields."
                )
            logical_key = logical_keys[0]
            children = conditions[logical_key]
            if not isinstance(children, (list, tuple)) or not children:
                raise ValueError(
                    f"{path}.{logical_key} must be a nonempty JSON list."
                )
            for index, child in enumerate(children):
                if not _validate_condition_block(
                    child, f"{path}.{logical_key}[{index}]"
                ):
                    raise ValueError(
                        f"{path}.{logical_key}[{index}] must contain a "
                        "condition."
                    )
            return True
        field = conditions.get("field")
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"{path} atomic condition requires a nonblank field.")
        op = str(conditions.get("op", "eq")).lower()
        if op not in TAG_CONDITION_OPERATORS:
            raise ValueError(f"{path} uses unsupported operator '{op}'.")
        if op not in {"is_null", "not_null"} and "value" not in conditions:
            raise ValueError(f"{path} operator '{op}' requires a value.")
        value = conditions.get("value")
        if op in {"in", "not_in"}:
            values = as_list(value)
            if not values:
                raise ValueError(
                    f"{path} operator '{op}' requires nonempty values."
                )
            try:
                set(values)
            except TypeError as exc:
                raise ValueError(
                    f"{path} operator '{op}' requires scalar values."
                ) from exc
        if op in {"has_token", "has_any_token", "has_all_tokens"}:
            token_values = [
                item
                for item in as_list(value)
                if not is_missing(item) and str(item).strip()
            ]
            if not token_values:
                raise ValueError(
                    f"{path} operator '{op}' requires nonempty token values."
                )
        if op in {"gt", "gte", "lt", "lte"}:
            number = to_number(value)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(number):
                raise ValueError(
                    f"{path} operator '{op}' requires a finite numeric value."
                )
        if op == "between":
            values = as_list(value)
            if len(values) != 2:
                raise ValueError(
                    f"{path} operator 'between' requires two values."
                )
            if any(
                isinstance(item, (bool, np.bool_))
                or not np.isfinite(to_number(item))
                for item in values
            ):
                raise ValueError(
                    f"{path} operator 'between' requires finite numeric values."
                )
        if op in {"contains", "startswith", "endswith", "regex"} and (
            is_missing(value) or not str(value)
        ):
            raise ValueError(f"{path} operator '{op}' requires a nonempty value.")
        if op == "regex":
            try:
                re.compile(str(value))
            except re.error as exc:
                raise ValueError(f"{path} contains an invalid regex: {exc}.") from exc
        return True
    if isinstance(conditions, (list, tuple)):
        if not conditions:
            return False
        for index, child in enumerate(conditions):
            if not _validate_condition_block(child, f"{path}[{index}]"):
                raise ValueError(f"{path}[{index}] must contain a condition.")
        return True
    raise ValueError(f"{path} must be a JSON object or list.")


def evaluate_conditions(df: pd.DataFrame, conditions: Any) -> pd.Series:
    """Return a boolean mask for JSON condition blocks.

    Called by `apply_tags` and tie-out lookup filtering. Supports implicit
    AND lists plus explicit `all` and `any` blocks for nested logic.
    """
    if not conditions:
        return pd.Series(True, index=df.index)
    if isinstance(conditions, Mapping):
        if "all" in conditions:
            mask = pd.Series(True, index=df.index)
            for item in conditions["all"]:
                mask &= evaluate_conditions(df, item)
            return mask.fillna(False).astype(bool)
        if "any" in conditions:
            mask = pd.Series(False, index=df.index)
            for item in conditions["any"]:
                mask |= evaluate_conditions(df, item)
            return mask.fillna(False).astype(bool)
        return _evaluate_condition(df, conditions).fillna(False).astype(bool)

    if not isinstance(conditions, (list, tuple)):
        raise ValueError("Tag conditions must be an object or list.")
    mask = pd.Series(True, index=df.index)
    for condition in conditions:
        mask &= evaluate_conditions(df, condition)
    return mask.fillna(False).astype(bool)


def _evaluate_condition(df: pd.DataFrame, condition: Mapping[str, Any]) -> pd.Series:
    """Evaluate one atomic condition against a DataFrame column."""
    field = condition.get("field")
    op = str(condition.get("op", "eq")).lower()
    value = condition.get("value")
    if field not in df.columns:
        return pd.Series(False, index=df.index)
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)
    series = df[field]

    if op == "eq":
        return series.apply(lambda item: compare_values(item, value))
    if op == "ne":
        return ~series.apply(lambda item: compare_values(item, value))
    if op == "in":
        values = set(as_list(value))
        return series.isin(values)
    if op == "not_in":
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
        return series.astype(str).str.contains(
            str(value), case=bool(condition.get("case", False)), regex=False, na=False
        )
    if op == "has_token":
        values = set(_normalize_tokens(value, condition))
        return series.apply(lambda item: bool(set(_split_tokens(item, condition)) & values))
    if op == "has_any_token":
        values = set(_normalize_tokens(value, condition))
        return series.apply(lambda item: bool(set(_split_tokens(item, condition)) & values))
    if op == "has_all_tokens":
        values = set(_normalize_tokens(value, condition))
        if not values:
            return pd.Series(False, index=df.index)
        return series.apply(lambda item: values.issubset(set(_split_tokens(item, condition))))
    if op == "startswith":
        left = series.astype(str)
        right = str(value)
        if not bool(condition.get("case", False)):
            left = left.str.lower()
            right = right.lower()
        return left.str.startswith(right, na=False)
    if op == "endswith":
        left = series.astype(str)
        right = str(value)
        if not bool(condition.get("case", False)):
            left = left.str.lower()
            right = right.lower()
        return left.str.endswith(right, na=False)
    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    if op == "regex":
        return series.astype(str).str.contains(
            str(value), regex=True, case=bool(condition.get("case", True)), na=False
        )
    raise ValueError(f"Unsupported tag condition operator: {op}")


def _apply_assignments(
    df: pd.DataFrame,
    mask: pd.Series,
    assignments: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
    tag_name: str,
    tracked_conflict_fields: set[str] | None = None,
) -> None:
    """Write derived field values for rows matched by a tag.

    Called from `apply_tags`. Assignments are intentionally simple so derived
    routing fields are auditable in `borrower_audit_raw.csv`.
    """
    for field, value_spec in assignments.items():
        if field not in df.columns:
            df[field] = pd.Series(pd.NA, index=df.index, dtype=object)
        if isinstance(value_spec, Mapping) and "from_field" in value_spec:
            source = value_spec["from_field"]
            if source not in df.columns:
                raise ValueError(f"Tag assignment references missing source field '{source}'.")
            values = df[source]
        else:
            values = pd.Series(value_spec, index=df.index)
        candidate = values if isinstance(values, pd.Series) else pd.Series(values, index=df.index)
        existing = df[field]
        comparable_existing = existing.astype(str)
        comparable_candidate = candidate.astype(str)
        conflict = mask & existing.notna() & candidate.notna() & comparable_existing.ne(comparable_candidate)
        if conflict.any():
            record_exception(
                exceptions,
                "WARNING",
                "tagging",
                "TAG_ASSIGNMENT_CONFLICT",
                "Multiple tags attempted to assign different values to the same field; the first assignment was retained.",
                field=field,
                source=tag_name,
                details=f"conflict_count={int(conflict.sum())}",
            )
            if field in (tracked_conflict_fields or set()):
                exceptions[-1]["_conflict_indices"] = list(
                    df.index[conflict]
                )
        write_mask = mask & existing.isna()
        df.loc[write_mask, field] = candidate.loc[write_mask]


def _normalize_tokens(value: Any, condition: Mapping[str, Any]) -> List[str]:
    """Normalize a condition value into comparable delimited tokens."""
    tokens: List[str] = []
    for item in as_list(value):
        tokens.extend(_split_tokens(item, condition))
    return tokens


def _split_tokens(value: Any, condition: Mapping[str, Any]) -> List[str]:
    """Split semicolon/comma/pipe token fields for subsector-style tags."""
    if pd.isna(value):
        return []
    delimiters = condition.get("delimiters", [";", "|", ","])
    text = str(value)
    for delimiter in delimiters:
        text = text.replace(str(delimiter), ";")
    case_sensitive = bool(condition.get("case", False))
    tokens = [token.strip() for token in text.split(";") if token.strip()]
    if not case_sensitive:
        tokens = [token.lower() for token in tokens]
    return tokens


def _evaluate_tieout(
    tag_name: str,
    tag_col: str,
    borrowers: pd.DataFrame,
    tieout: Mapping[str, Any],
    loaded: Mapping[str, LoadedTable],
    default_balance_field: str,
) -> Dict[str, Any]:
    """Calculate one tag reconciliation row.

    Called by `apply_tags` when a tag defines `tie_out`. Actual values come
    from the tagged borrower population; expected values come from either JSON
    constants or an external tie-out input table.
    """
    source_name = tieout.get("source")
    actual_field = tieout.get("actual_field", default_balance_field)
    tolerance = float(tieout.get("tolerance", 0.0))
    actual = float(pd.to_numeric(borrowers.loc[borrowers[tag_col], actual_field], errors="coerce").sum())
    expected = tieout.get("expected")
    expected_match_count = np.nan
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
            if key_field not in filtered.columns:
                raise ValueError(f"Tie-out source '{source_name}' missing key field '{key_field}'.")
            match_value = tieout.get("match_value", tag_name)
            filtered = filtered[filtered[key_field] == match_value]
        expected_match_count = int(len(filtered))
        expected = float(pd.to_numeric(filtered[amount_field], errors="coerce").sum())
    expected = float(expected)
    difference = actual - expected
    return {
        "tag": tag_name,
        "tag_column": tag_col,
        "model_eligible": np.nan,
        "exclude_from_model": np.nan,
        "borrower_count": int(borrowers[tag_col].sum()),
        "balance_field": actual_field,
        "balance": actual,
        "tie_out_name": tieout.get("name", tag_name),
        "expected": expected,
        "actual": actual,
        "difference": difference,
        "tolerance": tolerance,
        "passed": bool(abs(difference) <= tolerance),
        "expected_match_count": expected_match_count,
    }


def _tag_list(df: pd.DataFrame, tag_defs: List[Mapping[str, Any]], include_internal: bool) -> pd.Series:
    """Build semicolon-delimited tag audit columns."""
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


def _add_model_exclusion_breakdown(
    tag_summary: pd.DataFrame,
    borrowers: pd.DataFrame,
) -> pd.DataFrame:
    """Split each raw tag population into model-included and excluded amounts."""
    summary = tag_summary.copy()
    included_counts: List[int] = []
    included_balances: List[float] = []
    excluded_counts: List[int] = []
    excluded_balances: List[float] = []
    excluded_rows = borrowers["model_excluded"].fillna(False).astype(bool)
    for _, row in summary.iterrows():
        tag_column = row.get("tag_column")
        if tag_column not in borrowers.columns:
            tag_mask = pd.Series(False, index=borrowers.index)
        else:
            tag_mask = borrowers[tag_column].fillna(False).astype(bool)
        included = tag_mask & ~excluded_rows
        excluded = tag_mask & excluded_rows
        balance_field = row.get("balance_field")
        if balance_field in borrowers.columns:
            balances = pd.to_numeric(
                borrowers[balance_field], errors="coerce"
            )
        else:
            balances = pd.Series(np.nan, index=borrowers.index)
        included_counts.append(int(included.sum()))
        included_balances.append(float(balances.loc[included].sum()))
        excluded_counts.append(int(excluded.sum()))
        excluded_balances.append(float(balances.loc[excluded].sum()))
    summary["not_model_excluded_borrower_count"] = included_counts
    summary["not_model_excluded_balance"] = included_balances
    summary["model_excluded_borrower_count"] = excluded_counts
    summary["model_excluded_balance"] = excluded_balances
    return summary


def model_eligible_tag_names(scenario: Mapping[str, Any]) -> set[str]:
    """Return tags allowed to control module populations."""
    tags = normalize_tag_defs(scenario.get("tags", {}))
    return {str(tag["name"]) for tag in tags if tag.get("model_eligible", True)}


def assign_primary_modules(
    df: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Assign one primary stress module from active model tags.

    Raw tag flags and ``all_tags`` remain visible for auditability. Active
    non-excluded model tags remain in ``model_tags``; excluded rows receive no
    model tags or routing. Downstream stress modules use ``primary_module`` to
    avoid double-stressing borrowers that satisfy more than one model tag.
    """
    exceptions = exceptions if exceptions is not None else []
    result = df.copy()
    modules = scenario.get("modules", {})
    priority = [
        str(item)
        for item in scenario.get("module_order", ["CRE", "C&I", "Consumer"])
    ]
    priority_rank = {name: idx for idx, name in enumerate(priority)}
    allowed_tags = model_eligible_tag_names(scenario)
    module_specs = []
    for module_name, config in modules.items():
        if not config.get("enabled", True):
            continue
        module_specs.append(
            {
                "name": str(module_name),
                "rank": priority_rank.get(str(module_name), len(priority_rank) + len(module_specs)),
                "eligible_tags": [
                    str(tag)
                    for tag in as_list(config.get("eligible_tags"))
                    if str(tag) in allowed_tags
                ],
            }
        )
    module_specs.sort(key=lambda item: (item["rank"], item["name"]))
    enabled_modules = {spec["name"] for spec in module_specs}
    routable_modules = set(enabled_modules)
    if scenario.get("overlays"):
        routable_modules.add("Overlay")
    result["eligible_modules"] = ""
    # ``primary_module`` is an engine-derived routing decision. Never trust a
    # same-named input column from a prior run or source file.
    result["primary_module"] = pd.Series(pd.NA, index=result.index, dtype=object)

    borrower_cfg = scenario.get("borrower", {})
    borrower_id_field = borrower_cfg.get("borrower_id_field", "borrower_id")
    module_field = borrower_cfg.get("module_field", "model_module")
    portfolio_field = borrower_cfg.get("portfolio_field", "model_portfolio")
    cecl = scenario.get("cecl", {})
    cecl_portfolio_field = cecl.get("portfolio_field", "cecl_portfolio")
    if module_field not in result.columns:
        result[module_field] = pd.Series(pd.NA, index=result.index, dtype=object)
    else:
        result[module_field] = result[module_field].astype(object)
    if portfolio_field not in result.columns:
        result[portfolio_field] = pd.Series(pd.NA, index=result.index, dtype=object)
    else:
        result[portfolio_field] = result[portfolio_field].astype(object)
    if cecl_portfolio_field not in result.columns:
        result[cecl_portfolio_field] = pd.Series(pd.NA, index=result.index, dtype=object)
    else:
        result[cecl_portfolio_field] = result[cecl_portfolio_field].astype(object)

    for idx, row in result.iterrows():
        model_excluded = row.get("model_excluded", False)
        if not is_missing(model_excluded) and bool(model_excluded):
            result.at[idx, "eligible_modules"] = ""
            result.at[idx, "primary_module"] = pd.NA
            result.at[idx, module_field] = pd.NA
            result.at[idx, portfolio_field] = pd.NA
            result.at[idx, cecl_portfolio_field] = pd.NA
            continue
        active = []
        for spec in module_specs:
            if any(bool(row.get(f"tag_{stable_name(tag)}", False)) for tag in spec["eligible_tags"]):
                active.append(spec)
        if not active:
            existing_module = row.get(module_field)
            if not is_missing(existing_module) and str(existing_module).strip():
                existing_name = str(existing_module).strip()
                if existing_name not in routable_modules:
                    raise ValueError(
                        f"Input module field '{module_field}' references module "
                        f"'{existing_name}' that is not enabled and configured "
                        f"(borrower_id={row.get(borrower_id_field)})."
                    )
                if existing_name == "Overlay":
                    overlay_portfolios = {
                        str(portfolio)
                        for portfolio, config in scenario.get("overlays", {}).items()
                        if config.get("enabled", True)
                    }
                    existing_portfolio = row.get(portfolio_field)
                    portfolio_name = (
                        ""
                        if is_missing(existing_portfolio)
                        else str(existing_portfolio).strip()
                    )
                    if portfolio_name not in overlay_portfolios:
                        raise ValueError(
                            f"Input module field '{module_field}' routes borrower "
                            f"'{row.get(borrower_id_field)}' to Overlay portfolio "
                            f"'{portfolio_name}', which is not enabled and configured."
                        )
                result.at[idx, "primary_module"] = existing_name
            existing_portfolio = row.get(portfolio_field)
            if pd.notna(existing_portfolio):
                result.at[idx, cecl_portfolio_field] = existing_portfolio
            continue
        selected = active[0]
        # The selected module is the first active module after priority sort.
        # Other active modules remain in `eligible_modules` for audit review.
        result.at[idx, "eligible_modules"] = ";".join(spec["name"] for spec in active)
        if len(active) > 1:
            record_exception(
                exceptions,
                "WARNING",
                "tagging",
                "MODEL_MODULE_OVERLAP_RESOLVED",
                "Borrower matched multiple model modules; configured priority selected the primary module.",
                borrower_id=row.get(scenario.get("borrower", {}).get("borrower_id_field", "borrower_id")),
                module=selected["name"],
                details=f"eligible_modules={';'.join(spec['name'] for spec in active)}",
            )
        result.at[idx, "primary_module"] = selected["name"]
        result.at[idx, module_field] = selected["name"]
        result.at[idx, portfolio_field] = selected["name"]
        selected_module_config = modules.get(selected["name"], {})
        cecl_rollup = selected_module_config.get("cecl_portfolio_rollup")
        cecl_source_field = selected_module_config.get("cecl_portfolio_field")
        if cecl_rollup:
            result.at[idx, cecl_portfolio_field] = cecl_rollup
        elif cecl_source_field and cecl_source_field in result.columns and pd.notna(row.get(cecl_source_field)):
            result.at[idx, cecl_portfolio_field] = row.get(cecl_source_field)
        else:
            result.at[idx, cecl_portfolio_field] = selected["name"]
    return result


def resolve_cecl_level_tags(
    df: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
    *,
    emit_priority_warnings: bool = True,
) -> pd.DataFrame:
    """Resolve one CECL calibration tag after primary-module routing.

    CECL-level tags are scoped to a primary module so intentional cross-module
    tag overlap does not make the CECL population ambiguous. A uniquely lowest
    ``cecl_priority`` resolves same-module overlap; tied candidates still fail
    closed. Once any such tag is configured, every included CRE, C&I, or
    Overlay row must resolve to one tag in its selected module. Consumer and
    model-excluded rows retain their existing treatment and never enter tagged
    commercial calibration.
    """
    exceptions = exceptions if exceptions is not None else []
    result = df.copy()
    tag_defs = normalize_tag_defs(scenario.get("tags", {}))
    cecl_tags = [tag for tag in tag_defs if tag.get("cecl_level", False)]
    cecl = scenario.get("cecl", {})
    cecl_portfolio_field = (
        str(cecl.get("portfolio_field", "cecl_portfolio"))
        if isinstance(cecl, Mapping)
        else "cecl_portfolio"
    )
    result[CECL_LEVEL_TAG_FIELD] = pd.Series(
        pd.NA, index=result.index, dtype=object
    )

    if not cecl_tags:
        if cecl_portfolio_field in result.columns:
            result[CECL_LEVEL_TAG_FIELD] = result[cecl_portfolio_field].map(
                lambda value: (
                    value if is_missing(value) else str(value).strip()
                )
            )
        _validate_repeated_commercial_cecl_rows(result, scenario)
        return result

    borrower_id_field = scenario.get("borrower", {}).get(
        "borrower_id_field", "borrower_id"
    )
    loan_id_field = scenario.get("borrower", {}).get(
        "loan_id_field", "loan_id"
    )
    priority_resolved_indices: set[Any] = set()
    for idx, row in result.iterrows():
        excluded = row.get("model_excluded", False)
        if not is_missing(excluded) and bool(excluded):
            continue
        module_value = row.get("primary_module")
        module = (
            "" if is_missing(module_value) else str(module_value).strip()
        )
        if module == "Consumer":
            if cecl_portfolio_field in result.columns:
                portfolio = row.get(cecl_portfolio_field)
                if not is_missing(portfolio):
                    result.at[idx, CECL_LEVEL_TAG_FIELD] = str(
                        portfolio
                    ).strip()
            continue
        if module not in CECL_LEVEL_MODULES:
            continue

        candidates: List[Dict[str, Any]] = []
        for tag in cecl_tags:
            if tag.get("cecl_module") != module:
                continue
            column = f"tag_{stable_name(tag['name'])}"
            value = row.get(column, False)
            if not is_missing(value) and bool(value):
                candidates.append(tag)
        candidates.sort(
            key=lambda tag: (
                int(tag.get("cecl_priority", 0)), str(tag["name"])
            )
        )
        borrower_id = row.get(borrower_id_field)
        if not candidates:
            raise ValueError(
                "Model-included commercial row matched no CECL-level tag for "
                f"primary module '{module}' (borrower_id={borrower_id})."
            )
        priority_resolved = False
        if len(candidates) > 1:
            selected_priority = min(
                int(tag.get("cecl_priority", 0)) for tag in candidates
            )
            preferred = [
                tag
                for tag in candidates
                if int(tag.get("cecl_priority", 0)) == selected_priority
            ]
            candidate_details = ";".join(
                f"{tag['name']}={int(tag.get('cecl_priority', 0))}"
                for tag in candidates
            )
            if len(preferred) > 1:
                raise ValueError(
                    "Model-included commercial row matched multiple CECL-level "
                    f"tags for primary module '{module}' at the same winning "
                    f"priority (borrower_id={borrower_id}): "
                    f"{candidate_details}."
                )
            selected = preferred[0]
            priority_resolved = True
        else:
            selected = candidates[0]
        result.at[idx, CECL_LEVEL_TAG_FIELD] = str(selected["name"])
        route_audit: Dict[str, str] = {}
        if module == "Overlay" and priority_resolved:
            route_audit = _apply_selected_overlay_route(
                result, idx, row, selected, scenario
            )
        if priority_resolved:
            priority_resolved_indices.add(idx)
            if emit_priority_warnings:
                details = (
                    f"candidates={candidate_details}; "
                    f"selected={selected['name']}; "
                    f"selected_priority={selected_priority}"
                )
                if route_audit:
                    details += (
                        "; previous_model_portfolio="
                        f"{route_audit['previous_model_portfolio']}; "
                        "previous_cecl_portfolio="
                        f"{route_audit['previous_cecl_portfolio']}; "
                        "resolved_portfolio="
                        f"{route_audit['resolved_portfolio']}"
                    )
                record_exception(
                    exceptions,
                    "WARNING",
                    "tagging",
                    "CECL_LEVEL_TAG_OVERLAP_RESOLVED_BY_PRIORITY",
                    "Borrower matched multiple same-module CECL-level tags; configured priority selected one tag.",
                    borrower_id=borrower_id,
                    loan_id=row.get(loan_id_field, pd.NA),
                    module=module,
                    field=CECL_LEVEL_TAG_FIELD,
                    details=details,
                    scenario_variant=(
                        "all" if scenario.get("targeted_stress") else ""
                    ),
                )

    _reconcile_priority_resolved_assignment_conflicts(
        exceptions, priority_resolved_indices, cecl_tags, scenario
    )
    _validate_repeated_commercial_cecl_rows(result, scenario)
    return result


def _apply_selected_overlay_route(
    df: pd.DataFrame,
    idx: Any,
    row: Mapping[str, Any],
    selected_tag: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> Dict[str, str]:
    """Keep Overlay public routing aligned with the selected CECL tag."""
    borrower = scenario.get("borrower", {})
    module_field = borrower.get("module_field", "model_module")
    portfolio_field = borrower.get("portfolio_field", "model_portfolio")
    cecl = scenario.get("cecl", {})
    cecl_portfolio_field = (
        cecl.get("portfolio_field", "cecl_portfolio")
        if isinstance(cecl, Mapping)
        else "cecl_portfolio"
    )
    assignments = selected_tag.get("assign", {})
    if not isinstance(assignments, Mapping):
        raise ValueError(
            f"Priority-selected Overlay CECL tag '{selected_tag['name']}' "
            "must define routing assignments."
        )
    assigned_module = _resolved_assignment_value(
        row, assignments.get(module_field, pd.NA)
    )
    if is_missing(assigned_module) or str(assigned_module).strip() != "Overlay":
        raise ValueError(
            f"Priority-selected Overlay CECL tag '{selected_tag['name']}' "
            f"must assign '{module_field}' to 'Overlay'."
        )
    portfolio = _resolved_assignment_value(
        row, assignments.get(portfolio_field, pd.NA)
    )
    cecl_portfolio = _resolved_assignment_value(
        row, assignments.get(cecl_portfolio_field, pd.NA)
    )
    if is_missing(portfolio) or not str(portfolio).strip():
        raise ValueError(
            f"Priority-selected Overlay CECL tag '{selected_tag['name']}' "
            f"must assign a nonblank '{portfolio_field}'."
        )
    portfolio = str(portfolio).strip()
    if (
        not is_missing(cecl_portfolio)
        and str(cecl_portfolio).strip()
        and str(cecl_portfolio).strip() != portfolio
    ):
        raise ValueError(
            f"Selected Overlay CECL tag '{selected_tag['name']}' assigns "
            f"model portfolio '{portfolio}' but CECL portfolio "
            f"'{str(cecl_portfolio).strip()}'. Overlay routing fields "
            "must agree."
        )
    overlays = scenario.get("overlays", {})
    if not isinstance(overlays, Mapping):
        raise ValueError("Scenario overlays must be a JSON object.")
    enabled_overlays = {
        str(name)
        for name, config in overlays.items()
        if isinstance(config, Mapping) and config.get("enabled", True)
    }
    if portfolio not in enabled_overlays:
        raise ValueError(
            f"Selected Overlay CECL tag '{selected_tag['name']}' routes "
            f"to portfolio '{portfolio}', which is not enabled and "
            "configured."
        )
    previous_model_portfolio = _route_audit_value(row.get(portfolio_field))
    previous_cecl_portfolio = _route_audit_value(
        row.get(cecl_portfolio_field)
    )
    df.at[idx, module_field] = "Overlay"
    df.at[idx, portfolio_field] = portfolio
    df.at[idx, cecl_portfolio_field] = portfolio
    return {
        "previous_model_portfolio": previous_model_portfolio,
        "previous_cecl_portfolio": previous_cecl_portfolio,
        "resolved_portfolio": portfolio,
    }


def _route_audit_value(value: Any) -> str:
    if is_missing(value) or not str(value).strip():
        return "<blank>"
    return str(value).strip()


def _reconcile_priority_resolved_assignment_conflicts(
    exceptions: List[Dict[str, Any]],
    priority_resolved_indices: set[Any],
    cecl_tags: List[Dict[str, Any]],
    scenario: Mapping[str, Any],
) -> None:
    """Retain routing conflicts unless priority corrected those exact rows."""
    overlay_tag_names = {
        str(tag["name"])
        for tag in cecl_tags
        if tag.get("cecl_module") == "Overlay"
    }
    borrower = scenario.get("borrower", {})
    cecl = scenario.get("cecl", {})
    routing_fields = {
        borrower.get("module_field", "model_module"),
        borrower.get("portfolio_field", "model_portfolio"),
        (
            cecl.get("portfolio_field", "cecl_portfolio")
            if isinstance(cecl, Mapping)
            else "cecl_portfolio"
        ),
    }
    retained: List[Dict[str, Any]] = []
    for event in exceptions:
        conflict_indices = event.pop("_conflict_indices", None)
        if conflict_indices is None:
            retained.append(event)
            continue
        is_tracked_overlay_conflict = (
            event.get("code") == "TAG_ASSIGNMENT_CONFLICT"
            and str(event.get("source", "")) in overlay_tag_names
            and str(event.get("field", "")) in routing_fields
        )
        if not is_tracked_overlay_conflict:
            retained.append(event)
            continue
        unresolved = [
            idx
            for idx in conflict_indices
            if idx not in priority_resolved_indices
        ]
        if not unresolved:
            continue
        details = str(event.get("details", ""))
        event["details"] = re.sub(
            r"conflict_count=\d+",
            f"conflict_count={len(unresolved)}",
            details,
        )
        retained.append(event)
    exceptions[:] = retained


def _resolved_assignment_value(
    row: Mapping[str, Any], value_spec: Any
) -> Any:
    if isinstance(value_spec, Mapping) and "from_field" in value_spec:
        return row.get(value_spec["from_field"], pd.NA)
    return value_spec


def add_cecl_selection_summary(
    tag_summary: pd.DataFrame,
    resolved: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> pd.DataFrame:
    """Add resolved CECL populations beside inclusive raw tag populations."""
    summary = tag_summary.copy()
    summary["cecl_selected_borrower_count"] = np.nan
    summary["cecl_selected_balance"] = np.nan
    has_loan_counts = "loan_count" in summary.columns
    if has_loan_counts:
        summary["cecl_selected_loan_count"] = np.nan
    if summary.empty or "cecl_level" not in summary.columns:
        return summary

    borrower = scenario.get("borrower", {})
    borrower_id_field = borrower.get("borrower_id_field", "borrower_id")
    balance_field = borrower.get("balance_field", "outstanding_balance")
    cecl_rows = summary["cecl_level"].fillna(False).astype(bool)
    population_cecl_rows = cecl_rows.copy()
    if "tie_out_name" in summary.columns:
        population_cecl_rows &= summary["tie_out_name"].isna()
    summary.loc[
        population_cecl_rows, "cecl_selected_borrower_count"
    ] = 0
    summary.loc[population_cecl_rows, "cecl_selected_balance"] = 0.0
    if has_loan_counts:
        summary.loc[
            population_cecl_rows, "cecl_selected_loan_count"
        ] = 0

    if CECL_LEVEL_TAG_FIELD not in resolved.columns:
        return summary
    for tag_name in summary.loc[
        population_cecl_rows, "tag"
    ].dropna().unique():
        tag_rows = population_cecl_rows & summary["tag"].eq(tag_name)
        selected_mask = resolved[CECL_LEVEL_TAG_FIELD].astype(str).eq(
            str(tag_name)
        )
        if "model_excluded" in resolved.columns:
            selected_mask &= ~resolved["model_excluded"].fillna(False).astype(
                bool
            )
        if "cecl_module" in summary.columns and "primary_module" in resolved.columns:
            modules = {
                str(value).strip()
                for value in summary.loc[tag_rows, "cecl_module"].dropna()
                if str(value).strip()
            }
            if len(modules) == 1:
                selected_mask &= resolved["primary_module"].astype(str).eq(
                    next(iter(modules))
                )
        selected = resolved[selected_mask]
        borrower_count = (
            int(selected[borrower_id_field].nunique(dropna=True))
            if borrower_id_field in selected.columns
            else int(len(selected))
        )
        balance = (
            float(
                pd.to_numeric(
                    selected[balance_field], errors="coerce"
                ).sum()
            )
            if balance_field in selected.columns
            else 0.0
        )
        summary.loc[tag_rows, "cecl_selected_borrower_count"] = (
            borrower_count
        )
        summary.loc[tag_rows, "cecl_selected_balance"] = balance
        if has_loan_counts:
            summary.loc[tag_rows, "cecl_selected_loan_count"] = int(
                len(selected)
            )
    return summary


def _validate_repeated_commercial_cecl_rows(
    df: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> None:
    """Require loan-grain rows to agree with borrower-grain CECL routing."""
    borrower = scenario.get("borrower", {})
    borrower_id_field = borrower.get("borrower_id_field", "borrower_id")
    if borrower_id_field not in df.columns or "primary_module" not in df.columns:
        return
    included = pd.Series(True, index=df.index)
    if "model_excluded" in df.columns:
        included &= ~df["model_excluded"].fillna(False).astype(bool)
    commercial = included & df["primary_module"].astype(str).isin(
        CECL_LEVEL_MODULES
    )
    work = df.loc[commercial].copy()
    if work.empty:
        return

    if "base_bucket" in work.columns:
        work["_cecl_bucket_check"] = work["base_bucket"].map(
            lambda value: (
                "Unknown"
                if is_missing(value) or not str(value).strip()
                else str(value).strip()
            )
        )
    else:
        risk_rating_field = borrower.get("risk_rating_field", "risk_rating")
        if risk_rating_field in work.columns:
            work["_cecl_bucket_check"] = work[risk_rating_field].apply(
                risk_bucket_from_rating
            )
        else:
            work["_cecl_bucket_check"] = "Unknown"

    for borrower_id, group in work.groupby(
        borrower_id_field, dropna=False, sort=False
    ):
        if len(group) < 2:
            continue
        tags = sorted(
            {
                "<missing>"
                if is_missing(value) or not str(value).strip()
                else str(value).strip()
                for value in group[CECL_LEVEL_TAG_FIELD]
            }
        )
        if len(tags) > 1:
            raise ValueError(
                "Repeated commercial rows for one borrower resolved to "
                f"different CECL-level tags (borrower_id={borrower_id}): "
                f"{', '.join(tags)}."
            )
        buckets = sorted(
            {
                str(value).strip()
                for value in group["_cecl_bucket_check"].dropna()
                if str(value).strip()
            }
        )
        if len(buckets) > 1:
            raise ValueError(
                "Repeated commercial rows for one borrower resolve to "
                f"different base risk buckets (borrower_id={borrower_id}): "
                f"{', '.join(buckets)}."
            )
