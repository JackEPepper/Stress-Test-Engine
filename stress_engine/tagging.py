"""JSON-defined borrower tagging and tie-out checks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .exceptions import record_exception
from .io import LoadedTable
from .utils import as_list, compare_values, condition_fields, is_missing, stable_name, to_number


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
        )

        tagged = result[result[tag_col]]
        row = {
            "tag": name,
            "tag_column": tag_col,
            "model_eligible": bool(tag.get("model_eligible", True)),
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
    result["all_tags"] = _tag_list(result, tag_defs, include_internal=True)
    result["model_tags"] = _tag_list(result, tag_defs, include_internal=False)
    return result, tag_summary


def normalize_tag_defs(tags: Any) -> List[Dict[str, Any]]:
    """Convert the required object-style tag definitions to a list."""
    if not isinstance(tags, Mapping):
        raise ValueError("Scenario tags must be a JSON object keyed by tag name.")
    out = []
    for name, spec in tags.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"Tag '{name}' must be a JSON object.")
        if "name" in spec:
            raise ValueError(f"Tag '{name}' must use its object key as the name; remove the nested name field.")
        item = dict(spec)
        item["name"] = str(name)
        out.append(item)
    return out


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
    """Evaluate one atomic condition against a DataFrame column."""
    field = condition.get("field")
    op = str(condition.get("op", "eq")).lower()
    value = condition.get("value")
    if field not in df.columns:
        return pd.Series(False, index=df.index)
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

    The active model tags remain visible in ``model_tags`` for auditability, but
    downstream stress modules use ``primary_module`` to avoid double-stressing
    borrowers that satisfy more than one model tag.
    """
    exceptions = exceptions if exceptions is not None else []
    result = df.copy()
    modules = scenario.get("modules", {})
    priority = [
        str(item)
        for item in scenario.get("module_order", ["CRE", "C&I", "Consumer"])
    ]
    priority_rank = {name: idx for idx, name in enumerate(priority)}
    module_specs = []
    for module_name, config in modules.items():
        if not config.get("enabled", True):
            continue
        module_specs.append(
            {
                "name": str(module_name),
                "rank": priority_rank.get(str(module_name), len(priority_rank) + len(module_specs)),
                "eligible_tags": [str(tag) for tag in as_list(config.get("eligible_tags"))],
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
