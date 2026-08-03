"""Borrower universe construction and deterministic enrichment."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .cecl import (
    INVALID_BALANCE_COUNT_FIELD,
    reserve_basis_fields,
    reserve_missing_count_field,
)
from .exceptions import record_exception
from .io import LoadedTable
from .utils import as_list, condition_fields, ensure_columns, first_non_null, join_unique, parse_date_series


def build_borrowers(
    identity: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Collapse the loan-level identity file into one row per borrower.

    Called by `StressEngine.run` immediately after input loading. Only fields
    listed in scenario `sum_fields` are summed; all other fields follow explicit
    aggregation rules or the configured default. This protects borrower-level
    attributes from being accidentally added across multiple loans.
    """
    exceptions = exceptions if exceptions is not None else []
    config = scenario.get("borrower", {})
    borrower_id = config["borrower_id_field"]
    balance_field = config["balance_field"]
    balance_fields = set(config.get("sum_fields", [config["balance_field"]]))
    aggregations: Dict[str, Any] = dict(config.get("aggregation", {}))
    loan_id_field = config.get("loan_id_field")

    ensure_columns(identity, [borrower_id], "Identity input")
    identity = identity.copy()
    valid_balance = pd.Series(True, index=identity.index)
    if balance_field in identity.columns:
        balance_values = pd.to_numeric(identity[balance_field], errors="coerce")
        valid_balance = pd.Series(
            np.isfinite(balance_values), index=identity.index
        )
        identity[balance_field] = balance_values.where(valid_balance)
        # Keep negative balances visible in reported totals, but carry an
        # explicit counter so CECL never treats them as a valid zero balance
        # after borrower aggregation.
        identity[INVALID_BALANCE_COUNT_FIELD] = (
            (~valid_balance) | balance_values.lt(0)
        ).astype(int)
        balance_fields.add(INVALID_BALANCE_COUNT_FIELD)
    for reserve_field in reserve_basis_fields(scenario):
        if reserve_field not in identity.columns:
            continue
        count_field = reserve_missing_count_field(reserve_field)
        reserve_values = pd.to_numeric(identity[reserve_field], errors="coerce")
        finite = np.isfinite(reserve_values)
        identity[reserve_field] = reserve_values.where(
            finite & valid_balance
        )
        identity[count_field] = (~finite).astype(int)
        balance_fields.add(count_field)
    _separate_missing_borrower_ids(identity, borrower_id, exceptions)
    _record_identity_key_issues(identity, borrower_id, loan_id_field, exceptions)
    largest_fields = _largest_loan_fields(scenario, identity.columns)
    for field in balance_fields:
        if field in identity.columns:
            aggregations.setdefault(field, "sum")
    if loan_id_field and loan_id_field in identity.columns:
        aggregations.setdefault(loan_id_field, "list_unique")

    agg_map: Dict[str, Any] = {}
    for column in identity.columns:
        if column == borrower_id or str(column).startswith("_source_") or column in largest_fields:
            continue
        method = aggregations.get(column, "first")
        agg_map[column] = _aggregation_callable(method)

    borrowers = (
        identity.sort_values([borrower_id, "_source_row"], kind="mergesort")
        .groupby(borrower_id, dropna=False)
        .agg(agg_map)
        .reset_index()
    )
    loan_counts = identity.groupby(borrower_id, dropna=False).size().rename("loan_count").reset_index()
    borrowers = borrowers.merge(loan_counts, on=borrower_id, how="left")
    # Rating, maturity, and every tag-driving attribute must come from the same
    # loan. The largest balance is the deterministic representative loan.
    largest_rows = _largest_loan_rows(identity, borrower_id, balance_field, largest_fields)
    if not largest_rows.empty:
        borrowers = borrowers.merge(largest_rows, on=borrower_id, how="left")
    _record_largest_loan_conflicts(
        identity,
        borrower_id,
        balance_field,
        loan_id_field,
        largest_fields,
        exceptions,
    )
    borrowers = borrowers.sort_values([borrower_id], kind="mergesort").reset_index(drop=True)
    return borrowers


def _separate_missing_borrower_ids(
    identity: pd.DataFrame,
    borrower_id: str,
    exceptions: List[Dict[str, Any]],
) -> None:
    """Give null borrower IDs deterministic placeholders so rows never coalesce."""
    missing = identity[borrower_id].isna() | identity[borrower_id].astype(str).str.strip().eq("")
    for idx in identity.index[missing]:
        source_row = identity.at[idx, "_source_row"] if "_source_row" in identity.columns else idx + 1
        placeholder = f"__MISSING_BORROWER_ID_ROW_{source_row}"
        identity.at[idx, borrower_id] = placeholder
        record_exception(
            exceptions,
            "ERROR",
            "identity",
            "IDENTITY_BORROWER_ID_MISSING",
            "Identity row had no borrower ID; a deterministic placeholder preserved it as a separate exposure.",
            borrower_id=placeholder,
            field=borrower_id,
            details=f"source_row={source_row}",
        )


def _record_identity_key_issues(
    identity: pd.DataFrame,
    borrower_id: str,
    loan_id_field: str | None,
    exceptions: List[Dict[str, Any]],
) -> None:
    """Log duplicate or missing loan identifiers without stopping the run."""
    if not loan_id_field or loan_id_field not in identity.columns:
        return
    missing = identity[loan_id_field].isna() | identity[loan_id_field].astype(str).str.strip().eq("")
    if missing.any():
        record_exception(
            exceptions,
            "WARNING",
            "identity",
            "IDENTITY_LOAN_ID_MISSING",
            "Identity rows were missing loan IDs.",
            field=loan_id_field,
            details=f"missing_count={int(missing.sum())}",
        )
    duplicate = identity[loan_id_field].notna() & identity[loan_id_field].duplicated(keep=False)
    if duplicate.any():
        record_exception(
            exceptions,
            "WARNING",
            "identity",
            "IDENTITY_LOAN_ID_DUPLICATE",
            "Duplicate loan IDs were present in the identity input.",
            field=loan_id_field,
            details=f"duplicate_row_count={int(duplicate.sum())}",
        )


def _largest_loan_fields(scenario: Mapping[str, Any], columns: Sequence[str]) -> List[str]:
    """Return loan attributes that must be inherited from the largest loan."""
    config = scenario.get("borrower", {})
    aggregate_or_key_fields = {
        config.get("borrower_id_field"),
        config.get("balance_field"),
        *as_list(config.get("sum_fields")),
    }
    fields = {
        config.get("risk_rating_field", "risk_rating"),
        config.get("maturity_date_field", "maturity_date"),
        config.get("portfolio_field"),
        config.get("module_field"),
    }
    for tag in scenario.get("tags", {}).values():
        fields.update(condition_fields(tag.get("include", [])))
        fields.update(condition_fields(tag.get("exclude", [])))
        for value_spec in tag.get("assign", {}).values():
            if isinstance(value_spec, Mapping) and value_spec.get("from_field"):
                fields.add(value_spec["from_field"])
    return sorted(
        field
        for field in fields
        if field
        and field in columns
        and field not in aggregate_or_key_fields
    )


def _largest_loan_rows(
    identity: pd.DataFrame,
    borrower_id: str,
    balance_field: str,
    fields: Sequence[str],
) -> pd.DataFrame:
    """Select one coherent representative loan per borrower by largest balance."""
    if not fields:
        return pd.DataFrame(columns=[borrower_id])
    work = identity[[borrower_id, balance_field, *fields] + (["_source_row"] if "_source_row" in identity.columns else [])].copy()
    work["_selection_balance"] = pd.to_numeric(work[balance_field], errors="coerce").fillna(-np.inf)
    if "_source_row" not in work.columns:
        work["_source_row"] = np.arange(1, len(work) + 1)
    work = work.sort_values(
        [borrower_id, "_selection_balance", "_source_row"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return work.drop_duplicates(borrower_id, keep="first")[[borrower_id, *fields]]


def _record_largest_loan_conflicts(
    identity: pd.DataFrame,
    borrower_id: str,
    balance_field: str,
    loan_id_field: str | None,
    fields: Sequence[str],
    exceptions: List[Dict[str, Any]],
) -> None:
    """Log borrowers whose loans disagree on inherited rating/tag/maturity fields."""
    if not fields:
        return
    selected = _largest_loan_rows(
        identity,
        borrower_id,
        balance_field,
        list(dict.fromkeys([*fields, *([loan_id_field] if loan_id_field and loan_id_field in identity.columns else [])])),
    ).set_index(borrower_id)
    for value, group in identity.groupby(borrower_id, dropna=False, sort=True):
        if len(group) < 2:
            continue
        conflicting = [field for field in fields if group[field].dropna().astype(str).nunique() > 1]
        if not conflicting:
            continue
        selected_row = selected.loc[value]
        selected_loan = selected_row.get(loan_id_field, "") if loan_id_field else ""
        record_exception(
            exceptions,
            "WARNING",
            "borrower",
            "BORROWER_LOAN_ATTRIBUTE_CONFLICT",
            "Multiple loans for one borrower disagreed on rating, maturity, or tag-driving attributes; the largest loan was selected.",
            borrower_id=value,
            field=";".join(conflicting),
            source=str(selected_loan),
            details=f"loan_count={len(group)}; selected_loan={selected_loan}",
        )


def enrich_borrowers(
    borrowers: pd.DataFrame,
    loaded: Mapping[str, LoadedTable],
    scenario: Mapping[str, Any],
) -> pd.DataFrame:
    """Merge configured non-identity sources onto borrower rows.

    Account-keyed sources are crosswalked through identity first. Every source
    is then aggregated by borrower key to avoid many-to-many joins.
    """
    out = borrowers.copy()
    borrower_id = scenario["borrower"]["borrower_id_field"]
    source_specs = scenario.get("inputs", {}).get("sources", {})
    identity_table = loaded.get("identity")
    for name in sorted(source_specs):
        spec = source_specs[name]
        if spec.get("merge", True) is False:
            continue
        if name not in loaded:
            continue
        source_frame = loaded[name].frame
        group_key = spec.get("key", borrower_id)
        if spec.get("identity_key"):
            if identity_table is None:
                raise ValueError(f"Source '{name}' requires the identity input for account linkage.")
            source_frame = _link_source_to_identity(
                name,
                source_frame,
                spec,
                identity_table.frame,
                borrower_id,
            )
            group_key = borrower_id
        aggregated = aggregate_source(name, source_frame, spec, group_key)
        if aggregated.empty:
            continue
        out = _merge_source(out, aggregated, borrower_id)
    return out


def build_source_reconciliation(
    borrowers: pd.DataFrame,
    loaded: Mapping[str, LoadedTable],
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Report source-key coverage and cardinality without changing merge behavior."""
    borrower_id = scenario["borrower"]["borrower_id_field"]
    balance_field = scenario["borrower"]["balance_field"]
    source_specs = scenario.get("inputs", {}).get("sources", {})
    identity_table = loaded.get("identity")
    rows: List[Dict[str, Any]] = []

    for loaded_name, table in loaded.items():
        for issue in table.coercion_issues:
            record_exception(
                exceptions,
                "WARNING",
                "input",
                "INPUT_VALUE_COERCED_TO_MISSING",
                "Nonblank input values could not be parsed and were coerced to missing.",
                source=loaded_name,
                field=issue["field"],
                details=f"kind={issue['kind']}; count={issue['count']}; path={issue.get('path', '')}",
            )

    borrower_keys = set(borrowers[borrower_id].dropna().astype(str))
    for name in sorted(source_specs):
        spec = source_specs[name]
        table = loaded.get(name)
        if table is None:
            continue
        frame = table.frame
        merge_enabled = spec.get("merge", True) is not False
        key = spec.get("key", borrower_id)
        identity_key = spec.get("identity_key")
        base_row: Dict[str, Any] = {
            "source": name,
            "path": ";".join(str(path) for path in table.paths),
            "file_count": len(table.paths),
            "merge_enabled": merge_enabled,
            "key_field": key,
            "identity_key_field": identity_key or "",
            "source_row_count": int(len(frame)),
            "nonnull_key_row_count": np.nan,
            "null_key_row_count": np.nan,
            "unique_key_count": np.nan,
            "duplicate_key_row_count": np.nan,
            "expected_key_cardinality": "one" if spec.get("expect_unique_key", False) else "many",
            "matched_source_key_count": np.nan,
            "orphan_source_key_count": np.nan,
            "orphan_source_row_count": np.nan,
            "borrower_count": int(len(borrowers)),
            "matched_borrower_count": np.nan,
            "unmatched_borrower_count": np.nan,
            "matched_borrower_balance": np.nan,
            "unmatched_borrower_balance": np.nan,
            "entity_value_conflict_count": 0,
            "coercion_issue_count": int(sum(issue["count"] for issue in table.coercion_issues)),
        }
        if not merge_enabled:
            rows.append(base_row)
            continue
        if key not in frame.columns:
            record_exception(
                exceptions,
                "ERROR",
                "source_reconciliation",
                "SOURCE_KEY_FIELD_MISSING",
                "Merge-enabled source did not contain its configured key field.",
                source=name,
                field=key,
            )
            rows.append(base_row)
            continue

        null_key = frame[key].isna() | frame[key].astype(str).str.strip().eq("")
        nonnull = frame.loc[~null_key].copy()
        source_key_text = nonnull[key].astype(str)
        duplicate = source_key_text.duplicated(keep=False)
        source_keys = set(source_key_text)
        if identity_key:
            if identity_table is None:
                raise ValueError(f"Source '{name}' requires the identity input for account linkage.")
            linked = _link_source_to_identity(
                name,
                frame,
                spec,
                identity_table.frame,
                borrower_id,
            )
            matched_linked = linked.loc[linked[borrower_id].notna()].copy()
            matched_source_keys = set(matched_linked[key].dropna().astype(str))
            orphan_keys = source_keys - matched_source_keys
            matched_borrower_ids = set(matched_linked[borrower_id].dropna().astype(str))
            orphan_linked = linked.loc[
                linked[borrower_id].isna()
                & linked[key].notna()
                & linked[key].astype(str).str.strip().ne("")
            ].copy()
            conflicts = _source_entity_value_conflicts(matched_linked, borrower_id, spec)
            conflicts += _source_entity_value_conflicts(orphan_linked, key, spec)
        else:
            orphan_keys = source_keys - borrower_keys
            matched_borrower_ids = source_keys & borrower_keys
            conflicts = _source_entity_value_conflicts(frame, key, spec)
        orphan_rows = int(source_key_text.isin(orphan_keys).sum())
        matched_borrower = borrowers[borrower_id].astype(str).isin(matched_borrower_ids)
        base_row.update(
            {
                "nonnull_key_row_count": int((~null_key).sum()),
                "null_key_row_count": int(null_key.sum()),
                "unique_key_count": int(len(source_keys)),
                "duplicate_key_row_count": int(duplicate.sum()),
                "matched_source_key_count": int(len(source_keys - orphan_keys)),
                "orphan_source_key_count": int(len(orphan_keys)),
                "orphan_source_row_count": orphan_rows,
                "matched_borrower_count": int(matched_borrower.sum()),
                "unmatched_borrower_count": int((~matched_borrower).sum()),
                "matched_borrower_balance": float(
                    pd.to_numeric(borrowers.loc[matched_borrower, balance_field], errors="coerce").sum()
                ),
                "unmatched_borrower_balance": float(
                    pd.to_numeric(borrowers.loc[~matched_borrower, balance_field], errors="coerce").sum()
                ),
                "entity_value_conflict_count": conflicts,
            }
        )
        if null_key.any():
            record_exception(
                exceptions,
                "WARNING",
                "source_reconciliation",
                "SOURCE_NULL_KEYS",
                "Source rows with null merge keys could not enrich any borrower.",
                source=name,
                field=key,
                details=f"null_key_row_count={int(null_key.sum())}",
            )
        if orphan_keys:
            record_exception(
                exceptions,
                "WARNING",
                "source_reconciliation",
                "SOURCE_ORPHAN_KEYS",
                "Source keys did not match the configured borrower or identity key.",
                source=name,
                field=key,
                details=(
                    f"orphan_key_count={len(orphan_keys)}; orphan_row_count={orphan_rows}; "
                    f"identity_key={identity_key or borrower_id}"
                ),
            )
        if spec.get("expect_unique_key", False) and duplicate.any():
            record_exception(
                exceptions,
                "WARNING",
                "source_reconciliation",
                "SOURCE_KEY_CARDINALITY_VIOLATION",
                "Source expected one row per key but contained repeated keys; configured aggregation still determined the merged value.",
                source=name,
                field=key,
                details=f"duplicate_key_row_count={int(duplicate.sum())}",
            )
        if conflicts:
            record_exception(
                exceptions,
                "WARNING",
                "source_reconciliation",
                "SOURCE_ENTITY_VALUE_CONFLICT",
                "Repeated borrower/entity keys contained conflicting candidate values; deterministic row/date precedence selected the retained values.",
                source=name,
                field=key,
                details=f"conflict_group_count={conflicts}",
            )
        rows.append(base_row)
    return pd.DataFrame(rows)


def record_identity_data_issues(
    identity: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
) -> None:
    """Log raw identity balance and reserve issues before borrower aggregation."""
    borrower = scenario.get("borrower", {})
    balance_field = borrower.get("balance_field", "outstanding_balance")
    reserve_field = scenario.get("cecl", {}).get("reserve_field", "cecl_reserve")
    reserve_fields = reserve_basis_fields(scenario)
    if balance_field not in identity.columns:
        record_exception(
            exceptions,
            "ERROR",
            "identity",
            "IDENTITY_BALANCE_FIELD_MISSING",
            "Configured balance field was missing from identity data.",
            field=balance_field,
        )
    else:
        balances = pd.to_numeric(identity[balance_field], errors="coerce")
        invalid_balances = ~np.isfinite(balances)
        if invalid_balances.any():
            record_exception(
                exceptions,
                "WARNING",
                "identity",
                "IDENTITY_BALANCE_MISSING",
                "Identity rows had missing, invalid, or nonfinite balances and remain visible with unavailable dollar calculations.",
                field=balance_field,
                details=f"missing_count={int(invalid_balances.sum())}",
            )
        if (balances < 0).any():
            record_exception(
                exceptions,
                "WARNING",
                "identity",
                "IDENTITY_BALANCE_NEGATIVE",
                "Identity rows contained negative balances; values were retained for review.",
                field=balance_field,
                details=f"negative_count={int((balances < 0).sum())}",
            )
    for field in reserve_fields:
        is_current = field == reserve_field
        if field not in identity.columns:
            record_exception(
                exceptions,
                "ERROR",
                "cecl",
                (
                    "CECL_RESERVE_FIELD_MISSING"
                    if is_current
                    else "CECL_BASIS_FIELD_MISSING"
                ),
                "A configured CECL reserve-basis field was missing from identity data.",
                field=field,
            )
            continue
        reserve_values = pd.to_numeric(identity[field], errors="coerce")
        missing_reserve = ~np.isfinite(reserve_values)
        if missing_reserve.any():
            record_exception(
                exceptions,
                "WARNING",
                "cecl",
                (
                    "CECL_LOAN_RESERVE_MISSING_TREATED_AS_ZERO"
                    if is_current
                    else "CECL_HISTORY_RESERVE_MISSING_TREATED_AS_ZERO"
                ),
                "Individual loan CECL reserve values were missing and treated as zero in the selected reserve basis.",
                field=field,
                details=f"missing_count={int(missing_reserve.sum())}",
            )


def record_best_available_fallbacks(
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
) -> None:
    """Log when an ordered best-available chain selects a lower-priority field."""
    borrower_id = scenario["borrower"]["borrower_id_field"]
    for source_name, source_spec in scenario.get("inputs", {}).get("sources", {}).items():
        if source_spec.get("merge", True) is False:
            continue
        for output_field, method_spec in source_spec.get("aggregation", {}).items():
            if not isinstance(method_spec, Mapping) or method_spec.get("method") not in {
                "best_available",
                "best_available_unique_sum",
            }:
                continue
            candidates = _normalize_best_available_candidates(method_spec, output_field)
            if not candidates:
                continue
            audit_field = method_spec.get("candidate_output") or method_spec.get("source_field_output")
            if not audit_field or audit_field not in borrowers.columns:
                continue
            first_value = (
                candidates[0].get("label", candidates[0]["field"])
                if method_spec.get("candidate_output")
                else candidates[0]["field"]
            )
            for _, row in borrowers[borrowers[audit_field].notna()].iterrows():
                selected = str(row[audit_field])
                selected_parts = {part for part in selected.split(";") if part}
                if selected_parts and selected_parts == {str(first_value)}:
                    continue
                record_exception(
                    exceptions,
                    "INFO",
                    "enrichment",
                    "BEST_AVAILABLE_FALLBACK_USED",
                    "A lower-priority candidate supplied a best-available field because the preferred candidate was unusable.",
                    borrower_id=row.get(borrower_id),
                    field=output_field,
                    source=source_name,
                    details=f"preferred={first_value}; selected={selected}",
                )


def _source_entity_value_conflicts(frame: pd.DataFrame, key: str, spec: Mapping[str, Any]) -> int:
    """Count repeated entity keys whose configured candidate values disagree."""
    conflict_groups: set[tuple[Any, ...]] = set()
    for output_field, method_spec in spec.get("aggregation", {}).items():
        if not isinstance(method_spec, Mapping):
            continue
        method = method_spec.get("method")
        if method not in {"unique_sum", "best_available", "best_available_unique_sum"}:
            continue
        unique_fields = (
            [str(field) for field in as_list(method_spec.get("unique_fields"))]
            if method != "best_available"
            else []
        )
        if any(field not in frame.columns for field in unique_fields):
            continue
        candidate_specs: List[tuple[str, str | None]] = []
        if method == "unique_sum":
            candidate_specs = [(method_spec.get("field", output_field), None)]
        else:
            raw_candidates = as_list(method_spec.get("candidates"))
            if not raw_candidates:
                raw_candidates = [
                    {
                        "field": method_spec.get("field", output_field),
                        "date_field": method_spec.get("date_field"),
                    }
                ]
            for candidate in raw_candidates:
                if isinstance(candidate, str):
                    candidate_specs.append((candidate, None))
                elif isinstance(candidate, Mapping):
                    candidate_specs.append(
                        (
                            candidate.get("field"),
                            candidate.get("date_field"),
                        )
                    )
        for value_field, date_field in candidate_specs:
            if value_field not in frame.columns:
                continue
            group_fields = [key, *unique_fields]
            # Different values on different dates are expected history. A
            # conflict exists only when the same entity/date disagrees.
            if date_field and date_field in frame.columns:
                group_fields.append(date_field)
            for group_key, group in frame.groupby(group_fields, dropna=False, sort=True):
                if len(group) > 1 and group[value_field].dropna().astype(str).nunique() > 1:
                    conflict_groups.add(group_key if isinstance(group_key, tuple) else (group_key,))
    return len(conflict_groups)


def _link_source_to_identity(
    name: str,
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    identity: pd.DataFrame,
    borrower_id: str,
) -> pd.DataFrame:
    """Attach borrower IDs to account-keyed source rows through identity data."""
    source_key = spec.get("key", borrower_id)
    identity_key = spec.get("identity_key")
    if not identity_key:
        return frame.copy()
    if source_key == borrower_id:
        raise ValueError(
            f"Source '{name}' identity_key requires an account-level key other than '{borrower_id}'."
        )
    ensure_columns(frame, [source_key], f"Source '{name}'")
    ensure_columns(identity, [identity_key, borrower_id], "Identity input")
    if borrower_id in frame.columns:
        raise ValueError(
            f"Source '{name}' uses identity_key and must not also supply '{borrower_id}'."
        )

    crosswalk = identity[[identity_key, borrower_id]].copy()
    valid_identity_key = (
        crosswalk[identity_key].notna()
        & crosswalk[identity_key].astype(str).str.strip().ne("")
    )
    valid_borrower = (
        crosswalk[borrower_id].notna()
        & crosswalk[borrower_id].astype(str).str.strip().ne("")
    )
    crosswalk = crosswalk.loc[valid_identity_key & valid_borrower].drop_duplicates()
    borrower_counts = crosswalk.groupby(identity_key, dropna=False)[borrower_id].nunique()
    ambiguous = borrower_counts[borrower_counts > 1]
    if not ambiguous.empty:
        examples = ", ".join(str(value) for value in ambiguous.index[:5])
        raise ValueError(
            f"Identity key '{identity_key}' maps to multiple borrowers for source '{name}': {examples}"
        )
    crosswalk = crosswalk.drop_duplicates(identity_key, keep="first")
    if source_key != identity_key:
        crosswalk = crosswalk.rename(columns={identity_key: source_key})
    return frame.merge(crosswalk, on=source_key, how="left", validate="many_to_one", sort=False)


def aggregate_source(name: str, df: pd.DataFrame, spec: Mapping[str, Any], key: str) -> pd.DataFrame:
    """Aggregate one source table before merging it into borrowers.

    Called by `enrich_borrowers`. Account-keyed sources are linked to identity
    first, so every source is reduced to one row per borrower before merging.
    """
    if key not in df.columns:
        raise ValueError(f"Source '{name}' key column '{key}' is missing.")
    frame = df.copy()
    aggregation = spec.get("aggregation", {})
    if not aggregation:
        source_key = spec.get("key", key)
        aggregation = {
            column: "first"
            for column in frame.columns
            if column not in {key, source_key} and not str(column).startswith("_source_")
        }

    pieces: List[pd.DataFrame] = []
    for output_field, method_spec in aggregation.items():
        piece = _aggregate_field(frame, key, output_field, method_spec)
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=[key])
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=key, how="outer")
    return result.sort_values([key], kind="mergesort").reset_index(drop=True)


def _aggregate_field(df: pd.DataFrame, key: str, output_field: str, method_spec: Any) -> pd.DataFrame:
    """Aggregate one configured field from a source table.

    Called by `aggregate_source`. Special methods:
    - `latest` sorts by date and source row for deterministic tie-breaking.
    - `best_available` chooses across ordered candidate fields, skipping
      blanks and optionally zeros before using date/order tie-breaks.
    - `unique_sum` sums after dropping duplicate collateral/appraisal keys.
    """
    if isinstance(method_spec, str):
        method = method_spec
        field = output_field
        options: Mapping[str, Any] = {}
    else:
        options = method_spec
        method = options.get("method", "first")
        field = options.get("field", output_field)

    if method == "latest":
        date_field = options.get("date_field")
        if not date_field:
            raise ValueError(f"Latest aggregation for '{output_field}' requires date_field.")
        required = list(dict.fromkeys([key, field, date_field]))
        ensure_columns(df, required, f"Latest aggregation for '{output_field}'")
        work = df[required + (["_source_row"] if "_source_row" in df.columns else [])].copy()
        work[date_field] = parse_date_series(work[date_field])
        sort_cols = [key, date_field]
        if "_source_row" in work.columns:
            sort_cols.append("_source_row")
        # Mergesort preserves source order for equal dates, making latest-row
        # ties deterministic and auditable.
        work = work.sort_values(sort_cols, kind="mergesort")
        values = work.dropna(subset=[field]).groupby(key, dropna=False).tail(1)
        return values[[key, field]].rename(columns={field: output_field})

    if method == "best_available":
        return _best_available_field(df, key, output_field, options)

    if method == "best_available_unique_sum":
        return _best_available_unique_sum(df, key, output_field, options)

    if method == "unique_sum":
        unique_fields = as_list(options.get("unique_fields", [field]))
        required = [key, field] + [col for col in unique_fields if col != field]
        ensure_columns(df, required, f"Unique sum aggregation for '{output_field}'")
        # Appraisal/collateral amounts are summed only once per configured
        # unique collateral key, preventing duplicated collateral rows from
        # inflating borrower value.
        work = df[required].drop_duplicates([key] + unique_fields)
        work[field] = pd.to_numeric(work[field], errors="coerce")
        return work.groupby(key, dropna=False)[field].sum(min_count=1).reset_index(name=output_field)

    ensure_columns(df, [key, field], f"Aggregation for '{output_field}'")
    func = _aggregation_callable(method)
    return df.groupby(key, dropna=False)[field].agg(func).reset_index(name=output_field)


def _best_available_field(
    df: pd.DataFrame,
    key: str,
    output_field: str,
    options: Mapping[str, Any],
) -> pd.DataFrame:
    """Select the best nonblank candidate value for each borrower key.

    Called by `_aggregate_field` for reusable fields such as DSCR, FICO, and
    appraised value. `selection="latest"` uses the newest available candidate
    date first, then candidate order; `selection="precedence"` uses candidate
    order first, then the newest row inside that candidate.
    """
    candidates = _normalize_best_available_candidates(options, output_field)
    required = [key]
    for candidate in candidates:
        required.append(candidate["field"])
        if candidate.get("date_field"):
            required.append(candidate["date_field"])
    ensure_columns(df, list(dict.fromkeys(required)), f"Best available aggregation for '{output_field}'")

    treat_zero_as_missing = bool(options.get("treat_zero_as_missing", True))
    parts: List[pd.DataFrame] = []
    for order, candidate in enumerate(candidates):
        field = candidate["field"]
        date_field = candidate.get("date_field")
        columns = [key, field]
        if date_field:
            columns.append(date_field)
        columns.extend(_source_audit_columns(df))
        part = df[list(dict.fromkeys(columns))].copy()
        part["_candidate_order"] = order
        part["_candidate_field"] = field
        part["_candidate_label"] = candidate.get("label", field)
        part["_candidate_value"] = part[field]
        if date_field:
            part["_candidate_date"] = parse_date_series(part[date_field])
            part["_selected_date"] = part["_candidate_date"]
        else:
            part["_candidate_date"] = pd.Timestamp.min
            part["_selected_date"] = pd.NaT
        if "_source_row" not in part.columns:
            part["_source_row"] = 0
        part["_source_record"] = _source_record_series(part)
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=[key, output_field])

    work = pd.concat(parts, ignore_index=True, sort=False)
    valid = _valid_best_available_values(work["_candidate_value"], treat_zero_as_missing)
    work = work.loc[valid].copy()
    if work.empty:
        return pd.DataFrame(columns=_best_available_output_columns(key, output_field, options))

    work["_candidate_date"] = work["_candidate_date"].fillna(pd.Timestamp.min)
    if options.get("selection", "latest") == "precedence":
        sort_cols = [key, "_candidate_order", "_candidate_date", "_source_row"]
        ascending = [True, True, False, False]
    else:
        sort_cols = [key, "_candidate_date", "_candidate_order", "_source_row"]
        ascending = [True, False, True, False]
    chosen = work.sort_values(sort_cols, ascending=ascending, kind="mergesort").groupby(key, dropna=False).head(1)
    chosen = chosen.reset_index(drop=True)

    result = chosen[[key, "_candidate_value"]].rename(columns={"_candidate_value": output_field})
    date_output = options.get("date_output")
    if date_output:
        result[date_output] = chosen["_selected_date"]
    source_field_output = options.get("source_field_output")
    if source_field_output:
        result[source_field_output] = chosen["_candidate_field"]
    candidate_output = options.get("candidate_output")
    if candidate_output:
        result[candidate_output] = chosen["_candidate_label"]
    source_record_output = options.get("source_record_output")
    if source_record_output:
        result[source_record_output] = chosen["_source_record"]
    return result


def _best_available_unique_sum(
    df: pd.DataFrame,
    key: str,
    output_field: str,
    options: Mapping[str, Any],
) -> pd.DataFrame:
    """Select one best value per collateral/entity and sum those values by borrower."""
    candidates = _normalize_best_available_candidates(options, output_field)
    unique_fields = [str(field) for field in as_list(options.get("unique_fields"))]
    if not unique_fields:
        raise ValueError(f"Best available unique sum for '{output_field}' requires unique_fields.")
    required = [key, *unique_fields]
    for candidate in candidates:
        required.append(candidate["field"])
        if candidate.get("date_field"):
            required.append(candidate["date_field"])
    ensure_columns(df, list(dict.fromkeys(required)), f"Best available unique sum for '{output_field}'")

    parts: List[pd.DataFrame] = []
    for order, candidate in enumerate(candidates):
        field = candidate["field"]
        date_field = candidate.get("date_field")
        columns = [key, *unique_fields, field]
        if date_field:
            columns.append(date_field)
        columns.extend(_source_audit_columns(df))
        part = df[list(dict.fromkeys(columns))].copy()
        part["_candidate_order"] = order
        part["_candidate_field"] = field
        part["_candidate_label"] = candidate.get("label", field)
        part["_candidate_value"] = part[field]
        if date_field:
            part["_candidate_date"] = parse_date_series(part[date_field])
            part["_selected_date"] = part["_candidate_date"]
        else:
            part["_candidate_date"] = pd.Timestamp.min
            part["_selected_date"] = pd.NaT
        if "_source_row" not in part.columns:
            part["_source_row"] = 0
        part["_source_record"] = _source_record_series(part)
        parts.append(part)

    work = pd.concat(parts, ignore_index=True, sort=False)
    work = work.loc[
        _valid_best_available_values(
            work["_candidate_value"], bool(options.get("treat_zero_as_missing", True))
        )
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=_best_available_output_columns(key, output_field, options))
    work["_candidate_value"] = pd.to_numeric(work["_candidate_value"], errors="coerce")
    work = work[work["_candidate_value"].notna()].copy()
    work["_candidate_date"] = work["_candidate_date"].fillna(pd.Timestamp.min)
    group_fields = [key, *unique_fields]
    if options.get("selection", "precedence") == "latest":
        sort_cols = [*group_fields, "_candidate_date", "_candidate_order", "_source_row"]
        ascending = [*[True] * len(group_fields), False, True, False]
    else:
        sort_cols = [*group_fields, "_candidate_order", "_candidate_date", "_source_row"]
        ascending = [*[True] * len(group_fields), True, False, False]
    chosen = work.sort_values(sort_cols, ascending=ascending, kind="mergesort").groupby(
        group_fields, dropna=False, sort=True
    ).head(1)
    result = chosen.groupby(key, dropna=False)["_candidate_value"].sum(min_count=1).reset_index(name=output_field)
    date_output = options.get("date_output")
    if date_output:
        dates = chosen.groupby(key, dropna=False)["_selected_date"].max().reset_index(name=date_output)
        result = result.merge(dates, on=key, how="left")
    source_field_output = options.get("source_field_output")
    if source_field_output:
        sources = chosen.groupby(key, dropna=False)["_candidate_field"].agg(join_unique).reset_index(name=source_field_output)
        result = result.merge(sources, on=key, how="left")
    candidate_output = options.get("candidate_output")
    if candidate_output:
        labels = chosen.groupby(key, dropna=False)["_candidate_label"].agg(join_unique).reset_index(name=candidate_output)
        result = result.merge(labels, on=key, how="left")
    source_record_output = options.get("source_record_output")
    if source_record_output:
        records = chosen.groupby(key, dropna=False)["_source_record"].agg(join_unique).reset_index(
            name=source_record_output
        )
        result = result.merge(records, on=key, how="left")
    return result


def _normalize_best_available_candidates(options: Mapping[str, Any], output_field: str) -> List[Dict[str, Any]]:
    """Normalize string/dict candidate specs while preserving JSON order."""
    raw_candidates = as_list(options.get("candidates"))
    if not raw_candidates:
        raw_candidates = [{"field": options.get("field", output_field), "date_field": options.get("date_field")}]

    candidates: List[Dict[str, Any]] = []
    for candidate in raw_candidates:
        if isinstance(candidate, str):
            candidates.append({"field": candidate})
            continue
        field = candidate.get("field")
        if not field:
            raise ValueError(f"Best available aggregation for '{output_field}' includes a candidate without a field.")
        normalized = dict(candidate)
        normalized["field"] = field
        candidates.append(normalized)
    return candidates


def _valid_best_available_values(values: pd.Series, treat_zero_as_missing: bool) -> pd.Series:
    """Return rows whose candidate values can be used in a fallback chain."""
    valid = values.notna()
    text = values.astype(str).str.strip()
    valid &= text.ne("") & text.str.lower().ne("nan")
    if treat_zero_as_missing:
        numeric = pd.to_numeric(values, errors="coerce")
        valid &= numeric.ne(0) | numeric.isna()
    return valid


def _best_available_output_columns(key: str, output_field: str, options: Mapping[str, Any]) -> List[str]:
    """Return the empty-frame columns emitted by `best_available`."""
    columns = [key, output_field]
    for option in ("date_output", "source_field_output", "candidate_output", "source_record_output"):
        value = options.get(option)
        if value:
            columns.append(value)
    return columns


def _source_audit_columns(frame: pd.DataFrame) -> List[str]:
    """Return internal provenance columns available on a loaded frame."""
    return [
        column
        for column in ("_source_row", "_source_file", "_source_file_row")
        if column in frame.columns
    ]


def _source_record_series(frame: pd.DataFrame) -> pd.Series:
    """Format a physical file and row reference for selected source values."""
    if "_source_file" in frame.columns and "_source_file_row" in frame.columns:
        row_text = frame["_source_file_row"].apply(
            lambda value: str(int(value)) if pd.notna(value) else ""
        )
        return frame["_source_file"].fillna("").astype(str) + "#row=" + row_text
    if "_source_row" in frame.columns:
        row_text = frame["_source_row"].apply(
            lambda value: str(int(value)) if pd.notna(value) else ""
        )
        return "row=" + row_text
    return pd.Series("", index=frame.index, dtype=object)


def _aggregation_callable(method: Any) -> Any:
    """Map scenario aggregation names to pandas groupby callables."""
    method = str(method)
    if method == "first":
        return first_non_null
    if method == "last":
        return lambda values: values.dropna().iloc[-1] if not values.dropna().empty else np.nan
    if method == "sum":
        return lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)
    if method == "mean":
        return lambda values: pd.to_numeric(values, errors="coerce").mean()
    if method == "max":
        return _max_value
    if method == "min":
        return _min_value
    if method == "list_unique":
        return join_unique
    raise ValueError(f"Unsupported aggregation method: {method}")


def _max_value(values: pd.Series) -> Any:
    """Return max while preserving date values when possible."""
    clean = values.dropna()
    if clean.empty:
        return np.nan
    if pd.api.types.is_datetime64_any_dtype(clean):
        return clean.max()
    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.notna().all():
        return numeric.max()
    dates = pd.to_datetime(clean, errors="coerce")
    if dates.notna().all():
        return dates.max()
    if numeric.notna().any():
        return numeric.max()
    return clean.max()


def _min_value(values: pd.Series) -> Any:
    """Return min while preserving date values when possible."""
    clean = values.dropna()
    if clean.empty:
        return np.nan
    if pd.api.types.is_datetime64_any_dtype(clean):
        return clean.min()
    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.notna().all():
        return numeric.min()
    dates = pd.to_datetime(clean, errors="coerce")
    if dates.notna().all():
        return dates.min()
    if numeric.notna().any():
        return numeric.min()
    return clean.min()


def _merge_source(
    borrowers: pd.DataFrame,
    source: pd.DataFrame,
    borrower_id: str,
) -> pd.DataFrame:
    """Merge one borrower-level source and reject ambiguous field collisions."""
    conflicts = sorted((set(borrowers.columns) & set(source.columns)) - {borrower_id})
    if conflicts:
        raise ValueError(f"Enrichment columns already exist on borrowers: {', '.join(conflicts)}")
    return borrowers.merge(source, on=borrower_id, how="left", validate="one_to_one")
