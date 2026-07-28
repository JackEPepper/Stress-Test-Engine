"""Loan-level targeted external-shock selection and variant execution."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .borrower import aggregate_source
from .exceptions import record_exception
from .modules.base import initialize_results, targeted_override_column
from .modules.ci import _brg_key, _ebitda_reduction, run_ci
from .modules.consumer import run_consumer
from .modules.cre import run_cre
from .reporting import build_reports
from .tagging import apply_tags, assign_primary_modules, evaluate_conditions
from .utils import (
    as_list,
    condition_fields,
    get_levels,
    is_missing,
    lookup_parameter_with_source,
    to_number,
)


ALLOWED_PARAMETERS: Dict[str, set[str]] = {
    "C&I": {"ebitda_reduction", "interest_rate_stress"},
    "CRE": {
        "dscr_decline",
        "refinance_noi_decline",
        "treasury_rate",
        "credit_spread",
        "amortization_years",
        "cap_rate",
    },
    "Consumer": {
        "pd_increase_factor",
        "collateral_value_factor",
        "rushed_sale_discount",
        "closing_costs",
    },
}
OPERATIONS = {"replace", "add", "multiply"}
UNMATCHED_BEHAVIORS = {"base", "baseline_stress"}


def targeted_enabled(scenario: Mapping[str, Any]) -> bool:
    """Return whether the scenario opts into targeted loan-level execution."""
    config = scenario.get("targeted_stress", {})
    return isinstance(config, Mapping) and bool(config) and config.get("enabled", True)


def validate_targeted_config(scenario: Mapping[str, Any]) -> None:
    """Validate targeted-stress structure without loading external data."""
    if not targeted_enabled(scenario):
        return
    config = scenario["targeted_stress"]
    shocks = config.get("shocks", {})
    variants = config.get("variants", {})
    if not isinstance(shocks, Mapping) or not shocks:
        raise ValueError("targeted_stress.shocks must be a nonempty JSON object.")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("targeted_stress.variants must be a nonempty JSON object.")

    for shock_name, shock in shocks.items():
        if not isinstance(shock, Mapping):
            raise ValueError(f"Targeted shock '{shock_name}' must be a JSON object.")
        selector = shock.get("selector", shock.get("include"))
        if not selector:
            raise ValueError(f"Targeted shock '{shock_name}' must define selector or include.")
        _validate_selector(selector, f"targeted_stress.shocks.{shock_name}.selector")
        if shock.get("exclude"):
            _validate_selector(shock["exclude"], f"targeted_stress.shocks.{shock_name}.exclude")
        tiers = shock.get("tiers", {})
        if not isinstance(tiers, Mapping) or not tiers:
            raise ValueError(f"Targeted shock '{shock_name}' must define at least one tier.")
        default_tier = shock.get("default_tier")
        if default_tier is not None and str(default_tier) not in tiers:
            raise ValueError(
                f"Targeted shock '{shock_name}' default_tier '{default_tier}' is not defined."
            )
        for index, rule in enumerate(as_list(shock.get("tier_rules"))):
            if not isinstance(rule, Mapping) or not rule.get("tier"):
                raise ValueError(
                    f"Targeted shock '{shock_name}' tier_rules[{index}] requires tier."
                )
            if str(rule["tier"]) not in tiers:
                raise ValueError(
                    f"Targeted shock '{shock_name}' tier_rules[{index}] references "
                    f"unknown tier '{rule['tier']}'."
                )
            rule_selector = rule.get("selector", rule.get("include"))
            if not rule_selector:
                raise ValueError(
                    f"Targeted shock '{shock_name}' tier_rules[{index}] requires selector or include."
                )
            _validate_selector(
                rule_selector,
                f"targeted_stress.shocks.{shock_name}.tier_rules[{index}]",
            )
        for tier_name, tier in tiers.items():
            if not isinstance(tier, Mapping):
                raise ValueError(
                    f"Targeted shock '{shock_name}' tier '{tier_name}' must be a JSON object."
                )
            modules = tier.get("modules", tier)
            if not isinstance(modules, Mapping) or not modules:
                raise ValueError(
                    f"Targeted shock '{shock_name}' tier '{tier_name}' modules must be a nonempty object."
                )
            for module, parameters in modules.items():
                if module not in ALLOWED_PARAMETERS:
                    raise ValueError(
                        f"Targeted shock '{shock_name}' references unsupported module '{module}'."
                    )
                if not isinstance(parameters, Mapping) or not parameters:
                    raise ValueError(
                        f"Targeted shock '{shock_name}' {module} tier parameters must be a nonempty object."
                    )
                for parameter, spec in parameters.items():
                    if parameter not in ALLOWED_PARAMETERS[module]:
                        raise ValueError(
                            f"Unsupported targeted parameter '{module}.{parameter}'."
                        )
                    operation, values = _operation_spec(spec)
                    if operation not in OPERATIONS:
                        raise ValueError(
                            f"Unsupported targeted operation '{operation}' for {module}.{parameter}."
                        )
                    if values is None:
                        raise ValueError(
                            f"Targeted parameter '{module}.{parameter}' must define value or values."
                        )
                    _validate_assumption_values(
                        values,
                        get_levels(scenario),
                        (
                            f"targeted_stress.shocks.{shock_name}.tiers."
                            f"{tier_name}.{module}.{parameter}"
                        ),
                    )

    for variant_name, variant in variants.items():
        if str(variant_name).strip().casefold() == "baseline":
            raise ValueError(
                "Targeted variant name 'baseline' is reserved for the engine-generated baseline "
                "variant (case-insensitive)."
            )
        if not isinstance(variant, Mapping):
            raise ValueError(f"Targeted variant '{variant_name}' must be a JSON object.")
        behavior = str(variant.get("unmatched_behavior", "baseline_stress"))
        if behavior not in UNMATCHED_BEHAVIORS:
            raise ValueError(
                f"Targeted variant '{variant_name}' has unsupported unmatched_behavior '{behavior}'."
            )
        names = [str(item) for item in as_list(variant.get("shocks"))]
        if not names:
            raise ValueError(f"Targeted variant '{variant_name}' must list at least one shock.")
        unknown = [name for name in names if name not in shocks]
        if unknown:
            raise ValueError(
                f"Targeted variant '{variant_name}' references unknown shocks: {', '.join(unknown)}."
            )

    primary = str(config.get("primary_variant", "baseline"))
    if primary != "baseline" and primary not in variants:
        raise ValueError(f"targeted_stress.primary_variant '{primary}' is not defined.")


def _validate_selector(selector: Any, path: str) -> None:
    if isinstance(selector, list):
        for index, item in enumerate(selector):
            _validate_selector(item, f"{path}[{index}]")
        return
    if not isinstance(selector, Mapping):
        raise ValueError(f"{path} must be a selector object or list.")
    if "all" in selector and "any" in selector:
        raise ValueError(f"{path} cannot define both all and any.")
    if "all" in selector or "any" in selector:
        key = "all" if "all" in selector else "any"
        items = selector[key]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}.{key} must be a nonempty list.")
        for index, item in enumerate(items):
            _validate_selector(item, f"{path}.{key}[{index}]")
        return
    selector_type = str(selector.get("type", "condition"))
    if selector_type == "naics_prefix":
        if not selector.get("field"):
            raise ValueError(f"{path} naics_prefix requires field.")
        prefixes = as_list(selector.get("prefixes", selector.get("values")))
        if not prefixes:
            raise ValueError(f"{path} naics_prefix requires prefixes.")
        invalid = [value for value in prefixes if _normalize_naics(value) is None]
        if invalid:
            raise ValueError(
                f"{path} contains invalid NAICS prefixes; expected 2-6 digits: {invalid}."
            )
        return
    if selector_type == "external_list":
        required = ["source", "source_field", "exposure_field"]
        missing = [field for field in required if not selector.get(field)]
        if missing:
            raise ValueError(f"{path} external_list is missing: {', '.join(missing)}.")
        match = str(selector.get("match", "exact"))
        if match not in {"exact", "prefix"}:
            raise ValueError(f"{path} external_list match must be exact or prefix.")
        if selector.get("where") is not None:
            _validate_condition_block(selector["where"], f"{path}.where")
        return
    if selector_type != "condition":
        raise ValueError(f"{path} uses unsupported selector type '{selector_type}'.")
    _validate_atomic_condition(selector, path)


def _validate_condition_block(conditions: Any, path: str) -> None:
    """Validate the tag-condition grammar used by external-list filters."""
    if isinstance(conditions, list):
        if not conditions:
            raise ValueError(f"{path} must be a nonempty condition list.")
        for index, condition in enumerate(conditions):
            _validate_condition_block(condition, f"{path}[{index}]")
        return
    if not isinstance(conditions, Mapping):
        raise ValueError(f"{path} must be a condition object or list.")
    if "all" in conditions and "any" in conditions:
        raise ValueError(f"{path} cannot define both all and any.")
    if "all" in conditions or "any" in conditions:
        key = "all" if "all" in conditions else "any"
        items = conditions[key]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}.{key} must be a nonempty list.")
        for index, condition in enumerate(items):
            _validate_condition_block(condition, f"{path}.{key}[{index}]")
        return
    _validate_atomic_condition(conditions, path)


def _validate_atomic_condition(condition: Mapping[str, Any], path: str) -> None:
    """Reject structurally incomplete conditions before they can become no-ops."""
    if not condition.get("field"):
        raise ValueError(f"{path} condition requires field.")
    operation = str(condition.get("op", "eq")).lower()
    supported = {
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
    if operation not in supported:
        raise ValueError(f"{path} uses unsupported condition operator '{operation}'.")
    if operation not in {"is_null", "not_null"} and "value" not in condition:
        raise ValueError(f"{path} condition operator '{operation}' requires value.")
    value = condition.get("value")
    if (
        operation not in {"is_null", "not_null"}
        and (
            value is None
            or (
                not isinstance(value, (list, tuple, Mapping))
                and is_missing(value)
            )
        )
    ):
        raise ValueError(f"{path} condition operator '{operation}' requires a non-null value.")
    if operation in {"gt", "gte", "lt", "lte"}:
        _finite_targeted_number(value, f"{path}.value")
    if operation == "between":
        bounds = value
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{path} condition operator 'between' requires exactly two values.")
        for index, bound in enumerate(bounds):
            _finite_targeted_number(bound, f"{path}.value[{index}]")
    if operation in {
        "in",
        "not_in",
        "has_token",
        "has_any_token",
        "has_all_tokens",
    }:
        items = as_list(value)
        if not items:
            raise ValueError(f"{path} condition operator '{operation}' requires nonempty values.")
        if operation in {"in", "not_in"}:
            try:
                set(items)
            except TypeError:
                raise ValueError(
                    f"{path} condition operator '{operation}' requires scalar values."
                ) from None
    if operation == "regex":
        try:
            re.compile(str(value))
        except re.error as exc:
            raise ValueError(f"{path} condition regex is invalid: {exc}.") from None


def _validate_assumption_values(
    values: Any,
    levels: List[str],
    path: str,
) -> None:
    """Require complete level maps and finite numeric targeted shock values."""
    if isinstance(values, Mapping):
        expected = set(levels)
        fallback_keys = [key for key in ("default", "all", "*") if key in values]
        if len(fallback_keys) > 1:
            raise ValueError(
                f"{path} values must define at most one of: default, all, *."
            )
        allowed = expected | {"default", "all", "*"}
        unknown = [key for key in values if key not in allowed]
        if unknown:
            raise ValueError(
                f"{path} values contains unknown stress levels: "
                f"{', '.join(str(key) for key in unknown)}."
            )
        fallback = fallback_keys[0] if fallback_keys else None
        missing = [level for level in levels if level not in values and fallback is None]
        if missing:
            raise ValueError(
                f"{path} values is missing stress levels: {', '.join(missing)}."
            )
        for key, value in values.items():
            _finite_targeted_number(value, f"{path}.values.{key}")
        return
    _finite_targeted_number(values, f"{path}.value")


def _finite_targeted_number(value: Any, path: str) -> float:
    """Convert one targeted value and reject missing, nonnumeric, or nonfinite input."""
    number = to_number(value, np.nan)
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number):
        raise ValueError(f"{path} must be numeric and finite.")
    return number


def build_loan_context(
    scenario: Mapping[str, Any],
    loaded: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the canonical one-row-per-identity-loan modeling context."""
    borrower = scenario["borrower"]
    borrower_id = borrower["borrower_id_field"]
    loan_id = borrower.get("loan_id_field", "loan_id")
    loans = loaded["identity"].frame.copy()

    missing_borrower = loans[borrower_id].isna() | loans[borrower_id].astype(str).str.strip().eq("")
    for idx in loans.index[missing_borrower]:
        source_row = loans.at[idx, "_source_row"] if "_source_row" in loans.columns else idx + 1
        loans.at[idx, borrower_id] = f"__MISSING_BORROWER_ID_ROW_{source_row}"

    if loan_id not in loans.columns:
        loans[loan_id] = pd.NA
    text_ids = loans[loan_id].apply(_normalize_key)
    duplicate = text_ids.ne("") & text_ids.duplicated(keep=False)
    missing = text_ids.eq("")
    source_rows = loans.get("_source_row", pd.Series(np.arange(1, len(loans) + 1), index=loans.index))
    loans["_loan_id_ambiguous"] = duplicate | missing
    loans["_exposure_id"] = [
        value if value and not ambiguous else f"{value or 'missing_loan'}__row_{int(source_row)}"
        for value, ambiguous, source_row in zip(text_ids, loans["_loan_id_ambiguous"], source_rows)
    ]

    loans, tag_summary = apply_tags(loans, scenario, loaded, exceptions)
    if not tag_summary.empty:
        tag_summary["loan_count"] = tag_summary["borrower_count"]
        for idx, summary_row in tag_summary.iterrows():
            tag_column = summary_row.get("tag_column")
            if tag_column in loans.columns:
                selected = loans[loans[tag_column].fillna(False).astype(bool)]
                tag_summary.at[idx, "borrower_count"] = int(
                    selected[borrower_id].nunique(dropna=True)
                )
                tag_summary.at[idx, "loan_count"] = int(len(selected))
    loans = assign_primary_modules(loans, scenario, exceptions)
    loans = _enrich_loans(loans, scenario, loaded)
    return loans.sort_values(["_exposure_id"], kind="mergesort").reset_index(drop=True), tag_summary


def _enrich_loans(
    loans: pd.DataFrame,
    scenario: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> pd.DataFrame:
    """Merge borrower-keyed sources broadly and account-keyed sources directly."""
    out = loans.copy()
    borrower_id = scenario["borrower"]["borrower_id_field"]
    for name in sorted(scenario.get("inputs", {}).get("sources", {})):
        spec = scenario["inputs"]["sources"][name]
        if spec.get("merge", True) is False or name not in loaded:
            continue
        source = loaded[name].frame
        source_key = spec.get("key", borrower_id)
        exposure_key = spec.get("identity_key") or borrower_id
        aggregated = aggregate_source(name, source, spec, source_key)
        if source_key != exposure_key:
            aggregated = aggregated.rename(columns={source_key: exposure_key})
        conflicts = sorted((set(out.columns) & set(aggregated.columns)) - {exposure_key})
        if conflicts:
            raise ValueError(
                f"Loan enrichment columns already exist for source '{name}': {', '.join(conflicts)}"
            )
        if exposure_key not in out.columns:
            raise ValueError(
                f"Loan context is missing identity key '{exposure_key}' for source '{name}'."
            )
        out = out.merge(aggregated, on=exposure_key, how="left", validate="many_to_one", sort=False)
    return out


def run_targeted_stress(
    scenario: Mapping[str, Any],
    loaded: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run baseline plus configured loan-level targeted variants."""
    validate_targeted_config(scenario)
    context, tag_summary = build_loan_context(scenario, loaded, exceptions)
    baseline_scenario = _variant_scenario(scenario, "baseline")
    exception_start = len(exceptions)
    baseline, baseline_out_scope = _run_modules(
        initialize_results(context, baseline_scenario, exceptions),
        baseline_scenario,
        loaded,
        exceptions,
        return_out_of_scope=True,
    )
    _annotate_exceptions(
        exceptions, exception_start, "baseline", context, scenario
    )
    baseline["scenario_variant"] = "baseline"
    base_template = _base_template(baseline, baseline_scenario)

    variant_frames = [baseline]
    out_scope_frames: List[pd.DataFrame] = (
        [baseline_out_scope] if not baseline_out_scope.empty else []
    )
    selection_rows: List[Dict[str, Any]] = []
    assumption_rows: List[Dict[str, Any]] = []
    reports_by_name: Dict[str, List[pd.DataFrame]] = {}

    exception_start = len(exceptions)
    baseline_reports = build_reports(
        baseline.drop(columns=["scenario_variant"]),
        context,
        baseline_scenario,
        baseline_out_scope,
        exceptions,
    )
    _annotate_exceptions(
        exceptions, exception_start, "baseline", context, scenario
    )
    _collect_variant_reports(reports_by_name, baseline_reports, "baseline")

    config = scenario["targeted_stress"]
    for variant_name, variant_spec in config["variants"].items():
        variant_name = str(variant_name)
        variant_scenario = _variant_scenario(scenario, variant_name)
        exception_start = len(exceptions)
        resolved = _resolve_variant(
            context,
            scenario,
            loaded,
            variant_name,
            variant_spec,
            exceptions,
            selection_rows,
            assumption_rows,
        )
        _annotate_exceptions(
            exceptions, exception_start, variant_name, context, scenario
        )
        active = resolved["_targeted_active"].fillna(False).astype(bool)
        selection_error = resolved["_targeted_error"].fillna(False).astype(bool)
        behavior = str(variant_spec.get("unmatched_behavior", "baseline_stress"))
        variant = base_template.drop(columns=["scenario_variant"], errors="ignore").copy()
        if behavior == "baseline_stress":
            inactive = ~(active | selection_error)
            _copy_rows(variant, baseline.drop(columns=["scenario_variant"]), inactive)
        for column in resolved.columns:
            if column.startswith("_targeted_"):
                variant[column] = resolved[column]
        variant["_targeted_active"] = active
        exception_start = len(exceptions)
        variant, out_scope = _run_modules(
            variant,
            variant_scenario,
            loaded,
            exceptions,
            return_out_of_scope=True,
        )
        _annotate_exceptions(
            exceptions,
            exception_start,
            variant_name,
            context.loc[active],
            scenario,
        )
        variant["scenario_variant"] = variant_name
        selection_error_scope = _selection_error_scope(
            variant, selection_error, variant_scenario
        )
        if not selection_error_scope.empty:
            out_scope = pd.concat(
                [out_scope, selection_error_scope], ignore_index=True, sort=False
            )
        variant_frames.append(variant)
        if not out_scope.empty:
            out_scope_frames.append(out_scope)
        exception_start = len(exceptions)
        reports = build_reports(
            variant.drop(columns=["scenario_variant"]),
            context,
            variant_scenario,
            out_scope,
            exceptions,
        )
        _annotate_exceptions(
            exceptions, exception_start, variant_name, context, scenario
        )
        _collect_variant_reports(reports_by_name, reports, variant_name)

    variant_results = pd.concat(variant_frames, ignore_index=True, sort=False)
    reports = {
        name: pd.concat(frames, ignore_index=True, sort=False)
        for name, frames in reports_by_name.items()
    }
    selection_detail = pd.DataFrame(selection_rows)
    assumption_audit = pd.DataFrame(assumption_rows)
    reports["targeted_selection_detail"] = selection_detail
    reports["targeted_assumption_audit"] = assumption_audit
    reports["targeted_stress_summary"] = _selection_summary(selection_detail, scenario)
    reports["variant_comparison"] = _variant_comparison(reports)
    primary_name = str(config.get("primary_variant", "baseline"))
    primary = variant_results[variant_results["scenario_variant"] == primary_name].copy()
    return {
        "context": context,
        "results": primary.reset_index(drop=True),
        "variant_results": variant_results,
        "reports": reports,
        "tag_summary": tag_summary,
        "out_of_scope": (
            pd.concat(out_scope_frames, ignore_index=True, sort=False)
            if out_scope_frames
            else pd.DataFrame()
        ),
        "variant_names": ["baseline", *[str(name) for name in config["variants"]]],
        "primary_variant": primary_name,
    }


def _run_modules(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    loaded: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
    return_out_of_scope: bool = False,
) -> Any:
    out = results
    out_frames: List[pd.DataFrame] = []
    for module in scenario.get("module_order", ["CRE", "C&I", "Consumer"]):
        if module == "CRE":
            out, detail = run_cre(out, scenario, exceptions)
        elif module == "C&I":
            out, detail = run_ci(out, scenario, exceptions)
        elif module == "Consumer":
            out, detail = run_consumer(out, scenario, loaded, exceptions)
        else:
            raise ValueError(f"Unsupported module in module_order: {module}")
        if detail is not None and not detail.empty:
            out_frames.append(detail)
    detail = pd.concat(out_frames, ignore_index=True, sort=False) if out_frames else pd.DataFrame()
    return (out, detail) if return_out_of_scope else out


def _annotate_exceptions(
    exceptions: List[Dict[str, Any]],
    start: int,
    variant_name: str,
    loans: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> None:
    """Attach variant and unambiguous loan IDs to module exceptions."""
    borrower_id = scenario["borrower"]["borrower_id_field"]
    loan_id = scenario["borrower"].get("loan_id_field", "loan_id")
    unique: Dict[str, Any] = {}
    if borrower_id in loans.columns and loan_id in loans.columns:
        for value, group in loans.groupby(borrower_id, dropna=False):
            ids = group[loan_id].dropna().astype(str).unique()
            if len(ids) == 1:
                unique[str(value)] = ids[0]
    for row in exceptions[start:]:
        row.setdefault("scenario_variant", variant_name)
        if not row.get("scenario_variant"):
            row["scenario_variant"] = variant_name
        key = str(row.get("borrower_id", ""))
        if key in unique and not row.get("loan_id"):
            row["loan_id"] = unique[key]


def _variant_scenario(scenario: Mapping[str, Any], name: str) -> Dict[str, Any]:
    result = dict(scenario)
    result["_targeted_mode"] = True
    result["_scenario_variant"] = name
    return result


def _base_template(baseline: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    """Create an unstressed-per-level template while retaining base metrics."""
    out = baseline.copy()
    levels = get_levels(scenario)
    for level in levels:
        out[f"stressed_bucket_{level}"] = out["base_bucket"]
        out[f"out_of_scope_{level}"] = False
        for prefix in ("cre_", "ci_"):
            suffix = f"_{level}"
            for column in [col for col in out.columns if col.startswith(prefix) and col.endswith(suffix)]:
                out[column] = np.nan
        if "consumer_pd_unstressed" in out.columns:
            mappings = {
                f"consumer_pd_{level}": "consumer_pd_unstressed",
                f"consumer_lgd_{level}": "consumer_lgd_unstressed",
                f"consumer_lgd_ratio_{level}": "consumer_lgd_ratio_unstressed",
                f"consumer_el_{level}": "consumer_el_unstressed",
                f"consumer_stressed_collateral_value_{level}": "consumer_collateral_value_unstressed",
                f"consumer_proforma_cecl_{level}": "consumer_cecl_reserve_base",
            }
            for target, source in mappings.items():
                if source in out.columns:
                    out[target] = out[source]
    return out


def _copy_rows(target: pd.DataFrame, source: pd.DataFrame, mask: pd.Series) -> None:
    for column in source.columns:
        if column not in target.columns:
            target[column] = np.nan
        target.loc[mask, column] = source.loc[mask, column]


def _resolve_variant(
    context: pd.DataFrame,
    scenario: Mapping[str, Any],
    loaded: Mapping[str, Any],
    variant_name: str,
    variant_spec: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
    selection_rows: List[Dict[str, Any]],
    assumption_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    resolved = context.copy()
    resolved["_targeted_active"] = False
    resolved["_targeted_error"] = False
    effective: Dict[Tuple[int, str, str, str], float] = {}
    baseline_values: Dict[Tuple[int, str, str, str], float] = {}

    operation_sequence: Dict[Tuple[int, str, str, str], int] = {}
    shock_names = [str(item) for item in as_list(variant_spec.get("shocks"))]
    for shock_order, shock_name in enumerate(shock_names, start=1):
        shock = scenario["targeted_stress"]["shocks"][shock_name]
        selected, external_tier = _evaluate_selector(
            context,
            shock.get("selector", shock.get("include")),
            loaded,
            exceptions,
            shock_name,
        )
        if shock.get("exclude"):
            excluded, _ = _evaluate_selector(
                context, shock["exclude"], loaded, exceptions, shock_name
            )
            selected &= ~excluded
        tiers = _resolve_tiers(context, selected, external_tier, shock, loaded, exceptions, shock_name)

        for idx, row in context.iterrows():
            is_selected = bool(selected.at[idx])
            tier_name = tiers.at[idx] if is_selected else ""
            selection_rows.append(
                {
                    "scenario_variant": variant_name,
                    "shock_order": shock_order,
                    "shock": shock_name,
                    "exposure_id": row.get("_exposure_id"),
                    "loan_id": row.get(scenario["borrower"].get("loan_id_field", "loan_id")),
                    "borrower_id": row.get(scenario["borrower"]["borrower_id_field"]),
                    "primary_module": row.get("primary_module"),
                    "tier": tier_name,
                    "selected": is_selected,
                    "balance": row.get(scenario["borrower"]["balance_field"]),
                }
            )
            if not is_selected:
                continue
            module = str(row.get("primary_module", ""))
            tier = shock.get("tiers", {}).get(str(tier_name))
            if not tier:
                resolved.at[idx, "_targeted_error"] = True
                record_exception(
                    exceptions,
                    "ERROR",
                    "targeted_stress",
                    "TARGETED_TIER_UNRESOLVED",
                    "Selected loan did not resolve to a configured shock tier.",
                    borrower_id=row.get(scenario["borrower"]["borrower_id_field"]),
                    module=module,
                    source=shock_name,
                    details=f"loan_id={row.get(scenario['borrower'].get('loan_id_field', 'loan_id'))}; tier={tier_name}",
                )
                continue
            module_specs = tier.get("modules", tier).get(module, {})
            if not module_specs:
                record_exception(
                    exceptions,
                    "INFO",
                    "targeted_stress",
                    "TARGETED_MODULE_OVERRIDE_MISSING",
                    "Selected loan's primary module had no assumptions in the resolved tier.",
                    borrower_id=row.get(scenario["borrower"]["borrower_id_field"]),
                    module=module,
                    source=shock_name,
                    details=f"loan_id={row.get(scenario['borrower'].get('loan_id_field', 'loan_id'))}; tier={tier_name}",
                )
                continue
            for level in get_levels(scenario):
                for parameter, spec in module_specs.items():
                    operation, values = _operation_spec(spec)
                    shock_value = _level_value(values, level)
                    if is_missing(shock_value):
                        raise ValueError(
                            f"Targeted shock '{shock_name}' tier '{tier_name}' "
                            f"{module}.{parameter} has no value for stress level '{level}'."
                        )
                    shock_number = _finite_targeted_number(
                        shock_value,
                        (
                            f"Targeted shock '{shock_name}' tier '{tier_name}' "
                            f"{module}.{parameter}.{level}"
                        ),
                    )
                    key = (idx, module, parameter, level)
                    baseline = baseline_values.setdefault(
                        key, _baseline_parameter(row, scenario, module, parameter, level)
                    )
                    before = effective.get(key, baseline)
                    try:
                        after = _apply_operation(before, shock_number, operation)
                    except OverflowError as exc:
                        raise ValueError(
                            f"Targeted shock '{shock_name}' tier '{tier_name}' "
                            f"{module}.{parameter}.{level} produced a non-finite effective value."
                        ) from exc
                    if not np.isfinite(after):
                        raise ValueError(
                            f"Targeted shock '{shock_name}' tier '{tier_name}' "
                            f"{module}.{parameter}.{level} produced a non-finite effective value."
                        )
                    effective[key] = after
                    operation_sequence[key] = operation_sequence.get(key, 0) + 1
                    resolved.at[idx, targeted_override_column(module, parameter, level)] = after
                    resolved.at[idx, "_targeted_active"] = True
                    assumption_rows.append(
                        {
                            "scenario_variant": variant_name,
                            "shock_order": shock_order,
                            "operation_sequence": operation_sequence[key],
                            "shock": shock_name,
                            "tier": tier_name,
                            "exposure_id": row.get("_exposure_id"),
                            "loan_id": row.get(scenario["borrower"].get("loan_id_field", "loan_id")),
                            "borrower_id": row.get(scenario["borrower"]["borrower_id_field"]),
                            "module": module,
                            "stress_level": level,
                            "parameter": parameter,
                            "operation": operation,
                            "baseline_value": baseline,
                            "value_before_operation": before,
                            "shock_value": shock_number,
                            "effective_value": after,
                        }
                    )
    return resolved


def _selection_error_scope(
    results: pd.DataFrame,
    mask: pd.Series,
    scenario: Mapping[str, Any],
) -> pd.DataFrame:
    """Mark selected loans with unresolved tiers out of scope for every level."""
    if not mask.any():
        return pd.DataFrame()
    borrower = scenario["borrower"]
    borrower_id = borrower["borrower_id_field"]
    loan_id = borrower.get("loan_id_field", "loan_id")
    portfolio = borrower.get("portfolio_field", "portfolio")
    rows: List[Dict[str, Any]] = []
    for idx, row in results.loc[mask].iterrows():
        for level in get_levels(scenario):
            results.at[idx, f"out_of_scope_{level}"] = True
            rows.append(
                {
                    "borrower_id": row.get(borrower_id),
                    "loan_id": row.get(loan_id),
                    "portfolio": row.get(portfolio),
                    "module": row.get("primary_module"),
                    "stress_level": level,
                    "test": "targeted_selection",
                    "field": "tier",
                    "reason": "unresolved_targeted_tier",
                    "scenario_variant": scenario.get("_scenario_variant", ""),
                }
            )
    return pd.DataFrame(rows)


def _resolve_tiers(
    context: pd.DataFrame,
    selected: pd.Series,
    external_tier: pd.Series,
    shock: Mapping[str, Any],
    loaded: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
    shock_name: str,
) -> pd.Series:
    tiers = external_tier.copy()
    for rule in as_list(shock.get("tier_rules")):
        if not isinstance(rule, Mapping) or not rule.get("tier"):
            continue
        rule_mask, _ = _evaluate_selector(
            context,
            rule.get("selector", rule.get("include")),
            loaded,
            exceptions,
            shock_name,
        )
        write = selected & rule_mask & tiers.fillna("").astype(str).eq("")
        tiers.loc[write] = str(rule["tier"])
    default = shock.get("default_tier")
    if default is not None:
        tiers.loc[selected & tiers.fillna("").astype(str).eq("")] = str(default)
    return tiers.fillna("").astype(str)


def _evaluate_selector(
    frame: pd.DataFrame,
    selector: Any,
    loaded: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
    shock_name: str,
) -> tuple[pd.Series, pd.Series]:
    empty_tier = pd.Series("", index=frame.index, dtype=object)
    if isinstance(selector, list):
        mask = pd.Series(True, index=frame.index)
        tiers = empty_tier.copy()
        for item in selector:
            child_mask, child_tier = _evaluate_selector(frame, item, loaded, exceptions, shock_name)
            mask &= child_mask
            tiers = _combine_tiers(tiers, child_tier, mask, shock_name)
        return mask.fillna(False), tiers
    if "all" in selector:
        return _evaluate_selector(frame, list(selector["all"]), loaded, exceptions, shock_name)
    if "any" in selector:
        mask = pd.Series(False, index=frame.index)
        tiers = empty_tier.copy()
        for item in selector["any"]:
            child_mask, child_tier = _evaluate_selector(frame, item, loaded, exceptions, shock_name)
            combined_mask = mask | child_mask
            tiers = _combine_tiers(tiers, child_tier, combined_mask, shock_name)
            mask = combined_mask
        return mask.fillna(False), tiers

    selector_type = str(selector.get("type", "condition"))
    if selector_type == "naics_prefix":
        field = str(selector["field"])
        if field not in frame.columns:
            raise ValueError(
                f"Targeted shock '{shock_name}' NAICS selector references missing "
                f"loan-context field '{field}'."
            )
        prefixes = [_normalize_naics(value) for value in as_list(selector.get("prefixes", selector.get("values")))]
        values = frame[field].apply(_normalize_naics)
        mask = values.apply(
            lambda value: bool(
                isinstance(value, str)
                and value
                and any(value.startswith(prefix) for prefix in prefixes if prefix)
            )
        )
        return mask, empty_tier
    if selector_type == "external_list":
        return _external_list_selector(frame, selector, loaded, exceptions, shock_name)
    condition = {key: value for key, value in selector.items() if key != "type"}
    missing_fields = sorted(condition_fields(condition) - set(frame.columns))
    if missing_fields:
        raise ValueError(
            f"Targeted shock '{shock_name}' condition selector references missing "
            f"loan-context fields: {', '.join(missing_fields)}."
        )
    return evaluate_conditions(frame, condition), empty_tier


def _external_list_selector(
    frame: pd.DataFrame,
    selector: Mapping[str, Any],
    loaded: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
    shock_name: str,
) -> tuple[pd.Series, pd.Series]:
    source_name = str(selector["source"])
    source_field = str(selector["source_field"])
    exposure_field = str(selector["exposure_field"])
    if source_name not in loaded:
        raise ValueError(f"Targeted selector references unknown input source '{source_name}'.")
    source = loaded[source_name].frame
    if source_field not in source.columns:
        raise ValueError(
            f"Targeted selector source '{source_name}' is missing field '{source_field}'."
        )
    if exposure_field not in frame.columns:
        raise ValueError(f"Loan context is missing selector field '{exposure_field}'.")
    work = source.copy()
    if selector.get("where"):
        missing_where_fields = sorted(
            condition_fields(selector["where"]) - set(work.columns)
        )
        if missing_where_fields:
            raise ValueError(
                f"Targeted selector source '{source_name}' where filter references "
                f"missing fields: {', '.join(missing_where_fields)}."
            )
        work = work[evaluate_conditions(work, selector["where"])]
    keys = work[source_field].apply(_normalize_key)
    missing = keys.eq("")
    if missing.any():
        record_exception(
            exceptions,
            "WARNING",
            "targeted_stress",
            "TARGETED_EXTERNAL_LIST_NULL_KEY",
            "External selector rows with blank keys were ignored.",
            source=source_name,
            field=source_field,
            details=f"shock={shock_name}; count={int(missing.sum())}",
        )
    work = work.loc[~missing].copy()
    work["_match_key"] = keys.loc[~missing]
    work = work.drop_duplicates()
    tier_field = selector.get("tier_field")
    if tier_field and tier_field not in work.columns:
        raise ValueError(
            f"Targeted selector source '{source_name}' is missing tier_field '{tier_field}'."
        )
    if tier_field:
        conflicts = work.groupby("_match_key", dropna=False)[tier_field].apply(
            lambda values: values.dropna().astype(str).nunique()
        )
        bad = conflicts[conflicts > 1]
        if not bad.empty:
            raise ValueError(
                f"External selector '{source_name}' has conflicting tiers for keys: "
                f"{', '.join(str(value) for value in bad.index[:5])}"
            )

    exposure_values = frame[exposure_field].apply(_normalize_key)
    match_kind = str(selector.get("match", "exact"))
    tiers = pd.Series("", index=frame.index, dtype=object)
    if match_kind == "exact":
        source_keys = set(work["_match_key"])
        mask = exposure_values.isin(source_keys)
        if tier_field:
            tier_map = (
                work.dropna(subset=[tier_field])
                .drop_duplicates("_match_key")
                .set_index("_match_key")[tier_field]
                .astype(str)
            )
            tiers.loc[mask] = exposure_values.loc[mask].map(tier_map).fillna("")
        matched_source = set(exposure_values.loc[mask])
    else:
        records = work[["_match_key"] + ([tier_field] if tier_field else [])].to_dict("records")
        mask_values = []
        for idx, value in exposure_values.items():
            matches = [item for item in records if value and value.startswith(item["_match_key"])]
            mask_values.append(bool(matches))
            if tier_field:
                found = {str(item[tier_field]) for item in matches if pd.notna(item.get(tier_field))}
                if len(found) > 1:
                    raise ValueError(
                        f"External selector '{source_name}' assigns conflicting prefix tiers to '{value}'."
                    )
                if found:
                    tiers.at[idx] = next(iter(found))
        mask = pd.Series(mask_values, index=frame.index)
        matched_source = {
            key for key in work["_match_key"] if exposure_values.apply(lambda value: bool(value and value.startswith(key))).any()
        }
    orphan = set(work["_match_key"]) - matched_source
    if orphan:
        record_exception(
            exceptions,
            "WARNING",
            "targeted_stress",
            "TARGETED_EXTERNAL_LIST_ORPHAN_KEY",
            "External selector keys did not match the loan context.",
            source=source_name,
            field=source_field,
            details=f"shock={shock_name}; orphan_key_count={len(orphan)}",
        )
    if exposure_field == "loan_id" and "_loan_id_ambiguous" in frame.columns:
        ambiguous = frame["_loan_id_ambiguous"].fillna(False).astype(bool)
        if (mask & ambiguous).any():
            record_exception(
                exceptions,
                "ERROR",
                "targeted_stress",
                "TARGETED_EXTERNAL_LOAN_ID_AMBIGUOUS",
                "An external loan-ID selector matched a missing or duplicate identity loan ID; ambiguous rows were excluded.",
                source=source_name,
                field=exposure_field,
                details=f"shock={shock_name}; count={int((mask & ambiguous).sum())}",
            )
            mask &= ~ambiguous
            tiers.loc[ambiguous] = ""
    return mask.fillna(False), tiers


def _combine_tiers(
    left: pd.Series,
    right: pd.Series,
    applicable: pd.Series,
    shock_name: str,
) -> pd.Series:
    out = left.copy()
    left_text = left.fillna("").astype(str)
    right_text = right.fillna("").astype(str)
    conflict = applicable & left_text.ne("") & right_text.ne("") & left_text.ne(right_text)
    if conflict.any():
        raise ValueError(
            f"Targeted shock '{shock_name}' selector resolved conflicting tiers for "
            f"{int(conflict.sum())} loans."
        )
    write = applicable & left_text.eq("") & right_text.ne("")
    out.loc[write] = right_text.loc[write]
    return out


def _baseline_parameter(
    row: Mapping[str, Any],
    scenario: Mapping[str, Any],
    module: str,
    parameter: str,
    level: str,
) -> float:
    config = scenario.get("modules", {}).get(module, {})
    if module == "C&I":
        sector = row.get(config.get("sector_field", "ci_sector"))
        if parameter == "ebitda_reduction":
            brg = _brg_key(row.get(scenario["borrower"].get("risk_rating_field", "risk_rating")))
            return to_number(_ebitda_reduction(config, sector, brg, level)[0])
        return to_number(
            lookup_parameter_with_source(config.get("interest_rate_stress"), sector, level, np.nan)[0]
        )
    if module == "CRE":
        tests = config.get("tests", {})
        sector = row.get(config.get("subsector_field", "cre_subsector"))
        dscr = tests.get("dscr", {})
        refi = tests.get("refinance", {})
        ltv = tests.get("ltv", {})
        tables = {
            "dscr_decline": dscr.get("decline"),
            "refinance_noi_decline": refi.get("noi_decline", dscr.get("decline")),
            "credit_spread": refi.get("credit_spreads"),
            "amortization_years": refi.get("amortization_years"),
            "cap_rate": ltv.get("cap_rates"),
        }
        if parameter == "treasury_rate":
            return to_number(refi.get("treasury_rate"))
        return to_number(lookup_parameter_with_source(tables[parameter], sector, level, np.nan)[0])
    if module == "Consumer":
        segment = row.get(config.get("segment_field", ""))
        if parameter in {"rushed_sale_discount", "closing_costs"}:
            return to_number(config.get(parameter))
        return to_number(
            lookup_parameter_with_source(config.get(parameter), segment, level, np.nan)[0]
        )
    return np.nan


def _operation_spec(spec: Any) -> tuple[str, Any]:
    if isinstance(spec, Mapping):
        operation = str(spec.get("operation", spec.get("op", "replace"))).lower()
        values = spec.get("values", spec.get("value"))
        return operation, values
    return "replace", spec


def _level_value(values: Any, level: str) -> Any:
    if isinstance(values, Mapping):
        if level in values:
            return values[level]
        for key in ("default", "all", "*"):
            if key in values:
                return values[key]
        return np.nan
    return values


def _apply_operation(baseline: float, value: float, operation: str) -> float:
    if operation == "replace":
        return value
    if operation == "add":
        return baseline + value
    if operation == "multiply":
        return baseline * value
    raise ValueError(f"Unsupported targeted operation: {operation}")


def _normalize_key(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _normalize_naics(value: Any) -> str | None:
    text = _normalize_key(value)
    return text if re.fullmatch(r"\d{2,6}", text) else None


def _collect_variant_reports(
    target: Dict[str, List[pd.DataFrame]],
    reports: Mapping[str, pd.DataFrame],
    variant_name: str,
) -> None:
    for name, frame in reports.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        item = frame.copy()
        item.insert(0, "scenario_variant", variant_name)
        target.setdefault(name, []).append(item)


def _selection_summary(detail: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    selected = detail[detail["selected"]].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "scenario_variant",
                "shock",
                "tier",
                "primary_module",
                "borrower_count",
                "loan_count",
                "balance",
            ]
        )
    borrower_id = scenario["borrower"]["borrower_id_field"]
    if borrower_id != "borrower_id":
        selected[borrower_id] = selected["borrower_id"]
    return (
        selected.groupby(
            ["scenario_variant", "shock", "tier", "primary_module"], dropna=False
        )
        .agg(
            borrower_count=("borrower_id", "nunique"),
            loan_count=("exposure_id", "nunique"),
            balance=("balance", "sum"),
        )
        .reset_index()
    )


def _variant_comparison(reports: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    cecl = reports.get("cecl_summary", pd.DataFrame())
    if cecl.empty or "scenario_variant" not in cecl.columns:
        return pd.DataFrame()
    totals = cecl[
        cecl.get("bucket", pd.Series(index=cecl.index, dtype=object)).eq("Total")
    ].copy()
    if totals.empty:
        return pd.DataFrame()
    baseline = totals[totals["scenario_variant"] == "baseline"][
        ["portfolio", "stress_level", "proforma_cecl_reserve", "proforma_cecl_ratio"]
    ].rename(
        columns={
            "proforma_cecl_reserve": "baseline_cecl_reserve",
            "proforma_cecl_ratio": "baseline_cecl_ratio",
        }
    )
    changed = totals[totals["scenario_variant"] != "baseline"].merge(
        baseline, on=["portfolio", "stress_level"], how="left"
    )
    changed["delta_cecl_reserve"] = (
        pd.to_numeric(changed["proforma_cecl_reserve"], errors="coerce")
        - pd.to_numeric(changed["baseline_cecl_reserve"], errors="coerce")
    )
    changed["delta_cecl_ratio"] = (
        pd.to_numeric(changed["proforma_cecl_ratio"], errors="coerce")
        - pd.to_numeric(changed["baseline_cecl_ratio"], errors="coerce")
    )
    return changed[
        [
            "scenario_variant",
            "portfolio",
            "stress_level",
            "baseline_cecl_reserve",
            "proforma_cecl_reserve",
            "delta_cecl_reserve",
            "baseline_cecl_ratio",
            "proforma_cecl_ratio",
            "delta_cecl_ratio",
        ]
    ]
