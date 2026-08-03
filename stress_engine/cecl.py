"""Configurable current and portfolio-history CECL reserve bases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from .exceptions import record_exception
from .utils import pct, to_number


CURRENT_METHODS = {"in_place", "central_tendency"}
WEIGHT_TOLERANCE = 1e-9
DEFAULT_Z_SCORE_THRESHOLD = 2.0
EFFECTIVE_RESERVE_FIELD = "cecl_effective_reserve_base"
RESERVE_BASIS_METHOD_FIELD = "cecl_reserve_basis_method"
INVALID_BALANCE_COUNT_FIELD = "_cecl_invalid_balance_count"
RESERVED_INPUT_FIELDS = {
    "_source_file",
    "_source_file_row",
    "_source_row",
    "_portfolio_key",
    "_period_key",
}
CURRENT_CECL_RESERVED_FIELDS = RESERVED_INPUT_FIELDS | {
    INVALID_BALANCE_COUNT_FIELD,
    EFFECTIVE_RESERVE_FIELD,
    RESERVE_BASIS_METHOD_FIELD,
    "all_tags",
    "base_bucket",
    "calculated_cash_paid_for_interest",
    "calculated_cash_paid_for_interest_fallback_reason",
    "calculated_cash_paid_for_interest_source",
    "consumer_appraised_value",
    "consumer_cecl_reserve_base",
    "consumer_collateral_value_unstressed",
    "consumer_fico",
    "consumer_qualitative_reserve",
    "eligible_modules",
    "loan_count",
    "_exposure_id",
    "_loan_id_ambiguous",
    "model_excluded",
    "model_exclusion_tags",
    "model_tags",
    "module_applied",
    "primary_module",
    "scenario_variant",
    "_scenario_variant",
}
CURRENT_CECL_RESERVED_PREFIXES = (
    "_cecl_",
    "_cecl_reserve_missing_count__",
    "_source_",
    "_targeted_",
    "ci_available_cash_flow_",
    "ci_debt_service_",
    "ci_fccr_",
    "consumer_el_",
    "consumer_lgd_",
    "consumer_pd_",
    "consumer_proforma_cecl_",
    "consumer_stressed_collateral_value_",
    "cre_dscr_",
    "cre_ltv_",
    "cre_refi_dscr_",
    "out_of_scope_",
    "stressed_bucket_",
    "tag_",
)

RATIO_COLUMNS = [
    "portfolio",
    "bucket",
    "reserve_basis",
    "current_method",
    "history_enabled",
    "base_balance",
    "base_reserve",
    "reserve_ratio",
    "invalid_balance_count",
    "status",
    "exception_code",
]

AUDIT_COLUMNS = [
    "portfolio",
    "bucket",
    "reserve_basis",
    "current_method",
    "history_enabled",
    "period_method",
    "period",
    "source",
    "reserve_field",
    "weight",
    "observation_grain",
    "observation_count",
    "included_observation_count",
    "excluded_observation_count",
    "missing_reserve_count",
    "invalid_balance_count",
    "balance",
    "allocation_balance_source",
    "reserve",
    "raw_reserve_ratio",
    "raw_mean_reserve_ratio",
    "raw_std_reserve_ratio",
    "period_reserve_ratio",
    "weighted_ratio_component",
    "effective_reserve_ratio",
    "status",
    "exception_code",
    "basis_status",
    "basis_exception_code",
]


@dataclass(frozen=True)
class CeclReserveBasis:
    """One resolved basis shared by stress modules and CECL reporting."""

    method: str
    current_method: str
    history_enabled: bool
    effective_reserve: pd.Series
    method_by_row: pd.Series
    ratios: pd.DataFrame
    audit: pd.DataFrame
    required_fields: tuple[str, ...]
    exception_code: str = ""


def attach_cecl_reserve_basis(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
    history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, CeclReserveBasis]:
    """Attach the selected Base reserve before stress modules execute."""
    out = results.copy()
    cecl = scenario.get("cecl", {})
    portfolio_field = (
        str(cecl.get("portfolio_field", "cecl_portfolio"))
        if isinstance(cecl, Mapping)
        else "cecl_portfolio"
    )
    if portfolio_field in out.columns:
        out[portfolio_field] = _normalized_portfolios(out[portfolio_field])
    basis = build_cecl_reserve_basis(
        out, scenario, exceptions, history=history
    )
    out[EFFECTIVE_RESERVE_FIELD] = basis.effective_reserve.reindex(out.index)
    out[RESERVE_BASIS_METHOD_FIELD] = basis.method_by_row.reindex(out.index)
    return out, basis


def cecl_history_frame(
    scenario: Mapping[str, Any], loaded: Mapping[str, Any]
) -> pd.DataFrame | None:
    """Return the configured non-merged history table from loaded inputs."""
    basis = _basis_config(scenario)
    historical = _historical_config(basis)
    if not _history_enabled(historical):
        return None
    source = str(historical.get("source", "")).strip()
    item = loaded.get(source)
    if item is None:
        return None
    frame = getattr(item, "frame", item)
    return frame if isinstance(frame, pd.DataFrame) else None


def validate_cecl_config(scenario: Mapping[str, Any]) -> None:
    """Validate the optional scenario-level CECL reserve-basis contract."""
    cecl = scenario.get("cecl", {})
    if not isinstance(cecl, Mapping):
        raise ValueError("Scenario cecl must be a JSON object.")
    current_field = _exact_field_setting(
        cecl, "reserve_field", "cecl_reserve", "cecl.reserve_field"
    )
    portfolio_field = _exact_field_setting(
        cecl, "portfolio_field", "cecl_portfolio", "cecl.portfolio_field"
    )
    _validate_current_field_names(
        scenario, current_field, portfolio_field
    )
    configured_aggregation = scenario.get("borrower", {}).get(
        "aggregation", {}
    )
    if (
        isinstance(configured_aggregation, Mapping)
        and current_field in configured_aggregation
        and str(configured_aggregation[current_field]) != "sum"
    ):
        raise ValueError(
            f"Current CECL reserve field '{current_field}' must use borrower "
            "aggregation 'sum'."
        )
    basis = cecl.get("reserve_basis")
    if basis is None:
        return
    if not isinstance(basis, Mapping):
        raise ValueError("cecl.reserve_basis must be a JSON object.")
    if "weighted_history" in basis:
        raise ValueError(
            "cecl.reserve_basis.weighted_history is no longer supported; "
            "use reserve_basis.historical with a portfolio-level input source."
        )

    _validated_current_method(basis)
    central = basis.get("central_tendency", {})
    if not isinstance(central, Mapping):
        raise ValueError(
            "cecl.reserve_basis.central_tendency must be a JSON object."
        )
    threshold = _finite_number(
        central.get("z_score_threshold", DEFAULT_Z_SCORE_THRESHOLD),
        "cecl.reserve_basis.central_tendency.z_score_threshold",
    )
    if threshold <= 0:
        raise ValueError(
            "cecl.reserve_basis.central_tendency.z_score_threshold must be greater than zero."
        )
    grain = str(central.get("observation_grain", "borrower"))
    if grain != "borrower":
        raise ValueError(
            "cecl.reserve_basis.central_tendency.observation_grain currently supports only 'borrower'."
        )

    _validate_current_reserve_wiring(scenario, cecl)

    historical = basis.get("historical", {})
    if historical is None or not isinstance(historical, Mapping):
        raise ValueError(
            "cecl.reserve_basis.historical must be a JSON object."
        )
    enabled = historical.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "cecl.reserve_basis.historical.enabled must be a JSON boolean."
        )
    if not enabled:
        return

    source = _nonblank_setting(
        historical,
        "source",
        "cecl.reserve_basis.historical.source",
    )
    period_field = _configured_field(
        historical, "period_field", "period"
    )
    portfolio_field = _configured_field(
        historical,
        "portfolio_field",
        str(cecl.get("portfolio_field", "cecl_portfolio")),
    )
    reserve_field = _configured_field(
        historical, "reserve_field", "historical_cecl_reserve"
    )
    historical_fields = {period_field, portfolio_field, reserve_field}
    if len(historical_fields) != 3:
        raise ValueError(
            "CECL historical portfolio, period, and reserve fields must be distinct."
        )
    conflicts = sorted(historical_fields & RESERVED_INPUT_FIELDS)
    if conflicts:
        raise ValueError(
            "CECL historical fields cannot use reserved internal names: "
            f"{', '.join(conflicts)}."
        )

    current_period = historical.get("current_period")
    if not isinstance(current_period, Mapping):
        raise ValueError(
            "cecl.reserve_basis.historical.current_period must be a JSON object."
        )
    current_name = str(current_period.get("name", "")).strip()
    if not current_name:
        raise ValueError(
            "cecl.reserve_basis.historical.current_period.name must be nonblank."
        )
    current_weight = _positive_weight(
        current_period.get("weight"),
        "cecl.reserve_basis.historical.current_period.weight",
    )

    periods = historical.get("periods")
    if not isinstance(periods, list) or not periods:
        raise ValueError(
            "cecl.reserve_basis.historical.periods must be a nonempty JSON list."
        )
    names = {current_name}
    weights = [current_weight]
    for index, period in enumerate(periods):
        path = f"cecl.reserve_basis.historical.periods[{index}]"
        if not isinstance(period, Mapping):
            raise ValueError(f"{path} must be a JSON object.")
        if "reserve_field" in period or "period_method" in period:
            raise ValueError(
                f"{path} must reference a portfolio CSV period by name and weight only."
            )
        name = str(period.get("name", "")).strip()
        if not name:
            raise ValueError(f"{path}.name must be nonblank.")
        if name in names:
            raise ValueError(
                "CECL current and historical period names must be unique."
            )
        names.add(name)
        weights.append(_positive_weight(period.get("weight"), f"{path}.weight"))
    if not math.isclose(
        math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=WEIGHT_TOLERANCE
    ):
        raise ValueError(
            "CECL current and historical period weights must sum to 1."
        )

    sources = scenario.get("inputs", {}).get("sources", {})
    source_spec = sources.get(source) if isinstance(sources, Mapping) else None
    if not isinstance(source_spec, Mapping):
        raise ValueError(
            f"CECL historical source '{source}' must exist in inputs.sources."
        )
    if source_spec.get("merge") is not False:
        raise ValueError(
            f"CECL historical source '{source}' must set merge to false."
        )
    aliases = source_spec.get("column_aliases", {})
    canonical = set(aliases) if isinstance(aliases, Mapping) else set()
    required = {str(value) for value in source_spec.get("required_columns", [])}
    numeric = {str(value) for value in source_spec.get("numeric_columns", [])}
    strings = {str(value) for value in source_spec.get("string_columns", [])}
    expected = {period_field, portfolio_field, reserve_field}
    missing_aliases = sorted(expected - canonical)
    missing_required = sorted(expected - required)
    missing_numeric = sorted({reserve_field} - numeric)
    missing_strings = sorted({period_field, portfolio_field} - strings)
    if missing_aliases:
        raise ValueError(
            "CECL historical fields must be canonical source column aliases; "
            f"missing: {', '.join(missing_aliases)}."
        )
    if missing_required:
        raise ValueError(
            "CECL historical fields must be source required_columns; missing: "
            f"{', '.join(missing_required)}."
        )
    if missing_numeric:
        raise ValueError(
            "The CECL historical reserve field must be numeric; missing: "
            f"{', '.join(missing_numeric)}."
        )
    if missing_strings:
        raise ValueError(
            "CECL historical portfolio and period fields must be string_columns; "
            f"missing: {', '.join(missing_strings)}."
        )


def reserve_basis_fields(scenario: Mapping[str, Any]) -> tuple[str, ...]:
    """Return loan fields needed by CECL; history is never loan-grain."""
    cecl = scenario.get("cecl", {})
    if not isinstance(cecl, Mapping):
        return ("cecl_reserve",)
    return (str(cecl.get("reserve_field", "cecl_reserve")),)


def reserve_missing_count_field(reserve_field: str) -> str:
    """Return the internal loan-missing counter carried through aggregation."""
    return f"_cecl_reserve_missing_count__{reserve_field}"


def build_cecl_reserve_basis(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
    history: pd.DataFrame | None = None,
) -> CeclReserveBasis:
    """Resolve current and optional portfolio-history reserve ratios."""
    exceptions = exceptions if exceptions is not None else []
    cecl = scenario.get("cecl", {})
    if not isinstance(cecl, Mapping):
        cecl = {}
    basis_config = _basis_config(scenario)
    current_method = _current_method(basis_config)
    historical = _historical_config(basis_config)
    history_enabled = _history_enabled(historical)
    method = _basis_label(current_method, history_enabled)
    current_field = str(cecl.get("reserve_field", "cecl_reserve"))
    required_fields = (current_field,)
    portfolio_field = str(
        cecl.get(
            "portfolio_field",
            scenario.get("borrower", {}).get("portfolio_field", "portfolio"),
        )
    )
    balance_field = str(
        scenario.get("borrower", {}).get("balance_field", "outstanding_balance")
    )
    basis_results = results.copy()
    if portfolio_field in basis_results.columns:
        basis_results[portfolio_field] = _normalized_portfolios(
            basis_results[portfolio_field]
        )
    effective = pd.Series(np.nan, index=results.index, dtype=float)
    method_by_row = _row_basis_methods(
        basis_results, scenario, portfolio_field, method
    )

    if current_field not in basis_results.columns:
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            "CECL_RESERVE_FIELD_MISSING",
            "The configured current CECL reserve field is missing; CECL is unavailable.",
            field=current_field,
        )
        return _unavailable_basis(
            method,
            current_method,
            history_enabled,
            effective,
            method_by_row,
            required_fields,
            "CECL_RESERVE_FIELD_MISSING",
        )

    missing_count = int(
        _finite_numeric(basis_results[current_field]).isna().sum()
    )
    if missing_count and not _exception_exists(
        exceptions,
        "CECL_LOAN_RESERVE_MISSING_TREATED_AS_ZERO",
        current_field,
    ):
        record_exception(
            exceptions,
            "WARNING",
            "cecl",
            "CECL_LOAN_RESERVE_MISSING_TREATED_AS_ZERO",
            "Current loan CECL reserve values that are missing or invalid were treated as zero.",
            field=current_field,
            details=f"missing_count={missing_count}",
        )

    if (
        portfolio_field not in basis_results.columns
        or balance_field not in basis_results.columns
    ):
        missing = (
            portfolio_field
            if portfolio_field not in basis_results.columns
            else balance_field
        )
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            "CECL_BASIS_GROUP_FIELD_MISSING",
            "A grouping field required by CECL is missing; CECL is unavailable.",
            field=missing,
        )
        return _unavailable_basis(
            method,
            current_method,
            history_enabled,
            effective,
            method_by_row,
            required_fields,
            "CECL_BASIS_GROUP_FIELD_MISSING",
        )

    frame = basis_results.copy()
    if "model_excluded" in frame.columns:
        frame = frame[~frame["model_excluded"].fillna(False).astype(bool)].copy()
    missing_portfolio = frame[portfolio_field].isna() | frame[
        portfolio_field
    ].astype(str).str.strip().eq("")
    if missing_portfolio.any():
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            "CECL_PORTFOLIO_MISSING",
            "Rows without a CECL portfolio were excluded from reserve-basis calibration.",
            field=portfolio_field,
            details=f"missing_count={int(missing_portfolio.sum())}",
        )
        frame = frame.loc[~missing_portfolio].copy()
    if "base_bucket" not in frame.columns:
        frame["base_bucket"] = "Unknown"
    else:
        missing_bucket = frame["base_bucket"].isna() | frame[
            "base_bucket"
        ].astype(str).str.strip().eq("")
        frame.loc[missing_bucket, "base_bucket"] = "Unknown"

    consumer_portfolios = _consumer_portfolios(
        frame, scenario, portfolio_field
    )
    commercial_frame = frame[
        ~frame[portfolio_field].map(_portfolio_key).isin(consumer_portfolios)
    ]
    portfolio_balances: Dict[str, float] = {}
    portfolio_invalid: Dict[str, int] = {}
    for portfolio, group in commercial_frame.groupby(portfolio_field, dropna=False):
        key = _portfolio_key(portfolio)
        portfolio_balances[key] = float(_finite_numeric(group[balance_field]).sum())
        portfolio_invalid[key] = _invalid_balance_count(group, balance_field)

    history_lookup, history_global_code = _prepare_history(
        history,
        historical,
        history_enabled,
        exceptions,
        set(portfolio_balances),
    )
    threshold = to_number(
        basis_config.get("central_tendency", {}).get(
            "z_score_threshold", DEFAULT_Z_SCORE_THRESHOLD
        ),
        DEFAULT_Z_SCORE_THRESHOLD,
    )
    borrower_field = str(
        scenario.get("borrower", {}).get("borrower_id_field", "borrower_id")
    )
    current_period, historical_periods = _period_specs(
        historical, history_enabled
    )

    ratio_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    group_columns = [portfolio_field, "base_bucket"]
    for (portfolio, bucket), group in frame.groupby(group_columns, dropna=False):
        bucket_text = str(bucket)
        if bucket_text not in {
            "Pass",
            "Special Mention",
            "Substandard",
            "Unknown",
        }:
            continue
        portfolio_key = _portfolio_key(portfolio)
        is_consumer = portfolio_key in consumer_portfolios
        group_current_method = "in_place" if is_consumer else current_method
        group_history_enabled = history_enabled and not is_consumer
        group_method = _basis_label(
            group_current_method, group_history_enabled
        )
        group_current_period = (
            {"name": "current", "weight": 1.0}
            if is_consumer or not group_history_enabled
            else current_period
        )
        finite_balances = _finite_numeric(group[balance_field])
        eligible_group = group.loc[finite_balances.notna()]
        group_balance = float(finite_balances.sum())
        invalid_balance_count = _invalid_balance_count(group, balance_field)
        basis_invalid_balance_count = (
            invalid_balance_count
            if is_consumer
            else portfolio_invalid.get(portfolio_key, invalid_balance_count)
        )
        if invalid_balance_count:
            record_exception(
                exceptions,
                "ERROR",
                "cecl",
                "CECL_BALANCE_INVALID",
                "A CECL group contains negative, invalid, or nonfinite balances; its reserve basis is unavailable.",
                portfolio=portfolio,
                bucket=bucket_text,
                field=balance_field,
                details=f"invalid_balance_count={invalid_balance_count}",
            )

        observations = _borrower_observations(
            eligible_group, borrower_field, balance_field, current_field
        )
        current_values = _calculate_current_period(
            observations,
            group_current_method,
            threshold,
            group_balance,
        )
        selected: List[tuple[Dict[str, Any], Dict[str, Any]]] = [
            (group_current_period, current_values)
        ]
        group_audit: List[Dict[str, Any]] = [
            _audit_row(
                portfolio,
                bucket_text,
                group_method,
                group_current_method,
                group_history_enabled,
                group_current_method,
                group_current_period,
                "identity",
                current_field,
                "borrower",
                "current_bucket",
                invalid_balance_count,
                current_values,
            )
        ]

        if group_history_enabled:
            portfolio_balance = portfolio_balances.get(portfolio_key, np.nan)
            portfolio_balance_invalid = portfolio_invalid.get(portfolio_key, 0)
            for period in historical_periods:
                values = _historical_period_values(
                    history_lookup,
                    history_global_code,
                    portfolio_key,
                    period["name"],
                    portfolio_balance,
                    portfolio_balance_invalid,
                )
                selected.append((period, values))
                group_audit.append(
                    _audit_row(
                        portfolio,
                        bucket_text,
                        group_method,
                        group_current_method,
                        True,
                        "portfolio_history",
                        period,
                        str(historical.get("source", "")),
                        str(
                            historical.get(
                                "reserve_field", "historical_cecl_reserve"
                            )
                        ),
                        "portfolio",
                        "current_portfolio",
                        portfolio_balance_invalid,
                        values,
                    )
                )

        for period, values in selected:
            if values["status"] == "available":
                continue
            code = str(values.get("exception_code") or "CECL_BASIS_PERIOD_UNAVAILABLE")
            record_exception(
                exceptions,
                "ERROR" if group_balance > 0 else "WARNING",
                "cecl",
                code,
                "A configured CECL basis period is unavailable.",
                portfolio=portfolio,
                bucket=bucket_text,
                field=(
                    current_field
                    if period is group_current_period
                    else str(historical.get("reserve_field", "historical_cecl_reserve"))
                ),
                details=f"period={period['name']}",
            )

        direct_amount_basis = (
            group_current_method == "in_place" and not group_history_enabled
        )
        direct_values: pd.Series | None = None
        if direct_amount_basis:
            direct_values = _finite_numeric(group[current_field]).fillna(0.0)
            direct_values = direct_values.where(finite_balances.notna())
            effective.loc[group.index] = direct_values

        unavailable = [
            values for _, values in selected if values["status"] != "available"
        ]
        if basis_invalid_balance_count:
            ratio = np.nan
            reserve = np.nan
            status = "unavailable"
            code = "CECL_BALANCE_INVALID"
            effective.loc[group.index] = np.nan
        elif unavailable:
            ratio = np.nan
            reserve = np.nan
            status = "unavailable"
            history_error = next(
                (
                    str(values.get("exception_code"))
                    for values in unavailable
                    if str(values.get("exception_code", "")).startswith(
                        "CECL_HISTORY_"
                    )
                ),
                "",
            )
            code = history_error or str(
                unavailable[0].get("exception_code")
                or "CECL_BASIS_PERIOD_UNAVAILABLE"
            )
            if not direct_amount_basis:
                effective.loc[group.index] = np.nan
        elif direct_values is not None:
            reserve = _finite_fsum(direct_values.dropna())
            ratio = pct(reserve, group_balance) if pd.notna(reserve) else np.nan
            if np.isfinite(ratio) and np.isfinite(reserve):
                status = "available"
                code = ""
            else:
                ratio = np.nan
                reserve = np.nan
                status = "unavailable"
                code = "CECL_BASIS_RESULT_NONFINITE"
        else:
            ratio = _finite_fsum(
                period["weight"] * values["period_reserve_ratio"]
                for period, values in selected
            )
            reserve = group_balance * ratio
            if np.isfinite(ratio) and np.isfinite(reserve):
                status = "available"
                code = ""
                effective.loc[group.index] = finite_balances * ratio
            else:
                ratio = np.nan
                reserve = np.nan
                status = "unavailable"
                code = "CECL_BASIS_RESULT_NONFINITE"
                effective.loc[group.index] = np.nan

        ratio_rows.append(
            {
                "portfolio": portfolio,
                "bucket": bucket_text,
                "reserve_basis": group_method,
                "current_method": group_current_method,
                "history_enabled": group_history_enabled,
                "base_balance": group_balance,
                "base_reserve": reserve,
                "reserve_ratio": ratio,
                "invalid_balance_count": basis_invalid_balance_count,
                "status": status,
                "exception_code": code,
            }
        )
        for row in group_audit:
            row["effective_reserve_ratio"] = ratio
            row["basis_status"] = status
            row["basis_exception_code"] = code
            audit_rows.append(row)

    return CeclReserveBasis(
        method=method,
        current_method=current_method,
        history_enabled=history_enabled,
        effective_reserve=effective,
        method_by_row=method_by_row,
        ratios=pd.DataFrame(ratio_rows, columns=RATIO_COLUMNS),
        audit=pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS),
        required_fields=required_fields,
    )


def _basis_config(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    cecl = scenario.get("cecl", {})
    if not isinstance(cecl, Mapping):
        return {}
    basis = cecl.get("reserve_basis", {})
    return basis if isinstance(basis, Mapping) else {}


def _historical_config(basis: Mapping[str, Any]) -> Mapping[str, Any]:
    historical = basis.get("historical", {})
    return historical if isinstance(historical, Mapping) else {}


def _history_enabled(historical: Mapping[str, Any]) -> bool:
    return historical.get("enabled", False) is True


def _validated_current_method(basis: Mapping[str, Any]) -> str:
    legacy = basis.get("method")
    current = basis.get("current_method", legacy if legacy is not None else "in_place")
    if legacy is not None and "current_method" in basis and str(legacy) != str(current):
        raise ValueError(
            "cecl.reserve_basis.method and current_method cannot disagree."
        )
    method = str(current)
    if method not in CURRENT_METHODS:
        raise ValueError(
            "cecl.reserve_basis.current_method must be one of: "
            f"{', '.join(sorted(CURRENT_METHODS))}."
        )
    return method


def _current_method(basis: Mapping[str, Any]) -> str:
    method = basis.get("current_method", basis.get("method", "in_place"))
    return str(method) if str(method) in CURRENT_METHODS else "in_place"


def _basis_label(current_method: str, history_enabled: bool) -> str:
    return (
        f"{current_method}+portfolio_history"
        if history_enabled
        else current_method
    )


def _validate_current_reserve_wiring(
    scenario: Mapping[str, Any], cecl: Mapping[str, Any]
) -> None:
    field = str(cecl.get("reserve_field", "cecl_reserve"))
    identity = scenario.get("inputs", {}).get("identity", {})
    if not isinstance(identity, Mapping) or not identity:
        return
    aliases = identity.get("column_aliases", {})
    canonical = set(aliases) if isinstance(aliases, Mapping) else set()
    numeric = {str(value) for value in identity.get("numeric_columns", [])}
    required = {str(value) for value in identity.get("required_columns", [])}
    checks = [
        (canonical, "a canonical identity column alias"),
        (numeric, "an identity numeric_column"),
        (required, "an identity required_column"),
    ]
    for configured, description in checks:
        if field not in configured:
            raise ValueError(
                f"Current CECL reserve field '{field}' must be {description}."
            )


def _exact_field_setting(
    config: Mapping[str, Any], key: str, default: str, path: str
) -> str:
    """Return a canonical field setting that the loader will not rename."""
    raw = config.get(key, default)
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ValueError(
            f"{path} must be a nonblank string without surrounding whitespace."
        )
    return raw


def _validate_current_field_names(
    scenario: Mapping[str, Any], reserve_field: str, portfolio_field: str
) -> None:
    """Reject current CECL names that the borrower/stress pipeline overwrites."""
    if reserve_field == portfolio_field:
        raise ValueError(
            "cecl.reserve_field and cecl.portfolio_field must be distinct."
        )

    borrower = scenario.get("borrower", {})
    borrower_fields = {
        str(value)
        for value in (
            borrower.get("borrower_id_field", "borrower_id"),
            borrower.get("loan_id_field", "loan_id"),
            borrower.get("balance_field", "outstanding_balance"),
            borrower.get("module_field", "model_module"),
            borrower.get("portfolio_field", "model_portfolio"),
            borrower.get("risk_rating_field", "risk_rating"),
            borrower.get("maturity_date_field", "maturity_date"),
        )
        if value is not None and str(value)
    }
    generated = set(CURRENT_CECL_RESERVED_FIELDS)
    generated.update(
        f"stressed_bucket_{level}"
        for level in scenario.get("stress_levels", ["S1", "S2"])
    )
    generated.update(
        f"out_of_scope_{level}"
        for level in scenario.get("stress_levels", ["S1", "S2"])
    )

    reserve_conflicts = generated | borrower_fields
    if reserve_field in reserve_conflicts or reserve_field.startswith(
        CURRENT_CECL_RESERVED_PREFIXES
    ):
        raise ValueError(
            f"cecl.reserve_field '{reserve_field}' conflicts with a "
            "borrower or engine-owned field."
        )

    # A CECL portfolio may intentionally reuse borrower.portfolio_field, but
    # not keys, balances, routing fields, or generated outputs.
    allowed_portfolio = str(
        borrower.get("portfolio_field", "model_portfolio")
    )
    portfolio_conflicts = generated | (
        borrower_fields - {allowed_portfolio}
    )
    if portfolio_field in portfolio_conflicts or portfolio_field.startswith(
        CURRENT_CECL_RESERVED_PREFIXES
    ):
        raise ValueError(
            f"cecl.portfolio_field '{portfolio_field}' conflicts with a "
            "borrower or engine-owned field."
        )

    tags = scenario.get("tags", {})
    for tag in tags.values() if isinstance(tags, Mapping) else ():
        if not isinstance(tag, Mapping):
            continue
        assignments = tag.get("assign", {})
        if isinstance(assignments, Mapping) and reserve_field in assignments:
            raise ValueError(
                f"cecl.reserve_field '{reserve_field}' cannot also be a tag "
                "assignment target."
            )


def _nonblank_setting(
    config: Mapping[str, Any], key: str, path: str
) -> str:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"{path} must be nonblank.")
    return value


def _configured_field(
    config: Mapping[str, Any], key: str, default: str
) -> str:
    value = str(config.get(key, default)).strip()
    if not value:
        raise ValueError(f"cecl.reserve_basis.historical.{key} must be nonblank.")
    return value


def _positive_weight(value: Any, path: str) -> float:
    number = _finite_number(value, path)
    if number <= 0:
        raise ValueError(f"{path} must be greater than zero.")
    return number


def _period_specs(
    historical: Mapping[str, Any], enabled: bool
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not enabled:
        return {"name": "current", "weight": 1.0}, []
    current = historical.get("current_period", {})
    current_period = {
        "name": str(current.get("name", "current")).strip(),
        "weight": to_number(current.get("weight"), np.nan),
    }
    periods = [
        {
            "name": str(period.get("name", "")).strip(),
            "weight": to_number(period.get("weight"), np.nan),
        }
        for period in historical.get("periods", [])
        if isinstance(period, Mapping)
    ]
    return current_period, periods


def _prepare_history(
    history: pd.DataFrame | None,
    config: Mapping[str, Any],
    enabled: bool,
    exceptions: List[Dict[str, Any]],
    commercial_portfolios: set[str],
) -> tuple[Dict[tuple[str, str], Dict[str, Any]], str]:
    if not enabled:
        return {}, ""
    if not commercial_portfolios:
        # Portfolio history is a commercial-only feature. A Consumer-only run
        # must not require, validate, or log errors for a history table.
        return {}, ""
    source = str(config.get("source", "")).strip()
    portfolio_field = str(config.get("portfolio_field", "cecl_portfolio")).strip()
    period_field = str(config.get("period_field", "period")).strip()
    reserve_field = str(
        config.get("reserve_field", "historical_cecl_reserve")
    ).strip()
    if history is None:
        code = "CECL_HISTORY_SOURCE_UNAVAILABLE"
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            code,
            "The configured portfolio-level CECL history source is unavailable.",
            source=source,
        )
        return {}, code
    missing = [
        field
        for field in (portfolio_field, period_field, reserve_field)
        if field not in history.columns
    ]
    if missing:
        code = "CECL_HISTORY_FIELD_MISSING"
        for field in missing:
            record_exception(
                exceptions,
                "ERROR",
                "cecl",
                code,
                "A required portfolio-history field is missing.",
                source=source,
                field=field,
            )
        return {}, code

    configured_periods = {
        str(period.get("name", "")).strip()
        for period in config.get("periods", [])
        if isinstance(period, Mapping)
    }
    work = history[[portfolio_field, period_field, reserve_field]].copy()
    work["_portfolio_key"] = work[portfolio_field].map(_portfolio_key)
    work["_period_key"] = work[period_field].map(_period_key)
    blank = work["_portfolio_key"].eq("") | work["_period_key"].eq("")
    if blank.any():
        record_exception(
            exceptions,
            "WARNING",
            "cecl",
            "CECL_HISTORY_ROW_IGNORED",
            "Portfolio-history rows with blank portfolio or period values were ignored.",
            source=source,
            details=f"ignored_count={int(blank.sum())}",
        )
    relevant = (
        ~blank
        & work["_period_key"].isin(configured_periods)
        & work["_portfolio_key"].isin(commercial_portfolios)
    )
    work = work.loc[relevant].copy()
    duplicate = work.duplicated(["_portfolio_key", "_period_key"], keep=False)
    if duplicate.any():
        duplicate_keys = sorted(
            {
                f"{portfolio}/{period}"
                for portfolio, period in work.loc[
                    duplicate, ["_portfolio_key", "_period_key"]
                ].itertuples(index=False, name=None)
            }
        )
        raise ValueError(
            "CECL portfolio history must contain one unique row per "
            f"portfolio and period; duplicates: {', '.join(duplicate_keys)}."
        )
    lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for key, group in work.groupby(["_portfolio_key", "_period_key"], dropna=False):
        reserve = to_number(group[reserve_field].iloc[0], np.nan)
        if not np.isfinite(reserve) or reserve < 0:
            lookup[key] = {
                "status": "unavailable",
                "exception_code": "CECL_HISTORY_RESERVE_INVALID",
            }
            continue
        lookup[key] = {
            "status": "available",
            "exception_code": "",
            "reserve": float(reserve),
        }
    return lookup, ""


def _historical_period_values(
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    global_code: str,
    portfolio: str,
    period: str,
    current_portfolio_balance: float,
    invalid_portfolio_balance_count: int,
) -> Dict[str, Any]:
    item = lookup.get((portfolio, _period_key(period)))
    if global_code:
        return _unavailable_period(global_code, current_portfolio_balance)
    if item is None:
        return _unavailable_period(
            "CECL_HISTORY_PERIOD_MISSING", current_portfolio_balance
        )
    if item.get("status") != "available":
        return _unavailable_period(
            str(item.get("exception_code") or "CECL_HISTORY_RESERVE_INVALID"),
            current_portfolio_balance,
        )
    if (
        invalid_portfolio_balance_count
        or not np.isfinite(current_portfolio_balance)
        or current_portfolio_balance <= 0
    ):
        return _unavailable_period(
            "CECL_HISTORY_ALLOCATION_BALANCE_INVALID",
            current_portfolio_balance,
        )
    reserve = float(item["reserve"])
    ratio = reserve / current_portfolio_balance
    if not np.isfinite(ratio):
        return _unavailable_period(
            "CECL_HISTORY_RATIO_NONFINITE", current_portfolio_balance
        )
    return {
        "observation_count": 1,
        "included_observation_count": 1,
        "excluded_observation_count": 0,
        "missing_reserve_count": 0,
        "balance": current_portfolio_balance,
        "reserve": reserve,
        "raw_reserve_ratio": ratio,
        "raw_mean_reserve_ratio": np.nan,
        "raw_std_reserve_ratio": np.nan,
        "period_reserve_ratio": ratio,
        "status": "available",
        "exception_code": "",
    }


def _unavailable_period(code: str, balance: float) -> Dict[str, Any]:
    return {
        "observation_count": 0,
        "included_observation_count": 0,
        "excluded_observation_count": 0,
        "missing_reserve_count": 1,
        "balance": balance,
        "reserve": np.nan,
        "raw_reserve_ratio": np.nan,
        "raw_mean_reserve_ratio": np.nan,
        "raw_std_reserve_ratio": np.nan,
        "period_reserve_ratio": np.nan,
        "status": "unavailable",
        "exception_code": code,
    }


def _audit_row(
    portfolio: Any,
    bucket: str,
    reserve_basis: str,
    current_method: str,
    history_enabled: bool,
    period_method: str,
    period: Mapping[str, Any],
    source: str,
    reserve_field: str,
    observation_grain: str,
    allocation_balance_source: str,
    invalid_balance_count: int,
    values: Mapping[str, Any],
) -> Dict[str, Any]:
    weight = float(period["weight"])
    ratio = values.get("period_reserve_ratio", np.nan)
    return {
        "portfolio": portfolio,
        "bucket": bucket,
        "reserve_basis": reserve_basis,
        "current_method": current_method,
        "history_enabled": history_enabled,
        "period_method": period_method,
        "period": period["name"],
        "source": source,
        "reserve_field": reserve_field,
        "weight": weight,
        "observation_grain": observation_grain,
        **values,
        "invalid_balance_count": invalid_balance_count,
        "allocation_balance_source": allocation_balance_source,
        "weighted_ratio_component": (
            float(ratio) * weight if pd.notna(ratio) else np.nan
        ),
        "effective_reserve_ratio": np.nan,
        "basis_status": "unavailable",
        "basis_exception_code": "CECL_BASIS_PERIOD_UNAVAILABLE",
    }


def _borrower_observations(
    group: pd.DataFrame,
    borrower_field: str,
    balance_field: str,
    reserve_field: str,
) -> pd.DataFrame:
    work = pd.DataFrame(index=group.index)
    work["balance"] = pd.to_numeric(group[balance_field], errors="coerce")
    work["reserve_raw"] = _finite_numeric(group[reserve_field])
    work["reserve"] = work["reserve_raw"].fillna(0.0)
    missing_count_field = reserve_missing_count_field(reserve_field)
    if missing_count_field in group.columns:
        work["missing_reserve_count"] = pd.to_numeric(
            group[missing_count_field], errors="coerce"
        ).fillna(0.0)
    else:
        work["missing_reserve_count"] = work["reserve_raw"].isna().astype(int)
    if borrower_field in group.columns:
        borrower = group[borrower_field].astype(object)
        missing = borrower.isna() | borrower.astype(str).str.strip().eq("")
        fallback = pd.Series(group.index, index=group.index).map(
            lambda value: f"__row_{value}"
        )
        work["borrower"] = borrower.where(~missing, fallback)
    else:
        work["borrower"] = [f"__row_{index}" for index in group.index]
    return (
        work.groupby("borrower", dropna=False)
        .agg(
            balance=("balance", "sum"),
            reserve=("reserve", "sum"),
            missing_reserve_count=("missing_reserve_count", "sum"),
        )
        .reset_index(drop=True)
    )


def _calculate_current_period(
    observations: pd.DataFrame,
    method: str,
    threshold: float,
    group_balance: float,
) -> Dict[str, Any]:
    reserve = _finite_fsum(observations["reserve"])
    raw_ratio = pct(reserve, group_balance) if pd.notna(reserve) else np.nan
    valid = observations[
        pd.to_numeric(observations["balance"], errors="coerce").gt(0)
        & pd.to_numeric(observations["reserve"], errors="coerce").notna()
    ].copy()
    valid["ratio"] = valid["reserve"] / valid["balance"]
    valid = valid[np.isfinite(valid["ratio"])]
    observation_count = int(len(valid))
    mean = float(valid["ratio"].mean()) if observation_count else np.nan
    std = float(valid["ratio"].std(ddof=0)) if observation_count else np.nan
    if method == "central_tendency":
        if observation_count == 0:
            included = valid
        elif observation_count < 2 or std == 0.0:
            included = valid
        else:
            z_scores = (valid["ratio"] - mean) / std
            included = valid[z_scores.abs() <= threshold]
        ratio = (
            float(included["ratio"].mean()) if not included.empty else np.nan
        )
    else:
        included = valid
        ratio = raw_ratio
    status = "available" if pd.notna(ratio) else "unavailable"
    return {
        "observation_count": observation_count,
        "included_observation_count": int(len(included)),
        "excluded_observation_count": observation_count - int(len(included)),
        "missing_reserve_count": int(observations["missing_reserve_count"].sum()),
        "balance": group_balance,
        "reserve": reserve,
        "raw_reserve_ratio": raw_ratio,
        "raw_mean_reserve_ratio": mean,
        "raw_std_reserve_ratio": std,
        "period_reserve_ratio": ratio,
        "status": status,
        "exception_code": (
            "" if status == "available" else "CECL_BASIS_PERIOD_UNAVAILABLE"
        ),
    }


def _row_basis_methods(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    portfolio_field: str,
    commercial_method: str,
) -> pd.Series:
    methods = pd.Series(commercial_method, index=results.index, dtype=object)
    mask = _consumer_row_mask(results, scenario, portfolio_field)
    methods.loc[mask] = "in_place"
    return methods


def _consumer_portfolios(
    frame: pd.DataFrame,
    scenario: Mapping[str, Any],
    portfolio_field: str,
) -> set[str]:
    configured = scenario.get("cecl", {}).get("portfolios", {})
    portfolios = {
        _portfolio_key(name)
        for name, spec in configured.items()
        if isinstance(spec, Mapping) and spec.get("method") == "expected_loss"
    } if isinstance(configured, Mapping) else set()
    if portfolio_field in frame.columns:
        mask = _consumer_row_mask(frame, scenario, portfolio_field)
        portfolios.update(frame.loc[mask, portfolio_field].map(_portfolio_key))
    return portfolios


def _consumer_row_mask(
    frame: pd.DataFrame,
    scenario: Mapping[str, Any],
    portfolio_field: str,
) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    configured = scenario.get("cecl", {}).get("portfolios", {})
    expected_loss = {
        _portfolio_key(name)
        for name, spec in configured.items()
        if isinstance(spec, Mapping) and spec.get("method") == "expected_loss"
    } if isinstance(configured, Mapping) else set()
    if portfolio_field in frame.columns and expected_loss:
        mask |= frame[portfolio_field].map(_portfolio_key).isin(expected_loss)
    for field in ("primary_module", "model_module"):
        if field in frame.columns:
            mask |= frame[field].astype(str).str.strip().str.casefold().eq("consumer")
    if "module_applied" in frame.columns:
        mask |= frame["module_applied"].astype(str).str.contains(
            "Consumer", case=False, na=False
        )
    return mask


def _portfolio_key(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _normalized_portfolios(values: pd.Series) -> pd.Series:
    """Strip CECL portfolio keys while preserving genuine missing values."""
    return values.map(
        lambda value: value if pd.isna(value) else str(value).strip()
    )


def _period_key(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _finite_number(value: Any, path: str) -> float:
    number = to_number(value, np.nan)
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number):
        raise ValueError(f"{path} must be numeric and finite.")
    return float(number)


def _finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def _finite_fsum(values: Iterable[float]) -> float:
    try:
        total = math.fsum(float(value) for value in values)
    except (OverflowError, TypeError, ValueError):
        return np.nan
    return total if np.isfinite(total) else np.nan


def _invalid_balance_count(group: pd.DataFrame, balance_field: str) -> int:
    balances = pd.to_numeric(group[balance_field], errors="coerce")
    direct_count = int(((~np.isfinite(balances)) | balances.lt(0)).sum())
    if INVALID_BALANCE_COUNT_FIELD not in group.columns:
        return direct_count
    carried = pd.to_numeric(
        group[INVALID_BALANCE_COUNT_FIELD], errors="coerce"
    )
    carried = carried.where(np.isfinite(carried), 0.0).clip(lower=0.0)
    return max(direct_count, int(carried.sum()))


def _exception_exists(
    exceptions: Iterable[Mapping[str, Any]], code: str, field: str
) -> bool:
    return any(
        str(item.get("code", "")) == code
        and str(item.get("field", "")) == field
        for item in exceptions
    )


def _unavailable_basis(
    method: str,
    current_method: str,
    history_enabled: bool,
    effective: pd.Series,
    method_by_row: pd.Series,
    required_fields: tuple[str, ...],
    code: str,
) -> CeclReserveBasis:
    return CeclReserveBasis(
        method=method,
        current_method=current_method,
        history_enabled=history_enabled,
        effective_reserve=effective,
        method_by_row=method_by_row,
        ratios=pd.DataFrame(columns=RATIO_COLUMNS),
        audit=pd.DataFrame(columns=AUDIT_COLUMNS),
        required_fields=required_fields,
        exception_code=code,
    )
