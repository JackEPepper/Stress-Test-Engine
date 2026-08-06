"""Configurable current and CECL-level tag-history reserve bases."""

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
DEFAULT_ZERO_BALANCE_TOLERANCE = 1e-9
DEFAULT_Z_SCORE_THRESHOLD = 2.0
CECL_LEVEL_TAG_FIELD = "cecl_level_tag"
CANONICAL_BUCKETS = ("Pass", "Special Mention", "Substandard")
VALID_BUCKETS = frozenset((*CANONICAL_BUCKETS, "Unknown"))
HISTORY_SKIP_TOKENS = frozenset({"n/a", "#n/a"})
EFFECTIVE_RESERVE_FIELD = "cecl_effective_reserve_base"
RESERVE_BASIS_METHOD_FIELD = "cecl_reserve_basis_method"
INVALID_BALANCE_COUNT_FIELD = "_cecl_invalid_balance_count"
RESERVED_INPUT_FIELDS = {
    "_source_file",
    "_source_file_row",
    "_source_row",
    "_portfolio_key",
    "_period_key",
    "_tag_key",
    "_bucket_key",
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
    CECL_LEVEL_TAG_FIELD,
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
    "_raw_invalid_numeric__",
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
    CECL_LEVEL_TAG_FIELD,
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
    CECL_LEVEL_TAG_FIELD,
    "bucket",
    "reserve_basis",
    "current_method",
    "history_enabled",
    "period_method",
    "period",
    "source",
    "reserve_field",
    "weight",
    "effective_weight",
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
    out[EFFECTIVE_RESERVE_FIELD] = basis.effective_reserve.to_numpy()
    out[RESERVE_BASIS_METHOD_FIELD] = basis.method_by_row.to_numpy()
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
    zero_balance_tolerance = _finite_number(
        cecl.get(
            "zero_balance_tolerance", DEFAULT_ZERO_BALANCE_TOLERANCE
        ),
        "cecl.zero_balance_tolerance",
    )
    if zero_balance_tolerance < 0:
        raise ValueError(
            "cecl.zero_balance_tolerance must be greater than or equal to zero."
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
            "use reserve_basis.historical with a CECL-level tag input source."
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

    if not _configured_cecl_level_tags(scenario):
        raise ValueError(
            "Historical CECL ratios require at least one CECL-level scenario "
            "tag with cecl_level set to true."
        )

    legacy_fields = sorted(
        key for key in ("portfolio_field", "reserve_field") if key in historical
    )
    if legacy_fields:
        raise ValueError(
            "Portfolio-level historical CECL reserve inputs are no longer "
            "supported. Replace "
            f"{', '.join(legacy_fields)} with tag_field, bucket_field, and "
            "ratio_field, and provide one CECL ratio per tag, period, and "
            "risk bucket."
        )

    source = _nonblank_setting(
        historical,
        "source",
        "cecl.reserve_basis.historical.source",
    )
    tag_field = _configured_field(historical, "tag_field", "cecl_tag")
    period_field = _configured_field(
        historical, "period_field", "period"
    )
    bucket_field = _configured_field(
        historical, "bucket_field", "risk_bucket"
    )
    ratio_field = _configured_field(
        historical, "ratio_field", "historical_cecl_ratio"
    )
    historical_fields = {tag_field, period_field, bucket_field, ratio_field}
    if len(historical_fields) != 4:
        raise ValueError(
            "CECL historical tag, period, risk-bucket, and ratio fields must "
            "be distinct."
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
        if (
            "reserve_field" in period
            or "ratio_field" in period
            or "period_method" in period
        ):
            raise ValueError(
                f"{path} must reference a tag-ratio CSV period by name and weight only."
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
    dates = {str(value) for value in source_spec.get("date_columns", [])}
    expected = {tag_field, period_field, bucket_field, ratio_field}
    missing_aliases = sorted(expected - canonical)
    missing_required = sorted(expected - required)
    missing_numeric = sorted({ratio_field} - numeric)
    missing_strings = sorted({tag_field, period_field, bucket_field} - strings)
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
            "The CECL historical ratio field must be numeric; missing: "
            f"{', '.join(missing_numeric)}."
        )
    if ratio_field in dates:
        raise ValueError(
            "The CECL historical ratio field cannot also be a date column."
        )
    if missing_strings:
        raise ValueError(
            "CECL historical tag, period, and risk-bucket fields must be "
            "string_columns; "
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


def zero_balance_tolerance(scenario: Mapping[str, Any]) -> float:
    """Return the validated CECL/model balance rounding tolerance.

    Scenario validation rejects malformed values during normal engine runs.
    The defensive fallback keeps lower-level programmatic callers from
    crashing if they bypass that validation.
    """
    cecl = scenario.get("cecl", {})
    if not isinstance(cecl, Mapping):
        return DEFAULT_ZERO_BALANCE_TOLERANCE
    tolerance = to_number(
        cecl.get(
            "zero_balance_tolerance", DEFAULT_ZERO_BALANCE_TOLERANCE
        ),
        DEFAULT_ZERO_BALANCE_TOLERANCE,
    )
    if not np.isfinite(tolerance) or tolerance < 0:
        return DEFAULT_ZERO_BALANCE_TOLERANCE
    return float(tolerance)


def normalized_balance_values(
    frame: pd.DataFrame, scenario: Mapping[str, Any]
) -> pd.Series:
    """Coerce balances and normalize rounding-size values to exact zero."""
    balance_field = str(
        scenario.get("borrower", {}).get(
            "balance_field", "outstanding_balance"
        )
    )
    if balance_field not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    balances = _coerced_balance_values(frame[balance_field])
    balances = balances.where(np.isfinite(balances))
    tolerance = zero_balance_tolerance(scenario)
    return balances.mask(balances.abs().le(tolerance), 0.0)


def invalid_balance_mask(
    frame: pd.DataFrame, scenario: Mapping[str, Any]
) -> pd.Series:
    """Identify exposure rows whose balance cannot enter the model."""
    balance_field = str(
        scenario.get("borrower", {}).get(
            "balance_field", "outstanding_balance"
        )
    )
    if balance_field not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    balances = _coerced_balance_values(frame[balance_field])
    tolerance = zero_balance_tolerance(scenario)
    return pd.Series(
        (~np.isfinite(balances)) | balances.lt(-tolerance),
        index=frame.index,
        dtype=bool,
    )


def _coerced_balance_values(values: pd.Series) -> pd.Series:
    """Return ordinary float values for NumPy and nullable pandas inputs."""
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        array = numeric.to_numpy(dtype=float, na_value=np.nan)
    except (AttributeError, TypeError, ValueError):
        array = np.asarray(numeric, dtype=float)
    result = pd.Series(array, index=values.index, dtype=float)
    boolean_values = values.map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    return result.mask(boolean_values)


def build_cecl_reserve_basis(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
    history: pd.DataFrame | None = None,
) -> CeclReserveBasis:
    """Resolve current and optional tag-and-bucket historical CECL ratios."""
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
    output_index = results.index.copy()
    basis_results = results.reset_index(drop=True).copy()
    if portfolio_field in basis_results.columns:
        basis_results[portfolio_field] = _normalized_portfolios(
            basis_results[portfolio_field]
        )
    if CECL_LEVEL_TAG_FIELD in basis_results.columns:
        basis_results[CECL_LEVEL_TAG_FIELD] = _normalized_portfolios(
            basis_results[CECL_LEVEL_TAG_FIELD]
        )
    effective = pd.Series(np.nan, index=basis_results.index, dtype=float)
    method_by_row = _row_basis_methods(
        basis_results, scenario, portfolio_field, method
    )

    configured_level_tags = _configured_cecl_level_tags(scenario)
    if history_enabled and not configured_level_tags:
        code = "CECL_LEVEL_TAG_CONFIGURATION_MISSING"
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            code,
            "Historical CECL ratios require at least one configured CECL-level tag.",
            field=CECL_LEVEL_TAG_FIELD,
        )
        return _basis_with_output_index(
            _unavailable_basis(
                method,
                current_method,
                history_enabled,
                effective,
                method_by_row,
                required_fields,
                code,
            ),
            output_index,
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
        return _basis_with_output_index(
            _unavailable_basis(
                method,
                current_method,
                history_enabled,
                effective,
                method_by_row,
                required_fields,
                "CECL_RESERVE_FIELD_MISSING",
            ),
            output_index,
        )

    reserve_scope = pd.Series(True, index=basis_results.index)
    if "model_excluded" in basis_results.columns:
        reserve_scope &= ~basis_results["model_excluded"].fillna(False).astype(
            bool
        )
    reserve_scope &= ~invalid_balance_mask(basis_results, scenario)
    missing_count = int(
        _finite_numeric(basis_results[current_field])
        .isna()
        .loc[reserve_scope]
        .sum()
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
        return _basis_with_output_index(
            _unavailable_basis(
                method,
                current_method,
                history_enabled,
                effective,
                method_by_row,
                required_fields,
                "CECL_BASIS_GROUP_FIELD_MISSING",
            ),
            output_index,
        )

    frame = basis_results.copy()
    if "model_excluded" in frame.columns:
        frame = frame[~frame["model_excluded"].fillna(False).astype(bool)].copy()
    invalid_balances = invalid_balance_mask(frame, scenario)
    if invalid_balances.any():
        record_exception(
            exceptions,
            "WARNING",
            "cecl",
            "CECL_BALANCE_EXCLUDED",
            "Rows with missing, invalid, nonfinite, or materially negative balances were excluded from CECL reserve-basis calibration.",
            field=balance_field,
            details=f"excluded_row_count={int(invalid_balances.sum())}",
        )
        frame = frame.loc[~invalid_balances].copy()
    frame[balance_field] = normalized_balance_values(frame, scenario)
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
        frame["base_bucket"] = frame["base_bucket"].map(_bucket_key)

    consumer_mask = _consumer_row_mask(frame, scenario, portfolio_field)
    if CECL_LEVEL_TAG_FIELD not in frame.columns and configured_level_tags:
        consumer_only = _consumer_row_mask(
            frame, scenario, portfolio_field
        ).all()
        if not consumer_only:
            code = "CECL_LEVEL_TAG_MISSING"
            record_exception(
                exceptions,
                "ERROR",
                "cecl",
                code,
                "The resolved CECL-level tag field is missing; reserve-basis calibration is unavailable.",
                field=CECL_LEVEL_TAG_FIELD,
            )
            return _basis_with_output_index(
                _unavailable_basis(
                    method,
                    current_method,
                    history_enabled,
                    effective,
                    method_by_row,
                    required_fields,
                    code,
                ),
                output_index,
            )
        frame[CECL_LEVEL_TAG_FIELD] = frame[portfolio_field]
    if CECL_LEVEL_TAG_FIELD not in frame.columns:
        # Backward compatibility for programmatic callers that predate tag
        # routing. This is permitted only when the scenario has not opted into
        # CECL-level tags or tag history.
        frame[CECL_LEVEL_TAG_FIELD] = frame[portfolio_field]
    frame.loc[consumer_mask, CECL_LEVEL_TAG_FIELD] = frame.loc[
        consumer_mask, portfolio_field
    ]
    missing_level_tag = (
        ~consumer_mask
        & (
            frame[CECL_LEVEL_TAG_FIELD].isna()
            | frame[CECL_LEVEL_TAG_FIELD].astype(str).str.strip().eq("")
        )
    )
    if missing_level_tag.any():
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            "CECL_LEVEL_TAG_MISSING",
            "Commercial rows without a resolved CECL-level tag were excluded from reserve-basis calibration.",
            field=CECL_LEVEL_TAG_FIELD,
            details=f"missing_count={int(missing_level_tag.sum())}",
        )
        frame = frame.loc[~missing_level_tag].copy()
        consumer_mask = consumer_mask.reindex(frame.index, fill_value=False)

    consumer_frame = frame.loc[consumer_mask].copy()
    commercial_frame = frame.loc[~consumer_mask].copy()
    commercial_tags = {
        _level_tag_key(value)
        for value in commercial_frame[CECL_LEVEL_TAG_FIELD]
        if _level_tag_key(value)
    }
    history_lookup, history_global_code = _prepare_history(
        history,
        historical,
        history_enabled,
        exceptions,
        commercial_tags,
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

    def process_group(
        portfolio: Any,
        cecl_level_tag: Any,
        bucket_text: str,
        group: pd.DataFrame,
        *,
        is_consumer: bool,
        basis_invalid_balance_count: int,
    ) -> None:
        current_only = is_consumer or bucket_text == "Unknown"
        group_current_method = "in_place" if current_only else current_method
        group_history_enabled = history_enabled and not current_only
        group_method = _basis_label(
            group_current_method, group_history_enabled
        )
        method_by_row.loc[group.index] = group_method
        group_current_period = (
            {"name": "current", "weight": 1.0}
            if is_consumer or not group_history_enabled
            else current_period
        )
        finite_balances = _finite_numeric(group[balance_field])
        eligible_group = group.loc[finite_balances.notna()]
        group_balance = float(finite_balances.sum())

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
                cecl_level_tag,
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
                basis_invalid_balance_count,
                current_values,
            )
        ]

        if group_history_enabled:
            for period in historical_periods:
                values = _historical_period_values(
                    history_lookup,
                    history_global_code,
                    _level_tag_key(cecl_level_tag),
                    bucket_text,
                    period["name"],
                    group_balance,
                )
                selected.append((period, values))
                group_audit.append(
                    _audit_row(
                        portfolio,
                        cecl_level_tag,
                        bucket_text,
                        group_method,
                        group_current_method,
                        True,
                        "tag_bucket_history",
                        period,
                        str(historical.get("source", "")),
                        str(
                            historical.get(
                                "ratio_field", "historical_cecl_ratio"
                            )
                        ),
                        "cecl_level_tag_risk_bucket",
                        "current_cecl_tag_bucket",
                        basis_invalid_balance_count,
                        values,
                    )
                )

        _apply_effective_period_weights(selected, group_audit)
        for period, values in selected:
            if values["status"] in {"available", "skipped"}:
                continue
            code = str(values.get("exception_code") or "CECL_BASIS_PERIOD_UNAVAILABLE")
            # Empty current cells are retained so a later stressed migration
            # can identify a missing ratio. They are not errors unless they
            # currently carry balance; reporting performs the stressed check.
            if group_balance > 0:
                record_exception(
                    exceptions,
                    "ERROR",
                    "cecl",
                    code,
                    "A configured CECL basis period is unavailable.",
                    portfolio=portfolio,
                    bucket=bucket_text,
                    field=(
                        current_field
                        if period is group_current_period
                        else str(
                            historical.get(
                                "ratio_field", "historical_cecl_ratio"
                            )
                        )
                    ),
                    details=(
                        f"cecl_level_tag={_level_tag_key(cecl_level_tag)}; "
                        f"period={period['name']}"
                    ),
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
            values for _, values in selected if values["status"] == "unavailable"
        ]
        skipped_periods = [
            period
            for period, values in selected
            if values["status"] == "skipped"
        ]
        if unavailable:
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
                row["effective_weight"] * values["period_reserve_ratio"]
                for (_, values), row in zip(selected, group_audit)
                if values["status"] == "available"
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

        if status == "available" and skipped_periods:
            skipped_names = ", ".join(
                str(period["name"]) for period in skipped_periods
            )
            record_exception(
                exceptions,
                "WARNING",
                "cecl",
                "CECL_HISTORY_RATIO_SKIPPED_REWEIGHTED",
                "An N/A historical CECL ratio cell was skipped and the remaining period weights for its tag and bucket were normalized to 100%.",
                portfolio=portfolio,
                bucket=bucket_text,
                field=str(
                    historical.get(
                        "ratio_field", "historical_cecl_ratio"
                    )
                ),
                details=(
                    f"cecl_level_tag={_level_tag_key(cecl_level_tag)}; "
                    f"skipped_periods={skipped_names}"
                ),
            )

        ratio_rows.append(
            {
                "portfolio": portfolio,
                CECL_LEVEL_TAG_FIELD: cecl_level_tag,
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

    for (portfolio, bucket), group in consumer_frame.groupby(
        [portfolio_field, "base_bucket"], dropna=False, sort=False
    ):
        bucket_text = _bucket_key(bucket)
        if bucket_text not in VALID_BUCKETS:
            continue
        process_group(
            portfolio,
            portfolio,
            bucket_text,
            group,
            is_consumer=True,
            basis_invalid_balance_count=0,
        )

    for cecl_level_tag, tag_group in commercial_frame.groupby(
        CECL_LEVEL_TAG_FIELD, dropna=False, sort=False
    ):
        portfolio = _portfolio_label(tag_group[portfolio_field])
        observed_buckets = [
            bucket
            for bucket in (*CANONICAL_BUCKETS, "Unknown")
            if tag_group["base_bucket"].eq(bucket).any()
        ]
        buckets = list(CANONICAL_BUCKETS) if history_enabled else observed_buckets
        if "Unknown" in observed_buckets and "Unknown" not in buckets:
            buckets.append("Unknown")
        for bucket_text in buckets:
            group = tag_group.loc[
                tag_group["base_bucket"].eq(bucket_text)
            ].copy()
            process_group(
                portfolio,
                cecl_level_tag,
                bucket_text,
                group,
                is_consumer=False,
                basis_invalid_balance_count=0,
            )

    _warn_decreasing_effective_ladders(
        ratio_rows,
        commercial_frame,
        exceptions,
    )

    return _basis_with_output_index(
        CeclReserveBasis(
            method=method,
            current_method=current_method,
            history_enabled=history_enabled,
            effective_reserve=effective,
            method_by_row=method_by_row,
            ratios=pd.DataFrame(ratio_rows, columns=RATIO_COLUMNS),
            audit=pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS),
            required_fields=required_fields,
        ),
        output_index,
    )


def _warn_decreasing_effective_ladders(
    ratio_rows: List[Dict[str, Any]],
    commercial_frame: pd.DataFrame,
    exceptions: List[Dict[str, Any]],
) -> None:
    """Warn without changing results when an applied CECL ladder decreases."""
    commercial_tags = {
        _level_tag_key(value)
        for value in commercial_frame.get(
            CECL_LEVEL_TAG_FIELD, pd.Series(dtype=object)
        )
        if _level_tag_key(value)
    }
    by_tag: Dict[str, Dict[str, float]] = {}
    for row in ratio_rows:
        tag = _level_tag_key(row.get(CECL_LEVEL_TAG_FIELD))
        bucket = _bucket_key(row.get("bucket"))
        ratio = to_number(row.get("reserve_ratio"), np.nan)
        if (
            tag in commercial_tags
            and bucket in CANONICAL_BUCKETS
            and row.get("status") == "available"
            and np.isfinite(ratio)
        ):
            by_tag.setdefault(tag, {})[bucket] = float(ratio)

    decreasing_ladders: Dict[
        str, tuple[List[tuple[str, float]], List[tuple[str, str]]]
    ] = {}
    for tag, ratios in by_tag.items():
        ordered = [
            (bucket, ratios[bucket])
            for bucket in CANONICAL_BUCKETS
            if bucket in ratios
        ]
        decreases = [
            (earlier_bucket, later_bucket)
            for (earlier_bucket, earlier), (later_bucket, later) in zip(
                ordered, ordered[1:]
            )
            if later + WEIGHT_TOLERANCE < earlier
        ]
        if decreases:
            decreasing_ladders[tag] = (ordered, decreases)
    if not decreasing_ladders:
        return

    code = "CECL_RESERVE_RATIO_LADDER_INVALID"
    for tag in sorted(decreasing_ladders):
        ordered, decreases = decreasing_ladders[tag]
        ratio_details = ", ".join(
            f"{bucket}={ratio:.12g}" for bucket, ratio in ordered
        )
        decrease_details = ", ".join(
            f"{earlier}>{later}" for earlier, later in decreases
        )
        record_exception(
            exceptions,
            "WARNING",
            "cecl",
            code,
            "The applied commercial CECL ratio ladder decreases as credit quality worsens; calculated ratios were retained for case-by-case review.",
            details=(
                f"cecl_level_tag={tag}; ratios={ratio_details}; "
                f"decreases={decrease_details}"
            ),
        )


def _basis_config(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    cecl = scenario.get("cecl", {})
    if not isinstance(cecl, Mapping):
        return {}
    basis = cecl.get("reserve_basis", {})
    return basis if isinstance(basis, Mapping) else {}


def _configured_cecl_level_tags(scenario: Mapping[str, Any]) -> set[str]:
    tags = scenario.get("tags", {})
    if not isinstance(tags, Mapping):
        return set()
    return {
        str(name).strip()
        for name, spec in tags.items()
        if isinstance(spec, Mapping)
        and spec.get("cecl_level") is True
        and str(name).strip()
    }


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
        f"{current_method}+tag_bucket_history"
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
    commercial_tags: set[str],
) -> tuple[Dict[tuple[str, str, str], Dict[str, Any]], str]:
    if not enabled:
        return {}, ""
    if not commercial_tags:
        # Tag history is a commercial-only feature. A Consumer-only run
        # must not require, validate, or log errors for a history table.
        return {}, ""
    source = str(config.get("source", "")).strip()
    tag_field = str(config.get("tag_field", "cecl_tag")).strip()
    period_field = str(config.get("period_field", "period")).strip()
    bucket_field = str(config.get("bucket_field", "risk_bucket")).strip()
    ratio_field = str(
        config.get("ratio_field", "historical_cecl_ratio")
    ).strip()
    if history is None:
        code = "CECL_HISTORY_SOURCE_UNAVAILABLE"
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            code,
            "The configured CECL tag-ratio history source is unavailable.",
            source=source,
        )
        return {}, code
    missing = [
        field
        for field in (tag_field, period_field, bucket_field, ratio_field)
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
                "A required CECL tag-history field is missing.",
                source=source,
                field=field,
            )
        return {}, code

    configured_periods = {
        str(period.get("name", "")).strip()
        for period in config.get("periods", [])
        if isinstance(period, Mapping)
    }
    work = history[
        [tag_field, period_field, bucket_field, ratio_field]
    ].copy()
    work["_tag_key"] = work[tag_field].map(_level_tag_key)
    work["_period_key"] = work[period_field].map(_period_key)
    work["_bucket_key"] = work[bucket_field].map(_bucket_key)
    blank = (
        work["_tag_key"].eq("")
        | work["_period_key"].eq("")
        | work["_bucket_key"].eq("")
    )
    if blank.any():
        record_exception(
            exceptions,
            "WARNING",
            "cecl",
            "CECL_HISTORY_ROW_IGNORED",
            "CECL tag-history rows with blank tag, period, or risk-bucket values were ignored.",
            source=source,
            details=f"ignored_count={int(blank.sum())}",
        )
    relevant = (
        ~blank
        & work["_period_key"].isin(configured_periods)
        & work["_tag_key"].isin(commercial_tags)
    )
    work = work.loc[relevant].copy()
    invalid_buckets = sorted(
        set(work.loc[~work["_bucket_key"].isin(CANONICAL_BUCKETS), "_bucket_key"])
    )
    if invalid_buckets:
        raise ValueError(
            "CECL tag history risk buckets must be Pass, Special Mention, "
            "or Substandard; invalid values: "
            f"{', '.join(invalid_buckets)}."
        )
    duplicate = work.duplicated(
        ["_tag_key", "_period_key", "_bucket_key"], keep=False
    )
    if duplicate.any():
        duplicate_keys = sorted(
            {
                f"{tag}/{period}/{bucket}"
                for tag, period, bucket in work.loc[
                    duplicate,
                    ["_tag_key", "_period_key", "_bucket_key"],
                ].itertuples(index=False, name=None)
            }
        )
        raise ValueError(
            "CECL tag history must contain one unique row per tag, period, "
            f"and risk bucket; duplicates: {', '.join(duplicate_keys)}."
        )
    lookup: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for key, group in work.groupby(
        ["_tag_key", "_bucket_key", "_period_key"],
        dropna=False,
        sort=False,
    ):
        raw_ratio = group[ratio_field].iloc[0]
        if _is_history_skip_token(raw_ratio):
            lookup[key] = {
                "status": "skipped",
                "exception_code": "",
            }
            continue
        ratio = to_number(raw_ratio, np.nan)
        if (
            isinstance(raw_ratio, (bool, np.bool_))
            or not np.isfinite(ratio)
            or ratio < 0
            or ratio > 1
        ):
            lookup[key] = {
                "status": "unavailable",
                "exception_code": "CECL_HISTORY_RATIO_INVALID",
            }
            continue
        lookup[key] = {
            "status": "available",
            "exception_code": "",
            "ratio": float(ratio),
        }

    # Preserve decreasing historical ladders for case-by-case review, but make
    # the exact source values and transitions visible in the exception audit.
    grouped_keys: Dict[tuple[str, str], Dict[str, float]] = {}
    for (tag, bucket, period), item in lookup.items():
        if item.get("status") == "available":
            grouped_keys.setdefault((tag, period), {})[bucket] = float(
                item["ratio"]
            )
    for (tag, period), ratios in grouped_keys.items():
        ordered = [
            (bucket, ratios[bucket])
            for bucket in CANONICAL_BUCKETS
            if bucket in ratios
        ]
        decreases = [
            (earlier_bucket, later_bucket)
            for (earlier_bucket, earlier_ratio), (
                later_bucket,
                later_ratio,
            ) in zip(ordered, ordered[1:])
            if later_ratio + WEIGHT_TOLERANCE < earlier_ratio
        ]
        if decreases:
            ratio_details = ", ".join(
                f"{bucket}={ratio:.12g}" for bucket, ratio in ordered
            )
            decrease_details = ", ".join(
                f"{earlier}>{later}" for earlier, later in decreases
            )
            record_exception(
                exceptions,
                "WARNING",
                "cecl",
                "CECL_HISTORY_RATIO_LADDER_INVALID",
                "A historical CECL ratio ladder decreases as credit quality worsens; supplied ratios were retained for case-by-case review.",
                source=source,
                details=(
                    f"cecl_level_tag={tag}; period={period}; "
                    f"ratios={ratio_details}; decreases={decrease_details}"
                ),
            )
    return lookup, ""


def _historical_period_values(
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    global_code: str,
    cecl_level_tag: str,
    bucket: str,
    period: str,
    current_bucket_balance: float,
) -> Dict[str, Any]:
    item = lookup.get(
        (
            _level_tag_key(cecl_level_tag),
            _bucket_key(bucket),
            _period_key(period),
        )
    )
    if global_code:
        return _unavailable_period(global_code, current_bucket_balance)
    if item is None:
        return _unavailable_period(
            "CECL_HISTORY_TAG_BUCKET_PERIOD_MISSING", current_bucket_balance
        )
    if item.get("status") == "skipped":
        return _skipped_period(current_bucket_balance)
    if item.get("status") != "available":
        return _unavailable_period(
            str(item.get("exception_code") or "CECL_HISTORY_RATIO_INVALID"),
            current_bucket_balance,
        )
    ratio = float(item["ratio"])
    if not np.isfinite(ratio) or ratio < 0 or ratio > 1:
        return _unavailable_period(
            "CECL_HISTORY_RATIO_INVALID", current_bucket_balance
        )
    reserve = (
        current_bucket_balance * ratio
        if np.isfinite(current_bucket_balance)
        else np.nan
    )
    return {
        "observation_count": 1,
        "included_observation_count": 1,
        "excluded_observation_count": 0,
        "missing_reserve_count": 0,
        "balance": current_bucket_balance,
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


def _skipped_period(balance: float) -> Dict[str, Any]:
    return {
        "observation_count": 1,
        "included_observation_count": 0,
        "excluded_observation_count": 1,
        "missing_reserve_count": 0,
        "balance": balance,
        "reserve": np.nan,
        "raw_reserve_ratio": np.nan,
        "raw_mean_reserve_ratio": np.nan,
        "raw_std_reserve_ratio": np.nan,
        "period_reserve_ratio": np.nan,
        "status": "skipped",
        "exception_code": "",
    }


def _apply_effective_period_weights(
    selected: List[tuple[Dict[str, Any], Dict[str, Any]]],
    audit_rows: List[Dict[str, Any]],
) -> None:
    unavailable = any(
        values.get("status") == "unavailable" for _, values in selected
    )
    retained_weight = _finite_fsum(
        period["weight"]
        for period, values in selected
        if values.get("status") == "available"
    )
    can_normalize = (
        not unavailable
        and np.isfinite(retained_weight)
        and retained_weight > 0
    )
    for (period, values), row in zip(selected, audit_rows):
        period_status = values.get("status")
        if period_status == "skipped":
            effective_weight = 0.0
            component = 0.0
        elif not can_normalize:
            effective_weight = np.nan
            component = np.nan
        elif period_status == "available":
            effective_weight = float(period["weight"]) / retained_weight
            component = (
                effective_weight * float(values["period_reserve_ratio"])
            )
        else:
            effective_weight = np.nan
            component = np.nan
        row["effective_weight"] = effective_weight
        row["weighted_ratio_component"] = component


def _audit_row(
    portfolio: Any,
    cecl_level_tag: Any,
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
        CECL_LEVEL_TAG_FIELD: cecl_level_tag,
        "bucket": bucket,
        "reserve_basis": reserve_basis,
        "current_method": current_method,
        "history_enabled": history_enabled,
        "period_method": period_method,
        "period": period["name"],
        "source": source,
        "reserve_field": reserve_field,
        "weight": weight,
        "effective_weight": np.nan,
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


def _level_tag_key(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _bucket_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    canonical = {
        "pass": "Pass",
        "special mention": "Special Mention",
        "substandard": "Substandard",
        "unknown": "Unknown",
    }
    return canonical.get(text.casefold(), text)


def _portfolio_label(values: pd.Series) -> str:
    portfolios = sorted(
        {
            _portfolio_key(value)
            for value in values
            if _portfolio_key(value)
        }
    )
    return " | ".join(portfolios)


def _normalized_portfolios(values: pd.Series) -> pd.Series:
    """Strip CECL portfolio keys while preserving genuine missing values."""
    return values.map(
        lambda value: value if pd.isna(value) else str(value).strip()
    )


def _period_key(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _is_history_skip_token(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().casefold() in HISTORY_SKIP_TOKENS


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


def _exception_exists(
    exceptions: Iterable[Mapping[str, Any]], code: str, field: str
) -> bool:
    return any(
        str(item.get("code", "")) == code
        and str(item.get("field", "")) == field
        for item in exceptions
    )


def _basis_with_output_index(
    basis: CeclReserveBasis, output_index: pd.Index
) -> CeclReserveBasis:
    """Restore caller row labels after position-safe internal calculations."""
    return CeclReserveBasis(
        method=basis.method,
        current_method=basis.current_method,
        history_enabled=basis.history_enabled,
        effective_reserve=pd.Series(
            basis.effective_reserve.to_numpy(), index=output_index, dtype=float
        ),
        method_by_row=pd.Series(
            basis.method_by_row.to_numpy(), index=output_index, dtype=object
        ),
        ratios=basis.ratios,
        audit=basis.audit,
        required_fields=basis.required_fields,
        exception_code=basis.exception_code,
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
