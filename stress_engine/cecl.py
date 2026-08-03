"""Configurable CECL reserve-basis calculation and audit tables."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .exceptions import record_exception
from .utils import pct, to_number


RESERVE_BASIS_METHODS = {"in_place", "central_tendency", "weighted_history"}
PERIOD_METHODS = {"in_place", "central_tendency"}
WEIGHT_TOLERANCE = 1e-9
DEFAULT_Z_SCORE_THRESHOLD = 2.0
EFFECTIVE_RESERVE_FIELD = "cecl_effective_reserve_base"
RESERVE_BASIS_METHOD_FIELD = "cecl_reserve_basis_method"
INVALID_BALANCE_COUNT_FIELD = "_cecl_invalid_balance_count"

RATIO_COLUMNS = [
    "portfolio",
    "bucket",
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
    "period_method",
    "period",
    "reserve_field",
    "weight",
    "observation_grain",
    "observation_count",
    "included_observation_count",
    "excluded_observation_count",
    "missing_reserve_count",
    "invalid_balance_count",
    "balance",
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
    """One resolved CECL base shared by commercial and Consumer reporting."""

    method: str
    effective_reserve: pd.Series
    ratios: pd.DataFrame
    audit: pd.DataFrame
    required_fields: tuple[str, ...]
    exception_code: str = ""


def attach_cecl_reserve_basis(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, CeclReserveBasis]:
    """Attach the selected Base reserve before stress modules execute."""
    basis = build_cecl_reserve_basis(results, scenario, exceptions)
    out = results.copy()
    out[EFFECTIVE_RESERVE_FIELD] = basis.effective_reserve.reindex(out.index)
    out[RESERVE_BASIS_METHOD_FIELD] = basis.method
    return out, basis


def validate_cecl_config(scenario: Mapping[str, Any]) -> None:
    """Validate the optional scenario-level CECL reserve-basis contract."""
    cecl = scenario.get("cecl", {})
    if cecl is None:
        raise ValueError("Scenario cecl must be a JSON object.")
    if not isinstance(cecl, Mapping):
        raise ValueError("Scenario cecl must be a JSON object.")
    basis = cecl.get("reserve_basis")
    if basis is None:
        return
    if not isinstance(basis, Mapping):
        raise ValueError("cecl.reserve_basis must be a JSON object.")

    method = str(basis.get("method", "in_place"))
    if method not in RESERVE_BASIS_METHODS:
        raise ValueError(
            "cecl.reserve_basis.method must be one of: "
            f"{', '.join(sorted(RESERVE_BASIS_METHODS))}."
        )

    central = basis.get("central_tendency", {})
    if not isinstance(central, Mapping):
        raise ValueError(
            "cecl.reserve_basis.central_tendency must be a JSON object."
        )
    threshold = central.get("z_score_threshold", DEFAULT_Z_SCORE_THRESHOLD)
    threshold_number = _finite_number(
        threshold,
        "cecl.reserve_basis.central_tendency.z_score_threshold",
    )
    if threshold_number <= 0:
        raise ValueError(
            "cecl.reserve_basis.central_tendency.z_score_threshold must be greater than zero."
        )
    grain = str(central.get("observation_grain", "borrower"))
    if grain != "borrower":
        raise ValueError(
            "cecl.reserve_basis.central_tendency.observation_grain currently supports only 'borrower'."
        )

    history = basis.get("weighted_history", {})
    if history is None:
        history = {}
    if not isinstance(history, Mapping):
        raise ValueError(
            "cecl.reserve_basis.weighted_history must be a JSON object."
        )
    period_method = str(history.get("period_method", "in_place"))
    if period_method not in PERIOD_METHODS:
        raise ValueError(
            "cecl.reserve_basis.weighted_history.period_method must be one of: "
            f"{', '.join(sorted(PERIOD_METHODS))}."
        )
    current_field = str(cecl.get("reserve_field", "cecl_reserve"))
    fields: set[str] = set()
    validate_history = method == "weighted_history" or bool(history)
    if validate_history:
        periods = history.get("periods")
        if not isinstance(periods, list) or not periods:
            raise ValueError(
                "cecl.reserve_basis.weighted_history.periods must be a nonempty JSON list."
            )
        names: set[str] = set()
        weights: List[float] = []
        current_weight = 0.0
        for index, period in enumerate(periods):
            path = f"cecl.reserve_basis.weighted_history.periods[{index}]"
            if not isinstance(period, Mapping):
                raise ValueError(f"{path} must be a JSON object.")
            name = str(period.get("name", "")).strip()
            field = str(period.get("reserve_field", "")).strip()
            if not name:
                raise ValueError(f"{path}.name must be nonblank.")
            if not field:
                raise ValueError(f"{path}.reserve_field must be nonblank.")
            if name in names:
                raise ValueError(
                    "cecl.reserve_basis.weighted_history period names must be unique."
                )
            if field in fields:
                raise ValueError(
                    "cecl.reserve_basis.weighted_history reserve fields must be unique."
                )
            names.add(name)
            fields.add(field)
            weight = _finite_number(period.get("weight"), f"{path}.weight")
            if weight <= 0:
                raise ValueError(f"{path}.weight must be greater than zero.")
            weights.append(weight)
            if field == current_field:
                current_weight = weight
        if not math.isclose(
            math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=WEIGHT_TOLERANCE
        ):
            raise ValueError(
                "cecl.reserve_basis.weighted_history period weights must sum to 1."
            )
        if current_weight <= 0:
            raise ValueError(
                "cecl.reserve_basis.weighted_history must include the configured "
                "cecl.reserve_field with a positive weight."
            )
    if method == "central_tendency":
        fields.add(current_field)
    if not fields:
        return

    identity = scenario.get("inputs", {}).get("identity", {})
    if isinstance(identity, Mapping) and identity:
        aliases = identity.get("column_aliases", {})
        canonical = (
            {str(value) for value in aliases.keys()}
            if isinstance(aliases, Mapping)
            else set()
        )
        numeric = {str(value) for value in identity.get("numeric_columns", [])}
        required = {str(value) for value in identity.get("required_columns", [])}
        sum_fields = {
            str(value)
            for value in scenario.get("borrower", {}).get("sum_fields", [])
        }
        missing_aliases = sorted(fields - canonical)
        missing_numeric = sorted(fields - numeric)
        missing_required = sorted(fields - required)
        missing_sum = sorted(fields - sum_fields)
        if missing_aliases:
            raise ValueError(
                "CECL reserve-basis fields must be canonical identity "
                f"column aliases; missing: {', '.join(missing_aliases)}."
            )
        if missing_numeric:
            raise ValueError(
                "CECL reserve-basis fields must be identity numeric_columns; "
                f"missing: {', '.join(missing_numeric)}."
            )
        if missing_required:
            raise ValueError(
                "CECL reserve-basis fields must be identity required_columns "
                f"so missing period columns fail at load time; missing: {', '.join(missing_required)}."
            )
        if missing_sum:
            raise ValueError(
                "CECL reserve-basis fields must be borrower.sum_fields so "
                "multi-loan borrowers aggregate correctly; missing: "
                f"{', '.join(missing_sum)}."
            )


def reserve_basis_fields(scenario: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every reserve column required by the selected CECL basis."""
    cecl = scenario.get("cecl", {})
    current = str(cecl.get("reserve_field", "cecl_reserve"))
    basis = cecl.get("reserve_basis", {})
    if not isinstance(basis, Mapping) or basis.get("method", "in_place") != "weighted_history":
        return (current,)
    history = basis.get("weighted_history", {})
    periods = history.get("periods", []) if isinstance(history, Mapping) else []
    fields = [
        str(period.get("reserve_field", "")).strip()
        for period in periods
        if isinstance(period, Mapping)
    ]
    return tuple(dict.fromkeys(field for field in fields if field))


def reserve_missing_count_field(reserve_field: str) -> str:
    """Return the internal loan-missing counter carried through aggregation."""
    return f"_cecl_reserve_missing_count__{reserve_field}"


def build_cecl_reserve_basis(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> CeclReserveBasis:
    """Resolve effective reserves, group ratios, and a period-level audit."""
    exceptions = exceptions if exceptions is not None else []
    cecl = scenario.get("cecl", {})
    basis_config = cecl.get("reserve_basis", {})
    if not isinstance(basis_config, Mapping):
        basis_config = {}
    method = str(basis_config.get("method", "in_place"))
    current_field = str(cecl.get("reserve_field", "cecl_reserve"))
    required_fields = reserve_basis_fields(scenario)
    portfolio_field = str(
        cecl.get(
            "portfolio_field",
            scenario.get("borrower", {}).get("portfolio_field", "portfolio"),
        )
    )
    balance_field = str(
        scenario.get("borrower", {}).get("balance_field", "outstanding_balance")
    )
    effective = pd.Series(np.nan, index=results.index, dtype=float)

    missing_fields = [field for field in required_fields if field not in results.columns]
    if missing_fields:
        code = (
            "CECL_RESERVE_FIELD_MISSING"
            if missing_fields == [current_field]
            else "CECL_BASIS_FIELD_MISSING"
        )
        for field in missing_fields:
            record_exception(
                exceptions,
                "ERROR",
                "cecl",
                code,
                "A reserve field required by the selected CECL basis is missing; CECL is unavailable.",
                field=field,
                details=f"reserve_basis={method}",
            )
        return CeclReserveBasis(
            method=method,
            effective_reserve=effective,
            ratios=_empty_ratios(),
            audit=_empty_audit(),
            required_fields=required_fields,
            exception_code=code,
        )

    for field in required_fields:
        missing_count = int(
            _finite_numeric(results[field]).isna().sum()
        )
        if not missing_count:
            continue
        code = (
            "CECL_LOAN_RESERVE_MISSING_TREATED_AS_ZERO"
            if field == current_field
            else "CECL_HISTORY_RESERVE_MISSING_TREATED_AS_ZERO"
        )
        already_logged = any(
            str(item.get("code", "")) == code
            and str(item.get("field", "")) == field
            for item in exceptions
        )
        if not already_logged:
            record_exception(
                exceptions,
                "WARNING",
                "cecl",
                code,
                "Reserve-basis observations with missing or invalid values were treated as zero.",
                field=field,
                details=f"missing_count={missing_count}; reserve_basis={method}",
            )

    if portfolio_field not in results.columns or balance_field not in results.columns:
        missing = portfolio_field if portfolio_field not in results.columns else balance_field
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            "CECL_BASIS_GROUP_FIELD_MISSING",
            "A grouping field required by the selected CECL basis is missing; CECL is unavailable.",
            field=missing,
        )
        return CeclReserveBasis(
            method=method,
            effective_reserve=effective,
            ratios=_empty_ratios(),
            audit=_empty_audit(),
            required_fields=required_fields,
            exception_code="CECL_BASIS_GROUP_FIELD_MISSING",
        )

    frame = results.copy()
    if "model_excluded" in frame.columns:
        frame = frame[
            ~frame["model_excluded"].fillna(False).astype(bool)
        ].copy()
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
    periods, period_method = _period_specs(
        method, current_field, basis_config
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

    period_results: Dict[tuple[Any, str, str], Dict[str, Any]] = {}
    audit_rows: List[Dict[str, Any]] = []
    group_columns = [portfolio_field, "base_bucket"]
    for (portfolio, bucket), group in frame.groupby(group_columns, dropna=False):
        if str(bucket) not in {"Pass", "Special Mention", "Substandard", "Unknown"}:
            continue
        finite_balances = _finite_numeric(group[balance_field])
        invalid_balance_count = _invalid_balance_count(group, balance_field)
        eligible_group = group.loc[finite_balances.notna()]
        group_balance = float(finite_balances.sum())
        if invalid_balance_count:
            record_exception(
                exceptions,
                "ERROR",
                "cecl",
                "CECL_BALANCE_INVALID",
                "A CECL calibration group contains negative, invalid, or nonfinite balances; its reserve basis is unavailable.",
                portfolio=portfolio,
                bucket=str(bucket),
                field=balance_field,
                details=f"invalid_balance_count={invalid_balance_count}",
            )
        for period in periods:
            field = period["reserve_field"]
            observations = _borrower_observations(
                eligible_group, borrower_field, balance_field, field
            )
            values = _calculate_period(
                observations,
                period_method,
                threshold,
                group_balance,
            )
            if values["status"] != "available":
                record_exception(
                    exceptions,
                    "ERROR" if group_balance > 0 else "WARNING",
                    "cecl",
                    "CECL_BASIS_PERIOD_UNAVAILABLE",
                    "A configured CECL basis period produced no usable reserve ratio.",
                    portfolio=portfolio,
                    bucket=str(bucket),
                    field=field,
                    details=(
                        f"period={period['name']}; period_method={period_method}; "
                        f"observation_count={values['observation_count']}; "
                        f"included_observation_count={values['included_observation_count']}"
                    ),
                )
            key = (portfolio, str(bucket), period["name"])
            period_results[key] = values
            audit_rows.append(
                {
                    "portfolio": portfolio,
                    "bucket": str(bucket),
                    "reserve_basis": method,
                    "period_method": period_method,
                    "period": period["name"],
                    "reserve_field": field,
                    "weight": period["weight"],
                    "observation_grain": "borrower",
                    **values,
                    "invalid_balance_count": invalid_balance_count,
                    "weighted_ratio_component": (
                        values["period_reserve_ratio"] * period["weight"]
                        if pd.notna(values["period_reserve_ratio"])
                        else np.nan
                    ),
                    "effective_reserve_ratio": np.nan,
                    "basis_status": "unavailable",
                    "basis_exception_code": "CECL_BASIS_PERIOD_UNAVAILABLE",
                }
            )

    ratio_rows: List[Dict[str, Any]] = []
    for (portfolio, bucket), group in frame.groupby(group_columns, dropna=False):
        bucket_text = str(bucket)
        if bucket_text not in {"Pass", "Special Mention", "Substandard", "Unknown"}:
            continue
        finite_balances = _finite_numeric(group[balance_field])
        invalid_balance_count = _invalid_balance_count(group, balance_field)
        group_balance = float(finite_balances.sum())
        selected = [
            (period, period_results.get((portfolio, bucket_text, period["name"])))
            for period in periods
        ]
        unavailable = [
            item for item in selected if item[1] is None or item[1]["status"] != "available"
        ]
        direct_amount_basis = (
            method in {"in_place", "weighted_history"}
            and period_method == "in_place"
        )
        direct_values: pd.Series | None = None
        if direct_amount_basis:
            # Consumer can still use authoritative reserve dollars when a
            # zero-balance group has no meaningful commercial reserve ratio.
            direct_values = _weighted_row_values(
                group, [period for period, _ in selected]
            ).where(finite_balances.notna())
            effective.loc[group.index] = direct_values
        if invalid_balance_count:
            ratio = np.nan
            reserve = np.nan
            status = "unavailable"
            code = "CECL_BALANCE_INVALID"
            effective.loc[group.index] = np.nan
        elif unavailable:
            ratio = np.nan
            reserve = np.nan
            status = "unavailable"
            code = "CECL_BASIS_PERIOD_UNAVAILABLE"
        else:
            if direct_values is not None:
                reserve = _finite_fsum(direct_values.dropna())
                ratio = (
                    pct(reserve, group_balance)
                    if pd.notna(reserve)
                    else np.nan
                )
            else:
                ratio = _finite_fsum(
                    period["weight"] * values["period_reserve_ratio"]
                    for period, values in selected
                )
                reserve = group_balance * ratio
            if not np.isfinite(ratio) or not np.isfinite(reserve):
                ratio = np.nan
                reserve = np.nan
                status = "unavailable"
                code = "CECL_BASIS_RESULT_NONFINITE"
                effective.loc[group.index] = np.nan
            else:
                status = "available"
                code = ""
            if not direct_amount_basis and status == "available":
                effective.loc[group.index] = (
                    finite_balances * ratio
                )
        ratio_rows.append(
            {
                "portfolio": portfolio,
                "bucket": bucket_text,
                "base_balance": group_balance,
                "base_reserve": reserve,
                "reserve_ratio": ratio,
                "invalid_balance_count": invalid_balance_count,
                "status": status,
                "exception_code": code,
            }
        )
        for row in audit_rows:
            if _same_group(row["portfolio"], portfolio) and row["bucket"] == bucket_text:
                row["effective_reserve_ratio"] = ratio
                row["basis_status"] = status
                row["basis_exception_code"] = code

    ratios = pd.DataFrame(ratio_rows, columns=RATIO_COLUMNS)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return CeclReserveBasis(
        method=method,
        effective_reserve=effective,
        ratios=ratios,
        audit=audit,
        required_fields=required_fields,
    )


def _period_specs(
    method: str,
    current_field: str,
    basis_config: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], str]:
    if method != "weighted_history":
        return (
            [{"name": "current", "reserve_field": current_field, "weight": 1.0}],
            method,
        )
    history = basis_config.get("weighted_history", {})
    return (
        [
            {
                "name": str(period["name"]).strip(),
                "reserve_field": str(period["reserve_field"]).strip(),
                "weight": to_number(period["weight"]),
            }
            for period in history.get("periods", [])
        ],
        str(history.get("period_method", "in_place")),
    )


def _weighted_row_values(
    group: pd.DataFrame,
    periods: Sequence[Mapping[str, Any]],
) -> pd.Series:
    """Blend row reserves with compensated summation to avoid cent-level drift."""
    components = [
        (
            _finite_numeric(group[str(period["reserve_field"])])
            .fillna(0.0)
            .to_numpy(dtype=float)
            * float(period["weight"])
        )
        for period in periods
    ]
    values = [_finite_fsum(items) for items in zip(*components)]
    return pd.Series(values, index=group.index, dtype=float)


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
        borrower = borrower.where(~missing, pd.Series(group.index, index=group.index).map(lambda value: f"__row_{value}"))
        work["borrower"] = borrower
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


def _calculate_period(
    observations: pd.DataFrame,
    method: str,
    threshold: float,
    group_balance: float,
) -> Dict[str, Any]:
    reserve = _finite_fsum(observations["reserve"])
    raw_ratio = (
        pct(reserve, group_balance) if pd.notna(reserve) else np.nan
    )
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
        "exception_code": "" if status == "available" else "CECL_BASIS_PERIOD_UNAVAILABLE",
    }


def _same_group(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def _finite_number(value: Any, path: str) -> float:
    number = to_number(value, np.nan)
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number):
        raise ValueError(f"{path} must be numeric and finite.")
    return float(number)


def _finite_numeric(values: pd.Series) -> pd.Series:
    """Coerce reserve values and treat infinities as invalid/missing."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def _finite_fsum(values: Iterable[float]) -> float:
    """Return a finite compensated sum, or NaN on overflow/nonfinite input."""
    try:
        total = math.fsum(float(value) for value in values)
    except (OverflowError, TypeError, ValueError):
        return np.nan
    return total if np.isfinite(total) else np.nan


def _invalid_balance_count(group: pd.DataFrame, balance_field: str) -> int:
    """Return raw invalid balances, including counts carried through aggregation."""
    balances = pd.to_numeric(group[balance_field], errors="coerce")
    direct_count = int(((~np.isfinite(balances)) | balances.lt(0)).sum())
    if INVALID_BALANCE_COUNT_FIELD not in group.columns:
        return direct_count
    carried = pd.to_numeric(
        group[INVALID_BALANCE_COUNT_FIELD], errors="coerce"
    )
    carried = carried.where(np.isfinite(carried), 0.0).clip(lower=0.0)
    return max(direct_count, int(carried.sum()))


def _empty_ratios() -> pd.DataFrame:
    return pd.DataFrame(columns=RATIO_COLUMNS)


def _empty_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=AUDIT_COLUMNS)
