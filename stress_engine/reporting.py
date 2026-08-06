"""Portfolio summaries, CECL projection, and report table generation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .cecl import (
    CECL_LEVEL_TAG_FIELD,
    CeclReserveBasis,
    build_cecl_reserve_basis,
    invalid_balance_mask,
    normalized_balance_values,
)
from .modules.overlay import BUCKETS, apply_overlays
from .exceptions import record_exception
from .utils import get_levels, pct, to_number, weighted_average

REPORT_BUCKETS = [*BUCKETS, "Unknown"]
_CONSUMER_BASIS_VALUE_FIELD = "_cecl_consumer_basis_value"
BUCKET_SUMMARY_COLUMNS = [
    "portfolio",
    "stress_level",
    "bucket",
    "balance",
    "borrower_count",
    "source",
]
CECL_SUMMARY_COLUMNS = [
    "portfolio",
    "stress_level",
    "bucket",
    "method",
    "reserve_basis",
    "balance",
    "reserve_ratio",
    "proforma_cecl_reserve",
    "proforma_cecl_ratio",
    "cecl_reserve_status",
    "exception_code",
]

def build_reports(
    results: pd.DataFrame,
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
    out_of_scope: pd.DataFrame,
    exceptions: List[Dict[str, Any]] | None = None,
    reserve_basis: CeclReserveBasis | None = None,
) -> Dict[str, pd.DataFrame]:
    """Build all report DataFrames from stressed borrower results.

    Called once by `StressEngine.run` after all stress modules finish. This is
    also where overlays and CECL summaries are calculated.
    """
    exceptions = exceptions if exceptions is not None else []
    modeled_results = _model_included_rows(results, scenario)
    bucket_summary = build_bucket_summary(modeled_results, scenario)
    bucket_summary, overlay_summary = apply_overlays(
        bucket_summary, modeled_results, scenario, exceptions
    )
    # CECL can group more finely than migration reporting. In the example,
    # migration reports use CRE/C&I, while CECL can use CRE rollup or subsector.
    cecl_bucket_summary = build_cecl_bucket_summary(
        modeled_results, bucket_summary, scenario
    )
    reserve_basis = reserve_basis or build_cecl_reserve_basis(
        modeled_results, scenario, exceptions
    )
    cecl_summary = build_cecl_summary(
        modeled_results,
        cecl_bucket_summary,
        scenario,
        exceptions,
        reserve_basis,
    )
    reports = {
        "migration_summary": bucket_summary,
        "overlay_summary": overlay_summary,
        "cecl_summary": cecl_summary,
        "cecl_basis_summary": reserve_basis.audit,
        "cre_summary": build_cre_summary(modeled_results, scenario),
        "ci_summary": build_ci_summary(modeled_results, scenario),
        "consumer_summary": build_consumer_summary(
            modeled_results, scenario, reserve_basis
        ),
        "out_of_scope_summary": build_out_of_scope_summary(out_of_scope),
    }
    return reports


def build_bucket_summary(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    portfolio_field: str | None = None,
    include_consumer: bool = False,
) -> pd.DataFrame:
    """Summarize balances and counts by portfolio, level, and risk bucket."""
    portfolio_field = portfolio_field or scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    rows: List[Dict[str, Any]] = []
    frame = _model_included_rows(results, scenario)
    if not include_consumer:
        if "primary_module" in frame.columns:
            frame = frame[frame["primary_module"].astype(str).str.lower() != "consumer"]
        elif "module_applied" in frame.columns:
            frame = frame[~frame["module_applied"].astype(str).str.contains("Consumer", na=False)]
    levels = ["Base"] + get_levels(scenario)
    for level in levels:
        bucket_col = "base_bucket" if level == "Base" else f"stressed_bucket_{level}"
        if bucket_col not in results.columns:
            continue
        for (portfolio, bucket), group in frame.groupby([portfolio_field, bucket_col], dropna=False):
            if bucket not in REPORT_BUCKETS:
                continue
            rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": level,
                    "bucket": bucket,
                    "balance": float(pd.to_numeric(group[balance_field], errors="coerce").sum()),
                    **_population_counts(group, scenario),
                    "source": "model",
                }
            )
    columns = list(BUCKET_SUMMARY_COLUMNS)
    if scenario.get("_targeted_mode"):
        columns.insert(columns.index("source"), "loan_count")
    return pd.DataFrame(rows, columns=columns)


def _model_included_rows(
    frame: pd.DataFrame,
    scenario: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return rows that remain eligible for modeled reports."""
    included = pd.Series(True, index=frame.index)
    if "model_excluded" in frame.columns:
        included &= ~frame["model_excluded"].fillna(False).astype(bool)
    if scenario is not None:
        included &= ~invalid_balance_mask(frame, scenario)
    out = frame.loc[included]
    if scenario is not None:
        balance_field = str(
            scenario.get("borrower", {}).get(
                "balance_field", "outstanding_balance"
            )
        )
        if balance_field in out.columns:
            out = out.copy()
            out[balance_field] = normalized_balance_values(out, scenario)
    return out


def _cecl_public_portfolio_field(scenario: Mapping[str, Any]) -> str:
    """Return the public portfolio field used by CECL output rows."""
    cecl = scenario.get("cecl", {})
    if not isinstance(cecl, Mapping):
        cecl = {}
    borrower = scenario.get("borrower", {})
    if not isinstance(borrower, Mapping):
        borrower = {}
    return str(
        cecl.get(
            "portfolio_field",
            borrower.get("portfolio_field", "portfolio"),
        )
    )


def _normalized_cecl_key(value: Any) -> Any:
    """Strip external CECL keys while preserving genuine missing values."""
    return value if pd.isna(value) else str(value).strip()


def _has_configured_cecl_level_tags(scenario: Mapping[str, Any]) -> bool:
    """Return whether the scenario opts into explicit CECL tag grain."""
    tags = scenario.get("tags", {})
    return isinstance(tags, Mapping) and any(
        isinstance(spec, Mapping) and spec.get("cecl_level") is True
        for spec in tags.values()
    )


def _consumer_row_mask_for_reporting(
    frame: pd.DataFrame,
    scenario: Mapping[str, Any],
    portfolio_field: str,
) -> pd.Series:
    """Identify Consumer rows before CECL tag/bucket aggregation."""
    mask = pd.Series(False, index=frame.index)
    configured = scenario.get("cecl", {}).get("portfolios", {})
    expected_loss = {
        str(portfolio).strip()
        for portfolio, spec in configured.items()
        if isinstance(spec, Mapping) and spec.get("method") == "expected_loss"
    } if isinstance(configured, Mapping) else set()
    if portfolio_field in frame.columns and expected_loss:
        mask |= frame[portfolio_field].map(_normalized_cecl_key).isin(
            expected_loss
        )
    if "primary_module" in frame.columns:
        mask |= (
            frame["primary_module"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("consumer")
        )
    if "module_applied" in frame.columns:
        mask |= frame["module_applied"].astype(str).str.contains(
            "Consumer", case=False, na=False
        )
    return mask.fillna(False)


def build_cecl_bucket_summary(
    results: pd.DataFrame,
    migration_summary: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the bucket summary used only for CECL reserve calculations.

    Called by `build_reports`. Commercial balances retain both their public
    CECL portfolio and CECL-level tag so each tag/bucket ratio can be applied
    before the public report is rolled back up. Overlay migrations are
    reinserted only after their public portfolio resolves to one CECL tag.
    """
    portfolio_field = _cecl_public_portfolio_field(scenario)
    frame = _model_included_rows(results, scenario).copy()
    if portfolio_field not in frame.columns:
        raise ValueError(
            f"CECL public portfolio field '{portfolio_field}' is missing."
        )
    if CECL_LEVEL_TAG_FIELD not in frame.columns:
        if _has_configured_cecl_level_tags(scenario):
            raise ValueError(
                f"CECL level-tag field '{CECL_LEVEL_TAG_FIELD}' is missing."
            )
        # Legacy programmatic callers without CECL-level tag definitions use
        # their public CECL portfolio as the calibration tag, matching the
        # reserve-basis resolver's backward-compatible behavior.
        frame[CECL_LEVEL_TAG_FIELD] = frame[portfolio_field]
    frame[portfolio_field] = frame[portfolio_field].map(_normalized_cecl_key)
    frame[CECL_LEVEL_TAG_FIELD] = frame[CECL_LEVEL_TAG_FIELD].map(
        _normalized_cecl_key
    )
    # Consumer never uses commercial tag ratios. Giving it a deterministic
    # tag equal to its public portfolio keeps this internal summary rectangular
    # without changing the expected-loss calculation.
    consumer_rows = _consumer_row_mask_for_reporting(
        frame, scenario, portfolio_field
    )
    frame.loc[consumer_rows, CECL_LEVEL_TAG_FIELD] = frame.loc[
        consumer_rows, portfolio_field
    ]

    balance_field = scenario["borrower"]["balance_field"]
    levels = ["Base"] + get_levels(scenario)
    rows: List[Dict[str, Any]] = []
    for level in levels:
        bucket_field = (
            "base_bucket" if level == "Base" else f"stressed_bucket_{level}"
        )
        if bucket_field not in frame.columns:
            continue
        grouped = frame.groupby(
            [portfolio_field, CECL_LEVEL_TAG_FIELD, bucket_field],
            dropna=False,
        )
        for (portfolio, cecl_level_tag, bucket), group in grouped:
            if bucket not in REPORT_BUCKETS:
                continue
            rows.append(
                {
                    "portfolio": portfolio,
                    CECL_LEVEL_TAG_FIELD: cecl_level_tag,
                    "stress_level": level,
                    "bucket": bucket,
                    "balance": float(
                        pd.to_numeric(
                            group[balance_field], errors="coerce"
                        ).sum()
                    ),
                    **_population_counts(group, scenario),
                    "source": "model",
                }
            )
    internal_columns = [
        "portfolio",
        CECL_LEVEL_TAG_FIELD,
        "stress_level",
        "bucket",
        "balance",
        "borrower_count",
    ]
    if scenario.get("_targeted_mode"):
        internal_columns.append("loan_count")
    internal_columns.append("source")
    summary = pd.DataFrame(rows, columns=internal_columns)
    overlays = scenario.get("overlays", {})
    if isinstance(overlays, Mapping):
        overlay_portfolios = {
            _normalized_cecl_key(portfolio) for portfolio in overlays
        }
    else:
        overlay_portfolios = {
            _normalized_cecl_key(item["portfolio"]) for item in overlays
        }
    if overlay_portfolios and not migration_summary.empty:
        summary = summary[~summary["portfolio"].isin(overlay_portfolios)]
        overlay_frames: List[pd.DataFrame] = []
        for portfolio in sorted(overlay_portfolios, key=str):
            overlay_rows = migration_summary[
                migration_summary["portfolio"].map(_normalized_cecl_key)
                == _normalized_cecl_key(portfolio)
            ].copy()
            if overlay_rows.empty:
                continue
            portfolio_rows = frame[
                frame[portfolio_field] == _normalized_cecl_key(portfolio)
            ]
            missing_tag = (
                portfolio_rows[CECL_LEVEL_TAG_FIELD].isna()
                | portfolio_rows[CECL_LEVEL_TAG_FIELD]
                .astype(str)
                .str.strip()
                .eq("")
            )
            tags = sorted(
                {
                    str(value).strip()
                    for value in portfolio_rows[CECL_LEVEL_TAG_FIELD].dropna()
                    if str(value).strip()
                }
            )
            if missing_tag.any() or len(tags) != 1:
                raise ValueError(
                    "Overlay CECL portfolio "
                    f"'{portfolio}' must resolve to exactly one CECL-level "
                    f"tag on every modeled row; found {len(tags)} tags "
                    f"({', '.join(tags) or 'none'}) and "
                    f"{int(missing_tag.sum())} untagged rows."
                )
            overlay_rows["portfolio"] = overlay_rows["portfolio"].map(
                _normalized_cecl_key
            )
            overlay_rows[CECL_LEVEL_TAG_FIELD] = tags[0]
            overlay_frames.append(overlay_rows)
        if overlay_frames:
            summary = pd.concat(
                [summary, *overlay_frames], ignore_index=True, sort=False
            )

    return summary.reindex(columns=internal_columns)


def build_cecl_summary(
    results: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
    reserve_basis: CeclReserveBasis | None = None,
) -> pd.DataFrame:
    """Calculate proforma CECL reserves from the resolved reserve basis.

    Called by `build_reports`. Commercial balances are first valued at CECL
    tag/bucket grain, then rolled back into the existing public portfolio and
    bucket schema. This prevents two CECL-level tags in one public portfolio
    from sharing the wrong reserve ratio.
    """
    exceptions = exceptions if exceptions is not None else []
    cecl = scenario.get("cecl", {})
    reserve_field = cecl.get("reserve_field", "cecl_reserve")
    portfolio_field = _cecl_public_portfolio_field(scenario)
    balance_field = scenario["borrower"]["balance_field"]
    results = _model_included_rows(results, scenario).copy()
    if portfolio_field in results.columns:
        results[portfolio_field] = results[portfolio_field].map(
            _normalized_cecl_key
        )
    bucket_summary = bucket_summary.copy()
    if "portfolio" in bucket_summary.columns:
        bucket_summary["portfolio"] = bucket_summary["portfolio"].map(
            _normalized_cecl_key
        )
    method_by_portfolio = _cecl_methods(results, scenario)
    if CECL_LEVEL_TAG_FIELD not in bucket_summary.columns:
        public_portfolios = (
            set(bucket_summary["portfolio"].dropna())
            if "portfolio" in bucket_summary.columns
            else set()
        )
        commercial = {
            portfolio
            for portfolio in public_portfolios
            if method_by_portfolio.get(
                portfolio, "bucket_reserve_ratio"
            ) != "expected_loss"
        }
        if commercial and _has_configured_cecl_level_tags(scenario):
            raise ValueError(
                "CECL bucket summary is missing "
                f"'{CECL_LEVEL_TAG_FIELD}' for commercial portfolios: "
                f"{', '.join(sorted(map(str, commercial)))}."
            )
        bucket_summary[CECL_LEVEL_TAG_FIELD] = bucket_summary.get(
            "portfolio", pd.Series(index=bucket_summary.index, dtype=object)
        )
    bucket_summary[CECL_LEVEL_TAG_FIELD] = bucket_summary[
        CECL_LEVEL_TAG_FIELD
    ].map(_normalized_cecl_key)
    levels = ["Base"] + get_levels(scenario)
    rows: List[Dict[str, Any]] = []
    zero_balance_tolerance = to_number(
        cecl.get("zero_balance_tolerance", 1e-9), 1e-9
    )

    reserve_basis = reserve_basis or build_cecl_reserve_basis(
        results, scenario, exceptions
    )
    ratio_df = reserve_basis.ratios

    for portfolio in sorted(bucket_summary["portfolio"].dropna().unique()):
        method = method_by_portfolio.get(portfolio, "bucket_reserve_ratio")
        if method == "expected_loss":
            rows.extend(
                _consumer_cecl_rows(
                    results, portfolio, scenario, levels, reserve_basis
                )
            )
            continue
        portfolio_total_rows = []
        for level in levels:
            level_rows = bucket_summary[
                (bucket_summary["portfolio"] == portfolio)
                & (bucket_summary["stress_level"] == level)
            ]
            component_rows: List[Dict[str, Any]] = []
            for _, bucket_row in level_rows.iterrows():
                bucket = bucket_row["bucket"]
                cecl_level_tag = bucket_row[CECL_LEVEL_TAG_FIELD]
                balance = to_number(bucket_row["balance"], np.nan)
                ratio, status, exception_code, basis_label = _reserve_ratio_for(
                    portfolio,
                    cecl_level_tag,
                    bucket,
                    ratio_df,
                    reserve_basis.method,
                )
                if np.isfinite(balance) and abs(balance) <= zero_balance_tolerance:
                    balance = 0.0
                invalid_balance = (
                    not np.isfinite(balance)
                    or balance < -zero_balance_tolerance
                )
                basis_balance_invalid = (
                    status == "unavailable"
                    and exception_code == "CECL_BALANCE_INVALID"
                )
                if invalid_balance or basis_balance_invalid:
                    record_exception(
                        exceptions,
                        "WARNING",
                        "cecl",
                        "CECL_BUCKET_BALANCE_EXCLUDED",
                        "A malformed CECL bucket balance component was excluded; remaining valid components continued reporting.",
                        portfolio=portfolio,
                        stress_level=level,
                        bucket=bucket,
                        field=balance_field,
                        cecl_level_tag=cecl_level_tag,
                    )
                    continue
                if (
                    status == "unavailable"
                    and reserve_basis.exception_code
                ):
                    exception_code = reserve_basis.exception_code
                is_positive_balance = balance > zero_balance_tolerance
                if status == "available":
                    # Proforma reserve = stressed bucket balance times the
                    # selected ratio for this CECL tag/bucket component.
                    reserve = balance * ratio
                    reserve_ratio = ratio
                    proforma_ratio = pct(reserve, balance)
                elif is_positive_balance:
                    reserve = np.nan
                    reserve_ratio = np.nan
                    proforma_ratio = np.nan
                    record_exception(
                        exceptions,
                        "ERROR",
                        "cecl",
                        exception_code,
                        "CECL reserve ratio is unavailable for a required bucket calculation; proforma CECL reserve was not calculated.",
                        portfolio=portfolio,
                        stress_level=level,
                        bucket=bucket,
                        field=reserve_field,
                        cecl_level_tag=cecl_level_tag,
                    )
                else:
                    reserve = 0.0
                    reserve_ratio = np.nan
                    proforma_ratio = np.nan
                    status = "not_applicable_zero_balance"
                    exception_code = ""
                component_rows.append(
                    {
                        "portfolio": portfolio,
                        CECL_LEVEL_TAG_FIELD: cecl_level_tag,
                        "stress_level": level,
                        "bucket": bucket,
                        "method": method,
                        "reserve_basis": basis_label,
                        "balance": balance,
                        "reserve_ratio": reserve_ratio,
                        "proforma_cecl_reserve": reserve,
                        "proforma_cecl_ratio": proforma_ratio,
                        "cecl_reserve_status": status,
                        "exception_code": exception_code,
                    }
                )

            # Public CECL output remains portfolio/bucket grain. Combine the
            # separately valued tag components only after every tag used its
            # own basis ratio.
            public_bucket_rows: List[Dict[str, Any]] = []
            components = pd.DataFrame(component_rows)
            present_buckets = (
                set(components["bucket"].dropna())
                if not components.empty
                else set()
            )
            ordered_buckets = [
                bucket for bucket in REPORT_BUCKETS if bucket in present_buckets
            ]
            ordered_buckets.extend(
                sorted(present_buckets - set(ordered_buckets), key=str)
            )
            for bucket in ordered_buckets:
                bucket_components = components[
                    components["bucket"] == bucket
                ]
                bucket_balance = float(
                    pd.to_numeric(
                        bucket_components["balance"], errors="coerce"
                    ).sum()
                )
                bucket_unavailable = bucket_components[
                    "cecl_reserve_status"
                ].eq("unavailable").any()
                bucket_has_positive_balance = pd.to_numeric(
                    bucket_components["balance"], errors="coerce"
                ).gt(zero_balance_tolerance).any()
                bucket_codes = sorted(
                    {
                        str(code).strip()
                        for code in bucket_components["exception_code"].dropna()
                        if str(code).strip()
                    }
                )
                bucket_basis = _combined_basis(
                    {
                        str(value)
                        for value in bucket_components[
                            "reserve_basis"
                        ].dropna()
                        if str(value).strip()
                    }
                )
                if bucket_unavailable:
                    bucket_reserve = np.nan
                    bucket_ratio = np.nan
                    bucket_status = "unavailable"
                    bucket_exception_code = (
                        ";".join(bucket_codes)
                        or "CECL_RESERVE_RATIO_UNAVAILABLE"
                    )
                elif not bucket_has_positive_balance:
                    bucket_reserve = 0.0
                    bucket_ratio = np.nan
                    bucket_status = "not_applicable_zero_balance"
                    bucket_exception_code = ""
                else:
                    bucket_reserve = float(
                        pd.to_numeric(
                            bucket_components["proforma_cecl_reserve"],
                            errors="coerce",
                        ).sum()
                    )
                    bucket_ratio = pct(bucket_reserve, bucket_balance)
                    bucket_status = "available"
                    bucket_exception_code = ""
                public_row = {
                    "portfolio": portfolio,
                    "stress_level": level,
                    "bucket": bucket,
                    "method": method,
                    "reserve_basis": bucket_basis,
                    "balance": bucket_balance,
                    "reserve_ratio": bucket_ratio,
                    "proforma_cecl_reserve": bucket_reserve,
                    "proforma_cecl_ratio": bucket_ratio,
                    "cecl_reserve_status": bucket_status,
                    "exception_code": bucket_exception_code,
                }
                public_bucket_rows.append(public_row)
                # Unknown remains an internal CECL component so portfolio and
                # Aggregate totals reconcile, but it is intentionally omitted
                # from the public bucket-level CECL report.
                if bucket != "Unknown":
                    rows.append(public_row)

            total_balance = float(
                pd.to_numeric(
                    pd.Series(
                        [row["balance"] for row in public_bucket_rows],
                        dtype=float,
                    ),
                    errors="coerce",
                ).sum()
            )
            unavailable = any(
                row["cecl_reserve_status"] == "unavailable"
                for row in public_bucket_rows
            )
            unavailable_codes = sorted(
                {
                    str(row["exception_code"]).strip()
                    for row in public_bucket_rows
                    if str(row["exception_code"]).strip()
                }
            )
            level_basis_labels = {
                str(row["reserve_basis"])
                for row in public_bucket_rows
                if str(row["reserve_basis"]).strip()
            }
            if unavailable:
                total_reserve_value = np.nan
                total_ratio = np.nan
                total_status = "unavailable"
                total_exception_code = (
                    ";".join(unavailable_codes)
                    or "CECL_RESERVE_RATIO_UNAVAILABLE"
                )
            else:
                total_reserve_value = float(
                    pd.to_numeric(
                        pd.Series(
                            [
                                row["proforma_cecl_reserve"]
                                for row in public_bucket_rows
                            ],
                            dtype=float,
                        ),
                        errors="coerce",
                    ).sum()
                )
                total_ratio = pct(total_reserve_value, total_balance)
                total_status = "available"
                total_exception_code = ""
            portfolio_total_rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": level,
                    "bucket": "Total",
                    "method": method,
                    "reserve_basis": _combined_basis(level_basis_labels),
                    "balance": total_balance,
                    "reserve_ratio": np.nan,
                    "proforma_cecl_reserve": total_reserve_value,
                    "proforma_cecl_ratio": total_ratio,
                    "cecl_reserve_status": total_status,
                    "exception_code": total_exception_code,
                }
            )
        rows.extend(portfolio_total_rows)

    total_df = pd.DataFrame(rows, columns=CECL_SUMMARY_COLUMNS)
    aggregate_rows = []
    for level in levels:
        totals = total_df[(total_df["stress_level"] == level) & (total_df["bucket"] == "Total")]
        balance = float(pd.to_numeric(totals["balance"], errors="coerce").sum())
        aggregate_unavailable = totals.get("cecl_reserve_status", pd.Series(dtype=object)).eq("unavailable").any()
        unavailable_codes = sorted(
            {
                str(code).strip()
                for code in totals.loc[
                    totals.get(
                        "cecl_reserve_status",
                        pd.Series(index=totals.index, dtype=object),
                    ).eq("unavailable"),
                    "exception_code",
                ].dropna()
                if str(code).strip()
            }
        )
        aggregate_exception_code = (
            ";".join(unavailable_codes)
            if unavailable_codes
            else ("CECL_COMPONENT_UNAVAILABLE" if aggregate_unavailable else "")
        )
        reserve = np.nan if aggregate_unavailable else float(pd.to_numeric(totals["proforma_cecl_reserve"], errors="coerce").sum())
        aggregate_basis = _combined_basis(
            {
                str(value)
                for value in totals.get("reserve_basis", pd.Series(dtype=object)).dropna()
                if str(value).strip()
            }
        )
        aggregate_rows.append(
            {
                "portfolio": "Aggregate",
                "stress_level": level,
                "bucket": "Total",
                "method": "weighted_average",
                "reserve_basis": aggregate_basis,
                "balance": balance,
                "reserve_ratio": np.nan,
                "proforma_cecl_reserve": reserve,
                "proforma_cecl_ratio": np.nan if aggregate_unavailable else pct(reserve, balance),
                "cecl_reserve_status": "unavailable" if aggregate_unavailable else "available",
                "exception_code": aggregate_exception_code,
            }
        )
    return pd.concat([total_df, pd.DataFrame(aggregate_rows)], ignore_index=True)


def build_cre_summary(results: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    """Build weighted CRE metric and migration summary rows.

    Called by `build_reports`; values are weighted by borrower balance.
    """
    results = _model_included_rows(results, scenario)
    config = scenario.get("modules", {}).get("CRE", {})
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    subsector_field = config.get("subsector_field", "cre_subsector")
    dscr_field = config.get("tests", {}).get("dscr", {}).get("field", "dscr")
    levels = get_levels(scenario)
    rows = []
    frame = results[results.get("module_applied", "").astype(str).str.contains("CRE", na=False)] if "module_applied" in results.columns else results.iloc[0:0]
    if frame.empty:
        return pd.DataFrame()
    for keys, group in frame.groupby([portfolio_field, subsector_field], dropna=False):
        row = {
            "portfolio": keys[0],
            "subsector": keys[1],
            **_population_counts(group, scenario),
            "balance": float(pd.to_numeric(group[balance_field], errors="coerce").sum()),
            "unstressed_dscr": weighted_average(group[dscr_field], group[balance_field]) if dscr_field in group.columns else np.nan,
        }
        for level in levels:
            row[f"stressed_dscr_{level}"] = weighted_average(group.get(f"cre_dscr_{level}", _empty_series(group)), group[balance_field])
            row[f"stressed_ltv_{level}"] = weighted_average(group.get(f"cre_ltv_{level}", _empty_series(group)), group[balance_field])
            bucket_col = f"stressed_bucket_{level}"
            row[f"special_mention_balance_{level}"] = float(pd.to_numeric(group.loc[group[bucket_col] == "Special Mention", balance_field], errors="coerce").sum())
            row[f"substandard_balance_{level}"] = float(pd.to_numeric(group.loc[group[bucket_col] == "Substandard", balance_field], errors="coerce").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_ci_summary(results: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    """Build weighted C&I FCCR and migration summary rows."""
    results = _model_included_rows(results, scenario)
    config = scenario.get("modules", {}).get("C&I", {})
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    sector_field = config.get("sector_field", "ci_sector")
    levels = get_levels(scenario)
    frame = results[results.get("module_applied", "").astype(str).str.contains("C&I", na=False)] if "module_applied" in results.columns else results.iloc[0:0]
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for keys, group in frame.groupby([portfolio_field, sector_field], dropna=False):
        row = {
            "portfolio": keys[0],
            "sector": keys[1],
            **_population_counts(group, scenario),
            "balance": float(pd.to_numeric(group[balance_field], errors="coerce").sum()),
        }
        for level in levels:
            row[f"stressed_fccr_{level}"] = weighted_average(group.get(f"ci_fccr_{level}", _empty_series(group)), group[balance_field])
            bucket_col = f"stressed_bucket_{level}"
            row[f"special_mention_balance_{level}"] = float(pd.to_numeric(group.loc[group[bucket_col] == "Special Mention", balance_field], errors="coerce").sum())
            row[f"substandard_balance_{level}"] = float(pd.to_numeric(group.loc[group[bucket_col] == "Substandard", balance_field], errors="coerce").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_consumer_summary(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    reserve_basis: CeclReserveBasis | None = None,
) -> pd.DataFrame:
    """Build weighted consumer PD/LGD and effective CECL component rows.

    Reported Consumer CECL components use a monotonic carry-forward ladder:
    Base uses the current in-place reserve, and each stressed level retains the prior
    level's borrower contribution when the current level is unavailable or
    lower. This keeps the reported decomposition and CECL reserve aligned while
    preserving raw scope diagnostics from the modeled expected-loss columns.
    """
    results = _model_included_rows(results, scenario)
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    stress_levels = get_levels(scenario)
    levels = ["unstressed"] + stress_levels
    frame = results[results.get("module_applied", "").astype(str).str.contains("Consumer", na=False)] if "module_applied" in results.columns else results.iloc[0:0]
    rows = []
    if frame.empty:
        return pd.DataFrame()
    reserve_basis = reserve_basis or build_cecl_reserve_basis(
        results, scenario, []
    )
    results = _with_consumer_basis_values(results, reserve_basis)
    frame = results[
        results.get("module_applied", "")
        .astype(str)
        .str.contains("Consumer", na=False)
    ]
    for portfolio, group in frame.groupby(portfolio_field, dropna=False):
        basis_values, reserve_field_available = _consumer_basis_values(
            group, scenario, reserve_basis
        )
        effective_components = (
            _consumer_cecl_components(
                group,
                scenario,
                stress_levels,
                basis_values,
            )
            if reserve_field_available
            else {}
        )
        for level in levels:
            suffix = "unstressed" if level == "unstressed" else level
            report_level = "Base" if level == "unstressed" else level
            pd_field = f"consumer_pd_{suffix}"
            lgd_field = f"consumer_lgd_ratio_{suffix}"
            el_field = f"consumer_el_{suffix}"
            balance = float(pd.to_numeric(group[balance_field], errors="coerce").sum())
            el_values = pd.to_numeric(group.get(el_field, _empty_series(group)), errors="coerce")
            scope_mask = el_values.notna()
            if level != "unstressed":
                scope_mask &= ~group.get(
                    f"out_of_scope_{level}",
                    pd.Series(False, index=group.index),
                ).fillna(False).astype(bool)
            in_scope_counts = _population_counts(
                group.loc[scope_mask], scenario
            )
            in_scope_balance = float(pd.to_numeric(group.loc[scope_mask, balance_field], errors="coerce").sum())
            out_of_scope_balance = balance - in_scope_balance
            if reserve_field_available:
                components = effective_components[report_level]
                el = float(components["expected_loss"].sum())
                qualitative_reserve = float(
                    components["qualitative_reserve"].sum()
                )
                proforma_reserve = float(
                    components["proforma_cecl_reserve"].sum()
                )
            else:
                el = (
                    float(el_values.sum(min_count=1))
                    if scope_mask.any()
                    else np.nan
                )
                qualitative_reserve = float(
                    pd.to_numeric(
                        group.get(
                            "consumer_qualitative_reserve",
                            _empty_series(group),
                        ),
                        errors="coerce",
                    ).sum(min_count=1)
                )
                proforma_reserve = np.nan
            rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": report_level,
                    "reserve_basis": "in_place",
                    **_population_counts(group, scenario),
                    "balance": balance,
                    "in_scope_borrower_count": in_scope_counts[
                        "borrower_count"
                    ],
                    **(
                        {
                            "in_scope_loan_count": in_scope_counts[
                                "loan_count"
                            ]
                        }
                        if scenario.get("_targeted_mode")
                        else {}
                    ),
                    "in_scope_balance": in_scope_balance,
                    "out_of_scope_balance": out_of_scope_balance,
                    "weighted_average_pd": weighted_average(group.get(pd_field, _empty_series(group)), group[balance_field]),
                    "weighted_average_lgd_ratio": weighted_average(group.get(lgd_field, _empty_series(group)), group[balance_field]),
                    "expected_loss": el,
                    "expected_loss_ratio": (
                        np.nan if pd.isna(el) else pct(el, balance)
                    ),
                    "qualitative_reserve": qualitative_reserve,
                    "proforma_cecl_reserve": proforma_reserve,
                    "proforma_cecl_ratio": (
                        np.nan
                        if pd.isna(proforma_reserve)
                        else pct(proforma_reserve, balance)
                    ),
                    "calculation_status": (
                        "available"
                        if reserve_field_available
                        else (
                            "unavailable_missing_reserve_field"
                            if reserve_basis.exception_code
                            == "CECL_RESERVE_FIELD_MISSING"
                            else "unavailable_cecl_basis"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _consumer_cecl_components(
    group: pd.DataFrame,
    scenario: Mapping[str, Any],
    stress_levels: List[str],
    base_reserve: pd.Series | None = None,
) -> Dict[str, Dict[str, pd.Series]]:
    """Return borrower-level effective Consumer CECL components by level.

    The selected Base reserve is authoritative. When its quantitative calculation is
    unavailable, the reserve is treated as the residual qualitative
    contribution so the decomposition remains complete. At each later level,
    quantitative expected loss carries forward when unavailable and is floored
    at the prior level. Qualitative reserve is likewise prevented from falling,
    including when a configured qualitative floor first applies under stress.
    Proforma is always rebuilt from those two effective components.
    """
    reserve_field = scenario.get("cecl", {}).get(
        "reserve_field", "cecl_reserve"
    )
    if base_reserve is None:
        if reserve_field not in group.columns:
            return {}
        effective_proforma = pd.to_numeric(
            group[reserve_field], errors="coerce"
        ).fillna(0.0)
    else:
        effective_proforma = pd.to_numeric(
            base_reserve.reindex(group.index), errors="coerce"
        )
        if effective_proforma.isna().any():
            return {}
    base_el = pd.to_numeric(
        group.get("consumer_el_unstressed", _empty_series(group)),
        errors="coerce",
    )
    effective_el = base_el.fillna(0.0)
    effective_qualitative = effective_proforma - effective_el
    qualitative_floor = to_number(
        scenario.get("modules", {})
        .get("Consumer", {})
        .get("qualitative_reserve_floor"),
        np.nan,
    )
    qualitative_floor_values = pd.Series(
        qualitative_floor,
        index=group.index,
        dtype=float,
    )
    components: Dict[str, Dict[str, pd.Series]] = {
        "Base": {
            "expected_loss": effective_el.copy(),
            "qualitative_reserve": effective_qualitative.copy(),
            "proforma_cecl_reserve": effective_proforma.copy(),
        }
    }

    for level in stress_levels:
        raw_el = pd.to_numeric(
            group.get(f"consumer_el_{level}", _empty_series(group)),
            errors="coerce",
        )
        out_of_scope = group.get(
            f"out_of_scope_{level}",
            pd.Series(False, index=group.index),
        ).fillna(False).astype(bool)
        use_calculated_el = (
            raw_el.notna()
            & ~out_of_scope
        )
        candidate_el = raw_el.where(use_calculated_el, effective_el)
        candidate_qualitative = pd.concat(
            [
                effective_qualitative,
                qualitative_floor_values,
            ],
            axis=1,
        ).max(axis=1)
        effective_el = pd.concat(
            [effective_el, candidate_el], axis=1
        ).max(axis=1)
        effective_qualitative = pd.concat(
            [effective_qualitative, candidate_qualitative], axis=1
        ).max(axis=1)
        effective_proforma = effective_el + effective_qualitative
        components[level] = {
            "expected_loss": effective_el.copy(),
            "qualitative_reserve": effective_qualitative.copy(),
            "proforma_cecl_reserve": effective_proforma.copy(),
        }
    return components


def build_out_of_scope_summary(out_of_scope: pd.DataFrame) -> pd.DataFrame:
    """Aggregate loan-level missing-variable details for final reporting."""
    if out_of_scope is None or out_of_scope.empty:
        return pd.DataFrame(columns=["module", "stress_level", "test", "field", "reason", "count"])
    return (
        out_of_scope.groupby(["module", "stress_level", "test", "field", "reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )


def _cecl_methods(results: pd.DataFrame, scenario: Mapping[str, Any]) -> Dict[Any, str]:
    """Resolve CECL method by portfolio.

    Called by `build_cecl_summary`. Commercial portfolios use bucket reserve
    ratios; consumer portfolios use quantitative plus qualitative expected
    loss. Method detection uses the same portfolio field as CECL reporting so
    renamed and rolled-up Consumer portfolios retain that treatment.
    """
    cecl = scenario.get("cecl", {})
    portfolio_field = _cecl_public_portfolio_field(scenario)
    methods: Dict[Any, str] = {}
    configured = cecl.get("portfolios", {})
    for portfolio, spec in configured.items():
        methods[str(portfolio).strip()] = spec.get(
            "method", "bucket_reserve_ratio"
        )
    if "module_applied" in results.columns and portfolio_field in results.columns:
        for portfolio, group in results.groupby(portfolio_field, dropna=False):
            consumer_rows = group["module_applied"].astype(str).str.contains(
                "Consumer", na=False
            )
            if consumer_rows.any() and (~consumer_rows).any():
                raise ValueError(
                    f"CECL portfolio '{portfolio}' mixes Consumer and "
                    "non-Consumer rows; expected-loss and bucket-reserve-ratio "
                    "methods cannot share one CECL portfolio."
                )
            if consumer_rows.any():
                configured_method = methods.get(portfolio)
                if configured_method not in (None, "expected_loss"):
                    raise ValueError(
                        f"Consumer CECL portfolio '{portfolio}' must use the "
                        "'expected_loss' method."
                    )
                methods[portfolio] = "expected_loss"
            elif methods.get(portfolio) == "expected_loss":
                raise ValueError(
                    f"CECL portfolio '{portfolio}' uses the 'expected_loss' "
                    "method but contains no Consumer rows."
                )
    return methods


def _empty_ratio_frame() -> pd.DataFrame:
    """Return an empty CECL ratio frame with the expected columns."""
    return pd.DataFrame(
        columns=[
            "portfolio",
            CECL_LEVEL_TAG_FIELD,
            "bucket",
            "base_balance",
            "base_reserve",
            "reserve_ratio",
            "reserve_basis",
            "current_method",
            "history_enabled",
            "invalid_balance_count",
            "status",
            "exception_code",
        ]
    )


def _reserve_ratio_for(
    portfolio: Any,
    cecl_level_tag: Any,
    bucket: str,
    ratio_df: pd.DataFrame,
    fallback_basis: str = "in_place",
) -> Tuple[float, str, str, str]:
    """Lookup one resolved CECL tag/bucket ratio.

    ``portfolio`` remains an argument for call-site clarity and public-report
    context, but CECL calibration is intentionally keyed by tag and bucket.
    A CECL-level tag may therefore span more than one public report portfolio.
    """
    required = {"portfolio", CECL_LEVEL_TAG_FIELD, "bucket"}
    if ratio_df.empty or not required.issubset(ratio_df.columns):
        return (
            np.nan,
            "unavailable",
            "CECL_RESERVE_RATIO_UNAVAILABLE",
            fallback_basis,
        )
    ratio_match = ratio_df[
        ratio_df[CECL_LEVEL_TAG_FIELD].map(_normalized_cecl_key).eq(
            _normalized_cecl_key(cecl_level_tag)
        )
        & ratio_df["bucket"].astype(str).eq(str(bucket))
    ]
    tag_match = ratio_df[
        ratio_df[CECL_LEVEL_TAG_FIELD].map(_normalized_cecl_key).eq(
            _normalized_cecl_key(cecl_level_tag)
        )
    ]
    basis_label = (
        str(ratio_match["reserve_basis"].iloc[0])
        if not ratio_match.empty and "reserve_basis" in ratio_match.columns
        else (
            str(tag_match["reserve_basis"].iloc[0])
            if not tag_match.empty and "reserve_basis" in tag_match.columns
            else fallback_basis
        )
    )
    if len(ratio_match) > 1:
        return (
            np.nan,
            "unavailable",
            "CECL_RESERVE_RATIO_DUPLICATE",
            basis_label,
        )
    if not ratio_match.empty:
        configured_status = str(
            ratio_match.get(
                "status", pd.Series("available", index=ratio_match.index)
            ).iloc[0]
        ).strip()
        ratio = to_number(ratio_match["reserve_ratio"].iloc[0], np.nan)
        if configured_status in ("", "available") and np.isfinite(ratio):
            return ratio, "available", "", basis_label
    code = (
        str(ratio_match["exception_code"].iloc[0])
        if not ratio_match.empty
        and "exception_code" in ratio_match.columns
        and str(ratio_match["exception_code"].iloc[0]).strip()
        else "CECL_RESERVE_RATIO_UNAVAILABLE"
    )
    return np.nan, "unavailable", code, basis_label


def _consumer_cecl_rows(
    results: pd.DataFrame,
    portfolio: Any,
    scenario: Mapping[str, Any],
    levels: List[str],
    reserve_basis: CeclReserveBasis,
) -> List[Dict[str, Any]]:
    """Use Consumer expected loss as the CECL reserve output.

    Missing or lower stressed contributions carry the prior level forward.
    Detailed missing-input and out-of-scope records remain in their dedicated
    reports without making the CECL summary unavailable or reducing reserve.
    """
    portfolio_field = _cecl_public_portfolio_field(scenario)
    balance_field = scenario["borrower"]["balance_field"]
    positioned_results = _with_consumer_basis_values(
        results, reserve_basis
    )
    group = positioned_results[
        positioned_results[portfolio_field] == portfolio
    ]
    basis_values, reserve_field_available = _consumer_basis_values(
        group, scenario, reserve_basis
    )
    effective_components = (
        _consumer_cecl_components(
            group,
            scenario,
            get_levels(scenario),
            basis_values,
        )
        if reserve_field_available
        else {}
    )
    rows = []
    for level in levels:
        balance = float(pd.to_numeric(group[balance_field], errors="coerce").sum())
        if not reserve_field_available:
            reserve = np.nan
            status = "unavailable"
            exception_code = _basis_exception_for_group(
                group, portfolio, reserve_basis
            )
        else:
            reserve = float(
                effective_components[level][
                    "proforma_cecl_reserve"
                ].sum()
            )
            status = "available"
            exception_code = ""
        rows.append(
            {
                "portfolio": portfolio,
                "stress_level": level,
                "bucket": "Total",
                "method": "expected_loss",
                "reserve_basis": "in_place",
                "balance": balance,
                "reserve_ratio": np.nan,
                "proforma_cecl_reserve": reserve,
                "proforma_cecl_ratio": (
                    np.nan if pd.isna(reserve) else pct(reserve, balance)
                ),
                "cecl_reserve_status": status,
                "exception_code": exception_code,
            }
        )
    return rows


def _basis_exception_for_group(
    group: pd.DataFrame,
    portfolio: Any,
    reserve_basis: CeclReserveBasis,
) -> str:
    """Return the most specific unavailable-basis code for Consumer rows."""
    if reserve_basis.exception_code:
        return reserve_basis.exception_code
    buckets = {
        str(value)
        for value in group.get(
            "base_bucket", pd.Series(index=group.index, dtype=object)
        ).dropna()
    }
    ratios = reserve_basis.ratios
    if not ratios.empty:
        matches = ratios[
            ratios["portfolio"].eq(portfolio)
            & ratios["bucket"].astype(str).isin(buckets)
        ]
        codes = sorted(
            {
                str(value).strip()
                for value in matches.get(
                    "exception_code", pd.Series(dtype=object)
                ).dropna()
                if str(value).strip()
            }
        )
        if codes:
            return ";".join(codes)
    return "CECL_BASIS_PERIOD_UNAVAILABLE"


def _consumer_basis_values(
    group: pd.DataFrame,
    scenario: Mapping[str, Any],
    reserve_basis: CeclReserveBasis,
) -> tuple[pd.Series, bool]:
    """Use the immutable current in-place Consumer basis resolved pre-stress."""
    balance_field = scenario["borrower"]["balance_field"]
    tolerance = to_number(
        scenario.get("cecl", {}).get("zero_balance_tolerance", 1e-9),
        1e-9,
    )
    if _CONSUMER_BASIS_VALUE_FIELD in group.columns:
        values = pd.to_numeric(
            group[_CONSUMER_BASIS_VALUE_FIELD], errors="coerce"
        )
    elif len(reserve_basis.effective_reserve) == len(group):
        values = pd.Series(
            reserve_basis.effective_reserve.to_numpy(),
            index=group.index,
            dtype=float,
        )
    elif (
        reserve_basis.effective_reserve.index.is_unique
        and group.index.is_unique
    ):
        values = pd.to_numeric(
            reserve_basis.effective_reserve.reindex(group.index),
            errors="coerce",
        )
    else:
        values = pd.Series(np.nan, index=group.index, dtype=float)
    values = values.where(np.isfinite(values))
    balances = pd.to_numeric(group[balance_field], errors="coerce")
    invalid_balance = (~np.isfinite(balances)) | balances.lt(-tolerance)
    basis_unavailable = values.isna().any()
    return values.fillna(0.0), not bool(
        invalid_balance.any() or basis_unavailable
    )


def _with_consumer_basis_values(
    frame: pd.DataFrame, reserve_basis: CeclReserveBasis
) -> pd.DataFrame:
    """Attach basis amounts positionally before duplicate-label grouping."""
    out = frame.copy()
    if len(reserve_basis.effective_reserve) == len(out):
        out[_CONSUMER_BASIS_VALUE_FIELD] = (
            reserve_basis.effective_reserve.to_numpy()
        )
    elif (
        reserve_basis.effective_reserve.index.is_unique
        and out.index.is_unique
    ):
        out[_CONSUMER_BASIS_VALUE_FIELD] = (
            reserve_basis.effective_reserve.reindex(out.index).to_numpy()
        )
    else:
        out[_CONSUMER_BASIS_VALUE_FIELD] = np.nan
    return out


def _combined_basis(labels: set[str]) -> str:
    """Return one basis label, or mixed when portfolio methods differ."""
    clean = {str(label).strip() for label in labels if str(label).strip()}
    if not clean:
        return "in_place"
    if len(clean) == 1:
        return next(iter(clean))
    return "mixed"


def _empty_series(group: pd.DataFrame) -> pd.Series:
    """Return an aligned empty numeric series for optional report fields."""
    return pd.Series(index=group.index, dtype=float)


def _population_counts(group: pd.DataFrame, scenario: Mapping[str, Any]) -> Dict[str, int]:
    """Return legacy borrower counts or targeted distinct borrower/loan counts."""
    if not scenario.get("_targeted_mode"):
        return {"borrower_count": int(len(group))}
    borrower_id = scenario["borrower"]["borrower_id_field"]
    return {
        "borrower_count": int(group[borrower_id].nunique(dropna=True))
        if borrower_id in group.columns
        else int(len(group)),
        "loan_count": int(len(group)),
    }
