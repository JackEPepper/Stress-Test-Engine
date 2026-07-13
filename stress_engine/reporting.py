"""Portfolio summaries, CECL projection, and report table generation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .modules.overlay import BUCKETS, apply_overlays
from .exceptions import record_exception
from .utils import get_levels, pct, to_number, weighted_average

REPORT_BUCKETS = [*BUCKETS, "Unknown"]

def build_reports(
    results: pd.DataFrame,
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
    out_of_scope: pd.DataFrame,
    exceptions: List[Dict[str, Any]] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Build all report DataFrames from stressed borrower results.

    Called once by `StressEngine.run` after all stress modules finish. This is
    also where overlays and CECL summaries are calculated.
    """
    exceptions = exceptions if exceptions is not None else []
    bucket_summary = build_bucket_summary(results, scenario)
    bucket_summary, overlay_summary = apply_overlays(bucket_summary, results, scenario, exceptions)
    # CECL can group more finely than migration reporting. In the example,
    # migration reports use CRE/C&I, while CECL can use CRE rollup or subsector.
    cecl_bucket_summary = build_cecl_bucket_summary(results, bucket_summary, scenario)
    cecl_summary = build_cecl_summary(results, cecl_bucket_summary, scenario, exceptions)
    reports = {
        "migration_summary": bucket_summary,
        "overlay_summary": overlay_summary,
        "cecl_summary": cecl_summary,
        "cre_summary": build_cre_summary(results, scenario),
        "ci_summary": build_ci_summary(results, scenario),
        "consumer_summary": build_consumer_summary(results, scenario),
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
    frame = results
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
                    "borrower_count": int(len(group)),
                    "source": "model",
                }
            )
    return pd.DataFrame(rows)


def build_cecl_bucket_summary(
    results: pd.DataFrame,
    migration_summary: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the bucket summary used only for CECL reserve calculations.

    Called by `build_reports`. It groups modeled borrowers by `cecl_portfolio`
    and then reinserts overlay portfolio rows from the migration summary.
    """
    cecl = scenario.get("cecl", {})
    cecl_portfolio_field = cecl.get("portfolio_field", scenario["borrower"].get("portfolio_field", "portfolio"))
    summary = build_bucket_summary(results, scenario, cecl_portfolio_field, include_consumer=True)
    overlays = scenario.get("overlays", {})
    if isinstance(overlays, Mapping):
        overlay_portfolios = set(overlays.keys())
    else:
        overlay_portfolios = {item["portfolio"] for item in overlays}
    if overlay_portfolios and not migration_summary.empty:
        summary = summary[~summary["portfolio"].isin(overlay_portfolios)]
        overlay_rows = migration_summary[migration_summary["portfolio"].isin(overlay_portfolios)]
        summary = pd.concat([summary, overlay_rows], ignore_index=True)
    return summary


def build_cecl_summary(
    results: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Calculate proforma CECL reserves from loan-derived reserve ratios.

    Called by `build_reports`. Reserve ratios are always derived from loan data
    grouped by CECL portfolio and base bucket. Missing individual loan reserves
    are treated as zero and logged; missing aggregate bucket ratios remain
    unavailable rather than filled.
    """
    exceptions = exceptions if exceptions is not None else []
    cecl = scenario.get("cecl", {})
    reserve_field = cecl.get("reserve_field", "cecl_reserve")
    portfolio_field = cecl.get("portfolio_field", scenario["borrower"].get("portfolio_field", "portfolio"))
    balance_field = scenario["borrower"]["balance_field"]
    levels = ["Base"] + get_levels(scenario)
    method_by_portfolio = _cecl_methods(results, scenario)
    rows: List[Dict[str, Any]] = []
    zero_balance_tolerance = to_number(cecl.get("zero_balance_tolerance", 1e-9), 1e-9)

    ratio_df = _cecl_ratio_frame(results, scenario, exceptions)

    for portfolio in sorted(bucket_summary["portfolio"].dropna().unique()):
        method = method_by_portfolio.get(portfolio, "bucket_reserve_ratio")
        if method == "expected_loss":
            rows.extend(_consumer_cecl_rows(results, portfolio, scenario, levels, exceptions))
            continue
        portfolio_total_rows = []
        for level in levels:
            level_rows = bucket_summary[(bucket_summary["portfolio"] == portfolio) & (bucket_summary["stress_level"] == level)]
            total_balance = float(pd.to_numeric(level_rows["balance"], errors="coerce").sum())
            total_reserve = 0.0
            unavailable = False
            for _, bucket_row in level_rows.iterrows():
                bucket = bucket_row["bucket"]
                balance = to_number(bucket_row["balance"], 0.0)
                ratio, status, exception_code = _reserve_ratio_for(portfolio, bucket, ratio_df)
                is_positive_balance = balance > zero_balance_tolerance
                if status == "available":
                    # Proforma CECL reserve = stressed bucket balance times the
                    # in-place reserve ratio derived for that CECL portfolio/bucket.
                    reserve = balance * ratio
                    total_reserve += reserve
                    reserve_ratio = ratio
                    proforma_ratio = pct(reserve, balance)
                elif is_positive_balance:
                    reserve = np.nan
                    reserve_ratio = np.nan
                    proforma_ratio = np.nan
                    unavailable = True
                    record_exception(
                        exceptions,
                        "ERROR",
                        "cecl",
                        exception_code,
                        "CECL reserve ratio is unavailable for a positive-balance bucket; proforma CECL reserve was not calculated.",
                        portfolio=portfolio,
                        stress_level=level,
                        bucket=bucket,
                        field=reserve_field,
                    )
                else:
                    reserve = 0.0
                    reserve_ratio = np.nan
                    proforma_ratio = np.nan
                    status = "not_applicable_zero_balance"
                    exception_code = ""
                rows.append(
                    {
                        "portfolio": portfolio,
                        "stress_level": level,
                        "bucket": bucket,
                        "method": method,
                        "balance": balance,
                        "reserve_ratio": reserve_ratio,
                        "proforma_cecl_reserve": reserve,
                        "proforma_cecl_ratio": proforma_ratio,
                        "cecl_reserve_status": status,
                        "exception_code": exception_code,
                    }
                )
            if unavailable:
                total_reserve_value = np.nan
                total_ratio = np.nan
                total_status = "unavailable"
                total_exception_code = "CECL_RESERVE_RATIO_UNAVAILABLE"
            else:
                total_reserve_value = total_reserve
                total_ratio = pct(total_reserve, total_balance)
                total_status = "available"
                total_exception_code = ""
            portfolio_total_rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": level,
                    "bucket": "Total",
                    "method": method,
                    "balance": total_balance,
                    "reserve_ratio": np.nan,
                    "proforma_cecl_reserve": total_reserve_value,
                    "proforma_cecl_ratio": total_ratio,
                    "cecl_reserve_status": total_status,
                    "exception_code": total_exception_code,
                }
            )
        rows.extend(portfolio_total_rows)

    total_df = pd.DataFrame(rows)
    aggregate_rows = []
    for level in levels:
        totals = total_df[(total_df["stress_level"] == level) & (total_df["bucket"] == "Total")]
        balance = float(pd.to_numeric(totals["balance"], errors="coerce").sum())
        aggregate_unavailable = totals.get("cecl_reserve_status", pd.Series(dtype=object)).eq("unavailable").any()
        reserve = np.nan if aggregate_unavailable else float(pd.to_numeric(totals["proforma_cecl_reserve"], errors="coerce").sum())
        aggregate_rows.append(
            {
                "portfolio": "Aggregate",
                "stress_level": level,
                "bucket": "Total",
                "method": "weighted_average",
                "balance": balance,
                "reserve_ratio": np.nan,
                "proforma_cecl_reserve": reserve,
                "proforma_cecl_ratio": np.nan if aggregate_unavailable else pct(reserve, balance),
                "cecl_reserve_status": "unavailable" if aggregate_unavailable else "available",
                "exception_code": "CECL_RESERVE_RATIO_UNAVAILABLE" if aggregate_unavailable else "",
            }
        )
    return pd.concat([total_df, pd.DataFrame(aggregate_rows)], ignore_index=True)


def build_cre_summary(results: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    """Build weighted CRE metric and migration summary rows.

    Called by `build_reports`; values are weighted by borrower balance.
    """
    config = scenario.get("modules", {}).get("CRE", {})
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    subsector_field = config.get("subsector_field", "cre_subsector")
    dscr_field = config.get("tests", {}).get("dscr", {}).get("field", "dscr")
    ltv_field = config.get("current_ltv_field", "ltv")
    levels = get_levels(scenario)
    rows = []
    frame = results[results.get("module_applied", "").astype(str).str.contains("CRE", na=False)] if "module_applied" in results.columns else results.iloc[0:0]
    if frame.empty:
        return pd.DataFrame()
    for keys, group in frame.groupby([portfolio_field, subsector_field], dropna=False):
        row = {
            "portfolio": keys[0],
            "subsector": keys[1],
            "borrower_count": int(len(group)),
            "balance": float(pd.to_numeric(group[balance_field], errors="coerce").sum()),
            "unstressed_dscr": weighted_average(group[dscr_field], group[balance_field]) if dscr_field in group.columns else np.nan,
            "unstressed_ltv": weighted_average(group[ltv_field], group[balance_field]) if ltv_field in group.columns else np.nan,
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
    config = scenario.get("modules", {}).get("C&I", scenario.get("modules", {}).get("CI", {}))
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    sector_field = config.get("sector_field", "ci_sector")
    base_fccr_field = config.get("current_fccr_field", "fccr")
    levels = get_levels(scenario)
    frame = results[results.get("module_applied", "").astype(str).str.contains("C&I", na=False)] if "module_applied" in results.columns else results.iloc[0:0]
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for keys, group in frame.groupby([portfolio_field, sector_field], dropna=False):
        row = {
            "portfolio": keys[0],
            "sector": keys[1],
            "borrower_count": int(len(group)),
            "balance": float(pd.to_numeric(group[balance_field], errors="coerce").sum()),
            "unstressed_fccr": weighted_average(group[base_fccr_field], group[balance_field]) if base_fccr_field in group.columns else np.nan,
        }
        for level in levels:
            row[f"stressed_fccr_{level}"] = weighted_average(group.get(f"ci_fccr_{level}", _empty_series(group)), group[balance_field])
            bucket_col = f"stressed_bucket_{level}"
            row[f"special_mention_balance_{level}"] = float(pd.to_numeric(group.loc[group[bucket_col] == "Special Mention", balance_field], errors="coerce").sum())
            row[f"substandard_balance_{level}"] = float(pd.to_numeric(group.loc[group[bucket_col] == "Substandard", balance_field], errors="coerce").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_consumer_summary(results: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    """Build weighted consumer PD/LGD and expected-loss summary rows."""
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    reserve_field = scenario.get("cecl", {}).get("reserve_field", "cecl_reserve")
    levels = ["unstressed"] + get_levels(scenario)
    frame = results[results.get("module_applied", "").astype(str).str.contains("Consumer", na=False)] if "module_applied" in results.columns else results.iloc[0:0]
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for portfolio, group in frame.groupby(portfolio_field, dropna=False):
        for level in levels:
            suffix = "unstressed" if level == "unstressed" else level
            pd_field = f"consumer_pd_{suffix}"
            lgd_field = f"consumer_lgd_ratio_{suffix}"
            el_field = f"consumer_el_{suffix}"
            balance = float(pd.to_numeric(group[balance_field], errors="coerce").sum())
            el_values = pd.to_numeric(group.get(el_field, _empty_series(group)), errors="coerce")
            scope_mask = el_values.notna()
            in_scope_balance = float(pd.to_numeric(group.loc[scope_mask, balance_field], errors="coerce").sum())
            out_of_scope_balance = balance - in_scope_balance
            el = float(el_values.sum(min_count=1)) if scope_mask.any() else np.nan
            if level == "unstressed":
                proforma_reserve = float(pd.to_numeric(group.get(reserve_field, _empty_series(group)), errors="coerce").fillna(0.0).sum())
            else:
                proforma_values = pd.to_numeric(
                    group.get(f"consumer_proforma_cecl_{suffix}", _empty_series(group)), errors="coerce"
                )
                proforma_reserve = (
                    float(proforma_values.sum(min_count=1))
                    if out_of_scope_balance <= 1e-9 and proforma_values.notna().any()
                    else np.nan
                )
            rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": "Base" if level == "unstressed" else level,
                    "borrower_count": int(len(group)),
                    "balance": balance,
                    "in_scope_borrower_count": int(scope_mask.sum()),
                    "in_scope_balance": in_scope_balance,
                    "out_of_scope_balance": out_of_scope_balance,
                    "weighted_average_pd": weighted_average(group.get(pd_field, _empty_series(group)), group[balance_field]),
                    "weighted_average_lgd_ratio": weighted_average(group.get(lgd_field, _empty_series(group)), group[balance_field]),
                    "expected_loss": el,
                    "expected_loss_ratio": pct(el, balance),
                    "qualitative_reserve": float(
                        pd.to_numeric(group.get("consumer_qualitative_reserve", _empty_series(group)), errors="coerce").sum(min_count=1)
                    ),
                    "proforma_cecl_reserve": proforma_reserve,
                    "proforma_cecl_ratio": pct(proforma_reserve, balance),
                    "calculation_status": "available" if out_of_scope_balance <= 1e-9 else "unavailable_out_of_scope",
                }
            )
    return pd.DataFrame(rows)


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
    ratios; consumer portfolios default to expected loss.
    """
    cecl = scenario.get("cecl", {})
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    methods: Dict[Any, str] = {}
    configured = cecl.get("portfolios", {})
    if isinstance(configured, list):
        configured = {item["portfolio"]: item for item in configured}
    for portfolio, spec in configured.items():
        methods[portfolio] = spec.get("method", "bucket_reserve_ratio")
    if "module_applied" in results.columns:
        for portfolio, group in results.groupby(portfolio_field, dropna=False):
            if group["module_applied"].astype(str).str.contains("Consumer", na=False).any():
                methods.setdefault(portfolio, "expected_loss")
    return methods


def _cecl_ratio_frame(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Derive all CECL reserve ratios from loan-level data."""
    return _cecl_ratio_frame_from_loans(results, scenario, exceptions)


def _cecl_ratio_frame_from_loans(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Calculate reserve ratios by CECL portfolio and base bucket.

    Formula: reserve ratio = sum(loan CECL reserve) / sum(loan balance).
    Called by `_cecl_ratio_frame`.
    """
    cecl = scenario.get("cecl", {})
    reserve_field = cecl.get("reserve_field", "cecl_reserve")
    portfolio_field = cecl.get("portfolio_field", scenario["borrower"].get("portfolio_field", "portfolio"))
    balance_field = scenario["borrower"]["balance_field"]
    if reserve_field not in results.columns:
        record_exception(
            exceptions,
            "ERROR",
            "cecl",
            "CECL_RESERVE_FIELD_MISSING",
            "Configured CECL reserve field is missing; derived reserve ratios are unavailable.",
            field=reserve_field,
        )
        return _empty_ratio_frame()
    ratio_rows = []
    for (portfolio, bucket), group in results.groupby([portfolio_field, "base_bucket"], dropna=False):
        if bucket not in REPORT_BUCKETS:
            continue
        balance = float(pd.to_numeric(group[balance_field], errors="coerce").sum())
        reserve = float(pd.to_numeric(group[reserve_field], errors="coerce").fillna(0.0).sum())
        ratio_rows.append(
            {
                "portfolio": portfolio,
                "bucket": bucket,
                "base_balance": balance,
                "base_reserve": reserve,
                "reserve_ratio": pct(reserve, balance),
            }
        )
    if not ratio_rows:
        return _empty_ratio_frame()
    return pd.DataFrame(ratio_rows)


def _empty_ratio_frame() -> pd.DataFrame:
    """Return an empty CECL ratio frame with the expected columns."""
    return pd.DataFrame(columns=["portfolio", "bucket", "base_balance", "base_reserve", "reserve_ratio"])


def _reserve_ratio_for(portfolio: Any, bucket: str, ratio_df: pd.DataFrame) -> Tuple[float, str, str]:
    """Lookup a derived CECL reserve ratio for one CECL portfolio/bucket."""
    ratio_match = ratio_df[(ratio_df["portfolio"] == portfolio) & (ratio_df["bucket"] == bucket)]
    if not ratio_match.empty and pd.notna(ratio_match["reserve_ratio"].iloc[0]):
        return to_number(ratio_match["reserve_ratio"].iloc[0], 0.0), "available", ""
    return np.nan, "unavailable", "CECL_RESERVE_RATIO_UNAVAILABLE"


def _consumer_cecl_rows(
    results: pd.DataFrame,
    portfolio: Any,
    scenario: Mapping[str, Any],
    levels: List[str],
    exceptions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Use consumer expected loss as the CECL reserve output."""
    portfolio_field = scenario.get("cecl", {}).get(
        "portfolio_field", scenario["borrower"].get("portfolio_field", "portfolio")
    )
    balance_field = scenario["borrower"]["balance_field"]
    reserve_field = scenario.get("cecl", {}).get("reserve_field", "cecl_reserve")
    group = results[results[portfolio_field] == portfolio]
    rows = []
    for level in levels:
        balance = float(pd.to_numeric(group[balance_field], errors="coerce").sum())
        if level == "Base":
            if reserve_field in group.columns:
                reserve = float(pd.to_numeric(group[reserve_field], errors="coerce").fillna(0.0).sum())
                status = "available"
                exception_code = ""
            else:
                reserve = np.nan
                status = "unavailable"
                exception_code = "CECL_RESERVE_FIELD_MISSING"
            in_scope_balance = balance
            out_of_scope_balance = 0.0
        else:
            reserve_values = pd.to_numeric(
                group.get(f"consumer_proforma_cecl_{level}", _empty_series(group)), errors="coerce"
            )
            scope_mask = reserve_values.notna() & ~group.get(
                f"out_of_scope_{level}", pd.Series(False, index=group.index)
            ).fillna(False).astype(bool)
            in_scope_balance = float(pd.to_numeric(group.loc[scope_mask, balance_field], errors="coerce").sum())
            out_of_scope_balance = balance - in_scope_balance
            if out_of_scope_balance > 1e-9 or not scope_mask.any():
                reserve = np.nan
                status = "unavailable"
                exception_code = "CONSUMER_CECL_UNAVAILABLE_OUT_OF_SCOPE"
                record_exception(
                    exceptions,
                    "ERROR",
                    "cecl",
                    exception_code,
                    "Consumer stressed CECL was unavailable because part or all of the portfolio was out of scope.",
                    portfolio=portfolio,
                    stress_level=level,
                    field="consumer_proforma_cecl",
                    details=f"in_scope_balance={in_scope_balance}; out_of_scope_balance={out_of_scope_balance}",
                )
            else:
                reserve = float(reserve_values[scope_mask].sum(min_count=1))
                status = "available"
                exception_code = ""
        rows.append(
            {
                "portfolio": portfolio,
                "stress_level": level,
                "bucket": "Total",
                "method": "expected_loss",
                "balance": balance,
                "reserve_ratio": np.nan,
                "proforma_cecl_reserve": reserve,
                "proforma_cecl_ratio": pct(reserve, balance),
                "cecl_reserve_status": status,
                "exception_code": exception_code,
                "in_scope_balance": in_scope_balance,
                "out_of_scope_balance": out_of_scope_balance,
            }
        )
    return rows


def _empty_series(group: pd.DataFrame) -> pd.Series:
    """Return an aligned empty numeric series for optional report fields."""
    return pd.Series(index=group.index, dtype=float)
