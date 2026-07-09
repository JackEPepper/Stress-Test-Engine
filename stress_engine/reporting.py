"""Portfolio summaries, CECL projection, and report table generation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .modules.overlay import BUCKETS, apply_overlays
from .utils import get_levels, pct, to_number, weighted_average


def build_reports(
    results: pd.DataFrame,
    borrowers: pd.DataFrame,
    scenario: Mapping[str, Any],
    out_of_scope: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    bucket_summary = build_bucket_summary(results, scenario)
    bucket_summary, overlay_summary = apply_overlays(bucket_summary, results, scenario)
    cecl_summary = build_cecl_summary(results, bucket_summary, scenario)
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


def build_bucket_summary(results: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    rows: List[Dict[str, Any]] = []
    levels = ["Base"] + get_levels(scenario)
    for level in levels:
        bucket_col = "base_bucket" if level == "Base" else f"stressed_bucket_{level}"
        if bucket_col not in results.columns:
            continue
        for (portfolio, bucket), group in results.groupby([portfolio_field, bucket_col], dropna=False):
            if bucket not in BUCKETS:
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


def build_cecl_summary(
    results: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> pd.DataFrame:
    cecl = scenario.get("cecl", {})
    reserve_field = cecl.get("reserve_field", "cecl_reserve")
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    levels = ["Base"] + get_levels(scenario)
    method_by_portfolio = _cecl_methods(results, scenario)
    rows: List[Dict[str, Any]] = []

    base = results.copy()
    if reserve_field not in base.columns:
        base[reserve_field] = 0.0
    ratio_rows = []
    for (portfolio, bucket), group in base.groupby([portfolio_field, "base_bucket"], dropna=False):
        if bucket not in BUCKETS:
            continue
        balance = float(pd.to_numeric(group[balance_field], errors="coerce").sum())
        reserve = float(pd.to_numeric(group[reserve_field], errors="coerce").sum())
        ratio_rows.append(
            {
                "portfolio": portfolio,
                "bucket": bucket,
                "base_balance": balance,
                "base_reserve": reserve,
                "reserve_ratio": pct(reserve, balance),
            }
        )
    ratio_df = pd.DataFrame(ratio_rows)

    for portfolio in sorted(bucket_summary["portfolio"].dropna().unique()):
        method = method_by_portfolio.get(portfolio, "bucket_reserve_ratio")
        if method == "expected_loss":
            rows.extend(_consumer_cecl_rows(results, portfolio, scenario, levels))
            continue
        portfolio_total_rows = []
        for level in levels:
            level_rows = bucket_summary[(bucket_summary["portfolio"] == portfolio) & (bucket_summary["stress_level"] == level)]
            total_balance = float(pd.to_numeric(level_rows["balance"], errors="coerce").sum())
            total_reserve = 0.0
            for _, bucket_row in level_rows.iterrows():
                bucket = bucket_row["bucket"]
                balance = to_number(bucket_row["balance"], 0.0)
                ratio = _reserve_ratio_for(portfolio, bucket, ratio_df, cecl)
                reserve = balance * ratio
                total_reserve += reserve
                rows.append(
                    {
                        "portfolio": portfolio,
                        "stress_level": level,
                        "bucket": bucket,
                        "method": method,
                        "balance": balance,
                        "reserve_ratio": ratio,
                        "proforma_cecl_reserve": reserve,
                        "proforma_cecl_ratio": pct(reserve, balance),
                    }
                )
            portfolio_total_rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": level,
                    "bucket": "Total",
                    "method": method,
                    "balance": total_balance,
                    "reserve_ratio": np.nan,
                    "proforma_cecl_reserve": total_reserve,
                    "proforma_cecl_ratio": pct(total_reserve, total_balance),
                }
            )
        rows.extend(portfolio_total_rows)

    total_df = pd.DataFrame(rows)
    aggregate_rows = []
    for level in levels:
        totals = total_df[(total_df["stress_level"] == level) & (total_df["bucket"] == "Total")]
        balance = float(pd.to_numeric(totals["balance"], errors="coerce").sum())
        reserve = float(pd.to_numeric(totals["proforma_cecl_reserve"], errors="coerce").sum())
        aggregate_rows.append(
            {
                "portfolio": "Aggregate",
                "stress_level": level,
                "bucket": "Total",
                "method": "weighted_average",
                "balance": balance,
                "reserve_ratio": np.nan,
                "proforma_cecl_reserve": reserve,
                "proforma_cecl_ratio": pct(reserve, balance),
            }
        )
    return pd.concat([total_df, pd.DataFrame(aggregate_rows)], ignore_index=True)


def build_cre_summary(results: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
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
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
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
            el = float(pd.to_numeric(group.get(el_field, _empty_series(group)), errors="coerce").sum())
            rows.append(
                {
                    "portfolio": portfolio,
                    "stress_level": "Base" if level == "unstressed" else level,
                    "borrower_count": int(len(group)),
                    "balance": balance,
                    "weighted_average_pd": weighted_average(group.get(pd_field, _empty_series(group)), group[balance_field]),
                    "weighted_average_lgd_ratio": weighted_average(group.get(lgd_field, _empty_series(group)), group[balance_field]),
                    "expected_loss": el,
                    "expected_loss_ratio": pct(el, balance),
                }
            )
    return pd.DataFrame(rows)


def build_out_of_scope_summary(out_of_scope: pd.DataFrame) -> pd.DataFrame:
    if out_of_scope is None or out_of_scope.empty:
        return pd.DataFrame(columns=["module", "stress_level", "test", "field", "reason", "count"])
    return (
        out_of_scope.groupby(["module", "stress_level", "test", "field", "reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )


def _cecl_methods(results: pd.DataFrame, scenario: Mapping[str, Any]) -> Dict[Any, str]:
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


def _reserve_ratio_for(portfolio: Any, bucket: str, ratio_df: pd.DataFrame, cecl: Mapping[str, Any]) -> float:
    ratio_match = ratio_df[(ratio_df["portfolio"] == portfolio) & (ratio_df["bucket"] == bucket)]
    if not ratio_match.empty and pd.notna(ratio_match["reserve_ratio"].iloc[0]):
        return to_number(ratio_match["reserve_ratio"].iloc[0], 0.0)

    defaults = cecl.get("default_bucket_reserve_ratios", {})
    portfolio_defaults = {}
    if isinstance(defaults, Mapping):
        portfolio_defaults = defaults.get(str(portfolio), {}) if isinstance(defaults.get(str(portfolio), {}), Mapping) else {}
        if bucket in portfolio_defaults:
            return to_number(portfolio_defaults[bucket], 0.0)
        if bucket in defaults:
            return to_number(defaults[bucket], 0.0)

    observed = ratio_df[ratio_df["portfolio"] == portfolio]["reserve_ratio"].dropna()
    if observed.empty:
        return to_number(cecl.get("fallback_reserve_ratio", 0.0), 0.0)
    if bucket == "Pass":
        return float(observed.min())
    if bucket == "Special Mention":
        return float(max(observed.mean(), observed.min()))
    if bucket == "Substandard":
        multiplier = to_number(cecl.get("substandard_fallback_multiplier", 1.5), 1.5)
        return float(observed.max() * multiplier)
    return float(observed.mean())


def _consumer_cecl_rows(results: pd.DataFrame, portfolio: Any, scenario: Mapping[str, Any], levels: List[str]) -> List[Dict[str, Any]]:
    portfolio_field = scenario["borrower"].get("portfolio_field", "portfolio")
    balance_field = scenario["borrower"]["balance_field"]
    group = results[results[portfolio_field] == portfolio]
    rows = []
    for level in levels:
        suffix = "unstressed" if level == "Base" else level
        el_field = f"consumer_el_{suffix}"
        balance = float(pd.to_numeric(group[balance_field], errors="coerce").sum())
        reserve = float(pd.to_numeric(group.get(el_field, _empty_series(group)), errors="coerce").sum())
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
            }
        )
    return rows


def _empty_series(group: pd.DataFrame) -> pd.Series:
    return pd.Series(index=group.index, dtype=float)
