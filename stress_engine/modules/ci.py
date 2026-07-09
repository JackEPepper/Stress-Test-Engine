"""Commercial and Industrial stress module."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .base import module_population, record_out_of_scope
from ..utils import get_levels, is_missing, lookup_parameter, lower_metric_bucket, to_number, worse_bucket


def run_ci(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = scenario.get("modules", {}).get("C&I", scenario.get("modules", {}).get("CI", scenario.get("modules", {}).get("ci", {})))
    if not config or not config.get("enabled", True):
        return results, pd.DataFrame()

    out = results.copy()
    levels = get_levels(scenario)
    mask = module_population(out, scenario, config)
    out_scope: List[Dict[str, Any]] = []
    sector_field = config.get("sector_field", "ci_sector")
    fields = {
        "ebitda": config.get("ebitda_field", "ebitda"),
        "cash_taxes": config.get("cash_taxes_field", "cash_taxes"),
        "cash_distribution": config.get("cash_distribution_field", "cash_distribution"),
        "cash_dividends": config.get("cash_dividends_field", "cash_dividends"),
        "discretionary_dividends": config.get("discretionary_dividends_field", "discretionary_cash_dividends_distribution"),
        "cash_management_fees": config.get("cash_management_fees_field", "cash_management_fees"),
        "capex": config.get("unfinanced_capex_field", "unfinanced_capex"),
        "global_outstanding": config.get("global_total_outstanding_field", "global_total_outstanding"),
        "interest": config.get("cash_paid_interest_field", "cash_paid_for_interest"),
    }

    for level in levels:
        for idx, row in out.loc[mask].iterrows():
            out.at[idx, "module_applied"] = _append_module(out.at[idx, "module_applied"], "C&I")
            base_bucket = row.get("base_bucket", "Unknown")
            if base_bucket == "Substandard":
                out.at[idx, f"stressed_bucket_{level}"] = "Substandard"
                continue
            sector = row.get(sector_field, "default")
            sector_cfg = _sector_config(config, sector)
            result = _calculate_fccr(row, config, sector_cfg, fields, sector, level)
            if result["out_of_scope"]:
                out.at[idx, f"out_of_scope_{level}"] = True
                record_out_of_scope(
                    out_scope,
                    row,
                    scenario,
                    "C&I",
                    level,
                    "FCCR",
                    result["missing_fields"] or result["zero_fields"],
                    result["reason"],
                )
                continue
            out.at[idx, f"ci_available_cash_flow_{level}"] = result["available_cash_flow"]
            out.at[idx, f"ci_debt_service_{level}"] = result["debt_service"]
            out.at[idx, f"ci_fccr_{level}"] = result["fccr"]
            bucket = lower_metric_bucket(result["fccr"], config.get("cutoffs", {}))
            out.at[idx, f"stressed_bucket_{level}"] = worse_bucket(base_bucket, bucket)
    return out, pd.DataFrame(out_scope)


def _calculate_fccr(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    sector_cfg: Mapping[str, Any],
    fields: Mapping[str, str],
    sector: Any,
    level: str,
) -> Dict[str, Any]:
    base_bucket = row.get("base_bucket", "Pass")
    missing = [field for field in _fields_used(sector_cfg, fields) if field not in row or pd.isna(row[field])]

    ebitda = _value(row, fields["ebitda"])
    reduction = _ebitda_reduction(config, sector, base_bucket, level)
    stressed_ebitda = ebitda * (1 - reduction)
    tax_ratio = _ratio(_value(row, fields["cash_taxes"]), ebitda)
    distribution_ratio = _ratio(_value(row, fields["cash_distribution"]), ebitda)
    stressed_taxes = tax_ratio * stressed_ebitda
    stressed_distribution = distribution_ratio * stressed_ebitda

    non_disc_dividends = 0.0
    if sector_cfg.get("include_non_discretionary_dividends", False):
        non_disc = _value(row, fields["cash_dividends"]) - _value(row, fields["discretionary_dividends"])
        non_disc_dividends = _ratio(non_disc, ebitda) * stressed_ebitda

    available_cash_flow = (
        stressed_ebitda
        - stressed_taxes
        - stressed_distribution
        - non_disc_dividends
        - _value(row, fields["cash_management_fees"])
        - _value(row, fields["capex"])
    )

    principal_field = sector_cfg.get("principal_field", config.get("principal_repayments_field", "principal_repayments_paid"))
    interest_rate_stress = to_number(lookup_parameter(config.get("interest_rate_stress"), sector, level), 0.0)
    debt_service = (
        interest_rate_stress * _value(row, fields["global_outstanding"])
        + _value(row, fields["interest"])
        + _value(row, principal_field)
    )
    zero_fields: List[str] = []
    out_of_scope = False
    reason = ""
    if is_missing(available_cash_flow) or abs(available_cash_flow) < 1e-12:
        out_of_scope = True
        zero_fields.append("stressed_available_cash_flow")
        reason = "zero_or_missing_available_cash_flow"
    if is_missing(debt_service) or debt_service <= 0:
        out_of_scope = True
        zero_fields.append("debt_service")
        reason = "zero_or_missing_debt_service" if not reason else f"{reason};zero_or_missing_debt_service"
    fccr = np.nan if out_of_scope else available_cash_flow / debt_service
    return {
        "available_cash_flow": available_cash_flow,
        "debt_service": debt_service,
        "fccr": fccr,
        "out_of_scope": out_of_scope,
        "missing_fields": missing if out_of_scope else [],
        "zero_fields": zero_fields,
        "reason": reason,
    }


def _sector_config(config: Mapping[str, Any], sector: Any) -> Mapping[str, Any]:
    sectors = config.get("sectors", {})
    if str(sector) in sectors:
        return sectors[str(sector)]
    return sectors.get("default", {})


def _ebitda_reduction(config: Mapping[str, Any], sector: Any, bucket: str, level: str) -> float:
    table = config.get("ebitda_reduction", {})
    sector_table = table.get(str(sector), table.get("default", table)) if isinstance(table, Mapping) else table
    if isinstance(sector_table, Mapping) and bucket in sector_table:
        bucket_table = sector_table[bucket]
        if isinstance(bucket_table, Mapping):
            return to_number(bucket_table.get(level), 0.0)
        return to_number(bucket_table, 0.0)
    return to_number(lookup_parameter(sector_table, sector, level), 0.0)


def _fields_used(sector_cfg: Mapping[str, Any], fields: Mapping[str, str]) -> List[str]:
    used = [
        fields["ebitda"],
        fields["cash_taxes"],
        fields["cash_distribution"],
        fields["cash_management_fees"],
        fields["capex"],
        fields["global_outstanding"],
        fields["interest"],
        sector_cfg.get("principal_field", "principal_repayments_paid"),
    ]
    if sector_cfg.get("include_non_discretionary_dividends", False):
        used.extend([fields["cash_dividends"], fields["discretionary_dividends"]])
    return used


def _value(row: Mapping[str, Any], field: str) -> float:
    return to_number(row.get(field), 0.0)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _append_module(existing: Any, module: str) -> str:
    if not existing:
        return module
    pieces = [item for item in str(existing).split(";") if item]
    if module not in pieces:
        pieces.append(module)
    return ";".join(pieces)
