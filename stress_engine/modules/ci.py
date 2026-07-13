"""Commercial and Industrial stress module."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .base import module_population, record_out_of_scope
from ..exceptions import record_exception
from ..utils import get_levels, is_missing, lookup_parameter_with_source, lower_metric_bucket, to_number, worse_bucket


def run_ci(
    results: pd.DataFrame,
    scenario: Mapping[str, Any],
    exceptions: List[Dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run C&I FCCR stress for borrowers whose `primary_module` is C&I.

    Called from `StressEngine.run`. It delegates the actual formula to
    `_calculate_fccr` and records warning exceptions for any lenient zero
    substitutions that still leave the borrower in scope.
    """
    exceptions = exceptions if exceptions is not None else []
    config = scenario.get("modules", {}).get("C&I", scenario.get("modules", {}).get("CI", scenario.get("modules", {}).get("ci", {})))
    if not config or not config.get("enabled", True):
        return results, pd.DataFrame()
    config = dict(config)
    config["_module_name"] = "C&I"

    out = results.copy()
    levels = get_levels(scenario)
    mask = module_population(out, scenario, config)
    out_scope: List[Dict[str, Any]] = []
    fallback_events: set[tuple[str, str, str]] = set()
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
                # Substandard is the highest commercial bucket; keep it fixed.
                out.at[idx, f"stressed_bucket_{level}"] = "Substandard"
                continue
            sector = row.get(sector_field, "default")
            sector_cfg, used_sector_default = _sector_config(config, sector)
            if used_sector_default:
                record_exception(
                    exceptions,
                    "WARNING",
                    "C&I",
                    "CI_SECTOR_DEFAULT_USED",
                    "C&I sector was not explicitly configured; default sector settings were used.",
                    borrower_id=row.get(scenario["borrower"].get("borrower_id_field", "borrower_id")),
                    portfolio=row.get(scenario["borrower"].get("portfolio_field", "portfolio")),
                    stress_level=level,
                    module="C&I",
                    field=sector_field,
                    details=f"sector={sector}",
                )
            reduction, reduction_source = _ebitda_reduction(config, sector, base_bucket, level)
            rate_raw, rate_source = lookup_parameter_with_source(
                config.get("interest_rate_stress"), sector, level, np.nan
            )
            interest_rate_stress = to_number(rate_raw, np.nan)
            _log_ci_default(
                exceptions,
                fallback_events,
                "ebitda_reduction",
                sector,
                level,
                reduction_source,
            )
            _log_ci_default(
                exceptions,
                fallback_events,
                "interest_rate_stress",
                sector,
                level,
                rate_source,
            )
            invalid_assumptions = []
            if is_missing(reduction) or not 0 <= reduction <= 1:
                invalid_assumptions.append("ebitda_reduction")
            if is_missing(interest_rate_stress) or interest_rate_stress < 0:
                invalid_assumptions.append("interest_rate_stress")
            if invalid_assumptions:
                out.at[idx, f"out_of_scope_{level}"] = True
                record_out_of_scope(
                    out_scope,
                    row,
                    scenario,
                    "C&I",
                    level,
                    "FCCR",
                    invalid_assumptions,
                    "missing_or_invalid_scenario_assumption",
                )
                for field in invalid_assumptions:
                    record_exception(
                        exceptions,
                        "ERROR",
                        "C&I",
                        "CI_SCENARIO_ASSUMPTION_INVALID",
                        "C&I scenario assumption was missing or invalid; FCCR was not calculated.",
                        borrower_id=row.get(scenario["borrower"].get("borrower_id_field", "borrower_id")),
                        stress_level=level,
                        module="C&I",
                        field=field,
                        details=f"sector={sector}",
                    )
                continue
            result = _calculate_fccr(
                row,
                config,
                sector_cfg,
                fields,
                sector,
                level,
                reduction=reduction,
                interest_rate_stress=interest_rate_stress,
            )
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
            for missing_field in result["missing_fields"]:
                record_exception(
                    exceptions,
                    "WARNING",
                    "C&I",
                    "CI_MISSING_FIELD_ZERO_SUBSTITUTION",
                    "Missing C&I numeric field was treated as zero under the lenient in-scope rule.",
                    borrower_id=row.get(scenario["borrower"].get("borrower_id_field", "borrower_id")),
                    portfolio=row.get(scenario["borrower"].get("portfolio_field", "portfolio")),
                    stress_level=level,
                    module="C&I",
                    field=missing_field,
                )
            out.at[idx, f"ci_available_cash_flow_{level}"] = result["available_cash_flow"]
            out.at[idx, f"ci_debt_service_{level}"] = result["debt_service"]
            out.at[idx, f"ci_fccr_{level}"] = result["fccr"]
            bucket = lower_metric_bucket(result["fccr"], config.get("cutoffs", {}))
            out.at[idx, f"stressed_bucket_{level}"] = (
                "Unknown" if base_bucket == "Unknown" else worse_bucket(base_bucket, bucket)
            )
    return out, pd.DataFrame(out_scope)


def _calculate_fccr(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    sector_cfg: Mapping[str, Any],
    fields: Mapping[str, str],
    sector: Any,
    level: str,
    reduction: float | None = None,
    interest_rate_stress: float | None = None,
) -> Dict[str, Any]:
    """Calculate stressed available cash flow, debt service, and FCCR.

    Called once per C&I borrower/stress level from `run_ci`. Missing numeric
    inputs are treated as zero by `_value`; the loan is excluded only when the
    final available cash flow is zero/missing or debt service is nonpositive.
    """
    base_bucket = row.get("base_bucket", "Pass")
    missing = [field for field in _fields_used(sector_cfg, fields) if field not in row or pd.isna(row[field])]

    ebitda = _value(row, fields["ebitda"])
    reduction = _ebitda_reduction(config, sector, base_bucket, level)[0] if reduction is None else reduction
    # EBITDA is stressed first; taxes, distributions, and certain dividends
    # are then pro-formed as their original EBITDA ratios times stressed EBITDA.
    stressed_ebitda = ebitda * (1 - reduction)
    tax_ratio = _ratio(_value(row, fields["cash_taxes"]), ebitda)
    distribution_ratio = _ratio(_value(row, fields["cash_distribution"]), ebitda)
    stressed_taxes = tax_ratio * stressed_ebitda
    stressed_distribution = distribution_ratio * stressed_ebitda

    non_disc_dividends = 0.0
    if sector_cfg.get("include_non_discretionary_dividends", False):
        # Middle Market and ABL subtract non-discretionary dividends. The
        # non-discretionary amount is total dividends less discretionary items.
        non_disc = _value(row, fields["cash_dividends"]) - _value(row, fields["discretionary_dividends"])
        non_disc_dividends = _ratio(non_disc, ebitda) * stressed_ebitda

    # Available cash flow is stressed EBITDA less pro-forma cash outflows plus
    # unadjusted management fees and unfinanced CapEx.
    available_cash_flow = (
        stressed_ebitda
        - stressed_taxes
        - stressed_distribution
        - non_disc_dividends
        - _value(row, fields["cash_management_fees"])
        - _value(row, fields["capex"])
    )

    principal_field = sector_cfg.get("principal_field", config.get("principal_repayments_field", "principal_repayments_paid"))
    if interest_rate_stress is None:
        interest_rate_stress = to_number(
            lookup_parameter_with_source(config.get("interest_rate_stress"), sector, level, np.nan)[0], np.nan
        )
    # Debt service combines stressed incremental interest on global exposure,
    # cash interest, and the sector-specific principal repayment field.
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
        "missing_fields": missing,
        "zero_fields": zero_fields,
        "reason": reason,
    }


def _sector_config(config: Mapping[str, Any], sector: Any) -> tuple[Mapping[str, Any], bool]:
    """Return sector-specific formula options and whether default was used."""
    sectors = config.get("sectors", {})
    if str(sector) in sectors:
        return sectors[str(sector)], False
    return sectors.get("default", {}), True


def _ebitda_reduction(config: Mapping[str, Any], sector: Any, bucket: str, level: str) -> tuple[float, str]:
    """Resolve the EBITDA reduction by sector, base bucket, and stress level."""
    table = config.get("ebitda_reduction", {})
    if isinstance(table, Mapping) and str(sector) in table:
        sector_table = table[str(sector)]
        source = "sector"
    elif isinstance(table, Mapping) and "default" in table:
        sector_table = table["default"]
        source = "default"
    else:
        sector_table = table
        source = "scalar" if not isinstance(table, Mapping) else "missing"
    if isinstance(sector_table, Mapping) and bucket in sector_table:
        bucket_table = sector_table[bucket]
        if isinstance(bucket_table, Mapping):
            value = bucket_table.get(level)
            return to_number(value, np.nan), source if value is not None else "missing"
        return to_number(bucket_table, np.nan), source
    value, nested_source = lookup_parameter_with_source(sector_table, sector, level, np.nan)
    return to_number(value, np.nan), source if nested_source != "missing" else "missing"


def _log_ci_default(
    exceptions: List[Dict[str, Any]],
    events: set[tuple[str, str, str]],
    parameter: str,
    sector: Any,
    level: str,
    source: str,
) -> None:
    """Log configured C&I default-table usage once per sector and level."""
    event = (parameter, str(sector), level)
    if source != "default" or event in events:
        return
    events.add(event)
    record_exception(
        exceptions,
        "INFO",
        "C&I",
        "SCENARIO_DEFAULT_PARAMETER_USED",
        "C&I assumption used a configured default value rather than a sector-specific value.",
        stress_level=level,
        module="C&I",
        field=parameter,
        details=f"sector={sector}",
    )


def _fields_used(sector_cfg: Mapping[str, Any], fields: Mapping[str, str]) -> List[str]:
    """List fields whose missing values should be logged for the sector."""
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
    """Return numeric field value, using zero for C&I's lenient missing rule."""
    return to_number(row.get(field), 0.0)


def _ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, using zero when EBITDA denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _append_module(existing: Any, module: str) -> str:
    """Append the module name to the borrower-level module audit field."""
    if not existing:
        return module
    pieces = [item for item in str(existing).split(";") if item]
    if module not in pieces:
        pieces.append(module)
    return ";".join(pieces)
