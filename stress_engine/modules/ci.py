"""Commercial and Industrial stress module."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .base import append_module, module_population, record_out_of_scope, targeted_parameter
from ..exceptions import record_exception
from ..utils import (
    get_levels,
    get_metric_cutoffs,
    is_missing,
    lookup_parameter_with_source,
    lower_metric_bucket,
    to_number,
    worse_bucket,
)


DEFAULT_SECTOR_CONFIG = {
    "principal_field": "principal_repayments_paid",
    "include_non_discretionary_dividends": False,
    "use_calculated_cash_paid_for_interest": False,
}


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
    config = scenario.get("modules", {}).get("C&I", {})
    if not config or not config.get("enabled", True):
        return results, pd.DataFrame()
    config = dict(config)
    config["_module_name"] = "C&I"
    fccr_cutoffs = get_metric_cutoffs(scenario, "fccr")

    out = results.copy()
    out["calculated_cash_paid_for_interest"] = np.nan
    out["calculated_cash_paid_for_interest_source"] = pd.Series(
        pd.NA, index=out.index, dtype="object"
    )
    out["calculated_cash_paid_for_interest_fallback_reason"] = pd.Series(
        pd.NA, index=out.index, dtype="object"
    )
    levels = get_levels(scenario)
    mask = module_population(out, scenario, config)
    out_scope: List[Dict[str, Any]] = []
    fallback_events: set[tuple[str, str, str]] = set()
    sector_field = config.get("sector_field", "ci_sector")
    risk_rating_field = scenario.get("borrower", {}).get(
        "risk_rating_field", "risk_rating"
    )
    config["_risk_rating_field"] = risk_rating_field
    fields = {
        "ebitda": "ebitda",
        "cash_taxes": "cash_taxes",
        "cash_distribution": "cash_distribution",
        "cash_dividends": "cash_dividends",
        "discretionary_dividends": "discretionary_cash_dividends_distribution",
        "cash_management_fees": "cash_management_fees",
        "capex": "unfinanced_capex",
        "global_outstanding": "global_total_outstanding",
        "interest": "cash_paid_for_interest",
        "interest_expense": "interest_expense",
        "non_cash_interest_expense": "non_cash_interest_expense",
    }

    for level in levels:
        for idx, row in out.loc[mask].iterrows():
            out.at[idx, "module_applied"] = append_module(out.at[idx, "module_applied"], "C&I")
            base_bucket = row.get("base_bucket", "Unknown")
            sector = row.get(sector_field, "default")
            brg = _brg_key(row.get(risk_rating_field))
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
            if brg is None:
                out.at[idx, f"out_of_scope_{level}"] = True
                record_out_of_scope(
                    out_scope,
                    row,
                    scenario,
                    "C&I",
                    level,
                    "FCCR",
                    [risk_rating_field],
                    "missing_or_invalid_brg",
                )
                record_exception(
                    exceptions,
                    "ERROR",
                    "C&I",
                    "CI_BRG_INVALID",
                    "C&I borrower risk grade was missing or invalid; FCCR was not calculated.",
                    borrower_id=row.get(
                        scenario["borrower"].get("borrower_id_field", "borrower_id")
                    ),
                    portfolio=row.get(
                        scenario["borrower"].get("portfolio_field", "portfolio")
                    ),
                    stress_level=level,
                    module="C&I",
                    field=risk_rating_field,
                    details=(
                        f"sector={sector};risk_rating={row.get(risk_rating_field)};"
                        "expected=integral_1_to_7_or_numeric_8_plus"
                    ),
                )
                continue
            # Resolve and audit the effective assumptions before calculating;
            # invalid configuration fails this borrower/level closed instead
            # of allowing NaNs to leak into a migration bucket.
            reduction, reduction_source = _ebitda_reduction(
                config,
                sector,
                brg,
                level,
            )
            rate_raw, rate_source = lookup_parameter_with_source(
                config.get("interest_rate_stress"), sector, level, np.nan
            )
            interest_rate_stress = to_number(rate_raw, np.nan)
            reduction = targeted_parameter(
                row, scenario, "C&I", "ebitda_reduction", level, reduction
            )
            interest_rate_stress = targeted_parameter(
                row,
                scenario,
                "C&I",
                "interest_rate_stress",
                level,
                interest_rate_stress,
            )
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
                        details=(
                            f"sector={sector};brg={brg or 'invalid'};"
                            f"risk_rating={row.get(risk_rating_field)}"
                        ),
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
            out.at[idx, "calculated_cash_paid_for_interest"] = result["cash_interest"]
            out.at[idx, "calculated_cash_paid_for_interest_source"] = result["cash_interest_source"]
            out.at[idx, "calculated_cash_paid_for_interest_fallback_reason"] = (
                result["cash_interest_fallback_reason"] or pd.NA
            )
            if result["cash_interest_fallback_reason"]:
                record_exception(
                    exceptions,
                    "WARNING",
                    "C&I",
                    "CI_CALCULATED_CASH_INTEREST_FALLBACK",
                    "Calculated cash interest was unavailable or zero; the original "
                    "cash-paid-for-interest field was used.",
                    borrower_id=row.get(scenario["borrower"].get("borrower_id_field", "borrower_id")),
                    portfolio=row.get(scenario["borrower"].get("portfolio_field", "portfolio")),
                    stress_level=level,
                    module="C&I",
                    field=fields["interest"],
                    details=f"reason={result['cash_interest_fallback_reason']}",
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
            bucket = lower_metric_bucket(result["fccr"], fccr_cutoffs)
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
    brg = _brg_key(row.get(config.get("_risk_rating_field", "risk_rating")))
    cash_interest = _cash_interest(row, sector_cfg, fields)
    used_fields = [*_fields_used(sector_cfg, fields), *_cash_interest_fields(cash_interest, fields)]
    missing = [field for field in used_fields if field not in row or pd.isna(row[field])]

    ebitda = _value(row, fields["ebitda"])
    reduction = (
        _ebitda_reduction(config, sector, brg, level)[0]
        if reduction is None
        else reduction
    )
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

    principal_field = sector_cfg["principal_field"]
    if interest_rate_stress is None:
        interest_rate_stress = to_number(
            lookup_parameter_with_source(config.get("interest_rate_stress"), sector, level, np.nan)[0], np.nan
        )
    # Debt service combines stressed incremental interest on global exposure,
    # cash interest, and the sector-specific principal repayment field.
    debt_service = (
        interest_rate_stress * _value(row, fields["global_outstanding"])
        + cash_interest["value"]
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
        "cash_interest": cash_interest["value"],
        "cash_interest_source": cash_interest["source"],
        "cash_interest_fallback_reason": cash_interest["fallback_reason"],
        "out_of_scope": out_of_scope,
        "missing_fields": missing,
        "zero_fields": zero_fields,
        "reason": reason,
    }


def _sector_config(config: Mapping[str, Any], sector: Any) -> tuple[Mapping[str, Any], bool]:
    """Return complete sector formula options and whether a fallback was used."""
    sectors = config.get("sectors", {})
    if not isinstance(sectors, Mapping):
        raise ValueError("modules.C&I.sectors must be a JSON object.")
    if str(sector) in sectors:
        sector_config = sectors[str(sector)]
        used_default = False
    else:
        sector_config = sectors.get("default", {})
        used_default = True
    if not isinstance(sector_config, Mapping):
        source = f"sector '{sector}'" if not used_default else "the default sector"
        raise ValueError(f"modules.C&I.sectors configuration for {source} must be a JSON object.")
    merged = {**DEFAULT_SECTOR_CONFIG, **sector_config}
    principal_field = merged.get("principal_field")
    if not isinstance(principal_field, str) or not principal_field.strip():
        source = f"sector '{sector}'" if not used_default else "the default sector"
        raise ValueError(f"modules.C&I.sectors configuration for {source} requires principal_field.")
    return merged, used_default


def _ebitda_reduction(
    config: Mapping[str, Any],
    sector: Any,
    brg: str | None,
    level: str,
) -> tuple[float, str]:
    """Resolve EBITDA reduction by sector, normalized BRG, and stress level."""
    sector_table, source = _ebitda_reduction_table(config, sector)
    if not isinstance(sector_table, Mapping) or brg is None or brg not in sector_table:
        return np.nan, "missing"
    brg_table = sector_table[brg]
    if isinstance(brg_table, Mapping):
        value = brg_table.get(level)
        return to_number(value, np.nan), source if value is not None else "missing"
    return to_number(brg_table, np.nan), source


def _ebitda_reduction_table(
    config: Mapping[str, Any], sector: Any
) -> tuple[Any, str]:
    """Return the effective sector table and its audit source."""
    table = config.get("ebitda_reduction", {})
    if isinstance(table, Mapping) and str(sector) in table:
        return table[str(sector)], "sector"
    if isinstance(table, Mapping) and "default" in table:
        return table["default"], "default"
    return table, "direct" if isinstance(table, Mapping) else "missing"


def _brg_key(rating: Any) -> str | None:
    """Normalize BRGs 1-7 exactly and cap every finite BRG >=8 at 8."""
    numeric = to_number(rating, np.nan)
    if not np.isfinite(numeric) or numeric < 1:
        return None
    if numeric >= 8:
        return "8"
    if not float(numeric).is_integer():
        return None
    return str(int(numeric))


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
        sector_cfg.get("principal_field", DEFAULT_SECTOR_CONFIG["principal_field"]),
    ]
    if sector_cfg.get("include_non_discretionary_dividends", False):
        used.extend([fields["cash_dividends"], fields["discretionary_dividends"]])
    return used


def _cash_interest(
    row: Mapping[str, Any],
    sector_cfg: Mapping[str, Any],
    fields: Mapping[str, str],
) -> Dict[str, Any]:
    """Choose original or calculated cash interest for debt service."""
    original = _value(row, fields["interest"])
    if not sector_cfg.get("use_calculated_cash_paid_for_interest", False):
        return {
            "value": original,
            "source": fields["interest"],
            "fallback_reason": "",
        }

    interest_expense = row.get(fields["interest_expense"])
    non_cash_interest = row.get(fields["non_cash_interest_expense"])
    if is_missing(interest_expense) and is_missing(non_cash_interest):
        return {
            "value": original,
            "source": fields["interest"],
            "fallback_reason": "alternative_inputs_missing",
        }

    calculated = _value(row, fields["interest_expense"]) - _value(
        row, fields["non_cash_interest_expense"]
    )
    if abs(calculated) < 1e-12:
        return {
            "value": original,
            "source": fields["interest"],
            "fallback_reason": "calculated_value_zero",
        }
    return {
        "value": calculated,
        "source": "interest_expense_less_non_cash_interest_expense",
        "fallback_reason": "",
    }


def _cash_interest_fields(
    cash_interest: Mapping[str, Any], fields: Mapping[str, str]
) -> List[str]:
    """Return only cash-interest inputs that actually drove the calculation."""
    if cash_interest["source"] == fields["interest"]:
        return [fields["interest"]]
    return [fields["interest_expense"], fields["non_cash_interest_expense"]]


def _value(row: Mapping[str, Any], field: str) -> float:
    """Return numeric field value, using zero for C&I's lenient missing rule."""
    return to_number(row.get(field), 0.0)


def _ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, using zero when EBITDA denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator
