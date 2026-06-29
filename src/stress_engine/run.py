"""Run orchestration for the deterministic stress engine."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from stress_engine.aggregation.asset_quality import out_of_scope_summary, summarize
from stress_engine.aggregation.tag_tie_outs import tie_out_tag_populations
from stress_engine.audit.run_log import build_run_metadata
from stress_engine.cleaning.standardize import standardize_tables
from stress_engine.io.loaders import load_input_tables, read_csv, read_json
from stress_engine.io.writers import ensure_dir, write_csv, write_excel, write_json
from stress_engine.stress.formula_selection import apply_selected_stress
from stress_engine.tagging.loan_tags import add_tags, tags_as_list
from stress_engine.tagging.module_selection import select_modules_and_formulas
from stress_engine.validation.rule_sets import validate_dynamic_rules
from stress_engine.validation.schemas import (
    to_frame,
    validate_required_columns,
    validate_required_files,
    validate_unique_loan_ids,
)


def run_engine(
    config_path: Path,
    scenario_path: Path,
    input_dir: Path,
    output_dir: Path,
    external_source_dir: Path | None = None,
) -> dict:
    config = read_json(config_path)
    scenario = read_json(scenario_path)
    external_source_dir = external_source_dir or Path("external_sources")

    file_issues = validate_required_files(input_dir, config["input_files"])
    if file_issues:
        raise RuntimeError(_format_issues(file_issues))

    tables = load_input_tables(input_dir, config["input_files"])
    file_issues.extend(validate_required_columns(tables, config["required_columns"]))
    file_issues.extend(validate_unique_loan_ids(tables["loan_identity"]))
    if file_issues:
        raise RuntimeError(_format_issues(file_issues))

    tables = standardize_tables(tables)
    scenario_date = pd.Timestamp(scenario["as_of_date"])
    working = _join_tables(tables)
    working = add_tags(working, scenario_date, int(config.get("near_term_maturity_days", 365)))
    working = select_modules_and_formulas(working, config)
    working = validate_dynamic_rules(working, config)
    working["tags"] = working.apply(_add_scope_tag, axis=1)
    results = apply_selected_stress(working, scenario, config, tables["fico_pd_table"])
    results = _finalize_result_columns(results, scenario)

    external_paths = _external_paths(external_source_dir, config.get("external_source_files", {}))
    tie_outs = _load_and_tie_out(results, external_paths)
    tie_out_status = "not_run" if tie_outs.empty else ("pass" if (tie_outs["tie_out_status"] == "pass").all() else "fail")
    input_paths = {name: input_dir / file_name for name, file_name in config["input_files"].items()}
    metadata = build_run_metadata(
        config,
        scenario,
        config_path,
        scenario_path,
        input_paths,
        external_paths,
        tables,
        results,
        validation_issue_count=len(file_issues),
        tie_out_status=tie_out_status,
    )

    run_output_dir = output_dir / metadata["run_id"]
    ensure_dir(run_output_dir)
    outputs = _build_outputs(results, tie_outs, metadata)
    for file_name, frame in outputs.items():
        write_csv(run_output_dir / file_name, frame)
    write_json(run_output_dir / "run_metadata.json", metadata)
    write_excel(
        run_output_dir / "stress_report.xlsx",
        {
            "Run Summary": pd.DataFrame([metadata]),
            "Portfolio Summary": outputs["portfolio_summary.csv"],
            "Sector Summary": outputs["sector_summary.csv"],
            "Data Quality": outputs["data_quality_summary.csv"],
            "Out of Scope": outputs["out_of_scope_summary.csv"],
            "Tag Tie-Outs": outputs["tag_population_tie_out.csv"],
            "Scenario Inputs": pd.DataFrame(_flatten_dict(scenario), columns=["key", "value"]),
        },
    )
    return {"metadata": metadata, "output_dir": run_output_dir, "results": results, "outputs": outputs}


def _join_tables(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    loans = tables["loan_identity"].copy()
    cre = tables["cre_collateral"].rename(columns={"debt_service": "debt_service_cre"})
    ci = tables["ci_financials"].rename(columns={"debt_service": "debt_service_ci"})
    loans = loans.merge(cre, on="loan_id", how="left")
    loans = loans.merge(ci, on=["loan_id", "borrower_id"], how="left", suffixes=("", "_ci"))
    return loans


def _add_scope_tag(row: pd.Series) -> str:
    tags = tags_as_list(row.get("tags"))
    tags = [tag for tag in tags if tag not in {"in_scope", "out_of_scope"}]
    tags.append(str(row.get("scope_status")))
    return "|".join(sorted(set(tags)))


def _load_and_tie_out(results: pd.DataFrame, external_paths: Mapping[str, Path]) -> pd.DataFrame:
    target_path = external_paths.get("tag_population_targets")
    if not target_path or not target_path.exists():
        return pd.DataFrame()
    targets = read_csv(target_path)
    for column in ("source_value", "tolerance"):
        targets[column] = pd.to_numeric(targets[column], errors="coerce")
    return tie_out_tag_populations(results, targets)


def _external_paths(external_source_dir: Path, external_source_files: Mapping[str, str]) -> dict[str, Path]:
    return {name: external_source_dir / file_name for name, file_name in external_source_files.items()}


def _build_outputs(results: pd.DataFrame, tie_outs: pd.DataFrame, metadata: Mapping[str, object]) -> dict[str, pd.DataFrame]:
    data_quality = results[
        ["loan_id", "scope_status", "out_of_scope_reasons", "tags", "selected_stress_module", "selected_formula", "maturity_formula"]
    ].copy()
    return {
        "loan_level_results.csv": results,
        "portfolio_summary.csv": summarize(results, ["portfolio"]),
        "sector_summary.csv": summarize(results, ["portfolio", "sector"]),
        "selected_module_summary.csv": summarize(results, ["selected_stress_module"]),
        "data_quality_summary.csv": data_quality,
        "out_of_scope_summary.csv": out_of_scope_summary(results),
        "tag_population_tie_out.csv": tie_outs,
        "run_summary.csv": pd.DataFrame([metadata]),
    }


def _finalize_result_columns(results: pd.DataFrame, scenario: Mapping[str, object]) -> pd.DataFrame:
    finalized = results.copy()
    finalized.insert(0, "scenario_id", scenario.get("scenario_id"))
    for column in (
        "base_dscr",
        "stressed_dscr",
        "dscr_change",
        "base_fixed_charge_coverage",
        "stressed_fixed_charge_coverage",
        "fixed_charge_coverage_change",
        "base_el_rate",
        "base_expected_loss",
        "stressed_el_rate",
        "stressed_expected_loss",
        "expected_loss_change",
    ):
        if column not in finalized.columns:
            finalized[column] = 0.0
    return finalized


def _format_issues(issues: list[object]) -> str:
    return "\n".join(getattr(issue, "message", str(issue)) for issue in issues)


def _flatten_dict(payload: Mapping[str, object], prefix: str = "") -> list[tuple[str, object]]:
    rows = []
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_dict(value, name))
        else:
            rows.append((name, value))
    return rows
