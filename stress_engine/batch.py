"""Batch scenario expansion and execution."""

from __future__ import annotations

import copy
import itertools
import json
from decimal import Decimal, InvalidOperation
from math import prod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import output_dir_for
from .engine import OUTPUT_MANIFEST_KIND, StressEngine
from .io import write_csv, write_json
from .progress import ProgressReporter, ProgressStep
from .utils import (
    get_json_path,
    hash_json,
    json_safe,
    resolve_path,
    set_json_path_in_place,
    stable_name,
    to_number,
)
from .version import VERSION


_BATCH_MANIFEST_NAME = "batch_output_manifest.json"
_BATCH_MANIFEST_KIND = "stress_engine_batch_outputs"


def run_batch_scenarios(
    scenario: Mapping[str, Any],
    base_dir: str | Path,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
    run_comparison: bool = True,
    max_scenarios: int | None = None,
    mode_override: str | None = None,
    write_child_outputs: bool = True,
    progress: ProgressReporter | None = None,
) -> Dict[str, Any]:
    """Run a base scenario plus optional generated sensitivity scenarios.

    Called by the CLI when `--batch` is supplied. The existing `StressEngine`
    still performs each individual run; this layer only expands scenario JSON,
    assigns deterministic run ids, and aggregates batch-level reports.
    """
    base_dir = Path(base_dir).resolve()
    base_scenario = copy.deepcopy(dict(scenario))
    batch_config = _batch_config(base_scenario)
    if mode_override:
        batch_config["mode"] = mode_override

    # Expansion validates cardinality and numeric bounds before any expensive
    # child run starts, so an unsafe grid fails without partial model outputs.
    expanded = expand_batch_scenarios(
        base_scenario,
        max_scenarios=max_scenarios,
        mode_override=mode_override,
    )
    batch_dir = _batch_output_dir(base_scenario, base_dir, output_dir)
    reporter = progress if progress is not None else ProgressReporter()
    mode = str(mode_override or batch_config.get("mode", "grid")).lower()
    steps = [
        ProgressStep("base", "Run base scenario", 3.0),
        ProgressStep(
            "generated",
            f"Run {len(expanded)} generated scenarios",
            max(len(expanded), 1) * 3.0,
        ),
        ProgressStep("reports", "Aggregate batch reports", 2.0),
        ProgressStep("metadata", "Finalize batch audit metadata", 1.0),
    ]
    if write_outputs:
        steps.append(ProgressStep("outputs", "Write batch output artifacts", 2.5))
    reporter.start(
        f"{base_scenario.get('scenario_id', 'unnamed scenario')} | "
        f"{mode} batch | {len(expanded)} generated scenarios",
        steps,
    )
    reporter.update(
        f"Prepared {len(expanded):,} generated scenarios within the configured guardrail."
    )

    # The unmodified base run is retained as the common delta anchor for every
    # generated scenario, including targeted variant-aware CECL comparisons.
    base_output = batch_dir / "scenarios" / "base" if write_child_outputs else None
    with reporter.step("base"):
        base_result = StressEngine(base_scenario, base_dir).run(
            output_dir=base_output,
            write_outputs=write_outputs and write_child_outputs,
            run_comparison=run_comparison,
        )

    run_records: List[Dict[str, Any]] = []
    result_records: List[Dict[str, Any]] = []
    result_records.append(
        {
            "run_id": "base",
            "scenario_id": str(base_scenario.get("scenario_id", "base")),
            "scenario_label": "Base",
            "is_base": True,
            "variables": {},
            "variable_rows": [],
            "output_dir": str(batch_dir / "scenarios" / "base") if write_child_outputs else "",
            "result": base_result,
        }
    )

    # Child engines stay silent because the batch reporter owns user-facing
    # progress; their full in-memory results remain available for aggregation.
    with reporter.step("generated"):
        for index, record in enumerate(expanded, start=1):
            child_output = (
                batch_dir / "scenarios" / record["run_id"]
                if write_child_outputs
                else None
            )
            result = StressEngine(record["scenario"], base_dir).run(
                output_dir=child_output,
                write_outputs=write_outputs and write_child_outputs,
                run_comparison=run_comparison,
            )
            run_record = dict(record)
            run_record.pop("scenario", None)
            run_records.append(run_record)
            result_records.append(
                {
                    "run_id": record["run_id"],
                    "scenario_id": record["scenario_id"],
                    "scenario_label": record["scenario_label"],
                    "is_base": False,
                    "variables": record["variables"],
                    "variable_rows": record["variable_rows"],
                    "output_dir": (
                        str(child_output) if child_output is not None else ""
                    ),
                    "result": result,
                }
            )
            reporter.update(
                f"Completed scenario {index}/{len(expanded)}: {record['run_id']}.",
                completed=index,
                total=len(expanded),
            )

    # Batch reports are derived only after every child succeeds, preventing a
    # partial sensitivity table from looking like a complete population.
    with reporter.step("reports"):
        reports = _batch_reports(result_records, base_result)
        reporter.update(f"Built {len(reports):,} batch report tables.")
    with reporter.step("metadata"):
        metadata = _batch_metadata(
            base_scenario,
            batch_config,
            expanded,
            reports,
            result_records,
            base_result,
        )
    if write_outputs:
        with reporter.step("outputs"):
            child_directories = (
                [Path("scenarios") / record["run_id"] for record in result_records]
                if write_child_outputs
                else []
            )
            _write_batch_outputs(batch_dir, reports, metadata, child_directories)
            reporter.update("Wrote batch reports and audit metadata.")

    reporter.finish("Batch run complete")
    return {
        "batch_config": batch_config,
        "batch_runs": pd.DataFrame(run_records),
        "base_result": base_result,
        "results": result_records,
        "reports": reports,
        "metadata": metadata,
        "output_dir": batch_dir,
    }


def expand_batch_scenarios(
    scenario: Mapping[str, Any],
    max_scenarios: int | None = None,
    mode_override: str | None = None,
) -> List[Dict[str, Any]]:
    """Expand `scenario_batch` into deterministic child scenario records."""
    base = copy.deepcopy(dict(scenario))
    config = _batch_config(base)
    if not config:
        raise ValueError("Batch execution requires a 'scenario_batch' section.")
    mode = str(mode_override or config.get("mode", "grid")).lower()
    if mode not in {"grid", "paired"}:
        raise ValueError(f"Unsupported scenario_batch mode: {mode}")
    limit = _max_scenario_limit(config, max_scenarios)
    variables = _variable_specs(base, config, mode=mode, max_values=limit)
    scenario_count = _expansion_count(mode, variables)
    if scenario_count > limit:
        raise ValueError(f"Batch expansion produced {scenario_count} scenarios, above max_scenarios={limit}.")

    if mode == "grid":
        records = _grid_records(base, variables)
    else:
        records = _paired_records(base, variables)
    return records


def _max_scenario_limit(config: Mapping[str, Any], override: int | None) -> int:
    """Return a validated positive scenario-generation limit."""
    raw_limit = override if override is not None else config.get("max_scenarios", 500)
    return _positive_integer(raw_limit, "scenario_batch max_scenarios")


def _positive_integer(
    value: Any,
    path: str,
    maximum: int | None = None,
) -> int:
    """Parse a positive integer without lossy float conversion."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{path} must be a positive integer.")
    text = str(value).strip()
    if "e" in text.casefold():
        raise ValueError(f"{path} must be a positive integer without exponent notation.")
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{path} must be a positive integer.") from None
    if not number.is_finite() or number != number.to_integral_value() or number <= 0:
        raise ValueError(f"{path} must be a positive integer.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{path} exceeds max_scenarios={maximum}.")
    return int(number)


def _expansion_count(mode: str, variables: Sequence[Mapping[str, Any]]) -> int:
    """Return child scenario count without constructing child scenarios."""
    lengths = [len(variable["_values"]) for variable in variables]
    if mode == "grid":
        return prod(lengths)
    if len(set(lengths)) > 1:
        raise ValueError("Paired batch mode requires every variable to have the same number of values.")
    return lengths[0] if lengths else 0


def _batch_config(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the optional ``scenario_batch`` configuration."""
    return dict(scenario.get("scenario_batch", {}))


def _variable_specs(
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
    mode: str,
    max_values: int,
) -> List[Dict[str, Any]]:
    """Normalize variable specs and materialize their value lists."""
    variables = []
    grid_count = 1
    paired_count: int | None = None
    raw_variables = config.get("variables", [])
    if not raw_variables:
        raise ValueError("Grid and paired batch modes require at least one variable.")
    for index, raw in enumerate(raw_variables):
        spec = dict(raw)
        if "path" not in spec:
            raise ValueError("Every scenario_batch variable must define a JSON path.")
        spec.setdefault("name", spec["path"])
        spec["_index"] = index
        if not spec.get("allow_create", False):
            get_json_path(scenario, spec["path"])
        spec["_values"] = _variable_values(scenario, spec, max_values=max_values)
        if not spec["_values"]:
            raise ValueError(f"Batch variable '{spec['name']}' produced no values.")
        value_count = len(spec["_values"])
        if mode == "grid":
            grid_count *= value_count
            if grid_count > max_values:
                raise ValueError(
                    f"Batch expansion exceeds max_scenarios={max_values}."
                )
        elif paired_count is None:
            paired_count = value_count
        elif value_count != paired_count:
            raise ValueError(
                "Paired batch mode requires every variable to have the same number of values."
            )
        variables.append(spec)
    return variables


def _variable_values(
    scenario: Mapping[str, Any],
    spec: Mapping[str, Any],
    max_values: int,
) -> List[Any]:
    """Return explicit or generated values for one variable spec."""
    if "values" in spec:
        return _bounded_values(spec["values"], spec, max_values, "explicit")
    if "range" in spec:
        return _range_values(spec["range"], spec, max_values=max_values)
    if "linspace" in spec:
        return _linspace_values(spec["linspace"], spec, max_values=max_values)
    base_value = get_json_path(scenario, spec["path"])
    if "multipliers" in spec:
        base_number = to_number(base_value)
        multipliers = _bounded_values(
            spec["multipliers"], spec, max_values, "multiplier"
        )
        return [
            _clean_numeric(base_number * to_number(multiplier), spec)
            for multiplier in multipliers
        ]
    if "deltas" in spec:
        base_number = to_number(base_value)
        deltas = _bounded_values(
            spec["deltas"], spec, max_values, "delta"
        )
        return [
            _clean_numeric(base_number + to_number(delta), spec)
            for delta in deltas
        ]
    raise ValueError(f"Batch variable '{spec.get('name', spec.get('path'))}' must define values, range, linspace, multipliers, or deltas.")


def _bounded_values(
    values: Iterable[Any],
    variable_spec: Mapping[str, Any],
    max_values: int,
    source: str,
) -> List[Any]:
    """Collect at most ``max_values`` items from one batch value source."""
    if isinstance(values, (str, bytes, Mapping)):
        name = variable_spec.get("name", variable_spec.get("path"))
        raise ValueError(f"Batch variable '{name}' {source} values must be a list.")
    try:
        iterator = iter(values)
    except TypeError:
        name = variable_spec.get("name", variable_spec.get("path"))
        raise ValueError(f"Batch variable '{name}' {source} values must be iterable.") from None
    collected: List[Any] = []
    for value in iterator:
        if len(collected) >= max_values:
            name = variable_spec.get("name", variable_spec.get("path"))
            raise ValueError(
                f"Batch variable '{name}' exceeds max_scenarios={max_values} "
                f"while collecting {source} values."
            )
        collected.append(value)
    return collected


def _range_values(
    range_spec: Mapping[str, Any],
    variable_spec: Mapping[str, Any],
    max_values: int,
) -> List[Any]:
    """Generate inclusive numeric range values with deterministic rounding."""
    start = to_number(range_spec.get("start"))
    stop = to_number(range_spec.get("stop"))
    step = to_number(range_spec.get("step"))
    if not all(np.isfinite(value) for value in (start, stop, step)):
        raise ValueError("Batch range start, stop, and step must be finite numbers.")
    if step == 0:
        raise ValueError("Batch range step must be nonzero.")
    inclusive = bool(range_spec.get("inclusive", True))
    values: List[Any] = []
    current = start
    tolerance = abs(step) / 1_000_000
    while _range_value_in_bounds(current, stop, step, tolerance):
        if inclusive or (current < stop if step > 0 else current > stop):
            values.append(_clean_numeric(current, variable_spec))
            if len(values) > max_values:
                name = variable_spec.get("name", variable_spec.get("path"))
                raise ValueError(
                    f"Batch variable '{name}' exceeds max_scenarios={max_values} while generating range values."
                )
        if current == stop:
            break
        next_value = current + step
        if not np.isfinite(next_value):
            raise ValueError("Batch range arithmetic produced a non-finite value.")
        if next_value == current:
            raise ValueError("Batch range step is too small to advance from the current value.")
        current = next_value
    return values


def _range_value_in_bounds(
    current: float,
    stop: float,
    step: float,
    tolerance: float,
) -> bool:
    """Test a range boundary without overflow-prone adjusted endpoints."""
    if step > 0:
        return current <= stop or current - stop <= tolerance
    return current >= stop or stop - current <= tolerance


def _linspace_values(
    linspace_spec: Mapping[str, Any],
    variable_spec: Mapping[str, Any],
    max_values: int,
) -> List[Any]:
    """Generate evenly spaced numeric values."""
    start = to_number(linspace_spec.get("start"))
    stop = to_number(linspace_spec.get("stop"))
    if not all(np.isfinite(value) for value in (start, stop)):
        raise ValueError("Batch linspace start and stop must be finite numbers.")
    count = _positive_integer(
        linspace_spec.get("count", 0),
        "Batch linspace count",
        maximum=max_values,
    )
    if count == 1:
        return [_clean_numeric(start, variable_spec)]
    values = []
    for index in range(count):
        if index == 0:
            value = start
        elif index == count - 1:
            value = stop
        else:
            fraction = index / (count - 1)
            value = start * (1.0 - fraction) + stop * fraction
        if not np.isfinite(value):
            raise ValueError("Batch linspace interpolation produced a non-finite value.")
        values.append(_clean_numeric(value, variable_spec))
    return values


def _clean_numeric(value: float, spec: Mapping[str, Any]) -> Any:
    """Round generated numeric values for stable JSON and CSV output."""
    precision = int(spec.get("precision", 12))
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(
            f"Batch variable '{spec.get('name', spec.get('path'))}' generated a non-finite value."
        ) from None
    if not np.isfinite(numeric):
        raise ValueError(
            f"Batch variable '{spec.get('name', spec.get('path'))}' generated a non-finite value."
        )
    rounded = round(numeric, precision)
    return int(rounded) if rounded.is_integer() else rounded


def _grid_records(base: Mapping[str, Any], variables: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Expand variables by Cartesian product."""
    records = []
    value_lists = [variable["_values"] for variable in variables]
    for run_index, values in enumerate(itertools.product(*value_lists), start=1):
        overrides = [
            _override_from_variable(variable, value)
            for variable, value in zip(variables, values)
        ]
        records.append(_scenario_record(base, f"scenario_{run_index:04d}", run_index, overrides))
    return records


def _paired_records(base: Mapping[str, Any], variables: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Expand variables by pairing the Nth value of each variable."""
    lengths = {len(variable["_values"]) for variable in variables}
    if len(lengths) > 1:
        raise ValueError("Paired batch mode requires every variable to have the same number of values.")
    records = []
    count = next(iter(lengths), 0)
    for run_index in range(count):
        overrides = [
            _override_from_variable(variable, variable["_values"][run_index])
            for variable in variables
        ]
        records.append(_scenario_record(base, f"scenario_{run_index + 1:04d}", run_index + 1, overrides))
    return records


def _override_from_variable(variable: Mapping[str, Any], value: Any) -> Dict[str, Any]:
    """Build one override row from a variable/value pair."""
    return {
        "name": variable["name"],
        "path": variable["path"],
        "value": value,
        "allow_create": bool(variable.get("allow_create", False)),
    }


def _scenario_record(
    base: Mapping[str, Any],
    run_id: str,
    run_index: int,
    overrides: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Apply overrides to a child scenario and return its audit record."""
    child = copy.deepcopy(dict(base))
    variable_values: Dict[str, Any] = {}
    variable_rows = []
    # Validate paths against the untouched child as each override is applied;
    # the parallel rows preserve both machine-readable and report-friendly audit forms.
    for override in overrides:
        if not override.get("allow_create", False):
            get_json_path(child, override["path"])
        set_json_path_in_place(
            child,
            override["path"],
            override.get("value"),
            allow_create=bool(override.get("allow_create", False)),
        )
        variable_values[str(override["path"])] = override.get("value")
        variable_rows.append(
            {
                "run_id": run_id,
                "variable_name": str(override.get("name", override["path"])),
                "path": str(override["path"]),
                "value": json_safe(override.get("value")),
            }
        )

    # Give every child a stable identity and hash after all overrides so its
    # reports can be traced back to the exact generated scenario.
    base_id = str(base.get("scenario_id", "scenario"))
    scenario_id = f"{base_id}__{run_id}"
    child["scenario_id"] = scenario_id
    child.setdefault("_metadata", {})
    child["_metadata"].update(
        {
            "batch_run_id": run_id,
            "batch_variable_values": variable_values,
            "scenario_hash": hash_json({key: value for key, value in child.items() if key != "_metadata"}),
        }
    )
    return {
        "run_id": run_id,
        "run_index": run_index,
        "scenario_id": scenario_id,
        "scenario_label": run_id,
        "variables": variable_values,
        "variable_rows": variable_rows,
        "scenario": child,
    }


def _batch_reports(records: Sequence[Mapping[str, Any]], base_result: Mapping[str, Any] | None) -> Dict[str, pd.DataFrame]:
    """Build batch-level summary, variables, CECL, and exception reports."""
    base_lookup = _base_cecl_lookup(base_result)
    variable_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    cecl_rows: List[Dict[str, Any]] = []
    exception_rows: List[Dict[str, Any]] = []

    # Preserve full CECL and exception detail, but restrict the compact summary
    # to aggregate-total rows so one child has one headline result per level.
    for record in records:
        variable_columns = _variable_columns(record["variables"])
        for row in record.get("variable_rows", []):
            variable_rows.append(
                {
                    "run_id": record["run_id"],
                    "scenario_id": record["scenario_id"],
                    "scenario_label": record["scenario_label"],
                    "variable_name": row["variable_name"],
                    "path": row["path"],
                    "value": row["value"],
                }
            )

        result = record["result"]
        cecl = result["reports"].get("cecl_summary", pd.DataFrame())
        for row in cecl.to_dict(orient="records"):
            cecl_rows.append(_prefix_record(record, row, variable_columns))

        aggregate = cecl[(cecl.get("portfolio") == "Aggregate") & (cecl.get("bucket") == "Total")] if not cecl.empty else pd.DataFrame()
        for row in aggregate.to_dict(orient="records"):
            variant = row.get("scenario_variant", "baseline")
            base_values = base_lookup.get((variant, row.get("stress_level")), {})
            reserve = to_number(row.get("proforma_cecl_reserve"))
            ratio = to_number(row.get("proforma_cecl_ratio"))
            summary_rows.append(
                {
                    **_run_columns(record),
                    **variable_columns,
                    **(
                        {"scenario_variant": variant}
                        if "scenario_variant" in row
                        else {}
                    ),
                    "stress_level": row.get("stress_level"),
                    "balance": row.get("balance"),
                    "proforma_cecl_reserve": row.get("proforma_cecl_reserve"),
                    "proforma_cecl_ratio": row.get("proforma_cecl_ratio"),
                    "delta_cecl_reserve": reserve - to_number(base_values.get("proforma_cecl_reserve")) if base_values else np.nan,
                    "delta_cecl_ratio": ratio - to_number(base_values.get("proforma_cecl_ratio")) if base_values else np.nan,
                    "cecl_reserve_status": row.get("cecl_reserve_status"),
                    "exception_count": result["metadata"].get("exception_count", 0),
                    "output_dir": record.get("output_dir", ""),
                }
            )

        exceptions = result["reports"].get("exception_log", pd.DataFrame())
        for row in exceptions.to_dict(orient="records"):
            exception_rows.append(_prefix_record(record, row, variable_columns))

    return {
        "batch_summary": pd.DataFrame(summary_rows),
        "batch_variables": pd.DataFrame(variable_rows),
        "batch_cecl_summary": pd.DataFrame(cecl_rows),
        "batch_exceptions": pd.DataFrame(exception_rows),
    }


def _base_cecl_lookup(base_result: Mapping[str, Any] | None) -> Dict[Any, Dict[str, Any]]:
    """Return aggregate base CECL totals keyed by stress level."""
    if not base_result:
        return {}
    cecl = base_result["reports"].get("cecl_summary", pd.DataFrame())
    if cecl.empty:
        return {}
    aggregate = cecl[(cecl["portfolio"] == "Aggregate") & (cecl["bucket"] == "Total")]
    return {
        (row.get("scenario_variant", "baseline"), row["stress_level"]): row
        for row in aggregate.to_dict(orient="records")
    }


def _run_columns(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return common run-identifying columns for batch reports."""
    return {
        "run_id": record["run_id"],
        "scenario_id": record["scenario_id"],
        "scenario_label": record["scenario_label"],
        "is_base": bool(record["is_base"]),
    }


def _prefix_record(record: Mapping[str, Any], row: Mapping[str, Any], variable_columns: Mapping[str, Any]) -> Dict[str, Any]:
    """Prefix one child report row with run and variable identifiers."""
    return {**_run_columns(record), **variable_columns, **dict(row)}


def _variable_columns(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Return wide variable columns for easy pivoting/filtering."""
    return {
        f"var_{stable_name(path)}": json_safe(value)
        for path, value in values.items()
    }


def _batch_metadata(
    scenario: Mapping[str, Any],
    batch_config: Mapping[str, Any],
    expanded: Sequence[Mapping[str, Any]],
    reports: Mapping[str, pd.DataFrame],
    result_records: Sequence[Mapping[str, Any]],
    base_result: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Build metadata for the batch wrapper outputs."""
    metadata_source = base_result or (result_records[0]["result"] if result_records else {})
    exception_report = reports.get("batch_exceptions", pd.DataFrame())
    # Batch controls need both a portfolio-wide severity rollup and child-level
    # hashes/counts for drill-through to the scenario that produced an issue.
    severity_counts = (
        exception_report["severity"].value_counts().sort_index().to_dict()
        if "severity" in exception_report.columns
        else {}
    )
    run_metadata = []
    for record in result_records:
        child_metadata = record["result"].get("metadata", {})
        run_metadata.append(
            {
                "run_id": record["run_id"],
                "scenario_id": record["scenario_id"],
                "is_base": bool(record["is_base"]),
                "variables": json_safe(record.get("variables", {})),
                "scenario_hash": child_metadata.get("scenario_hash"),
                "exception_count": child_metadata.get("exception_count", 0),
                "output_hashes": child_metadata.get("output_hashes", {}),
            }
        )
    return {
        "engine_version": VERSION,
        "base_scenario_id": scenario.get("scenario_id"),
        "scenario_files": scenario.get("_metadata", {}).get("scenario_files", []),
        "base_scenario_hash": scenario.get("_metadata", {}).get("scenario_hash"),
        "batch_hash": hash_json(batch_config),
        "batch_mode": batch_config.get("mode", "grid"),
        "generated_scenario_count": len(expanded),
        "input_files": metadata_source.get("metadata", {}).get("input_files", []),
        "exception_count": int(len(exception_report)),
        "exception_counts_by_severity": {
            str(key): int(value) for key, value in severity_counts.items()
        },
        "runs": run_metadata,
        "report_hashes": {
            name: hash_json(frame.fillna("").to_dict(orient="records"))
            for name, frame in reports.items()
        },
    }


def _write_batch_outputs(
    output_dir: Path,
    reports: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    child_directories: Sequence[Path],
) -> None:
    """Write batch-level CSV/JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_manifest = _read_batch_manifest(output_dir)
    sort_columns = {
        "batch_summary": ["run_id", "scenario_variant", "stress_level"],
        "batch_variables": ["run_id", "path"],
        "batch_cecl_summary": ["run_id", "portfolio", "stress_level", "bucket"],
        "batch_exceptions": ["run_id", "severity", "stage", "code"],
    }
    current_files = {f"{name}.csv" for name in reports}
    current_files.update({"batch_metadata.json", _BATCH_MANIFEST_NAME})
    current_children = {path.as_posix() for path in child_directories}
    for name, frame in reports.items():
        write_csv(frame, output_dir / f"{name}.csv", sort_columns.get(name, []))
    write_json(metadata, output_dir / "batch_metadata.json")
    _remove_stale_batch_outputs(output_dir, prior_manifest, current_files, current_children)
    write_json(
        {
            "engine_version": VERSION,
            "kind": _BATCH_MANIFEST_KIND,
            "files": sorted(current_files),
            "child_directories": sorted(current_children),
        },
        output_dir / _BATCH_MANIFEST_NAME,
    )


def _read_batch_manifest(output_dir: Path) -> Dict[str, Any]:
    """Read an engine-created batch manifest, ignoring invalid or unrelated JSON."""
    manifest = output_dir / _BATCH_MANIFEST_NAME
    if not manifest.is_file():
        return {}
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, dict) or data.get("kind") != _BATCH_MANIFEST_KIND:
        return {}
    return data


def _remove_stale_batch_outputs(
    output_dir: Path,
    prior_manifest: Mapping[str, Any],
    current_files: set[str],
    current_children: set[str],
) -> None:
    """Remove only files and child directories owned by the prior batch run."""
    prior_files = _manifest_strings(prior_manifest, "files")
    for filename in sorted(prior_files - current_files):
        path = _direct_child(output_dir, filename)
        if path is not None and path.is_file():
            path.unlink()

    prior_children = _manifest_strings(prior_manifest, "child_directories")
    for relative in sorted(prior_children - current_children):
        child = _scenario_child(output_dir, relative)
        if child is not None:
            _remove_engine_child_outputs(child)

    scenarios_dir = output_dir / "scenarios"
    if scenarios_dir.is_dir():
        try:
            scenarios_dir.rmdir()
        except OSError:
            pass


def _manifest_strings(manifest: Mapping[str, Any], field: str) -> set[str]:
    """Return one manifest string-list field, treating malformed data as empty."""
    values = manifest.get(field, [])
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str)}


def _direct_child(parent: Path, name: str) -> Path | None:
    """Resolve a direct child filename while rejecting paths and traversal."""
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        return None
    return parent / relative.name


def _scenario_child(output_dir: Path, relative: str) -> Path | None:
    """Resolve a manifest child only when it is directly below ``scenarios``."""
    path = Path(relative)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "scenarios":
        return None
    if path.parts[1] in {"", ".", ".."}:
        return None
    scenarios_dir = (output_dir / "scenarios").resolve()
    child = (output_dir / path).resolve()
    if child.parent != scenarios_dir:
        return None
    return child


def _remove_engine_child_outputs(child: Path) -> None:
    """Remove files named by one child engine manifest, preserving user files."""
    if not child.is_dir():
        return
    manifest = child / "output_manifest.json"
    owned_files: Sequence[Any] = []
    if manifest.is_file():
        try:
            with manifest.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if (
                isinstance(data, dict)
                and data.get("kind") == OUTPUT_MANIFEST_KIND
                and isinstance(data.get("files"), list)
            ):
                owned_files = data["files"]
        except (OSError, ValueError, TypeError):
            pass
    for filename in sorted({str(value) for value in owned_files}):
        path = _direct_child(child, filename)
        if path is not None and path.is_file():
            path.unlink()
    try:
        child.rmdir()
    except OSError:
        pass


def _batch_output_dir(scenario: Mapping[str, Any], base_dir: Path, override: str | Path | None) -> Path:
    """Resolve batch output directory from CLI override or batch config."""
    if override:
        return resolve_path(override, base_dir)
    config = _batch_config(scenario)
    if config.get("output_directory"):
        return resolve_path(config["output_directory"], base_dir)
    single_dir = output_dir_for(dict(scenario), base_dir)
    return single_dir.parent / f"{single_dir.name}_batch"
