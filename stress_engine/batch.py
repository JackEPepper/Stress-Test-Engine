"""Batch scenario expansion and execution."""

from __future__ import annotations

import copy
import itertools
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import output_dir_for
from .engine import ENGINE_VERSION, StressEngine
from .io import write_csv, write_json
from .utils import hash_json, json_safe, resolve_path, stable_name, to_number


_PATH_TOKEN_RE = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")
_BATCH_SECTION_NAMES = ("scenario_batch", "batch")


def run_batch_scenarios(
    scenario: Mapping[str, Any],
    base_dir: str | Path,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
    run_comparison: bool = True,
    max_scenarios: int | None = None,
    mode_override: str | None = None,
    write_child_outputs: bool = True,
) -> Dict[str, Any]:
    """Run a base scenario plus optional generated sensitivity scenarios.

    Called by the CLI when `--batch` is supplied. The existing `StressEngine`
    still performs each individual run; this layer only expands scenario JSON,
    assigns deterministic run ids, and aggregates batch-level reports.
    """
    base_dir = Path(base_dir).resolve()
    base_scenario = copy.deepcopy(dict(scenario))
    batch_config = _batch_config(base_scenario)
    expanded = expand_batch_scenarios(base_scenario, max_scenarios=max_scenarios, mode_override=mode_override)
    batch_dir = _batch_output_dir(base_scenario, base_dir, output_dir)

    include_base = bool(batch_config.get("include_base", True))
    delta_from_base = bool(batch_config.get("delta_from_base", True))
    base_result = None
    if include_base or delta_from_base:
        base_output = batch_dir / "scenarios" / "base" if write_child_outputs else None
        base_result = StressEngine(base_scenario, base_dir).run(
            output_dir=base_output,
            write_outputs=write_outputs and write_child_outputs,
            run_comparison=run_comparison,
        )

    run_records: List[Dict[str, Any]] = []
    result_records: List[Dict[str, Any]] = []
    if include_base and base_result is not None:
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

    for record in expanded:
        child_output = batch_dir / "scenarios" / record["run_id"] if write_child_outputs else None
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
                "output_dir": str(child_output) if child_output is not None else "",
                "result": result,
            }
        )

    reports = _batch_reports(result_records, base_result)
    metadata = _batch_metadata(base_scenario, batch_config, expanded, reports, result_records, base_result)
    if write_outputs:
        _write_batch_outputs(batch_dir, reports, metadata)

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
    if mode == "named":
        records = _named_records(base, config)
    else:
        variables = _variable_specs(base, config)
        if mode == "grid":
            records = _grid_records(base, variables)
        elif mode == "paired":
            records = _paired_records(base, variables)
        else:
            raise ValueError(f"Unsupported scenario_batch mode: {mode}")

    limit = max_scenarios if max_scenarios is not None else int(config.get("max_scenarios", 500))
    if len(records) > limit:
        raise ValueError(f"Batch expansion produced {len(records)} scenarios, above max_scenarios={limit}.")
    run_ids = [record["run_id"] for record in records]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        raise ValueError(f"Batch scenario run IDs must be unique: {', '.join(duplicates)}")
    return records


def _batch_config(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the optional batch config under supported section names."""
    for name in _BATCH_SECTION_NAMES:
        config = scenario.get(name)
        if config:
            return dict(config)
    return {}


def _variable_specs(scenario: Mapping[str, Any], config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalize variable specs and materialize their value lists."""
    variables = []
    raw_variables = config.get("variables", [])
    if not raw_variables:
        raise ValueError("Grid and paired batch modes require at least one variable.")
    for index, raw in enumerate(raw_variables):
        spec = dict(raw)
        if "path" not in spec:
            raise ValueError("Every scenario_batch variable must define a JSON path.")
        spec.setdefault("name", spec["path"])
        spec["_index"] = index
        spec["_values"] = _variable_values(scenario, spec)
        if not spec["_values"]:
            raise ValueError(f"Batch variable '{spec['name']}' produced no values.")
        if not spec.get("allow_create", False):
            _get_json_path(scenario, spec["path"])
        variables.append(spec)
    return variables


def _variable_values(scenario: Mapping[str, Any], spec: Mapping[str, Any]) -> List[Any]:
    """Return explicit or generated values for one variable spec."""
    if "values" in spec:
        return list(spec["values"])
    if "range" in spec:
        return _range_values(spec["range"], spec)
    if "linspace" in spec:
        return _linspace_values(spec["linspace"], spec)
    base_value = _get_json_path(scenario, spec["path"])
    if "multipliers" in spec:
        base_number = to_number(base_value)
        return [_clean_numeric(base_number * to_number(multiplier), spec) for multiplier in spec["multipliers"]]
    if "deltas" in spec:
        base_number = to_number(base_value)
        return [_clean_numeric(base_number + to_number(delta), spec) for delta in spec["deltas"]]
    raise ValueError(f"Batch variable '{spec.get('name', spec.get('path'))}' must define values, range, linspace, multipliers, or deltas.")


def _range_values(range_spec: Mapping[str, Any], variable_spec: Mapping[str, Any]) -> List[Any]:
    """Generate inclusive numeric range values with deterministic rounding."""
    start = to_number(range_spec.get("start"))
    stop = to_number(range_spec.get("stop"))
    step = to_number(range_spec.get("step"))
    if step == 0 or np.isnan(step):
        raise ValueError("Batch range step must be nonzero.")
    inclusive = bool(range_spec.get("inclusive", True))
    values: List[Any] = []
    current = start
    tolerance = abs(step) / 1_000_000
    while current <= stop + tolerance if step > 0 else current >= stop - tolerance:
        if inclusive or (current < stop if step > 0 else current > stop):
            values.append(_clean_numeric(current, variable_spec))
        current += step
    return values


def _linspace_values(linspace_spec: Mapping[str, Any], variable_spec: Mapping[str, Any]) -> List[Any]:
    """Generate evenly spaced numeric values."""
    start = to_number(linspace_spec.get("start"))
    stop = to_number(linspace_spec.get("stop"))
    count = int(linspace_spec.get("count", 0))
    if count <= 0:
        raise ValueError("Batch linspace count must be positive.")
    if count == 1:
        return [_clean_numeric(start, variable_spec)]
    step = (stop - start) / (count - 1)
    return [_clean_numeric(start + step * index, variable_spec) for index in range(count)]


def _clean_numeric(value: float, spec: Mapping[str, Any]) -> Any:
    """Round generated numeric values for stable JSON and CSV output."""
    precision = int(spec.get("precision", spec.get("round", 12)))
    rounded = round(float(value), precision)
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


def _named_records(base: Mapping[str, Any], config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand explicitly named override sets."""
    records = []
    named = config.get("scenarios", config.get("named_scenarios", []))
    if not named:
        raise ValueError("Named batch mode requires at least one named scenario.")
    for run_index, item in enumerate(named, start=1):
        label = str(item.get("label", item.get("name", f"scenario_{run_index:04d}")))
        run_id = stable_name(item.get("run_id", label)) or f"scenario_{run_index:04d}"
        overrides = _named_overrides(item)
        records.append(_scenario_record(base, run_id, run_index, overrides, label=label))
    return records


def _named_overrides(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalize one named scenario's path/value overrides."""
    raw = item.get("overrides", item.get("variables", {}))
    if isinstance(raw, Mapping):
        return [
            {"name": path, "path": path, "value": value, "allow_create": bool(item.get("allow_create", False))}
            for path, value in raw.items()
        ]
    return [
        {
            "name": override.get("name", override.get("path")),
            "path": override["path"],
            "value": override.get("value"),
            "allow_create": bool(override.get("allow_create", item.get("allow_create", False))),
        }
        for override in raw
    ]


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
    label: str | None = None,
) -> Dict[str, Any]:
    """Apply overrides to a child scenario and return its audit record."""
    child = copy.deepcopy(dict(base))
    variable_values: Dict[str, Any] = {}
    variable_rows = []
    for override in overrides:
        if not override.get("allow_create", False):
            _get_json_path(child, override["path"])
        _set_json_path(child, override["path"], override.get("value"), allow_create=bool(override.get("allow_create", False)))
        variable_values[str(override["path"])] = override.get("value")
        variable_rows.append(
            {
                "run_id": run_id,
                "variable_name": str(override.get("name", override["path"])),
                "path": str(override["path"]),
                "value": json_safe(override.get("value")),
            }
        )

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
        "scenario_label": label or run_id,
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
            base_values = base_lookup.get(row.get("stress_level"), {})
            reserve = to_number(row.get("proforma_cecl_reserve"))
            ratio = to_number(row.get("proforma_cecl_ratio"))
            summary_rows.append(
                {
                    **_run_columns(record),
                    **variable_columns,
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
    return {row["stress_level"]: row for row in aggregate.to_dict(orient="records")}


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
        "engine_version": ENGINE_VERSION,
        "base_scenario_id": scenario.get("scenario_id"),
        "scenario_files": scenario.get("_metadata", {}).get("scenario_files", []),
        "base_scenario_hash": scenario.get("_metadata", {}).get("scenario_hash"),
        "batch_hash": hash_json(batch_config),
        "batch_mode": batch_config.get("mode", "grid"),
        "generated_scenario_count": len(expanded),
        "input_files": metadata_source.get("metadata", {}).get("input_files", []),
        "runs": run_metadata,
        "report_hashes": {
            name: hash_json(frame.fillna("").to_dict(orient="records"))
            for name, frame in reports.items()
        },
    }


def _write_batch_outputs(output_dir: Path, reports: Mapping[str, pd.DataFrame], metadata: Mapping[str, Any]) -> None:
    """Write batch-level CSV/JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sort_columns = {
        "batch_summary": ["run_id", "stress_level"],
        "batch_variables": ["run_id", "path"],
        "batch_cecl_summary": ["run_id", "portfolio", "stress_level", "bucket"],
        "batch_exceptions": ["run_id", "severity", "stage", "code"],
    }
    for name, frame in reports.items():
        write_csv(frame, output_dir / f"{name}.csv", sort_columns.get(name, []))
    write_json(metadata, output_dir / "batch_metadata.json")


def _batch_output_dir(scenario: Mapping[str, Any], base_dir: Path, override: str | Path | None) -> Path:
    """Resolve batch output directory from CLI override or batch config."""
    if override:
        return resolve_path(override, base_dir)
    config = _batch_config(scenario)
    if config.get("output_directory"):
        return resolve_path(config["output_directory"], base_dir)
    single_dir = output_dir_for(dict(scenario), base_dir)
    return single_dir.parent / f"{single_dir.name}_batch"


def _get_json_path(data: Mapping[str, Any], path: str) -> Any:
    """Return a value from dotted/list-index JSON path syntax."""
    cursor: Any = data
    for token in _path_tokens(path):
        cursor = cursor[token]
    return cursor


def _set_json_path(data: Dict[str, Any], path: str, value: Any, allow_create: bool) -> None:
    """Set a JSON path in place, optionally creating missing dict keys."""
    tokens = _path_tokens(path)
    cursor: Any = data
    for token, next_token in zip(tokens[:-1], tokens[1:]):
        if isinstance(token, int):
            cursor = cursor[token]
            continue
        if token not in cursor:
            if not allow_create:
                raise KeyError(path)
            cursor[token] = [] if isinstance(next_token, int) else {}
        cursor = cursor[token]
    final = tokens[-1]
    if isinstance(final, int):
        cursor[final] = copy.deepcopy(value)
    else:
        if final not in cursor and not allow_create:
            raise KeyError(path)
        cursor[final] = copy.deepcopy(value)


def _path_tokens(path: str) -> List[Any]:
    """Tokenize flattened JSON paths such as `modules.CRE.x[0].value`."""
    tokens: List[Any] = []
    for match in _PATH_TOKEN_RE.finditer(path):
        tokens.append(match.group(1) if match.group(1) is not None else int(match.group(2)))
    if not tokens:
        raise ValueError(f"Invalid JSON path: {path}")
    return tokens
