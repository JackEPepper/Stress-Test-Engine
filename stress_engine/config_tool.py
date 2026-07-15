"""Utilities for maintaining scenario configuration files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


LIST_FIELDS = ("values", "multipliers", "deltas")


def batch_config_from_csv(
    csv_path: str | Path,
    *,
    mode: str = "grid",
    output_directory: str | None = None,
    max_scenarios: int = 500,
) -> Dict[str, Any]:
    """Convert a variable-definition CSV into a ``scenario_batch`` payload.

    Each row requires ``path`` and may define exactly one of ``values``,
    ``range_*``, ``linspace_*``, ``multipliers``, or ``deltas``. List cells can
    be JSON arrays or pipe-delimited values.
    """
    source = Path(csv_path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Variable CSV contains no data rows: {source}")

    variables = [_variable_from_row(row, index + 2) for index, row in enumerate(rows)]
    config: Dict[str, Any] = {
        "mode": mode,
        "max_scenarios": max_scenarios,
        "variables": variables,
    }
    if output_directory:
        config["output_directory"] = output_directory
    return {"scenario_batch": config}


def _variable_from_row(row: Mapping[str, str | None], row_number: int) -> Dict[str, Any]:
    path = _cell(row, "path")
    if not path:
        raise ValueError(f"CSV row {row_number} is missing required 'path'.")
    variable: Dict[str, Any] = {"name": _cell(row, "name") or path, "path": path}

    methods = 0
    for field in LIST_FIELDS:
        raw = _cell(row, field)
        if raw:
            variable[field] = _parse_list(raw, row_number, field)
            methods += 1

    range_values = {key: _cell(row, f"range_{key}") for key in ("start", "stop", "step")}
    if any(range_values.values()):
        if not all(range_values.values()):
            raise ValueError(f"CSV row {row_number} must provide range_start, range_stop, and range_step.")
        variable["range"] = {key: _parse_json_scalar(value) for key, value in range_values.items()}
        inclusive = _cell(row, "range_inclusive")
        if inclusive:
            variable["range"]["inclusive"] = _parse_bool(inclusive, row_number, "range_inclusive")
        methods += 1

    linspace_values = {key: _cell(row, f"linspace_{key}") for key in ("start", "stop", "count")}
    if any(linspace_values.values()):
        if not all(linspace_values.values()):
            raise ValueError(f"CSV row {row_number} must provide linspace_start, linspace_stop, and linspace_count.")
        variable["linspace"] = {
            "start": _parse_json_scalar(linspace_values["start"]),
            "stop": _parse_json_scalar(linspace_values["stop"]),
            "count": int(linspace_values["count"]),
        }
        methods += 1

    if methods != 1:
        raise ValueError(f"CSV row {row_number} must define exactly one value-generation method; found {methods}.")
    precision = _cell(row, "precision")
    if precision:
        variable["precision"] = int(precision)
    allow_create = _cell(row, "allow_create")
    if allow_create:
        variable["allow_create"] = _parse_bool(allow_create, row_number, "allow_create")
    return variable


def _cell(row: Mapping[str, str | None], name: str) -> str:
    return str(row.get(name) or "").strip()


def _parse_list(value: str, row_number: int, field: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [_parse_json_scalar(item.strip()) for item in value.split("|") if item.strip()]
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"CSV row {row_number} field '{field}' must be a nonempty JSON array or pipe-delimited list.")
    return parsed


def _parse_json_scalar(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, (dict, list)):
        raise ValueError(f"Expected a scalar value, got: {value}")
    return parsed


def _parse_bool(value: str, row_number: int, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"CSV row {row_number} field '{field}' must be true or false.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and maintain credit stress scenario configuration.")
    commands = parser.add_subparsers(dest="command", required=True)

    csv_parser = commands.add_parser("batch-csv", help="Convert a variable CSV to scenario_batch JSON.")
    csv_parser.add_argument("csv_path")
    csv_parser.add_argument("output_json")
    csv_parser.add_argument("--mode", choices=("grid", "paired"), default="grid")
    csv_parser.add_argument("--output-directory")
    csv_parser.add_argument("--max-scenarios", type=int, default=500)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = batch_config_from_csv(
        args.csv_path,
        mode=args.mode,
        output_directory=args.output_directory,
        max_scenarios=args.max_scenarios,
    )
    _write_json(Path(args.output_json), payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
