"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_scenario
from .engine import StressEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a JSON-defined credit stress scenario.")
    parser.add_argument("scenario", nargs="+", help="Scenario JSON file(s). Later files override earlier files.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument(
        "--previous-scenario",
        action="append",
        default=[],
        help="Previous scenario JSON for scenario/data marginal impact reporting. May be used multiple times.",
    )
    parser.add_argument("--no-comparison", action="store_true", help="Disable previous-scenario comparison.")
    parser.add_argument("--no-write", action="store_true", help="Run calculations without writing output files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario, base_dir = load_scenario([Path(item) for item in args.scenario])
    if args.previous_scenario:
        scenario.setdefault("comparison", {})
        existing = scenario["comparison"].get("previous_scenarios", [])
        if isinstance(existing, str):
            existing = [existing]
        scenario["comparison"]["previous_scenarios"] = list(existing) + args.previous_scenario
    engine = StressEngine(scenario, base_dir)
    result = engine.run(
        output_dir=args.output_dir,
        write_outputs=not args.no_write,
        run_comparison=not args.no_comparison,
    )
    outputs = scenario.get("outputs", {}).get("directory", "outputs/latest") if not args.output_dir else args.output_dir
    print(f"Completed stress run with {len(result['results'])} borrowers.")
    if not args.no_write:
        print(f"Outputs written to: {Path(outputs).resolve() if Path(outputs).is_absolute() else (base_dir / outputs).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
