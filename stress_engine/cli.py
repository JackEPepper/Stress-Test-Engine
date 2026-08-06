"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .progress import ConsoleProgressReporter, ProgressReporter
from .version import VERSION


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for scenario execution."""
    parser = argparse.ArgumentParser(
        prog="credit-stress",
        description="Run a JSON-defined credit stress scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  credit-stress examples/scenario.json --no-comparison\n"
            "  credit-stress examples/scenario.json --no-write --progress\n"
            "  credit-stress examples/scenario.json examples/scenario_batch.json "
            "--batch --no-write-child-outputs"
        ),
    )
    # Scenario composition and output controls apply to both standard and batch
    # runs; repeated scenario files are merged in the order supplied.
    parser.add_argument(
        "scenario",
        nargs="+",
        help=(
            "Scenario JSON manifest/file(s). Includes load first; later files "
            "override earlier files."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument(
        "--previous-scenario",
        action="append",
        default=[],
        help=(
            "Previous scenario JSON for scenario/data marginal impact "
            "reporting. May be used multiple times."
        ),
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Disable previous-scenario comparison.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run calculations without writing output files.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Expand and run scenario_batch variables.",
    )
    parser.add_argument(
        "--batch-mode",
        choices=["grid", "paired"],
        help="Override scenario_batch mode.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        help="Maximum generated child scenarios allowed in a batch.",
    )
    parser.add_argument(
        "--no-write-child-outputs",
        action="store_true",
        help=(
            "For batch runs, write only batch-level reports and skip each "
            "child run's full output folder."
        ),
    )
    # Progress is tri-state: explicit on/off wins, while the unset state follows
    # stderr interactivity so redirected machine-readable output stays clean.
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        help=(
            "Show live step timings and ETAs (automatically enabled in an "
            "interactive terminal)."
        ),
    )
    progress.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Suppress live progress while retaining the final summary.",
    )
    parser.set_defaults(show_progress=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point called by `python -m stress_engine`."""
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "openpyxl", "pandas"}:
            raise
        print(
            f"ERROR: Missing runtime dependency '{exc.name}'. "
            "Install the project first with: python -m pip install -e .",
            file=sys.stderr,
            flush=True,
        )
        return 2
    except ImportError as exc:
        if "openpyxl" not in str(exc).casefold():
            raise
        print(
            "ERROR: Excel support is unavailable or incompatible: "
            f"{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


def _run(args: argparse.Namespace) -> int:
    """Execute parsed CLI arguments and print a concise completion summary."""
    scenario_paths = [Path(item) for item in args.scenario]
    _preflight_scenario_paths(scenario_paths)

    # Keep heavy numerical imports out of parser startup so `--help` and
    # `--version` remain available before runtime setup is complete.
    from .batch import run_batch_scenarios
    from .config import load_scenario
    from .engine import StressEngine

    scenario, base_dir = load_scenario(scenario_paths)
    _add_previous_scenarios(scenario, args.previous_scenario)
    progress = _progress_reporter(args.show_progress)

    if args.batch:
        result = run_batch_scenarios(
            scenario,
            base_dir,
            output_dir=args.output_dir,
            write_outputs=not args.no_write,
            run_comparison=not args.no_comparison,
            max_scenarios=args.max_scenarios,
            mode_override=args.batch_mode,
            write_child_outputs=not args.no_write_child_outputs,
            progress=progress,
        )
        print(
            "Completed batch run with "
            f"{result['metadata']['generated_scenario_count']} generated scenarios."
        )
        print(_control_summary(result["metadata"]))
        if not args.no_write:
            print(f"Batch outputs written to: {_display_path(result['output_dir'])}")
        return 0

    engine = StressEngine(scenario, base_dir)
    result = engine.run(
        output_dir=args.output_dir,
        write_outputs=not args.no_write,
        run_comparison=not args.no_comparison,
        progress=progress,
    )
    outputs = (
        args.output_dir
        if args.output_dir
        else scenario.get("outputs", {}).get("directory", "outputs/latest")
    )
    if "variant_results" in result:
        variant_count = result["variant_results"]["scenario_variant"].nunique()
        print(
            f"Completed targeted stress run with {len(result['results'])} loans "
            f"in primary variant and {variant_count} variants."
        )
    else:
        print(f"Completed stress run with {len(result['results'])} borrowers.")
    print(_control_summary(result["metadata"]))
    if not args.no_write:
        destination = (
            Path(outputs).resolve()
            if Path(outputs).is_absolute()
            else (base_dir / outputs).resolve()
        )
        print(f"Outputs written to: {_display_path(destination)}")
    return 0


def _add_previous_scenarios(
    scenario: dict[str, Any], previous_scenarios: list[str]
) -> None:
    """Append repeatable CLI comparison paths to the merged scenario."""
    if not previous_scenarios:
        return
    comparison = scenario.get("comparison")
    if comparison is None:
        comparison = {}
        scenario["comparison"] = comparison
    if not isinstance(comparison, dict):
        raise ValueError("Scenario comparison must be a JSON object.")
    existing = comparison.get("previous_scenarios", [])
    if existing is None:
        existing = []
    if isinstance(existing, str):
        existing = [existing]
    elif not isinstance(existing, list):
        raise ValueError(
            "Scenario comparison.previous_scenarios must be a string or list."
        )
    comparison["previous_scenarios"] = (
        list(existing) + [str(_absolute_cli_path(path)) for path in previous_scenarios]
    )


def _absolute_cli_path(path: str | Path) -> Path:
    """Resolve a CLI path from the working directory, even when it is missing."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _preflight_scenario_paths(paths: list[Path]) -> None:
    """Fail clearly on missing top-level files before numerical imports."""
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Scenario file not found: {path.resolve()}")


def _progress_reporter(show_progress: bool | None) -> ProgressReporter:
    """Enable progress in terminals, with explicit CLI flags taking priority."""
    if show_progress is None:
        is_terminal = getattr(sys.stderr, "isatty", lambda: False)
        show_progress = bool(is_terminal())
    return ConsoleProgressReporter() if show_progress else ProgressReporter()


def _control_summary(metadata: dict[str, Any]) -> str:
    """Format exception counts without implying that warnings are successes."""
    counts = metadata.get("exception_counts_by_severity", {})
    errors = int(counts.get("ERROR", 0))
    warnings = int(counts.get("WARNING", 0))
    information = int(counts.get("INFO", 0))
    return (
        f"Controls: {_count_label(errors, 'error')}, "
        f"{_count_label(warnings, 'warning')}, "
        f"{_count_label(information, 'informational event')}."
    )


def _count_label(count: int, singular: str) -> str:
    """Return a count with a grammatically singular or plural label."""
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def _display_path(path: str | Path) -> Path:
    """Prefer a concise working-directory-relative path when available."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return resolved


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
