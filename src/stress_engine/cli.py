"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from stress_engine.run import run_engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic credit stress engine.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--external-source-dir", default=Path("external_sources"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        result = run_engine(args.config, args.scenario, args.input_dir, args.output_dir, args.external_source_dir)
    except Exception as exc:
        print(f"Stress engine failed: {exc}")
        return 1

    metadata = result["metadata"]
    print(f"Run complete: {metadata['run_id']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Out-of-scope loans: {metadata['out_of_scope_loan_count']}")
    print(f"External tie-out status: {metadata['external_source_tie_out_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
