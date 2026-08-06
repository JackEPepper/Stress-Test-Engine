from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from stress_engine.cli import (
    _add_previous_scenarios,
    _control_summary,
    build_parser,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"


class CommandLineTest(unittest.TestCase):
    def test_cli_comparison_paths_are_resolved_from_working_directory(self):
        scenario = {"comparison": {"previous_scenarios": None}}
        _add_previous_scenarios(scenario, ["prior/scenario.json"])

        configured = scenario["comparison"]["previous_scenarios"]
        self.assertEqual(
            configured,
            [str((Path.cwd() / "prior" / "scenario.json").resolve())],
        )

    def test_parser_has_stable_product_name_and_progress_controls(self):
        parser = build_parser()
        self.assertEqual(parser.prog, "credit-stress")
        self.assertIsNone(parser.parse_args([str(SCENARIO)]).show_progress)
        self.assertTrue(
            parser.parse_args([str(SCENARIO), "--progress"]).show_progress
        )
        self.assertFalse(
            parser.parse_args([str(SCENARIO), "--no-progress"]).show_progress
        )

    def test_forced_progress_uses_stderr_and_summary_uses_stdout(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(
                [
                    str(SCENARIO),
                    "--no-comparison",
                    "--no-write",
                    "--progress",
                ]
            )

        self.assertEqual(status, 0)
        self.assertIn("Completed stress run with 15 borrowers.", stdout.getvalue())
        self.assertIn("Controls: 0 errors, 5 warnings", stdout.getvalue())
        self.assertNotIn("Credit Stress Engine |", stdout.getvalue())
        self.assertIn("Credit Stress Engine | example_2026q2", stderr.getvalue())
        self.assertIn("Load and profile input tables", stderr.getvalue())
        self.assertIn("Stress run complete", stderr.getvalue())

    def test_control_summary_always_exposes_error_and_warning_counts(self):
        self.assertEqual(
            _control_summary(
                {
                    "exception_count": 7,
                    "exception_counts_by_severity": {
                        "ERROR": 1,
                        "WARNING": 2,
                        "INFO": 4,
                    },
                }
            ),
            "Controls: 1 error, 2 warnings, 4 informational events.",
        )
        self.assertEqual(
            _control_summary({"exception_count": 0}),
            "Controls: 0 errors, 0 warnings, 0 informational events.",
        )

    def test_expected_runtime_errors_return_a_concise_message(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        missing = ROOT / "examples" / "missing-scenario.json"
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([str(missing), "--no-progress"])

        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("ERROR:", stderr.getvalue())
        self.assertIn("missing-scenario.json", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_incompatible_openpyxl_returns_a_concise_message(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "stress_engine.cli._run",
            side_effect=ImportError(
                "pandas requires a newer version of openpyxl"
            ),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([str(SCENARIO), "--no-progress"])

        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("ERROR: Excel support is unavailable", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_comparison_failures_are_visible_in_progress_and_controls(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        missing = ROOT / "examples" / "missing-previous.json"
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(
                [
                    str(SCENARIO),
                    "--previous-scenario",
                    str(missing),
                    "--no-write",
                    "--progress",
                ]
            )

        self.assertEqual(status, 0)
        self.assertIn("Controls: 1 error", stdout.getvalue())
        self.assertIn("Comparison finished with 1 error", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
