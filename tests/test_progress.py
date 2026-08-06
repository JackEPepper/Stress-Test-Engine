from __future__ import annotations

import io
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from stress_engine.config import load_scenario
from stress_engine.engine import StressEngine, _previous_scenarios
from stress_engine.progress import (
    ConsoleProgressReporter,
    ProgressReporter,
    ProgressStep,
    _format_duration,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _RecordingProgress(ProgressReporter):
    def __init__(self) -> None:
        self.planned: list[str] = []
        self.events: list[tuple[str, str]] = []

    def start(self, title: str, steps: Sequence[ProgressStep]) -> None:
        self.events.append(("run_started", title))
        self.planned = [step.key for step in steps]

    @contextmanager
    def step(self, key: str) -> Iterator[None]:
        self.events.append(("step_started", key))
        try:
            yield
        except BaseException:
            self.events.append(("step_failed", key))
            raise
        else:
            self.events.append(("step_completed", key))

    def finish(self, summary: str = "Run complete") -> None:
        self.events.append(("run_completed", summary))


class ConsoleProgressReporterTest(unittest.TestCase):
    def test_reports_planned_estimates_updates_and_actual_timings(self):
        stream = io.StringIO()
        clock = _Clock()
        reporter = ConsoleProgressReporter(
            stream,
            clock=clock,
            initial_seconds_per_weight=1.0,
        )
        reporter.start(
            "example | standard run",
            [
                ProgressStep("load", "Load inputs"),
                ProgressStep("report", "Build reports"),
            ],
        )

        with reporter.step("load"):
            clock.advance(0.5)
            reporter.update("Loaded 3 tables.", completed=1, total=2)
        with reporter.step("report"):
            clock.advance(1.0)
        reporter.finish()

        output = stream.getvalue()
        self.assertIn("Credit Stress Engine | example | standard run", output)
        self.assertIn("estimated runtime 2.0s", output)
        self.assertIn("[1/2] Load inputs ... est. 1.0s | ETA 2.0s", output)
        self.assertIn("Loaded 3 tables. | ETA 1.5s", output)
        self.assertIn("[1/2] Load inputs ... DONE 0.5s", output)
        self.assertIn("[2/2] Build reports ... DONE 1.0s", output)
        self.assertIn("Run complete in 1.5s.", output)

    def test_failed_step_is_visible_and_exception_is_not_swallowed(self):
        stream = io.StringIO()
        clock = _Clock()
        reporter = ConsoleProgressReporter(stream, clock=clock)
        reporter.start("failure", [ProgressStep("load", "Load inputs")])

        with self.assertRaisesRegex(ValueError, "bad input"):
            with reporter.step("load"):
                clock.advance(0.25)
                raise ValueError("bad input")

        self.assertIn("Load inputs ... FAILED after 0.2s", stream.getvalue())

    def test_duration_format_is_compact_across_runtime_scales(self):
        self.assertEqual(_format_duration(0.01), "<0.1s")
        self.assertEqual(_format_duration(1.24), "1.2s")
        self.assertEqual(_format_duration(65), "1m 05s")
        self.assertEqual(_format_duration(3_661), "1h 01m")


class EngineProgressIntegrationTest(unittest.TestCase):
    def test_configured_comparison_paths_are_anchored_to_scenario_folder(self):
        scenario = {
            "comparison": {
                "previous_scenarios": ["prior/scenario.json"],
            }
        }
        self.assertEqual(
            _previous_scenarios(scenario, ROOT / "examples"),
            [str((ROOT / "examples" / "prior" / "scenario.json").resolve())],
        )

    def test_progress_is_ordered_and_does_not_change_outputs(self):
        scenario, base_dir = load_scenario(SCENARIO)
        quiet = StressEngine(scenario, base_dir).run(
            write_outputs=False,
            run_comparison=False,
        )
        reporter = _RecordingProgress()
        visible = StressEngine(scenario, base_dir).run(
            write_outputs=False,
            run_comparison=False,
            progress=reporter,
        )

        self.assertEqual(
            reporter.planned,
            ["inputs", "population", "enrichment", "stress", "reports", "metadata"],
        )
        self.assertEqual(
            [value for event, value in reporter.events if event == "step_completed"],
            reporter.planned,
        )
        self.assertEqual(reporter.events[-1][0], "run_completed")
        self.assertEqual(
            visible["metadata"]["output_hashes"],
            quiet["metadata"]["output_hashes"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
