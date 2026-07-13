from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stress_engine.batch import expand_batch_scenarios, run_batch_scenarios
from stress_engine.config import load_scenario


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"
BATCH = ROOT / "examples" / "scenario_batch.json"


class BatchScenarioTest(unittest.TestCase):
    def test_grid_expansion_is_deterministic(self):
        scenario, _ = load_scenario([SCENARIO, BATCH])
        expanded = expand_batch_scenarios(scenario)

        self.assertEqual(len(expanded), 6)
        self.assertEqual(expanded[0]["run_id"], "scenario_0001")
        self.assertEqual(expanded[-1]["run_id"], "scenario_0006")
        self.assertEqual(expanded[0]["variables"]["modules.CRE.tests.dscr.decline.default.S1"], 0.03)
        self.assertEqual(expanded[0]["variables"]["modules.Consumer.pd_increase_factor.S2"], 1.25)
        self.assertEqual(expanded[-1]["variables"]["modules.CRE.tests.dscr.decline.default.S1"], 0.07)
        self.assertEqual(expanded[-1]["variables"]["modules.Consumer.pd_increase_factor.S2"], 1.5)

    def test_batch_run_writes_summary_reports_without_child_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenario, base_dir = load_scenario([SCENARIO, BATCH])
            result = run_batch_scenarios(
                scenario,
                base_dir,
                output_dir=tmp,
                write_outputs=True,
                run_comparison=False,
                write_child_outputs=False,
            )

            self.assertEqual(result["metadata"]["generated_scenario_count"], 6)
            self.assertEqual(result["metadata"]["engine_version"], "0.2.0")
            self.assertTrue(result["metadata"]["input_files"])
            self.assertEqual(len(result["metadata"]["runs"]), 7)
            output_files = {path.name for path in Path(tmp).iterdir()}
            self.assertIn("batch_summary.csv", output_files)
            self.assertIn("batch_variables.csv", output_files)
            self.assertIn("batch_cecl_summary.csv", output_files)
            self.assertIn("batch_exceptions.csv", output_files)
            self.assertIn("batch_metadata.json", output_files)
            self.assertFalse((Path(tmp) / "scenarios").exists())

            summary = pd.read_csv(Path(tmp) / "batch_summary.csv")
            variables = pd.read_csv(Path(tmp) / "batch_variables.csv")
            self.assertEqual(len(summary), 21)
            self.assertEqual(len(variables), 12)
            self.assertIn("delta_cecl_ratio", summary.columns)
            self.assertTrue(summary["run_id"].eq("base").any())

    def test_batch_max_scenario_guardrail(self):
        scenario, _ = load_scenario([SCENARIO, BATCH])
        with self.assertRaises(ValueError):
            expand_batch_scenarios(scenario, max_scenarios=2)

    def test_named_batch_rejects_duplicate_run_ids(self):
        scenario, _ = load_scenario(SCENARIO)
        scenario["scenario_batch"] = {
            "mode": "named",
            "scenarios": [
                {"run_id": "duplicate", "overrides": {"modules.Consumer.pd_increase_factor.S1": 1.2}},
                {"run_id": "duplicate", "overrides": {"modules.Consumer.pd_increase_factor.S1": 1.3}},
            ],
        }
        with self.assertRaises(ValueError):
            expand_batch_scenarios(scenario)


if __name__ == "__main__":
    unittest.main()
