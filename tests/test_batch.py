from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from stress_engine.batch import expand_batch_scenarios, run_batch_scenarios
from stress_engine.config import load_scenario


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"
BATCH = ROOT / "examples" / "scenario_batch.json"


class _FakeStressEngine:
    """Small output-writing stand-in used to exercise batch cleanup."""

    def __init__(self, scenario, base_dir):
        self.scenario = scenario

    def run(self, output_dir=None, write_outputs=True, run_comparison=True):
        if write_outputs and output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "engine_owned.csv").write_text("value\n1\n", encoding="utf-8")
            (destination / "output_manifest.json").write_text(
                json.dumps({"files": ["engine_owned.csv", "output_manifest.json"]}),
                encoding="utf-8",
            )
        cecl = pd.DataFrame(
            [
                {
                    "portfolio": "Aggregate",
                    "bucket": "Total",
                    "stress_level": "S1",
                    "balance": 1.0,
                    "proforma_cecl_reserve": 0.1,
                    "proforma_cecl_ratio": 0.1,
                    "cecl_reserve_status": "available",
                }
            ]
        )
        return {
            "reports": {"cecl_summary": cecl, "exception_log": pd.DataFrame()},
            "metadata": {"input_files": [], "exception_count": 0, "output_hashes": {}},
        }


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
        self.assertEqual(expanded[-1]["variables"]["modules.Consumer.pd_increase_factor.S2"], 1.75)

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
            self.assertIn("batch_output_manifest.json", output_files)
            self.assertFalse((Path(tmp) / "scenarios").exists())

            summary = pd.read_csv(Path(tmp) / "batch_summary.csv")
            variables = pd.read_csv(Path(tmp) / "batch_variables.csv")
            self.assertEqual(len(summary), 21)
            self.assertEqual(len(variables), 12)
            self.assertIn("delta_cecl_ratio", summary.columns)
            self.assertTrue(summary["run_id"].eq("base").any())

    def test_batch_max_scenario_guardrail(self):
        scenario, _ = load_scenario([SCENARIO, BATCH])
        with patch("stress_engine.batch.itertools.product", side_effect=AssertionError("product materialized")):
            with self.assertRaisesRegex(ValueError, "produced 6 scenarios"):
                expand_batch_scenarios(scenario, max_scenarios=2)

    def test_paired_guardrail_runs_before_child_construction(self):
        scenario = {
            "scenario_id": "paired_guard",
            "first": 0,
            "second": 0,
            "scenario_batch": {
                "mode": "paired",
                "variables": [
                    {"path": "first", "values": [1, 2, 3]},
                    {"path": "second", "values": [4, 5, 6]},
                ],
            },
        }
        with patch("stress_engine.batch._scenario_record", side_effect=AssertionError("child constructed")):
            with self.assertRaisesRegex(ValueError, "produced 3 scenarios"):
                expand_batch_scenarios(scenario, max_scenarios=2)

    def test_reused_batch_directory_removes_only_stale_engine_outputs(self):
        scenario = {
            "scenario_id": "cleanup",
            "value": 0,
            "scenario_batch": {
                "mode": "grid",
                "variables": [{"path": "value", "values": [1, 2, 3]}],
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch("stress_engine.batch.StressEngine", _FakeStressEngine):
            output_dir = Path(tmp)
            run_batch_scenarios(scenario, ROOT, output_dir=output_dir, run_comparison=False)
            retained_user_file = output_dir / "scenarios" / "scenario_0003" / "keep.txt"
            retained_user_file.write_text("user-owned", encoding="utf-8")
            user_file = output_dir / "notes.txt"
            user_file.write_text("user-owned", encoding="utf-8")

            scenario["scenario_batch"]["variables"][0]["values"] = [1]
            run_batch_scenarios(scenario, ROOT, output_dir=output_dir, run_comparison=False)

            self.assertFalse((output_dir / "scenarios" / "scenario_0002").exists())
            retained_directory = output_dir / "scenarios" / "scenario_0003"
            self.assertTrue(retained_user_file.is_file())
            self.assertFalse((retained_directory / "engine_owned.csv").exists())
            self.assertFalse((retained_directory / "output_manifest.json").exists())
            self.assertTrue(user_file.is_file())

            manifest = json.loads((output_dir / "batch_output_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["child_directories"], ["scenarios/base", "scenarios/scenario_0001"])

            run_batch_scenarios(
                scenario,
                ROOT,
                output_dir=output_dir,
                run_comparison=False,
                write_child_outputs=False,
            )
            self.assertFalse((output_dir / "scenarios" / "base").exists())
            self.assertFalse((output_dir / "scenarios" / "scenario_0001").exists())
            self.assertTrue(retained_user_file.is_file())
            self.assertTrue(user_file.is_file())
            manifest = json.loads((output_dir / "batch_output_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["child_directories"], [])

if __name__ == "__main__":
    unittest.main()
