from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from stress_engine.batch import (
    _max_scenario_limit,
    _remove_engine_child_outputs,
    expand_batch_scenarios,
    run_batch_scenarios,
)
from stress_engine.config import load_scenario
from stress_engine.engine import OUTPUT_MANIFEST_KIND, StressEngine


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"
BATCH = ROOT / "examples" / "scenario_batch.json"


class _ExplodesPastBound:
    def __iter__(self):
        yield 1
        yield 2
        yield 3
        raise AssertionError("ITERATED_PAST_CAP")


class _MustNotIterate:
    def __iter__(self):
        raise AssertionError("UNNECESSARY_VARIABLE_MATERIALIZED")


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
                json.dumps(
                    {
                        "kind": OUTPUT_MANIFEST_KIND,
                        "files": ["engine_owned.csv", "output_manifest.json"],
                    }
                ),
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

        self.assertEqual(len(expanded), 48)
        self.assertEqual(expanded[0]["run_id"], "scenario_0001")
        self.assertEqual(expanded[-1]["run_id"], "scenario_0048")
        self.assertEqual(
            expanded[0]["variables"],
            {
                "modules.CRE.tests.dscr.decline.default.S1": 0.03,
                "modules.Consumer.pd_increase_factor.S2": 1.25,
                "modules.C&I.interest_rate_stress.S1": 0.005,
                "modules.CRE.tests.ltv.cap_rates.default.S2": 0.08,
                "modules.Consumer.collateral_value_factor.S1": 0.855,
            },
        )
        self.assertEqual(
            expanded[-1]["variables"],
            {
                "modules.CRE.tests.dscr.decline.default.S1": 0.07,
                "modules.Consumer.pd_increase_factor.S2": 1.75,
                "modules.C&I.interest_rate_stress.S1": 0.015,
                "modules.CRE.tests.ltv.cap_rates.default.S2": 0.09,
                "modules.Consumer.collateral_value_factor.S1": 0.945,
            },
        )

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

            self.assertEqual(result["metadata"]["generated_scenario_count"], 48)
            self.assertEqual(result["metadata"]["engine_version"], "0.2.0")
            self.assertTrue(result["metadata"]["input_files"])
            self.assertEqual(len(result["metadata"]["runs"]), 49)
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
            self.assertEqual(len(summary), 147)
            self.assertEqual(len(variables), 240)
            self.assertEqual(
                set(variables["path"]),
                {
                    "modules.CRE.tests.dscr.decline.default.S1",
                    "modules.Consumer.pd_increase_factor.S2",
                    "modules.C&I.interest_rate_stress.S1",
                    "modules.CRE.tests.ltv.cap_rates.default.S2",
                    "modules.Consumer.collateral_value_factor.S1",
                },
            )
            self.assertIn("delta_cecl_ratio", summary.columns)
            self.assertTrue(summary["run_id"].eq("base").any())

    def test_batch_max_scenario_guardrail(self):
        scenario, _ = load_scenario([SCENARIO, BATCH])
        with patch("stress_engine.batch.itertools.product", side_effect=AssertionError("product materialized")):
            with self.assertRaisesRegex(ValueError, "max_scenarios=2"):
                expand_batch_scenarios(scenario, max_scenarios=2)

    def test_generated_range_is_capped_during_value_generation(self):
        scenario = {
            "scenario_id": "huge_range",
            "value": 0,
            "scenario_batch": {
                "mode": "grid",
                "variables": [
                    {
                        "path": "value",
                        "range": {"start": 0, "stop": 1_000_000_000_000, "step": 1},
                    }
                ],
            },
        }
        with patch("stress_engine.batch._clean_numeric", side_effect=lambda value, spec: value) as clean:
            with self.assertRaisesRegex(ValueError, "max_scenarios=2"):
                expand_batch_scenarios(scenario, max_scenarios=2)
        self.assertEqual(clean.call_count, 3)

    def test_generated_values_reject_nonfinite_bounds_before_iteration(self):
        for generator in (
            {"range": {"start": "-Infinity", "stop": 10, "step": 1}},
            {"linspace": {"start": 0, "stop": "Infinity", "count": 2}},
        ):
            with self.subTest(generator=generator):
                scenario = {
                    "scenario_id": "nonfinite",
                    "value": 0,
                    "scenario_batch": {
                        "mode": "grid",
                        "variables": [{"path": "value", **generator}],
                    },
                }
                with self.assertRaisesRegex(ValueError, "finite"):
                    expand_batch_scenarios(scenario, max_scenarios=2)

    def test_linspace_count_is_capped_before_materialization(self):
        scenario = {
            "scenario_id": "huge_linspace",
            "value": 0,
            "scenario_batch": {
                "mode": "grid",
                "variables": [
                    {
                        "path": "value",
                        "linspace": {"start": 0, "stop": 1, "count": 1_000_000_000},
                    }
                ],
            },
        }
        with self.assertRaisesRegex(ValueError, "max_scenarios=2"):
            expand_batch_scenarios(scenario, max_scenarios=2)

    def test_every_iterable_value_source_is_bounded(self):
        for source in ("values", "multipliers", "deltas"):
            with self.subTest(source=source):
                scenario = {
                    "scenario_id": "bounded_iterable",
                    "value": 1,
                    "scenario_batch": {
                        "mode": "grid",
                        "variables": [
                            {"path": "value", source: _ExplodesPastBound()}
                        ],
                    },
                }
                with self.assertRaisesRegex(ValueError, "max_scenarios=2"):
                    expand_batch_scenarios(scenario, max_scenarios=2)

    def test_grid_limit_stops_before_unnecessary_variables(self):
        scenario = {
            "scenario_id": "incremental_grid_limit",
            "first": 0,
            "second": 0,
            "third": 0,
            "scenario_batch": {
                "mode": "grid",
                "variables": [
                    {"path": "first", "values": [1, 2]},
                    {"path": "second", "values": [3, 4]},
                    {"path": "third", "values": _MustNotIterate()},
                ],
            },
        }
        with self.assertRaisesRegex(ValueError, "max_scenarios=2"):
            expand_batch_scenarios(scenario, max_scenarios=2)

    def test_positive_integer_limits_are_exact_and_linspace_rejects_coercion(self):
        exact = 9_007_199_254_740_993
        self.assertEqual(_max_scenario_limit({}, exact), exact)
        self.assertEqual(_max_scenario_limit({}, 10**400), 10**400)

        for count in (True, 2.9, "Infinity", "1e100000"):
            with self.subTest(count=count):
                scenario = {
                    "scenario_id": "invalid_count",
                    "value": 0,
                    "scenario_batch": {
                        "mode": "grid",
                        "variables": [
                            {
                                "path": "value",
                                "linspace": {"start": 0, "stop": 1, "count": count},
                            }
                        ],
                    },
                }
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    expand_batch_scenarios(scenario, max_scenarios=3)

    def test_extreme_range_endpoint_and_linspace_remain_finite(self):
        largest = float.fromhex("0x1.fffffffffffffp+1023")
        range_scenario = {
            "scenario_id": "largest_range",
            "value": 0,
            "scenario_batch": {
                "mode": "grid",
                "variables": [
                    {
                        "path": "value",
                        "range": {
                            "start": largest,
                            "stop": largest,
                            "step": largest,
                        },
                    }
                ],
            },
        }
        expanded = expand_batch_scenarios(range_scenario, max_scenarios=3)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(float(expanded[0]["variables"]["value"]), largest)

        linspace_scenario = {
            "scenario_id": "wide_linspace",
            "value": 0,
            "scenario_batch": {
                "mode": "grid",
                "variables": [
                    {
                        "path": "value",
                        "linspace": {
                            "start": -1e308,
                            "stop": 1e308,
                            "count": 3,
                        },
                    }
                ],
            },
        }
        values = [
            record["variables"]["value"]
            for record in expand_batch_scenarios(
                linspace_scenario, max_scenarios=3
            )
        ]
        self.assertTrue(all(np.isfinite(float(value)) for value in values))
        self.assertEqual(values[1], 0)

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
            with self.assertRaisesRegex(ValueError, "max_scenarios=2"):
                expand_batch_scenarios(scenario, max_scenarios=2)

    def test_generated_child_is_revalidated_after_enabling_a_module(self):
        scenario = {
            "scenario_id": "enable_module",
            "inputs": {"identity": {}},
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "balance",
            },
            "tags": {},
            "modules": {
                "CRE": {"enabled": False},
                "Consumer": {"enabled": False},
            },
            "module_order": ["CRE"],
            "scenario_batch": {
                "mode": "grid",
                "variables": [
                    {
                        "path": "modules.Consumer.enabled",
                        "values": [True],
                    }
                ],
            },
        }
        expanded = expand_batch_scenarios(scenario)

        with self.assertRaisesRegex(ValueError, "missing: Consumer"):
            StressEngine(expanded[0]["scenario"], ROOT)

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

    def test_unmarked_child_manifest_cannot_authorize_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / "scenarios" / "foreign"
            child.mkdir(parents=True)
            foreign_file = child / "keep.txt"
            foreign_file.write_text("user-owned", encoding="utf-8")
            manifest = child / "output_manifest.json"
            manifest.write_text(
                json.dumps({"files": ["keep.txt", "output_manifest.json"]}),
                encoding="utf-8",
            )

            _remove_engine_child_outputs(child)

            self.assertEqual(foreign_file.read_text(encoding="utf-8"), "user-owned")
            self.assertTrue(manifest.is_file())

if __name__ == "__main__":
    unittest.main()
