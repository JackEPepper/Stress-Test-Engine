from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stress_engine.comparison import build_comparison_report
from stress_engine.config import load_scenario
from stress_engine.engine import StressEngine


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"


class SampleScenarioRunTest(unittest.TestCase):
    def _run(self, write_outputs: bool = False):
        scenario, base_dir = load_scenario(SCENARIO)
        return StressEngine(scenario, base_dir).run(write_outputs=write_outputs, run_comparison=False)

    def test_sample_run_outputs_expected_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenario, base_dir = load_scenario(SCENARIO)
            result = StressEngine(scenario, base_dir).run(output_dir=tmp, write_outputs=True, run_comparison=False)

            borrowers = result["borrowers"]
            self.assertEqual(len(borrowers), 11)
            b001 = borrowers.loc[borrowers["borrower_id"] == "B001"].iloc[0]
            self.assertEqual(float(b001["outstanding_balance"]), 1250000.0)
            self.assertEqual(int(b001["loan_count"]), 2)
            overlap = borrowers.loc[borrowers["borrower_id"] == "B011"].iloc[0]
            self.assertIn("CI_Model", overlap["model_tags"])
            self.assertNotIn("CRE_Model", overlap["model_tags"])
            self.assertIn("CRE_Subsector_Retail", overlap["all_tags"])
            self.assertIn("CI_Sector_Middle_Market", overlap["all_tags"])

            tag_summary = result["reports"]["tag_summary"]
            tieouts = tag_summary[tag_summary["tie_out_name"].notna()]
            self.assertTrue(tieouts["passed"].astype(bool).all())

            out_scope = result["reports"]["out_of_scope_detail"]
            self.assertEqual(set(out_scope["borrower_id"]), {"B004"})
            self.assertEqual(set(out_scope["field"]), {"dscr"})

            consumer = result["reports"]["consumer_summary"]
            s1 = float(consumer.loc[consumer["stress_level"] == "S1", "expected_loss"].iloc[0])
            s2 = float(consumer.loc[consumer["stress_level"] == "S2", "expected_loss"].iloc[0])
            self.assertGreater(s2, s1)
            self.assertGreater(s1, 0)

            output_files = {path.name for path in Path(tmp).iterdir()}
            self.assertIn("borrower_audit_raw.csv", output_files)
            self.assertIn("stressed_borrower_results.csv", output_files)
            self.assertIn("cecl_summary.csv", output_files)
            self.assertIn("metadata.json", output_files)

    def test_repeated_runs_are_deterministic_for_core_reports(self):
        first = self._run()
        second = self._run()
        for report_name in ["migration_summary", "cecl_summary", "consumer_summary", "out_of_scope_summary"]:
            left = first["reports"][report_name].fillna("").sort_index(axis=1).reset_index(drop=True)
            right = second["reports"][report_name].fillna("").sort_index(axis=1).reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right, check_dtype=False)

    def test_changed_scenario_variable_produces_marginal_report(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["modules"]["CRE"]["tests"]["dscr"]["decline"]["Traditional Office"]["S1"] = 0.20
        result = StressEngine(scenario, base_dir).run(write_outputs=False, run_comparison=False)
        diff = build_comparison_report(
            scenario,
            result["reports"],
            [SCENARIO],
            max_variable_reruns=3,
        )
        scenario_rows = diff[diff["change_kind"] == "scenario_variable"]
        self.assertFalse(scenario_rows.empty)
        self.assertTrue((scenario_rows["marginal_impact"].astype(float) != 0).any())


if __name__ == "__main__":
    unittest.main()
