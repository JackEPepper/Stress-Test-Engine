import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stress_engine.run import run_engine


class RunEngineTests(unittest.TestCase):
    def test_example_run_writes_first_milestone_outputs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "reports"
            result = run_engine(
                root / "config" / "base_config.json",
                root / "scenarios" / "example_severe_case.json",
                root / "data" / "raw",
                output_dir,
                root / "external_sources",
            )
            run_dir = result["output_dir"]
            self.assertTrue((run_dir / "loan_level_results.csv").exists())
            self.assertTrue((run_dir / "sector_summary.csv").exists())
            self.assertTrue((run_dir / "tag_population_tie_out.csv").exists())
            self.assertEqual(result["metadata"]["out_of_scope_loan_count"], 2)
            self.assertEqual(result["metadata"]["external_source_tie_out_status"], "pass")
            loan_results = result["results"]
            self.assertIn("selected_stress_module", loan_results.columns)
            self.assertIn("selected_formula", loan_results.columns)
            self.assertIn("scope_status", loan_results.columns)


if __name__ == "__main__":
    unittest.main()
