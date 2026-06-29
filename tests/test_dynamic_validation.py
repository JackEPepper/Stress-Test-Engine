import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stress_engine.validation.rule_sets import validate_dynamic_rules


class DynamicValidationTests(unittest.TestCase):
    def test_formula_specific_missing_field_marks_out_of_scope(self):
        frame = pd.DataFrame(
            [
                {
                    "loan_id": "L001",
                    "balance": 100.0,
                    "maturity_date": pd.Timestamp("2026-12-31"),
                    "selected_stress_module": "ci",
                    "selected_formula": "formula_2",
                    "maturity_formula": "longer_term",
                    "revenue": 1000.0,
                    "ebitda": 100.0,
                }
            ]
        )
        config = {
            "dynamic_required_fields": {
                "ci": {
                    "formula_2": {
                        "longer_term": ["loan_id", "balance", "maturity_date", "revenue", "ebitda", "debt_service_ci"]
                    }
                }
            },
            "positive_fields": {"ci": ["balance"]},
        }
        result = validate_dynamic_rules(frame, config)
        self.assertEqual(result.iloc[0]["scope_status"], "out_of_scope")
        self.assertIn("missing_required_field:debt_service_ci", result.iloc[0]["out_of_scope_reasons"])


if __name__ == "__main__":
    unittest.main()
