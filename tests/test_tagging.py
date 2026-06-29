import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stress_engine.tagging.loan_tags import add_tags, tags_as_list
from stress_engine.tagging.module_selection import select_modules_and_formulas


class TaggingTests(unittest.TestCase):
    def test_multiple_tags_can_select_one_module(self):
        frame = pd.DataFrame(
            [
                {
                    "loan_id": "L001",
                    "portfolio": "c&i",
                    "product_type": "revolver",
                    "sector": "wholesale",
                    "cre_sector": "office",
                    "ebitda": 100.0,
                    "fico": float("nan"),
                    "maturity_date": pd.Timestamp("2026-09-30"),
                }
            ]
        )
        tagged = add_tags(frame, pd.Timestamp("2026-06-30"), 365)
        selected = select_modules_and_formulas(tagged, {"primary_module_priority": ["cre", "ci", "consumer"], "ci_formula_rules": []})
        tags = tags_as_list(selected.iloc[0]["tags"])
        self.assertIn("eligible_cre", tags)
        self.assertIn("eligible_ci", tags)
        self.assertEqual(selected.iloc[0]["selected_stress_module"], "cre")
        self.assertEqual(selected.iloc[0]["maturity_formula"], "near_term")


if __name__ == "__main__":
    unittest.main()
