import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stress_engine.validation.schemas import validate_required_columns, validate_unique_loan_ids


class ValidationTests(unittest.TestCase):
    def test_required_columns_are_enforced(self):
        tables = {"loan_identity": pd.DataFrame({"loan_id": ["L001"]})}
        issues = validate_required_columns(tables, {"loan_identity": ["loan_id", "balance"]})
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "missing_required_column")

    def test_duplicate_loan_ids_are_rejected(self):
        frame = pd.DataFrame({"loan_id": ["L001", "L001", "L002"]})
        issues = validate_unique_loan_ids(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "duplicate_loan_id")


if __name__ == "__main__":
    unittest.main()
