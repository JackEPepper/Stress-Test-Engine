from __future__ import annotations

import unittest

import pandas as pd

from stress_engine.reporting import build_bucket_summary


class PandasCompatibilityTest(unittest.TestCase):
    def test_bucket_summary_emits_only_observed_categorical_populations(self):
        results = pd.DataFrame(
            {
                "portfolio": pd.Categorical(
                    ["CRE"], categories=["CRE", "Unused Portfolio"]
                ),
                "base_bucket": pd.Categorical(
                    ["Pass"], categories=["Pass", "Substandard"]
                ),
                "balance": [100.0],
            }
        )
        scenario = {
            "borrower": {
                "borrower_id_field": "borrower_id",
                "portfolio_field": "portfolio",
                "balance_field": "balance",
            },
            "stress_levels": [],
        }

        summary = build_bucket_summary(results, scenario)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["portfolio"], "CRE")
        self.assertEqual(summary.iloc[0]["bucket"], "Pass")
        self.assertEqual(float(summary.iloc[0]["balance"]), 100.0)
        self.assertEqual(int(summary.iloc[0]["borrower_count"]), 1)


if __name__ == "__main__":
    unittest.main()
