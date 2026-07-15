from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stress_engine.borrower import build_borrowers
from stress_engine.comparison import _cecl_impact_rows
from stress_engine.config import load_scenario
from stress_engine.engine import StressEngine
from stress_engine.modules.ci import run_ci
from stress_engine.modules.consumer import run_consumer
from stress_engine.tagging import evaluate_conditions
from stress_engine.utils import to_number


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"


class HardeningRegressionTest(unittest.TestCase):
    def test_largest_loan_supplies_rating_maturity_and_tag_fields(self):
        scenario, _ = load_scenario(SCENARIO)
        identity = pd.DataFrame(
            [
                {
                    "loan_id": "small",
                    "borrower_id": "X",
                    "subsector": "Middle Market",
                    "tag_hint": "",
                    "status": "Active",
                    "risk_rating": 8,
                    "maturity_date": pd.Timestamp("2026-01-01"),
                    "outstanding_balance": 100.0,
                    "cecl_reserve": 5.0,
                    "_source_row": 1,
                },
                {
                    "loan_id": "large",
                    "borrower_id": "X",
                    "subsector": "Retail",
                    "tag_hint": "",
                    "status": "Active",
                    "risk_rating": 6,
                    "maturity_date": pd.Timestamp("2029-01-01"),
                    "outstanding_balance": 900.0,
                    "cecl_reserve": 9.0,
                    "_source_row": 2,
                },
            ]
        )
        exceptions = []
        borrower = build_borrowers(identity, scenario, exceptions).iloc[0]
        self.assertEqual(borrower["subsector"], "Retail")
        self.assertEqual(float(borrower["risk_rating"]), 6.0)
        self.assertEqual(borrower["maturity_date"], pd.Timestamp("2029-01-01"))
        self.assertEqual(float(borrower["outstanding_balance"]), 1000.0)
        self.assertIn("BORROWER_LOAN_ATTRIBUTE_CONFLICT", {row["code"] for row in exceptions})

    def test_missing_ratings_remain_visible_as_unknown(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["borrower"]["risk_rating_field"] = "missing_rating"
        result = StressEngine(scenario, base_dir).run(write_outputs=False, run_comparison=False)

        base_migration = result["reports"]["migration_summary"]
        base_migration = base_migration[base_migration["stress_level"] == "Base"]
        self.assertEqual(float(base_migration["balance"].sum()), 6100000.0)
        positive_base = base_migration[pd.to_numeric(base_migration["balance"], errors="coerce") > 0]
        self.assertEqual(set(positive_base["bucket"]), {"Unknown"})
        aggregate_base = result["reports"]["cecl_summary"]
        aggregate_base = aggregate_base[
            (aggregate_base["portfolio"] == "Aggregate")
            & (aggregate_base["stress_level"] == "Base")
            & (aggregate_base["bucket"] == "Total")
        ].iloc[0]
        self.assertEqual(float(aggregate_base["balance"]), 6400000.0)
        self.assertEqual(float(aggregate_base["proforma_cecl_reserve"]), 76000.0)
        self.assertIn("RISK_RATING_MISSING", set(result["reports"]["exception_log"]["code"]))

    def test_out_of_scope_consumer_stress_cecl_is_unavailable_not_zero(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["modules"]["Consumer"]["fico_field"] = "does_not_exist"
        result = StressEngine(scenario, base_dir).run(write_outputs=False, run_comparison=False)
        cecl = result["reports"]["cecl_summary"]
        consumer = cecl[(cecl["portfolio"] == "Consumer") & (cecl["bucket"] == "Total")].set_index("stress_level")
        self.assertEqual(float(consumer.at["Base", "proforma_cecl_reserve"]), 4000.0)
        self.assertTrue(pd.isna(consumer.at["S1", "proforma_cecl_reserve"]))
        self.assertEqual(consumer.at["S1", "cecl_reserve_status"], "unavailable")

    def test_unconfigured_ci_sector_uses_logged_canonical_fallback(self):
        scenario = {
            "stress_levels": ["S1"],
            "borrower": {
                "borrower_id_field": "borrower_id",
                "portfolio_field": "portfolio",
            },
            "modules": {
                "C&I": {
                    "sector_field": "ci_sector",
                    "ebitda_reduction": {"default": {"Pass": {"S1": 0.1}}},
                    "interest_rate_stress": {"S1": 0.01},
                    "cutoffs": {"special_mention": 1.15, "substandard": 1.0},
                    "sectors": {},
                }
            },
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "CI-1",
                    "portfolio": "C&I",
                    "primary_module": "C&I",
                    "module_applied": "",
                    "base_bucket": "Pass",
                    "stressed_bucket_S1": "Pass",
                    "out_of_scope_S1": False,
                    "ci_sector": "Unconfigured Sector",
                    "ebitda": 100.0,
                    "cash_taxes": 10.0,
                    "cash_distribution": 5.0,
                    "cash_management_fees": 0.0,
                    "unfinanced_capex": 5.0,
                    "global_total_outstanding": 500.0,
                    "cash_paid_for_interest": 10.0,
                    "principal_repayments_paid": 20.0,
                }
            ]
        )
        exceptions = []

        stressed, out_of_scope = run_ci(results, scenario, exceptions)

        self.assertFalse(bool(stressed.at[0, "out_of_scope_S1"]))
        self.assertAlmostEqual(float(stressed.at[0, "ci_debt_service_S1"]), 35.0)
        self.assertTrue(out_of_scope.empty)
        self.assertIn("CI_SECTOR_DEFAULT_USED", {row["code"] for row in exceptions})

    def test_malformed_consumer_pd_lookup_is_logged_and_out_of_scope(self):
        scenario = {
            "stress_levels": ["S1"],
            "borrower": {
                "borrower_id_field": "borrower_id",
                "portfolio_field": "portfolio",
                "balance_field": "outstanding_balance",
            },
            "modules": {
                "Consumer": {
                    "pd_lookup_source": "fico_pd_lookup",
                    "fico_field": "fico_score",
                    "appraisal_field": "current_appraised_value",
                    "pd_increase_factor": {"S1": 1.25},
                    "collateral_value_factor": {"S1": 0.9},
                    "rushed_sale_discount": 0.05,
                    "closing_costs": 0.02,
                }
            },
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "CON-1",
                    "portfolio": "Consumer",
                    "primary_module": "Consumer",
                    "module_applied": "",
                    "out_of_scope_S1": False,
                    "fico_score": 700,
                    "current_appraised_value": 100.0,
                    "outstanding_balance": 120.0,
                }
            ]
        )
        inputs = {"fico_pd_lookup": SimpleNamespace(frame=pd.DataFrame({"score": [700], "pd": [0.02]}))}
        exceptions = []

        stressed, out_of_scope = run_consumer(results, scenario, inputs, exceptions)

        self.assertTrue(bool(stressed.at[0, "out_of_scope_S1"]))
        self.assertEqual(set(out_of_scope["reason"]), {"missing_pd_lookup"})
        self.assertIn("CONSUMER_PD_LOOKUP_COLUMNS_INVALID", {row["code"] for row in exceptions})

    def test_tieout_difference_is_logged_without_stopping_run(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["tags"]["CRE_Subsector_Retail"]["tie_out"]["expected"] = 1.0
        result = StressEngine(scenario, base_dir).run(write_outputs=False, run_comparison=False)
        self.assertIn("TAG_TIEOUT_DIFFERENCE", set(result["reports"]["exception_log"]["code"]))
        tieout = result["reports"]["tag_summary"]
        tieout = tieout[
            (tieout["tag"] == "CRE_Subsector_Retail") & tieout["tie_out_name"].notna()
        ].iloc[0]
        self.assertFalse(bool(tieout["passed"]))

    def test_reused_output_directory_removes_stale_comparison(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["modules"]["Consumer"]["pd_increase_factor"]["S1"] = 2.0
        scenario["comparison"] = {"previous_scenarios": [str(SCENARIO)]}
        with tempfile.TemporaryDirectory() as tmp:
            StressEngine(scenario, base_dir).run(output_dir=tmp, write_outputs=True, run_comparison=True)
            comparison = Path(tmp) / "scenario_diff.csv"
            self.assertTrue(comparison.exists())
            scenario["comparison"] = {}
            StressEngine(scenario, base_dir).run(output_dir=tmp, write_outputs=True, run_comparison=False)
            self.assertFalse(comparison.exists())

    def test_comparison_reports_available_to_unavailable_transition(self):
        previous = pd.DataFrame(
            [{"portfolio": "CRE", "stress_level": "S1", "bucket": "Total", "proforma_cecl_reserve": 100.0,
              "proforma_cecl_ratio": 0.01, "cecl_reserve_status": "available"}]
        )
        changed = pd.DataFrame(
            [{"portfolio": "CRE", "stress_level": "S1", "bucket": "Total", "proforma_cecl_reserve": np.nan,
              "proforma_cecl_ratio": np.nan, "cecl_reserve_status": "unavailable"}]
        )
        rows = _cecl_impact_rows("prior", "scenario_variable", "x", 1, 2, previous, changed)
        self.assertIn("cecl_reserve_status", {row["metric"] for row in rows})
        self.assertIn("proforma_cecl_reserve", {row["metric"] for row in rows})

    def test_literal_contains_and_percent_parsing(self):
        frame = pd.DataFrame({"value": ["abc", "a.c"]})
        mask = evaluate_conditions(frame, [{"field": "value", "op": "contains", "value": "."}])
        self.assertEqual(mask.tolist(), [False, True])
        self.assertEqual(to_number("2%"), 0.02)


if __name__ == "__main__":
    unittest.main()
