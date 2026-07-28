from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stress_engine.borrower import build_borrowers
from stress_engine.comparison import _cecl_impact_rows
from stress_engine.config import load_scenario
from stress_engine.engine import OUTPUT_MANIFEST_KIND, StressEngine
from stress_engine.io import write_csv
from stress_engine.modules.ci import run_ci
from stress_engine.modules.consumer import run_consumer
from stress_engine.reporting import build_consumer_summary
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
            "cutoffs": {
                "fccr": {"special_mention": 1.15, "substandard": 1.0}
            },
            "borrower": {
                "borrower_id_field": "borrower_id",
                "portfolio_field": "portfolio",
                "risk_rating_field": "risk_rating",
            },
            "modules": {
                "C&I": {
                    "sector_field": "ci_sector",
                    "ebitda_reduction": {"default": {"6": {"S1": 0.1}}},
                    "interest_rate_stress": {"S1": 0.01},
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
                    "risk_rating": 6,
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

    def test_ci_brg_reductions_cover_one_through_eight_and_cap_higher_grades(self):
        reductions = {
            str(brg): {"S1": brg / 100.0}
            for brg in range(1, 9)
        }
        scenario = {
            "stress_levels": ["S1"],
            "cutoffs": {
                "fccr": {"special_mention": 1.15, "substandard": 1.0}
            },
            "borrower": {
                "borrower_id_field": "borrower_id",
                "portfolio_field": "portfolio",
                "risk_rating_field": "brg",
            },
            "modules": {
                "C&I": {
                    "sector_field": "ci_sector",
                    "ebitda_reduction": {"default": reductions},
                    "interest_rate_stress": {"S1": 0.0},
                    "sectors": {
                        "Test Sector": {
                            "principal_field": "principal_repayments_paid",
                        }
                    },
                }
            },
        }
        rows = []
        for brg in [*range(1, 9), 12]:
            base_bucket = (
                "Pass"
                if brg < 7
                else "Special Mention"
                if brg == 7
                else "Substandard"
            )
            rows.append(
                {
                    "borrower_id": f"CI-{brg}",
                    "portfolio": "C&I",
                    "primary_module": "C&I",
                    "module_applied": "",
                    "brg": brg,
                    "base_bucket": base_bucket,
                    "stressed_bucket_S1": base_bucket,
                    "out_of_scope_S1": False,
                    "ci_sector": "Test Sector",
                    "ebitda": 100.0,
                    "cash_taxes": 0.0,
                    "cash_distribution": 0.0,
                    "cash_management_fees": 0.0,
                    "unfinanced_capex": 0.0,
                    "global_total_outstanding": 0.0,
                    "cash_paid_for_interest": 10.0,
                    "principal_repayments_paid": 0.0,
                }
            )

        stressed, out_of_scope = run_ci(pd.DataFrame(rows), scenario, [])
        stressed = stressed.set_index("borrower_id")

        self.assertTrue(out_of_scope.empty)
        for brg in range(1, 9):
            self.assertAlmostEqual(
                float(stressed.at[f"CI-{brg}", "ci_available_cash_flow_S1"]),
                100.0 - brg,
            )
        self.assertEqual(
            float(stressed.at["CI-12", "ci_available_cash_flow_S1"]),
            float(stressed.at["CI-8", "ci_available_cash_flow_S1"]),
        )
        for borrower_id in ("CI-8", "CI-12"):
            self.assertFalse(pd.isna(stressed.at[borrower_id, "ci_fccr_S1"]))
            self.assertEqual(
                stressed.at[borrower_id, "stressed_bucket_S1"],
                "Substandard",
            )

    def test_ci_invalid_brgs_are_data_errors_not_assumption_errors(self):
        scenario = {
            "stress_levels": ["S1"],
            "cutoffs": {
                "fccr": {"special_mention": 1.15, "substandard": 1.0}
            },
            "borrower": {
                "borrower_id_field": "borrower_id",
                "portfolio_field": "portfolio",
                "risk_rating_field": "brg",
            },
            "modules": {
                "C&I": {
                    "sector_field": "ci_sector",
                    "ebitda_reduction": {
                        "default": {
                            "1": {"S1": 0.1},
                            "8": {"S1": 0.2},
                            # A valid BRG without its own key must not fall
                            # back to a bucket-named assumption.
                            "Pass": {"S1": 0.3},
                        }
                    },
                    "interest_rate_stress": {"S1": 0.0},
                    "sectors": {
                        "Test Sector": {
                            "principal_field": "principal_repayments_paid",
                        }
                    },
                }
            },
        }
        base_row = {
            "portfolio": "C&I",
            "primary_module": "C&I",
            "module_applied": "",
            "stressed_bucket_S1": "Pass",
            "out_of_scope_S1": False,
            "ci_sector": "Test Sector",
            "ebitda": 100.0,
            "cash_taxes": 0.0,
            "cash_distribution": 0.0,
            "cash_management_fees": 0.0,
            "unfinanced_capex": 0.0,
            "global_total_outstanding": 0.0,
            "cash_paid_for_interest": 10.0,
            "principal_repayments_paid": 0.0,
        }
        invalid_brgs = [
            ("CI-BRG-MISSING", np.nan, "Unknown"),
            ("CI-BRG-ZERO", 0, "Pass"),
            ("CI-BRG-FRACTIONAL", 7.5, "Substandard"),
            ("CI-BRG-INFINITE", np.inf, "Substandard"),
        ]
        rows = [
            {
                **base_row,
                "borrower_id": borrower_id,
                "brg": brg,
                "base_bucket": base_bucket,
            }
            for borrower_id, brg, base_bucket in invalid_brgs
        ]
        rows.append(
            {
                **base_row,
                "borrower_id": "CI-BRG-2-NOT-CONFIGURED",
                "brg": 2,
                "base_bucket": "Pass",
            }
        )
        exceptions = []

        stressed, out_of_scope = run_ci(pd.DataFrame(rows), scenario, exceptions)

        self.assertTrue(stressed["out_of_scope_S1"].astype(bool).all())
        invalid_ids = {borrower_id for borrower_id, _, _ in invalid_brgs}
        invalid_detail = out_of_scope[
            out_of_scope["borrower_id"].isin(invalid_ids)
        ]
        self.assertEqual(set(invalid_detail["field"]), {"brg"})
        self.assertEqual(set(invalid_detail["reason"]), {"missing_or_invalid_brg"})
        invalid_exceptions = [
            row for row in exceptions if row["code"] == "CI_BRG_INVALID"
        ]
        self.assertEqual(
            {row["borrower_id"] for row in invalid_exceptions},
            invalid_ids,
        )

        missing_assumption = out_of_scope[
            out_of_scope["borrower_id"] == "CI-BRG-2-NOT-CONFIGURED"
        ]
        self.assertEqual(set(missing_assumption["field"]), {"ebitda_reduction"})
        self.assertEqual(
            set(missing_assumption["reason"]),
            {"missing_or_invalid_scenario_assumption"},
        )
        assumption_errors = [
            row
            for row in exceptions
            if row["code"] == "CI_SCENARIO_ASSUMPTION_INVALID"
        ]
        self.assertEqual(
            {row["borrower_id"] for row in assumption_errors},
            {"CI-BRG-2-NOT-CONFIGURED"},
        )

    def test_abl_calculated_cash_interest_and_fallbacks(self):
        scenario = {
            "stress_levels": ["S1"],
            "cutoffs": {
                "fccr": {"special_mention": 1.15, "substandard": 1.0}
            },
            "borrower": {
                "borrower_id_field": "borrower_id",
                "portfolio_field": "portfolio",
                "risk_rating_field": "risk_rating",
            },
            "modules": {
                "C&I": {
                    "sector_field": "ci_sector",
                    "ebitda_reduction": {"default": {"6": {"S1": 0.0}}},
                    "interest_rate_stress": {"S1": 0.0},
                    "sectors": {
                        "Asset-Based Lending": {
                            "principal_field": "required_principal_paid_period",
                            "include_non_discretionary_dividends": True,
                            "use_calculated_cash_paid_for_interest": True,
                        },
                        "Middle Market": {
                            "principal_field": "required_principal_paid_period",
                            "include_non_discretionary_dividends": True,
                        },
                    },
                }
            },
        }
        base_row = {
            "portfolio": "C&I",
            "primary_module": "C&I",
            "module_applied": "",
            "risk_rating": 6,
            "base_bucket": "Pass",
            "stressed_bucket_S1": "Pass",
            "out_of_scope_S1": False,
            "ci_sector": "Asset-Based Lending",
            "ebitda": 100.0,
            "cash_taxes": 0.0,
            "cash_distribution": 0.0,
            "cash_dividends": 0.0,
            "discretionary_cash_dividends_distribution": 0.0,
            "cash_management_fees": 0.0,
            "unfinanced_capex": 0.0,
            "global_total_outstanding": 0.0,
            "required_principal_paid_period": 10.0,
        }
        results = pd.DataFrame(
            [
                {
                    **base_row,
                    "borrower_id": "ABL-CALCULATED",
                    "interest_expense": 30.0,
                    "non_cash_interest_expense": 5.0,
                    "cash_paid_for_interest": 99.0,
                },
                {
                    **base_row,
                    "borrower_id": "ABL-MISSING",
                    "interest_expense": np.nan,
                    "non_cash_interest_expense": np.nan,
                    "cash_paid_for_interest": 20.0,
                },
                {
                    **base_row,
                    "borrower_id": "ABL-ZERO",
                    "interest_expense": 5.0,
                    "non_cash_interest_expense": 5.0,
                    "cash_paid_for_interest": 20.0,
                },
                {
                    **base_row,
                    "borrower_id": "MM-ORIGINAL",
                    "ci_sector": "Middle Market",
                    "interest_expense": 30.0,
                    "non_cash_interest_expense": 5.0,
                    "cash_paid_for_interest": 20.0,
                },
                {
                    **base_row,
                    "borrower_id": "ABL-ONE-MISSING",
                    "interest_expense": 30.0,
                    "non_cash_interest_expense": np.nan,
                    "cash_paid_for_interest": 99.0,
                },
            ]
        )
        exceptions = []

        stressed, out_of_scope = run_ci(results, scenario, exceptions)
        stressed = stressed.set_index("borrower_id")

        self.assertTrue(out_of_scope.empty)
        self.assertEqual(
            float(stressed.at["ABL-CALCULATED", "calculated_cash_paid_for_interest"]),
            25.0,
        )
        self.assertEqual(
            stressed.at["ABL-CALCULATED", "calculated_cash_paid_for_interest_source"],
            "interest_expense_less_non_cash_interest_expense",
        )
        self.assertEqual(float(stressed.at["ABL-CALCULATED", "ci_debt_service_S1"]), 35.0)

        for borrower_id, reason in (
            ("ABL-MISSING", "alternative_inputs_missing"),
            ("ABL-ZERO", "calculated_value_zero"),
        ):
            self.assertEqual(
                float(stressed.at[borrower_id, "calculated_cash_paid_for_interest"]),
                20.0,
            )
            self.assertEqual(
                stressed.at[borrower_id, "calculated_cash_paid_for_interest_source"],
                "cash_paid_for_interest",
            )
            self.assertEqual(
                stressed.at[borrower_id, "calculated_cash_paid_for_interest_fallback_reason"],
                reason,
            )
            self.assertEqual(float(stressed.at[borrower_id, "ci_debt_service_S1"]), 30.0)

        self.assertEqual(
            float(stressed.at["MM-ORIGINAL", "calculated_cash_paid_for_interest"]),
            20.0,
        )
        self.assertEqual(
            stressed.at["MM-ORIGINAL", "calculated_cash_paid_for_interest_source"],
            "cash_paid_for_interest",
        )
        self.assertEqual(float(stressed.at["MM-ORIGINAL", "ci_debt_service_S1"]), 30.0)

        self.assertEqual(
            float(stressed.at["ABL-ONE-MISSING", "calculated_cash_paid_for_interest"]),
            30.0,
        )
        self.assertEqual(float(stressed.at["ABL-ONE-MISSING", "ci_debt_service_S1"]), 40.0)
        fallback_rows = [
            row for row in exceptions if row["code"] == "CI_CALCULATED_CASH_INTEREST_FALLBACK"
        ]
        self.assertEqual({row["borrower_id"] for row in fallback_rows}, {"ABL-MISSING", "ABL-ZERO"})
        substitution_rows = [
            row for row in exceptions if row["code"] == "CI_MISSING_FIELD_ZERO_SUBSTITUTION"
        ]
        self.assertEqual(
            {(row["borrower_id"], row["field"]) for row in substitution_rows},
            {("ABL-ONE-MISSING", "non_cash_interest_expense")},
        )

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

    def test_consumer_baseline_el_uses_liquidation_adjustments(self):
        scenario = {
            "stress_levels": ["S1"],
            "borrower": {
                "borrower_id_field": "borrower_id",
                "portfolio_field": "portfolio",
                "balance_field": "outstanding_balance",
            },
            "cecl": {"reserve_field": "cecl_reserve"},
            "modules": {
                "Consumer": {
                    "pd_lookup_source": "fico_pd_lookup",
                    "fico_field": "fico_score",
                    "appraisal_field": "current_appraised_value",
                    "pd_increase_factor": {"S1": 1.25},
                    "collateral_value_factor": {"S1": 0.9},
                    "rushed_sale_discount": 0.05,
                    "closing_costs": 0.02,
                    "pd_cap": 1.0,
                }
            },
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "CON-BASE",
                    "portfolio": "Consumer",
                    "primary_module": "Consumer",
                    "module_applied": "",
                    "out_of_scope_S1": False,
                    "fico_score": 700,
                    "current_appraised_value": 100.0,
                    "outstanding_balance": 120.0,
                    "cecl_reserve": 5.0,
                }
            ]
        )
        inputs = {
            "fico_pd_lookup": SimpleNamespace(
                frame=pd.DataFrame(
                    {
                        "min_score": [600],
                        "max_score": [800],
                        "pd": [0.02],
                    }
                )
            )
        }

        stressed, out_of_scope = run_consumer(results, scenario, inputs, [])
        summary = build_consumer_summary(stressed, scenario).set_index(
            "stress_level"
        )

        self.assertTrue(out_of_scope.empty)
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_collateral_value_unstressed"]),
            93.1,
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_lgd_unstressed"]),
            26.9,
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_el_unstressed"]),
            0.538,
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_qualitative_reserve"]),
            4.462,
        )
        self.assertAlmostEqual(
            float(summary.at["Base", "expected_loss"])
            + float(summary.at["Base", "qualitative_reserve"]),
            float(summary.at["Base", "proforma_cecl_reserve"]),
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_stressed_collateral_value_S1"]),
            83.79,
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_el_S1"]),
            0.90525,
        )
        self.assertAlmostEqual(
            float(summary.at["S1", "proforma_cecl_reserve"]),
            5.36725,
        )

        scenario["modules"]["Consumer"]["closing_costs"] = 1.1
        exceptions = []
        invalid, invalid_out_of_scope = run_consumer(
            results,
            scenario,
            inputs,
            exceptions,
        )
        self.assertTrue(pd.isna(invalid.at[0, "consumer_el_unstressed"]))
        self.assertTrue(pd.isna(invalid.at[0, "consumer_qualitative_reserve"]))
        self.assertTrue(bool(invalid.at[0, "out_of_scope_S1"]))
        self.assertEqual(
            set(invalid_out_of_scope["reason"]),
            {"missing_or_invalid_scenario_assumption"},
        )
        self.assertIn(
            ("CONSUMER_SCENARIO_ASSUMPTION_INVALID", "closing_costs"),
            {(row["code"], row["field"]) for row in exceptions},
        )

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
            manifest = json.loads((Path(tmp) / "output_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], OUTPUT_MANIFEST_KIND)

    def test_unmarked_output_manifest_cannot_authorize_deletion(self):
        scenario, base_dir = load_scenario(SCENARIO)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            foreign_file = output_dir / "foreign_report.csv"
            foreign_file.write_text("must remain\n", encoding="utf-8")
            (output_dir / "output_manifest.json").write_text(
                json.dumps(
                    {
                        "engine_version": "legacy",
                        "files": ["foreign_report.csv", "output_manifest.json"],
                    }
                ),
                encoding="utf-8",
            )

            StressEngine(scenario, base_dir).run(
                output_dir=output_dir,
                write_outputs=True,
                run_comparison=False,
            )

            self.assertEqual(foreign_file.read_text(encoding="utf-8"), "must remain\n")
            manifest = json.loads(
                (output_dir / "output_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["kind"], OUTPUT_MANIFEST_KIND)

    def test_csv_export_neutralizes_formulas_without_mutating_frame(self):
        frame = pd.DataFrame(
            {
                "=formula_header": [
                    "=1+1",
                    "  +SUM(1,1)",
                    "\t-2+3",
                    " @SUM(1,1)",
                    "-42",
                    "safe",
                ],
                "numeric": [-6, -5, -4, -3, -2, -1],
            }
        )
        original = frame.copy(deep=True)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "safe.csv"
            write_csv(frame, output)
            exported = pd.read_csv(output, keep_default_na=False)

        self.assertIn("'=formula_header", exported.columns)
        self.assertEqual(
            exported["'=formula_header"].tolist(),
            [
                "'=1+1",
                "'  +SUM(1,1)",
                "'\t-2+3",
                "' @SUM(1,1)",
                "'-42",
                "safe",
            ],
        )
        self.assertEqual(exported["numeric"].tolist(), [-6, -5, -4, -3, -2, -1])
        pd.testing.assert_frame_equal(frame, original)

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
