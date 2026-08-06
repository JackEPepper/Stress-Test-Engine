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
from stress_engine.reporting import (
    build_cecl_summary,
    build_consumer_summary,
    build_reports,
)
from stress_engine.tagging import evaluate_conditions
from stress_engine.utils import compare_values, parse_date_series, to_number


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"


class HardeningRegressionTest(unittest.TestCase):
    def test_all_model_excluded_population_returns_schema_stable_empty_reports(self):
        scenario = {
            "stress_levels": ["S1"],
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "balance",
                "portfolio_field": "model_portfolio",
            },
            "cecl": {
                "portfolio_field": "cecl_portfolio",
                "reserve_field": "cecl_reserve",
            },
            "modules": {},
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "B-ARR",
                    "balance": 100.0,
                    "cecl_reserve": 5.0,
                    "model_portfolio": pd.NA,
                    "cecl_portfolio": pd.NA,
                    "primary_module": pd.NA,
                    "module_applied": "",
                    "base_bucket": "Pass",
                    "stressed_bucket_S1": "Pass",
                    "model_excluded": True,
                }
            ]
        )

        reports = build_reports(
            results,
            results,
            scenario,
            pd.DataFrame(),
            [],
        )

        migration = reports["migration_summary"]
        self.assertTrue(migration.empty)
        self.assertEqual(
            migration.columns.tolist(),
            [
                "portfolio",
                "stress_level",
                "bucket",
                "balance",
                "borrower_count",
                "source",
            ],
        )
        cecl = reports["cecl_summary"]
        self.assertEqual(
            cecl[["portfolio", "stress_level", "bucket"]].to_dict(
                orient="records"
            ),
            [
                {
                    "portfolio": "Aggregate",
                    "stress_level": "Base",
                    "bucket": "Total",
                },
                {
                    "portfolio": "Aggregate",
                    "stress_level": "S1",
                    "bucket": "Total",
                },
            ],
        )
        self.assertTrue(cecl["balance"].eq(0.0).all())
        self.assertTrue(cecl["proforma_cecl_reserve"].eq(0.0).all())

    def test_largest_loan_supplies_rating_maturity_and_tag_fields(self):
        scenario, _ = load_scenario(SCENARIO)
        scenario["borrower"]["sum_fields"].append("secondary_exposure")
        scenario["tags"]["Summed_Exposure_Control"] = {
            "model_eligible": False,
            "include": {
                "field": "secondary_exposure",
                "op": "gt",
                "value": 0,
            },
        }
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
                    "secondary_exposure": 40.0,
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
                    "secondary_exposure": 60.0,
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
        self.assertEqual(float(borrower["secondary_exposure"]), 100.0)
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

    def test_out_of_scope_consumer_stress_cecl_carries_base_without_scope_exception(
        self,
    ):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["modules"]["Consumer"]["fico_field"] = "does_not_exist"
        result = StressEngine(scenario, base_dir).run(write_outputs=False, run_comparison=False)
        cecl = result["reports"]["cecl_summary"]
        consumer = cecl[(cecl["portfolio"] == "Consumer") & (cecl["bucket"] == "Total")].set_index("stress_level")
        self.assertEqual(float(consumer.at["Base", "proforma_cecl_reserve"]), 4000.0)
        for level in ("S1", "S2"):
            self.assertEqual(
                float(consumer.at[level, "proforma_cecl_reserve"]), 4000.0
            )
            self.assertEqual(
                consumer.at[level, "cecl_reserve_status"], "available"
            )
            self.assertEqual(consumer.at[level, "exception_code"], "")
        self.assertNotIn("in_scope_balance", cecl.columns)
        self.assertNotIn("out_of_scope_balance", cecl.columns)
        exception_codes = set(result["reports"]["exception_log"]["code"])
        self.assertNotIn(
            "CONSUMER_CECL_UNAVAILABLE_OUT_OF_SCOPE", exception_codes
        )
        self.assertFalse(result["reports"]["out_of_scope_detail"].empty)
        consumer_summary = result["reports"]["consumer_summary"].set_index(
            "stress_level"
        )
        for level in ("Base", "S1", "S2"):
            self.assertEqual(
                float(consumer_summary.at[level, "expected_loss"]), 0.0
            )
            self.assertEqual(
                float(
                    consumer_summary.at[level, "qualitative_reserve"]
                ),
                4000.0,
            )
            self.assertEqual(
                float(
                    consumer_summary.at[
                        level, "proforma_cecl_reserve"
                    ]
                ),
                4000.0,
            )
            self.assertEqual(
                consumer_summary.at[level, "calculation_status"],
                "available",
            )
            self.assertEqual(
                float(
                    consumer_summary.at[level, "in_scope_balance"]
                ),
                0.0,
            )
            self.assertEqual(
                float(
                    consumer_summary.at[level, "out_of_scope_balance"]
                ),
                300000.0,
            )
            self.assertAlmostEqual(
                float(consumer_summary.at[level, "expected_loss"])
                + float(
                    consumer_summary.at[level, "qualitative_reserve"]
                ),
                float(
                    consumer_summary.at[
                        level, "proforma_cecl_reserve"
                    ]
                ),
            )
        self.assertEqual(
            list(consumer["proforma_cecl_reserve"]),
            sorted(consumer["proforma_cecl_reserve"]),
        )
        aggregate = cecl[
            (cecl["portfolio"] == "Aggregate")
            & (cecl["bucket"] == "Total")
        ].set_index("stress_level")
        self.assertEqual(
            list(aggregate["proforma_cecl_reserve"]),
            sorted(aggregate["proforma_cecl_reserve"]),
        )

    def test_consumer_qualitative_floor_preserves_base_and_monotonic_stress(
        self,
    ):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["modules"]["Consumer"][
            "qualitative_reserve_floor"
        ] = 5000.0
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False,
            run_comparison=False,
        )
        summary = result["reports"]["consumer_summary"].set_index(
            "stress_level"
        )
        cecl = result["reports"]["cecl_summary"]
        consumer_cecl = cecl[
            (cecl["portfolio"] == "Consumer")
            & (cecl["bucket"] == "Total")
        ].set_index("stress_level")
        expected = {
            "Base": (0.0, 4000.0, 4000.0),
            "S1": (377.85, 5000.0, 5377.85),
            "S2": (1403.04, 5000.0, 6403.04),
        }
        for level, (
            quantitative,
            qualitative,
            proforma,
        ) in expected.items():
            self.assertAlmostEqual(
                float(summary.at[level, "expected_loss"]),
                quantitative,
            )
            self.assertAlmostEqual(
                float(summary.at[level, "qualitative_reserve"]),
                qualitative,
            )
            self.assertAlmostEqual(
                float(summary.at[level, "proforma_cecl_reserve"]),
                proforma,
            )
            self.assertAlmostEqual(
                float(
                    consumer_cecl.at[
                        level, "proforma_cecl_reserve"
                    ]
                ),
                proforma,
            )
        self.assertEqual(
            list(summary["proforma_cecl_reserve"]),
            sorted(summary["proforma_cecl_reserve"]),
        )

        scenario["modules"]["Consumer"][
            "fico_field"
        ] = "does_not_exist"
        out_of_scope_result = StressEngine(scenario, base_dir).run(
            write_outputs=False,
            run_comparison=False,
        )
        out_of_scope_summary = out_of_scope_result["reports"][
            "consumer_summary"
        ].set_index("stress_level")
        expected_out_of_scope = {
            "Base": (0.0, 4000.0, 4000.0),
            "S1": (0.0, 5000.0, 5000.0),
            "S2": (0.0, 5000.0, 5000.0),
        }
        for level, (
            quantitative,
            qualitative,
            proforma,
        ) in expected_out_of_scope.items():
            self.assertAlmostEqual(
                float(
                    out_of_scope_summary.at[level, "expected_loss"]
                ),
                quantitative,
            )
            self.assertAlmostEqual(
                float(
                    out_of_scope_summary.at[
                        level, "qualitative_reserve"
                    ]
                ),
                qualitative,
            )
            self.assertAlmostEqual(
                float(
                    out_of_scope_summary.at[
                        level, "proforma_cecl_reserve"
                    ]
                ),
                proforma,
            )

    def test_consumer_cecl_carries_partial_stress_but_not_missing_field(
        self,
    ):
        scenario = {
            "stress_levels": ["S1", "S2"],
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "balance",
                "portfolio_field": "model_portfolio",
            },
            "cecl": {
                "portfolio_field": "cecl_portfolio",
                "reserve_field": "cecl_reserve",
                "portfolios": {
                    "Consumer": {"method": "expected_loss"}
                },
            },
            "modules": {},
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "C1",
                    "balance": 100.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "cecl_reserve": 10.0,
                    "module_applied": "Consumer",
                    "base_bucket": "Pass",
                    "consumer_el_unstressed": 8.0,
                    "consumer_el_S1": 10.0,
                    "consumer_el_S2": 13.0,
                    "consumer_qualitative_reserve": 2.0,
                    "consumer_proforma_cecl_S1": 12.0,
                    "consumer_proforma_cecl_S2": 15.0,
                },
                {
                    "borrower_id": "C2",
                    "balance": 200.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "cecl_reserve": 20.0,
                    "module_applied": "Consumer",
                    "base_bucket": "Pass",
                    "consumer_el_unstressed": 15.0,
                    "consumer_el_S1": np.nan,
                    "consumer_el_S2": 18.0,
                    "consumer_qualitative_reserve": 5.0,
                    "consumer_proforma_cecl_S1": np.nan,
                    "consumer_proforma_cecl_S2": 23.0,
                },
                {
                    "borrower_id": "C3",
                    "balance": 300.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "cecl_reserve": 30.0,
                    "module_applied": "Consumer",
                    "base_bucket": "Pass",
                    "consumer_el_unstressed": 20.0,
                    "consumer_el_S1": 25.0,
                    "consumer_el_S2": np.nan,
                    "consumer_qualitative_reserve": 10.0,
                    "consumer_proforma_cecl_S1": 35.0,
                    "consumer_proforma_cecl_S2": np.nan,
                },
            ]
        )
        bucket_summary = pd.DataFrame([{"portfolio": "Consumer"}])
        exceptions = []

        cecl = build_cecl_summary(
            results, bucket_summary, scenario, exceptions
        )
        consumer = cecl[
            (cecl["portfolio"] == "Consumer")
            & (cecl["bucket"] == "Total")
        ].set_index("stress_level")
        self.assertEqual(
            float(consumer.at["Base", "proforma_cecl_reserve"]), 60.0
        )
        self.assertEqual(
            float(consumer.at["S1", "proforma_cecl_reserve"]), 67.0
        )
        self.assertEqual(
            float(consumer.at["S2", "proforma_cecl_reserve"]), 73.0
        )
        self.assertAlmostEqual(
            float(consumer.at["S1", "proforma_cecl_ratio"]),
            67.0 / 600.0,
        )
        self.assertEqual(
            list(consumer["proforma_cecl_reserve"]),
            sorted(consumer["proforma_cecl_reserve"]),
        )
        self.assertTrue(
            consumer["cecl_reserve_status"].eq("available").all()
        )
        self.assertTrue(consumer["exception_code"].eq("").all())
        self.assertNotIn("in_scope_balance", cecl.columns)
        self.assertNotIn("out_of_scope_balance", cecl.columns)
        self.assertEqual(exceptions, [])

        consumer_report = build_consumer_summary(results, scenario).set_index(
            "stress_level"
        )
        expected_components = {
            "Base": (43.0, 17.0, 60.0),
            "S1": (50.0, 17.0, 67.0),
            "S2": (56.0, 17.0, 73.0),
        }
        for level, (expected_loss, qualitative, proforma) in expected_components.items():
            self.assertEqual(
                float(consumer_report.at[level, "expected_loss"]),
                expected_loss,
            )
            self.assertEqual(
                float(
                    consumer_report.at[level, "qualitative_reserve"]
                ),
                qualitative,
            )
            self.assertEqual(
                float(
                    consumer_report.at[
                        level, "proforma_cecl_reserve"
                    ]
                ),
                proforma,
            )
            self.assertAlmostEqual(
                expected_loss + qualitative,
                proforma,
            )
        self.assertEqual(
            float(consumer_report.at["Base", "in_scope_balance"]),
            600.0,
        )
        self.assertEqual(
            float(consumer_report.at["S1", "out_of_scope_balance"]),
            200.0,
        )
        self.assertEqual(
            float(consumer_report.at["S2", "out_of_scope_balance"]),
            300.0,
        )
        self.assertEqual(
            consumer_report.at["S1", "calculation_status"], "available"
        )
        targeted_scenario = dict(scenario)
        targeted_scenario["_targeted_mode"] = True
        targeted_results = results.copy()
        targeted_results["borrower_id"] = "C-SHARED"
        targeted_report = build_consumer_summary(
            targeted_results, targeted_scenario
        ).set_index("stress_level")
        self.assertEqual(
            int(targeted_report.at["S2", "borrower_count"]), 1
        )
        self.assertEqual(int(targeted_report.at["S2", "loan_count"]), 3)
        self.assertEqual(
            int(targeted_report.at["S2", "in_scope_borrower_count"]), 1
        )
        self.assertEqual(
            int(targeted_report.at["S2", "in_scope_loan_count"]), 2
        )

        missing_reserve_exceptions = []
        missing_reserve = build_cecl_summary(
            results.drop(columns=["cecl_reserve"]),
            bucket_summary,
            scenario,
            missing_reserve_exceptions,
        )
        missing_consumer = missing_reserve[
            (missing_reserve["portfolio"] == "Consumer")
            & (missing_reserve["bucket"] == "Total")
        ]
        self.assertTrue(
            missing_consumer["proforma_cecl_reserve"].isna().all()
        )
        self.assertTrue(
            missing_consumer["proforma_cecl_ratio"].isna().all()
        )
        self.assertTrue(
            missing_consumer["cecl_reserve_status"].eq("unavailable").all()
        )
        self.assertTrue(
            missing_consumer["exception_code"].eq(
                "CECL_RESERVE_FIELD_MISSING"
            ).all()
        )
        self.assertIn(
            "CECL_RESERVE_FIELD_MISSING",
            {row["code"] for row in missing_reserve_exceptions},
        )
        missing_aggregate = missing_reserve[
            (missing_reserve["portfolio"] == "Aggregate")
            & (missing_reserve["bucket"] == "Total")
        ]
        self.assertTrue(
            missing_aggregate["exception_code"].eq(
                "CECL_RESERVE_FIELD_MISSING"
            ).all()
        )

    def test_consumer_cecl_carries_prior_level_when_raw_el_declines(self):
        scenario = {
            "stress_levels": ["S1", "S2"],
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "balance",
                "portfolio_field": "model_portfolio",
            },
            "cecl": {
                "portfolio_field": "cecl_portfolio",
                "reserve_field": "cecl_reserve",
                "portfolios": {
                    "Consumer": {"method": "expected_loss"}
                },
            },
            "modules": {},
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "C1",
                    "balance": 100.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "cecl_reserve": 10.0,
                    "module_applied": "Consumer",
                    "base_bucket": "Pass",
                    "out_of_scope_S1": False,
                    "out_of_scope_S2": False,
                    "consumer_el_unstressed": 8.0,
                    "consumer_el_S1": 20.0,
                    "consumer_el_S2": 15.0,
                    "consumer_qualitative_reserve": 2.0,
                    "consumer_proforma_cecl_S1": 22.0,
                    "consumer_proforma_cecl_S2": 17.0,
                }
            ]
        )

        report = build_consumer_summary(results, scenario).set_index(
            "stress_level"
        )
        self.assertEqual(
            list(report["proforma_cecl_reserve"]),
            [10.0, 22.0, 22.0],
        )
        self.assertEqual(
            list(report["expected_loss"]),
            [8.0, 20.0, 20.0],
        )
        self.assertEqual(
            list(report["qualitative_reserve"]),
            [2.0, 2.0, 2.0],
        )
        for level in ("Base", "S1", "S2"):
            self.assertAlmostEqual(
                float(report.at[level, "expected_loss"])
                + float(report.at[level, "qualitative_reserve"]),
                float(report.at[level, "proforma_cecl_reserve"]),
            )

        cecl = build_cecl_summary(
            results,
            pd.DataFrame([{"portfolio": "Consumer"}]),
            scenario,
            [],
        )
        consumer = cecl[
            (cecl["portfolio"] == "Consumer")
            & (cecl["bucket"] == "Total")
        ].set_index("stress_level")
        self.assertEqual(
            list(consumer["proforma_cecl_reserve"]),
            [10.0, 22.0, 22.0],
        )
        self.assertEqual(float(results.at[0, "consumer_el_S2"]), 15.0)
        self.assertEqual(
            float(results.at[0, "consumer_proforma_cecl_S2"]), 17.0
        )

        flagged_out_of_scope = results.copy()
        flagged_out_of_scope["out_of_scope_S1"] = True
        flagged_report = build_consumer_summary(
            flagged_out_of_scope, scenario
        ).set_index("stress_level")
        self.assertEqual(
            list(flagged_report["proforma_cecl_reserve"]),
            [10.0, 10.0, 17.0],
        )
        self.assertEqual(
            list(flagged_report["expected_loss"]),
            [8.0, 8.0, 15.0],
        )
        self.assertEqual(
            float(flagged_report.at["S1", "in_scope_balance"]),
            0.0,
        )
        self.assertEqual(
            float(flagged_report.at["S1", "out_of_scope_balance"]),
            100.0,
        )

    def test_mixed_consumer_commercial_cecl_portfolio_is_rejected(self):
        scenario = {
            "stress_levels": ["S1"],
            "borrower": {
                "balance_field": "balance",
                "portfolio_field": "model_portfolio",
            },
            "cecl": {
                "portfolio_field": "cecl_portfolio",
                "reserve_field": "cecl_reserve",
            },
        }
        results = pd.DataFrame(
            [
                {
                    "balance": 100.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Shared",
                    "cecl_reserve": 5.0,
                    "module_applied": "Consumer",
                    "base_bucket": "Pass",
                },
                {
                    "balance": 200.0,
                    "model_portfolio": "CRE",
                    "cecl_portfolio": "Shared",
                    "cecl_reserve": 10.0,
                    "module_applied": "CRE",
                    "base_bucket": "Pass",
                },
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            "mixes Consumer and non-Consumer rows",
        ):
            build_cecl_summary(
                results,
                pd.DataFrame([{"portfolio": "Shared"}]),
                scenario,
                [],
            )

        scenario["cecl"]["portfolios"] = {
            "Shared": {"method": "bucket_reserve_ratio"}
        }
        with self.assertRaisesRegex(
            ValueError,
            "must use the 'expected_loss' method",
        ):
            build_cecl_summary(
                results.iloc[[0]].copy(),
                pd.DataFrame([{"portfolio": "Shared"}]),
                scenario,
                [],
            )

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

    def test_mixed_date_formats_and_timezones_parse_consistently(self):
        values = pd.Series(
            [
                "2024-01-31",
                "02/15/2024",
                "2024-03-01T00:00:00-05:00",
                "not-a-date",
            ]
        )

        parsed = parse_date_series(values)

        self.assertEqual(parsed.iloc[0], pd.Timestamp("2024-01-31"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2024-02-15"))
        self.assertEqual(parsed.iloc[2], pd.Timestamp("2024-03-01 05:00:00"))
        self.assertTrue(pd.isna(parsed.iloc[3]))

    def test_nullable_string_equality_masks_are_boolean(self):
        frame = pd.DataFrame(
            {"value": pd.Series([pd.NA, "equal", "different"], dtype="string")}
        )

        equal = evaluate_conditions(
            frame, {"field": "value", "op": "eq", "value": "equal"}
        )
        not_equal = evaluate_conditions(
            frame, {"field": "value", "op": "ne", "value": "equal"}
        )

        self.assertEqual(equal.tolist(), [False, True, False])
        self.assertEqual(not_equal.tolist(), [True, False, True])
        self.assertEqual(equal.dtype, bool)
        self.assertEqual(not_equal.dtype, bool)

    def test_compare_values_always_returns_bool_for_missing_scalars(self):
        cases = [
            (pd.NA, "value", False),
            ("value", pd.NA, False),
            (pd.NA, pd.NA, True),
            (np.nan, None, True),
            ("value", "value", True),
        ]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                result = compare_values(left, right)
                self.assertIs(type(result), bool)
                self.assertEqual(result, expected)

    def test_nullable_string_text_predicates_never_match_missing_values(self):
        frame = pd.DataFrame(
            {
                "value": pd.Series(
                    [pd.NA, "banana", "<literal", "literal>"],
                    dtype="string",
                )
            }
        )
        cases = [
            ("contains", "na", [False, True, False, False]),
            ("startswith", "<", [False, False, True, False]),
            ("endswith", ">", [False, False, False, True]),
            ("regex", r"^<", [False, False, True, False]),
        ]
        for operation, value, expected in cases:
            with self.subTest(operation=operation):
                mask = evaluate_conditions(
                    frame,
                    {"field": "value", "op": operation, "value": value},
                )
                self.assertEqual(mask.tolist(), expected)
                self.assertEqual(mask.dtype, bool)


if __name__ == "__main__":
    unittest.main()
