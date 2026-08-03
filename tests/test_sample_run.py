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

            identity_columns = set(pd.read_csv(ROOT / "examples" / "data" / "loans.csv").columns)
            self.assertNotIn("portfolio", identity_columns)
            self.assertNotIn("module", identity_columns)
            self.assertNotIn("cre_subsector", identity_columns)
            self.assertNotIn("ci_sector", identity_columns)
            self.assertIn("subsector", identity_columns)
            self.assertIn("tag_hint", identity_columns)
            self.assertIn("tag_hint_2", identity_columns)
            self.assertNotIn("cecl_reserve_prior_1", identity_columns)
            self.assertNotIn("cecl_reserve_prior_2", identity_columns)

            history = pd.read_csv(
                ROOT / "examples" / "data" / "cecl_history.csv"
            )
            self.assertEqual(
                set(history.columns),
                {"cecl_portfolio", "period", "historical_cecl_reserve"},
            )
            self.assertNotIn("Consumer", set(history["cecl_portfolio"]))
            self.assertFalse(
                history.duplicated(["cecl_portfolio", "period"]).any()
            )
            expected_history_keys = {
                (portfolio, period)
                for portfolio in {
                    "Asset-Based Lending",
                    "BCC",
                    "CRE",
                    "EF",
                    "Middle Market",
                    "Sponsor and Specialty",
                }
                for period in {"2026Q1", "2025Q4"}
            }
            self.assertEqual(
                set(history[["cecl_portfolio", "period"]].itertuples(index=False, name=None)),
                expected_history_keys,
            )
            self.assertIs(
                scenario["inputs"]["sources"]["cecl_history"]["merge"],
                False,
            )

            financial_columns = set(pd.read_csv(ROOT / "examples" / "data" / "financials.csv").columns)
            collateral_columns = set(pd.read_csv(ROOT / "examples" / "data" / "collateral.csv").columns)
            consumer_columns = set(pd.read_csv(ROOT / "examples" / "data" / "consumer_scores_1.csv").columns)
            self.assertTrue({"current_dscr", "prior_dscr", "origination_dscr", "noi"}.isdisjoint(financial_columns))
            self.assertNotIn("fccr", financial_columns)
            self.assertTrue(
                {"interest_expense", "non_cash_interest_expense"}.issubset(financial_columns)
            )
            self.assertTrue(
                {"Current DSCR", "Prior DSCR", "Origination DSCR", "Net Operating Income"}.issubset(
                    collateral_columns
                )
            )
            self.assertTrue({"current_dscr", "prior_dscr", "origination_dscr", "noi"}.isdisjoint(collateral_columns))
            self.assertTrue(
                {"collateral_id", "current_appraised_value_raw", "prior_appraised_value"}.issubset(
                    collateral_columns
                )
            )
            self.assertTrue(
                {"current_fico_score", "collateral_id", "current_appraised_value_raw"}.issubset(
                    consumer_columns
                )
            )
            self.assertIn("account_number", consumer_columns)
            self.assertNotIn("borrower_id", consumer_columns)

            financial_import = scenario["inputs"]["sources"]["financials"]
            collateral_import = scenario["inputs"]["sources"]["collateral"]
            consumer_import = scenario["inputs"]["sources"]["consumer_scores"]
            self.assertNotIn("fccr", financial_import["numeric_columns"])
            self.assertNotIn("aggregation", financial_import)
            self.assertEqual(
                financial_import["column_aliases"]["interest_expense"],
                "interest_expense",
            )
            self.assertEqual(
                financial_import["column_aliases"]["non_cash_interest_expense"],
                "non_cash_interest_expense",
            )
            self.assertIn("interest_expense", financial_import["numeric_columns"])
            self.assertIn("non_cash_interest_expense", financial_import["numeric_columns"])
            self.assertIn("dscr", collateral_import["aggregation"])
            self.assertIn("noi", collateral_import["aggregation"])
            self.assertIn("current_appraised_value", collateral_import["aggregation"])
            self.assertNotIn("consumer_appraised_value", collateral_import["aggregation"])
            self.assertEqual(collateral_import["column_aliases"]["current_dscr"], "Current DSCR")
            self.assertEqual(collateral_import["column_aliases"]["noi"], "Net Operating Income")
            self.assertEqual(len(consumer_import["paths"]), 2)
            self.assertEqual(consumer_import["key"], "loan_id")
            self.assertEqual(consumer_import["identity_key"], "loan_id")
            self.assertEqual(consumer_import["column_aliases"]["loan_id"], "account_number")
            self.assertNotIn("borrower_id", consumer_import["column_aliases"])
            self.assertIn("fico_score", consumer_import["aggregation"])
            self.assertIn("consumer_appraised_value", consumer_import["aggregation"])
            self.assertTrue(
                scenario["modules"]["C&I"]["sectors"]["Asset-Based Lending"][
                    "use_calculated_cash_paid_for_interest"
                ]
            )
            self.assertEqual(
                set(scenario["modules"]["C&I"]["ebitda_reduction"]["default"]),
                {str(brg) for brg in range(1, 9)},
            )

            borrowers = result["borrowers"]
            self.assertEqual(len(borrowers), 15)
            b001 = borrowers.loc[borrowers["borrower_id"] == "B001"].iloc[0]
            self.assertEqual(float(b001["outstanding_balance"]), 1250000.0)
            self.assertEqual(int(b001["loan_count"]), 2)
            self.assertEqual(float(b001["dscr"]), 1.20)
            self.assertEqual(b001["dscr_source_field"], "prior_dscr")
            self.assertEqual(b001["dscr_candidate"], "prior")
            overlap = borrowers.loc[borrowers["borrower_id"] == "B011"].iloc[0]
            self.assertIn("CI_Model", overlap["model_tags"])
            self.assertIn("CRE_Model", overlap["model_tags"])
            self.assertIn("CRE_Subsector_Retail", overlap["all_tags"])
            self.assertIn("CI_Sector_Middle_Market", overlap["all_tags"])
            self.assertEqual(overlap["subsector"], "Retail")
            self.assertTrue(pd.isna(overlap["tag_hint"]))
            self.assertEqual(overlap["tag_hint_2"], "Middle Market")
            self.assertEqual(overlap["eligible_modules"], "CRE;C&I")
            self.assertEqual(overlap["primary_module"], "CRE")
            self.assertEqual(overlap["model_portfolio"], "CRE")
            self.assertEqual(overlap["model_module"], "CRE")
            self.assertEqual(overlap["cecl_portfolio"], "CRE")
            self.assertEqual(overlap["cre_subsector"], "Retail")
            self.assertEqual(overlap["ci_sector"], "Middle Market")
            self.assertEqual(float(overlap["dscr"]), 1.25)
            self.assertEqual(overlap["dscr_source_field"], "prior_dscr")
            b002 = borrowers.loc[borrowers["borrower_id"] == "B002"].iloc[0]
            self.assertEqual(float(b002["current_appraised_value"]), 1500000.0)
            self.assertTrue(pd.isna(b002["consumer_appraised_value"]))

            consumer_borrower = borrowers.loc[borrowers["borrower_id"] == "B007"].iloc[0]
            self.assertEqual(float(consumer_borrower["fico_score"]), 680.0)
            self.assertEqual(consumer_borrower["fico_source_field"], "current_fico_score")
            self.assertTrue(consumer_borrower["fico_source_record"].endswith("consumer_scores_2.csv#row=1"))
            self.assertEqual(float(consumer_borrower["consumer_appraised_value"]), 340000.0)
            self.assertEqual(
                consumer_borrower["consumer_appraisal_source_field"],
                "current_appraised_value_raw",
            )
            self.assertEqual(consumer_borrower["consumer_appraisal_candidate"], "current")
            self.assertTrue(
                consumer_borrower["consumer_appraisal_source_record"].endswith(
                    "consumer_scores_2.csv#row=1"
                )
            )

            reconciliation = result["reports"]["source_reconciliation"]
            consumer_reconciliation = reconciliation.loc[reconciliation["source"] == "consumer_scores"].iloc[0]
            self.assertEqual(int(consumer_reconciliation["file_count"]), 2)
            self.assertEqual(consumer_reconciliation["key_field"], "loan_id")
            self.assertEqual(consumer_reconciliation["identity_key_field"], "loan_id")
            self.assertEqual(int(consumer_reconciliation["unique_key_count"]), 1)
            self.assertEqual(int(consumer_reconciliation["matched_source_key_count"]), 1)
            self.assertEqual(int(consumer_reconciliation["orphan_source_key_count"]), 0)
            self.assertEqual(int(consumer_reconciliation["matched_borrower_count"]), 1)
            self.assertEqual(int(consumer_reconciliation["unmatched_borrower_count"]), 14)
            self.assertEqual(float(consumer_reconciliation["matched_borrower_balance"]), 300000.0)
            consumer_metadata = [
                item for item in result["metadata"]["input_files"] if item["name"] == "consumer_scores"
            ]
            self.assertEqual(len(consumer_metadata), 2)

            tag_summary = result["reports"]["tag_summary"]
            tieouts = tag_summary[tag_summary["tie_out_name"].notna()]
            self.assertTrue(tieouts["passed"].astype(bool).all())

            overlay_summary = result["reports"]["overlay_summary"]
            bcc_overlay = overlay_summary[
                (overlay_summary["portfolio"] == "BCC")
                & (overlay_summary["stress_level"] == "S1")
            ].iloc[0]
            self.assertEqual(bcc_overlay["source_names"], "CRE;C&I")
            self.assertEqual(bcc_overlay["source_weights"], "CRE=0.65;C&I=0.35")
            self.assertEqual(
                bcc_overlay["source_selection"],
                "CRE(tags=CRE_Model;primary_module=required);C&I(tags=CI_Model;primary_module=required)",
            )

            out_scope = result["reports"]["out_of_scope_detail"]
            self.assertEqual(set(out_scope["borrower_id"]), {"B004"})
            self.assertEqual(set(out_scope["field"]), {"dscr"})

            overlap_result = result["results"].loc[result["results"]["borrower_id"] == "B011"].iloc[0]
            self.assertEqual(overlap_result["module_applied"], "CRE")
            self.assertGreater(float(overlap_result["cre_dscr_S1"]), 0)
            self.assertTrue(pd.isna(overlap_result["ci_fccr_S1"]))

            abl_result = result["results"].loc[result["results"]["borrower_id"] == "B010"].iloc[0]
            self.assertEqual(float(abl_result["calculated_cash_paid_for_interest"]), 24000.0)
            self.assertEqual(
                abl_result["calculated_cash_paid_for_interest_source"],
                "interest_expense_less_non_cash_interest_expense",
            )
            self.assertTrue(
                pd.isna(abl_result["calculated_cash_paid_for_interest_fallback_reason"])
            )
            self.assertEqual(float(abl_result["ci_debt_service_S1"]), 63500.0)

            for borrower_id in ("B014", "B015"):
                grade_eight = result["results"].loc[
                    result["results"]["borrower_id"] == borrower_id
                ].iloc[0]
                self.assertFalse(pd.isna(grade_eight["ci_fccr_S1"]))
                self.assertFalse(pd.isna(grade_eight["ci_fccr_S2"]))
                self.assertEqual(grade_eight["stressed_bucket_S1"], "Substandard")
                self.assertEqual(grade_eight["stressed_bucket_S2"], "Substandard")

            consumer = result["reports"]["consumer_summary"]
            s1 = float(consumer.loc[consumer["stress_level"] == "S1", "expected_loss"].iloc[0])
            s2 = float(consumer.loc[consumer["stress_level"] == "S2", "expected_loss"].iloc[0])
            self.assertGreater(s2, s1)
            self.assertGreater(s1, 0)

            input_summary = result["reports"]["input_summary"]
            self.assertIn(
                "cecl_history", set(input_summary["dataset"].astype(str))
            )
            self.assertNotIn("cecl_reserve_prior_1", result["borrowers"].columns)
            self.assertNotIn("cecl_reserve_prior_2", result["borrowers"].columns)
            basis_audit = result["reports"]["cecl_basis_summary"]
            consumer_basis = basis_audit[
                basis_audit["portfolio"].eq("Consumer")
            ]
            self.assertEqual(set(consumer_basis["period"]), {"current"})
            self.assertEqual(set(consumer_basis["reserve_basis"]), {"in_place"})
            commercial_history = basis_audit[
                basis_audit["period"].isin(["2026Q1", "2025Q4"])
            ]
            self.assertFalse(commercial_history.empty)
            self.assertEqual(
                set(commercial_history["period_method"]),
                {"portfolio_history"},
            )

            output_files = {path.name for path in Path(tmp).iterdir()}
            self.assertIn("borrower_audit_raw.csv", output_files)
            self.assertIn("stressed_borrower_results.csv", output_files)
            self.assertIn("cecl_summary.csv", output_files)
            self.assertIn("cecl_basis_summary.csv", output_files)
            self.assertIn("exception_log.csv", output_files)
            self.assertIn("metadata.json", output_files)
            self.assertIn("output_manifest.json", output_files)
            self.assertIn("source_reconciliation.csv", output_files)

            cecl = result["reports"]["cecl_summary"]
            self.assertNotIn("in_scope_balance", cecl.columns)
            self.assertNotIn("out_of_scope_balance", cecl.columns)
            consumer_cecl = cecl[
                (cecl["portfolio"] == "Consumer")
                & (cecl["bucket"] == "Total")
            ].set_index("stress_level")
            self.assertEqual(
                float(
                    consumer_cecl.at[
                        "Base", "proforma_cecl_reserve"
                    ]
                ),
                4000.0,
            )
            self.assertAlmostEqual(
                float(
                    consumer_cecl.at[
                        "S1", "proforma_cecl_reserve"
                    ]
                ),
                4377.85,
            )
            self.assertAlmostEqual(
                float(
                    consumer_cecl.at[
                        "S2", "proforma_cecl_reserve"
                    ]
                ),
                5403.04,
            )
            unavailable = cecl[cecl["cecl_reserve_status"] == "unavailable"]
            self.assertTrue(unavailable.empty)
            cre_s2_sub = cecl[
                (cecl["portfolio"] == "CRE")
                & (cecl["stress_level"] == "S2")
                & (cecl["bucket"] == "Substandard")
            ].iloc[0]
            self.assertFalse(pd.isna(cre_s2_sub["reserve_ratio"]))
            self.assertFalse(pd.isna(cre_s2_sub["proforma_cecl_reserve"]))

            exceptions = result["reports"]["exception_log"]
            self.assertNotIn("CECL_RESERVE_RATIO_UNAVAILABLE", set(exceptions["code"]))
            self.assertNotIn("CI_CALCULATED_CASH_INTEREST_FALLBACK", set(exceptions["code"]))
            self.assertIn("CECL_LOAN_RESERVE_MISSING_TREATED_AS_ZERO", set(exceptions["code"]))
            self.assertEqual(result["metadata"]["exception_count"], len(exceptions))

            consumer_base = cecl[
                (cecl["portfolio"] == "Consumer")
                & (cecl["stress_level"] == "Base")
                & (cecl["bucket"] == "Total")
            ].iloc[0]
            self.assertEqual(float(consumer_base["proforma_cecl_reserve"]), 4000.0)
            aggregate_base = cecl[
                (cecl["portfolio"] == "Aggregate")
                & (cecl["stress_level"] == "Base")
                & (cecl["bucket"] == "Total")
            ].iloc[0]
            self.assertEqual(float(aggregate_base["proforma_cecl_reserve"]), 76000.0)

    def test_arr_retains_sponsor_population_but_is_excluded_from_models(self):
        scenario, base_dir = load_scenario(SCENARIO)
        # Use a deterministic sample attribute to designate B006 as ARR without
        # changing the shared example input used by the other regression tests.
        scenario["tags"]["ARR"]["include"] = {
            "all": [
                {
                    "field": "subsector",
                    "op": "has_token",
                    "value": "Sponsor and Specialty",
                },
                {
                    "field": "naics_code",
                    "op": "eq",
                    "value": 213112,
                },
            ]
        }
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False, run_comparison=False
        )

        borrowers = result["borrowers"].set_index("borrower_id")
        arr = borrowers.loc["B006"]
        sponsor_control = borrowers.loc["B014"]
        self.assertTrue(bool(arr["tag_arr"]))
        self.assertTrue(bool(arr["tag_ci_sector_sponsor_and_specialty"]))
        self.assertTrue(bool(arr["tag_ci_model"]))
        self.assertIn("CI_Sector_Sponsor_and_Specialty", arr["all_tags"])
        self.assertIn("ARR", arr["all_tags"])
        self.assertEqual(arr["ci_sector"], "Sponsor and Specialty")
        self.assertTrue(bool(arr["model_excluded"]))
        self.assertEqual(arr["model_exclusion_tags"], "ARR")
        self.assertEqual(arr["model_tags"], "")
        self.assertEqual(arr["eligible_modules"], "")
        for field in (
            "primary_module",
            "model_module",
            "model_portfolio",
            "cecl_portfolio",
        ):
            self.assertTrue(pd.isna(arr[field]), msg=field)

        self.assertFalse(bool(sponsor_control["model_excluded"]))
        self.assertEqual(sponsor_control["primary_module"], "C&I")
        self.assertEqual(
            sponsor_control["cecl_portfolio"], "Sponsor and Specialty"
        )

        stressed = result["results"].set_index("borrower_id")
        self.assertEqual(stressed.at["B006", "module_applied"], "")
        self.assertTrue(pd.isna(stressed.at["B006", "ci_fccr_S1"]))
        self.assertTrue(pd.isna(stressed.at["B006", "ci_fccr_S2"]))
        self.assertEqual(stressed.at["B014", "module_applied"], "C&I")

        tag_summary = result["reports"]["tag_summary"]
        populations = tag_summary[tag_summary["tie_out_name"].isna()].set_index(
            "tag"
        )
        self.assertEqual(int(populations.at["ARR", "borrower_count"]), 1)
        self.assertEqual(float(populations.at["ARR", "balance"]), 500000.0)
        self.assertEqual(
            int(
                populations.at[
                    "ARR", "not_model_excluded_borrower_count"
                ]
            ),
            0,
        )
        self.assertEqual(
            float(populations.at["ARR", "not_model_excluded_balance"]),
            0.0,
        )
        self.assertEqual(
            float(populations.at["ARR", "model_excluded_balance"]), 500000.0
        )
        self.assertEqual(
            int(
                populations.at[
                    "CI_Sector_Sponsor_and_Specialty", "borrower_count"
                ]
            ),
            2,
        )
        self.assertEqual(
            float(
                populations.at["CI_Sector_Sponsor_and_Specialty", "balance"]
            ),
            600000.0,
        )
        self.assertEqual(
            float(
                populations.at[
                    "CI_Sector_Sponsor_and_Specialty",
                    "not_model_excluded_balance",
                ]
            ),
            100000.0,
        )
        self.assertEqual(
            float(
                populations.at[
                    "CI_Sector_Sponsor_and_Specialty",
                    "model_excluded_balance",
                ]
            ),
            500000.0,
        )
        sponsor_tieout = tag_summary[
            (
                tag_summary["tag"]
                == "CI_Sector_Sponsor_and_Specialty"
            )
            & tag_summary["tie_out_name"].notna()
        ].iloc[0]
        self.assertTrue(bool(sponsor_tieout["passed"]))
        self.assertEqual(float(sponsor_tieout["actual"]), 600000.0)

        ci_sponsor = result["reports"]["ci_summary"].loc[
            result["reports"]["ci_summary"]["sector"]
            == "Sponsor and Specialty"
        ].iloc[0]
        self.assertEqual(int(ci_sponsor["borrower_count"]), 1)
        self.assertEqual(float(ci_sponsor["balance"]), 100000.0)

        ci_base = result["reports"]["migration_summary"]
        ci_base = ci_base[
            (ci_base["portfolio"] == "C&I")
            & (ci_base["stress_level"] == "Base")
        ]
        self.assertEqual(float(ci_base["balance"].sum()), 1450000.0)

        cecl_sponsor_total = result["reports"]["cecl_summary"]
        cecl_sponsor_total = cecl_sponsor_total[
            (
                cecl_sponsor_total["portfolio"]
                == "Sponsor and Specialty"
            )
            & (cecl_sponsor_total["stress_level"] == "Base")
            & (cecl_sponsor_total["bucket"] == "Total")
        ].iloc[0]
        self.assertEqual(float(cecl_sponsor_total["balance"]), 100000.0)

    def test_repeated_runs_are_deterministic_for_core_reports(self):
        first = self._run()
        second = self._run()
        for report_name in ["migration_summary", "cecl_summary", "consumer_summary", "out_of_scope_summary", "exception_log"]:
            left = first["reports"][report_name].fillna("").sort_index(axis=1).reset_index(drop=True)
            right = second["reports"][report_name].fillna("").sort_index(axis=1).reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right, check_dtype=False)

    def test_changed_scenario_variable_produces_marginal_report(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["modules"]["Consumer"]["pd_increase_factor"]["S1"] = 2.00
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

    def test_cre_cecl_can_remain_at_subsector_level(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["modules"]["CRE"].pop("cecl_portfolio_rollup")
        result = StressEngine(scenario, base_dir).run(write_outputs=False, run_comparison=False)
        overlap = result["borrowers"].loc[result["borrowers"]["borrower_id"] == "B011"].iloc[0]
        self.assertEqual(overlap["cecl_portfolio"], "Retail")
        cecl = result["reports"]["cecl_summary"]
        self.assertIn("Retail", set(cecl["portfolio"]))
        retail_rows = cecl[(cecl["portfolio"] == "Retail") & (cecl["bucket"] == "Total")]
        self.assertFalse(retail_rows.empty)

    def test_consumer_expected_loss_components_roll_into_cecl_totals(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["modules"]["Consumer"]["cecl_portfolio_rollup"] = "Retail Consumer"
        scenario["cecl"]["portfolios"] = {}
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False,
            run_comparison=False,
        )

        consumer_summary = result["reports"]["consumer_summary"].set_index(
            "stress_level"
        )
        cecl = result["reports"]["cecl_summary"]
        consumer_cecl = cecl[
            (cecl["portfolio"] == "Retail Consumer")
            & (cecl["bucket"] == "Total")
        ].set_index("stress_level")
        consumer_result = result["results"][
            result["results"]["module_applied"]
            .astype(str)
            .str.contains("Consumer", na=False)
        ].iloc[0]

        self.assertEqual(set(consumer_cecl["method"]), {"expected_loss"})
        self.assertEqual(set(consumer_cecl["reserve_basis"]), {"in_place"})
        self.assertEqual(
            float(consumer_result["consumer_cecl_reserve_base"]),
            float(consumer_result["cecl_effective_reserve_base"]),
        )
        for level in ("S1", "S2"):
            self.assertAlmostEqual(
                float(consumer_result[f"consumer_proforma_cecl_{level}"]),
                float(consumer_result[f"consumer_el_{level}"])
                + float(consumer_result["consumer_qualitative_reserve"]),
            )
        for level in ("Base", "S1", "S2"):
            quantitative = float(consumer_summary.at[level, "expected_loss"])
            qualitative = float(
                consumer_summary.at[level, "qualitative_reserve"]
            )
            self.assertAlmostEqual(
                float(consumer_cecl.at[level, "proforma_cecl_reserve"]),
                quantitative + qualitative,
            )

            portfolio_totals = cecl[
                (cecl["stress_level"] == level)
                & (cecl["bucket"] == "Total")
                & (cecl["portfolio"] != "Aggregate")
            ]
            aggregate = cecl[
                (cecl["portfolio"] == "Aggregate")
                & (cecl["stress_level"] == level)
                & (cecl["bucket"] == "Total")
            ].iloc[0]
            self.assertAlmostEqual(
                float(aggregate["proforma_cecl_reserve"]),
                float(
                    pd.to_numeric(
                        portfolio_totals["proforma_cecl_reserve"],
                        errors="raise",
                    ).sum()
                ),
            )

    def test_overlay_source_weights_change_tagged_source_ratios(self):
        scenario, base_dir = load_scenario(SCENARIO)
        scenario["overlays"]["BCC"]["sources"][0]["weight"] = 1.0
        scenario["overlays"]["BCC"]["sources"][1]["weight"] = 0.0
        cre_only = StressEngine(scenario, base_dir).run(write_outputs=False, run_comparison=False)
        cre_ratio = float(
            cre_only["reports"]["overlay_summary"].loc[
                (cre_only["reports"]["overlay_summary"]["portfolio"] == "BCC")
                & (cre_only["reports"]["overlay_summary"]["stress_level"] == "S1"),
                "weighted_source_stressed_substandard_ratio",
            ].iloc[0]
        )

        scenario, base_dir = load_scenario(SCENARIO)
        scenario["overlays"]["BCC"]["sources"][0]["weight"] = 0.0
        scenario["overlays"]["BCC"]["sources"][1]["weight"] = 1.0
        ci_only = StressEngine(scenario, base_dir).run(write_outputs=False, run_comparison=False)
        ci_ratio = float(
            ci_only["reports"]["overlay_summary"].loc[
                (ci_only["reports"]["overlay_summary"]["portfolio"] == "BCC")
                & (ci_only["reports"]["overlay_summary"]["stress_level"] == "S1"),
                "weighted_source_stressed_substandard_ratio",
            ].iloc[0]
        )

        self.assertNotEqual(cre_ratio, ci_ratio)


if __name__ == "__main__":
    unittest.main()
