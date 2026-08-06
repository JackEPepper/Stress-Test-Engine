from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stress_engine.config import load_scenario, validate_scenario
from stress_engine.batch import _base_cecl_lookup
from stress_engine.comparison import _cecl_impact_rows
from stress_engine.engine import StressEngine
from stress_engine.targeted import (
    _baseline_parameter,
    _copy_rows,
    _evaluate_selector,
    _resolve_variant,
    validate_targeted_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _targeted_validation_scenario(values):
    return {
        "stress_levels": ["S1", "S2"],
        "targeted_stress": {
            "shocks": {
                "shock": {
                    "selector": {
                        "type": "condition",
                        "field": "loan_id",
                        "op": "eq",
                        "value": "L1",
                    },
                    "default_tier": "high",
                    "tiers": {
                        "high": {
                            "modules": {
                                "C&I": {
                                    "ebitda_reduction": {
                                        "operation": "add",
                                        "values": values,
                                    }
                                }
                            }
                        }
                    },
                }
            },
            "variants": {
                "variant": {
                    "shocks": ["shock"],
                    "unmatched_behavior": "base",
                }
            },
        },
    }


class TargetedStressTest(unittest.TestCase):
    def test_copy_rows_preserves_new_nullable_and_datetime_columns(self):
        target = pd.DataFrame({"amount": [1.0, 2.0]})
        source = pd.DataFrame(
            {
                "amount": [10.0, 20.0],
                "label": pd.Series(["copied", "not copied"], dtype="string"),
                "as_of_date": pd.to_datetime(
                    ["2026-01-31", "2026-02-28"]
                ),
            }
        )

        _copy_rows(target, source, pd.Series([True, False]))

        self.assertEqual(target["amount"].tolist(), [10.0, 2.0])
        self.assertEqual(target.at[0, "label"], "copied")
        self.assertTrue(pd.isna(target.at[1, "label"]))
        self.assertEqual(target.at[0, "as_of_date"], pd.Timestamp("2026-01-31"))
        self.assertTrue(pd.isna(target.at[1, "as_of_date"]))

    def test_cecl_priority_overlap_is_logged_once_at_loan_grain(self):
        scenario, base_dir = load_scenario(
            ROOT / "examples" / "targeted_stress.json"
        )
        bcc_name = "Business_Credit_Center_Overlay"
        scenario["tags"][bcc_name]["include"] = {
            "field": "subsector",
            "op": "has_any_token",
            "value": ["Business Credit Center", "Equipment Finance"],
        }
        bcc_spec = scenario["tags"].pop(bcc_name)
        scenario["tags"] = {bcc_name: bcc_spec, **scenario["tags"]}

        result = StressEngine(scenario, base_dir).run(
            write_outputs=False, run_comparison=False
        )

        context = result["loan_context"].set_index("loan_id")
        ef = context.loc["L009"]
        self.assertEqual(ef["borrower_id"], "B008")
        self.assertTrue(bool(ef["tag_equipment_finance_overlay"]))
        self.assertTrue(bool(ef["tag_business_credit_center_overlay"]))
        self.assertEqual(ef["cecl_level_tag"], "Equipment_Finance_Overlay")
        self.assertEqual(ef["model_portfolio"], "EF")
        self.assertEqual(ef["cecl_portfolio"], "EF")

        events = result["reports"]["exception_log"]
        events = events[
            events["code"].eq(
                "CECL_LEVEL_TAG_OVERLAP_RESOLVED_BY_PRIORITY"
            )
        ]
        self.assertEqual(len(events), 1)
        event = events.iloc[0]
        self.assertEqual(event["borrower_id"], "B008")
        self.assertEqual(event["loan_id"], "L009")
        self.assertEqual(event["scenario_variant"], "all")

        tag_summary = result["reports"]["tag_summary"]
        bcc = tag_summary[
            tag_summary["tag"].eq(bcc_name)
            & tag_summary["tie_out_name"].isna()
        ].iloc[0]
        self.assertEqual(int(bcc["borrower_count"]), 2)
        self.assertEqual(int(bcc["loan_count"]), 2)
        self.assertEqual(int(bcc["cecl_selected_borrower_count"]), 1)
        self.assertEqual(int(bcc["cecl_selected_loan_count"]), 1)
        self.assertEqual(float(bcc["cecl_selected_balance"]), 200000.0)

    def test_targeted_example_runs_baseline_layered_and_isolated_variants(self):
        scenario, base_dir = load_scenario(ROOT / "examples" / "targeted_stress.json")
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False, run_comparison=False
        )

        self.assertEqual(
            set(result["variant_results"]["scenario_variant"]),
            {"baseline", "oil_layered", "tariff_isolated", "oil_and_tariffs"},
        )
        summary = result["reports"]["targeted_stress_summary"]
        tariff = summary[
            (summary["scenario_variant"] == "tariff_isolated")
            & (summary["shock"] == "tariff_shock")
        ]
        self.assertEqual(int(tariff["loan_count"].sum()), 3)
        self.assertEqual(int(tariff["borrower_count"].sum()), 3)
        self.assertIn("scenario_variant", result["reports"]["migration_summary"])
        self.assertIn("loan_count", result["reports"]["migration_summary"])
        cre_population = result["reports"]["tag_summary"]
        cre_population = cre_population[
            (cre_population["tag"] == "CRE_Model")
            & cre_population["tie_out_name"].isna()
        ].iloc[0]
        self.assertEqual(int(cre_population["borrower_count"]), 6)
        self.assertEqual(int(cre_population["loan_count"]), 7)
        self.assertEqual(
            int(cre_population["not_model_excluded_borrower_count"]),
            6,
        )
        self.assertEqual(
            int(cre_population["not_model_excluded_loan_count"]),
            7,
        )

    def test_arr_exclusion_blocks_targeted_selection_and_assumptions(self):
        scenario, base_dir = load_scenario(
            ROOT / "examples" / "targeted_stress.json"
        )
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

        context = result["loan_context"].set_index("loan_id")
        arr = context.loc["L007"]
        self.assertTrue(bool(arr["tag_arr"]))
        self.assertTrue(bool(arr["tag_ci_sector_sponsor_and_specialty"]))
        self.assertTrue(bool(arr["model_excluded"]))
        self.assertEqual(arr["model_exclusion_tags"], "ARR")
        self.assertTrue(pd.isna(arr["primary_module"]))

        selection = result["reports"]["targeted_selection_detail"]
        arr_oil = selection[
            (selection["loan_id"] == "L007")
            & (selection["shock"] == "oil_shock")
        ]
        self.assertEqual(len(arr_oil), 2)
        self.assertTrue(arr_oil["selector_matched"].astype(bool).all())
        self.assertTrue(arr_oil["model_excluded"].astype(bool).all())
        self.assertTrue(arr_oil["selection_reason"].eq("model_excluded").all())
        self.assertFalse(arr_oil["selected"].astype(bool).any())
        control_oil = selection[
            (selection["loan_id"] == "L006")
            & (selection["shock"] == "oil_shock")
        ]
        self.assertEqual(len(control_oil), 2)
        self.assertTrue(control_oil["selected"].astype(bool).all())

        assumption_audit = result["reports"]["targeted_assumption_audit"]
        self.assertFalse((assumption_audit["loan_id"] == "L007").any())
        self.assertTrue((assumption_audit["loan_id"] == "L006").any())

        arr_results = result["variant_results"][
            result["variant_results"]["loan_id"] == "L007"
        ]
        self.assertEqual(len(arr_results), 4)
        self.assertTrue(arr_results["model_excluded"].astype(bool).all())
        self.assertTrue(arr_results["module_applied"].eq("").all())
        self.assertTrue(arr_results["ci_fccr_S1"].isna().all())
        self.assertTrue(arr_results["ci_fccr_S2"].isna().all())

    def test_all_model_excluded_targeted_population_returns_empty_model_reports(
        self,
    ):
        scenario, base_dir = load_scenario(
            ROOT / "examples" / "targeted_stress.json"
        )
        scenario["tags"]["Exclude_All_For_Test"] = {
            "model_eligible": False,
            "exclude_from_model": True,
            "include": {
                "field": "outstanding_balance",
                "op": "gt",
                "value": 0,
            },
        }
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False, run_comparison=False
        )

        self.assertTrue(
            result["loan_context"]["model_excluded"].astype(bool).all()
        )
        migration = result["reports"]["migration_summary"]
        self.assertTrue(migration.empty)
        self.assertIn("scenario_variant", migration.columns)
        self.assertIn("loan_count", migration.columns)
        cecl = result["reports"]["cecl_summary"]
        aggregate = cecl[
            (cecl["portfolio"] == "Aggregate")
            & (cecl["bucket"] == "Total")
        ]
        self.assertEqual(len(aggregate), 12)
        self.assertTrue(aggregate["balance"].eq(0.0).all())
        self.assertTrue(
            aggregate["proforma_cecl_reserve"].eq(0.0).all()
        )
        selection = result["reports"]["targeted_selection_detail"]
        self.assertFalse(selection["selected"].astype(bool).any())
        selector_matches = selection[selection["selector_matched"]]
        self.assertFalse(selector_matches.empty)
        self.assertTrue(
            selector_matches["selection_reason"].eq("model_excluded").all()
        )

    def test_only_selected_loan_of_multi_loan_borrower_is_shocked(self):
        scenario, base_dir = load_scenario(ROOT / "examples" / "scenario.json")
        scenario["targeted_stress"] = {
            "enabled": True,
            "shocks": {
                "single_cre": {
                    "selector": {
                        "type": "condition",
                        "field": "loan_id",
                        "op": "eq",
                        "value": "L001",
                    },
                    "default_tier": "high",
                    "tiers": {
                        "high": {
                            "modules": {
                                "CRE": {
                                    "dscr_decline": {
                                        "operation": "add",
                                        "values": {"S1": 0.40, "S2": 0.40},
                                    }
                                }
                            }
                        }
                    },
                }
            },
            "variants": {
                "isolated": {
                    "shocks": ["single_cre"],
                    "unmatched_behavior": "base",
                }
            },
        }
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False, run_comparison=False
        )
        rows = result["variant_results"]
        isolated = rows[rows["scenario_variant"] == "isolated"].set_index("loan_id")
        self.assertEqual(isolated.loc["L001", "stressed_bucket_S1"], "Substandard")
        self.assertEqual(isolated.loc["L002", "stressed_bucket_S1"], "Pass")
        detail = result["reports"]["targeted_selection_detail"]
        matched = detail[
            (detail["scenario_variant"] == "isolated") & detail["selected"]
        ]
        self.assertEqual(matched["loan_id"].tolist(), ["L001"])

    def test_ordered_shocks_compose_assumptions(self):
        scenario, base_dir = load_scenario(ROOT / "examples" / "targeted_stress.json")
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False, run_comparison=False
        )
        audit = result["reports"]["targeted_assumption_audit"]
        rows = audit[
            (audit["scenario_variant"] == "oil_and_tariffs")
            & (audit["loan_id"] == "L006")
            & (audit["stress_level"] == "S1")
            & (audit["parameter"] == "ebitda_reduction")
        ]
        self.assertEqual(rows["shock"].tolist(), ["oil_shock", "tariff_shock"])
        self.assertTrue(rows["baseline_value"].eq(0.10).all())
        self.assertAlmostEqual(float(rows.iloc[0]["effective_value"]), 0.18)
        self.assertAlmostEqual(float(rows.iloc[1]["value_before_operation"]), 0.18)
        self.assertAlmostEqual(float(rows.iloc[1]["effective_value"]), 0.25)

    def test_ci_rr_six_point_five_targeted_baseline_uses_distinct_assumption(self):
        scenario = {
            "borrower": {"risk_rating_field": "risk_rating"},
            "modules": {
                "C&I": {
                    "sector_field": "ci_sector",
                    "ebitda_reduction": {
                        "default": {
                            "6": {"S1": 0.10},
                            "6.5": {"S1": 0.165},
                            "7": {"S1": 0.20},
                        }
                    },
                }
            },
        }
        row = {"ci_sector": "Test Sector", "risk_rating": "6.5"}

        self.assertAlmostEqual(
            _baseline_parameter(
                row,
                scenario,
                "C&I",
                "ebitda_reduction",
                "S1",
            ),
            0.165,
        )

    def test_consumer_targeted_parameters_are_applied_per_loan(self):
        scenario, base_dir = load_scenario(ROOT / "examples" / "scenario.json")
        scenario["targeted_stress"] = {
            "shocks": {
                "consumer_shock": {
                    "selector": {
                        "type": "condition",
                        "field": "loan_id",
                        "op": "eq",
                        "value": "L008",
                    },
                    "default_tier": "high",
                    "tiers": {
                        "high": {
                            "modules": {
                                "Consumer": {
                                    "pd_increase_factor": {
                                        "operation": "replace",
                                        "values": {"S1": 2.0, "S2": 2.5},
                                    },
                                    "collateral_value_factor": {
                                        "operation": "replace",
                                        "values": {"S1": 0.7, "S2": 0.6},
                                    },
                                    "rushed_sale_discount": {
                                        "operation": "replace",
                                        "values": {"S1": 0.1, "S2": 0.15},
                                    },
                                    "closing_costs": {
                                        "operation": "replace",
                                        "values": {"S1": 0.03, "S2": 0.04},
                                    },
                                }
                            }
                        }
                    },
                }
            },
            "variants": {
                "consumer_parallel": {
                    "shocks": ["consumer_shock"],
                    "unmatched_behavior": "base",
                }
            },
        }
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False, run_comparison=False
        )
        row = result["variant_results"][
            (result["variant_results"]["scenario_variant"] == "consumer_parallel")
            & (result["variant_results"]["loan_id"] == "L008")
        ].iloc[0]
        self.assertAlmostEqual(
            float(row["consumer_pd_S1"]),
            min(float(row["consumer_pd_unstressed"]) * 2.0, 1.0),
        )
        self.assertLess(
            float(row["consumer_stressed_collateral_value_S1"]),
            float(row["consumer_collateral_value_unstressed"]),
        )
        consumer_summary = result["reports"]["consumer_summary"]
        for variant in ("baseline", "consumer_parallel"):
            variant_summary = consumer_summary[
                consumer_summary["scenario_variant"] == variant
            ].set_index("stress_level")
            self.assertEqual(
                list(variant_summary["proforma_cecl_reserve"]),
                sorted(variant_summary["proforma_cecl_reserve"]),
            )
            for level in ("Base", "S1", "S2"):
                self.assertAlmostEqual(
                    float(variant_summary.at[level, "expected_loss"])
                    + float(
                        variant_summary.at[
                            level, "qualitative_reserve"
                        ]
                    ),
                    float(
                        variant_summary.at[
                            level, "proforma_cecl_reserve"
                        ]
                    ),
                )

        invalid_s2 = copy.deepcopy(scenario)
        invalid_s2["targeted_stress"]["shocks"]["consumer_shock"][
            "tiers"
        ]["high"]["modules"]["Consumer"]["pd_increase_factor"][
            "values"
        ]["S2"] = -1.0
        invalid_result = StressEngine(invalid_s2, base_dir).run(
            write_outputs=False,
            run_comparison=False,
        )
        invalid_row = invalid_result["variant_results"][
            (
                invalid_result["variant_results"]["scenario_variant"]
                == "consumer_parallel"
            )
            & (invalid_result["variant_results"]["loan_id"] == "L008")
        ].iloc[0]
        self.assertTrue(bool(invalid_row["out_of_scope_S2"]))
        invalid_summary = invalid_result["reports"]["consumer_summary"]
        invalid_summary = invalid_summary[
            invalid_summary["scenario_variant"] == "consumer_parallel"
        ].set_index("stress_level")
        self.assertAlmostEqual(
            float(invalid_summary.at["Base", "proforma_cecl_reserve"]),
            4000.0,
        )
        self.assertAlmostEqual(
            float(invalid_summary.at["S1", "proforma_cecl_reserve"]),
            7689.04,
        )
        self.assertAlmostEqual(
            float(invalid_summary.at["S2", "proforma_cecl_reserve"]),
            7689.04,
        )
        self.assertAlmostEqual(
            float(invalid_summary.at["S2", "expected_loss"]),
            3689.04,
        )
        self.assertAlmostEqual(
            float(invalid_summary.at["S2", "qualitative_reserve"]),
            4000.0,
        )
        self.assertEqual(
            float(invalid_summary.at["S2", "in_scope_balance"]),
            0.0,
        )
        self.assertEqual(
            float(invalid_summary.at["S2", "out_of_scope_balance"]),
            300000.0,
        )
        invalid_cecl = invalid_result["reports"]["cecl_summary"]
        invalid_cecl = invalid_cecl[
            (
                invalid_cecl["scenario_variant"]
                == "consumer_parallel"
            )
            & (invalid_cecl["portfolio"] == "Consumer")
            & (invalid_cecl["bucket"] == "Total")
        ].set_index("stress_level")
        self.assertEqual(
            list(invalid_cecl["proforma_cecl_reserve"]),
            sorted(invalid_cecl["proforma_cecl_reserve"]),
        )
        self.assertAlmostEqual(
            float(invalid_cecl.at["S2", "proforma_cecl_reserve"]),
            float(invalid_cecl.at["S1", "proforma_cecl_reserve"]),
        )

    def test_cre_refinance_parameters_and_invalid_effective_value(self):
        scenario, base_dir = load_scenario(ROOT / "examples" / "scenario.json")
        cre_parameters = {
            "refinance_noi_decline": {
                "operation": "replace",
                "values": {"S1": 0.20, "S2": 0.30},
            },
            "treasury_rate": {
                "operation": "replace",
                "values": {"S1": 0.05, "S2": 0.06},
            },
            "credit_spread": {
                "operation": "replace",
                "values": {"S1": 0.03, "S2": 0.04},
            },
            "amortization_years": {
                "operation": "replace",
                "values": {"S1": 15, "S2": 10},
            },
            "cap_rate": {
                "operation": "replace",
                "values": {"S1": 0.10, "S2": 0.12},
            },
        }
        scenario["targeted_stress"] = {
            "primary_variant": "cre_refi",
            "shocks": {
                "refi": {
                    "selector": {
                        "type": "condition",
                        "field": "loan_id",
                        "op": "eq",
                        "value": "L003",
                    },
                    "default_tier": "high",
                    "tiers": {
                        "high": {"modules": {"CRE": cre_parameters}}
                    },
                }
            },
            "variants": {
                "cre_refi": {
                    "shocks": ["refi"],
                    "unmatched_behavior": "baseline_stress",
                }
            },
        }
        result = StressEngine(scenario, base_dir).run(
            write_outputs=False, run_comparison=False
        )
        self.assertEqual(set(result["results"]["scenario_variant"]), {"cre_refi"})
        audit = result["reports"]["targeted_assumption_audit"]
        applied = set(
            audit[
                (audit["scenario_variant"] == "cre_refi")
                & (audit["loan_id"] == "L003")
            ]["parameter"]
        )
        self.assertEqual(applied, set(cre_parameters))
        row = result["results"][result["results"]["loan_id"] == "L003"].iloc[0]
        self.assertTrue(pd.notna(row["cre_refi_dscr_S1"]))
        self.assertTrue(pd.notna(row["cre_ltv_S1"]))

        invalid = copy.deepcopy(scenario)
        invalid["targeted_stress"]["shocks"]["refi"]["tiers"]["high"]["modules"][
            "CRE"
        ]["amortization_years"]["values"] = {"S1": 0, "S2": 0}
        invalid_result = StressEngine(invalid, base_dir).run(
            write_outputs=False, run_comparison=False
        )
        invalid_row = invalid_result["results"][
            invalid_result["results"]["loan_id"] == "L003"
        ].iloc[0]
        self.assertTrue(bool(invalid_row["out_of_scope_S1"]))
        out_scope = invalid_result["reports"]["out_of_scope_detail"]
        self.assertTrue(
            (
                (out_scope["scenario_variant"] == "cre_refi")
                & (out_scope["loan_id"] == "L003")
                & (out_scope["field"] == "amortization_years")
            ).any()
        )

    def test_invalid_naics_prefix_and_parameter_are_rejected(self):
        scenario, _ = load_scenario(ROOT / "examples" / "scenario.json")
        base_target = {
            "shocks": {
                "bad": {
                    "selector": {
                        "type": "naics_prefix",
                        "field": "naics_code",
                        "prefixes": ["2"],
                    },
                    "default_tier": "high",
                    "tiers": {
                        "high": {
                            "modules": {
                                "C&I": {
                                    "ebitda_reduction": {
                                        "operation": "add",
                                        "values": {"S1": 0.1},
                                    }
                                }
                            }
                        }
                    },
                }
            },
            "variants": {
                "bad_variant": {
                    "shocks": ["bad"],
                    "unmatched_behavior": "base",
                }
            },
        }
        bad_naics = copy.deepcopy(scenario)
        bad_naics["targeted_stress"] = base_target
        with self.assertRaisesRegex(ValueError, "2-6 digits"):
            validate_scenario(bad_naics)

        bad_parameter = copy.deepcopy(scenario)
        bad_parameter["targeted_stress"] = copy.deepcopy(base_target)
        bad_parameter["targeted_stress"]["shocks"]["bad"]["selector"]["prefixes"] = [
            "21"
        ]
        modules = bad_parameter["targeted_stress"]["shocks"]["bad"]["tiers"]["high"][
            "modules"
        ]
        modules["C&I"]["unsupported"] = modules["C&I"].pop("ebitda_reduction")
        with self.assertRaisesRegex(ValueError, "Unsupported targeted parameter"):
            validate_scenario(bad_parameter)

    def test_naics_prefix_lengths_and_nested_external_list_selection(self):
        frame = pd.DataFrame(
            {
                "loan_id": ["L2", "L3", "L4", "L5", "L6", "BAD", "MISS"],
                "naics_code": ["21", "211", "2111", "21112", "211120", "abc", None],
            }
        )
        for prefix in ["21", "211", "2111", "21112", "211120"]:
            mask, _ = _evaluate_selector(
                frame,
                {
                    "type": "naics_prefix",
                    "field": "naics_code",
                    "prefixes": [prefix],
                },
                {},
                [],
                "length_test",
            )
            self.assertTrue(mask.iloc[len(prefix) - 2])
            self.assertFalse(mask.iloc[5])
            self.assertFalse(mask.iloc[6])

        loaded = {
            "partner": SimpleNamespace(
                frame=pd.DataFrame(
                    {"loan_id": ["L3", "L6"], "impact_tier": ["medium", "high"]}
                )
            )
        }
        mask, tiers = _evaluate_selector(
            frame,
            {
                "all": [
                    {
                        "type": "naics_prefix",
                        "field": "naics_code",
                        "prefixes": ["21"],
                    },
                    {
                        "type": "external_list",
                        "source": "partner",
                        "source_field": "loan_id",
                        "exposure_field": "loan_id",
                        "tier_field": "impact_tier",
                    },
                ]
            },
            loaded,
            [],
            "combined",
        )
        self.assertEqual(frame.loc[mask, "loan_id"].tolist(), ["L3", "L6"])
        self.assertEqual(tiers.loc[mask].tolist(), ["medium", "high"])

    def test_external_list_conflicting_tiers_are_rejected(self):
        frame = pd.DataFrame({"loan_id": ["L1"]})
        loaded = {
            "partner": SimpleNamespace(
                frame=pd.DataFrame(
                    {"loan_id": ["L1", "L1"], "impact_tier": ["low", "high"]}
                )
            )
        }
        with self.assertRaisesRegex(ValueError, "conflicting tiers"):
            _evaluate_selector(
                frame,
                {
                    "type": "external_list",
                    "source": "partner",
                    "source_field": "loan_id",
                    "exposure_field": "loan_id",
                    "tier_field": "impact_tier",
                },
                loaded,
                [],
                "conflict",
            )

    def test_missing_fields_in_nested_selectors_and_external_where_are_rejected(self):
        frame = pd.DataFrame({"loan_id": ["L1"], "status": ["active"]})
        nested_selector = {
            "all": [
                {
                    "type": "condition",
                    "field": "status",
                    "op": "eq",
                    "value": "active",
                },
                {
                    "any": [
                        {
                            "type": "condition",
                            "field": "missing_region",
                            "op": "eq",
                            "value": "US",
                        }
                    ]
                },
            ]
        }
        with self.assertRaisesRegex(ValueError, "missing_region"):
            _evaluate_selector(frame, nested_selector, {}, [], "nested")

        loaded = {
            "partner": SimpleNamespace(
                frame=pd.DataFrame({"loan_id": ["L1"], "region": ["US"]})
            )
        }
        with self.assertRaisesRegex(ValueError, "missing_country"):
            _evaluate_selector(
                frame,
                {
                    "type": "external_list",
                    "source": "partner",
                    "source_field": "loan_id",
                    "exposure_field": "loan_id",
                    "where": {
                        "all": [
                            {
                                "field": "region",
                                "op": "eq",
                                "value": "US",
                            },
                            {
                                "any": [
                                    {
                                        "field": "missing_country",
                                        "op": "eq",
                                        "value": "US",
                                    }
                                ]
                            },
                        ]
                    },
                },
                loaded,
                [],
                "external_where",
            )

    def test_condition_selectors_require_operands_and_valid_between_bounds(self):
        scenario = _targeted_validation_scenario({"S1": 0.1, "S2": 0.2})
        selector = scenario["targeted_stress"]["shocks"]["shock"]["selector"]
        selector.pop("value")
        selector["op"] = "gt"
        with self.assertRaisesRegex(ValueError, "requires value"):
            validate_targeted_config(scenario)

        invalid_operands = [
            ("gt", "not-a-number", "numeric and finite"),
            ("gt", None, "requires a non-null value"),
            ("between", [0, "not-a-number"], "numeric and finite"),
            ("in", [], "requires nonempty values"),
        ]
        for operation, value, message in invalid_operands:
            with self.subTest(operation=operation, value=value):
                scenario = _targeted_validation_scenario(
                    {"S1": 0.1, "S2": 0.2}
                )
                selector = scenario["targeted_stress"]["shocks"]["shock"][
                    "selector"
                ]
                selector["op"] = operation
                selector["value"] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_targeted_config(scenario)

        scenario = _targeted_validation_scenario({"S1": 0.1, "S2": 0.2})
        scenario["targeted_stress"]["shocks"]["shock"]["selector"] = {
            "type": "external_list",
            "source": "partner",
            "source_field": "loan_id",
            "exposure_field": "loan_id",
            "where": {
                "field": "balance",
                "op": "between",
                "value": [100],
            },
        }
        with self.assertRaisesRegex(ValueError, "requires exactly two values"):
            validate_targeted_config(scenario)

    def test_targeted_tiers_and_parameter_maps_must_be_nonempty(self):
        scenario = _targeted_validation_scenario({"S1": 0.1, "S2": 0.2})
        tier = scenario["targeted_stress"]["shocks"]["shock"]["tiers"]["high"]
        tier["modules"] = {}
        with self.assertRaisesRegex(ValueError, "modules must be a nonempty object"):
            validate_targeted_config(scenario)

        scenario = _targeted_validation_scenario({"S1": 0.1, "S2": 0.2})
        parameters = scenario["targeted_stress"]["shocks"]["shock"]["tiers"]["high"][
            "modules"
        ]
        parameters["C&I"] = {}
        with self.assertRaisesRegex(ValueError, "parameters must be a nonempty object"):
            validate_targeted_config(scenario)

    def test_targeted_value_maps_require_resolved_levels_and_finite_numbers(self):
        cases = [
            ({"S1": 0.1}, "missing stress levels"),
            (
                {"S1": 0.1, "S2": 0.2, "S3": 0.3},
                "unknown stress levels",
            ),
            ("not-a-number", "numeric and finite"),
            ({"S1": np.inf, "S2": 0.2}, "numeric and finite"),
            ({"S1": np.nan, "S2": 0.2}, "numeric and finite"),
        ]
        for values, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    validate_targeted_config(_targeted_validation_scenario(values))

    def test_targeted_value_maps_accept_one_finite_wildcard(self):
        for wildcard in ("default", "all", "*"):
            with self.subTest(wildcard=wildcard):
                validate_targeted_config(
                    _targeted_validation_scenario({wildcard: 0.1})
                )

        with self.assertRaisesRegex(ValueError, "at most one"):
            validate_targeted_config(
                _targeted_validation_scenario({"default": 0.1, "*": 0.2})
            )

    def test_nonfinite_composed_targeted_value_is_a_runtime_error(self):
        scenario = _targeted_validation_scenario(2.0)
        scenario["stress_levels"] = ["S1"]
        parameter_spec = scenario["targeted_stress"]["shocks"]["shock"]["tiers"][
            "high"
        ]["modules"]["C&I"].pop("ebitda_reduction")
        parameter_spec["operation"] = "multiply"
        scenario["targeted_stress"]["shocks"]["shock"]["tiers"]["high"]["modules"][
            "C&I"
        ]["interest_rate_stress"] = parameter_spec
        scenario["borrower"] = {
            "borrower_id_field": "borrower_id",
            "loan_id_field": "loan_id",
            "balance_field": "balance",
            "risk_rating_field": "risk_rating",
        }
        scenario["modules"] = {
            "C&I": {
                "sector_field": "ci_sector",
                "interest_rate_stress": {"S1": 1e308},
            }
        }
        context = pd.DataFrame(
            [
                {
                    "_exposure_id": "L1",
                    "loan_id": "L1",
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "primary_module": "C&I",
                    "ci_sector": "Test",
                    "risk_rating": 6,
                }
            ]
        )
        validate_targeted_config(scenario)
        with self.assertRaisesRegex(ValueError, "non-finite effective value"):
            _resolve_variant(
                context,
                scenario,
                {},
                "variant",
                scenario["targeted_stress"]["variants"]["variant"],
                [],
                [],
                [],
            )

    def test_baseline_variant_name_is_reserved_case_insensitively(self):
        scenario = _targeted_validation_scenario({"S1": 0.1, "S2": 0.2})
        scenario["targeted_stress"]["variants"] = {
            "BaSeLiNe": {
                "shocks": ["shock"],
                "unmatched_behavior": "base",
            }
        }
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_targeted_config(scenario)

    def test_batch_and_comparison_keys_include_variant(self):
        previous = pd.DataFrame(
            [
                {
                    "scenario_variant": variant,
                    "portfolio": "Aggregate",
                    "stress_level": "S1",
                    "bucket": "Total",
                    "proforma_cecl_reserve": reserve,
                    "proforma_cecl_ratio": reserve / 1000,
                    "cecl_reserve_status": "available",
                }
                for variant, reserve in [("baseline", 100.0), ("oil", 120.0)]
            ]
        )
        changed = previous.copy()
        changed.loc[changed["scenario_variant"] == "oil", "proforma_cecl_reserve"] = 130.0
        rows = _cecl_impact_rows(
            "prior.json",
            "scenario_variable",
            "targeted_stress",
            "old",
            "new",
            previous,
            changed,
        )
        reserve_rows = [
            row for row in rows if row["metric"] == "proforma_cecl_reserve"
        ]
        self.assertEqual(len(reserve_rows), 1)
        self.assertEqual(reserve_rows[0]["scenario_variant"], "oil")
        self.assertEqual(reserve_rows[0]["marginal_impact"], 10.0)

        lookup = _base_cecl_lookup({"reports": {"cecl_summary": previous}})
        self.assertEqual(lookup[("baseline", "S1")]["proforma_cecl_reserve"], 100.0)
        self.assertEqual(lookup[("oil", "S1")]["proforma_cecl_reserve"], 120.0)


if __name__ == "__main__":
    unittest.main()
