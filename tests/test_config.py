from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stress_engine.borrower import build_borrowers, build_source_reconciliation, enrich_borrowers
from stress_engine.config import load_scenario, validate_scenario
from stress_engine.io import load_inputs, read_table
from stress_engine.tagging import apply_tags, assign_primary_modules


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"


class ScenarioConfigTest(unittest.TestCase):
    @staticmethod
    def _minimal_scenario() -> dict:
        return {
            "inputs": {"identity": {}},
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "balance",
            },
            "tags": {},
            "modules": {},
        }

    def test_example_manifest_loads_all_fragments(self):
        scenario, base_dir = load_scenario(SCENARIO)

        self.assertEqual(base_dir, SCENARIO.parent)
        self.assertEqual(scenario["scenario_id"], "example_2026q2")
        self.assertEqual(set(scenario["modules"]), {"CRE", "C&I", "Consumer"})
        self.assertEqual(len(scenario["tags"]), 18)
        self.assertNotIn("$include", scenario)
        self.assertEqual(len(scenario["_metadata"]["scenario_files"]), 9)

        shipped_parent = dict(
            scenario["tags"]["CI_Sector_Sponsor_and_Specialty"]
        )
        shipped_parent.pop("tie_out")
        shipped_arr, _ = apply_tags(
            pd.DataFrame(
                [
                    {
                        "subsector": "Sponsor and Specialty;ARR",
                        "tag_hint": pd.NA,
                        "tag_hint_2": pd.NA,
                        "balance": 100.0,
                    }
                ]
            ),
            {
                "borrower": {"balance_field": "balance"},
                "tags": {
                    "CI_Sector_Sponsor_and_Specialty": shipped_parent,
                    "ARR": scenario["tags"]["ARR"],
                },
            },
            {},
            [],
        )
        self.assertTrue(
            bool(
                shipped_arr.at[
                    0, "tag_ci_sector_sponsor_and_specialty"
                ]
            )
        )
        self.assertTrue(bool(shipped_arr.at[0, "tag_arr"]))
        self.assertTrue(bool(shipped_arr.at[0, "model_excluded"]))
        self.assertEqual(
            shipped_arr.at[0, "ci_sector"], "Sponsor and Specialty"
        )

    def test_arr_tag_is_nested_under_sponsor_and_vetoes_model_routing(self):
        scenario = {
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "balance",
                "module_field": "model_module",
                "portfolio_field": "model_portfolio",
            },
            "cecl": {"portfolio_field": "cecl_portfolio"},
            "tags": {
                "CI_Model": {
                    "model_eligible": True,
                    "include": {
                        "field": "subsector",
                        "op": "has_token",
                        "value": "Sponsor and Specialty",
                    },
                    "assign": {
                        "model_module": "C&I",
                        "model_portfolio": "C&I",
                    },
                },
                "CI_Sector_Sponsor_and_Specialty": {
                    "model_eligible": False,
                    "include": {
                        "field": "subsector",
                        "op": "has_token",
                        "value": "Sponsor and Specialty",
                    },
                    "assign": {"ci_sector": "Sponsor and Specialty"},
                },
                "ARR": {
                    "model_eligible": False,
                    "exclude_from_model": True,
                    "include": {
                        "all": [
                            {
                                "field": "subsector",
                                "op": "has_token",
                                "value": "Sponsor and Specialty",
                            },
                            {
                                "any": [
                                    {
                                        "field": "subsector",
                                        "op": "has_token",
                                        "value": "ARR",
                                    },
                                    {
                                        "field": "tag_hint",
                                        "op": "has_token",
                                        "value": "ARR",
                                    },
                                    {
                                        "field": "tag_hint_2",
                                        "op": "has_token",
                                        "value": "ARR",
                                    },
                                ]
                            },
                        ]
                    },
                },
            },
            "modules": {
                "C&I": {
                    "enabled": True,
                    "eligible_tags": ["CI_Model"],
                    "cecl_portfolio_field": "ci_sector",
                }
            },
            "module_order": ["C&I"],
        }
        borrowers = pd.DataFrame(
            [
                {
                    "borrower_id": "B-ARR",
                    "subsector": "Sponsor and Specialty",
                    "tag_hint": pd.NA,
                    "tag_hint_2": "ARR",
                    "balance": 300.0,
                },
                {
                    "borrower_id": "B-SPONSOR",
                    "subsector": "Sponsor and Specialty",
                    "tag_hint": pd.NA,
                    "tag_hint_2": pd.NA,
                    "balance": 200.0,
                },
                {
                    "borrower_id": "B-NONSPONSOR",
                    "subsector": "Middle Market",
                    "tag_hint": "ARR",
                    "tag_hint_2": pd.NA,
                    "balance": 100.0,
                },
            ]
        )

        tagged, summary = apply_tags(borrowers, scenario, {}, [])
        assigned = assign_primary_modules(tagged, scenario)
        arr = assigned.loc[assigned["borrower_id"] == "B-ARR"].iloc[0]
        sponsor = assigned.loc[
            assigned["borrower_id"] == "B-SPONSOR"
        ].iloc[0]
        nonsponsor = assigned.loc[
            assigned["borrower_id"] == "B-NONSPONSOR"
        ].iloc[0]

        self.assertTrue(bool(arr["tag_arr"]))
        self.assertTrue(bool(arr["tag_ci_sector_sponsor_and_specialty"]))
        self.assertEqual(arr["ci_sector"], "Sponsor and Specialty")
        self.assertEqual(
            set(arr["all_tags"].split(";")),
            {"CI_Model", "CI_Sector_Sponsor_and_Specialty", "ARR"},
        )
        self.assertTrue(bool(arr["model_excluded"]))
        self.assertEqual(arr["model_exclusion_tags"], "ARR")
        self.assertEqual(arr["model_tags"], "")
        self.assertEqual(arr["eligible_modules"], "")
        self.assertTrue(pd.isna(arr["primary_module"]))
        self.assertTrue(pd.isna(arr["model_module"]))
        self.assertTrue(pd.isna(arr["model_portfolio"]))
        self.assertTrue(pd.isna(arr["cecl_portfolio"]))

        self.assertFalse(bool(sponsor["model_excluded"]))
        self.assertEqual(sponsor["primary_module"], "C&I")
        self.assertEqual(sponsor["cecl_portfolio"], "Sponsor and Specialty")
        self.assertFalse(bool(nonsponsor["tag_arr"]))
        self.assertFalse(bool(nonsponsor["model_excluded"]))

        populations = summary[summary["tie_out_name"].isna()].set_index("tag")
        self.assertEqual(int(populations.at["ARR", "borrower_count"]), 1)
        self.assertEqual(float(populations.at["ARR", "balance"]), 300.0)
        self.assertTrue(bool(populations.at["ARR", "exclude_from_model"]))
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
            int(populations.at["ARR", "model_excluded_borrower_count"]), 1
        )
        self.assertEqual(
            float(populations.at["ARR", "model_excluded_balance"]), 300.0
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
            500.0,
        )
        self.assertEqual(
            float(
                populations.at[
                    "CI_Sector_Sponsor_and_Specialty",
                    "not_model_excluded_balance",
                ]
            ),
            200.0,
        )
        self.assertEqual(
            float(
                populations.at[
                    "CI_Sector_Sponsor_and_Specialty",
                    "model_excluded_balance",
                ]
            ),
            300.0,
        )

    def test_model_exclusion_tag_flags_require_json_booleans(self):
        for spec, message in (
            (
                {"exclude_from_model": "true"},
                "exclude_from_model must be a JSON boolean",
            ),
            (
                {"exclude_from_model": True, "model_eligible": True},
                "cannot be both model_eligible and exclude_from_model",
            ),
            (
                {"exclude_from_model": True, "model_eligible": False},
                "must define a nonempty include condition",
            ),
            (
                {
                    "exclude_from_model": True,
                    "model_eligible": False,
                    "include": {"field": "sentinel", "all": []},
                },
                "logical conditions must contain exactly one",
            ),
            (
                {
                    "exclude_from_model": True,
                    "model_eligible": False,
                    "include": {
                        "field": "sentinel",
                        "op": "not_in",
                        "value": [],
                    },
                },
                "requires nonempty values",
            ),
            (
                {
                    "exclude_from_model": True,
                    "model_eligible": False,
                    "include": {
                        "field": "sentinel",
                        "op": "in",
                        "value": [["nested"]],
                    },
                },
                "requires scalar values",
            ),
            (
                {
                    "exclude_from_model": True,
                    "model_eligible": False,
                    "include": {
                        "field": "sentinel",
                        "op": "gt",
                        "value": "not-a-number",
                    },
                },
                "requires a finite numeric value",
            ),
            (
                {
                    "exclude_from_model": True,
                    "model_eligible": False,
                    "include": {
                        "field": "sentinel",
                        "op": "regex",
                        "value": "[",
                    },
                },
                "contains an invalid regex",
            ),
        ):
            with self.subTest(spec=spec):
                scenario = self._minimal_scenario()
                scenario["tags"] = {"ARR": spec}
                with self.assertRaisesRegex(ValueError, message):
                    validate_scenario(scenario)

    def test_model_exclusion_tag_fails_closed_on_missing_input_field(self):
        scenario = self._minimal_scenario()
        scenario["tags"] = {
            "ARR": {
                "model_eligible": False,
                "exclude_from_model": True,
                "include": {
                    "field": "arr_indicator",
                    "op": "eq",
                    "value": True,
                },
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "Model-exclusion tag 'ARR' references missing condition fields",
        ):
            apply_tags(
                pd.DataFrame(
                    [{"borrower_id": "B1", "balance": 100.0}]
                ),
                scenario,
                {},
                [],
            )

    def test_non_model_eligible_tag_cannot_assign_stress_module(self):
        for assignments, field in (
            ({"model_module": "C&I"}, "model_module"),
            ({"model_portfolio": "C&I"}, "model_portfolio"),
            ({"cecl_portfolio": "Sponsor and Specialty"}, "cecl_portfolio"),
        ):
            with self.subTest(assignments=assignments):
                scenario = self._minimal_scenario()
                scenario["tags"] = {
                    "Audit Only": {
                        "model_eligible": False,
                        "include": {
                            "field": "category",
                            "op": "eq",
                            "value": "audit",
                        },
                        "assign": assignments,
                    }
                }

                with self.assertRaisesRegex(
                    ValueError,
                    f"cannot assign modeled routing fields: {field}",
                ):
                    validate_scenario(scenario)

    def test_borrower_and_cecl_config_are_consolidated(self):
        manifest = json.loads(SCENARIO.read_text(encoding="utf-8"))
        inputs_fragment = json.loads(
            (SCENARIO.parent / "scenario" / "inputs.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertIn("borrower", inputs_fragment)
        self.assertIn("cecl", manifest)
        self.assertNotIn("scenario/borrower.json", manifest["$include"])
        self.assertNotIn("scenario/cecl.json", manifest["$include"])
        self.assertFalse((SCENARIO.parent / "scenario" / "borrower.json").exists())
        self.assertFalse((SCENARIO.parent / "scenario" / "cecl.json").exists())

        scenario, _ = load_scenario(SCENARIO)
        self.assertEqual(scenario["borrower"], inputs_fragment["borrower"])
        self.assertEqual(scenario["cecl"], manifest["cecl"])

    def test_migration_cutoffs_live_only_in_master_manifest(self):
        manifest = json.loads(SCENARIO.read_text(encoding="utf-8"))
        cre_fragment = json.loads(
            (SCENARIO.parent / "scenario" / "modules" / "cre.json").read_text(
                encoding="utf-8"
            )
        )
        ci_fragment = json.loads(
            (SCENARIO.parent / "scenario" / "modules" / "ci.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            "dscr": {"special_mention": 1.15, "substandard": 1.0},
            "fccr": {"special_mention": 1.15, "substandard": 1.0},
            "ltv": {"special_mention": 0.75, "substandard": 0.9},
        }

        self.assertEqual(manifest["cutoffs"], expected)
        self.assertNotIn("cutoffs", ci_fragment["modules"]["C&I"])
        for test_config in cre_fragment["modules"]["CRE"]["tests"].values():
            self.assertNotIn("cutoffs", test_config)

        scenario, _ = load_scenario(SCENARIO)
        self.assertEqual(scenario["cutoffs"], expected)

    def test_nested_commercial_cutoffs_are_rejected(self):
        scenario = {
            "inputs": {"identity": {}},
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "balance",
            },
            "tags": {},
            "cutoffs": {
                "fccr": {"special_mention": 1.15, "substandard": 1.0}
            },
            "modules": {
                "C&I": {
                    "cutoffs": {
                        "special_mention": 1.15,
                        "substandard": 1.0,
                    }
                }
            },
        }

        with self.assertRaisesRegex(
            ValueError, r"remove: modules\.C&I\.cutoffs"
        ):
            validate_scenario(scenario)

    def test_omitted_module_order_retains_default_behavior(self):
        scenario = self._minimal_scenario()
        scenario["modules"] = {
            "Consumer": {"enabled": True, "eligible_tags": ["Consumer Eligible"]},
            "CRE": {"enabled": True, "eligible_tags": ["CRE Eligible"]},
        }
        scenario["tags"] = {
            "Consumer Eligible": {},
            "CRE Eligible": {},
        }
        scenario["run"] = {"cutoff_date": "2026-01-01"}
        scenario["cutoffs"] = {
            "dscr": {"special_mention": 1.15, "substandard": 1.0},
            "ltv": {"special_mention": 0.75, "substandard": 0.9},
        }

        validate_scenario(scenario)
        assigned = assign_primary_modules(
            pd.DataFrame(
                [
                    {
                        "borrower_id": "B1",
                        "tag_consumer_eligible": True,
                        "tag_cre_eligible": True,
                    }
                ]
            ),
            scenario,
        )

        self.assertEqual(assigned.at[0, "primary_module"], "CRE")

    def test_disabled_module_cannot_take_primary_priority(self):
        scenario = self._minimal_scenario()
        scenario["modules"] = {
            "CRE": {"enabled": False, "eligible_tags": ["Shared Eligible"]},
            "Consumer": {"enabled": True, "eligible_tags": ["Shared Eligible"]},
        }
        scenario["tags"] = {"Shared Eligible": {}}

        validate_scenario(scenario)
        assigned = assign_primary_modules(
            pd.DataFrame(
                [{"borrower_id": "B1", "tag_shared_eligible": True}]
            ),
            scenario,
        )

        self.assertEqual(assigned.at[0, "primary_module"], "Consumer")
        self.assertEqual(assigned.at[0, "eligible_modules"], "Consumer")

    def test_non_model_eligible_tag_cannot_route_a_module(self):
        scenario = self._minimal_scenario()
        scenario["tags"] = {
            "Audit Only": {
                "model_eligible": False,
                "include": {
                    "field": "category",
                    "op": "eq",
                    "value": "audit",
                },
            }
        }
        scenario["modules"] = {
            "C&I": {
                "enabled": True,
                "eligible_tags": ["Audit Only"],
            }
        }
        scenario["module_order"] = ["C&I"]

        assigned = assign_primary_modules(
            pd.DataFrame(
                [
                    {
                        "borrower_id": "B1",
                        "tag_audit_only": True,
                    }
                ]
            ),
            scenario,
        )

        self.assertEqual(assigned.at[0, "eligible_modules"], "")
        self.assertTrue(pd.isna(assigned.at[0, "primary_module"]))

    def test_module_order_must_be_a_nonempty_list(self):
        for module_order in (None, [], "Consumer"):
            with self.subTest(module_order=module_order):
                scenario = self._minimal_scenario()
                scenario["module_order"] = module_order

                with self.assertRaisesRegex(
                    ValueError, "module_order must be a nonempty JSON list"
                ):
                    validate_scenario(scenario)

    def test_module_order_rejects_unsupported_and_duplicate_names(self):
        invalid_orders = [
            (["Consumer", "Unknown"], "contains unsupported modules"),
            (["Consumer", "Consumer"], "must contain unique module names"),
        ]
        for module_order, message in invalid_orders:
            with self.subTest(module_order=module_order):
                scenario = self._minimal_scenario()
                scenario["module_order"] = module_order

                with self.assertRaisesRegex(ValueError, message):
                    validate_scenario(scenario)

    def test_module_order_cannot_skip_an_enabled_configured_module(self):
        scenario = self._minimal_scenario()
        scenario["modules"]["Consumer"] = {"enabled": True}
        scenario["module_order"] = ["CRE"]

        with self.assertRaisesRegex(
            ValueError,
            "include every enabled configured module exactly once; missing: Consumer",
        ):
            validate_scenario(scenario)

    def test_module_order_may_omit_a_disabled_configured_module(self):
        scenario = self._minimal_scenario()
        scenario["modules"]["Consumer"] = {"enabled": False}
        scenario["module_order"] = ["CRE"]

        validate_scenario(scenario)

    def test_module_configuration_rejects_unknown_and_malformed_modules(self):
        cases = [
            (
                {"TypoModule": {"enabled": True}},
                "unsupported module configurations",
            ),
            ({"Consumer": True}, "configured module must be a JSON object"),
            ({"Consumer": {}}, "configured module must be a nonempty JSON object"),
            (
                {"Consumer": {"enabled": "false"}},
                "'enabled' values must be JSON booleans",
            ),
        ]
        for modules, message in cases:
            with self.subTest(modules=modules):
                scenario = self._minimal_scenario()
                scenario["modules"] = modules

                with self.assertRaisesRegex(ValueError, message):
                    validate_scenario(scenario)

        scenario = self._minimal_scenario()
        scenario["modules"] = []
        with self.assertRaisesRegex(ValueError, "modules must be a JSON object"):
            validate_scenario(scenario)

    def test_input_module_fallback_requires_an_enabled_configured_module(self):
        scenario = self._minimal_scenario()
        scenario["modules"] = {"Consumer": {"enabled": False}}
        validate_scenario(scenario)

        for module_name in ("Consumer", "Consmer"):
            with self.subTest(module_name=module_name):
                with self.assertRaisesRegex(
                    ValueError, "not enabled and configured"
                ):
                    assign_primary_modules(
                        pd.DataFrame(
                            [
                                {
                                    "borrower_id": "B1",
                                    "model_module": module_name,
                                }
                            ]
                        ),
                        scenario,
                    )

    def test_preexisting_primary_module_is_recomputed(self):
        scenario = self._minimal_scenario()
        assigned = assign_primary_modules(
            pd.DataFrame(
                [
                    {
                        "borrower_id": "B1",
                        "primary_module": "Consmer",
                        "model_module": pd.NA,
                    }
                ]
            ),
            scenario,
        )

        self.assertTrue(pd.isna(assigned.at[0, "primary_module"]))

    def test_overlay_routing_requires_an_enabled_executable_portfolio(self):
        scenario = self._minimal_scenario()
        scenario["overlays"] = {
            "EF": {
                "enabled": True,
                "sources": [{"name": "C&I", "weight": 1.0}],
            }
        }
        validate_scenario(scenario)

        valid = assign_primary_modules(
            pd.DataFrame(
                [
                    {
                        "borrower_id": "B1",
                        "model_module": "Overlay",
                        "model_portfolio": "EF",
                    }
                ]
            ),
            scenario,
        )
        self.assertEqual(valid.at[0, "primary_module"], "Overlay")

        with self.assertRaisesRegex(ValueError, "not enabled and configured"):
            assign_primary_modules(
                pd.DataFrame(
                    [
                        {
                            "borrower_id": "B2",
                            "model_module": "Overlay",
                            "model_portfolio": "EFTypo",
                        }
                    ]
                ),
                scenario,
            )

        invalid = self._minimal_scenario()
        invalid["overlays"] = {"EF": {"enabled": True}}
        with self.assertRaisesRegex(ValueError, "nonempty sources list"):
            validate_scenario(invalid)

        invalid = self._minimal_scenario()
        invalid["overlays"] = {
            "EF": {
                "enabled": True,
                "sources": [{"name": "C&I", "weight": 0}],
            }
        }
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            validate_scenario(invalid)

    def test_example_aliases_import_the_intended_columns(self):
        scenario, base_dir = load_scenario(SCENARIO)
        specs = {"identity": scenario["inputs"]["identity"], **scenario["inputs"]["sources"]}

        for name, spec in specs.items():
            aliases = spec.get("column_aliases", {})
            relative_paths = [spec["path"]] if "path" in spec else spec["paths"]
            for relative_path in relative_paths:
                raw_columns = set(pd.read_csv(base_dir / relative_path, nrows=0).columns)
                self.assertTrue(
                    set(aliases.values()).issubset(raw_columns),
                    msg=f"{name}:{relative_path} must contain every configured source column",
                )

            loaded = read_table(name, spec, base_dir)
            canonical_columns = {
                column for column in loaded.frame.columns if not str(column).startswith("_source_")
            }
            self.assertEqual(set(aliases), canonical_columns)

    def test_manifest_values_override_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base = {
                "scenario_id": "base",
                "stress_levels": ["S1"],
                "inputs": {"identity": {}},
                "borrower": {"borrower_id_field": "id", "balance_field": "balance"},
                "tags": {},
                "modules": {},
            }
            (directory / "base.json").write_text(json.dumps(base), encoding="utf-8")
            manifest = {"$include": "base.json", "scenario_id": "override"}
            (directory / "scenario.json").write_text(json.dumps(manifest), encoding="utf-8")

            scenario, _ = load_scenario(directory / "scenario.json")

            self.assertEqual(scenario["scenario_id"], "override")

    def test_include_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "a.json").write_text(json.dumps({"$include": "b.json"}), encoding="utf-8")
            (directory / "b.json").write_text(json.dumps({"$include": "a.json"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "include cycle"):
                load_scenario(directory / "a.json")

    def test_source_column_aliases_preserve_canonical_engine_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "source.csv").write_text(
                "Borrower Number,Current DSCR,Unused Vendor Note\n001,1.25,ignore me\n",
                encoding="utf-8",
            )
            loaded = read_table(
                "aliased",
                {
                    "path": "source.csv",
                    "type": "csv",
                    "column_aliases": {
                        "borrower_id": "Borrower Number",
                        "current_dscr": "Current DSCR",
                    },
                    "string_columns": ["borrower_id"],
                    "numeric_columns": ["current_dscr"],
                    "required_columns": ["borrower_id", "current_dscr"],
                },
                directory,
            )

            self.assertEqual(loaded.frame.at[0, "borrower_id"], "001")
            self.assertEqual(float(loaded.frame.at[0, "current_dscr"]), 1.25)
            self.assertNotIn("Borrower Number", loaded.frame.columns)
            self.assertNotIn("Unused Vendor Note", loaded.frame.columns)
            self.assertNotIn("Unused Vendor Note", loaded.profile["field"].tolist())

    def test_multi_file_input_source_concatenates_in_listed_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "first.csv").write_text(
                "Borrower Number,Score,First File Only\n001,680,ignore\n",
                encoding="utf-8",
            )
            (directory / "second.csv").write_text(
                "Borrower Number,Score,Second File Only\n002,720,ignore\n",
                encoding="utf-8",
            )

            loaded = read_table(
                "consumer",
                {
                    "paths": ["first.csv", "second.csv"],
                    "column_aliases": {"borrower_id": "Borrower Number", "fico_score": "Score"},
                    "string_columns": ["borrower_id"],
                    "numeric_columns": ["fico_score"],
                },
                directory,
            )

            self.assertEqual(loaded.frame["borrower_id"].tolist(), ["001", "002"])
            self.assertEqual([path.name for path in loaded.paths], ["first.csv", "second.csv"])
            self.assertEqual(loaded.file_row_counts, [1, 1])
            self.assertEqual(loaded.frame["_source_file_row"].tolist(), [1, 1])
            self.assertNotIn("First File Only", loaded.frame.columns)
            self.assertNotIn("Second File Only", loaded.frame.columns)

    def test_unused_columns_are_ignored_across_identity_and_source_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "identity.csv").write_text(
                "Account Number,Customer Number,Balance,Unused Identity\n"
                "A001,B001,100,ignore\n",
                encoding="utf-8",
            )
            (directory / "borrower_source.csv").write_text(
                "Customer Number,Metric,outstanding_balance\n"
                "B001,1.5,999\n",
                encoding="utf-8",
            )
            (directory / "account_source.csv").write_text(
                "Account Number,FICO,borrower_id,Other Vendor Field\n"
                "A001,700,WRONG,ignore\n",
                encoding="utf-8",
            )
            scenario = {
                "inputs": {
                    "identity": {
                        "path": "identity.csv",
                        "column_aliases": {
                            "loan_id": "Account Number",
                            "borrower_id": "Customer Number",
                            "outstanding_balance": "Balance",
                        },
                        "numeric_columns": ["outstanding_balance"],
                        "required_columns": [
                            "loan_id",
                            "borrower_id",
                            "outstanding_balance",
                        ],
                    },
                    "sources": {
                        "borrower_metrics": {
                            "path": "borrower_source.csv",
                            "key": "borrower_id",
                            "column_aliases": {
                                "borrower_id": "Customer Number",
                                "metric": "Metric",
                            },
                            "numeric_columns": ["metric"],
                        },
                        "consumer": {
                            "path": "account_source.csv",
                            "key": "loan_id",
                            "identity_key": "loan_id",
                            "column_aliases": {
                                "loan_id": "Account Number",
                                "fico_score": "FICO",
                            },
                            "numeric_columns": ["fico_score"],
                        },
                    },
                },
                "borrower": {
                    "borrower_id_field": "borrower_id",
                    "loan_id_field": "loan_id",
                    "balance_field": "outstanding_balance",
                    "sum_fields": ["outstanding_balance"],
                },
            }

            loaded = load_inputs(scenario, directory)
            exceptions = []
            borrowers = build_borrowers(loaded["identity"].frame, scenario, exceptions)
            enriched = enrich_borrowers(borrowers, loaded, scenario)
            build_source_reconciliation(enriched, loaded, scenario, exceptions)

            self.assertEqual(float(enriched.at[0, "outstanding_balance"]), 100.0)
            self.assertEqual(float(enriched.at[0, "metric"]), 1.5)
            self.assertEqual(float(enriched.at[0, "fico_score"]), 700.0)
            for unused in (
                "Unused Identity",
                "outstanding_balance",
                "borrower_id",
                "Other Vendor Field",
            ):
                if unused == "outstanding_balance":
                    self.assertNotIn(unused, loaded["borrower_metrics"].frame.columns)
                elif unused == "borrower_id":
                    self.assertNotIn(unused, loaded["consumer"].frame.columns)
                else:
                    self.assertFalse(
                        any(unused in item.frame.columns for item in loaded.values())
                    )
            self.assertEqual(exceptions, [])

    def test_multi_file_input_rejects_duplicate_resolved_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "source.csv").write_text("Borrower Number\n001\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "same file more than once"):
                read_table(
                    "consumer",
                    {
                        "paths": ["source.csv", "./source.csv"],
                        "column_aliases": {"borrower_id": "Borrower Number"},
                    },
                    directory,
                )

    def test_multi_file_best_available_conflict_is_reconciled(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            header = "Borrower Number,Current Score,Score Date\n"
            (directory / "first.csv").write_text(header + "001,680,2026-06-30\n", encoding="utf-8")
            (directory / "second.csv").write_text(header + "001,690,2026-06-30\n", encoding="utf-8")
            source_spec = {
                "paths": ["first.csv", "second.csv"],
                "key": "borrower_id",
                "column_aliases": {
                    "borrower_id": "Borrower Number",
                    "current_fico_score": "Current Score",
                    "current_fico_date": "Score Date",
                },
                "string_columns": ["borrower_id"],
                "date_columns": ["current_fico_date"],
                "numeric_columns": ["current_fico_score"],
                "aggregation": {
                    "fico_score": {
                        "method": "best_available",
                        "candidates": [
                            {"field": "current_fico_score", "date_field": "current_fico_date"}
                        ],
                    }
                },
            }
            loaded = read_table("consumer", source_spec, directory)
            exceptions = []
            reconciliation = build_source_reconciliation(
                pd.DataFrame({"borrower_id": ["001"], "outstanding_balance": [100.0]}),
                {"consumer": loaded},
                {
                    "borrower": {
                        "borrower_id_field": "borrower_id",
                        "balance_field": "outstanding_balance",
                    },
                    "inputs": {"sources": {"consumer": source_spec}},
                },
                exceptions,
            )

            self.assertEqual(int(reconciliation.at[0, "entity_value_conflict_count"]), 1)
            self.assertIn("SOURCE_ENTITY_VALUE_CONFLICT", {item["code"] for item in exceptions})

    def test_account_keyed_source_links_and_aggregates_by_borrower(self):
        identity = pd.DataFrame(
            {
                "loan_id": ["A1", "A2", "A2", "B2-ACCOUNT"],
                "borrower_id": ["B1", "B1", "B1", "B2"],
            }
        )
        source = pd.DataFrame(
            [
                {
                    "loan_id": "A1",
                    "current_fico_score": 680,
                    "current_fico_date": "2026-01-01",
                    "collateral_id": "C1",
                    "current_appraisal_date": "2025-01-01",
                    "current_appraised_value_raw": 100.0,
                    "_source_row": 1,
                },
                {
                    "loan_id": "A2",
                    "current_fico_score": 720,
                    "current_fico_date": "2026-06-01",
                    "collateral_id": "C1",
                    "current_appraisal_date": "2025-02-01",
                    "current_appraised_value_raw": 100.0,
                    "_source_row": 2,
                },
                {
                    "loan_id": "A2",
                    "current_fico_score": 720,
                    "current_fico_date": "2026-06-01",
                    "collateral_id": "C2",
                    "current_appraisal_date": "2025-02-01",
                    "current_appraised_value_raw": 200.0,
                    "_source_row": 3,
                },
                {
                    "loan_id": "UNKNOWN",
                    "current_fico_score": 800,
                    "current_fico_date": "2026-06-01",
                    "collateral_id": "ORPHAN",
                    "current_appraisal_date": "2025-02-01",
                    "current_appraised_value_raw": 999.0,
                    "_source_row": 4,
                },
                {
                    "loan_id": "UNKNOWN",
                    "current_fico_score": 810,
                    "current_fico_date": "2026-06-01",
                    "collateral_id": "ORPHAN",
                    "current_appraisal_date": "2025-02-01",
                    "current_appraised_value_raw": 998.0,
                    "_source_row": 5,
                },
            ]
        )
        source_spec = {
            "key": "loan_id",
            "identity_key": "loan_id",
            "aggregation": {
                "fico_score": {
                    "method": "best_available",
                    "selection": "precedence",
                    "candidates": [
                        {
                            "field": "current_fico_score",
                            "date_field": "current_fico_date",
                        }
                    ],
                },
                "consumer_appraised_value": {
                    "method": "best_available_unique_sum",
                    "selection": "precedence",
                    "unique_fields": ["collateral_id"],
                    "candidates": [
                        {
                            "field": "current_appraised_value_raw",
                            "date_field": "current_appraisal_date",
                        }
                    ],
                },
            },
        }
        scenario = {
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "outstanding_balance",
            },
            "inputs": {"sources": {"consumer": source_spec}},
        }
        borrowers = pd.DataFrame(
            {
                "borrower_id": ["B1", "B2"],
                "outstanding_balance": [500.0, 100.0],
            }
        )
        loaded = {
            "identity": SimpleNamespace(
                frame=identity,
                paths=[Path("identity.csv")],
                coercion_issues=[],
            ),
            "consumer": SimpleNamespace(
                frame=source,
                paths=[Path("consumer.csv")],
                coercion_issues=[],
            ),
        }

        enriched = enrich_borrowers(borrowers, loaded, scenario).set_index("borrower_id")

        self.assertEqual(float(enriched.at["B1", "fico_score"]), 720.0)
        self.assertEqual(float(enriched.at["B1", "consumer_appraised_value"]), 300.0)
        self.assertTrue(pd.isna(enriched.at["B2", "fico_score"]))

        exceptions = []
        reconciliation = build_source_reconciliation(
            enriched.reset_index(), loaded, scenario, exceptions
        ).iloc[0]
        self.assertEqual(reconciliation["key_field"], "loan_id")
        self.assertEqual(reconciliation["identity_key_field"], "loan_id")
        self.assertEqual(int(reconciliation["unique_key_count"]), 3)
        self.assertEqual(int(reconciliation["matched_source_key_count"]), 2)
        self.assertEqual(int(reconciliation["orphan_source_key_count"]), 1)
        self.assertEqual(int(reconciliation["orphan_source_row_count"]), 2)
        self.assertEqual(int(reconciliation["matched_borrower_count"]), 1)
        self.assertEqual(int(reconciliation["unmatched_borrower_count"]), 1)
        self.assertEqual(float(reconciliation["matched_borrower_balance"]), 500.0)
        self.assertEqual(float(reconciliation["unmatched_borrower_balance"]), 100.0)
        self.assertIn("SOURCE_ORPHAN_KEYS", {item["code"] for item in exceptions})
        self.assertEqual(int(reconciliation["entity_value_conflict_count"]), 2)
        self.assertIn("SOURCE_ENTITY_VALUE_CONFLICT", {item["code"] for item in exceptions})

    def test_account_link_rejects_identity_key_shared_by_multiple_borrowers(self):
        source_spec = {
            "key": "loan_id",
            "identity_key": "loan_id",
            "aggregation": {"fico_score": {"method": "first", "field": "fico_score"}},
        }
        scenario = {
            "borrower": {
                "borrower_id_field": "borrower_id",
                "balance_field": "outstanding_balance",
            },
            "inputs": {"sources": {"consumer": source_spec}},
        }
        loaded = {
            "identity": SimpleNamespace(
                frame=pd.DataFrame(
                    {
                        "loan_id": ["A1", "A1"],
                        "borrower_id": ["B1", "B2"],
                    }
                )
            ),
            "consumer": SimpleNamespace(
                frame=pd.DataFrame({"loan_id": ["A1"], "fico_score": [700]})
            ),
        }
        borrowers = pd.DataFrame(
            {
                "borrower_id": ["B1", "B2"],
                "outstanding_balance": [100.0, 100.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "maps to multiple borrowers"):
            enrich_borrowers(borrowers, loaded, scenario)

    def test_missing_configured_source_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "source.csv").write_text("Different Header\nvalue\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing configured aliased columns"):
                read_table(
                    "aliased",
                    {
                        "path": "source.csv",
                        "column_aliases": {"borrower_id": "Borrower Number"},
                    },
                    directory,
                )

    def test_unmapped_source_columns_are_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "source.csv").write_text(
                "Borrower Number,Unexpected Column,borrower_id\n001,value,wrong value\n",
                encoding="utf-8",
            )

            loaded = read_table(
                "aliased",
                {
                    "path": "source.csv",
                    "column_aliases": {"borrower_id": "Borrower Number"},
                    "string_columns": ["borrower_id"],
                    "required_columns": ["borrower_id"],
                },
                directory,
            )

            imported_columns = [
                column
                for column in loaded.frame.columns
                if not str(column).startswith("_source_")
            ]
            self.assertEqual(imported_columns, ["borrower_id"])
            self.assertEqual(loaded.frame.at[0, "borrower_id"], "001")
            self.assertEqual(loaded.coercion_issues, [])


if __name__ == "__main__":
    unittest.main()
