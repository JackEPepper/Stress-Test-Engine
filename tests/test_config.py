from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stress_engine.borrower import build_borrowers, build_source_reconciliation, enrich_borrowers
from stress_engine.config import load_scenario
from stress_engine.config_tool import batch_config_from_csv
from stress_engine.io import load_inputs, read_table


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenario.json"


class ScenarioConfigTest(unittest.TestCase):
    def test_example_manifest_loads_all_fragments(self):
        scenario, base_dir = load_scenario(SCENARIO)

        self.assertEqual(base_dir, SCENARIO.parent)
        self.assertEqual(scenario["scenario_id"], "example_2026q2")
        self.assertEqual(set(scenario["modules"]), {"CRE", "C&I", "Consumer"})
        self.assertEqual(len(scenario["tags"]), 17)
        self.assertNotIn("$include", scenario)
        self.assertEqual(len(scenario["_metadata"]["scenario_files"]), 11)

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

    def test_batch_variable_csv_converts_to_engine_shape(self):
        payload = batch_config_from_csv(
            ROOT / "examples" / "scenario_variables.csv",
            output_directory="outputs/example_batch",
            max_scenarios=20,
        )

        batch = payload["scenario_batch"]
        self.assertEqual(batch["mode"], "grid")
        self.assertEqual(batch["max_scenarios"], 20)
        self.assertEqual(batch["variables"][0]["range"], {"start": 0.03, "stop": 0.07, "step": 0.02})
        self.assertEqual(batch["variables"][1]["values"], [1.25, 1.75])

    def test_committed_batch_json_matches_the_variable_csv(self):
        generated = batch_config_from_csv(
            ROOT / "examples" / "scenario_variables.csv",
            output_directory="outputs/example_batch",
            max_scenarios=20,
        )
        committed = json.loads((ROOT / "examples" / "scenario_batch.json").read_text(encoding="utf-8"))

        self.assertEqual(committed, generated)

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
