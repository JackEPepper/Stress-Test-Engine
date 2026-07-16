from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stress_engine.borrower import build_source_reconciliation
from stress_engine.config import load_scenario
from stress_engine.config_tool import batch_config_from_csv
from stress_engine.io import read_table


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

    def test_every_example_input_column_has_an_explicit_alias(self):
        scenario, base_dir = load_scenario(SCENARIO)
        specs = {"identity": scenario["inputs"]["identity"], **scenario["inputs"]["sources"]}

        for name, spec in specs.items():
            aliases = spec.get("column_aliases", {})
            relative_paths = [spec["path"]] if "path" in spec else spec["paths"]
            for relative_path in relative_paths:
                raw_columns = set(pd.read_csv(base_dir / relative_path, nrows=0).columns)
                self.assertEqual(
                    set(aliases.values()),
                    raw_columns,
                    msg=f"{name}:{relative_path} must map every source column exactly once",
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
                "Borrower Number,Current DSCR\n001,1.25\n",
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

    def test_multi_file_input_source_concatenates_in_listed_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "first.csv").write_text("Borrower Number,Score\n001,680\n", encoding="utf-8")
            (directory / "second.csv").write_text("Borrower Number,Score\n002,720\n", encoding="utf-8")

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

    def test_unmapped_source_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "source.csv").write_text(
                "Borrower Number,Unexpected Column\n001,value\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "columns missing from column_aliases"):
                read_table(
                    "aliased",
                    {
                        "path": "source.csv",
                        "column_aliases": {"borrower_id": "Borrower Number"},
                    },
                    directory,
                )


if __name__ == "__main__":
    unittest.main()
