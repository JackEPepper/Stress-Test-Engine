from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from stress_engine.borrower import (
    build_borrowers,
    split_identity_balance_scope,
)
from stress_engine.cecl import (
    INVALID_BALANCE_COUNT_FIELD,
    attach_cecl_reserve_basis,
    build_cecl_reserve_basis,
    validate_cecl_config,
)
from stress_engine.config import load_scenario
from stress_engine.engine import StressEngine
from stress_engine.reporting import build_cecl_summary, build_consumer_summary


ROOT = Path(__file__).resolve().parents[1]


def _scenario(
    *, current_method: str = "in_place", tolerance: float = 0.01
) -> dict:
    return {
        "borrower": {
            "borrower_id_field": "borrower_id",
            "loan_id_field": "loan_id",
            "balance_field": "balance",
            "portfolio_field": "model_portfolio",
            "module_field": "model_module",
            "sum_fields": ["balance", "cecl_reserve"],
        },
        "cecl": {
            "reserve_field": "cecl_reserve",
            "portfolio_field": "cecl_portfolio",
            "zero_balance_tolerance": tolerance,
            "reserve_basis": {
                "current_method": current_method,
                "central_tendency": {"z_score_threshold": 2.0},
                "historical": {"enabled": False},
            },
        },
        "tags": {},
        "modules": {},
        "stress_levels": ["S1", "S2"],
    }


class IdentityBalanceScopeTest(unittest.TestCase):
    def test_zero_balance_tolerance_must_be_finite_and_nonnegative(self):
        for value in (-0.01, np.inf, "not-a-number"):
            with self.subTest(value=value):
                scenario = _scenario(tolerance=value)
                with self.assertRaises(ValueError):
                    validate_cecl_config(scenario)

    def test_split_scope_normalizes_tolerance_and_describes_invalid_rows(self):
        scenario = _scenario(tolerance=0.01)
        identity = pd.DataFrame(
            [
                {
                    "loan_id": "positive",
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "model_portfolio": "Direct",
                    "cecl_portfolio": "Fallback",
                },
                {
                    "loan_id": "zero",
                    "borrower_id": "B2",
                    "balance": 0.0,
                    "model_portfolio": "Direct",
                    "cecl_portfolio": "Fallback",
                },
                {
                    "loan_id": "tiny-positive",
                    "borrower_id": "B3",
                    "balance": 0.005,
                    "model_portfolio": "Direct",
                    "cecl_portfolio": "Fallback",
                },
                {
                    "loan_id": "tiny-negative",
                    "borrower_id": "B4",
                    "balance": -0.005,
                    "model_portfolio": "Direct",
                    "cecl_portfolio": "Fallback",
                },
                {
                    "loan_id": "negative-boundary",
                    "borrower_id": "B5",
                    "balance": -0.01,
                    "model_portfolio": "Direct",
                    "cecl_portfolio": "Fallback",
                },
                {
                    "loan_id": "missing",
                    "borrower_id": "B6",
                    "balance": np.nan,
                    "model_portfolio": "",
                    "cecl_portfolio": "Missing Portfolio",
                },
                {
                    "loan_id": "nonnumeric",
                    "borrower_id": "B7",
                    "balance": "not-a-number",
                    "model_portfolio": np.nan,
                    "cecl_portfolio": "Text Portfolio",
                },
                {
                    "loan_id": "infinite",
                    "borrower_id": "B8",
                    "balance": np.inf,
                    "model_portfolio": "",
                    "cecl_portfolio": "Infinite Portfolio",
                },
                {
                    "loan_id": "material-negative",
                    "borrower_id": "B9",
                    "balance": -0.0101,
                    "model_portfolio": "Negative Direct",
                    "cecl_portfolio": "Negative Fallback",
                },
                {
                    "loan_id": "boolean",
                    "borrower_id": "B10",
                    "balance": True,
                    "model_portfolio": "Boolean Direct",
                    "cecl_portfolio": "Boolean Fallback",
                },
            ]
        )
        identity["_source_file"] = "identity.csv"
        identity["_source_file_row"] = np.arange(1, len(identity) + 1)
        identity["_source_row"] = np.arange(1, len(identity) + 1)
        identity.index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        original = identity.copy(deep=True)

        included, detail = split_identity_balance_scope(identity, scenario)

        self.assertEqual(
            included["loan_id"].tolist(),
            [
                "positive",
                "zero",
                "tiny-positive",
                "tiny-negative",
                "negative-boundary",
            ],
        )
        included = included.set_index("loan_id")
        self.assertEqual(float(included.at["positive", "balance"]), 100.0)
        for loan_id in (
            "zero",
            "tiny-positive",
            "tiny-negative",
            "negative-boundary",
        ):
            self.assertEqual(float(included.at[loan_id, "balance"]), 0.0)

        self.assertEqual(
            detail["loan_id"].tolist(),
            [
                "missing",
                "nonnumeric",
                "infinite",
                "material-negative",
                "boolean",
            ],
        )
        detail = detail.set_index("loan_id")
        self.assertTrue(
            detail.loc[["missing", "nonnumeric", "infinite", "boolean"],
                       "balance_issue"]
            .eq("missing_invalid_or_nonfinite")
            .all()
        )
        self.assertEqual(
            detail.at["material-negative", "balance_issue"],
            "negative_below_tolerance",
        )
        self.assertEqual(
            detail.at["material-negative", "input_value"], -0.0101
        )
        self.assertTrue(pd.isna(detail.at["missing", "input_value"]))
        self.assertEqual(
            detail.at["nonnumeric", "input_value"], "not-a-number"
        )
        self.assertTrue(np.isposinf(detail.at["infinite", "input_value"]))
        self.assertIs(detail.at["boolean", "input_value"], True)

        self.assertTrue(detail["module"].eq("Input").all())
        self.assertTrue(detail["stress_level"].eq("All").all())
        self.assertTrue(detail["test"].eq("Model population").all())
        self.assertTrue(detail["field"].eq("balance").all())
        self.assertTrue(detail["reason"].eq("invalid_balance").all())
        self.assertEqual(
            detail.at["missing", "portfolio"], "Missing Portfolio"
        )
        self.assertEqual(
            detail.at["material-negative", "portfolio"], "Negative Direct"
        )
        self.assertTrue(detail["_source_file"].eq("identity.csv").all())
        self.assertEqual(
            detail["_source_file_row"].astype(int).tolist(),
            [6, 7, 8, 9, 10],
        )
        self.assertEqual(
            detail["_source_row"].astype(int).tolist(),
            [6, 7, 8, 9, 10],
        )
        pd.testing.assert_frame_equal(identity, original)

    def test_build_borrowers_keeps_valid_sibling_and_drops_invalid_exposures(self):
        scenario = _scenario()
        identity = pd.DataFrame(
            [
                {
                    "_source_row": 1,
                    "loan_id": "B1-invalid",
                    "borrower_id": "B1",
                    "balance": np.inf,
                    "cecl_reserve": 50.0,
                    "model_portfolio": "C&I",
                    "model_module": "C&I",
                    "cecl_portfolio": "C&I",
                },
                {
                    "_source_row": 2,
                    "loan_id": "B1-valid",
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "cecl_reserve": 5.0,
                    "model_portfolio": "C&I",
                    "model_module": "C&I",
                    "cecl_portfolio": "C&I",
                },
                {
                    "_source_row": 3,
                    "loan_id": "B2-valid",
                    "borrower_id": "B2",
                    "balance": 200.0,
                    "cecl_reserve": 20.0,
                    "model_portfolio": "CRE",
                    "model_module": "CRE",
                    "cecl_portfolio": "CRE",
                },
                {
                    "_source_row": 4,
                    "loan_id": "B3-invalid",
                    "borrower_id": "B3",
                    "balance": -1.0,
                    "cecl_reserve": 30.0,
                    "model_portfolio": "CRE",
                    "model_module": "CRE",
                    "cecl_portfolio": "CRE",
                },
            ]
        )

        borrowers = build_borrowers(identity, scenario, [])

        self.assertEqual(borrowers["borrower_id"].tolist(), ["B1", "B2"])
        borrowers = borrowers.set_index("borrower_id")
        self.assertEqual(float(borrowers.at["B1", "balance"]), 100.0)
        self.assertEqual(float(borrowers.at["B1", "cecl_reserve"]), 5.0)
        self.assertEqual(borrowers.at["B1", "loan_id"], "B1-valid")
        self.assertEqual(int(borrowers.at["B1", "loan_count"]), 1)
        self.assertEqual(float(borrowers.at["B2", "balance"]), 200.0)
        self.assertEqual(float(borrowers.at["B2", "cecl_reserve"]), 20.0)
        self.assertEqual(int(borrowers.at["B2", "loan_count"]), 1)
        self.assertTrue(
            borrowers[INVALID_BALANCE_COUNT_FIELD].astype(int).eq(0).all()
        )


class CeclBalanceIsolationTest(unittest.TestCase):
    def test_duplicate_consumer_labels_keep_positionally_aligned_reserves(self):
        scenario = _scenario()
        scenario["cecl"]["portfolios"] = {
            "Consumer": {"method": "expected_loss"}
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "C1",
                    "balance": 100.0,
                    "cecl_reserve": 10.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "cecl_level_tag": "Consumer",
                    "base_bucket": "Pass",
                    "module_applied": "Consumer",
                    "consumer_el_unstressed": 5.0,
                    "consumer_el_S1": 6.0,
                    "consumer_el_S2": 7.0,
                },
                {
                    "borrower_id": "C2",
                    "balance": 200.0,
                    "cecl_reserve": 20.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "cecl_level_tag": "Consumer",
                    "base_bucket": "Pass",
                    "module_applied": "Consumer",
                    "consumer_el_unstressed": 10.0,
                    "consumer_el_S1": 12.0,
                    "consumer_el_S2": 14.0,
                },
            ],
            index=[3, 3],
        )

        consumer = build_consumer_summary(results, scenario).set_index(
            "stress_level"
        )
        cecl = build_cecl_summary(
            results,
            pd.DataFrame([{"portfolio": "Consumer"}]),
            scenario,
            [],
        )
        cecl_base = cecl[
            cecl["portfolio"].eq("Consumer")
            & cecl["stress_level"].eq("Base")
        ].iloc[0]

        self.assertEqual(
            float(consumer.at["Base", "proforma_cecl_reserve"]), 30.0
        )
        self.assertEqual(float(cecl_base["proforma_cecl_reserve"]), 30.0)
        self.assertEqual(cecl_base["cecl_reserve_status"], "available")

    def test_duplicate_row_labels_do_not_receive_sibling_reserves(self):
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "bad",
                    "balance": np.inf,
                    "cecl_reserve": 100.0,
                    "cecl_portfolio": "C&I",
                    "cecl_level_tag": "Tag A",
                    "base_bucket": "Pass",
                },
                {
                    "borrower_id": "good",
                    "balance": 100.0,
                    "cecl_reserve": 5.0,
                    "cecl_portfolio": "C&I",
                    "cecl_level_tag": "Tag A",
                    "base_bucket": "Pass",
                },
            ],
            index=[7, 7],
        )

        attached, basis = attach_cecl_reserve_basis(
            results, _scenario(), []
        )

        effective = basis.effective_reserve.to_numpy()
        self.assertTrue(np.isnan(effective[0]))
        self.assertEqual(float(effective[1]), 5.0)
        attached_effective = attached[
            "cecl_effective_reserve_base"
        ].to_numpy()
        self.assertTrue(np.isnan(attached_effective[0]))
        self.assertEqual(float(attached_effective[1]), 5.0)

    def test_invalid_rows_do_not_poison_valid_tags_for_current_methods(self):
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "A-valid",
                    "balance": 100.0,
                    "cecl_reserve": 10.0,
                    "cecl_portfolio": "C&I",
                    "model_portfolio": "C&I",
                    "cecl_level_tag": "Tag A",
                    "base_bucket": "Pass",
                    "model_module": "C&I",
                    INVALID_BALANCE_COUNT_FIELD: 99,
                },
                {
                    "borrower_id": "A-invalid",
                    "balance": np.inf,
                    "cecl_reserve": 1000.0,
                    "cecl_portfolio": "C&I",
                    "model_portfolio": "C&I",
                    "cecl_level_tag": "Tag A",
                    "base_bucket": "Pass",
                    "model_module": "C&I",
                },
                {
                    "borrower_id": "B-valid",
                    "balance": 200.0,
                    "cecl_reserve": 40.0,
                    "cecl_portfolio": "C&I",
                    "model_portfolio": "C&I",
                    "cecl_level_tag": "Tag B",
                    "base_bucket": "Pass",
                    "model_module": "C&I",
                },
                {
                    "borrower_id": "B-invalid",
                    "balance": -1.0,
                    "cecl_reserve": 1000.0,
                    "cecl_portfolio": "C&I",
                    "model_portfolio": "C&I",
                    "cecl_level_tag": "Tag B",
                    "base_bucket": "Pass",
                    "model_module": "C&I",
                },
            ]
        )

        for current_method in ("in_place", "central_tendency"):
            with self.subTest(current_method=current_method):
                exceptions: list[dict] = []
                basis = build_cecl_reserve_basis(
                    results,
                    _scenario(current_method=current_method),
                    exceptions,
                )

                ratios = basis.ratios.set_index("cecl_level_tag")
                self.assertEqual(set(ratios.index), {"Tag A", "Tag B"})
                self.assertTrue(ratios["status"].eq("available").all())
                self.assertTrue(ratios["exception_code"].eq("").all())
                self.assertTrue(
                    ratios["invalid_balance_count"].astype(int).eq(0).all()
                )
                self.assertAlmostEqual(
                    float(ratios.at["Tag A", "base_balance"]), 100.0
                )
                self.assertAlmostEqual(
                    float(ratios.at["Tag A", "reserve_ratio"]), 0.10
                )
                self.assertAlmostEqual(
                    float(ratios.at["Tag B", "base_balance"]), 200.0
                )
                self.assertAlmostEqual(
                    float(ratios.at["Tag B", "reserve_ratio"]), 0.20
                )

                self.assertAlmostEqual(
                    float(basis.effective_reserve.loc[0]), 10.0
                )
                self.assertTrue(pd.isna(basis.effective_reserve.loc[1]))
                self.assertAlmostEqual(
                    float(basis.effective_reserve.loc[2]), 40.0
                )
                self.assertTrue(pd.isna(basis.effective_reserve.loc[3]))

                exclusion_events = [
                    row
                    for row in exceptions
                    if row["code"] == "CECL_BALANCE_EXCLUDED"
                ]
                self.assertEqual(len(exclusion_events), 1)
                self.assertEqual(
                    exclusion_events[0]["details"], "excluded_row_count=2"
                )
                self.assertNotIn(
                    "CECL_BALANCE_INVALID",
                    {row["code"] for row in exceptions},
                )

    def test_engine_excludes_bad_loan_and_keeps_valid_sibling_reportable(self):
        scenario, base_dir = load_scenario(
            ROOT / "examples" / "scenario.json"
        )
        identity = pd.read_csv(
            ROOT / "examples" / "data" / "loans.csv",
            dtype=str,
            keep_default_na=False,
        )
        identity.loc[
            identity["loan_id"].eq("L001"), "outstanding_balance"
        ] = "not-a-balance"

        with tempfile.TemporaryDirectory() as tmp:
            identity_path = Path(tmp) / "loans.csv"
            identity.to_csv(identity_path, index=False)
            scenario["inputs"]["identity"]["path"] = str(identity_path)
            run = StressEngine(scenario, base_dir).run(
                write_outputs=False, run_comparison=False
            )

        identity_profile = run["reports"]["input_summary"]
        identity_profile = identity_profile[
            identity_profile["dataset"].eq("identity")
        ]
        self.assertEqual(len(identity_profile), len(identity.columns))
        self.assertFalse(
            identity_profile["field"]
            .astype(str)
            .str.startswith("_raw_invalid_numeric__")
            .any()
        )
        identity_metadata = next(
            row
            for row in run["metadata"]["input_files"]
            if row["name"] == "identity"
        )
        self.assertEqual(identity_metadata["columns"], len(identity.columns))

        b001 = run["borrowers"].set_index("borrower_id").loc["B001"]
        self.assertEqual(float(b001["outstanding_balance"]), 250_000.0)
        self.assertEqual(float(b001["cecl_reserve"]), 2_500.0)
        self.assertEqual(int(b001["loan_count"]), 1)

        detail = run["reports"]["out_of_scope_detail"]
        input_scope = detail[
            detail["module"].eq("Input")
            & detail["field"].eq("outstanding_balance")
        ]
        self.assertEqual(len(input_scope), 1)
        excluded = input_scope.iloc[0]
        self.assertEqual(excluded["borrower_id"], "B001")
        self.assertEqual(excluded["loan_id"], "L001")
        self.assertEqual(excluded["stress_level"], "All")
        self.assertEqual(excluded["reason"], "invalid_balance")
        self.assertEqual(excluded["input_value"], "not-a-balance")

        scope_summary = run["reports"]["out_of_scope_summary"]
        population_scope = scope_summary[
            scope_summary["module"].eq("Input")
            & scope_summary["field"].eq("outstanding_balance")
        ]
        self.assertEqual(int(population_scope.iloc[0]["count"]), 1)

        cecl = run["reports"]["cecl_summary"]
        aggregate = cecl[
            cecl["portfolio"].eq("Aggregate")
            & cecl["bucket"].eq("Total")
        ].set_index("stress_level")
        self.assertEqual(set(aggregate.index), {"Base", "S1", "S2"})
        self.assertTrue(aggregate["balance"].eq(5_400_000.0).all())
        self.assertTrue(
            aggregate["cecl_reserve_status"].eq("available").all()
        )
        self.assertTrue(aggregate["exception_code"].eq("").all())

        exception_codes = set(
            run["reports"]["exception_log"]["code"].astype(str)
        )
        self.assertIn("IDENTITY_BALANCE_MISSING", exception_codes)
        self.assertNotIn("CECL_BALANCE_INVALID", exception_codes)

    def test_targeted_engine_excludes_bad_exposure_from_every_variant(self):
        scenario, base_dir = load_scenario(
            ROOT / "examples" / "targeted_stress.json"
        )
        identity = pd.read_csv(
            ROOT / "examples" / "data" / "loans.csv",
            dtype=str,
            keep_default_na=False,
        )
        identity.loc[
            identity["loan_id"].eq("L001"), "outstanding_balance"
        ] = "not-a-balance"

        with tempfile.TemporaryDirectory() as tmp:
            identity_path = Path(tmp) / "loans.csv"
            identity.to_csv(identity_path, index=False)
            scenario["inputs"]["identity"]["path"] = str(identity_path)
            run = StressEngine(scenario, base_dir).run(
                write_outputs=False, run_comparison=False
            )

        self.assertNotIn("L001", set(run["loan_context"]["loan_id"]))
        self.assertIn("L002", set(run["loan_context"]["loan_id"]))
        self.assertNotIn("L001", set(run["variant_results"]["loan_id"]))
        variant_names = set(run["variant_results"]["scenario_variant"])
        self.assertEqual(
            int(run["variant_results"]["loan_id"].eq("L002").sum()),
            len(variant_names),
        )

        detail = run["reports"]["out_of_scope_detail"]
        input_scope = detail[
            detail["module"].eq("Input")
            & detail["field"].eq("outstanding_balance")
        ]
        self.assertEqual(len(input_scope), 1)
        self.assertEqual(input_scope.iloc[0]["loan_id"], "L001")
        self.assertEqual(input_scope.iloc[0]["scenario_variant"], "all")
        self.assertEqual(
            input_scope.iloc[0]["input_value"], "not-a-balance"
        )
        input_scope_summary = run["reports"]["out_of_scope_summary"]
        input_scope_summary = input_scope_summary[
            input_scope_summary["module"].eq("Input")
            & input_scope_summary["field"].eq("outstanding_balance")
        ]
        self.assertEqual(len(input_scope_summary), 1)
        self.assertEqual(
            input_scope_summary.iloc[0]["scenario_variant"], "all"
        )
        self.assertEqual(int(input_scope_summary.iloc[0]["count"]), 1)

        cecl = run["reports"]["cecl_summary"]
        aggregate = cecl[
            cecl["portfolio"].eq("Aggregate")
            & cecl["bucket"].eq("Total")
        ]
        self.assertEqual(
            set(aggregate["scenario_variant"]), variant_names
        )
        self.assertTrue(aggregate["balance"].eq(5_400_000.0).all())
        baseline = aggregate[aggregate["scenario_variant"].eq("baseline")]
        self.assertTrue(
            baseline["cecl_reserve_status"].eq("available").all()
        )
        self.assertTrue(baseline["exception_code"].eq("").all())
        self.assertFalse(
            aggregate["exception_code"]
            .astype(str)
            .str.contains("CECL_BALANCE_INVALID")
            .any()
        )
        self.assertNotIn(
            "CECL_BALANCE_INVALID",
            set(run["reports"]["exception_log"]["code"].astype(str)),
        )

    def test_all_invalid_identity_population_finishes_with_scope_audit(self):
        scenario, base_dir = load_scenario(
            ROOT / "examples" / "scenario.json"
        )
        identity = pd.read_csv(
            ROOT / "examples" / "data" / "loans.csv",
            dtype=str,
            keep_default_na=False,
        )
        identity["outstanding_balance"] = "not-a-balance"

        with tempfile.TemporaryDirectory() as tmp:
            identity_path = Path(tmp) / "loans.csv"
            identity.to_csv(identity_path, index=False)
            scenario["inputs"]["identity"]["path"] = str(identity_path)
            run = StressEngine(scenario, base_dir).run(
                write_outputs=False, run_comparison=False
            )

        self.assertTrue(run["borrowers"].empty)
        self.assertTrue(run["results"].empty)
        input_scope = run["reports"]["out_of_scope_detail"]
        self.assertEqual(len(input_scope), len(identity))
        self.assertTrue(input_scope["module"].eq("Input").all())
        self.assertTrue(input_scope["stress_level"].eq("All").all())

        cecl = run["reports"]["cecl_summary"]
        self.assertTrue(cecl["portfolio"].eq("Aggregate").all())
        self.assertTrue(cecl["balance"].eq(0.0).all())
        self.assertTrue(cecl["proforma_cecl_reserve"].eq(0.0).all())
        self.assertTrue(cecl["cecl_reserve_status"].eq("available").all())
        self.assertTrue(cecl["exception_code"].eq("").all())


if __name__ == "__main__":
    unittest.main()
