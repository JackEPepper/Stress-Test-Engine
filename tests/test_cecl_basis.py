from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from stress_engine.borrower import build_borrowers, record_identity_data_issues
from stress_engine.cecl import (
    attach_cecl_reserve_basis,
    build_cecl_reserve_basis,
)
from stress_engine.config import validate_scenario
from stress_engine.engine import StressEngine
from stress_engine.modules.consumer import run_consumer
from stress_engine.reporting import (
    build_cecl_bucket_summary,
    build_cecl_summary,
    build_consumer_summary,
)


def _scenario(reserve_basis: dict | None = None) -> dict:
    reserve_fields = [
        "cecl_reserve",
        "cecl_reserve_prior_1",
        "cecl_reserve_prior_2",
    ]
    cecl = {
        "reserve_field": "cecl_reserve",
        "portfolio_field": "cecl_portfolio",
    }
    if reserve_basis is not None:
        cecl["reserve_basis"] = reserve_basis
    return {
        "inputs": {
            "identity": {
                "column_aliases": {
                    "borrower_id": "borrower_id",
                    "balance": "balance",
                    "cecl_portfolio": "cecl_portfolio",
                    **{field: field for field in reserve_fields},
                },
                "numeric_columns": ["balance", *reserve_fields],
                "required_columns": [
                    "borrower_id",
                    "balance",
                    "cecl_portfolio",
                    *reserve_fields,
                ],
            }
        },
        "borrower": {
            "borrower_id_field": "borrower_id",
            "balance_field": "balance",
            "portfolio_field": "model_portfolio",
            "sum_fields": ["balance", *reserve_fields],
        },
        "tags": {},
        "modules": {},
        "stress_levels": ["S1", "S2"],
        "cecl": cecl,
    }


def _commercial_frame(
    balances: list[float],
    reserves: list[float],
    *,
    portfolio: str = "CRE",
    bucket: str = "Pass",
) -> pd.DataFrame:
    count = len(balances)
    return pd.DataFrame(
        {
            "borrower_id": [f"B{index + 1}" for index in range(count)],
            "balance": balances,
            "cecl_portfolio": [portfolio] * count,
            "model_portfolio": [portfolio] * count,
            "base_bucket": [bucket] * count,
            "stressed_bucket_S1": [bucket] * count,
            "stressed_bucket_S2": [bucket] * count,
            "cecl_reserve": reserves,
            "module_applied": ["CRE"] * count,
        }
    )


def _weighted_basis(
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    *,
    period_method: str = "in_place",
    z_score_threshold: float | None = None,
) -> dict:
    config: dict = {
        "method": "weighted_history",
        "weighted_history": {
            "period_method": period_method,
            "periods": [
                {
                    "name": "current",
                    "reserve_field": "cecl_reserve",
                    "weight": weights[0],
                },
                {
                    "name": "prior_1",
                    "reserve_field": "cecl_reserve_prior_1",
                    "weight": weights[1],
                },
                {
                    "name": "prior_2",
                    "reserve_field": "cecl_reserve_prior_2",
                    "weight": weights[2],
                },
            ],
        },
    }
    if z_score_threshold is not None:
        config["central_tendency"] = {
            "z_score_threshold": z_score_threshold
        }
    return config


class CeclReserveBasisTest(unittest.TestCase):
    def test_omitted_basis_and_explicit_in_place_are_legacy_compatible(self):
        results = _commercial_frame(
            [100.0, 300.0, 200.0],
            [1.0, 9.0, np.nan],
        )

        omitted = build_cecl_reserve_basis(results, _scenario(), [])
        explicit = build_cecl_reserve_basis(
            results,
            _scenario({"method": "in_place"}),
            [],
        )

        self.assertEqual(omitted.method, "in_place")
        self.assertEqual(explicit.method, "in_place")
        pd.testing.assert_series_equal(
            omitted.effective_reserve,
            pd.Series([1.0, 9.0, 0.0], index=results.index),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            explicit.effective_reserve,
            omitted.effective_reserve,
            check_names=False,
        )
        pd.testing.assert_frame_equal(explicit.ratios, omitted.ratios)
        self.assertAlmostEqual(float(omitted.ratios.iloc[0]["reserve_ratio"]), 1 / 60)

    def test_central_tendency_is_arithmetic_not_balance_weighted(self):
        results = _commercial_frame(
            [100.0, 1_000.0, 10_000.0],
            [1.0, 20.0, 300.0],
        )
        scenario = _scenario(
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 10.0},
            }
        )

        basis = build_cecl_reserve_basis(results, scenario, [])

        self.assertEqual(basis.method, "central_tendency")
        self.assertAlmostEqual(float(basis.ratios.iloc[0]["reserve_ratio"]), 0.02)
        np.testing.assert_allclose(
            basis.effective_reserve.to_numpy(),
            np.array([2.0, 20.0, 200.0]),
        )
        self.assertAlmostEqual(float(basis.effective_reserve.sum()), 222.0)
        audit = basis.audit.iloc[0]
        self.assertEqual(int(audit["observation_count"]), 3)
        self.assertEqual(int(audit["included_observation_count"]), 3)
        self.assertEqual(int(audit["excluded_observation_count"]), 0)
        self.assertAlmostEqual(float(audit["raw_mean_reserve_ratio"]), 0.02)

    def test_central_tendency_uses_population_zscore_and_trims_once(self):
        results = _commercial_frame(
            [100.0] * 6,
            [1.0, 1.0, 1.0, 1.0, 1.0, 50.0],
        )
        scenario = _scenario(
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 2.0},
            }
        )

        basis = build_cecl_reserve_basis(results, scenario, [])

        # The 50% observation has population z = sqrt(5), while each retained
        # 1% observation has abs(z) = 1/sqrt(5).
        self.assertAlmostEqual(
            float(basis.audit.iloc[0]["raw_std_reserve_ratio"]),
            0.18261221816248283,
        )
        self.assertEqual(int(basis.audit.iloc[0]["included_observation_count"]), 5)
        self.assertEqual(int(basis.audit.iloc[0]["excluded_observation_count"]), 1)
        self.assertAlmostEqual(float(basis.ratios.iloc[0]["reserve_ratio"]), 0.01)
        np.testing.assert_allclose(
            basis.effective_reserve.to_numpy(), np.ones(6)
        )
        self.assertAlmostEqual(float(basis.effective_reserve.sum()), 6.0)

    def test_central_tendency_handles_singleton_zero_variance_and_boundary(self):
        results = pd.concat(
            [
                _commercial_frame(
                    [100.0], [3.0], portfolio="Singleton"
                ),
                _commercial_frame(
                    [100.0, 200.0, 300.0],
                    [2.0, 4.0, 6.0],
                    portfolio="Identical",
                ),
                _commercial_frame(
                    [100.0, 100.0],
                    [1.0, 3.0],
                    portfolio="Boundary",
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = [f"B{index}" for index in results.index]
        scenario = _scenario(
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 1.0},
            }
        )

        basis = build_cecl_reserve_basis(results, scenario, [])
        rates = basis.ratios.set_index("portfolio")["reserve_ratio"]

        self.assertAlmostEqual(float(rates["Singleton"]), 0.03)
        self.assertAlmostEqual(float(rates["Identical"]), 0.02)
        # The two Boundary observations have z-scores exactly -1 and +1 and
        # must be retained because the configured comparison is inclusive.
        self.assertAlmostEqual(float(rates["Boundary"]), 0.02)
        boundary_audit = basis.audit[basis.audit["portfolio"] == "Boundary"].iloc[0]
        self.assertEqual(int(boundary_audit["included_observation_count"]), 2)
        self.assertEqual(int(boundary_audit["excluded_observation_count"]), 0)
        identical_audit = basis.audit[basis.audit["portfolio"] == "Identical"].iloc[0]
        self.assertEqual(float(identical_audit["raw_std_reserve_ratio"]), 0.0)

    def test_central_tendency_reports_when_trimming_removes_every_observation(self):
        results = _commercial_frame([100.0, 100.0], [1.0, 3.0])
        scenario = _scenario(
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 0.5},
            }
        )
        exceptions: list[dict] = []

        basis = build_cecl_reserve_basis(results, scenario, exceptions)

        self.assertTrue(basis.effective_reserve.isna().all())
        self.assertEqual(basis.ratios.iloc[0]["status"], "unavailable")
        self.assertEqual(
            basis.audit.iloc[0]["basis_exception_code"],
            "CECL_BASIS_PERIOD_UNAVAILABLE",
        )
        self.assertIn(
            "CECL_BASIS_PERIOD_UNAVAILABLE",
            {row["code"] for row in exceptions},
        )

    def test_central_tendency_isolated_by_portfolio_and_base_bucket(self):
        results = pd.concat(
            [
                _commercial_frame([100.0, 100.0], [1.0, 1.0], bucket="Pass"),
                _commercial_frame(
                    [100.0, 100.0],
                    [5.0, 5.0],
                    bucket="Substandard",
                ),
                _commercial_frame(
                    [100.0, 100.0],
                    [8.0, 8.0],
                    portfolio="C&I",
                    bucket="Pass",
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = [f"B{index}" for index in results.index]
        scenario = _scenario(
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 2.0},
            }
        )

        basis = build_cecl_reserve_basis(results, scenario, [])
        rates = basis.ratios.set_index(["portfolio", "bucket"])["reserve_ratio"]

        self.assertAlmostEqual(float(rates[("CRE", "Pass")]), 0.01)
        self.assertAlmostEqual(float(rates[("CRE", "Substandard")]), 0.05)
        self.assertAlmostEqual(float(rates[("C&I", "Pass")]), 0.08)

    def test_weighted_history_supports_equal_and_arbitrary_weights(self):
        results = _commercial_frame([100.0, 300.0], [1.0, 9.0])
        results["cecl_reserve_prior_1"] = [2.0, 12.0]
        results["cecl_reserve_prior_2"] = [4.0, 18.0]

        equal = build_cecl_reserve_basis(
            results,
            _scenario(_weighted_basis()),
            [],
        )
        arbitrary = build_cecl_reserve_basis(
            results,
            _scenario(_weighted_basis((0.2, 0.3, 0.5))),
            [],
        )

        np.testing.assert_allclose(
            equal.effective_reserve.to_numpy(),
            np.array([7 / 3, 13.0]),
        )
        self.assertAlmostEqual(float(equal.effective_reserve.sum()), 46 / 3)
        self.assertAlmostEqual(float(equal.ratios.iloc[0]["reserve_ratio"]), 23 / 600)
        np.testing.assert_allclose(
            arbitrary.effective_reserve.to_numpy(),
            np.array([2.8, 14.4]),
        )
        self.assertAlmostEqual(float(arbitrary.effective_reserve.sum()), 17.2)
        self.assertAlmostEqual(float(arbitrary.ratios.iloc[0]["reserve_ratio"]), 0.043)
        self.assertEqual(set(equal.audit["period"]), {"current", "prior_1", "prior_2"})
        self.assertAlmostEqual(float(equal.audit["weight"].sum()), 1.0)
        self.assertTrue(equal.audit["status"].eq("available").all())

    def test_weighted_history_supports_an_arbitrary_period_count(self):
        reserve_basis = _weighted_basis((0.1, 0.2, 0.3))
        reserve_basis["weighted_history"]["periods"].append(
            {
                "name": "prior_3",
                "reserve_field": "cecl_reserve_prior_3",
                "weight": 0.4,
            }
        )
        scenario = _scenario(reserve_basis)
        identity = scenario["inputs"]["identity"]
        identity["column_aliases"]["cecl_reserve_prior_3"] = "Prior 3"
        identity["numeric_columns"].append("cecl_reserve_prior_3")
        identity["required_columns"].append("cecl_reserve_prior_3")
        scenario["borrower"]["sum_fields"].append("cecl_reserve_prior_3")
        results = _commercial_frame([100.0], [10.0])
        results["cecl_reserve_prior_1"] = 20.0
        results["cecl_reserve_prior_2"] = 30.0
        results["cecl_reserve_prior_3"] = 40.0

        validate_scenario(scenario)
        basis = build_cecl_reserve_basis(results, scenario, [])

        self.assertEqual(len(basis.audit), 4)
        self.assertAlmostEqual(float(basis.effective_reserve.iloc[0]), 30.0)
        self.assertAlmostEqual(float(basis.ratios.iloc[0]["reserve_ratio"]), 0.3)

    def test_weighted_history_can_trim_each_period_before_blending(self):
        results = _commercial_frame(
            [100.0] * 6,
            [1.0, 1.0, 1.0, 1.0, 1.0, 50.0],
        )
        results["cecl_reserve_prior_1"] = [2.0, 2.0, 2.0, 2.0, 2.0, 60.0]
        results["cecl_reserve_prior_2"] = [3.0, 3.0, 3.0, 3.0, 3.0, 70.0]
        scenario = _scenario(
            _weighted_basis(
                period_method="central_tendency",
                z_score_threshold=2.0,
            )
        )

        basis = build_cecl_reserve_basis(results, scenario, [])

        self.assertAlmostEqual(float(basis.ratios.iloc[0]["reserve_ratio"]), 0.02)
        np.testing.assert_allclose(
            basis.effective_reserve.to_numpy(), np.full(6, 2.0)
        )
        self.assertTrue(basis.audit["period_method"].eq("central_tendency").all())
        self.assertTrue(basis.audit["excluded_observation_count"].eq(1).all())

    def test_weighted_history_missing_cells_are_zero_and_audited(self):
        results = _commercial_frame([100.0, 100.0], [10.0, 20.0])
        results["cecl_reserve_prior_1"] = [np.inf, 30.0]
        results["cecl_reserve_prior_2"] = [30.0, np.nan]
        exceptions: list[dict] = []

        basis = build_cecl_reserve_basis(
            results,
            _scenario(_weighted_basis()),
            exceptions,
        )

        np.testing.assert_allclose(
            basis.effective_reserve.to_numpy(),
            np.array([40 / 3, 50 / 3]),
        )
        self.assertAlmostEqual(float(basis.effective_reserve.sum()), 30.0)
        self.assertAlmostEqual(float(basis.ratios.iloc[0]["reserve_ratio"]), 0.15)
        missing = basis.audit[basis.audit["missing_reserve_count"] > 0]
        self.assertEqual(set(missing["period"]), {"prior_1", "prior_2"})
        self.assertEqual(int(missing["missing_reserve_count"].sum()), 2)
        self.assertTrue(exceptions)

    def test_weighted_history_missing_configured_field_is_unavailable(self):
        results = _commercial_frame([100.0, 300.0], [1.0, 9.0])
        results["cecl_reserve_prior_1"] = [2.0, 12.0]
        exceptions: list[dict] = []

        basis = build_cecl_reserve_basis(
            results,
            _scenario(_weighted_basis()),
            exceptions,
        )

        self.assertTrue(basis.effective_reserve.isna().all())
        self.assertTrue(basis.ratios["reserve_ratio"].isna().all())
        self.assertTrue(basis.ratios["status"].eq("unavailable").all())
        self.assertTrue(basis.ratios["exception_code"].astype(str).str.len().gt(0).all())
        self.assertIn(
            "cecl_reserve_prior_2",
            " ".join(str(value) for row in exceptions for value in row.values()),
        )

    def test_invalid_reserve_basis_configuration_is_rejected(self):
        invalid_configs = [
            {"method": "unknown"},
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 0.0},
            },
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": np.inf},
            },
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": True},
            },
            _weighted_basis((0.2, 0.3, 0.4)),
            _weighted_basis((0.0, 0.5, 0.5)),
            _weighted_basis((0.5, 0.5, 0.0)),
            _weighted_basis((-0.1, 0.6, 0.5)),
            _weighted_basis(period_method="unsupported"),
        ]
        for reserve_basis in invalid_configs:
            with self.subTest(reserve_basis=reserve_basis):
                with self.assertRaises(ValueError):
                    validate_scenario(_scenario(reserve_basis))

        duplicate_name = _weighted_basis()
        duplicate_name["weighted_history"]["periods"][1]["name"] = "current"
        with self.assertRaises(ValueError):
            validate_scenario(_scenario(duplicate_name))

        duplicate_field = _weighted_basis()
        duplicate_field["weighted_history"]["periods"][1][
            "reserve_field"
        ] = "cecl_reserve"
        with self.assertRaises(ValueError):
            validate_scenario(_scenario(duplicate_field))

    def test_weighted_history_fields_require_alias_numeric_and_sum_wiring(self):
        for location in (
            "column_aliases",
            "numeric_columns",
            "required_columns",
            "sum_fields",
        ):
            with self.subTest(location=location):
                scenario = _scenario(_weighted_basis())
                if location == "sum_fields":
                    scenario["borrower"][location].remove(
                        "cecl_reserve_prior_2"
                    )
                else:
                    identity = scenario["inputs"]["identity"]
                    if location == "column_aliases":
                        identity[location].pop("cecl_reserve_prior_2")
                    else:
                        identity[location].remove("cecl_reserve_prior_2")
                with self.assertRaises(ValueError):
                    validate_scenario(scenario)

    def test_central_tendency_requires_current_reserve_aggregation_wiring(self):
        basis = {
            "method": "central_tendency",
            "central_tendency": {"z_score_threshold": 2.0},
        }
        for location in (
            "column_aliases",
            "numeric_columns",
            "required_columns",
            "sum_fields",
        ):
            with self.subTest(location=location):
                scenario = _scenario(basis)
                if location == "sum_fields":
                    scenario["borrower"][location].remove("cecl_reserve")
                elif location == "column_aliases":
                    scenario["inputs"]["identity"][location].pop(
                        "cecl_reserve"
                    )
                else:
                    scenario["inputs"]["identity"][location].remove(
                        "cecl_reserve"
                    )
                with self.assertRaises(ValueError):
                    validate_scenario(scenario)

    def test_borrower_aggregation_preserves_loan_missing_counts_for_audit(self):
        scenario = _scenario(_weighted_basis())
        identity = pd.DataFrame(
            [
                {
                    "_source_row": 1,
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "cecl_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "cecl_reserve": 1.0,
                    "cecl_reserve_prior_1": np.nan,
                    "cecl_reserve_prior_2": 3.0,
                },
                {
                    "_source_row": 2,
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "cecl_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "cecl_reserve": 9.0,
                    "cecl_reserve_prior_1": 4.0,
                    "cecl_reserve_prior_2": 5.0,
                },
            ]
        )

        borrowers = build_borrowers(identity, scenario, [])
        basis = build_cecl_reserve_basis(borrowers, scenario, [])

        prior_1 = basis.audit[basis.audit["period"] == "prior_1"].iloc[0]
        self.assertEqual(int(prior_1["missing_reserve_count"]), 1)

    def test_weighted_history_accepts_vendor_headers_for_canonical_aliases(self):
        scenario = _scenario(_weighted_basis())
        aliases = scenario["inputs"]["identity"]["column_aliases"]
        aliases["cecl_reserve"] = "Current CECL Reserve"
        aliases["cecl_reserve_prior_1"] = "Prior Quarter CECL Reserve"
        aliases["cecl_reserve_prior_2"] = "Prior Year-End CECL Reserve"

        # column_aliases maps canonical engine names to vendor source headers;
        # validation must inspect the keys rather than requiring both to match.
        validate_scenario(scenario)

    def test_weighted_history_normalizes_validated_strings_at_runtime(self):
        reserve_basis = _weighted_basis()
        periods = reserve_basis["weighted_history"]["periods"]
        periods[0].update(
            {"name": " current ", "reserve_field": " cecl_reserve ", "weight": "50%"}
        )
        periods[1].update({"weight": "25%"})
        periods[2].update({"weight": "25%"})
        scenario = _scenario(reserve_basis)
        results = _commercial_frame([100.0], [10.0])
        results["cecl_reserve_prior_1"] = 20.0
        results["cecl_reserve_prior_2"] = 30.0

        validate_scenario(scenario)
        basis = build_cecl_reserve_basis(results, scenario, [])

        self.assertEqual(set(basis.audit["period"]), {"current", "prior_1", "prior_2"})
        self.assertAlmostEqual(float(basis.effective_reserve.iloc[0]), 17.5)

    def test_nonfinite_loan_reserve_is_zeroed_before_borrower_sum(self):
        scenario = _scenario({"method": "in_place"})
        identity = pd.DataFrame(
            [
                {
                    "_source_row": 1,
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "cecl_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "cecl_reserve": np.inf,
                },
                {
                    "_source_row": 2,
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "cecl_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "cecl_reserve": 10.0,
                },
            ]
        )

        exceptions: list[dict] = []
        record_identity_data_issues(identity, scenario, exceptions)
        borrowers = build_borrowers(identity, scenario, exceptions)
        standard = build_cecl_reserve_basis(borrowers, scenario, [])
        targeted = build_cecl_reserve_basis(identity, scenario, [])

        self.assertEqual(float(standard.effective_reserve.sum()), 10.0)
        self.assertEqual(float(targeted.effective_reserve.sum()), 10.0)
        self.assertEqual(int(standard.audit.iloc[0]["missing_reserve_count"]), 1)
        self.assertIn(
            "CECL_LOAN_RESERVE_MISSING_TREATED_AS_ZERO",
            {row["code"] for row in exceptions},
        )

    def test_missing_cecl_portfolio_is_excluded_and_logged(self):
        results = _commercial_frame([100.0], [10.0])
        results["cecl_portfolio"] = np.nan
        exceptions: list[dict] = []

        basis = build_cecl_reserve_basis(results, _scenario(), exceptions)

        self.assertTrue(basis.effective_reserve.isna().all())
        self.assertTrue(basis.ratios.empty)
        self.assertIn("CECL_PORTFOLIO_MISSING", {row["code"] for row in exceptions})

    def test_nonfinite_loan_balance_is_excluded_before_borrower_sum(self):
        scenario = _scenario({"method": "in_place"})
        identity = pd.DataFrame(
            [
                {
                    "_source_row": 1,
                    "borrower_id": "B1",
                    "balance": np.inf,
                    "cecl_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "cecl_reserve": 10.0,
                },
                {
                    "_source_row": 2,
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "cecl_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "cecl_reserve": 5.0,
                },
            ]
        )

        borrowers = build_borrowers(identity, scenario, [])
        standard = build_cecl_reserve_basis(borrowers, scenario, [])
        targeted = build_cecl_reserve_basis(identity, scenario, [])

        for basis in (standard, targeted):
            self.assertEqual(basis.ratios.iloc[0]["status"], "unavailable")
            self.assertEqual(
                basis.ratios.iloc[0]["exception_code"],
                "CECL_BALANCE_INVALID",
            )
            self.assertEqual(
                int(basis.ratios.iloc[0]["invalid_balance_count"]), 1
            )
            self.assertTrue(basis.effective_reserve.isna().all())

    def test_engine_rejects_null_cecl_object_before_execution(self):
        scenario = _scenario()
        scenario["cecl"] = None

        with self.assertRaisesRegex(
            ValueError, "Scenario cecl must be a JSON object"
        ):
            StressEngine(scenario, ".")

    def test_targeted_mode_central_tendency_uses_borrower_observations(self):
        # B1 has two loan rows. At borrower grain its combined 25.5% ratio has
        # z=2 and remains inside a 2.1 threshold. At loan grain the 50% loan has
        # z=sqrt(5)>2.1 and would be incorrectly removed.
        results = _commercial_frame(
            [100.0] * 6,
            [1.0, 50.0, 1.0, 1.0, 1.0, 1.0],
        )
        results["borrower_id"] = ["B1", "B1", "B2", "B3", "B4", "B5"]
        scenario = _scenario(
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 2.1},
            }
        )
        scenario["_targeted_mode"] = True

        basis = build_cecl_reserve_basis(results, scenario, [])

        self.assertAlmostEqual(float(basis.ratios.iloc[0]["reserve_ratio"]), 0.059)
        np.testing.assert_allclose(
            basis.effective_reserve.to_numpy(), np.full(6, 5.9)
        )
        self.assertEqual(int(basis.audit.iloc[0]["observation_count"]), 5)
        self.assertEqual(
            basis.audit.iloc[0]["observation_grain"], "borrower"
        )


class CeclReserveBasisReportingTest(unittest.TestCase):
    def test_aggregated_invalid_balance_cannot_become_zero_balance_cecl(self):
        scenario = _scenario({"method": "in_place"})
        identity = pd.DataFrame(
            [
                {
                    "_source_row": 1,
                    "borrower_id": "B1",
                    "balance": np.inf,
                    "cecl_portfolio": "CRE",
                    "model_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "cecl_reserve": 10.0,
                    "module_applied": "CRE",
                }
            ]
        )
        borrowers = build_borrowers(identity, scenario, [])
        basis = build_cecl_reserve_basis(borrowers, scenario, [])
        bucket_summary = build_cecl_bucket_summary(
            borrowers, pd.DataFrame(), scenario
        )

        cecl = build_cecl_summary(
            borrowers, bucket_summary, scenario, [], basis
        )
        base = cecl[
            (cecl["portfolio"] == "CRE")
            & (cecl["stress_level"] == "Base")
            & (cecl["bucket"] == "Total")
        ].iloc[0]
        aggregate = cecl[
            (cecl["portfolio"] == "Aggregate")
            & (cecl["stress_level"] == "Base")
        ].iloc[0]

        self.assertTrue(pd.isna(borrowers.iloc[0]["balance"]))
        self.assertEqual(
            float(bucket_summary.iloc[0]["balance"]), 0.0
        )
        self.assertEqual(base["cecl_reserve_status"], "unavailable")
        self.assertEqual(base["exception_code"], "CECL_BALANCE_INVALID")
        self.assertTrue(pd.isna(base["proforma_cecl_reserve"]))
        self.assertEqual(aggregate["cecl_reserve_status"], "unavailable")

    def test_nonfinite_commercial_balance_cannot_report_available_cecl(self):
        scenario = _scenario(
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 2.0},
            }
        )
        results = _commercial_frame([np.inf], [10.0])
        bucket_summary = pd.DataFrame(
            [
                {
                    "portfolio": "CRE",
                    "stress_level": "Base",
                    "bucket": "Pass",
                    "balance": np.inf,
                }
            ]
        )

        cecl = build_cecl_summary(results, bucket_summary, scenario, [])
        base = cecl[
            (cecl["portfolio"] == "CRE")
            & (cecl["stress_level"] == "Base")
            & (cecl["bucket"] == "Total")
        ].iloc[0]
        aggregate = cecl[
            (cecl["portfolio"] == "Aggregate")
            & (cecl["stress_level"] == "Base")
        ].iloc[0]

        self.assertEqual(base["cecl_reserve_status"], "unavailable")
        self.assertTrue(pd.isna(base["proforma_cecl_reserve"]))
        self.assertEqual(aggregate["cecl_reserve_status"], "unavailable")

    def test_negative_commercial_and_consumer_balances_are_unavailable(self):
        scenario = _scenario({"method": "central_tendency"})
        commercial = _commercial_frame([-100.0], [10.0])
        commercial_buckets = pd.DataFrame(
            [
                {
                    "portfolio": "CRE",
                    "stress_level": "Base",
                    "bucket": "Pass",
                    "balance": -100.0,
                }
            ]
        )
        commercial_cecl = build_cecl_summary(
            commercial, commercial_buckets, scenario, []
        )
        commercial_total = commercial_cecl[
            (commercial_cecl["portfolio"] == "CRE")
            & (commercial_cecl["stress_level"] == "Base")
            & (commercial_cecl["bucket"] == "Total")
        ].iloc[0]

        scenario["cecl"]["portfolios"] = {
            "Consumer": {"method": "expected_loss"}
        }
        consumer = _commercial_frame(
            [-100.0], [10.0], portfolio="Consumer"
        )
        consumer["model_portfolio"] = "Consumer"
        consumer["module_applied"] = "Consumer"
        consumer["consumer_el_unstressed"] = 5.0
        consumer_cecl = build_cecl_summary(
            consumer,
            pd.DataFrame([{"portfolio": "Consumer"}]),
            scenario,
            [],
        )
        consumer_total = consumer_cecl[
            (consumer_cecl["portfolio"] == "Consumer")
            & (consumer_cecl["stress_level"] == "Base")
        ].iloc[0]

        for row in (commercial_total, consumer_total):
            self.assertEqual(row["cecl_reserve_status"], "unavailable")
            self.assertEqual(row["exception_code"], "CECL_BALANCE_INVALID")
            self.assertTrue(pd.isna(row["proforma_cecl_reserve"]))

    def test_zero_balance_consumer_bucket_does_not_block_valid_portfolio(self):
        scenario = _scenario(
            {
                "method": "central_tendency",
                "central_tendency": {"z_score_threshold": 2.0},
            }
        )
        scenario["cecl"]["portfolios"] = {
            "Consumer": {"method": "expected_loss"}
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "C1",
                    "balance": 100.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "base_bucket": "Pass",
                    "cecl_reserve": 10.0,
                    "module_applied": "Consumer",
                    "consumer_el_unstressed": 4.0,
                },
                {
                    "borrower_id": "C2",
                    "balance": 0.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "base_bucket": "Substandard",
                    "cecl_reserve": 0.0,
                    "module_applied": "Consumer",
                    "consumer_el_unstressed": 0.0,
                },
            ]
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
        total = cecl[
            (cecl["portfolio"] == "Consumer") & (cecl["bucket"] == "Total")
        ].set_index("stress_level")

        self.assertEqual(float(consumer.at["Base", "proforma_cecl_reserve"]), 10.0)
        self.assertEqual(float(total.at["Base", "proforma_cecl_reserve"]), 10.0)
        self.assertEqual(total.at["Base", "cecl_reserve_status"], "available")

    def test_consumer_module_uses_the_selected_weighted_base(self):
        scenario = _scenario(_weighted_basis())
        scenario["stress_levels"] = ["S1"]
        scenario["modules"] = {
            "Consumer": {
                "pd_lookup_source": "fico_pd_lookup",
                "fico_field": "fico_score",
                "appraisal_field": "appraised_value",
                "pd_increase_factor": {"S1": 1.0},
                "collateral_value_factor": {"S1": 0.9},
                "rushed_sale_discount": 0.0,
                "closing_costs": 0.0,
                "pd_cap": 1.0,
            }
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "C1",
                    "balance": 100.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "base_bucket": "Pass",
                    "primary_module": "Consumer",
                    "module_applied": "",
                    "out_of_scope_S1": False,
                    "cecl_reserve": 10.0,
                    "cecl_reserve_prior_1": 20.0,
                    "cecl_reserve_prior_2": 30.0,
                    "fico_score": 700.0,
                    "appraised_value": 100.0,
                }
            ]
        )
        inputs = {
            "fico_pd_lookup": SimpleNamespace(
                frame=pd.DataFrame(
                    {"min_score": [600], "max_score": [800], "pd": [0.02]}
                )
            )
        }

        prepared, basis = attach_cecl_reserve_basis(results, scenario, [])
        stressed, out_of_scope = run_consumer(prepared, scenario, inputs, [])

        self.assertTrue(out_of_scope.empty)
        self.assertAlmostEqual(float(basis.effective_reserve.iloc[0]), 20.0)
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_cecl_reserve_base"]), 20.0
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_qualitative_reserve"]), 20.0
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_proforma_cecl_S1"]),
            float(stressed.at[0, "consumer_el_S1"]) + 20.0,
        )

    def test_weighted_history_consumer_reconciles_and_remains_monotonic(self):
        scenario = _scenario(_weighted_basis())
        scenario["cecl"]["portfolios"] = {
            "Consumer": {"method": "expected_loss"}
        }
        results = pd.DataFrame(
            [
                {
                    "borrower_id": "C1",
                    "balance": 100.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "base_bucket": "Pass",
                    "cecl_reserve": 10.0,
                    "cecl_reserve_prior_1": 20.0,
                    "cecl_reserve_prior_2": 30.0,
                    "module_applied": "Consumer",
                    "consumer_el_unstressed": 8.0,
                    "consumer_el_S1": 12.0,
                    "consumer_el_S2": 11.0,
                    "consumer_qualitative_reserve": 2.0,
                    "consumer_proforma_cecl_S1": 14.0,
                    "consumer_proforma_cecl_S2": 13.0,
                    "out_of_scope_S1": False,
                    "out_of_scope_S2": False,
                },
                {
                    "borrower_id": "C2",
                    "balance": 100.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "base_bucket": "Pass",
                    "cecl_reserve": 20.0,
                    "cecl_reserve_prior_1": 30.0,
                    "cecl_reserve_prior_2": 40.0,
                    "module_applied": "Consumer",
                    "consumer_el_unstressed": 15.0,
                    "consumer_el_S1": np.nan,
                    "consumer_el_S2": 18.0,
                    "consumer_qualitative_reserve": 5.0,
                    "consumer_proforma_cecl_S1": np.nan,
                    "consumer_proforma_cecl_S2": 23.0,
                    "out_of_scope_S1": True,
                    "out_of_scope_S2": False,
                },
            ]
        )

        consumer_summary = build_consumer_summary(results, scenario).set_index(
            "stress_level"
        )
        cecl = build_cecl_summary(
            results,
            pd.DataFrame([{"portfolio": "Consumer"}]),
            scenario,
            [],
        )
        consumer_cecl = cecl[
            (cecl["portfolio"] == "Consumer") & (cecl["bucket"] == "Total")
        ].set_index("stress_level")

        expected = {
            "Base": (23.0, 27.0, 50.0),
            "S1": (27.0, 27.0, 54.0),
            "S2": (30.0, 27.0, 57.0),
        }
        for level, (quantitative, qualitative, proforma) in expected.items():
            self.assertAlmostEqual(
                float(consumer_summary.at[level, "expected_loss"]), quantitative
            )
            self.assertAlmostEqual(
                float(consumer_summary.at[level, "qualitative_reserve"]), qualitative
            )
            self.assertAlmostEqual(
                float(consumer_summary.at[level, "proforma_cecl_reserve"]), proforma
            )
            self.assertAlmostEqual(
                quantitative + qualitative,
                float(consumer_summary.at[level, "proforma_cecl_reserve"]),
            )
            self.assertAlmostEqual(
                float(consumer_cecl.at[level, "proforma_cecl_reserve"]), proforma
            )
        self.assertEqual(
            list(consumer_cecl["proforma_cecl_reserve"]), [50.0, 54.0, 57.0]
        )
        self.assertEqual(float(results["cecl_reserve"].sum()), 30.0)


if __name__ == "__main__":
    unittest.main()
