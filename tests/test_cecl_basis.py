from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import pandas as pd

from stress_engine.borrower import build_borrowers, record_identity_data_issues
from stress_engine.cecl import (
    attach_cecl_reserve_basis,
    build_cecl_reserve_basis,
    cecl_history_frame,
)
from stress_engine.config import validate_scenario
from stress_engine.engine import StressEngine
from stress_engine.io import load_inputs
from stress_engine.modules.consumer import run_consumer
from stress_engine.reporting import (
    build_cecl_bucket_summary,
    build_cecl_summary,
    build_consumer_summary,
)


def _scenario(reserve_basis: dict | None = None) -> dict:
    cecl = {
        "reserve_field": "cecl_reserve",
        "portfolio_field": "cecl_portfolio",
    }
    if reserve_basis is not None:
        cecl["reserve_basis"] = reserve_basis
    history_enabled = (
        isinstance(reserve_basis, dict)
        and isinstance(reserve_basis.get("historical"), dict)
        and reserve_basis["historical"].get("enabled") is True
    )
    tags = (
        {
            "CRE": {
                "model_eligible": False,
                "cecl_level": True,
                "cecl_module": "Overlay",
                "include": [],
            }
        }
        if history_enabled
        else {}
    )
    return {
        "inputs": {
            "identity": {
                "column_aliases": {
                    "borrower_id": "borrower_id",
                    "balance": "balance",
                    "cecl_portfolio": "cecl_portfolio",
                    "cecl_reserve": "cecl_reserve",
                },
                "numeric_columns": ["balance", "cecl_reserve"],
                "required_columns": [
                    "borrower_id",
                    "balance",
                    "cecl_portfolio",
                    "cecl_reserve",
                ],
            },
            "sources": {},
        },
        "borrower": {
            "borrower_id_field": "borrower_id",
            "balance_field": "balance",
            "portfolio_field": "model_portfolio",
            "sum_fields": ["balance", "cecl_reserve"],
        },
        "tags": tags,
        "modules": {},
        "stress_levels": ["S1", "S2"],
        "cecl": cecl,
    }


def _history_source_spec() -> dict:
    return {
        "path": "cecl_history.csv",
        "type": "csv",
        "merge": False,
        "column_aliases": {
            "cecl_tag": "cecl_tag",
            "period": "period",
            "risk_bucket": "risk_bucket",
            "historical_cecl_ratio": "historical_cecl_ratio",
        },
        "string_columns": ["cecl_tag", "period", "risk_bucket"],
        "numeric_columns": ["historical_cecl_ratio"],
        "required_columns": [
            "cecl_tag",
            "period",
            "risk_bucket",
            "historical_cecl_ratio",
        ],
    }


def _with_history_source(scenario: dict) -> dict:
    scenario["inputs"]["sources"]["cecl_history"] = _history_source_spec()
    return scenario


def _history_frame(
    rows: list[tuple[object, object, object, object]],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "cecl_tag",
            "period",
            "risk_bucket",
            "historical_cecl_ratio",
        ],
    )


def _history_period(
    tag: object,
    period: object,
    pass_ratio: object,
    special_mention_ratio: object | None = None,
    substandard_ratio: object | None = None,
) -> list[tuple[object, object, object, object]]:
    """Return one complete tag-period risk ladder for test inputs."""
    special = (
        pass_ratio
        if special_mention_ratio is None
        else special_mention_ratio
    )
    substandard = (
        special if substandard_ratio is None else substandard_ratio
    )
    return [
        (tag, period, "Pass", pass_ratio),
        (tag, period, "Special Mention", special),
        (tag, period, "Substandard", substandard),
    ]


def _build_basis(
    results: pd.DataFrame,
    scenario: dict,
    history_rows: list[tuple[object, object, object, object]],
    exceptions: list[dict] | None = None,
):
    return build_cecl_reserve_basis(
        results,
        scenario,
        exceptions if exceptions is not None else [],
        history=_history_frame(history_rows),
    )


def _commercial_frame(
    balances: list[float],
    reserves: list[float],
    *,
    portfolio: str = "CRE",
    bucket: str = "Pass",
    cecl_tag: str | None = None,
) -> pd.DataFrame:
    count = len(balances)
    resolved_tag = cecl_tag if cecl_tag is not None else portfolio
    return pd.DataFrame(
        {
            "borrower_id": [f"B{index + 1}" for index in range(count)],
            "balance": balances,
            "cecl_portfolio": [portfolio] * count,
            "cecl_level_tag": [resolved_tag] * count,
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
    current_method: str = "in_place",
    z_score_threshold: float | None = None,
) -> dict:
    config: dict = {
        "current_method": current_method,
        "historical": {
            "enabled": True,
            "source": "cecl_history",
            "tag_field": "cecl_tag",
            "period_field": "period",
            "bucket_field": "risk_bucket",
            "ratio_field": "historical_cecl_ratio",
            "current_period": {
                "name": "2026Q2",
                "weight": weights[0],
            },
            "periods": [
                {
                    "name": "2026Q1",
                    "weight": weights[1],
                },
                {
                    "name": "2025Q4",
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
            _scenario({"current_method": "in_place"}),
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
                "current_method": "central_tendency",
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
                "current_method": "central_tendency",
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
                "current_method": "central_tendency",
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
                "current_method": "central_tendency",
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

    def test_central_tendency_isolated_by_cecl_tag_and_base_bucket(self):
        results = pd.concat(
            [
                _commercial_frame(
                    [100.0, 100.0],
                    [1.0, 1.0],
                    portfolio="Shared",
                    bucket="Pass",
                    cecl_tag="Tag A",
                ),
                _commercial_frame(
                    [100.0, 100.0],
                    [5.0, 5.0],
                    portfolio="Shared",
                    bucket="Substandard",
                    cecl_tag="Tag A",
                ),
                _commercial_frame(
                    [100.0, 100.0],
                    [8.0, 8.0],
                    portfolio="Shared",
                    bucket="Pass",
                    cecl_tag="Tag B",
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = [f"B{index}" for index in results.index]
        scenario = _scenario(
            {
                "current_method": "central_tendency",
                "central_tendency": {"z_score_threshold": 2.0},
            }
        )

        basis = build_cecl_reserve_basis(results, scenario, [])
        rates = basis.ratios.set_index(
            ["cecl_level_tag", "bucket"]
        )["reserve_ratio"]

        self.assertAlmostEqual(float(rates[("Tag A", "Pass")]), 0.01)
        self.assertAlmostEqual(
            float(rates[("Tag A", "Substandard")]), 0.05
        )
        self.assertAlmostEqual(float(rates[("Tag B", "Pass")]), 0.08)
        self.assertEqual(set(basis.ratios["portfolio"]), {"Shared"})

    def test_history_uses_direct_tag_bucket_ratios_and_trims_only_current(self):
        pass_rows = _commercial_frame(
            [100.0] * 6,
            [1.0, 1.0, 1.0, 1.0, 1.0, 50.0],
            bucket="Pass",
        )
        substandard = _commercial_frame(
            [200.0], [10.0], bucket="Substandard"
        )
        results = pd.concat([pass_rows, substandard], ignore_index=True)
        results["borrower_id"] = [f"B{index}" for index in results.index]
        scenario = _with_history_source(
            _scenario(
                _weighted_basis(
                    (0.5, 0.3, 0.2),
                    current_method="central_tendency",
                    z_score_threshold=2.0,
                )
            )
        )

        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", 0.02, 0.03, 0.06)
            + _history_period("CRE", "2025Q4", 0.04, 0.05, 0.08),
        )
        rates = basis.ratios.set_index("bucket")["reserve_ratio"]

        # Historical ratios are consumed as supplied at tag x bucket grain;
        # they are neither divided by the $800 tag balance nor z-trimmed.
        self.assertAlmostEqual(float(rates["Pass"]), 0.019)
        self.assertAlmostEqual(float(rates["Substandard"]), 0.059)
        self.assertAlmostEqual(
            float(basis.effective_reserve.iloc[:6].sum()), 11.4
        )
        self.assertAlmostEqual(float(basis.effective_reserve.iloc[6]), 11.8)
        self.assertAlmostEqual(float(basis.effective_reserve.sum()), 23.2)

        current_pass = basis.audit[
            (basis.audit["bucket"] == "Pass")
            & (basis.audit["period"] == "2026Q2")
        ].iloc[0]
        self.assertEqual(int(current_pass["observation_count"]), 6)
        self.assertEqual(int(current_pass["included_observation_count"]), 5)
        self.assertEqual(int(current_pass["excluded_observation_count"]), 1)
        history = basis.audit[
            basis.audit["period"].isin(["2026Q1", "2025Q4"])
            & basis.audit["bucket"].isin(["Pass", "Substandard"])
        ]
        self.assertEqual(
            set(history["period_reserve_ratio"]),
            {0.02, 0.04, 0.06, 0.08},
        )
        self.assertTrue(history["excluded_observation_count"].fillna(0).eq(0).all())

    def test_history_current_method_toggle_changes_only_current_quarter(self):
        results = _commercial_frame(
            [100.0] * 6,
            [1.0, 1.0, 1.0, 1.0, 1.0, 50.0],
        )
        history = _history_period("CRE", "2026Q1", 0.02) + _history_period(
            "CRE", "2025Q4", 0.04
        )
        central_scenario = _with_history_source(
            _scenario(
                _weighted_basis(
                    (0.5, 0.3, 0.2),
                    current_method="central_tendency",
                    z_score_threshold=2.0,
                )
            )
        )
        in_place_scenario = _with_history_source(
            _scenario(_weighted_basis((0.5, 0.3, 0.2)))
        )

        central = _build_basis(results, central_scenario, history)
        in_place = _build_basis(results, in_place_scenario, history)

        self.assertAlmostEqual(
            float(central.ratios.iloc[0]["reserve_ratio"]), 0.019
        )
        self.assertAlmostEqual(
            float(in_place.ratios.iloc[0]["reserve_ratio"]),
            0.059833333333333336,
        )
        for basis in (central, in_place):
            historical = basis.audit[basis.audit["period"] != "2026Q2"]
            self.assertEqual(
                set(historical["period_reserve_ratio"]),
                {0.02, 0.04},
            )

    def test_history_supports_arbitrary_period_count_and_weights(self):
        reserve_basis = _weighted_basis((0.1, 0.2, 0.3))
        reserve_basis["historical"]["periods"].append(
            {"name": "2025Q3", "weight": 0.4}
        )
        scenario = _with_history_source(_scenario(reserve_basis))
        results = _commercial_frame([100.0], [1.0])

        validate_scenario(scenario)
        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", 0.02)
            + _history_period("CRE", "2025Q4", 0.03)
            + _history_period("CRE", "2025Q3", 0.04),
        )

        self.assertEqual(set(basis.audit["period"]), {"2026Q2", "2026Q1", "2025Q4", "2025Q3"})
        self.assertAlmostEqual(float(basis.ratios.iloc[0]["reserve_ratio"]), 0.03)
        self.assertAlmostEqual(float(basis.effective_reserve.iloc[0]), 3.0)

    def test_missing_commercial_history_period_is_unavailable_without_reweighting(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = _commercial_frame([100.0], [10.0])
        exceptions: list[dict] = []

        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", 0.20),
            exceptions,
        )

        self.assertTrue(basis.effective_reserve.isna().all())
        self.assertTrue(basis.ratios["reserve_ratio"].isna().all())
        self.assertTrue(basis.ratios["status"].eq("unavailable").all())
        self.assertIn(
            "CECL_HISTORY_TAG_BUCKET_PERIOD_MISSING",
            {row["code"] for row in exceptions},
        )

    def test_duplicate_tag_period_bucket_history_rows_are_rejected(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = _commercial_frame([100.0], [10.0])

        with self.assertRaisesRegex(ValueError, "duplicate|unique"):
            _build_basis(
                results,
                scenario,
                _history_period("CRE", "2026Q1", 0.02)
                + [("CRE", "2026Q1", "Pass", 0.02)]
                + _history_period("CRE", "2025Q4", 0.04),
            )

    def test_invalid_history_ratios_make_only_affected_tag_unavailable(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = pd.concat(
            [
                _commercial_frame(
                    [100.0], [1.0], portfolio="Shared", cecl_tag="CRE Tag"
                ),
                _commercial_frame(
                    [100.0], [3.0], portfolio="Shared", cecl_tag="C&I Tag"
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = ["CRE-1", "CI-1"]
        invalid_values = (
            -1.0,
            1.01,
            np.inf,
            np.nan,
            True,
            "",
            "NA",
            "not-a-ratio",
        )
        for value in invalid_values:
            with self.subTest(historical_cecl_ratio=value):
                exceptions: list[dict] = []
                basis = _build_basis(
                    results,
                    scenario,
                    _history_period("CRE Tag", "2026Q1", value)
                    + _history_period("CRE Tag", "2025Q4", 0.04)
                    + _history_period("C&I Tag", "2026Q1", 0.02)
                    + _history_period("C&I Tag", "2025Q4", 0.04),
                    exceptions,
                )
                ratios = basis.ratios[basis.ratios["bucket"] == "Pass"].set_index(
                    "cecl_level_tag"
                )
                self.assertEqual(ratios.at["CRE Tag", "status"], "unavailable")
                self.assertEqual(ratios.at["C&I Tag", "status"], "available")
                self.assertTrue(any("HISTORY" in row["code"] for row in exceptions))

    def test_history_ratio_percent_text_is_parsed_as_decimal(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = _commercial_frame([100.0], [10.0])

        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", "2%", "3%", "4%")
            + _history_period("CRE", "2025Q4", "4%", "5%", "6%"),
        )

        passed = basis.ratios[basis.ratios["bucket"] == "Pass"].iloc[0]
        self.assertAlmostEqual(float(passed["reserve_ratio"]), 0.16 / 3)
        history = basis.audit[
            (basis.audit["bucket"] == "Pass")
            & basis.audit["period"].isin(["2026Q1", "2025Q4"])
        ]
        self.assertEqual(set(history["period_reserve_ratio"]), {0.02, 0.04})

    def test_na_history_ratio_skips_exact_cell_and_reweights(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = pd.concat(
            [
                _commercial_frame([100.0], [10.0], bucket="Pass"),
                _commercial_frame(
                    [100.0], [20.0], bucket="Special Mention"
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = ["P1", "SM1"]
        exceptions: list[dict] = []

        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", "N/A", 0.20, 0.30)
            + _history_period("CRE", "2025Q4", 0.30, 0.30, 0.30),
            exceptions,
        )

        ratios = basis.ratios.set_index("bucket")["reserve_ratio"]
        self.assertAlmostEqual(float(ratios["Pass"]), 0.20)
        self.assertAlmostEqual(
            float(ratios["Special Mention"]), (0.20 + 0.20 + 0.30) / 3
        )
        pass_audit = basis.audit[basis.audit["bucket"] == "Pass"].set_index(
            "period"
        )
        self.assertTrue(pass_audit["weight"].eq(1 / 3).all())
        self.assertAlmostEqual(
            float(pass_audit.at["2026Q2", "effective_weight"]), 0.5
        )
        self.assertEqual(
            float(pass_audit.at["2026Q1", "effective_weight"]), 0.0
        )
        self.assertAlmostEqual(
            float(pass_audit.at["2025Q4", "effective_weight"]), 0.5
        )
        self.assertEqual(pass_audit.at["2026Q1", "status"], "skipped")
        self.assertEqual(
            float(pass_audit.at["2026Q1", "weighted_ratio_component"]),
            0.0,
        )
        self.assertTrue(pass_audit["basis_status"].eq("available").all())
        sm_audit = basis.audit[
            basis.audit["bucket"] == "Special Mention"
        ]
        self.assertTrue(sm_audit["effective_weight"].eq(1 / 3).all())
        warnings = [
            row
            for row in exceptions
            if row["code"] == "CECL_HISTORY_RATIO_SKIPPED_REWEIGHTED"
        ]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["severity"], "WARNING")
        self.assertEqual(warnings[0]["portfolio"], "CRE")
        self.assertEqual(warnings[0]["bucket"], "Pass")
        self.assertIn("cecl_level_tag=CRE", warnings[0]["details"])
        self.assertIn("skipped_periods=2026Q1", warnings[0]["details"])

    def test_all_na_history_ratios_reweight_to_current_quarter(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = _commercial_frame([100.0], [10.0])

        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", "N/A")
            + _history_period("CRE", "2025Q4", "#N/A"),
        )

        passed = basis.ratios[basis.ratios["bucket"] == "Pass"].iloc[0]
        self.assertEqual(passed["status"], "available")
        self.assertAlmostEqual(float(passed["reserve_ratio"]), 0.10)
        audit = basis.audit[basis.audit["bucket"] == "Pass"].set_index(
            "period"
        )
        self.assertEqual(float(audit.at["2026Q2", "effective_weight"]), 1.0)
        self.assertEqual(float(audit.at["2026Q1", "effective_weight"]), 0.0)
        self.assertEqual(float(audit.at["2025Q4", "effective_weight"]), 0.0)

    def test_decreasing_history_ladder_warns_and_remains_available(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = pd.concat(
            [
                _commercial_frame([100.0], [1.0], bucket="Pass"),
                _commercial_frame(
                    [100.0], [2.0], bucket="Special Mention"
                ),
                _commercial_frame(
                    [100.0], [3.0], bucket="Substandard"
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = ["P1", "SM1", "SS1"]
        exceptions: list[dict] = []

        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", 0.03, 0.02, 0.04)
            + _history_period("CRE", "2025Q4", 0.04, 0.05, 0.06),
            exceptions,
        )

        ratios = basis.ratios.set_index("bucket")
        self.assertTrue(ratios["status"].eq("available").all())
        self.assertTrue(ratios["exception_code"].eq("").all())
        self.assertTrue(basis.audit["basis_status"].eq("available").all())
        self.assertTrue(basis.audit["basis_exception_code"].eq("").all())
        self.assertAlmostEqual(
            float(ratios.at["Pass", "reserve_ratio"]), 0.08 / 3
        )
        self.assertAlmostEqual(
            float(ratios.at["Special Mention", "reserve_ratio"]), 0.03
        )
        self.assertAlmostEqual(
            float(ratios.at["Substandard", "reserve_ratio"]), 0.13 / 3
        )
        event = next(
            row
            for row in exceptions
            if row["code"] == "CECL_HISTORY_RATIO_LADDER_INVALID"
        )
        self.assertEqual(event["severity"], "WARNING")
        self.assertIn("period=2026Q1", event["details"])
        self.assertIn("Pass=0.03", event["details"])
        self.assertIn("Special Mention=0.02", event["details"])
        self.assertIn("decreases=Pass>Special Mention", event["details"])

    def test_skipped_ladder_cell_still_warns_without_blocking(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = pd.concat(
            [
                _commercial_frame([100.0], [1.0], bucket="Pass"),
                _commercial_frame(
                    [100.0], [2.0], bucket="Special Mention"
                ),
                _commercial_frame(
                    [100.0], [3.0], bucket="Substandard"
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = ["P1", "SM1", "SS1"]
        exceptions: list[dict] = []

        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", 0.03, "N/A", 0.02)
            + _history_period("CRE", "2025Q4", 0.04, 0.05, 0.06),
            exceptions,
        )

        ratios = basis.ratios.set_index("bucket")
        self.assertTrue(ratios["status"].eq("available").all())
        self.assertAlmostEqual(
            float(ratios.at["Special Mention", "reserve_ratio"]), 0.035
        )
        event = next(
            row
            for row in exceptions
            if row["code"] == "CECL_HISTORY_RATIO_LADDER_INVALID"
        )
        self.assertEqual(event["severity"], "WARNING")
        self.assertIn("decreases=Pass>Substandard", event["details"])
        self.assertIn(
            "CECL_HISTORY_RATIO_SKIPPED_REWEIGHTED",
            {row["code"] for row in exceptions},
        )

    def test_decreasing_final_ladder_warns_and_remains_available(self):
        results = pd.concat(
            [
                _commercial_frame(
                    [100.0], [10.0], bucket="Pass", cecl_tag="CRE Tag"
                ),
                _commercial_frame(
                    [100.0], [1.0], bucket="Substandard", cecl_tag="CRE Tag"
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = ["B1", "B2"]
        results["stressed_bucket_S1"] = "Substandard"
        results["stressed_bucket_S2"] = "Substandard"
        exceptions: list[dict] = []

        basis = build_cecl_reserve_basis(results, _scenario(), exceptions)
        buckets = build_cecl_bucket_summary(
            results, pd.DataFrame(), _scenario()
        )
        cecl = build_cecl_summary(
            results, buckets, _scenario(), exceptions, basis
        )

        self.assertTrue(basis.ratios["status"].eq("available").all())
        self.assertTrue(basis.ratios["exception_code"].eq("").all())
        self.assertTrue(basis.audit["basis_status"].eq("available").all())
        self.assertTrue(basis.audit["basis_exception_code"].eq("").all())
        self.assertEqual(list(basis.effective_reserve), [10.0, 1.0])
        event = next(
            row
            for row in exceptions
            if row["code"] == "CECL_RESERVE_RATIO_LADDER_INVALID"
        )
        self.assertEqual(event["severity"], "WARNING")
        self.assertIn("Pass=0.1", event["details"])
        self.assertIn("Substandard=0.01", event["details"])
        self.assertIn("decreases=Pass>Substandard", event["details"])
        totals = cecl[
            cecl["portfolio"].eq("CRE") & cecl["bucket"].eq("Total")
        ].set_index("stress_level")
        self.assertTrue(totals["cecl_reserve_status"].eq("available").all())
        self.assertTrue(totals["exception_code"].eq("").all())
        self.assertTrue(totals["balance"].eq(200.0).all())
        self.assertEqual(
            list(totals.loc[["Base", "S1", "S2"], "proforma_cecl_reserve"]),
            [11.0, 2.0, 2.0],
        )

    def test_invalid_reserve_basis_configuration_is_rejected(self):
        invalid_configs = [
            {"current_method": "unknown"},
            {
                "current_method": "central_tendency",
                "central_tendency": {"z_score_threshold": 0.0},
            },
            {
                "current_method": "central_tendency",
                "central_tendency": {"z_score_threshold": np.inf},
            },
            {
                "current_method": "central_tendency",
                "central_tendency": {"z_score_threshold": True},
            },
            _weighted_basis((0.2, 0.3, 0.4)),
            _weighted_basis((0.0, 0.5, 0.5)),
            _weighted_basis((0.5, 0.5, 0.0)),
            _weighted_basis((-0.1, 0.6, 0.5)),
        ]
        for reserve_basis in invalid_configs:
            with self.subTest(reserve_basis=reserve_basis):
                with self.assertRaises(ValueError):
                    validate_scenario(
                        _with_history_source(_scenario(reserve_basis))
                    )

        duplicate_name = _weighted_basis()
        duplicate_name["historical"]["periods"][1]["name"] = "2026Q1"
        with self.assertRaises(ValueError):
            validate_scenario(_with_history_source(_scenario(duplicate_name)))

        current_name = _weighted_basis()
        current_name["historical"]["periods"][0]["name"] = "2026Q2"
        with self.assertRaises(ValueError):
            validate_scenario(_with_history_source(_scenario(current_name)))

    def test_history_requires_a_configured_cecl_level_tag(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        scenario["tags"] = {}

        with self.assertRaisesRegex(ValueError, "CECL-level.*tag"):
            validate_scenario(scenario)

        basis = build_cecl_reserve_basis(
            _commercial_frame([100.0], [10.0]),
            scenario,
            [],
            history=_history_frame(
                _history_period("CRE", "2026Q1", 0.10)
                + _history_period("CRE", "2025Q4", 0.10)
            ),
        )
        self.assertEqual(
            basis.exception_code, "CECL_LEVEL_TAG_CONFIGURATION_MISSING"
        )
        self.assertTrue(basis.effective_reserve.isna().all())

    def test_explicit_tag_scenario_rejects_missing_resolved_tag_field(self):
        scenario = _scenario()
        scenario["tags"] = {
            "CRE": {
                "model_eligible": False,
                "cecl_level": True,
                "cecl_module": "Overlay",
                "include": [],
            }
        }
        results = _commercial_frame([100.0], [10.0]).drop(
            columns=["cecl_level_tag"]
        )
        exceptions: list[dict] = []

        basis = build_cecl_reserve_basis(results, scenario, exceptions)

        self.assertEqual(basis.exception_code, "CECL_LEVEL_TAG_MISSING")
        self.assertTrue(basis.effective_reserve.isna().all())
        self.assertIn(
            "CECL_LEVEL_TAG_MISSING", {row["code"] for row in exceptions}
        )

    def test_history_source_requires_nonmerged_alias_and_type_wiring(self):
        mutations = (
            ("source", lambda scenario: scenario["cecl"]["reserve_basis"]["historical"].update(source="missing")),
            ("merge", lambda scenario: scenario["inputs"]["sources"]["cecl_history"].update(merge=True)),
            ("column_aliases", lambda scenario: scenario["inputs"]["sources"]["cecl_history"]["column_aliases"].pop("historical_cecl_ratio")),
            ("string_columns", lambda scenario: scenario["inputs"]["sources"]["cecl_history"]["string_columns"].remove("period")),
            ("numeric_columns", lambda scenario: scenario["inputs"]["sources"]["cecl_history"]["numeric_columns"].remove("historical_cecl_ratio")),
            ("date_columns", lambda scenario: scenario["inputs"]["sources"]["cecl_history"].update(date_columns=["historical_cecl_ratio"])),
            ("required_columns", lambda scenario: scenario["inputs"]["sources"]["cecl_history"]["required_columns"].remove("cecl_tag")),
        )
        for location, mutate in mutations:
            with self.subTest(location=location):
                scenario = _with_history_source(_scenario(_weighted_basis()))
                mutate(scenario)
                with self.assertRaises(ValueError):
                    validate_scenario(scenario)

    def test_legacy_loan_history_schema_is_rejected_with_migration_message(self):
        legacy = {
            "method": "weighted_history",
            "weighted_history": {
                "period_method": "central_tendency",
                "periods": [
                    {
                        "name": "prior",
                        "reserve_field": "cecl_reserve_prior",
                        "weight": 1.0,
                    }
                ],
            },
        }
        with self.assertRaisesRegex(ValueError, "histor|portfolio|CSV"):
            validate_scenario(_scenario(legacy))

    def test_old_portfolio_dollar_history_schema_is_rejected(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        historical = scenario["cecl"]["reserve_basis"]["historical"]
        historical.pop("tag_field")
        historical.pop("bucket_field")
        historical.pop("ratio_field")
        historical.update(
            {
                "portfolio_field": "cecl_portfolio",
                "reserve_field": "historical_cecl_reserve",
            }
        )

        with self.assertRaisesRegex(
            ValueError, "Portfolio-level historical CECL reserve inputs"
        ):
            validate_scenario(scenario)

    def test_explicit_basis_requires_current_reserve_input_wiring(self):
        for current_method in ("in_place", "central_tendency"):
            basis = {
                "current_method": current_method,
                "central_tendency": {"z_score_threshold": 2.0},
            }
            for location in (
                "column_aliases",
                "numeric_columns",
                "required_columns",
            ):
                with self.subTest(
                    current_method=current_method, location=location
                ):
                    scenario = _scenario(basis)
                    if location == "column_aliases":
                        scenario["inputs"]["identity"][location].pop(
                            "cecl_reserve"
                        )
                    else:
                        scenario["inputs"]["identity"][location].remove(
                            "cecl_reserve"
                        )
                    with self.assertRaises(ValueError):
                        validate_scenario(scenario)

    def test_current_reserve_is_always_summed_for_multi_loan_consumer(self):
        scenario = _with_history_source(
            _scenario(_weighted_basis(current_method="central_tendency"))
        )
        scenario["borrower"]["sum_fields"].remove("cecl_reserve")
        scenario["cecl"]["portfolios"] = {
            "Consumer": {"method": "expected_loss"}
        }
        identity = pd.DataFrame(
            [
                {
                    "_source_row": 1,
                    "borrower_id": "C1",
                    "balance": 100.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "base_bucket": "Pass",
                    "primary_module": "Consumer",
                    "cecl_reserve": 10.0,
                },
                {
                    "_source_row": 2,
                    "borrower_id": "C1",
                    "balance": 200.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "base_bucket": "Pass",
                    "primary_module": "Consumer",
                    "cecl_reserve": 20.0,
                },
            ]
        )

        validate_scenario(scenario)
        borrowers = build_borrowers(identity, scenario, [])
        basis = build_cecl_reserve_basis(
            borrowers, scenario, [], history=None
        )

        self.assertEqual(float(borrowers.at[0, "balance"]), 300.0)
        self.assertEqual(float(borrowers.at[0, "cecl_reserve"]), 30.0)
        self.assertEqual(float(basis.effective_reserve.iloc[0]), 30.0)
        self.assertEqual(basis.method_by_row.iloc[0], "in_place")

    def test_non_sum_current_reserve_aggregation_is_rejected(self):
        scenario = _scenario({"current_method": "in_place"})
        scenario["borrower"]["aggregation"] = {
            "cecl_reserve": "first"
        }

        with self.assertRaisesRegex(ValueError, "aggregation 'sum'"):
            validate_scenario(scenario)

    def test_current_cecl_fields_reject_loader_owned_names(self):
        for setting in ("reserve_field", "portfolio_field"):
            for field in (
                "_source_file",
                "_source_file_row",
                "_source_row",
                "_portfolio_key",
                "_period_key",
                "_cecl_invalid_balance_count",
                "cecl_effective_reserve_base",
                "cecl_reserve_basis_method",
                "_cecl_reserve_missing_count__custom",
                "base_bucket",
                "module_applied",
                "primary_module",
                "loan_count",
                "stressed_bucket_S1",
                "out_of_scope_S1",
                "tag_custom",
                "_source_custom",
                "_targeted_active",
                "_scenario_variant",
                "scenario_variant",
                "consumer_qualitative_reserve",
                "consumer_el_S1",
                "cre_dscr_S1",
                "ci_fccr_S1",
                "calculated_cash_paid_for_interest",
                "_exposure_id",
                "_loan_id_ambiguous",
            ):
                scenario = _scenario({"current_method": "in_place"})
                scenario["cecl"][setting] = field
                with self.subTest(setting=setting, field=field):
                    with self.assertRaisesRegex(
                        ValueError, "conflicts with"
                    ):
                        validate_scenario(scenario)

    def test_current_cecl_fields_reject_collisions_and_whitespace(self):
        for reserve_field, portfolio_field in (
            ("cecl_portfolio", "cecl_portfolio"),
            ("borrower_id", "cecl_portfolio"),
            ("balance", "cecl_portfolio"),
            ("model_portfolio", "cecl_portfolio"),
            ("cecl_reserve", "borrower_id"),
            ("cecl_reserve", "balance"),
            ("cecl_reserve", "model_module"),
        ):
            scenario = _scenario({"current_method": "in_place"})
            scenario["cecl"].update(
                {
                    "reserve_field": reserve_field,
                    "portfolio_field": portfolio_field,
                }
            )
            with self.subTest(
                reserve_field=reserve_field,
                portfolio_field=portfolio_field,
            ):
                with self.assertRaisesRegex(
                    ValueError, "distinct|conflicts with"
                ):
                    validate_scenario(scenario)

        for setting in ("reserve_field", "portfolio_field"):
            scenario = _scenario({"current_method": "in_place"})
            scenario["cecl"][setting] = " cecl_custom "
            with self.subTest(setting=setting, whitespace=True):
                with self.assertRaisesRegex(
                    ValueError, "without surrounding whitespace"
                ):
                    validate_scenario(scenario)

    def test_current_reserve_cannot_be_overwritten_by_tag_assignment(self):
        scenario = _scenario({"current_method": "in_place"})
        scenario["tags"] = {
            "Bad": {"assign": {"cecl_reserve": 999.0}}
        }

        with self.assertRaisesRegex(ValueError, "tag assignment target"):
            validate_scenario(scenario)

    def test_cecl_portfolio_may_reuse_borrower_portfolio_field(self):
        for portfolio_field in (
            "model_portfolio",
            "ci_portfolio",
            "cre_subsector",
            "consumer_segment",
            "overlay_custom",
            "calculated_custom",
            "_custom_portfolio",
        ):
            scenario = _scenario({"current_method": "in_place"})
            scenario["borrower"]["portfolio_field"] = portfolio_field
            scenario["cecl"]["portfolio_field"] = portfolio_field
            with self.subTest(portfolio_field=portfolio_field):
                validate_scenario(scenario)

    def test_consumer_only_load_skips_missing_history_file(self):
        scenario = _with_history_source(
            _scenario(_weighted_basis(current_method="central_tendency"))
        )
        scenario["cecl"]["portfolios"] = {
            "Consumer": {"method": "expected_loss"}
        }
        scenario["inputs"]["identity"]["path"] = "identity.csv"
        scenario["inputs"]["sources"]["cecl_history"]["path"] = (
            "missing_history.csv"
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "identity.csv").write_text(
                "borrower_id,balance,cecl_portfolio,cecl_reserve\n"
                "C1,100,Consumer,10\n",
                encoding="utf-8",
            )
            validate_scenario(scenario)
            loaded = load_inputs(scenario, directory)

        self.assertIn("identity", loaded)
        self.assertNotIn("cecl_history", loaded)
        borrowers = build_borrowers(loaded["identity"].frame, scenario, [])
        exceptions: list[dict] = []
        basis = build_cecl_reserve_basis(
            borrowers, scenario, exceptions, history=None
        )
        self.assertEqual(float(basis.effective_reserve.iloc[0]), 10.0)
        self.assertEqual(basis.method_by_row.iloc[0], "in_place")
        self.assertFalse(
            any("HISTORY" in str(row.get("code", "")) for row in exceptions)
        )

    def test_history_source_accepts_vendor_headers_for_canonical_aliases(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        aliases = scenario["inputs"]["sources"]["cecl_history"][
            "column_aliases"
        ]
        aliases.update(
            {
                "cecl_tag": "CECL Segment",
                "period": "Quarter",
                "risk_bucket": "Risk Bucket",
                "historical_cecl_ratio": "CECL Ratio",
            }
        )

        # column_aliases maps canonical engine names to vendor source headers;
        # validation must inspect the keys rather than requiring names to match.
        validate_scenario(scenario)

    def test_csv_loader_preserves_only_explicit_na_skip_tokens(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        scenario["inputs"]["identity"]["path"] = "identity.csv"
        scenario["inputs"]["sources"]["cecl_history"]["path"] = (
            "history.csv"
        )
        converter_calls: list[str] = []

        def ratio_converter(value: object) -> object:
            converter_calls.append(str(value))
            return value

        scenario["inputs"]["sources"]["cecl_history"]["read_options"] = {
            "na_values": ["N/A", "#N/A"],
            "dtype": "string",
            "converters": {
                "historical_cecl_ratio": ratio_converter
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "identity.csv").write_text(
                "borrower_id,balance,cecl_portfolio,cecl_reserve\n"
                "B1,100,CRE,10\n",
                encoding="utf-8",
            )
            (directory / "history.csv").write_text(
                "cecl_tag,period,risk_bucket,historical_cecl_ratio\n"
                "CRE,2026Q1,Pass,N/A\n"
                "CRE,2026Q1,Special Mention,NA\n"
                "CRE,2026Q1,Substandard,0.30\n"
                "CRE,2025Q4,Pass,0.40\n"
                "CRE,2025Q4,Special Mention,0.40\n"
                "CRE,2025Q4,Substandard,#N/A\n"
                "CRE,NA,Pass,0.99\n",
                encoding="utf-8",
            )
            validate_scenario(scenario)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_inputs(scenario, directory)

        self.assertNotIn("N/A", converter_calls)
        self.assertNotIn("#N/A", converter_calls)
        self.assertIn("0.40", converter_calls)
        self.assertFalse(
            any(
                "converter and dtype" in str(item.message).casefold()
                for item in caught
            )
        )

        history = cecl_history_frame(scenario, loaded)
        self.assertIsNotNone(history)
        q1 = history[history["period"].eq("2026Q1")].set_index(
            "risk_bucket"
        )
        self.assertEqual(q1.at["Pass", "historical_cecl_ratio"], "N/A")
        self.assertTrue(
            pd.isna(q1.at["Special Mention", "historical_cecl_ratio"])
        )
        q4 = history[history["period"].eq("2025Q4")].set_index(
            "risk_bucket"
        )
        self.assertEqual(
            q4.at["Substandard", "historical_cecl_ratio"], "N/A"
        )
        self.assertEqual(int(history["period"].isna().sum()), 1)
        issues = loaded["cecl_history"].coercion_issues
        self.assertEqual(sum(issue["count"] for issue in issues), 1)

        basis = build_cecl_reserve_basis(
            _commercial_frame([100.0], [10.0]),
            scenario,
            [],
            history=history,
        )
        passed = basis.ratios[basis.ratios["bucket"] == "Pass"].iloc[0]
        special = basis.ratios[
            basis.ratios["bucket"] == "Special Mention"
        ].iloc[0]
        self.assertAlmostEqual(float(passed["reserve_ratio"]), 0.25)
        self.assertEqual(special["status"], "unavailable")
        self.assertEqual(
            special["exception_code"], "CECL_HISTORY_RATIO_INVALID"
        )

    def test_history_normalizes_validated_weights_and_join_keys_at_runtime(self):
        reserve_basis = _weighted_basis()
        history = reserve_basis["historical"]
        history["current_period"].update(
            {"name": " 2026Q2 ", "weight": "50%"}
        )
        history["periods"][0].update({"name": " 2026Q1 ", "weight": "25%"})
        history["periods"][1].update({"name": " 2025Q4 ", "weight": "25%"})
        scenario = _with_history_source(_scenario(reserve_basis))
        results = _commercial_frame([100.0], [10.0])

        validate_scenario(scenario)
        basis = _build_basis(
            results,
            scenario,
            _history_period(" CRE ", " 2026Q1 ", 0.20)
            + _history_period(" CRE ", " 2025Q4 ", 0.30),
        )

        self.assertEqual(
            set(basis.audit["period"]), {"2026Q2", "2026Q1", "2025Q4"}
        )
        self.assertAlmostEqual(float(basis.effective_reserve.iloc[0]), 17.5)

    def test_nonfinite_loan_reserve_is_zeroed_before_borrower_sum(self):
        scenario = _scenario({"current_method": "in_place"})
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
        scenario = _scenario({"current_method": "in_place"})
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
        targeted_exceptions: list[dict] = []
        targeted = build_cecl_reserve_basis(
            identity, scenario, targeted_exceptions
        )

        self.assertEqual(float(borrowers.iloc[0]["balance"]), 100.0)
        self.assertEqual(float(borrowers.iloc[0]["cecl_reserve"]), 5.0)
        for basis in (standard, targeted):
            self.assertEqual(basis.ratios.iloc[0]["status"], "available")
            self.assertAlmostEqual(
                float(basis.ratios.iloc[0]["reserve_ratio"]), 0.05
            )
            self.assertEqual(
                int(basis.ratios.iloc[0]["invalid_balance_count"]), 0
            )
        self.assertAlmostEqual(float(standard.effective_reserve.sum()), 5.0)
        self.assertTrue(pd.isna(targeted.effective_reserve.iloc[0]))
        self.assertAlmostEqual(float(targeted.effective_reserve.iloc[1]), 5.0)
        self.assertIn(
            "CECL_BALANCE_EXCLUDED",
            {row["code"] for row in targeted_exceptions},
        )
        self.assertNotIn(
            "CECL_BALANCE_INVALID",
            {row["code"] for row in targeted_exceptions},
        )

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
                "current_method": "central_tendency",
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

    def test_tag_history_has_standard_and_targeted_borrower_grain_parity(self):
        scenario = _with_history_source(
            _scenario(
                _weighted_basis(
                    (0.5, 0.3, 0.2),
                    current_method="central_tendency",
                    z_score_threshold=2.0,
                )
            )
        )
        identity = pd.DataFrame(
            [
                {
                    "_source_row": index + 1,
                    "borrower_id": borrower,
                    "balance": balance,
                    "cecl_portfolio": "CRE",
                    "cecl_level_tag": "CRE",
                    "model_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "cecl_reserve": reserve,
                    "module_applied": "CRE",
                }
                for index, (borrower, balance, reserve) in enumerate(
                    [
                        ("B1", 50.0, 0.5),
                        ("B1", 50.0, 0.5),
                        ("B2", 100.0, 1.0),
                        ("B3", 100.0, 1.0),
                        ("B4", 100.0, 1.0),
                        ("B5", 100.0, 1.0),
                        ("B6", 100.0, 50.0),
                    ]
                )
            ]
        )
        borrowers = build_borrowers(identity, scenario, [])
        history = _history_period("CRE", "2026Q1", 0.02) + _history_period(
            "CRE", "2025Q4", 0.04
        )

        standard = _build_basis(borrowers, scenario, history)
        targeted_scenario = dict(scenario)
        targeted_scenario["_targeted_mode"] = True
        targeted = _build_basis(identity, targeted_scenario, history)

        for basis in (standard, targeted):
            self.assertAlmostEqual(
                float(basis.ratios.iloc[0]["reserve_ratio"]),
                0.019,
            )
            self.assertAlmostEqual(float(basis.effective_reserve.sum()), 11.4)
            current = basis.audit[basis.audit["period"] == "2026Q2"].iloc[0]
            self.assertEqual(int(current["observation_count"]), 6)
            self.assertEqual(int(current["included_observation_count"]), 5)
            self.assertEqual(int(current["excluded_observation_count"]), 1)


    def test_current_tag_whitespace_is_normalized_before_history_join(self):
        scenario = _scenario(_weighted_basis((0.5, 0.25, 0.25)))
        results = _commercial_frame([100.0, 100.0], [10.0, 10.0])
        results["cecl_portfolio"] = ["CRE", " CRE "]
        results["cecl_level_tag"] = ["CRE", " CRE "]

        prepared, basis = attach_cecl_reserve_basis(
            results,
            scenario,
            [],
            history=_history_frame(
                _history_period("CRE", "2026Q1", 0.10)
                + _history_period("CRE", "2025Q4", 0.10)
            ),
        )

        self.assertEqual(set(prepared["cecl_portfolio"]), {"CRE"})
        self.assertEqual(set(basis.ratios["cecl_level_tag"]), {"CRE"})
        self.assertEqual(len(basis.ratios), 3)
        self.assertAlmostEqual(float(basis.effective_reserve.sum()), 20.0)

    def test_history_field_names_must_be_distinct_and_noninternal(self):
        for fields in (
            {
                "tag_field": "x",
                "period_field": "x",
                "bucket_field": "x",
                "ratio_field": "x",
            },
            {"tag_field": "_portfolio_key"},
            {"period_field": "_source_file"},
            {"bucket_field": "_source_file_row"},
            {"ratio_field": "_source_row"},
        ):
            scenario = _with_history_source(
                _scenario(_weighted_basis())
            )
            scenario["cecl"]["reserve_basis"]["historical"].update(fields)
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(
                    ValueError, "distinct|reserved internal"
                ):
                    validate_scenario(scenario)

    def test_blank_history_keys_are_ignored_with_an_audit_warning(self):
        scenario = _scenario(_weighted_basis())
        results = _commercial_frame([100.0], [10.0])
        exceptions: list[dict] = []

        basis = _build_basis(
            results,
            scenario,
            [("", "2026Q1", "Pass", 0.99), ("CRE", "", "Pass", 0.99)]
            + _history_period("CRE", "2026Q1", 0.10)
            + _history_period("CRE", "2025Q4", 0.10),
            exceptions,
        )

        self.assertAlmostEqual(float(basis.effective_reserve.iloc[0]), 10.0)
        ignored = [
            row
            for row in exceptions
            if row["code"] == "CECL_HISTORY_ROW_IGNORED"
        ]
        self.assertEqual(len(ignored), 1)
        self.assertIn("ignored_count=2", ignored[0]["details"])


class CeclReserveBasisReportingTest(unittest.TestCase):
    def test_unknown_basis_is_unique_hidden_and_included_in_totals(self):
        scenario = _scenario()
        results = pd.concat(
            [
                _commercial_frame([100.0], [1.0], bucket="Pass"),
                _commercial_frame([50.0], [2.0], bucket="Unknown"),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = ["P1", "U1"]
        exceptions: list[dict] = []
        basis = build_cecl_reserve_basis(results, scenario, exceptions)
        buckets = build_cecl_bucket_summary(
            results, pd.DataFrame(), scenario
        )

        cecl = build_cecl_summary(
            results, buckets, scenario, exceptions, basis
        )

        unknown_basis = basis.ratios[basis.ratios["bucket"].eq("Unknown")]
        self.assertEqual(len(unknown_basis), 1)
        self.assertFalse(cecl["bucket"].eq("Unknown").any())
        for level in ("Base", "S1", "S2"):
            with self.subTest(level=level):
                portfolio_total = cecl[
                    cecl["portfolio"].eq("CRE")
                    & cecl["stress_level"].eq(level)
                    & cecl["bucket"].eq("Total")
                ].iloc[0]
                aggregate_total = cecl[
                    cecl["portfolio"].eq("Aggregate")
                    & cecl["stress_level"].eq(level)
                    & cecl["bucket"].eq("Total")
                ].iloc[0]
                for row in (portfolio_total, aggregate_total):
                    self.assertEqual(float(row["balance"]), 150.0)
                    self.assertEqual(
                        float(row["proforma_cecl_reserve"]), 3.0
                    )
                    self.assertEqual(
                        row["cecl_reserve_status"], "available"
                    )
                    self.assertEqual(row["exception_code"], "")
        self.assertNotIn(
            "CECL_RESERVE_RATIO_DUPLICATE",
            {row["code"] for row in exceptions},
        )

    def test_public_portfolio_rollup_applies_each_tag_ratio_first(self):
        scenario = _scenario()
        results = pd.concat(
            [
                _commercial_frame(
                    [100.0], [1.0], portfolio="C&I", cecl_tag="Tag A"
                ),
                _commercial_frame(
                    [100.0], [3.0], portfolio="C&I", cecl_tag="Tag B"
                ),
            ],
            ignore_index=True,
        )
        results["borrower_id"] = ["A1", "B1"]
        basis = build_cecl_reserve_basis(results, scenario, [])
        buckets = build_cecl_bucket_summary(
            results, pd.DataFrame(), scenario
        )

        cecl = build_cecl_summary(
            results, buckets, scenario, [], basis
        )

        base_pass = cecl[
            (cecl["portfolio"] == "C&I")
            & (cecl["stress_level"] == "Base")
            & (cecl["bucket"] == "Pass")
        ].iloc[0]
        self.assertAlmostEqual(float(base_pass["balance"]), 200.0)
        self.assertAlmostEqual(
            float(base_pass["proforma_cecl_reserve"]), 4.0
        )
        self.assertAlmostEqual(float(base_pass["reserve_ratio"]), 0.02)

    def test_reweighted_history_remains_available_in_public_totals(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
        results = _commercial_frame([100.0], [10.0])
        exceptions: list[dict] = []
        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", "N/A", 0.20, 0.30)
            + _history_period("CRE", "2025Q4", 0.30, 0.30, 0.30),
            exceptions,
        )
        buckets = build_cecl_bucket_summary(
            results, pd.DataFrame(), scenario
        )

        cecl = build_cecl_summary(
            results, buckets, scenario, exceptions, basis
        )

        for level in ("Base", "S1", "S2"):
            with self.subTest(level=level):
                pass_bucket = cecl[
                    (cecl["portfolio"] == "CRE")
                    & (cecl["stress_level"] == level)
                    & (cecl["bucket"] == "Pass")
                ].iloc[0]
                portfolio_total = cecl[
                    (cecl["portfolio"] == "CRE")
                    & (cecl["stress_level"] == level)
                    & (cecl["bucket"] == "Total")
                ].iloc[0]
                aggregate = cecl[
                    (cecl["portfolio"] == "Aggregate")
                    & (cecl["stress_level"] == level)
                ].iloc[0]
                self.assertEqual(
                    pass_bucket["cecl_reserve_status"], "available"
                )
                self.assertAlmostEqual(
                    float(pass_bucket["proforma_cecl_reserve"]), 20.0
                )
                self.assertAlmostEqual(
                    float(pass_bucket["reserve_ratio"]), 0.20
                )
                for row in (portfolio_total, aggregate):
                    self.assertEqual(
                        row["cecl_reserve_status"], "available"
                    )
                    self.assertAlmostEqual(
                        float(row["proforma_cecl_reserve"]), 20.0
                    )

    def test_all_invalid_balances_are_excluded_without_fake_zero_cecl(self):
        scenario = _scenario(_weighted_basis())
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
        borrowers["cecl_level_tag"] = "CRE"
        basis = _build_basis(
            borrowers,
            scenario,
            _history_period("CRE", "2026Q1", 0.10)
            + _history_period("CRE", "2025Q4", 0.10),
        )
        bucket_summary = build_cecl_bucket_summary(
            borrowers, pd.DataFrame(), scenario
        )

        cecl = build_cecl_summary(
            borrowers, bucket_summary, scenario, [], basis
        )
        self.assertTrue(borrowers.empty)
        self.assertTrue(basis.ratios.empty)
        self.assertTrue(bucket_summary.empty)
        self.assertTrue(cecl["portfolio"].eq("Aggregate").all())
        self.assertTrue(cecl["balance"].eq(0.0).all())
        self.assertTrue(cecl["proforma_cecl_reserve"].eq(0.0).all())
        self.assertTrue(cecl["cecl_reserve_status"].eq("available").all())
        self.assertTrue(cecl["exception_code"].eq("").all())

    def test_nonfinite_bucket_component_is_excluded_from_cecl(self):
        scenario = _scenario(
            {
                "current_method": "central_tendency",
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

        exceptions: list[dict] = []
        cecl = build_cecl_summary(
            results, bucket_summary, scenario, exceptions
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

        for row in (base, aggregate):
            self.assertEqual(row["cecl_reserve_status"], "available")
            self.assertEqual(row["exception_code"], "")
            self.assertEqual(float(row["balance"]), 0.0)
            self.assertEqual(float(row["proforma_cecl_reserve"]), 0.0)
        self.assertIn(
            "CECL_BUCKET_BALANCE_EXCLUDED",
            {row["code"] for row in exceptions},
        )

    def test_tolerance_sized_bucket_balance_is_normalized_to_zero(self):
        scenario = _scenario()
        scenario["cecl"]["zero_balance_tolerance"] = 1e-9
        results = _commercial_frame([100.0], [10.0])
        bucket_summary = pd.DataFrame(
            [
                {
                    "portfolio": "CRE",
                    "cecl_level_tag": "CRE",
                    "stress_level": "Base",
                    "bucket": "Pass",
                    "balance": -5e-10,
                }
            ]
        )
        exceptions: list[dict] = []

        cecl = build_cecl_summary(
            results, bucket_summary, scenario, exceptions
        )
        base_pass = cecl[
            (cecl["portfolio"] == "CRE")
            & (cecl["stress_level"] == "Base")
            & (cecl["bucket"] == "Pass")
        ].iloc[0]
        aggregate = cecl[
            (cecl["portfolio"] == "Aggregate")
            & (cecl["stress_level"] == "Base")
        ].iloc[0]

        for row in (base_pass, aggregate):
            self.assertEqual(float(row["balance"]), 0.0)
            self.assertEqual(float(row["proforma_cecl_reserve"]), 0.0)
            self.assertEqual(row["exception_code"], "")
        self.assertEqual(
            base_pass["cecl_reserve_status"],
            "not_applicable_zero_balance",
        )
        self.assertEqual(aggregate["cecl_reserve_status"], "available")
        self.assertNotIn(
            "CECL_BUCKET_BALANCE_EXCLUDED",
            {row["code"] for row in exceptions},
        )

    def test_empty_current_bucket_only_blocks_a_positive_stressed_balance(self):
        scenario = _scenario(_weighted_basis())
        results = _commercial_frame([0.0], [0.0])
        basis = _build_basis(
            results,
            scenario,
            _history_period("CRE", "2026Q1", 0.10, 0.20, 0.30)
            + _history_period("CRE", "2025Q4", 0.10, 0.20, 0.30),
        )
        cecl = build_cecl_summary(
            results,
            pd.DataFrame(
                [
                    {
                        "portfolio": "CRE",
                        "cecl_level_tag": "CRE",
                        "stress_level": "Base",
                        "bucket": "Pass",
                        "balance": 0.0,
                    },
                    {
                        "portfolio": "CRE",
                        "cecl_level_tag": "CRE",
                        "stress_level": "S1",
                        "bucket": "Special Mention",
                        "balance": 100.0,
                    },
                ]
            ),
            scenario,
            [],
            basis,
        )
        base = cecl[
            (cecl["portfolio"] == "CRE")
            & (cecl["stress_level"] == "Base")
            & (cecl["bucket"] == "Total")
        ].iloc[0]

        stressed = cecl[
            (cecl["portfolio"] == "CRE")
            & (cecl["stress_level"] == "S1")
            & (cecl["bucket"] == "Total")
        ].iloc[0]

        self.assertEqual(base["cecl_reserve_status"], "available")
        self.assertEqual(float(base["proforma_cecl_reserve"]), 0.0)
        self.assertEqual(base["exception_code"], "")
        self.assertEqual(stressed["cecl_reserve_status"], "unavailable")
        self.assertEqual(
            stressed["exception_code"], "CECL_BASIS_PERIOD_UNAVAILABLE"
        )

    def test_invalid_base_bucket_balance_does_not_block_valid_population(self):
        scenario = _scenario(_weighted_basis())
        identity = pd.DataFrame(
            [
                {
                    "_source_row": 1,
                    "borrower_id": "B1",
                    "balance": 100.0,
                    "cecl_portfolio": "CRE",
                    "base_bucket": "Pass",
                    "stressed_bucket_S1": "Pass",
                    "stressed_bucket_S2": "Pass",
                    "cecl_reserve": 10.0,
                    "module_applied": "CRE",
                },
                {
                    "_source_row": 2,
                    "borrower_id": "B2",
                    "balance": np.inf,
                    "cecl_portfolio": "CRE",
                    "base_bucket": "Substandard",
                    "stressed_bucket_S1": "Pass",
                    "stressed_bucket_S2": "Pass",
                    "cecl_reserve": 5.0,
                    "module_applied": "CRE",
                },
            ]
        )
        borrowers = build_borrowers(identity, scenario, [])
        borrowers["cecl_level_tag"] = "CRE"
        basis = _build_basis(
            borrowers,
            scenario,
            _history_period("CRE", "2026Q1", 0.10)
            + _history_period("CRE", "2025Q4", 0.10),
        )
        buckets = build_cecl_bucket_summary(
            borrowers, pd.DataFrame(), scenario
        )
        cecl = build_cecl_summary(
            borrowers, buckets, scenario, [], basis
        )

        for level in ("Base", "S1", "S2"):
            total = cecl[
                (cecl["portfolio"] == "CRE")
                & (cecl["stress_level"] == level)
                & (cecl["bucket"] == "Total")
            ].iloc[0]
            self.assertEqual(total["cecl_reserve_status"], "available")
            self.assertEqual(total["exception_code"], "")
            self.assertEqual(float(total["balance"]), 100.0)
            self.assertAlmostEqual(
                float(total["proforma_cecl_reserve"]), 10.0
            )
        history_audit = basis.audit[
            basis.audit["period_method"] == "tag_bucket_history"
        ]
        self.assertFalse(history_audit.empty)
        self.assertTrue(history_audit["invalid_balance_count"].eq(0).all())

    def test_unavailable_commercial_rows_retain_configured_basis_label(self):
        scenario = _scenario(_weighted_basis())
        results = _commercial_frame([100.0], [10.0]).drop(
            columns=["cecl_reserve"]
        )
        basis = build_cecl_reserve_basis(results, scenario, [], history=None)
        cecl = build_cecl_summary(
            results,
            pd.DataFrame(
                [
                    {
                        "portfolio": "CRE",
                        "cecl_level_tag": "CRE",
                        "stress_level": "Base",
                        "bucket": "Pass",
                        "balance": 100.0,
                    }
                ]
            ),
            scenario,
            [],
            basis,
        )
        base = cecl[
            (cecl["portfolio"] == "CRE")
            & (cecl["stress_level"] == "Base")
            & (cecl["bucket"] == "Total")
        ].iloc[0]

        self.assertEqual(
            base["reserve_basis"], "in_place+tag_bucket_history"
        )
        self.assertEqual(base["cecl_reserve_status"], "unavailable")

    def test_negative_rows_and_bucket_components_are_excluded(self):
        scenario = _scenario({"current_method": "central_tendency"})
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
        exceptions: list[dict] = []
        commercial_cecl = build_cecl_summary(
            commercial, commercial_buckets, scenario, exceptions
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

        self.assertEqual(
            commercial_total["cecl_reserve_status"], "available"
        )
        self.assertEqual(commercial_total["exception_code"], "")
        self.assertEqual(float(commercial_total["balance"]), 0.0)
        self.assertEqual(
            float(commercial_total["proforma_cecl_reserve"]), 0.0
        )
        self.assertEqual(consumer_total["cecl_reserve_status"], "available")
        self.assertEqual(consumer_total["exception_code"], "")
        self.assertEqual(float(consumer_total["balance"]), 0.0)
        self.assertEqual(
            float(consumer_total["proforma_cecl_reserve"]), 0.0
        )
        self.assertIn(
            "CECL_BUCKET_BALANCE_EXCLUDED",
            {row["code"] for row in exceptions},
        )

    def test_zero_balance_consumer_bucket_does_not_block_valid_portfolio(self):
        scenario = _scenario(
            {
                "current_method": "central_tendency",
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

    def test_consumer_ignores_top_level_central_tendency(self):
        scenario = _scenario(
            {
                "current_method": "central_tendency",
                "central_tendency": {"z_score_threshold": 10.0},
            }
        )
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
                },
                {
                    "borrower_id": "C2",
                    "balance": 900.0,
                    "model_portfolio": "Consumer",
                    "cecl_portfolio": "Consumer",
                    "base_bucket": "Pass",
                    "cecl_reserve": 180.0,
                    "module_applied": "Consumer",
                },
            ]
        )

        prepared, basis = attach_cecl_reserve_basis(results, scenario, [])

        # A commercial central-tendency calculation would assign a 15% ratio
        # and total $150. Consumer must preserve the two authoritative current
        # reserve amounts and identify its own basis as in-place.
        np.testing.assert_allclose(
            basis.effective_reserve.to_numpy(), np.array([10.0, 180.0])
        )
        self.assertAlmostEqual(float(basis.effective_reserve.sum()), 190.0)
        self.assertTrue(
            prepared["cecl_reserve_basis_method"].eq("in_place").all()
        )

    def test_consumer_reporting_uses_pre_stress_immutable_current_basis(self):
        scenario = _scenario({"current_method": "in_place"})
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
                    "consumer_el_S1": 5.0,
                    "consumer_el_S2": 6.0,
                    "out_of_scope_S1": False,
                    "out_of_scope_S2": False,
                }
            ]
        )
        basis = build_cecl_reserve_basis(results, scenario, [])

        # Simulate any later module mutation of the configured source column.
        # Reports must retain the current value captured before stress.
        results["cecl_reserve"] = 999.0
        consumer = build_consumer_summary(
            results, scenario, basis
        ).set_index("stress_level")
        cecl = build_cecl_summary(
            results,
            pd.DataFrame([{"portfolio": "Consumer"}]),
            scenario,
            [],
            basis,
        )
        consumer_cecl = cecl[
            (cecl["portfolio"] == "Consumer")
            & (cecl["bucket"] == "Total")
        ].set_index("stress_level")

        self.assertEqual(
            float(consumer.at["Base", "proforma_cecl_reserve"]), 10.0
        )
        self.assertEqual(
            float(consumer_cecl.at["Base", "proforma_cecl_reserve"]),
            10.0,
        )

    def test_consumer_module_uses_current_in_place_under_history_option(self):
        scenario = _with_history_source(
            _scenario(
                _weighted_basis(
                    current_method="central_tendency",
                    z_score_threshold=2.0,
                )
            )
        )
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

        prepared, basis = attach_cecl_reserve_basis(
            results,
            scenario,
            [],
            history=_history_frame(
                _history_period("Consumer", "2026Q1", 0.10)
                + _history_period("Consumer", "2025Q4", 0.20)
            ),
        )
        stressed, out_of_scope = run_consumer(prepared, scenario, inputs, [])

        self.assertTrue(out_of_scope.empty)
        self.assertAlmostEqual(float(basis.effective_reserve.iloc[0]), 10.0)
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_cecl_reserve_base"]), 10.0
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_qualitative_reserve"]), 10.0
        )
        self.assertAlmostEqual(
            float(stressed.at[0, "consumer_proforma_cecl_S1"]),
            float(stressed.at[0, "consumer_el_S1"]) + 10.0,
        )
        self.assertTrue(
            prepared["cecl_reserve_basis_method"].eq("in_place").all()
        )

    def test_consumer_only_history_option_does_not_require_history_input(self):
        scenario = _with_history_source(
            _scenario(_weighted_basis(current_method="central_tendency"))
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
                    "primary_module": "Consumer",
                    "cecl_reserve": 10.0,
                }
            ]
        )
        exceptions: list[dict] = []

        basis = build_cecl_reserve_basis(
            results, scenario, exceptions, history=None
        )

        self.assertEqual(float(basis.effective_reserve.iloc[0]), 10.0)
        self.assertEqual(basis.method_by_row.iloc[0], "in_place")
        self.assertNotIn(
            "CECL_HISTORY_SOURCE_UNAVAILABLE",
            {row["code"] for row in exceptions},
        )
        self.assertFalse(
            basis.audit["period"].isin(["2026Q1", "2025Q4"]).any()
        )

    def test_consumer_current_basis_reconciles_and_remains_monotonic_with_history_enabled(self):
        scenario = _with_history_source(_scenario(_weighted_basis()))
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

        exceptions: list[dict] = []
        basis = _build_basis(results, scenario, [], exceptions)
        self.assertFalse(
            any("HISTORY" in str(row.get("code", "")) for row in exceptions)
        )
        consumer_summary = build_consumer_summary(
            results, scenario, basis
        ).set_index(
            "stress_level"
        )
        cecl = build_cecl_summary(
            results,
            pd.DataFrame([{"portfolio": "Consumer"}]),
            scenario,
            [],
            basis,
        )
        consumer_cecl = cecl[
            (cecl["portfolio"] == "Consumer") & (cecl["bucket"] == "Total")
        ].set_index("stress_level")

        expected = {
            "Base": (23.0, 7.0, 30.0),
            "S1": (27.0, 7.0, 34.0),
            "S2": (30.0, 7.0, 37.0),
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
            self.assertEqual(consumer_summary.at[level, "reserve_basis"], "in_place")
            self.assertEqual(consumer_cecl.at[level, "reserve_basis"], "in_place")
        self.assertEqual(
            list(consumer_cecl["proforma_cecl_reserve"]), [30.0, 34.0, 37.0]
        )
        self.assertEqual(float(results["cecl_reserve"].sum()), 30.0)


if __name__ == "__main__":
    unittest.main()
