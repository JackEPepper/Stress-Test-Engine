"""Stress module dispatch."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from stress_engine.stress.ci import apply_ci
from stress_engine.stress.consumer import apply_consumer
from stress_engine.stress.cre import apply_cre


def apply_selected_stress(
    frame: pd.DataFrame, scenario: Mapping[str, object], config: Mapping[str, object], fico_pd_table: pd.DataFrame
) -> pd.DataFrame:
    stressed = frame.copy()
    result_rows = []
    for _, row in stressed.iterrows():
        if row.get("scope_status") != "in_scope":
            result_rows.append({})
            continue
        module = row.get("selected_stress_module")
        if module == "cre":
            result_rows.append(apply_cre(row, scenario, config))
        elif module == "ci":
            result_rows.append(apply_ci(row, scenario, config))
        elif module == "consumer":
            result_rows.append(apply_consumer(row, scenario, config, fico_pd_table))
        else:
            result_rows.append({})

    results = pd.DataFrame(result_rows)
    if results.empty:
        return stressed
    for column in results.columns:
        stressed[column] = results[column]
    stressed["expected_loss_change"] = stressed["stressed_expected_loss"].fillna(0) - stressed["base_expected_loss"].fillna(0)
    return stressed
