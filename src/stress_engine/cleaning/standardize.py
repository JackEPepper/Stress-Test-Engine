"""Data standardization routines."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = {
    "balance",
    "commitment",
    "interest_rate",
    "fico",
    "current_el_rate",
    "noi",
    "appraised_value",
    "occupancy_rate",
    "debt_service",
    "ltv",
    "dscr",
    "revenue",
    "ebitda",
    "total_debt",
    "cash",
    "interest_expense",
    "fico_min",
    "fico_max",
    "base_pd",
}


TEXT_COLUMNS = {
    "loan_id",
    "borrower_id",
    "portfolio",
    "product_type",
    "sector",
    "risk_rating",
    "collateral_id",
    "cre_sector",
    "industry",
}


def standardize_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {name: standardize_table(frame) for name, frame in tables.items()}


def standardize_table(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    for column in cleaned.columns:
        if column in TEXT_COLUMNS:
            cleaned[column] = cleaned[column].astype(str).str.strip()
        elif column in NUMERIC_COLUMNS:
            cleaned[column] = pd.to_numeric(cleaned[column].replace("", np.nan), errors="coerce")
        elif column == "maturity_date":
            cleaned[column] = pd.to_datetime(cleaned[column].replace("", np.nan), errors="coerce")

    for column in ("portfolio", "product_type", "sector", "cre_sector", "industry"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip().str.lower()

    return cleaned
