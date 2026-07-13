"""Deterministic exception and fallback logging."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import pandas as pd


EXCEPTION_COLUMNS = [
    "severity",
    "stage",
    "code",
    "message",
    "portfolio",
    "stress_level",
    "bucket",
    "borrower_id",
    "module",
    "field",
    "source",
    "details",
]


def record_exception(
    exceptions: List[Dict[str, Any]],
    severity: str,
    stage: str,
    code: str,
    message: str,
    **context: Any,
) -> None:
    """Append one structured exception/warning row.

    Called throughout reporting and C&I/overlay logic when a bad state or
    fallback-style behavior should be visible in `exception_log.csv`.
    """
    row = {column: "" for column in EXCEPTION_COLUMNS}
    row.update(
        {
            "severity": severity,
            "stage": stage,
            "code": code,
            "message": message,
        }
    )
    for key, value in context.items():
        if key in row:
            row[key] = "" if _is_scalar_missing(value) else value
        else:
            row["details"] = f"{row['details']}; {key}={value}".strip("; ")
    exceptions.append(row)


def exception_frame(exceptions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert collected exception rows to a consistently shaped DataFrame."""
    if not exceptions:
        return pd.DataFrame(columns=EXCEPTION_COLUMNS)
    return pd.DataFrame(exceptions, columns=EXCEPTION_COLUMNS)


def _is_scalar_missing(value: Any) -> bool:
    """Check scalar missing values without evaluating array-like truth values."""
    if isinstance(value, (list, tuple, set, Mapping)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
