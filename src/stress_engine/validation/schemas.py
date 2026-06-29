"""File-level validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    table: str
    code: str
    message: str


def validate_required_files(input_dir: Path, input_files: Mapping[str, str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for table_name, file_name in input_files.items():
        if not (input_dir / file_name).exists():
            issues.append(
                ValidationIssue(
                    level="error",
                    table=table_name,
                    code="missing_required_file",
                    message=f"Missing required input file: {file_name}",
                )
            )
    return issues


def validate_required_columns(
    tables: Mapping[str, pd.DataFrame], required_columns: Mapping[str, Iterable[str]]
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for table_name, columns in required_columns.items():
        if table_name not in tables:
            continue
        missing = sorted(set(columns) - set(tables[table_name].columns))
        if missing:
            issues.append(
                ValidationIssue(
                    level="error",
                    table=table_name,
                    code="missing_required_column",
                    message=f"{table_name} is missing columns: {', '.join(missing)}",
                )
            )
    return issues


def validate_unique_loan_ids(loan_identity: pd.DataFrame) -> List[ValidationIssue]:
    if "loan_id" not in loan_identity.columns:
        return []
    duplicated = loan_identity["loan_id"][loan_identity["loan_id"].duplicated()].dropna().unique()
    if len(duplicated) == 0:
        return []
    return [
        ValidationIssue(
            level="error",
            table="loan_identity",
            code="duplicate_loan_id",
            message=f"Duplicate loan_id values found: {', '.join(map(str, duplicated))}",
        )
    ]


def to_frame(issues: List[ValidationIssue]) -> pd.DataFrame:
    return pd.DataFrame([issue.__dict__ for issue in issues], columns=["level", "table", "code", "message"])
