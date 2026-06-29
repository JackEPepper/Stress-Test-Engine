"""Output writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def write_excel(path: Path, sheets: Mapping[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                safe_name = sheet_name[:31]
                frame.to_excel(writer, sheet_name=safe_name, index=False)
    except ImportError:
        return
