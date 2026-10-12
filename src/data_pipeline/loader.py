"""Module 1 — Data Loading & Reference for Micro-Credit Pulse.

Reads ``data/raw/alternative_credit_scoring.csv``, validates 29-column schema,
and exposes cohort statistics from the EDA doc / raw data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from features._shared import load_raw_data, validate_required_columns, REQUIRED_BASE_COLUMNS


def load_reference_dataset(path: str = "Data/raw/alternative_credit_scoring.csv") -> pd.DataFrame:
    """Load and validate the raw 2,000-record base dataset.

    Args:
        path: Path to the CSV (case-insensitive on Windows).

    Returns:
        Validated DataFrame with all 29 base columns.
    """
    # Note: EDA doc uses ``Data/raw/``; _shared uses same case-insensitive path.
    df = load_raw_data(path)
    validate_required_columns(df, REQUIRED_BASE_COLUMNS, context="Module 1 loader")
    return df


def get_cohort_split(df: pd.DataFrame) -> Dict[str, int]:
    """Return observed area_type counts (from EDA / raw)."""
    return df["area_type"].value_counts().to_dict()


def get_target_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return primary target distribution."""
    return df["loan_approved"].value_counts().to_dict()


def main() -> None:
    df = load_reference_dataset()
    print("Module 1 — Data loaded:", df.shape)
    print("Cohort split:", get_cohort_split(df))
    print("Target summary:", get_target_summary(df))


if __name__ == "__main__":
    main()
