"""Tests for the feature synthesis modules (urban_features & rural_features)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import (
    synthesize_all_features,
    synthesize_rural_features,
    synthesize_urban_features,
)
from features._shared import (
    REQUIRED_BASE_COLUMNS,
    load_raw_data,
    validate_required_columns,
)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Fixture providing the raw 2,000-record dataset."""
    return load_raw_data("Data/raw/alternative_credit_scoring.csv")


def test_schema_validation_passes_on_raw(raw_df: pd.DataFrame) -> None:
    """Raw dataframe contains all 29 required base columns."""
    validate_required_columns(raw_df, REQUIRED_BASE_COLUMNS)
    assert len(raw_df) == 2000
    assert len(raw_df.columns) == 29


def test_schema_validation_fails_on_missing_column(raw_df: pd.DataFrame) -> None:
    """Dropped column raises ValueError."""
    dropped = raw_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_required_columns(dropped, REQUIRED_BASE_COLUMNS)


def test_urban_features_shape_and_types(raw_df: pd.DataFrame) -> None:
    """Urban synthesis produces expected columns and preserves length."""
    urban = synthesize_urban_features(raw_df, random_state=42)
    assert len(urban) == 2000
    assert "digital_engagement_index" in urban.columns
    assert "upi_transaction_velocity" in urban.columns
    assert "wallet_volatility_score" in urban.columns

    # Rural rows must be NaN in urban-only features
    rural_rows = raw_df["area_type"] == "Rural"
    assert urban.loc[rural_rows, "digital_engagement_index"].isna().all()

    # Urban rows must NOT be NaN
    urban_rows = raw_df["area_type"] == "Urban"
    assert urban.loc[urban_rows, "digital_engagement_index"].notna().all()


def test_rural_features_shape_and_types(raw_df: pd.DataFrame) -> None:
    """Rural synthesis produces expected columns and preserves length."""
    rural = synthesize_rural_features(raw_df, random_state=42)
    assert len(rural) == 2000
    assert "crop_harvest_topup_ratio" in rural.columns
    assert "rainfall_anomaly_index" in rural.columns
    assert "cooperative_payout_consistency_score" in rural.columns

    # Urban rows must be NaN in rural-only features
    urban_rows = raw_df["area_type"] == "Urban"
    assert rural.loc[urban_rows, "rainfall_anomaly_index"].isna().all()

    # Rural rows must NOT be NaN
    rural_rows = raw_df["area_type"] == "Rural"
    assert rural.loc[rural_rows, "rainfall_anomaly_index"].notna().all()

    # Semi-Urban rows have coarsened values (~30% NaN)
    semi_rows = raw_df["area_type"] == "Semi-Urban"
    nan_rate = rural.loc[semi_rows, "crop_harvest_topup_ratio"].isna().mean()
    assert 0.15 < nan_rate < 0.45


def test_structural_zeros_preserved(raw_df: pd.DataFrame) -> None:
    """Structural zeros in base data remain intact and flags match."""
    rural = synthesize_rural_features(raw_df, random_state=42)
    rural_rows = raw_df["area_type"] == "Rural"

    # Flag correctly identifies positive earnings
    flag = rural.loc[rural_rows, "gig_income_presence_flag"]
    actual = (raw_df.loc[rural_rows, "gig_monthly_earnings"] > 0).astype("int8")
    assert (flag == actual).all()


def test_synthesize_all_features_end_to_end(tmp_path: Path) -> None:
    """Full orchestration runs end-to-end and outputs valid parquet."""
    out_file = tmp_path / "test_features.parquet"
    df = synthesize_all_features(
        input_path="Data/raw/alternative_credit_scoring.csv",
        output_path=str(out_file),
        random_state=42,
    )

    assert df.shape[0] == 2000
    assert df.shape[1] == 104  # 29 raw + 75 synthesized
    assert out_file.exists()

    # Round-trip read
    reloaded = pd.read_parquet(out_file)
    assert reloaded.shape == (2000, 104)


def test_determinism() -> None:
    """Identical seeds yield bit-for-bit identical frames."""
    raw = load_raw_data("Data/raw/alternative_credit_scoring.csv")
    df1 = synthesize_urban_features(raw, random_state=123)
    df2 = synthesize_urban_features(raw, random_state=123)
    for col in df1.columns:
        assert df1[col].equals(df2[col]), f"Mismatch in {col}"
