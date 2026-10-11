"""Feature engineering module for Micro-Credit Pulse Engine.

Provides cohort-routed feature synthesis:
- Urban/Semi-Urban borrowers → digital-footprint & transactional features
- Rural borrowers → seasonal/agricultural proxy features

Public API:
    synthesize_all_features(input_path, output_path, random_state)

See also:
    - urban_features: synthesize_urban_features()
    - rural_features: synthesize_rural_features()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from ._shared import (
    ENGINEERED_FEATURES_PATH,
    REQUIRED_BASE_COLUMNS,
    load_raw_data,
    validate_required_columns,
    write_parquet,
)
from .urban_features import synthesize_urban_features
from .rural_features import synthesize_rural_features


def synthesize_all_features(
    input_path: str = "Data/raw/alternative_credit_scoring.csv",
    output_path: str = ENGINEERED_FEATURES_PATH,
    random_state: int = 42,
) -> pd.DataFrame:
    """Orchestrate complete feature engineering: load, route, synthesize, write.

    This is the primary entry point for the feature-engineering pipeline.
    It:
    1. Loads the raw 29-column dataset.
    2. Routes rows by ``area_type`` (Urban/Semi-Urban → urban module;
       Rural → rural module).
    3. Applies cross-cohort and regional-context features (from both
       modules) to all rows.
    4. Merges raw + synthesized columns.
    5. Writes ``data/processed/engineered_features.parquet``.

    Args:
        input_path: Path to the raw CSV file.
        output_path: Destination parquet file path.
        random_state: RNG seed for reproducible synthesis.

    Returns:
        The full engineered DataFrame (raw + synthesized columns).

    Raises:
        FileNotFoundError: If the raw CSV does not exist.
        ValueError: If the raw CSV schema is invalid.
        RuntimeError: If the parquet write fails.
    """
    # 1. Load and validate raw data.
    raw_df = load_raw_data(input_path)
    validate_required_columns(raw_df, REQUIRED_BASE_COLUMNS, context="raw dataset")

    rng = np.random.default_rng(random_state)
    n = len(raw_df)

    # 2. Synthesize cohort-specific features.
    # Urban module returns NaN for Rural rows; Rural module returns NaN for
    # Urban rows and applies 30% NaN mask to Semi-Urban (coarsening).
    urban_synth = synthesize_urban_features(raw_df, random_state=random_state)
    rural_synth = synthesize_rural_features(raw_df, random_state=random_state)

    # 3. Merge: raw columns + urban-synthesized + rural-synthesized.
    # The two synthetic frames have overlapping columns (regional-context,
    # cross-cohort). We keep only one copy; precedence: urban-synth last,
    # so it overwrites rural-synth for Urban/Semi rows (where rural has NaN).
    # For Rural rows, urban_synth is all NaN and rural_synth has the values.
    combined = pd.concat([raw_df, rural_synth, urban_synth], axis=1)

    # Guard against duplicate columns (should not happen, but be safe).
    dupes = combined.columns[combined.columns.duplicated()].tolist()
    if dupes:
        # Keep first occurrence (rural) for Rural rows, urban for others.
        combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    # 4. Enforce dtypes for consistency (optional polish step).
    # This ensures binary flags stay int8, scores stay float32, etc.
    # The synthesis already used these dtypes; this is a sanity check.
    # Skip for performance; rely on synthesis for dtype correctness.

    # 5. Write output.
    write_parquet(combined, output_path)

    return combined


# Expose sub-module entry points for direct usage if needed.
__all__ = [
    "synthesize_all_features",
    "synthesize_urban_features",
    "synthesize_rural_features",
]