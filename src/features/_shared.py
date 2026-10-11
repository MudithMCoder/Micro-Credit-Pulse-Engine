"""Shared utilities for the Micro-Credit Pulse feature-synthesis modules.

Holds the raw-dataset schema contract, cohort-mask helpers, categorical
samplers, and small numeric utilities used by both ``urban_features`` and
``rural_features``. Nothing in here references target/label columns — the
leakage guard from ``docs/Synthesized_Features_Research.md`` (Part 6) is
enforced by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: Default location of the raw base dataset (Windows is case-insensitive, so
#: ``Data/raw`` and ``data/raw`` resolve to the same folder).
RAW_DATA_PATH = "Data/raw/alternative_credit_scoring.csv"

#: Default output for the engineered (raw + synthesized) feature set.
ENGINEERED_FEATURES_PATH = "data/processed/engineered_features.parquet"

#: Reference/label columns that synthesis formulas must never read or emit.
LEAKAGE_COLUMNS = frozenset(
    {"alternative_credit_score", "credit_risk_label", "loan_approved"}
)

#: The 29-column schema of the raw dataset (see ``docs/Dataset Doc.md``).
REQUIRED_BASE_COLUMNS: Tuple[str, ...] = (
    "age",
    "monthly_income_inr",
    "area_type",
    "upi_monthly_avg_txn_count",
    "upi_avg_txn_amount",
    "upi_failed_txn_ratio",
    "mobile_recharge_consistency_score",
    "mobile_recharge_lapse_days_max",
    "utility_bills_on_time_pct",
    "electricity_bill_avg_amount",
    "cooperative_meeting_attendance_pct",
    "shg_savings_amount_inr",
    "microfinance_loans_repaid_count",
    "microfinance_default_rate",
    "gig_monthly_earnings",
    "gig_platform_rating",
    "jandhan_avg_balance",
    "jandhan_zero_balance_months",
    "monthly_data_usage_gb",
    "telecom_bill_late_payments",
    "banking_app_login_frequency_monthly",
    "fintech_apps_count",
    "rent_on_time_payment_pct",
    "rent_to_income_ratio",
    "social_referral_count",
    "insurance_policy_count",
    "alternative_credit_score",
    "credit_risk_label",
    "loan_approved",
)


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    context: str = "frame",
) -> None:
    """Raise ``ValueError`` if any required column is missing from ``df``.

    Args:
        df: DataFrame to validate.
        required: Column names that must all be present.
        context: Human-readable label used in the error message.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required columns: {sorted(missing)}"
        )


def load_raw_data(
    path: str = RAW_DATA_PATH,
    required_columns: Sequence[str] = REQUIRED_BASE_COLUMNS,
) -> pd.DataFrame:
    """Load the raw alternative-credit CSV and validate its schema.

    Args:
        path: Path to the raw CSV file.
        required_columns: Columns that must be present after loading.

    Returns:
        The loaded raw DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If the file cannot be parsed as CSV.
        ValueError: If the schema validation fails.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at '{csv_path}'. "
            "Expected the alternative-credit CSV under data/raw/."
        )
    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to read raw dataset '{csv_path}': {exc}") from exc
    validate_required_columns(df, required_columns, context=f"Raw dataset '{path}'")
    return df


def write_parquet(df: pd.DataFrame, path: str = ENGINEERED_FEATURES_PATH) -> None:
    """Write a DataFrame to parquet, auto-creating the parent directory.

    Args:
        df: DataFrame to persist.
        path: Destination parquet file path.

    Raises:
        RuntimeError: If the write fails.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False, engine="pyarrow")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to write parquet to '{out_path}': {exc}") from exc


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    default: float = 0.0,
) -> np.ndarray:
    """Element-wise division that returns ``default`` where denominator is 0.

    Args:
        numerator: Numerator array.
        denominator: Denominator array (same shape).
        default: Value to use where ``denominator <= 0``.

    Returns:
        Float array of the quotient with zero-division guarded.
    """
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    safe_den = np.where(den == 0.0, 1.0, den)
    return np.where(den > 0.0, num / safe_den, default)


def bernoulli_flags(rng: np.random.Generator, probability: np.ndarray, n: int) -> np.ndarray:
    """Draw per-row 0/1 flags from row-wise Bernoulli probabilities.

    Args:
        rng: Seeded numpy random generator.
        probability: Array of success probabilities (clipped to [0, 1]).
        n: Number of draws (should match ``len(probability)``).

    Returns:
        int8 array of 0/1 flags.
    """
    p = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
    return (rng.random(n) < p).astype("int8")


def sample_categorical(
    rng: np.random.Generator,
    categories: Sequence[str],
    probabilities: Union[Sequence[float], np.ndarray],
    n: int,
) -> np.ndarray:
    """Draw ``n`` categorical samples from row-wise probabilities.

    Args:
        rng: Seeded numpy random generator.
        categories: Category labels (column order matches ``probabilities``).
        probabilities: 1-D distribution (tiled across rows) or a 2-D
            ``(n, len(categories))`` matrix of per-row distributions.
        n: Number of samples to draw.

    Returns:
        Object array of sampled category labels.
    """
    probs = np.asarray(probabilities, dtype="float64")
    if probs.ndim == 1:
        probs = np.tile(probs / probs.sum(), (n, 1))
    cum = np.cumsum(probs, axis=1)
    cum[:, -1] = 1.0  # guard against float drift so every row sums to 1
    idx = (rng.random(n)[:, None] > cum).sum(axis=1)
    idx = np.clip(idx, 0, len(categories) - 1)
    return np.asarray(categories, dtype=object)[idx]


def sample_categorical_by_area(
    rng: np.random.Generator,
    df: pd.DataFrame,
    options: Mapping[str, Tuple[Sequence[str], Sequence[float]]],
) -> np.ndarray:
    """Draw categorical samples with area-specific distributions.

    Args:
        rng: Seeded numpy random generator.
        df: DataFrame whose ``area_type`` column drives the distribution.
        options: Mapping of area name to ``(categories, probabilities)``.
            All areas must list categories in the same order.

    Returns:
        Object array of sampled category labels, one per row.

    Raises:
        ValueError: If an observed ``area_type`` has no configured options.
    """
    areas = df["area_type"].to_numpy()
    area_names = list(options.keys())
    unmatched = set(np.unique(areas[~np.isin(areas, area_names)]))
    if unmatched:
        raise ValueError(f"No categorical options configured for area_type: {unmatched}")
    tier = np.select([areas == name for name in area_names], np.arange(len(area_names)))
    categories = options[area_names[0]][0]
    prob_matrix = np.stack(
        [np.asarray(options[name][1], dtype="float64") for name in area_names]
    )
    return sample_categorical(rng, categories, prob_matrix[tier], len(df))


def apply_cohort_mask(block: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Set NA wherever ``mask`` is False, preserving integer semantics.

    Integer columns are recast to pandas nullable dtypes (``Int8``/``Int16``)
    so cohort-specific counts/flags can carry NA without becoming floats.
    Float columns become ``float32`` with NaN; object columns keep NaN.

    Args:
        block: DataFrame of synthesized columns.
        mask: Boolean array of shape ``(n_rows,)`` broadcast across columns,
            or ``(n_rows, n_cols)`` for per-cell masks.

    Returns:
        A new DataFrame with non-appliable cells set to NA.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        mask = np.tile(mask[:, None], (1, block.shape[1]))
    out = pd.DataFrame(index=block.index)
    for pos, column in enumerate(block.columns):
        series = block[column]
        col_mask = mask[:, pos]
        if series.dtype.kind in "ib":
            width = {1: 8, 2: 16, 4: 32, 8: 64}.get(series.dtype.itemsize, 64)
            nullable = "Int8" if series.dtype.kind == "b" else f"Int{width}"
            out[column] = (
                series.where(col_mask, other=np.nan).astype("float64").astype(nullable)
            )
        elif series.dtype.kind == "f":
            out[column] = series.where(col_mask, other=np.nan).astype("float32")
        else:
            out[column] = series.where(col_mask, other=np.nan)
    return out
