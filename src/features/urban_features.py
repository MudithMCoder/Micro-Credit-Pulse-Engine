"""Urban / Semi-Urban cohort feature synthesis.

Per ``docs/Synthesized_Features_Research.md`` (Part 2 — Digital
Footprint & Transactional Metrics, plus Urban-side cross-cohort from
Part 3 and regional context from Part 4). All synthesis formulas derive
from base columns; the leakage guard (no reads from
``alternative_credit_score``, ``credit_risk_label``, ``loan_approved``) is
enforced by design.

Public entry point: ``synthesize_urban_features(df, random_state=42)``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from ._shared import (
    ENGINEERED_FEATURES_PATH,
    REQUIRED_BASE_COLUMNS,
    apply_cohort_mask,
    load_raw_data,
    safe_divide,
    validate_required_columns,
    write_parquet,
)

#: Columns synthesized specifically for Urban / Semi-Urban borrowers.
SYNTHETIC_URBAN_COLUMNS: List[str] = [
    # 2.1 UPI / digital-payment behavioural
    "digital_engagement_index",
    "upi_transaction_velocity",
    "wallet_volatility_score",
    "cross_platform_payment_count",
    "recurring_payment_subscription_count",
    "utility_payment_regularity_score",
    "rent_payment_advance_days",
    # 2.2 Banking / fintech engagement
    "banking_app_session_duration_avg",
    "loan_product_inquiry_count",
    "credit_score_inquiry_own_count",
    "investment_app_ownership_flag",
    "insurance_app_purchase_flag",
    # 2.3 E-commerce / platform engagement
    "ecommerce_monthly_spend_inr",
    "ecommerce_order_frequency_monthly",
    "ecommerce_return_rate",
    "food_delivery_order_count",
    "subscription_service_count",
    # 2.4 Employment / income stability
    "employer_industry_category",
    "job_tenure_months",
    "salary_dispersion_index",
    "side_gig_income_presence",
    "freelance_income_ratio",
    # 3.1 Urban-side household (part of split cross-cohort)
    "vehicle_ownership_flag",
    # 3.2 Urban-side telecom behaviour
    "mobile_recharge_frequency_monthly",
    "telecom_bill_payment_advance_days",
    "data_usage_consistency_score",
    # 3.4 Urban-side government-ID / verification
    "pan_card_availability_flag",
    "driving_license_ownership",
    # 3.3 Urban-side social / referral
    "referrer_loan_default_flag",
    "family_credit_history_known",
    # 4 Regional / macro (cohort-neutral — computed here as urban-side owner)
    "district_urbanization_level",
    "mfi_branch_density",
    "bank_branch_per_capita",
    "pincode_agrarian_dominance",
    "msme_density_score",
    "festival_season_indicator",
    "monsoon_onset_lag_days",
    "harvest_calendar_region",
    "tourism_season_indicator",
]


def _build_rng(random_state: int = 42) -> np.random.Generator:
    return np.random.default_rng(random_state)


def synthesize_urban_features(
    df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Synthesize the Urban / Semi-Urban feature set from the raw frame.

    Args:
        df: Raw 29-column DataFrame. Must include ``area_type``.
        random_state: Reproducibility seed for all stochastic synthesis.

    Returns:
        A new DataFrame with the same index and ``SYNTHETIC_URBAN_COLUMNS``
        appended. Non-applicable rows (``area_type == 'Rural'``) receive
        ``np.nan`` via ``apply_cohort_mask``.

    Raises:
        ValueError: If required base columns are missing.
        RuntimeError: If column names collide with synthesized names.
    """
    validate_required_columns(df, REQUIRED_BASE_COLUMNS, context="urban synthesis")
    if "area_type" not in df.columns:
        raise ValueError("Cohort routing requires column 'area_type'")

    # Guard against accidental overwrite of base columns (leakage guard
    # also prevents synthesizing from target/label fields).
    for col in SYNTHETIC_URBAN_COLUMNS:
        if col in df.columns:
            raise ValueError(f"Synthetic column '{col}' already exists in input.")

    rng = _build_rng(random_state)
    n = len(df)

    # ----- 2.1 UPI / digital-payment behavioural -----
    txn_count = df["upi_monthly_avg_txn_count"].to_numpy(dtype="float64")
    avg_amt = df["upi_avg_txn_amount"].to_numpy(dtype="float64")

    # Normalised engagement (0-1) combining volume and value.
    raw_engagement = np.log1p(txn_count * avg_amt) / np.log1p(10000.0)
    digital_engagement_index = np.clip(
        raw_engagement + rng.normal(0.0, 0.05, n), 0.0, 1.0
    ).astype("float32")

    upi_transaction_velocity = (txn_count / 30.0).astype("float32")

    failed = df["upi_failed_txn_ratio"].to_numpy(dtype="float64")
    zero_bal_months = df["jandhan_zero_balance_months"].to_numpy(dtype="float64")
    wallet_volatility_score = np.clip(
        0.30 * failed + 0.10 * zero_bal_months + rng.normal(0.0, 0.03, n),
        0.0,
        1.0,
    ).astype("float32")

    fintech = df["fintech_apps_count"].to_numpy(dtype="int16")
    cross_platform_payment_count = np.clip(
        np.minimum(fintech, 5) + rng.integers(0, 2, size=n), 0, 10
    ).astype("int16")

    rent_on_time = df["rent_on_time_payment_pct"].to_numpy(dtype="float64")
    util_on_time = df["utility_bills_on_time_pct"].to_numpy(dtype="float64")
    recurring_payment_subscription_count = (
        (rent_on_time > 0.70).astype("int8")
        + (util_on_time > 0.70).astype("int8")
        + rng.integers(0, 2, size=n).astype("int8")
    ).astype("int8")

    utility_payment_regularity_score = np.clip(
        df["utility_bills_on_time_pct"].to_numpy("float64") + rng.normal(0.0, 0.05, n),
        0.0,
        1.0,
    ).astype("float32")

    rent_payment_advance_days = np.where(
        rent_on_time > 0.80,
        5.0 + rng.integers(0, 5, size=n).astype("float32"),
        0.0,
    ).astype("float32")

    # ----- 2.2 Banking / fintech engagement -----
    login_freq = df["banking_app_login_frequency_monthly"].to_numpy("float64")
    banking_app_session_duration_avg = np.clip(
        login_freq * (2.0 + rng.normal(0.0, 0.5, n)), 0.5, 30.0
    ).astype("float32")

    # Intent proxy (leakage-safe — uses only income / age, never target).
    income = df["monthly_income_inr"].to_numpy("float64")
    age = df["age"].to_numpy("float64")
    lambda_poisson = np.clip(income / 10000.0, 0.1, 3.0)
    loan_product_inquiry_count = rng.poisson(lambda_poisson).astype("int8")

    credit_score_inquiry_own_count = np.clip(
        (login_freq / 10.0).astype("int16"), 0, 3
    ).astype("int8")

    investment_app_ownership_flag = (
        ((fintech >= 3) & (age >= 25)).astype("int8")
    )
    insurance_app_purchase_flag = (
        (df["insurance_policy_count"].to_numpy("int8") > 0).astype("int8")
    )

    # ----- 2.3 E-commerce / platform -----
    # Spend derived from income + UPI value as disposable-income signal.
    ecommerce_monthly_spend_inr = (
        income * (0.05 + 0.10 * avg_amt / np.clip(income, 1.0, None))
    ).astype("float32")

    ecommerce_order_frequency_monthly = np.clip(
        txn_count / 5.0, 1, 30
    ).astype("int16")

    ecommerce_return_rate = np.clip(
        0.02 + age * 0.001 + rng.normal(0.0, 0.02, n), 0.0, 0.30
    ).astype("float32")

    # Food-delivery proxy by income tier (discretionary-spending indicator).
    food_delivery_order_count = np.where(
        income > 15000, rng.poisson(4, n),
        np.where(income > 8000, rng.poisson(2, n), 0)
    ).astype("int8")

    # Subscription count — regular-cash-flow-commitment indicator.
    subscription_service_count = np.select(
        [income > 20000, income > 10000],
        [rng.integers(2, 5, n), rng.integers(1, 3, n)],
        default=0,
    ).astype("int8")

    # ----- 2.4 Employment / income stability -----
    # Weighted-categorical draw using income as selection weight.
    industry_cats = ["IT", "Manufacturing", "Retail", "Services", "Construction"]
    # Per-row probability weights scaled by income tier.
    # Simplify: just use a single draw per row (vectorized).
    u = rng.random(n)
    employer_industry_category = np.select(
        [
            u < 0.25,
            u < 0.45,  # 0.25 + 0.20
            u < 0.70,  # 0.45 + 0.25
            u < 0.90,  # 0.70 + 0.20
        ],
        [
            np.full(n, "IT", dtype=object),
            np.full(n, "Manufacturing", dtype=object),
            np.full(n, "Retail", dtype=object),
            np.full(n, "Services", dtype=object),
        ],
        default=np.full(n, "Construction", dtype=object),
    )
    employer_industry_category = pd.Series(
        employer_industry_category, index=df.index, dtype=object
    )

    job_tenure_months = np.clip(
        age - 22.0 + rng.normal(0.0, 12.0, n), 1.0, 480.0
    ).astype("int16")

    salary_dispersion_index = np.clip(
        0.10 + (income / 50000.0) * rng.random(n), 0.0, 1.0
    ).astype("float32")

    side_gig_income_presence = (
        (df["gig_monthly_earnings"].to_numpy("float64") > 0.0).astype("int8")
    )

    gig_earnings = df["gig_monthly_earnings"].to_numpy("float64")
    freelance_income_ratio = safe_divide(
        gig_earnings,
        income + gig_earnings + 1.0,
        default=0.0,
    ).astype("float32")

    # ----- 3.1 Urban-side household / asset -----
    vehicle_ownership_flag = (
        ((income > 15000) | (age > 35)).astype("int8")
    )

    # ----- 3.2 Telecom behaviour -----
    recharge_consistency = df["mobile_recharge_consistency_score"].to_numpy(
        "float64"
    )
    mobile_recharge_frequency_monthly = np.clip(
        2.0 + (1.0 - recharge_consistency) * 6.0, 1, 12
    ).astype("int8")

    telecom_late = df["telecom_bill_late_payments"].to_numpy("int8")
    telecom_bill_payment_advance_days = np.where(
        telecom_late == 0,
        3.0 + rng.integers(0, 5, n).astype("float32"),
        -telecom_late.astype("float32"),
    ).astype("float32")

    data_usage = df["monthly_data_usage_gb"].to_numpy("float64")
    data_usage_consistency_score = np.clip(
        1.0 - np.clip(data_usage / 50.0, 0.0, 1.0) + rng.normal(0.0, 0.10, n),
        0.0,
        1.0,
    ).astype("float32")

    # ----- 3.4 Government-ID / verification -----
    # Age-weighted flags with mild stochasticity.
    pan_card_availability_flag = np.clip(
        (age >= 25).astype("float64") * 0.85 + rng.normal(0.0, 0.05, n),
        0.0,
        1.0,
    )
    pan_card_availability_flag = (pan_card_availability_flag > 0.5).astype("int8")
    driving_license_ownership = np.clip(
        (age >= 18).astype("float64") * 0.70 + rng.normal(0.0, 0.08, n),
        0.0,
        1.0,
    )
    driving_license_ownership = (driving_license_ownership > 0.5).astype("int8")

    # ----- 3.3 Social / referral -----
    # Default propagation from microfinance rate (leakage-safe — uses base rate, not label).
    def_rate = df["microfinance_default_rate"].to_numpy("float64")
    referrer_loan_default_flag = (
        (rng.random(n) < def_rate).astype("int8")
    )
    family_credit_history_known = (
        (rng.random(n) < 0.30).astype("int8")
    )

    # ----- 4 Macroeconomic / regional -----
    # Cohort-neutral; values vary by area for realism.
    district_urbanization_level = np.clip(
        np.where(
            df["area_type"].to_numpy() == "Urban",
            0.60 + rng.uniform(-0.15, 0.15, n),
            np.where(
                df["area_type"].to_numpy() == "Semi-Urban",
                0.40 + rng.uniform(-0.15, 0.15, n),
                0.15 + rng.uniform(-0.10, 0.10, n),
            ),
        ),
        0.0,
        1.0,
    ).astype("float32")

    mfi_branch_density = np.clip(
        rng.integers(0, 6, size=n) + (district_urbanization_level * 3).astype("float32"),
        0,
        10,
    ).astype("int8")

    bank_branch_per_capita = np.clip(
        0.5 + 2.5 * district_urbanization_level + rng.normal(0.0, 0.3, n),
        0.0,
        5.0,
    ).astype("float32")

    pincode_agrarian_dominance = np.clip(
        np.where(
            df["area_type"].to_numpy() == "Rural",
            0.65 + rng.normal(0.0, 0.15, n),
            np.where(
                df["area_type"].to_numpy() == "Semi-Urban",
                0.40 + rng.normal(0.0, 0.15, n),
                0.10 + rng.normal(0.0, 0.10, n),
            ),
        ),
        0.0,
        1.0,
    ).astype("float32")

    msme_density_score = np.clip(
        0.25 + 0.50 * district_urbanization_level + rng.normal(0.0, 0.10, n),
        0.0,
        1.0,
    ).astype("float32")

    festival_season_indicator = (
        (rng.random(n) < 0.25).astype("int8")
    )

    # Seasonal calendar — simulated lag from normal monsoon onset.
    monsoon_onset_lag_days = np.clip(
        rng.integers(-15, 16, size=n).astype("float32"), -30.0, 30.0
    ).astype("float32")

    harvest_calendar_region = np.select(
        [
            rng.random(n) < 0.45,
            rng.random(n) < 0.35,
        ],
        [
            np.full(n, "Kharif", dtype=object),
            np.full(n, "Rabi", dtype=object),
        ],
        default=np.full(n, "Zaid", dtype=object),
    )
    harvest_calendar_region = pd.Series(
        harvest_calendar_region, index=df.index, dtype=object
    )

    tourism_season_indicator = (
        (rng.random(n) < 0.15).astype("int8")
    )

    # ----- Assemble synthetic frame -----
    synthetic = pd.DataFrame({
        "digital_engagement_index": digital_engagement_index,
        "upi_transaction_velocity": upi_transaction_velocity,
        "wallet_volatility_score": wallet_volatility_score,
        "cross_platform_payment_count": cross_platform_payment_count,
        "recurring_payment_subscription_count": recurring_payment_subscription_count,
        "utility_payment_regularity_score": utility_payment_regularity_score,
        "rent_payment_advance_days": rent_payment_advance_days,
        "banking_app_session_duration_avg": banking_app_session_duration_avg,
        "loan_product_inquiry_count": loan_product_inquiry_count,
        "credit_score_inquiry_own_count": credit_score_inquiry_own_count,
        "investment_app_ownership_flag": investment_app_ownership_flag,
        "insurance_app_purchase_flag": insurance_app_purchase_flag,
        "ecommerce_monthly_spend_inr": ecommerce_monthly_spend_inr,
        "ecommerce_order_frequency_monthly": ecommerce_order_frequency_monthly,
        "ecommerce_return_rate": ecommerce_return_rate,
        "food_delivery_order_count": food_delivery_order_count,
        "subscription_service_count": subscription_service_count,
        "employer_industry_category": employer_industry_category,
        "job_tenure_months": job_tenure_months,
        "salary_dispersion_index": salary_dispersion_index,
        "side_gig_income_presence": side_gig_income_presence,
        "freelance_income_ratio": freelance_income_ratio,
        "vehicle_ownership_flag": vehicle_ownership_flag,
        "mobile_recharge_frequency_monthly": mobile_recharge_frequency_monthly,
        "telecom_bill_payment_advance_days": telecom_bill_payment_advance_days,
        "data_usage_consistency_score": data_usage_consistency_score,
        "pan_card_availability_flag": pan_card_availability_flag,
        "driving_license_ownership": driving_license_ownership,
        "referrer_loan_default_flag": referrer_loan_default_flag,
        "family_credit_history_known": family_credit_history_known,
        "district_urbanization_level": district_urbanization_level,
        "mfi_branch_density": mfi_branch_density,
        "bank_branch_per_capita": bank_branch_per_capita,
        "pincode_agrarian_dominance": pincode_agrarian_dominance,
        "msme_density_score": msme_density_score,
        "festival_season_indicator": festival_season_indicator,
        "monsoon_onset_lag_days": monsoon_onset_lag_days,
        "harvest_calendar_region": harvest_calendar_region,
        "tourism_season_indicator": tourism_season_indicator,
    }, index=df.index)

    # Ensure column order matches declaration.
    synthetic = synthetic[[c for c in SYNTHETIC_URBAN_COLUMNS if c in synthetic.columns]]

    # Cohort mask: Rural rows → NaN for urban-only features.
    applicable = (df["area_type"].to_numpy() != "Rural")
    masked = apply_cohort_mask(synthetic, applicable)

    return masked


def synthesize_urban_features_main(
    input_path: str = "Data/raw/alternative_credit_scoring.csv",
    output_path: str = "data/processed/engineered_features.parquet",
    random_state: int = 42,
) -> pd.DataFrame:
    """Convenience entry point: load raw, synthesize urban cohort, write parquet.

    This is the standalone execution path used when the feature module is
    invoked directly (e.g. by ``main.py`` or manual CLI usage).
    """
    raw_df = load_raw_data(input_path)
    synthetic_df = synthesize_urban_features(raw_df, random_state=random_state)
    combined = pd.concat([raw_df, synthetic_df], axis=1)
    # Avoid duplicate column names if any overlap occurred (should not, guarded).
    combined = combined.loc[:, ~combined.columns.duplicated()]
    write_parquet(combined, output_path)
    return combined
