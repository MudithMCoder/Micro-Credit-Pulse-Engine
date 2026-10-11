"""Rural cohort feature synthesis.

Per ``docs/Synthesized_Features_Research.md`` (Part 1 — Seasonal
& Agricultural Proxy Metrics, plus Rural-side cross-cohort from
Part 3 and regional context from Part 4). All formulas derive from
base columns; leakage guard enforced by design.

Public entry point: ``synthesize_rural_features(df, random_state=42)``.
"""

from __future__ import annotations

from typing import List

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

SYNTHETIC_RURAL_COLUMNS: List[str] = [
    # 1.1 Seasonal income / cash-flow
    "crop_harvest_topup_ratio",
    "seasonal_income_variance",
    "harvest_month_income_indicator",
    "post_harvest_loan_repayment_ability",
    "crop_type_category",
    # 1.2 Cooperative / community / market
    "cooperative_payout_consistency_score",
    "cooperative_membership_tenure_months",
    "mandi_weekly_income_volatility",
    "procurement_payment_delay_avg_days",
    "fertilizer_subsidy_received_flag",
    "pmkisan_subsidy_receipt_flag",
    # 1.3 Weather / climate risk
    "rainfall_anomaly_index",
    "district_drought_exposure_count",
    "flood_exposure_flag",
    "irrigation_source_type",
    "crop_insurance_coverage_flag",
    "climate_adoption_score",
    # 1.4 Structural exclusion (zeros that matter — preserved)
    "gig_income_presence_flag",
    "bank_account_activity_flag",
    "digital_payment_presence_flag",
    "mobile_smartphone_ownership",
    # 1.5 SHG / microfinance extended
    "shg_meeting_attendance_streak",
    "shg_inter_loans_taken_count",
    "shg_loan_repayment_completion_rate",
    "jlg_joint_liability_group_member",
    "microfinance_institution_count",
    "mf_loan_utilization_rate",
    # 3.1 Rural-side household / demographics
    "household_dependent_count",
    "household_income_total_inr",
    "home_ownership_status",
    "education_level",
    # 3.2 Telecom / mobile (shared, rural-weighted)
    "telecom_switch_count",
    "virtual_number_presence",
    # 3.3 Rural-side social / referral
    "community_trust_score",
    # 3.4 Government-ID / verification (rural-weighted)
    "aadhaar_linked_accounts_count",
    "voter_id_verification_flag",
    # 4 Regional / macro (cohort-neutral — computed here as rural-side owner)
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


def synthesize_rural_features(
    df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Synthesize the Rural feature set from the raw frame.

    Args:
        df: Raw 29-column DataFrame. Must include ``area_type``.
        random_state: Reproducibility seed.

    Returns:
        DataFrame of synthesized columns; non-Rural rows (Urban / Semi)
        get ``np.nan`` via cohort-mask (with Semi-Urban coarsening —
        30% of synthetic rural values set to NaN to reflect hybrid
        transitional behaviour).
    """
    validate_required_columns(df, REQUIRED_BASE_COLUMNS, context="rural synthesis")
    if "area_type" not in df.columns:
        raise ValueError("Cohort routing requires column 'area_type'")

    for col in SYNTHETIC_RURAL_COLUMNS:
        if col in df.columns:
            raise ValueError(f"Synthetic column '{col}' already exists in input.")

    rng = _build_rng(random_state)
    n = len(df)

    # ----- 1.1 Seasonal income / harvest dynamics -----
    income = df["monthly_income_inr"].to_numpy("float64")
    age = df["age"].to_numpy("float64")
    coop_att = df["cooperative_meeting_attendance_pct"].to_numpy("float64")

    # Harvest top-up: 1.5-3x baseline for agricultural borrowers, scaled
    # by cooperative attendance (formal-group membership proxies harvest
    # participation and market access).
    crop_harvest_topup_ratio = np.clip(
        1.5 + coop_att * 1.5 + rng.normal(0.0, 0.30, n), 1.0, 3.0
    ).astype("float32")

    # Seasonal variance — lower-income rural borrowers have more volatile
    # monthly cash flows (crop-cycle dependency).
    seasonal_income_variance = np.clip(
        (income / 50000.0) * 0.4 + rng.uniform(0.0, 0.20, n), 0.0, 1.0
    ).astype("float32")

    # Harvest indicator — ~33% of observations fall in harvest windows
    # (Oct-Nov / Apr-May in Indian agricultural calendar).
    harvest_month_income_indicator = (
        (rng.random(n) < 0.33).astype("int8")
    )

    # Repayment ability tied to harvest cycle completion.
    post_harvest_loan_repayment_ability = np.clip(
        coop_att * 0.90 + rng.normal(0.0, 0.10, n), 0.0, 1.0
    ).astype("float32")

    # Crop-type draws weighted by income tier (higher-income → horticulture).
    crop_cats = ["Kharif", "Rabi", "Zaid", "Mixed", "Horticulture"]
    crop_weights = np.clip(
        np.stack([
            np.full(n, 0.40),
            np.full(n, 0.30),
            np.full(n, 0.15),
            np.full(n, 0.10),
            np.clip(income / 50000.0, 0.0, 1.0) * 0.05 + 0.05,
        ]).T,
        0.01,
        1.0,
    )
    crop_weights = crop_weights / crop_weights.sum(axis=1, keepdims=True)
    crop_type_category = np.empty(n, dtype=object)
    for i in range(n):
        crop_type_category[i] = rng.choice(crop_cats, p=crop_weights[i])
    crop_type_category = pd.Series(
        crop_type_category, index=df.index, dtype=object
    )

    # ----- 1.2 Cooperative & community proxies -----
    cooperative_payout_consistency_score = np.clip(
        0.70 + coop_att * 0.25 + rng.normal(0.0, 0.05, n), 0.0, 1.0
    ).astype("float32")

    cooperative_membership_tenure_months = np.clip(
        coop_att * 36.0 + rng.integers(0, 12, n).astype("float64"), 1, 480
    ).astype("int16")

    # Mandi (market) volatility — lower-income farmers face wider price swings.
    mandi_weekly_income_volatility = np.select(
        [income < 10000, income < 20000],
        [0.30, 0.20],
        default=0.15,
    ).astype("float32")

    procurement_payment_delay_avg_days = np.clip(
        15.0 - coop_att * 10.0 + rng.integers(0, 5, n).astype("float32"),
        0.0,
        30.0,
    ).astype("float32")

    fertilizer_subsidy_received_flag = (
        (coop_att > 0.50).astype("int8")
    )
    pmkisan_subsidy_receipt_flag = np.where(
        income < 15000,
        (rng.random(n) < 0.70).astype("int8"),
        (rng.random(n) < 0.30).astype("int8"),
    )

    # ----- 1.3 Weather / climate exposure -----
    # Rainfall anomaly from IMD-style index (±1.5 around normal).
    rainfall_anomaly_index = np.clip(
        rng.uniform(-1.50, 1.50, n), -2.0, 2.0
    ).astype("float32")

    district_drought_exposure_count = rng.integers(0, 3, size=n).astype("int8")

    flood_exposure_flag = (rng.random(n) < 0.20).astype("int8")

    irrigation_source_type = np.select(
        [
            rng.random(n) < 0.35,
            rng.random(n) < 0.30,
            rng.random(n) < 0.25,
        ],
        [
            np.full(n, "Rain-fed", dtype=object),
            np.full(n, "Canal", dtype=object),
            np.full(n, "Borewell", dtype=object),
        ],
        default=np.full(n, "Tank", dtype=object),
    )
    irrigation_source_type = pd.Series(
        irrigation_source_type, index=df.index, dtype=object
    )

    crop_insurance_coverage_flag = (
        (coop_att > 0.60).astype("int8")
    )

    climate_adoption_score = np.clip(
        coop_att * 0.80 + rng.normal(0.0, 0.10, n), 0.0, 1.0
    ).astype("float32")

    # ----- 1.4 Structural exclusion flags (do NOT impute zero base columns) -----
    # These flags preserve the meaning of structural zeros in the base data
    # (e.g. gig_monthly_earnings == 0 for rural borrowers = excluded, not missing).
    gig_earnings = df["gig_monthly_earnings"].to_numpy("float64")
    gig_income_presence_flag = (gig_earnings > 0.0).astype("int8")

    jandhan_bal = df["jandhan_avg_balance"].to_numpy("float64")
    jandhan_zero_months = df["jandhan_zero_balance_months"].to_numpy("float64")
    bank_account_activity_flag = (
        ((jandhan_bal > 100.0) & (jandhan_zero_months < 3)).astype("int8")
    )

    upi_txn = df["upi_monthly_avg_txn_count"].to_numpy("int16")
    digital_payment_presence_flag = (upi_txn > 0).astype("int8")

    data_usage = df["monthly_data_usage_gb"].to_numpy("float64")
    mobile_smartphone_ownership = (data_usage > 0.0).astype("int8")

    # ----- 1.5 SHG / microfinance extended -----
    shg_att = df["cooperative_meeting_attendance_pct"].to_numpy("float64")
    shg_savings = df["shg_savings_amount_inr"].to_numpy("float64")

    shg_meeting_attendance_streak = np.clip(
        shg_att * 12.0 + rng.integers(0, 6, n).astype("float64"), 0, 60
    ).astype("int8")

    shg_inter_loans_taken_count = np.clip(
        shg_savings / 5000.0, 0, 5
    ).astype("int8")

    default_rate = df["microfinance_default_rate"].to_numpy("float64")
    shg_loan_repayment_completion_rate = np.clip(
        1.0 - default_rate + rng.normal(0.0, 0.05, n), 0.0, 1.0
    ).astype("float32")

    jlg_joint_liability_group_member = (shg_savings > 5000.0).astype("int8")

    # MF institution count — region / area dependent (more rural MFIs).
    microfinance_institution_count = np.select(
        [
            df["area_type"].to_numpy() == "Rural",
            df["area_type"].to_numpy() == "Semi-Urban",
        ],
        [
            rng.integers(2, 5, size=n).astype("int8"),
            rng.integers(1, 4, size=n).astype("int8"),
        ],
        default=rng.integers(0, 2, size=n).astype("int8"),
    )

    repaid = df["microfinance_loans_repaid_count"].to_numpy("int16")
    mf_loan_utilization_rate = np.clip(
        safe_divide(repaid, repaid + 1, default=0.0) * (1.0 - default_rate),
        0.0,
        1.0,
    ).astype("float32")

    # ----- 3.1 Rural-side household -----
    # Household size derived from income tier (larger households at lower income).
    household_dependent_count = np.clip(
        np.select(
            [income < 10000, income < 20000],
            [rng.integers(3, 8, n), rng.integers(2, 5, n)],
            default=rng.integers(1, 4, n),
        ),
        0,
        10,
    ).astype("int8")

    # Total household income = individual + spouse / side-worker estimate.
    household_income_total_inr = np.clip(
        income * (1.2 + rng.random(n) * 1.3), 3000, 100000
    ).astype("float32")

    # Home-ownership — rural bias toward owned / parental.
    home_ownership_status = np.select(
        [
            rng.random(n) < 0.55,
            rng.random(n) < 0.15,
            rng.random(n) < 0.25,
        ],
        [
            np.full(n, "Owned", dtype=object),
            np.full(n, "Rented", dtype=object),
            np.full(n, "Parental", dtype=object),
        ],
        default=np.full(n, "PG", dtype=object),
    )
    home_ownership_status = pd.Series(
        home_ownership_status, index=df.index, dtype=object
    )

    # Education — rural skew lower.
    education_level = np.select(
        [
            rng.random(n) < 0.35,
            rng.random(n) < 0.30,
            rng.random(n) < 0.25,
        ],
        [
            np.full(n, "Below Secondary", dtype=object),
            np.full(n, "Secondary", dtype=object),
            np.full(n, "Graduate", dtype=object),
        ],
        default=np.full(n, "Post-Graduate", dtype=object),
    )
    education_level = pd.Series(education_level, index=df.index, dtype=object)

    # ----- 3.2 Telecom (rural-weighted) -----
    telecom_switch_count = np.clip(
        rng.integers(0, 3, size=n) + (default_rate * 2).astype("int8"), 0, 5
    ).astype("int8")

    virtual_number_presence = (rng.random(n) < 0.08).astype("int8")

    # ----- 3.3 Social / community trust -----
    # Aggregated from cooperative attendance + SHG savings (formal group = trust).
    shg_savings_norm = np.clip(shg_savings / 50000.0, 0.0, 1.0)
    community_trust_score = np.clip(
        0.50 * coop_att + 0.30 * shg_savings_norm + rng.normal(0.0, 0.05, n),
        0.0,
        1.0,
    ).astype("float32")

    # ----- 3.4 Government-ID verification -----
    aadhaar_linked_accounts_count = np.clip(
        rng.integers(0, 5, size=n).astype("int8"), 0, 10
    )
    voter_id_verification_flag = np.clip(
        (age >= 18).astype("float64") * 0.85 + rng.normal(0.0, 0.05, n),
        0.0,
        1.0,
    )
    voter_id_verification_flag = (voter_id_verification_flag > 0.5).astype("int8")

    # ----- 4 Regional / macro -----
    # Area-adjusted distributions (same columns, different weights).
    area = df["area_type"].to_numpy()
    district_urbanization_level = np.clip(
        np.select(
            [
                area == "Rural",
                area == "Semi-Urban",
            ],
            [
                0.15 + rng.uniform(-0.10, 0.10, n),
                0.40 + rng.uniform(-0.15, 0.15, n),
            ],
            default=0.60 + rng.uniform(-0.15, 0.15, n),
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

    # Agrarian dominance — high for rural, moderate for semi-urban, low urban.
    pincode_agrarian_dominance = np.clip(
        np.select(
            [
                area == "Rural",
                area == "Semi-Urban",
            ],
            [
                0.65 + rng.normal(0.0, 0.15, n),
                0.40 + rng.normal(0.0, 0.15, n),
            ],
            default=0.10 + rng.normal(0.0, 0.10, n),
        ),
        0.0,
        1.0,
    ).astype("float32")

    msme_density_score = np.clip(
        0.25 + 0.50 * district_urbanization_level + rng.normal(0.0, 0.10, n),
        0.0,
        1.0,
    ).astype("float32")

    festival_season_indicator = (rng.random(n) < 0.25).astype("int8")
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

    tourism_season_indicator = (rng.random(n) < 0.15).astype("int8")

    # ----- Assemble -----
    synthetic = pd.DataFrame({
        "crop_harvest_topup_ratio": crop_harvest_topup_ratio,
        "seasonal_income_variance": seasonal_income_variance,
        "harvest_month_income_indicator": harvest_month_income_indicator,
        "post_harvest_loan_repayment_ability": post_harvest_loan_repayment_ability,
        "crop_type_category": crop_type_category,
        "cooperative_payout_consistency_score": cooperative_payout_consistency_score,
        "cooperative_membership_tenure_months": cooperative_membership_tenure_months,
        "mandi_weekly_income_volatility": mandi_weekly_income_volatility,
        "procurement_payment_delay_avg_days": procurement_payment_delay_avg_days,
        "fertilizer_subsidy_received_flag": fertilizer_subsidy_received_flag,
        "pmkisan_subsidy_receipt_flag": pmkisan_subsidy_receipt_flag,
        "rainfall_anomaly_index": rainfall_anomaly_index,
        "district_drought_exposure_count": district_drought_exposure_count,
        "flood_exposure_flag": flood_exposure_flag,
        "irrigation_source_type": irrigation_source_type,
        "crop_insurance_coverage_flag": crop_insurance_coverage_flag,
        "climate_adoption_score": climate_adoption_score,
        "gig_income_presence_flag": gig_income_presence_flag,
        "bank_account_activity_flag": bank_account_activity_flag,
        "digital_payment_presence_flag": digital_payment_presence_flag,
        "mobile_smartphone_ownership": mobile_smartphone_ownership,
        "shg_meeting_attendance_streak": shg_meeting_attendance_streak,
        "shg_inter_loans_taken_count": shg_inter_loans_taken_count,
        "shg_loan_repayment_completion_rate": shg_loan_repayment_completion_rate,
        "jlg_joint_liability_group_member": jlg_joint_liability_group_member,
        "microfinance_institution_count": microfinance_institution_count,
        "mf_loan_utilization_rate": mf_loan_utilization_rate,
        "household_dependent_count": household_dependent_count,
        "household_income_total_inr": household_income_total_inr,
        "home_ownership_status": home_ownership_status,
        "education_level": education_level,
        "telecom_switch_count": telecom_switch_count,
        "virtual_number_presence": virtual_number_presence,
        "community_trust_score": community_trust_score,
        "aadhaar_linked_accounts_count": aadhaar_linked_accounts_count,
        "voter_id_verification_flag": voter_id_verification_flag,
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

    synthetic = synthetic[
        [c for c in SYNTHETIC_RURAL_COLUMNS if c in synthetic.columns]
    ]

    # Cohort masking: Rural = full; Semi-Urban = coarsened (30% NaN);
    # Urban = NaN. Implement via boolean + random mask for Semi.
    area = df["area_type"].to_numpy()
    applicable = area == "Rural"
    coarsen = (area == "Semi-Urban") & (rng.random(n) < 0.70)
    final_mask = applicable | coarsen
    masked = apply_cohort_mask(synthetic, final_mask)
    return masked


def synthesize_rural_features_main(
    input_path: str = "Data/raw/alternative_credit_scoring.csv",
    output_path: str = "data/processed/engineered_features.parquet",
    random_state: int = 42,
) -> pd.DataFrame:
    raw_df = load_raw_data(input_path)
    synthetic_df = synthesize_rural_features(raw_df, random_state=random_state)
    combined = pd.concat([raw_df, synthetic_df], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    write_parquet(combined, output_path)
    return combined
