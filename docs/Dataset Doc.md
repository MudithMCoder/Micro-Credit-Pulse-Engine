# Dataset Documentation: alternative_credit_scoring.csv

## Source
`Data/raw/alternative_credit_scoring.csv` (original path: `Data/raw/alternative_credit_scoring.csv`)

## Overview
Synthetic / imported alternative-credit dataset for the Micro-Credit Pulse engine. Contains **2,000 borrower records** with **29 columns** covering demographic, digital-finance, cooperative, microfinance, and risk-label fields.

## Schema (29 columns)

| # | Column | Type (inferred) | Description |
|---|--------|-----------------|-------------|
| 1 | `age` | int | Borrower age (18–61 observed) |
| 2 | `monthly_income_inr` | float | Reported monthly income in INR |
| 3 | `area_type` | categorical | `Urban` / `Semi-Urban` / `Rural` |
| 4 | `upi_monthly_avg_txn_count` | int | Monthly UPI transaction count |
| 5 | `upi_avg_txn_amount` | float | Average UPI transaction amount (INR) |
| 6 | `upi_failed_txn_ratio` | float | Ratio of failed UPI transactions (0–0.57) |
| 7 | `mobile_recharge_consistency_score` | float | Consistency of mobile-recharge pattern (0–1) |
| 8 | `mobile_recharge_lapse_days_max` | int | Maximum lapse days between recharges |
| 9 | `utility_bills_on_time_pct` | float | % of utility bills paid on time (0–1) |
| 10 | `electricity_bill_avg_amount` | float | Average electricity bill (INR) |
| 11 | `cooperative_meeting_attendance_pct` | float | Cooperative meeting attendance (0–1) |
| 12 | `shg_savings_amount_inr` | float | Self-Help Group savings (INR) |
| 13 | `microfinance_loans_repaid_count` | int | Repayed microfinance loans |
| 14 | `microfinance_default_rate` | float | Historical default rate on micro-loans (0–0.55) |
| 15 | `gig_monthly_earnings` | float | Gig-economy monthly earnings (INR) |
| 16 | `gig_platform_rating` | float | Platform rating (1–5 observed) |
| 17 | `jandhan_avg_balance` | float | Avg Jan Dhan / bank balance (INR) |
| 18 | `jandhan_zero_balance_months` | int | Months with zero Jan Dhan balance |
| 19 | `monthly_data_usage_gb` | float | Monthly mobile-data usage (GB) |
| 20 | `telecom_bill_late_payments` | int | Late telecom payments count |
| 21 | `banking_app_login_frequency_monthly` | int | Monthly banking-app logins |
| 22 | `fintech_apps_count` | int | Installed fintech apps |
| 23 | `rent_on_time_payment_pct` | float | Rent paid on-time % (0–1) |
| 24 | `rent_to_income_ratio` | float | Rent / income ratio (0–0.69) |
| 25 | `social_referral_count` | int | Social-referral count |
| 26 | `insurance_policy_count` | int | Insurance policies held |
| 27 | `alternative_credit_score` | int | Alternative credit score (462–827) |
| 28 | `credit_risk_label` | categorical | `Very Low Risk` / `Low Risk` / `Medium Risk` / `High Risk` |
| 29 | `loan_approved` | binary (0/1) | 1 = approved, 0 = rejected (1,368 / 632 split) |

## Distribution Highlights

- **Records:** 2,000
- **Area split:** Rural 609 (30.5%), Semi-Urban 701 (35.0%), Urban 690 (34.5%)
- **Credit risk:** Low Risk 713, Medium Risk 1,028, High Risk 198, Very Low Risk 61
- **Loan approval:** Approved 1,368 (68.4%), Rejected 632 (31.6%)
- **Age range:** ~18–61 years
- **Income range:** ~3,000–34,000 INR/month
- **Alternative credit score:** 462–827
- **Microfinance default rate:** 0.0–0.55 (mean ~0.10)

## Null / Missing
No explicit null values detected in header rows; some numeric fields contain `0.0` for rural borrowers (e.g., `gig_monthly_earnings`, `jandhan_avg_balance`) reflecting structural exclusion from digital/gig channels — consistent with the project’s urban-bias research focus.

## Usage in Pipeline
- Raw input for `src/data_pipeline/generator.py` (reference / comparison dataset)
- Source for governance validation rules (`src/data_pipeline/governance.py`)
- Baseline for urban vs rural feature-engineering routes (`src/features/urban_features.py`, `src/features/rural_features.py`)
- Training / evaluation data for dual XGBoost cohort models (`src/models/train.py`)

## Notes
- File uses comma delimiter, no embedded commas within quoted fields in observed rows.
- Numerical fields are float/int; categorical fields are unquoted strings.
- Dataset is intended for synthetic / demonstration use within the Micro-Credit Pulse engine; no live borrower PII is included.
