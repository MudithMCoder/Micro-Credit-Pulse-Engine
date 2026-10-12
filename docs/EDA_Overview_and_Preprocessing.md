# EDA Overview & Preprocessing / Feature Engineering Guide

Source: `notebooks/EDA.ipynb` (2,000 × 104 engineered features; 29 base + 75 synthesized).

---

## 1. EDA Overview (from notebook)

- **Cohort split:** Urban 690 (34.5%), Semi-Urban 701 (35.0%), Rural 609 (30.5%).
- **Target:** `loan_approved` binary; approval ~68.4% overall → class imbalance ~2.16:1 (`scale_pos_weight` = 2.16).
- **Credit risk:** Very Low 61 / Low 713 / Medium 1,028 / High 198.
- **Income:** Mean ~19.1K INR across cohorts (range ~3K–35K); rural mean slightly lower (~19.1K vs 19.3K urban) — not a strong separator by income alone.
- **Approval by cohort:** Rural 70.3%, Semi-Urban 68.3%, Urban 66.8% — rural slightly higher (likely due to lower default exposure via cooperative structures, not just income).
- **Structural zeros preserved** (not missing): `gig_monthly_earnings` ~55% zero across all cohorts; `jandhan_avg_balance` ~22–25% zero; rural digital-payment zeros reflect structural exclusion.
- **Cohort feature masking (proof from cell 3):** Urban/Semi-Urban hold digital footprint; Rural masked (100% NaN on `digital_engagement_index`, `wallet_volatility_score`, `ecommerce_monthly_spend`). Rural holds seasonal proxies; Urban masked (100% NaN on `crop_harvest_topup_ratio`, `community_trust_score`). Semi-Urban coarsened (~32% NaN).
- **Feature correlation with approval:** Positive: `side_gig_income_presence` (0.458), `gig_income_presence_flag` (0.456), `gig_platform_rating` (0.445), `rent_on_time_payment_pct` (0.286), `community_trust_score` (0.256). Negative: `procurement_payment_delay_avg_days` (-0.236), `telecom_bill_late_payments` (-0.211), `microfinance_default_rate` (-0.142).

---

## 2. Data Preprocessing (do before training / processing)

1. **Load:** `data/processed/engineered_features.parquet` (or `data/raw/alternative_credit_scoring.csv` for base). Use `pd.read_parquet()` / `pd.read_csv()`.
2. **Cohort routing:** Split by `area_type` (Urban / Semi-Urban / Rural). Do not mix feature subsets across cohorts when building dual models.
3. **Structural-zero flags (do NOT impute to mean):** Create indicators `gig_income_presence_flag`, `digital_payment_presence_flag` from zero-checks on `gig_monthly_earnings`, `upi_monthly_avg_txn_count`, `jandhan_avg_balance`. Preserve 0.0 values.
4. **Leakage exclusion:** Drop from feature sets: `alternative_credit_score`, `credit_risk_label`, `loan_approved` (target).
5. **Class imbalance:** Use `scale_pos_weight = 2.16` (XGBoost) or stratified sampling; do NOT downsample and lose rural representation.
6. **Missing / masked values:** Cohort-specific — urban digital cols may have NaN for rural rows; rural seasonal cols NaN for urban. Route by cohort before imputation; for semi-urban (~30% NaN), use median imputation by cohort or leave masked for dual-model routing.
7. **Bounds / sanity:** Confirm `age` 18–61, `monthly_income_inr` 3K–35K, `upi_failed_txn_ratio` 0–1, probability cols 0–1.
8. **Reproducibility:** `random_state=42` for split / synthesis.

---

## 3. Feature Engineering (cohort-routed per spec)

### Urban / Semi-Urban (Digital Footprint) — `src/features/urban_features.py`
- `digital_engagement_index` = `upi_monthly_avg_txn_count` × `upi_avg_txn_amount`
- Risk flag: `upi_failed_txn_ratio` > 0.2 → `failed_txn_volatility_flag`
- Banking/fintech adoption score from `banking_app_login_frequency_monthly`, `fintech_apps_count`, `mobile_recharge_consistency_score`
- Wallet volatility / ecommerce spend proxies

### Rural (Seasonal / Community) — `src/features/rural_features.py`
- `community_trust_score` from `cooperative_meeting_attendance_pct` + `shg_savings_amount_inr`
- Microfinance reliability: `microfinance_loans_repaid_count` vs `microfinance_default_rate`
- Seasonal: `crop_harvest_topup_ratio`, rainfall anomaly index, cooperative payout consistency
- Structural-exclusion flags (see preprocessing #3)

### Output
- Save engineered features to `data/processed/engineered_features.parquet` (already exists: 104 cols).
- For model input: use cohort-routed subsets (exclude masked NaN cols per cohort) or dual-model architecture.

---

## 4. Modeling & SHAP Reminders (from EDA cell 9 / spec Module 4)
- Dual XGBoost (`scale_pos_weight` 2.16).
- Top positive drivers: gig-income presence, rent-on-time, community trust.
- Top negative drivers: procurement delay, telecom late payments, microfinance default.
- SHAP extraction per prediction; save to `data/processed/model_predictions_with_shap.csv`.

---

Skipped: full HTML visualization note, plotly `fig.show()` fixes (nbformat missing — not data-related), additional statistical tests beyond correlation/boxplots shown. Risk: semi-urban masked features (~32% NaN) require careful routing; do not impute seasonal proxies into urban records or vice versa.

Co-Authored-By: Claude Code <noreply@anthropic.com>
🤖 Generated with [Claude Code](https://claude.com/claude-code)
