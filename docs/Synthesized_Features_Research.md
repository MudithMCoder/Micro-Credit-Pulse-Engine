# Synthesized Features Research: Rural vs Urban Credit Scoring

## Executive Summary

This document outlines the **features that should be synthesized** to enhance the Micro-Credit Pulse alternative credit scoring dataset. The research addresses the core challenge: **Urban Bias of Alternative Credit Data** — the fact that most alternative credit scoring models favor urban borrowers who have rich digital footprints, while rural borrowers with informal income sources and seasonal cash flows are systematically underserved.

---

## Part 1: Rural Borrower Features (Seasonal & Agricultural Metrics)

### 1.1 Seasonal Income & Cash Flow Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `crop_harvest_topup_ratio` | float | Ratio of harvest-season income boost to baseline monthly income | Synthesized: harvest months show 1.5-3x income multiplier for agricultural borrowers |
| `seasonal_income_variance` | float | Standard deviation of monthly income across seasons (0-1) | Higher for farmers, lower for salaried workers |
| `harvest_month_income_indicator` | binary | Binary flag for harvest season (Oct-Nov, Apr-May in India) | 1 if income recorded during harvest months |
| `post_harvest_loan_repayment_ability` | float | Likelihood of repayment based on harvest timing (0-1) | Correlates with crop cycle completion |
| `crop_type_category` | categorical | Type of crop: Kharif / Rabi / Zaid / Mixed / Horticulture | Determines income seasonality pattern |

### 1.2 Agricultural Cooperative & Community Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `cooperative_payout_consistency_score` | float | Consistency of cooperative society payouts (0-1) | 0.8+ for regular monthly dairy/poultry cooperatives |
| `cooperative_membership_tenure_months` | int | Months of active cooperative membership | Longer tenure = more reliable payment history |
| `mandi_weekly_income_volatility` | float | Weekly income volatility from market (mandi) sales | Measured from price fluctuations |
| `procurement_payment_delay_avg_days` | int | Average days between produce delivery and payment | Longer delays indicate liquidity stress |
| `fertilizer_subsidy_received_flag` | binary | Whether farmer receives government fertilizer subsidy | 1 = access to welfare programs = lower risk |
| `pmkisan_subsidy_receipt_flag` | binary | PM-KISAN direct income support receipt | 1 = eligible for government support |

### 1.3 Weather & Climate Risk Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `rainfall_anomaly_index` | float | Deviation from normal rainfall (-2 to +2 scale) | Negative = drought risk, positive = flood risk |
| `district_drought_exposure_count` | int | Number of drought years in district (last 5 years) | 0-5 scale |
| `flood_exposure_flag` | binary | Whether borrower's district is flood-prone | 1 for flood-prone areas |
| `irrigation_source_type` | categorical | Source: Rain-fed / Canal / Borewell / Tank | Rain-fed = highest income volatility |
| `crop_insurance_coverage_flag` | binary | Whether crop insurance is active | 1 = risk mitigation present |
| `climate_adoption_score` | float | Adoption of climate-smart practices (0-1) | Higher = lower long-term risk |

### 1.4 Structural Exclusion Indicators (Zeros that matter)

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `gig_income_presence_flag` | binary | Whether gig_monthly_earnings > 0 | 0 for structurally excluded rural borrowers |
| `bank_account_activity_flag` | boolean | Whether jandhan_avg_balance > 100 INR | Inactive accounts flag financial exclusion |
| `digital_payment_presence_flag` | binary | Whether UPI transaction count > 0 | 0 = completely cash-based economy |
| `mobile_smartphone_ownership` | binary | Whether smartphone ownership (enables digital payments) | 0 for feature-phone only users |

### 1.5 Self-Help Group (SHG) & Microfinance Extended Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `shg_meeting_attendance_streak` | int | Consecutive months of SHG meeting attendance | 6+ months = high community trust |
| `shg_inter_loans_taken_count` | int | Number of intra-SHG loans taken | Shows active participation |
| `shg_loan_repayment_completion_rate` | float | % of SHG loans repaid on time | 0-1, similar to microfinance_default_rate |
| `jlg_joint_liability_group_member` | binary | Whether part of JLG for joint liability | 1 = access to group lending |
| `microfinance_institution_count` | int | Number of MFIs active in borrower area | More options = better access |
| `mf_loan_utilization_rate` | float | Utilization of previous microfinance (0-1) | High utilization with repayment = good signal |

---

## Part 2: Urban Borrower Features (Digital Footprint & Transactional Metrics)

### 2.1 UPI & Digital Payment Behavioral Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `digital_engagement_index` | float | UPI txn count × avg txn amount (normalized) | Combines volume and value |
| `upi_transaction_velocity` | float | Transactions per day (avg monthly / 30) | Higher = more integrated into digital economy |
| `wallet_volatility_score` | float | Standard deviation of wallet balance across month | Lower = more stable financial behavior |
| `cross_platform_payment_count` | int | Number of different UPI apps used | 3+ apps = sophisticated user |
| `recurring_payment_subscription_count` | int | Number of active recurring payments (SIP, rent, EMI) | Shows financial discipline |
| `utility_payment_regularity_score` | float | Consistency of utility bill payments (derived from on_time_pct) | 0.8+ = reliable |
| `rent_payment_advance_days` | int | Days before due date rent is paid | Positive = financial discipline |

### 2.2 Banking App & Fintech Engagement Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `banking_app_session_duration_avg` | float | Average time spent in banking app (minutes) | Longer = more engaged |
| `fintech_apps_installed_count` | int | Number of fintech apps (existing: fintech_apps_count) | Expand with sub-category |
| `loan_product_inquiry_count` | int | Number of loan product views in last 30 days | High intent = potential borrower |
| `credit_score_inquiry_own_count` | int | Times borrower checked own credit score | Shows financial awareness |
| `investment_app_ownership_flag` | boolean | Whether investment/trading app installed | 1 = financial planning orientation |
| `insurance_app_purchase_flag` | boolean | Whether purchased insurance via app | Shows risk management behavior |

### 2.3 E-Commerce & Platform Engagement Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `ecommerce_monthly_spend_inr` | float | Total monthly e-commerce spending | Higher spend + income ratio = disposable income signal |
| `ecommerce_order_frequency_monthly` | int | Number of orders per month | Behavioral pattern |
| `ecommerce_return_rate` | float | Percentage of orders returned | High return rate = decision inconsistency |
| `food_delivery_order_count` | int | Monthly food delivery orders | Discretionary spending indicator |
| `subscription_service_count` | int | Active subscriptions (OTT, music, news) | Regular cash flow commitment |

### 2.4 Employment & Income Stability Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `employer_industry_category` | categorical | Industry: IT / Manufacturing / Retail / Services / Construction | Industry risk profile |
| `job_tenure_months` | int | Months at current employer | Longer = more stable |
| `salary_dispersion_index` | float | Variation in monthly salary deposits | Low = stable income |
| `side_gig_income_presence` | binary | Whether gig_monthly_earnings > 0 | Urban side gig indicator |
| `freelance_income_ratio` | float | Freelance / total income ratio | Higher = variable income |

---

## Part 3: Cross-Cohort Features (Applicable to Both)

### 3.1 Demographic & Household Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `household_dependent_count` | int | Number of dependents | Higher = more financial obligation |
| `household_income_total_inr` | float | Combined household monthly income | More accurate than individual |
| `home_ownership_status` | categorical | Owned / Rented / Parental / PG | Stability indicator |
| `vehicle_ownership_flag` | boolean | Two-wheeler / Four-wheeler ownership | Asset indicator |
| `education_level` | categorical | Below Secondary / Secondary / Graduate / Post-Graduate | Employment quality indicator |

### 3.2 Mobile & Telecom Behavioral Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `mobile_recharge_frequency_monthly` | int | Number of recharges per month | Regular = consistent cash flow |
| `telecom_bill_payment_advance_days` | int | Days before due date bill paid | Financial discipline signal |
| `data_usage_consistency_score` | float | Month-over-month data usage variance | Low variance = predictable behavior |
| `telecom_switch_count` | int | Number of SIM swaps in last 2 years | High = unstable |
| `virtual_number_presence` | binary | Whether using VoIP/virtual number | Potential fraud indicator |

### 3.3 Social Graph & Referral Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `social_referral_count` | int | Existing (social_referral_count) - expand | Add quality metrics |
| `referrer_loan_default_flag` | binary | Whether referrer has defaulted | Risk propagation |
| `community_trust_score` | float | Aggregated trust from SHG/cooperative members | Community reputation |
| `family_credit_history_known` | binary | Whether family has known credit history | Hidden leverage |

### 3.4 Government ID & Verification Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `aadhaar_linked_accounts_count` | int | Number of Aadhaar-linked accounts | Financial inclusion depth |
| `pan_card_availability_flag` | boolean | Whether PAN card obtained | Tax compliance indicator |
| ` voter_id_verification_flag` | boolean | Whether Voter ID verified | Identity verification |
| `driving_license_ownership` | boolean | Whether driving license held | Asset/employability indicator |

---

## Part 4: Macroeconomic & Regional Context Features

### 4.1 Regional Economic Indicators

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `district_urbanization_level` | float | % urban population in district (0-1) | Structural factor |
| `mfi_branch_density` | int | Microfinance branches per 1000 households | Access to formal credit |
| `bank_branch_per_capita` | float | Bank branches per 10,000 population | Traditional banking access |
| `pincode_agrarian_dominance` | float | % employment in agriculture | Structural rural indicator |
| `msme_density_score` | float | MSME establishments per capita | Local economy vibrancy |

### 4.2 Seasonal Calendar Features

| Feature Name | Type | Description | Data Source Logic |
|--------------|------|-------------|-------------------|
| `festival_season_indicator` | binary | Whether data point is in festival season (Oct-Dec) | Spike in spending/borrowing |
| `monsoon_onset_lag_days` | int | Days from normal monsoon onset | Agricultural impact |
| `harvest_calendar_region` | categorical | Rabi/Kharif/Zaid harvest timing by region | Cash flow timing |
| `tourism_season_indicator` | binary | Whether region is tourism-dependent | Seasonal employment |

---

## Part 5: Priority Feature Synthesis Recommendations

### High Priority (Core to Urban Bias Research)

1. **For Rural Borrowers:**
   - `crop_harvest_topup_ratio` — addresses seasonal income bias
   - `rainfall_anomaly_index` — environmental risk quantification
   - `cooperative_payout_consistency_score` — formalizes informal credit history
   - Structural exclusion flags — ensures zero ≠ missing

2. **For Urban Borrowers:**
   - `digital_engagement_index` — aggregates digital footprint
   - `wallet_volatility_score` — behavioral stability metric
   - `recurring_payment_subscription_count` — financial discipline

### Medium Priority (Enhances Model Performance)

3. Cross-cohort features:
   - `household_dependent_count`
   - `home_ownership_status`
   - `education_level`

### Lower Priority (Nice-to-Have for Full Picture)

4. Regional context:
   - `district_urbanization_level`
   - `mfi_branch_density`

---

## Part 6: Implementation Notes

### Data Synthesis Strategy

- **Rural seasonal features**: Use sinusoidal patterns with 6-month periodicity to simulate crop cycles
- **Cooperative metrics**: Model as binomial distributions with 0.7-0.9 success rates
- **Rainfall anomaly**: Use historical district-level data; synthesize using uniform distribution ±1.5 from mean
- **Structural zeros**: Maintain exact zeros for rural borrowers in `gig_monthly_earnings`, `jandhan_avg_balance` — do not impute

### Validation Requirements

- Verify no data leakage from `alternative_credit_score` or `credit_risk_label` into synthesized features
- Ensure cross-cohort feature distributions are statistically distinct (p < 0.05 for t-test)
- Validate structural zero columns remain zero for appropriate borrower types

---

## Appendix: References

- Reserve Bank of India (RBI) - Priority Sector Lending Guidelines
- GSMA Mobile Money Metrics
- MicroFinance Institutions Network (MFIN) - Indian Microfinance Statistics
- Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) Program Data
- India Meteorological Department (IMD) - Rainfall Anomaly Index
- NITI Aayog - Financial Inclusion Data

---

*Document prepared for Micro-Credit Pulse Engine*
*Research Focus: Addressing Urban Bias in Alternative Credit Data*