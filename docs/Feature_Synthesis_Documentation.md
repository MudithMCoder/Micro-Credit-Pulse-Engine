# Feature Synthesis — Created Today

## Created Files
- `src/features/urban_features.py` — Urban/Semi-Urban digital-footprint synthesis (~27 features)
- `src/features/rural_features.py` — Rural seasonal/agricultural proxy synthesis (~37 features, structural-exclusion flags preserved)
- `src/features/_shared.py` — schema validation, safe division, cohort masking
- `src/features/__init__.py` — `synthesize_all_features()` orchestration
- `tests/test_features.py` — pytest suite (determinism, routing, leakage guard, structural zeros)
- `requirements.txt` — pandas, numpy, pyarrow

## What was synthesized (from docs/Synthesized_Features_Research.md)
- **Urban:** digital_engagement_index, wallet_volatility_score, recurring_payment_subscription_count, ecommerce_monthly_spend_inr, investment_app_ownership_flag, etc.
- **Rural:** crop_harvest_topup_ratio, rainfall_anomaly_index, cooperative_payout_consistency_score, gig_income_presence_flag (preserves structural zeros), shg_loan_repayment_completion_rate, etc.
- **Cross-cohort / Regional:** household_dependent_count, district_urbanization_level, festival_season_indicator, etc.

## How to run
```bash
pip install -r requirements.txt
python -c "
import sys; sys.path.insert(0,'src')
from features import synthesize_all_features
df = synthesize_all_features(random_state=42)
print('Shape:', df.shape)
"
```

## How to test
```bash
pytest tests/test_features.py -v
```
(Requires `pytest`; install with `pip install pytest` if not present.)

## Verification performed
- 2,000 rows × 104 columns (29 original + 75 synthesized)
- Cohort routing correct (Urban NaN for rural-only; Rural NaN for urban-only; Semi-Urban ~32% NaN coarsened)
- Determinism verified (same seed = identical frame)
- Structural zeros preserved (rural `gig_monthly_earnings` exactly 0 where original was 0)
- Leaking guard passed (0 features with |r| > 0.90 vs target/label)
- Schema validation raises `ValueError` correctly

Output: `data/processed/engineered_features.parquet`

---

## How the synthesized data is used for further processing

### Do NOT overwrite `Data/raw/alternative_credit_scoring.csv`
The raw CSV (2,000 × 29) is the **immutable source of truth** — it must stay untouched for governance, reproducibility, and audit. The 75 synthesized features are derived/computed, not imported, so they belong in the **processed** layer (`data/processed/`) rather than the **raw** layer.

### Architecture of usage (from CLAUDE.md)
```text
Data/raw/alternative_credit_scoring.csv
    ↓  load → synthesize (this pipeline)
Data/processed/engineered_features.parquet  ← 29 raw + 75 synthesized
    ↓  read → train / explain / advise
src/models/train.py  → dual XGBoost (Urban & Rural cohorts)
src/models/explainability.py → SHAP (top 3 positive / negative drivers)
src/genai/advisor.py → Loan Officer advisory summaries
```

### Processing paths
1. **Model training (`src/models/train.py`)**: Load `engineered_features.parquet`; exclude leakage columns (`alternative_credit_score`, `credit_risk_label`, `loan_approved`); use `scale_pos_weight` (1,368 / 632 ≈ 2.16) to handle class imbalance; train cohort-routed XGBoost on the 75 synthesized features plus eligible base columns.
2. **SHAP explainability (`src/models/explainability.py`)**: Use `shap.TreeExplainer` on the trained model; extract top 3 positive / top 3 negative drivers per applicant; output to `data/processed/model_predictions_with_shap.csv`.
3. **GenAI advisory (`src/genai/advisor.py`)**: Consume predictions + SHAP factors + `credit_risk_label`; feed to LLM via structured prompt (system guardrails from `prompts.py`); write summaries to `data/processed/final_loan_portfolio_report.csv`.
4. **BI / Dashboard**: Power BI / Tableau can read `engineered_features.parquet` or `final_loan_portfolio_report.csv` directly via parquet/csv connectors.

### Alternative ways (if you choose not to merge into parquet)
- **Side table**: Keep raw CSV clean; write synthesized features to `data/processed/synthesized_features.parquet`; merge at training time via `pd.merge(..., on='applicant_id')`. (This pipeline already produces the merged version for simplicity.)
- **In-memory pipeline**: Run `synthesize_urban_features()` / `synthesize_rural_features()` inside `main.py` sequentially (data generation → governance audit → feature pipeline → model scoring + SHAP → GenAI → final report). No file write required at intermediate steps except the parquet output for persistence.
- **Feature selection / dimensionality reduction**: Before modeling, apply `SelectKBest` or `PCA` on the 75 synthesized columns to reduce noise; the `structured_output` (`JSON` / Pydantic) from GenAI can also filter features by importance.

### Key rule
The synthesized features are **computational artifacts** — they must be reproducible (same `random_state` = same values) and version-controlled with the pipeline code (`src/features/`), not injected back into the raw dataset.
