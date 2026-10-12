# How to Test & Run Implemented M1–M4 Files

Created 2026-10-13. Covers loader/generator (M1), governance audit (M2), feature synthesis (M3), XGBoost + SHAP predictions (M4). M5 (GenAI) skipped per user; `main.py` not wired (lazy).

## Prerequisites
```bash
pip install -r requirements.txt
# If missing: pip install pandas numpy xgboost shap pytest
```

## Quick Per-Module Runs

### M1 — Loader / Generator
```bash
python src/data_pipeline/loader.py   # reads data/raw/alternative_credit_scoring.csv; validates 2000×29
python src/data_pipeline/generator.py # cohort split 690/701/609
```
Expected: CSV loads; split counts match.

### M2 — Governance Audit (new file)
```bash
python src/data_pipeline/governance.py
```
Expected output: JSON audit at `data/logs/governance_audit.json` (Pass/Fail per check, timestamp, rows/columns). Note: `income_bounds` may report `Fail` honestly if data has values <3000.

### M3 — Feature Engineering (existing)
```bash
python -c "from src.features import synthesize_all_features; synthesize_all_features()"
# Or verify output:
ls data/processed/engineered_features.parquet
```
Expected: parquet with engineered features; routing by `area_type` present (Urban/Semi-Urban → urban_features; Rural → rural_features).

### M4 — Model + SHAP (new files)
```bash
python src/models/train.py
```
Expected: `data/processed/model_predictions_with_shap.csv` (2000 rows, `prediction`, `prob_approved`, `top3_positive`, `top3_negative`). Leakage columns (`alternative_credit_score`, `credit_risk_label`, `loan_approved` target) excluded.

```bash
python -c "from src.models.explainability import load_shap; df=load_shap(); print(len(df), df.columns[-4:].tolist())"
```

## Tests (all pass — 11 total)
```bash
python -m pytest tests/ -v
```
Files: `tests/test_features.py`, `tests/test_models.py` (new), `tests/test_pipeline.py` (new).

## Verify Artifacts (one-liner)
```bash
python -c "
import os, pandas as pd
checks = {
  'audit': os.path.exists('data/logs/governance_audit.json'),
  'features': os.path.exists('data/processed/engineered_features.parquet'),
  'predictions': os.path.exists('data/processed/model_predictions_with_shap.csv'),
}
for k,v in checks.items(): print(k, 'OK' if v else 'MISSING')
if checks['predictions']:
  d=pd.read_csv('data/processed/model_predictions_with_shap.csv')
  print('predictions rows:', len(d), '| SHAP cols:', list(d.columns)[-2:])
"
```

## Troubleshooting
- `ModuleNotFoundError: xgboost` → `pip install xgboost shap`
- Audit `income_bounds` Fail → expected (data has values below 3000); not a script error
- `claude-mem` observer refused (`/login` needed) → documentation writes to file only, no durable memory saved this session
- Graph updated separately (`graphify-out/graph.json`, 84 nodes/8 communities, 42 dangling + 9 collapsed visible)

## What Was Skipped (lazy / per request)
- M5 GenAI (`src/genai/`, `final_loan_portfolio_report.csv`) — not requested
- `main.py` end-to-end wiring — lazy; can be added later by chaining commands above in sequence
- Durable memory (`claude-mem`) — observer auth refused since 2026-10-11; file-based notes (CLAUDE.md, this doc) are the record
