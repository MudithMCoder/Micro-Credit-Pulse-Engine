# Micro-Credit Pulse Engine

Alternative credit scoring with dual-path AI for urban & rural borrowers.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-11%2F11%20passing-brightgreen.svg)

## 🎯 Mission

Solve the **urban bias** in alternative credit data by implementing context-appropriate feature engineering:

- **Urban/Semi-Urban**: 30-day rolling digital footprints (UPI, e-commerce, utility consistency)
- **Rural**: Seasonal burst metrics (harvest ratios, cooperative payouts, rainfall anomaly indexes)

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run full pipeline (M1–M4, skip GenAI if no API key)
python main.py --skip-genai

# Run all tests
pytest tests/ -v
```

## 📦 Features

| Module | Description |
|--------|-------------|
| **M1** | Data loading & validation (2,000 × 29 CSV) |
| **M2** | Governance audit with schema & bounds checks |
| **M3** | Dual feature engineering: urban digital + rural seasonal |
| **M4** | XGBoost models + SHAP explainability (90% acc, 0.9647 AUC) |
| **M5** | GenAI loan officer advisories (requires Gemini API key) |

## 📁 Output Artifacts

- `data/logs/governance_audit.json` — Data quality audit (M2)
- `data/processed/engineered_features.parquet` — 104 feature columns (M3)
- `data/processed/model_predictions_with_shap.csv` — 400 test predictions + SHAP (M4)
- `data/processed/model_evaluations.json` — Model metrics + plots (M4)
- `data/processed/final_loan_portfolio_report.csv` — Predictions + AI advisories (M5)

## 🏗️ Architecture

```
Micro Credit Pulse Engine/
├── data/          # raw CSV, processed parquet, audit logs
├── src/           # 5 modules: data_pipeline, features, models, genai
├── tests/         # 11/11 passing test suite
├── docs/          # Full documentation (PROJECT_DOCUMENTATION.md)
└── main.py        # End-to-end pipeline orchestrator
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[PROJECT_DOCUMENTATION.md](docs/Project-Documentation.md)** — Complete project reference covering architecture, modules, dataset, performance, setup, and future enhancements
- **Dataset Doc.md** — Full 29-column schema and distributions of the raw alternative credit dataset
- **Feature Synthesis Documentation.md** — Details on engineered features for urban and rural cohorts
- **EDA Overview.md** — Exploratory analysis and preprocessing steps
- **Gen AI Doc.md** — Loan advisor module specifications and prompt engineering
- **Synthesized Features Research.md** — Research foundation for urban-bias mitigation features

## 🔧 Tech Stack

- **Language**: Python 3.10+
- **Data**: pandas, numpy, pyarrow
- **ML**: xgboost, scikit-learn, shap
- **Explainability**: SHAP TreeExplainer
- **GenAI**: Google Gemini API, pydantic
- **Testing**: pytest
- **Visualization**: matplotlib

## ⚠️ Known Limitations

- GenAI requires external API key (free tier available)
- Semi-Urban coarsening (~30% NaN) needs careful routing
- Single train/test split (no cross-validation)
- Simple label encoding for categorical features

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 90.0% |
| **AUC** | 0.9647 |
| **Test Samples** | 400 (80/20 split) |

**Top Positive Drivers**: gig income presence, rent-on-time payments, community trust  
**Top Negative Drivers**: procurement delays, telecom late payments, microfinance defaults

## 🛠️ Development

```bash
# Full pipeline
python main.py --help  # shows: --skip-governance --skip-train --run-genai --random-state

# Feature synthesis only
python -c "from features import synthesize_all_features; df = synthesize_all_features(random_state=42); print(df.shape)"

# Run tests
pytest tests/ -v  # 11 tests passing
```

## 📄 License

See individual module licenses in source code headers.

## 🙋 Support

- Review test logs for debugging
- Check `docs/` for usage examples  
- Verify API key configuration for GenAI

---

*Built with ❤️ by expert Principal Data Engineer and Machine Learning Architect. Focus: Solving urban bias in alternative credit data with dual-path AI pipelines.*