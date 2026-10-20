# Module 5: GenAI Loan Officer Advisor

**Status**: ✅ BUILT (2026-10-13)
**Location**: `src/genai/`
**Description**: AI-driven loan officer advisory generation using Google Gemini API with structured output schemas

## Overview

The GenAI Loan Officer Advisor (M5) provides AI-powered, plain-language advisory summaries for loan officers, combining machine learning predictions with environmental and seasonal context for better credit decisions. This module addresses the project's goal of solving urban bias by providing context-aware explanations for both urban (digital behavior) and rural (seasonal factors) borrowers.

## What Was Built

### Core Components

#### 1. `src/genai/prompts.py`
- **Purpose**: Pydantic schemas for structured LLM output
- **Classes**:
  - `LoanOfficerAdvisory`: Main advisory model with 4 fields
  - `AdvisoryBatch`: Batch processing support
- **Guarded Output**: JSON schema ensures consistent, parseable responses

#### 2. `src/genai/advisor.py`
- **Purpose**: Google Gemini API integration for advisory generation
- **Key Features**:
  - Reads `model_predictions_with_shap.csv` (400 test samples)
  - Generates per-applicant 3-bullet advisories
  - Handles SHAP drivers for explainability
  - Writes `final_loan_portfolio_report.csv`
  - Graceful fallback on API errors

## Technical Implementation

### Data Flow

```
M4 Output: model_predictions_with_shap.csv
    ↓
M5 Input: Each test record (applicant_id, area_type, prediction, top3_positive/negative drivers)
    ↓
Gemini API: Structured generation per applicant
    ↓
M5 Output: final_loan_portfolio_report.csv
    (predictions + 3-part advisories + confidence_level)
```

### Structured Output Schema

Each advisory contains:

```json
{
  "applicant_id": 123,
  "risk_assessment": "Plain-language 1-2 sentence risk summary",
  "environmental_context": "Seasonal factors affecting repayment",
  "recommendation": "Actionable loan officer suggestion",
  "confidence_level": "High/Medium/Low"
}
```

### Key Features

#### Guardrails
- **Tone**: Objective, concise, professional financial risk analyst
- **Context**: Urban → digital patterns, Rural → seasonal/factors
- **Output Format**: Strictly 3-bullet format (risk, environment, recommendation)

#### Explainability Integration
- **SHAP Drivers**: Top 3 positive/negative drivers from XGBoost model
- **Area Type Awareness**: Different context for Urban vs Rural borrowers
- **Confidence Scoring**: Based on data clarity and driver consistency

#### Error Handling
- **API Key Check**: Validates `GEMINI_API_KEY` or `GOOGLE_API_KEY` env var
- **Fallback**: Returns error message and low confidence if API fails
- **Partial Success**: Continues processing even if some advisories fail

## Configuration & Setup

### Prerequisites

1. **Install GenAI Dependencies**
   ```bash
   pip install pydantic google-genai
   ```

2. **Set Environment Variable**
   ```bash
   # Linux/Mac
   export GEMINI_API_KEY=your_key_here
   
   # Windows PowerShell
   $env:GEMINI_API_KEY="your_key_here"
   ```

3. **Get Free Gemini API Key**
   - Visit: https://aistudio.google.com/app/apikey
   - Sign in with Google account
   - Copy generated API key

### Running Module 5

#### Full Pipeline (with M5)
```bash
cd "C:\Users\Mudith\Desktop\Micro Credit Pulse Engine"
python main.py
```

#### Skip GenAI (if API not configured)
```bash
python main.py --skip-genai
```

#### Test with Limited Samples
```powershell
$env:GRAPHIFY_GEMINI_MODEL="gemini-flash-latest"
python -m src.genai.advisor
```

## Generated Outputs

### Primary Output: `data/processed/final_loan_portfolio_report.csv`

**Structure**:
- **Prediction Columns**: `loan_approved`, `prediction`, `prob_approved`
- **Advisory Columns**: `risk_assessment`, `environmental_context`, `recommendation`, `confidence_level`
- **SHAP Drivers**: `top3_positive`, `top3_negative` (from M4)
- **Context**: `area_type` for routing awareness

**Sample Preview**:
```csv
applicant_id,loan_approved,prediction,prob_approved,area_type,risk_assessment,environmental_context,recommendation,confidence_level,top3_positive,top3_negative
123,1,1,0.987,Rural,"Low risk - strong community ties","Harvest season coincides with repayment timing","Offer 60-day grace period aligned with agricultural calendar","High","shg_savings_amount_inr,cooperative_meeting_attendance_pct,rent_on_time_payment_pct","microfinance_default_rate,jandhan_avg_balance,gig_monthly_earnings"
```

### Advisory Categories

#### Risk Assessment Examples
- **Urban**: "Low risk - consistent digital banking behavior"
- **Rural**: "Moderate risk - seasonal income variability"

#### Environmental Context Examples
- **Urban**: "High utility bill payment consistency, low fintech adoption"
- **Rural**: "Drought conditions in region X, harvest timing favorable"

#### Recommendation Examples
- **Grace Periods**: "Offer 60-day grace period aligned with harvest"
- **Collateral**: "Require additional guarantor due to income volatility"

## Integration with Power BI/Tableau

The `final_loan_portfolio_report.csv` is ready for BI tool ingestion:

- **Dashboard Metrics**: Applicant count, approval rate, confidence distribution
- **Explainability**: SHAP drivers as feature importance
- **Geographic**: Area_type breakdown for urban/rural insights
- **Seasonal**: Environmental context for time-based analysis

## Usage Examples

### 1. Generate Advisories for Specific Applicants
```python
from src.genai import generate_advisories

advisories = generate_advisories(
    predictions_path="data/processed/model_predictions_with_shap.csv",
    max_samples=10  # For testing
)
```

### 2. Export for Power BI
```python
from src.genai import generate_portfolio_report

report_path = generate_portfolio_report(
    output_path="reports/loan_portfolio_analysis.csv"
)
```

### 3. Configuration Testing
```python
import os
# Test with invalid API key
os.environ["GEMINI_API_KEY"] = "invalid_key"
try:
    from src.genai import generate_advisories
    generate_advisories(max_samples=2)
except ValueError as e:
    print(f"Expected error: {e}")
```

## Error Handling & Troubleshooting

### Common Issues

1. **Missing API Key**
   - **Error**: `GEMINI_API_KEY or GOOGLE_API_KEY not set`
   - **Fix**: Set environment variable as described above

2. **Invalid API Key**
   - **Error**: Google API authentication failure
   - **Fix**: Get valid key from Google AI Studio

3. **Model Not Trained**
   - **Error**: `FileNotFoundError: 'data/processed/model_predictions_with_shap.csv'`
   - **Fix**: Run pipeline with `--skip-genai` first to generate M4 predictions

### Debug Output
```bash
python -m src.genai.advisor
```

This tests the advisor with 5 samples and shows setup instructions if API key is missing.

## Technical Notes

### Architecture Decisions

1. **Pydantic Schemas**: Ensure consistent, validated output structure
2. **Gemini Integration**: Uses Google's generative AI with structured output
3. **Fallback Strategy**: Graceful degradation on API failures
4. **Testability**: Environment-based configuration for development/testing

### Performance Considerations

- **Processing**: Each advisory makes separate API call (can be optimized)
- **Rate Limits**: Gemini API has usage limits - consider batch processing
- **Caching**: Advisories could be cached for consistent predictions

### Security Considerations

- **API Key Management**: Key stored in environment variables, not code
- **Error Masking**: API errors don't expose sensitive information
- **Input Validation**: All inputs validated before processing

## Testing & Verification

### Unit Tests
```bash
pytest tests/test_features.py tests/test_models.py -v
```

### Integration Test
```bash
python main.py --skip-genai
```

### GenAI Module Test (with API key)
```bash
python -m src.genai.advisor
```

## Future Enhancements

1. **Batch Processing**: Process multiple advisories per API call
2. **Caching**: Cache generated advisories for consistent results
3. **Multiple Models**: Support different Gemini models (gemini-1.5-pro)
4. **Regional Context**: Add more granular environmental factors
5. **Customization**: Allow loan officers to customize advisory format

## Dependencies

### Required Libraries
- `pydantic`: Structured output validation
- `google-genai`: Gemini API integration
- `pandas`, `numpy`: Data processing
- `shap`, `xgboost`: From M4 predictions

### Updated `requirements.txt`
```txt
# Existing (M1-M4)
pandas
numpy
pyarrow
scikit-learn
xgboost
shap
pytest
matplotlib

# Added (M5)
pydantic
google-genai
```

## Related Documentation

- **M1-M4 Docs**: `docs/` folder
- **Test Suite**: `docs/TEST_RUN.md`
- **Dataset Info**: `docs/Dataset Doc.md`
- **Graph Documentation**: `graphify-out/GRAPH_REPORT.md`

## Quick Start

```bash
# 1. Set up environment
cd "C:\Users\Mudith\Desktop\Micro Credit Pulse Engine"
export GEMINI_API_KEY=your_key

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run full pipeline (M1-M5)
python main.py

# 4. Access results
ls data/processed/
# Shows: final_loan_portfolio_report.csv
```

## Conclusion

Module 5 successfully completes the Micro-Credit Pulse Engine by adding AI-powered loan officer advisory generation. The GenAI advisor:

- ✅ Provides human-readable explanations for ML predictions
- ✅ Maintains urban/rural contextual awareness
- ✅ Integrates seamlessly with M1-M4 pipeline
- ✅ Generates BI-ready output files
- ✅ Handles errors gracefully
- ✅ Supports both development and production environments

The system is production-ready and can be deployed with a Google Gemini API key to generate actionable loan officer advisories at scale.