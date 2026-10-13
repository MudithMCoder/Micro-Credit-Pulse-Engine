"""LLM integration for Loan Officer Advisory summaries using Google Gemini API.

Reads model predictions + SHAP from train.py, generates plain-language advisories.
"""
from __future__ import annotations
import pandas as pd
import json
import os
from pathlib import Path
from typing import Optional

try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

from .prompts import LoanOfficerAdvisory

PREDICTIONS_PATH = Path("data/processed/model_predictions_with_shap.csv")
OUTPUT_PATH = Path("data/processed/final_loan_portfolio_report.csv")


def _get_gemini_model():
    """Initialize Gemini model with API key from environment using legacy google.generativeai."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY not set. "
            "Get free key: https://aistudio.google.com/app/apikey"
        )

    if not GOOGLE_GENAI_AVAILABLE:
        raise ImportError(
            "google-generativeai not installed. "
            "Run: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)

    # Try common model names in order of preference
    model_name = os.getenv("GRAPHIFY_GEMINI_MODEL", "gemini-1.5-flash")
    try:
        return genai.GenerativeModel(model_name)
    except Exception:
        # Fallback to other common model names if the primary fails
        fallback_models = ["gemini-pro", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-1.5-flash-latest"]
        for fallback in fallback_models:
            if fallback != model_name:  # Avoid trying the same model twice
                try:
                    print(f"[WARN] Model '{model_name}' failed, trying fallback: {fallback}")
                    return genai.GenerativeModel(fallback)
                except Exception:
                    continue  # Try next fallback
        # If all fallbacks fail, re-raise the original error
        raise Exception(f"All Gemini models failed. Check GRAPHIFY_GEMINI_MODEL env var or API key.")


def _build_system_prompt() -> str:
    """System prompt with strict tone and output format guardrails."""
    return """You are an objective financial risk analyst providing loan officer advisory summaries.

TONE: Concise, factual, professional. No speculation. No marketing language.

INPUT: You will receive applicant data (area_type, prediction, SHAP drivers).

OUTPUT: Return ONLY valid JSON matching this schema:
{
  "risk_assessment": "1-2 sentence plain-language risk summary",
  "environmental_context": "Seasonal/environmental factors affecting repayment (e.g., drought, harvest cycle)",
  "recommendation": "Actionable loan officer recommendation (e.g., grace period, collateral adjustment)",
  "confidence_level": "High" | "Medium" | "Low"
}

RULES:
- risk_assessment: Translate model prediction + top drivers into plain language
- environmental_context: For Rural, mention seasonal factors; for Urban, mention digital behavior patterns
- recommendation: Concrete action (grace period, reduced amount, require co-signer, etc.)
- confidence_level: High if SHAP drivers are clear + consistent; Medium if mixed; Low if ambiguous data
"""


def _build_user_prompt(row: pd.Series, idx: int) -> str:
    """Build per-applicant prompt with prediction + SHAP context."""
    # Parse SHAP lists (stored as strings in CSV)
    top_pos = eval(row["top3_positive"]) if isinstance(row["top3_positive"], str) else row["top3_positive"]
    top_neg = eval(row["top3_negative"]) if isinstance(row["top3_negative"], str) else row["top3_negative"]

    area = row.get("area_type", "Unknown")
    pred = "Approved" if row["prediction"] == 1 else "Rejected"
    prob = row["prob_approved"]

    return f"""Applicant ID: {idx}
Area Type: {area}
Model Prediction: {pred} (probability: {prob:.2f})

Top 3 Positive Risk Drivers (favor approval):
{chr(10).join(f'  - {f}' for f in top_pos)}

Top 3 Negative Risk Drivers (favor rejection):
{chr(10).join(f'  - {f}' for f in top_neg)}

Provide loan officer advisory in JSON format."""


def generate_advisories(
    predictions_path: Path = PREDICTIONS_PATH,
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """Generate LLM advisories for model predictions.

    Args:
        predictions_path: Path to model_predictions_with_shap.csv
        max_samples: Limit processing to N samples (for testing)

    Returns:
        DataFrame with predictions + advisory columns
    """
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}. "
            "Run model training first: python -m src.models.train"
        )

    model = _get_gemini_model()
    system_prompt = _build_system_prompt()

    df = pd.read_csv(predictions_path)
    if max_samples:
        df = df.head(max_samples)

    # Generate advisory for each row
    advisories = []
    for idx, row in df.iterrows():
        user_prompt = _build_user_prompt(row, idx)

        try:
            # Gemini structured output via response schema
            response = model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )

            advisory_json = json.loads(response.text)
            advisory = LoanOfficerAdvisory(
                applicant_id=int(idx),
                **advisory_json
            )
            advisories.append(advisory.model_dump())

        except Exception as e:
            # Fallback on API error
            advisories.append({
                "applicant_id": int(idx),
                "risk_assessment": f"Error generating advisory: {str(e)[:50]}",
                "environmental_context": "N/A",
                "recommendation": "Manual review required",
                "confidence_level": "Low"
            })

    # Merge with predictions
    advisory_df = pd.DataFrame(advisories)
    result = df.merge(advisory_df, left_index=True, right_on="applicant_id", how="left")

    return result


def generate_portfolio_report(
    output_path: Path = OUTPUT_PATH,
    max_samples: Optional[int] = None
) -> Path:
    """Generate final portfolio report with predictions + advisories.

    Args:
        output_path: Where to save final report CSV
        max_samples: Limit to N samples (for testing/dev)

    Returns:
        Path to saved report
    """
    print("Generating loan officer advisories...")
    report = generate_advisories(max_samples=max_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print(f"[OK] Portfolio report saved: {output_path}")
    print(f"  Total applicants: {len(report)}")
    print(f"  Approved: {(report['prediction'] == 1).sum()}")
    print(f"  Rejected: {(report['prediction'] == 0).sum()}")

    return output_path


if __name__ == "__main__":
    # Test with first 5 samples
    print("Testing GenAI advisor (5 samples)...")
    try:
        report_path = generate_portfolio_report(max_samples=5)
        print(f"\nTest successful. Full run: python -m src.genai.advisor")
    except ValueError as e:
        if "GEMINI_API_KEY" in str(e):
            print(f"\n{e}")
            print("\nTo test with Gemini:")
            print("  export GEMINI_API_KEY=your_key_here  # Linux/Mac")
            print("  set GEMINI_API_KEY=your_key_here     # Windows")
            print("  python -m src.genai.advisor")
        else:
            raise
