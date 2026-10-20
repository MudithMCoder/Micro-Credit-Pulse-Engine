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
    from google import genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

from .prompts import AdvisoryContent, LoanOfficerAdvisory

PREDICTIONS_PATH = Path("data/processed/model_predictions_with_shap.csv")
OUTPUT_PATH = Path("data/processed/final_loan_portfolio_report.csv")


def _get_gemini_client():
    """Create the Gemini API client using credentials from the environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY not set. "
            "Get free key: https://aistudio.google.com/app/apikey"
        )

    if not GOOGLE_GENAI_AVAILABLE:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    return genai.Client(api_key=api_key)


def _build_system_prompt() -> str:
    """System prompt with strict tone and output format guardrails."""
    return """You are an objective financial risk analyst providing loan officer advisory summaries.

TONE: Concise, factual, professional. No speculation. No marketing language.

INPUT: You will receive applicant data (area_type, prediction, SHAP drivers).

OUTPUT: Return ONLY valid JSON matching this schema:
{
  "risk_assessment": "1-2 sentence plain-language risk summary, maximum 200 characters",
  "environmental_context": "Seasonal/environmental repayment context, maximum 150 characters",
  "recommendation": "Actionable loan officer recommendation, maximum 200 characters",
  "confidence_level": "High" | "Medium" | "Low"
}

Character limits are strict maximums, including spaces and punctuation:
- risk_assessment: at most 200 characters
- environmental_context: at most 150 characters
- recommendation: at most 200 characters

RULES:
- risk_assessment: Translate model prediction + top drivers into plain language within 200 characters
- environmental_context: For Rural, mention seasonal factors; for Urban, mention digital behavior patterns; at most 150 characters
- recommendation: Concrete action (grace period, reduced amount, require co-signer, etc.) within 200 characters
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
    max_samples: Optional[int] = None,
    first_10_only: bool = True,
) -> pd.DataFrame:
    """Generate LLM advisories for model predictions.

    Args:
        predictions_path: Path to model_predictions_with_shap.csv
        max_samples: Limit returned data to N samples (for testing)
        first_10_only: Generate advisories for only the first 10 rows while retaining all rows

    Returns:
        DataFrame with predictions + advisory columns
    """
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}. "
            "Run model training first: python -m src.models.train"
        )

    client = _get_gemini_client()
    model_name = os.getenv("GRAPHIFY_GEMINI_MODEL", "gemini-flash-latest")
    system_prompt = _build_system_prompt()

    df = pd.read_csv(predictions_path)
    if max_samples is not None:
        df = df.head(max_samples).copy()

    generation_df = df.head(10) if first_10_only else df

    # Generate advisory for each selected row
    advisories = []
    for idx, row in generation_df.iterrows():
        user_prompt = _build_user_prompt(row, idx)

        try:
            # Request JSON output; Pydantic validates the returned fields below.
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\n{user_prompt}",
                config={
                    "response_mime_type": "application/json",
                    "response_schema": AdvisoryContent,
                },
            )

            advisory_json = json.loads(response.text)
            advisory = LoanOfficerAdvisory(
                applicant_id=int(idx),
                **advisory_json
            )
            advisories.append(advisory.model_dump())

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                print(f"[WARN] Rate limit / quota exceeded (429). Falling back to rule-based advisory for applicant {idx}.")
                # Rule-based fallback based on prediction and area type for free/quota protection
                pred_label = "Approved" if row.get("prediction", 1) == 1 else "Rejected"
                area_lbl = row.get("area_type", "Unknown")
                advisories.append({
                    "applicant_id": int(idx),
                    "risk_assessment": f"Applicant evaluated as {pred_label} in {area_lbl} cohort based on digital and financial indicators.",
                    "environmental_context": "Standard seasonal/digital behavior profile observed for cohort." if area_lbl != "Rural" else "Agricultural harvest and cooperative cycle considered.",
                    "recommendation": "Proceed with standard review and verify income documentation." if pred_label == "Approved" else "Manual review required; verify risk drivers.",
                    "confidence_level": "Medium"
                })
            else:
                # Fallback on other API errors
                advisories.append({
                    "applicant_id": int(idx),
                    "risk_assessment": f"Error generating advisory: {err_msg[:200]}",
                    "environmental_context": "N/A",
                    "recommendation": "Manual review required",
                    "confidence_level": "Low"
                })

    # Merge with predictions, retaining rows that were not sent for generation.
    advisory_df = pd.DataFrame(advisories).set_index("applicant_id")
    result = df.join(advisory_df, how="left")

    return result


def generate_portfolio_report(
    output_path: Path = OUTPUT_PATH,
    max_samples: Optional[int] = None,
    first_10_only: bool = False,
) -> Path:
    """Generate final portfolio report with predictions + advisories.

    Args:
        output_path: Where to save final report CSV
        max_samples: Limit to N samples (for testing/dev)

    Returns:
        Path to saved report
    """
    print("Generating loan officer advisories...")
    report = generate_advisories(
        max_samples=max_samples,
        first_10_only=first_10_only,
    )

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
