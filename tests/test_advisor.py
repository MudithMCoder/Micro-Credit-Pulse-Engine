"""Offline tests for Gemini advisory generation."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genai import advisor
from genai.prompts import AdvisoryContent


def _predictions(path: Path, count: int = 2) -> None:
    pd.DataFrame({
        "prediction": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0][:count],
        "prob_approved": [0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2][:count],
        "top3_positive": ["['payment_history']"] * count,
        "top3_negative": ["['late_payments']"] * count,
        "area_type": ["Rural", "Urban", "Rural", "Urban", "Rural", "Urban", "Rural", "Urban", "Rural", "Urban", "Rural", "Urban"][:count],
    }).to_csv(path, index=False)


def test_generate_advisories_uses_configured_model_and_area(tmp_path, monkeypatch):
    predictions_path = tmp_path / "predictions.csv"
    _predictions(predictions_path, count=2)
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return type("Response", (), {"text": json.dumps({
                "risk_assessment": "Assessment",
                "environmental_context": "Context",
                "recommendation": "Review repayment schedule",
                "confidence_level": "Medium",
            })})()

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.models = FakeModels()

    monkeypatch.setattr(advisor.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GRAPHIFY_GEMINI_MODEL", "test-model")

    result = advisor.generate_advisories(predictions_path, max_samples=1)

    assert len(result) == 1
    assert result.loc[0, "area_type"] == "Rural"
    assert result.loc[0, "environmental_context"] == "Context"
    assert calls[0]["model"] == "test-model"
    assert calls[0]["config"]["response_mime_type"] == "application/json"
    assert calls[0]["config"]["response_schema"] == AdvisoryContent
    assert "Area Type: Rural" in calls[0]["contents"]


def test_generate_advisories_first_10_only_mode(tmp_path, monkeypatch):
    predictions_path = tmp_path / "predictions.csv"
    _predictions(predictions_path, count=12)
    generation_counts = []

    class FakeModels:
        def generate_content(self, **kwargs):
            generation_counts.append(1)
            return type("Response", (), {"text": json.dumps({
                "risk_assessment": "Short risk summary",
                "environmental_context": "Stable seasonal factors",
                "recommendation": "Proceed with standard review",
                "confidence_level": "High",
            })})()

    class FakeClient:
        def __init__(self, *, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(advisor.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = advisor.generate_advisories(predictions_path, first_10_only=True)

    assert len(result) == 12
    assert len(generation_counts) == 10
    # First 10 rows should have generated advisories
    for i in range(10):
        assert result.loc[i, "risk_assessment"] == "Short risk summary"
    # Rows 10 and 11 should have NaN in advisory columns because they weren't generated
    for i in range(10, 12):
        assert pd.isna(result.loc[i, "risk_assessment"])
        assert pd.isna(result.loc[i, "recommendation"])


def test_generate_advisories_handles_rate_limit_gracefully(tmp_path, monkeypatch):
    predictions_path = tmp_path / "predictions.csv"
    _predictions(predictions_path, count=1)

    class FakeModels:
        def generate_content(self, **kwargs):
            raise Exception("429 RESOURCE_EXHAUSTED. Quota exceeded.")

    class FakeClient:
        def __init__(self, *, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(advisor.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = advisor.generate_advisories(predictions_path)

    assert len(result) == 1
    # Should fall back to structured rule-based advisory instead of failing/error
    assert "cohort" in result.loc[0, "risk_assessment"]
    assert result.loc[0, "confidence_level"] == "Medium"
