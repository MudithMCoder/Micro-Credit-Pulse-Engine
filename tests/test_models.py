"""Tests for models (Module 4)."""
from pathlib import Path
import pandas as pd

def test_predictions_exist():
    assert Path("data/processed/model_predictions_with_shap.csv").exists()
    df = pd.read_csv("data/processed/model_predictions_with_shap.csv")
    assert len(df)==2000
    assert "top3_positive" in df.columns
