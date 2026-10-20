"""Tests for models (Module 4)."""
from pathlib import Path
import pandas as pd

def test_predictions_exist():
    assert Path("data/processed/model_predictions_with_shap.csv").exists()
    df = pd.read_csv("data/processed/model_predictions_with_shap.csv")
    assert len(df)==400  # Test set only (80/20 split)
    assert "top3_positive" in df.columns
    assert "top3_negative" in df.columns
    assert "split" in df.columns
    assert all(df["split"] == "test")
    assert {"cv_accuracy_mean", "cv_accuracy_std", "cv_roc_auc_mean", "cv_roc_auc_std"} <= set(df.columns)
