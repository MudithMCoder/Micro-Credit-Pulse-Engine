"""Pipeline tests (M2+M3 routing + M4)."""
from pathlib import Path
import pandas as pd

def test_governance_audit():
    assert Path("data/logs/governance_audit.json").exists()

def test_engineered_features():
    assert Path("data/processed/engineered_features.parquet").exists()

def test_predictions():
    assert Path("data/processed/model_predictions_with_shap.csv").exists()
