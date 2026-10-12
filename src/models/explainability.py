"""SHAP interpreter helper (Module 4)."""
from __future__ import annotations
import pandas as pd

def load_shap(path="data/processed/model_predictions_with_shap.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # top3_positive / top3_negative stored as str; keep for report
    return df

def top_positive(row) -> list:
    s = str(row.get("top3_positive","[]"))
    # Simple parse not needed if already list-col; CSV may quote
    try:
        import ast
        return ast.literal_eval(s)
    except: return []
