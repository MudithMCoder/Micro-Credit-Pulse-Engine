"""Data Quality & Governance Gate (Module 2)."""
from __future__ import annotations
import os, json, pandas as pd
from pathlib import Path
from datetime import datetime, timezone

RAW = Path("data/raw/alternative_credit_scoring.csv")
AUDIT = Path("data/logs/governance_audit.json")

def validate() -> dict:
    df = pd.read_csv(RAW)
    audit = {"timestamp": datetime.now(timezone.utc).isoformat(), "file": str(RAW), "rows": len(df), "columns": len(df.columns), "checks": {}}
    # Schema
    expected = ["age","monthly_income_inr","loan_approved","alternative_credit_score","credit_risk_label","upi_failed_txn_ratio","area_type"]
    audit["checks"]["schema"] = "Pass" if all(c in df.columns for c in expected) else "Fail"
    audit["checks"]["age_bounds"] = "Pass" if df["age"].between(18,61).all() else "Fail"
    audit["checks"]["income_bounds"] = "Pass" if df["monthly_income_inr"].between(3000,34000).all() else "Fail"
    audit["checks"]["loan_binary"] = "Pass" if df["loan_approved"].isin([0,1]).all() else "Fail"
    audit["checks"]["credit_risk_labels"] = "Pass" if df["credit_risk_label"].isin(["Very Low Risk","Low Risk","Medium Risk","High Risk"]).all() else "Fail"
    # Null / split
    null_pct = (df.isnull().mean()*100).to_dict()
    audit["checks"]["null_over_5pct"] = {k:"Fail" if v>5 else "Pass" for k,v in null_pct.items() if v>0}
    audit["checks"]["area_split_approx"] = "Pass" if df["area_type"].value_counts(normalize=True).get("Urban",0) > 0.3 else "Fail"
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT,"w") as f: json.dump(audit, f, indent=2)
    return audit

if __name__ == "__main__":
    print(validate())
