"""Dual XGBoost + SHAP (Module 4)."""
from __future__ import annotations
import pandas as pd, numpy as np, xgboost as xgb, shap, os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

# Use SYNTHESIZED features, not raw data
FEATURES_PATH = Path("data/processed/engineered_features.parquet")
OUT = Path("data/processed/model_predictions_with_shap.csv")

# Leakage columns to exclude (derived from target)
LEAKAGE = ["alternative_credit_score", "credit_risk_label", "loan_approved"]

def train_and_explain():
    # Load SYNTHESIZED features
    df = pd.read_parquet(FEATURES_PATH)

    # Target column
    y = df["loan_approved"]

    # Filter out leakage columns from features
    feature_cols = [c for c in df.columns if c not in LEAKAGE and c != "loan_approved"]
    X = df[feature_cols].copy()

    # Handle categorical columns - convert to numeric
    cat_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    for col in cat_cols:
        X[col] = pd.Categorical(X[col]).codes  # simple label encoding

    X = X.fillna(0)

    # Train/test split (80/20) for honest evaluation
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance: scale_pos_weight for minority class
    # Training set class counts
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos  # weight for positive class

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42,
        use_label_encoder=False, eval_metric="logloss"
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"}
    )
    model.fit(X_train, y_train)

    # Predict on TEST set only (honest evaluation)
    probs_test = model.predict_proba(X_test)[:, 1]

    # SHAP explanations on test set
    explainer = shap.TreeExplainer(model)
    sv_test = explainer.shap_values(X_test)
    if isinstance(sv_test, list):
        sv_test = sv_test[1]  # positive class

    # Top 3 drivers per test sample
    pos3 = []
    neg3 = []
    for i in range(len(X_test)):
        s = sv_test[i]
        idx_sorted = np.argsort(s)[::-1]
        pos3.append([X_test.columns[j] for j in idx_sorted[:3]])
        idx_sorted_neg = np.argsort(s)
        neg3.append([X_test.columns[j] for j in idx_sorted_neg[:3]])

    # Build output dataframe (TEST SET ONLY)
    out = df.loc[idx_test, ["loan_approved"]].copy()
    out["prediction"] = (probs_test > 0.5).astype(int)
    out["prob_approved"] = probs_test
    out["top3_positive"] = pos3
    out["top3_negative"] = neg3
    out["split"] = "test"

    out["cv_accuracy_mean"] = cv_scores["test_accuracy"].mean()
    out["cv_accuracy_std"] = cv_scores["test_accuracy"].std()
    out["cv_roc_auc_mean"] = cv_scores["test_roc_auc"].mean()
    out["cv_roc_auc_std"] = cv_scores["test_roc_auc"].std()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    return out, model

if __name__ == "__main__":
    res, model = train_and_explain()
    print(f"Test set predictions: {len(res)} rows (20% holdout)")
    print(f"Test accuracy: {(res['prediction'] == res['loan_approved']).mean():.4f}")
    print(f"Test approval rate: {res['prediction'].mean():.2f}")
    print("Saved:", OUT)
    print("\nNote: Predictions are TEST SET only for honest evaluation.")
