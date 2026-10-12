"""Model evaluation for XGBoost classifier trained in train.py.
Outputs confusion matrix, classification report, AUC, ROC curve (data + plots)."""
from __future__ import annotations
import pandas as pd, numpy as np, json, os, matplotlib.pyplot as plt, io
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, PrecisionRecallDisplay, ConfusionMatrixDisplay

def evaluate_model(pred_path: str = "data/processed/model_predictions_with_shap.csv", plot: bool = True) -> dict:
    df = pd.read_csv(pred_path)
    y_true = df["loan_approved"]
    y_pred = df["prediction"]
    y_prob = df["prob_approved"]
    # Ensure binary
    if len(np.unique(y_true)) < 2:
        raise ValueError("Only one class present in true labels")
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()
    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    # ROC curve (fpr, tpr, thresholds)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": roc_auc
    }
    # Classification report (dict)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    summary = {
        "confusion_matrix": cm_list,
        "roc_auc": roc_auc,
        "roc_curve": roc_data,
        "classification_report": report
    }
    # Save summary
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_evaluations.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate plots
    if plot:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion Matrix heatmap
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected (0)", "Approved (1)"])
        disp.plot(ax=axes[0], cmap='Blues', values_format='d')
        axes[0].set_title(f'Confusion Matrix\n(AUC: {roc_auc:.4f})', fontsize=12, weight='bold')

        # ROC Curve
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate', fontsize=10)
        axes[1].set_ylabel('True Positive Rate', fontsize=10)
        axes[1].set_title('ROC Curve', fontsize=12, weight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "model_evaluation_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plots saved: {plot_path}")

    return summary

if __name__ == "__main__":
    print("Evaluating model predictions...")
    try:
        summary = evaluate_model()
        print(f"Model AUC: {summary['roc_auc']:.4f}")
        print("Confusion matrix:")
        for row in summary["confusion_matrix"]:
            print(row)
        print("Classification report:")
        for label, metrics in summary["classification_report"].items():
            if label not in ("accuracy", "macro avg", "weighted avg"):
                print(f"{label}: {metrics}")
        print("\nSaved evaluations to data/processed/model_evaluations.json")
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise