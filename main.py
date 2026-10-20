"""Main pipeline orchestrator for Micro-Credit Pulse Engine.

Executes the full file-based chain:
1. Load & validate raw data (M1)
2. Governance audit (M2)
3. Feature synthesis by cohort (M3)
4. Model training + SHAP (M4)
5. Model evaluation with plots
6. GenAI loan officer advisory (M5)

Usage:
    python main.py [--skip-governance] [--skip-train] [--run-genai] [--random-state SEED]
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_pipeline(
    skip_governance: bool = False,
    skip_train: bool = False,
    run_genai: bool = False,
    random_state: int = 42
) -> None:
    """Execute the full M1-M5 pipeline."""
    print("=" * 60)
    print("Micro-Credit Pulse Engine - Full Pipeline")
    print("=" * 60)

    # M1: Data loading & validation
    print("\n[M1] Loading raw data...")
    from data_pipeline.loader import load_reference_dataset, get_cohort_split

    raw_df = load_reference_dataset()
    cohort_split = get_cohort_split(raw_df)
    print(f"[OK] Loaded {len(raw_df)} records")
    print(f"  Cohort split: {cohort_split}")

    # M2: Governance audit
    if not skip_governance:
        print("\n[M2] Running governance audit...")
        from data_pipeline.governance import validate

        audit = validate()
        failed = [k for k, v in audit["checks"].items() if v == "Fail"]
        if failed:
            print(f"[WARN] Audit completed with failures: {failed}")
        else:
            print(f"[OK] Governance audit passed")
        print(f"  Audit log: data/logs/governance_audit.json")
    else:
        print("\n[M2] Governance audit skipped (--skip-governance)")

    # M3: Feature synthesis (cohort-routed)
    print("\n[M3] Synthesizing features by cohort...")
    from features import synthesize_all_features

    engineered_df = synthesize_all_features(
        input_path="data/raw/alternative_credit_scoring.csv",
        output_path="data/processed/engineered_features.parquet",
        random_state=random_state
    )
    print(f"[OK] Engineered {engineered_df.shape[1]} features ({len(engineered_df)} rows)")
    print(f"  Output: data/processed/engineered_features.parquet")

    # M4: Model training + SHAP
    if not skip_train:
        print("\n[M4] Training XGBoost model + SHAP...")
        from models.train import train_and_explain

        predictions_df, model = train_and_explain()
        print(f"[OK] Model trained on test set ({len(predictions_df)} test samples)")
        print(f"  5-fold CV accuracy: {predictions_df['cv_accuracy_mean'].iloc[0]:.4f} ± {predictions_df['cv_accuracy_std'].iloc[0]:.4f}")
        print(f"  5-fold CV ROC AUC: {predictions_df['cv_roc_auc_mean'].iloc[0]:.4f} ± {predictions_df['cv_roc_auc_std'].iloc[0]:.4f}")
        print(f"  Test accuracy: {(predictions_df['prediction'] == predictions_df['loan_approved']).mean():.4f}")
        print(f"  Predictions: data/processed/model_predictions_with_shap.csv")

        # Model evaluation with plots
        print("\n[M4+] Evaluating model performance...")
        from models.evaluations import evaluate_model

        eval_summary = evaluate_model(plot=True)
        print(f"[OK] Evaluation complete")
        print(f"  AUC: {eval_summary['roc_auc']:.4f}")
        print(f"  Confusion Matrix: {eval_summary['confusion_matrix']}")
        print(f"  Outputs: data/processed/model_evaluations.json")
        print(f"           data/processed/model_evaluation_plots.png")
    else:
        print("\n[M4] Model training skipped (--skip-train)")

    # M5: GenAI loan officer advisory
    if run_genai:
        print("\n[M5] Generating loan officer advisory summaries...")
        from genai import generate_portfolio_report

        output_path = generate_portfolio_report()
        print(f"[OK] Advisory report saved: {output_path}")
    else:
        print("\n[M5] GenAI advisory skipped (use --run-genai to enable)")

    # Summary
    print("\n" + "=" * 60)
    print("[OK] Pipeline complete!")
    print("=" * 60)
    print("\nArtifacts generated:")
    print("  [OK] data/logs/governance_audit.json (M2 audit)")
    print("  [OK] data/processed/engineered_features.parquet (M3 features)")
    if not skip_train:
        print("  [OK] data/processed/model_predictions_with_shap.csv (M4 predictions)")
        print("  [OK] data/processed/model_evaluations.json (M4 metrics)")
        print("  [OK] data/processed/model_evaluation_plots.png (M4 plots)")
    if run_genai:
        print("  [OK] data/processed/final_loan_portfolio_report.csv (M5 report)")

    print("\nNext steps:")
    print("  - Review audit: cat data/logs/governance_audit.json")
    print("  - View plots: open data/processed/model_evaluation_plots.png")
    print("  - Run tests: pytest tests/ -v")
    if run_genai:
        print("  - View advisory report: cat data/processed/final_loan_portfolio_report.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Micro-Credit Pulse Engine - Full Pipeline Orchestrator"
    )
    parser.add_argument(
        "--skip-governance",
        action="store_true",
        help="Skip M2 governance audit step"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip M4 model training and evaluation"
    )
    parser.add_argument(
        "--run-genai",
        action="store_true",
        help="Run M5 GenAI loan officer advisory (requires GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            skip_governance=args.skip_governance,
            skip_train=args.skip_train,
            run_genai=args.run_genai,
            random_state=args.random_state
        )
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()