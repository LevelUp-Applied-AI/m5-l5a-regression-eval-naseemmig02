"""
Challenge Tier 2: Config-Driven Model Sweep
============================================
Reads a YAML config file, instantiates each model inside a Pipeline,
runs cross-validation, and prints a formatted results table.

Run: python challenge_tier2.py
Requires: pip install pyyaml
"""

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Model registry ───────────────────────────────────────────────────────────
# Maps type strings from the config to sklearn classes.
# Add any new model class here to support it in the config.

MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "Ridge":              Ridge,
    "Lasso":              Lasso,
}

# ── Classifier vs Regressor detection ────────────────────────────────────────

CLASSIFIERS = {"LogisticRegression"}
REGRESSORS  = {"Ridge", "Lasso"}


def load_config(config_path="model_config.yaml"):
    """Load model configurations from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_pipeline_from_config(model_cfg):
    """Instantiate a sklearn Pipeline from a single model config dict.

    Args:
        model_cfg: dict with keys 'type' and 'params'.

    Returns:
        sklearn Pipeline with StandardScaler + model.
    """
    model_type   = model_cfg["type"]
    model_params = model_cfg.get("params", {})

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_type](**model_params)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  model)
    ])


def run_model_sweep(config_path="model_config.yaml",
                    X_cls=None, y_cls=None,
                    X_reg=None, y_reg=None,
                    cv_folds=5):
    """
    Read config, run cross-validation for each model, return results table.

    Classifiers are evaluated on X_cls / y_cls with StratifiedKFold + accuracy.
    Regressors  are evaluated on X_reg / y_reg with KFold + r2.

    Args:
        config_path : path to YAML config file.
        X_cls, y_cls: classification features and labels.
        X_reg, y_reg: regression features and target.
        cv_folds    : number of CV folds.

    Returns:
        DataFrame with columns: Model, Type, Metric, Mean, Std, Min, Max.
    """
    config = load_config(config_path)
    rows   = []

    cls_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    reg_cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for model_cfg in config["models"]:
        name       = model_cfg["name"]
        model_type = model_cfg["type"]

        try:
            pipeline = build_pipeline_from_config(model_cfg)
        except ValueError as e:
            print(f"  [SKIP] {name}: {e}")
            continue

        # ── Choose data and scorer based on model type ────────────────────
        if model_type in CLASSIFIERS:
            if X_cls is None:
                print(f"  [SKIP] {name}: no classification data provided.")
                continue
            X, y     = X_cls, y_cls
            cv       = cls_cv
            scoring  = "accuracy"
            metric   = "Accuracy"

        elif model_type in REGRESSORS:
            if X_reg is None:
                print(f"  [SKIP] {name}: no regression data provided.")
                continue
            X, y     = X_reg, y_reg
            cv       = reg_cv
            scoring  = "r2"
            metric   = "R²"

        else:
            print(f"  [SKIP] {name}: type '{model_type}' not mapped to classifier/regressor.")
            continue

        # ── Run CV ────────────────────────────────────────────────────────
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

        rows.append({
            "Model":  name,
            "Type":   model_type,
            "Metric": metric,
            "Mean":   round(scores.mean(), 4),
            "Std":    round(scores.std(),  4),
            "Min":    round(scores.min(),  4),
            "Max":    round(scores.max(),  4),
        })

    return pd.DataFrame(rows)


def print_results_table(df):
    """Print a clean formatted results table sorted by Mean score."""
    if df.empty:
        print("No results to display.")
        return

    # Print classifiers and regressors separately
    for model_type, group in df.groupby("Type"):
        group_sorted = group.sort_values("Mean", ascending=False).reset_index(drop=True)
        print(f"\n{'=' * 72}")
        print(f"  {model_type} Results")
        print(f"{'=' * 72}")
        print(f"{'#':<4} {'Model':<40} {'Metric':<10} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
        print("-" * 72)
        for i, row in group_sorted.iterrows():
            print(f"{i+1:<4} {row['Model']:<40} {row['Metric']:<10} "
                  f"{row['Mean']:>7.4f} {row['Std']:>7.4f} "
                  f"{row['Min']:>7.4f} {row['Max']:>7.4f}")

    # Overall best per category
    print(f"\n{'=' * 72}")
    print("  Best Model per Category")
    print(f"{'=' * 72}")
    for model_type, group in df.groupby("Type"):
        best = group.loc[group["Mean"].idxmax()]
        print(f"  {model_type:<22} → {best['Model']} "
              f"({best['Metric']} {best['Mean']:.4f} ± {best['Std']:.4f})")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("data/telecom_churn.csv")

    numeric_features = [
        "tenure", "monthly_charges", "total_charges",
        "num_support_calls", "senior_citizen",
        "has_partner", "has_dependents"
    ]
    reg_features = [
        "tenure", "total_charges", "num_support_calls",
        "senior_citizen", "has_partner", "has_dependents"
    ]

    # Classification data
    df_cls = df[numeric_features + ["churned"]].dropna()
    X_cls  = df_cls.drop(columns=["churned"])
    y_cls  = df_cls["churned"]

    # Regression data
    df_reg = df[reg_features + ["monthly_charges"]].dropna()
    X_reg  = df_reg.drop(columns=["monthly_charges"])
    y_reg  = df_reg["monthly_charges"]

    print("Running config-driven model sweep...")
    print(f"Classification: {X_cls.shape[0]} rows | Regression: {X_reg.shape[0]} rows\n")

    results = run_model_sweep(
        config_path="model_config.yaml",
        X_cls=X_cls, y_cls=y_cls,
        X_reg=X_reg, y_reg=y_reg,
        cv_folds=5
    )

    print_results_table(results)

    # Save to CSV
    results.to_csv("model_sweep_results.csv", index=False)
    print("\nFull results saved to model_sweep_results.csv")