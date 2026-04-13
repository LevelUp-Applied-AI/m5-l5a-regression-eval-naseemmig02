"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score
)


# ---------------------------------------------------------------------------
# Task 1: Load Data
# ---------------------------------------------------------------------------

def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['churned'].value_counts(normalize=True).round(3)}")
    return df


# ---------------------------------------------------------------------------
# Task 2: Split Data
# ---------------------------------------------------------------------------

def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification where applicable.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratify only for discrete/classification targets
    stratify = y if y.nunique() <= 10 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    if stratify is not None:
        print(f"Train churn rate: {y_train.mean():.3f} | Test churn rate: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Task 3: Logistic Regression Pipeline
# ---------------------------------------------------------------------------

def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    disp.plot()
    plt.title("Confusion Matrix — Logistic Regression")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }


# ---------------------------------------------------------------------------
# Task 4: Ridge Regression Pipeline
# ---------------------------------------------------------------------------

def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\n--- Ridge Regression Metrics ---")
    print(f"MAE: {mae:.4f} | R²: {r2:.4f}")

    return {"mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Task 5: Lasso Regularization Comparison
# ---------------------------------------------------------------------------

def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression.

    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])


def compare_ridge_lasso(ridge_pipe, lasso_pipe, X_train, y_train):
    """Fit both pipelines and print coefficients side by side.

    Args:
        ridge_pipe: Fitted or unfitted Ridge pipeline.
        lasso_pipe: Fitted or unfitted Lasso pipeline.
        X_train: Training features.
        y_train: Training target.
    """
    ridge_pipe.fit(X_train, y_train)
    lasso_pipe.fit(X_train, y_train)

    feature_names = X_train.columns.tolist()
    ridge_coefs   = ridge_pipe.named_steps["model"].coef_
    lasso_coefs   = lasso_pipe.named_steps["model"].coef_

    print(f"\n--- Ridge vs Lasso Coefficients ---")
    print(f"{'Feature':<25} {'Ridge':>10} {'Lasso':>10}")
    print("-" * 47)
    for name, rc, lc in zip(feature_names, ridge_coefs, lasso_coefs):
        zeroed = " <- zeroed" if lc == 0.0 else ""
        print(f"{name:<25} {rc:>10.4f} {lc:>10.4f}{zeroed}")

    zeroed_features = [n for n, lc in zip(feature_names, lasso_coefs) if lc == 0.0]
    if zeroed_features:
        print(f"\nLasso zeroed out: {zeroed_features}")
        # These features have weak marginal predictive power for monthly_charges
        # once correlated features (e.g. total_charges, tenure) are already in the model.
    else:
        print("\nNo features fully zeroed by Lasso at alpha=0.1")


# ---------------------------------------------------------------------------
# Task 6: Cross-Validation
# ---------------------------------------------------------------------------

def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_splitter, scoring="accuracy"
    )

    print(f"\n--- {cv}-Fold Stratified Cross-Validation ---")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean: {scores.mean():.3f} +/- {scores.std():.3f}")

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Task 1: Load ────────────────────────────────────────────────────────
    df = load_data()

    if df is not None:
        print(f"\nLoaded {len(df)} rows, {df.shape[1]} columns")

        numeric_features = [
            "tenure", "monthly_charges", "total_charges",
            "num_support_calls", "senior_citizen",
            "has_partner", "has_dependents"
        ]

        # ── Task 2 & 3: Classification — predict churned ────────────────────
        print("\n" + "=" * 55)
        print("CLASSIFICATION: Predicting Churn")
        print("=" * 55)

        df_cls = df[numeric_features + ["churned"]].dropna()
        split  = split_data(df_cls, "churned")

        if split:
            X_train, X_test, y_train, y_test = split

            pipe    = build_logistic_pipeline()
            metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
            print(f"\nLogistic Regression metrics: {metrics}")

            # ── Task 6: Cross-Validation ─────────────────────────────────────
            scores = run_cross_validation(pipe, X_train, y_train)
            print(f"\nCV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

        # ── Task 4 & 5: Regression — predict monthly_charges ────────────────
        print("\n" + "=" * 55)
        print("REGRESSION: Predicting Monthly Charges")
        print("=" * 55)

        reg_features = [
            "tenure", "total_charges", "num_support_calls",
            "senior_citizen", "has_partner", "has_dependents"
        ]

        df_reg    = df[reg_features + ["monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")

        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg

            ridge_pipe  = build_ridge_pipeline()
            reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
            print(f"\nRidge Regression metrics: {reg_metrics}")

            # ── Task 5: Lasso comparison ─────────────────────────────────────
            lasso_pipe = build_lasso_pipeline()
            compare_ridge_lasso(ridge_pipe, lasso_pipe, X_tr, y_tr)


# ---------------------------------------------------------------------------
# Task 7: Summary of Findings
# ---------------------------------------------------------------------------
"""
=== Task 7: Summary of Findings ===

1. IMPORTANT FEATURES FOR CHURN:
   - tenure: Long-tenured customers are far less likely to churn. It's the
     strongest single predictor.
   - num_support_calls: A high-signal behavioral feature — customers who call
     support repeatedly are frustrated and likely to leave.
   - total_charges: Correlated with tenure but adds marginal signal about
     overall spend commitment.
   - has_partner / has_dependents: Customers with family ties churn less,
     likely because switching costs are higher for households.

2. MODEL PERFORMANCE:
   With class_weight="balanced" the model compensates for the ~16% churn
   minority class by upweighting it during training. This raises recall at
   the cost of precision — the model catches more real churners but also
   produces more false alarms (predicting churn for loyal customers).
   For this business problem, RECALL is the more important metric: a missed
   churner (false negative) means a lost customer with no chance to intervene,
   whereas a false positive only means an unnecessary (but cheap) retention
   offer. Erring toward higher recall is the right trade-off here.

3. NEXT STEPS TO IMPROVE PERFORMANCE:
   - Add a DummyClassifier baseline (Thursday's Integration Task) to confirm
     the model genuinely beats naive majority-class prediction.
   - Engineer richer features: contract type (month-to-month vs annual),
     payment method, and internet service tier are strong churn signals in
     telecom literature.
   - Try tree-based models (RandomForest, GradientBoosting) which capture
     non-linear interactions (e.g. high monthly_charges AND short tenure)
     that logistic regression cannot model without manual feature engineering.
   - Tune the classification threshold (Challenge Tier 1): lowering the
     threshold below 0.5 further increases recall and may improve F1 on this
     imbalanced dataset.
"""