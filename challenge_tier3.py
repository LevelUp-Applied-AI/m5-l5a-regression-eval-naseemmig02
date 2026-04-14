"""
Challenge Tier 3: Logistic Regression from Scratch
====================================================
Binary logistic regression implemented using only NumPy.
Includes sigmoid, log-loss, gradient descent with L2 regularization,
and a fit/predict interface — then compared against sklearn.

Run: python challenge_tier3.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =============================================================================
# Custom Logistic Regression (NumPy only)
# =============================================================================

class LogisticRegressionNumpy:
    """
    Binary logistic regression via gradient descent with L2 regularization.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates.
    n_iterations : int
        Number of gradient descent steps.
    lambda_l2 : float
        L2 regularization strength (0 = no regularization).
    verbose : bool
        If True, print loss every 100 iterations.
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000,
                 lambda_l2=0.01, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.lambda_l2     = lambda_l2
        self.verbose       = verbose
        self.weights       = None
        self.bias          = None
        self.loss_history  = []

    # ── Core math ─────────────────────────────────────────────────────────

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation: maps any real value to (0, 1).
        σ(z) = 1 / (1 + e^(-z))
        Clipped for numerical stability to avoid log(0).
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def log_loss(self, y_true, y_pred_proba):
        """
        Binary cross-entropy loss with L2 penalty.

        L = -1/n * Σ [y·log(ŷ) + (1-y)·log(1-ŷ)] + (λ/2n) * Σ w²

        The L2 term penalizes large weights to prevent overfitting.
        Bias is excluded from regularization (standard practice).
        """
        n = len(y_true)
        # Clip probabilities to avoid log(0)
        y_pred_proba = np.clip(y_pred_proba, 1e-9, 1 - 1e-9)

        cross_entropy = -np.mean(
            y_true * np.log(y_pred_proba) +
            (1 - y_true) * np.log(1 - y_pred_proba)
        )
        l2_penalty = (self.lambda_l2 / (2 * n)) * np.sum(self.weights ** 2)

        return cross_entropy + l2_penalty

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Fit the model using gradient descent.

        Gradient of log-loss w.r.t. weights:
            dL/dw = (1/n) * Xᵀ(ŷ - y) + (λ/n) * w    ← L2 term
            dL/db = (1/n) * Σ(ŷ - y)                   ← bias, no L2

        Args:
            X : np.ndarray of shape (n_samples, n_features)
            y : np.ndarray of shape (n_samples,), binary {0, 1}
        """
        n_samples, n_features = X.shape

        # Initialize weights to zero (same as sklearn default)
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

        for i in range(self.n_iterations):
            # Forward pass
            z            = X @ self.weights + self.bias
            y_pred_proba = self.sigmoid(z)

            # Compute gradients
            error     = y_pred_proba - y
            dw        = (X.T @ error) / n_samples + (self.lambda_l2 / n_samples) * self.weights
            db        = np.mean(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # Track loss
            loss = self.log_loss(y, y_pred_proba)
            self.loss_history.append(loss)

            if self.verbose and (i + 1) % 100 == 0:
                print(f"  Iteration {i+1:>5} | Loss: {loss:.6f}")

        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """Return probability of class 1 for each sample."""
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Return binary predictions using the given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)


# =============================================================================
# Comparison with sklearn
# =============================================================================

def compare_with_sklearn(X_train, X_test, y_train, y_test):
    """
    Train both implementations and compare predictions and coefficients.
    """

    # ── Our implementation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training: NumPy Logistic Regression")
    print("=" * 60)

    custom_model = LogisticRegressionNumpy(
        learning_rate=0.1,
        n_iterations=2000,
        lambda_l2=0.01,
        verbose=True
    )
    custom_model.fit(X_train, y_train)

    custom_preds  = custom_model.predict(X_test)
    custom_acc    = accuracy_score(y_test, custom_preds)

    print(f"\nFinal loss: {custom_model.loss_history[-1]:.6f}")
    print(f"Accuracy  : {custom_acc:.4f}")
    print(f"\nClassification Report (NumPy):")
    print(classification_report(y_test, custom_preds))

    # ── sklearn implementation ────────────────────────────────────────────
    print("=" * 60)
    print("Training: sklearn LogisticRegression")
    print("=" * 60)

    sklearn_model = LogisticRegression(
        random_state=42,
        max_iter=2000,
        C=1.0 / 0.01,          # sklearn uses C = 1/lambda
        solver="lbfgs",
        class_weight=None       # match our implementation (no balancing)
    )
    sklearn_model.fit(X_train, y_train)

    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_acc   = accuracy_score(y_test, sklearn_preds)

    print(f"Accuracy  : {sklearn_acc:.4f}")
    print(f"\nClassification Report (sklearn):")
    print(classification_report(y_test, sklearn_preds))

    # ── Coefficient comparison ────────────────────────────────────────────
    print("=" * 60)
    print("Coefficient Comparison")
    print("=" * 60)

    feature_names  = [f"feature_{i}" for i in range(X_train.shape[1])]
    sklearn_coefs  = sklearn_model.coef_[0]
    custom_coefs   = custom_model.weights

    print(f"\n{'Feature':<15} {'NumPy':>12} {'sklearn':>12} {'Diff':>12}")
    print("-" * 53)
    for name, cw, sw in zip(feature_names, custom_coefs, sklearn_coefs):
        diff = abs(cw - sw)
        print(f"{name:<15} {cw:>12.6f} {sw:>12.6f} {diff:>12.6f}")

    print(f"\n{'bias/intercept':<15} {custom_model.bias:>12.6f} "
          f"{sklearn_model.intercept_[0]:>12.6f} "
          f"{abs(custom_model.bias - sklearn_model.intercept_[0]):>12.6f}")

    # ── Prediction agreement ──────────────────────────────────────────────
    agreement = np.mean(custom_preds == sklearn_preds)
    print(f"\nPrediction agreement: {agreement:.2%} of test samples match")

    # ── Loss curve ────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(custom_model.loss_history, color="#2196F3", linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("Log-Loss")
        plt.title("Gradient Descent Convergence — NumPy Logistic Regression")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("convergence_curve.png", dpi=150)
        plt.show()
        print("\nConvergence curve saved to convergence_curve.png")
    except ImportError:
        print("matplotlib not available — skipping convergence plot.")

    # ── Divergence analysis ───────────────────────────────────────────────
    print("""
Why results may diverge from sklearn
--------------------------------------
1. SOLVER: sklearn's lbfgs uses second-order (quasi-Newton) optimization
   which converges faster and more precisely than our first-order gradient
   descent. Our weights may not fully converge in 2000 iterations.

2. REGULARIZATION CONVENTION: sklearn uses C = 1/lambda where larger C means
   LESS regularization. We use lambda directly where larger = MORE.
   C=100 here matches lambda=0.01 in our model.

3. CLASS WEIGHTING: sklearn's class_weight="balanced" reweights the gradient
   contribution of each sample. Our implementation treats all samples equally,
   so predictions on the minority class (churners) will differ the most.

4. CONVERGENCE CRITERIA: sklearn's lbfgs stops when the gradient norm falls
   below a tolerance (default 1e-4). Our implementation always runs the full
   n_iterations regardless of convergence — we may over- or under-shoot.

5. INTERCEPT SCALING: sklearn internally applies a small scaling to the
   intercept term during optimization to improve conditioning. We treat bias
   identically to weights (except excluding it from L2).
""")

    return custom_model, sklearn_model


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # Load and prepare data
    df = pd.read_csv("data/telecom_churn.csv")

    numeric_features = [
        "tenure", "monthly_charges", "total_charges",
        "num_support_calls", "senior_citizen",
        "has_partner", "has_dependents"
    ]

    df_cls = df[numeric_features + ["churned"]].dropna()
    X      = df_cls.drop(columns=["churned"]).values.astype(float)
    y      = df_cls["churned"].values.astype(float)

    # Scale features (our model has no built-in scaler unlike the Pipeline)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset: {X_train.shape[0]} train | {X_test.shape[0]} test samples")
    print(f"Churn rate: {y_train.mean():.3f} train | {y_test.mean():.3f} test")

    compare_with_sklearn(X_train, X_test, y_train, y_test)