"""
Challenge Tier 1: Threshold Tuning
===================================
The default classification threshold is 0.5. For churn prediction,
missing a churner (false negative) is more costly than a false alarm.
This script explores how precision, recall, and F1 change across thresholds.

Run: python challenge_tier1.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Load & prepare data ──────────────────────────────────────────────────────

df = pd.read_csv("data/telecom_churn.csv")

numeric_features = [
    "tenure", "monthly_charges", "total_charges",
    "num_support_calls", "senior_citizen",
    "has_partner", "has_dependents"
]

df_cls = df[numeric_features + ["churned"]].dropna()

X = df_cls.drop(columns=["churned"])
y = df_cls["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Build & fit pipeline ─────────────────────────────────────────────────────

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        random_state=42, max_iter=1000, class_weight="balanced"
    ))
])

pipe.fit(X_train, y_train)

# ── Get probability scores ───────────────────────────────────────────────────

y_proba = pipe.predict_proba(X_test)[:, 1]  # probability of churn (class 1)

# ── Evaluate at multiple thresholds ─────────────────────────────────────────

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

results = []
print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 44)

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    p  = precision_score(y_test, y_pred, zero_division=0)
    r  = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
    print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")

# ── Find best threshold by F1 ────────────────────────────────────────────────

results_df   = pd.DataFrame(results)
best_row     = results_df.loc[results_df["f1"].idxmax()]
best_thresh  = best_row["threshold"]

print(f"\n✓ Best threshold by F1: {best_thresh:.2f}")
print(f"  Precision: {best_row['precision']:.3f}")
print(f"  Recall:    {best_row['recall']:.3f}")
print(f"  F1:        {best_row['f1']:.3f}")

print(f"""
Why does the best threshold differ from 0.5?
---------------------------------------------
With class_weight="balanced", the model already leans toward predicting churn
more aggressively. On an imbalanced dataset (~16% churn), the predicted
probabilities for the minority class cluster below 0.5, so the default 0.5
cutoff under-predicts churn. Lowering the threshold shifts the boundary to
where the probability distribution of churners actually lies, recovering
recall without collapsing precision entirely. The F1-maximizing threshold
balances this trade-off optimally for this dataset.
""")

# ── Plot precision, recall, F1 vs threshold ──────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(results_df["threshold"], results_df["precision"],
        marker="o", label="Precision", color="#2196F3")
ax.plot(results_df["threshold"], results_df["recall"],
        marker="o", label="Recall",    color="#F44336")
ax.plot(results_df["threshold"], results_df["f1"],
        marker="o", label="F1",        color="#4CAF50", linewidth=2)

ax.axvline(x=best_thresh, color="gray", linestyle="--",
           label=f"Best threshold ({best_thresh:.2f})")

ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision, Recall & F1 vs Classification Threshold\n(Churn Prediction)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("threshold_tuning.png", dpi=150)
plt.show()
print("Plot saved to threshold_tuning.png")