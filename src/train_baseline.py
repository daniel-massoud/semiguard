# Trains a baseline pass/fail classifier on the SECOM training set.
# We use a Random Forest because it handles the high feature count (400+ sensors)
# without much tuning, gives us calibrated-ish probabilities, and is robust to
# the kind of noisy redundant features that semiconductor fabs produce.
#
# The model is evaluated on the held-out validation set with metrics that
# actually matter for a rare-failure setting: precision, recall, PR-AUC, and
# the precision we get at a fixed review-capacity budget.

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Paths are relative to this script in src/
here = os.path.dirname(os.path.abspath(__file__))
processed = os.path.join(here, "..", "data", "processed")
models_dir = os.path.join(here, "..", "results", "models")
os.makedirs(models_dir, exist_ok=True)

# --- Load the cleaned splits we wrote in Block 2.
train = pd.read_csv(os.path.join(processed, "train.csv"))
val   = pd.read_csv(os.path.join(processed, "val.csv"))

# Everything that starts with "sensor_" is a feature. The other two columns
# are timestamp and label, which we keep separate.
feature_cols = [c for c in train.columns if c.startswith("sensor_")]

X_train, y_train = train[feature_cols].values, train["label"].values
X_val,   y_val   = val[feature_cols].values,   val["label"].values

print(f"train: {X_train.shape}, failures: {y_train.sum()} ({y_train.mean() * 100:.1f}%)")
print(f"val:   {X_val.shape}, failures: {y_val.sum()} ({y_val.mean() * 100:.1f}%)")

# --- Train the model.
# class_weight="balanced" tells the forest to upweight the rare failure class
# so it doesnt just learn "always predict pass". This is the simplest honest
# way to deal with imbalance without resampling tricks.
print("\ntraining random forest...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)

# --- Predict probabilities, not classes. We need the score, not a yes/no,
# so we can pick a threshold that matches our review budget later.
val_probs = model.predict_proba(X_val)[:, 1]

# --- Compute the metrics that actually matter.
pr_auc = average_precision_score(y_val, val_probs)
roc    = roc_auc_score(y_val, val_probs)

# At the default 0.5 threshold, how does it look?
val_preds_default = (val_probs >= 0.5).astype(int)
print("\n--- at default threshold 0.5 ---")
print(classification_report(y_val, val_preds_default, target_names=["pass", "fail"], digits=3))
print("confusion matrix [[TN FP][FN TP]]:")
print(confusion_matrix(y_val, val_preds_default))

# The interesting question: if a human can only review the top K% of wafers
# flagged as risky per day, what fraction of those flagged are actually fails?
# That is "precision at K", and it's the most production-relevant metric here.
print(f"\nPR-AUC: {pr_auc:.3f}")
print(f"ROC-AUC: {roc:.3f}")

print("\n--- precision at fixed review capacity ---")
for review_fraction in [0.05, 0.10, 0.20]:
    # Take the top-scored wafers, where "top" means we have budget for
    # review_fraction of all wafers seen.
    k = int(len(val_probs) * review_fraction)
    top_k_indices = np.argsort(val_probs)[-k:]
    precision_at_k = y_val[top_k_indices].mean()
    recall_at_k    = y_val[top_k_indices].sum() / y_val.sum()
    print(f"  review top {int(review_fraction * 100):2d}%  ->  "
          f"precision = {precision_at_k:.2f}, recall = {recall_at_k:.2f}")

# --- Save the model and the features it expects.
joblib.dump(model, os.path.join(models_dir, "baseline_model.pkl"))
with open(os.path.join(models_dir, "baseline_features.json"), "w") as f:
    json.dump(feature_cols, f)

# --- Also save the val-set probabilities and labels. We'll need these in
# Block 9 to calibrate the operating threshold against the review budget.
np.savez(
    os.path.join(models_dir, "baseline_val_scores.npz"),
    probs=val_probs,
    labels=y_val,
)

print(f"\nsaved model and artifacts to {models_dir}")
