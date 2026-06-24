# Computes three OOD scores for every wafer in the validation set, based on
# the baseline Random Forest's behavior rather than the input distribution.
#
# These are model-aware scores: they catch cases where the model is uncertain,
# which is fundamentally different from cases where the input looks anomalous.
# A wafer can be perfectly normal-looking but still trigger high uncertainty
# if it lands near the decision boundary, and we want to flag those too.

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

here = os.path.dirname(os.path.abspath(__file__))
processed = os.path.join(here, "..", "data", "processed")
models_dir = os.path.join(here, "..", "results", "models")

# --- Load baseline model and val data
baseline_model = joblib.load(os.path.join(models_dir, "baseline_model.pkl"))
val = pd.read_csv(os.path.join(processed, "val.csv"))
sensor_cols = [c for c in val.columns if c.startswith("sensor_")]
X_val = val[sensor_cols].values
y_val = val["label"].values

# --- 1) Inverse max class probability (MSP analog).
# predict_proba gives [P(pass), P(fail)] per wafer. The closer the max is to
# 0.5, the more confused the model is.  We score 1 - max(p) so higher means
# more uncertain. This is the classical OOD baseline from Hendrycks & Gimpel.
probs = baseline_model.predict_proba(X_val)
max_class_prob = probs.max(axis=1)
inverse_msp = 1.0 - max_class_prob

# --- 2) Prediction entropy.
# Higher entropy = more uniform distribution = more uncertain.
# Add a tiny epsilon to avoid log(0) — harmless but defensive.
eps = 1e-12
entropy = -np.sum(probs * np.log(probs + eps), axis=1)

# --- 3) Tree-level disagreement (epistemic uncertainty for forests).
# Each of the 300 trees in the forest votes; the final prediction is the
# average. But the VARIANCE of those votes is itself a signal: when trees
# agree the ensemble is confident, when they disagree the ensemble is in
# territory it doesnt know well. This is invisible in the averaged output.
per_tree_fail_probs = np.array([
    tree.predict_proba(X_val)[:, 1] for tree in baseline_model.estimators_
])
# shape: (n_trees, n_val). We want the spread across trees for each wafer.
tree_disagreement = per_tree_fail_probs.std(axis=0)

# --- Print PR-AUC for each OOD score against actual failures. Worth
# remembering: these scores arent supposed to predict failures directly,
# theyre supposed to predict UNCERTAINTY. Some signal vs failures means
# uncertain predictions are also more likely to be wrong — which is the
# whole reason this is useful for production review.
print("OOD scores vs actual failures (PR-AUC):")
for name, score in [
    ("inverse_msp", inverse_msp),
    ("entropy", entropy),
    ("tree_disagreement", tree_disagreement),
]:
    ap = average_precision_score(y_val, score)
    print(f"  {name:20s}  {ap:.3f}")

# --- Append to the val scores CSV that Blocks 4 and 5 built up
val_scores_path = os.path.join(processed, "val_anomaly_scores.csv")
val_scores = pd.read_csv(val_scores_path)
val_scores["inverse_msp"]        = inverse_msp
val_scores["entropy"]            = entropy
val_scores["tree_disagreement"]  = tree_disagreement
val_scores.to_csv(val_scores_path, index=False)

print(f"\nappended OOD scores to {val_scores_path}")
print(f"columns now: {list(val_scores.columns)}")
