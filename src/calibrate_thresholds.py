# Calibrates operating thresholds for production review. The core idea:
# a review team can only inspect N wafers per shift, so we set the
# detection threshold to flag exactly that fraction of incoming wafers,
# then measure how many failures we expect to actually catch.
#
# Three scoring strategies are compared:
#   1. Baseline probability alone (what a vanilla ML deployment would use)
#   2. Anomaly ensemble alone (input-based unsupervised signal)
#   3. Combined: rank-average of baseline + anomaly ensemble
#
# All confidence intervals come from bootstrap resampling so the report
# can quote real uncertainty intervals instead of point estimates.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from sklearn.metrics import precision_score, recall_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

here = os.path.dirname(os.path.abspath(__file__))
processed = os.path.join(here, "..", "data", "processed")
models_dir = os.path.join(here, "..", "results", "models")
figures = os.path.join(here, "..", "results", "figures")
os.makedirs(figures, exist_ok=True)

# --- Load all the val-set scores we accumulated through Blocks 3-6
scores = pd.read_csv(os.path.join(processed, "val_anomaly_scores.csv"))

# Build the three candidate scoring strategies
# 1. Baseline alone (already in the file)
baseline = scores["baseline_prob"].values

# 2. Anomaly ensemble: rank-average of the 5 input-anomaly detectors
input_detectors = ["mahalanobis", "isolation_forest", "lof", "ocsvm", "autoencoder"]
ranks_anomaly = np.column_stack([rankdata(scores[d]) for d in input_detectors])
anomaly_ensemble = ranks_anomaly.mean(axis=1)

# 3. Combined: rank-average of baseline + anomaly ensemble.
# We give them equal weight here. In production this weight could itself
# be tuned, but equal-weight is a defensible neutral starting point.
ranks_combined = np.column_stack([
    rankdata(baseline),
    rankdata(anomaly_ensemble),
])
combined = ranks_combined.mean(axis=1)

y_val = scores["is_failure"].values
n_val = len(scores)
n_failures = int(y_val.sum())

scoring_strategies = {
    "baseline_only":    baseline,
    "anomaly_only":     anomaly_ensemble,
    "baseline_and_anomaly": combined,
}


def evaluate_at_budget(score, labels, review_fraction):
    """At the given review budget, what threshold do we use, and what does
    it catch? Returns precision (fraction of reviews that are real fails)
    and recall (fraction of all fails captured)."""
    n = len(score)
    n_to_review = max(1, int(round(n * review_fraction)))
    threshold = np.partition(score, -n_to_review)[-n_to_review]

    flagged = score >= threshold
    # Clip to exactly n_to_review in case of ties at the threshold
    if flagged.sum() > n_to_review:
        # Tie-break by descending score order
        sorted_idx = np.argsort(-score)[:n_to_review]
        flagged = np.zeros(n, dtype=bool)
        flagged[sorted_idx] = True

    tp = int(((labels == 1) & flagged).sum())
    fp = int(((labels == 0) & flagged).sum())
    fn = int(((labels == 1) & ~flagged).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(labels.sum(), 1)

    return {
        "threshold": float(threshold),
        "n_reviewed": int(flagged.sum()),
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": float(precision),
        "recall":    float(recall),
    }


def bootstrap_ci(score, labels, review_fraction, n_boot=500, seed=0):
    """Bootstrap a 95% CI for precision and recall at the given budget.

    Resampling rows of the val set gives us the sampling variability we'd
    expect if we re-ran the experiment on a different cohort of wafers.
    """
    rng = np.random.RandomState(seed)
    precisions, recalls = [], []
    n = len(score)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        result = evaluate_at_budget(score[idx], labels[idx], review_fraction)
        precisions.append(result["precision"])
        recalls.append(result["recall"])
    return {
        "precision_low":  float(np.percentile(precisions, 2.5)),
        "precision_high": float(np.percentile(precisions, 97.5)),
        "recall_low":     float(np.percentile(recalls, 2.5)),
        "recall_high":    float(np.percentile(recalls, 97.5)),
    }


# --- Evaluate every strategy at a range of review budgets
print(f"validation set: {n_val} wafers, {n_failures} failures "
      f"({n_failures/n_val*100:.1f}% base rate)\n")

review_fractions = [0.05, 0.10, 0.15, 0.20, 0.30]
all_results = []

for strategy_name, score in scoring_strategies.items():
    print(f"=== {strategy_name} ===")
    print(f"{'budget':>8} {'thresh':>8} {'reviewed':>9} {'TP':>4} "
          f"{'FP':>4} {'precision':>20} {'recall':>20}")
    for rf in review_fractions:
        point = evaluate_at_budget(score, y_val, rf)
        ci = bootstrap_ci(score, y_val, rf)
        prec_str = f"{point['precision']:.2f} [{ci['precision_low']:.2f}, {ci['precision_high']:.2f}]"
        rec_str  = f"{point['recall']:.2f} [{ci['recall_low']:.2f}, {ci['recall_high']:.2f}]"
        print(f"{int(rf*100):>7}% {point['threshold']:>8.2f} "
              f"{point['n_reviewed']:>9} {point['true_positives']:>4} "
              f"{point['false_positives']:>4} {prec_str:>20} {rec_str:>20}")
        all_results.append({
            "strategy": strategy_name,
            "review_fraction": rf,
            **point,
            **ci,
        })
    print()

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(processed, "calibration_results.csv"), index=False)


# --- The headline plot: catch rate vs review budget for each strategy,
# with confidence bands. This single chart answers: "for any budget you
# pick, which strategy wins, and by how much?"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

colors = {
    "baseline_only":         "#3b82f6",
    "anomaly_only":          "#f59e0b",
    "baseline_and_anomaly":  "#ef4444",
}
labels_pretty = {
    "baseline_only":         "baseline only",
    "anomaly_only":          "anomaly only",
    "baseline_and_anomaly":  "baseline + anomaly",
}

for strategy_name in scoring_strategies:
    sub = results_df[results_df["strategy"] == strategy_name].sort_values("review_fraction")
    x = sub["review_fraction"].values * 100
    color = colors[strategy_name]

    axes[0].plot(x, sub["recall"], marker="o", color=color, label=labels_pretty[strategy_name])
    axes[0].fill_between(x, sub["recall_low"], sub["recall_high"], color=color, alpha=0.15)

    axes[1].plot(x, sub["precision"], marker="o", color=color, label=labels_pretty[strategy_name])
    axes[1].fill_between(x, sub["precision_low"], sub["precision_high"], color=color, alpha=0.15)

axes[0].axhline(1.0, color="#0f172a", linestyle=":", alpha=0.4, label="catch everything")
axes[0].set_xlabel("review budget (% of wafers)")
axes[0].set_ylabel("recall (% of failures caught)")
axes[0].set_title("how many failures do we catch?")
axes[0].legend(loc="lower right")

# Baseline rate line: if we just sampled randomly, what precision would we get?
axes[1].axhline(n_failures / n_val, color="#0f172a", linestyle=":", alpha=0.4,
                label=f"random ({n_failures/n_val*100:.1f}%)")
axes[1].set_xlabel("review budget (% of wafers)")
axes[1].set_ylabel("precision (% of reviews that are real failures)")
axes[1].set_title("how efficient is each review?")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(figures, "calibration_curves.png"), dpi=140, bbox_inches="tight")
plt.show()


# --- Save the calibrated thresholds for the production pipeline (Block 10).
# We pick 10% review budget as the recommended default and bundle everything.
recommended_budget = 0.10

production_config = {
    "recommended_review_budget": recommended_budget,
    "strategies": {},
}
for strategy_name, score in scoring_strategies.items():
    point = evaluate_at_budget(score, y_val, recommended_budget)
    ci    = bootstrap_ci(score, y_val, recommended_budget)
    production_config["strategies"][strategy_name] = {
        "threshold": point["threshold"],
        "expected_precision": point["precision"],
        "expected_recall":    point["recall"],
        "precision_ci_95":    [ci["precision_low"], ci["precision_high"]],
        "recall_ci_95":       [ci["recall_low"], ci["recall_high"]],
    }

with open(os.path.join(processed, "production_thresholds.json"), "w") as f:
    json.dump(production_config, f, indent=2)

print(f"\nrecommended config at {int(recommended_budget*100)}% review budget:")
print(json.dumps(production_config["strategies"], indent=2))
print(f"\nsaved production_thresholds.json and calibration_curves.png")
