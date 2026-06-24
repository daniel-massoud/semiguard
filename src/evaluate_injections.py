# Validates SemiGuard by injecting each known anomaly type into the
# validation set and checking how often each detector flags the injected
# rows. This is the controlled experiment that turns "the detectors
# produce scores" into "the detectors work, and heres which one works
# for which failure mode".

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from anomaly_injectors import ANOMALY_INJECTORS

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

processed = os.path.join(here, "..", "data", "processed")
models_dir = os.path.join(here, "..", "results", "models")
figures = os.path.join(here, "..", "results", "figures")
os.makedirs(figures, exist_ok=True)

# --- Load every trained detector
baseline_model    = joblib.load(os.path.join(models_dir, "baseline_model.pkl"))
mahalanobis_model = joblib.load(os.path.join(models_dir, "anomaly_mahalanobis.pkl"))
isoforest_model   = joblib.load(os.path.join(models_dir, "anomaly_isoforest.pkl"))
lof_model         = joblib.load(os.path.join(models_dir, "anomaly_lof.pkl"))
ocsvm_model       = joblib.load(os.path.join(models_dir, "anomaly_ocsvm.pkl"))

# Autoencoder needs to be rebuilt with the same architecture before loading
class Autoencoder(nn.Module):
    def __init__(self, n_in, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, bottleneck), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(),
            nn.Linear(128, n_in),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

val = pd.read_csv(os.path.join(processed, "val.csv"))
sensor_cols = [c for c in val.columns if c.startswith("sensor_")]

ae = Autoencoder(n_in=len(sensor_cols))
ae.load_state_dict(torch.load(os.path.join(models_dir, "autoencoder.pt"),
                              map_location="cpu"))
ae.eval()


def score_all_detectors(X):
    """Run every detector on a feature matrix and return a dict of scores.

    All scores are oriented so that higher = more anomalous, matching the
    convention from Blocks 4-6.
    """
    scores = {}
    scores["mahalanobis"]      = mahalanobis_model.mahalanobis(X)
    scores["isolation_forest"] = -isoforest_model.score_samples(X)
    scores["lof"]              = -lof_model.score_samples(X)
    scores["ocsvm"]            = -ocsvm_model.decision_function(X)

    with torch.no_grad():
        Xt = torch.tensor(X.astype(np.float32))
        recon = ae(Xt).numpy()
    scores["autoencoder"] = ((recon - X) ** 2).mean(axis=1)

    # Baseline-derived OOD: we keep just one since Block 6 showed all three
    # are perfectly correlated in a binary classifier.
    probs = baseline_model.predict_proba(X)
    scores["tree_disagreement"] = np.array([
        t.predict_proba(X)[:, 1] for t in baseline_model.estimators_
    ]).std(axis=0)

    return scores


# --- Run each anomaly injector, score the corrupted dataset, and compute
# AUROC of each detector vs the injection mask. AUROC works well here
# because we have a balanced two-class problem (injected vs clean).
print("running injection experiments...\n")
results = []

for anomaly_name, injector in ANOMALY_INJECTORS.items():
    corrupted_df, mask, params = injector(val, sensor_cols)
    X_corrupt = corrupted_df[sensor_cols].values
    detector_scores = score_all_detectors(X_corrupt)

    n_corrupted = int(mask.sum())
    print(f"{anomaly_name}: {n_corrupted}/{len(val)} rows corrupted, params={params}")

    for detector_name, scores in detector_scores.items():
        auroc = roc_auc_score(mask, scores)
        results.append({
            "anomaly_type": anomaly_name,
            "detector": detector_name,
            "auroc": auroc,
            "n_corrupted": n_corrupted,
        })

results_df = pd.DataFrame(results)

# --- Pivot into the diagnostic table: detectors as columns, anomaly types as
# rows. AUROC of 1.0 = perfect detection, 0.5 = random.
pivot = results_df.pivot(index="anomaly_type", columns="detector", values="auroc")
# Reorder columns into a logical grouping
detector_order = ["mahalanobis", "isolation_forest", "lof", "ocsvm",
                  "autoencoder", "tree_disagreement"]
pivot = pivot[detector_order]

print("\n--- DETECTION AUROC PER ANOMALY TYPE x DETECTOR ---")
print(pivot.round(3).to_string())

# --- Heatmap version of the table
fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
    cbar_kws={"label": "AUROC (0.5 = random, 1.0 = perfect)"}, ax=ax,
)
ax.set_title("which detector catches which anomaly type?")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(figures, "injection_auroc_matrix.png"),
            dpi=140, bbox_inches="tight")
plt.show()

# --- Also rank-average the input-anomaly detectors and check the ensemble.
# If the ensemble is at least as good as the best individual on every
# anomaly type, then "always use the ensemble" is a defensible default.
from scipy.stats import rankdata

print("\n--- ENSEMBLE PERFORMANCE ---")
input_detectors = ["mahalanobis", "isolation_forest", "lof", "ocsvm", "autoencoder"]
for anomaly_name, injector in ANOMALY_INJECTORS.items():
    corrupted_df, mask, _ = injector(val, sensor_cols)
    X_corrupt = corrupted_df[sensor_cols].values
    scores = score_all_detectors(X_corrupt)

    ranked = np.column_stack([rankdata(scores[d]) for d in input_detectors])
    ensemble = ranked.mean(axis=1)
    auroc_ens = roc_auc_score(mask, ensemble)
    best_individual = max(roc_auc_score(mask, scores[d]) for d in input_detectors)
    print(f"  {anomaly_name:15s}  ensemble: {auroc_ens:.3f}  "
          f"vs best individual: {best_individual:.3f}")

# --- Save the full results for the report
results_df.to_csv(os.path.join(processed, "injection_results.csv"), index=False)
pivot.to_csv(os.path.join(processed, "injection_results_matrix.csv"))

print(f"\nsaved injection_results.csv and matrix to {processed}")
print(f"saved injection_auroc_matrix.png to {figures}")
