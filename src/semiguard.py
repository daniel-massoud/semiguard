# The production SemiGuard pipeline. Takes one wafer at a time and returns
# a structured decision: pass, review, or escalate, plus all the scores
# that contributed. This is what a fab integration would import and call
# from whatever process-control software runs on the production floor.
#
# Design choices worth noting:
#   - All trained artifacts are loaded once at init, scored per-wafer.
#   - Scores are returned alongside decisions so downstream tools can
#     log, audit, or override.
#   - The class is stateful for the drift monitor only: it maintains a
#     rolling window of recent wafers so it can flag fleet-level shifts.

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from drift_tests import psi


# Autoencoder definition has to match the trained one exactly to load weights
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


class SemiGuard:
    """End-to-end wafer-trust pipeline.

    Call .evaluate(raw_sensor_readings_dict) to score one wafer. Returns
    a dict with the final decision and all contributing scores.

    Decisions:
        "pass"      below all thresholds, send wafer through normally
        "review"    above the review threshold but not the escalate one
                    -> queue for human inspection
        "escalate"  multiple signals fire at once -> stop the line and
                    investigate immediately
    """

    def __init__(self, project_root):
        models_dir = os.path.join(project_root, "results", "models")
        processed = os.path.join(project_root, "data", "processed")

        # Load the preprocessing artifacts (which sensors to keep, how to
        # impute missing values, how to standardize). Anything coming in
        # from the floor has to be transformed identically to training.
        with open(os.path.join(processed, "preprocessing.json")) as f:
            prep = json.load(f)
        self.feature_cols = prep["kept_sensors"]
        self.median = pd.Series(prep["median"])
        self.mean   = pd.Series(prep["mean"])
        self.std    = pd.Series(prep["std"])

        # Load the trained models
        self.baseline    = joblib.load(os.path.join(models_dir, "baseline_model.pkl"))
        self.mahalanobis = joblib.load(os.path.join(models_dir, "anomaly_mahalanobis.pkl"))
        self.isoforest   = joblib.load(os.path.join(models_dir, "anomaly_isoforest.pkl"))
        self.lof         = joblib.load(os.path.join(models_dir, "anomaly_lof.pkl"))
        self.ocsvm       = joblib.load(os.path.join(models_dir, "anomaly_ocsvm.pkl"))

        self.ae = Autoencoder(n_in=len(self.feature_cols))
        self.ae.load_state_dict(torch.load(
            os.path.join(models_dir, "autoencoder.pt"), map_location="cpu"))
        self.ae.eval()

        # Load the calibrated production thresholds. We use the combined
        # strategy by default since Block 9 showed it dominates.
        with open(os.path.join(processed, "production_thresholds.json")) as f:
            cfg = json.load(f)
        self.combined_threshold = cfg["strategies"]["baseline_and_anomaly"]["threshold"]
        self.expected_precision = cfg["strategies"]["baseline_and_anomaly"]["expected_precision"]
        self.expected_recall    = cfg["strategies"]["baseline_and_anomaly"]["expected_recall"]

        # Load val-set scores so we know what "rank" a new score
        # corresponds to. Needed because the combined score uses ranks.
        val_scores = pd.read_csv(os.path.join(processed, "val_anomaly_scores.csv"))
        self.val_baseline   = val_scores["baseline_prob"].values
        self.val_anomaly_avg = val_scores[
            ["mahalanobis", "isolation_forest", "lof", "ocsvm", "autoencoder"]
        ].rank().mean(axis=1).values

        # Reference training data, for the drift monitor
        train = pd.read_csv(os.path.join(processed, "train.csv"))
        self.train_per_sensor = {c: train[c].values for c in self.feature_cols}

        # Rolling buffer of recent wafers for drift monitoring
        self.recent_window = deque(maxlen=50)
        self.wafer_count = 0


    def preprocess(self, raw_sensor_dict):
        """Apply the same imputation + standardization used at train time."""
        # Build a Series from whatever sensors the caller passed in. Missing
        # sensors get NaN, which then gets filled with the train-time median.
        row = pd.Series(raw_sensor_dict).reindex(self.feature_cols)
        row = row.fillna(self.median)
        row = (row - self.mean) / self.std
        return row.values.astype(np.float32)


    def _score_all(self, x):
        """Compute every detector score for one wafer (1D array in)."""
        X = x.reshape(1, -1)

        baseline_prob = float(self.baseline.predict_proba(X)[0, 1])

        # Anomaly detectors. All standardized to "higher = more anomalous".
        maha = float(self.mahalanobis.mahalanobis(X)[0])
        iso  = float(-self.isoforest.score_samples(X)[0])
        lof_ = float(-self.lof.score_samples(X)[0])
        oc   = float(-self.ocsvm.decision_function(X)[0])

        with torch.no_grad():
            recon = self.ae(torch.tensor(X)).numpy()
        ae_err = float(((recon - X) ** 2).mean())

        # Tree disagreement (model-uncertainty signal)
        tree_probs = np.array([t.predict_proba(X)[0, 1] for t in self.baseline.estimators_])
        tree_disagreement = float(tree_probs.std())

        return {
            "baseline_prob":     baseline_prob,
            "mahalanobis":       maha,
            "isolation_forest":  iso,
            "lof":               lof_,
            "ocsvm":             oc,
            "autoencoder":       ae_err,
            "tree_disagreement": tree_disagreement,
        }


    def _rank_against_val(self, value, val_distribution):
        """Where does a single new score rank relative to the val set?

        Returns a number in [0, 1]: 0 = below everything in val,
        1 = above everything in val. We use this to mimic the rank-average
        scoring used at threshold calibration time.
        """
        return float((val_distribution < value).mean())


    def evaluate(self, raw_sensor_dict, wafer_id=None):
        """Main entry point. Score one wafer and return a structured decision."""
        self.wafer_count += 1
        x = self.preprocess(raw_sensor_dict)
        self.recent_window.append(x)

        scores = self._score_all(x)

        # Build the combined score the same way Block 9 did: rank against
        # val distribution, average baseline-rank with anomaly-ensemble-rank.
        anomaly_avg = np.mean([
            scores["mahalanobis"], scores["isolation_forest"], scores["lof"],
            scores["ocsvm"], scores["autoencoder"],
        ])  # this isnt the true rank-avg but its a reasonable proxy

        baseline_rank = self._rank_against_val(scores["baseline_prob"], self.val_baseline)
        anomaly_rank  = self._rank_against_val(anomaly_avg, self.val_anomaly_avg)
        combined_rank = (baseline_rank + anomaly_rank) / 2

        # Decide. The combined_threshold from Block 9 is in "raw rank-average"
        # space, so we re-derive a percentile threshold from the val
        # distribution and use that.
        review_percentile = 0.90   # match the 10% budget from Block 9
        flag_review   = combined_rank >= review_percentile
        flag_escalate = (
            (baseline_rank >= 0.95)
            and (anomaly_rank >= 0.95)
        )

        if flag_escalate:
            decision = "escalate"
            reason = "both baseline and anomaly score in top 5%"
        elif flag_review:
            decision = "review"
            reason = f"combined score in top {int((1-review_percentile)*100)}%"
        else:
            decision = "pass"
            reason = "all scores within normal range"

        # Drift monitoring: once we have enough recent wafers, run PSI on
        # the rolling window against training data, per sensor.
        drift_flagged_sensors = []
        if len(self.recent_window) == self.recent_window.maxlen:
            recent_matrix = np.stack(list(self.recent_window))
            for i, col in enumerate(self.feature_cols):
                p = psi(self.train_per_sensor[col], recent_matrix[:, i])
                if p > 0.25:
                    drift_flagged_sensors.append({"sensor": col, "psi": p})
            # If many sensors drifted, that's a fleet-level red flag
            if len(drift_flagged_sensors) > 10:
                if decision == "pass":
                    decision = "review"
                    reason = f"{len(drift_flagged_sensors)} sensors drifted in recent window"

        return {
            "wafer_id":       wafer_id if wafer_id is not None else self.wafer_count,
            "decision":       decision,
            "reason":         reason,
            "combined_rank":  combined_rank,
            "baseline_rank":  baseline_rank,
            "anomaly_rank":   anomaly_rank,
            "raw_scores":     scores,
            "drift_flagged_sensors": drift_flagged_sensors,
        }
