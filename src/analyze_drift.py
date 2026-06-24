# Runs PSI, KS, and MMD on the SECOM training set vs the validation set,
# both per-sensor and over weekly time windows. The expected outcome:
# substantial drift, because we already know the failure rate dropped from
# 8.1% in train to 3.5% in val (and we saw the failure-rate-over-time plot
# in Block 2 wandering visibly).

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)  # so we can import drift_tests as a sibling

from drift_tests import psi, ks_statistic, mmd_rbf

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

processed = os.path.join(here, "..", "data", "processed")
figures = os.path.join(here, "..", "results", "figures")
os.makedirs(figures, exist_ok=True)

train = pd.read_csv(os.path.join(processed, "train.csv"))
val   = pd.read_csv(os.path.join(processed, "val.csv"),  parse_dates=["timestamp"])
test  = pd.read_csv(os.path.join(processed, "test.csv"), parse_dates=["timestamp"])

sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
print(f"sensors to monitor: {len(sensor_cols)}")

# ---------- per-sensor: train vs val ----------
print("\nrunning per-sensor PSI and KS (train vs val)...")
rows = []
for col in sensor_cols:
    p     = psi(train[col].values, val[col].values)
    ks, pv = ks_statistic(train[col].values, val[col].values)
    rows.append({"sensor": col, "psi": p, "ks_stat": ks, "ks_pvalue": pv})
drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)

# Categorize PSI into the standard buckets
drift_df["severity"] = pd.cut(
    drift_df["psi"],
    bins=[-np.inf, 0.1, 0.25, np.inf],
    labels=["stable", "moderate", "significant"],
)
counts = drift_df["severity"].value_counts()

print(f"\n  stable (PSI < 0.10):           {counts.get('stable', 0):>3} sensors")
print(f"  moderate (0.10 <= PSI < 0.25): {counts.get('moderate', 0):>3} sensors")
print(f"  significant (PSI >= 0.25):     {counts.get('significant', 0):>3} sensors")
print(f"\nKS test: {(drift_df['ks_pvalue'] < 0.05).sum()} / {len(drift_df)} "
      f"sensors significant at p<0.05")
print(f"  (Bonferroni-corrected at alpha=0.05: "
      f"{(drift_df['ks_pvalue'] < 0.05 / len(drift_df)).sum()} sensors)")

# ---------- top-N drift plot ----------
top_n = 25
top_drift = drift_df.head(top_n)

fig, ax = plt.subplots(figsize=(8, 6))
colors = top_drift["severity"].map(
    {"stable": "#94a3b8", "moderate": "#f59e0b", "significant": "#ef4444"}
)
ax.barh(top_drift["sensor"], top_drift["psi"], color=colors)
ax.axvline(0.10, color="#0f172a", linestyle=":", alpha=0.5, label="moderate")
ax.axvline(0.25, color="#0f172a", linestyle="--", alpha=0.7, label="significant")
ax.set_xlabel("PSI (train -> val)")
ax.set_title(f"top {top_n} sensors by drift")
ax.legend(loc="lower right")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(figures, "drift_top_sensors.png"), dpi=140, bbox_inches="tight")
plt.show()

# ---------- multivariate MMD on top-drifting sensors ----------
# Run MMD on the 20 most-drifting sensors as a multivariate joint check.
# This will catch joint shifts a univariate test cant see.
top_sensors = drift_df["sensor"].head(20).tolist()
mmd_joint = mmd_rbf(train[top_sensors].values, val[top_sensors].values)
print(f"\nmultivariate MMD on top-20 drifting sensors (train vs val): {mmd_joint:.4f}")

# Compare against test set too, so we can see if drift continues
mmd_train_test = mmd_rbf(train[top_sensors].values, test[top_sensors].values)
print(f"multivariate MMD on same sensors  (train vs test):           {mmd_train_test:.4f}")

# ---------- drift over time: weekly windows ----------
# Combine val+test into a "post-deployment" timeline and chunk into weeks.
# For each week, compute mean PSI across all sensors and the failure rate.
# If drift correlates with failure rate over time, the production system
# should treat high-drift weeks as low-trust periods.
print("\nrunning weekly drift over post-train timeline...")
post_train = pd.concat([val, test], ignore_index=True).sort_values("timestamp")
post_train["week"] = post_train["timestamp"].dt.to_period("W").dt.start_time

weekly = []
for week_start, chunk in post_train.groupby("week"):
    if len(chunk) < 20:
        continue   # too few wafers in this week to give a stable estimate
    psi_per_sensor = [psi(train[c].values, chunk[c].values) for c in sensor_cols]
    weekly.append({
        "week": week_start,
        "n_wafers":           len(chunk),
        "mean_psi":           float(np.mean(psi_per_sensor)),
        "p95_psi":            float(np.percentile(psi_per_sensor, 95)),
        "failure_rate":       float(chunk["label"].mean()),
        "n_sensors_shifted":  int(sum(p > 0.25 for p in psi_per_sensor)),
    })
weekly_df = pd.DataFrame(weekly)
print(weekly_df.round(4).to_string(index=False))

# ---------- drift-over-time visualization ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

axes[0].plot(weekly_df["week"], weekly_df["p95_psi"], marker="o",
             color="#0f172a", label="95th-percentile PSI across sensors")
axes[0].axhline(0.25, color="#ef4444", linestyle="--", alpha=0.7, label="significant")
axes[0].axhline(0.10, color="#f59e0b", linestyle=":",  alpha=0.7, label="moderate")
axes[0].set_ylabel("PSI")
axes[0].set_title("weekly drift vs failure rate (post-training period)")
axes[0].legend(fontsize=8, loc="upper left")

axes[1].plot(weekly_df["week"], weekly_df["failure_rate"], marker="o", color="#ef4444")
axes[1].set_ylabel("failure rate")
axes[1].set_xlabel("week")

plt.tight_layout()
plt.savefig(os.path.join(figures, "drift_over_time.png"), dpi=140, bbox_inches="tight")
plt.show()

# ---------- save everything ----------
drift_df.to_csv(os.path.join(processed, "drift_per_sensor.csv"), index=False)
weekly_df.to_csv(os.path.join(processed, "drift_weekly.csv"), index=False)

print(f"\nsaved drift_per_sensor.csv and drift_weekly.csv to {processed}")
print(f"saved drift_top_sensors.png and drift_over_time.png to {figures}")
