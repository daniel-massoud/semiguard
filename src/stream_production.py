# Streams the held-out test set through the SemiGuard pipeline one wafer at
# a time, simulating production deployment. Prints a running log and a
# final summary showing how many wafers went to each decision bucket and
# how many real failures we caught.

import os
import sys
import time
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from semiguard import SemiGuard

project_root = os.path.dirname(here)

print("loading SemiGuard pipeline...")
guard = SemiGuard(project_root=project_root)
print("ready.\n")

# Test set was held back until now — pipeline has never seen these wafers
test = pd.read_csv(os.path.join(project_root, "data", "processed", "test.csv"))
sensor_cols = guard.feature_cols

print(f"streaming {len(test)} test wafers...\n")
print(f"{'wafer':>5}  {'decision':>9}  {'comb':>5}  {'baseln':>6}  {'anom':>5}  "
      f"{'true':>5}  reason")
print("-" * 100)

results = []
start = time.time()

for i, row in test.iterrows():
    raw = {col: row[col] for col in sensor_cols}
    result = guard.evaluate(raw, wafer_id=i)
    result["true_label"] = int(row["label"])
    results.append(result)

    # Print every 10th wafer so the log doesnt explode but you can still
    # see things happening live.
    if i % 10 == 0 or result["decision"] != "pass":
        print(f"{i:>5}  {result['decision']:>9}  "
              f"{result['combined_rank']:>5.2f}  "
              f"{result['baseline_rank']:>6.2f}  "
              f"{result['anomaly_rank']:>5.2f}  "
              f"{result['true_label']:>5}  {result['reason']}")

elapsed = time.time() - start

# --- Summary
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print(f"streamed {len(results)} wafers in {elapsed:.1f}s "
      f"({len(results)/elapsed:.0f} wafers/sec)\n")

decision_counts = results_df["decision"].value_counts()
print("decisions:")
for decision, count in decision_counts.items():
    pct = count / len(results_df) * 100
    print(f"  {decision:>10}  {count:>4}  ({pct:.1f}%)")

# How many real failures did each bucket contain?
print("\nreal failures per bucket:")
for decision in ["escalate", "review", "pass"]:
    bucket = results_df[results_df["decision"] == decision]
    if len(bucket) == 0:
        continue
    n_fail = int(bucket["true_label"].sum())
    n_total = len(bucket)
    pct = n_fail / max(n_total, 1) * 100
    print(f"  {decision:>10}  {n_fail}/{n_total} wafers were real failures ({pct:.1f}%)")

# Headline metrics
n_failures_total = int(results_df["true_label"].sum())
n_caught = int(results_df[results_df["decision"] != "pass"]["true_label"].sum())
n_reviewed = int((results_df["decision"] != "pass").sum())

print(f"\nheadline:")
print(f"  total failures in test set:         {n_failures_total}")
print(f"  failures caught (review+escalate):  {n_caught} ({n_caught/max(n_failures_total,1)*100:.0f}%)")
print(f"  wafers flagged for review:          {n_reviewed} ({n_reviewed/len(results_df)*100:.0f}%)")
print(f"  precision (real fails / flagged):   {n_caught/max(n_reviewed,1)*100:.0f}%")

# Save the full stream log for the dashboard to load in Block 11
results_df_flat = results_df.copy()
# Flatten the nested raw_scores dict into individual columns
scores_expanded = pd.json_normalize(results_df_flat["raw_scores"])
results_df_flat = pd.concat([
    results_df_flat.drop(columns=["raw_scores", "drift_flagged_sensors"]),
    scores_expanded
], axis=1)
out_path = os.path.join(project_root, "data", "processed", "production_stream_log.csv")
results_df_flat.to_csv(out_path, index=False)
print(f"\nsaved stream log to {out_path}")
