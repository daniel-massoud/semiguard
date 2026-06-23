# Downloads SECOM directly from the UCI archive and saves two CSVs to data/raw.
# The labels file has the format:  -1 "19/07/2008 11:55:00"
# i.e. a signed integer followed by a quoted timestamp. We parse it by hand
# because the quotes confuse pandas-style whitespace splitting.

import os
import urllib.request
import pandas as pd

features_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
labels_url   = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

here = os.path.dirname(os.path.abspath(__file__))
raw_folder = os.path.join(here, "..", "data", "raw")
os.makedirs(raw_folder, exist_ok=True)

# --- Sensor readings: 1567 rows x 590 columns of floats, "NaN" for missing.
print("Downloading sensor readings...")
sensor_readings = pd.read_csv(
    features_url,
    sep=r"\s+",
    header=None,
    na_values="NaN",
    engine="python",
)
sensor_readings.columns = [f"sensor_{i}" for i in range(sensor_readings.shape[1])]

# --- Labels: download the raw text, then parse one line at a time.
print("Downloading labels...")
with urllib.request.urlopen(labels_url) as response:
    raw_text = response.read().decode("utf-8")

parsed_rows = []
for line in raw_text.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    # Split into the label (first whitespace-delimited token) and the rest.
    label_str, _, timestamp_str = line.partition(" ")
    # Strip the surrounding double quotes from the timestamp.
    timestamp_str = timestamp_str.strip().strip(chr(34))
    parsed_rows.append({
        "label": int(label_str),         # -1 = pass, +1 = fail
        "timestamp_text": timestamp_str,
    })

labels = pd.DataFrame(parsed_rows)
labels["timestamp"] = pd.to_datetime(labels["timestamp_text"], format="%d/%m/%Y %H:%M:%S")
labels = labels[["label", "timestamp"]]

# --- Sanity check
n_wafers   = len(sensor_readings)
n_sensors  = sensor_readings.shape[1]
n_failures = (labels["label"] == 1).sum()

assert len(labels) == n_wafers, f"row count mismatch: {len(labels)} labels vs {n_wafers} wafers"

print(f"  wafers:     {n_wafers}")
print(f"  sensors:    {n_sensors}")
print(f"  failures:   {n_failures} ({n_failures / n_wafers * 100:.1f}%)")
print(f"  date range: {labels['timestamp'].min()}  ->  {labels['timestamp'].max()}")

sensor_readings.to_csv(os.path.join(raw_folder, "secom_features.csv"), index=False)
labels.to_csv(os.path.join(raw_folder, "secom_targets.csv"), index=False)

print(f"\nSaved to {os.path.abspath(raw_folder)}")

