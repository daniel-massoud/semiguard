# Synthetic anomaly injectors. Each function takes a clean (pre-processed)
# dataframe and returns a corrupted copy plus a boolean mask of which rows
# were corrupted. These mimic real fab failure modes:
#
#   stuck_sensor    -> calibration locked, sensor reports a constant value
#   sensor_spike    -> transient extreme reading (electrical fault, etc.)
#   sensor_noise    -> sensor becomes noisy (loose connection, EMI)
#   sensor_drift    -> gradual offset over time (chamber contamination)
#   global_shift    -> all sensors shift slightly (recalibration event)
#
# We work on the STANDARDIZED feature space because thats what the
# detectors operate on. A "spike to value 8" in standardized space means
# "8 standard deviations from the training mean" — extreme by construction.

import numpy as np
import pandas as pd


def _pick_rows(n_total, fraction, rng):
    """Sample row indices to corrupt without replacement."""
    n_corrupt = max(1, int(n_total * fraction))
    return rng.choice(n_total, n_corrupt, replace=False)


def stuck_sensor(df, sensor_cols, fraction=0.05, n_sensors_stuck=3, seed=0):
    """Pick N random sensors and freeze them at their median for some rows."""
    rng = np.random.RandomState(seed)
    out = df.copy()
    mask = np.zeros(len(df), dtype=bool)

    corrupt_idx = _pick_rows(len(df), fraction, rng)
    mask[corrupt_idx] = True

    stuck_sensors = rng.choice(sensor_cols, n_sensors_stuck, replace=False)
    for sensor in stuck_sensors:
        # Standardized features have median ~0, so "stuck" means stuck at 0.
        out.loc[out.index[corrupt_idx], sensor] = 0.0
    return out, mask, {"sensors_affected": list(stuck_sensors)}


def sensor_spike(df, sensor_cols, fraction=0.05, spike_magnitude=8.0, seed=1):
    """Inject extreme spikes (~8 sigma) in a randomly chosen sensor per row."""
    rng = np.random.RandomState(seed)
    out = df.copy()
    mask = np.zeros(len(df), dtype=bool)

    corrupt_idx = _pick_rows(len(df), fraction, rng)
    mask[corrupt_idx] = True

    for row in corrupt_idx:
        sensor = rng.choice(sensor_cols)
        sign = rng.choice([-1, 1])
        out.iloc[row, out.columns.get_loc(sensor)] = sign * spike_magnitude
    return out, mask, {"magnitude_sigmas": spike_magnitude}


def sensor_noise(df, sensor_cols, fraction=0.05, n_sensors_noisy=5,
                 noise_std=3.0, seed=2):
    """Add Gaussian noise (noise_std sigma) to N sensors on some rows."""
    rng = np.random.RandomState(seed)
    out = df.copy()
    mask = np.zeros(len(df), dtype=bool)

    corrupt_idx = _pick_rows(len(df), fraction, rng)
    mask[corrupt_idx] = True

    noisy_sensors = rng.choice(sensor_cols, n_sensors_noisy, replace=False)
    for sensor in noisy_sensors:
        col_idx = out.columns.get_loc(sensor)
        noise = rng.normal(0, noise_std, size=len(corrupt_idx))
        out.iloc[corrupt_idx, col_idx] = out.iloc[corrupt_idx, col_idx].values + noise
    return out, mask, {"sensors_affected": list(noisy_sensors),
                       "noise_std_sigmas": noise_std}


def sensor_drift(df, sensor_cols, fraction=0.25, n_sensors_drifting=4,
                 max_drift=2.5, seed=3):
    """Gradual linear drift on N sensors over the LAST `fraction` of rows.

    Unlike the others this is contiguous: imagine a slowly-degrading sensor
    that gets worse over the course of a week. The drift starts at 0 and
    grows linearly to max_drift sigmas by the end.
    """
    rng = np.random.RandomState(seed)
    out = df.copy()
    mask = np.zeros(len(df), dtype=bool)

    start = int(len(df) * (1 - fraction))
    mask[start:] = True

    drifting_sensors = rng.choice(sensor_cols, n_sensors_drifting, replace=False)
    drift_curve = np.linspace(0, max_drift, len(df) - start)
    for sensor in drifting_sensors:
        col_idx = out.columns.get_loc(sensor)
        original = out.iloc[start:, col_idx].values
        out.iloc[start:, col_idx] = original + drift_curve
    return out, mask, {"sensors_affected": list(drifting_sensors),
                       "max_drift_sigmas": max_drift}


def global_shift(df, sensor_cols, fraction=0.2, shift_magnitude=0.5, seed=4):
    """Shift ALL sensors by a small amount on some rows (recalibration event)."""
    rng = np.random.RandomState(seed)
    out = df.copy()
    mask = np.zeros(len(df), dtype=bool)

    corrupt_idx = _pick_rows(len(df), fraction, rng)
    mask[corrupt_idx] = True

    # Apply the same shift to every sensor, every corrupted row. This is
    # subtle per-sensor but coherent across the joint distribution —
    # exactly the kind of thing MMD catches and per-sensor tests miss.
    for sensor in sensor_cols:
        col_idx = out.columns.get_loc(sensor)
        out.iloc[corrupt_idx, col_idx] = out.iloc[corrupt_idx, col_idx].values + shift_magnitude
    return out, mask, {"shift_magnitude_sigmas": shift_magnitude}


# Registry so the evaluation script can iterate over all anomaly types
ANOMALY_INJECTORS = {
    "stuck_sensor": stuck_sensor,
    "sensor_spike": sensor_spike,
    "sensor_noise": sensor_noise,
    "sensor_drift": sensor_drift,
    "global_shift": global_shift,
}
