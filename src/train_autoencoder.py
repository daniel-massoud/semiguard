# Trains a small autoencoder on PASSING wafers only and uses reconstruction
# error as an anomaly score. The intuition: the model learns to compress
# and rebuild normal sensor patterns. When a failing or anomalous wafer
# arrives, the model has never seen its pattern, so reconstruction is poor
# and the squared error is high. That error is our anomaly score.
#
# We deliberately keep the network small (444 -> 128 -> 32 -> 128 -> 444).
# With only ~860 passing training wafers, a bigger network would memorize
# the training set and lose its ability to flag truly anomalous inputs.

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Reproducibility. Without these, every run gives different scores.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

here = os.path.dirname(os.path.abspath(__file__))
processed = os.path.join(here, "..", "data", "processed")
models_dir = os.path.join(here, "..", "results", "models")
os.makedirs(models_dir, exist_ok=True)

# --- Load splits and keep only passing wafers from train. This is the
# crucial step: the model should learn "normal", not "average of everything".
train = pd.read_csv(os.path.join(processed, "train.csv"))
val   = pd.read_csv(os.path.join(processed, "val.csv"))

sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
train_pass_only = train[train["label"] == 0]

X_train = train_pass_only[sensor_cols].values.astype(np.float32)
X_val   = val[sensor_cols].values.astype(np.float32)
y_val   = val["label"].values

n_features = X_train.shape[1]
print(f"training on {len(X_train)} passing wafers, {n_features} features")
print(f"validating on {len(X_val)} wafers ({y_val.sum()} failures)")


# --- The autoencoder. Symmetric encoder/decoder with a 32-dim bottleneck.
# We use Tanh in the bottleneck because our features are standardized
# (roughly zero-mean, unit-variance), so values are bounded around [-3, 3].
class Autoencoder(nn.Module):
    def __init__(self, n_in, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.ReLU(),
            nn.Linear(128, n_in),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# --- Training setup. CPU is fine here — the dataset is tiny.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Autoencoder(n_in=n_features).to(device)

X_train_tensor = torch.tensor(X_train, device=device)
dataset = TensorDataset(X_train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# --- Train. We hold out 10% of the passing training data as an internal
# validation set — NOT the same as our overall val split. This is used only
# to monitor overfitting and pick the best epoch.
n_internal_val = max(50, len(X_train) // 10)
internal_val = X_train_tensor[-n_internal_val:]
internal_train = X_train_tensor[:-n_internal_val]
internal_loader = DataLoader(TensorDataset(internal_train), batch_size=32, shuffle=True)

train_losses, val_losses = [], []
best_val_loss = float("inf")
best_state = None

n_epochs = 100
print(f"\ntraining for up to {n_epochs} epochs...")
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for (batch,) in internal_loader:
        optimizer.zero_grad()
        reconstruction = model(batch)
        loss = loss_fn(reconstruction, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch)
    epoch_loss /= len(internal_train)

    model.eval()
    with torch.no_grad():
        val_recon = model(internal_val)
        val_loss = loss_fn(val_recon, internal_val).item()

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)

    # Track the best model so far. We don't want the last epoch — we want
    # the epoch where the internal val loss was lowest (early stopping).
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:3d}  train MSE {epoch_loss:.4f}  val MSE {val_loss:.4f}")

print(f"\nbest internal val MSE: {best_val_loss:.4f}")
model.load_state_dict(best_state)
model.eval()


# --- Score the overall val split: reconstruct each wafer and compute
# the per-sample mean squared error. Higher = more anomalous.
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val, device=device)
    val_reconstruction = model(X_val_tensor)
    per_sample_mse = ((val_reconstruction - X_val_tensor) ** 2).mean(dim=1).cpu().numpy()

# Quick check that the reconstruction error actually separates the classes.
from sklearn.metrics import average_precision_score
ap = average_precision_score(y_val, per_sample_mse)
print(f"\nautoencoder reconstruction error PR-AUC vs failures: {ap:.3f}")

# Mean reconstruction error on passes vs fails — should be visibly different.
mean_pass = per_sample_mse[y_val == 0].mean()
mean_fail = per_sample_mse[y_val == 1].mean()
print(f"  mean recon error on passes: {mean_pass:.4f}")
print(f"  mean recon error on fails:  {mean_fail:.4f}  ({mean_fail / mean_pass:.2f}x higher)")


# --- Save the trained model, the training curves (for the report), and
# the val-set anomaly scores (for Block 9 calibration).
torch.save(model.state_dict(), os.path.join(models_dir, "autoencoder.pt"))

with open(os.path.join(models_dir, "autoencoder_meta.json"), "w") as f:
    json.dump({
        "n_features": n_features,
        "bottleneck": 32,
        "best_internal_val_mse": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, f, indent=2)

# Append the AE score to the val scores we already saved in Block 4.
val_scores_path = os.path.join(processed, "val_anomaly_scores.csv")
val_scores = pd.read_csv(val_scores_path)
val_scores["autoencoder"] = per_sample_mse
val_scores.to_csv(val_scores_path, index=False)

print(f"\nsaved model and updated val_anomaly_scores.csv")
