import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import glob
import os
import datetime

# Fixed seeds so the reported held-out R2 is reproducible run-to-run.
# (With this small 3-run dataset the metric is otherwise high-variance —
# see the reproducibility/limitations note in the README.)
torch.manual_seed(42)
np.random.seed(42)

# 1. Load and Balance the Dataset
# Repo-relative data dir so this runs from a fresh clone on any machine.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.environ.get("NEUROTRACTION_DATA", os.path.join(_THIS_DIR, "..", "data"))
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {data_dir}. Set NEUROTRACTION_DATA or add data/.")

# Load each recording separately, preserving row order and tagging the source
# recording. Rows within a recording are a time series (adjacent rows highly
# correlated), and each fixed-velocity run ends in a constant fully-slipping
# tail. Two honest consequences:
#   * A pure random row split leaks (neighbors land in both train and test) and
#     inflates R2 — this is what the earlier 0.945 number did.
#   * A pure temporal tail split makes the test set the constant slip=1.0 tail
#     (zero variance -> R2 undefined/0), which is also not representative.
# With only 3 short scripted runs there is no clean episode-level split. We use
# a per-recording strided hold-out (every 5th sample -> test) which keeps both
# grip and slip regimes in the test set and is far less leaky than random row
# shuffling. The honest caveat: adjacent strided samples are still ~0.05 s apart,
# so this is a limited proxy for a true multi-recording hold-out. See README.
frames = []
for rec_id, f in enumerate(sorted(csv_files)):
    d = pd.read_csv(f).rename(columns=lambda x: x.strip())
    d = d.loc[:, ~d.columns.duplicated()].dropna().reset_index(drop=True)
    d["_rec"] = rec_id
    d["_order"] = np.arange(len(d))
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

FEATURES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'v_enc']

# Per-recording strided hold-out: every 5th row (by time order) -> test.
test_mask = np.zeros(len(df), dtype=bool)
for rec_id, g in df.groupby("_rec"):
    ordered = g.sort_values("_order").index
    test_mask[ordered[4::5]] = True   # indices 4,9,14,... within each recording

train_df = df[~test_mask]
test_df = df[test_mask]

is_traction_train = (train_df['slip_ratio'] < 0.1).values
traction_count = int(np.sum(is_traction_train))
slip_count = len(train_df) - traction_count
weight_for_traction = slip_count / max(traction_count, 1)

print(f"Split: {len(train_df)} train / {len(test_df)} test rows "
      f"(per-recording strided hold-out over {df['_rec'].nunique()} recordings)")
print(f"Test slip std: {test_df['slip_ratio'].std():.3f} (must be >0 for a meaningful R2)")
print(f"Train balance: {traction_count} Traction vs {slip_count} Slip")

X_train_raw = train_df[FEATURES].values
X_test_raw = test_df[FEATURES].values
y_train = train_df['slip_ratio'].values.reshape(-1, 1)
y_test = test_df['slip_ratio'].values.reshape(-1, 1)
weight_train = is_traction_train

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Convert to Tensors
X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test); y_test = torch.FloatTensor(y_test)

# 3. Architecture (Twin of your Inference)
class TractionNet(nn.Module):
    def __init__(self):
        super(TractionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

model = TractionNet()

# 4. CUSTOM WEIGHTED LOSS
# Instead of standard MSE, we manually weight the "Traction" errors
def weighted_mse_loss(pred, target, is_traction_mask):
    weights = torch.ones_like(target)
    # Give high weight to the rare "Traction" cases
    weights[is_traction_mask] = weight_for_traction
    return torch.mean(weights * (pred - target)**2)

optimizer = optim.Adam(model.parameters(), lr=0.001)
# Add a scheduler: It slows down learning as we get closer to the goal for "Fine Tuning"
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

# 5. Training Loop
epochs = 2000 # Increased for better convergence
is_traction_tensor = torch.BoolTensor(weight_train).unsqueeze(1)

print("🚀 Starting Better Training...")
for epoch in range(epochs):
    model.train()
    pred = model(X_train)
    loss = weighted_mse_loss(pred, y_train, is_traction_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    scheduler.step(loss) # Adjust LR based on loss

    if (epoch+1) % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LR: {current_lr}")

# 6. Held-out evaluation (test set the model NEVER trained on)
from sklearn.metrics import r2_score
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).numpy()
holdout_r2 = r2_score(y_test, y_test_pred)
# Binary grip/slip accuracy at the 0.1 slip threshold, on held-out data.
pred_traction = (y_test_pred.ravel() < 0.1)
true_traction = (np.asarray(y_test).ravel() < 0.1)
holdout_bin_acc = float(np.mean(pred_traction == true_traction))
print(f"Held-out R2:              {holdout_r2:.4f}")
print(f"Held-out binary accuracy: {holdout_bin_acc:.4f}")

# 7. Save (Sync Names)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_MODELS_DIR = os.path.join(_THIS_DIR, "models")
_SCALERS_DIR = os.path.join(_THIS_DIR, "scalers")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_SCALERS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(_MODELS_DIR, f"traction_model_v2_{timestamp}.pth"))
joblib.dump(scaler, os.path.join(_SCALERS_DIR, f"scaler_v2_{timestamp}.pkl"))
print("✅ Improved Model and Scaler Saved.")