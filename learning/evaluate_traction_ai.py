import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report
import joblib
import glob
import os

# 1. ARCHITECTURE (The "Sequential" Version - Matches v2 Training)
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
    def forward(self, x):
        return self.net(x)

# 2. Setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_THIS_DIR, 'models')
SCALERS_DIR = os.path.join(_THIS_DIR, 'scalers')
PLOTS_DIR = os.path.join(_THIS_DIR, 'plots')
DATA_DIR = os.environ.get("NEUROTRACTION_DATA", os.path.join(_THIS_DIR, '..', 'data'))
os.makedirs(PLOTS_DIR, exist_ok=True)

# 3. Auto-select latest assets (Looking for v2 specifically)
try:
    # This architecture matches the v2 training script — load a v2 model/scaler.
    LATEST_MODEL = max(glob.glob(os.path.join(MODELS_DIR, "*v2*.pth")), key=os.path.getctime)
    LATEST_SCALER = max(glob.glob(os.path.join(SCALERS_DIR, "*v2*.pkl")), key=os.path.getctime)
    
    print(f"🚀 Evaluating Model: {os.path.basename(LATEST_MODEL)}")
    print(f"📏 Using Scaler:     {os.path.basename(LATEST_SCALER)}")
except ValueError:
    print("❌ Error: No models or scalers found!")
    exit()

# 4. Load Model and Data
model = TractionNet()
model.load_state_dict(torch.load(LATEST_MODEL))
model.eval()
scaler = joblib.load(LATEST_SCALER)

# 5. Reproduce the SAME per-recording strided hold-out used in training
#    (every 5th row by time order within each recording -> test), and evaluate
#    ONLY on that test partition. The scaler was fit on the training partition,
#    so applying it to the test rows here is leakage-free.
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
frames = []
for rec_id, f in enumerate(csv_files):
    d = pd.read_csv(f)
    d.columns = d.columns.str.strip()
    d = d.loc[:, ~d.columns.duplicated()].dropna().reset_index(drop=True)
    d["_rec"] = rec_id
    d["_order"] = np.arange(len(d))
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

test_mask = np.zeros(len(df), dtype=bool)
for rec_id, g in df.groupby("_rec"):
    ordered = g.sort_values("_order").index
    test_mask[ordered[4::5]] = True
test_df = df[test_mask]

X_raw = test_df[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'v_enc']].values
y_true = test_df['slip_ratio'].values

X_scaled = scaler.transform(X_raw)
X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    y_pred = model(X_tensor).numpy().flatten()

# 6. Metrics & Printing (held-out test set only)
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
r2 = r2_score(y_true, y_pred)

print(f"\n--- Held-out Test Report (v2 Weighted, 20% split) ---")
print(f"Test samples: {len(y_true)}")
print(f"RMSE (Error): {rmse:.4f}")
print(f"R2 Score:     {r2:.4f}")

# Binary grip/slip threshold — MUST match training definition (slip_ratio < 0.1 = traction).
SLIP_THRESHOLD = 0.1
y_true_slip = y_true >= SLIP_THRESHOLD   # True = slipping
y_pred_slip = y_pred >= SLIP_THRESHOLD
conf_matrix = confusion_matrix(y_true_slip, y_pred_slip)

print(f"\n--- Safety Detection (Binary, threshold={SLIP_THRESHOLD}) ---")
print(classification_report(y_true_slip, y_pred_slip, target_names=["Traction", "SLIP"]))

# 7. Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(y_true, y_pred, alpha=0.1, color='teal')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title(f'Prediction Correlation (R²: {r2:.2f})')

plt.subplot(2, 2, 2)
plt.scatter(X_raw[:, 6], y_true - y_pred, alpha=0.3, color='orange')  # X_raw col 6 = v_enc (test set)
plt.axhline(0, color='black')
plt.title('Error vs. Robot Speed (held-out)')

plt.subplot(2, 2, 3)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Grip', 'Slip'], yticklabels=['Grip', 'Slip'])
plt.title('v2 Weighted Confusion Matrix')

plt.subplot(2, 2, 4)
plt.hist(y_true - y_pred, bins=50, color='purple', alpha=0.7)
plt.title('Error Histogram')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "v2_evaluation_report.png"))
print(f"\n✅ Balanced Report Saved: plots/v2_evaluation_report.png")