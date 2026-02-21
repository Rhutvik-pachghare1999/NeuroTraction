import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import glob
import os
import datetime

# 1. Load and Balance the Dataset
data_dir = "/home/rhutvik/portfolio_projects/ros2/data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

df = pd.concat([pd.read_csv(f).rename(columns=lambda x: x.strip()) for f in csv_files], ignore_index=True)
df = df.loc[:, ~df.columns.duplicated()].dropna()

# --- THE PRO FIX: CALCULATE WEIGHTS ---
# We want to give the "Traction" (slip < 0.1) more importance
is_traction = (df['slip_ratio'] < 0.1).values
traction_count = np.sum(is_traction)
slip_count = len(df) - traction_count
weight_for_traction = slip_count / max(traction_count, 1)

print(f"Dataset Balance: {traction_count} Traction vs {slip_count} Slip")
print(f"Applying Importance Weight of {weight_for_traction:.2f}x to Traction rows.")

X = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'v_enc']].values
y = df['slip_ratio'].values.reshape(-1, 1)

# 2. Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(
    X, y, is_traction, test_size=0.2, random_state=42
)

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

# 6. Save (Sync Names)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), f"models/traction_model_v2_{timestamp}.pth")
joblib.dump(scaler, f"scalers/scaler_v2_{timestamp}.pkl")
print("✅ Improved Model and Scaler Saved.")