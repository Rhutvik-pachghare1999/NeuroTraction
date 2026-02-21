import torch                                            # core Neural Network lib (pytorch)
import torch.nn as nn                                   # Contains building blocks for layers (linrear, Relu)
import torch.optim as optim                             # optimizer (logic that turns the knobs))
import pandas as pd                                     # for reading CSV files
import numpy as np                                      # for handling numerical infinity
from sklearn.model_selection import train_test_split    # for splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler        # for scaling the data
import joblib                                           # for saving the model
import glob 
import os
import datetime

# 1. Load the Dataset
data_dir = "/home/rhutvik/portfolio_projects/ros2/data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No Dataset files found")

df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    # CRITICAL: Strip spaces from each file BEFORE combining them
    temp_df.columns = temp_df.columns.str.strip()
    df_list.append(temp_df)

# Now they will align perfectly
df = pd.concat(df_list, ignore_index=True)

# 2. Cleanup & Optimization
# Remove any row where 'v_enc' is exactly 0 if 'ax' is high (sensor glitches)
# and drop duplicate columns
df = df.loc[:, ~df.columns.duplicated()]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
# ---------------------------------------------------------

print(f"Loaded {len(csv_files)} files.")
print(f"Total clean rows: {len(df)}")

X_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'v_enc']
X = df[X_cols].values            
y = df['slip_ratio'].values.reshape(-1, 1)                              

# 2. Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test  = torch.FloatTensor(y_test)

# 3. Build the Neural Network Architecture
class TractionNet(nn.Module):
    def __init__(self):
        super(TractionNet,self).__init__()
        self.layer1 = nn.Linear(7,32)      
        self.layer2 = nn.Linear(32,16) 
        self.layer3 = nn.Linear(16,8)
        self.layer4 = nn.Linear(8,4)   
        self.output = nn.Linear(4,1)      
        self.relu = nn.ReLU()              
        self.sigmoid = nn.Sigmoid()        

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.sigmoid(self.output(x))
        return x
 
# 4. Training setup
model = TractionNet() 
criterion = nn.MSELoss()  
# Lowered learning rate to 0.0005 for better stability
optimizer = optim.Adam(model.parameters(), lr=0.0005)  

# 5 Training loop
epochs = 1000
print("Starting Training...")

for epoch in range(epochs):
    predictions = model(X_train)                    
    loss = criterion(predictions, y_train)   

    # Check if loss is nan manually to stop early if it breaks
    if torch.isnan(loss):
        print(f"Loss became NaN at epoch {epoch+1}. Check your data for errors.")
        break

    optimizer.zero_grad()                           
    loss.backward()                                 
    optimizer.step()                                

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
# 6. Save the Model
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"traction_model_{timestamp}_loss_{loss.item():.4f}.pth"
model_save_path = os.path.join("/home/rhutvik/portfolio_projects/ros2/learning/models", model_name)


scaler_name = f"scaler_{timestamp}.pkl"
scaler_save_path = os.path.join("/home/rhutvik/portfolio_projects/ros2/learning/scalers", scaler_name)
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved to {scaler_save_path}")

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

