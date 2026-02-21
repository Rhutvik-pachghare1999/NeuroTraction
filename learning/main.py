import pandas as pd
import glob
import os
import numpy as np

# =========================
# 1. LOAD ALL DATASETS
# =========================

data_dir = "/home/rhutvik/portfolio_projects/ros2/data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

if not csv_files:
    raise FileNotFoundError("No dataset files found.")

df_list = []

for file in csv_files:
    temp_df = pd.read_csv(file)
    temp_df.columns = temp_df.columns.str.strip()
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# =========================
# 2. CLEAN DATA
# =========================

df = df.loc[:, ~df.columns.duplicated()]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Keep only valid slip ratios
df = df[df['slip_ratio'] <= 1.0]
df = df[df['slip_ratio'] >= 0.0]

print(f"Loaded {len(csv_files)} files.")
print(f"Total clean rows: {len(df)}")

# =========================
# 3. CHECK IF REGRESSION OR CLASSIFICATION
# =========================

unique_values = df['slip_ratio'].nunique()
print(f"\nUnique slip_ratio values: {unique_values}")

# If too many unique values → it's regression
if unique_values > 20:
    print("\nSlip ratio is continuous. Converting to classes...")

    # Define bins (edit if needed)
    bins = [0.0, 0.1, 0.4, 1.0]
    labels = [0, 1, 2]  # 0=No slip, 1=Medium, 2=High

    df['slip_class'] = pd.cut(
        df['slip_ratio'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

else:
    print("\nSlip ratio already appears categorical.")
    df['slip_class'] = df['slip_ratio']

# =========================
# 4. CLASS DISTRIBUTION
# =========================

print("\nNumber of classes:", df['slip_class'].nunique())

print("\nClass counts:")
print(df['slip_class'].value_counts().sort_index())

print("\nClass distribution (%):")
print((df['slip_class'].value_counts(normalize=True) * 100).sort_index())

# =========================
# 5. BASIC WARNING CHECK
# =========================

class_percent = df['slip_class'].value_counts(normalize=True) * 100

if class_percent.max() > 70:
    print("\n⚠ WARNING: Severe class imbalance detected.")
    print("Your model will likely predict the dominant class.")
