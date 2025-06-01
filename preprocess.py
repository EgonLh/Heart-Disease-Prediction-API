import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load raw data
df = pd.read_csv("data/heart.csv")

# Basic cleaning (example: drop rows with missing values)
df = df.dropna()

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed data as CSV (combine features & target)
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df["target"] = y.values
processed_df.to_csv("data/processed_data.csv", index=False)

print("Preprocessing done. Saved to data/processed_data.csv")
