import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/heart.csv")

df = df.dropna()

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df["target"] = y.values
processed_df.to_csv("data/processed_data.csv", index=False)

print("Preprocessing done. Saved to data/processed_data.csv")
