import pandas as pd
import joblib
from sklearn.metrics import classification_report
import json

# Load model and data
model = joblib.load("models/model.pkl")
df = pd.read_csv("data/processed_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Predict
y_pred = model.predict(X)

# Classification report
report = classification_report(y, y_pred, output_dict=True)

# Save report to JSON
with open("reports/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Evaluation report saved to reports/evaluation_report.json")
