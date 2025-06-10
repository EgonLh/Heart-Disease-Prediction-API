import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os

df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/model.pkl")
os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump({
        "accuracy": acc,
        "report": report
    }, f, indent=4)
params = {
    "model_type": "RandomForest",
    "n_estimators": 100,
    "test_size": 0.2,
    "random_state": 42
}
with open("params.yaml", "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

print(f"✅ Training complete. Accuracy: {acc:.2f}")
