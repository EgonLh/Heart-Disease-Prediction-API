import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

import os

# Load training data as reference
reference = pd.read_csv("data/heart.csv")
print("Reference data loaded with shape:", reference.shape)

# Load recent logged predictions as current data
current = pd.read_csv("logs/predictions_log.csv")
print("Current data loaded with shape:", current.shape)

# Check columns (debug)
print("Reference columns:", reference.columns.tolist())
print("Current columns:", current.columns.tolist())

# Align columns, excluding 'target' if missing from current
features = reference.columns.drop("target")
reference = reference[features]
current = current[features]

# Create a report
report = Report(metrics=[
    DataDriftPreset()
])

report.run(reference_data=reference, current_data=current)

# Save HTML report
os.makedirs("reports", exist_ok=True)
report.save_html("reports/data_drift_report.html")
