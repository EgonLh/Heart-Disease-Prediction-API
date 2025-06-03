import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

import os
reference = pd.read_csv("data/heart.csv")
print("Reference data loaded with shape:", reference.shape)

current = pd.read_csv("logs/predictions_log.csv")
print("Current data loaded with shape:", current.shape)

print("Reference columns:", reference.columns.tolist())
print("Current columns:", current.columns.tolist())
features = reference.columns.drop("target")
reference = reference[features]
current = current[features]

report = Report(metrics=[
    DataDriftPreset()
])

report.run(reference_data=reference, current_data=current)

os.makedirs("reports", exist_ok=True)
report.save_html("reports/data_drift_report.html")
