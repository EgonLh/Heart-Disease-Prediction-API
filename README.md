# Heart Disease Prediction API

This project serves a machine learning model that predicts the likelihood of heart disease based on patient clinical data. It leverages a trained classifier, exposes a REST API with FastAPI, and logs prediction data for monitoring and versioning.

---

## Features

- **Baseline ML model** trained on the Heart Disease dataset.
- **Prediction API** built with FastAPI for serving model predictions.
- **Input validation** using Pydantic schemas.
- **Prediction logging** with timestamps, saved as CSV for audit and analysis.
- **Probability thresholding** (optional) to tune sensitivity and reduce false positives.
- **Data versioning** and pipeline automation with DVC (in progress).
- **Monitoring setup** with Prometheus and Grafana (planned).
- **Containerization** with Docker for consistent deployment (planned).
- **CI/CD integration** via GitHub Actions (optional).

---

├── app.py                 # FastAPI application serving the model
├── models/
│   └── model.pkl          # Trained model artifact
├── data/
│   └── raw_data.csv       # Original dataset
├── logs/
│   └── predictions_log.csv # Logs of predictions with timestamps
├── preprocess.py          # Data preprocessing script
├── train.py               # Model training script
├── evaluate.py            # Model evaluation script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container setup (planned)
├── dvc.yaml               # Pipeline configuration (planned)
└── README.md

This project is actively maintained with plans to improve:
Integration of DVC pipelines for reproducible training and version control.
Containerization with Docker for easier deployment.
Monitoring and alerting setup using Prometheus and Grafana dashboards.
Exploration of CI/CD automation using GitHub Actions.
Model improvements with additional data and calibration to reduce false positives.
Let me know if you want it more casual, or want to add badges, GIFs, or anything else!
