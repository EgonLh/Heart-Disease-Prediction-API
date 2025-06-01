from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Load model
model = joblib.load("models/model.pkl")

app = FastAPI()

# Input Schema
class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# Create logs dir if not exists
os.makedirs("logs", exist_ok=True)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds')
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions made')
FAILED_PREDICTIONS = Counter('failed_predictions_total', 'Total number of failed predictions')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    resp_time = time.time() - start_time

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(resp_time)

    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
@app.post("/predict")
def predict(data: HeartData):
    try:
        features = pd.DataFrame([{
            "age": data.age,
            "sex": data.sex,
            "cp": data.cp,
            "trestbps": data.trestbps,
            "chol": data.chol,
            "fbs": data.fbs,
            "restecg": data.restecg,
            "thalach": data.thalach,
            "exang": data.exang,
            "oldpeak": data.oldpeak,
            "slope": data.slope,
            "ca": data.ca,
            "thal": data.thal
        }])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]

        # Convert data to dict
        input_data = data.dict()
        input_data.update({
            "prediction": int(prediction),
            "probability": float(probability),
            "timestamp": datetime.now().isoformat()
        })

        # Append to log CSV
        log_df = pd.DataFrame([input_data])
        log_df.to_csv("logs/predictions_log.csv", mode="a", header=not os.path.exists("logs/predictions_log.csv"), index=False)

        PREDICTIONS_TOTAL.inc()

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }

    except Exception as e:
        FAILED_PREDICTIONS.inc()
        return {"error": str(e)}
