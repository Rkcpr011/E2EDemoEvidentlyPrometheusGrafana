"""
Bicycle Demand Monitoring — Project Scaffold
this is a bolierplate script to create the project structure and files for the demo.
Run: python create_project.py
this will automatically create the folders and files with the code snippets from the monitoring demo.
"""

import os

# ── folder structure ──────────────────────────────────────────────
FOLDERS = [
    "bicycle_monitoring/data",
    "bicycle_monitoring/model",
    "bicycle_monitoring/monitoring",
    "bicycle_monitoring/api",
    "bicycle_monitoring/grafana/provisioning/datasources",
    "bicycle_monitoring/grafana/provisioning/dashboards",
    "bicycle_monitoring/reports",
    "bicycle_monitoring/mlruns",
]

# ── file templates ────────────────────────────────────────────────
FILES = {

# ── data_loader.py ────────────────────────────────────────────────
"bicycle_monitoring/data/data_loader.py": '''\

"""
data_loader.py
Downloads the UCI Bike Sharing dataset and returns
reference (train) and current (production) DataFrames.
"""

import io
import zipfile
from datetime import datetime, time

import pandas as pd
import requests


RAW_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

NUMERICAL_FEATURES   = ["temp", "atemp", "hum", "windspeed", "hr", "weekday"]
CATEGORICAL_FEATURES = ["season", "holiday", "workingday"]
TARGET               = "cnt"


def load_raw_data() -> pd.DataFrame:
    """Download and parse the hourly bike-sharing CSV."""
    content = requests.get(RAW_URL, timeout=30).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        df = pd.read_csv(
            arc.open("hour.csv"),
            header=0,
            sep=",",
            parse_dates=["dteday"],
            index_col="dteday",
        )
    # combine date + hour into a single datetime index
    df.index = df.apply(
        lambda row: datetime.combine(row.name, time(hour=int(row["hr"]))), axis=1
    )
    return df


def get_reference_and_current(df: pd.DataFrame):
    """
    Split raw data into reference and current windows.
    Reference : Jan 1–28  2011  (model training baseline)
    Current   : Jan 29 – Feb 28 2011  (simulated production)
    """
    reference = df.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"]
    current   = df.loc["2011-01-29 00:00:00":"2011-02-28 23:00:00"]
    return reference, current


if __name__ == "__main__":
    print("Downloading dataset …")
    df = load_raw_data()
    ref, cur = get_reference_and_current(df)
    print(f"Reference rows : {len(ref)}")
    print(f"Current rows   : {len(cur)}")
    print("data_loader OK")
''',

# ── train.py ──────────────────────────────────────────────────────
"bicycle_monitoring/model/train.py": '''\
"""
train.py
Trains a RandomForest regressor on reference data and
logs the run to MLflow.  Returns the fitted model.
"""

import mlflow
import mlflow.sklearn
from sklearn import ensemble

from data.data_loader import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    get_reference_and_current,
    load_raw_data,
)

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def train_model(reference):
    """Fit RandomForestRegressor and log to MLflow."""
    with mlflow.start_run(run_name="bike_rf"):
        params = {"n_estimators": 50, "random_state": 0}
        mlflow.log_params(params)

        model = ensemble.RandomForestRegressor(**params)
        model.fit(reference[FEATURES], reference[TARGET])

        mlflow.sklearn.log_model(model, "model")
        print("Model logged to MLflow.")
    return model


if __name__ == "__main__":
    df  = load_raw_data()
    ref, _ = get_reference_and_current(df)
    train_model(ref)
''',

# ── predict.py ────────────────────────────────────────────────────
"bicycle_monitoring/model/predict.py": '''\
"""
predict.py
Adds a 'prediction' column to reference and current DataFrames.
"""

from data.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def add_predictions(model, reference, current):
    """Return copies of both DataFrames with predictions attached."""
    ref = reference.copy()
    cur = current.copy()
    ref["prediction"] = model.predict(ref[FEATURES])
    cur["prediction"] = model.predict(cur[FEATURES])
    return ref, cur
''',

# ── monitor.py ────────────────────────────────────────────────────
"bicycle_monitoring/monitoring/monitor.py": '''\
"""
monitor.py
Runs Evidently reports (regression performance, target drift,
data drift) and returns structured metric dicts.
"""

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.report import Report

from data.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

PREDICTION = "prediction"

WEEKS = {
    "week1": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week2": ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
    "week3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}


def _column_mapping(include_target: bool = True) -> ColumnMapping:
    cm = ColumnMapping()
    cm.numerical_features   = NUMERICAL_FEATURES
    cm.categorical_features = CATEGORICAL_FEATURES
    if include_target:
        cm.target     = TARGET
        cm.prediction = PREDICTION
    return cm


def run_regression_report(current_slice, reference) -> dict:
    report = Report(metrics=[RegressionPreset()])
    report.run(
        current_data=current_slice,
        reference_data=reference,
        column_mapping=_column_mapping(),
    )
    result = report.as_dict()
    metrics = result["metrics"][0]["result"]
    return {
        "rmse"     : metrics["current"]["rmse"],
        "mae"      : metrics["current"]["mae"],
        "r2"       : metrics["current"]["r2_score"],
    }


def run_data_drift_report(current_slice, reference) -> dict:
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        current_data=current_slice,
        reference_data=reference,
        column_mapping=_column_mapping(include_target=False),
    )
    result = report.as_dict()
    drift  = result["metrics"][0]["result"]
    return {
        "drift_detected"      : drift["dataset_drift"],
        "drifted_feature_count": drift["number_of_drifted_columns"],
        "drift_share"         : drift["share_of_drifted_columns"],
    }


def run_all_weeks(reference, current) -> dict:
    """Run regression + data drift for each week. Returns nested dict."""
    results = {}
    for week, (start, end) in WEEKS.items():
        slice_ = current.loc[start:end]
        results[week] = {
            "regression" : run_regression_report(slice_, reference),
            "data_drift" : run_data_drift_report(slice_, reference),
        }
        print(f"{week} done — RMSE={results[week]['regression']['rmse']:.2f}")
    return results
''',

# ── metrics.py ────────────────────────────────────────────────────
"bicycle_monitoring/monitoring/metrics.py": '''\
"""
metrics.py
Defines Prometheus Gauge metrics.
Call update_metrics(results) after each monitoring run.
"""

from prometheus_client import Gauge

# Regression metrics
GAUGE_RMSE  = Gauge("bike_model_rmse",  "Root Mean Squared Error",  ["week"])
GAUGE_MAE   = Gauge("bike_model_mae",   "Mean Absolute Error",      ["week"])
GAUGE_R2    = Gauge("bike_model_r2",    "R² score",                 ["week"])

# Drift metrics
GAUGE_DRIFT         = Gauge("bike_data_drift_detected",       "1 if drift detected else 0", ["week"])
GAUGE_DRIFT_SHARE   = Gauge("bike_data_drift_share",          "Share of drifted features",  ["week"])
GAUGE_DRIFTED_COUNT = Gauge("bike_data_drifted_feature_count","Number of drifted features", ["week"])


def update_metrics(results: dict):
    """
    results: output of monitor.run_all_weeks()
    {
      "week1": {"regression": {...}, "data_drift": {...}},
      ...
    }
    """
    for week, data in results.items():
        reg   = data["regression"]
        drift = data["data_drift"]

        GAUGE_RMSE.labels(week=week).set(reg["rmse"])
        GAUGE_MAE.labels(week=week).set(reg["mae"])
        GAUGE_R2.labels(week=week).set(reg["r2"])

        GAUGE_DRIFT.labels(week=week).set(int(drift["drift_detected"]))
        GAUGE_DRIFT_SHARE.labels(week=week).set(drift["drift_share"])
        GAUGE_DRIFTED_COUNT.labels(week=week).set(drift["drifted_feature_count"])

    print("Prometheus metrics updated.")
''',

# ── api/main.py ───────────────────────────────────────────────────
"bicycle_monitoring/api/main.py": '''\
"""
api/main.py
FastAPI app exposing:
  GET /run-monitor  — triggers full monitoring pipeline
  GET /metrics      — Prometheus text exposition
  GET /health       — liveness probe
"""

from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# local imports (run from bicycle_monitoring/ root)
from data.data_loader import load_raw_data, get_reference_and_current
from model.train import train_model
from model.predict import add_predictions
from monitoring.monitor import run_all_weeks
from monitoring.metrics import update_metrics

app = FastAPI(title="Bike Demand Monitor")

# load + train once at startup
_df          = load_raw_data()
_reference_raw, _current_raw = get_reference_and_current(_df)
_model       = train_model(_reference_raw)
_reference, _current = add_predictions(_model, _reference_raw, _current_raw)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/run-monitor")
def run_monitor():
    """Run Evidently reports and push results to Prometheus gauges."""
    results = run_all_weeks(_reference, _current)
    update_metrics(results)
    return {"status": "done", "results": results}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
''',

# ── prometheus.yml ────────────────────────────────────────────────
"bicycle_monitoring/prometheus.yml": '''\
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "bike_monitor"
    static_configs:
      - targets: ["fastapi:8000"]   # service name in docker-compose
''',

# ── docker-compose.yml ────────────────────────────────────────────
"bicycle_monitoring/docker-compose.yml": '''\
version: "3.9"

services:

  fastapi:
    build: .
    container_name: bike_fastapi
    ports:
      - "8000:8000"
    volumes:
      - ./mlruns:/app/mlruns
      - ./reports:/app/reports
    environment:
      - PYTHONPATH=/app

  prometheus:
    image: prom/prometheus:latest
    container_name: bike_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    container_name: bike_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning

depends_on_note: |
  prometheus depends on fastapi being up.
  grafana depends on prometheus being up.
  For local dev, start manually or add healthchecks.
''',

# ── Dockerfile ────────────────────────────────────────────────────
"bicycle_monitoring/Dockerfile": '''\
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
''',

# ── requirements.txt ─────────────────────────────────────────────
"bicycle_monitoring/requirements.txt": '''\
evidently==0.4.30
fastapi==0.111.0
uvicorn==0.29.0
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
mlflow==2.13.0
prometheus-client==0.20.0
requests==2.31.0
''',

# ── grafana datasource ────────────────────────────────────────────
"bicycle_monitoring/grafana/provisioning/datasources/prometheus.yml": '''\
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    access: proxy
    isDefault: true
''',

# ── grafana dashboard provisioning ───────────────────────────────
"bicycle_monitoring/grafana/provisioning/dashboards/dashboard.yml": '''\
apiVersion: 1

providers:
  - name: "default"
    folder: ""
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
''',

# ── __init__ files ────────────────────────────────────────────────
"bicycle_monitoring/data/__init__.py":       "",
"bicycle_monitoring/model/__init__.py":      "",
"bicycle_monitoring/monitoring/__init__.py": "",
"bicycle_monitoring/api/__init__.py":        "",

# ── .env ─────────────────────────────────────────────────────────
"bicycle_monitoring/.env": '''\
MLFLOW_TRACKING_URI=./mlruns
''',

# ── .gitignore ────────────────────────────────────────────────────
"bicycle_monitoring/.gitignore": '''\
__pycache__/
*.pyc
.env
mlruns/
reports/
.DS_Store
''',

}  # end FILES


# ── scaffold runner ───────────────────────────────────────────────
def create_project():
    print("\n🚲  Creating Bicycle Demand Monitoring project …\n")

    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"  📁  {folder}/")
    print()

    for filepath, content in FILES.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  📄  {filepath}")

    print("\n✅  Project created successfully!\n")
    print("Next steps:")
    print("  cd bicycle_monitoring")
    print("  pip install -r requirements.txt")
    print("  uvicorn api.main:app --reload   # local run")
    print("  docker-compose up --build       # full stack\n")


if __name__ == "__main__":
    create_project()
