"""
metrics.py
Defines Prometheus Gauge metrics.
Call update_metrics(results) after each monitoring run.
"""
from prometheus_client import Gauge

GAUGE_RMSE = Gauge("bike_model_rmse", "RMSE", ["week"])
GAUGE_MAE = Gauge("bike_model_mae", "MAE", ["week"])
GAUGE_R2 = Gauge("bike_model_r2", "R2", ["week"])
GAUGE_DRIFT = Gauge("bike_data_drift_detected", "Drift detected", ["week"])
GAUGE_DRIFT_SHARE = Gauge("bike_data_drift_share", "Drift share", ["week"])
GAUGE_DRIFTED_COUNT = Gauge("bike_data_drifted_feature_count", "Drifted features", ["week"])

def update_metrics(results):
    for week, data in results.items():
        reg = data["regression"]
        drift = data["data_drift"]
        GAUGE_RMSE.labels(week=week).set(reg["rmse"])
        GAUGE_MAE.labels(week=week).set(reg["mae"])
        GAUGE_R2.labels(week=week).set(reg["r2"])
        GAUGE_DRIFT.labels(week=week).set(int(drift["drift_detected"]))
        GAUGE_DRIFT_SHARE.labels(week=week).set(drift["drift_share"])
        GAUGE_DRIFTED_COUNT.labels(week=week).set(drift["drifted_feature_count"])
    print("Prometheus metrics updated.")