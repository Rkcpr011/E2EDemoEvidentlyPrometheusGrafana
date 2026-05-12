"""
monitor.py
Runs Evidently reports (regression performance, target drift,
data drift) and returns structured metric dicts.
"""
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report
from data.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

PREDICTION = "prediction"
WEEKS = {
    "week1": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week2": ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
    "week3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}

def _column_mapping(include_target=True):
    cm = ColumnMapping()
    cm.numerical_features = NUMERICAL_FEATURES
    cm.categorical_features = CATEGORICAL_FEATURES
    if include_target:
        cm.target = TARGET
        cm.prediction = PREDICTION
    return cm

def run_regression_report(current_slice, reference):
    report = Report(metrics=[RegressionPreset()])
    report.run(current_data=current_slice, reference_data=reference, column_mapping=_column_mapping())
    result = report.as_dict()
    metrics = result["metrics"][0]["result"]
    return {
        "rmse": metrics["current"]["rmse"],
        "mae": metrics["current"]["mae"],
        "r2": metrics["current"]["r2_score"],
    }

def run_data_drift_report(current_slice, reference):
    report = Report(metrics=[DataDriftPreset()])
    report.run(current_data=current_slice, reference_data=reference, column_mapping=_column_mapping(include_target=False))
    result = report.as_dict()
    drift = result["metrics"][0]["result"]
    return {
        "drift_detected": drift["dataset_drift"],
        "drifted_feature_count": drift["number_of_drifted_columns"],
        "drift_share": drift["share_of_drifted_columns"],
    }

def run_all_weeks(reference, current):
    results = {}
    for week, (start, end) in WEEKS.items():
        slice_ = current.loc[start:end]
        results[week] = {
            "regression": run_regression_report(slice_, reference),
            "data_drift": run_data_drift_report(slice_, reference),
        }
        print(f"{week} done")
    return results