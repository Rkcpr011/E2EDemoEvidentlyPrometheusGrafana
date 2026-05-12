"""
train.py
Trains a RandomForest regressor on reference data and
logs the run to MLflow.  Returns the fitted model.
"""

import mlflow
import mlflow.sklearn
from sklearn import ensemble
from data.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET, get_reference_and_current, load_raw_data


FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def train_model(reference):
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    with mlflow.start_run(run_name="bike_rf"):
        params = {"n_estimators": 50, "random_state": 0}
        mlflow.log_params(params)
        model = ensemble.RandomForestRegressor(**params)
        model.fit(reference[FEATURES], reference[TARGET])
        mlflow.sklearn.log_model(model, "model")
    return model
