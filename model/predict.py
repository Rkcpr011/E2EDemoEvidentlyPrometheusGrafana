"""
predict.py
Adds a 'prediction' column to reference and current DataFrames.
"""
from data.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def add_predictions(model, reference, current):
    ref = reference.copy()
    cur = current.copy()
    ref["prediction"] = model.predict(ref[FEATURES])
    cur["prediction"] = model.predict(cur[FEATURES])
    return ref, cur
