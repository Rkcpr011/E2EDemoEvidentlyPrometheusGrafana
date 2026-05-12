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
NUMERICAL_FEATURES = ["temp", "atemp", "hum", "windspeed", "hr", "weekday"]
CATEGORICAL_FEATURES = ["season", "holiday", "workingday"]
TARGET = "cnt"

def load_raw_data():
    content = requests.get(RAW_URL, timeout=30).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        df = pd.read_csv(arc.open("hour.csv"), header=0, sep=",", parse_dates=["dteday"], index_col="dteday")
    df.index = df.apply(lambda row: datetime.combine(row.name, time(hour=int(row["hr"]))), axis=1)
    return df

def get_reference_and_current(df):
    reference = df.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"]
    current = df.loc["2011-01-29 00:00:00":"2011-02-28 23:00:00"]
    return reference, current