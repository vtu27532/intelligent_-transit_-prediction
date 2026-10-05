"""Shared feature preparation and model-loading helpers for TransitPredict."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "trips.csv"
MODEL_PATH = ROOT / "data" / "delay_model.joblib"
METRICS_PATH = ROOT / "data" / "model_metrics.json"

FEATURE_COLUMNS = ["route", "stop", "hour", "day_of_week", "weather", "traffic_level", "holiday_flag"]
CATEGORICAL_COLUMNS = ["route", "stop", "weather", "traffic_level"]
NUMERIC_COLUMNS = ["hour", "day_of_week", "holiday_flag"]


def make_feature_frame(rows: list[dict]) -> pd.DataFrame:
    """Keep prediction inputs in the same shape used during model training."""
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def load_model():
    """Load the best trained scikit-learn pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run python train.py first.")
    return joblib.load(MODEL_PATH)
