"""Validated inference with a native XGBoost model and an explicit input schema."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "models/inference_schema.json").read_text(encoding="utf-8"))
MODEL_PATH = ROOT / "models/xgb_climate_model.ubj"
DISTRICTS = tuple(SCHEMA["district_categories"])
NUMERIC_FEATURES = tuple(SCHEMA["raw_features"][1:])
INPUT_FEATURES = tuple(name for name in NUMERIC_FEATURES if name not in {"week_sin", "week_cos"})


def load_example():
    """Return a historical observation, never live surveillance data."""
    return json.loads((ROOT / "data/demo/kandy_2025_12_13.json").read_text(encoding="utf-8"))


def _number(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def encode_record(record):
    """One-hot district followed by numeric features in the training order."""
    missing = [key for key in ("district", "week", *INPUT_FEATURES) if key not in record]
    if missing:
        raise ValueError("Missing inputs: " + ", ".join(missing))
    if record["district"] not in DISTRICTS:
        raise ValueError("Choose one of the five supported districts.")
    week = _number("week", record["week"])
    if not week.is_integer() or not 1 <= week <= 53:
        raise ValueError("Epidemiological week must be an integer from 1 to 53.")
    values = {}
    for name in INPUT_FEATURES:
        value = _number(name, record[name])
        if name.startswith("cases") and (value < 0 or not value.is_integer()):
            raise ValueError(f"{name} must be a non-negative whole number.")
        if name.startswith("rainfall") and value < 0:
            raise ValueError(f"{name} must be non-negative.")
        if name.startswith("humidity") and not 0 <= value <= 100:
            raise ValueError(f"{name} must be between 0 and 100 percent.")
        values[name] = value
    # Preserve the original training convention, including week 53.
    angle = 2 * math.pi * week / SCHEMA["seasonal_period"]
    values.update(week_sin=math.sin(angle), week_cos=math.cos(angle))
    one_hot = [float(record["district"] == district) for district in DISTRICTS]
    encoded = np.asarray([one_hot + [values[name] for name in NUMERIC_FEATURES]], dtype=np.float64)
    if not np.isfinite(encoded).all():
        raise ValueError("Inputs must remain finite after conversion.")
    return encoded


@lru_cache(maxsize=1)
def load_model():
    """Load the recorded native artifact without pickle/joblib deserialization."""
    data = MODEL_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != SCHEMA["native_model_sha256"]:
        raise ValueError("The model file does not match its recorded schema.")
    model = xgb.Booster(params={"nthread": 1})
    model.load_model(bytearray(data))
    if model.num_features() != len(DISTRICTS) + len(NUMERIC_FEATURES):
        raise ValueError("The model and input schema have different feature counts.")
    return model


def predict_next_week(record):
    inputs = encode_record(record)
    prediction = float(load_model().predict(xgb.DMatrix(inputs))[0])
    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite forecast.")
    return max(0.0, prediction)
