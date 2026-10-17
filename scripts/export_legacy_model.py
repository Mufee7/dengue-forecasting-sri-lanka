"""Extract only XGBoost's UBJSON model data; never unpickle Python objects.

This is a one-time, hash-bound migration for the repository's known snapshot.
It is deliberately not a general pickle converter.
"""
import csv
import hashlib
import json
from pathlib import Path
import pickletools

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "models/xgb_climate_model.pkl"
expected = "a2b9fad4434e7ec4a34ceb8657c11ac1ce580651c98206bfe4af489d923ab68f"
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected, "Unexpected source artifact"
buffers = [arg for op, arg, _ in pickletools.genops(raw)
           if op.name == "BINBYTES" and len(arg) > 1000]
assert len(buffers) == 1
snapshot = buffers[0]
marker = b"L" + (5).to_bytes(8, "big", signed=True) + b"Model"
assert snapshot.count(marker) == 1
model_data = snapshot[snapshot.index(marker) + len(marker):-1]
assert model_data.startswith(b"{L" + (7).to_bytes(8, "big") + b"learner")
assert snapshot.endswith(b"}}")
(ROOT / "models/xgb_climate_model.ubj").write_bytes(model_data)

feature_source = ROOT / "models/model_b_features.pkl"
assert hashlib.sha256(feature_source.read_bytes()).hexdigest() == "e70d7cbb671920610c15c4d497a72faac1d95870c2435ae1d4e6b92c7a6ec857"
assert hashlib.sha256((ROOT / "models/climate_preprocessor.pkl").read_bytes()).hexdigest() == "64311ffef31f84b928df637fca649af3c6950a722da78a442edf8724c1663b67", "Unexpected preprocessing artifact"
feature_ops = list(pickletools.genops(feature_source.read_bytes()))
assert all(op.name in {"PROTO", "FRAME", "EMPTY_LIST", "MEMOIZE", "MARK",
                      "SHORT_BINUNICODE", "APPENDS", "STOP"}
           for op, _, _ in feature_ops)
features = [arg for op, arg, _ in feature_ops if op.name == "SHORT_BINUNICODE"]
assert len(features) == 22 and features[0] == "district"
schema = {
    "raw_features": features,
    "district_categories": ["Colombo", "Gampaha", "Jaffna", "Kalutara", "Kandy"],
    "transformation": "district one-hot first, then numeric raw_features in order",
    "seasonal_period": 52,
    "source_artifacts_sha256": {
        "xgb_climate_model.pkl": expected,
        "model_b_features.pkl": hashlib.sha256(feature_source.read_bytes()).hexdigest(),
        "climate_preprocessor.pkl": "64311ffef31f84b928df637fca649af3c6950a722da78a442edf8724c1663b67"
    },
    "native_model_sha256": hashlib.sha256(model_data).hexdigest(),
    "preprocessing_note": "Recreated from the inspected saved ColumnTransformer: OneHotEncoder(drop=None, sparse_output=False), then remainder=passthrough. No scaling. Legacy objects were not executed."
}
(ROOT / "models/inference_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
with (ROOT / "data/processed/dengue_forecasting_features.csv").open(newline="") as stream:
    row = next(row for row in csv.DictReader(stream)
               if row["district"] == "Kandy" and row["start.date"] == "2025-12-13")
sample = {"district": row["district"], "week": int(row["week"])}
sample.update({name: float(row[name]) for name in features[1:] if name not in {"week_sin", "week_cos"}})
sample.update({"observation_date": row["start.date"], "actual_next_week_cases": float(row["cases_next_week"]), "notebook_prediction_rounded": 77.44})
(ROOT / "data/demo").mkdir(exist_ok=True)
(ROOT / "data/demo/kandy_2025_12_13.json").write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"native_bytes": len(model_data), "raw_features": len(features), "example": sample["observation_date"]}))
