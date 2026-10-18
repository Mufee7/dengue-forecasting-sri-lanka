# Serving artifact provenance

The demonstration uses `models/xgb_climate_model.ubj` and `models/inference_schema.json`. The original three `.pkl` files are retained as historical project artifacts; the app does not load them.

## Native model

The original XGBoost pickle contains a single UBJSON memory snapshot. A one-time, SHA-256-bound migration reads pickle **opcodes as data** using `pickletools`; it never calls `pickle.load`, `pickle.loads`, or `joblib.load`. It extracts the existing XGBoost-produced `Model` document from that known snapshot, without modifying the tree data. The model's stored version is 3.3.0.

`scripts/export_legacy_model.py` records and checks the exact source artifact hashes before exporting. It is intentionally specific to these artifacts, not a general converter. Normal app startup does not need to run the migration.

Native JSON/UBJSON model storage is XGBoost's documented format for model IO. See [XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).

## Preprocessing

The inspected saved scikit-learn 1.6.1 `ColumnTransformer` uses a dense `OneHotEncoder` for district, with no dropped category, followed by passthrough numeric features. There is no scaling. The recreated transform is recorded explicitly in the schema:

1. District indicators in order: Colombo, Gampaha, Jaffna, Kalutara, Kandy.
2. The remaining 21 numeric features in the saved feature-list order.
3. Seasonal values use `sin(2π × week / 52)` and `cos(2π × week / 52)`, preserving the original app's week-53 behavior.

The runtime checks the model checksum and expected feature count. Unsupported districts are rejected instead of silently using an all-zero district encoding.

## Verification boundary

The golden example comes from the committed processed CSV: Kandy, observation date 13 December 2025. Notebook section 6.2 records a rounded forecast of **77.44**, with 86 actual cases in the following week. The test uses that recorded result as its independent reference.

This verifies a saved-model serving path, not an end-to-end retraining reproduction. The original notebook is a Colab-oriented snapshot with absolute Drive paths, precomputed comparison results, and cells that write application files. Do not run its file-generation cells over the maintained app. Full training reconstruction, data-source refresh, and reevaluation of all reported metrics remain separate work.
