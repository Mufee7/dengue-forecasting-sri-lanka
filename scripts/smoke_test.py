"""Run a historical forecast from the repository root without starting a server."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from inference import load_example, predict_next_week

example = load_example()
print(json.dumps({
    "district": example["district"],
    "observation_date": example["observation_date"],
    "predicted_cases": predict_next_week(example),
    "actual_next_week_cases": example["actual_next_week_cases"],
    "reference_prediction_rounded": example["notebook_prediction_rounded"],
}, indent=2))
