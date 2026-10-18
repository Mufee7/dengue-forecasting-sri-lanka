import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inference import DISTRICTS, encode_record, load_example, load_model, predict_next_week


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.example = load_example()

    def test_historical_forecast_matches_saved_notebook(self):
        # Independent reference from saved notebook output, cell 6.2.
        self.assertAlmostEqual(predict_next_week(self.example), 77.44, delta=0.005)

    def test_supported_districts_have_finite_forecasts(self):
        for district in DISTRICTS:
            with self.subTest(district=district):
                prediction = predict_next_week(dict(self.example, district=district))
                self.assertTrue(math.isfinite(prediction))
                self.assertGreaterEqual(prediction, 0)

    def test_schema_matches_model_and_encoded_district(self):
        encoded = encode_record(self.example)
        self.assertEqual(encoded.shape, (1, load_model().num_features()))
        self.assertEqual(encoded[0, :5].tolist(), [0, 0, 0, 0, 1])

    def test_rejects_invalid_observations(self):
        invalid = [
            ("district", "Matara"), ("week", 0), ("week", 54), ("week", 1.5),
            ("cases", -1), ("cases_lag_1", 2.5), ("cases", True),
            ("rainfall_total", -0.1), ("humidity_mean_lag_8", 101),
            ("humidity_mean", -1), ("temperature_mean", float("inf")),
            ("cases", float("nan")), ("temperature_mean", None),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    predict_next_week(dict(self.example, **{field: value}))

    def test_missing_input_explains_problem(self):
        self.example.pop("cases_lag_4")
        with self.assertRaisesRegex(ValueError, "Missing inputs: cases_lag_4"):
            predict_next_week(self.example)

    def test_input_is_not_mutated(self):
        before = copy.deepcopy(self.example)
        predict_next_week(self.example)
        self.assertEqual(self.example, before)

    def test_week_53_preserves_training_encoding(self):
        encoded = encode_record(dict(self.example, week=53))
        self.assertAlmostEqual(encoded[0, 9], math.sin(2 * math.pi * 53 / 52))
        self.assertAlmostEqual(encoded[0, 10], math.cos(2 * math.pi * 53 / 52))


class StreamlitTests(unittest.TestCase):
    def test_historical_example_can_be_predicted_in_the_app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.metric[0].value, "77 cases")
        self.assertIn("Kandy", app.metric[0].label)


if __name__ == "__main__":
    unittest.main()
