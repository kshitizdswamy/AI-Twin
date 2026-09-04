import unittest
import numpy as np
import pandas as pd
from src.preprocessing.data_generator import generate_synthetic_activity_data
from src.preprocessing.usage_preprocess import preprocess_usage_data, prepare_feature_matrices
from src.models.usage_model import ScreenTimeUsagePredictor


class TestUsageModel(unittest.TestCase):

    def test_synthetic_data_generation(self):
        df = generate_synthetic_activity_data(num_days=30)
        self.assertEqual(len(df), 30)
        self.assertIn("screen_time_mins", df.columns)
        self.assertIn("social_media_mins", df.columns)

    def test_usage_preprocessing(self):
        df_raw = generate_synthetic_activity_data(num_days=40)
        df_proc = preprocess_usage_data(df_raw)
        self.assertTrue("next_day_screen_time" in df_proc.columns)
        self.assertTrue("prev_day_screen_time" in df_proc.columns)
        self.assertTrue("rolling_7d_avg" in df_proc.columns)

    def test_predictor_train_predict(self):
        df_raw = generate_synthetic_activity_data(num_days=60)
        df_proc = preprocess_usage_data(df_raw)
        cols = [
            "screen_time_mins", "social_media_mins", "productivity_mins",
            "entertainment_mins", "gaming_mins", "notifications", "unlocks",
            "battery_drain_pct", "day_of_week", "is_weekend", "prev_day_screen_time", "rolling_7d_avg"
        ]
        X, y, scaler = prepare_feature_matrices(df_proc, cols)

        predictor = ScreenTimeUsagePredictor(n_estimators=10, random_state=42)
        res = predictor.fit(X, y)
        self.assertTrue(predictor.is_trained)

        preds = predictor.predict(X[:5])
        self.assertEqual(len(preds), 5)

        anomalies = predictor.detect_anomalies(X[:5])
        self.assertEqual(len(anomalies), 5)


if __name__ == "__main__":
    unittest.main()
