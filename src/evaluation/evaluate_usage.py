import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.evaluation.metrics import calculate_classification_metrics, calculate_regression_metrics
from src.models.usage_model import ScreenTimeUsagePredictor
from src.preprocessing.data_generator import generate_synthetic_activity_data
from src.preprocessing.usage_preprocess import prepare_feature_matrices, preprocess_usage_data


def evaluate_screen_time_model(
    model_path: str = "models/usage_model.joblib",
    config_path: str = "configs/usage_config.yaml"
):
    """Run thorough evaluation of trained screen time predictor & anomaly detector."""
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please run training pipeline first.")
        return

    predictor = ScreenTimeUsagePredictor()
    predictor.load(model_path)

    # Generate evaluation test dataset
    df = generate_synthetic_activity_data(num_days=90, seed=123)
    processed = preprocess_usage_data(df)
    feature_cols = predictor.feature_names or [
        "screen_time_mins", "social_media_mins", "productivity_mins",
        "entertainment_mins", "gaming_mins", "notifications", "unlocks",
        "battery_drain_pct", "day_of_week", "is_weekend", "prev_day_screen_time", "rolling_7d_avg"
    ]

    scaler = joblib.load("models/scaler.joblib")
    X = processed[feature_cols].values
    X_scaled = scaler.transform(X)
    y = processed["next_day_screen_time"].values
    y_anomaly = processed["is_anomaly_stat"].values

    preds = predictor.predict(X_scaled)
    anomaly_preds = predictor.detect_anomalies(X_scaled)

    reg_metrics = calculate_regression_metrics(y, preds)
    cls_metrics = calculate_classification_metrics(y_anomaly, anomaly_preds)

    print("=== Usage Prediction Model Evaluation ===")
    print(f"Regression Accuracy: {reg_metrics['accuracy_pct']}% (Target: >=83%)")
    print(f"R2 Score: {reg_metrics['r2']:.4f}")
    print(f"MAE: {reg_metrics['mae']:.2f} mins")
    print("\n=== Anomaly Detection Evaluation ===")
    print(f"Accuracy: {cls_metrics['accuracy']}")
    print(f"Precision: {cls_metrics['precision']}")
    print(f"Recall: {cls_metrics['recall']}")
    print(f"F1 Score: {cls_metrics['f1_score']}")

    return reg_metrics, cls_metrics


if __name__ == "__main__":
    evaluate_screen_time_model()
