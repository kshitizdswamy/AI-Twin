import os
import sys
import joblib
import yaml

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.model_selection import train_test_split
from src.preprocessing.data_generator import generate_synthetic_activity_data
from src.preprocessing.usage_preprocess import preprocess_usage_data, prepare_feature_matrices
from src.models.usage_model import ScreenTimeUsagePredictor


def train_pipeline(config_path: str = "configs/usage_config.yaml", train_config_path: str = "configs/training_config.yaml"):
    """Full training pipeline for screen time forecasting and anomaly detection."""
    print("Starting AI Twin Model Training Pipeline...")

    # Load Configs
    with open(config_path, "r") as f:
        usage_config = yaml.safe_load(f)

    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    feature_cols = usage_config["feature_columns"]
    target_col = usage_config["target_column"]
    num_days = train_config.get("num_synthetic_days", 180)
    seed = train_config.get("random_seed", 42)

    # Step 1: Generate Dataset
    print(f"Generating {num_days} days of synthetic user activity logs...")
    raw_df = generate_synthetic_activity_data(num_days=num_days, seed=seed)

    # Step 2: Preprocess Features
    processed_df = preprocess_usage_data(raw_df, target_col=target_col)

    # Step 3: Prepare Matrices
    X_scaled, y, scaler = prepare_feature_matrices(processed_df, feature_cols, target_col)

    # Save Scaler
    models_dir = train_config.get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Step 4: Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=train_config.get("test_size", 0.2), random_state=seed
    )

    # Step 5: Train Model
    predictor = ScreenTimeUsagePredictor(
        model_type="random_forest",
        n_estimators=train_config.get("n_estimators", 100),
        random_state=seed
    )
    train_res = predictor.fit(X_train, y_train, feature_names=feature_cols)
    print(f"[SUCCESS] Model Training Completed. Training R2: {train_res['train_r2']:.4f}, MAE: {train_res['train_mae']:.2f}")

    # Step 6: Evaluate Model
    metrics = predictor.evaluate(X_test, y_test)
    print("Held-Out Test Evaluation:")
    print(f"   - R2 Score: {metrics['r2_score']:.4f}")
    print(f"   - MAE: {metrics['mae']:.2f} mins")
    print(f"   - RMSE: {metrics['rmse']:.2f} mins")
    print(f"   - Forecast Accuracy: {metrics['accuracy_pct']}%")

    # Step 7: Anomaly Detection Check
    anomalies_detected = predictor.detect_anomalies(X_test)
    print(f"[ALERT] Anomaly Detection: Flagged {sum(anomalies_detected)} anomalous days out of {len(X_test)} test samples.")

    # Step 8: Save Model Artifact
    model_save_path = train_config.get("usage_model_path", "models/usage_model.joblib")
    predictor.save(model_save_path)
    print(f"[SAVED] Model saved successfully to '{model_save_path}'.")

    return metrics


if __name__ == "__main__":
    train_pipeline()
