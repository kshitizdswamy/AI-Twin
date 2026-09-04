import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ScreenTimeUsagePredictor:
    """
    ML Prediction Model to forecast screen time & detect digital wellbeing usage anomalies.
    """

    def __init__(self, model_type: str = "gradient_boosting", n_estimators: int = 150, random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state

        if model_type == "gradient_boosting":
            self.regressor = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                random_state=random_state
            )
        else:
            self.regressor = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=8,
                min_samples_split=4,
                random_state=random_state
            )

        self.anomaly_detector = IsolationForest(
            contamination=0.10,
            random_state=random_state
        )

        self.is_trained = False
        self.feature_names = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None) -> Dict[str, float]:
        """Train the screen time forecasting regressor and anomaly detector."""
        self.regressor.fit(X_train, y_train)
        self.anomaly_detector.fit(X_train)

        self.is_trained = True
        self.feature_names = feature_names or []

        y_pred = self.regressor.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)

        return {"train_r2": float(r2), "train_mae": float(mae)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast screen time predictions in minutes."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() or load() first.")
        return self.regressor.predict(X)

    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in usage behavior.
        Returns 1 for anomaly, 0 for normal behavior.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        preds = self.anomaly_detector.predict(X)
        return np.where(preds == -1, 1, 0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate performance metrics on held-out test dataset."""
        y_pred = self.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # Forecast accuracy metric percentage (1 - MAPE)
        mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))))
        accuracy_pct = round(max(0.0, (1 - mape) * 100), 2)

        return {
            "r2_score": float(r2),
            "mae": float(mae),
            "rmse": rmse,
            "mape": mape,
            "accuracy_pct": accuracy_pct
        }

    def save(self, filepath: str):
        """Serialize trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifact = {
            "regressor": self.regressor,
            "anomaly_detector": self.anomaly_detector,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "model_type": self.model_type
        }
        joblib.dump(artifact, filepath)

    def load(self, filepath: str):
        """Load trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        artifact = joblib.load(filepath)
        self.regressor = artifact["regressor"]
        self.anomaly_detector = artifact["anomaly_detector"]
        self.is_trained = artifact["is_trained"]
        self.feature_names = artifact.get("feature_names", [])
        self.model_type = artifact.get("model_type", "gradient_boosting")
