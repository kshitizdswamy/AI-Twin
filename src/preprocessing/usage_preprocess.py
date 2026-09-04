from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_usage_data(
    df: pd.DataFrame,
    target_col: str = "next_day_screen_time",
    anomaly_threshold_std: float = 1.8
) -> pd.DataFrame:
    """
    Process raw activity logs dataframe to generate lag features,
    rolling statistics, ratios, anomaly labels, and future target values.
    """
    df = df.copy()

    # Sort by date if available
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Calculate lag features
    df["prev_day_screen_time"] = df["screen_time_mins"].shift(1)
    df["prev_day_social"] = df["social_media_mins"].shift(1)
    df["prev_day_notifications"] = df["notifications"].shift(1)

    # Rolling window stats
    df["rolling_3d_avg"] = df["screen_time_mins"].rolling(window=3, min_periods=1).mean()
    df["rolling_7d_avg"] = df["screen_time_mins"].rolling(window=7, min_periods=1).mean()
    df["rolling_7d_std"] = df["screen_time_mins"].rolling(window=7, min_periods=1).std().fillna(0)

    # Ratio & change features
    df["prod_ratio"] = df["productivity_mins"] / (df["screen_time_mins"] + 1.0)
    df["social_ratio"] = df["social_media_mins"] / (df["screen_time_mins"] + 1.0)
    df["screen_time_change"] = (df["screen_time_mins"] - df["prev_day_screen_time"]).fillna(0)

    # Target: Next day's screen time
    df[target_col] = df["screen_time_mins"].shift(-1)

    # Statistical Anomaly Detection flag based on rolling std
    mean_val = df["screen_time_mins"].mean()
    std_val = df["screen_time_mins"].std()
    df["is_anomaly_stat"] = (df["screen_time_mins"] > (mean_val + anomaly_threshold_std * std_val)).astype(int)

    # Drop rows with NaN targets/lags
    df_clean = df.dropna(subset=[target_col, "prev_day_screen_time"]).reset_index(drop=True)
    return df_clean


def prepare_feature_matrices(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "next_day_screen_time"
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Extract X (feature matrix) and y (target vector) and apply standard scaling.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
