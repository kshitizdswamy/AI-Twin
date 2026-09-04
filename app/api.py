import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.chatbot import Chatbot
from src.models.sentiment_model import SentimentEmotionAnalyzer
from src.models.usage_model import ScreenTimeUsagePredictor
from src.preprocessing.data_generator import generate_synthetic_activity_data

app = FastAPI(
    title="Smart AI Twin API",
    description="REST API for AI Twin Behavioral Insights, Screen Time Forecasting, and Sentiment NLP",
    version="1.0.0"
)

chatbot = Chatbot()
nlp_analyzer = SentimentEmotionAnalyzer(use_transformers=False)
usage_predictor = None
scaler = None


@app.on_event("startup")
def load_ml_models():
    global usage_predictor, scaler
    model_path = "models/usage_model.joblib"
    scaler_path = "models/scaler.joblib"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            usage_predictor = ScreenTimeUsagePredictor()
            usage_predictor.load(model_path)
            scaler = joblib.load(scaler_path)
            print("✅ FastAPI Startup: Successfully loaded trained usage predictor & scaler.")
        except Exception as e:
            print(f"⚠️ FastAPI Startup Warning: Could not load usage predictor: {e}")
    else:
        print("ℹ️ FastAPI Startup: Trained usage model files not found yet. Run training script to enable predictions.")


# Request & Response Data Models
class UsageInput(BaseModel):
    screen_time_mins: float = Field(..., example=280.5)
    social_media_mins: float = Field(..., example=95.0)
    productivity_mins: float = Field(..., example=110.0)
    entertainment_mins: float = Field(..., example=55.0)
    gaming_mins: float = Field(..., example=20.5)
    notifications: int = Field(..., example=140)
    unlocks: int = Field(..., example=65)
    battery_drain_pct: float = Field(..., example=45.0)
    day_of_week: int = Field(..., example=2, description="0=Monday, 6=Sunday")
    is_weekend: int = Field(..., example=0)
    prev_day_screen_time: float = Field(..., example=260.0)
    rolling_7d_avg: float = Field(..., example=275.0)


class TextInput(BaseModel):
    text: str = Field(..., example="Had a very productive day working on machine learning models!")


class ChatQuery(BaseModel):
    message: str = Field(..., example="How is my screen time looking today?")
    context: Optional[Dict[str, Any]] = None


@app.get("/")
def root():
    return {
        "service": "Smart AI Twin API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": ["/health", "/predict/usage", "/analyze/text", "/chat", "/analytics"]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": usage_predictor is not None and usage_predictor.is_trained
    }


@app.post("/predict/usage")
def predict_usage(input_data: UsageInput):
    """Predict next day's screen time and detect abnormal digital usage spikes."""
    if usage_predictor is None or not usage_predictor.is_trained:
        # Fallback heuristic calculation if model not yet trained
        predicted = round(input_data.screen_time_mins * 0.95 + input_data.notifications * 0.2, 1)
        is_anomaly = 1 if input_data.screen_time_mins > 420 else 0
        return {
            "predicted_next_day_screen_time": predicted,
            "units": "minutes",
            "is_anomaly": bool(is_anomaly),
            "anomaly_risk": "High" if is_anomaly else "Normal",
            "note": "Using rule-based fallback predictor. Run python src/models/train.py to train ML model."
        }

    try:
        features = np.array([[
            input_data.screen_time_mins,
            input_data.social_media_mins,
            input_data.productivity_mins,
            input_data.entertainment_mins,
            input_data.gaming_mins,
            input_data.notifications,
            input_data.unlocks,
            input_data.battery_drain_pct,
            input_data.day_of_week,
            input_data.is_weekend,
            input_data.prev_day_screen_time,
            input_data.rolling_7d_avg
        ]])

        features_scaled = scaler.transform(features)
        prediction = float(usage_predictor.predict(features_scaled)[0])
        anomalies = usage_predictor.detect_anomalies(features_scaled)[0]

        return {
            "predicted_next_day_screen_time": round(prediction, 1),
            "units": "minutes",
            "is_anomaly": bool(anomalies == 1),
            "anomaly_risk": "High" if anomalies == 1 else "Normal"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/analyze/text")
def analyze_text(input_data: TextInput):
    """Analyze sentiment, emotion breakdown, and generate text summary."""
    try:
        result = nlp_analyzer.analyze_sentiment_and_emotion(input_data.text)
        summary = nlp_analyzer.summarize_text(input_data.text)
        result["summary"] = summary
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis error: {str(e)}")


@app.post("/chat")
def chat_endpoint(query: ChatQuery):
    """Interact with Smart AI Twin Chatbot."""
    try:
        response = chatbot.get_response(query.message, user_context=query.context)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/analytics")
def get_analytics(days: int = 30):
    """Return historical user activity trends & behavioral insights for dashboard."""
    df = generate_synthetic_activity_data(num_days=days)
    records = df.to_dict(orient="records")

    avg_screen_time = round(float(df["screen_time_mins"].mean()), 1)
    total_anomalies = int(df["is_anomaly"].sum())

    return {
        "days_analyzed": days,
        "average_daily_screen_time": avg_screen_time,
        "total_anomalies_flagged": total_anomalies,
        "recent_logs": records[-7:]
    }
