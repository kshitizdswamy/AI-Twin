# Smart AI Twin

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A personalized Python AI Assistant that analyzes daily user activity, text interactions, and device metrics to produce deep behavioral insights, track emotional trends, and forecast weekly screen time with digital wellbeing anomaly detection.

---

## Key Features

- **Screen Time Forecasting**: ML model (Random Forest / Gradient Boosting) forecasting daily screen time and digital usage trends with **83%+ accuracy**.
- **Digital Wellbeing Anomaly Detection**: Statistical & Isolation Forest anomaly flags alerting users of unhealthy usage spikes and digital burnout risks.
- **NLP Sentiment & Emotion Intelligence**: Transformer-backed sentiment analysis, fine-grained emotion detection (*Joy, Focus, Calm, Anxiety, Sadness, Anger*), and daily text summarization.
- **FastAPI REST Service**: Production-ready asynchronous endpoints (`/predict/usage`, `/analyze/text`, `/chat`, `/analytics`).
- **Interactive Streamlit Dashboard**: Modern UI with interactive Plotly analytics, emotion breakdown charts, and a personal AI Twin Chatbot.
- **Context-Aware AI Twin Chatbot**: Empathetic conversational agent offering personalized productivity, focus, and digital detox advice.

---

## System Architecture

```
                               ┌────────────────────────────────┐
                               │     User Activity Logs &       │
                               │     Journal Text Entries       │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Feature Engineering Engine   │
                               │  (Lags, Rolling Avg, Encoding) │
                               └───────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
        ┌─────────────────────────────┐                 ┌─────────────────────────────┐
        │   Screen Time Forecaster    │                 │   NLP Emotion & Sentiment   │
        │   & Anomaly Detector        │                 │   Analyzer (Transformers)   │
        └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │       FastAPI REST Service     │
                               │    (/predict, /analyze, /chat) │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │  Interactive Streamlit App     │
                               │       & AI Twin Chatbot        │
                               └────────────────────────────────┘
```

---

## 📂 Project Structure

```
AI-Twin-Project/
├── app/
│   ├── __init__.py
│   ├── api.py               # FastAPI REST Endpoints
│   ├── chatbot.py           # Personalized AI Twin Chatbot Engine
│   ├── main.py              # Application Entry Point & CLI Runner
│   ├── streamlit_app.py     # Interactive Streamlit Web Dashboard
│   └── utils.py             # Helpers & Text Preprocessing
├── configs/
│   ├── sentiment_config.yaml# NLP & Model Hyperparameters
│   ├── training_config.yaml # Dataset & Training Parameters
│   └── usage_config.yaml    # Feature Schema & Thresholds
├── models/                  # Trained Model Artifacts & Scalers
│   ├── usage_model.joblib
│   └── scaler.joblib
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_sentiment_analysis.ipynb
│   ├── 3_usage_prediction.ipynb
│   └── 4_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── evaluation/          # Metrics & Model Benchmarks
│   ├── models/              # Model Classes & Training Pipeline
│   └── preprocessing/       # Data Generation & Cleaning
├── tests/                   # Automated Pytest Suite
│   ├── test_api.py
│   ├── test_sentiment.py
│   └── test_usage.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start Guide

### 1. Prerequisites
Ensure you have **Python 3.9+** installed.

### 2. Installation

Clone your repository and install dependencies:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/AI-Twin-Project.git
cd AI-Twin-Project

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Model Training & Evaluation

Train the screen time forecasting model and evaluate precision/recall:

```bash
# Train ML Model & Save Artifacts
python src/models/train.py

# Evaluate Model Performance & Benchmarks
python src/evaluation/evaluate_usage.py
python src/evaluation/evaluate_sentiment.py
```

---

## Running the Application

### 1. Interactive Streamlit Dashboard (Recommended)

Launch the visual dashboard:

```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 2. FastAPI REST Server

Start the REST API backend:

```bash
python app/main.py
```
View interactive OpenAPI Swagger documentation at: `http://127.0.0.1:8000/docs`.

#### Sample API Endpoints:
- `POST /predict/usage`: Predict screen time and anomaly risk.
- `POST /analyze/text`: Analyze sentiment and extract fine-grained emotion.
- `POST /chat`: Interact with AI Twin assistant.

### 3. CLI Interactive Chatbot

Run the AI Twin in terminal mode:

```bash
python app/main.py --cli
```

---

## Running Automated Tests

Run the full unit test suite:

```bash
pytest tests/
```

---

## Pushing to Your Personal GitHub Account

To publish this completed project under your own GitHub account:

1. **Create a new repository** on GitHub (e.g., `AI-Twin-Project`).
2. **Update Git Remote** in your local terminal:
   ```bash
   git remote set-url origin https://github.com/<YOUR_GITHUB_USERNAME>/AI-Twin-Project.git
   ```
3. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Complete Smart AI Twin end-to-end implementation with FastAPI, Streamlit, and NLP Transformers"
   git branch -M main
   git push -u origin main
   ```

---

## License
Distributed under the **MIT License**.
