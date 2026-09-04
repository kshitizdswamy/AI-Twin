import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.metrics import calculate_classification_metrics
from src.models.sentiment_model import SentimentEmotionAnalyzer


def evaluate_sentiment_analyzer():
    """Evaluate sentiment and emotion classification on benchmark text samples."""
    analyzer = SentimentEmotionAnalyzer(use_transformers=False)

    test_cases = [
        ("Had an amazingly productive coding session today!", "Positive", "Joy"),
        ("Overwhelmed with deadlines and feeling super stressed out.", "Negative", "Anxiety"),
        ("Routine day at work, checked emails and updated tasks.", "Neutral", "Calm"),
        ("Felt sad and lonely missing my family.", "Negative", "Sadness"),
        ("Learned a lot reading NLP research papers today.", "Positive", "Focus")
    ]

    true_sentiments = []
    pred_sentiments = []

    true_emotions = []
    pred_emotions = []

    print("=== Sentiment & Emotion Model Benchmarks ===")
    for text, exp_sent, exp_emo in test_cases:
        res = analyzer.analyze_sentiment_and_emotion(text)
        true_sentiments.append(exp_sent)
        pred_sentiments.append(res["sentiment"])

        true_emotions.append(exp_emo)
        pred_emotions.append(res["emotion"])

        print(f"Text: '{text}'")
        print(f" -> Predicted Sentiment: {res['sentiment']} (Exp: {exp_sent}), Emotion: {res['emotion']} (Exp: {exp_emo})")

    label_map = {"Positive": 0, "Negative": 1, "Neutral": 2}
    y_true_s = [label_map.get(s, 2) for s in true_sentiments]
    y_pred_s = [label_map.get(s, 2) for s in pred_sentiments]

    metrics = calculate_classification_metrics(y_true_s, y_pred_s)
    print("\nSentiment Benchmarking Accuracy:", metrics["accuracy"])
    return metrics


if __name__ == "__main__":
    evaluate_sentiment_analyzer()
