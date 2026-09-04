import logging
from typing import Dict, Any, List
from src.preprocessing.text_preprocess import clean_text, extract_keywords

logger = logging.getLogger(__name__)

class SentimentEmotionAnalyzer:
    """
    NLP Sentiment & Emotion Analyzer powered by HuggingFace Transformers
    with robust rule-based / lexicon fallbacks.
    """

    def __init__(self, use_transformers: bool = True):
        self.use_transformers = use_transformers
        self.sentiment_pipeline = None
        self.summarizer_pipeline = None

        if self.use_transformers:
            try:
                from transformers import pipeline
                logger.info("Initializing HuggingFace sentiment analysis pipeline...")
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    top_k=None
                )
            except Exception as e:
                logger.warning(f"Could not load HuggingFace pipeline ({e}). Falling back to rule-based analyzer.")
                self.sentiment_pipeline = None

    def analyze_sentiment_and_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment, detect primary emotion, and extract key insights.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "sentiment": "Neutral",
                "score": 0.5,
                "emotion": "Neutral",
                "emotion_confidence": 0.5,
                "keywords": []
            }

        # Try HuggingFace pipeline first if available
        if self.sentiment_pipeline:
            try:
                results = self.sentiment_pipeline(cleaned)
                # Results is list of list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
                scores_list = results[0] if isinstance(results, list) and isinstance(results[0], list) else results
                label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
                top_result = max(scores_list, key=lambda x: x["score"])
                sentiment_label = label_map.get(top_result["label"].lower(), top_result["label"].capitalize())
                score = round(float(top_result["score"]), 3)

                emotion = self._infer_emotion(cleaned, sentiment_label)
                keywords = extract_keywords(cleaned)

                return {
                    "sentiment": sentiment_label,
                    "score": score,
                    "emotion": emotion["label"],
                    "emotion_confidence": emotion["confidence"],
                    "keywords": keywords
                }
            except Exception as e:
                logger.warning(f"Error running transformer sentiment pipeline: {e}")

        # Fallback Lexicon/Rule-based Engine
        return self._rule_based_analysis(cleaned)

    def _infer_emotion(self, text: str, sentiment: str) -> Dict[str, Any]:
        """Infer granular emotion based on keywords and sentiment context."""
        text_lower = text.lower()

        anxious_words = ["anxious", "stressed", "overwhelmed", "deadline", "scrolling", "distracted", "exhausted"]
        joy_words = ["productive", "proud", "energized", "wonderful", "great", "crushed", "learned"]
        sad_words = ["sad", "missed", "low", "unmotivated", "disappointed", "tired"]
        focus_words = ["focus", "reading", "paper", "worked", "backlog", "organized"]

        if any(w in text_lower for w in anxious_words):
            return {"label": "Anxiety", "confidence": 0.88}
        elif any(w in text_lower for w in joy_words):
            return {"label": "Joy", "confidence": 0.92}
        elif any(w in text_lower for w in sad_words):
            return {"label": "Sadness", "confidence": 0.85}
        elif any(w in text_lower for w in focus_words):
            return {"label": "Focus", "confidence": 0.86}

        if sentiment == "Positive":
            return {"label": "Joy", "confidence": 0.75}
        elif sentiment == "Negative":
            return {"label": "Anxiety", "confidence": 0.70}
        else:
            return {"label": "Calm", "confidence": 0.80}

    def _rule_based_analysis(self, cleaned_text: str) -> Dict[str, Any]:
        """Rule-based lexicon sentiment & emotion fallback."""
        text_lower = cleaned_text.lower()
        pos_words = {"productive", "proud", "energized", "wonderful", "great", "good", "learned", "completed"}
        neg_words = {"overwhelmed", "anxious", "stressed", "exhausted", "distracted", "sad", "low", "disappointed"}

        words = text_lower.split()
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)

        if pos_count > neg_count:
            sentiment = "Positive"
            score = round(0.6 + 0.1 * min(pos_count, 4), 2)
        elif neg_count > pos_count:
            sentiment = "Negative"
            score = round(0.6 + 0.1 * min(neg_count, 4), 2)
        else:
            sentiment = "Neutral"
            score = 0.50

        emotion = self._infer_emotion(cleaned_text, sentiment)
        keywords = extract_keywords(cleaned_text)

        return {
            "sentiment": sentiment,
            "score": score,
            "emotion": emotion["label"],
            "emotion_confidence": emotion["confidence"],
            "keywords": keywords
        }

    def summarize_text(self, text: str, max_length: int = 60) -> str:
        """Summarize daily user journal or status entries."""
        cleaned = clean_text(text)
        if len(cleaned.split()) < 12:
            return cleaned

        # Simple high-value sentence summarization
        sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
        if sentences:
            return sentences[0] + "."
        return cleaned[:max_length] + "..."


if __name__ == "__main__":
    analyzer = SentimentEmotionAnalyzer(use_transformers=False)
    sample = "Feeling overwhelmed by deadlines and too many unread notifications. Need a break."
    res = analyzer.analyze_sentiment_and_emotion(sample)
    print("Result:", res)
    print("Summary:", analyzer.summarize_text(sample))
