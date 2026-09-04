import random
from typing import Dict, Any
from src.models.sentiment_model import SentimentEmotionAnalyzer


class Chatbot:
    """
    Personalized AI Twin Chatbot analyzing user sentiment, emotion state,
    and digital wellbeing activity to generate empathetic & actionable advice.
    """

    def __init__(self, name: str = "Smart AI Twin"):
        self.name = name
        self.nlp_analyzer = SentimentEmotionAnalyzer(use_transformers=False)

    def get_response(self, user_input: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate contextual AI Twin response given user message and activity context.
        """
        if not user_input or not user_input.strip():
            return {
                "bot_name": self.name,
                "reply": "I am listening. Feel free to share your thoughts, how your day is going, or ask about your screen time trends.",
                "detected_emotion": "Neutral",
                "sentiment": "Neutral"
            }

        # Analyze message NLP
        analysis = self.nlp_analyzer.analyze_sentiment_and_emotion(user_input)
        emotion = analysis["emotion"]
        sentiment = analysis["sentiment"]

        # Parse context if provided
        screen_time = user_context.get("screen_time_mins", 280) if user_context else 280
        is_anomaly = user_context.get("is_anomaly", False) if user_context else False

        reply = self._generate_personalized_reply(user_input, emotion, sentiment, screen_time, is_anomaly)

        return {
            "bot_name": self.name,
            "reply": reply,
            "detected_emotion": emotion,
            "sentiment": sentiment,
            "confidence": analysis.get("emotion_confidence", 0.85),
            "keywords": analysis.get("keywords", [])
        }

    def _generate_personalized_reply(
        self, text: str, emotion: str, sentiment: str, screen_time: float, is_anomaly: bool
    ) -> str:
        text_lower = text.lower()

        # Query about screen time / stats
        if any(w in text_lower for w in ["screen time", "usage", "predict", "forecast", "phone", "hours"]):
            hours = round(screen_time / 60, 1)
            status = "slightly elevated today" if screen_time > 360 else "within a balanced range"
            return (
                f"Based on your recent activity logs, your daily screen usage is around {hours} hours ({status}). "
                f"The predictive model forecasts upcoming screen time based on your productivity and entertainment balance."
            )

        # Query about detox / break
        if any(w in text_lower for w in ["detox", "break", "stress", "tired", "anxious", "overwhelmed"]):
            return (
                f"I notice you might be feeling {emotion.lower()}. "
                f"When screen time increases or stress builds up, taking a short 15-minute walk or practicing a brief breathing exercise can significantly help reset your focus."
            )

        # Emotion specific responses
        if emotion == "Anxiety":
            return (
                "I understand. It sounds like workload or deadlines are feeling demanding right now. "
                "Try breaking down your tasks into small 15-minute segments and turn on Do Not Disturb to reduce notification interruptions."
            )
        elif emotion == "Joy":
            return (
                "That is great news. It is good to see you maintaining a strong momentum and positive focus today."
            )
        elif emotion == "Focus":
            return (
                "You are currently in a deep focus state. Make sure to stay hydrated and protect your focus blocks."
            )
        elif emotion == "Sadness":
            return (
                "Take it easy on yourself today. Stepping away from screens for a brief rest or catching up with a friend can help."
            )
        else:
            return (
                f"Thanks for sharing. I am monitoring your activity metrics and wellbeing trends. "
                f"Let me know if you would like insights on your habits or screen time predictions."
            )


if __name__ == "__main__":
    bot = Chatbot()
    res = bot.get_response("I am feeling super stressed about my project deadline tomorrow!")
    print("Bot Response:", res)
