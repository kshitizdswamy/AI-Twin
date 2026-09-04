import unittest
from src.models.sentiment_model import SentimentEmotionAnalyzer
from src.preprocessing.text_preprocess import clean_text, tokenize_sentences, extract_keywords


class TestSentimentAnalysis(unittest.TestCase):

    def setUp(self):
        self.analyzer = SentimentEmotionAnalyzer(use_transformers=False)

    def test_text_cleaning(self):
        dirty = "Check out https://example.com! Feeling #awesome & happy!  "
        cleaned = clean_text(dirty)
        self.assertEqual(cleaned, "Check out Feeling awesome happy!")

    def test_sentiment_classification(self):
        pos_text = "I had a great productive day accomplishing my goals!"
        res = self.analyzer.analyze_sentiment_and_emotion(pos_text)
        self.assertIn(res["sentiment"], ["Positive", "Neutral"])
        self.assertIn("emotion", res)

    def test_summarization(self):
        text = "This is sentence one. This is sentence two about productivity. This is sentence three."
        summary = self.analyzer.summarize_text(text)
        self.assertTrue(len(summary) > 0)


if __name__ == "__main__":
    unittest.main()
