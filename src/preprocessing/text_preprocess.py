import re
from typing import List


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, special characters, extra whitespace,
    and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special non-alphanumeric characters except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_sentences(text: str) -> List[str]:
    """Split input text into individual sentences."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract simple top keywords excluding common English stop words."""
    stopwords = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "both", "through", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "upon",
        "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "s", "t", "can", "will", "just", "don", "should", "now", "i", "me", "my",
        "myself", "we", "our", "ours", "you", "your", "yours", "he", "him", "his",
        "she", "her", "hers", "it", "its", "they", "them", "their", "am", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "feeling", "day", "got", "worked", "spent", "need"
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [w for w in words if w not in stopwords]

    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_n]]
