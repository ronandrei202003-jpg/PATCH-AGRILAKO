import re
import langid
from transformers import pipeline

_translation_pipeline = None
_sentiment_pipeline = None

def get_translation_pipeline():
    global _translation_pipeline
    if _translation_pipeline is None:
        _translation_pipeline = pipeline(
            "translation",
            model="facebook/nllb-200-1.3B",
            tokenizer="facebook/nllb-200-1.3B",
            src_lang="tgl_Latn",
            tgt_lang="eng_Latn",
            max_length=100
        )
    return _translation_pipeline

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
    return _sentiment_pipeline

# Sentiment label mapping
sentiment_label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "POSITIVE": "Positive",
}

# --- Utility Functions ---
def clean_text(text):
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

# --- Main Functions ---
def translate_text(text):
    try:
        cleaned = clean_text(text)
        if detect_language(cleaned) == "tl":
            return get_translation_pipeline()(cleaned)[0]["translation_text"]
        return cleaned
    except Exception as e:
        return f"[Translation failed] {e}"

def analyze_sentiment(text):
    try:
        result = get_sentiment_pipeline()(text)
        return sentiment_label_map.get(result[0]['label'], result[0]['label'])
    except Exception as e:
        return f"[Sentiment failed] {e}"

def analyze_feedback(feedback_text):
    # Handle empty or default message
    if not feedback_text or feedback_text.strip() == "" or feedback_text.strip().lower() == "no comment provided":
        return {
            "original": feedback_text,
            "translated": feedback_text,
            "sentiment": "Neutral",
        }
    
    translated = translate_text(feedback_text)
    sentiment = analyze_sentiment(translated)
    return {
        "original": feedback_text,
        "translated": translated,
        "sentiment": sentiment,
    }
