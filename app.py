"""
Gradio Demo — Twitter Sentiment Analysis

A clean, interactive web interface for the trained GRU sentiment model.
Accepts a tweet and returns the predicted sentiment with confidence score.

Usage:
    python app.py
"""

import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr

from preprocessing import clean_text

# ─── Load Model & Tokenizer ────────────────────────────────────────

MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100

print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading tokenizer …")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# ─── Prediction Function ───────────────────────────────────────────


def predict_sentiment(text: str) -> dict:
    """Predict the sentiment of a tweet.

    Args:
        text: Raw tweet string.

    Returns:
        Dictionary with class labels as keys and confidence scores as values.
    """
    if not text or not text.strip():
        return {"Positive 😊": 0.0, "Negative 😞": 0.0}

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = float(model.predict(padded, verbose=0)[0][0])

    return {
        "Positive 😊": round(prob, 4),
        "Negative 😞": round(1 - prob, 4),
    }


# ─── Gradio Interface ──────────────────────────────────────────────

examples = [
    ["I absolutely love this product! Best purchase ever!"],
    ["This is the worst experience I have ever had."],
    ["Just had an amazing day at the beach with friends 🌊"],
    ["I'm so disappointed with the customer service."],
    ["The weather is nice today, feeling great!"],
    ["Can't believe how terrible this movie was, total waste of time."],
    ["Congratulations on your new job! So happy for you!"],
    ["My flight got cancelled again... so frustrated 😤"],
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Enter a Tweet",
        placeholder="Type or paste a tweet here …",
        lines=3,
    ),
    outputs=gr.Label(
        label="Sentiment Prediction",
        num_top_classes=2,
    ),
    title="🐦 Twitter Sentiment Analysis",
    description=(
        "A GRU-based deep learning model trained on **1.6 million tweets** "
        "to classify text as **Positive** or **Negative**.\n\n"
        "Enter any tweet below (or click one of the examples) to see the prediction."
    ),
    examples=examples,
    theme=gr.themes.Soft(),
)

# ─── Launch ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
