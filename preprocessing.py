"""
Text preprocessing utilities for Twitter Sentiment Analysis.

Provides text cleaning, tokenization, and padding functions
for preparing tweet data for model training and inference.
"""

import re
import pickle
from typing import Optional, Tuple, List

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --- Text Cleaning ---


def clean_text(text: str) -> str:
    """Clean a single tweet by removing noise while preserving sentiment signals.

    Steps:
        1. Lowercase the text
        2. Remove URLs
        3. Remove @mentions
        4. Remove special characters (keep letters, numbers, spaces, #)
        5. Collapse multiple whitespace

    Args:
        text: Raw tweet string.

    Returns:
        Cleaned tweet string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove @mentions
    text = re.sub(r"[^a-zA-Z0-9\s#]", "", text)  # Keep alphanumeric + #
    text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
    return text


def clean_texts_batch(texts: List[str]) -> List[str]:
    """Apply clean_text to a list of tweets.

    Args:
        texts: List of raw tweet strings.

    Returns:
        List of cleaned tweet strings.
    """
    return [clean_text(t) for t in texts]


# --- Tokenization & Padding ---


def preprocess_texts(
    texts: List[str],
    tokenizer: Optional[Tokenizer] = None,
    vocab_size: int = 10000,
    max_len: int = 100,
) -> Tuple[np.ndarray, Tokenizer]:
    """Tokenize and pad a list of texts.

    If no tokenizer is provided, a new one is fitted on the input texts.
    Otherwise, the provided tokenizer is used for encoding only.

    Args:
        texts:      List of raw or pre-cleaned tweet strings.
        tokenizer:  Pre-fitted Keras Tokenizer (None to create a new one).
        vocab_size: Maximum vocabulary size for a new tokenizer.
        max_len:    Max sequence length for padding/truncating.

    Returns:
        Tuple of (padded_sequences, tokenizer).
    """
    cleaned_texts = clean_texts_batch(texts)

    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(cleaned_texts)

    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=max_len)

    return padded, tokenizer


def load_tokenizer(path: str = "tokenizer.pkl") -> Tokenizer:
    """Load a saved Keras Tokenizer from a pickle file.

    Args:
        path: File path to the pickled tokenizer.

    Returns:
        The deserialized Tokenizer object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
