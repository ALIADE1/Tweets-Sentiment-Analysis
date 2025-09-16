import re
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s#\U0001F600-\U0001F64F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_texts(texts, tokenizer=None, vocab_size=10000, max_len=100):
    cleaned_texts = [clean_text(t) for t in texts]
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    return padded, tokenizer