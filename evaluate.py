"""
Model Evaluation — Twitter Sentiment Analysis

Loads the trained GRU model and tokenizer, evaluates on a held-out test
split, and generates:
  - Classification report (printed)
  - Confusion matrix heatmap  → results/confusion_matrix.png
  - ROC curve with AUC        → results/roc_curve.png
  - Sample predictions table   → results/sample_predictions.png

Usage:
    python evaluate.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
)

from preprocessing import clean_text, preprocess_texts

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ─── Configuration ──────────────────────────────────────────────────
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
CLEAN_DATA_PATH = "clean_data.csv"
RAW_DATA_PATH = "data.csv"
MAX_LEN = 100
VOCAB_SIZE = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_artifacts():
    """Load model, tokenizer, and dataset.

    Tries to load clean_data.csv first. If not found, falls back to
    the raw data.csv and cleans it (same steps as EDA.ipynb).
    """
    print("Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading tokenizer …")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading data …")
    if os.path.exists(CLEAN_DATA_PATH):
        df = pd.read_csv(CLEAN_DATA_PATH)
    elif os.path.exists(RAW_DATA_PATH):
        print(
            f"  '{CLEAN_DATA_PATH}' not found — loading raw '{RAW_DATA_PATH}' instead …"
        )
        df = pd.read_csv(RAW_DATA_PATH, encoding="latin1", header=None)
        df.columns = ["target", "id", "date", "query", "user", "text"]
        df["target"] = df["target"].apply(lambda x: 1 if x == 4 else 0)
        df = df[["target", "text"]]
    else:
        raise FileNotFoundError(
            f"Neither '{CLEAN_DATA_PATH}' nor '{RAW_DATA_PATH}' found.\n"
            "Please run EDA.ipynb first, or download data.csv from "
            "http://help.sentiment140.com/for-students"
        )

    return model, tokenizer, df


def prepare_test_data(df, tokenizer):
    """Reproduce the same train/test split used during training."""
    df["text_clean"] = df["text"].apply(clean_text)
    X, _ = preprocess_texts(
        df["text_clean"].tolist(),
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
    )
    y = df["target"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # keep raw texts aligned with test indices
    _, texts_test = train_test_split(
        df["text"].values, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_test, y_test, texts_test


def plot_confusion_matrix(y_true, y_pred):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted", fontsize=13)
    plt.ylabel("Actual", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()
    print("  ✓ Saved results/confusion_matrix.png")


def plot_roc_curve(y_true, y_prob):
    """Generate and save an ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"GRU Model (AUC = {roc_auc:.4f})")
    plt.plot(
        [0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random baseline"
    )
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curve", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved results/roc_curve.png  (AUC = {roc_auc:.4f})")
    return roc_auc


def plot_sample_predictions(texts, y_true, y_pred, y_prob, n=10):
    """Save a table image showing sample correct & incorrect predictions."""
    idx_correct = np.where(y_true == y_pred)[0]
    idx_wrong = np.where(y_true != y_pred)[0]

    samples = []
    for i in np.random.choice(
        idx_correct, min(n // 2, len(idx_correct)), replace=False
    ):
        samples.append(
            (
                texts[i][:80],
                "Positive" if y_true[i] else "Negative",
                "Positive" if y_pred[i] else "Negative",
                f"{y_prob[i]:.3f}",
                "✓",
            )
        )
    for i in np.random.choice(idx_wrong, min(n // 2, len(idx_wrong)), replace=False):
        samples.append(
            (
                texts[i][:80],
                "Positive" if y_true[i] else "Negative",
                "Positive" if y_pred[i] else "Negative",
                f"{y_prob[i]:.3f}",
                "✗",
            )
        )

    df_samples = pd.DataFrame(
        samples, columns=["Tweet", "Actual", "Predicted", "Confidence", ""]
    )
    fig, ax = plt.subplots(figsize=(16, max(3, len(df_samples) * 0.6)))
    ax.axis("off")
    table = ax.table(
        cellText=df_samples.values,
        colLabels=df_samples.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title("Sample Predictions", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("results/sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved results/sample_predictions.png")


# ─── Main ───────────────────────────────────────────────────────────


def main():
    model, tokenizer, df = load_artifacts()
    X_test, y_test, texts_test = prepare_test_data(df, tokenizer)

    print("\nRunning predictions on test set …")
    y_prob = model.predict(X_test, batch_size=512, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{'=' * 50}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test F1-Score : {f1:.4f}")
    print(f"{'=' * 50}\n")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # --- Plots ---
    print("Generating evaluation plots …")
    plot_confusion_matrix(y_test, y_pred)
    roc_auc = plot_roc_curve(y_test, y_prob)
    plot_sample_predictions(texts_test, y_test, y_pred, y_prob)

    # --- Summary file ---
    with open("results/metrics.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test F1-Score: {f1:.4f}\n")
        f.write(f"ROC AUC:       {roc_auc:.4f}\n")
    print("\n  ✓ Saved results/metrics.txt")
    print("\nDone! All evaluation artifacts are in the results/ folder.")


if __name__ == "__main__":
    main()
