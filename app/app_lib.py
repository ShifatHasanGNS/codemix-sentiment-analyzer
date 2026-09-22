"""Shared constants, cached model loaders, and UI helpers for app_pages/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import torch

from src import config
from src.data.preprocess import load_split
from src.data.tokenizer import CodeMixTokenizer
from src.features.classical_features import transform
from src.features.embeddings import build_embedding_matrix, load_word2vec
from src.models.bert_model import encode_batch
from src.training.train_neural import _forward, build_model
from src.utils.io_utils import load_checkpoint

CLASSICAL_MODELS = ("ngram", "bow", "tfidf")
NEURAL_MODELS = ("ann", "rnn", "lstm", "attention", "transformer")

# row height used to size no-scroll st.dataframe tables
TABLE_ROW_HEIGHT = 38

MODEL_DISPLAY = {
    "ngram": ("Classical", "N-Gram"),
    "bow": ("Classical", "Bag-of-Words"),
    "tfidf": ("Classical", "TF-IDF"),
    "ann": ("Neural", "ANN"),
    "rnn": ("Neural", "RNN"),
    "lstm": ("Neural", "LSTM"),
    "attention": ("Neural", "Attention"),
    "transformer": ("Neural", "Transformer"),
    "bert": ("Pretrained", "BERT"),
}

SENTIMENT_STYLE = {
    "positive": ("green", ":material/sentiment_satisfied:"),
    "negative": ("red", ":material/sentiment_dissatisfied:"),
    "neutral": ("gray", ":material/sentiment_neutral:"),
}


def display_name(model_name: str) -> str:
    category, name = MODEL_DISPLAY.get(model_name, ("", model_name))
    return f"{category} – {name}" if category else name


def short_name(model_name: str) -> str:
    """Bare model name, no category prefix -- for tight display spaces."""
    return MODEL_DISPLAY.get(model_name, ("", model_name))[1]


def style(is_dark: bool) -> str:
    """Minimal CSS scoped to specific widget keys."""
    shadow = "rgba(0, 0, 0, 0.45)" if is_dark else "rgba(0, 0, 0, 0.08)"
    shadow_sm = "rgba(0, 0, 0, 0.5)" if is_dark else "rgba(0, 0, 0, 0.10)"
    sep_color = "rgba(161, 161, 170, 0.35)" if is_dark else "rgba(113, 113, 122, 0.30)"
    return f"""
<style>
.main .block-container {{ animation: cm-fade-in 0.3s ease-out; }}
@keyframes cm-fade-in {{
  from {{ opacity: 0; transform: translateY(4px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

.st-key-verdict_card, [class*="st-key-model_card_"] {{
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.st-key-verdict_card:hover, [class*="st-key-model_card_"]:hover {{
  transform: translateY(-2px);
  box-shadow: 0 6px 16px {shadow};
}}

.stButton button, .stFormSubmitButton button {{
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}}
.stButton button:hover, .stFormSubmitButton button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 2px 8px {shadow_sm};
}}

.st-key-review_text textarea {{
  font-size: 1.15rem;
  line-height: 1.5;
}}

.cm-soft-sep {{
  width: 50%;
  height: 3px;
  margin: 0.9rem auto;
  border: none;
  border-radius: 999px;
  background: {sep_color};
}}
</style>
"""


def soft_divider() -> None:
    """A subtle, soft-edged half-width separator."""
    st.html('<div class="cm-soft-sep"></div>')


@st.cache_resource
def load_all_models():
    models = {
        "classical": {},
        "neural": {},
        "bert": None,
        "tokenizer": None,
        "embedding_matrix": None,
    }

    for name in CLASSICAL_MODELS:
        path = config.MODELS_SAVED_DIR / f"classical_{name}.pt"
        if path.exists():
            checkpoint = load_checkpoint(path)
            models["classical"][name] = (
                checkpoint["vectorizer"],
                checkpoint["classifier"],
            )

    tokenizer_path = config.MODELS_SAVED_DIR / "tokenizer.json"
    word2vec_path = config.MODELS_SAVED_DIR / "word2vec.model"
    if tokenizer_path.exists() and word2vec_path.exists():
        tokenizer = CodeMixTokenizer.load(tokenizer_path)
        w2v = load_word2vec(word2vec_path)
        models["tokenizer"] = tokenizer
        models["embedding_matrix"] = build_embedding_matrix(
            w2v, tokenizer.token_to_id, config.EMBEDDING_DIM
        )

        for name in NEURAL_MODELS:
            path = config.MODELS_SAVED_DIR / f"{name}.pt"
            if path.exists():
                checkpoint = load_checkpoint(path)
                model = build_model(
                    name, checkpoint["vocab_size"], models["embedding_matrix"]
                )
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                models["neural"][name] = model

    bert_path = config.MODELS_SAVED_DIR / "bert_finetuned"
    if bert_path.exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
        bert_model.eval()
        models["bert"] = (bert_tokenizer, bert_model)

    return models


@st.cache_data
def load_model_performance():
    """Each model's overall test-set accuracy, sorted best-first."""
    path = config.RESULTS_DIR / "metrics_comparison.csv"
    if not path.exists():
        return {}
    metrics_df = pd.read_csv(path)
    overall = metrics_df[metrics_df["language"] == "overall"].sort_values(
        "accuracy", ascending=False
    )
    return dict(zip(overall["model"], overall["accuracy"]))


@st.cache_data
def load_sample_pool():
    """Test-set sentences grouped by language, for the example buttons."""
    try:
        test_df = load_split("test")
    except FileNotFoundError:
        return {}
    pool = {}
    for language in config.LANGUAGE_CONDITIONS:
        examples = test_df[test_df["language"] == language]["text"].tolist()
        if examples:
            pool[language] = examples
    return pool


def predict_with_model(model_name: str, models: dict, text: str):
    if model_name in CLASSICAL_MODELS:
        vectorizer, classifier = models["classical"][model_name]
        probs = classifier.predict_proba(transform(vectorizer, [text]))[0]
        best = probs.argmax()
        return classifier.classes_[best], float(probs[best])

    if model_name in NEURAL_MODELS:
        tokenizer = models["tokenizer"]
        model = models["neural"][model_name]
        input_ids = torch.tensor(
            [tokenizer.encode(text, config.MAX_SEQUENCE_LENGTH)], dtype=torch.long
        )
        lengths = torch.tensor(
            [max(1, min(len(tokenizer.tokenize(text)), config.MAX_SEQUENCE_LENGTH))]
        )
        embedding_matrix_tensor = torch.as_tensor(
            models["embedding_matrix"], dtype=torch.float32
        )
        with torch.no_grad():
            logits = _forward(
                model, model_name, input_ids, lengths, embedding_matrix_tensor
            )
            probs = torch.softmax(logits[0], dim=-1)
        best = int(probs.argmax())
        return config.LABELS[best], float(probs[best])

    if model_name == "bert":
        bert_tokenizer, bert_model = models["bert"]
        encoded = encode_batch(bert_tokenizer, [text], config.BERT_MAX_SEQUENCE_LENGTH)
        with torch.no_grad():
            logits = bert_model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1)
        best = int(probs.argmax())
        return config.LABELS[best], float(probs[best])

    raise ValueError(f"Unknown model_name: {model_name!r}")
