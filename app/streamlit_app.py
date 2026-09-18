"""
Streamlit UI for the Code-Mix Sentiment Analyzer.

Lets a user type/paste a sentence in English, Bangla, Banglish, or
code-switched form, pick one or more trained models, and see each model's
predicted sentiment + confidence side by side, plus a comparison chart.

Run with: streamlit run app/streamlit_app.py
"""

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


@st.cache_resource
def load_all_models():
    models = {"classical": {}, "neural": {}, "bert": None, "tokenizer": None, "embedding_matrix": None}

    for name in CLASSICAL_MODELS:
        path = config.MODELS_SAVED_DIR / f"classical_{name}.pt"
        if path.exists():
            checkpoint = load_checkpoint(path)
            models["classical"][name] = (checkpoint["vectorizer"], checkpoint["classifier"])

    tokenizer_path = config.MODELS_SAVED_DIR / "tokenizer.json"
    word2vec_path = config.MODELS_SAVED_DIR / "word2vec.model"
    if tokenizer_path.exists() and word2vec_path.exists():
        tokenizer = CodeMixTokenizer.load(tokenizer_path)
        w2v = load_word2vec(word2vec_path)
        models["tokenizer"] = tokenizer
        models["embedding_matrix"] = build_embedding_matrix(w2v, tokenizer.token_to_id, config.EMBEDDING_DIM)

        for name in NEURAL_MODELS:
            path = config.MODELS_SAVED_DIR / f"{name}.pt"
            if path.exists():
                checkpoint = load_checkpoint(path)
                model = build_model(name, checkpoint["vocab_size"], models["embedding_matrix"])
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


def predict_with_model(model_name: str, models: dict, text: str):
    if model_name in CLASSICAL_MODELS:
        vectorizer, classifier = models["classical"][model_name]
        probs = classifier.predict_proba(transform(vectorizer, [text]))[0]
        best = probs.argmax()
        return classifier.classes_[best], float(probs[best])

    if model_name in NEURAL_MODELS:
        tokenizer = models["tokenizer"]
        model = models["neural"][model_name]
        input_ids = torch.tensor([tokenizer.encode(text, config.MAX_SEQUENCE_LENGTH)], dtype=torch.long)
        lengths = torch.tensor([max(1, min(len(tokenizer.tokenize(text)), config.MAX_SEQUENCE_LENGTH))])
        embedding_matrix_tensor = torch.as_tensor(models["embedding_matrix"], dtype=torch.float32)
        with torch.no_grad():
            logits = _forward(model, model_name, input_ids, lengths, embedding_matrix_tensor)
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


def render_comparison_chart(results: dict) -> None:
    st.caption("Predicted-label confidence by model")
    chart_data = pd.Series(results, name="confidence").to_frame()
    st.bar_chart(chart_data, y="confidence", color="primary", y_label="Confidence")


def main():
    st.set_page_config(
        page_title="Code-Mix Sentiment Analyzer",
        page_icon=":material/reviews:",
        layout="wide",
    )
    st.title("Code-Mix Sentiment Analyzer")
    st.caption(
        "Sentiment classification for English, Bangla, Banglish, and "
        "Bangla-English code-switched product reviews."
    )

    models = load_all_models()
    available = list(models["classical"].keys()) + list(models["neural"].keys())
    if models["bert"] is not None:
        available.append("bert")

    if not available:
        st.error(
            "No trained models found in models_saved/. Run the training scripts "
            "(see CLAUDE.md Phases 5, 7, 8) before launching this app.",
            icon=":material/error:",
        )
        return

    st.sidebar.header("Models")
    selected_models = st.sidebar.multiselect("Choose one or more models", options=available, default=available)

    with st.sidebar.expander("Sample sentences by language condition", icon=":material/translate:"):
        try:
            test_df = load_split("test")
            for language in config.LANGUAGE_CONDITIONS:
                examples = test_df[test_df["language"] == language]["text"]
                if len(examples) > 0:
                    st.markdown(f"**{language}**")
                    st.caption(examples.iloc[0])
        except FileNotFoundError:
            st.caption("data/processed/test.csv not found.")

    text = st.text_area("Enter a product review", height=120, placeholder="e.g. product ta khub valo, kintu delivery slow silo")

    if st.button("Analyze", icon=":material/query_stats:", type="primary") and text.strip():
        if not selected_models:
            st.warning("Select at least one model in the sidebar.", icon=":material/warning:")
            return

        results = {}
        columns = st.columns(len(selected_models))
        for column, model_name in zip(columns, selected_models):
            label, confidence = predict_with_model(model_name, models, text)
            results[model_name] = confidence
            with column:
                st.metric(model_name, label, f"{confidence:.0%} confidence")

        render_comparison_chart(results)


if __name__ == "__main__":
    main()
