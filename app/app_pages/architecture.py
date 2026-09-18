"""
Model Architecture page: a layperson-friendly, somewhat-technical guide to
how each of the 9 models actually works, grouped into the 3 families used
throughout the project (classical / from-scratch neural / pretrained), plus
a side-by-side comparison table.

All hyperparameters/facts quoted here are read live from src.config or the
saved tokenizer, not hardcoded copies -- if the config changes, this page
stays accurate without a manual edit.
"""

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent))
sys.path.insert(0, str(_APP_DIR))

import json

import pandas as pd
import streamlit as st

from app_lib import TABLE_ROW_HEIGHT, display_name, load_model_performance, short_name
from diagrams import diagram_dot
from src import config

is_dark = st.context.theme.type == "dark"


@st.cache_data
def load_vocab_size():
    path = config.MODELS_SAVED_DIR / "tokenizer.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return len(json.load(f)["token_to_id"])


st.title("How do these models work?")
st.markdown(
    "A plain-language guide to each model this app compares -- what it actually "
    "does to a sentence, and how it differs from its neighbors."
)

vocab_size = load_vocab_size()
performance = load_model_performance()

st.space("small")

classical_tab, neural_tab, pretrained_tab = st.tabs(
    ["Classical", "Neural (from scratch)", "Pretrained"]
)

# --- Classical ------------------------------------------------------------

with classical_tab:
    st.markdown(
        "These three don't understand grammar or meaning at all -- they turn each "
        "review into a big list of *\"which words or short phrases appeared, and how "
        "many times,\"* then a Logistic Regression classifier learns which words tend "
        "to go with which sentiment. No neural network, no learned word meanings."
    )

    with st.expander(f"**{display_name('ngram')}**", icon=":material/format_list_numbered:"):
        st.markdown(
            f"Counts short **phrases of 1-2 words in a row** (like treating \"not good\" "
            "as one unit, not just \"not\" and \"good\" separately), using up to "
            f"**{config.CLASSICAL_MAX_FEATURES:,}** of the most common phrases as its "
            "vocabulary. Catches meaning that changes when words combine -- Bag-of-Words "
            "can't tell \"not good\" from \"good\" once it drops word order."
        )
        st.graphviz_chart(diagram_dot("ngram", is_dark))

    with st.expander(f"**{display_name('bow')}**", icon=":material/inventory_2:"):
        st.markdown(
            "The simplest of the three: counts single words only, **ignoring order "
            "entirely** -- \"bag\" of words, meaning only *which* words and *how many* "
            "matter, never their sequence. \"product bad\" and \"bad product\" look "
            "identical to this model."
        )
        st.graphviz_chart(diagram_dot("bow", is_dark))

    with st.expander(f"**{display_name('tfidf')}**", icon=":material/balance:"):
        text = (
            "Same phrase-counting idea as N-Gram, but instead of raw counts it "
            "**down-weights words that show up in almost every review** (like "
            "\"product\" or \"the\") and **up-weights words distinctive to a "
            "particular review** -- paying more attention to the unusual words."
        )
        if "tfidf" in performance:
            text += f" This is the best performer of all 9 models here (**{performance['tfidf']:.0%} test accuracy**)."
        st.markdown(text)
        st.graphviz_chart(diagram_dot("tfidf", is_dark))

# --- Neural (from scratch) -------------------------------------------------

with neural_tab:
    st.markdown(
        "These five are neural networks (PyTorch modules) built and trained **entirely "
        f"from this project's own {config.TARGET_DATASET_SIZE // 1000}k-review dataset** "
        "-- no outside knowledge, random weights at the start. Each word is represented "
        f"as a learned **{config.EMBEDDING_DIM}-number vector** (a Word2Vec \"embedding\" "
        "trained on this project's own text) instead of just a raw count."
    )

    with st.expander(f"**{display_name('ann')}**", icon=":material/scatter_plot:"):
        st.markdown(
            "**Averages together** every word's embedding vector into one single vector "
            "-- which throws away word order, similar in spirit to Bag-of-Words but with "
            "learned word meanings instead of raw counts -- then passes that through two "
            f"small layers ({' → '.join(str(d) for d in config.ANN_HIDDEN_DIMS)} neurons) "
            "to predict sentiment. Because it ignores order, this is the project's "
            "*neural lower bound* -- the baseline the sequence-aware models below need to beat."
        )
        st.graphviz_chart(diagram_dot("ann", is_dark))

    with st.expander(f"**{display_name('rnn')}**", icon=":material/arrow_forward:"):
        st.markdown(
            "Reads the review **one word at a time, left to right**, carrying forward a "
            f"running \"memory\" ({config.HIDDEN_DIM} numbers) updated after every word -- "
            "the first model here aware of word order. Its weakness: plain RNNs tend to "
            "**forget things said many words earlier** -- memory fades over a long sentence."
        )
        st.graphviz_chart(diagram_dot("rnn", is_dark))

    with st.expander(f"**{display_name('lstm')}**", icon=":material/menu_book:"):
        st.markdown(
            "An upgraded RNN with an internal \"notebook\" and **gates that decide what "
            "to remember, what to forget, and what's worth writing down** -- built "
            "specifically to fix the RNN's forgetting problem over longer text. This one "
            "also reads the sentence **in both directions** (left-to-right *and* "
            "right-to-left) and combines both readings."
        )
        st.graphviz_chart(diagram_dot("lstm", is_dark))

    with st.expander(f"**{display_name('attention')}**", icon=":material/visibility:"):
        st.markdown(
            "Built on top of an LSTM, with one more trick: instead of relying only on "
            "the LSTM's final summary, it **looks back at every single word's hidden "
            "state and decides, per prediction, which words mattered most** (\"attention "
            "weights\") -- a bit like re-reading a sentence and highlighting the "
            "important parts before answering."
        )
        st.graphviz_chart(diagram_dot("attention", is_dark))

    with st.expander(f"**{display_name('transformer')}**", icon=":material/hub:"):
        st.markdown(
            "A different strategy altogether: instead of reading word by word in "
            "sequence, it looks at the **whole sentence at once** and lets every word "
            "directly compare itself to every other word simultaneously "
            "(\"self-attention\"), figuring out which words relate to which. Since that "
            "gives it no inherent sense of order, each word also gets an explicit "
            f"position signal. Uses {config.TRANSFORMER_NUM_LAYERS} such layers with "
            f"{config.TRANSFORMER_NUM_HEADS} attention \"heads\" each. This is the same "
            "core idea behind large language models, just a much smaller version "
            "trained only on this project's data."
        )
        st.graphviz_chart(diagram_dot("transformer", is_dark))

# --- Pretrained -------------------------------------------------------------

with pretrained_tab:
    st.markdown(
        "The odd one out: unlike the other 8, this wasn't built or trained from "
        "scratch by this project."
    )
    with st.expander(f"**{display_name('bert')}**", icon=":material/psychology:"):
        st.markdown(
            f"**{config.BERT_MODEL_NAME}** is a Transformer Google already pretrained on "
            "a huge collection of multilingual text -- a big head start in the form of "
            "prior \"reading experience\" across many languages, including Bangla. This "
            f"project fine-tuned it briefly on a **{config.BERT_TRAIN_SUBSAMPLE_SIZE:,}-review "
            "sample** to teach it this specific task."
        )
        st.caption(
            f"Worth knowing: the other 8 models train on the full ~"
            f"{int(config.TARGET_DATASET_SIZE * (1 - config.TEST_HOLDOUT_SPLIT) * 0.8 / 1000)}"
            f"k-review training fold, while BERT fine-tunes on just "
            f"{config.BERT_TRAIN_SUBSAMPLE_SIZE:,} rows (mBERT is far slower per step on "
            "CPU) -- so its pretrained head start and its much smaller task-specific "
            "training set pull in opposite directions, and its numbers here aren't "
            "directly apples-to-apples with the others."
        )
        st.graphviz_chart(diagram_dot("bert", is_dark))

st.space("small")
st.subheader("Side-by-side comparison", divider="gray")

model_order = ["ngram", "bow", "tfidf", "ann", "rnn", "lstm", "attention", "transformer", "bert"]
comparison_df = pd.DataFrame([
    {
        "Model": short_name(name),
        "Family": {"ngram": "Classical", "bow": "Classical", "tfidf": "Classical",
                    "bert": "Pretrained"}.get(name, "Neural"),
        "Word order?": {
            "ngram": "Partial (phrases)", "bow": "No", "tfidf": "Partial (phrases)", "ann": "No",
            "rnn": "Yes, sequential", "lstm": "Yes, sequential", "attention": "Yes, sequential",
            "transformer": "Yes, all at once", "bert": "Yes, all at once",
        }[name],
        "Long-range memory": {
            "ngram": "-", "bow": "-", "tfidf": "-", "ann": "-", "rnn": "Weak (fades)",
            "lstm": "Strong (gated)", "attention": "Strong (gated + lookback)",
            "transformer": "Strong (whole sequence)", "bert": "Strong (whole sequence)",
        }[name],
        "Pretrained on outside data?": "Yes" if name == "bert" else "No",
        "Test accuracy": performance.get(name),
    }
    for name in model_order
])

st.dataframe(
    comparison_df,
    hide_index=True,
    width="stretch",
    row_height=TABLE_ROW_HEIGHT,
    height=(len(comparison_df) + 1) * TABLE_ROW_HEIGHT + 3,
    column_config={
        "Test accuracy": st.column_config.ProgressColumn(
            "Test accuracy", format="percent", min_value=0, max_value=1
        ),
    },
)

if vocab_size:
    st.caption(
        f"Shared vocabulary for the 5 from-scratch neural models: {vocab_size:,} tokens "
        f"(words seen at least {config.MIN_TOKEN_FREQ} times in training). BERT uses its "
        "own pretrained subword vocabulary instead."
    )
