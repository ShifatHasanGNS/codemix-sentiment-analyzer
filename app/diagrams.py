"""Builds per-model pipeline diagrams as DOT strings for st.graphviz_chart."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config


def _palette(is_dark: bool) -> dict:
    if is_dark:
        return {
            "bg": "#09090B",
            "fill": "#18181B",
            "border": "#3F3F46",
            "text": "#FAFAFA",
            "edge": "#71717A",
            "edge_text": "#A1A1AA",
            "accent_fill": "#1E3A5F",
            "accent_border": "#60A5FA",
            "accent_text": "#FAFAFA",
        }
    return {
        "bg": "#FFFFFF",
        "fill": "#FAFAFA",
        "border": "#D4D4D8",
        "text": "#09090B",
        "edge": "#A1A1AA",
        "edge_text": "#52525B",
        "accent_fill": "#DBEAFE",
        "accent_border": "#2563EB",
        "accent_text": "#09090B",
    }


_FONT = "Inter,Helvetica,Arial,sans-serif"


def _header(p: dict) -> str:
    return (
        f"digraph G {{\n"
        f'  rankdir=LR; bgcolor="{p["bg"]}"; nodesep=0.35; ranksep=0.45;\n'
        f'  node [shape=box style="rounded,filled" fillcolor="{p["fill"]}" '
        f'color="{p["border"]}" fontcolor="{p["text"]}" fontname="{_FONT}" '
        f"fontsize=13 margin=0.18];\n"
        f'  edge [color="{p["edge"]}" fontcolor="{p["edge_text"]}" '
        f'fontname="{_FONT}" fontsize=11];\n'
    )


def _node(node_id: str, label: str, p: dict, accent: bool = False) -> str:
    if accent:
        return (
            f'  {node_id} [label="{label}" fillcolor="{p["accent_fill"]}" '
            f'color="{p["accent_border"]}" fontcolor="{p["accent_text"]}" '
            f"penwidth=1.5];\n"
        )
    return f'  {node_id} [label="{label}"];\n'


def _chain(node_ids: list) -> str:
    return "  " + " -> ".join(node_ids) + ";\n"


def _classical_diagram(is_dark: bool, count_label: str, weight_note: str = "") -> str:
    p = _palette(is_dark)
    dot = _header(p)
    dot += _node("a", "Review text", p)
    dot += _node("b", "Tokenize", p)
    dot += _node("c", count_label, p)
    if weight_note:
        dot += _node("d", weight_note, p)
        dot += _node("e", "Feature vector", p)
        dot += _node("f", "Logistic\nRegression", p)
        dot += _node("g", "3-class\nprediction", p)
        dot += _chain(["a", "b", "c", "d", "e", "f", "g"])
    else:
        dot += _node("e", "Feature vector", p)
        dot += _node("f", "Logistic\nRegression", p)
        dot += _node("g", "3-class\nprediction", p)
        dot += _chain(["a", "b", "c", "e", "f", "g"])
    dot += "}\n"
    return dot


def _ann_diagram(is_dark: bool) -> str:
    p = _palette(is_dark)
    hidden = " → ".join(str(d) for d in config.ANN_HIDDEN_DIMS)
    dot = _header(p)
    dot += _node("a", "Review text", p)
    dot += _node("b", "Tokenize", p)
    dot += _node("c", f"Word2Vec embedding\nper word ({config.EMBEDDING_DIM}-dim)", p)
    dot += _node("d", "Average all\nword vectors", p)
    dot += _node("e", f"Dense layers\n({hidden})", p)
    dot += _node("f", "3-class\nprediction", p)
    dot += _chain(["a", "b", "c", "d", "e", "f"])
    dot += "}\n"
    return dot


def _rnn_diagram(is_dark: bool) -> str:
    p = _palette(is_dark)
    dot = _header(p)
    dot += _node("a", "Review text", p)
    dot += _node("b", "Tokenize", p)
    dot += _node("c", f"Word2Vec embedding\nper word ({config.EMBEDDING_DIM}-dim)", p)
    dot += _node(
        "d", f"RNN reads word by word\n(hidden state, {config.HIDDEN_DIM}-dim)", p
    )
    dot += _node("e", "Final hidden\nstate", p)
    dot += _node("f", "Linear", p)
    dot += _node("g", "3-class\nprediction", p)
    dot += _chain(["a", "b", "c", "d", "e", "f", "g"])
    dot += '  d -> d [label="  per word,\\l  memory fades\\l  over distance\\l"];\n'
    dot += "}\n"
    return dot


def _lstm_diagram(is_dark: bool) -> str:
    p = _palette(is_dark)
    dot = _header(p)
    dot += _node("a", "Review text", p)
    dot += _node("b", "Tokenize", p)
    dot += _node("c", f"Word2Vec embedding\nper word ({config.EMBEDDING_DIM}-dim)", p)
    dot += _node("d1", f"LSTM forward →\n(hidden {config.HIDDEN_DIM}-dim)", p)
    dot += _node("d2", f"LSTM backward ←\n(hidden {config.HIDDEN_DIM}-dim)", p)
    dot += _node("e", f"Concatenate\n({config.HIDDEN_DIM * 2}-dim)", p)
    dot += _node("f", "Linear", p)
    dot += _node("g", "3-class\nprediction", p)
    dot += "  a -> b -> c;\n"
    dot += "  c -> d1; c -> d2;\n"
    dot += '  d1 -> d1 [label="gated memory"];\n'
    dot += '  d2 -> d2 [label="gated memory"];\n'
    dot += "  d1 -> e; d2 -> e;\n"
    dot += "  e -> f -> g;\n"
    dot += "}\n"
    return dot


def _attention_diagram(is_dark: bool) -> str:
    p = _palette(is_dark)
    dot = _header(p)
    dot += _node("a", "Review text", p)
    dot += _node("b", "Tokenize", p)
    dot += _node("c", f"Word2Vec embedding\nper word ({config.EMBEDDING_DIM}-dim)", p)
    dot += _node(
        "d",
        f"LSTM keeps every step's\nhidden state h₁ … h_T ({config.HIDDEN_DIM}-dim)",
        p,
    )
    dot += _node("e", "Attention: weighs\neach hᵢ by relevance", p)
    dot += _node("f", "Context vector\n(weighted sum)", p)
    dot += _node("g", "Linear", p)
    dot += _node("h", "3-class\nprediction", p)
    dot += _chain(["a", "b", "c", "d", "e", "f", "g", "h"])
    dot += '  d -> d [label="per word"];\n'
    dot += "}\n"
    return dot


def _transformer_diagram(is_dark: bool) -> str:
    p = _palette(is_dark)
    dot = _header(p)
    dot += _node("a", "Review text", p)
    dot += _node("b", "Tokenize", p)
    dot += _node("c1", f"Embedding\n({config.EMBEDDING_DIM}-dim)", p)
    dot += _node("c2", "Positional\nencoding", p)
    dot += _node("plus", "+", p)
    dot += _node(
        "d",
        f"Self-attention\n×{config.TRANSFORMER_NUM_LAYERS} layers, "
        f"{config.TRANSFORMER_NUM_HEADS} heads each",
        p,
    )
    dot += _node("e", "Mean-pool\n(non-pad tokens)", p)
    dot += _node("f", "Linear", p)
    dot += _node("g", "3-class\nprediction", p)
    dot += "  a -> b -> c1;\n"
    dot += "  c1 -> plus; c2 -> plus;\n"
    dot += "  plus -> d -> e -> f -> g;\n"
    dot += "}\n"
    return dot


def _bert_diagram(is_dark: bool) -> str:
    p = _palette(is_dark)
    dot = _header(p)
    dot += _node("a", "Review text", p)
    dot += _node("b", "mBERT's own subword\ntokenizer", p)
    dot += _node(
        "c",
        "Pretrained mBERT encoder\n(already trained by Google on\nmany languages, incl. Bangla)",
        p,
        accent=True,
    )
    dot += _node("d", "[CLS] token's\nvector", p)
    dot += _node(
        "e",
        f"New classifier head\n(fine-tuned here on {config.BERT_TRAIN_SUBSAMPLE_SIZE // 1000}k reviews)",
        p,
    )
    dot += _node("f", "3-class\nprediction", p)
    dot += _chain(["a", "b", "c", "d", "e", "f"])
    dot += "}\n"
    return dot


_BUILDERS = {
    "ngram": lambda is_dark: _classical_diagram(
        is_dark, "Count 1-2 word\nphrases (≤20k)"
    ),
    "bow": lambda is_dark: _classical_diagram(is_dark, "Count single\nwords only"),
    "tfidf": lambda is_dark: _classical_diagram(
        is_dark,
        "Count 1-2 word\nphrases",
        "Weight: common words ↓\ndistinctive words ↑",
    ),
    "ann": _ann_diagram,
    "rnn": _rnn_diagram,
    "lstm": _lstm_diagram,
    "attention": _attention_diagram,
    "transformer": _transformer_diagram,
    "bert": _bert_diagram,
}


def diagram_dot(model_name: str, is_dark: bool) -> str:
    return _BUILDERS[model_name](is_dark)
