"""Analyzer page: pick/type a review, choose models, compare predictions."""

import random
import sys
from collections import Counter
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent))
sys.path.insert(0, str(_APP_DIR))

import pandas as pd
import streamlit as st
from app_lib import (
    SENTIMENT_STYLE,
    TABLE_ROW_HEIGHT,
    display_name,
    load_all_models,
    load_model_performance,
    load_sample_pool,
    predict_with_model,
    short_name,
    soft_divider,
)


def render_verdict(results: dict) -> None:
    """Consensus summary: majority-vote label across selected models."""
    labels = [label for label, _ in results.values()]
    overall_label, votes = Counter(labels).most_common(1)[0]
    agreeing_confidences = [
        conf for label, conf in results.values() if label == overall_label
    ]
    avg_confidence = sum(agreeing_confidences) / len(agreeing_confidences)
    color, icon = SENTIMENT_STYLE[overall_label]

    with st.container(border=True, key="verdict_card"):
        st.caption("Overall verdict")
        st.badge(overall_label.capitalize(), icon=icon, color=color)
        total = len(results)
        agreement = (
            "unanimous" if votes == total else f"{votes} of {total} models agree"
        )
        st.caption(f"{agreement} · {avg_confidence:.0%} average confidence")


def render_model_cards(results: dict) -> None:
    """Compact cards: label + confidence as text only."""
    st.subheader("Per-model breakdown", divider="gray")
    with st.container(horizontal=True, horizontal_alignment="left", gap="small"):
        for model_name, (label, confidence) in results.items():
            color, icon = SENTIMENT_STYLE[label]
            with st.container(border=True, width=200, key=f"model_card_{model_name}"):
                st.markdown(f"**{display_name(model_name)}**")
                with st.container(
                    horizontal=True, vertical_alignment="center", gap="small"
                ):
                    st.badge(label.capitalize(), icon=icon, color=color)
                    st.caption(f"{confidence:.0%}")


def render_comparison_chart(results: dict) -> None:
    st.subheader("Confidence by model", divider="gray")
    with st.container(border=True):
        chart_data = (
            pd.Series(
                {short_name(name): conf for name, (_, conf) in results.items()},
                name="confidence",
            )
            .sort_values()
            .to_frame()
        )
        st.bar_chart(
            chart_data,
            y="confidence",
            color="primary",
            x_label="",
            y_label="Confidence",
            horizontal=True,
        )


def _use_sample(language: str, pool: dict) -> None:
    st.session_state.review_text = random.choice(pool[language])


st.title("Code-Mix Sentiment Analyzer")
st.markdown(
    "Compare sentiment predictions across 9 models, for English, Bangla, Banglish, and code-switched reviews."
)

models = load_all_models()
available = list(models["classical"].keys()) + list(models["neural"].keys())
if models["bert"] is not None:
    available.append("bert")

if not available:
    st.error(
        "No trained models found. Run the training scripts first (see README.md).",
        icon=":material/error:",
    )
    st.stop()

performance = load_model_performance()

with st.sidebar:
    st.subheader("Models")
    selected_models = st.pills(
        "Models",
        options=available,
        default=available,
        format_func=display_name,
        selection_mode="multi",
        width="stretch",
        label_visibility="collapsed",
    )
    st.caption(f"{len(selected_models)} of {len(available)} active for analysis")

    if performance:
        soft_divider()
        st.subheader("Test accuracy", width="content")
        performance_df = pd.DataFrame(
            {
                "Model": [short_name(name) for name in performance],
                "Accuracy": list(performance.values()),
            }
        )
        st.dataframe(
            performance_df,
            hide_index=True,
            width="stretch",
            row_height=TABLE_ROW_HEIGHT,
            height=(len(performance_df) + 1) * TABLE_ROW_HEIGHT + 3,
            column_config={
                "Accuracy": st.column_config.ProgressColumn(
                    "Accuracy", format="percent", min_value=0, max_value=1
                ),
            },
        )

    soft_divider()
    st.caption("9 models · 150k real reviews.")
    st.page_link(
        "app_pages/architecture.py",
        label="How do these models work?",
        icon=":material/school:",
    )

sample_pool = load_sample_pool()
if sample_pool:
    st.caption("Try a random example:")
    with st.container(horizontal=True, gap="small"):
        for language in sample_pool:
            st.button(
                language.replace("_", "-").capitalize(),
                key=f"sample_{language}",
                on_click=_use_sample,
                args=(language, sample_pool),
            )

with st.form("analyze_form", border=False):
    st.text_area(
        "Product review",
        height=120,
        placeholder="e.g. product ta khub valo, kintu delivery slow silo",
        key="review_text",
    )
    submitted = st.form_submit_button(
        "Analyze", icon=":material/query_stats:", type="primary"
    )

text = st.session_state.get("review_text", "")

if submitted:
    if not text.strip():
        st.warning("Enter a review before analyzing.", icon=":material/warning:")
        st.stop()
    if not selected_models:
        st.warning(
            "Select at least one model in the sidebar.", icon=":material/warning:"
        )
        st.stop()

    with st.spinner("Running selected models…"):
        results = {
            name: predict_with_model(name, models, text) for name in selected_models
        }

    st.space("small")
    render_verdict(results)
    st.space("small")
    render_model_cards(results)
    st.space("small")
    render_comparison_chart(results)
else:
    st.info(
        "Enter a review above and click **Analyze**.", icon=":material/arrow_upward:"
    )
