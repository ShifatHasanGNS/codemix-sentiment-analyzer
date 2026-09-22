"""Entry point: sets theme, then dispatches to the selected page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from app_lib import style

st.set_page_config(
    page_title="Code-Mix Sentiment Analyzer",
    page_icon=":material/reviews:",
    layout="wide",
)
st.html(style(is_dark=st.context.theme.type == "dark"))

page = st.navigation(
    [
        st.Page(
            "app_pages/analyzer.py",
            title="Analyzer",
            icon=":material/reviews:",
            default=True,
        ),
        st.Page(
            "app_pages/architecture.py",
            title="How models work",
            icon=":material/school:",
        ),
    ],
    position="top",
)
page.run()
