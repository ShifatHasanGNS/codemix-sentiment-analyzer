"""
Entry point for the Code-Mix Sentiment Analyzer's Streamlit UI. Sets shared
page config/theme, then hands off to whichever page is selected -- see
app_pages/analyzer.py (predictions) and app_pages/architecture.py (a
layperson's guide to how each model works).

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from app_lib import style

st.set_page_config(page_title="Code-Mix Sentiment Analyzer", page_icon=":material/reviews:", layout="wide")
st.html(style(is_dark=st.context.theme.type == "dark"))

page = st.navigation(
    [
        st.Page("app_pages/analyzer.py", title="Analyzer", icon=":material/reviews:", default=True),
        st.Page("app_pages/architecture.py", title="How models work", icon=":material/school:"),
    ],
    position="top",
)
page.run()
