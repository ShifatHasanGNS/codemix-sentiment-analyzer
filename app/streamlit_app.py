"""
Streamlit UI for the Code-Mix Sentiment Analyzer.

Lets a user type/paste a sentence in English, Bangla, Banglish, or
code-switched form, pick one or more trained models, and see each model's
predicted sentiment + confidence side by side, plus a comparison chart.

Run with: streamlit run app/streamlit_app.py

`streamlit run` starts a long-lived local dev server, so per CLAUDE.md's
"long-running commands" ground rule, implement this file, then hand the run
command to the human to launch in their own terminal (rather than starting/
backgrounding the server in-session) and ask them to confirm it loaded and
report any errors before considering this phase done.

Layout plan (fill in with real Streamlit calls):
- st.title / st.caption: project name + one-line description.
- st.sidebar: model selection (multiselect over N-Gram, BoW, TF-IDF, ANN,
  RNN, LSTM, Attention, Transformer, BERT) + maybe a language-form hint.
- Main area:
    - st.text_area for input sentence.
    - On submit: for each selected model, load its saved checkpoint/pipeline
      (cache with st.cache_resource so models aren't reloaded every run),
      run inference, and display label + confidence in st.columns.
    - Embed a Matplotlib bar chart (st.pyplot) comparing confidence scores
      across the selected models for the current input.
    - Optional expander: browse a few sample sentences per language
      condition from data/processed/test.csv.

TODO:
- load_all_models() -> dict[str, Any]
    Cached loader (st.cache_resource) that loads every trained
    model/vectorizer/tokenizer needed for inference, keyed by model name.
- predict_with_model(model_name: str, models: dict, text: str) -> (label: str, confidence: float)
    Dispatches to the right preprocessing + inference path depending on
    whether model_name is a classical baseline, a from-scratch neural model,
    or the BERT benchmark.
- render_comparison_chart(results: dict[str, float]) -> None
    Builds and displays the Matplotlib bar chart via st.pyplot.
- main()
    Wires up the layout described above.
"""


def load_all_models():
    raise NotImplementedError


def predict_with_model(model_name: str, models: dict, text: str):
    raise NotImplementedError


def render_comparison_chart(results: dict) -> None:
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
