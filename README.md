# Code-Mix Sentiment Analyzer

A comparative study of classical and neural NLP architectures — N-Gram, Bag-of-Words,
TF-IDF, Word2Vec, ANN, RNN, LSTM, Attention, Transformer (from scratch, PyTorch), and a
fine-tuned multilingual BERT — for sentiment classification of real e-commerce product
reviews written in English, Bangla, Banglish, and Bangla–English code-switched text.

Course project for **Natural Language Processing Laboratory (CSE 4122)**, Department of
Computer Science and Engineering, Khulna University of Engineering and Technology.

**Authors:** Md. Shifat Hasan (2107067), Siam Basher (2107078)

**Data collection is 100% code-driven.** The corpus is downloaded and built entirely by
code from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset),
a large, real, public dataset of Bangla/English/Banglish/code-mixed product reviews — no
sentences are hand-written or translated. See `data/README.md` for the full pipeline.

**Text only** — this project works on review text only. BanglishRev's raw records may
include product-image fields alongside the reviews; those are dropped during dataset
construction and no image data is downloaded, stored, or used anywhere in this project.

The full pipeline is implemented end to end: a 150,000-review label-balanced corpus,
nine trained sentiment models (three classical + logistic regression, five from-scratch
PyTorch neural networks, and a fine-tuned multilingual BERT), stratified 5-fold
cross-validation, a held-out test evaluation broken down by language condition, and an
interactive Streamlit app for live inference and model comparison. See
[`docs/report/report.pdf`](docs/report/report.pdf) for the full write-up (methodology,
results, discussion, limitations).

## Project Structure

```
codemix-sentiment-analyzer/
├── data/
│   ├── raw/                 # Downloaded BanglishRev cache (git-ignored, multi-GB)
│   └── processed/           # cv_pool.csv (k-fold CV pool) + test.csv (held-out) -- tracked
├── src/
│   ├── config.py             # Central paths, constants, hyperparameters
│   ├── data/                 # Dataset building, tokenizer, preprocessing
│   ├── features/             # N-Gram/BoW/TF-IDF features + Word2Vec embeddings
│   ├── models/                # ANN, RNN, LSTM, Attention, Transformer, BERT wrapper
│   ├── generation/            # Small autoregressive text-completion demo
│   ├── training/               # Training loops for classical / neural / BERT models
│   ├── evaluation/             # Metrics computation + qualitative error analysis
│   └── utils/                  # Seeding, I/O helpers
├── app/
│   ├── streamlit_app.py        # Entry point: theme, navigation between pages
│   ├── app_lib.py               # Shared model loading, inference, styling helpers
│   ├── app_pages/                # Analyzer page + Model Architecture explainer page
│   └── diagrams.py               # Auto-generated per-model pipeline diagrams
├── scripts/
│   ├── build_dataset.py        # CLI entry point to assemble the dataset
│   └── run_pipeline.py         # CLI entry point to train/evaluate everything
├── results/                   # Saved metrics tables, charts, error analysis reports
├── models_saved/              # Trained model checkpoints
├── docs/
│   ├── idea/                    # Original project proposal PDFs
│   └── report/                  # LaTeX source + compiled PDF academic report
├── .streamlit/config.toml      # Light/dark/system theme
├── tests/                     # Lightweight smoke tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Why Streamlit (not Gradio)

Streamlit was chosen over Gradio because the UI needs to show **several models' predictions
side by side with a comparison chart** in one coherent dashboard layout (columns, tabs,
embedded Matplotlib figures). Streamlit's layout primitives (`st.columns`, `st.tabs`,
`st.sidebar`, native chart embedding) fit a multi-model comparison view more naturally than
Gradio, which is optimized around a single input → single output demo pattern. Both are
pure-Python and require no separate frontend code, so the switch costs nothing.

## Getting Started

Trained model checkpoints and the processed dataset (`data/processed/cv_pool.csv`,
`data/processed/test.csv`) are already committed to this repository, so the app runs
directly after cloning without retraining anything:

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

The fine-tuned BERT checkpoint is excluded from the repo (over GitHub's file-size limit);
run `python -m src.training.train_bert` after cloning if you want it too. To rebuild the
dataset or retrain everything from scratch instead:

```bash
python scripts/build_dataset.py      # rebuild the 150k-row corpus (downloads BanglishRev, several minutes)
python scripts/run_pipeline.py       # retrain + re-evaluate all 9 models (long-running)
```
