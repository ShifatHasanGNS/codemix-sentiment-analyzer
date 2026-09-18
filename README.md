# Code-Mix Sentiment Analyzer

A comparative study of classical and neural NLP architectures — N-Gram, Bag-of-Words,
TF-IDF, Word2Vec, ANN, RNN, LSTM, Attention, Transformer (from scratch, PyTorch), and a
fine-tuned multilingual BERT — for sentiment classification of real e-commerce product
reviews written in English, Bangla, Banglish, and Bangla–English code-switched text.

**Data collection is 100% code-driven.** The corpus is downloaded and built entirely by
code from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset),
a large, real, public dataset of Bangla/English/Banglish/code-mixed product reviews — no
sentences are hand-written or translated. See `data/README.md` for the full pipeline.

See the full project proposal (`Code-Mix_Sentiment_Analyzer_Proposal.md`, shared separately)
for objectives, methodology, evaluation plan, and timeline.

**Text only** — this project works on review text only. BanglishRev's raw records may
include product-image fields alongside the reviews; those are dropped during dataset
construction and no image data is downloaded, stored, or used anywhere in this project.

This repository is currently an **unimplemented scaffold**: every module has its structure,
docstrings, and function/class signatures in place as comments/TODOs, with no logic filled
in yet. See `CLAUDE.md` for a task-by-task implementation guide.

## Project Structure

```
codemix-sentiment-analyzer/
├── data/                   # Raw and processed dataset (git-ignored except structure)
│   ├── raw/                 # Original collected/written sentences
│   └── processed/           # cv_pool.csv (k-fold CV pool) + test.csv (held-out)
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
│   └── streamlit_app.py        # Interactive UI: input text, pick model(s), compare
├── scripts/
│   ├── build_dataset.py        # CLI entry point to assemble the dataset
│   └── run_pipeline.py         # CLI entry point to train/evaluate everything
├── notebooks/                # Scratch/exploratory notebooks (git-ignored contents)
├── results/                  # Saved metrics tables, charts, error analysis reports
├── models_saved/             # Trained model checkpoints (git-ignored)
├── tests/                    # Lightweight smoke tests
├── requirements.txt
├── .gitignore
├── README.md
└── CLAUDE.md                 # Segmented task guide for AI-assisted implementation
```

## Why Streamlit (not Gradio)

Streamlit was chosen over Gradio because the UI needs to show **several models' predictions
side by side with a comparison chart** in one coherent dashboard layout (columns, tabs,
embedded Matplotlib figures). Streamlit's layout primitives (`st.columns`, `st.tabs`,
`st.sidebar`, native chart embedding) fit a multi-model comparison view more naturally than
Gradio, which is optimized around a single input → single output demo pattern. Both are
pure-Python and require no separate frontend code, so the switch costs nothing.

## Getting Started

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

python -m nltk.downloader punkt stopwords words      # one-time NLTK data download

# 1. Build the dataset (once src/data/* is implemented)
python scripts/build_dataset.py

# 2. Train + evaluate everything (once src/training/* is implemented)
python scripts/run_pipeline.py

# 3. Launch the UI
streamlit run app/streamlit_app.py
```

The `nltk.downloader`, `build_dataset.py`, `run_pipeline.py`, and `streamlit run` steps
above are all long-running (network download, model training, or a persistent dev server).
If an AI assistant is implementing this project, it should write the code for each phase
and then hand these commands to you to run yourself, pausing until you confirm they've
finished — see the "long-running commands" ground rule in `CLAUDE.md`.

## Implementation Order

Follow the phases in `CLAUDE.md` — each phase only touches a small, self-contained set of
files, so it can be implemented (and reviewed) independently of the others.
