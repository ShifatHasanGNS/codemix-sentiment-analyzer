# CLAUDE.md — Implementation Guide for This Repository

This file exists so that an AI assistant (or a human) can pick up **one task at a time**
without needing to read the entire repository first. Each phase below is self-contained:
it names exactly which files to open, what they depend on, and what "done" looks like.
Work through phases in order — later phases assume earlier ones are finished — but within
a single session, only load the files listed for the phase you are doing.

## Ground Rules (apply to every phase)

- **Framework:** PyTorch only. Never introduce TensorFlow/Keras.
- **No pretrained weights** inside `src/models/ann.py` … `src/models/transformer.py` —
  those five are trained from random initialization. The only pretrained component in the
  whole project is `src/models/bert_model.py`.
- **Text-only project — no image data, anywhere.** This is a pure NLP project. BanglishRev
  is an e-commerce review dataset, so the raw source records may include product-image
  URLs/fields alongside the review text — `flatten_reviews()` must drop any such
  image-related fields and keep only the text columns listed in the schema
  (`data/README.md`). Never add code that downloads, stores, displays, or otherwise
  processes images (product photos, review screenshots, etc.) in any phase — not in the
  dataset pipeline, not in features/models, not in the Streamlit app. Don't add
  image-handling libraries (Pillow, OpenCV, torchvision transforms for images, etc.) to
  `requirements.txt`.
- **Long-running commands are handed off to the human, not run by the assistant.** Some
  commands in this project are slow or resource-heavy: fetching BanglishRev from Hugging
  Face (Phase 1, GBs of raw JSON), training the classical/neural/BERT models (Phases 5, 7,
  8, 11), and running the Streamlit dev server (Phase 12). For any such command:
  1. Finish implementing the code for the current phase first.
  2. Print/state the exact command to run (e.g. `python scripts/build_dataset.py`,
     `python scripts/run_pipeline.py`, `streamlit run app/streamlit_app.py`).
  3. Do **not** execute it yourself in the session — end your turn there and ask the human
     to run it in their own terminal.
  4. Wait for the human to report back (e.g. "done", "training finished", paste output/
     errors) before continuing to the next phase or checking the "Done when" condition.
     This applies whenever a command downloads a non-trivial amount of data, installs
     packages, trains a model, or starts a long-lived server — not to quick, near-instant
     sanity checks (e.g. `python -c "from src import config"`, a tiny inline unit test on a
     handful of rows), which are fine to run directly.
- **Language scope:** every dataset example is auto-tagged (not manually assigned) into one
  of 4 language conditions — English, Bangla, Banglish, Code-switched — see Phase 1. Any
  function that touches raw text should be agnostic to which condition it's given (don't
  hardcode English-only assumptions like whitespace-only tokenization everywhere; the custom
  tokenizer must handle Bangla script + Romanized Bangla too).
- **Data collection is 100% code-driven.** The corpus comes from the real, public
  BanglishRev dataset, fetched, labeled, tagged, filtered, and split entirely by code — never
  write, translate, or hand-label example sentences. See Phase 1.
- **Labels:** 3-class sentiment — `positive`, `negative`, `neutral`.
- **Config, not hardcoding:** paths, hyperparameters, and file locations belong in
  `src/config.py`. Other modules import from it rather than redefining constants.
- **Determinism:** any training script should call `src.utils.seed.set_seed(...)` first.
- **Every module currently contains only docstrings/comments/TODOs and stub signatures
  (`pass` / `raise NotImplementedError`).** Implementing a phase means filling in real logic
  behind those existing signatures — keep the public function/class names and signatures
  unless you have a good reason to change them (and if you do, update callers in the same
  phase).

---

## Phase 0 — Environment & Config

**Files:** `requirements.txt`, `src/config.py`, `src/utils/seed.py`, `src/utils/io_utils.py`
**Depends on:** nothing.
**Goal:** Fill in `config.py` with actual paths/hyperparameters/label list; implement the
seeding helper and basic I/O helpers (load/save JSON, CSV, pickle).
**Done when:** `python -c "from src import config"` runs with no errors and every constant
referenced by later phases exists.

## Phase 1 — Dataset Construction (fully automated)

**Files:** `src/data/dataset_builder.py`, `src/config.py` (dataset constants), `data/README.md`,
`scripts/build_dataset.py`
**Depends on:** Phase 0.
**Goal:** Download BanglishRev (`download_banglishrev`), flatten it to one row per review,
**keeping text/metadata columns only — drop any image/photo fields present in the raw
source** (`flatten_reviews`), derive a 3-class sentiment label from each review's star rating
(`map_rating_to_label`), auto-detect the language condition of each review
(`detect_language_condition` — Unicode-script + English-dictionary heuristic), restrict to a
short list of universally familiar product categories (`filter_familiar_categories`), clean
and deduplicate (`clean_and_dedupe`), then subsample down to a small, balanced corpus
(`subsample_balanced`). Write the result to `data/raw/dataset.csv`, then produce stratified
train/val/test splits into `data/processed/`. See `data/README.md` for the exact schema
(text-only — no image columns).
**Note:** the first run needs internet access and downloads a multi-GB raw file from Hugging
Face (`src.config.HF_DATASET_ID`) — per the "long-running commands" ground rule above,
implement the code, then hand `python scripts/build_dataset.py` to the human to run and wait
for confirmation it finished before checking the "Done when" condition. Nothing in this phase
should involve writing or translating sentences by hand, and nothing in this phase should
fetch or persist images.
**Done when:** `python scripts/build_dataset.py` produces `data/processed/{train,val,test}.csv`
matching the documented schema, with roughly balanced labels and all 4 auto-detected language
conditions represented in every split.

## Phase 2 — Tokenizer & Preprocessing

**Files:** `src/data/tokenizer.py`, `src/data/preprocess.py`
**Depends on:** Phase 1.
**Goal:** Manual tokenizer covering English, Bangla script, and Banglish/code-switched text;
shared cleaning utilities (lowercasing for Latin script, punctuation handling, optional
stopword removal) used by every downstream feature/model.
**Done when:** the tokenizer round-trips all 4 language forms from `data/processed/train.csv`
without crashing and produces non-trivial token lists.

## Phase 3 — Classical Features (N-Gram / BoW / TF-IDF)

**Files:** `src/features/classical_features.py`
**Depends on:** Phase 2.
**Goal:** Fit/transform functions for N-Gram counts, Bag-of-Words, and TF-IDF vectors over
the tokenized corpus.
**Done when:** each `fit_*`/`transform_*` pair produces a feature matrix whose row count
matches the input example count.

## Phase 4 — Word2Vec Embeddings

**Files:** `src/features/embeddings.py`
**Depends on:** Phase 2.
**Goal:** Train a Word2Vec model (gensim) on the tokenized training corpus; provide a
lookup function returning an embedding matrix aligned to a given vocabulary.
**Done when:** embeddings can be fetched for the tokenizer's vocabulary with a documented
fallback for out-of-vocabulary tokens.

## Phase 5 — Classical Baseline Training

**Files:** `src/training/train_classical.py`
**Depends on:** Phases 3, 0.
**Goal:** Train Naive Bayes / Logistic Regression on top of N-Gram, BoW, and TF-IDF
features; save fitted vectorizers + classifiers to `models_saved/`.
**Done when:** running the script produces 3 saved baseline models and prints train/val
accuracy for each. These are fast (CPU, small corpus), so it's fine to run the script
directly rather than handing it off.

## Phase 6 — From-Scratch Neural Models

**Files:** `src/models/ann.py`, `src/models/rnn.py`, `src/models/lstm.py`,
`src/models/attention.py`, `src/models/transformer.py`
**Depends on:** Phases 2, 4, 0.
**Goal:** Implement each architecture as a `torch.nn.Module` per the course material —
ANN over pooled embeddings, RNN/LSTM over token sequences, an attention layer usable on top
of the RNN/LSTM, and a small Transformer encoder with positional encoding — each ending in
a 3-class classification head.
**Done when:** a forward pass on a dummy batch of the correct shape returns logits of shape
`(batch_size, 3)` for every model.

## Phase 7 — Neural Model Training

**Files:** `src/training/train_neural.py`
**Depends on:** Phase 6.
**Goal:** Shared training loop (data loading, optimizer, loss, checkpointing) parameterized
to run against any of the five models in `src/models/`.
**Done when:** the script can train each of the five models end-to-end on
`data/processed/train.csv` and save the best checkpoint (by validation accuracy) to
`models_saved/`. Training five models is the kind of long-running work covered by the
ground rule above — implement the script, then hand the training command to the human and
wait rather than running the full training loop yourself in-session.

## Phase 8 — Pretrained BERT Benchmark

**Files:** `src/models/bert_model.py`, `src/training/train_bert.py`
**Depends on:** Phase 1, 0.
**Goal:** Load a pretrained multilingual BERT (mBERT or XLM-R) via `transformers`,
fine-tune on the same train/val split, save the fine-tuned checkpoint.
**Done when:** fine-tuning runs to completion and validation accuracy is logged. BERT
fine-tuning is the slowest step in the project (even on a small subsample) — implement the
script, then hand the run command to the human per the "long-running commands" ground rule
rather than fine-tuning in-session.

## Phase 9 — Sequence Generation Bonus

**Files:** `src/generation/text_completion.py`
**Depends on:** Phase 7 (needs a trained LSTM or Transformer checkpoint).
**Goal:** Small next-word/next-sentence completion function built on the trained
LSTM/Transformer, used only for qualitative demonstration.
**Done when:** given a prompt string, the function returns a short generated continuation.

## Phase 10 — Evaluation & Error Analysis

**Files:** `src/evaluation/metrics.py`, `src/evaluation/error_analysis.py`
**Depends on:** Phases 5, 7, 8.
**Goal:** Compute accuracy/precision/recall/F1 per model, broken down by language condition
(English/Bangla/Banglish/code-switched); collect representative misclassified examples per
model into a readable report.
**Done when:** `results/` contains a metrics table (CSV/JSON) and an error-analysis report
covering every trained model.

## Phase 11 — Pipeline Orchestration

**Files:** `scripts/run_pipeline.py`
**Depends on:** Phases 1–10 (or whichever subset exists so far).
**Goal:** A single CLI entry point that runs dataset building → feature/embedding prep →
training (classical, neural, BERT) → evaluation, with flags to run a subset (e.g.
`--models ann,lstm` or `--skip-bert` for a faster local pass).
**Done when:** `python scripts/run_pipeline.py --help` documents all flags and a full run
populates `models_saved/` and `results/`. `--help` is quick and fine to run directly; an
actual full (or even partial) pipeline run is long-running — hand the command to the human
per the ground rule above and wait for it to finish before moving to evaluation/UI work.

## Phase 12 — Streamlit UI

**Files:** `app/streamlit_app.py`
**Depends on:** Phase 11 (needs trained models to load).
**Goal:** Text input box, model-selection widget, per-model prediction + confidence display,
a Matplotlib confidence-comparison bar chart, and an optional sample-sentence browser by
language condition.
**Done when:** `streamlit run app/streamlit_app.py` loads without error and produces a
prediction for a typed sentence using at least one trained model. `streamlit run` starts a
long-lived dev server — hand this command to the human to run in their own terminal per the
ground rule above rather than starting/backgrounding it yourself; ask them to confirm it
loaded and to paste back any error before you consider this phase done.

## Phase 13 — Tests & Polish

**Files:** `tests/test_smoke.py`
**Depends on:** whichever phases are implemented.
**Goal:** Minimal smoke tests (import checks, tiny forward passes, tokenizer round-trip) —
not full unit-test coverage, just enough to catch broken imports/signatures.

---

## Quick Reference: "I need to work on X, what do I open?"

| Task                            | Files to open                                                                                               |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Dataset schema / content        | `data/README.md`, `src/data/dataset_builder.py`                                                             |
| Tokenization behavior           | `src/data/tokenizer.py`, `src/data/preprocess.py`                                                           |
| A specific model's architecture | the single file in `src/models/` for that model                                                             |
| Training loop bugs (neural)     | `src/training/train_neural.py`, `src/models/config.py`                                                      |
| Metrics / results formatting    | `src/evaluation/metrics.py`, `src/evaluation/error_analysis.py`                                             |
| UI behavior                     | `app/streamlit_app.py` only (it imports from `src/` but you rarely need to edit `src/` for UI-only changes) |

Do not open files outside the current phase's list unless a stub function you're
implementing explicitly imports from them.
