# Project Details

Technical reference for the Code-Mix Sentiment Analyzer: what each file does, how data
flows through the pipeline, and why key design decisions were made. See the top-level
[`README.md`](../README.md) for the project summary and [`data/README.md`](../data/README.md)
for the dataset schema, and [`docs/report/report.pdf`](report/report.pdf) for the full
academic write-up (methodology, results, discussion).

## Contents

- [1. Overview](#1-overview)
- [2. End-to-end workflow](#2-end-to-end-workflow)
- [3. `src/config.py`](#3-srcconfigpy)
- [4. `src/data/`](#4-srcdata)
- [5. `src/features/`](#5-srcfeatures)
- [6. `src/models/`](#6-srcmodels)
- [7. `src/training/`](#7-srctraining)
- [8. `src/evaluation/`](#8-srcevaluation)
- [9. `src/generation/`](#9-srcgeneration)
- [10. `src/utils/`](#10-srcutils)
- [11. `app/` (Streamlit UI)](#11-app-streamlit-ui)
- [12. `scripts/`](#12-scripts)
- [13. Outputs (`models_saved/`, `results/`)](#13-outputs-models_saved-results)

---

## 1. Overview

The project compares nine sentiment-classification techniques on a 150,000-review corpus
of real Bangla / English / Banglish / code-switched e-commerce product reviews
(sourced from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset)):

| Family | Models |
|---|---|
| Classical (Logistic Regression + hand-built features) | N-Gram, Bag-of-Words, TF-IDF |
| Neural, from scratch (PyTorch) | ANN, RNN, LSTM, Attention, Transformer |
| Pretrained | Fine-tuned `bert-base-multilingual-cased` (mBERT) |

Every model is evaluated on the same held-out test set, broken down by language
condition, to answer the project's central question: **which technique is most robust to
code-switched input?**

```
codemix-sentiment-analyzer/
├── data/                 raw + processed corpus (see data/README.md)
├── src/
│   ├── config.py         central paths, constants, hyperparameters
│   ├── data/              dataset building, tokenizer, preprocessing
│   ├── features/          N-Gram/BoW/TF-IDF + Word2Vec embeddings
│   ├── models/             ANN/RNN/LSTM/Attention/Transformer/BERT architectures
│   ├── training/            training loops (classical / neural / BERT)
│   ├── evaluation/           metrics + qualitative error analysis
│   ├── generation/            bonus autoregressive text-completion demo
│   └── utils/                 seeding, I/O helpers
├── app/                  Streamlit inference + architecture-explainer UI
├── scripts/              CLI entry points (build_dataset.py, run_pipeline.py)
├── models_saved/         trained checkpoints
├── results/              metrics tables, chart, error-analysis report
└── docs/                 this file, project idea PDFs, LaTeX report
```

---

## 2. End-to-end workflow

```
                 python scripts/build_dataset.py
                              │
   ┌──────────────────────────┴───────────────────────────────┐
   │ src/data/dataset_builder.py                              │
   │  download_banglishrev → flatten_reviews →                │
   │  map_rating_to_label → detect_language_condition →       │
   │  filter_familiar_categories (no-op) → clean_and_dedupe → │
   │  subsample_balanced → assemble_dataset (raw/dataset.csv) │
   │  → create_cv_splits (processed/cv_pool.csv, test.csv)    │
   └──────────────────────────┬───────────────────────────────┘
                              │
                 python scripts/run_pipeline.py
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
 train_classical.py    train_neural.py         train_bert.py
 (ngram/bow/tfidf,      (ANN/RNN/LSTM/           (mBERT fine-tune,
  5-fold CV + refit      Attention/Transformer,   single held-out
  on full CV pool)       5-fold CV + refit)       split, no CV)
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                              │
                    run_evaluation() in run_pipeline.py
                              │
        predicts every available checkpoint on test.csv,
        breaks down accuracy/precision/recall/F1 by language
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
 metrics_comparison    metrics_comparison.png   error_analysis.md
 .csv (+ cv_summary     (grouped bar chart)      (misclassified
 .csv if CV was run)                             examples + text-
                                                  completion demo)
                              │
                  streamlit run app/streamlit_app.py
                  (loads models_saved/* for live inference
                   + an architecture explainer page)
```

Training is long-running once real models are selected, so `build_dataset.py` and
`run_pipeline.py` are meant to be run directly in a terminal rather than driven inline by
an AI session — `--help`, and re-running only the evaluation step against checkpoints
that already exist in `models_saved/`, are the exceptions safe to run either way.

---

## 3. `src/config.py`

Central configuration; every other module imports paths/constants/hyperparameters from
here instead of hardcoding them, so changing one value updates the whole pipeline.

- **Paths**: `PROJECT_ROOT` (resolved from `__file__`) and derived `DATA_DIR`,
  `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`, `MODELS_SAVED_DIR`, `RESULTS_DIR`. Loads `.env`
  via `load_dotenv` early (before any network call) so `HF_TOKEN` is available to
  `huggingface_hub`'s `hf_hub_download` for authenticated, higher-rate-limit downloads;
  the token is never logged or printed.
- **Dataset constants**: `LABELS = [positive, negative, neutral]`,
  `LANGUAGE_CONDITIONS = [english, bangla, banglish, code_switched]`,
  `TEST_HOLDOUT_SPLIT = 0.10`, `N_FOLDS = 5`, `RANDOM_SEED = 42`.
- **BanglishRev constants**: `HF_DATASET_ID`, `RATING_TO_LABEL` (1-2★→negative,
  3★→neutral, 4-5★→positive). `ALLOWED_CATEGORIES = None` disables category filtering
  by design — an earlier 5-bucket "familiar categories" keyword allowlist was found (by
  sampling the raw corpus) to exclude ~74% of it, including everyday items (Smartwatches,
  T-Shirts, Shampoo, Toothpaste, Diapers, Coffee) just as "universally familiar" as the
  original 5 buckets, so the whole corpus is used unfiltered.
  `BANGLA_UNICODE_RANGE = (0x0980, 0x09FF)` and `LANGUAGE_DETECTION_MIN_TOKEN_RATIO = 0.6`
  drive the language-detection heuristic. `TARGET_DATASET_SIZE = 150_000` sits
  comfortably under the ~57k natural ceiling for the scarcest label ("neutral");
  code-switched reviews are intrinsically rare (~1,600 of ~1.74M raw reviews) and are
  used at natural scale via `subsample_balanced()`'s best-effort-then-redistribute logic
  rather than forced into an equal bucket.
- **Shared feature/model hyperparameters**: `MAX_SEQUENCE_LENGTH=64`,
  `EMBEDDING_DIM=100`, `BATCH_SIZE=64`, `LEARNING_RATE=1e-3`, `NUM_EPOCHS=4` (chosen
  because at ~108k rows/fold each epoch has ~67x more gradient updates than a small
  corpus would, keeping 5-fold × 5-model CV to roughly an hour on CPU),
  `CLASSICAL_MAX_FEATURES=20_000`, `MIN_TOKEN_FREQ=3` (prevents typo/rare tokens from
  bloating the embedding table at 100k+ documents). Architecture constants
  (`DROPOUT=0.3`, `HIDDEN_DIM=64`, `ANN_HIDDEN_DIMS=[64,32]`, `LSTM_NUM_LAYERS=1`,
  `LSTM_BIDIRECTIONAL=True`, `TRANSFORMER_NUM_HEADS=4`, `TRANSFORMER_NUM_LAYERS=2`,
  `TRANSFORMER_FF_DIM=128`) are kept deliberately small so training stays reasonably
  fast on CPU at this corpus's scale.
- **Text-completion bonus**: `TEXT_COMPLETION_LM_EPOCHS=3`,
  `TEXT_COMPLETION_TRAIN_SUBSAMPLE_SIZE=5_000` — bounded since this is a non-scored
  qualitative demo, not worth full-corpus training time.
- **Pretrained BERT benchmark**: `BERT_MODEL_NAME = "bert-base-multilingual-cased"`,
  `BERT_MAX_SEQUENCE_LENGTH=96` (longer than the word-level tokenizer's budget since
  mBERT subword-tokenizes Bangla into more pieces per word), `BERT_BATCH_SIZE=8`
  (smaller than the neural models' batch size since mBERT's ~178M params are far more
  expensive per step on CPU), `BERT_NUM_EPOCHS=2`. BERT deliberately uses a single
  held-out split rather than 5-fold CV (an estimated 3-5+ extra CPU hours for little
  reporting benefit), with `BERT_TRAIN_SUBSAMPLE_SIZE=5_000` /
  `BERT_VAL_SUBSAMPLE_SIZE=1_000` drawn non-overlapping from the CV pool, evaluated
  against the same held-out `test.csv` as every other model.

---

## 4. `src/data/`

### `dataset_builder.py`
Assembles the corpus entirely by code from BanglishRev, text-only by construction. The
source repo stores review text and images as separate files (`reviews v1.json` ~1.9GB
plus 109 "Review Images N.zip" archives); `download_banglishrev(cache_dir=None)` fetches
only the exact JSON filename via `hf_hub_download`, so image archives are never touched.

- `flatten_reviews(raw_data)` — accepts either a file path or an already-parsed list
  (the latter used directly by tests without network access). Reads only
  `Current Rating`, `Review Content`, and category fields (`Category`/`Parent Category`/
  `Root Category`) out of each review dict, ignoring engagement/moderation fields
  (Buyer ID, Likes, Dislikes, Reply, Images). Drops rows with unparseable/out-of-range
  ratings or empty text here, since this is the one place still holding raw/untrusted
  values — downstream code can then assume `rating` is a clean int.
- `map_rating_to_label(rating)` — applies `config.RATING_TO_LABEL`.
- `_english_word_set()` — lazily loads/caches NLTK's `words` corpus lowercased; drops
  ~165 one/two-letter entries (abbreviations, archaic words, stray letters like "ar",
  "s", "m") that coincidentally match common romanized-Bangla syllables and would
  inflate false "english" matches on Banglish text, except for a curated
  `_SHORT_ENGLISH_WORDS` allowlist.
- `_script_of(token)` — classifies a token as bangla/latin/other by its first matching
  character, using `str.isalpha()` (not ASCII a-z) so stylized Unicode letters — e.g. a
  real review writing "onak kharap" as "𝒐𝒏𝒂𝒌 𝒌𝒉𝒂𝒓𝒂𝒑" in Mathematical Bold Italic — still
  register as latin script rather than contributing no signal.
- `detect_language_condition(text)` — computes bangla/latin token ratios against
  `LANGUAGE_DETECTION_MIN_TOKEN_RATIO`: ≥threshold Bangla → `"bangla"`; ≥threshold Latin
  with most tokens real English words → `"english"`, else `"banglish"`; neither
  dominates → `"code_switched"`; zero alphabetic tokens (emoji/digits only) → defaults
  to `"english"` as a harmless fallback. Symmetric thresholding avoids misclassifying an
  English review with one stray Bangla character as code-switched.
- `filter_familiar_categories(df, allowed_categories)` — a no-op when
  `allowed_categories` is falsy (the project default, see `config.ALLOWED_CATEGORIES`);
  otherwise does a case-insensitive substring match. Kept for traceability/override.
- `clean_and_dedupe(df)` — strips control characters, enforces
  `MIN_REVIEW_LENGTH_CHARS`, drops "content-free" reviews (zero alphabetic tokens — e.g.
  pure mojibake like "??? ??????" seen in this corpus), since such rows carry no
  learnable signal and would otherwise become an all-padding input downstream that
  drives `AdditiveAttention`'s softmax to `0/0 = NaN`, permanently corrupting model
  weights via backprop. Also drops exact-text duplicates.
- `subsample_balanced(df, target_size, per_group_cap=None)` — a two-level, label-first
  design: `target_size` splits evenly across the 3 labels, and only *within* each label
  does a best-effort-then-redistribute pass run across its 4 language groups (so
  code-switched contributes everything it has, and any shortfall is filled from other
  language conditions *of that same label*). This matters at scale because the raw
  label distribution is heavily skewed (~82% positive) — a single flat redistribution
  across all 12 (label, language) cells would let positive's leftover pool dominate and
  silently break label balance (verified empirically before choosing this design).
- `assemble_dataset(cache_dir=None)` — orchestrates the full pipeline above, computes
  `per_group_cap` as 3× each language condition's even split of its label quota (so
  plentiful conditions like english/bangla/banglish can absorb scarce code-switched's
  shortfall without an overly tight ceiling), assigns `id` values like `cm000001`, and
  writes `data/raw/dataset.csv`.
- `create_cv_splits(df, n_folds=None, test_holdout=None)` — stratifies by
  `label + "_" + language` jointly where possible (falling back to label-only
  stratification if a group is too small — a real risk for scarce code_switched),
  carves off a held-out test split via `train_test_split`, and assigns a `fold` column
  (0..n_folds-1) to the rest via `StratifiedKFold`. Writes `data/processed/test.csv` and
  `data/processed/cv_pool.csv`.
- `get_fold(cv_pool_df, fold_index)` — splits a fold-tagged pool into that fold's
  `(train_df, val_df)` pair.

Depends on `src.config`, `src.utils.io_utils`. Used by `scripts/build_dataset.py` and
(`get_fold`) by both training loops.

### `preprocess.py`
Shared text-cleaning utilities used before tokenization/feature extraction.

- `_BANGLA_STOPWORDS` / `_BANGLISH_STOPWORDS` — small hand-curated frozensets of common
  function words; NLTK has no offline Bangla/Banglish stopword corpus, so these lists
  aren't exhaustive but strip the highest-frequency noise words per language condition.
- `_english_stopword_set()` — lazily loads/caches NLTK's English stopwords, raising a
  clear `LookupError` with a download hint if the corpus is missing.
- `clean_text(text, lowercase=True)` — normalizes smart quotes, collapses whitespace,
  and (if `lowercase`) lowercases only Latin-script runs, leaving Bangla script
  untouched since it has no case distinction.
- `remove_stopwords(tokens, language)` — validates `language` against
  `config.LANGUAGE_CONDITIONS`, then filters against the union of applicable stopword
  sets; for `"code_switched"` text all three sets (English, Bangla, Banglish) apply
  since either language's function words may appear.
- `load_split(split_name)` — loads `data/processed/{split_name}.csv` for `"cv_pool"`
  (has a `fold` column) or `"test"` (the untouched 10% held-out set).

Depends on `src.config`, `src.utils.io_utils.load_csv`. Used throughout training,
evaluation, and `scripts/run_pipeline.py`.

### `tokenizer.py`
`CodeMixTokenizer` — a manually implemented, script-aware tokenizer for mixed
Bangla-English vocabulary, handling all 4 language conditions. Used by classical
features, Word2Vec training, and every from-scratch neural model's input pipeline —
**not** used by the BERT benchmark, which has its own subword tokenizer.

- `_TOKEN_RE` matches one of: a run of Bangla-script letters, a run of other-script
  "word" letters (`_LETTER_RUN`, with internal apostrophes for contractions like
  "don't"), a run of digits, or any single other character (punctuation/emoji) as its
  own token. `_LETTER_RUN` uses `\w`'s general Unicode letter class rather than plain
  `[A-Za-z]` so stylized Unicode letters survive tokenization (the same "onak kharap" →
  Mathematical Bold Italic example as in `dataset_builder.py`) — an ASCII-only pattern
  would silently drop such a review to zero tokens, producing an all-padding input that
  drives attention's softmax to NaN (see `src/models/attention.py`'s defensive masking
  for anything that still slips through at inference).
- Class state: `token_to_id` / `id_to_token` dicts seeded with `PAD_TOKEN="<pad>"`
  (id 0) and `UNK_TOKEN="<unk>"` (id 1).
- `tokenize(text)` — applies `_TOKEN_RE`, optionally drops non-word tokens if
  `remove_punctuation`, optionally lowercases ASCII-only tokens (leaving Bangla script
  as-is) if `lowercase`.
- `build_vocab(corpus, min_freq=1)` — counts token frequencies, keeps tokens with
  count ≥ `min_freq`, sorts by (frequency desc, token asc) for deterministic ordering
  across runs given the same corpus, rebuilds `token_to_id` / `id_to_token`.
- `encode(text, max_length)` — maps tokens to ids (`UNK_TOKEN` id for OOV), truncates or
  pads to `max_length`.
- `vocab_size` property; `save(path)` / `load(path)` persist/restore full tokenizer
  state as JSON via `src.utils.io_utils`.

Depended on by `src/features/classical_features.py`, `src/features/embeddings.py`, all
`src/models/*` except BERT, `src/training/train_neural.py`, and
`scripts/run_pipeline.py`.

---

## 5. `src/features/`

### `classical_features.py`
Builds N-Gram / Bag-of-Words / TF-IDF vectorizers, all using
`CodeMixTokenizer.tokenize` as the `tokenizer=` callable (with `token_pattern=None` to
disable sklearn's default regex tokenizer), so classical and neural pipelines tokenize
identically.

- `fit_ngram_vectorizer(corpus, n_range=(1,2))` → `CountVectorizer` with
  `ngram_range=(1,2)`.
- `fit_bow_vectorizer(corpus)` → `CountVectorizer` with `ngram_range=(1,1)`.
- `fit_tfidf_vectorizer(corpus)` → `TfidfVectorizer` with `ngram_range=(1,2)`.
- All capped at `max_features=config.CLASSICAL_MAX_FEATURES` (20,000).
- `transform(vectorizer, corpus)` → sparse feature matrix.

Each vectorizer is paired with a `LogisticRegression` classifier in
`src/training/train_classical.py`. Used at inference time by `scripts/run_pipeline.py`
and `app/app_lib.py`.

### `embeddings.py`
Word2Vec embeddings trained on the project's own corpus via `gensim`, used as the input
representation for the ANN/RNN/LSTM/Attention/Transformer models (in place of a
randomly-initialized embedding table).

- `train_word2vec(tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1)` →
  `gensim.models.Word2Vec(..., seed=config.RANDOM_SEED, workers=1)`. `workers=1` is
  deliberately forced to keep gensim training deterministic given the seed
  (multi-worker training is non-deterministic even with a fixed seed). `sg=1` selects
  skip-gram.
- `save_word2vec` / `load_word2vec` — thin wrappers around `model.save` /
  `Word2Vec.load`.
- `build_embedding_matrix(word2vec_model, vocab, vector_size)` — builds a
  `(vocab_size, vector_size)` float32 matrix aligned to a token→id `vocab` (e.g.
  `CodeMixTokenizer.token_to_id`): the pad token gets an all-zero row (never
  contributes gradient), and any other token missing from the Word2Vec vocab
  (including `<unk>`) gets a small seeded random vector
  (`rng.normal(scale=0.1, ...)`, `rng = np.random.default_rng(RANDOM_SEED)`) rather
  than zeros — deliberately, so the model can distinguish OOV tokens from padding and
  from each other.

Used by `src/training/train_neural.py`, `app/app_lib.py`, and
`scripts/run_pipeline.py` to build the embedding matrix fed into RNN/LSTM/Attention/ANN.

---

## 6. `src/models/`

All five from-scratch models are instantiated by `src/training/train_neural.py`'s
`build_model()` factory, wired with the hyperparameters in `src/config.py`.

### `ann.py` — `ANNClassifier`
Feed-forward baseline over a **pooled** input vector (mean-pooled Word2Vec embeddings,
computed by `train_neural._pool_embeddings`, since ANN has no learned embedding layer
or sequence encoder). `__init__(input_dim, hidden_dims, num_classes=3, dropout=0.3)`
stacks `Linear → ReLU → Dropout` per entry in `hidden_dims` (`ANN_HIDDEN_DIMS=[64,32]`),
ending in `Linear(prev_dim, num_classes)`. `forward(x)` → logits `(batch, num_classes)`.
No word-order or sequence modeling — intentionally the project's neural lower bound.

### `rnn.py` — `RNNClassifier`
`__init__(vocab_size, embedding_dim, hidden_dim, num_classes=3, pretrained_embeddings=None, freeze_embeddings=False)`.
Embedding layer either loaded from `pretrained_embeddings` (the Word2Vec matrix, via
`nn.Embedding.from_pretrained`) or randomly initialized (`padding_idx=0` either way).
Single-layer `nn.RNN(embedding_dim, hidden_dim, batch_first=True)`. `forward` packs the
sequence via `pack_padded_sequence` when `lengths` is given (variable-length,
`enforce_sorted=False`), else runs unpacked; classifies from `hidden[-1]` (final hidden
state) through `Linear(hidden_dim, num_classes)`. Vanilla RNN baseline — no gating, so
no long-range memory. Uses `HIDDEN_DIM=64`, `EMBEDDING_DIM=100`.

### `lstm.py` — `LSTMClassifier`
`__init__(vocab_size, embedding_dim, hidden_dim, num_classes=3, num_layers=1, bidirectional=False, pretrained_embeddings=None, freeze_embeddings=False)`.
Same embedding setup as RNN. `nn.LSTM(..., num_layers=num_layers, bidirectional=bidirectional, batch_first=True)`;
classifier input dim is `hidden_dim * num_directions`. Project config uses
`LSTM_NUM_LAYERS=1`, `LSTM_BIDIRECTIONAL=True`, so this project's LSTM is bidirectional
single-layer with output dim `hidden_dim*2`. `forward` packs by `lengths` as in RNN;
final hidden-state extraction takes the last layer's forward/backward final states
(shape `(num_layers*num_directions, batch, hidden_dim)`), concatenated along the
feature dim when bidirectional. Purpose: gated long-range memory vs. the vanilla RNN,
tested for whether it helps specifically on code-switched input. Also the encoder
reused by `src/generation/text_completion.py`'s LM head and composed inside
`attention.py`.

### `attention.py` — `AdditiveAttention`, `AttentionClassifier`
Manually implemented Bahdanau-style additive attention
(`score(h) = v^T tanh(W h)`) layered on top of an LSTM encoder, so the model can weight
relevant tokens instead of relying only on the final hidden state.

- `AdditiveAttention(hidden_dim)`: `W = Linear(hidden_dim, hidden_dim)`,
  `v = Linear(hidden_dim, 1, bias=False)`. `forward(encoder_outputs, mask)` computes
  per-position scores, masking invalid (pad) positions to `-inf` before softmax —
  **except** when a row is *fully* masked (e.g. text that tokenized to nothing): that
  row is left fully unmasked instead, because masking every position to `-inf` produces
  an all-`-inf` softmax input → NaN weights → NaN gradients that permanently corrupt
  the model on the next backward pass. Such empty rows are filtered out of training
  data (`dataset_builder.clean_and_dedupe`), but live Streamlit inference can still hand
  the model arbitrary/empty text, so this guard is defensive, inference-time
  robustness — "meaningless-but-finite attention over padding" is preferred over NaN.
  Returns `(context, weights)` via a weighted `bmm` sum.
- `AttentionClassifier(vocab_size, embedding_dim, hidden_dim, num_classes=3, pretrained_embeddings=None, freeze_embeddings=False)`:
  embedding → single-layer unidirectional `nn.LSTM` encoder → `AdditiveAttention` →
  `Linear(hidden_dim, num_classes)`. `forward(..., return_attention=False)` optionally
  also returns attention weights, used by the Streamlit app to visualize which tokens
  the model focused on.

### `transformer.py` — `PositionalEncoding`, `TransformerEncoderClassifier`
From-scratch Transformer encoder using PyTorch's own `nn.TransformerEncoderLayer` /
`nn.TransformerEncoder` blocks (still counts as "from scratch" per this project's
convention: random init, no pretrained weights — the same level as RNN/LSTM using
`nn.RNN`/`nn.LSTM` rather than hand-rolled recurrence).

- `PositionalEncoding(embedding_dim, max_len=512)` — standard sinusoidal positional
  encoding, precomputed buffer `pe` of shape `(1, max_len, embedding_dim)`, added to
  embeddings in `forward`.
- `TransformerEncoderClassifier(vocab_size, embedding_dim, num_heads, num_layers, ff_dim, num_classes=3, max_len=128, dropout=0.1)`:
  embedding scaled by `sqrt(embedding_dim)` (standard Transformer scaling) → positional
  encoding added → `TransformerEncoder` with a `src_key_padding_mask` built from
  `input_ids == 0` (or a passed `attention_mask`) → mean-pool over valid (non-pad)
  positions (masked sum divided by count, clamped `min=1e-6` to avoid divide-by-zero) →
  dropout → `Linear(embedding_dim, num_classes)`. Config values:
  `TRANSFORMER_NUM_HEADS=4`, `TRANSFORMER_NUM_LAYERS=2`, `TRANSFORMER_FF_DIM=128`,
  `EMBEDDING_DIM=100`, `MAX_SEQUENCE_LENGTH=64`. Also reused as an alternative encoder
  for `src/generation/text_completion.py`'s LM head.

### `bert_model.py`
Thin wrapper around HuggingFace `AutoModelForSequenceClassification` /
`AutoTokenizer` for `bert-base-multilingual-cased` (`BERT_MODEL_NAME`).

- `load_bert_and_tokenizer(model_name)` → `(tokenizer, model)` with
  `num_labels=len(config.LABELS)=3`.
- `encode_batch(tokenizer, texts, max_length)` — tokenizes with
  `padding="max_length", truncation=True, return_tensors="pt"`.

The only model in the project using pretrained weights and its own native subword
tokenizer (every other model shares `CodeMixTokenizer`) — used as an upper-bound
benchmark against the 5 from-scratch models. Used by `src/training/train_bert.py`
(fine-tuning), and by `app/app_lib.py` / `scripts/run_pipeline.py` (`_predict_bert`) for
inference.

---

## 7. `src/training/`

### `train_bert.py`
Fine-tunes pretrained mBERT as the strongest benchmark model.

- `build_dataloaders(train_df, val_df, tokenizer, batch_size)` → two `DataLoader`s;
  tokenizes via `encode_batch`, wraps `(input_ids, attention_mask, label)` into a
  `TensorDataset`.
- `_evaluate(model, loader, device)` → accuracy, no-grad eval loop.
- `fine_tune(num_epochs=config.BERT_NUM_EPOCHS, learning_rate=2e-5)` →
  `{best_val_accuracy, history}`. Main training entry point.
- `main()` — CLI wrapper (`--epochs`, `--lr`).

Algorithm flow: seeds RNG → loads the `cv_pool` split → draws
`BERT_TRAIN_SUBSAMPLE_SIZE` (5,000) train rows and a disjoint
`BERT_VAL_SUBSAMPLE_SIZE` (1,000) val rows (`.sample()` + `.drop(train_df.index)`) →
loads mBERT + tokenizer → trains with `AdamW` for `num_epochs` (default 2), tracking
best validation accuracy and checkpointing the best `state_dict` (moved to CPU) each
epoch → reloads the best state → saves both model and tokenizer to
`models_saved/bert_finetuned/` via HuggingFace's `save_pretrained`.

Unlike every other model, BERT deliberately **skips k-fold cross-validation** — full
5-fold CV for mBERT (~178M params) at this corpus's 150k-row scale was estimated to cost
an additional 3-5+ hours of CPU time for little added reporting value, so it stays on a
single held-out fine-tune. It is still evaluated against the same held-out
`data/processed/test.csv` as every other model, so comparisons in
`results/metrics_comparison.csv` remain apples-to-apples at inference time even though
the validation methodology differed.

### `train_classical.py`
Trains the three classical baselines — N-Gram, Bag-of-Words, TF-IDF — each paired with
the same `LogisticRegression` classifier.

- `_FIT_FUNCTIONS` — dict mapping `"ngram"/"bow"/"tfidf"` to their respective
  `fit_*_vectorizer` functions.
- `load_data()` → `(cv_pool_df, test_df)`.
- `_make_classifier()` → `LogisticRegression(max_iter=200, random_state=...)`.
  `max_iter=200` is capped below sklearn's usual 1000+ default because at ~100k rows ×
  20k features an unbounded fit can run long, and 200 lbfgs iterations converges
  comfortably at this bag-of-words scale.
- `train_one_baseline(feature_name, train_texts, train_labels, val_texts, val_labels)` →
  `(vectorizer, classifier, val_accuracy)`.
- `run_cross_validation(feature_name, cv_pool)` → dict `{model_name, fold_accuracies,
  mean_accuracy, std_accuracy}`, looping `config.N_FOLDS` (5) folds via `get_fold`.
- `fit_final_classical(feature_name, texts, labels)` → `(vectorizer, classifier)` fit on
  the *entire* CV pool (no held-out split) — this becomes the deployed checkpoint.
- `main()` — runs CV then full-pool refit for all 3 feature types, saves each as
  `models_saved/classical_{name}.pt`.

Design rationale: Logistic Regression (rather than mixing in Naive Bayes per feature
type) is used uniformly across all three feature types so the model comparison isolates
the effect of the *feature representation* (n-gram vs BoW vs TF-IDF), not a difference
in classifier.

### `train_neural.py`
Shared, model-agnostic training loop for all five from-scratch PyTorch architectures
(ANN, RNN, LSTM, Attention, Transformer), parameterized by a `model_name` string. The
largest module in the project (308 lines pre-edit).

- `_ASSETS_CACHE` — dict keyed by fold index (0-4) or `"final"`, caching the
  expensive-to-build tokenizer/Word2Vec/embedding-matrix/dataloaders per fold so they're
  built once and reused across all 5 models' CV runs rather than rebuilt per model.
- `_EncodedDataset(Dataset)` — precomputes token-id tensors (padded/truncated to
  `MAX_SEQUENCE_LENGTH`), true sequence lengths, and integer labels for a dataframe.
- `_collate(batch)` — stacks a batch's `(input_ids, lengths, labels)`.
- `build_model(model_name, vocab_size, embedding_matrix=None)` — factory dispatching to
  `ANNClassifier`, `RNNClassifier`, `LSTMClassifier`, `AttentionClassifier`, or
  `TransformerEncoderClassifier`, each wired with its `config.py` hyperparameters.
- `build_dataloaders(train_df, val_df, tokenizer, batch_size)` →
  `(train_loader, val_loader_or_None)`.
- `_build_assets(train_df, val_df, cache_key)` → dict `{tokenizer, embedding_matrix,
  train_loader, val_loader, vocab_size, class_weights}`. Builds `CodeMixTokenizer` vocab
  and trains Word2Vec **from `train_df` only, never `val_df`** — the textbook-correct
  k-fold approach that avoids leaking validation-fold vocabulary into feature
  extraction. Also computes inverse-frequency class weights
  (`label_counts.sum() / (num_classes * label_counts)`) to counteract the corpus's skew
  toward the "positive" label — without this, cross-entropy on a weak/small model tends
  to collapse to predicting only the majority class. When `cache_key == "final"`, also
  persists the tokenizer to `models_saved/tokenizer.json` and Word2Vec to
  `models_saved/word2vec.model` — the artifacts the Streamlit app, the text-completion
  demo, and error analysis all load downstream.
- `_pool_embeddings(input_ids, embedding_matrix_tensor)` — mean-pools a Word2Vec
  lookup over non-pad tokens; this is the ANN model's actual input.
- `_forward(model, model_name, input_ids, lengths, embedding_matrix_tensor)` —
  dispatches ANN to pooled embeddings, Transformer to raw `input_ids`, and RNN/LSTM/
  Attention to `(input_ids, lengths)`. Also imported directly by
  `scripts/run_pipeline.py` and `app/app_lib.py` for inference.
- `_evaluate(...)` → accuracy over a loader, no-grad.
- `train_one_model(model_name, train_loader, val_loader, embedding_matrix, vocab_size, class_weights, num_epochs, learning_rate)` →
  `{state_dict, best_val_accuracy, history}`. Standard Adam + weighted cross-entropy
  loop; if `val_loader` is `None` (the full-pool refit path), there's no per-epoch
  validation signal, so it just runs all epochs and returns final weights (CV, run
  separately at the same epoch budget, already validated that epoch count).
- `run_cross_validation(model_name)` — CV summary dict, looping `config.N_FOLDS` folds.
- `refit_final(model_name)` — trains on the full CV pool (dropping the `fold` column)
  with `cache_key="final"`, saves `{model_name, state_dict, vocab_size}` to
  `models_saved/{model_name}.pt`.
- `main()` — CLI (`--models`, `--epochs`, `--lr`) running CV + refit for each selected
  model.

### Cross-validation and refit strategy (shared by classical + neural)
For each classical/neural model: run `N_FOLDS=5` k-fold cross-validation on the CV pool
to get a mean/std accuracy estimate (written to `results/cv_summary.csv`), then
**separately** refit on the *entire* CV pool (no held-out split) to produce the actual
deployed checkpoint in `models_saved/`. The CV numbers estimate generalization; the
refit model is what's evaluated on the untouched `test.csv` and served in the Streamlit
app.

---

## 8. `src/evaluation/`

### `metrics.py`
Computes and persists quantitative classification metrics for every trained model, both
overall and broken down by language condition — this breakdown is what lets the project
answer its central question ("which technique is most robust to code-switched input?")
directly from the output tables/chart.

- `compute_metrics(y_true, y_pred)` → `{accuracy, precision, recall, f1}` using
  macro-averaged precision/recall/F1 (`zero_division=0`) over the fixed
  `config.LABELS` set.
- `evaluate_model_by_language(model_name, predictions_df)` → DataFrame with one row per
  language condition plus an `"overall"` row.
- `build_comparison_table(all_model_results)` → concatenates every model's per-language
  DataFrame, saves to `results/metrics_comparison.csv`.
- `summarize_cv_results(fold_results)` → `{model, n_folds, mean_accuracy, std_accuracy}`
  sorted by mean accuracy descending, saved to `results/cv_summary.csv`. This reports
  the *validation* methodology (k-fold CV on the training pool) — a separate,
  complementary artifact to `metrics_comparison.csv`, which reports the final refit
  models' performance on the untouched held-out test set.
- `plot_comparison_chart(comparison_table)` → grouped bar chart (Matplotlib), one group
  per model, one bar per language condition, saved to `results/metrics_comparison.png`.
  Uses a fixed-order, adjacent-pair colorblind-safe categorical palette (english=blue
  `#2a78d6`, bangla=orange `#eb6834`, banglish=aqua/green `#1baf7a`,
  code_switched=yellow `#eda100`).

### `error_analysis.py`
Produces the qualitative half of evaluation — representative misclassified examples per
model with a heuristic likely-cause note, plus qualitative text-generation samples —
written to `results/error_analysis.md`.

- `collect_misclassified_examples(model_name, predictions_df, num_examples=5)` →
  DataFrame of up to 5 wrong predictions, tagged with `model` and a `note` column.
  Spreads the sample across language conditions (rather than taking the first N rows)
  via a per-language quota + top-up from remaining wrong rows, so the report isn't
  dominated by whichever language condition happens to have the most errors.
- `_likely_cause_note(row)` — heuristic string explaining a probable failure reason:
  code-switched input → "unseen intra-sentence language-mixing pattern"; banglish →
  "out-of-vocabulary tokens (no dictionary/script cues)"; predicted the majority class
  "positive" → "possible class-imbalance bias"; otherwise → generic ambiguous/sarcastic-
  phrasing note.
- `write_error_report(all_examples, output_path)` — builds the Markdown report: a
  "Misclassified Examples" section per model, followed by a "Text-Completion Bonus
  (Phase 9, qualitative only)" section listing prompt→completion pairs.
- `collect_generation_samples(prompts)` — runs `TextCompletionModel.generate` on each
  prompt using both the `lstm` and `transformer` checkpoints (if they exist), returning
  `{model_name, prompt, completion}` dicts. Returns `[]` if the shared tokenizer
  checkpoint doesn't exist yet. The text-completion outputs are explicitly **not scored
  numerically** — included purely as illustrative examples, distinct from the scored
  classification metrics.

---

## 9. `src/generation/`

### `text_completion.py`
A small bonus (non-scored) autoregressive text-generation demo built on top of an
already-trained LSTM or Transformer sentiment-classification checkpoint, reusing its
embedding + encoder as a warm start for a lightweight next-token language model.

- `_SUPPORTED_BASE_MODELS = ("lstm", "transformer")`.
- `_NextTokenDataset(Dataset)` — builds `(input_ids[:-1], input_ids[1:])`
  teacher-forcing pairs from raw texts using the base classifier's own tokenizer/vocab.
- `TextCompletionModel.__init__(base_model_checkpoint_path, tokenizer, lm_epochs=config.TEXT_COMPLETION_LM_EPOCHS, learning_rate=1e-3)` —
  loads the base checkpoint, validates its `model_name` is lstm/transformer,
  reconstructs the base model via `train_neural.build_model`, loads its trained
  weights, then **extracts only the embedding layer and encoder**
  (discarding the classification head) and attaches a fresh `nn.Linear` LM head sized
  to the vocab. Immediately trains that head via `_train_lm_head`.
- `_encode_sequence(input_ids)` — returns per-timestep hidden states
  `(batch, seq_len, hidden_dim)`; for LSTM this is the raw LSTM output, for Transformer
  it scales embeddings by `sqrt(embedding_dim)`, adds positional encoding, and runs the
  encoder with a padding mask.
- `_train_lm_head(num_epochs, learning_rate)` — subsamples the CV pool to
  `config.TEXT_COMPLETION_TRAIN_SUBSAMPLE_SIZE` (5,000) rows, trains embedding+encoder+
  LM-head jointly via teacher-forced next-token cross-entropy (`ignore_index=0` for
  `<pad>` targets) for `config.TEXT_COMPLETION_LM_EPOCHS` (3) epochs.
- `generate(prompt, max_new_tokens=20, temperature=1.0)` → str. Greedy
  (`temperature<=0`) or multinomial-sampled autoregressive decoding, one token at a
  time, feeding the growing context (truncated to `MAX_SEQUENCE_LENGTH`) back through
  `_encode_sequence` + LM head; stops early on `<pad>`.

The base checkpoint was trained purely for 3-class sentiment classification and has no
next-token head of its own — this class exists solely to *retrofit* one via a
warm-started, briefly-trained linear head, good enough for qualitative "does this look
plausible" demos. Its output feeds into `error_analysis.py`'s report, never scored
numerically.

---

## 10. `src/utils/`

### `seed.py`
`set_seed(seed=config.RANDOM_SEED)` — seeds Python's `random`, `numpy`, and `torch`
(CPU and, if available, all CUDA devices), and forces
`torch.backends.cudnn.deterministic=True` / `benchmark=False` for reproducible runs.
Called first by every training script (classical, neural, BERT).

### `io_utils.py`
Small I/O helpers shared across the project so scripts don't repeat boilerplate.

- `ensure_dir(path)` — wraps `os.makedirs(exist_ok=True)`.
- `load_json` / `save_json` (UTF-8, `ensure_ascii=False`, indented) and `load_csv` /
  `save_csv` (pandas, no index) — standard read/write, auto-creating the parent
  directory on save.
- `save_checkpoint` / `load_checkpoint` — wrap `torch.save` / `torch.load`;
  `load_checkpoint` passes `weights_only=False` because project checkpoints can hold
  arbitrary picklable objects (e.g. sklearn vectorizers/classifiers for the classical
  baselines), not just tensors, and PyTorch ≥2.6 defaults to a tensor-only loader — safe
  here because every checkpoint is a first-party artifact this project's own training
  scripts wrote to `models_saved/`, never an externally sourced file.

---

## 11. `app/` (Streamlit UI)

### `streamlit_app.py`
Entry point (`streamlit run app/streamlit_app.py`). Inserts both the project root and
`app/` onto `sys.path` (so `app_lib` and `src.*` are both importable regardless of
Streamlit's working directory), calls `st.set_page_config(...)`, injects custom CSS via
`st.html(style(is_dark=...))` (dark/light detected from `st.context.theme.type`), then
builds an `st.navigation([...])` with two pages — `app_pages/analyzer.py` (default,
"Analyzer") and `app_pages/architecture.py` ("How models work") — and calls `page.run()`.
Kept as a thin entry point specifically so the page modules hold all page-specific
logic.

### `app_lib.py`
Shared constants, cached model loaders, and inference/styling helpers used by both
pages — deliberately kept out of `streamlit_app.py` so importing this module doesn't
also trigger `st.navigation()`.

Key constants: `CLASSICAL_MODELS`, `NEURAL_MODELS`; `TABLE_ROW_HEIGHT=38` (used so
every no-scroll `st.dataframe` table's height formula `(rows+1)*TABLE_ROW_HEIGHT+3`
stays in sync with the actual rendered row height); `MODEL_DISPLAY` (internal key →
`(category, display name)`, e.g. `"tfidf" → ("Classical", "TF-IDF")`); `SENTIMENT_STYLE`
(label → `(badge color, Material icon)`, green=positive / red=negative / gray=neutral).

Key functions:
- `display_name(model_name)` / `short_name(model_name)` — full "Category – Name" string
  vs. bare model name (used in tight spaces, e.g. the confidence chart's label gutter,
  where the full name would be truncated).
- `style(is_dark)` — returns a `<style>` block of minimal, explicitly-scoped CSS
  (targets Streamlit's auto-generated `.st-key-<key>` classes rather than broad
  selectors): fade-in on `.block-container`, hover-lift on cards/buttons, larger review
  `textarea` font, a soft half-width divider class. Shadow colors flip for dark mode
  since a black shadow is invisible on a dark background.
- `soft_divider()` — a lighter-weight alternative to `st.divider()`'s full-width rule.
- `load_all_models()` (`@st.cache_resource`, runs once per server process/session) —
  loads every checkpoint that exists on disk into
  `{"classical": {...}, "neural": {...}, "bert": ..., "tokenizer": ..., "embedding_matrix": ...}`.
  Missing checkpoints are silently skipped (via `Path.exists()`), letting the app run
  with whatever subset of the 9 models has actually been trained.
- `load_model_performance()` (`@st.cache_data`) — reads
  `results/metrics_comparison.csv`, filters to `language == "overall"` rows, sorts
  descending by accuracy, returns `{model: accuracy}` for the sidebar accuracy table.
- `load_sample_pool()` (`@st.cache_data`) — loads the held-out test split, groups review
  texts by language condition, for the "Try a random example" buttons.
- `predict_with_model(model_name, models, text)` → `(label, confidence)` — dispatches to
  classical (`predict_proba` on the transformed text), neural (tokenize/encode →
  `train_neural._forward` → softmax), or BERT (`encode_batch` → forward pass →
  softmax) prediction paths depending on which bucket `model_name` falls into.

### `diagrams.py`
Generates a Graphviz DOT string per model, rendered client-side via
`st.graphviz_chart` (no `graphviz` pip package or system `dot` binary required, since
Streamlit renders DOT text directly). Each diagram was manually checked stage-by-stage
against the real `forward()` pass it depicts — simplified to real processing stages,
not every tensor op, but nothing shown is invented or out of order.

- `_palette(is_dark)` — dark/light hex color dict (background, node fill/border/text,
  edge color, an "accent" variant).
- `_header(p)`, `_node(node_id, label, p, accent=False)`, `_chain(node_ids)` — DOT
  boilerplate helpers.
- Per-model diagram builders (`_classical_diagram`, `_ann_diagram`, `_rnn_diagram`,
  `_lstm_diagram`, `_attention_diagram`, `_transformer_diagram`, `_bert_diagram`), each
  visualizing that model's real processing stages — e.g. the RNN diagram adds a
  self-loop edge labeled "memory fades over distance"; the LSTM diagram shows two
  parallel forward/backward direction nodes with "gated memory" self-loops that
  concatenate before the linear layer; the BERT diagram shows mBERT's own tokenizer →
  pretrained encoder (accent node) → `[CLS]` vector → new fine-tuned classifier head.
- `diagram_dot(model_name, is_dark)` — public entry point dispatching through a
  `_BUILDERS` map.

### `app_pages/analyzer.py`
The default/main page. Lets the user type or pick a sample review, choose which
available trained models to run, and view predictions side by side with an overall
consensus and a confidence bar chart.

- `render_verdict(results)` — computes the majority-vote label across selected models'
  predictions, averages confidence among models that agree with that label, renders it
  as a caption + colored badge + agreement caption (e.g. "unanimous · 92% average
  confidence" or "3 of 5 models agree · ...").
- `render_model_cards(results)` — one bordered card per model, showing display name,
  colored sentiment badge, and confidence percentage. Text-only (no progress bar) since
  the bar comparison already lives in the chart below.
- `render_comparison_chart(results)` — a horizontal `st.bar_chart` of
  `{short_name: confidence}` sorted ascending, so long model names stay legible instead
  of being rotated.
- `_use_sample(language, pool)` — button callback filling the review textbox with a
  random review from `pool[language]`.

Page flow: load models → if none exist, error + stop; sidebar with a multi-select of
available models (`st.pills`), a live "N of M active" caption, an optional accuracy
table, and a link to the architecture page; per-language "try a random example"
buttons; an `st.form` with a text area + "Analyze" button; on submit, validates
non-empty text and ≥1 selected model, runs `predict_with_model` per selected model, then
renders verdict, cards, and chart.

### `app_pages/architecture.py`
An educational "How do these models work?" page — explains all 9 models grouped into 3
families (Classical, Neural-from-scratch, Pretrained), each paired with its
`diagrams.py` diagram, plus a side-by-side comparison table. All hyperparameters/facts
quoted on this page are read live from `src.config` or the saved tokenizer JSON rather
than hardcoded copies, so the page stays accurate automatically if the config changes.

- `load_vocab_size()` (`@st.cache_data`) — reads `models_saved/tokenizer.json`, returns
  `len(token_to_id)` or `None` if not yet built.
- Three tabs (Classical / Neural / Pretrained), each with an `st.expander` per model
  explaining its mechanism and quoting live config values (e.g. TF-IDF's expander
  dynamically appends "(best performer, X% test accuracy)" if performance data is
  available; BERT's expander notes its numbers aren't directly apples-to-apples with
  the other 8 models since it uses a different train-set size and no k-fold CV).
- A final comparison table (`Model`, `Family`, "Word order?", "Long-range memory",
  "Pretrained on outside data?", live "Test accuracy") and a caption reporting the
  shared neural-model vocabulary size.

---

## 12. `scripts/`

### `build_dataset.py`
CLI entry point that downloads BanglishRev, builds, and splits the dataset.

```bash
python scripts/build_dataset.py [--seed SEED] [--target-size N] [--cache-dir DIR]
```

- `--seed` (default `config.RANDOM_SEED`=42) — also written back into
  `config.RANDOM_SEED`.
- `--target-size` (default `config.TARGET_DATASET_SIZE`=150,000) — target total row
  count, written back into `config.TARGET_DATASET_SIZE`.
- `--cache-dir` (default `None`) — optional override for the Hugging Face download
  cache directory.

`main()` seeds RNG, applies the overrides, calls `assemble_dataset(cache_dir=...)`
(writes `data/raw/dataset.csv`), then `create_cv_splits(df)` to get
`(cv_pool, test_df)`, and prints summary stats (row counts, label×language breakdown
per split, per-fold sizes).

The first run needs internet access to fetch BanglishRev's raw review JSON (~1.9GB)
from Hugging Face; later runs reuse the local `huggingface_hub` cache. At
`TARGET_DATASET_SIZE=150,000` this must flatten/tag the *entire* ~1.74M-row raw corpus
rather than a small sample, so expect a few minutes even with the raw file cached.

### `run_pipeline.py`
CLI entry point that runs the full pipeline end to end: verify processed data exists →
train selected classical/neural models via 5-fold CV plus a full-pool refit (BERT
instead trains on a single held-out split, no CV) → evaluate every trained/available
model on the held-out test set → write result artifacts.

```bash
python scripts/run_pipeline.py [--models ngram,bow,tfidf,ann,rnn,lstm,attention,transformer,bert]
                                [--skip-bert] [--skip-classical] [--epochs N]
```

- `--models` (default: all 9, comma-joined) — validated against `ALL_MODELS`, raising
  `ValueError` on an unknown name.
- `--skip-bert` — drops `"bert"` from the selected list.
- `--skip-classical` — drops all of `CLASSICAL_MODELS` from the selected list.
- `--epochs N` — overrides `config.NUM_EPOCHS` (neural) / `config.BERT_NUM_EPOCHS`
  (BERT) if given.

Key functions:
- `_require_processed_data()` — raises `FileNotFoundError` if `cv_pool.csv`/`test.csv`
  are missing, hinting to run `build_dataset.py` first.
- `_predict_classical` / `_predict_neural` / `_predict_bert` — per-family prediction
  helpers, each loading the relevant checkpoint(s) and batching inputs
  (`_PREDICT_BATCH_SIZE=128` for neural, `config.BERT_BATCH_SIZE` for BERT) so
  evaluation doesn't run the whole test set through a model in one unbatched forward
  pass.
- `_checkpoint_exists(model_name)` — per-family existence check.
- `run_evaluation(models_to_evaluate, cv_results_by_model=None)` — loads the test
  split, filters to models with existing checkpoints (printing a skip notice for the
  rest), builds the shared tokenizer/embedding matrix once if any neural model is
  included, predicts for every available model, computes per-language metrics and
  misclassified examples, builds/plots the comparison table
  (`results/metrics_comparison.{csv,png}`), collects qualitative generation samples for
  one fixed prompt per language condition, writes `results/error_analysis.md`, and — if
  CV results were passed in — also writes `results/cv_summary.csv`.
- `main()` — parses args, applies filters, then sequentially: cross-validates + refits +
  checkpoints selected classical models, cross-validates + refits selected neural
  models, fine-tunes BERT if selected, then calls `run_evaluation` on the full
  originally-selected model list.

Training (aside from `--skip-bert`) is long-running once real models are selected, so
this script is meant to be run directly in a terminal by a human rather than driven
inline in an AI session; `--help`, and re-running only the evaluation step against
checkpoints that already exist in `models_saved/`, are the exceptions safe to run either
way.

---

## 13. Outputs (`models_saved/`, `results/`)

`models_saved/` holds one checkpoint per model:
`classical_{ngram,bow,tfidf}.pt` (vectorizer + `LogisticRegression`),
`{ann,rnn,lstm,attention,transformer}.pt` (`state_dict` + `vocab_size` +
`model_name`), `tokenizer.json` / `word2vec.model` (shared neural-pipeline
artifacts, built once during the `"final"` refit), and `bert_finetuned/` (a full
`transformers`-format directory: config, weights, tokenizer files). The fine-tuned
BERT checkpoint is excluded from version control (over GitHub's file-size limit); the
rest are committed so the Streamlit app runs immediately after cloning.

`results/` holds the pipeline's evaluation artifacts:
- `metrics_comparison.csv` / `.png` — every model's accuracy/precision/recall/F1,
  overall and per language condition, as a table and a grouped bar chart.
- `cv_summary.csv` — each model's k-fold CV mean/std accuracy (classical + neural only;
  BERT isn't cross-validated).
- `error_analysis.md` — up to 5 misclassified examples per model (spread across
  language conditions, each with a heuristic likely-cause note) plus qualitative
  LSTM/Transformer text-completion samples for 4 fixed prompts.
