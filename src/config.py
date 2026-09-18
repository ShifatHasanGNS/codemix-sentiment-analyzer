"""
Central configuration for the Code-Mix Sentiment Analyzer project.

Every other module should import constants/paths from here instead of
redefining or hardcoding them, so that changing a path or hyperparameter
in one place updates the whole pipeline.
"""

from pathlib import Path

from dotenv import load_dotenv

# --- Base paths ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Loads .env (if present) into the process environment -- e.g. HF_TOKEN,
# which huggingface_hub's hf_hub_download picks up automatically for
# authenticated, higher-rate-limit downloads. src.config is imported before
# any network call in this project, so this runs early enough regardless of
# entry point. Never logs/prints the token's value.
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_SAVED_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Dataset constants ---------------------------------------------------

LABELS = ["positive", "negative", "neutral"]
LANGUAGE_CONDITIONS = ["english", "bangla", "banglish", "code_switched"]

# Held-out final test set, carved off before k-fold CV; the remaining pool
# gets a `fold` column (0..N_FOLDS-1) instead of a fixed train/val split.
TEST_HOLDOUT_SPLIT = 0.10
N_FOLDS = 5

RANDOM_SEED = 42

# --- BanglishRev source constants ---------------------------------------

HF_DATASET_ID = "BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset"

RATING_TO_LABEL = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

# No category restriction: an earlier 5-bucket "familiar categories" keyword
# filter was found (by sampling the raw corpus) to exclude ~74% of it,
# including everyday items like Smartwatches, T-Shirts, Shampoo, Toothpaste,
# Diapers, and Coffee that are just as "universally familiar" as the
# original 5 buckets -- the whole corpus is ordinary e-commerce consumer
# goods by construction. None disables filter_familiar_categories()'s
# restriction (see its docstring in src/data/dataset_builder.py).
ALLOWED_CATEGORIES = None

# Unicode block for Bangla script, used by the language-detection heuristic.
BANGLA_UNICODE_RANGE = (0x0980, 0x09FF)

# Fraction of alphabetic tokens that must fall on one side of the
# Bangla-script / English-dictionary split for a review to count as
# "mostly" that language, rather than code-switched.
LANGUAGE_DETECTION_MIN_TOKEN_RATIO = 0.6

# Target corpus size, balanced by label (comfortably under the ~57k natural
# ceiling for the scarcest label, "neutral"). Code-switched reviews are
# intrinsically rare in this corpus (~1,600 total across the full ~1.74M
# raw reviews, regardless of category scope) and are used at their natural
# scale rather than forced into an equal language-condition bucket --
# subsample_balanced()'s best-effort-then-redistribute algorithm already
# handles this correctly. See data/README.md's "Known limitations" section.
TARGET_DATASET_SIZE = 150_000
MIN_REVIEW_LENGTH_CHARS = 8

# --- Shared feature/model hyperparameters --------------------------------

MAX_SEQUENCE_LENGTH = 64
EMBEDDING_DIM = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# At ~108k rows/fold, each epoch is ~67x more gradient updates than the
# original 1600-row corpus; fewer epochs are needed to converge, and this
# keeps 5-fold x 5-model CV bounded to roughly an hour on CPU.
NUM_EPOCHS = 4

# Bounds the vocabulary size of the classical N-Gram/BoW/TF-IDF vectorizers.
CLASSICAL_MAX_FEATURES = 20_000

# Tokenizer vocab building drops tokens seen fewer than this many times in
# the training corpus. At 100k+ documents, min_freq=1 would keep every
# typo/rare token as its own vocabulary entry, bloating the embedding table
# with mostly-noise, low-signal entries.
MIN_TOKEN_FREQ = 3

# From-scratch neural model architecture (kept small -- see CLAUDE.md's
# "modest hyperparameters" ground rule -- so training stays reasonably fast
# on CPU even at this corpus's scale).
DROPOUT = 0.3
HIDDEN_DIM = 64
ANN_HIDDEN_DIMS = [64, 32]
LSTM_NUM_LAYERS = 1
LSTM_BIDIRECTIONAL = True
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_FF_DIM = 128

# Phase 9 bonus: epochs used to briefly train a next-token LM head on top
# of a trained LSTM/Transformer checkpoint's (reused) embedding + encoder,
# for qualitative-only text-completion demos. Bounded subsample of the CV
# pool (rather than all 135k rows) since this is a non-scored qualitative
# demo, not worth the extra training time at full corpus scale.
TEXT_COMPLETION_LM_EPOCHS = 3
TEXT_COMPLETION_TRAIN_SUBSAMPLE_SIZE = 5_000

# --- Pretrained BERT benchmark --------------------------------------------

BERT_MODEL_NAME = "bert-base-multilingual-cased"
# mBERT subword-tokenizes Bangla script into more pieces per word than the
# project's own word-level tokenizer, so it gets a longer sequence budget.
BERT_MAX_SEQUENCE_LENGTH = 96
# Smaller than the from-scratch models' BATCH_SIZE since mBERT (~178M
# params) is far more expensive per step on CPU.
BERT_BATCH_SIZE = 8
BERT_NUM_EPOCHS = 2
# BERT stays on a single held-out split rather than k-fold CV (agreed
# tradeoff: full 5-fold CV for mBERT at this corpus's scale would add an
# estimated 3-5+ hours of CPU time for little reporting benefit). Train/val
# subsamples are drawn from the CV pool, non-overlapping; evaluated on the
# same held-out test.csv as every other model.
BERT_TRAIN_SUBSAMPLE_SIZE = 5_000
BERT_VAL_SUBSAMPLE_SIZE = 1_000
