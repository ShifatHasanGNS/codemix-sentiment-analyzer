"""Central configuration: paths, constants, hyperparameters."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_SAVED_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"

LABELS = ["positive", "negative", "neutral"]
LANGUAGE_CONDITIONS = ["english", "bangla", "banglish", "code_switched"]

# Held-out test set carved off before k-fold CV; the rest gets a `fold` column.
TEST_HOLDOUT_SPLIT = 0.10
N_FOLDS = 5

RANDOM_SEED = 42

HF_DATASET_ID = "BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset"

RATING_TO_LABEL = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

# None = no category restriction (see filter_familiar_categories() docstring).
ALLOWED_CATEGORIES = None

BANGLA_UNICODE_RANGE = (0x0980, 0x09FF)

# Min fraction of alphabetic tokens on one side of the script/dictionary split
# for a review to count as "mostly" that language rather than code-switched.
LANGUAGE_DETECTION_MIN_TOKEN_RATIO = 0.6

TARGET_DATASET_SIZE = 150_000
MIN_REVIEW_LENGTH_CHARS = 8

MAX_SEQUENCE_LENGTH = 64
EMBEDDING_DIM = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 4

CLASSICAL_MAX_FEATURES = 20_000

# Vocab-building drops tokens seen fewer than this many times.
MIN_TOKEN_FREQ = 3

DROPOUT = 0.3
HIDDEN_DIM = 64
ANN_HIDDEN_DIMS = [64, 32]
LSTM_NUM_LAYERS = 1
LSTM_BIDIRECTIONAL = True
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_FF_DIM = 128

# Text-completion LM head: brief extra training on a subsample, qualitative only.
TEXT_COMPLETION_LM_EPOCHS = 3
TEXT_COMPLETION_TRAIN_SUBSAMPLE_SIZE = 5_000

BERT_MODEL_NAME = "bert-base-multilingual-cased"
BERT_MAX_SEQUENCE_LENGTH = 96
BERT_BATCH_SIZE = 8
BERT_NUM_EPOCHS = 2
# Single held-out split, not k-fold CV (full CV for mBERT is too slow on CPU).
BERT_TRAIN_SUBSAMPLE_SIZE = 5_000
BERT_VAL_SUBSAMPLE_SIZE = 1_000
