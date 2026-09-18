"""
Central configuration for the Code-Mix Sentiment Analyzer project.

Every other module should import constants/paths from here instead of
redefining or hardcoding them, so that changing a path or hyperparameter
in one place updates the whole pipeline.

TODO:
- Define base paths:
    PROJECT_ROOT, DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_SAVED_DIR, RESULTS_DIR
- Define dataset constants:
    LABELS = ["positive", "negative", "neutral"]
    LANGUAGE_CONDITIONS = ["english", "bangla", "banglish", "code_switched"]
    TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT ratios
    RANDOM_SEED
- Define BanglishRev source constants (see src/data/dataset_builder.py):
    HF_DATASET_ID = "BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset"
    RATING_TO_LABEL = {1: "negative", 2: "negative", 3: "neutral",
                        4: "positive", 5: "positive"}
    ALLOWED_CATEGORIES = [...]   # short list of universally familiar product
                                  # categories, e.g. Fashion, Electronics,
                                  # Grocery, Home & Living, Beauty
    BANGLA_UNICODE_RANGE = (0x0980, 0x09FF)   # for the language-detection heuristic
    LANGUAGE_DETECTION_MIN_TOKEN_RATIO = 0.6  # threshold used to decide
                                                # "mostly Bangla script" / "mostly
                                                # English-dictionary tokens" etc.
    TARGET_DATASET_SIZE = 2000   # final small, balanced corpus size (tune as needed)
    MIN_REVIEW_LENGTH_CHARS = 8  # drop very short/low-signal reviews
- Define shared feature/model hyperparameters (can also live per-model if preferred):
    MAX_SEQUENCE_LENGTH
    EMBEDDING_DIM (for from-scratch models / Word2Vec)
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS (defaults; can be overridden per training script)
- Define BERT_MODEL_NAME (e.g. a multilingual checkpoint identifier such as
    "bert-base-multilingual-cased" or an XLM-R variant) for src/models/bert_model.py.
- Consider loading overrides from a YAML/JSON config file if useful, but keep
    sensible hardcoded defaults so the project runs without extra setup.
"""

# TODO: implement constants described above.
