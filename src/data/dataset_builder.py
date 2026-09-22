"""Assembles the Code-Mix Sentiment corpus from BanglishRev, entirely via code.

Pipeline: download_banglishrev -> flatten_reviews -> map_rating_to_label ->
detect_language_condition -> filter_familiar_categories -> clean_and_dedupe ->
subsample_balanced -> assemble_dataset (orchestrates, writes data/raw/dataset.csv)
-> create_cv_splits (writes data/processed/). See data/README.md for schema.
"""

import json
import re

import pandas as pd

from src import config
from src.utils.io_utils import ensure_dir, save_csv

_REVIEW_RATING_FIELD = "Current Rating"
_REVIEW_TEXT_FIELD = "Review Content"
_PRODUCT_CATEGORY_FIELDS = ("Category", "Parent Category", "Root Category")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_ENGLISH_WORDS = None  # lazily-built cache; see _english_word_set()

# Short English function words kept despite the length>=3 filter below.
_SHORT_ENGLISH_WORDS = frozenset(
    [
        "a",
        "i",
        "am",
        "an",
        "as",
        "at",
        "be",
        "by",
        "do",
        "go",
        "he",
        "hi",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "no",
        "of",
        "ok",
        "on",
        "or",
        "so",
        "to",
        "up",
        "us",
        "we",
    ]
)


def download_banglishrev(cache_dir: str | None = None) -> str:
    """Fetch BanglishRev's `reviews v1.json` by exact filename (never the image archives) and return its local path."""
    from huggingface_hub import hf_hub_download

    if cache_dir is None:
        cache_dir = str(config.DATA_RAW_DIR / "hf_cache")
    ensure_dir(cache_dir)

    return hf_hub_download(
        repo_id=config.HF_DATASET_ID,
        repo_type="dataset",
        filename="reviews v1.json",
        cache_dir=cache_dir,
    )


def flatten_reviews(raw_data) -> pd.DataFrame:
    """Flatten BanglishRev's per-product JSON (path or parsed list) into one row per review: product_category, rating (int), text."""
    if isinstance(raw_data, (str,)):
        with open(raw_data, "r", encoding="utf-8") as f:
            products = json.load(f)
    else:
        products = raw_data

    rows = []
    for product in products:
        category = None
        for field in _PRODUCT_CATEGORY_FIELDS:
            value = product.get(field)
            if value:
                category = value
                break
        if category is None:
            continue

        for review in product.get("Reviews", []):
            text = review.get(_REVIEW_TEXT_FIELD)
            if not text or not text.strip():
                continue

            try:
                rating = int(review.get(_REVIEW_RATING_FIELD))
            except (TypeError, ValueError):
                continue
            if rating not in config.RATING_TO_LABEL:
                continue

            rows.append(
                {
                    "product_category": category,
                    "rating": rating,
                    "text": text.strip(),
                }
            )

    return pd.DataFrame(rows, columns=["product_category", "rating", "text"])


def map_rating_to_label(rating: int) -> str:
    """Map a 1-5 star rating to positive/negative/neutral via src.config.RATING_TO_LABEL."""
    try:
        return config.RATING_TO_LABEL[int(rating)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"rating must be an int in {sorted(config.RATING_TO_LABEL)}, got {rating!r}"
        )


def _english_word_set():
    """Lazily load/cache NLTK's English word list; short entries dropped (they falsely match romanized-Bangla) except a curated allowlist."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        from nltk.corpus import words as nltk_words

        try:
            all_words = frozenset(w.lower() for w in nltk_words.words())
        except LookupError:
            raise LookupError(
                "NLTK 'words' corpus not found. Run "
                "`python -m nltk.downloader words` once, then retry."
            )
        _ENGLISH_WORDS = frozenset(w for w in all_words if len(w) >= 3) | (
            _SHORT_ENGLISH_WORDS & all_words
        )
    return _ENGLISH_WORDS


def _script_of(token: str) -> str:
    """Classify a token as 'bangla' or 'latin' by its first matching character; uses str.isalpha() so stylized Unicode letters still count as latin."""
    bangla_lo, bangla_hi = config.BANGLA_UNICODE_RANGE
    for ch in token:
        code = ord(ch)
        if bangla_lo <= code <= bangla_hi:
            return "bangla"
        if ch.isalpha():
            return "latin"
    return "other"


def detect_language_condition(text: str) -> str:
    """Tag a review as english/bangla/banglish/code_switched via a Unicode-script + English-dictionary ratio heuristic (see src.config threshold)."""
    threshold = config.LANGUAGE_DETECTION_MIN_TOKEN_RATIO
    tokens = [t for t in _WORD_RE.findall(text)]
    scripts = [_script_of(t) for t in tokens]
    bangla_count = scripts.count("bangla")
    latin_count = scripts.count("latin")
    total = bangla_count + latin_count

    if total == 0:
        return "english"  # no alphabetic signal at all; harmless default

    bangla_ratio = bangla_count / total
    if bangla_ratio >= threshold:
        return "bangla"

    latin_ratio = latin_count / total
    if latin_ratio >= threshold:
        latin_tokens = [t.lower() for t, s in zip(tokens, scripts) if s == "latin"]
        english_words = _english_word_set()
        english_ratio = sum(t in english_words for t in latin_tokens) / len(
            latin_tokens
        )
        return "english" if english_ratio >= threshold else "banglish"

    return "code_switched"


def filter_familiar_categories(df: pd.DataFrame, allowed_categories) -> pd.DataFrame:
    """Optionally restrict to reviews whose category matches (case-insensitive substring) one of `allowed_categories`; None (default) disables it."""
    if not allowed_categories:
        return df

    def matches(category: str) -> bool:
        category_lower = str(category).lower()
        return any(keyword.lower() in category_lower for keyword in allowed_categories)

    mask = df["product_category"].apply(matches)
    return df[mask].reset_index(drop=True)


def clean_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip control chars, drop too-short/duplicate/content-free (zero-alphabetic-token, e.g. mojibake) reviews."""
    df = df.copy()
    df["text"] = (
        df["text"]
        .str.replace(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", regex=True)
        .str.strip()
    )
    df = df[df["text"].str.len() >= config.MIN_REVIEW_LENGTH_CHARS]
    df = df[df["text"].apply(lambda t: len(_WORD_RE.findall(t)) > 0)]
    df = df.drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def subsample_balanced(
    df: pd.DataFrame, target_size: int, per_group_cap: int | None = None
) -> pd.DataFrame:
    """Seeded subsample to `target_size`, balanced by label first, then best-effort-redistributed across language conditions within each label."""
    labels = df["label"].unique()
    n_labels = max(len(labels), 1)
    label_quota = target_size // n_labels

    sampled_parts = []
    for label in labels:
        subset = df[df["label"] == label]
        language_groups = list(subset.groupby("language", sort=False))
        n_languages = max(len(language_groups), 1)
        cap = per_group_cap if per_group_cap is not None else label_quota
        base_quota = max(1, min(cap, label_quota // n_languages))

        chosen_parts = []
        remaining_pool = []
        for _, group_df in language_groups:
            take = min(base_quota, len(group_df))
            chosen = group_df.sample(n=take, random_state=config.RANDOM_SEED)
            chosen_parts.append(chosen)
            leftover = group_df.drop(chosen.index)
            if len(leftover) > 0:
                remaining_pool.append(leftover)

        total = sum(len(c) for c in chosen_parts)
        shortfall = label_quota - total
        if shortfall > 0 and remaining_pool:
            pool = pd.concat(remaining_pool)
            extra = pool.sample(
                n=min(shortfall, len(pool)), random_state=config.RANDOM_SEED
            )
            chosen_parts.append(extra)

        sampled_parts.append(pd.concat(chosen_parts))

    result = pd.concat(sampled_parts).sample(frac=1, random_state=config.RANDOM_SEED)
    return result.reset_index(drop=True)


def assemble_dataset(cache_dir: str | None = None) -> pd.DataFrame:
    """Run the full collection pipeline and write data/raw/dataset.csv."""
    raw_path = download_banglishrev(cache_dir=cache_dir)
    df = flatten_reviews(raw_path)

    df["label"] = df["rating"].apply(map_rating_to_label)
    df["language"] = df["text"].apply(detect_language_condition)

    df = filter_familiar_categories(df, config.ALLOWED_CATEGORIES)
    df = clean_and_dedupe(df)

    # Cap each language condition at 3x its even split, so plentiful ones absorb scarce ones' shortfall.
    label_quota = config.TARGET_DATASET_SIZE // len(config.LABELS)
    per_group_cap = max(50, (label_quota // len(config.LANGUAGE_CONDITIONS)) * 3)
    df = subsample_balanced(df, config.TARGET_DATASET_SIZE, per_group_cap)

    df.insert(0, "id", [f"cm{idx:06d}" for idx in range(len(df))])
    df = df[["id", "product_category", "rating", "label", "language", "text"]]

    save_csv(df, config.DATA_RAW_DIR / "dataset.csv")
    return df


def create_cv_splits(
    df: pd.DataFrame, n_folds: int | None = None, test_holdout: float | None = None
):
    """Carve off a stratified held-out test set, then add a `fold` column (0..n_folds-1) to the rest; writes both to data/processed/."""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    n_folds = n_folds or config.N_FOLDS
    test_holdout = (
        test_holdout if test_holdout is not None else config.TEST_HOLDOUT_SPLIT
    )

    strata = df["label"] + "_" + df["language"]
    try:
        cv_pool, test_df = train_test_split(
            df,
            test_size=test_holdout,
            random_state=config.RANDOM_SEED,
            stratify=strata,
        )
        fold_strata_source = "joint"
    except ValueError:
        print(
            "Warning: some (label, language) group too small to stratify jointly for "
            "the test split; falling back to stratifying by label only."
        )
        cv_pool, test_df = train_test_split(
            df,
            test_size=test_holdout,
            random_state=config.RANDOM_SEED,
            stratify=df["label"],
        )
        fold_strata_source = "label"

    cv_pool = cv_pool.reset_index(drop=True)
    fold_strata = (
        cv_pool["label"] + "_" + cv_pool["language"]
        if fold_strata_source == "joint"
        else cv_pool["label"]
    )
    try:
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED
        )
        fold_of = pd.Series(index=cv_pool.index, dtype=int)
        for fold_idx, (_, val_indices) in enumerate(skf.split(cv_pool, fold_strata)):
            fold_of.iloc[val_indices] = fold_idx
    except ValueError:
        print(
            "Warning: some (label, language) group too small to stratify jointly for "
            "k-fold assignment; falling back to stratifying by label only."
        )
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED
        )
        fold_of = pd.Series(index=cv_pool.index, dtype=int)
        for fold_idx, (_, val_indices) in enumerate(
            skf.split(cv_pool, cv_pool["label"])
        ):
            fold_of.iloc[val_indices] = fold_idx
    cv_pool["fold"] = fold_of.values

    ensure_dir(config.DATA_PROCESSED_DIR)
    save_csv(cv_pool, config.DATA_PROCESSED_DIR / "cv_pool.csv")
    save_csv(test_df.reset_index(drop=True), config.DATA_PROCESSED_DIR / "test.csv")

    return cv_pool, test_df.reset_index(drop=True)


def get_fold(cv_pool_df: pd.DataFrame, fold_index: int):
    """Split a fold-tagged CV pool (see create_cv_splits) into that fold's
    (train_df, val_df), by filtering the `fold` column."""
    train_df = (
        cv_pool_df[cv_pool_df["fold"] != fold_index]
        .drop(columns=["fold"])
        .reset_index(drop=True)
    )
    val_df = (
        cv_pool_df[cv_pool_df["fold"] == fold_index]
        .drop(columns=["fold"])
        .reset_index(drop=True)
    )
    return train_df, val_df
