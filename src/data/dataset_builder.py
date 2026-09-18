"""
Assembles the Code-Mix Sentiment corpus -- entirely via code, no manual
writing or translation.

Source: BanglishRev (Hugging Face dataset id in src.config.HF_DATASET_ID),
a large-scale, real, publicly available dataset of Bangla / English /
Banglish / code-mixed e-commerce product reviews (~1.74M reviews across
~128k products), released under CC-BY-NC-SA-4.0
(https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset).
Product reviews are a universally familiar topic, which satisfies the
"corpus must be familiar to everyone" course requirement without needing any
hand-written or translated sentences.

Text only: the raw BanglishRev repo stores review text and review images as
completely separate files -- a single `reviews v1.json` (~1.9GB) plus 109
"Review Images N.zip" archives. download_banglishrev() fetches only
`reviews v1.json` by exact filename, so the image archives are never
downloaded, touched, or referenced anywhere in this project.

Pipeline (all automated):
    1. download_banglishrev()      -- fetch the raw dataset via code
    2. flatten_reviews()           -- raw nested JSON -> one row per review
    3. map_rating_to_label()       -- star rating -> positive/neutral/negative
    4. detect_language_condition() -- heuristic tagging into one of
                                       english / bangla / banglish / code_switched
    5. filter_familiar_categories()-- no-op by default (see its docstring);
                                       kept for traceability/override
    6. clean_and_dedupe()          -- drop empty/too-short/duplicate reviews
    7. subsample_balanced()        -- cap total size and balance by label
                                       (language left at its natural
                                       distribution -- see src.config)
    8. assemble_dataset()          -- orchestrates 1-7, writes data/raw/dataset.csv
    9. create_cv_splits()          -- stratified held-out test split, then a
                                       `fold` column (0..N_FOLDS-1) on the
                                       rest, writes data/processed/

Expected output schema (see data/README.md for full details), one row per
example: id, product_category, rating, label, language, text
"""

import json
import re

import pandas as pd

from src import config
from src.utils.io_utils import ensure_dir, save_csv

# A raw JSON record's "Reviews" entries carry engagement/moderation fields
# (Buyer ID, Likes, Dislikes, Reply, Images, ...) that this text-only
# project has no use for. Only these are ever read out of a review dict.
_REVIEW_RATING_FIELD = "Current Rating"
_REVIEW_TEXT_FIELD = "Review Content"
_PRODUCT_CATEGORY_FIELDS = ("Category", "Parent Category", "Root Category")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_ENGLISH_WORDS = None  # lazily-built cache; see _english_word_set()

# Common short (<3-char) English function words worth keeping even though
# most other 1-2 letter entries in NLTK's raw word list are dropped below.
_SHORT_ENGLISH_WORDS = frozenset([
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi",
    "if", "in", "is", "it", "me", "my", "no", "of", "ok", "on", "or", "so",
    "to", "up", "us", "we",
])


def download_banglishrev(cache_dir: str = None) -> str:
    """Fetch BanglishRev's review JSON and return its local file path.

    Downloads only the `reviews v1.json` file (~1.9GB) by exact filename via
    huggingface_hub, never the dataset repo's 109 "Review Images N.zip"
    archives -- this keeps the fetch text-only by construction rather than
    by filtering after the fact.
    """
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
    """Flatten BanglishRev's per-product JSON into one row per review.

    `raw_data` is either a path to the downloaded JSON file (as returned by
    download_banglishrev) or an already-parsed list of product dicts (used
    directly by tests on a handful of rows, without touching the network).

    Output columns: product_category, rating (int), text (str). Rows with
    an unparseable/out-of-range rating or empty review text are dropped
    here, since this is the one place that still has the raw, untrusted
    values -- downstream steps can then assume `rating` is a clean int.
    """
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

            rows.append({
                "product_category": category,
                "rating": rating,
                "text": text.strip(),
            })

    return pd.DataFrame(rows, columns=["product_category", "rating", "text"])


def map_rating_to_label(rating: int) -> str:
    """Map a 1-5 star rating to a 3-class sentiment label.

    Thresholds (src.config.RATING_TO_LABEL): 1-2 -> negative, 3 -> neutral,
    4-5 -> positive.
    """
    try:
        return config.RATING_TO_LABEL[int(rating)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"rating must be an int in {sorted(config.RATING_TO_LABEL)}, got {rating!r}")


def _english_word_set():
    """Lazily load and cache NLTK's English word list, lowercased.

    NLTK's raw `words` corpus includes ~165 one- and two-letter entries
    (abbreviations, archaic words, stray letters like "ar", "s", "m") that
    coincidentally match common romanized-Bangla syllables ("ar", "ta",
    "re") far too often, inflating false "english" matches on Banglish
    text. Those are dropped except for a small curated allowlist of
    genuinely common short English words.
    """
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
        _ENGLISH_WORDS = frozenset(
            w for w in all_words if len(w) >= 3
        ) | (_SHORT_ENGLISH_WORDS & all_words)
    return _ENGLISH_WORDS


def _script_of(token: str) -> str:
    """Classify one alphabetic token as 'bangla' or 'latin' by its first
    Bangla-range or other-alphabetic character (mixed-script tokens are
    rare in practice; the first matching character decides).

    Uses `str.isalpha()` rather than an ASCII a-z check, so that stylized
    Unicode letters (e.g. "Mathematical Bold Italic" glyphs, sometimes used
    to dodge naive filters -- real example seen in this corpus:
    "onak kharap" written as "𝒐𝒏𝒂𝒌 𝒌𝒉𝒂𝒓𝒂𝒑") are still recognized as latin
    script rather than silently contributing no signal at all.
    """
    bangla_lo, bangla_hi = config.BANGLA_UNICODE_RANGE
    for ch in token:
        code = ord(ch)
        if bangla_lo <= code <= bangla_hi:
            return "bangla"
        if ch.isalpha():
            return "latin"
    return "other"


def detect_language_condition(text: str) -> str:
    """Heuristically tag a review's language condition.

    Returns one of "english", "bangla", "banglish", "code_switched", using
    a Unicode-script + English-dictionary heuristic:
        - script ratio >= config.LANGUAGE_DETECTION_MIN_TOKEN_RATIO toward
          Bangla script -> "bangla"
        - script ratio >= that same threshold toward Latin script, and most
          of those Latin tokens are real English words -> "english";
          otherwise (mostly non-dictionary, i.e. romanized Bangla) ->
          "banglish"
        - neither script dominates (roughly balanced mix) -> "code_switched"

    Symmetric thresholding (rather than "any Bangla character at all means
    code-switched") avoids misclassifying an otherwise-English review that
    contains one stray Bangla character/typo.
    """
    threshold = config.LANGUAGE_DETECTION_MIN_TOKEN_RATIO
    tokens = [t for t in _WORD_RE.findall(text)]
    scripts = [_script_of(t) for t in tokens]
    bangla_count = scripts.count("bangla")
    latin_count = scripts.count("latin")
    total = bangla_count + latin_count

    if total == 0:
        # No alphabetic signal at all (e.g. emoji/digits only); such rows
        # are rare after MIN_REVIEW_LENGTH_CHARS filtering and are given an
        # arbitrary, harmless default rather than crashing the pipeline.
        return "english"

    bangla_ratio = bangla_count / total
    if bangla_ratio >= threshold:
        return "bangla"

    latin_ratio = latin_count / total
    if latin_ratio >= threshold:
        latin_tokens = [t.lower() for t, s in zip(tokens, scripts) if s == "latin"]
        english_words = _english_word_set()
        english_ratio = sum(t in english_words for t in latin_tokens) / len(latin_tokens)
        return "english" if english_ratio >= threshold else "banglish"

    return "code_switched"


def filter_familiar_categories(df: pd.DataFrame, allowed_categories) -> pd.DataFrame:
    """Optionally restrict to reviews whose raw category text matches one of
    `allowed_categories` (case-insensitive substring match).

    `allowed_categories=None` (the project default, src.config.ALLOWED_CATEGORIES)
    disables the restriction entirely and returns `df` unchanged. An earlier
    5-bucket keyword allowlist was found -- by sampling the raw corpus -- to
    exclude ~74% of it, including everyday items (Smartwatches, T-Shirts,
    Shampoo, Toothpaste, Diapers, Coffee) that are just as "universally
    familiar" as the original 5 buckets; the whole corpus is ordinary
    e-commerce consumer goods by construction. Kept as a real (non-default)
    option for anyone who wants to re-narrow the category scope.
    """
    if not allowed_categories:
        return df

    def matches(category: str) -> bool:
        category_lower = str(category).lower()
        return any(keyword.lower() in category_lower for keyword in allowed_categories)

    mask = df["product_category"].apply(matches)
    return df[mask].reset_index(drop=True)


def clean_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty/too-short/duplicate/content-free reviews and strip
    control characters.

    "Content-free" means zero alphabetic tokens under _WORD_RE -- e.g. a
    review that's pure mojibake ("??? ?????? ...", a garbled-encoding
    artifact seen in this corpus). Such rows pass the length check but
    carry no learnable signal, and downstream they become an all-padding
    input that drives attention softmax to 0/0 = NaN, permanently
    corrupting a model's weights via backprop. Filtering them here, once,
    is cheaper and more robust than handling it at every consumer.
    """
    df = df.copy()
    df["text"] = df["text"].str.replace(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", regex=True).str.strip()
    df = df[df["text"].str.len() >= config.MIN_REVIEW_LENGTH_CHARS]
    df = df[df["text"].apply(lambda t: len(_WORD_RE.findall(t)) > 0)]
    df = df.drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def subsample_balanced(df: pd.DataFrame, target_size: int, per_group_cap: int = None) -> pd.DataFrame:
    """Randomly (seeded) subsample to `target_size` total rows, balanced by
    label, with each label's rows spread as evenly as possible across
    language conditions.

    Two-level, label-first design: `target_size` is split evenly across the
    3 labels, and only *within* a label does the existing best-effort-then-
    redistribute logic run across its 4 language groups (so a naturally
    scarce condition like code-switched contributes everything it has, and
    the shortfall is filled from the other language conditions *of that
    same label*). This label-first structure matters at scale: this
    corpus's raw label distribution is heavily skewed (~82% positive), so a
    single flat redistribution across all 12 (label, language) cells would
    let positive's much larger leftover pool dominate the redistribution
    and silently break label balance -- verified this would happen before
    picking this design (see the docstring's real numbers in src.config).
    """
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
            extra = pool.sample(n=min(shortfall, len(pool)), random_state=config.RANDOM_SEED)
            chosen_parts.append(extra)

        sampled_parts.append(pd.concat(chosen_parts))

    result = pd.concat(sampled_parts).sample(frac=1, random_state=config.RANDOM_SEED)
    return result.reset_index(drop=True)


def assemble_dataset(cache_dir: str = None) -> pd.DataFrame:
    """Run the full collection pipeline and write data/raw/dataset.csv."""
    raw_path = download_banglishrev(cache_dir=cache_dir)
    df = flatten_reviews(raw_path)

    df["label"] = df["rating"].apply(map_rating_to_label)
    df["language"] = df["text"].apply(detect_language_condition)

    df = filter_familiar_categories(df, config.ALLOWED_CATEGORIES)
    df = clean_and_dedupe(df)

    # Per-language cap within a label: allow a language condition to supply
    # up to 3x its even split of that label's quota, so plentiful conditions
    # (english/bangla/banglish) can absorb the shortfall left by a scarce
    # one (code_switched) without an overly tight ceiling.
    label_quota = config.TARGET_DATASET_SIZE // len(config.LABELS)
    per_group_cap = max(50, (label_quota // len(config.LANGUAGE_CONDITIONS)) * 3)
    df = subsample_balanced(df, config.TARGET_DATASET_SIZE, per_group_cap)

    df.insert(0, "id", [f"cm{idx:06d}" for idx in range(len(df))])
    df = df[["id", "product_category", "rating", "label", "language", "text"]]

    save_csv(df, config.DATA_RAW_DIR / "dataset.csv")
    return df


def create_cv_splits(df: pd.DataFrame, n_folds: int = None, test_holdout: float = None):
    """Carve off a stratified held-out test set, then assign every remaining
    row a `fold` column (0..n_folds-1) for k-fold cross-validation.

    Writes data/processed/test.csv (held-out, untouched by any training/CV
    step) and data/processed/cv_pool.csv (the rest, with the `fold` column).
    Both the test split and the fold assignment stratify by (label,
    language) jointly where possible, falling back to label alone if some
    (label, language) group is too small to stratify (a real possibility
    for the naturally scarce code_switched condition).
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    n_folds = n_folds or config.N_FOLDS
    test_holdout = test_holdout if test_holdout is not None else config.TEST_HOLDOUT_SPLIT

    strata = df["label"] + "_" + df["language"]
    try:
        cv_pool, test_df = train_test_split(
            df, test_size=test_holdout, random_state=config.RANDOM_SEED, stratify=strata,
        )
        fold_strata_source = "joint"
    except ValueError:
        print("Warning: some (label, language) group too small to stratify jointly for "
              "the test split; falling back to stratifying by label only.")
        cv_pool, test_df = train_test_split(
            df, test_size=test_holdout, random_state=config.RANDOM_SEED, stratify=df["label"],
        )
        fold_strata_source = "label"

    cv_pool = cv_pool.reset_index(drop=True)
    fold_strata = (
        cv_pool["label"] + "_" + cv_pool["language"] if fold_strata_source == "joint" else cv_pool["label"]
    )
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
        fold_of = pd.Series(index=cv_pool.index, dtype=int)
        for fold_idx, (_, val_indices) in enumerate(skf.split(cv_pool, fold_strata)):
            fold_of.iloc[val_indices] = fold_idx
    except ValueError:
        print("Warning: some (label, language) group too small to stratify jointly for "
              "k-fold assignment; falling back to stratifying by label only.")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
        fold_of = pd.Series(index=cv_pool.index, dtype=int)
        for fold_idx, (_, val_indices) in enumerate(skf.split(cv_pool, cv_pool["label"])):
            fold_of.iloc[val_indices] = fold_idx
    cv_pool["fold"] = fold_of.values

    ensure_dir(config.DATA_PROCESSED_DIR)
    save_csv(cv_pool, config.DATA_PROCESSED_DIR / "cv_pool.csv")
    save_csv(test_df.reset_index(drop=True), config.DATA_PROCESSED_DIR / "test.csv")

    return cv_pool, test_df.reset_index(drop=True)


def get_fold(cv_pool_df: pd.DataFrame, fold_index: int):
    """Split a fold-tagged CV pool (see create_cv_splits) into that fold's
    (train_df, val_df), by filtering the `fold` column."""
    train_df = cv_pool_df[cv_pool_df["fold"] != fold_index].drop(columns=["fold"]).reset_index(drop=True)
    val_df = cv_pool_df[cv_pool_df["fold"] == fold_index].drop(columns=["fold"]).reset_index(drop=True)
    return train_df, val_df
