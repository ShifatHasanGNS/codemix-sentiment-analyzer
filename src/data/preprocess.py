"""Shared text-cleaning utilities used before tokenization/feature extraction."""

import re

from src import config
from src.utils.io_utils import load_csv

_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)
_LATIN_RUN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

# Hand-curated (no offline corpus exists for these, unlike NLTK's English list).
_BANGLA_STOPWORDS = frozenset(
    [
        "এবং",
        "ও",
        "না",
        "যে",
        "এই",
        "সে",
        "তার",
        "এটি",
        "করে",
        "হয়",
        "থেকে",
        "জন্য",
        "কিন্তু",
        "আমি",
        "আপনি",
        "তুমি",
        "ইহা",
        "ওই",
        "কি",
        "কেন",
        "কোথায়",
        "সঙ্গে",
        "পর",
        "আগে",
        "এখন",
        "তবে",
        "তাই",
        "একটি",
    ]
)
_BANGLISH_STOPWORDS = frozenset(
    [
        "ami",
        "tumi",
        "apni",
        "ei",
        "oi",
        "ba",
        "na",
        "ke",
        "kintu",
        "jonno",
        "theke",
        "hoy",
        "kore",
        "ekta",
        "khub",
        "onek",
        "ekdom",
        "kono",
        "sob",
        "eta",
        "oke",
        "tar",
        "amar",
        "tomar",
    ]
)

_ENGLISH_STOPWORDS = None  # lazily-built cache; see _english_stopword_set()


def _english_stopword_set():
    global _ENGLISH_STOPWORDS
    if _ENGLISH_STOPWORDS is None:
        from nltk.corpus import stopwords

        try:
            _ENGLISH_STOPWORDS = frozenset(stopwords.words("english"))
        except LookupError:
            raise LookupError(
                "NLTK 'stopwords' corpus not found. Run "
                "`python -m nltk.downloader stopwords` once, then retry."
            )
    return _ENGLISH_STOPWORDS


def clean_text(text: str, lowercase: bool = True) -> str:
    """Normalize smart quotes, collapse whitespace, lowercase Latin runs only (Bangla has no case)."""
    text = (text or "").translate(_QUOTE_MAP)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = _LATIN_RUN_RE.sub(lambda m: m.group(0).lower(), text)
    return text


def remove_stopwords(tokens: list, language: str) -> list:
    """Drop stopwords for the given language condition; code_switched uses all three sets."""
    if language not in config.LANGUAGE_CONDITIONS:
        raise ValueError(
            f"language must be one of {config.LANGUAGE_CONDITIONS}, got {language!r}"
        )

    stopword_sets = []
    if language in ("english", "code_switched"):
        stopword_sets.append(_english_stopword_set())
    if language in ("bangla", "code_switched"):
        stopword_sets.append(_BANGLA_STOPWORDS)
    if language in ("banglish", "code_switched"):
        stopword_sets.append(_BANGLISH_STOPWORDS)

    return [t for t in tokens if not any(t.lower() in s for s in stopword_sets)]


def load_split(split_name: str):
    """Load data/processed/{split_name}.csv ("cv_pool" or "test")."""
    if split_name not in ("cv_pool", "test"):
        raise ValueError(
            f"split_name must be one of 'cv_pool', 'test', got {split_name!r}"
        )
    return load_csv(config.DATA_PROCESSED_DIR / f"{split_name}.csv")
