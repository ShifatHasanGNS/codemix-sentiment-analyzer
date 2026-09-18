"""
Classical statistical feature extraction: N-Gram counts, Bag-of-Words, TF-IDF.

Each representation is paired (in src/training/train_classical.py) with a
simple classifier (e.g. Naive Bayes / Logistic Regression from scikit-learn)
to form the project's statistical baselines.

All three vectorizers use src.data.tokenizer.CodeMixTokenizer for splitting
text, so classical and neural pipelines tokenize identically.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src import config
from src.data.tokenizer import CodeMixTokenizer

_tokenizer = CodeMixTokenizer()


def fit_ngram_vectorizer(corpus: list, n_range=(1, 2)):
    vectorizer = CountVectorizer(
        tokenizer=_tokenizer.tokenize, token_pattern=None,
        ngram_range=n_range, max_features=config.CLASSICAL_MAX_FEATURES,
    )
    vectorizer.fit(corpus)
    return vectorizer


def fit_bow_vectorizer(corpus: list):
    vectorizer = CountVectorizer(
        tokenizer=_tokenizer.tokenize, token_pattern=None,
        ngram_range=(1, 1), max_features=config.CLASSICAL_MAX_FEATURES,
    )
    vectorizer.fit(corpus)
    return vectorizer


def fit_tfidf_vectorizer(corpus: list):
    vectorizer = TfidfVectorizer(
        tokenizer=_tokenizer.tokenize, token_pattern=None,
        ngram_range=(1, 2), max_features=config.CLASSICAL_MAX_FEATURES,
    )
    vectorizer.fit(corpus)
    return vectorizer


def transform(vectorizer, corpus: list):
    return vectorizer.transform(corpus)
