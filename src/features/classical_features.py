"""
Classical statistical feature extraction: N-Gram counts, Bag-of-Words, TF-IDF.

Each representation is paired (in src/training/train_classical.py) with a
simple classifier (e.g. Naive Bayes / Logistic Regression from scikit-learn)
to form the project's statistical baselines.

TODO:
- fit_ngram_vectorizer(corpus: list[str], n_range=(1, 2)) -> vectorizer
- fit_bow_vectorizer(corpus: list[str]) -> vectorizer
- fit_tfidf_vectorizer(corpus: list[str]) -> vectorizer
    All three can likely reuse sklearn.feature_extraction.text.CountVectorizer /
    TfidfVectorizer with a custom `tokenizer=` callable from
    src.data.tokenizer.CodeMixTokenizer, keeping tokenization consistent
    with the neural pipeline.
- transform(vectorizer, corpus: list[str]) -> scipy.sparse matrix / ndarray
"""


def fit_ngram_vectorizer(corpus: list, n_range=(1, 2)):
    raise NotImplementedError


def fit_bow_vectorizer(corpus: list):
    raise NotImplementedError


def fit_tfidf_vectorizer(corpus: list):
    raise NotImplementedError


def transform(vectorizer, corpus: list):
    raise NotImplementedError
