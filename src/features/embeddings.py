"""
Word2Vec embeddings trained on the project's own corpus (via gensim),
used as an alternative to randomly initialized embeddings inside the
from-scratch neural models.

TODO:
- train_word2vec(tokenized_corpus: list[list[str]], vector_size: int,
                  window: int, min_count: int, sg: int) -> gensim.models.Word2Vec
- save_word2vec(model, path) / load_word2vec(path)
- build_embedding_matrix(word2vec_model, vocab: dict[str, int], vector_size: int)
    -> numpy.ndarray of shape (vocab_size, vector_size), with a documented
    strategy for out-of-vocabulary tokens (e.g. random init or zero vector).
"""


def train_word2vec(tokenized_corpus: list, vector_size: int = 100,
                    window: int = 5, min_count: int = 1, sg: int = 1):
    raise NotImplementedError


def save_word2vec(model, path: str) -> None:
    raise NotImplementedError


def load_word2vec(path: str):
    raise NotImplementedError


def build_embedding_matrix(word2vec_model, vocab: dict, vector_size: int):
    raise NotImplementedError
