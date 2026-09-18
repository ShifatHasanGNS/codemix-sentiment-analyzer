"""
Word2Vec embeddings trained on the project's own corpus (via gensim),
used as an alternative to randomly initialized embeddings inside the
from-scratch neural models.
"""

import numpy as np
from gensim.models import Word2Vec

from src import config
from src.data.tokenizer import PAD_TOKEN


def train_word2vec(tokenized_corpus: list, vector_size: int = 100,
                    window: int = 5, min_count: int = 1, sg: int = 1):
    return Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        seed=config.RANDOM_SEED,
        workers=1,  # workers=1 keeps gensim's training deterministic given `seed`
    )


def save_word2vec(model, path: str) -> None:
    model.save(str(path))


def load_word2vec(path: str):
    return Word2Vec.load(str(path))


def build_embedding_matrix(word2vec_model, vocab: dict, vector_size: int):
    """Build a (vocab_size, vector_size) embedding matrix aligned to
    `vocab` (a token -> id mapping, e.g. CodeMixTokenizer.token_to_id).

    OOV strategy: the pad token gets an all-zero row (so it never
    contributes gradient signal); every other token missing from the
    Word2Vec vocabulary (including <unk>) gets a small seeded random
    vector, rather than zeros, so the model can still tell OOV tokens
    apart from padding and from each other.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    matrix = np.zeros((len(vocab), vector_size), dtype=np.float32)

    for token, idx in vocab.items():
        if token == PAD_TOKEN:
            continue
        if token in word2vec_model.wv:
            matrix[idx] = word2vec_model.wv[token]
        else:
            matrix[idx] = rng.normal(scale=0.1, size=vector_size)

    return matrix
