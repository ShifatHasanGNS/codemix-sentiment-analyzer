# Word2Vec embeddings trained on the project's own corpus, via gensim.

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
        workers=1,  # keeps training deterministic given `seed`
    )


def save_word2vec(model, path: str) -> None:
    model.save(str(path))


def load_word2vec(path: str):
    return Word2Vec.load(str(path))


def build_embedding_matrix(word2vec_model, vocab: dict, vector_size: int):
    # Pad token -> all-zero row; other OOV tokens -> small seeded random vector (not zero, so they stay distinguishable).
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
