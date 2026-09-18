"""
ANN baseline: feed-forward network over a pooled representation of the input
(e.g. averaged Word2Vec embeddings, or a Bag-of-Words/TF-IDF vector).

Expected to underperform the sequence models since it has no notion of word
order or long-range dependency -- serves as the project's neural lower bound.

TODO:
- class ANNClassifier(torch.nn.Module)
    - __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int = 3,
               dropout: float = 0.3)
    - forward(self, x) -> logits of shape (batch_size, num_classes)
"""

import torch.nn as nn


class ANNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
