"""
ANN baseline: feed-forward network over a pooled representation of the input
(e.g. averaged Word2Vec embeddings, or a Bag-of-Words/TF-IDF vector).

Expected to underperform the sequence models since it has no notion of word
order or long-range dependency -- serves as the project's neural lower bound.
"""

import torch.nn as nn


class ANNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
