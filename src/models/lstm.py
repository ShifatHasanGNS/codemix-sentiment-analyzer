"""
LSTM classifier: same role as the RNN baseline but with gated long-range
memory, used to test whether better long-range retention helps on
code-switched input specifically.

TODO:
- class LSTMClassifier(torch.nn.Module)
    - __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
               num_classes: int = 3, num_layers: int = 1, bidirectional: bool = False,
               pretrained_embeddings=None, freeze_embeddings: bool = False)
    - forward(self, input_ids, lengths=None) -> logits of shape (batch_size, num_classes)
"""

import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int = 3, num_layers: int = 1, bidirectional: bool = False,
                 pretrained_embeddings=None, freeze_embeddings: bool = False):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids, lengths=None):
        raise NotImplementedError
