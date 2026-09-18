"""
Basic RNN classifier: processes the token sequence step by step, maintaining
a hidden state, and classifies from the final hidden state (or pooled states).

TODO:
- class RNNClassifier(torch.nn.Module)
    - __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
               num_classes: int = 3, pretrained_embeddings=None,
               freeze_embeddings: bool = False)
    - forward(self, input_ids, lengths=None) -> logits of shape (batch_size, num_classes)
      (accept `lengths` if using pack_padded_sequence for variable-length inputs)
"""

import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int = 3, pretrained_embeddings=None,
                 freeze_embeddings: bool = False):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids, lengths=None):
        raise NotImplementedError
