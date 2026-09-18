"""
Small Transformer encoder implemented entirely from scratch (no pretrained
transformer library), consisting of a limited number of self-attention
layers with positional encoding, ending in a classification head.

TODO:
- class PositionalEncoding(torch.nn.Module)
    - __init__(self, embedding_dim: int, max_len: int)
    - forward(self, x) -> x + positional encodings
- class TransformerEncoderClassifier(torch.nn.Module)
    - __init__(self, vocab_size: int, embedding_dim: int, num_heads: int,
               num_layers: int, ff_dim: int, num_classes: int = 3,
               max_len: int = 128, dropout: float = 0.1)
        Can use torch.nn.TransformerEncoderLayer/TransformerEncoder as the
        underlying self-attention blocks (still "from scratch" in the sense
        of no pretrained weights) or a fully custom multi-head attention
        implementation -- pick one and document the choice here.
    - forward(self, input_ids, attention_mask=None) -> logits of shape
      (batch_size, num_classes)  (e.g. via mean-pooling or a [CLS]-style token)
"""

import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 512):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int,
                 num_layers: int, ff_dim: int, num_classes: int = 3,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids, attention_mask=None):
        raise NotImplementedError
