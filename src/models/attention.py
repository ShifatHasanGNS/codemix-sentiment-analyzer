"""
Manually implemented attention mechanism layered on top of an RNN/LSTM
encoder, so the model can selectively weight relevant tokens in the
sequence rather than relying only on the final hidden state.

Designed to be composed with src.models.lstm.LSTMClassifier (or the RNN
variant) rather than duplicating the recurrent encoder here.

TODO:
- class AdditiveAttention(torch.nn.Module)
    - __init__(self, hidden_dim: int)
    - forward(self, encoder_outputs, mask=None) -> (context_vector, attention_weights)
- class AttentionClassifier(torch.nn.Module)
    - __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
               num_classes: int = 3, pretrained_embeddings=None,
               freeze_embeddings: bool = False)
        Wraps an LSTM/RNN encoder + AdditiveAttention + classification head.
    - forward(self, input_ids, lengths=None) -> logits of shape (batch_size, num_classes)
      (optionally also return attention_weights for inspection/UI display)
"""

import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, encoder_outputs, mask=None):
        raise NotImplementedError


class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int = 3, pretrained_embeddings=None,
                 freeze_embeddings: bool = False):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids, lengths=None):
        raise NotImplementedError
