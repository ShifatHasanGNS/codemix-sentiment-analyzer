"""
Small Transformer encoder implemented entirely from scratch (no pretrained
transformer library), consisting of a limited number of self-attention
layers with positional encoding, ending in a classification head.

Uses torch.nn.TransformerEncoderLayer/TransformerEncoder as the underlying
multi-head self-attention blocks -- this is still "from scratch" in the
project's sense (random init, no pretrained weights loaded), it just reuses
PyTorch's standard building block instead of re-deriving scaled dot-product
attention by hand, matching the level of the RNN/LSTM models above (which
likewise use torch.nn.RNN/LSTM rather than a hand-rolled recurrence).
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embedding_dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int,
                 num_layers: int, ff_dim: int, num_classes: int = 3,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            padding_mask = input_ids == 0  # padding_idx=0; True = ignore
        else:
            padding_mask = attention_mask == 0

        embedded = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        embedded = self.positional_encoding(embedded)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-6)

        return self.classifier(self.dropout(pooled))
