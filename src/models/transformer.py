# Small from-scratch Transformer encoder (random init, no pretrained weights) + classification head.

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
        self.register_buffer("pe", pe.unsqueeze(0))

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
            padding_mask = input_ids == 0
        else:
            padding_mask = attention_mask == 0

        embedded = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        embedded = self.positional_encoding(embedded)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-6)

        return self.classifier(self.dropout(pooled))
