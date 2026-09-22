# LSTM classifier: gated long-range memory vs. the vanilla RNN baseline.

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
        num_layers: int = 1,
        bidirectional: bool = False,
        pretrained_embeddings=None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            weight = torch.as_tensor(pretrained_embeddings, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                weight, freeze=freeze_embeddings, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.num_directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, input_ids, lengths=None):
        embedded = self.embedding(input_ids)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)

        # Last layer; concat both directions if bidirectional.
        if self.num_directions == 2:
            final = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            final = hidden[-1]

        return self.classifier(final)
