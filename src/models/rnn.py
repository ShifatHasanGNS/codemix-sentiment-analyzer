# Vanilla RNN: classifies from the final hidden state.

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int = 3, pretrained_embeddings=None,
                 freeze_embeddings: bool = False):
        super().__init__()
        if pretrained_embeddings is not None:
            weight = torch.as_tensor(pretrained_embeddings, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                weight, freeze=freeze_embeddings, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths=None):
        embedded = self.embedding(input_ids)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.rnn(packed)
        else:
            _, hidden = self.rnn(embedded)

        return self.classifier(hidden[-1])
