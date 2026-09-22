# Additive attention over an LSTM encoder's outputs, instead of using only the final hidden state.

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, mask=None):
        scores = self.v(torch.tanh(self.W(encoder_outputs))).squeeze(-1)
        if mask is not None:
            # Fully-masked rows (all-padding input) are left unmasked to avoid softmax NaN.
            fully_masked = ~mask.any(dim=-1, keepdim=True)
            effective_mask = mask | fully_masked
            scores = scores.masked_fill(~effective_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class AttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
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

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = AdditiveAttention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths=None, return_attention: bool = False):
        embedded = self.embedding(input_ids)
        encoder_outputs, _ = self.encoder(embedded)

        mask = input_ids != 0  # padding_idx=0
        context, weights = self.attention(encoder_outputs, mask)
        logits = self.classifier(context)

        return (logits, weights) if return_attention else logits
