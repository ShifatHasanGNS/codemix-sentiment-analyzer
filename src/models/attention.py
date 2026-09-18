"""
Manually implemented attention mechanism layered on top of an RNN/LSTM
encoder, so the model can selectively weight relevant tokens in the
sequence rather than relying only on the final hidden state.

Designed to be composed with src.models.lstm.LSTMClassifier (or the RNN
variant) rather than duplicating the recurrent encoder here.
"""

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention: score(h) = v^T tanh(W h)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, mask=None):
        # encoder_outputs: (batch, seq_len, hidden_dim)
        # mask: (batch, seq_len), True at valid (non-pad) positions
        scores = self.v(torch.tanh(self.W(encoder_outputs))).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            # A fully-masked row (every position padding, e.g. text that
            # tokenized to nothing) would otherwise mask every score to
            # -inf, making softmax produce NaN that permanently corrupts
            # the model's weights on the very next backward pass. Such
            # rows are filtered out of the training data (see
            # dataset_builder.clean_and_dedupe), but live inference (the
            # Streamlit app) can still hand the model arbitrary text, so
            # this guard leaves those rows fully unmasked -- an attention
            # distribution over padding is meaningless either way, but
            # "meaningless and finite" beats "NaN that poisons training".
            fully_masked = ~mask.any(dim=-1, keepdim=True)
            effective_mask = mask | fully_masked
            scores = scores.masked_fill(~effective_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class AttentionClassifier(nn.Module):
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
