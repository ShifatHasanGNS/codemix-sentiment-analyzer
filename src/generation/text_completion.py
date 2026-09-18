"""
Small bonus sequence-generation demo: autoregressive next-word / next-sentence
completion built on top of a trained LSTM or Transformer checkpoint (reusing
its encoder, with a language-modeling head instead of / alongside the
classification head).

The base checkpoint was trained for 3-class classification, not language
modeling, so it has no next-token head of its own. TextCompletionModel reuses
its embedding + encoder weights as a warm start and briefly trains a small
linear LM head (config.TEXT_COMPLETION_LM_EPOCHS epochs) on the same train
split via teacher forcing -- enough for qualitative "does this look
plausible" demos, not a serious generative model.

This is evaluated qualitatively only (see src/evaluation/error_analysis.py
docstring) -- not scored numerically.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data.preprocess import load_split
from src.training.train_neural import build_model
from src.utils.io_utils import load_checkpoint
from src.utils.seed import set_seed

_SUPPORTED_BASE_MODELS = ("lstm", "transformer")


class _NextTokenDataset(Dataset):
    """(input_ids[:-1], input_ids[1:]) teacher-forcing pairs, using the same
    tokenizer/vocab the base classifier was trained with."""

    def __init__(self, texts, tokenizer, max_length):
        self.examples = [
            torch.tensor(tokenizer.encode(text, max_length), dtype=torch.long)
            for text in texts
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return ids[:-1], ids[1:]


class TextCompletionModel:
    def __init__(self, base_model_checkpoint_path: str, tokenizer,
                 lm_epochs: int = config.TEXT_COMPLETION_LM_EPOCHS, learning_rate: float = 1e-3):
        checkpoint = load_checkpoint(base_model_checkpoint_path)
        model_name = checkpoint["model_name"]
        if model_name not in _SUPPORTED_BASE_MODELS:
            raise ValueError(
                f"base_model_checkpoint_path must be a {_SUPPORTED_BASE_MODELS} "
                f"checkpoint, got model_name={model_name!r}"
            )

        set_seed(config.RANDOM_SEED)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model = build_model(model_name, checkpoint["vocab_size"])
        base_model.load_state_dict(checkpoint["state_dict"])
        self.embedding = base_model.embedding.to(self.device)

        if model_name == "lstm":
            self.encoder = base_model.lstm.to(self.device)
            num_directions = 2 if self.encoder.bidirectional else 1
            hidden_dim = self.encoder.hidden_size * num_directions
        else:
            self.encoder = base_model.encoder.to(self.device)
            self.positional_encoding = base_model.positional_encoding.to(self.device)
            self.embedding_dim = base_model.embedding_dim
            hidden_dim = base_model.embedding_dim

        self.lm_head = nn.Linear(hidden_dim, checkpoint["vocab_size"]).to(self.device)
        self._train_lm_head(lm_epochs, learning_rate)

    def _encode_sequence(self, input_ids):
        """Per-step hidden states, shape (batch, seq_len, hidden_dim)."""
        embedded = self.embedding(input_ids)
        if self.model_name == "lstm":
            outputs, _ = self.encoder(embedded)
            return outputs

        embedded = embedded * (self.embedding_dim ** 0.5)
        embedded = self.positional_encoding(embedded)
        padding_mask = input_ids == 0
        return self.encoder(embedded, src_key_padding_mask=padding_mask)

    def _train_lm_head(self, num_epochs, learning_rate):
        train_df = load_split("cv_pool")
        if len(train_df) > config.TEXT_COMPLETION_TRAIN_SUBSAMPLE_SIZE:
            train_df = train_df.sample(
                n=config.TEXT_COMPLETION_TRAIN_SUBSAMPLE_SIZE, random_state=config.RANDOM_SEED
            )
        dataset = _NextTokenDataset(train_df["text"].tolist(), self.tokenizer, config.MAX_SEQUENCE_LENGTH)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        params = list(self.embedding.parameters()) + list(self.encoder.parameters()) + list(self.lm_head.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad> targets

        for epoch in range(num_epochs):
            total_loss = 0.0
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                logits = self.lm_head(self._encode_sequence(inputs))
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[text_completion:{self.model_name}] lm epoch {epoch + 1}/{num_epochs} "
                  f"loss={total_loss / len(loader):.4f}")

    def generate(self, prompt: str, max_new_tokens: int = 20, temperature: float = 1.0) -> str:
        unk_id = self.tokenizer.token_to_id["<unk>"]
        pad_id = self.tokenizer.token_to_id["<pad>"]
        token_ids = [self.tokenizer.token_to_id.get(t, unk_id) for t in self.tokenizer.tokenize(prompt)]
        if not token_ids:
            token_ids = [unk_id]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = token_ids[-config.MAX_SEQUENCE_LENGTH:]
                input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
                next_logits = self.lm_head(self._encode_sequence(input_ids))[0, -1]

                if temperature <= 0:
                    next_id = int(next_logits.argmax())
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1))

                if next_id == pad_id:
                    break
                token_ids.append(next_id)

        generated_tokens = [self.tokenizer.id_to_token.get(i, "<unk>") for i in token_ids]
        return " ".join(generated_tokens)
