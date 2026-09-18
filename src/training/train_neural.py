"""
Shared training loop for the five from-scratch neural models (ANN, RNN,
LSTM, Attention, Transformer), parameterized by model name so the same
script trains any of them.

TODO:
- build_model(model_name: str, vocab_size: int, embedding_matrix=None) -> torch.nn.Module
    model_name in {"ann", "rnn", "lstm", "attention", "transformer"}, dispatching
    to the matching class in src.models.*.
- build_dataloaders(train_df, val_df, tokenizer, batch_size) -> (train_loader, val_loader)
    Wraps a torch.utils.data.Dataset that encodes text via the tokenizer and
    returns (input_ids, length, label) tuples.
- train_one_model(model_name: str, num_epochs: int, learning_rate: float) -> dict
    Standard PyTorch train/val loop (forward, loss via CrossEntropyLoss,
    backward, optimizer step); tracks best validation accuracy; saves the
    best checkpoint to models_saved/{model_name}.pt via src.utils.io_utils.
- main()
    CLI entry point looping over all five models (or a subset via an
    argparse flag), calling train_one_model for each.
"""


def build_model(model_name: str, vocab_size: int, embedding_matrix=None):
    raise NotImplementedError


def build_dataloaders(train_df, val_df, tokenizer, batch_size: int):
    raise NotImplementedError


def train_one_model(model_name: str, num_epochs: int = 10, learning_rate: float = 1e-3):
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
