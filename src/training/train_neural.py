"""Shared k-fold CV training loop for the 5 from-scratch neural models; per-fold tokenizer/Word2Vec are built from that fold's train partition only, to avoid val-fold leakage."""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data.dataset_builder import get_fold
from src.data.preprocess import load_split
from src.data.tokenizer import CodeMixTokenizer
from src.features.embeddings import (
    build_embedding_matrix,
    save_word2vec,
    train_word2vec,
)
from src.models.ann import ANNClassifier
from src.models.attention import AttentionClassifier
from src.models.lstm import LSTMClassifier
from src.models.rnn import RNNClassifier
from src.models.transformer import TransformerEncoderClassifier
from src.utils.io_utils import ensure_dir, save_checkpoint
from src.utils.seed import set_seed

MODEL_NAMES = ["ann", "rnn", "lstm", "attention", "transformer"]

# Keyed by fold index or "final"; reused across all 5 models so assets are built once per fold.
_ASSETS_CACHE = {}


class _EncodedDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.input_ids = [
            torch.tensor(
                tokenizer.encode(text, config.MAX_SEQUENCE_LENGTH), dtype=torch.long
            )
            for text in df["text"]
        ]
        self.lengths = [
            max(1, min(len(tokenizer.tokenize(text)), config.MAX_SEQUENCE_LENGTH))
            for text in df["text"]
        ]
        self.labels = [config.LABELS.index(label) for label in df["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.lengths[idx], self.labels[idx]


def _collate(batch):
    input_ids = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return input_ids, lengths, labels


def build_model(model_name: str, vocab_size: int, embedding_matrix=None):
    num_classes = len(config.LABELS)

    if model_name == "ann":
        return ANNClassifier(
            config.EMBEDDING_DIM, config.ANN_HIDDEN_DIMS, num_classes, config.DROPOUT
        )
    if model_name == "rnn":
        return RNNClassifier(
            vocab_size,
            config.EMBEDDING_DIM,
            config.HIDDEN_DIM,
            num_classes,
            pretrained_embeddings=embedding_matrix,
        )
    if model_name == "lstm":
        return LSTMClassifier(
            vocab_size,
            config.EMBEDDING_DIM,
            config.HIDDEN_DIM,
            num_classes,
            num_layers=config.LSTM_NUM_LAYERS,
            bidirectional=config.LSTM_BIDIRECTIONAL,
            pretrained_embeddings=embedding_matrix,
        )
    if model_name == "attention":
        return AttentionClassifier(
            vocab_size,
            config.EMBEDDING_DIM,
            config.HIDDEN_DIM,
            num_classes,
            pretrained_embeddings=embedding_matrix,
        )
    if model_name == "transformer":
        return TransformerEncoderClassifier(
            vocab_size,
            config.EMBEDDING_DIM,
            num_heads=config.TRANSFORMER_NUM_HEADS,
            num_layers=config.TRANSFORMER_NUM_LAYERS,
            ff_dim=config.TRANSFORMER_FF_DIM,
            num_classes=num_classes,
            max_len=config.MAX_SEQUENCE_LENGTH,
            dropout=config.DROPOUT,
        )
    raise ValueError(f"model_name must be one of {MODEL_NAMES}, got {model_name!r}")


def build_dataloaders(train_df, val_df, tokenizer, batch_size: int):
    train_loader = DataLoader(
        _EncodedDataset(train_df, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = None
    if val_df is not None:
        val_loader = DataLoader(
            _EncodedDataset(val_df, tokenizer),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
        )
    return train_loader, val_loader


def _build_assets(train_df, val_df, cache_key):
    """Build tokenizer/Word2Vec/embedding-matrix/dataloaders from train_df only, cached by cache_key."""
    if cache_key in _ASSETS_CACHE:
        return _ASSETS_CACHE[cache_key]

    set_seed(config.RANDOM_SEED)
    tokenizer = CodeMixTokenizer()
    tokenizer.build_vocab(train_df["text"].tolist(), min_freq=config.MIN_TOKEN_FREQ)

    tokenized_corpus = [tokenizer.tokenize(t) for t in train_df["text"]]
    w2v = train_word2vec(
        tokenized_corpus,
        vector_size=config.EMBEDDING_DIM,
        min_count=config.MIN_TOKEN_FREQ,
    )
    embedding_matrix = build_embedding_matrix(
        w2v, tokenizer.token_to_id, config.EMBEDDING_DIM
    )

    train_loader, val_loader = build_dataloaders(
        train_df, val_df, tokenizer, config.BATCH_SIZE
    )

    # Inverse-frequency class weights counteract the corpus's skew toward "positive".
    label_counts = torch.bincount(
        torch.tensor(train_loader.dataset.labels), minlength=len(config.LABELS)
    ).float()
    class_weights = label_counts.sum() / (len(config.LABELS) * label_counts)

    if cache_key == "final":
        # Saved tokenizer/Word2Vec are loaded by the app, text-completion demo, and error analysis.
        ensure_dir(config.MODELS_SAVED_DIR)
        tokenizer.save(config.MODELS_SAVED_DIR / "tokenizer.json")
        save_word2vec(w2v, config.MODELS_SAVED_DIR / "word2vec.model")

    assets = {
        "tokenizer": tokenizer,
        "embedding_matrix": embedding_matrix,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "vocab_size": tokenizer.vocab_size,
        "class_weights": class_weights,
    }
    _ASSETS_CACHE[cache_key] = assets
    return assets


def _pool_embeddings(input_ids, embedding_matrix_tensor):
    """Mean-pool a Word2Vec embedding lookup over non-pad tokens, for the ANN model's input."""
    embedded = F.embedding(input_ids, embedding_matrix_tensor, padding_idx=0)
    mask = (input_ids != 0).unsqueeze(-1).float()
    return (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


def _forward(model, model_name, input_ids, lengths, embedding_matrix_tensor):
    if model_name == "ann":
        return model(_pool_embeddings(input_ids, embedding_matrix_tensor))
    if model_name == "transformer":
        return model(input_ids)
    return model(input_ids, lengths)


def _evaluate(model, model_name, loader, embedding_matrix_tensor, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids, lengths, labels = (
                input_ids.to(device),
                lengths.to(device),
                labels.to(device),
            )
            logits = _forward(
                model, model_name, input_ids, lengths, embedding_matrix_tensor
            )
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_one_model(
    model_name: str,
    train_loader,
    val_loader,
    embedding_matrix,
    vocab_size: int,
    class_weights,
    num_epochs: int,
    learning_rate: float,
):
    """Train model_name on already-built assets; val_loader=None means the full-pool refit (no per-epoch validation)."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}, got {model_name!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_matrix_tensor = torch.as_tensor(embedding_matrix, dtype=torch.float32).to(
        device
    )

    model = build_model(model_name, vocab_size, embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_accuracy = -1.0
    best_state = None
    val_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        for input_ids, lengths, labels in train_loader:
            input_ids, lengths, labels = (
                input_ids.to(device),
                lengths.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            logits = _forward(
                model, model_name, input_ids, lengths, embedding_matrix_tensor
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            val_accuracy = _evaluate(
                model, model_name, val_loader, embedding_matrix_tensor, device
            )
            val_accuracy_history.append(val_accuracy)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(
                f"[{model_name}] epoch {epoch + 1}/{num_epochs}  val_acc={val_accuracy:.4f}"
            )
        else:
            print(
                f"[{model_name}] epoch {epoch + 1}/{num_epochs}  (final refit, no val)"
            )

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "state_dict": best_state if val_loader is not None else final_state,
        "best_val_accuracy": best_val_accuracy if val_loader is not None else None,
        "history": val_accuracy_history,
    }


def run_cross_validation(model_name: str):
    """k-fold CV for one model; returns per-fold accuracies plus mean/std."""
    cv_pool = load_split("cv_pool")
    fold_accuracies = []

    for fold_index in range(config.N_FOLDS):
        train_df, val_df = get_fold(cv_pool, fold_index)
        assets = _build_assets(train_df, val_df, cache_key=fold_index)
        result = train_one_model(
            model_name,
            assets["train_loader"],
            assets["val_loader"],
            assets["embedding_matrix"],
            assets["vocab_size"],
            assets["class_weights"],
            config.NUM_EPOCHS,
            config.LEARNING_RATE,
        )
        fold_accuracies.append(result["best_val_accuracy"])
        print(
            f"-> {model_name} fold {fold_index + 1}/{config.N_FOLDS}: "
            f"best_val_acc={result['best_val_accuracy']:.4f}"
        )

    return {
        "model_name": model_name,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
    }


def refit_final(model_name: str):
    """Train on the full CV pool (no held-out val) for the deployed checkpoint."""
    cv_pool = load_split("cv_pool").drop(columns=["fold"])
    assets = _build_assets(cv_pool, None, cache_key="final")
    result = train_one_model(
        model_name,
        assets["train_loader"],
        None,
        assets["embedding_matrix"],
        assets["vocab_size"],
        assets["class_weights"],
        config.NUM_EPOCHS,
        config.LEARNING_RATE,
    )

    save_checkpoint(
        {
            "model_name": model_name,
            "state_dict": result["state_dict"],
            "vocab_size": assets["vocab_size"],
        },
        config.MODELS_SAVED_DIR / f"{model_name}.pt",
    )

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=",".join(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]

    cv_results = []
    for model_name in models_to_run:
        cv_result = run_cross_validation(model_name)
        cv_results.append(cv_result)
        print(
            f"-> {model_name}: CV mean_acc={cv_result['mean_accuracy']:.4f} "
            f"(+/- {cv_result['std_accuracy']:.4f}) over {config.N_FOLDS} folds"
        )

        refit_final(model_name)
        print(
            f"-> {model_name}: refit on full CV pool -> models_saved/{model_name}.pt\n"
        )

    print("=== CV Summary ===")
    for cv_result in cv_results:
        print(
            f"{cv_result['model_name']:12s}  mean_acc={cv_result['mean_accuracy']:.4f} "
            f"(+/- {cv_result['std_accuracy']:.4f})"
        )

    return cv_results


if __name__ == "__main__":
    main()
