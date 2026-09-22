"""Fine-tunes mBERT on a bounded subsample (single split, no k-fold CV -- full CV at this scale is too slow on CPU)."""

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data.preprocess import load_split
from src.models.bert_model import encode_batch, load_bert_and_tokenizer
from src.utils.io_utils import ensure_dir
from src.utils.seed import set_seed


def build_dataloaders(train_df, val_df, tokenizer, batch_size: int):
    loaders = []
    for df, shuffle in [(train_df, True), (val_df, False)]:
        encoded = encode_batch(
            tokenizer, df["text"].tolist(), config.BERT_MAX_SEQUENCE_LENGTH
        )
        labels = torch.tensor(
            [config.LABELS.index(label) for label in df["label"]], dtype=torch.long
        )
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    return tuple(loaders)


def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def fine_tune(num_epochs: int = config.BERT_NUM_EPOCHS, learning_rate: float = 2e-5):
    set_seed(config.RANDOM_SEED)
    cv_pool = load_split("cv_pool")

    train_df = cv_pool.sample(
        n=config.BERT_TRAIN_SUBSAMPLE_SIZE, random_state=config.RANDOM_SEED
    )
    val_df = cv_pool.drop(train_df.index).sample(
        n=config.BERT_VAL_SUBSAMPLE_SIZE, random_state=config.RANDOM_SEED
    )
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    print(
        f"BERT fine-tuning: {len(train_df)}-row train / {len(val_df)}-row val "
        f"subsample of the {len(cv_pool)}-row CV pool (see src.config.BERT_TRAIN_SUBSAMPLE_SIZE)."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_bert_and_tokenizer(config.BERT_MODEL_NAME)
    model.to(device)

    train_loader, val_loader = build_dataloaders(
        train_df, val_df, tokenizer, config.BERT_BATCH_SIZE
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_accuracy = -1.0
    best_state = None
    val_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            outputs.loss.backward()
            optimizer.step()

        val_accuracy = _evaluate(model, val_loader, device)
        val_accuracy_history.append(val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(f"[bert] epoch {epoch + 1}/{num_epochs}  val_acc={val_accuracy:.4f}")

    model.load_state_dict(best_state)
    save_dir = config.MODELS_SAVED_DIR / "bert_finetuned"
    ensure_dir(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return {"best_val_accuracy": best_val_accuracy, "history": val_accuracy_history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.BERT_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    result = fine_tune(num_epochs=args.epochs, learning_rate=args.lr)
    print(
        f"\nbest_val_acc={result['best_val_accuracy']:.4f} "
        f"-> saved to models_saved/bert_finetuned/"
    )
    return result


if __name__ == "__main__":
    main()
