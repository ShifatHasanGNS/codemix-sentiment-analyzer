"""
Fine-tunes the pretrained multilingual BERT benchmark
(src.models.bert_model) on the project's train/val split.

TODO:
- build_dataloaders(train_df, val_df, tokenizer, batch_size) -> (train_loader, val_loader)
    Uses src.models.bert_model.encode_batch for tokenization.
- fine_tune(num_epochs: int, learning_rate: float) -> dict
    Standard fine-tuning loop (small learning rate, e.g. 2e-5; optionally a
    linear warmup/decay schedule); tracks best validation accuracy; saves
    checkpoint to models_saved/bert_finetuned/.
- main()
    CLI entry point calling fine_tune() with sensible defaults.
"""


def build_dataloaders(train_df, val_df, tokenizer, batch_size: int):
    raise NotImplementedError


def fine_tune(num_epochs: int = 3, learning_rate: float = 2e-5):
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
