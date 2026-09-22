# Pretrained mBERT wrapper -- the only model here using pretrained weights.

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config


def load_bert_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(config.LABELS)
    )
    return tokenizer, model


def encode_batch(tokenizer, texts: list, max_length: int):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
