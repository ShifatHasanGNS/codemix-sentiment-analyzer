"""
Pretrained multilingual BERT wrapper (mBERT or XLM-R, via HuggingFace
transformers), fine-tuned on the project's classification task and used as
an upper-bound benchmark against the five from-scratch models.

This is the ONLY model in the project that uses pretrained weights and its
own native (sub-word) tokenizer, per the project's scope constraints.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config


def load_bert_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(config.LABELS))
    return tokenizer, model


def encode_batch(tokenizer, texts: list, max_length: int):
    return tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
