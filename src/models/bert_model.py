"""
Pretrained multilingual BERT wrapper (mBERT or XLM-R, via HuggingFace
transformers), fine-tuned on the project's classification task and used as
an upper-bound benchmark against the five from-scratch models.

This is the ONLY model in the project that uses pretrained weights and its
own native (sub-word) tokenizer, per the project's scope constraints.

TODO:
- load_bert_and_tokenizer(model_name: str) -> (tokenizer, model)
    model_name should come from src.config.BERT_MODEL_NAME, loaded via
    transformers.AutoTokenizer / AutoModelForSequenceClassification with
    num_labels=3.
- encode_batch(tokenizer, texts: list[str], max_length: int) -> dict of tensors
    (input_ids, attention_mask, ...) ready for the model's forward pass.
"""


def load_bert_and_tokenizer(model_name: str):
    raise NotImplementedError


def encode_batch(tokenizer, texts: list, max_length: int):
    raise NotImplementedError
