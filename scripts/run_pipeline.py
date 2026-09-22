#!/usr/bin/env python
"""
Usage:
    python scripts/run_pipeline.py [--models ngram,bow,tfidf,ann,rnn,lstm,attention,transformer,bert]
                                    [--skip-bert] [--skip-classical] [--epochs N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src import config
from src.data.preprocess import load_split
from src.data.tokenizer import CodeMixTokenizer
from src.evaluation.error_analysis import (
    collect_generation_samples, collect_misclassified_examples, write_error_report,
)
from src.evaluation.metrics import (
    build_comparison_table, evaluate_model_by_language, plot_comparison_chart, summarize_cv_results,
)
from src.features.classical_features import transform
from src.features.embeddings import build_embedding_matrix, load_word2vec
from src.models.bert_model import encode_batch
from src.training.train_neural import _forward, build_model
from src.utils.io_utils import load_checkpoint

CLASSICAL_MODELS = ("ngram", "bow", "tfidf")
NEURAL_MODELS = ("ann", "rnn", "lstm", "attention", "transformer")
ALL_MODELS = CLASSICAL_MODELS + NEURAL_MODELS + ("bert",)

# one generation prompt per language condition, for the error-analysis report
_GENERATION_PROMPTS = ["this product is", "ei jinis ta", "পণ্যটি", "product ta khub"]

_PREDICT_BATCH_SIZE = 128


def _require_processed_data():
    missing = [name for name in ("cv_pool", "test")
               if not (config.DATA_PROCESSED_DIR / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing data/processed/{{{','.join(missing)}}}.csv -- "
            f"run `python scripts/build_dataset.py` first."
        )


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _predict_classical(model_name: str, texts: list) -> list:
    checkpoint = load_checkpoint(config.MODELS_SAVED_DIR / f"classical_{model_name}.pt")
    return list(checkpoint["classifier"].predict(transform(checkpoint["vectorizer"], texts)))


def _predict_neural(model_name: str, texts: list, tokenizer, embedding_matrix, vocab_size: int) -> list:
    checkpoint = load_checkpoint(config.MODELS_SAVED_DIR / f"{model_name}.pt")
    model = build_model(model_name, vocab_size, embedding_matrix)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    embedding_matrix_tensor = torch.as_tensor(embedding_matrix, dtype=torch.float32)

    predicted = []
    with torch.no_grad():
        for batch_texts in _batched(texts, _PREDICT_BATCH_SIZE):
            input_ids = torch.stack(
                [torch.tensor(tokenizer.encode(t, config.MAX_SEQUENCE_LENGTH)) for t in batch_texts]
            )
            lengths = torch.tensor(
                [max(1, min(len(tokenizer.tokenize(t)), config.MAX_SEQUENCE_LENGTH)) for t in batch_texts]
            )
            logits = _forward(model, model_name, input_ids, lengths, embedding_matrix_tensor)
            predicted.extend(config.LABELS[i] for i in logits.argmax(dim=-1).tolist())
    return predicted


def _predict_bert(texts: list) -> list:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = config.MODELS_SAVED_DIR / "bert_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()

    predicted = []
    with torch.no_grad():
        for batch_texts in _batched(texts, config.BERT_BATCH_SIZE):
            encoded = encode_batch(tokenizer, batch_texts, config.BERT_MAX_SEQUENCE_LENGTH)
            logits = model(**encoded).logits
            predicted.extend(config.LABELS[i] for i in logits.argmax(dim=-1).tolist())
    return predicted


def _checkpoint_exists(model_name: str) -> bool:
    if model_name in CLASSICAL_MODELS:
        return (config.MODELS_SAVED_DIR / f"classical_{model_name}.pt").exists()
    if model_name in NEURAL_MODELS:
        return (config.MODELS_SAVED_DIR / f"{model_name}.pt").exists()
    if model_name == "bert":
        return (config.MODELS_SAVED_DIR / "bert_finetuned").exists()
    return False


def run_evaluation(models_to_evaluate, cv_results_by_model: dict = None):
    """Predicts on the held-out test set and writes results/* reports."""
    test_df = load_split("test")
    texts = test_df["text"].tolist()

    available = [m for m in models_to_evaluate if _checkpoint_exists(m)]
    skipped = sorted(set(models_to_evaluate) - set(available))
    if skipped:
        print(f"Skipping evaluation for {skipped} -- no saved checkpoint found "
              f"(train them first, or drop them from --models).")
    if not available:
        print("No trained checkpoints available to evaluate.")
        return None

    tokenizer, embedding_matrix, vocab_size = None, None, None
    if any(m in NEURAL_MODELS for m in available):
        tokenizer = CodeMixTokenizer.load(config.MODELS_SAVED_DIR / "tokenizer.json")
        w2v = load_word2vec(config.MODELS_SAVED_DIR / "word2vec.model")
        embedding_matrix = build_embedding_matrix(w2v, tokenizer.token_to_id, config.EMBEDDING_DIM)
        vocab_size = tokenizer.vocab_size

    all_model_results, misclassified_by_model = {}, {}
    for model_name in available:
        if model_name in CLASSICAL_MODELS:
            predicted = _predict_classical(model_name, texts)
        elif model_name in NEURAL_MODELS:
            predicted = _predict_neural(model_name, texts, tokenizer, embedding_matrix, vocab_size)
        else:
            predicted = _predict_bert(texts)

        predictions_df = test_df[["text", "language", "label"]].copy()
        predictions_df = predictions_df.rename(columns={"label": "true_label"})
        predictions_df["predicted_label"] = predicted

        all_model_results[model_name] = evaluate_model_by_language(model_name, predictions_df)
        misclassified_by_model[model_name] = collect_misclassified_examples(model_name, predictions_df)
        print(f"Evaluated {model_name}: overall accuracy = "
              f"{all_model_results[model_name].set_index('language').loc['overall', 'accuracy']:.4f}")

    comparison_table = build_comparison_table(all_model_results)
    plot_comparison_chart(comparison_table)
    generation_samples = collect_generation_samples(_GENERATION_PROMPTS)
    write_error_report(
        {"misclassified": misclassified_by_model, "generation_samples": generation_samples},
        config.RESULTS_DIR / "error_analysis.md",
    )

    wrote = ["results/metrics_comparison.csv", "results/metrics_comparison.png", "results/error_analysis.md"]
    if cv_results_by_model:
        summarize_cv_results(cv_results_by_model)
        wrote.append("results/cv_summary.csv")
    print(f"\nWrote {', '.join(wrote)} ({len(available)} models evaluated).")
    return comparison_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS))
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    _require_processed_data()

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in selected:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model {m!r} in --models; choose from {ALL_MODELS}")
    if args.skip_bert:
        selected = [m for m in selected if m != "bert"]
    if args.skip_classical:
        selected = [m for m in selected if m not in CLASSICAL_MODELS]

    classical_selected = [m for m in selected if m in CLASSICAL_MODELS]
    neural_selected = [m for m in selected if m in NEURAL_MODELS]
    bert_selected = "bert" in selected

    cv_results_by_model = {}

    if classical_selected:
        print(f"=== Cross-validating + refitting classical baselines: {classical_selected} ===")
        from src.training.train_classical import fit_final_classical, run_cross_validation
        from src.utils.io_utils import save_checkpoint

        cv_pool = load_split("cv_pool")
        for model_name in classical_selected:
            cv_result = run_cross_validation(model_name, cv_pool)
            cv_results_by_model[model_name] = cv_result
            print(f"{model_name}: CV mean_acc={cv_result['mean_accuracy']:.4f} "
                  f"(+/- {cv_result['std_accuracy']:.4f})")

            vectorizer, classifier = fit_final_classical(
                model_name, cv_pool["text"].tolist(), cv_pool["label"].tolist()
            )
            save_checkpoint({"feature_name": model_name, "vectorizer": vectorizer, "classifier": classifier},
                             config.MODELS_SAVED_DIR / f"classical_{model_name}.pt")

    if neural_selected:
        print(f"\n=== Cross-validating + refitting neural models: {neural_selected} ===")
        from src.training.train_neural import refit_final, run_cross_validation

        if args.epochs:
            config.NUM_EPOCHS = args.epochs
        for model_name in neural_selected:
            cv_result = run_cross_validation(model_name)
            cv_results_by_model[model_name] = cv_result
            print(f"{model_name}: CV mean_acc={cv_result['mean_accuracy']:.4f} "
                  f"(+/- {cv_result['std_accuracy']:.4f})")
            refit_final(model_name)

    if bert_selected:
        print("\n=== Fine-tuning BERT (single held-out split, no CV) ===")
        from src.training.train_bert import fine_tune

        epochs = args.epochs or config.BERT_NUM_EPOCHS
        result = fine_tune(num_epochs=epochs)
        print(f"bert: val_acc={result['best_val_accuracy']:.4f}")

    print("\n=== Evaluation ===")
    run_evaluation(selected, cv_results_by_model)


if __name__ == "__main__":
    main()
