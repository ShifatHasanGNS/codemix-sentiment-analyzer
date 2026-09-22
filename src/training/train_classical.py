"""Trains classical N-Gram/BoW/TF-IDF + Logistic Regression baselines via k-fold CV, then refits on the full pool."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src import config
from src.data.dataset_builder import get_fold
from src.data.preprocess import load_split
from src.features.classical_features import (
    fit_bow_vectorizer,
    fit_ngram_vectorizer,
    fit_tfidf_vectorizer,
    transform,
)
from src.utils.io_utils import save_checkpoint
from src.utils.seed import set_seed

_FIT_FUNCTIONS = {
    "ngram": fit_ngram_vectorizer,
    "bow": fit_bow_vectorizer,
    "tfidf": fit_tfidf_vectorizer,
}


def load_data():
    return load_split("cv_pool"), load_split("test")


def _make_classifier():
    return LogisticRegression(max_iter=200, random_state=config.RANDOM_SEED)


def train_one_baseline(feature_name, train_texts, train_labels, val_texts, val_labels):
    """Same classifier for all feature types so the comparison isolates the feature representation."""
    vectorizer = _FIT_FUNCTIONS[feature_name](train_texts)
    train_features = transform(vectorizer, train_texts)
    val_features = transform(vectorizer, val_texts)

    classifier = _make_classifier()
    classifier.fit(train_features, train_labels)

    val_accuracy = accuracy_score(val_labels, classifier.predict(val_features))
    return vectorizer, classifier, val_accuracy


def run_cross_validation(feature_name: str, cv_pool):
    """k-fold CV for one feature type; returns per-fold accuracies plus mean/std."""
    fold_accuracies = []
    for fold_index in range(config.N_FOLDS):
        train_df, val_df = get_fold(cv_pool, fold_index)
        _, _, val_accuracy = train_one_baseline(
            feature_name,
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            val_df["text"].tolist(),
            val_df["label"].tolist(),
        )
        fold_accuracies.append(val_accuracy)
        print(
            f"[{feature_name}] fold {fold_index + 1}/{config.N_FOLDS}  val_acc={val_accuracy:.4f}"
        )

    return {
        "model_name": feature_name,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
    }


def fit_final_classical(feature_name: str, texts: list, labels: list):
    """Fit on the full CV pool (no held-out val) for the deployed model."""
    vectorizer = _FIT_FUNCTIONS[feature_name](texts)
    classifier = _make_classifier()
    classifier.fit(transform(vectorizer, texts), labels)
    return vectorizer, classifier


def main():
    set_seed(config.RANDOM_SEED)
    cv_pool, _ = load_data()

    cv_results = []
    for feature_name in _FIT_FUNCTIONS:
        cv_result = run_cross_validation(feature_name, cv_pool)
        cv_results.append(cv_result)
        print(
            f"-> {feature_name}: CV mean_acc={cv_result['mean_accuracy']:.4f} "
            f"(+/- {cv_result['std_accuracy']:.4f}) over {config.N_FOLDS} folds"
        )

        vectorizer, classifier = fit_final_classical(
            feature_name, cv_pool["text"].tolist(), cv_pool["label"].tolist()
        )
        save_path = config.MODELS_SAVED_DIR / f"classical_{feature_name}.pt"
        save_checkpoint(
            {
                "feature_name": feature_name,
                "vectorizer": vectorizer,
                "classifier": classifier,
            },
            save_path,
        )
        print(f"-> {feature_name}: refit on full CV pool -> {save_path}\n")

    return cv_results


if __name__ == "__main__":
    main()
