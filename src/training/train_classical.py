"""
Trains the classical statistical baselines: N-Gram, Bag-of-Words, and TF-IDF
features, each paired with a simple classifier (e.g. Naive Bayes / Logistic
Regression from scikit-learn).

TODO:
- load_data() -> train_df, val_df   (via src.data.preprocess.load_split)
- train_one_baseline(feature_name: str, train_texts, train_labels,
                      val_texts, val_labels) -> (vectorizer, classifier, val_accuracy)
    feature_name in {"ngram", "bow", "tfidf"}, dispatching to the matching
    fit_*_vectorizer function in src.features.classical_features.
- main()
    Loops over all three feature types, trains + evaluates each, saves the
    (vectorizer, classifier) pair to models_saved/ via src.utils.io_utils,
    and prints a small summary table.
"""


def load_data():
    raise NotImplementedError


def train_one_baseline(feature_name, train_texts, train_labels, val_texts, val_labels):
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
