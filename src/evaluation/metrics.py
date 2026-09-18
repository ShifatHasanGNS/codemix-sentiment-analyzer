"""
Computes classification metrics for every trained model, broken down overall
and per language condition (English / Bangla / Banglish / code-switched),
so the project's central research question -- "which technique is most
robust to code-switched input?" -- can be answered directly from the output.

TODO:
- compute_metrics(y_true: list, y_pred: list) -> dict
    accuracy, precision, recall, f1 (macro-averaged across the 3 classes).
- evaluate_model_by_language(model_name: str, predictions_df) -> pandas.DataFrame
    predictions_df expected to have columns: text, language, true_label,
    predicted_label. Groups by `language` (plus an "overall" row) and applies
    compute_metrics to each group.
- build_comparison_table(all_model_results: dict[str, pandas.DataFrame]) -> pandas.DataFrame
    Combines every model's per-language metrics into one wide comparison
    table, written to results/metrics_comparison.csv.
- plot_comparison_chart(comparison_table) -> matplotlib.figure.Figure
    Bar chart of (e.g.) accuracy per model per language condition, saved to
    results/metrics_comparison.png.
"""


def compute_metrics(y_true: list, y_pred: list) -> dict:
    raise NotImplementedError


def evaluate_model_by_language(model_name: str, predictions_df):
    raise NotImplementedError


def build_comparison_table(all_model_results: dict):
    raise NotImplementedError


def plot_comparison_chart(comparison_table):
    raise NotImplementedError
