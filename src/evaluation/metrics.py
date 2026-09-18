"""
Computes classification metrics for every trained model, broken down overall
and per language condition (English / Bangla / Banglish / code-switched),
so the project's central research question -- "which technique is most
robust to code-switched input?" -- can be answered directly from the output.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src import config
from src.utils.io_utils import ensure_dir, save_csv


def compute_metrics(y_true: list, y_pred: list) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=config.LABELS, average="macro", zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model_by_language(model_name: str, predictions_df) -> pd.DataFrame:
    rows = []
    for language, group in predictions_df.groupby("language"):
        metrics = compute_metrics(group["true_label"], group["predicted_label"])
        rows.append({"model": model_name, "language": language, "n": len(group), **metrics})

    overall_metrics = compute_metrics(predictions_df["true_label"], predictions_df["predicted_label"])
    rows.append({"model": model_name, "language": "overall", "n": len(predictions_df), **overall_metrics})

    return pd.DataFrame(rows, columns=["model", "language", "n", "accuracy", "precision", "recall", "f1"])


def build_comparison_table(all_model_results: dict) -> pd.DataFrame:
    table = pd.concat(all_model_results.values(), ignore_index=True)
    ensure_dir(config.RESULTS_DIR)
    save_csv(table, config.RESULTS_DIR / "metrics_comparison.csv")
    return table


def summarize_cv_results(fold_results: dict) -> pd.DataFrame:
    """Build a per-model k-fold CV summary (mean +/- std validation
    accuracy) from the `cv_result` dicts returned by
    src.training.train_classical/train_neural's run_cross_validation(),
    and save it to results/cv_summary.csv.

    This reports the *validation* methodology (k-fold CV on the training
    pool); it's a separate artifact from metrics_comparison.csv, which
    reports the final refit models' performance on the untouched test set.
    """
    rows = [
        {
            "model": cv_result["model_name"],
            "n_folds": len(cv_result["fold_accuracies"]),
            "mean_accuracy": cv_result["mean_accuracy"],
            "std_accuracy": cv_result["std_accuracy"],
        }
        for cv_result in fold_results.values()
    ]
    table = pd.DataFrame(rows, columns=["model", "n_folds", "mean_accuracy", "std_accuracy"])
    table = table.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)

    ensure_dir(config.RESULTS_DIR)
    save_csv(table, config.RESULTS_DIR / "cv_summary.csv")
    return table


def plot_comparison_chart(comparison_table):
    """Grouped bar chart: accuracy per model, grouped by language condition
    -- this is the chart that actually answers the project's central
    question ("which technique is most robust to code-switched input?"),
    rather than just an overall-accuracy ranking.

    Colors are the first 4 slots of the dataviz skill's validated
    categorical palette (fixed order, not cycled); two of those slots sit
    below 3:1 contrast against the chart surface, so per the skill's
    "relief rule" this chart ships alongside metrics_comparison.csv (the
    table view) rather than leaning on in-chart text contrast alone.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    languages = [l for l in config.LANGUAGE_CONDITIONS if l in comparison_table["language"].unique()]
    # Fixed-order categorical palette (dataviz skill reference palette,
    # slots 1-4: blue, orange, aqua, yellow), validated via
    # scripts/validate_palette.js for adjacent-pair CVD safety.
    palette = {
        "english": "#2a78d6", "bangla": "#eb6834",
        "banglish": "#1baf7a", "code_switched": "#eda100",
    }

    models = (
        comparison_table[comparison_table["language"] == "overall"]
        .sort_values("accuracy", ascending=False)["model"].tolist()
    )
    pivot = comparison_table[comparison_table["language"] != "overall"].pivot(
        index="model", columns="language", values="accuracy"
    ).loc[models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    bar_width = 0.8 / len(languages)

    for i, language in enumerate(languages):
        offset = (i - (len(languages) - 1) / 2) * bar_width
        ax.bar(x + offset, pivot[language], width=bar_width * 0.9,
               label=language, color=palette[language], edgecolor="none")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", color="#3a3a3a")
    ax.set_title("Model comparison: test accuracy by language condition", color="#1a1a1a")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", color="#3a3a3a")
    ax.tick_params(axis="y", colors="#3a3a3a")
    ax.yaxis.grid(True, linewidth=1, color="#e0e0e0")
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.legend(title="Language condition", frameon=False, loc="upper right")
    fig.tight_layout()

    ensure_dir(config.RESULTS_DIR)
    fig.savefig(config.RESULTS_DIR / "metrics_comparison.png", dpi=150)
    return fig
