"""
Qualitative error analysis: collects representative misclassified examples
per model (ideally covering more than one language condition each) with a
short note on the likely cause (e.g. unseen code-switch pattern, out-of-
vocabulary tokens, class imbalance, tokenization artifact).

Also used to qualitatively showcase a handful of outputs from the bonus
sequence-generation demo (src.generation.text_completion) -- not scored
numerically, just presented as examples in the final report.

TODO:
- collect_misclassified_examples(model_name: str, predictions_df,
                                  num_examples: int = 5) -> pandas.DataFrame
    Sample a few wrong predictions per model, ideally spread across language
    conditions, for manual annotation of likely failure cause.
- write_error_report(all_examples: dict[str, pandas.DataFrame], output_path: str) -> None
    Renders a readable Markdown report (model -> examples -> notes) to
    results/error_analysis.md.
- collect_generation_samples(prompts: list[str]) -> list[dict]
    Runs src.generation.text_completion.TextCompletionModel.generate on a
    few fixed prompts (one per language condition, where sensible) and
    records the outputs for inclusion in the same report.
"""


def collect_misclassified_examples(model_name: str, predictions_df, num_examples: int = 5):
    raise NotImplementedError


def write_error_report(all_examples: dict, output_path: str) -> None:
    raise NotImplementedError


def collect_generation_samples(prompts: list):
    raise NotImplementedError
