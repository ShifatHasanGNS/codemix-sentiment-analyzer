"""Qualitative error analysis: misclassified examples per model with a likely-cause note, plus generation demo samples."""

from pathlib import Path

import pandas as pd

from src import config
from src.utils.io_utils import ensure_dir


def collect_misclassified_examples(
    model_name: str, predictions_df, num_examples: int = 5
):
    wrong = predictions_df[
        predictions_df["true_label"] != predictions_df["predicted_label"]
    ].copy()
    wrong["model"] = model_name
    if len(wrong) == 0:
        return wrong.assign(note=[])

    # Spread the sample across language conditions rather than taking the first N rows.
    per_language_quota = max(1, num_examples // max(len(wrong["language"].unique()), 1))
    sampled = wrong.groupby("language", group_keys=False).apply(
        lambda g: g.sample(
            n=min(per_language_quota, len(g)), random_state=config.RANDOM_SEED
        )
    )
    if len(sampled) < num_examples:
        remaining = wrong.drop(sampled.index)
        extra = remaining.sample(
            n=min(num_examples - len(sampled), len(remaining)),
            random_state=config.RANDOM_SEED,
        )
        sampled = pd.concat([sampled, extra])
    sampled = sampled.head(num_examples).copy()

    sampled["note"] = sampled.apply(_likely_cause_note, axis=1)
    return sampled


def _likely_cause_note(row) -> str:
    if row["language"] == "code_switched":
        return "Code-switched input -- likely an unseen intra-sentence language-mixing pattern."
    if row["language"] == "banglish":
        return "Romanized Bangla -- likely out-of-vocabulary tokens (no dictionary/script cues to lean on)."
    if row["predicted_label"] == "positive":
        return "Predicted the majority class ('positive') -- possible class-imbalance bias."
    return "Misclassified with no obvious script/vocabulary cause -- possibly ambiguous or sarcastic phrasing."


def write_error_report(all_examples: dict, output_path: str) -> None:
    lines = ["# Error Analysis Report", ""]

    lines.append("## Misclassified Examples\n")
    for model_name, examples_df in all_examples.get("misclassified", {}).items():
        lines.append(f"### {model_name}\n")
        if len(examples_df) == 0:
            lines.append("_No misclassified examples in the test set._\n")
            continue
        for _, row in examples_df.iterrows():
            lines.append(
                f'- **[{row["language"]}]** "{row["text"]}"\n'
                f"  true=`{row['true_label']}`, predicted=`{row['predicted_label']}` -- {row['note']}"
            )
        lines.append("")

    generation_samples = all_examples.get("generation_samples")
    if generation_samples:
        lines.append("## Text-Completion Bonus (Phase 9, qualitative only)\n")
        for sample in generation_samples:
            lines.append(f'- **[{sample["model_name"]}]** prompt="{sample["prompt"]}"')
            lines.append(f'  -> "{sample["completion"]}"')
        lines.append("")

    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def collect_generation_samples(prompts: list):
    """Runs TextCompletionModel.generate on fixed prompts using the LSTM and Transformer checkpoints."""
    from src.data.tokenizer import CodeMixTokenizer
    from src.generation.text_completion import TextCompletionModel

    tokenizer_path = config.MODELS_SAVED_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        return []
    tokenizer = CodeMixTokenizer.load(tokenizer_path)

    samples = []
    for model_name in ("lstm", "transformer"):
        checkpoint_path = config.MODELS_SAVED_DIR / f"{model_name}.pt"
        if not checkpoint_path.exists():
            continue
        completion_model = TextCompletionModel(checkpoint_path, tokenizer)
        for prompt in prompts:
            completion = completion_model.generate(
                prompt, max_new_tokens=15, temperature=0.8
            )
            samples.append(
                {"model_name": model_name, "prompt": prompt, "completion": completion}
            )

    return samples
