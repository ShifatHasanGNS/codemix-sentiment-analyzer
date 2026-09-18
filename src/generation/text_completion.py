"""
Small bonus sequence-generation demo: autoregressive next-word / next-sentence
completion built on top of a trained LSTM or Transformer checkpoint (reusing
its encoder, with a language-modeling head instead of / alongside the
classification head).

This is evaluated qualitatively only (see src/evaluation/error_analysis.py
docstring) -- not scored numerically.

TODO:
- class TextCompletionModel
    - __init__(self, base_model_checkpoint_path: str, tokenizer)
        Loads a trained LSTM/Transformer checkpoint and adapts/attaches a
        simple LM head (predicting the next token) if one isn't already
        present from training.
    - generate(self, prompt: str, max_new_tokens: int = 20,
               temperature: float = 1.0) -> str
        Greedy or temperature-sampled autoregressive generation starting
        from the prompt, returned as a decoded string.
"""


class TextCompletionModel:
    def __init__(self, base_model_checkpoint_path: str, tokenizer):
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int = 20,
                 temperature: float = 1.0) -> str:
        raise NotImplementedError
