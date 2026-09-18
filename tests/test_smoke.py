"""
Lightweight smoke tests -- not full coverage, just enough to catch broken
imports/signatures as modules get implemented phase by phase (see CLAUDE.md).

TODO (add these as the corresponding phase gets implemented; skip/guard
tests for phases not yet done rather than leaving the whole suite failing):
- test_config_imports(): `from src import config` succeeds and expected
  constants exist (LABELS, LANGUAGE_CONDITIONS, etc.).
- test_tokenizer_roundtrip(): CodeMixTokenizer.tokenize() returns a
  non-empty list for a short sentence in each of the 4 language conditions.
- test_model_forward_shapes(): for each from-scratch model class in
  src.models, a forward pass on a small dummy batch returns logits of shape
  (batch_size, 3).
- test_metrics_basic(): compute_metrics() on a tiny hand-crafted
  y_true/y_pred pair returns the expected accuracy.
"""
