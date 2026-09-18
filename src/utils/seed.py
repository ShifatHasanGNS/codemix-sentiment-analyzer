"""
Reproducibility helper.

TODO:
- Implement set_seed(seed: int) -> None that seeds:
    python's `random`, `numpy`, `torch` (CPU and CUDA if available),
    and sets torch.backends.cudnn.deterministic / benchmark appropriately.
- Every training script (classical, neural, BERT) should call this first,
  using src.config.RANDOM_SEED as the default.
"""


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducible runs. See module docstring."""
    raise NotImplementedError
