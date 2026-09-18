"""
Reproducibility helper.

Every training script (classical, neural, BERT) should call set_seed()
first, using src.config.RANDOM_SEED as the default.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
