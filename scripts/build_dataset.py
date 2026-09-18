#!/usr/bin/env python
"""
CLI entry point: downloads BanglishRev, builds, and splits the dataset --
fully automated, no manual data entry.

Usage:
    python scripts/build_dataset.py [--seed SEED] [--target-size N] [--cache-dir DIR]

Note: the first run needs internet access to fetch BanglishRev's review
JSON (~1.9GB) from Hugging Face; subsequent runs reuse the local
huggingface_hub cache. This is a long-running command (see CLAUDE.md's
"long-running commands" ground rule): run it in your own terminal. At the
current TARGET_DATASET_SIZE (150,000) it must flatten/tag the full ~1.74M
raw reviews (not a small sample), so expect a few minutes even with the
raw file cached.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.dataset_builder import assemble_dataset, create_cv_splits
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--target-size", type=int, default=config.TARGET_DATASET_SIZE)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    config.RANDOM_SEED = args.seed
    config.TARGET_DATASET_SIZE = args.target_size

    df = assemble_dataset(cache_dir=args.cache_dir)
    cv_pool, test_df = create_cv_splits(df)

    print(f"\nAssembled {len(df)} total reviews -> data/raw/dataset.csv")
    print(f"CV pool: {len(cv_pool)} rows ({config.N_FOLDS} folds) | "
          f"Held-out test: {len(test_df)} rows\n")

    print(f"--- cv_pool ({len(cv_pool)} rows) ---")
    print(cv_pool.groupby(["label", "language"]).size().unstack(fill_value=0))
    print(f"\nfold sizes: {cv_pool['fold'].value_counts().sort_index().to_dict()}\n")

    print(f"--- test ({len(test_df)} rows) ---")
    print(test_df.groupby(["label", "language"]).size().unstack(fill_value=0))
    print()


if __name__ == "__main__":
    main()
