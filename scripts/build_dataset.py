#!/usr/bin/env python
"""
Usage:
    python scripts/build_dataset.py [--seed SEED] [--target-size N] [--cache-dir DIR]
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
