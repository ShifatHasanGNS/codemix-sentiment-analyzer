#!/usr/bin/env python
"""
CLI entry point: downloads BanglishRev, builds, and splits the dataset --
fully automated, no manual data entry.

Usage:
    python scripts/build_dataset.py

Note: the first run needs internet access to fetch the BanglishRev dataset
from Hugging Face (src.config.HF_DATASET_ID) -- a multi-GB download; subsequent
runs can reuse the local cache (e.g. via `datasets`' default cache dir, or a
project-local cache path passed to download_banglishrev()).

This is a long-running command (see CLAUDE.md's "long-running commands" ground
rule): implement this script, then have the human run it in their own terminal
and report back once it's done, rather than running it inline in an AI session.

TODO:
- Parse optional CLI args via argparse, e.g. --seed, --target-size
  (overriding src.config.TARGET_DATASET_SIZE), --cache-dir.
- Call src.data.dataset_builder.assemble_dataset() (which internally runs
  download -> flatten -> label -> language-tag -> filter -> clean ->
  subsample, writing data/raw/dataset.csv) then
  src.data.dataset_builder.split_dataset(df), writing data/processed/.
- Print a short summary: total rows, and counts per split x label x
  language, so class/language balance can be sanity-checked at a glance.
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
