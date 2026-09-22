"""Small I/O helpers shared across the project (JSON, CSV, torch checkpoints)."""

import json
import os

import pandas as pd
import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    ensure_dir(os.path.dirname(str(path)) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path):
    return pd.read_csv(path)


def save_csv(df, path):
    ensure_dir(os.path.dirname(str(path)) or ".")
    df.to_csv(path, index=False)


def save_checkpoint(state_dict, path):
    ensure_dir(os.path.dirname(str(path)) or ".")
    torch.save(state_dict, path)


def load_checkpoint(path, map_location=None):
    # weights_only=False: checkpoints may hold non-tensor objects (e.g. sklearn classifiers); always first-party.
    return torch.load(path, map_location=map_location, weights_only=False)
