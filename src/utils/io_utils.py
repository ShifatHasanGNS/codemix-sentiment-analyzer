"""
Small I/O helpers shared across the project (load/save JSON, CSV, pickle,
and torch checkpoints), so scripts don't repeat boilerplate.

TODO:
- load_json(path) / save_json(obj, path)
- load_csv(path) -> pandas.DataFrame / save_csv(df, path)
- save_checkpoint(model_state_dict, path) / load_checkpoint(path, map_location=...)
- ensure_dir(path) -> creates a directory (and parents) if it doesn't exist
"""


def load_json(path):
    raise NotImplementedError


def save_json(obj, path):
    raise NotImplementedError


def load_csv(path):
    raise NotImplementedError


def save_csv(df, path):
    raise NotImplementedError


def save_checkpoint(state_dict, path):
    raise NotImplementedError


def load_checkpoint(path, map_location=None):
    raise NotImplementedError


def ensure_dir(path):
    raise NotImplementedError
