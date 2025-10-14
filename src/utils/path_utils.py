import os
from typing import Any


def normalize_path(value: Any, default: str, name: str) -> str:
    """Normalize a config path value to a string path.

    Accepts str, os.PathLike, or dicts with keys like 'path'|'dir'|'directory'.
    Raises TypeError if value cannot be normalized.
    """
    if value is None:
        return default
    if isinstance(value, (str, os.PathLike)):
        return str(value)
    if isinstance(value, dict):
        for key in ("path", "dir", "directory"):
            v = value.get(key)
            if isinstance(v, (str, os.PathLike)):
                return str(v)
    raise TypeError(f"Config '{name}' must be a str path; got {type(value).__name__}: {value}")


def ensure_dir(path: str) -> str:
    """Ensure a directory exists and return the path string."""
    os.makedirs(path, exist_ok=True)
    return path
