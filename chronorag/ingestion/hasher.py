import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(content: Union[str, bytes, Path]) -> str:
    """Computes a SHA-256 hex digest for a string, bytes, or file path."""
    hasher = hashlib.sha256()
    if isinstance(content, Path):
        with open(content, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    elif isinstance(content, str):
        hasher.update(content.encode("utf-8"))
    elif isinstance(content, bytes):
        hasher.update(content)
    else:
        raise TypeError(f"Unsupported content type for hashing: {type(content)}")
    return hasher.hexdigest()
