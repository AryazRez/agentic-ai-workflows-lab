from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from typing import Any


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not exist.
    Safe to call repeatedly.
    """
    os.makedirs(path, exist_ok=True)


def _to_jsonable(obj: Any) -> Any:
    """
    Convert dataclasses (and nested dataclasses) into plain Python types
    that json.dump can serialize.
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def read_json(path: str) -> Any:
    """
    Read a JSON file and return the parsed object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    """
    Write an object to a JSON file with stable formatting.
    Supports dataclasses.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, ensure_ascii=False)


def write_text(path: str, text: str) -> None:
    """
    Write plain text to a file.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def timestamp() -> str:
    """
    Return an ISO-8601 UTC timestamp (timezone-aware).
    """
    return datetime.now(timezone.utc).isoformat()

