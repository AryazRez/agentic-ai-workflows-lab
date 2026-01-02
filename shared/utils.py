import json
import os
from datetime import datetime
from dataclasses import is_dataclass, asdict
from typing import Any


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


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


def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def timestamp():
    return datetime.utcnow().isoformat()
