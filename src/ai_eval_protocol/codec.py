from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_document(path: Path) -> dict[str, Any]:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_non_finite_number,
    )
    if not isinstance(raw, dict):
        raise ValueError("protocol document must be a JSON object")
    return raw


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def canonical_json(document: dict[str, Any]) -> str:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(document: dict[str, Any]) -> str:
    payload = canonical_json(document).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
