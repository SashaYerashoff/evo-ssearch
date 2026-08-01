"""Stable identity helpers for CLIP-like embedding spaces.

Vectors from different models (or different revisions of the same model) are
not comparable even when their dimensions happen to match.  Keep the identity
small enough to persist beside every archive vector and derive a filesystem /
FAISS-safe fingerprint from the canonical fields.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional


_IDENTITY_FIELDS = (
    "backend",
    "model",
    "revision",
    "dimension",
    "contract",
)


def normalize_embedding_space(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    normalized: Dict[str, Any] = {}
    for key in ("backend", "model", "revision", "contract"):
        text = str(value.get(key) or "").strip()
        if text:
            normalized[key] = text
    raw_dimension = value.get("dimension")
    if raw_dimension is not None and not isinstance(raw_dimension, bool):
        try:
            dimension = int(raw_dimension)
        except (TypeError, ValueError):
            dimension = 0
        if dimension > 0:
            normalized["dimension"] = dimension
    return normalized


def embedding_space_fingerprint(value: Any) -> str:
    normalized = normalize_embedding_space(value)
    canonical = {
        key: normalized[key]
        for key in _IDENTITY_FIELDS
        if key in normalized
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def identified_embedding_space(value: Any) -> Dict[str, Any]:
    normalized = normalize_embedding_space(value)
    if normalized:
        normalized["fingerprint"] = embedding_space_fingerprint(normalized)
    return normalized


def embedding_space_requires_identity(value: Any) -> bool:
    normalized = normalize_embedding_space(value)
    backend = str(normalized.get("backend") or "").lower()
    model = str(normalized.get("model") or "").lower()
    return backend not in {"", "openai_clip"} or "siglip" in model


def embedding_spaces_match(
    expected: Any,
    archived: Any,
    *,
    allow_legacy_openai_clip: bool = True,
) -> bool:
    expected_space = normalize_embedding_space(expected)
    archived_space = normalize_embedding_space(archived)
    if not expected_space:
        return True
    if not archived_space:
        return bool(
            allow_legacy_openai_clip
            and not embedding_space_requires_identity(expected_space)
        )
    for key in _IDENTITY_FIELDS:
        expected_value: Optional[Any] = expected_space.get(key)
        archived_value: Optional[Any] = archived_space.get(key)
        if expected_value is None:
            continue
        if archived_value is None:
            if embedding_space_requires_identity(expected_space):
                return False
            continue
        if expected_value != archived_value:
            return False
    if embedding_space_requires_identity(expected_space):
        return bool(archived_space.get("backend") and archived_space.get("model"))
    return True
