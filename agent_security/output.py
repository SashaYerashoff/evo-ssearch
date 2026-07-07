from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .policy import ToolPolicy


_DEFAULT_SENSITIVE_FIELDS = frozenset(
    {
        "access_token",
        "approval_id",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "token",
    }
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REDACTED = "[REDACTED]"


def sanitize_output(value: Any, policy: ToolPolicy) -> Any:
    sensitive_fields = {
        _normalized_key(key)
        for key in _DEFAULT_SENSITIVE_FIELDS | policy.redact_fields
    }
    state = {"items": 0, "truncated": False}
    sanitized = _sanitize(
        value,
        sensitive_fields=sensitive_fields,
        max_items=policy.max_output_items,
        max_string_chars=policy.max_output_string_chars,
        state=state,
    )
    if state["truncated"] and isinstance(sanitized, dict):
        sanitized.setdefault("_truncated", True)
    return _bound_serialized(sanitized, policy.max_output_bytes)


def _sanitize(
    value: Any,
    *,
    sensitive_fields: set[str],
    max_items: int,
    max_string_chars: int,
    state: dict[str, Any],
) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if state["items"] >= max_items:
                state["truncated"] = True
                break
            state["items"] += 1
            text_key = str(key)
            if _is_sensitive(text_key, sensitive_fields):
                result[text_key] = _REDACTED
            else:
                result[text_key] = _sanitize(
                    item,
                    sensitive_fields=sensitive_fields,
                    max_items=max_items,
                    max_string_chars=max_string_chars,
                    state=state,
                )
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        result = []
        for item in value:
            if state["items"] >= max_items:
                state["truncated"] = True
                break
            state["items"] += 1
            result.append(
                _sanitize(
                    item,
                    sensitive_fields=sensitive_fields,
                    max_items=max_items,
                    max_string_chars=max_string_chars,
                    state=state,
                )
            )
        return result
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        cleaned = _CONTROL_CHARACTERS.sub("", value)
        if len(cleaned) > max_string_chars:
            state["truncated"] = True
            return cleaned[:max_string_chars]
        return cleaned
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _is_sensitive(key: str, sensitive_fields: set[str]) -> bool:
    normalized = _normalized_key(key)
    if normalized in sensitive_fields:
        return True
    return any(
        marker in normalized
        for marker in ("password", "secret", "token", "api_key", "private_key")
    )


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _bound_serialized(value: Any, max_bytes: int) -> Any:
    encoded = _json_bytes(value)
    if len(encoded) <= max_bytes:
        return value

    envelope = {"_truncated": True, "preview": ""}
    overhead = len(_json_bytes(envelope))
    if max_bytes <= overhead:
        return 0

    available = max_bytes - overhead
    preview = encoded[:available].decode("utf-8", errors="ignore")
    envelope["preview"] = preview
    while len(_json_bytes(envelope)) > max_bytes and envelope["preview"]:
        envelope["preview"] = envelope["preview"][:-1]
    return envelope


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
