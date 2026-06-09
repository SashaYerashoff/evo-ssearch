from __future__ import annotations

import hashlib
import hmac
import secrets


MIN_TOKEN_BYTES = 32


def _generate_token(num_bytes: int) -> str:
    if num_bytes < MIN_TOKEN_BYTES:
        raise ValueError(f"tokens require at least {MIN_TOKEN_BYTES} random bytes")
    return secrets.token_urlsafe(num_bytes)


def generate_session_token(num_bytes: int = MIN_TOKEN_BYTES) -> str:
    return _generate_token(num_bytes)


def digest_session_token(token: str) -> str:
    if not token:
        raise ValueError("session token is required")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


session_token_digest_for_storage = digest_session_token


def verify_session_token_digest(token: str, stored_digest: str) -> bool:
    if not token or not stored_digest:
        return False
    candidate = digest_session_token(token)
    return hmac.compare_digest(candidate, stored_digest)


def generate_csrf_token(num_bytes: int = MIN_TOKEN_BYTES) -> str:
    return _generate_token(num_bytes)


def verify_csrf_token(expected_token: str, provided_token: str) -> bool:
    if not expected_token or not provided_token:
        return False
    return hmac.compare_digest(expected_token, provided_token)
