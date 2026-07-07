"""Validated database settings with secret-safe representations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

CURRENT_SCHEMA_REVISION = "20260614_0006"

_SECRET_QUERY_KEYS = {
    "api_key",
    "apikey",
    "pass",
    "password",
    "secret",
    "token",
}
_KEYWORD_SECRET_RE = re.compile(
    r"(?i)\b(password|pass|secret|token|api_key)\s*=\s*"
    r"(?:'[^']*'|\"[^\"]*\"|[^\s]+)"
)


class DatabaseConfigurationError(ValueError):
    """Raised when database environment settings are absent or invalid."""


def redact_dsn(dsn: str) -> str:
    """Return a DSN safe for diagnostics without exposing credentials."""

    value = str(dsn or "")
    if "://" not in value:
        return _KEYWORD_SECRET_RE.sub(lambda match: f"{match.group(1)}=***", value)

    try:
        parts = urlsplit(value)
    except ValueError:
        return "<redacted-dsn>"

    netloc = parts.netloc
    if "@" in netloc:
        userinfo, separator, hostinfo = netloc.rpartition("@")
        if ":" in userinfo:
            username, _, _password = userinfo.partition(":")
            userinfo = f"{username}:***"
        netloc = f"{userinfo}{separator}{hostinfo}"

    query_items = []
    for key, item_value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key = key.lower()
        is_secret = normalized_key in _SECRET_QUERY_KEYS or any(
            marker in normalized_key
            for marker in ("password", "secret", "token", "api_key", "apikey")
        )
        redacted_value = "***" if is_secret else item_value
        query_items.append((key, redacted_value))
    query = urlencode(query_items, doseq=True)
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _first_value(environ: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _read_int(
    environ: Mapping[str, str],
    names: tuple[str, ...],
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = _first_value(environ, names)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DatabaseConfigurationError(f"{names[0]} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise DatabaseConfigurationError(
            f"{names[0]} must be between {minimum} and {maximum}"
        )
    return value


def _read_float(
    environ: Mapping[str, str],
    names: tuple[str, ...],
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = _first_value(environ, names)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise DatabaseConfigurationError(f"{names[0]} must be numeric") from exc
    if not minimum <= value <= maximum:
        raise DatabaseConfigurationError(
            f"{names[0]} must be between {minimum:g} and {maximum:g}"
        )
    return value


def _validate_dsn(dsn: str) -> None:
    if "://" not in dsn:
        if "=" not in dsn:
            raise DatabaseConfigurationError(
                "database DSN must be a PostgreSQL URI or libpq conninfo string"
            )
        return
    try:
        scheme = urlsplit(dsn).scheme.lower()
    except ValueError as exc:
        raise DatabaseConfigurationError("database DSN is malformed") from exc
    if scheme not in {"postgres", "postgresql"}:
        raise DatabaseConfigurationError("database DSN must use PostgreSQL")


@dataclass(frozen=True, repr=False)
class DatabaseSettings:
    """Connection-pool and session timeout settings.

    The DSN field is excluded from dataclass-generated representations. The
    custom representation includes only a redacted form.
    """

    dsn: str = field(repr=False)
    pool_min_size: int = 1
    pool_max_size: int = 10
    pool_timeout_seconds: float = 5.0
    pool_max_waiting: int = 20
    connect_timeout_seconds: int = 5
    statement_timeout_ms: int = 15_000
    lock_timeout_ms: int = 5_000
    idle_transaction_timeout_ms: int = 30_000
    application_name: str = "eva-ai"
    expected_schema_revision: str = CURRENT_SCHEMA_REVISION

    def __post_init__(self) -> None:
        _validate_dsn(self.dsn)
        if not 0 <= self.pool_min_size <= self.pool_max_size:
            raise DatabaseConfigurationError(
                "pool_min_size must be non-negative and no greater than pool_max_size"
            )
        if not 1 <= self.pool_max_size <= 100:
            raise DatabaseConfigurationError("pool_max_size must be between 1 and 100")
        if self.pool_max_waiting < 0:
            raise DatabaseConfigurationError("pool_max_waiting must be non-negative")
        if self.pool_timeout_seconds <= 0:
            raise DatabaseConfigurationError("pool_timeout_seconds must be positive")
        if not 1 <= self.connect_timeout_seconds <= 300:
            raise DatabaseConfigurationError(
                "connect_timeout_seconds must be between 1 and 300"
            )
        for field_name, value in (
            ("statement_timeout_ms", self.statement_timeout_ms),
            ("lock_timeout_ms", self.lock_timeout_ms),
            ("idle_transaction_timeout_ms", self.idle_transaction_timeout_ms),
        ):
            if not 100 <= value <= 3_600_000:
                raise DatabaseConfigurationError(
                    f"{field_name} must be between 100 and 3600000"
                )
        if not self.application_name or len(self.application_name) > 63:
            raise DatabaseConfigurationError(
                "application_name must contain between 1 and 63 characters"
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.expected_schema_revision):
            raise DatabaseConfigurationError("expected_schema_revision is invalid")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "DatabaseSettings":
        source = os.environ if environ is None else environ
        dsn = _first_value(
            source,
            (
                "EVA_DATABASE_DSN",
                "EVA_DATABASE_URL",
                "EVOSSEARCH_DATABASE_DSN",
                "EVOSSEARCH_DATABASE_URL",
                "DATABASE_URL",
            ),
        )
        if dsn is None:
            raise DatabaseConfigurationError(
                "EVA_DATABASE_DSN is required when PostgreSQL is enabled"
            )

        pool_min_size = _read_int(
            source,
            ("EVA_DB_POOL_MIN_SIZE", "EVOSSEARCH_DB_POOL_MIN_SIZE"),
            1,
            minimum=0,
            maximum=100,
        )
        pool_max_size = _read_int(
            source,
            ("EVA_DB_POOL_MAX_SIZE", "EVOSSEARCH_DB_POOL_MAX_SIZE"),
            10,
            minimum=1,
            maximum=100,
        )
        settings = cls(
            dsn=dsn,
            pool_min_size=pool_min_size,
            pool_max_size=pool_max_size,
            pool_timeout_seconds=_read_float(
                source,
                ("EVA_DB_POOL_TIMEOUT_SECONDS",),
                5.0,
                minimum=0.1,
                maximum=300.0,
            ),
            pool_max_waiting=_read_int(
                source,
                ("EVA_DB_POOL_MAX_WAITING",),
                20,
                minimum=0,
                maximum=10_000,
            ),
            connect_timeout_seconds=_read_int(
                source,
                ("EVA_DB_CONNECT_TIMEOUT_SECONDS",),
                5,
                minimum=1,
                maximum=300,
            ),
            statement_timeout_ms=_read_int(
                source,
                ("EVA_DB_STATEMENT_TIMEOUT_MS",),
                15_000,
                minimum=100,
                maximum=3_600_000,
            ),
            lock_timeout_ms=_read_int(
                source,
                ("EVA_DB_LOCK_TIMEOUT_MS",),
                5_000,
                minimum=100,
                maximum=3_600_000,
            ),
            idle_transaction_timeout_ms=_read_int(
                source,
                ("EVA_DB_IDLE_TRANSACTION_TIMEOUT_MS",),
                30_000,
                minimum=100,
                maximum=3_600_000,
            ),
            application_name=_first_value(
                source, ("EVA_DB_APPLICATION_NAME",)
            )
            or "eva-ai",
            expected_schema_revision=_first_value(
                source, ("EVA_DB_EXPECTED_SCHEMA_REVISION",)
            )
            or CURRENT_SCHEMA_REVISION,
        )
        return settings

    @property
    def min_pool_size(self) -> int:
        return self.pool_min_size

    @property
    def max_pool_size(self) -> int:
        return self.pool_max_size

    def __repr__(self) -> str:
        return (
            "DatabaseSettings("
            f"dsn={redact_dsn(self.dsn)!r}, "
            f"pool_min_size={self.pool_min_size!r}, "
            f"pool_max_size={self.pool_max_size!r}, "
            f"pool_timeout_seconds={self.pool_timeout_seconds!r}, "
            f"pool_max_waiting={self.pool_max_waiting!r}, "
            f"connect_timeout_seconds={self.connect_timeout_seconds!r}, "
            f"statement_timeout_ms={self.statement_timeout_ms!r}, "
            f"lock_timeout_ms={self.lock_timeout_ms!r}, "
            f"idle_transaction_timeout_ms={self.idle_transaction_timeout_ms!r}, "
            f"application_name={self.application_name!r}, "
            f"expected_schema_revision={self.expected_schema_revision!r}"
            ")"
        )
