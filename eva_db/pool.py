"""Lazy, bounded psycopg3 connection pool and readiness checks."""

from __future__ import annotations

import importlib
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

from .settings import DatabaseSettings


class DatabaseDependencyError(RuntimeError):
    """Raised when optional PostgreSQL runtime dependencies are unavailable."""


class DatabaseState(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    MIGRATION_MISMATCH = "migration_mismatch"
    UNSAFE_RUNTIME_ROLE = "unsafe_runtime_role"


@dataclass(frozen=True)
class DatabaseCheckResult:
    state: DatabaseState
    detail: str
    latency_ms: float
    current_revision: str | None = None
    expected_revision: str | None = None
    current_user: str | None = None
    session_user: str | None = None
    unsafe_reason: str | None = None

    @property
    def ready(self) -> bool:
        return self.state is DatabaseState.READY

    @property
    def available(self) -> bool:
        return self.state is not DatabaseState.UNAVAILABLE


def _validated_uuid(value: str | uuid.UUID, field_name: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _validated_context_text(value: str | None, field_name: str) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text or len(text) > 128 or "\x00" in text:
        raise ValueError(f"{field_name} must contain 1 to 128 safe characters")
    return text


@dataclass(frozen=True)
class TransactionContext:
    """Server-derived identity propagated to PostgreSQL transaction-local GUCs."""

    tenant_id: str | uuid.UUID
    actor_id: str | uuid.UUID
    request_id: str | None = None
    agent_session_id: str | uuid.UUID | None = None

    def as_database_values(self) -> tuple[str, str, str, str]:
        session_id = (
            ""
            if self.agent_session_id is None
            else _validated_uuid(self.agent_session_id, "agent_session_id")
        )
        return (
            _validated_uuid(self.tenant_id, "tenant_id"),
            _validated_uuid(self.actor_id, "actor_id"),
            _validated_context_text(self.request_id, "request_id"),
            session_id,
        )


class PsycopgPool:
    """A bounded psycopg_pool wrapper with explicit transaction boundaries."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self._pool: Any | None = None
        self._opened = False
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"pool_min_size={self.settings.pool_min_size}, "
            f"pool_max_size={self.settings.pool_max_size}, "
            f"opened={self._opened})"
        )

    @staticmethod
    def _load_pool_class() -> Any:
        try:
            importlib.import_module("psycopg")
            module = importlib.import_module("psycopg_pool")
        except ModuleNotFoundError as exc:
            if exc.name not in {"psycopg", "psycopg_pool"}:
                raise
            raise DatabaseDependencyError(
                "PostgreSQL support requires requirements-db.txt"
            ) from None
        return module.ConnectionPool

    def _configure_connection(self, connection: Any) -> None:
        settings = self.settings
        connection.execute(
            """
            SELECT
                set_config('statement_timeout', %s, false),
                set_config('lock_timeout', %s, false),
                set_config('idle_in_transaction_session_timeout', %s, false)
            """,
            (
                f"{settings.statement_timeout_ms}ms",
                f"{settings.lock_timeout_ms}ms",
                f"{settings.idle_transaction_timeout_ms}ms",
            ),
        )
        connection.commit()

    def _ensure_pool(self) -> Any:
        with self._lock:
            if self._pool is None:
                pool_class = self._load_pool_class()
                self._pool = pool_class(
                    conninfo=self.settings.dsn,
                    min_size=self.settings.pool_min_size,
                    max_size=self.settings.pool_max_size,
                    timeout=self.settings.pool_timeout_seconds,
                    max_waiting=self.settings.pool_max_waiting,
                    kwargs={
                        "application_name": self.settings.application_name,
                        "connect_timeout": self.settings.connect_timeout_seconds,
                    },
                    configure=self._configure_connection,
                    name="eva-db",
                    open=False,
                )
            return self._pool

    def open(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._opened:
                return
            pool = self._ensure_pool()
            try:
                pool.open(wait=wait, timeout=self.settings.pool_timeout_seconds)
            except Exception:
                if getattr(pool, "closed", False):
                    self._pool = None
                raise
            self._opened = True

    def close(self) -> None:
        with self._lock:
            if self._pool is None:
                return
            self._pool.close(timeout=self.settings.pool_timeout_seconds)
            self._pool = None
            self._opened = False

    @contextmanager
    def connection(self) -> Iterator[Any]:
        self.open()
        assert self._pool is not None
        with self._pool.connection(
            timeout=self.settings.pool_timeout_seconds
        ) as connection:
            yield connection

    @contextmanager
    def transaction(
        self,
        context: TransactionContext | None = None,
        *,
        readonly: bool = False,
    ) -> Iterator[Any]:
        """Open a transaction and apply trusted identity as transaction-local GUCs."""

        with self.connection() as connection:
            with connection.transaction():
                if readonly:
                    connection.execute("SET TRANSACTION READ ONLY")
                if context is not None:
                    tenant_id, actor_id, request_id, agent_session_id = (
                        context.as_database_values()
                    )
                    connection.execute(
                        """
                        SELECT
                            set_config('eva.tenant_id', %s, true),
                            set_config('eva.actor_id', %s, true),
                            set_config('eva.request_id', %s, true),
                            set_config('eva.agent_session_id', %s, true)
                        """,
                        (tenant_id, actor_id, request_id, agent_session_id),
                    )
                yield connection

    def check_health(self) -> DatabaseCheckResult:
        started = time.monotonic()
        try:
            with self.connection() as connection:
                row = connection.execute("SELECT 1").fetchone()
            if row is None or row[0] != 1:
                raise RuntimeError("database probe returned no row")
        except Exception as exc:
            return DatabaseCheckResult(
                state=DatabaseState.UNAVAILABLE,
                detail=f"database unavailable ({_safe_exception_name(exc)})",
                latency_ms=_elapsed_ms(started),
            )
        return DatabaseCheckResult(
            state=DatabaseState.READY,
            detail="database connection available",
            latency_ms=_elapsed_ms(started),
        )

    def check_readiness(self) -> DatabaseCheckResult:
        started = time.monotonic()
        expected = self.settings.expected_schema_revision
        try:
            with self.connection() as connection:
                probe = connection.execute("SELECT 1").fetchone()
                revision_rows = connection.execute(
                    "SELECT version_num FROM alembic_version ORDER BY version_num"
                ).fetchall()
            if probe is None or probe[0] != 1:
                raise RuntimeError("database probe returned no row")
        except Exception as exc:
            if _is_missing_revision_table(exc):
                return DatabaseCheckResult(
                    state=DatabaseState.MIGRATION_MISMATCH,
                    detail="database schema is not initialized",
                    latency_ms=_elapsed_ms(started),
                    expected_revision=expected,
                )
            return DatabaseCheckResult(
                state=DatabaseState.UNAVAILABLE,
                detail=f"database unavailable ({_safe_exception_name(exc)})",
                latency_ms=_elapsed_ms(started),
                expected_revision=expected,
            )

        revisions = tuple(str(row[0]) for row in revision_rows)
        current = ",".join(revisions) or None
        if revisions != (expected,):
            return DatabaseCheckResult(
                state=DatabaseState.MIGRATION_MISMATCH,
                detail="database schema revision does not match application",
                latency_ms=_elapsed_ms(started),
                current_revision=current,
                expected_revision=expected,
            )
        return DatabaseCheckResult(
            state=DatabaseState.READY,
            detail="database and schema revision are ready",
            latency_ms=_elapsed_ms(started),
            current_revision=current,
            expected_revision=expected,
        )

    def check_runtime_role(self, *, strict: bool = False) -> DatabaseCheckResult:
        """Verify the connected principal is not a privileged deployment role."""

        started = time.monotonic()
        try:
            with self.connection() as connection:
                row = connection.execute(
                    """
                    SELECT
                        current_user,
                        session_user,
                        rolsuper,
                        rolcreaterole,
                        rolcreatedb,
                        rolbypassrls
                    FROM pg_roles
                    WHERE rolname = current_user
                    """
                ).fetchone()
            if row is None:
                raise RuntimeError("current database role was not found")
        except Exception as exc:
            return DatabaseCheckResult(
                state=DatabaseState.UNAVAILABLE,
                detail=f"database unavailable ({_safe_exception_name(exc)})",
                latency_ms=_elapsed_ms(started),
            )

        current_user = str(row[0])
        session_user = str(row[1])
        unsafe_reason = _unsafe_runtime_role_reason(
            current_user=current_user,
            is_superuser=bool(row[2]),
            can_create_role=bool(row[3]),
            can_create_db=bool(row[4]),
            bypasses_rls=bool(row[5]),
        )
        if unsafe_reason and strict:
            return DatabaseCheckResult(
                state=DatabaseState.UNSAFE_RUNTIME_ROLE,
                detail="database runtime role is unsafe for production",
                latency_ms=_elapsed_ms(started),
                current_user=current_user,
                session_user=session_user,
                unsafe_reason=unsafe_reason,
            )
        return DatabaseCheckResult(
            state=DatabaseState.READY,
            detail=(
                "database runtime role is acceptable"
                if unsafe_reason is None
                else "database runtime role is unsafe; strict mode is disabled"
            ),
            latency_ms=_elapsed_ms(started),
            current_user=current_user,
            session_user=session_user,
            unsafe_reason=unsafe_reason,
        )

    def health(self) -> DatabaseCheckResult:
        return self.check_health()

    def readiness(self) -> DatabaseCheckResult:
        return self.check_readiness()


PostgresPool = PsycopgPool


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 3)


def _safe_exception_name(exc: Exception) -> str:
    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else "DatabaseError"


def _unsafe_runtime_role_reason(
    *,
    current_user: str,
    is_superuser: bool,
    can_create_role: bool,
    can_create_db: bool,
    bypasses_rls: bool,
) -> str | None:
    if current_user in {"postgres", "eva_owner", "eva_migrator"}:
        return f"forbidden runtime role: {current_user}"
    if is_superuser:
        return "runtime role is superuser"
    if bypasses_rls:
        return "runtime role bypasses row-level security"
    if can_create_role:
        return "runtime role can create roles"
    if can_create_db:
        return "runtime role can create databases"
    return None


def _is_missing_revision_table(exc: Exception) -> bool:
    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate == "42P01":
        return True
    cause = getattr(exc, "__cause__", None)
    return getattr(cause, "sqlstate", None) == "42P01"
