"""PostgreSQL foundation for EVA AI.

This package deliberately imports without psycopg, psycopg_pool, SQLAlchemy, or
Alembic installed. Optional database dependencies are loaded only when a pool is
opened or migrations are executed.
"""

from .pool import (
    DatabaseCheckResult,
    DatabaseDependencyError,
    DatabaseState,
    PostgresPool,
    PsycopgPool,
    TransactionContext,
)
from .settings import (
    CURRENT_SCHEMA_REVISION,
    DatabaseConfigurationError,
    DatabaseSettings,
    redact_dsn,
)

__all__ = [
    "CURRENT_SCHEMA_REVISION",
    "DatabaseCheckResult",
    "DatabaseConfigurationError",
    "DatabaseDependencyError",
    "DatabaseSettings",
    "DatabaseState",
    "PostgresPool",
    "PsycopgPool",
    "TransactionContext",
    "redact_dsn",
]
