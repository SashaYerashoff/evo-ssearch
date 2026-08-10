"""Alembic environment for the EVA PostgreSQL schema."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from eva_db.alembic_url import sqlalchemy_database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _database_url() -> str:
    dsn = (
        os.getenv("EVA_DATABASE_DSN")
        or os.getenv("EVA_DATABASE_URL")
        or os.getenv("EVOSSEARCH_DATABASE_DSN")
        or os.getenv("EVOSSEARCH_DATABASE_URL")
        or os.getenv("DATABASE_URL")
    )
    if not dsn:
        raise RuntimeError("EVA_DATABASE_DSN is required to run migrations")
    return sqlalchemy_database_url(dsn)


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    try:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                transaction_per_migration=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
