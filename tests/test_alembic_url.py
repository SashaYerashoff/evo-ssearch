from __future__ import annotations

from urllib.parse import unquote, urlsplit

import pytest

from eva_db.alembic_url import sqlalchemy_database_url


def test_postgresql_uri_selects_psycopg_driver() -> None:
    value = sqlalchemy_database_url("postgresql://eva:secret@db.internal:5433/eva")

    assert value == "postgresql+psycopg://eva:secret@db.internal:5433/eva"


def test_libpq_conninfo_becomes_unredacted_sqlalchemy_url() -> None:
    value = sqlalchemy_database_url(
        "host=127.0.0.1 port=15433 dbname=eva user=postgres "
        "password='space secret' sslmode=disable connect_timeout=5"
    )
    parts = urlsplit(value)

    assert parts.scheme == "postgresql+psycopg"
    assert parts.hostname == "127.0.0.1"
    assert parts.port == 15433
    assert parts.username == "postgres"
    assert unquote(parts.password or "") == "space secret"
    assert parts.path == "/eva"
    assert "sslmode=disable" in parts.query
    assert "connect_timeout=5" in parts.query
    assert "***" not in value


@pytest.mark.parametrize("value", ["", "not-a-dsn", "sqlite:///tmp/eva.db"])
def test_invalid_database_dsn_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        sqlalchemy_database_url(value)
