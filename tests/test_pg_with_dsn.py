from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pg_with_dsn_test",
    ROOT / "scripts" / "pg_with_dsn.py",
)
assert SPEC and SPEC.loader
pg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pg
SPEC.loader.exec_module(pg)


def test_keyword_conninfo_becomes_individual_libpq_environment_variables():
    dsn = "host=db.internal port=5433 dbname=eva user=migrator password='space secret'"

    environment = pg.postgres_environment(
        dsn,
        {
            pg.DSN_ENV: dsn,
            "EVA_INSTALL_MIGRATION_DSN": dsn,
            "PGHOST": "stale-host",
            "PATH": "/usr/bin",
        },
    )

    assert environment["PGHOST"] == "db.internal"
    assert environment["PGPORT"] == "5433"
    assert environment["PGDATABASE"] == "eva"
    assert environment["PGUSER"] == "migrator"
    assert environment["PGPASSWORD"] == "space secret"
    assert environment["PATH"] == "/usr/bin"
    assert pg.DSN_ENV not in environment
    assert "EVA_INSTALL_MIGRATION_DSN" not in environment


def test_percent_encoded_postgres_uri_is_supported():
    environment = pg.postgres_environment(
        "postgresql://eva%20user:p%40ss@db.internal:5444/eva%20archive?sslmode=require",
        {},
    )

    assert environment["PGUSER"] == "eva user"
    assert environment["PGPASSWORD"] == "p@ss"
    assert environment["PGHOST"] == "db.internal"
    assert environment["PGPORT"] == "5444"
    assert environment["PGDATABASE"] == "eva archive"
    assert environment["PGSSLMODE"] == "require"


def test_unsupported_conninfo_parameter_fails_closed_without_showing_value():
    with pytest.raises(pg.DsnError, match="keepalives") as raised:
        pg.postgres_environment(
            "host=db.internal dbname=eva keepalives=1",
            {},
        )

    assert "db.internal" not in str(raised.value)


def test_cli_keeps_dsn_out_of_child_argv_and_removes_source_environment():
    dsn = "host=db.internal dbname=eva user=migrator password=secret"
    captured = {}

    def run(command, *, env, check):
        captured["command"] = list(command)
        captured["environment"] = dict(env)
        return type("Completed", (), {"returncode": 0})()

    with (
        patch.dict(pg.os.environ, {pg.DSN_ENV: dsn}, clear=True),
        patch.object(pg.subprocess, "run", side_effect=run),
    ):
        assert pg.main(["--", "psql", "-Atc", "select 1"]) == 0

    assert captured["command"] == ["psql", "-Atc", "select 1"]
    assert dsn not in captured["command"]
    assert pg.DSN_ENV not in captured["environment"]
    assert captured["environment"]["PGPASSWORD"] == "secret"
