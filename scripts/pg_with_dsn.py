#!/usr/bin/env python3
"""Run a libpq command without placing its connection string in argv.

The connection string is read only from ``EVA_PG_CONNECT_DSN``.  It is parsed
into the ordinary libpq ``PG*`` environment variables, removed from the child
environment, and never printed.  URI and keyword/value conninfo forms are
supported; parsing uses psycopg/libpq when available and a bounded stdlib
fallback during early offline-installer stages.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import Mapping
from urllib.parse import parse_qsl, unquote, urlsplit


DSN_ENV = "EVA_PG_CONNECT_DSN"
SENSITIVE_DSN_ENV = (
    DSN_ENV,
    "EVA_INSTALL_MIGRATION_DSN",
    "EVA_MIGRATION_DATABASE_DSN",
    "EVA_PATCH_PG_DSN",
    "EVA_DATABASE_DSN",
    "EVA_AUDIT_DATABASE_DSN",
    "EVA_WORKER_DATABASE_DSN",
    "EVOSSEARCH_DATABASE_DSN",
)
PG_ENV_BY_KEY = {
    "service": "PGSERVICE",
    "user": "PGUSER",
    "password": "PGPASSWORD",
    "passfile": "PGPASSFILE",
    "channel_binding": "PGCHANNELBINDING",
    "connect_timeout": "PGCONNECT_TIMEOUT",
    "dbname": "PGDATABASE",
    "host": "PGHOST",
    "hostaddr": "PGHOSTADDR",
    "port": "PGPORT",
    "client_encoding": "PGCLIENTENCODING",
    "options": "PGOPTIONS",
    "application_name": "PGAPPNAME",
    "sslmode": "PGSSLMODE",
    "sslnegotiation": "PGSSLNEGOTIATION",
    "sslcompression": "PGSSLCOMPRESSION",
    "sslcert": "PGSSLCERT",
    "sslkey": "PGSSLKEY",
    "sslcertmode": "PGSSLCERTMODE",
    "sslrootcert": "PGSSLROOTCERT",
    "sslcrl": "PGSSLCRL",
    "sslcrldir": "PGSSLCRLDIR",
    "sslsni": "PGSSLSNI",
    "requirepeer": "PGREQUIREPEER",
    "require_auth": "PGREQUIREAUTH",
    "min_protocol_version": "PGMINPROTOCOLVERSION",
    "max_protocol_version": "PGMAXPROTOCOLVERSION",
    "ssl_min_protocol_version": "PGSSLMINPROTOCOLVERSION",
    "ssl_max_protocol_version": "PGSSLMAXPROTOCOLVERSION",
    "gssencmode": "PGGSSENCMODE",
    "krbsrvname": "PGKRBSRVNAME",
    "gsslib": "PGGSSLIB",
    "gssdelegation": "PGGSSDELEGATION",
    "target_session_attrs": "PGTARGETSESSIONATTRS",
    "load_balance_hosts": "PGLOADBALANCEHOSTS",
}


class DsnError(ValueError):
    pass


def _stdlib_keyword_conninfo(dsn: str) -> dict[str, str]:
    try:
        tokens = shlex.split(dsn, posix=True)
    except ValueError as exc:
        raise DsnError("invalid quoted keyword/value connection string") from exc
    parsed: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            raise DsnError("keyword/value connection string contains a token without '='")
        key, value = token.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise DsnError("connection string contains an empty keyword")
        if key in parsed:
            raise DsnError(f"connection string repeats keyword {key}")
        parsed[key] = value
    return parsed


def _stdlib_uri_conninfo(dsn: str) -> dict[str, str]:
    parsed_uri = urlsplit(dsn)
    if parsed_uri.scheme not in {"postgres", "postgresql"}:
        raise DsnError("PostgreSQL URI must use postgres:// or postgresql://")
    if "," in parsed_uri.netloc:
        raise DsnError("multi-host PostgreSQL URIs require psycopg/libpq parsing")
    parsed: dict[str, str] = {}
    if parsed_uri.username is not None:
        parsed["user"] = unquote(parsed_uri.username)
    if parsed_uri.password is not None:
        parsed["password"] = unquote(parsed_uri.password)
    if parsed_uri.hostname is not None:
        parsed["host"] = unquote(parsed_uri.hostname)
    try:
        port = parsed_uri.port
    except ValueError as exc:
        raise DsnError("PostgreSQL URI contains an invalid port") from exc
    if port is not None:
        parsed["port"] = str(port)
    if parsed_uri.path and parsed_uri.path != "/":
        parsed["dbname"] = unquote(parsed_uri.path.lstrip("/"))
    for key, value in parse_qsl(parsed_uri.query, keep_blank_values=True):
        parsed[key.lower()] = value
    return parsed


def parse_conninfo(dsn: str) -> dict[str, str]:
    normalized = str(dsn or "").strip()
    if not normalized:
        raise DsnError(f"{DSN_ENV} is empty")
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError:
        conninfo_to_dict = None
    if conninfo_to_dict is not None:
        try:
            return {
                str(key).lower(): str(value)
                for key, value in conninfo_to_dict(normalized).items()
                if value is not None
            }
        except Exception as exc:
            raise DsnError("libpq rejected the PostgreSQL connection string") from exc
    if normalized.startswith(("postgres://", "postgresql://")):
        return _stdlib_uri_conninfo(normalized)
    return _stdlib_keyword_conninfo(normalized)


def postgres_environment(
    dsn: str,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = parse_conninfo(dsn)
    unsupported = sorted(set(values) - set(PG_ENV_BY_KEY))
    if unsupported:
        raise DsnError(
            "connection string uses parameters with no safe libpq environment "
            f"mapping: {', '.join(unsupported)}"
        )
    environment = dict(os.environ if base is None else base)
    for variable in SENSITIVE_DSN_ENV:
        environment.pop(variable, None)
    for variable in set(PG_ENV_BY_KEY.values()):
        environment.pop(variable, None)
    for key, value in values.items():
        environment[PG_ENV_BY_KEY[key]] = value
    return environment


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        print("usage: pg_with_dsn.py -- COMMAND [ARG ...]", file=sys.stderr)
        return 2
    try:
        environment = postgres_environment(os.environ.get(DSN_ENV, ""))
    except DsnError as exc:
        print(f"PostgreSQL connection setup failed: {exc}", file=sys.stderr)
        return 2
    return subprocess.run(arguments, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
