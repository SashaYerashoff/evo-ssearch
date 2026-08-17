#!/usr/bin/env python3
"""Capture and verify the no-data-loss contract around EVA migrations.

The database DSN is accepted only through ``EVA_DATABASE_DSN`` so credentials
never appear in the process list.  The generated manifest contains counts and
cryptographic fingerprints, not row values or secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

import psycopg
from psycopg import sql


MANIFEST_VERSION = 1
MANAGED_SCHEMAS = ("agent", "archive", "audit", "iam", "jobs")
KEY_CAPTURE_LIMIT = 100_000
CONTENT_PROTECTED_TABLES = frozenset(
    {
        "archive.probes",
        "archive.runtime_state",
        "iam.permissions",
        "iam.role_permissions",
        "iam.roles",
        "iam.sessions",
        "iam.user_channel_grants",
        "iam.user_roles",
        "iam.users",
    }
)


class PreservationError(RuntimeError):
    pass


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, memoryview):
        value = bytes(value)
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _digest(values: Sequence[Any]) -> str:
    encoded = json.dumps(
        [_json_value(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _table_identifier(schema: str, table: str) -> sql.Identifier:
    return sql.Identifier(schema, table)


def _table_names(connection: psycopg.Connection[Any]) -> list[tuple[str, str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_schema, table_name
              FROM information_schema.tables
             WHERE table_type = 'BASE TABLE'
               AND table_schema = ANY(%s)
             ORDER BY table_schema, table_name
            """,
            (list(MANAGED_SCHEMAS),),
        )
        return [(str(schema), str(table)) for schema, table in cursor.fetchall()]


def _columns(
    connection: psycopg.Connection[Any], schema: str, table: str
) -> list[tuple[str, str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name, data_type
              FROM information_schema.columns
             WHERE table_schema = %s AND table_name = %s
             ORDER BY ordinal_position
            """,
            (schema, table),
        )
        return [(str(name), str(data_type)) for name, data_type in cursor.fetchall()]


def _primary_key(
    connection: psycopg.Connection[Any], schema: str, table: str
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT attribute.attname
              FROM pg_index index_definition
              JOIN pg_class relation ON relation.oid = index_definition.indrelid
              JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
              JOIN unnest(index_definition.indkey) WITH ORDINALITY key(attnum, position)
                ON true
              JOIN pg_attribute attribute
                ON attribute.attrelid = relation.oid
               AND attribute.attnum = key.attnum
             WHERE namespace.nspname = %s
               AND relation.relname = %s
               AND index_definition.indisprimary
             ORDER BY key.position
            """,
            (schema, table),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def _fetch_rows(
    connection: psycopg.Connection[Any],
    schema: str,
    table: str,
    columns: Sequence[str],
    order_by: Sequence[str],
) -> Iterable[tuple[Any, ...]]:
    selected = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    ordering = sql.SQL(", ").join(sql.Identifier(column) for column in order_by)
    query = sql.SQL("SELECT {} FROM {} ORDER BY {}").format(
        selected,
        _table_identifier(schema, table),
        ordering,
    )
    with connection.cursor(name=f"eva_preserve_{schema}_{table}") as cursor:
        cursor.execute(query)
        while rows := cursor.fetchmany(2_000):
            yield from rows


def _capture_table(
    connection: psycopg.Connection[Any],
    schema: str,
    table: str,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    qualified = f"{schema}.{table}"
    column_rows = _columns(connection, schema, table)
    columns = [name for name, _type in column_rows]
    column_types = {name: kind for name, kind in column_rows}
    primary_key = _primary_key(connection, schema, table)
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("SELECT count(*) FROM {}").format(_table_identifier(schema, table))
        )
        row_count = int(cursor.fetchone()[0])

    result: dict[str, Any] = {
        "columns": columns,
        "primary_key": primary_key,
        "row_count": row_count,
    }
    capture_keys = bool(
        primary_key
        and (
            row_count <= KEY_CAPTURE_LIMIT
            or (baseline and baseline.get("key_hashes"))
        )
    )
    if capture_keys:
        key_hashes: list[str] = []
        protected_rows: dict[str, str] = {}
        protected_columns = list((baseline or {}).get("columns") or columns)
        selected = (
            protected_columns
            if qualified in CONTENT_PROTECTED_TABLES
            else primary_key
        )
        pk_positions = [selected.index(column) for column in primary_key]
        for row in _fetch_rows(connection, schema, table, selected, primary_key):
            key = tuple(row[position] for position in pk_positions)
            key_hash = _digest(key)
            key_hashes.append(key_hash)
            if qualified in CONTENT_PROTECTED_TABLES:
                protected_rows[key_hash] = _digest(row)
        result["key_hashes"] = key_hashes
        if qualified in CONTENT_PROTECTED_TABLES:
            result["protected_rows"] = protected_rows
    elif len(primary_key) == 1 and column_types.get(primary_key[0]) in {
        "bigint",
        "integer",
        "smallint",
    }:
        key = sql.Identifier(primary_key[0])
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT min({0}), max({0}), sum({0}) FROM {1}").format(
                    key,
                    _table_identifier(schema, table),
                )
            )
            minimum, maximum, total = cursor.fetchone()
        result["numeric_key_guard"] = {
            "column": primary_key[0],
            "minimum": minimum,
            "maximum": maximum,
            "sum": str(total) if total is not None else None,
        }

    if qualified == "archive.detections":
        baseline_evidence = (baseline or {}).get("evidence_counts") or {}
        evidence_columns = [
            name
            for name in ("thumbnail_b64", "image_path")
            if name in columns and (not baseline_evidence or name in baseline_evidence)
        ]
        expressions = [sql.SQL("count(*)")]
        expressions.extend(
            sql.SQL("count({})").format(sql.Identifier(column))
            for column in evidence_columns
        )
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT {} FROM {}").format(
                    sql.SQL(", ").join(expressions),
                    _table_identifier(schema, table),
                )
            )
            values = cursor.fetchone()
        result["evidence_counts"] = {
            "rows": int(values[0]),
            **{
                column: int(values[index + 1])
                for index, column in enumerate(evidence_columns)
            },
        }
    return result


def _can_bypass_row_security(connection: psycopg.Connection[Any]) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                  FROM pg_roles
                 WHERE pg_has_role(current_user, oid, 'USAGE')
                   AND (rolbypassrls OR rolsuper)
            )
            """
        )
        return bool(cursor.fetchone()[0])


def _force_rls_tables(connection: psycopg.Connection[Any]) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT namespace.nspname, relation.relname
              FROM pg_class relation
              JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = ANY(%s)
               AND relation.relkind IN ('r', 'p')
               AND relation.relforcerowsecurity
             ORDER BY namespace.nspname, relation.relname
            """,
            (list(MANAGED_SCHEMAS),),
        )
        return [f"{schema}.{table}" for schema, table in cursor.fetchall()]


def assert_row_visibility(connection: psycopg.Connection[Any]) -> None:
    """Refuse to certify preservation this connection cannot actually observe.

    EVA tenant tables run ``FORCE ROW LEVEL SECURITY`` with a policy keyed on
    ``current_setting('eva.tenant_id')``.  With that GUC unset the predicate is
    NULL, so a role without BYPASSRLS reads every protected table as empty.
    Capturing under such a role produces zero counts before *and* after the
    migration, which compares clean and prints a preservation guarantee that was
    never checked.  A silent false negative here is worse than no guard at all.
    """

    if _can_bypass_row_security(connection):
        return
    blind = _force_rls_tables(connection)
    if not blind:
        return
    raise PreservationError(
        f"{len(blind)} EVA table(s) enforce FORCE ROW LEVEL SECURITY and this "
        "connection cannot bypass it, so every protected row reads as absent "
        f"(for example {', '.join(blind[:3])}). Capturing now would certify "
        "preservation without inspecting a single tenant row. Re-run with a "
        "BYPASSRLS or superuser migration identity."
    )


def capture(
    connection: psycopg.Connection[Any],
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    assert_row_visibility(connection)
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database(), current_setting('server_version')")
        database, server_version = cursor.fetchone()
        cursor.execute("SELECT version_num FROM public.alembic_version LIMIT 1")
        revision_row = cursor.fetchone()
    baseline_tables = (baseline or {}).get("tables") or {}
    tables = {}
    for schema, table in _table_names(connection):
        qualified = f"{schema}.{table}"
        tables[qualified] = _capture_table(
            connection,
            schema,
            table,
            baseline_tables.get(qualified),
        )
    return {
        "manifest_version": MANIFEST_VERSION,
        "database": str(database),
        "postgresql": str(server_version),
        "schema_revision": str(revision_row[0]) if revision_row else "",
        "tables": tables,
    }


def compare_manifests(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if before.get("manifest_version") != MANIFEST_VERSION:
        return ["unsupported preservation manifest version"]
    if before.get("database") != after.get("database"):
        errors.append("database identity changed during migration")
    before_tables = before.get("tables") or {}
    after_tables = after.get("tables") or {}
    for qualified, expected in sorted(before_tables.items()):
        actual = after_tables.get(qualified)
        if actual is None:
            errors.append(f"pre-existing table disappeared: {qualified}")
            continue
        expected_count = int(expected.get("row_count") or 0)
        actual_count = int(actual.get("row_count") or 0)
        if actual_count < expected_count:
            errors.append(
                f"{qualified} lost rows: before={expected_count} after={actual_count}"
            )
        expected_keys = set(expected.get("key_hashes") or [])
        if expected_keys:
            missing = expected_keys - set(actual.get("key_hashes") or [])
            if missing:
                errors.append(f"{qualified} lost {len(missing)} pre-existing primary key(s)")
        expected_rows = expected.get("protected_rows") or {}
        actual_rows = actual.get("protected_rows") or {}
        changed = [
            key for key, digest in expected_rows.items() if actual_rows.get(key) != digest
        ]
        if changed:
            errors.append(f"{qualified} changed {len(changed)} protected row(s)")
        expected_evidence = expected.get("evidence_counts") or {}
        actual_evidence = actual.get("evidence_counts") or {}
        for label, count in expected_evidence.items():
            if int(actual_evidence.get(label) or 0) < int(count or 0):
                errors.append(
                    f"{qualified} lost {label} evidence: before={count} "
                    f"after={actual_evidence.get(label, 0)}"
                )
        numeric = expected.get("numeric_key_guard") or {}
        if numeric and expected_count:
            actual_numeric = actual.get("numeric_key_guard") or {}
            if actual_count == expected_count and any(
                str(actual_numeric.get(key)) != str(numeric.get(key))
                for key in ("column", "minimum", "maximum", "sum")
            ):
                errors.append(f"{qualified} primary-key population changed")
    return errors


def _connection() -> psycopg.Connection[Any]:
    dsn = str(os.environ.get("EVA_DATABASE_DSN") or "").strip()
    if not dsn:
        raise PreservationError("EVA_DATABASE_DSN is required in the process environment")
    return psycopg.connect(
        dsn,
        connect_timeout=15,
        options="-c default_transaction_read_only=on",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    capture_parser = subcommands.add_parser("capture")
    capture_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subcommands.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "capture":
            with _connection() as connection:
                current = capture(connection)
            _write_manifest(args.output, current)
            print(
                "OK: captured pre-migration preservation inventory "
                f"for {len(current['tables'])} table(s)"
            )
            return 0
        baseline = json.loads(args.input.read_text(encoding="utf-8"))
        with _connection() as connection:
            current = capture(connection, baseline=baseline)
        errors = compare_manifests(baseline, current)
        if errors:
            for error in errors:
                print(f"FAIL: {error}", file=os.sys.stderr)
            return 1
        print(
            "OK: all pre-existing users, probes, runtime state and archive rows "
            "survived the migration"
        )
        return 0
    except (OSError, ValueError, psycopg.Error, PreservationError) as exc:
        print(f"FAIL: database preservation guard: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
