#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional developer convenience
    load_dotenv = None  # type: ignore[assignment]

from eva_db import DatabaseSettings, PsycopgPool  # noqa: E402


@dataclass(frozen=True, slots=True)
class RuntimeLogin:
    login_role: str
    group_role: str
    password_env: str


RUNTIME_LOGINS = (
    RuntimeLogin("eva_migrator_login", "eva_migrator", "EVA_MIGRATOR_PASSWORD"),
    RuntimeLogin("eva_api_login", "eva_api", "EVA_API_PASSWORD"),
    RuntimeLogin("eva_audit_login", "eva_audit_writer", "EVA_AUDIT_PASSWORD"),
    RuntimeLogin("eva_worker_login", "eva_worker", "EVA_WORKER_PASSWORD"),
    RuntimeLogin("eva_backup_login", "eva_backup", "EVA_BACKUP_PASSWORD"),
)


def _load_local_env() -> None:
    if load_dotenv is None:
        return
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create least-privilege EVA PostgreSQL login roles and grant them "
            "membership in the NOLOGIN runtime groups created by migrations."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate passwords and current connection role without changing DB.",
    )
    return parser


def _required_password(env_name: str) -> str:
    value = os.getenv(env_name, "")
    if len(value) < 20:
        raise SystemExit(f"{env_name} must contain at least 20 characters.")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise SystemExit(f"{env_name} contains unsafe characters.")
    return value


def _assert_privileged_connection(pool: PsycopgPool) -> None:
    result = pool.check_runtime_role(strict=False)
    if result.unsafe_reason is None:
        raise SystemExit(
            "This command must be run with a privileged deployment DSN "
            "(for example postgres or eva_migrator_login)."
        )


def _create_or_update_login(connection, login: RuntimeLogin, password: str) -> None:
    from psycopg import sql

    connection.execute(
        sql.SQL(
            """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = {login_role_name}
            ) THEN
                CREATE ROLE {login_role}
                    LOGIN
                    INHERIT
                    NOSUPERUSER
                    NOCREATEDB
                    NOCREATEROLE
                    NOREPLICATION
                    NOBYPASSRLS;
            END IF;
        END
        $$
        """
        ).format(
            login_role_name=sql.Literal(login.login_role),
            login_role=sql.Identifier(login.login_role),
        )
    )
    connection.execute(
        sql.SQL(
            """
        ALTER ROLE {login_role}
            WITH
            LOGIN
            INHERIT
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            NOREPLICATION
            NOBYPASSRLS
            PASSWORD {password}
        """
        ).format(
            login_role=sql.Identifier(login.login_role),
            password=sql.Literal(password),
        )
    )
    connection.execute(
        sql.SQL("GRANT {group_role} TO {login_role}").format(
            group_role=sql.Identifier(login.group_role),
            login_role=sql.Identifier(login.login_role),
        )
    )


def _env_matrix() -> str:
    return "\n".join(
        (
            "# Runtime DSNs; replace passwords with the matching secret values.",
            "EVA_DATABASE_DSN=postgresql://eva_api_login:${EVA_API_PASSWORD}@db/eva",
            (
                "EVA_AUDIT_DATABASE_DSN="
                "postgresql://eva_audit_login:${EVA_AUDIT_PASSWORD}@db/eva"
            ),
            (
                "EVA_WORKER_DATABASE_DSN="
                "postgresql://eva_worker_login:${EVA_WORKER_PASSWORD}@db/eva"
            ),
            "EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true",
        )
    )


def main(argv: list[str] | None = None) -> int:
    _load_local_env()
    args = _parser().parse_args(argv)
    passwords = {
        login.password_env: _required_password(login.password_env)
        for login in RUNTIME_LOGINS
    }
    pool = PsycopgPool(DatabaseSettings.from_env())
    try:
        _assert_privileged_connection(pool)
        if args.check_only:
            print("Role bootstrap preflight OK.")
            return 0
        with pool.connection() as connection:
            with connection.transaction():
                for login in RUNTIME_LOGINS:
                    _create_or_update_login(
                        connection,
                        login,
                        passwords[login.password_env],
                    )
        print("Runtime database login roles are ready.")
        print(_env_matrix())
        return 0
    finally:
        pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
