#!/usr/bin/env python3
"""Create/update a scoped operator or engineer for live integration smoke.

This is a dev/acceptance helper, not a production credential policy. It keeps
the restricted-help smoke repeatable without hand-editing users in the UI.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

from eva_db import DatabaseSettings, PsycopgPool  # noqa: E402
from security import Permission, Role  # noqa: E402
from security.postgres_identity import PostgresIdentityRepository  # noqa: E402


def _load_local_env() -> None:
    if load_dotenv is None:
        return
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap a scoped EVA operator/engineer for live acceptance tests."
    )
    parser.add_argument(
        "--tenant-id",
        default=os.getenv("EVOSSEARCH_AUTH_TENANT_ID", ""),
        help="Tenant UUID; defaults to EVOSSEARCH_AUTH_TENANT_ID.",
    )
    parser.add_argument(
        "--actor-username",
        default=os.getenv("EVA_ADMIN_USERNAME", "admin"),
        help="Existing admin username used for attribution.",
    )
    parser.add_argument("--username", default="operator-smoke")
    parser.add_argument("--display-name", default="Live Smoke Operator")
    parser.add_argument(
        "--role",
        choices=(Role.OPERATOR.value, Role.ENGINEER.value),
        default=Role.OPERATOR.value,
        help="Scoped role to assign; engineer is required for probe/prompt Apply coverage.",
    )
    parser.add_argument(
        "--channel-id",
        default=os.getenv("EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID", ""),
        help="Allowed channel id; defaults to EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID.",
    )
    parser.add_argument(
        "--password-env",
        default="EVA_LIVE_OPERATOR_PASSWORD",
        help="Environment variable containing the smoke user's password.",
    )
    parser.add_argument(
        "--set-password",
        action="store_true",
        help="Reset password even if the smoke user already exists.",
    )
    parser.add_argument(
        "--base-url",
        default="https://127.0.0.1:5443",
        help="Base URL printed in the smoke command.",
    )
    return parser


def _tenant_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        raise SystemExit("A valid --tenant-id UUID is required.") from None


def _channel_id(raw: str) -> int:
    try:
        value = int(str(raw).strip())
    except Exception:
        raise SystemExit("A valid --channel-id is required.") from None
    if value <= 0:
        raise SystemExit("A valid --channel-id is required.")
    return value


def _password(env_name: str) -> str:
    value = os.getenv(env_name, "")
    if value:
        return value
    password = getpass.getpass(f"Password for smoke user ({env_name}): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")
    return password


def _admin_actor(
    repository: PostgresIdentityRepository,
    tenant_id: uuid.UUID,
    username: str,
) -> str:
    actor = repository.get_user_by_username(tenant_id, username)
    if actor is None:
        raise SystemExit(f"Admin actor not found: {username}")
    if not actor.is_active:
        raise SystemExit(f"Admin actor is inactive: {username}")
    if Permission.USERS_MANAGE.value not in actor.permissions:
        raise SystemExit(f"Actor {username!r} lacks {Permission.USERS_MANAGE.value}.")
    return actor.user_id


def main(argv: list[str] | None = None) -> int:
    _load_local_env()
    args = _parser().parse_args(argv)
    tenant_id = _tenant_id(args.tenant_id)
    channel_id = _channel_id(args.channel_id)

    pool = PsycopgPool(DatabaseSettings.from_env())
    repository = PostgresIdentityRepository(pool)
    try:
        actor_id = _admin_actor(repository, tenant_id, args.actor_username)
        existing = repository.get_user_by_username(
            tenant_id,
            args.username,
            actor_user_id=actor_id,
        )
        if existing is None:
            password = _password(args.password_env)
            record = repository.create_user(
                tenant_id,
                actor_user_id=actor_id,
                username=args.username,
                password=password,
                roles=[args.role],
                display_name=args.display_name,
                allowed_channel_ids=[channel_id],
                is_active=True,
            )
            action = "created"
        else:
            updates: dict[str, object] = {
                "display_name": args.display_name,
                "roles": [args.role],
                "allowed_channel_ids": [channel_id],
                "is_active": True,
            }
            if args.set_password:
                updates["password"] = _password(args.password_env)
            record = repository.update_user(
                tenant_id,
                existing.user_id,
                actor_user_id=actor_id,
                **updates,
            )
            action = "updated"
    finally:
        pool.close()

    print(
        f"Smoke user {action}: tenant={tenant_id} role={args.role} "
        f"user={record.username} id={record.user_id} channel={channel_id}"
    )
    print("\n# live agent acceptance:")
    print(f"export EVA_LIVE_BASE_URL={args.base_url}")
    print(f"export EVA_LIVE_USER={record.username}")
    print(f"export EVA_LIVE_CHANNEL_REF={channel_id}")
    include = "non_admin" if args.role == Role.OPERATOR.value else "probe_apply,prompt_preview"
    print(f"export EVA_LIVE_INCLUDE={include}")
    print(f"# export {args.password_env}=<operator password>")
    print(
        ".venv/bin/pytest -q tests/integration/test_live_agent.py -s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
