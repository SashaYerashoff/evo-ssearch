#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional developer convenience
    load_dotenv = None  # type: ignore[assignment]

from eva_db import DatabaseSettings, PsycopgPool  # noqa: E402
from security import ALL_CHANNELS, Permission, Role  # noqa: E402
from security.postgres_identity import (  # noqa: E402
    IdentityRecord,
    PostgresIdentityRepository,
)


def _load_local_env() -> None:
    if load_dotenv is None:
        return
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage EVA tenant users, roles, channel grants, and sessions."
    )
    parser.add_argument(
        "--tenant-id",
        default=os.getenv("EVOSSEARCH_AUTH_TENANT_ID", ""),
        help="Tenant UUID; defaults to EVOSSEARCH_AUTH_TENANT_ID.",
    )
    parser.add_argument(
        "--actor-user-id",
        default=os.getenv("EVA_ADMIN_USER_ID", ""),
        help="Admin actor user UUID for attribution.",
    )
    parser.add_argument(
        "--actor-username",
        default=os.getenv("EVA_ADMIN_USERNAME", "admin"),
        help="Admin actor username used when --actor-user-id is absent.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List users.")
    _add_json_alias(list_parser)
    list_parser.add_argument(
        "--active-only",
        action="store_true",
        help="Hide inactive users.",
    )

    show_parser = subparsers.add_parser("show", help="Show one user.")
    _add_json_alias(show_parser)
    show_parser.add_argument("user", help="User UUID or username.")

    create_parser = subparsers.add_parser("create", help="Create a user.")
    _add_json_alias(create_parser)
    create_parser.add_argument("username")
    create_parser.add_argument("--display-name")
    create_parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        default=[],
        help="Role name; may be repeated. Defaults to viewer.",
    )
    create_parser.add_argument(
        "--channels",
        default="",
        help="Comma-separated positive channel ids.",
    )
    create_parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Grant all-channel access; valid for admin users.",
    )
    create_parser.add_argument(
        "--inactive",
        action="store_true",
        help="Create disabled account.",
    )
    create_parser.add_argument(
        "--password-env",
        default="EVA_USER_PASSWORD",
        help="Environment variable containing the new password.",
    )

    update_parser = subparsers.add_parser("update", help="Update a user.")
    _add_json_alias(update_parser)
    update_parser.add_argument("user", help="User UUID or username.")
    update_parser.add_argument("--display-name")
    update_parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        default=None,
        help="Replacement role; may be repeated.",
    )
    update_parser.add_argument(
        "--channels",
        help="Replacement comma-separated positive channel ids.",
    )
    update_parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Grant all-channel access; valid for admin users.",
    )
    update_parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate the account.",
    )
    update_parser.add_argument(
        "--deactivate",
        action="store_true",
        help="Deactivate the account and revoke sessions.",
    )
    update_parser.add_argument(
        "--set-password",
        action="store_true",
        help="Read a replacement password from --password-env or prompt.",
    )
    update_parser.add_argument(
        "--password-env",
        default="EVA_USER_PASSWORD",
        help="Environment variable containing the replacement password.",
    )

    revoke_parser = subparsers.add_parser(
        "revoke-sessions",
        help="Revoke all active sessions for a user.",
    )
    _add_json_alias(revoke_parser)
    revoke_parser.add_argument("user", help="User UUID or username.")
    revoke_parser.add_argument(
        "--reason",
        default="admin_revoked",
        help="Revocation reason stored with session rows.",
    )
    return parser


def _add_json_alias(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def _tenant_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        raise SystemExit("A valid --tenant-id UUID is required.") from None


def _parse_roles(raw_roles: Iterable[str]) -> list[str]:
    roles = [
        token.strip().lower()
        for item in raw_roles
        for token in str(item).split(",")
        if token.strip()
    ]
    if not roles:
        return [Role.VIEWER.value]
    for role in roles:
        Role(role)
    return roles


def _parse_channels(raw: str, *, all_channels: bool) -> list[int | str]:
    if all_channels:
        return [ALL_CHANNELS]
    if not raw.strip():
        return []
    channels: list[int | str] = []
    for token in raw.split(","):
        value = token.strip()
        if not value:
            continue
        try:
            channel_id = int(value)
        except ValueError:
            raise SystemExit(f"invalid channel id: {value!r}") from None
        if channel_id <= 0:
            raise SystemExit(f"invalid channel id: {value!r}")
        channels.append(channel_id)
    return channels


def _password(env_name: str, *, prompt: str) -> str:
    value = os.getenv(env_name, "")
    if value:
        return value
    password = getpass.getpass(prompt)
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")
    return password


def _actor_user_id(
    repository: PostgresIdentityRepository,
    tenant_id: uuid.UUID,
    args: argparse.Namespace,
) -> str:
    if args.actor_user_id:
        actor_id = uuid.UUID(str(args.actor_user_id))
        actor = repository.get_user(
            tenant_id,
            actor_id,
            actor_user_id=actor_id,
        )
    else:
        actor = repository.get_user_by_username(
            tenant_id,
            args.actor_username,
        )
    if actor is None:
        raise SystemExit(
            "Admin actor was not found. Pass --actor-user-id or set "
            "EVA_ADMIN_USERNAME to an existing admin user."
        )
    if not actor.is_active:
        raise SystemExit("Admin actor is inactive.")
    if Permission.USERS_MANAGE.value not in actor.permissions:
        raise SystemExit("Admin actor must have users:manage permission.")
    return actor.user_id


def _resolve_user(
    repository: PostgresIdentityRepository,
    tenant_id: uuid.UUID,
    actor_user_id: str,
    user: str,
) -> IdentityRecord:
    try:
        record = repository.get_user(
            tenant_id,
            uuid.UUID(str(user)),
            actor_user_id=actor_user_id,
        )
    except ValueError:
        record = repository.get_user_by_username(
            tenant_id,
            user,
            actor_user_id=actor_user_id,
        )
    if record is None:
        raise SystemExit(f"user not found: {user}")
    return record


def _record_payload(record: IdentityRecord) -> dict[str, Any]:
    return {
        "id": record.user_id,
        "tenant_id": record.tenant_id,
        "username": record.username,
        "display_name": record.display_name,
        "roles": sorted(record.roles),
        "permissions": sorted(record.permissions),
        "allowed_channel_ids": sorted(
            record.allowed_channel_ids,
            key=lambda value: str(value),
        ),
        "is_active": record.is_active,
    }


def _emit(args: argparse.Namespace, payload: Any) -> None:
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return
    if isinstance(payload, list):
        for item in payload:
            print(_format_record(item))
        return
    if isinstance(payload, dict) and "user" in payload:
        print(_format_record(payload["user"]))
        return
    print(payload)


def _format_record(payload: dict[str, Any]) -> str:
    channels = ",".join(str(item) for item in payload["allowed_channel_ids"]) or "-"
    roles = ",".join(payload["roles"]) or "-"
    status = "active" if payload["is_active"] else "inactive"
    return (
        f"{payload['id']}  {payload['username']}  {status}  "
        f"roles={roles}  channels={channels}"
    )


def main(argv: list[str] | None = None) -> int:
    _load_local_env()
    args = _parser().parse_args(argv)
    tenant_id = _tenant_id(args.tenant_id)
    pool = PsycopgPool(DatabaseSettings.from_env())
    repository = PostgresIdentityRepository(pool)
    try:
        actor_user_id = _actor_user_id(repository, tenant_id, args)
        if args.command == "list":
            records = repository.list_users(
                tenant_id,
                actor_user_id=actor_user_id,
                include_inactive=not args.active_only,
            )
            _emit(args, [_record_payload(record) for record in records])
            return 0
        if args.command == "show":
            record = _resolve_user(repository, tenant_id, actor_user_id, args.user)
            _emit(args, {"user": _record_payload(record)})
            return 0
        if args.command == "create":
            record = repository.create_user(
                tenant_id,
                actor_user_id=actor_user_id,
                username=args.username,
                password=_password(
                    args.password_env,
                    prompt=f"Password for {args.username}: ",
                ),
                roles=_parse_roles(args.roles),
                display_name=args.display_name,
                allowed_channel_ids=_parse_channels(
                    args.channels,
                    all_channels=args.all_channels,
                ),
                is_active=not args.inactive,
            )
            _emit(args, {"user": _record_payload(record)})
            return 0
        if args.command == "update":
            target = _resolve_user(repository, tenant_id, actor_user_id, args.user)
            updates: dict[str, Any] = {}
            if args.display_name is not None:
                updates["display_name"] = args.display_name
            if args.roles is not None:
                updates["roles"] = _parse_roles(args.roles)
            if args.channels is not None or args.all_channels:
                updates["allowed_channel_ids"] = _parse_channels(
                    args.channels or "",
                    all_channels=args.all_channels,
                )
            if args.activate and args.deactivate:
                raise SystemExit("--activate and --deactivate are mutually exclusive")
            if args.activate:
                updates["is_active"] = True
            if args.deactivate:
                updates["is_active"] = False
            if args.set_password:
                updates["password"] = _password(
                    args.password_env,
                    prompt=f"New password for {target.username}: ",
                )
            if not updates:
                raise SystemExit("No update fields were provided.")
            record = repository.update_user(
                tenant_id,
                target.user_id,
                actor_user_id=actor_user_id,
                **updates,
            )
            _emit(args, {"user": _record_payload(record)})
            return 0
        if args.command == "revoke-sessions":
            target = _resolve_user(repository, tenant_id, actor_user_id, args.user)
            count = repository.revoke_user_sessions(
                tenant_id,
                target.user_id,
                actor_user_id=actor_user_id,
                reason=args.reason,
            )
            _emit(args, f"Revoked sessions: {count}")
            return 0
    finally:
        pool.close()
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
