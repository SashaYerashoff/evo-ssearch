#!/usr/bin/env python3
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

from eva_db import DatabaseSettings, PsycopgPool  # noqa: E402
from security.postgres_identity import PostgresIdentityRepository  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the first EVA tenant administrator."
    )
    parser.add_argument(
        "--tenant-id",
        default=os.getenv("EVOSSEARCH_AUTH_TENANT_ID", ""),
        help="Tenant UUID; defaults to EVOSSEARCH_AUTH_TENANT_ID.",
    )
    parser.add_argument("--username", required=True)
    parser.add_argument("--display-name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        tenant_id = uuid.UUID(str(args.tenant_id))
    except ValueError:
        print("A valid --tenant-id UUID is required.", file=sys.stderr)
        return 2

    password = os.getenv("EVA_BOOTSTRAP_ADMIN_PASSWORD")
    if not password:
        password = getpass.getpass("Administrator password: ")
        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            print("Passwords do not match.", file=sys.stderr)
            return 2
    if len(password) < 12:
        print("Password must contain at least 12 characters.", file=sys.stderr)
        return 2

    settings = DatabaseSettings.from_env()
    pool = PsycopgPool(settings)
    try:
        identity = PostgresIdentityRepository(pool).bootstrap_admin(
            tenant_id,
            args.username,
            password,
            args.display_name,
        )
    finally:
        pool.close()

    print(
        f"Administrator ready: tenant={identity.tenant_id} "
        f"user={identity.username} id={identity.user_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
