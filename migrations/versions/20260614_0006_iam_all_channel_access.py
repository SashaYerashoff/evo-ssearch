"""Add explicit all-channel IAM grants.

Revision ID: 20260614_0006
Revises: 20260612_0005
Create Date: 2026-06-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260614_0006"
down_revision: str | None = "20260612_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _execute(sql: str) -> None:
    op.execute(sql)


def upgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        """
        ALTER TABLE iam.users
        ADD COLUMN IF NOT EXISTS all_channel_access boolean NOT NULL DEFAULT false
        """
    )
    _execute(
        """
        UPDATE iam.users AS u
        SET all_channel_access = true
        WHERE EXISTS (
            SELECT 1
            FROM iam.user_roles AS ur
            JOIN iam.roles AS r
              ON r.tenant_id = ur.tenant_id AND r.id = ur.role_id
            WHERE ur.tenant_id = u.tenant_id
              AND ur.user_id = u.id
              AND r.name = 'admin'
        )
        """
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute("ALTER TABLE iam.users DROP COLUMN IF EXISTS all_channel_access")
    _execute("RESET ROLE")
