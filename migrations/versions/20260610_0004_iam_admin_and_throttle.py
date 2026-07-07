"""Add durable IAM login throttling support.

Revision ID: 20260610_0004
Revises: 20260609_0003
Create Date: 2026-06-10
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260610_0004"
down_revision: str | None = "20260609_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _execute(sql: str) -> None:
    op.execute(sql)


def upgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        """
        CREATE TABLE iam.login_attempts (
            tenant_id uuid NOT NULL,
            throttle_key text NOT NULL,
            failed_attempts integer NOT NULL CHECK (failed_attempts >= 0),
            window_started_at timestamptz NOT NULL,
            locked_until timestamptz,
            updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, throttle_key),
            CHECK (throttle_key = btrim(throttle_key) AND throttle_key <> ''),
            CHECK (length(throttle_key) <= 512)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_iam_login_attempts_locked
        ON iam.login_attempts (tenant_id, locked_until)
        WHERE locked_until IS NOT NULL
        """
    )
    _execute("ALTER TABLE iam.login_attempts ENABLE ROW LEVEL SECURITY")
    _execute("ALTER TABLE iam.login_attempts FORCE ROW LEVEL SECURITY")
    _execute(
        """
        CREATE POLICY iam_login_attempts_tenant_isolation
        ON iam.login_attempts
        USING (
            tenant_id = NULLIF(current_setting('eva.tenant_id', true), '')::uuid
        )
        WITH CHECK (
            tenant_id = NULLIF(current_setting('eva.tenant_id', true), '')::uuid
        )
        """
    )
    _execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON iam.login_attempts TO eva_api"
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute("DROP TABLE IF EXISTS iam.login_attempts")
    _execute("RESET ROLE")
