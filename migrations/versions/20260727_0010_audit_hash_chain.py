"""Prepare tenant-scoped audit hash-chain writes.

Revision ID: 20260727_0010
Revises: 20260726_0009
Create Date: 2026-07-27
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260727_0010"
down_revision: str | None = "20260726_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _execute(sql: str) -> None:
    op.execute(sql)


def upgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        """
        ALTER TABLE audit.events
        ADD CONSTRAINT ck_audit_events_previous_hash_length
            CHECK (
                previous_event_hash IS NULL
                OR octet_length(previous_event_hash) = 32
            ),
        ADD CONSTRAINT ck_audit_events_hash_length
            CHECK (
                event_hash IS NULL
                OR octet_length(event_hash) = 32
            )
        """
    )
    _execute(
        """
        CREATE INDEX ix_audit_events_tenant_sequence
        ON audit.events (tenant_id, sequence_number DESC)
        INCLUDE (event_hash)
        """
    )
    _execute(
        """
        GRANT SELECT (tenant_id, sequence_number, event_hash)
        ON audit.events TO eva_audit_writer
        """
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        """
        REVOKE SELECT (tenant_id, sequence_number, event_hash)
        ON audit.events FROM eva_audit_writer
        """
    )
    _execute(
        "DROP INDEX IF EXISTS audit.ix_audit_events_tenant_sequence"
    )
    _execute(
        """
        ALTER TABLE audit.events
        DROP CONSTRAINT IF EXISTS ck_audit_events_hash_length,
        DROP CONSTRAINT IF EXISTS ck_audit_events_previous_hash_length
        """
    )
    _execute("RESET ROLE")
