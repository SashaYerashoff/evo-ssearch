"""Repair the source/channel archive paging index on upgraded databases.

Revision ID: 20260805_0013
Revises: 20260805_0012
Create Date: 2026-08-05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260805_0013"
down_revision: str | None = "20260805_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _execute(sql: str) -> None:
    op.execute(sql)


def upgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    # Some databases upgraded from the field bundle are missing the original
    # source/channel index from 0005.  Use a new name so every upgrade obtains
    # the paging order needed by archive readers, even when an older index with
    # a similar name exists but lacks the deterministic id suffix.
    _execute(
        """
        CREATE INDEX IF NOT EXISTS ix_archive_detections_source_channel_event_id
        ON archive.detections (
            tenant_id,
            source,
            channel_id,
            event_timestamp_ms DESC,
            id DESC
        )
        """
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        "DROP INDEX IF EXISTS archive.ix_archive_detections_source_channel_event_id"
    )
    _execute("RESET ROLE")
