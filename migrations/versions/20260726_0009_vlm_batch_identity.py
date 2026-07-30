"""Index stable VLM batch membership stored in archive payloads.

Revision ID: 20260726_0009
Revises: 20260726_0008
Create Date: 2026-07-26
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260726_0009"
down_revision: str | None = "20260726_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _execute(sql: str) -> None:
    op.execute(sql)


def upgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        """
        CREATE INDEX ix_archive_detections_vlm_batch
        ON archive.detections (
            tenant_id,
            channel_id,
            (payload_json->>'batch_id'),
            event_timestamp_ms DESC,
            id DESC
        )
        WHERE source = 'vlm_summary'
          AND payload_json ? 'batch_id'
        """
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute("DROP INDEX IF EXISTS archive.ix_archive_detections_vlm_batch")
    _execute("RESET ROLE")
