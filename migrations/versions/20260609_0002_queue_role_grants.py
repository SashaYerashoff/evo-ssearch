"""Allow API-side inference admission without worker privileges.

Revision ID: 20260609_0002
Revises: 20260609_0001
Create Date: 2026-06-09
"""

from __future__ import annotations

from alembic import op

revision: str = "20260609_0002"
down_revision: str | None = "20260609_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "GRANT UPDATE ON jobs.inference_jobs TO eva_api"
    )
    op.execute(
        "GRANT SELECT ON public.alembic_version "
        "TO eva_api, eva_worker, eva_agent_reader, eva_audit_writer, eva_backup"
    )
    op.execute(
        "GRANT SELECT (id) ON audit.events TO eva_audit_writer"
    )


def downgrade() -> None:
    op.execute(
        "REVOKE SELECT (id) ON audit.events FROM eva_audit_writer"
    )
    op.execute(
        "REVOKE SELECT ON public.alembic_version "
        "FROM eva_api, eva_worker, eva_agent_reader, eva_audit_writer, eva_backup"
    )
    op.execute(
        "REVOKE UPDATE ON jobs.inference_jobs FROM eva_api"
    )
