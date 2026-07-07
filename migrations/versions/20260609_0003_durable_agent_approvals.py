"""Harden durable agent action approvals.

Revision ID: 20260609_0003
Revises: 20260609_0002
Create Date: 2026-06-09
"""

from __future__ import annotations

from alembic import op

revision: str = "20260609_0003"
down_revision: str | None = "20260609_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agent.action_plans
        ADD CONSTRAINT ck_agent_action_plans_arguments_hash_sha256
        CHECK (octet_length(arguments_hash) = 32)
        """
    )
    op.execute(
        """
        ALTER TABLE agent.action_approvals
        ADD CONSTRAINT ck_agent_action_approvals_plan_hash_sha256
        CHECK (octet_length(plan_arguments_hash) = 32)
        """
    )
    op.execute(
        """
        ALTER TABLE agent.action_approvals
        ADD CONSTRAINT ck_agent_action_approvals_token_hash_sha256
        CHECK (octet_length(approval_token_hash) = 32)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX ux_agent_action_approvals_one_active_per_plan
        ON agent.action_approvals (tenant_id, plan_id)
        WHERE status = 'active'
        """
    )
    op.execute(
        """
        CREATE INDEX ix_agent_action_plans_session_pending
        ON agent.action_plans (tenant_id, session_id, expires_at)
        WHERE status IN ('pending', 'approved')
        """
    )
    op.execute(
        """
        CREATE INDEX ix_agent_action_plans_expiring
        ON agent.action_plans (tenant_id, expires_at)
        WHERE status IN ('pending', 'approved')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent.ix_agent_action_plans_expiring")
    op.execute("DROP INDEX IF EXISTS agent.ix_agent_action_plans_session_pending")
    op.execute(
        "DROP INDEX IF EXISTS agent.ux_agent_action_approvals_one_active_per_plan"
    )
    op.execute(
        """
        ALTER TABLE agent.action_approvals
        DROP CONSTRAINT IF EXISTS ck_agent_action_approvals_token_hash_sha256
        """
    )
    op.execute(
        """
        ALTER TABLE agent.action_approvals
        DROP CONSTRAINT IF EXISTS ck_agent_action_approvals_plan_hash_sha256
        """
    )
    op.execute(
        """
        ALTER TABLE agent.action_plans
        DROP CONSTRAINT IF EXISTS ck_agent_action_plans_arguments_hash_sha256
        """
    )
