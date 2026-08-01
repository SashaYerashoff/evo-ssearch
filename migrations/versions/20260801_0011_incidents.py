"""Add durable incident records and incident operator permission.

Revision ID: 20260801_0011
Revises: 20260727_0010
Create Date: 2026-08-01
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260801_0011"
down_revision: str | None = "20260727_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _execute(sql: str) -> None:
    op.execute(sql)


def upgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        """
        CREATE TABLE archive.incidents (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            revision integer NOT NULL DEFAULT 1 CHECK (revision > 0),
            state text NOT NULL CHECK (
                state IN (
                    'candidate', 'draft', 'following',
                    'ended', 'reported', 'closed'
                )
            ),
            title text NOT NULL,
            channel_ids bigint[] NOT NULL,
            possible_start_ms bigint NOT NULL CHECK (possible_start_ms >= 0),
            observed_start_ms bigint CHECK (
                observed_start_ms IS NULL OR observed_start_ms >= possible_start_ms
            ),
            observed_end_ms bigint CHECK (
                observed_end_ms IS NULL
                OR observed_end_ms >= COALESCE(observed_start_ms, possible_start_ms)
            ),
            possible_end_ms bigint CHECK (
                possible_end_ms IS NULL
                OR possible_end_ms >= COALESCE(observed_end_ms, observed_start_ms, possible_start_ms)
            ),
            anchor_ref jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(anchor_ref) = 'object'),
            timeline_refs jsonb NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(timeline_refs) = 'array'),
            evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(evidence_refs) = 'array'),
            qualia_refs jsonb NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(qualia_refs) = 'array'),
            coverage_json jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(coverage_json) = 'object'),
            uncertainties_json jsonb NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(uncertainties_json) = 'array'),
            report_json jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(report_json) = 'object'),
            follow_policy_json jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(follow_policy_json) = 'object'),
            created_by uuid NOT NULL,
            updated_by uuid NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            CHECK (title = btrim(title) AND title <> ''),
            CHECK (length(title) <= 200),
            CHECK (cardinality(channel_ids) BETWEEN 1 AND 32),
            CHECK (array_position(channel_ids, NULL) IS NULL),
            CHECK (0 < ALL(channel_ids)),
            CHECK (octet_length(anchor_ref::text) <= 262144),
            CHECK (octet_length(timeline_refs::text) <= 262144),
            CHECK (octet_length(evidence_refs::text) <= 262144),
            CHECK (octet_length(qualia_refs::text) <= 262144),
            CHECK (octet_length(coverage_json::text) <= 262144),
            CHECK (octet_length(uncertainties_json::text) <= 262144),
            CHECK (octet_length(report_json::text) <= 262144),
            CHECK (octet_length(follow_policy_json::text) <= 262144)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incidents_channels
        ON archive.incidents USING gin (channel_ids)
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incidents_time
        ON archive.incidents (
            tenant_id,
            possible_start_ms DESC,
            (COALESCE(possible_end_ms, observed_end_ms, observed_start_ms, possible_start_ms)) DESC
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incidents_state_updated
        ON archive.incidents (tenant_id, state, updated_at DESC)
        """
    )
    _execute("ALTER TABLE archive.incidents ENABLE ROW LEVEL SECURITY")
    _execute("ALTER TABLE archive.incidents FORCE ROW LEVEL SECURITY")
    _execute(
        """
        CREATE POLICY archive_incidents_tenant_isolation
        ON archive.incidents
        USING (
            tenant_id = NULLIF(current_setting('eva.tenant_id', true), '')::uuid
        )
        WITH CHECK (
            tenant_id = NULLIF(current_setting('eva.tenant_id', true), '')::uuid
        )
        """
    )
    _execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON archive.incidents "
        "TO eva_api, eva_worker"
    )
    _execute(
        "GRANT SELECT ON archive.incidents TO eva_agent_reader, eva_backup"
    )

    _execute(
        """
        INSERT INTO iam.permissions (key, description, risk)
        VALUES ('incidents:manage', 'incidents manage', 'write')
        ON CONFLICT (key) DO UPDATE
        SET description = EXCLUDED.description,
            risk = EXCLUDED.risk
        """
    )
    _execute(
        """
        INSERT INTO iam.role_permissions (
            tenant_id, role_id, permission_key, assigned_by
        )
        SELECT role.tenant_id, role.id, 'incidents:manage', NULL
        FROM iam.roles AS role
        WHERE role.name IN ('admin', 'engineer', 'operator')
        ON CONFLICT (tenant_id, role_id, permission_key) DO NOTHING
        """
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        """
        DELETE FROM iam.role_permissions
        WHERE permission_key = 'incidents:manage'
        """
    )
    _execute(
        "DELETE FROM iam.permissions WHERE key = 'incidents:manage'"
    )
    _execute("DROP TABLE IF EXISTS archive.incidents")
    _execute("RESET ROLE")
