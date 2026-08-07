"""Add durable incident lifecycle axes and append-only temporal ledgers.

Revision ID: 20260805_0012
Revises: 20260801_0011
Create Date: 2026-08-05

The legacy ``archive.incidents.state`` column remains the compatibility
projection used by the v0 API.  Existing records are deliberately backfilled
to ``unknown`` on every new lifecycle axis: a migration cannot infer risk,
perception, case, or attention semantics from the old combined state.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260805_0012"
down_revision: str | None = "20260801_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_PERCEPTION_VALUES = "'unknown', 'observed', 'not_observed', 'ended'"
_RISK_VALUES = "'unknown', 'active', 'contained', 'resolved', 'occurred'"
_CASE_VALUES = (
    "'unknown', 'candidate', 'open', 'closed', 'dismissed', 'false_positive'"
)
_ATTENTION_VALUES = "'unknown', 'inactive', 'follow', 'critical'"
_LEGACY_VALUES = "'candidate', 'draft', 'following', 'ended', 'reported', 'closed'"


def _execute(sql: str) -> None:
    op.execute(sql)


def _nullable_key_check(column: str, maximum: int = 200) -> str:
    return (
        f"{column} IS NULL OR ("
        f"{column} = btrim({column}) AND {column} <> '' "
        f"AND length({column}) <= {maximum})"
    )


def _tenant_policy(table: str) -> None:
    _execute(f"ALTER TABLE archive.{table} ENABLE ROW LEVEL SECURITY")
    _execute(f"ALTER TABLE archive.{table} FORCE ROW LEVEL SECURITY")
    _execute(
        f"""
        CREATE POLICY archive_{table}_tenant_isolation
        ON archive.{table}
        USING (
            tenant_id = NULLIF(current_setting('eva.tenant_id', true), '')::uuid
        )
        WITH CHECK (
            tenant_id = NULLIF(current_setting('eva.tenant_id', true), '')::uuid
        )
        """
    )


def upgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        f"""
        ALTER TABLE archive.incidents
            ADD COLUMN perception_state text NOT NULL DEFAULT 'unknown'
                CONSTRAINT archive_incidents_perception_state_check
                CHECK (perception_state IN ({_PERCEPTION_VALUES})),
            ADD COLUMN risk_state text NOT NULL DEFAULT 'unknown'
                CONSTRAINT archive_incidents_risk_state_check
                CHECK (risk_state IN ({_RISK_VALUES})),
            ADD COLUMN case_state text NOT NULL DEFAULT 'unknown'
                CONSTRAINT archive_incidents_case_state_check
                CHECK (case_state IN ({_CASE_VALUES})),
            ADD COLUMN attention_state text NOT NULL DEFAULT 'unknown'
                CONSTRAINT archive_incidents_attention_state_check
                CHECK (attention_state IN ({_ATTENTION_VALUES})),
            ADD COLUMN identity_key text,
            ADD COLUMN idempotency_key text,
            ADD CONSTRAINT archive_incidents_identity_key_check
                CHECK ({_nullable_key_check('identity_key')}),
            ADD CONSTRAINT archive_incidents_idempotency_key_check
                CHECK ({_nullable_key_check('idempotency_key')})
        """
    )
    _execute(
        """
        CREATE UNIQUE INDEX ux_archive_incidents_identity_key
        ON archive.incidents (tenant_id, identity_key)
        WHERE identity_key IS NOT NULL
        """
    )
    _execute(
        """
        CREATE UNIQUE INDEX ux_archive_incidents_idempotency_key
        ON archive.incidents (tenant_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incidents_lifecycle_updated
        ON archive.incidents (
            tenant_id, case_state, risk_state, perception_state, updated_at DESC
        )
        """
    )

    _execute(
        f"""
        CREATE TABLE archive.incident_observations (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            incident_id uuid NOT NULL,
            idempotency_key text NOT NULL,
            source_kind text NOT NULL,
            observed_at_ms bigint NOT NULL CHECK (observed_at_ms >= 0),
            channel_id bigint CHECK (channel_id IS NULL OR channel_id > 0),
            perception_state text NOT NULL DEFAULT 'unknown'
                CHECK (perception_state IN ({_PERCEPTION_VALUES})),
            source_ref jsonb NOT NULL DEFAULT '{{}}'::jsonb
                CHECK (jsonb_typeof(source_ref) = 'object'),
            payload_json jsonb NOT NULL DEFAULT '{{}}'::jsonb
                CHECK (jsonb_typeof(payload_json) = 'object'),
            created_by uuid NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            FOREIGN KEY (tenant_id, incident_id)
                REFERENCES archive.incidents (tenant_id, id) ON DELETE RESTRICT,
            UNIQUE (tenant_id, incident_id, idempotency_key),
            CHECK (idempotency_key = btrim(idempotency_key) AND idempotency_key <> ''),
            CHECK (length(idempotency_key) <= 200),
            CHECK (source_kind = btrim(source_kind) AND source_kind <> ''),
            CHECK (length(source_kind) <= 80),
            CHECK (octet_length(source_ref::text) <= 262144),
            CHECK (octet_length(payload_json::text) <= 262144)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incident_observations_time
        ON archive.incident_observations (
            tenant_id, incident_id, observed_at_ms ASC, id ASC
        )
        """
    )

    _execute(
        f"""
        CREATE TABLE archive.incident_episodes (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            incident_id uuid NOT NULL,
            idempotency_key text NOT NULL,
            episode_key text NOT NULL,
            perception_state text NOT NULL DEFAULT 'unknown'
                CHECK (perception_state IN ({_PERCEPTION_VALUES})),
            semantic_key text,
            entity_key text,
            zone_key text,
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
                OR possible_end_ms >= COALESCE(
                    observed_end_ms, observed_start_ms, possible_start_ms
                )
            ),
            routine_before_ref jsonb NOT NULL DEFAULT '{{}}'::jsonb
                CHECK (jsonb_typeof(routine_before_ref) = 'object'),
            routine_after_ref jsonb NOT NULL DEFAULT '{{}}'::jsonb
                CHECK (jsonb_typeof(routine_after_ref) = 'object'),
            evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb
                CHECK (jsonb_typeof(evidence_refs) = 'array'),
            coverage_json jsonb NOT NULL DEFAULT '{{}}'::jsonb
                CHECK (jsonb_typeof(coverage_json) = 'object'),
            created_by uuid NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            FOREIGN KEY (tenant_id, incident_id)
                REFERENCES archive.incidents (tenant_id, id) ON DELETE RESTRICT,
            UNIQUE (tenant_id, incident_id, idempotency_key),
            UNIQUE (tenant_id, incident_id, episode_key),
            CHECK (idempotency_key = btrim(idempotency_key) AND idempotency_key <> ''),
            CHECK (length(idempotency_key) <= 200),
            CHECK (episode_key = btrim(episode_key) AND episode_key <> ''),
            CHECK (length(episode_key) <= 200),
            CHECK ({_nullable_key_check('semantic_key', 160)}),
            CHECK ({_nullable_key_check('entity_key', 160)}),
            CHECK ({_nullable_key_check('zone_key', 160)}),
            CHECK (octet_length(routine_before_ref::text) <= 262144),
            CHECK (octet_length(routine_after_ref::text) <= 262144),
            CHECK (octet_length(evidence_refs::text) <= 262144),
            CHECK (octet_length(coverage_json::text) <= 262144)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incident_episodes_time
        ON archive.incident_episodes (
            tenant_id, incident_id, possible_start_ms ASC, id ASC
        )
        """
    )

    _execute(
        """
        CREATE TABLE archive.incident_relations (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            subject_incident_id uuid NOT NULL,
            object_incident_id uuid NOT NULL,
            idempotency_key text NOT NULL,
            relation_type text NOT NULL CHECK (
                relation_type IN (
                    'series_member', 'caused_by', 'concurrent_with',
                    'possible_same_as', 'merged_into', 'split_from', 'supersedes'
                )
            ),
            relation_state text NOT NULL DEFAULT 'candidate' CHECK (
                relation_state IN ('candidate', 'confirmed', 'rejected')
            ),
            confidence text NOT NULL DEFAULT 'unknown' CHECK (
                confidence IN ('unknown', 'low', 'medium', 'high')
            ),
            rationale text NOT NULL DEFAULT '',
            payload_json jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(payload_json) = 'object'),
            created_by uuid NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            FOREIGN KEY (tenant_id, subject_incident_id)
                REFERENCES archive.incidents (tenant_id, id) ON DELETE RESTRICT,
            FOREIGN KEY (tenant_id, object_incident_id)
                REFERENCES archive.incidents (tenant_id, id) ON DELETE RESTRICT,
            UNIQUE (tenant_id, subject_incident_id, idempotency_key),
            CHECK (subject_incident_id <> object_incident_id),
            CHECK (idempotency_key = btrim(idempotency_key) AND idempotency_key <> ''),
            CHECK (length(idempotency_key) <= 200),
            CHECK (rationale = btrim(rationale)),
            CHECK (length(rationale) <= 2000),
            CHECK (octet_length(payload_json::text) <= 262144)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incident_relations_subject
        ON archive.incident_relations (
            tenant_id, subject_incident_id, relation_type, created_at DESC
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incident_relations_object
        ON archive.incident_relations (
            tenant_id, object_incident_id, relation_type, created_at DESC
        )
        """
    )

    _execute(
        f"""
        CREATE TABLE archive.incident_transitions (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            incident_id uuid NOT NULL,
            idempotency_key text NOT NULL,
            axis text NOT NULL CHECK (
                axis IN ('perception', 'risk', 'case', 'attention', 'legacy')
            ),
            from_state text,
            to_state text NOT NULL,
            incident_revision integer NOT NULL CHECK (incident_revision > 0),
            transitioned_at_ms bigint NOT NULL CHECK (transitioned_at_ms >= 0),
            reason text NOT NULL DEFAULT '',
            source_kind text NOT NULL,
            source_ref jsonb NOT NULL DEFAULT '{{}}'::jsonb
                CHECK (jsonb_typeof(source_ref) = 'object'),
            payload_json jsonb NOT NULL DEFAULT '{{}}'::jsonb
                CHECK (jsonb_typeof(payload_json) = 'object'),
            created_by uuid NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            FOREIGN KEY (tenant_id, incident_id)
                REFERENCES archive.incidents (tenant_id, id) ON DELETE RESTRICT,
            UNIQUE (tenant_id, incident_id, idempotency_key),
            CHECK (idempotency_key = btrim(idempotency_key) AND idempotency_key <> ''),
            CHECK (length(idempotency_key) <= 200),
            CHECK (source_kind = btrim(source_kind) AND source_kind <> ''),
            CHECK (length(source_kind) <= 80),
            CHECK (reason = btrim(reason)),
            CHECK (length(reason) <= 2000),
            CHECK (octet_length(source_ref::text) <= 262144),
            CHECK (octet_length(payload_json::text) <= 262144),
            CHECK (
                (axis = 'perception' AND to_state IN ({_PERCEPTION_VALUES}))
                OR (axis = 'risk' AND to_state IN ({_RISK_VALUES}))
                OR (axis = 'case' AND to_state IN ({_CASE_VALUES}))
                OR (axis = 'attention' AND to_state IN ({_ATTENTION_VALUES}))
                OR (axis = 'legacy' AND to_state IN ({_LEGACY_VALUES}))
            ),
            CHECK (
                from_state IS NULL
                OR (axis = 'perception' AND from_state IN ({_PERCEPTION_VALUES}))
                OR (axis = 'risk' AND from_state IN ({_RISK_VALUES}))
                OR (axis = 'case' AND from_state IN ({_CASE_VALUES}))
                OR (axis = 'attention' AND from_state IN ({_ATTENTION_VALUES}))
                OR (axis = 'legacy' AND from_state IN ({_LEGACY_VALUES}))
            )
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_incident_transitions_time
        ON archive.incident_transitions (
            tenant_id, incident_id, transitioned_at_ms ASC, id ASC
        )
        """
    )

    for table in (
        "incident_observations",
        "incident_episodes",
        "incident_relations",
        "incident_transitions",
    ):
        _tenant_policy(table)

    # These four ledgers are append-only for runtime identities.  Corrections
    # are represented by a later observation/relation/transition, never an
    # in-place rewrite or deletion.
    _execute(
        "GRANT SELECT, INSERT ON "
        "archive.incident_observations, archive.incident_episodes, "
        "archive.incident_relations, archive.incident_transitions "
        "TO eva_api, eva_worker"
    )
    _execute(
        "GRANT SELECT ON "
        "archive.incident_observations, archive.incident_episodes, "
        "archive.incident_relations, archive.incident_transitions "
        "TO eva_agent_reader, eva_backup"
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    for table in (
        "incident_transitions",
        "incident_relations",
        "incident_episodes",
        "incident_observations",
    ):
        _execute(f"DROP TABLE IF EXISTS archive.{table}")
    _execute(
        """
        ALTER TABLE archive.incidents
            DROP COLUMN IF EXISTS idempotency_key,
            DROP COLUMN IF EXISTS identity_key,
            DROP COLUMN IF EXISTS attention_state,
            DROP COLUMN IF EXISTS case_state,
            DROP COLUMN IF EXISTS risk_state,
            DROP COLUMN IF EXISTS perception_state
        """
    )
    _execute("RESET ROLE")
