"""Add compact attention telemetry and scheduler audit storage.

Revision ID: 20260726_0008
Revises: 20260725_0007
Create Date: 2026-07-26
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260726_0008"
down_revision: str | None = "20260725_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLES = (
    "attention_embedding_snapshots",
    "attention_probe_scores",
    "attention_intervals",
    "attention_interval_links",
    "attention_episodes",
    "attention_scheduler_decisions",
    "attention_probe_lineage",
)


def _execute(sql: str) -> None:
    op.execute(sql)


def upgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    _execute(
        """
        CREATE TABLE archive.attention_embedding_snapshots (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            channel_id bigint NOT NULL CHECK (channel_id > 0),
            captured_at_ms bigint NOT NULL CHECK (captured_at_ms >= 0),
            embedding_ref text NOT NULL,
            embedding_model text NOT NULL,
            frame_ref text,
            cadence_ms integer CHECK (
                cadence_ms IS NULL OR cadence_ms BETWEEN 1 AND 3600000
            ),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            CHECK (embedding_ref = btrim(embedding_ref) AND embedding_ref <> ''),
            CHECK (length(embedding_ref) <= 1024),
            CHECK (embedding_model = btrim(embedding_model) AND embedding_model <> ''),
            CHECK (length(embedding_model) <= 160),
            CHECK (frame_ref IS NULL OR length(frame_ref) <= 1024)
        )
        """
    )
    _execute(
        """
        CREATE UNIQUE INDEX ux_archive_attention_snapshot_cadence
        ON archive.attention_embedding_snapshots (
            tenant_id, channel_id, captured_at_ms, embedding_model
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_snapshot_channel_time
        ON archive.attention_embedding_snapshots (
            tenant_id, channel_id, captured_at_ms DESC
        )
        """
    )

    _execute(
        """
        CREATE TABLE archive.attention_probe_scores (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            embedding_snapshot_id uuid NOT NULL,
            scored_at_ms bigint NOT NULL CHECK (scored_at_ms >= 0),
            probe_id text NOT NULL,
            probe_version text NOT NULL,
            pos_score double precision NOT NULL CHECK (pos_score BETWEEN -1 AND 1),
            neg_score double precision NOT NULL CHECK (neg_score BETWEEN -1 AND 1),
            margin double precision NOT NULL CHECK (margin BETWEEN -2 AND 2),
            pos_floor double precision CHECK (
                pos_floor IS NULL OR pos_floor BETWEEN 0 AND 1
            ),
            margin_threshold double precision CHECK (
                margin_threshold IS NULL OR margin_threshold BETWEEN 0 AND 2
            ),
            threshold_state text NOT NULL CHECK (
                threshold_state IN (
                    'hit', 'below_pos', 'below_margin', 'below_both',
                    'not_evaluated', 'suppressed'
                )
            ),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            CONSTRAINT fk_attention_probe_score_snapshot
                FOREIGN KEY (tenant_id, embedding_snapshot_id)
                REFERENCES archive.attention_embedding_snapshots (tenant_id, id)
                ON DELETE CASCADE,
            CHECK (probe_id = btrim(probe_id) AND probe_id <> ''),
            CHECK (length(probe_id) <= 160),
            CHECK (probe_version = btrim(probe_version) AND probe_version <> ''),
            CHECK (length(probe_version) <= 160),
            CHECK (
                threshold_state = 'not_evaluated'
                OR (pos_floor IS NOT NULL AND margin_threshold IS NOT NULL)
            )
        )
        """
    )
    _execute(
        """
        CREATE UNIQUE INDEX ux_archive_attention_probe_score_snapshot
        ON archive.attention_probe_scores (
            tenant_id, embedding_snapshot_id, probe_id, probe_version
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_probe_score_probe_time
        ON archive.attention_probe_scores (
            tenant_id, probe_id, scored_at_ms DESC
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_probe_score_state_time
        ON archive.attention_probe_scores (
            tenant_id, threshold_state, scored_at_ms DESC
        )
        """
    )

    _execute(
        """
        CREATE TABLE archive.attention_intervals (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            channel_id bigint NOT NULL CHECK (channel_id > 0),
            started_at_ms bigint NOT NULL CHECK (started_at_ms >= 0),
            ended_at_ms bigint NOT NULL CHECK (ended_at_ms >= started_at_ms),
            state text NOT NULL CHECK (
                state IN ('quiet', 'motion', 'mixed', 'unknown', 'degraded')
            ),
            sample_count integer NOT NULL CHECK (sample_count > 0),
            expected_sample_count integer CHECK (
                expected_sample_count IS NULL OR expected_sample_count > 0
            ),
            motion_mean double precision NOT NULL CHECK (motion_mean >= 0),
            motion_max double precision NOT NULL CHECK (motion_max >= 0),
            motion_p95 double precision NOT NULL CHECK (motion_p95 >= 0),
            motion_integral double precision NOT NULL CHECK (motion_integral >= 0),
            moving_fraction double precision NOT NULL CHECK (
                moving_fraction BETWEEN 0 AND 1
            ),
            quiet_fraction double precision NOT NULL CHECK (
                quiet_fraction BETWEEN 0 AND 1
            ),
            activity_x_max double precision NOT NULL CHECK (activity_x_max >= 0),
            peak_at_ms bigint,
            baseline_ref text,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            CHECK (moving_fraction + quiet_fraction <= 1.000001),
            CHECK (
                peak_at_ms IS NULL
                OR peak_at_ms BETWEEN started_at_ms AND ended_at_ms
            ),
            CHECK (baseline_ref IS NULL OR length(baseline_ref) <= 256)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_interval_channel_time
        ON archive.attention_intervals (
            tenant_id, channel_id, started_at_ms DESC, ended_at_ms DESC
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_interval_state_time
        ON archive.attention_intervals (
            tenant_id, state, ended_at_ms DESC
        )
        """
    )

    _execute(
        """
        CREATE TABLE archive.attention_interval_links (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            interval_id uuid NOT NULL,
            occurred_at_ms bigint NOT NULL CHECK (occurred_at_ms >= 0),
            kind text NOT NULL CHECK (kind IN ('embedding', 'vlm_apex')),
            role text NOT NULL CHECK (
                role IN (
                    'support', 'control', 'pre', 'onset', 'apex',
                    'post', 'companion'
                )
            ),
            embedding_snapshot_id uuid,
            apex_ref text,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            CONSTRAINT fk_attention_link_interval
                FOREIGN KEY (tenant_id, interval_id)
                REFERENCES archive.attention_intervals (tenant_id, id)
                ON DELETE CASCADE,
            CONSTRAINT fk_attention_link_snapshot
                FOREIGN KEY (tenant_id, embedding_snapshot_id)
                REFERENCES archive.attention_embedding_snapshots (tenant_id, id)
                ON DELETE CASCADE,
            CHECK (
                (
                    kind = 'embedding'
                    AND embedding_snapshot_id IS NOT NULL
                    AND apex_ref IS NULL
                )
                OR (
                    kind = 'vlm_apex'
                    AND embedding_snapshot_id IS NULL
                    AND apex_ref IS NOT NULL
                    AND apex_ref = btrim(apex_ref)
                    AND apex_ref <> ''
                )
            ),
            CHECK (apex_ref IS NULL OR length(apex_ref) <= 1024)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_link_interval_time
        ON archive.attention_interval_links (
            tenant_id, interval_id, occurred_at_ms ASC
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_link_snapshot
        ON archive.attention_interval_links (
            tenant_id, embedding_snapshot_id
        )
        WHERE embedding_snapshot_id IS NOT NULL
        """
    )

    _execute(
        """
        CREATE TABLE archive.attention_episodes (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            channel_id bigint NOT NULL CHECK (channel_id > 0),
            started_at_ms bigint NOT NULL CHECK (started_at_ms >= 0),
            ended_at_ms bigint NOT NULL CHECK (ended_at_ms >= started_at_ms),
            trigger text NOT NULL,
            status text NOT NULL,
            record_json jsonb NOT NULL CHECK (jsonb_typeof(record_json) = 'object'),
            canonical_json text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            CHECK (trigger = btrim(trigger) AND trigger <> ''),
            CHECK (length(trigger) <= 80),
            CHECK (status = btrim(status) AND status <> ''),
            CHECK (length(status) <= 40),
            CHECK (length(canonical_json) <= 262144),
            CHECK (canonical_json::jsonb = record_json)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_episode_channel_time
        ON archive.attention_episodes (
            tenant_id, channel_id, started_at_ms DESC, ended_at_ms DESC
        )
        """
    )

    _execute(
        """
        CREATE TABLE archive.attention_scheduler_decisions (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            channel_id bigint CHECK (channel_id IS NULL OR channel_id > 0),
            episode_id uuid,
            decided_at_ms bigint NOT NULL CHECK (decided_at_ms >= 0),
            action text NOT NULL,
            record_json jsonb NOT NULL CHECK (jsonb_typeof(record_json) = 'object'),
            canonical_json text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            CONSTRAINT fk_attention_decision_episode
                FOREIGN KEY (tenant_id, episode_id)
                REFERENCES archive.attention_episodes (tenant_id, id)
                ON DELETE SET NULL (episode_id),
            CHECK (action = btrim(action) AND action <> ''),
            CHECK (length(action) <= 80),
            CHECK (length(canonical_json) <= 262144),
            CHECK (canonical_json::jsonb = record_json)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_decision_channel_time
        ON archive.attention_scheduler_decisions (
            tenant_id, channel_id, decided_at_ms DESC
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_decision_episode
        ON archive.attention_scheduler_decisions (tenant_id, episode_id)
        WHERE episode_id IS NOT NULL
        """
    )

    _execute(
        """
        CREATE TABLE archive.attention_probe_lineage (
            tenant_id uuid NOT NULL,
            id uuid NOT NULL,
            probe_id text NOT NULL,
            channel_id bigint NOT NULL CHECK (channel_id > 0),
            created_at_ms bigint NOT NULL CHECK (created_at_ms >= 0),
            expires_at_ms bigint,
            lifecycle_state text NOT NULL CHECK (
                lifecycle_state IN (
                    'created', 'active', 'expired', 'retired',
                    'promoted', 'rejected'
                )
            ),
            parent_alert_ref text,
            parent_probe_id text,
            record_json jsonb NOT NULL CHECK (jsonb_typeof(record_json) = 'object'),
            canonical_json text NOT NULL,
            recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, id),
            CHECK (probe_id = btrim(probe_id) AND probe_id <> ''),
            CHECK (length(probe_id) <= 160),
            CHECK (expires_at_ms IS NULL OR expires_at_ms >= created_at_ms),
            CHECK (parent_alert_ref IS NULL OR length(parent_alert_ref) <= 1024),
            CHECK (parent_probe_id IS NULL OR length(parent_probe_id) <= 160),
            CHECK (length(canonical_json) <= 262144),
            CHECK (canonical_json::jsonb = record_json)
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_probe_channel_time
        ON archive.attention_probe_lineage (
            tenant_id, channel_id, created_at_ms DESC
        )
        """
    )
    _execute(
        """
        CREATE INDEX ix_archive_attention_probe_id_time
        ON archive.attention_probe_lineage (
            tenant_id, probe_id, created_at_ms DESC
        )
        """
    )

    for table in TABLES:
        qualified = f"archive.{table}"
        policy = f"archive_{table}_tenant_isolation"
        _execute(f"ALTER TABLE {qualified} ENABLE ROW LEVEL SECURITY")
        _execute(f"ALTER TABLE {qualified} FORCE ROW LEVEL SECURITY")
        _execute(
            f"""
            CREATE POLICY {policy} ON {qualified}
            USING (
                tenant_id = NULLIF(current_setting('eva.tenant_id', true), '')::uuid
            )
            WITH CHECK (
                tenant_id = NULLIF(current_setting('eva.tenant_id', true), '')::uuid
            )
            """
        )

    _execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON "
        + ", ".join(f"archive.{table}" for table in TABLES)
        + " TO eva_api, eva_worker"
    )
    _execute(
        "GRANT SELECT ON "
        + ", ".join(f"archive.{table}" for table in TABLES)
        + " TO eva_agent_reader, eva_backup"
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    _execute("SET LOCAL ROLE eva_owner")
    for table in reversed(TABLES):
        _execute(f"DROP TABLE IF EXISTS archive.{table}")
    _execute("RESET ROLE")
