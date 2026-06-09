"""Create the secure PostgreSQL control-plane foundation.

Revision ID: 20260609_0001
Revises:
Create Date: 2026-06-09

Cluster roles are retained on downgrade because they can be shared with
deployment-managed login principals and backup tooling outside these schemas.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260609_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME_ROLES = (
    "eva_owner",
    "eva_migrator",
    "eva_api",
    "eva_worker",
    "eva_agent_reader",
    "eva_audit_writer",
    "eva_backup",
)

SCHEMAS = ("iam", "agent", "audit", "jobs")

TENANT_TABLES = (
    ("iam", "users"),
    ("iam", "sessions"),
    ("iam", "roles"),
    ("iam", "user_roles"),
    ("iam", "role_permissions"),
    ("iam", "user_channel_grants"),
    ("agent", "sessions"),
    ("agent", "messages"),
    ("agent", "tool_runs"),
    ("agent", "action_plans"),
    ("agent", "action_approvals"),
    ("audit", "events"),
    ("jobs", "inference_jobs"),
    ("jobs", "job_attempts"),
    ("jobs", "outbox"),
)


def _execute(sql: str) -> None:
    op.execute(sql)


def upgrade() -> None:
    _execute(
        """
        DO $roles$
        DECLARE
            role_name text;
        BEGIN
            FOREACH role_name IN ARRAY ARRAY[
                'eva_owner',
                'eva_migrator',
                'eva_api',
                'eva_worker',
                'eva_agent_reader',
                'eva_audit_writer',
                'eva_backup'
            ]
            LOOP
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
                    EXECUTE format(
                        'CREATE ROLE %I NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS',
                        role_name
                    );
                END IF;
            END LOOP;
        END
        $roles$
        """
    )
    _execute("GRANT eva_owner TO eva_migrator")

    for schema in SCHEMAS:
        _execute(f"CREATE SCHEMA IF NOT EXISTS {schema} AUTHORIZATION eva_owner")
        _execute(f"REVOKE ALL ON SCHEMA {schema} FROM PUBLIC")

    _execute("SET LOCAL ROLE eva_owner")

    _execute(
        """
        CREATE TABLE iam.users (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            username text NOT NULL,
            password_hash text NOT NULL,
            display_name text,
            email text,
            is_active boolean NOT NULL DEFAULT true,
            failed_login_count integer NOT NULL DEFAULT 0
                CHECK (failed_login_count >= 0),
            locked_until timestamptz,
            last_login_at timestamptz,
            password_changed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            UNIQUE (tenant_id, id),
            CHECK (username = btrim(username) AND username <> ''),
            CHECK (password_hash <> '')
        )
        """
    )
    _execute(
        "CREATE UNIQUE INDEX uq_iam_users_tenant_username "
        "ON iam.users (tenant_id, lower(username))"
    )

    _execute(
        """
        CREATE TABLE iam.sessions (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            user_id uuid NOT NULL,
            token_hash bytea NOT NULL UNIQUE,
            csrf_token_hash bytea NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            last_seen_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            expires_at timestamptz NOT NULL,
            revoked_at timestamptz,
            revoke_reason text,
            client_ip inet,
            user_agent text,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, user_id)
                REFERENCES iam.users (tenant_id, id) ON DELETE CASCADE,
            CHECK (expires_at > created_at)
        )
        """
    )
    _execute(
        "CREATE INDEX ix_iam_sessions_active "
        "ON iam.sessions (tenant_id, user_id, expires_at) WHERE revoked_at IS NULL"
    )

    _execute(
        """
        CREATE TABLE iam.roles (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            name text NOT NULL,
            description text,
            is_system boolean NOT NULL DEFAULT false,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, name),
            CHECK (name = lower(name) AND name ~ '^[a-z][a-z0-9_.-]{1,62}$')
        )
        """
    )

    _execute(
        """
        CREATE TABLE iam.permissions (
            key text PRIMARY KEY,
            description text NOT NULL,
            risk text NOT NULL DEFAULT 'read'
                CHECK (risk IN ('read', 'write', 'external_side_effect')),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            CHECK (key ~ '^[a-z][a-z0-9_.:-]{2,127}$')
        )
        """
    )

    _execute(
        """
        CREATE TABLE iam.user_roles (
            tenant_id uuid NOT NULL,
            user_id uuid NOT NULL,
            role_id uuid NOT NULL,
            assigned_by uuid,
            assigned_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, user_id, role_id),
            FOREIGN KEY (tenant_id, user_id)
                REFERENCES iam.users (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, role_id)
                REFERENCES iam.roles (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, assigned_by)
                REFERENCES iam.users (tenant_id, id) ON DELETE SET NULL
        )
        """
    )

    _execute(
        """
        CREATE TABLE iam.role_permissions (
            tenant_id uuid NOT NULL,
            role_id uuid NOT NULL,
            permission_key text NOT NULL,
            assigned_by uuid,
            assigned_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, role_id, permission_key),
            FOREIGN KEY (tenant_id, role_id)
                REFERENCES iam.roles (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (permission_key)
                REFERENCES iam.permissions (key) ON DELETE RESTRICT,
            FOREIGN KEY (tenant_id, assigned_by)
                REFERENCES iam.users (tenant_id, id) ON DELETE SET NULL
        )
        """
    )

    _execute(
        """
        CREATE TABLE iam.user_channel_grants (
            tenant_id uuid NOT NULL,
            user_id uuid NOT NULL,
            channel_id bigint NOT NULL CHECK (channel_id > 0),
            access_level text NOT NULL DEFAULT 'view'
                CHECK (access_level IN ('view', 'operate', 'manage')),
            granted_by uuid,
            granted_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            expires_at timestamptz,
            PRIMARY KEY (tenant_id, user_id, channel_id),
            FOREIGN KEY (tenant_id, user_id)
                REFERENCES iam.users (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, granted_by)
                REFERENCES iam.users (tenant_id, id) ON DELETE SET NULL
        )
        """
    )
    _execute(
        "CREATE INDEX ix_iam_channel_grants_channel "
        "ON iam.user_channel_grants (tenant_id, channel_id, user_id)"
    )

    _execute(
        """
        CREATE TABLE agent.sessions (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            user_id uuid NOT NULL,
            title text,
            status text NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'archived', 'deleted')),
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            deleted_at timestamptz,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, user_id)
                REFERENCES iam.users (tenant_id, id) ON DELETE RESTRICT
        )
        """
    )
    _execute(
        "CREATE INDEX ix_agent_sessions_user_updated "
        "ON agent.sessions (tenant_id, user_id, updated_at DESC)"
    )

    _execute(
        """
        CREATE TABLE agent.messages (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            session_id uuid NOT NULL,
            sequence_number bigint NOT NULL CHECK (sequence_number >= 0),
            role text NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
            content text NOT NULL DEFAULT '',
            tool_call_id text,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, session_id, sequence_number),
            FOREIGN KEY (tenant_id, session_id)
                REFERENCES agent.sessions (tenant_id, id) ON DELETE CASCADE
        )
        """
    )

    _execute(
        """
        CREATE TABLE agent.tool_runs (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            session_id uuid NOT NULL,
            actor_user_id uuid NOT NULL,
            request_id text,
            tool_name text NOT NULL,
            normalized_arguments_hash bytea NOT NULL,
            required_permission text NOT NULL,
            permission_decision text NOT NULL
                CHECK (permission_decision IN ('allow', 'deny')),
            duration_ms integer CHECK (duration_ms >= 0),
            result_class text,
            audit_event_id uuid,
            safe_metadata jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(safe_metadata) = 'object'),
            started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            finished_at timestamptz,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, session_id)
                REFERENCES agent.sessions (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, actor_user_id)
                REFERENCES iam.users (tenant_id, id) ON DELETE RESTRICT
        )
        """
    )
    _execute(
        "CREATE INDEX ix_agent_tool_runs_session_started "
        "ON agent.tool_runs (tenant_id, session_id, started_at DESC)"
    )

    _execute(
        """
        CREATE TABLE agent.action_plans (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            session_id uuid NOT NULL,
            actor_user_id uuid NOT NULL,
            action text NOT NULL,
            required_permission text NOT NULL,
            normalized_arguments jsonb NOT NULL
                CHECK (jsonb_typeof(normalized_arguments) = 'object'),
            arguments_hash bytea NOT NULL,
            trusted_diff jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(trusted_diff) = 'object'),
            status text NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'approved', 'executed', 'expired', 'cancelled')),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            expires_at timestamptz NOT NULL,
            executed_at timestamptz,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, session_id)
                REFERENCES agent.sessions (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, actor_user_id)
                REFERENCES iam.users (tenant_id, id) ON DELETE RESTRICT,
            CHECK (expires_at > created_at)
        )
        """
    )
    _execute(
        "CREATE INDEX ix_agent_action_plans_pending "
        "ON agent.action_plans (tenant_id, actor_user_id, expires_at) "
        "WHERE status = 'pending'"
    )

    _execute(
        """
        CREATE TABLE agent.action_approvals (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            plan_id uuid NOT NULL,
            approved_by uuid NOT NULL,
            plan_arguments_hash bytea NOT NULL,
            approval_token_hash bytea NOT NULL UNIQUE,
            status text NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'consumed', 'expired', 'revoked')),
            approved_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            expires_at timestamptz NOT NULL,
            consumed_at timestamptz,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, plan_id)
                REFERENCES agent.action_plans (tenant_id, id) ON DELETE CASCADE,
            FOREIGN KEY (tenant_id, approved_by)
                REFERENCES iam.users (tenant_id, id) ON DELETE RESTRICT,
            CHECK (expires_at > approved_at),
            CHECK ((status = 'consumed') = (consumed_at IS NOT NULL))
        )
        """
    )
    _execute(
        "CREATE INDEX ix_agent_action_approvals_active "
        "ON agent.action_approvals (tenant_id, plan_id, expires_at) "
        "WHERE status = 'active'"
    )

    _execute(
        """
        CREATE TABLE audit.events (
            id uuid PRIMARY KEY,
            sequence_number bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
            tenant_id uuid NOT NULL,
            occurred_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            request_id text,
            actor_user_id uuid,
            actor_roles text[] NOT NULL DEFAULT ARRAY[]::text[],
            source_ip inet,
            action text NOT NULL,
            target_type text,
            target_id text,
            channel_id bigint CHECK (channel_id > 0),
            result text NOT NULL CHECK (result IN ('success', 'failure', 'denied')),
            safe_details jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(safe_details) = 'object'),
            previous_event_hash bytea,
            event_hash bytea,
            UNIQUE (tenant_id, id)
        )
        """
    )
    _execute(
        "CREATE INDEX ix_audit_events_tenant_time "
        "ON audit.events (tenant_id, occurred_at DESC, sequence_number DESC)"
    )
    _execute(
        "CREATE INDEX ix_audit_events_actor_time "
        "ON audit.events (tenant_id, actor_user_id, occurred_at DESC)"
    )
    _execute(
        """
        CREATE FUNCTION audit.reject_event_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        BEGIN
            RAISE EXCEPTION 'audit events are append-only'
                USING ERRCODE = '55000';
        END
        $function$
        """
    )
    _execute(
        """
        CREATE TRIGGER audit_events_append_only
        BEFORE UPDATE OR DELETE ON audit.events
        FOR EACH ROW EXECUTE FUNCTION audit.reject_event_mutation()
        """
    )
    _execute(
        """
        ALTER TABLE agent.tool_runs
        ADD CONSTRAINT fk_agent_tool_runs_audit_event
        FOREIGN KEY (tenant_id, audit_event_id)
        REFERENCES audit.events (tenant_id, id) ON DELETE RESTRICT
        """
    )

    _execute(
        """
        CREATE TABLE jobs.inference_jobs (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            channel_id bigint NOT NULL CHECK (channel_id > 0),
            workload_class text NOT NULL
                CHECK (workload_class IN ('heartbeat', 'event', 'manual')),
            model_id text NOT NULL,
            prompt_version_id uuid,
            media_object_ids uuid[] NOT NULL DEFAULT ARRAY[]::uuid[],
            priority smallint NOT NULL DEFAULT 100
                CHECK (priority BETWEEN 0 AND 1000),
            deadline_at timestamptz,
            state text NOT NULL DEFAULT 'queued'
                CHECK (state IN (
                    'queued',
                    'leased',
                    'succeeded',
                    'dead_letter',
                    'dropped'
                )),
            attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 100),
            lease_owner text,
            lease_expires_at timestamptz,
            idempotency_key text NOT NULL,
            payload jsonb NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(payload) = 'object'),
            result_metadata jsonb
                CHECK (result_metadata IS NULL OR jsonb_typeof(result_metadata) = 'object'),
            error_metadata jsonb
                CHECK (error_metadata IS NULL OR jsonb_typeof(error_metadata) = 'object'),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            started_at timestamptz,
            finished_at timestamptz,
            updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, idempotency_key),
            CHECK (
                (state = 'leased' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
                OR state <> 'leased'
            )
        )
        """
    )
    _execute(
        "CREATE INDEX ix_jobs_inference_claim "
        "ON jobs.inference_jobs "
        "(priority DESC, available_at, created_at) "
        "WHERE state = 'queued'"
    )
    _execute(
        "CREATE INDEX ix_jobs_inference_expired_leases "
        "ON jobs.inference_jobs (lease_expires_at) WHERE state = 'leased'"
    )
    _execute(
        "CREATE INDEX ix_jobs_inference_channel "
        "ON jobs.inference_jobs (tenant_id, channel_id, created_at DESC)"
    )

    _execute(
        """
        CREATE TABLE jobs.job_attempts (
            id uuid PRIMARY KEY,
            tenant_id uuid NOT NULL,
            job_id uuid NOT NULL,
            attempt_number integer NOT NULL CHECK (attempt_number >= 1),
            worker_id text NOT NULL,
            state text NOT NULL
                CHECK (state IN ('started', 'succeeded', 'failed', 'abandoned')),
            started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            finished_at timestamptz,
            error_class text,
            safe_error_metadata jsonb
                CHECK (
                    safe_error_metadata IS NULL
                    OR jsonb_typeof(safe_error_metadata) = 'object'
                ),
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, job_id, attempt_number),
            FOREIGN KEY (tenant_id, job_id)
                REFERENCES jobs.inference_jobs (tenant_id, id) ON DELETE CASCADE
        )
        """
    )

    _execute(
        """
        CREATE TABLE jobs.outbox (
            id uuid PRIMARY KEY,
            sequence_number bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
            tenant_id uuid NOT NULL,
            aggregate_type text NOT NULL,
            aggregate_id uuid NOT NULL,
            event_type text NOT NULL,
            deduplication_key text,
            payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
            occurred_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            published_at timestamptz,
            publish_attempts integer NOT NULL DEFAULT 0 CHECK (publish_attempts >= 0),
            lease_owner text,
            lease_expires_at timestamptz,
            last_error_class text,
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, deduplication_key)
        )
        """
    )
    _execute(
        "CREATE INDEX ix_jobs_outbox_unpublished "
        "ON jobs.outbox (available_at, sequence_number) WHERE published_at IS NULL"
    )

    for schema, table in TENANT_TABLES:
        qualified = f"{schema}.{table}"
        policy = f"{schema}_{table}_tenant_isolation"
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

    for schema in SCHEMAS:
        _execute(
            f"GRANT USAGE ON SCHEMA {schema} "
            "TO eva_api, eva_worker, eva_agent_reader, eva_audit_writer, eva_backup"
        )

    _execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA iam TO eva_api"
    )
    _execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA agent TO eva_api"
    )
    _execute(
        "GRANT SELECT, INSERT ON jobs.inference_jobs, jobs.outbox TO eva_api"
    )
    _execute("GRANT SELECT ON audit.events TO eva_api")

    _execute(
        "GRANT SELECT, UPDATE ON jobs.inference_jobs TO eva_worker"
    )
    _execute(
        "GRANT SELECT, INSERT, UPDATE ON jobs.job_attempts, jobs.outbox TO eva_worker"
    )

    _execute(
        "GRANT SELECT ON agent.sessions, agent.messages, agent.tool_runs, "
        "agent.action_plans, agent.action_approvals TO eva_agent_reader"
    )
    _execute(
        "GRANT SELECT ON iam.permissions, iam.user_channel_grants TO eva_agent_reader"
    )
    _execute("GRANT INSERT ON audit.events TO eva_audit_writer")

    for schema in SCHEMAS:
        _execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO eva_backup")
        _execute(f"GRANT SELECT ON ALL SEQUENCES IN SCHEMA {schema} TO eva_backup")

    _execute(
        "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO eva_audit_writer"
    )
    _execute(
        "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA jobs TO eva_api, eva_worker"
    )

    _execute(
        "ALTER DEFAULT PRIVILEGES FOR ROLE eva_owner IN SCHEMA audit "
        "REVOKE UPDATE, DELETE ON TABLES FROM eva_audit_writer"
    )
    _execute("RESET ROLE")


def downgrade() -> None:
    for schema in ("jobs", "audit", "agent", "iam"):
        _execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
