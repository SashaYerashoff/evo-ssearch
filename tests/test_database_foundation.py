import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from eva_db import (
    CURRENT_SCHEMA_REVISION,
    DatabaseConfigurationError,
    DatabaseDependencyError,
    DatabaseState,
    DatabaseSettings,
    PsycopgPool,
    TransactionContext,
    redact_dsn,
)
from eva_db.pool import _unsafe_runtime_role_reason
from archive_store import PostgresDetectionsStore


ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
WSGI_SOURCE = (ROOT / "wsgi.py").read_text(encoding="utf-8")
ARCHIVE_STORE_SOURCE = (ROOT / "archive_store.py").read_text(encoding="utf-8")
MIGRATION = ROOT / "migrations" / "versions" / (
    "20260609_0001_secure_foundation.py"
)
QUEUE_GRANTS_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260609_0002_queue_role_grants.py"
)
DURABLE_APPROVALS_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260609_0003_durable_agent_approvals.py"
)
IAM_ADMIN_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260610_0004_iam_admin_and_throttle.py"
)
ARCHIVE_RUNTIME_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260612_0005_archive_runtime.py"
)
IAM_ALL_CHANNEL_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260614_0006_iam_all_channel_access.py"
)
ALERT_FEEDBACK_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260725_0007_alert_feedback.py"
)
ATTENTION_STORAGE_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260726_0008_attention_storage.py"
)
VLM_BATCH_IDENTITY_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260726_0009_vlm_batch_identity.py"
)
AUDIT_HASH_CHAIN_MIGRATION = ROOT / "migrations" / "versions" / (
    "20260727_0010_audit_hash_chain.py"
)


class DatabaseSettingsTests(unittest.TestCase):
    def test_redact_uri_and_keyword_dsn(self):
        uri = (
            "postgresql://eva:correct-horse@db.internal/eva"
            "?sslmode=require&password=query-secret"
        )
        redacted_uri = redact_dsn(uri)
        self.assertNotIn("correct-horse", redacted_uri)
        self.assertNotIn("query-secret", redacted_uri)
        self.assertIn("eva:***@", redacted_uri)
        self.assertIn("password=%2A%2A%2A", redacted_uri)

        keyword = "host=db dbname=eva user=eva password='space secret'"
        redacted_keyword = redact_dsn(keyword)
        self.assertNotIn("space secret", redacted_keyword)
        self.assertIn("password=***", redacted_keyword)

    def test_settings_parse_bounds_and_repr_is_secret_safe(self):
        settings = DatabaseSettings.from_env(
            {
                "EVA_DATABASE_DSN": "postgresql://eva:top-secret@db/eva",
                "EVA_DB_POOL_MIN_SIZE": "2",
                "EVA_DB_POOL_MAX_SIZE": "12",
                "EVA_DB_POOL_MAX_WAITING": "40",
                "EVA_DB_STATEMENT_TIMEOUT_MS": "20000",
            }
        )
        self.assertEqual(settings.pool_min_size, 2)
        self.assertEqual(settings.pool_max_size, 12)
        self.assertEqual(settings.expected_schema_revision, CURRENT_SCHEMA_REVISION)
        self.assertNotIn("top-secret", repr(settings))
        self.assertIn("***", repr(settings))
        self.assertNotIn("top-secret", repr(PsycopgPool(settings)))

    def test_settings_require_postgresql_and_bounded_pool(self):
        with self.assertRaises(DatabaseConfigurationError):
            DatabaseSettings.from_env({})
        with self.assertRaises(DatabaseConfigurationError):
            DatabaseSettings.from_env(
                {"EVA_DATABASE_DSN": "sqlite:///tmp/eva.sqlite3"}
            )
        with self.assertRaises(DatabaseConfigurationError):
            DatabaseSettings.from_env(
                {
                    "EVA_DATABASE_DSN": "postgresql://db/eva",
                    "EVA_DB_POOL_MIN_SIZE": "20",
                    "EVA_DB_POOL_MAX_SIZE": "10",
                }
            )


class OptionalDependencyTests(unittest.TestCase):
    def test_package_import_does_not_import_postgresql_dependencies(self):
        script = (
            "import builtins\n"
            "real_import = builtins.__import__\n"
            "def guarded(name, *args, **kwargs):\n"
            "    if name.split('.')[0] in "
            "{'psycopg', 'psycopg_pool', 'sqlalchemy', 'alembic'}:\n"
            "        raise AssertionError(name)\n"
            "    return real_import(name, *args, **kwargs)\n"
            "builtins.__import__ = guarded\n"
            "import eva_db\n"
            "print(eva_db.CURRENT_SCHEMA_REVISION)\n"
        )
        import subprocess

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(CURRENT_SCHEMA_REVISION, completed.stdout)

    def test_repository_ci_runs_drift_compile_and_full_tests(self):
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ubuntu-24.04", workflow)
        self.assertIn("scripts/check_docs_drift.sh", workflow)
        self.assertIn("python -m compileall", workflow)
        self.assertIn("python -m pytest -q", workflow)
        self.assertIn('EVOSSEARCH_OFFLINE_MODE: "true"', workflow)

    def test_wsgi_starts_scheduled_retention_and_cap_has_no_orphan_delete_path(self):
        self.assertIn("ensure_archive_retention_thread()", WSGI_SOURCE)
        self.assertNotIn("_trim_to_cap", ARCHIVE_STORE_SOURCE)
        self.assertIn("deleted_image_paths", ARCHIVE_STORE_SOURCE)

    def test_pool_reports_actionable_error_when_driver_is_missing(self):
        settings = DatabaseSettings(dsn="postgresql://db/eva")
        pool = PsycopgPool(settings)
        real_import_module = importlib.import_module

        def missing_driver(name, package=None):
            if name == "psycopg":
                error = ModuleNotFoundError("No module named 'psycopg'")
                error.name = "psycopg"
                raise error
            return real_import_module(name, package)

        with patch("eva_db.pool.importlib.import_module", side_effect=missing_driver):
            with self.assertRaisesRegex(
                DatabaseDependencyError, "requirements-db.txt"
            ):
                pool.open()


class TransactionContextTests(unittest.TestCase):
    def test_context_validates_uuid_values(self):
        context = TransactionContext(
            tenant_id="f3c3533e-bf17-46a1-a543-696d95b8cf6f",
            actor_id="fc3dbc49-151a-44c9-a70b-3035e900c496",
            request_id="request-123",
        )
        self.assertEqual(
            context.as_database_values(),
            (
                "f3c3533e-bf17-46a1-a543-696d95b8cf6f",
                "fc3dbc49-151a-44c9-a70b-3035e900c496",
                "request-123",
                "",
            ),
        )
        with self.assertRaisesRegex(ValueError, "tenant_id"):
            TransactionContext(
                tenant_id="tenant-from-request-json",
                actor_id="fc3dbc49-151a-44c9-a70b-3035e900c496",
            ).as_database_values()


class RuntimeRoleSafetyTests(unittest.TestCase):
    def test_unsafe_runtime_role_reasons(self):
        self.assertEqual(
            _unsafe_runtime_role_reason(
                current_user="postgres",
                is_superuser=False,
                can_create_role=False,
                can_create_db=False,
                bypasses_rls=False,
            ),
            "forbidden runtime role: postgres",
        )
        self.assertEqual(
            _unsafe_runtime_role_reason(
                current_user="eva_api_login",
                is_superuser=False,
                can_create_role=False,
                can_create_db=False,
                bypasses_rls=True,
            ),
            "runtime role bypasses row-level security",
        )
        self.assertIsNone(
            _unsafe_runtime_role_reason(
                current_user="eva_api_login",
                is_superuser=False,
                can_create_role=False,
                can_create_db=False,
                bypasses_rls=False,
            )
        )


class ArchiveChannelFilterTests(unittest.TestCase):
    def test_multi_channel_scope_builds_parameterized_in_clause(self):
        store = object.__new__(PostgresDetectionsStore)
        store.tenant_id = "f3c3533e-bf17-46a1-a543-696d95b8cf6f"

        where_sql, params = store._build_where(channel_ids=[9, 7, 9])

        self.assertIn("channel_id IN (%s,%s)", where_sql)
        self.assertEqual(params, [store.tenant_id, 7, 9])

    def test_explicit_empty_channel_scope_matches_no_rows(self):
        store = object.__new__(PostgresDetectionsStore)
        store.tenant_id = "f3c3533e-bf17-46a1-a543-696d95b8cf6f"

        where_sql, params = store._build_where(channel_ids=[])

        self.assertIn("1 = 0", where_sql)
        self.assertEqual(params, [store.tenant_id])

    def test_batch_identity_scope_is_parameterized(self):
        store = object.__new__(PostgresDetectionsStore)
        store.tenant_id = "f3c3533e-bf17-46a1-a543-696d95b8cf6f"

        where_sql, params = store._build_where(
            channel_id=7,
            source="vlm_summary",
            batch_id="vlm-7c6512",
        )

        self.assertIn("payload_json->>'batch_id' = %s", where_sql)
        self.assertEqual(
            params,
            [store.tenant_id, 7, "vlm_summary", "vlm-7c6512"],
        )

    def test_parent_alert_scope_is_parameterized(self):
        store = object.__new__(PostgresDetectionsStore)
        store.tenant_id = "f3c3533e-bf17-46a1-a543-696d95b8cf6f"

        where_sql, params = store._build_where(
            channel_id=7,
            source="vlm_alert",
            parent_alert_id="vlm-alert-exact",
        )

        self.assertIn("payload_json->>'parent_alert_id' = %s", where_sql)
        self.assertEqual(
            params,
            [store.tenant_id, 7, "vlm_alert", "vlm-alert-exact"],
        )


class MigrationContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MIGRATION.read_text(encoding="utf-8")

    def test_initial_migration_contains_required_schemas_and_tables(self):
        for schema in ("iam", "agent", "audit", "jobs"):
            self.assertIn(f'"{schema}"', self.source)
        self.assertIn("CREATE SCHEMA IF NOT EXISTS {schema}", self.source)

        required_tables = {
            "iam": (
                "users",
                "sessions",
                "roles",
                "permissions",
                "user_roles",
                "role_permissions",
                "user_channel_grants",
            ),
            "agent": (
                "sessions",
                "messages",
                "tool_runs",
                "action_plans",
                "action_approvals",
            ),
            "audit": ("events",),
            "jobs": ("inference_jobs", "job_attempts", "outbox"),
        }
        for schema, tables in required_tables.items():
            for table in tables:
                self.assertIn(f"CREATE TABLE {schema}.{table}", self.source)

    def test_migration_has_security_and_operational_boundaries(self):
        for role in (
            "eva_owner",
            "eva_migrator",
            "eva_api",
            "eva_worker",
            "eva_agent_reader",
            "eva_audit_writer",
            "eva_backup",
        ):
            self.assertIn(role, self.source)
        self.assertIn("ENABLE ROW LEVEL SECURITY", self.source)
        self.assertIn("FORCE ROW LEVEL SECURITY", self.source)
        self.assertIn("current_setting('eva.tenant_id', true)", self.source)
        self.assertIn("audit events are append-only", self.source)
        self.assertIn("WHERE state = 'queued'", self.source)
        self.assertIn("idempotency_key", self.source)
        self.assertIn("approval_token_hash", self.source)
        self.assertIn("fk_agent_tool_runs_audit_event", self.source)
        self.assertIn("channel_id bigint", self.source)
        self.assertIn("'heartbeat', 'event', 'manual'", self.source)
        self.assertIn("'dead_letter'", self.source)

    def test_migration_does_not_add_deferred_data_plane_schemas(self):
        lowered = self.source.lower()
        self.assertNotIn("create extension vector", lowered)
        self.assertNotIn("create schema events", lowered)
        self.assertNotIn("create schema vectors", lowered)

    def test_alembic_config_and_revision_agree(self):
        self.assertTrue((ROOT / "alembic.ini").is_file())
        self.assertTrue((ROOT / "migrations" / "env.py").is_file())
        queue_grants_source = QUEUE_GRANTS_MIGRATION.read_text(encoding="utf-8")
        durable_approvals_source = DURABLE_APPROVALS_MIGRATION.read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'revision: str = "20260609_0002"',
            queue_grants_source,
        )
        self.assertIn(
            'down_revision: str | None = "20260609_0001"',
            queue_grants_source,
        )
        self.assertIn(
            "GRANT UPDATE ON jobs.inference_jobs TO eva_api",
            queue_grants_source,
        )
        self.assertIn(
            "GRANT SELECT ON public.alembic_version",
            queue_grants_source,
        )
        self.assertIn(
            "GRANT SELECT (id) ON audit.events TO eva_audit_writer",
            queue_grants_source,
        )
        self.assertIn(
            'revision: str = "20260609_0003"',
            durable_approvals_source,
        )
        self.assertIn(
            'down_revision: str | None = "20260609_0002"',
            durable_approvals_source,
        )
        self.assertIn("octet_length(arguments_hash) = 32", durable_approvals_source)
        self.assertIn(
            "ux_agent_action_approvals_one_active_per_plan",
            durable_approvals_source,
        )
        iam_admin_source = IAM_ADMIN_MIGRATION.read_text(encoding="utf-8")
        self.assertIn(
            'revision: str = "20260610_0004"',
            iam_admin_source,
        )
        self.assertIn(
            'down_revision: str | None = "20260609_0003"',
            iam_admin_source,
        )
        self.assertIn("CREATE TABLE iam.login_attempts", iam_admin_source)
        self.assertIn("GRANT SELECT, INSERT, UPDATE, DELETE", iam_admin_source)
        archive_runtime_source = ARCHIVE_RUNTIME_MIGRATION.read_text(
            encoding="utf-8"
        )
        self.assertIn('revision: str = "20260612_0005"', archive_runtime_source)
        self.assertIn(
            'down_revision: str | None = "20260610_0004"',
            archive_runtime_source,
        )
        self.assertIn("CREATE SCHEMA IF NOT EXISTS archive", archive_runtime_source)
        self.assertIn("CREATE TABLE archive.detections", archive_runtime_source)
        self.assertIn("CREATE TABLE archive.probes", archive_runtime_source)
        self.assertIn("CREATE TABLE archive.runtime_state", archive_runtime_source)
        self.assertIn("ix_archive_detections_source_ts", archive_runtime_source)
        iam_all_channel_source = IAM_ALL_CHANNEL_MIGRATION.read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'revision: str = "20260614_0006"',
            iam_all_channel_source,
        )
        self.assertIn(
            'down_revision: str | None = "20260612_0005"',
            iam_all_channel_source,
        )
        self.assertIn("all_channel_access", iam_all_channel_source)
        alert_feedback_source = ALERT_FEEDBACK_MIGRATION.read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'revision: str = "20260725_0007"',
            alert_feedback_source,
        )
        self.assertIn(
            'down_revision: str | None = "20260614_0006"',
            alert_feedback_source,
        )
        self.assertIn(
            "CREATE TABLE archive.alert_feedback",
            alert_feedback_source,
        )
        self.assertIn(
            "archive_alert_feedback_tenant_isolation",
            alert_feedback_source,
        )
        self.assertIn("'benign_activity'", alert_feedback_source)
        self.assertIn("'poor_visual_quality'", alert_feedback_source)
        attention_storage_source = ATTENTION_STORAGE_MIGRATION.read_text(
            encoding="utf-8"
        )
        self.assertIn('revision: str = "20260726_0008"', attention_storage_source)
        self.assertIn(
            'down_revision: str | None = "20260725_0007"',
            attention_storage_source,
        )
        self.assertIn(
            "CREATE TABLE archive.attention_embedding_snapshots",
            attention_storage_source,
        )
        self.assertIn(
            "CREATE TABLE archive.attention_intervals",
            attention_storage_source,
        )
        vlm_batch_identity_source = VLM_BATCH_IDENTITY_MIGRATION.read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'revision: str = "20260726_0009"',
            vlm_batch_identity_source,
        )
        self.assertIn(
            'down_revision: str | None = "20260726_0008"',
            vlm_batch_identity_source,
        )
        self.assertIn(
            "ix_archive_detections_vlm_batch",
            vlm_batch_identity_source,
        )
        self.assertIn(
            "payload_json->>'batch_id'",
            vlm_batch_identity_source,
        )
        audit_hash_chain_source = AUDIT_HASH_CHAIN_MIGRATION.read_text(
            encoding="utf-8"
        )
        self.assertIn(
            f'revision: str = "{CURRENT_SCHEMA_REVISION}"',
            audit_hash_chain_source,
        )
        self.assertIn(
            'down_revision: str | None = "20260726_0009"',
            audit_hash_chain_source,
        )
        self.assertIn(
            "GRANT SELECT (tenant_id, sequence_number, event_hash)",
            audit_hash_chain_source,
        )
        self.assertIn(
            "ix_audit_events_tenant_sequence",
            audit_hash_chain_source,
        )
        self.assertIn(
            "octet_length(event_hash) = 32",
            audit_hash_chain_source,
        )
        self.assertIn(
            "CREATE TABLE archive.attention_probe_scores",
            attention_storage_source,
        )
        self.assertIn("ix_archive_detections_source_channel_ts", archive_runtime_source)
        self.assertIn("ENABLE ROW LEVEL SECURITY", archive_runtime_source)
        self.assertIn("current_setting('eva.tenant_id', true)", archive_runtime_source)


@unittest.skipUnless(
    os.getenv("EVA_TEST_DATABASE_DSN"),
    "set EVA_TEST_DATABASE_DSN for live PostgreSQL readiness test",
)
class PostgreSQLIntegrationTests(unittest.TestCase):
    def test_live_database_readiness(self):
        settings = DatabaseSettings(
            dsn=os.environ["EVA_TEST_DATABASE_DSN"],
            pool_min_size=0,
            pool_max_size=1,
        )
        pool = PsycopgPool(settings)
        try:
            result = pool.check_readiness()
        finally:
            pool.close()
        self.assertTrue(result.ready, result)

    def test_live_database_runtime_role_check(self):
        settings = DatabaseSettings(
            dsn=os.environ["EVA_TEST_DATABASE_DSN"],
            pool_min_size=0,
            pool_max_size=1,
        )
        pool = PsycopgPool(settings)
        try:
            relaxed = pool.check_runtime_role(strict=False)
            strict = pool.check_runtime_role(strict=True)
        finally:
            pool.close()

        self.assertEqual(relaxed.state, DatabaseState.READY)
        self.assertTrue(relaxed.current_user)
        if relaxed.unsafe_reason:
            self.assertEqual(strict.state, DatabaseState.UNSAFE_RUNTIME_ROLE)
            self.assertFalse(strict.ready)
        else:
            self.assertEqual(strict.state, DatabaseState.READY)
            self.assertTrue(strict.ready)


if __name__ == "__main__":
    unittest.main()
