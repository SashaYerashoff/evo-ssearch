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
    DatabaseSettings,
    PsycopgPool,
    TransactionContext,
    redact_dsn,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATION = ROOT / "migrations" / "versions" / (
    "20260609_0001_secure_foundation.py"
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
        self.assertIn(f'revision: str = "{CURRENT_SCHEMA_REVISION}"', self.source)


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


if __name__ == "__main__":
    unittest.main()
