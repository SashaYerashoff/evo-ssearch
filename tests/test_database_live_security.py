import os
import unittest
from uuid import uuid4


@unittest.skipUnless(
    os.getenv("EVA_TEST_DATABASE_DSN"),
    "set EVA_TEST_DATABASE_DSN for live PostgreSQL security tests",
)
class PostgreSQLSecurityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg

        cls.psycopg = psycopg
        cls.dsn = os.environ["EVA_TEST_DATABASE_DSN"]

    def setUp(self):
        self.connection = self.psycopg.connect(self.dsn)

    def tearDown(self):
        self.connection.rollback()
        self.connection.close()

    def _set_runtime_context(self, role: str, tenant_id: str) -> None:
        self.assertIn(
            role,
            {
                "eva_api",
                "eva_worker",
                "eva_agent_reader",
                "eva_audit_writer",
                "eva_owner",
            },
        )
        self.connection.execute(f"SET LOCAL ROLE {role}")
        self.connection.execute(
            "SELECT set_config('eva.tenant_id', %s, true)",
            (tenant_id,),
        )

    def test_runtime_roles_cannot_bypass_rls(self):
        rows = self.connection.execute(
            """
            SELECT rolname, rolsuper, rolcreaterole, rolcreatedb, rolbypassrls
            FROM pg_roles
            WHERE rolname LIKE 'eva_%'
            ORDER BY rolname
            """
        ).fetchall()

        self.assertGreaterEqual(len(rows), 7)
        for role_name, is_super, can_create_role, can_create_db, bypasses_rls in rows:
            with self.subTest(role=role_name):
                self.assertFalse(is_super)
                self.assertFalse(can_create_role)
                self.assertFalse(can_create_db)
                self.assertFalse(bypasses_rls)

    def test_queue_roles_have_separate_admission_and_worker_grants(self):
        privileges = self.connection.execute(
            """
            SELECT
                has_table_privilege(
                    'eva_api',
                    'jobs.inference_jobs',
                    'SELECT,INSERT,UPDATE'
                ),
                has_table_privilege(
                    'eva_api',
                    'jobs.job_attempts',
                    'INSERT'
                ),
                has_table_privilege(
                    'eva_worker',
                    'jobs.inference_jobs',
                    'SELECT,UPDATE'
                ),
                has_table_privilege(
                    'eva_worker',
                    'jobs.job_attempts',
                    'SELECT,INSERT,UPDATE'
                )
            """
        ).fetchone()

        self.assertEqual(privileges, (True, False, True, True))
        self.assertTrue(
            self.connection.execute(
                """
                SELECT has_table_privilege(
                    'eva_api',
                    'public.alembic_version',
                    'SELECT'
                )
                """
            ).fetchone()[0]
        )
        self.assertTrue(
            self.connection.execute(
                """
                SELECT has_column_privilege(
                    'eva_audit_writer',
                    'audit.events',
                    'id',
                    'SELECT'
                )
                """
            ).fetchone()[0]
        )

    def test_api_role_is_tenant_isolated(self):
        tenant_a = str(uuid4())
        tenant_b = str(uuid4())
        user_id = str(uuid4())

        self._set_runtime_context("eva_api", tenant_a)
        self.connection.execute(
            """
            INSERT INTO iam.users (id, tenant_id, username, password_hash)
            VALUES (%s, %s, 'operator', 'argon2-placeholder')
            """,
            (user_id, tenant_a),
        )
        own_count = self.connection.execute(
            "SELECT count(*) FROM iam.users WHERE id = %s",
            (user_id,),
        ).fetchone()[0]

        self.connection.execute("RESET ROLE")
        self._set_runtime_context("eva_api", tenant_b)
        other_count = self.connection.execute(
            "SELECT count(*) FROM iam.users WHERE id = %s",
            (user_id,),
        ).fetchone()[0]

        self.assertEqual(own_count, 1)
        self.assertEqual(other_count, 0)

    def test_audit_events_are_append_only(self):
        tenant_id = str(uuid4())
        event_id = str(uuid4())

        self._set_runtime_context("eva_audit_writer", tenant_id)
        self.connection.execute(
            """
            INSERT INTO audit.events (
                id,
                tenant_id,
                source_ip,
                action,
                target_type,
                result
            )
            VALUES (%s, %s, '127.0.0.1', 'test.read', 'test', 'success')
            """,
            (event_id, tenant_id),
        )
        self.connection.execute("RESET ROLE")
        self._set_runtime_context("eva_owner", tenant_id)

        with self.assertRaises(self.psycopg.errors.ObjectNotInPrerequisiteState):
            self.connection.execute(
                "UPDATE audit.events SET result = 'failure' WHERE id = %s",
                (event_id,),
            )


if __name__ == "__main__":
    unittest.main()
