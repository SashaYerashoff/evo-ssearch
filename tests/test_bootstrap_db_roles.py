from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from scripts import bootstrap_db_roles


class BootstrapDbRolesTests(unittest.TestCase):
    def test_required_password_rejects_short_or_missing_values(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                bootstrap_db_roles._required_password("EVA_API_PASSWORD")
        with patch.dict(os.environ, {"EVA_API_PASSWORD": "short"}, clear=True):
            with self.assertRaises(SystemExit):
                bootstrap_db_roles._required_password("EVA_API_PASSWORD")

    def test_required_password_reads_safe_secret(self):
        with patch.dict(
            os.environ,
            {"EVA_API_PASSWORD": "x" * 20},
            clear=True,
        ):
            self.assertEqual(
                bootstrap_db_roles._required_password("EVA_API_PASSWORD"),
                "x" * 20,
            )

    def test_env_matrix_mentions_strict_runtime_role_gate(self):
        matrix = bootstrap_db_roles._env_matrix()
        self.assertIn("eva_api_login", matrix)
        self.assertIn("eva_audit_login", matrix)
        self.assertIn("eva_worker_login", matrix)
        self.assertIn("EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true", matrix)

    def test_migrator_receives_alembic_revision_table_dml(self):
        connection = MagicMock()

        bootstrap_db_roles._grant_migration_revision_access(connection)

        connection.execute.assert_called_once_with(
            "GRANT SELECT, INSERT, UPDATE, DELETE "
            "ON TABLE public.alembic_version TO eva_migrator"
        )


if __name__ == "__main__":
    unittest.main()
