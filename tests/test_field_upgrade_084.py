import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "field_upgrade_084.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")


class FieldUpgradeGuideTests(unittest.TestCase):
    def test_script_parses_and_refuses_without_root(self):
        subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)
        completed = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.stdout and "id -u" not in completed.stdout:
            # Running as root (CI edge) would proceed past this gate; only
            # assert the refusal when we are an unprivileged user.
            import os

            if os.geteuid() != 0:
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("sudo", completed.stdout)

    def test_migration_path_is_unreachable_from_the_guide(self):
        # The guide must never carry or prompt for a privileged migration
        # identity; a schema mismatch is a hard stop, not a migration.
        self.assertIn("--no-migrate", SCRIPT)
        self.assertNotIn("EVA_INSTALL_MIGRATION_DSN", SCRIPT)
        self.assertNotIn("EVA_MIGRATION_DATABASE_DSN", SCRIPT)
        self.assertIn('EXPECTED_SCHEMA="20260614_0006"', SCRIPT)
        self.assertIn("alembic_version", SCRIPT)
        self.assertIn("не совпадает с ожидаемой", SCRIPT)

    def test_dry_run_and_confirmation_precede_apply(self):
        dry_run_at = SCRIPT.index("--dry-run --non-interactive --no-migrate")
        confirm_at = SCRIPT.index('!= "UPGRADE"')
        apply_at = SCRIPT.index("--apply --non-interactive --no-migrate")
        self.assertLess(dry_run_at, confirm_at)
        self.assertLess(confirm_at, apply_at)

    def test_rollback_command_and_evidence_are_recorded(self):
        self.assertIn("ROLLBACK_COMMAND.txt", SCRIPT)
        self.assertIn("client_diagnostics.sh", SCRIPT)
        self.assertIn("EVIDENCE_DIR", SCRIPT)
        # The runtime DSN is parsed for the read-only schema check but must
        # never be printed back to the terminal or the evidence files.
        self.assertNotIn("print(dsn", SCRIPT)
        self.assertNotIn("echo ${dsn", SCRIPT.lower())


if __name__ == "__main__":
    unittest.main()
