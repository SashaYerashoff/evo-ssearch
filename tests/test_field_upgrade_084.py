import subprocess
import unittest
import re
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
        self.assertIn("does not match ${EXPECTED_SCHEMA}", SCRIPT)

    def test_runtime_terminal_and_ui_sources_have_no_cyrillic_copy(self):
        roots = [ROOT / "scripts", ROOT / "static", ROOT / "templates", ROOT / "react-ui"]
        suffixes = {".sh", ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css"}
        cyrillic = re.compile(r"[\u0400-\u04ff]")
        offenders = []
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if "node_modules" in path.parts:
                    continue
                if path.is_file() and path.suffix in suffixes:
                    if cyrillic.search(path.read_text(encoding="utf-8", errors="ignore")):
                        offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

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
        self.assertIn('2>&1 | tee "${EVIDENCE_DIR}/apply.txt"', SCRIPT)
        self.assertIn("ROLLBACK HANDOFF:", SCRIPT)
        # The runtime DSN is parsed for the read-only schema check but must
        # never be printed back to the terminal or the evidence files.
        self.assertNotIn("print(dsn", SCRIPT)
        self.assertNotIn("echo ${dsn", SCRIPT.lower())

    def test_exact_bundle_and_dependency_neutral_adopt_are_gated(self):
        self.assertIn('EXPECTED_VERSION="β 0.8.4"', SCRIPT)
        self.assertIn('MANIFEST_VERSION=', SCRIPT)
        self.assertIn('MANIFEST_TREE_STATUS=', SCRIPT)
        self.assertIn('release bundle must be clean', SCRIPT)
        self.assertIn('DEPLOYED_VERSION=', SCRIPT)
        self.assertIn('this exact ${EXPECTED_VERSION} bundle is already installed', SCRIPT)
        self.assertIn('adopt-upgrade candidate', SCRIPT)
        self.assertNotIn('"β 0.8.0"|"β 0.8.1"', SCRIPT)
        self.assertNotIn('unsupported deployed VERSION', SCRIPT)
        self.assertIn('MANIFEST_COMMIT=', SCRIPT)
        self.assertIn('.eva-bundle-commit', SCRIPT)
        self.assertIn('cmp -s "${APP_DIR}/${requirements_file}"', SCRIPT)
        self.assertIn('-m pip check', SCRIPT)
        self.assertIn('uv pip check --python', SCRIPT)
        self.assertIn('neither python -m pip nor uv is available', SCRIPT)

    def test_post_upgrade_http_check_rejects_error_statuses(self):
        self.assertIn('curl -fsS -m 10 "${BASE_URL}/health"', SCRIPT)
        self.assertIn('curl -fsS -m 10 "${BASE_URL}/ready?load=1"', SCRIPT)
        self.assertIn('payload.get("status") == "ready"', SCRIPT)
        self.assertIn('payload.get("version") == sys.argv[1]', SCRIPT)
        last_state_refresh = SCRIPT.rindex('systemctl is-active "${SERVICE_NAME}"')
        health_loop = SCRIPT.index('for _attempt in 1 2 3 4 5 6 7 8 9')
        self.assertGreater(last_state_refresh, health_loop)

    def test_verification_commands_are_required_before_apply(self):
        self.assertIn('for required_command in cmp curl find grep sed systemctl tee', SCRIPT)


if __name__ == "__main__":
    unittest.main()
