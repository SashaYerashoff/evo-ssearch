import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "update_bundle.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")
BUILD_SCRIPT = (ROOT / "scripts" / "build_patch_bundle.sh").read_text(encoding="utf-8")


class UpdateBundleTests(unittest.TestCase):
    def test_shell_parses_and_help_is_safe(self):
        subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)
        completed = subprocess.run(
            ["bash", str(SCRIPT_PATH), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unpacked EVA AI bundle", completed.stdout)

    def test_detects_both_systemd_contexts(self):
        self.assertIn("systemctl --user", SCRIPT)
        self.assertIn('MODE="system"', SCRIPT)
        self.assertIn("find_user_service", SCRIPT)
        self.assertIn("find_system_service", SCRIPT)
        self.assertIn("user-systemd dev mode skips", SCRIPT)
        self.assertIn("Production installer dry-run", SCRIPT)

    def test_adopt_update_never_migrates_or_dumps_database(self):
        self.assertIn("--no-migrate", SCRIPT)
        self.assertIn("--skip-pg-dump", SCRIPT)
        self.assertNotIn("--run-migrations", SCRIPT)
        self.assertIn("default_transaction_read_only=on", SCRIPT)
        self.assertIn('${HOME}/.local/state/eva-ai/0.8.4-backups', SCRIPT)
        self.assertIn('if [[ "${MODE}" == "user" ]]', SCRIPT)

    def test_human_confirmation_and_restart_are_separate(self):
        confirmation = SCRIPT.index("Type UPDATE")
        install = SCRIPT.index('"${BUNDLE_DIR}/scripts/install_patch.sh"')
        restart_prompt = SCRIPT.index("Restart %s.service now?")
        self.assertLess(confirmation, install)
        self.assertLess(install, restart_prompt)
        self.assertIn("is up and running", SCRIPT)

    def test_builder_places_entrypoint_at_bundle_root(self):
        self.assertIn('"${BUNDLE_DIR}/update.sh"', BUILD_SCRIPT)
        self.assertIn('chmod 0755 "${BUNDLE_DIR}/update.sh"', BUILD_SCRIPT)


if __name__ == "__main__":
    unittest.main()
