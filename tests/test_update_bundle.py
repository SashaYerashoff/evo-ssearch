import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "update_bundle.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")
BUILD_SCRIPT = (ROOT / "scripts" / "build_patch_bundle.sh").read_text(encoding="utf-8")
ROLLBACK_SCRIPT = (ROOT / "scripts" / "rollback.sh").read_text(encoding="utf-8")
INSTALL_SCRIPT = (ROOT / "scripts" / "install_patch.sh").read_text(encoding="utf-8")
VERIFY_SCRIPT = (ROOT / "scripts" / "verify_patch.sh").read_text(encoding="utf-8")


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

    def test_config_is_discovered_from_active_systemd_unit_before_path_fallbacks(self):
        discovery = SCRIPT.index("discover_systemd_env_file()")
        environment_files = SCRIPT.index("-p EnvironmentFiles --value")
        system_fallback = SCRIPT.index('path_is_file "/etc/eva-ai/eva-ai.env"')
        app_fallback = SCRIPT.index('path_is_file "${APP_DIR}/.env"')
        self.assertLess(discovery, environment_files)
        self.assertLess(environment_files, system_fallback)
        self.assertLess(system_fallback, app_fallback)
        self.assertIn('ENV_FILE_SOURCE="systemd EnvironmentFiles"', SCRIPT)
        self.assertIn("Config source:", SCRIPT)

    def test_selected_config_is_cross_checked_with_active_runtime(self):
        self.assertIn('"${BASE_URL}/ready"', SCRIPT)
        self.assertIn("active runtime identity loaded from /ready", SCRIPT)
        self.assertIn("selected config does not match the active runtime agent profile", SCRIPT)
        self.assertIn("selected config does not match the active Luxriot endpoint", SCRIPT)
        self.assertIn("never rewrites model or server endpoints", SCRIPT)
        self.assertIn("WARN: active service reports %s while %s/VERSION is %s", SCRIPT)
        self.assertIn("--verified-adopt-existing-config", SCRIPT)

    def test_legacy_update_disables_unconfigured_archive_retention(self):
        self.assertIn("ARCHIVE_RETENTION_POLICY_MISSING=false", SCRIPT)
        self.assertIn("legacy config has no archive retention policy", SCRIPT)
        self.assertIn("EVOSSEARCH_ARCHIVE_RETENTION_ENABLED=false", SCRIPT)
        self.assertIn("review retention before enabling pruning", SCRIPT)

    def test_adopt_update_never_migrates_or_dumps_database(self):
        self.assertIn("--no-migrate", SCRIPT)
        self.assertIn("--skip-pg-dump", SCRIPT)
        self.assertNotIn("--run-migrations", SCRIPT)
        self.assertIn("default_transaction_read_only=on", SCRIPT)
        self.assertIn('${HOME}/.local/state/eva-ai/0.8.7-backups', SCRIPT)
        self.assertIn('if [[ "${MODE}" == "user" ]]', SCRIPT)

    def test_release_identity_and_schema_gate_match_the_current_tree(self):
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertIn(f'EXPECTED_VERSION="{version}"', SCRIPT)
        self.assertIn('EXPECTED_SCHEMA="20260805_0013"', SCRIPT)

    def test_human_confirmation_and_restart_are_separate(self):
        confirmation = SCRIPT.index("Install %s now?")
        install = SCRIPT.index('"${BUNDLE_DIR}/scripts/install_patch.sh"')
        restart_prompt = SCRIPT.index("Restart %s.service now?")
        self.assertLess(confirmation, install)
        self.assertLess(install, restart_prompt)
        self.assertIn("[y/N]", SCRIPT)
        self.assertIn("is up and running", SCRIPT)
        self.assertIn("ready_json_matches_version", SCRIPT)
        self.assertIn('payload.get("version") == sys.argv[1]', SCRIPT)
        self.assertNotIn('grep -Fq "${EXPECTED_VERSION}"', SCRIPT)

    def test_same_version_hotfix_is_idempotent_by_bundle_commit(self):
        self.assertIn("same-version hotfix", SCRIPT)
        self.assertIn(".eva-bundle-commit", SCRIPT)
        self.assertIn("this exact ${EXPECTED_VERSION} bundle is already installed", SCRIPT)

    def test_adopt_compatibility_is_schema_and_dependency_gated_not_version_allowlisted(self):
        self.assertIn("adopt-upgrade candidate", SCRIPT)
        self.assertIn("Compatibility is determined by the exact requirements and read-only schema gates", SCRIPT)
        self.assertNotIn('"β 0.8.0"|"β 0.8.1"', SCRIPT)
        self.assertNotIn("unsupported installed version", SCRIPT)
        requirements_at = SCRIPT.index("for requirements_file in requirements.txt requirements-db.txt")
        schema_at = SCRIPT.index('SCHEMA_VERSION="$(target_python')
        confirmation_at = SCRIPT.index("Install %s now?")
        self.assertLess(requirements_at, confirmation_at)
        self.assertLess(schema_at, confirmation_at)
        self.assertIn("active runtime /ready (no DSN stored in selected file)", SCRIPT)

    def test_post_restart_wait_allows_slow_model_restore(self):
        self.assertIn("READY_DEADLINE=$((SECONDS + 240))", SCRIPT)
        self.assertIn("while (( SECONDS < READY_DEADLINE ))", SCRIPT)
        self.assertIn('"${BASE_URL}/ready?load=1"', SCRIPT)

    def test_agent_context_mismatch_requires_confirmation_but_is_not_rewritten(self):
        context_check = SCRIPT.index("Agent context compatibility decision")
        confirmation = SCRIPT.index("Install %s now?")
        self.assertLess(context_check, confirmation)
        self.assertIn("Continue without changing the inference configuration? [y/N]", SCRIPT)
        self.assertIn("short-context update declined; nothing was changed", SCRIPT)
        self.assertIn("Inference was not changed", SCRIPT)
        self.assertNotIn("EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS={temporary_agent_context}", SCRIPT)
        self.assertNotIn('key == "EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS"', SCRIPT)

    def test_agent_context_probe_uses_profile_auth_and_unknown_requires_confirmation(self):
        self.assertIn("EVOSSEARCH_LM_PROFILE_AGENT_API_KEY", SCRIPT)
        self.assertIn("EVOSSEARCH_LM_API_KEY", SCRIPT)
        self.assertIn('Authorization: Bearer ${AGENT_LM_API_KEY}', SCRIPT)
        unknown_prompt = SCRIPT.index("Continue with the operator-verified Agent configuration? [y/N]")
        confirmation = SCRIPT.index("Install %s now?")
        self.assertLess(unknown_prompt, confirmation)
        self.assertIn("unknown agent context declined; nothing was changed", SCRIPT)
        self.assertIn("Agent context: UNVERIFIED", SCRIPT)

    def test_update_never_writes_model_or_server_settings(self):
        self.assertNotIn("DEFAULT_AGENT_MODEL", SCRIPT)
        self.assertNotIn("AGENT_MODEL_TO_PERSIST", SCRIPT)
        self.assertNotIn("adopted_agent_model", SCRIPT)
        self.assertNotIn("EVOSSEARCH_LM_PROFILE_AGENT_MODEL={", SCRIPT)
        self.assertNotIn('env "EVOSSEARCH_LM_PROFILE_AGENT_MODEL=', SCRIPT)
        self.assertIn("no configuration was or will be modified", SCRIPT)

    def test_update_fingerprints_and_preserves_inference_policy(self):
        self.assertIn("inference_policy_fingerprint()", SCRIPT)
        for prefix in (
            "EVOSSEARCH_LM_",
            "EVOSSEARCH_AGENT_",
            "EVOSSEARCH_INFERENCE_",
        ):
            self.assertIn(prefix, SCRIPT)
        before = SCRIPT.index('INFERENCE_POLICY_HASH_BEFORE="$(inference_policy_fingerprint)"')
        confirmation = SCRIPT.index("Install %s now?")
        install = SCRIPT.index('"${BUNDLE_DIR}/scripts/install_patch.sh"')
        after = SCRIPT.index('INFERENCE_POLICY_HASH_AFTER="$(inference_policy_fingerprint)"')
        self.assertLess(before, confirmation)
        self.assertLess(confirmation, install)
        self.assertLess(install, after)
        self.assertIn("inference policy changed during the code update", SCRIPT)
        self.assertIn("automatic rollback is armed", SCRIPT)
        self.assertIn("inference policy fingerprint preserved", SCRIPT)

    def test_model_preflight_describes_topology_and_warns_instead_of_stopping(self):
        preflight = SCRIPT.index("Model/server configuration preflight (read-only)")
        confirmation = SCRIPT.index("Install %s now?")
        self.assertLess(preflight, confirmation)
        self.assertIn("describe_lm_profile", SCRIPT)
        self.assertIn("EVOSSEARCH_LM_PROFILES", SCRIPT)
        self.assertIn("dedicated vLLM server", SCRIPT)
        self.assertIn("LM Studio or llama.cpp beside EVA", SCRIPT)
        self.assertIn("but the server at %s currently serves", SCRIPT)
        self.assertIn("could not reach %s for profile %s; continuing", SCRIPT)
        self.assertIn("EVA keeps using its configured profile defaults", SCRIPT)

    def test_degraded_runtime_warns_and_is_accepted_after_update(self):
        self.assertIn("PREUPGRADE_DEGRADED=true", SCRIPT)
        self.assertNotIn("restore readiness first", SCRIPT)
        self.assertIn("POST_UPDATE_DEGRADED=true", SCRIPT)
        self.assertIn("matching the pre-update state", SCRIPT)

    def test_dependency_import_preflight_gates_before_install_but_excludes_opencv(self):
        preflight = SCRIPT.index("Python dependency preflight (read-only)")
        confirmation = SCRIPT.index("Install %s now?")
        self.assertLess(preflight, confirmation)
        for module_name in ('"torch"', '"transformers"', '"clip"', '"faiss"', '"psycopg"', '"gunicorn"'):
            self.assertIn(module_name, SCRIPT)
        self.assertNotIn('"cv2"', SCRIPT)
        self.assertIn("cannot import modules required by", SCRIPT)
        self.assertIn("the bundled rescue wheel will be used", SCRIPT)

    def test_branded_runtime_version_mismatch_warns_instead_of_stopping(self):
        self.assertNotIn('stop "active service reports', SCRIPT)
        self.assertIn("via the EVOSSEARCH_APP_VERSION override", SCRIPT)
        self.assertIn("field builds may brand the runtime version differently", SCRIPT)

    def test_system_update_authenticates_before_confirmation_and_stop(self):
        sudo_check = SCRIPT.index('sudo -v || stop "sudo authentication failed; service was not stopped"')
        confirmation = SCRIPT.index("Install %s now?")
        service_stop = SCRIPT.index('systemctl_write stop "${SERVICE_NAME}.service"')
        self.assertLess(sudo_check, confirmation)
        self.assertLess(confirmation, service_stop)

    def test_builder_places_entrypoint_at_bundle_root(self):
        self.assertIn('"${BUNDLE_DIR}/update.sh"', BUILD_SCRIPT)
        self.assertIn('chmod 0755 "${BUNDLE_DIR}/update.sh"', BUILD_SCRIPT)

    def test_media_runtime_is_required_and_checked_before_confirmation(self):
        media_check = SCRIPT.index('MEDIA_RUNTIME="')
        checksum = SCRIPT.index("sha256sum -c SHA256SUMS")
        confirmation = SCRIPT.index("Install %s now?")
        self.assertLess(media_check, confirmation)
        self.assertLess(checksum, confirmation)
        self.assertIn("bundled ffmpeg failed the decode smoke test", SCRIPT)
        self.assertIn("bundled OpenCV wheel is incompatible", SCRIPT)
        self.assertIn("--ffmpeg-archive FILE", BUILD_SCRIPT)
        self.assertIn("--opencv-wheel FILE", BUILD_SCRIPT)
        self.assertIn("--media-runtime-dir DIR", BUILD_SCRIPT)
        self.assertIn("Media runtime checksum verification failed", BUILD_SCRIPT)

    def test_system_opencv_preflight_removes_root_owned_temp_payload_as_root(self):
        helper = SCRIPT.index("remove_temp_path()")
        payload = SCRIPT.index('CV_PAYLOAD_TEST_DIR="$(mktemp -d)"')
        confirmation = SCRIPT.index("Install %s now?")
        self.assertLess(helper, payload)
        self.assertLess(payload, confirmation)
        self.assertIn('as_root rm -rf -- "${path}"', SCRIPT)
        self.assertEqual(SCRIPT.count('remove_temp_path "${CV_PAYLOAD_TEST_DIR}"'), 3)

    def test_post_stop_failure_arms_automatic_rollback(self):
        stop_service = SCRIPT.index('systemctl_write stop "${SERVICE_NAME}.service"')
        armed = SCRIPT.index("ROLLBACK_ARMED=true")
        install = SCRIPT.index('"${BUNDLE_DIR}/scripts/install_patch.sh"')
        self.assertLess(stop_service, armed)
        self.assertLess(armed, install)
        self.assertIn("trap automatic_rollback EXIT", SCRIPT)
        self.assertIn("--no-verify", SCRIPT)
        self.assertIn("database and runtime data were untouched", SCRIPT)
        self.assertIn("ready_json_reports_version", SCRIPT)
        self.assertIn("/ready remains degraded", SCRIPT)
        self.assertIn("SECONDS + 60", SCRIPT)

    def test_manual_rollback_supports_user_systemd_and_exact_restore(self):
        self.assertIn("systemctl --user", ROLLBACK_SCRIPT)
        self.assertIn("restore_code_snapshot.py", ROLLBACK_SCRIPT)
        for excluded in (".git", ".local", ".venv*", ".env", "dist", "*.sqlite3", "*.db", "*.log"):
            self.assertIn(f'--exclude="${{APP_BASE}}/{excluded}"', ROLLBACK_SCRIPT)

    def test_manual_rollback_streams_custom_dump_and_removes_new_unit(self):
        self.assertIn('pg_restore --exit-on-error', ROLLBACK_SCRIPT)
        self.assertIn('database is already at the recorded pre-update revision; dump restore skipped', ROLLBACK_SCRIPT)
        self.assertIn('DROP SCHEMA IF EXISTS archive CASCADE', ROLLBACK_SCRIPT)
        self.assertIn('DROP TABLE IF EXISTS public.alembic_version CASCADE', ROLLBACK_SCRIPT)
        self.assertIn('exact dump restore requires a PostgreSQL superuser DSN', ROLLBACK_SCRIPT)
        self.assertIn('restored complete PostgreSQL dump with original ownership', ROLLBACK_SCRIPT)
        self.assertIn('UNIT_PREEXISTED="$(read_state_var unit_preexisted)"', ROLLBACK_SCRIPT)
        self.assertIn('removed service unit created by the failed installation', ROLLBACK_SCRIPT)
        self.assertIn('START_SERVICE=false', ROLLBACK_SCRIPT)
        self.assertIn('cp -a -- "${BACKUP_DIR}/eva-ai.env" "${ENV_FILE}"', ROLLBACK_SCRIPT)

    def test_patch_backup_records_pre_update_database_revision(self):
        self.assertIn('database_revision.txt', INSTALL_SCRIPT)
        self.assertIn('SELECT version_num FROM public.alembic_version LIMIT 1', INSTALL_SCRIPT)

    def test_patch_preserves_external_vlm_policy_and_probe_state(self):
        for key in (
            "EVOSSEARCH_LUXRIOT_SUMMARY_STATE_FILE",
            "EVOSSEARCH_LUXRIOT_ROLLUP_CACHE_FILE",
            "EVOSSEARCH_PROBE_CHANNEL_GROUPS_FILE",
        ):
            self.assertIn(key, INSTALL_SCRIPT)
        self.assertIn('runtime-state.tsv', INSTALL_SCRIPT)
        self.assertIn('restored runtime state', ROLLBACK_SCRIPT)

    def test_patch_bundle_can_ship_siglip2_and_installer_places_it_in_cache(self):
        self.assertIn('--siglip2-cache-repo', BUILD_SCRIPT)
        self.assertIn('models--google--siglip2-base-patch16-224', BUILD_SCRIPT)
        self.assertIn('models--google--siglip2-base-patch16-224', INSTALL_SCRIPT)
        self.assertIn('installed offline SigLIP2 cache', INSTALL_SCRIPT)

    def test_all_code_snapshots_exclude_runtime_private_and_large_trees(self):
        for source in (SCRIPT, INSTALL_SCRIPT, ROLLBACK_SCRIPT):
            for excluded in (
                ".git", "*/.git", ".local", "*/.local", ".venv*", "*/.venv*",
                ".env", ".env.*", "dist", "node_modules", "*/node_modules",
                "*.sqlite3", "*.db", "*.log",
            ):
                self.assertIn(f'--exclude="${{APP_BASE}}/{excluded}"', source)
            self.assertNotIn('--exclude="${APP_BASE}/*/dist"', source)

    def test_update_installs_and_rolls_back_react_production_build(self):
        self.assertIn('REACT_BUILD_SOURCE="${SOURCE_DIR}/react-ui/dist"', INSTALL_SCRIPT)
        self.assertIn('installed React production build', INSTALL_SCRIPT)
        self.assertIn('${SOURCE_DIR}/react-ui/dist/index.html', SCRIPT)
        self.assertIn('${APP_DIR}/react-ui/dist', SCRIPT)
        payload_check = SCRIPT.index('React production build is missing from the offline bundle')
        confirmation = SCRIPT.index('Install %s now?')
        self.assertLess(payload_check, confirmation)
        self.assertIn('<div id="root"></div>', VERIFY_SCRIPT)
        self.assertIn('/ui-assets/assets/', VERIFY_SCRIPT)
        self.assertIn('React command console and hashed frontend asset are served', VERIFY_SCRIPT)

    def test_restore_helper_deletes_new_code_but_keeps_runtime_data(self):
        helper = ROOT / "scripts" / "restore_code_snapshot.py"
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            app = root / "app"
            snapshot_root = root / "snapshot" / "app"
            app.mkdir()
            snapshot_root.mkdir(parents=True)
            (snapshot_root / "old.py").write_text("old\n", encoding="utf-8")
            (snapshot_root / ".venv.broken-old").symlink_to("/mnt/retired/eva-venv")
            archive = root / "code.tgz"
            with tarfile.open(archive, "w:gz") as handle:
                handle.add(snapshot_root, arcname="app")
            (app / "old.py").write_text("changed\n", encoding="utf-8")
            (app / "new.py").write_text("new\n", encoding="utf-8")
            (app / "video").mkdir()
            (app / "video" / "evidence.mp4").write_bytes(b"evidence")
            (app / ".venv.broken-current").symlink_to("/mnt/current/eva-venv")
            subprocess.run(
                ["python3", str(helper), "--archive", str(archive), "--app-dir", str(app)],
                check=True,
            )
            self.assertEqual((app / "old.py").read_text(encoding="utf-8"), "old\n")
            self.assertFalse((app / "new.py").exists())
            self.assertEqual((app / "video" / "evidence.mp4").read_bytes(), b"evidence")
            self.assertEqual((app / ".venv.broken-current").readlink(), Path("/mnt/current/eva-venv"))
            self.assertFalse((app / ".venv.broken-old").is_symlink())


if __name__ == "__main__":
    unittest.main()
