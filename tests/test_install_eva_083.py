from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
INSTALLER_PATH = ROOT / "scripts" / "install_eva_083.py"
SPEC = importlib.util.spec_from_file_location("install_eva_083", INSTALLER_PATH)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)


COMPLETE_ENV = {
    "EVOSSEARCH_LUXRIOT_BASE_URL": "http://evo.internal:8080",
    "EVOSSEARCH_LUXRIOT_USERNAME": "operator",
    "EVOSSEARCH_LUXRIOT_PASSWORD": "EVO-SECRET-DO-NOT-PRINT",
    "EVA_DATABASE_DSN": "postgresql://api:API-SECRET@db.internal/eva",
    "EVA_AUDIT_DATABASE_DSN": "postgresql://audit:AUDIT-SECRET@db.internal/eva",
    "EVA_WORKER_DATABASE_DSN": "postgresql://worker:WORKER-SECRET@db.internal/eva",
    "EVOSSEARCH_LM_PROFILES": "agent,vlm",
    "EVOSSEARCH_LM_AGENT_PROFILE_ID": "agent",
    "EVOSSEARCH_LM_VLM_PROFILE_ID": "vlm",
    "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL": "http://lm.internal:1234/v1",
    "EVOSSEARCH_LM_PROFILE_AGENT_MODEL": "qwen3.5-9b-mtp",
    "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL": "http://vlm.internal:8001/v1",
    "EVOSSEARCH_LM_PROFILE_VLM_MODEL": "qwen/qwen3-vl-4b",
}


def env_text(values=None, *, prefix=""):
    rows = [prefix] if prefix else []
    for key, value in (values or COMPLETE_ENV).items():
        rows.append(f'{key}="{value}"')
    return "\n".join(rows) + "\n"


def make_source(root: Path) -> Path:
    source = root / "source"
    for relative in (
        "migrations",
        "static/js",
        "templates",
        "scripts/install_assets",
    ):
        (source / relative).mkdir(parents=True, exist_ok=True)
    files = {
        "VERSION": "β 0.8.3\n",
        "run_prod.sh": "#!/bin/sh\n",
        "wsgi.py": "app = None\n",
        "requirements.txt": "example==1\n",
        "alembic.ini": "[alembic]\n",
        "static/js/app.js": "// static\n",
        "templates/index.html": "<!doctype html>\n",
        "scripts/preflight_patch.sh": "#!/bin/sh\nexit 0\n",
        "scripts/install_patch.sh": "#!/bin/sh\nexit 0\n",
        "scripts/verify_patch.sh": "#!/bin/sh\nexit 0\n",
        "scripts/rollback.sh": "#!/bin/sh\nexit 0\n",
        "scripts/install_assets/eva-ai.service.in": (
            "[Service]\nUser=@SERVICE_USER@\nGroup=@SERVICE_GROUP@\n"
            "WorkingDirectory=@APP_DIR@\nEnvironmentFile=@ENV_FILE@\n"
            "Environment=EVOSSEARCH_CONFIG_ENV_FILE=@ENV_FILE@\n"
            "ExecStart=@APP_DIR@/run_prod.sh\n"
        ),
    }
    for relative, content in files.items():
        path = source / relative
        path.write_text(content, encoding="utf-8")
        if path.suffix == ".sh":
            path.chmod(0o755)
    return source


def make_options(
    root: Path,
    source: Path,
    *,
    env_file: Path | None,
    migrate=False,
    app_with_venv=True,
):
    app = root / "app"
    app.mkdir(parents=True, exist_ok=True)
    if app_with_venv:
        python = app / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
    bundle = root / "bundle"
    bundle.mkdir()
    return installer.InstallerOptions(
        source_dir=source,
        bundle_dir=bundle,
        app_dir=app,
        env_file=env_file,
        backup_root=root / "backups",
        service_name="eva-ai",
        service_user="eva",
        service_group="eva",
        unit_file=root / "eva-ai.service",
        unit_template=source / "scripts/install_assets/eva-ai.service.in",
        lock_file=root / "eva-ai-installer.lock",
        base_url="http://127.0.0.1:5000",
        python_bin=sys.executable,
        dry_run=True,
        non_interactive=True,
        migrate=migrate,
        start=True,
        verify=True,
    )


class OfflineInstallerUnitTests(unittest.TestCase):
    def test_discovers_existing_app_dotenv_before_source_and_preserves_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            app = root / "app"
            app.mkdir()
            app_env = app / ".env"
            source_env = source / ".env"
            app_env.write_text("APP_SENTINEL=1\n", encoding="utf-8")
            source_env.write_text("SOURCE_SENTINEL=1\n", encoding="utf-8")

            result = installer.discover_env_file(
                explicit=None,
                app_dir=app,
                source_dir=source,
                environ={},
            )

            self.assertEqual(result.source, app_env)
            self.assertEqual(result.target, app_env)
            self.assertEqual(result.existing, {"APP_SENTINEL": "1"})

    def test_explicit_missing_env_copies_source_dotenv_to_requested_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            (source / ".env").write_text("SITE_SETTING=keep\n", encoding="utf-8")
            target = root / "etc/eva-ai.env"

            result = installer.discover_env_file(
                explicit=target,
                app_dir=root / "app",
                source_dir=source,
                environ={},
            )

            self.assertEqual(result.source, source / ".env")
            self.assertEqual(result.target, target)
            self.assertEqual(result.source_kind, "copy")

    def test_env_update_never_overwrites_existing_values_and_is_idempotent(self):
        original = 'CUSTOM_FLAG="keep me"\nEVOSSEARCH_HOST="10.0.0.8"\n'
        first = installer.render_env_update(
            original,
            {"EVOSSEARCH_HOST": "127.0.0.1", "NEW_KEY": "new value"},
        )

        self.assertIn('EVOSSEARCH_HOST="10.0.0.8"', first)
        self.assertIn("NEW_KEY='new value'", first)
        parsed = installer.parse_env_text(first)
        second_updates = {key: value for key, value in {"NEW_KEY": "new value"}.items() if not parsed.get(key)}
        second = installer.render_env_update(first, second_updates)
        self.assertEqual(second, first)

    def test_noninteractive_configuration_accepts_environment_without_echoing_secrets(self):
        resolution = installer.EnvResolution(None, Path("/tmp/eva-ai.env"), "", {})

        values, updates, missing = installer.prepare_env_values(
            resolution,
            environ=COMPLETE_ENV,
            non_interactive=True,
        )

        self.assertEqual(missing, [])
        self.assertEqual(values["EVOSSEARCH_LUXRIOT_PASSWORD"], COMPLETE_ENV["EVOSSEARCH_LUXRIOT_PASSWORD"])
        self.assertIn("EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED", updates)
        self.assertEqual(values["EVOSSEARCH_GUNICORN_THREADS"], "8")
        self.assertEqual(values["EVOSSEARCH_LUXRIOT_LIVE_MEDIA_MAX_SECONDS"], "120")
        self.assertEqual(values["EVOSSEARCH_LUXRIOT_LIVE_MEDIA_MAX_BYTES"], "268435456")
        self.assertEqual(values["EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_SECONDS"], "60")
        rendered = installer.render_env_update("", updates)
        self.assertIn("EVO-SECRET-DO-NOT-PRINT", rendered)

    def test_env_dsn_references_are_resolved_in_memory_but_raw_file_is_preserved(self):
        existing = dict(COMPLETE_ENV)
        existing["EVA_API_PASSWORD"] = "expanded-password"
        existing["EVA_DATABASE_DSN"] = "postgresql://api:${EVA_API_PASSWORD}@db.internal/eva"
        raw = env_text(existing, prefix="# keep comments")
        resolution = installer.EnvResolution(Path("/x/.env"), Path("/x/.env"), raw, existing)

        values, updates, missing = installer.prepare_env_values(
            resolution,
            environ={},
            non_interactive=True,
        )

        self.assertEqual(missing, [])
        self.assertEqual(values["EVA_DATABASE_DSN"], "postgresql://api:expanded-password@db.internal/eva")
        self.assertEqual(updates, {})
        self.assertEqual(installer.render_env_update(raw, updates), raw)

    def test_noninteractive_missing_configuration_is_explicit(self):
        resolution = installer.EnvResolution(None, Path("/tmp/eva-ai.env"), "", {})

        _values, _updates, missing = installer.prepare_env_values(
            resolution,
            environ={},
            non_interactive=True,
        )

        self.assertIn("EVOSSEARCH_LUXRIOT_PASSWORD", missing)
        self.assertIn("EVA_DATABASE_DSN", missing)
        self.assertIn("EVOSSEARCH_LM_PROFILE_<VLM>_BASE_URL", missing)

    def test_interactive_fresh_configuration_prompts_for_evo_postgres_and_lm(self):
        resolution = installer.EnvResolution(None, Path("/tmp/eva-ai.env"), "", {})

        def answer(label):
            if label.startswith("Luxriot Evo base URL"):
                return "http://evo.local:8080"
            if label.startswith("Luxriot Evo username"):
                return "admin"
            if label.startswith("Agent OpenAI-compatible endpoint"):
                return ""
            if label.startswith("Agent model id"):
                return ""
            if label.startswith("VLM OpenAI-compatible endpoint"):
                return "http://vlm.local:8001/v1"
            if label.startswith("VLM model id"):
                return ""
            raise AssertionError(label)

        with patch.object(
            installer.getpass,
            "getpass",
            side_effect=(
                "luxriot-password",
                "postgresql://api:secret@db/eva",
                "postgresql://audit:secret@db/eva",
                "postgresql://worker:secret@db/eva",
            ),
        ) as secret_prompt:
            values, _updates, missing = installer.prepare_env_values(
                resolution,
                environ={},
                non_interactive=False,
                input_fn=answer,
            )

        self.assertEqual(missing, [])
        self.assertEqual(secret_prompt.call_count, 4)
        self.assertEqual(values["EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL"], "http://127.0.0.1:1234/v1")
        self.assertEqual(values["EVOSSEARCH_LM_PROFILE_VLM_BASE_URL"], "http://vlm.local:8001/v1")

    def test_multi_vlm_profile_satisfies_existing_endpoint_and_model_contract(self):
        existing = dict(COMPLETE_ENV)
        existing.pop("EVOSSEARCH_LM_PROFILE_VLM_BASE_URL")
        existing.pop("EVOSSEARCH_LM_PROFILE_VLM_MODEL")
        existing["EVOSSEARCH_LM_PROFILE_VLM_A1_BASE_URL"] = "http://vlm-a:8001/v1"
        existing["EVOSSEARCH_LM_PROFILE_VLM_A1_MODEL"] = "qwen3-vl-4b"
        resolution = installer.EnvResolution(Path("/x/.env"), Path("/x/.env"), "", existing)

        _values, updates, missing = installer.prepare_env_values(
            resolution,
            environ={},
            non_interactive=True,
        )

        self.assertEqual(missing, [])
        self.assertNotIn("EVOSSEARCH_LM_PROFILE_VLM_BASE_URL", updates)
        self.assertNotIn("EVOSSEARCH_LM_PROFILE_VLM_MODEL", updates)
        self.assertNotIn("EVOSSEARCH_LM_VLM_PROFILE_ID", updates)

    def test_migrate_requires_distinct_privileged_dsn_and_never_falls_back_to_runtime(self):
        values = {"EVA_DATABASE_DSN": COMPLETE_ENV["EVA_DATABASE_DSN"]}
        updates = {}

        migration_dsn, source, error = installer.prepare_migration_dsn(
            values,
            updates,
            environ={},
            migrate=True,
            non_interactive=True,
        )

        self.assertIsNone(migration_dsn)
        self.assertIsNone(source)
        self.assertIn("EVA_INSTALL_MIGRATION_DSN", error)
        self.assertNotIn("EVA_MIGRATION_DATABASE_DSN", updates)

    def test_process_only_migration_dsn_is_distinct_redacted_and_not_persisted(self):
        values = {"EVA_DATABASE_DSN": COMPLETE_ENV["EVA_DATABASE_DSN"]}
        updates = {}
        privileged = "postgresql://migrator:MIGRATION-SECRET@db.internal/eva"

        migration_dsn, source, error = installer.prepare_migration_dsn(
            values,
            updates,
            environ={"EVA_INSTALL_MIGRATION_DSN": privileged},
            migrate=True,
            non_interactive=True,
        )

        self.assertEqual(migration_dsn, privileged)
        self.assertEqual(source, "EVA_INSTALL_MIGRATION_DSN (process-only)")
        self.assertIsNone(error)
        self.assertNotIn("EVA_MIGRATION_DATABASE_DSN", values)
        self.assertEqual(updates, {})

    def test_migration_dsn_equal_to_runtime_is_rejected(self):
        runtime = COMPLETE_ENV["EVA_DATABASE_DSN"]
        values = {"EVA_DATABASE_DSN": runtime}

        _migration_dsn, _source, error = installer.prepare_migration_dsn(
            values,
            {},
            environ={"EVA_INSTALL_MIGRATION_DSN": runtime},
            migrate=True,
            non_interactive=True,
        )

        self.assertEqual(error, "migration DSN must be distinct from runtime EVA_DATABASE_DSN")

    def test_apply_without_migration_dsn_fails_before_service_account_or_service_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=True)
            options.dry_run = False
            prepared = installer.prepare_install(options, environ={})

            with (
                patch.object(installer.os, "geteuid", return_value=0),
                patch.object(installer, "_ensure_service_account") as mutate_host,
            ):
                with self.assertRaisesRegex(
                    installer.InstallerError,
                    "Refusing --migrate without distinct",
                ):
                    installer.apply_install(prepared)

            mutate_host.assert_not_called()
            self.assertFalse((root / "backups").exists())

    def test_no_migrate_requires_no_privileged_dsn(self):
        migration_dsn, source, error = installer.prepare_migration_dsn(
            {"EVA_DATABASE_DSN": COMPLETE_ENV["EVA_DATABASE_DSN"]},
            {},
            environ={},
            migrate=False,
            non_interactive=True,
        )

        self.assertIsNone(migration_dsn)
        self.assertIsNone(source)
        self.assertIsNone(error)

    def test_systemd_template_uses_selected_paths_and_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            options = make_options(root, source, env_file=root / "site.env")
            template = options.unit_template.read_text(encoding="utf-8")

            rendered = installer.render_unit(template, options)

            self.assertIn(f"WorkingDirectory={options.app_dir}", rendered)
            self.assertIn(f"EnvironmentFile={options.env_file}", rendered)
            self.assertIn(
                f"Environment=EVOSSEARCH_CONFIG_ENV_FILE={options.env_file}",
                rendered,
            )
            self.assertIn("User=eva", rendered)
            self.assertNotRegex(rendered, r"@[A-Z_]+@")

    def test_preflight_rejects_luxriot_password_placeholder_without_echoing_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            placeholder_env = dict(COMPLETE_ENV)
            placeholder_env["EVOSSEARCH_LUXRIOT_PASSWORD"] = "changeme"
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(placeholder_env), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)

            prepared = installer.prepare_install(options, environ={})
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                installer.print_prepared(prepared)

            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]
            self.assertIn(
                "EVOSSEARCH_LUXRIOT_PASSWORD contains an obvious placeholder value",
                failures,
            )
            self.assertNotIn("changeme", stdout.getvalue().lower())

    def test_plan_reuses_existing_patch_migration_verify_and_rollback_mechanisms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(prefix="CUSTOM_SETTING=preserve"), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=True)

            prepared = installer.prepare_install(options, environ={})
            plan = "\n".join(action.description for action in prepared.actions)

            self.assertIn("preflight_patch.sh", plan)
            self.assertIn("install_patch.sh", plan)
            self.assertIn("Alembic current -> upgrade head -> current", plan)
            self.assertIn("verify_patch.sh", plan)
            self.assertIn("rollback.sh", plan)

    def test_fresh_install_without_wheelhouse_is_blocked_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(), encoding="utf-8")
            options = make_options(
                root,
                source,
                env_file=env_file,
                migrate=False,
                app_with_venv=False,
            )

            prepared = installer.prepare_install(options, environ={})
            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]

            self.assertTrue(any("fresh install requires bundle/wheelhouse" in message for message in failures))

    def test_apply_preflight_requires_root_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            options.dry_run = False

            with patch.object(installer.os, "geteuid", return_value=1000):
                prepared = installer.prepare_install(options, environ={})

            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]
            self.assertIn("--apply requires root (run with sudo)", failures)

    def test_latest_backup_refuses_path_outside_configured_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup_root = root / "backups"
            backup_root.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (backup_root / "LATEST").write_text(str(outside), encoding="utf-8")

            with self.assertRaisesRegex(installer.InstallerError, "outside configured backup root"):
                installer._latest_backup(backup_root)

    def test_second_apply_lock_fails_nonblocking_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_file = root / "installer.lock"
            mutation_marker = root / "mutated"

            with installer.install_lock(lock_file):
                with self.assertRaisesRegex(installer.InstallerError, "another EVA AI installer"):
                    with installer.install_lock(lock_file):
                        mutation_marker.write_text("should not happen", encoding="utf-8")

            self.assertFalse(mutation_marker.exists())

    def test_command_runner_redacts_secret_from_stdout_and_stderr(self):
        secret = "UNIQUE-INSTALLER-SECRET"
        runner = installer.CommandRunner([secret])
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            runner.run((
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); print(sys.argv[1], file=sys.stderr)",
                secret,
            ))

        self.assertNotIn(secret, stdout.getvalue())
        self.assertNotIn(secret, stderr.getvalue())
        self.assertIn("***", stdout.getvalue())
        self.assertIn("***", stderr.getvalue())


class OfflineInstallerCliTests(unittest.TestCase):
    def test_fresh_noninteractive_dry_run_uses_process_env_and_wheelhouse_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            bundle = root / "bundle"
            wheelhouse = bundle / "wheelhouse"
            wheelhouse.mkdir(parents=True)
            (wheelhouse / "offline-placeholder.whl").write_bytes(b"wheel")
            app = root / "new-app"
            env_file = root / "etc/eva-ai.env"
            unit_file = root / "eva-ai.service"
            process_env = {"PATH": os.environ.get("PATH", ""), **COMPLETE_ENV}

            completed = subprocess.run(
                (
                    sys.executable,
                    INSTALLER_PATH,
                    "--dry-run",
                    "--non-interactive",
                    "--source-dir", source,
                    "--bundle-dir", bundle,
                    "--app-dir", app,
                    "--env-file", env_file,
                    "--unit-file", unit_file,
                    "--backup-root", root / "backups",
                    "--no-migrate",
                    "--no-start",
                ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=process_env,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("offline wheelhouse contains 1 artifact(s)", completed.stdout)
            self.assertIn(f"create {env_file} with mode 0600", completed.stdout)
            self.assertFalse(app.exists())
            self.assertFalse(env_file.exists())
            self.assertFalse(unit_file.exists())
            for secret in ("EVO-SECRET-DO-NOT-PRINT", "API-SECRET", "AUDIT-SECRET", "WORKER-SECRET"):
                self.assertNotIn(secret, completed.stdout + completed.stderr)

    def test_dry_run_is_default_has_required_flags_and_does_not_mutate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            app = root / "app"
            python = app / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
            env_file = root / "eva-ai.env"
            original = env_text(prefix="CUSTOM_SETTING=preserve")
            env_file.write_text(original, encoding="utf-8")
            unit_file = root / "eva-ai.service"
            lock_file = root / "installer.lock"

            completed = subprocess.run(
                (
                    sys.executable,
                    INSTALLER_PATH,
                    "--non-interactive",
                    "--source-dir", source,
                    "--app-dir", app,
                    "--env-file", env_file,
                    "--unit-file", unit_file,
                    "--lock-file", lock_file,
                    "--backup-root", root / "backups",
                    "--no-migrate",
                    "--no-start",
                ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("MODE: DRY-RUN (no changes)", completed.stdout)
            self.assertIn("--apply", completed.stdout)
            self.assertIn("install_patch.sh", completed.stdout)
            self.assertEqual(env_file.read_text(encoding="utf-8"), original)
            self.assertFalse(unit_file.exists())
            self.assertFalse(lock_file.exists())
            for secret in ("EVO-SECRET-DO-NOT-PRINT", "API-SECRET", "AUDIT-SECRET", "WORKER-SECRET"):
                self.assertNotIn(secret, completed.stdout + completed.stderr)

    def test_dry_run_noninteractive_missing_env_is_blocked_without_creating_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            app = root / "app"
            python = app / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
            env_file = root / "missing.env"

            completed = subprocess.run(
                (
                    sys.executable,
                    INSTALLER_PATH,
                    "--dry-run",
                    "--non-interactive",
                    "--source-dir", source,
                    "--app-dir", app,
                    "--env-file", env_file,
                    "--unit-file", root / "eva-ai.service",
                    "--backup-root", root / "backups",
                    "--no-migrate",
                    "--no-start",
                ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"PATH": os.environ.get("PATH", "")},
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("required configuration keys are missing", completed.stdout)
            self.assertIn("no changes made", completed.stderr)
            self.assertFalse(env_file.exists())

    def test_help_exposes_dry_run_apply_env_and_noninteractive_contract(self):
        completed = subprocess.run(
            (sys.executable, INSTALLER_PATH, "--help"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        for flag in ("--dry-run", "--apply", "--env-file", "--non-interactive"):
            self.assertIn(flag, completed.stdout)


if __name__ == "__main__":
    unittest.main()
