from __future__ import annotations

import importlib.util
import inspect
import io
import os
import stat
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
    "EVOSSEARCH_ARCHIVE_RETENTION_ENABLED": "true",
}


def env_text(values=None, *, prefix=""):
    rows = [prefix] if prefix else []
    for key, value in (values or COMPLETE_ENV).items():
        rows.append(f'{key}="{value}"')
    return "\n".join(rows) + "\n"


def make_siglip_cache(cache_root: Path) -> Path:
    snapshot = (
        cache_root
        / "models--google--siglip2-base-patch16-224/snapshots"
        / "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"test")
    return snapshot


def make_source(root: Path) -> Path:
    source = root / "source"
    for relative in (
        "migrations",
        "react-ui/dist",
        "static/js",
        "templates",
        "scripts/install_assets",
    ):
        (source / relative).mkdir(parents=True, exist_ok=True)
    files = {
        # Stay in lockstep with the real tree: the installer's expected
        # version now derives from the repo VERSION file.
        "VERSION": (ROOT / "VERSION").read_text(encoding="utf-8"),
        "run_prod.sh": "#!/bin/sh\n",
        "wsgi.py": "app = None\n",
        "requirements.txt": "example==1\n",
        "alembic.ini": "[alembic]\n",
        "static/js/app.js": "// static\n",
        "templates/index.html": "<!doctype html>\n",
        "react-ui/dist/index.html": "<!doctype html><div id=\"root\"></div>\n",
        "scripts/preflight_patch.sh": "#!/bin/sh\nexit 0\n",
        "scripts/install_patch.sh": "#!/bin/sh\nexit 0\n",
        "scripts/install_media_runtime.sh": "#!/bin/sh\nexit 0\n",
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
    make_siglip_cache(source / "models/huggingface")
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
    make_siglip_cache(bundle / "models/huggingface")
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
    def test_release_identity_and_schema_match_the_current_tree(self):
        self.assertEqual(
            installer.EXPECTED_VERSION,
            (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        )
        self.assertEqual(installer.EXPECTED_SCHEMA, "20260805_0013")

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

    def test_fresh_runtime_directories_are_bounded_owned_and_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {
                "EVOSSEARCH_MODEL_CACHE_DIR": str(root / "models/hf"),
                "EVOSSEARCH_OPENAI_CLIP_CACHE_DIR": str(root / "models/clip"),
                "EVOSSEARCH_INFERENCE_QUEUE_SPOOL_DIR": str(root / "spool"),
            }
            identity = type("Identity", (), {"pw_uid": 1234})()
            group = type("Group", (), {"gr_gid": 1235})()
            with (
                patch.object(installer.pwd, "getpwnam", return_value=identity),
                patch.object(installer.grp, "getgrnam", return_value=group),
                patch.object(installer.os, "chown") as chown,
            ):
                created = installer._ensure_runtime_directories(
                    values,
                    user="eva",
                    group="eva",
                )

            self.assertEqual(len(created), 3)
            self.assertEqual(chown.call_count, 3)
            for path in created:
                self.assertTrue(path.is_dir())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o750)

    def test_preinstall_file_backup_preserves_owner_and_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "eva-ai.env"
            source.write_text("SETTING=value\n", encoding="utf-8")
            source_stat = source.stat()

            with patch.object(installer.os, "chown") as chown:
                backup = installer._backup_file(source)

            self.assertIsNotNone(backup)
            assert backup is not None
            self.assertEqual(backup.read_text(encoding="utf-8"), "SETTING=value\n")
            chown.assert_called_once_with(
                backup,
                source_stat.st_uid,
                source_stat.st_gid,
            )

    def test_adopt_updates_release_identity_and_appends_siglip_runtime_defaults(self):
        existing = dict(COMPLETE_ENV)
        existing.update({
            "EVOSSEARCH_APP_VERSION": "β 0.8.1",
            "EVOSSEARCH_HOST": "10.20.30.40",
        })
        raw = env_text(existing, prefix="# preserve site settings")
        resolution = installer.EnvResolution(Path("/x/.env"), Path("/x/.env"), raw, existing)

        values, updates, missing = installer.prepare_env_values(
            resolution,
            environ={},
            non_interactive=True,
        )
        rendered = installer.render_env_update(raw, updates)

        self.assertEqual(missing, [])
        self.assertEqual(
            updates,
            {
                "EVOSSEARCH_APP_VERSION": installer.EXPECTED_VERSION,
                "EVOSSEARCH_UI_MODE": "react",
                "EVOSSEARCH_MODEL_CACHE_DIR": "/var/lib/eva-ai/models/huggingface",
                "EVOSSEARCH_PRODUCTION_CLIP_MODEL": "google/siglip2-base-patch16-224",
                "EVOSSEARCH_CLIP_MODEL": "google/siglip2-base-patch16-224",
                "EVOSSEARCH_CLIP_MODEL_REVISION": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED": "false",
            },
        )
        self.assertEqual(values["EVOSSEARCH_APP_VERSION"], installer.EXPECTED_VERSION)
        self.assertEqual(values["EVOSSEARCH_UI_MODE"], "react")
        self.assertIn(f"EVOSSEARCH_APP_VERSION='{installer.EXPECTED_VERSION}'", rendered)
        self.assertIn("EVOSSEARCH_UI_MODE='react'", rendered)
        self.assertIn('EVOSSEARCH_HOST="10.20.30.40"', rendered)

    def test_adopt_preserves_explicit_embedding_model_and_cache(self):
        existing = dict(COMPLETE_ENV)
        existing.update({
            "EVOSSEARCH_MODEL_CACHE_DIR": "/srv/eva/models",
            "EVOSSEARCH_PRODUCTION_CLIP_MODEL": "site/model",
            "EVOSSEARCH_CLIP_MODEL": "site/model",
            "EVOSSEARCH_CLIP_MODEL_REVISION": "site-revision",
            "EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED": "true",
        })
        raw = env_text(existing)
        resolution = installer.EnvResolution(Path("/x/.env"), Path("/x/.env"), raw, existing)

        values, updates, missing = installer.prepare_env_values(
            resolution,
            environ={},
            non_interactive=True,
        )

        self.assertEqual(missing, [])
        self.assertEqual(values["EVOSSEARCH_MODEL_CACHE_DIR"], "/srv/eva/models")
        self.assertEqual(values["EVOSSEARCH_CLIP_MODEL"], "site/model")
        self.assertEqual(values["EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED"], "true")
        self.assertNotIn("EVOSSEARCH_MODEL_CACHE_DIR", updates)
        self.assertNotIn("EVOSSEARCH_CLIP_MODEL", updates)

    def test_legacy_adopt_disables_unconfigured_archive_retention(self):
        existing = dict(COMPLETE_ENV)
        existing.pop("EVOSSEARCH_ARCHIVE_RETENTION_ENABLED")
        raw = env_text(existing, prefix="# legacy site without retention policy")
        resolution = installer.EnvResolution(Path("/x/.env"), Path("/x/.env"), raw, existing)

        values, updates, missing = installer.prepare_env_values(
            resolution,
            environ={},
            non_interactive=True,
        )
        rendered = installer.render_env_update(raw, updates)

        self.assertEqual(missing, [])
        self.assertEqual(values["EVOSSEARCH_ARCHIVE_RETENTION_ENABLED"], "false")
        self.assertEqual(updates["EVOSSEARCH_ARCHIVE_RETENTION_ENABLED"], "false")
        self.assertIn("EVOSSEARCH_ARCHIVE_RETENTION_ENABLED='false'", rendered)

    def test_legacy_adopt_preserves_explicit_archive_retention_window(self):
        existing = dict(COMPLETE_ENV)
        existing.pop("EVOSSEARCH_ARCHIVE_RETENTION_ENABLED")
        existing["EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS"] = "30"
        raw = env_text(existing)
        resolution = installer.EnvResolution(Path("/x/.env"), Path("/x/.env"), raw, existing)

        values, updates, missing = installer.prepare_env_values(
            resolution,
            environ={},
            non_interactive=True,
        )

        self.assertEqual(missing, [])
        self.assertNotIn("EVOSSEARCH_ARCHIVE_RETENTION_ENABLED", updates)
        self.assertEqual(values["EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS"], "30")

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
        self.assertEqual(values["EVOSSEARCH_OFFLINE_MODE"], "true")
        self.assertEqual(values["EVOSSEARCH_TRUSTED_PROXY_HOPS"], "1")
        self.assertEqual(values["EVOSSEARCH_PROBE_POS_FLOOR_DEFAULT"], "0.05")
        self.assertEqual(values["EVOSSEARCH_PROBE_MARGIN_DEFAULT"], "0.02")
        self.assertEqual(
            values["EVOSSEARCH_CLIP_MODEL"],
            "google/siglip2-base-patch16-224",
        )
        self.assertEqual(values["EVOSSEARCH_INFERENCE_QUEUE_ENABLED"], "true")
        self.assertEqual(values["EVOSSEARCH_INFERENCE_WORKER_COUNT"], "1")
        rendered = installer.render_env_update("", updates)
        self.assertIn("EVO-SECRET-DO-NOT-PRINT", rendered)

    def test_env_dsn_references_are_resolved_in_memory_but_raw_file_is_preserved(self):
        existing = dict(COMPLETE_ENV)
        existing["EVOSSEARCH_APP_VERSION"] = installer.EXPECTED_VERSION
        existing["EVOSSEARCH_UI_MODE"] = "react"
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
        self.assertNotIn("EVA_DATABASE_DSN", updates)
        self.assertEqual(
            updates["EVOSSEARCH_MODEL_CACHE_DIR"],
            "/var/lib/eva-ai/models/huggingface",
        )
        self.assertIn("${EVA_API_PASSWORD}", installer.render_env_update(raw, updates))

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

    def test_selected_nonstandard_agent_profile_satisfies_agent_contract(self):
        existing = dict(COMPLETE_ENV)
        existing.pop("EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL")
        existing.pop("EVOSSEARCH_LM_PROFILE_AGENT_MODEL")
        existing["EVOSSEARCH_LM_PROFILES"] = "chat,vlm"
        existing["EVOSSEARCH_LM_AGENT_PROFILE_ID"] = "chat"
        existing["EVOSSEARCH_LM_PROFILE_CHAT_BASE_URL"] = "http://lm.internal:1234/v1"
        existing["EVOSSEARCH_LM_PROFILE_CHAT_MODEL"] = "qwen3.5-9b-mtp"
        resolution = installer.EnvResolution(Path("/x/.env"), Path("/x/.env"), "", existing)

        _values, updates, missing = installer.prepare_env_values(
            resolution,
            environ={},
            non_interactive=True,
        )

        self.assertEqual(missing, [])
        self.assertNotIn("EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL", updates)
        self.assertNotIn("EVOSSEARCH_LM_PROFILE_AGENT_MODEL", updates)

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

    def test_interactive_migration_dsn_is_process_only_and_not_persisted(self):
        values = {"EVA_DATABASE_DSN": COMPLETE_ENV["EVA_DATABASE_DSN"]}
        updates = {}
        privileged = "postgresql://migrator:INTERACTIVE-SECRET@db.internal/eva"

        with patch.object(installer.getpass, "getpass", return_value=privileged):
            migration_dsn, source, error = installer.prepare_migration_dsn(
                values,
                updates,
                environ={},
                migrate=True,
                non_interactive=False,
            )

        self.assertEqual(migration_dsn, privileged)
        self.assertEqual(source, "interactive process-only value")
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

    def test_apply_proves_revision_table_privileges_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=True)
            options.dry_run = False
            prepared = installer.prepare_install(
                options,
                environ={
                    "EVA_INSTALL_MIGRATION_DSN": (
                        "postgresql://migrator:MIGRATION-SECRET@db.internal/eva"
                    )
                },
            )

            with (
                patch.object(installer.os, "geteuid", return_value=0),
                patch.object(
                    installer,
                    "_verify_migration_capability",
                    side_effect=installer.InstallerError("revision privilege denied"),
                ) as verify_db,
                patch.object(installer, "_ensure_service_account") as mutate_host,
            ):
                with self.assertRaisesRegex(installer.InstallerError, "revision privilege denied"):
                    installer.apply_install(prepared)

            verify_db.assert_called_once()
            mutate_host.assert_not_called()
            self.assertFalse((root / "backups").exists())

    def test_migration_capability_check_rolls_back_noop_revision_update(self):
        runner = type("Runner", (), {"commands": []})()

        def run(command, **_kwargs):
            runner.commands.append([str(item) for item in command])
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        runner.run = run
        installer._verify_migration_capability(
            runner,
            "postgresql://migrator:secret@db.internal/eva",
        )

        command = runner.commands[0]
        sql = command[command.index("--command") + 1]
        self.assertIn("SELECT version_num FROM public.alembic_version", sql)
        self.assertIn("UPDATE public.alembic_version SET version_num = version_num", sql)
        self.assertIn("SET LOCAL ROLE eva_owner", sql)
        self.assertIn("EVA schemas are absent or are not owned by eva_owner", sql)
        self.assertIn("CREATE TABLE archive.__eva_migration_preflight", sql)
        self.assertIn("ROLLBACK", sql)

    def test_apply_staging_failure_restores_preexisting_env_without_backup_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            original = env_text()
            env_file.write_text(original, encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            options.dry_run = False
            options.unit_file.write_text("[Service]\nUser=site-eva\n", encoding="utf-8")
            prepared = installer.prepare_install(options, environ={})
            prepared.values["NEW_SITE_KEY"] = "staged"
            prepared.updates["NEW_SITE_KEY"] = "staged"

            with (
                patch.object(installer.os, "geteuid", return_value=0),
                patch.object(
                    installer.CommandRunner,
                    "run",
                    side_effect=installer.InstallerError("preflight failed"),
                ),
            ):
                with self.assertRaisesRegex(installer.InstallerError, "preflight failed"):
                    installer.apply_install(prepared)

            self.assertEqual(env_file.read_text(encoding="utf-8"), original)
            self.assertFalse(list(root.glob("eva-ai.env.preinstall-*.bak")))

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

    def test_explicit_live_evo_check_can_accept_heuristic_matched_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            site_env = dict(COMPLETE_ENV)
            site_env["EVOSSEARCH_LUXRIOT_PASSWORD"] = "changeme"
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(site_env), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            options.verify_luxriot_credential = True

            with patch.object(
                installer,
                "_verify_luxriot_credential",
                return_value=(True, "http_200"),
            ) as verify:
                prepared = installer.prepare_install(options, environ={})

            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]
            warnings = [finding.message for finding in prepared.findings if finding.level == "WARN"]
            self.assertFalse(any("LUXRIOT_PASSWORD" in message for message in failures))
            self.assertTrue(any("authenticated read-only Evo" in message for message in warnings))
            verify.assert_called_once()

    def test_failed_live_evo_check_keeps_placeholder_gate_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            site_env = dict(COMPLETE_ENV)
            site_env["EVOSSEARCH_LUXRIOT_PASSWORD"] = "changeme"
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(site_env), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            options.verify_luxriot_credential = True

            with patch.object(
                installer,
                "_verify_luxriot_credential",
                return_value=(False, "HTTPError"),
            ):
                prepared = installer.prepare_install(options, environ={})

            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]
            self.assertTrue(any("authenticated Evo check failed" in message for message in failures))
            self.assertFalse(any("changeme" in message.lower() for message in failures))

    def test_automatic_rollback_restores_database_and_previous_service_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=True)
            prepared = installer.prepare_install(
                options,
                environ={
                    "EVA_INSTALL_MIGRATION_DSN": (
                        "postgresql://migrator:MIGRATION-SECRET@db.internal/eva"
                    )
                },
            )
            calls = []

            class Runner:
                def run(self, command, *, env=None, cwd=None):
                    calls.append(([str(item) for item in command], dict(env or {})))
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            backup = root / "backups/patch-1"
            backup.mkdir(parents=True)
            self.assertTrue(installer._automatic_rollback(prepared, Runner(), backup))

            command, environment = calls[0]
            self.assertIn("--restore-db", command)
            self.assertNotIn("--no-start", command)
            self.assertEqual(environment["EVA_PATCH_CONFIRM_DB_RESTORE"], "yes")
            self.assertEqual(environment["EVA_PATCH_PG_DSN"], prepared.migration_dsn)

    def test_verified_code_only_adopt_warns_but_does_not_block_existing_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            placeholder_env = dict(COMPLETE_ENV)
            placeholder_env["EVOSSEARCH_LUXRIOT_PASSWORD"] = "changeme"
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(placeholder_env), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            (options.app_dir / "VERSION").write_text("β 0.8.3\n", encoding="utf-8")
            options.adopt_existing_config = True

            prepared = installer.prepare_install(options, environ={})
            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]
            warnings = [finding.message for finding in prepared.findings if finding.level == "WARN"]

            self.assertNotIn(
                "EVOSSEARCH_LUXRIOT_PASSWORD contains an obvious placeholder value",
                failures,
            )
            self.assertIn(
                "EVOSSEARCH_LUXRIOT_PASSWORD looks like a placeholder but is preserved by verified code-only adopt",
                warnings,
            )

    def test_verified_adopt_does_not_weaken_fresh_install_or_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            placeholder_env = dict(COMPLETE_ENV)
            placeholder_env["EVOSSEARCH_LUXRIOT_PASSWORD"] = "changeme"
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(placeholder_env), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            options.adopt_existing_config = True

            prepared = installer.prepare_install(options, environ={})
            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]
            self.assertIn(
                "EVOSSEARCH_LUXRIOT_PASSWORD contains an obvious placeholder value",
                failures,
            )

            (options.app_dir / "VERSION").write_text("β 0.8.3\n", encoding="utf-8")
            options.migrate = True
            prepared = installer.prepare_install(
                options,
                environ={
                    "EVA_INSTALL_MIGRATION_DSN": "postgresql://migrator:secret@db.internal/eva",
                },
            )
            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]
            self.assertIn(
                "EVOSSEARCH_LUXRIOT_PASSWORD contains an obvious placeholder value",
                failures,
            )

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
            self.assertIn("checksummed offline FFmpeg runtime", plan)
            self.assertIn("Alembic current -> upgrade head -> current", plan)
            self.assertIn("verify_patch.sh", plan)
            self.assertIn("rollback.sh", plan)

    def test_included_media_runtime_is_verified_before_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            runtime = options.bundle_dir / "runtime"
            runtime.mkdir(parents=True)

            prepared = installer.prepare_install(options, environ={})
            failures = [finding.message for finding in prepared.findings if finding.level == "FAIL"]

            self.assertTrue(any("bundled media runtime path is missing" in row for row in failures))

    def test_media_runtime_install_precedes_migration_and_service_start(self):
        source = inspect.getsource(installer.apply_install)
        self.assertLess(source.index("_install_media_runtime"), source.index("alembic ="))
        self.assertLess(
            source.index("_install_media_runtime"),
            source.index('(\"systemctl\", \"enable\"'),
        )

    def test_media_runtime_uses_opencv_overlay_only_when_venv_needs_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            (options.bundle_dir / "runtime").mkdir()
            helper = options.bundle_dir / "scripts/install_media_runtime.sh"
            helper.parent.mkdir(parents=True)
            helper.write_text("#!/bin/sh\n", encoding="utf-8")
            helper.chmod(0o755)
            prepared = installer.prepare_install(options, environ={})
            calls = []

            class Runner:
                def run(self, command, **_kwargs):
                    calls.append([str(item) for item in command])
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch.object(installer, "_venv_has_healthy_opencv", return_value=False):
                self.assertTrue(installer._install_media_runtime(prepared, Runner()))
            self.assertIn("--with-opencv-overlay", calls[-1])

            with patch.object(installer, "_venv_has_healthy_opencv", return_value=True):
                self.assertTrue(installer._install_media_runtime(prepared, Runner()))
            self.assertNotIn("--with-opencv-overlay", calls[-1])

    def test_adopt_plan_preserves_existing_systemd_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            env_file = root / "eva-ai.env"
            env_file.write_text(env_text(), encoding="utf-8")
            options = make_options(root, source, env_file=env_file, migrate=False)
            existing_unit = "[Service]\nUser=site-eva\nProtectSystem=full\n"
            options.unit_file.write_text(existing_unit, encoding="utf-8")

            prepared = installer.prepare_install(options, environ={})
            plan = "\n".join(action.description for action in prepared.actions)

            self.assertIn("preserve existing systemd unit", plan)
            self.assertIn("preserve the account selected by the existing systemd unit", plan)
            self.assertEqual(options.unit_file.read_text(encoding="utf-8"), existing_unit)

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
    def test_bundle_builder_excludes_local_runtime_and_private_release_artifacts(self):
        builder = (ROOT / "scripts" / "build_patch_bundle.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('"--exclude=.local"', builder)
        self.assertIn('"--exclude=.env"', builder)
        self.assertIn('"--exclude=.env.*"', builder)

    def test_fresh_noninteractive_dry_run_uses_process_env_and_wheelhouse_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            bundle = root / "bundle"
            wheelhouse = bundle / "wheelhouse"
            wheelhouse.mkdir(parents=True)
            (wheelhouse / "offline-placeholder.whl").write_bytes(b"wheel")
            make_siglip_cache(bundle / "models/huggingface")
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
