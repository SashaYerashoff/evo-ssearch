from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


deploy = _load("eva_offline_deploy", ROOT / "scripts/eva_offline_deploy.py")
report = _load("eva_deployment_report", ROOT / "scripts/eva_deployment_report.py")


def test_auto_detection_prefers_systemd_working_directory_and_environment(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "VERSION").write_text("β 0.8.5\n", encoding="utf-8")
    env_file = tmp_path / "eva-ai.env"
    env_file.write_text("EVOSSEARCH_PORT=5050\n", encoding="utf-8")
    unit = tmp_path / "eva-ai.service"
    properties = {
        "LoadState": "loaded",
        "WorkingDirectory": str(app),
        "EnvironmentFiles": f"{env_file} (ignore_errors=no)",
        "FragmentPath": str(unit),
        "User": "site-eva",
        "Group": "site-eva",
    }

    with patch.object(deploy, "_systemd_property", side_effect=lambda _service, key: properties[key]):
        detected = deploy.detect_existing()

    assert detected is not None
    assert detected.app_dir == app
    assert detected.env_file == env_file
    assert detected.base_url == "http://127.0.0.1:5050"
    assert detected.service_user == "site-eva"


def test_auto_detection_returns_fresh_when_no_unit_or_install():
    with patch.object(deploy, "_systemd_property", return_value=""):
        assert deploy.detect_existing() is None
        assert deploy.detect_deployment().mode == "install"


def test_auto_detection_resumes_journaled_incomplete_fresh_install(tmp_path, capsys):
    install_root = tmp_path / "eva-ai"
    app = install_root / "app"
    app.mkdir(parents=True)
    (app / "VERSION").write_text("β 0.8.7\n", encoding="utf-8")
    state = tmp_path / "install-state.json"
    state.write_text(
        json.dumps(
            {
                "format": 1,
                "status": "failed",
                "failed_phase": "python_environments",
                "target": {"install_root": str(install_root)},
            }
        ),
        encoding="utf-8",
    )
    properties = {
        "LoadState": "not-found",
        "WorkingDirectory": str(app),
    }

    with patch.object(
        deploy,
        "_systemd_property",
        side_effect=lambda _service, key: properties.get(key, ""),
    ):
        assert deploy.detect_existing(installer_state=state) is None
        detected = deploy.detect_deployment(installer_state=state)

    assert detected.mode == "resume"
    assert "resuming INSTALL engine" in capsys.readouterr().out


def test_auto_detection_does_not_trust_unjournaled_app_without_env(tmp_path):
    app = tmp_path / "eva-ai" / "app"
    app.mkdir(parents=True)
    (app / "VERSION").write_text("β 0.8.7\n", encoding="utf-8")
    missing_state = tmp_path / "missing-install-state.json"
    properties = {
        "LoadState": "not-found",
        "WorkingDirectory": str(app),
    }

    with (
        patch.object(
            deploy,
            "_systemd_property",
            side_effect=lambda _service, key: properties.get(key, ""),
        ),
        pytest.raises(deploy.DeployError, match="environment file was not found"),
    ):
        deploy.detect_existing(installer_state=missing_state)


def test_auto_detection_resumes_loaded_service_with_failed_journal(tmp_path, capsys):
    app = tmp_path / "eva-ai" / "app"
    app.mkdir(parents=True)
    (app / "VERSION").write_text("β 0.8.7\n", encoding="utf-8")
    state = tmp_path / "install-state.json"
    state.write_text(
        json.dumps(
            {
                "format": 1,
                "status": "failed",
                "target": {"install_root": str(app.parent)},
            }
        ),
        encoding="utf-8",
    )
    properties = {
        "LoadState": "loaded",
        "WorkingDirectory": str(app),
        "EnvironmentFiles": "",
    }

    with patch.object(
        deploy,
        "_systemd_property",
        side_effect=lambda _service, key: properties.get(key, ""),
    ):
        detected = deploy.detect_deployment(installer_state=state)

    assert detected.mode == "resume"
    assert detected.existing is None
    assert detected.incomplete is not None
    assert detected.incomplete.install_root == app.parent
    assert "resuming INSTALL engine" in capsys.readouterr().out


def test_auto_detection_does_not_apply_stale_failed_journal_to_other_install(tmp_path):
    journal_root = tmp_path / "failed-install"
    app = tmp_path / "live-install" / "app"
    app.mkdir(parents=True)
    (app / "VERSION").write_text("beta 0.8.7\n", encoding="utf-8")
    env_file = tmp_path / "eva-ai.env"
    env_file.write_text("EVOSSEARCH_PORT=5000\n", encoding="utf-8")
    state = tmp_path / "install-state.json"
    state.write_text(
        json.dumps(
            {
                "format": 1,
                "status": "failed",
                "target": {"install_root": str(journal_root)},
            }
        ),
        encoding="utf-8",
    )
    properties = {
        "LoadState": "loaded",
        "WorkingDirectory": str(app),
        "EnvironmentFiles": str(env_file),
        "FragmentPath": str(tmp_path / "eva-ai.service"),
        "User": "eva",
        "Group": "eva",
    }

    with patch.object(
        deploy,
        "_systemd_property",
        side_effect=lambda _service, key: properties.get(key, ""),
    ):
        detected = deploy.detect_deployment(installer_state=state)

    assert detected.mode == "update"
    assert detected.existing is not None


def test_auto_detection_completed_journal_remains_update(tmp_path):
    install_root = tmp_path / "eva-ai"
    app = install_root / "app"
    app.mkdir(parents=True)
    (app / "VERSION").write_text("beta 0.8.7\n", encoding="utf-8")
    env_file = tmp_path / "eva-ai.env"
    env_file.write_text("EVOSSEARCH_PORT=5000\n", encoding="utf-8")
    state = tmp_path / "install-state.json"
    state.write_text(
        json.dumps(
            {
                "format": 1,
                "status": "complete",
                "target": {"install_root": str(install_root)},
            }
        ),
        encoding="utf-8",
    )
    properties = {
        "LoadState": "loaded",
        "WorkingDirectory": str(app),
        "EnvironmentFiles": str(env_file),
        "FragmentPath": str(tmp_path / "eva-ai.service"),
        "User": "eva",
        "Group": "eva",
    }

    with patch.object(
        deploy,
        "_systemd_property",
        side_effect=lambda _service, key: properties.get(key, ""),
    ):
        detected = deploy.detect_deployment(installer_state=state)

    assert detected.mode == "update"
    assert detected.existing is not None


def test_auto_mode_dispatches_interrupted_install_to_resume_engine(tmp_path, capsys):
    bundle = tmp_path / "bundle"
    (bundle / "repo").mkdir(parents=True)
    interrupted = deploy.IncompleteFreshInstall(
        install_root=Path("/opt/eva-ai"),
        status="failed",
        failed_phase="operator_runtime_canary",
    )
    detection = deploy.DeploymentDetection(
        mode="resume",
        incomplete=interrupted,
    )

    with (
        patch.object(deploy, "_verify_bundle"),
        patch.object(deploy, "detect_deployment", return_value=detection),
        patch.object(deploy, "_fresh") as fresh,
        patch.object(deploy, "_update") as update,
    ):
        result = deploy.main(("--bundle-root", str(bundle), "--yes"))

    assert result == 0
    fresh.assert_called_once_with(bundle.resolve(), assume_yes=True, passthrough=[])
    update.assert_not_called()
    assert "Mode:   RESUME" in capsys.readouterr().out


def test_explicit_update_rejects_interrupted_fresh_install(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "repo").mkdir(parents=True)
    detection = deploy.DeploymentDetection(
        mode="resume",
        incomplete=deploy.IncompleteFreshInstall(
            install_root=Path("/opt/eva-ai"),
            status="failed",
            failed_phase="systemd_units",
        ),
    )

    with (
        patch.object(deploy, "_verify_bundle"),
        patch.object(deploy, "detect_deployment", return_value=detection),
        patch.object(deploy, "_fresh") as fresh,
        patch.object(deploy, "_update") as update,
    ):
        result = deploy.main(
            ("--bundle-root", str(bundle), "--mode", "update", "--yes")
        )

    assert result == 1
    fresh.assert_not_called()
    update.assert_not_called()


def test_update_compatibility_accepts_ubuntu_26_python_314(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "offline-dependencies.json").write_text(
        json.dumps(
            {
                "target": {"architecture": "amd64"},
                "update_compatibility": {
                    "os_releases": ["24.04", "26.04"],
                    "python_versions": ["3.12", "3.13", "3.14"],
                },
            }
        ),
        encoding="utf-8",
    )
    os_release = tmp_path / "os-release"
    os_release.write_text('ID=ubuntu\nVERSION_ID="26.04"\n', encoding="utf-8")
    deployment = deploy.ExistingDeployment(
        service="eva-ai",
        app_dir=tmp_path / "app",
        env_file=tmp_path / "eva-ai.env",
        unit_file=tmp_path / "eva-ai.service",
        service_user="eva",
        service_group="eva",
        base_url="http://127.0.0.1:5000",
    )

    with (
        patch.object(deploy.platform, "machine", return_value="x86_64"),
        patch.object(deploy, "_deployment_python_version", return_value="3.14"),
    ):
        deploy._assert_update_compatibility(
            bundle,
            deployment,
            os_release_path=os_release,
        )


def test_update_compatibility_rejects_unbundled_python_before_mutation(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "offline-dependencies.json").write_text(
        json.dumps(
            {
                "target": {"architecture": "amd64"},
                "update_compatibility": {
                    "os_releases": ["24.04", "26.04"],
                    "python_versions": ["3.12", "3.13"],
                },
            }
        ),
        encoding="utf-8",
    )
    os_release = tmp_path / "os-release"
    os_release.write_text("ID=ubuntu\nVERSION_ID=24.04\n", encoding="utf-8")
    deployment = deploy.ExistingDeployment(
        service="eva-ai",
        app_dir=tmp_path / "app",
        env_file=tmp_path / "eva-ai.env",
        unit_file=tmp_path / "eva-ai.service",
        service_user="eva",
        service_group="eva",
        base_url="http://127.0.0.1:5000",
    )

    with (
        patch.object(deploy.platform, "machine", return_value="x86_64"),
        patch.object(deploy, "_deployment_python_version", return_value="3.14"),
        pytest.raises(deploy.DeployError, match="CPython 3.14"),
    ):
        deploy._assert_update_compatibility(
            bundle,
            deployment,
            os_release_path=os_release,
        )


def test_common_bundle_verification_catches_corruption_before_either_path(
    tmp_path, capsys
):
    bundle = tmp_path / "bundle"
    critical_file = bundle / "repo" / "VERSION"
    required_files = (
        bundle / "SOURCE_REVISION.json",
        bundle / "START_EVA_AI.sh",
        bundle / "eva_offline_deploy.py",
        bundle / "offline_bundle_dependencies.py",
        bundle / "offline-dependencies.json",
        bundle / "install_port_appliance.py",
        bundle / "migration-plans" / "0006-to-0013.sql",
        bundle / "apt" / "Packages.gz",
        bundle / "repo" / "react-ui" / "dist" / "index.html",
        bundle / "repo" / "requirements-cuda.txt",
        bundle / "repo" / "scripts" / "database_preservation_guard.py",
        bundle / "repo" / "scripts" / "pg_with_dsn.py",
        bundle / "repo" / "migrations" / "versions" / "20260801_0011_incidents.py",
        bundle / "repo" / "migrations" / "versions" / "20260805_0012_incident_temporal_memory.py",
        bundle / "repo" / "migrations" / "versions" / "20260805_0013_archive_source_channel_page_index.py",
    )
    for path in (*required_files, critical_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload\n", encoding="utf-8")
    (bundle / "wheelhouse").mkdir()
    digest = hashlib.sha256(critical_file.read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "format": 2,
                "release_flavor": deploy.EXPECTED_FLAVOR,
                "schema_head": deploy.EXPECTED_SCHEMA,
                "installation_modes": ["fresh", "resume", "update", "report"],
                "critical_sha256": {"repo/VERSION": digest},
            }
        ),
        encoding="utf-8",
    )

    with patch.object(deploy, "verify_dependencies"):
        deploy._verify_bundle(bundle)
    output = capsys.readouterr().out
    assert "Verifying offline bundle payload" in output
    assert "payload verification 100%" in output
    assert "Offline bundle payload verification: OK" in output
    critical_file.write_text("corrupted\n", encoding="utf-8")

    try:
        with patch.object(deploy, "verify_dependencies"):
            deploy._verify_bundle(bundle)
    except deploy.DeployError as exc:
        assert "Checksum mismatch" in str(exc)
    else:
        raise AssertionError("corrupted critical file was accepted")


def test_common_bundle_verification_requires_update_pack_identity_in_critical_map(tmp_path):
    bundle = tmp_path / "bundle"
    critical_file = bundle / "repo" / "VERSION"
    required_files = (
        bundle / "SOURCE_REVISION.json",
        bundle / "START_EVA_AI.sh",
        bundle / "eva_offline_deploy.py",
        bundle / "offline_bundle_dependencies.py",
        bundle / "offline-dependencies.json",
        bundle / "install_port_appliance.py",
        bundle / "migration-plans" / "0006-to-0013.sql",
        bundle / "apt" / "Packages.gz",
        bundle / "repo" / "react-ui" / "dist" / "index.html",
        bundle / "repo" / "requirements-cuda.txt",
        bundle / "repo" / "scripts" / "database_preservation_guard.py",
        bundle / "repo" / "scripts" / "pg_with_dsn.py",
        bundle / "repo" / "migrations" / "versions" / "20260801_0011_incidents.py",
        bundle / "repo" / "migrations" / "versions" / "20260805_0012_incident_temporal_memory.py",
        bundle / "repo" / "migrations" / "versions" / "20260805_0013_archive_source_channel_page_index.py",
    )
    for path in (*required_files, critical_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload\n", encoding="utf-8")
    (bundle / "wheelhouse").mkdir()
    digest = hashlib.sha256(critical_file.read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "format": 2,
                "release_flavor": deploy.EXPECTED_FLAVOR,
                "schema_head": deploy.EXPECTED_SCHEMA,
                "installation_modes": ["fresh", "resume", "update", "report"],
                "critical_sha256": {"repo/VERSION": digest},
                "update_packages": [
                    {
                        "name": "georgia",
                        "archive": "updates/georgia/update.tar.gz",
                        "archive_sha256": "a" * 64,
                        "checksum": "updates/georgia/update.tar.gz.sha256",
                        "expanded_manifest": "updates/georgia/update/manifest.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with patch.object(deploy, "verify_dependencies"):
        try:
            deploy._verify_bundle(bundle)
        except deploy.DeployError as exc:
            assert "has no critical archive" in str(exc)
        else:
            raise AssertionError("unbound update package was accepted")


def test_fresh_child_receives_content_bound_preflight_stamp(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "install_port_appliance.py").write_text("# installer\n", encoding="utf-8")
    manifest = bundle / "manifest.json"
    manifest.write_text('{"format": 2}\n', encoding="utf-8")
    calls = []

    def capture(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return type("Completed", (), {"returncode": 0})()

    with (
        patch.object(deploy.os, "geteuid", return_value=0),
        patch.object(deploy, "_run", side_effect=capture),
        patch.object(deploy, "detect_existing", return_value=None),
    ):
        deploy._fresh(bundle, assume_yes=True, passthrough=())

    assert len(calls) == 1
    child_env = calls[0][1]["env"]
    assert child_env[deploy.PREFLIGHT_STAMP_ENV] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()


def test_report_evaluation_requires_react_schema_evo_and_inference():
    payload = {
        "eva": {
            "service": {"ok": True},
            "health": {"ok": True},
            "ready": {"ok": True},
            "ui": {"ok": True},
        },
        "database": {"ok": True, "current_revision": report.EXPECTED_SCHEMA},
        "luxriot": {"configured": True, "ready_check": {"ok": True}},
        "inference": {"profiles": [{"id": "agent", "ok": True}, {"id": "vlm", "ok": True}]},
        "streams": {
            "ok": True,
            "channels": 4,
            "records": 12,
            "latest_event_timestamp_ms": 200,
        },
    }
    baseline = {"streams": {"channels": 4, "latest_event_timestamp_ms": 100}}

    assessment = report.evaluate(payload, baseline)

    assert assessment == {"status": "PASS", "failures": [], "warnings": []}


def test_report_maps_database_uri_to_libpq_environment_without_argv_secret():
    dsn = "postgresql://reporter:secret@db.internal:5433/eva?sslmode=require"
    with patch.object(
        report,
        "_command",
        return_value={"ok": True, "stdout": report.EXPECTED_SCHEMA, "stderr": ""},
    ) as command:
        result = report._psql(dsn, "SELECT version_num FROM alembic_version")

    assert result["ok"] is True
    argv = command.call_args.args[0]
    environment = command.call_args.kwargs["env"]
    assert dsn not in argv
    assert dsn not in environment.values()
    assert environment["PGHOST"] == "db.internal"
    assert environment["PGPORT"] == "5433"
    assert environment["PGDATABASE"] == "eva"
    assert environment["PGUSER"] == "reporter"
    assert environment["PGPASSWORD"] == "secret"
    assert environment["PGSSLMODE"] == "require"


def test_report_fails_when_previously_active_streams_do_not_resume():
    payload = {
        "eva": {
            "service": {"ok": True},
            "health": {"ok": True},
            "ready": {"ok": True},
            "ui": {"ok": True},
        },
        "database": {"ok": True, "current_revision": report.EXPECTED_SCHEMA},
        "luxriot": {"configured": True, "ready_check": {"ok": True}},
        "inference": {"profiles": [{"id": "agent", "ok": True}]},
        "streams": {"ok": True, "channels": 4, "latest_event_timestamp_ms": 100},
    }
    baseline = {"streams": {"channels": 4, "latest_event_timestamp_ms": 100}}

    assessment = report.evaluate(payload, baseline)

    assert assessment["status"] == "FAIL"
    assert any("did not produce" in item for item in assessment["failures"])


def test_text_report_contains_field_handoff_facts():
    payload = {
        "generated_at": "2026-08-04T00:00:00+00:00",
        "host": {"hostname": "eva-georgia", "kernel": "6.8.0", "gpu": {"devices": ["RTX 5070 Ti"]}},
        "eva": {
            "version": "β 0.8.5",
            "service": {"ok": True},
            "ready": {"ok": True},
            "ui": {"ok": True},
        },
        "database": {"ok": True, "current_revision": report.EXPECTED_SCHEMA},
        "luxriot": {"configured": True, "ready_check": {"ok": True}},
        "inference": {"profiles": [{"id": "vlm-1", "ok": True}]},
        "streams": {"channels": 50, "records": 900, "window_minutes": 15},
    }

    text = report.render_text(payload, {"status": "PASS", "failures": [], "warnings": []})

    for expected in (
        "RESULT: PASS",
        "Kernel version: 6.8.0",
        "UI updated and running: YES (React)",
        f"Migrations successful: YES ({report.EXPECTED_SCHEMA})",
        "Luxriot Evo: REACHABLE",
        "Streams: WORKING",
    ):
        assert expected in text


def test_baseline_json_contract_is_secret_free(tmp_path):
    path = tmp_path / "baseline.json"
    payload = {"streams": {"channels": 2, "latest_event_timestamp_ms": 123}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert report._baseline_expected(loaded) == (True, 123)


def test_update_runs_reviewed_dry_run_then_apply_without_rewriting_site_profiles(tmp_path):
    bundle = tmp_path / "bundle"
    source = bundle / "repo"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    installer = scripts / "install_eva_083.py"
    installer.write_text("# installer\n", encoding="utf-8")
    (bundle / "SOURCE_REVISION.json").write_text(
        json.dumps({"commit": "a" * 40}), encoding="utf-8"
    )
    app = tmp_path / "app"
    app.mkdir()
    env_file = tmp_path / "eva-ai.env"
    env_file.write_text(
        "\n".join(
            (
                "EVA_MIGRATION_DATABASE_DSN=postgresql://migrator:secret@db/eva",
                "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL=http://external-vlm:8000/v1",
                "EVOSSEARCH_LM_PROFILE_VLM_MODEL=qwen-vlm",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    unit = tmp_path / "eva-ai.service"
    deployment = deploy.ExistingDeployment(
        service="eva-ai",
        app_dir=app,
        env_file=env_file,
        unit_file=unit,
        service_user="missing-test-user",
        service_group="missing-test-user",
        base_url="http://127.0.0.1:5000",
    )
    commands = []

    def capture_run(argv, **_kwargs):
        commands.append([str(item) for item in argv])
        return type("Completed", (), {"returncode": 0})()

    with (
        patch.object(deploy.os, "geteuid", return_value=0),
        patch.object(deploy, "DEFAULT_REPORT_ROOT", tmp_path / "reports"),
            patch.object(deploy, "DEFAULT_BACKUP_ROOT", tmp_path / "backups"),
            patch.object(deploy, "_run", side_effect=capture_run),
            patch.object(deploy, "_report"),
            patch.object(deploy, "_assert_update_compatibility"),
        ):
        deploy._update(bundle, deployment, assume_yes=True, wait_streams=0)

    installer_commands = [row for row in commands if str(installer) in row]
    assert len(installer_commands) == 2
    assert "--dry-run" in installer_commands[0]
    assert "--apply" in installer_commands[1]
    for row in installer_commands:
        assert row[row.index("--env-file") + 1] == str(env_file)
        assert "http://external-vlm:8000/v1" not in row
        assert "qwen-vlm" not in row
    assert (app / ".eva-bundle-commit").read_text(encoding="utf-8").strip() == "a" * 40


def test_update_forwards_explicit_live_evo_credential_verification(tmp_path):
    bundle = tmp_path / "bundle"
    source = bundle / "repo"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    installer = scripts / "install_eva_083.py"
    installer.write_text("# installer\n", encoding="utf-8")
    app = tmp_path / "app"
    app.mkdir()
    env_file = tmp_path / "eva-ai.env"
    env_file.write_text(
        "EVA_MIGRATION_DATABASE_DSN=postgresql://migrator:secret@db/eva\n",
        encoding="utf-8",
    )
    deployment = deploy.ExistingDeployment(
        service="eva-ai",
        app_dir=app,
        env_file=env_file,
        unit_file=tmp_path / "eva-ai.service",
        service_user="eva",
        service_group="eva",
        base_url="http://127.0.0.1:5000",
    )
    commands = []

    def capture_run(argv, **_kwargs):
        commands.append([str(item) for item in argv])
        return type("Completed", (), {"returncode": 0})()

    with (
        patch.object(deploy.os, "geteuid", return_value=0),
        patch.object(deploy, "DEFAULT_REPORT_ROOT", tmp_path / "reports"),
            patch.object(deploy, "DEFAULT_BACKUP_ROOT", tmp_path / "backups"),
            patch.object(deploy, "_run", side_effect=capture_run),
            patch.object(deploy, "_report"),
            patch.object(deploy, "_assert_update_compatibility"),
        ):
            deploy._update(
            bundle,
            deployment,
            assume_yes=True,
            wait_streams=0,
            verify_luxriot_credential=True,
            )

    installer_commands = [row for row in commands if str(installer) in row]
    assert len(installer_commands) == 2
    assert all("--verify-luxriot-credential" in row for row in installer_commands)
