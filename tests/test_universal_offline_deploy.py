from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch


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


def test_common_bundle_verification_catches_corruption_before_either_path(tmp_path):
    bundle = tmp_path / "bundle"
    critical_file = bundle / "repo" / "VERSION"
    required_files = (
        bundle / "SOURCE_REVISION.json",
        bundle / "START_EVA_AI.sh",
        bundle / "eva_offline_deploy.py",
        bundle / "install_port_appliance.py",
        bundle / "migration-plans" / "0006-to-0011.sql",
        bundle / "apt" / "Packages.gz",
        bundle / "repo" / "react-ui" / "dist" / "index.html",
        bundle / "repo" / "migrations" / "versions" / "20260801_0011_incidents.py",
    )
    for path in (*required_files, critical_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload\n", encoding="utf-8")
    (bundle / "wheelhouse").mkdir()
    digest = hashlib.sha256(critical_file.read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "release_flavor": deploy.EXPECTED_FLAVOR,
                "schema_head": deploy.EXPECTED_SCHEMA,
                "critical_sha256": {"repo/VERSION": digest},
            }
        ),
        encoding="utf-8",
    )

    deploy._verify_bundle(bundle)
    critical_file.write_text("corrupted\n", encoding="utf-8")

    try:
        deploy._verify_bundle(bundle)
    except deploy.DeployError as exc:
        assert "Checksum mismatch" in str(exc)
    else:
        raise AssertionError("corrupted critical file was accepted")


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
