from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "install_port_appliance.py"
SPEC = importlib.util.spec_from_file_location("install_port_appliance", MODULE_PATH)
assert SPEC and SPEC.loader
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)


def test_usb_builder_builds_react_for_node_free_runtime():
    builder = (ROOT / "scripts" / "build_port_usb_bundle.sh").read_text(
        encoding="utf-8"
    )

    assert 'npm --prefix "${REACT_UI_ROOT}" run build' in builder
    assert 'dist/index.html' in builder
    assert not any(
        line.strip() == "--exclude=react-ui/ \\"
        for line in builder.splitlines()
    )
    assert "--exclude=react-ui/node_modules/" in builder
    assert "--delete-excluded" in builder
    assert 'EXPECTED_BRANCH="${EVA_PORT_EXPECTED_BRANCH:-feature/maritime-port-specs}"' in builder
    assert "port client bundle requires a clean committed working tree" in builder
    assert "SOURCE_REVISION.json" in builder
    for local_only_pattern in (
        "--exclude='.env*'",
        "--exclude='.venv*'",
        "--exclude=.eva-runtime",
        "--exclude=node_modules/",
        "--exclude=/dist/",
        "--exclude=probes_store.json",
        "--exclude='*.sqlite3'",
    ):
        assert local_only_pattern in builder


def test_port_profile_shares_bounded_gpu_with_siglip2():
    assert installer.PORT_ENV["CUDA_VISIBLE_DEVICES"] == "0"
    assert installer.PORT_ENV["EVOSSEARCH_EMBEDDER"] == "clip"
    assert installer.PORT_ENV["EVOSSEARCH_CLIP_MODEL"] == "google/siglip2-base-patch16-224"
    assert installer.PORT_ENV["EVOSSEARCH_CLIP_DEVICE"] == "cuda"
    assert installer.PORT_ENV["EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS"] == "1000"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_SUMMARY_MAX_BATCH_FRAMES"] == "16"
    assert installer.PORT_ENV["EVOSSEARCH_EMBEDDER_EAGER_LOAD"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_UI_MODE"] == "react"


def test_port_payload_requires_maritime_runtime_and_react_assets():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for relative in (
        "SOURCE_REVISION.json",
        "repo/camera_scene.py",
        "repo/maritime_profiles.py",
        "repo/docs/maritime_port_profile.md",
        "repo/react-ui/dist/index.html",
    ):
        assert f'"{relative}"' in source

    finalizer = (ROOT / "scripts" / "finalize_port_usb_bundle.py").read_text(
        encoding="utf-8"
    )
    assert '"release_flavor": "ventspils-maritime-client"' in finalizer
    assert "Refusing to finalize an uncommitted port client bundle" in finalizer


def test_predeploy_gate_runs_react_tests_and_production_build():
    gate = (ROOT / "scripts" / "predeploy_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert '"${NPM}" --prefix "${REACT_ROOT}" test -- --run' in gate
    assert '"${NPM}" --prefix "${REACT_ROOT}" run build' in gate
    assert "React build did not produce dist/index.html" in gate


def test_port_profile_has_bounded_queue_and_context():
    assert installer.PORT_ENV["EVOSSEARCH_INFERENCE_QUEUE_ENABLED"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_INFERENCE_WORKER_COUNT"] == "1"
    assert installer.PORT_ENV["EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS"] == "32768"
    assert installer.PORT_ENV["EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS"] == "30000"
    assert installer.PORT_ENV["EVOSSEARCH_LM_VIDEO_REPETITION_PENALTY"] == "1.08"


def test_port_vlm_uses_stable_vision_backend_and_content_watchdog(tmp_path):
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "--mm-encoder-attn-backend FLASH_ATTN" in source
    assert "--mm-processor-cache-gb 0" in source
    assert "--gpu-memory-utilization 0.72" in source
    assert "--max-num-seqs 4" in source
    assert "--enforce-eager" not in source
    assert "ExecStartPost={app_dir}/.venv/bin/python {app_dir}/scripts/wait_openai_endpoint.py --timeout 240" in source
    assert "eva-vlm-vision-watchdog.timer" in source
    assert "OnFailure=eva-vlm-vision-recover.service" in source

    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
    )
    passwords = {
        "EVA_MIGRATOR_PASSWORD": "a" * 64,
        "EVA_API_PASSWORD": "b" * 64,
        "EVA_AUDIT_PASSWORD": "c" * 64,
        "EVA_WORKER_PASSWORD": "d" * 64,
        "EVA_BACKUP_PASSWORD": "e" * 64,
    }
    values = installer.render_runtime_env(answers, {}, passwords)
    assert values["EVOSSEARCH_LM_PROFILE_AGENT_MAX_INFLIGHT"] == "8"
    assert values["EVOSSEARCH_LM_PROFILE_VLM_MAX_INFLIGHT"] == "8"
    assert values["EVOSSEARCH_LM_VISION_HEALTH_STATE_FILE"] == str(
        answers.data_root / "state" / "vlm-vision-health.json"
    )
    assert values["EVOSSEARCH_LM_VISION_HEALTH_MAX_AGE_SEC"] == "180"


def test_vision_smoke_png_and_response_contract():
    png = installer._vision_smoke_png()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")

    response = {
        "choices": [
            {
                "message": {
                    "content": "VISION_OK 7391 RED GREEN BLUE",
                }
            }
        ]
    }

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://vlm.local/v1/chat/completions"
        assert timeout == 12
        payload = json.loads(request.data)
        assert payload["model"] == "vlm-test"
        image_url = payload["messages"][0]["content"][0]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")
        return FakeResponse(json.dumps(response).encode())

    with patch.object(installer.urllib.request, "urlopen", fake_urlopen):
        installer._verify_vlm_vision(
            "http://vlm.local/v1",
            "vlm-test",
            timeout_sec=12,
        )


def test_vision_smoke_rejects_text_only_hallucination():
    response = {
        "choices": [
            {
                "message": {
                    "content": "VISION_OK 1234 RED BLUE",
                }
            }
        ]
    }

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    with patch.object(
        installer.urllib.request,
        "urlopen",
        return_value=FakeResponse(json.dumps(response).encode()),
    ):
        with pytest.raises(installer.InstallError, match="did not perceive"):
            installer._verify_vlm_vision("http://vlm.local/v1", "vlm-test")


def test_rendered_env_is_sorted_and_shell_quoted():
    rendered = installer.render_env({"Z_KEY": "last", "A_KEY": "first"})
    assert rendered.index("A_KEY='first'") < rendered.index("Z_KEY='last'")


def test_deep_review_is_fail_closed_without_endpoint(tmp_path):
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
        local_deep=False,
        deep_url="",
        deep_model="",
    )
    passwords = {
        "EVA_MIGRATOR_PASSWORD": "a" * 64,
        "EVA_API_PASSWORD": "b" * 64,
        "EVA_AUDIT_PASSWORD": "c" * 64,
        "EVA_WORKER_PASSWORD": "d" * 64,
        "EVA_BACKUP_PASSWORD": "e" * 64,
    }
    values = installer.render_runtime_env(answers, {}, passwords)
    assert values["EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_ENABLED"] == "false"
    assert values["EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_ENABLED"] == "false"
    assert values["EVOSSEARCH_LM_PROFILE_VLM_BASE_URL"] == installer.DEFAULT_VLM_URL
    assert "EVOSSEARCH_PROBE_STORE_FILE" not in values


def test_hardware_detection_falls_back_to_linux_pci_sysfs(tmp_path):
    pci_device = tmp_path / "0000:01:00.0"
    pci_device.mkdir()
    (pci_device / "vendor").write_text("0x10de\n", encoding="ascii")

    real_glob = Path.glob

    def fake_glob(path: Path, pattern: str):
        if str(path) == "/sys/bus/pci/devices" and pattern == "*/vendor":
            return iter((pci_device / "vendor",))
        return real_glob(path, pattern)

    with patch.object(installer.shutil, "which", return_value=None), patch.object(
        installer.Path, "glob", fake_glob
    ):
        hardware = installer.detect_hardware()

    assert hardware.nvidia_pci is True
    assert hardware.gpu_lines == []


def test_offline_apt_can_explicitly_filter_nvidia_driver(tmp_path, capsys):
    apt = tmp_path / "apt"
    apt.mkdir()
    (apt / "package-names.txt").write_text(
        "python3\nnvidia-driver-590-open\npostgresql\n",
        encoding="utf-8",
    )

    installer.install_offline_apt(
        tmp_path,
        installer.Runner(dry_run=True),
        include_nvidia=False,
    )

    commands = capsys.readouterr().out
    assert "python3" in commands
    assert "postgresql" in commands
    assert "nvidia-driver-590-open" not in commands


def test_external_vlm_still_requires_nvidia_for_local_siglip2_cuda(tmp_path):
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
        local_vlm=False,
        vlm_url="http://external-vlm.local/v1",
        vlm_model="external-vlm",
    )

    assert installer.local_siglip2_cuda_selected() is True
    assert installer.requires_local_nvidia(answers) is True

    values = installer.render_runtime_env(
        answers,
        {},
        {
            "EVA_MIGRATOR_PASSWORD": "a" * 64,
            "EVA_API_PASSWORD": "b" * 64,
            "EVA_AUDIT_PASSWORD": "c" * 64,
            "EVA_WORKER_PASSWORD": "d" * 64,
            "EVA_BACKUP_PASSWORD": "e" * 64,
        },
    )
    assert values["EVOSSEARCH_LM_PROFILE_VLM_BASE_URL"] == answers.vlm_url
    assert values["EVOSSEARCH_CLIP_DEVICE"] == "cuda"


def test_cpu_siglip_profile_does_not_require_nvidia_with_external_vlm():
    cpu_env = {
        **installer.PORT_ENV,
        "EVOSSEARCH_CLIP_DEVICE": "cpu",
    }
    assert installer.local_siglip2_cuda_selected(cpu_env) is False


def test_external_inference_skips_local_model_payloads(tmp_path, capsys):
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
        local_vlm=False,
        local_deep=False,
        vlm_url="http://vlm.local/v1",
        deep_url="",
        deep_model="",
    )

    installer.sync_payload(tmp_path / "bundle", answers, installer.Runner(dry_run=True))

    commands = capsys.readouterr().out
    assert "models/clip/ViT-B-32.pt" in commands
    assert "rsync -a --delete --delete-delay --exclude=.venv" in commands
    assert "qwen3-vl-4b-awq" not in commands
    assert "qwen3.5-9b-mtp" not in commands
    assert "llama.cpp" not in commands


def test_offline_apt_is_staged_locally_for_apt_sandbox(tmp_path):
    bundle = tmp_path / "usb"
    source = bundle / "apt"
    source.mkdir(parents=True)
    (source / "Packages.gz").write_bytes(b"package-index")
    (source / "package-names.txt").write_text("python3\n", encoding="utf-8")
    (source / "python3.deb").write_bytes(b"deb")
    apt_root = tmp_path / "apt-cache"
    (apt_root / "repos").mkdir(parents=True)

    class RecordingRunner:
        dry_run = False

        def __init__(self):
            self.commands = []

        def run(self, command, **kwargs):
            self.commands.append(tuple(str(item) for item in command))
            return CompletedProcess(command, 0, "", "")

    runner = RecordingRunner()
    installer.install_offline_apt(
        bundle,
        runner,
        include_nvidia=False,
        apt_root=apt_root,
    )

    source_line = (apt_root / "eva-ai-offline.list").read_text(encoding="utf-8")
    assert str(bundle) not in source_line
    assert str(apt_root / "repos") in source_line
    assert any(
        command[:8] == (
            "install",
            "-d",
            "-o",
            "_apt",
            "-g",
            "root",
            "-m",
            "0700",
        )
        for command in runner.commands
    )


def test_runtime_env_drops_legacy_token_and_repairs_partial_tenant_ids(tmp_path):
    tenant_id = "903b65df-8fc7-44a6-9dff-230857df81b8"
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
    )
    passwords = {
        "EVA_MIGRATOR_PASSWORD": "a" * 64,
        "EVA_API_PASSWORD": "b" * 64,
        "EVA_AUDIT_PASSWORD": "c" * 64,
        "EVA_WORKER_PASSWORD": "d" * 64,
        "EVA_BACKUP_PASSWORD": "e" * 64,
    }
    values = installer.render_runtime_env(
        answers,
        {
            "EVOSSEARCH_ARCHIVE_TENANT_ID": tenant_id,
            "EVOSSEARCH_ADMIN_TOKEN": "legacy",
        },
        passwords,
    )
    assert "EVOSSEARCH_ADMIN_TOKEN" not in values
    assert {values[key] for key in installer.TENANT_ID_KEYS} == {tenant_id}


def test_runtime_env_refuses_conflicting_tenant_ids():
    with pytest.raises(installer.InstallError, match="tenant IDs disagree"):
        installer.resolve_tenant_id(
            {
                "EVOSSEARCH_AUTH_TENANT_ID": "903b65df-8fc7-44a6-9dff-230857df81b8",
                "EVOSSEARCH_ARCHIVE_TENANT_ID": "680a0f47-702a-4e91-8497-364358493491",
            }
        )


def test_installer_bootstraps_admin_before_starting_runtime():
    source = MODULE_PATH.read_text(encoding="utf-8")
    apply_body = source.split("def apply_install(", 1)[1].split(
        "def build_parser(", 1
    )[0]
    assert apply_body.index('"administrator"') < apply_body.index(
        '"services_and_readiness"'
    )
    assert "/ready" in source


def test_bootstrap_deb_has_noninteractive_packaging_boundary(tmp_path):
    if not shutil.which("dpkg-deb"):
        pytest.skip("dpkg-deb is unavailable")
    subprocess.run(
        (
            sys.executable,
            ROOT / "scripts" / "build_appliance_installer_deb.py",
            "--output-dir",
            tmp_path,
        ),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    package = tmp_path / "eva-ai-appliance-installer_0.8.5_amd64.deb"
    assert package.is_file()
    control = subprocess.run(
        ("dpkg-deb", "--field", package),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    assert "Package: eva-ai-appliance-installer" in control
    assert "PostgreSQL" not in control
    contents = subprocess.run(
        ("dpkg-deb", "--contents", package),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    assert "usr/sbin/eva-ai-install" in contents
    assert "usr/sbin/eva-ai-doctor" in contents
    assert "models/" not in contents


def test_install_journal_is_secret_free_and_records_failure(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"version":"test"}', encoding="utf-8")
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="evo-password",
        admin_password="admin-password",
    )
    journal = installer.InstallJournal(
        path=tmp_path / "state.json",
        secrets_to_redact=(answers.evo_password, answers.admin_password),
    )
    journal.begin(bundle, answers)
    journal.mark(
        "offline_apt",
        "failed",
        f"permission denied for {answers.evo_password}",
    )
    raw = journal.path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert "evo-password" not in raw
    assert "admin-password" not in raw
    assert payload["status"] == "failed"
    assert payload["failed_phase"] == "offline_apt"


def test_noninteractive_install_has_no_hidden_password_prompt(tmp_path):
    evo_secret = tmp_path / "evo.secret"
    admin_secret = tmp_path / "admin.secret"
    evo_secret.write_text("evo-password\n", encoding="utf-8")
    admin_secret.write_text("long-admin-password\n", encoding="utf-8")
    args = installer.build_parser().parse_args(
        (
            "--non-interactive",
            "--evo-url",
            "evo.local",
            "--evo-username",
            "operator",
            "--evo-password-file",
            str(evo_secret),
            "--admin-password-file",
            str(admin_secret),
            "--admin-username",
            "admins",
            "--quiet-window-start",
            "01:30",
            "--quiet-window-end",
            "04:30",
        )
    )

    answers = installer.gather_answers(True, args)

    assert answers.evo_password == "evo-password"
    assert answers.admin_password == "long-admin-password"
    assert answers.admin_username == "admins"
    assert answers.quiet_enabled is True
    assert answers.quiet_start == "01:30"
    assert answers.quiet_end == "04:30"
