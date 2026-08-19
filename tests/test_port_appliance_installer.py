from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "install_port_appliance.py"
SPEC = importlib.util.spec_from_file_location("install_port_appliance", MODULE_PATH)
assert SPEC and SPEC.loader
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)

FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "finalize_port_usb_bundle",
    ROOT / "scripts" / "finalize_port_usb_bundle.py",
)
assert FINALIZER_SPEC and FINALIZER_SPEC.loader
finalizer = importlib.util.module_from_spec(FINALIZER_SPEC)
sys.modules[FINALIZER_SPEC.name] = finalizer
FINALIZER_SPEC.loader.exec_module(finalizer)

DEPENDENCIES_SPEC = importlib.util.spec_from_file_location(
    "offline_bundle_dependencies",
    ROOT / "scripts" / "offline_bundle_dependencies.py",
)
assert DEPENDENCIES_SPEC and DEPENDENCIES_SPEC.loader
dependencies = importlib.util.module_from_spec(DEPENDENCIES_SPEC)
sys.modules[DEPENDENCIES_SPEC.name] = dependencies
DEPENDENCIES_SPEC.loader.exec_module(dependencies)


def test_fresh_entrypoint_reads_manifest_before_rendering_target(tmp_path, capsys):
    manifest = {
        "target": {
            "gpu": "RTX 5070 Ti",
            "os": "Ubuntu 24.04 LTS amd64",
        }
    }
    with (
        patch.object(installer, "read_manifest", return_value=manifest),
        patch.object(
            installer,
            "validate_target_host",
            side_effect=installer.InstallError("stop after entrypoint contract"),
        ),
    ):
        result = installer.main(("--bundle-root", str(tmp_path)))

    assert result == 1
    output = capsys.readouterr()
    assert "RTX 5070 Ti" in output.out
    assert "stop after entrypoint contract" in output.err


def test_fresh_host_validation_uses_declared_ubuntu_release(tmp_path):
    os_release = tmp_path / "os-release"
    os_release.write_text('ID=ubuntu\nVERSION_ID="26.04"\n', encoding="utf-8")
    manifest = {
        "offline_dependencies": {
            "target": {"architecture": "amd64", "os_release": "26.04"}
        }
    }
    with patch.object(installer.platform, "machine", return_value="x86_64"):
        installer.validate_target_host(manifest, os_release=os_release)

    os_release.write_text('ID=ubuntu\nVERSION_ID="24.04"\n', encoding="utf-8")
    with (
        patch.object(installer.platform, "machine", return_value="x86_64"),
        pytest.raises(installer.InstallError, match="requires Ubuntu Server 26.04"),
    ):
        installer.validate_target_host(manifest, os_release=os_release)


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
    assert 'SIGLIP2_CACHE_TARGET="${STAGING_ROOT}/models/huggingface"' in builder
    assert "xargs -0 -r sha256sum > SHA256SUMS" in builder
    assert "sha256sum -c SHA256SUMS" in builder
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

    universal_builder = (ROOT / "scripts" / "build_universal_usb_bundle.sh").read_text(
        encoding="utf-8"
    )
    assert "EVA_UNIVERSAL_UPDATE_SEED" in universal_builder
    assert "stale updates exist in staging" in universal_builder


def test_port_profile_shares_bounded_gpu_with_siglip2():
    assert installer.PORT_ENV["CUDA_VISIBLE_DEVICES"] == "0"
    assert installer.PORT_ENV["EVOSSEARCH_EMBEDDER"] == "clip"
    assert installer.PORT_ENV["EVOSSEARCH_CLIP_MODEL"] == "google/siglip2-base-patch16-224"
    assert installer.PORT_ENV["EVOSSEARCH_CLIP_DEVICE"] == "cuda"
    assert installer.PORT_ENV["EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS"] == "1000"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_SUMMARY_MAX_BATCH_FRAMES"] == "16"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_ATTENTION_MAX_VLM_FRAMES"] == "8"
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
    assert '"ventspils-maritime-client", "universal-offline"' in finalizer
    assert "Refusing to finalize an uncommitted port client bundle" in finalizer
    assert '"START_EVA_AI.sh"' in finalizer
    assert '"manifest.txt"' in finalizer
    assert '"format": 2' in finalizer
    assert '"installation_modes": ["fresh", "update", "report"]' in finalizer
    assert '"offline-dependencies.json"' in finalizer
    assert '"update_packages": update_packages' in finalizer
    assert "Update archive checksum mismatch" in finalizer
    assert '"models/huggingface/SHA256SUMS"' in finalizer


def test_finalizer_verifies_packaged_siglip2_checksum_manifest(tmp_path):
    cache_root = tmp_path / "models" / "huggingface"
    payload = (
        cache_root
        / "models--google--siglip2-base-patch16-224"
        / "blobs"
        / "model"
    )
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"siglip2-test-weights")
    relative = payload.relative_to(cache_root)
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    checksum = cache_root / "SHA256SUMS"
    checksum.write_text(f"{expected}  {relative}\n", encoding="utf-8")

    assert finalizer.verify_siglip2_checksum_manifest(tmp_path) == 1

    payload.write_bytes(b"corrupted")
    finalizer.digest.cache_clear()
    with pytest.raises(SystemExit, match="checksum mismatch"):
        finalizer.verify_siglip2_checksum_manifest(tmp_path)


def test_finalizer_rejects_missing_siglip2_checksum_manifest(tmp_path):
    with pytest.raises(SystemExit, match="checksum manifest is missing"):
        finalizer.verify_siglip2_checksum_manifest(tmp_path)


def test_finalizer_binds_standalone_update_pack_into_release_manifest(tmp_path):
    package = tmp_path / "updates" / "georgia-d725c87"
    bundle_name = "eva-ai-georgia-upgrade-d725c87"
    expanded = package / bundle_name
    archive = package / f"{bundle_name}.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"immutable update archive")
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (package / f"{archive.name}.sha256").write_text(
        f"{archive_digest}  {archive.name}\n",
        encoding="utf-8",
    )
    (package / "START_GEORGIA_REHEARSAL.sh").write_text(
        "#!/usr/bin/env bash\n",
        encoding="utf-8",
    )
    (expanded / "repo" / "scripts").mkdir(parents=True)
    (expanded / "manifest.txt").write_text(
        "\n".join(
            (
                f"bundle_name={bundle_name}",
                "version=β 0.8.7",
                f"git_commit={'d' * 40}",
                "working_tree_status=clean",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for filename in ("database_preservation_guard.py", "pg_with_dsn.py"):
        (expanded / "repo" / "scripts" / filename).write_text(
            "# safety payload\n",
            encoding="utf-8",
        )

    packages, critical = finalizer.bundled_update_packages(tmp_path)

    assert packages == [
        {
            "name": "georgia-d725c87",
            "bundle_name": bundle_name,
            "version": "β 0.8.7",
            "git_commit": "d" * 40,
            "archive": f"updates/georgia-d725c87/{bundle_name}.tar.gz",
            "archive_sha256": archive_digest,
            "checksum": f"updates/georgia-d725c87/{bundle_name}.tar.gz.sha256",
            "expanded_manifest": f"updates/georgia-d725c87/{bundle_name}/manifest.txt",
            "launchers": ["updates/georgia-d725c87/START_GEORGIA_REHEARSAL.sh"],
        }
    ]
    assert f"updates/georgia-d725c87/{bundle_name}/repo/scripts/pg_with_dsn.py" in critical


def test_predeploy_gate_runs_react_tests_and_production_build():
    gate = (ROOT / "scripts" / "predeploy_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert '"${NPM}" --prefix "${REACT_ROOT}" test -- --run' in gate
    assert '"${NPM}" --prefix "${REACT_ROOT}" run build' in gate
    assert "React build did not produce dist/index.html" in gate


def test_port_profile_has_bounded_queue_and_context():
    assert installer.PORT_ENV["EVOSSEARCH_INFERENCE_QUEUE_ENABLED"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_INFERENCE_WORKER_COUNT"] == "3"
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
    assert "ExecStartPost={app_dir}/.venv/bin/python {app_dir}/scripts/wait_openai_endpoint.py --timeout 720" in source
    assert "TimeoutStartSec=780" in source
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
    assert values["EVOSSEARCH_LM_PROFILE_AGENT_MAX_INFLIGHT"] == "4"
    assert values["EVOSSEARCH_LM_PROFILE_VLM_MAX_INFLIGHT"] == "4"
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


def test_offline_bundle_includes_python_headers_for_native_wheels():
    package_input = (
        ROOT / "deployment" / "port_4070s" / "apt-packages-ubuntu-24.04.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert "python3-dev" in installer.APT_PACKAGES
    assert "python3-dev" in package_input


def test_nginx_configuration_restarts_an_existing_process(tmp_path, capsys):
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
    )

    installer.configure_nginx(answers, installer.Runner(dry_run=True))

    commands = capsys.readouterr().out
    assert "+ nginx -t" in commands
    assert "+ systemctl enable nginx" in commands
    assert "+ systemctl restart nginx" in commands
    assert "systemctl enable --now nginx" not in commands


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
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip().removeprefix("β ")
    package = tmp_path / f"eva-ai-appliance-installer_{version}_amd64.deb"
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
    assert "usr/sbin/eva-ai-deploy" in contents
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


def test_bundle_local_vlm_capability_comes_from_dependency_contract(tmp_path):
    manifest = tmp_path / "offline-dependencies.json"
    manifest.write_text(
        json.dumps({"pip_resolution": {"vllm": "external"}}),
        encoding="utf-8",
    )
    assert installer.bundle_supports_local_vlm(tmp_path) is False

    manifest.write_text(
        json.dumps({"pip_resolution": {"vllm": "0.25.0"}}),
        encoding="utf-8",
    )
    assert installer.bundle_supports_local_vlm(tmp_path) is True


def test_noninteractive_external_only_bundle_rejects_implicit_local_vlm(tmp_path):
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
        )
    )

    with pytest.raises(installer.InstallError, match="requires --external-vlm-url"):
        installer.gather_answers(True, args, local_vlm_available=False)


def _spark_manifest(archive: str = ""):
    runtime = {
        "engine": "docker",
        "base_image": installer.SPARK_RUNTIME_BASE_IMAGE,
        "base_manifest_digest": installer.SPARK_RUNTIME_BASE_MANIFEST_DIGEST,
        "base_image_id": installer.SPARK_RUNTIME_BASE_IMAGE_ID,
        "image": installer.SPARK_RUNTIME_IMAGE,
        "image_id": installer.SPARK_RUNTIME_IMAGE_ID,
        "model": installer.SPARK_VLM_REPO,
        "model_revision": installer.SPARK_VLM_REVISION,
        "numpy": installer.SPARK_NUMPY_VERSION,
        "pip_constraint": installer.SPARK_PIP_CONSTRAINT,
        "weight_quantization": "online-fp8-w8a8",
        "kv_cache_dtype": "bfloat16",
        "vision_attention_dtype": "bfloat16",
    }
    if archive:
        runtime["archive"] = archive
    return {"container_runtime": runtime}


def test_spark_runtime_contract_is_immutable_and_path_safe():
    contract = installer.spark_runtime_contract(
        _spark_manifest("container/eva-spark-runtime-0.8.7-arm64.tar.zst")
    )
    assert contract["image_id"] == installer.SPARK_RUNTIME_IMAGE_ID

    changed = _spark_manifest()
    changed["container_runtime"]["image"] = "nvcr.io/nvidia/vllm:latest"
    with pytest.raises(installer.InstallError, match="does not match"):
        installer.spark_runtime_contract(changed)

    with pytest.raises(installer.InstallError, match="unsafe"):
        installer.spark_runtime_contract(_spark_manifest("../runtime.tar"))


def test_spark_vendor_numpy_contract_is_consistent_across_release_inputs():
    runtime = json.loads(
        (
            ROOT / "deployment" / "spark_gb10" / "runtime-container.json"
        ).read_text(encoding="utf-8")
    )
    constraints = (
        ROOT / "deployment" / "spark_gb10" / "constraints-spark-gb10.txt"
    ).read_text(encoding="utf-8")

    assert runtime["numpy"] == installer.SPARK_NUMPY_VERSION
    assert runtime["numpy"] == finalizer.SPARK_NUMPY_VERSION
    assert runtime["numpy"] == dependencies.SPARK_VENDOR_RUNTIME_PACKAGES["numpy"]
    assert runtime["pip_constraint"] == installer.SPARK_PIP_CONSTRAINT
    assert runtime["pip_constraint"] == finalizer.SPARK_PIP_CONSTRAINT
    assert f"numpy=={runtime['numpy']}" in constraints.splitlines()


def test_spark_runtime_probe_binds_vendor_numpy_and_pip_constraint():
    payload = {
        "numpy": installer.SPARK_NUMPY_VERSION,
        "pip_constraint": installer.SPARK_PIP_CONSTRAINT,
        "torch": "2.13.0a0+nv26.07",
        "torchvision": "0.28.0a0+nv26.07",
        "vllm": "0.24.0.dev",
        "ffmpeg": "/usr/bin/ffmpeg",
        "ffmpeg_returncode": 0,
        "cuda_available": True,
        "cuda": "13.3",
        "device": "NVIDIA GB10",
    }
    responses = [
        CompletedProcess([], 0, "", ""),
        CompletedProcess(
            [],
            0,
            f"arm64|{installer.SPARK_RUNTIME_IMAGE_ID}\n",
            "",
        ),
        CompletedProcess([], 0, json.dumps(payload) + "\n", ""),
    ]
    with (
        patch.object(installer.shutil, "which", return_value="/usr/bin/docker"),
        patch.object(installer, "_spark_image_present", return_value=True),
        patch.object(installer.subprocess, "run", side_effect=responses),
    ):
        result = installer.validate_spark_container_runtime(_spark_manifest())

    assert result is not None
    assert result["numpy"] == "2.1.0"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("numpy", "2.3.5", "incompatible NumPy"),
        ("pip_constraint", "", "unexpected pip constraint"),
    ),
)
def test_spark_runtime_probe_rejects_vendor_dependency_drift(field, value, message):
    payload = {
        "numpy": installer.SPARK_NUMPY_VERSION,
        "pip_constraint": installer.SPARK_PIP_CONSTRAINT,
        "torch": "2.13.0a0+nv26.07",
        "torchvision": "0.28.0a0+nv26.07",
        "vllm": "0.24.0.dev",
        "ffmpeg": "/usr/bin/ffmpeg",
        "ffmpeg_returncode": 0,
        "cuda_available": True,
        "cuda": "13.3",
        "device": "NVIDIA GB10",
    }
    payload[field] = value
    responses = [
        CompletedProcess([], 0, "", ""),
        CompletedProcess(
            [],
            0,
            f"arm64|{installer.SPARK_RUNTIME_IMAGE_ID}\n",
            "",
        ),
        CompletedProcess([], 0, json.dumps(payload) + "\n", ""),
    ]
    with (
        patch.object(installer.shutil, "which", return_value="/usr/bin/docker"),
        patch.object(installer, "_spark_image_present", return_value=True),
        patch.object(installer.subprocess, "run", side_effect=responses),
        pytest.raises(installer.InstallError, match=message),
    ):
        installer.validate_spark_container_runtime(_spark_manifest())


def test_site_timezone_requires_iana_name():
    assert installer._timezone_name("Europe/Riga") == "Europe/Riga"
    with pytest.raises(installer.InstallError, match="IANA name"):
        installer._timezone_name("Riga")


def test_spark_python_environment_is_built_inside_pinned_container(tmp_path, capsys):
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
        local_vlm=False,
        local_deep=False,
    )
    installer.install_python_envs(
        tmp_path / "bundle",
        answers,
        installer.Runner(dry_run=True),
        architecture="arm64",
    )
    output = capsys.readouterr().out
    assert "docker run --rm --network host --ipc host" in output
    assert f"--entrypoint python3 {installer.SPARK_RUNTIME_IMAGE_ID}" in output
    assert "venv --system-site-packages" in output
    assert "/eva-bundle/wheelhouse" in output
    assert "/eva-bundle/constraints-spark-gb10.txt" in output
    assert "+ python3 -m venv" not in output


def test_factory_spark_install_loads_bundled_runtime_before_canary(tmp_path):
    archive_relative = "container/eva-spark-runtime-0.8.7-arm64.tar.zst"
    archive = tmp_path / archive_relative
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"verified offline OCI image")
    runner = installer.Runner(dry_run=False)

    with (
        patch.object(installer, "_spark_image_present", side_effect=[False, True]),
        patch.object(installer, "validate_spark_container_runtime") as canary,
        patch.object(runner, "run") as run,
    ):
        installer.ensure_spark_runtime(
            tmp_path,
            _spark_manifest(archive_relative),
            runner,
        )

    run.assert_called_once_with(("docker", "load", "--input", archive))
    canary.assert_called_once()


def test_spark_database_migration_names_postgres_role_inside_container(tmp_path):
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
        local_vlm=True,
        local_deep=False,
    )
    calls = []

    class RecordingRunner:
        dry_run = False

        def run(self, command, **kwargs):
            calls.append((tuple(str(item) for item in command), kwargs))
            return CompletedProcess(command, 0, "", "")

    existing = {
        "EVA_MIGRATOR_PASSWORD": "migrator",
        "EVA_API_PASSWORD": "api",
        "EVA_AUDIT_PASSWORD": "audit",
        "EVA_WORKER_PASSWORD": "worker",
        "EVA_BACKUP_PASSWORD": "backup",
    }
    account = SimpleNamespace(pw_uid=128, pw_gid=127)
    with patch.object(installer.pwd, "getpwnam", return_value=account):
        installer.prepare_database(
            answers,
            RecordingRunner(),
            db_was_present=True,
            existing_env=existing,
            architecture="arm64",
        )

    container_calls = [
        (command, kwargs)
        for command, kwargs in calls
        if command and command[0] == "docker"
    ]
    assert len(container_calls) == 2
    for command, kwargs in container_calls:
        assert "--user" in command
        assert command[command.index("--user") + 1] == "128:127"
        assert kwargs["env"]["EVA_DATABASE_DSN"] == (
            "postgresql://postgres@/eva?host=/var/run/postgresql"
        )


def test_spark_systemd_uses_separate_pinned_gpu_container(tmp_path):
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
        local_vlm=False,
        local_deep=False,
    )
    written = {}

    class RecordingRunner:
        dry_run = False

        def run(self, command, **_kwargs):
            return CompletedProcess(command, 0, "", "")

    def record(path, content, mode):
        written[str(path)] = (content, mode)

    account = SimpleNamespace(pw_uid=991, pw_gid=991)
    with patch.object(installer.pwd, "getpwnam", return_value=account), patch.object(
        installer, "_atomic_write", side_effect=record
    ):
        installer.install_systemd_units(
            answers,
            RecordingRunner(),
            architecture="arm64",
        )

    unit, mode = written["/etc/systemd/system/eva-ai.service"]
    assert mode == 0o644
    assert "Requires=postgresql.service docker.service" in unit
    assert f"--name {installer.SPARK_RUNTIME_CONTAINER_NAME}" in unit
    assert f"--user 991:991" in unit
    assert "--gpus all" in unit
    assert installer.SPARK_RUNTIME_IMAGE_ID in unit
    assert "eva-vllm.service" not in unit
    assert f"docker stop -t 110 {installer.SPARK_RUNTIME_CONTAINER_NAME}" in unit


def test_spark_systemd_installs_its_own_local_vlm(tmp_path):
    answers = installer.Answers(
        install_root=tmp_path / "opt",
        data_root=tmp_path / "data",
        config_root=tmp_path / "etc",
        evo_url="http://evo.local",
        evo_username="operator",
        evo_password="secret",
        local_vlm=True,
        local_deep=False,
    )
    written = {}

    class RecordingRunner:
        dry_run = False

        def run(self, command, **_kwargs):
            return CompletedProcess(command, 0, "", "")

    account = SimpleNamespace(pw_uid=991, pw_gid=991)
    with patch.object(installer.pwd, "getpwnam", return_value=account), patch.object(
        installer,
        "_atomic_write",
        side_effect=lambda path, content, mode: written.setdefault(
            str(path), (content, mode)
        ),
    ):
        installer.install_systemd_units(
            answers,
            RecordingRunner(),
            architecture="arm64",
        )

    app_unit = written["/etc/systemd/system/eva-ai.service"][0]
    vlm_unit = written["/etc/systemd/system/eva-vllm.service"][0]
    assert "Wants=eva-vllm.service" in app_unit
    assert "--name eva-vllm" in vlm_unit
    assert installer.SPARK_RUNTIME_IMAGE_ID in vlm_unit
    assert "/models/qwen3-vl-4b" in vlm_unit
    assert "--quantization fp8" in vlm_unit
    assert "--max-num-seqs 4" in vlm_unit
    assert "--limit-mm-per-prompt.image 8" in vlm_unit
    assert "VLLM_USE_DEEP_GEMM=0" in vlm_unit


def test_spark_noninteractive_defaults_to_bundled_vlm_and_no_deep(tmp_path):
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
        )
    )

    answers = installer.gather_answers(True, args, architecture="arm64")

    assert answers.local_vlm is True
    assert answers.vlm_url == installer.DEFAULT_VLM_URL
    assert answers.local_deep is False
    assert answers.deep_url == ""


def test_container_env_is_literal_not_shell_quoted():
    rendered = installer.render_container_env(
        {"A_URL": "http://127.0.0.1:8080/v1", "B_SECRET": "abc#123"}
    )
    assert "A_URL=http://127.0.0.1:8080/v1" in rendered
    assert "B_SECRET=abc#123" in rendered
    assert "'" not in rendered


def test_finalizer_binds_required_spark_image_archive(tmp_path):
    shutil.copy2(
        ROOT / "deployment" / "spark_gb10" / "runtime-container.json",
        tmp_path / "runtime-container.json",
    )
    archive = tmp_path / finalizer.SPARK_RUNTIME_ARCHIVE
    archive.parent.mkdir()
    archive.write_bytes(b"offline OCI archive")

    archive_manifest = json.dumps(
        [{"Config": installer.SPARK_RUNTIME_IMAGE_ID.removeprefix("sha256:")}]
    )
    with patch.object(
        finalizer.subprocess,
        "run",
        return_value=CompletedProcess([], 0, archive_manifest, ""),
    ):
        runtime, critical = finalizer.spark_runtime_payload(tmp_path, "arm64")

    assert runtime is not None
    assert runtime["image_id"] == installer.SPARK_RUNTIME_IMAGE_ID
    assert runtime["archive"] == finalizer.SPARK_RUNTIME_ARCHIVE
    assert runtime["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert finalizer.SPARK_RUNTIME_ARCHIVE in critical


def test_finalizer_rejects_spark_bundle_without_runtime_archive(tmp_path):
    shutil.copy2(
        ROOT / "deployment" / "spark_gb10" / "runtime-container.json",
        tmp_path / "runtime-container.json",
    )

    with pytest.raises(SystemExit, match="runtime archive is missing"):
        finalizer.spark_runtime_payload(tmp_path, "arm64")
