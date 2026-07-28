from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "install_port_appliance.py"
SPEC = importlib.util.spec_from_file_location("install_port_appliance", MODULE_PATH)
assert SPEC and SPEC.loader
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)


def test_usb_builder_excludes_the_unused_react_workspace():
    builder = (ROOT / "scripts" / "build_port_usb_bundle.sh").read_text(
        encoding="utf-8"
    )

    assert "--exclude=react-ui/" in builder
    assert "--delete-excluded" in builder


def test_port_profile_keeps_gpu_for_vllm_and_clip_on_cpu():
    assert installer.PORT_ENV["CUDA_VISIBLE_DEVICES"] == "-1"
    assert installer.PORT_ENV["EVOSSEARCH_EMBEDDER"] == "clip"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS"] == "1000"
    assert installer.PORT_ENV["EVOSSEARCH_LUXRIOT_SUMMARY_MAX_BATCH_FRAMES"] == "16"


def test_port_profile_has_bounded_queue_and_context():
    assert installer.PORT_ENV["EVOSSEARCH_INFERENCE_QUEUE_ENABLED"] == "true"
    assert installer.PORT_ENV["EVOSSEARCH_INFERENCE_WORKER_COUNT"] == "1"
    assert installer.PORT_ENV["EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS"] == "32768"
    assert installer.PORT_ENV["EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS"] == "30000"


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


def test_external_inference_does_not_install_nvidia_driver(tmp_path, capsys):
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
    assert "qwen3-vl-4b-awq" not in commands
    assert "qwen3.5-9b-mtp" not in commands
    assert "llama.cpp" not in commands
