from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eva_appliance_doctor.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("eva_appliance_doctor", SCRIPT)
assert SPEC and SPEC.loader
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


def test_vllm_tool_calling_contract_accepts_port_unit() -> None:
    result = doctor._vllm_tool_calling_contract(
        "ExecStart=/opt/eva/vllm serve /models/qwen "
        "--enable-auto-tool-choice --tool-call-parser hermes\n"
    )

    assert result == {
        "ok": True,
        "unit_exec_start_present": True,
        "auto_tool_choice": True,
        "tool_call_parser": "hermes",
    }


def test_vllm_tool_calling_contract_rejects_non_agentic_unit() -> None:
    result = doctor._vllm_tool_calling_contract(
        "ExecStart=/opt/eva/vllm serve /models/qwen --max-model-len 32768\n"
    )

    assert result["ok"] is False
    assert result["unit_exec_start_present"] is True
    assert result["auto_tool_choice"] is False
    assert result["tool_call_parser"] is None


def test_vllm_tool_calling_contract_requires_parser_value() -> None:
    result = doctor._vllm_tool_calling_contract(
        "ExecStart=/opt/eva/vllm serve /models/qwen "
        "--enable-auto-tool-choice --tool-call-parser\n"
    )

    assert result["ok"] is False
    assert result["tool_call_parser"] is None


def test_external_vlm_profile_does_not_require_a_local_vllm_unit() -> None:
    result = doctor._vllm_runtime_contract(
        {"EVOSSEARCH_LM_PROFILE_VLM_BASE_URL": "http://vlm.internal:8000/v1"}
    )

    assert result == {
        "ok": True,
        "status": "external_profile",
        "local_unit_required": False,
    }
