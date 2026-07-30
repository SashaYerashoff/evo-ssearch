from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_appliance_config.py"
SPEC = importlib.util.spec_from_file_location("validate_appliance_config", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def complete_values(tmp_path: Path) -> dict[str, str]:
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "ViT-B-32.pt").write_bytes(b"weights")
    tenant = "903b65df-8fc7-44a6-9dff-230857df81b8"
    return {
        "EVA_DATABASE_DSN": "postgresql://api:secret@127.0.0.1/eva",
        "EVA_AUDIT_DATABASE_DSN": "postgresql://audit:secret@127.0.0.1/eva",
        "EVA_WORKER_DATABASE_DSN": "postgresql://worker:secret@127.0.0.1/eva",
        "EVA_MIGRATION_DATABASE_DSN": "postgresql://migrator:secret@127.0.0.1/eva",
        "EVOSSEARCH_AUTH_ENABLED": "true",
        "EVOSSEARCH_AUTH_COOKIE_SECURE": "true",
        "EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED": "true",
        "EVOSSEARCH_AUTH_TENANT_ID": tenant,
        "EVOSSEARCH_ARCHIVE_TENANT_ID": tenant,
        "EVOSSEARCH_INFERENCE_QUEUE_TENANT_ID": tenant,
        "EVOSSEARCH_LUXRIOT_BASE_URL": "http://evo.local",
        "EVOSSEARCH_LUXRIOT_USERNAME": "operator",
        "EVOSSEARCH_LUXRIOT_PASSWORD": "secret",
        "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL": "http://127.0.0.1:1234/v1",
        "EVOSSEARCH_LM_PROFILE_AGENT_MODEL": "qwen/qwen3-vl-4b",
        "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL": "http://127.0.0.1:1234/v1",
        "EVOSSEARCH_LM_PROFILE_VLM_MODEL": "qwen/qwen3-vl-4b",
        "EVOSSEARCH_OPENAI_CLIP_CACHE_DIR": str(clip),
    }


def test_complete_appliance_configuration_passes(tmp_path):
    assert validator.validate(complete_values(tmp_path)) == []


def test_partial_field_recovery_configuration_fails_closed(tmp_path):
    values = complete_values(tmp_path)
    values.pop("EVA_DATABASE_DSN")
    values["EVOSSEARCH_AUTH_ENABLED"] = "false"
    values["EVOSSEARCH_AUTH_COOKIE_SECURE"] = "false"
    values["EVOSSEARCH_ADMIN_TOKEN"] = "legacy"

    errors = validator.validate(values)

    assert any("EVA_DATABASE_DSN" in error for error in errors)
    assert "named-user authentication must be enabled" in errors
    assert any("legacy EVOSSEARCH_ADMIN_TOKEN" in error for error in errors)
    assert any("secure authentication cookies" in error for error in errors)


def test_mismatched_tenant_ids_fail_closed(tmp_path):
    values = complete_values(tmp_path)
    values["EVOSSEARCH_ARCHIVE_TENANT_ID"] = (
        "680a0f47-702a-4e91-8497-364358493491"
    )
    assert any("tenant IDs must match" in error for error in validator.validate(values))
