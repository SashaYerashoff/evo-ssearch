import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import oldapp
from oldapp import (
    ENV_SECRET_REDACTION,
    _redact_env_map,
    _restore_redacted_env_secrets,
    app,
    config,
)


class SecuritySmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self._orig_auth_enabled = config.AUTH_ENABLED
        self._orig_admin_token = config.ADMIN_TOKEN
        self._orig_settings_local_only = config.SETTINGS_LOCAL_ONLY
        self._orig_secure_deployment_required = getattr(
            config,
            "SECURE_DEPLOYMENT_REQUIRED",
            False,
        )
        self._orig_db_strict_runtime_roles = getattr(
            config,
            "DB_STRICT_RUNTIME_ROLES",
            False,
        )
        config.AUTH_ENABLED = False

    def tearDown(self) -> None:
        config.AUTH_ENABLED = self._orig_auth_enabled
        config.ADMIN_TOKEN = self._orig_admin_token
        config.SETTINGS_LOCAL_ONLY = self._orig_settings_local_only
        config.SECURE_DEPLOYMENT_REQUIRED = self._orig_secure_deployment_required
        config.DB_STRICT_RUNTIME_ROLES = self._orig_db_strict_runtime_roles

    def test_search_missing_json_returns_api_error(self) -> None:
        resp = self.client.post("/search")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("error", payload)

    def test_settings_masks_luxriot_password(self) -> None:
        headers = {}
        if config.ADMIN_TOKEN:
            headers["X-Admin-Token"] = config.ADMIN_TOKEN
        resp = self.client.get("/settings", headers=headers)
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload.get("success"))
        settings = payload.get("settings", {})
        self.assertEqual(settings.get("luxriotPassword"), "")
        self.assertIn("luxriotPasswordSet", settings)

    def test_env_editor_redacts_and_preserves_secrets(self) -> None:
        current = {
            "EVOSSEARCH_LM_API_KEY": "lm-secret",
            "EVOSSEARCH_LUXRIOT_PASSWORD": "camera-secret",
            "EVOSSEARCH_PORT": "5000",
        }
        redacted = _redact_env_map(current)

        self.assertEqual(redacted["EVOSSEARCH_LM_API_KEY"], ENV_SECRET_REDACTION)
        self.assertEqual(redacted["EVOSSEARCH_LUXRIOT_PASSWORD"], ENV_SECRET_REDACTION)
        self.assertEqual(redacted["EVOSSEARCH_PORT"], "5000")
        self.assertNotIn("lm-secret", str(redacted))
        self.assertNotIn("camera-secret", str(redacted))

        submitted = dict(redacted)
        submitted["EVOSSEARCH_PORT"] = "5001"
        restored = _restore_redacted_env_secrets(submitted, current)
        self.assertEqual(restored["EVOSSEARCH_LM_API_KEY"], "lm-secret")
        self.assertEqual(restored["EVOSSEARCH_LUXRIOT_PASSWORD"], "camera-secret")
        self.assertEqual(restored["EVOSSEARCH_PORT"], "5001")

    def test_env_endpoint_never_returns_secret_values(self) -> None:
        with (
            patch.object(config, "ADMIN_TOKEN", "admin-secret"),
            patch.object(config, "LM_API_KEY", "lm-secret"),
            patch.object(config, "LUXRIOT_PASSWORD", "camera-secret"),
        ):
            resp = self.client.get(
                "/settings/env",
                headers={"X-Admin-Token": "admin-secret"},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        serialized = str(payload)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("lm-secret", serialized)
        self.assertNotIn("camera-secret", serialized)
        self.assertEqual(
            payload["envVariables"]["EVOSSEARCH_ADMIN_TOKEN"],
            ENV_SECRET_REDACTION,
        )

    def test_secure_deployment_gate_requires_named_auth_and_strict_roles(self) -> None:
        config.SECURE_DEPLOYMENT_REQUIRED = True
        config.AUTH_ENABLED = False

        disabled = oldapp._check_auth_ready()
        self.assertFalse(disabled["ok"])
        self.assertTrue(disabled["required"])
        self.assertEqual(disabled["status"], "disabled")

        config.AUTH_ENABLED = True
        config.DB_STRICT_RUNTIME_ROLES = False
        misconfigured = oldapp._check_auth_ready()
        self.assertFalse(misconfigured["ok"])
        self.assertTrue(misconfigured["required"])
        self.assertEqual(misconfigured["status"], "misconfigured")
        self.assertIn("STRICT_RUNTIME_ROLES", misconfigured["error"])

    def test_mutating_endpoint_requires_admin_token(self) -> None:
        config.ADMIN_TOKEN = ""
        resp = self.client.post("/probes/delete", json={"id": "missing-probe"})
        self.assertEqual(resp.status_code, 503)
        payload = resp.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("error", payload)

    def test_mutating_endpoint_allows_valid_admin_token(self) -> None:
        config.ADMIN_TOKEN = "unit-token"
        denied = self.client.post("/probes/delete", json={"id": "missing-probe"})
        self.assertEqual(denied.status_code, 401)

        allowed = self.client.post(
            "/probes/delete",
            headers={"X-Admin-Token": "unit-token"},
            json={"id": "missing-probe"},
        )
        # Route-specific validation after auth guard.
        self.assertEqual(allowed.status_code, 404)

    def test_image_endpoint_blocks_non_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_dir = root / ".clip_index" / "clip"
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / "index.faiss").write_bytes(b"stub")

            resp = self.client.get(f"/image/etc/passwd?folder={root}")
            self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
