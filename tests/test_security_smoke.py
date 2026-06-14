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
        self._orig_auth_cookie_secure = getattr(
            config,
            "AUTH_COOKIE_SECURE",
            True,
        )
        self._orig_allowed_roots = list(getattr(config, "ALLOWED_ROOTS", []))
        self._orig_luxriot_password = getattr(config, "LUXRIOT_PASSWORD", "")
        self._orig_db_strict_runtime_roles = getattr(
            config,
            "DB_STRICT_RUNTIME_ROLES",
            False,
        )
        config.AUTH_ENABLED = False
        config.ALLOWED_ROOTS = []

    def tearDown(self) -> None:
        config.AUTH_ENABLED = self._orig_auth_enabled
        config.ADMIN_TOKEN = self._orig_admin_token
        config.SETTINGS_LOCAL_ONLY = self._orig_settings_local_only
        config.SECURE_DEPLOYMENT_REQUIRED = self._orig_secure_deployment_required
        config.AUTH_COOKIE_SECURE = self._orig_auth_cookie_secure
        config.ALLOWED_ROOTS = self._orig_allowed_roots
        config.LUXRIOT_PASSWORD = self._orig_luxriot_password
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
            "EVA_DATABASE_DSN": "postgresql://eva:db-secret@db.internal/eva",
            "EVOSSEARCH_PORT": "5000",
        }
        redacted = _redact_env_map(current)

        self.assertEqual(redacted["EVOSSEARCH_LM_API_KEY"], ENV_SECRET_REDACTION)
        self.assertEqual(redacted["EVOSSEARCH_LUXRIOT_PASSWORD"], ENV_SECRET_REDACTION)
        self.assertEqual(redacted["EVA_DATABASE_DSN"], ENV_SECRET_REDACTION)
        self.assertEqual(redacted["EVOSSEARCH_PORT"], "5000")
        self.assertNotIn("lm-secret", str(redacted))
        self.assertNotIn("camera-secret", str(redacted))
        self.assertNotIn("db-secret", str(redacted))

        submitted = dict(redacted)
        submitted["EVOSSEARCH_PORT"] = "5001"
        restored = _restore_redacted_env_secrets(submitted, current)
        self.assertEqual(restored["EVOSSEARCH_LM_API_KEY"], "lm-secret")
        self.assertEqual(restored["EVOSSEARCH_LUXRIOT_PASSWORD"], "camera-secret")
        self.assertEqual(
            restored["EVA_DATABASE_DSN"],
            "postgresql://eva:db-secret@db.internal/eva",
        )
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

    def test_secure_deployment_gate_checks_cookie_roots_and_placeholder_secrets(self) -> None:
        config.SECURE_DEPLOYMENT_REQUIRED = True
        config.AUTH_COOKIE_SECURE = False
        config.ALLOWED_ROOTS = []
        config.ADMIN_TOKEN = "12345"
        config.LUXRIOT_PASSWORD = "123"

        result = oldapp._check_deployment_security_ready()

        self.assertFalse(result["ok"])
        self.assertTrue(result["required"])
        self.assertEqual(result["status"], "misconfigured")
        self.assertGreaterEqual(len(result["issues"]), 4)
        serialized = " ".join(result["issues"])
        self.assertIn("AUTH_COOKIE_SECURE", serialized)
        self.assertIn("ALLOWED_ROOTS", serialized)
        self.assertIn("ADMIN_TOKEN", serialized)
        self.assertIn("LUXRIOT_PASSWORD", serialized)

    def test_secure_deployment_gate_accepts_hardened_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.SECURE_DEPLOYMENT_REQUIRED = True
            config.AUTH_COOKIE_SECURE = True
            config.ALLOWED_ROOTS = [tmp_dir]
            config.ADMIN_TOKEN = ""
            config.LUXRIOT_PASSWORD = "client-password-with-enough-length-2026"

            result = oldapp._check_deployment_security_ready()

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["allowed_roots_count"], 1)

    def test_secure_ready_response_hides_component_details(self) -> None:
        config.SECURE_DEPLOYMENT_REQUIRED = True
        config.AUTH_ENABLED = False
        with (
            patch(
                "oldapp._check_database_ready",
                return_value={
                    "ok": False,
                    "status": "error",
                    "required": True,
                    "path": "/srv/eva/private/archive",
                    "tenant_id": "tenant-secret",
                },
            ),
            patch(
                "oldapp._check_luxriot_ready",
                return_value={
                    "ok": False,
                    "status": "error",
                    "required": True,
                    "base_url": "http://camera.internal:8080",
                    "error": "ConnectionError camera.internal",
                },
            ),
        ):
            response = self.client.get("/ready")

        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(
            payload["checks"]["database"],
            {"ok": False, "status": "error", "required": True},
        )
        self.assertEqual(
            payload["checks"]["luxriot"],
            {"ok": False, "status": "error", "required": True},
        )
        serialized = str(payload)
        self.assertNotIn("/srv/eva/private", serialized)
        self.assertNotIn("camera.internal", serialized)
        self.assertNotIn("tenant-secret", serialized)

    def test_secure_ready_details_requires_named_auth(self) -> None:
        config.SECURE_DEPLOYMENT_REQUIRED = True
        config.AUTH_ENABLED = False

        response = self.client.get("/ready?details=1")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json()["error"],
            "Named-user authentication is disabled",
        )

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

    def test_image_endpoint_does_not_expose_send_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_dir = root / ".clip_index" / "clip"
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / "index.faiss").write_bytes(b"stub")
            image_path = root / "frame.jpg"
            image_path.write_bytes(b"stub-jpeg")

            with patch(
                "oldapp.send_file",
                side_effect=RuntimeError("/srv/eva/private/frame.jpg failed"),
            ):
                resp = self.client.get(
                    f"/image?folder={root}&image_path={image_path.name}"
                )

        self.assertEqual(resp.status_code, 500)
        self.assertIn(b"Image unavailable", resp.data)
        self.assertNotIn(b"/srv/eva/private", resp.data)

    def test_describe_image_not_found_does_not_echo_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_dir = root / ".clip_index" / "clip"
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / "index.faiss").write_bytes(b"stub")

            resp = self.client.post(
                "/describe_image",
                json={
                    "folder": str(root),
                    "image_path": "/srv/eva/private/missing.jpg",
                },
            )

        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload["error"], "Image not found")
        self.assertNotIn("/srv/eva/private", str(payload))

    def test_video_understanding_not_found_does_not_echo_path(self) -> None:
        resp = self.client.post(
            "/video_understanding",
            json={"video": "/srv/eva/private/missing.mp4"},
        )

        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertEqual(payload["error"], "Video file not found")
        self.assertNotIn("/srv/eva/private", str(payload))


if __name__ == "__main__":
    unittest.main()
