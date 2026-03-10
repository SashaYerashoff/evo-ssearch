import tempfile
import unittest
from pathlib import Path

from oldapp import app, config


class SecuritySmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self._orig_admin_token = config.ADMIN_TOKEN
        self._orig_settings_local_only = config.SETTINGS_LOCAL_ONLY

    def tearDown(self) -> None:
        config.ADMIN_TOKEN = self._orig_admin_token
        config.SETTINGS_LOCAL_ONLY = self._orig_settings_local_only

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
