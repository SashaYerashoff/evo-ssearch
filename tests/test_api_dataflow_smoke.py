import base64
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import patch

from oldapp import app, config


def _normalize_frontend_path(raw_path: str) -> str:
    path = re.sub(r"\$\{[^}]+\}", "", raw_path).strip()
    path = path.split("?", 1)[0]
    if path.endswith("/") and path != "/":
        path = path[:-1]
    return path


def _route_to_regex(route_path: str) -> re.Pattern[str]:
    normalized = route_path.rstrip("/") or "/"
    if normalized == "/":
        return re.compile(r"^/$")
    parts = normalized.strip("/").split("/")
    regex_parts = []
    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            regex_parts.append(r"[^/]+")
        else:
            regex_parts.append(re.escape(part))
    return re.compile(r"^/" + "/".join(regex_parts) + r"$")


def _route_matches_frontend(route_path: str, frontend_path: str) -> bool:
    route_norm = route_path.rstrip("/") or "/"
    front_norm = frontend_path.rstrip("/") or "/"
    if _route_to_regex(route_norm).match(front_norm):
        return True
    if "<" in route_norm and ">" in route_norm:
        dynamic_prefix = route_norm.split("<", 1)[0].rstrip("/")
        if dynamic_prefix and (front_norm == dynamic_prefix or front_norm.startswith(dynamic_prefix + "/")):
            return True
    return False


def _collect_frontend_and_backend_paths() -> Tuple[Set[str], Set[str]]:
    root = Path(__file__).resolve().parent.parent
    sources = [
        root.joinpath("oldapp.py"),
        root.joinpath("templates", "index.html"),
        root.joinpath("static", "js", "app.js"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in sources if path.exists())
    fetch_paths = re.findall(r"fetch\(\s*['\"](/[^'\"]+)['\"]", source)
    fetch_paths.extend(re.findall(r"fetch\(\s*`(/[^`]*)`", source))
    src_paths = re.findall(r"\.src\s*=\s*['\"](/[^'\"]+)['\"]", source)
    src_paths.extend(re.findall(r"\.src\s*=\s*`(/[^`]*)`", source))
    frontend_paths = {_normalize_frontend_path(path) for path in fetch_paths + src_paths}
    backend_routes = {path for _, path in re.findall(r"@app\.route\(\s*([`'\"])(/[^`'\"]+)\1", source)}
    return frontend_paths, backend_routes


class ApiDataflowSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self._orig_admin_token = config.ADMIN_TOKEN
        self._orig_settings_local_only = config.SETTINGS_LOCAL_ONLY

    def tearDown(self) -> None:
        config.ADMIN_TOKEN = self._orig_admin_token
        config.SETTINGS_LOCAL_ONLY = self._orig_settings_local_only

    def test_frontend_endpoints_map_to_backend_routes(self) -> None:
        frontend_paths, backend_routes = _collect_frontend_and_backend_paths()
        unmatched = sorted(
            path
            for path in frontend_paths
            if not any(_route_matches_frontend(route, path) for route in backend_routes)
        )
        self.assertEqual(unmatched, [], f"Frontend endpoints missing backend routes: {unmatched}")

    def test_backend_only_endpoints_are_known(self) -> None:
        frontend_paths, backend_routes = _collect_frontend_and_backend_paths()
        backend_only = {
            route
            for route in backend_routes
            if not any(_route_matches_frontend(route, path) for path in frontend_paths)
        }
        allowed_backend_only = {
            "/",
            "/branding/logo",
            "/agent/skills/create",
            "/detections/image",
            "/favicon.ico",
            "/health",
            "/auth/login",
            "/auth/logout",
            "/auth/me",
            "/image",
            "/image/<path:filepath>",
            "/js/app.js",
            "/lm/models",
            "/ready",
        }
        unexpected_backend_only = backend_only - allowed_backend_only
        self.assertEqual(unexpected_backend_only, set(), f"Unexpected backend-only endpoints: {sorted(unexpected_backend_only)}")

    def test_non_mutating_endpoints_return_validation_errors_not_500(self) -> None:
        config.ADMIN_TOKEN = ""
        checks: List[Tuple[str, str, Dict[str, Any], Set[int]]] = [
            ("GET", "/", {}, {200}),
            ("GET", "/health", {}, {200}),
            ("GET", "/ready", {}, {200, 503}),
            ("GET", "/settings", {}, {200, 403}),
            ("POST", "/check_index", {"json": {}}, {400}),
            ("POST", "/search", {"json": {}}, {400}),
            ("POST", "/search_by_image", {"data": {}}, {400}),
            ("POST", "/video_understanding", {"json": {}}, {400}),
            ("POST", "/describe_image", {"json": {}}, {400}),
            ("GET", "/comments", {}, {400}),
            ("POST", "/commented_images", {"json": {}}, {400}),
            ("POST", "/segment_from_point", {"json": {}}, {400}),
            ("GET", "/probes/list", {}, {200}),
            ("GET", "/probes/status", {}, {200}),
            ("GET", "/luxriot/session", {}, {200}),
        ]
        for method, path, kwargs, allowed in checks:
            with self.subTest(endpoint=path, method=method):
                req_kwargs = dict(kwargs)
                resp = self.client.open(path=path, method=method, data=req_kwargs.get("data"), json=req_kwargs.get("json"))
                if 503 not in allowed:
                    self.assertLess(resp.status_code, 500)
                self.assertIn(resp.status_code, allowed)

    def test_health_and_ready_payloads_are_structured(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")

        ready = self.client.get("/ready")
        self.assertIn(ready.status_code, {200, 503})
        payload = ready.get_json()
        self.assertIn(payload["status"], {"ready", "not_ready"})
        self.assertIn("postgresql", payload["checks"])
        self.assertIn("authentication", payload["checks"])
        self.assertIn("database", payload["checks"])
        self.assertIn("embedder", payload["checks"])
        self.assertIn("luxriot", payload["checks"])

    def test_mutating_frontend_endpoints_require_token(self) -> None:
        config.ADMIN_TOKEN = ""
        checks: List[Tuple[str, Dict[str, Any]]] = [
            ("/settings", {"json": {}}),
            ("/index", {"json": {}}),
            ("/comments", {"json": {}}),
            ("/probes/save", {"json": {}}),
            ("/probes/start_capture", {"json": {}}),
            ("/probes/stop_capture", {"json": {}}),
            ("/probes/run", {"json": {}}),
            ("/probes/query", {"json": {}}),
            ("/probes/delete", {"json": {}}),
            ("/luxriot/start_capture", {"json": {}}),
            ("/luxriot/stop_capture", {"json": {}}),
            ("/luxriot/flush_capture", {"json": {}}),
        ]
        for path, kwargs in checks:
            with self.subTest(endpoint=path):
                resp = self.client.post(path, **kwargs)
                self.assertEqual(resp.status_code, 503)
                payload_raw = resp.get_json(silent=True)
                payload = payload_raw if isinstance(payload_raw, dict) else {}
                self.assertIn("error", payload)

    def test_luxriot_read_endpoints_work_with_stubs(self) -> None:
        with patch("oldapp.luxriot_manager.get_channels", return_value=[{"id": 7, "name": "Cam 7"}]):
            channels_resp = self.client.get("/luxriot/channels?force=1")
            self.assertEqual(channels_resp.status_code, 200)
            channels_payload = channels_resp.get_json()
            self.assertIsInstance(channels_payload, dict)
            self.assertEqual(len(channels_payload.get("channels", [])), 1)

        encoded = base64.b64encode(b"mock-jpeg-bytes").decode("ascii")
        with patch("oldapp.luxriot_manager.get_snapshot_base64", return_value=(encoded, {"width": 320, "height": 180})):
            snapshot_resp = self.client.get("/luxriot/snapshot/7")
            self.assertEqual(snapshot_resp.status_code, 200)
            self.assertEqual(snapshot_resp.headers.get("Content-Type"), "image/jpeg")
            self.assertEqual(snapshot_resp.headers.get("X-Image-Width"), "320")
            self.assertEqual(snapshot_resp.headers.get("X-Image-Height"), "180")
            self.assertEqual(snapshot_resp.data, b"mock-jpeg-bytes")

    def test_luxriot_capture_flow_with_token_and_stubs(self) -> None:
        config.ADMIN_TOKEN = "unit-token"
        headers = {"X-Admin-Token": "unit-token"}

        with patch("oldapp.luxriot_manager.start_session", return_value={"running": True, "channel_id": 7}):
            start_resp = self.client.post("/luxriot/start_capture", headers=headers, json={"channel_id": 7, "batch_size": 12})
            self.assertEqual(start_resp.status_code, 200)
            start_payload = start_resp.get_json()
            self.assertIsInstance(start_payload, dict)
            self.assertTrue(start_payload.get("success"))

        with patch("oldapp.luxriot_manager.stop_session", return_value={"running": False, "channel_id": 7}):
            stop_resp = self.client.post("/luxriot/stop_capture", headers=headers, json={"channel_id": 7})
            self.assertEqual(stop_resp.status_code, 200)
            stop_payload = stop_resp.get_json()
            self.assertIsInstance(stop_payload, dict)
            self.assertTrue(stop_payload.get("success"))

        flush_result = {"success": True, "summary": "stub summary", "alerts": [], "frame_count": 16}
        with patch("oldapp.luxriot_manager.flush_session", return_value=flush_result):
            flush_resp = self.client.post("/luxriot/flush_capture", headers=headers, json={"channel_id": 7})
            self.assertEqual(flush_resp.status_code, 200)
            flush_payload = flush_resp.get_json()
            self.assertIsInstance(flush_payload, dict)
            self.assertTrue(flush_payload.get("success"))

    def test_probe_tracking_flow_with_token_and_stubs(self) -> None:
        config.ADMIN_TOKEN = "unit-token"
        headers = {"X-Admin-Token": "unit-token"}

        with patch("oldapp.luxriot_manager.start_probe_capture", return_value={"running": True, "channel_id": 103}):
            start_resp = self.client.post("/probes/start_capture", headers=headers, json={"channel_id": 103, "fps": 1.5})
            self.assertEqual(start_resp.status_code, 200)
            start_payload = start_resp.get_json()
            self.assertIsInstance(start_payload, dict)
            self.assertTrue(start_payload.get("success"))

        query_result = {
            "status": {"frames": 12, "window_sec": 300.0},
            "results": [
                {
                    "pos_score": 0.88,
                    "neg_score": 0.12,
                    "margin": 0.76,
                    "timestamp_ms": 1700000000000,
                    "image_path": "/tmp/frame.jpg",
                }
            ],
        }
        with patch("oldapp.probe_manager.query", return_value=query_result), patch(
            "oldapp.luxriot_manager.send_bookmark_event", return_value={"success": True}
        ):
            query_resp = self.client.post(
                "/probes/query",
                headers=headers,
                json={"channel_id": 103, "positives": ["person"], "bookmark": True},
            )
            self.assertEqual(query_resp.status_code, 200)
            query_payload = query_resp.get_json()
            self.assertIsInstance(query_payload, dict)
            self.assertEqual(len(query_payload.get("results", [])), 1)
            self.assertEqual(len(query_payload.get("recent_hits", [])), 1)

        with patch("oldapp.luxriot_manager.stop_probe_capture", return_value={"running": False, "channel_id": 103}):
            stop_resp = self.client.post("/probes/stop_capture", headers=headers, json={"channel_id": 103})
            self.assertEqual(stop_resp.status_code, 200)
            stop_payload = stop_resp.get_json()
            self.assertIsInstance(stop_payload, dict)
            self.assertTrue(stop_payload.get("success"))


if __name__ == "__main__":
    unittest.main()
