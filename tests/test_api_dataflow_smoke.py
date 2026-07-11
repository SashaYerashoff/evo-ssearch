import base64
import hashlib
import inspect
import re
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import patch

from PIL import Image

from oldapp import (
    _MUTATION_ENDPOINT_PERMISSIONS,
    _SENSITIVE_ENDPOINT_PERMISSIONS,
    _build_detection_search_result,
    _env_precedence_report,
    _store_vlm_summary_archive_frames,
    ProbesStore,
    app,
    config,
)


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
    # URL-builder helpers (media broker, attention stream, image fetch) hand
    # root-relative template literals to fetch()/media elements indirectly.
    builder_paths = re.findall(r"return\s+`(/[^`]*)`", source)
    frontend_paths = {
        _normalize_frontend_path(path)
        for path in fetch_paths + src_paths + builder_paths
    }
    backend_routes = {path for _, path in re.findall(r"@app\.route\(\s*([`'\"])(/[^`'\"]+)\1", source)}
    return frontend_paths, backend_routes


class ApiDataflowSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self._orig_auth_enabled = config.AUTH_ENABLED
        self._orig_admin_token = config.ADMIN_TOKEN
        self._orig_settings_local_only = config.SETTINGS_LOCAL_ONLY
        self._orig_offline_video_enabled = config.OFFLINE_VIDEO_ENABLED
        self._orig_probe_snap_enabled = config.PROBE_SNAP_ENABLED
        self._orig_indexed_folder_enabled = config.INDEXED_FOLDER_ENABLED
        config.AUTH_ENABLED = False
        config.OFFLINE_VIDEO_ENABLED = True
        config.PROBE_SNAP_ENABLED = True
        config.INDEXED_FOLDER_ENABLED = True

    def tearDown(self) -> None:
        config.AUTH_ENABLED = self._orig_auth_enabled
        config.ADMIN_TOKEN = self._orig_admin_token
        config.SETTINGS_LOCAL_ONLY = self._orig_settings_local_only
        config.OFFLINE_VIDEO_ENABLED = self._orig_offline_video_enabled
        config.PROBE_SNAP_ENABLED = self._orig_probe_snap_enabled
        config.INDEXED_FOLDER_ENABLED = self._orig_indexed_folder_enabled

    def test_frontend_endpoints_map_to_backend_routes(self) -> None:
        frontend_paths, backend_routes = _collect_frontend_and_backend_paths()
        unmatched = sorted(
            path
            for path in frontend_paths
            if not any(_route_matches_frontend(route, path) for route in backend_routes)
        )
        self.assertEqual(unmatched, [], f"Frontend endpoints missing backend routes: {unmatched}")

    def test_probe_runtime_patch_preserves_operator_edits_and_never_resurrects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProbesStore(Path(temp_dir) / "probes.json")
            original = store.upsert_probe(
                {
                    "id": "probe-1",
                    "name": "Door watch",
                    "bookmark": True,
                    "pos_floor": 0.2,
                }
            )
            stale_daemon_copy = dict(original)
            store.upsert_probe(
                {
                    **original,
                    "bookmark": False,
                    "pos_floor": 0.8,
                }
            )

            patched = store.patch_probe_runtime(
                "probe-1",
                {
                    "last_hit": {"timestamp_ms": 123},
                    "recent_hits": [{"timestamp_ms": 123}],
                    "bookmark": True,
                },
            )

            self.assertIsNotNone(patched)
            self.assertFalse(patched["bookmark"])
            self.assertEqual(patched["pos_floor"], 0.8)
            self.assertEqual(patched["last_hit"]["timestamp_ms"], 123)
            self.assertTrue(store.delete_probe("probe-1"))
            self.assertIsNone(
                store.patch_probe_runtime(
                    "probe-1",
                    {"last_hit": stale_daemon_copy},
                )
            )
            self.assertEqual(store.list_probes(), [])

    def test_env_precedence_reports_process_difference_without_values(self) -> None:
        file_secret_hash = hashlib.sha256(b"different-secret").hexdigest()
        report = _env_precedence_report(
            file_map={
                "EVOSSEARCH_LUXRIOT_PASSWORD": "file-secret",
                "EVOSSEARCH_PORT": "5443",
            },
            process_keys={
                "EVOSSEARCH_LUXRIOT_PASSWORD",
                "EVOSSEARCH_HOST",
            },
            process_value_hashes={
                "EVOSSEARCH_LUXRIOT_PASSWORD": file_secret_hash,
            },
        )

        self.assertEqual(
            report["different_process_and_file_keys"],
            ["EVOSSEARCH_LUXRIOT_PASSWORD"],
        )
        self.assertIn("EVOSSEARCH_HOST", report["process_environment_keys"])
        serialized = str(report)
        self.assertNotIn("file-secret", serialized)

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
            "/detections/thumbnail/<int:detection_id>",
            "/detections/diagnostics",
            "/favicon.ico",
            "/health",
            "/auth/login",
            "/auth/logout",
            "/auth/me",
            "/auth/roles",
            "/auth/sessions",
            "/auth/sessions/<session_id>/revoke",
            "/auth/users",
            "/auth/users/<user_id>",
            "/auth/users/<user_id>/revoke-sessions",
            "/audit/events",
            "/image",
            "/image/<path:filepath>",
            "/js/app.js",
            "/lm/admission",
            "/lm/models",
            "/luxriot/recent_frame/<int:channel_id>",
            "/ready",
        }
        unexpected_backend_only = backend_only - allowed_backend_only
        self.assertEqual(unexpected_backend_only, set(), f"Unexpected backend-only endpoints: {sorted(unexpected_backend_only)}")

    def test_vlm_summary_archive_frames_write_detection_records(self) -> None:
        class Store:
            def __init__(self) -> None:
                self.records: List[Dict[str, Any]] = []

            def add_detections(self, records: List[Dict[str, Any]]) -> int:
                self.records.extend(records)
                return len(records)

        store = Store()
        entry = {
            "channel_id": 7,
            "run_id": "run-7",
            "summary": "Observed motion near the entrance.\nALERTS_JSON: {}",
            "frame_count": 12,
            "batch_size": 12,
            "created_at": 100.0,
            "batch_start_ms": 100000,
            "batch_end_ms": 105000,
            "duration_sec": 1.25,
            "prompt": "Watch for people.",
            "alert_counts": {"normal": 1},
            "alert_total": 1,
            "bookmarks_sent": 1,
            "archive_frames": [
                {
                    "anchor_role": "first",
                    "frame_index": 0,
                    "timestamp_ms": 100000,
                    "thumbnail": "frame-one",
                    "width": 1280,
                    "height": 720,
                },
                {
                    "anchor_role": "last",
                    "frame_index": 11,
                    "timestamp_ms": 105000,
                    "thumbnail": "frame-two",
                    "width": 1280,
                    "height": 720,
                },
            ],
        }

        with (
            patch("oldapp.detections_store", store),
            patch("oldapp._embed_thumbnail_b64", return_value=None),
            patch("oldapp._apply_archive_retention", return_value={"ok": True}),
        ):
            result = _store_vlm_summary_archive_frames(entry)

        self.assertEqual(result["attempted"], 3)
        self.assertEqual(result["inserted"], 3)
        self.assertEqual(result["summary_frames"], 2)
        self.assertEqual(result["alert_frames"], 1)
        self.assertEqual([record["source"] for record in store.records], ["vlm_summary", "vlm_summary", "vlm_alert"])
        self.assertEqual(store.records[0]["probe_id"], "vlm_summary:7")
        self.assertEqual(store.records[2]["probe_id"], "vlm_alert:7")
        self.assertTrue(store.records[2]["bookmark_sent"])
        self.assertIn("run-7", store.records[0]["dedupe_key"])

    def test_vlm_summary_archive_frames_write_one_record_per_alert_event(self) -> None:
        class Store:
            def __init__(self) -> None:
                self.records: List[Dict[str, Any]] = []

            def add_detections(self, records: List[Dict[str, Any]]) -> int:
                self.records.extend(records)
                return len(records)

        store = Store()
        entry = {
            "channel_id": 7,
            "run_id": "run-7",
            "summary": "Two independent test triggers.",
            "frame_count": 12,
            "batch_size": 12,
            "created_at": 100.0,
            "batch_start_ms": 100000,
            "batch_end_ms": 105000,
            "alert_counts": {"info": 1, "low": 1},
            "alert_total": 2,
            "bookmarks_sent": 2,
            "alert_events": [
                {
                    "title": "Thumbs up",
                    "description": "Person shows a thumbs-up gesture.",
                    "severity": "info",
                    "timestamp_ms": 100100,
                    "delivery_status": "sent",
                },
                {
                    "title": "Union Jack mug drink",
                    "description": "Person drinks from a mug with Union Jack art.",
                    "severity": "low",
                    "timestamp_ms": 104900,
                    "delivery_status": "sent",
                },
            ],
            "archive_frames": [
                {"anchor_role": "first", "frame_index": 0, "timestamp_ms": 100000, "thumbnail": "frame-one"},
                {"anchor_role": "last", "frame_index": 11, "timestamp_ms": 105000, "thumbnail": "frame-two"},
            ],
        }

        with (
            patch("oldapp.detections_store", store),
            patch("oldapp._embed_thumbnail_b64", return_value=None),
            patch("oldapp._apply_archive_retention", return_value={"ok": True}),
        ):
            result = _store_vlm_summary_archive_frames(entry)

        alert_records = [record for record in store.records if record["source"] == "vlm_alert"]
        self.assertEqual(result["alert_frames"], 2)
        self.assertEqual(len(alert_records), 2)
        self.assertEqual([record["severity"] for record in alert_records], ["info", "low"])
        self.assertEqual(alert_records[0]["payload"]["alert_event"]["title"], "Thumbs up")
        self.assertEqual(alert_records[1]["payload"]["alert_event"]["title"], "Union Jack mug drink")

    def test_vlm_alert_archive_anchor_prefers_snapshot_reference(self) -> None:
        class Store:
            def __init__(self) -> None:
                self.records: List[Dict[str, Any]] = []

            def add_detections(self, records: List[Dict[str, Any]]) -> int:
                self.records.extend(records)
                return len(records)

        store = Store()
        entry = {
            "channel_id": 120,
            "run_id": "run-drift",
            "summary": "Vehicle drifts through Snapshots 8-12 with tire smoke.",
            "frame_count": 12,
            "batch_size": 12,
            "created_at": 100.0,
            "batch_start_ms": 100000,
            "batch_end_ms": 112000,
            "alert_counts": {"high": 1},
            "alert_total": 1,
            "bookmarks_sent": 1,
            "alert_events": [
                {
                    "title": "Vehicle drift with tire smoke",
                    "description": "Silver sedan performs a drift and emits tire smoke in Snapshots 8-12.",
                    "severity": "high",
                    "timestamp_ms": 100000,
                    "delivery_status": "sent",
                },
            ],
            "vector_signal": {
                "version": 1,
                "semantics": "vector_homeostasis_attention_signal_not_visual_proof",
                "clip_probe_signals": [
                    {
                        "name": "vehicle drift candidate",
                        "probe_id": "probe-drift",
                        "p": 0.41,
                        "n": 0.12,
                        "m": 0.29,
                        "apex_frame": 11,
                    }
                ],
                "road_cv_cues": [
                    {
                        "cue_type": "road_motion_burst",
                        "score": 0.73,
                        "apex_frame": 10,
                        "zone_name": "auto_motion_zone",
                    }
                ],
            },
            "archive_frames": [
                {
                    "anchor_role": "first" if frame_index == 0 else "sample",
                    "frame_index": frame_index,
                    "timestamp_ms": 100000 + frame_index * 1000,
                    "thumbnail": f"frame-{frame_index}",
                }
                for frame_index in range(12)
            ],
        }

        with (
            patch("oldapp.detections_store", store),
            patch("oldapp._embed_thumbnail_b64", return_value=None),
            patch("oldapp._apply_archive_retention", return_value={"ok": True}),
        ):
            result = _store_vlm_summary_archive_frames(entry)

        alert_records = [record for record in store.records if record["source"] == "vlm_alert"]
        self.assertEqual(result["alert_frames"], 1)
        self.assertEqual(len(alert_records), 1)
        payload = alert_records[0]["payload"]
        self.assertEqual(alert_records[0]["thumbnail_b64"], "frame-11")
        self.assertEqual(payload["anchor_frame_index"], 11)
        self.assertEqual(payload["anchor_snapshot_hint"], 12)
        self.assertEqual(payload["anchor_selection"], "alert_snapshot_reference")
        self.assertEqual(payload["vector_signal"]["clip_probe_signals"][0]["name"], "vehicle drift candidate")
        self.assertEqual(payload["vector_signal"]["road_cv_cues"][0]["cue_type"], "road_motion_burst")

    def test_detections_list_passes_archive_source_filter(self) -> None:
        captured: Dict[str, Any] = {}

        class Store:
            def list_detections(self, **kwargs):
                captured.update(kwargs)
                return [], 0

        with patch("oldapp.detections_store", Store()):
            response = self.client.get(
                "/detections/list?source=vlm_summary&channel_id=7&hours=1"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(captured["source"], "vlm_summary")
        self.assertEqual(response.get_json()["filters"]["source"], "vlm_summary")

    def test_detections_list_normalizes_probe_source_aliases(self) -> None:
        captured: Dict[str, Any] = {}

        class Store:
            def list_detections(self, **kwargs):
                captured.update(kwargs)
                return [], 0

        with patch("oldapp.detections_store", Store()):
            response = self.client.get(
                "/detections/list?source=probes_run&channel_id=7&since_ms=1000&until_ms=2000"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(captured["source"], "probe")
        self.assertEqual(captured["since_ms"], 1000)
        self.assertEqual(captured["until_ms"], 2000)
        self.assertEqual(response.get_json()["filters"]["source"], "probe")

    def test_detection_text_search_passes_archive_source_filter(self) -> None:
        captured: Dict[str, Any] = {}

        def _search(**kwargs):
            captured.update(kwargs)
            return []

        with (
            patch("oldapp.get_clip_text_embedding", return_value=object()),
            patch("oldapp._search_detections_archive", side_effect=_search),
        ):
            response = self.client.post(
                "/detections/search_text",
                json={
                    "query": "person near entrance",
                    "source": "vlm_summary",
                    "channel_id": 7,
                    "hours": 1,
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(captured["source"], "vlm_summary")
        self.assertEqual(response.get_json()["filters"]["source"], "vlm_summary")

    def test_detection_search_result_preserves_vlm_payload_for_review(self) -> None:
        result = _build_detection_search_result(
            item={
                "id": 42,
                "timestamp_ms": 1780000012345,
                "probe_id": "vlm_summary:112",
                "probe_name": "VLM summary ch 112",
                "channel_id": 112,
                "severity": "info",
                "source": "vlm_summary",
                "payload": {
                    "source": "vlm_summary",
                    "run_id": "run-demo",
                    "batch_start_ms": 1780000010000,
                    "batch_end_ms": 1780000020000,
                    "frame_timestamp_ms": 1780000012345,
                    "frame_index": 4,
                    "anchor_role": "sample",
                    "summary": "A public lobby scene is calm.",
                    "summary_truncated": False,
                },
            },
            score=0.91,
            clip_score=0.91,
            dino_score=None,
            mode="clip",
            alpha=0.0,
            dino_fallback=False,
        )

        self.assertEqual(result["source"], "vlm_summary")
        self.assertEqual(result["payload"]["summary"], "A public lobby scene is calm.")
        self.assertEqual(result["summary"], "A public lobby scene is calm.")
        self.assertEqual(result["run_id"], "run-demo")
        self.assertEqual(result["batch_start_ms"], 1780000010000)
        self.assertEqual(result["frame_timestamp_ms"], 1780000012345)

    def test_detections_diagnostics_reports_sources_without_thumbnail_payloads(self) -> None:
        class Store:
            def summarize_by_source(self, **kwargs):
                return [
                    {
                        "source": "vlm_summary",
                        "row_count": 2,
                        "thumbnail_count": 2,
                        "clip_count": 2,
                        "dino_count": 0,
                        "channel_count": 1,
                        "oldest_timestamp_ms": 100000,
                        "newest_timestamp_ms": 105000,
                    }
                ]

            def list_detections(self, **kwargs):
                return [
                    {
                        "id": 11,
                        "source": "vlm_summary",
                        "channel_id": 7,
                        "probe_id": "vlm_summary:7",
                        "probe_name": "VLM summary ch 7",
                        "timestamp_ms": 105000,
                        "severity": "info",
                        "thumbnail": "abcdef",
                        "has_clip": True,
                        "has_dino": False,
                        "shard_key": "ch7:19700101",
                    }
                ], 1

            def storage_summary(self):
                return {
                    "backend": "test",
                    "row_count": 2,
                    "thumbnail_bytes": 6,
                    "clip_vec_bytes": 16,
                    "dino_vec_bytes": 0,
                    "payload_json_bytes": 10,
                    "oldest_timestamp_ms": 100000,
                    "newest_timestamp_ms": 105000,
                }

        with patch("oldapp.detections_store", Store()):
            response = self.client.get(
                "/detections/diagnostics?channel_id=7&source=vlm_summary&hours=1"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["sources"][0]["source"], "vlm_summary")
        self.assertEqual(payload["recent"][0]["thumbnail_chars"], 6)
        self.assertTrue(payload["recent"][0]["has_thumbnail"])
        self.assertNotIn("thumbnail", payload["recent"][0])

    def test_non_public_routes_are_declared_in_security_surface(self) -> None:
        public_endpoints = {
            "static",
            "home",
            "favicon",
            "branding_logo",
            "serve_app_js",
            "health",
            "ready",
            "auth_login",
        }
        internally_guarded_reads = {
            "auth_me",
            "auth_roles",
            "auth_users",
            "auth_user",
            "auth_sessions",
            "get_settings",
            "get_settings_env",
        }
        internally_guarded_mutations = {
            "auth_logout",
            "auth_users",
            "auth_user",
            "auth_user_revoke_sessions",
            "auth_session_revoke",
        }
        read_guard_markers = (
            "_settings_guard(",
            "_auth_admin_guard(",
            "_session_guard(",
        )
        mutation_guard_markers = (
            "_mutation_guard",
            "_auth_admin_guard(",
            "_settings_guard(write=True",
        )
        read_only_post_endpoints = {
            "check_index",
            "get_commented_images",
            "video_understanding",
            "describe_image",
            "search",
            "search_by_image",
            "search_by_mask",
            "segment_from_point",
            "detections_search_text",
            "detections_search_image",
            "luxriot_snapshot_capture",
        }
        for rule in app.url_map.iter_rules():
            endpoint = str(rule.endpoint)
            if endpoint in public_endpoints:
                continue
            methods = set(rule.methods or set()) - {"HEAD", "OPTIONS"}
            source = inspect.getsource(app.view_functions[endpoint])
            has_get = "GET" in methods
            for method in methods:
                with self.subTest(endpoint=endpoint, method=method, rule=rule.rule):
                    if method == "GET":
                        guarded = (
                            endpoint in _SENSITIVE_ENDPOINT_PERMISSIONS
                            or endpoint in internally_guarded_reads
                            or any(marker in source for marker in read_guard_markers)
                        )
                        self.assertTrue(
                            guarded,
                            f"{endpoint} GET reads sensitive data but is not guarded",
                        )
                        continue

                    if method in {"POST", "PATCH", "PUT", "DELETE"}:
                        mixed_route = has_get
                        if endpoint in read_only_post_endpoints and not mixed_route:
                            guarded = endpoint in _SENSITIVE_ENDPOINT_PERMISSIONS
                        else:
                            guarded = (
                                endpoint in _MUTATION_ENDPOINT_PERMISSIONS
                                or endpoint in internally_guarded_mutations
                                or any(marker in source for marker in mutation_guard_markers)
                            )
                        self.assertTrue(
                            guarded,
                            f"{endpoint} {method} mutates but is not guarded",
                        )
                        continue

                    self.fail(f"Unhandled HTTP method {method} for {endpoint}")

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

    def test_deployment_feature_flags_disable_unstable_surfaces_server_side(self) -> None:
        config.OFFLINE_VIDEO_ENABLED = False
        config.PROBE_SNAP_ENABLED = False
        config.INDEXED_FOLDER_ENABLED = False

        with patch("oldapp.luxriot_manager.capture_snapshot_base64") as capture:
            video = self.client.post("/video_understanding", json={})
            snap = self.client.post("/luxriot/snapshot/7/capture", json={})
            search = self.client.post("/search", json={"folder": "/tmp", "query": "person"})
            folder_describe = self.client.post(
                "/describe_image",
                json={"folder": "/tmp", "image_path": "/tmp/frame.jpg"},
            )

        self.assertEqual(video.status_code, 404)
        self.assertEqual(video.get_json()["error"], "offline_video_disabled")
        self.assertEqual(snap.status_code, 404)
        self.assertEqual(snap.get_json()["error"], "probe_snap_disabled")
        self.assertEqual(search.status_code, 404)
        self.assertEqual(folder_describe.status_code, 404)
        capture.assert_not_called()

    def test_indexed_folder_flag_does_not_disable_archive_image_description_uploads(self) -> None:
        config.INDEXED_FOLDER_ENABLED = False
        jpeg = BytesIO()
        Image.new("RGB", (24, 16), (20, 30, 40)).save(jpeg, format="JPEG")
        jpeg.seek(0)
        profile = {
            "id": "vlm-local",
            "kind": "vlm",
            "base_url": "http://localhost:1234/v1",
            "model": "qwen-vl-test",
            "timeout": 30,
        }

        with (
            patch("oldapp._resolve_lm_profile", return_value=profile),
            patch("oldapp._call_lm_chat", return_value="archive frame summary"),
        ):
            resp = self.client.post(
                "/describe_image",
                data={"image": (jpeg, "archive-frame.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200, resp.get_json())
        self.assertEqual(resp.get_json()["summary"], "archive frame summary")

    def test_describe_image_accepts_uploaded_file(self) -> None:
        jpeg = BytesIO()
        Image.new("RGB", (32, 20), (12, 34, 56)).save(jpeg, format="JPEG")
        jpeg.seek(0)
        profile = {
            "id": "vlm-local",
            "kind": "vlm",
            "base_url": "http://localhost:1234/v1",
            "model": "qwen-vl-test",
            "timeout": 30,
        }

        with (
            patch("oldapp._resolve_lm_profile", return_value=profile),
            patch("oldapp._call_lm_chat", return_value="uploaded image summary"),
        ):
            resp = self.client.post(
                "/describe_image",
                data={
                    "image": (jpeg, "sample.jpg"),
                    "prompt": "Describe this upload",
                    "model": "qwen-vl-test",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200, resp.get_json())
        payload = resp.get_json()
        self.assertEqual(payload["summary"], "uploaded image summary")
        self.assertEqual(payload["model"], "qwen-vl-test")
        self.assertEqual(payload["profile_id"], "vlm-local")
        self.assertEqual(payload["filename"], "sample.jpg")
        self.assertTrue(payload["uploaded"])
        self.assertIsInstance(payload.get("thumbnail"), str)
        self.assertGreater(len(payload.get("thumbnail") or ""), 100)

    def test_video_understanding_auto_balances_uploaded_file(self) -> None:
        thumb_buf = BytesIO()
        Image.new("RGB", (40, 24), (80, 120, 160)).save(thumb_buf, format="JPEG")
        thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode("ascii")
        profiles = {
            "default": {
                "id": "default",
                "kind": "general",
                "base_url": "http://default.local/v1",
                "model": "default-model",
                "api_key": "",
                "timeout": 120,
            },
            "vlm-a": {
                "id": "vlm-a",
                "kind": "vlm",
                "base_url": "http://vlm-a.local/v1",
                "model": "qwen-vlm-a",
                "api_key": "",
                "timeout": 300,
                "enabled": True,
            },
            "vlm-b": {
                "id": "vlm-b",
                "kind": "vlm",
                "base_url": "http://vlm-b.local/v1",
                "model": "qwen-vlm-b",
                "api_key": "",
                "timeout": 300,
                "enabled": True,
            },
        }
        captured: Dict[str, Any] = {}

        def fake_call(messages, model_override=None, profile_id=None):
            captured["messages"] = messages
            captured["model_override"] = model_override
            captured["profile_id"] = profile_id
            return "auto balanced video summary"

        with (
            patch.object(config, "LM_PROFILES", profiles),
            patch.object(config, "LM_VLM_PROFILE_ID", "vlm-a"),
            patch.object(config, "LM_VLM_BALANCER_ENABLED", True),
            patch.object(config, "LM_VLM_BALANCER_PROFILES", ("vlm-a", "vlm-b")),
            patch(
                "oldapp._sample_video_frames",
                return_value=(
                    [{"index": 0, "time_sec": 0.0, "thumbnail": thumb_b64}],
                    25.0,
                    1.0,
                ),
            ),
            patch("oldapp._call_video_understanding", side_effect=fake_call),
        ):
            resp = self.client.post(
                "/video_understanding",
                data={
                    "video": (BytesIO(b"not-a-real-video"), "sample.mp4"),
                    "prompt": "Describe this upload",
                    "model": "__auto__",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200, resp.get_json())
        payload = resp.get_json()
        self.assertEqual(payload["summary"], "auto balanced video summary")
        self.assertEqual(payload["model_selection"], "auto")
        self.assertIn(payload["assigned_profile_id"], {"vlm-a", "vlm-b"})
        self.assertEqual(captured["model_override"], payload["assigned_profile_id"])
        self.assertIsNone(captured["profile_id"])
        self.assertNotEqual(captured["model_override"], "__auto__")

    def test_video_understanding_defaults_to_agent_profile_for_offline_media(self) -> None:
        thumb_buf = BytesIO()
        Image.new("RGB", (40, 24), (80, 120, 160)).save(thumb_buf, format="JPEG")
        thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode("ascii")
        profiles = {
            "default": {
                "id": "default",
                "kind": "general",
                "base_url": "http://default.local/v1",
                "model": "default-model",
                "api_key": "",
                "timeout": 120,
            },
            "agent": {
                "id": "agent",
                "kind": "agent",
                "base_url": "http://agent.local/v1",
                "model": "qwen3.5-9b-mtp",
                "api_key": "",
                "timeout": 600,
                "enabled": True,
            },
            "vlm": {
                "id": "vlm",
                "kind": "vlm",
                "base_url": "http://vlm.local/v1",
                "model": "qwen3-vl-4b",
                "api_key": "",
                "timeout": 300,
                "enabled": True,
            },
        }
        captured: Dict[str, Any] = {}

        def fake_call(messages, model_override=None, profile_id=None):
            captured["messages"] = messages
            captured["model_override"] = model_override
            captured["profile_id"] = profile_id
            return "agent default video summary"

        with (
            patch.object(config, "LM_PROFILES", profiles),
            patch.object(config, "LM_AGENT_PROFILE_ID", "agent"),
            patch.object(config, "LM_VLM_PROFILE_ID", "vlm"),
            patch.object(config, "LM_VLM_BALANCER_ENABLED", False),
            patch(
                "oldapp._sample_video_frames",
                return_value=(
                    [{"index": 0, "time_sec": 0.0, "thumbnail": thumb_b64}],
                    25.0,
                    1.0,
                ),
            ),
            patch("oldapp._call_video_understanding", side_effect=fake_call),
        ):
            resp = self.client.post(
                "/video_understanding",
                data={
                    "video": (BytesIO(b"not-a-real-video"), "sample.mp4"),
                    "prompt": "Describe this upload",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200, resp.get_json())
        payload = resp.get_json()
        self.assertEqual(payload["summary"], "agent default video summary")
        self.assertEqual(payload["model_selection"], "default_agent")
        self.assertEqual(payload["assigned_profile_id"], "agent")
        self.assertEqual(payload["model"], "qwen3.5-9b-mtp")
        self.assertEqual(payload["profile_id"], "agent")
        self.assertEqual(captured["model_override"], "agent")
        self.assertIsNone(captured["profile_id"])

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
        self.assertIn("lm_profiles", payload["checks"])
        self.assertIn("embedder", payload["checks"])
        self.assertIn("luxriot", payload["checks"])

    def test_mutating_frontend_endpoints_require_token(self) -> None:
        config.ADMIN_TOKEN = ""
        checks: List[Tuple[str, Dict[str, Any]]] = [
            ("/settings", {"json": {}}),
            ("/index", {"json": {}}),
            ("/comments", {"json": {}}),
            ("/probes/save", {"json": {}}),
            ("/probes/cast", {"json": {}}),
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

    def test_probe_cast_creates_copies_with_token(self) -> None:
        config.ADMIN_TOKEN = "unit-token"
        headers = {"X-Admin-Token": "unit-token"}
        saved_probes = []

        def _upsert(probe):
            stored = dict(probe)
            stored["id"] = stored.get("id") or f"probe-{len(saved_probes) + 1}"
            saved_probes.append(stored)
            return stored

        with (
            patch("oldapp.probes_store.list_probes", return_value=[]),
            patch("oldapp.probes_store.upsert_probe", side_effect=_upsert),
        ):
            response = self.client.post(
                "/probes/cast",
                headers=headers,
                json={
                    "name": "Door watch",
                    "channel_ids": [101, 102],
                    "positives": ["person near door"],
                    "bookmark": False,
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["counts"]["created"], 2)
        self.assertEqual(payload["counts"]["failed"], 0)
        self.assertEqual([probe["channel_id"] for probe in saved_probes], [101, 102])
        self.assertTrue(all(probe.get("cast_group_id") for probe in saved_probes))


if __name__ == "__main__":
    unittest.main()
