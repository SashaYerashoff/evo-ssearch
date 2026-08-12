import base64
import hashlib
import inspect
import re
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import patch

import oldapp
import numpy as np
from PIL import Image

from oldapp import (
    _MUTATION_ENDPOINT_PERMISSIONS,
    _SENSITIVE_ENDPOINT_PERMISSIONS,
    _build_detection_search_result,
    _bound_rollup_messages,
    _attention_batch_from_event,
    _env_precedence_report,
    _is_outdated_luxriot_json_prompt,
    _expired_stored_probe_lineage_payload,
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
    def test_cover_first_shipped_batch_state_contract_is_upgraded(self) -> None:
        old_contract = (
            "Machine-readable current-batch state for EVA memory, navigation, and alert actions:\n"
            "The alerts array contains current matches.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"cover":{},"scene":{},"alerts":[]}'
        )
        custom_contract = 'Return BATCH_STATE_JSON: {"cover":{},"alerts":[]}'
        canonical_contract = (
            "Machine-readable current-batch state for EVA memory, navigation, and alert actions:\n"
            "In BATCH_STATE_JSON the first two members MUST be version and alerts.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[],"cover":{}}'
        )

        self.assertTrue(_is_outdated_luxriot_json_prompt(old_contract))
        self.assertFalse(_is_outdated_luxriot_json_prompt(custom_contract))
        self.assertFalse(_is_outdated_luxriot_json_prompt(canonical_contract))

    def setUp(self) -> None:
        self.client = app.test_client()
        self._orig_auth_enabled = config.AUTH_ENABLED
        self._orig_admin_token = config.ADMIN_TOKEN
        self._orig_settings_local_only = config.SETTINGS_LOCAL_ONLY
        self._orig_host = config.HOST
        self._orig_port = config.PORT
        self._orig_offline_video_enabled = config.OFFLINE_VIDEO_ENABLED
        self._orig_probe_snap_enabled = config.PROBE_SNAP_ENABLED
        self._orig_indexed_folder_enabled = config.INDEXED_FOLDER_ENABLED
        config.AUTH_ENABLED = False
        config.OFFLINE_VIDEO_ENABLED = True
        config.PROBE_SNAP_ENABLED = True
        config.INDEXED_FOLDER_ENABLED = True

    def test_rollup_context_bounds_string_and_list_text_content(self):
        messages = [
            {"role": "system", "content": "S" * 6000},
            {
                "role": "user",
                "content": [{"type": "text", "text": "U" * 6000}],
            },
            {"role": "user", "content": "final instruction"},
        ]

        bounded = _bound_rollup_messages(messages, 5000)
        bounded_chars = sum(
            len(message["content"])
            if isinstance(message.get("content"), str)
            else sum(
                len(str(part.get("text") or ""))
                for part in message.get("content") or []
                if isinstance(part, dict) and part.get("type") == "text"
            )
            for message in bounded
        )

        self.assertLessEqual(bounded_chars, 5000)
        self.assertEqual(messages[0]["content"], "S" * 6000)
        self.assertEqual(bounded[-1]["content"], "final instruction")
        self.assertIn("older rollup source text compacted", bounded[0]["content"])

    def test_fast_attention_outcome_keeps_episode_and_decision_atomic(self):
        episode_id = "cccd01a6-0a88-4d79-b0bd-fc0286812129"
        batch = _attention_batch_from_event(
            "scheduler_decision",
            {
                "id": episode_id,
                "channel_id": 112,
                "episode_id": episode_id,
                "decided_at_ms": 2000,
                "action": "fast_vlm_no_alert",
                "record": {"frame_count": 6},
                "episode": {
                    "id": episode_id,
                    "channel_id": 112,
                    "started_at_ms": 1000,
                    "ended_at_ms": 1900,
                    "trigger": "fast_vlm_alert",
                    "status": "closed",
                    "record": {"frame_count": 6},
                },
            },
        )

        self.assertEqual(len(batch.episodes), 1)
        self.assertEqual(len(batch.decisions), 1)
        self.assertEqual(batch.decisions[0].episode_id, batch.episodes[0].id)

    def tearDown(self) -> None:
        config.AUTH_ENABLED = self._orig_auth_enabled
        config.ADMIN_TOKEN = self._orig_admin_token
        config.SETTINGS_LOCAL_ONLY = self._orig_settings_local_only
        config.HOST = self._orig_host
        config.PORT = self._orig_port
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

            store.upsert_probe({"id": "probe-2"})
            store.upsert_probe({"id": "probe-3"})
            self.assertEqual(
                store.delete_probes(["probe-2", "probe-3", "probe-3"]),
                2,
            )
            self.assertEqual(store.list_probes(), [])

    def test_archive_disk_guard_reports_low_space_before_snapshot_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(config, "DETECTIONS_ARCHIVE_DIR", temp_dir),
                patch.object(config, "DETECTIONS_ARCHIVE_ENABLED", True),
                patch.object(config, "ARCHIVE_DISK_MIN_FREE_GB", 0.0),
                patch.object(config, "ARCHIVE_DISK_MIN_FREE_PERCENT", 5.0),
            ):
                archive = oldapp._AdaptiveDetectionArchive()
            with patch(
                "oldapp.shutil.disk_usage",
                return_value=type(
                    "Usage",
                    (),
                    {"total": 1000, "used": 960, "free": 40},
                )(),
            ):
                status = archive.disk_status(refresh=True)

        self.assertFalse(status["ok"])
        self.assertEqual(status["status"], "low_space")
        self.assertEqual(status["free_percent"], 4.0)

    def test_probe_list_hides_expired_temporary_rows_but_keeps_disabled_saved_probes(self) -> None:
        store = type(
            "ProbeStore",
            (),
            {
                "list_probes": staticmethod(
                    lambda: [
                        {
                            "id": "saved-disabled",
                            "channel_id": 7,
                            "enabled": False,
                            "temporary": False,
                        },
                        {
                            "id": "temporary-active",
                            "channel_id": 7,
                            "enabled": True,
                            "temporary": True,
                            "expires_at_ms": 9_999_999_999_999,
                        },
                        {
                            "id": "temporary-expired",
                            "channel_id": 7,
                            "enabled": False,
                            "temporary": True,
                            "expires_at_ms": 1,
                        },
                    ]
                )
            },
        )()

        with patch("oldapp.probes_store", store):
            response = self.client.get("/probes/list")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            [probe["id"] for probe in payload["probes"]],
            ["saved-disabled", "temporary-active"],
        )
        self.assertEqual(payload["counts"]["persistent"], 1)
        self.assertEqual(payload["counts"]["temporary_active"], 1)
        self.assertEqual(payload["counts"]["temporary_expired_hidden"], 1)
        self.assertEqual(
            payload["defaults"]["pos_floor"],
            config.PROBE_POS_FLOOR_DEFAULT,
        )
        self.assertEqual(
            payload["defaults"]["margin"],
            config.PROBE_MARGIN_DEFAULT,
        )
        self.assertEqual(
            payload["defaults"]["embedding_backend"],
            "siglip2" if "siglip2" in config.CLIP_MODEL.lower() else "openai_clip",
        )

    def test_probe_list_keeps_one_card_image_but_compacts_recent_hit_history(self) -> None:
        store = type(
            "ProbeStore",
            (),
            {
                "list_probes": staticmethod(
                    lambda: [
                        {
                            "id": "probe-with-history",
                            "channel_id": 7,
                            "last_hit": {
                                "timestamp_ms": 3_000,
                                "pos_score": 0.8,
                                "thumbnail": "latest-card-image",
                            },
                            "recent_hits": [
                                {
                                    "timestamp_ms": 3_000,
                                    "pos_score": 0.8,
                                    "thumbnail": "duplicate-latest-image",
                                    "clip_vec": [0.1, 0.2],
                                },
                                {
                                    "timestamp_ms": 2_000,
                                    "pos_score": 0.7,
                                    "thumbnail": "older-image",
                                    "embedding": [0.3, 0.4],
                                },
                            ],
                        }
                    ]
                )
            },
        )()

        with patch("oldapp.probes_store", store):
            response = self.client.get("/probes/list")

        self.assertEqual(response.status_code, 200)
        probe = response.get_json()["probes"][0]
        self.assertEqual(probe["last_hit"]["thumbnail"], "latest-card-image")
        self.assertEqual(
            [hit["pos_score"] for hit in probe["recent_hits"]],
            [0.8, 0.7],
        )
        for hit in probe["recent_hits"]:
            self.assertNotIn("thumbnail", hit)
            self.assertNotIn("clip_vec", hit)
            self.assertNotIn("embedding", hit)

    def test_expired_probe_lineage_omits_heavy_runtime_thumbnails(self) -> None:
        payload = _expired_stored_probe_lineage_payload(
            {
                "id": "temporary-expired",
                "channel_id": 7,
                "temporary": True,
                "created_at_ms": 1_000,
                "expires_at_ms": 2_000,
                "parent_alert_id": "alert-1",
                "last_hit": {
                    "timestamp_ms": 1_900,
                    "pos_score": 0.8,
                    "thumbnail": "x" * 300_000,
                },
                "recent_hits": [
                    {
                        "timestamp_ms": 1_900,
                        "pos_score": 0.8,
                        "thumbnail": "y" * 300_000,
                    }
                ],
            },
            now_ms=3_000,
        )

        record = payload["record"]
        self.assertNotIn("last_hit", record)
        self.assertNotIn("recent_hits", record)
        self.assertEqual(record["runtime_evidence"]["recent_hit_count"], 1)
        self.assertEqual(
            record["runtime_evidence"]["last_hit"]["pos_score"],
            0.8,
        )
        self.assertNotIn(
            "thumbnail",
            record["runtime_evidence"]["last_hit"],
        )

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

    def test_runtime_config_exposes_pre_dotenv_provenance(self) -> None:
        self.assertTrue(hasattr(config, "ENV_KEYS_BEFORE_DOTENV"))
        self.assertTrue(hasattr(config, "ENV_VALUE_HASHES_BEFORE_DOTENV"))
        self.assertTrue(hasattr(config, "CONFIG_ENV_FILE_BEFORE_DOTENV"))

    def test_secure_settings_source_fails_closed_when_undeclared(self) -> None:
        with patch.object(config, "SECURE_DEPLOYMENT_REQUIRED", True), patch.object(
            config,
            "CONFIG_ENV_FILE_BEFORE_DOTENV",
            "",
        ):
            report = _env_precedence_report(file_map={}, file_path=Path(".env"))

        self.assertFalse(report["write_allowed"])
        self.assertEqual(report["config_source_status"], "undeclared")
        self.assertIsNone(report["persistence_source"])

    def test_secure_settings_write_refuses_undeclared_source(self) -> None:
        config.ADMIN_TOKEN = "unit-token"
        headers = {"X-Admin-Token": "unit-token"}
        with patch.object(config, "SECURE_DEPLOYMENT_REQUIRED", True), patch.object(
            config,
            "CONFIG_ENV_FILE_BEFORE_DOTENV",
            "",
        ), patch("oldapp._write_env_file_atomic") as write_env:
            response = self.client.post(
                "/settings",
                headers=headers,
                json={"host": "0.0.0.0"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["code"], "settings_source_undeclared")
        write_env.assert_not_called()

    def test_environment_editor_reads_persisted_file_not_runtime_overlay(self) -> None:
        config.ADMIN_TOKEN = "unit-token"
        headers = {"X-Admin-Token": "unit-token"}
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "EVOSSEARCH_PORT=5999\nEVOSSEARCH_LUXRIOT_PASSWORD=file-secret\n",
                encoding="utf-8",
            )
            with patch("oldapp._settings_env_path", return_value=env_path):
                response = self.client.get("/settings/env", headers=headers)

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["envVariables"]["EVOSSEARCH_PORT"], "5999")
        self.assertEqual(
            payload["envVariables"]["EVOSSEARCH_LUXRIOT_PASSWORD"],
            oldapp.ENV_SECRET_REDACTION,
        )
        self.assertNotIn("file-secret", payload["envText"])

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
            "/detections/<int:detection_id>/feedback",
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
            # The production React client builds these paths through its typed
            # incidents API rather than direct fetch() calls in the legacy bundle.
            "/incidents/draft",
            "/incidents",
            "/incidents/<incident_id>",
            "/incidents/<incident_id>/observations",
            "/incidents/<incident_id>/temporal",
            "/incidents/<incident_id>/follow",
            "/incidents/<incident_id>/stop-follow",
            "/incidents/<incident_id>/review",
            "/incidents/<incident_id>/series/<relation_id>/review",
            "/incidents/<incident_id>/export",
            "/js/app.js",
            "/lm/admission",
            "/lm/models",
            "/luxriot/recent_frame/<int:channel_id>",
            "/luxriot/rollups/l3-schedule",
            "/ready",
            "/reports/false-positives",
            "/reports/false-positives/export",
            "/ui-assets/<path:filename>",
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
                    "clip_embedding": [1.0, 0.0],
                    "embedding_space": {
                        "backend": "siglip2",
                        "model": "google/siglip2-base-patch16-224",
                        "dimension": 2,
                    },
                    "embedding_ref": "probe-buffer:7:1",
                    "embedding_status": "ready",
                    "width": 1280,
                    "height": 720,
                },
                {
                    "anchor_role": "last",
                    "frame_index": 11,
                    "timestamp_ms": 105000,
                    "thumbnail": "frame-two",
                    "clip_embedding": [0.0, 1.0],
                    "embedding_space": {
                        "backend": "siglip2",
                        "model": "google/siglip2-base-patch16-224",
                        "dimension": 2,
                    },
                    "embedding_ref": "probe-buffer:7:2",
                    "embedding_status": "ready",
                    "width": 1280,
                    "height": 720,
                },
            ],
        }

        with (
            patch("oldapp.detections_store", store),
            patch(
                "oldapp._embed_thumbnail_b64_with_space",
                side_effect=AssertionError("live archive frame was embedded twice"),
            ),
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
        self.assertEqual(store.records[0]["clip_vec"], [1.0, 0.0])
        self.assertEqual(
            store.records[0]["payload"]["embedding_ref"],
            "probe-buffer:7:1",
        )

    def test_cv_apex_and_companion_receive_independent_clip_embeddings(self) -> None:
        class Store:
            def __init__(self) -> None:
                self.records: List[Dict[str, Any]] = []

            def add_detections(self, records: List[Dict[str, Any]]) -> int:
                self.records.extend(records)
                return len(records)

        store = Store()
        embedded: List[str] = []

        def embed(thumbnail: Any, embedder: str) -> tuple[str, Dict[str, Any]]:
            self.assertEqual(embedder, "clip")
            value = str(thumbnail)
            embedded.append(value)
            return f"vec:{value}", {
                "backend": "siglip2",
                "model": "google/siglip2-base-patch16-224",
                "revision": "test-revision",
                "dimension": 768,
                "contract": "siglip2-torchvision-lower64-v1",
                "fingerprint": "test-fingerprint",
            }

        entry = {
            "channel_id": 7,
            "run_id": "run-7",
            "summary": "Fast movement with a sharper comparison frame.",
            "frame_count": 1,
            "created_at": 100.0,
            "batch_start_ms": 100000,
            "batch_end_ms": 100999,
            "archive_frames": [
                {
                    "anchor_role": "burst_apex",
                    "frame_index": 0,
                    "timestamp_ms": 100300,
                    "thumbnail": "cv-apex",
                    "source_frame_index": 2,
                    "selection_source": "capture_cv_frame_delta",
                    "selector_enabled": True,
                },
                {
                    "anchor_role": "burst_companion",
                    "frame_index": 0,
                    "timestamp_ms": 100600,
                    "thumbnail": "sharp-companion",
                    "source_frame_index": 3,
                    "companion_of_timestamp_ms": 100300,
                },
            ],
        }

        with (
            patch("oldapp.detections_store", store),
            patch("oldapp._embed_thumbnail_b64_with_space", side_effect=embed),
            patch("oldapp._apply_archive_retention", return_value={"ok": True}),
        ):
            result = _store_vlm_summary_archive_frames(entry)

        self.assertEqual(result["summary_frames"], 2)
        self.assertEqual(embedded, ["cv-apex", "sharp-companion"])
        self.assertEqual(
            [record["clip_vec"] for record in store.records],
            ["vec:cv-apex", "vec:sharp-companion"],
        )
        self.assertTrue(store.records[0]["payload"]["selector_enabled"])
        self.assertEqual(store.records[0]["payload"]["source_frame_index"], 2)
        self.assertEqual(store.records[1]["payload"]["source_frame_index"], 3)

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
        parent_ids = [
            record["payload"]["parent_alert_id"]
            for record in alert_records
        ]
        self.assertEqual(
            parent_ids,
            [
                record["payload"]["alert_event"]["id"]
                for record in alert_records
            ],
        )
        self.assertEqual(len(set(parent_ids)), 2)

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

    def test_detections_list_passes_exact_parent_alert_filter(self) -> None:
        captured: Dict[str, Any] = {}

        class Store:
            def list_detections(self, **kwargs):
                captured.update(kwargs)
                return [], 0

        with patch("oldapp.detections_store", Store()):
            response = self.client.get(
                "/detections/list?source=vlm_alert&channel_id=7"
                "&parent_alert_id=vlm-alert-exact"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(captured["parent_alert_id"], "vlm-alert-exact")
        self.assertEqual(
            response.get_json()["filters"]["parent_alert_id"],
            "vlm-alert-exact",
        )

    def test_local_probe_bookmark_is_suppressed_before_send(self) -> None:
        with (
            patch.object(
                oldapp.luxriot_manager,
                "is_local_channel",
                return_value=True,
            ),
            patch.object(oldapp.luxriot_manager, "send_bookmark_event") as send,
            patch("oldapp._embed_thumbnail_b64") as embed,
        ):
            sent, gate = oldapp._maybe_send_probe_bookmark(
                {
                    "id": "probe-local",
                    "name": "Local camera watch",
                    "channel_id": 900001,
                    "bookmark": True,
                },
                {
                    "timestamp_ms": 1_785_000_000_000,
                    "pos_score": 0.8,
                    "neg_score": 0.1,
                    "margin": 0.7,
                    "thumbnail": "jpeg",
                },
                source="probe_daemon",
            )

        self.assertFalse(sent)
        self.assertEqual(gate["reason"], "local_source_no_recorder")
        send.assert_not_called()
        embed.assert_not_called()

    def test_probe_bookmark_uses_current_embedding_and_records_delivery_timing(self) -> None:
        timestamp_ms = int(time.time() * 1000.0) - 25
        with (
            patch.object(oldapp.luxriot_manager, "is_local_channel", return_value=False),
            patch.object(
                oldapp.luxriot_manager,
                "send_bookmark_event",
                return_value={"success": True},
            ) as send,
            patch("oldapp._embed_thumbnail_b64") as embed,
        ):
            sent, gate = oldapp._maybe_send_probe_bookmark(
                {
                    "id": f"probe-live-{timestamp_ms}",
                    "name": "Immediate collision watch",
                    "channel_id": 112,
                    "bookmark": True,
                    "bookmark_cooldown_sec": 0,
                    "bookmark_dedupe_window_sec": 1,
                },
                {
                    "timestamp_ms": timestamp_ms,
                    "pos_score": 0.9,
                    "neg_score": 0.1,
                    "margin": 0.8,
                    "thumbnail": "jpeg",
                    "clip_vec": np.asarray([1.0, 0.0], dtype=np.float32),
                },
                source="probe_realtime",
            )

        self.assertTrue(sent)
        self.assertEqual(gate["reason"], "sent")
        self.assertGreaterEqual(gate["bookmark_ack_at_ms"], gate["bookmark_attempted_at_ms"])
        self.assertEqual(
            gate["bookmark_delivery_ms"],
            gate["bookmark_ack_at_ms"] - gate["bookmark_attempted_at_ms"],
        )
        self.assertGreaterEqual(gate["event_to_bookmark_ack_ms"], 0)
        send.assert_called_once()
        embed.assert_not_called()

    def test_fast_vlm_alert_waits_for_post_roll_after_burst(self) -> None:
        runtime = oldapp._FastVlmAlertRuntime()
        try:
            trigger_ms = 100_000
            with patch.object(runtime, "_submit") as submit:
                runtime.observe(
                    112,
                    {
                        "timestamp_ms": trigger_ms,
                        "capture_selection": {"selection_mode": "burst"},
                    },
                )
                submit.assert_not_called()
                runtime.observe(
                    112,
                    {
                        "timestamp_ms": trigger_ms + runtime.post_roll_ms,
                        "capture_selection": {"selection_mode": "normal"},
                    },
                )
                submit.assert_called_once()
                channel_id, episode = submit.call_args.args
                self.assertEqual(channel_id, 112)
                self.assertEqual(episode["trigger_timestamp_ms"], trigger_ms)
                self.assertEqual(
                    episode["observed_post_timestamp_ms"],
                    trigger_ms + runtime.post_roll_ms,
                )
        finally:
            runtime.shutdown()

    def test_realtime_probe_lane_coalesces_stale_channel_work(self) -> None:
        runtime = oldapp._RealtimeProbeBookmarkRuntime()
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        seen: List[int] = []

        def evaluate(_channel_id: int, observation: Dict[str, Any]) -> None:
            seen.append(int(observation["timestamp_ms"]))
            if len(seen) == 1:
                started.set()
                self.assertTrue(release.wait(2.0))
            if len(seen) == 2:
                completed.set()

        try:
            with patch.object(runtime, "_evaluate", side_effect=evaluate):
                runtime.submit(112, {"timestamp_ms": 1})
                self.assertTrue(started.wait(2.0))
                runtime.submit(112, {"timestamp_ms": 2})
                runtime.submit(112, {"timestamp_ms": 3})
                release.set()
                self.assertTrue(completed.wait(2.0))
            self.assertEqual(seen, [1, 3])
            self.assertEqual(runtime.status()["coalesced_total"], 2)
        finally:
            runtime.shutdown()

    def test_fast_vlm_alert_routes_large_semantic_change_with_cv_motion(self) -> None:
        runtime = oldapp._FastVlmAlertRuntime()
        try:
            runtime.semantic_delta_threshold = 0.2
            runtime.min_moving_fraction = 0.15
            with patch.object(runtime, "_submit") as submit:
                runtime.observe(
                    118,
                    {
                        "timestamp_ms": 100_000,
                        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                        "capture_selection": {"selection_mode": "normal"},
                        "motion_aggregate": {"moving_fraction": 0.5},
                    },
                )
                runtime.observe(
                    118,
                    {
                        "timestamp_ms": 101_000,
                        "embedding": np.asarray([0.7, 0.714], dtype=np.float32),
                        "capture_selection": {"selection_mode": "normal"},
                        "motion_aggregate": {"moving_fraction": 0.5},
                    },
                )
                submit.assert_not_called()
                runtime.observe(
                    118,
                    {
                        "timestamp_ms": 101_000 + runtime.post_roll_ms,
                        "embedding": np.asarray([0.7, 0.714], dtype=np.float32),
                        "capture_selection": {"selection_mode": "normal"},
                        "motion_aggregate": {"moving_fraction": 0.5},
                    },
                )
                submit.assert_called_once()
                _channel, episode = submit.call_args.args
                self.assertEqual(episode["reason"], "semantic_motion_change")
                self.assertGreaterEqual(episode["semantic_delta"], 0.2)
        finally:
            runtime.shutdown()

    def test_detections_list_passes_multiple_channel_filters(self) -> None:
        captured: Dict[str, Any] = {}

        class Store:
            def list_detections(self, **kwargs):
                captured.update(kwargs)
                return [], 0

        with patch("oldapp.detections_store", Store()):
            response = self.client.get(
                "/detections/list?channel_id=7&channel_id=9&since_ms=1000"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(captured["channel_ids"], [7, 9])
        self.assertNotIn("channel_id", captured)
        self.assertEqual(response.get_json()["filters"]["channel_ids"], [7, 9])

    def test_detections_list_can_skip_thumbnail_payload_for_history_reader(self) -> None:
        captured: Dict[str, Any] = {}

        class Store:
            def list_detections(self, **kwargs):
                captured.update(kwargs)
                return [], 0

        with patch("oldapp.detections_store", Store()):
            response = self.client.get(
                "/detections/list?source=vlm_summary&channel_id=7&since_ms=1000"
                "&until_ms=2000&include_thumbnail=0&batch_id=vlm-test-batch"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertFalse(captured["include_thumbnail"])
        self.assertEqual(captured["batch_id"], "vlm-test-batch")
        self.assertFalse(response.get_json()["filters"]["include_thumbnail"])
        self.assertEqual(response.get_json()["filters"]["batch_id"], "vlm-test-batch")

    def test_luxriot_history_returns_compact_postgres_batch_page(self) -> None:
        captured: Dict[str, Any] = {}

        class Store:
            def list_vlm_summary_batches(self, **kwargs):
                captured.update(kwargs)
                return [
                    {
                        "channel_id": 7,
                        "run_id": "run-a",
                        "created_at": 2.0,
                        "batch_start_ms": 1000,
                        "batch_end_ms": 2000,
                        "summary": "A person crossed the road.",
                    }
                ], 3

        with patch("oldapp.detections_store", Store()):
            response = self.client.get(
                "/luxriot/history?channel_id=7&from_ts=1&to_ts=2&limit=1&offset=0"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["storage"], "postgres")
        self.assertEqual(payload["run"], "all")
        self.assertEqual(payload["total"], 3)
        self.assertTrue(payload["has_more"])
        self.assertEqual(payload["logs"][0]["summary"], "A person crossed the road.")
        self.assertEqual(captured["since_ms"], 1000)
        self.assertEqual(captured["until_ms"], 2000)
        self.assertTrue(captured["return_page_info"])
        self.assertTrue(payload["total_exact"])

    def test_luxriot_history_exposes_lower_bound_without_claiming_exact_total(self) -> None:
        class Store:
            def list_vlm_summary_batches(self, **_kwargs):
                return ([{"channel_id": 7, "summary": "Newest batch"}], 2, {
                    "has_more": True,
                    "total_exact": False,
                    "scanned_rows": 256,
                })

        with patch("oldapp.detections_store", Store()):
            response = self.client.get(
                "/luxriot/history?channel_id=7&limit=1&offset=0"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["total"], 2)
        self.assertTrue(payload["has_more"])
        self.assertFalse(payload["total_exact"])

    def test_luxriot_rollup_api_never_exposes_internal_memory_or_signal_digest(self) -> None:
        rollup_payload = {
            "channel_id": 7,
            "routine_context": {"routine": "internal homeostasis"},
            "levels": {
                "L1": [
                    {
                        "rollup_id": "l1-7",
                        "summary": "### Period Overview\nRoutine window.",
                        "operator_summary": "### Period Overview\nRoutine window.",
                        "memory_update": {"routine_baseline": "internal homeostasis"},
                        "signal_digest": {"watchlist": ["internal item"]},
                        "highlights": ["internal source excerpt"],
                        "llm_input_stats": {"source_chars_selected": 100},
                    }
                ]
            },
        }
        with patch("oldapp.luxriot_manager.summary_rollups", return_value=rollup_payload):
            response = self.client.get("/luxriot/rollups?channel_id=7&target_level=L1")

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertNotIn("routine_context", payload)
        row = payload["levels"]["L1"][0]
        self.assertEqual(row["summary"], "### Period Overview\nRoutine window.")
        self.assertNotIn("operator_summary", row)
        self.assertNotIn("memory_update", row)
        self.assertNotIn("signal_digest", row)
        self.assertNotIn("highlights", row)
        self.assertNotIn("llm_input_stats", row)

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

    def test_detections_list_rejects_unknown_source_filter(self) -> None:
        class Store:
            def list_detections(self, **_kwargs):
                raise AssertionError("invalid source must not reach the archive store")

        with patch("oldapp.detections_store", Store()):
            response = self.client.get("/detections/list?source=everything")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "source must be one of: semantic_snapshot, probe, vlm_summary, vlm_alert"},
        )

    def test_detections_summary_rejects_unknown_source_filter(self) -> None:
        class Store:
            def summarize_by_probe(self, **_kwargs):
                raise AssertionError("invalid source must not reach the archive store")

        with patch("oldapp.detections_store", Store()):
            response = self.client.get("/detections/summary?source=everything")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "source must be one of: semantic_snapshot, probe, vlm_summary, vlm_alert"},
        )

    def test_detection_routes_keep_empty_source_as_no_filter(self) -> None:
        captured_sources = []

        class Store:
            def list_detections(self, **kwargs):
                captured_sources.append(kwargs.get("source"))
                return [], 0

            def summarize_by_probe(self, **kwargs):
                captured_sources.append(kwargs.get("source"))
                return []

        with patch("oldapp.detections_store", Store()):
            list_response = self.client.get("/detections/list?source=%20")
            summary_response = self.client.get("/detections/summary?source=")

        self.assertEqual(list_response.status_code, 200, list_response.get_json())
        self.assertEqual(summary_response.status_code, 200, summary_response.get_json())
        self.assertEqual(captured_sources, [None, None])

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

    def test_detection_archive_rejects_reversed_time_range_before_search(self) -> None:
        text_response = self.client.post(
            "/detections/search_text",
            json={"query": "person", "since_ms": 2_000, "until_ms": 1_000},
        )
        list_response = self.client.get(
            "/detections/list?since_ms=2000&until_ms=1000"
        )

        self.assertEqual(text_response.status_code, 400, text_response.get_json())
        self.assertEqual(list_response.status_code, 400, list_response.get_json())
        self.assertIn("since_ms", text_response.get_json()["error"])
        self.assertIn("since_ms", list_response.get_json()["error"])

    def test_detection_text_search_accepts_continuous_clip_archive(self) -> None:
        captured: Dict[str, Any] = {}

        def _search(**kwargs):
            captured.update(kwargs)
            return [], {"search_strategy": "hourly_sharded_exact"}

        with (
            patch("oldapp.get_clip_text_embedding", return_value=object()),
            patch("oldapp._search_detections_archive", side_effect=_search),
        ):
            response = self.client.post(
                "/detections/search_text",
                json={
                    "query": "person near entrance",
                    "source": "semantic_snapshot",
                    "channel_ids": [7, 8],
                    "hours": 24,
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(captured["source"], "semantic_snapshot")
        self.assertEqual(captured["channel_ids"], [7, 8])
        self.assertEqual(
            response.get_json()["coverage"]["search_strategy"],
            "hourly_sharded_exact",
        )

    def test_continuous_clip_search_ranks_every_matching_shard(self) -> None:
        embedding_space = {
            "backend": "siglip2",
            "model": "google/siglip2-base-patch16-224",
            "revision": "test-revision",
            "dimension": 2,
        }
        fingerprint = oldapp.embedding_space_fingerprint(embedding_space)
        ch7_shard = f"semantic:e{fingerprint}:ch7:1970010100"
        ch8_shard = f"semantic:e{fingerprint}:ch8:1970010100"

        class Index:
            def __init__(self, name):
                self.name = name
                self.d = 2

        rows = {
            1: {
                "id": 1,
                "timestamp_ms": 1_000,
                "probe_id": "semantic-snapshot:ch7",
                "probe_name": "Semantic snapshot",
                "channel_id": 7,
                "source": "semantic_snapshot",
                "shard_key": ch7_shard,
                "thumbnail": "dGVzdA==",
                "payload": {"embedding_space": embedding_space},
            },
            2: {
                "id": 2,
                "timestamp_ms": 2_000,
                "probe_id": "semantic-snapshot:ch7",
                "probe_name": "Semantic snapshot",
                "channel_id": 7,
                "source": "semantic_snapshot",
                "shard_key": ch7_shard,
                "thumbnail": "dGVzdA==",
                "payload": {"embedding_space": embedding_space},
            },
            3: {
                "id": 3,
                "timestamp_ms": 3_000,
                "probe_id": "semantic-snapshot:ch8",
                "probe_name": "Semantic snapshot",
                "channel_id": 8,
                "source": "semantic_snapshot",
                "shard_key": ch8_shard,
                "thumbnail": "dGVzdA==",
                "payload": {"embedding_space": embedding_space},
            },
            4: {
                "id": 4,
                "timestamp_ms": 4_000,
                "probe_id": "semantic-snapshot:ch8",
                "probe_name": "Semantic snapshot",
                "channel_id": 8,
                "source": "semantic_snapshot",
                "shard_key": ch8_shard,
                "thumbnail": "dGVzdA==",
                "payload": {"embedding_space": embedding_space},
            },
        }

        class Store:
            def summarize_shards(self, **_kwargs):
                return [
                    {
                        "shard_key": ch7_shard,
                        "clip_count": 2,
                    },
                    {
                        "shard_key": ch8_shard,
                        "clip_count": 2,
                    },
                ]

            def fetch_detections_by_ids(
                self,
                ids,
                *,
                include_vectors=False,
                include_thumbnail=False,
            ):
                return [dict(rows[item]) for item in ids if item in rows]

        indexes = {
            ch7_shard: (
                Index("ch7"),
                oldapp.np.asarray([1, 2], dtype=oldapp.np.int64),
            ),
            ch8_shard: (
                Index("ch8"),
                oldapp.np.asarray([3, 4], dtype=oldapp.np.int64),
            ),
        }

        def search(index, _query, k):
            if index.name == "ch7":
                scores = [0.8, 0.7]
            else:
                scores = [0.95, 0.6]
            return (
                oldapp.np.asarray([scores[:k]], dtype=oldapp.np.float32),
                oldapp.np.asarray(
                    [list(range(min(k, len(scores))))],
                    dtype=oldapp.np.int64,
                ),
            )

        with (
            patch("oldapp.detections_store", Store()),
            patch.object(
                oldapp.detection_clip_shard_cache,
                "get",
                side_effect=lambda key: indexes[key],
            ),
            patch("oldapp._faiss_search", side_effect=search),
            patch(
                "oldapp.get_probe_embedding_space",
                return_value=embedding_space,
            ),
        ):
            results, coverage = oldapp._search_semantic_snapshot_shards(
                clip_query_vec=oldapp.np.asarray(
                    [1.0, 0.0],
                    dtype=oldapp.np.float32,
                ),
                dino_query_vec=None,
                mode="clip",
                probe_id=None,
                channel_id=None,
                channel_ids=[7, 8],
                since_ms=0,
                until_ms=5_000,
                limit=2,
                sort_by="similarity",
            )

        self.assertEqual(
            [item["detection_id"] for item in results],
            [3, 1],
        )
        self.assertEqual(coverage["scanned_candidates"], 4)
        self.assertEqual(coverage["shards_searched"], 2)
        self.assertFalse(coverage["truncated"])

    def test_detection_search_falls_back_to_healthy_channels(self) -> None:
        class QueryCanceled(Exception):
            pass

        def search(**kwargs):
            if kwargs.get("channel_ids") == [7, 8]:
                raise QueryCanceled("canceling statement due to statement timeout")
            if kwargs.get("channel_id") == 7:
                return ([{
                    "detection_id": 71,
                    "channel_id": 7,
                    "timestamp_ms": 1_000,
                    "similarity": 0.9,
                }], {"scanned_candidates": 1, "total_candidates": 1})
            raise QueryCanceled("canceling statement due to statement timeout")

        with patch("oldapp._search_detections_archive", side_effect=search):
            results, coverage = oldapp._search_detections_archive_resilient(
                clip_query_vec=oldapp.np.asarray([1.0, 0.0], dtype=oldapp.np.float32),
                dino_query_vec=None,
                mode="clip",
                probe_id=None,
                channel_id=None,
                channel_ids=[7, 8],
                source="vlm_summary",
                since_ms=0,
                until_ms=5_000,
                limit=12,
                sort_by="similarity",
                candidate_limit=20_000,
            )

        self.assertEqual([item["detection_id"] for item in results], [71])
        self.assertEqual(coverage["searched_channel_ids"], [7])
        self.assertEqual(coverage["failed_channel_ids"], [8])
        self.assertTrue(coverage["partial"])

    def test_detection_search_omits_rows_without_visual_evidence(self) -> None:
        rows = {
            1: {"id": 1, "channel_id": 7, "timestamp_ms": 1_000},
            2: {"id": 2, "channel_id": 7, "timestamp_ms": 2_000, "thumbnail": "dGVzdA=="},
        }

        class Store:
            def fetch_detections_by_ids(self, ids, **_kwargs):
                return [rows[item] for item in ids]

        stats = {}
        with patch("oldapp.detections_store", Store()):
            results = oldapp._finalize_detection_search_results(
                clip_hits=[(1, 0.95), (2, 0.90)],
                candidate_map=rows,
                dino_query_vec=None,
                mode="clip",
                sort_by="similarity",
                limit=2,
                stats=stats,
            )

        self.assertEqual([item["detection_id"] for item in results], [2])
        self.assertEqual(stats["visual_evidence_excluded"], 1)

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
            "react_ui_asset",
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
            ("GET", "/probes/list", {}, {200, 503}),
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

    def test_probe_status_returns_prethreshold_live_pnm(self) -> None:
        class Session:
            def status(self):
                return {
                    "running": True,
                    "probe_last_error": None,
                    "capture_last_error": None,
                }

        probe = {
            "id": "probe-live",
            "name": "Person in headphones",
            "channel_id": 7,
            "positives": ["person wearing headphones"],
            "negatives": ["person without headphones"],
            "pos_floor": 0.5,
            "margin": 0.1,
        }
        scores = {
            "frames_indexed": 2,
            "results": [
                {"timestamp_ms": 1000, "pos_score": 0.42, "neg_score": 0.39, "margin": 0.03},
                {"timestamp_ms": 2000, "pos_score": 0.47, "neg_score": 0.41, "margin": 0.06},
            ],
        }
        with (
            patch.object(oldapp.probes_store, "list_probes", return_value=[probe]),
            patch.object(oldapp.probe_manager, "status", return_value={"frames": 2}),
            patch.object(oldapp.probe_manager, "score_frames", return_value=scores),
            patch.object(oldapp.luxriot_manager, "sessions", {7: Session()}),
            patch("oldapp._probe_embedding_calibration_state", return_value="calibrated"),
            patch("oldapp.time.time", return_value=2.5),
        ):
            response = self.client.get(
                "/probes/status?channel_id=7&probe_id=probe-live"
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["runtime_state"], "running")
        self.assertEqual(payload["semantic_state"], "ready")
        self.assertEqual(payload["live_signal"]["pos_score"], 0.47)
        self.assertEqual(payload["live_signal"]["threshold_state"], "below_both")
        self.assertEqual(payload["live_signal"]["age_ms"], 500)
        self.assertFalse(payload["live_signal"]["stale"])
        self.assertEqual(
            payload["live_signal"]["frame_url"],
            "/probes/signal_frame/7/2000",
        )
        self.assertEqual(len(payload["signal_history"]), 2)

    def test_probe_signal_frame_returns_exact_buffered_jpeg_without_cache(self) -> None:
        encoded = base64.b64encode(b"exact-semantic-jpeg").decode("ascii")
        with patch.object(
            oldapp.probe_manager,
            "frame_thumbnail",
            return_value=encoded,
        ) as frame_thumbnail:
            response = self.client.get("/probes/signal_frame/7/2000")

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.data, b"exact-semantic-jpeg")
        self.assertEqual(response.content_type, "image/jpeg")
        self.assertIn("no-store", response.headers["Cache-Control"])
        self.assertEqual(response.headers["X-EVA-Frame-Timestamp-Ms"], "2000")
        frame_thumbnail.assert_called_once_with(7, 2000)

    def test_probe_signal_frame_does_not_substitute_a_newer_frame(self) -> None:
        with patch.object(
            oldapp.probe_manager,
            "frame_thumbnail",
            return_value=None,
        ):
            response = self.client.get("/probes/signal_frame/7/1500")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"], "semantic_frame_unavailable")

    def test_probe_status_exposes_embedder_failure_instead_of_false_running(self) -> None:
        class Session:
            def status(self):
                return {
                    "running": True,
                    "probe_last_error": "SigLIP2 model is not available offline",
                }

        with (
            patch.object(oldapp.probe_manager, "status", return_value={"frames": 0}),
            patch.object(oldapp.luxriot_manager, "sessions", {7: Session()}),
        ):
            response = self.client.get("/probes/status?channel_id=7")

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["runtime_state"], "running")
        self.assertEqual(payload["semantic_state"], "degraded")
        self.assertIn("SigLIP2", payload["semantic_error"])

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

    def test_host_only_settings_patch_keeps_visible_port_when_browser_sends_blank(self) -> None:
        config.ADMIN_TOKEN = "unit-token"
        headers = {"X-Admin-Token": "unit-token"}
        running_host = config.HOST
        running_port = config.PORT
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "EVOSSEARCH_HOST=127.0.0.1\n"
                "EVOSSEARCH_PORT=5081\n"
                "EXTERNAL_SENTINEL=preserve-me\n",
                encoding="utf-8",
            )
            with patch("oldapp._settings_env_path", return_value=env_path), patch(
                "oldapp._write_completion_audit_or_error",
                return_value=None,
            ):
                response = self.client.post(
                    "/settings",
                    headers=headers,
                    json={"host": "0.0.0.0", "port": ""},
                )

            self.assertEqual(response.status_code, 200, response.get_json())
            saved = oldapp._read_env_file_map(env_path)
            self.assertEqual(saved["EVOSSEARCH_HOST"], "0.0.0.0")
            self.assertEqual(saved["EVOSSEARCH_PORT"], "5081")
            self.assertEqual(saved["EXTERNAL_SENTINEL"], "preserve-me")
            self.assertEqual(config.HOST, running_host)
            self.assertEqual(config.PORT, running_port)
            payload = response.get_json()
            self.assertEqual(payload["restartRequiredFields"], ["host"])
            self.assertNotIn("host", payload["runtimeAppliedFields"])

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
