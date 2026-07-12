import unittest
import base64
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace

import oldapp
from PIL import Image
from unittest.mock import patch
from security import ALL_CHANNELS, Permission, Role, digest_session_token
from security.http_auth import AuthenticationService


TENANT_ID = "59da6ca3-51b7-4d91-9190-aae06b76d846"
USER_ID = "361fe45f-f277-42f8-ae35-eaa0fc81cf38"


@dataclass(frozen=True)
class _Identity:
    user_id: str = USER_ID
    tenant_id: str = TENANT_ID
    username: str = "engineer"
    display_name: str | None = "Pilot Engineer"
    roles: frozenset[str] = frozenset({Role.ENGINEER.value})
    permissions: frozenset[str] = frozenset(
        {
            Permission.AGENT_USE.value,
            Permission.DETECTIONS_VIEW.value,
            Permission.PROBES_MANAGE.value,
            Permission.STREAMS_VIEW.value,
        }
    )
    allowed_channel_ids: frozenset[int] = frozenset({7})
    is_active: bool = True


@dataclass(frozen=True)
class _Session:
    identity: _Identity
    csrf_digest: str
    expires_at: datetime


@dataclass(frozen=True)
class _ManagedSession:
    session_id: str
    tenant_id: str
    user_id: str
    username: str
    created_at: datetime
    last_seen_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    revoke_reason: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None


@dataclass(frozen=True)
class _AuditRow:
    event_id: str = "event-1"
    action: str = "auth.login"

    def to_dict(self):
        return {
            "id": self.event_id,
            "action": self.action,
            "details": {"safe": "visible"},
        }


@dataclass(frozen=True)
class _AuditPage:
    events: tuple[_AuditRow, ...] = (_AuditRow(),)
    next_cursor: str | None = "cursor-1"


class _AuditReader:
    def __init__(self) -> None:
        self.calls = []
        self.error = None

    def list_events(self, context, **kwargs):
        if self.error is not None:
            raise self.error
        self.calls.append((context, kwargs))
        return _AuditPage()


class _Repository:
    def __init__(self) -> None:
        self.identity = _Identity()
        self.sessions = {}
        self.session_inventory = []
        self.users = {USER_ID: self.identity}
        self.revoked_user_sessions = []
        self.revoked_session_ids = []

    def authenticate(self, tenant_id, username, password):
        if (
            tenant_id == TENANT_ID
            and username.lower() == "engineer"
            and password == "correct-password"
        ):
            return self.identity
        return None

    def create_session(
        self,
        identity,
        token,
        csrf_token,
        expires_at,
        *,
        client_ip=None,
        user_agent=None,
    ):
        session_id = f"session-{len(self.sessions) + 1}"
        self.sessions[token] = _Session(
            identity=identity,
            csrf_digest=digest_session_token(csrf_token),
            expires_at=expires_at,
        )
        self.session_inventory.append(
            _ManagedSession(
                session_id=session_id,
                tenant_id=identity.tenant_id,
                user_id=identity.user_id,
                username=identity.username,
                created_at=datetime(2026, 1, 1, 10, 0, 0),
                last_seen_at=datetime(2026, 1, 1, 10, 5, 0),
                expires_at=expires_at,
                client_ip=client_ip,
                user_agent=user_agent,
            )
        )
        return session_id

    def resolve_session(self, tenant_id, token):
        if tenant_id != TENANT_ID:
            return None
        return self.sessions.get(token)

    def revoke_session(self, tenant_id, token, reason):
        if tenant_id != TENANT_ID:
            return False
        return self.sessions.pop(token, None) is not None

    def list_users(self, tenant_id, *, actor_user_id, include_inactive=True):
        if tenant_id != TENANT_ID:
            return ()
        users = list(self.users.values())
        if not include_inactive:
            users = [user for user in users if user.is_active]
        return tuple(users)

    def get_user(self, tenant_id, user_id, *, actor_user_id):
        if tenant_id != TENANT_ID:
            return None
        return self.users.get(user_id)

    def create_user(
        self,
        tenant_id,
        *,
        actor_user_id,
        username,
        password,
        roles,
        display_name=None,
        allowed_channel_ids=(),
        is_active=True,
    ):
        if tenant_id != TENANT_ID:
            raise RuntimeError("wrong tenant")
        del actor_user_id, password
        user_id = f"user-{len(self.users) + 1}"
        user = _Identity(
            user_id=user_id,
            username=username,
            display_name=display_name,
            roles=frozenset(str(role) for role in roles),
            permissions=frozenset({Permission.STREAMS_VIEW.value}),
            allowed_channel_ids=frozenset(allowed_channel_ids),
            is_active=bool(is_active),
        )
        self.users[user_id] = user
        return user

    def update_user(self, tenant_id, user_id, *, actor_user_id, **updates):
        if tenant_id != TENANT_ID:
            raise RuntimeError("wrong tenant")
        del actor_user_id
        user = self.users.get(user_id)
        if user is None:
            raise LookupError("user not found")
        replacements = {}
        if "display_name" in updates:
            replacements["display_name"] = updates["display_name"]
        if "roles" in updates:
            replacements["roles"] = frozenset(str(role) for role in updates["roles"])
        if "allowed_channel_ids" in updates:
            replacements["allowed_channel_ids"] = frozenset(
                updates["allowed_channel_ids"]
            )
        if "is_active" in updates:
            replacements["is_active"] = bool(updates["is_active"])
        updated = replace(user, **replacements)
        self.users[user_id] = updated
        return updated

    def revoke_user_sessions(self, tenant_id, user_id, *, actor_user_id, reason):
        if tenant_id != TENANT_ID:
            raise RuntimeError("wrong tenant")
        if user_id not in self.users:
            raise LookupError("user not found")
        self.revoked_user_sessions.append((user_id, actor_user_id, reason))
        return 2

    def list_sessions(
        self,
        tenant_id,
        *,
        actor_user_id,
        user_id=None,
        active_only=True,
    ):
        if tenant_id != TENANT_ID:
            raise RuntimeError("wrong tenant")
        del actor_user_id
        sessions = [
            session
            for session in self.session_inventory
            if user_id is None or session.user_id == user_id
        ]
        if active_only:
            sessions = [
                session for session in sessions if session.revoked_at is None
            ]
        return tuple(sessions)

    def revoke_session_by_id(
        self,
        tenant_id,
        session_id,
        *,
        actor_user_id,
        reason,
    ):
        if tenant_id != TENANT_ID:
            raise RuntimeError("wrong tenant")
        del actor_user_id
        for index, session in enumerate(self.session_inventory):
            if session.session_id != session_id or session.revoked_at is not None:
                continue
            self.session_inventory[index] = replace(
                session,
                revoked_at=datetime(2026, 1, 1, 11, 0, 0),
                revoke_reason=reason,
            )
            self.revoked_session_ids.append((session_id, reason))
            return True
        return False


class _AuditWriter:
    def __init__(self) -> None:
        self.events = []
        self.error = None

    def write(self, event):
        if self.error is not None:
            raise self.error
        self.events.append(event)
        return f"event-{len(self.events)}"


class _AgentRunner:
    def __init__(self) -> None:
        self.calls = []
        self.approvals = []

    def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        yield 'data: {"type":"done","session_id":"session-1"}\n\n'

    def approve_action_plan(self, plan_id, tool_context):
        self.approvals.append((plan_id, tool_context))
        return {"status": "applied", "plan_id": plan_id}


class HttpAuthRouteTests(unittest.TestCase):
    def test_video_lm_generation_preflight_runs_after_admission_before_http(self):
        order = []

        class Admission:
            def __enter__(self):
                order.append("admitted")
                return "ticket"

            def __exit__(self, _exc_type, _exc, _tb):
                order.append("released")
                return False

        class Superseded(RuntimeError):
            superseded = True

        marker = Superseded("old generation")

        def preflight():
            order.append("preflight")
            raise marker

        profile = {
            "id": "vlm-test",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "qwen-test",
            "api_key": "",
            "timeout": 30,
            "kind": "vlm",
        }
        with (
            patch("oldapp._resolve_lm_profile", return_value=profile),
            patch.object(oldapp._lm_admission_controller, "admission", return_value=Admission()),
            patch("oldapp.requests.post") as post,
        ):
            with self.assertRaises(Superseded):
                oldapp._call_video_understanding([], preflight=preflight)

        self.assertEqual(order, ["admitted", "preflight", "released"])
        post.assert_not_called()

    def setUp(self) -> None:
        self.original = {
            "AUTH_ENABLED": oldapp.config.AUTH_ENABLED,
            "AUTH_TENANT_ID": oldapp.config.AUTH_TENANT_ID,
            "AUTH_COOKIE_SECURE": oldapp.config.AUTH_COOKIE_SECURE,
            "OFFLINE_VIDEO_ENABLED": oldapp.config.OFFLINE_VIDEO_ENABLED,
            "PROBE_SNAP_ENABLED": oldapp.config.PROBE_SNAP_ENABLED,
            "INDEXED_FOLDER_ENABLED": oldapp.config.INDEXED_FOLDER_ENABLED,
        }
        oldapp.config.AUTH_ENABLED = True
        oldapp.config.AUTH_TENANT_ID = TENANT_ID
        oldapp.config.AUTH_COOKIE_SECURE = False
        oldapp.config.OFFLINE_VIDEO_ENABLED = True
        oldapp.config.PROBE_SNAP_ENABLED = True
        oldapp.config.INDEXED_FOLDER_ENABLED = True
        self.repository = _Repository()
        self.audit = _AuditWriter()
        oldapp._auth_service = AuthenticationService(
            self.repository,
            tenant_id=TENANT_ID,
        )
        oldapp._identity_repository = self.repository
        oldapp._audit_writer = self.audit
        self.audit_reader = _AuditReader()
        oldapp._audit_reader = self.audit_reader
        self.client = oldapp.app.test_client()

    def tearDown(self) -> None:
        oldapp.config.AUTH_ENABLED = self.original["AUTH_ENABLED"]
        oldapp.config.AUTH_TENANT_ID = self.original["AUTH_TENANT_ID"]
        oldapp.config.AUTH_COOKIE_SECURE = self.original["AUTH_COOKIE_SECURE"]
        oldapp.config.OFFLINE_VIDEO_ENABLED = self.original["OFFLINE_VIDEO_ENABLED"]
        oldapp.config.PROBE_SNAP_ENABLED = self.original["PROBE_SNAP_ENABLED"]
        oldapp.config.INDEXED_FOLDER_ENABLED = self.original["INDEXED_FOLDER_ENABLED"]
        oldapp._auth_service = None
        oldapp._identity_repository = None
        oldapp._audit_writer = None
        oldapp._audit_reader = None
        oldapp._audit_db_pool = None

    def _login(self):
        response = self.client.post(
            "/auth/login",
            json={
                "username": "engineer",
                "password": "correct-password",
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        csrf_cookie = self.client.get_cookie(oldapp.config.AUTH_CSRF_COOKIE)
        self.assertIsNotNone(csrf_cookie)
        return response, csrf_cookie.value

    def test_malformed_login_is_audited(self) -> None:
        response = self.client.post("/auth/login", json={"username": "engineer"})

        self.assertEqual(response.status_code, 400)
        self.assertTrue(
            any(
                event.action == "auth.login"
                and event.result == "denied"
                and event.details.get("reason") == "missing_credentials"
                for event in self.audit.events
            )
        )

    def test_login_me_and_logout(self) -> None:
        login, csrf_token = self._login()

        session_header = next(
            value
            for value in login.headers.getlist("Set-Cookie")
            if value.startswith(f"{oldapp.config.AUTH_SESSION_COOKIE}=")
        )
        self.assertIn("HttpOnly", session_header)
        self.assertIn("SameSite=Strict", session_header)

        me = self.client.get("/auth/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.get_json()["user"]["username"], "engineer")

        missing_csrf = self.client.post("/auth/logout")
        self.assertEqual(missing_csrf.status_code, 403)

        logout = self.client.post(
            "/auth/logout",
            headers={"X-CSRF-Token": csrf_token},
        )
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        self.assertTrue(
            any(
                event.action == "auth.logout.completed"
                and event.result == "success"
                and event.details.get("revoked") is True
                for event in self.audit.events
            )
        )

    def test_logout_does_not_stop_background_captures(self) -> None:
        _, csrf_token = self._login()

        with (
            patch("oldapp.luxriot_manager.stop_session") as stop_video,
            patch("oldapp.luxriot_manager.stop_probe_capture") as stop_probe,
            patch("oldapp.luxriot_manager.stop_all_streams") as stop_all,
        ):
            logout = self.client.post(
                "/auth/logout",
                headers={"X-CSRF-Token": csrf_token},
            )

        self.assertEqual(logout.status_code, 200, logout.get_json())
        stop_video.assert_not_called()
        stop_probe.assert_not_called()
        stop_all.assert_not_called()

    def test_mutation_requires_csrf_permission_and_channel_grant(self) -> None:
        _, csrf_token = self._login()

        missing_csrf = self.client.post(
            "/probes/delete",
            json={"id": "missing-probe", "channel_id": 7},
        )
        self.assertEqual(missing_csrf.status_code, 403)

        forbidden_channel = self.client.post(
            "/probes/delete",
            headers={"X-CSRF-Token": csrf_token},
            json={"id": "missing-probe", "channel_id": 8},
        )
        self.assertEqual(forbidden_channel.status_code, 403)

        with (
            patch(
                "oldapp.probes_store.list_probes",
                return_value=[{"id": "missing-probe", "channel_id": 7}],
            ),
            patch("oldapp.probes_store.delete_probe", return_value=False),
        ):
            allowed = self.client.post(
                "/probes/delete",
                headers={"X-CSRF-Token": csrf_token},
                json={"id": "missing-probe", "channel_id": 7},
            )
        self.assertEqual(allowed.status_code, 404)
        self.assertTrue(
            any(
                event.action == "http.probes_delete.mutate"
                and event.result == "success"
                for event in self.audit.events
            )
        )

    def test_probe_cast_checks_every_selected_channel(self) -> None:
        _, csrf_token = self._login()

        with patch("oldapp.probes_store.upsert_probe") as upsert_probe:
            forbidden = self.client.post(
                "/probes/cast",
                headers={"X-CSRF-Token": csrf_token},
                json={
                    "name": "Door watch",
                    "channel_ids": [7, 8],
                    "positives": ["person near door"],
                    "bookmark": False,
                },
            )

        self.assertEqual(forbidden.status_code, 403)
        upsert_probe.assert_not_called()

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
            allowed = self.client.post(
                "/probes/cast",
                headers={"X-CSRF-Token": csrf_token},
                json={
                    "name": "Door watch",
                    "channel_ids": [7],
                    "positives": ["person near door"],
                    "bookmark": False,
                },
            )

        self.assertEqual(allowed.status_code, 200, allowed.get_json())
        self.assertEqual([probe["channel_id"] for probe in saved_probes], [7])

    def test_sensitive_reads_require_login_and_channel_scope(self) -> None:
        anonymous = self.client.get("/luxriot/channels")
        self.assertEqual(anonymous.status_code, 401)
        anonymous_media = self.client.head("/luxriot/media/live/7")
        self.assertEqual(anonymous_media.status_code, 401)

        self._login()
        allowed = self.client.get("/luxriot/channels")
        self.assertNotEqual(allowed.status_code, 401)

        denied_snapshot = self.client.get("/luxriot/snapshot/8")
        self.assertEqual(denied_snapshot.status_code, 403)
        denied_media = self.client.head("/luxriot/media/archive/8?time_ms=1700000000000")
        self.assertEqual(denied_media.status_code, 403)
        denied_archive_snapshot = self.client.get(
            "/luxriot/archive_snapshot/8?time_ms=1700000000000"
        )
        self.assertEqual(denied_archive_snapshot.status_code, 403)

        self.assertTrue(
            any(
                event.action == "http.luxriot_channels.access"
                and event.result == "success"
                for event in self.audit.events
            )
        )
        self.assertTrue(
            any(
                event.action == "http.luxriot_snapshot.access"
                and event.result == "denied"
                and event.channel_id == 8
                for event in self.audit.events
            )
        )
        self.assertTrue(
            any(
                event.action == "http.luxriot_media.access"
                and event.result == "denied"
                and event.channel_id == 8
                for event in self.audit.events
            )
        )
        self.assertTrue(
            any(
                event.action == "http.luxriot_archive_snapshot.access"
                and event.result == "denied"
                and event.channel_id == 8
                for event in self.audit.events
            )
        )

    def test_road_scene_overlay_requires_diagnostics_permission(self) -> None:
        self._login()
        denied = self.client.get("/road/scene_overlay/7")
        self.assertEqual(denied.status_code, 403)

    def test_luxriot_recent_frame_prefers_eva_buffer(self) -> None:
        self._login()
        image = Image.new("RGB", (32, 24), (80, 120, 160))
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG")
        snapshot_b64 = base64.b64encode(jpeg.getvalue()).decode("ascii")

        fake_session = SimpleNamespace(
            recent_frame_items=lambda limit=1: [
                {
                    "thumbnail": snapshot_b64,
                    "captured_at": time.time(),
                    "width": 32,
                    "height": 24,
                }
            ],
            status=lambda: {
                "running": True,
                "recent_frame_count": 1,
                "last_error": None,
                "active_capture_source": "live_segment",
            },
        )
        with oldapp.luxriot_manager.cache_lock:
            had_video = 7 in oldapp.luxriot_manager.sessions
            old_video = oldapp.luxriot_manager.sessions.get(7)
            oldapp.luxriot_manager.sessions[7] = fake_session
        try:
            response = self.client.get("/luxriot/recent_frame/7?fallback=0")
        finally:
            with oldapp.luxriot_manager.cache_lock:
                if had_video:
                    oldapp.luxriot_manager.sessions[7] = old_video
                else:
                    oldapp.luxriot_manager.sessions.pop(7, None)

        self.assertEqual(response.status_code, 200, response.get_json(silent=True))
        self.assertEqual(response.headers.get("Content-Type"), "image/jpeg")
        self.assertEqual(response.headers.get("X-EVA-Frame-Source"), "eva_recent")
        self.assertEqual(response.headers.get("X-EVA-Signal"), "fresh")
        self.assertGreater(len(response.data), 20)

    def test_luxriot_recent_frame_rejects_stale_eva_buffer(self) -> None:
        self._login()
        image = Image.new("RGB", (32, 24), (80, 120, 160))
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG")
        snapshot_b64 = base64.b64encode(jpeg.getvalue()).decode("ascii")
        fake_session = SimpleNamespace(
            recent_frame_items=lambda limit=1: [
                {
                    "thumbnail": snapshot_b64,
                    "captured_at": time.time() - 600,
                    "width": 32,
                    "height": 24,
                }
            ],
            status=lambda: {
                "running": True,
                "recent_frame_count": 1,
                "last_error": "Luxriot snapshot failed",
                "active_capture_source": "live_segment",
            },
        )
        with oldapp.luxriot_manager.cache_lock:
            had_video = 7 in oldapp.luxriot_manager.sessions
            old_video = oldapp.luxriot_manager.sessions.get(7)
            oldapp.luxriot_manager.sessions[7] = fake_session
        try:
            response = self.client.get("/luxriot/recent_frame/7?fallback=0&max_age_sec=45")
        finally:
            with oldapp.luxriot_manager.cache_lock:
                if had_video:
                    oldapp.luxriot_manager.sessions[7] = old_video
                else:
                    oldapp.luxriot_manager.sessions.pop(7, None)

        self.assertEqual(response.status_code, 503)
        body = response.get_json()
        self.assertEqual(body["error_code"], "signal_lost")
        self.assertEqual(body["last_error"], "Luxriot snapshot failed")
        self.assertGreater(body["last_frame_age_sec"], 45)

    def test_luxriot_recent_frame_cycle_skips_stale_candidates(self) -> None:
        self._login()
        image = Image.new("RGB", (32, 24), (80, 120, 160))
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG")
        snapshot_b64 = base64.b64encode(jpeg.getvalue()).decode("ascii")
        fake_session = SimpleNamespace(
            recent_frame_items=lambda limit=1: [
                {
                    "thumbnail": snapshot_b64,
                    "captured_at": 900.0,
                    "width": 32,
                    "height": 24,
                },
                {
                    "thumbnail": snapshot_b64,
                    "captured_at": 990.0,
                    "width": 32,
                    "height": 24,
                },
            ],
            status=lambda: {
                "running": True,
                "recent_frame_count": 2,
                "last_error": None,
                "active_capture_source": "live_segment",
            },
        )
        with oldapp.luxriot_manager.cache_lock:
            had_video = 7 in oldapp.luxriot_manager.sessions
            old_video = oldapp.luxriot_manager.sessions.get(7)
            oldapp.luxriot_manager.sessions[7] = fake_session
        try:
            with patch("oldapp.time.time", return_value=1000.0):
                response = self.client.get("/luxriot/recent_frame/7?mode=cycle&fps=1&fallback=0&max_age_sec=45")
        finally:
            with oldapp.luxriot_manager.cache_lock:
                if had_video:
                    oldapp.luxriot_manager.sessions[7] = old_video
                else:
                    oldapp.luxriot_manager.sessions.pop(7, None)

        self.assertEqual(response.status_code, 200, response.get_json(silent=True))
        self.assertEqual(response.headers.get("X-EVA-Signal"), "fresh")
        self.assertLess(float(response.headers.get("X-EVA-Frame-Age-Sec")), 45)

    def test_luxriot_recent_frame_rejects_frozen_signal(self) -> None:
        self._login()
        image = Image.new("RGB", (32, 24), (80, 120, 160))
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG")
        snapshot_b64 = base64.b64encode(jpeg.getvalue()).decode("ascii")
        fake_session = SimpleNamespace(
            recent_frame_items=lambda limit=1: [
                {
                    "thumbnail": snapshot_b64,
                    "captured_at": time.time(),
                    "width": 32,
                    "height": 24,
                }
            ],
            status=lambda: {
                "running": True,
                "recent_frame_count": 1,
                "last_error": None,
                "active_capture_source": "live_segment",
                "frozen_signal": True,
                "frozen_signal_age_sec": 22.0,
                "frozen_frame_count": 5,
            },
        )
        with oldapp.luxriot_manager.cache_lock:
            had_video = 7 in oldapp.luxriot_manager.sessions
            old_video = oldapp.luxriot_manager.sessions.get(7)
            oldapp.luxriot_manager.sessions[7] = fake_session
        try:
            response = self.client.get("/luxriot/recent_frame/7?fallback=0&max_age_sec=45")
        finally:
            with oldapp.luxriot_manager.cache_lock:
                if had_video:
                    oldapp.luxriot_manager.sessions[7] = old_video
                else:
                    oldapp.luxriot_manager.sessions.pop(7, None)

        self.assertEqual(response.status_code, 503)
        body = response.get_json()
        self.assertEqual(body["error_code"], "signal_frozen")
        self.assertTrue(body["frozen_signal"])
        self.assertEqual(body["frozen_frame_count"], 5)

    def test_road_scene_overlay_returns_bounded_preview_for_engineer(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.STREAMS_VIEW.value,
                    Permission.DIAGNOSTICS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        self._login()
        image = Image.new("RGB", (64, 48), (24, 28, 32))
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG")
        snapshot_b64 = base64.b64encode(jpeg.getvalue()).decode("ascii")

        zone = SimpleNamespace(
            enabled=True,
            polygon=((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)),
            expected_flow=(1.0, 0.0),
        )
        result = SimpleNamespace(
            scene_card=SimpleNamespace(zones=(zone,)),
            as_dict=lambda: {
                "confidence": "medium",
                "reason": "motion-history zone inferred",
                "frame_count": 12,
                "motion_pair_count": 6,
                "scene_cut_count": 0,
                "zone_area_ratio": 0.42,
                "flow_dominance": 0.7,
                "scene_card": {
                    "channel_id": 7,
                    "zones": [
                        {
                            "name": "auto_motion_zone",
                            "polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
                            "expected_flow": [1.0, 0.0],
                        }
                    ],
                },
            },
        )
        fake_session = SimpleNamespace(
            lock=threading.Lock(),
            frames=[
                {
                    "thumbnail": snapshot_b64,
                    "captured_at": time.time() - (12 - index),
                }
                for index in range(12)
            ],
        )
        with oldapp.luxriot_manager.cache_lock:
            had_video = 7 in oldapp.luxriot_manager.sessions
            old_video = oldapp.luxriot_manager.sessions.get(7)
            had_probe = 7 in oldapp.luxriot_manager.probe_sessions
            old_probe = oldapp.luxriot_manager.probe_sessions.get(7)
            oldapp.luxriot_manager.sessions[7] = fake_session
            oldapp.luxriot_manager.probe_sessions.pop(7, None)
        try:
            with patch("oldapp.infer_scene_card_from_frames", return_value=result):
                response = self.client.get("/road/scene_overlay/7?seconds=999&frames=999&mb=999")
        finally:
            with oldapp.luxriot_manager.cache_lock:
                if had_video:
                    oldapp.luxriot_manager.sessions[7] = old_video
                else:
                    oldapp.luxriot_manager.sessions.pop(7, None)
                if had_probe:
                    oldapp.luxriot_manager.probe_sessions[7] = old_probe
                else:
                    oldapp.luxriot_manager.probe_sessions.pop(7, None)

        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(body["channel_id"], 7)
        self.assertEqual(body["source"], "eva_capture_buffer")
        self.assertTrue(body["overlay_b64"])
        self.assertEqual(body["scene"]["confidence"], "medium")
        self.assertEqual(body["budget"]["frames"], 120)
        self.assertEqual(body["budget"]["max_edge"], 240)
        self.assertEqual(body["snapshot_meta"]["width"], 64)
        self.assertEqual(body["snapshot_meta"]["height"], 48)
        self.assertEqual(body["snapshot_meta"]["frame_count"], 12)

    def test_channel_lists_and_stream_status_are_filtered(self) -> None:
        self._login()
        streams = {
            "video_streams": [
                {"channel_id": 7},
                {"channel_id": 8},
            ],
            "analytics_streams": [{"channel_id": 8}],
            "paused_analytics_channels": [7, 8],
            "video_history_channels": [7, 8],
            "running_total": 3,
        }
        with (
            patch(
                "oldapp.luxriot_manager.get_channels",
                return_value=[{"id": 7}, {"id": 8}],
            ),
            patch(
                "oldapp.luxriot_manager.streams_status",
                return_value=streams,
            ),
        ):
            channels = self.client.get("/luxriot/channels")
            stream_status = self.client.get("/luxriot/streams")

        self.assertEqual(channels.get_json()["channels"], [{"id": 7}])
        payload = stream_status.get_json()
        self.assertEqual(payload["video_streams"], [{"channel_id": 7}])
        self.assertEqual(payload["analytics_streams"], [])
        self.assertEqual(payload["paused_analytics_channels"], [7])
        self.assertEqual(payload["video_history_channels"], [7])
        self.assertEqual(payload["running_total"], 1)

    def test_stop_stream_response_is_filtered_to_channel_grants(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.CAPTURE_MANAGE.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()
        streams = {
            "video_streams": [
                {"channel_id": 7},
                {"channel_id": 8},
            ],
            "analytics_streams": [{"channel_id": 8}],
            "paused_analytics_channels": [7, 8],
            "video_history_channels": [7, 8],
            "running_total": 3,
        }
        with (
            patch(
                "oldapp.luxriot_manager.stop_stream",
                return_value={"stopped": True, "channel_id": 7},
            ) as stop_stream,
            patch(
                "oldapp.luxriot_manager.streams_status",
                return_value=streams,
            ),
        ):
            response = self.client.post(
                "/luxriot/streams/stop",
                headers={"X-CSRF-Token": csrf_token},
                json={"channel_id": 7},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        stop_stream.assert_called_once()
        payload = response.get_json()["streams"]
        self.assertEqual(payload["video_streams"], [{"channel_id": 7}])
        self.assertEqual(payload["analytics_streams"], [])
        self.assertEqual(payload["paused_analytics_channels"], [7])
        self.assertEqual(payload["video_history_channels"], [7])
        self.assertEqual(payload["running_total"], 1)

    def test_luxriot_start_capture_writes_completion_audit(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.CAPTURE_MANAGE.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()

        with patch(
            "oldapp.luxriot_manager.start_session",
            return_value={"running": True, "channel_id": 7},
        ) as start_session:
            response = self.client.post(
                "/luxriot/start_capture",
                headers={"X-CSRF-Token": csrf_token},
                json={
                    "channel_id": 7,
                    "batch_size": 16,
                    "interval_sec": 2.5,
                    "prompt": "operator sensitive prompt",
                    "system_prompt": "system sensitive prompt",
                    "model": "qwen35-4b-q4",
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(start_session.call_args.kwargs["interval_sec"], 2.5)
        event = next(
            event
            for event in self.audit.events
            if event.action == "luxriot.capture.start.completed"
        )
        self.assertEqual(event.result, "success")
        self.assertEqual(event.channel_id, 7)
        self.assertTrue(event.details["prompt_supplied"])
        self.assertTrue(event.details["system_prompt_supplied"])
        self.assertTrue(event.details["model_supplied"])
        self.assertTrue(event.details["interval_sec_supplied"])
        self.assertEqual(event.details["interval_sec"], 2.5)
        self.assertNotIn("operator sensitive prompt", str(event.details))
        self.assertNotIn("system sensitive prompt", str(event.details))

    def test_luxriot_start_capture_auto_balances_to_profile(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.CAPTURE_MANAGE.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()
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
                "model": "qwen-vlm",
                "api_key": "",
                "timeout": 300,
                "enabled": True,
            },
            "vlm-b": {
                "id": "vlm-b",
                "kind": "vlm",
                "base_url": "http://vlm-b.local/v1",
                "model": "qwen-vlm",
                "api_key": "",
                "timeout": 300,
                "enabled": True,
            },
        }
        captured = {}

        def fake_start_session(channel_id, **kwargs):
            captured["channel_id"] = channel_id
            captured.update(kwargs)
            return {"running": True, "channel_id": channel_id, "model": kwargs.get("model_hint")}

        with (
            patch.object(oldapp.config, "LM_PROFILES", profiles),
            patch.object(oldapp.config, "LM_VLM_PROFILE_ID", "vlm-a"),
            patch.object(oldapp.config, "LM_VLM_BALANCER_ENABLED", True),
            patch.object(oldapp.config, "LM_VLM_BALANCER_PROFILES", ("vlm-a", "vlm-b")),
            patch("oldapp.luxriot_manager.start_session", side_effect=fake_start_session),
        ):
            response = self.client.post(
                "/luxriot/start_capture",
                headers={"X-CSRF-Token": csrf_token},
                json={"channel_id": 7, "batch_size": 16, "model": "__auto__"},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertIn(captured["model_hint"], {"vlm-a", "vlm-b"})
        payload = response.get_json()
        self.assertEqual(payload["session"]["model_selection"], "auto")
        self.assertEqual(payload["session"]["assigned_profile_id"], captured["model_hint"])
        event = next(
            event
            for event in self.audit.events
            if event.action == "luxriot.capture.start.completed"
        )
        self.assertEqual(event.details["model_selection"], "auto")
        self.assertEqual(event.details["assigned_profile_id"], captured["model_hint"])
        self.assertTrue(event.details["balancer_enabled"])

    def test_luxriot_start_capture_accepts_fps_alias(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.CAPTURE_MANAGE.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()

        with patch(
            "oldapp.luxriot_manager.start_session",
            return_value={"running": True, "channel_id": 7},
        ) as start_session:
            response = self.client.post(
                "/luxriot/start_capture",
                headers={"X-CSRF-Token": csrf_token},
                json={"channel_id": 7, "batch_size": 12, "fps": 0.2},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(start_session.call_args.kwargs["interval_sec"], 5.0)

    def test_scoped_detection_queries_require_owned_channel(self) -> None:
        self._login()

        missing_scope = self.client.get("/detections/list")
        forbidden_scope = self.client.get("/detections/list?channel_id=8")
        allowed_scope = self.client.get("/detections/list?channel_id=7")

        self.assertEqual(missing_scope.status_code, 403)
        self.assertEqual(forbidden_scope.status_code, 403)
        self.assertEqual(allowed_scope.status_code, 200)

    def test_archive_not_ready_response_is_sanitized(self) -> None:
        self._login()

        with patch(
            "oldapp.detections_store.list_detections",
            side_effect=oldapp.ArchiveStoreNotReady(
                "Archive storage is not migrated yet. Apply database migration 20260612_0005."
            ),
        ):
            response = self.client.get("/detections/list?channel_id=7")

        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(payload["not_ready"], "archive_store")
        self.assertEqual(payload["required_revision"], "20260612_0005")
        self.assertNotIn("archive.detections", payload["error"])

    def test_luxriot_prompt_bookmark_settings_require_bookmark_permission(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.PROMPTS_MANAGE.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()

        with patch("oldapp.luxriot_manager.update_prompt_settings") as update_settings:
            denied = self.client.post(
                "/luxriot/prompt_settings",
                headers={"X-CSRF-Token": csrf_token},
                json={"channel_id": 7, "bookmark_enabled": True},
            )

        self.assertEqual(denied.status_code, 403)
        update_settings.assert_not_called()

        with patch("oldapp.luxriot_manager.update_prompt_settings") as update_settings:
            denied_json_prompt = self.client.post(
                "/luxriot/prompt_settings",
                headers={"X-CSRF-Token": csrf_token},
                json={"channel_id": 7, "json_alert_prompt": "emit every vehicle as alert"},
            )

        self.assertEqual(denied_json_prompt.status_code, 403)
        update_settings.assert_not_called()

        with patch(
            "oldapp.luxriot_manager.update_prompt_settings",
            return_value={"stream_system_prompt": "ok"},
        ) as update_settings:
            allowed_prompt_only = self.client.post(
                "/luxriot/prompt_settings",
                headers={"X-CSRF-Token": csrf_token},
                json={
                    "channel_id": 7,
                    "stream_system_prompt": "plain prompt",
                    "alert_policy_prompt": "watch for visible falls near stairs",
                },
            )

        self.assertEqual(allowed_prompt_only.status_code, 200, allowed_prompt_only.get_json())
        self.assertEqual(
            update_settings.call_args.kwargs["alert_policy_prompt"],
            "watch for visible falls near stairs",
        )
        self.assertIsNone(update_settings.call_args.kwargs["bookmark_enabled"])
        self.assertIsNone(update_settings.call_args.kwargs["bookmark_cooldown_sec"])

        with patch("oldapp.luxriot_manager.update_prompt_settings") as update_settings:
            denied_bookmark_reset = self.client.post(
                "/luxriot/prompt_settings",
                headers={"X-CSRF-Token": csrf_token},
                json={"channel_id": 7, "clear_override_fields": ["bookmark_enabled"]},
            )

        self.assertEqual(denied_bookmark_reset.status_code, 403)
        update_settings.assert_not_called()

        with patch(
            "oldapp.luxriot_manager.update_prompt_settings",
            return_value={"stream_system_prompt": "inherited"},
        ) as update_settings:
            allowed_prompt_reset = self.client.post(
                "/luxriot/prompt_settings",
                headers={"X-CSRF-Token": csrf_token},
                json={"channel_id": 7, "clear_override_fields": ["stream_system_prompt"]},
            )

        self.assertEqual(allowed_prompt_reset.status_code, 200, allowed_prompt_reset.get_json())
        self.assertEqual(
            update_settings.call_args.kwargs["clear_override_fields"],
            ["stream_system_prompt"],
        )

    def test_luxriot_rollups_reject_invalid_target_level_as_bad_request(self) -> None:
        self.repository.identity = replace(
            self.repository.identity,
            permissions=frozenset(
                {
                    Permission.REPORTS_VIEW.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
        )
        self.repository.users[USER_ID] = self.repository.identity
        self._login()

        response = self.client.get("/luxriot/rollups?channel_id=7&target_level=L9")

        self.assertEqual(response.status_code, 400)
        self.assertIn("target_level", response.get_json()["error"])

    def test_probe_bookmark_save_requires_bookmark_permission(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.PROBES_MANAGE.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()

        with patch("oldapp.probes_store.upsert_probe") as upsert_probe:
            response = self.client.post(
                "/probes/save",
                headers={"X-CSRF-Token": csrf_token},
                json={
                    "channel_id": 7,
                    "name": "door",
                    "positives": ["person at door"],
                    "bookmark": True,
                },
            )

        self.assertEqual(response.status_code, 403)
        upsert_probe.assert_not_called()

    def test_probe_bookmark_query_requires_bookmark_permission(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.PROBES_RUN.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()

        with patch("oldapp.probe_manager.query") as query:
            response = self.client.post(
                "/probes/query",
                headers={"X-CSRF-Token": csrf_token},
                json={
                    "channel_id": 7,
                    "positives": ["person"],
                    "bookmark": True,
                },
            )

        self.assertEqual(response.status_code, 403)
        query.assert_not_called()

    def test_probe_bookmark_run_requires_bookmark_permission(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.PROBES_RUN.value,
                    Permission.STREAMS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()
        probe = {
            "id": "probe-7",
            "channel_id": 7,
            "positives": ["person"],
            "bookmark": True,
        }

        with (
            patch("oldapp.probes_store.list_probes", return_value=[probe]),
            patch("oldapp.probe_manager.query") as query,
        ):
            response = self.client.post(
                "/probes/run",
                headers={"X-CSRF-Token": csrf_token},
                json={"id": "probe-7"},
            )

        self.assertEqual(response.status_code, 403)
        query.assert_not_called()

    def test_probe_id_ownership_is_checked_before_mutation(self) -> None:
        _, csrf_token = self._login()
        with (
            patch(
                "oldapp.probes_store.list_probes",
                return_value=[
                    {
                        "id": "probe-8",
                        "channel_id": 8,
                    }
                ],
            ),
            patch(
                "oldapp.probes_store.delete_probe",
                return_value=True,
            ) as delete_probe,
        ):
            response = self.client.post(
                "/probes/delete",
                headers={"X-CSRF-Token": csrf_token},
                json={"id": "probe-8"},
            )

        self.assertEqual(response.status_code, 403)
        delete_probe.assert_not_called()

    def test_probe_id_lookup_failure_denies_scoped_mutation(self) -> None:
        _, csrf_token = self._login()
        with (
            patch(
                "oldapp.probes_store.list_probes",
                side_effect=RuntimeError("probe store unavailable"),
            ),
            patch("oldapp.probes_store.delete_probe") as delete_probe,
        ):
            response = self.client.post(
                "/probes/delete",
                headers={"X-CSRF-Token": csrf_token},
                json={"id": "probe-7", "channel_id": 7},
            )

        self.assertEqual(response.status_code, 403)
        delete_probe.assert_not_called()

    def test_probe_id_missing_owner_denies_scoped_mutation(self) -> None:
        _, csrf_token = self._login()
        with (
            patch("oldapp.probes_store.list_probes", return_value=[]),
            patch("oldapp.probes_store.delete_probe") as delete_probe,
        ):
            response = self.client.post(
                "/probes/delete",
                headers={"X-CSRF-Token": csrf_token},
                json={"id": "missing-probe", "channel_id": 7},
            )

        self.assertEqual(response.status_code, 403)
        delete_probe.assert_not_called()

    def test_legacy_folder_routes_require_all_channel_for_scoped_users(self) -> None:
        _, csrf_token = self._login()

        image = self.client.get("/image?folder=/tmp&image_path=frame.jpg")
        search = self.client.post(
            "/search",
            headers={"X-CSRF-Token": csrf_token},
            json={"folder": "/tmp", "query": "person"},
        )
        describe = self.client.post(
            "/describe_image",
            headers={"X-CSRF-Token": csrf_token},
            json={"folder": "/tmp", "image_path": "/tmp/frame.jpg"},
        )

        self.assertEqual(image.status_code, 403)
        self.assertEqual(search.status_code, 403)
        self.assertEqual(describe.status_code, 403)

    def test_lm_models_requires_authenticated_diagnostics_permission(self) -> None:
        anonymous = self.client.get("/lm/models")
        anonymous_admission = self.client.get("/lm/admission")
        self.assertEqual(anonymous.status_code, 401)
        self.assertEqual(anonymous_admission.status_code, 401)

        self.repository.identity = _Identity(
            permissions=frozenset(
                {
                    Permission.DIAGNOSTICS_VIEW.value,
                }
            ),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        self._login()
        with (
            patch(
                "oldapp._fetch_lm_model_catalog",
                return_value={
                    "profiles": [{"id": "vlm-4b", "model": "qwen35-4b"}],
                    "models": ["qwen35-4b"],
                },
            ),
            patch(
                "oldapp._get_agent_config_payload",
                return_value={"model": "qwen35-9b"},
            ),
            patch("oldapp._lm_admission_profiles", return_value=[]),
        ):
            scoped_engineer = self.client.get("/lm/models?force=1")
            admission = self.client.get("/lm/admission")
        self.assertEqual(scoped_engineer.status_code, 200)
        self.assertEqual(admission.status_code, 200)
        self.assertTrue(admission.get_json()["enabled"])
        event = next(
            event for event in self.audit.events if event.action == "lm.models.completed"
        )
        self.assertEqual(event.result, "success")
        self.assertEqual(event.details["profile_count"], 1)
        self.assertEqual(event.details["model_count"], 1)
        self.assertTrue(event.details["force"])

    def test_lm_admission_reports_served_model_match_without_credentials(self) -> None:
        self.repository.identity = _Identity(
            permissions=frozenset({Permission.DIAGNOSTICS_VIEW.value}),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        self._login()
        profiles = {
            "match": {
                "id": "match",
                "kind": "vlm",
                "base_url": "https://user:password@lm-match.example/v1",
                "model": "model-a",
                "api_key": "bearer-secret",
                "timeout": 30,
            },
            "mismatch": {
                "id": "mismatch",
                "kind": "agent",
                "base_url": "https://lm-mismatch.example/v1",
                "model": "model-b",
                "api_key": "",
                "timeout": 30,
            },
            "unreachable": {
                "id": "unreachable",
                "kind": "vlm",
                "base_url": "https://lm-down.example/v1",
                "model": "model-c",
                "api_key": "down-secret",
                "timeout": 30,
            },
        }

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        def get_models(url, **_kwargs):
            if "lm-match.example" in url:
                return Response({"data": [{"id": "model-a", "max_model_len": 32768}]})
            if "lm-mismatch.example" in url:
                return Response({"data": [{"id": "actually-served"}]})
            raise oldapp.requests.Timeout("credential-bearing endpoint unavailable")

        with oldapp._lm_served_models_cache_lock:
            oldapp._lm_served_models_cache.clear()
        with (
            patch("oldapp._configured_lm_profiles", return_value=profiles),
            patch("oldapp.requests.get", side_effect=get_models) as get,
        ):
            response = self.client.get("/lm/admission")
            cached_response = self.client.get("/lm/admission")

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(cached_response.status_code, 200, cached_response.get_json())
        self.assertEqual(get.call_count, 3)
        rows = {row["id"]: row for row in response.get_json()["profiles"]}
        self.assertIs(rows["match"]["model_match"], True)
        self.assertEqual(rows["match"]["served_models"], ["model-a"])
        self.assertEqual(rows["match"]["served_context_length"], 32768)
        self.assertIs(rows["mismatch"]["model_match"], False)
        self.assertEqual(rows["mismatch"]["served_models"], ["actually-served"])
        self.assertEqual(rows["unreachable"]["model_match"], "unknown")
        self.assertEqual(rows["unreachable"]["served_models"], [])
        serialized = str(response.get_json())
        for secret in ("user", "password", "bearer-secret", "down-secret"):
            self.assertNotIn(secret, serialized)
        match_call = next(call for call in get.call_args_list if "lm-match.example" in call.args[0])
        self.assertEqual(match_call.kwargs["headers"]["Authorization"], "Bearer bearer-secret")
        self.assertEqual(match_call.kwargs["timeout"], 3.0)

    def test_settings_env_write_audits_keys_without_secret_values(self) -> None:
        self.repository.identity = _Identity(
            roles=frozenset({Role.ADMIN.value}),
            permissions=frozenset(permission.value for permission in Permission),
            allowed_channel_ids=frozenset({ALL_CHANNELS}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()
        writes = []

        class _FakePath:
            def __init__(self, value):
                self.value = value

            def write_text(self, text, encoding=None):
                writes.append((self.value, text, encoding))
                return len(text)

        with (
            patch("oldapp.Path", _FakePath),
            patch(
                "oldapp._effective_env_map",
                return_value={"EVOSSEARCH_LUXRIOT_PASSWORD": "old-secret"},
            ),
            patch("oldapp._read_env_file_map", return_value={}),
        ):
            response = self.client.post(
                "/settings/env",
                headers={"X-CSRF-Token": csrf_token},
                json={
                    "envVariables": {
                        "EVOSSEARCH_LUXRIOT_PASSWORD": "new-secret",
                        "EVOSSEARCH_LM_BASE_URL": "http://llm.internal:8080",
                    }
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(writes[0][0], ".env")
        event = next(
            event
            for event in self.audit.events
            if event.action == "settings.env.write.completed"
        )
        self.assertEqual(event.result, "success")
        self.assertIn("EVOSSEARCH_LUXRIOT_PASSWORD", event.details["keys"])
        self.assertIn("EVOSSEARCH_LM_BASE_URL", event.details["keys"])
        self.assertNotIn("new-secret", str(event.details))
        self.assertNotIn("llm.internal", str(event.details))

    def test_detection_image_requires_owned_metadata(self) -> None:
        self._login()
        with patch(
            "oldapp.detections_store.channel_ids_for_image_path",
            return_value=frozenset({8}),
        ):
            response = self.client.get(
                "/detections/image?image_path=archive/forbidden.jpg"
            )

        self.assertEqual(response.status_code, 403)

    def test_detection_thumbnail_requires_owned_metadata(self) -> None:
        self._login()
        with patch(
            "oldapp.detections_store.fetch_detections_by_ids",
            return_value=[{"id": 501, "channel_id": 8, "thumbnail": "ZmFrZQ=="}],
        ):
            response = self.client.get("/detections/thumbnail/501")

        self.assertEqual(response.status_code, 403)

    def test_detection_thumbnail_serves_authorized_thumbnail(self) -> None:
        self._login()
        with patch(
            "oldapp.detections_store.fetch_detections_by_ids",
            return_value=[{"id": 501, "channel_id": 7, "thumbnail": "ZmFrZQ=="}],
        ):
            response = self.client.get("/detections/thumbnail/501")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertEqual(response.data, b"fake")

    def test_audit_outage_blocks_mutation_before_handler(self) -> None:
        _, csrf_token = self._login()
        self.audit.error = RuntimeError("audit unavailable")

        response = self.client.post(
            "/probes/delete",
            headers={"X-CSRF-Token": csrf_token},
            json={"id": "missing-probe", "channel_id": 7},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "Audit service unavailable")

    def test_agent_receives_server_created_tool_context(self) -> None:
        _, csrf_token = self._login()
        runner = _AgentRunner()
        original_runner = oldapp._agent_runner
        oldapp._agent_runner = runner
        self.addCleanup(setattr, oldapp, "_agent_runner", original_runner)

        response = self.client.post(
            "/agent/chat",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "message": "inspect channel 7",
                "actor_id": "forged-admin",
                "allowed_channel_ids": ["*"],
            },
        )
        response.get_data()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(runner.calls), 1)
        context = runner.calls[0]["tool_context"]
        self.assertEqual(context.actor_id, USER_ID)
        self.assertEqual(context.tenant_id, TENANT_ID)
        self.assertEqual(context.allowed_channel_ids, frozenset({"7"}))
        self.assertNotEqual(context.actor_id, "forged-admin")

    def test_agent_action_plan_execute_uses_server_context(self) -> None:
        _, csrf_token = self._login()
        runner = _AgentRunner()
        original_runner = oldapp._agent_runner
        oldapp._agent_runner = runner
        self.addCleanup(setattr, oldapp, "_agent_runner", original_runner)

        response = self.client.post(
            "/agent/action-plans/plan-123/execute",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "session_id": "session-1",
                "preview": False,
                "actor_id": "forged-admin",
                "arguments": {"channel_id": 8},
            },
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["result"]["status"], "applied")
        self.assertEqual(len(runner.approvals), 1)
        plan_id, context = runner.approvals[0]
        self.assertEqual(plan_id, "plan-123")
        self.assertEqual(context.actor_id, USER_ID)
        self.assertEqual(context.tenant_id, TENANT_ID)
        self.assertEqual(context.session_id, "session-1")
        self.assertEqual(context.allowed_channel_ids, frozenset({"7"}))

    def test_non_admin_cannot_manage_users(self) -> None:
        self._login()

        response = self.client.get("/auth/users")

        self.assertEqual(response.status_code, 403)

    def test_admin_can_manage_users_and_revoke_sessions(self) -> None:
        self.repository.identity = _Identity(
            roles=frozenset({Role.ADMIN.value}),
            permissions=frozenset(permission.value for permission in Permission),
            allowed_channel_ids=frozenset({ALL_CHANNELS}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()

        missing_csrf = self.client.post(
            "/auth/users",
            json={
                "username": "viewer",
                "password": "correct-password",
                "roles": [Role.VIEWER.value],
                "allowedChannelIds": [7],
            },
        )
        self.assertEqual(missing_csrf.status_code, 403)

        created = self.client.post(
            "/auth/users",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "username": "viewer",
                "password": "correct-password",
                "displayName": "Pilot Viewer",
                "roles": [Role.VIEWER.value],
                "allowedChannelIds": [7],
            },
        )
        self.assertEqual(created.status_code, 201, created.get_json())
        created_user = created.get_json()["user"]
        self.assertEqual(created_user["username"], "viewer")
        self.assertEqual(created_user["allowedChannelIds"], [7])

        operator_all = self.client.post(
            "/auth/users",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "username": "operator-all",
                "password": "correct-password",
                "displayName": "All Channel Operator",
                "roles": [Role.OPERATOR.value],
                "allowedChannelIds": [ALL_CHANNELS],
            },
        )
        self.assertEqual(operator_all.status_code, 201, operator_all.get_json())
        self.assertEqual(
            operator_all.get_json()["user"]["allowedChannelIds"],
            [ALL_CHANNELS],
        )

        listed = self.client.get("/auth/users")
        self.assertEqual(listed.status_code, 200)
        self.assertIn(
            created_user["id"],
            {user["id"] for user in listed.get_json()["users"]},
        )

        patched = self.client.patch(
            f"/auth/users/{created_user['id']}",
            headers={"X-CSRF-Token": csrf_token},
            json={
                "roles": [Role.OPERATOR.value],
                "allowedChannelIds": [7, 8],
                "isActive": False,
            },
        )
        self.assertEqual(patched.status_code, 200, patched.get_json())
        self.assertFalse(patched.get_json()["user"]["isActive"])
        self.assertEqual(
            patched.get_json()["user"]["allowedChannelIds"],
            [7, 8],
        )

        revoked = self.client.post(
            f"/auth/users/{created_user['id']}/revoke-sessions",
            headers={"X-CSRF-Token": csrf_token},
            json={"reason": "pilot_rotation"},
        )
        self.assertEqual(revoked.status_code, 200, revoked.get_json())
        self.assertEqual(revoked.get_json()["revokedSessions"], 2)
        self.assertEqual(
            self.repository.revoked_user_sessions[-1],
            (created_user["id"], USER_ID, "pilot_rotation"),
        )
        self.assertTrue(
            any(
                event.action == "auth.users.update.completed"
                and event.target_id == created_user["id"]
                for event in self.audit.events
            )
        )

    def test_admin_can_list_and_revoke_single_session(self) -> None:
        self.repository.identity = _Identity(
            roles=frozenset({Role.ADMIN.value}),
            permissions=frozenset(permission.value for permission in Permission),
            allowed_channel_ids=frozenset({ALL_CHANNELS}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        _, csrf_token = self._login()
        session_id = self.repository.session_inventory[0].session_id

        listed = self.client.get("/auth/sessions")
        self.assertEqual(listed.status_code, 200, listed.get_json())
        sessions = listed.get_json()["sessions"]
        self.assertEqual(sessions[0]["id"], session_id)
        self.assertEqual(sessions[0]["userId"], USER_ID)
        self.assertIsNone(sessions[0]["revokedAt"])

        missing_csrf = self.client.post(
            f"/auth/sessions/{session_id}/revoke",
            json={"reason": "device_lost"},
        )
        self.assertEqual(missing_csrf.status_code, 403)

        revoked = self.client.post(
            f"/auth/sessions/{session_id}/revoke",
            headers={"X-CSRF-Token": csrf_token},
            json={"reason": "device_lost"},
        )
        self.assertEqual(revoked.status_code, 200, revoked.get_json())
        self.assertEqual(
            self.repository.revoked_session_ids[-1],
            (session_id, "device_lost"),
        )
        self.assertTrue(
            any(
                event.action == "auth.sessions.revoke.completed"
                and event.target_id == session_id
                for event in self.audit.events
            )
        )

    def test_admin_can_read_audit_events(self) -> None:
        self.repository.identity = _Identity(
            roles=frozenset({Role.ADMIN.value}),
            permissions=frozenset(permission.value for permission in Permission),
            allowed_channel_ids=frozenset({ALL_CHANNELS}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        self._login()

        response = self.client.get("/audit/events?limit=10&result=success")

        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body["events"][0]["action"], "auth.login")
        self.assertEqual(body["nextCursor"], "cursor-1")
        context, kwargs = self.audit_reader.calls[-1]
        self.assertEqual(context.user_id, USER_ID)
        self.assertEqual(kwargs["limit"], "10")
        self.assertEqual(kwargs["result"], "success")

    def test_non_admin_cannot_read_audit_events(self) -> None:
        self._login()

        response = self.client.get("/audit/events")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.audit_reader.calls, [])

    def test_scoped_audit_viewer_cannot_read_tenant_wide_audit(self) -> None:
        self.repository.identity = _Identity(
            roles=frozenset({Role.VIEWER.value}),
            permissions=frozenset({Permission.AUDIT_VIEW.value}),
            allowed_channel_ids=frozenset({7}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        self._login()

        response = self.client.get("/audit/events")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.audit_reader.calls, [])

    def test_audit_reader_validation_errors_are_400(self) -> None:
        self.repository.identity = _Identity(
            roles=frozenset({Role.ADMIN.value}),
            permissions=frozenset(permission.value for permission in Permission),
            allowed_channel_ids=frozenset({ALL_CHANNELS}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        self._login()
        self.audit_reader.error = ValueError("limit must be between 1 and 100")

        response = self.client.get("/audit/events?limit=101")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"],
            "limit must be between 1 and 100",
        )

    def test_audit_outage_blocks_audit_event_reads(self) -> None:
        self.repository.identity = _Identity(
            roles=frozenset({Role.ADMIN.value}),
            permissions=frozenset(permission.value for permission in Permission),
            allowed_channel_ids=frozenset({ALL_CHANNELS}),
        )
        self.repository.users[USER_ID] = self.repository.identity
        self._login()
        self.audit.error = RuntimeError("audit unavailable")

        response = self.client.get("/audit/events")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "Audit service unavailable")
        self.assertEqual(self.audit_reader.calls, [])


if __name__ == "__main__":
    unittest.main()
