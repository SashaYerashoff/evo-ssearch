import unittest
from dataclasses import dataclass
from datetime import datetime

import oldapp
from unittest.mock import patch
from security import Permission, Role, digest_session_token
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


class _Repository:
    def __init__(self) -> None:
        self.identity = _Identity()
        self.sessions = {}

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
        return session_id

    def resolve_session(self, tenant_id, token):
        if tenant_id != TENANT_ID:
            return None
        return self.sessions.get(token)

    def revoke_session(self, tenant_id, token, reason):
        if tenant_id != TENANT_ID:
            return False
        return self.sessions.pop(token, None) is not None


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

    def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        yield 'data: {"type":"done","session_id":"session-1"}\n\n'


class HttpAuthRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = {
            "AUTH_ENABLED": oldapp.config.AUTH_ENABLED,
            "AUTH_TENANT_ID": oldapp.config.AUTH_TENANT_ID,
            "AUTH_COOKIE_SECURE": oldapp.config.AUTH_COOKIE_SECURE,
        }
        oldapp.config.AUTH_ENABLED = True
        oldapp.config.AUTH_TENANT_ID = TENANT_ID
        oldapp.config.AUTH_COOKIE_SECURE = False
        self.repository = _Repository()
        self.audit = _AuditWriter()
        oldapp._auth_service = AuthenticationService(
            self.repository,
            tenant_id=TENANT_ID,
        )
        oldapp._audit_writer = self.audit
        self.client = oldapp.app.test_client()

    def tearDown(self) -> None:
        oldapp.config.AUTH_ENABLED = self.original["AUTH_ENABLED"]
        oldapp.config.AUTH_TENANT_ID = self.original["AUTH_TENANT_ID"]
        oldapp.config.AUTH_COOKIE_SECURE = self.original["AUTH_COOKIE_SECURE"]
        oldapp._auth_service = None
        oldapp._audit_writer = None
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

    def test_sensitive_reads_require_login_and_channel_scope(self) -> None:
        anonymous = self.client.get("/luxriot/channels")
        self.assertEqual(anonymous.status_code, 401)

        self._login()
        allowed = self.client.get("/luxriot/channels")
        self.assertNotEqual(allowed.status_code, 401)

        denied_snapshot = self.client.get("/luxriot/snapshot/8")
        self.assertEqual(denied_snapshot.status_code, 403)

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

    def test_scoped_detection_queries_require_owned_channel(self) -> None:
        self._login()

        missing_scope = self.client.get("/detections/list")
        forbidden_scope = self.client.get("/detections/list?channel_id=8")
        allowed_scope = self.client.get("/detections/list?channel_id=7")

        self.assertEqual(missing_scope.status_code, 403)
        self.assertEqual(forbidden_scope.status_code, 403)
        self.assertEqual(allowed_scope.status_code, 200)

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


if __name__ == "__main__":
    unittest.main()
