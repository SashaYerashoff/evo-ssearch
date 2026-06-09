import builtins
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from unittest.mock import patch

from security import (
    ALL_CHANNELS,
    ALL_PERMISSIONS,
    REDACTED,
    Argon2idPasswordHasher,
    AuditEventBuilder,
    AuthContext,
    ChannelAccessDenied,
    InMemoryLoginThrottleRepository,
    LoginThrottlePolicy,
    LoginThrottleService,
    PasswordHashingUnavailable,
    Permission,
    Role,
    digest_session_token,
    generate_csrf_token,
    generate_session_token,
    permissions_for_roles,
    require_channel_authorization,
    role_permissions,
    verify_csrf_token,
    verify_session_token_digest,
)


class MutableClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_context(
    *,
    roles: set[str] | None = None,
    permissions: set[str] | None = None,
    channels: set[str | int] | None = None,
) -> AuthContext:
    return AuthContext(
        user_id="user-1",
        tenant_id="tenant-1",
        roles=frozenset(roles or {Role.OPERATOR.value}),
        permissions=frozenset(
            permissions
            or {permission.value for permission in role_permissions(Role.OPERATOR)}
        ),
        allowed_channel_ids=frozenset(channels or {"channel-1"}),
        request_id="request-1",
    )


class RoleAndAuthorizationTests(unittest.TestCase):
    def test_role_catalogue_assigns_expected_boundaries(self) -> None:
        viewer = role_permissions(Role.VIEWER)
        operator = role_permissions(Role.OPERATOR)
        engineer = role_permissions(Role.ENGINEER)

        self.assertIn(Permission.STREAMS_VIEW, viewer)
        self.assertNotIn(Permission.AGENT_USE, viewer)
        self.assertIn(Permission.AGENT_USE, operator)
        self.assertIn(Permission.PROBES_RUN, operator)
        self.assertNotIn(Permission.PROBES_MANAGE, operator)
        self.assertIn(Permission.PROBES_MANAGE, engineer)
        self.assertIn(Permission.PROMPTS_MANAGE, engineer)
        self.assertNotIn(Permission.USERS_MANAGE, engineer)
        self.assertEqual(role_permissions(Role.ADMIN), ALL_PERMISSIONS)
        self.assertEqual(
            permissions_for_roles({Role.VIEWER, Role.OPERATOR}),
            operator,
        )

    def test_auth_context_is_immutable_and_normalized(self) -> None:
        roles = {"Operator"}
        context = AuthContext(
            user_id="user-1",
            tenant_id="tenant-1",
            roles=roles,  # type: ignore[arg-type]
            permissions={"STREAMS:VIEW"},  # type: ignore[arg-type]
            allowed_channel_ids={"channel-1"},  # type: ignore[arg-type]
            request_id="request-1",
        )
        roles.add("admin")

        self.assertEqual(context.roles, frozenset({"operator"}))
        self.assertEqual(context.permissions, frozenset({"streams:view"}))
        with self.assertRaises(FrozenInstanceError):
            context.user_id = "attacker"  # type: ignore[misc]

    def test_permission_does_not_bypass_cross_channel_scope(self) -> None:
        context = make_context(channels={"channel-1"})

        require_channel_authorization(
            context,
            channel_id="channel-1",
            permission=Permission.STREAMS_VIEW,
        )
        with self.assertRaises(ChannelAccessDenied):
            require_channel_authorization(
                context,
                channel_id="channel-2",
                permission=Permission.STREAMS_VIEW,
            )

    def test_all_channel_grant_is_explicit(self) -> None:
        context = make_context(channels={ALL_CHANNELS})
        require_channel_authorization(
            context,
            channel_id=999,
            permission=Permission.DETECTIONS_VIEW,
        )


class PasswordHasherTests(unittest.TestCase):
    def test_missing_argon2_dependency_fails_closed(self) -> None:
        original_import = builtins.__import__

        def reject_argon2(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "argon2" or name.startswith("argon2."):
                raise ModuleNotFoundError("forced missing argon2")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=reject_argon2):
            with self.assertRaises(PasswordHashingUnavailable):
                Argon2idPasswordHasher()


class TokenTests(unittest.TestCase):
    def test_session_storage_uses_digest_not_bearer_token(self) -> None:
        token = generate_session_token()
        stored_digest = digest_session_token(token)

        self.assertNotEqual(token, stored_digest)
        self.assertNotIn(token, stored_digest)
        self.assertEqual(len(stored_digest), 64)
        self.assertTrue(verify_session_token_digest(token, stored_digest))
        self.assertFalse(
            verify_session_token_digest(generate_session_token(), stored_digest)
        )

    def test_csrf_tokens_use_constant_time_verification_contract(self) -> None:
        token = generate_csrf_token()

        self.assertTrue(verify_csrf_token(token, token))
        self.assertFalse(verify_csrf_token(token, generate_csrf_token()))
        self.assertFalse(verify_csrf_token(token, ""))


class LoginThrottleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = MutableClock(100.0)
        self.repository = InMemoryLoginThrottleRepository()
        self.service = LoginThrottleService(
            self.repository,
            policy=LoginThrottlePolicy(
                max_attempts=3,
                window_seconds=60,
                lockout_seconds=120,
            ),
            clock=self.clock,
        )

    def test_failures_lock_then_expire(self) -> None:
        key = "tenant-1:user@example.test:192.0.2.10"

        self.assertTrue(self.service.check(key).allowed)
        self.assertEqual(self.service.record_failure(key).attempts_remaining, 2)
        self.assertEqual(self.service.record_failure(key).attempts_remaining, 1)
        locked = self.service.record_failure(key)

        self.assertFalse(locked.allowed)
        self.assertEqual(locked.retry_after_seconds, 120)
        self.clock.advance(119)
        self.assertFalse(self.service.check(key).allowed)
        self.clock.advance(1)
        self.assertTrue(self.service.check(key).allowed)

    def test_success_and_failure_window_clear_attempts(self) -> None:
        key = "login-key"
        self.service.record_failure(key)
        self.service.record_success(key)
        self.assertEqual(self.service.check(key).attempts_remaining, 3)

        self.service.record_failure(key)
        self.clock.advance(60)
        self.assertEqual(self.service.check(key).attempts_remaining, 3)


class AuditEventTests(unittest.TestCase):
    def test_anonymous_event_can_be_bound_to_known_tenant(self) -> None:
        event = AuditEventBuilder().build(
            context=None,
            tenant_id="tenant-1",
            source_ip="192.0.2.20",
            action="auth.login",
            target_type="user",
            result="denied",
        )

        self.assertIsNone(event.actor_user_id)
        self.assertEqual(event.tenant_id, "tenant-1")

    def test_builder_uses_context_identity_and_recursively_redacts(self) -> None:
        context = make_context()
        timestamp = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
        builder = AuditEventBuilder(clock=lambda: timestamp)
        details = {
            "password": "correct horse battery staple",
            "api_key": "api-secret",
            "private_key": "private-secret",
            "prompt_id": "prompt-7",
            "prompt": "complete sensitive prompt",
            "nested": {
                "sessionToken": "bearer-secret",
                "image_bytes": b"raw-image",
                "safe": "retained",
            },
        }

        event = builder.build(
            context=context,
            source_ip="192.0.2.10",
            action="probe.update",
            target_type="probe",
            target_id="probe-7",
            channel_id="channel-1",
            result="success",
            details=details,
        )

        self.assertEqual(event.actor_user_id, context.user_id)
        self.assertEqual(event.tenant_id, context.tenant_id)
        self.assertEqual(event.request_id, context.request_id)
        self.assertEqual(event.details["password"], REDACTED)
        self.assertEqual(event.details["api_key"], REDACTED)
        self.assertEqual(event.details["private_key"], REDACTED)
        self.assertEqual(event.details["prompt"], REDACTED)
        self.assertEqual(event.details["prompt_id"], "prompt-7")
        self.assertEqual(event.details["nested"]["sessionToken"], REDACTED)
        self.assertEqual(event.details["nested"]["image_bytes"], REDACTED)
        self.assertEqual(event.details["nested"]["safe"], "retained")
        with self.assertRaises(TypeError):
            event.details["password"] = "restored"  # type: ignore[index]

        record = event.to_dict()
        self.assertEqual(record["timestamp"], timestamp.isoformat())
        self.assertEqual(record["details"]["nested"]["safe"], "retained")


if __name__ == "__main__":
    unittest.main()
