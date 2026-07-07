import unittest
from dataclasses import dataclass
from datetime import datetime, timezone

from security import AuthContext, Permission, Role, digest_session_token
from security.http_auth import (
    AuthenticationService,
    InvalidCredentials,
    LoginThrottled,
)
from security.throttling import (
    InMemoryLoginThrottleRepository,
    LoginThrottlePolicy,
    LoginThrottleService,
)


@dataclass(frozen=True)
class _Identity:
    user_id: str = "user-1"
    tenant_id: str = "tenant-1"
    username: str = "operator"
    display_name: str | None = "Operator"
    roles: frozenset[str] = frozenset({Role.OPERATOR.value})
    permissions: frozenset[str] = frozenset({Permission.AGENT_USE.value})
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
        self.session = None
        self.revoked = False

    def authenticate(self, tenant_id, username, password):
        if (
            tenant_id == self.identity.tenant_id
            and username.lower() == self.identity.username
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
        self.session_token = token
        self.session = _Session(
            identity=identity,
            csrf_digest=digest_session_token(csrf_token),
            expires_at=expires_at,
        )
        return "session-1"

    def resolve_session(self, tenant_id, token):
        if (
            not self.revoked
            and tenant_id == self.identity.tenant_id
            and token == getattr(self, "session_token", None)
        ):
            return self.session
        return None

    def revoke_session(self, tenant_id, token, reason):
        if tenant_id != self.identity.tenant_id or token != self.session_token:
            return False
        self.revoked = True
        return True


class AuthenticationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = _Repository()
        self.throttle = LoginThrottleService(
            InMemoryLoginThrottleRepository(),
            policy=LoginThrottlePolicy(
                max_attempts=2,
                window_seconds=60,
                lockout_seconds=120,
            ),
        )
        self.service = AuthenticationService(
            self.repository,
            tenant_id="tenant-1",
            throttle=self.throttle,
        )

    def test_login_resolve_csrf_and_logout(self) -> None:
        login = self.service.login(
            username="Operator",
            password="correct-password",
            client_ip="192.0.2.10",
            user_agent="test",
        )
        resolved = self.service.resolve(
            login.session_token,
            request_id="request-1",
        )

        self.assertIsNotNone(resolved)
        session, context = resolved
        self.assertIsInstance(context, AuthContext)
        self.assertEqual(context.allowed_channel_ids, frozenset({7}))
        self.assertTrue(
            self.service.validate_csrf(
                session,
                cookie_token=login.csrf_token,
                header_token=login.csrf_token,
            )
        )
        self.assertFalse(
            self.service.validate_csrf(
                session,
                cookie_token=login.csrf_token,
                header_token="different",
            )
        )
        self.assertTrue(self.service.logout(login.session_token))
        self.assertIsNone(
            self.service.resolve(login.session_token, request_id="request-2")
        )

    def test_invalid_credentials_are_throttled(self) -> None:
        with self.assertRaises(InvalidCredentials):
            self.service.login(
                username="operator",
                password="wrong",
                client_ip="192.0.2.10",
            )
        with self.assertRaises(LoginThrottled):
            self.service.login(
                username="operator",
                password="wrong-again",
                client_ip="192.0.2.10",
            )


if __name__ == "__main__":
    unittest.main()
