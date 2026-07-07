from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from security.context import AuthContext, ChannelId
from security.throttling import (
    InMemoryLoginThrottleRepository,
    LoginThrottleService,
)
from security.tokens import (
    generate_csrf_token,
    generate_session_token,
    verify_session_token_digest,
)


class AuthenticationError(RuntimeError):
    pass


class InvalidCredentials(AuthenticationError):
    pass


class LoginThrottled(AuthenticationError):
    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__("login is temporarily throttled")


@runtime_checkable
class IdentityRecord(Protocol):
    user_id: str
    tenant_id: str
    username: str
    display_name: str | None
    roles: frozenset[str]
    permissions: frozenset[str]
    allowed_channel_ids: frozenset[ChannelId]
    is_active: bool


@runtime_checkable
class SessionRecord(Protocol):
    identity: IdentityRecord
    csrf_digest: str
    expires_at: datetime


@runtime_checkable
class IdentityRepository(Protocol):
    def authenticate(
        self,
        tenant_id: str,
        username: str,
        password: str,
    ) -> IdentityRecord | None:
        ...

    def create_session(
        self,
        identity: IdentityRecord,
        token: str,
        csrf_token: str,
        expires_at: datetime,
        *,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> str:
        ...

    def resolve_session(
        self,
        tenant_id: str,
        token: str,
    ) -> SessionRecord | None:
        ...

    def revoke_session(
        self,
        tenant_id: str,
        token: str,
        reason: str,
    ) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class LoginSession:
    identity: IdentityRecord
    session_id: str
    session_token: str
    csrf_token: str
    expires_at: datetime


class AuthenticationService:
    def __init__(
        self,
        repository: IdentityRepository,
        *,
        tenant_id: str,
        session_ttl: timedelta = timedelta(hours=12),
        throttle: LoginThrottleService | None = None,
    ) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if session_ttl <= timedelta(0):
            raise ValueError("session_ttl must be positive")
        self._repository = repository
        self._tenant_id = tenant_id
        self._session_ttl = session_ttl
        self._throttle = throttle or LoginThrottleService(
            InMemoryLoginThrottleRepository()
        )

    def login(
        self,
        *,
        username: str,
        password: str,
        client_ip: str,
        user_agent: str | None = None,
    ) -> LoginSession:
        normalized_username = username.strip()
        throttle_key = self._throttle_key(normalized_username, client_ip)
        decision = self._throttle.check(throttle_key)
        if not decision.allowed:
            raise LoginThrottled(decision.retry_after_seconds)

        identity = self._repository.authenticate(
            self._tenant_id,
            normalized_username,
            password,
        )
        if identity is None:
            failed = self._throttle.record_failure(throttle_key)
            if not failed.allowed:
                raise LoginThrottled(failed.retry_after_seconds)
            raise InvalidCredentials("invalid username or password")

        self._throttle.record_success(throttle_key)
        session_token = generate_session_token()
        csrf_token = generate_csrf_token()
        expires_at = datetime.now(timezone.utc) + self._session_ttl
        session_id = self._repository.create_session(
            identity,
            session_token,
            csrf_token,
            expires_at,
            client_ip=client_ip,
            user_agent=user_agent,
        )
        return LoginSession(
            identity=identity,
            session_id=session_id,
            session_token=session_token,
            csrf_token=csrf_token,
            expires_at=expires_at,
        )

    def resolve(
        self,
        session_token: str,
        *,
        request_id: str,
    ) -> tuple[SessionRecord, AuthContext] | None:
        if not session_token:
            return None
        session = self._repository.resolve_session(
            self._tenant_id,
            session_token,
        )
        if session is None:
            return None
        identity = session.identity
        context = AuthContext(
            user_id=identity.user_id,
            tenant_id=identity.tenant_id,
            roles=identity.roles,
            permissions=identity.permissions,
            allowed_channel_ids=identity.allowed_channel_ids,
            request_id=request_id,
        )
        return session, context

    def validate_csrf(
        self,
        session: SessionRecord,
        *,
        cookie_token: str,
        header_token: str,
    ) -> bool:
        if not cookie_token or cookie_token != header_token:
            return False
        return verify_session_token_digest(
            header_token,
            session.csrf_digest,
        )

    def logout(self, session_token: str, *, reason: str = "logout") -> bool:
        if not session_token:
            return False
        return self._repository.revoke_session(
            self._tenant_id,
            session_token,
            reason,
        )

    def _throttle_key(self, username: str, client_ip: str) -> str:
        return f"{self._tenant_id}:{username.lower()}:{client_ip}"
