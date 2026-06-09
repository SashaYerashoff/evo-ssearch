"""Durable PostgreSQL identity and session repository."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from eva_db import PsycopgPool, TransactionContext
from security.context import ALL_CHANNELS, ChannelId
from security.passwords import PasswordHasher, create_password_hasher
from security.permissions import Permission, Role, ROLE_PERMISSIONS
from security.tokens import digest_session_token


NIL_UUID = uuid.UUID(int=0)
_ROLE_NAMESPACE = uuid.UUID("4a994294-10ab-43fe-b88c-80b160dedcd2")

_ROLE_DESCRIPTIONS = {
    Role.ADMIN: "Full tenant administration",
    Role.ENGINEER: "System and inference configuration",
    Role.OPERATOR: "Operational use of EVA AI",
    Role.VIEWER: "Read-only operational access",
}

_WRITE_PERMISSIONS = {
    Permission.BOOKMARKS_CREATE,
    Permission.CAPTURE_MANAGE,
    Permission.MODELS_MANAGE,
    Permission.PROBES_MANAGE,
    Permission.PROMPTS_MANAGE,
    Permission.SETTINGS_MANAGE,
    Permission.USERS_MANAGE,
}
_EXTERNAL_SIDE_EFFECT_PERMISSIONS = {
    Permission.DATA_EXPORT,
    Permission.PROBES_RUN,
}


class IdentityBootstrapConflict(ValueError):
    """Raised when a tenant was already bootstrapped for another administrator."""


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    user_id: str
    tenant_id: str
    username: str
    display_name: str | None
    roles: frozenset[str]
    permissions: frozenset[str]
    allowed_channel_ids: frozenset[ChannelId]
    is_active: bool


@dataclass(frozen=True, slots=True)
class SessionRecord:
    identity: IdentityRecord
    csrf_digest: str
    expires_at: datetime


class PostgresIdentityRepository:
    """Tenant-isolated identity persistence using the shared bounded pool."""

    def __init__(
        self,
        pool: PsycopgPool,
        password_hasher: PasswordHasher | None = None,
    ) -> None:
        self._pool = pool
        self._password_hasher = password_hasher or create_password_hasher()

    def bootstrap_admin(
        self,
        tenant_id: uuid.UUID,
        username: str,
        password: str,
        display_name: str | None = None,
    ) -> IdentityRecord:
        tenant = _require_uuid(tenant_id, "tenant_id")
        normalized_username = _normalize_username(username)
        if not password:
            raise ValueError("password is required")
        normalized_display_name = _normalize_optional_text(
            display_name,
            "display_name",
            maximum=255,
        )
        context = _transaction_context(tenant, NIL_UUID)

        with self._pool.transaction(context) as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (str(tenant),),
            )
            role_ids = self._seed_authorization_catalogue(connection, tenant)
            existing_admins = connection.execute(
                """
                SELECT u.id, u.username
                FROM iam.users AS u
                JOIN iam.user_roles AS ur
                  ON ur.tenant_id = u.tenant_id AND ur.user_id = u.id
                JOIN iam.roles AS r
                  ON r.tenant_id = ur.tenant_id AND r.id = ur.role_id
                WHERE u.tenant_id = %s AND r.name = %s
                ORDER BY lower(u.username), u.id
                """,
                (tenant, Role.ADMIN.value),
            ).fetchall()
            if existing_admins:
                matching = [
                    row
                    for row in existing_admins
                    if str(row[1]).casefold() == normalized_username.casefold()
                ]
                if len(existing_admins) != 1 or len(matching) != 1:
                    raise IdentityBootstrapConflict(
                        "tenant already has a different bootstrap administrator"
                    )
                return self._load_identity(connection, tenant, matching[0][0])

            username_owner = connection.execute(
                """
                SELECT id
                FROM iam.users
                WHERE tenant_id = %s AND lower(username) = lower(%s)
                """,
                (tenant, normalized_username),
            ).fetchone()
            if username_owner is not None:
                raise IdentityBootstrapConflict(
                    "bootstrap username already belongs to a non-admin user"
                )

            user_id = uuid.uuid4()
            password_hash = self._password_hasher.hash(password)
            connection.execute(
                """
                INSERT INTO iam.users (
                    id, tenant_id, username, password_hash, display_name
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    tenant,
                    normalized_username,
                    password_hash,
                    normalized_display_name,
                ),
            )
            connection.execute(
                """
                INSERT INTO iam.user_roles (
                    tenant_id, user_id, role_id, assigned_by
                )
                VALUES (%s, %s, %s, NULL)
                """,
                (tenant, user_id, role_ids[Role.ADMIN]),
            )
            self._set_actor_id(connection, user_id)
            return self._load_identity(connection, tenant, user_id)

    def authenticate(
        self,
        tenant_id: uuid.UUID,
        username: str,
        password: str,
    ) -> IdentityRecord | None:
        tenant = _require_uuid(tenant_id, "tenant_id")
        normalized_username = _normalize_username(username)
        if not password:
            return None

        with self._pool.transaction(
            _transaction_context(tenant, NIL_UUID)
        ) as connection:
            row = connection.execute(
                """
                SELECT id, password_hash
                FROM iam.users
                WHERE tenant_id = %s
                  AND lower(username) = lower(%s)
                  AND is_active
                  AND (locked_until IS NULL OR locked_until <= clock_timestamp())
                FOR UPDATE
                """,
                (tenant, normalized_username),
            ).fetchone()
            if row is None or not self._password_hasher.verify(row[1], password):
                return None

            user_id = _require_uuid(row[0], "user_id")
            self._set_actor_id(connection, user_id)
            password_hash = row[1]
            replacement_hash = (
                self._password_hasher.hash(password)
                if self._password_hasher.needs_rehash(password_hash)
                else password_hash
            )
            connection.execute(
                """
                UPDATE iam.users
                SET last_login_at = clock_timestamp(),
                    failed_login_count = 0,
                    password_hash = %s,
                    password_changed_at = CASE
                        WHEN password_hash = %s THEN password_changed_at
                        ELSE clock_timestamp()
                    END,
                    updated_at = clock_timestamp()
                WHERE tenant_id = %s AND id = %s AND is_active
                """,
                (replacement_hash, replacement_hash, tenant, user_id),
            )
            return self._load_identity(connection, tenant, user_id)

    def create_session(
        self,
        identity: IdentityRecord,
        token: str,
        csrf_token: str,
        expires_at: datetime,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> str:
        if not identity.is_active:
            raise ValueError("cannot create a session for an inactive identity")
        expiry = _require_future_datetime(expires_at)
        normalized_user_agent = _normalize_optional_text(
            user_agent,
            "user_agent",
            maximum=2048,
        )
        session_id = uuid.uuid4()
        token_digest = _sha256_digest(token, "token")
        csrf_digest = _sha256_digest(csrf_token, "csrf_token")

        with self._pool.transaction(
            _transaction_context(identity.tenant_id, identity.user_id)
        ) as connection:
            connection.execute(
                """
                INSERT INTO iam.sessions (
                    id,
                    tenant_id,
                    user_id,
                    token_hash,
                    csrf_token_hash,
                    expires_at,
                    client_ip,
                    user_agent
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    identity.tenant_id,
                    identity.user_id,
                    token_digest,
                    csrf_digest,
                    expiry,
                    client_ip,
                    normalized_user_agent,
                ),
            )
        return str(session_id)

    def resolve_session(
        self,
        tenant_id: uuid.UUID,
        token: str,
    ) -> SessionRecord | None:
        tenant = _require_uuid(tenant_id, "tenant_id")
        token_digest = _sha256_digest(token, "token")

        with self._pool.transaction(
            _transaction_context(tenant, NIL_UUID)
        ) as connection:
            row = connection.execute(
                """
                SELECT s.user_id, s.csrf_token_hash, s.expires_at
                FROM iam.sessions AS s
                JOIN iam.users AS u
                  ON u.tenant_id = s.tenant_id AND u.id = s.user_id
                WHERE s.tenant_id = %s
                  AND s.token_hash = %s
                  AND s.revoked_at IS NULL
                  AND s.expires_at > clock_timestamp()
                  AND u.is_active
                  AND (u.locked_until IS NULL OR u.locked_until <= clock_timestamp())
                FOR UPDATE OF s
                """,
                (tenant, token_digest),
            ).fetchone()
            if row is None:
                return None

            user_id = _require_uuid(row[0], "user_id")
            self._set_actor_id(connection, user_id)
            connection.execute(
                """
                UPDATE iam.sessions
                SET last_seen_at = clock_timestamp()
                WHERE tenant_id = %s
                  AND token_hash = %s
                  AND revoked_at IS NULL
                  AND expires_at > clock_timestamp()
                """,
                (tenant, token_digest),
            )
            identity = self._load_identity(connection, tenant, user_id)
            if not identity.is_active:
                return None
            return SessionRecord(
                identity=identity,
                csrf_digest=bytes(row[1]).hex(),
                expires_at=row[2],
            )

    def revoke_session(
        self,
        tenant_id: uuid.UUID,
        token: str,
        reason: str,
    ) -> bool:
        tenant = _require_uuid(tenant_id, "tenant_id")
        token_digest = _sha256_digest(token, "token")
        normalized_reason = _normalize_required_text(
            reason,
            "reason",
            maximum=512,
        )

        with self._pool.transaction(
            _transaction_context(tenant, NIL_UUID)
        ) as connection:
            row = connection.execute(
                """
                SELECT user_id
                FROM iam.sessions
                WHERE tenant_id = %s
                  AND token_hash = %s
                  AND revoked_at IS NULL
                FOR UPDATE
                """,
                (tenant, token_digest),
            ).fetchone()
            if row is None:
                return False

            user_id = _require_uuid(row[0], "user_id")
            self._set_actor_id(connection, user_id)
            result = connection.execute(
                """
                UPDATE iam.sessions
                SET revoked_at = clock_timestamp(), revoke_reason = %s
                WHERE tenant_id = %s
                  AND token_hash = %s
                  AND revoked_at IS NULL
                """,
                (normalized_reason, tenant, token_digest),
            )
            return bool(result.rowcount)

    def _seed_authorization_catalogue(
        self,
        connection: Any,
        tenant_id: uuid.UUID,
    ) -> dict[Role, uuid.UUID]:
        for permission in Permission:
            connection.execute(
                """
                INSERT INTO iam.permissions (key, description, risk)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE
                SET description = EXCLUDED.description,
                    risk = EXCLUDED.risk
                """,
                (
                    permission.value,
                    permission.value.replace(":", " "),
                    _permission_risk(permission),
                ),
            )

        role_ids: dict[Role, uuid.UUID] = {}
        for role in Role:
            proposed_id = uuid.uuid5(
                _ROLE_NAMESPACE,
                f"{tenant_id}:{role.value}",
            )
            row = connection.execute(
                """
                INSERT INTO iam.roles (
                    id, tenant_id, name, description, is_system
                )
                VALUES (%s, %s, %s, %s, true)
                ON CONFLICT (tenant_id, name) DO UPDATE
                SET description = EXCLUDED.description,
                    is_system = true,
                    updated_at = clock_timestamp()
                RETURNING id
                """,
                (
                    proposed_id,
                    tenant_id,
                    role.value,
                    _ROLE_DESCRIPTIONS[role],
                ),
            ).fetchone()
            role_id = _require_uuid(row[0], "role_id")
            role_ids[role] = role_id
            for permission in ROLE_PERMISSIONS[role]:
                connection.execute(
                    """
                    INSERT INTO iam.role_permissions (
                        tenant_id, role_id, permission_key, assigned_by
                    )
                    VALUES (%s, %s, %s, NULL)
                    ON CONFLICT (tenant_id, role_id, permission_key) DO NOTHING
                    """,
                    (tenant_id, role_id, permission.value),
                )
        return role_ids

    def _load_identity(
        self,
        connection: Any,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | str,
    ) -> IdentityRecord:
        user = connection.execute(
            """
            SELECT id, tenant_id, username, display_name, is_active
            FROM iam.users
            WHERE tenant_id = %s AND id = %s
            """,
            (tenant_id, user_id),
        ).fetchone()
        if user is None:
            raise LookupError("identity no longer exists")

        roles = frozenset(
            str(row[0])
            for row in connection.execute(
                """
                SELECT r.name
                FROM iam.user_roles AS ur
                JOIN iam.roles AS r
                  ON r.tenant_id = ur.tenant_id AND r.id = ur.role_id
                WHERE ur.tenant_id = %s AND ur.user_id = %s
                ORDER BY r.name
                """,
                (tenant_id, user_id),
            ).fetchall()
        )
        permissions = frozenset(
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT rp.permission_key
                FROM iam.user_roles AS ur
                JOIN iam.role_permissions AS rp
                  ON rp.tenant_id = ur.tenant_id AND rp.role_id = ur.role_id
                WHERE ur.tenant_id = %s AND ur.user_id = %s
                ORDER BY rp.permission_key
                """,
                (tenant_id, user_id),
            ).fetchall()
        )
        if Role.ADMIN.value in roles:
            allowed_channel_ids: frozenset[ChannelId] = frozenset({ALL_CHANNELS})
        else:
            allowed_channel_ids = frozenset(
                int(row[0])
                for row in connection.execute(
                    """
                    SELECT channel_id
                    FROM iam.user_channel_grants
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND (expires_at IS NULL OR expires_at > clock_timestamp())
                    ORDER BY channel_id
                    """,
                    (tenant_id, user_id),
                ).fetchall()
            )
        return IdentityRecord(
            user_id=str(_require_uuid(user[0], "user_id")),
            tenant_id=str(_require_uuid(user[1], "tenant_id")),
            username=str(user[2]),
            display_name=None if user[3] is None else str(user[3]),
            roles=roles,
            permissions=permissions,
            allowed_channel_ids=allowed_channel_ids,
            is_active=bool(user[4]),
        )

    @staticmethod
    def _set_actor_id(connection: Any, actor_id: uuid.UUID) -> None:
        connection.execute(
            "SELECT set_config('eva.actor_id', %s, true)",
            (str(actor_id),),
        )


def _transaction_context(
    tenant_id: uuid.UUID,
    actor_id: uuid.UUID,
) -> TransactionContext:
    return TransactionContext(tenant_id=tenant_id, actor_id=actor_id)


def _require_uuid(value: uuid.UUID | str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _normalize_username(username: str) -> str:
    return _normalize_required_text(username, "username", maximum=255)


def _normalize_required_text(value: str, field_name: str, *, maximum: int) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if "\x00" in normalized or len(normalized) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} safe characters")
    return normalized


def _normalize_optional_text(
    value: str | None,
    field_name: str,
    *,
    maximum: int,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if "\x00" in normalized or len(normalized) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} safe characters")
    return normalized


def _sha256_digest(value: str, field_name: str) -> bytes:
    if not value:
        raise ValueError(f"{field_name} is required")
    return bytes.fromhex(digest_session_token(value))


def _require_future_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("expires_at must be timezone-aware")
    if value <= datetime.now(timezone.utc):
        raise ValueError("expires_at must be in the future")
    return value


def _permission_risk(permission: Permission) -> str:
    if permission in _EXTERNAL_SIDE_EFFECT_PERMISSIONS:
        return "external_side_effect"
    if permission in _WRITE_PERMISSIONS:
        return "write"
    return "read"
