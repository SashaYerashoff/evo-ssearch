"""Durable PostgreSQL identity and session repository."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
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
    Permission.INCIDENTS_MANAGE,
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
_UNSET = object()


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
    session_id: str
    identity: IdentityRecord
    csrf_digest: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class IdentitySessionRecord:
    session_id: str
    tenant_id: str
    user_id: str
    username: str
    created_at: datetime
    last_seen_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    revoke_reason: str | None
    client_ip: str | None
    user_agent: str | None


class PostgresIdentityRepository:
    """Tenant-isolated identity persistence using the shared bounded pool."""

    _SESSION_LAST_SEEN_WRITE_INTERVAL_SECONDS = 30

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
                    id,
                    tenant_id,
                    username,
                    password_hash,
                    display_name,
                    all_channel_access
                )
                VALUES (%s, %s, %s, %s, %s, true)
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

    def list_users(
        self,
        tenant_id: uuid.UUID | str,
        *,
        actor_user_id: uuid.UUID | str,
        include_inactive: bool = True,
    ) -> tuple[IdentityRecord, ...]:
        tenant = _require_uuid(tenant_id, "tenant_id")
        actor = _require_uuid(actor_user_id, "actor_user_id")
        with self._pool.transaction(
            _transaction_context(tenant, actor),
            readonly=True,
        ) as connection:
            rows = connection.execute(
                """
                SELECT id
                FROM iam.users
                WHERE tenant_id = %s
                  AND (%s OR is_active)
                ORDER BY lower(username), id
                """,
                (tenant, bool(include_inactive)),
            ).fetchall()
            return tuple(
                self._load_identity(connection, tenant, row[0])
                for row in rows
            )

    def get_user(
        self,
        tenant_id: uuid.UUID | str,
        user_id: uuid.UUID | str,
        *,
        actor_user_id: uuid.UUID | str,
    ) -> IdentityRecord | None:
        tenant = _require_uuid(tenant_id, "tenant_id")
        target_user = _require_uuid(user_id, "user_id")
        actor = _require_uuid(actor_user_id, "actor_user_id")
        with self._pool.transaction(
            _transaction_context(tenant, actor),
            readonly=True,
        ) as connection:
            exists = connection.execute(
                """
                SELECT 1
                FROM iam.users
                WHERE tenant_id = %s AND id = %s
                """,
                (tenant, target_user),
            ).fetchone()
            if exists is None:
                return None
            return self._load_identity(connection, tenant, target_user)

    def get_user_by_username(
        self,
        tenant_id: uuid.UUID | str,
        username: str,
        *,
        actor_user_id: uuid.UUID | str = NIL_UUID,
    ) -> IdentityRecord | None:
        tenant = _require_uuid(tenant_id, "tenant_id")
        actor = _require_uuid(actor_user_id, "actor_user_id")
        normalized_username = _normalize_username(username)
        with self._pool.transaction(
            _transaction_context(tenant, actor),
            readonly=True,
        ) as connection:
            row = connection.execute(
                """
                SELECT id
                FROM iam.users
                WHERE tenant_id = %s AND lower(username) = lower(%s)
                """,
                (tenant, normalized_username),
            ).fetchone()
            if row is None:
                return None
            return self._load_identity(connection, tenant, row[0])

    def create_user(
        self,
        tenant_id: uuid.UUID | str,
        *,
        actor_user_id: uuid.UUID | str,
        username: str,
        password: str,
        roles: Iterable[str | Role],
        display_name: str | None = None,
        allowed_channel_ids: Iterable[ChannelId] = (),
        is_active: bool = True,
    ) -> IdentityRecord:
        tenant = _require_uuid(tenant_id, "tenant_id")
        actor = _require_uuid(actor_user_id, "actor_user_id")
        normalized_username = _normalize_username(username)
        _require_password_strength(password)
        normalized_display_name = _normalize_optional_text(
            display_name,
            "display_name",
            maximum=255,
        )
        normalized_roles = _normalize_roles(roles)
        normalized_channels = _normalize_channel_ids(
            allowed_channel_ids,
            roles=normalized_roles,
        )
        all_channel_access = _has_all_channel_access(normalized_channels)

        with self._pool.transaction(
            _transaction_context(tenant, actor)
        ) as connection:
            role_ids = self._seed_authorization_catalogue(connection, tenant)
            duplicate = connection.execute(
                """
                SELECT id
                FROM iam.users
                WHERE tenant_id = %s AND lower(username) = lower(%s)
                """,
                (tenant, normalized_username),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("username already exists")

            user_id = uuid.uuid4()
            connection.execute(
                """
                INSERT INTO iam.users (
                    id,
                    tenant_id,
                    username,
                    password_hash,
                    display_name,
                    is_active,
                    all_channel_access
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    tenant,
                    normalized_username,
                    self._password_hasher.hash(password),
                    normalized_display_name,
                    bool(is_active),
                    all_channel_access,
                ),
            )
            self._replace_roles(
                connection,
                tenant_id=tenant,
                user_id=user_id,
                actor_user_id=actor,
                role_ids=role_ids,
                roles=normalized_roles,
            )
            self._replace_channel_grants(
                connection,
                tenant_id=tenant,
                user_id=user_id,
                actor_user_id=actor,
                channel_ids=normalized_channels,
            )
            return self._load_identity(connection, tenant, user_id)

    def update_user(
        self,
        tenant_id: uuid.UUID | str,
        user_id: uuid.UUID | str,
        *,
        actor_user_id: uuid.UUID | str,
        display_name: str | None | object = _UNSET,
        password: str | None | object = _UNSET,
        roles: Iterable[str | Role] | object = _UNSET,
        allowed_channel_ids: Iterable[ChannelId] | object = _UNSET,
        is_active: bool | object = _UNSET,
    ) -> IdentityRecord:
        tenant = _require_uuid(tenant_id, "tenant_id")
        target_user = _require_uuid(user_id, "user_id")
        actor = _require_uuid(actor_user_id, "actor_user_id")
        normalized_roles = (
            None if roles is _UNSET else _normalize_roles(_cast_roles(roles))
        )
        normalized_channels = (
            None
            if allowed_channel_ids is _UNSET
            else _normalize_channel_ids(
                _cast_channels(allowed_channel_ids),
                roles=normalized_roles or frozenset(),
                allow_admin_all=(
                    normalized_roles is not None
                    and Role.ADMIN in normalized_roles
                ),
            )
        )
        normalized_display_name = (
            _UNSET
            if display_name is _UNSET
            else _normalize_optional_text(
                None if display_name is None else str(display_name),
                "display_name",
                maximum=255,
            )
        )
        if password is not _UNSET:
            if password is None:
                raise ValueError("password cannot be null")
            _require_password_strength(str(password))
        if (
            target_user == actor
            and is_active is not _UNSET
            and not bool(is_active)
        ):
            raise ValueError("cannot deactivate your own account")
        if (
            target_user == actor
            and normalized_roles is not None
            and Role.ADMIN not in normalized_roles
        ):
            raise ValueError("cannot remove your own admin role")

        with self._pool.transaction(
            _transaction_context(tenant, actor)
        ) as connection:
            exists = connection.execute(
                """
                SELECT 1
                FROM iam.users
                WHERE tenant_id = %s AND id = %s
                FOR UPDATE
                """,
                (tenant, target_user),
            ).fetchone()
            if exists is None:
                raise LookupError("user not found")

            updates: list[str] = ["updated_at = clock_timestamp()"]
            params: list[Any] = []
            if normalized_display_name is not _UNSET:
                updates.append("display_name = %s")
                params.append(normalized_display_name)
            if password is not _UNSET:
                updates.append("password_hash = %s")
                updates.append("password_changed_at = clock_timestamp()")
                params.append(self._password_hasher.hash(str(password)))
            if is_active is not _UNSET:
                updates.append("is_active = %s")
                params.append(bool(is_active))
            if normalized_channels is not None:
                updates.append("all_channel_access = %s")
                params.append(_has_all_channel_access(normalized_channels))
            if len(updates) > 1:
                params.extend([tenant, target_user])
                connection.execute(
                    f"""
                    UPDATE iam.users
                    SET {", ".join(updates)}
                    WHERE tenant_id = %s AND id = %s
                    """,
                    tuple(params),
                )

            if normalized_roles is not None:
                role_ids = self._seed_authorization_catalogue(
                    connection,
                    tenant,
                )
                self._replace_roles(
                    connection,
                    tenant_id=tenant,
                    user_id=target_user,
                    actor_user_id=actor,
                    role_ids=role_ids,
                    roles=normalized_roles,
                )
            if normalized_channels is not None:
                self._replace_channel_grants(
                    connection,
                    tenant_id=tenant,
                    user_id=target_user,
                    actor_user_id=actor,
                    channel_ids=normalized_channels,
                )
            if is_active is not _UNSET and not bool(is_active):
                self._revoke_user_sessions(
                    connection,
                    tenant_id=tenant,
                    user_id=target_user,
                    reason="account_deactivated",
                )
            elif (
                password is not _UNSET
                or normalized_roles is not None
                or normalized_channels is not None
            ):
                self._revoke_user_sessions(
                    connection,
                    tenant_id=tenant,
                    user_id=target_user,
                    reason="account_security_changed",
                )
            return self._load_identity(connection, tenant, target_user)

    def revoke_user_sessions(
        self,
        tenant_id: uuid.UUID | str,
        user_id: uuid.UUID | str,
        *,
        actor_user_id: uuid.UUID | str,
        reason: str,
    ) -> int:
        tenant = _require_uuid(tenant_id, "tenant_id")
        target_user = _require_uuid(user_id, "user_id")
        actor = _require_uuid(actor_user_id, "actor_user_id")
        normalized_reason = _normalize_required_text(
            reason,
            "reason",
            maximum=512,
        )
        with self._pool.transaction(
            _transaction_context(tenant, actor)
        ) as connection:
            exists = connection.execute(
                """
                SELECT 1
                FROM iam.users
                WHERE tenant_id = %s AND id = %s
                """,
                (tenant, target_user),
            ).fetchone()
            if exists is None:
                raise LookupError("user not found")
            return self._revoke_user_sessions(
                connection,
                tenant_id=tenant,
                user_id=target_user,
                reason=normalized_reason,
            )

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
                SELECT s.id, s.user_id, s.csrf_token_hash, s.expires_at
                FROM iam.sessions AS s
                JOIN iam.users AS u
                  ON u.tenant_id = s.tenant_id AND u.id = s.user_id
                WHERE s.tenant_id = %s
                  AND s.token_hash = %s
                  AND s.revoked_at IS NULL
                  AND s.expires_at > clock_timestamp()
                  AND u.is_active
                  AND (u.locked_until IS NULL OR u.locked_until <= clock_timestamp())
                """,
                (tenant, token_digest),
            ).fetchone()
            if row is None:
                return None

            session_id = _require_uuid(row[0], "session_id")
            user_id = _require_uuid(row[1], "user_id")
            self._set_actor_id(connection, user_id)
            connection.execute(
                """
                UPDATE iam.sessions
                SET last_seen_at = clock_timestamp()
                WHERE tenant_id = %s
                  AND token_hash = %s
                  AND revoked_at IS NULL
                  AND expires_at > clock_timestamp()
                  AND last_seen_at < clock_timestamp()
                      - (%s * INTERVAL '1 second')
                """,
                (
                    tenant,
                    token_digest,
                    self._SESSION_LAST_SEEN_WRITE_INTERVAL_SECONDS,
                ),
            )
            identity = self._load_identity(connection, tenant, user_id)
            if not identity.is_active:
                return None
            return SessionRecord(
                session_id=str(session_id),
                identity=identity,
                csrf_digest=bytes(row[2]).hex(),
                expires_at=row[3],
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

    def list_sessions(
        self,
        tenant_id: uuid.UUID | str,
        *,
        actor_user_id: uuid.UUID | str,
        user_id: uuid.UUID | str | None = None,
        active_only: bool = True,
    ) -> tuple[IdentitySessionRecord, ...]:
        tenant = _require_uuid(tenant_id, "tenant_id")
        actor = _require_uuid(actor_user_id, "actor_user_id")
        target_user = (
            None
            if user_id is None or str(user_id).strip() == ""
            else _require_uuid(user_id, "user_id")
        )
        with self._pool.transaction(
            _transaction_context(tenant, actor),
            readonly=True,
        ) as connection:
            rows = connection.execute(
                """
                SELECT
                    s.id,
                    s.tenant_id,
                    s.user_id,
                    u.username,
                    s.created_at,
                    s.last_seen_at,
                    s.expires_at,
                    s.revoked_at,
                    s.revoke_reason,
                    host(s.client_ip),
                    s.user_agent
                FROM iam.sessions AS s
                JOIN iam.users AS u
                  ON u.tenant_id = s.tenant_id AND u.id = s.user_id
                WHERE s.tenant_id = %s
                  AND (%s::uuid IS NULL OR s.user_id = %s::uuid)
                  AND (
                    NOT %s
                    OR (
                      s.revoked_at IS NULL
                      AND s.expires_at > clock_timestamp()
                    )
                  )
                ORDER BY s.last_seen_at DESC, s.created_at DESC
                LIMIT 1000
                """,
                (tenant, target_user, target_user, bool(active_only)),
            ).fetchall()
            return tuple(
                IdentitySessionRecord(
                    session_id=str(row[0]),
                    tenant_id=str(row[1]),
                    user_id=str(row[2]),
                    username=str(row[3]),
                    created_at=row[4],
                    last_seen_at=row[5],
                    expires_at=row[6],
                    revoked_at=row[7],
                    revoke_reason=row[8],
                    client_ip=row[9],
                    user_agent=row[10],
                )
                for row in rows
            )

    def revoke_session_by_id(
        self,
        tenant_id: uuid.UUID | str,
        session_id: uuid.UUID | str,
        *,
        actor_user_id: uuid.UUID | str,
        reason: str,
    ) -> bool:
        tenant = _require_uuid(tenant_id, "tenant_id")
        target_session = _require_uuid(session_id, "session_id")
        actor = _require_uuid(actor_user_id, "actor_user_id")
        normalized_reason = _normalize_required_text(
            reason,
            "reason",
            maximum=512,
        )
        with self._pool.transaction(
            _transaction_context(tenant, actor)
        ) as connection:
            result = connection.execute(
                """
                UPDATE iam.sessions
                SET revoked_at = clock_timestamp(),
                    revoke_reason = %s
                WHERE tenant_id = %s
                  AND id = %s
                  AND revoked_at IS NULL
                """,
                (normalized_reason, tenant, target_session),
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
            SELECT
                id,
                tenant_id,
                username,
                display_name,
                is_active,
                all_channel_access
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
        if Role.ADMIN.value in roles or bool(user[5]):
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

    def _replace_roles(
        self,
        connection: Any,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        actor_user_id: uuid.UUID,
        role_ids: dict[Role, uuid.UUID],
        roles: frozenset[Role],
    ) -> None:
        if not roles:
            raise ValueError("at least one role is required")
        connection.execute(
            """
            DELETE FROM iam.user_roles
            WHERE tenant_id = %s AND user_id = %s
            """,
            (tenant_id, user_id),
        )
        for role in sorted(roles, key=lambda item: item.value):
            connection.execute(
                """
                INSERT INTO iam.user_roles (
                    tenant_id, user_id, role_id, assigned_by
                )
                VALUES (%s, %s, %s, %s)
                """,
                (tenant_id, user_id, role_ids[role], actor_user_id),
            )

    def _replace_channel_grants(
        self,
        connection: Any,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        actor_user_id: uuid.UUID,
        channel_ids: frozenset[ChannelId],
    ) -> None:
        connection.execute(
            """
            DELETE FROM iam.user_channel_grants
            WHERE tenant_id = %s AND user_id = %s
            """,
            (tenant_id, user_id),
        )
        if _has_all_channel_access(channel_ids):
            return
        for channel_id in sorted(_numeric_channel_ids(channel_ids)):
            connection.execute(
                """
                INSERT INTO iam.user_channel_grants (
                    tenant_id,
                    user_id,
                    channel_id,
                    access_level,
                    granted_by
                )
                VALUES (%s, %s, %s, 'view', %s)
                """,
                (tenant_id, user_id, channel_id, actor_user_id),
            )

    def _revoke_user_sessions(
        self,
        connection: Any,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        reason: str,
    ) -> int:
        result = connection.execute(
            """
            UPDATE iam.sessions
            SET revoked_at = clock_timestamp(),
                revoke_reason = %s
            WHERE tenant_id = %s
              AND user_id = %s
              AND revoked_at IS NULL
            """,
            (reason, tenant_id, user_id),
        )
        return int(getattr(result, "rowcount", 0) or 0)

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


def _cast_roles(value: object) -> Iterable[str | Role]:
    if value is _UNSET or value is None:
        raise ValueError("roles are required")
    if isinstance(value, (str, Role)):
        return (value,)
    if not isinstance(value, Iterable):
        raise ValueError("roles must be a list")
    return value


def _cast_channels(value: object) -> Iterable[ChannelId]:
    if value is _UNSET or value is None:
        raise ValueError("allowed_channel_ids are required")
    if isinstance(value, (str, int)):
        return (value,)
    if not isinstance(value, Iterable):
        raise ValueError("allowed_channel_ids must be a list")
    return value


def _normalize_roles(roles: Iterable[str | Role]) -> frozenset[Role]:
    normalized: set[Role] = set()
    for role in roles:
        try:
            normalized.add(
                Role(
                    str(role.value if isinstance(role, Role) else role)
                    .strip()
                    .lower()
                )
            )
        except ValueError as exc:
            raise ValueError(f"unknown role: {role}") from exc
    if not normalized:
        raise ValueError("at least one role is required")
    return frozenset(normalized)


def _normalize_channel_ids(
    channel_ids: Iterable[ChannelId],
    *,
    roles: frozenset[Role],
    allow_admin_all: bool = False,
) -> frozenset[ChannelId]:
    normalized: set[ChannelId] = set()
    for raw_channel_id in channel_ids:
        if str(raw_channel_id).strip() == ALL_CHANNELS:
            normalized = {ALL_CHANNELS}
            continue
        if ALL_CHANNELS in normalized:
            continue
        try:
            channel_id = int(raw_channel_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("channel ids must be positive integers") from exc
        if channel_id <= 0:
            raise ValueError("channel ids must be positive integers")
        normalized.add(channel_id)
    return frozenset(normalized)


def _has_all_channel_access(channel_ids: Iterable[ChannelId]) -> bool:
    return any(str(channel_id).strip() == ALL_CHANNELS for channel_id in channel_ids)


def _numeric_channel_ids(channel_ids: Iterable[ChannelId]) -> frozenset[int]:
    numeric: set[int] = set()
    for raw_channel_id in channel_ids:
        if str(raw_channel_id).strip() == ALL_CHANNELS:
            continue
        channel_id = int(raw_channel_id)
        if channel_id > 0:
            numeric.add(channel_id)
    return frozenset(numeric)


def _require_password_strength(password: str) -> None:
    if not isinstance(password, str) or len(password) < 12 or "\x00" in password:
        raise ValueError("password must contain at least 12 safe characters")


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
