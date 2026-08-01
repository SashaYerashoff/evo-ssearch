from __future__ import annotations

from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Iterable, Mapping

from security.context import ALL_CHANNELS, AuthContext, ChannelId


class Role(StrEnum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    OPERATOR = "operator"
    VIEWER = "viewer"


class Permission(StrEnum):
    STREAMS_VIEW = "streams:view"
    DETECTIONS_VIEW = "detections:view"
    REPORTS_VIEW = "reports:view"
    AGENT_USE = "agent:use"
    PROBES_RUN = "probes:run"
    BOOKMARKS_CREATE = "bookmarks:create"
    INCIDENTS_MANAGE = "incidents:manage"
    PROBES_MANAGE = "probes:manage"
    PROMPTS_MANAGE = "prompts:manage"
    MODELS_MANAGE = "models:manage"
    CAPTURE_MANAGE = "capture:manage"
    DIAGNOSTICS_VIEW = "diagnostics:view"
    USERS_MANAGE = "users:manage"
    SETTINGS_VIEW = "settings:view"
    SETTINGS_MANAGE = "settings:manage"
    AUDIT_VIEW = "audit:view"
    DATA_EXPORT = "data:export"


_VIEWER_PERMISSIONS = frozenset(
    {
        Permission.STREAMS_VIEW,
        Permission.DETECTIONS_VIEW,
        Permission.REPORTS_VIEW,
    }
)
_OPERATOR_PERMISSIONS = _VIEWER_PERMISSIONS | frozenset(
    {
        Permission.AGENT_USE,
        Permission.PROBES_RUN,
        Permission.BOOKMARKS_CREATE,
        Permission.INCIDENTS_MANAGE,
    }
)
_ENGINEER_PERMISSIONS = _VIEWER_PERMISSIONS | frozenset(
    {
        Permission.AGENT_USE,
        Permission.PROBES_RUN,
        Permission.INCIDENTS_MANAGE,
        Permission.PROBES_MANAGE,
        Permission.PROMPTS_MANAGE,
        Permission.MODELS_MANAGE,
        Permission.CAPTURE_MANAGE,
        Permission.DIAGNOSTICS_VIEW,
        Permission.SETTINGS_VIEW,
    }
)
ALL_PERMISSIONS = frozenset(Permission)

ROLE_PERMISSIONS: Mapping[Role, frozenset[Permission]] = MappingProxyType(
    {
        Role.ADMIN: ALL_PERMISSIONS,
        Role.ENGINEER: _ENGINEER_PERMISSIONS,
        Role.OPERATOR: _OPERATOR_PERMISSIONS,
        Role.VIEWER: _VIEWER_PERMISSIONS,
    }
)


class AuthorizationDenied(PermissionError):
    """Base error for server-side authorization failures."""


class PermissionDenied(AuthorizationDenied):
    def __init__(self, permission: str | Permission) -> None:
        self.permission = _permission_value(permission)
        super().__init__(f"missing permission: {self.permission}")


class ChannelAccessDenied(AuthorizationDenied):
    def __init__(self, channel_id: ChannelId) -> None:
        self.channel_id = channel_id
        super().__init__(f"channel access denied: {channel_id}")


def _enum_value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _permission_value(permission: str | Permission) -> str:
    return _enum_value(permission).strip().lower()


def role_permissions(role: str | Role) -> frozenset[Permission]:
    try:
        normalized = Role(_enum_value(role).strip().lower())
    except ValueError as exc:
        raise ValueError(f"unknown role: {role}") from exc
    return ROLE_PERMISSIONS[normalized]


def permissions_for_roles(
    roles: Iterable[str | Role],
) -> frozenset[Permission]:
    permissions: set[Permission] = set()
    for role in roles:
        permissions.update(role_permissions(role))
    return frozenset(permissions)


def has_permission(
    context: AuthContext,
    permission: str | Permission,
) -> bool:
    return _permission_value(permission) in context.permissions


def require_permission(
    context: AuthContext,
    permission: str | Permission,
) -> None:
    if not has_permission(context, permission):
        raise PermissionDenied(permission)


def can_access_channel(context: AuthContext, channel_id: ChannelId) -> bool:
    return (
        ALL_CHANNELS in context.allowed_channel_ids
        or channel_id in context.allowed_channel_ids
    )


def require_channel_access(
    context: AuthContext,
    channel_id: ChannelId,
) -> None:
    if not can_access_channel(context, channel_id):
        raise ChannelAccessDenied(channel_id)


def authorize_channel(
    context: AuthContext,
    *,
    channel_id: ChannelId,
    permission: str | Permission,
) -> bool:
    return has_permission(context, permission) and can_access_channel(
        context, channel_id
    )


def require_channel_authorization(
    context: AuthContext,
    *,
    channel_id: ChannelId,
    permission: str | Permission,
) -> None:
    require_permission(context, permission)
    require_channel_access(context, channel_id)
