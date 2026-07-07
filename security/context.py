from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, TypeAlias


ChannelId: TypeAlias = str | int
ALL_CHANNELS = "*"


def _string_value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Server-created identity and authorization state for one request."""

    user_id: str
    tenant_id: str
    roles: frozenset[str]
    permissions: frozenset[str]
    allowed_channel_ids: frozenset[ChannelId]
    request_id: str

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.tenant_id:
            raise ValueError("tenant_id is required")
        if not self.request_id:
            raise ValueError("request_id is required")

        roles = frozenset(_string_value(role).strip().lower() for role in self.roles)
        permissions = frozenset(
            _string_value(permission).strip().lower()
            for permission in self.permissions
        )
        channels = frozenset(self.allowed_channel_ids)
        if "" in roles or "" in permissions:
            raise ValueError("roles and permissions cannot contain empty values")
        if None in channels:
            raise ValueError("allowed_channel_ids cannot contain None")

        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "permissions", permissions)
        object.__setattr__(self, "allowed_channel_ids", channels)

    @classmethod
    def from_roles(
        cls,
        *,
        user_id: str,
        tenant_id: str,
        roles: Iterable[str | Enum],
        allowed_channel_ids: Iterable[ChannelId],
        request_id: str,
    ) -> AuthContext:
        from security.permissions import permissions_for_roles

        normalized_roles = frozenset(_string_value(role) for role in roles)
        return cls(
            user_id=user_id,
            tenant_id=tenant_id,
            roles=normalized_roles,
            permissions=frozenset(
                permission.value
                for permission in permissions_for_roles(normalized_roles)
            ),
            allowed_channel_ids=frozenset(allowed_channel_ids),
            request_id=request_id,
        )
