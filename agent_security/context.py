from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping


def _string_set(values: Iterable[str], field_name: str) -> frozenset[str]:
    normalized = frozenset(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} cannot contain empty values")
    return normalized


@dataclass(frozen=True, slots=True, init=False)
class ToolExecutionContext:
    """Authenticated server-created values that are never model arguments."""

    actor_id: str
    tenant_id: str
    roles: frozenset[str]
    permissions: frozenset[str]
    allowed_channel_ids: frozenset[str]
    agent_session_id: str | None
    request_id: str | None
    client_metadata: Mapping[str, Any]

    def __init__(
        self,
        actor_id: str,
        tenant_id: str,
        roles: Iterable[str] = (),
        permissions: Iterable[str] = (),
        allowed_channels: Iterable[str] | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        client_metadata: Mapping[str, Any] | None = None,
        *,
        allowed_channel_ids: Iterable[str] | None = None,
        agent_session_id: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        actor = str(actor_id).strip()
        tenant = str(tenant_id).strip()
        if not actor or not tenant:
            raise ValueError("actor_id and tenant_id are required")
        if allowed_channels is not None and allowed_channel_ids is not None:
            if set(allowed_channels) != set(allowed_channel_ids):
                raise ValueError(
                    "allowed_channels and allowed_channel_ids cannot disagree"
                )
        if session_id is not None and agent_session_id is not None:
            if session_id != agent_session_id:
                raise ValueError(
                    "session_id and agent_session_id cannot disagree"
                )

        channels = (
            allowed_channel_ids
            if allowed_channel_ids is not None
            else allowed_channels or ()
        )
        metadata = dict(client_metadata or {})
        if client_ip is not None:
            existing_ip = metadata.get("client_ip")
            if existing_ip is not None and existing_ip != client_ip:
                raise ValueError("client_ip and client_metadata cannot disagree")
            metadata["client_ip"] = client_ip

        object.__setattr__(self, "actor_id", actor)
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "roles", _string_set(roles, "roles"))
        object.__setattr__(
            self,
            "permissions",
            _string_set(permissions, "permissions"),
        )
        object.__setattr__(
            self,
            "allowed_channel_ids",
            _string_set(channels, "allowed_channel_ids"),
        )
        object.__setattr__(
            self,
            "agent_session_id",
            agent_session_id if agent_session_id is not None else session_id,
        )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(
            self,
            "client_metadata",
            MappingProxyType(metadata),
        )

    @property
    def allowed_channels(self) -> frozenset[str]:
        return self.allowed_channel_ids

    @property
    def session_id(self) -> str | None:
        return self.agent_session_id

    @property
    def client_ip(self) -> str | None:
        value = self.client_metadata.get("client_ip")
        return str(value) if value is not None else None
