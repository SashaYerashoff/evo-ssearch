from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from security.context import AuthContext, ChannelId


REDACTED = "[REDACTED]"
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_FORBIDDEN_EXACT_FIELDS = frozenset(
    {
        "apikey",
        "accesskey",
        "authorization",
        "clientsecret",
        "completeprompt",
        "cookie",
        "credential",
        "credentials",
        "csrftoken",
        "imagebase64",
        "imagebytes",
        "newpassword",
        "oldpassword",
        "password",
        "passwordhash",
        "passphrase",
        "prompt",
        "privatekey",
        "rawimage",
        "refreshtoken",
        "secret",
        "sessiontoken",
        "setcookie",
        "systemprompt",
        "token",
        "userprompt",
    }
)
FORBIDDEN_AUDIT_FIELDS = _FORBIDDEN_EXACT_FIELDS


def _canonical_field_name(field_name: object) -> str:
    return _NON_ALPHANUMERIC.sub("", str(field_name).lower())


def _is_forbidden_field(field_name: object) -> bool:
    canonical = _canonical_field_name(field_name)
    if canonical in _FORBIDDEN_EXACT_FIELDS:
        return True
    if canonical.endswith(
        (
            "accesskey",
            "credential",
            "credentials",
            "password",
            "passphrase",
            "privatekey",
            "secret",
            "token",
        )
    ):
        return True
    if canonical.endswith(("prompttext", "promptcontent")):
        return True
    return canonical.startswith("rawimage")


def redact_audit_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
    if not details:
        return {}

    def redact(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): REDACTED if _is_forbidden_field(key) else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, (bytes, bytearray, memoryview)):
            return REDACTED
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [redact(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return [redact(item) for item in sorted(value, key=repr)]
        return value

    return redact(details)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AuditEvent:
    timestamp: datetime
    request_id: str | None
    actor_user_id: str | None
    actor_roles: tuple[str, ...]
    tenant_id: str | None
    source_ip: str
    action: str
    target_type: str
    target_id: str | None
    channel_id: ChannelId | None
    result: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "actor_user_id": self.actor_user_id,
            "actor_roles": list(self.actor_roles),
            "tenant_id": self.tenant_id,
            "source_ip": self.source_ip,
            "action": self.action,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "channel_id": self.channel_id,
            "result": self.result,
            "details": _thaw(self.details),
        }


class AuditEventBuilder:
    def __init__(
        self,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._clock = clock

    def build(
        self,
        *,
        context: AuthContext | None,
        source_ip: str,
        action: str,
        target_type: str,
        result: str,
        target_id: str | None = None,
        channel_id: ChannelId | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        timestamp = self._clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("audit timestamps must be timezone-aware")
        if not source_ip:
            raise ValueError("source_ip is required")
        if not action:
            raise ValueError("action is required")
        if not target_type:
            raise ValueError("target_type is required")
        if not result:
            raise ValueError("result is required")

        return AuditEvent(
            timestamp=timestamp,
            request_id=context.request_id if context else None,
            actor_user_id=context.user_id if context else None,
            actor_roles=tuple(sorted(context.roles)) if context else (),
            tenant_id=context.tenant_id if context else None,
            source_ip=source_ip,
            action=action,
            target_type=target_type,
            target_id=target_id,
            channel_id=channel_id,
            result=result,
            details=_freeze(redact_audit_details(details)),
        )


def build_audit_event(
    *,
    context: AuthContext | None,
    source_ip: str,
    action: str,
    target_type: str,
    result: str,
    target_id: str | None = None,
    channel_id: ChannelId | None = None,
    details: Mapping[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> AuditEvent:
    builder = AuditEventBuilder(
        clock=(lambda: timestamp) if timestamp is not None else (
            lambda: datetime.now(timezone.utc)
        )
    )
    return builder.build(
        context=context,
        source_ip=source_ip,
        action=action,
        target_type=target_type,
        result=result,
        target_id=target_id,
        channel_id=channel_id,
        details=details,
    )
