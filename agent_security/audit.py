from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ToolAuditEvent:
    timestamp: datetime
    phase: str
    operation: str
    tool_name: str
    actor_id: str
    tenant_id: str
    request_id: str | None
    session_id: str | None
    risk: str | None = None
    required_permission: str | None = None
    arguments_hash: str | None = None
    code: str | None = None
    duration_ms: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        return self.phase

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "operation": self.operation,
            "tool_name": self.tool_name,
            "actor_id": self.actor_id,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "risk": self.risk,
            "required_permission": self.required_permission,
            "arguments_hash": self.arguments_hash,
            "code": self.code,
            "duration_ms": self.duration_ms,
            "details": dict(self.details),
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]
