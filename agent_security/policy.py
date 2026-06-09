from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Iterable, Mapping


class ToolRisk(str, Enum):
    READ = "read"
    WRITE = "write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


ChannelScopeResolver = Callable[[Mapping[str, Any]], Iterable[str]]
ArgumentNormalizer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


_SINGULAR_CHANNEL_ARGUMENTS = ("channel_id", "channel")
_LIST_CHANNEL_ARGUMENTS = ("channel_ids", "channels", "channel_scope")


def resolve_channel_arguments(arguments: Mapping[str, Any]) -> frozenset[str]:
    """Resolve common singular and list channel arguments."""

    resolved: set[str] = set()
    for key in _SINGULAR_CHANNEL_ARGUMENTS:
        if key not in arguments or arguments[key] is None:
            continue
        value = arguments[key]
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            raise ValueError(f"{key} must contain one channel ID")
        channel_id = str(value).strip()
        if not channel_id:
            raise ValueError(f"{key} cannot be empty")
        resolved.add(channel_id)

    for key in _LIST_CHANNEL_ARGUMENTS:
        if key not in arguments or arguments[key] is None:
            continue
        value = arguments[key]
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(
            value, Iterable
        ):
            raise ValueError(f"{key} must contain a list of channel IDs")
        for item in value:
            channel_id = str(item).strip()
            if not channel_id:
                raise ValueError(f"{key} cannot contain empty channel IDs")
            resolved.add(channel_id)
    return frozenset(resolved)


@dataclass(frozen=True, slots=True)
class RateLimit:
    max_calls: int
    window_seconds: float
    scope: str = "actor"

    def __post_init__(self) -> None:
        if self.max_calls <= 0 or self.window_seconds <= 0:
            raise ValueError("rate limits must be positive")
        if self.scope not in {"actor", "tenant", "session"}:
            raise ValueError("rate limit scope must be actor, tenant, or session")


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    required_permission: str
    risk: ToolRisk | str = ToolRisk.READ
    approval_required: bool = False
    approval_permission: str | None = None
    allowed_arguments: frozenset[str] = frozenset()
    required_arguments: frozenset[str] = frozenset()
    channel_resolver: ChannelScopeResolver | None = resolve_channel_arguments
    channel_required: bool = False
    max_rows: int | None = None
    default_rows: int | None = None
    row_limit_arguments: tuple[str, ...] = ("limit", "row_limit", "max_rows")
    max_time_window: timedelta | int | float | None = None
    default_time_window: timedelta | int | float | None = None
    time_window_arguments: tuple[tuple[str, str], ...] = (
        ("start_time", "end_time"),
        ("since", "until"),
        ("from", "to"),
    )
    time_window_object_arguments: tuple[str, ...] = ("time_window", "window")
    duration_arguments: tuple[str, ...] = (
        "time_window_seconds",
        "window_seconds",
    )
    max_output_bytes: int = 65_536
    max_output_items: int = 1_000
    max_output_string_chars: int = 16_384
    redact_fields: frozenset[str] = frozenset()
    timeout_seconds: float | None = None
    rate_limit: RateLimit | None = None
    output_data_classification: str = "internal"
    audit_event_type: str = "agent_tool"
    plan_ttl_seconds: float = 300.0
    approval_ttl_seconds: float = 120.0
    argument_normalizer: ArgumentNormalizer | None = None

    def __post_init__(self) -> None:
        permission = self.required_permission.strip()
        if not permission:
            raise ValueError("required_permission must be explicit")
        object.__setattr__(self, "required_permission", permission)
        object.__setattr__(self, "risk", ToolRisk(self.risk))
        object.__setattr__(
            self, "allowed_arguments", frozenset(self.allowed_arguments)
        )
        object.__setattr__(
            self, "required_arguments", frozenset(self.required_arguments)
        )
        object.__setattr__(self, "redact_fields", frozenset(self.redact_fields))

        if not self.required_arguments.issubset(self.allowed_arguments):
            raise ValueError("required_arguments must be allowed_arguments")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive")
        if self.default_rows is not None and self.default_rows <= 0:
            raise ValueError("default_rows must be positive")
        if (
            self.max_rows is not None
            and self.default_rows is not None
            and self.default_rows > self.max_rows
        ):
            raise ValueError("default_rows cannot exceed max_rows")

        max_window = _as_timedelta(self.max_time_window, "max_time_window")
        default_window = _as_timedelta(
            self.default_time_window, "default_time_window"
        )
        if max_window is not None and default_window is not None:
            if default_window > max_window:
                raise ValueError(
                    "default_time_window cannot exceed max_time_window"
                )
        object.__setattr__(self, "max_time_window", max_window)
        object.__setattr__(self, "default_time_window", default_window)

        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if self.max_output_items <= 0:
            raise ValueError("max_output_items must be positive")
        if self.max_output_string_chars <= 0:
            raise ValueError("max_output_string_chars must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.plan_ttl_seconds <= 0 or self.approval_ttl_seconds <= 0:
            raise ValueError("plan and approval TTLs must be positive")


def _as_timedelta(
    value: timedelta | int | float | None, field_name: str
) -> timedelta | None:
    if value is None:
        return None
    result = value if isinstance(value, timedelta) else timedelta(seconds=value)
    if result.total_seconds() <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result
