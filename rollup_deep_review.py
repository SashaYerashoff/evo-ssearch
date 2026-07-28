"""Safe scheduling and OpenAI-compatible client for L3 deep review.

This module has no UI or agent-tool dependencies.  Operators can persist the
plain schedule record through a service-owned runtime-state store, while the
Luxriot manager remains responsible for activity/alert/debt admission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


_DAY_NAMES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}
_TIME_RE = re.compile(r"^(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d)$")
_URL_USERINFO_RE = re.compile(
    r"(?P<scheme>\b[a-z][a-z0-9+.-]*://)[^\s/@]+@",
    re.IGNORECASE,
)
_QUERY_SECRET_RE = re.compile(
    r"(?P<prefix>[?&](?:password|token|access_token|api[_-]?key)=)"
    r"[^&\s\"'<>]+",
    re.IGNORECASE,
)


def _bounded_text(value: object, maximum: int = 500) -> str:
    text = " ".join(str(value or "").split())
    text = _URL_USERINFO_RE.sub(r"\g<scheme><redacted>@", text)
    text = _QUERY_SECRET_RE.sub(r"\g<prefix><redacted>", text)
    return text[: max(1, int(maximum))]


def _parse_time(value: object, field_name: str) -> Tuple[str, int]:
    text = str(value or "").strip()
    match = _TIME_RE.fullmatch(text)
    if not match:
        raise ValueError(f"{field_name} must use HH:MM local time")
    minute_of_day = int(match.group("hour")) * 60 + int(match.group("minute"))
    return text, minute_of_day


def parse_schedule_days(value: object) -> Tuple[int, ...]:
    if isinstance(value, str):
        tokens: Sequence[object] = [
            token for token in re.split(r"[\s,;]+", value) if token
        ]
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        tokens = value
    else:
        tokens = ()
    days = set()
    for token in tokens:
        normalized = str("" if token is None else token).strip().lower()
        if not normalized:
            continue
        if normalized in _DAY_NAMES:
            days.add(_DAY_NAMES[normalized])
            continue
        try:
            parsed = int(normalized)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid schedule day: {token}") from exc
        if not 0 <= parsed <= 6:
            raise ValueError("schedule days must be between 0 (Monday) and 6")
        days.add(parsed)
    if not days:
        raise ValueError("schedule days must not be empty")
    return tuple(sorted(days))


def _bounded_float(
    value: object,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum:g} and {maximum:g}"
        )
    return parsed


def _strict_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{field_name} must be boolean")


@dataclass(frozen=True)
class QuietWindowSchedule:
    """Operator-defined local-time admission window for proposal-only L3 work."""

    enabled: bool = False
    timezone: str = "UTC"
    start_local: str = "01:00"
    end_local: str = "05:00"
    days: Tuple[int, ...] = tuple(range(7))
    max_deferral_seconds: float = 86_400.0
    poll_seconds: float = 60.0
    max_activity_x: float = 1.5
    alert_lookback_seconds: float = 900.0
    max_l0_coverage_debt: float = 0.75

    def __post_init__(self) -> None:
        timezone_name = str(self.timezone or "").strip()
        if not timezone_name:
            raise ValueError("timezone must not be empty")
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {timezone_name}") from exc
        start_text, start_minute = _parse_time(self.start_local, "start_local")
        end_text, end_minute = _parse_time(self.end_local, "end_local")
        if start_minute == end_minute:
            raise ValueError("quiet window start and end must differ")
        object.__setattr__(self, "timezone", timezone_name)
        object.__setattr__(self, "start_local", start_text)
        object.__setattr__(self, "end_local", end_text)
        object.__setattr__(self, "days", parse_schedule_days(self.days))
        for field_name, minimum, maximum in (
            ("max_deferral_seconds", 60.0, 604_800.0),
            ("poll_seconds", 5.0, 3600.0),
            ("max_activity_x", 0.0, 1000.0),
            ("alert_lookback_seconds", 0.0, 86_400.0),
            ("max_l0_coverage_debt", 0.0, 10.0),
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_float(
                    getattr(self, field_name),
                    field_name,
                    minimum,
                    maximum,
                ),
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        fallback: Optional["QuietWindowSchedule"] = None,
    ) -> "QuietWindowSchedule":
        base = fallback or cls()
        return cls(
            enabled=_strict_bool(
                value.get("enabled", base.enabled),
                "enabled",
            ),
            timezone=str(value.get("timezone", base.timezone)),
            start_local=str(value.get("start_local", base.start_local)),
            end_local=str(value.get("end_local", base.end_local)),
            days=parse_schedule_days(value.get("days", base.days)),
            max_deferral_seconds=value.get(
                "max_deferral_seconds", base.max_deferral_seconds
            ),
            poll_seconds=value.get("poll_seconds", base.poll_seconds),
            max_activity_x=value.get("max_activity_x", base.max_activity_x),
            alert_lookback_seconds=value.get(
                "alert_lookback_seconds", base.alert_lookback_seconds
            ),
            max_l0_coverage_debt=value.get(
                "max_l0_coverage_debt", base.max_l0_coverage_debt
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "timezone": self.timezone,
            "start_local": self.start_local,
            "end_local": self.end_local,
            "days": list(self.days),
            "max_deferral_seconds": float(self.max_deferral_seconds),
            "poll_seconds": float(self.poll_seconds),
            "max_activity_x": float(self.max_activity_x),
            "alert_lookback_seconds": float(self.alert_lookback_seconds),
            "max_l0_coverage_debt": float(self.max_l0_coverage_debt),
        }

    def window_status(self, now: float) -> dict[str, Any]:
        local_now = datetime.fromtimestamp(float(now), tz=ZoneInfo(self.timezone))
        local_minute = local_now.hour * 60 + local_now.minute
        _start_text, start_minute = _parse_time(
            self.start_local, "start_local"
        )
        _end_text, end_minute = _parse_time(self.end_local, "end_local")
        weekday = local_now.weekday()
        if start_minute < end_minute:
            schedule_day = weekday
            inside_clock = start_minute <= local_minute < end_minute
        else:
            if local_minute >= start_minute:
                schedule_day = weekday
                inside_clock = True
            elif local_minute < end_minute:
                schedule_day = (weekday - 1) % 7
                inside_clock = True
            else:
                schedule_day = weekday
                inside_clock = False
        allowed = bool(
            self.enabled and inside_clock and schedule_day in set(self.days)
        )
        reason = (
            "disabled"
            if not self.enabled
            else "inside_quiet_window"
            if allowed
            else "outside_quiet_window"
        )
        return {
            "allowed": allowed,
            "reason": reason,
            "local_time": local_now.isoformat(timespec="seconds"),
            "local_weekday": int(weekday),
            "schedule_day": int(schedule_day),
            "timezone": self.timezone,
        }


@dataclass(frozen=True)
class DeepReviewClientConfig:
    base_url: str
    model: str
    api_key: str = ""
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 300.0
    max_tokens: int = 2048
    temperature: float = 0.1

    def __post_init__(self) -> None:
        base_url = str(self.base_url or "").strip().rstrip("/")
        model = str(self.model or "").strip()
        if not base_url:
            raise ValueError("deep-review base_url is not configured")
        if not re.match(r"^https?://", base_url, flags=re.IGNORECASE):
            raise ValueError("deep-review base_url must use http or https")
        if not model:
            raise ValueError("deep-review model is not configured")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", str(self.api_key or "").strip())
        object.__setattr__(
            self,
            "connect_timeout_seconds",
            _bounded_float(
                self.connect_timeout_seconds,
                "connect_timeout_seconds",
                0.25,
                60.0,
            ),
        )
        object.__setattr__(
            self,
            "read_timeout_seconds",
            _bounded_float(
                self.read_timeout_seconds,
                "read_timeout_seconds",
                1.0,
                3600.0,
            ),
        )
        max_tokens = int(self.max_tokens)
        if not 128 <= max_tokens <= 32768:
            raise ValueError("max_tokens must be between 128 and 32768")
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(
            self,
            "temperature",
            _bounded_float(self.temperature, "temperature", 0.0, 2.0),
        )


class OpenAICompatibleDeepReviewClient:
    """Small OpenAI-compatible text client with bounded connect/read timeouts."""

    def __init__(self, config: DeepReviewClientConfig) -> None:
        self.config = config

    def __call__(
        self,
        messages: Sequence[Mapping[str, Any]],
        _model_hint: Optional[str] = None,
        **_kwargs: Any,
    ) -> str:
        endpoint = f"{self.config.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_tokens),
            "stream": False,
        }
        response = None
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=(
                    float(self.config.connect_timeout_seconds),
                    float(self.config.read_timeout_seconds),
                ),
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            status = getattr(response, "status_code", None)
            suffix = f" (HTTP {status})" if status is not None else ""
            raise RuntimeError(
                "L3 deep-review request failed"
                + suffix
                + ": "
                + _bounded_text(exc)
            ) from exc
        choices = data.get("choices") if isinstance(data, Mapping) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, Mapping) else {}
        content = message.get("content") if isinstance(message, Mapping) else ""
        if isinstance(content, list):
            parts = [
                str(part.get("text") or "")
                for part in content
                if isinstance(part, Mapping)
                and str(part.get("type") or "") == "text"
            ]
            text = "\n".join(parts).strip()
        else:
            text = str(content or "").strip()
        if not text:
            raise RuntimeError("L3 deep-review response contained no text")
        return text


__all__ = [
    "DeepReviewClientConfig",
    "OpenAICompatibleDeepReviewClient",
    "QuietWindowSchedule",
    "parse_schedule_days",
]
