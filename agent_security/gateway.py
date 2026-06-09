from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable, Mapping, Protocol

from .approvals import (
    ApprovalStore,
    InMemoryPlanApprovalStore,
    PlanStore,
    ToolApproval,
    ToolPlan,
    hash_arguments,
    thaw_value,
    utc_now,
)
from .audit import ToolAuditEvent
from .context import ToolExecutionContext
from .errors import (
    ApprovalArgumentMismatchError,
    ApprovalConsumedError,
    ApprovalError,
    ApprovalExpiredError,
    ApprovalRequiredError,
    AuditUnavailableError,
    AuthorizationError,
    ChannelAccessDeniedError,
    ContextInjectionError,
    InvalidToolArgumentsError,
    PermissionDeniedError,
    PlanExpiredError,
    RateLimitExceededError,
    ToolGatewayError,
    ToolTimeoutError,
)
from .output import sanitize_output
from .policy import RateLimit, ToolPolicy
from .registry import ToolDefinition, ToolRegistry


AuditCallback = Callable[[ToolAuditEvent], None]
Clock = Callable[[], datetime]

_CONTEXT_ARGUMENT_NAMES = frozenset(
    {
        "actor_id",
        "user_id",
        "tenant_id",
        "roles",
        "permissions",
        "allowed_channels",
        "allowed_channel_ids",
        "session_id",
        "agent_session_id",
        "request_id",
        "client_metadata",
        "client_ip",
        "ip_address",
    }
)


class RateLimiter(Protocol):
    def check_and_record(
        self,
        tool_name: str,
        context: ToolExecutionContext,
        limit: RateLimit,
    ) -> bool: ...


class InMemoryRateLimiter:
    def __init__(self, clock: Clock = utc_now) -> None:
        self._clock = clock
        self._calls: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def check_and_record(
        self,
        tool_name: str,
        context: ToolExecutionContext,
        limit: RateLimit,
    ) -> bool:
        scope_id = {
            "actor": context.actor_id,
            "tenant": context.tenant_id,
            "session": context.session_id or context.actor_id,
        }[limit.scope]
        key = (tool_name, scope_id)
        now = self._clock().timestamp()
        cutoff = now - limit.window_seconds
        with self._lock:
            calls = self._calls[key]
            while calls and calls[0] <= cutoff:
                calls.popleft()
            if len(calls) >= limit.max_calls:
                return False
            calls.append(now)
            return True


class ToolGateway:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        audit_callback: AuditCallback | None = None,
        plan_store: PlanStore | None = None,
        approval_store: ApprovalStore | None = None,
        clock: Clock = utc_now,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.registry = registry
        self._audit_callback = audit_callback
        self._audit_events: list[ToolAuditEvent] = []
        self._audit_lock = RLock()
        self._clock = clock
        if plan_store is None and approval_store is None:
            memory_store = InMemoryPlanApprovalStore(clock)
            plan_store = memory_store
            approval_store = memory_store
        elif plan_store is None or approval_store is None:
            raise ValueError(
                "plan_store and approval_store must be provided together"
            )
        self._plan_store = plan_store
        self._approval_store = approval_store
        self._rate_limiter = rate_limiter or InMemoryRateLimiter(clock)
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = RLock()

    def available_tools(
        self, context: ToolExecutionContext
    ) -> tuple[ToolDefinition, ...]:
        self._require_context(context)
        return tuple(
            definition
            for definition in self.registry.definitions()
            if definition.policy.required_permission in context.permissions
        )

    @property
    def audit_events(self) -> tuple[ToolAuditEvent, ...]:
        with self._audit_lock:
            return tuple(self._audit_events)

    def create_plan(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        context: ToolExecutionContext,
        *,
        ttl_seconds: float | None = None,
    ) -> ToolPlan:
        started = time.monotonic()
        definition: ToolDefinition | None = None
        arguments_hash: str | None = None
        try:
            self._require_context(context)
            definition = self.registry.get(tool_name)
            if not definition.policy.approval_required:
                raise ApprovalError("tool does not require an approval plan")
            normalized = self._normalize_arguments(
                arguments, definition.policy, now=self._clock()
            )
            arguments_hash = hash_arguments(normalized)
            self._authorize(definition, normalized, context)
            self._emit(
                "allow",
                "plan",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            ttl = self._bounded_ttl(
                ttl_seconds, definition.policy.plan_ttl_seconds
            )
            plan = self._plan_store.create_plan(
                action=tool_name,
                normalized_arguments=normalized,
                arguments_hash=arguments_hash,
                context=context,
                required_permission=definition.policy.required_permission,
                expires_at=self._clock() + timedelta(seconds=ttl),
            )
            self._emit(
                "result",
                "plan",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
                details={"plan_id": plan.plan_id},
            )
            return plan
        except ToolGatewayError as exc:
            self._emit_failure(
                exc,
                "plan",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            raise
        except Exception as exc:
            self._emit_exception(
                exc,
                "plan",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            raise

    plan = create_plan

    def approve(
        self,
        plan_id: str,
        context: ToolExecutionContext,
        *,
        ttl_seconds: float | None = None,
    ) -> ToolApproval:
        started = time.monotonic()
        definition: ToolDefinition | None = None
        tool_name = "<unknown>"
        arguments_hash: str | None = None
        try:
            self._require_context(context)
            plan = self._plan_store.get_plan(plan_id)
            tool_name = plan.action
            arguments_hash = plan.arguments_hash
            definition = self.registry.get(plan.action)
            now = self._clock()
            if plan.expires_at <= now:
                raise PlanExpiredError("plan has expired")
            if (
                plan.actor_id != context.actor_id
                or plan.tenant_id != context.tenant_id
            ):
                raise ApprovalError("plan is not bound to this actor and tenant")
            if plan.required_permission != definition.policy.required_permission:
                raise ApprovalError("tool policy changed; create a new plan")
            approval_permission = (
                definition.policy.approval_permission
                or definition.policy.required_permission
            )
            if approval_permission not in context.permissions:
                raise PermissionDeniedError(
                    f"missing permission: {approval_permission}"
                )
            normalized = thaw_value(plan.normalized_arguments)
            self._authorize(definition, normalized, context)
            self._emit(
                "allow",
                "approve",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            ttl = self._bounded_ttl(
                ttl_seconds, definition.policy.approval_ttl_seconds
            )
            approval = self._approval_store.create_approval(
                plan=plan,
                approved_by=context,
                expires_at=now + timedelta(seconds=ttl),
            )
            self._emit(
                "result",
                "approve",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
                details={"plan_id": plan.plan_id},
            )
            return approval
        except ToolGatewayError as exc:
            self._emit_failure(
                exc,
                "approve",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            raise
        except Exception as exc:
            self._emit_exception(
                exc,
                "approve",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            raise

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        context: ToolExecutionContext,
        *,
        approval_id: str | None = None,
    ) -> Any:
        started = time.monotonic()
        definition: ToolDefinition | None = None
        arguments_hash: str | None = None
        allowed = False
        try:
            self._require_context(context)
            definition = self.registry.get(tool_name)
            policy = definition.policy
            self._reject_context_injection(arguments)

            if policy.approval_required:
                if not approval_id:
                    normalized = self._normalize_arguments(
                        arguments, policy, now=self._clock()
                    )
                    arguments_hash = hash_arguments(normalized)
                    self._authorize(definition, normalized, context)
                    raise ApprovalRequiredError(
                        "a stored one-time approval is required"
                    )
                normalized, arguments_hash = self._approved_arguments(
                    tool_name,
                    arguments,
                    context,
                    approval_id,
                    definition,
                )
            else:
                if approval_id is not None:
                    raise ApprovalError(
                        "approval_id is not accepted for this tool"
                    )
                normalized = self._normalize_arguments(
                    arguments, policy, now=self._clock()
                )
                arguments_hash = hash_arguments(normalized)

            self._authorize(definition, normalized, context)
            self._enforce_rate_limit(tool_name, context, policy)
            if policy.approval_required:
                self._approval_store.consume_approval(
                    approval_id=approval_id,
                    actor_id=context.actor_id,
                    tenant_id=context.tenant_id,
                    action=tool_name,
                    arguments_hash=arguments_hash,
                    now=self._clock(),
                )

            self._emit(
                "allow",
                "execute",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            allowed = True
            raw_result = self._invoke(
                definition,
                context,
                normalized,
            )
            result = sanitize_output(raw_result, policy)
            self._emit(
                "result",
                "execute",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            return result
        except ToolGatewayError as exc:
            self._emit_failure(
                exc,
                "execute",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
                allowed=allowed,
            )
            raise
        except Exception as exc:
            self._emit_exception(
                exc,
                "execute",
                tool_name,
                context,
                definition,
                arguments_hash,
                started,
            )
            raise

    def close(self, *, wait: bool = False) -> None:
        with self._executor_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> ToolGateway:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _approved_arguments(
        self,
        tool_name: str,
        supplied_arguments: Mapping[str, Any] | None,
        context: ToolExecutionContext,
        approval_id: str,
        definition: ToolDefinition,
    ) -> tuple[dict[str, Any], str]:
        approval = self._approval_store.get_approval(approval_id)
        now = self._clock()
        if approval.consumed_at is not None:
            raise ApprovalConsumedError("approval has already been consumed")
        if approval.expires_at <= now:
            raise ApprovalExpiredError("approval has expired")
        if (
            approval.actor_id != context.actor_id
            or approval.tenant_id != context.tenant_id
            or approval.action != tool_name
        ):
            raise ApprovalError(
                "approval is not valid for this actor, tenant, or action"
            )
        plan = self._plan_store.get_plan(approval.plan_id)
        if plan.expires_at <= now:
            raise PlanExpiredError("plan has expired")
        if (
            plan.action != tool_name
            or plan.actor_id != context.actor_id
            or plan.tenant_id != context.tenant_id
            or plan.arguments_hash != approval.arguments_hash
        ):
            raise ApprovalError("approval and stored plan do not match")
        if plan.required_permission != definition.policy.required_permission:
            raise ApprovalError("tool policy changed; create a new plan")

        stored_arguments = thaw_value(plan.normalized_arguments)
        if supplied_arguments:
            supplied_normalized = self._normalize_arguments(
                supplied_arguments,
                definition.policy,
                now=now,
            )
            if hash_arguments(supplied_normalized) != plan.arguments_hash:
                raise ApprovalArgumentMismatchError(
                    "execution arguments differ from the approved plan"
                )
        return stored_arguments, plan.arguments_hash

    def _authorize(
        self,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> None:
        policy = definition.policy
        if policy.required_permission not in context.permissions:
            raise PermissionDeniedError(
                f"missing permission: {policy.required_permission}"
            )
        if policy.channel_resolver is None:
            channels = frozenset()
        else:
            try:
                channels = frozenset(
                    str(channel).strip()
                    for channel in policy.channel_resolver(arguments)
                )
            except (TypeError, ValueError) as exc:
                raise InvalidToolArgumentsError(str(exc)) from exc
            if any(not channel for channel in channels):
                raise InvalidToolArgumentsError(
                    "channel resolver returned an empty channel ID"
                )
        if policy.channel_required and not channels:
            raise ChannelAccessDeniedError("a channel scope is required")
        if "*" not in context.allowed_channel_ids:
            denied = channels - context.allowed_channel_ids
            if denied:
                raise ChannelAccessDeniedError(
                    "requested channels are outside the actor's grants",
                    details={"denied_channel_ids": sorted(denied)},
                )

    def _normalize_arguments(
        self,
        arguments: Mapping[str, Any] | None,
        policy: ToolPolicy,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise InvalidToolArgumentsError("tool arguments must be an object")
        self._reject_context_injection(arguments)
        unknown = set(arguments) - policy.allowed_arguments
        if unknown:
            raise InvalidToolArgumentsError(
                f"unknown tool arguments: {', '.join(sorted(map(str, unknown)))}"
            )
        missing = policy.required_arguments - set(arguments)
        if missing:
            raise InvalidToolArgumentsError(
                f"missing tool arguments: {', '.join(sorted(missing))}"
            )
        normalized = {
            str(key): self._normalize_json_value(value)
            for key, value in arguments.items()
        }
        self._normalize_common_channels(normalized)
        self._normalize_row_limit(normalized, policy)
        self._normalize_time_windows(normalized, policy, now)
        if policy.argument_normalizer is not None:
            try:
                normalized = dict(policy.argument_normalizer(normalized))
            except (TypeError, ValueError) as exc:
                raise InvalidToolArgumentsError(str(exc)) from exc
            if set(normalized) - policy.allowed_arguments:
                raise InvalidToolArgumentsError(
                    "argument normalizer introduced unknown arguments"
                )
        return normalized

    def _reject_context_injection(
        self, arguments: Mapping[str, Any] | None
    ) -> None:
        if arguments is None:
            return
        found = self._find_context_keys(arguments)
        if found:
            raise ContextInjectionError(
                "server execution context cannot be supplied in tool arguments",
                details={"fields": sorted(found)},
            )

    def _find_context_keys(self, value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            for key, item in value.items():
                text_key = str(key)
                if text_key in _CONTEXT_ARGUMENT_NAMES:
                    found.add(text_key)
                found.update(self._find_context_keys(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                found.update(self._find_context_keys(item))
        return found

    def _normalize_json_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise InvalidToolArgumentsError(
                    "tool arguments cannot contain NaN or infinity"
                )
            return value
        if isinstance(value, datetime):
            return self._format_datetime(self._aware_datetime(value))
        if isinstance(value, date):
            combined = datetime.combine(
                value, datetime_time.min, tzinfo=timezone.utc
            )
            return self._format_datetime(combined)
        if isinstance(value, Enum):
            return self._normalize_json_value(value.value)
        if isinstance(value, Mapping):
            return {
                str(key): self._normalize_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._normalize_json_value(item) for item in value]
        raise InvalidToolArgumentsError(
            f"unsupported tool argument type: {type(value).__name__}"
        )

    def _normalize_common_channels(self, arguments: dict[str, Any]) -> None:
        for key in ("channel_id", "channel"):
            if key in arguments and arguments[key] is not None:
                value = arguments[key]
                if isinstance(value, (list, dict)):
                    raise InvalidToolArgumentsError(
                        f"{key} must contain one channel ID"
                    )
                channel_id = str(value).strip()
                if not channel_id:
                    raise InvalidToolArgumentsError(f"{key} cannot be empty")
                arguments[key] = channel_id
        for key in ("channel_ids", "channels", "channel_scope"):
            if key not in arguments or arguments[key] is None:
                continue
            value = arguments[key]
            if not isinstance(value, list):
                raise InvalidToolArgumentsError(
                    f"{key} must contain a list of channel IDs"
                )
            normalized = []
            seen = set()
            for item in value:
                channel_id = str(item).strip()
                if not channel_id:
                    raise InvalidToolArgumentsError(
                        f"{key} cannot contain empty channel IDs"
                    )
                if channel_id not in seen:
                    seen.add(channel_id)
                    normalized.append(channel_id)
            arguments[key] = normalized

    def _normalize_row_limit(
        self, arguments: dict[str, Any], policy: ToolPolicy
    ) -> None:
        present = [
            key for key in policy.row_limit_arguments if key in arguments
        ]
        if len(present) > 1:
            raise InvalidToolArgumentsError(
                "only one row limit argument may be supplied"
            )
        if present:
            key = present[0]
            value = arguments[key]
            if isinstance(value, bool):
                raise InvalidToolArgumentsError(f"{key} must be an integer")
            try:
                limit = int(value)
            except (TypeError, ValueError) as exc:
                raise InvalidToolArgumentsError(
                    f"{key} must be an integer"
                ) from exc
            if limit <= 0:
                raise InvalidToolArgumentsError(f"{key} must be positive")
            if policy.max_rows is not None:
                limit = min(limit, policy.max_rows)
            arguments[key] = limit
            return

        default = policy.default_rows
        if default is None:
            default = policy.max_rows
        if default is None:
            return
        for key in policy.row_limit_arguments:
            if key in policy.allowed_arguments:
                arguments[key] = default
                return

    def _normalize_time_windows(
        self,
        arguments: dict[str, Any],
        policy: ToolPolicy,
        now: datetime,
    ) -> None:
        max_window = policy.max_time_window
        default_window = policy.default_time_window or max_window

        for key in policy.duration_arguments:
            if key not in arguments:
                continue
            value = arguments[key]
            if isinstance(value, bool):
                raise InvalidToolArgumentsError(f"{key} must be numeric")
            try:
                seconds = float(value)
            except (TypeError, ValueError) as exc:
                raise InvalidToolArgumentsError(
                    f"{key} must be numeric"
                ) from exc
            if not math.isfinite(seconds) or seconds <= 0:
                raise InvalidToolArgumentsError(f"{key} must be positive")
            if max_window is not None:
                seconds = min(seconds, max_window.total_seconds())
            arguments[key] = int(seconds) if seconds.is_integer() else seconds

        handled_pair = False
        for start_key, end_key in policy.time_window_arguments:
            if start_key not in policy.allowed_arguments:
                continue
            if end_key not in policy.allowed_arguments:
                continue
            if (
                start_key in arguments
                or end_key in arguments
                or (not handled_pair and default_window is not None)
            ):
                self._normalize_time_pair(
                    arguments,
                    start_key,
                    end_key,
                    max_window=max_window,
                    default_window=default_window,
                    now=now,
                )
                handled_pair = True
                break

        for key in policy.time_window_object_arguments:
            if key not in arguments:
                continue
            value = arguments[key]
            if not isinstance(value, Mapping):
                raise InvalidToolArgumentsError(f"{key} must be an object")
            window = dict(value)
            start_key = "start" if "start" in window else "start_time"
            end_key = "end" if "end" in window else "end_time"
            self._normalize_time_pair(
                window,
                start_key,
                end_key,
                max_window=max_window,
                default_window=default_window,
                now=now,
            )
            arguments[key] = window

    def _normalize_time_pair(
        self,
        arguments: dict[str, Any],
        start_key: str,
        end_key: str,
        *,
        max_window: timedelta | None,
        default_window: timedelta | None,
        now: datetime,
    ) -> None:
        start = (
            self._parse_datetime(arguments[start_key], start_key)
            if start_key in arguments
            else None
        )
        end = (
            self._parse_datetime(arguments[end_key], end_key)
            if end_key in arguments
            else None
        )
        effective_window = default_window or max_window
        if start is None and end is None:
            if effective_window is None:
                return
            end = self._aware_datetime(now)
            start = end - effective_window
        elif start is None:
            if effective_window is None:
                raise InvalidToolArgumentsError(
                    f"{start_key} is required with {end_key}"
                )
            start = end - effective_window
        elif end is None:
            end = self._aware_datetime(now)
            if end < start:
                raise InvalidToolArgumentsError(
                    f"{start_key} cannot be in the future"
                )
            if effective_window is not None and end - start > effective_window:
                start = end - effective_window
        if start > end:
            raise InvalidToolArgumentsError(
                f"{start_key} must not be after {end_key}"
            )
        if max_window is not None and end - start > max_window:
            start = end - max_window
        arguments[start_key] = self._format_datetime(start)
        arguments[end_key] = self._format_datetime(end)

    def _parse_datetime(self, value: Any, field_name: str) -> datetime:
        if isinstance(value, datetime):
            return self._aware_datetime(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except (OverflowError, OSError, ValueError) as exc:
                raise InvalidToolArgumentsError(
                    f"{field_name} is not a valid timestamp"
                ) from exc
        if not isinstance(value, str):
            raise InvalidToolArgumentsError(
                f"{field_name} must be an ISO-8601 timestamp"
            )
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise InvalidToolArgumentsError(
                f"{field_name} must be an ISO-8601 timestamp"
            ) from exc
        return self._aware_datetime(parsed)

    def _aware_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _format_datetime(self, value: datetime) -> str:
        return (
            self._aware_datetime(value)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

    def _enforce_rate_limit(
        self,
        tool_name: str,
        context: ToolExecutionContext,
        policy: ToolPolicy,
    ) -> None:
        if policy.rate_limit is None:
            return
        if not self._rate_limiter.check_and_record(
            tool_name, context, policy.rate_limit
        ):
            raise RateLimitExceededError("tool rate limit exceeded")

    def _invoke(
        self,
        definition: ToolDefinition,
        context: ToolExecutionContext,
        arguments: Mapping[str, Any],
    ) -> Any:
        timeout = definition.policy.timeout_seconds
        if timeout is None:
            return definition.handler(context, dict(arguments))
        executor = self._get_executor()
        future = executor.submit(definition.handler, context, dict(arguments))
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            future.cancel()
            raise ToolTimeoutError(
                f"tool exceeded its {timeout:g}s timeout"
            ) from exc

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=8,
                    thread_name_prefix="agent-tool",
                )
            return self._executor

    def _require_context(self, context: ToolExecutionContext) -> None:
        if not isinstance(context, ToolExecutionContext):
            raise AuthorizationError(
                "a server-created ToolExecutionContext is required"
            )

    def _bounded_ttl(
        self, requested: float | None, policy_maximum: float
    ) -> float:
        if requested is None:
            return policy_maximum
        if requested <= 0:
            raise InvalidToolArgumentsError("TTL must be positive")
        return min(float(requested), policy_maximum)

    def _emit_failure(
        self,
        exc: ToolGatewayError,
        operation: str,
        tool_name: str,
        context: ToolExecutionContext,
        definition: ToolDefinition | None,
        arguments_hash: str | None,
        started: float,
        *,
        allowed: bool = False,
    ) -> None:
        self._emit(
            "error" if allowed else "deny",
            operation,
            tool_name,
            context,
            definition,
            arguments_hash,
            started,
            code=exc.code,
            details=exc.details,
        )

    def _emit_exception(
        self,
        exc: Exception,
        operation: str,
        tool_name: str,
        context: ToolExecutionContext,
        definition: ToolDefinition | None,
        arguments_hash: str | None,
        started: float,
    ) -> None:
        self._emit(
            "error",
            operation,
            tool_name,
            context,
            definition,
            arguments_hash,
            started,
            code=type(exc).__name__,
        )

    def _emit(
        self,
        phase: str,
        operation: str,
        tool_name: str,
        context: ToolExecutionContext,
        definition: ToolDefinition | None,
        arguments_hash: str | None,
        started: float,
        *,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        actor_id = getattr(context, "actor_id", "<invalid>")
        tenant_id = getattr(context, "tenant_id", "<invalid>")
        event = ToolAuditEvent(
            timestamp=self._clock(),
            phase=phase,
            operation=operation,
            tool_name=tool_name,
            actor_id=actor_id,
            tenant_id=tenant_id,
            request_id=getattr(context, "request_id", None),
            session_id=getattr(context, "session_id", None),
            risk=(
                definition.policy.risk.value if definition is not None else None
            ),
            required_permission=(
                definition.policy.required_permission
                if definition is not None
                else None
            ),
            arguments_hash=arguments_hash,
            code=code,
            duration_ms=(time.monotonic() - started) * 1000,
            details=dict(details or {}),
        )
        with self._audit_lock:
            self._audit_events.append(event)
        if self._audit_callback is None:
            return
        try:
            self._audit_callback(event)
        except Exception as exc:
            raise AuditUnavailableError(
                "audit sink is unavailable; tool execution is blocked"
            ) from exc
