from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from .context import ToolExecutionContext
from .errors import (
    ApprovalConsumedError,
    ApprovalError,
    ApprovalExpiredError,
    PlanExpiredError,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_arguments(arguments: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    return value


def thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ToolPlan:
    plan_id: str
    action: str
    actor_id: str
    tenant_id: str
    required_permission: str
    normalized_arguments: Mapping[str, Any]
    arguments_hash: str
    created_at: datetime
    expires_at: datetime

    @property
    def args_hash(self) -> str:
        return self.arguments_hash


@dataclass(frozen=True, slots=True)
class ToolApproval:
    approval_id: str
    plan_id: str
    action: str
    actor_id: str
    tenant_id: str
    arguments_hash: str
    approved_by_actor_id: str
    created_at: datetime
    expires_at: datetime
    consumed_at: datetime | None = None

    @property
    def args_hash(self) -> str:
        return self.arguments_hash


class PlanStore(Protocol):
    def create_plan(
        self,
        *,
        action: str,
        normalized_arguments: Mapping[str, Any],
        arguments_hash: str,
        context: ToolExecutionContext,
        required_permission: str,
        expires_at: datetime,
    ) -> ToolPlan: ...

    def get_plan(
        self,
        plan_id: str,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolPlan: ...


class ApprovalStore(Protocol):
    def create_approval(
        self,
        *,
        plan: ToolPlan,
        approved_by: ToolExecutionContext,
        expires_at: datetime,
    ) -> ToolApproval: ...

    def get_approval(
        self,
        approval_id: str,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolApproval: ...

    def consume_approval(
        self,
        *,
        approval_id: str,
        actor_id: str,
        tenant_id: str,
        action: str,
        arguments_hash: str,
        now: datetime,
    ) -> ToolApproval: ...


class InMemoryPlanApprovalStore(PlanStore, ApprovalStore):
    """Thread-safe reference store for tests and single-process deployments."""

    def __init__(self, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock
        self._plans: dict[str, ToolPlan] = {}
        self._approvals: dict[str, ToolApproval] = {}
        self._lock = RLock()

    def create_plan(
        self,
        *,
        action: str,
        normalized_arguments: Mapping[str, Any],
        arguments_hash: str,
        context: ToolExecutionContext,
        required_permission: str,
        expires_at: datetime,
    ) -> ToolPlan:
        now = self._clock()
        if expires_at <= now:
            raise ValueError("plan expiry must be in the future")
        plan = ToolPlan(
            plan_id=secrets.token_urlsafe(24),
            action=action,
            actor_id=context.actor_id,
            tenant_id=context.tenant_id,
            required_permission=required_permission,
            normalized_arguments=freeze_value(normalized_arguments),
            arguments_hash=arguments_hash,
            created_at=now,
            expires_at=expires_at,
        )
        with self._lock:
            self._plans[plan.plan_id] = plan
        return plan

    def get_plan(
        self,
        plan_id: str,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolPlan:
        with self._lock:
            plan = self._plans.get(plan_id)
        if plan is None:
            raise ApprovalError("unknown plan")
        return plan

    def create_approval(
        self,
        *,
        plan: ToolPlan,
        approved_by: ToolExecutionContext,
        expires_at: datetime,
    ) -> ToolApproval:
        now = self._clock()
        if plan.expires_at <= now:
            raise PlanExpiredError("plan has expired")
        if expires_at <= now:
            raise ValueError("approval expiry must be in the future")
        if (
            approved_by.actor_id != plan.actor_id
            or approved_by.tenant_id != plan.tenant_id
        ):
            raise ApprovalError("approval actor does not own the plan")
        approval = ToolApproval(
            approval_id=secrets.token_urlsafe(32),
            plan_id=plan.plan_id,
            action=plan.action,
            actor_id=plan.actor_id,
            tenant_id=plan.tenant_id,
            arguments_hash=plan.arguments_hash,
            approved_by_actor_id=approved_by.actor_id,
            created_at=now,
            expires_at=min(expires_at, plan.expires_at),
        )
        with self._lock:
            self._approvals[approval.approval_id] = approval
        return approval

    def get_approval(
        self,
        approval_id: str,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolApproval:
        with self._lock:
            approval = self._approvals.get(approval_id)
        if approval is None:
            raise ApprovalError("unknown approval")
        return approval

    def consume_approval(
        self,
        *,
        approval_id: str,
        actor_id: str,
        tenant_id: str,
        action: str,
        arguments_hash: str,
        now: datetime,
    ) -> ToolApproval:
        with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None:
                raise ApprovalError("unknown approval")
            if approval.consumed_at is not None:
                raise ApprovalConsumedError("approval has already been consumed")
            if approval.expires_at <= now:
                raise ApprovalExpiredError("approval has expired")
            if (
                approval.actor_id != actor_id
                or approval.tenant_id != tenant_id
                or approval.action != action
                or approval.arguments_hash != arguments_hash
            ):
                raise ApprovalError(
                    "approval is not valid for this actor, tenant, or action"
                )
            consumed = replace(approval, consumed_at=now)
            self._approvals[approval_id] = consumed
            return consumed
