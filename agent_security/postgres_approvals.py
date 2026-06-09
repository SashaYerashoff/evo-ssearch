"""PostgreSQL-backed durable agent action plans and approvals."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from eva_db import PsycopgPool, TransactionContext

from .approvals import ApprovalStore, PlanStore, ToolApproval, ToolPlan, freeze_value
from .context import ToolExecutionContext
from .errors import (
    ApprovalConsumedError,
    ApprovalError,
    ApprovalExpiredError,
    PlanExpiredError,
)


class PostgresPlanApprovalStore(PlanStore, ApprovalStore):
    """Tenant-scoped durable store for server-owned approval workflows."""

    def __init__(self, pool: PsycopgPool) -> None:
        self._pool = pool

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
        db_context = _transaction_context(context)
        plan_id = str(uuid.uuid4())
        arguments_hash_bytes = _hash_bytes(arguments_hash, "arguments_hash")
        now = _utc_now()
        if expires_at <= now:
            raise ValueError("plan expiry must be in the future")
        session_id = _context_uuid(context.agent_session_id, "agent_session_id")
        actor_id = _context_uuid(context.actor_id, "actor_id")
        tenant_id = _context_uuid(context.tenant_id, "tenant_id")
        payload = _plain_mapping(normalized_arguments)

        with self._pool.transaction(db_context) as connection:
            self._upsert_session(
                connection,
                tenant_id=tenant_id,
                actor_id=actor_id,
                session_id=session_id,
            )
            row = connection.execute(
                """
                INSERT INTO agent.action_plans (
                    id,
                    tenant_id,
                    session_id,
                    actor_user_id,
                    action,
                    required_permission,
                    normalized_arguments,
                    arguments_hash,
                    status,
                    created_at,
                    expires_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    'pending', %s, %s
                )
                RETURNING
                    id::text,
                    action,
                    actor_user_id::text,
                    tenant_id::text,
                    required_permission,
                    normalized_arguments,
                    encode(arguments_hash, 'hex'),
                    created_at,
                    expires_at
                """,
                (
                    plan_id,
                    tenant_id,
                    session_id,
                    actor_id,
                    str(action),
                    str(required_permission),
                    _jsonb(payload),
                    arguments_hash_bytes,
                    now,
                    expires_at,
                ),
            ).fetchone()
        return _row_to_plan(row)

    def get_plan(
        self,
        plan_id: str,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolPlan:
        db_context = _required_transaction_context(context)
        plan_uuid = _context_uuid(plan_id, "plan_id")
        with self._pool.transaction(db_context, readonly=True) as connection:
            row = connection.execute(
                """
                SELECT
                    id::text,
                    action,
                    actor_user_id::text,
                    tenant_id::text,
                    required_permission,
                    normalized_arguments,
                    encode(arguments_hash, 'hex'),
                    created_at,
                    expires_at
                FROM agent.action_plans
                WHERE tenant_id = %s AND id = %s
                """,
                (db_context.tenant_id, plan_uuid),
            ).fetchone()
        if row is None:
            raise ApprovalError("unknown plan")
        return _row_to_plan(row)

    def create_approval(
        self,
        *,
        plan: ToolPlan,
        approved_by: ToolExecutionContext,
        expires_at: datetime,
    ) -> ToolApproval:
        db_context = _transaction_context(approved_by)
        plan_id = _context_uuid(plan.plan_id, "plan_id")
        approval_row_id = str(uuid.uuid4())
        raw_token = secrets.token_urlsafe(32)
        token_hash = _approval_token_hash(raw_token)
        now = _utc_now()
        if expires_at <= now:
            raise ValueError("approval expiry must be in the future")
        plan_hash = _hash_bytes(plan.arguments_hash, "arguments_hash")
        actor_id = _context_uuid(approved_by.actor_id, "actor_id")
        tenant_id = _context_uuid(approved_by.tenant_id, "tenant_id")

        with self._pool.transaction(db_context) as connection:
            locked = self._locked_plan(connection, tenant_id, plan_id)
            stored_plan = _row_to_plan(locked)
            status = str(locked[9])
            if stored_plan.expires_at <= now:
                self._expire_plan(connection, tenant_id, plan_id, now)
                raise PlanExpiredError("plan has expired")
            if stored_plan.actor_id != actor_id or stored_plan.tenant_id != tenant_id:
                raise ApprovalError("approval actor does not own the plan")
            if stored_plan.arguments_hash != plan.arguments_hash:
                raise ApprovalError("plan arguments changed")
            if status == "executed":
                raise ApprovalConsumedError("plan has already been executed")
            if status not in {"pending", "approved"}:
                raise ApprovalError(f"plan is {status}")

            approval_expires_at = min(expires_at, stored_plan.expires_at)
            row = connection.execute(
                """
                INSERT INTO agent.action_approvals (
                    id,
                    tenant_id,
                    plan_id,
                    approved_by,
                    plan_arguments_hash,
                    approval_token_hash,
                    status,
                    approved_at,
                    expires_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'active', %s, %s)
                RETURNING
                    plan_id::text,
                    %s::text AS approval_token,
                    %s::text AS action,
                    %s::text AS actor_user_id,
                    tenant_id::text,
                    encode(plan_arguments_hash, 'hex'),
                    approved_by::text,
                    approved_at,
                    expires_at,
                    consumed_at,
                    status,
                    plan_arguments_hash
                """,
                (
                    approval_row_id,
                    tenant_id,
                    plan_id,
                    actor_id,
                    plan_hash,
                    token_hash,
                    now,
                    approval_expires_at,
                    raw_token,
                    stored_plan.action,
                    stored_plan.actor_id,
                ),
            ).fetchone()
            connection.execute(
                """
                UPDATE agent.action_plans
                SET status = 'approved'
                WHERE tenant_id = %s AND id = %s AND status = 'pending'
                """,
                (tenant_id, plan_id),
            )
        return _row_to_approval(row)

    def get_approval(
        self,
        approval_id: str,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolApproval:
        db_context = _required_transaction_context(context)
        row = self._approval_by_token(
            approval_id,
            db_context,
            readonly=True,
        )
        if row is None:
            raise ApprovalError("unknown approval")
        return _row_to_approval(row, raw_token=approval_id)

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
        context = TransactionContext(
            tenant_id=_context_uuid(tenant_id, "tenant_id"),
            actor_id=_context_uuid(actor_id, "actor_id"),
        )
        expected_hash = _hash_bytes(arguments_hash, "arguments_hash")
        with self._pool.transaction(context) as connection:
            row = self._approval_by_token_row(
                connection,
                approval_id,
                tenant_id=context.tenant_id,
                for_update=True,
            )
            if row is None:
                raise ApprovalError("unknown approval")
            approval = _row_to_approval(row, raw_token=approval_id)
            if approval.consumed_at is not None or str(row[10]) == "consumed":
                raise ApprovalConsumedError("approval has already been consumed")
            if approval.expires_at <= now or str(row[10]) == "expired":
                raise ApprovalExpiredError("approval has expired")
            if (
                approval.actor_id != context.actor_id
                or approval.tenant_id != context.tenant_id
                or approval.action != action
                or approval.arguments_hash != arguments_hash
            ):
                raise ApprovalError(
                    "approval is not valid for this actor, tenant, or action"
                )
            if bytes(row[11]) != expected_hash:
                raise ApprovalError("approval hash mismatch")

            plan_row = self._locked_plan(connection, context.tenant_id, approval.plan_id)
            plan = _row_to_plan(plan_row)
            plan_status = str(plan_row[9])
            if plan.expires_at <= now:
                self._expire_plan(connection, context.tenant_id, approval.plan_id, now)
                raise PlanExpiredError("plan has expired")
            if plan_status == "executed":
                raise ApprovalConsumedError("plan has already been executed")
            if plan_status not in {"pending", "approved"}:
                raise ApprovalError(f"plan is {plan_status}")
            if (
                plan.action != action
                or plan.actor_id != context.actor_id
                or plan.tenant_id != context.tenant_id
                or plan.arguments_hash != arguments_hash
            ):
                raise ApprovalError("approval and stored plan do not match")

            consumed_row = connection.execute(
                """
                UPDATE agent.action_approvals
                SET status = 'consumed',
                    consumed_at = %s
                WHERE tenant_id = %s
                  AND approval_token_hash = %s
                  AND status = 'active'
                RETURNING
                    plan_id::text,
                    %s::text AS approval_token,
                    %s::text AS action,
                    %s::text AS actor_user_id,
                    tenant_id::text,
                    encode(plan_arguments_hash, 'hex'),
                    approved_by::text,
                    approved_at,
                    expires_at,
                    consumed_at,
                    status,
                    plan_arguments_hash
                """,
                (
                    now,
                    context.tenant_id,
                    _approval_token_hash(approval_id),
                    approval_id,
                    action,
                    actor_id,
                ),
            ).fetchone()
            if consumed_row is None:
                raise ApprovalConsumedError("approval has already been consumed")
            connection.execute(
                """
                UPDATE agent.action_plans
                SET status = 'executed',
                    executed_at = %s
                WHERE tenant_id = %s AND id = %s
                """,
                (now, context.tenant_id, approval.plan_id),
            )
        return _row_to_approval(consumed_row, raw_token=approval_id)

    def _upsert_session(
        self,
        connection: Any,
        *,
        tenant_id: str,
        actor_id: str,
        session_id: str,
    ) -> None:
        row = connection.execute(
            """
            INSERT INTO agent.sessions (
                id,
                tenant_id,
                user_id,
                status,
                metadata
            )
            VALUES (%s, %s, %s, 'active', '{}'::jsonb)
            ON CONFLICT (tenant_id, id)
            DO UPDATE SET updated_at = clock_timestamp()
            WHERE agent.sessions.user_id = EXCLUDED.user_id
            RETURNING id
            """,
            (session_id, tenant_id, actor_id),
        ).fetchone()
        if row is None:
            raise ApprovalError("agent session is owned by another actor")

    def _locked_plan(
        self,
        connection: Any,
        tenant_id: str,
        plan_id: str,
    ) -> Any:
        row = connection.execute(
            """
            SELECT
                id::text,
                action,
                actor_user_id::text,
                tenant_id::text,
                required_permission,
                normalized_arguments,
                encode(arguments_hash, 'hex'),
                created_at,
                expires_at,
                status
            FROM agent.action_plans
            WHERE tenant_id = %s AND id = %s
            FOR UPDATE
            """,
            (tenant_id, plan_id),
        ).fetchone()
        if row is None:
            raise ApprovalError("unknown plan")
        return row

    def _expire_plan(
        self,
        connection: Any,
        tenant_id: str,
        plan_id: str,
        now: datetime,
    ) -> None:
        connection.execute(
            """
            UPDATE agent.action_plans
            SET status = 'expired'
            WHERE tenant_id = %s
              AND id = %s
              AND status IN ('pending', 'approved')
              AND expires_at <= %s
            """,
            (tenant_id, plan_id, now),
        )

    def _approval_by_token(
        self,
        token: str,
        context: TransactionContext,
        *,
        readonly: bool,
    ) -> Any:
        with self._pool.transaction(context, readonly=readonly) as connection:
            return self._approval_by_token_row(
                connection,
                token,
                tenant_id=context.tenant_id,
                for_update=False,
            )

    def _approval_by_token_row(
        self,
        connection: Any,
        token: str,
        *,
        tenant_id: str,
        for_update: bool,
    ) -> Any:
        lock_clause = "FOR UPDATE" if for_update else ""
        return connection.execute(
            f"""
            SELECT
                a.plan_id::text,
                %s::text AS approval_token,
                p.action,
                p.actor_user_id::text,
                a.tenant_id::text,
                encode(a.plan_arguments_hash, 'hex'),
                a.approved_by::text,
                a.approved_at,
                a.expires_at,
                a.consumed_at,
                a.status,
                a.plan_arguments_hash
            FROM agent.action_approvals a
            JOIN agent.action_plans p
              ON p.tenant_id = a.tenant_id
             AND p.id = a.plan_id
            WHERE a.tenant_id = %s
              AND a.approval_token_hash = %s
            {lock_clause}
            """,
            (token, tenant_id, _approval_token_hash(token)),
        ).fetchone()


PostgreSQLPlanApprovalStore = PostgresPlanApprovalStore


def _row_to_plan(row: Any) -> ToolPlan:
    return ToolPlan(
        plan_id=str(row[0]),
        action=str(row[1]),
        actor_id=str(row[2]),
        tenant_id=str(row[3]),
        required_permission=str(row[4]),
        normalized_arguments=freeze_value(dict(row[5] or {})),
        arguments_hash=str(row[6]),
        created_at=_aware_utc(row[7]),
        expires_at=_aware_utc(row[8]),
    )


def _row_to_approval(row: Any, *, raw_token: str | None = None) -> ToolApproval:
    return ToolApproval(
        approval_id=str(raw_token if raw_token is not None else row[1]),
        plan_id=str(row[0]),
        action=str(row[2]),
        actor_id=str(row[3]),
        tenant_id=str(row[4]),
        arguments_hash=str(row[5]),
        approved_by_actor_id=str(row[6]),
        created_at=_aware_utc(row[7]),
        expires_at=_aware_utc(row[8]),
        consumed_at=None if row[9] is None else _aware_utc(row[9]),
    )


def _transaction_context(context: ToolExecutionContext) -> TransactionContext:
    return TransactionContext(
        tenant_id=_context_uuid(context.tenant_id, "tenant_id"),
        actor_id=_context_uuid(context.actor_id, "actor_id"),
        request_id=context.request_id,
        agent_session_id=(
            None
            if context.agent_session_id is None
            else _context_uuid(context.agent_session_id, "agent_session_id")
        ),
    )


def _required_transaction_context(
    context: ToolExecutionContext | None,
) -> TransactionContext:
    if context is None:
        raise ApprovalError("PostgreSQL approval store requires execution context")
    return _transaction_context(context)


def _context_uuid(value: str | uuid.UUID | None, field_name: str) -> str:
    if value is None:
        raise ApprovalError(f"{field_name} is required")
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise ApprovalError(f"{field_name} must be a UUID") from exc


def _hash_bytes(value: str, field_name: str) -> bytes:
    try:
        raw = bytes.fromhex(str(value))
    except ValueError as exc:
        raise ApprovalError(f"{field_name} must be a SHA-256 hex digest") from exc
    if len(raw) != 32:
        raise ApprovalError(f"{field_name} must be a SHA-256 hex digest")
    return raw


def _approval_token_hash(token: str) -> bytes:
    return hashlib.sha256(str(token).encode("utf-8")).digest()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    return value


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(_plain_value(value))
