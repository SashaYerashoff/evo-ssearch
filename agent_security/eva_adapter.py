from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from security import Permission

from .audit import ToolAuditEvent
from .context import ToolExecutionContext
from .errors import (
    AuditUnavailableError,
    ChannelAccessDeniedError,
    InvalidToolArgumentsError,
    ToolGatewayError,
)
from .gateway import ToolGateway
from .approvals import hash_arguments
from .policy import RateLimit, ToolPolicy, ToolRisk
from .registry import ToolRegistry


_TRUE_ARG_STRINGS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_ARG_STRINGS = frozenset({"0", "false", "no", "n", "off"})


def _coerce_bool_argument(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_ARG_STRINGS:
            return True
        if normalized in _FALSE_ARG_STRINGS:
            return False
        return default
    return bool(value)


_PREVIEW_ONLY_TOOLS = frozenset(
    {
        "create_probe",
        "delete_probes",
        "update_probe",
        "update_prompt_settings",
        "restore_video_summary_history",
    }
)
_HIDDEN_UNTIL_APPROVALS = frozenset({"create_bookmark"})
_SINGLE_CHANNEL_FOR_SCOPED_ACTORS = frozenset(
    {
        "search_archive",
        "get_visual_window_signals",
        "calibrate_probe_from_archive",
        "prepare_probe_calibration_batch",
        "get_detections",
        "get_detection_summary",
        "build_research_batch",
        "describe_frame",
        "get_prompt_settings",
        "update_prompt_settings",
        "get_video_summaries",
        "list_attention_bursts",
        "count_video_summary_events",
        "track_visual_state_transitions",
        "generate_report",
    }
)

_TOOL_PERMISSIONS: dict[str, Permission] = {
    "search_archive": Permission.DETECTIONS_VIEW,
    "get_visual_window_signals": Permission.DETECTIONS_VIEW,
    "calibrate_probe_from_archive": Permission.DETECTIONS_VIEW,
    "prepare_probe_calibration_batch": Permission.DETECTIONS_VIEW,
    "get_detections": Permission.DETECTIONS_VIEW,
    "get_detection_summary": Permission.DETECTIONS_VIEW,
    "list_channels": Permission.STREAMS_VIEW,
    "normalize_time_window": Permission.AGENT_USE,
    "lookup_help": Permission.AGENT_USE,
    "list_video_summary_channels": Permission.STREAMS_VIEW,
    "list_probes": Permission.REPORTS_VIEW,
    "survey_channels": Permission.STREAMS_VIEW,
    "build_research_batch": Permission.DETECTIONS_VIEW,
    "create_probe": Permission.PROBES_MANAGE,
    "deploy_summary": Permission.REPORTS_VIEW,
    "delete_probes": Permission.PROBES_MANAGE,
    "update_probe": Permission.PROBES_MANAGE,
    "describe_frame": Permission.DETECTIONS_VIEW,
    "get_prompt_settings": Permission.STREAMS_VIEW,
    "update_prompt_settings": Permission.PROMPTS_MANAGE,
    "get_video_summaries": Permission.STREAMS_VIEW,
    "list_attention_bursts": Permission.STREAMS_VIEW,
    "count_video_summary_events": Permission.STREAMS_VIEW,
    "track_visual_state_transitions": Permission.DETECTIONS_VIEW,
    "create_bookmark": Permission.BOOKMARKS_CREATE,
    "generate_report": Permission.REPORTS_VIEW,
    "restore_video_summary_history": Permission.CAPTURE_MANAGE,
    "get_video_summary_restore_status": Permission.STREAMS_VIEW,
}

_WRITE_TOOLS = _PREVIEW_ONLY_TOOLS | _HIDDEN_UNTIL_APPROVALS
_CHANNEL_REQUIRED_TOOLS = frozenset(
    {
        "create_probe",
        "delete_probes",
        "update_probe",
        "get_video_summaries",
        "list_attention_bursts",
        "count_video_summary_events",
        "track_visual_state_transitions",
        "get_visual_window_signals",
        "create_bookmark",
    }
)


def _preview_apply_requires_approval(arguments: Mapping[str, Any]) -> bool:
    return arguments.get("preview", True) is not True


class EvaAgentToolAdapter:
    """Authorize and audit legacy EVA agent tools before dispatch."""

    def __init__(
        self,
        legacy_tools: Any,
        tool_schemas: Sequence[Mapping[str, Any]],
        *,
        audit_callback: Callable[[ToolAuditEvent], None],
        plan_store: Any | None = None,
        approval_store: Any | None = None,
    ) -> None:
        self._legacy_tools = legacy_tools
        self._audit_callback = audit_callback
        self._schemas = {
            str(item["function"]["name"]): copy.deepcopy(dict(item))
            for item in tool_schemas
        }
        self._local = threading.local()
        registry = ToolRegistry()
        for name, schema in self._schemas.items():
            parameters = schema["function"].get("parameters") or {}
            properties = parameters.get("properties") or {}
            allowed_arguments = set(map(str, properties))
            if name in {"delete_probes", "update_probe"}:
                allowed_arguments.update({"channel_id", "channel_ids"})
            policy = ToolPolicy(
                required_permission=_TOOL_PERMISSIONS[name].value,
                risk=(
                    ToolRisk.EXTERNAL_SIDE_EFFECT
                    if name == "create_bookmark"
                    else ToolRisk.WRITE
                    if name in _WRITE_TOOLS
                    else ToolRisk.READ
                ),
                approval_required=name in _WRITE_TOOLS,
                approval_required_when=(
                    _preview_apply_requires_approval
                    if name in _PREVIEW_ONLY_TOOLS
                    else None
                ),
                allowed_arguments=frozenset(allowed_arguments),
                required_arguments=frozenset(
                    map(str, parameters.get("required") or ())
                ),
                channel_required=name in _CHANNEL_REQUIRED_TOOLS,
                max_rows=self._max_rows(name),
                default_rows=self._default_rows(name),
                time_window_arguments=() if name == "normalize_time_window" else (
                    ("start_time", "end_time"),
                    ("since", "until"),
                    ("from", "to"),
                ),
                time_window_object_arguments=() if name == "normalize_time_window" else ("time_window", "window"),
                duration_arguments=() if name == "normalize_time_window" else (
                    "time_window_seconds",
                    "window_seconds",
                ),
                max_output_bytes=self._max_output_bytes(name),
                max_output_items=self._max_output_items(name),
                max_output_string_chars=24_000,
                timeout_seconds=self._timeout_seconds(name),
                rate_limit=RateLimit(
                    max_calls=120,
                    window_seconds=60,
                    scope="actor",
                ),
            )
            registry.register(
                name,
                self._handler(name),
                policy,
                description=str(schema["function"].get("description") or ""),
                input_schema=parameters,
            )
        self.gateway = ToolGateway(
            registry,
            audit_callback=audit_callback,
            plan_store=plan_store,
            approval_store=approval_store,
        )

    @staticmethod
    def _max_rows(name: str) -> int | None:
        return {
            "search_archive": 48,
            "get_detections": 100,
            "list_video_summary_channels": 100,
            "get_video_summaries": 100,
            "list_attention_bursts": 100,
            "count_video_summary_events": 120,
            "track_visual_state_transitions": 120,
            "calibrate_probe_from_archive": 120,
            "prepare_probe_calibration_batch": 120,
        }.get(name)

    @staticmethod
    def _max_output_items(name: str) -> int:
        # Video-summary responses contain bounded semantic entries, evidence
        # rows, and nested coverage contracts. The generic 500-item ceiling can
        # be exhausted by coverage metadata before it reaches entries/image_url,
        # producing false empty summaries and blank UI previews. Bytes and row
        # limits remain independently bounded.
        #
        # search_archive and get_detections have the identical shape: up to
        # 48-100 uncompacted detection rows (each carrying its full
        # vlm_summary/vlm_alert payload — state_observations/
        # state_transition_events/vector_signal can be sizeable) followed by
        # (for search_archive) a trailing `coverage` object. The sanitizer
        # walks keys in order and stops counting once the budget is spent,
        # so a full page of rows silently ate `coverage` before the
        # sanitizer ever reached it — both the operator UI and the model
        # then saw "coverage: not reported" and a spurious `_truncated`
        # flag on searches that were not actually truncated. This is a
        # coverage-honesty gate (docs/tuktuk/grammar_pin.md); see
        # docs/tuktuk/grammar_review_questions.md (Resolved,
        # "search_archive coverage truncation").
        return {
            "get_video_summaries": 4_000,
            "list_video_summary_channels": 2_000,
            "search_archive": 4_000,
            "get_detections": 4_000,
        }.get(name, 500)

    @staticmethod
    def _max_output_bytes(name: str) -> int:
        # Independent of _max_output_items: sanitize_output's final byte
        # check (_bound_serialized) replaces the *entire* result with a
        # useless {"_truncated": true, "preview": "<64KB of raw JSON>"}
        # envelope once the serialized size exceeds this cap — wiping out
        # coverage/count/scope alongside the rows, regardless of key order.
        # Real archive rows run large (~9-11KB each with full vlm_summary
        # payload/state arrays): search_archive at its default page (12
        # rows) measured ~141KB, and get_detections at its own default
        # (20 rows) measured ~221KB — both routinely exceeding the generic
        # 96,000-byte default in normal operation, not just at max_rows.
        # get_detections is one of the most frequently called tools, so
        # this was silently returning an empty preview envelope on a large
        # fraction of ordinary calls. See
        # docs/tuktuk/grammar_review_questions.md (Resolved,
        # "search_archive coverage truncation").
        return {
            "search_archive": 2_000_000,
            "get_detections": 2_000_000,
        }.get(name, 96_000)

    @staticmethod
    def _default_rows(name: str) -> int | None:
        return {
            "search_archive": 12,
            "get_detections": 20,
            "list_video_summary_channels": 16,
            "get_video_summaries": 20,
            "list_attention_bursts": 24,
            "count_video_summary_events": 40,
            "track_visual_state_transitions": 40,
            "calibrate_probe_from_archive": 24,
            "prepare_probe_calibration_batch": 24,
        }.get(name)

    @staticmethod
    def _timeout_seconds(name: str) -> float:
        return {
            "survey_channels": 300.0,
            "describe_frame": 120.0,
            "search_archive": 90.0,
            "get_visual_window_signals": 90.0,
            "calibrate_probe_from_archive": 90.0,
            "prepare_probe_calibration_batch": 90.0,
            "build_research_batch": 90.0,
            "get_video_summaries": 90.0,
            "list_attention_bursts": 90.0,
            "count_video_summary_events": 90.0,
            "track_visual_state_transitions": 90.0,
            "list_video_summary_channels": 90.0,
            "generate_report": 90.0,
            "restore_video_summary_history": 180.0,
        }.get(name, 45.0)

    def close(self) -> None:
        self.gateway.close()

    def available_tool_schemas(
        self,
        context: ToolExecutionContext,
    ) -> list[dict[str, Any]]:
        allowed_names = {
            definition.name
            for definition in self.gateway.available_tools(context)
        } - _HIDDEN_UNTIL_APPROVALS
        if "*" not in context.allowed_channel_ids:
            # The durable backfill worker is deployment-global and its status
            # contains channel IDs plus aggregate progress across the job. Do
            # not expose a misleading partially-filtered global status to a
            # channel-scoped actor.
            allowed_names.discard("get_video_summary_restore_status")
        return [
            self._model_schema(name)
            for name in self._schemas
            if name in allowed_names
        ]

    def visible_probes(
        self,
        context: ToolExecutionContext,
    ) -> list[dict[str, Any]]:
        probes = self._legacy_tools._ps.list_probes()
        if "*" in context.allowed_channel_ids:
            return [dict(item) for item in probes]
        return [
            dict(item)
            for item in probes
            if str(item.get("channel_id")) in context.allowed_channel_ids
        ]

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any] | None,
        context: ToolExecutionContext,
        *,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any:
        try:
            prepared = self._prepare_arguments(name, arguments or {}, context)
        except ToolGatewayError as exc:
            self._audit_preparation_denial(
                name,
                arguments or {},
                context,
                exc,
            )
            raise
        self._local.progress_cb = progress_cb
        try:
            result = self.gateway.execute(name, prepared, context)
        finally:
            self._local.progress_cb = None
        return self._with_approval_plan(name, prepared, context, result)

    def create_plan(
        self,
        name: str,
        arguments: Mapping[str, Any] | None,
        context: ToolExecutionContext,
        *,
        ttl_seconds: float | None = None,
    ):
        try:
            prepared = self._prepare_arguments(name, arguments or {}, context)
        except ToolGatewayError as exc:
            self._audit_preparation_denial(
                name,
                arguments or {},
                context,
                exc,
            )
            raise
        return self.gateway.create_plan(
            name,
            prepared,
            context,
            ttl_seconds=ttl_seconds,
        )

    def approve_and_execute(
        self,
        plan_id: str,
        context: ToolExecutionContext,
        *,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any:
        plan = self.gateway._plan_store.get_plan(plan_id, context=context)
        self._local.progress_cb = progress_cb
        try:
            approval = self.gateway.approve(plan_id, context)
            result = self.gateway.execute(
                plan.action,
                None,
                context,
                approval_id=approval.approval_id,
            )
            receipt = {
                "type": "agent_action_applied",
                "plan_id": plan.plan_id,
                "tool": plan.action,
                "status": "applied",
                "result_status": (
                    str(result.get("status"))
                    if isinstance(result, Mapping) and result.get("status") is not None
                    else None
                ),
            }
            if isinstance(result, Mapping):
                enriched = dict(result)
                enriched["action_receipt"] = receipt
                return enriched
            return {
                "status": "applied",
                "result": result,
                "action_receipt": receipt,
            }
        finally:
            self._local.progress_cb = None

    def _with_approval_plan(
        self,
        name: str,
        prepared_arguments: Mapping[str, Any],
        context: ToolExecutionContext,
        result: Any,
    ) -> Any:
        if (
            name not in _PREVIEW_ONLY_TOOLS
            or not isinstance(result, Mapping)
            or result.get("status") != "preview"
            or prepared_arguments.get("preview", True) is not True
        ):
            return result
        apply_arguments = dict(prepared_arguments)
        apply_arguments["preview"] = False
        plan = self.gateway.create_plan(name, apply_arguments, context)
        enriched = dict(result)
        enriched["approval"] = {
            "plan_id": plan.plan_id,
            "action": plan.action,
            "expires_at": plan.expires_at.isoformat(),
            "required_permission": plan.required_permission,
        }
        return enriched

    def _audit_preparation_denial(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
        error: ToolGatewayError,
    ) -> None:
        permission = _TOOL_PERMISSIONS.get(name)
        risk = (
            ToolRisk.EXTERNAL_SIDE_EFFECT
            if name == "create_bookmark"
            else ToolRisk.WRITE
            if name in _WRITE_TOOLS
            else ToolRisk.READ
        )
        try:
            arguments_hash = hash_arguments(dict(arguments))
        except Exception:
            arguments_hash = None
        event = ToolAuditEvent(
            timestamp=datetime.now(timezone.utc),
            phase="deny",
            operation="execute",
            tool_name=name,
            actor_id=context.actor_id,
            tenant_id=context.tenant_id,
            request_id=context.request_id,
            session_id=context.session_id,
            actor_roles=tuple(sorted(context.roles)),
            source_ip=context.client_ip,
            risk=risk.value,
            required_permission=permission.value if permission else None,
            arguments_hash=arguments_hash,
            code=error.code,
            details=error.details,
        )
        try:
            self._audit_callback(event)
        except Exception as exc:
            raise AuditUnavailableError(
                "audit sink is unavailable; tool execution is blocked"
            ) from exc

    def _handler(self, name: str):
        def execute_legacy(
            context: ToolExecutionContext,
            arguments: Mapping[str, Any],
        ) -> Any:
            set_perms = getattr(self._legacy_tools, "_set_trusted_permissions", None)
            clear_perms = getattr(self._legacy_tools, "_clear_trusted_permissions", None)
            if callable(set_perms):
                set_perms(context.permissions)
            try:
                result = self._legacy_tools.execute(
                    name,
                    dict(arguments),
                    progress_cb=getattr(self._local, "progress_cb", None),
                )
                return self._filter_result(name, result, context)
            finally:
                if callable(clear_perms):
                    clear_perms()

        return execute_legacy

    def _prepare_arguments(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if name not in self._schemas:
            return dict(arguments)
        prepared = copy.deepcopy(dict(arguments))

        if name == "lookup_help":
            # Permissions never travel through model/tool args; strip any attempt.
            # Trusted permissions are passed to the tool via execution context.
            prepared.pop("_granted_permissions", None)

        if name == "list_channels" and "now" in prepared:
            now = prepared.pop("now")
            prepared.setdefault("force", _coerce_bool_argument(now, default=False))

        if name == "generate_report":
            since_ms = prepared.pop("since_ms", None)
            until_ms = prepared.pop("until_ms", None)
            try:
                if since_ms is not None and prepared.get("from_ts") is None:
                    prepared["from_ts"] = float(since_ms) / 1000.0
                if until_ms is not None and prepared.get("to_ts") is None:
                    prepared["to_ts"] = float(until_ms) / 1000.0
            except (TypeError, ValueError) as exc:
                raise InvalidToolArgumentsError(
                    "generate_report since_ms/until_ms must be Unix milliseconds"
                ) from exc

        if name == "search_archive":
            scope = str(prepared.get("scope") or "detections").strip()
            if scope != "detections":
                raise InvalidToolArgumentsError(
                    "agent filesystem search is disabled; use detections scope"
                )
            prepared.pop("folder", None)
        if name == "describe_frame" and prepared.get("image_path"):
            raise InvalidToolArgumentsError(
                "agent filesystem image paths are disabled"
            )

        self._resolve_channel_reference(prepared)
        if name == "describe_frame":
            self._resolve_detection_channel(prepared)
        if name == "update_probe":
            self._resolve_update_probe_channel(prepared)
        if name == "delete_probes":
            self._resolve_delete_probe_channels(prepared, context)
        scoped_channels = self._scoped_channels(context)
        if name == "get_video_summary_restore_status" and scoped_channels is not None:
            raise ChannelAccessDeniedError(
                "video-summary restoration status is deployment-wide; ask an all-channel administrator",
                details={"scope": "deployment"},
            )
        if name == "survey_channels" and scoped_channels is not None:
            prepared.setdefault("channel_ids", sorted(scoped_channels))
        elif name == "list_video_summary_channels" and scoped_channels is not None:
            prepared.setdefault("channel_ids", sorted(scoped_channels))
        elif name in {"calibrate_probe_from_archive", "prepare_probe_calibration_batch"} and scoped_channels is not None:
            prepared.setdefault("channel_ids", sorted(scoped_channels))
            if name == "prepare_probe_calibration_batch":
                self._filter_probe_batch_items_for_scope(prepared, scoped_channels)
        elif name == "generate_report" and scoped_channels is not None:
            prepared.setdefault("channel_ids", sorted(scoped_channels))
        elif name == "restore_video_summary_history" and scoped_channels is not None:
            prepared.setdefault("channel_ids", sorted(scoped_channels))
        elif (
            name in _SINGLE_CHANNEL_FOR_SCOPED_ACTORS
            and scoped_channels is not None
            and not self._argument_has_channel(prepared)
        ):
            if len(scoped_channels) != 1:
                raise ChannelAccessDeniedError(
                    "choose an explicit channel for this tool",
                    details={"allowed_channel_ids": sorted(scoped_channels)},
                )
            prepared["channel_id"] = next(iter(scoped_channels))

        if name in _PREVIEW_ONLY_TOOLS:
            prepared.setdefault("preview", True)
        return prepared

    def _resolve_channel_reference(self, arguments: dict[str, Any]) -> None:
        if arguments.get("channel_id") is not None:
            return
        if not any(
            str(arguments.get(key) or "").strip()
            for key in ("channel_ref", "channel", "channel_title", "channel_name")
        ):
            return
        channel_id = self._legacy_tools._resolve_channel_id(
            arguments,
            required=False,
        )
        if channel_id is not None:
            arguments["channel_id"] = channel_id

    def _resolve_detection_channel(self, arguments: dict[str, Any]) -> None:
        detection_id = arguments.get("detection_id")
        if detection_id is None:
            return
        try:
            normalized_id = int(detection_id)
        except (TypeError, ValueError) as exc:
            raise InvalidToolArgumentsError(
                "detection_id must be an integer"
            ) from exc
        records = self._legacy_tools._ds.fetch_detections_by_ids(
            [normalized_id],
            include_vectors=False,
        )
        if not records:
            raise InvalidToolArgumentsError("detection does not exist")
        channel_id = records[0].get("channel_id")
        if channel_id is None:
            raise InvalidToolArgumentsError(
                "detection has no channel ownership metadata"
            )
        arguments["channel_id"] = channel_id

    def _resolve_update_probe_channel(self, arguments: dict[str, Any]) -> None:
        probes = self._legacy_tools._ps.list_probes()
        probe_id = str(arguments.get("probe_id") or "").strip()
        probe_name = str(arguments.get("probe_name") or "").strip().lower()
        matches = [
            probe
            for probe in probes
            if (probe_id and str(probe.get("id") or "") == probe_id)
            or (
                not probe_id
                and probe_name
                and str(probe.get("name") or "").strip().lower() == probe_name
            )
        ]
        if len(matches) != 1:
            raise InvalidToolArgumentsError(
                "update_probe must resolve to exactly one existing probe"
            )
        arguments["channel_id"] = matches[0].get("channel_id")

    def _resolve_delete_probe_channels(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> None:
        probes = self._legacy_tools._ps.list_probes()
        delete_all = bool(arguments.get("delete_all", False))
        if delete_all:
            if "*" not in context.allowed_channel_ids:
                raise ChannelAccessDeniedError(
                    "delete_all is unavailable to channel-scoped actors"
                )
            targets = probes
        else:
            wanted = {
                str(item).strip()
                for item in arguments.get("probe_ids") or ()
                if str(item).strip()
            }
            targets = [
                probe
                for probe in probes
                if str(probe.get("id") or "") in wanted
            ]
        arguments["channel_ids"] = sorted(
            {
                str(probe.get("channel_id"))
                for probe in targets
                if probe.get("channel_id") is not None
            }
        )

    @staticmethod
    def _filter_probe_batch_items_for_scope(
        arguments: dict[str, Any],
        scoped_channels: frozenset[str],
    ) -> None:
        items = arguments.get("items")
        if not isinstance(items, list):
            return
        filtered_items: list[dict[str, Any]] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            item = copy.deepcopy(raw)
            if item.get("channel_ids"):
                allowed = [
                    channel_id
                    for channel_id in item.get("channel_ids") or []
                    if str(channel_id) in scoped_channels
                ]
                item["channel_ids"] = allowed
                if not allowed:
                    continue
            elif item.get("channel_id") is not None:
                if str(item.get("channel_id")) not in scoped_channels:
                    continue
            filtered_items.append(item)
        arguments["items"] = filtered_items

    @staticmethod
    def _argument_has_channel(arguments: Mapping[str, Any]) -> bool:
        return (
            arguments.get("channel_id") is not None
            or bool(arguments.get("channel_ids"))
        )

    @staticmethod
    def _scoped_channels(
        context: ToolExecutionContext,
    ) -> set[str] | None:
        if "*" in context.allowed_channel_ids:
            return None
        return set(context.allowed_channel_ids)

    def _filter_result(
        self,
        name: str,
        result: Any,
        context: ToolExecutionContext,
    ) -> Any:
        scoped_channels = self._scoped_channels(context)
        if scoped_channels is None or not isinstance(result, dict):
            return result
        if name == "list_channels":
            channels = [
                item
                for item in result.get("channels") or ()
                if str(item.get("id")) in scoped_channels
            ]
            return {**result, "count": len(channels), "channels": channels}
        if name == "list_probes":
            probes = [
                item
                for item in result.get("probes") or ()
                if str(item.get("channel_id")) in scoped_channels
            ]
            return {**result, "count": len(probes), "probes": probes}
        return result

    def _model_schema(self, name: str) -> dict[str, Any]:
        schema = copy.deepcopy(self._schemas[name])
        parameters = schema["function"].get("parameters") or {}
        properties = parameters.get("properties") or {}
        if name == "search_archive":
            properties.pop("folder", None)
            scope = properties.get("scope")
            if isinstance(scope, dict):
                scope["enum"] = ["detections"]
                scope["description"] = "Search the authorized detections archive."
        if name == "describe_frame":
            properties.pop("image_path", None)
        if name in _PREVIEW_ONLY_TOOLS and "preview" in properties:
            properties["preview"] = {
                "type": "boolean",
                "enum": [True],
                "description": (
                    "Must be true for model calls. The UI may show an Apply "
                    "button for the returned approval.plan_id; the model must "
                    "not call this tool with preview=false."
                ),
            }
        return schema
