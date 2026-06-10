import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import atexit
import base64
import copy
import gc
import html as html_lib
import json
import math
import pickle
import secrets
import socket
import threading
import time
import uuid
import requests
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
from urllib.parse import unquote
from threading import Lock

import numpy as np
import torch
import cv2
import clip
import faiss
from PIL import Image
from transformers import AutoModel, AutoProcessor
from flask import Flask, g, request, jsonify, send_file, make_response, render_template, Response, stream_with_context
from flask_cors import CORS

from config import config
from detection_store import DetectionsStore
from embedders.dino_encoder import DINOEncoder
from eva_db import DatabaseSettings, PsycopgPool
from inference_queue import (
    LuxriotInferenceQueueRuntime,
    PostgresInferenceQueueRepository,
)
from luxriot_connector import LuxriotManager
from probe_manager import ProbeManager
from security import (
    ALL_CHANNELS,
    AuditEvent,
    AuditEventBuilder,
    AuthContext,
    AuthenticationService,
    InvalidCredentials,
    LoginThrottled,
    Permission,
    ROLE_PERMISSIONS,
    Role,
    require_channel_access,
    require_permission,
)
from agent_security import ToolExecutionContext, ToolGatewayError
from agent_security.audit import ToolAuditEvent
if TYPE_CHECKING:
    from heads.mask2former_head import Mask2FormerHead
try:
    from heads.mask2former_head import Mask2FormerHead as _Mask2FormerHeadRuntime
except Exception:  # pragma: no cover - optional dependency
    _Mask2FormerHeadRuntime = None  # type: ignore[misc]

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config["MAX_CONTENT_LENGTH"] = max(1, int(config.MAX_FILE_SIZE_MB)) * 1024 * 1024
if config.CORS_ALLOWED_ORIGINS:
    CORS(app, resources={r"/*": {"origins": list(config.CORS_ALLOWED_ORIGINS)}})

# Global embedder state
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model: Optional[torch.nn.Module] = None
clip_preprocess = None
clip_processor: Optional[Any] = None
clip_backend_kind = "openai_clip"
clip_runtime_model = ""
clip_runtime_device = device
dino_encoder: Optional[DINOEncoder] = None
mask2former_head: Optional["Mask2FormerHead"] = None
_mask2former_lock = Lock()
_mask2former_failed = False
SUPPORTED_EMBEDDERS = {"clip", "dino", "fusion"}
EMBEDDER_SUBDIRS: Dict[str, str] = {"clip": "clip", "dino": "dino"}
FaissIndexBundle = Tuple[Optional[faiss.Index], Optional[List[str]], Optional[List[Dict[str, Any]]], Dict[str, Any]]

if hasattr(Image, "Resampling"):
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:  # pragma: no cover - Pillow compatibility fallback
    RESAMPLE_NEAREST = Image.NEAREST  # type: ignore[attr-defined]
    RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]
    RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]
active_embedder = config.EMBEDDER if config.EMBEDDER in SUPPORTED_EMBEDDERS else "clip"
if active_embedder == "fusion" and not config.FUSION_ENABLED:
    active_embedder = "clip"

LOCAL_HOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
TRUE_BOOL_STRINGS = {"1", "true", "yes", "on"}
FALSE_BOOL_STRINGS = {"0", "false", "no", "off"}
PROBE_ROI_MIN_SIDE = 0.02
PROBE_ROI_PADDING = 0.05
_auth_service: Optional[AuthenticationService] = None
_auth_service_lock = Lock()
_identity_repository: Optional[Any] = None
_identity_repository_lock = Lock()
_audit_writer: Optional[Any] = None
_audit_writer_lock = Lock()
_audit_db_pool: Optional[PsycopgPool] = None
_audit_db_pool_lock = Lock()

_MUTATION_ENDPOINT_PERMISSIONS: Dict[str, Optional[Permission]] = {
    "auth_logout": None,
    "save_comment": Permission.BOOKMARKS_CREATE,
    "index_folder": Permission.MODELS_MANAGE,
    "index_segments": Permission.MODELS_MANAGE,
    "luxriot_start_capture": Permission.CAPTURE_MANAGE,
    "luxriot_prompt_settings": Permission.PROMPTS_MANAGE,
    "luxriot_stop_capture": Permission.CAPTURE_MANAGE,
    "luxriot_flush_capture": Permission.CAPTURE_MANAGE,
    "luxriot_stop_stream": Permission.CAPTURE_MANAGE,
    "luxriot_stop_all_streams": Permission.CAPTURE_MANAGE,
    "luxriot_bookmark": Permission.BOOKMARKS_CREATE,
    "probes_query": Permission.PROBES_RUN,
    "probes_start_capture": Permission.CAPTURE_MANAGE,
    "probes_stop_capture": Permission.CAPTURE_MANAGE,
    "probes_save": Permission.PROBES_MANAGE,
    "probes_delete": Permission.PROBES_MANAGE,
    "probes_run": Permission.PROBES_RUN,
    "agent_chat": Permission.AGENT_USE,
    "agent_action_plan_execute": Permission.AGENT_USE,
    "agent_config": Permission.MODELS_MANAGE,
    "agent_skills_create": Permission.PROMPTS_MANAGE,
    "agent_skill_detail": Permission.PROMPTS_MANAGE,
    "agent_session": Permission.AGENT_USE,
    "save_settings": Permission.SETTINGS_MANAGE,
    "save_settings_env": Permission.SETTINGS_MANAGE,
}
_SENSITIVE_ENDPOINT_PERMISSIONS: Dict[str, Permission] = {
    "serve_image": Permission.DETECTIONS_VIEW,
    "serve_detection_image": Permission.DETECTIONS_VIEW,
    "get_comments": Permission.REPORTS_VIEW,
    "get_commented_images": Permission.REPORTS_VIEW,
    "check_index": Permission.DIAGNOSTICS_VIEW,
    "video_understanding": Permission.STREAMS_VIEW,
    "describe_image": Permission.DETECTIONS_VIEW,
    "search": Permission.DETECTIONS_VIEW,
    "search_by_image": Permission.DETECTIONS_VIEW,
    "search_by_mask": Permission.DETECTIONS_VIEW,
    "segment_from_point": Permission.DETECTIONS_VIEW,
    "detections_search_text": Permission.DETECTIONS_VIEW,
    "detections_search_image": Permission.DETECTIONS_VIEW,
    "detections_list": Permission.DETECTIONS_VIEW,
    "detections_summary": Permission.DETECTIONS_VIEW,
    "luxriot_channels": Permission.STREAMS_VIEW,
    "luxriot_prompt_settings": Permission.STREAMS_VIEW,
    "luxriot_snapshot": Permission.STREAMS_VIEW,
    "luxriot_session_status": Permission.STREAMS_VIEW,
    "luxriot_summary_rollups": Permission.REPORTS_VIEW,
    "luxriot_streams_status": Permission.STREAMS_VIEW,
    "probes_status": Permission.STREAMS_VIEW,
    "probes_list": Permission.REPORTS_VIEW,
    "probes_bench": Permission.DIAGNOSTICS_VIEW,
    "agent_sessions": Permission.AGENT_USE,
    "agent_config": Permission.AGENT_USE,
    "agent_skills": Permission.AGENT_USE,
    "agent_skill_detail": Permission.AGENT_USE,
    "agent_session": Permission.AGENT_USE,
}
_CHANNEL_REQUIRED_FOR_SCOPED_ENDPOINTS = frozenset(
    {
        "detections_list",
        "detections_search_image",
        "detections_search_text",
        "detections_summary",
        "luxriot_prompt_settings",
    }
)
_ALL_CHANNELS_REQUIRED_ENDPOINTS = frozenset(
    {
        "luxriot_stop_all_streams",
    }
)
_DEFAULT_CHANNEL_ENDPOINTS = frozenset(
    {
        "luxriot_bookmark",
        "luxriot_flush_capture",
        "luxriot_session_status",
        "luxriot_start_capture",
        "luxriot_stop_capture",
        "luxriot_stop_stream",
        "luxriot_summary_rollups",
        "probes_query",
        "probes_save",
        "probes_start_capture",
        "probes_status",
        "probes_stop_capture",
    }
)


def _json_body() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_BOOL_STRINGS:
            return True
        if normalized in FALSE_BOOL_STRINGS:
            return False
        return default
    return bool(value)


def _normalize_probe_roi_norm(raw: Any, min_side: float = PROBE_ROI_MIN_SIDE) -> Optional[Tuple[float, float, float, float]]:
    values: Optional[Tuple[float, float, float, float]] = None
    try:
        if isinstance(raw, dict):
            values = (
                _to_float(raw.get("x"), default=math.nan),
                _to_float(raw.get("y"), default=math.nan),
                _to_float(raw.get("w"), default=math.nan),
                _to_float(raw.get("h"), default=math.nan),
            )
        elif isinstance(raw, (list, tuple)) and len(raw) == 4:
            values = (
                _to_float(raw[0], default=math.nan),
                _to_float(raw[1], default=math.nan),
                _to_float(raw[2], default=math.nan),
                _to_float(raw[3], default=math.nan),
            )
    except Exception:
        return None
    if values is None:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    x, y, w, h = values
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    min_size = max(0.001, float(min_side))
    if w < min_size or h < min_size:
        return None
    if x + w > 1.0:
        x = max(0.0, 1.0 - w)
    if y + h > 1.0:
        y = max(0.0, 1.0 - h)
    return (round(x, 6), round(y, 6), round(w, 6), round(h, 6))


def _probe_roi_norm_to_payload(roi_norm: Optional[Tuple[float, float, float, float]]) -> Optional[Dict[str, float]]:
    if roi_norm is None:
        return None
    x, y, w, h = roi_norm
    return {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}


def _parse_probe_roi(payload: Mapping[str, Any]) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
    roi_enabled_explicit = "roi_enabled" in payload
    enabled = _coerce_bool(payload.get("roi_enabled"), False)
    roi_raw: Any = payload.get("roi_norm")
    legacy = payload.get("roi")
    if isinstance(legacy, dict):
        if "enabled" in legacy and "roi_enabled" not in payload:
            enabled = _coerce_bool(legacy.get("enabled"), enabled)
        if roi_raw is None:
            roi_raw = legacy.get("norm")
            if roi_raw is None and all(key in legacy for key in ("x", "y", "w", "h")):
                roi_raw = legacy
    roi_norm = _normalize_probe_roi_norm(roi_raw)
    if roi_norm is not None and not roi_enabled_explicit:
        enabled = True
    if not enabled or roi_norm is None:
        return False, None
    return True, roi_norm


def _is_local_request() -> bool:
    return (request.remote_addr or "") in LOCAL_HOSTS


def _request_admin_token() -> str:
    auth_header = (request.headers.get("Authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return (request.headers.get("X-Admin-Token") or "").strip()


def _has_admin_access() -> bool:
    configured = (config.ADMIN_TOKEN or "").strip()
    if not configured:
        return False
    provided = _request_admin_token()
    if not provided:
        return False
    return secrets.compare_digest(provided, configured)


def _auth_enabled() -> bool:
    return bool(getattr(config, "AUTH_ENABLED", False))


def _request_id() -> str:
    value = str(request.headers.get("X-Request-ID") or "").strip()
    if value and len(value) <= 128 and all(
        character.isalnum() or character in "._:-" for character in value
    ):
        return value
    return str(uuid.uuid4())


def _source_ip() -> str:
    return str(request.remote_addr or "0.0.0.0")


def _get_identity_repository() -> Any:
    global _identity_repository
    with _identity_repository_lock:
        if _identity_repository is None:
            from security.postgres_identity import PostgresIdentityRepository

            _identity_repository = PostgresIdentityRepository(
                _get_control_plane_db_pool()
            )
        return _identity_repository


def _get_auth_service() -> AuthenticationService:
    global _auth_service
    with _auth_service_lock:
        if _auth_service is None:
            if not str(getattr(config, "AUTH_TENANT_ID", "") or "").strip():
                raise RuntimeError("EVOSSEARCH_AUTH_TENANT_ID is required")
            from security.postgres_throttling import PostgresLoginThrottleRepository
            from security.throttling import LoginThrottleService

            throttle = LoginThrottleService(
                PostgresLoginThrottleRepository(
                    _get_control_plane_db_pool(),
                    config.AUTH_TENANT_ID,
                )
            )
            _auth_service = AuthenticationService(
                _get_identity_repository(),
                tenant_id=config.AUTH_TENANT_ID,
                session_ttl=timedelta(
                    hours=int(config.AUTH_SESSION_TTL_HOURS)
                ),
                throttle=throttle,
            )
        return _auth_service


def _get_audit_writer() -> Any:
    global _audit_writer
    with _audit_writer_lock:
        if _audit_writer is None:
            from security.postgres_audit import PostgresAuditWriter

            _audit_writer = PostgresAuditWriter(_get_audit_db_pool())
        return _audit_writer


def _audit_database_dsn() -> str:
    return str(
        os.getenv("EVA_AUDIT_DATABASE_DSN")
        or os.getenv("EVOSSEARCH_AUDIT_DATABASE_DSN")
        or ""
    ).strip()


def _get_audit_db_pool() -> PsycopgPool:
    global _audit_db_pool
    with _audit_db_pool_lock:
        if _audit_db_pool is None:
            dsn = _audit_database_dsn()
            if not dsn:
                raise RuntimeError(
                    "EVA_AUDIT_DATABASE_DSN is required for durable audit"
                )
            base_settings = DatabaseSettings.from_env()
            _audit_db_pool = PsycopgPool(
                replace(
                    base_settings,
                    dsn=dsn,
                    pool_min_size=0,
                    pool_max_size=min(4, base_settings.pool_max_size),
                    application_name="eva-ai-audit",
                )
            )
        return _audit_db_pool


def _current_auth_context() -> Optional[AuthContext]:
    context = getattr(g, "auth_context", None)
    return context if isinstance(context, AuthContext) else None


def _probe_channel_ids(probe_ids: Iterable[Any]) -> Set[int]:
    wanted = {
        str(probe_id).strip()
        for probe_id in probe_ids
        if str(probe_id or "").strip()
    }
    if not wanted:
        return set()
    try:
        probes = probes_store.list_probes()
    except Exception:
        return set()
    return {
        int(probe["channel_id"])
        for probe in probes
        if str(probe.get("id") or "") in wanted
        and _to_optional_int(probe.get("channel_id")) is not None
    }


def _request_channel_ids() -> Set[int]:
    candidates: List[Any] = []
    view_args = request.view_args or {}
    candidates.extend(
        view_args.get(key) for key in ("channel_id", "channel") if key in view_args
    )
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        candidates.extend(
            payload.get(key) for key in ("channel_id", "channel") if key in payload
        )
    form = request.form
    candidates.extend(
        form.get(key) for key in ("channel_id", "channel") if key in form
    )
    candidates.extend(
        request.args.get(key) for key in ("channel_id", "channel") if key in request.args
    )
    channel_ids: Set[int] = set()
    for candidate in candidates:
        try:
            channel_id = int(candidate)
        except (TypeError, ValueError):
            continue
        if channel_id > 0:
            channel_ids.add(channel_id)

    endpoint = str(request.endpoint or "")
    probe_ids: List[Any] = []
    for source in (view_args, payload, form, request.args):
        if not isinstance(source, Mapping):
            continue
        if "probe_id" in source:
            probe_ids.append(source.get("probe_id"))
        if endpoint in {"probes_delete", "probes_run", "probes_save"}:
            if "id" in source:
                probe_ids.append(source.get("id"))
    channel_ids.update(_probe_channel_ids(probe_ids))

    if endpoint == "serve_detection_image":
        image_path = str(request.args.get("image_path") or "").strip()
        if image_path:
            try:
                channel_ids.update(
                    detections_store.channel_ids_for_image_path(image_path)
                )
            except Exception:
                pass

    if not channel_ids and endpoint in _DEFAULT_CHANNEL_ENDPOINTS:
        default_channel = _to_optional_int(config.LUXRIOT_DEFAULT_CHANNEL_ID)
        if default_channel is not None and default_channel > 0:
            channel_ids.add(default_channel)
    return channel_ids


def _audit_scope_details(
    details: Mapping[str, Any],
    channel_ids: Set[int],
) -> Dict[str, Any]:
    scoped = dict(details)
    if len(channel_ids) > 1:
        scoped["channel_ids"] = sorted(channel_ids)
    return scoped


def _channel_for_audit(channel_ids: Set[int]) -> Optional[int]:
    return next(iter(channel_ids)) if len(channel_ids) == 1 else None


def _is_channel_scoped(context: AuthContext) -> bool:
    return ALL_CHANNELS not in context.allowed_channel_ids


def _can_access_context_channel(
    context: Optional[AuthContext],
    channel_id: Any,
) -> bool:
    if context is None:
        return True
    normalized = _to_optional_int(channel_id)
    if normalized is None:
        return False
    try:
        require_channel_access(context, normalized)
    except PermissionError:
        return False
    return True


def _write_security_audit(
    *,
    context: Optional[AuthContext],
    action: str,
    result: str,
    target_type: str,
    target_id: Optional[str] = None,
    channel_id: Optional[int] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    event = AuditEventBuilder().build(
        context=context,
        tenant_id=(
            None
            if context is not None
            else str(getattr(config, "AUTH_TENANT_ID", "") or "").strip() or None
        ),
        source_ip=_source_ip(),
        action=action,
        target_type=target_type,
        target_id=target_id,
        channel_id=channel_id,
        result=result,
        details=details,
    )
    _get_audit_writer().write(event)


def _write_agent_tool_audit(event: ToolAuditEvent) -> None:
    details = {
        "phase": event.phase,
        "operation": event.operation,
        "risk": event.risk,
        "required_permission": event.required_permission,
        "arguments_hash": event.arguments_hash,
        "code": event.code,
        "duration_ms": event.duration_ms,
        **dict(event.details),
    }
    audit_event = AuditEvent(
        timestamp=event.timestamp,
        request_id=event.request_id,
        actor_user_id=event.actor_id,
        actor_roles=tuple(sorted(event.actor_roles)),
        tenant_id=event.tenant_id,
        source_ip=event.source_ip or "0.0.0.0",
        action=f"agent.tool.{event.operation}.{event.tool_name}",
        target_type="agent_tool",
        target_id=event.tool_name,
        channel_id=None,
        result=event.phase,
        details=details,
    )
    _get_audit_writer().write(audit_event)


def _auth_failure_response(message: str, status: int):
    return jsonify({"success": False, "error": message}), status


def _session_guard(
    *,
    permission: Optional[Permission],
    require_csrf: bool,
    action: str,
):
    context = _current_auth_context()
    channel_ids = _request_channel_ids()
    channel_id = _channel_for_audit(channel_ids)
    if getattr(g, "auth_resolution_error", None):
        return _auth_failure_response("Authentication service unavailable", 503)
    if context is None:
        try:
            _write_security_audit(
                context=None,
                action=action,
                result="denied",
                target_type="route",
                target_id=request.path,
                channel_id=channel_id,
                details=_audit_scope_details(
                    {
                        "method": request.method,
                        "reason": "authentication_required",
                    },
                    channel_ids,
                ),
            )
        except Exception:
            return _auth_failure_response("Audit service unavailable", 503)
        return _auth_failure_response("Authentication required", 401)

    if require_csrf:
        session_record = getattr(g, "auth_session", None)
        cookie_token = str(
            request.cookies.get(config.AUTH_CSRF_COOKIE) or ""
        )
        header_token = str(request.headers.get("X-CSRF-Token") or "")
        try:
            valid_csrf = bool(
                session_record
                and _get_auth_service().validate_csrf(
                    session_record,
                    cookie_token=cookie_token,
                    header_token=header_token,
                )
            )
        except Exception:
            return _auth_failure_response("Authentication service unavailable", 503)
        if not valid_csrf:
            try:
                _write_security_audit(
                    context=context,
                    action=action,
                    result="denied",
                    target_type="route",
                    target_id=request.path,
                    channel_id=channel_id,
                    details=_audit_scope_details(
                        {"method": request.method, "reason": "csrf_rejected"},
                        channel_ids,
                    ),
                )
            except Exception:
                return _auth_failure_response("Audit service unavailable", 503)
            return _auth_failure_response("CSRF token required", 403)

    try:
        if permission is not None:
            require_permission(context, permission)
        if (
            _is_channel_scoped(context)
            and str(request.endpoint or "") in _ALL_CHANNELS_REQUIRED_ENDPOINTS
        ):
            raise PermissionError("all-channel access is required")
        if (
            _is_channel_scoped(context)
            and str(request.endpoint or "")
            in _CHANNEL_REQUIRED_FOR_SCOPED_ENDPOINTS
            and not channel_ids
        ):
            raise PermissionError("an explicit authorized channel is required")
        if (
            _is_channel_scoped(context)
            and str(request.endpoint or "") == "serve_detection_image"
            and not channel_ids
        ):
            raise PermissionError(
                "detection image ownership metadata is required"
            )
        for requested_channel_id in channel_ids:
            require_channel_access(context, requested_channel_id)
    except PermissionError as exc:
        try:
            _write_security_audit(
                context=context,
                action=action,
                result="denied",
                target_type="route",
                target_id=request.path,
                channel_id=channel_id,
                details=_audit_scope_details(
                    {"method": request.method, "reason": type(exc).__name__},
                    channel_ids,
                ),
            )
        except Exception:
            return _auth_failure_response("Audit service unavailable", 503)
        return _auth_failure_response("Access denied", 403)

    try:
        _write_security_audit(
            context=context,
            action=action,
            result="success",
            target_type="route",
            target_id=request.path,
            channel_id=channel_id,
            details=_audit_scope_details(
                {"method": request.method, "phase": "authorized"},
                channel_ids,
            ),
        )
    except Exception:
        return _auth_failure_response("Audit service unavailable", 503)
    return None


@app.before_request
def _bind_request_security_context() -> None:
    g.request_id = _request_id()
    g.auth_context = None
    g.auth_session = None
    g.auth_resolution_error = None
    if not _auth_enabled():
        return
    session_token = str(
        request.cookies.get(config.AUTH_SESSION_COOKIE) or ""
    )
    if not session_token:
        resolved = None
    else:
        try:
            resolved = _get_auth_service().resolve(
                session_token,
                request_id=g.request_id,
            )
            if resolved is not None:
                g.auth_session, g.auth_context = resolved
        except Exception as exc:
            g.auth_resolution_error = type(exc).__name__

    permission = _SENSITIVE_ENDPOINT_PERMISSIONS.get(str(request.endpoint or ""))
    if permission is not None:
        return _session_guard(
            permission=permission,
            require_csrf=request.method not in {"GET", "HEAD", "OPTIONS"},
            action=f"http.{request.endpoint}.access",
        )


@app.after_request
def _attach_request_security_headers(response):
    response.headers["X-Request-ID"] = str(
        getattr(g, "request_id", "") or uuid.uuid4()
    )
    return response


def _settings_guard(write: bool = False):
    if _auth_enabled():
        return _session_guard(
            permission=(
                Permission.SETTINGS_MANAGE
                if write
                else Permission.SETTINGS_VIEW
            ),
            require_csrf=write,
            action=(
                "settings.write.authorize"
                if write
                else "settings.read"
            ),
        )
    if write:
        return _mutation_guard()
    if config.ADMIN_TOKEN:
        if _has_admin_access():
            return None
        return jsonify({"success": False, "error": "Admin token required"}), 401
    if config.SETTINGS_LOCAL_ONLY and not _is_local_request():
        return jsonify({"success": False, "error": "Remote settings access is disabled."}), 403
    return None


def _mutation_guard():
    if _auth_enabled():
        endpoint = str(request.endpoint or "")
        permission = (
            _MUTATION_ENDPOINT_PERMISSIONS[endpoint]
            if endpoint in _MUTATION_ENDPOINT_PERMISSIONS
            else Permission.SETTINGS_MANAGE
        )
        return _session_guard(
            permission=permission,
            require_csrf=True,
            action=f"http.{endpoint or 'unknown'}.mutate",
        )
    configured = (config.ADMIN_TOKEN or "").strip()
    if not configured:
        return jsonify(
            {
                "success": False,
                "error": "Admin token is required for mutating endpoints. Set EVOSSEARCH_ADMIN_TOKEN.",
            }
        ), 503
    if _has_admin_access():
        return None
    return jsonify({"success": False, "error": "Admin token required"}), 401


def _mutation_guard_error():
    guard = _mutation_guard()
    if guard is None:
        return None
    body, status = guard
    payload = body.get_json(silent=True) if hasattr(body, "get_json") else {}
    message = (payload or {}).get("error") or "Admin token required"
    return jsonify({"error": message}), status


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_folder_path(folder_raw: Any, require_index: bool = False) -> Path:
    folder_text = str(folder_raw or "").strip()
    if not folder_text:
        raise ValueError("No folder specified")
    folder_path = Path(folder_text).expanduser().resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError("Invalid folder path")
    if config.ALLOWED_ROOTS:
        allowed_roots = [Path(item).expanduser().resolve() for item in config.ALLOWED_ROOTS]
        if not any(_path_within(folder_path, root) for root in allowed_roots):
            raise ValueError("Folder path is outside configured allowed roots")
    if require_index:
        index_root = folder_path / config.INDEX_FOLDER_NAME
        has_index = (index_root / "index.faiss").exists() or any(
            (index_root / subdir / "index.faiss").exists() for subdir in EMBEDDER_SUBDIRS.values()
        )
        if not has_index:
            raise ValueError("Folder is not indexed")
    return folder_path


@app.errorhandler(413)
def payload_too_large(_: Exception):
    return jsonify({"error": f"Payload too large (max {config.MAX_FILE_SIZE_MB} MB)."}), 413


def init_clip() -> None:
    """Load the CLIP-like model lazily for embedding extraction."""
    global clip_model, clip_preprocess, clip_processor, clip_backend_kind, clip_runtime_model, clip_runtime_device
    if clip_model is not None:
        if clip_backend_kind == "openai_clip" and clip_preprocess is not None:
            return
        if clip_backend_kind == "siglip2" and clip_processor is not None:
            return

    preferred_device = device
    requested_model = str(config.CLIP_MODEL or "").strip() or "ViT-B/32"
    if _is_siglip2_clip_model(requested_model):
        try:
            model_obj, processor_obj = _load_siglip2_clip_model(requested_model, preferred_device)
            clip_model = model_obj
            clip_processor = processor_obj
            clip_preprocess = None
            clip_backend_kind = "siglip2"
            clip_runtime_model = requested_model
            clip_runtime_device = preferred_device
            return
        except Exception as exc:
            fallback_model = "ViT-B/32"
            print(
                f"SigLIP2 model '{requested_model}' failed to load ({exc}). "
                f"Falling back to CLIP '{fallback_model}'."
            )
            fallback_error: Optional[Exception] = None
            fallback_device = preferred_device
            if fallback_device.startswith("cuda"):
                _release_cuda_memory()
            try:
                model_obj, preprocess_obj = _load_openai_clip_model(fallback_model, fallback_device)
            except Exception as fallback_exc:
                fallback_error = fallback_exc
                if fallback_device.startswith("cuda"):
                    _release_cuda_memory()
                    fallback_device = "cpu"
                    model_obj, preprocess_obj = _load_openai_clip_model(fallback_model, fallback_device)
                else:
                    raise
            clip_model = model_obj
            clip_preprocess = preprocess_obj
            clip_processor = None
            clip_backend_kind = "openai_clip"
            clip_runtime_model = fallback_model
            clip_runtime_device = fallback_device
            if fallback_error is not None:
                print(f"CLIP fallback recovered on {fallback_device} after initial failure: {fallback_error}")
            return

    fallback_device = preferred_device
    initial_error: Optional[Exception] = None
    try:
        model_obj, preprocess_obj = _load_openai_clip_model(requested_model, fallback_device)
    except Exception as exc:
        initial_error = exc
        if fallback_device.startswith("cuda"):
            _release_cuda_memory()
            fallback_device = "cpu"
            model_obj, preprocess_obj = _load_openai_clip_model(requested_model, fallback_device)
        else:
            raise
    clip_model = model_obj
    clip_preprocess = preprocess_obj
    clip_processor = None
    clip_backend_kind = "openai_clip"
    clip_runtime_model = requested_model
    clip_runtime_device = fallback_device
    if initial_error is not None:
        print(f"CLIP model '{requested_model}' loaded on {fallback_device} after retry: {initial_error}")


def _is_siglip2_clip_model(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    return "siglip2" in normalized


def _release_cuda_memory() -> None:
    if not torch.cuda.is_available():
        return
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _load_openai_clip_model(model_name: str, target_device: str) -> Tuple[torch.nn.Module, Any]:
    model, preprocess = clip.load(model_name, device=target_device)
    cast(torch.nn.Module, model).eval()
    return cast(torch.nn.Module, model), preprocess


def _load_siglip2_clip_model(model_name: str, target_device: str) -> Tuple[torch.nn.Module, Any]:
    model = AutoModel.from_pretrained(model_name)
    cast(torch.nn.Module, model).to(target_device)
    cast(torch.nn.Module, model).eval()
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    return cast(torch.nn.Module, model), processor


def _normalize_l2_embeddings(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _processor_to_device(batch: Mapping[str, Any], target_device: str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(target_device)
        else:
            moved[key] = value
    return moved


def _clip_image_embeddings_from_pils(images: Sequence[Image.Image]) -> np.ndarray:
    ensure_embedder_loaded("clip")
    if not images:
        return np.zeros((0, 0), dtype=np.float32)

    normalized_images = [img.convert("RGB") for img in images]
    with torch.no_grad():
        if clip_backend_kind == "siglip2":
            if clip_processor is None or clip_model is None:
                raise RuntimeError("SigLIP2 clip backend is not initialized")
            processor_inputs = cast(Any, clip_processor)(images=normalized_images, return_tensors="pt")
            model_inputs = _processor_to_device(cast(Mapping[str, Any], processor_inputs), clip_runtime_device)
            image_features = cast(Any, clip_model).get_image_features(**model_inputs)
        else:
            if clip_preprocess is None or clip_model is None:
                raise RuntimeError("CLIP backend is not initialized")
            image_batch = torch.stack([clip_preprocess(img) for img in normalized_images], dim=0).to(clip_runtime_device)  # type: ignore[operator]
            image_features = cast(Any, clip_model).encode_image(image_batch)
        image_features = _normalize_l2_embeddings(cast(torch.Tensor, image_features))
    return image_features.cpu().numpy().astype(np.float32, copy=False)


def _clip_text_embeddings(texts: Sequence[str]) -> np.ndarray:
    ensure_embedder_loaded("clip")
    prepared = [str(text or "").strip() for text in texts if str(text or "").strip()]
    if not prepared:
        return np.zeros((0, 0), dtype=np.float32)

    with torch.no_grad():
        if clip_backend_kind == "siglip2":
            if clip_processor is None or clip_model is None:
                raise RuntimeError("SigLIP2 clip backend is not initialized")
            # SigLIP2 tokenization quality is better when text is lower-cased and max_length=64.
            normalized_texts = [text.lower() for text in prepared]
            processor_inputs = cast(Any, clip_processor)(
                text=normalized_texts,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            model_inputs = _processor_to_device(cast(Mapping[str, Any], processor_inputs), clip_runtime_device)
            text_features = cast(Any, clip_model).get_text_features(**model_inputs)
        else:
            if clip_model is None:
                raise RuntimeError("CLIP backend is not initialized")
            text_tokens = clip.tokenize(prepared).to(clip_runtime_device)
            text_features = cast(Any, clip_model).encode_text(text_tokens)
        text_features = _normalize_l2_embeddings(cast(torch.Tensor, text_features))
    return text_features.cpu().numpy().astype(np.float32, copy=False)


def init_dino() -> None:
    """Load the DINO encoder lazily."""
    global dino_encoder
    if dino_encoder is not None:
        return
    weights_path = (config.DINO_WEIGHTS_PATH or "").strip()
    if not weights_path:
        raise RuntimeError(
            "EVOSSEARCH_DINO_WEIGHTS_PATH is not set. Provide a local DINO checkpoint to keep inference on GPU."
        )
    weights_file = Path(weights_path).expanduser()
    if not weights_file.exists():
        raise FileNotFoundError(f"DINO weights file not found: {weights_file}")
    config.DINO_WEIGHTS_PATH = str(weights_file.resolve())

    device_hint = (config.DINO_DEVICE or "").strip()
    if not device_hint:
        if torch.cuda.is_available():
            device_hint = "cuda:0"
        else:
            raise RuntimeError("CUDA device required for DINO encoder; none detected.")
    if not device_hint.startswith("cuda"):
        raise RuntimeError("DINO encoder must run on CUDA. Set EVOSSEARCH_DINO_DEVICE accordingly.")
    config.DINO_DEVICE = device_hint

    dino_encoder = DINOEncoder(
        model_name=config.DINO_MODEL,
        batch_size=config.BATCH_SIZE,
        weights_path=config.DINO_WEIGHTS_PATH,
        device=config.DINO_DEVICE,
    )


def ensure_embedder_loaded(embedder: Optional[str] = None) -> None:
    target = embedder or active_embedder
    if target == "clip":
        init_clip()
    elif target == "dino":
        init_dino()
    elif target == "fusion":
        init_clip()
        init_dino()
    else:
        raise ValueError(f"Unsupported embedder: {target}")


def reset_embedder_runtime_state() -> None:
    """Clear loaded embedding backends so they can be re-initialized."""
    global clip_model, clip_preprocess, clip_processor, clip_backend_kind, clip_runtime_model, clip_runtime_device, dino_encoder
    clip_model = None
    clip_preprocess = None
    clip_processor = None
    clip_backend_kind = "openai_clip"
    clip_runtime_model = ""
    clip_runtime_device = device
    dino_encoder = None


def warm_start_embedder() -> Optional[str]:
    """
    Warm start embedding backend without aborting process on heavy model failures.

    In fusion mode we intentionally warm only CLIP/SigLIP and defer DINO to first DINO/fusion call.
    This prevents startup crashes when GPU memory is tight.
    """
    global active_embedder
    requested = active_embedder
    try:
        if requested == "fusion":
            ensure_embedder_loaded("clip")
            return "Fusion warm-up loaded CLIP backend; DINO will load on demand."
        ensure_embedder_loaded(requested)
        return None
    except Exception as exc:
        if requested != "clip":
            try:
                ensure_embedder_loaded("clip")
                active_embedder = "clip"
                return f"{requested} warm-up failed ({exc}); fell back to CLIP backend."
            except Exception as clip_exc:
                return f"Embedder warm-up failed ({exc}); CLIP fallback also failed ({clip_exc})."
        return f"Embedder warm-up failed: {exc}"


def ensure_mask_head() -> Optional["Mask2FormerHead"]:
    global mask2former_head, _mask2former_failed
    if not config.MASK2FORMER_ENABLED:
        return None
    if _Mask2FormerHeadRuntime is None:
        if not _mask2former_failed:
            print("Mask2Former head unavailable: transformers vision deps missing; disabling head.")
            _mask2former_failed = True
        return None
    if mask2former_head is not None:
        return mask2former_head
    with _mask2former_lock:
        if mask2former_head is not None:
            return mask2former_head
        try:
            target_device = config.MASK2FORMER_DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
            created_head = _Mask2FormerHeadRuntime(
                model_name=config.MASK2FORMER_MODEL,
                device=target_device,
                max_size=config.MASK2FORMER_MAX_SIZE,
            )
            mask2former_head = cast(Any, created_head)
        except Exception as exc:
            _mask2former_failed = True
            print(f"Mask2Former head initialization failed: {exc}")
            config.MASK2FORMER_ENABLED = False
            return None
    return mask2former_head


def _faiss_add_vectors(index: faiss.Index, vectors: np.ndarray) -> None:
    """Typed wrapper for FAISS add; FAISS runtime monkey-patches signatures."""
    cast(Any, index).add(vectors)


def _faiss_search(index: faiss.Index, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Typed wrapper for FAISS search; returns (distances, indices)."""
    query_arr = np.asarray(query, dtype=np.float32)
    if query_arr.ndim != 2:
        raise ValueError("FAISS query must be a 2D float32 array")
    try:
        index_dim = int(getattr(index, "d"))
    except Exception:
        index_dim = None
    if index_dim is not None and int(query_arr.shape[1]) != index_dim:
        raise ValueError(
            f"Embedding dimension mismatch: query dim {int(query_arr.shape[1])} vs index dim {index_dim}. "
            "Rebuild index for the selected CLIP model."
        )
    distances, labels = cast(Any, index).search(query_arr, int(k))
    return np.asarray(distances), np.asarray(labels)


def _faiss_reconstruct(index: faiss.Index, idx: int) -> np.ndarray:
    """Typed wrapper for FAISS reconstruct."""
    return np.asarray(cast(Any, index).reconstruct(int(idx)), dtype=np.float32)


def get_image_embedding(image_path: Union[str, Path], embedder: Optional[str] = None) -> np.ndarray:
    """Extract an embedding for an image path using the selected backend."""
    target = embedder or active_embedder
    if target == "fusion":
        target = "clip"
    ensure_embedder_loaded(target)
    if target == "clip":
        pil_image = Image.open(image_path).convert("RGB")
        embeddings = _clip_image_embeddings_from_pils([pil_image])
        if embeddings.size == 0:
            raise RuntimeError("Failed to produce clip embedding from image")
        return embeddings[0]
    else:
        assert dino_encoder is not None
        features = dino_encoder.encode_images(image_path)
        return features[0]


def get_image_embedding_from_pil(pil_image: Image.Image, embedder: Optional[str] = None) -> np.ndarray:
    """Extract an embedding from a PIL image using the selected backend."""
    target = embedder or active_embedder
    if target == "fusion":
        target = "clip"
    ensure_embedder_loaded(target)
    if target == "clip":
        embeddings = _clip_image_embeddings_from_pils([pil_image])
        if embeddings.size == 0:
            raise RuntimeError("Failed to produce clip embedding from image")
        return embeddings[0]
    else:
        assert dino_encoder is not None
        features = dino_encoder.encode_images(pil_image)
        return features[0]


def get_clip_text_embedding(text: str) -> np.ndarray:
    """Extract a text embedding from the CLIP-like backend, including SigLIP2."""
    embeddings = _clip_text_embeddings([text])
    if embeddings.size == 0:
        raise RuntimeError("Text query is empty")
    return embeddings[0]


def get_text_embedding(text: str) -> np.ndarray:
    """Extract a CLIP text embedding. Only available when CLIP or fusion backend is active."""
    if active_embedder not in {"clip", "fusion"}:
        raise RuntimeError("Text search is only supported when the CLIP backend is active.")
    return get_clip_text_embedding(text)


def get_probe_image_embedding_from_pil(pil_image: Image.Image) -> np.ndarray:
    """Probe image embeddings always use the CLIP-like backend so SigLIP2 can drive probe matching."""
    return get_image_embedding_from_pil(pil_image, embedder="clip")


def get_probe_text_embedding(text: str) -> np.ndarray:
    """Probe text embeddings always use the CLIP-like backend so they remain available outside clip search mode."""
    return get_clip_text_embedding(text)


def _build_index_metadata(embedder: str, additional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ensure_embedder_loaded(embedder)
    base: Dict[str, Any]
    if embedder == "clip":
        runtime_model = str(clip_runtime_model or config.CLIP_MODEL or "unknown")
        if clip_backend_kind == "siglip2":
            projection_dim = getattr(getattr(clip_model, "config", None), "projection_dim", 1152)
            embed_dim = int(projection_dim if projection_dim else 1152)
            library = "google/siglip2"
        else:
            embed_dim = int(getattr(getattr(clip_model, "visual", None), "output_dim", 512))  # type: ignore[union-attr]
            library = "openai/CLIP"
        base = {
            "embedder": "clip",
            "model": runtime_model,
            "requested_model": str(config.CLIP_MODEL),
            "embedding_dim": embed_dim,
            "library": library,
            "backend": clip_backend_kind,
            "device": clip_runtime_device,
        }
    else:
        assert dino_encoder is not None
        base = {
            "embedder": "dino",
            "model": dino_encoder.metadata.name,
            "embedding_dim": dino_encoder.metadata.embed_dim,
            "config_model": config.DINO_MODEL,
            "library": "facebookresearch/dinov3",
            "device": str(dino_encoder.device),
        }
    if additional:
        base.update(additional)
    base.setdefault("created_at", time.time())
    return base


def _index_targets() -> List[str]:
    if config.INDEX_MODE == 'dual':
        return list(EMBEDDER_SUBDIRS.keys())
    return [config.INDEX_MODE]


def _collect_image_entries(folder_path: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    entries: List[Tuple[Path, Dict[str, Any]]] = []
    for ext in config.SUPPORTED_EXTENSIONS:
        for img_path in sorted(folder_path.glob(f'*{ext}')):
            try:
                stat = img_path.stat()
            except FileNotFoundError:
                continue
            metadata = {
                'path': str(img_path),
                'mtime': stat.st_mtime,
                'size': stat.st_size,
            }
            entries.append((img_path, metadata))
    return entries


def _create_index_for_embedder(entries: List[Tuple[Path, Dict[str, Any]]], embedder: str) -> Optional[Tuple[faiss.Index, List[str], List[Dict[str, Any]], Dict[str, Any]]]:
    image_paths: List[str] = []
    image_metadata: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []

    for img_path, metadata in entries:
        try:
            embedding = get_image_embedding(img_path, embedder=embedder)
        except Exception as exc:
            print(f"Error processing {img_path} for {embedder}: {exc}")
            continue
        embeddings.append(embedding)
        image_paths.append(str(img_path))
        image_metadata.append(metadata)

    if not embeddings:
        return None

    embeddings_array = np.array(embeddings, dtype='float32')
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    _faiss_add_vectors(index, embeddings_array)
    index_meta = _build_index_metadata(embedder, {'image_count': len(image_paths)})
    return index, image_paths, image_metadata, index_meta


def create_index(folder_path):
    """Create FAISS indexes for the configured embedder targets."""
    folder_path = Path(folder_path)
    entries = _collect_image_entries(folder_path)
    if not entries:
        return {}

    results: Dict[str, Tuple[faiss.Index, List[str], List[Dict[str, Any]], Dict[str, Any]]] = {}
    for embedder in _index_targets():
        if embedder not in EMBEDDER_SUBDIRS:
            continue
        data = _create_index_for_embedder(entries, embedder)
        if data is not None:
            results[embedder] = data
    return results


@app.route('/')
def home():
    """Serve the frontend."""
    min_val, max_val, default_val = config.MIN_RESULTS, config.MAX_RESULTS, config.DEFAULT_RESULTS
    options: Set[int] = {min_val, default_val, max_val}
    if max_val <= 20:
        options.update(i for i in range(min_val, max_val + 1) if i % 2 == 0 or i % 3 == 0)
    else:
        options.update(i for i in [6, 12, 18, 24, 30] if min_val <= i <= max_val)
    result_options_html = '\n                            '.join(
        f'<option value="{i}" {"selected" if i == default_val else ""}>{i}</option>'
        for i in sorted(options)
    )

    luxriot_batch_options = '\n                            '.join(
        f'<option value="{size}" {"selected" if size == config.LUXRIOT_BATCH_SIZES[0] else ""}>{size}</option>'
        for size in config.LUXRIOT_BATCH_SIZES
    )

    default_video_frames = max(1, int(config.LM_VIDEO_DEFAULT_FRAMES))
    max_video_frames = max(default_video_frames, int(config.LM_VIDEO_MAX_FRAMES))
    video_frame_options_set: Set[int] = {default_video_frames}
    for _raw in getattr(config, "LM_VIDEO_FRAME_OPTIONS", ()):
        try:
            _opt = int(_raw)
        except (TypeError, ValueError):
            continue
        if 1 <= _opt <= max_video_frames:
            video_frame_options_set.add(_opt)
    video_frame_options_html = '\n                            '.join(
        f'<option value="{count}" {"selected" if count == default_video_frames else ""}>{count}</option>'
        for count in sorted(video_frame_options_set)
    )

    rollup_default = str(
        getattr(config, "LUXRIOT_ROLLUP_LLM_SYSTEM_PROMPT",
                getattr(config, "LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT", "")) or ""
    )

    response = make_response(render_template(
        'index.html',
        timestamp=str(int(time.time())),
        app_version=config.APP_VERSION or '',
        result_options_html=result_options_html,
        segment_threshold_percent=min(99, max(40, int(round(float(config.DINO_HEATMAP_THRESHOLD) * 100)))),
        segments_enabled_checked='checked' if config.DINO_SEGMENTS_ENABLED else '',
        segment_min_patches_default=max(1, int(config.DINO_SEGMENT_MIN_PATCHES)),
        luxriot_batch_options=luxriot_batch_options,
        luxriot_snapshot_interval=config.LUXRIOT_SNAPSHOT_INTERVAL,
        luxriot_snapshot_max_edge=config.LUXRIOT_SNAPSHOT_MAX_EDGE,
        video_frame_options_html=video_frame_options_html,
        luxriot_system_prompt_default=LUXRIOT_SYSTEM_PROMPT_DEFAULT or '',
        luxriot_rollup_prompt_l1=getattr(config, 'LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT', rollup_default) or rollup_default,
        luxriot_rollup_prompt_l2=getattr(config, 'LUXRIOT_ROLLUP_L2_SYSTEM_PROMPT', rollup_default) or rollup_default,
        luxriot_rollup_prompt_l3=getattr(config, 'LUXRIOT_ROLLUP_L3_SYSTEM_PROMPT', rollup_default) or rollup_default,
        luxriot_json_alert_prompt=getattr(config, 'LUXRIOT_ALERTS_JSON_PROMPT', LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT) or LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT,
        auth_enabled=bool(config.AUTH_ENABLED),
    ))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/favicon.ico')
def favicon():
    """Serve favicon if present; otherwise return no-content to avoid noisy 404 logs."""
    icon_path = Path(__file__).resolve().parent / 'static' / 'images' / 'favicon.ico'
    if icon_path.exists():
        return send_file(icon_path, mimetype='image/x-icon', max_age=86400)
    return ('', 204)


@app.route('/branding/logo')
def branding_logo():
    """Serve application branding logo."""
    logo_path = Path(__file__).resolve().parent / 'static' / 'images' / 'lxrt-logo-darktheme.png'
    if logo_path.exists():
        return send_file(logo_path, mimetype='image/png', max_age=86400)
    return ('', 204)


@app.route('/js/app.js')
def serve_app_js():
    """Serve app.js with runtime config values injected (5 Luxriot defaults)."""
    js_path = Path(__file__).resolve().parent / 'static' / 'js' / 'app.js'
    js = js_path.read_text(encoding='utf-8')
    luxriot_default_batch = config.LUXRIOT_BATCH_SIZES[0] if config.LUXRIOT_BATCH_SIZES else 12
    js = js.replace('{luxriot_default_channel}', str(config.LUXRIOT_DEFAULT_CHANNEL_ID))
    js = js.replace('{luxriot_base_url_json}', json.dumps(str(config.LUXRIOT_BASE_URL or "")))
    js = js.replace('{luxriot_snapshot_interval}', str(config.LUXRIOT_SNAPSHOT_INTERVAL))
    js = js.replace('{luxriot_snapshot_max_edge}', str(config.LUXRIOT_SNAPSHOT_MAX_EDGE))
    js = js.replace('{luxriot_batch_default}', str(luxriot_default_batch))
    js = js.replace('{auth_enabled_json}', json.dumps(bool(config.AUTH_ENABLED)))
    js = js.replace(
        '{auth_csrf_cookie_json}',
        json.dumps(str(config.AUTH_CSRF_COOKIE)),
    )
    response = make_response(js)
    response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


@app.route('/image', methods=['GET'])
@app.route('/image/<path:filepath>', methods=['GET'])
def serve_image(filepath: str = ""):
    """Serve image files from indexed folders and keep legacy detection-archive URLs working."""
    try:
        folder_raw = request.args.get('folder')
        source_path = request.args.get('image_path') or filepath
        if not source_path:
            return "Missing image path", 400

        decoded = unquote(source_path)
        path_obj = Path(decoded)
        folder_path: Optional[Path] = None
        if folder_raw:
            folder_path = _resolve_folder_path(folder_raw, require_index=True)
            if not path_obj.is_absolute():
                path_obj = folder_path / path_obj
            abs_path = path_obj.resolve()
        else:
            if not path_obj.is_absolute():
                return "Missing folder parameter", 400
            abs_path = path_obj.expanduser().resolve()
        if abs_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return "Unsupported file type", 403
        if folder_path is not None:
            if not _path_within(abs_path, folder_path):
                return "Access denied", 403
        else:
            if not _path_within(abs_path, detection_archive.root):
                return "Access denied", 403
        if not abs_path.exists() or not abs_path.is_file():
            return "Image not found", 404
        return send_file(str(abs_path))
    except ValueError as exc:
        return str(exc), 400
    except Exception as exc:
        return f"Error serving image: {exc}", 500


@app.route('/detections/image', methods=['GET'])
def serve_detection_image():
    image_path = request.args.get('image_path')
    try:
        resolved = detection_archive.resolve_archive_image_path(image_path)
        return send_file(str(resolved))
    except ValueError as exc:
        message = str(exc)
        status = 404 if message.lower() == "image not found" else 400
        return message, status
    except Exception as exc:
        return f"Error serving detection image: {exc}", 500


def _index_directory(folder_path: Union[str, Path], embedder: str) -> Path:
    root = Path(folder_path) / config.INDEX_FOLDER_NAME
    return root / EMBEDDER_SUBDIRS[embedder]


def _segment_index_directory(folder_path: Union[str, Path]) -> Path:
    return _index_directory(folder_path, 'dino') / 'segments'


def _image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def _encode_jpeg(img: Image.Image, max_edge: Optional[int] = None, quality: int = 85) -> str:
    if max_edge and max(img.size) > max_edge:
        scale = max_edge / float(max(img.size))
        img = img.resize((int(img.width * scale), int(img.height * scale)), RESAMPLE_LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def _create_overlay_rgba(alpha_image: Image.Image, color: Tuple[int, int, int], opacity_scale: float = 1.0) -> Image.Image:
    alpha = alpha_image.convert('L')
    scale = float(opacity_scale)
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if scale <= 0:
        alpha_arr.fill(0.0)
    elif scale < 0.999:
        alpha_arr = np.clip(alpha_arr * scale, 0.0, 255.0)
    alpha = Image.fromarray(alpha_arr.astype(np.uint8), mode='L')
    overlay = Image.new('RGBA', alpha.size, color + (0,))
    overlay.putalpha(alpha)
    return overlay


_SEGMENT_COLOR_TABLE: Tuple[Tuple[int, int, int], ...] = (
    (244, 67, 54),
    (30, 136, 229),
    (102, 187, 106),
    (255, 202, 40),
    (171, 71, 188),
    (255, 112, 67),
    (66, 165, 245),
    (38, 166, 154),
    (156, 204, 101),
    (255, 238, 88),
    (239, 83, 80),
    (126, 87, 194),
    (0, 188, 212),
    (255, 171, 145),
    (156, 39, 176),
    (124, 179, 66),
)


def _class_color(class_id: int) -> Tuple[int, int, int]:
    table = _SEGMENT_COLOR_TABLE
    return table[class_id % len(table)]


def _sample_video_frames(
    video_path: Union[str, Path],
    max_frames: int,
    sample_fps: Optional[float],
    max_edge: int,
) -> Tuple[List[Dict[str, Any]], float, Optional[float]]:
    path_obj = Path(video_path)
    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (total_frames / fps) if fps > 0 else None
    target_frames = max(1, min(int(max_frames), config.LM_VIDEO_MAX_FRAMES))

    if total_frames > 0:
        if sample_fps and sample_fps > 0 and fps > 0:
            step = max(1, int(round(fps / sample_fps)))
            frame_indices = list(range(0, total_frames, step))
        else:
            frame_indices = list(np.linspace(0, total_frames - 1, num=target_frames, dtype=int))
        frame_indices = sorted(set(frame_indices))[:target_frames]
    else:
        frame_indices = list(range(target_frames))

    frames: List[Dict[str, Any]] = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(
                {
                    'index': int(idx),
                    'time_sec': round(idx / fps, 2) if fps > 0 else None,
                    'thumbnail': _encode_jpeg(pil_img, max_edge=max_edge, quality=85),
                    'width': pil_img.width,
                    'height': pil_img.height,
                }
            )
            if len(frames) >= target_frames:
                break
    finally:
        cap.release()
    return frames, fps, duration


def _build_video_messages(video_path: str, frames: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, Any]]:
    prompt = (user_prompt or '').strip() or "Summarize the key events, people, and objects in this video."
    intro = f"Video file: {Path(video_path).name}. {len(frames)} sampled frames are provided."
    user_content: List[Dict[str, Any]] = [{'type': 'text', 'text': f"{intro}\n\nTask: {prompt}"}]
    for idx, frame in enumerate(frames):
        ts = frame.get('time_sec')
        ts_label = f"{ts:.2f}s" if isinstance(ts, (int, float)) else "n/a"
        user_content.append({'type': 'text', 'text': f"Frame {idx + 1} (t={ts_label})"})
        user_content.append(
            {
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{frame['thumbnail']}",
                    'detail': 'high',
                },
            }
        )
    system_msg = (
        "You analyze short videos using sampled frames. Provide a concise, factual summary, key events, "
        "notable objects, people, and any scene changes. Avoid repeating the same detail for each frame."
    )
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': system_msg}]},
        {'role': 'user', 'content': user_content},
    ]


def _build_luxriot_messages(channel_label: str, frames: List[Dict[str, Any]], user_prompt: str, system_prompt: str) -> List[Dict[str, Any]]:
    prompt = (user_prompt or '').strip() or "Describe notable activity, people, vehicles, and anomalies."
    intro = (
        f"Live snapshots from Luxriot channel {channel_label}. "
        f"{len(frames)} snapshots captured roughly every {config.LUXRIOT_SNAPSHOT_INTERVAL}s."
    )
    user_content: List[Dict[str, Any]] = [{'type': 'text', 'text': f"{intro}\n\nTask: {prompt}"}]
    for idx, frame in enumerate(frames):
        ts_raw = frame.get('captured_at') or frame.get('time_sec')
        ts_label = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_raw)) if isinstance(ts_raw, (int, float)) else 'n/a'
        user_content.append({'type': 'text', 'text': f"Snapshot {idx + 1} (captured at {ts_label})"})
        thumbnail = frame.get('thumbnail')
        if thumbnail:
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{thumbnail}",
                        'detail': 'high',
                    },
                }
            )
    system_msg = system_prompt.strip() or LUXRIOT_SYSTEM_PROMPT_DEFAULT
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': system_msg}]},
        {'role': 'user', 'content': user_content},
    ]


def _call_lm_chat(messages: List[Dict[str, Any]], model_override: Optional[str] = None) -> str:
    base_url = (config.LM_BASE_URL or '').rstrip('/')
    if not base_url:
        raise RuntimeError("EVOSSEARCH_LM_BASE_URL is not configured.")
    endpoint = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.LM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LM_API_KEY}"

    target_model = (model_override or config.LM_MODEL).strip()
    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": float(config.LM_VIDEO_TEMPERATURE),
        "max_tokens": int(config.LM_VIDEO_MAX_TOKENS),
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=config.LM_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"LM Studio request failed: {exc}") from exc

    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        content_text = "\n".join(parts).strip()
    else:
        content_text = str(content).strip()
    return content_text or "(empty response from model)"


def _call_video_understanding(messages: List[Dict[str, Any]], model_override: Optional[str] = None) -> str:
    return _call_lm_chat(messages, model_override=model_override)


def _parse_lm_alerts(text: str, default_channel_id: int, default_ts_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """Extract alert objects from LM output; expects optional JSON with an alerts array."""
    import json
    import re
    now_ms = int(time.time() * 1000)
    base_ts_ms = default_ts_ms or now_ms

    def _validate_alert(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        title = (raw.get('title') or '').strip() or 'External event'
        description = (raw.get('description') or '').strip()
        severity = str(raw.get('severity') or 'critical').lower()
        allowed_sev = {'info', 'low', 'normal', 'high', 'critical'}
        if severity not in allowed_sev:
            severity = 'critical'
        state = str(raw.get('state') or 'new').lower()
        allowed_state = {'none', 'new', 'inprogress', 'closed', 'hidden'}
        if state not in allowed_state:
            state = 'new'
        channel_id = raw.get('channel_id') or default_channel_id
        try:
            channel_id = int(channel_id)
        except Exception:
            channel_id = default_channel_id
        timestamp_ms = raw.get('timestamp_ms') or base_ts_ms
        try:
            timestamp_ms = int(timestamp_ms)
        except Exception:
            timestamp_ms = base_ts_ms
        return {
            'title': title,
            'description': description,
            'severity': severity,
            'state': state,
            'channel_id': channel_id,
            'timestamp_ms': timestamp_ms,
        }

    def _extract_balanced_json(blob: str, start_idx: int) -> Optional[Tuple[str, int]]:
        if not isinstance(blob, str) or start_idx < 0 or start_idx >= len(blob):
            return None
        idx = start_idx
        while idx < len(blob) and blob[idx] != '{':
            idx += 1
        if idx >= len(blob) or blob[idx] != '{':
            return None
        depth = 0
        in_string = False
        escaped = False
        end_idx = idx
        while end_idx < len(blob):
            ch = blob[end_idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return blob[idx:end_idx + 1], end_idx + 1
            end_idx += 1
        return None

    def _extract_candidates(blob: str) -> List[Any]:
        candidates: List[Any] = []
        seen: Set[str] = set()

        def _add_candidate(raw: Any) -> None:
            try:
                key = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            except Exception:
                key = repr(raw)
            if key in seen:
                return
            seen.add(key)
            candidates.append(raw)

        try:
            parsed = json.loads(blob)
            _add_candidate(parsed)
        except Exception:
            pass

        for match in re.finditer(r"```json(.*?)```", blob, flags=re.DOTALL | re.IGNORECASE):
            try:
                _add_candidate(json.loads(match.group(1)))
            except Exception:
                continue

        marker = "ALERTS_JSON:"
        lowered = blob.lower()
        marker_lower = marker.lower()
        search_pos = 0
        while True:
            marker_idx = lowered.find(marker_lower, search_pos)
            if marker_idx < 0:
                break
            start_idx = marker_idx + len(marker)
            chunk = _extract_balanced_json(blob, start_idx)
            if chunk:
                json_blob, next_idx = chunk
                try:
                    _add_candidate(json.loads(json_blob))
                except Exception:
                    pass
                search_pos = max(next_idx, marker_idx + 1)
            else:
                search_pos = marker_idx + 1

        for match in re.finditer(r"\{\s*\"alerts\"\s*:", blob, flags=re.IGNORECASE):
            chunk = _extract_balanced_json(blob, match.start())
            if not chunk:
                continue
            json_blob, _ = chunk
            try:
                _add_candidate(json.loads(json_blob))
            except Exception:
                continue

        return candidates

    alerts: List[Dict[str, Any]] = []
    for candidate in _extract_candidates(text or ''):
        if isinstance(candidate, dict) and isinstance(candidate.get('alerts'), list):
            for raw_alert in candidate['alerts']:
                validated = _validate_alert(raw_alert)
                if validated:
                    alerts.append(validated)
            if alerts:
                break

    return alerts


PROBE_MAX_STORED_HITS = getattr(config, 'PROBE_MAX_STORED_HITS', 30)
PROBE_DAEMON_INTERVAL_SEC = getattr(config, 'PROBE_DAEMON_INTERVAL_SEC', 5)
PROBE_BENCH_BATCH = getattr(config, 'PROBE_BENCH_BATCH', 16)
LEGACY_LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT = (
    "{\n"
    "  \"alerts\": [\n"
    "    {\n"
    "      \"title\": \"Event title\",\n"
    "      \"description\": \"twitter-sized event description\",\n"
    "      \"severity\": \"info\",\n"
    "      \"state\": \"new\",\n"
    "      \"channel_id\": \"channel ID\",\n"
    "      \"timestamp_ms\": 1772202050000\n"
    "    }\n"
    "  ]\n"
    "}"
)
LEGACY_LUXRIOT_ROLLUP_PROMPT_DEFAULT = (
    "You are a CCTV operations summarizer. Consolidate multiple short L0 summaries into one clear L1 rollup. "
    "Remove repetition, keep concrete scene changes and timestamps, and avoid boilerplate."
)
LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT = (
    "Optional bookmark output (emit only when a Task-defined trigger is observed in this batch):\n"
    "- If no trigger match: emit no JSON block.\n"
    "- If a trigger matches: append exactly one block at the end, prefixed with ALERTS_JSON:, using this schema:\n"
    "ALERTS_JSON:\n"
    "{\n"
    "  \"alerts\": [\n"
    "    {\n"
    "      \"title\": \"Short event title\",\n"
    "      \"description\": \"<= 240 chars, concrete and actionable\",\n"
    "      \"severity\": \"info|low|normal|high|critical\",\n"
    "      \"state\": \"new\",\n"
    "      \"channel_id\": {channel_id},\n"
    "      \"timestamp_ms\": 0\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "Rules: max 3 alerts; do not alert routine micro-movements unless explicitly requested; "
    "timestamp_ms should be observed batch epoch in milliseconds (or 0 if unknown)."
)
PREVIOUS_LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT = (
    "Optional bookmark output (emit only when a Task-defined trigger is observed in this batch):\n"
    "- If no trigger match: emit no JSON block.\n"
    "- If a trigger matches: append exactly one block at the end, prefixed with ALERTS_JSON:, using this schema:\n"
    "ALERTS_JSON:\n"
    "{\n"
    "  \"alerts\": [\n"
    "    {\n"
    "      \"title\": \"Short event title\",\n"
    "      \"description\": \"<= 240 chars, concrete and actionable\",\n"
    "      \"severity\": \"info|low|normal|high|critical\",\n"
    "      \"state\": \"new\",\n"
    "      \"channel_id\": {channel_id},\n"
    "      \"timestamp_ms\": 1772202050000\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "Rules: max 3 alerts; do not alert routine micro-movements unless explicitly requested; timestamp_ms should be batch time in ms."
)
OUTDATED_LUXRIOT_ALERTS_JSON_PROMPTS = {
    LEGACY_LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT.strip(),
    PREVIOUS_LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT.strip(),
}
LUXRIOT_SYSTEM_PROMPT_DEFAULT = (
    "You are a CCTV operator assistant for Luxriot.\n"
    "Return Markdown with exactly these sections and order:\n"
    "### Scene description\n"
    "1-2 short paragraphs describing stable scene context.\n"
    "### Activity description\n"
    "1-2 short paragraphs describing what changed in this batch; reference snapshot numbers or timestamps when possible.\n"
    "### Worth to remember\n"
    "2-6 concise bullet points with context useful for future rollups.\n"
    "Rules: separate routine baseline from deviations; keep it factual and concise; avoid repetition; "
    "emit alerts JSON only when a Task-defined trigger is observed in this batch."
)

current_stream_prompt = str(getattr(config, 'LUXRIOT_SYSTEM_PROMPT_DEFAULT', '') or '').strip()
if not current_stream_prompt:
    config.LUXRIOT_SYSTEM_PROMPT_DEFAULT = LUXRIOT_SYSTEM_PROMPT_DEFAULT

current_json_prompt = str(getattr(config, 'LUXRIOT_ALERTS_JSON_PROMPT', '') or '').strip()
if (not current_json_prompt) or (current_json_prompt in OUTDATED_LUXRIOT_ALERTS_JSON_PROMPTS):
    config.LUXRIOT_ALERTS_JSON_PROMPT = LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT

luxriot_manager = LuxriotManager(
    config=config,
    lm_callback=_call_video_understanding,
    message_builder=cast(Any, _build_luxriot_messages),
    jpeg_encoder=_encode_jpeg,
    alert_parser=_parse_lm_alerts,
    probe_manager=None,  # will be assigned after probe_manager init
)
try:
    with luxriot_manager.cache_lock:
        changed_prompt_defaults = False
        if not str(luxriot_manager.system_prompt or '').strip():
            luxriot_manager.system_prompt = LUXRIOT_SYSTEM_PROMPT_DEFAULT
            changed_prompt_defaults = True
        if str(luxriot_manager.default_json_alert_prompt or '').strip() in OUTDATED_LUXRIOT_ALERTS_JSON_PROMPTS:
            luxriot_manager.default_json_alert_prompt = LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT
            changed_prompt_defaults = True
        desired_rollup_prompts = {
            'L1': str(getattr(config, 'LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT', '') or '').strip(),
            'L2': str(getattr(config, 'LUXRIOT_ROLLUP_L2_SYSTEM_PROMPT', '') or '').strip(),
            'L3': str(getattr(config, 'LUXRIOT_ROLLUP_L3_SYSTEM_PROMPT', '') or '').strip(),
        }
        legacy_rollup_prompt = LEGACY_LUXRIOT_ROLLUP_PROMPT_DEFAULT.strip()
        if (
            not str(luxriot_manager.rollup_llm_system_prompt or '').strip()
            or str(luxriot_manager.rollup_llm_system_prompt or '').strip() == legacy_rollup_prompt
        ):
            base_rollup_prompt = str(getattr(config, 'LUXRIOT_ROLLUP_LLM_SYSTEM_PROMPT', '') or '').strip()
            if not base_rollup_prompt:
                base_rollup_prompt = desired_rollup_prompts.get('L1') or legacy_rollup_prompt
            luxriot_manager.rollup_llm_system_prompt = base_rollup_prompt
            changed_prompt_defaults = True
        for level in ('L1', 'L2', 'L3'):
            current_level_prompt = str(luxriot_manager.rollup_llm_system_prompts.get(level) or '').strip()
            default_level_prompt = desired_rollup_prompts.get(level) or luxriot_manager.rollup_llm_system_prompt
            if not current_level_prompt or current_level_prompt == legacy_rollup_prompt:
                luxriot_manager.rollup_llm_system_prompts[level] = default_level_prompt
                changed_prompt_defaults = True
        for channel_id, raw_overrides in list(luxriot_manager.channel_prompt_overrides.items()):
            if not isinstance(raw_overrides, Mapping):
                continue
            channel_overrides = dict(raw_overrides)
            channel_changed = False
            if str(channel_overrides.get('json_alert_prompt') or '').strip() in OUTDATED_LUXRIOT_ALERTS_JSON_PROMPTS:
                channel_overrides['json_alert_prompt'] = LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT
                channel_changed = True
            rollup_overrides_raw = channel_overrides.get('rollup_prompts')
            if isinstance(rollup_overrides_raw, Mapping):
                rollup_overrides = dict(rollup_overrides_raw)
                rollup_changed = False
                for level in ('L1', 'L2', 'L3'):
                    raw_level_prompt = str(rollup_overrides.get(level) or '').strip()
                    if (not raw_level_prompt) or raw_level_prompt == legacy_rollup_prompt:
                        fallback_level_prompt = desired_rollup_prompts.get(level) or luxriot_manager.rollup_llm_system_prompts.get(level, '')
                        if fallback_level_prompt:
                            rollup_overrides[level] = fallback_level_prompt
                            rollup_changed = True
                if rollup_changed:
                    channel_overrides['rollup_prompts'] = rollup_overrides
                    channel_changed = True
            if channel_changed:
                luxriot_manager.channel_prompt_overrides[int(channel_id)] = channel_overrides
                changed_prompt_defaults = True
        if changed_prompt_defaults:
            luxriot_manager._persist_summary_state_locked()
except Exception:
    pass

probe_manager = ProbeManager(
    embed_image_fn=get_probe_image_embedding_from_pil,
    embed_text_fn=get_probe_text_embedding,
    jpeg_encoder=_encode_jpeg,
)
luxriot_manager.probe_manager = probe_manager
probe_daemon_thread: Optional[threading.Thread] = None
probe_daemon_stop = threading.Event()


class ProbesStore:
    def __init__(self, path: Union[str, Path] = "probes_store.json") -> None:
        self.path = Path(path)
        self.data: Dict[str, Any] = {"probes": []}
        self.lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        with self.lock:
            if self.path.exists():
                try:
                    loaded = json.loads(self.path.read_text(encoding='utf-8'))
                    if isinstance(loaded, dict):
                        self.data = loaded
                    else:
                        self.data = {"probes": []}
                except Exception:
                    self.data = {"probes": []}

    def _save_locked(self) -> None:
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2), encoding='utf-8')
        tmp_path.replace(self.path)

    def list_probes(self) -> List[Dict[str, Any]]:
        with self.lock:
            probes = self.data.get("probes", [])
            return copy.deepcopy(probes if isinstance(probes, list) else [])

    def upsert_probe(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.data.setdefault("probes", [])
            probe_list: List[Dict[str, Any]] = self.data["probes"]
            stored_probe = copy.deepcopy(probe)
            if not stored_probe.get("id"):
                stored_probe["id"] = f"probe-{uuid.uuid4().hex[:12]}"
            existing = None
            for idx, item in enumerate(probe_list):
                if item.get("id") == stored_probe["id"]:
                    existing = idx
                    break
            if existing is None:
                probe_list.append(stored_probe)
            else:
                probe_list[existing] = stored_probe
            self._save_locked()
            return copy.deepcopy(stored_probe)

    def delete_probe(self, probe_id: str) -> bool:
        with self.lock:
            probes = self.data.get("probes", [])
            new_probes = [p for p in probes if p.get("id") != probe_id]
            if len(new_probes) == len(probes):
                return False
            self.data["probes"] = new_probes
            self._save_locked()
            return True


probes_store = ProbesStore()
detections_store = DetectionsStore()
APP_STARTED_AT = time.time()
_control_plane_db_pool: Optional[PsycopgPool] = None
_control_plane_db_lock = Lock()
_inference_worker_db_pool: Optional[PsycopgPool] = None
_inference_worker_db_lock = Lock()
_inference_queue_runtime: Optional[LuxriotInferenceQueueRuntime] = None
_inference_queue_lock = Lock()


def _component_result(
    ok: bool,
    status: str,
    *,
    required: bool = True,
    **details: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": bool(ok),
        "status": status,
        "required": bool(required),
    }
    for key, value in details.items():
        if value is not None:
            payload[key] = value
    return payload


def _embedder_loaded_state() -> Dict[str, Any]:
    if active_embedder == "clip":
        loaded = clip_model is not None and (clip_preprocess is not None or clip_processor is not None)
    elif active_embedder == "dino":
        loaded = dino_encoder is not None
    elif active_embedder == "fusion":
        loaded = (
            clip_model is not None
            and (clip_preprocess is not None or clip_processor is not None)
            and dino_encoder is not None
        )
    else:
        return _component_result(False, "unsupported", embedder=active_embedder)
    return _component_result(
        loaded,
        "loaded" if loaded else "not_loaded",
        embedder=active_embedder,
        clip_model=clip_runtime_model or None,
        backend=clip_backend_kind if clip_model is not None else None,
    )


def _check_database_ready() -> Dict[str, Any]:
    try:
        with detections_store.lock:
            conn = detections_store._connect()
            try:
                conn.execute("SELECT 1").fetchone()
            finally:
                conn.close()
        return _component_result(True, "reachable", path=str(detections_store.path))
    except Exception as exc:
        return _component_result(False, "error", error=str(exc), path=str(detections_store.path))


def _postgres_database_configured() -> bool:
    return any(
        str(os.getenv(name) or "").strip()
        for name in (
            "EVA_DATABASE_DSN",
            "EVA_DATABASE_URL",
            "EVOSSEARCH_DATABASE_DSN",
            "EVOSSEARCH_DATABASE_URL",
            "DATABASE_URL",
        )
    )


def _get_control_plane_db_pool() -> PsycopgPool:
    global _control_plane_db_pool
    with _control_plane_db_lock:
        if _control_plane_db_pool is None:
            _control_plane_db_pool = PsycopgPool(DatabaseSettings.from_env())
        return _control_plane_db_pool


def _inference_worker_database_dsn() -> str:
    return str(
        os.getenv("EVA_WORKER_DATABASE_DSN")
        or os.getenv("EVOSSEARCH_WORKER_DATABASE_DSN")
        or ""
    ).strip()


def _get_inference_worker_db_pool() -> PsycopgPool:
    global _inference_worker_db_pool
    with _inference_worker_db_lock:
        if _inference_worker_db_pool is None:
            dsn = _inference_worker_database_dsn()
            if not dsn:
                raise RuntimeError(
                    "EVA_WORKER_DATABASE_DSN is required when local inference "
                    "workers are enabled"
                )
            base_settings = DatabaseSettings.from_env()
            worker_count = max(
                1,
                int(getattr(config, "INFERENCE_WORKER_COUNT", 0)),
            )
            _inference_worker_db_pool = PsycopgPool(
                replace(
                    base_settings,
                    dsn=dsn,
                    pool_min_size=0,
                    pool_max_size=min(64, worker_count + 1),
                    application_name="eva-ai-inference-worker",
                )
            )
        return _inference_worker_db_pool


def _configure_inference_queue() -> Optional[LuxriotInferenceQueueRuntime]:
    global _inference_queue_runtime
    if not bool(getattr(config, "INFERENCE_QUEUE_ENABLED", False)):
        return None
    with _inference_queue_lock:
        if _inference_queue_runtime is not None:
            return _inference_queue_runtime
        tenant_id = str(
            getattr(config, "INFERENCE_QUEUE_TENANT_ID", "") or ""
        ).strip()
        if not tenant_id:
            raise RuntimeError(
                "EVOSSEARCH_INFERENCE_QUEUE_TENANT_ID is required when the "
                "inference queue is enabled"
            )
        api_repository = PostgresInferenceQueueRepository(
            _get_control_plane_db_pool(),
            tenant_id,
        )
        worker_count = int(getattr(config, "INFERENCE_WORKER_COUNT", 0))
        worker_repository = (
            PostgresInferenceQueueRepository(
                _get_inference_worker_db_pool(),
                tenant_id,
            )
            if worker_count > 0
            else None
        )
        runtime = LuxriotInferenceQueueRuntime(
            manager=luxriot_manager,
            enqueue_repository=api_repository,
            worker_repository=worker_repository,
            tenant_id=tenant_id,
            capacity=int(config.INFERENCE_QUEUE_CAPACITY),
            spool_directory=config.INFERENCE_QUEUE_SPOOL_DIR,
            default_model=config.LM_MODEL,
            worker_count=worker_count,
            poll_interval_seconds=float(
                config.INFERENCE_WORKER_POLL_INTERVAL_SEC
            ),
            lease_seconds=float(config.INFERENCE_WORKER_LEASE_SEC),
            spool_retention_hours=float(
                config.INFERENCE_SPOOL_RETENTION_HOURS
            ),
        )
        runtime.start()
        luxriot_manager.set_summary_dispatcher(runtime.enqueue_summary)
        _inference_queue_runtime = runtime
        return runtime


def _check_postgres_ready() -> Dict[str, Any]:
    if not _postgres_database_configured():
        return _component_result(False, "not_configured", required=False)
    try:
        result = _get_control_plane_db_pool().check_readiness()
        return _component_result(
            result.ready,
            result.state.value,
            detail=result.detail,
            latency_ms=result.latency_ms,
            current_revision=result.current_revision,
            expected_revision=result.expected_revision,
        )
    except Exception as exc:
        return _component_result(
            False,
            "unavailable",
            error=type(exc).__name__,
        )


def _check_auth_ready() -> Dict[str, Any]:
    if not _auth_enabled():
        return _component_result(False, "disabled", required=False)
    tenant_id = str(getattr(config, "AUTH_TENANT_ID", "") or "").strip()
    if not tenant_id:
        return _component_result(
            False,
            "misconfigured",
            error="EVOSSEARCH_AUTH_TENANT_ID is required",
        )
    try:
        uuid.UUID(tenant_id)
    except ValueError:
        return _component_result(
            False,
            "misconfigured",
            error="EVOSSEARCH_AUTH_TENANT_ID must be a UUID",
        )
    if not _postgres_database_configured():
        return _component_result(
            False,
            "misconfigured",
            error="PostgreSQL is required when authentication is enabled",
        )
    if not _audit_database_dsn():
        return _component_result(
            False,
            "misconfigured",
            error="EVA_AUDIT_DATABASE_DSN is required when authentication is enabled",
        )
    postgres = _check_postgres_ready()
    try:
        audit_database = _get_audit_db_pool().check_health()
    except Exception as exc:
        return _component_result(
            False,
            "unavailable",
            error=f"audit database unavailable ({type(exc).__name__})",
            tenant_id=tenant_id,
        )
    return _component_result(
        bool(postgres.get("ok")) and audit_database.ready,
        (
            "ready"
            if postgres.get("ok") and audit_database.ready
            else "unavailable"
        ),
        tenant_id=tenant_id,
        audit_latency_ms=audit_database.latency_ms,
    )


def _check_inference_queue_ready() -> Dict[str, Any]:
    if not bool(getattr(config, "INFERENCE_QUEUE_ENABLED", False)):
        return _component_result(False, "disabled", required=False)
    try:
        runtime = _configure_inference_queue()
        if runtime is None:
            return _component_result(False, "disabled", required=False)
        status = runtime.status()
    except Exception as exc:
        return _component_result(
            False,
            "unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )
    worker_count = int(status.get("worker_count") or 0)
    workers_alive = int(status.get("workers_alive") or 0)
    workers_ready = worker_count == 0 or workers_alive == worker_count
    return _component_result(
        workers_ready,
        "ready" if workers_ready else "worker_unavailable",
        **status,
    )


def _luxriot_configured() -> bool:
    base_url = str(getattr(config, "LUXRIOT_BASE_URL", "") or "").strip()
    if not base_url or "luxriot-host" in base_url:
        return False
    return bool(str(getattr(config, "LUXRIOT_USERNAME", "") or "").strip())


def _check_luxriot_ready(timeout_sec: float = 2.0) -> Dict[str, Any]:
    base_url = str(getattr(config, "LUXRIOT_BASE_URL", "") or "").strip().rstrip("/")
    if not _luxriot_configured():
        return _component_result(False, "not_configured", required=False, base_url=base_url or None)
    try:
        auth = requests.auth.HTTPDigestAuth(config.LUXRIOT_USERNAME, config.LUXRIOT_PASSWORD)
        resp = requests.get(
            f"{base_url}/channels",
            params={"health": 0},
            headers={"Accept": "application/json"},
            auth=auth,
            timeout=(1.0, max(1.0, float(timeout_sec))),
            stream=True,
        )
        try:
            ok = 200 <= int(resp.status_code) < 400
            return _component_result(
                ok,
                "reachable" if ok else "http_error",
                status_code=int(resp.status_code),
                base_url=base_url,
            )
        finally:
            resp.close()
    except Exception as exc:
        return _component_result(False, "error", error=str(exc), base_url=base_url)


_configure_inference_queue()


@app.route('/health', methods=['GET'])
def health():
    """Liveness endpoint for process supervisors and load balancers."""
    return jsonify(
        {
            "status": "ok",
            "version": config.APP_VERSION,
            "uptime_sec": round(max(0.0, time.time() - APP_STARTED_AT), 3),
        }
    )


@app.route('/ready', methods=['GET'])
def ready():
    """Readiness endpoint with component-level dependency status."""
    load_embedder = str(request.args.get("load") or "").strip().lower() in TRUE_BOOL_STRINGS
    strict = str(request.args.get("strict") or "").strip().lower() in TRUE_BOOL_STRINGS

    checks: Dict[str, Dict[str, Any]] = {
        "database": _check_database_ready(),
        "postgresql": _check_postgres_ready(),
        "authentication": _check_auth_ready(),
        "inference_queue": _check_inference_queue_ready(),
        "embedder": _embedder_loaded_state(),
        "luxriot": _check_luxriot_ready(),
    }

    if load_embedder and not checks["embedder"].get("ok"):
        try:
            ensure_embedder_loaded(active_embedder)
            checks["embedder"] = _embedder_loaded_state()
        except Exception as exc:
            checks["embedder"] = _component_result(
                False,
                "load_failed",
                embedder=active_embedder,
                error=str(exc),
            )

    required_names = ["database", "embedder"]
    if checks["postgresql"].get("required"):
        required_names.append("postgresql")
    if checks["authentication"].get("required"):
        required_names.append("authentication")
    if checks["inference_queue"].get("required"):
        required_names.append("inference_queue")
    if strict or checks["luxriot"].get("required"):
        required_names.append("luxriot")

    is_ready = all(bool(checks[name].get("ok")) for name in required_names)
    status_code = 200 if is_ready else 503
    return jsonify(
        {
            "status": "ready" if is_ready else "not_ready",
            "version": config.APP_VERSION,
            "required": required_names,
            "checks": checks,
        }
    ), status_code


def _identity_payload(identity: Any) -> Dict[str, Any]:
    return {
        "id": identity.user_id,
        "tenantId": identity.tenant_id,
        "username": identity.username,
        "displayName": identity.display_name,
        "isActive": bool(getattr(identity, "is_active", True)),
        "roles": sorted(identity.roles),
        "permissions": sorted(identity.permissions),
        "allowedChannelIds": sorted(
            identity.allowed_channel_ids,
            key=lambda value: str(value),
        ),
    }


def _role_payload(role: Role) -> Dict[str, Any]:
    permissions = ROLE_PERMISSIONS[role]
    return {
        "name": role.value,
        "permissions": sorted(permission.value for permission in permissions),
    }


def _auth_admin_guard(*, write: bool, action: str):
    return _session_guard(
        permission=Permission.USERS_MANAGE,
        require_csrf=write,
        action=action,
    )


def _body_value(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return None


def _has_body_value(data: Mapping[str, Any], *names: str) -> bool:
    return any(name in data for name in names)


def _parse_roles_payload(value: Any, *, default: Sequence[str] | None = None) -> List[str]:
    raw = default if value is None and default is not None else value
    if isinstance(raw, str):
        roles = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        roles = [str(item) for item in raw]
    else:
        raise ValueError("roles must be a list")
    if not roles:
        raise ValueError("at least one role is required")
    return roles


def _parse_channel_ids_payload(value: Any) -> List[Union[int, str]]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        raw_items: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = value
    else:
        raise ValueError("allowedChannelIds must be a list")
    parsed: List[Union[int, str]] = []
    for item in raw_items:
        if str(item).strip() == ALL_CHANNELS:
            parsed.append(ALL_CHANNELS)
            continue
        try:
            channel_id = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("allowedChannelIds must contain positive integers") from exc
        if channel_id <= 0:
            raise ValueError("allowedChannelIds must contain positive integers")
        parsed.append(channel_id)
    return parsed


def _audit_identity_admin_result(
    *,
    action: str,
    result: str,
    target_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    _write_security_audit(
        context=_current_auth_context(),
        action=action,
        result=result,
        target_type="iam_user",
        target_id=target_id,
        details=details,
    )


@app.route('/auth/login', methods=['POST'])
def auth_login():
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    data = _json_body()
    username = str(data.get("username") or "").strip()
    password = str(data.get("password") or "")
    if not username or not password:
        return _auth_failure_response("Username and password are required", 400)
    try:
        login = _get_auth_service().login(
            username=username,
            password=password,
            client_ip=_source_ip(),
            user_agent=str(request.headers.get("User-Agent") or "")[:1024] or None,
        )
    except LoginThrottled as exc:
        try:
            _write_security_audit(
                context=None,
                action="auth.login",
                result="denied",
                target_type="user",
                details={"reason": "throttled"},
            )
        except Exception:
            return _auth_failure_response("Audit service unavailable", 503)
        response = _auth_failure_response("Too many login attempts", 429)
        response[0].headers["Retry-After"] = str(exc.retry_after_seconds)
        return response
    except InvalidCredentials:
        try:
            _write_security_audit(
                context=None,
                action="auth.login",
                result="failure",
                target_type="user",
                details={"reason": "invalid_credentials"},
            )
        except Exception:
            return _auth_failure_response("Audit service unavailable", 503)
        return _auth_failure_response("Invalid username or password", 401)
    except Exception:
        return _auth_failure_response("Authentication service unavailable", 503)

    context = AuthContext(
        user_id=login.identity.user_id,
        tenant_id=login.identity.tenant_id,
        roles=login.identity.roles,
        permissions=login.identity.permissions,
        allowed_channel_ids=login.identity.allowed_channel_ids,
        request_id=g.request_id,
    )
    try:
        _write_security_audit(
            context=context,
            action="auth.login",
            result="success",
            target_type="session",
            target_id=login.session_id,
        )
    except Exception:
        try:
            _get_auth_service().logout(
                login.session_token,
                reason="audit_unavailable",
            )
        except Exception:
            pass
        return _auth_failure_response("Audit service unavailable", 503)

    max_age = max(
        1,
        int((login.expires_at - datetime.now(timezone.utc)).total_seconds()),
    )
    response = make_response(
        jsonify(
            {
                "success": True,
                "user": _identity_payload(login.identity),
                "expiresAt": login.expires_at.isoformat(),
                "csrfHeader": "X-CSRF-Token",
            }
        )
    )
    response.set_cookie(
        config.AUTH_SESSION_COOKIE,
        login.session_token,
        max_age=max_age,
        secure=bool(config.AUTH_COOKIE_SECURE),
        httponly=True,
        samesite="Strict",
        path="/",
    )
    response.set_cookie(
        config.AUTH_CSRF_COOKIE,
        login.csrf_token,
        max_age=max_age,
        secure=bool(config.AUTH_COOKIE_SECURE),
        httponly=False,
        samesite="Strict",
        path="/",
    )
    return response


@app.route('/auth/me', methods=['GET'])
def auth_me():
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    guard = _session_guard(
        permission=None,
        require_csrf=False,
        action="auth.me.read",
    )
    if guard is not None:
        return guard
    session_record = g.auth_session
    return jsonify(
        {
            "success": True,
            "user": _identity_payload(session_record.identity),
            "expiresAt": session_record.expires_at.isoformat(),
            "csrfHeader": "X-CSRF-Token",
        }
    )


@app.route('/auth/roles', methods=['GET'])
def auth_roles():
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    guard = _auth_admin_guard(write=False, action="auth.roles.read")
    if guard is not None:
        return guard
    return jsonify(
        {
            "success": True,
            "roles": [_role_payload(role) for role in Role],
        }
    )


@app.route('/auth/users', methods=['GET', 'POST'])
def auth_users():
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    guard = _auth_admin_guard(
        write=request.method == "POST",
        action=(
            "auth.users.create"
            if request.method == "POST"
            else "auth.users.list"
        ),
    )
    if guard is not None:
        return guard
    context = _current_auth_context()
    if context is None:
        return _auth_failure_response("Authentication required", 401)

    repository = _get_identity_repository()
    if request.method == "GET":
        include_inactive = _coerce_bool(
            request.args.get("includeInactive"),
            default=True,
        )
        try:
            users = repository.list_users(
                context.tenant_id,
                actor_user_id=context.user_id,
                include_inactive=include_inactive,
            )
        except Exception:
            return _auth_failure_response("Identity service unavailable", 503)
        return jsonify(
            {
                "success": True,
                "users": [_identity_payload(user) for user in users],
            }
        )

    data = _json_body()
    username = str(data.get("username") or "").strip()
    password = str(data.get("password") or "")
    display_name = _body_value(data, "displayName", "display_name")
    is_active = _coerce_bool(_body_value(data, "isActive", "is_active"), True)
    try:
        roles = _parse_roles_payload(data.get("roles"), default=[Role.VIEWER.value])
        channel_ids = _parse_channel_ids_payload(
            _body_value(data, "allowedChannelIds", "allowed_channel_ids")
        )
        if not username or not password:
            raise ValueError("username and password are required")
        user = repository.create_user(
            context.tenant_id,
            actor_user_id=context.user_id,
            username=username,
            password=password,
            display_name=None if display_name is None else str(display_name),
            roles=roles,
            allowed_channel_ids=channel_ids,
            is_active=is_active,
        )
        _audit_identity_admin_result(
            action="auth.users.create.completed",
            result="success",
            target_id=user.user_id,
            details={
                "username": user.username,
                "roles": sorted(user.roles),
                "is_active": user.is_active,
            },
        )
    except ValueError as exc:
        return _auth_failure_response(str(exc), 400)
    except Exception:
        return _auth_failure_response("Identity service unavailable", 503)

    return jsonify({"success": True, "user": _identity_payload(user)}), 201


@app.route('/auth/users/<user_id>', methods=['GET', 'PATCH'])
def auth_user(user_id: str):
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    guard = _auth_admin_guard(
        write=request.method == "PATCH",
        action=(
            "auth.users.update"
            if request.method == "PATCH"
            else "auth.users.read"
        ),
    )
    if guard is not None:
        return guard
    context = _current_auth_context()
    if context is None:
        return _auth_failure_response("Authentication required", 401)

    repository = _get_identity_repository()
    if request.method == "GET":
        try:
            user = repository.get_user(
                context.tenant_id,
                user_id,
                actor_user_id=context.user_id,
            )
        except ValueError as exc:
            return _auth_failure_response(str(exc), 400)
        except Exception:
            return _auth_failure_response("Identity service unavailable", 503)
        if user is None:
            return _auth_failure_response("User not found", 404)
        return jsonify({"success": True, "user": _identity_payload(user)})

    data = _json_body()
    updates: Dict[str, Any] = {}
    try:
        if _has_body_value(data, "displayName", "display_name"):
            updates["display_name"] = _body_value(
                data,
                "displayName",
                "display_name",
            )
        if _has_body_value(data, "password"):
            updates["password"] = str(data.get("password") or "")
        if _has_body_value(data, "roles"):
            updates["roles"] = _parse_roles_payload(data.get("roles"))
        if _has_body_value(data, "allowedChannelIds", "allowed_channel_ids"):
            updates["allowed_channel_ids"] = _parse_channel_ids_payload(
                _body_value(data, "allowedChannelIds", "allowed_channel_ids")
            )
        if _has_body_value(data, "isActive", "is_active"):
            updates["is_active"] = _coerce_bool(
                _body_value(data, "isActive", "is_active"),
                default=True,
            )
        if not updates:
            raise ValueError("no user fields to update")
        user = repository.update_user(
            context.tenant_id,
            user_id,
            actor_user_id=context.user_id,
            **updates,
        )
        _audit_identity_admin_result(
            action="auth.users.update.completed",
            result="success",
            target_id=user.user_id,
            details={
                "updated_fields": sorted(updates),
                "roles": sorted(user.roles),
                "is_active": user.is_active,
            },
        )
    except LookupError:
        return _auth_failure_response("User not found", 404)
    except ValueError as exc:
        return _auth_failure_response(str(exc), 400)
    except Exception:
        return _auth_failure_response("Identity service unavailable", 503)
    return jsonify({"success": True, "user": _identity_payload(user)})


@app.route('/auth/users/<user_id>/revoke-sessions', methods=['POST'])
def auth_user_revoke_sessions(user_id: str):
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    guard = _auth_admin_guard(
        write=True,
        action="auth.users.revoke_sessions",
    )
    if guard is not None:
        return guard
    context = _current_auth_context()
    if context is None:
        return _auth_failure_response("Authentication required", 401)
    data = _json_body()
    reason = str(data.get("reason") or "admin_revoked").strip()
    try:
        revoked = _get_identity_repository().revoke_user_sessions(
            context.tenant_id,
            user_id,
            actor_user_id=context.user_id,
            reason=reason,
        )
        _audit_identity_admin_result(
            action="auth.users.revoke_sessions.completed",
            result="success",
            target_id=user_id,
            details={"revoked_sessions": revoked, "reason": reason},
        )
    except LookupError:
        return _auth_failure_response("User not found", 404)
    except ValueError as exc:
        return _auth_failure_response(str(exc), 400)
    except Exception:
        return _auth_failure_response("Identity service unavailable", 503)
    return jsonify({"success": True, "revokedSessions": revoked})


@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    session_token = str(
        request.cookies.get(config.AUTH_SESSION_COOKIE) or ""
    )
    try:
        _get_auth_service().logout(session_token, reason="logout")
    except Exception:
        return _auth_failure_response("Authentication service unavailable", 503)
    response = make_response(jsonify({"success": True}))
    response.delete_cookie(config.AUTH_SESSION_COOKIE, path="/")
    response.delete_cookie(config.AUTH_CSRF_COOKIE, path="/")
    return response

# Agent runner — instantiated lazily on first /agent/chat request so that
# all helper functions (get_text_embedding, _search_detections_archive, etc.)
# are fully defined before the runner captures them as callables.
_agent_runner: Optional[Any] = None
_agent_runner_lock = threading.Lock()
_agent_runtime_model_override: Optional[str] = None
_skills_root = Path(__file__).resolve().parent / "skills"
_lm_models_cache_lock = threading.Lock()
_lm_models_cache_payload: Optional[Dict[str, Any]] = None
_lm_models_cache_expires_at = 0.0


def _slugify_skill_name(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or "").strip())
    parts = [part for part in text.split("-") if part]
    return "-".join(parts)[:80]


def _normalize_skill_token(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    token = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in raw)
    while "--" in token:
        token = token.replace("--", "-")
    return token.strip("-_")[:80]


def _candidate_skill_slugs(value: str) -> List[str]:
    base = _normalize_skill_token(value)
    if not base:
        return []
    variants: List[str] = []
    for candidate in (base, base.replace("-", "_"), base.replace("_", "-")):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _skill_title_from_markdown(content: str, fallback_slug: str) -> str:
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback_slug.replace("-", " ").title()
        break
    return fallback_slug.replace("-", " ").title()


def _skill_summary_from_markdown(content: str) -> str:
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        return line[:180]
    return "No summary yet."


def _skill_markdown_template(name: str) -> str:
    title = str(name or "New Skill").strip() or "New Skill"
    return (
        f"# {title}\n\n"
        "Goal: describe when this playbook should be used.\n\n"
        "Default order:\n"
        "1. Clarify missing inputs if needed.\n"
        "2. Inspect the relevant context.\n"
        "3. Use the right tools in a safe order.\n"
        "4. Summarize the result for the operator.\n\n"
        "Notes:\n"
        "- Add decision rules here.\n"
        "- Add embedder/model-specific guidance here.\n"
    )


def _apply_skill_title_to_markdown(name: str, content: str, slug: str) -> str:
    title = str(name or "").strip() or slug.replace("-", " ").title()
    body = str(content or "").strip()
    if not body:
        return _skill_markdown_template(title)
    lines = body.splitlines()
    for idx, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith("#"):
            lines[idx] = f"# {title}"
            return "\n".join(lines).strip()
        return f"# {title}\n\n{body}".strip()
    return _skill_markdown_template(title)


def _resolve_skill_path(slug: str) -> Path:
    candidates = _candidate_skill_slugs(slug)
    if not candidates:
        raise ValueError("Invalid skill slug")
    root = _skills_root.resolve()
    existing_paths: List[Path] = []
    for candidate in candidates:
        skill_file = (_skills_root / candidate / "SKILL.md").resolve()
        if root not in skill_file.parents:
            raise ValueError("Skill path is outside skills root")
        if skill_file.exists():
            existing_paths.append(skill_file)
    if existing_paths:
        return existing_paths[0]
    fallback = (_skills_root / candidates[0] / "SKILL.md").resolve()
    if root not in fallback.parents:
        raise ValueError("Skill path is outside skills root")
    return fallback


def _list_skill_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not _skills_root.exists():
        return records
    for skill_file in sorted(_skills_root.rglob("SKILL.md")):
        try:
            content = skill_file.read_text(encoding="utf-8")
        except Exception:
            continue
        slug = skill_file.parent.name
        stat = skill_file.stat()
        records.append({
            "slug": slug,
            "name": _skill_title_from_markdown(content, slug),
            "summary": _skill_summary_from_markdown(content),
            "path": str(skill_file.relative_to(Path(__file__).resolve().parent)),
            "updated_at": stat.st_mtime,
        })
    return records


def _load_skill_record(slug: str) -> Dict[str, Any]:
    skill_file = _resolve_skill_path(slug)
    if not skill_file.exists():
        raise FileNotFoundError(f"Skill not found: {slug}")
    content = skill_file.read_text(encoding="utf-8")
    stat = skill_file.stat()
    return {
        "slug": skill_file.parent.name,
        "name": _skill_title_from_markdown(content, skill_file.parent.name),
        "summary": _skill_summary_from_markdown(content),
        "content": content,
        "path": str(skill_file.relative_to(Path(__file__).resolve().parent)),
        "updated_at": stat.st_mtime,
    }


def _save_skill_record(slug: str, name: str, content: str) -> Dict[str, Any]:
    skill_file = _resolve_skill_path(slug)
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    final_content = _apply_skill_title_to_markdown(name, content, skill_file.parent.name)
    skill_file.write_text(final_content.rstrip() + "\n", encoding="utf-8")
    return _load_skill_record(skill_file.parent.name)


def _get_agent_runner() -> Any:
    """Return (creating if needed) the singleton AgentRunner."""
    global _agent_runner
    if _agent_runner is not None:
        return _agent_runner
    with _agent_runner_lock:
        if _agent_runner is not None:
            return _agent_runner
        from agent import AgentRunner
        approval_store = None
        if _auth_enabled():
            from agent_security import PostgresPlanApprovalStore

            approval_store = PostgresPlanApprovalStore(_get_control_plane_db_pool())

        def _agent_search_folder(
            *, query: str, folder: str, limit: int = 12, sort_by: str = "similarity"
        ) -> List[Dict[str, Any]]:
            idx_bundle = load_index(folder, embedder='clip')
            index, image_paths, image_metadata, _ = idx_bundle
            if index is None or not image_paths:
                return []
            vec = get_text_embedding(query)
            from agent import _strip_thumbnails  # local import is fine here
            results = _build_ranked_results(
                index=index,
                image_paths=image_paths,
                image_metadata=image_metadata,
                query_vec=vec,
                limit=limit,
                sort_by=sort_by,
            )
            return results

        def _agent_search_detections(
            *, query: str, probe_id: Optional[str] = None,
            channel_id: Optional[int] = None,
            since_ms: Optional[int] = None,
            until_ms: Optional[int] = None,
            limit: int = 12,
            sort_by: str = "similarity",
            candidate_limit: int = 20000,
            mode: str = "clip",
        ) -> List[Dict[str, Any]]:
            vec = get_text_embedding(query)
            return _search_detections_archive(
                clip_query_vec=vec,
                dino_query_vec=None,
                mode=mode,
                probe_id=probe_id,
                channel_id=channel_id,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                sort_by=sort_by,
                candidate_limit=candidate_limit,
            )

        _agent_runner = AgentRunner(
            embed_text_fn=lambda text: get_text_embedding(text),
            embed_image_fn=lambda img: get_image_embedding_from_pil(img, embedder='clip'),
            call_lm_fn=_call_lm_chat,
            encode_jpeg_fn=_encode_jpeg,
            probes_store=probes_store,
            detections_store=detections_store,
            luxriot_manager=luxriot_manager,
            search_indexed_folder_fn=_agent_search_folder,
            search_detections_fn=_agent_search_detections,
            lm_base_url=config.LM_BASE_URL,
            lm_model=_agent_runtime_model_override or config.LM_MODEL,
            lm_api_key=config.LM_API_KEY,
            lm_timeout=config.LM_TIMEOUT,
            tool_audit_callback=_write_agent_tool_audit,
            tool_plan_store=approval_store,
            tool_approval_store=approval_store,
        )
        return _agent_runner


def _get_agent_config_payload() -> Dict[str, Any]:
    return {
        "model": str(_agent_runtime_model_override or config.LM_MODEL or "").strip(),
        "default_model": str(config.LM_MODEL or "").strip(),
        "override_model": str(_agent_runtime_model_override or "").strip() or None,
        "source": "runtime_override" if _agent_runtime_model_override else "config",
    }


def _agent_session_owner() -> Dict[str, str]:
    if not _auth_enabled():
        return {}
    context = _current_auth_context()
    if context is None:
        return {}
    return {
        "tenant_id": context.tenant_id,
        "actor_id": context.user_id,
    }


def _fetch_lm_model_catalog(force: bool = False) -> Dict[str, Any]:
    global _lm_models_cache_payload, _lm_models_cache_expires_at

    now = time.monotonic()
    with _lm_models_cache_lock:
        if (
            not force
            and _lm_models_cache_payload is not None
            and now < _lm_models_cache_expires_at
        ):
            return copy.deepcopy(_lm_models_cache_payload)

    default_model = str(config.LM_MODEL or "").strip()
    fallback_models: List[str] = []
    for candidate in (
        default_model,
        str(_agent_runtime_model_override or "").strip(),
    ):
        if candidate and candidate not in fallback_models:
            fallback_models.append(candidate)

    payload: Dict[str, Any] = {
        "models": fallback_models,
        "default_model": default_model,
        "source": "fallback",
        "error": None,
        "fetched_at": time.time(),
    }

    try:
        base_url = (config.LM_BASE_URL or "").rstrip("/")
        if not base_url:
            raise RuntimeError("EVOSSEARCH_LM_BASE_URL is not configured.")
        headers = {"Content-Type": "application/json"}
        if config.LM_API_KEY:
            headers["Authorization"] = f"Bearer {config.LM_API_KEY}"
        timeout = (3.05, min(10.0, max(5.0, float(config.LM_TIMEOUT or 120))))
        response = requests.get(f"{base_url}/models", headers=headers, timeout=timeout)
        response.raise_for_status()
        raw = response.json()
        items = raw.get("data") if isinstance(raw, Mapping) else None
        model_ids: List[str] = []
        if isinstance(items, Sequence):
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                model_id = str(item.get("id") or item.get("model") or "").strip()
                if model_id and model_id not in model_ids:
                    model_ids.append(model_id)
        if not model_ids:
            model_ids = list(fallback_models)
        payload.update({
            "models": model_ids,
            "source": "lm_studio",
            "error": None,
        })
    except Exception as exc:
        payload["error"] = str(exc)

    with _lm_models_cache_lock:
        _lm_models_cache_payload = copy.deepcopy(payload)
        _lm_models_cache_expires_at = now + 15.0
    return payload


DETECTIONS_SEARCH_MAX_CANDIDATES = 100000
DETECTIONS_SEARCH_DEFAULT_HOURS = 24.0
DETECTIONS_SEARCH_SHARD_OVERFETCH = 200
DETECTIONS_SEARCH_DINO_POOL_MIN = 64
DETECTIONS_SEARCH_DINO_POOL_MULTIPLIER = 8


class _DetectionClipShardCache:
    def __init__(self, store: DetectionsStore) -> None:
        self.store = store
        self.lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def clear(self) -> None:
        with self.lock:
            self._cache.clear()

    def get(self, shard_key: str) -> Tuple[Optional[faiss.Index], Optional[np.ndarray]]:
        shard = str(shard_key or "").strip()
        if not shard:
            return None, None
        version = self.store.shard_version(shard, embedder="clip")
        if version[0] <= 0:
            with self.lock:
                self._cache.pop(shard, None)
            return None, None

        with self.lock:
            cached = self._cache.get(shard)
            if cached and cached.get("version") == version:
                return cached.get("index"), cached.get("ids")

        ids, vectors = self.store.load_shard_vectors(shard, embedder="clip")
        if not ids or vectors.size == 0:
            with self.lock:
                self._cache.pop(shard, None)
            return None, None

        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        _faiss_add_vectors(index, vectors)
        ids_arr = np.asarray(ids, dtype=np.int64)

        with self.lock:
            self._cache[shard] = {
                "version": version,
                "index": index,
                "ids": ids_arr,
            }
        return index, ids_arr


detection_clip_shard_cache = _DetectionClipShardCache(detections_store)


def _thumbnail_to_pil_image(thumbnail_b64: Any) -> Optional[Image.Image]:
    raw_value = str(thumbnail_b64 or "").strip()
    if not raw_value:
        return None
    try:
        decoded = base64.b64decode(raw_value)
        with Image.open(BytesIO(decoded)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            else:
                img = img.copy()
            return img
    except Exception:
        return None


def _embed_thumbnail_b64(thumbnail_b64: Any, embedder: str) -> Optional[np.ndarray]:
    pil_img = _thumbnail_to_pil_image(thumbnail_b64)
    if pil_img is None:
        return None
    try:
        return get_image_embedding_from_pil(pil_img, embedder=embedder)
    except Exception:
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except Exception:
        return None


def _probe_identity(probe_like: Mapping[str, Any]) -> str:
    probe_id = str(probe_like.get("id") or "").strip()
    if probe_id:
        return probe_id
    channel_id = int(probe_like.get("channel_id") or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    name = str(probe_like.get("name") or "probe").strip().lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in name).strip("-")
    if not slug:
        slug = "probe"
    return f"adhoc:{channel_id}:{slug[:48]}"


def _probe_bookmark_identity(probe_like: Mapping[str, Any]) -> str:
    base = _probe_identity(probe_like)
    roi_enabled, roi_norm = _parse_probe_roi(probe_like)
    if roi_enabled and roi_norm is not None:
        x, y, w, h = roi_norm
        return f"{base}:roi:{x:.4f}:{y:.4f}:{w:.4f}:{h:.4f}"
    return base


def _slug_token(value: Any, fallback: str) -> str:
    token = str(value or "").strip().lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in token).strip("-")
    return slug[:64] if slug else fallback


class _AdaptiveDetectionArchive:
    def __init__(self) -> None:
        root_raw = str(getattr(config, "DETECTIONS_ARCHIVE_DIR", "detections_archive") or "detections_archive").strip()
        root_path = Path(root_raw).expanduser()
        if not root_path.is_absolute():
            root_path = (Path.cwd() / root_path).resolve()
        else:
            root_path = root_path.resolve()

        self.archive_enabled = bool(getattr(config, "DETECTIONS_ARCHIVE_ENABLED", True))
        self.retention_enabled = bool(getattr(config, "DETECTIONS_RETENTION_ENABLED", True))
        self.drop_skipped = bool(getattr(config, "DETECTIONS_RETENTION_DROP_SKIPPED", True))
        self.root = root_path
        self.window_ms = int(max(500.0, float(getattr(config, "DETECTIONS_RETENTION_WINDOW_SEC", 6.0)) * 1000.0))
        self.force_keep_ms = int(max(1000.0, float(getattr(config, "DETECTIONS_RETENTION_FORCE_KEEP_SEC", 20.0)) * 1000.0))
        self.sim_high = float(getattr(config, "DETECTIONS_RETENTION_SIMILARITY_HIGH", 0.985))
        self.sim_low = float(getattr(config, "DETECTIONS_RETENTION_SIMILARITY_LOW", 0.94))
        self.margin_delta_thr = float(getattr(config, "DETECTIONS_RETENTION_MARGIN_DELTA", 0.08))
        self.score_delta_thr = float(getattr(config, "DETECTIONS_RETENTION_SCORE_DELTA", 0.08))
        self.jpeg_quality = int(getattr(config, "DETECTIONS_ARCHIVE_JPEG_QUALITY", 88))

        self._lock = threading.RLock()
        self._state: Dict[str, Dict[str, Any]] = {}
        if self.archive_enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"Detections archive disabled (cannot create root {self.root}): {exc}")
                self.archive_enabled = False

    @staticmethod
    def _normalize_vec(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vec is None:
            return None
        try:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            return None
        return (arr / norm).astype(np.float32, copy=False)

    @staticmethod
    def _state_key(channel_id: int, probe_id: str, source: str) -> str:
        return f"{int(channel_id)}:{str(probe_id)}:{str(source)}"

    def _resolve_existing_path(self, image_path: Optional[str]) -> Optional[str]:
        raw_path = str(image_path or "").strip()
        if not raw_path:
            return None
        try:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = (self.root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if candidate.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                return None
            if not candidate.exists() or not candidate.is_file():
                return None
            return str(candidate)
        except Exception:
            return None

    def _write_snapshot(
        self,
        *,
        channel_id: int,
        probe_id: str,
        source: str,
        timestamp_ms: int,
        thumbnail_b64: Any,
    ) -> Optional[str]:
        if not self.archive_enabled:
            return None
        pil_img = _thumbnail_to_pil_image(thumbnail_b64)
        if pil_img is None:
            return None
        ts_sec = max(0.0, float(timestamp_ms) / 1000.0)
        date_key = time.strftime("%Y%m%d", time.localtime(ts_sec))
        probe_slug = _slug_token(probe_id, "probe")
        source_slug = _slug_token(source, "probe")
        out_dir = self.root / f"ch{int(channel_id)}" / date_key / probe_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{int(timestamp_ms)}_{source_slug}_{uuid.uuid4().hex[:8]}.jpg"
        out_path = out_dir / filename
        pil_img.save(str(out_path), format="JPEG", quality=self.jpeg_quality)
        return str(out_path)

    def _update_state_locked(
        self,
        *,
        key: str,
        timestamp_ms: int,
        clip_vec: Optional[np.ndarray],
        pos_score: float,
        neg_score: float,
        margin: float,
        image_path: Optional[str],
    ) -> None:
        self._state[key] = {
            "timestamp_ms": int(timestamp_ms),
            "clip_vec": clip_vec.copy() if clip_vec is not None else None,
            "pos_score": float(pos_score),
            "neg_score": float(neg_score),
            "margin": float(margin),
            "image_path": image_path,
        }

    def _decide_keep_locked(
        self,
        *,
        key: str,
        timestamp_ms: int,
        clip_vec: Optional[np.ndarray],
        pos_score: float,
        neg_score: float,
        margin: float,
    ) -> Tuple[bool, str, Optional[float]]:
        if not self.retention_enabled:
            return True, "retention_disabled", None

        prev = self._state.get(key)
        if prev is None:
            return True, "bootstrap", None

        dt_ms = max(0, int(timestamp_ms) - int(prev.get("timestamp_ms") or 0))
        if dt_ms >= self.force_keep_ms:
            return True, "force_interval", None

        prev_vec = prev.get("clip_vec")
        sim: Optional[float] = None
        if isinstance(prev_vec, np.ndarray) and clip_vec is not None and clip_vec.shape == prev_vec.shape:
            sim = float(np.clip(np.dot(clip_vec, prev_vec), -1.0, 1.0))

        margin_delta = abs(float(margin) - float(prev.get("margin") or 0.0))
        pos_delta = abs(float(pos_score) - float(prev.get("pos_score") or 0.0))
        neg_delta = abs(float(neg_score) - float(prev.get("neg_score") or 0.0))

        if sim is not None and sim <= self.sim_low:
            return True, "novel_scene", sim
        if margin_delta >= self.margin_delta_thr or pos_delta >= self.score_delta_thr or neg_delta >= self.score_delta_thr:
            return True, "score_shift", sim
        if dt_ms >= self.window_ms:
            return True, "window_anchor", sim
        if sim is not None and sim >= self.sim_high:
            return False, "high_similarity_skip", sim
        return False, "within_window_skip", sim

    def handle_hit(
        self,
        *,
        probe_id: str,
        channel_id: int,
        source: str,
        timestamp_ms: int,
        clip_vec: Optional[np.ndarray],
        thumbnail_b64: Any,
        pos_score: float,
        neg_score: float,
        margin: float,
        image_path: Optional[str],
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        normalized_vec = self._normalize_vec(clip_vec)
        resolved_path = self._resolve_existing_path(image_path)
        key = self._state_key(channel_id, probe_id, source)

        with self._lock:
            previous = self._state.get(key)
            previous_path = None
            if isinstance(previous, dict):
                previous_path = str(previous.get("image_path") or "").strip() or None
            keep, reason, similarity = self._decide_keep_locked(
                key=key,
                timestamp_ms=timestamp_ms,
                clip_vec=normalized_vec,
                pos_score=pos_score,
                neg_score=neg_score,
                margin=margin,
            )
            if resolved_path:
                # Existing source path already preserved on disk; do not drop it.
                keep = True
                reason = "external_path"
            saved_path = resolved_path
            if keep and not saved_path:
                saved_path = self._write_snapshot(
                    channel_id=channel_id,
                    probe_id=probe_id,
                    source=source,
                    timestamp_ms=timestamp_ms,
                    thumbnail_b64=thumbnail_b64,
                )
                if saved_path:
                    reason = f"{reason}_snapshot_saved"
            if not keep and not saved_path and previous_path:
                # Reuse previous anchor snapshot so expanded view remains useful across dense hit sequences.
                saved_path = previous_path
                reason = f"{reason}_reuse_snapshot"
            if keep:
                self._update_state_locked(
                    key=key,
                    timestamp_ms=timestamp_ms,
                    clip_vec=normalized_vec,
                    pos_score=pos_score,
                    neg_score=neg_score,
                    margin=margin,
                    image_path=saved_path,
                )

        keep_record = keep or (not self.drop_skipped)
        meta = {
            "decision": reason,
            "kept": bool(keep),
            "record_persisted": bool(keep_record),
            "similarity_to_last_kept": similarity,
        }
        return keep_record, saved_path, meta

    def resolve_archive_image_path(self, image_path: Any) -> Path:
        if not self.archive_enabled:
            raise ValueError("detections archive is disabled")
        raw = str(image_path or "").strip()
        if not raw:
            raise ValueError("Missing image_path")
        path_obj = Path(raw).expanduser()
        if not path_obj.is_absolute():
            path_obj = (self.root / path_obj).resolve()
        else:
            path_obj = path_obj.resolve()
        if path_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported image file type")
        if not _path_within(path_obj, self.root):
            raise ValueError("image_path is outside detections archive")
        if not path_obj.exists() or not path_obj.is_file():
            raise ValueError("Image not found")
        return path_obj


detection_archive = _AdaptiveDetectionArchive()


class _ProbeBookmarkGate:
    def __init__(self) -> None:
        self.cooldown_ms = int(max(0.0, float(getattr(config, "PROBE_BOOKMARK_COOLDOWN_SEC", 8.0)) * 1000.0))
        self.dedupe_window_ms = int(
            max(500.0, float(getattr(config, "PROBE_BOOKMARK_DEDUPE_WINDOW_SEC", 20.0)) * 1000.0)
        )
        self.sim_high = float(getattr(config, "PROBE_BOOKMARK_SIM_HIGH", 0.985))
        self.margin_delta_thr = float(getattr(config, "PROBE_BOOKMARK_MARGIN_DELTA", 0.08))
        self.score_delta_thr = float(getattr(config, "PROBE_BOOKMARK_SCORE_DELTA", 0.08))
        self.max_frame_gap = int(max(1, int(getattr(config, "PROBE_BOOKMARK_MAX_FRAME_GAP", 8))))
        self.max_states = 4096
        self.keep_states = 2500
        self._lock = threading.RLock()
        self._state: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _normalize_vec(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vec is None:
            return None
        try:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            return None
        return (arr / norm).astype(np.float32, copy=False)

    @staticmethod
    def _state_key(channel_id: int, probe_key: str) -> str:
        return f"{int(channel_id)}:{probe_key}"

    @staticmethod
    def _estimate_frame_interval_ms(fps_hint: Optional[float]) -> int:
        if fps_hint is not None and fps_hint > 0:
            return max(1, int(round(1000.0 / fps_hint)))
        snapshot_interval = max(1, int(getattr(config, "LUXRIOT_SNAPSHOT_INTERVAL", 5)))
        return snapshot_interval * 1000

    def probe_config(self, probe_like: Mapping[str, Any]) -> Dict[str, Any]:
        cooldown_sec = _to_optional_float(probe_like.get("bookmark_cooldown_sec"))
        dedupe_window_sec = _to_optional_float(probe_like.get("bookmark_dedupe_window_sec"))
        return {
            "cooldown_ms": int(
                max(
                    0.0,
                    (cooldown_sec if cooldown_sec is not None else (self.cooldown_ms / 1000.0)) * 1000.0,
                )
            ),
            "dedupe_window_ms": int(
                max(
                    500.0,
                    (dedupe_window_sec if dedupe_window_sec is not None else (self.dedupe_window_ms / 1000.0)) * 1000.0,
                )
            ),
            "sim_high": self.sim_high,
            "margin_delta_thr": self.margin_delta_thr,
            "score_delta_thr": self.score_delta_thr,
            "max_frame_gap": self.max_frame_gap,
        }

    def _prune_locked(self) -> None:
        if len(self._state) <= self.max_states:
            return
        newest = sorted(
            self._state.items(),
            key=lambda item: int(item[1].get("timestamp_ms") or 0),
            reverse=True,
        )[: self.keep_states]
        self._state = dict(newest)

    def evaluate(
        self,
        *,
        channel_id: int,
        probe_key: str,
        timestamp_ms: int,
        clip_vec: Optional[np.ndarray],
        pos_score: float,
        neg_score: float,
        margin: float,
        fps_hint: Optional[float],
        probe_config: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        key = self._state_key(channel_id, probe_key)
        ts_ms = int(timestamp_ms) if int(timestamp_ms) > 0 else int(time.time() * 1000)
        frame_interval_ms = self._estimate_frame_interval_ms(fps_hint)
        normalized_vec = self._normalize_vec(clip_vec)
        cfg = dict(probe_config or {})
        cooldown_ms = int(max(0, int(cfg.get("cooldown_ms", self.cooldown_ms) or self.cooldown_ms)))
        dedupe_window_ms = int(max(500, int(cfg.get("dedupe_window_ms", self.dedupe_window_ms) or self.dedupe_window_ms)))
        sim_high = float(cfg.get("sim_high", self.sim_high) or self.sim_high)
        margin_delta_thr = float(cfg.get("margin_delta_thr", self.margin_delta_thr) or self.margin_delta_thr)
        score_delta_thr = float(cfg.get("score_delta_thr", self.score_delta_thr) or self.score_delta_thr)
        max_frame_gap = int(max(1, int(cfg.get("max_frame_gap", self.max_frame_gap) or self.max_frame_gap)))

        with self._lock:
            prev = self._state.get(key)
            if prev is None:
                return True, {
                    "reason": "bootstrap",
                    "timestamp_ms": ts_ms,
                    "dt_ms": None,
                    "similarity": None,
                    "frame_gap": None,
                }

            prev_ts = int(prev.get("timestamp_ms") or 0)
            dt_ms = max(0, ts_ms - prev_ts)
            frame_gap = float(dt_ms) / float(frame_interval_ms)
            similarity: Optional[float] = None

            prev_vec = prev.get("clip_vec")
            if (
                isinstance(prev_vec, np.ndarray)
                and normalized_vec is not None
                and prev_vec.shape == normalized_vec.shape
            ):
                similarity = float(np.clip(np.dot(prev_vec, normalized_vec), -1.0, 1.0))

            margin_delta = abs(float(margin) - float(prev.get("margin") or 0.0))
            pos_delta = abs(float(pos_score) - float(prev.get("pos_score") or 0.0))
            neg_delta = abs(float(neg_score) - float(prev.get("neg_score") or 0.0))

            if cooldown_ms > 0 and dt_ms < cooldown_ms:
                return False, {
                    "reason": "cooldown",
                    "timestamp_ms": ts_ms,
                    "dt_ms": dt_ms,
                    "similarity": similarity,
                    "frame_gap": frame_gap,
                }

            stable_scores = (
                margin_delta < margin_delta_thr
                and pos_delta < score_delta_thr
                and neg_delta < score_delta_thr
            )
            if (
                dt_ms < dedupe_window_ms
                and similarity is not None
                and similarity >= sim_high
                and stable_scores
                and frame_gap <= float(max_frame_gap)
            ):
                return False, {
                    "reason": "similar_recent_hit",
                    "timestamp_ms": ts_ms,
                    "dt_ms": dt_ms,
                    "similarity": similarity,
                    "frame_gap": frame_gap,
                }

            return True, {
                "reason": "novel_or_spaced",
                "timestamp_ms": ts_ms,
                "dt_ms": dt_ms,
                "similarity": similarity,
                "frame_gap": frame_gap,
            }

    def mark_sent(
        self,
        *,
        channel_id: int,
        probe_key: str,
        timestamp_ms: int,
        clip_vec: Optional[np.ndarray],
        pos_score: float,
        neg_score: float,
        margin: float,
    ) -> None:
        key = self._state_key(channel_id, probe_key)
        normalized_vec = self._normalize_vec(clip_vec)
        with self._lock:
            self._state[key] = {
                "timestamp_ms": int(timestamp_ms),
                "clip_vec": normalized_vec.copy() if normalized_vec is not None else None,
                "pos_score": float(pos_score),
                "neg_score": float(neg_score),
                "margin": float(margin),
            }
            self._prune_locked()


probe_bookmark_gate = _ProbeBookmarkGate()


def _select_probe_bookmark_hit(hits: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    best_hit: Optional[Mapping[str, Any]] = None
    best_key: Optional[Tuple[int, float]] = None
    for hit in hits:
        ts_ms = _to_int(hit.get("timestamp_ms"), 0)
        margin = _to_float(hit.get("margin"), -1.0)
        key = (ts_ms, margin)
        if best_key is None or key > best_key:
            best_hit = hit
            best_key = key
    return best_hit


def _maybe_send_probe_bookmark(
    probe_like: Mapping[str, Any],
    hit: Mapping[str, Any],
    *,
    source: str,
) -> Tuple[bool, Dict[str, Any]]:
    if not bool(probe_like.get("bookmark", False)):
        return False, {"reason": "bookmark_disabled", "source": source}

    channel_id = _to_int(probe_like.get("channel_id"), config.LUXRIOT_DEFAULT_CHANNEL_ID)
    probe_key = _probe_bookmark_identity(probe_like)
    probe_name = str(probe_like.get("name") or "probe")
    severity = str(probe_like.get("severity") or "critical")
    ts_ms = _to_int(hit.get("timestamp_ms"), int(time.time() * 1000))
    pos_score = _to_float(hit.get("pos_score"), 0.0)
    neg_score = _to_float(hit.get("neg_score"), 0.0)
    margin = _to_float(hit.get("margin"), 0.0)
    fps_hint = _to_optional_float(probe_like.get("fps"))
    clip_vec = _embed_thumbnail_b64(hit.get("thumbnail"), "clip")
    gate_config = probe_bookmark_gate.probe_config(probe_like)

    allow, gate_meta = probe_bookmark_gate.evaluate(
        channel_id=channel_id,
        probe_key=probe_key,
        timestamp_ms=ts_ms,
        clip_vec=clip_vec,
        pos_score=pos_score,
        neg_score=neg_score,
        margin=margin,
        fps_hint=fps_hint,
        probe_config=gate_config,
    )
    gate_meta["source"] = source
    gate_meta["cooldown_sec"] = round(float(gate_config.get("cooldown_ms", 0)) / 1000.0, 3)
    gate_meta["dedupe_window_sec"] = round(float(gate_config.get("dedupe_window_ms", 0)) / 1000.0, 3)
    if not allow:
        gate_meta["sent"] = False
        return False, gate_meta

    try:
        luxriot_manager.send_bookmark_event(
            channel_id=channel_id,
            title=f"Probe hit: {probe_name}",
            description=f"pos {pos_score:.3f} / neg {neg_score:.3f} · margin {margin:.3f}",
            severity=severity,
            state="new",
            timestamp_ms=ts_ms,
        )
    except Exception as exc:
        gate_meta["sent"] = False
        gate_meta["reason"] = "send_error"
        gate_meta["error"] = str(exc)
        return False, gate_meta

    probe_bookmark_gate.mark_sent(
        channel_id=channel_id,
        probe_key=probe_key,
        timestamp_ms=ts_ms,
        clip_vec=clip_vec,
        pos_score=pos_score,
        neg_score=neg_score,
        margin=margin,
    )
    gate_meta["sent"] = True
    gate_meta["reason"] = "sent"
    return True, gate_meta


def _store_probe_hits(
    probe_like: Dict[str, Any],
    hits: Sequence[Dict[str, Any]],
    *,
    source: str,
    bookmark_sent: bool = False,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> int:
    if not hits:
        return 0
    probe_id = _probe_identity(probe_like)
    channel_id = int(probe_like.get("channel_id") or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    probe_name = str(probe_like.get("name") or "").strip() or probe_id
    severity = str(probe_like.get("severity") or "normal").strip().lower() or "normal"
    bookmark_enabled = bool(probe_like.get("bookmark", False))
    now_ms = int(time.time() * 1000)
    records: List[Dict[str, Any]] = []
    def _resolve_archive_thumbnail(hit_thumb: Any, ts_ms: int) -> Any:
        try:
            hq_thumb = luxriot_manager.probe_frame_thumbnail(channel_id, ts_ms)
            if hq_thumb:
                return hq_thumb
        except Exception:
            pass
        return hit_thumb

    def _hit_sort_key(item: Tuple[int, Dict[str, Any]]) -> int:
        return _to_int(item[1].get("timestamp_ms"), now_ms)
    ordered_hits = sorted(
        enumerate(hits),
        key=_hit_sort_key,
    )
    for idx, hit in ordered_hits:
        ts_raw = hit.get("timestamp_ms")
        ts_ms = _to_int(ts_raw, now_ms)
        pos_score = _to_float(hit.get("pos_score"), 0.0)
        neg_score = _to_float(hit.get("neg_score"), 0.0)
        margin = _to_float(hit.get("margin"), 0.0)
        thumbnail_b64 = hit.get("thumbnail")
        archive_thumbnail_b64 = _resolve_archive_thumbnail(thumbnail_b64, ts_ms)
        clip_vec = _embed_thumbnail_b64(thumbnail_b64, "clip")
        raw_image_path = str(hit.get("image_path") or hit.get("path") or "").strip() or None
        keep_record, saved_image_path, retention_meta = detection_archive.handle_hit(
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            timestamp_ms=ts_ms,
            clip_vec=clip_vec,
            thumbnail_b64=archive_thumbnail_b64,
            pos_score=pos_score,
            neg_score=neg_score,
            margin=margin,
            image_path=raw_image_path,
        )
        if not keep_record:
            continue
        image_path = saved_image_path or raw_image_path
        payload = {
            "hit_index": idx,
            "probe_window_sec": probe_like.get("window_sec"),
            "probe_fps": probe_like.get("fps"),
            "source": source,
            "image_path": image_path,
            "retention": retention_meta,
            "hit": {
                "timestamp_ms": ts_ms,
                "channel_id": channel_id,
                "pos_score": pos_score,
                "neg_score": neg_score,
                "margin": margin,
            },
        }
        if isinstance(extra_payload, dict) and extra_payload:
            payload["context"] = extra_payload
        records.append(
            {
                "dedupe_key": f"{probe_id}:{source}:{ts_ms}:{pos_score:.4f}:{neg_score:.4f}:{margin:.4f}",
                "timestamp_ms": ts_ms,
                "probe_id": probe_id,
                "probe_name": probe_name,
                "channel_id": channel_id,
                "severity": severity,
                "bookmark_enabled": bookmark_enabled,
                "bookmark_sent": bookmark_sent,
                "pos_score": pos_score,
                "neg_score": neg_score,
                "margin": margin,
                "thumbnail_b64": thumbnail_b64,
                "clip_vec": clip_vec,
                "image_path": image_path,
                "source": source,
                "payload": payload,
            }
        )
    if not records:
        return 0
    try:
        return detections_store.add_detections(records)
    except Exception as exc:
        print(f"Detections store write failed for {probe_id}: {exc}")
        return 0


def _build_image_messages(image_path: str, prompt: str) -> List[Dict[str, Any]]:
    user_prompt = (prompt or '').strip() or "Describe the main content of this image clearly and concisely."
    user_content = [
        {'type': 'text', 'text': f"Image: {Path(image_path).name}\n\nTask: {user_prompt}"},
        {
            'type': 'image_url',
            'image_url': {
                'url': f"data:image/jpeg;base64,{_encode_jpeg(Image.open(image_path), max_edge=config.THUMBNAIL_SIZE[0])}",
                'detail': 'high',
            },
        },
    ]
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'You are an expert visual analyst. Be concise and factual.'}]},
        {'role': 'user', 'content': user_content},
    ]


def _rgb_to_hex(color: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _probe_daemon() -> None:
    """Background runner: execute enabled probes across channels."""
    while not probe_daemon_stop.is_set():
        try:
            probes = probes_store.list_probes()
            # Group probes by channel
            by_channel: Dict[int, List[Dict[str, Any]]] = {}
            for p in probes:
                if p.get("enabled") is False:
                    continue
                ch = int(p.get("channel_id", config.LUXRIOT_DEFAULT_CHANNEL_ID))
                by_channel.setdefault(ch, []).append(p)
            for ch, plist in by_channel.items():
                try:
                    if luxriot_manager.is_probe_capture_paused(ch):
                        continue
                    # Ensure capture running for this channel
                    try:
                        fps_desired = max([p.get('fps') or 0 for p in plist] or [0])
                        luxriot_manager.start_probe_capture(
                            ch,
                            fps=fps_desired if fps_desired > 0 else None,
                            clear_pause=False,
                        )
                    except Exception as exc:
                        print(f"Probe daemon failed to start capture for channel {ch}: {exc}")
                    for probe in plist:
                        probe_roi_enabled, probe_roi_norm = _parse_probe_roi(probe)
                        result = probe_manager.query(
                            probe.get('channel_id', config.LUXRIOT_DEFAULT_CHANNEL_ID),
                            probe.get('positives', []),
                            probe.get('negatives', []),
                            probe.get('pos_floor', 0.2),
                            probe.get('margin', 0.05),
                            probe.get('top_k', 6),
                            window_sec=probe.get('window_sec', 300.0),
                            image_probe=probe.get('image_probe'),
                            roi_norm=probe_roi_norm if probe_roi_enabled else None,
                            roi_padding=PROBE_ROI_PADDING,
                        )
                        if 'error' in result:
                            continue
                        hits = result.get('results') or []
                        if hits:
                            probe['last_hit'] = hits[0]
                            recent = probe.get('recent_hits') or []
                            recent = (hits + recent)[:PROBE_MAX_STORED_HITS]
                            probe['recent_hits'] = recent
                            bookmark_sent = False
                            bookmark_gate: Dict[str, Any] = {"reason": "bookmark_disabled", "source": "probe_daemon"}
                            if probe.get('bookmark'):
                                bookmark_hit = _select_probe_bookmark_hit(cast(Sequence[Mapping[str, Any]], hits))
                                if bookmark_hit is not None:
                                    bookmark_sent, bookmark_gate = _maybe_send_probe_bookmark(
                                        probe,
                                        bookmark_hit,
                                        source='probe_daemon',
                                    )
                                    if (not bookmark_sent) and str(bookmark_gate.get("reason") or "") == "send_error":
                                        print(
                                            "Probe daemon failed to send bookmark for probe {}: {}".format(
                                                probe.get('id'),
                                                bookmark_gate.get("error") or "unknown error",
                                            )
                                        )
                            probe['bookmark_gate'] = bookmark_gate
                            probe['bookmark_gate_updated_at_ms'] = int(time.time() * 1000)
                            _store_probe_hits(
                                probe,
                                hits,
                                source='probe_daemon',
                                bookmark_sent=bookmark_sent,
                                extra_payload={
                                    'frames_indexed': result.get('frames_indexed'),
                                    'roi_enabled': probe_roi_enabled,
                                    'roi_norm': _probe_roi_norm_to_payload(probe_roi_norm),
                                    'bookmark_gate': bookmark_gate,
                                },
                            )
                            probes_store.upsert_probe(probe)
                except Exception as exc:
                    print(f"Probe daemon channel loop error (channel {ch}): {exc}")
                    continue
        except Exception as exc:
            print(f"Probe daemon loop error: {exc}")
        probe_daemon_stop.wait(PROBE_DAEMON_INTERVAL_SEC)

def _render_segmentation_overlay(
    seg_map: np.ndarray,
    label_lookup: Optional[Dict[int, str]],
    highlight_id: Optional[int],
) -> Tuple[Optional[Image.Image], List[Dict[str, Any]]]:
    if seg_map is None or not isinstance(seg_map, np.ndarray):
        return None, []
    seg_int = np.asarray(seg_map, dtype=np.int32)
    if seg_int.ndim != 2:
        return None, []
    height, width = seg_int.shape
    if height == 0 or width == 0:
        return None, []

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    legend: List[Dict[str, Any]] = []

    labels = label_lookup or {}
    unique_ids = np.unique(seg_int)
    for class_id in unique_ids:
        class_int = int(class_id)
        color = _class_color(class_int)
        mask = seg_int == class_int
        alpha = 220 if highlight_id is not None and class_int == highlight_id else 150
        rgba[..., :3][mask] = color
        rgba[..., 3][mask] = alpha
        legend.append(
            {
                'id': class_int,
                'label': labels.get(class_int, f'class_{class_int}'),
                'color': _rgb_to_hex(color),
                'highlight': bool(highlight_id is not None and class_int == highlight_id),
            }
        )

    legend.sort(key=lambda entry: (0 if entry['highlight'] else 1, entry['label']))

    overlay_img = Image.fromarray(rgba, mode='RGBA')
    return overlay_img, legend


def _available_indexes(folder_path: Union[str, Path]) -> List[str]:
    available: List[str] = []
    root = Path(folder_path) / config.INDEX_FOLDER_NAME
    has_clip = False
    has_dino = False
    for embedder in EMBEDDER_SUBDIRS:
        embed_dir = _index_directory(folder_path, embedder)
        if (embed_dir / 'index.faiss').exists():
            available.append(embedder)
            if embedder == 'clip':
                has_clip = True
            elif embedder == 'dino':
                has_dino = True
            continue
        if embedder == 'clip' and (root / 'index.faiss').exists():
            available.append(embedder)
            has_clip = True
    if config.FUSION_ENABLED and has_clip and has_dino:
        available.append('fusion')
    return available


def save_index(index_results: Dict[str, Tuple[faiss.Index, List[str], List[Dict[str, Any]], Dict[str, Any]]], folder_path) -> None:
    """Persist FAISS indexes for one or more embedders."""
    for embedder, (index, image_paths, image_metadata, index_meta) in index_results.items():
        embed_dir = _index_directory(folder_path, embedder)
        embed_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(embed_dir / 'index.faiss'))

        with open(embed_dir / 'paths.pkl', 'wb') as f:
            pickle.dump(image_paths, f)

        with open(embed_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(image_metadata, f)

        meta_path = embed_dir / 'meta.json'
        meta_path.write_text(json.dumps(index_meta, indent=2), encoding='utf-8')


def load_index(folder_path: Union[str, Path], embedder: Optional[str] = None) -> FaissIndexBundle:
    """Load FAISS index, metadata, and embedder info for the requested backend."""
    target = embedder or active_embedder
    if target not in EMBEDDER_SUBDIRS:
        return None, None, None, {}

    root = Path(folder_path) / config.INDEX_FOLDER_NAME
    embed_dir = _index_directory(folder_path, target)
    legacy_dir = root if target == 'clip' else None
    if not embed_dir.exists():
        if legacy_dir is None or not (legacy_dir / 'index.faiss').exists():
            return None, None, None, {}
        embed_dir = legacy_dir

    meta: Dict[str, Any] = {}
    meta_path = embed_dir / 'meta.json'
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            meta = {}

    try:
        index = faiss.read_index(str(embed_dir / 'index.faiss'))

        with open(embed_dir / 'paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)

        image_metadata = None
        metadata_file = embed_dir / 'metadata.pkl'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    image_metadata = pickle.load(f)
            except Exception:
                image_metadata = None

        return index, image_paths, image_metadata, meta
    except Exception as exc:
        print(f"Error loading index for {target}: {exc}")
        return None, None, None, meta


def save_segment_index(
    folder_path: Union[str, Path],
    embeddings: np.ndarray,
    segment_metadata: List[Dict[str, Any]],
) -> None:
    segment_dir = _segment_index_directory(folder_path)
    segment_dir.mkdir(parents=True, exist_ok=True)

    index_path = segment_dir / 'segment_index.faiss'
    metadata_path = segment_dir / 'segments.pkl'
    info_path = segment_dir / 'meta.json'

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Segment embeddings must be a 2D array")

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != embeddings.shape[1]:
            raise ValueError(
                f"Segment embedding dimension mismatch: existing index expects {index.d}, got {embeddings.shape[1]}"
            )
        with open(metadata_path, 'rb') as fh:
            existing_meta: List[Dict[str, Any]] = pickle.load(fh)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        existing_meta = []

    _faiss_add_vectors(index, embeddings)
    existing_meta.extend(segment_metadata)

    faiss.write_index(index, str(index_path))
    with open(metadata_path, 'wb') as fh:
        pickle.dump(existing_meta, fh)

    info = {
        'embedder': 'dino',
        'type': 'segments',
        'embedding_dim': embeddings.shape[1],
        'segment_count': len(existing_meta),
        'updated_at': time.time(),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')


def load_segment_index(folder_path: Union[str, Path]):
    segment_dir = _segment_index_directory(folder_path)
    index_path = segment_dir / 'segment_index.faiss'
    metadata_path = segment_dir / 'segments.pkl'
    info_path = segment_dir / 'meta.json'

    if not index_path.exists() or not metadata_path.exists():
        return None, [], {}

    try:
        index = faiss.read_index(str(index_path))
        with open(metadata_path, 'rb') as fh:
            segment_meta: List[Dict[str, Any]] = pickle.load(fh)
        meta_info = {}
        if info_path.exists():
            try:
                meta_info = json.loads(info_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                meta_info = {}
        return index, segment_meta, meta_info
    except Exception as exc:
        print(f"Error loading segment index: {exc}")
        return None, [], {}


def load_comments(folder_path):
    """Load comments from JSON file."""
    index_path = Path(folder_path) / config.INDEX_FOLDER_NAME
    comments_file = index_path / 'comments.json'

    if not comments_file.exists():
        return {}

    try:
        return json.loads(comments_file.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def save_comments(folder_path, comments_data):
    """Persist comments to JSON."""
    index_path = Path(folder_path) / config.INDEX_FOLDER_NAME
    index_path.mkdir(exist_ok=True)
    comments_file = index_path / 'comments.json'

    try:
        comments_file.write_text(json.dumps(comments_data, ensure_ascii=False, indent=2), encoding='utf-8')
        return True
    except Exception as exc:
        print(f"Error saving comments: {exc}")
        return False


def get_image_comments(folder_path, image_path):
    """Fetch list of comments for a particular image."""
    comments_data = load_comments(folder_path)
    return comments_data.get(image_path, [])


def add_image_comment(folder_path, image_path, comment):
    """Append a comment to the image's history."""
    comments_data = load_comments(folder_path)

    if image_path not in comments_data:
        comments_data[image_path] = []

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comments_data[image_path].append(f"[{timestamp}] {comment}")

    return save_comments(folder_path, comments_data)


def _prepare_metadata_map(image_paths: Optional[List[str]], image_metadata: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not image_paths:
        return meta_map
    for idx, path in enumerate(image_paths):
        info: Dict[str, Any] = {}
        if image_metadata and idx < len(image_metadata):
            meta = image_metadata[idx] or {}
            if isinstance(meta, dict):
                info = meta
        meta_map[path] = info
    return meta_map


def _build_result_entry(img_path: str, similarity: float, metadata: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    result = {
        'path': img_path,
        'filename': os.path.basename(img_path),
        'similarity': float(similarity),
        'thumbnail': '',
        'metadata': dict(metadata or {}),
    }
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=config.THUMBNAIL_QUALITY)
            result['thumbnail'] = base64.b64encode(buffer.getvalue()).decode()
    except Exception as img_error:
        # Keep the result entry even if thumbnail creation fails so search never collapses to empty.
        # Frontend can still show filename/path and let operators inspect the source image directly.
        result['metadata']['thumbnail_error'] = str(img_error)
        print(f"Warning: thumbnail generation failed for {img_path}: {img_error}")
    if extra:
        result.update(extra)
    return result


def _should_rerank(sort_by: str) -> bool:
    return config.RERANK_ENABLED and config.RERANK_TOP_K > 0 and sort_by != 'time'


def _candidate_pool_size(limit: int, total: int, sort_by: str) -> int:
    if total <= 0:
        return 0
    limit = max(0, min(limit, total))
    if limit == 0:
        return 0
    if _should_rerank(sort_by):
        return min(total, max(limit, config.RERANK_TOP_K))
    return limit


def _collect_candidates(indices: np.ndarray, similarities: np.ndarray, max_index: int) -> List[Tuple[int, float]]:
    seen: Set[int] = set()
    candidates: List[Tuple[int, float]] = []
    flat_indices = indices[0] if indices.ndim > 1 else indices
    flat_sims = similarities[0] if similarities.ndim > 1 else similarities
    for idx, sim in zip(flat_indices, flat_sims):
        idx_int = int(idx)
        if 0 <= idx_int < max_index and idx_int not in seen:
            seen.add(idx_int)
            candidates.append((idx_int, float(sim)))
    return candidates


def _reconstruct_vectors(index: faiss.Index, ids: Sequence[int]) -> Optional[np.ndarray]:
    vectors: List[np.ndarray] = []
    for idx in ids:
        try:
            vec = _faiss_reconstruct(index, int(idx))
        except (AttributeError, RuntimeError, IndexError, TypeError):
            return None
        vectors.append(vec)
    if not vectors:
        return None
    try:
        return np.stack(vectors, axis=0)
    except ValueError:
        return None


def _rerank_candidates(index: faiss.Index, query_vec: np.ndarray, candidates: List[Tuple[int, float]], sort_by: str) -> List[Tuple[int, float, float, bool]]:
    if not candidates:
        return []

    if not _should_rerank(sort_by):
        return [(idx, score, score, False) for idx, score in candidates]

    rerank_count = min(len(candidates), config.RERANK_TOP_K)
    pool_ids = [idx for idx, _ in candidates[:rerank_count]]
    reconstructed = _reconstruct_vectors(index, pool_ids)
    if reconstructed is None:
        return [(idx, score, score, False) for idx, score in candidates]

    pool_vectors = reconstructed.astype(np.float32, copy=False)
    pool_norms = np.linalg.norm(pool_vectors, axis=1, keepdims=True)
    pool_norms[pool_norms == 0] = 1.0
    pool_vectors = pool_vectors / pool_norms

    query = np.asarray(query_vec, dtype=np.float32)
    if query.ndim > 1:
        query = query.reshape(-1)
    q_norm = np.linalg.norm(query)
    if q_norm == 0.0:
        q_norm = 1.0
    query = query / q_norm

    rerank_scores = pool_vectors @ query
    reranked = [
        (pool_ids[i], float(rerank_scores[i]), candidates[i][1], True)
        for i in range(len(pool_ids))
    ]
    reranked.sort(key=lambda item: item[1], reverse=True)

    remaining = [
        (idx, score, score, False) for idx, score in candidates[len(pool_ids):]
    ]
    return reranked + remaining


def _build_ranked_results(
    index: faiss.Index,
    query_vec: np.ndarray,
    indices: np.ndarray,
    similarities: np.ndarray,
    image_paths: Sequence[str],
    metadata_map: Dict[str, Dict[str, Any]],
    limit: int,
    sort_by: str,
) -> List[Dict[str, Any]]:
    candidates = _collect_candidates(indices, similarities, len(image_paths))
    baseline = [(idx, score, score, False) for idx, score in candidates]
    ranked = _rerank_candidates(index, query_vec, candidates, sort_by)

    if not ranked and baseline:
        ranked = baseline

    if ranked and len(ranked) != len(baseline) and baseline:
        ranked = baseline

    if not ranked:
        return []

    results: List[Dict[str, Any]] = []
    max_results = min(limit, len(ranked)) if limit > 0 else len(ranked)
    skipped = 0

    for idx, score, original_score, applied in ranked[:max_results]:
        path = image_paths[idx]
        extra = None
        if config.RERANK_ENABLED:
            extra = {
                'rerank': {
                    'applied': applied,
                    'score': float(score),
                    'original_score': float(original_score),
                }
            }
        entry = _build_result_entry(path, float(score), metadata_map.get(path, {}), extra=extra)
        if entry:
            results.append(entry)
        else:
            skipped += 1

    if skipped:
        print(f"Warning: skipped {skipped}/{max_results} ranked entries while building results.")

    return results


def _load_fusion_indexes(folder_path: Union[str, Path]) -> Tuple[FaissIndexBundle, FaissIndexBundle]:
    clip_data = load_index(folder_path, embedder='clip')
    dino_data = load_index(folder_path, embedder='dino')
    return clip_data, dino_data


def _merge_metadata_maps(primary: Dict[str, Dict[str, Any]], secondary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged = {path: dict(meta) for path, meta in primary.items()}
    for path, meta in secondary.items():
        if path not in merged:
            merged[path] = dict(meta)
        else:
            for key, value in meta.items():
                merged[path].setdefault(key, value)
    return merged


def _fuse_results(
    clip_data: FaissIndexBundle,
    dino_data: FaissIndexBundle,
    clip_vec: np.ndarray,
    dino_vec: np.ndarray,
    limit: int,
    sort_by: str,
) -> List[Dict[str, Any]]:
    clip_index, clip_paths, clip_metadata, _ = clip_data
    dino_index, dino_paths, dino_metadata, _ = dino_data
    if clip_index is None or dino_index is None:
        return []

    clip_map = _prepare_metadata_map(clip_paths, clip_metadata)
    dino_map = _prepare_metadata_map(dino_paths, dino_metadata)
    metadata_map = _merge_metadata_maps(clip_map, dino_map)

    limit = max(1, limit)
    alpha = max(0.0, min(1.0, float(config.FUSION_ALPHA)))

    def _search(index: Optional[faiss.Index], vec: np.ndarray, paths: Optional[Sequence[str]]) -> Dict[str, float]:
        if index is None or not paths:
            return {}
        k = min(limit * 2, len(paths))
        sims, inds = _faiss_search(index, vec.reshape(1, -1), k)
        scores: Dict[str, float] = {}
        for idx, sim in zip(inds[0], sims[0]):
            idx_int = int(idx)
            if 0 <= idx_int < len(paths):
                scores[paths[idx_int]] = float(sim)
        return scores

    clip_scores = _search(clip_index, clip_vec, clip_paths)
    dino_scores = _search(dino_index, dino_vec, dino_paths)

    combined: List[Tuple[str, float, float, float]] = []
    for path in set(list(clip_scores.keys()) + list(dino_scores.keys())):
        clip_sim = clip_scores.get(path, 0.0)
        dino_sim = dino_scores.get(path, 0.0)
        combined_score = (1.0 - alpha) * clip_sim + alpha * dino_sim
        combined.append((path, combined_score, clip_sim, dino_sim))

    if sort_by == 'time':
        combined.sort(key=lambda item: metadata_map.get(item[0], {}).get('mtime', 0), reverse=True)
    else:
        combined.sort(key=lambda item: item[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for path, score, clip_sim, dino_sim in combined:
        entry = _build_result_entry(
            path,
            score,
            metadata_map.get(path, {}),
            extra={
                'fusion': {
                    'clip_similarity': clip_sim,
                    'dino_similarity': dino_sim,
                    'alpha': alpha,
                }
            },
        )
        if entry:
            results.append(entry)
        if len(results) >= limit:
            break
    return results


def _normalize_detection_search_mode(requested: Optional[str]) -> str:
    mode = (requested or active_embedder or "clip").strip().lower()
    if mode == "fusion" and not config.FUSION_ENABLED:
        mode = "clip"
    if mode not in {"clip", "dino", "fusion"}:
        mode = "clip"
    return mode


def _parse_detection_filters(payload: Dict[str, Any], default_hours: float = DETECTIONS_SEARCH_DEFAULT_HOURS) -> Dict[str, Any]:
    probe_raw = str(payload.get("probe_id") or "").strip()
    probe_id = probe_raw or None

    channel_raw = str(payload.get("channel_id") or "").strip()
    channel_id: Optional[int] = None
    if channel_raw:
        channel_id = int(channel_raw)

    since_raw = payload.get("since_ms")
    until_raw = payload.get("until_ms")
    since_ms = _to_optional_int(since_raw)
    until_ms = _to_optional_int(until_raw)

    if since_ms is None:
        hours_raw = payload.get("hours")
        parsed_hours = _to_optional_float(hours_raw)
        hours = parsed_hours if parsed_hours is not None else float(default_hours)
        if hours > 0:
            since_ms = int(time.time() * 1000 - (hours * 3600 * 1000))

    return {
        "probe_id": probe_id,
        "channel_id": channel_id,
        "since_ms": since_ms,
        "until_ms": until_ms,
    }


def _backfill_clip_vectors_for_filters(
    probe_id: Optional[str],
    channel_id: Optional[int],
    since_ms: Optional[int],
    until_ms: Optional[int],
    *,
    expected_dim: Optional[int] = None,
    max_backfill: int = 2000,
) -> int:
    detections, _ = detections_store.list_detections(
        probe_id=probe_id,
        channel_id=channel_id,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=max_backfill,
        offset=0,
    )
    vector_by_id: Dict[int, np.ndarray] = {}
    if expected_dim is not None and detections:
        det_ids = [_to_optional_int(item.get("id")) for item in detections]
        rows = detections_store.fetch_detections_by_ids([det_id for det_id in det_ids if det_id is not None], include_vectors=True)
        for row in rows:
            det_id = _to_optional_int(row.get("id"))
            clip_vec = row.get("clip_vec")
            if det_id is None or not isinstance(clip_vec, np.ndarray):
                continue
            vector_by_id[det_id] = clip_vec
    pending: List[Tuple[int, Sequence[float]]] = []
    for item in detections:
        det_id = _to_optional_int(item.get("id"))
        if det_id is None:
            continue
        has_clip = bool(item.get("has_clip"))
        if has_clip:
            if expected_dim is None:
                continue
            existing = vector_by_id.get(det_id)
            if isinstance(existing, np.ndarray) and existing.ndim == 1 and int(existing.shape[0]) == expected_dim:
                continue
        thumb = item.get("thumbnail")
        if not thumb:
            continue
        vec = _embed_thumbnail_b64(thumb, "clip")
        if vec is None:
            continue
        if expected_dim is not None and vec.ndim == 1 and int(vec.shape[0]) != expected_dim:
            continue
        pending.append((det_id, cast(Sequence[float], vec)))
    if not pending:
        return 0
    updated = detections_store.update_clip_embeddings(pending)
    if updated > 0:
        detection_clip_shard_cache.clear()
    return updated


def _search_detection_clip_shards(
    candidates: Sequence[Dict[str, Any]],
    clip_query_vec: np.ndarray,
    limit: int,
) -> Tuple[List[Tuple[int, float]], Dict[int, Dict[str, Any]]]:
    candidate_map: Dict[int, Dict[str, Any]] = {}
    allowed_by_shard: Dict[str, Set[int]] = {}
    for item in candidates:
        det_id = _to_optional_int(item.get("id"))
        if det_id is None:
            continue
        candidate_map[det_id] = item
        shard = str(item.get("shard_key") or "").strip()
        allowed_by_shard.setdefault(shard, set()).add(det_id)

    ranked: List[Tuple[int, float]] = []
    seen: Set[int] = set()
    per_shard_k = max(DETECTIONS_SEARCH_SHARD_OVERFETCH, limit * 20)
    query_dim = int(clip_query_vec.shape[0]) if clip_query_vec.ndim == 1 else 0

    for shard_key, allowed_ids in allowed_by_shard.items():
        if not shard_key:
            continue
        index_obj, shard_ids = detection_clip_shard_cache.get(shard_key)
        if index_obj is None or shard_ids is None or shard_ids.size == 0:
            continue
        if query_dim > 0:
            index_dim = _to_optional_int(getattr(index_obj, "d", None))
            if index_dim is not None and index_dim != query_dim:
                continue
        k = min(int(shard_ids.size), per_shard_k)
        if k <= 0:
            continue
        sims, inds = _faiss_search(index_obj, clip_query_vec.reshape(1, -1), k)
        for local_idx, sim in zip(inds[0], sims[0]):
            local_int = int(local_idx)
            if local_int < 0 or local_int >= int(shard_ids.size):
                continue
            det_id = int(shard_ids[local_int])
            if det_id not in allowed_ids or det_id in seen:
                continue
            seen.add(det_id)
            ranked.append((det_id, float(sim)))

    if len(seen) < len(candidate_map):
        remaining = [det_id for det_id in candidate_map.keys() if det_id not in seen]
        if remaining:
            vec_rows = detections_store.fetch_detections_by_ids(remaining, include_vectors=True)
            fallback_ranked: List[Tuple[int, float]] = []
            for row in vec_rows:
                clip_vec = row.get("clip_vec")
                if not isinstance(clip_vec, np.ndarray):
                    continue
                if clip_vec.shape != clip_query_vec.shape:
                    continue
                det_id = _to_optional_int(row.get("id"))
                if det_id is None:
                    continue
                fallback_ranked.append((det_id, float(np.dot(clip_query_vec, clip_vec))))
            fallback_ranked.sort(key=lambda item: item[1], reverse=True)
            for det_id, score in fallback_ranked:
                if det_id in seen:
                    continue
                seen.add(det_id)
                ranked.append((det_id, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked, candidate_map


def _ensure_dino_vectors_for_ids(detection_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    ids_clean = [int(det_id) for det_id in detection_ids if det_id is not None]
    if not ids_clean:
        return {}
    rows = detections_store.fetch_detections_by_ids(ids_clean, include_vectors=True)
    dino_map: Dict[int, np.ndarray] = {}
    pending_updates: List[Tuple[int, Sequence[float]]] = []
    for row in rows:
        det_id = _to_optional_int(row.get("id"))
        if det_id is None:
            continue
        existing = row.get("dino_vec")
        if existing is not None:
            dino_map[det_id] = existing
            continue
        thumb = row.get("thumbnail")
        if not thumb:
            continue
        vec = _embed_thumbnail_b64(thumb, "dino")
        if vec is None:
            continue
        dino_map[det_id] = vec
        pending_updates.append((det_id, cast(Sequence[float], vec)))
    if pending_updates:
        detections_store.update_dino_embeddings(pending_updates)
    return dino_map


def _build_detection_search_result(
    item: Dict[str, Any],
    score: float,
    clip_score: float,
    dino_score: Optional[float],
    mode: str,
    alpha: float,
    dino_fallback: bool,
) -> Dict[str, Any]:
    ts_ms = int(item.get("timestamp_ms") or 0)
    ts_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000.0)) if ts_ms > 0 else "n/a"
    probe_label = str(item.get("probe_name") or item.get("probe_id") or "probe")
    image_path = str(item.get("image_path") or "").strip()
    payload_obj = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    if not image_path and isinstance(payload_obj, dict):
        image_path = str(payload_obj.get("image_path") or "").strip()

    result: Dict[str, Any] = {
        "path": image_path,
        "filename": f"{probe_label} · {ts_label}",
        "similarity": float(score),
        "thumbnail": item.get("thumbnail") or "",
        "metadata": {
            "mtime": ts_ms,
            "detection_id": item.get("id"),
            "source": item.get("source"),
            "probe_id": item.get("probe_id"),
            "probe_name": item.get("probe_name"),
            "channel_id": item.get("channel_id"),
        },
        "is_detection": True,
        "detection_id": item.get("id"),
        "timestamp_ms": ts_ms,
        "channel_id": item.get("channel_id"),
        "probe_id": item.get("probe_id"),
        "probe_name": item.get("probe_name"),
        "severity": item.get("severity"),
        "pos_score": float(item.get("pos_score") or 0.0),
        "neg_score": float(item.get("neg_score") or 0.0),
        "margin": float(item.get("margin") or 0.0),
        "source": item.get("source"),
        "shard_key": item.get("shard_key"),
        "search_mode": mode,
        "dino_fallback": bool(dino_fallback),
    }
    if mode in {"fusion", "dino"}:
        result["fusion"] = {
            "clip_similarity": float(clip_score),
            "dino_similarity": float(dino_score if dino_score is not None else clip_score),
            "alpha": float(alpha),
            "dino_fallback": bool(dino_fallback),
        }
    return result


def _search_detections_archive(
    *,
    clip_query_vec: np.ndarray,
    dino_query_vec: Optional[np.ndarray],
    mode: str,
    probe_id: Optional[str],
    channel_id: Optional[int],
    since_ms: Optional[int],
    until_ms: Optional[int],
    limit: int,
    sort_by: str,
    candidate_limit: int,
) -> List[Dict[str, Any]]:
    limit = max(1, min(config.MAX_RESULTS, int(limit or config.DEFAULT_RESULTS)))
    candidate_limit = max(limit, min(DETECTIONS_SEARCH_MAX_CANDIDATES, int(candidate_limit or 20000)))
    clip_dim = int(clip_query_vec.shape[0]) if clip_query_vec.ndim == 1 else None

    candidates = detections_store.list_vector_candidates(
        probe_id=probe_id,
        channel_id=channel_id,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=candidate_limit,
        only_with_clip=True,
        include_vectors=False,
    )
    if not candidates:
        updated = _backfill_clip_vectors_for_filters(
            probe_id,
            channel_id,
            since_ms,
            until_ms,
            expected_dim=clip_dim,
            max_backfill=min(candidate_limit, 2000),
        )
        if updated > 0:
            candidates = detections_store.list_vector_candidates(
                probe_id=probe_id,
                channel_id=channel_id,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=candidate_limit,
                only_with_clip=True,
                include_vectors=False,
            )
    if not candidates:
        return []

    clip_hits, candidate_map = _search_detection_clip_shards(candidates, clip_query_vec, limit)
    if not clip_hits:
        updated = _backfill_clip_vectors_for_filters(
            probe_id,
            channel_id,
            since_ms,
            until_ms,
            expected_dim=clip_dim,
            max_backfill=min(candidate_limit, 2000),
        )
        if updated > 0:
            candidates = detections_store.list_vector_candidates(
                probe_id=probe_id,
                channel_id=channel_id,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=candidate_limit,
                only_with_clip=True,
                include_vectors=False,
            )
            clip_hits, candidate_map = _search_detection_clip_shards(candidates, clip_query_vec, limit)
    if not clip_hits:
        return []

    alpha = max(0.0, min(1.0, float(config.FUSION_ALPHA)))
    if mode == "clip":
        alpha = 0.0
    elif mode == "dino":
        alpha = 1.0

    dino_scores: Dict[int, float] = {}
    if mode in {"dino", "fusion"} and dino_query_vec is not None:
        pool_size = min(len(clip_hits), max(DETECTIONS_SEARCH_DINO_POOL_MIN, limit * DETECTIONS_SEARCH_DINO_POOL_MULTIPLIER))
        pool_ids = [det_id for det_id, _ in clip_hits[:pool_size]]
        dino_vectors = _ensure_dino_vectors_for_ids(pool_ids)
        for det_id in pool_ids:
            vec = dino_vectors.get(det_id)
            if vec is None:
                continue
            dino_scores[det_id] = float(np.dot(dino_query_vec, vec))

    scored: List[Tuple[int, float, float, Optional[float], bool]] = []
    for det_id, clip_score in clip_hits:
        dino_score = dino_scores.get(det_id)
        dino_fallback = False
        if mode == "clip":
            final_score = clip_score
        elif dino_query_vec is None:
            dino_fallback = True
            final_score = clip_score
        elif mode == "dino":
            if dino_score is None:
                dino_fallback = True
                final_score = clip_score
            else:
                final_score = dino_score
        else:
            if dino_score is None:
                dino_fallback = True
                final_score = clip_score
            else:
                final_score = (1.0 - alpha) * clip_score + alpha * dino_score
        scored.append((det_id, float(final_score), float(clip_score), dino_score, dino_fallback))

    if sort_by == "time":
        scored.sort(key=lambda row: int(candidate_map.get(row[0], {}).get("timestamp_ms") or 0), reverse=True)
    else:
        scored.sort(key=lambda row: row[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for det_id, final_score, clip_score, dino_score, dino_fallback in scored[:limit]:
        item = candidate_map.get(det_id)
        if not item:
            continue
        results.append(
            _build_detection_search_result(
                item=item,
                score=final_score,
                clip_score=clip_score,
                dino_score=dino_score,
                mode=mode,
                alpha=alpha,
                dino_fallback=dino_fallback,
            )
        )
    return results


def _build_segment_search_results(
    segment_index: Optional[faiss.Index],
    segment_metadata: Sequence[Dict[str, Any]],
    query_vec: np.ndarray,
    limit: int,
) -> List[Dict[str, Any]]:
    if segment_index is None or not segment_metadata:
        return []

    if limit <= 0:
        return []

    k = min(limit, len(segment_metadata))
    if k == 0:
        return []

    similarities, indices = _faiss_search(segment_index, query_vec.reshape(1, -1), k)

    results: List[Dict[str, Any]] = []
    for idx, sim in zip(indices[0], similarities[0]):
        if not (0 <= idx < len(segment_metadata)):
            continue
        seg_meta = segment_metadata[idx] or {}
        image_path = seg_meta.get('image_path')
        if not image_path:
            continue
        metadata = {
            'segment_id': seg_meta.get('segment_id'),
            'segment_label': seg_meta.get('label'),
            'segment_area': seg_meta.get('area'),
            'segment_fraction': seg_meta.get('patch_fraction'),
        }
        entry = _build_result_entry(image_path, float(sim), metadata)
        if entry is None:
            continue
        entry['segment'] = {
            'id': seg_meta.get('segment_id'),
            'label': seg_meta.get('label'),
            'source': seg_meta.get('source'),
            'area': seg_meta.get('area'),
            'patch_fraction': seg_meta.get('patch_fraction'),
        }
        results.append(entry)
    return results


def _parse_targets(raw_value: Optional[Union[str, Sequence[str]]]) -> Set[str]:
    acceptable = {'images', 'segments'}
    if raw_value is None:
        return {'images'}
    if isinstance(raw_value, (list, tuple, set)):
        tokens = {str(item).lower() for item in raw_value}
    else:
        tokens = {str(raw_value).lower()}
    targets = {token for token in tokens if token in acceptable}
    return targets or {'images'}


def _has_non_full_segments(segments: Dict[str, Any]) -> bool:
    return any(str(seg_id) != 'full' for seg_id in segments.keys())


def _encode_mask_segments_with_fallback(
    image_input: Union[str, Path, Image.Image],
    mask_image: Image.Image,
    segment_ids: Optional[List[str]] = None,
    min_patches: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, Union[np.ndarray, float, int]]], int, bool]:
    ensure_embedder_loaded('dino')
    if dino_encoder is None:
        raise RuntimeError('DINO encoder is not available')

    requested_min = max(1, int(min_patches if min_patches is not None else config.DINO_SEGMENT_MIN_PATCHES))
    segments = dino_encoder.encode_masked(
        image_input,
        mask_image,
        segment_ids=segment_ids,
        min_patches=requested_min,
    )
    if requested_min <= 1:
        return segments, requested_min, False

    require_non_full = True
    if segment_ids is not None:
        normalized_ids = {str(seg).strip().lower() for seg in segment_ids if str(seg).strip()}
        require_non_full = bool(normalized_ids - {'full'})

    if segments and (not require_non_full or _has_non_full_segments(segments)):
        return segments, requested_min, False

    relaxed = dino_encoder.encode_masked(
        image_input,
        mask_image,
        segment_ids=segment_ids,
        min_patches=1,
    )
    if relaxed and (not require_non_full or _has_non_full_segments(relaxed)):
        print(
            f"Mask encoding fallback: relaxed min_patches from {requested_min} to 1 "
            f"(segment_ids={segment_ids if segment_ids is not None else 'auto'})"
        )
        return relaxed, 1, True

    return segments or relaxed, requested_min, False


def _mask_search_pipeline(
    image_input: Union[str, Path, Image.Image],
    mask_image: Image.Image,
    folder: Union[str, Path],
    limit: int,
    sort_by: str,
    target_modes: Set[str],
    segment_ids: Optional[List[str]] = None,
    label_map: Optional[Dict[str, Any]] = None,
):
    segments, _, _ = _encode_mask_segments_with_fallback(
        image_input,
        mask_image,
        segment_ids=segment_ids,
        min_patches=config.DINO_SEGMENT_MIN_PATCHES,
    )

    if not segments:
        return [], segments

    image_index, image_paths, image_metadata, _ = load_index(folder, embedder='dino')
    if image_index is None or not image_paths:
        raise RuntimeError('DINO index not available for the requested folder')

    metadata_map = _prepare_metadata_map(image_paths, image_metadata)
    segment_index, segment_metadata, _ = load_segment_index(folder)

    label_map = label_map or {}

    segments_response: List[Dict[str, Any]] = []
    for seg_id, info in segments.items():
        if seg_id == 'full':
            if segment_ids is not None and 'full' not in segment_ids:
                continue
            if segment_ids is None and any(key != 'full' for key in segments.keys()):
                continue

        embedding = np.asarray(info.get('embedding'))
        if embedding.size == 0:
            continue

        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            continue
        embedding = embedding / norm

        entry: Dict[str, Any] = {
            'segment_id': seg_id,
            'label': label_map.get(str(seg_id)) or label_map.get(seg_id),
            'patch_count': int(info.get('patch_count', 0)),
            'patch_fraction': float(info.get('patch_fraction', 0.0)),
        }

        if 'images' in target_modes:
            k = _candidate_pool_size(limit, len(image_paths), sort_by)
            similarities, indices = _faiss_search(image_index, embedding.reshape(1, -1), k)
            entry['image_results'] = _build_ranked_results(
                image_index,
                embedding,
                indices,
                similarities,
                image_paths,
                metadata_map,
                limit,
                sort_by,
            )

        if 'segments' in target_modes:
            entry['segment_results'] = _build_segment_search_results(
                segment_index,
                segment_metadata,
                embedding,
                limit,
            )

        segments_response.append(entry)

    return segments_response, segments


@app.route('/comments', methods=['GET'])
def get_comments():
    """Get comments for a specific image"""
    folder_raw = request.args.get('folder')
    image_path = request.args.get('image_path')
    
    if not folder_raw or not image_path:
        return jsonify({'error': 'Missing folder or image_path parameter'}), 400
    
    try:
        folder = str(_resolve_folder_path(folder_raw))
        comments = get_image_comments(folder, image_path)
        return jsonify({'comments': comments})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as e:
        print(f"Error getting comments: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/comments', methods=['POST'])
def save_comment():
    """Save a new comment for an image"""
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    folder_raw = data.get('folder')
    image_path = data.get('image_path')
    comment = data.get('comment', '').strip()
    
    if not folder_raw or not image_path or not comment:
        return jsonify({'error': 'Missing folder, image_path, or comment'}), 400
    
    # Basic input sanitization
    if len(comment) > config.MAX_COMMENT_LENGTH:
        return jsonify({'error': f'Comment too long (max {config.MAX_COMMENT_LENGTH} characters)'}), 400
    
    try:
        folder = str(_resolve_folder_path(folder_raw))
        success = add_image_comment(folder, image_path, comment)
        if success:
            comments = get_image_comments(folder, image_path)
            return jsonify({'success': True, 'comments': comments})
        else:
            return jsonify({'error': 'Failed to save comment'}), 500
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as e:
        print(f"Error saving comment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/commented_images', methods=['POST'])
def get_commented_images():
    """Get all images that have comments in the indexed folder"""
    payload = _json_body()
    folder_raw = payload.get('folder')
    if not folder_raw:
        return jsonify({'error': 'No folder specified'}), 400

    try:
        folder = str(_resolve_folder_path(folder_raw))
        available = _available_indexes(folder)
        image_paths: List[str] = []
        image_metadata: List[Dict[str, Any]] = []
        metadata_map: Dict[str, Dict[str, Any]] = {}

        if available:
            targets = [active_embedder] + [m for m in available if m != active_embedder]
            for emb in targets:
                idx_obj, paths, metas, meta_info = load_index(folder, embedder=emb)
                if idx_obj is not None:
                    image_paths = paths or []
                    image_metadata = metas or []
                    metadata_map = _prepare_metadata_map(image_paths, image_metadata)
                    break

        comments_data = load_comments(folder)
        if not comments_data:
            return jsonify({'results': []})

        results: List[Dict[str, Any]] = []
        for image_path, comment_list in comments_data.items():
            if not Path(image_path).exists():
                continue
            entry = _build_result_entry(
                image_path,
                similarity=1.0,
                metadata=metadata_map.get(image_path, {}),
                extra={
                    'comment_count': len(comment_list),
                    'latest_comment': comment_list[-1] if comment_list else '',
                },
            )
            if entry:
                results.append(entry)

        results.sort(key=lambda x: (x.get('metadata', {}).get('mtime', 0), x.get('comment_count', 0)), reverse=True)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_index', methods=['POST'])
def check_index():
    """Check if folder is indexed"""
    data = _json_body()
    folder_raw = data.get('folder')
    if not folder_raw:
        return jsonify({'error': 'No folder specified'}), 400
    try:
        folder = str(_resolve_folder_path(folder_raw))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    
    available = _available_indexes(folder)
    if active_embedder == 'fusion':
        indexed = 'fusion' in available or ({'clip', 'dino'}.issubset(set(available)))
    else:
        indexed = active_embedder in available
    response = {
        'indexed': indexed,
        'available_modes': available,
    }
    if not response['indexed'] and available:
        response['existing_embedder'] = available
    return jsonify(response)

@app.route('/index', methods=['POST'])
def index_folder():
    """Index a folder"""
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    folder_raw = data.get('folder')
    if not folder_raw:
        return jsonify({'error': 'Invalid folder path'}), 400
    
    try:
        folder = str(_resolve_folder_path(folder_raw))
        index_results = create_index(folder)
        if not index_results:
            return jsonify({'error': 'No images found in folder'}), 400

        save_index(index_results, folder)

        counts = {embedder: len(data[1]) for embedder, data in index_results.items()}
        metadata = {embedder: data[3] for embedder, data in index_results.items()}
        modes = list(index_results.keys())
        if config.FUSION_ENABLED and 'clip' in counts and 'dino' in counts:
            fusion_count = min(counts['clip'], counts['dino'])
            counts['fusion'] = fusion_count
            metadata['fusion'] = {'alpha': config.FUSION_ALPHA}
            modes.append('fusion')

        active_count = counts.get(active_embedder)
        if active_count is None and counts:
            active_count = next(iter(counts.values()))
        response = {
            'success': True,
            'counts': counts,
            'meta': metadata,
            'count': active_count or 0,
            'active_meta': metadata.get(active_embedder),
            'modes': modes,
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _parse_segment_ids(raw_value: Optional[Union[str, List[str], List[int]]]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in text.split(',') if part.strip()]


def _parse_segment_labels(raw_value: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return {str(k): v for k, v in raw_value.items()}
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    return {}


def _normalize_mask_image(mask_img: Image.Image) -> Image.Image:
    """Normalize uploaded/decoded mask images to a meaningful single-channel mask.

    If an alpha channel exists and is informative (not fully opaque), use alpha because
    UI overlays are RGBA with semantic mask stored in transparency.
    """
    bands = mask_img.getbands()
    if "A" in bands:
        try:
            alpha = mask_img.getchannel("A")
            alpha_np = np.asarray(alpha)
            if alpha_np.size and np.any(alpha_np > 0) and np.any(alpha_np < 255):
                return alpha.convert("L")
        except Exception:
            pass
    if mask_img.mode != "L":
        return mask_img.convert("L")
    return mask_img


def _load_mask_from_request() -> Optional[Image.Image]:
    mask_file = request.files.get('mask')
    if mask_file:
        return _normalize_mask_image(Image.open(mask_file.stream))
    payload = _json_body()
    mask_base64 = request.form.get('mask') or payload.get('mask')
    if mask_base64:
        try:
            mask_bytes = base64.b64decode(mask_base64)
            return _normalize_mask_image(Image.open(BytesIO(mask_bytes)))
        except Exception as exc:
            raise ValueError(f"Invalid mask payload: {exc}")
    return None


@app.route('/index_segments', methods=['POST'])
def index_segments():
    """Index DINO segment embeddings derived from a mask."""
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    if not config.DINO_SEGMENTS_ENABLED:
        return jsonify({'error': 'Segment indexing is disabled. Enable EVOSSEARCH_DINO_SEGMENTS_ENABLED to use this feature.'}), 400

    data = request.form if request.form else _json_body()

    folder_raw = data.get('folder')
    image_path = data.get('image_path')
    if not folder_raw or not image_path:
        return jsonify({'error': 'Both folder and image_path are required'}), 400
    try:
        folder_path = _resolve_folder_path(folder_raw)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    image_obj = Path(image_path).expanduser().resolve()
    if not image_obj.exists() or not image_obj.is_file():
        return jsonify({'error': f'Image file not found: {image_path}'}), 400
    if image_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return jsonify({'error': 'Unsupported image file type'}), 400
    if not _path_within(image_obj, folder_path):
        return jsonify({'error': 'image_path must be inside folder'}), 400

    mask_image = _load_mask_from_request()
    if mask_image is None:
        return jsonify({'error': 'Mask is required for segment indexing'}), 400

    segment_ids = _parse_segment_ids(data.get('segment_ids'))
    label_map = _parse_segment_labels(data.get('segment_labels'))

    try:
        segments, min_patches_used, min_patches_relaxed = _encode_mask_segments_with_fallback(
            str(image_obj),
            mask_image,
            segment_ids=segment_ids,
            min_patches=config.DINO_SEGMENT_MIN_PATCHES,
        )
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 500

    if not segments:
        return jsonify({'error': 'No valid segments were produced from the provided mask'}), 400

    entries: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    for seg_id, info in segments.items():
        if seg_id == 'full':
            continue
        embedding = info.get('embedding')
        if embedding is None:
            continue
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        entries.append(
            {
                'image_path': image_path,
                'segment_id': seg_id,
                'label': label_map.get(seg_id),
                'area': int(info.get('patch_count', 0)),
                'patch_fraction': float(info.get('patch_fraction', 0.0)),
                'source': 'index_segments',
                'created_at': time.time(),
            }
        )

    if not embeddings:
        segment_keys = [str(key) for key in segments.keys()]
        print(
            "index_segments: no non-full segments "
            f"(keys={segment_keys}, min_patches_used={min_patches_used}, relaxed={min_patches_relaxed})"
        )
        return jsonify(
            {
                'error': 'Mask did not yield any segments beyond the full image aggregate',
                'min_patches_used': min_patches_used,
                'min_patches_relaxed': min_patches_relaxed,
                'hint': 'Select a larger region or lower Min Segment Patches in settings.',
            }
        ), 400

    embedding_matrix = np.stack(embeddings, axis=0)
    save_segment_index(str(folder_path), embedding_matrix, entries)

    return jsonify(
        {
            'success': True,
            'segments_indexed': [
                {
                    'segment_id': entry['segment_id'],
                    'label': entry.get('label'),
                    'patch_count': entry.get('area'),
                    'patch_fraction': entry.get('patch_fraction'),
                }
                for entry in entries
            ],
            'min_patches_used': min_patches_used,
            'min_patches_relaxed': min_patches_relaxed,
        }
    )


@app.route('/video_understanding', methods=['POST'])
def video_understanding():
    data = _json_body()
    video_path = (data.get('video') or '').strip()
    if not video_path:
        return jsonify({'error': 'Provide a video path.'}), 400
    video_obj = Path(video_path).expanduser().resolve()
    if not video_obj.exists() or not video_obj.is_file():
        return jsonify({'error': f'Video file not found: {video_path}'}), 400
    if config.ALLOWED_ROOTS:
        allowed_roots = [Path(item).expanduser().resolve() for item in config.ALLOWED_ROOTS]
        if not any(_path_within(video_obj, root) for root in allowed_roots):
            return jsonify({'error': 'Video path is outside configured allowed roots'}), 400
    max_frames = data.get('frame_count') or config.LM_VIDEO_DEFAULT_FRAMES
    try:
        max_frames_int = int(max_frames)
    except (TypeError, ValueError):
        max_frames_int = config.LM_VIDEO_DEFAULT_FRAMES
    if max_frames_int < 1:
        max_frames_int = 1
    max_frames_int = min(max_frames_int, config.LM_VIDEO_MAX_FRAMES)

    sample_fps_raw = data.get('sample_fps')
    try:
        sample_fps_val = float(sample_fps_raw) if sample_fps_raw is not None else None
        if sample_fps_val is not None and sample_fps_val <= 0:
            sample_fps_val = None
    except (TypeError, ValueError):
        sample_fps_val = None

    user_prompt = data.get('prompt') or ''
    model_hint = (data.get('model') or '').strip()

    try:
        frames, fps, duration = _sample_video_frames(
            str(video_obj),
            max_frames=max_frames_int,
            sample_fps=sample_fps_val,
            max_edge=config.LM_VIDEO_MAX_EDGE,
        )
        if not frames:
            return jsonify({'error': 'No frames could be extracted from the video.'}), 400
        messages = _build_video_messages(str(video_obj), frames, user_prompt)
        summary = _call_video_understanding(messages, model_override=model_hint or None)
        return jsonify(
            {
                'summary': summary,
                'frames': [
                    {
                        'index': f['index'],
                        'time_sec': f['time_sec'],
                        'thumbnail': f['thumbnail'],
                    }
                    for f in frames
                ],
                'fps': fps,
                'duration_sec': duration,
                'model': model_hint or config.LM_MODEL,
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/describe_image', methods=['POST'])
def describe_image():
    data = _json_body()
    folder_raw = data.get('folder')
    image_path = (data.get('image_path') or '').strip()
    prompt = data.get('prompt') or ''
    model_hint = (data.get('model') or '').strip()
    if not image_path:
        return jsonify({'error': 'image_path is required'}), 400
    try:
        if folder_raw:
            folder_path = _resolve_folder_path(folder_raw, require_index=True)
            path_obj = Path(image_path).expanduser().resolve()
            if path_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                return jsonify({'error': 'Unsupported image file type'}), 400
            if not path_obj.exists() or not path_obj.is_file():
                return jsonify({'error': f'Image not found: {image_path}'}), 400
            if not _path_within(path_obj, folder_path):
                return jsonify({'error': 'image_path must be inside folder'}), 400
        else:
            path_obj = detection_archive.resolve_archive_image_path(image_path)
        messages = _build_image_messages(str(path_obj), prompt)
        summary = _call_lm_chat(messages, model_override=model_hint or None)
        with Image.open(path_obj) as src:
            thumb = _encode_jpeg(src, max_edge=config.THUMBNAIL_SIZE[0])
        return jsonify(
            {
                'summary': summary,
                'thumbnail': thumb,
                'model': model_hint or config.LM_MODEL,
                'image_path': str(path_obj),
            }
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
@app.route('/search', methods=['POST'])
def search():
    """Search for images using text queries."""
    data = _json_body()
    folder_raw = data.get('folder')
    query = data.get('query')
    limit = data.get('limit', 10)
    sort_by = data.get('sort_by', 'similarity')  # 'similarity' or 'time'
    if not folder_raw or not query:
        return jsonify({'error': 'Missing folder or query'}), 400
    try:
        folder = str(_resolve_folder_path(folder_raw, require_index=True))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    fusion_active = active_embedder == 'fusion' and config.FUSION_ENABLED
    search_mode = 'clip' if fusion_active else active_embedder

    if search_mode != 'clip':
        return jsonify({'error': 'Text search is only available when using the CLIP backend.'}), 400

    try:
        limit = int(limit)
        if limit < config.MIN_RESULTS or limit > config.MAX_RESULTS:
            limit = config.DEFAULT_RESULTS
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS

    index, image_paths, image_metadata, index_meta = load_index(folder, embedder=search_mode)
    if index is None or not image_paths:
        message = 'Folder not indexed for the current backend'
        available = _available_indexes(folder)
        if available:
            message += f" (available: {', '.join(available)})"
        return jsonify({'error': message}), 400

    try:
        text_embedding = get_text_embedding(query)
    except RuntimeError as err:
        return jsonify({'error': str(err)}), 400

    try:
        k = _candidate_pool_size(limit, len(image_paths), sort_by)
        if k == 0:
            print(f"Text search: query='{query}' candidates=0 returned=0 folder='{folder}'")
            return jsonify({'results': []})
        similarities, indices = _faiss_search(index, text_embedding.reshape(1, -1), k)

        metadata_map = _prepare_metadata_map(image_paths, image_metadata)
        candidate_count = len(_collect_candidates(indices, similarities, len(image_paths)))
        results = _build_ranked_results(
            index,
            text_embedding,
            indices,
            similarities,
            image_paths,
            metadata_map,
            limit,
            sort_by,
        )

        if sort_by == 'time':
            results.sort(key=lambda item: item['metadata'].get('mtime', 0), reverse=True)

        print(
            f"Text search: query='{query}' candidates={candidate_count} "
            f"returned={len(results)} folder='{folder}'"
        )
        return jsonify({'results': results})
    except Exception as e:
        print(f"Text search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/search_by_image', methods=['POST'])
def search_by_image():
    """Search for images using an uploaded image"""
    folder_raw = request.form.get('folder')
    limit = request.form.get('limit', 12)
    sort_by = request.form.get('sort_by', 'similarity')  # 'similarity' or 'time'

    if not folder_raw:
        return jsonify({'error': 'Missing folder'}), 400
    try:
        folder = str(_resolve_folder_path(folder_raw, require_index=True))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        limit = int(limit)
        if limit < config.MIN_RESULTS or limit > config.MAX_RESULTS:
            limit = config.DEFAULT_RESULTS
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS

    file = request.files.get('image')
    image_path = request.form.get('image_path')

    if not file and not image_path:
        return jsonify({'error': 'No image uploaded or path provided'}), 400

    if file and file.filename == '':
        file = None

    fusion_active = active_embedder == 'fusion' and config.FUSION_ENABLED

    clip_data: Optional[FaissIndexBundle] = None
    dino_data: Optional[FaissIndexBundle] = None
    index: Optional[faiss.Index] = None
    image_paths: List[str] = []
    image_metadata: List[Dict[str, Any]] = []

    if fusion_active:
        clip_data, dino_data = _load_fusion_indexes(folder)
        clip_index, clip_paths, clip_metadata, _ = clip_data
        dino_index, dino_paths, dino_metadata, _ = dino_data
        if clip_index is None or dino_index is None:
            message = 'Fusion requires both CLIP and DINO indexes'
            available = _available_indexes(folder)
            if available:
                message += f" (available: {', '.join(available)})"
            return jsonify({'error': message}), 400
    else:
        index, loaded_paths, loaded_metadata, index_meta = load_index(folder, embedder=active_embedder)
        if index is None or not loaded_paths:
            message = 'Folder not indexed for the current backend'
            available = _available_indexes(folder)
            if available:
                message += f" (available: {', '.join(available)})"
            return jsonify({'error': message}), 400
        image_paths = loaded_paths
        image_metadata = loaded_metadata or []

    try:
        dino_vec: Optional[np.ndarray] = None
        if file:
            uploaded_image = Image.open(file.stream)
            if uploaded_image.mode != 'RGB':
                uploaded_image = uploaded_image.convert('RGB')
            clip_vec = get_image_embedding_from_pil(uploaded_image, embedder='clip') if fusion_active else get_image_embedding_from_pil(uploaded_image, embedder=active_embedder)
            if fusion_active:
                dino_vec = get_image_embedding_from_pil(uploaded_image, embedder='dino')
        else:
            image_obj = Path(str(image_path)).expanduser().resolve()
            if image_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                return jsonify({'error': 'Unsupported image file type'}), 400
            if not image_obj.exists() or not image_obj.is_file():
                return jsonify({'error': f'Image file not found: {image_path}'}), 400
            if not _path_within(image_obj, Path(folder)):
                return jsonify({'error': 'image_path must be inside folder'}), 400
            clip_vec = get_image_embedding(image_obj, embedder='clip') if fusion_active else get_image_embedding(image_obj, embedder=active_embedder)
            if fusion_active:
                dino_vec = get_image_embedding(image_obj, embedder='dino')

        if fusion_active:
            if clip_data is None or dino_data is None or dino_vec is None:
                return jsonify({'error': 'Fusion search requires both CLIP and DINO query embeddings'}), 500
            results = _fuse_results(clip_data, dino_data, clip_vec, dino_vec, limit, sort_by)
            return jsonify({'results': results})

        if index is None or not image_paths:
            return jsonify({'error': 'Backend index is unavailable for image search'}), 400
        k = _candidate_pool_size(limit, len(image_paths), sort_by)
        if k == 0:
            print(f"Image search: candidates=0 returned=0 folder='{folder}'")
            return jsonify({'results': []})
        similarities, indices = _faiss_search(index, clip_vec.reshape(1, -1), k)

        metadata_map = _prepare_metadata_map(image_paths, image_metadata)
        candidate_count = len(_collect_candidates(indices, similarities, len(image_paths)))
        results = _build_ranked_results(
            index,
            clip_vec,
            indices,
            similarities,
            image_paths,
            metadata_map,
            limit,
            sort_by,
        )

        if sort_by == 'time':
            results.sort(key=lambda item: item['metadata'].get('mtime', 0), reverse=True)

        print(
            f"Image search: candidates={candidate_count} returned={len(results)} "
            f"folder='{folder}'"
        )
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search_by_mask', methods=['POST'])
def search_by_mask():
    """Search using a masked region of an image leveraging DINO segment embeddings."""
    data = request.form if request.form else _json_body()

    if not config.DINO_SEGMENTS_ENABLED:
        return jsonify({'error': 'Segment search is disabled. Enable EVOSSEARCH_DINO_SEGMENTS_ENABLED to use this feature.'}), 400

    folder_raw = data.get('folder')
    if not folder_raw:
        return jsonify({'error': 'Missing folder'}), 400
    try:
        folder_path = _resolve_folder_path(folder_raw, require_index=True)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    folder = str(folder_path)

    limit = data.get('limit', config.DEFAULT_RESULTS)
    sort_by = data.get('sort_by', 'similarity')
    target_modes_raw = data.get('targets') or data.get('target')
    segment_ids = _parse_segment_ids(data.get('segment_ids'))
    label_map = _parse_segment_labels(data.get('segment_labels'))

    try:
        limit = int(limit)
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS

    try:
        limit = max(config.MIN_RESULTS, min(limit, config.MAX_RESULTS))
    except Exception:
        limit = config.DEFAULT_RESULTS

    target_modes = _parse_targets(target_modes_raw)

    image_source = data.get('image_path')
    uploaded_image = request.files.get('image')

    if not image_source and uploaded_image is None:
        return jsonify({'error': 'Provide image_path or upload an image file'}), 400

    mask_image = _load_mask_from_request()
    if mask_image is None:
        return jsonify({'error': 'Mask is required for masked search'}), 400

    ensure_embedder_loaded('dino')
    if dino_encoder is None:
        return jsonify({'error': 'DINO encoder is not available'}), 500

    if uploaded_image:
        query_image = Image.open(uploaded_image.stream)
        if query_image.mode != 'RGB':
            query_image = query_image.convert('RGB')
        image_input: Union[Image.Image, str] = query_image
    else:
        image_obj = Path(str(image_source)).expanduser().resolve()
        if image_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return jsonify({'error': 'Unsupported image file type'}), 400
        if not image_obj.exists():
            return jsonify({'error': f'Image file not found: {image_source}'}), 400
        if not _path_within(image_obj, folder_path):
            return jsonify({'error': 'image_path must be inside folder'}), 400
        image_input = str(image_obj)

    try:
        segment_map = dino_encoder.encode_masked(
            image_input,
            mask_image,
            segment_ids=segment_ids,
            min_patches=config.DINO_SEGMENT_MIN_PATCHES,
        )
    except Exception as exc:
        return jsonify({'error': f'Failed to compute masked embeddings: {exc}'}), 500

    if not segment_map:
        return jsonify({'error': 'Mask did not produce any valid segments'}), 400

    try:
        segments_response, _ = _mask_search_pipeline(
            image_input,
            mask_image,
            folder,
            limit,
            sort_by,
            target_modes,
            segment_ids,
            label_map,
        )
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 400

    if not segments_response:
        return jsonify({'error': 'No segment results produced'}), 400

    return jsonify(
        {
            'segments': segments_response,
            'targets': list(target_modes),
            'segment_count': len(segments_response),
        }
    )


@app.route('/segment_from_point', methods=['POST'])
def segment_from_point():
    """Derive a mask from a clicked point and run masked search."""
    data = request.form if request.form else _json_body()
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    folder_raw = data.get('folder')
    if not folder_raw:
        return jsonify({'error': 'Missing folder'}), 400
    try:
        folder_path = _resolve_folder_path(folder_raw, require_index=True)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    folder = str(folder_path)

    image_path = data.get('image_path')
    uploaded_image = request.files.get('image')
    if not image_path and uploaded_image is None:
        return jsonify({'error': 'Provide image_path or upload an image file'}), 400

    x_raw = data.get('x')
    y_raw = data.get('y')
    if x_raw is None or y_raw is None:
        return jsonify({'error': 'Invalid or missing x/y coordinates'}), 400
    try:
        x_norm = float(x_raw)
        y_norm = float(y_raw)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid or missing x/y coordinates'}), 400

    try:
        threshold = float(data.get('threshold', config.DINO_HEATMAP_THRESHOLD))
    except (TypeError, ValueError):
        threshold = config.DINO_HEATMAP_THRESHOLD
    threshold = min(max(threshold, 0.0), 0.99)

    limit = data.get('limit', config.DEFAULT_RESULTS)
    sort_by = data.get('sort_by', 'similarity')
    target_modes = _parse_targets(data.get('targets') or data.get('target'))
    segment_ids = _parse_segment_ids(data.get('segment_ids'))
    label_map = _parse_segment_labels(data.get('segment_labels'))
    if not isinstance(label_map, dict):
        label_map = {}

    try:
        limit = int(limit)
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS
    try:
        limit = max(config.MIN_RESULTS, min(limit, config.MAX_RESULTS))
    except Exception:
        limit = config.DEFAULT_RESULTS

    ensure_embedder_loaded('dino')
    if dino_encoder is None:
        return jsonify({'error': 'DINO encoder is not available'}), 500

    pil_image: Optional[Image.Image] = None
    if uploaded_image:
        pil_image = Image.open(uploaded_image.stream)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
    else:
        image_obj = Path(str(image_path)).expanduser().resolve()
        if image_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return jsonify({'error': 'Unsupported image file type'}), 400
        if not image_obj.exists() or not image_obj.is_file():
            return jsonify({'error': f'Image file not found: {image_path}'}), 400
        if not _path_within(image_obj, folder_path):
            return jsonify({'error': 'image_path must be inside folder'}), 400
        with Image.open(image_obj) as src:
            pil_image = src.convert('RGB')

    assert pil_image is not None
    image_input: Union[str, Path, Image.Image] = pil_image

    try:
        heatmap, _, grid, patch_coords = dino_encoder.patch_similarity_map(
            image_input,
            x_norm,
            y_norm,
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    flat = heatmap.reshape(-1)
    quantile = np.quantile(flat, threshold) if flat.size else 0.0
    mask_bool = heatmap >= quantile
    if not mask_bool.any() and flat.size:
        mask_bool.flat[int(np.argmax(flat))] = True

    coarse_mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    mask_fraction = float(np.count_nonzero(coarse_mask_uint8)) / float(coarse_mask_uint8.size) if coarse_mask_uint8.size else 0.0

    heatmap_norm = heatmap - heatmap.min()
    if heatmap_norm.max() > 0:
        heatmap_norm = heatmap_norm / heatmap_norm.max()

    crop_size = getattr(dino_encoder, 'crop_size', 224)
    mask_img = Image.fromarray(coarse_mask_uint8).resize((crop_size, crop_size), resample=RESAMPLE_NEAREST)

    base_size = pil_image.size
    overlay_mask_source = Image.fromarray(coarse_mask_uint8).resize(base_size, resample=RESAMPLE_NEAREST)
    refinement_source = 'dino_heatmap'
    refined_label: Optional[str] = None
    segment_value = 255
    refine_result: Optional[Dict[str, Any]] = None

    head = ensure_mask_head()
    if head is not None:
        try:
            refine_result = head.refine(pil_image, coarse_mask_uint8, (x_norm, y_norm))
            if refine_result and isinstance(refine_result, dict):
                refined_mask_arr = refine_result.get('mask')
                if isinstance(refined_mask_arr, np.ndarray) and refined_mask_arr.any():
                    refinement_source = 'mask2former'
                    refined_label = refine_result.get('label')
                    mask_fraction = float(refine_result.get('mask_fraction', mask_fraction))
                    segment_value = int(refine_result.get('segment_value', 255))
                    overlay_mask_source = Image.fromarray(refined_mask_arr.astype(np.uint8), mode='L')
                    if overlay_mask_source.size != base_size:
                        overlay_mask_source = overlay_mask_source.resize(base_size, resample=RESAMPLE_NEAREST)
                    mask_img = overlay_mask_source.resize((crop_size, crop_size), resample=RESAMPLE_NEAREST)
                    if refined_label:
                        label_map[str(segment_value)] = refined_label
        except Exception as exc:
            print(f"Mask2Former refinement error: {exc}")

    # Update mask coverage fraction based on overlay image size
    overlay_mask_np = np.asarray(overlay_mask_source)
    if overlay_mask_np.size:
        mask_fraction = float(np.count_nonzero(overlay_mask_np)) / float(overlay_mask_np.size)

    heatmap_alpha = Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize(base_size, resample=RESAMPLE_BILINEAR)
    heatmap_overlay = _create_overlay_rgba(heatmap_alpha, (255, 155, 40), 0.9)
    mask_overlay = _create_overlay_rgba(overlay_mask_source, (94, 196, 255), 0.6)
    segmentation_overlay_img: Optional[Image.Image] = None
    legend_entries: List[Dict[str, Any]] = []

    if refine_result:
        seg_map = refine_result.get('segmentation')
        class_labels_raw = refine_result.get('class_labels')
        class_labels = {}
        if isinstance(class_labels_raw, dict):
            class_labels = {int(k): str(v) for k, v in class_labels_raw.items()}
        seg_overlay: Optional[Image.Image] = None
        if isinstance(seg_map, np.ndarray):
            seg_overlay, legend_entries = _render_segmentation_overlay(
                seg_map,
                class_labels,
                int(refine_result.get('class_id', segment_value)),
            )
        if seg_overlay is not None:
            if seg_overlay.size != base_size:
                seg_overlay = seg_overlay.resize(base_size, resample=RESAMPLE_NEAREST)
            segmentation_overlay_img = seg_overlay

    try:
        segments_response, segment_map = _mask_search_pipeline(
            image_input,
            mask_img,
            folder,
            limit,
            sort_by,
            target_modes,
            segment_ids,
            label_map,
        )
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 400

    if not segments_response:
        return jsonify({'error': 'No segment results produced'}), 400

    selected_key = next((key for key in segment_map.keys() if key != 'full'), 'full')
    selected_meta = segment_map.get(selected_key, {})

    overlay = {
        'grid_size': grid,
        'patch_coords': {'x': patch_coords[0], 'y': patch_coords[1]},
        'threshold': threshold,
        'heatmap_png': _image_to_base64(heatmap_overlay),
        'mask_png': _image_to_base64(mask_overlay),
        'mask_raw_png': _image_to_base64(overlay_mask_source.convert('L')),
        'mask_fraction': mask_fraction,
        'refinement': refinement_source,
        'refined_label': refined_label,
        'segment_value': segment_value,
        'patch_count': int(selected_meta.get('patch_count', 0)),
        'patch_fraction': float(selected_meta.get('patch_fraction', 0.0)),
    }
    if segmentation_overlay_img is not None:
        overlay['segmentation_png'] = _image_to_base64(segmentation_overlay_img)
    if legend_entries:
        overlay['legend'] = legend_entries

    return jsonify(
        {
            'segments': segments_response,
            'targets': list(target_modes),
            'segment_count': len(segments_response),
            'overlay': overlay,
        }
    )


@app.route('/luxriot/channels', methods=['GET'])
def luxriot_channels():
    """Fetch available Luxriot channels (cached briefly)."""
    force = str(request.args.get('force', '')).lower() in {'1', 'true', 'yes'}
    try:
        channels = luxriot_manager.get_channels(force=force)
        context = _current_auth_context()
        if _auth_enabled() and context is not None:
            channels = [
                channel
                for channel in channels
                if _can_access_context_channel(context, channel.get("id"))
            ]
        return jsonify({'channels': channels})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/snapshot/<int:channel_id>', methods=['GET'])
def luxriot_snapshot(channel_id: int):
    stream_type = request.args.get('stream', 'mainStream')
    try:
        encoded, meta = luxriot_manager.get_snapshot_base64(channel_id, stream_type=stream_type)
        img_bytes = base64.b64decode(encoded)
        response = make_response(img_bytes)
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-store, must-revalidate'
        response.headers['X-Image-Width'] = str(meta.get('width'))
        response.headers['X-Image-Height'] = str(meta.get('height'))
        return response
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/start_capture', methods=['POST'])
def luxriot_start_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or data.get('id') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    batch_size = data.get('batch_size')
    prompt = data.get('prompt') or ''
    model_hint = (data.get('model') or '').strip() or None
    system_prompt = (data.get('system_prompt') or '').strip() or None
    try:
        status = luxriot_manager.start_session(
            channel_id,
            batch_size=batch_size,
            prompt=prompt,
            model_hint=model_hint,
            system_prompt=system_prompt,
        )
        return jsonify({'success': True, 'session': status})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/prompt_settings', methods=['GET', 'POST'])
def luxriot_prompt_settings():
    if request.method == 'GET':
        channel_id = request.args.get('channel_id', default=None, type=int)
        try:
            return jsonify(luxriot_manager.get_prompt_settings(channel_id=channel_id))
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    channel_id = _to_optional_int(data.get('channel_id'))
    if channel_id is None:
        channel_id = _to_optional_int(data.get('channel'))
    if channel_id is None:
        channel_id = _to_optional_int(data.get('id'))
    if channel_id is None:
        return jsonify({'error': 'Provide a valid channel_id'}), 400

    stream_system_prompt: Optional[str] = None
    if 'stream_system_prompt' in data:
        raw_stream_prompt = data.get('stream_system_prompt')
        stream_system_prompt = '' if raw_stream_prompt is None else str(raw_stream_prompt)
    elif 'system_prompt' in data:
        raw_stream_prompt = data.get('system_prompt')
        stream_system_prompt = '' if raw_stream_prompt is None else str(raw_stream_prompt)

    json_alert_prompt: Optional[str] = None
    if 'json_alert_prompt' in data:
        raw_json_prompt = data.get('json_alert_prompt')
        json_alert_prompt = '' if raw_json_prompt is None else str(raw_json_prompt)
    elif 'json_prompt' in data:
        raw_json_prompt = data.get('json_prompt')
        json_alert_prompt = '' if raw_json_prompt is None else str(raw_json_prompt)

    bookmark_enabled: Optional[bool] = None
    if 'bookmark_enabled' in data:
        bookmark_enabled = _coerce_bool(data.get('bookmark_enabled'), default=False)
    elif 'enable_bookmarks' in data:
        bookmark_enabled = _coerce_bool(data.get('enable_bookmarks'), default=False)

    bookmark_cooldown_sec: Optional[float] = None
    if 'bookmark_cooldown_sec' in data:
        bookmark_cooldown_sec = max(0.0, _to_float(data.get('bookmark_cooldown_sec'), default=0.0))
    elif 'cooldown_sec' in data:
        bookmark_cooldown_sec = max(0.0, _to_float(data.get('cooldown_sec'), default=0.0))

    rollup_prompt_updates: Optional[Dict[str, Any]] = None
    rollup_prompts_raw = data.get('rollup_prompts')
    if isinstance(rollup_prompts_raw, Mapping):
        rollup_prompt_updates = {
            str(level): value
            for level, value in rollup_prompts_raw.items()
        }
    else:
        inline_updates: Dict[str, Any] = {}
        for level in ('L1', 'L2', 'L3'):
            field_name = f'rollup_prompt_{level.lower()}'
            if field_name in data:
                inline_updates[level] = data.get(field_name)
        if inline_updates:
            rollup_prompt_updates = inline_updates

    try:
        settings = luxriot_manager.update_prompt_settings(
            channel_id=channel_id,
            stream_system_prompt=stream_system_prompt,
            rollup_prompts=rollup_prompt_updates,
            json_alert_prompt=json_alert_prompt,
            bookmark_enabled=bookmark_enabled,
            bookmark_cooldown_sec=bookmark_cooldown_sec,
        )
        return jsonify({'success': True, **settings})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/stop_capture', methods=['POST'])
def luxriot_stop_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    try:
        state = luxriot_manager.stop_session(channel_id)
        return jsonify({'success': True, 'session': state})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/flush_capture', methods=['POST'])
def luxriot_flush_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    try:
        result = luxriot_manager.flush_session(channel_id)
        if not result.get('success'):
            return jsonify(result), 400
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/session', methods=['GET'])
def luxriot_session_status():
    channel_id = request.args.get('channel_id', default=config.LUXRIOT_DEFAULT_CHANNEL_ID, type=int)
    run_selector = (request.args.get('run') or '').strip() or None
    from_ts = request.args.get('from_ts', default=None, type=float)
    to_ts = request.args.get('to_ts', default=None, type=float)
    limit = request.args.get('limit', default=None, type=int)
    try:
        status = luxriot_manager.session_status(
            channel_id,
            run_selector=run_selector,
            start_ts=from_ts,
            end_ts=to_ts,
            limit=limit,
        )
        return jsonify(status)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/rollups', methods=['GET'])
def luxriot_summary_rollups():
    channel_id = request.args.get('channel_id', default=config.LUXRIOT_DEFAULT_CHANNEL_ID, type=int)
    run_selector = (request.args.get('run') or '').strip() or None
    from_ts = request.args.get('from_ts', default=None, type=float)
    to_ts = request.args.get('to_ts', default=None, type=float)
    level_limit = request.args.get('level_limit', default=60, type=int)
    try:
        rollups = luxriot_manager.summary_rollups(
            channel_id=channel_id,
            run_selector=run_selector,
            start_ts=from_ts,
            end_ts=to_ts,
            level_limit=level_limit,
        )
        return jsonify(rollups)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/streams', methods=['GET'])
def luxriot_streams_status():
    try:
        status = luxriot_manager.streams_status()
        context = _current_auth_context()
        if _auth_enabled() and context is not None:
            for key in ("video_streams", "analytics_streams"):
                status[key] = [
                    item
                    for item in status.get(key) or []
                    if _can_access_context_channel(
                        context,
                        item.get("channel_id"),
                    )
                ]
            for key in (
                "paused_analytics_channels",
                "video_history_channels",
            ):
                status[key] = [
                    channel_id
                    for channel_id in status.get(key) or []
                    if _can_access_context_channel(context, channel_id)
                ]
            status["running_total"] = sum(
                len(status.get(key) or [])
                for key in ("video_streams", "analytics_streams")
            )
        return jsonify(status)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/streams/stop', methods=['POST'])
def luxriot_stop_stream():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    stream_type = (data.get('stream_type') or 'both').strip().lower()
    pause_analytics = _coerce_bool(data.get('pause_analytics'), True)
    try:
        result = luxriot_manager.stop_stream(channel_id, stream_type=stream_type, pause_analytics=pause_analytics)
        return jsonify({'success': True, 'result': result, 'streams': luxriot_manager.streams_status()})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/streams/stop_all', methods=['POST'])
def luxriot_stop_all_streams():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    stop_video = _coerce_bool(data.get('stop_video'), True)
    stop_analytics = _coerce_bool(data.get('stop_analytics'), True)
    pause_analytics = _coerce_bool(data.get('pause_analytics'), True)
    if not stop_video and not stop_analytics:
        return jsonify({'error': 'Select at least one stream type to stop'}), 400
    try:
        result = luxriot_manager.stop_all_streams(
            stop_video=stop_video,
            stop_analytics=stop_analytics,
            pause_analytics=pause_analytics,
        )
        return jsonify({'success': True, 'result': result, 'streams': luxriot_manager.streams_status()})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/bookmark', methods=['POST'])
def luxriot_bookmark():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    title = (data.get('title') or '').strip() or 'External event'
    description = data.get('description') or ''
    severity = (data.get('severity') or 'critical').strip().lower()
    state = (data.get('state') or 'new').strip().lower()
    timestamp_ms = data.get('timestamp_ms')
    try:
        if timestamp_ms is not None:
            timestamp_ms = int(timestamp_ms)
    except Exception:
        timestamp_ms = None
    try:
        result = luxriot_manager.send_bookmark_event(
            channel_id=channel_id,
            title=title,
            description=description,
            severity=severity,
            state=state,
            timestamp_ms=timestamp_ms,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/probes/query', methods=['POST'])
def probes_query():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    positives = data.get('positives') or []
    negatives = data.get('negatives') or []
    try:
        pos_floor = float(data.get('pos_floor', 0.2))
    except Exception:
        pos_floor = 0.2
    try:
        margin_thr = float(data.get('margin', 0.05))
    except Exception:
        margin_thr = 0.05
    try:
        top_k = int(data.get('top_k', 5))
    except Exception:
        top_k = 5
    try:
        window_sec = float(data.get('window_sec', 0))
    except Exception:
        window_sec = 0
    probe_roi_enabled, probe_roi_norm = _parse_probe_roi(data)
    probe_like = {
        "id": data.get('id'),
        "name": (data.get('name') or 'probe'),
        "channel_id": channel_id,
        "severity": (data.get('severity') or 'critical'),
        "bookmark": bool(data.get('bookmark')),
        "window_sec": window_sec,
        "fps": data.get('fps'),
        "roi_enabled": probe_roi_enabled,
        "roi_norm": _probe_roi_norm_to_payload(probe_roi_norm),
    }
    result = probe_manager.query(
        channel_id,
        positives,
        negatives,
        pos_floor,
        margin_thr,
        top_k,
        window_sec=window_sec,
        image_probe=data.get('image_probe'),
        roi_norm=probe_roi_norm if probe_roi_enabled else None,
        roi_padding=PROBE_ROI_PADDING,
    )
    status_code = 200 if 'error' not in result else 400
    hits = result.get('results') or []
    bookmark_sent = False
    bookmark_gate: Dict[str, Any] = {"reason": "bookmark_disabled", "source": "probes_query"}
    if hits and data.get('bookmark'):
        bookmark_hit = _select_probe_bookmark_hit(cast(Sequence[Mapping[str, Any]], hits))
        if bookmark_hit is not None:
            bookmark_sent, bookmark_gate = _maybe_send_probe_bookmark(
                probe_like,
                bookmark_hit,
                source='probes_query',
            )
    result['bookmark_gate'] = bookmark_gate
    if hits:
        # trim recent hits (kept in request payload only; not persisted unless saved)
        recent_hits = data.get('recent_hits') or []
        recent_hits = (recent_hits + hits)[:PROBE_MAX_STORED_HITS]
        result['recent_hits'] = recent_hits
        result['persisted_hits'] = _store_probe_hits(
            probe_like,
            hits,
            source='probes_query',
            bookmark_sent=bookmark_sent,
            extra_payload={
                'frames_indexed': result.get('frames_indexed'),
                'roi_enabled': probe_roi_enabled,
                'roi_norm': _probe_roi_norm_to_payload(probe_roi_norm),
                'bookmark_gate': bookmark_gate,
            },
        )
    else:
        result['persisted_hits'] = 0
    return jsonify(result), status_code


@app.route('/probes/status', methods=['GET'])
def probes_status():
    channel_id = request.args.get('channel_id', default=config.LUXRIOT_DEFAULT_CHANNEL_ID, type=int)
    try:
        return jsonify(probe_manager.status(channel_id))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/probes/start_capture', methods=['POST'])
def probes_start_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    try:
        fps = data.get('fps')
        fps_val = None
        try:
            if fps is not None:
                fps_val = float(fps)
        except Exception:
            fps_val = None
        clear_pause = _coerce_bool(data.get('clear_pause'), True)
        state = luxriot_manager.start_probe_capture(channel_id, fps=fps_val, clear_pause=clear_pause)
        return jsonify({'success': True, 'state': state})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/probes/stop_capture', methods=['POST'])
def probes_stop_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    try:
        pause = _coerce_bool(data.get('pause'), True)
        state = luxriot_manager.stop_probe_capture(channel_id, pause=pause)
        return jsonify({'success': True, 'state': state})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/probes/save', methods=['POST'])
def probes_save():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    positives = [str(x).strip() for x in (data.get('positives') or []) if str(x).strip()]
    negatives = [str(x).strip() for x in (data.get('negatives') or []) if str(x).strip()]
    image_probe = data.get('image_probe') or {}
    if not positives and not (image_probe.get('data') and image_probe.get('enabled', True) is not False):
        return jsonify({'error': 'Provide at least one positive (text or image).'}), 400

    def _float(val, default):
        try:
            return float(val)
        except Exception:
            return default

    def _int(val, default):
        try:
            return int(val)
        except Exception:
            return default

    probe_roi_enabled, probe_roi_norm = _parse_probe_roi(data)

    existing_probe: Dict[str, Any] = {}
    probe_id_raw = data.get('id')
    if probe_id_raw:
        try:
            existing_probe = next(
                (p for p in probes_store.list_probes() if str(p.get('id')) == str(probe_id_raw)),
                {},
            )
        except Exception:
            existing_probe = {}

    probe = {
        "id": data.get('id') or None,
        "name": (data.get('name') or '').strip() or f"probe-{int(time.time())}",
        "channel_id": channel_id,
        "positives": positives,
        "negatives": negatives,
        "pos_floor": _float(data.get('pos_floor'), 0.2),
        "margin": max(0.0, _float(data.get('margin'), 0.05)),
        "bookmark_cooldown_sec": max(
            0.0,
            _float(
                data.get('bookmark_cooldown_sec'),
                existing_probe.get('bookmark_cooldown_sec', config.PROBE_BOOKMARK_COOLDOWN_SEC),
            ),
        ),
        "bookmark_dedupe_window_sec": max(
            0.5,
            _float(
                data.get('bookmark_dedupe_window_sec'),
                existing_probe.get('bookmark_dedupe_window_sec', config.PROBE_BOOKMARK_DEDUPE_WINDOW_SEC),
            ),
        ),
        "top_k": _int(data.get('top_k'), 6),
        "window_sec": _float(data.get('window_sec'), 300.0),
        "severity": (data.get('severity') or 'critical').lower(),
        "bookmark": bool(data.get('bookmark', True)),
        "enabled": bool(data.get('enabled', True)),
        "image_probe": image_probe,
        "roi_enabled": probe_roi_enabled,
        "roi_norm": _probe_roi_norm_to_payload(probe_roi_norm),
        "pairs": data.get('pairs') or [],
        "last_hit": data.get('last_hit'),
        "recent_hits": (data.get('recent_hits') or [])[:PROBE_MAX_STORED_HITS],
        "bookmark_gate": existing_probe.get("bookmark_gate"),
        "bookmark_gate_updated_at_ms": existing_probe.get("bookmark_gate_updated_at_ms"),
    }
    saved = probes_store.upsert_probe(probe)
    return jsonify({'success': True, 'probe': saved})


@app.route('/probes/list', methods=['GET'])
def probes_list():
    probes = probes_store.list_probes()
    context = _current_auth_context()
    if _auth_enabled() and context is not None:
        probes = [
            probe
            for probe in probes
            if _can_access_context_channel(
                context,
                probe.get("channel_id"),
            )
        ]
    response = jsonify({'probes': probes})
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/probes/delete', methods=['POST'])
def probes_delete():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    probe_id = data.get('id')
    if not probe_id:
        return jsonify({'error': 'Provide probe id'}), 400
    ok = probes_store.delete_probe(probe_id)
    if not ok:
        return jsonify({'error': 'Probe not found'}), 404
    return jsonify({'success': True})


@app.route('/probes/run', methods=['POST'])
def probes_run():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    probe_id = data.get('id')
    if not probe_id:
        return jsonify({'error': 'Provide probe id'}), 400
    probes = {p.get('id'): p for p in probes_store.list_probes()}
    probe = probes.get(probe_id)
    if not probe:
        return jsonify({'error': 'Probe not found'}), 404
    probe_roi_enabled, probe_roi_norm = _parse_probe_roi(probe)
    result = probe_manager.query(
        probe.get('channel_id', config.LUXRIOT_DEFAULT_CHANNEL_ID),
        probe.get('positives', []),
        probe.get('negatives', []),
        probe.get('pos_floor', 0.2),
        probe.get('margin', 0.05),
        probe.get('top_k', 6),
        window_sec=probe.get('window_sec', 300.0),
        image_probe=probe.get('image_probe'),
        roi_norm=probe_roi_norm if probe_roi_enabled else None,
        roi_padding=PROBE_ROI_PADDING,
    )
    if 'error' in result:
        return jsonify(result), 400
    hits = result.get('results') or []
    bookmark_sent = False
    bookmark_gate: Dict[str, Any] = {"reason": "bookmark_disabled", "source": "probes_run"}
    if hits:
        probe['last_hit'] = hits[0]
        # keep a short rolling history of hits for UI while capping thumbnails
        recent = probe.get('recent_hits') or []
        recent = (hits + recent)[:PROBE_MAX_STORED_HITS]
        probe['recent_hits'] = recent
        if probe.get('bookmark'):
            bookmark_hit = _select_probe_bookmark_hit(cast(Sequence[Mapping[str, Any]], hits))
            if bookmark_hit is not None:
                bookmark_sent, bookmark_gate = _maybe_send_probe_bookmark(
                    probe,
                    bookmark_hit,
                    source='probes_run',
                )
        probe['bookmark_gate'] = bookmark_gate
        probe['bookmark_gate_updated_at_ms'] = int(time.time() * 1000)
        probes_store.upsert_probe(probe)
        persisted_hits = _store_probe_hits(
            probe,
            hits,
            source='probes_run',
            bookmark_sent=bookmark_sent,
            extra_payload={
                'frames_indexed': result.get('frames_indexed'),
                'roi_enabled': probe_roi_enabled,
                'roi_norm': _probe_roi_norm_to_payload(probe_roi_norm),
                'bookmark_gate': bookmark_gate,
            },
        )
    else:
        persisted_hits = 0
    return jsonify(
        {
            'results': hits,
            'status': result.get('status'),
            'probe': probe,
            'persisted_hits': persisted_hits,
            'bookmark_gate': bookmark_gate,
        }
    )


@app.route('/probes/bench', methods=['GET'])
def probes_bench():
    """Lightweight throughput estimate (image embedding)."""
    try:
        import torch  # type: ignore
        init_clip()
    except Exception:
        return jsonify({
            "error": "PyTorch/CLIP not available; install torch+clip to run benchmark."
        }), 400
    batch = int(request.args.get('batch', PROBE_BENCH_BATCH))
    batch = max(4, min(64, batch))
    try:
        # Use a repeated random image batch so benchmark works across CLIP and SigLIP2 backends.
        target_size = 224
        if clip_backend_kind == "siglip2" and clip_processor is not None:
            size_info = getattr(getattr(clip_processor, "image_processor", None), "size", None)
            if isinstance(size_info, dict):
                h = _to_optional_int(size_info.get("height")) or _to_optional_int(size_info.get("shortest_edge"))
                w = _to_optional_int(size_info.get("width")) or h
                if h is not None and w is not None:
                    target_size = max(128, min(512, max(h, w)))
        rnd = np.random.randint(0, 256, (target_size, target_size, 3), dtype=np.uint8)
        probe_image = Image.fromarray(rnd, mode='RGB')
        images = [probe_image] * batch
        started = time.time()
        feats = _clip_image_embeddings_from_pils(images)
        _ = feats.shape[0]
        elapsed = time.time() - started
        fps = batch / elapsed if elapsed > 0 else 0
        return jsonify({
            "batch": batch,
            "elapsed_sec": round(elapsed, 3),
            "approx_fps": round(fps, 1),
            "device": clip_runtime_device,
            "backend": clip_backend_kind,
            "model": clip_runtime_model or config.CLIP_MODEL,
            "resolution": target_size,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/detections/search_text', methods=['POST'])
def detections_search_text():
    data = _json_body()
    query = str(data.get('query') or '').strip()
    if not query:
        return jsonify({'error': 'query is required'}), 400

    try:
        filters = _parse_detection_filters(data)
    except Exception as exc:
        return jsonify({'error': f'Invalid detection filters: {exc}'}), 400

    try:
        limit = int(data.get('limit', config.DEFAULT_RESULTS))
    except Exception:
        limit = config.DEFAULT_RESULTS
    if limit < config.MIN_RESULTS or limit > config.MAX_RESULTS:
        limit = config.DEFAULT_RESULTS

    try:
        candidate_limit = int(data.get('candidate_limit', 20000))
    except Exception:
        candidate_limit = 20000

    sort_by = str(data.get('sort_by') or 'similarity').strip().lower()
    if sort_by not in {'similarity', 'time'}:
        sort_by = 'similarity'

    mode_requested = _normalize_detection_search_mode(str(data.get('embedder') or active_embedder))
    # DINO/Fusion cannot encode text prompts; fallback to CLIP retrieval.
    mode = 'clip'

    try:
        clip_query_vec = get_text_embedding(query)
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Failed to embed text query: {exc}'}), 500

    try:
        results = _search_detections_archive(
            clip_query_vec=clip_query_vec,
            dino_query_vec=None,
            mode=mode,
            probe_id=filters['probe_id'],
            channel_id=filters['channel_id'],
            since_ms=filters['since_ms'],
            until_ms=filters['until_ms'],
            limit=limit,
            sort_by=sort_by,
            candidate_limit=candidate_limit,
        )
        return jsonify(
            {
                'results': results,
                'mode_requested': mode_requested,
                'mode_used': mode,
                'filters': filters,
                'query': query,
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/detections/search_image', methods=['POST'])
def detections_search_image():
    mode = _normalize_detection_search_mode(request.form.get('embedder') or active_embedder)

    filters_payload = {
        'probe_id': request.form.get('probe_id'),
        'channel_id': request.form.get('channel_id'),
        'since_ms': request.form.get('since_ms'),
        'until_ms': request.form.get('until_ms'),
        'hours': request.form.get('hours'),
    }
    try:
        filters = _parse_detection_filters(filters_payload)
    except Exception as exc:
        return jsonify({'error': f'Invalid detection filters: {exc}'}), 400

    try:
        limit = int(request.form.get('limit', config.DEFAULT_RESULTS))
    except Exception:
        limit = config.DEFAULT_RESULTS
    if limit < config.MIN_RESULTS or limit > config.MAX_RESULTS:
        limit = config.DEFAULT_RESULTS

    try:
        candidate_limit = int(request.form.get('candidate_limit', 20000))
    except Exception:
        candidate_limit = 20000

    sort_by = str(request.form.get('sort_by') or 'similarity').strip().lower()
    if sort_by not in {'similarity', 'time'}:
        sort_by = 'similarity'

    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify({'error': 'image file is required'}), 400

    try:
        pil_image = Image.open(file.stream)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
    except Exception as exc:
        return jsonify({'error': f'Failed to read uploaded image: {exc}'}), 400

    try:
        clip_query_vec = get_image_embedding_from_pil(pil_image, embedder='clip')
    except Exception as exc:
        return jsonify({'error': f'Failed to embed image query with CLIP: {exc}'}), 500

    dino_query_vec: Optional[np.ndarray] = None
    if mode in {'dino', 'fusion'}:
        try:
            dino_query_vec = get_image_embedding_from_pil(pil_image, embedder='dino')
        except Exception as exc:
            print(f"Detections image search: DINO query embedding unavailable, fallback to CLIP only ({exc})")
            dino_query_vec = None

    try:
        results = _search_detections_archive(
            clip_query_vec=clip_query_vec,
            dino_query_vec=dino_query_vec,
            mode=mode,
            probe_id=filters['probe_id'],
            channel_id=filters['channel_id'],
            since_ms=filters['since_ms'],
            until_ms=filters['until_ms'],
            limit=limit,
            sort_by=sort_by,
            candidate_limit=candidate_limit,
        )
        return jsonify(
            {
                'results': results,
                'mode_used': mode,
                'filters': filters,
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/detections/list', methods=['GET'])
def detections_list():
    probe_id_raw = (request.args.get('probe_id') or '').strip()
    probe_id = probe_id_raw or None

    channel_id_raw = (request.args.get('channel_id') or '').strip()
    channel_id: Optional[int] = None
    if channel_id_raw:
        try:
            channel_id = int(channel_id_raw)
        except Exception:
            return jsonify({'error': 'channel_id must be an integer'}), 400

    since_ms_raw = (request.args.get('since_ms') or '').strip()
    since_ms: Optional[int] = None
    if since_ms_raw:
        try:
            since_ms = int(since_ms_raw)
        except Exception:
            return jsonify({'error': 'since_ms must be an integer'}), 400

    until_ms_raw = (request.args.get('until_ms') or '').strip()
    until_ms: Optional[int] = None
    if until_ms_raw:
        try:
            until_ms = int(until_ms_raw)
        except Exception:
            return jsonify({'error': 'until_ms must be an integer'}), 400

    hours_raw = (request.args.get('hours') or '').strip()
    if since_ms is None:
        try:
            hours = float(hours_raw) if hours_raw else 24.0
        except Exception:
            return jsonify({'error': 'hours must be numeric'}), 400
        if hours > 0:
            since_ms = int(time.time() * 1000 - (hours * 3600 * 1000))

    try:
        limit = int(request.args.get('limit', 50))
    except Exception:
        limit = 50
    try:
        offset = int(request.args.get('offset', 0))
    except Exception:
        offset = 0

    try:
        detections, total = detections_store.list_detections(
            probe_id=probe_id,
            channel_id=channel_id,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=offset,
        )
        return jsonify(
            {
                'detections': detections,
                'total': total,
                'limit': max(1, min(500, int(limit or 50))),
                'offset': max(0, int(offset or 0)),
                'has_more': max(0, int(offset or 0)) + len(detections) < total,
                'filters': {
                    'probe_id': probe_id,
                    'channel_id': channel_id,
                    'since_ms': since_ms,
                    'until_ms': until_ms,
                },
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/detections/summary', methods=['GET'])
def detections_summary():
    channel_id_raw = (request.args.get('channel_id') or '').strip()
    channel_id: Optional[int] = None
    if channel_id_raw:
        try:
            channel_id = int(channel_id_raw)
        except Exception:
            return jsonify({'error': 'channel_id must be an integer'}), 400

    since_ms_raw = (request.args.get('since_ms') or '').strip()
    since_ms: Optional[int] = None
    if since_ms_raw:
        try:
            since_ms = int(since_ms_raw)
        except Exception:
            return jsonify({'error': 'since_ms must be an integer'}), 400
    else:
        hours_raw = (request.args.get('hours') or '').strip()
        try:
            hours = float(hours_raw) if hours_raw else 24.0
        except Exception:
            return jsonify({'error': 'hours must be numeric'}), 400
        if hours > 0:
            since_ms = int(time.time() * 1000 - (hours * 3600 * 1000))

    try:
        limit = int(request.args.get('limit', 100))
    except Exception:
        limit = 100

    try:
        summary = detections_store.summarize_by_probe(since_ms=since_ms, channel_id=channel_id, limit=limit)
        return jsonify(
            {
                'summary': summary,
                'count': len(summary),
                'filters': {
                    'channel_id': channel_id,
                    'since_ms': since_ms,
                },
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


ENV_PREFIX = "EVOSSEARCH_"


def _bool_to_env(value: Any) -> str:
    return "true" if bool(value) else "false"


def _read_env_file_map(path: Union[str, Path] = ".env") -> Dict[str, str]:
    env_map: Dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return env_map
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return env_map
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key_raw, value_raw = raw_line.split("=", 1)
        key = key_raw.strip()
        if not key:
            continue
        env_map[key] = value_raw.strip()
    return env_map


def _parse_env_editor_text(raw_text: Any) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    text = str(raw_text or "")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key_raw, value_raw = raw_line.split("=", 1)
        key = key_raw.strip()
        if not key or not key.startswith(ENV_PREFIX):
            continue
        parsed[key] = value_raw.strip()
    return parsed


def _serialize_env_map(env_map: Dict[str, str]) -> str:
    keys_sorted = sorted(env_map.keys())
    return "\n".join(f"{key}={env_map[key]}" for key in keys_sorted)


ENV_SECRET_REDACTION = "__EVOSSEARCH_SECRET_SET__"
ENV_SECRET_KEY_PARTS = ("PASSWORD", "TOKEN", "SECRET", "API_KEY", "PRIVATE_KEY")


def _is_secret_env_key(key: str) -> bool:
    normalized = str(key or "").strip().upper()
    return any(part in normalized for part in ENV_SECRET_KEY_PARTS)


def _redact_env_map(env_map: Mapping[str, str]) -> Dict[str, str]:
    return {
        str(key): (
            ENV_SECRET_REDACTION
            if _is_secret_env_key(str(key)) and bool(str(value or ""))
            else str(value or "")
        )
        for key, value in env_map.items()
    }


def _restore_redacted_env_secrets(
    target_env: Mapping[str, str],
    current_env: Mapping[str, str],
) -> Dict[str, str]:
    restored = {str(key): str(value) for key, value in target_env.items()}
    for key, value in list(restored.items()):
        if not _is_secret_env_key(key) or value != ENV_SECRET_REDACTION:
            continue
        restored[key] = str(current_env.get(key) or "")
    return restored


def _runtime_env_map() -> Dict[str, str]:
    sev = config.LUXRIOT_SEVERITY_MAP or {}
    env: Dict[str, str] = {
        "EVOSSEARCH_HOST": str(config.HOST),
        "EVOSSEARCH_PORT": str(config.PORT),
        "EVOSSEARCH_DEBUG": _bool_to_env(config.DEBUG),
        "EVOSSEARCH_EMBEDDER": str(config.EMBEDDER),
        "EVOSSEARCH_CLIP_MODEL": str(config.CLIP_MODEL),
        "EVOSSEARCH_DINO_MODEL": str(config.DINO_MODEL),
        "EVOSSEARCH_EMB_DIM_DINO": str(config.EMB_DIM_DINO),
        "EVOSSEARCH_DINO_WEIGHTS_PATH": str(config.DINO_WEIGHTS_PATH),
        "EVOSSEARCH_DINO_DEVICE": str(config.DINO_DEVICE),
        "EVOSSEARCH_INDEX_MODE": str(config.INDEX_MODE),
        "EVOSSEARCH_FUSION_ENABLED": _bool_to_env(config.FUSION_ENABLED),
        "EVOSSEARCH_FUSION_ALPHA": str(config.FUSION_ALPHA),
        "EVOSSEARCH_RERANK_ENABLED": _bool_to_env(config.RERANK_ENABLED),
        "EVOSSEARCH_RERANK_TOP_K": str(config.RERANK_TOP_K),
        "EVOSSEARCH_DINO_SEGMENTS_ENABLED": _bool_to_env(config.DINO_SEGMENTS_ENABLED),
        "EVOSSEARCH_DINO_SEGMENT_MIN_PATCHES": str(config.DINO_SEGMENT_MIN_PATCHES),
        "EVOSSEARCH_DINO_HEATMAP_THRESHOLD": str(config.DINO_HEATMAP_THRESHOLD),
        "EVOSSEARCH_M2F_ENABLED": _bool_to_env(config.MASK2FORMER_ENABLED),
        "EVOSSEARCH_M2F_MODEL": str(config.MASK2FORMER_MODEL),
        "EVOSSEARCH_M2F_DEVICE": str(config.MASK2FORMER_DEVICE),
        "EVOSSEARCH_M2F_MAX_SIZE": str(config.MASK2FORMER_MAX_SIZE),
        "EVOSSEARCH_LM_BASE_URL": str(config.LM_BASE_URL),
        "EVOSSEARCH_LM_MODEL": str(config.LM_MODEL),
        "EVOSSEARCH_LM_API_KEY": str(config.LM_API_KEY),
        "EVOSSEARCH_LM_TIMEOUT": str(config.LM_TIMEOUT),
        "EVOSSEARCH_LM_VIDEO_DEFAULT_FRAMES": str(config.LM_VIDEO_DEFAULT_FRAMES),
        "EVOSSEARCH_LM_VIDEO_MAX_FRAMES": str(config.LM_VIDEO_MAX_FRAMES),
        "EVOSSEARCH_LM_VIDEO_MAX_EDGE": str(config.LM_VIDEO_MAX_EDGE),
        "EVOSSEARCH_LM_VIDEO_MAX_TOKENS": str(config.LM_VIDEO_MAX_TOKENS),
        "EVOSSEARCH_LM_VIDEO_TEMPERATURE": str(config.LM_VIDEO_TEMPERATURE),
        "EVOSSEARCH_LUXRIOT_BASE_URL": str(config.LUXRIOT_BASE_URL),
        "EVOSSEARCH_LUXRIOT_USERNAME": str(config.LUXRIOT_USERNAME),
        "EVOSSEARCH_LUXRIOT_PASSWORD": str(config.LUXRIOT_PASSWORD),
        "EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID": str(config.LUXRIOT_DEFAULT_CHANNEL_ID),
        "EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL": str(config.LUXRIOT_SNAPSHOT_INTERVAL),
        "EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE": str(config.LUXRIOT_SNAPSHOT_MAX_EDGE),
        "EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES": str(config.LUXRIOT_MAX_BUFFER_FRAMES),
        "EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS": _bool_to_env(config.LUXRIOT_AUTO_BOOKMARKS),
        "EVOSSEARCH_LUXRIOT_SEV_INFO": str(sev.get("info", "info")),
        "EVOSSEARCH_LUXRIOT_SEV_LOW": str(sev.get("low", "low")),
        "EVOSSEARCH_LUXRIOT_SEV_NORMAL": str(sev.get("normal", "normal")),
        "EVOSSEARCH_LUXRIOT_SEV_HIGH": str(sev.get("high", "high")),
        "EVOSSEARCH_LUXRIOT_SEV_CRITICAL": str(sev.get("critical", "critical")),
        "EVOSSEARCH_PROBE_MAX_FRAMES": str(config.PROBE_MAX_FRAMES),
        "EVOSSEARCH_PROBE_THUMB_MAX_EDGE": str(config.PROBE_THUMB_MAX_EDGE),
        "EVOSSEARCH_PROBE_BOOKMARK_COOLDOWN_SEC": str(config.PROBE_BOOKMARK_COOLDOWN_SEC),
        "EVOSSEARCH_PROBE_BOOKMARK_DEDUPE_WINDOW_SEC": str(config.PROBE_BOOKMARK_DEDUPE_WINDOW_SEC),
        "EVOSSEARCH_PROBE_BOOKMARK_SIM_HIGH": str(config.PROBE_BOOKMARK_SIM_HIGH),
        "EVOSSEARCH_PROBE_BOOKMARK_MARGIN_DELTA": str(config.PROBE_BOOKMARK_MARGIN_DELTA),
        "EVOSSEARCH_PROBE_BOOKMARK_SCORE_DELTA": str(config.PROBE_BOOKMARK_SCORE_DELTA),
        "EVOSSEARCH_PROBE_BOOKMARK_MAX_FRAME_GAP": str(config.PROBE_BOOKMARK_MAX_FRAME_GAP),
        "EVOSSEARCH_DETECTIONS_ARCHIVE_ENABLED": _bool_to_env(config.DETECTIONS_ARCHIVE_ENABLED),
        "EVOSSEARCH_DETECTIONS_ARCHIVE_DIR": str(config.DETECTIONS_ARCHIVE_DIR),
        "EVOSSEARCH_DETECTIONS_ARCHIVE_JPEG_QUALITY": str(config.DETECTIONS_ARCHIVE_JPEG_QUALITY),
        "EVOSSEARCH_DETECTIONS_RETENTION_ENABLED": _bool_to_env(config.DETECTIONS_RETENTION_ENABLED),
        "EVOSSEARCH_DETECTIONS_RETENTION_DROP_SKIPPED": _bool_to_env(config.DETECTIONS_RETENTION_DROP_SKIPPED),
        "EVOSSEARCH_DETECTIONS_RETENTION_WINDOW_SEC": str(config.DETECTIONS_RETENTION_WINDOW_SEC),
        "EVOSSEARCH_DETECTIONS_RETENTION_FORCE_KEEP_SEC": str(config.DETECTIONS_RETENTION_FORCE_KEEP_SEC),
        "EVOSSEARCH_DETECTIONS_RETENTION_SIMILARITY_HIGH": str(config.DETECTIONS_RETENTION_SIMILARITY_HIGH),
        "EVOSSEARCH_DETECTIONS_RETENTION_SIMILARITY_LOW": str(config.DETECTIONS_RETENTION_SIMILARITY_LOW),
        "EVOSSEARCH_DETECTIONS_RETENTION_MARGIN_DELTA": str(config.DETECTIONS_RETENTION_MARGIN_DELTA),
        "EVOSSEARCH_DETECTIONS_RETENTION_SCORE_DELTA": str(config.DETECTIONS_RETENTION_SCORE_DELTA),
        "EVOSSEARCH_MIN_RESULTS": str(config.MIN_RESULTS),
        "EVOSSEARCH_MAX_RESULTS": str(config.MAX_RESULTS),
        "EVOSSEARCH_DEFAULT_RESULTS": str(config.DEFAULT_RESULTS),
        "EVOSSEARCH_BATCH_SIZE": str(config.BATCH_SIZE),
        "EVOSSEARCH_THUMBNAIL_QUALITY": str(config.THUMBNAIL_QUALITY),
        "EVOSSEARCH_INDEX_FOLDER": str(config.INDEX_FOLDER_NAME),
        "EVOSSEARCH_MAX_COMMENT_LENGTH": str(config.MAX_COMMENT_LENGTH),
        "EVOSSEARCH_MAX_FILE_SIZE_MB": str(config.MAX_FILE_SIZE_MB),
        "EVOSSEARCH_ADMIN_TOKEN": str(config.ADMIN_TOKEN),
        "EVOSSEARCH_SETTINGS_LOCAL_ONLY": _bool_to_env(config.SETTINGS_LOCAL_ONLY),
        "EVOSSEARCH_CORS_ALLOWED_ORIGINS": ",".join(config.CORS_ALLOWED_ORIGINS),
        "EVOSSEARCH_ALLOWED_ROOTS": os.pathsep.join(config.ALLOWED_ROOTS),
    }
    return env


def _effective_env_map() -> Dict[str, str]:
    runtime_map = _runtime_env_map()
    file_map = _read_env_file_map(".env")
    merged = dict(runtime_map)
    for key, value in file_map.items():
        if key.startswith(ENV_PREFIX) and key not in merged:
            merged[key] = value
    return merged


def _preserve_additional_env_lines(known_keys: Set[str]) -> str:
    existing_map = _read_env_file_map(".env")
    extra_evos = [
        f"{key}={value}"
        for key, value in sorted(existing_map.items())
        if key.startswith(ENV_PREFIX) and key not in known_keys
    ]
    extra_other = [
        f"{key}={value}"
        for key, value in sorted(existing_map.items())
        if not key.startswith(ENV_PREFIX)
    ]
    chunks: List[str] = []
    if extra_evos:
        chunks.append("# Additional EVOSSEARCH variables")
        chunks.extend(extra_evos)
    if extra_other:
        if chunks:
            chunks.append("")
        chunks.append("# Preserved external variables")
        chunks.extend(extra_other)
    return ("\n" + "\n".join(chunks) + "\n") if chunks else ""


@app.route('/settings/env', methods=['GET'])
def get_settings_env():
    guard = _settings_guard(write=False)
    if guard is not None:
        return guard
    try:
        env_map = _redact_env_map(_effective_env_map())
        return jsonify(
            {
                'success': True,
                'envVariables': env_map,
                'envText': _serialize_env_map(env_map),
                'count': len(env_map),
            }
        )
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/settings/env', methods=['POST'])
def save_settings_env():
    guard = _settings_guard(write=True)
    if guard is not None:
        return guard
    data = _json_body()
    try:
        parsed_from_text = _parse_env_editor_text(data.get('envText', ''))
        if parsed_from_text:
            target_env = parsed_from_text
        else:
            payload_obj = data.get('envVariables')
            if not isinstance(payload_obj, dict):
                return jsonify({'success': False, 'error': 'Provide envText or envVariables'}), 400
            target_env = {
                str(k).strip(): str(v).strip()
                for k, v in payload_obj.items()
                if str(k).strip().startswith(ENV_PREFIX)
            }
        if not target_env:
            return jsonify({'success': False, 'error': 'No EVOSSEARCH_* entries to save'}), 400

        target_env = _restore_redacted_env_secrets(target_env, _effective_env_map())
        existing_map = _read_env_file_map(".env")
        preserved_other = {
            key: value
            for key, value in existing_map.items()
            if not key.startswith(ENV_PREFIX)
        }
        merged_map = dict(preserved_other)
        merged_map.update(target_env)

        env_lines = [f"{key}={merged_map[key]}" for key in sorted(merged_map.keys())]
        header = "# evo-ssearch Configuration\n# Managed by settings env editor\n\n"
        Path(".env").write_text(header + "\n".join(env_lines) + "\n", encoding="utf-8")

        return jsonify(
            {
                'success': True,
                'message': 'Environment variables saved to .env. Restart the server to apply all changes.',
                'count': len(target_env),
            }
        )
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current configuration settings"""
    guard = _settings_guard(write=False)
    if guard is not None:
        return guard
    try:
        requested_embedder = config.EMBEDDER if config.EMBEDDER in SUPPORTED_EMBEDDERS else active_embedder
        settings = {
            'host': config.HOST,
            'port': config.PORT,
            'debug': config.DEBUG,
            'appVersion': config.APP_VERSION,
            'embedder': requested_embedder,
            'clipModel': config.CLIP_MODEL,
            'dinoModel': config.DINO_MODEL,
            'dinoEmbedDim': config.EMB_DIM_DINO,
            'dinoWeightsPath': config.DINO_WEIGHTS_PATH,
            'indexMode': config.INDEX_MODE,
            'fusionEnabled': config.FUSION_ENABLED,
            'fusionAlpha': config.FUSION_ALPHA,
            'rerankEnabled': config.RERANK_ENABLED,
            'rerankTopK': config.RERANK_TOP_K,
            'segmentsEnabled': config.DINO_SEGMENTS_ENABLED,
            'segmentMinPatches': config.DINO_SEGMENT_MIN_PATCHES,
            'segmentThreshold': config.DINO_HEATMAP_THRESHOLD,
            'luxriotBaseUrl': config.LUXRIOT_BASE_URL,
            'luxriotUsername': config.LUXRIOT_USERNAME,
            'luxriotPassword': '',
            'luxriotPasswordSet': bool(config.LUXRIOT_PASSWORD),
            'luxriotSnapshotInterval': config.LUXRIOT_SNAPSHOT_INTERVAL,
            'luxriotSnapshotMaxEdge': config.LUXRIOT_SNAPSHOT_MAX_EDGE,
            'luxriotDefaultChannelId': config.LUXRIOT_DEFAULT_CHANNEL_ID,
            'luxriotMaxBufferFrames': config.LUXRIOT_MAX_BUFFER_FRAMES,
            'luxriotAutoBookmarks': config.LUXRIOT_AUTO_BOOKMARKS,
            'luxriotSeverityMap': config.LUXRIOT_SEVERITY_MAP,
            'probeBookmarkCooldownSec': config.PROBE_BOOKMARK_COOLDOWN_SEC,
            'probeBookmarkDedupeWindowSec': config.PROBE_BOOKMARK_DEDUPE_WINDOW_SEC,
            'probeBookmarkSimHigh': config.PROBE_BOOKMARK_SIM_HIGH,
            'probeBookmarkMarginDelta': config.PROBE_BOOKMARK_MARGIN_DELTA,
            'probeBookmarkScoreDelta': config.PROBE_BOOKMARK_SCORE_DELTA,
            'probeBookmarkMaxFrameGap': config.PROBE_BOOKMARK_MAX_FRAME_GAP,
            'luxriotBatchSizes': list(config.LUXRIOT_BATCH_SIZES),
            'minResults': config.MIN_RESULTS,
            'maxResults': config.MAX_RESULTS,
            'defaultResults': config.DEFAULT_RESULTS,
            'batchSize': config.BATCH_SIZE,
            'thumbnailQuality': config.THUMBNAIL_QUALITY,
            'maxCommentLength': config.MAX_COMMENT_LENGTH,
            'maxFileSize': config.MAX_FILE_SIZE_MB,
            'indexFolderName': config.INDEX_FOLDER_NAME,
            'settingsLocalOnly': config.SETTINGS_LOCAL_ONLY,
            'adminTokenSet': bool(config.ADMIN_TOKEN),
            'corsAllowedOrigins': list(config.CORS_ALLOWED_ORIGINS),
            'allowedRoots': list(config.ALLOWED_ROOTS),
            'envCount': len(_effective_env_map()),
        }
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/settings', methods=['POST'])
def save_settings():
    """Save configuration settings to .env file"""
    guard = _settings_guard(write=True)
    if guard is not None:
        return guard
    try:
        data = _json_body()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        global active_embedder, clip_model, clip_preprocess, clip_processor, clip_backend_kind, clip_runtime_model, dino_encoder, probe_bookmark_gate

        required_fields = ['host', 'port', 'debug', 'clipModel', 'minResults', 'maxResults', 'defaultResults']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        debug_enabled = _coerce_bool(data.get('debug', config.DEBUG), config.DEBUG)

        try:
            port = int(data['port'])
            if not (1000 <= port <= 65535):
                return jsonify({'success': False, 'error': 'Port must be between 1000 and 65535'}), 400

            min_results = int(data['minResults'])
            max_results = int(data['maxResults'])
            default_results = int(data['defaultResults'])

            if not (1 <= min_results <= max_results):
                return jsonify({'success': False, 'error': 'Min results must be less than or equal to max results'}), 400

            if not (min_results <= default_results <= max_results):
                return jsonify({'success': False, 'error': 'Default results must be between min and max results'}), 400
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid number format: {str(e)}'}), 400

        fusion_enabled = _coerce_bool(data.get('fusionEnabled', config.FUSION_ENABLED), config.FUSION_ENABLED)

        try:
            fusion_alpha = float(data.get('fusionAlpha', config.FUSION_ALPHA))
        except (TypeError, ValueError):
            fusion_alpha = config.FUSION_ALPHA
        fusion_alpha = min(1.0, max(0.0, fusion_alpha))

        rerank_enabled = _coerce_bool(data.get('rerankEnabled', config.RERANK_ENABLED), config.RERANK_ENABLED)

        try:
            rerank_top_k = int(data.get('rerankTopK', config.RERANK_TOP_K))
        except (TypeError, ValueError):
            rerank_top_k = config.RERANK_TOP_K
        if rerank_top_k < 1:
            rerank_top_k = 1

        segments_enabled = _coerce_bool(data.get('segmentsEnabled', config.DINO_SEGMENTS_ENABLED), config.DINO_SEGMENTS_ENABLED)

        try:
            segment_min_patches = int(data.get('segmentMinPatches', config.DINO_SEGMENT_MIN_PATCHES))
        except (TypeError, ValueError):
            segment_min_patches = config.DINO_SEGMENT_MIN_PATCHES
        if segment_min_patches < 1:
            segment_min_patches = 1

        try:
            segment_threshold = float(data.get('segmentThreshold', config.DINO_HEATMAP_THRESHOLD))
        except (TypeError, ValueError):
            segment_threshold = config.DINO_HEATMAP_THRESHOLD
        segment_threshold = min(0.99, max(0.0, segment_threshold))

        luxriot_base_url = str(data.get('luxriotBaseUrl', config.LUXRIOT_BASE_URL)).strip().rstrip('/')
        luxriot_username = str(data.get('luxriotUsername', config.LUXRIOT_USERNAME)).strip()
        luxriot_password_raw = data.get('luxriotPassword', None)
        if luxriot_password_raw is None:
            luxriot_password = config.LUXRIOT_PASSWORD
        else:
            luxriot_password = str(luxriot_password_raw).strip() or config.LUXRIOT_PASSWORD
        try:
            luxriot_snapshot_interval = int(data.get('luxriotSnapshotInterval', config.LUXRIOT_SNAPSHOT_INTERVAL))
        except (TypeError, ValueError):
            luxriot_snapshot_interval = config.LUXRIOT_SNAPSHOT_INTERVAL
        if luxriot_snapshot_interval < 1:
            luxriot_snapshot_interval = 5
        try:
            luxriot_snapshot_max_edge = int(data.get('luxriotSnapshotMaxEdge', config.LUXRIOT_SNAPSHOT_MAX_EDGE))
        except (TypeError, ValueError):
            luxriot_snapshot_max_edge = config.LUXRIOT_SNAPSHOT_MAX_EDGE
        if luxriot_snapshot_max_edge < 640:
            luxriot_snapshot_max_edge = 640
        try:
            luxriot_default_channel_id = int(data.get('luxriotDefaultChannelId', config.LUXRIOT_DEFAULT_CHANNEL_ID))
        except (TypeError, ValueError):
            luxriot_default_channel_id = config.LUXRIOT_DEFAULT_CHANNEL_ID
        try:
            luxriot_max_buffer_frames = int(data.get('luxriotMaxBufferFrames', config.LUXRIOT_MAX_BUFFER_FRAMES))
        except (TypeError, ValueError):
            luxriot_max_buffer_frames = config.LUXRIOT_MAX_BUFFER_FRAMES
        if luxriot_max_buffer_frames < 12:
            luxriot_max_buffer_frames = 12
        luxriot_auto_bookmarks = _coerce_bool(
            data.get('luxriotAutoBookmarks', config.LUXRIOT_AUTO_BOOKMARKS),
            config.LUXRIOT_AUTO_BOOKMARKS,
        )
        try:
            probe_bookmark_cooldown_sec = float(data.get('probeBookmarkCooldownSec', config.PROBE_BOOKMARK_COOLDOWN_SEC))
        except (TypeError, ValueError):
            probe_bookmark_cooldown_sec = config.PROBE_BOOKMARK_COOLDOWN_SEC
        probe_bookmark_cooldown_sec = max(0.0, probe_bookmark_cooldown_sec)
        try:
            probe_bookmark_dedupe_window_sec = float(
                data.get('probeBookmarkDedupeWindowSec', config.PROBE_BOOKMARK_DEDUPE_WINDOW_SEC)
            )
        except (TypeError, ValueError):
            probe_bookmark_dedupe_window_sec = config.PROBE_BOOKMARK_DEDUPE_WINDOW_SEC
        probe_bookmark_dedupe_window_sec = max(0.5, probe_bookmark_dedupe_window_sec)
        try:
            probe_bookmark_sim_high = float(data.get('probeBookmarkSimHigh', config.PROBE_BOOKMARK_SIM_HIGH))
        except (TypeError, ValueError):
            probe_bookmark_sim_high = config.PROBE_BOOKMARK_SIM_HIGH
        probe_bookmark_sim_high = min(0.9999, max(0.5, probe_bookmark_sim_high))
        try:
            probe_bookmark_margin_delta = float(data.get('probeBookmarkMarginDelta', config.PROBE_BOOKMARK_MARGIN_DELTA))
        except (TypeError, ValueError):
            probe_bookmark_margin_delta = config.PROBE_BOOKMARK_MARGIN_DELTA
        probe_bookmark_margin_delta = max(0.0, probe_bookmark_margin_delta)
        try:
            probe_bookmark_score_delta = float(data.get('probeBookmarkScoreDelta', config.PROBE_BOOKMARK_SCORE_DELTA))
        except (TypeError, ValueError):
            probe_bookmark_score_delta = config.PROBE_BOOKMARK_SCORE_DELTA
        probe_bookmark_score_delta = max(0.0, probe_bookmark_score_delta)
        try:
            probe_bookmark_max_frame_gap = int(data.get('probeBookmarkMaxFrameGap', config.PROBE_BOOKMARK_MAX_FRAME_GAP))
        except (TypeError, ValueError):
            probe_bookmark_max_frame_gap = config.PROBE_BOOKMARK_MAX_FRAME_GAP
        if probe_bookmark_max_frame_gap < 1:
            probe_bookmark_max_frame_gap = 1
        severity_map = data.get('luxriotSeverityMap', {}) or {}
        merged_sev = dict(config.LUXRIOT_SEVERITY_MAP)
        for key in ['info', 'low', 'normal', 'high', 'critical']:
            if key in severity_map:
                merged_sev[key] = str(severity_map[key] or merged_sev.get(key, key)).lower()

        embedder = str(data.get('embedder', active_embedder)).strip().lower()
        if embedder == 'fusion' and not fusion_enabled:
            embedder = 'clip'
        if embedder not in SUPPORTED_EMBEDDERS:
            embedder = 'clip'
        dino_model = str(data.get('dinoModel', config.DINO_MODEL)).strip() or config.DINO_MODEL
        try:
            dino_dim = int(data.get('dinoEmbedDim', config.EMB_DIM_DINO))
        except (TypeError, ValueError):
            dino_dim = config.EMB_DIM_DINO

        dino_weights_path = data.get('dinoWeightsPath', config.DINO_WEIGHTS_PATH) or ''
        dino_device = str(data.get('dinoDevice', config.DINO_DEVICE)).strip()

        index_mode = str(data.get('indexMode', config.INDEX_MODE)).strip().lower()
        if index_mode not in {'clip', 'dino', 'dual'}:
            index_mode = 'clip'

        batch_size = int(data.get('batchSize', config.BATCH_SIZE))
        thumbnail_quality = int(data.get('thumbnailQuality', config.THUMBNAIL_QUALITY))
        max_comment_length = int(data.get('maxCommentLength', config.MAX_COMMENT_LENGTH))
        max_file_size = int(data.get('maxFileSize', config.MAX_FILE_SIZE_MB))
        index_folder = data.get('indexFolderName', config.INDEX_FOLDER_NAME)

        env_content = f"""# evo-ssearch Configuration
# Generated by settings panel

# Server Configuration
EVOSSEARCH_HOST={data['host']}
EVOSSEARCH_PORT={port}
EVOSSEARCH_DEBUG={str(debug_enabled).lower()}
EVOSSEARCH_APP_VERSION="{config.APP_VERSION}"

# Embedder configuration
EVOSSEARCH_EMBEDDER={embedder}
EVOSSEARCH_CLIP_MODEL={data['clipModel']}
EVOSSEARCH_DINO_MODEL={dino_model}
EVOSSEARCH_EMB_DIM_DINO={dino_dim}
EVOSSEARCH_DINO_WEIGHTS_PATH={dino_weights_path}
EVOSSEARCH_DINO_DEVICE={dino_device}
EVOSSEARCH_INDEX_MODE={index_mode}
EVOSSEARCH_FUSION_ENABLED={str(fusion_enabled).lower()}
EVOSSEARCH_FUSION_ALPHA={fusion_alpha:.4f}
EVOSSEARCH_RERANK_ENABLED={str(rerank_enabled).lower()}
EVOSSEARCH_RERANK_TOP_K={rerank_top_k}
EVOSSEARCH_DINO_SEGMENTS_ENABLED={str(segments_enabled).lower()}
EVOSSEARCH_DINO_SEGMENT_MIN_PATCHES={segment_min_patches}
EVOSSEARCH_DINO_HEATMAP_THRESHOLD={segment_threshold:.4f}
EVOSSEARCH_M2F_ENABLED={str(config.MASK2FORMER_ENABLED).lower()}
EVOSSEARCH_M2F_MODEL={config.MASK2FORMER_MODEL}
EVOSSEARCH_M2F_DEVICE={config.MASK2FORMER_DEVICE}
EVOSSEARCH_M2F_MAX_SIZE={config.MASK2FORMER_MAX_SIZE}

# LM Studio / video understanding
EVOSSEARCH_LM_BASE_URL={config.LM_BASE_URL}
EVOSSEARCH_LM_MODEL={config.LM_MODEL}
EVOSSEARCH_LM_API_KEY={config.LM_API_KEY}
EVOSSEARCH_LM_TIMEOUT={config.LM_TIMEOUT}
EVOSSEARCH_LM_VIDEO_DEFAULT_FRAMES={config.LM_VIDEO_DEFAULT_FRAMES}
EVOSSEARCH_LM_VIDEO_MAX_FRAMES={config.LM_VIDEO_MAX_FRAMES}
EVOSSEARCH_LM_VIDEO_MAX_EDGE={config.LM_VIDEO_MAX_EDGE}
EVOSSEARCH_LM_VIDEO_MAX_TOKENS={config.LM_VIDEO_MAX_TOKENS}
EVOSSEARCH_LM_VIDEO_TEMPERATURE={config.LM_VIDEO_TEMPERATURE}

# Luxriot Evo integration
EVOSSEARCH_LUXRIOT_BASE_URL={luxriot_base_url}
EVOSSEARCH_LUXRIOT_USERNAME={luxriot_username}
EVOSSEARCH_LUXRIOT_PASSWORD={luxriot_password}
EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID={luxriot_default_channel_id}
EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL={luxriot_snapshot_interval}
EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE={luxriot_snapshot_max_edge}
EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES={luxriot_max_buffer_frames}
EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS={str(luxriot_auto_bookmarks).lower()}
EVOSSEARCH_LUXRIOT_SEV_INFO={merged_sev['info']}
EVOSSEARCH_LUXRIOT_SEV_LOW={merged_sev['low']}
EVOSSEARCH_LUXRIOT_SEV_NORMAL={merged_sev['normal']}
EVOSSEARCH_LUXRIOT_SEV_HIGH={merged_sev['high']}
EVOSSEARCH_LUXRIOT_SEV_CRITICAL={merged_sev['critical']}

# Probe / monitoring
EVOSSEARCH_PROBE_MAX_FRAMES={config.PROBE_MAX_FRAMES}
EVOSSEARCH_PROBE_THUMB_MAX_EDGE={config.PROBE_THUMB_MAX_EDGE}
EVOSSEARCH_PROBE_BOOKMARK_COOLDOWN_SEC={probe_bookmark_cooldown_sec}
EVOSSEARCH_PROBE_BOOKMARK_DEDUPE_WINDOW_SEC={probe_bookmark_dedupe_window_sec}
EVOSSEARCH_PROBE_BOOKMARK_SIM_HIGH={probe_bookmark_sim_high}
EVOSSEARCH_PROBE_BOOKMARK_MARGIN_DELTA={probe_bookmark_margin_delta}
EVOSSEARCH_PROBE_BOOKMARK_SCORE_DELTA={probe_bookmark_score_delta}
EVOSSEARCH_PROBE_BOOKMARK_MAX_FRAME_GAP={probe_bookmark_max_frame_gap}

# Detections archive / adaptive retention
EVOSSEARCH_DETECTIONS_ARCHIVE_ENABLED={str(config.DETECTIONS_ARCHIVE_ENABLED).lower()}
EVOSSEARCH_DETECTIONS_ARCHIVE_DIR={config.DETECTIONS_ARCHIVE_DIR}
EVOSSEARCH_DETECTIONS_ARCHIVE_JPEG_QUALITY={config.DETECTIONS_ARCHIVE_JPEG_QUALITY}
EVOSSEARCH_DETECTIONS_RETENTION_ENABLED={str(config.DETECTIONS_RETENTION_ENABLED).lower()}
EVOSSEARCH_DETECTIONS_RETENTION_DROP_SKIPPED={str(config.DETECTIONS_RETENTION_DROP_SKIPPED).lower()}
EVOSSEARCH_DETECTIONS_RETENTION_WINDOW_SEC={config.DETECTIONS_RETENTION_WINDOW_SEC}
EVOSSEARCH_DETECTIONS_RETENTION_FORCE_KEEP_SEC={config.DETECTIONS_RETENTION_FORCE_KEEP_SEC}
EVOSSEARCH_DETECTIONS_RETENTION_SIMILARITY_HIGH={config.DETECTIONS_RETENTION_SIMILARITY_HIGH}
EVOSSEARCH_DETECTIONS_RETENTION_SIMILARITY_LOW={config.DETECTIONS_RETENTION_SIMILARITY_LOW}
EVOSSEARCH_DETECTIONS_RETENTION_MARGIN_DELTA={config.DETECTIONS_RETENTION_MARGIN_DELTA}
EVOSSEARCH_DETECTIONS_RETENTION_SCORE_DELTA={config.DETECTIONS_RETENTION_SCORE_DELTA}

# Search result limits
EVOSSEARCH_MIN_RESULTS={min_results}
EVOSSEARCH_MAX_RESULTS={max_results}
EVOSSEARCH_DEFAULT_RESULTS={default_results}

# Processing configuration
EVOSSEARCH_BATCH_SIZE={batch_size}
EVOSSEARCH_THUMBNAIL_QUALITY={thumbnail_quality}

# File system configuration
EVOSSEARCH_INDEX_FOLDER={index_folder}

# Comment system configuration
EVOSSEARCH_MAX_COMMENT_LENGTH={max_comment_length}

# Security configuration
EVOSSEARCH_MAX_FILE_SIZE_MB={max_file_size}
EVOSSEARCH_ADMIN_TOKEN={config.ADMIN_TOKEN}
EVOSSEARCH_SETTINGS_LOCAL_ONLY={str(config.SETTINGS_LOCAL_ONLY).lower()}
EVOSSEARCH_CORS_ALLOWED_ORIGINS={','.join(config.CORS_ALLOWED_ORIGINS)}
EVOSSEARCH_ALLOWED_ROOTS={os.pathsep.join(config.ALLOWED_ROOTS)}
"""

        known_env_keys = set(_parse_env_editor_text(env_content).keys())
        env_content = env_content.rstrip() + _preserve_additional_env_lines(known_env_keys)
        if not env_content.endswith("\n"):
            env_content += "\n"

        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)

        config.HOST = data['host']
        config.PORT = port
        config.DEBUG = debug_enabled
        config.EMBEDDER = embedder
        config.CLIP_MODEL = data['clipModel']
        config.DINO_MODEL = dino_model
        config.EMB_DIM_DINO = dino_dim
        config.DINO_WEIGHTS_PATH = dino_weights_path
        config.DINO_DEVICE = dino_device
        config.INDEX_MODE = index_mode
        config.MIN_RESULTS = min_results
        config.MAX_RESULTS = max_results
        config.DEFAULT_RESULTS = default_results
        config.BATCH_SIZE = batch_size
        config.THUMBNAIL_QUALITY = thumbnail_quality
        config.MAX_COMMENT_LENGTH = max_comment_length
        config.MAX_FILE_SIZE_MB = max_file_size
        config.INDEX_FOLDER_NAME = index_folder
        app.config["MAX_CONTENT_LENGTH"] = max(1, int(config.MAX_FILE_SIZE_MB)) * 1024 * 1024
        config.FUSION_ENABLED = fusion_enabled
        config.FUSION_ALPHA = fusion_alpha
        config.RERANK_ENABLED = rerank_enabled
        config.RERANK_TOP_K = rerank_top_k
        config.DINO_SEGMENTS_ENABLED = segments_enabled
        config.DINO_SEGMENT_MIN_PATCHES = segment_min_patches
        config.DINO_HEATMAP_THRESHOLD = segment_threshold
        config.LUXRIOT_BASE_URL = luxriot_base_url
        config.LUXRIOT_USERNAME = luxriot_username
        config.LUXRIOT_PASSWORD = luxriot_password
        config.LUXRIOT_DEFAULT_CHANNEL_ID = luxriot_default_channel_id
        config.LUXRIOT_SNAPSHOT_INTERVAL = luxriot_snapshot_interval
        config.LUXRIOT_SNAPSHOT_MAX_EDGE = luxriot_snapshot_max_edge
        config.LUXRIOT_MAX_BUFFER_FRAMES = luxriot_max_buffer_frames
        config.LUXRIOT_AUTO_BOOKMARKS = luxriot_auto_bookmarks
        config.LUXRIOT_SEVERITY_MAP = merged_sev
        config.PROBE_BOOKMARK_COOLDOWN_SEC = probe_bookmark_cooldown_sec
        config.PROBE_BOOKMARK_DEDUPE_WINDOW_SEC = probe_bookmark_dedupe_window_sec
        config.PROBE_BOOKMARK_SIM_HIGH = probe_bookmark_sim_high
        config.PROBE_BOOKMARK_MARGIN_DELTA = probe_bookmark_margin_delta
        config.PROBE_BOOKMARK_SCORE_DELTA = probe_bookmark_score_delta
        config.PROBE_BOOKMARK_MAX_FRAME_GAP = probe_bookmark_max_frame_gap
        probe_bookmark_gate = _ProbeBookmarkGate()

        active_embedder = embedder
        if active_embedder == 'fusion' and not config.FUSION_ENABLED:
            active_embedder = 'clip'
        reset_embedder_runtime_state()
        warmup_warning = warm_start_embedder()
        message = 'Settings saved successfully. Restart the server if issues persist.'
        payload: Dict[str, Any] = {'success': True, 'message': message}
        if warmup_warning:
            payload['warning'] = warmup_warning
        return jsonify(payload)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _stop_probe_daemon_thread() -> None:
    global probe_daemon_thread
    probe_daemon_stop.set()
    if probe_daemon_thread is not None and probe_daemon_thread.is_alive():
        probe_daemon_thread.join(timeout=1.5)


def _port_is_available(host: str, port: int) -> bool:
    bind_host = (host or "").strip() or "0.0.0.0"
    try:
        port_num = int(port)
    except Exception:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port_num))
            return True
        except OSError:
            return False


@app.route('/agent/chat', methods=['POST'])
def agent_chat():
    """SSE streaming agent chat endpoint."""
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    message = str(data.get('message') or '').strip()
    if not message:
        return jsonify({'error': 'message is required'}), 400
    session_id = str(data.get('session_id') or '').strip() or None
    image_b64  = str(data.get('image_b64') or '').strip() or None
    tool_context = None
    auth_context = _current_auth_context()
    if _auth_enabled() and auth_context is not None:
        tool_context = ToolExecutionContext(
            actor_id=auth_context.user_id,
            tenant_id=auth_context.tenant_id,
            roles=auth_context.roles,
            permissions=auth_context.permissions,
            allowed_channel_ids={
                str(channel_id)
                for channel_id in auth_context.allowed_channel_ids
            },
            agent_session_id=session_id,
            request_id=auth_context.request_id,
            client_ip=_source_ip(),
        )

    try:
        runner = _get_agent_runner()
    except Exception as exc:
        return jsonify({'error': f'Agent unavailable: {exc}'}), 503

    def _generate():
        yield from runner.stream_chat(
            session_id=session_id,
            message=message,
            image_b64=image_b64,
            tool_context=tool_context,
        )

    response = Response(
        stream_with_context(_generate()),
        mimetype='text/event-stream',
    )
    response.headers['Cache-Control']     = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection']        = 'keep-alive'
    return response


@app.route('/agent/action-plans/<plan_id>/execute', methods=['POST'])
def agent_action_plan_execute(plan_id: str):
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    auth_context = _current_auth_context()
    if _auth_enabled() and auth_context is None:
        return jsonify({'error': 'Authentication required'}), 401
    if not _auth_enabled() or auth_context is None:
        return jsonify({'error': 'Durable approvals require named users'}), 403
    tool_context = ToolExecutionContext(
        actor_id=auth_context.user_id,
        tenant_id=auth_context.tenant_id,
        roles=auth_context.roles,
        permissions=auth_context.permissions,
        allowed_channel_ids={
            str(channel_id)
            for channel_id in auth_context.allowed_channel_ids
        },
        request_id=auth_context.request_id,
        client_ip=_source_ip(),
    )
    try:
        runner = _get_agent_runner()
        result = runner.approve_action_plan(plan_id, tool_context)
        return jsonify({'success': True, 'result': result})
    except ToolGatewayError as exc:
        status = 403 if getattr(exc, 'code', '') in {'permission_denied', 'channel_access_denied'} else 409
        return jsonify({'success': False, 'error': str(exc), 'code': getattr(exc, 'code', 'tool_error')}), status
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/agent/sessions', methods=['GET'])
def agent_sessions():
    try:
        runner = _get_agent_runner()
    except Exception as exc:
        return jsonify({'error': f'Agent unavailable: {exc}'}), 503
    return jsonify({
        'sessions': runner.store.list_sessions(**_agent_session_owner())
    })


@app.route('/agent/config', methods=['GET', 'POST'])
def agent_config():
    global _agent_runner, _agent_runtime_model_override
    if request.method == 'GET':
        return jsonify(_get_agent_config_payload())

    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    raw_model = str(data.get('model') or '').strip()
    default_model = str(config.LM_MODEL or '').strip()
    with _agent_runner_lock:
        _agent_runtime_model_override = raw_model if raw_model and raw_model != default_model else None
        _agent_runner = None
    return jsonify({'success': True, **_get_agent_config_payload()})


@app.route('/lm/models', methods=['GET'])
def lm_models():
    force = str(request.args.get('force') or '').strip().lower() in TRUE_BOOL_STRINGS
    payload = _fetch_lm_model_catalog(force=force)
    payload['agent'] = _get_agent_config_payload()
    return jsonify(payload)


@app.route('/agent/skills', methods=['GET'])
def agent_skills():
    try:
        return jsonify({'skills': _list_skill_records()})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/agent/skills/create', methods=['POST'])
def agent_skills_create():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    raw_name = str(data.get('name') or '').strip()
    raw_slug = str(data.get('slug') or '').strip()
    slug = _slugify_skill_name(raw_slug or raw_name)
    if not raw_name:
        return jsonify({'error': 'name is required'}), 400
    if not slug:
        return jsonify({'error': 'Could not derive a valid skill slug'}), 400
    try:
        skill_file = _resolve_skill_path(slug)
        if skill_file.exists():
            return jsonify({'error': f'Skill already exists: {slug}'}), 409
        skill = _save_skill_record(slug, raw_name, str(data.get('content') or ''))
        return jsonify({'success': True, 'skill': skill})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/agent/skills/<slug>', methods=['GET', 'POST'])
def agent_skill_detail(slug: str):
    if request.method == 'GET':
        try:
            return jsonify(_load_skill_record(slug))
        except FileNotFoundError:
            return jsonify({'error': 'Skill not found'}), 404
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        current = _load_skill_record(slug)
    except FileNotFoundError:
        return jsonify({'error': 'Skill not found'}), 404
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    try:
        updated = _save_skill_record(
            slug=current['slug'],
            name=str(data.get('name') or current.get('name') or current['slug']),
            content=str(data.get('content') or current.get('content') or ''),
        )
        return jsonify({'success': True, 'skill': updated})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/agent/session/<session_id>', methods=['GET', 'DELETE'])
def agent_session(session_id: str):
    try:
        runner = _get_agent_runner()
    except Exception as exc:
        return jsonify({'error': f'Agent unavailable: {exc}'}), 503
    if request.method == 'DELETE':
        guard = _mutation_guard_error()
        if guard is not None:
            return guard
        ok = runner.store.delete_session(
            session_id,
            **_agent_session_owner(),
        )
        if not ok:
            return jsonify({'error': 'Session not found'}), 404
        return jsonify({'status': 'deleted'})
    session = runner.store.get_session(
        session_id,
        **_agent_session_owner(),
    )
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(session)


@atexit.register
def _shutdown_background_workers() -> None:
    global _audit_db_pool, _audit_writer, _control_plane_db_pool
    global _identity_repository
    global _inference_queue_runtime, _inference_worker_db_pool
    try:
        if _inference_queue_runtime is not None:
            _inference_queue_runtime.stop()
            _inference_queue_runtime = None
    except Exception:
        pass
    try:
        if _inference_worker_db_pool is not None:
            _inference_worker_db_pool.close()
            _inference_worker_db_pool = None
    except Exception:
        pass
    try:
        if _audit_db_pool is not None:
            _audit_db_pool.close()
            _audit_db_pool = None
            _audit_writer = None
    except Exception:
        pass
    try:
        if _control_plane_db_pool is not None:
            _control_plane_db_pool.close()
            _control_plane_db_pool = None
            _identity_repository = None
    except Exception:
        pass
    try:
        luxriot_manager.stop_all_streams(stop_video=True, stop_analytics=True, pause_analytics=False)
    except Exception:
        pass
    try:
        luxriot_manager.persist_summary_state()
    except Exception:
        pass
    try:
        luxriot_manager.persist_rollup_cache()
    except Exception:
        pass
    try:
        _stop_probe_daemon_thread()
    except Exception:
        pass


if __name__ == '__main__':
    if not _port_is_available(config.HOST, config.PORT):
        print(f"Startup aborted: {config.HOST}:{config.PORT} is already in use.")
        print("Stop the existing server process or change EVOSSEARCH_PORT before starting oldapp.py.")
        raise SystemExit(1)
    warmup_warning = warm_start_embedder()
    if warmup_warning:
        print(f"Embedder warm-up warning: {warmup_warning}")
    config.print_startup_info()
    if probe_daemon_thread is None:
        probe_daemon_thread = threading.Thread(target=_probe_daemon, daemon=True)
        probe_daemon_thread.start()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
