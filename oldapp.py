import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import atexit
import base64
import copy
import gc
import hashlib
import html as html_lib
import json
import math
import pickle
import re
import secrets
import socket
import tempfile
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
from PIL import Image
from transformers import AutoModel, AutoProcessor
from flask import Flask, g, request, jsonify, send_file, make_response, render_template, Response, stream_with_context
from flask_cors import CORS

from config import config
from agent_postgres_store import PostgresAgentStore
from archive_store import (
    ARCHIVE_RUNTIME_REVISION,
    ArchiveStoreNotReady,
    PostgresDetectionsStore,
    PostgresProbesStore,
    PostgresRuntimeStateStore,
)
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
_clip_module: Optional[Any] = None
dino_encoder: Optional[DINOEncoder] = None
mask2former_head: Optional["Mask2FormerHead"] = None
_mask2former_lock = Lock()
_mask2former_failed = False
SUPPORTED_EMBEDDERS = {"clip", "dino", "fusion"}
EMBEDDER_SUBDIRS: Dict[str, str] = {"clip": "clip", "dino": "dino"}


class _FaissTypingStub:
    Index = Any


faiss: Any = _FaissTypingStub()
_faiss_module: Optional[Any] = None
FaissIndexBundle = Tuple[Optional[faiss.Index], Optional[List[str]], Optional[List[Dict[str, Any]]], Dict[str, Any]]


def _get_faiss() -> Any:
    """Load FAISS only for index/search paths.

    Some older CPU-only hosts can import the rest of the control plane but crash
    with SIGILL when importing faiss-cpu wheels. Keeping this lazy lets auth,
    audit, settings, Luxriot, and remote LLM smoke tests run on those hosts.
    """
    global faiss, _faiss_module
    if not _local_vision_stack_enabled():
        raise RuntimeError(
            "Local vision stack is disabled. Set EVOSSEARCH_LOCAL_VISION_ENABLED=true "
            "on a host that supports the installed CLIP/FAISS wheels."
        )
    if _faiss_module is None:
        import importlib

        _faiss_module = importlib.import_module("faiss")
        faiss = _faiss_module
    return _faiss_module


def _get_clip_module() -> Any:
    """Load clip-anytorch only when CLIP embeddings are actually needed."""
    global _clip_module
    if not _local_vision_stack_enabled():
        raise RuntimeError(
            "Local vision stack is disabled. Set EVOSSEARCH_LOCAL_VISION_ENABLED=true "
            "on a host that supports the installed CLIP/FAISS wheels."
        )
    if _clip_module is None:
        import importlib

        _clip_module = importlib.import_module("clip")
    return _clip_module


def _local_vision_stack_enabled() -> bool:
    return str(
        os.getenv("EVOSSEARCH_LOCAL_VISION_ENABLED", "true") or "true"
    ).strip().lower() in TRUE_BOOL_STRINGS

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
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
_auth_service: Optional[AuthenticationService] = None
_auth_service_lock = Lock()
_identity_repository: Optional[Any] = None
_identity_repository_lock = Lock()
_audit_writer: Optional[Any] = None
_audit_writer_lock = Lock()
_audit_reader: Optional[Any] = None
_audit_reader_lock = Lock()
_audit_db_pool: Optional[PsycopgPool] = None
_audit_db_pool_lock = Lock()
_control_plane_db_pool: Optional[PsycopgPool] = None
_control_plane_db_lock = Lock()
_inference_worker_db_pool: Optional[PsycopgPool] = None
_inference_worker_db_lock = Lock()
_inference_queue_runtime: Optional[LuxriotInferenceQueueRuntime] = None
_inference_queue_lock = Lock()


def _experimental_embedding_models_enabled() -> bool:
    return bool(getattr(config, "EXPERIMENTAL_EMBEDDERS_ENABLED", False))


def _production_clip_model() -> str:
    model = str(getattr(config, "PRODUCTION_CLIP_MODEL", "ViT-B/32") or "").strip()
    return model or "ViT-B/32"


def _normalize_clip_model_for_policy(model_name: Any) -> str:
    requested = str(model_name or "").strip()
    if not requested:
        requested = _production_clip_model()
    if _experimental_embedding_models_enabled():
        return requested
    return _production_clip_model()


def _normalize_embedder_for_policy(embedder: Any, fusion_enabled: bool = False) -> str:
    requested = str(embedder or "clip").strip().lower()
    if requested not in SUPPORTED_EMBEDDERS:
        requested = "clip"
    if not _experimental_embedding_models_enabled():
        return "clip"
    if requested == "fusion" and not fusion_enabled:
        return "clip"
    return requested


def _normalize_index_mode_for_policy(index_mode: Any) -> str:
    requested = str(index_mode or "clip").strip().lower()
    if requested not in {"clip", "dino", "dual"}:
        requested = "clip"
    if not _experimental_embedding_models_enabled():
        return "clip"
    return requested

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
    "probes_cast": Permission.PROBES_MANAGE,
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
    "serve_detection_thumbnail": Permission.DETECTIONS_VIEW,
    "get_comments": Permission.REPORTS_VIEW,
    "get_commented_images": Permission.REPORTS_VIEW,
    "check_index": Permission.DIAGNOSTICS_VIEW,
    "video_understanding": Permission.STREAMS_VIEW,
    "describe_image": Permission.DETECTIONS_VIEW,
    "get_settings_env": Permission.SETTINGS_MANAGE,
    "lm_models": Permission.DIAGNOSTICS_VIEW,
    "settings_archive_capacity": Permission.DIAGNOSTICS_VIEW,
    "search": Permission.DETECTIONS_VIEW,
    "search_by_image": Permission.DETECTIONS_VIEW,
    "search_by_mask": Permission.DETECTIONS_VIEW,
    "segment_from_point": Permission.DETECTIONS_VIEW,
    "detections_search_text": Permission.DETECTIONS_VIEW,
    "detections_search_image": Permission.DETECTIONS_VIEW,
    "detections_list": Permission.DETECTIONS_VIEW,
    "detections_summary": Permission.DETECTIONS_VIEW,
    "detections_diagnostics": Permission.DETECTIONS_VIEW,
    "luxriot_channels": Permission.STREAMS_VIEW,
    "luxriot_prompt_settings": Permission.STREAMS_VIEW,
    "luxriot_snapshot": Permission.STREAMS_VIEW,
    "luxriot_snapshot_capture": Permission.STREAMS_VIEW,
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
    "audit_events": Permission.AUDIT_VIEW,
}
_CHANNEL_REQUIRED_FOR_SCOPED_ENDPOINTS = frozenset(
    {
        "detections_list",
        "detections_search_image",
        "detections_search_text",
        "detections_summary",
        "describe_image",
        "search",
        "serve_image",
        "luxriot_prompt_settings",
        "probes_delete",
        "probes_run",
        "probes_save",
        "probes_cast",
    }
)
_ALL_CHANNELS_REQUIRED_ENDPOINTS = frozenset(
    {
        "audit_events",
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


def _archive_store_not_ready_response(exc: ArchiveStoreNotReady):
    app.logger.warning(
        "Archive store not ready request_id=%s error=%s",
        getattr(g, "request_id", ""),
        exc,
    )
    return jsonify(
        {
            "error": (
                "Archive storage is not ready. Apply the required database "
                "migration before using archive search."
            ),
            "not_ready": "archive_store",
            "required_revision": ARCHIVE_RUNTIME_REVISION,
        }
    ), 503


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


def _archive_store_mode() -> str:
    return "postgres"


def _archive_tenant_id() -> str:
    return str(
        getattr(config, "ARCHIVE_TENANT_ID", "")
        or getattr(config, "AUTH_TENANT_ID", "")
        or ""
    ).strip()


def _archive_store_required() -> bool:
    return True


class _UnavailablePostgresStore:
    """Fail-closed stand-in used when PostgreSQL runtime storage is unavailable."""

    backend = "postgres"

    def __init__(self, component: str, exc: Optional[Exception] = None) -> None:
        self.component = str(component or "archive")
        self.error = type(exc).__name__ if exc is not None else None
        self.detail = str(exc)[:240] if exc is not None else None

    def health(self) -> Dict[str, Any]:
        status = "error" if self.error else "not_configured"
        payload: Dict[str, Any] = {
            "ok": False,
            "status": status,
            "backend": self.backend,
            "component": self.component,
            "required_backend": "postgres",
        }
        if self.error:
            payload["error"] = self.error
        if self.detail:
            payload["detail"] = self.detail
        return payload

    def load_state(self, key: str) -> Optional[Dict[str, Any]]:
        raise ArchiveStoreNotReady(self._message())

    def save_state(self, key: str, payload: Mapping[str, Any]) -> None:
        raise ArchiveStoreNotReady(self._message())

    def _message(self) -> str:
        reason = f" ({self.error})" if self.error else ""
        return f"PostgreSQL {self.component} store is unavailable{reason}."

    def __getattr__(self, name: str) -> Any:
        def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ArchiveStoreNotReady(self._message())

        return _raise


def _postgres_archive_enabled() -> bool:
    if not _postgres_database_configured():
        return False
    if not _archive_tenant_id():
        return False
    return True


def _build_luxriot_runtime_state_store() -> Optional[PostgresRuntimeStateStore]:
    if not _postgres_archive_enabled():
        return cast(Any, _UnavailablePostgresStore("runtime_state"))
    try:
        return PostgresRuntimeStateStore(
            _get_control_plane_db_pool(),
            _archive_tenant_id(),
        )
    except Exception as exc:
        return cast(Any, _UnavailablePostgresStore("runtime_state", exc))


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


def _get_audit_reader() -> Any:
    global _audit_reader
    with _audit_reader_lock:
        if _audit_reader is None:
            from security.postgres_audit_reader import PostgresAuditReader

            _audit_reader = PostgresAuditReader(_get_control_plane_db_pool())
        return _audit_reader


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
        g.channel_resolution_error = "probe_lookup_failed"
        return set()
    matched: Set[str] = set()
    channel_ids: Set[int] = set()
    for probe in probes:
        probe_id = str(probe.get("id") or "")
        if probe_id not in wanted:
            continue
        matched.add(probe_id)
        channel_id = _to_optional_int(probe.get("channel_id"))
        if channel_id is not None:
            channel_ids.add(channel_id)
    if matched != wanted:
        g.channel_resolution_error = "probe_owner_missing"
    return channel_ids


def _image_channel_ids_for_request_value(value: Any) -> Set[int]:
    image_path = str(value or "").strip()
    if not image_path:
        return set()
    try:
        return {
            int(channel_id)
            for channel_id in detections_store.channel_ids_for_image_path(image_path)
            if _to_optional_int(channel_id) is not None
        }
    except Exception:
        g.channel_resolution_error = "image_owner_lookup_failed"
        return set()


def _detection_channel_ids_for_request_value(value: Any) -> Set[int]:
    detection_id = _to_optional_int(value)
    if detection_id is None:
        return set()
    try:
        rows = detections_store.fetch_detections_by_ids([detection_id], include_vectors=False)
    except Exception:
        g.channel_resolution_error = "detection_owner_lookup_failed"
        return set()
    channel_ids: Set[int] = set()
    for row in rows:
        channel_id = _to_optional_int(row.get("channel_id"))
        if channel_id is not None:
            channel_ids.add(channel_id)
    if not channel_ids:
        g.channel_resolution_error = "detection_owner_missing"
    return channel_ids


def _request_image_channel_ids(
    endpoint: str,
    view_args: Mapping[str, Any],
    payload: Any,
    form: Mapping[str, Any],
) -> Set[int]:
    values: List[Any] = []
    detection_ids: List[Any] = []
    if endpoint == "serve_detection_image":
        values.append(request.args.get("image_path"))
    if endpoint == "serve_detection_thumbnail":
        detection_ids.append(view_args.get("detection_id"))
    if endpoint == "serve_image":
        values.append(request.args.get("image_path"))
        values.append(view_args.get("filepath"))
    if endpoint == "describe_image":
        if isinstance(payload, Mapping):
            values.append(payload.get("image_path"))
        values.append(form.get("image_path"))
    channel_ids: Set[int] = set()
    for value in values:
        channel_ids.update(_image_channel_ids_for_request_value(value))
    for value in detection_ids:
        channel_ids.update(_detection_channel_ids_for_request_value(value))
    return channel_ids


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
        for key in ("channel_ids", "channels"):
            raw_values = payload.get(key)
            if isinstance(raw_values, (list, tuple, set)):
                candidates.extend(raw_values)
            elif raw_values is not None:
                candidates.append(raw_values)
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
        if endpoint in {"probes_delete", "probes_run"}:
            if "id" in source:
                probe_ids.append(source.get("id"))
    channel_ids.update(_probe_channel_ids(probe_ids))
    channel_ids.update(
        _request_image_channel_ids(endpoint, view_args, payload, form)
    )

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


def _filter_stream_status_for_context(
    status: Mapping[str, Any],
    context: Optional[AuthContext],
) -> Dict[str, Any]:
    filtered = dict(status)
    if not _auth_enabled() or context is None:
        return filtered
    for key in ("video_streams", "analytics_streams"):
        filtered[key] = [
            item
            for item in filtered.get(key) or []
            if _can_access_context_channel(
                context,
                item.get("channel_id") if isinstance(item, Mapping) else None,
            )
        ]
    for key in (
        "paused_analytics_channels",
        "video_history_channels",
    ):
        filtered[key] = [
            channel_id
            for channel_id in filtered.get(key) or []
            if _can_access_context_channel(context, channel_id)
        ]
    filtered["running_total"] = sum(
        len(filtered.get(key) or [])
        for key in ("video_streams", "analytics_streams")
    )
    return filtered


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


def _audit_fingerprint(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _audit_key_details(name: str, values: Iterable[Any], *, limit: int = 100) -> Dict[str, Any]:
    keys = sorted({str(value).strip() for value in values if str(value).strip()})
    details: Dict[str, Any] = {
        f"{name}_count": len(keys),
        name: keys[:limit],
    }
    if len(keys) > limit:
        details[f"{name}_truncated"] = True
    return details


def _write_completion_audit_or_error(
    *,
    action: str,
    result: str = "success",
    target_type: str = "route",
    target_id: Optional[str] = None,
    channel_id: Optional[int] = None,
    details: Optional[Mapping[str, Any]] = None,
):
    if not _auth_enabled():
        return None
    try:
        _write_security_audit(
            context=_current_auth_context(),
            action=action,
            result=result,
            target_type=target_type,
            target_id=target_id,
            channel_id=channel_id,
            details=details,
        )
    except Exception:
        return _auth_failure_response("Audit service unavailable", 503)
    return None


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
        if _is_channel_scoped(context) and getattr(
            g,
            "channel_resolution_error",
            None,
        ):
            raise PermissionError(str(g.channel_resolution_error))
        if (
            _is_channel_scoped(context)
            and str(request.endpoint or "")
            in _CHANNEL_REQUIRED_FOR_SCOPED_ENDPOINTS
            and not channel_ids
        ):
            raise PermissionError("an explicit authorized channel is required")
        if (
            _is_channel_scoped(context)
            and str(request.endpoint or "") in {"serve_detection_image", "serve_detection_thumbnail"}
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


_INDEXED_FOLDER_ENDPOINTS = {
    "get_comments",
    "save_comment",
    "get_commented_images",
    "check_index",
    "index_folder",
    "index_segments",
    "search",
    "search_by_image",
    "search_by_mask",
    "segment_from_point",
}


def _request_has_field(field: str) -> bool:
    if field in request.args or field in request.form:
        return True
    data = request.get_json(silent=True) if request.is_json else None
    return isinstance(data, Mapping) and field in data


def _disabled_feature_response():
    endpoint = str(request.endpoint or "")
    if endpoint == "video_understanding" and not bool(getattr(config, "OFFLINE_VIDEO_ENABLED", False)):
        return jsonify({"error": "offline_video_disabled"}), 404
    if endpoint == "luxriot_snapshot_capture" and not bool(getattr(config, "PROBE_SNAP_ENABLED", False)):
        return jsonify({"error": "probe_snap_disabled"}), 404

    indexed_folder_enabled = bool(getattr(config, "INDEXED_FOLDER_ENABLED", False))
    if indexed_folder_enabled:
        return None
    if endpoint in _INDEXED_FOLDER_ENDPOINTS:
        return jsonify({"error": "indexed_folder_disabled"}), 404
    if endpoint == "describe_image" and _request_has_field("folder"):
        return jsonify({"error": "indexed_folder_disabled"}), 404
    if endpoint == "serve_image" and "folder" in request.args:
        return "Not found", 404
    return None


@app.before_request
def _bind_request_security_context() -> None:
    g.request_id = _request_id()
    g.auth_context = None
    g.auth_session = None
    g.auth_resolution_error = None
    g.channel_resolution_error = None
    disabled_feature = _disabled_feature_response()
    if disabled_feature is not None:
        return disabled_feature
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


def _permission_guard_error(permission: Permission, *, action: str):
    if not _auth_enabled():
        return None
    guard = _session_guard(
        permission=permission,
        require_csrf=False,
        action=action,
    )
    if guard is None:
        return None
    body, status = guard
    payload = body.get_json(silent=True) if hasattr(body, "get_json") else {}
    message = (payload or {}).get("error") or "Access denied"
    return jsonify({"error": message}), status


def _current_request_has_permission(permission: Permission) -> bool:
    if not _auth_enabled():
        return True
    context = _current_auth_context()
    return bool(context and permission.value in context.permissions)


def _bookmark_permission_guard_error(*, action: str):
    return _permission_guard_error(Permission.BOOKMARKS_CREATE, action=action)


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
    requested_model = _normalize_clip_model_for_policy(config.CLIP_MODEL)
    config.CLIP_MODEL = requested_model
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
    clip_module = _get_clip_module()
    model, preprocess = clip_module.load(model_name, device=target_device)
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
            text_tokens = _get_clip_module().tokenize(prepared).to(clip_runtime_device)
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


def _index_metadata_compatible(embedder: str, meta: Mapping[str, Any]) -> bool:
    if not meta:
        return True
    if embedder == "clip":
        expected_model = _normalize_clip_model_for_policy(config.CLIP_MODEL)
        meta_model = str(meta.get("model") or meta.get("requested_model") or "").strip()
        if meta_model and meta_model != expected_model:
            return False
        if not _experimental_embedding_models_enabled():
            backend = str(meta.get("backend") or "").strip().lower()
            if backend and backend != "openai_clip":
                return False
        return True
    if embedder == "dino":
        if not _experimental_embedding_models_enabled():
            return False
        meta_model = str(meta.get("config_model") or meta.get("model") or "").strip()
        return not meta_model or meta_model == str(config.DINO_MODEL)
    return True


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
    index = _get_faiss().IndexFlatIP(embeddings_array.shape[1])
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
        luxriot_alert_policy_prompt=getattr(config, 'LUXRIOT_ALERT_POLICY_PROMPT', '') or '',
        luxriot_rollup_prompt_l1=getattr(config, 'LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT', rollup_default) or rollup_default,
        luxriot_rollup_prompt_l2=getattr(config, 'LUXRIOT_ROLLUP_L2_SYSTEM_PROMPT', rollup_default) or rollup_default,
        luxriot_rollup_prompt_l3=getattr(config, 'LUXRIOT_ROLLUP_L3_SYSTEM_PROMPT', rollup_default) or rollup_default,
        luxriot_json_alert_prompt=getattr(config, 'LUXRIOT_ALERTS_JSON_PROMPT', LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT) or LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT,
        auth_enabled=bool(config.AUTH_ENABLED),
        offline_video_hidden_class='' if getattr(config, "OFFLINE_VIDEO_ENABLED", False) else 'deployment-hidden',
        probe_snap_hidden_class='' if getattr(config, "PROBE_SNAP_ENABLED", False) else 'deployment-hidden',
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
        app.logger.info(
            "Image request rejected request_id=%s error=%s",
            getattr(g, "request_id", ""),
            exc,
        )
        return "Invalid image request", 400
    except Exception as exc:
        app.logger.exception(
            "Image serving failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return "Image unavailable", 500


@app.route('/detections/image', methods=['GET'])
def serve_detection_image():
    image_path = request.args.get('image_path')
    try:
        resolved = detection_archive.resolve_archive_image_path(image_path)
        return send_file(str(resolved))
    except ValueError as exc:
        message = str(exc).lower()
        status = 404 if message == "image not found" else 400
        app.logger.info(
            "Detection image request rejected request_id=%s error=%s",
            getattr(g, "request_id", ""),
            exc,
        )
        return ("Image not found" if status == 404 else "Invalid image request"), status
    except Exception as exc:
        app.logger.exception(
            "Detection image serving failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return "Image unavailable", 500


def _strip_image_data_url_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("data:image/") and "," in text:
        return text.split(",", 1)[1].strip()
    return text


@app.route('/detections/thumbnail/<int:detection_id>', methods=['GET'])
def serve_detection_thumbnail(detection_id: int):
    try:
        rows = detections_store.fetch_detections_by_ids([detection_id], include_vectors=False)
        if not rows:
            return "Image not found", 404
        thumbnail_b64 = _strip_image_data_url_prefix(rows[0].get("thumbnail"))
        if not thumbnail_b64:
            return "Image not found", 404
        try:
            image_bytes = base64.b64decode(thumbnail_b64, validate=True)
        except Exception as exc:
            raise ValueError("Invalid thumbnail data") from exc
        return Response(image_bytes, mimetype="image/jpeg")
    except ValueError as exc:
        app.logger.info(
            "Detection thumbnail request rejected request_id=%s detection_id=%s error=%s",
            getattr(g, "request_id", ""),
            detection_id,
            exc,
        )
        return "Invalid image request", 400
    except Exception:
        app.logger.exception(
            "Detection thumbnail serving failed request_id=%s detection_id=%s",
            getattr(g, "request_id", ""),
            detection_id,
        )
        return "Image unavailable", 500


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


def _uploaded_file_suffix(file_obj: Any, fallback: str = "") -> str:
    filename = str(getattr(file_obj, "filename", "") or "").strip()
    suffix = Path(filename).suffix.lower()
    return suffix or fallback


def _save_upload_to_temp(file_obj: Any, *, allowed_suffixes: Set[str], prefix: str) -> Path:
    if file_obj is None or not str(getattr(file_obj, "filename", "") or "").strip():
        raise ValueError("No upload supplied")
    suffix = _uploaded_file_suffix(file_obj)
    if suffix not in allowed_suffixes:
        raise ValueError("Unsupported uploaded file type")
    tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        file_obj.save(str(tmp_path))
        if tmp_path.stat().st_size <= 0:
            raise ValueError("Uploaded file is empty")
        return tmp_path
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _cleanup_temp_upload(path_obj: Optional[Path]) -> None:
    if path_obj is None:
        return
    try:
        path_obj.unlink(missing_ok=True)
    except Exception:
        pass


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


def _configured_lm_profiles() -> Dict[str, Dict[str, Any]]:
    profiles = getattr(config, "LM_PROFILES", None)
    if isinstance(profiles, Mapping) and profiles:
        return {
            str(profile_id): dict(profile)
            for profile_id, profile in profiles.items()
            if str(profile_id).strip()
        }
    return {
        "default": {
            "id": "default",
            "kind": "general",
            "base_url": config.LM_BASE_URL,
            "model": config.LM_MODEL,
            "api_key": config.LM_API_KEY,
            "timeout": config.LM_TIMEOUT,
        }
    }


def _default_lm_profile_id(kind: str = "general") -> str:
    profiles = _configured_lm_profiles()
    configured = (
        getattr(config, "LM_AGENT_PROFILE_ID", "")
        if kind == "agent"
        else getattr(config, "LM_VLM_PROFILE_ID", "")
        if kind in {"vlm", "vision", "video"}
        else "default"
    )
    profile_id = str(configured or "").strip()
    if profile_id in profiles:
        return profile_id
    if kind in {"vlm", "vision", "video"}:
        for candidate_id, profile in profiles.items():
            if str(profile.get("kind") or "").lower() in {"vlm", "vision", "video"}:
                return candidate_id
    if kind == "agent":
        for candidate_id, profile in profiles.items():
            if str(profile.get("kind") or "").lower() == "agent":
                return candidate_id
    return "default" if "default" in profiles else next(iter(profiles))


def _lm_profile_selector_value(profile: Mapping[str, Any]) -> str:
    profile_id = str(profile.get("id") or "").strip()
    model = str(profile.get("model") or "").strip()
    if profile_id and profile_id != "default":
        return profile_id
    return model or profile_id


LM_AUTO_BALANCE_SELECTOR = "__auto__"
LM_AUTO_BALANCE_LABEL = "Auto balance"
LM_AUTO_BALANCE_ALIASES = {
    LM_AUTO_BALANCE_SELECTOR,
    "auto",
    "auto-balance",
    "auto_balance",
}
VLM_PROFILE_KINDS = {"vlm", "vision", "video"}


def _lm_profile_enabled(profile: Mapping[str, Any]) -> bool:
    raw = profile.get("enabled")
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"false", "0", "no", "off"}


def _is_auto_lm_selector(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in LM_AUTO_BALANCE_ALIASES


def _configured_vlm_balancer_profile_ids() -> List[str]:
    profiles = _configured_lm_profiles()
    raw_configured = getattr(config, "LM_VLM_BALANCER_PROFILES", ())
    if isinstance(raw_configured, str):
        configured_ids = [
            item.strip()
            for item in raw_configured.split(",")
            if item.strip()
        ]
    else:
        configured_ids = [
            str(item).strip()
            for item in (raw_configured or ())
            if str(item).strip()
        ]

    profile_ids: List[str] = []
    if configured_ids:
        candidates = configured_ids
    else:
        candidates = [
            profile_id
            for profile_id, profile in profiles.items()
            if str(profile.get("kind") or "").strip().lower() in VLM_PROFILE_KINDS
        ]
        if not candidates:
            candidates = [_default_lm_profile_id("vlm")]

    for profile_id in candidates:
        if profile_id in profile_ids:
            continue
        profile = profiles.get(profile_id)
        if not isinstance(profile, Mapping):
            continue
        if not _lm_profile_enabled(profile):
            continue
        if not str(profile.get("base_url") or "").strip():
            continue
        if not str(profile.get("model") or "").strip():
            continue
        profile_ids.append(profile_id)
    return profile_ids


def _vlm_balancer_enabled() -> bool:
    return bool(getattr(config, "LM_VLM_BALANCER_ENABLED", False))


def _stable_vlm_profile_for_channel(channel_id: int, profile_ids: Sequence[str]) -> Optional[str]:
    return _stable_vlm_profile_for_key(str(int(channel_id)), profile_ids)


def _stable_vlm_profile_for_key(key: str, profile_ids: Sequence[str]) -> Optional[str]:
    if not profile_ids:
        return None
    normalized_key = str(key or "default").strip() or "default"
    digest = hashlib.sha256(f"vlm:{normalized_key}".encode("utf-8")).digest()
    slot = int.from_bytes(digest[:8], "big") % len(profile_ids)
    return str(profile_ids[slot])


def _resolve_vlm_auto_model_hint(
    requested_model_hint: Optional[str],
    *,
    assignment_key: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    raw_hint = str(requested_model_hint or "").strip()
    profiles = _configured_lm_profiles()
    if raw_hint and not _is_auto_lm_selector(raw_hint):
        return raw_hint, {
            "mode": "manual",
            "requested": raw_hint,
            "assigned_profile_id": raw_hint if raw_hint in profiles else None,
            "balancer_enabled": _vlm_balancer_enabled(),
        }

    default_profile = _resolve_lm_profile(kind="vlm")
    default_selector = _lm_profile_selector_value(default_profile)
    if not _vlm_balancer_enabled():
        return (default_selector if raw_hint else None), {
            "mode": "default",
            "requested": raw_hint or None,
            "assigned_profile_id": str(default_profile.get("id") or "").strip() or None,
            "balancer_enabled": False,
            "profile_count": len(_configured_vlm_balancer_profile_ids()),
        }

    profile_ids = _configured_vlm_balancer_profile_ids()
    selected_profile_id = _stable_vlm_profile_for_key(assignment_key, profile_ids)
    if not selected_profile_id:
        return (default_selector if raw_hint else None), {
            "mode": "default",
            "requested": raw_hint or None,
            "assigned_profile_id": str(default_profile.get("id") or "").strip() or None,
            "balancer_enabled": True,
            "profile_count": 0,
            "reason": "no_vlm_profiles",
        }
    return selected_profile_id, {
        "mode": "auto",
        "requested": raw_hint or None,
        "assigned_profile_id": selected_profile_id,
        "balancer_enabled": True,
        "profile_count": len(profile_ids),
    }


def _resolve_luxriot_vlm_model_hint(
    channel_id: int,
    requested_model_hint: Optional[str],
) -> Tuple[Optional[str], Dict[str, Any]]:
    return _resolve_vlm_auto_model_hint(
        requested_model_hint,
        assignment_key=str(int(channel_id)),
    )


def _resolve_offline_lm_model_hint(
    requested_model_hint: Optional[str],
    *,
    assignment_key: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    raw_hint = str(requested_model_hint or "").strip()
    if raw_hint:
        if _is_auto_lm_selector(raw_hint):
            return _resolve_vlm_auto_model_hint(raw_hint, assignment_key=assignment_key)
        profiles = _configured_lm_profiles()
        return raw_hint, {
            "mode": "manual",
            "requested": raw_hint,
            "assigned_profile_id": raw_hint if raw_hint in profiles else None,
            "balancer_enabled": _vlm_balancer_enabled(),
        }

    agent_profile = _resolve_lm_profile(kind="agent")
    agent_selector = _lm_profile_selector_value(agent_profile)
    return agent_selector, {
        "mode": "default_agent",
        "requested": None,
        "assigned_profile_id": str(agent_profile.get("id") or "").strip() or None,
        "balancer_enabled": False,
        "profile_count": 1,
    }


def _offline_vlm_assignment_key(kind: str, value: Any) -> str:
    raw = str(value or "").strip()
    if len(raw) > 240:
        raw = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"offline:{kind}:{raw or 'request'}"


def _lm_profile_env_key(profile_id: str, suffix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", profile_id).strip("_").upper()
    return f"EVOSSEARCH_LM_PROFILE_{normalized or 'DEFAULT'}_{suffix}"


def _resolve_lm_profile(
    *,
    profile_id: Optional[str] = None,
    model_override: Optional[str] = None,
    kind: str = "general",
) -> Dict[str, Any]:
    profiles = _configured_lm_profiles()
    selected_profile_id = str(profile_id or "").strip()
    selected_model_override = str(model_override or "").strip()
    if not selected_profile_id and selected_model_override in profiles:
        selected_profile_id = selected_model_override
        selected_model_override = ""
    if not selected_profile_id:
        selected_profile_id = _default_lm_profile_id(kind)
    if selected_profile_id not in profiles:
        selected_profile_id = _default_lm_profile_id(kind)
    profile = dict(profiles[selected_profile_id])
    profile["id"] = selected_profile_id
    profile["base_url"] = str(profile.get("base_url") or "").rstrip("/")
    profile["model"] = selected_model_override or str(
        profile.get("model") or config.LM_MODEL
    ).strip()
    profile["api_key"] = str(profile.get("api_key") or "").strip()
    try:
        profile["timeout"] = min(
            3600,
            max(1, int(profile.get("timeout") or config.LM_TIMEOUT)),
        )
    except (TypeError, ValueError):
        profile["timeout"] = int(config.LM_TIMEOUT)
    profile["kind"] = str(profile.get("kind") or kind or "general").lower()
    return profile


def _public_lm_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(profile.get("id") or ""),
        "kind": str(profile.get("kind") or "general"),
        "model": str(profile.get("model") or ""),
        "selector": _lm_profile_selector_value(profile),
        "timeout": int(profile.get("timeout") or config.LM_TIMEOUT),
        "enabled": _lm_profile_enabled(profile),
        "gpu": str(profile.get("gpu") or ""),
    }


def _call_lm_chat(
    messages: List[Dict[str, Any]],
    model_override: Optional[str] = None,
    *,
    profile_id: Optional[str] = None,
    profile_kind: str = "vlm",
) -> str:
    profile = _resolve_lm_profile(
        profile_id=profile_id,
        model_override=model_override,
        kind=profile_kind,
    )
    base_url = str(profile.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("LM profile base URL is not configured.")
    if not str(profile.get("model") or "").strip():
        raise RuntimeError(f"LM profile {profile['id']} model is not configured.")
    endpoint = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if profile.get("api_key"):
        headers["Authorization"] = f"Bearer {profile['api_key']}"

    payload = {
        "model": str(profile.get("model") or "").strip(),
        "messages": messages,
        "temperature": float(config.LM_VIDEO_TEMPERATURE),
        "max_tokens": int(config.LM_VIDEO_MAX_TOKENS),
    }
    response: Optional[requests.Response] = None

    def _response_error_detail(resp: Any) -> str:
        try:
            data = resp.json()
        except Exception:
            data = None
        if isinstance(data, Mapping):
            error = data.get("error")
            if isinstance(error, Mapping):
                message = str(error.get("message") or "").strip()
                error_type = str(error.get("type") or "").strip()
                if message and error_type:
                    return f"{message} ({error_type})"
                if message:
                    return message
            message = str(data.get("message") or "").strip()
            if message:
                return message
        text = str(getattr(resp, "text", "") or "").strip()
        if text:
            return text[:500]
        return "empty error response"

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=int(profile.get("timeout") or config.LM_TIMEOUT),
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        resp = getattr(exc, "response", None) or response
        detail = _response_error_detail(resp) if resp is not None else str(exc)
        status = getattr(resp, "status_code", None)
        status_text = f"HTTP {status}" if status else "HTTP error"
        raise RuntimeError(
            f"LM request failed for profile {profile['id']} "
            f"(model {payload['model']}): {status_text}; {detail}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"LM request failed for profile {profile['id']} "
            f"(model {payload['model']}): {exc}"
        ) from exc

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


def _call_video_understanding(
    messages: List[Dict[str, Any]],
    model_override: Optional[str] = None,
    *,
    profile_id: Optional[str] = None,
) -> str:
    return _call_lm_chat(
        messages,
        model_override=model_override,
        profile_id=profile_id,
        profile_kind="vlm",
    )


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
        severity = str(raw.get('severity') or 'normal').strip().lower()
        severity_aliases = {
            'information': 'info',
            'informational': 'info',
            'warn': 'low',
            'warning': 'low',
            'medium': 'normal',
            'moderate': 'normal',
            'danger': 'high',
            'emergency': 'critical',
        }
        severity = severity_aliases.get(severity, severity)
        allowed_sev = {'info', 'low', 'normal', 'high', 'critical'}
        if severity not in allowed_sev:
            severity = 'normal'
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

    def _extract_prose_alerts(blob: str) -> List[Dict[str, Any]]:
        severity_aliases = {
            "info": "info",
            "information": "info",
            "informational": "info",
            "low": "low",
            "warn": "low",
            "warning": "low",
            "normal": "normal",
            "moderate": "normal",
            "high": "high",
            "danger": "high",
            "critical": "critical",
            "emergency": "critical",
        }
        pattern = re.compile(
            r"^\s*(?:[-*•]|\d+[.)])?\s*"
            r"(?P<label>info(?:rmation(?:al)?)?|low|warn(?:ing)?|normal|moderate|high|critical|danger|emergency)"
            r"\s*(?:level|alert|severity)?\s*[:\-–]\s*(?P<description>.+?)\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        out: List[Dict[str, Any]] = []
        for match in pattern.finditer(blob or ""):
            raw_label = str(match.group("label") or "").strip().lower()
            severity = severity_aliases.get(raw_label, "normal")
            description = " ".join(str(match.group("description") or "").strip().split())
            if not description:
                continue
            title = re.split(r"\s*\(|[.;]\s*", description, maxsplit=1)[0].strip()
            if not title:
                title = description
            if len(title) > 80:
                title = title[:77].rstrip() + "..."
            out.append(
                {
                    "title": title,
                    "description": description,
                    "severity": severity,
                    "state": "new",
                    "channel_id": default_channel_id,
                    "timestamp_ms": base_ts_ms,
                }
            )
        return out

    def _alert_token_set(raw: Dict[str, Any]) -> Set[str]:
        text = f"{raw.get('title') or ''} {raw.get('description') or ''}".lower()
        stop_words = {
            "the", "and", "from", "with", "that", "this", "into", "onto", "person",
            "detected", "observed", "visible", "snapshot", "snapshots", "approx",
            "event", "level", "alert",
        }
        tokens = {
            token.rstrip("s")
            for token in re.findall(r"[a-zа-яё0-9]{3,}", text, flags=re.IGNORECASE)
            if token not in stop_words
        }
        return {token for token in tokens if token}

    def _is_near_duplicate_alert(candidate: Dict[str, Any], existing_alerts: Sequence[Dict[str, Any]]) -> bool:
        candidate_tokens = _alert_token_set(candidate)
        if len(candidate_tokens) < 2:
            return False
        candidate_severity = str(candidate.get("severity") or "").lower()
        for existing in existing_alerts:
            if str(existing.get("severity") or "").lower() != candidate_severity:
                continue
            existing_tokens = _alert_token_set(existing)
            if len(candidate_tokens & existing_tokens) >= 2:
                return True
        return False

    alerts: List[Dict[str, Any]] = []
    seen_alerts: Set[str] = set()
    raw_alerts: List[Tuple[str, Any]] = []
    for candidate in _extract_candidates(text or ''):
        if isinstance(candidate, dict) and isinstance(candidate.get('alerts'), list):
            for raw_alert in candidate['alerts']:
                raw_alerts.append(("json", raw_alert))
    raw_alerts.extend(("prose", raw_alert) for raw_alert in _extract_prose_alerts(text or ''))
    for source, raw_alert in raw_alerts:
        validated = _validate_alert(raw_alert)
        if not validated:
            continue
        if source == "prose" and _is_near_duplicate_alert(validated, alerts):
            continue
        alert_key = json.dumps(
            {
                "title": validated.get("title"),
                "description": validated.get("description"),
                "severity": validated.get("severity"),
                "state": validated.get("state"),
                "channel_id": validated.get("channel_id"),
                "timestamp_ms": validated.get("timestamp_ms"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if alert_key in seen_alerts:
            continue
        seen_alerts.add(alert_key)
        alerts.append(validated)

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
PREVIOUS_LUXRIOT_ROLLUP_PROMPT_DEFAULTS = {
    "L1": (
        "You are a CCTV operations analyst. Summarize multiple L0 batch notes for one short time window.\n"
        "Return Markdown using exactly these sections:\n"
        "### Window snapshot\n"
        "### Scene baseline\n"
        "### Key changes\n"
        "### Alerts/signals\n"
        "### Operator notes\n"
        "Rules: keep factual language; deduplicate repeated observations; include timestamps when available; "
        "avoid phrases like 'L1 rollup from L0'."
    ),
    "L2": (
        "You are a CCTV operations analyst. Summarize multiple L1 summaries into one hour-scale view.\n"
        "Return Markdown using exactly these sections:\n"
        "### Window snapshot\n"
        "### Routine baseline\n"
        "### Significant changes\n"
        "### Alerts/signals\n"
        "### Operator notes\n"
        "Rules: preserve meaningful deviations from routine; avoid repeating unchanged background details; "
        "keep concise, operator-facing language."
    ),
    "L3": (
        "You are a CCTV operations analyst. Summarize multiple L2 summaries into a longer period narrative.\n"
        "Return Markdown using exactly these sections:\n"
        "### Window snapshot\n"
        "### Persistent patterns\n"
        "### Notable events\n"
        "### Risks and follow-ups\n"
        "### Operator notes\n"
        "Rules: emphasize trend shifts and durable signals; remove duplicate wording; focus on actionable context."
    ),
}
PREVIOUS_LUXRIOT_GENERIC_ROLLUP_PROMPT_DEFAULTS = {
    "L2": (
        "You are a CCTV operations summarizer. Consolidate multiple short L1 summaries into one clear L2 rollup. "
        "Remove repetition, keep concrete scene changes and timestamps, and avoid boilerplate."
    ),
    "L3": (
        "You are a CCTV operations summarizer. Consolidate multiple short L2 summaries into one clear L3 rollup. "
        "Remove repetition, keep concrete scene changes and timestamps, and avoid boilerplate."
    ),
}
LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT = (
    "Machine-readable alert output for operator review:\n"
    "- Always append exactly one block at the end, prefixed with ALERTS_JSON:.\n"
    "- If no trigger matches, use {\"alerts\": []}.\n"
    "- If one or more triggers match, include one alert object per distinct visible trigger using this schema:\n"
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
    "Alert candidates are defined by the Alert review policy and by visible immediate safety/security hazards. "
    "General hazards include physical violence, a person falling/collapsing or appearing to need urgent help, "
    "dangerous vehicle behavior, forced entry, property damage, theft-like tampering, weapon/fire/smoke, "
    "critical camera obstruction, or crowd escalation. "
    "Do not classify behavior as illegal/unlawful; describe visible facts requiring operator review. "
    "Do not alert on littering, loitering, casual gestures, routine walking/parking/deliveries, or ambiguous movement "
    "unless the Alert review policy explicitly asks for that review signal. "
    "Rules: emit one alert object per distinct visible trigger in the batch, up to 8 objects; "
    "do not merge unrelated triggers into one alert; do not output a prose-only Alerts section or Warning Level list; "
    "if a matching event is described anywhere in the prose summary, it must also appear in ALERTS_JSON; "
    "evaluate every operator-defined trigger independently against the current snapshots; "
    "if two distinct triggers are visible in the same batch, emit two alert objects; "
    "timestamp_ms should be observed batch epoch in milliseconds (or 0 if unknown)."
)
PREVIOUS_LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT_V2 = (
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
    PREVIOUS_LUXRIOT_ALERTS_JSON_PROMPT_DEFAULT_V2.strip(),
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
    "the backend appends current-observation and ALERTS_JSON instructions; follow that final output contract."
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
    runtime_state_store=_build_luxriot_runtime_state_store(),
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
        outdated_rollup_prompts = {
            level: {
                legacy_rollup_prompt,
                str(PREVIOUS_LUXRIOT_ROLLUP_PROMPT_DEFAULTS.get(level, '') or '').strip(),
                str(PREVIOUS_LUXRIOT_GENERIC_ROLLUP_PROMPT_DEFAULTS.get(level, '') or '').strip(),
            }
            for level in ('L1', 'L2', 'L3')
        }
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
            if not current_level_prompt or current_level_prompt in outdated_rollup_prompts.get(level, set()):
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
                    if (not raw_level_prompt) or raw_level_prompt in outdated_rollup_prompts.get(level, set()):
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
    backend = "json"

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

    def health(self) -> Dict[str, Any]:
        try:
            with self.lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            return {
                "ok": True,
                "status": "reachable",
                "backend": self.backend,
                "path": str(self.path),
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "backend": self.backend,
                "path": str(self.path),
                "error": str(exc),
            }


def _build_archive_stores() -> Tuple[Any, Any]:
    if not _postgres_archive_enabled():
        unavailable = _UnavailablePostgresStore("archive")
        return unavailable, unavailable
    try:
        pool = _get_control_plane_db_pool()
        tenant_id = _archive_tenant_id()
        return (
            PostgresProbesStore(pool, tenant_id),
            PostgresDetectionsStore(
                pool,
                tenant_id,
                max_records=int(getattr(config, "ARCHIVE_MAX_RECORDS", 5000000)),
            )
        )
    except Exception as exc:
        unavailable = _UnavailablePostgresStore("archive", exc)
        return unavailable, unavailable


probes_store, detections_store = _build_archive_stores()
APP_STARTED_AT = time.time()


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


_PUBLIC_READY_KEYS = frozenset({"ok", "status", "required"})


def _public_ready_checks(
    checks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    public_checks: Dict[str, Dict[str, Any]] = {}
    for name, payload in checks.items():
        public_checks[str(name)] = {
            key: payload[key]
            for key in _PUBLIC_READY_KEYS
            if key in payload
        }
    return public_checks


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
    required_postgres = _archive_store_required()
    store_checks: Dict[str, Dict[str, Any]] = {}

    for name, store in (
        ("detections", detections_store),
        ("probes", probes_store),
    ):
        try:
            health_fn = getattr(store, "health")
            health = dict(health_fn())
        except Exception as exc:
            health = {"ok": False, "status": "error", "error": type(exc).__name__}
        if required_postgres and health.get("backend") != "postgres":
            health["ok"] = False
            health["status"] = "not_postgres"
            health["required_backend"] = "postgres"
        store_checks[name] = health

    runtime_state_store = getattr(luxriot_manager, "runtime_state_store", None)
    if runtime_state_store is not None:
        try:
            runtime_state = dict(runtime_state_store.health())
        except Exception as exc:
            runtime_state = {
                "ok": False,
                "status": "error",
                "error": type(exc).__name__,
            }
    else:
        runtime_state = {
            "ok": not required_postgres,
            "status": "not_configured",
            "backend": "postgres",
            "required_backend": "postgres" if required_postgres else None,
        }
    store_checks["runtime_state"] = runtime_state

    ok = all(bool(item.get("ok")) for item in store_checks.values())
    first_status = next(
        (
            str(item.get("status") or "error")
            for item in store_checks.values()
            if not item.get("ok")
        ),
        "reachable",
    )
    return _component_result(
        ok,
        "reachable" if ok else first_status,
        backend=str(store_checks["detections"].get("backend") or ""),
        archive_store_mode=_archive_store_mode(),
        tenant_id=_archive_tenant_id() or None,
        stores=store_checks,
        retention={
            "enabled": bool(getattr(config, "ARCHIVE_RETENTION_ENABLED", True)),
            "row_retention_days": float(getattr(config, "ARCHIVE_ROW_RETENTION_DAYS", 90.0)),
            "thumbnail_retention_days": float(getattr(config, "ARCHIVE_THUMBNAIL_RETENTION_DAYS", 14.0)),
            "max_records": int(getattr(config, "ARCHIVE_MAX_RECORDS", 5000000)),
            "prune_interval_sec": float(getattr(config, "ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC", 3600.0)),
            "last_result": dict(_archive_retention_last_result),
        },
    )


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
            default_model=_lm_profile_selector_value(_resolve_lm_profile(kind="vlm")),
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
        pool = _get_control_plane_db_pool()
        result = pool.check_readiness()
        role_result = pool.check_runtime_role(
            strict=bool(getattr(config, "DB_STRICT_RUNTIME_ROLES", False))
        )
        ok = result.ready and role_result.ready
        return _component_result(
            ok,
            role_result.state.value if result.ready else result.state.value,
            detail=result.detail,
            latency_ms=result.latency_ms,
            current_revision=result.current_revision,
            expected_revision=result.expected_revision,
            runtime_role_ok=role_result.ready,
            runtime_role_status=role_result.state.value,
            runtime_user=role_result.current_user,
            session_user=role_result.session_user,
            runtime_unsafe_reason=role_result.unsafe_reason,
            strict_runtime_roles=bool(
                getattr(config, "DB_STRICT_RUNTIME_ROLES", False)
            ),
        )
    except Exception as exc:
        return _component_result(
            False,
            "unavailable",
            error=type(exc).__name__,
        )


def _secret_is_weak(value: str) -> bool:
    secret = str(value or "").strip()
    if not secret:
        return False
    if len(secret) < 32:
        return True
    lowered = secret.lower()
    weak_values = {
        "123",
        "1234",
        "12345",
        "admin",
        "password",
        "changeme",
        "change-me",
        "secret",
        "unit-token",
    }
    return lowered in weak_values or len(set(secret)) < 8


def _check_deployment_security_ready() -> Dict[str, Any]:
    secure_required = bool(
        getattr(config, "SECURE_DEPLOYMENT_REQUIRED", False)
    )
    issues: List[str] = []
    if not bool(getattr(config, "AUTH_COOKIE_SECURE", False)):
        issues.append("EVOSSEARCH_AUTH_COOKIE_SECURE=true is required behind TLS")
    if not config.ALLOWED_ROOTS:
        issues.append("EVOSSEARCH_ALLOWED_ROOTS must whitelist deployment data roots")
    else:
        invalid_roots = [
            str(root)
            for root in config.ALLOWED_ROOTS
            if not Path(root).expanduser().exists()
        ]
        if invalid_roots:
            issues.append("ALLOWED_ROOTS contains missing paths: " + ", ".join(invalid_roots[:3]))
    if _secret_is_weak(getattr(config, "ADMIN_TOKEN", "")):
        issues.append("EVOSSEARCH_ADMIN_TOKEN is set but weak; rotate or unset under named auth")
    if _secret_is_weak(getattr(config, "LUXRIOT_PASSWORD", "")):
        issues.append("EVOSSEARCH_LUXRIOT_PASSWORD appears to be a placeholder")

    ok = not secure_required or not issues
    return _component_result(
        ok,
        "ready" if ok else "misconfigured",
        required=secure_required,
        issues=issues,
        secure_deployment_required=secure_required,
        auth_cookie_secure=bool(getattr(config, "AUTH_COOKIE_SECURE", False)),
        allowed_roots_count=len(config.ALLOWED_ROOTS),
        admin_token_set=bool(getattr(config, "ADMIN_TOKEN", "")),
        settings_local_only=bool(getattr(config, "SETTINGS_LOCAL_ONLY", False)),
    )


def _check_auth_ready() -> Dict[str, Any]:
    secure_required = bool(
        getattr(config, "SECURE_DEPLOYMENT_REQUIRED", False)
    )
    if not _auth_enabled():
        return _component_result(
            False,
            "disabled",
            required=secure_required,
            error=(
                "EVOSSEARCH_AUTH_ENABLED=true is required for secure deployment"
                if secure_required
                else None
            ),
        )
    if secure_required and not bool(
        getattr(config, "DB_STRICT_RUNTIME_ROLES", False)
    ):
        return _component_result(
            False,
            "misconfigured",
            error=(
                "EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true is required for "
                "secure deployment"
            ),
        )
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
        audit_pool = _get_audit_db_pool()
        audit_database = audit_pool.check_health()
        audit_role = audit_pool.check_runtime_role(
            strict=bool(getattr(config, "DB_STRICT_RUNTIME_ROLES", False))
        )
    except Exception as exc:
        return _component_result(
            False,
            "unavailable",
            error=f"audit database unavailable ({type(exc).__name__})",
            tenant_id=tenant_id,
        )
    ok = bool(postgres.get("ok")) and audit_database.ready and audit_role.ready
    return _component_result(
        ok,
        "ready" if ok else "unavailable",
        tenant_id=tenant_id,
        audit_latency_ms=audit_database.latency_ms,
        audit_runtime_role_ok=audit_role.ready,
        audit_runtime_role_status=audit_role.state.value,
        audit_runtime_user=audit_role.current_user,
        audit_runtime_unsafe_reason=audit_role.unsafe_reason,
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


def _check_lm_profiles_ready(timeout_sec: float = 1.0) -> Dict[str, Any]:
    profiles = _configured_lm_profiles()
    named_profile_ids = [
        profile_id
        for profile_id in profiles
        if str(profile_id).strip() and str(profile_id).strip() != "default"
    ]
    if not _vlm_balancer_enabled() and not named_profile_ids:
        return _component_result(
            False,
            "not_configured",
            required=False,
            profile_count=0,
            required_profile_ids=[],
            profiles=[],
        )
    configured_required_ids: List[str] = []
    if _vlm_balancer_enabled():
        configured_required_ids.extend(_configured_vlm_balancer_profile_ids())

    required_ids: List[str] = []
    for profile_id in configured_required_ids:
        if profile_id and profile_id not in required_ids:
            required_ids.append(profile_id)

    profile_payloads: List[Dict[str, Any]] = []
    any_checked = False
    required_ok = True
    enabled_profile_ids = [
        profile_id
        for profile_id, profile in profiles.items()
        if _lm_profile_enabled(profile)
        and (_vlm_balancer_enabled() or profile_id in named_profile_ids)
    ]
    for profile_id in enabled_profile_ids:
        try:
            profile = _resolve_lm_profile(profile_id=profile_id)
        except Exception as exc:
            profile_required = profile_id in required_ids
            required_ok = required_ok and not profile_required
            profile_payloads.append(
                {
                    "id": profile_id,
                    "required": profile_required,
                    "ok": False,
                    "status": "config_error",
                    "error": type(exc).__name__,
                }
            )
            continue

        profile_required = profile_id in required_ids
        base_url = str(profile.get("base_url") or "").rstrip("/")
        model = str(profile.get("model") or "").strip()
        public_profile = _public_lm_profile(profile)
        profile_status: Dict[str, Any] = {
            "id": profile_id,
            "kind": public_profile.get("kind"),
            "selector": public_profile.get("selector"),
            "model": model,
            "gpu": public_profile.get("gpu") or "",
            "base_url": base_url,
            "required": profile_required,
            "ok": False,
        }
        if not base_url or not model:
            profile_status["status"] = "not_configured"
            if profile_required:
                required_ok = False
            profile_payloads.append(profile_status)
            continue

        any_checked = True
        headers = {"Accept": "application/json"}
        if profile.get("api_key"):
            headers["Authorization"] = f"Bearer {profile['api_key']}"
        start = time.monotonic()
        try:
            resp = requests.get(
                f"{base_url}/models",
                headers=headers,
                timeout=(0.5, max(0.5, float(timeout_sec))),
                stream=True,
            )
            try:
                status_code = int(resp.status_code)
                ok = 200 <= status_code < 400
                profile_status.update(
                    {
                        "ok": ok,
                        "status": "reachable" if ok else "http_error",
                        "status_code": status_code,
                        "latency_ms": round(
                            max(0.0, time.monotonic() - start) * 1000.0,
                            3,
                        ),
                    }
                )
            finally:
                resp.close()
        except Exception as exc:
            profile_status.update(
                {
                    "ok": False,
                    "status": "error",
                    "error": type(exc).__name__,
                }
            )
        if profile_required and not profile_status.get("ok"):
            required_ok = False
        profile_payloads.append(profile_status)

    required = bool(required_ids)
    if required:
        ok = required_ok
    else:
        ok = any(bool(profile.get("ok")) for profile in profile_payloads)
    return _component_result(
        ok,
        "ready" if ok else ("not_configured" if not any_checked else "unavailable"),
        required=required,
        profile_count=len(profile_payloads),
        required_profile_ids=required_ids,
        profiles=profile_payloads,
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
_luxriot_restore_result: Dict[str, Any] = {}
try:
    _luxriot_restore_result = luxriot_manager.restore_desired_live_sessions()
except Exception as exc:
    _luxriot_restore_result = {
        "ok": False,
        "status": "error",
        "error": type(exc).__name__,
    }


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
    details_requested = str(request.args.get("details") or "").strip().lower() in TRUE_BOOL_STRINGS
    secure_required = bool(getattr(config, "SECURE_DEPLOYMENT_REQUIRED", False))
    details_allowed = not secure_required
    if secure_required and (details_requested or load_embedder):
        if not _auth_enabled():
            return _auth_failure_response(
                "Named-user authentication is disabled",
                503,
            )
        guard = _session_guard(
            permission=Permission.DIAGNOSTICS_VIEW,
            require_csrf=False,
            action="ready.details",
        )
        if guard is not None:
            return guard
        details_allowed = True
    embedder_required = str(
        os.getenv("EVOSSEARCH_EMBEDDER_REQUIRED", "true") or "true"
    ).strip().lower() in TRUE_BOOL_STRINGS

    checks: Dict[str, Dict[str, Any]] = {
        "database": _check_database_ready(),
        "postgresql": _check_postgres_ready(),
        "authentication": _check_auth_ready(),
        "deployment_security": _check_deployment_security_ready(),
        "inference_queue": _check_inference_queue_ready(),
        "lm_profiles": _check_lm_profiles_ready(),
        "embedder": _embedder_loaded_state(),
        "luxriot": _check_luxriot_ready(),
        "luxriot_restore": _component_result(
            bool(_luxriot_restore_result.get("ok", True)),
            str(_luxriot_restore_result.get("status") or "unknown"),
            required=False,
            **{
                key: value
                for key, value in _luxriot_restore_result.items()
                if key not in {"ok", "status"}
            },
        ),
    }

    if load_embedder and details_allowed and not checks["embedder"].get("ok"):
        try:
            ensure_embedder_loaded(active_embedder)
            checks["embedder"] = _embedder_loaded_state()
        except Exception as exc:
            app.logger.warning(
                "Readiness embedder load failed request_id=%s error=%s",
                getattr(g, "request_id", ""),
                exc,
            )
            checks["embedder"] = _component_result(
                False,
                "load_failed",
                embedder=active_embedder,
                error=type(exc).__name__,
            )

    required_names = ["database"]
    if embedder_required:
        required_names.append("embedder")
    else:
        checks["embedder"]["required"] = False
    if checks["postgresql"].get("required"):
        required_names.append("postgresql")
    if checks["authentication"].get("required"):
        required_names.append("authentication")
    if checks["deployment_security"].get("required"):
        required_names.append("deployment_security")
    if checks["inference_queue"].get("required"):
        required_names.append("inference_queue")
    if checks["lm_profiles"].get("required"):
        required_names.append("lm_profiles")
    if strict or checks["luxriot"].get("required"):
        required_names.append("luxriot")

    is_ready = all(bool(checks[name].get("ok")) for name in required_names)
    status_code = 200 if is_ready else 503
    return jsonify(
        {
            "status": "ready" if is_ready else "not_ready",
            "version": config.APP_VERSION,
            "required": required_names,
            "checks": checks if details_allowed else _public_ready_checks(checks),
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


def _identity_session_payload(session: Any) -> Dict[str, Any]:
    return {
        "id": str(getattr(session, "session_id", "")),
        "tenantId": str(getattr(session, "tenant_id", "")),
        "userId": str(getattr(session, "user_id", "")),
        "username": str(getattr(session, "username", "")),
        "createdAt": getattr(session, "created_at").isoformat()
        if getattr(session, "created_at", None) is not None
        else None,
        "lastSeenAt": getattr(session, "last_seen_at").isoformat()
        if getattr(session, "last_seen_at", None) is not None
        else None,
        "expiresAt": getattr(session, "expires_at").isoformat()
        if getattr(session, "expires_at", None) is not None
        else None,
        "revokedAt": getattr(session, "revoked_at").isoformat()
        if getattr(session, "revoked_at", None) is not None
        else None,
        "revokeReason": getattr(session, "revoke_reason", None),
        "clientIp": getattr(session, "client_ip", None),
        "userAgent": getattr(session, "user_agent", None),
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
        try:
            _write_security_audit(
                context=None,
                action="auth.login",
                result="denied",
                target_type="user",
                details={"reason": "missing_credentials"},
            )
        except Exception:
            return _auth_failure_response("Audit service unavailable", 503)
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


@app.route('/auth/sessions', methods=['GET'])
def auth_sessions():
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    guard = _auth_admin_guard(write=False, action="auth.sessions.list")
    if guard is not None:
        return guard
    context = _current_auth_context()
    if context is None:
        return _auth_failure_response("Authentication required", 401)
    user_id = (
        str(request.args.get("userId") or request.args.get("user_id") or "").strip()
        or None
    )
    active_only = _coerce_bool(
        request.args.get("activeOnly") or request.args.get("active_only"),
        default=True,
    )
    try:
        sessions = _get_identity_repository().list_sessions(
            context.tenant_id,
            actor_user_id=context.user_id,
            user_id=user_id,
            active_only=active_only,
        )
    except ValueError as exc:
        return _auth_failure_response(str(exc), 400)
    except Exception:
        return _auth_failure_response("Identity service unavailable", 503)
    return jsonify(
        {
            "success": True,
            "sessions": [_identity_session_payload(session) for session in sessions],
        }
    )


@app.route('/auth/sessions/<session_id>/revoke', methods=['POST'])
def auth_session_revoke(session_id: str):
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    guard = _auth_admin_guard(write=True, action="auth.sessions.revoke")
    if guard is not None:
        return guard
    context = _current_auth_context()
    if context is None:
        return _auth_failure_response("Authentication required", 401)
    data = _json_body()
    reason = str(data.get("reason") or "admin_revoked").strip()
    try:
        revoked = _get_identity_repository().revoke_session_by_id(
            context.tenant_id,
            session_id,
            actor_user_id=context.user_id,
            reason=reason,
        )
        if not revoked:
            return _auth_failure_response("Session not found", 404)
        _write_security_audit(
            context=context,
            action="auth.sessions.revoke.completed",
            result="success",
            target_type="iam_session",
            target_id=session_id,
            details={"reason": reason},
        )
    except ValueError as exc:
        return _auth_failure_response(str(exc), 400)
    except Exception:
        return _auth_failure_response("Identity service unavailable", 503)
    return jsonify({"success": True, "revoked": True})


@app.route('/audit/events', methods=['GET'])
def audit_events():
    if not _auth_enabled():
        return _auth_failure_response("Named-user authentication is disabled", 503)
    context = _current_auth_context()
    if context is None:
        return _auth_failure_response("Authentication required", 401)
    try:
        page = _get_audit_reader().list_events(
            context,
            limit=request.args.get("limit"),
            cursor=request.args.get("cursor"),
            since=request.args.get("since"),
            until=request.args.get("until"),
            actor_user_id=(
                request.args.get("actorUserId")
                or request.args.get("actor_user_id")
            ),
            action=request.args.get("action"),
            target_type=(
                request.args.get("targetType")
                or request.args.get("target_type")
            ),
            target_id=(
                request.args.get("targetId")
                or request.args.get("target_id")
            ),
            channel_id=(
                request.args.get("channelId")
                or request.args.get("channel_id")
            ),
            result=request.args.get("result"),
            request_id=(
                request.args.get("requestId")
                or request.args.get("request_id")
            ),
        )
    except ValueError as exc:
        return _auth_failure_response(str(exc), 400)
    except Exception:
        return _auth_failure_response("Audit service unavailable", 503)
    return jsonify(
        {
            "success": True,
            "events": [event.to_dict() for event in page.events],
            "nextCursor": page.next_cursor,
        }
    )


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
    context = _current_auth_context()
    try:
        revoked = _get_auth_service().logout(session_token, reason="logout")
    except Exception:
        return _auth_failure_response("Authentication service unavailable", 503)
    try:
        _write_security_audit(
            context=context,
            action="auth.logout.completed",
            result="success",
            target_type="session",
            details={"revoked": bool(revoked)},
        )
    except Exception:
        return _auth_failure_response("Audit service unavailable", 503)
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
        from agent import (
            AGENT_MAX_MESSAGES_PER_SESSION,
            AGENT_MAX_SESSIONS,
            AGENT_SESSION_TTL_DAYS,
            AgentRunner,
        )
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
            source: Optional[str] = None,
            since_ms: Optional[int] = None,
            until_ms: Optional[int] = None,
            limit: int = 12,
            sort_by: str = "similarity",
            candidate_limit: int = 20000,
            mode: str = "clip",
            include_coverage: bool = False,
        ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
            vec = get_text_embedding(query)
            payload = _search_detections_archive(
                clip_query_vec=vec,
                dino_query_vec=None,
                mode=mode,
                probe_id=probe_id,
                channel_id=channel_id,
                source=_normalize_archive_source_filter(source),
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                sort_by=sort_by,
                candidate_limit=candidate_limit,
                include_coverage=include_coverage,
            )
            if include_coverage and isinstance(payload, tuple):
                results, coverage = payload
                return {"results": results, "coverage": coverage}
            return cast(List[Dict[str, Any]], payload)

        def _agent_tool_lm_chat(messages: List[Dict[str, Any]]) -> str:
            return _call_lm_chat(
                messages,
                model_override=_agent_runtime_model_override,
                profile_kind="agent",
            )

        agent_profile = _resolve_lm_profile(
            model_override=_agent_runtime_model_override,
            kind="agent",
        )
        _agent_runner = AgentRunner(
            embed_text_fn=lambda text: get_text_embedding(text),
            embed_image_fn=lambda img: get_image_embedding_from_pil(img, embedder='clip'),
            call_lm_fn=_agent_tool_lm_chat,
            encode_jpeg_fn=_encode_jpeg,
            probes_store=probes_store,
            detections_store=detections_store,
            luxriot_manager=luxriot_manager,
            search_indexed_folder_fn=_agent_search_folder,
            search_detections_fn=_agent_search_detections,
            lm_base_url=str(agent_profile.get("base_url") or ""),
            lm_model=str(agent_profile.get("model") or ""),
            lm_api_key=str(agent_profile.get("api_key") or ""),
            lm_timeout=int(agent_profile.get("timeout") or config.LM_TIMEOUT),
            store=PostgresAgentStore(
                _get_control_plane_db_pool(),
                max_sessions=AGENT_MAX_SESSIONS,
                max_messages_per_session=AGENT_MAX_MESSAGES_PER_SESSION,
                session_ttl_days=AGENT_SESSION_TTL_DAYS,
            ),
            tool_audit_callback=_write_agent_tool_audit,
            tool_plan_store=approval_store,
            tool_approval_store=approval_store,
        )
        return _agent_runner


def _get_agent_config_payload() -> Dict[str, Any]:
    profile = _resolve_lm_profile(
        model_override=_agent_runtime_model_override,
        kind="agent",
    )
    default_profile = _resolve_lm_profile(kind="agent")
    selected_value = (
        str(_agent_runtime_model_override or "").strip()
        or _lm_profile_selector_value(profile)
    )
    default_value = _lm_profile_selector_value(default_profile)
    return {
        "model": selected_value,
        "resolved_model": str(profile.get("model") or "").strip(),
        "profile_id": str(profile.get("id") or "").strip(),
        "default_model": default_value,
        "default_resolved_model": str(default_profile.get("model") or "").strip(),
        "default_profile_id": str(default_profile.get("id") or "").strip(),
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

    default_profile = _resolve_lm_profile(kind="vlm")
    default_model = _lm_profile_selector_value(default_profile)
    agent_default_profile = _resolve_lm_profile(kind="agent")
    agent_default_model = _lm_profile_selector_value(agent_default_profile)
    fallback_models: List[str] = []
    profiles = [
        _resolve_lm_profile(profile_id=profile_id)
        for profile_id in _configured_lm_profiles()
    ]
    for profile in profiles:
        for candidate in (
            _lm_profile_selector_value(profile),
            str(profile.get("model") or "").strip(),
        ):
            if candidate and candidate not in fallback_models:
                fallback_models.append(candidate)
    agent_selector = str(_agent_runtime_model_override or "").strip()
    if agent_selector and agent_selector not in fallback_models:
        fallback_models.append(agent_selector)

    payload: Dict[str, Any] = {
        "models": fallback_models,
        "default_model": default_model,
        "default_profile_id": str(default_profile.get("id") or "").strip(),
        "agent_default_model": agent_default_model,
        "agent_default_profile_id": str(agent_default_profile.get("id") or "").strip(),
        "offline_default_model": agent_default_model,
        "offline_default_profile_id": str(agent_default_profile.get("id") or "").strip(),
        "profiles": [_public_lm_profile(profile) for profile in profiles],
        "auto_model_selector": LM_AUTO_BALANCE_SELECTOR,
        "auto_model_label": LM_AUTO_BALANCE_LABEL,
        "vlm_balancer": {
            "enabled": _vlm_balancer_enabled(),
            "profile_ids": _configured_vlm_balancer_profile_ids(),
        },
        "profile_errors": {},
        "source": "fallback",
        "error": None,
        "fetched_at": time.time(),
    }

    profile_errors: Dict[str, str] = {}
    fetched_any = False
    try:
        model_ids: List[str] = []
        for candidate in fallback_models:
            if candidate and candidate not in model_ids:
                model_ids.append(candidate)
        available_by_profile: Dict[str, List[str]] = {}
        for profile in profiles:
            profile_id = str(profile.get("id") or "").strip()
            base_url = str(profile.get("base_url") or "").rstrip("/")
            if not base_url:
                profile_errors[profile_id] = "base URL is not configured"
                continue
            headers = {"Content-Type": "application/json"}
            if profile.get("api_key"):
                headers["Authorization"] = f"Bearer {profile['api_key']}"
            timeout_value = float(profile.get("timeout") or config.LM_TIMEOUT or 120)
            timeout = (3.05, min(10.0, max(5.0, timeout_value)))
            try:
                response = requests.get(
                    f"{base_url}/models",
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                raw = response.json()
                items = raw.get("data") if isinstance(raw, Mapping) else None
                profile_models: List[str] = []
                if isinstance(items, Sequence):
                    for item in items:
                        if not isinstance(item, Mapping):
                            continue
                        model_id = str(item.get("id") or item.get("model") or "").strip()
                        if not model_id:
                            continue
                        if model_id not in profile_models:
                            profile_models.append(model_id)
                        if model_id not in model_ids:
                            model_ids.append(model_id)
                if profile_models:
                    available_by_profile[profile_id] = profile_models
                    fetched_any = True
            except Exception as exc:
                profile_errors[profile_id] = str(exc)
        if available_by_profile:
            by_id = {
                str(profile.get("id") or ""): dict(profile)
                for profile in payload["profiles"]
                if isinstance(profile, Mapping)
            }
            for profile_id, models in available_by_profile.items():
                if profile_id in by_id:
                    by_id[profile_id]["available_models"] = models
            payload["profiles"] = list(by_id.values())
        payload.update({
            "models": model_ids,
            "source": "lm_profiles" if fetched_any else "fallback",
            "profile_errors": profile_errors,
            "error": None if not profile_errors else "; ".join(
                f"{profile_id}: {error}"
                for profile_id, error in profile_errors.items()
            ),
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
ARCHIVE_SOURCE_FILTERS = {"probe", "vlm_summary", "vlm_alert"}
ARCHIVE_SOURCE_ALIASES = {
    "probe": "probe",
    "probes_run": "probe",
    "probes_query": "probe",
    "probe_daemon": "probe",
    "detection": "probe",
    "detections": "probe",
    "vlm_summary": "vlm_summary",
    "video_description": "vlm_summary",
    "video_descriptions": "vlm_summary",
    "vlm_alert": "vlm_alert",
    "alert": "vlm_alert",
    "alerts": "vlm_alert",
}


def _normalize_archive_source_filter(value: Any) -> Optional[str]:
    source = str(value or "").strip().lower()
    source = ARCHIVE_SOURCE_ALIASES.get(source, source)
    return source if source in ARCHIVE_SOURCE_FILTERS else None


def _archive_source_label(value: Any) -> str:
    source = _normalize_archive_source_filter(value)
    if source == "probe":
        return "Probe hit"
    if source == "vlm_summary":
        return "Video description"
    if source == "vlm_alert":
        return "VLM alert"
    return "Archive frame"


def _archive_item_type(value: Any) -> str:
    source = _normalize_archive_source_filter(value)
    if source == "probe":
        return "probe_detection"
    if source == "vlm_summary":
        return "video_description_frame"
    if source == "vlm_alert":
        return "video_description_alert"
    return "archive_frame"


class _DetectionClipShardCache:
    def __init__(self, store: Any) -> None:
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

        index = _get_faiss().IndexFlatIP(int(vectors.shape[1]))
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
    raw_value = _strip_image_data_url_prefix(thumbnail_b64)
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


def _parse_luxriot_capture_interval_sec(data: Mapping[str, Any]) -> Optional[float]:
    interval_keys = (
        "interval_sec",
        "snapshot_interval_sec",
        "sample_interval_sec",
        "capture_interval_sec",
    )
    interval_present = False
    interval_sec: Optional[float] = None
    for key in interval_keys:
        if key not in data:
            continue
        interval_present = True
        interval_sec = _to_optional_float(data.get(key))
        break
    if interval_present and interval_sec is None:
        raise ValueError("Provide a valid positive interval_sec")

    fps_keys = ("fps", "target_fps", "sample_fps")
    fps_present = False
    fps: Optional[float] = None
    for key in fps_keys:
        if key not in data:
            continue
        fps_present = True
        fps = _to_optional_float(data.get(key))
        break
    if interval_sec is None and fps_present:
        if fps is None or fps <= 0 or not math.isfinite(fps):
            raise ValueError("Provide a valid positive fps")
        interval_sec = 1.0 / fps

    if interval_sec is None:
        return None
    if interval_sec <= 0 or not math.isfinite(interval_sec):
        raise ValueError("Provide a valid positive interval_sec")
    return max(0.2, min(300.0, float(interval_sec)))


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
_archive_retention_lock = threading.RLock()
_archive_retention_last_run = 0.0
_archive_retention_last_result: Dict[str, Any] = {}


def _delete_retained_archive_images(image_paths: Sequence[Any]) -> Dict[str, Any]:
    deleted = 0
    skipped = 0
    errors = 0
    root = detection_archive.root
    seen: Set[str] = set()
    for raw_path in image_paths:
        text = str(raw_path or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        try:
            path = Path(text).expanduser()
            if not path.is_absolute():
                path = (root / path).resolve()
            else:
                path = path.resolve()
            if not _path_within(path, root):
                skipped += 1
                continue
            if path.exists() and path.is_file():
                path.unlink()
                deleted += 1
            else:
                skipped += 1
        except Exception:
            errors += 1
    return {
        "files_deleted": deleted,
        "files_skipped": skipped,
        "file_delete_errors": errors,
    }


def _apply_archive_retention(*, force: bool = False) -> Dict[str, Any]:
    global _archive_retention_last_run, _archive_retention_last_result
    if not bool(getattr(config, "ARCHIVE_RETENTION_ENABLED", True)):
        return {"ok": True, "status": "disabled"}
    prune_fn = getattr(detections_store, "apply_retention", None)
    if not callable(prune_fn):
        return {"ok": True, "status": "unsupported", "backend": getattr(detections_store, "backend", "unknown")}
    now = time.time()
    interval = float(getattr(config, "ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC", 3600.0))
    with _archive_retention_lock:
        if (
            not force
            and _archive_retention_last_run > 0
            and now - _archive_retention_last_run < interval
        ):
            cached = dict(_archive_retention_last_result)
            cached["status"] = "cached"
            cached["next_run_in_sec"] = max(0.0, interval - (now - _archive_retention_last_run))
            return cached
        try:
            result = dict(
                prune_fn(
                    row_retention_days=float(getattr(config, "ARCHIVE_ROW_RETENTION_DAYS", 90.0)),
                    thumbnail_retention_days=float(getattr(config, "ARCHIVE_THUMBNAIL_RETENTION_DAYS", 14.0)),
                    max_records=int(getattr(config, "ARCHIVE_MAX_RECORDS", 5000000)),
                    batch_size=int(getattr(config, "ARCHIVE_RETENTION_BATCH_SIZE", 5000)),
                )
            )
            deleted_paths = result.pop("deleted_image_paths", [])
            result.update(_delete_retained_archive_images(deleted_paths))
            result["status"] = "applied"
            _archive_retention_last_run = now
            _archive_retention_last_result = dict(result)
            return result
        except Exception as exc:
            result = {
                "ok": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _archive_retention_last_run = now
            _archive_retention_last_result = dict(result)
            return result


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
    if _auth_enabled() and not bool(probe_like.get("bookmark_authorized", False)):
        return False, {"reason": "bookmark_not_authorized", "source": source}

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
    origin_source = str(source or "probe").strip().lower() or "probe"
    archive_source = _normalize_archive_source_filter(origin_source) or "probe"
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
            source=archive_source,
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
            "source": archive_source,
            "origin": origin_source,
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
                "dedupe_key": f"{probe_id}:{archive_source}:{origin_source}:{ts_ms}:{pos_score:.4f}:{neg_score:.4f}:{margin:.4f}",
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
                "source": archive_source,
                "payload": payload,
            }
        )
    if not records:
        return 0
    try:
        inserted = detections_store.add_detections(records)
        _apply_archive_retention()
        return inserted
    except Exception as exc:
        print(f"Detections store write failed for {probe_id}: {exc}")
        return 0


_VLM_ARCHIVE_SEVERITY_ORDER = ("critical", "high", "normal", "low", "info")


def _vlm_archive_alert_counts(raw_counts: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if isinstance(raw_counts, Mapping):
        for raw_severity, raw_count in raw_counts.items():
            severity = str(raw_severity or "").strip().lower()
            if severity not in _VLM_ARCHIVE_SEVERITY_ORDER:
                severity = "normal"
            count = _to_int(raw_count, 0)
            if count > 0:
                counts[severity] = counts.get(severity, 0) + count
    return {
        severity: int(counts[severity])
        for severity in _VLM_ARCHIVE_SEVERITY_ORDER
        if counts.get(severity, 0) > 0
    }


def _vlm_archive_top_severity(alert_counts: Mapping[str, int]) -> str:
    for severity in _VLM_ARCHIVE_SEVERITY_ORDER:
        if int(alert_counts.get(severity, 0) or 0) > 0:
            return severity
    return "normal"


def _vlm_archive_alert_events(raw_events: Any, channel_id: int) -> List[Dict[str, Any]]:
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes, bytearray)):
        return []
    events: List[Dict[str, Any]] = []
    for raw_event in raw_events[:32]:
        if not isinstance(raw_event, Mapping):
            continue
        title = str(raw_event.get("title") or "Event").strip()[:120] or "Event"
        description = str(raw_event.get("description") or "").strip()[:300]
        severity = str(raw_event.get("severity") or "normal").strip().lower()
        if severity not in _VLM_ARCHIVE_SEVERITY_ORDER:
            severity = "normal"
        state = str(raw_event.get("state") or "new").strip().lower()[:20] or "new"
        timestamp_ms = _to_optional_int(raw_event.get("timestamp_ms"))
        event: Dict[str, Any] = {
            "title": title,
            "description": description,
            "severity": severity,
            "state": state,
            "channel_id": channel_id,
        }
        if timestamp_ms is not None:
            event["timestamp_ms"] = int(max(0, timestamp_ms))
        status = str(raw_event.get("delivery_status") or "").strip().lower()
        if status:
            event["delivery_status"] = status[:40]
        error = str(raw_event.get("error") or "").strip()
        if error:
            event["error"] = error[:240]
        events.append(event)
    return events


def _vlm_archive_excerpt(value: Any, limit: int) -> Tuple[str, bool]:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip(), True


def _vlm_summary_frame_records(entry: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], int, int]:
    raw_frames = entry.get("archive_frames")
    if (
        not isinstance(raw_frames, Sequence)
        or isinstance(raw_frames, (str, bytes, bytearray))
    ):
        return [], 0, 0

    channel_id = _to_int(entry.get("channel_id"), 0)
    if channel_id < 1:
        return [], 0, 0

    created_at = _to_optional_float(entry.get("created_at")) or time.time()
    created_ms = int(created_at * 1000.0)
    batch_start_ms = _to_int(entry.get("batch_start_ms"), created_ms)
    batch_end_ms = _to_int(entry.get("batch_end_ms"), batch_start_ms)
    run_id = str(entry.get("run_id") or "").strip() or "manual"
    summary_excerpt, summary_truncated = _vlm_archive_excerpt(entry.get("summary"), 4000)
    prompt_excerpt, prompt_truncated = _vlm_archive_excerpt(entry.get("prompt"), 1000)
    alert_counts = _vlm_archive_alert_counts(entry.get("alert_counts"))
    alert_total = _to_int(entry.get("alert_total"), sum(alert_counts.values()))
    bookmarks_sent = _to_int(entry.get("bookmarks_sent"), 0)
    alert_events = _vlm_archive_alert_events(entry.get("alert_events"), channel_id)
    frame_count = _to_int(entry.get("frame_count"), 0)
    batch_size = _to_int(entry.get("batch_size"), 0)

    base_payload = {
        "run_id": run_id,
        "batch_start_ms": batch_start_ms,
        "batch_end_ms": batch_end_ms,
        "frame_count": frame_count,
        "batch_size": batch_size,
        "duration_sec": _to_float(entry.get("duration_sec"), 0.0),
        "summary": summary_excerpt,
        "summary_truncated": summary_truncated,
        "prompt": prompt_excerpt,
        "prompt_truncated": prompt_truncated,
        "alert_counts": alert_counts,
        "alert_total": alert_total,
        "bookmarks_sent": bookmarks_sent,
        "state_observations": list(entry.get("state_observations") or [])[:64]
        if isinstance(entry.get("state_observations"), Sequence)
        and not isinstance(entry.get("state_observations"), (str, bytes, bytearray))
        else [],
        "state_transition_events": list(entry.get("state_transition_events") or [])[:32]
        if isinstance(entry.get("state_transition_events"), Sequence)
        and not isinstance(entry.get("state_transition_events"), (str, bytes, bytearray))
        else [],
        "state_transition_total": _to_int(entry.get("state_transition_total"), 0),
    }

    records: List[Dict[str, Any]] = []
    valid_frames: List[Dict[str, Any]] = []
    for fallback_index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            continue
        thumbnail_b64 = str(raw_frame.get("thumbnail") or raw_frame.get("thumbnail_b64") or "").strip()
        if not thumbnail_b64:
            continue
        frame_index = _to_int(raw_frame.get("frame_index"), fallback_index)
        timestamp_ms = _to_int(raw_frame.get("timestamp_ms"), batch_start_ms)
        timestamp_ms = max(0, timestamp_ms)
        anchor_role = str(raw_frame.get("anchor_role") or "sample").strip().lower() or "sample"
        width = _to_optional_int(raw_frame.get("width"))
        height = _to_optional_int(raw_frame.get("height"))
        frame_payload = {
            **base_payload,
            "source": "vlm_summary",
            "anchor_role": anchor_role,
            "frame_index": frame_index,
            "frame_timestamp_ms": timestamp_ms,
            "captured_at": _to_optional_float(raw_frame.get("captured_at")),
            "width": width,
            "height": height,
        }
        clip_vec = _embed_thumbnail_b64(thumbnail_b64, "clip")
        records.append(
            {
                "dedupe_key": (
                    f"vlm_summary:{channel_id}:{run_id}:{batch_start_ms}:{batch_end_ms}:"
                    f"{frame_index}:{timestamp_ms}:{anchor_role}"
                ),
                "timestamp_ms": timestamp_ms,
                "probe_id": f"vlm_summary:{channel_id}",
                "probe_name": f"VLM summary ch {channel_id}",
                "channel_id": channel_id,
                "severity": "info",
                "bookmark_enabled": False,
                "bookmark_sent": False,
                "pos_score": 0.0,
                "neg_score": 0.0,
                "margin": 0.0,
                "thumbnail_b64": thumbnail_b64,
                "clip_vec": clip_vec,
                "source": "vlm_summary",
                "payload": frame_payload,
            }
        )
        valid_frames.append(
            {
                "timestamp_ms": timestamp_ms,
                "frame_index": frame_index,
                "anchor_role": anchor_role,
                "thumbnail_b64": thumbnail_b64,
                "clip_vec": clip_vec,
                "payload": frame_payload,
            }
        )

    summary_count = len(records)
    alert_count = 0
    if valid_frames and (alert_total > 0 or bookmarks_sent > 0 or alert_events):
        events_for_archive = list(alert_events)
        if not events_for_archive:
            events_for_archive = [
                {
                    "title": "VLM alert batch",
                    "description": "",
                    "severity": _vlm_archive_top_severity(alert_counts),
                    "state": "new",
                    "channel_id": channel_id,
                    "timestamp_ms": int(valid_frames[-1]["timestamp_ms"]),
                    "delivery_status": "aggregate",
                }
            ]
        for event_index, alert_event in enumerate(events_for_archive[:32]):
            event_ts = _to_int(alert_event.get("timestamp_ms"), int(valid_frames[-1]["timestamp_ms"]))
            anchor = min(
                valid_frames,
                key=lambda frame: abs(int(frame.get("timestamp_ms") or 0) - int(event_ts)),
            )
            severity = str(alert_event.get("severity") or "normal").strip().lower()
            if severity not in _VLM_ARCHIVE_SEVERITY_ORDER:
                severity = "normal"
            event_payload = {
                "title": str(alert_event.get("title") or "Event"),
                "description": str(alert_event.get("description") or ""),
                "severity": severity,
                "state": str(alert_event.get("state") or "new"),
                "timestamp_ms": event_ts,
                "delivery_status": str(alert_event.get("delivery_status") or ""),
            }
            if alert_event.get("error"):
                event_payload["error"] = str(alert_event.get("error") or "")[:240]
            event_hash = hashlib.sha1(
                json.dumps(event_payload, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="ignore")
            ).hexdigest()[:12]
            alert_payload = {
                **base_payload,
                "source": "vlm_alert",
                "severity": severity,
                "alert_event": event_payload,
                "alert_event_index": event_index,
                "anchor_role": "alert_anchor",
                "anchor_frame_index": anchor["frame_index"],
                "anchor_frame_timestamp_ms": anchor["timestamp_ms"],
                "anchor_source_role": anchor["anchor_role"],
            }
            records.append(
                {
                    "dedupe_key": (
                        f"vlm_alert:{channel_id}:{run_id}:{batch_start_ms}:{batch_end_ms}:"
                        f"{anchor['timestamp_ms']}:{severity}:{event_index}:{event_hash}"
                    ),
                    "timestamp_ms": int(anchor["timestamp_ms"]),
                    "probe_id": f"vlm_alert:{channel_id}",
                    "probe_name": f"VLM alert ch {channel_id}: {event_payload['title'][:64]}",
                    "channel_id": channel_id,
                    "severity": severity,
                    "bookmark_enabled": False,
                    "bookmark_sent": bookmarks_sent > 0,
                    "pos_score": 0.0,
                    "neg_score": 0.0,
                    "margin": 0.0,
                    "thumbnail_b64": anchor["thumbnail_b64"],
                    "clip_vec": anchor["clip_vec"],
                    "source": "vlm_alert",
                    "payload": alert_payload,
                }
            )
        alert_count = len(events_for_archive[:32])
    return records, summary_count, alert_count


def _store_vlm_summary_archive_frames(entry: Mapping[str, Any]) -> Dict[str, Any]:
    records, summary_count, alert_count = _vlm_summary_frame_records(entry)
    if not records:
        return {
            "attempted": 0,
            "inserted": 0,
            "summary_frames": 0,
            "alert_frames": 0,
        }
    try:
        inserted = detections_store.add_detections(records)
        _apply_archive_retention()
        return {
            "attempted": len(records),
            "inserted": int(inserted),
            "summary_frames": summary_count,
            "alert_frames": alert_count,
        }
    except Exception as exc:
        print(f"VLM summary archive write failed: {exc}")
        return {
            "attempted": len(records),
            "inserted": 0,
            "summary_frames": summary_count,
            "alert_frames": alert_count,
            "error": str(exc)[:240] or exc.__class__.__name__,
        }


luxriot_manager.set_summary_archive_callback(_store_vlm_summary_archive_frames)


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

        _get_faiss().write_index(index, str(embed_dir / 'index.faiss'))

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

    if not _index_metadata_compatible(target, meta):
        print(f"Index metadata for {target} does not match current embedding model; rebuild index.")
        return None, None, None, meta

    try:
        index = _get_faiss().read_index(str(embed_dir / 'index.faiss'))

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
        index = _get_faiss().read_index(str(index_path))
        if index.d != embeddings.shape[1]:
            raise ValueError(
                f"Segment embedding dimension mismatch: existing index expects {index.d}, got {embeddings.shape[1]}"
            )
        with open(metadata_path, 'rb') as fh:
            existing_meta: List[Dict[str, Any]] = pickle.load(fh)
    else:
        index = _get_faiss().IndexFlatIP(embeddings.shape[1])
        existing_meta = []

    _faiss_add_vectors(index, embeddings)
    existing_meta.extend(segment_metadata)

    _get_faiss().write_index(index, str(index_path))
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
        index = _get_faiss().read_index(str(index_path))
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
    source = _normalize_archive_source_filter(payload.get("source"))

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
        "source": source,
        "since_ms": since_ms,
        "until_ms": until_ms,
    }


def _backfill_clip_vectors_for_filters(
    probe_id: Optional[str],
    channel_id: Optional[int],
    source: Optional[str],
    since_ms: Optional[int],
    until_ms: Optional[int],
    *,
    expected_dim: Optional[int] = None,
    max_backfill: int = 2000,
) -> int:
    detections, _ = detections_store.list_detections(
        probe_id=probe_id,
        channel_id=channel_id,
        source=source,
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
            vec_rows = detections_store.fetch_detections_by_ids(
                remaining,
                include_vectors=True,
                include_thumbnail=False,
            )
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

    origin = str(payload_obj.get("origin") or payload_obj.get("source") or "").strip()
    result: Dict[str, Any] = {
        "path": image_path,
        "filename": f"{probe_label} · {ts_label}",
        "similarity": float(score),
        "thumbnail": item.get("thumbnail") or "",
        "metadata": {
            "mtime": ts_ms,
            "detection_id": item.get("id"),
            "source": item.get("source"),
            "origin": origin,
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
        "source_label": _archive_source_label(item.get("source")),
        "archive_item_type": _archive_item_type(item.get("source")),
        "origin": origin,
        "shard_key": item.get("shard_key"),
        "search_mode": mode,
        "dino_fallback": bool(dino_fallback),
    }
    if payload_obj:
        result["payload"] = payload_obj
        summary_excerpt = str(payload_obj.get("summary") or "").strip()
        if summary_excerpt:
            result["summary"] = summary_excerpt
        for key in (
            "run_id",
            "batch_start_ms",
            "batch_end_ms",
            "frame_timestamp_ms",
            "anchor_frame_timestamp_ms",
            "frame_index",
            "anchor_frame_index",
            "anchor_role",
            "anchor_source_role",
            "summary_truncated",
        ):
            if key in payload_obj:
                result[key] = payload_obj.get(key)
    if mode in {"fusion", "dino"}:
        result["fusion"] = {
            "clip_similarity": float(clip_score),
            "dino_similarity": float(dino_score if dino_score is not None else clip_score),
            "alpha": float(alpha),
            "dino_fallback": bool(dino_fallback),
        }
    return result


def _build_detection_search_coverage(
    *,
    candidates: Sequence[Dict[str, Any]],
    total_candidates: Optional[int],
    candidate_limit: int,
    since_ms: Optional[int],
    until_ms: Optional[int],
    limit: int,
    source: Optional[str],
    channel_id: Optional[int],
) -> Dict[str, Any]:
    scanned = len(candidates)
    total = max(int(total_candidates), scanned) if total_candidates is not None else scanned
    timestamps = [
        int(item.get("timestamp_ms") or 0)
        for item in candidates
        if _to_optional_int(item.get("timestamp_ms")) is not None
    ]
    newest_ms = max(timestamps) if timestamps else None
    oldest_ms = min(timestamps) if timestamps else None
    truncated = total > scanned
    note = "Search ranked the full candidate set for the requested filters."
    if truncated:
        note = (
            "Search ranked a limited newest-first candidate window; older matching archive rows "
            "may exist outside this search pass."
        )
    return {
        "candidate_limit": int(candidate_limit),
        "scanned_candidates": int(scanned),
        "total_candidates": int(total),
        "truncated": bool(truncated),
        "result_limit": int(limit),
        "source": source,
        "channel_id": channel_id,
        "requested_since_ms": since_ms,
        "requested_until_ms": until_ms,
        "scanned_oldest_ms": oldest_ms,
        "scanned_newest_ms": newest_ms,
        "must_state_coverage": bool(truncated),
        "note": note,
    }


def _search_detections_archive(
    *,
    clip_query_vec: np.ndarray,
    dino_query_vec: Optional[np.ndarray],
    mode: str,
    probe_id: Optional[str],
    channel_id: Optional[int],
    source: Optional[str],
    since_ms: Optional[int],
    until_ms: Optional[int],
    limit: int,
    sort_by: str,
    candidate_limit: int,
    include_coverage: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    limit = max(1, min(config.MAX_RESULTS, int(limit or config.DEFAULT_RESULTS)))
    candidate_limit = max(limit, min(DETECTIONS_SEARCH_MAX_CANDIDATES, int(candidate_limit or 20000)))
    clip_dim = int(clip_query_vec.shape[0]) if clip_query_vec.ndim == 1 else None
    total_candidates: Optional[int] = None
    try:
        total_candidates = detections_store.count_vector_candidates(
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            only_with_clip=True,
        )
    except AttributeError:
        total_candidates = None

    candidates = detections_store.list_vector_candidates(
        probe_id=probe_id,
        channel_id=channel_id,
        source=source,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=candidate_limit,
        only_with_clip=True,
        include_vectors=False,
        include_thumbnail=False,
    )
    if not candidates:
        updated = _backfill_clip_vectors_for_filters(
            probe_id,
            channel_id,
            source,
            since_ms,
            until_ms,
            expected_dim=clip_dim,
            max_backfill=min(candidate_limit, 2000),
        )
        if updated > 0:
            candidates = detections_store.list_vector_candidates(
                probe_id=probe_id,
                channel_id=channel_id,
                source=source,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=candidate_limit,
                only_with_clip=True,
                include_vectors=False,
                include_thumbnail=False,
            )
    if not candidates:
        coverage = _build_detection_search_coverage(
            candidates=[],
            total_candidates=total_candidates,
            candidate_limit=candidate_limit,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            source=source,
            channel_id=channel_id,
        )
        return ([], coverage) if include_coverage else []

    clip_hits, candidate_map = _search_detection_clip_shards(candidates, clip_query_vec, limit)
    if not clip_hits:
        updated = _backfill_clip_vectors_for_filters(
            probe_id,
            channel_id,
            source,
            since_ms,
            until_ms,
            expected_dim=clip_dim,
            max_backfill=min(candidate_limit, 2000),
        )
        if updated > 0:
            candidates = detections_store.list_vector_candidates(
                probe_id=probe_id,
                channel_id=channel_id,
                source=source,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=candidate_limit,
                only_with_clip=True,
                include_vectors=False,
                include_thumbnail=False,
            )
            clip_hits, candidate_map = _search_detection_clip_shards(candidates, clip_query_vec, limit)
    if not clip_hits:
        coverage = _build_detection_search_coverage(
            candidates=candidates,
            total_candidates=total_candidates,
            candidate_limit=candidate_limit,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            source=source,
            channel_id=channel_id,
        )
        return ([], coverage) if include_coverage else []

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
        dino_dim = int(dino_query_vec.shape[0]) if dino_query_vec.ndim == 1 else 0
        for det_id in pool_ids:
            vec = dino_vectors.get(det_id)
            if vec is None:
                continue
            if dino_dim <= 0 or vec.ndim != 1 or int(vec.shape[0]) != dino_dim:
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

    top_scored = scored[:limit]
    hydrated_by_id: Dict[int, Dict[str, Any]] = {}
    try:
        hydrated_rows = detections_store.fetch_detections_by_ids(
            [det_id for det_id, *_rest in top_scored],
            include_vectors=False,
            include_thumbnail=True,
        )
        hydrated_by_id = {
            int(row["id"]): row
            for row in hydrated_rows
            if isinstance(row, Mapping) and _to_optional_int(row.get("id")) is not None
        }
    except Exception:
        hydrated_by_id = {}

    results: List[Dict[str, Any]] = []
    for det_id, final_score, clip_score, dino_score, dino_fallback in top_scored:
        item = hydrated_by_id.get(det_id) or candidate_map.get(det_id)
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
    coverage = _build_detection_search_coverage(
        candidates=candidates,
        total_candidates=total_candidates,
        candidate_limit=candidate_limit,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=limit,
        source=source,
        channel_id=channel_id,
    )
    return (results, coverage) if include_coverage else results


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
    is_multipart = bool(request.files)
    data = request.form if is_multipart else _json_body()
    upload_obj = request.files.get('video') or request.files.get('file')
    uploaded_temp_path: Optional[Path] = None
    uploaded_original_name = str(getattr(upload_obj, "filename", "") or "").strip()
    diagnostics: Dict[str, Any] = {
        "request_id": getattr(g, "request_id", ""),
        "multipart": is_multipart,
        "upload": bool(uploaded_original_name),
    }

    if upload_obj is not None and uploaded_original_name:
        diagnostics["filename"] = Path(uploaded_original_name).name
        diagnostics["suffix"] = Path(uploaded_original_name).suffix.lower()
        try:
            uploaded_temp_path = _save_upload_to_temp(
                upload_obj,
                allowed_suffixes=SUPPORTED_VIDEO_EXTENSIONS,
                prefix="eva-video-",
            )
            diagnostics["upload_bytes"] = uploaded_temp_path.stat().st_size
        except ValueError as exc:
            diagnostics["stage"] = "upload"
            diagnostics["reason"] = type(exc).__name__
            return jsonify({'error': str(exc), 'diagnostics': diagnostics}), 400
        video_obj = uploaded_temp_path
        video_path = uploaded_original_name
    else:
        video_path = (data.get('video') or '').strip()
        diagnostics["source"] = "server_path"
        if not video_path:
            diagnostics["stage"] = "input"
            return jsonify({'error': 'Provide a video path or upload a video file.', 'diagnostics': diagnostics}), 400
        video_obj = Path(video_path).expanduser().resolve()
        if not video_obj.exists() or not video_obj.is_file():
            diagnostics["stage"] = "path"
            return jsonify({'error': 'Video file not found', 'diagnostics': diagnostics}), 400
        if video_obj.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            diagnostics["stage"] = "suffix"
            diagnostics["suffix"] = video_obj.suffix.lower()
            return jsonify({'error': 'Unsupported video file type', 'diagnostics': diagnostics}), 400
        if config.ALLOWED_ROOTS:
            allowed_roots = [Path(item).expanduser().resolve() for item in config.ALLOWED_ROOTS]
            if not any(_path_within(video_obj, root) for root in allowed_roots):
                diagnostics["stage"] = "allowed_roots"
                diagnostics["allowed_roots_count"] = len(allowed_roots)
                return jsonify({'error': 'Video path is outside configured allowed roots', 'diagnostics': diagnostics}), 400

    max_frames = data.get('frame_count') or config.LM_VIDEO_DEFAULT_FRAMES
    try:
        max_frames_int = int(max_frames)
    except (TypeError, ValueError):
        max_frames_int = config.LM_VIDEO_DEFAULT_FRAMES
    if max_frames_int < 1:
        max_frames_int = 1
    max_frames_int = min(max_frames_int, config.LM_VIDEO_MAX_FRAMES)
    diagnostics["frame_count_requested"] = max_frames_int

    sample_fps_raw = data.get('sample_fps')
    try:
        sample_fps_val = float(sample_fps_raw) if sample_fps_raw is not None else None
        if sample_fps_val is not None and sample_fps_val <= 0:
            sample_fps_val = None
    except (TypeError, ValueError):
        sample_fps_val = None
    if sample_fps_val is not None:
        diagnostics["sample_fps"] = sample_fps_val

    user_prompt = data.get('prompt') or ''
    model_hint = (data.get('model') or '').strip()
    profile_hint = (
        str(data.get('profile_id') or data.get('profileId') or '').strip()
        or None
    )
    effective_model_hint = model_hint or None
    model_selection: Dict[str, Any] = {
        "mode": "manual" if (model_hint or profile_hint) else "default",
        "requested": model_hint or profile_hint or None,
        "assigned_profile_id": profile_hint,
        "balancer_enabled": _vlm_balancer_enabled(),
    }
    if profile_hint and _is_auto_lm_selector(effective_model_hint):
        effective_model_hint = None
    if not profile_hint:
        assignment_key = _offline_vlm_assignment_key(
            "video",
            uploaded_original_name or video_path or str(video_obj),
        )
        effective_model_hint, model_selection = _resolve_offline_lm_model_hint(
            model_hint or None,
            assignment_key=assignment_key,
        )

    try:
        lm_profile = _resolve_lm_profile(
            profile_id=profile_hint,
            model_override=effective_model_hint,
            kind="vlm",
        )
        diagnostics["profile_id"] = str(lm_profile.get('id') or '')
        diagnostics["model"] = str(lm_profile.get('model') or '')
        diagnostics["model_selection"] = model_selection.get("mode")
        diagnostics["assigned_profile_id"] = model_selection.get("assigned_profile_id")
        frames, fps, duration = _sample_video_frames(
            str(video_obj),
            max_frames=max_frames_int,
            sample_fps=sample_fps_val,
            max_edge=config.LM_VIDEO_MAX_EDGE,
        )
        diagnostics["fps"] = round(float(fps or 0.0), 3)
        if duration is not None:
            diagnostics["duration_sec"] = round(float(duration), 3)
        diagnostics["frames_extracted"] = len(frames)
        if not frames:
            diagnostics["stage"] = "frame_sampling"
            audit_error = _write_completion_audit_or_error(
                action="lm.video_understanding.completed",
                result="failure",
                target_type="video",
                target_id=_audit_fingerprint(video_obj),
                details={
                    "reason": "no_frames",
                    "frame_count_requested": max_frames_int,
                    "sample_fps_supplied": sample_fps_raw is not None,
                },
            )
            if audit_error is not None:
                return audit_error
            return jsonify({'error': 'No frames could be extracted from the video.', 'diagnostics': diagnostics}), 400
        messages = _build_video_messages(video_path or str(video_obj), frames, user_prompt)
        summary = _call_video_understanding(
            messages,
            model_override=effective_model_hint,
            profile_id=profile_hint,
        )
        audit_error = _write_completion_audit_or_error(
            action="lm.video_understanding.completed",
            result="success",
            target_type="video",
            target_id=_audit_fingerprint(video_obj),
            details={
                "frames": len(frames),
                "frame_count_requested": max_frames_int,
                "sample_fps_supplied": sample_fps_raw is not None,
                "prompt_supplied": bool(str(user_prompt).strip()),
                "profile_id": str(lm_profile.get('id') or ''),
                "model": str(lm_profile.get('model') or ''),
                "model_selection": model_selection.get("mode"),
                "assigned_profile_id": model_selection.get("assigned_profile_id"),
                "balancer_enabled": model_selection.get("balancer_enabled"),
                "balancer_profile_count": model_selection.get("profile_count"),
            },
        )
        if audit_error is not None:
            return audit_error
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
                'model': str(lm_profile.get('model') or ''),
                'profile_id': str(lm_profile.get('id') or ''),
                'model_selector': _lm_profile_selector_value(lm_profile),
                'model_selection': model_selection.get("mode"),
                'assigned_profile_id': model_selection.get("assigned_profile_id"),
                'uploaded': bool(uploaded_temp_path),
                'filename': uploaded_original_name or Path(video_obj).name,
                'diagnostics': diagnostics,
            }
        )
    except Exception as exc:
        diagnostics["stage"] = "inference"
        diagnostics["reason"] = type(exc).__name__
        audit_error = _write_completion_audit_or_error(
            action="lm.video_understanding.completed",
            result="failure",
            target_type="video",
            target_id=_audit_fingerprint(video_obj if 'video_obj' in locals() else video_path),
            details={
                "reason": type(exc).__name__,
                "model_supplied": bool(model_hint),
                "profile_supplied": bool(profile_hint),
            },
        )
        if audit_error is not None:
            return audit_error
        app.logger.exception(
            "Video understanding failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'video_understanding_failed', 'diagnostics': diagnostics}), 500
    finally:
        _cleanup_temp_upload(uploaded_temp_path)


@app.route('/describe_image', methods=['POST'])
def describe_image():
    is_multipart = bool(request.files)
    data = request.form if is_multipart else _json_body()
    folder_raw = data.get('folder')
    upload_obj = request.files.get('image') or request.files.get('file')
    uploaded_temp_path: Optional[Path] = None
    uploaded_original_name = str(getattr(upload_obj, "filename", "") or "").strip()
    image_path = (data.get('image_path') or '').strip()
    prompt = data.get('prompt') or ''
    model_hint = (data.get('model') or '').strip()
    profile_hint = (
        str(data.get('profile_id') or data.get('profileId') or '').strip()
        or None
    )
    effective_model_hint = model_hint or None
    model_selection: Dict[str, Any] = {
        "mode": "manual" if (model_hint or profile_hint) else "default",
        "requested": model_hint or profile_hint or None,
        "assigned_profile_id": profile_hint,
        "balancer_enabled": _vlm_balancer_enabled(),
    }
    if profile_hint and _is_auto_lm_selector(effective_model_hint):
        effective_model_hint = None
    if upload_obj is not None and uploaded_original_name:
        try:
            uploaded_temp_path = _save_upload_to_temp(
                upload_obj,
                allowed_suffixes=set(config.SUPPORTED_EXTENSIONS),
                prefix="eva-image-",
            )
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        image_path = uploaded_original_name
    elif not image_path:
        return jsonify({'error': 'image_path or image upload is required'}), 400
    try:
        if uploaded_temp_path is not None:
            path_obj = uploaded_temp_path
            with Image.open(path_obj) as uploaded_img:
                uploaded_img.verify()
        elif folder_raw:
            folder_path = _resolve_folder_path(folder_raw, require_index=True)
            path_obj = Path(image_path).expanduser().resolve()
            if path_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                return jsonify({'error': 'Unsupported image file type'}), 400
            if not path_obj.exists() or not path_obj.is_file():
                return jsonify({'error': 'Image not found'}), 400
            if not _path_within(path_obj, folder_path):
                return jsonify({'error': 'image_path must be inside folder'}), 400
        else:
            path_obj = detection_archive.resolve_archive_image_path(image_path)
        if not profile_hint:
            assignment_key = _offline_vlm_assignment_key(
                "image",
                uploaded_original_name or image_path or str(path_obj),
            )
            effective_model_hint, model_selection = _resolve_offline_lm_model_hint(
                model_hint or None,
                assignment_key=assignment_key,
            )
        lm_profile = _resolve_lm_profile(
            profile_id=profile_hint,
            model_override=effective_model_hint,
            kind="vlm",
        )
        messages = _build_image_messages(str(path_obj), prompt)
        summary = _call_lm_chat(
            messages,
            model_override=effective_model_hint,
            profile_id=profile_hint,
            profile_kind="vlm",
        )
        with Image.open(path_obj) as src:
            thumb = _encode_jpeg(src, max_edge=config.THUMBNAIL_SIZE[0])
        audit_error = _write_completion_audit_or_error(
            action="lm.describe_image.completed",
            result="success",
            target_type="image",
            target_id=_audit_fingerprint(path_obj),
            details={
                "folder_supplied": bool(folder_raw),
                "prompt_supplied": bool(str(prompt).strip()),
                "thumbnail_returned": bool(thumb),
                "profile_id": str(lm_profile.get('id') or ''),
                "model": str(lm_profile.get('model') or ''),
                "model_selection": model_selection.get("mode"),
                "assigned_profile_id": model_selection.get("assigned_profile_id"),
                "balancer_enabled": model_selection.get("balancer_enabled"),
                "balancer_profile_count": model_selection.get("profile_count"),
            },
        )
        if audit_error is not None:
            return audit_error
        return jsonify(
            {
                'summary': summary,
                'thumbnail': thumb,
                'model': str(lm_profile.get('model') or ''),
                'profile_id': str(lm_profile.get('id') or ''),
                'model_selector': _lm_profile_selector_value(lm_profile),
                'model_selection': model_selection.get("mode"),
                'assigned_profile_id': model_selection.get("assigned_profile_id"),
                'uploaded': bool(uploaded_temp_path),
                'filename': uploaded_original_name or Path(path_obj).name,
            }
        )
    except ValueError as exc:
        audit_error = _write_completion_audit_or_error(
            action="lm.describe_image.completed",
            result="failure",
            target_type="image",
            target_id=_audit_fingerprint(image_path),
            details={
                "reason": type(exc).__name__,
                "folder_supplied": bool(folder_raw),
                "model_supplied": bool(model_hint),
                "profile_supplied": bool(profile_hint),
            },
        )
        if audit_error is not None:
            return audit_error
        app.logger.info(
            "Describe image request rejected request_id=%s error=%s",
            getattr(g, "request_id", ""),
            exc,
        )
        return jsonify({'error': 'Invalid image request'}), 400
    except Exception as exc:
        audit_error = _write_completion_audit_or_error(
            action="lm.describe_image.completed",
            result="failure",
            target_type="image",
            target_id=_audit_fingerprint(path_obj if 'path_obj' in locals() else image_path),
            details={
                "reason": type(exc).__name__,
                "folder_supplied": bool(folder_raw),
                "model_supplied": bool(model_hint),
                "profile_supplied": bool(profile_hint),
            },
        )
        if audit_error is not None:
            return audit_error
        app.logger.exception(
            "Describe image failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'image_description_failed'}), 500
    finally:
        _cleanup_temp_upload(uploaded_temp_path)


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


@app.route('/luxriot/snapshot/<int:channel_id>/capture', methods=['POST'])
def luxriot_snapshot_capture(channel_id: int):
    data = _json_body()
    stream_type = str(data.get('stream') or request.args.get('stream') or 'mainStream').strip() or 'mainStream'
    roi_enabled, roi_norm = _parse_probe_roi(data)
    quality = data.get('quality') or 92
    try:
        encoded, meta = luxriot_manager.capture_snapshot_base64(
            channel_id,
            stream_type=stream_type,
            roi_norm=roi_norm if roi_enabled else None,
            quality=int(quality),
        )
        return jsonify(
            {
                "success": True,
                "snapshot_b64": encoded,
                "meta": meta,
                "channel_id": channel_id,
                "filename": (
                    f"probe_snap_ch{channel_id}_"
                    f"{int(meta.get('captured_at_ms') or int(time.time() * 1000))}.jpg"
                ),
            }
        )
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


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
    requested_model_hint = (data.get('model') or '').strip() or None
    try:
        interval_sec = _parse_luxriot_capture_interval_sec(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    model_hint, model_selection = _resolve_luxriot_vlm_model_hint(
        channel_id,
        requested_model_hint,
    )
    system_prompt = (data.get('system_prompt') or '').strip() or None
    try:
        status = luxriot_manager.start_session(
            channel_id,
            batch_size=batch_size,
            prompt=prompt,
            model_hint=model_hint,
            system_prompt=system_prompt,
            interval_sec=interval_sec,
        )
        if isinstance(status, Mapping):
            status = dict(status)
            status["model_selection"] = model_selection.get("mode")
            status["assigned_profile_id"] = model_selection.get("assigned_profile_id")
        audit_error = _write_completion_audit_or_error(
            action="luxriot.capture.start.completed",
            result="success",
            target_type="luxriot_capture",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={
                "batch_size_supplied": batch_size is not None,
                "prompt_supplied": bool(str(prompt).strip()),
                "system_prompt_supplied": bool(system_prompt),
                "model_supplied": bool(requested_model_hint),
                "interval_sec_supplied": interval_sec is not None,
                "interval_sec": interval_sec,
                "model_selection": model_selection.get("mode"),
                "assigned_profile_id": model_selection.get("assigned_profile_id"),
                "balancer_enabled": model_selection.get("balancer_enabled"),
                "balancer_profile_count": model_selection.get("profile_count"),
                "session_running": bool(status.get("running"))
                if isinstance(status, Mapping)
                else None,
            },
        )
        if audit_error is not None:
            return audit_error
        return jsonify({'success': True, 'session': status})
    except Exception as exc:
        audit_error = _write_completion_audit_or_error(
            action="luxriot.capture.start.completed",
            result="failure",
            target_type="luxriot_capture",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={"reason": type(exc).__name__},
        )
        if audit_error is not None:
            return audit_error
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

    alert_policy_prompt: Optional[str] = None
    if 'alert_policy_prompt' in data:
        raw_alert_policy_prompt = data.get('alert_policy_prompt')
        alert_policy_prompt = '' if raw_alert_policy_prompt is None else str(raw_alert_policy_prompt)
    elif 'alert_prompt' in data:
        raw_alert_policy_prompt = data.get('alert_prompt')
        alert_policy_prompt = '' if raw_alert_policy_prompt is None else str(raw_alert_policy_prompt)

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
    if json_alert_prompt is not None or bookmark_enabled is not None or bookmark_cooldown_sec is not None:
        bookmark_guard = _bookmark_permission_guard_error(
            action="http.luxriot_prompt_settings.bookmark_settings",
        )
        if bookmark_guard is not None:
            return bookmark_guard

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
            alert_policy_prompt=alert_policy_prompt,
            rollup_prompts=rollup_prompt_updates,
            json_alert_prompt=json_alert_prompt,
            bookmark_enabled=bookmark_enabled,
            bookmark_cooldown_sec=bookmark_cooldown_sec,
        )
        audit_error = _write_completion_audit_or_error(
            action="luxriot.prompt_settings.update.completed",
            result="success",
            target_type="luxriot_prompt_settings",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={
                "stream_system_prompt_updated": stream_system_prompt is not None,
                "alert_policy_prompt_updated": alert_policy_prompt is not None,
                "rollup_prompts_updated": bool(rollup_prompt_updates),
                "json_alert_prompt_updated": json_alert_prompt is not None,
                "bookmark_enabled_updated": bookmark_enabled is not None,
                "bookmark_cooldown_updated": bookmark_cooldown_sec is not None,
                "rollup_levels": sorted(rollup_prompt_updates.keys())
                if isinstance(rollup_prompt_updates, Mapping)
                else [],
            },
        )
        if audit_error is not None:
            return audit_error
        return jsonify({'success': True, **settings})
    except Exception as exc:
        audit_error = _write_completion_audit_or_error(
            action="luxriot.prompt_settings.update.completed",
            result="failure",
            target_type="luxriot_prompt_settings",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={"reason": type(exc).__name__},
        )
        if audit_error is not None:
            return audit_error
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
        audit_error = _write_completion_audit_or_error(
            action="luxriot.capture.stop.completed",
            result="success",
            target_type="luxriot_capture",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={
                "session_running": bool(state.get("running"))
                if isinstance(state, Mapping)
                else None,
            },
        )
        if audit_error is not None:
            return audit_error
        return jsonify({'success': True, 'session': state})
    except Exception as exc:
        audit_error = _write_completion_audit_or_error(
            action="luxriot.capture.stop.completed",
            result="failure",
            target_type="luxriot_capture",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={"reason": type(exc).__name__},
        )
        if audit_error is not None:
            return audit_error
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
            audit_error = _write_completion_audit_or_error(
                action="luxriot.capture.flush.completed",
                result="failure",
                target_type="luxriot_capture",
                target_id=str(channel_id),
                channel_id=channel_id,
                details={"reason": "flush_unsuccessful"},
            )
            if audit_error is not None:
                return audit_error
            return jsonify(result), 400
        audit_error = _write_completion_audit_or_error(
            action="luxriot.capture.flush.completed",
            result="success",
            target_type="luxriot_capture",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={
                "items": len(result.get("items") or [])
                if isinstance(result, Mapping)
                else None,
            },
        )
        if audit_error is not None:
            return audit_error
        return jsonify(result)
    except Exception as exc:
        audit_error = _write_completion_audit_or_error(
            action="luxriot.capture.flush.completed",
            result="failure",
            target_type="luxriot_capture",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={"reason": type(exc).__name__},
        )
        if audit_error is not None:
            return audit_error
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
        return jsonify(
            _filter_stream_status_for_context(
                status,
                _current_auth_context(),
            )
        )
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
        audit_error = _write_completion_audit_or_error(
            action="luxriot.stream.stop.completed",
            result="success",
            target_type="luxriot_stream",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={
                "stream_type": stream_type,
                "pause_analytics": pause_analytics,
            },
        )
        if audit_error is not None:
            return audit_error
        return jsonify({
            'success': True,
            'result': result,
            'streams': _filter_stream_status_for_context(
                luxriot_manager.streams_status(),
                _current_auth_context(),
            ),
        })
    except ValueError as exc:
        audit_error = _write_completion_audit_or_error(
            action="luxriot.stream.stop.completed",
            result="failure",
            target_type="luxriot_stream",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={"reason": type(exc).__name__},
        )
        if audit_error is not None:
            return audit_error
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        audit_error = _write_completion_audit_or_error(
            action="luxriot.stream.stop.completed",
            result="failure",
            target_type="luxriot_stream",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={"reason": type(exc).__name__},
        )
        if audit_error is not None:
            return audit_error
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
        audit_error = _write_completion_audit_or_error(
            action="luxriot.stream.stop_all.completed",
            result="success",
            target_type="luxriot_streams",
            details={
                "stop_video": stop_video,
                "stop_analytics": stop_analytics,
                "pause_analytics": pause_analytics,
            },
        )
        if audit_error is not None:
            return audit_error
        return jsonify({
            'success': True,
            'result': result,
            'streams': _filter_stream_status_for_context(
                luxriot_manager.streams_status(),
                _current_auth_context(),
            ),
        })
    except Exception as exc:
        audit_error = _write_completion_audit_or_error(
            action="luxriot.stream.stop_all.completed",
            result="failure",
            target_type="luxriot_streams",
            details={"reason": type(exc).__name__},
        )
        if audit_error is not None:
            return audit_error
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
        audit_error = _write_completion_audit_or_error(
            action="luxriot.bookmark.create.completed",
            result="success",
            target_type="luxriot_bookmark",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={
                "severity": severity,
                "state": state,
                "timestamp_supplied": timestamp_ms is not None,
                "title_length": len(title),
                "description_length": len(str(description)),
                "success": bool(result.get("success"))
                if isinstance(result, Mapping)
                else None,
            },
        )
        if audit_error is not None:
            return audit_error
        return jsonify(result)
    except Exception as exc:
        audit_error = _write_completion_audit_or_error(
            action="luxriot.bookmark.create.completed",
            result="failure",
            target_type="luxriot_bookmark",
            target_id=str(channel_id),
            channel_id=channel_id,
            details={
                "reason": type(exc).__name__,
                "severity": severity,
                "state": state,
            },
        )
        if audit_error is not None:
            return audit_error
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
        "bookmark": _coerce_bool(data.get('bookmark'), default=False),
        "bookmark_authorized": False,
        "window_sec": window_sec,
        "fps": data.get('fps'),
        "roi_enabled": probe_roi_enabled,
        "roi_norm": _probe_roi_norm_to_payload(probe_roi_norm),
    }
    if probe_like["bookmark"]:
        bookmark_guard = _bookmark_permission_guard_error(
            action="http.probes_query.bookmark",
        )
        if bookmark_guard is not None:
            return bookmark_guard
        probe_like["bookmark_authorized"] = True
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


def _probe_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _probe_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return default


def _probe_text_values(raw: Any) -> List[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def _find_probe_by_id(
    probe_id: Any,
    probes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized = str(probe_id or "").strip()
    if not normalized:
        return {}
    try:
        source = probes if probes is not None else probes_store.list_probes()
        return dict(
            next(
                (p for p in source if str(p.get("id")) == normalized),
                {},
            )
        )
    except Exception:
        return {}


def _probe_bookmark_requested(
    data: Mapping[str, Any],
    existing_probe: Mapping[str, Any],
) -> bool:
    bookmark_field_present = 'bookmark' in data
    if bookmark_field_present:
        return _coerce_bool(data.get('bookmark'), default=False)
    if existing_probe:
        return bool(existing_probe.get('bookmark', False))
    return False


def _probe_bookmark_settings_touched(data: Mapping[str, Any]) -> bool:
    return any(
        key in data
        for key in (
            'bookmark_cooldown_sec',
            'bookmark_dedupe_window_sec',
        )
    )


def _probe_bookmark_guard_error(
    data: Mapping[str, Any],
    existing_probe: Mapping[str, Any],
    *,
    action: str,
):
    bookmark_requested = _probe_bookmark_requested(data, existing_probe)
    bookmark_settings_touched = _probe_bookmark_settings_touched(data)
    if bookmark_requested or (bool(existing_probe.get('bookmark')) and bookmark_settings_touched):
        return _bookmark_permission_guard_error(action=action)
    return None


def _build_probe_payload(
    data: Mapping[str, Any],
    *,
    existing_probe: Optional[Mapping[str, Any]] = None,
    channel_id_override: Optional[int] = None,
    probe_id_override: Optional[str] = None,
    force_new: bool = False,
    name_override: Optional[str] = None,
    cast_group_id: Optional[str] = None,
    cast_base_name: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        channel_id = int(channel_id_override if channel_id_override is not None else (data.get('channel_id') or config.LUXRIOT_DEFAULT_CHANNEL_ID))
    except Exception as exc:
        raise ValueError('Provide a valid channel_id') from exc
    if channel_id <= 0:
        raise ValueError('Provide a valid channel_id')

    positives = _probe_text_values(data.get('positives'))
    negatives = _probe_text_values(data.get('negatives'))
    raw_image_probe = data.get('image_probe') or {}
    image_probe = dict(raw_image_probe) if isinstance(raw_image_probe, Mapping) else {}
    if not positives and not (image_probe.get('data') and image_probe.get('enabled', True) is not False):
        raise ValueError('Provide at least one positive (text or image).')

    existing = dict(existing_probe or {})
    probe_roi_enabled, probe_roi_norm = _parse_probe_roi(data)
    bookmark_requested = _probe_bookmark_requested(data, existing)
    bookmark_authorized = bool(
        bookmark_requested
        and _current_request_has_permission(Permission.BOOKMARKS_CREATE)
    )
    pairs = data.get('pairs') or []
    if not isinstance(pairs, list):
        pairs = []
    recent_hits = data.get('recent_hits') or []
    if not isinstance(recent_hits, list):
        recent_hits = []

    if force_new:
        probe_id = None
    elif probe_id_override is not None:
        probe_id = str(probe_id_override).strip() or None
    else:
        probe_id = data.get('id') or None

    name = (name_override if name_override is not None else (data.get('name') or '')).strip()
    if not name:
        name = f"probe-{int(time.time())}"

    probe = {
        "id": probe_id,
        "name": name,
        "channel_id": channel_id,
        "positives": positives,
        "negatives": negatives,
        "pos_floor": _probe_float(data.get('pos_floor'), 0.2),
        "margin": max(0.0, _probe_float(data.get('margin'), 0.05)),
        "bookmark_cooldown_sec": max(
            0.0,
            _probe_float(
                data.get('bookmark_cooldown_sec'),
                existing.get('bookmark_cooldown_sec', config.PROBE_BOOKMARK_COOLDOWN_SEC),
            ),
        ),
        "bookmark_dedupe_window_sec": max(
            0.5,
            _probe_float(
                data.get('bookmark_dedupe_window_sec'),
                existing.get('bookmark_dedupe_window_sec', config.PROBE_BOOKMARK_DEDUPE_WINDOW_SEC),
            ),
        ),
        "top_k": _probe_int(data.get('top_k'), 6),
        "window_sec": _probe_float(data.get('window_sec'), 300.0),
        "severity": (data.get('severity') or 'critical').lower(),
        "bookmark": bookmark_requested,
        "bookmark_authorized": bookmark_authorized,
        "enabled": _coerce_bool(data.get('enabled'), True),
        "image_probe": image_probe,
        "roi_enabled": probe_roi_enabled,
        "roi_norm": _probe_roi_norm_to_payload(probe_roi_norm),
        "pairs": pairs,
        "last_hit": data.get('last_hit'),
        "recent_hits": recent_hits[:PROBE_MAX_STORED_HITS],
        "bookmark_gate": existing.get("bookmark_gate"),
        "bookmark_gate_updated_at_ms": existing.get("bookmark_gate_updated_at_ms"),
    }
    if cast_group_id:
        probe["cast_group_id"] = str(cast_group_id)
    if cast_base_name:
        probe["cast_base_name"] = str(cast_base_name)
    return probe


def _matching_probe_for_channel(
    probes: Sequence[Mapping[str, Any]],
    *,
    name: str,
    channel_id: int,
) -> Dict[str, Any]:
    normalized_name = str(name or "").strip().casefold()
    for probe in probes:
        try:
            probe_channel = int(probe.get("channel_id"))
        except Exception:
            continue
        if probe_channel != channel_id:
            continue
        if str(probe.get("name") or "").strip().casefold() == normalized_name:
            return dict(probe)
    return {}


@app.route('/probes/save', methods=['POST'])
def probes_save():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    existing_probe = _find_probe_by_id(data.get('id'))
    bookmark_guard = _probe_bookmark_guard_error(
        data,
        existing_probe,
        action="http.probes_save.bookmark_settings",
    )
    if bookmark_guard is not None:
        return bookmark_guard
    try:
        probe = _build_probe_payload(data, existing_probe=existing_probe)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    saved = probes_store.upsert_probe(probe)
    return jsonify({'success': True, 'probe': saved})


@app.route('/probes/cast', methods=['POST'])
def probes_cast():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    raw_channel_ids = data.get("channel_ids")
    if not isinstance(raw_channel_ids, (list, tuple)):
        return jsonify({"error": "Provide channel_ids as a non-empty list."}), 400
    channel_ids: List[int] = []
    seen_channels: Set[int] = set()
    for raw_channel_id in raw_channel_ids:
        try:
            channel_id = int(raw_channel_id)
        except Exception:
            continue
        if channel_id <= 0 or channel_id in seen_channels:
            continue
        seen_channels.add(channel_id)
        channel_ids.append(channel_id)
    if not channel_ids:
        return jsonify({"error": "Select at least one valid channel."}), 400
    if len(channel_ids) > 500:
        return jsonify({"error": "Too many channels for one cast operation."}), 400

    conflict_policy = str(data.get("conflict") or "skip").strip().lower()
    if conflict_policy not in {"skip", "create", "update"}:
        return jsonify({"error": "Unsupported conflict policy."}), 400
    copy_roi = _coerce_bool(data.get("copy_roi"), False)
    cast_group_id = str(data.get("cast_group_id") or uuid.uuid4().hex).strip()
    base_name = str(data.get("name") or "").strip() or f"probe-{int(time.time())}"

    base_payload = dict(data)
    base_payload.pop("id", None)
    base_payload.pop("channel_ids", None)
    base_payload.pop("channels", None)
    base_payload.pop("conflict", None)
    base_payload.pop("copy_roi", None)
    base_payload["name"] = base_name
    if not copy_roi:
        base_payload["roi_enabled"] = False
        base_payload["roi_norm"] = None
        base_payload.pop("roi", None)

    try:
        existing_probes = probes_store.list_probes()
    except Exception as exc:
        app.logger.exception("Probe cast failed to list probes request_id=%s", getattr(g, "request_id", ""))
        return jsonify({"error": "Probe store is unavailable."}), 503

    for channel_id in channel_ids:
        existing_probe = (
            _matching_probe_for_channel(existing_probes, name=base_name, channel_id=channel_id)
            if conflict_policy == "update"
            else {}
        )
        bookmark_guard = _probe_bookmark_guard_error(
            base_payload,
            existing_probe,
            action="http.probes_cast.bookmark_settings",
        )
        if bookmark_guard is not None:
            return bookmark_guard
        try:
            _build_probe_payload(
                base_payload,
                existing_probe=existing_probe,
                channel_id_override=channel_id,
                probe_id_override=str(existing_probe.get("id") or "") if existing_probe else None,
                force_new=not existing_probe,
                cast_group_id=cast_group_id,
                cast_base_name=base_name,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    created: List[Dict[str, Any]] = []
    updated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for channel_id in channel_ids:
        matching_probe = _matching_probe_for_channel(
            existing_probes,
            name=base_name,
            channel_id=channel_id,
        )
        if matching_probe and conflict_policy == "skip":
            skipped.append(
                {
                    "channel_id": channel_id,
                    "probe_id": matching_probe.get("id"),
                    "reason": "matching_probe_exists",
                }
            )
            continue
        updating = bool(matching_probe and conflict_policy == "update")
        try:
            probe = _build_probe_payload(
                base_payload,
                existing_probe=matching_probe if updating else {},
                channel_id_override=channel_id,
                probe_id_override=str(matching_probe.get("id") or "") if updating else None,
                force_new=not updating,
                cast_group_id=cast_group_id,
                cast_base_name=base_name,
            )
            saved = probes_store.upsert_probe(probe)
            item = {
                "channel_id": channel_id,
                "probe_id": saved.get("id"),
                "name": saved.get("name"),
            }
            if updating:
                updated.append(item)
            else:
                created.append(item)
        except Exception as exc:
            failed.append({"channel_id": channel_id, "error": str(exc)})

    status = 207 if failed and (created or updated or skipped) else (500 if failed else 200)
    audit_error = _write_completion_audit_or_error(
        action="probes.cast.completed",
        result="partial" if failed and status == 207 else ("failure" if failed else "success"),
        target_type="probe_cast",
        target_id=cast_group_id,
        channel_id=channel_ids[0] if len(channel_ids) == 1 else None,
        details={
            "conflict": conflict_policy,
            "copy_roi": copy_roi,
            "created": len(created),
            "updated": len(updated),
            "skipped": len(skipped),
            "failed": len(failed),
            **_audit_key_details("channel_ids", channel_ids),
        },
    )
    if audit_error is not None:
        return audit_error
    return jsonify(
        {
            "success": not failed,
            "cast_group_id": cast_group_id,
            "created": created,
            "updated": updated,
            "skipped": skipped,
            "failed": failed,
            "counts": {
                "created": len(created),
                "updated": len(updated),
                "skipped": len(skipped),
                "failed": len(failed),
            },
        }
    ), status


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
    if probe.get('bookmark'):
        bookmark_guard = _bookmark_permission_guard_error(
            action="http.probes_run.bookmark",
        )
        if bookmark_guard is not None:
            return bookmark_guard
        probe['bookmark_authorized'] = True
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
        app.logger.exception(
            "Probe benchmark failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({"error": "probe_benchmark_failed"}), 500


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
        clip_query_vec = get_clip_text_embedding(query)
    except RuntimeError as exc:
        app.logger.info(
            "Detection text query embedding unavailable request_id=%s error=%s",
            getattr(g, "request_id", ""),
            exc,
        )
        return jsonify({'error': 'text_embedding_unavailable'}), 400
    except Exception as exc:
        app.logger.exception(
            "Detection text query embedding failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'text_embedding_failed'}), 500

    try:
        search_payload = _search_detections_archive(
            clip_query_vec=clip_query_vec,
            dino_query_vec=None,
            mode=mode,
            probe_id=filters['probe_id'],
            channel_id=filters['channel_id'],
            source=filters['source'],
            since_ms=filters['since_ms'],
            until_ms=filters['until_ms'],
            limit=limit,
            sort_by=sort_by,
            candidate_limit=candidate_limit,
            include_coverage=True,
        )
        if isinstance(search_payload, tuple):
            results, coverage = search_payload
        else:
            results = cast(List[Dict[str, Any]], search_payload)
            coverage = {}
        return jsonify(
            {
                'results': results,
                'coverage': coverage,
                'mode_requested': mode_requested,
                'mode_used': mode,
                'filters': filters,
                'query': query,
            }
        )
    except ArchiveStoreNotReady as exc:
        return _archive_store_not_ready_response(exc)
    except Exception as exc:
        app.logger.exception(
            "Detection text archive query failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'archive_query_failed'}), 500


@app.route('/detections/search_image', methods=['POST'])
def detections_search_image():
    mode = _normalize_detection_search_mode(request.form.get('embedder') or active_embedder)

    filters_payload = {
        'probe_id': request.form.get('probe_id'),
        'channel_id': request.form.get('channel_id'),
        'source': request.form.get('source'),
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
        app.logger.info(
            "Detection image query upload rejected request_id=%s error=%s",
            getattr(g, "request_id", ""),
            exc,
        )
        return jsonify({'error': 'invalid_uploaded_image'}), 400

    try:
        clip_query_vec = get_image_embedding_from_pil(pil_image, embedder='clip')
    except Exception as exc:
        app.logger.exception(
            "Detection image query CLIP embedding failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'image_embedding_failed'}), 500

    dino_query_vec: Optional[np.ndarray] = None
    if mode in {'dino', 'fusion'}:
        try:
            dino_query_vec = get_image_embedding_from_pil(pil_image, embedder='dino')
        except Exception as exc:
            print(f"Detections image search: DINO query embedding unavailable, fallback to CLIP only ({exc})")
            dino_query_vec = None

    try:
        search_payload = _search_detections_archive(
            clip_query_vec=clip_query_vec,
            dino_query_vec=dino_query_vec,
            mode=mode,
            probe_id=filters['probe_id'],
            channel_id=filters['channel_id'],
            source=filters['source'],
            since_ms=filters['since_ms'],
            until_ms=filters['until_ms'],
            limit=limit,
            sort_by=sort_by,
            candidate_limit=candidate_limit,
            include_coverage=True,
        )
        if isinstance(search_payload, tuple):
            results, coverage = search_payload
        else:
            results = cast(List[Dict[str, Any]], search_payload)
            coverage = {}
        return jsonify(
            {
                'results': results,
                'coverage': coverage,
                'mode_used': mode,
                'filters': filters,
            }
        )
    except ArchiveStoreNotReady as exc:
        return _archive_store_not_ready_response(exc)
    except Exception as exc:
        app.logger.exception(
            "Detection image archive query failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'archive_query_failed'}), 500


@app.route('/detections/list', methods=['GET'])
def detections_list():
    probe_id_raw = (request.args.get('probe_id') or '').strip()
    probe_id = probe_id_raw or None
    source = _normalize_archive_source_filter(request.args.get('source'))

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
            source=source,
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
                    'source': source,
                    'since_ms': since_ms,
                    'until_ms': until_ms,
                },
            }
        )
    except ArchiveStoreNotReady as exc:
        return _archive_store_not_ready_response(exc)
    except Exception as exc:
        app.logger.exception(
            "Detection archive list failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'archive_query_failed'}), 500


@app.route('/detections/summary', methods=['GET'])
def detections_summary():
    source = _normalize_archive_source_filter(request.args.get('source'))
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

    until_ms_raw = (request.args.get('until_ms') or '').strip()
    until_ms: Optional[int] = None
    if until_ms_raw:
        try:
            until_ms = int(until_ms_raw)
        except Exception:
            return jsonify({'error': 'until_ms must be an integer'}), 400

    try:
        limit = int(request.args.get('limit', 100))
    except Exception:
        limit = 100

    try:
        summary = detections_store.summarize_by_probe(
            since_ms=since_ms,
            channel_id=channel_id,
            source=source,
            limit=limit,
            until_ms=until_ms,
        )
        return jsonify(
            {
                'summary': summary,
                'count': len(summary),
                'filters': {
                    'channel_id': channel_id,
                    'source': source,
                    'since_ms': since_ms,
                    'until_ms': until_ms,
                },
            }
        )
    except ArchiveStoreNotReady as exc:
        return _archive_store_not_ready_response(exc)
    except Exception as exc:
        app.logger.exception(
            "Detection archive summary failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'archive_query_failed'}), 500


@app.route('/detections/diagnostics', methods=['GET'])
def detections_diagnostics():
    source = _normalize_archive_source_filter(request.args.get('source'))
    channel_id_raw = (request.args.get('channel_id') or '').strip()
    channel_id: Optional[int] = None
    if channel_id_raw:
        try:
            channel_id = int(channel_id_raw)
        except Exception:
            return jsonify({'error': 'channel_id must be an integer'}), 400

    since_ms = _to_optional_int(request.args.get('since_ms'))
    until_ms = _to_optional_int(request.args.get('until_ms'))
    if since_ms is None:
        hours_raw = (request.args.get('hours') or '').strip()
        try:
            hours = float(hours_raw) if hours_raw else 24.0
        except Exception:
            return jsonify({'error': 'hours must be numeric'}), 400
        if hours > 0:
            since_ms = int(time.time() * 1000 - (hours * 3600 * 1000))

    try:
        limit = int(request.args.get('limit', 8))
    except Exception:
        limit = 8
    limit = max(1, min(50, limit))

    try:
        source_summary_fn = getattr(detections_store, "summarize_by_source", None)
        source_summary = (
            list(source_summary_fn(since_ms=since_ms, channel_id=channel_id))
            if callable(source_summary_fn)
            else []
        )
        recent_rows, total = detections_store.list_detections(
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=0,
        )
        recent: List[Dict[str, Any]] = []
        for row in recent_rows:
            thumbnail = str(row.get("thumbnail") or "")
            recent.append(
                {
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "channel_id": row.get("channel_id"),
                    "probe_id": row.get("probe_id"),
                    "probe_name": row.get("probe_name"),
                    "timestamp_ms": row.get("timestamp_ms"),
                    "severity": row.get("severity"),
                    "has_thumbnail": bool(thumbnail),
                    "thumbnail_chars": len(thumbnail),
                    "has_clip": bool(row.get("has_clip")),
                    "has_dino": bool(row.get("has_dino")),
                    "shard_key": row.get("shard_key"),
                }
            )
        return jsonify(
            {
                "ok": True,
                "filters": {
                    "channel_id": channel_id,
                    "source": source,
                    "since_ms": since_ms,
                    "until_ms": until_ms,
                    "limit": limit,
                },
                "storage": _archive_storage_summary(),
                "sources": source_summary,
                "recent": recent,
                "recent_total": total,
            }
        )
    except ArchiveStoreNotReady as exc:
        return _archive_store_not_ready_response(exc)
    except Exception:
        app.logger.exception(
            "Detection archive diagnostics failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'error': 'archive_diagnostics_failed'}), 500


ENV_PREFIX = "EVOSSEARCH_"
_ENV_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@%+=,-]*$")


def _bool_to_env(value: Any) -> str:
    return "true" if bool(value) else "false"


def _decode_env_value(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        body = text[1:-1]
        if text[0] == '"':
            body = (
                body
                .replace("\\n", "\n")
                .replace('\\"', '"')
                .replace("\\$", "$")
                .replace("\\\\", "\\")
            )
        return body
    return text


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
        env_map[key] = _decode_env_value(value_raw)
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
        parsed[key] = _decode_env_value(value_raw)
    return parsed


def _serialize_env_map(env_map: Dict[str, str]) -> str:
    keys_sorted = sorted(env_map.keys())
    return "\n".join(f"{key}={_quote_env_value(env_map[key])}" for key in keys_sorted)


def _quote_env_value(value: Any) -> str:
    text = str(value or "")
    if text and "\n" not in text and _ENV_SAFE_VALUE_RE.fullmatch(text):
        return text
    escaped = (
        text
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
        .replace("$", "\\$")
    )
    return f'"{escaped}"'


def _write_env_file_atomic(content: str, path: Union[str, Path] = ".env") -> None:
    env_path = Path(path)
    if not hasattr(env_path, "with_name"):
        env_path.write_text(content, encoding="utf-8")
        return
    tmp_path = env_path.with_name(f".{env_path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    os.replace(tmp_path, env_path)
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass


ENV_SECRET_REDACTION = "__EVOSSEARCH_SECRET_SET__"
ENV_SECRET_KEY_PARTS = (
    "PASSWORD",
    "TOKEN",
    "SECRET",
    "API_KEY",
    "PRIVATE_KEY",
    "DSN",
    "DATABASE_URL",
)


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
    lm_profiles = _configured_lm_profiles()
    lm_profile_ids = [
        profile_id
        for profile_id in lm_profiles
        if str(profile_id).strip() and str(profile_id).strip() != "default"
    ]
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
        "EVOSSEARCH_LM_PROFILES": ",".join(lm_profile_ids),
        "EVOSSEARCH_LM_AGENT_PROFILE_ID": str(
            getattr(config, "LM_AGENT_PROFILE_ID", "")
        ),
        "EVOSSEARCH_LM_VLM_PROFILE_ID": str(
            getattr(config, "LM_VLM_PROFILE_ID", "")
        ),
        "EVOSSEARCH_LM_VLM_BALANCER_ENABLED": _bool_to_env(
            getattr(config, "LM_VLM_BALANCER_ENABLED", False)
        ),
        "EVOSSEARCH_LM_VLM_BALANCER_PROFILES": ",".join(
            _configured_vlm_balancer_profile_ids()
        ),
        "EVOSSEARCH_LM_VIDEO_DEFAULT_FRAMES": str(config.LM_VIDEO_DEFAULT_FRAMES),
        "EVOSSEARCH_LM_VIDEO_MAX_FRAMES": str(config.LM_VIDEO_MAX_FRAMES),
        "EVOSSEARCH_LM_VIDEO_MAX_EDGE": str(config.LM_VIDEO_MAX_EDGE),
        "EVOSSEARCH_LM_VIDEO_MAX_TOKENS": str(config.LM_VIDEO_MAX_TOKENS),
        "EVOSSEARCH_LM_VIDEO_INPUT_WARNING_CHARS": str(
            getattr(config, "LM_VIDEO_INPUT_WARNING_CHARS", 24000)
        ),
        "EVOSSEARCH_LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS": str(
            getattr(config, "LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS", 2500000)
        ),
        "EVOSSEARCH_LM_VIDEO_TEMPERATURE": str(config.LM_VIDEO_TEMPERATURE),
        "EVOSSEARCH_AGENT_CONTEXT_CHARS_PER_TOKEN": os.getenv(
            "EVOSSEARCH_AGENT_CONTEXT_CHARS_PER_TOKEN", "4"
        ),
        "EVOSSEARCH_AGENT_CONTEXT_HISTORY_BUDGET_TOKENS": os.getenv(
            "EVOSSEARCH_AGENT_CONTEXT_HISTORY_BUDGET_TOKENS", "12000"
        ),
        "EVOSSEARCH_AGENT_CONTEXT_WARNING_TOKENS": os.getenv(
            "EVOSSEARCH_AGENT_CONTEXT_WARNING_TOKENS", "45000"
        ),
        "EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS": os.getenv(
            "EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS", "60000"
        ),
        "EVOSSEARCH_OFFLINE_VIDEO_ENABLED": _bool_to_env(getattr(config, "OFFLINE_VIDEO_ENABLED", False)),
        "EVOSSEARCH_PROBE_SNAP_ENABLED": _bool_to_env(getattr(config, "PROBE_SNAP_ENABLED", False)),
        "EVOSSEARCH_INDEXED_FOLDER_ENABLED": _bool_to_env(getattr(config, "INDEXED_FOLDER_ENABLED", False)),
        "EVOSSEARCH_LUXRIOT_BASE_URL": str(config.LUXRIOT_BASE_URL),
        "EVOSSEARCH_LUXRIOT_USERNAME": str(config.LUXRIOT_USERNAME),
        "EVOSSEARCH_LUXRIOT_PASSWORD": str(config.LUXRIOT_PASSWORD),
        "EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID": str(config.LUXRIOT_DEFAULT_CHANNEL_ID),
        "EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL": str(config.LUXRIOT_SNAPSHOT_INTERVAL),
        "EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE": str(config.LUXRIOT_SNAPSHOT_MAX_EDGE),
        "EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES": str(config.LUXRIOT_MAX_BUFFER_FRAMES),
        "EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS": str(
            getattr(config, "LUXRIOT_SUMMARY_RETENTION_DAYS", 7.0)
        ),
        "EVOSSEARCH_LUXRIOT_SUMMARY_HISTORY_LIMIT": str(
            getattr(config, "LUXRIOT_SUMMARY_HISTORY_LIMIT", 600)
        ),
        "EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS": _bool_to_env(config.LUXRIOT_AUTO_BOOKMARKS),
        "EVOSSEARCH_LUXRIOT_ALERTS_MAX_PER_BATCH": str(getattr(config, "LUXRIOT_ALERTS_MAX_PER_BATCH", 8)),
        "EVOSSEARCH_LUXRIOT_ALERT_POLICY_PROMPT": str(getattr(config, "LUXRIOT_ALERT_POLICY_PROMPT", "")),
        "EVOSSEARCH_LUXRIOT_STATE_TRANSITIONS_ENABLED": _bool_to_env(
            getattr(config, "LUXRIOT_STATE_TRANSITIONS_ENABLED", True)
        ),
        "EVOSSEARCH_LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES": str(
            getattr(config, "LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES", 2)
        ),
        "EVOSSEARCH_LUXRIOT_STATE_TRANSITION_ALERT_EVENTS": _bool_to_env(
            getattr(config, "LUXRIOT_STATE_TRANSITION_ALERT_EVENTS", True)
        ),
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
        "EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED": _bool_to_env(
            getattr(config, "SECURE_DEPLOYMENT_REQUIRED", False)
        ),
        "EVOSSEARCH_ARCHIVE_STORE": str(getattr(config, "ARCHIVE_STORE", "auto")),
        "EVOSSEARCH_ARCHIVE_TENANT_ID": str(
            getattr(config, "ARCHIVE_TENANT_ID", "")
        ),
        "EVOSSEARCH_ARCHIVE_RETENTION_ENABLED": _bool_to_env(
            getattr(config, "ARCHIVE_RETENTION_ENABLED", True)
        ),
        "EVOSSEARCH_ARCHIVE_MAX_RECORDS": str(
            getattr(config, "ARCHIVE_MAX_RECORDS", 5000000)
        ),
        "EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS": str(
            getattr(config, "ARCHIVE_ROW_RETENTION_DAYS", 90.0)
        ),
        "EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS": str(
            getattr(config, "ARCHIVE_THUMBNAIL_RETENTION_DAYS", 14.0)
        ),
        "EVOSSEARCH_ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC": str(
            getattr(config, "ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC", 3600.0)
        ),
        "EVOSSEARCH_ARCHIVE_RETENTION_BATCH_SIZE": str(
            getattr(config, "ARCHIVE_RETENTION_BATCH_SIZE", 5000)
        ),
        "EVOSSEARCH_ARCHIVE_ESTIMATE_CHANNELS": str(
            getattr(config, "ARCHIVE_ESTIMATE_CHANNELS", 50)
        ),
        "EVOSSEARCH_ARCHIVE_ESTIMATE_FRAMES_PER_BATCH": str(
            getattr(config, "ARCHIVE_ESTIMATE_FRAMES_PER_BATCH", 2.5)
        ),
        "EVOSSEARCH_ARCHIVE_ESTIMATE_AVG_JPEG_KB": str(
            getattr(config, "ARCHIVE_ESTIMATE_AVG_JPEG_KB", 100.0)
        ),
        "EVOSSEARCH_ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY": str(
            getattr(config, "ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY", 250.0)
        ),
        "EVOSSEARCH_CORS_ALLOWED_ORIGINS": ",".join(config.CORS_ALLOWED_ORIGINS),
        "EVOSSEARCH_ALLOWED_ROOTS": os.pathsep.join(config.ALLOWED_ROOTS),
    }
    for profile_id in lm_profile_ids:
        profile = lm_profiles[profile_id]
        env[_lm_profile_env_key(profile_id, "KIND")] = str(
            profile.get("kind") or "general"
        )
        env[_lm_profile_env_key(profile_id, "BASE_URL")] = str(
            profile.get("base_url") or ""
        )
        env[_lm_profile_env_key(profile_id, "MODEL")] = str(profile.get("model") or "")
        env[_lm_profile_env_key(profile_id, "API_KEY")] = str(profile.get("api_key") or "")
        env[_lm_profile_env_key(profile_id, "TIMEOUT")] = str(
            profile.get("timeout") or config.LM_TIMEOUT
        )
        env[_lm_profile_env_key(profile_id, "ENABLED")] = _bool_to_env(
            _lm_profile_enabled(profile)
        )
        env[_lm_profile_env_key(profile_id, "GPU")] = str(profile.get("gpu") or "")
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
        f"{key}={_quote_env_value(value)}"
        for key, value in sorted(existing_map.items())
        if key.startswith(ENV_PREFIX) and key not in known_keys
    ]
    extra_other = [
        f"{key}={_quote_env_value(value)}"
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


def _archive_storage_summary() -> Dict[str, Any]:
    summary_fn = getattr(detections_store, "storage_summary", None)
    if not callable(summary_fn):
        return {
            "available": False,
            "backend": getattr(detections_store, "backend", "unknown"),
        }
    try:
        summary = dict(summary_fn())
        summary["available"] = True
        return summary
    except Exception as exc:
        app.logger.warning(
            "Archive storage summary failed request_id=%s error=%s",
            getattr(g, "request_id", ""),
            exc,
        )
        return {
            "available": False,
            "backend": getattr(detections_store, "backend", "unknown"),
            "error": type(exc).__name__,
        }


def _archive_capacity_estimate(
    *,
    channels: Optional[int] = None,
    batch_size: Optional[int] = None,
    snapshot_interval_sec: Optional[float] = None,
    frames_per_batch: Optional[float] = None,
    avg_jpeg_kb: Optional[float] = None,
    probe_records_per_channel_day: Optional[float] = None,
    summary_retention_days: Optional[float] = None,
    summary_history_limit: Optional[int] = None,
    frame_retention_days: Optional[float] = None,
    thumbnail_retention_days: Optional[float] = None,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    channel_count = max(
        1,
        int(channels if channels is not None else getattr(config, "ARCHIVE_ESTIMATE_CHANNELS", 50)),
    )
    default_batch = config.LUXRIOT_BATCH_SIZES[0] if config.LUXRIOT_BATCH_SIZES else 12
    batch = max(1, int(batch_size if batch_size is not None else default_batch))
    interval = max(
        0.2,
        float(snapshot_interval_sec if snapshot_interval_sec is not None else config.LUXRIOT_SNAPSHOT_INTERVAL),
    )
    frames = max(
        0.0,
        float(frames_per_batch if frames_per_batch is not None else getattr(config, "ARCHIVE_ESTIMATE_FRAMES_PER_BATCH", 2.5)),
    )
    jpeg_kb = max(
        1.0,
        float(avg_jpeg_kb if avg_jpeg_kb is not None else getattr(config, "ARCHIVE_ESTIMATE_AVG_JPEG_KB", 100.0)),
    )
    probe_daily = max(
        0.0,
        float(
            probe_records_per_channel_day
            if probe_records_per_channel_day is not None
            else getattr(config, "ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY", 250.0)
        ),
    )
    summary_days = max(
        0.0,
        float(summary_retention_days if summary_retention_days is not None else getattr(config, "LUXRIOT_SUMMARY_RETENTION_DAYS", 7.0)),
    )
    summary_cap = max(
        40,
        int(summary_history_limit if summary_history_limit is not None else getattr(config, "LUXRIOT_SUMMARY_HISTORY_LIMIT", 600)),
    )
    frame_days = max(
        0.0,
        float(frame_retention_days if frame_retention_days is not None else getattr(config, "ARCHIVE_ROW_RETENTION_DAYS", 90.0)),
    )
    thumb_days = max(
        0.0,
        float(thumbnail_retention_days if thumbnail_retention_days is not None else getattr(config, "ARCHIVE_THUMBNAIL_RETENTION_DAYS", 14.0)),
    )
    record_cap = max(
        1000,
        int(max_records if max_records is not None else getattr(config, "ARCHIVE_MAX_RECORDS", 5000000)),
    )

    batches_per_channel_day = 86400.0 / max(1.0, interval * batch)
    l0_per_channel_day = batches_per_channel_day
    l1_window = max(1.0, float(getattr(config, "LUXRIOT_ROLLUP_L1_WINDOW_SEC", 900)))
    l2_window = max(1.0, float(getattr(config, "LUXRIOT_ROLLUP_L2_WINDOW_SEC", 3600)))
    l3_window = max(1.0, float(getattr(config, "LUXRIOT_ROLLUP_L3_WINDOW_SEC", 21600)))
    summary_per_channel_day = (
        l0_per_channel_day
        + 86400.0 / l1_window
        + 86400.0 / l2_window
        + 86400.0 / l3_window
    )
    summary_rows_day = channel_count * summary_per_channel_day
    frame_rows_day = channel_count * (
        batches_per_channel_day * frames + probe_daily
    )
    uncapped_frame_rows = frame_rows_day * frame_days
    retained_frame_rows = min(float(record_cap), uncapped_frame_rows)
    retained_thumbnail_rows = min(
        retained_frame_rows,
        frame_rows_day * min(frame_days, thumb_days),
    )
    summary_rows_retained = channel_count * min(
        float(summary_cap),
        summary_per_channel_day * summary_days,
    )

    jpeg_bytes = jpeg_kb * 1024.0
    db_thumbnail_bytes = retained_thumbnail_rows * jpeg_bytes * 4.0 / 3.0
    fs_jpeg_bytes = retained_frame_rows * jpeg_bytes
    vector_and_meta_bytes = retained_frame_rows * 8192.0
    summary_bytes = summary_rows_retained * 4096.0
    db_total_bytes = db_thumbnail_bytes + vector_and_meta_bytes + summary_bytes
    total_bytes = db_total_bytes + fs_jpeg_bytes

    return {
        "inputs": {
            "channels": channel_count,
            "batch_size": batch,
            "snapshot_interval_sec": interval,
            "frames_per_batch": frames,
            "avg_jpeg_kb": jpeg_kb,
            "probe_records_per_channel_day": probe_daily,
            "summary_retention_days": summary_days,
            "summary_history_limit": summary_cap,
            "frame_retention_days": frame_days,
            "thumbnail_retention_days": thumb_days,
            "max_records": record_cap,
        },
        "daily": {
            "batches_per_channel": batches_per_channel_day,
            "summary_rows": summary_rows_day,
            "frame_rows": frame_rows_day,
        },
        "retained": {
            "summary_rows": summary_rows_retained,
            "frame_rows": retained_frame_rows,
            "thumbnail_rows": retained_thumbnail_rows,
            "capped_by_max_records": uncapped_frame_rows > float(record_cap),
        },
        "bytes": {
            "database": db_total_bytes,
            "database_thumbnails": db_thumbnail_bytes,
            "database_vectors_meta": vector_and_meta_bytes,
            "database_summaries": summary_bytes,
            "archive_files": fs_jpeg_bytes,
            "total": total_bytes,
        },
    }


@app.route('/settings/archive_capacity', methods=['GET'])
def settings_archive_capacity():
    guard = _settings_guard(write=False)
    if guard is not None:
        return guard
    try:
        def _optional_float(name: str) -> Optional[float]:
            raw = request.args.get(name)
            if raw is None or str(raw).strip() == "":
                return None
            return float(raw)

        def _optional_int(name: str) -> Optional[int]:
            raw = request.args.get(name)
            if raw is None or str(raw).strip() == "":
                return None
            return int(float(raw))

        estimate = _archive_capacity_estimate(
            channels=_optional_int("channels"),
            batch_size=_optional_int("batch_size"),
            snapshot_interval_sec=_optional_float("snapshot_interval_sec"),
            frames_per_batch=_optional_float("frames_per_batch"),
            avg_jpeg_kb=_optional_float("avg_jpeg_kb"),
            probe_records_per_channel_day=_optional_float("probe_records_per_channel_day"),
            summary_retention_days=_optional_float("summary_retention_days"),
            summary_history_limit=_optional_int("summary_history_limit"),
            frame_retention_days=_optional_float("frame_retention_days"),
            thumbnail_retention_days=_optional_float("thumbnail_retention_days"),
            max_records=_optional_int("max_records"),
        )
        return jsonify(
            {
                "success": True,
                "estimate": estimate,
                "current": _archive_storage_summary(),
                "retention": dict(_archive_retention_last_result),
            }
        )
    except Exception as exc:
        app.logger.info(
            "Archive capacity estimate failed request_id=%s error=%s",
            getattr(g, "request_id", ""),
            exc,
        )
        return jsonify({"success": False, "error": "Invalid archive capacity request"}), 400


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
        app.logger.exception(
            "Settings env read failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'success': False, 'error': 'settings_env_unavailable'}), 500


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

        header = "# evo-ssearch Configuration\n# Managed by settings env editor\n\n"
        _write_env_file_atomic(header + _serialize_env_map(merged_map) + "\n")

        audit_error = _write_completion_audit_or_error(
            action="settings.env.write.completed",
            result="success",
            target_type="settings_env",
            details=_audit_key_details("keys", target_env.keys()),
        )
        if audit_error is not None:
            return audit_error
        return jsonify(
            {
                'success': True,
                'message': 'Environment variables saved to .env. Restart the server to apply all changes.',
                'count': len(target_env),
            }
        )
    except Exception as exc:
        app.logger.exception(
            "Settings env write failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'success': False, 'error': 'settings_env_unavailable'}), 500


@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current configuration settings"""
    guard = _settings_guard(write=False)
    if guard is not None:
        return guard
    try:
        experimental_embedders_enabled = _experimental_embedding_models_enabled()
        requested_embedder = _normalize_embedder_for_policy(
            config.EMBEDDER if config.EMBEDDER in SUPPORTED_EMBEDDERS else active_embedder,
            bool(config.FUSION_ENABLED),
        )
        clip_model = _normalize_clip_model_for_policy(config.CLIP_MODEL)
        index_mode = _normalize_index_mode_for_policy(config.INDEX_MODE)
        settings = {
            'host': config.HOST,
            'port': config.PORT,
            'debug': config.DEBUG,
            'appVersion': config.APP_VERSION,
            'experimentalEmbeddersEnabled': experimental_embedders_enabled,
            'productionClipModel': _production_clip_model(),
            'embedder': requested_embedder,
            'clipModel': clip_model,
            'dinoModel': config.DINO_MODEL,
            'dinoEmbedDim': config.EMB_DIM_DINO,
            'dinoWeightsPath': config.DINO_WEIGHTS_PATH,
            'indexMode': index_mode,
            'fusionEnabled': bool(config.FUSION_ENABLED) if experimental_embedders_enabled else False,
            'fusionAlpha': config.FUSION_ALPHA,
            'rerankEnabled': config.RERANK_ENABLED,
            'rerankTopK': config.RERANK_TOP_K,
            'segmentsEnabled': bool(config.DINO_SEGMENTS_ENABLED) if experimental_embedders_enabled else False,
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
            'luxriotSummaryRetentionDays': getattr(config, "LUXRIOT_SUMMARY_RETENTION_DAYS", 7.0),
            'luxriotSummaryHistoryLimit': getattr(config, "LUXRIOT_SUMMARY_HISTORY_LIMIT", 600),
            'luxriotSummaryArchiveFramesPerBatch': getattr(
                config,
                "LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH",
                4,
            ),
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
            'offlineVideoEnabled': bool(getattr(config, "OFFLINE_VIDEO_ENABLED", False)),
            'probeSnapEnabled': bool(getattr(config, "PROBE_SNAP_ENABLED", False)),
            'indexedFolderEnabled': bool(getattr(config, "INDEXED_FOLDER_ENABLED", False)),
            'corsAllowedOrigins': list(config.CORS_ALLOWED_ORIGINS),
            'allowedRoots': list(config.ALLOWED_ROOTS),
            'archiveRetentionEnabled': getattr(config, "ARCHIVE_RETENTION_ENABLED", True),
            'archiveMaxRecords': getattr(config, "ARCHIVE_MAX_RECORDS", 5000000),
            'archiveRowRetentionDays': getattr(config, "ARCHIVE_ROW_RETENTION_DAYS", 90.0),
            'archiveThumbnailRetentionDays': getattr(config, "ARCHIVE_THUMBNAIL_RETENTION_DAYS", 14.0),
            'archiveRetentionPruneIntervalSec': getattr(config, "ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC", 3600.0),
            'archiveRetentionBatchSize': getattr(config, "ARCHIVE_RETENTION_BATCH_SIZE", 5000),
            'archiveEstimateChannels': getattr(config, "ARCHIVE_ESTIMATE_CHANNELS", 50),
            'archiveEstimateFramesPerBatch': getattr(config, "ARCHIVE_ESTIMATE_FRAMES_PER_BATCH", 2.5),
            'archiveEstimateAvgJpegKb': getattr(config, "ARCHIVE_ESTIMATE_AVG_JPEG_KB", 100.0),
            'archiveEstimateProbeRecordsPerChannelDay': getattr(
                config,
                "ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY",
                250.0,
            ),
            'archiveCapacityEstimate': _archive_capacity_estimate(),
            'archiveStorageSummary': _archive_storage_summary(),
            'envCount': len(_effective_env_map()),
        }
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        app.logger.exception(
            "Settings read failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'success': False, 'error': 'settings_unavailable'}), 500

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
        experimental_embedders_enabled = _experimental_embedding_models_enabled()

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
        if not experimental_embedders_enabled:
            fusion_enabled = False

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
        if not experimental_embedders_enabled:
            segments_enabled = False

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
        try:
            luxriot_summary_retention_days = float(
                data.get(
                    'luxriotSummaryRetentionDays',
                    getattr(config, "LUXRIOT_SUMMARY_RETENTION_DAYS", 7.0),
                )
            )
        except (TypeError, ValueError):
            luxriot_summary_retention_days = float(getattr(config, "LUXRIOT_SUMMARY_RETENTION_DAYS", 7.0))
        luxriot_summary_retention_days = max(0.0, min(3650.0, luxriot_summary_retention_days))
        try:
            luxriot_summary_history_limit = int(
                data.get(
                    'luxriotSummaryHistoryLimit',
                    getattr(config, "LUXRIOT_SUMMARY_HISTORY_LIMIT", 600),
                )
            )
        except (TypeError, ValueError):
            luxriot_summary_history_limit = int(getattr(config, "LUXRIOT_SUMMARY_HISTORY_LIMIT", 600))
        luxriot_summary_history_limit = max(40, min(1000000, luxriot_summary_history_limit))
        try:
            luxriot_summary_archive_frames_per_batch = int(
                data.get(
                    'luxriotSummaryArchiveFramesPerBatch',
                    getattr(config, "LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH", 4),
                )
            )
        except (TypeError, ValueError):
            luxriot_summary_archive_frames_per_batch = int(
                getattr(config, "LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH", 4)
            )
        luxriot_summary_archive_frames_per_batch = max(
            1,
            min(16, luxriot_summary_archive_frames_per_batch),
        )
        archive_retention_enabled = _coerce_bool(
            data.get('archiveRetentionEnabled', getattr(config, "ARCHIVE_RETENTION_ENABLED", True)),
            bool(getattr(config, "ARCHIVE_RETENTION_ENABLED", True)),
        )
        try:
            archive_max_records = int(data.get('archiveMaxRecords', getattr(config, "ARCHIVE_MAX_RECORDS", 5000000)))
        except (TypeError, ValueError):
            archive_max_records = int(getattr(config, "ARCHIVE_MAX_RECORDS", 5000000))
        archive_max_records = max(1000, min(500000000, archive_max_records))
        try:
            archive_row_retention_days = float(
                data.get('archiveRowRetentionDays', getattr(config, "ARCHIVE_ROW_RETENTION_DAYS", 90.0))
            )
        except (TypeError, ValueError):
            archive_row_retention_days = float(getattr(config, "ARCHIVE_ROW_RETENTION_DAYS", 90.0))
        archive_row_retention_days = max(0.0, min(3650.0, archive_row_retention_days))
        try:
            archive_thumbnail_retention_days = float(
                data.get('archiveThumbnailRetentionDays', getattr(config, "ARCHIVE_THUMBNAIL_RETENTION_DAYS", 14.0))
            )
        except (TypeError, ValueError):
            archive_thumbnail_retention_days = float(getattr(config, "ARCHIVE_THUMBNAIL_RETENTION_DAYS", 14.0))
        archive_thumbnail_retention_days = max(0.0, min(3650.0, archive_thumbnail_retention_days))
        try:
            archive_retention_prune_interval_sec = float(
                data.get(
                    'archiveRetentionPruneIntervalSec',
                    getattr(config, "ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC", 3600.0),
                )
            )
        except (TypeError, ValueError):
            archive_retention_prune_interval_sec = float(getattr(config, "ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC", 3600.0))
        archive_retention_prune_interval_sec = max(60.0, min(86400.0, archive_retention_prune_interval_sec))
        try:
            archive_retention_batch_size = int(
                data.get('archiveRetentionBatchSize', getattr(config, "ARCHIVE_RETENTION_BATCH_SIZE", 5000))
            )
        except (TypeError, ValueError):
            archive_retention_batch_size = int(getattr(config, "ARCHIVE_RETENTION_BATCH_SIZE", 5000))
        archive_retention_batch_size = max(100, min(50000, archive_retention_batch_size))
        try:
            archive_estimate_channels = int(data.get('archiveEstimateChannels', getattr(config, "ARCHIVE_ESTIMATE_CHANNELS", 50)))
        except (TypeError, ValueError):
            archive_estimate_channels = int(getattr(config, "ARCHIVE_ESTIMATE_CHANNELS", 50))
        archive_estimate_channels = max(1, min(10000, archive_estimate_channels))
        try:
            archive_estimate_frames_per_batch = float(
                data.get('archiveEstimateFramesPerBatch', getattr(config, "ARCHIVE_ESTIMATE_FRAMES_PER_BATCH", 2.5))
            )
        except (TypeError, ValueError):
            archive_estimate_frames_per_batch = float(getattr(config, "ARCHIVE_ESTIMATE_FRAMES_PER_BATCH", 2.5))
        archive_estimate_frames_per_batch = max(0.0, min(32.0, archive_estimate_frames_per_batch))
        try:
            archive_estimate_avg_jpeg_kb = float(
                data.get('archiveEstimateAvgJpegKb', getattr(config, "ARCHIVE_ESTIMATE_AVG_JPEG_KB", 100.0))
            )
        except (TypeError, ValueError):
            archive_estimate_avg_jpeg_kb = float(getattr(config, "ARCHIVE_ESTIMATE_AVG_JPEG_KB", 100.0))
        archive_estimate_avg_jpeg_kb = max(1.0, min(5000.0, archive_estimate_avg_jpeg_kb))
        try:
            archive_estimate_probe_records_per_channel_day = float(
                data.get(
                    'archiveEstimateProbeRecordsPerChannelDay',
                    getattr(config, "ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY", 250.0),
                )
            )
        except (TypeError, ValueError):
            archive_estimate_probe_records_per_channel_day = float(
                getattr(config, "ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY", 250.0)
            )
        archive_estimate_probe_records_per_channel_day = max(
            0.0,
            min(100000.0, archive_estimate_probe_records_per_channel_day),
        )
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

        clip_model = _normalize_clip_model_for_policy(data.get('clipModel', config.CLIP_MODEL))
        embedder = _normalize_embedder_for_policy(data.get('embedder', active_embedder), fusion_enabled)
        dino_model = str(data.get('dinoModel', config.DINO_MODEL)).strip() or config.DINO_MODEL
        try:
            dino_dim = int(data.get('dinoEmbedDim', config.EMB_DIM_DINO))
        except (TypeError, ValueError):
            dino_dim = config.EMB_DIM_DINO

        dino_weights_path = data.get('dinoWeightsPath', config.DINO_WEIGHTS_PATH) or ''
        dino_device = str(data.get('dinoDevice', config.DINO_DEVICE)).strip()

        index_mode = _normalize_index_mode_for_policy(data.get('indexMode', config.INDEX_MODE))

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
EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED={str(experimental_embedders_enabled).lower()}
EVOSSEARCH_PRODUCTION_CLIP_MODEL={_production_clip_model()}
EVOSSEARCH_EMBEDDER={embedder}
EVOSSEARCH_CLIP_MODEL={clip_model}
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
EVOSSEARCH_LM_VIDEO_INPUT_WARNING_CHARS={getattr(config, "LM_VIDEO_INPUT_WARNING_CHARS", 24000)}
EVOSSEARCH_LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS={getattr(config, "LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS", 2500000)}
EVOSSEARCH_LM_VIDEO_TEMPERATURE={config.LM_VIDEO_TEMPERATURE}
EVOSSEARCH_AGENT_CONTEXT_CHARS_PER_TOKEN={os.getenv("EVOSSEARCH_AGENT_CONTEXT_CHARS_PER_TOKEN", "4")}
EVOSSEARCH_AGENT_CONTEXT_HISTORY_BUDGET_TOKENS={os.getenv("EVOSSEARCH_AGENT_CONTEXT_HISTORY_BUDGET_TOKENS", "12000")}
EVOSSEARCH_AGENT_CONTEXT_WARNING_TOKENS={os.getenv("EVOSSEARCH_AGENT_CONTEXT_WARNING_TOKENS", "45000")}
EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS={os.getenv("EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS", "60000")}
EVOSSEARCH_OFFLINE_VIDEO_ENABLED={str(getattr(config, "OFFLINE_VIDEO_ENABLED", False)).lower()}
EVOSSEARCH_PROBE_SNAP_ENABLED={str(getattr(config, "PROBE_SNAP_ENABLED", False)).lower()}
EVOSSEARCH_INDEXED_FOLDER_ENABLED={str(getattr(config, "INDEXED_FOLDER_ENABLED", False)).lower()}

# Luxriot Evo integration
EVOSSEARCH_LUXRIOT_BASE_URL={luxriot_base_url}
EVOSSEARCH_LUXRIOT_USERNAME={luxriot_username}
EVOSSEARCH_LUXRIOT_PASSWORD={luxriot_password}
EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID={luxriot_default_channel_id}
EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL={luxriot_snapshot_interval}
EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE={luxriot_snapshot_max_edge}
EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES={luxriot_max_buffer_frames}
EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS={luxriot_summary_retention_days}
EVOSSEARCH_LUXRIOT_SUMMARY_HISTORY_LIMIT={luxriot_summary_history_limit}
EVOSSEARCH_LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH={luxriot_summary_archive_frames_per_batch}
EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS={str(luxriot_auto_bookmarks).lower()}
EVOSSEARCH_LUXRIOT_ALERTS_MAX_PER_BATCH={getattr(config, "LUXRIOT_ALERTS_MAX_PER_BATCH", 8)}
EVOSSEARCH_LUXRIOT_ALERT_POLICY_PROMPT={getattr(config, "LUXRIOT_ALERT_POLICY_PROMPT", "")}
EVOSSEARCH_LUXRIOT_STATE_TRANSITIONS_ENABLED={str(getattr(config, "LUXRIOT_STATE_TRANSITIONS_ENABLED", True)).lower()}
EVOSSEARCH_LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES={getattr(config, "LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES", 2)}
EVOSSEARCH_LUXRIOT_STATE_TRANSITION_ALERT_EVENTS={str(getattr(config, "LUXRIOT_STATE_TRANSITION_ALERT_EVENTS", True)).lower()}
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
EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED={str(getattr(config, "SECURE_DEPLOYMENT_REQUIRED", False)).lower()}
EVOSSEARCH_ARCHIVE_STORE={getattr(config, "ARCHIVE_STORE", "auto")}
EVOSSEARCH_ARCHIVE_TENANT_ID={getattr(config, "ARCHIVE_TENANT_ID", "")}
EVOSSEARCH_ARCHIVE_RETENTION_ENABLED={str(archive_retention_enabled).lower()}
EVOSSEARCH_ARCHIVE_MAX_RECORDS={archive_max_records}
EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS={archive_row_retention_days}
EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS={archive_thumbnail_retention_days}
EVOSSEARCH_ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC={archive_retention_prune_interval_sec}
EVOSSEARCH_ARCHIVE_RETENTION_BATCH_SIZE={archive_retention_batch_size}
EVOSSEARCH_ARCHIVE_ESTIMATE_CHANNELS={archive_estimate_channels}
EVOSSEARCH_ARCHIVE_ESTIMATE_FRAMES_PER_BATCH={archive_estimate_frames_per_batch}
EVOSSEARCH_ARCHIVE_ESTIMATE_AVG_JPEG_KB={archive_estimate_avg_jpeg_kb}
EVOSSEARCH_ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY={archive_estimate_probe_records_per_channel_day}
EVOSSEARCH_CORS_ALLOWED_ORIGINS={','.join(config.CORS_ALLOWED_ORIGINS)}
EVOSSEARCH_ALLOWED_ROOTS={os.pathsep.join(config.ALLOWED_ROOTS)}
"""

        parsed_env_content = _parse_env_editor_text(env_content)
        known_env_keys = set(parsed_env_content.keys())
        env_content = (
            "# evo-ssearch Configuration\n"
            "# Generated by settings panel\n\n"
            + _serialize_env_map(parsed_env_content)
            + _preserve_additional_env_lines(known_env_keys)
        )
        if not env_content.endswith("\n"):
            env_content += "\n"

        _write_env_file_atomic(env_content)

        config.HOST = data['host']
        config.PORT = port
        config.DEBUG = debug_enabled
        embedder_state_changed = (
            str(config.EMBEDDER) != str(embedder)
            or str(config.CLIP_MODEL) != str(clip_model)
            or str(config.INDEX_MODE) != str(index_mode)
            or bool(config.FUSION_ENABLED) != bool(fusion_enabled)
            or bool(config.DINO_SEGMENTS_ENABLED) != bool(segments_enabled)
        )

        config.EXPERIMENTAL_EMBEDDERS_ENABLED = experimental_embedders_enabled
        config.PRODUCTION_CLIP_MODEL = _production_clip_model()
        config.EMBEDDER = embedder
        config.CLIP_MODEL = clip_model
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
        config.LUXRIOT_SUMMARY_RETENTION_DAYS = luxriot_summary_retention_days
        config.LUXRIOT_SUMMARY_HISTORY_LIMIT = luxriot_summary_history_limit
        config.LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH = luxriot_summary_archive_frames_per_batch
        config.LUXRIOT_AUTO_BOOKMARKS = luxriot_auto_bookmarks
        config.LUXRIOT_SEVERITY_MAP = merged_sev
        luxriot_manager.summary_retention_days = luxriot_summary_retention_days
        luxriot_manager.summary_history_limit = luxriot_summary_history_limit
        luxriot_manager.summary_archive_frames_per_batch = luxriot_summary_archive_frames_per_batch
        config.ARCHIVE_RETENTION_ENABLED = archive_retention_enabled
        config.ARCHIVE_MAX_RECORDS = archive_max_records
        config.ARCHIVE_ROW_RETENTION_DAYS = archive_row_retention_days
        config.ARCHIVE_THUMBNAIL_RETENTION_DAYS = archive_thumbnail_retention_days
        config.ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC = archive_retention_prune_interval_sec
        config.ARCHIVE_RETENTION_BATCH_SIZE = archive_retention_batch_size
        config.ARCHIVE_ESTIMATE_CHANNELS = archive_estimate_channels
        config.ARCHIVE_ESTIMATE_FRAMES_PER_BATCH = archive_estimate_frames_per_batch
        config.ARCHIVE_ESTIMATE_AVG_JPEG_KB = archive_estimate_avg_jpeg_kb
        config.ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY = archive_estimate_probe_records_per_channel_day
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
        if embedder_state_changed:
            try:
                probe_manager.clear_all()
            except Exception:
                pass
            try:
                detection_clip_shard_cache.clear()
            except Exception:
                pass
        warmup_warning = warm_start_embedder()
        message = 'Settings saved successfully. Restart the server if issues persist.'
        payload: Dict[str, Any] = {'success': True, 'message': message}
        if warmup_warning:
            app.logger.warning(
                "Embedder warmup warning after settings save request_id=%s warning=%s",
                getattr(g, "request_id", ""),
                warmup_warning,
            )
            payload['warning'] = 'Embedder warmup failed; restart or check server logs.'
        audit_details = _audit_key_details("fields", data.keys())
        audit_details.update(
            {
                "embedder": embedder,
                "index_mode": index_mode,
                "debug": debug_enabled,
                "luxriot_password_supplied": luxriot_password_raw is not None,
                "warmup_warning": bool(warmup_warning),
            }
        )
        audit_error = _write_completion_audit_or_error(
            action="settings.write.completed",
            result="success",
            target_type="settings",
            details=audit_details,
        )
        if audit_error is not None:
            return audit_error
        return jsonify(payload)

    except Exception as e:
        app.logger.exception(
            "Settings write failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({'success': False, 'error': 'settings_update_failed'}), 500


def _stop_probe_daemon_thread() -> None:
    global probe_daemon_thread
    probe_daemon_stop.set()
    if probe_daemon_thread is not None and probe_daemon_thread.is_alive():
        probe_daemon_thread.join(timeout=1.5)


def ensure_probe_daemon_thread() -> None:
    """Start the saved-probe daemon for production entrypoints."""
    global probe_daemon_thread
    if probe_daemon_thread is not None and probe_daemon_thread.is_alive():
        return
    probe_daemon_stop.clear()
    probe_daemon_thread = threading.Thread(
        target=_probe_daemon,
        name="eva-probe-daemon",
        daemon=True,
    )
    probe_daemon_thread.start()


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
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('session_id') or '').strip() or None
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
        agent_session_id=session_id,
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
    default_profile = _resolve_lm_profile(kind="agent")
    default_values = {
        _lm_profile_selector_value(default_profile),
        str(default_profile.get("model") or "").strip(),
    }
    default_values.discard("")
    with _agent_runner_lock:
        _agent_runtime_model_override = raw_model if raw_model and raw_model not in default_values else None
        _agent_runner = None
    return jsonify({'success': True, **_get_agent_config_payload()})


@app.route('/lm/models', methods=['GET'])
def lm_models():
    force = str(request.args.get('force') or '').strip().lower() in TRUE_BOOL_STRINGS
    payload = _fetch_lm_model_catalog(force=force)
    payload['agent'] = _get_agent_config_payload()
    profiles = payload.get("profiles") if isinstance(payload, Mapping) else None
    models = payload.get("models") if isinstance(payload, Mapping) else None
    audit_error = _write_completion_audit_or_error(
        action="lm.models.completed",
        result="success",
        target_type="lm_catalog",
        details={
            "force": force,
            "profile_count": len(profiles) if isinstance(profiles, Sequence) else 0,
            "model_count": len(models) if isinstance(models, Sequence) else 0,
            "has_error": bool(payload.get("error")) if isinstance(payload, Mapping) else False,
        },
    )
    if audit_error is not None:
        return audit_error
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
    global _audit_db_pool, _audit_reader, _audit_writer, _control_plane_db_pool
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
            _audit_reader = None
            _identity_repository = None
    except Exception:
        pass
    try:
        luxriot_manager.stop_all_streams(
            stop_video=True,
            stop_analytics=True,
            pause_analytics=False,
            update_desired=False,
        )
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
    ensure_probe_daemon_thread()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
