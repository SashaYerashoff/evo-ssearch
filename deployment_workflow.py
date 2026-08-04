"""Durable, bounded commissioning state for the EVA Protocol Deploy workflow.

The agent model is deliberately not the workflow database.  This module keeps
the site inventory, operator scope, survey receipts, deployment plan, and
commissioning result in the tenant-scoped runtime-state store.  Tool results
can therefore stay compact enough for a small 4B agent while a deployment is
resumed across chat turns or service restarts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from rollup_deep_review import QuietWindowSchedule
from maritime_profiles import (
    MARITIME_CHANNEL_ROLES,
    MARITIME_L0_PROMPT,
    MARITIME_ROLLUP_PROMPTS,
    maritime_requirement,
    maritime_role_label,
)


DEPLOYMENT_STATE_VERSION = 1
DEPLOYMENT_INDEX_KEY = "protocol_deploy:index:v1"
DEPLOYMENT_KEY_PREFIX = "protocol_deploy:v1:"
COUNTED_STATE_PROFILES_KEY = "counted_state_profiles:v1"
MAX_DEPLOYMENT_CHANNELS = 8
MAX_GROUPS = 8
MAX_REQUIREMENT_PACKS = 16
MAX_ALERTS_PER_PACK = 6
_SEVERITIES = frozenset({"ignore", "log", "info", "low", "normal", "high", "critical"})
_NOVELTY = frozenset({"low", "balanced", "high"})
_DEPLOYMENT_PROFILES = frozenset({"general", "maritime"})
_STARTER_POLICY_MODES = frozenset({"none", "shadow"})
_COUNTER_MODES = frozenset(
    {"none", "count_transitions", "measure_duration", "count_and_duration"}
)
_TRANSITIONS = frozenset(
    {"positive_to_negative", "negative_to_positive", "any"}
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class DeploymentWorkflowError(ValueError):
    """Operator-visible deployment workflow validation error."""


def _bounded_text(value: Any, maximum: int) -> str:
    return " ".join(str(value or "").split())[:maximum]


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise DeploymentWorkflowError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DeploymentWorkflowError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise DeploymentWorkflowError(f"{field} must be a positive integer")
    return parsed


def _bounded_float(
    value: Any,
    *,
    field: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DeploymentWorkflowError(f"{field} must be numeric") from exc
    if not minimum <= parsed <= maximum:
        raise DeploymentWorkflowError(
            f"{field} must be between {minimum:g} and {maximum:g}"
        )
    return parsed


def _slug(value: Any, fallback: str = "watch") -> str:
    normalized = _SLUG_RE.sub("-", str(value or "").strip().casefold()).strip("-")
    return (normalized or fallback)[:48]


def _stable_metric_id(
    deployment_id: str,
    channel_id: int,
    name: str,
) -> str:
    digest = hashlib.blake2s(
        f"{deployment_id}\x1f{channel_id}\x1f{name}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"metric-{digest}"


def _normalize_channel_ids(
    values: Any,
    *,
    maximum: int = MAX_DEPLOYMENT_CHANNELS,
) -> List[int]:
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray)
    ):
        raise DeploymentWorkflowError("channel_ids must be a list")
    normalized: List[int] = []
    seen = set()
    for value in values:
        channel_id = _positive_int(value, "channel_id")
        if channel_id in seen:
            continue
        seen.add(channel_id)
        normalized.append(channel_id)
    if not normalized:
        raise DeploymentWorkflowError("select at least one channel")
    if len(normalized) > maximum:
        raise DeploymentWorkflowError(
            f"Protocol Deploy supports at most {maximum} selected channels"
        )
    return normalized


def _normalize_groups(
    groups: Any,
    *,
    selected_channel_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    if groups in (None, []):
        return []
    if not isinstance(groups, Sequence) or isinstance(
        groups, (str, bytes, bytearray)
    ):
        raise DeploymentWorkflowError("groups must be a list")
    if len(groups) > MAX_GROUPS:
        raise DeploymentWorkflowError(f"at most {MAX_GROUPS} groups are supported")
    allowed = set(int(item) for item in selected_channel_ids)
    claimed = set()
    normalized: List[Dict[str, Any]] = []
    for raw in groups:
        if not isinstance(raw, Mapping):
            raise DeploymentWorkflowError("each group must be an object")
        name = _bounded_text(raw.get("name"), 80)
        if not name:
            raise DeploymentWorkflowError("group name is required")
        channel_ids = _normalize_channel_ids(
            raw.get("channel_ids") or [],
            maximum=MAX_DEPLOYMENT_CHANNELS,
        )
        if any(channel_id not in allowed for channel_id in channel_ids):
            raise DeploymentWorkflowError(
                f"group {name!r} contains a channel outside the selected scope"
            )
        overlap = claimed.intersection(channel_ids)
        if overlap:
            raise DeploymentWorkflowError(
                f"channels may belong to only one deployment group: {sorted(overlap)}"
            )
        claimed.update(channel_ids)
        normalized.append({"name": name, "channel_ids": channel_ids})
    return normalized


def _normalize_alert(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    name = _bounded_text(raw.get("name") or f"watch {index}", 100)
    description = _bounded_text(raw.get("description") or name, 600)
    severity = str(raw.get("severity") or "normal").strip().lower()
    if severity not in _SEVERITIES:
        raise DeploymentWorkflowError(f"unsupported alert severity: {severity}")
    positive_query = _bounded_text(raw.get("positive_query"), 300)
    contrast_query = _bounded_text(raw.get("contrast_query"), 300)
    counter_mode = str(raw.get("counter_mode") or "none").strip().lower()
    if counter_mode not in _COUNTER_MODES:
        raise DeploymentWorkflowError(f"unsupported counter_mode: {counter_mode}")
    if (
        counter_mode == "count_transitions"
        and raw.get("duration_state") is not None
    ):
        # Small heads often encode “count transitions and duration” as a
        # transition counter plus an explicit duration_state.  Preserve both
        # operator intents instead of silently discarding dwell time.
        counter_mode = "count_and_duration"
    count_transition = str(
        raw.get("count_transition") or "positive_to_negative"
    ).strip().lower()
    if count_transition not in _TRANSITIONS:
        raise DeploymentWorkflowError(
            f"unsupported count_transition: {count_transition}"
        )
    duration_state = str(
        raw.get("duration_state") or "positive"
    ).strip().lower()
    if duration_state not in {"positive", "negative"}:
        raise DeploymentWorkflowError(
            "duration_state must be positive or negative"
        )
    if counter_mode != "none" and (not positive_query or not contrast_query):
        raise DeploymentWorkflowError(
            f"counted state {name!r} requires positive_query and contrast_query"
        )
    return {
        "name": name,
        "description": description,
        "severity": severity,
        "positive_query": positive_query,
        "contrast_query": contrast_query,
        "positive_label": _bounded_text(
            raw.get("positive_label") or "positive", 60
        ),
        "negative_label": _bounded_text(
            raw.get("negative_label") or "negative", 60
        ),
        "counter_mode": counter_mode,
        "count_transition": count_transition,
        "duration_state": duration_state,
        "min_state_samples": max(
            1,
            min(20, int(raw.get("min_state_samples") or 2)),
        ),
        "min_state_duration_sec": _bounded_float(
            raw.get("min_state_duration_sec"),
            field="min_state_duration_sec",
            default=20.0 if counter_mode != "none" else 2.0,
            minimum=0.0,
            maximum=600.0,
        ),
        "merge_gap_sec": _bounded_float(
            raw.get("merge_gap_sec"),
            field="merge_gap_sec",
            default=15.0 if counter_mode != "none" else 3.0,
            minimum=0.0,
            maximum=600.0,
        ),
        "alert_after_sec": _bounded_float(
            raw.get("alert_after_sec"),
            field="alert_after_sec",
            default=0.0,
            minimum=0.0,
            maximum=604_800.0,
        ),
    }


def _normalize_requirements(
    requirements: Any,
    *,
    selected_channel_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    if requirements in (None, []):
        return []
    if not isinstance(requirements, Sequence) or isinstance(
        requirements, (str, bytes, bytearray)
    ):
        raise DeploymentWorkflowError("requirements must be a list")
    if len(requirements) > MAX_REQUIREMENT_PACKS:
        raise DeploymentWorkflowError(
            f"at most {MAX_REQUIREMENT_PACKS} requirement packs are supported"
        )
    allowed = set(int(item) for item in selected_channel_ids)
    normalized: List[Dict[str, Any]] = []
    for raw in requirements:
        if not isinstance(raw, Mapping):
            raise DeploymentWorkflowError("each requirement pack must be an object")
        channel_ids = _normalize_channel_ids(
            raw.get("channel_ids") or [],
            maximum=MAX_DEPLOYMENT_CHANNELS,
        )
        if any(channel_id not in allowed for channel_id in channel_ids):
            raise DeploymentWorkflowError(
                "requirement pack contains a channel outside the selected scope"
            )
        alerts_raw = raw.get("alerts") or []
        if not isinstance(alerts_raw, Sequence) or isinstance(
            alerts_raw, (str, bytes, bytearray)
        ):
            raise DeploymentWorkflowError("alerts must be a list")
        if len(alerts_raw) > MAX_ALERTS_PER_PACK:
            raise DeploymentWorkflowError(
                f"a requirement pack supports at most {MAX_ALERTS_PER_PACK} alerts"
            )
        alerts = [
            _normalize_alert(alert, index)
            for index, alert in enumerate(alerts_raw, start=1)
            if isinstance(alert, Mapping)
        ]
        unexpected_severity = str(
            raw.get("unexpected_severity") or "info"
        ).strip().lower()
        if unexpected_severity not in _SEVERITIES:
            raise DeploymentWorkflowError(
                f"unsupported unexpected_severity: {unexpected_severity}"
            )
        novelty = str(
            raw.get("novelty_sensitivity") or "balanced"
        ).strip().lower()
        if novelty not in _NOVELTY:
            raise DeploymentWorkflowError(
                f"unsupported novelty_sensitivity: {novelty}"
            )
        normalized.append(
            {
                "name": _bounded_text(raw.get("name") or "channel policy", 100),
                "channel_ids": channel_ids,
                "expected_routine": _bounded_text(
                    raw.get("expected_routine"), 800
                ),
                "unexpected_severity": unexpected_severity,
                "novelty_sensitivity": novelty,
                "alerts": alerts,
            }
        )
    return normalized


def _dedupe_requirement_packs(
    requirements: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Drop model-duplicated policy packs without merging channel scopes."""

    accepted: List[Dict[str, Any]] = []
    seen: List[tuple[tuple[tuple[str, str, str], ...], frozenset[int]]] = []
    warnings: List[str] = []
    for raw in requirements:
        pack = copy.deepcopy(dict(raw))
        name_key = str(pack.get("name") or "").strip().casefold().replace("_", " ")
        if name_key in {"quiet window", "consolidation window"}:
            warnings.append(
                f"ignored requirement pack {pack.get('name')!r}: quiet window is a separate field"
            )
            continue
        signature = tuple(
            sorted(
                (
                    str(alert.get("name") or "").strip().casefold(),
                    str(alert.get("positive_query") or "").strip().casefold(),
                    str(alert.get("contrast_query") or "").strip().casefold(),
                )
                for alert in (pack.get("alerts") or [])
                if isinstance(alert, Mapping)
            )
        )
        scope = frozenset(int(item) for item in (pack.get("channel_ids") or []))
        duplicate = bool(
            signature
            and any(
                signature == prior_signature and bool(scope & prior_scope)
                for prior_signature, prior_scope in seen
            )
        )
        if duplicate:
            warnings.append(
                f"ignored duplicated requirement pack {pack.get('name')!r} with overlapping channel scope"
            )
            continue
        accepted.append(pack)
        if signature:
            seen.append((signature, scope))
    return accepted, warnings


def _merge_requirement_corrections(
    existing: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge corrections by pack scope/name and alert name, preserving siblings."""

    merged = [copy.deepcopy(dict(pack)) for pack in existing]
    for raw_update in updates:
        update = copy.deepcopy(dict(raw_update))
        update_scope = tuple(sorted(int(item) for item in update.get("channel_ids") or []))
        update_name = str(update.get("name") or "").strip().casefold()
        update_alert_names = {
            str(alert.get("name") or "").strip().casefold()
            for alert in (update.get("alerts") or [])
            if isinstance(alert, Mapping)
        }
        match_index: Optional[int] = None
        for index, current in enumerate(merged):
            current_scope = tuple(
                sorted(int(item) for item in current.get("channel_ids") or [])
            )
            current_name = str(current.get("name") or "").strip().casefold()
            current_alert_names = {
                str(alert.get("name") or "").strip().casefold()
                for alert in (current.get("alerts") or [])
                if isinstance(alert, Mapping)
            }
            if current_scope == update_scope and (
                current_name == update_name
                or bool(current_alert_names & update_alert_names)
            ):
                match_index = index
                break
        if match_index is None:
            merged.append(update)
            continue
        current = merged[match_index]
        alerts_by_name = {
            str(alert.get("name") or "").strip().casefold(): copy.deepcopy(dict(alert))
            for alert in (current.get("alerts") or [])
            if isinstance(alert, Mapping)
        }
        for alert in (update.get("alerts") or []):
            if not isinstance(alert, Mapping):
                continue
            alerts_by_name[str(alert.get("name") or "").strip().casefold()] = (
                copy.deepcopy(dict(alert))
            )
        combined = copy.deepcopy(current)
        for key in (
            "name",
            "channel_ids",
            "expected_routine",
            "unexpected_severity",
            "novelty_sensitivity",
        ):
            if update.get(key) not in (None, "", []):
                combined[key] = copy.deepcopy(update.get(key))
        combined["alerts"] = list(alerts_by_name.values())
        merged[match_index] = combined
    return merged


def _normalize_quiet_window(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, {}):
        return None
    if not isinstance(value, Mapping):
        raise DeploymentWorkflowError("quiet_window must be an object")
    days = value.get("days", list(range(7)))
    if not isinstance(days, Sequence) or isinstance(
        days, (str, bytes, bytearray)
    ):
        raise DeploymentWorkflowError("quiet_window.days must be a list")
    normalized_days = sorted({int(item) for item in days})
    if not normalized_days or any(item < 0 or item > 6 for item in normalized_days):
        raise DeploymentWorkflowError(
            "quiet_window.days must contain 0 (Monday) through 6"
        )
    candidate = {
        "enabled": bool(value.get("enabled", True)),
        "timezone": _bounded_text(value.get("timezone") or "UTC", 80),
        "start_local": _bounded_text(value.get("start_local") or "01:00", 5),
        "end_local": _bounded_text(value.get("end_local") or "05:00", 5),
        "days": normalized_days,
        "max_deferral_seconds": _bounded_float(
            value.get("max_deferral_seconds"),
            field="max_deferral_seconds",
            default=86_400.0,
            minimum=60.0,
            maximum=604_800.0,
        ),
    }
    try:
        return QuietWindowSchedule.from_mapping(candidate).as_dict()
    except ValueError as exc:
        raise DeploymentWorkflowError(str(exc)) from exc


def _normalize_deployment_profile(value: Any) -> str:
    profile = str(value or "general").strip().lower()
    if profile not in _DEPLOYMENT_PROFILES:
        raise DeploymentWorkflowError(f"unsupported deployment_profile: {profile}")
    return profile


def _normalize_channel_roles(
    value: Any,
    *,
    selected_channel_ids: Sequence[int],
    deployment_profile: str,
) -> List[Dict[str, Any]]:
    if value in (None, []):
        return []
    if deployment_profile != "maritime":
        raise DeploymentWorkflowError("channel_roles require deployment_profile=maritime")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DeploymentWorkflowError("channel_roles must be a list")
    selected = set(int(channel_id) for channel_id in selected_channel_ids)
    roles: List[Dict[str, Any]] = []
    seen = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise DeploymentWorkflowError("each channel role must be an object")
        channel_id = _positive_int(raw.get("channel_id"), "channel_id")
        if channel_id not in selected:
            raise DeploymentWorkflowError("channel role is outside the selected scope")
        if channel_id in seen:
            raise DeploymentWorkflowError("each selected channel may have only one role")
        role = str(raw.get("role") or "").strip().lower()
        if role not in MARITIME_CHANNEL_ROLES:
            raise DeploymentWorkflowError(f"unsupported maritime channel role: {role}")
        seen.add(channel_id)
        roles.append(
            {
                "channel_id": channel_id,
                "role": role,
                "label": _bounded_text(raw.get("label") or maritime_role_label(role), 100),
                "location": _bounded_text(raw.get("location"), 160),
            }
        )
    return roles


def _next_action(state: Mapping[str, Any]) -> str:
    stage = str(state.get("stage") or "inventory")
    return {
        "inventory": "configure_scope",
        "scope_configured": "survey",
        "surveyed": "collect_requirements",
        "requirements_partial": "collect_requirements",
        "requirements_configured": "preview_plan",
        "plan_ready": "apply_in_ui",
        "applied": "wait_for_commissioning",
        "commissioning_pending": "wait_for_commissioning",
        "commissioned": "review_commissioning",
    }.get(stage, "status")


def compact_deployment_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    surveys = state.get("surveys") if isinstance(state.get("surveys"), list) else []
    plan = state.get("plan") if isinstance(state.get("plan"), Mapping) else {}
    commissioning = (
        state.get("commissioning")
        if isinstance(state.get("commissioning"), Mapping)
        else {}
    )
    selected_channel_ids = [
        int(item) for item in (state.get("selected_channel_ids") or [])
    ]
    covered_requirement_ids = {
        int(channel_id)
        for pack in (state.get("requirements") or [])
        if isinstance(pack, Mapping)
        for channel_id in (pack.get("channel_ids") or [])
    }
    missing_requirement_ids = [
        channel_id
        for channel_id in selected_channel_ids
        if channel_id not in covered_requirement_ids
    ]
    return {
        "deployment_id": state.get("deployment_id"),
        "version": state.get("version"),
        "stage": state.get("stage"),
        "next_action": _next_action(state),
        "deployment_profile": state.get("deployment_profile") or "general",
        "starter_policy_mode": state.get("starter_policy_mode") or "none",
        "target_channel_count": state.get("target_channel_count"),
        "selected_channel_ids": list(state.get("selected_channel_ids") or []),
        "channel_roles": copy.deepcopy(list(state.get("channel_roles") or [])),
        "groups": copy.deepcopy(list(state.get("groups") or [])),
        "available_channels": copy.deepcopy(list(state.get("available_channels") or []))[:16],
        "survey_count": len(surveys),
        "surveyed_channel_ids": [
            int(row.get("channel_id"))
            for row in surveys
            if isinstance(row, Mapping) and row.get("channel_id") is not None
        ],
        "requirement_pack_count": len(state.get("requirements") or []),
        "missing_requirement_channel_ids": missing_requirement_ids,
        "requirement_warnings": list(state.get("requirement_warnings") or [])[:8],
        "quiet_window": copy.deepcopy(state.get("quiet_window")),
        "plan_summary": {
            "channel_count": len(plan.get("channels") or []),
            "group_count": len(plan.get("groups") or []),
            "probe_count": len(plan.get("probes") or []),
            "counted_state_count": len(plan.get("counted_states") or []),
            "starts_live": bool(plan.get("start_live", True)) if plan else None,
        },
        "applied_at_ms": state.get("applied_at_ms"),
        "commissioning": {
            key: commissioning.get(key)
            for key in (
                "status",
                "due_at_ms",
                "started_at_ms",
                "completed_at_ms",
                "coverage_ready",
                "proposal_count",
                "last_error",
            )
            if commissioning.get(key) is not None
        },
        "updated_at_ms": state.get("updated_at_ms"),
    }


class ProtocolDeploymentStore:
    """Runtime-state backed workflow store with a test/dev memory fallback."""

    def __init__(self, runtime_state_store: Optional[Any] = None) -> None:
        self.runtime_state_store = runtime_state_store
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(deployment_id: str) -> str:
        normalized = str(deployment_id or "").strip()
        if not re.fullmatch(r"deploy-[a-f0-9]{12}", normalized):
            raise DeploymentWorkflowError("invalid deployment_id")
        return f"{DEPLOYMENT_KEY_PREFIX}{normalized}"

    def _load_raw(self, key: str) -> Optional[Dict[str, Any]]:
        loader = getattr(self.runtime_state_store, "load_state", None)
        if callable(loader):
            payload = loader(key)
            return copy.deepcopy(payload) if isinstance(payload, Mapping) else None
        payload = self._memory.get(key)
        return copy.deepcopy(payload) if isinstance(payload, Mapping) else None

    def _save_raw(self, key: str, payload: Mapping[str, Any]) -> None:
        saver = getattr(self.runtime_state_store, "save_state", None)
        if callable(saver):
            saver(key, dict(payload))
            return
        self._memory[key] = copy.deepcopy(dict(payload))

    def _index_ids(self) -> List[str]:
        payload = self._load_raw(DEPLOYMENT_INDEX_KEY) or {}
        return [
            str(item)
            for item in (payload.get("deployment_ids") or [])
            if re.fullmatch(r"deploy-[a-f0-9]{12}", str(item))
        ][:100]

    def _save_index(self, deployment_ids: Sequence[str]) -> None:
        self._save_raw(
            DEPLOYMENT_INDEX_KEY,
            {
                "version": DEPLOYMENT_STATE_VERSION,
                "deployment_ids": list(dict.fromkeys(deployment_ids))[:100],
                "updated_at_ms": int(time.time() * 1000),
            },
        )

    def save(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(dict(state))
        deployment_id = str(payload.get("deployment_id") or "")
        payload["version"] = DEPLOYMENT_STATE_VERSION
        payload["updated_at_ms"] = int(time.time() * 1000)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > 240_000:
            raise DeploymentWorkflowError("deployment state is too large")
        with self._lock:
            self._save_raw(self._key(deployment_id), payload)
            ids = [deployment_id] + [
                item for item in self._index_ids() if item != deployment_id
            ]
            self._save_index(ids)
        return copy.deepcopy(payload)

    def load(self, deployment_id: str) -> Dict[str, Any]:
        with self._lock:
            state = self._load_raw(self._key(deployment_id))
        if not isinstance(state, Mapping):
            raise DeploymentWorkflowError("deployment was not found")
        return copy.deepcopy(dict(state))

    def latest_unfinished(
        self,
        deployment_profile: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        requested_profile = (
            _normalize_deployment_profile(deployment_profile)
            if deployment_profile is not None
            else None
        )
        with self._lock:
            for deployment_id in self._index_ids():
                state = self._load_raw(self._key(deployment_id))
                if not isinstance(state, Mapping):
                    continue
                state_profile = str(state.get("deployment_profile") or "general")
                if requested_profile is not None and state_profile != requested_profile:
                    continue
                if str(state.get("stage") or "") not in {"commissioned", "cancelled"}:
                    return copy.deepcopy(dict(state))
        return None

    def list_states(self) -> List[Dict[str, Any]]:
        states: List[Dict[str, Any]] = []
        with self._lock:
            for deployment_id in self._index_ids():
                state = self._load_raw(self._key(deployment_id))
                if isinstance(state, Mapping):
                    states.append(copy.deepcopy(dict(state)))
        return states

    def start(
        self,
        available_channels: Sequence[Mapping[str, Any]],
        *,
        target_channel_count: int = MAX_DEPLOYMENT_CHANNELS,
        resume_latest: bool = True,
        deployment_profile: str = "general",
    ) -> Dict[str, Any]:
        profile = _normalize_deployment_profile(deployment_profile)
        target = max(1, min(MAX_DEPLOYMENT_CHANNELS, int(target_channel_count or 8)))
        if resume_latest:
            existing = self.latest_unfinished(profile)
            # A target supplied by the operator describes the new scope cap.
            # Never silently resume a draft that was created for a different
            # cap: this is especially confusing when a previous two-channel
            # dry run is followed by a four/eight-channel commissioning run.
            if (
                existing is not None
                and int(existing.get("target_channel_count") or MAX_DEPLOYMENT_CHANNELS)
                == target
            ):
                return existing
        channels: List[Dict[str, Any]] = []
        seen = set()
        for raw in available_channels:
            if not isinstance(raw, Mapping):
                continue
            try:
                channel_id = _positive_int(raw.get("id"), "channel id")
            except DeploymentWorkflowError:
                continue
            if channel_id in seen:
                continue
            seen.add(channel_id)
            channels.append(
                {
                    "id": channel_id,
                    "title": _bounded_text(
                        raw.get("title") or raw.get("name") or f"channel-{channel_id}",
                        160,
                    ),
                    "type": _bounded_text(raw.get("type"), 80),
                }
            )
        now_ms = int(time.time() * 1000)
        state = {
            "version": DEPLOYMENT_STATE_VERSION,
            "deployment_id": f"deploy-{uuid.uuid4().hex[:12]}",
            "stage": "inventory",
            "deployment_profile": profile,
            "starter_policy_mode": "none",
            "target_channel_count": target,
            "available_channels": channels[:100],
            "selected_channel_ids": [],
            "groups": [],
            "channel_roles": [],
            "surveys": [],
            "requirements": [],
            "requirement_warnings": [],
            "quiet_window": None,
            "plan": None,
            "commissioning": {"status": "not_scheduled"},
            "created_at_ms": now_ms,
            "updated_at_ms": now_ms,
        }
        return self.save(state)

    def configure(
        self,
        deployment_id: str,
        *,
        channel_ids: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Mapping[str, Any]]] = None,
        requirements: Optional[Sequence[Mapping[str, Any]]] = None,
        quiet_window: Optional[Mapping[str, Any]] = None,
        channel_roles: Optional[Sequence[Mapping[str, Any]]] = None,
        starter_policy_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self.load(deployment_id)
        selected = list(state.get("selected_channel_ids") or [])
        if channel_ids is not None:
            target = max(
                1,
                min(
                    MAX_DEPLOYMENT_CHANNELS,
                    int(state.get("target_channel_count") or MAX_DEPLOYMENT_CHANNELS),
                ),
            )
            selected = _normalize_channel_ids(channel_ids, maximum=target)
            available = {
                int(item.get("id"))
                for item in (state.get("available_channels") or [])
                if isinstance(item, Mapping) and item.get("id") is not None
            }
            if any(channel_id not in available for channel_id in selected):
                raise DeploymentWorkflowError(
                    "selected channels must come from the deployment inventory"
                )
            state["selected_channel_ids"] = selected
            state["groups"] = _normalize_groups(
                groups or [],
                selected_channel_ids=selected,
            )
            state["surveys"] = [
                row
                for row in (state.get("surveys") or [])
                if isinstance(row, Mapping)
                and int(row.get("channel_id") or 0) in set(selected)
            ]
            state["requirements"] = []
            state["requirement_warnings"] = []
            state["channel_roles"] = []
            state["plan"] = None
            state["stage"] = "scope_configured"
        elif groups is not None:
            if not selected:
                raise DeploymentWorkflowError("configure channel scope first")
            state["groups"] = _normalize_groups(
                groups,
                selected_channel_ids=selected,
            )
            state["plan"] = None

        if channel_roles is not None:
            if not selected:
                raise DeploymentWorkflowError("configure channel scope first")
            state["channel_roles"] = _normalize_channel_roles(
                channel_roles,
                selected_channel_ids=selected,
                deployment_profile=str(state.get("deployment_profile") or "general"),
            )
            state["plan"] = None
        if starter_policy_mode is not None:
            mode = str(starter_policy_mode or "none").strip().lower()
            if mode not in _STARTER_POLICY_MODES:
                raise DeploymentWorkflowError(f"unsupported starter_policy_mode: {mode}")
            if mode != "none" and str(state.get("deployment_profile") or "general") != "maritime":
                raise DeploymentWorkflowError("starter shadow policies require maritime deployment")
            state["starter_policy_mode"] = mode
            state["plan"] = None
            if mode == "shadow" and selected and state.get("surveys"):
                state["stage"] = "requirements_configured"

        if requirements is not None:
            if not selected:
                raise DeploymentWorkflowError("configure channel scope first")
            normalized_requirements = _normalize_requirements(
                requirements,
                selected_channel_ids=selected,
            )
            normalized_requirements, warnings = _dedupe_requirement_packs(
                normalized_requirements
            )
            if (
                str(state.get("stage") or "")
                in {
                    "requirements_partial",
                    "requirements_configured",
                    "plan_ready",
                }
                and state.get("requirements")
            ):
                merged_requirements = _merge_requirement_corrections(
                    [
                        pack
                        for pack in (state.get("requirements") or [])
                        if isinstance(pack, Mapping)
                    ],
                    normalized_requirements,
                )
                normalized_requirements, merge_warnings = _dedupe_requirement_packs(
                    merged_requirements
                )
                warnings = list(state.get("requirement_warnings") or []) + warnings + merge_warnings
            state["requirements"] = normalized_requirements
            state["requirement_warnings"] = list(dict.fromkeys(warnings))[:16]
            state["plan"] = None
            covered = {
                int(channel_id)
                for pack in state["requirements"]
                for channel_id in (pack.get("channel_ids") or [])
            }
            missing = [channel_id for channel_id in selected if channel_id not in covered]
            if state["requirements"] and not missing:
                state["stage"] = "requirements_configured"
            elif state["requirements"]:
                state["stage"] = "requirements_partial"
            else:
                state["stage"] = "surveyed"
        if quiet_window is not None:
            state["quiet_window"] = _normalize_quiet_window(quiet_window)
            state["plan"] = None
        return self.save(state)

    def record_survey(
        self,
        deployment_id: str,
        survey_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        state = self.load(deployment_id)
        selected = set(int(item) for item in state.get("selected_channel_ids") or [])
        if not selected:
            raise DeploymentWorkflowError("configure channel scope before survey")
        surveys: List[Dict[str, Any]] = []
        for raw in survey_result.get("channels") or []:
            if not isinstance(raw, Mapping):
                continue
            channel_id = int(raw.get("channel_id") or 0)
            if channel_id not in selected:
                continue
            surveys.append(
                {
                    "channel_id": channel_id,
                    "title": _bounded_text(raw.get("title"), 160),
                    "sample_count": int(raw.get("sample_count") or 0),
                    "duration_sec": float(raw.get("duration_sec") or 0.0),
                    "survey": _bounded_text(raw.get("survey"), 2_000),
                    "error": _bounded_text(raw.get("error"), 300),
                }
            )
        state["surveys"] = surveys
        state["stage"] = "surveyed"
        state["plan"] = None
        return self.save(state)

    def build_plan(
        self,
        deployment_id: str,
        *,
        start_live: bool = True,
        commissioning_after_minutes: int = 15,
        probe_pos_floor: float = 0.05,
        probe_margin: float = 0.02,
        probe_embedding_space: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self.load(deployment_id)
        selected = [int(item) for item in state.get("selected_channel_ids") or []]
        requirements = list(state.get("requirements") or [])
        deployment_profile = str(state.get("deployment_profile") or "general")
        channel_roles = {
            int(row.get("channel_id")): dict(row)
            for row in (state.get("channel_roles") or [])
            if isinstance(row, Mapping) and row.get("channel_id") is not None
        }
        if not selected:
            raise DeploymentWorkflowError("deployment has no selected channels")
        if deployment_profile == "maritime":
            missing_roles = [channel_id for channel_id in selected if channel_id not in channel_roles]
            if missing_roles:
                raise DeploymentWorkflowError(
                    "assign a maritime role to every selected channel: "
                    + ", ".join(str(channel_id) for channel_id in missing_roles)
                )
            if str(state.get("starter_policy_mode") or "none") == "shadow":
                starter_requirements: List[Dict[str, Any]] = []
                for channel_id in selected:
                    starter = maritime_requirement(
                        str(channel_roles[channel_id].get("role") or ""),
                        channel_id,
                    )
                    normalized = _normalize_requirements(
                        [starter],
                        selected_channel_ids=selected,
                    )[0]
                    normalized["starter_policy"] = True
                    starter_requirements.append(normalized)
                # Operator-authored requirements win the four-probe budget;
                # starter watches fill only the remaining slots.
                requirements.extend(starter_requirements)
        if not requirements:
            raise DeploymentWorkflowError(
                "collect operator alert/routine requirements before preview"
            )
        covered_requirement_ids = {
            int(channel_id)
            for pack in requirements
            if isinstance(pack, Mapping)
            for channel_id in (pack.get("channel_ids") or [])
        }
        missing_requirement_ids = [
            channel_id
            for channel_id in selected
            if channel_id not in covered_requirement_ids
        ]
        if missing_requirement_ids:
            raise DeploymentWorkflowError(
                "collect requirements for every selected channel before preview: "
                + ", ".join(str(item) for item in missing_requirement_ids)
            )
        requirement_by_channel: Dict[int, List[Dict[str, Any]]] = {
            channel_id: [] for channel_id in selected
        }
        for pack in requirements:
            if not isinstance(pack, Mapping):
                continue
            for channel_id in pack.get("channel_ids") or []:
                if int(channel_id) in requirement_by_channel:
                    requirement_by_channel[int(channel_id)].append(dict(pack))

        channel_plans: List[Dict[str, Any]] = []
        probes: List[Dict[str, Any]] = []
        counted_states: List[Dict[str, Any]] = []
        channel_probe_counts: Dict[int, int] = {channel_id: 0 for channel_id in selected}
        for channel_id in selected:
            packs = requirement_by_channel.get(channel_id) or []
            lines = [
                f"EVA Protocol Deploy profile {deployment_id} for channel {channel_id}.",
                "This is an action policy, not evidence about the current scene.",
            ]
            for pack in packs:
                routine = str(pack.get("expected_routine") or "").strip()
                if routine:
                    lines.append(f"Expected visible routine: {routine}")
                lines.append(
                    "Unexpected visible activity: "
                    f"novelty={pack.get('novelty_sensitivity')}; "
                    f"default severity={pack.get('unexpected_severity')}."
                )
                for alert in pack.get("alerts") or []:
                    if not isinstance(alert, Mapping):
                        continue
                    lines.append(
                        f"- [{alert.get('severity')}] {alert.get('name')}: "
                        f"{alert.get('description')}"
                    )
                    positive = str(alert.get("positive_query") or "").strip()
                    contrast = str(alert.get("contrast_query") or "").strip()
                    probe_payload: Optional[Dict[str, Any]] = None
                    if (
                        positive
                        and contrast
                        and channel_probe_counts.get(channel_id, 0) < 4
                    ):
                        probe_name = (
                            f"{_slug(alert.get('name'))} [{deployment_id[-6:]}]"
                        )
                        probe_payload = {
                            "name": probe_name,
                            "channel_id": channel_id,
                            "positives": [positive],
                            "negatives": [contrast],
                            "pos_floor": float(probe_pos_floor),
                            "margin": max(0.0, float(probe_margin)),
                            "top_k": 6,
                            "window_sec": 300.0,
                            "severity": str(alert.get("severity") or "normal"),
                            "bookmark": False,
                            "enabled": True,
                            "origin": "agent",
                            "attention_only": bool(pack.get("starter_policy")),
                            "starter_policy": bool(pack.get("starter_policy")),
                            "deployment_id": deployment_id,
                        }
                        if probe_embedding_space:
                            probe_payload["embedding_space"] = copy.deepcopy(
                                dict(probe_embedding_space)
                            )
                        probes.append(probe_payload)
                        channel_probe_counts[channel_id] = (
                            channel_probe_counts.get(channel_id, 0) + 1
                        )
                    counter_mode = str(alert.get("counter_mode") or "none")
                    if counter_mode != "none":
                        metric_id = _stable_metric_id(
                            deployment_id,
                            channel_id,
                            str(alert.get("name") or "metric"),
                        )
                        metric = {
                            "id": metric_id,
                            "deployment_id": deployment_id,
                            "name": str(alert.get("name") or metric_id),
                            "channel_id": channel_id,
                            "subject_query": str(alert.get("name") or ""),
                            "positive_state_query": positive,
                            "negative_state_query": contrast,
                            "positive_label": str(
                                alert.get("positive_label") or "positive"
                            ),
                            "negative_label": str(
                                alert.get("negative_label") or "negative"
                            ),
                            "counter_mode": counter_mode,
                            "count_transition": str(
                                alert.get("count_transition")
                                or "positive_to_negative"
                            ),
                            "duration_state": str(
                                alert.get("duration_state") or "positive"
                            ),
                            "min_state_samples": int(
                                alert.get("min_state_samples") or 2
                            ),
                            "min_state_duration_sec": float(
                                alert.get("min_state_duration_sec") or 20.0
                            ),
                            "merge_gap_sec": float(
                                alert.get("merge_gap_sec") or 15.0
                            ),
                            "alert_after_sec": float(
                                alert.get("alert_after_sec") or 0.0
                            ),
                            "severity": str(alert.get("severity") or "normal"),
                            "enabled": True,
                        }
                        counted_states.append(metric)
                        if probe_payload is not None:
                            probe_payload["metric_profile_id"] = metric_id
                        lines.append(
                            "  Counted-state metric "
                            f"{metric_id}: count={metric['count_transition']}; "
                            f"duration={metric['duration_state']}; unknown/no-coverage "
                            "must remain separate."
                        )
            channel_plan: Dict[str, Any] = {
                "channel_id": channel_id,
                "alert_policy_prompt": "\n".join(lines),
                "novelty_profiles": [
                        {
                            "name": pack.get("name"),
                            "sensitivity": pack.get("novelty_sensitivity"),
                            "unexpected_severity": pack.get(
                                "unexpected_severity"
                            ),
                        }
                        for pack in packs
                    ],
            }
            if deployment_profile == "maritime":
                role = channel_roles[channel_id]
                role_name = str(role.get("role") or "")
                location = str(role.get("location") or "").strip()
                role_card = (
                    f"\n\nChannel operating card: role={role_name}; "
                    f"location={location or 'operator confirmation pending'}. "
                    "PTZ presets and spatial zones are separate scene epochs and must not share absence claims."
                )
                channel_plan.update(
                    {
                        "channel_role": role_name,
                        "channel_location": location,
                        "stream_system_prompt": MARITIME_L0_PROMPT + role_card,
                        "rollup_prompts": copy.deepcopy(MARITIME_ROLLUP_PROMPTS),
                        "coverage_contract": "ptz_scene_epoch_v1",
                    }
                )
            channel_plans.append(channel_plan)
        plan = {
            "version": 1,
            "deployment_id": deployment_id,
            "deployment_profile": deployment_profile,
            "starter_policy_mode": state.get("starter_policy_mode") or "none",
            "channels": channel_plans,
            "groups": copy.deepcopy(list(state.get("groups") or [])),
            "probes": probes[: MAX_DEPLOYMENT_CHANNELS * 4],
            "counted_states": counted_states[: MAX_DEPLOYMENT_CHANNELS * 4],
            "quiet_window": copy.deepcopy(state.get("quiet_window")),
            "start_live": bool(start_live),
            "commissioning_after_minutes": max(
                1, min(120, int(commissioning_after_minutes or 15))
            ),
            "generated_at_ms": int(time.time() * 1000),
        }
        state["plan"] = plan
        state["stage"] = "plan_ready"
        return self.save(state)

    def save_counted_profiles(
        self,
        profiles: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        with self._lock:
            current = self._load_raw(COUNTED_STATE_PROFILES_KEY) or {}
            by_id = {
                str(item.get("id")): dict(item)
                for item in (current.get("profiles") or [])
                if isinstance(item, Mapping) and item.get("id")
            }
            for profile in profiles:
                profile_id = str(profile.get("id") or "").strip()
                if profile_id:
                    by_id[profile_id] = copy.deepcopy(dict(profile))
            payload = {
                "version": 1,
                "profiles": list(by_id.values())[:256],
                "updated_at_ms": int(time.time() * 1000),
            }
            self._save_raw(COUNTED_STATE_PROFILES_KEY, payload)
        return copy.deepcopy(payload["profiles"])

    def list_counted_profiles(
        self,
        *,
        channel_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            payload = self._load_raw(COUNTED_STATE_PROFILES_KEY) or {}
        rows = [
            copy.deepcopy(dict(item))
            for item in (payload.get("profiles") or [])
            if isinstance(item, Mapping)
        ]
        if channel_id is not None:
            rows = [
                row
                for row in rows
                if int(row.get("channel_id") or 0) == int(channel_id)
            ]
        return rows

    def mark_applied(
        self,
        deployment_id: str,
        *,
        receipt: Mapping[str, Any],
    ) -> Dict[str, Any]:
        state = self.load(deployment_id)
        now_ms = int(time.time() * 1000)
        plan = state.get("plan") if isinstance(state.get("plan"), Mapping) else {}
        delay_minutes = int(plan.get("commissioning_after_minutes") or 15)
        state["stage"] = "commissioning_pending"
        state["applied_at_ms"] = now_ms
        state["apply_receipt"] = copy.deepcopy(dict(receipt))
        state["commissioning"] = {
            "status": "pending",
            "due_at_ms": now_ms + delay_minutes * 60_000,
            "coverage_ready": False,
        }
        return self.save(state)

    def record_commissioning(
        self,
        deployment_id: str,
        result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        state = self.load(deployment_id)
        commissioning = copy.deepcopy(dict(result))
        commissioning.setdefault("completed_at_ms", int(time.time() * 1000))
        commissioning.setdefault("status", "complete")
        commissioning["proposal_count"] = len(
            commissioning.get("proposals") or []
        )
        state["commissioning"] = commissioning
        state["stage"] = (
            "commissioned"
            if commissioning.get("status") == "complete"
            else "commissioning_pending"
        )
        return self.save(state)


def aggregate_counted_state_metric(
    profile: Mapping[str, Any],
    transition_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Aggregate stable segments independently from delivered alert count."""

    transitions = [
        dict(item)
        for item in (transition_result.get("transitions") or [])
        if isinstance(item, Mapping)
    ]
    segments = [
        dict(item)
        for item in (transition_result.get("segments") or [])
        if isinstance(item, Mapping)
    ]
    positive_label = str(profile.get("positive_label") or "positive")
    negative_label = str(profile.get("negative_label") or "negative")
    transition_mode = str(
        profile.get("count_transition") or "positive_to_negative"
    )
    count = 0
    for transition in transitions:
        before = str(transition.get("from_state") or "")
        after = str(transition.get("to_state") or "")
        if transition_mode == "any" and before != after:
            count += 1
        elif (
            transition_mode == "positive_to_negative"
            and before == positive_label
            and after == negative_label
        ):
            count += 1
        elif (
            transition_mode == "negative_to_positive"
            and before == negative_label
            and after == positive_label
        ):
            count += 1

    duration_by_state: Dict[str, float] = {}
    for segment in segments:
        state = str(segment.get("state") or "unknown")
        duration_by_state[state] = duration_by_state.get(state, 0.0) + max(
            0.0, float(segment.get("duration_sec") or 0.0)
        )
    duration_state = str(profile.get("duration_state") or "positive")
    duration_label = (
        positive_label
        if duration_state == "positive"
        else negative_label
        if duration_state == "negative"
        else duration_state
    )
    measured = sum(duration_by_state.values())
    window = transition_result.get("time_window")
    requested = (
        float(window.get("duration_sec") or 0.0)
        if isinstance(window, Mapping)
        else 0.0
    )
    unknown = max(0.0, requested - measured) if requested > 0 else duration_by_state.get("unknown", 0.0)
    duration_sec = round(duration_by_state.get(duration_label, 0.0), 3)
    unknown_sec = round(unknown, 3)
    if duration_sec >= 3600:
        duration_human = f"{duration_sec / 3600.0:.2f} h"
    elif duration_sec >= 60:
        duration_human = f"{duration_sec / 60.0:.1f} min"
    else:
        duration_human = f"{duration_sec:.1f} s"
    note = (
        "Counts and dwell are sampled visual-state estimates. Alert cooldown "
        "and deduplication do not affect this episode count; unknown/no-coverage "
        "time is reported separately."
    )
    return {
        "metric_id": profile.get("id"),
        "name": profile.get("name"),
        "metric_name": profile.get("name"),
        "channel_id": profile.get("channel_id"),
        "counter_mode": profile.get("counter_mode"),
        "count_transition": transition_mode,
        "transition_count": count,
        "event_count": count,
        "duration_state": duration_label,
        "duration_sec": duration_sec,
        "duration_human": duration_human,
        "duration_by_state_sec": {
            key: round(value, 3)
            for key, value in sorted(duration_by_state.items())
        },
        "unknown_or_uncovered_sec": unknown_sec,
        "unknown_duration_sec": unknown_sec,
        "coverage": copy.deepcopy(transition_result.get("coverage")),
        "frame_count": transition_result.get("frame_count"),
        "operator_note": note,
        "notes": note,
        "transitions": transitions[:24],
        "segments": segments[:24],
    }
