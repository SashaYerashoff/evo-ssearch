"""Probe provenance and operator-defined channel grouping for the Probes tab.

Two concerns live here, both deliberately free of Flask, the embedding runtime,
and the agent tool surface so they can be unit-tested in isolation:

``origin``
    Who created a probe: an operator, the agent (after operator approval), or
    the automatic VLM-alert follow-up lifecycle.  Before this module the only
    machine-readable distinction was ``temporary``, which is set exclusively by
    :mod:`alert_probe_lifecycle`; an agent-created probe became indistinguishable
    from an operator-created one the moment its approval was applied.

    Note the deliberate naming overlap: ``AlertProbe.origin`` in
    :mod:`alert_probe_lifecycle` is a *lineage guard* whose value is
    ``"vlm_alert"``, and it is written into the probe payload as ``source``.
    The ``origin`` key on a stored probe is this module's authorship field and
    only ever holds one of :data:`PROBE_ORIGINS`.

``ChannelGroupStore``
    Operator-authored grouping of channels ("Perimeter", "Berth 3").  Luxriot
    exposes no group concept through the connector, so grouping is EVA-side
    state.  It is intentionally file-backed rather than a new archive table:
    production runs a single Gunicorn worker, the data is small operator UI
    organisation, and this keeps the feature off the tenant schema while an
    unrelated schema revision is in flight.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

ORIGIN_OPERATOR = "operator"
ORIGIN_AGENT = "agent"
ORIGIN_AUTO = "auto"

#: Every authorship value the UI, the filters, and the stores may see.
PROBE_ORIGINS = (ORIGIN_OPERATOR, ORIGIN_AGENT, ORIGIN_AUTO)

#: Legacy ``source`` values written before ``origin`` existed that mean "auto".
_LEGACY_AUTO_SOURCES = frozenset({"vlm_alert", "alert_probe", "auto"})

#: Fields that describe how a probe came to exist.  ``_build_probe_payload``
#: rebuilds a probe from the request body, so these must be carried over from
#: the stored probe or an ordinary operator edit would silently strip a probe's
#: authorship and its alert lineage.
PROBE_PROVENANCE_FIELDS = (
    "origin",
    "origin_meta",
    "temporary",
    "source",
    "generation",
    "generated_fallback",
    "confidence",
    "parent_alert_id",
    "parent_alert_title",
    "parent_alert_description",
    "parent_alert_timestamp_ms",
    "created_at_ms",
    "expires_at_ms",
    "cooldown_until_ms",
    "runtime_status",
    "lifecycle",
)

MAX_GROUPS = 128
MAX_GROUP_NAME_LEN = 80
MAX_CHANNELS_PER_GROUP = 512


class ChannelGroupError(ValueError):
    """Raised when a channel group create/update request is malformed."""


def normalize_probe_origin(probe: Mapping[str, Any]) -> str:
    """Return the authorship of ``probe``, backfilling probes stored earlier.

    Probes written before this field existed carry no ``origin``.  The only
    reliable legacy signal is the alert lifecycle's ``temporary`` flag and its
    ``parent_alert_id``/``source`` lineage; everything else predates the agent
    approval path being distinguishable and is attributed to the operator.
    """

    if not isinstance(probe, Mapping):
        return ORIGIN_OPERATOR
    raw = str(probe.get("origin") or "").strip().lower()
    if raw in PROBE_ORIGINS:
        return raw
    if probe.get("temporary") or probe.get("parent_alert_id"):
        return ORIGIN_AUTO
    if str(probe.get("source") or "").strip().lower() in _LEGACY_AUTO_SOURCES:
        return ORIGIN_AUTO
    return ORIGIN_OPERATOR


def coerce_probe_origin(value: Any, *, default: str = ORIGIN_OPERATOR) -> str:
    """Return ``value`` if it is a known origin, else ``default``."""

    candidate = str(value or "").strip().lower()
    if candidate in PROBE_ORIGINS:
        return candidate
    return default if default in PROBE_ORIGINS else ORIGIN_OPERATOR


def annotate_probe_origin(probe: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``probe`` with a resolved ``origin`` key."""

    annotated = dict(probe or {})
    annotated["origin"] = normalize_probe_origin(probe)
    return annotated


def carry_probe_provenance(
    target: Dict[str, Any],
    existing: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Copy provenance/lineage fields from ``existing`` into ``target`` in place.

    Only fields absent from ``target`` are filled, so an explicit new value
    (for example the agent stamping ``origin`` on first save) still wins.
    """

    if not isinstance(existing, Mapping):
        return target
    for field in PROBE_PROVENANCE_FIELDS:
        if field in target and target[field] is not None:
            continue
        if field in existing:
            target[field] = copy.deepcopy(existing[field])
    return target


def _coerce_channel_ids(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ChannelGroupError("channel_ids must be a list of channel ids")
    seen: Dict[int, None] = {}
    for item in value:
        try:
            channel_id = int(str(item).strip())
        except (TypeError, ValueError) as exc:
            raise ChannelGroupError(f"invalid channel id: {item!r}") from exc
        if channel_id <= 0:
            raise ChannelGroupError(f"invalid channel id: {item!r}")
        seen.setdefault(channel_id, None)
    if len(seen) > MAX_CHANNELS_PER_GROUP:
        raise ChannelGroupError(
            f"a group may hold at most {MAX_CHANNELS_PER_GROUP} channels"
        )
    return list(seen)


def _coerce_group_name(value: Any) -> str:
    name = " ".join(str(value or "").split()).strip()
    if not name:
        raise ChannelGroupError("group name must not be empty")
    if len(name) > MAX_GROUP_NAME_LEN:
        raise ChannelGroupError(
            f"group name must be at most {MAX_GROUP_NAME_LEN} characters"
        )
    return name


class ChannelGroupStore:
    """File-backed operator-defined channel groups.

    A channel belongs to at most one group.  Assigning it to a new group
    removes it from whichever group held it before, so the board never renders
    the same channel twice.
    """

    backend = "json"

    def __init__(self, path: Union[str, Path] = "probe_channel_groups.json") -> None:
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data: Dict[str, Any] = {"groups": []}
        self._load()

    def _load(self) -> None:
        with self.lock:
            if not self.path.exists():
                return
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {"groups": []}
                return
            groups = loaded.get("groups") if isinstance(loaded, Mapping) else None
            self.data = {"groups": list(groups) if isinstance(groups, list) else []}

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def _groups_locked(self) -> List[Dict[str, Any]]:
        groups = self.data.get("groups")
        if not isinstance(groups, list):
            groups = []
            self.data["groups"] = groups
        return groups

    def list_groups(self) -> List[Dict[str, Any]]:
        with self.lock:
            groups = [
                copy.deepcopy(group)
                for group in self._groups_locked()
                if isinstance(group, Mapping)
            ]
        groups.sort(
            key=lambda group: (
                int(group.get("position") or 0),
                str(group.get("name") or "").casefold(),
            )
        )
        return groups

    def group_id_by_channel(self) -> Dict[int, str]:
        """Return ``{channel_id: group_id}`` for fast board assembly."""

        assignment: Dict[int, str] = {}
        for group in self.list_groups():
            group_id = str(group.get("id") or "")
            for channel_id in group.get("channel_ids") or []:
                try:
                    assignment[int(channel_id)] = group_id
                except (TypeError, ValueError):
                    continue
        return assignment

    def upsert_group(
        self,
        *,
        group_id: Optional[str] = None,
        name: Optional[str] = None,
        channel_ids: Any = None,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized_id = str(group_id or "").strip()
        now_ms = int(time.time() * 1000.0)
        with self.lock:
            groups = self._groups_locked()
            existing_index: Optional[int] = None
            for index, group in enumerate(groups):
                if isinstance(group, Mapping) and str(group.get("id") or "") == normalized_id:
                    existing_index = index
                    break
            if normalized_id and existing_index is None:
                raise ChannelGroupError("group not found")
            if existing_index is None and len(groups) >= MAX_GROUPS:
                raise ChannelGroupError(f"at most {MAX_GROUPS} groups are supported")

            current = dict(groups[existing_index]) if existing_index is not None else {}
            if name is not None or existing_index is None:
                current["name"] = _coerce_group_name(
                    name if name is not None else current.get("name")
                )
            if channel_ids is not None or existing_index is None:
                current["channel_ids"] = _coerce_channel_ids(
                    channel_ids if channel_ids is not None else current.get("channel_ids")
                )
            if position is not None:
                try:
                    current["position"] = int(position)
                except (TypeError, ValueError) as exc:
                    raise ChannelGroupError("position must be an integer") from exc
            elif existing_index is None:
                current["position"] = len(groups)

            if existing_index is None:
                current["id"] = f"grp-{uuid.uuid4().hex[:12]}"
                current["created_at_ms"] = now_ms
            current["updated_at_ms"] = now_ms

            claimed = set(current.get("channel_ids") or [])
            for index, group in enumerate(groups):
                if index == existing_index or not isinstance(group, Mapping):
                    continue
                retained = [
                    channel_id
                    for channel_id in (group.get("channel_ids") or [])
                    if int(channel_id) not in claimed
                ]
                if len(retained) != len(group.get("channel_ids") or []):
                    updated = dict(group)
                    updated["channel_ids"] = retained
                    updated["updated_at_ms"] = now_ms
                    groups[index] = updated

            if existing_index is None:
                groups.append(current)
            else:
                groups[existing_index] = current
            self._save_locked()
            return copy.deepcopy(current)

    def delete_group(self, group_id: str) -> bool:
        normalized = str(group_id or "").strip()
        if not normalized:
            return False
        with self.lock:
            groups = self._groups_locked()
            retained = [
                group
                for group in groups
                if not (isinstance(group, Mapping) and str(group.get("id") or "") == normalized)
            ]
            if len(retained) == len(groups):
                return False
            self.data["groups"] = retained
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
