"""Shared incident commands for HTTP and the secured EVA agent surface.

The command layer owns orchestration between durable incident state and the
process-local attention lease.  It deliberately does not perform authorization:
HTTP guards and ``EvaAgentToolAdapter`` resolve tenant/channel authority before
calling it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from incident_attention import PromptBudgetError, compact_incident_context
from incident_presentation import (
    build_follow_result,
    build_incident_synopsis,
    classify_follow_heartbeat,
    semantic_timeline,
)
from incident_service import IncidentDraftAssembler, IncidentDraftRequest
from incident_store import IncidentRevisionConflict


_GENERIC_INCIDENT_KEYS = {
    "event",
    "transition",
    "state_transition",
    "alert",
    "vlm_alert",
    "vlm_structured_alert",
    "batch_state",
    "motion_peak",
}

_INCIDENT_COVER_ROLE_PRIORITY = {
    "apex": 0,
    "alert": 1,
    "onset": 2,
    "event": 3,
    "post": 4,
    "anchor": 5,
    "control": 6,
}
_LOGGER = logging.getLogger(__name__)


def _positive_ints(values: Any) -> list[int]:
    channels: list[int] = []
    for value in values or ():
        try:
            channel_id = int(value)
        except (TypeError, ValueError):
            continue
        if channel_id > 0 and channel_id not in channels:
            channels.append(channel_id)
    return channels


def incident_narrative(incident: Mapping[str, Any]) -> str:
    return str(build_incident_synopsis(incident).get("description") or "").strip()


def incident_storage_record(draft: Mapping[str, Any]) -> dict[str, Any]:
    bounds = (
        draft.get("time_bounds")
        if isinstance(draft.get("time_bounds"), Mapping)
        else {}
    )
    report = {
        "severity": str(draft.get("severity") or "info"),
        "source_title": str(draft.get("title") or "")[:200],
        "apex_ms": bounds.get("apex_ms"),
        "provenance": dict(draft.get("provenance") or {}),
        "qualia_digest": dict(draft.get("qualia_digest") or {}),
    }
    provisional = {
        "title": draft.get("title"),
        "timeline_refs": list(draft.get("timeline") or []),
        "qualia_refs": [dict(draft.get("qualia_digest") or {})],
        "coverage": dict(draft.get("coverage") or {}),
        "uncertainties": list(draft.get("uncertainties") or []),
        "report": report,
    }
    synopsis = build_incident_synopsis(provisional)
    report["summary"] = str(synopsis.get("description") or "")
    report["synopsis"] = synopsis
    record = {
        "state": "draft",
        "perception_state": "observed",
        "risk_state": "unknown",
        "case_state": "candidate",
        "attention_state": "inactive",
        "title": str(synopsis.get("title") or "Incident draft")[:200],
        "channel_ids": list(draft.get("channel_ids") or []),
        "possible_start_ms": bounds.get("possible_start_ms"),
        "observed_start_ms": bounds.get("observed_start_ms"),
        "observed_end_ms": bounds.get("observed_end_ms"),
        "possible_end_ms": bounds.get("possible_end_ms"),
        "anchor_ref": dict(draft.get("anchor") or {}),
        "timeline_refs": list(draft.get("timeline") or []),
        "evidence_refs": list(draft.get("evidence") or []),
        "qualia_refs": [dict(draft.get("qualia_digest") or {})],
        "coverage": dict(draft.get("coverage") or {}),
        "uncertainties": list(draft.get("uncertainties") or []),
        "report": report,
        "follow_policy": {},
    }
    identity_payload = json.dumps(
        {
            "channel_ids": record["channel_ids"],
            "possible_start_ms": record["possible_start_ms"],
            "observed_start_ms": record["observed_start_ms"],
            "observed_end_ms": record["observed_end_ms"],
            "possible_end_ms": record["possible_end_ms"],
            "anchor_ref": record["anchor_ref"],
            "timeline_refs": record["timeline_refs"],
            "evidence_refs": record["evidence_refs"],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    record["idempotency_key"] = (
        "incident-draft:" + hashlib.sha256(identity_payload).hexdigest()
    )
    return record


class IncidentCommandService:
    """Coordinate durable incident records with bounded runtime focus leases."""

    def __init__(
        self,
        incident_store: Any,
        detections_store: Any,
        attention_store: Any,
        focus_runtime: Any,
        *,
        wall_clock_ms: Callable[[], int] | None = None,
        draft_assembler_factory: Callable[..., Any] = IncidentDraftAssembler,
    ) -> None:
        self.incident_store = incident_store
        self.detections_store = detections_store
        self.attention_store = attention_store
        self.focus_runtime = focus_runtime
        self._wall_clock_ms = wall_clock_ms or (lambda: int(time.time() * 1000.0))
        self._draft_assembler_factory = draft_assembler_factory

    def build_draft(
        self,
        *,
        channel_id: int,
        anchor_detection_id: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> dict[str, Any]:
        return self._draft_assembler_factory(
            self.detections_store,
            self.attention_store,
        ).assemble(
            IncidentDraftRequest(
                channel_id=int(channel_id),
                anchor_detection_id=anchor_detection_id,
                since_ms=since_ms,
                until_ms=until_ms,
            )
        )

    def store_draft(
        self,
        draft: Mapping[str, Any],
        *,
        actor_id: str,
    ) -> dict[str, Any]:
        created = self.incident_store.create_incident(
            incident_storage_record(draft),
            actor_id=actor_id,
        )
        self._materialize_primary_episode(created, actor_id=actor_id)
        self._materialize_series_candidate(created, actor_id=actor_id)
        return created

    @staticmethod
    def _primary_semantic_key(incident: Mapping[str, Any]) -> str:
        for item in semantic_timeline(incident):
            key = str(item.get("semantic_key") or "").strip().lower()
            if key and key not in _GENERIC_INCIDENT_KEYS:
                return key[:160]
        return ""

    def _materialize_primary_episode(
        self,
        incident: Mapping[str, Any],
        *,
        actor_id: str | None,
    ) -> dict[str, Any] | None:
        appender = getattr(self.incident_store, "append_episode", None)
        if not callable(appender):
            return None
        incident_id = str(incident.get("id") or "").strip()
        possible_start = self._optional_int(incident.get("possible_start_ms"))
        if not incident_id or possible_start is None:
            return None
        observed_start = self._optional_int(incident.get("observed_start_ms"))
        observed_end = self._optional_int(incident.get("observed_end_ms"))
        possible_end = self._optional_int(incident.get("possible_end_ms"))
        semantic_key = self._primary_semantic_key(incident)
        start_for_duration = observed_start if observed_start is not None else possible_start
        duration_ms = (
            max(0, observed_end - start_for_duration)
            if observed_end is not None
            else 0
        )
        disposition = (
            "long_incident_candidate"
            if duration_ms >= 15 * 60 * 1000
            else "continuing_incident"
            if observed_end is None
            else "unclassified_keep"
        )
        episode_seed = f"{semantic_key or 'unverified'}:{start_for_duration}"
        episode_key = "episode:" + hashlib.sha256(episode_seed.encode("utf-8")).hexdigest()
        coverage = (
            dict(incident.get("coverage") or {})
            if isinstance(incident.get("coverage"), Mapping)
            else {}
        )
        coverage.update(
            {
                "scale_disposition": disposition,
                "duration_ms_at_materialization": duration_ms,
                "operator_review_required": True,
            }
        )
        payload = {
            "incident_id": incident_id,
            "idempotency_key": f"primary:{episode_key}",
            "episode_key": episode_key,
            "perception_state": str(incident.get("perception_state") or "unknown"),
            "semantic_key": semantic_key or None,
            "possible_start_ms": possible_start,
            "observed_start_ms": observed_start,
            "observed_end_ms": observed_end,
            "possible_end_ms": possible_end,
            "routine_before_ref": {},
            "routine_after_ref": {},
            "evidence_refs": [
                dict(item)
                for item in incident.get("evidence_refs") or []
                if isinstance(item, Mapping)
            ][:512],
            "coverage": coverage,
        }
        return appender(payload, actor_id=actor_id)

    def _materialize_series_candidate(
        self,
        incident: Mapping[str, Any],
        *,
        actor_id: str | None,
    ) -> dict[str, Any] | None:
        appender = getattr(self.incident_store, "append_relation", None)
        lister = getattr(self.incident_store, "list_incidents", None)
        if not callable(appender) or not callable(lister):
            return None
        semantic_key = self._primary_semantic_key(incident)
        channel_ids = _positive_ints(incident.get("channel_ids"))
        incident_id = str(incident.get("id") or "").strip()
        current_start = self._optional_int(
            incident.get("observed_start_ms") or incident.get("possible_start_ms")
        )
        if not semantic_key or not channel_ids or not incident_id or current_start is None:
            return None
        since_ms = max(0, current_start - 30 * 24 * 60 * 60 * 1000)
        candidates, _total = lister(
            channel_ids=channel_ids,
            since_ms=since_ms,
            until_ms=max(0, current_start - 1),
            limit=100,
            offset=0,
        )
        matches = [
            item
            for item in candidates
            if isinstance(item, Mapping)
            and str(item.get("id") or "") != incident_id
            and self._primary_semantic_key(item) == semantic_key
        ]
        if not matches:
            return None

        def prior_end(item: Mapping[str, Any]) -> int:
            return self._optional_int(
                item.get("possible_end_ms")
                or item.get("observed_end_ms")
                or item.get("observed_start_ms")
                or item.get("possible_start_ms")
            ) or 0

        prior = max(matches, key=prior_end)
        prior_id = str(prior.get("id") or "")
        gap_ms = max(0, current_start - prior_end(prior))
        relation_seed = f"{prior_id}:{incident_id}:{semantic_key}"
        relation_key = hashlib.sha256(relation_seed.encode("utf-8")).hexdigest()
        series_key = hashlib.sha256(
            f"{channel_ids[0]}:{semantic_key}".encode("utf-8")
        ).hexdigest()[:32]
        return appender(
            {
                "subject_incident_id": incident_id,
                "object_incident_id": prior_id,
                "idempotency_key": f"series:{relation_key}",
                "relation_type": "series_member",
                "relation_state": "candidate",
                "confidence": "medium",
                "rationale": (
                    "Exact semantic track recurred in a distinct, non-overlapping "
                    "incident window; operator confirmation is required."
                ),
                "payload": {
                    "series_key": series_key,
                    "semantic_key": semantic_key,
                    "gap_ms": gap_ms,
                    "automatic_merge": False,
                    "operator_review_required": True,
                },
            },
            actor_id=actor_id,
        )

    def ensure_temporal_projection(
        self,
        incident: Mapping[str, Any],
        *,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        """Backfill append-only episode/series ledgers without changing the case.

        This is safe to replay after process restart. Repository uniqueness and
        deterministic idempotency keys make concurrent maintenance workers
        converge on the same rows.
        """

        incident_id = str(incident.get("id") or "").strip()
        episode_lister = getattr(self.incident_store, "list_episodes", None)
        relation_lister = getattr(self.incident_store, "list_relations", None)
        if not incident_id or not callable(episode_lister):
            return {"supported": False, "episode_created": False, "relation_created": False}

        episodes, episode_total = episode_lister(incident_id, limit=1, offset=0)
        episode_created = False
        episode_ready = bool(
            self._optional_int(incident.get("possible_end_ms")) is not None
            or str(incident.get("perception_state") or "") == "ended"
            or str(incident.get("state") or "") in {"ended", "reported", "closed"}
        )
        if episode_ready and not episode_total and not episodes:
            episode_created = self._materialize_primary_episode(
                incident,
                actor_id=actor_id,
            ) is not None

        relation_created = False
        existing_series = False
        relations_rejected = 0
        if callable(relation_lister):
            relations, _relation_total = relation_lister(
                incident_id,
                limit=500,
                offset=0,
            )
            rejected_relation_ids = {
                str(
                    (item.get("payload") or {}).get("supersedes_relation_id")
                    or ""
                )
                for item in relations
                if isinstance(item, Mapping)
                and isinstance(item.get("payload"), Mapping)
                and str(item.get("relation_state") or "") == "rejected"
            }
            for item in relations:
                if (
                    not isinstance(item, Mapping)
                    or str(item.get("subject_incident_id") or "") != incident_id
                    or str(item.get("relation_type") or "") != "series_member"
                    or str(item.get("relation_state") or "") != "candidate"
                    or str(item.get("id") or "") in rejected_relation_ids
                ):
                    continue
                payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
                relation_semantic_key = str(payload.get("semantic_key") or "").strip().lower()
                if relation_semantic_key and relation_semantic_key not in _GENERIC_INCIDENT_KEYS:
                    continue
                rejection_idempotency = "reject-series:" + hashlib.sha256(
                    str(item.get("id") or "").encode("utf-8")
                ).hexdigest()
                appender = getattr(self.incident_store, "append_relation", None)
                if callable(appender):
                    appender(
                        {
                            "subject_incident_id": incident_id,
                            "object_incident_id": str(item.get("object_incident_id") or ""),
                            "idempotency_key": rejection_idempotency,
                            "relation_type": "series_member",
                            "relation_state": "rejected",
                            "confidence": "high",
                            "rationale": (
                                "Generic transport metadata cannot establish a semantic incident series."
                            ),
                            "payload": {
                                "supersedes_relation_id": str(item.get("id") or ""),
                                "semantic_key": relation_semantic_key or None,
                                "automatic_merge": False,
                            },
                        },
                        actor_id=actor_id,
                    )
                    rejected_relation_ids.add(str(item.get("id") or ""))
                    relations_rejected += 1
            existing_series = any(
                isinstance(item, Mapping)
                and str(item.get("subject_incident_id") or "") == incident_id
                and str(item.get("relation_type") or "") == "series_member"
                and str(item.get("relation_state") or "") in {"candidate", "confirmed"}
                and str(item.get("id") or "") not in rejected_relation_ids
                for item in relations
            )
        if not existing_series:
            relation_created = self._materialize_series_candidate(
                incident,
                actor_id=actor_id,
            ) is not None
        return {
            "supported": True,
            "episode_created": episode_created,
            "relation_created": relation_created,
            "relations_rejected": relations_rejected,
        }

    def create_draft(
        self,
        *,
        channel_id: int,
        actor_id: str,
        anchor_detection_id: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> dict[str, Any]:
        draft = self.build_draft(
            channel_id=channel_id,
            anchor_detection_id=anchor_detection_id,
            since_ms=since_ms,
            until_ms=until_ms,
        )
        return self.store_draft(draft, actor_id=actor_id)

    @staticmethod
    def draft_digest(draft: Mapping[str, Any]) -> str:
        """Stable binding used to prevent a preview/apply evidence race."""

        payload = json.dumps(
            incident_storage_record(draft),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, incident_id: str) -> dict[str, Any]:
        incident = self.incident_store.get_incident(incident_id)
        if not isinstance(incident, Mapping):
            raise LookupError("incident not found")
        return dict(incident)

    def temporal_context(self, incident: Mapping[str, Any]) -> dict[str, Any]:
        """Return the bounded, effective temporal-memory projection for operators.

        Episodes and relations are immutable ledgers.  A rejected relation does
        not delete the candidate it corrects, so this read projection removes
        superseded candidates while preserving correction counts for audit.  It
        never merges incidents and never mutates the ledgers.
        """

        incident_id = str(incident.get("id") or "").strip()
        if not incident_id:
            raise ValueError("incident id is required")
        episode_lister = getattr(self.incident_store, "list_episodes", None)
        relation_lister = getattr(self.incident_store, "list_relations", None)
        transition_lister = getattr(self.incident_store, "list_transitions", None)
        if not callable(episode_lister) or not callable(relation_lister):
            return {
                "supported": False,
                "incident_id": incident_id,
                "episodes": [],
                "episode_total": 0,
                "series_links": [],
                "relation_total": 0,
                "correction_count": 0,
                "lifecycle_history": [],
                "transition_total": 0,
            }

        raw_episodes, episode_total = episode_lister(
            incident_id,
            limit=250,
            offset=0,
        )
        raw_relations, relation_total = relation_lister(
            incident_id,
            limit=500,
            offset=0,
        )
        raw_transitions: list[Mapping[str, Any]] = []
        transition_total = 0
        if callable(transition_lister):
            loaded_transitions, transition_total = transition_lister(
                incident_id,
                limit=500,
                offset=0,
            )
            raw_transitions = [
                item for item in loaded_transitions if isinstance(item, Mapping)
            ]
        superseded_relation_ids = {
            str((item.get("payload") or {}).get("supersedes_relation_id") or "")
            for item in raw_relations
            if isinstance(item, Mapping)
            and str(item.get("relation_state") or "") in {"confirmed", "rejected"}
            and isinstance(item.get("payload"), Mapping)
        }
        superseded_relation_ids.discard("")

        episodes: list[dict[str, Any]] = []
        for raw in raw_episodes:
            if not isinstance(raw, Mapping):
                continue
            coverage = (
                dict(raw.get("coverage") or {})
                if isinstance(raw.get("coverage"), Mapping)
                else {}
            )
            semantic_key = str(raw.get("semantic_key") or "").strip().lower()
            if semantic_key in _GENERIC_INCIDENT_KEYS:
                semantic_key = ""
            episodes.append(
                {
                    "id": str(raw.get("id") or ""),
                    "episode_key": str(raw.get("episode_key") or ""),
                    "perception_state": str(raw.get("perception_state") or "unknown"),
                    "semantic_key": semantic_key or None,
                    "entity_key": raw.get("entity_key"),
                    "zone_key": raw.get("zone_key"),
                    "possible_start_ms": self._optional_int(raw.get("possible_start_ms")),
                    "observed_start_ms": self._optional_int(raw.get("observed_start_ms")),
                    "observed_end_ms": self._optional_int(raw.get("observed_end_ms")),
                    "possible_end_ms": self._optional_int(raw.get("possible_end_ms")),
                    "scale_disposition": str(
                        coverage.get("scale_disposition") or "unclassified_keep"
                    ),
                    "operator_review_required": coverage.get("operator_review_required") is not False,
                    "nested_context": coverage.get("nested_context") is True,
                    "composition_parent": coverage.get("composition_parent") is True,
                    "source_level": str(coverage.get("source_level") or "") or None,
                    "composition_id": str(coverage.get("composition_id") or "") or None,
                    "automatic_merge": coverage.get("automatic_merge") is True,
                    "evidence_count": len(
                        [
                            item
                            for item in raw.get("evidence_refs") or []
                            if isinstance(item, Mapping)
                        ]
                    ),
                }
            )

        series_links: list[dict[str, Any]] = []
        for raw in raw_relations:
            if not isinstance(raw, Mapping):
                continue
            relation_id = str(raw.get("id") or "")
            relation_state = str(raw.get("relation_state") or "")
            if (
                str(raw.get("relation_type") or "") != "series_member"
                or relation_state not in {"candidate", "confirmed"}
                or relation_id in superseded_relation_ids
            ):
                continue
            payload = (
                dict(raw.get("payload") or {})
                if isinstance(raw.get("payload"), Mapping)
                else {}
            )
            semantic_key = str(payload.get("semantic_key") or "").strip().lower()
            if not semantic_key or semantic_key in _GENERIC_INCIDENT_KEYS:
                continue
            subject_id = str(raw.get("subject_incident_id") or "")
            object_id = str(raw.get("object_incident_id") or "")
            related_id = object_id if subject_id == incident_id else subject_id
            series_links.append(
                {
                    "relation_id": relation_id,
                    "relation_state": relation_state,
                    "confidence": str(raw.get("confidence") or "unknown"),
                    "related_incident_id": related_id,
                    "direction": "prior" if subject_id == incident_id else "later",
                    "series_key": str(payload.get("series_key") or ""),
                    "semantic_key": semantic_key,
                    "gap_ms": max(0, self._optional_int(payload.get("gap_ms")) or 0),
                    "automatic_merge": False,
                    "operator_review_required": payload.get("operator_review_required") is not False,
                    "rationale": str(raw.get("rationale") or "")[:500],
                }
            )

        series_links.sort(
            key=lambda item: (
                int(item.get("gap_ms") or 0),
                str(item.get("related_incident_id") or ""),
            )
        )
        lifecycle_history = [
            {
                "id": str(item.get("id") or ""),
                "axis": str(item.get("axis") or "unknown"),
                "from_state": item.get("from_state"),
                "to_state": str(item.get("to_state") or "unknown"),
                "incident_revision": self._optional_int(item.get("incident_revision")),
                "transitioned_at_ms": self._optional_int(item.get("transitioned_at_ms")),
                "reason": str(item.get("reason") or "")[:500],
                "source_kind": str(item.get("source_kind") or "unknown"),
            }
            for item in raw_transitions[-100:]
        ]
        return {
            "supported": True,
            "incident_id": incident_id,
            "episodes": episodes,
            "episode_total": int(episode_total),
            "series_links": series_links,
            "relation_total": int(relation_total),
            "correction_count": len(superseded_relation_ids),
            "lifecycle_history": lifecycle_history,
            "transition_total": int(transition_total),
        }

    def review_series_relation(
        self,
        incident_id: str,
        relation_id: str,
        *,
        actor_id: str,
        action: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Append an operator correction for a candidate recurrence relation."""

        normalized_action = str(action or "").strip().lower()
        if normalized_action not in {"confirm", "reject"}:
            raise ValueError("action must be confirm or reject")
        normalized_note = str(note or "").strip()
        if "\x00" in normalized_note or len(normalized_note) > 1_000:
            raise ValueError("note must contain at most 1000 safe characters")
        incident = self.get(incident_id)
        relation_lister = getattr(self.incident_store, "list_relations", None)
        relation_appender = getattr(self.incident_store, "append_relation", None)
        if not callable(relation_lister) or not callable(relation_appender):
            raise RuntimeError("incident relation ledger is unavailable")
        relations, _total = relation_lister(incident_id, limit=500, offset=0)
        relation = next(
            (
                item
                for item in relations
                if isinstance(item, Mapping)
                and str(item.get("id") or "") == str(relation_id or "").strip()
            ),
            None,
        )
        if not isinstance(relation, Mapping):
            raise LookupError("incident relation not found")
        if str(relation.get("relation_type") or "") != "series_member":
            raise ValueError("only series_member relations can be reviewed here")
        if str(relation.get("relation_state") or "") != "candidate":
            raise ValueError("only candidate series relations can be reviewed")
        endpoints = {
            str(relation.get("subject_incident_id") or ""),
            str(relation.get("object_incident_id") or ""),
        }
        if str(incident.get("id") or incident_id) not in endpoints:
            raise LookupError("incident relation not found")
        payload = (
            dict(relation.get("payload") or {})
            if isinstance(relation.get("payload"), Mapping)
            else {}
        )
        payload.update(
            {
                "supersedes_relation_id": str(relation.get("id") or relation_id),
                "operator_review_required": False,
                "operator_action": normalized_action,
                "operator_note": normalized_note,
                "automatic_merge": False,
            }
        )
        relation_state = "confirmed" if normalized_action == "confirm" else "rejected"
        return relation_appender(
            {
                "subject_incident_id": str(relation.get("subject_incident_id") or ""),
                "object_incident_id": str(relation.get("object_incident_id") or ""),
                "idempotency_key": (
                    f"series-review:{str(relation.get('id') or relation_id)}:"
                    f"{relation_state}"
                ),
                "relation_type": "series_member",
                "relation_state": relation_state,
                "confidence": str(relation.get("confidence") or "unknown"),
                "rationale": normalized_note or f"Operator {normalized_action}ed recurrence series.",
                "payload": payload,
            },
            actor_id=actor_id,
        )

    def ingest_l0_temporal_observations(
        self,
        channel_id: int,
        heartbeat: Mapping[str, Any],
        temporal_observations: list[Mapping[str, Any]],
        *,
        actor_id: str | None = None,
        max_new_incidents: int = 4,
        tracked_limit: int = 64,
    ) -> dict[str, Any]:
        """Associate grounded L0 episode observations with durable candidates.

        New/high-signal L0 events create operator-review candidates. Continuity
        extends the same semantic episode until a grounded resolution or return
        to routine ends *perception*; case closure remains an operator decision.
        """

        normalized_channel = int(channel_id)
        if normalized_channel <= 0:
            raise ValueError("channel_id must be positive")
        batch_id = str(heartbeat.get("batch_id") or "").strip()
        if not batch_id:
            raise ValueError("heartbeat batch_id is required")
        batch_start_ms = self._optional_int(heartbeat.get("batch_start_ms"))
        batch_end_ms = self._optional_int(heartbeat.get("batch_end_ms"))
        if batch_start_ms is None or batch_end_ms is None:
            raise ValueError("heartbeat batch bounds are required")
        if batch_end_ms < batch_start_ms:
            raise ValueError("heartbeat batch end must not precede its start")
        lister = getattr(self.incident_store, "list_incidents", None)
        creator = getattr(self.incident_store, "create_incident", None)
        updater = getattr(self.incident_store, "update_incident", None)
        observation_appender = getattr(self.incident_store, "append_observation", None)
        if not all(callable(item) for item in (lister, creator, updater, observation_appender)):
            return {"supported": False, "created": 0, "associated": 0, "ended": 0}

        bounded_tracked = max(1, min(256, int(tracked_limit)))
        records, total = lister(
            channel_ids=[normalized_channel],
            case_states=["candidate", "open"],
            limit=bounded_tracked,
            offset=0,
        )
        active_by_key: dict[str, dict[str, Any]] = {}
        for raw in records:
            if not isinstance(raw, Mapping):
                continue
            semantic_key = self._primary_semantic_key(raw)
            if (
                semantic_key
                and str(raw.get("perception_state") or "unknown") != "ended"
                and self._optional_int(raw.get("possible_end_ms")) is None
                and semantic_key not in active_by_key
            ):
                active_by_key[semantic_key] = dict(raw)

        homeostasis = self._heartbeat_homeostasis(heartbeat)
        created = 0
        associated = 0
        ended = 0
        skipped = 0
        failures: list[str] = []
        max_new = max(0, min(16, int(max_new_incidents)))

        event_rows = [
            dict(item)
            for item in temporal_observations
            if isinstance(item, Mapping) and str(item.get("kind") or "") == "event"
        ][:16]
        routine_rows = [
            dict(item)
            for item in temporal_observations
            if isinstance(item, Mapping)
            and str(item.get("kind") or "") == "routine_gap"
        ][:16]

        for observation in event_rows:
            semantic_key = str(observation.get("semantic_key") or "").strip().lower()[:160]
            observation_id = str(observation.get("observation_id") or "").strip()
            event_state = str(observation.get("state") or "uncertain").strip().lower()
            label = str(observation.get("label") or semantic_key or "Incident candidate").strip()[:200]
            if not semantic_key or semantic_key in _GENERIC_INCIDENT_KEYS or not observation_id:
                skipped += 1
                continue
            incident = active_by_key.get(semantic_key)
            trigger_kind = str(observation.get("trigger_kind") or "").strip().lower()
            if trigger_kind == "episode_event":
                report = (
                    incident.get("report")
                    if isinstance(incident, Mapping)
                    and isinstance(incident.get("report"), Mapping)
                    else {}
                )
                if incident is None or str(report.get("priority") or "") not in {
                    "operator_criterion",
                    "safety",
                }:
                    # Preserve the observation in L0-L3 temporal memory, but
                    # wait for wider-scale composition. An ordinary episode
                    # must not create a case or keep a legacy noisy candidate
                    # artificially fresh. It may continue an already grounded
                    # operator/safety incident with the same canonical key.
                    skipped += 1
                    continue
            if incident is None and event_state in {"resolved", "finished"}:
                skipped += 1
                continue
            try:
                if incident is None:
                    if created >= max_new or len(active_by_key) >= bounded_tracked:
                        skipped += 1
                        continue
                    incident = creator(
                        self._l0_incident_record(
                            normalized_channel,
                            heartbeat,
                            observation,
                            homeostasis,
                        ),
                        actor_id=actor_id,
                    )
                    incident = dict(incident)
                    active_by_key[semantic_key] = incident
                    created += 1
                else:
                    incident = self._extend_l0_incident(
                        incident,
                        observation,
                        actor_id=actor_id,
                    )
                    active_by_key[semantic_key] = incident
                self._append_l0_temporal_observation(
                    incident,
                    observation,
                    homeostasis,
                    actor_id=actor_id,
                )
                associated += 1
                if event_state in {"resolved", "finished"}:
                    active_by_key.pop(semantic_key, None)
                    ended += 1
            except Exception as exc:
                failures.append(f"{semantic_key}:{type(exc).__name__}")

        for routine in routine_rows:
            applies_to = {
                str(item or "").strip().lower()
                for item in routine.get("applies_to") or []
                if str(item or "").strip()
            }
            for semantic_key, incident in list(active_by_key.items()):
                if applies_to and semantic_key not in applies_to:
                    continue
                try:
                    closed = self._end_l0_incident_at_routine(
                        incident,
                        routine,
                        actor_id=actor_id,
                    )
                    self._append_l0_temporal_observation(
                        closed,
                        routine,
                        homeostasis,
                        actor_id=actor_id,
                    )
                    active_by_key.pop(semantic_key, None)
                    ended += 1
                except Exception as exc:
                    failures.append(f"{semantic_key}:routine:{type(exc).__name__}")

        return {
            "supported": True,
            "created": created,
            "associated": associated,
            "ended": ended,
            "skipped": skipped,
            "tracked": len(records),
            "matching_total": int(total),
            "failures": failures[:8],
        }

    def ingest_rollup_incident_compositions(
        self,
        channel_id: int,
        rollup: Mapping[str, Any],
        *,
        actor_id: str | None = None,
        tracked_limit: int = 128,
    ) -> dict[str, Any]:
        """Attach L2 context to an existing grounded incident, replay-safely.

        Deterministic consolidation may enrich a case which already exists due
        to an operator criterion or an independent safety signal.  It must not
        create attention from ordinary scene narration, and it never merges or
        closes incidents automatically.
        """

        normalized_channel = int(channel_id)
        if normalized_channel <= 0:
            raise ValueError("channel_id must be positive")
        if str(rollup.get("level") or "").strip().upper() != "L2":
            return {
                "supported": True,
                "compositions": 0,
                "attached": 0,
                "episodes": 0,
                "skipped": "not_l2",
            }
        compositions = [
            dict(item)
            for item in rollup.get("incident_compositions") or []
            if isinstance(item, Mapping)
        ][:32]
        episodes_by_id = {
            str(item.get("episode_id") or ""): dict(item)
            for item in rollup.get("incident_ledger") or []
            if isinstance(item, Mapping)
            and str(item.get("episode_id") or "").strip()
        }
        if not compositions or not episodes_by_id:
            return {
                "supported": True,
                "compositions": len(compositions),
                "attached": 0,
                "episodes": 0,
                "skipped": "no_compositions" if not compositions else "no_episode_ledger",
            }
        lister = getattr(self.incident_store, "list_incidents", None)
        episode_appender = getattr(self.incident_store, "append_episode", None)
        observation_appender = getattr(self.incident_store, "append_observation", None)
        if not all(callable(item) for item in (lister, episode_appender, observation_appender)):
            return {
                "supported": False,
                "compositions": len(compositions),
                "attached": 0,
                "episodes": 0,
            }

        starts = [self._optional_int(item.get("start_ms")) for item in compositions]
        ends = [self._optional_int(item.get("end_ms")) for item in compositions]
        bounded_starts = [value for value in starts if value is not None]
        bounded_ends = [value for value in ends if value is not None]
        if not bounded_starts or not bounded_ends:
            return {
                "supported": True,
                "compositions": len(compositions),
                "attached": 0,
                "episodes": 0,
                "skipped": "missing_bounds",
            }
        context_pad_ms = 5 * 60 * 1000
        incidents, _total = lister(
            channel_ids=[normalized_channel],
            case_states=["candidate", "open"],
            since_ms=max(0, min(bounded_starts) - context_pad_ms),
            until_ms=max(bounded_ends) + context_pad_ms,
            limit=max(1, min(256, int(tracked_limit))),
            offset=0,
        )

        def observation_ids(incident: Mapping[str, Any]) -> set[str]:
            values: set[str] = set()
            anchor = incident.get("anchor_ref")
            if isinstance(anchor, Mapping):
                value = str(anchor.get("observation_id") or "").strip()
                if value:
                    values.add(value)
            for item in incident.get("timeline_refs") or []:
                if not isinstance(item, Mapping):
                    continue
                value = str(item.get("observation_id") or "").strip()
                if value:
                    values.add(value)
            return values

        def grounded_priority(incident: Mapping[str, Any]) -> tuple[int, int]:
            report = incident.get("report")
            report = report if isinstance(report, Mapping) else {}
            priority = str(report.get("priority") or "").strip().lower()
            severity = str(report.get("severity") or "").strip().lower()
            severity_rank = {
                "": 0,
                "info": 1,
                "low": 2,
                "normal": 3,
                "medium": 3,
                "high": 4,
                "critical": 5,
                "emergency": 5,
            }.get(severity, 0)
            if priority == "safety":
                return (2, severity_rank)
            if priority == "operator_criterion" and severity_rank >= 4:
                return (1, severity_rank)
            return (0, severity_rank)

        grounded = [
            (dict(item), observation_ids(item), grounded_priority(item))
            for item in incidents
            if isinstance(item, Mapping) and grounded_priority(item)[0] > 0
        ]
        attached = 0
        episode_count = 0
        skipped = 0
        failures: list[str] = []
        parent_incident_ids: list[str] = []
        rollup_id = str(rollup.get("rollup_id") or "").strip()
        for composition in compositions:
            composition_id = str(composition.get("composition_id") or "").strip()
            if (
                not composition_id
                or str(composition.get("promotion_policy") or "")
                != "extend_grounded_anchor"
                or composition.get("automatic_merge") is not False
            ):
                skipped += 1
                continue
            parent_observation_ids = {
                str(value or "").strip()
                for value in (
                    composition.get("parent_observation_ids")
                    or composition.get("anchor_observation_ids")
                    or []
                )
                if str(value or "").strip()
            }
            matches = [
                item
                for item in grounded
                if parent_observation_ids.intersection(item[1])
            ]
            if not matches:
                skipped += 1
                continue
            parent, _known_observations, _priority = max(
                matches,
                key=lambda item: (
                    item[2][0],
                    item[2][1],
                    int(item[0].get("revision") or 0),
                ),
            )
            parent_id = str(parent.get("id") or "").strip()
            if not parent_id:
                skipped += 1
                continue
            nested_ids = [
                str(value or "").strip()
                for value in composition.get("nested_episode_ids") or []
                if str(value or "").strip()
            ][:127]
            parent_episode_id = str(
                composition.get("parent_episode_id") or ""
            ).strip()
            composition_episode_ids = list(
                dict.fromkeys(
                    [
                        *([parent_episode_id] if parent_episode_id else []),
                        *nested_ids,
                    ]
                )
            )
            try:
                for episode_id in composition_episode_ids:
                    episode = episodes_by_id.get(episode_id)
                    if not episode:
                        continue
                    nested_context = episode_id != parent_episode_id
                    start_ms = self._optional_int(episode.get("start_ms"))
                    end_ms = self._optional_int(
                        episode.get("boundary_at_ms")
                        or episode.get("last_observed_ms")
                    )
                    if start_ms is None:
                        continue
                    end_ms = max(start_ms, end_ms or start_ms)
                    status = str(episode.get("status") or "open").strip().lower()
                    is_ended = status != "open"
                    episode_digest = hashlib.sha256(
                        f"{composition_id}:{episode_id}".encode("utf-8")
                    ).hexdigest()
                    evidence_refs = [
                        {
                            "kind": "vlm_snapshot",
                            "ref": str(value),
                            "role": "context" if nested_context else "event",
                        }
                        for value in episode.get("evidence_refs") or []
                        if str(value or "").strip()
                    ][:128]
                    episode_appender(
                        {
                            "incident_id": parent_id,
                            "idempotency_key": f"l2-composition:{episode_digest}",
                            "episode_key": f"l2:{episode_digest}",
                            "perception_state": "ended" if is_ended else "observed",
                            "semantic_key": str(episode.get("semantic_key") or "")[:160] or None,
                            "possible_start_ms": start_ms,
                            "observed_start_ms": start_ms,
                            "observed_end_ms": end_ms if is_ended else None,
                            "possible_end_ms": end_ms if is_ended else None,
                            "routine_before_ref": {},
                            "routine_after_ref": {},
                            "evidence_refs": evidence_refs,
                            "coverage": {
                                "source_level": "L2",
                                "rollup_id": rollup_id,
                                "composition_id": composition_id,
                                "scale_disposition": str(
                                    episode.get("scale_disposition")
                                    or "unclassified_keep"
                                ),
                                "nested_context": nested_context,
                                "composition_parent": not nested_context,
                                "automatic_merge": False,
                                "operator_review_required": True,
                            },
                        },
                        actor_id=actor_id,
                    )
                    episode_count += 1
                observation_appender(
                    {
                        "incident_id": parent_id,
                        "idempotency_key": f"rollup-composition:{composition_id}"[:200],
                        "source_kind": "rollup_l2_composition",
                        "observed_at_ms": self._optional_int(composition.get("end_ms"))
                        or self._wall_clock_ms(),
                        "channel_id": normalized_channel,
                        "perception_state": str(parent.get("perception_state") or "observed"),
                        "source_ref": {
                            "rollup_id": rollup_id,
                            "composition_id": composition_id,
                        },
                        "payload": {
                            "semantic_keys": list(composition.get("semantic_keys") or [])[:32],
                            "parent_episode_id": composition.get("parent_episode_id"),
                            "nested_episode_ids": nested_ids,
                            "promotion_policy": "extend_grounded_anchor",
                            "automatic_merge": False,
                            "operator_review_required": True,
                        },
                    },
                    actor_id=actor_id,
                )
                attached += 1
                parent_incident_ids.append(parent_id)
            except Exception as exc:
                failures.append(f"{composition_id}:{type(exc).__name__}")
        return {
            "supported": True,
            "compositions": len(compositions),
            "attached": attached,
            "episodes": episode_count,
            "skipped": skipped,
            "parent_incident_ids": list(dict.fromkeys(parent_incident_ids))[:32],
            "failures": failures[:8],
        }

    @staticmethod
    def _heartbeat_homeostasis(heartbeat: Mapping[str, Any]) -> dict[str, Any]:
        vector_signal = heartbeat.get("vector_signal")
        vector_signal = vector_signal if isinstance(vector_signal, Mapping) else {}
        capture = vector_signal.get("capture_attention")
        capture = capture if isinstance(capture, Mapping) else {}
        seconds = [
            item for item in capture.get("seconds") or [] if isinstance(item, Mapping)
        ][:120]
        activity_values: list[float] = []
        for item in seconds:
            try:
                value = float(item.get("activity_x"))
            except (TypeError, ValueError):
                continue
            if value >= 0:
                activity_values.append(value)
        return {
            "sample_count": len(seconds),
            "activity_x_max": max(activity_values, default=0.0),
            "activity_x_mean": (
                sum(activity_values) / len(activity_values)
                if activity_values
                else 0.0
            ),
            "burst_count": sum(
                str(item.get("mode") or "").strip().lower() == "burst"
                for item in seconds
            ),
        }

    def _l0_incident_record(
        self,
        channel_id: int,
        heartbeat: Mapping[str, Any],
        observation: Mapping[str, Any],
        homeostasis: Mapping[str, Any],
    ) -> dict[str, Any]:
        observation_id = str(observation.get("observation_id") or "").strip()
        semantic_key = str(observation.get("semantic_key") or "").strip().lower()[:160]
        start_ms = self._optional_int(observation.get("start_ms")) or int(heartbeat["batch_start_ms"])
        end_ms = max(start_ms, self._optional_int(observation.get("end_ms")) or int(heartbeat["batch_end_ms"]))
        label = str(observation.get("label") or semantic_key or "Incident candidate").strip()[:200]
        trigger_kind = str(observation.get("trigger_kind") or "legacy_event").strip().lower()
        if trigger_kind not in {
            "legacy_event",
            "episode_event",
            "safety_event",
            "safety_alert",
            "operator_alert",
        }:
            trigger_kind = "legacy_event"
        severity = str(observation.get("severity") or "info").strip().lower()[:32] or "info"
        report_source = {
            "operator_alert": "operator_alert_l0",
            "safety_alert": "safety_alert_l0",
            "safety_event": "safety_event_l0",
        }.get(trigger_kind, "vlm_l0_temporal")
        evidence_refs = [
            {"kind": "vlm_snapshot", "ref": str(item), "role": "event"}
            for item in observation.get("evidence_refs") or []
            if str(item or "").strip()
        ][:32]
        return {
            "state": "candidate",
            "perception_state": "observed",
            "risk_state": "unknown",
            "case_state": "candidate",
            "attention_state": "inactive",
            "identity_key": f"l0:{channel_id}:{observation_id}"[:200],
            "idempotency_key": f"l0-incident:{observation_id}"[:200],
            "title": label,
            "channel_ids": [channel_id],
            "possible_start_ms": start_ms,
            "observed_start_ms": start_ms,
            "observed_end_ms": end_ms,
            "possible_end_ms": None,
            "anchor_ref": {
                "batch_id": str(heartbeat.get("batch_id") or ""),
                "observation_id": observation_id,
            },
            "timeline_refs": [self._l0_timeline_ref(observation)],
            "evidence_refs": evidence_refs,
            "qualia_refs": [dict(homeostasis)],
            "coverage": {
                "status": "covered",
                "source_batch_id": str(heartbeat.get("batch_id") or ""),
            },
            "uncertainties": [
                (
                    "Grounded L0 match to an operator-configured alert criterion; "
                    "operator confirmation is required."
                    if trigger_kind == "operator_alert"
                    else "Grounded L0 safety/security candidate; operator confirmation is required."
                    if trigger_kind in {"safety_alert", "safety_event"}
                    else "Automated L0 incident candidate; operator confirmation is required."
                )
            ],
            "report": {
                "severity": severity,
                "summary": label,
                "source": report_source,
                "priority": (
                    "operator_criterion"
                    if trigger_kind == "operator_alert"
                    else "safety"
                    if trigger_kind in {"safety_alert", "safety_event"}
                    else "context"
                ),
            },
            "follow_policy": {},
        }

    @staticmethod
    def _l0_timeline_ref(observation: Mapping[str, Any]) -> dict[str, Any]:
        item = {
            "observation_id": str(observation.get("observation_id") or ""),
            "timestamp_ms": observation.get("start_ms"),
            "start_ms": observation.get("start_ms"),
            "end_ms": observation.get("end_ms"),
            "semantic_key": str(observation.get("semantic_key") or "")[:160],
            "label": str(observation.get("label") or "")[:240],
            "state": str(observation.get("state") or "uncertain")[:32],
            "source": "vlm_l0_temporal",
        }
        trigger_kind = str(observation.get("trigger_kind") or "").strip().lower()
        if trigger_kind:
            item["trigger_kind"] = trigger_kind[:32]
        severity = str(observation.get("severity") or "").strip().lower()
        if severity:
            item["severity"] = severity[:32]
        operator_criterion = str(observation.get("operator_criterion") or "").strip()
        if operator_criterion:
            item["operator_criterion"] = operator_criterion[:220]
        return item

    def _extend_l0_incident(
        self,
        incident: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        actor_id: str | None,
    ) -> dict[str, Any]:
        observation_id = str(observation.get("observation_id") or "").strip()
        timeline = [
            dict(item)
            for item in incident.get("timeline_refs") or []
            if isinstance(item, Mapping)
        ]
        if any(str(item.get("observation_id") or "") == observation_id for item in timeline):
            return dict(incident)
        timeline.append(self._l0_timeline_ref(observation))
        timeline = timeline[-512:]
        evidence = [
            dict(item)
            for item in incident.get("evidence_refs") or []
            if isinstance(item, Mapping)
        ]
        seen_refs = {str(item.get("ref") or "") for item in evidence}
        for raw_ref in observation.get("evidence_refs") or []:
            ref = str(raw_ref or "").strip()
            if ref and ref not in seen_refs:
                evidence.append({"kind": "vlm_snapshot", "ref": ref, "role": "event"})
                seen_refs.add(ref)
        event_state = str(observation.get("state") or "uncertain").strip().lower()
        event_end_ms = self._optional_int(observation.get("end_ms"))
        existing_end_ms = self._optional_int(incident.get("observed_end_ms"))
        effective_end_ms = max(
            value for value in (event_end_ms, existing_end_ms) if value is not None
        )
        terminal = event_state in {"resolved", "finished"}
        current_legacy = str(incident.get("state") or "candidate")
        changes: dict[str, Any] = {
            "timeline_refs": timeline,
            "evidence_refs": evidence[-512:],
            "observed_end_ms": effective_end_ms,
            "perception_state": "ended" if terminal else "observed",
        }
        trigger_kind = str(observation.get("trigger_kind") or "").strip().lower()
        incoming_priority = (
            "operator_criterion"
            if trigger_kind == "operator_alert"
            else "safety"
            if trigger_kind in {"safety_alert", "safety_event"}
            else ""
        )
        report = (
            dict(incident.get("report") or {})
            if isinstance(incident.get("report"), Mapping)
            else {}
        )
        priority_rank = {"": 0, "context": 0, "safety": 1, "operator_criterion": 2}
        if priority_rank.get(incoming_priority, 0) > priority_rank.get(
            str(report.get("priority") or ""), 0
        ):
            report["priority"] = incoming_priority
            report["source"] = (
                "operator_alert_l0"
                if trigger_kind == "operator_alert"
                else "safety_alert_l0"
                if trigger_kind == "safety_alert"
                else "safety_event_l0"
            )
            incoming_severity = str(observation.get("severity") or "").strip().lower()
            severity_rank = {
                "": 0,
                "info": 1,
                "low": 2,
                "normal": 3,
                "medium": 3,
                "high": 4,
                "critical": 5,
                "emergency": 5,
            }
            if severity_rank.get(incoming_severity, 0) > severity_rank.get(
                str(report.get("severity") or "").strip().lower(), 0
            ):
                report["severity"] = incoming_severity
            changes["report"] = report
        if terminal:
            changes["possible_end_ms"] = effective_end_ms
            if current_legacy in {"candidate", "draft"}:
                changes["state"] = "ended"
        return self.incident_store.update_incident(
            str(incident.get("id") or ""),
            expected_revision=int(incident.get("revision") or 0),
            changes=changes,
            actor_id=actor_id,
            transition={
                "transitioned_at_ms": effective_end_ms,
                "reason": "L0 temporal episode resolved" if terminal else "L0 temporal episode continued",
                "source_kind": "vlm_l0_temporal",
                "source_ref": {"observation_id": observation_id},
                "payload": {"semantic_key": observation.get("semantic_key")},
            },
        )

    def _end_l0_incident_at_routine(
        self,
        incident: Mapping[str, Any],
        routine: Mapping[str, Any],
        *,
        actor_id: str | None,
    ) -> dict[str, Any]:
        boundary_ms = self._optional_int(routine.get("end_ms") or routine.get("start_ms"))
        if boundary_ms is None:
            raise ValueError("routine boundary timestamp is required")
        current_legacy = str(incident.get("state") or "candidate")
        changes: dict[str, Any] = {
            "perception_state": "ended",
            "possible_end_ms": boundary_ms,
        }
        if current_legacy in {"candidate", "draft"}:
            changes["state"] = "ended"
        return self.incident_store.update_incident(
            str(incident.get("id") or ""),
            expected_revision=int(incident.get("revision") or 0),
            changes=changes,
            actor_id=actor_id,
            transition={
                "transitioned_at_ms": boundary_ms,
                "reason": "grounded return to routine ended perceptual episode",
                "source_kind": "routine_boundary",
                "source_ref": {
                    "observation_id": str(routine.get("observation_id") or ""),
                },
                "payload": {
                    "semantic_key": self._primary_semantic_key(incident),
                    "case_closed": False,
                },
            },
        )

    def _append_l0_temporal_observation(
        self,
        incident: Mapping[str, Any],
        observation: Mapping[str, Any],
        homeostasis: Mapping[str, Any],
        *,
        actor_id: str | None,
    ) -> None:
        observation_id = str(observation.get("observation_id") or "").strip()
        if not observation_id:
            return
        perception_state = (
            "ended"
            if str(observation.get("kind") or "") == "routine_gap"
            or str(observation.get("state") or "") in {"resolved", "finished"}
            else "observed"
        )
        self.incident_store.append_observation(
            {
                "incident_id": str(incident.get("id") or ""),
                "idempotency_key": f"temporal:{observation_id}"[:200],
                "source_kind": (
                    "routine_boundary"
                    if str(observation.get("kind") or "") == "routine_gap"
                    else str(observation.get("trigger_kind") or "vlm_l0_event")
                ),
                "observed_at_ms": self._optional_int(observation.get("end_ms")) or self._wall_clock_ms(),
                "channel_id": _positive_ints(incident.get("channel_ids"))[0],
                "perception_state": perception_state,
                "source_ref": {"observation_id": observation_id},
                "payload": {
                    "semantic_key": observation.get("semantic_key"),
                    "label": str(observation.get("label") or "")[:240],
                    "state": observation.get("state"),
                    "trigger_kind": observation.get("trigger_kind"),
                    "severity": observation.get("severity"),
                    "operator_criterion": observation.get("operator_criterion"),
                    "homeostasis": dict(homeostasis),
                },
            },
            actor_id=actor_id,
        )

    @staticmethod
    def _focus_context(incident: Mapping[str, Any]) -> str:
        """Bound prior incident evidence carried into live VLM batches."""

        report = (
            incident.get("report")
            if isinstance(incident.get("report"), Mapping)
            else {}
        )
        raw_timeline = incident.get("timeline_refs")
        timeline_items = raw_timeline if isinstance(raw_timeline, list) else []
        timeline = [
            {
                key: item.get(key)
                for key in (
                    "timestamp_ms",
                    "start_ms",
                    "end_ms",
                    "semantic_key",
                    "label",
                    "description",
                    "confidence",
                )
                if item.get(key) is not None
            }
            for item in timeline_items[-8:]
            if isinstance(item, Mapping)
        ]
        raw_uncertainties = incident.get("uncertainties_json") or incident.get("uncertainties")
        uncertainty_items = raw_uncertainties if isinstance(raw_uncertainties, list) else []
        payload = {
            "title": str(incident.get("title") or "")[:180],
            "summary": str(report.get("summary") or "")[:600],
            "possible_start_ms": incident.get("possible_start_ms"),
            "observed_start_ms": incident.get("observed_start_ms"),
            "observed_end_ms": incident.get("observed_end_ms"),
            "timeline": timeline,
            "uncertainties": [
                str(item)[:180]
                for item in uncertainty_items[:6]
            ],
        }
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        try:
            return compact_incident_context(rendered, max_tokens=800)
        except PromptBudgetError:
            # Preserve a valid semantic record even under a pathological title
            # or timeline. Arbitrary slicing can produce invalid JSON and poison
            # every following VLM turn.
            return json.dumps(
                {
                    "title": str(incident.get("title") or "")[:180],
                    "possible_start_ms": incident.get("possible_start_ms"),
                    "observed_start_ms": incident.get("observed_start_ms"),
                    "context_status": "compacted_to_identity",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )

    def follow(
        self,
        incident_id: str,
        *,
        actor_id: str,
        mode: str = "follow",
        ttl_seconds: int = 300,
        expected_revision: int | None = None,
    ) -> tuple[dict[str, Any], Any]:
        normalized_mode = str(mode or "follow").strip().lower()
        if normalized_mode not in {"follow", "critical"}:
            raise ValueError("mode must be follow or critical")
        ttl = int(ttl_seconds)
        if ttl < 60 or ttl > 8 * 60 * 60:
            raise ValueError("ttl_seconds must be between 60 and 28800")
        incident = self.get(incident_id)
        revision = self._expected_revision(incident, expected_revision)
        channel_ids = self._required_channels(incident)

        previous_lease = None
        lease_manager = getattr(self.focus_runtime, "incident_focus_leases", None)
        lease_getter = getattr(lease_manager, "get", None)
        if callable(lease_getter):
            try:
                previous_lease = lease_getter(incident_id)
            except Exception:
                previous_lease = None

        focus_args = {
            "level": normalized_mode,
            "ttl_seconds": ttl,
        }
        try:
            lease = self.focus_runtime.start_incident_focus(
                incident_id,
                channel_ids,
                context=self._focus_context(incident),
                **focus_args,
            )
        except TypeError as exc:
            # Compatibility seam for external/custom runtimes implementing
            # the pre-context signature. EVA's runtime always accepts it.
            if "context" not in str(exc):
                raise
            lease = self.focus_runtime.start_incident_focus(
                incident_id,
                channel_ids,
                **focus_args,
            )
        try:
            focus_payload = self._active_focus_payload(lease, ttl, incident)
            updated = self.incident_store.update_incident(
                incident_id,
                expected_revision=revision,
                changes={
                    "state": "following",
                    "case_state": "open",
                    "attention_state": normalized_mode,
                    "follow_policy": focus_payload,
                },
                actor_id=actor_id,
                transition={
                    "transitioned_at_ms": int(focus_payload["started_at_ms"]),
                    "reason": "incident follow started",
                    "source_kind": "follow_started",
                    "source_ref": {"mode": normalized_mode},
                    "payload": {"ttl_seconds": ttl},
                },
            )
        except Exception:
            # Starting focus precedes persistence so capacity errors remain
            # side-effect free. If persistence loses the revision race, remove
            # a newly-created lease. Preserve a pre-existing focus lease rather
            # than turning a failed refresh into an accidental stop.
            if previous_lease is None:
                self.focus_runtime.stop_incident_focus(incident_id)
            raise
        return dict(updated), lease

    def stop_follow(
        self,
        incident_id: str,
        *,
        actor_id: str,
        expected_revision: int | None = None,
        stop_reason: str = "operator",
    ) -> tuple[dict[str, Any], bool]:
        incident = self.get(incident_id)
        revision = self._expected_revision(incident, expected_revision)
        self._required_channels(incident)
        updated = self._finalize_follow_record(
            incident,
            revision=revision,
            actor_id=actor_id,
            stop_reason=stop_reason,
        )
        stopped = bool(self.focus_runtime.stop_incident_focus(incident_id))
        return dict(updated), stopped

    def review_incident(
        self,
        incident_id: str,
        *,
        actor_id: str,
        action: str,
        expected_revision: int | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Apply one explicit operator lifecycle decision.

        Review changes the materialized incident row and appends an immutable
        observation.  It never rewrites episode/relation ledgers and it never
        infers visual absence, event end, or risk from model silence.
        """

        normalized_action = str(action or "").strip().lower()
        if normalized_action not in {
            "confirm",
            "resolve",
            "dismiss",
            "false_positive",
            "reopen",
        }:
            raise ValueError(
                "action must be confirm, resolve, dismiss, false_positive, or reopen"
            )
        normalized_note = str(note or "").strip()
        if "\x00" in normalized_note or len(normalized_note) > 1_000:
            raise ValueError("note must contain at most 1000 safe characters")

        incident = self.get(incident_id)
        revision = self._expected_revision(incident, expected_revision)
        previous = {
            key: str(incident.get(key) or "unknown")
            for key in (
                "state",
                "perception_state",
                "risk_state",
                "case_state",
                "attention_state",
            )
        }
        terminal_action = normalized_action in {
            "resolve",
            "dismiss",
            "false_positive",
        }
        follow = (
            dict(incident.get("follow_policy") or {})
            if isinstance(incident.get("follow_policy"), Mapping)
            else {}
        )
        focus_stopped = False
        if terminal_action and (
            str(incident.get("state") or "") == "following"
            or follow.get("active") is True
        ):
            incident, _stopped = self.stop_follow(
                incident_id,
                actor_id=actor_id,
                expected_revision=revision,
                stop_reason=f"operator_review_{normalized_action}",
            )
            focus_stopped = True
            revision = int(incident.get("revision") or revision + 1)
            follow = (
                dict(incident.get("follow_policy") or {})
                if isinstance(incident.get("follow_policy"), Mapping)
                else {}
            )

        changes: dict[str, Any]
        if normalized_action == "confirm":
            if str(incident.get("case_state") or "") in {
                "closed",
                "dismissed",
                "false_positive",
            }:
                raise ValueError("reopen a historical incident before confirming it")
            changes = {
                "state": "reported",
                "case_state": "open",
            }
        elif normalized_action == "resolve":
            changes = {
                "state": "closed",
                "case_state": "closed",
                "risk_state": "resolved",
                "attention_state": "inactive",
            }
        elif normalized_action == "dismiss":
            changes = {
                "state": "closed",
                "case_state": "dismissed",
                "risk_state": "resolved",
                "attention_state": "inactive",
            }
        elif normalized_action == "false_positive":
            changes = {
                "state": "closed",
                "case_state": "false_positive",
                "risk_state": "resolved",
                "attention_state": "inactive",
            }
        else:
            current_risk = str(incident.get("risk_state") or "unknown")
            changes = {
                "state": "reported",
                "case_state": "open",
                "risk_state": "unknown" if current_risk == "resolved" else current_risk,
                "attention_state": "inactive",
            }

        if normalized_action in {"resolve", "dismiss", "false_positive", "reopen"}:
            if follow:
                follow.update(
                    {
                        "active": False,
                        "inactive_reason": f"operator_review_{normalized_action}",
                    }
                )
                changes["follow_policy"] = follow

        report = (
            dict(incident.get("report") or {})
            if isinstance(incident.get("report"), Mapping)
            else {}
        )
        reviewed_at_ms = self._wall_clock_ms()
        report["last_operator_review"] = {
            "action": normalized_action,
            "reviewed_at_ms": reviewed_at_ms,
            "note": normalized_note,
        }
        changes["report"] = report
        updated = self.incident_store.update_incident(
            incident_id,
            expected_revision=revision,
            changes=changes,
            actor_id=actor_id,
            transition={
                "transitioned_at_ms": reviewed_at_ms,
                "reason": f"operator review: {normalized_action}",
                "source_kind": "operator_review",
                "source_ref": {"action": normalized_action},
                "payload": {
                    "note": normalized_note,
                    "ground_truth": "operator_review",
                },
            },
        )

        if (terminal_action or normalized_action == "reopen") and not focus_stopped:
            self.focus_runtime.stop_incident_focus(incident_id)

        appender = getattr(self.incident_store, "append_observation", None)
        if callable(appender):
            current = {
                key: str(updated.get(key) or "unknown")
                for key in (
                    "state",
                    "perception_state",
                    "risk_state",
                    "case_state",
                    "attention_state",
                )
            }
            observation = {
                "incident_id": incident_id,
                "idempotency_key": (
                    f"operator-review:{int(updated.get('revision') or revision + 1)}:"
                    f"{normalized_action}"
                ),
                "source_kind": "operator_review",
                "observed_at_ms": reviewed_at_ms,
                "channel_id": self._required_channels(updated)[0],
                "perception_state": str(updated.get("perception_state") or "unknown"),
                "source_ref": {"incident_revision": int(updated.get("revision") or 0)},
                "payload": {
                    "action": normalized_action,
                    "note": normalized_note,
                    "previous": previous,
                    "current": current,
                    "ground_truth": "operator_review",
                },
            }
            try:
                try:
                    appender(observation, actor_id=actor_id)
                except TypeError:
                    appender(observation)
            except Exception:
                # The materialized lifecycle update is already authoritative.
                # Audit logging at the HTTP boundary still records completion;
                # a diagnostic ledger failure must not invite a duplicate click.
                pass
        return dict(updated)

    def reconcile_expired_follow(
        self,
        incident: Mapping[str, Any],
        *,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        """Durably finalize a lazily expired process-local Follow lease."""

        follow = incident.get("follow_policy") if isinstance(incident.get("follow_policy"), Mapping) else {}
        expires_at_ms = self._optional_int(follow.get("expires_at_ms"))
        if (
            str(incident.get("state") or "") != "following"
            or follow.get("active") is not True
            or expires_at_ms is None
            or expires_at_ms > self._wall_clock_ms()
        ):
            return dict(incident)
        try:
            return self._finalize_follow_record(
                incident,
                revision=int(incident.get("revision") or 0),
                actor_id=actor_id,
                stop_reason="ttl_expired",
            )
        except IncidentRevisionConflict:
            return self.get(str(incident.get("id") or ""))

    def observation_for_heartbeat(
        self,
        incident: Mapping[str, Any],
        heartbeat: Mapping[str, Any],
    ) -> dict[str, Any]:
        return classify_follow_heartbeat(incident, heartbeat)

    def _follow_observations(self, incident: Mapping[str, Any], ended_at_ms: int) -> list[dict[str, Any]]:
        loader = getattr(self.incident_store, "list_observations", None)
        if not callable(loader):
            return []
        follow = incident.get("follow_policy") if isinstance(incident.get("follow_policy"), Mapping) else {}
        since_ms = self._optional_int(follow.get("started_at_ms"))
        try:
            observations, _total = loader(
                str(incident.get("id") or ""),
                since_ms=since_ms,
                until_ms=int(ended_at_ms),
                limit=2000,
                offset=0,
            )
        except TypeError:
            observations, _total = loader(str(incident.get("id") or ""))
        return [dict(item) for item in observations if isinstance(item, Mapping)]

    def _finalize_follow_record(
        self,
        incident: Mapping[str, Any],
        *,
        revision: int,
        actor_id: str | None,
        stop_reason: str,
    ) -> dict[str, Any]:
        ended_at_ms = self._wall_clock_ms()
        observations = self._follow_observations(incident, ended_at_ms)
        result = build_follow_result(
            incident,
            observations,
            ended_at_ms=ended_at_ms,
            stop_reason=stop_reason,
        )
        outcome = str(result.get("outcome") or "inconclusive")
        previous_follow = dict(incident.get("follow_policy") or {})
        previous_follow.update(
            {
                "active": False,
                "stopped_at_ms": ended_at_ms,
                "stop_reason": str(stop_reason or "operator")[:80],
                "last_result": result,
            }
        )
        report = dict(incident.get("report") or {}) if isinstance(incident.get("report"), Mapping) else {}
        report["follow_result"] = result
        changes: dict[str, Any] = {
            "state": "ended" if outcome == "resolved" else "draft",
            "attention_state": "inactive",
            "follow_policy": previous_follow,
            "report": report,
        }
        if outcome == "resolved":
            changes["perception_state"] = "ended"
            changes["risk_state"] = "resolved"
            last_observation_ms = self._optional_int(result.get("last_observation_ms"))
            observed_start_ms = self._optional_int(incident.get("observed_start_ms"))
            if last_observation_ms is not None and (
                observed_start_ms is None or last_observation_ms >= observed_start_ms
            ):
                changes["observed_end_ms"] = last_observation_ms
        elif outcome == "continuing":
            changes["perception_state"] = "observed"

        projected = {**dict(incident), **changes, "report": report}
        synopsis = build_incident_synopsis(projected)
        report["summary"] = str(synopsis.get("description") or "")
        report["synopsis"] = synopsis
        changes["title"] = str(synopsis.get("title") or incident.get("title") or "Incident")[:200]
        updated = self.incident_store.update_incident(
            str(incident.get("id") or ""),
            expected_revision=revision,
            changes=changes,
            actor_id=actor_id,
            transition={
                "transitioned_at_ms": ended_at_ms,
                "reason": f"incident follow stopped: {str(stop_reason or 'operator')[:80]}",
                "source_kind": "follow_completed",
                "source_ref": {"follow_run_id": str(previous_follow.get("run_id") or "legacy")},
                "payload": {
                    "outcome": outcome,
                    "stop_reason": str(stop_reason or "operator")[:80],
                },
            },
        )
        appender = getattr(self.incident_store, "append_observation", None)
        run_id = str(previous_follow.get("run_id") or "legacy")
        if callable(appender):
            completion_observation = {
                "incident_id": str(incident.get("id") or ""),
                "idempotency_key": f"follow:{run_id}:completed",
                "source_kind": "follow_completed",
                "observed_at_ms": ended_at_ms,
                "channel_id": _positive_ints(incident.get("channel_ids"))[0],
                "perception_state": str(changes.get("perception_state") or incident.get("perception_state") or "unknown"),
                "source_ref": {"follow_run_id": run_id},
                "payload": result,
            }
            try:
                try:
                    appender(completion_observation, actor_id=actor_id)
                except TypeError:
                    appender(completion_observation)
            except Exception:
                # The materialized incident state is authoritative and was
                # already committed. A diagnostic ledger write must not turn
                # successful Follow finalization into a 500 response.
                pass
        return dict(updated)

    def public_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        channel_ids = _positive_ints(record.get("channel_ids"))
        timeline = [
            dict(item)
            for item in record.get("timeline_refs") or []
            if isinstance(item, Mapping)
        ]
        evidence = [
            dict(item)
            for item in record.get("evidence_refs") or []
            if isinstance(item, Mapping)
        ]
        report = (
            dict(record.get("report") or {})
            if isinstance(record.get("report"), Mapping)
            else {}
        )
        qualia_refs = [
            dict(item)
            for item in record.get("qualia_refs") or []
            if isinstance(item, Mapping)
        ]
        follow = (
            dict(record.get("follow_policy") or {})
            if isinstance(record.get("follow_policy"), Mapping)
            else {}
        )
        if follow.get("active") is True:
            expires_at_ms = self._optional_int(follow.get("expires_at_ms"))
            inactive_reason: str | None = None
            if expires_at_ms is not None and expires_at_ms <= self._wall_clock_ms():
                inactive_reason = "ttl_expired"
            lease_manager = getattr(self.focus_runtime, "incident_focus_leases", None)
            lease_getter = getattr(lease_manager, "get", None)
            if inactive_reason is None and callable(lease_getter):
                try:
                    if lease_getter(str(record.get("id") or "")) is None:
                        inactive_reason = "runtime_lease_absent"
                except Exception:
                    pass
            if inactive_reason is not None:
                follow["active"] = False
                follow["inactive_reason"] = inactive_reason
        bounds = {
            "possible_start": record.get("possible_start_ms"),
            "possible_start_ms": record.get("possible_start_ms"),
            "observed_start": record.get("observed_start_ms"),
            "observed_start_ms": record.get("observed_start_ms"),
            "apex": report.get("apex_ms"),
            "apex_ms": report.get("apex_ms"),
            "observed_end": record.get("observed_end_ms"),
            "observed_end_ms": record.get("observed_end_ms"),
            "possible_end": record.get("possible_end_ms"),
            "possible_end_ms": record.get("possible_end_ms"),
        }
        semantic_keys = list(
            dict.fromkeys(
                str(item.get("semantic_key") or "").strip()
                for item in timeline
                if str(item.get("semantic_key") or "").strip()
            )
        )
        synopsis = build_incident_synopsis(record)
        public = {
            **dict(record),
            "incident_id": str(record.get("id") or ""),
            "channel_id": channel_ids[0] if channel_ids else None,
            "channels": channel_ids,
            "channel_ids": channel_ids,
            "severity": str(report.get("severity") or "info"),
            "stored_title": str(record.get("title") or ""),
            "title": str(synopsis.get("title") or record.get("title") or "Incident"),
            "summary": str(synopsis.get("description") or report.get("summary") or "").strip(),
            "synopsis": synopsis,
            "homeostasis": dict(synopsis.get("homeostasis") or {}),
            "key_moments": list(synopsis.get("key_moments") or []),
            "follow_result": dict(synopsis.get("follow_result") or {}),
            "time_bounds": bounds,
            "timeline": timeline,
            "events": timeline,
            "evidence": evidence,
            "qualia_digest": qualia_refs[0] if qualia_refs else {},
            "semantic_keys": semantic_keys,
            "follow": follow,
            "follow_policy": follow,
        }
        if public.get("state") == "following" and follow.get("active") is False:
            public["state"] = "draft"
        if not public["summary"]:
            public["summary"] = incident_narrative(public)
        return public

    def public_review_records(
        self,
        records: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Project a review page and resolve all compact image refs in one query."""

        cover_hints: list[dict[str, Any]] = []
        for record in records:
            channel_ids = _positive_ints(record.get("channel_ids"))
            timestamp_ms = self._optional_int(record.get("possible_start_ms"))
            candidates = [
                (index, item)
                for index, item in enumerate(record.get("evidence_refs") or [])
                if isinstance(item, Mapping)
                and str(item.get("kind") or "").strip().lower() == "vlm_snapshot"
                and str(item.get("ref") or "").strip()
            ]
            if not channel_ids or timestamp_ms is None or not candidates:
                continue
            _, selected = min(
                candidates,
                key=lambda pair: (
                    _INCIDENT_COVER_ROLE_PRIORITY.get(
                        str(pair[1].get("role") or "evidence").lower(),
                        7,
                    ),
                    pair[0],
                ),
            )
            cover_hints.append(
                {
                    "ref": str(selected.get("ref") or "").strip(),
                    "channel_id": channel_ids[0],
                    "timestamp_ms": timestamp_ms,
                }
            )
        refs = list(
            dict.fromkeys(
                str(item.get("ref") or "").strip()
                for record in records
                for item in record.get("evidence_refs") or []
                if isinstance(item, Mapping)
                and str(item.get("kind") or "").strip().lower() == "vlm_snapshot"
                and str(item.get("ref") or "").strip()
            )
        )
        fast_resolver = getattr(
            self.detections_store,
            "resolve_vlm_snapshot_cover_refs",
            None,
        )
        resolver = getattr(self.detections_store, "resolve_vlm_snapshot_refs", None)
        resolved: Mapping[str, Mapping[str, Any]] = {}
        if cover_hints and callable(fast_resolver):
            try:
                candidate = fast_resolver(cover_hints)
                if isinstance(candidate, Mapping):
                    resolved = candidate
            except Exception as exc:
                # Incident text and lifecycle state remain useful if archive media
                # is temporarily unavailable; keep the review endpoint operational.
                _LOGGER.warning(
                    "Incident review cover resolver failed: %s",
                    type(exc).__name__,
                )
                resolved = {}
        elif refs and callable(resolver):
            try:
                candidate = resolver(refs)
                if isinstance(candidate, Mapping):
                    resolved = candidate
            except Exception as exc:
                _LOGGER.warning(
                    "Legacy incident review cover resolver failed: %s",
                    type(exc).__name__,
                )
                resolved = {}
        return [
            self.public_review_record(record, resolved_snapshot_refs=resolved)
            for record in records
        ]

    def public_review_record(
        self,
        record: Mapping[str, Any],
        *,
        resolved_snapshot_refs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return the bounded projection used by the incident review board.

        The board may render hundreds of incidents.  Sending complete timelines,
        evidence manifests and qualia payloads there would duplicate the incident
        report and waste both browser memory and response time.  Detail remains
        available through ``GET /incidents/<id>`` when an operator opens a card.
        """

        now_ms = self._wall_clock_ms()
        channel_ids = _positive_ints(record.get("channel_ids"))
        report = (
            dict(record.get("report") or {})
            if isinstance(record.get("report"), Mapping)
            else {}
        )
        coverage = (
            dict(record.get("coverage") or {})
            if isinstance(record.get("coverage"), Mapping)
            else {}
        )
        timeline = [
            item
            for item in record.get("timeline_refs") or []
            if isinstance(item, Mapping)
        ]
        evidence = [
            item
            for item in record.get("evidence_refs") or []
            if isinstance(item, Mapping)
        ]
        uncertainties = list(record.get("uncertainties") or [])
        follow = (
            dict(record.get("follow_policy") or {})
            if isinstance(record.get("follow_policy"), Mapping)
            else {}
        )
        expires_at_ms = self._optional_int(follow.get("expires_at_ms"))
        if follow.get("active") is True and expires_at_ms is not None and expires_at_ms <= now_ms:
            follow["active"] = False
            follow["inactive_reason"] = "ttl_expired"

        def item_timestamp(item: Mapping[str, Any]) -> int | None:
            for key in ("timestamp_ms", "occurred_at_ms", "start_ms", "end_ms"):
                timestamp = self._optional_int(item.get(key))
                if timestamp is not None and timestamp > 0:
                    return timestamp
            return None

        # Prefer a meaningful apex/alert frame over the archive anchor while still
        # retaining a deterministic fallback when only the anchor is available.
        cover_candidates: list[tuple[int, int, int, str]] = []
        for item in evidence:
            ref = str(item.get("ref") or "").strip()
            resolved_ref = (
                resolved_snapshot_refs.get(ref)
                if ref and isinstance(resolved_snapshot_refs, Mapping)
                else None
            )
            resolved_ref = resolved_ref if isinstance(resolved_ref, Mapping) else {}
            detection_id = self._optional_int(
                item.get("detection_id")
                or item.get("id")
                or resolved_ref.get("detection_id")
            )
            if detection_id is None or detection_id <= 0:
                continue
            role = str(item.get("role") or item.get("kind") or "evidence").lower()
            timestamp = item_timestamp(item) or self._optional_int(resolved_ref.get("timestamp_ms")) or 0
            cover_candidates.append((_INCIDENT_COVER_ROLE_PRIORITY.get(role, 7), -timestamp, detection_id, role))
        anchor = record.get("anchor_ref")
        if isinstance(anchor, Mapping):
            detection_id = self._optional_int(anchor.get("detection_id") or anchor.get("id"))
            if detection_id is not None and detection_id > 0:
                timestamp = item_timestamp(anchor) or 0
                cover_candidates.append((_INCIDENT_COVER_ROLE_PRIORITY["anchor"], -timestamp, detection_id, "anchor"))
        cover = None
        if cover_candidates:
            _, negative_timestamp, detection_id, role = min(cover_candidates)
            cover = {
                "detection_id": detection_id,
                "timestamp_ms": -negative_timestamp or None,
                "role": role,
            }

        possible_start_ms = self._optional_int(record.get("possible_start_ms"))
        observed_start_ms = self._optional_int(record.get("observed_start_ms"))
        observed_end_ms = self._optional_int(record.get("observed_end_ms"))
        possible_end_ms = self._optional_int(record.get("possible_end_ms"))
        evidence_timestamps = [
            timestamp
            for timestamp in (item_timestamp(item) for item in [*timeline, *evidence])
            if timestamp is not None
        ]
        last_evidence_ms = max(
            evidence_timestamps
            + [
                timestamp
                for timestamp in (observed_end_ms, observed_start_ms, possible_start_ms)
                if timestamp is not None
            ],
            default=None,
        )
        observed_effective_end = observed_end_ms or last_evidence_ms
        observed_duration_ms = (
            max(0, observed_effective_end - observed_start_ms)
            if observed_start_ms is not None and observed_effective_end is not None
            else None
        )

        state = str(record.get("state") or "draft").lower()
        perception_state = str(record.get("perception_state") or "unknown").lower()
        risk_state = str(record.get("risk_state") or "unknown").lower()
        case_state = str(record.get("case_state") or "candidate").lower()
        attention_state = str(record.get("attention_state") or "inactive").lower()
        history = (
            state in {"closed", "dismissed", "false_positive"}
            or case_state in {"closed", "dismissed", "false_positive"}
        )
        active = (
            not history
            and (
                state == "following"
                or follow.get("active") is True
                or risk_state in {"active", "critical"}
                or attention_state in {"follow", "critical"}
                # An operator-confirmed case remains operationally active even
                # when its visual episode has ended; case closure is an
                # independent decision and must not be inferred from a boundary.
                or case_state == "open"
            )
        )
        review_state = "history" if history else "active" if active else "needs_review"
        case_end_ms = possible_end_ms or observed_end_ms or last_evidence_ms
        case_duration_ms = None
        if possible_start_ms is not None:
            case_duration_ms = max(
                0,
                (case_end_ms if history and case_end_ms is not None else now_ms) - possible_start_ms,
            )

        semantic_keys = list(
            dict.fromkeys(
                str(item.get("semantic_key") or "").strip()
                for item in timeline
                if str(item.get("semantic_key") or "").strip()
            )
        )[:6]
        synopsis = build_incident_synopsis(record)
        summary = str(synopsis.get("description") or report.get("summary") or "").strip()
        if not summary:
            summary = str(record.get("title") or "Incident awaiting review").strip()

        return {
            "id": str(record.get("id") or ""),
            "incident_id": str(record.get("id") or ""),
            "revision": int(record.get("revision") or 0),
            "title": str(synopsis.get("title") or record.get("title") or "Incident")[:200],
            "summary": summary[:900],
            "outcome": str(synopsis.get("outcome") or "awaiting_review"),
            "homeostasis": dict(synopsis.get("homeostasis") or {}),
            "follow_result": dict(synopsis.get("follow_result") or {}),
            "severity": str(report.get("severity") or "info"),
            "source": str(report.get("source") or ""),
            "priority": str(report.get("priority") or ""),
            "review_state": review_state,
            "state": "draft" if state == "following" and follow.get("active") is False else state,
            "perception_state": perception_state,
            "risk_state": risk_state,
            "case_state": case_state,
            "attention_state": attention_state,
            "channel_id": channel_ids[0] if channel_ids else None,
            "channels": channel_ids,
            "channel_ids": channel_ids,
            "possible_start_ms": possible_start_ms,
            "observed_start_ms": observed_start_ms,
            "observed_end_ms": observed_end_ms,
            "possible_end_ms": possible_end_ms,
            "last_evidence_ms": last_evidence_ms,
            "observed_duration_ms": observed_duration_ms,
            "case_duration_ms": case_duration_ms,
            "cover": cover,
            "evidence_count": len(evidence),
            "timeline_count": len(timeline),
            "uncertainty_count": len(uncertainties),
            "semantic_keys": semantic_keys,
            "coverage": {
                key: coverage.get(key)
                for key in ("status", "covered_fraction_estimate", "gap_count")
                if coverage.get(key) is not None
            },
            "follow": {
                key: follow.get(key)
                for key in ("active", "mode", "started_at_ms", "expires_at_ms", "inactive_reason")
                if follow.get(key) is not None
            },
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        }

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _expected_revision(
        incident: Mapping[str, Any],
        expected_revision: int | None,
    ) -> int:
        actual = int(incident.get("revision") or 0)
        expected = actual if expected_revision is None else int(expected_revision)
        if expected <= 0:
            raise ValueError("expected_revision must be positive")
        if expected != actual:
            raise IncidentRevisionConflict(
                str(incident.get("id") or ""),
                expected,
                actual,
            )
        return expected

    @staticmethod
    def _required_channels(incident: Mapping[str, Any]) -> list[int]:
        channel_ids = _positive_ints(incident.get("channel_ids"))
        if not channel_ids:
            raise ValueError("incident has no channel ownership metadata")
        return channel_ids

    def _active_focus_payload(
        self,
        lease: Any,
        ttl_seconds: int,
        incident: Mapping[str, Any],
    ) -> dict[str, Any]:
        now_ms = self._wall_clock_ms()
        last_observed_ms = self._optional_int(
            incident.get("observed_end_ms") or incident.get("observed_start_ms")
        )
        return {
            "active": True,
            "mode": str(
                getattr(getattr(lease, "level", None), "value", None)
                or "follow"
            ),
            "channel_ids": [
                int(value) for value in getattr(lease, "channel_ids", ())
            ],
            "started_at_ms": now_ms,
            "updated_at_ms": now_ms,
            "expires_at_ms": now_ms + int(ttl_seconds) * 1000,
            "ttl_seconds": int(ttl_seconds),
            "run_id": uuid.uuid4().hex,
            "relationship": (
                "recurrence_watch"
                if last_observed_ms is not None and now_ms - last_observed_ms > 2 * 60 * 1000
                else "continuation"
            ),
            "incident_last_observed_ms": last_observed_ms,
        }


__all__ = [
    "IncidentCommandService",
    "incident_narrative",
    "incident_storage_record",
]
