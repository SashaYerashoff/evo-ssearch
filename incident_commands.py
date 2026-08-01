"""Shared incident commands for HTTP and the secured EVA agent surface.

The command layer owns orchestration between durable incident state and the
process-local attention lease.  It deliberately does not perform authorization:
HTTP guards and ``EvaAgentToolAdapter`` resolve tenant/channel authority before
calling it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from incident_service import IncidentDraftAssembler, IncidentDraftRequest
from incident_store import IncidentRevisionConflict


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
    timeline = [
        item
        for item in incident.get("timeline_refs") or incident.get("timeline") or []
        if isinstance(item, Mapping) and str(item.get("label") or "").strip()
    ]
    coverage = (
        incident.get("coverage")
        if isinstance(incident.get("coverage"), Mapping)
        else {}
    )
    if not timeline:
        return (
            "No grounded event transition was recovered around the selected "
            "anchor. Archive coverage is "
            f"{str(coverage.get('status') or 'unknown')}; operator review is required."
        )
    labels = [str(item.get("label") or "").strip() for item in timeline[:4]]
    narrative = " → ".join(labels)
    if len(timeline) > len(labels):
        narrative += f" → {len(timeline) - len(labels)} more grounded timeline item(s)"
    return (
        f"EVA connected {len(timeline)} evidence-linked timeline item(s): "
        f"{narrative}. Archive coverage is "
        f"{str(coverage.get('status') or 'unknown')}; P/N/M and motion values "
        "remain attention signals, not visual proof."
    )


def incident_storage_record(draft: Mapping[str, Any]) -> dict[str, Any]:
    bounds = (
        draft.get("time_bounds")
        if isinstance(draft.get("time_bounds"), Mapping)
        else {}
    )
    report = {
        "severity": str(draft.get("severity") or "info"),
        "apex_ms": bounds.get("apex_ms"),
        "provenance": dict(draft.get("provenance") or {}),
        "qualia_digest": dict(draft.get("qualia_digest") or {}),
    }
    provisional = {
        "timeline_refs": list(draft.get("timeline") or []),
        "coverage": dict(draft.get("coverage") or {}),
    }
    report["summary"] = incident_narrative(provisional)
    return {
        "state": "draft",
        "title": str(draft.get("title") or "Incident draft")[:200],
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
        return self.incident_store.create_incident(
            incident_storage_record(draft),
            actor_id=actor_id,
        )

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
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )[:2400]

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
            updated = self.incident_store.update_incident(
                incident_id,
                expected_revision=revision,
                changes={
                    "state": "following",
                    "follow_policy": self._active_focus_payload(lease, ttl),
                },
                actor_id=actor_id,
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
        previous_follow = dict(incident.get("follow_policy") or {})
        previous_follow.update(
            {
                "active": False,
                "stopped_at_ms": self._wall_clock_ms(),
                "stop_reason": str(stop_reason or "operator")[:80],
            }
        )

        # Persist first. A stale revision therefore cannot silently remove the
        # live lease while the durable incident still says it is following.
        updated = self.incident_store.update_incident(
            incident_id,
            expected_revision=revision,
            changes={"state": "draft", "follow_policy": previous_follow},
            actor_id=actor_id,
        )
        stopped = bool(self.focus_runtime.stop_incident_focus(incident_id))
        return dict(updated), stopped

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
        public = {
            **dict(record),
            "incident_id": str(record.get("id") or ""),
            "channel_id": channel_ids[0] if channel_ids else None,
            "channels": channel_ids,
            "channel_ids": channel_ids,
            "severity": str(report.get("severity") or "info"),
            "summary": str(report.get("summary") or "").strip(),
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

    def _active_focus_payload(self, lease: Any, ttl_seconds: int) -> dict[str, Any]:
        now_ms = self._wall_clock_ms()
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
        }


__all__ = [
    "IncidentCommandService",
    "incident_narrative",
    "incident_storage_record",
]
