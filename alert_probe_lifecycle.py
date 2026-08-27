"""Lifecycle policy for temporary probes derived from a VLM alert.

This module deliberately has no dependency on Flask, the agent tool surface, or
the embedding runtime.  It is an internal admission/lifecycle component that
can be called by ``oldapp`` after an alert has been parsed.

The lifecycle is intentionally one-way:

    VLM alert (generation 0) -> temporary probes

A probe hit cannot be passed back as a parent and create another generation of
probes.  This keeps an alert/probe feedback loop from growing recursively.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_VALID_TERMINAL_STATUSES = frozenset({"active", "expired", "retired"})


class AlertProbeValidationError(ValueError):
    """Raised when an alert or one of its proposed probe specs is malformed."""


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _validated_text(value: Any, *, field: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise AlertProbeValidationError(f"{field} entries must be strings")
    text = _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).strip())
    if not text:
        raise AlertProbeValidationError(f"{field} entries must not be empty")
    if len(text) > max_length:
        raise AlertProbeValidationError(
            f"{field} entries must be at most {max_length} characters"
        )
    if not _normalize_text(text):
        raise AlertProbeValidationError(f"{field} entries must contain letters or numbers")
    return text


def _validated_text_tuple(
    value: Any,
    *,
    field: str,
    max_items: int,
    max_length: int,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AlertProbeValidationError(f"{field} must be a sequence of strings")
    if not value and allow_empty:
        return ()
    if not value:
        raise AlertProbeValidationError(f"{field} must contain at least one entry")
    if len(value) > max_items:
        raise AlertProbeValidationError(f"{field} must contain at most {max_items} entries")

    result = []
    seen = set()
    for item in value:
        text = _validated_text(item, field=field, max_length=max_length)
        normalized = _normalize_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    if not result:
        raise AlertProbeValidationError(f"{field} must contain at least one unique entry")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AlertProbeSpec:
    """A positive/negative contrast proposed for one background probe."""

    positives: Tuple[str, ...]
    negatives: Tuple[str, ...]
    label: str = ""

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        max_terms_per_polarity: int = 4,
        max_text_length: int = 240,
        max_label_length: int = 120,
        allow_empty_negatives: bool = False,
    ) -> "AlertProbeSpec":
        if isinstance(value, cls):
            positives = _validated_text_tuple(
                value.positives,
                field="positives",
                max_items=max_terms_per_polarity,
                max_length=max_text_length,
            )
            negatives = _validated_text_tuple(
                value.negatives,
                field="negatives",
                max_items=max_terms_per_polarity,
                max_length=max_text_length,
                allow_empty=allow_empty_negatives,
            )
            label = (
                ""
                if not value.label
                else _validated_text(
                    value.label,
                    field="label",
                    max_length=max_label_length,
                )
            )
            spec = cls(positives=positives, negatives=negatives, label=label)
        elif isinstance(value, Mapping):
            positives = _validated_text_tuple(
                value.get("positives"),
                field="positives",
                max_items=max_terms_per_polarity,
                max_length=max_text_length,
            )
            negatives = _validated_text_tuple(
                value.get("negatives"),
                field="negatives",
                max_items=max_terms_per_polarity,
                max_length=max_text_length,
                allow_empty=allow_empty_negatives,
            )
            label_value = value.get("label", "")
            if label_value in (None, ""):
                label = ""
            else:
                label = _validated_text(
                    label_value,
                    field="label",
                    max_length=max_label_length,
                )
            spec = cls(positives=positives, negatives=negatives, label=label)
        else:
            raise AlertProbeValidationError("each probe spec must be a mapping")

        positive_keys = {_normalize_text(text) for text in spec.positives}
        negative_keys = {_normalize_text(text) for text in spec.negatives}
        overlap = positive_keys & negative_keys
        if overlap:
            raise AlertProbeValidationError(
                "positive and negative texts must not overlap: "
                + ", ".join(sorted(overlap))
            )
        return spec

    def canonical_key(self) -> str:
        positives = sorted(_normalize_text(text) for text in self.positives)
        negatives = sorted(_normalize_text(text) for text in self.negatives)
        return "p:" + "|".join(positives) + ";n:" + "|".join(negatives)

    def token_signature(self) -> frozenset[str]:
        result = set()
        for polarity, values in (("p", self.positives), ("n", self.negatives)):
            for text in values:
                for token in _normalize_text(text).split():
                    result.add(f"{polarity}:{token}")
        return frozenset(result)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "positives": list(self.positives),
            "negatives": list(self.negatives),
        }


@dataclass(frozen=True, slots=True)
class AlertDerivedProbe:
    """Serializable runtime record for a temporary alert-derived probe."""

    probe_id: str
    parent_alert_id: str
    channel_id: int
    spec: AlertProbeSpec
    fingerprint: str
    created_at: float
    expires_at: float
    cooldown_until: float
    status: str = "active"
    ended_at: Optional[float] = None
    end_reason: str = ""
    origin: str = "vlm_alert"
    generation: int = 0
    generated_fallback: bool = False
    confidence: str = "standard"
    parent_alert_title: str = ""
    parent_alert_description: str = ""
    parent_alert_severity: str = "normal"
    parent_alert_timestamp_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "parent_alert_id": self.parent_alert_id,
            "channel_id": self.channel_id,
            "origin": self.origin,
            "generation": self.generation,
            "generated_fallback": self.generated_fallback,
            "confidence": self.confidence,
            "parent_alert_title": self.parent_alert_title,
            "parent_alert_description": self.parent_alert_description,
            "parent_alert_severity": self.parent_alert_severity,
            "parent_alert_timestamp_ms": self.parent_alert_timestamp_ms,
            "status": self.status,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "cooldown_until": self.cooldown_until,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "fingerprint": self.fingerprint,
            "spec": self.spec.to_dict(),
        }

    def to_store_payload(
        self,
        *,
        pos_floor: float = 0.05,
        margin: float = 0.02,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """Return a payload directly consumable by ``probes_store.upsert_probe``.

        Automatic probes never request recorder bookmarks.  Their alert lineage
        and expiry remain explicit top-level fields so a daemon can cheaply
        suppress or retire stale rows without decoding an opaque blob.
        """

        label = self.spec.label or self.parent_alert_title or "VLM alert follow-up"
        return {
            "id": self.probe_id,
            "name": f"[temporary] {label}"[:160],
            "channel_id": self.channel_id,
            "positives": list(self.spec.positives),
            "negatives": list(self.spec.negatives),
            "pos_floor": float(pos_floor),
            "margin": max(0.0, float(margin)),
            "top_k": max(1, int(top_k)),
            "window_sec": max(1.0, self.expires_at - self.created_at),
            "severity": self.parent_alert_severity,
            "bookmark": False,
            "bookmark_authorized": False,
            "enabled": self.status == "active",
            "image_probe": {},
            "roi_enabled": False,
            "roi_norm": None,
            "pairs": [],
            "last_hit": None,
            "recent_hits": [],
            "temporary": True,
            "runtime_status": self.status,
            # ``origin`` is probe authorship (operator/agent/auto) as consumed by
            # the Probes board; ``source`` keeps this lifecycle's own lineage
            # guard value, which is a different axis with a similar name.
            "origin": "auto",
            "source": self.origin,
            "generation": self.generation,
            "generated_fallback": self.generated_fallback,
            "confidence": self.confidence,
            "parent_alert_id": self.parent_alert_id,
            "parent_alert_title": self.parent_alert_title,
            "parent_alert_description": self.parent_alert_description,
            "parent_alert_timestamp_ms": self.parent_alert_timestamp_ms,
            "created_at_ms": int(round(self.created_at * 1000.0)),
            "expires_at_ms": int(round(self.expires_at * 1000.0)),
            "cooldown_until_ms": int(round(self.cooldown_until * 1000.0)),
            "lifecycle": {
                "version": 1,
                "status": self.status,
                "end_reason": self.end_reason,
                "ended_at_ms": (
                    None if self.ended_at is None else int(round(self.ended_at * 1000.0))
                ),
                "fingerprint": self.fingerprint,
            },
        }


@dataclass(frozen=True, slots=True)
class AlertProbeAdmission:
    """Result returned to an internal alert-processing caller."""

    accepted: bool
    reason: str
    parent_alert_id: str
    channel_id: int
    probes: Tuple[AlertDerivedProbe, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "parent_alert_id": self.parent_alert_id,
            "channel_id": self.channel_id,
            "probes": [probe.to_dict() for probe in self.probes],
        }

    def store_payloads(
        self,
        *,
        pos_floor: float = 0.05,
        margin: float = 0.02,
        top_k: int = 6,
    ) -> Tuple[Dict[str, Any], ...]:
        """Return accepted probes in the existing probe-store shape."""

        if not self.accepted:
            return ()
        return tuple(
            probe.to_store_payload(
                pos_floor=pos_floor,
                margin=margin,
                top_k=top_k,
            )
            for probe in self.probes
        )


@dataclass(frozen=True, slots=True)
class AlertEventContext:
    """Normalized subset of an EVA VLM ``alert_event``."""

    parent_alert_id: str
    channel_id: int
    title: str
    description: str
    severity: str
    timestamp_ms: Optional[int]

    @classmethod
    def from_mapping(
        cls,
        event: Mapping[str, Any],
        *,
        channel_id: Optional[int] = None,
    ) -> "AlertEventContext":
        if not isinstance(event, Mapping):
            raise AlertProbeValidationError("alert event must be a mapping")
        raw_channel = channel_id if channel_id is not None else event.get("channel_id")
        normalized_channel = AlertProbeLifecycle._validate_channel_id(raw_channel)
        title = _validated_text(
            event.get("title") or "Event",
            field="alert title",
            max_length=120,
        )
        raw_description = event.get("description")
        description = (
            ""
            if raw_description in (None, "")
            else _validated_text(
                raw_description,
                field="alert description",
                max_length=600,
            )
        )
        severity = _normalize_text(str(event.get("severity") or "normal"))[:20] or "normal"
        raw_timestamp = event.get("timestamp_ms")
        if raw_timestamp is None:
            timestamp_ms = None
        elif isinstance(raw_timestamp, bool):
            raise AlertProbeValidationError("alert timestamp_ms must be an integer")
        else:
            try:
                timestamp_ms = int(raw_timestamp)
            except (TypeError, ValueError) as exc:
                raise AlertProbeValidationError(
                    "alert timestamp_ms must be an integer"
                ) from exc
            if timestamp_ms < 0 or str(timestamp_ms) != str(raw_timestamp).strip():
                raise AlertProbeValidationError("alert timestamp_ms must be an integer")

        explicit_id = str(event.get("id") or event.get("alert_id") or "").strip()
        if explicit_id:
            parent_alert_id = AlertProbeLifecycle._validate_alert_id(explicit_id)
        else:
            identity = json.dumps(
                {
                    "channel_id": normalized_channel,
                    "timestamp_ms": timestamp_ms,
                    "title": _normalize_text(title),
                    "description": _normalize_text(description),
                    "severity": severity,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=12).hexdigest()
            parent_alert_id = f"vlm-alert-{digest}"
        return cls(
            parent_alert_id=parent_alert_id,
            channel_id=normalized_channel,
            title=title,
            description=description,
            severity=severity,
            timestamp_ms=timestamp_ms,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_alert_id": self.parent_alert_id,
            "channel_id": self.channel_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "timestamp_ms": self.timestamp_ms,
        }


def derive_parent_alert_id(
    event: Mapping[str, Any],
    *,
    channel_id: Optional[int] = None,
) -> str:
    """Return the shared archive/probe lineage id for one VLM alert event."""

    return AlertEventContext.from_mapping(
        event,
        channel_id=channel_id,
    ).parent_alert_id


SimilarityFn = Callable[[AlertProbeSpec, AlertProbeSpec], float]


class AlertProbeLifecycle:
    """Admit, expire, deduplicate, and inspect temporary alert probes.

    Admission is transactional: an alert produces between ``min_probes`` and
    ``max_probes`` active probes, or it produces none.  Caps and deduplication
    are checked before any record is inserted.

    ``similarity_fn`` can bridge to an embedding-based text similarity routine.
    The stdlib-only default uses polarity-aware token Jaccard similarity.
    """

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        *,
        min_probes: int = 2,
        max_probes: int = 4,
        per_channel_cap: int = 8,
        global_cap: int = 64,
        default_ttl_seconds: float = 300.0,
        min_ttl_seconds: float = 5.0,
        max_ttl_seconds: float = 3600.0,
        cooldown_seconds: float = 5.0,
        semantic_dedupe_threshold: float = 0.8,
        history_limit: int = 2048,
        clock: Callable[[], float] = time.time,
        similarity_fn: Optional[SimilarityFn] = None,
    ) -> None:
        if min_probes < 1 or max_probes < min_probes:
            raise ValueError("probe count limits are invalid")
        if per_channel_cap < max_probes:
            raise ValueError("per_channel_cap must permit one complete alert admission")
        if global_cap < per_channel_cap:
            raise ValueError("global_cap must be at least per_channel_cap")
        if not 0.0 <= semantic_dedupe_threshold <= 1.0:
            raise ValueError("semantic_dedupe_threshold must be between 0 and 1")
        if min_ttl_seconds <= 0 or max_ttl_seconds < min_ttl_seconds:
            raise ValueError("TTL limits are invalid")
        if not min_ttl_seconds <= default_ttl_seconds <= max_ttl_seconds:
            raise ValueError("default TTL must be inside TTL limits")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if history_limit < global_cap:
            raise ValueError("history_limit must be at least global_cap")

        self.min_probes = int(min_probes)
        self.max_probes = int(max_probes)
        self.per_channel_cap = int(per_channel_cap)
        self.global_cap = int(global_cap)
        self.default_ttl_seconds = float(default_ttl_seconds)
        self.min_ttl_seconds = float(min_ttl_seconds)
        self.max_ttl_seconds = float(max_ttl_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.semantic_dedupe_threshold = float(semantic_dedupe_threshold)
        self.history_limit = int(history_limit)
        self._clock = clock
        self._similarity_fn = similarity_fn or self._default_similarity
        self._records: Dict[str, AlertDerivedProbe] = {}
        self._parent_alerts: Dict[str, Tuple[str, ...]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _default_similarity(left: AlertProbeSpec, right: AlertProbeSpec) -> float:
        if left.canonical_key() == right.canonical_key():
            return 1.0
        left_terms = left.token_signature()
        right_terms = right.token_signature()
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / float(len(left_terms | right_terms))

    @staticmethod
    def _fingerprint(spec: AlertProbeSpec) -> str:
        return hashlib.blake2s(spec.canonical_key().encode("utf-8"), digest_size=12).hexdigest()

    @staticmethod
    def _probe_id(
        *,
        parent_alert_id: str,
        channel_id: int,
        ordinal: int,
        fingerprint: str,
    ) -> str:
        seed = f"{parent_alert_id}\x1f{channel_id}\x1f{ordinal}\x1f{fingerprint}"
        digest = hashlib.blake2s(seed.encode("utf-8"), digest_size=12).hexdigest()
        return f"alert-probe-{digest}"

    def _now(self, now: Optional[float]) -> float:
        value = self._clock() if now is None else now
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise AlertProbeValidationError("now must be a finite timestamp") from exc
        if not math.isfinite(result):
            raise AlertProbeValidationError("now must be a finite timestamp")
        return result

    def _validate_ttl(self, ttl_seconds: Optional[float]) -> float:
        raw = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        try:
            ttl = float(raw)
        except (TypeError, ValueError) as exc:
            raise AlertProbeValidationError("ttl_seconds must be a finite number") from exc
        if not math.isfinite(ttl):
            raise AlertProbeValidationError("ttl_seconds must be a finite number")
        if ttl < self.min_ttl_seconds or ttl > self.max_ttl_seconds:
            raise AlertProbeValidationError(
                f"ttl_seconds must be between {self.min_ttl_seconds:g} "
                f"and {self.max_ttl_seconds:g}"
            )
        return ttl

    @staticmethod
    def _validate_alert_id(value: Any) -> str:
        if not isinstance(value, str):
            raise AlertProbeValidationError("parent_alert_id must be a string")
        result = value.strip()
        if not result:
            raise AlertProbeValidationError("parent_alert_id must not be empty")
        if len(result) > 200:
            raise AlertProbeValidationError("parent_alert_id must be at most 200 characters")
        return result

    @staticmethod
    def _validate_channel_id(value: Any) -> int:
        if isinstance(value, bool):
            raise AlertProbeValidationError("channel_id must be a positive integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise AlertProbeValidationError("channel_id must be a positive integer") from exc
        if result <= 0 or str(result) != str(value).strip():
            raise AlertProbeValidationError("channel_id must be a positive integer")
        return result

    def _expire_locked(self, now: float) -> Tuple[AlertDerivedProbe, ...]:
        expired = []
        for probe_id in sorted(self._records):
            probe = self._records[probe_id]
            if probe.status == "active" and probe.expires_at <= now:
                closed = replace(
                    probe,
                    status="expired",
                    ended_at=probe.expires_at,
                    end_reason="ttl_elapsed",
                )
                self._records[probe_id] = closed
                expired.append(closed)
        self._prune_history_locked(now)
        return tuple(expired)

    def _prune_history_locked(self, now: float) -> None:
        excess = len(self._records) - self.history_limit
        if excess <= 0:
            return
        removable = sorted(
            (
                probe
                for probe in self._records.values()
                if probe.status != "active" and probe.cooldown_until <= now
            ),
            key=lambda probe: (
                probe.ended_at if probe.ended_at is not None else probe.expires_at,
                probe.probe_id,
            ),
        )
        removed_ids = {
            probe.probe_id
            for probe in removable[:excess]
        }
        if not removed_ids:
            return
        for probe_id in removed_ids:
            self._records.pop(probe_id, None)
        for alert_id, probe_ids in tuple(self._parent_alerts.items()):
            retained = tuple(probe_id for probe_id in probe_ids if probe_id in self._records)
            if retained:
                self._parent_alerts[alert_id] = retained
            else:
                self._parent_alerts.pop(alert_id, None)

    def _active_locked(self) -> Tuple[AlertDerivedProbe, ...]:
        return tuple(
            self._records[probe_id]
            for probe_id in sorted(self._records)
            if self._records[probe_id].status == "active"
        )

    def _dedupe_candidates_locked(
        self,
        specs: Sequence[AlertProbeSpec],
        *,
        channel_id: int,
        now: float,
        allow_related_batch_specs: bool = False,
    ) -> Optional[str]:
        comparison_records = [
            probe
            for probe in self._records.values()
            if probe.channel_id == channel_id
            and (probe.status == "active" or probe.cooldown_until > now)
        ]
        for index, spec in enumerate(specs):
            if not allow_related_batch_specs:
                for prior in specs[:index]:
                    if self._safe_similarity(spec, prior) >= self.semantic_dedupe_threshold:
                        return "duplicate_probe_specs_in_alert"
            for prior in comparison_records:
                if self._safe_similarity(spec, prior.spec) >= self.semantic_dedupe_threshold:
                    if prior.status == "active":
                        return "semantically_duplicate_active_probe"
                    return "semantically_duplicate_probe_in_cooldown"
        return None

    def _safe_similarity(self, left: AlertProbeSpec, right: AlertProbeSpec) -> float:
        try:
            score = float(self._similarity_fn(left, right))
        except Exception as exc:
            raise AlertProbeValidationError("probe similarity evaluation failed") from exc
        if not math.isfinite(score):
            raise AlertProbeValidationError("probe similarity must be finite")
        return max(0.0, min(1.0, score))

    def admit_alert(
        self,
        *,
        parent_alert_id: str,
        channel_id: int,
        specs: Sequence[Any],
        ttl_seconds: Optional[float] = None,
        origin: str = "vlm_alert",
        generation: int = 0,
        source_probe_id: Optional[str] = None,
        now: Optional[float] = None,
        alert_title: str = "",
        alert_description: str = "",
        alert_severity: str = "normal",
        alert_timestamp_ms: Optional[int] = None,
        generated_fallback: bool = False,
    ) -> AlertProbeAdmission:
        """Admit 2-4 probes for a root VLM alert, or reject the whole batch.

        ``origin``, ``generation``, and ``source_probe_id`` are explicit guards
        for internal callers.  Anything other than a generation-zero VLM alert
        is rejected before probe text is considered.
        """

        alert_id = self._validate_alert_id(parent_alert_id)
        channel = self._validate_channel_id(channel_id)
        current = self._now(now)
        normalized_title = (
            ""
            if not alert_title
            else _validated_text(alert_title, field="alert title", max_length=120)
        )
        normalized_description = (
            ""
            if not alert_description
            else _validated_text(
                alert_description,
                field="alert description",
                max_length=600,
            )
        )
        normalized_severity = _normalize_text(str(alert_severity or "normal"))[:20] or "normal"
        if alert_timestamp_ms is not None:
            if isinstance(alert_timestamp_ms, bool):
                raise AlertProbeValidationError("alert_timestamp_ms must be an integer")
            try:
                normalized_timestamp_ms = int(alert_timestamp_ms)
            except (TypeError, ValueError) as exc:
                raise AlertProbeValidationError(
                    "alert_timestamp_ms must be an integer"
                ) from exc
        else:
            normalized_timestamp_ms = None

        if origin != "vlm_alert" or generation != 0 or source_probe_id:
            return AlertProbeAdmission(
                accepted=False,
                reason="recursive_probe_derivation_forbidden",
                parent_alert_id=alert_id,
                channel_id=channel,
            )
        if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
            raise AlertProbeValidationError("specs must be a sequence")
        if len(specs) < self.min_probes or len(specs) > self.max_probes:
            raise AlertProbeValidationError(
                f"each alert must propose {self.min_probes}-{self.max_probes} probes"
            )
        validated_specs = tuple(
            AlertProbeSpec.from_value(
                spec,
                allow_empty_negatives=bool(generated_fallback),
            )
            for spec in specs
        )
        ttl = self._validate_ttl(ttl_seconds)

        with self._lock:
            self._expire_locked(current)
            if alert_id in self._parent_alerts:
                probes = tuple(
                    self._records[probe_id]
                    for probe_id in self._parent_alerts[alert_id]
                    if probe_id in self._records
                )
                return AlertProbeAdmission(
                    accepted=False,
                    reason="parent_alert_already_processed",
                    parent_alert_id=alert_id,
                    channel_id=channel,
                    probes=probes,
                )

            duplicate_reason = self._dedupe_candidates_locked(
                validated_specs,
                channel_id=channel,
                now=current,
                allow_related_batch_specs=bool(generated_fallback),
            )
            if duplicate_reason:
                return AlertProbeAdmission(
                    accepted=False,
                    reason=duplicate_reason,
                    parent_alert_id=alert_id,
                    channel_id=channel,
                )

            active = self._active_locked()
            channel_active_count = sum(1 for probe in active if probe.channel_id == channel)
            if channel_active_count + len(validated_specs) > self.per_channel_cap:
                return AlertProbeAdmission(
                    accepted=False,
                    reason="per_channel_cap_exceeded",
                    parent_alert_id=alert_id,
                    channel_id=channel,
                )
            if len(active) + len(validated_specs) > self.global_cap:
                return AlertProbeAdmission(
                    accepted=False,
                    reason="global_cap_exceeded",
                    parent_alert_id=alert_id,
                    channel_id=channel,
                )

            expires_at = current + ttl
            cooldown_until = expires_at + self.cooldown_seconds
            created = []
            for ordinal, spec in enumerate(validated_specs):
                fingerprint = self._fingerprint(spec)
                probe = AlertDerivedProbe(
                    probe_id=self._probe_id(
                        parent_alert_id=alert_id,
                        channel_id=channel,
                        ordinal=ordinal,
                        fingerprint=fingerprint,
                    ),
                    parent_alert_id=alert_id,
                    channel_id=channel,
                    spec=spec,
                    fingerprint=fingerprint,
                    created_at=current,
                    expires_at=expires_at,
                    cooldown_until=cooldown_until,
                    generated_fallback=bool(generated_fallback),
                    confidence="low" if generated_fallback else "standard",
                    parent_alert_title=normalized_title,
                    parent_alert_description=normalized_description,
                    parent_alert_severity=normalized_severity,
                    parent_alert_timestamp_ms=normalized_timestamp_ms,
                )
                self._records[probe.probe_id] = probe
                created.append(probe)
            self._parent_alerts[alert_id] = tuple(probe.probe_id for probe in created)
            return AlertProbeAdmission(
                accepted=True,
                reason="admitted",
                parent_alert_id=alert_id,
                channel_id=channel,
                probes=tuple(created),
            )

    def admit_alert_event(
        self,
        event: Mapping[str, Any],
        *,
        specs: Optional[Sequence[Any]] = None,
        channel_id: Optional[int] = None,
        ttl_seconds: Optional[float] = None,
        now: Optional[float] = None,
        allow_generated_fallback: bool = False,
    ) -> AlertProbeAdmission:
        """Ingest an EVA alert event and return store-ready lifecycle records.

        Probe contrasts may be supplied explicitly or in ``event.probe_specs``.
        Missing specs are rejected instead of fabricating weak CLIP negatives
        from natural-language negation.
        """

        context = AlertEventContext.from_mapping(event, channel_id=channel_id)
        raw_specs = specs
        if raw_specs is None:
            candidate = event.get("probe_specs")
            raw_specs = (
                candidate
                if isinstance(candidate, Sequence)
                and not isinstance(candidate, (str, bytes, bytearray))
                else None
            )
        if raw_specs is None:
            if not allow_generated_fallback:
                raise AlertProbeValidationError(
                    "alert event must provide 2-4 explicit positive/negative probe_specs "
                    "or opt in to generated fallback probes"
                )
            raw_specs = self.build_fallback_specs(context)
        raw_origin = str(event.get("origin") or event.get("source") or "vlm_alert").strip()
        raw_generation = event.get("generation", 0)
        try:
            generation = int(raw_generation)
        except (TypeError, ValueError):
            generation = 1
        source_probe_id = str(event.get("source_probe_id") or "").strip() or None
        return self.admit_alert(
            parent_alert_id=context.parent_alert_id,
            channel_id=context.channel_id,
            specs=raw_specs,
            ttl_seconds=ttl_seconds,
            now=now,
            origin=raw_origin,
            generation=generation,
            source_probe_id=source_probe_id,
            alert_title=context.title,
            alert_description=context.description,
            alert_severity=context.severity,
            alert_timestamp_ms=context.timestamp_ms,
            generated_fallback=bool(
                allow_generated_fallback
                and specs is None
                and not event.get("probe_specs")
            ),
        )

    @staticmethod
    def build_fallback_specs(
        event: AlertEventContext | Mapping[str, Any],
        *,
        channel_id: Optional[int] = None,
    ) -> Tuple[AlertProbeSpec, AlertProbeSpec]:
        """Build two low-confidence positive-only probes for today's alert schema.

        This fallback is deliberately explicit and must be enabled by the
        caller.  It avoids synthetic natural-language negatives because CLIP
        does not reliably interpret negation.  The richer second variant gives
        the text encoder scene context while the first keeps the alert title
        concise.
        """

        context = (
            event
            if isinstance(event, AlertEventContext)
            else AlertEventContext.from_mapping(event, channel_id=channel_id)
        )
        concise = context.title
        detail_parts = [f"security camera view of {context.title}"]
        if context.description:
            detail_parts.append(context.description)
        detailed = ". ".join(detail_parts)
        if len(detailed) > 240:
            detailed = detailed[:237].rstrip() + "..."
        return (
            AlertProbeSpec(
                label=f"{context.title} — concise"[:120],
                positives=(concise,),
                negatives=(),
            ),
            AlertProbeSpec(
                label=f"{context.title} — scene context"[:120],
                positives=(detailed,),
                negatives=(),
            ),
        )

    def expire(self, *, now: Optional[float] = None) -> Tuple[AlertDerivedProbe, ...]:
        """Expire due probes and return the records changed by this call."""

        current = self._now(now)
        with self._lock:
            return self._expire_locked(current)

    def retire_probe(
        self,
        probe_id: str,
        *,
        reason: str = "retired",
        now: Optional[float] = None,
    ) -> Optional[AlertDerivedProbe]:
        """Retire an active probe while retaining its cooldown and lineage."""

        current = self._now(now)
        reason_text = _validated_text(reason, field="reason", max_length=120)
        with self._lock:
            self._expire_locked(current)
            probe = self._records.get(str(probe_id))
            if probe is None:
                return None
            if probe.status != "active":
                return probe
            retired = replace(
                probe,
                status="retired",
                ended_at=current,
                end_reason=reason_text,
            )
            self._records[probe.probe_id] = retired
            return retired

    def active_probes(
        self,
        *,
        channel_id: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Tuple[AlertDerivedProbe, ...]:
        current = self._now(now)
        channel = None if channel_id is None else self._validate_channel_id(channel_id)
        with self._lock:
            self._expire_locked(current)
            probes = self._active_locked()
            if channel is not None:
                probes = tuple(probe for probe in probes if probe.channel_id == channel)
            return probes

    def status(
        self,
        *,
        channel_id: Optional[int] = None,
        include_inactive: bool = True,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return a stable, JSON-safe runtime snapshot."""

        current = self._now(now)
        channel = None if channel_id is None else self._validate_channel_id(channel_id)
        with self._lock:
            self._expire_locked(current)
            records: Iterable[AlertDerivedProbe] = self._records.values()
            if channel is not None:
                records = (probe for probe in records if probe.channel_id == channel)
            if not include_inactive:
                records = (probe for probe in records if probe.status == "active")
            ordered = sorted(
                records,
                key=lambda probe: (
                    probe.channel_id,
                    probe.created_at,
                    probe.parent_alert_id,
                    probe.probe_id,
                ),
            )
            counts = {status: 0 for status in sorted(_VALID_TERMINAL_STATUSES)}
            for probe in ordered:
                counts[probe.status] += 1
            return {
                "version": self.SNAPSHOT_VERSION,
                "now": current,
                "channel_id": channel,
                "limits": {
                    "min_probes_per_alert": self.min_probes,
                    "max_probes_per_alert": self.max_probes,
                    "per_channel_cap": self.per_channel_cap,
                    "global_cap": self.global_cap,
                    "default_ttl_seconds": self.default_ttl_seconds,
                    "cooldown_seconds": self.cooldown_seconds,
                    "semantic_dedupe_threshold": self.semantic_dedupe_threshold,
                    "history_limit": self.history_limit,
                },
                "counts": counts,
                "probes": [probe.to_dict() for probe in ordered],
            }

    def dumps(
        self,
        *,
        channel_id: Optional[int] = None,
        include_inactive: bool = True,
        now: Optional[float] = None,
    ) -> str:
        """Return canonical JSON suitable for logs, snapshots, or DB payloads."""

        return json.dumps(
            self.status(
                channel_id=channel_id,
                include_inactive=include_inactive,
                now=now,
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


__all__ = [
    "AlertDerivedProbe",
    "AlertEventContext",
    "AlertProbeAdmission",
    "AlertProbeLifecycle",
    "AlertProbeSpec",
    "AlertProbeValidationError",
]
