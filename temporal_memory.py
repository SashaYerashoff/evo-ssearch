"""Deterministic temporal-memory primitives for semantic video observations.

The module is deliberately independent from the VLM, rollup scheduler, incident
store, and runtime state.  It gives those layers server-owned identities and a
gap-aware way to preserve concurrent episodes without treating missing coverage
as evidence that the scene returned to routine.

All timestamps are integer milliseconds on one shared source timeline.  DTOs
contain only bounded text and durable references; pixels and arbitrary payloads
do not belong here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Iterable, Sequence


MAX_ID_CHARS = 160
MAX_KEY_CHARS = 160
MAX_LABEL_CHARS = 240
MAX_REASON_CHARS = 240
MAX_EVIDENCE_REFS = 32
MAX_OBSERVATIONS_PER_EPISODE = 512
MAX_EPISODES_PER_SERIES = 128
MAX_SEGMENTATION_OBSERVATIONS = 4096
MAX_CHILD_DISPOSITIONS = 4096
DEFAULT_MAX_OBSERVED_GAP_MS = 15 * 60 * 1000


class ObservationKind(str, Enum):
    EVENT = "event"
    ROUTINE_GAP = "routine_gap"
    COVERAGE_GAP = "coverage_gap"


class ObservationState(str, Enum):
    NEW = "new"
    CONTINUING = "continuing"
    RESOLVED = "resolved"
    UNCERTAIN = "uncertain"


class EpisodeStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    ENDED_BY_ROUTINE = "ended_by_routine"
    ENDED_BY_OBSERVED_GAP = "ended_by_observed_gap"


class ChildDispositionKind(str, Enum):
    STARTED_EPISODE = "started_episode"
    CONTINUED_EPISODE = "continued_episode"
    RESOLVED_EPISODE = "resolved_episode"
    ROUTINE_GAP = "routine_gap"
    COVERAGE_GAP_KEEP = "coverage_gap_keep"
    UNCLASSIFIED_KEEP = "unclassified_keep"


class AttentionBudget(IntEnum):
    """Pure 2/4/8-point temporal attention semantics.

    ``BOUNDARIES`` keeps a first/last pair, ``EVOLUTION`` samples four points,
    and ``DENSE`` samples eight.  The enum does not imply scheduler policy.
    """

    BOUNDARIES = 2
    EVOLUTION = 4
    DENSE = 8


def _timestamp(value: int, name: str) -> int:
    try:
        timestamp = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if timestamp < 0:
        raise ValueError(f"{name} must be non-negative")
    return timestamp


def _positive_channel(value: int) -> int:
    try:
        channel_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("channel_id must be an integer") from exc
    if channel_id <= 0:
        raise ValueError("channel_id must be positive")
    return channel_id


def _bounded_text(value: object, name: str, limit: int, *, required: bool) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise ValueError(f"{name} must not contain NUL characters")
    if required and not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > limit:
        raise ValueError(f"{name} must contain at most {limit} characters")
    return text


def _identifier(value: object, name: str) -> str:
    return _bounded_text(value, name, MAX_ID_CHARS, required=True)


def _semantic_key(value: object, name: str = "semantic_key") -> str:
    text = _bounded_text(value, name, MAX_KEY_CHARS, required=True)
    return " ".join(text.casefold().split())


def _stable_id(prefix: str, *parts: object) -> str:
    canonical = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:32]
    return f"{prefix}-{digest}"


def _bounded_unique_refs(
    values: Iterable[object],
    name: str,
    *,
    limit: int,
    semantic: bool = False,
) -> tuple[str, ...]:
    refs: list[str] = []
    for raw in values:
        ref = _semantic_key(raw, name) if semantic else _identifier(raw, name)
        if ref not in refs:
            refs.append(ref)
        if len(refs) > limit:
            raise ValueError(f"{name} must contain at most {limit} items")
    return tuple(refs)


@dataclass(frozen=True)
class TemporalObservation:
    """One bounded observation with an identity derived outside the model."""

    observation_id: str
    channel_id: int
    source_batch_id: str
    ordinal: int
    kind: ObservationKind
    start_ms: int
    end_ms: int
    state: ObservationState | None = None
    semantic_key: str = ""
    label: str = ""
    applies_to: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        observation_id = _identifier(self.observation_id, "observation_id")
        object.__setattr__(self, "channel_id", _positive_channel(self.channel_id))
        object.__setattr__(
            self, "source_batch_id", _identifier(self.source_batch_id, "source_batch_id")
        )
        try:
            ordinal = int(self.ordinal)
        except (TypeError, ValueError) as exc:
            raise ValueError("ordinal must be an integer") from exc
        if ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        object.__setattr__(self, "ordinal", ordinal)
        expected_id = _stable_id(
            "obs", "v1", self.channel_id, self.source_batch_id, self.ordinal
        )
        if observation_id != expected_id:
            raise ValueError("observation_id is not the server-derived identity")
        object.__setattr__(self, "observation_id", observation_id)

        kind = self.kind if isinstance(self.kind, ObservationKind) else ObservationKind(self.kind)
        object.__setattr__(self, "kind", kind)
        start_ms = _timestamp(self.start_ms, "start_ms")
        end_ms = _timestamp(self.end_ms, "end_ms")
        if end_ms < start_ms:
            raise ValueError("end_ms must not precede start_ms")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)

        state = self.state
        if state is not None and not isinstance(state, ObservationState):
            state = ObservationState(state)
        if kind is ObservationKind.EVENT and state is None:
            raise ValueError("event observations require state")
        if kind is not ObservationKind.EVENT and state is not None:
            raise ValueError("gap observations must not carry event state")
        object.__setattr__(self, "state", state)

        semantic_key = str(self.semantic_key or "").strip()
        if kind is ObservationKind.EVENT:
            semantic_key = _semantic_key(semantic_key)
        elif semantic_key:
            semantic_key = _semantic_key(semantic_key)
        object.__setattr__(self, "semantic_key", semantic_key)
        label = _bounded_text(
            self.label,
            "label",
            MAX_LABEL_CHARS,
            required=kind is ObservationKind.EVENT,
        )
        object.__setattr__(self, "label", label)
        object.__setattr__(
            self,
            "applies_to",
            _bounded_unique_refs(
                self.applies_to,
                "applies_to",
                limit=MAX_EVIDENCE_REFS,
                semantic=True,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _bounded_unique_refs(
                self.evidence_refs,
                "evidence_refs",
                limit=MAX_EVIDENCE_REFS,
            ),
        )

    def applies_to_key(self, semantic_key: str) -> bool:
        return not self.applies_to or _semantic_key(semantic_key) in self.applies_to

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "channel_id": self.channel_id,
            "source_batch_id": self.source_batch_id,
            "ordinal": self.ordinal,
            "kind": self.kind.value,
            "state": self.state.value if self.state is not None else None,
            "semantic_key": self.semantic_key,
            "label": self.label,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "applies_to": list(self.applies_to),
            "evidence_refs": list(self.evidence_refs),
        }


def make_observation(
    *,
    channel_id: int,
    source_batch_id: str,
    ordinal: int,
    kind: ObservationKind | str,
    start_ms: int,
    end_ms: int | None = None,
    state: ObservationState | str | None = None,
    semantic_key: str = "",
    label: str = "",
    applies_to: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
) -> TemporalObservation:
    """Create an observation whose identity cannot be supplied by the model.

    Identity is content-independent: reprocessing the same channel, stable batch,
    and child ordinal produces the same ID even if a label is reworded.
    """

    normalized_channel = _positive_channel(channel_id)
    normalized_batch = _identifier(source_batch_id, "source_batch_id")
    try:
        normalized_ordinal = int(ordinal)
    except (TypeError, ValueError) as exc:
        raise ValueError("ordinal must be an integer") from exc
    if normalized_ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    normalized_kind = kind if isinstance(kind, ObservationKind) else ObservationKind(kind)
    observation_id = _stable_id(
        "obs", "v1", normalized_channel, normalized_batch, normalized_ordinal
    )
    return TemporalObservation(
        observation_id=observation_id,
        channel_id=normalized_channel,
        source_batch_id=normalized_batch,
        ordinal=normalized_ordinal,
        kind=normalized_kind,
        state=state,
        semantic_key=semantic_key,
        label=label,
        start_ms=start_ms,
        end_ms=start_ms if end_ms is None else end_ms,
        applies_to=tuple(applies_to),
        evidence_refs=tuple(evidence_refs),
    )


@dataclass(frozen=True)
class TemporalEpisode:
    """A perceptual episode; it is deliberately not an operator incident."""

    episode_id: str
    channel_id: int
    semantic_key: str
    label: str
    status: EpisodeStatus
    start_ms: int
    last_observed_ms: int
    boundary_at_ms: int | None
    boundary_observation_id: str | None
    observation_ids: tuple[str, ...]
    coverage_gap_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _positive_channel(self.channel_id))
        object.__setattr__(self, "semantic_key", _semantic_key(self.semantic_key))
        object.__setattr__(
            self,
            "label",
            _bounded_text(self.label, "label", MAX_LABEL_CHARS, required=True),
        )
        status = (
            self.status
            if isinstance(self.status, EpisodeStatus)
            else EpisodeStatus(self.status)
        )
        object.__setattr__(self, "status", status)
        start_ms = _timestamp(self.start_ms, "start_ms")
        last_observed_ms = _timestamp(self.last_observed_ms, "last_observed_ms")
        if last_observed_ms < start_ms:
            raise ValueError("last_observed_ms must not precede start_ms")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "last_observed_ms", last_observed_ms)
        if self.boundary_at_ms is not None:
            boundary_at_ms = _timestamp(self.boundary_at_ms, "boundary_at_ms")
            if boundary_at_ms < last_observed_ms:
                raise ValueError("boundary_at_ms must not precede last_observed_ms")
            object.__setattr__(self, "boundary_at_ms", boundary_at_ms)
        if status is EpisodeStatus.OPEN and self.boundary_at_ms is not None:
            raise ValueError("open episodes must not carry a boundary")
        if status is not EpisodeStatus.OPEN and self.boundary_at_ms is None:
            raise ValueError("ended episodes require boundary_at_ms")
        if self.boundary_observation_id is not None:
            object.__setattr__(
                self,
                "boundary_observation_id",
                _identifier(self.boundary_observation_id, "boundary_observation_id"),
            )
        observation_ids = _bounded_unique_refs(
            self.observation_ids,
            "observation_ids",
            limit=MAX_OBSERVATIONS_PER_EPISODE,
        )
        object.__setattr__(
            self,
            "observation_ids",
            observation_ids,
        )
        if not self.observation_ids:
            raise ValueError("observation_ids must not be empty")
        episode_id = _identifier(self.episode_id, "episode_id")
        expected_id = _stable_id(
            "ep", "v1", self.channel_id, self.observation_ids[0]
        )
        if episode_id != expected_id:
            raise ValueError("episode_id is not the server-derived identity")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(
            self,
            "coverage_gap_ids",
            _bounded_unique_refs(
                self.coverage_gap_ids,
                "coverage_gap_ids",
                limit=MAX_EVIDENCE_REFS,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _bounded_unique_refs(
                self.evidence_refs,
                "evidence_refs",
                limit=MAX_EVIDENCE_REFS,
            ),
        )

    @property
    def observed_span_ms(self) -> int:
        return self.last_observed_ms - self.start_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "channel_id": self.channel_id,
            "semantic_key": self.semantic_key,
            "label": self.label,
            "status": self.status.value,
            "start_ms": self.start_ms,
            "last_observed_ms": self.last_observed_ms,
            "boundary_at_ms": self.boundary_at_ms,
            "boundary_observation_id": self.boundary_observation_id,
            "observed_span_ms": self.observed_span_ms,
            "observation_ids": list(self.observation_ids),
            "coverage_gap_ids": list(self.coverage_gap_ids),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class TemporalSeries:
    """An explicit relation between episodes; segmentation never invents one."""

    series_id: str
    channel_id: int
    series_key: str
    episode_ids: tuple[str, ...]
    first_observed_ms: int
    last_observed_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _positive_channel(self.channel_id))
        object.__setattr__(self, "series_key", _semantic_key(self.series_key, "series_key"))
        object.__setattr__(
            self,
            "episode_ids",
            _bounded_unique_refs(
                self.episode_ids,
                "episode_ids",
                limit=MAX_EPISODES_PER_SERIES,
            ),
        )
        if not self.episode_ids:
            raise ValueError("episode_ids must not be empty")
        series_id = _identifier(self.series_id, "series_id")
        expected_id = _stable_id(
            "series", "v1", self.channel_id, self.episode_ids[0]
        )
        if series_id != expected_id:
            raise ValueError("series_id is not the server-derived identity")
        object.__setattr__(self, "series_id", series_id)
        first = _timestamp(self.first_observed_ms, "first_observed_ms")
        last = _timestamp(self.last_observed_ms, "last_observed_ms")
        if last < first:
            raise ValueError("last_observed_ms must not precede first_observed_ms")
        object.__setattr__(self, "first_observed_ms", first)
        object.__setattr__(self, "last_observed_ms", last)

    def to_dict(self) -> dict[str, object]:
        return {
            "series_id": self.series_id,
            "channel_id": self.channel_id,
            "series_key": self.series_key,
            "episode_ids": list(self.episode_ids),
            "first_observed_ms": self.first_observed_ms,
            "last_observed_ms": self.last_observed_ms,
        }


@dataclass(frozen=True)
class ChildDisposition:
    child_id: str
    disposition: ChildDispositionKind
    episode_id: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_id", _identifier(self.child_id, "child_id"))
        disposition = (
            self.disposition
            if isinstance(self.disposition, ChildDispositionKind)
            else ChildDispositionKind(self.disposition)
        )
        object.__setattr__(self, "disposition", disposition)
        if self.episode_id is not None:
            object.__setattr__(
                self, "episode_id", _identifier(self.episode_id, "episode_id")
            )
        object.__setattr__(
            self,
            "reason",
            _bounded_text(self.reason, "reason", MAX_REASON_CHARS, required=False),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "child_id": self.child_id,
            "disposition": self.disposition.value,
            "episode_id": self.episode_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TemporalSegmentation:
    episodes: tuple[TemporalEpisode, ...]
    dispositions: tuple[ChildDisposition, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes],
            "dispositions": [item.to_dict() for item in self.dispositions],
        }


@dataclass
class _EpisodeBuilder:
    episode_id: str
    channel_id: int
    semantic_key: str
    label: str
    start_ms: int
    last_observed_ms: int
    observation_ids: list[str] = field(default_factory=list)
    coverage_gap_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_observation(cls, observation: TemporalObservation) -> _EpisodeBuilder:
        return cls(
            episode_id=_stable_id(
                "ep",
                "v1",
                observation.channel_id,
                observation.observation_id,
            ),
            channel_id=observation.channel_id,
            semantic_key=observation.semantic_key,
            label=observation.label,
            start_ms=observation.start_ms,
            last_observed_ms=observation.end_ms,
        )

    def append(self, observation: TemporalObservation) -> None:
        if len(self.observation_ids) >= MAX_OBSERVATIONS_PER_EPISODE:
            raise ValueError(
                "episode exceeds MAX_OBSERVATIONS_PER_EPISODE; split before appending"
            )
        self.observation_ids.append(observation.observation_id)
        self.last_observed_ms = max(self.last_observed_ms, observation.end_ms)
        for ref in observation.evidence_refs:
            if ref not in self.evidence_refs:
                if len(self.evidence_refs) >= MAX_EVIDENCE_REFS:
                    break
                self.evidence_refs.append(ref)

    def add_coverage_gap(self, observation_id: str) -> None:
        if observation_id in self.coverage_gap_ids:
            return
        if len(self.coverage_gap_ids) < MAX_EVIDENCE_REFS:
            self.coverage_gap_ids.append(observation_id)

    def freeze(
        self,
        status: EpisodeStatus,
        *,
        boundary_at_ms: int | None = None,
        boundary_observation_id: str | None = None,
    ) -> TemporalEpisode:
        return TemporalEpisode(
            episode_id=self.episode_id,
            channel_id=self.channel_id,
            semantic_key=self.semantic_key,
            label=self.label,
            status=status,
            start_ms=self.start_ms,
            last_observed_ms=self.last_observed_ms,
            boundary_at_ms=boundary_at_ms,
            boundary_observation_id=boundary_observation_id,
            observation_ids=tuple(self.observation_ids),
            coverage_gap_ids=tuple(self.coverage_gap_ids),
            evidence_refs=tuple(self.evidence_refs),
        )


def _sort_key(observation: TemporalObservation) -> tuple[object, ...]:
    kind_order = {
        ObservationKind.COVERAGE_GAP: 0,
        ObservationKind.ROUTINE_GAP: 1,
        ObservationKind.EVENT: 2,
    }
    return (
        observation.start_ms,
        observation.end_ms,
        observation.source_batch_id,
        observation.ordinal,
        kind_order[observation.kind],
        observation.observation_id,
    )


def _overlap_duration(
    start_ms: int,
    end_ms: int,
    intervals: Sequence[tuple[int, int]],
) -> int:
    clipped = sorted(
        (max(start_ms, left), min(end_ms, right))
        for left, right in intervals
        if right > start_ms and left < end_ms
    )
    if not clipped:
        return 0
    covered = 0
    current_start, current_end = clipped[0]
    for left, right in clipped[1:]:
        if left <= current_end:
            current_end = max(current_end, right)
            continue
        covered += max(0, current_end - current_start)
        current_start, current_end = left, right
    return covered + max(0, current_end - current_start)


def _coverage_intervals_for(
    coverage_gaps: Sequence[TemporalObservation],
    channel_id: int,
    semantic_key: str,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (gap.start_ms, gap.end_ms)
        for gap in coverage_gaps
        if gap.channel_id == channel_id and gap.applies_to_key(semantic_key)
    )


def complete_child_dispositions(
    child_ids: Sequence[str],
    proposed: Iterable[ChildDisposition] = (),
) -> tuple[ChildDisposition, ...]:
    """Return exactly one disposition for every input child, in input order.

    Missing classifications are preserved as ``unclassified_keep``.  Unknown or
    duplicate proposal IDs are rejected so aggregation cannot silently erase or
    double-account for a child.
    """

    if len(child_ids) > MAX_CHILD_DISPOSITIONS:
        raise ValueError(
            f"child_ids must contain at most {MAX_CHILD_DISPOSITIONS} items"
        )
    normalized_children = tuple(_identifier(value, "child_id") for value in child_ids)
    if len(normalized_children) != len(set(normalized_children)):
        raise ValueError("child_ids must be unique")
    allowed = set(normalized_children)
    by_id: dict[str, ChildDisposition] = {}
    for item in proposed:
        if not isinstance(item, ChildDisposition):
            raise ValueError("proposed items must be ChildDisposition values")
        if item.child_id not in allowed:
            raise ValueError(f"disposition references unknown child_id {item.child_id}")
        if item.child_id in by_id:
            raise ValueError(f"duplicate disposition for child_id {item.child_id}")
        by_id[item.child_id] = item
    return tuple(
        by_id.get(
            child_id,
            ChildDisposition(
                child_id=child_id,
                disposition=ChildDispositionKind.UNCLASSIFIED_KEEP,
                reason="no explicit aggregation disposition",
            ),
        )
        for child_id in normalized_children
    )


def segment_observations(
    observations: Iterable[TemporalObservation],
    *,
    max_observed_gap_ms: int = DEFAULT_MAX_OBSERVED_GAP_MS,
) -> TemporalSegmentation:
    """Segment event streams while keeping parallel semantic keys independent.

    A routine marker ends applicable open episodes.  A coverage marker never
    ends one and its interval is subtracted from the observed-gap clock.  Thus a
    long unobserved interval is uncertainty, not evidence of restored routine.
    """

    try:
        normalized_gap = int(max_observed_gap_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_observed_gap_ms must be an integer") from exc
    if normalized_gap <= 0:
        raise ValueError("max_observed_gap_ms must be positive")

    ordered = list(observations)
    if len(ordered) > MAX_SEGMENTATION_OBSERVATIONS:
        raise ValueError(
            "observations must contain at most "
            f"{MAX_SEGMENTATION_OBSERVATIONS} items"
        )
    if any(not isinstance(item, TemporalObservation) for item in ordered):
        raise ValueError("observations must contain TemporalObservation values")
    ids = [item.observation_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("observation_id values must be unique")
    ordered.sort(key=_sort_key)
    ordered_ids = [item.observation_id for item in ordered]

    active: dict[tuple[int, str], _EpisodeBuilder] = {}
    coverage_gaps: list[TemporalObservation] = []
    episodes: list[TemporalEpisode] = []
    proposed: list[ChildDisposition] = []

    for observation in ordered:
        if observation.kind is ObservationKind.COVERAGE_GAP:
            coverage_gaps.append(observation)
            for builder in active.values():
                if (
                    builder.channel_id == observation.channel_id
                    and observation.applies_to_key(builder.semantic_key)
                ):
                    builder.add_coverage_gap(observation.observation_id)
            proposed.append(
                ChildDisposition(
                    child_id=observation.observation_id,
                    disposition=ChildDispositionKind.COVERAGE_GAP_KEEP,
                    reason="coverage absence is not evidence of routine",
                )
            )
            continue

        if observation.kind is ObservationKind.ROUTINE_GAP:
            for key, builder in tuple(active.items()):
                if builder.channel_id != observation.channel_id:
                    continue
                if not observation.applies_to_key(builder.semantic_key):
                    continue
                episodes.append(
                    builder.freeze(
                        EpisodeStatus.ENDED_BY_ROUTINE,
                        boundary_at_ms=max(
                            builder.last_observed_ms, observation.start_ms
                        ),
                        boundary_observation_id=observation.observation_id,
                    )
                )
                active.pop(key)
            proposed.append(
                ChildDisposition(
                    child_id=observation.observation_id,
                    disposition=ChildDispositionKind.ROUTINE_GAP,
                    reason="observed routine is an episode boundary, not risk resolution",
                )
            )
            continue

        key = (observation.channel_id, observation.semantic_key)
        builder = active.get(key)
        started = builder is None
        if builder is not None:
            raw_gap = max(0, observation.start_ms - builder.last_observed_ms)
            covered_ms = _overlap_duration(
                builder.last_observed_ms,
                observation.start_ms,
                _coverage_intervals_for(
                    coverage_gaps, observation.channel_id, observation.semantic_key
                ),
            )
            observed_gap_ms = max(0, raw_gap - covered_ms)
            if observed_gap_ms > normalized_gap:
                episodes.append(
                    builder.freeze(
                        EpisodeStatus.ENDED_BY_OBSERVED_GAP,
                        boundary_at_ms=observation.start_ms,
                    )
                )
                builder = None
                active.pop(key)
                started = True
        if builder is None:
            builder = _EpisodeBuilder.from_observation(observation)
            active[key] = builder
        builder.append(observation)

        if observation.state is ObservationState.RESOLVED:
            episodes.append(
                builder.freeze(
                    EpisodeStatus.RESOLVED,
                    boundary_at_ms=observation.end_ms,
                    boundary_observation_id=observation.observation_id,
                )
            )
            active.pop(key)
            disposition = ChildDispositionKind.RESOLVED_EPISODE
            reason = "explicit resolved observation ended the episode"
        elif observation.state is ObservationState.UNCERTAIN:
            disposition = ChildDispositionKind.UNCLASSIFIED_KEEP
            reason = "uncertain event observation retained without forced classification"
        elif started:
            disposition = ChildDispositionKind.STARTED_EPISODE
            reason = "first observation after a server-owned boundary"
        else:
            disposition = ChildDispositionKind.CONTINUED_EPISODE
            reason = "compatible observation continued the open episode"
        proposed.append(
            ChildDisposition(
                child_id=observation.observation_id,
                disposition=disposition,
                episode_id=builder.episode_id,
                reason=reason,
            )
        )

    episodes.extend(builder.freeze(EpisodeStatus.OPEN) for builder in active.values())
    episodes.sort(key=lambda item: (item.start_ms, item.channel_id, item.episode_id))
    disposition_by_id = {item.child_id: item for item in proposed}
    dispositions = complete_child_dispositions(
        ordered_ids,
        (disposition_by_id[child_id] for child_id in ordered_ids),
    )
    return TemporalSegmentation(tuple(episodes), dispositions)


def build_series(
    episodes: Iterable[TemporalEpisode],
    *,
    series_key: str | None = None,
) -> TemporalSeries:
    """Explicitly link distinct episodes without merging their identities."""

    values = list(episodes)
    if not values:
        raise ValueError("episodes must not be empty")
    if len(values) > MAX_EPISODES_PER_SERIES:
        raise ValueError(
            f"episodes must contain at most {MAX_EPISODES_PER_SERIES} items"
        )
    if any(not isinstance(item, TemporalEpisode) for item in values):
        raise ValueError("episodes must contain TemporalEpisode values")
    by_id: dict[str, TemporalEpisode] = {}
    for episode in values:
        previous = by_id.get(episode.episode_id)
        if previous is not None and previous != episode:
            raise ValueError(f"conflicting episode payload for {episode.episode_id}")
        by_id[episode.episode_id] = episode
    values = sorted(
        by_id.values(), key=lambda item: (item.start_ms, item.episode_id)
    )
    channels = {item.channel_id for item in values}
    if len(channels) != 1:
        raise ValueError("series episodes must belong to one channel")
    semantic_keys = {item.semantic_key for item in values}
    if series_key is None:
        if len(semantic_keys) != 1:
            raise ValueError(
                "series_key is required when episodes have different semantic keys"
            )
        normalized_series_key = next(iter(semantic_keys))
    else:
        normalized_series_key = _semantic_key(series_key, "series_key")
    channel_id = next(iter(channels))
    first_episode = values[0]
    return TemporalSeries(
        series_id=_stable_id(
            "series",
            "v1",
            channel_id,
            first_episode.episode_id,
        ),
        channel_id=channel_id,
        series_key=normalized_series_key,
        episode_ids=tuple(item.episode_id for item in values),
        first_observed_ms=min(item.start_ms for item in values),
        last_observed_ms=max(item.last_observed_ms for item in values),
    )


def select_attention_observations(
    observations: Iterable[TemporalObservation],
    budget: AttentionBudget | int,
) -> tuple[TemporalObservation, ...]:
    """Select 2, 4, or 8 evenly distributed temporal observations.

    First and last observations are always retained.  This function only defines
    deterministic selection semantics; it neither changes runtime cadence nor
    decides which attention budget a caller should use.
    """

    try:
        normalized_budget = (
            budget if isinstance(budget, AttentionBudget) else AttentionBudget(int(budget))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("attention budget must be exactly 2, 4, or 8") from exc
    ordered = list(observations)
    if any(not isinstance(item, TemporalObservation) for item in ordered):
        raise ValueError("observations must contain TemporalObservation values")
    ids = [item.observation_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("observation_id values must be unique")
    ordered.sort(key=_sort_key)
    count = int(normalized_budget)
    if len(ordered) <= count:
        return tuple(ordered)
    last_index = len(ordered) - 1
    indices = tuple((index * last_index) // (count - 1) for index in range(count))
    return tuple(ordered[index] for index in indices)
