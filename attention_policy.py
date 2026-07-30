"""Deterministic attention planning primitives for EVA live video.

The module deliberately has no dependency on capture, database, model, or image
libraries.  Dense CV samples are reduced to compact intervals and linked to
already-persisted embedding snapshots.  The resulting records can be printed,
stored, replayed, and used by a scheduler without retaining image payloads.

All timestamps are integer milliseconds on the source timeline.  Callers are
responsible for mapping source clocks to a common monotonic timeline.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping, Sequence


class MotionKind(str, Enum):
    QUIET = "quiet"
    MOTION = "motion"


class AttentionMode(str, Enum):
    QUIET = "quiet"
    WATCH = "watch"
    ACTIVE = "active"
    BURST = "burst"
    DEGRADED = "degraded"


class EpisodeRole(str, Enum):
    CONTROL = "control"
    PRE = "pre"
    ONSET = "onset"
    APEX = "apex"
    POST = "post"
    CURRENT = "current"


@dataclass(frozen=True)
class ModeProfile:
    """Port scheduling contract for one homeostatic mode.

    ``cadence_ms`` is the earliest VLM admission cadence.  It does not control
    the independent 1 Hz embedding/archive path.  ``deadline_ms`` is the hard
    coverage deadline; reaching ``hard_accumulator_cap`` also forces admission.
    """

    mode: AttentionMode
    cadence_ms: int
    deadline_ms: int
    min_frames: int
    target_frames: int
    max_frames: int
    embedding_cadence_ms: int = 1_000
    dispatch_enabled: bool = True
    hard_accumulator_cap: int = 16

    def __post_init__(self) -> None:
        if self.cadence_ms <= 0:
            raise ValueError("cadence_ms must be positive")
        if self.deadline_ms < self.cadence_ms:
            raise ValueError("deadline_ms must be at least cadence_ms")
        if not 1 <= self.min_frames <= self.target_frames <= self.max_frames:
            raise ValueError("frame targets must satisfy 1 <= min <= target <= max")
        if not 4 <= self.min_frames <= 16:
            raise ValueError("port min_frames must be between 4 and 16")
        if not 4 <= self.max_frames <= 16:
            raise ValueError("port max_frames must be between 4 and 16")
        if self.embedding_cadence_ms <= 0:
            raise ValueError("embedding_cadence_ms must be positive")
        if self.hard_accumulator_cap != 16:
            raise ValueError("port hard_accumulator_cap must be exactly 16")
        if self.hard_accumulator_cap < self.max_frames:
            raise ValueError("hard_accumulator_cap cannot be lower than max_frames")

    def due_at(self, last_dispatch_ms: int) -> int:
        return _timestamp(last_dispatch_ms, "last_dispatch_ms") + self.cadence_ms

    def deadline_at(self, last_dispatch_ms: int) -> int:
        return _timestamp(last_dispatch_ms, "last_dispatch_ms") + self.deadline_ms

    def should_force_dispatch(
        self,
        *,
        accumulator_size: int,
        now_ms: int,
        last_dispatch_ms: int,
    ) -> bool:
        if not self.dispatch_enabled:
            return False
        return (
            int(accumulator_size) >= self.hard_accumulator_cap
            or _timestamp(now_ms, "now_ms") >= self.deadline_at(last_dispatch_ms)
        )


@dataclass(frozen=True)
class PortAttentionPreset:
    """Deterministic deployment preset for one 4070-class VLM over 8 channels."""

    name: str
    channel_limit: int
    steady_l0_per_minute: float
    profiles: tuple[ModeProfile, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("preset name must not be empty")
        if self.channel_limit <= 0:
            raise ValueError("channel_limit must be positive")
        steady = float(self.steady_l0_per_minute)
        if not math.isfinite(steady) or steady <= 0:
            raise ValueError("steady_l0_per_minute must be positive")
        modes = [profile.mode for profile in self.profiles]
        if len(modes) != len(set(modes)):
            raise ValueError("preset profiles must have unique modes")
        missing = set(AttentionMode).difference(modes)
        if missing:
            raise ValueError(
                "preset is missing modes: "
                + ", ".join(sorted(mode.value for mode in missing))
            )
        if any(profile.embedding_cadence_ms != 1_000 for profile in self.profiles):
            raise ValueError("port embeddings must remain fixed at 1000 ms in every mode")
        if any(profile.hard_accumulator_cap != 16 for profile in self.profiles):
            raise ValueError("port accumulator hard cap must remain 16 in every mode")

    def profile_for_mode(self, mode: AttentionMode | str) -> ModeProfile:
        normalized = (
            mode
            if isinstance(mode, AttentionMode)
            else AttentionMode(str(mode).strip().lower())
        )
        return next(profile for profile in self.profiles if profile.mode is normalized)


PORT_EIGHT_CHANNEL_PRESET = PortAttentionPreset(
    name="port-4070s-8ch",
    channel_limit=8,
    steady_l0_per_minute=6.0,
    profiles=(
        ModeProfile(AttentionMode.QUIET, 10_000, 120_000, 6, 8, 8),
        ModeProfile(AttentionMode.WATCH, 5_000, 90_000, 6, 8, 10),
        ModeProfile(AttentionMode.ACTIVE, 2_500, 60_000, 8, 12, 12),
        ModeProfile(AttentionMode.BURST, 1_000, 30_000, 10, 16, 16),
        ModeProfile(
            AttentionMode.DEGRADED,
            15_000,
            120_000,
            4,
            6,
            6,
            dispatch_enabled=True,
        ),
    ),
)


def profile_for_mode(
    mode: AttentionMode | str,
    preset: PortAttentionPreset = PORT_EIGHT_CHANNEL_PRESET,
) -> ModeProfile:
    """Return the stable port cadence/deadline/frame contract for ``mode``."""

    return preset.profile_for_mode(mode)


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _unit(value: float, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _non_negative(value: float, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _timestamp(value: int, name: str = "timestamp_ms") -> int:
    timestamp = int(value)
    if timestamp < 0:
        raise ValueError(f"{name} must be non-negative")
    return timestamp


@dataclass(frozen=True)
class CvSample:
    """One cheap CV observation; it never contains an image."""

    timestamp_ms: int
    motion: float
    sharpness: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", _timestamp(self.timestamp_ms))
        object.__setattr__(self, "motion", _unit(self.motion, "motion"))
        object.__setattr__(
            self, "sharpness", _non_negative(self.sharpness, "sharpness")
        )


@dataclass(frozen=True)
class ProbeScore:
    """P/N/M values computed against a saved image embedding."""

    probe_id: str
    positive: float
    negative: float
    margin: float
    probe_version: str = ""

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must not be empty")
        object.__setattr__(self, "positive", _finite(self.positive, "positive"))
        object.__setattr__(self, "negative", _finite(self.negative, "negative"))
        object.__setattr__(self, "margin", _finite(self.margin, "margin"))


@dataclass(frozen=True)
class EmbeddingSnapshot:
    """Reference to one persisted embedding snapshot.

    Only durable identifiers and scalar probe results belong here.  Pixels,
    JPEG/base64 payloads, and arbitrary metadata are intentionally absent.
    """

    channel_id: str
    snapshot_id: str
    timestamp_ms: int
    embedding_ref: str
    frame_hash: str = ""
    probe_scores: tuple[ProbeScore, ...] = ()

    def __post_init__(self) -> None:
        if not self.channel_id:
            raise ValueError("channel_id must not be empty")
        if not self.snapshot_id:
            raise ValueError("snapshot_id must not be empty")
        if not self.embedding_ref:
            raise ValueError("embedding_ref must not be empty")
        object.__setattr__(self, "timestamp_ms", _timestamp(self.timestamp_ms))
        object.__setattr__(self, "probe_scores", tuple(self.probe_scores))


@dataclass(frozen=True)
class ApexMarker:
    """A CV/VLM apex timestamp which will be linked to a saved snapshot."""

    apex_id: str
    timestamp_ms: int

    def __post_init__(self) -> None:
        if not self.apex_id:
            raise ValueError("apex_id must not be empty")
        object.__setattr__(self, "timestamp_ms", _timestamp(self.timestamp_ms))


@dataclass(frozen=True)
class ApexLink:
    apex_id: str
    apex_timestamp_ms: int
    snapshot_id: str
    snapshot_timestamp_ms: int
    distance_ms: int


@dataclass(frozen=True)
class MotionInterval:
    """Compact representation of a run of motion or quiet CV samples."""

    channel_id: str
    kind: MotionKind
    start_ms: int
    end_ms: int
    sample_count: int
    mean_motion: float
    peak_motion: float
    peak_timestamp_ms: int
    mean_sharpness: float
    linked_snapshot_ids: tuple[str, ...] = ()
    apex_links: tuple[ApexLink, ...] = ()

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def printable(self) -> str:
        return json.dumps(
            {
                "apex_links": [
                    {
                        "apex_id": link.apex_id,
                        "apex_timestamp_ms": link.apex_timestamp_ms,
                        "distance_ms": link.distance_ms,
                        "snapshot_id": link.snapshot_id,
                        "snapshot_timestamp_ms": link.snapshot_timestamp_ms,
                    }
                    for link in self.apex_links
                ],
                "channel_id": self.channel_id,
                "end_ms": self.end_ms,
                "kind": self.kind.value,
                "linked_snapshot_ids": list(self.linked_snapshot_ids),
                "mean_motion": round(self.mean_motion, 6),
                "mean_sharpness": round(self.mean_sharpness, 6),
                "peak_motion": round(self.peak_motion, 6),
                "peak_timestamp_ms": self.peak_timestamp_ms,
                "sample_count": self.sample_count,
                "start_ms": self.start_ms,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class AggregationConfig:
    motion_enter: float = 0.35
    motion_exit: float = 0.20
    max_sample_gap_ms: int = 1_500
    snapshot_link_tolerance_ms: int = 600
    apex_link_tolerance_ms: int = 1_500

    def __post_init__(self) -> None:
        enter = _unit(self.motion_enter, "motion_enter")
        exit_ = _unit(self.motion_exit, "motion_exit")
        if exit_ >= enter:
            raise ValueError("motion_exit must be lower than motion_enter")
        if self.max_sample_gap_ms <= 0:
            raise ValueError("max_sample_gap_ms must be positive")
        if self.snapshot_link_tolerance_ms < 0:
            raise ValueError("snapshot_link_tolerance_ms must be non-negative")
        if self.apex_link_tolerance_ms < 0:
            raise ValueError("apex_link_tolerance_ms must be non-negative")


@dataclass
class _IntervalBuilder:
    channel_id: str
    kind: MotionKind
    samples: list[CvSample]

    def finish(self) -> MotionInterval:
        peak = max(self.samples, key=lambda sample: (sample.motion, -sample.timestamp_ms))
        return MotionInterval(
            channel_id=self.channel_id,
            kind=self.kind,
            start_ms=self.samples[0].timestamp_ms,
            end_ms=self.samples[-1].timestamp_ms,
            sample_count=len(self.samples),
            mean_motion=sum(sample.motion for sample in self.samples) / len(self.samples),
            peak_motion=peak.motion,
            peak_timestamp_ms=peak.timestamp_ms,
            mean_sharpness=(
                sum(sample.sharpness for sample in self.samples) / len(self.samples)
            ),
        )


def aggregate_cv_intervals(
    channel_id: str,
    samples: Iterable[CvSample],
    snapshots: Iterable[EmbeddingSnapshot] = (),
    apex_markers: Iterable[ApexMarker] = (),
    config: AggregationConfig | None = None,
) -> tuple[MotionInterval, ...]:
    """Aggregate dense CV samples and link only durable embedding snapshots.

    Hysteresis is applied between ``motion_enter`` and ``motion_exit``.  A large
    timestamp gap terminates the current interval; it is not silently described
    as quiet.  Snapshots in small sampling gaps may be linked to the nearest
    interval boundary within ``snapshot_link_tolerance_ms``.
    """

    if not channel_id:
        raise ValueError("channel_id must not be empty")
    policy = config or AggregationConfig()
    ordered_samples = sorted(samples, key=lambda sample: sample.timestamp_ms)
    if not ordered_samples:
        return ()
    if len({sample.timestamp_ms for sample in ordered_samples}) != len(ordered_samples):
        raise ValueError("CV sample timestamps must be unique per aggregation window")

    ordered_snapshots = sorted(snapshots, key=lambda snapshot: snapshot.timestamp_ms)
    for snapshot in ordered_snapshots:
        if snapshot.channel_id != channel_id:
            raise ValueError("all snapshots must belong to channel_id")
    if len({snapshot.snapshot_id for snapshot in ordered_snapshots}) != len(
        ordered_snapshots
    ):
        raise ValueError("snapshot_id values must be unique")

    markers = sorted(apex_markers, key=lambda marker: (marker.timestamp_ms, marker.apex_id))
    builders: list[MotionInterval] = []
    first = ordered_samples[0]
    current_kind = (
        MotionKind.MOTION
        if first.motion >= policy.motion_enter
        else MotionKind.QUIET
    )
    current = _IntervalBuilder(channel_id, current_kind, [first])

    for sample in ordered_samples[1:]:
        previous_sample = current.samples[-1]
        gap = sample.timestamp_ms - previous_sample.timestamp_ms
        if gap > policy.max_sample_gap_ms:
            builders.append(current.finish())
            current_kind = (
                MotionKind.MOTION
                if sample.motion >= policy.motion_enter
                else MotionKind.QUIET
            )
            current = _IntervalBuilder(channel_id, current_kind, [sample])
            continue

        next_kind = current.kind
        if current.kind is MotionKind.QUIET and sample.motion >= policy.motion_enter:
            next_kind = MotionKind.MOTION
        elif current.kind is MotionKind.MOTION and sample.motion <= policy.motion_exit:
            next_kind = MotionKind.QUIET

        if next_kind is not current.kind:
            builders.append(current.finish())
            current = _IntervalBuilder(channel_id, next_kind, [sample])
        else:
            current.samples.append(sample)
    builders.append(current.finish())

    intervals: list[MotionInterval] = []
    for interval in builders:
        linked: list[EmbeddingSnapshot] = []
        for snapshot in ordered_snapshots:
            if interval.start_ms <= snapshot.timestamp_ms <= interval.end_ms:
                linked.append(snapshot)
                continue
            distance = min(
                abs(snapshot.timestamp_ms - interval.start_ms),
                abs(snapshot.timestamp_ms - interval.end_ms),
            )
            if distance <= policy.snapshot_link_tolerance_ms:
                # A snapshot near two interval boundaries belongs to the nearest
                # interval; ties resolve to the earlier interval deterministically.
                nearest_index = min(
                    range(len(builders)),
                    key=lambda index: (
                        min(
                            abs(snapshot.timestamp_ms - builders[index].start_ms),
                            abs(snapshot.timestamp_ms - builders[index].end_ms),
                        ),
                        index,
                    ),
                )
                if builders[nearest_index] is interval:
                    linked.append(snapshot)

        apex_links: list[ApexLink] = []
        if interval.kind is MotionKind.MOTION and linked:
            for marker in markers:
                if not interval.start_ms <= marker.timestamp_ms <= interval.end_ms:
                    continue
                nearest = min(
                    linked,
                    key=lambda snapshot: (
                        abs(snapshot.timestamp_ms - marker.timestamp_ms),
                        snapshot.timestamp_ms,
                        snapshot.snapshot_id,
                    ),
                )
                distance = abs(nearest.timestamp_ms - marker.timestamp_ms)
                if distance <= policy.apex_link_tolerance_ms:
                    apex_links.append(
                        ApexLink(
                            apex_id=marker.apex_id,
                            apex_timestamp_ms=marker.timestamp_ms,
                            snapshot_id=nearest.snapshot_id,
                            snapshot_timestamp_ms=nearest.timestamp_ms,
                            distance_ms=distance,
                        )
                    )

        intervals.append(
            replace(
                interval,
                linked_snapshot_ids=tuple(
                    snapshot.snapshot_id
                    for snapshot in sorted(
                        linked, key=lambda item: (item.timestamp_ms, item.snapshot_id)
                    )
                ),
                apex_links=tuple(apex_links),
            )
        )
    return tuple(intervals)


@dataclass(frozen=True)
class AttentionVector:
    """Normalized, independently inspectable inputs to attention control."""

    timestamp_ms: int
    motion_intensity: float = 0.0
    motion_persistence: float = 0.0
    burst: float = 0.0
    probe_positive: float = 0.0
    probe_margin: float = 0.0
    probe_novelty: float = 0.0
    uncertainty: float = 0.0
    alert_persistence: float = 0.0
    signal_staleness: float = 0.0
    source_health: float = 1.0
    redundancy: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", _timestamp(self.timestamp_ms))
        for name in (
            "motion_intensity",
            "motion_persistence",
            "burst",
            "probe_positive",
            "probe_margin",
            "probe_novelty",
            "uncertainty",
            "alert_persistence",
            "signal_staleness",
            "source_health",
            "redundancy",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))

    def as_dict(self) -> dict[str, float | int]:
        return {
            "alert_persistence": self.alert_persistence,
            "burst": self.burst,
            "motion_intensity": self.motion_intensity,
            "motion_persistence": self.motion_persistence,
            "probe_margin": self.probe_margin,
            "probe_novelty": self.probe_novelty,
            "probe_positive": self.probe_positive,
            "redundancy": self.redundancy,
            "signal_staleness": self.signal_staleness,
            "source_health": self.source_health,
            "timestamp_ms": self.timestamp_ms,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class AttentionWeights:
    motion_intensity: float = 0.20
    motion_persistence: float = 0.08
    burst: float = 0.18
    probe_positive: float = 0.10
    probe_margin: float = 0.08
    probe_novelty: float = 0.08
    uncertainty: float = 0.06
    alert_persistence: float = 0.10
    signal_staleness: float = 0.05
    coverage_debt: float = 0.20
    health_penalty: float = 0.30
    redundancy_penalty: float = 0.12


@dataclass(frozen=True)
class AttentionPolicyConfig:
    weights: AttentionWeights = field(default_factory=AttentionWeights)
    watch_enter: float = 0.28
    watch_exit: float = 0.16
    active_enter: float = 0.58
    active_exit: float = 0.38
    burst_enter: float = 0.72
    burst_exit: float = 0.38
    degraded_enter_health: float = 0.35
    degraded_exit_health: float = 0.70
    quiet_target_interval_ms: int = 240_000
    watch_target_interval_ms: int = 60_000
    active_target_interval_ms: int = 15_000
    burst_target_interval_ms: int = 5_000
    degraded_target_interval_ms: int = 300_000
    min_mode_dwell_ms: int = 5_000
    burst_min_dwell_ms: int = 2_000
    burst_cooldown_ms: int = 15_000
    degraded_min_dwell_ms: int = 10_000
    max_coverage_debt: float = 2.0
    bootstrap_coverage_debt: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "watch_enter",
            "watch_exit",
            "active_enter",
            "active_exit",
            "burst_enter",
            "burst_exit",
            "degraded_enter_health",
            "degraded_exit_health",
        ):
            _unit(getattr(self, name), name)
        if self.watch_exit >= self.watch_enter:
            raise ValueError("watch_exit must be lower than watch_enter")
        if self.active_exit >= self.active_enter:
            raise ValueError("active_exit must be lower than active_enter")
        if self.burst_exit >= self.burst_enter:
            raise ValueError("burst_exit must be lower than burst_enter")
        if self.degraded_enter_health >= self.degraded_exit_health:
            raise ValueError(
                "degraded_enter_health must be lower than degraded_exit_health"
            )
        for name in (
            "quiet_target_interval_ms",
            "watch_target_interval_ms",
            "active_target_interval_ms",
            "burst_target_interval_ms",
            "degraded_target_interval_ms",
            "min_mode_dwell_ms",
            "burst_min_dwell_ms",
            "burst_cooldown_ms",
            "degraded_min_dwell_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        _non_negative(self.max_coverage_debt, "max_coverage_debt")
        _non_negative(self.bootstrap_coverage_debt, "bootstrap_coverage_debt")


@dataclass(frozen=True)
class AttentionState:
    channel_id: str
    mode: AttentionMode
    mode_since_ms: int
    cooldown_until_ms: int
    last_vlm_ms: int | None
    vector: AttentionVector
    priority: float
    coverage_debt: float


@dataclass(frozen=True)
class AttentionDecision:
    state: AttentionState
    components: tuple[tuple[str, float], ...]
    reasons: tuple[str, ...]

    @property
    def channel_id(self) -> str:
        return self.state.channel_id

    @property
    def mode(self) -> AttentionMode:
        return self.state.mode

    @property
    def priority(self) -> float:
        return self.state.priority

    @property
    def coverage_debt(self) -> float:
        return self.state.coverage_debt

    def printable(self) -> str:
        return json.dumps(
            {
                "channel_id": self.channel_id,
                "components": {
                    name: round(value, 6) for name, value in self.components
                },
                "cooldown_until_ms": self.state.cooldown_until_ms,
                "coverage_debt": round(self.coverage_debt, 6),
                "last_vlm_ms": self.state.last_vlm_ms,
                "mode": self.mode.value,
                "mode_since_ms": self.state.mode_since_ms,
                "priority": round(self.priority, 6),
                "reasons": list(self.reasons),
                "vector": {
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in self.state.vector.as_dict().items()
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class HomeostaticAttentionPolicy:
    """State transition and priority policy with no external side effects."""

    def __init__(self, config: AttentionPolicyConfig | None = None) -> None:
        self.config = config or AttentionPolicyConfig()

    def evaluate(
        self,
        channel_id: str,
        vector: AttentionVector,
        previous: AttentionState | None = None,
        *,
        last_vlm_ms: int | None = None,
    ) -> AttentionDecision:
        if not channel_id:
            raise ValueError("channel_id must not be empty")
        now = vector.timestamp_ms
        if previous is not None and previous.channel_id != channel_id:
            raise ValueError("previous state belongs to another channel")
        if previous is not None and previous.vector.timestamp_ms > now:
            raise ValueError("attention timestamps must be monotonic")

        effective_last_vlm = (
            last_vlm_ms
            if last_vlm_ms is not None
            else previous.last_vlm_ms if previous is not None else None
        )
        if effective_last_vlm is not None:
            effective_last_vlm = _timestamp(effective_last_vlm, "last_vlm_ms")
            if effective_last_vlm > now:
                raise ValueError("last_vlm_ms cannot be in the future")

        event_drive = max(
            vector.burst,
            0.54 * vector.motion_intensity
            + 0.18 * vector.motion_persistence
            + 0.12 * vector.probe_positive
            + 0.08 * vector.probe_novelty
            + 0.08 * vector.alert_persistence,
        )
        mode, mode_reason, cooldown_until = self._next_mode(
            vector, event_drive, previous
        )
        mode_since = (
            previous.mode_since_ms
            if previous is not None and previous.mode is mode
            else now
        )
        target_interval = self._target_interval(mode)
        if effective_last_vlm is None:
            coverage_debt = min(
                self.config.max_coverage_debt,
                self.config.bootstrap_coverage_debt,
            )
            coverage_reason = "coverage=bootstrap"
        elif target_interval == 0:
            coverage_debt = self.config.max_coverage_debt
            coverage_reason = "coverage=immediate"
        else:
            coverage_debt = min(
                self.config.max_coverage_debt,
                max(0.0, (now - effective_last_vlm) / target_interval),
            )
            coverage_reason = (
                f"coverage={now - effective_last_vlm}ms/{target_interval}ms"
            )

        weights = self.config.weights
        components = (
            ("alert_persistence", weights.alert_persistence * vector.alert_persistence),
            ("burst", weights.burst * vector.burst),
            ("coverage_debt", weights.coverage_debt * coverage_debt),
            (
                "health_penalty",
                -weights.health_penalty * (1.0 - vector.source_health),
            ),
            ("motion_intensity", weights.motion_intensity * vector.motion_intensity),
            (
                "motion_persistence",
                weights.motion_persistence * vector.motion_persistence,
            ),
            ("probe_margin", weights.probe_margin * vector.probe_margin),
            ("probe_novelty", weights.probe_novelty * vector.probe_novelty),
            ("probe_positive", weights.probe_positive * vector.probe_positive),
            ("redundancy_penalty", -weights.redundancy_penalty * vector.redundancy),
            (
                "signal_staleness",
                weights.signal_staleness * vector.signal_staleness,
            ),
            ("uncertainty", weights.uncertainty * vector.uncertainty),
        )
        priority = max(0.0, min(2.0, sum(value for _, value in components)))
        if mode is AttentionMode.DEGRADED:
            priority = 0.0
        state = AttentionState(
            channel_id=channel_id,
            mode=mode,
            mode_since_ms=mode_since,
            cooldown_until_ms=cooldown_until,
            last_vlm_ms=effective_last_vlm,
            vector=vector,
            priority=priority,
            coverage_debt=coverage_debt,
        )
        reasons = (
            mode_reason,
            f"event_drive={event_drive:.3f}",
            coverage_reason,
            f"priority={priority:.3f}",
        )
        return AttentionDecision(state, components, reasons)

    def record_dispatch(
        self, state: AttentionState, timestamp_ms: int
    ) -> AttentionState:
        """Return replayable state after a VLM job was admitted."""

        timestamp = _timestamp(timestamp_ms)
        if timestamp < state.vector.timestamp_ms:
            raise ValueError("dispatch cannot predate the evaluated state")
        return replace(state, last_vlm_ms=timestamp, coverage_debt=0.0)

    def _next_mode(
        self,
        vector: AttentionVector,
        event_drive: float,
        previous: AttentionState | None,
    ) -> tuple[AttentionMode, str, int]:
        cfg = self.config
        now = vector.timestamp_ms
        cooldown = previous.cooldown_until_ms if previous is not None else 0
        age = now - previous.mode_since_ms if previous is not None else 0

        if previous is not None and previous.mode is AttentionMode.DEGRADED:
            if (
                vector.source_health < cfg.degraded_exit_health
                or age < cfg.degraded_min_dwell_ms
            ):
                return AttentionMode.DEGRADED, "mode=degraded:hold_hysteresis", cooldown
        elif vector.source_health <= cfg.degraded_enter_health:
            return AttentionMode.DEGRADED, "mode=degraded:source_health", cooldown

        if vector.burst >= cfg.burst_enter:
            if previous is None or previous.mode is AttentionMode.BURST or now >= cooldown:
                new_cooldown = max(cooldown, now + cfg.burst_cooldown_ms)
                reason = (
                    "mode=burst:hold"
                    if previous is not None and previous.mode is AttentionMode.BURST
                    else "mode=burst:enter"
                )
                return AttentionMode.BURST, reason, new_cooldown

        if previous is not None and previous.mode is AttentionMode.BURST:
            if vector.burst > cfg.burst_exit or age < cfg.burst_min_dwell_ms:
                return AttentionMode.BURST, "mode=burst:hold_hysteresis", cooldown

        desired = (
            AttentionMode.ACTIVE
            if event_drive >= cfg.active_enter
            else AttentionMode.WATCH
            if event_drive >= cfg.watch_enter
            else AttentionMode.QUIET
        )
        if previous is None or previous.mode in (
            AttentionMode.BURST,
            AttentionMode.DEGRADED,
        ):
            return desired, f"mode={desired.value}:classified", cooldown

        if previous.mode is AttentionMode.ACTIVE:
            if desired is AttentionMode.BURST:
                return desired, "mode=burst:upgrade", cooldown
            if event_drive > cfg.active_exit or age < cfg.min_mode_dwell_ms:
                return AttentionMode.ACTIVE, "mode=active:hold_hysteresis", cooldown
        elif previous.mode is AttentionMode.WATCH:
            if desired is AttentionMode.ACTIVE:
                return AttentionMode.ACTIVE, "mode=active:upgrade", cooldown
            if event_drive > cfg.watch_exit or age < cfg.min_mode_dwell_ms:
                return AttentionMode.WATCH, "mode=watch:hold_hysteresis", cooldown
        elif previous.mode is AttentionMode.QUIET and desired is not AttentionMode.QUIET:
            return desired, f"mode={desired.value}:upgrade", cooldown

        return desired, f"mode={desired.value}:classified", cooldown

    def _target_interval(self, mode: AttentionMode) -> int:
        return {
            AttentionMode.QUIET: self.config.quiet_target_interval_ms,
            AttentionMode.WATCH: self.config.watch_target_interval_ms,
            AttentionMode.ACTIVE: self.config.active_target_interval_ms,
            AttentionMode.BURST: self.config.burst_target_interval_ms,
            AttentionMode.DEGRADED: self.config.degraded_target_interval_ms,
        }[mode]


@dataclass(frozen=True)
class AttentionCandidate:
    channel_id: str
    decision: AttentionDecision
    estimated_units: float = 1.0
    episode_id: str = ""
    ready_at_ms: int = 0

    def __post_init__(self) -> None:
        if self.channel_id != self.decision.channel_id:
            raise ValueError("candidate and decision channel_id must match")
        if _non_negative(self.estimated_units, "estimated_units") == 0:
            raise ValueError("estimated_units must be positive")
        object.__setattr__(self, "ready_at_ms", _timestamp(self.ready_at_ms))


@dataclass(frozen=True)
class GlobalBudgetConfig:
    total_units: float
    max_jobs: int = 16
    fairness_fraction: float = 0.25
    urgent_priority: float = 0.62

    def __post_init__(self) -> None:
        if _non_negative(self.total_units, "total_units") == 0:
            raise ValueError("total_units must be positive")
        if self.max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        _unit(self.fairness_fraction, "fairness_fraction")
        _non_negative(self.urgent_priority, "urgent_priority")


@dataclass(frozen=True)
class AllocationEntry:
    channel_id: str
    episode_id: str
    phase: str
    estimated_units: float
    priority: float
    coverage_debt: float


@dataclass(frozen=True)
class AttentionAllocation:
    selected: tuple[AllocationEntry, ...]
    rejected: tuple[tuple[str, str], ...]
    used_units: float
    total_units: float

    def printable(self) -> str:
        return json.dumps(
            {
                "rejected": [
                    {"channel_id": channel_id, "reason": reason}
                    for channel_id, reason in self.rejected
                ],
                "selected": [
                    {
                        "channel_id": item.channel_id,
                        "coverage_debt": round(item.coverage_debt, 6),
                        "episode_id": item.episode_id,
                        "estimated_units": round(item.estimated_units, 6),
                        "phase": item.phase,
                        "priority": round(item.priority, 6),
                    }
                    for item in self.selected
                ],
                "total_units": round(self.total_units, 6),
                "used_units": round(self.used_units, 6),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def allocate_global_attention(
    candidates: Iterable[AttentionCandidate],
    now_ms: int,
    config: GlobalBudgetConfig,
) -> AttentionAllocation:
    """Allocate one global budget with explicit urgent/fairness/priority phases."""

    now = _timestamp(now_ms, "now_ms")
    ordered = sorted(candidates, key=lambda item: item.channel_id)
    if len({item.channel_id for item in ordered}) != len(ordered):
        raise ValueError("only one candidate per channel is allowed in a cycle")

    rejected: dict[str, str] = {}
    eligible: list[AttentionCandidate] = []
    for candidate in ordered:
        if candidate.decision.mode is AttentionMode.DEGRADED:
            rejected[candidate.channel_id] = "degraded_source"
        elif candidate.ready_at_ms > now:
            rejected[candidate.channel_id] = f"not_ready_until:{candidate.ready_at_ms}"
        elif candidate.estimated_units > config.total_units:
            rejected[candidate.channel_id] = "cost_exceeds_global_budget"
        else:
            eligible.append(candidate)

    selected: list[AllocationEntry] = []
    selected_channels: set[str] = set()
    used = 0.0

    def admit(
        pool: Sequence[AttentionCandidate], phase: str, phase_limit: float
    ) -> None:
        nonlocal used
        for candidate in pool:
            if len(selected) >= config.max_jobs:
                return
            if candidate.channel_id in selected_channels:
                continue
            if used + candidate.estimated_units > config.total_units + 1e-9:
                continue
            phase_used = sum(
                item.estimated_units for item in selected if item.phase == phase
            )
            if phase_used + candidate.estimated_units > phase_limit + 1e-9:
                continue
            selected.append(
                AllocationEntry(
                    channel_id=candidate.channel_id,
                    episode_id=candidate.episode_id,
                    phase=phase,
                    estimated_units=candidate.estimated_units,
                    priority=candidate.decision.priority,
                    coverage_debt=candidate.decision.coverage_debt,
                )
            )
            selected_channels.add(candidate.channel_id)
            used += candidate.estimated_units

    fairness_units = config.total_units * config.fairness_fraction
    event_units = config.total_units - fairness_units
    urgent = sorted(
        (
            item
            for item in eligible
            if item.decision.mode is AttentionMode.BURST
            or item.decision.priority >= config.urgent_priority
        ),
        key=lambda item: (
            -item.decision.priority,
            -item.decision.coverage_debt,
            item.channel_id,
        ),
    )
    admit(urgent, "urgent", event_units)

    overdue = sorted(
        (item for item in eligible if item.decision.coverage_debt >= 1.0),
        key=lambda item: (
            -item.decision.coverage_debt,
            item.decision.state.last_vlm_ms
            if item.decision.state.last_vlm_ms is not None
            else -1,
            item.channel_id,
        ),
    )
    admit(overdue, "fairness", fairness_units)

    remainder = sorted(
        eligible,
        key=lambda item: (
            -item.decision.priority,
            -item.decision.coverage_debt,
            item.channel_id,
        ),
    )
    admit(remainder, "priority", config.total_units)

    for candidate in eligible:
        if candidate.channel_id not in selected_channels:
            rejected[candidate.channel_id] = (
                "max_jobs_reached"
                if len(selected) >= config.max_jobs
                else "global_budget_exhausted"
            )
    return AttentionAllocation(
        selected=tuple(selected),
        rejected=tuple(sorted(rejected.items())),
        used_units=used,
        total_units=config.total_units,
    )


@dataclass(frozen=True)
class InferenceCost:
    """Preflight cost for one L0 request in both scarce resource dimensions."""

    prompt_tokens: int
    vision_tokens: int
    expected_output_tokens: int
    slot_seconds: float

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "vision_tokens", "expected_output_tokens"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.total_tokens <= 0:
            raise ValueError("inference must consume at least one token")
        if _non_negative(self.slot_seconds, "slot_seconds") == 0:
            raise ValueError("slot_seconds must be positive")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.vision_tokens + self.expected_output_tokens


@dataclass(frozen=True)
class CostBudgetConfig:
    """Token bucket sized to six steady reference L0 requests per minute."""

    steady_l0_per_minute: float = 6.0
    reference_l0_tokens: int = 8_192
    reference_l0_slot_seconds: float = 10.0
    burst_borrow_l0: float = 2.0
    fairness_fraction: float = 0.25
    urgent_priority: float = 0.62
    max_jobs_per_cycle: int = 8
    refill_window_ms: int = 60_000

    def __post_init__(self) -> None:
        if _non_negative(
            self.steady_l0_per_minute, "steady_l0_per_minute"
        ) == 0:
            raise ValueError("steady_l0_per_minute must be positive")
        if self.reference_l0_tokens <= 0:
            raise ValueError("reference_l0_tokens must be positive")
        if _non_negative(
            self.reference_l0_slot_seconds, "reference_l0_slot_seconds"
        ) == 0:
            raise ValueError("reference_l0_slot_seconds must be positive")
        _non_negative(self.burst_borrow_l0, "burst_borrow_l0")
        _unit(self.fairness_fraction, "fairness_fraction")
        _non_negative(self.urgent_priority, "urgent_priority")
        if self.max_jobs_per_cycle <= 0:
            raise ValueError("max_jobs_per_cycle must be positive")
        if self.refill_window_ms <= 0:
            raise ValueError("refill_window_ms must be positive")

    @property
    def token_capacity(self) -> float:
        return self.reference_l0_tokens * self.steady_l0_per_minute

    @property
    def slot_seconds_capacity(self) -> float:
        return self.reference_l0_slot_seconds * self.steady_l0_per_minute

    @property
    def token_borrow_limit(self) -> float:
        return self.reference_l0_tokens * self.burst_borrow_l0

    @property
    def slot_seconds_borrow_limit(self) -> float:
        return self.reference_l0_slot_seconds * self.burst_borrow_l0

    def equivalent_l0(self, cost: InferenceCost) -> float:
        return max(
            cost.total_tokens / self.reference_l0_tokens,
            cost.slot_seconds / self.reference_l0_slot_seconds,
        )


@dataclass(frozen=True)
class CostBudgetState:
    timestamp_ms: int
    available_tokens: float
    available_slot_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", _timestamp(self.timestamp_ms))
        object.__setattr__(
            self,
            "available_tokens",
            _finite(self.available_tokens, "available_tokens"),
        )
        object.__setattr__(
            self,
            "available_slot_seconds",
            _finite(self.available_slot_seconds, "available_slot_seconds"),
        )

    def debt_l0(self, config: CostBudgetConfig) -> float:
        return max(
            max(0.0, -self.available_tokens) / config.reference_l0_tokens,
            max(0.0, -self.available_slot_seconds)
            / config.reference_l0_slot_seconds,
        )


def initial_cost_budget_state(
    timestamp_ms: int,
    config: CostBudgetConfig | None = None,
    *,
    fill_fraction: float = 1.0,
) -> CostBudgetState:
    policy = config or CostBudgetConfig()
    fraction = _unit(fill_fraction, "fill_fraction")
    return CostBudgetState(
        timestamp_ms=_timestamp(timestamp_ms),
        available_tokens=policy.token_capacity * fraction,
        available_slot_seconds=policy.slot_seconds_capacity * fraction,
    )


@dataclass(frozen=True)
class CostedAttentionCandidate:
    channel_id: str
    decision: AttentionDecision
    cost: InferenceCost
    episode_id: str = ""
    ready_at_ms: int = 0

    def __post_init__(self) -> None:
        if self.channel_id != self.decision.channel_id:
            raise ValueError("candidate and decision channel_id must match")
        object.__setattr__(self, "ready_at_ms", _timestamp(self.ready_at_ms))


@dataclass(frozen=True)
class CostAllocationEntry:
    channel_id: str
    episode_id: str
    phase: str
    cost: InferenceCost
    equivalent_l0: float
    priority: float
    coverage_debt: float


@dataclass(frozen=True)
class CostAwareAllocation:
    selected: tuple[CostAllocationEntry, ...]
    rejected: tuple[tuple[str, str], ...]
    state_before: CostBudgetState
    state_after: CostBudgetState
    repaid_tokens: float
    repaid_slot_seconds: float

    def printable(self, config: CostBudgetConfig | None = None) -> str:
        policy = config or CostBudgetConfig()
        return json.dumps(
            {
                "burst_debt_l0_after": round(
                    self.state_after.debt_l0(policy), 6
                ),
                "burst_debt_l0_before": round(
                    self.state_before.debt_l0(policy), 6
                ),
                "rejected": [
                    {"channel_id": channel_id, "reason": reason}
                    for channel_id, reason in self.rejected
                ],
                "repaid_slot_seconds": round(self.repaid_slot_seconds, 6),
                "repaid_tokens": round(self.repaid_tokens, 6),
                "selected": [
                    {
                        "channel_id": item.channel_id,
                        "coverage_debt": round(item.coverage_debt, 6),
                        "episode_id": item.episode_id,
                        "equivalent_l0": round(item.equivalent_l0, 6),
                        "phase": item.phase,
                        "priority": round(item.priority, 6),
                        "slot_seconds": round(item.cost.slot_seconds, 6),
                        "tokens": item.cost.total_tokens,
                    }
                    for item in self.selected
                ],
                "state_after": {
                    "available_slot_seconds": round(
                        self.state_after.available_slot_seconds, 6
                    ),
                    "available_tokens": round(
                        self.state_after.available_tokens, 6
                    ),
                    "timestamp_ms": self.state_after.timestamp_ms,
                },
                "state_before": {
                    "available_slot_seconds": round(
                        self.state_before.available_slot_seconds, 6
                    ),
                    "available_tokens": round(
                        self.state_before.available_tokens, 6
                    ),
                    "timestamp_ms": self.state_before.timestamp_ms,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def replenish_cost_budget(
    state: CostBudgetState,
    now_ms: int,
    config: CostBudgetConfig | None = None,
) -> tuple[CostBudgetState, float, float]:
    """Refill the bucket; negative burst balances are repaid before new credit."""

    policy = config or CostBudgetConfig()
    now = _timestamp(now_ms, "now_ms")
    if now < state.timestamp_ms:
        raise ValueError("budget timestamps must be monotonic")
    elapsed = now - state.timestamp_ms
    token_refill = policy.token_capacity * elapsed / policy.refill_window_ms
    slot_refill = policy.slot_seconds_capacity * elapsed / policy.refill_window_ms
    repaid_tokens = min(token_refill, max(0.0, -state.available_tokens))
    repaid_slots = min(slot_refill, max(0.0, -state.available_slot_seconds))
    replenished = CostBudgetState(
        timestamp_ms=now,
        available_tokens=min(
            policy.token_capacity, state.available_tokens + token_refill
        ),
        available_slot_seconds=min(
            policy.slot_seconds_capacity,
            state.available_slot_seconds + slot_refill,
        ),
    )
    return replenished, repaid_tokens, repaid_slots


def allocate_cost_aware_attention(
    candidates: Iterable[CostedAttentionCandidate],
    now_ms: int,
    state: CostBudgetState,
    config: CostBudgetConfig | None = None,
) -> CostAwareAllocation:
    """Allocate token and slot-seconds jointly with burst borrowing and fairness."""

    policy = config or CostBudgetConfig()
    now = _timestamp(now_ms, "now_ms")
    before, repaid_tokens, repaid_slots = replenish_cost_budget(state, now, policy)
    ordered = sorted(candidates, key=lambda candidate: candidate.channel_id)
    if len({candidate.channel_id for candidate in ordered}) != len(ordered):
        raise ValueError("only one costed candidate per channel is allowed")
    rejected: dict[str, str] = {}
    eligible: list[CostedAttentionCandidate] = []
    for candidate in ordered:
        profile = profile_for_mode(candidate.decision.mode)
        if not profile.dispatch_enabled:
            rejected[candidate.channel_id] = "mode_dispatch_disabled"
        elif candidate.ready_at_ms > now:
            rejected[candidate.channel_id] = (
                f"not_ready_until:{candidate.ready_at_ms}"
            )
        elif (
            candidate.cost.total_tokens
            > policy.token_capacity + policy.token_borrow_limit
            or candidate.cost.slot_seconds
            > policy.slot_seconds_capacity + policy.slot_seconds_borrow_limit
        ):
            rejected[candidate.channel_id] = "cost_exceeds_steady_plus_burst_limit"
        else:
            eligible.append(candidate)

    tokens = before.available_tokens
    slots = before.available_slot_seconds
    selected: list[CostAllocationEntry] = []
    selected_ids: set[str] = set()

    def can_admit(
        candidate: CostedAttentionCandidate, *, allow_borrow: bool
    ) -> bool:
        token_floor = -policy.token_borrow_limit if allow_borrow else 0.0
        slot_floor = (
            -policy.slot_seconds_borrow_limit if allow_borrow else 0.0
        )
        return (
            tokens - candidate.cost.total_tokens >= token_floor - 1e-9
            and slots - candidate.cost.slot_seconds >= slot_floor - 1e-9
        )

    def admit(candidate: CostedAttentionCandidate, phase: str) -> bool:
        nonlocal tokens, slots
        if len(selected) >= policy.max_jobs_per_cycle:
            return False
        if candidate.channel_id in selected_ids:
            return False
        borrow = candidate.decision.mode is AttentionMode.BURST
        if not can_admit(candidate, allow_borrow=borrow):
            return False
        tokens -= candidate.cost.total_tokens
        slots -= candidate.cost.slot_seconds
        selected_ids.add(candidate.channel_id)
        selected.append(
            CostAllocationEntry(
                channel_id=candidate.channel_id,
                episode_id=candidate.episode_id,
                phase=phase,
                cost=candidate.cost,
                equivalent_l0=policy.equivalent_l0(candidate.cost),
                priority=candidate.decision.priority,
                coverage_debt=candidate.decision.coverage_debt,
            )
        )
        return True

    overdue = sorted(
        (
            candidate
            for candidate in eligible
            if candidate.decision.coverage_debt >= 1.0
            and candidate.decision.mode is not AttentionMode.BURST
        ),
        key=lambda candidate: (
            -candidate.decision.coverage_debt,
            candidate.decision.state.last_vlm_ms
            if candidate.decision.state.last_vlm_ms is not None
            else -1,
            candidate.channel_id,
        ),
    )
    fairness_tokens = max(0.0, tokens) * policy.fairness_fraction
    fairness_slots = max(0.0, slots) * policy.fairness_fraction
    fair_tokens_used = 0.0
    fair_slots_used = 0.0
    for candidate in overdue:
        within_reserve = (
            fair_tokens_used + candidate.cost.total_tokens <= fairness_tokens + 1e-9
            and fair_slots_used + candidate.cost.slot_seconds
            <= fairness_slots + 1e-9
        )
        first_fair_candidate = not any(
            item.phase == "fairness" for item in selected
        )
        if not within_reserve and not first_fair_candidate:
            continue
        if admit(candidate, "fairness"):
            fair_tokens_used += candidate.cost.total_tokens
            fair_slots_used += candidate.cost.slot_seconds

    urgent = sorted(
        (
            candidate
            for candidate in eligible
            if candidate.decision.mode is AttentionMode.BURST
            or candidate.decision.priority >= policy.urgent_priority
        ),
        key=lambda candidate: (
            -candidate.decision.priority,
            -candidate.decision.coverage_debt,
            candidate.channel_id,
        ),
    )
    for candidate in urgent:
        admit(candidate, "urgent")

    priority = sorted(
        eligible,
        key=lambda candidate: (
            -candidate.decision.priority,
            -candidate.decision.coverage_debt,
            candidate.channel_id,
        ),
    )
    for candidate in priority:
        admit(candidate, "priority")

    for candidate in eligible:
        if candidate.channel_id in selected_ids:
            continue
        rejected[candidate.channel_id] = (
            "max_jobs_reached"
            if len(selected) >= policy.max_jobs_per_cycle
            else "token_budget_exhausted"
            if tokens < candidate.cost.total_tokens
            else "slot_seconds_budget_exhausted"
        )
    phase_order = {"urgent": 0, "fairness": 1, "priority": 2}
    selected.sort(
        key=lambda item: (
            phase_order[item.phase],
            -item.priority,
            item.channel_id,
        )
    )
    after = CostBudgetState(
        timestamp_ms=now,
        available_tokens=tokens,
        available_slot_seconds=slots,
    )
    return CostAwareAllocation(
        selected=tuple(selected),
        rejected=tuple(sorted(rejected.items())),
        state_before=before,
        state_after=after,
        repaid_tokens=repaid_tokens,
        repaid_slot_seconds=repaid_slots,
    )


@dataclass(frozen=True)
class EpisodeConfig:
    pre_roll_ms: int = 5_000
    post_roll_ms: int = 5_000
    control_lookback_ms: int = 30_000
    trigger_tolerance_ms: int = 2_000
    onset_tolerance_ms: int = 1_500
    apex_tolerance_ms: int = 1_500
    max_apex_frames: int = 2
    max_frames: int = 8

    def __post_init__(self) -> None:
        for name in (
            "pre_roll_ms",
            "post_roll_ms",
            "control_lookback_ms",
            "trigger_tolerance_ms",
            "onset_tolerance_ms",
            "apex_tolerance_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_apex_frames <= 0:
            raise ValueError("max_apex_frames must be positive")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")


@dataclass(frozen=True)
class EpisodeFrame:
    snapshot_id: str
    timestamp_ms: int
    embedding_ref: str
    roles: tuple[EpisodeRole, ...]
    frame_hash: str = ""
    probe_scores: tuple[ProbeScore, ...] = ()


@dataclass(frozen=True)
class AttentionEpisode:
    channel_id: str
    trigger_timestamp_ms: int
    interval_start_ms: int | None
    interval_end_ms: int | None
    frames: tuple[EpisodeFrame, ...]
    reasons: tuple[str, ...]

    def printable(self) -> str:
        return json.dumps(
            {
                "channel_id": self.channel_id,
                "frames": [
                    {
                        "embedding_ref": frame.embedding_ref,
                        "frame_hash": frame.frame_hash,
                        "probe_scores": [
                            {
                                "margin": round(score.margin, 6),
                                "negative": round(score.negative, 6),
                                "positive": round(score.positive, 6),
                                "probe_id": score.probe_id,
                                "probe_version": score.probe_version,
                            }
                            for score in frame.probe_scores
                        ],
                        "roles": [role.value for role in frame.roles],
                        "snapshot_id": frame.snapshot_id,
                        "timestamp_ms": frame.timestamp_ms,
                    }
                    for frame in self.frames
                ],
                "interval_end_ms": self.interval_end_ms,
                "interval_start_ms": self.interval_start_ms,
                "reasons": list(self.reasons),
                "trigger_timestamp_ms": self.trigger_timestamp_ms,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class ModelFrameCandidate:
    """Embedding-backed snapshot metadata eligible for one model request."""

    channel_id: str
    snapshot_id: str
    timestamp_ms: int
    embedding_ref: str
    frame_hash: str = ""
    roles: tuple[EpisodeRole, ...] = ()
    motion_score: float = 0.0
    probe_score: float = 0.0
    salience: float = 0.0
    sharpness_score: float = 0.0
    estimated_tokens: int = 512
    embedding: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.channel_id:
            raise ValueError("channel_id must not be empty")
        if not self.snapshot_id:
            raise ValueError("snapshot_id must not be empty")
        if not self.embedding_ref:
            raise ValueError("embedding_ref must not be empty")
        object.__setattr__(self, "timestamp_ms", _timestamp(self.timestamp_ms))
        object.__setattr__(
            self,
            "roles",
            tuple(
                role if isinstance(role, EpisodeRole) else EpisodeRole(str(role))
                for role in self.roles
            ),
        )
        for name in (
            "motion_score",
            "probe_score",
            "salience",
            "sharpness_score",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if int(self.estimated_tokens) <= 0:
            raise ValueError("estimated_tokens must be positive")
        object.__setattr__(self, "estimated_tokens", int(self.estimated_tokens))
        vector = tuple(
            _finite(value, f"embedding[{index}]")
            for index, value in enumerate(self.embedding)
        )
        object.__setattr__(self, "embedding", vector)


@dataclass(frozen=True)
class FrameSelectorConfig:
    redundancy_weight: float = 0.42
    temporal_coverage_weight: float = 0.12
    required_roles: tuple[EpisodeRole, ...] = ()

    def __post_init__(self) -> None:
        _non_negative(self.redundancy_weight, "redundancy_weight")
        _non_negative(self.temporal_coverage_weight, "temporal_coverage_weight")
        normalized = tuple(
            role if isinstance(role, EpisodeRole) else EpisodeRole(str(role))
            for role in self.required_roles
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("required_roles must be unique")
        object.__setattr__(self, "required_roles", normalized)


@dataclass(frozen=True)
class SelectedModelFrame:
    channel_id: str
    snapshot_id: str
    timestamp_ms: int
    embedding_ref: str
    frame_hash: str
    roles: tuple[EpisodeRole, ...]
    motion_score: float
    probe_score: float
    selection_score: float
    redundancy: float
    estimated_tokens: int


@dataclass(frozen=True)
class ModelFrameSelection:
    channel_id: str
    mode: AttentionMode
    frames: tuple[SelectedModelFrame, ...]
    token_budget: int
    estimated_tokens: int
    missing_roles: tuple[EpisodeRole, ...]
    trimmed_snapshot_ids: tuple[str, ...]
    preflight_ok: bool
    reasons: tuple[str, ...]

    def printable(self) -> str:
        return json.dumps(
            {
                "channel_id": self.channel_id,
                "estimated_tokens": self.estimated_tokens,
                "frames": [
                    {
                        "embedding_ref": frame.embedding_ref,
                        "estimated_tokens": frame.estimated_tokens,
                        "frame_hash": frame.frame_hash,
                        "motion_score": round(frame.motion_score, 6),
                        "probe_score": round(frame.probe_score, 6),
                        "redundancy": round(frame.redundancy, 6),
                        "roles": [role.value for role in frame.roles],
                        "selection_score": round(frame.selection_score, 6),
                        "snapshot_id": frame.snapshot_id,
                        "timestamp_ms": frame.timestamp_ms,
                    }
                    for frame in self.frames
                ],
                "missing_roles": [role.value for role in self.missing_roles],
                "mode": self.mode.value,
                "preflight_ok": self.preflight_ok,
                "reasons": list(self.reasons),
                "token_budget": self.token_budget,
                "trimmed_snapshot_ids": list(self.trimmed_snapshot_ids),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _candidate_similarity(
    left: ModelFrameCandidate, right: ModelFrameCandidate
) -> float:
    if left.snapshot_id == right.snapshot_id:
        return 1.0
    if left.frame_hash and left.frame_hash == right.frame_hash:
        return 1.0
    if (
        left.embedding
        and right.embedding
        and len(left.embedding) == len(right.embedding)
    ):
        dot = sum(a * b for a, b in zip(left.embedding, right.embedding))
        left_norm = math.sqrt(sum(value * value for value in left.embedding))
        right_norm = math.sqrt(sum(value * value for value in right.embedding))
        if left_norm > 0.0 and right_norm > 0.0:
            return max(0.0, min(1.0, dot / (left_norm * right_norm)))
    return 0.0


def select_model_frames(
    channel_id: str,
    candidates: Iterable[ModelFrameCandidate],
    *,
    mode: AttentionMode | str,
    token_budget: int,
    preset: PortAttentionPreset = PORT_EIGHT_CHANNEL_PRESET,
    config: FrameSelectorConfig | None = None,
) -> ModelFrameSelection:
    """Choose 6-16 chronological model frames, then run token preflight.

    The latest candidate is automatically tagged ``current``.  Other mandatory
    roles must be assigned by the episode builder from CV interval/apex links.
    Images are never accepted; output retains only snapshot and embedding refs.
    """

    if not channel_id:
        raise ValueError("channel_id must not be empty")
    if int(token_budget) <= 0:
        raise ValueError("token_budget must be positive")
    normalized_mode = (
        mode if isinstance(mode, AttentionMode) else AttentionMode(str(mode).lower())
    )
    profile = preset.profile_for_mode(normalized_mode)
    policy = config or FrameSelectorConfig()
    automatic_required = (
        (
            EpisodeRole.CONTROL,
            EpisodeRole.PRE,
            EpisodeRole.ONSET,
            EpisodeRole.APEX,
            EpisodeRole.POST,
            EpisodeRole.CURRENT,
        )
        if normalized_mode in (AttentionMode.ACTIVE, AttentionMode.BURST)
        else (EpisodeRole.CONTROL, EpisodeRole.CURRENT)
    )
    required_roles = policy.required_roles or automatic_required
    ordered = sorted(
        (candidate for candidate in candidates if candidate.channel_id == channel_id),
        key=lambda candidate: (candidate.timestamp_ms, candidate.snapshot_id),
    )
    if len({candidate.snapshot_id for candidate in ordered}) != len(ordered):
        raise ValueError("snapshot_id values must be unique per selection")
    if not ordered:
        return ModelFrameSelection(
            channel_id=channel_id,
            mode=normalized_mode,
            frames=(),
            token_budget=int(token_budget),
            estimated_tokens=0,
            missing_roles=required_roles,
            trimmed_snapshot_ids=(),
            preflight_ok=False,
            reasons=("no_embedding_backed_candidates",),
        )

    latest_id = ordered[-1].snapshot_id

    def roles_for(candidate: ModelFrameCandidate) -> tuple[EpisodeRole, ...]:
        roles = list(candidate.roles)
        if candidate.snapshot_id == latest_id and EpisodeRole.CURRENT not in roles:
            roles.append(EpisodeRole.CURRENT)
        return tuple(roles)

    roles_by_id = {candidate.snapshot_id: roles_for(candidate) for candidate in ordered}

    def base_score(candidate: ModelFrameCandidate) -> float:
        return (
            0.42 * candidate.salience
            + 0.28 * candidate.motion_score
            + 0.20 * candidate.probe_score
            + 0.10 * candidate.sharpness_score
        )

    event_roles = (
        EpisodeRole.PRE,
        EpisodeRole.ONSET,
        EpisodeRole.APEX,
        EpisodeRole.POST,
    )
    present_event_roles = tuple(
        role
        for role in event_roles
        if any(role in roles_by_id[candidate.snapshot_id] for candidate in ordered)
    )
    anchor_roles = tuple(dict.fromkeys((*required_roles, *present_event_roles)))
    missing_roles = tuple(
        role
        for role in required_roles
        if not any(role in roles_by_id[candidate.snapshot_id] for candidate in ordered)
    )
    selected: dict[str, ModelFrameCandidate] = {}
    protected: set[str] = set()
    selection_metrics: dict[str, tuple[float, float]] = {}

    for role in anchor_roles:
        role_candidates = [
            candidate
            for candidate in ordered
            if role in roles_by_id[candidate.snapshot_id]
        ]
        if not role_candidates:
            continue
        if role is EpisodeRole.CURRENT:
            chosen = max(
                role_candidates,
                key=lambda candidate: (
                    candidate.timestamp_ms,
                    base_score(candidate),
                    -candidate.estimated_tokens,
                    candidate.snapshot_id,
                ),
            )
        else:
            chosen = max(
                role_candidates,
                key=lambda candidate: (
                    base_score(candidate),
                    -candidate.estimated_tokens,
                    -candidate.timestamp_ms,
                    candidate.snapshot_id,
                ),
            )
        selected[chosen.snapshot_id] = chosen
        protected.add(chosen.snapshot_id)
        selection_metrics.setdefault(chosen.snapshot_id, (base_score(chosen), 0.0))

    target = min(profile.max_frames, max(profile.min_frames, profile.target_frames))
    while len(selected) < target:
        remaining = [
            candidate
            for candidate in ordered
            if candidate.snapshot_id not in selected
        ]
        if not remaining:
            break
        scored: list[tuple[float, float, ModelFrameCandidate]] = []
        for candidate in remaining:
            redundancy = max(
                (
                    _candidate_similarity(candidate, chosen)
                    for chosen in selected.values()
                ),
                default=0.0,
            )
            total_span = max(1, ordered[-1].timestamp_ms - ordered[0].timestamp_ms)
            temporal_distance = min(
                (
                    abs(candidate.timestamp_ms - chosen.timestamp_ms)
                    for chosen in selected.values()
                ),
                default=total_span,
            )
            temporal_coverage = min(1.0, temporal_distance / total_span)
            score = (
                base_score(candidate)
                + policy.temporal_coverage_weight * temporal_coverage
                - policy.redundancy_weight * redundancy
            )
            scored.append((score, redundancy, candidate))
        score, redundancy, chosen = max(
            scored,
            key=lambda item: (
                item[0],
                -item[1],
                -item[2].estimated_tokens,
                -item[2].timestamp_ms,
                item[2].snapshot_id,
            ),
        )
        selected[chosen.snapshot_id] = chosen
        selection_metrics[chosen.snapshot_id] = (score, redundancy)

    trimmed: list[str] = []

    def selected_tokens() -> int:
        return sum(candidate.estimated_tokens for candidate in selected.values())

    while selected_tokens() > token_budget:
        removable = [
            candidate
            for snapshot_id, candidate in selected.items()
            if snapshot_id not in protected
        ]
        if not removable:
            break
        removed = min(
            removable,
            key=lambda candidate: (
                selection_metrics[candidate.snapshot_id][0],
                -candidate.estimated_tokens,
                candidate.timestamp_ms,
                candidate.snapshot_id,
            ),
        )
        trimmed.append(removed.snapshot_id)
        del selected[removed.snapshot_id]

    if len(selected) < profile.min_frames:
        refill = [
            candidate
            for candidate in ordered
            if candidate.snapshot_id not in selected
            and candidate.snapshot_id not in trimmed
        ]
        refill.sort(
            key=lambda candidate: (
                -base_score(candidate),
                candidate.estimated_tokens,
                candidate.timestamp_ms,
                candidate.snapshot_id,
            )
        )
        for candidate in refill:
            if len(selected) >= profile.min_frames:
                break
            if selected_tokens() + candidate.estimated_tokens > token_budget:
                continue
            redundancy = max(
                (
                    _candidate_similarity(candidate, chosen)
                    for chosen in selected.values()
                ),
                default=0.0,
            )
            selected[candidate.snapshot_id] = candidate
            selection_metrics[candidate.snapshot_id] = (
                base_score(candidate) - policy.redundancy_weight * redundancy,
                redundancy,
            )

    frames = tuple(
        SelectedModelFrame(
            channel_id=candidate.channel_id,
            snapshot_id=candidate.snapshot_id,
            timestamp_ms=candidate.timestamp_ms,
            embedding_ref=candidate.embedding_ref,
            frame_hash=candidate.frame_hash,
            roles=roles_by_id[candidate.snapshot_id],
            motion_score=candidate.motion_score,
            probe_score=candidate.probe_score,
            selection_score=selection_metrics[candidate.snapshot_id][0],
            redundancy=selection_metrics[candidate.snapshot_id][1],
            estimated_tokens=candidate.estimated_tokens,
        )
        for candidate in sorted(
            selected.values(),
            key=lambda candidate: (candidate.timestamp_ms, candidate.snapshot_id),
        )
    )
    estimated_tokens = sum(frame.estimated_tokens for frame in frames)
    preflight_ok = (
        not missing_roles
        and profile.min_frames <= len(frames) <= profile.max_frames
        and estimated_tokens <= token_budget
    )
    reasons: list[str] = [
        f"profile={normalized_mode.value}:{profile.min_frames}/{profile.target_frames}/{profile.max_frames}",
        f"frames={len(frames)}",
        f"tokens={estimated_tokens}/{token_budget}",
    ]
    if missing_roles:
        reasons.append(
            "missing_roles=" + ",".join(role.value for role in missing_roles)
        )
    if trimmed:
        reasons.append("token_preflight_trimmed=" + ",".join(trimmed))
    if len(frames) < profile.min_frames:
        reasons.append(f"below_min_frames={profile.min_frames}")
    if estimated_tokens > token_budget:
        reasons.append("mandatory_anchors_exceed_token_budget")
    return ModelFrameSelection(
        channel_id=channel_id,
        mode=normalized_mode,
        frames=frames,
        token_budget=int(token_budget),
        estimated_tokens=estimated_tokens,
        missing_roles=missing_roles,
        trimmed_snapshot_ids=tuple(trimmed),
        preflight_ok=preflight_ok,
        reasons=tuple(reasons),
    )


def build_control_episode(
    channel_id: str,
    trigger_timestamp_ms: int,
    snapshots: Iterable[EmbeddingSnapshot],
    *,
    lookback_ms: int = 30_000,
) -> AttentionEpisode:
    """Build a quiet-channel audit from the latest saved embedding snapshot."""

    if not channel_id:
        raise ValueError("channel_id must not be empty")
    trigger = _timestamp(trigger_timestamp_ms, "trigger_timestamp_ms")
    if lookback_ms < 0:
        raise ValueError("lookback_ms must be non-negative")
    candidates = sorted(
        (
            snapshot
            for snapshot in snapshots
            if snapshot.channel_id == channel_id
            and trigger - lookback_ms <= snapshot.timestamp_ms <= trigger
        ),
        key=lambda snapshot: (snapshot.timestamp_ms, snapshot.snapshot_id),
    )
    if not candidates:
        return AttentionEpisode(
            channel_id=channel_id,
            trigger_timestamp_ms=trigger,
            interval_start_ms=None,
            interval_end_ms=None,
            frames=(),
            reasons=("no_saved_snapshot_for_control",),
        )
    snapshot = candidates[-1]
    return AttentionEpisode(
        channel_id=channel_id,
        trigger_timestamp_ms=trigger,
        interval_start_ms=None,
        interval_end_ms=None,
        frames=(
            EpisodeFrame(
                snapshot_id=snapshot.snapshot_id,
                timestamp_ms=snapshot.timestamp_ms,
                embedding_ref=snapshot.embedding_ref,
                roles=(EpisodeRole.CONTROL,),
                frame_hash=snapshot.frame_hash,
                probe_scores=snapshot.probe_scores,
            ),
        ),
        reasons=("control_only:quiet_audit", "roles=control"),
    )


def build_attention_episode(
    channel_id: str,
    trigger_timestamp_ms: int,
    intervals: Iterable[MotionInterval],
    snapshots: Iterable[EmbeddingSnapshot],
    config: EpisodeConfig | None = None,
) -> AttentionEpisode:
    """Select control/pre/onset/apex/post references from saved snapshots only."""

    if not channel_id:
        raise ValueError("channel_id must not be empty")
    trigger = _timestamp(trigger_timestamp_ms, "trigger_timestamp_ms")
    policy = config or EpisodeConfig()
    channel_intervals = sorted(
        (
            interval
            for interval in intervals
            if interval.channel_id == channel_id
        ),
        key=lambda interval: (interval.start_ms, interval.end_ms),
    )
    channel_snapshots = sorted(
        (
            snapshot
            for snapshot in snapshots
            if snapshot.channel_id == channel_id
        ),
        key=lambda snapshot: (snapshot.timestamp_ms, snapshot.snapshot_id),
    )
    snapshot_by_id = {snapshot.snapshot_id: snapshot for snapshot in channel_snapshots}
    if len(snapshot_by_id) != len(channel_snapshots):
        raise ValueError("snapshot_id values must be unique per channel")

    motion_intervals = [
        interval
        for interval in channel_intervals
        if interval.kind is MotionKind.MOTION
    ]
    containing = [
        interval
        for interval in motion_intervals
        if interval.start_ms <= trigger <= interval.end_ms
    ]
    if containing:
        event = min(
            containing,
            key=lambda interval: (
                abs(interval.peak_timestamp_ms - trigger),
                interval.start_ms,
            ),
        )
    elif motion_intervals:
        event = min(
            motion_intervals,
            key=lambda interval: (
                min(
                    abs(trigger - interval.start_ms),
                    abs(trigger - interval.end_ms),
                ),
                interval.start_ms,
            ),
        )
        distance = min(
            abs(trigger - event.start_ms),
            abs(trigger - event.end_ms),
        )
        if distance > policy.trigger_tolerance_ms:
            event = None
    else:
        event = None

    if event is None:
        return AttentionEpisode(
            channel_id=channel_id,
            trigger_timestamp_ms=trigger,
            interval_start_ms=None,
            interval_end_ms=None,
            frames=(),
            reasons=("no_motion_interval_near_trigger",),
        )

    selected: dict[str, tuple[EmbeddingSnapshot, list[EpisodeRole]]] = {}
    reasons: list[str] = []

    def add(snapshot: EmbeddingSnapshot | None, role: EpisodeRole) -> None:
        if snapshot is None:
            reasons.append(f"missing_role:{role.value}")
            return
        existing = selected.get(snapshot.snapshot_id)
        if existing is None:
            selected[snapshot.snapshot_id] = (snapshot, [role])
        elif role not in existing[1]:
            existing[1].append(role)

    linked_event_snapshots = [
        snapshot_by_id[snapshot_id]
        for snapshot_id in event.linked_snapshot_ids
        if snapshot_id in snapshot_by_id
    ]
    all_before = [
        snapshot
        for snapshot in channel_snapshots
        if event.start_ms - policy.pre_roll_ms
        <= snapshot.timestamp_ms
        < event.start_ms
    ]
    add(all_before[-1] if all_before else None, EpisodeRole.PRE)

    onset_pool = [
        snapshot
        for snapshot in linked_event_snapshots
        if abs(snapshot.timestamp_ms - event.start_ms) <= policy.onset_tolerance_ms
    ]
    add(
        min(
            onset_pool,
            key=lambda snapshot: (
                abs(snapshot.timestamp_ms - event.start_ms),
                snapshot.timestamp_ms,
                snapshot.snapshot_id,
            ),
        )
        if onset_pool
        else None,
        EpisodeRole.ONSET,
    )

    linked_apex_ids: list[str] = []
    for link in sorted(
        event.apex_links,
        key=lambda item: (
            abs(item.apex_timestamp_ms - trigger),
            item.apex_timestamp_ms,
            item.snapshot_id,
        ),
    ):
        if (
            link.snapshot_id in snapshot_by_id
            and link.snapshot_id not in linked_apex_ids
            and link.distance_ms <= policy.apex_tolerance_ms
        ):
            linked_apex_ids.append(link.snapshot_id)
        if len(linked_apex_ids) >= policy.max_apex_frames:
            break
    if linked_apex_ids:
        for snapshot_id in linked_apex_ids:
            add(snapshot_by_id[snapshot_id], EpisodeRole.APEX)
    else:
        apex_pool = [
            snapshot
            for snapshot in linked_event_snapshots
            if abs(snapshot.timestamp_ms - event.peak_timestamp_ms)
            <= policy.apex_tolerance_ms
        ]
        add(
            min(
                apex_pool,
                key=lambda snapshot: (
                    abs(snapshot.timestamp_ms - event.peak_timestamp_ms),
                    snapshot.timestamp_ms,
                    snapshot.snapshot_id,
                ),
            )
            if apex_pool
            else None,
            EpisodeRole.APEX,
        )
        if apex_pool:
            reasons.append("apex_fallback:cv_peak")

    post_pool = [
        snapshot
        for snapshot in channel_snapshots
        if event.end_ms
        < snapshot.timestamp_ms
        <= event.end_ms + policy.post_roll_ms
    ]
    add(post_pool[0] if post_pool else None, EpisodeRole.POST)

    quiet_before = [
        interval
        for interval in channel_intervals
        if interval.kind is MotionKind.QUIET
        and interval.end_ms < event.start_ms
        and interval.end_ms >= event.start_ms - policy.control_lookback_ms
    ]
    control_pool: list[EmbeddingSnapshot] = []
    if quiet_before:
        control_interval = quiet_before[-1]
        control_pool = [
            snapshot_by_id[snapshot_id]
            for snapshot_id in control_interval.linked_snapshot_ids
            if snapshot_id in snapshot_by_id
        ]
    add(control_pool[-1] if control_pool else None, EpisodeRole.CONTROL)

    role_order = {
        EpisodeRole.CONTROL: 0,
        EpisodeRole.PRE: 1,
        EpisodeRole.ONSET: 2,
        EpisodeRole.APEX: 3,
        EpisodeRole.POST: 4,
    }
    frames = tuple(
        EpisodeFrame(
            snapshot_id=snapshot.snapshot_id,
            timestamp_ms=snapshot.timestamp_ms,
            embedding_ref=snapshot.embedding_ref,
            roles=tuple(sorted(roles, key=role_order.__getitem__)),
            frame_hash=snapshot.frame_hash,
            probe_scores=snapshot.probe_scores,
        )
        for snapshot, roles in sorted(
            selected.values(),
            key=lambda item: (item[0].timestamp_ms, item[0].snapshot_id),
        )[: policy.max_frames]
    )
    selected_roles = {role for frame in frames for role in frame.roles}
    if len(frames) >= policy.max_frames and len(selected) > policy.max_frames:
        reasons.append("frame_limit_applied")
    reasons.append(
        "roles="
        + ",".join(
            role.value for role in EpisodeRole if role in selected_roles
        )
    )
    return AttentionEpisode(
        channel_id=channel_id,
        trigger_timestamp_ms=trigger,
        interval_start_ms=event.start_ms,
        interval_end_ms=event.end_ms,
        frames=frames,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class CoordinatorConfig:
    """Runtime-neutral bounds for the small in-memory planning facade."""

    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    attention: AttentionPolicyConfig = field(default_factory=AttentionPolicyConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    budget: GlobalBudgetConfig = field(
        default_factory=lambda: GlobalBudgetConfig(
            total_units=4.0,
            max_jobs=4,
            fairness_fraction=0.25,
        )
    )
    max_channels: int = 16
    cv_retention_ms: int = 120_000
    snapshot_retention_ms: int = 300_000
    expected_snapshot_interval_ms: int = 1_000
    persistence_window_ms: int = 5_000
    minimum_dispatch_gap_ms: int = 3_000
    burst_postroll_wait_ms: int = 2_000
    control_audit_lookback_ms: int = 30_000
    default_job_units: float = 1.0
    active_priority_dispatch: float = 0.52

    def __post_init__(self) -> None:
        if self.max_channels <= 0:
            raise ValueError("max_channels must be positive")
        for name in (
            "cv_retention_ms",
            "snapshot_retention_ms",
            "expected_snapshot_interval_ms",
            "persistence_window_ms",
            "minimum_dispatch_gap_ms",
            "burst_postroll_wait_ms",
            "control_audit_lookback_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.expected_snapshot_interval_ms == 0:
            raise ValueError("expected_snapshot_interval_ms must be positive")
        if _non_negative(self.default_job_units, "default_job_units") == 0:
            raise ValueError("default_job_units must be positive")
        _non_negative(self.active_priority_dispatch, "active_priority_dispatch")


@dataclass(frozen=True)
class DispatchReadiness:
    channel_id: str
    due: bool
    reason: str
    trigger_timestamp_ms: int
    estimated_units: float
    decision: AttentionDecision | None

    def printable(self) -> str:
        return json.dumps(
            {
                "channel_id": self.channel_id,
                "decision": (
                    json.loads(self.decision.printable())
                    if self.decision is not None
                    else None
                ),
                "due": self.due,
                "estimated_units": round(self.estimated_units, 6),
                "reason": self.reason,
                "trigger_timestamp_ms": self.trigger_timestamp_ms,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class PlannedDispatch:
    channel_id: str
    episode_id: str
    phase: str
    estimated_units: float
    decision: AttentionDecision
    episode: AttentionEpisode


@dataclass(frozen=True)
class AttentionPlan:
    now_ms: int
    allocation: AttentionAllocation
    jobs: tuple[PlannedDispatch, ...]
    not_due: tuple[tuple[str, str], ...]

    def printable(self) -> str:
        return json.dumps(
            {
                "allocation": json.loads(self.allocation.printable()),
                "jobs": [
                    {
                        "channel_id": job.channel_id,
                        "decision": json.loads(job.decision.printable()),
                        "episode": json.loads(job.episode.printable()),
                        "episode_id": job.episode_id,
                        "estimated_units": round(job.estimated_units, 6),
                        "phase": job.phase,
                    }
                    for job in self.jobs
                ],
                "not_due": [
                    {"channel_id": channel_id, "reason": reason}
                    for channel_id, reason in self.not_due
                ],
                "now_ms": self.now_ms,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass
class _ChannelRuntime:
    samples: list[CvSample] = field(default_factory=list)
    snapshots: list[EmbeddingSnapshot] = field(default_factory=list)
    apex_markers: list[ApexMarker] = field(default_factory=list)
    state: AttentionState | None = None
    decision: AttentionDecision | None = None
    capture_mode: str = "quiet"
    last_activity_x: float = 0.0
    inflight_job_id: str = ""
    inflight_since_ms: int | None = None
    last_completed_ms: int | None = None
    last_completion_success: bool | None = None


class AttentionCoordinator:
    """Compact integration facade for shadow-mode or runtime wiring.

    It keeps only bounded CV scalars, embedding references, and policy state.
    It performs no I/O and never owns image bytes.
    """

    _CAPTURE_MODES = frozenset({"quiet", "normal", "watch", "active", "burst"})

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self.config = config or CoordinatorConfig()
        self.policy = HomeostaticAttentionPolicy(self.config.attention)
        self._channels: dict[str, _ChannelRuntime] = {}
        self._lock = threading.RLock()

    def observe_cv(
        self,
        channel_id: str,
        timestamp_ms: int,
        motion_score: float,
        activity_x: float,
        mode: str,
        *,
        source_health: float = 1.0,
        uncertainty: float = 0.0,
        alert_persistence: float = 0.0,
        probe_novelty: float = 0.0,
        redundancy: float = 0.0,
        sharpness: float = 0.0,
    ) -> AttentionDecision:
        """Observe one dense CV scalar sample and update channel state."""

        timestamp = _timestamp(timestamp_ms)
        capture_mode = str(mode).strip().lower()
        if capture_mode not in self._CAPTURE_MODES:
            raise ValueError(
                "mode must be one of: " + ", ".join(sorted(self._CAPTURE_MODES))
            )
        activity = _non_negative(activity_x, "activity_x")
        sample = CvSample(timestamp, motion_score, sharpness)
        with self._lock:
            runtime = self._runtime(channel_id)
            if runtime.samples and runtime.samples[-1].timestamp_ms >= timestamp:
                raise ValueError("CV observations must have increasing timestamps")
            runtime.samples.append(sample)
            runtime.capture_mode = capture_mode
            runtime.last_activity_x = activity
            self._trim(runtime, timestamp)

            recent = [
                item
                for item in runtime.samples
                if item.timestamp_ms >= timestamp - self.config.persistence_window_ms
            ]
            persistence = (
                sum(
                    item.motion >= self.config.aggregation.motion_exit
                    for item in recent
                )
                / len(recent)
            )
            activity_signal = 1.0 - math.exp(-max(0.0, activity - 1.0))
            burst_signal = (
                1.0
                if capture_mode == "burst"
                else min(1.0, max(0.0, activity - 2.0) / 3.0)
            )
            latest_snapshot = runtime.snapshots[-1] if runtime.snapshots else None
            staleness = (
                1.0
                if latest_snapshot is None
                else min(
                    1.0,
                    max(0, timestamp - latest_snapshot.timestamp_ms)
                    / self.config.expected_snapshot_interval_ms,
                )
            )
            positive, margin = self._latest_probe_signal(latest_snapshot)
            vector = AttentionVector(
                timestamp_ms=timestamp,
                motion_intensity=max(sample.motion, activity_signal),
                motion_persistence=persistence,
                burst=burst_signal,
                probe_positive=positive,
                probe_margin=margin,
                probe_novelty=probe_novelty,
                uncertainty=uncertainty,
                alert_persistence=alert_persistence,
                signal_staleness=staleness,
                source_health=source_health,
                redundancy=redundancy,
            )
            decision = self.policy.evaluate(
                channel_id,
                vector,
                runtime.state,
            )
            runtime.state = decision.state
            runtime.decision = decision
            return decision

    def link_snapshot(
        self,
        channel_id: str,
        timestamp_ms: int,
        frame_ref: str,
        *,
        snapshot_id: str | None = None,
        frame_hash: str = "",
        probe_scores: Iterable[ProbeScore | Mapping[str, object]] = (),
        meta: Mapping[str, object] | None = None,
    ) -> EmbeddingSnapshot:
        """Link one embedding/frame reference; image payloads are rejected."""

        timestamp = _timestamp(timestamp_ms)
        metadata = dict(meta or {})
        forbidden = {
            "image",
            "image_bytes",
            "jpeg",
            "jpg",
            "base64",
            "data",
            "pixels",
            "thumbnail",
            "thumb",
        }
        blocked = forbidden.intersection(key.lower() for key in metadata)
        if blocked:
            raise ValueError(
                "image payload metadata is not allowed: " + ", ".join(sorted(blocked))
            )
        allowed = {"snapshot_id", "frame_hash", "probe_scores"}
        unknown = set(metadata).difference(allowed)
        if unknown:
            raise ValueError(
                "unsupported snapshot metadata: " + ", ".join(sorted(unknown))
            )
        effective_id = str(
            snapshot_id
            or metadata.get("snapshot_id")
            or f"{channel_id}:{timestamp}"
        )
        effective_hash = str(frame_hash or metadata.get("frame_hash") or "")
        provided_scores = tuple(probe_scores)
        score_values = (
            provided_scores
            if provided_scores
            else metadata.get("probe_scores", ())
        )
        parsed_scores = tuple(self._parse_probe_score(value) for value in score_values)
        snapshot = EmbeddingSnapshot(
            channel_id=channel_id,
            snapshot_id=effective_id,
            timestamp_ms=timestamp,
            embedding_ref=str(frame_ref),
            frame_hash=effective_hash,
            probe_scores=parsed_scores,
        )
        with self._lock:
            runtime = self._runtime(channel_id)
            if runtime.snapshots and runtime.snapshots[-1].timestamp_ms >= timestamp:
                raise ValueError("snapshot timestamps must be increasing")
            runtime.snapshots.append(snapshot)
            self._trim(runtime, timestamp)
        return snapshot

    def link_apex(
        self, channel_id: str, timestamp_ms: int, apex_id: str | None = None
    ) -> ApexMarker:
        marker = ApexMarker(apex_id or f"{channel_id}:apex:{timestamp_ms}", timestamp_ms)
        with self._lock:
            runtime = self._runtime(channel_id)
            if (
                runtime.apex_markers
                and runtime.apex_markers[-1].timestamp_ms > marker.timestamp_ms
            ):
                raise ValueError("apex timestamps must be monotonic")
            runtime.apex_markers.append(marker)
            self._trim(runtime, marker.timestamp_ms)
        return marker

    def should_dispatch(
        self, channel_id: str, now_ms: int | None = None
    ) -> DispatchReadiness:
        with self._lock:
            runtime = self._channels.get(channel_id)
            if runtime is None or runtime.decision is None:
                return DispatchReadiness(
                    channel_id, False, "no_attention_state", int(now_ms or 0), 0.0, None
                )
            now = (
                runtime.decision.state.vector.timestamp_ms
                if now_ms is None
                else _timestamp(now_ms, "now_ms")
            )
            return self._should_dispatch(channel_id, runtime, now)

    def plan_due(
        self,
        now_ms: int,
        budget: GlobalBudgetConfig | None = None,
    ) -> AttentionPlan:
        """Plan a deterministic cross-channel cycle without dispatch side effects."""

        now = _timestamp(now_ms, "now_ms")
        with self._lock:
            readiness = [
                self._should_dispatch(channel_id, runtime, now)
                for channel_id, runtime in sorted(self._channels.items())
                if runtime.decision is not None
            ]
            candidates = [
                AttentionCandidate(
                    item.channel_id,
                    item.decision,
                    estimated_units=item.estimated_units,
                    episode_id=self._episode_id(item.channel_id, item.trigger_timestamp_ms),
                )
                for item in readiness
                if item.due and item.decision is not None
            ]
            allocation = allocate_global_attention(
                candidates, now, budget or self.config.budget
            )
            selected_by_channel = {
                item.channel_id: item for item in allocation.selected
            }
            jobs: list[PlannedDispatch] = []
            for item in readiness:
                allocation_item = selected_by_channel.get(item.channel_id)
                if allocation_item is None or item.decision is None:
                    continue
                episode = self.select_episode(
                    item.channel_id, item.trigger_timestamp_ms
                )
                jobs.append(
                    PlannedDispatch(
                        channel_id=item.channel_id,
                        episode_id=allocation_item.episode_id,
                        phase=allocation_item.phase,
                        estimated_units=allocation_item.estimated_units,
                        decision=item.decision,
                        episode=episode,
                    )
                )
            return AttentionPlan(
                now_ms=now,
                allocation=allocation,
                jobs=tuple(jobs),
                not_due=tuple(
                    (item.channel_id, item.reason)
                    for item in readiness
                    if not item.due
                ),
            )

    def select_episode(
        self, channel_id: str, trigger_timestamp_ms: int
    ) -> AttentionEpisode:
        with self._lock:
            runtime = self._channels.get(channel_id)
            if runtime is None:
                return AttentionEpisode(
                    channel_id,
                    _timestamp(trigger_timestamp_ms),
                    None,
                    None,
                    (),
                    ("unknown_channel",),
                )
            intervals = aggregate_cv_intervals(
                channel_id,
                runtime.samples,
                runtime.snapshots,
                runtime.apex_markers,
                self.config.aggregation,
            )
            episode = build_attention_episode(
                channel_id,
                trigger_timestamp_ms,
                intervals,
                runtime.snapshots,
                self.config.episode,
            )
            if episode.frames:
                return episode
            return build_control_episode(
                channel_id,
                trigger_timestamp_ms,
                runtime.snapshots,
                lookback_ms=self.config.control_audit_lookback_ms,
            )

    def mark_dispatched(
        self,
        channel_id: str,
        timestamp_ms: int,
        job_id: str | None = None,
    ) -> str:
        timestamp = _timestamp(timestamp_ms)
        with self._lock:
            runtime = self._channels.get(channel_id)
            if runtime is None or runtime.state is None:
                raise ValueError("channel has no attention state")
            if runtime.inflight_job_id:
                raise ValueError("channel already has an inflight VLM job")
            effective_job_id = job_id or self._episode_id(channel_id, timestamp)
            runtime.inflight_job_id = effective_job_id
            runtime.inflight_since_ms = timestamp
            runtime.state = self.policy.record_dispatch(runtime.state, timestamp)
            if runtime.decision is not None:
                runtime.decision = replace(runtime.decision, state=runtime.state)
            return effective_job_id

    def mark_completed(
        self,
        channel_id: str,
        timestamp_ms: int,
        *,
        job_id: str | None = None,
        success: bool = True,
    ) -> None:
        timestamp = _timestamp(timestamp_ms)
        with self._lock:
            runtime = self._channels.get(channel_id)
            if runtime is None or not runtime.inflight_job_id:
                raise ValueError("channel has no inflight VLM job")
            if job_id is not None and job_id != runtime.inflight_job_id:
                raise ValueError("job_id does not match the inflight VLM job")
            if (
                runtime.inflight_since_ms is not None
                and timestamp < runtime.inflight_since_ms
            ):
                raise ValueError("completion cannot predate dispatch")
            runtime.inflight_job_id = ""
            runtime.inflight_since_ms = None
            runtime.last_completed_ms = timestamp
            runtime.last_completion_success = bool(success)

    def status(self, channel_id: str | None = None) -> dict[str, object]:
        with self._lock:
            if channel_id is not None:
                runtime = self._channels.get(channel_id)
                return (
                    self._channel_status(channel_id, runtime)
                    if runtime is not None
                    else {"channel_id": channel_id, "known": False}
                )
            return {
                "channel_count": len(self._channels),
                "max_channels": self.config.max_channels,
                "channels": [
                    self._channel_status(key, runtime)
                    for key, runtime in sorted(self._channels.items())
                ],
            }

    def _runtime(self, channel_id: str) -> _ChannelRuntime:
        if not channel_id:
            raise ValueError("channel_id must not be empty")
        runtime = self._channels.get(channel_id)
        if runtime is None:
            if len(self._channels) >= self.config.max_channels:
                raise ValueError(
                    f"channel capacity exceeded ({self.config.max_channels})"
                )
            runtime = _ChannelRuntime()
            self._channels[channel_id] = runtime
        return runtime

    def _trim(self, runtime: _ChannelRuntime, now_ms: int) -> None:
        sample_floor = now_ms - self.config.cv_retention_ms
        snapshot_floor = now_ms - self.config.snapshot_retention_ms
        runtime.samples = [
            sample for sample in runtime.samples if sample.timestamp_ms >= sample_floor
        ]
        runtime.snapshots = [
            snapshot
            for snapshot in runtime.snapshots
            if snapshot.timestamp_ms >= snapshot_floor
        ]
        runtime.apex_markers = [
            marker
            for marker in runtime.apex_markers
            if marker.timestamp_ms >= sample_floor
        ]

    def _latest_probe_signal(
        self, snapshot: EmbeddingSnapshot | None
    ) -> tuple[float, float]:
        if snapshot is None or not snapshot.probe_scores:
            return 0.0, 0.0
        return (
            min(1.0, max(0.0, max(score.positive for score in snapshot.probe_scores))),
            min(1.0, max(0.0, max(score.margin for score in snapshot.probe_scores))),
        )

    def _parse_probe_score(
        self, value: ProbeScore | Mapping[str, object]
    ) -> ProbeScore:
        if isinstance(value, ProbeScore):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("probe_scores must contain ProbeScore or mappings")
        return ProbeScore(
            probe_id=str(value["probe_id"]),
            positive=float(value["positive"]),
            negative=float(value["negative"]),
            margin=float(value["margin"]),
            probe_version=str(value.get("probe_version", "")),
        )

    def _should_dispatch(
        self, channel_id: str, runtime: _ChannelRuntime, now: int
    ) -> DispatchReadiness:
        decision = runtime.decision
        assert decision is not None
        state = decision.state
        if now < state.vector.timestamp_ms:
            raise ValueError("now_ms cannot predate the channel state")
        if runtime.inflight_job_id:
            return DispatchReadiness(
                channel_id,
                False,
                f"inflight:{runtime.inflight_job_id}",
                state.vector.timestamp_ms,
                self.config.default_job_units,
                decision,
            )
        if state.mode is AttentionMode.DEGRADED:
            return DispatchReadiness(
                channel_id,
                False,
                "degraded_source",
                state.vector.timestamp_ms,
                self.config.default_job_units,
                decision,
            )
        if not runtime.snapshots:
            return DispatchReadiness(
                channel_id,
                False,
                "no_saved_embedding_snapshot",
                state.vector.timestamp_ms,
                self.config.default_job_units,
                decision,
            )
        if (
            state.last_vlm_ms is not None
            and now - state.last_vlm_ms < self.config.minimum_dispatch_gap_ms
        ):
            return DispatchReadiness(
                channel_id,
                False,
                "minimum_dispatch_gap",
                state.vector.timestamp_ms,
                self.config.default_job_units,
                decision,
            )

        trigger = (
            runtime.apex_markers[-1].timestamp_ms
            if runtime.apex_markers
            else state.vector.timestamp_ms
        )
        if state.mode is AttentionMode.BURST:
            ready_at = trigger + self.config.burst_postroll_wait_ms
            if now < ready_at:
                return DispatchReadiness(
                    channel_id,
                    False,
                    f"collecting_postroll_until:{ready_at}",
                    trigger,
                    self.config.default_job_units,
                    decision,
                )
            return DispatchReadiness(
                channel_id,
                True,
                "burst_ready",
                trigger,
                self.config.default_job_units,
                decision,
            )
        if state.coverage_debt >= 1.0:
            return DispatchReadiness(
                channel_id,
                True,
                "coverage_due",
                trigger,
                self.config.default_job_units,
                decision,
            )
        if (
            state.mode is AttentionMode.ACTIVE
            and state.priority >= self.config.active_priority_dispatch
        ):
            return DispatchReadiness(
                channel_id,
                True,
                "active_priority",
                trigger,
                self.config.default_job_units,
                decision,
            )
        return DispatchReadiness(
            channel_id,
            False,
            f"not_due:{state.mode.value}",
            trigger,
            self.config.default_job_units,
            decision,
        )

    def _episode_id(self, channel_id: str, trigger_timestamp_ms: int) -> str:
        return f"{channel_id}:attention:{trigger_timestamp_ms}"

    def _channel_status(
        self, channel_id: str, runtime: _ChannelRuntime
    ) -> dict[str, object]:
        intervals = aggregate_cv_intervals(
            channel_id,
            runtime.samples,
            runtime.snapshots,
            runtime.apex_markers,
            self.config.aggregation,
        )
        return {
            "apex_markers": len(runtime.apex_markers),
            "capture_mode": runtime.capture_mode,
            "channel_id": channel_id,
            "cv_intervals": len(intervals),
            "cv_samples": len(runtime.samples),
            "decision": (
                json.loads(runtime.decision.printable())
                if runtime.decision is not None
                else None
            ),
            "inflight_job_id": runtime.inflight_job_id or None,
            "known": True,
            "last_activity_x": round(runtime.last_activity_x, 6),
            "last_completed_ms": runtime.last_completed_ms,
            "last_completion_success": runtime.last_completion_success,
            "snapshots": len(runtime.snapshots),
        }


__all__ = [
    "AggregationConfig",
    "AllocationEntry",
    "ApexLink",
    "ApexMarker",
    "AttentionAllocation",
    "AttentionCandidate",
    "AttentionCoordinator",
    "AttentionDecision",
    "AttentionEpisode",
    "AttentionMode",
    "AttentionPlan",
    "AttentionPolicyConfig",
    "AttentionState",
    "AttentionVector",
    "AttentionWeights",
    "CostAllocationEntry",
    "CostAwareAllocation",
    "CostBudgetConfig",
    "CostBudgetState",
    "CostedAttentionCandidate",
    "CoordinatorConfig",
    "CvSample",
    "DispatchReadiness",
    "EmbeddingSnapshot",
    "EpisodeConfig",
    "EpisodeFrame",
    "EpisodeRole",
    "FrameSelectorConfig",
    "GlobalBudgetConfig",
    "HomeostaticAttentionPolicy",
    "InferenceCost",
    "ModeProfile",
    "ModelFrameCandidate",
    "ModelFrameSelection",
    "MotionInterval",
    "MotionKind",
    "PORT_EIGHT_CHANNEL_PRESET",
    "PlannedDispatch",
    "PortAttentionPreset",
    "ProbeScore",
    "SelectedModelFrame",
    "aggregate_cv_intervals",
    "allocate_cost_aware_attention",
    "allocate_global_attention",
    "build_attention_episode",
    "build_control_episode",
    "initial_cost_budget_state",
    "profile_for_mode",
    "replenish_cost_budget",
    "select_model_frames",
]
