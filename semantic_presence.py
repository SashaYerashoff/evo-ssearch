"""Bounded semantic-presence homeostasis over existing pooled embeddings.

This module deliberately does not detect or count objects.  It turns a small,
bounded text bank into per-channel score traces and adaptive baselines.  Patch
localization is a separate experiment; these values remain attention signals,
not visual proof.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Mapping, Sequence, Tuple


SEMANTIC_PRESENCE_PROMPTS: Mapping[str, Tuple[str, ...]] = {
    "person": ("a person", "people"),
    "vehicle": ("a road vehicle", "a car, bus, or truck"),
    "animal": ("an animal", "a cat or dog"),
    "smoke": ("visible smoke",),
    "fire": ("visible flames or fire",),
}


def normalize_presence_labels(
    values: Iterable[object],
    *,
    maximum: int = 10,
) -> Tuple[str, ...]:
    labels: List[str] = []
    for value in values:
        label = " ".join(str(value or "").strip().lower().split())
        if not label or len(label) > 48 or label in labels:
            continue
        labels.append(label)
        if len(labels) >= max(1, int(maximum)):
            break
    return tuple(labels)


@dataclass
class _PresenceState:
    score: float
    baseline: float
    deviation: float
    samples: int
    timestamp_ms: int
    trend: str
    history: Deque[Dict[str, Any]]


class SemanticPresenceTracker:
    """Maintain small, resettable per-channel semantic score baselines."""

    def __init__(
        self,
        labels: Sequence[str],
        *,
        maximum_classes: int = 10,
        history_size: int = 60,
        warmup_samples: int = 30,
        alpha: float = 0.025,
        noise_floor: float = 0.01,
        prompts: Mapping[str, Sequence[str]] = SEMANTIC_PRESENCE_PROMPTS,
    ) -> None:
        self.maximum_classes = max(1, min(16, int(maximum_classes)))
        self.history_size = max(10, min(300, int(history_size)))
        self.warmup_samples = max(3, min(600, int(warmup_samples)))
        self.alpha = max(0.001, min(0.5, float(alpha)))
        self.noise_floor = max(0.0001, min(0.25, float(noise_floor)))
        self.core_labels = normalize_presence_labels(
            labels,
            maximum=self.maximum_classes,
        )
        self._prompts = {
            str(label).strip().lower(): tuple(
                prompt
                for prompt in (" ".join(str(item or "").split()) for item in items)
                if prompt
            )
            for label, items in prompts.items()
        }
        self._channel_labels: Dict[int, Tuple[str, ...]] = {}
        self._states: Dict[Tuple[int, str], _PresenceState] = {}
        self._last_error: Dict[int, str] = {}
        self._lock = threading.RLock()

    def labels_for_channel(self, channel_id: int) -> Tuple[str, ...]:
        with self._lock:
            dynamic = self._channel_labels.get(int(channel_id), ())
        return normalize_presence_labels(
            (*self.core_labels, *dynamic),
            maximum=self.maximum_classes,
        )

    def set_channel_labels(self, channel_id: int, labels: Sequence[str]) -> Tuple[str, ...]:
        """Install bounded candidate labels; intended for future grounded L1 proposals."""

        dynamic = tuple(
            label
            for label in normalize_presence_labels(
                labels,
                maximum=self.maximum_classes,
            )
            if label not in self.core_labels
        )
        with self._lock:
            self._channel_labels[int(channel_id)] = dynamic
        return self.labels_for_channel(channel_id)

    def prompt_plan(self, channel_id: int) -> List[Tuple[str, Tuple[str, ...]]]:
        plan: List[Tuple[str, Tuple[str, ...]]] = []
        for label in self.labels_for_channel(channel_id):
            label_prompts = self._prompts.get(label) or (f"a visible {label}",)
            plan.append((label, tuple(label_prompts)))
        return plan

    def note_error(self, channel_id: int, error: object) -> None:
        message = " ".join(str(error or "semantic presence unavailable").split())[:240]
        with self._lock:
            self._last_error[int(channel_id)] = message

    def update(
        self,
        channel_id: int,
        timestamp_ms: int,
        scores: Mapping[str, float],
    ) -> Dict[str, Any]:
        channel = int(channel_id)
        timestamp = max(0, int(timestamp_ms))
        with self._lock:
            self._last_error.pop(channel, None)
            for label in self.labels_for_channel(channel):
                raw_score = scores.get(label)
                try:
                    score = float(raw_score) if raw_score is not None else math.nan
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(score):
                    continue
                key = (channel, label)
                previous = self._states.get(key)
                if previous is None:
                    baseline = score
                    deviation = 0.0
                    samples = 1
                    trend = "warming_up"
                    history: Deque[Dict[str, Any]] = deque(maxlen=self.history_size)
                else:
                    scale = max(previous.deviation, self.noise_floor)
                    delta = score - previous.baseline
                    samples = previous.samples + 1
                    if previous.samples < self.warmup_samples:
                        trend = "warming_up"
                    elif delta >= 3.0 * scale:
                        trend = "above_baseline"
                    elif delta <= -3.0 * scale:
                        trend = "below_baseline"
                    else:
                        trend = "routine"
                    clamped = min(
                        previous.baseline + 3.0 * scale,
                        max(previous.baseline - 3.0 * scale, score),
                    )
                    baseline = (
                        (1.0 - self.alpha) * previous.baseline
                        + self.alpha * clamped
                    )
                    deviation = (
                        (1.0 - self.alpha) * previous.deviation
                        + self.alpha * abs(clamped - baseline)
                    )
                    history = previous.history
                history.append(
                    {
                        "timestamp_ms": timestamp,
                        "score": round(score, 6),
                        "baseline": round(baseline, 6),
                    }
                )
                self._states[key] = _PresenceState(
                    score=score,
                    baseline=baseline,
                    deviation=deviation,
                    samples=samples,
                    timestamp_ms=timestamp,
                    trend=trend,
                    history=history,
                )
        return self.status(channel, now_ms=timestamp)

    def status(self, channel_id: int, *, now_ms: int | None = None) -> Dict[str, Any]:
        channel = int(channel_id)
        current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        classes: List[Dict[str, Any]] = []
        with self._lock:
            labels = self.labels_for_channel(channel)
            for label in labels:
                state = self._states.get((channel, label))
                prompts = list(dict.fromkeys(self._prompts.get(label) or (f"a visible {label}",)))
                if state is None:
                    classes.append(
                        {
                            "key": label,
                            "label": label,
                            "prompts": prompts,
                            "state": "warming_up",
                            "warmup": True,
                            "samples": 0,
                            "history": [],
                        }
                    )
                    continue
                delta = state.score - state.baseline
                scale = max(state.deviation, self.noise_floor)
                classes.append(
                    {
                        "key": label,
                        "label": label,
                        "prompts": prompts,
                        "score": round(state.score, 6),
                        "baseline": round(state.baseline, 6),
                        "deviation": round(state.deviation, 6),
                        "delta": round(delta, 6),
                        "z": round(delta / scale, 3),
                        "state": state.trend,
                        "warmup": state.samples < self.warmup_samples,
                        "samples": state.samples,
                        "timestamp_ms": state.timestamp_ms,
                        "history": list(state.history),
                    }
                )
            timestamps = [
                int(item.get("timestamp_ms") or 0)
                for item in classes
                if int(item.get("timestamp_ms") or 0) > 0
            ]
            error = self._last_error.get(channel)
        latest = max(timestamps, default=0)
        state = (
            "degraded"
            if error
            else "warming_up"
            if not latest or any(bool(item.get("warmup")) for item in classes)
            else "ready"
        )
        return {
            "enabled": True,
            "shadow": True,
            "state": state,
            "channel_id": channel,
            "timestamp_ms": latest or None,
            "age_ms": max(0, current_ms - latest) if latest else None,
            "semantics": "pooled_embedding_attention_signal_not_object_detection",
            "classes": classes,
            "error": error,
        }

    def clear_channel(self, channel_id: int) -> None:
        channel = int(channel_id)
        with self._lock:
            self._states = {
                key: state
                for key, state in self._states.items()
                if key[0] != channel
            }
            self._last_error.pop(channel, None)

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
            self._last_error.clear()

