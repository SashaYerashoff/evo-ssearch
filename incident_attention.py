"""Pure attention allocation and prompt-budget planning for active incidents.

This module deliberately has no database, capture, model, or HTTP dependencies.
It answers two bounded questions:

* which active incidents are foreground, hot, or parked; and
* which semantic representation of each incident fits beside protected prompt
  blocks without truncating those blocks or cutting arbitrary text.

Parking is an attention decision only.  It never resolves, closes, or otherwise
mutates an incident.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


ALERT_CONTRACT_BLOCK = "alert_contract"
BATCH_STATE_BLOCK = "batch_state"


class AttentionTier(str, Enum):
    FOREGROUND = "foreground"
    HOT = "hot"
    PARKED = "parked"


class ContextCompactionTier(str, Enum):
    FULL = "full"
    DIGEST = "digest"
    PARKED = "parked"


class PromptBudgetError(ValueError):
    """The protected prompt envelope cannot fit without unsafe truncation."""


def estimate_text_tokens(value: str) -> int:
    """Return a deterministic conservative token estimate.

    The planner accepts an injected tokenizer-specific estimator in production.
    This dependency-free default counts at most three UTF-8 bytes per estimated
    token, which is intentionally more conservative than the usual English-text
    four-characters heuristic and remains measurable in unit tests.
    """

    data = str(value or "").encode("utf-8", errors="replace")
    return 0 if not data else int(math.ceil(len(data) / 3.0))


@dataclass(frozen=True)
class IncidentAttentionPolicy:
    """Deterministic 2/4/8 attention policy."""

    normal_foreground_limit: int = 2
    hard_foreground_limit: int = 4
    hot_unresolved_limit: int = 8
    max_tracked_incidents: int = 64

    def __post_init__(self) -> None:
        values = (
            self.normal_foreground_limit,
            self.hard_foreground_limit,
            self.hot_unresolved_limit,
            self.max_tracked_incidents,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError("incident attention limits must be positive")
        if self.normal_foreground_limit > self.hard_foreground_limit:
            raise ValueError("normal foreground limit must not exceed hard limit")
        if self.hard_foreground_limit > self.hot_unresolved_limit:
            raise ValueError("hard foreground limit must not exceed hot limit")
        if self.hot_unresolved_limit > self.max_tracked_incidents:
            raise ValueError("hot limit must not exceed tracked incident limit")


@dataclass(frozen=True)
class IncidentAttentionCandidate:
    incident_id: str
    level: str = "follow"
    context: str = ""
    operator_selected: bool = False
    unresolved: bool = True
    incumbent_tier: AttentionTier | str | None = None
    resolution_debt: int = 0
    updated_at_ms: int = 0
    expires_at_ms: int = 0

    def __post_init__(self) -> None:
        incident_id = str(self.incident_id or "").strip()
        if not incident_id:
            raise ValueError("incident_id must not be empty")
        level = str(self.level or "follow").strip().lower()
        if level not in {"follow", "critical"}:
            raise ValueError("level must be follow or critical")
        incumbent = self.incumbent_tier
        if incumbent is not None and not isinstance(incumbent, AttentionTier):
            try:
                incumbent = AttentionTier(str(incumbent))
            except ValueError as exc:
                raise ValueError("incumbent_tier must be foreground, hot, or parked") from exc
        debt = int(self.resolution_debt)
        if debt < 0:
            raise ValueError("resolution_debt must be non-negative")
        object.__setattr__(self, "incident_id", incident_id)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "context", str(self.context or ""))
        object.__setattr__(self, "incumbent_tier", incumbent)
        object.__setattr__(self, "resolution_debt", debt)
        object.__setattr__(self, "updated_at_ms", max(0, int(self.updated_at_ms)))
        object.__setattr__(self, "expires_at_ms", max(0, int(self.expires_at_ms)))

    @property
    def hard_priority(self) -> bool:
        return self.level == "critical" or bool(self.operator_selected)


@dataclass(frozen=True)
class IncidentRankDecision:
    incident_id: str
    rank: int
    tier: AttentionTier
    level: str
    operator_selected: bool
    unresolved: bool
    incumbent_tier: AttentionTier | None
    resolution_debt: int
    reasons: tuple[str, ...]
    resolution_inferred: bool = False


@dataclass(frozen=True)
class IncidentAttentionAllocation:
    effective_level: str
    foreground_incident_ids: tuple[str, ...]
    hot_incident_ids: tuple[str, ...]
    parked_incident_ids: tuple[str, ...]
    all_incident_ids: tuple[str, ...]
    decisions: tuple[IncidentRankDecision, ...]
    foreground_limit: int
    hot_limit: int

    def tier_for(self, incident_id: str) -> AttentionTier:
        normalized = str(incident_id or "").strip()
        for decision in self.decisions:
            if decision.incident_id == normalized:
                return decision.tier
        raise KeyError(normalized)


def _candidate_rank_key(candidate: IncidentAttentionCandidate) -> tuple[Any, ...]:
    incumbent_rank = {
        AttentionTier.FOREGROUND: 2,
        AttentionTier.HOT: 1,
        AttentionTier.PARKED: 0,
        None: 0,
    }[candidate.incumbent_tier]
    return (
        -int(candidate.level == "critical"),
        -int(candidate.operator_selected),
        -int(candidate.unresolved),
        -incumbent_rank,
        -int(candidate.resolution_debt),
        -int(candidate.updated_at_ms),
        candidate.incident_id,
    )


def allocate_incident_attention(
    candidates: Sequence[IncidentAttentionCandidate],
    policy: IncidentAttentionPolicy | None = None,
) -> IncidentAttentionAllocation:
    """Rank candidates and partition them without changing lifecycle state."""

    selected_policy = policy or IncidentAttentionPolicy()
    normalized = tuple(candidates)
    if len(normalized) > selected_policy.max_tracked_incidents:
        raise ValueError(
            "incident candidates exceed max_tracked_incidents "
            f"({selected_policy.max_tracked_incidents})"
        )
    ids = [candidate.incident_id for candidate in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("incident candidates must have unique incident ids")

    ranked = tuple(sorted(normalized, key=_candidate_rank_key))
    hard_present = any(candidate.hard_priority for candidate in ranked)
    foreground_limit = (
        selected_policy.hard_foreground_limit
        if hard_present
        else selected_policy.normal_foreground_limit
    )
    hot_candidates = tuple(
        candidate
        for candidate in ranked
        if candidate.unresolved or candidate.hard_priority
    )[: selected_policy.hot_unresolved_limit]
    hot_ids = tuple(candidate.incident_id for candidate in hot_candidates)
    hot_id_set = set(hot_ids)
    foreground_ids = hot_ids[:foreground_limit]
    parked_ids = tuple(
        candidate.incident_id
        for candidate in ranked
        if candidate.incident_id not in hot_id_set
    )

    decisions: list[IncidentRankDecision] = []
    for rank, candidate in enumerate(ranked, start=1):
        if candidate.incident_id in foreground_ids:
            tier = AttentionTier.FOREGROUND
        elif candidate.incident_id in hot_ids:
            tier = AttentionTier.HOT
        else:
            tier = AttentionTier.PARKED
        reasons: list[str] = []
        if candidate.level == "critical":
            reasons.append("critical")
        if candidate.operator_selected:
            reasons.append("operator_selected")
        if candidate.unresolved:
            reasons.append("unresolved")
        if candidate.incumbent_tier is not None:
            reasons.append(f"incumbent_{candidate.incumbent_tier.value}")
        if candidate.resolution_debt:
            reasons.append(f"resolution_debt_{candidate.resolution_debt}")
        if tier is AttentionTier.PARKED:
            reasons.append("attention_capacity_only")
        decisions.append(
            IncidentRankDecision(
                incident_id=candidate.incident_id,
                rank=rank,
                tier=tier,
                level=candidate.level,
                operator_selected=bool(candidate.operator_selected),
                unresolved=bool(candidate.unresolved),
                incumbent_tier=candidate.incumbent_tier,
                resolution_debt=candidate.resolution_debt,
                reasons=tuple(reasons),
            )
        )

    return IncidentAttentionAllocation(
        effective_level=(
            "critical"
            if any(candidate.level == "critical" for candidate in ranked)
            else "follow"
        ),
        foreground_incident_ids=foreground_ids,
        hot_incident_ids=hot_ids,
        parked_incident_ids=parked_ids,
        all_incident_ids=tuple(candidate.incident_id for candidate in ranked),
        decisions=tuple(decisions),
        foreground_limit=foreground_limit,
        hot_limit=selected_policy.hot_unresolved_limit,
    )


@dataclass(frozen=True)
class PromptEnvelopeBudget:
    context_window_tokens: int = 32_768
    max_text_tokens: int = 16_384
    max_vision_tokens: int = 4_096
    max_output_tokens: int = 2_048
    max_incident_tokens: int = 4_096

    def __post_init__(self) -> None:
        values = (
            self.context_window_tokens,
            self.max_text_tokens,
            self.max_vision_tokens,
            self.max_output_tokens,
            self.max_incident_tokens,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError("prompt budgets must be non-negative")
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        if (
            self.max_text_tokens
            + self.max_vision_tokens
            + self.max_output_tokens
            > self.context_window_tokens
        ):
            raise ValueError("text, vision, and output budgets exceed context window")
        if self.max_incident_tokens > self.max_text_tokens:
            raise ValueError("incident text budget must not exceed text budget")


@dataclass(frozen=True)
class ProtectedPromptBlock:
    name: str
    text: str

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("protected prompt block name must not be empty")
        text = str(self.text or "")
        if not text:
            raise ValueError("protected prompt block text must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "text", text)


@dataclass(frozen=True)
class PlannedProtectedBlock:
    name: str
    text: str
    token_estimate: int
    sha256: str


@dataclass(frozen=True)
class PlannedIncidentContext:
    incident_id: str
    attention_tier: AttentionTier
    compaction_tier: ContextCompactionTier
    text: str
    token_estimate: int
    source_sha256: str


@dataclass(frozen=True)
class IncidentPromptEnvelopePlan:
    allocation: IncidentAttentionAllocation
    protected_blocks: tuple[PlannedProtectedBlock, ...]
    incident_contexts: tuple[PlannedIncidentContext, ...]
    omitted_incident_ids: tuple[str, ...]
    text_tokens_used: int
    incident_tokens_used: int
    vision_tokens: int
    output_tokens: int
    budget: PromptEnvelopeBudget

    @property
    def context_strings(self) -> tuple[str, ...]:
        return tuple(item.text for item in self.incident_contexts)


_FULL_CONTEXT_KEYS = (
    "title",
    "summary",
    "possible_start_ms",
    "observed_start_ms",
    "observed_end_ms",
    "timeline",
    "uncertainties",
)
_TIMELINE_KEYS = (
    "timestamp_ms",
    "start_ms",
    "end_ms",
    "semantic_key",
    "label",
    "description",
    "confidence",
    "state",
)


def _source_sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def _decoded_context(value: str) -> Mapping[str, Any] | None:
    try:
        decoded = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _full_context_payload(
    candidate: IncidentAttentionCandidate,
    *,
    attention_tier: AttentionTier | None = None,
) -> dict[str, Any]:
    source = _decoded_context(candidate.context)
    payload: dict[str, Any] = {
        "incident_id": candidate.incident_id,
        "context_compaction": ContextCompactionTier.FULL.value,
        "unresolved": bool(candidate.unresolved),
        "resolution_inferred": False,
    }
    if attention_tier is not None:
        payload["attention_tier"] = attention_tier.value
    if source is None:
        if candidate.context:
            payload["context"] = candidate.context
        return payload
    for key in _FULL_CONTEXT_KEYS:
        value = source.get(key)
        if value in (None, "", [], {}):
            continue
        if key == "timeline" and isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            payload[key] = [
                {
                    field: row.get(field)
                    for field in _TIMELINE_KEYS
                    if row.get(field) is not None
                }
                for row in list(value)[-4:]
                if isinstance(row, Mapping)
            ]
        elif key == "uncertainties" and isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            payload[key] = list(value)[:2]
        else:
            payload[key] = value
    return payload


def _digest_context_payload(
    candidate: IncidentAttentionCandidate,
    *,
    attention_tier: AttentionTier | None = None,
) -> dict[str, Any]:
    source = _decoded_context(candidate.context) or {}
    timeline = source.get("timeline")
    labels: list[str] = []
    if isinstance(timeline, Sequence) and not isinstance(
        timeline, (str, bytes, bytearray)
    ):
        for row in list(timeline)[-2:]:
            if not isinstance(row, Mapping):
                continue
            label = str(row.get("label") or row.get("semantic_key") or "").strip()
            if label:
                labels.append(label)
    payload: dict[str, Any] = {
        "incident_id": candidate.incident_id,
        "context_compaction": ContextCompactionTier.DIGEST.value,
        "unresolved": bool(candidate.unresolved),
        "resolution_inferred": False,
        "context_sha256": _source_sha256(candidate.context),
    }
    if attention_tier is not None:
        payload["attention_tier"] = attention_tier.value
    for key in (
        "title",
        "possible_start_ms",
        "observed_start_ms",
        "observed_end_ms",
    ):
        if source.get(key) not in (None, ""):
            payload[key] = source.get(key)
    if labels:
        payload["recent_timeline_labels"] = labels
    uncertainties = source.get("uncertainties")
    if isinstance(uncertainties, Sequence) and not isinstance(
        uncertainties, (str, bytes, bytearray)
    ):
        payload["uncertainty_count"] = len(uncertainties)
    return payload


def _parked_context_payload(
    candidate: IncidentAttentionCandidate,
    *,
    attention_tier: AttentionTier | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "incident_id": candidate.incident_id,
        "context_compaction": ContextCompactionTier.PARKED.value,
        "unresolved": bool(candidate.unresolved),
        "resolution_inferred": False,
        "context_sha256": _source_sha256(candidate.context),
    }
    if attention_tier is not None:
        payload["attention_tier"] = attention_tier.value
    return payload


def _render_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def compact_incident_context(
    value: str | None,
    *,
    max_tokens: int = 800,
    token_estimator: Callable[[str], int] = estimate_text_tokens,
) -> str:
    """Bound a stored lease context by semantic tiers, never by slicing text."""

    text = str(value or "").strip()
    if not text:
        return ""
    budget = max(1, int(max_tokens))
    looks_structured = text.startswith("{") or text.startswith("[")
    if token_estimator(text) <= budget and (
        not looks_structured or _decoded_context(text) is not None
    ):
        return text
    candidate = IncidentAttentionCandidate(incident_id="lease-context", context=text)
    for payload in (
        _full_context_payload(candidate),
        _digest_context_payload(candidate),
        _parked_context_payload(candidate),
    ):
        rendered = _render_payload(payload)
        if token_estimator(rendered) <= budget:
            return rendered
    raise PromptBudgetError("incident lease context cannot fit semantic stub")


class IncidentPromptEnvelopePlanner:
    """Plan an atomic protected envelope plus tiered incident context."""

    def __init__(
        self,
        policy: IncidentAttentionPolicy | None = None,
        *,
        token_estimator: Callable[[str], int] = estimate_text_tokens,
        required_protected_blocks: Sequence[str] = (
            ALERT_CONTRACT_BLOCK,
            BATCH_STATE_BLOCK,
        ),
    ) -> None:
        self.policy = policy or IncidentAttentionPolicy()
        self.token_estimator = token_estimator
        self.required_protected_blocks = tuple(
            str(name or "").strip() for name in required_protected_blocks
        )

    def plan(
        self,
        candidates: Sequence[IncidentAttentionCandidate],
        *,
        protected_blocks: Sequence[ProtectedPromptBlock],
        budget: PromptEnvelopeBudget | None = None,
        vision_tokens: int = 0,
        output_tokens: int = 0,
    ) -> IncidentPromptEnvelopePlan:
        selected_budget = budget or PromptEnvelopeBudget()
        requested_vision = max(0, int(vision_tokens))
        requested_output = max(0, int(output_tokens))
        if requested_vision > selected_budget.max_vision_tokens:
            raise PromptBudgetError("requested vision tokens exceed vision budget")
        if requested_output > selected_budget.max_output_tokens:
            raise PromptBudgetError("requested output tokens exceed output budget")

        raw_blocks = tuple(protected_blocks)
        names = [block.name for block in raw_blocks]
        if len(names) != len(set(names)):
            raise PromptBudgetError("protected prompt block names must be unique")
        missing = [
            name for name in self.required_protected_blocks if name not in set(names)
        ]
        if missing:
            raise PromptBudgetError(
                "missing protected prompt blocks: " + ", ".join(missing)
            )
        planned_blocks = tuple(
            PlannedProtectedBlock(
                name=block.name,
                text=block.text,
                token_estimate=self.token_estimator(block.text),
                sha256=_source_sha256(block.text),
            )
            for block in raw_blocks
        )
        protected_tokens = sum(block.token_estimate for block in planned_blocks)
        if protected_tokens > selected_budget.max_text_tokens:
            raise PromptBudgetError(
                "protected prompt blocks exceed text budget; refusing truncation"
            )

        normalized_candidates = tuple(candidates)
        allocation = allocate_incident_attention(normalized_candidates, self.policy)
        by_id = {
            candidate.incident_id: candidate for candidate in normalized_candidates
        }
        incident_capacity = min(
            selected_budget.max_incident_tokens,
            selected_budget.max_text_tokens - protected_tokens,
        )
        variants_by_id: dict[str, tuple[PlannedIncidentContext, ...]] = {}

        # The scheduler may keep eight unresolved incidents hot, but the model
        # prompt only receives the first hard foreground window (normally four).
        # Incidents 5-8 remain measurable scheduler state and are not allowed to
        # consume the visual reasoning envelope.
        prompt_incident_ids = allocation.all_incident_ids[
            : self.policy.hard_foreground_limit
        ]

        for incident_id in prompt_incident_ids:
            candidate = by_id[incident_id]
            attention_tier = allocation.tier_for(incident_id)
            variants: list[tuple[ContextCompactionTier, dict[str, Any]]]
            if attention_tier is AttentionTier.FOREGROUND:
                variants = [
                    (
                        ContextCompactionTier.FULL,
                        _full_context_payload(
                            candidate,
                            attention_tier=attention_tier,
                        ),
                    ),
                    (
                        ContextCompactionTier.DIGEST,
                        _digest_context_payload(
                            candidate,
                            attention_tier=attention_tier,
                        ),
                    ),
                    (
                        ContextCompactionTier.PARKED,
                        _parked_context_payload(
                            candidate,
                            attention_tier=attention_tier,
                        ),
                    ),
                ]
            elif attention_tier is AttentionTier.HOT:
                variants = [
                    (
                        ContextCompactionTier.DIGEST,
                        _digest_context_payload(
                            candidate,
                            attention_tier=attention_tier,
                        ),
                    ),
                    (
                        ContextCompactionTier.PARKED,
                        _parked_context_payload(
                            candidate,
                            attention_tier=attention_tier,
                        ),
                    ),
                ]
            else:
                variants = [
                    (
                        ContextCompactionTier.PARKED,
                        _parked_context_payload(
                            candidate,
                            attention_tier=attention_tier,
                        ),
                    ),
                ]

            rendered_variants: list[PlannedIncidentContext] = []
            for compaction_tier, payload in variants:
                rendered = _render_payload(payload)
                rendered_variants.append(
                    PlannedIncidentContext(
                        incident_id=incident_id,
                        attention_tier=attention_tier,
                        compaction_tier=compaction_tier,
                        text=rendered,
                        token_estimate=self.token_estimator(rendered),
                        source_sha256=_source_sha256(candidate.context),
                    )
                )
            variants_by_id[incident_id] = tuple(rendered_variants)

        # Breadth comes before detail: reserve an atomic semantic stub for each
        # incident in priority order, then spend the remainder upgrading those
        # representations.  One rich incident can therefore never silently
        # crowd a parallel foreground incident out of the prompt.
        admitted_by_id: dict[str, PlannedIncidentContext] = {}
        omitted_ids: list[str] = list(
            allocation.all_incident_ids[len(prompt_incident_ids) :]
        )
        incident_used = 0
        for index, incident_id in enumerate(prompt_incident_ids):
            stub = variants_by_id[incident_id][-1]
            if incident_used + stub.token_estimate > incident_capacity:
                omitted_ids[0:0] = list(prompt_incident_ids[index:])
                break
            admitted_by_id[incident_id] = stub
            incident_used += stub.token_estimate

        for incident_id in prompt_incident_ids:
            current = admitted_by_id.get(incident_id)
            if current is None:
                continue
            for richer in variants_by_id[incident_id][:-1]:
                delta = richer.token_estimate - current.token_estimate
                if incident_used + delta > incident_capacity:
                    continue
                admitted_by_id[incident_id] = richer
                incident_used += delta
                break

        planned_contexts = tuple(
            admitted_by_id[incident_id]
            for incident_id in prompt_incident_ids
            if incident_id in admitted_by_id
        )

        return IncidentPromptEnvelopePlan(
            allocation=allocation,
            protected_blocks=planned_blocks,
            incident_contexts=planned_contexts,
            omitted_incident_ids=tuple(omitted_ids),
            text_tokens_used=protected_tokens + incident_used,
            incident_tokens_used=incident_used,
            vision_tokens=requested_vision,
            output_tokens=requested_output,
            budget=selected_budget,
        )


__all__ = [
    "ALERT_CONTRACT_BLOCK",
    "AttentionTier",
    "BATCH_STATE_BLOCK",
    "ContextCompactionTier",
    "IncidentAttentionAllocation",
    "IncidentAttentionCandidate",
    "IncidentAttentionPolicy",
    "IncidentPromptEnvelopePlan",
    "IncidentPromptEnvelopePlanner",
    "IncidentRankDecision",
    "PlannedIncidentContext",
    "PlannedProtectedBlock",
    "PromptBudgetError",
    "PromptEnvelopeBudget",
    "ProtectedPromptBlock",
    "allocate_incident_attention",
    "compact_incident_context",
    "estimate_text_tokens",
]
