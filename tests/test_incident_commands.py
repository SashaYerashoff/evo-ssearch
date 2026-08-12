from __future__ import annotations

from types import SimpleNamespace

import pytest

from incident_commands import IncidentCommandService, incident_storage_record
from incident_store import IncidentRevisionConflict


INCIDENT_ID = "00000000-0000-0000-0000-000000000117"


def _record(*, revision: int = 1, state: str = "following"):
    return {
        "id": INCIDENT_ID,
        "revision": revision,
        "state": state,
        "perception_state": "observed",
        "risk_state": "unknown",
        "case_state": "open",
        "attention_state": "follow" if state == "following" else "inactive",
        "title": "Craft crossing the port gate",
        "channel_ids": [112],
        "timeline_refs": [],
        "evidence_refs": [],
        "qualia_refs": [],
        "coverage": {"status": "covered"},
        "report": {},
        "follow_policy": {"active": state == "following", "mode": "follow"},
    }


class _Store:
    def __init__(self):
        self.record = _record()
        self.conflict = False
        self.update_calls = []
        self.observations = []

    def get_incident(self, incident_id):
        return dict(self.record) if incident_id == INCIDENT_ID else None

    def update_incident(
        self,
        incident_id,
        *,
        expected_revision,
        changes,
        actor_id,
        transition=None,
    ):
        self.update_calls.append((incident_id, expected_revision, dict(changes), actor_id))
        if self.conflict:
            raise IncidentRevisionConflict(incident_id, expected_revision, expected_revision + 1)
        self.record.update(dict(changes))
        self.record["revision"] += 1
        return dict(self.record)

    def list_observations(self, incident_id, **_kwargs):
        assert incident_id == INCIDENT_ID
        return [dict(item) for item in self.observations], len(self.observations)

    def append_observation(self, observation, **_kwargs):
        self.observations.append(dict(observation))
        return dict(observation)


class _Runtime:
    def __init__(self):
        self.active = {INCIDENT_ID: object()}
        self.stop_calls = []
        self.start_context = None

    def start_incident_focus(self, incident_id, channel_ids, *, level, ttl_seconds, context=None):
        lease = SimpleNamespace(level=SimpleNamespace(value=level), channel_ids=tuple(channel_ids))
        self.active[incident_id] = lease
        self.start_context = context
        return lease

    def stop_incident_focus(self, incident_id):
        self.stop_calls.append(incident_id)
        return self.active.pop(incident_id, None) is not None


def _service(store, runtime):
    return IncidentCommandService(store, object(), object(), runtime, wall_clock_ms=lambda: 10_000)


def test_stop_revision_conflict_keeps_runtime_focus_lease():
    store = _Store()
    store.conflict = True
    runtime = _Runtime()

    with pytest.raises(IncidentRevisionConflict):
        _service(store, runtime).stop_follow(
            INCIDENT_ID,
            actor_id="operator-1",
            expected_revision=1,
        )

    assert INCIDENT_ID in runtime.active
    assert runtime.stop_calls == []


def test_stop_persists_before_removing_runtime_focus():
    store = _Store()
    runtime = _Runtime()

    updated, removed = _service(store, runtime).stop_follow(
        INCIDENT_ID,
        actor_id="operator-1",
        expected_revision=1,
    )

    assert removed is True
    assert runtime.stop_calls == [INCIDENT_ID]
    assert updated["revision"] == 2
    assert updated["state"] == "draft"
    assert updated["attention_state"] == "inactive"
    assert updated["follow_policy"]["active"] is False
    assert updated["follow_policy"]["stopped_at_ms"] == 10_000


def test_follow_uses_explicit_optimistic_revision():
    store = _Store()
    runtime = _Runtime()

    updated, lease = _service(store, runtime).follow(
        INCIDENT_ID,
        actor_id="operator-1",
        mode="critical",
        ttl_seconds=120,
        expected_revision=1,
    )

    assert updated["revision"] == 2
    assert updated["state"] == "following"
    assert updated["case_state"] == "open"
    assert updated["attention_state"] == "critical"
    assert updated["follow_policy"]["mode"] == "critical"
    assert updated["follow_policy"]["expires_at_ms"] == 130_000
    assert lease.level.value == "critical"
    assert "Craft crossing the port gate" in runtime.start_context


def test_operator_confirm_opens_case_without_inventing_risk_or_perception():
    store = _Store()
    store.record.update(
        {
            "state": "draft",
            "case_state": "candidate",
            "perception_state": "unknown",
            "risk_state": "unknown",
            "attention_state": "inactive",
            "follow_policy": {},
        }
    )
    runtime = _Runtime()

    updated = _service(store, runtime).review_incident(
        INCIDENT_ID,
        actor_id="operator-1",
        action="confirm",
        expected_revision=1,
        note="Grounded entry reviewed on the archive frames.",
    )

    assert updated["state"] == "reported"
    assert updated["case_state"] == "open"
    assert updated["perception_state"] == "unknown"
    assert updated["risk_state"] == "unknown"
    assert runtime.stop_calls == []
    observation = store.observations[-1]
    assert observation["source_kind"] == "operator_review"
    assert observation["payload"]["action"] == "confirm"
    assert observation["payload"]["previous"]["case_state"] == "candidate"
    assert observation["payload"]["current"]["case_state"] == "open"


def test_false_positive_stops_follow_and_reopen_is_explicit():
    store = _Store()
    runtime = _Runtime()
    service = _service(store, runtime)

    closed = service.review_incident(
        INCIDENT_ID,
        actor_id="operator-1",
        action="false_positive",
        expected_revision=1,
    )

    assert closed["state"] == "closed"
    assert closed["case_state"] == "false_positive"
    assert closed["risk_state"] == "resolved"
    assert closed["attention_state"] == "inactive"
    assert runtime.stop_calls == [INCIDENT_ID]
    assert any(item.get("source_kind") == "follow_completed" for item in store.observations)
    assert store.observations[-1]["payload"]["action"] == "false_positive"

    reopened = service.review_incident(
        INCIDENT_ID,
        actor_id="operator-1",
        action="reopen",
        expected_revision=closed["revision"],
    )
    assert reopened["state"] == "reported"
    assert reopened["case_state"] == "open"
    assert reopened["risk_state"] == "unknown"


def test_draft_digest_is_stable_and_content_bound():
    service = _service(_Store(), _Runtime())
    draft = {
        "title": "Person enters",
        "channel_ids": [112],
        "time_bounds": {"observed_start_ms": 1_000, "observed_end_ms": 2_000},
        "timeline": [{"timestamp_ms": 1_500, "label": "Person enters"}],
        "evidence": [{"kind": "detection", "detection_id": 41}],
        "coverage": {"status": "covered"},
    }

    first = service.draft_digest(draft)
    assert first == service.draft_digest(dict(reversed(list(draft.items()))))
    changed = {**draft, "title": "Vehicle enters"}
    assert service.draft_digest(changed) != first
    assert incident_storage_record(draft)["idempotency_key"].startswith(
        "incident-draft:"
    )


def test_store_draft_materializes_long_episode_and_candidate_series_without_merge():
    prior = {
        **_record(state="draft"),
        "id": "00000000-0000-0000-0000-000000000116",
        "possible_start_ms": 1_000,
        "observed_start_ms": 1_000,
        "observed_end_ms": 2_000,
        "timeline_refs": [
            {
                "timestamp_ms": 1_000,
                "semantic_key": "person_entry",
                "label": "Person enters the room",
                "source": "state_transition",
            }
        ],
    }

    class _TemporalStore:
        def __init__(self):
            self.episodes = []
            self.relations = []
            self.created = None

        def create_incident(self, record, **_kwargs):
            self.created = {
                "id": INCIDENT_ID,
                "revision": 1,
                **dict(record),
            }
            return dict(self.created)

        def append_episode(self, record, **_kwargs):
            self.episodes.append(dict(record))
            return dict(record)

        def append_relation(self, record, **_kwargs):
            self.relations.append(dict(record))
            return dict(record)

        def list_incidents(self, **_kwargs):
            return [dict(prior)], 1

    draft = {
        "title": "Person enters and remains",
        "channel_ids": [112],
        "time_bounds": {
            "possible_start_ms": 3_000,
            "observed_start_ms": 3_000,
            "observed_end_ms": 3_000 + 16 * 60 * 1_000,
            "possible_end_ms": 3_000 + 16 * 60 * 1_000,
        },
        "timeline": [
            {
                "timestamp_ms": 3_000,
                "semantic_key": "person_entry",
                "label": "Person enters the room",
                "source": "state_transition",
            }
        ],
        "evidence": [{"detection_id": 41}],
        "coverage": {"status": "covered"},
    }
    store = _TemporalStore()

    created = _service(store, _Runtime()).store_draft(
        draft,
        actor_id="operator-1",
    )

    assert created["id"] == INCIDENT_ID
    assert len(store.episodes) == 1
    assert store.episodes[0]["semantic_key"] == "person_entry"
    assert (
        store.episodes[0]["coverage"]["scale_disposition"]
        == "long_incident_candidate"
    )
    assert len(store.relations) == 1
    assert store.relations[0]["relation_type"] == "series_member"
    assert store.relations[0]["relation_state"] == "candidate"
    assert store.relations[0]["payload"]["automatic_merge"] is False


def test_generic_vlm_transport_key_is_rejected_as_series_evidence():
    relation_id = "00000000-0000-0000-0000-000000000119"

    class _GenericStore:
        def __init__(self):
            self.appended = []

        def list_episodes(self, *_args, **_kwargs):
            return [{"episode_key": "legacy"}], 1

        def list_relations(self, *_args, **_kwargs):
            return [
                {
                    "id": relation_id,
                    "subject_incident_id": INCIDENT_ID,
                    "object_incident_id": "00000000-0000-0000-0000-000000000116",
                    "relation_type": "series_member",
                    "relation_state": "candidate",
                    "payload": {"semantic_key": "vlm_alert"},
                }
            ], 1

        def append_relation(self, record, **_kwargs):
            self.appended.append(dict(record))
            return dict(record)

        def list_incidents(self, **_kwargs):
            raise AssertionError("generic semantic keys must not start series search")

    store = _GenericStore()
    service = _service(store, _Runtime())
    incident = {
        **_record(state="draft"),
        "timeline_refs": [
            {
                "timestamp_ms": 1_000,
                "semantic_key": "vlm_alert",
                "label": "Person enters",
                "source": "vlm_alert",
            }
        ],
    }

    result = service.ensure_temporal_projection(incident)

    assert service._primary_semantic_key(incident) == ""
    assert result["relation_created"] is False
    assert result["relations_rejected"] == 1
    assert store.appended[0]["relation_state"] == "rejected"
    assert store.appended[0]["payload"]["supersedes_relation_id"] == relation_id


def test_operator_review_appends_series_correction_without_merging_incidents():
    relation_id = "00000000-0000-0000-0000-000000000119"
    correction_id = "00000000-0000-0000-0000-000000000120"

    class _SeriesStore(_Store):
        def __init__(self):
            super().__init__()
            self.relations = [
                {
                    "id": relation_id,
                    "subject_incident_id": INCIDENT_ID,
                    "object_incident_id": "00000000-0000-0000-0000-000000000116",
                    "relation_type": "series_member",
                    "relation_state": "candidate",
                    "confidence": "medium",
                    "rationale": "Same semantic track recurred.",
                    "payload": {
                        "semantic_key": "person_entry",
                        "series_key": "series-person-entry",
                        "gap_ms": 120_000,
                        "operator_review_required": True,
                    },
                }
            ]

        def list_relations(self, *_args, **_kwargs):
            return [dict(item) for item in self.relations], len(self.relations)

        def list_episodes(self, *_args, **_kwargs):
            return [], 0

        def append_relation(self, record, **_kwargs):
            stored = {"id": correction_id, **dict(record)}
            self.relations.append(stored)
            return dict(stored)

    store = _SeriesStore()
    service = _service(store, _Runtime())

    correction = service.review_series_relation(
        INCIDENT_ID,
        relation_id,
        actor_id="operator-1",
        action="confirm",
        note="Same person and behavior; keep as a recurrence series.",
    )
    temporal = service.temporal_context(store.record)

    assert correction["relation_state"] == "confirmed"
    assert correction["payload"]["supersedes_relation_id"] == relation_id
    assert correction["payload"]["automatic_merge"] is False
    assert len(temporal["series_links"]) == 1
    assert temporal["series_links"][0]["relation_id"] == correction_id
    assert temporal["series_links"][0]["relation_state"] == "confirmed"
    assert temporal["correction_count"] == 1


def test_l0_temporal_ingestion_creates_continues_and_ends_candidate_at_routine():
    class _AutomaticStore:
        def __init__(self):
            self.records = []
            self.observations = []
            self.transitions = []

        def list_incidents(self, **_kwargs):
            return [dict(item) for item in self.records], len(self.records)

        def create_incident(self, record, **_kwargs):
            stored = {
                "id": INCIDENT_ID,
                "revision": 1,
                **dict(record),
            }
            self.records.append(stored)
            return dict(stored)

        def update_incident(
            self,
            incident_id,
            *,
            expected_revision,
            changes,
            actor_id,
            transition=None,
        ):
            record = next(item for item in self.records if item["id"] == incident_id)
            assert record["revision"] == expected_revision
            record.update(dict(changes))
            record["revision"] += 1
            if transition:
                self.transitions.append(dict(transition))
            return dict(record)

        def append_observation(self, observation, **_kwargs):
            if not any(
                item["idempotency_key"] == observation["idempotency_key"]
                for item in self.observations
            ):
                self.observations.append(dict(observation))
            return dict(observation)

    store = _AutomaticStore()
    service = _service(store, _Runtime())
    base_heartbeat = {
        "batch_id": "batch-1",
        "batch_start_ms": 1_000,
        "batch_end_ms": 2_000,
        "vector_signal": {
            "capture_attention": {
                "seconds": [
                    {"activity_x": 1.2, "mode": "normal"},
                    {"activity_x": 4.5, "mode": "burst"},
                ]
            }
        },
    }
    first = {
        "observation_id": "obs-1",
        "kind": "event",
        "state": "new",
        "semantic_key": "person enter",
        "label": "Person enters the room",
        "start_ms": 1_000,
        "end_ms": 2_000,
        "evidence_refs": ["batch-1:snapshot:2"],
    }

    created = service.ingest_l0_temporal_observations(
        112, base_heartbeat, [first]
    )
    continued = service.ingest_l0_temporal_observations(
        112,
        {**base_heartbeat, "batch_id": "batch-2", "batch_start_ms": 3_000, "batch_end_ms": 4_000},
        [
            {
                **first,
                "observation_id": "obs-2",
                "state": "continuing",
                "start_ms": 3_000,
                "end_ms": 4_000,
                "evidence_refs": ["batch-2:snapshot:1"],
            }
        ],
    )
    ended = service.ingest_l0_temporal_observations(
        112,
        {**base_heartbeat, "batch_id": "batch-3", "batch_start_ms": 5_000, "batch_end_ms": 6_000},
        [
            {
                "observation_id": "routine-1",
                "kind": "routine_gap",
                "semantic_key": "desk routine",
                "label": "Room returned to routine",
                "start_ms": 6_000,
                "end_ms": 6_000,
                "applies_to": ["person enter"],
            }
        ],
    )

    assert created["created"] == 1
    assert continued["created"] == 0
    assert continued["associated"] == 1
    assert ended["ended"] == 1
    assert len(store.records) == 1
    assert store.records[0]["perception_state"] == "ended"
    assert store.records[0]["case_state"] == "candidate"
    assert store.records[0]["possible_end_ms"] == 6_000
    assert len(store.records[0]["timeline_refs"]) == 2
    assert len(store.observations) == 3
    assert store.records[0]["qualia_refs"][0]["activity_x_max"] == 4.5

    held_for_rollup = service.ingest_l0_temporal_observations(
        112,
        {
            **base_heartbeat,
            "batch_id": "batch-4",
            "batch_start_ms": 7_000,
            "batch_end_ms": 8_000,
        },
        [
            {
                **first,
                "observation_id": "obs-episode-only",
                "semantic_key": "person exit",
                "label": "Person leaves the room",
                "trigger_kind": "episode_event",
                "start_ms": 7_000,
                "end_ms": 8_000,
            }
        ],
    )
    assert held_for_rollup["created"] == 0
    assert held_for_rollup["skipped"] == 1
    assert len(store.records) == 1


def test_l0_operator_alert_candidate_preserves_admission_priority():
    service = _service(_Store(), _Runtime())
    record = service._l0_incident_record(
        112,
        {
            "batch_id": "batch-alert-1",
            "batch_start_ms": 1_000,
            "batch_end_ms": 2_000,
        },
        {
            "observation_id": "obs-alert-1",
            "kind": "event",
            "state": "new",
            "semantic_key": "person thumbs_up",
            "label": "Thumb-up gesture detected",
            "start_ms": 1_000,
            "end_ms": 2_000,
            "trigger_kind": "operator_alert",
            "severity": "info",
            "operator_criterion": "you spot a thumbs-up gesture",
            "evidence_refs": ["batch-alert-1:snapshot:3"],
        },
        {"sample_count": 1, "activity_x_max": 2.0},
    )

    assert record["report"]["source"] == "operator_alert_l0"
    assert record["report"]["priority"] == "operator_criterion"
    assert record["timeline_refs"][0]["trigger_kind"] == "operator_alert"
    assert record["timeline_refs"][0]["operator_criterion"] == "you spot a thumbs-up gesture"


def test_episode_event_does_not_refresh_legacy_candidate_but_grounded_signal_upgrades_it():
    class _PriorityStore:
        def __init__(self):
            self.records = []
            self.observations = []

        def list_incidents(self, **_kwargs):
            return [dict(item) for item in self.records], len(self.records)

        def create_incident(self, record, **_kwargs):
            stored = {**dict(record), "id": "incident-priority", "revision": 1}
            self.records.append(stored)
            return dict(stored)

        def update_incident(self, incident_id, *, expected_revision, changes, **_kwargs):
            record = next(item for item in self.records if item["id"] == incident_id)
            assert record["revision"] == expected_revision
            record.update(dict(changes))
            record["revision"] += 1
            return dict(record)

        def append_observation(self, observation, **_kwargs):
            self.observations.append(dict(observation))
            return dict(observation)

    store = _PriorityStore()
    service = _service(store, _Runtime())
    heartbeat = {
        "batch_id": "batch-priority-1",
        "batch_start_ms": 1_000,
        "batch_end_ms": 2_000,
    }
    legacy = {
        "observation_id": "obs-legacy",
        "kind": "event",
        "state": "new",
        "semantic_key": "person thumbs_up",
        "label": "Person gesture",
        "start_ms": 1_000,
        "end_ms": 2_000,
    }
    service.ingest_l0_temporal_observations(112, heartbeat, [legacy])
    initial_revision = store.records[0]["revision"]

    skipped = service.ingest_l0_temporal_observations(
        112,
        {**heartbeat, "batch_id": "batch-priority-2", "batch_end_ms": 3_000},
        [{**legacy, "observation_id": "obs-episode", "trigger_kind": "episode_event"}],
    )

    assert skipped["associated"] == 0
    assert skipped["skipped"] == 1
    assert store.records[0]["revision"] == initial_revision

    upgraded = service.ingest_l0_temporal_observations(
        112,
        {**heartbeat, "batch_id": "batch-priority-3", "batch_end_ms": 4_000},
        [
            {
                **legacy,
                "observation_id": "obs-operator",
                "trigger_kind": "operator_alert",
                "operator_criterion": "you spot a thumbs-up gesture",
            }
        ],
    )

    assert upgraded["associated"] == 1
    assert store.records[0]["report"]["priority"] == "operator_criterion"
    assert store.records[0]["report"]["source"] == "operator_alert_l0"


def test_open_automatic_incident_waits_for_boundary_before_episode_materialization():
    class _ProjectionStore:
        def __init__(self):
            self.episodes = []

        def list_episodes(self, *_args, **_kwargs):
            return [], 0

        def list_relations(self, *_args, **_kwargs):
            return [], 0

        def append_episode(self, record, **_kwargs):
            self.episodes.append(dict(record))
            return dict(record)

        def list_incidents(self, **_kwargs):
            return [], 0

    store = _ProjectionStore()
    service = _service(store, _Runtime())
    open_incident = {
        **_record(state="candidate"),
        "case_state": "candidate",
        "perception_state": "observed",
        "possible_start_ms": 1_000,
        "observed_start_ms": 1_000,
        "observed_end_ms": 2_000,
        "possible_end_ms": None,
        "timeline_refs": [
            {"semantic_key": "person enter", "label": "Person enters"}
        ],
    }

    result = service.ensure_temporal_projection(open_incident)

    assert result["episode_created"] is False
    assert store.episodes == []


def test_l2_composition_replay_attaches_context_only_to_grounded_safety_case():
    safety = {
        **_record(state="candidate"),
        "case_state": "candidate",
        "possible_start_ms": 10_000,
        "observed_start_ms": 10_000,
        "observed_end_ms": 12_000,
        "anchor_ref": {"observation_id": "obs-collision"},
        "timeline_refs": [
            {
                "observation_id": "obs-collision",
                "semantic_key": "vehicle collision",
            }
        ],
        "report": {"priority": "safety", "severity": "critical"},
    }

    class _CompositionStore:
        def __init__(self, records):
            self.records = list(records)
            self.episodes = {}
            self.observations = {}
            self.list_calls = []

        def list_incidents(self, **kwargs):
            self.list_calls.append(dict(kwargs))
            return [dict(item) for item in self.records], len(self.records)

        def append_episode(self, record, **_kwargs):
            self.episodes.setdefault(record["idempotency_key"], dict(record))
            return dict(self.episodes[record["idempotency_key"]])

        def append_observation(self, record, **_kwargs):
            self.observations.setdefault(record["idempotency_key"], dict(record))
            return dict(self.observations[record["idempotency_key"]])

    rollup = {
        "rollup_id": "l2-ch112-w3600-0",
        "channel_id": 112,
        "level": "L2",
        "incident_ledger": [
            {
                "episode_id": "episode-collision",
                "semantic_key": "vehicle collision",
                "status": "open",
                "start_ms": 10_000,
                "last_observed_ms": 12_000,
                "observation_ids": ["obs-collision"],
                "priority": "safety",
                "severity": "critical",
            },
            {
                "episode_id": "episode-phone",
                "semantic_key": "person phone_call",
                "status": "ended_by_routine",
                "start_ms": 20_000,
                "last_observed_ms": 30_000,
                "boundary_at_ms": 31_000,
                "observation_ids": ["obs-phone"],
                "evidence_refs": ["batch-phone:snapshot:4"],
                "priority": "context",
                "scale_disposition": "routine_at_this_scale",
            },
        ],
        "incident_compositions": [
            {
                "composition_id": "composition-crash-phone",
                "parent_episode_id": "episode-collision",
                "parent_observation_ids": ["obs-collision"],
                "nested_episode_ids": ["episode-phone"],
                "semantic_keys": ["vehicle collision", "person phone_call"],
                "start_ms": 10_000,
                "end_ms": 31_000,
                "promotion_policy": "extend_grounded_anchor",
                "automatic_merge": False,
            }
        ],
    }
    store = _CompositionStore([safety])
    service = _service(store, _Runtime())

    first = service.ingest_rollup_incident_compositions(112, rollup)
    replay = service.ingest_rollup_incident_compositions(112, rollup)

    assert first["attached"] == 1
    assert replay["attached"] == 1
    assert len(store.episodes) == 1
    episode = next(iter(store.episodes.values()))
    assert episode["incident_id"] == INCIDENT_ID
    assert episode["semantic_key"] == "person phone_call"
    assert episode["perception_state"] == "ended"
    assert episode["coverage"]["nested_context"] is True
    assert episode["coverage"]["automatic_merge"] is False
    assert len(store.observations) == 1
    assert store.list_calls[0]["case_states"] == ["candidate", "open"]


def test_l2_composition_cannot_promote_context_or_info_operator_case():
    info_operator = {
        **_record(state="candidate"),
        "case_state": "candidate",
        "possible_start_ms": 10_000,
        "observed_start_ms": 10_000,
        "anchor_ref": {"observation_id": "obs-thumb"},
        "timeline_refs": [{"observation_id": "obs-thumb"}],
        "report": {"priority": "operator_criterion", "severity": "info"},
    }

    class _NoPromotionStore:
        def __init__(self):
            self.episodes = []
            self.observations = []

        def list_incidents(self, **_kwargs):
            return [dict(info_operator)], 1

        def append_episode(self, record, **_kwargs):
            self.episodes.append(dict(record))

        def append_observation(self, record, **_kwargs):
            self.observations.append(dict(record))

    rollup = {
        "rollup_id": "l2-ch112-w3600-0",
        "level": "L2",
        "incident_ledger": [
            {
                "episode_id": "episode-thumb",
                "semantic_key": "person thumbs_up",
                "start_ms": 10_000,
                "last_observed_ms": 11_000,
            },
            {
                "episode_id": "episode-head",
                "semantic_key": "person head_turn",
                "start_ms": 12_000,
                "last_observed_ms": 13_000,
            },
        ],
        "incident_compositions": [
            {
                "composition_id": "composition-info",
                "parent_observation_ids": ["obs-thumb"],
                "nested_episode_ids": ["episode-head"],
                "start_ms": 10_000,
                "end_ms": 13_000,
                "promotion_policy": "extend_grounded_anchor",
                "automatic_merge": False,
            }
        ],
    }
    store = _NoPromotionStore()

    result = _service(store, _Runtime()).ingest_rollup_incident_compositions(
        112,
        rollup,
    )

    assert result["attached"] == 0
    assert result["skipped"] == 1
    assert store.episodes == []
    assert store.observations == []


def test_review_projection_is_compact_and_keeps_independent_lifecycle_axes():
    service = IncidentCommandService(
        _Store(),
        object(),
        object(),
        _Runtime(),
        wall_clock_ms=lambda: 90_000,
    )
    record = {
        **_record(state="draft"),
        "case_state": "candidate",
        "possible_start_ms": 10_000,
        "observed_start_ms": 20_000,
        "observed_end_ms": 40_000,
        "possible_end_ms": 50_000,
        "timeline_refs": [
            {"timestamp_ms": 20_000, "semantic_key": "craft_entry", "label": "Craft enters"},
        ],
        "evidence_refs": [
            {"timestamp_ms": 30_000, "detection_id": 44, "role": "apex"},
        ],
        "anchor_ref": {"timestamp_ms": 20_000, "detection_id": 41},
        "uncertainties": ["Operator confirmation required"],
        "report": {"severity": "high", "summary": "Craft crossed the gate."},
    }

    review = service.public_review_record(record)

    assert review["review_state"] == "needs_review"
    assert review["perception_state"] == "observed"
    assert review["case_state"] == "candidate"
    assert review["cover"] == {"detection_id": 44, "timestamp_ms": 30_000, "role": "apex"}
    assert review["observed_duration_ms"] == 20_000
    assert review["case_duration_ms"] == 80_000
    assert review["semantic_keys"] == ["craft_entry"]
    assert review["uncertainty_count"] == 1
    assert "timeline" not in review
    assert "evidence" not in review


def test_operator_confirmed_ended_episode_stays_active_until_case_closure():
    service = IncidentCommandService(
        _Store(),
        object(),
        object(),
        _Runtime(),
        wall_clock_ms=lambda: 90_000,
    )
    record = {
        **_record(state="reported"),
        "case_state": "open",
        "possible_start_ms": 10_000,
        "observed_start_ms": 20_000,
        "observed_end_ms": 40_000,
        "possible_end_ms": 50_000,
    }

    review = service.public_review_record(record)

    assert review["review_state"] == "active"
    assert review["observed_duration_ms"] == 20_000


def test_review_page_resolves_compact_vlm_snapshot_refs_to_cover_images():
    class _Detections:
        def __init__(self):
            self.calls = []

        def resolve_vlm_snapshot_refs(self, refs):
            self.calls.append(list(refs))
            return {
                "vlm-batch-1:snapshot:3": {
                    "detection_id": 9300,
                    "timestamp_ms": 30_000,
                }
            }

    detections = _Detections()
    service = IncidentCommandService(
        _Store(),
        detections,
        object(),
        _Runtime(),
        wall_clock_ms=lambda: 90_000,
    )
    record = {
        **_record(state="draft"),
        "evidence_refs": [
            {
                "kind": "vlm_snapshot",
                "ref": "vlm-batch-1:snapshot:3",
                "role": "event",
            }
        ],
    }

    review = service.public_review_records([record])[0]

    assert detections.calls == [["vlm-batch-1:snapshot:3"]]
    assert review["cover"] == {
        "detection_id": 9300,
        "timestamp_ms": 30_000,
        "role": "event",
    }


def test_review_page_prefers_time_bounded_cover_resolver():
    class _Detections:
        def __init__(self):
            self.hints = []

        def resolve_vlm_snapshot_cover_refs(self, hints):
            self.hints = [dict(item) for item in hints]
            return {
                "vlm-batch-1:snapshot:3": {
                    "detection_id": 9300,
                    "timestamp_ms": 30_000,
                }
            }

        def resolve_vlm_snapshot_refs(self, _refs):
            raise AssertionError("slow tenant-wide resolver must not run")

    detections = _Detections()
    service = IncidentCommandService(
        _Store(),
        detections,
        object(),
        _Runtime(),
        wall_clock_ms=lambda: 90_000,
    )
    record = {
        **_record(state="draft"),
        "possible_start_ms": 10_000,
        "evidence_refs": [
            {
                "kind": "vlm_snapshot",
                "ref": "vlm-batch-1:snapshot:3",
                "role": "event",
            },
            {
                "kind": "vlm_snapshot",
                "ref": "vlm-batch-later:snapshot:8",
                "role": "post",
            },
        ],
    }

    review = service.public_review_records([record])[0]

    assert detections.hints == [
        {
            "ref": "vlm-batch-1:snapshot:3",
            "channel_id": 112,
            "timestamp_ms": 10_000,
        }
    ]
    assert review["cover"] == {
        "detection_id": 9300,
        "timestamp_ms": 30_000,
        "role": "event",
    }


def test_expired_follow_is_durably_finalized_with_operator_result():
    store = _Store()
    store.record["follow_policy"] = {
        "active": True,
        "mode": "follow",
        "run_id": "run-1",
        "relationship": "recurrence_watch",
        "started_at_ms": 1_000,
        "expires_at_ms": 9_000,
    }
    store.observations = [
        {
            "incident_id": INCIDENT_ID,
            "observed_at_ms": 8_000,
            "source_ref": {"follow_run_id": "run-1"},
            "payload": {"association": "neutral"},
        }
    ]

    updated = _service(store, _Runtime()).reconcile_expired_follow(store.record)

    assert updated["state"] == "draft"
    assert updated["attention_state"] == "inactive"
    assert updated["follow_policy"]["active"] is False
    assert updated["report"]["follow_result"]["outcome"] == "recurrence_not_confirmed"
    assert store.observations[-1]["source_kind"] == "follow_completed"
