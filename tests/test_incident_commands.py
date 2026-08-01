from __future__ import annotations

from types import SimpleNamespace

import pytest

from incident_commands import IncidentCommandService
from incident_store import IncidentRevisionConflict


INCIDENT_ID = "00000000-0000-0000-0000-000000000117"


def _record(*, revision: int = 1, state: str = "following"):
    return {
        "id": INCIDENT_ID,
        "revision": revision,
        "state": state,
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

    def get_incident(self, incident_id):
        return dict(self.record) if incident_id == INCIDENT_ID else None

    def update_incident(self, incident_id, *, expected_revision, changes, actor_id):
        self.update_calls.append((incident_id, expected_revision, dict(changes), actor_id))
        if self.conflict:
            raise IncidentRevisionConflict(incident_id, expected_revision, expected_revision + 1)
        self.record.update(dict(changes))
        self.record["revision"] += 1
        return dict(self.record)


class _Runtime:
    def __init__(self):
        self.active = {INCIDENT_ID: object()}
        self.stop_calls = []

    def start_incident_focus(self, incident_id, channel_ids, *, level, ttl_seconds):
        lease = SimpleNamespace(level=SimpleNamespace(value=level), channel_ids=tuple(channel_ids))
        self.active[incident_id] = lease
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
    assert updated["follow_policy"]["mode"] == "critical"
    assert updated["follow_policy"]["expires_at_ms"] == 130_000
    assert lease.level.value == "critical"


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
