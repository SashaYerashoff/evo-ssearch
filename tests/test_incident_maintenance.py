from __future__ import annotations

from incident_maintenance import IncidentMaintenanceWorker


class _Store:
    def __init__(self, records):
        self.records = [dict(item) for item in records]
        self.calls = 0

    def list_incidents(self, **kwargs):
        self.calls += 1
        assert kwargs["states"] == ["following"]
        assert kwargs["limit"] <= 500
        return [dict(item) for item in self.records], len(self.records)


class _Service:
    def __init__(self, store):
        self.store = store
        self.calls = []

    def reconcile_expired_follow(self, record):
        self.calls.append(record["id"])
        for stored in self.store.records:
            if stored["id"] != record["id"]:
                continue
            follow = dict(stored.get("follow_policy") or {})
            if follow.get("expires_at_ms", 0) <= 10_000:
                follow["active"] = False
                stored["follow_policy"] = follow
                stored["state"] = "draft"
            return dict(stored)
        return dict(record)


def _record(index: int, *, expired: bool) -> dict:
    return {
        "id": f"incident-{index}",
        "state": "following",
        "follow_policy": {
            "active": True,
            "expires_at_ms": 9_000 if expired else 20_000,
        },
    }


def test_maintenance_finalizes_only_expired_follow_rows_with_bounded_scan():
    store = _Store([_record(index, expired=index < 4) for index in range(8)])
    service = _Service(store)
    worker = IncidentMaintenanceWorker(
        store,
        lambda: service,
        interval_sec=15,
        batch_size=8,
    )

    status = worker.run_once()

    assert status["records_scanned"] == 8
    assert status["records_finalized"] == 4
    assert status["errors"] == 0
    assert len(service.calls) == 8


def test_maintenance_replay_after_restart_is_idempotent():
    store = _Store([_record(1, expired=True)])
    first = _Service(store)
    second = _Service(store)

    first_status = IncidentMaintenanceWorker(store, lambda: first).run_once()
    second_status = IncidentMaintenanceWorker(store, lambda: second).run_once()

    assert first_status["records_finalized"] == 1
    assert second_status["records_finalized"] == 0
    assert store.records[0]["state"] == "draft"
    assert store.records[0]["follow_policy"]["active"] is False


def test_maintenance_contains_one_record_failure():
    store = _Store([_record(1, expired=True), _record(2, expired=True)])

    class _Flaky(_Service):
        def reconcile_expired_follow(self, record):
            if record["id"] == "incident-1":
                raise RuntimeError("broken record")
            return super().reconcile_expired_follow(record)

    status = IncidentMaintenanceWorker(store, lambda: _Flaky(store)).run_once()

    assert status["records_scanned"] == 2
    assert status["records_finalized"] == 1
    assert status["errors"] == 1
    assert status["last_error"] == "RuntimeError"


def test_maintenance_backfills_temporal_projection_in_bounded_pages():
    store = _Store([_record(index, expired=False) for index in range(3)])

    class _Projecting(_Service):
        def ensure_temporal_projection(self, record):
            return {
                "episode_created": True,
                "relation_created": record["id"] != "incident-0",
            }

    worker = IncidentMaintenanceWorker(
        store,
        lambda: _Projecting(store),
        batch_size=2,
    )
    original_list = store.list_incidents

    def list_incidents(**kwargs):
        if kwargs.get("states") == ["following"]:
            return original_list(**kwargs)
        offset = int(kwargs.get("offset") or 0)
        limit = int(kwargs.get("limit") or 2)
        return store.records[offset:offset + limit], len(store.records)

    store.list_incidents = list_incidents

    first = worker.run_once()
    second = worker.run_once()

    assert first["projections_scanned"] == 2
    assert first["episodes_materialized"] == 2
    assert first["series_candidates_materialized"] == 1
    assert second["projections_scanned"] == 3
    assert second["episodes_materialized"] == 3
    assert second["series_candidates_materialized"] == 2


def test_maintenance_reuses_projection_until_incident_revision_changes():
    records = [
        {
            **_record(index, expired=False),
            "revision": 1,
        }
        for index in range(2)
    ]
    store = _Store(records)

    class _Projecting(_Service):
        def __init__(self, incident_store):
            super().__init__(incident_store)
            self.projection_calls = []

        def ensure_temporal_projection(self, record):
            self.projection_calls.append((record["id"], record["revision"]))
            return {"episode_created": False, "relation_created": False}

    service = _Projecting(store)
    worker = IncidentMaintenanceWorker(store, lambda: service, batch_size=2)
    original_list = store.list_incidents

    def list_incidents(**kwargs):
        if kwargs.get("states") == ["following"]:
            return original_list(**kwargs)
        return [dict(item) for item in store.records], len(store.records)

    store.list_incidents = list_incidents

    first = worker.run_once()
    second = worker.run_once()
    store.records[1]["revision"] = 2
    third = worker.run_once()

    assert first["projections_reused"] == 0
    assert second["projections_reused"] == 2
    assert third["projections_reused"] == 3
    assert service.projection_calls == [
        ("incident-0", 1),
        ("incident-1", 1),
        ("incident-1", 2),
    ]


def test_temporal_projection_page_is_capped_below_realtime_follow_page():
    store = _Store([])
    requested_limits = []

    def list_incidents(**kwargs):
        if kwargs.get("states") == ["following"]:
            return [], 0
        requested_limits.append(int(kwargs["limit"]))
        return [], 0

    store.list_incidents = list_incidents

    class _Projecting(_Service):
        def ensure_temporal_projection(self, record):
            return {"episode_created": False, "relation_created": False}

    worker = IncidentMaintenanceWorker(store, lambda: _Projecting(store), batch_size=64)

    worker.run_once()

    assert worker.status()["batch_size"] == 64
    assert worker.status()["projection_batch_size"] == 8
    assert requested_limits == [8]
