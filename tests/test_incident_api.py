from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import oldapp


INCIDENT_ID = "00000000-0000-0000-0000-000000000117"


def _draft():
    return {
        "state": "draft",
        "title": "Person enters the monitored area",
        "severity": "high",
        "primary_channel_id": 112,
        "channel_ids": [112],
        "anchor": {"type": "detection", "detection_id": 41, "timestamp_ms": 1_785_000_030_000},
        "time_bounds": {
            "possible_start_ms": 1_785_000_000_000,
            "observed_start_ms": 1_785_000_010_000,
            "apex_ms": 1_785_000_030_000,
            "observed_end_ms": 1_785_000_050_000,
            "possible_end_ms": 1_785_000_060_000,
        },
        "timeline": [
            {
                "timestamp_ms": 1_785_000_010_000,
                "semantic_key": "person_entry",
                "label": "Person enters",
                "severity": "high",
                "confidence": "medium",
                "source": "state_transition",
            }
        ],
        "evidence": [{"kind": "detection", "detection_id": 41, "role": "alert"}],
        "qualia_digest": {"ground_truth": False, "probe_count": 1},
        "coverage": {"status": "covered", "summary_rows": 2, "gaps": []},
        "uncertainties": ["Operator confirmation required."],
        "provenance": {"connected_summary_rows": 2},
    }


class _Assembler:
    def __init__(self, *_args, **_kwargs):
        pass

    def assemble(self, request):
        assert request.channel_id == 112
        assert request.anchor_detection_id == 41
        return _draft()


class _IncidentStore:
    def __init__(self):
        self.record = None

    def create_incident(self, record, **_kwargs):
        self.record = {"id": INCIDENT_ID, "revision": 1, **dict(record)}
        return dict(self.record)

    def get_incident(self, incident_id):
        return dict(self.record) if self.record and incident_id == INCIDENT_ID else None

    def update_incident(self, incident_id, *, expected_revision, changes, **_kwargs):
        assert incident_id == INCIDENT_ID
        assert expected_revision == self.record["revision"]
        self.record.update(dict(changes))
        self.record["revision"] += 1
        return dict(self.record)


class _Runtime:
    def __init__(self):
        self.active = {}

    def start_incident_focus(self, incident_id, channel_ids, *, level, ttl_seconds):
        lease = SimpleNamespace(level=SimpleNamespace(value=level), channel_ids=tuple(channel_ids))
        self.active[incident_id] = lease
        return lease

    def stop_incident_focus(self, incident_id):
        return self.active.pop(incident_id, None) is not None


def test_incident_draft_follow_stop_and_export_contracts():
    store = _IncidentStore()
    runtime = _Runtime()
    previous_auth = oldapp.config.AUTH_ENABLED
    previous_token = oldapp.config.ADMIN_TOKEN
    oldapp.config.AUTH_ENABLED = False
    oldapp.config.ADMIN_TOKEN = "incident-test-token"
    headers = {"X-Admin-Token": "incident-test-token"}
    try:
        with (
            patch.object(oldapp, "incident_store", store),
            patch.object(oldapp, "luxriot_manager", runtime),
            patch.object(oldapp, "IncidentDraftAssembler", _Assembler),
        ):
            client = oldapp.app.test_client()
            response = client.post(
                "/incidents/draft",
                json={"channel_id": 112, "anchor_detection_id": 41},
                headers=headers,
            )
            assert response.status_code == 201, response.get_json()
            incident = response.get_json()["incident"]
            assert incident["incident_id"] == INCIDENT_ID
            assert incident["time_bounds"]["observed_start"] == 1_785_000_010_000
            assert incident["semantic_keys"] == ["person_entry"]
            assert "attention signals, not visual proof" in incident["summary"]

            response = client.post(
                f"/incidents/{INCIDENT_ID}/follow",
                json={"mode": "critical", "ttl_seconds": 300},
                headers=headers,
            )
            assert response.status_code == 200, response.get_json()
            followed = response.get_json()["incident"]
            assert followed["state"] == "following"
            assert followed["follow"]["active"] is True
            assert followed["follow"]["mode"] == "critical"

            response = client.post(
                f"/incidents/{INCIDENT_ID}/stop-follow",
                json={},
                headers=headers,
            )
            assert response.status_code == 200, response.get_json()
            stopped = response.get_json()["incident"]
            assert stopped["state"] == "draft"
            assert stopped["follow"]["active"] is False

            response = client.get(f"/incidents/{INCIDENT_ID}/export?format=md")
            assert response.status_code == 200
            assert response.mimetype == "text/markdown"
            assert b"Person enters" in response.data

            response = client.get(f"/incidents/{INCIDENT_ID}/export?format=xml")
            assert response.status_code == 200
            assert response.mimetype == "application/xml"
            assert b"groundTruthStatus=\"operator_review_required\"" in response.data
    finally:
        oldapp.config.AUTH_ENABLED = previous_auth
        oldapp.config.ADMIN_TOKEN = previous_token


def test_incident_draft_rejects_unbounded_request():
    previous_auth = oldapp.config.AUTH_ENABLED
    previous_token = oldapp.config.ADMIN_TOKEN
    oldapp.config.AUTH_ENABLED = False
    oldapp.config.ADMIN_TOKEN = "incident-test-token"
    try:
        response = oldapp.app.test_client().post(
            "/incidents/draft",
            json={"channel_id": 112},
            headers={"X-Admin-Token": "incident-test-token"},
        )
        assert response.status_code == 400
        assert "anchor" in response.get_json()["error"]
    finally:
        oldapp.config.AUTH_ENABLED = previous_auth
        oldapp.config.ADMIN_TOKEN = previous_token
