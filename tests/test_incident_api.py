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
        self.appended_observations = []
        self.transitions = []
        self.list_calls = []
        self.relations = [
            {
                "id": "00000000-0000-0000-0000-000000000120",
                "subject_incident_id": INCIDENT_ID,
                "object_incident_id": "00000000-0000-0000-0000-000000000121",
                "relation_type": "series_member",
                "relation_state": "candidate",
                "confidence": "medium",
                "rationale": "The same semantic track recurred after routine.",
                "payload": {
                    "semantic_key": "person_entry",
                    "series_key": "series-person-entry",
                    "gap_ms": 120_000,
                    "operator_review_required": True,
                },
            }
        ]

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
        transition = _kwargs.get("transition")
        if isinstance(transition, dict):
            self.transitions.append(
                {
                    "id": f"transition-{len(self.transitions) + 1}",
                    "axis": "case",
                    "from_state": None,
                    "to_state": self.record.get("case_state", "unknown"),
                    "incident_revision": self.record["revision"],
                    "transitioned_at_ms": transition.get("transitioned_at_ms"),
                    "reason": transition.get("reason", ""),
                    "source_kind": transition.get("source_kind", "unknown"),
                }
            )
        return dict(self.record)

    def list_incidents(self, **_kwargs):
        self.list_calls.append(dict(_kwargs))
        rows = [dict(self.record)] if self.record else []
        return rows, len(rows)

    def list_observations(self, incident_id, **_kwargs):
        if not self.record or incident_id != INCIDENT_ID:
            return [], 0
        return [
            {
                "id": "00000000-0000-0000-0000-000000000118",
                "incident_id": INCIDENT_ID,
                "idempotency_key": "l0:vlm-batch",
                "source_kind": "vlm_l0_heartbeat",
                "observed_at_ms": 1_785_000_040_000,
                "channel_id": 112,
                "perception_state": "unknown",
                "source_ref": {"batch_id": "vlm-batch"},
                "payload": {"coverage_gap": False},
            }
        ], 1

    def list_episodes(self, incident_id, **_kwargs):
        if not self.record or incident_id != INCIDENT_ID:
            return [], 0
        return [
            {
                "id": "00000000-0000-0000-0000-000000000119",
                "incident_id": INCIDENT_ID,
                "episode_key": "episode:person-entry",
                "perception_state": "observed",
                "semantic_key": "person_entry",
                "possible_start_ms": 1_785_000_000_000,
                "observed_start_ms": 1_785_000_010_000,
                "observed_end_ms": 1_785_000_050_000,
                "possible_end_ms": 1_785_000_060_000,
                "evidence_refs": [{"detection_id": 41}],
                "coverage": {
                    "scale_disposition": "unclassified_keep",
                    "operator_review_required": True,
                },
            }
        ], 1

    def list_relations(self, incident_id, **_kwargs):
        if not self.record or incident_id != INCIDENT_ID:
            return [], 0
        return [dict(item) for item in self.relations], len(self.relations)

    def list_transitions(self, incident_id, **_kwargs):
        if not self.record or incident_id != INCIDENT_ID:
            return [], 0
        return [dict(item) for item in self.transitions], len(self.transitions)

    def append_relation(self, relation, **_kwargs):
        stored = {
            "id": "00000000-0000-0000-0000-000000000122",
            **dict(relation),
        }
        self.relations.append(stored)
        return dict(stored)

    def append_observation(self, observation, **_kwargs):
        self.appended_observations.append(dict(observation))
        return dict(observation)


class _Runtime:
    def __init__(self):
        self.active = {}

    def start_incident_focus(self, incident_id, channel_ids, *, level, ttl_seconds):
        lease = SimpleNamespace(level=SimpleNamespace(value=level), channel_ids=tuple(channel_ids))
        self.active[incident_id] = lease
        return lease

    def stop_incident_focus(self, incident_id):
        return self.active.pop(incident_id, None) is not None

    def incident_focus_status(self):
        return {
            "active": len(self.active),
            "attention_policy": {
                "normal_foreground_limit": 2,
                "hard_foreground_limit": 4,
                "hot_unresolved_limit": 8,
            },
        }

    def incident_focus_for_channel(self, channel_id):
        hot = tuple(self.active) if int(channel_id) == 112 else ()
        return SimpleNamespace(hot_incident_ids=hot) if hot else None


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
            assert incident["summary"] == "Person enters."
            assert incident["title"] == "Person enters monitored area"
            assert incident["homeostasis"]["interpretation"] == "attention signals, not visual proof"

            response = client.get("/incidents?channel_id=112&case_state=candidate")
            assert response.status_code == 200, response.get_json()
            listed = response.get_json()
            assert listed["total"] == 1
            assert listed["incidents"][0]["incident_id"] == INCIDENT_ID
            assert listed["attention"]["attention_policy"]["hot_unresolved_limit"] == 8

            response = client.get("/incidents?view=review&channel_id=112&limit=50")
            assert response.status_code == 200, response.get_json()
            review_page = response.get_json()
            assert review_page["view"] == "review"
            review = review_page["incidents"][0]
            assert review["incident_id"] == INCIDENT_ID
            assert review["review_state"] == "needs_review"
            assert review["cover"]["detection_id"] == 41
            assert review["cover"]["role"] == "alert"
            assert review["observed_duration_ms"] == 40_000
            assert review["evidence_count"] == 1
            assert "timeline" not in review
            assert "evidence" not in review
            assert store.list_calls[-1]["top_level_only"] is True

            response = client.get(f"/incidents/{INCIDENT_ID}/observations")
            assert response.status_code == 200, response.get_json()
            observation_page = response.get_json()
            assert observation_page["total"] == 1
            assert observation_page["observations"][0]["source_kind"] == "vlm_l0_heartbeat"

            response = client.get(f"/incidents/{INCIDENT_ID}/temporal")
            assert response.status_code == 200, response.get_json()
            temporal = response.get_json()
            assert temporal["episode_total"] == 1
            assert temporal["episodes"][0]["scale_disposition"] == "unclassified_keep"
            assert temporal["episodes"][0]["evidence_count"] == 1
            assert temporal["series_links"][0]["related_incident_id"].endswith("0121")
            assert temporal["series_links"][0]["automatic_merge"] is False

            response = client.post(
                f"/incidents/{INCIDENT_ID}/series/"
                "00000000-0000-0000-0000-000000000120/review",
                json={"action": "confirm", "note": "Same recurring episode."},
                headers=headers,
            )
            assert response.status_code == 200, response.get_json()
            reviewed_series = response.get_json()
            assert reviewed_series["relation"]["relation_state"] == "confirmed"
            assert reviewed_series["temporal"]["series_links"][0]["relation_state"] == "confirmed"
            assert reviewed_series["temporal"]["correction_count"] == 1

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

            heartbeat = oldapp._append_l0_incident_observations(
                112,
                {
                    "batch_id": "vlm-batch-1",
                    "batch_start_ms": 1_785_000_030_000,
                    "batch_end_ms": 1_785_000_040_000,
                    "batch_state": {"version": 2, "events": []},
                },
            )
            assert heartbeat["inserted"] == 1
            assert heartbeat["hot_eligible"] == 1
            assert store.appended_observations[0]["perception_state"] == "unknown"
            assert store.appended_observations[0]["idempotency_key"] == "l0:vlm-batch-1"

            response = client.post(
                f"/incidents/{INCIDENT_ID}/stop-follow",
                json={},
                headers=headers,
            )
            assert response.status_code == 200, response.get_json()
            stopped = response.get_json()["incident"]
            assert stopped["state"] == "draft"
            assert stopped["follow"]["active"] is False

            response = client.post(
                f"/incidents/{INCIDENT_ID}/review",
                json={
                    "action": "confirm",
                    "expected_revision": stopped["revision"],
                    "note": "Reviewed against the stored entry frame.",
                },
                headers=headers,
            )
            assert response.status_code == 200, response.get_json()
            confirmed = response.get_json()["incident"]
            assert confirmed["state"] == "reported"
            assert confirmed["case_state"] == "open"
            assert confirmed["risk_state"] == "unknown"
            assert store.appended_observations[-1]["source_kind"] == "operator_review"
            response = client.get("/incidents?view=review&channel_id=112&limit=50")
            assert response.status_code == 200, response.get_json()
            assert response.get_json()["incidents"][0]["review_state"] == "active"

            response = client.post(
                f"/incidents/{INCIDENT_ID}/review",
                json={
                    "action": "false_positive",
                    "expected_revision": confirmed["revision"],
                },
                headers=headers,
            )
            assert response.status_code == 200, response.get_json()
            historical = response.get_json()["incident"]
            assert historical["state"] == "closed"
            assert historical["case_state"] == "false_positive"

            response = client.get("/incidents?view=review&channel_id=112&limit=50")
            assert response.status_code == 200, response.get_json()
            assert response.get_json()["incidents"][0]["review_state"] == "history"

            response = client.post(
                f"/incidents/{INCIDENT_ID}/review",
                json={
                    "action": "reopen",
                    "expected_revision": historical["revision"],
                },
                headers=headers,
            )
            assert response.status_code == 200, response.get_json()
            reopened = response.get_json()["incident"]
            assert reopened["state"] == "reported"
            assert reopened["case_state"] == "open"

            response = client.get(f"/incidents/{INCIDENT_ID}/export?format=md")
            assert response.status_code == 200
            assert response.mimetype == "text/markdown"
            assert b"Person enters" in response.data
            assert b"Temporal memory" in response.data
            assert b"Lifecycle history" in response.data

            response = client.get(f"/incidents/{INCIDENT_ID}/export?format=xml")
            assert response.status_code == 200
            assert response.mimetype == "application/xml"
            assert b"groundTruthStatus=\"operator_review_required\"" in response.data
            assert b"<temporalMemory" in response.data
            assert b"<lifecycleHistory>" in response.data
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


def test_incident_list_rejects_unknown_view():
    response = oldapp.app.test_client().get("/incidents?view=everything")
    assert response.status_code == 400
    assert response.get_json()["error"] == "view must be full or review"
