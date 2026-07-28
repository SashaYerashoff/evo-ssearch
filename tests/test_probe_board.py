"""Probe authorship backfill and operator-defined channel grouping."""

from __future__ import annotations

import json

import pytest

from alert_probe_lifecycle import derive_parent_alert_id
from probe_board import (
    ORIGIN_AGENT,
    ORIGIN_AUTO,
    ORIGIN_OPERATOR,
    PROBE_ORIGINS,
    ChannelGroupError,
    ChannelGroupStore,
    annotate_probe_origin,
    carry_probe_provenance,
    coerce_probe_origin,
    normalize_probe_origin,
)


class TestProbeOrigin:
    def test_explicit_origin_wins(self):
        for origin in PROBE_ORIGINS:
            assert normalize_probe_origin({"origin": origin}) == origin

    def test_origin_is_case_and_whitespace_tolerant(self):
        assert normalize_probe_origin({"origin": "  Agent "}) == ORIGIN_AGENT

    def test_unknown_origin_falls_back_to_operator(self):
        assert normalize_probe_origin({"origin": "definitely-not-an-origin"}) == ORIGIN_OPERATOR

    @pytest.mark.parametrize(
        "probe",
        [
            {"temporary": True},
            {"parent_alert_id": "vlm-alert-abc"},
            {"source": "vlm_alert"},
        ],
    )
    def test_legacy_alert_probes_backfill_to_auto(self, probe):
        """Probes stored before ``origin`` existed keep an honest badge."""
        assert normalize_probe_origin(probe) == ORIGIN_AUTO

    def test_legacy_plain_probe_backfills_to_operator(self):
        assert normalize_probe_origin({"name": "fighting", "channel_id": 112}) == ORIGIN_OPERATOR

    def test_non_mapping_is_safe(self):
        assert normalize_probe_origin(None) == ORIGIN_OPERATOR

    def test_annotate_does_not_mutate_the_source(self):
        probe = {"temporary": True}
        annotated = annotate_probe_origin(probe)
        assert annotated["origin"] == ORIGIN_AUTO
        assert "origin" not in probe

    def test_coerce_respects_default(self):
        assert coerce_probe_origin("", default=ORIGIN_AGENT) == ORIGIN_AGENT
        assert coerce_probe_origin("auto", default=ORIGIN_AGENT) == ORIGIN_AUTO
        assert coerce_probe_origin("nonsense", default="also-nonsense") == ORIGIN_OPERATOR


def test_parent_alert_id_is_stable_and_respects_an_explicit_id():
    event = {
        "channel_id": 112,
        "title": "Person down",
        "description": "Person remains on the ground.",
        "severity": "high",
        "timestamp_ms": 1_785_000_000_000,
    }
    derived = derive_parent_alert_id(event)
    assert derived.startswith("vlm-alert-")
    assert derive_parent_alert_id(dict(event)) == derived
    assert derive_parent_alert_id({**event, "id": "alert-explicit"}) == "alert-explicit"


class TestProvenanceCarryOver:
    def test_operator_edit_keeps_agent_authorship(self):
        """A rebuilt payload must not relabel an agent probe as operator-made."""
        rebuilt = {"name": "renamed", "channel_id": 112}
        carry_probe_provenance(rebuilt, {"origin": ORIGIN_AGENT})
        assert rebuilt["origin"] == ORIGIN_AGENT

    def test_operator_edit_keeps_alert_lineage(self):
        rebuilt = {"name": "renamed"}
        carry_probe_provenance(
            rebuilt,
            {
                "origin": ORIGIN_AUTO,
                "temporary": True,
                "parent_alert_id": "vlm-alert-abc",
                "parent_alert_title": "smoke near berth 3",
                "expires_at_ms": 1_700_000_000_000,
            },
        )
        assert rebuilt["temporary"] is True
        assert rebuilt["parent_alert_id"] == "vlm-alert-abc"
        assert rebuilt["expires_at_ms"] == 1_700_000_000_000

    def test_explicit_new_value_is_not_overwritten(self):
        rebuilt = {"origin": ORIGIN_AGENT}
        carry_probe_provenance(rebuilt, {"origin": ORIGIN_OPERATOR})
        assert rebuilt["origin"] == ORIGIN_AGENT

    def test_carry_is_a_deep_copy(self):
        existing = {"lifecycle": {"status": "active"}}
        rebuilt = {}
        carry_probe_provenance(rebuilt, existing)
        rebuilt["lifecycle"]["status"] = "retired"
        assert existing["lifecycle"]["status"] == "active"

    def test_missing_existing_probe_is_a_no_op(self):
        rebuilt = {"name": "fresh"}
        assert carry_probe_provenance(rebuilt, None) == {"name": "fresh"}


class TestAlertProbeLifecycleStampsAuto:
    def test_store_payload_declares_auto_origin(self):
        """The alert lifecycle is the only writer of background VLM probes."""
        from alert_probe_lifecycle import AlertProbeLifecycle

        lifecycle = AlertProbeLifecycle()
        admission = lifecycle.admit_alert_event(
            {
                "id": "vlm-alert-1",
                "channel_id": 112,
                "title": "smoke near berth 3",
                "description": "grey plume rising",
                "timestamp_ms": 1_700_000_000_000,
            },
            specs=[
                {"label": "smoke", "positives": ["smoke plume"], "negatives": ["clear sky"]},
                {"label": "fire", "positives": ["open flame"], "negatives": ["empty quay"]},
            ],
        )
        assert admission.accepted, admission
        payload = admission.probes[0].to_store_payload()
        assert payload["origin"] == ORIGIN_AUTO
        assert payload["temporary"] is True
        # The lifecycle's own lineage guard keeps its separate meaning.
        assert payload["source"] == "vlm_alert"
        assert normalize_probe_origin(payload) == ORIGIN_AUTO


@pytest.fixture()
def group_store(tmp_path):
    return ChannelGroupStore(tmp_path / "groups.json")


class TestChannelGroupStore:
    def test_create_and_list(self, group_store):
        group = group_store.upsert_group(name="Perimeter", channel_ids=[112, 120])
        assert group["id"].startswith("grp-")
        assert group["channel_ids"] == [112, 120]
        assert [g["name"] for g in group_store.list_groups()] == ["Perimeter"]

    def test_channel_ids_are_deduped_and_coerced(self, group_store):
        group = group_store.upsert_group(name="Gates", channel_ids=["112", 112, " 120 "])
        assert group["channel_ids"] == [112, 120]

    def test_a_channel_belongs_to_one_group_only(self, group_store):
        """Claiming a channel must remove it from its previous group."""
        first = group_store.upsert_group(name="Perimeter", channel_ids=[112, 120])
        second = group_store.upsert_group(name="Berth 3", channel_ids=[120])
        groups = {g["id"]: g for g in group_store.list_groups()}
        assert groups[first["id"]]["channel_ids"] == [112]
        assert groups[second["id"]]["channel_ids"] == [120]
        assert group_store.group_id_by_channel() == {112: first["id"], 120: second["id"]}

    def test_rename_keeps_channels(self, group_store):
        group = group_store.upsert_group(name="Perimeter", channel_ids=[112])
        renamed = group_store.upsert_group(group_id=group["id"], name="Outer perimeter")
        assert renamed["channel_ids"] == [112]
        assert renamed["name"] == "Outer perimeter"

    def test_delete_only_removes_the_group(self, group_store):
        group = group_store.upsert_group(name="Perimeter", channel_ids=[112])
        assert group_store.delete_group(group["id"]) is True
        assert group_store.delete_group(group["id"]) is False
        assert group_store.list_groups() == []
        assert group_store.group_id_by_channel() == {}

    def test_groups_list_in_creation_order(self, group_store):
        group_store.upsert_group(name="A", channel_ids=[1])
        group_store.upsert_group(name="B", channel_ids=[2])
        assert [g["name"] for g in group_store.list_groups()] == ["A", "B"]

    def test_explicit_position_overrides_creation_order(self, group_store):
        first = group_store.upsert_group(name="A", channel_ids=[1])
        group_store.upsert_group(name="B", channel_ids=[2])
        group_store.upsert_group(group_id=first["id"], position=5)
        assert [g["name"] for g in group_store.list_groups()] == ["B", "A"]

    def test_state_survives_a_reload(self, tmp_path):
        path = tmp_path / "groups.json"
        ChannelGroupStore(path).upsert_group(name="Perimeter", channel_ids=[112])
        assert [g["name"] for g in ChannelGroupStore(path).list_groups()] == ["Perimeter"]

    def test_corrupt_file_does_not_crash_the_board(self, tmp_path):
        path = tmp_path / "groups.json"
        path.write_text("{ not json", encoding="utf-8")
        assert ChannelGroupStore(path).list_groups() == []

    def test_writes_are_atomic_json(self, tmp_path):
        path = tmp_path / "groups.json"
        ChannelGroupStore(path).upsert_group(name="Perimeter", channel_ids=[112])
        assert json.loads(path.read_text(encoding="utf-8"))["groups"][0]["name"] == "Perimeter"
        assert not path.with_suffix(".json.tmp").exists()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": ""},
            {"name": "   "},
            {"name": "x" * 81},
            {"name": "ok", "channel_ids": [0]},
            {"name": "ok", "channel_ids": [-1]},
            {"name": "ok", "channel_ids": "112"},
            {"name": "ok", "channel_ids": ["not-a-number"]},
        ],
    )
    def test_invalid_input_is_rejected(self, group_store, kwargs):
        with pytest.raises(ChannelGroupError):
            group_store.upsert_group(**kwargs)

    def test_updating_an_unknown_group_is_rejected(self, group_store):
        with pytest.raises(ChannelGroupError):
            group_store.upsert_group(group_id="grp-missing", name="ghost")

    def test_health_reports_backend_and_path(self, group_store):
        health = group_store.health()
        assert health["ok"] is True
        assert health["backend"] == "json"
