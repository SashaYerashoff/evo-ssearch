import copy

import pytest

from agent import AgentTools
from deployment_workflow import (
    DeploymentWorkflowError,
    ProtocolDeploymentStore,
    aggregate_counted_state_metric,
)


class _RuntimeState:
    def __init__(self):
        self.data = {}

    def load_state(self, key):
        return copy.deepcopy(self.data.get(key))

    def save_state(self, key, payload):
        self.data[key] = copy.deepcopy(dict(payload))


class _ProbeStore:
    def __init__(self):
        self.rows = []

    def list_probes(self):
        return copy.deepcopy(self.rows)

    def upsert_probe(self, payload):
        saved = copy.deepcopy(dict(payload))
        existing_index = next(
            (
                index
                for index, item in enumerate(self.rows)
                if saved.get("id")
                and str(item.get("id")) == str(saved.get("id"))
            ),
            None,
        )
        if existing_index is None:
            saved["id"] = saved.get("id") or f"probe-{len(self.rows) + 1}"
            self.rows.append(saved)
        else:
            self.rows[existing_index] = saved
        return copy.deepcopy(saved)


class _GroupStore:
    def __init__(self):
        self.rows = []

    def list_groups(self):
        return copy.deepcopy(self.rows)

    def upsert_group(self, *, group_id=None, name=None, channel_ids=None):
        saved = {
            "id": group_id or f"group-{len(self.rows) + 1}",
            "name": name,
            "channel_ids": list(channel_ids or []),
        }
        existing_index = next(
            (
                index
                for index, item in enumerate(self.rows)
                if str(item.get("id")) == str(saved["id"])
            ),
            None,
        )
        if existing_index is None:
            self.rows.append(saved)
        else:
            self.rows[existing_index] = saved
        return copy.deepcopy(saved)


class _Manager:
    def __init__(self):
        self.prompts = {
            11: {
                "alert_policy_prompt": "KEEP EXISTING POLICY",
                "stream_system_prompt": "KEEP STREAM CORE",
                "rollup_prompts": {"L1": "", "L2": "", "L3": ""},
            },
            12: {
                "alert_policy_prompt": "",
                "stream_system_prompt": "",
                "rollup_prompts": {"L1": "", "L2": "", "L3": ""},
            },
        }
        self.sessions = {}
        self.schedule = None

    def get_channels(self, force=False):
        return [
            {"id": 11, "title": "Gate"},
            {"id": 12, "title": "Workstation"},
        ]

    def get_prompt_settings(self, channel_id=None):
        return {"current": copy.deepcopy(self.prompts[int(channel_id)])}

    def update_prompt_settings(
        self,
        channel_id=None,
        alert_policy_prompt=None,
        stream_system_prompt=None,
        rollup_prompts=None,
    ):
        channel = self.prompts[int(channel_id)]
        channel["alert_policy_prompt"] = alert_policy_prompt
        if stream_system_prompt is not None:
            channel["stream_system_prompt"] = stream_system_prompt
        if rollup_prompts is not None:
            channel["rollup_prompts"] = copy.deepcopy(dict(rollup_prompts))
        return {"status": "updated"}

    def start_session(self, channel_id):
        self.sessions[int(channel_id)] = object()

    def set_rollup_l3_deep_schedule(self, schedule, persist=True):
        self.schedule = copy.deepcopy(dict(schedule))
        return {"schedule": copy.deepcopy(self.schedule)}

    def summary_rollups(self, **_kwargs):
        return {
            "levels": {
                "L1": [
                    {
                        "summary": "Bounded first-window review",
                        "generation_status": "generated",
                        "window_start": 1.0,
                        "window_end": 901.0,
                    }
                ]
            }
        }


class _DetectionStore:
    def count_vector_candidates(self, **_kwargs):
        return 120


def _configured_state(store):
    state = store.start(
        [
            {"id": 11, "title": "Gate"},
            {"id": 12, "title": "Workstation"},
            {"id": 13, "title": "Yard"},
        ],
        resume_latest=False,
    )
    deployment_id = state["deployment_id"]
    store.configure(
        deployment_id,
        channel_ids=[11, 12],
        groups=[
            {"name": "Perimeter", "channel_ids": [11]},
            {"name": "Operations", "channel_ids": [12]},
        ],
    )
    store.record_survey(
        deployment_id,
        {
            "channels": [
                {
                    "channel_id": 11,
                    "title": "Gate",
                    "sample_count": 4,
                    "survey": "SCENE: gate",
                },
                {
                    "channel_id": 12,
                    "title": "Workstation",
                    "sample_count": 4,
                    "survey": "SCENE: desk",
                },
            ]
        },
    )
    return deployment_id


def test_protocol_deploy_builds_bounded_durable_plan():
    runtime = _RuntimeState()
    store = ProtocolDeploymentStore(runtime)
    deployment_id = _configured_state(store)

    store.configure(
        deployment_id,
        requirements=[
            {
                "name": "Workstation routine",
                "channel_ids": [12],
                "expected_routine": "person usually seated at workstation",
                "unexpected_severity": "info",
                "novelty_sensitivity": "balanced",
                "alerts": [
                    {
                        "name": "Workstation occupancy",
                        "description": "Track visible occupied and empty desk states",
                        "severity": "low",
                        "positive_query": "person seated at occupied workstation",
                        "contrast_query": "empty chair at workstation",
                        "counter_mode": "count_and_duration",
                        "positive_label": "occupied",
                        "negative_label": "away",
                        "count_transition": "positive_to_negative",
                        "duration_state": "positive",
                    }
                ],
            },
            {
                "name": "Gate routine",
                "channel_ids": [11],
                "unexpected_severity": "normal",
                "novelty_sensitivity": "high",
                "alerts": [],
            },
        ],
        quiet_window={
            "enabled": True,
            "timezone": "Europe/Riga",
            "start_local": "02:00",
            "end_local": "05:00",
            "days": [0, 1, 2, 3, 4, 5, 6],
        },
    )
    planned = store.build_plan(deployment_id)
    plan = planned["plan"]

    assert planned["stage"] == "plan_ready"
    assert len(plan["channels"]) == 2
    assert plan["groups"] == [
        {"name": "Perimeter", "channel_ids": [11]},
        {"name": "Operations", "channel_ids": [12]},
    ]
    assert len(plan["probes"]) == 1
    assert len(plan["counted_states"]) == 1
    assert plan["probes"][0]["metric_profile_id"] == plan["counted_states"][0]["id"]
    assert "operator" not in plan["channels"][1]["alert_policy_prompt"].lower()
    assert plan["quiet_window"]["timezone"] == "Europe/Riga"

    receipt = {"status": "applied"}
    store.mark_applied(deployment_id, receipt=receipt)
    reloaded = ProtocolDeploymentStore(runtime).load(deployment_id)
    assert reloaded["stage"] == "commissioning_pending"
    assert reloaded["commissioning"]["status"] == "pending"


def test_protocol_deploy_rejects_more_than_eight_channels_and_bad_duration_state():
    store = ProtocolDeploymentStore()
    state = store.start(
        [{"id": index, "title": str(index)} for index in range(1, 10)],
        resume_latest=False,
    )
    with pytest.raises(DeploymentWorkflowError, match="at most 8"):
        store.configure(state["deployment_id"], channel_ids=list(range(1, 10)))

    store.configure(state["deployment_id"], channel_ids=[1])
    with pytest.raises(DeploymentWorkflowError, match="duration_state"):
        store.configure(
            state["deployment_id"],
            requirements=[
                {
                    "name": "bad metric",
                    "channel_ids": [1],
                    "alerts": [
                        {
                            "name": "state",
                            "description": "state",
                            "severity": "low",
                            "positive_query": "occupied chair",
                            "contrast_query": "empty chair",
                            "counter_mode": "measure_duration",
                            "duration_state": "invented",
                        }
                    ],
                }
            ],
        )


def test_protocol_deploy_drops_overlapping_duplicate_requirement_pack():
    store = ProtocolDeploymentStore()
    deployment_id = _configured_state(store)
    configured = store.configure(
        deployment_id,
        requirements=[
            {
                "name": "Operations",
                "channel_ids": [12],
                "alerts": [
                    {
                        "name": "Occupancy",
                        "description": "Visible workstation occupancy",
                        "severity": "log",
                        "positive_query": "person at workstation",
                        "contrast_query": "empty workstation",
                        "counter_mode": "count_transitions",
                        "duration_state": "positive",
                    }
                ],
            },
            {
                "name": "quiet_window",
                "channel_ids": [11, 12],
                "alerts": [
                    {
                        "name": "Occupancy",
                        "description": "Visible workstation occupancy",
                        "severity": "log",
                        "positive_query": "person at workstation",
                        "contrast_query": "empty workstation",
                        "counter_mode": "count_transitions",
                        "duration_state": "positive",
                    }
                ],
            },
        ],
    )

    assert len(configured["requirements"]) == 1
    assert configured["requirements"][0]["alerts"][0]["counter_mode"] == "count_and_duration"
    assert configured["requirement_warnings"]
    assert "quiet window is a separate field" in configured["requirement_warnings"][0]


def test_maritime_deploy_builds_ptz_prompts_and_shadow_starter_probes():
    store = ProtocolDeploymentStore()
    state = store.start(
        [
            {"id": 41, "title": "Sea gate"},
            {"id": 42, "title": "West beach"},
        ],
        deployment_profile="maritime",
        resume_latest=False,
    )
    deployment_id = state["deployment_id"]
    configured = store.configure(
        deployment_id,
        channel_ids=[41, 42],
        channel_roles=[
            {
                "channel_id": 41,
                "role": "maritime_gate",
                "location": "Liepaja north gate",
            },
            {"channel_id": 42, "role": "maritime_coast"},
        ],
        starter_policy_mode="shadow",
        quiet_window={
            "enabled": True,
            "timezone": "Europe/Riga",
            "start_local": "01:00",
            "end_local": "04:00",
            "days": [0, 1, 2, 3, 4, 5, 6],
        },
    )

    planned = store.build_plan(deployment_id)
    plan = planned["plan"]

    assert configured["deployment_profile"] == "maritime"
    assert plan["deployment_profile"] == "maritime"
    assert len(plan["channels"]) == 2
    assert len(plan["probes"]) == 8
    assert all(probe["attention_only"] for probe in plan["probes"])
    gate = next(row for row in plan["channels"] if row["channel_id"] == 41)
    assert gate["channel_role"] == "maritime_gate"
    assert "Camera-global motion is not vessel motion" in gate["stream_system_prompt"]
    assert "Liepaja north gate" in gate["stream_system_prompt"]
    assert "not observed" in gate["rollup_prompts"]["L3"]
    assert plan["quiet_window"]["timezone"] == "Europe/Riga"


def test_maritime_deploy_requires_a_role_for_every_selected_channel():
    store = ProtocolDeploymentStore()
    state = store.start(
        [{"id": 41, "title": "Sea gate"}, {"id": 42, "title": "Beach"}],
        deployment_profile="maritime",
        resume_latest=False,
    )
    store.configure(
        state["deployment_id"],
        channel_ids=[41, 42],
        channel_roles=[{"channel_id": 41, "role": "maritime_gate"}],
        starter_policy_mode="shadow",
    )

    with pytest.raises(DeploymentWorkflowError, match="assign a maritime role"):
        store.build_plan(state["deployment_id"])


def test_maritime_composite_apply_installs_prompt_layers_and_shadow_probes(monkeypatch):
    store = ProtocolDeploymentStore()
    state = store.start(
        [{"id": 11, "title": "Sea gate"}],
        deployment_profile="maritime",
        resume_latest=False,
    )
    deployment_id = state["deployment_id"]
    store.configure(
        deployment_id,
        channel_ids=[11],
        channel_roles=[{
            "channel_id": 11,
            "role": "maritime_gate",
            "location": "West coast gate",
        }],
        starter_policy_mode="shadow",
    )
    manager = _Manager()
    probes = _ProbeStore()
    tools = AgentTools(
        detections_store=_DetectionStore(),
        probes_store=probes,
        luxriot_manager=manager,
        embed_text_fn=lambda _text: None,
        embed_image_fn=lambda _image: None,
        call_lm_fn=lambda *_args, **_kwargs: "",
        encode_jpeg_fn=lambda *_args, **_kwargs: "",
        search_indexed_folder_fn=lambda **_kwargs: [],
        search_detections_fn=lambda **_kwargs: [],
        deployment_store=store,
    )
    monkeypatch.setattr(tools, "_schedule_deployment_commissioning", lambda _deployment_id: None)

    applied = tools.execute(
        "apply_deployment_plan",
        {"deployment_id": deployment_id, "preview": False},
    )

    assert applied["status"] == "applied"
    assert "KEEP STREAM CORE" in manager.prompts[11]["stream_system_prompt"]
    assert "visual aggregation core" in manager.prompts[11]["stream_system_prompt"]
    assert "eight-hour maritime consolidation" in manager.prompts[11]["rollup_prompts"]["L3"]
    assert len(probes.rows) == 4
    assert all(probe["attention_only"] for probe in probes.rows)


def test_counted_metric_keeps_alert_delivery_and_unknown_time_out_of_count():
    profile = {
        "id": "metric-1",
        "name": "Workstation occupancy",
        "channel_id": 12,
        "positive_label": "occupied",
        "negative_label": "away",
        "counter_mode": "count_and_duration",
        "count_transition": "positive_to_negative",
        "duration_state": "positive",
    }
    result = aggregate_counted_state_metric(
        profile,
        {
            "time_window": {"duration_sec": 70},
            "coverage": {"status": "partial"},
            "frame_count": 65,
            "transitions": [
                {"from_state": "occupied", "to_state": "away"},
                {"from_state": "away", "to_state": "occupied"},
                {"from_state": "occupied", "to_state": "away"},
            ],
            "segments": [
                {"state": "occupied", "duration_sec": 30},
                {"state": "away", "duration_sec": 20},
                {"state": "unknown", "duration_sec": 10},
            ],
        },
    )

    assert result["event_count"] == 2
    assert result["duration_sec"] == 30
    assert result["duration_human"] == "30.0 s"
    assert result["unknown_duration_sec"] == 10
    assert "cooldown" in result["notes"]


def test_composite_apply_is_idempotent_and_preserves_existing_alert_policy(monkeypatch):
    runtime = _RuntimeState()
    deployment_store = ProtocolDeploymentStore(runtime)
    deployment_id = _configured_state(deployment_store)
    deployment_store.configure(
        deployment_id,
        requirements=[
            {
                "name": "Gate",
                "channel_ids": [11],
                "unexpected_severity": "normal",
                "novelty_sensitivity": "balanced",
                "alerts": [
                    {
                        "name": "Gate occupied",
                        "description": "A person is visibly waiting at the gate",
                        "severity": "low",
                        "positive_query": "person waiting at gate",
                        "contrast_query": "clear unattended gate",
                        "counter_mode": "count_transitions",
                    }
                ],
            }
        ],
        quiet_window={
            "enabled": True,
            "timezone": "Europe/Riga",
            "start_local": "02:00",
            "end_local": "05:00",
            "days": [0, 1, 2, 3, 4, 5, 6],
        },
    )
    manager = _Manager()
    probes = _ProbeStore()
    groups = _GroupStore()
    tools = AgentTools(
        detections_store=_DetectionStore(),
        probes_store=probes,
        luxriot_manager=manager,
        embed_text_fn=lambda _text: None,
        embed_image_fn=lambda _image: None,
        call_lm_fn=lambda *_args, **_kwargs: "",
        encode_jpeg_fn=lambda *_args, **_kwargs: "",
        search_indexed_folder_fn=lambda **_kwargs: [],
        search_detections_fn=lambda **_kwargs: [],
        channel_group_store=groups,
        deployment_store=deployment_store,
    )
    monkeypatch.setattr(
        tools,
        "_schedule_deployment_commissioning",
        lambda _deployment_id: None,
    )

    preview = tools.execute(
        "apply_deployment_plan",
        {"deployment_id": deployment_id, "preview": True},
    )
    assert preview["status"] == "preview"
    assert preview["diff"]["probe_count"] == 1

    applied = tools.execute(
        "apply_deployment_plan",
        {"deployment_id": deployment_id, "preview": False},
    )
    assert applied["status"] == "applied"
    assert "KEEP EXISTING POLICY" in manager.prompts[11]["alert_policy_prompt"]
    assert f"EVA_PROTOCOL_DEPLOY:{deployment_id}:BEGIN" in manager.prompts[11][
        "alert_policy_prompt"
    ]
    assert len(probes.rows) == 1
    assert len(groups.rows) == 2
    assert set(manager.sessions) == {11, 12}
    assert manager.schedule["timezone"] == "Europe/Riga"
    assert len(deployment_store.list_counted_profiles(channel_id=11)) == 1

    # A retry after a lost receipt updates the marked section and same probe.
    applied_again = tools.execute(
        "apply_deployment_plan",
        {"deployment_id": deployment_id, "preview": False},
    )
    assert applied_again["status"] == "applied"
    assert len(probes.rows) == 1
    assert (
        manager.prompts[11]["alert_policy_prompt"].count(
            f"EVA_PROTOCOL_DEPLOY:{deployment_id}:BEGIN"
        )
        == 1
    )


def test_first_commissioning_pass_runs_l1_and_returns_proposals_only(monkeypatch):
    deployment_store = ProtocolDeploymentStore()
    deployment_id = _configured_state(deployment_store)
    deployment_store.configure(
        deployment_id,
        requirements=[
            {
                "name": "Gate",
                "channel_ids": [11],
                "alerts": [
                    {
                        "name": "Gate occupied",
                        "description": "Person waiting at gate",
                        "severity": "low",
                        "positive_query": "person waiting at gate",
                        "contrast_query": "clear unattended gate",
                    }
                ],
            }
        ],
    )
    deployment_store.build_plan(deployment_id)
    tools = AgentTools(
        detections_store=_DetectionStore(),
        probes_store=_ProbeStore(),
        luxriot_manager=_Manager(),
        embed_text_fn=lambda _text: None,
        embed_image_fn=lambda _image: None,
        call_lm_fn=lambda *_args, **_kwargs: "",
        encode_jpeg_fn=lambda *_args, **_kwargs: "",
        search_indexed_folder_fn=lambda **_kwargs: [],
        search_detections_fn=lambda **_kwargs: [],
        deployment_store=deployment_store,
    )
    deployment_store.mark_applied(
        deployment_id,
        receipt={"status": "applied"},
    )
    monkeypatch.setattr(
        tools,
        "_calibrate_probe_from_archive",
        lambda _args: {
            "contrast_query_effective": "clear unattended gate",
            "channels": [
                {
                    "suggested_thresholds": {
                        "safe_to_apply": True,
                        "pos_floor": 0.24,
                        "margin_thr": 0.06,
                        "separation_quality": "good",
                        "recommended_action": "review_then_apply",
                    },
                    "warnings": [],
                }
            ],
        },
    )
    monkeypatch.setattr(
        tools,
        "_track_visual_state_transitions",
        lambda _args: {
            "counts": {"appearance_count": 2, "disappearance_count": 1},
            "transitions": [
                {
                    "from_state": "negative",
                    "to_state": "positive",
                    "to_ms": 1_000,
                },
                {
                    "from_state": "positive",
                    "to_state": "negative",
                    "to_ms": 20_000,
                },
                {
                    "from_state": "negative",
                    "to_state": "positive",
                    "to_ms": 101_000,
                },
            ],
        },
    )

    result = tools._run_deployment_commissioning(deployment_id)

    assert result["status"] == "complete"
    assert result["proposal_only"] is True
    assert result["l1_reviews"][0]["generation_status"] == "generated"
    proposal = result["proposals"][0]
    assert proposal["status"] == "threshold_proposal"
    assert proposal["cadence_proposal"]["median_episode_gap_sec"] == 100
    changes = proposal["recommended_probe_args"]["changes"]
    assert changes["pos_floor"] == 0.24
    assert changes["margin_thr"] == 0.06
    assert changes["bookmark_dedupe_window_sec"] == 40
    assert changes["bookmark_cooldown_sec"] == 20
