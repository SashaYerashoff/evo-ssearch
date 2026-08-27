import ast
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

from luxriot_connector import (
    DEFAULT_ALERTS_JSON_PROMPT,
    LuxriotManager,
    compact_alert_topic,
)


def build_manager(directory: Path, alert_parser=None, **config_overrides: Any) -> LuxriotManager:
    config = SimpleNamespace(
        LUXRIOT_SYSTEM_PROMPT_DEFAULT="Describe only what is visible in the current frames.",
        LUXRIOT_ALERTS_JSON_PROMPT="",
        LUXRIOT_SUMMARY_HISTORY_LIMIT=100,
        LUXRIOT_SUMMARY_RETENTION_DAYS=0,
        LUXRIOT_AUTO_BOOKMARKS=False,
        LUXRIOT_BOOKMARK_COOLDOWN_SEC=60.0,
        LUXRIOT_ALERTS_MAX_PER_BATCH=8,
        LUXRIOT_SUMMARY_STATE_FILE=str(directory / "summaries.json"),
        LUXRIOT_ROLLUP_CACHE_FILE=str(directory / "rollups.json"),
        LUXRIOT_ROLLUP_L1_LLM_ENABLED=False,
        LUXRIOT_ROLLUP_LLM_LEVELS="",
        LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS=8000,
        LUXRIOT_ROLLUP_LLM_CHAR_BUDGET=12000,
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL=1,
        LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT=100,
        LUXRIOT_ROLLUP_TIME_ONLY=True,
        LUXRIOT_SNAPSHOT_INTERVAL=5,
        LUXRIOT_SNAPSHOT_MAX_EDGE=800,
        LUXRIOT_MAX_BUFFER_FRAMES=180,
        LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH=4,
        LUXRIOT_BASE_URL="http://luxriot.invalid",
        LUXRIOT_USERNAME="",
        LUXRIOT_PASSWORD="",
    )
    for key, value in config_overrides.items():
        setattr(config, key, value)
    return LuxriotManager(
        config=config,
        lm_callback=lambda _messages, _model: "summary",
        message_builder=lambda _channel, frames, prompt, system_prompt: [
            {
                "frame_count": len(frames),
                "prompt": prompt,
                "system_prompt": system_prompt,
            }
        ],
        jpeg_encoder=lambda _image, **_kwargs: "jpeg",
        alert_parser=alert_parser,
    )


def load_lm_alert_parser():
    source = Path(__file__).resolve().parent.parent.joinpath("oldapp.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    namespace = {
        "time": time,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Sequence": Sequence,
        "Set": Set,
        "Tuple": Tuple,
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_lm_alerts":
            exec(compile(ast.Module([node], []), "oldapp.py", "exec"), namespace)
            return namespace["_parse_lm_alerts"]
    raise AssertionError("_parse_lm_alerts not found")


def install_channel_memory(manager: LuxriotManager, channel_id: int = 7) -> None:
    now = time.time()
    manager._update_channel_routine_context(
        channel_id=channel_id,
        rollup_id="l1-memory",
        window_end=now - 1.0,
        level="L1",
        memory_update={
            "active_watchlist": ["watch the east gate"],
            "preserved_deviations": [
                "vehicle drifting was seen in a prior window"
            ],
        },
        summary_text="",
    )
    manager._update_channel_routine_context(
        channel_id=channel_id,
        rollup_id="l2-memory",
        window_end=now,
        level="L2",
        summary_text=(
            "MEMORY_UPDATE_JSON:\n"
            "{"
            "\"routine_baseline\":\"parking lot is usually empty overnight\","
            "\"active_watchlist\":[\"watch the east gate\"],"
            "\"preserved_deviations\":[\"vehicle drifting was seen in a prior window\"],"
            "\"alert_tuning_notes\":[\"keep drifting visible\"],"
            "\"ignore_as_routine\":[\"parked maintenance vehicles\"]"
            "}"
        ),
    )


class VlmAlertPromptContractTests(unittest.TestCase):
    def test_bookmark_topic_is_generic_and_never_exceeds_three_words(self):
        self.assertEqual(
            compact_alert_topic(
                "Snapshot 2: Red car drifts sharply right, tires kicking up smoke."
            ),
            "Red car drifts",
        )
        self.assertEqual(
            compact_alert_topic("ALERT — Person giving thumbs-up gesture: visible evidence"),
            "Person giving thumbs-up",
        )

    def test_l0_output_budget_cannot_exceed_actual_generation_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                LUXRIOT_L0_OUTPUT_BUDGET_TOKENS=1536,
                LM_VIDEO_MAX_TOKENS=512,
            )

        self.assertEqual(manager.l0_prompt_budget.max_output_tokens, 512)

    def test_batch_contract_prioritizes_alerts_before_optional_memory(self):
        schema = DEFAULT_ALERTS_JSON_PROMPT.split(
            "Required top-level key order:",
            1,
        )[1]

        self.assertLess(schema.index("alerts"), schema.index("events"))
        self.assertLess(schema.index("alerts"), schema.index("memory_pass"))
        self.assertIn("roughly 100-220 words", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("Snapshot N:", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("for EVERY supplied snapshot", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("ALERT — <title>", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("### Scene, ### Episode, ### Alerts, ### Routine", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("### Deviations, ### Worth to remember", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("one COMPLETE, compact JSON object", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn(
            "always finish with literal BATCH_STATE_JSON:",
            DEFAULT_ALERTS_JSON_PROMPT,
        )
        self.assertIn("including empty arrays", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("only allowed top-level JSON keys", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("probe legends", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("under 12 words", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertIn("Never claim complete coverage", DEFAULT_ALERTS_JSON_PROMPT)
        self.assertTrue(
            DEFAULT_ALERTS_JSON_PROMPT.rstrip().endswith(
                '"scene":{},"routines":[],"memory_pass":[]}'
            )
        )

    def test_batch_state_recovers_complete_alert_from_truncated_json_prefix(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
        ]
        summary = (
            "A person gives a thumbs-up gesture.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[{"title":"Thumbs up","severity":"info",'
            '"state":"new","channel_id":7,"timestamp_ms":0,'
            '"snapshot_indices":[2]}],"events":['
        )

        state = LuxriotManager._extract_batch_state(summary, frames)

        self.assertEqual(state["contract_status"], "partial_prefix")
        self.assertEqual(len(state["alerts"]), 1)
        self.assertEqual(state["alerts"][0]["title"], "Thumbs up")
        self.assertEqual(state["alerts"][0]["snapshot_indices"], [2])

    def test_batch_state_recovers_markerless_v2_prefix_and_strips_prompt_leak(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
        ]
        summary = (
            "### Scene description\n"
            "A person gives a thumbs-up. Coverage is complete and matched.\n\n"
            "### Episode update\nThumbs-up remains visible.\n\n"
            '{"version":2,"alerts":[{"title":"Thumbs up",'
            '"description":"Visible thumbs-up gesture", "severity":"info",'
            '"snapshot_indices":[2]}],"events":[],"observed_states":[],'
            '"cover":{"snapshot_index":2,"kind":"event"},'
            '"scene":{"status":"matched","summary":"Person at desk"},'
            '"routines":[],"memory_pass":[],"batch_end_ms":101000,'
            '"clip_probe_legend":['
        )

        state = LuxriotManager._extract_batch_state(summary, frames)
        canonical = LuxriotManager._render_reconciled_batch_state_summary(
            summary,
            state,
        )

        self.assertEqual(state["contract_status"], "partial_prefix_markerless")
        self.assertEqual(state["alerts"][0]["title"], "Thumbs up")
        self.assertIn("BATCH_STATE_JSON:", canonical)
        self.assertNotIn("clip_probe_legend", canonical)
        self.assertNotIn("batch_end_ms", canonical)
        self.assertNotIn("Coverage is complete", canonical)
        payload = json.loads(canonical.split("BATCH_STATE_JSON:\n", 1)[1])
        self.assertEqual(
            list(payload),
            [
                "version",
                "alerts",
                "events",
                "observed_states",
                "cover",
                "scene",
                "routines",
                "memory_pass",
            ],
        )
        self.assertEqual(payload["alerts"][0]["snapshot_indices"], [2])

    def test_batch_state_does_not_treat_markerless_input_metadata_as_state(self):
        frames = [{"thumbnail": "frame-one", "captured_at": 100.0}]
        summary = (
            "Scene notes.\n"
            '{"version":2,"channel_id":7,"batch_end_ms":100000,'
            '"clip_probe_legend":[]}'
        )

        state = LuxriotManager._extract_batch_state(summary, frames)

        self.assertEqual(state["contract_status"], "missing_fallback")

    def test_batch_state_accepts_complete_markerless_ordered_v2_object(self):
        frames = [{"thumbnail": "frame-one", "captured_at": 100.0}]
        summary = (
            "A person remains at the desk.\n"
            '{"version":2,"alerts":[],"events":[],"observed_states":[],'
            '"cover":{"snapshot_index":1,"kind":"routine"},'
            '"scene":{"status":"matched","summary":"Person at desk"},'
            '"routines":[],"memory_pass":[]}'
        )

        state = LuxriotManager._extract_batch_state(summary, frames)
        canonical = LuxriotManager._render_reconciled_batch_state_summary(
            summary,
            state,
        )

        self.assertEqual(state["contract_status"], "parsed_markerless")
        self.assertIn("BATCH_STATE_JSON:", canonical)
        self.assertEqual(
            json.loads(canonical.split("BATCH_STATE_JSON:\n", 1)[1])["version"],
            2,
        )

    def test_truncated_event_prefix_can_reconcile_explicit_operator_alert(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
        ]
        summary = (
            "A person gives a thumbs-up gesture.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[],"events":[{"event_id":"thumbs_up",'
            '"label":"person gives thumbs up","state":"new",'
            '"snapshot_indices":[2],"summary":"Visible thumbs-up gesture",'
            '"novelty":"novel","pass_up":true}],"observed_states":['
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Alert if a person shows a thumbs-up gesture, severity info."
                ),
            )
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

        self.assertEqual(state["contract_status"], "partial_prefix")
        self.assertEqual(reconciled["contract_status"], "parsed_alert_reconciled")
        self.assertEqual(reconciled["alerts"][0]["severity"], "info")
        self.assertIn("BATCH_STATE_JSON:", patched_summary)

    def test_make_alert_on_policy_promotes_grounded_episode_observation(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(6)
        ]
        summary = (
            "### Scene\nA person sits at a desk.\n\n"
            "### Episode\n"
            "Snapshot 1: Person seated, hands resting.\n"
            "Snapshot 2: Raises right hand, thumb up.\n"
            "Snapshot 3: Hand remains raised.\n"
            "Snapshot 4: Lowering hand.\n\n"
            "### Alerts\nNone\n\n"
            "### Routine\nSeated desk work.\n\n"
            "### Deviations\nThumb-up gesture (Snap 2).\n\n"
            "### Worth to remember\nNone\n\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[],"events":[],"observed_states":[],'
            '"cover":{"snapshot_index":3,"kind":"routine","confidence":"low"},'
            '"scene":{"status":"uncertain","summary":""},'
            '"routines":[],"memory_pass":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=112,
                alert_policy_prompt=(
                    "Make alert on thumbs up gesture, severity - critical"
                ),
            )
            state = manager._extract_batch_state(summary, frames)
            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                112,
                summary,
                state,
            )

        self.assertEqual(reconciled["contract_status"], "parsed_alert_reconciled")
        self.assertEqual(reconciled["alerts"][0]["title"], "thumbs up gesture")
        self.assertEqual(reconciled["alerts"][0]["severity"], "critical")
        self.assertEqual(reconciled["alerts"][0]["snapshot_indices"], [2])
        self.assertIn("BATCH_STATE_JSON:", patched_summary)

    def test_grounded_plaintext_alert_repairs_empty_json_alert_array(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(4)
        ]
        summary = (
            "### Scene description\nA person sits at a desk.\n\n"
            "### Episode update\n"
            "Snapshot 1: Person is seated.\n"
            "Snapshot 2: Person raises two fingers.\n"
            "Snapshot 3: The V hand gesture remains visible.\n"
            "Snapshot 4: The hand lowers.\n"
            "ALERT — Victory gesture: person makes a visible victory hand gesture "
            "(snapshots 2,3).\n\n"
            "### Routine and deviations\nRoutine desk work; gesture is a deviation.\n\n"
            "### Worth to remember\nNone\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[],"events":[],"observed_states":[],'
            '"cover":{"snapshot_index":2,"kind":"event","confidence":"high"},'
            '"scene":{"status":"matched","summary":"Person at desk"},'
            '"routines":[],"memory_pass":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Alert when a person makes a victory gesture, severity info."
                ),
            )
            state = manager._extract_batch_state(summary, frames)
            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

        self.assertEqual(reconciled["contract_status"], "parsed_alert_reconciled")
        self.assertEqual(
            reconciled["alert_reconciliation"]["source"],
            "grounded_plaintext_alert",
        )
        self.assertEqual(reconciled["alerts"][0]["snapshot_indices"], [2, 3])
        self.assertIn("BATCH_STATE_JSON:", patched_summary)
    def test_final_live_prompt_places_prior_memory_before_batch_state_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            install_channel_memory(manager)
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt="Watch for visible public safety hazards near the entrance.",
            )

            base_prompt = manager.get_effective_stream_system_prompt(7)
            final_prompt = manager.compose_live_system_prompt(7, base_prompt)

            self.assertIn("Alert review policy", final_prompt)
            self.assertIn("Active Channel Memory", final_prompt)
            self.assertIn("BATCH_STATE_JSON:", final_prompt)
            self.assertLess(
                final_prompt.index("Alert review policy"),
                final_prompt.index("Active Channel Memory"),
                "operator alert criteria should be separate from routine memory/prior",
            )
            self.assertLess(
                final_prompt.index("Active Channel Memory"),
                final_prompt.index("BATCH_STATE_JSON:"),
                "channel memory/prior must be before the final BATCH_STATE_JSON output contract",
            )
            self.assertLess(
                final_prompt.index("Current-batch observation contract"),
                final_prompt.index("BATCH_STATE_JSON:"),
                "BATCH_STATE_JSON must stay last after current-state instructions",
            )

    def test_vector_signal_prompt_marks_cues_as_attention_not_visual_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            install_channel_memory(manager)

            final_prompt = manager.compose_live_system_prompt(
                7,
                manager.get_effective_stream_system_prompt(7),
                vector_signal={
                    "version": 1,
                    "channel_id": 7,
                    "semantics": "vector_homeostasis_attention_signal_not_visual_proof",
                    "clip_probe_signals": [
                        {
                            "name": "vehicle drift candidate",
                            "probe_id": "probe-drift",
                            "p": 0.42,
                            "n": 0.18,
                            "m": 0.24,
                            "apex_frame": 2,
                        }
                    ],
                },
            )

            self.assertIn("VECTOR_SIGNALS_JSON", final_prompt)
            self.assertIn("secondary attention/arousal signal", final_prompt)
            self.assertIn("not visual proof", final_prompt)
            self.assertLess(final_prompt.index("Active Channel Memory"), final_prompt.index("VECTOR_SIGNALS_JSON"))
            self.assertLess(final_prompt.index("VECTOR_SIGNALS_JSON"), final_prompt.index("Current-batch observation contract"))
            self.assertLess(final_prompt.index("VECTOR_SIGNALS_JSON"), final_prompt.index("BATCH_STATE_JSON:"))

    def test_default_alert_contract_has_no_private_scene_entities(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            final_prompt = manager.compose_live_system_prompt(7, manager.get_effective_stream_system_prompt(7))
            lowered = final_prompt.lower()

            for forbidden in ("orlandina", "sphynx", "union jack", "british flag", "sasha"):
                self.assertNotIn(forbidden, lowered)

    def test_unified_batch_state_preserves_cover_and_evidence_references(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
            {"thumbnail": "frame-three", "captured_at": 102.0},
        ]
        summary = (
            "A person enters through the gate.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,'
            '"cover":{"snapshot_index":2,"kind":"transition","reason":"clearest gate crossing","confidence":"high"},'
            '"scene":{"status":"matched","summary":"Gate entrance"},'
            '"events":[{"event_id":"gate-entry","label":"person enters","state":"new",'
            '"snapshot_indices":[1,2],"summary":"Person crosses the gate","novelty":"novel","pass_up":true}],'
            '"observed_states":[],"routines":[],"memory_pass":["gate entry"],'
            '"alerts":[{"title":"Gate entry","severity":"low","snapshot_indices":[2,3]}]}'
        )

        state = LuxriotManager._extract_batch_state(summary, frames)
        archived = LuxriotManager._summary_archive_frames(
            frames,
            batch_start_ms=100000,
            batch_end_ms=102000,
            sample_count=2,
            batch_state=state,
        )

        self.assertEqual(state["contract_status"], "parsed")
        self.assertEqual(state["cover"]["snapshot_index"], 2)
        self.assertEqual(state["cover"]["source"], "model")
        self.assertEqual(state["alerts"][0]["snapshot_indices"], [2, 3])
        self.assertEqual(
            [frame["snapshot_index"] for frame in archived],
            [1, 2, 3],
        )
        self.assertTrue(archived[1]["is_cover"])

    def test_batch_state_accepts_marker_alias_and_fails_conflicting_state_closed(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
        ]
        summary = (
            "Current scene.\n"
            "BATCHSTATEJSON:\n"
            '{"version":1,"cover":{"snapshot_index":1},'
            '"events":[],"observed_states":[{"key":"cat_orlandina",'
            '"label":"Cat Orlandina","state":"absent","snapshot_indices":[1,2],'
            '"evidence":"Cat is visible on the shelf in both snapshots"}],'
            '"routines":["routine visibly reinforced in this batch"],'
            '"memory_pass":["grounded item important for later consolidation"],'
            '"alerts":[]}'
        )

        state = LuxriotManager._extract_batch_state(summary, frames)

        self.assertEqual(state["contract_status"], "parsed")
        self.assertEqual(state["observed_states"][0]["state"], "unknown")
        self.assertEqual(
            state["observed_states"][0]["validation_issues"],
            ["state_evidence_conflict"],
        )
        self.assertEqual(state["routines"], [])
        self.assertEqual(state["memory_pass"], [])

    def test_batch_state_recovers_complete_terminal_json_fence_without_marker(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
        ]
        summary = (
            "A cat remains on the shelf.\n\n"
            "```json\n"
            '{"version":1,"cover":{"snapshot_index":1,"kind":"routine"},'
            '"scene":{"status":"matched","summary":"Cat on shelf"},'
            '"events":[],"observed_states":[{"key":"cat","label":"cat",'
            '"state":"present","snapshot_indices":[1],'
            '"evidence":"Cat visible on shelf."}],'
            '"routines":["Cat resting"],"memory_pass":[],"alerts":[]}\n'
            "```"
        )

        state = LuxriotManager._extract_batch_state(summary, frames)
        canonical = LuxriotManager._render_reconciled_batch_state_summary(
            summary,
            state,
        )

        self.assertEqual(state["contract_status"], "parsed_terminal_fence")
        self.assertEqual(state["cover"]["snapshot_index"], 1)
        self.assertEqual(state["observed_states"][0]["state"], "present")
        self.assertIn("BATCH_STATE_JSON:", canonical)
        self.assertNotIn("```", canonical)
        rendered_json = canonical.split("BATCH_STATE_JSON:\n", 1)[1]
        self.assertLess(rendered_json.index('"alerts"'), rendered_json.index('"events"'))
        self.assertLess(rendered_json.index('"alerts"'), rendered_json.index('"cover"'))

    def test_batch_state_rejects_unrelated_terminal_json_fence(self):
        frames = [{"thumbnail": "frame-one", "captured_at": 100.0}]
        summary = "Scene notes.\n```json\n{\"objects\":[\"cat\"]}\n```"

        state = LuxriotManager._extract_batch_state(summary, frames)

        self.assertEqual(state["contract_status"], "missing_fallback")
        self.assertEqual(state["observed_states"], [])

    def test_batch_state_rejects_prose_state_conflict_and_ungrounded_absence(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
        ]
        summary = (
            "No visual evidence of a person named Sasha in the current snapshots.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":1},"events":[],'
            '"observed_states":['
            '{"key":"person_sasha","label":"person Sasha","state":"present",'
            '"snapshot_indices":[1,2],"evidence":"person visible in both snapshots"},'
            '{"key":"cat_orlandina","label":"cat Orlandina","state":"absent",'
            '"snapshot_indices":[1,2],"evidence":"cat not visible"}'
            '],"routines":[],"memory_pass":[],"alerts":[]}'
        )

        state = LuxriotManager._extract_batch_state(summary, frames)
        by_key = {item["key"]: item for item in state["observed_states"]}

        self.assertEqual(by_key["person sasha"]["state"], "unknown")
        self.assertIn(
            "summary_state_conflict",
            by_key["person sasha"]["validation_issues"],
        )
        self.assertEqual(by_key["cat orlandina"]["state"], "unknown")
        self.assertIn(
            "absent_scope_unverified",
            by_key["cat orlandina"]["validation_issues"],
        )

    def test_state_conflict_does_not_cross_sentence_boundaries(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
        ]
        summary = (
            "No other people or animals are visible. "
            "A grey Sphynx cat (Orlandina) is visible on the shelf.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":1},"events":[],'
            '"observed_states":[{"key":"cat_orlandina",'
            '"label":"Sphynx cat (Orlandina)","state":"present",'
            '"snapshot_indices":[1,2],'
            '"evidence":"Grey Sphynx cat visible on shelf in both snapshots"}],'
            '"routines":[],"memory_pass":[],"alerts":[]}'
        )

        state = LuxriotManager._extract_batch_state(summary, frames)

        self.assertEqual(state["observed_states"][0]["state"], "present")
        self.assertNotIn(
            "validation_issues",
            state["observed_states"][0],
        )

    def test_alert_policy_prompt_is_separate_from_stream_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            settings = manager.update_prompt_settings(
                channel_id=7,
                stream_system_prompt="Summarize visible activity.",
                alert_policy_prompt="Pay special attention to people falling near stairs.",
            )

            self.assertEqual(settings["stream_system_prompt"], "Summarize visible activity.")
            self.assertEqual(settings["alert_policy_prompt"], "Pay special attention to people falling near stairs.")
            final_prompt = manager.compose_live_system_prompt(7, manager.get_effective_stream_system_prompt(7))
            self.assertIn("Pay special attention to people falling near stairs.", final_prompt)
            self.assertIn("BATCH_STATE_JSON:", final_prompt)
            self.assertIn("Mandatory current-batch alert reconciliation:", final_prompt)
            self.assertLess(
                final_prompt.index("Pay special attention to people falling near stairs."),
                final_prompt.index("BATCH_STATE_JSON:"),
            )
            self.assertLess(
                final_prompt.index("Mandatory current-batch alert reconciliation:"),
                final_prompt.index("BATCH_STATE_JSON:"),
            )

    def test_channel_alert_criteria_follow_memory_and_remain_adjacent_to_batch_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                stream_system_prompt="Describe visible activity.",
                alert_policy_prompt="Alert when a person drinks from the British flag mug.",
            )
            manager._render_capture_homeostasis_prompt = lambda _channel_id: (
                "CURRENT HOMEOSTASIS TEST LAYER"
            )

            final_prompt = manager.compose_live_system_prompt(
                7,
                manager.get_effective_stream_system_prompt(7),
            )

            homeostasis_at = final_prompt.index("CURRENT HOMEOSTASIS TEST LAYER")
            criteria_at = final_prompt.rindex(
                "Alert when a person drinks from the British flag mug."
            )
            contract_at = final_prompt.index("BATCH_STATE_JSON:")
            self.assertLess(homeostasis_at, criteria_at)
            self.assertLess(criteria_at, contract_at)
            self.assertEqual(
                final_prompt.count(
                    "Alert when a person drinks from the British flag mug."
                ),
                2,
            )
            self.assertIn(
                "alerts must contain the corresponding object",
                final_prompt[criteria_at:contract_at],
            )

    def test_backend_reconciles_explicit_current_event_omitted_by_small_vlm(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(4)
        ]
        summary = (
            "The person lifts a mug with a Union Jack design and takes a sip. "
            "The action is a minor, non-alertable routine.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":3,"kind":"event",'
            '"reason":"Person drinking from mug with British flag art","confidence":"high"},'
            '"scene":{"status":"matched","summary":"Person seated and drinking"},'
            '"events":[{"event_id":"drink_from_mug","label":"drinking from mug",'
            '"state":"new","snapshot_indices":[3],"summary":'
            '"Person lifts mug with British flag design to mouth and sips.",'
            '"novelty":"expected_variation","pass_up":true}],'
            '"observed_states":[],"routines":[],"memory_pass":[],'
            '"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Create alert if person shows thumbs up, or drinks from the mug "
                    "with the British flag art on it. Alert level for the thumbs up - "
                    "info, for the mug - warning"
                ),
            )
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

            self.assertEqual(reconciled["contract_status"], "parsed_alert_reconciled")
            self.assertEqual(reconciled["alert_reconciliation"]["source"], "current_structured_event")
            self.assertEqual(len(reconciled["alerts"]), 1)
            self.assertEqual(reconciled["alerts"][0]["severity"], "low")
            self.assertEqual(reconciled["alerts"][0]["snapshot_indices"], [3])
            self.assertEqual(
                reconciled["alerts"][0]["source"],
                "backend_policy_reconciliation",
            )
            self.assertEqual(
                len(manager._structured_alert_payloads(patched_summary)),
                1,
            )
            self.assertNotIn("```", patched_summary)

    def test_backend_reconciles_compact_priority_policy_from_settings(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(3)
        ]
        summary = (
            "A person gives a thumbs-up gesture.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":2,"kind":"event"},'
            '"events":[{"event_id":"thumbs_up","label":"thumbs-up gesture",'
            '"state":"new","snapshot_indices":[2],'
            '"summary":"Person visibly gives a thumbs-up gesture.",'
            '"novelty":"novel","pass_up":true}],'
            '"observed_states":[],"routines":[],"memory_pass":[],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Alert when see thumbs up gesture, priority low"
                ),
            )
            state = manager._extract_batch_state(summary, frames)
            _patched_summary, reconciled = (
                manager._reconcile_operator_alert_contract(
                    7,
                    summary,
                    state,
                )
            )

        self.assertEqual(
            reconciled["contract_status"],
            "parsed_alert_reconciled",
        )
        self.assertEqual(len(reconciled["alerts"]), 1)
        self.assertEqual(reconciled["alerts"][0]["severity"], "low")
        self.assertEqual(reconciled["alerts"][0]["snapshot_indices"], [2])
        self.assertEqual(
            reconciled["alerts"][0]["source"],
            "backend_policy_reconciliation",
        )

    def test_backend_repairs_grounded_episode_alert_omitted_from_machine_state(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(4)
        ]
        summary = (
            "### Scene description\n"
            "A person stands beside a restricted area.\n\n"
            "### Episode update\n"
            "Person enters the restricted area in snapshot 2.\n\n"
            "### Routine and deviations\n"
            "Deviation: entry into the restricted area.\n\n"
            "### Worth to remember\nNone\n\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[],"events":[],"observed_states":[],'
            '"cover":{"snapshot_index":2,"kind":"event"},'
            '"scene":{"status":"uncertain","summary":""},'
            '"routines":[],"memory_pass":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=112,
                alert_policy_prompt=(
                    "Alert when a person enters the restricted area, severity - normal"
                ),
            )
            state = manager._extract_batch_state(summary, frames)
            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                112,
                summary,
                state,
            )

        self.assertEqual(reconciled["contract_status"], "parsed_alert_reconciled")
        self.assertEqual(
            reconciled["alert_reconciliation"]["source"],
            "grounded_episode_narrative",
        )
        self.assertEqual(len(reconciled["alerts"]), 1)
        self.assertEqual(reconciled["alerts"][0]["snapshot_indices"], [2])
        self.assertEqual(reconciled["alerts"][0]["severity"], "normal")
        rendered = json.loads(patched_summary.split("BATCH_STATE_JSON:\n", 1)[1])
        self.assertEqual(len(rendered["alerts"]), 1)

    def test_backend_does_not_repair_negated_or_unanchored_episode_prose(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(3)
        ]
        machine_state = (
            'BATCH_STATE_JSON:\n{"version":2,"alerts":[],"events":[],'
            '"observed_states":[],"cover":{},"scene":{},'
            '"routines":[],"memory_pass":[]}'
        )
        summaries = (
            (
                "### Episode update\n"
                "No person enters the restricted area in snapshot 2.\n\n"
                + machine_state
            ),
            (
                "### Episode update\n"
                "Person enters the restricted area.\n\n"
                + machine_state
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=112,
                alert_policy_prompt=(
                    "Alert when a person enters the restricted area, severity - normal"
                ),
            )
            repaired = []
            for summary in summaries:
                state = manager._extract_batch_state(summary, frames)
                _patched, reconciled = manager._reconcile_operator_alert_contract(
                    112,
                    summary,
                    state,
                )
                repaired.append(reconciled)

        self.assertEqual(repaired[0]["alerts"], [])
        self.assertEqual(repaired[1]["alerts"], [])

    def test_backend_repairs_closed_episode_transition_using_valid_cover(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(4)
        ]
        scenarios = (
            (
                "Alert when a person enters the restricted area, severity - high",
                "Person enters the restricted area then exits.",
                2,
                "high",
            ),
            (
                "Alert when a vehicle door opens, severity - normal",
                "Vehicle door opens then closes.",
                3,
                "normal",
            ),
            (
                "Alert when a worker makes a warning gesture, severity - low",
                "Worker makes a warning sign then lowers the hand.",
                1,
                "low",
            ),
        )
        for policy, episode_update, cover_index, severity in scenarios:
            with self.subTest(policy=policy):
                summary = (
                    "### Scene description\nCurrent monitored scene.\n\n"
                    f"### Episode update\n{episode_update}\n\n"
                    "### Routine and deviations\nA current deviation is described above.\n\n"
                    "### Worth to remember\nNone\n\n"
                    "BATCH_STATE_JSON:\n"
                    '{"version":2,"alerts":[],"events":[],'
                    '"observed_states":[],"cover":{"snapshot_index":'
                    f"{cover_index}"
                    ',"kind":"routine","confidence":"low"},'
                    '"scene":{"status":"uncertain","summary":""},'
                    '"routines":[],"memory_pass":[]}'
                )
                with tempfile.TemporaryDirectory() as temp:
                    manager = build_manager(Path(temp))
                    manager.update_prompt_settings(
                        channel_id=112,
                        alert_policy_prompt=policy,
                    )
                    state = manager._extract_batch_state(summary, frames)
                    patched_summary, reconciled = (
                        manager._reconcile_operator_alert_contract(
                            112,
                            summary,
                            state,
                        )
                    )

                self.assertEqual(
                    reconciled["contract_status"],
                    "parsed_alert_reconciled",
                )
                self.assertEqual(
                    reconciled["alert_reconciliation"]["source"],
                    "grounded_episode_transition",
                )
                self.assertEqual(
                    reconciled["alerts"][0]["snapshot_indices"],
                    [cover_index],
                )
                self.assertEqual(reconciled["alerts"][0]["severity"], severity)
                rendered = json.loads(
                    patched_summary.split("BATCH_STATE_JSON:\n", 1)[1]
                )
                self.assertEqual(len(rendered["alerts"]), 1)

    def test_backend_does_not_repair_historical_episode_transition(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(3)
        ]
        summary = (
            "### Episode update\n"
            "In the previous batch, vehicle door opened then closed.\n\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[],"events":[],"observed_states":[],'
            '"cover":{"snapshot_index":2},"scene":{},'
            '"routines":[],"memory_pass":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=112,
                alert_policy_prompt=(
                    "Alert when a vehicle door opens, severity - normal"
                ),
            )
            state = manager._extract_batch_state(summary, frames)
            _patched, reconciled = manager._reconcile_operator_alert_contract(
                112,
                summary,
                state,
            )

        self.assertEqual(reconciled["alerts"], [])

    def test_backend_reconciles_each_distinct_operator_criterion(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(4)
        ]
        summary = (
            "A cat and a person enter; the person makes two gestures.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[],"events":['
            '{"event_id":"cat_entry","label":"cat entering",'
            '"state":"new","snapshot_indices":[1],'
            '"summary":"Cat enters the scene."},'
            '{"event_id":"person_entry","label":"person entering",'
            '"state":"new","snapshot_indices":[2],'
            '"summary":"Person enters the scene."},'
            '{"event_id":"thumbs_up","label":"thumbs-up gesture",'
            '"state":"new","snapshot_indices":[3],'
            '"summary":"Person shows a thumbs-up gesture."},'
            '{"event_id":"victory","label":"victory gesture",'
            '"state":"new","snapshot_indices":[4],'
            '"summary":"Person shows a victory gesture."}],'
            '"observed_states":[],"cover":{"snapshot_index":3,'
            '"kind":"event"},"scene":{"status":"matched"},'
            '"routines":[],"memory_pass":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Alert if cat entering or leaving scene, severity - high\n"
                    "Alert if person entering or leaving scene, severity - info\n"
                    "Alert if you spot a thumbs-up gesture, severity - info\n"
                    "Alert is you spot a victory gesture, severity - normal"
                ),
            )
            state = manager._extract_batch_state(summary, frames)
            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

        self.assertEqual(reconciled["contract_status"], "parsed_alert_reconciled")
        self.assertEqual(reconciled["alert_reconciliation"]["count"], 4)
        self.assertEqual(
            [alert["severity"] for alert in reconciled["alerts"]],
            ["high", "info", "info", "normal"],
        )
        self.assertEqual(
            [alert["snapshot_indices"] for alert in reconciled["alerts"]],
            [[1], [2], [3], [4]],
        )
        rendered_json = patched_summary.split("BATCH_STATE_JSON:\n", 1)[1]
        self.assertLess(rendered_json.index('"alerts"'), rendered_json.index('"events"'))

    def test_backend_alert_reconciliation_rejects_weak_object_only_match(self):
        frames = [
            {"thumbnail": "frame-one", "captured_at": 100.0},
            {"thumbnail": "frame-two", "captured_at": 101.0},
        ]
        summary = (
            "A person drinks from a plain mug.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":2,"kind":"event"},'
            '"events":[{"event_id":"drink","label":"drinking from mug","state":"new",'
            '"snapshot_indices":[2],"summary":"Person drinks from a plain mug.",'
            '"novelty":"routine","pass_up":false}],'
            '"observed_states":[],"routines":[],"memory_pass":[],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Create alert if the person drinks from the mug with British flag art."
                ),
            )
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

            self.assertEqual(patched_summary, summary)
            self.assertEqual(reconciled["contract_status"], "parsed")
            self.assertEqual(reconciled["alerts"], [])

    def test_backend_reconciles_validated_present_state_and_inline_severity(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(3)
        ]
        summary = (
            "A person in a white shirt is seated at a desk, then stands.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":2,"kind":"transition"},'
            '"events":[{"event_id":"stand","label":"person stands up",'
            '"state":"new","snapshot_indices":[2],'
            '"summary":"Person rises from the desk.","novelty":"novel"}],'
            '"observed_states":[{"key":"person_in_white_shirt",'
            '"label":"person in white shirt","state":"present",'
            '"snapshot_indices":[1],'
            '"evidence":"Person in white shirt is seated at desk."}],'
            '"routines":[],"memory_pass":[],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Alert when person in white shirt is sitting in front of the camera, "
                    "severity high."
                ),
            )
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

            self.assertEqual(reconciled["contract_status"], "parsed_alert_reconciled")
            self.assertEqual(
                reconciled["alert_reconciliation"]["source"],
                "current_structured_state",
            )
            self.assertEqual(reconciled["alerts"][0]["severity"], "high")
            self.assertEqual(reconciled["alerts"][0]["snapshot_indices"], [1])
            self.assertEqual(len(manager._structured_alert_payloads(patched_summary)), 1)

    def test_backend_does_not_reconcile_unknown_observed_state(self):
        frames = [{"thumbnail": "frame-one", "captured_at": 100.0}]
        summary = (
            "The watched person cannot be confirmed.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":1,"kind":"routine"},'
            '"events":[],"observed_states":[{"key":"person_in_white_shirt",'
            '"label":"person in white shirt","state":"unknown",'
            '"snapshot_indices":[1],"evidence":"View is obstructed."}],'
            '"routines":[],"memory_pass":[],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Alert when person in white shirt is sitting in front of the camera, "
                    "severity high."
                ),
            )
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

            self.assertEqual(patched_summary, summary)
            self.assertEqual(reconciled["alerts"], [])

    def test_backend_does_not_turn_negated_movement_into_operator_alert(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(2)
        ]
        summary = (
            "The cat remains stationary on the shelf.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":1,"kind":"routine"},'
            '"events":[{"event_id":"cat_static","label":"cat remains seated",'
            '"state":"continuing","snapshot_indices":[1,2],'
            '"summary":"No movement detected; cat remains in the same position.",'
            '"novelty":"routine"}],'
            '"observed_states":[{"key":"cat_orlandina","label":"cat Orlandina",'
            '"state":"present","snapshot_indices":[1,2],'
            '"evidence":"Cat visible on shelf in both snapshots."}],'
            '"routines":[],"memory_pass":[],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Monitor presence changes. Detect person movement and cat movement "
                    "as significant events requiring alert."
                ),
            )
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

            self.assertEqual(patched_summary, summary)
            self.assertEqual(reconciled["alerts"], [])

    def test_backend_alert_reconciliation_never_promotes_memory_or_prose_without_event_evidence(self):
        frames = [{"thumbnail": "frame-one", "captured_at": 100.0}]
        summary = (
            "Prior memory mentions a person drinking from a British flag mug.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":1,"kind":"routine"},'
            '"events":[],"observed_states":[],"routines":[],'
            '"memory_pass":["Person drinking from British flag mug"],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=(
                    "Create alert if the person drinks from the mug with British flag art."
                ),
            )
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = manager._reconcile_operator_alert_contract(
                7,
                summary,
                state,
            )

            self.assertEqual(patched_summary, summary)
            self.assertEqual(reconciled["alerts"], [])

    def test_backend_reconciles_grounded_drift_general_hazard(self):
        frames = [
            {"thumbnail": f"frame-{index}", "captured_at": 100.0 + index}
            for index in range(3)
        ]
        summary = (
            "An orange car drifts through the intersection with visible smoke.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":2,"kind":"event"},'
            '"events":[{"event_id":"drift","label":"orange car drifting with smoke",'
            '"state":"continuing","snapshot_indices":[2,3],'
            '"summary":"Orange car performs repeated drifting with smoke.",'
            '"novelty":"expected_variation","pass_up":true}],'
            '"observed_states":[],"routines":[],"memory_pass":[],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = (
                manager._reconcile_general_hazard_alert_contract(
                    118,
                    summary,
                    state,
                )
            )

            self.assertEqual(reconciled["contract_status"], "parsed_alert_reconciled")
            self.assertEqual(
                reconciled["alert_reconciliation"]["rule"],
                "dangerous_vehicle_behavior",
            )
            self.assertEqual(reconciled["alerts"][0]["title"], "Dangerous vehicle behavior")
            self.assertEqual(reconciled["alerts"][0]["severity"], "normal")
            self.assertEqual(reconciled["alerts"][0]["snapshot_indices"], [2, 3])
            self.assertEqual(
                reconciled["alerts"][0]["source"],
                "backend_general_hazard_reconciliation",
            )
            self.assertEqual(len(manager._structured_alert_payloads(patched_summary)), 1)

    def test_backend_general_hazard_reconciliation_ignores_routine_vehicle_crossing(self):
        frames = [{"thumbnail": "frame-one", "captured_at": 100.0}]
        summary = (
            "A car crosses the intersection normally.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":1,"kind":"routine"},'
            '"events":[{"event_id":"crossing","label":"car crossing intersection",'
            '"state":"new","snapshot_indices":[1],'
            '"summary":"Car crosses under traffic light control.",'
            '"novelty":"routine","pass_up":false}],'
            '"observed_states":[],"routines":[],"memory_pass":[],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = (
                manager._reconcile_general_hazard_alert_contract(
                    118,
                    summary,
                    state,
                )
            )

            self.assertEqual(patched_summary, summary)
            self.assertEqual(reconciled["alerts"], [])

    def test_backend_general_hazard_reconciliation_ignores_smoke_detector_object(self):
        frames = [{"thumbnail": "frame-one", "captured_at": 100.0}]
        summary = (
            "A smoke detector remains mounted above the intersection office.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":1,"kind":"routine"},'
            '"events":[{"event_id":"detector","label":"smoke detector present",'
            '"state":"continuing","snapshot_indices":[1],'
            '"summary":"Static smoke detector remains mounted on the wall.",'
            '"novelty":"routine","pass_up":false}],'
            '"observed_states":[],"routines":[],"memory_pass":[],"alerts":[]}'
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            state = manager._extract_batch_state(summary, frames)

            patched_summary, reconciled = (
                manager._reconcile_general_hazard_alert_contract(
                    118,
                    summary,
                    state,
                )
            )

            self.assertEqual(patched_summary, summary)
            self.assertEqual(reconciled["alerts"], [])

    def test_old_persisted_batch_contract_is_replaced_by_current_default(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            old_contract = (
                "BATCH_STATE_JSON:\n"
                '{"cover":{"kind":"event|transition|routine|coverage_issue",'
                '"confidence":"high|medium|low"},"alerts":[]}'
            )

            normalized = manager._normalize_json_alert_prompt(old_contract)

            self.assertNotEqual(normalized, old_contract)
            self.assertIn(
                "kind=event|transition|routine|coverage_issue",
                normalized,
            )
            self.assertIn(
                "one per distinct current visible operator criterion",
                normalized,
            )

    def test_persisted_v1_contract_is_upgraded_without_alert_policy_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            old_contract = (
                "Machine-readable current-batch state for EVA memory, navigation, "
                "and alert actions:\nBATCH_STATE_JSON:\n"
                '{"version": 1, "events": [], "observed_states": [], '
                '"routines": [], "memory_pass": [], "alerts": []}'
            )

            normalized = manager._normalize_json_alert_prompt(old_contract)

            self.assertEqual(normalized, DEFAULT_ALERTS_JSON_PROMPT)
            self.assertIn('"version":2', normalized)
            self.assertIn("state=continuing|returned|uncertain", normalized)
            self.assertIn("applies_to_event_keys", normalized)

    def test_legacy_stream_alert_prompt_returns_migration_suggestion(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            settings = manager.update_prompt_settings(
                channel_id=7,
                stream_system_prompt=(
                    "Describe visible activity.\n"
                    "Alerts:\n"
                    "Warning Level: describe risky events in prose.\n"
                    "- Alert when a person falls near stairs.\n"
                    "- Watch for smoke near waste bins.\n"
                ),
                alert_policy_prompt="",
            )

            health = settings["prompt_health"]
            self.assertTrue(health["needs_migration"])
            self.assertTrue(health["legacy_prose_alert_format"])
            self.assertTrue(health["legacy_alert_criteria_in_stream"])
            self.assertIn("person falls near stairs", health["suggested_alert_policy_prompt"])
            self.assertIn("smoke near waste bins", health["suggested_alert_policy_prompt"])
            self.assertNotIn("Warning Level", health["suggested_stream_system_prompt"])
            self.assertNotIn("Alert when", health["suggested_stream_system_prompt"])

    def test_criteria_only_monitor_prompt_migrates_out_of_stream_role(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            health = manager._legacy_alert_prompt_health(
                (
                    "Monitor workspace scene for presence changes of Sasha and cats.\n"
                    "Alert when Sasha or either cat enters or leaves.\n"
                    "Create alert if Sasha shows thumbs up."
                ),
                "",
            )

            self.assertTrue(health["needs_migration"])
            self.assertEqual(health["suggested_stream_system_prompt"], "")
            self.assertIn(
                "Monitor workspace scene for presence changes",
                health["suggested_alert_policy_prompt"],
            )

    def test_l0_alert_role_language_is_not_misclassified_as_watch_criteria(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            health = manager._legacy_alert_prompt_health(
                (
                    "You are EVA's visual-semantic intellectual core. "
                    "Alert criteria and deployment rules are supplied separately. "
                    "Match grounded current events with alert profiles and raise alerts "
                    "if an event matches criteria. Alerts may regulate later attention."
                ),
                "Watch for smoke near the east gate.",
            )

            self.assertFalse(health["needs_migration"])
            self.assertFalse(health["legacy_alert_criteria_in_stream"])
            self.assertEqual(health["warnings"], [])

    def test_prompt_settings_expose_compact_memory_metabolism_not_raw_memory_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            install_channel_memory(manager)

            settings = manager.get_prompt_settings(channel_id=7)
            metabolism = settings["memory_metabolism"]
            current = metabolism["current_state"]

            self.assertEqual(metabolism["status"], "active")
            self.assertEqual(current["source_level"], "L2")
            self.assertEqual(current["active_watchlist_count"], 1)
            self.assertEqual(current["preserved_deviations_count"], 0)
            self.assertEqual(
                [stage["level"] for stage in metabolism["stages"]],
                ["L0", "L1", "L2", "L3"],
            )
            self.assertTrue(metabolism["stages"][1]["applies_to_live_memory"])
            self.assertFalse(metabolism["stages"][2]["applies_to_live_memory"])
            self.assertFalse(metabolism["stages"][3]["applies_to_live_memory"])
            self.assertNotIn("backend_memory", settings["prompt_layers"]["stream"])
            self.assertNotIn("active_memory", settings["prompt_layers"]["rollups"]["L1"])
            self.assertTrue(
                settings["prompt_layers"]["stream"]["memory_context"]["present"]
            )
            self.assertEqual(current["alert_tuning_notes_count"], 0)
            self.assertEqual(current["ignore_as_routine_count"], 0)
            self.assertEqual(current["held_tuning_proposals_count"], 0)
            self.assertEqual(
                current["held_routine_suppression_proposals_count"],
                0,
            )
            live_prompt = manager._get_channel_routine_prompt(7)
            self.assertIn("watch the east gate", live_prompt)
            self.assertNotIn("vehicle drifting", live_prompt)
            self.assertNotIn("parking lot is usually empty", live_prompt)
            self.assertNotIn("keep drifting visible", live_prompt)
            self.assertNotIn("parked maintenance vehicles", live_prompt)

    def test_new_l1_replaces_and_can_clear_short_lived_watchlist(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            now = time.time()
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l1-watch",
                window_end=now - 1.0,
                level="L1",
                memory_update={"active_watchlist": ["old unresolved item"]},
                summary_text="",
            )
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l1-watch-a",
                window_end=now,
                level="L1",
                memory_update={"active_watchlist": ["check unresolved east gate event"]},
                summary_text="",
            )
            self.assertIn("east gate", manager._get_channel_routine_prompt(7))

            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l1-watch-b",
                window_end=now + 900,
                level="L1",
                memory_update={},
                summary_text="",
            )
            self.assertNotIn(
                "active_watchlist",
                manager.channel_routine_context[7]["memory"],
            )
            self.assertEqual(manager._get_channel_routine_prompt(7), "")

    def test_live_memory_expires_watchlist_and_rejects_schema_baseline_placeholder(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            now = time.time()
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l2-placeholder",
                window_end=now,
                level="L2",
                memory_update={
                    "routine_baseline": "normal pattern for this channel, if grounded",
                    "active_watchlist": ["old unresolved item"],
                    "preserved_deviations": ["grounded prior deviation"],
                },
                summary_text="",
            )
            manager.channel_routine_context[7]["memory_field_updated_at"][
                "active_watchlist"
            ] = now - 3601

            live_prompt = manager._get_channel_routine_prompt(7)
            self.assertNotIn("normal pattern for this channel", live_prompt)
            self.assertNotIn("old unresolved item", live_prompt)
            self.assertNotIn("grounded prior deviation", live_prompt)

    def test_channel_memory_text_marks_prior_context_not_current_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            install_channel_memory(manager)

            memory_prompt = manager._get_channel_routine_prompt(7)
            normalized = " ".join(memory_prompt.lower().split())

            self.assertIn("prior", normalized)
            self.assertIn("not a current observation", normalized)
            self.assertIn("do not assert", normalized)
            self.assertIn("from memory", normalized)

    def test_alert_diagnostics_expose_json_prose_and_parser_counts_separately(self):
        parser = load_lm_alert_parser()
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), alert_parser=parser)
            manager.default_bookmark_enabled = True
            manager.default_bookmark_cooldown_sec = 0.0
            summary = (
                "Window summary.\n\n"
                "Alerts:\n"
                "Info: Person waves at the gate.\n\n"
                "BATCH_STATE_JSON:\n"
                "{\"version\":1,\"cover\":{\"snapshot_index\":2},\"events\":[],"
                "\"observed_states\":[],\"routines\":[],\"memory_pass\":[],"
                "\"alerts\":[{\"title\":\"Vehicle drifting\",\"description\":\"Vehicle drifting in the lot.\","
                "\"severity\":\"high\",\"state\":\"new\",\"channel_id\":7,\"timestamp_ms\":0,"
                "\"snapshot_indices\":[2]}]}"
            )

            with patch.object(
                manager,
                "send_bookmark_event",
                side_effect=lambda **_kwargs: {"success": True},
            ):
                result = manager.process_summary_alerts(7, summary, default_ts_ms=1_781_700_000_000)

            diagnostics = result.as_dict()
            self.assertEqual(diagnostics["json_alert_count"], 1)
            self.assertEqual(diagnostics["prose_alert_count"], 1)
            self.assertEqual(diagnostics["parser_alert_count"], 1)
            self.assertEqual(len(diagnostics["alert_events"]), 1)
            self.assertEqual(diagnostics["alert_events"][0]["title"], "Vehicle drifting")
            self.assertEqual(result.alert_events[0]["snapshot_indices"], [2])
            self.assertTrue(
                result.alert_events[0]["id"].startswith("vlm-alert-")
            )
            self.assertEqual(
                diagnostics["alert_events"][0]["id"],
                result.alert_events[0]["id"],
            )

    def test_alert_parser_uses_first_evidence_snapshot_unless_anchor_is_explicit(self):
        parser = load_lm_alert_parser()
        summary = (
            "BATCH_STATE_JSON:\n"
            '{"version":1,"cover":{"snapshot_index":3},"events":[],'
            '"observed_states":[],"routines":[],"memory_pass":[],'
            '"alerts":[{"title":"Vessel convergence","severity":"high",'
            '"snapshot_indices":[2,3]}]}'
        )

        parsed = parser(summary, 7, 1_781_700_060_000)

        self.assertEqual(parsed[0]["snapshot_indices"], [2, 3])
        self.assertEqual(parsed[0]["anchor_snapshot"], 2)


if __name__ == "__main__":
    unittest.main()
