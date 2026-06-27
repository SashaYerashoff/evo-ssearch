import ast
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

from luxriot_connector import LuxriotManager


def build_manager(directory: Path, alert_parser=None) -> LuxriotManager:
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
    manager._update_channel_routine_context(
        channel_id=channel_id,
        rollup_id="l2-memory",
        window_end=1234.0,
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
    def test_final_live_prompt_places_prior_memory_before_alerts_json_contract(self):
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
            self.assertIn("ALERTS_JSON:", final_prompt)
            self.assertLess(
                final_prompt.index("Alert review policy"),
                final_prompt.index("Active Channel Memory"),
                "operator alert criteria should be separate from routine memory/prior",
            )
            self.assertLess(
                final_prompt.index("Active Channel Memory"),
                final_prompt.index("ALERTS_JSON:"),
                "channel memory/prior must be before the final ALERTS_JSON output contract",
            )
            self.assertLess(
                final_prompt.index("Current-batch observation contract"),
                final_prompt.index("ALERTS_JSON:"),
                "ALERTS_JSON must stay last after current-state instructions",
            )

    def test_default_alert_contract_has_no_private_scene_entities(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            final_prompt = manager.compose_live_system_prompt(7, manager.get_effective_stream_system_prompt(7))
            lowered = final_prompt.lower()

            for forbidden in ("orlandina", "sphynx", "union jack", "british flag", "sasha"):
                self.assertNotIn(forbidden, lowered)

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
            self.assertIn("ALERTS_JSON:", final_prompt)

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
                "ALERTS_JSON:\n"
                "{\"alerts\":[{\"title\":\"Vehicle drifting\",\"description\":\"Vehicle drifting in the lot.\","
                "\"severity\":\"high\",\"state\":\"new\",\"channel_id\":7,\"timestamp_ms\":0}]}"
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
            self.assertEqual(diagnostics["parser_alert_count"], 2)


if __name__ == "__main__":
    unittest.main()
