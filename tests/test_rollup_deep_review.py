import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rollup_deep_review import (
    DeepReviewClientConfig,
    OpenAICompatibleDeepReviewClient,
    QuietWindowSchedule,
)
from tests.test_luxriot_inference_runtime import (
    MemoryRuntimeStateStore,
    build_manager,
    operator_rollup_response,
)


class QuietWindowScheduleTests(unittest.TestCase):
    def test_cross_midnight_window_uses_the_start_day(self):
        schedule = QuietWindowSchedule(
            enabled=True,
            timezone="UTC",
            start_local="23:00",
            end_local="02:00",
            days=(0,),
        )
        monday_late = datetime(
            2026, 7, 27, 23, 30, tzinfo=timezone.utc
        ).timestamp()
        tuesday_early = datetime(
            2026, 7, 28, 1, 30, tzinfo=timezone.utc
        ).timestamp()
        tuesday_late = datetime(
            2026, 7, 28, 23, 30, tzinfo=timezone.utc
        ).timestamp()

        self.assertTrue(schedule.window_status(monday_late)["allowed"])
        self.assertTrue(schedule.window_status(tuesday_early)["allowed"])
        self.assertFalse(schedule.window_status(tuesday_late)["allowed"])

    def test_persisted_false_string_is_not_treated_as_true(self):
        schedule = QuietWindowSchedule.from_mapping({"enabled": "false"})
        self.assertFalse(schedule.enabled)
        with self.assertRaisesRegex(ValueError, "boolean"):
            QuietWindowSchedule.from_mapping({"enabled": "sometimes"})


class DeepReviewClientTests(unittest.TestCase):
    def test_openai_client_uses_bounded_split_timeout_and_configured_model(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "review"}}]
        }
        client = OpenAICompatibleDeepReviewClient(
            DeepReviewClientConfig(
                base_url="http://127.0.0.1:1240/v1",
                model="qwen-9b-cpu",
                connect_timeout_seconds=2,
                read_timeout_seconds=20,
            )
        )

        with patch("rollup_deep_review.requests.post", return_value=response) as post:
            result = client([{"role": "user", "content": "audit"}])

        self.assertEqual(result, "review")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["timeout"], (2.0, 20.0))
        self.assertEqual(kwargs["json"]["model"], "qwen-9b-cpu")
        self.assertFalse(kwargs["json"]["stream"])


class LuxriotDeepReviewRoutingTests(unittest.TestCase):
    def _manager(self, directory, *, state_store=None, lm_callback=None):
        return build_manager(
            Path(directory),
            runtime_state_store=state_store,
            lm_callback=lm_callback,
            config_overrides={
                "LUXRIOT_ROLLUP_LLM_MODEL": "agent-profile",
                "LUXRIOT_ROLLUP_LLM_LEVELS": "L1,L2,L3",
                "LUXRIOT_ROLLUP_L3_DEEP_ENABLED": True,
                "LUXRIOT_ROLLUP_L3_DEEP_BASE_URL": "http://127.0.0.1:1240/v1",
                "LUXRIOT_ROLLUP_L3_DEEP_MODEL": "qwen-9b-cpu",
                "LUXRIOT_ROLLUP_L3_QUIET_WINDOW_ENABLED": False,
            },
        )

    def test_level_routes_and_models_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(temp)

            self.assertIs(manager._rollup_callback_for_level("L1"), manager.lm_callback)
            self.assertIs(manager._rollup_callback_for_level("L2"), manager.lm_callback)
            self.assertEqual(
                manager._rollup_callback_for_level("L3").__func__,
                manager._call_l3_deep_review.__func__,
            )
            self.assertEqual(
                manager._get_rollup_model_hint_locked(7, "L1"),
                "agent-profile",
            )
            self.assertEqual(
                manager._get_rollup_model_hint_locked(7, "L3"),
                "qwen-9b-cpu",
            )

    def test_operator_schedule_is_validated_persisted_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = MemoryRuntimeStateStore()
            manager = self._manager(temp, state_store=state_store)
            result = manager.set_rollup_l3_deep_schedule(
                {
                    "enabled": True,
                    "timezone": "Europe/Riga",
                    "start_local": "00:30",
                    "end_local": "04:30",
                    "days": ["mon", "wed", "fri"],
                    "max_deferral_seconds": 7200,
                    "poll_seconds": 30,
                    "max_activity_x": 1.25,
                    "alert_lookback_seconds": 600,
                    "max_l0_coverage_debt": 0.5,
                }
            )

            saved = state_store.payloads[manager.ROLLUP_L3_DEEP_SCHEDULE_KEY]
            self.assertEqual(saved["timezone"], "Europe/Riga")
            self.assertEqual(saved["days"], [0, 2, 4])
            self.assertEqual(result["source"], "runtime_state")
            status = manager.streams_status()["rollup_l3_deep_review"]
            self.assertEqual(status["concurrency"], 1)
            self.assertEqual(status["schedule_source"], "runtime_state")
            self.assertEqual(status["model"], "qwen-9b-cpu")

    def test_unavailable_deep_model_persists_proposal_only_closure_without_agent_fallback(self):
        agent_calls = []

        def agent_callback(_messages, _model):
            agent_calls.append(True)
            return operator_rollup_response("Agent fallback must not run.")

        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(temp, lm_callback=agent_callback)
            node = {
                "rollup_id": "l3-ch7-w1800-100",
                "window_start": 100.0,
                "window_end": 1900.0,
                "window_sec": 1800,
                "item_count": 1,
                "frame_count": 12,
                "source_tokens": 1000,
                "run_ids": ["run-7"],
                "source_ids": ["l2-a"],
                "source_signature": "sig-l2-a",
                "summary": operator_rollup_response(
                    "Deterministic source-only closure."
                ),
            }
            manager._apply_rollup_llm_summaries(
                channel_id=7,
                level="L3",
                source_level="L2",
                node_children_pairs=[(node, [dict(node, level="L2")])],
                max_new=1,
            )

            self.assertEqual(agent_calls, [])
            cached = manager._get_cached_rollup_record(node["rollup_id"])
            self.assertIsNotNone(cached)
            self.assertEqual(cached["summary_kind"], "review_pending")
            self.assertEqual(cached["generation_status"], "review_deferred")
            self.assertTrue(cached["review_only"])
            self.assertTrue(cached["proposals_only"])
            self.assertFalse(cached["mutations_applied"])

    def test_deep_gate_requires_quiet_activity_no_alerts_and_no_l0_debt(self):
        now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(temp)
            manager.rollup_l3_deep_schedule = QuietWindowSchedule(
                enabled=True,
                timezone="UTC",
                start_local="00:00",
                end_local="23:59",
                days=tuple(range(7)),
                max_activity_x=1.5,
                max_l0_coverage_debt=0.5,
            )
            manager.sessions[7] = SimpleNamespace(
                status=lambda: {
                    "summary_queue_depth": 1,
                    "summary_inflight": True,
                    "capture_apex_last_selection": {"activity_x": 2.0},
                }
            )
            manager.summary_history[7] = [
                {"created_at": now - 100, "alert_total": 1},
                *[
                    {"created_at": now - 99 + index, "alert_total": 0}
                    for index in range(40)
                ],
            ]
            with patch.object(
                manager.attention_coordinator,
                "status",
                return_value={
                    "channels": [
                        {
                            "last_activity_x": 2.0,
                            "decision": {
                                "mode": "active",
                                "coverage_debt": 0.8,
                            },
                            "inflight_job_id": "l0-job",
                        }
                    ]
                },
            ):
                blocked = manager._l3_deep_admission_gate(now)

            self.assertFalse(blocked["allowed"])
            self.assertEqual(
                blocked["reasons"],
                [
                    "activity_above_quiet_gate",
                    "l0_attention_debt",
                    "recent_alerts",
                ],
            )

    def test_l3_memory_is_stored_as_a_proposal_but_not_applied_live(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(temp)
            summary = operator_rollup_response(
                "Deep review proposal.",
                memory={
                    "routine_baseline": "do not apply this L3 proposal",
                    "alert_tuning_notes": ["proposal only"],
                },
            )
            manager._put_cached_rollup_summary(
                "l3-ch7-w1800-100",
                summary,
                channel_id=7,
                level="L3",
                source_level="L2",
                window_start=100.0,
                window_end=1900.0,
                window_sec=1800,
                summary_kind="llm",
                generation_status="ready",
                format_version=2,
                review_only=True,
                proposals_only=True,
                mutations_applied=False,
            )
            cached = manager._get_cached_rollup_record("l3-ch7-w1800-100")
            manager._refresh_channel_memory_from_rollups(7, [cached])

            self.assertNotIn(7, manager.channel_routine_context)
            self.assertIn("routine_baseline", cached["memory_update"])
            self.assertFalse(cached["mutations_applied"])


if __name__ == "__main__":
    unittest.main()
