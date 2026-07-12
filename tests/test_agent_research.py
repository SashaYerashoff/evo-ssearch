import unittest

from agent_research import (
    continuation_tool_defaults,
    operator_requests_continuation,
    research_state_from_inventory,
    trusted_research_message,
    usable_research_state,
)


class AgentResearchLedgerTests(unittest.TestCase):
    def test_continuation_language_is_detected_without_matching_ordinary_text(self):
        self.assertTrue(operator_requests_continuation("Continue with the remaining channels"))
        self.assertTrue(operator_requests_continuation("давай дальше по остальным"))
        self.assertTrue(operator_requests_continuation("продолжи"))
        self.assertTrue(operator_requests_continuation("Продолжай проверку остальных каналов"))
        self.assertTrue(operator_requests_continuation("check the remaining cameras"))
        self.assertTrue(operator_requests_continuation("следующий чанк"))
        self.assertFalse(operator_requests_continuation("Show channel 118"))

    def test_ordinary_questions_do_not_trigger_continuation(self):
        # Each of these used to hijack the frozen research window.
        self.assertFalse(operator_requests_continuation("какая продолжительность события на канале 5?"))
        self.assertFalse(operator_requests_continuation("что было дальше по времени?"))
        self.assertFalse(operator_requests_continuation("опиши остальную сцену"))
        self.assertFalse(operator_requests_continuation("summarize the remaining footage quality"))
        self.assertFalse(operator_requests_continuation("сколько длилось событие?"))

    def test_first_chunk_conserves_named_scope(self):
        state = research_state_from_inventory(
            {
                "time_window": {"from_ts": 100.0, "to_ts": 200.0},
                "requested_channel_ids": [1, 2, 3, 4, 5],
                "candidate_channels": [{"channel_id": 1}, {"channel_id": 2}],
                "inactive_channel_ids": [3],
                "deferred_channel_ids": [4],
                "unchecked_channel_ids": [5],
            },
            now=1_000.0,
        )

        self.assertEqual(state["completed_channel_ids"], [1, 2, 3])
        self.assertEqual(state["remaining_channel_ids"], [4, 5])
        self.assertEqual(state["status"], "pending")
        self.assertTrue(usable_research_state(state, now=1_001.0))

    def test_next_chunk_never_repeats_completed_and_keeps_unattempted_ids(self):
        previous = {
            "version": 1,
            "kind": "video_summary_inventory",
            "status": "pending",
            "frozen_window": {"from_ts": 100.0, "to_ts": 200.0},
            "requested_channel_ids": [1, 2, 3, 4, 5],
            "completed_channel_ids": [1, 2],
            "remaining_channel_ids": [3, 4, 5],
            "updated_at": 1_000.0,
        }

        state = research_state_from_inventory(
            {
                "time_window": {"from_ts": 100.0, "to_ts": 200.0},
                "requested_channel_ids": [3, 4],
                "candidate_channels": [{"channel_id": 3}],
                "deferred_channel_ids": [4],
            },
            previous=previous,
            continuation=True,
            now=1_010.0,
        )

        self.assertEqual(state["completed_channel_ids"], [1, 2, 3])
        self.assertEqual(state["remaining_channel_ids"], [4, 5])
        self.assertEqual(
            continuation_tool_defaults(state, now=1_011.0),
            {"channel_ids": [4, 5], "from_ts": 100.0, "to_ts": 200.0},
        )
        message = trusted_research_message(state, now=1_011.0)
        self.assertIn('"remaining_channel_ids":[4,5]', message)
        self.assertIn('"completed_channel_ids":[1,2,3]', message)

    def test_window_drift_is_flagged_and_frozen_window_wins(self):
        previous = {
            "version": 1,
            "kind": "video_summary_inventory",
            "status": "pending",
            "frozen_window": {"from_ts": 100.0, "to_ts": 200.0},
            "requested_channel_ids": [1, 2],
            "completed_channel_ids": [1],
            "remaining_channel_ids": [2],
            "updated_at": 1_000.0,
        }

        state = research_state_from_inventory(
            {
                "time_window": {"from_ts": 90.0, "to_ts": 250.0},
                "requested_channel_ids": [2],
                "candidate_channels": [{"channel_id": 2}],
            },
            previous=previous,
            continuation=True,
            now=1_010.0,
        )

        self.assertTrue(state["window_mismatch"])
        self.assertEqual(state["frozen_window"], previous["frozen_window"])
        self.assertEqual(state["status"], "complete")
        self.assertFalse(usable_research_state(state, now=1_011.0))


if __name__ == "__main__":
    unittest.main()
