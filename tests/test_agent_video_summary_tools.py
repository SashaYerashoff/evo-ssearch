import unittest

from agent import AgentTools


class _SummaryManager:
    def __init__(self):
        self.channels = [
            {"id": 7, "title": "Kitchen"},
            {"id": 8, "title": "Door"},
            {"id": 9, "title": "Quiet"},
        ]
        self.logs_by_channel = {
            7: [
                {"created_at": 100.0, "summary": "before window", "frame_count": 2},
                {"created_at": 150.0, "summary": "person enters", "frame_count": 3, "alert_counts": {"normal": 1}},
                {"created_at": 400.0, "summary": "after window", "frame_count": 4},
            ],
            8: [
                {"created_at": 175.0, "summary": "door light changed", "frame_count": 2},
            ],
            9: [],
        }

    def get_channels(self, force=False):
        return list(self.channels)

    def session_status(self, channel_id, run_selector=None, start_ts=None, end_ts=None, limit=None):
        logs = [
            dict(row)
            for row in self.logs_by_channel.get(int(channel_id), [])
            if (start_ts is None or float(row["created_at"]) >= float(start_ts))
            and (end_ts is None or float(row["created_at"]) <= float(end_ts))
        ]
        return {"running": False, "channel_id": channel_id, "logs": logs, "selected_run": None}

    def summary_rollups(self, channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
        # Intentionally include out-of-window nodes to verify AgentTools performs strict post-filtering.
        nodes = [
            {
                "level": "L1",
                "window_start": 90.0,
                "window_end": 99.0,
                "summary": "before",
                "frame_count": 1,
            },
            {
                "level": "L1",
                "window_start": 140.0,
                "window_end": 160.0,
                "summary": "inside",
                "frame_count": 3,
                "alert_counts": {"normal": 1},
                "alert_total": 1,
            },
            {
                "level": "L1",
                "window_start": 301.0,
                "window_end": 330.0,
                "summary": "after",
                "frame_count": 1,
            },
        ]
        return {
            "running": False,
            "selected_run": None,
            "run_filter_id": None,
            "levels": {"L0": [], "L1": nodes, "L2": [], "L3": []},
        }


def _tools(manager=None, search_detections_fn=None):
    return AgentTools(
        detections_store=object(),
        probes_store=object(),
        luxriot_manager=manager or _SummaryManager(),
        embed_text_fn=lambda _text: None,
        embed_image_fn=lambda _image: None,
        call_lm_fn=lambda *_args, **_kwargs: "",
        encode_jpeg_fn=lambda *_args, **_kwargs: "",
        search_indexed_folder_fn=lambda **_kwargs: [],
        search_detections_fn=search_detections_fn or (lambda **_kwargs: []),
    )


class AgentVideoSummaryToolTests(unittest.TestCase):
    def test_archive_search_passes_source_and_labels_vlm_summary_results(self):
        captured = {}

        def search_detections(**kwargs):
            captured.update(kwargs)
            return [
                {
                    "detection_id": 101,
                    "timestamp_ms": 1781389900000,
                    "source": "vlm_summary",
                    "probe_name": "VLM summary frame",
                    "channel_id": 7,
                    "score": 0.81,
                }
            ]

        result = _tools(search_detections_fn=search_detections).execute(
            "search_archive",
            {
                "query": "person at desk",
                "scope": "detections",
                "source": "vlm_summary",
                "channel_id": 7,
            },
        )

        self.assertEqual(captured["source"], "vlm_summary")
        self.assertEqual(result["source"], "vlm_summary")
        self.assertEqual(result["source_label"], "Video-description frame")
        self.assertEqual(result["results"][0]["archive_item_type"], "video_description_frame")

    def test_normalize_time_window_last_night_returns_seconds_and_milliseconds(self):
        tools = _tools()
        result = tools.execute(
            "normalize_time_window",
            {
                "date": "2026-06-14",
                "start_time": "01:30",
                "end_time": "08:30",
                "timezone": "Europe/Riga",
            },
        )

        self.assertEqual(result["from_ts"], 1781389800)
        self.assertEqual(result["to_ts"], 1781415000)
        self.assertEqual(result["since_ms"], 1781389800000)
        self.assertEqual(result["until_ms"], 1781415000000)

    def test_get_video_summaries_accepts_milliseconds_and_filters_window(self):
        manager = _SummaryManager()

        def epoch_rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            return {
                "running": False,
                "selected_run": None,
                "run_filter_id": None,
                "levels": {
                    "L0": [],
                    "L1": [
                        {
                            "level": "L1",
                            "window_start": 1781389700.0,
                            "window_end": 1781389799.0,
                            "summary": "before",
                        },
                        {
                            "level": "L1",
                            "window_start": 1781389900.0,
                            "window_end": 1781390100.0,
                            "summary": "inside",
                            "frame_count": 3,
                        },
                        {
                            "level": "L1",
                            "window_start": 1781392000.0,
                            "window_end": 1781392100.0,
                            "summary": "after",
                        },
                    ],
                    "L2": [],
                    "L3": [],
                },
            }

        manager.summary_rollups = epoch_rollups
        result = _tools(manager).execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L1",
                "from_ts": 1_781_389_800_000,
                "to_ts": 1_781_391_000_000,
                "limit": 10,
            },
        )

        self.assertEqual(result["time_window"]["normalized_input_units"]["from_ts"], "milliseconds")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["total_in_window"], 1)
        self.assertEqual(result["entries"][0]["summary"], "inside")

    def test_list_video_summary_channels_returns_active_candidates_and_confirmation_flag(self):
        result = _tools().execute(
            "list_video_summary_channels",
            {
                "from_ts": 100.0,
                "to_ts": 300.0,
                "limit": 10,
            },
        )

        self.assertEqual(result["active_count"], 2)
        self.assertEqual(result["inactive_count"], 1)
        self.assertFalse(result["requires_confirmation"])
        self.assertEqual(
            [row["channel_id"] for row in result["candidate_channels"]],
            [7, 8],
        )


if __name__ == "__main__":
    unittest.main()
