import unittest

from tests.integration.real_data_manifest import (
    _coverage_from_logs,
    build_frozen_manifest,
)


class _Session:
    def __init__(self, *, all_channels=True):
        self.all_channels = all_channels
        self.calls = []

    def whoami(self):
        return {
            "user": {
                "username": "admin" if self.all_channels else "operator",
                "roles": ["admin"] if self.all_channels else ["operator"],
                "allowedChannelIds": ["*"] if self.all_channels else ["112"],
            }
        }

    def get_json(self, path, *, params=None):
        params = dict(params or {})
        self.calls.append((path, params))
        if path == "/luxriot/channels":
            return {
                "channels": [
                    {"id": 112, "title": "Zenbook webcam"},
                    {"id": 118, "title": "emu-1"},
                ]
            }
        if path == "/luxriot/streams":
            return {
                "desired_video_channels": [112],
                "video_streams": [
                    {"channel_id": 112, "running": True, "last_summary_at": 119.0}
                ],
            }
        if path == "/luxriot/session":
            channel_id = int(params["channel_id"])
            return {
                "channel_id": channel_id,
                "logs": (
                    [
                        {
                            "run_id": "run-112",
                            "batch_start_ms": 100_000,
                            "batch_end_ms": 110_000,
                        },
                        {
                            "run_id": "run-112",
                            "batch_start_ms": 105_000,
                            "batch_end_ms": 120_000,
                        },
                    ]
                    if channel_id == 112
                    else []
                ),
            }
        if path == "/detections/list":
            total_by_source = {"vlm_summary": 3, "vlm_alert": 1, "probe": 0}
            total = total_by_source[params["source"]]
            if total <= 0:
                detections = []
            elif int(params.get("offset") or 0) == total - 1:
                detections = [{"timestamp_ms": 101_000}]
            else:
                detections = [{"timestamp_ms": 199_000}]
            return {"total": total, "detections": detections}
        raise AssertionError(path)


class FrozenManifestTests(unittest.TestCase):
    def test_union_coverage_does_not_count_overlapping_time_twice(self):
        coverage = _coverage_from_logs(
            [
                {"batch_start_ms": 100_000, "batch_end_ms": 110_000},
                {"batch_start_ms": 105_000, "batch_end_ms": 120_000},
            ],
            from_ts=100.0,
            to_ts=200.0,
        )

        self.assertEqual(coverage["union_covered_sec"], 20.0)
        self.assertEqual(coverage["gap_count"], 1)
        self.assertEqual(coverage["gaps"][0]["from_ts"], 120.0)
        self.assertEqual(coverage["gaps"][0]["to_ts"], 200.0)

    def test_manifest_freezes_window_and_compares_memory_to_archive(self):
        session = _Session()

        manifest = build_frozen_manifest(
            session,
            from_ts=100.0,
            to_ts=200.0,
            channel_ids=[112],
        )

        self.assertEqual(manifest["missing_requested_channel_ids"], [])
        channel = manifest["channels"]["112"]
        self.assertTrue(channel["desired"])
        self.assertEqual(channel["summary_memory"]["log_count"], 2)
        self.assertEqual(channel["summary_memory"]["union_covered_sec"], 20.0)
        self.assertEqual(channel["archive"]["vlm_summary"]["total"], 3)
        self.assertEqual(
            channel["archive"]["vlm_summary"]["first_timestamp_ms"],
            101_000,
        )
        self.assertEqual(
            channel["archive"]["vlm_summary"]["last_timestamp_ms"],
            199_000,
        )
        self.assertTrue(
            all(
                call_params.get("until_ms") == 200_000
                for path, call_params in session.calls
                if path == "/detections/list"
            )
        )

    def test_manifest_refuses_non_admin_scope_by_default(self):
        with self.assertRaisesRegex(PermissionError, "all-channel"):
            build_frozen_manifest(
                _Session(all_channels=False),
                from_ts=100.0,
                to_ts=200.0,
                channel_ids=[112],
            )


if __name__ == "__main__":
    unittest.main()
