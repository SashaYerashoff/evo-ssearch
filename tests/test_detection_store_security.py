import tempfile
import time
import unittest
from pathlib import Path

from detection_store import DetectionsStore


class DetectionStoreSecurityTests(unittest.TestCase):
    def test_summary_keeps_probe_hits_separate_from_vlm_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DetectionsStore(
                path=str(Path(temp_dir) / "detections.sqlite3")
            )
            now_ms = int(time.time() * 1000)
            self.assertTrue(
                store.add_detection(
                    {
                        "dedupe_key": "probe-event-1",
                        "timestamp_ms": now_ms,
                        "probe_id": "shared-id",
                        "probe_name": "Shared",
                        "channel_id": 7,
                        "source": "probe",
                    }
                )
            )
            self.assertTrue(
                store.add_detection(
                    {
                        "dedupe_key": "legacy-probe-event-1",
                        "timestamp_ms": now_ms + 2,
                        "probe_id": "legacy-id",
                        "probe_name": "Legacy",
                        "channel_id": 7,
                        "source": "probes_run",
                    }
                )
            )
            conn = store._connect()
            try:
                conn.execute(
                    "UPDATE detections SET source = 'probes_run' WHERE dedupe_key = ?",
                    ("legacy-probe-event-1",),
                )
                conn.commit()
            finally:
                conn.close()
            self.assertTrue(
                store.add_detection(
                    {
                        "dedupe_key": "vlm-event-1",
                        "timestamp_ms": now_ms + 1,
                        "probe_id": "shared-id",
                        "probe_name": "Shared",
                        "channel_id": 7,
                        "source": "vlm_summary",
                    }
                )
            )

            all_rows = store.summarize_by_probe(since_ms=now_ms - 1000)
            hit_count_by_source = {}
            for row in all_rows:
                hit_count_by_source[row["source"]] = (
                    hit_count_by_source.get(row["source"], 0) + row["hit_count"]
                )

            self.assertEqual(hit_count_by_source["probe"], 2)
            self.assertEqual(hit_count_by_source["vlm_summary"], 1)
            self.assertEqual(
                store.summarize_by_probe(since_ms=now_ms - 1000, source="probe")[0]["source"],
                "probe",
            )
            rows, total = store.list_detections(since_ms=now_ms - 1000, source="probe")
            self.assertEqual(total, 2)
            self.assertEqual({row["source"] for row in rows}, {"probe", "probes_run"})

    def test_image_path_resolves_channel_ownership(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DetectionsStore(
                path=str(Path(temp_dir) / "detections.sqlite3")
            )
            image_path = str(Path(temp_dir) / "event.jpg")
            inserted = store.add_detection(
                {
                    "dedupe_key": "event-1",
                    "timestamp_ms": int(time.time() * 1000),
                    "probe_id": "probe-1",
                    "probe_name": "Door",
                    "channel_id": 7,
                    "image_path": image_path,
                }
            )

            self.assertTrue(inserted)
            self.assertEqual(
                store.channel_ids_for_image_path(image_path),
                frozenset({7}),
            )
            self.assertEqual(
                store.channel_ids_for_image_path("/missing.jpg"),
                frozenset(),
            )


if __name__ == "__main__":
    unittest.main()
