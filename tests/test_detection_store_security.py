import tempfile
import time
import unittest
from pathlib import Path

from detection_store import DetectionsStore


class DetectionStoreSecurityTests(unittest.TestCase):
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
