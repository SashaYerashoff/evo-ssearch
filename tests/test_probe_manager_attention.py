import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from probe_manager import ProbeBuffer, ProbeManager


class ProbeManagerAttentionTests(unittest.TestCase):
    def _manager(self):
        text_calls = []

        def embed_image(image):
            level = float(image.getpixel((0, 0))[0]) / 255.0
            return np.asarray([level, 1.0 - level], dtype=np.float32)

        def embed_text(text):
            text_calls.append(text)
            if "bright" in text.casefold():
                return np.asarray([1.0, 0.0], dtype=np.float32)
            return np.asarray([0.0, 1.0], dtype=np.float32)

        manager = ProbeManager(
            embed_image_fn=embed_image,
            embed_text_fn=embed_text,
            jpeg_encoder=lambda _image, **_kwargs: "jpeg",
        )
        return manager, text_calls

    def test_add_frame_returns_stable_embedding_reference(self):
        manager, _calls = self._manager()
        with patch.object(ProbeBuffer, "_rebuild_index", return_value=None):
            first = manager.add_frame(
                7,
                Image.new("RGB", (4, 4), color=(255, 255, 255)),
                1_000,
            )
            second = manager.add_frame(
                7,
                Image.new("RGB", (4, 4), color=(0, 0, 0)),
                2_000,
            )
        self.assertEqual(first["embedding_ref"], "probe-buffer:7:1")
        self.assertEqual(second["embedding_ref"], "probe-buffer:7:2")
        self.assertEqual(first["timestamp_ms"], 1_000)

    def test_score_frames_returns_pnm_for_every_embedding_in_window(self):
        manager, _calls = self._manager()
        with patch.object(ProbeBuffer, "_rebuild_index", return_value=None):
            manager.add_frame(
                7,
                Image.new("RGB", (4, 4), color=(255, 255, 255)),
                1_000,
            )
            manager.add_frame(
                7,
                Image.new("RGB", (4, 4), color=(0, 0, 0)),
                2_000,
            )
        scored = manager.score_frames(
            7,
            ["bright foreground"],
            ["dark background"],
            min_ts_ms=1_000,
            max_ts_ms=2_000,
        )
        rows = scored["results"]
        self.assertEqual([row["timestamp_ms"] for row in rows], [1_000, 2_000])
        self.assertGreater(rows[0]["margin"], 0.9)
        self.assertLess(rows[1]["margin"], -0.9)

    def test_text_embeddings_are_cached_across_probe_queries(self):
        manager, calls = self._manager()
        with patch.object(ProbeBuffer, "_rebuild_index", return_value=None):
            manager.add_frame(
                7,
                Image.new("RGB", (4, 4), color=(255, 255, 255)),
                1_000,
            )
        for _ in range(2):
            manager.score_frames(
                7,
                ["bright foreground"],
                ["dark background"],
            )
        self.assertEqual(calls, ["bright foreground", "dark background"])


if __name__ == "__main__":
    unittest.main()
