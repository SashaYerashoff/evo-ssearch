import base64
import threading
import time
import unittest
from io import BytesIO
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

    def test_frame_thumbnail_returns_only_the_exact_scored_frame(self):
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

        self.assertEqual(manager.frame_thumbnail(7, 1_000), "jpeg")
        self.assertEqual(manager.frame_thumbnail(7, 2_000), "jpeg")
        self.assertIsNone(manager.frame_thumbnail(7, 1_500))
        self.assertIsNone(manager.frame_thumbnail(8, 2_000))

    def test_live_append_does_not_rebuild_unused_faiss_index(self):
        manager, _calls = self._manager()
        with patch.object(ProbeBuffer, "_rebuild_index") as rebuild:
            manager.add_frame(
                7,
                Image.new("RGB", (4, 4), color=(255, 255, 255)),
                1_000,
            )
        rebuild.assert_not_called()
        self.assertIsNone(manager.buffers[7].index)

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

    def test_probe_texts_are_batched_and_reused_after_prewarm(self):
        scalar_calls = []
        batch_calls = []

        def embed_text(text):
            scalar_calls.append(text)
            raise AssertionError("scalar text encoder should not be used")

        def embed_texts(texts):
            batch_calls.append(list(texts))
            return np.asarray(
                [
                    [1.0, 0.0] if "bright" in text else [0.0, 1.0]
                    for text in texts
                ],
                dtype=np.float32,
            )

        manager = ProbeManager(
            embed_image_fn=lambda _image: np.asarray([1.0, 0.0], dtype=np.float32),
            embed_text_fn=embed_text,
            embed_texts_fn=embed_texts,
            jpeg_encoder=lambda _image, **_kwargs: "jpeg",
        )
        with patch.object(ProbeBuffer, "_rebuild_index", return_value=None):
            manager.add_frame(
                7,
                Image.new("RGB", (4, 4), color=(255, 255, 255)),
                1_000,
            )

        self.assertEqual(
            manager.prewarm_texts(["bright foreground", "dark background"]),
            2,
        )
        for _ in range(2):
            manager.score_frames(
                7,
                ["bright foreground"],
                ["dark background"],
            )
        self.assertEqual(
            batch_calls,
            [["bright foreground", "dark background"]],
        )
        self.assertEqual(scalar_calls, [])

    def test_async_text_prewarm_never_blocks_caller_and_populates_cache(self):
        encoder_started = threading.Event()
        release_encoder = threading.Event()

        def embed_texts(texts):
            encoder_started.set()
            release_encoder.wait(timeout=2.0)
            return np.asarray([[1.0, 0.0] for _text in texts], dtype=np.float32)

        manager = ProbeManager(
            embed_image_fn=lambda _image: np.asarray([1.0, 0.0], dtype=np.float32),
            embed_text_fn=lambda _text: np.asarray([1.0, 0.0], dtype=np.float32),
            embed_texts_fn=embed_texts,
            jpeg_encoder=lambda _image, **_kwargs: "jpeg",
        )
        started = time.monotonic()
        scheduled = manager.prewarm_texts_async(["person near window"])
        elapsed = time.monotonic() - started

        self.assertEqual(scheduled, 1)
        self.assertLess(elapsed, 0.1)
        self.assertTrue(encoder_started.wait(timeout=1.0))
        self.assertFalse(manager.texts_cached(["person near window"]))

        release_encoder.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if manager.texts_cached(["person near window"]):
                break
            time.sleep(0.01)
        self.assertTrue(manager.texts_cached(["person near window"]))

    def test_frame_metadata_keeps_cache_hits_off_embedding_space_reader(self):
        space_calls = []
        embedding_space = {
            "backend": "siglip2",
            "model": "test/siglip2",
            "dimension": 2,
        }
        manager = ProbeManager(
            embed_image_fn=lambda _image: np.asarray([1.0, 0.0], dtype=np.float32),
            embed_image_with_metadata_fn=lambda _image: (
                np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_space,
            ),
            embed_text_fn=lambda _text: np.asarray([1.0, 0.0], dtype=np.float32),
            embedding_space_fn=lambda: space_calls.append(True) or embedding_space,
            jpeg_encoder=lambda _image, **_kwargs: "jpeg",
        )
        with patch.object(ProbeBuffer, "_rebuild_index", return_value=None):
            manager.add_frame(
                7,
                Image.new("RGB", (4, 4), color=(255, 255, 255)),
                1_000,
            )

        for _ in range(2):
            manager.score_frames(7, ["bright foreground"], [])

        self.assertEqual(space_calls, [])

    def test_non_roi_query_scores_snapshot_without_blocking_ingestion(self):
        manager, _calls = self._manager()
        manager.add_frame(
            7,
            Image.new("RGB", (4, 4), color=(255, 255, 255)),
            1_000,
        )
        query_started = threading.Event()
        release_query = threading.Event()
        append_completed = threading.Event()
        original_query = ProbeBuffer.query

        def blocked_query(buffer, *args, **kwargs):
            query_started.set()
            release_query.wait(timeout=2.0)
            return original_query(buffer, *args, **kwargs)

        with patch.object(ProbeBuffer, "query", blocked_query):
            query_thread = threading.Thread(
                target=lambda: manager.query(
                    7,
                    ["bright foreground"],
                    ["dark background"],
                    0.1,
                    0.0,
                    1,
                ),
                daemon=True,
            )
            query_thread.start()
            self.assertTrue(query_started.wait(timeout=0.5))
            append_thread = threading.Thread(
                target=lambda: (
                    manager.add_frame(
                        7,
                        Image.new("RGB", (4, 4), color=(0, 0, 0)),
                        2_000,
                    ),
                    append_completed.set(),
                ),
                daemon=True,
            )
            append_thread.start()
            self.assertTrue(append_completed.wait(timeout=0.5))
            release_query.set()
            query_thread.join(timeout=1.0)
            append_thread.join(timeout=1.0)

        self.assertEqual(manager.status(7)["frames"], 2)

    def test_roi_query_backfills_only_bounded_fresh_frames(self):
        image_calls = []

        def embed_image(image):
            image_calls.append(image.size)
            level = float(image.getpixel((0, 0))[0]) / 255.0
            return np.asarray([level, 1.0 - level], dtype=np.float32)

        def encode(image, **_kwargs):
            output = BytesIO()
            image.save(output, format="PNG")
            return base64.b64encode(output.getvalue()).decode("ascii")

        manager = ProbeManager(
            embed_image_fn=embed_image,
            embed_text_fn=lambda _text: np.asarray([1.0, 0.0], dtype=np.float32),
            jpeg_encoder=encode,
        )
        for index in range(5):
            manager.add_frame(
                7,
                Image.new("RGB", (8, 4), color=(255, 255, 255)),
                (index + 1) * 1_000,
            )
        image_calls.clear()

        first = manager.query(
            7,
            ["bright"],
            [],
            0.1,
            0.0,
            10,
            roi_norm=(0.5, 0.0, 0.5, 1.0),
            roi_padding=0.0,
            roi_embedding_budget=2,
        )
        self.assertEqual(len(image_calls), 2)
        self.assertEqual(
            {row["timestamp_ms"] for row in first["results"]},
            {4_000, 5_000},
        )

        second = manager.query(
            7,
            ["bright"],
            [],
            0.1,
            0.0,
            10,
            roi_norm=(0.5, 0.0, 0.5, 1.0),
            roi_padding=0.0,
            roi_embedding_budget=2,
        )
        self.assertEqual(len(image_calls), 4)
        self.assertEqual(len(second["results"]), 4)

    def test_realtime_roi_score_seeds_daemon_query_cache(self):
        image_calls = []

        def embed_image(image):
            image_calls.append(image.size)
            level = float(image.getpixel((0, 0))[0]) / 255.0
            return np.asarray([level, 1.0 - level], dtype=np.float32)

        def encode(image, **_kwargs):
            output = BytesIO()
            image.save(output, format="PNG")
            return base64.b64encode(output.getvalue()).decode("ascii")

        manager = ProbeManager(
            embed_image_fn=embed_image,
            embed_text_fn=lambda _text: np.asarray([1.0, 0.0], dtype=np.float32),
            jpeg_encoder=encode,
        )
        frame = Image.new("RGB", (8, 4), color=(255, 255, 255))
        saved = manager.add_frame(7, frame, 1_000)
        image_calls.clear()
        scored = manager.score_current_frame(
            7,
            1_000,
            ["bright"],
            [],
            embedding=saved["embedding"],
            thumbnail_b64=saved["thumbnail"],
            roi_norm=(0.5, 0.0, 0.5, 1.0),
            roi_padding=0.0,
        )
        self.assertNotIn("error", scored)
        self.assertEqual(len(image_calls), 1)

        queried = manager.query(
            7,
            ["bright"],
            [],
            0.1,
            0.0,
            10,
            roi_norm=(0.5, 0.0, 0.5, 1.0),
            roi_padding=0.0,
            roi_embedding_budget=0,
        )
        self.assertEqual(len(image_calls), 1)
        self.assertEqual(len(queried["results"]), 1)

    def test_roi_query_scores_snapshot_without_blocking_ingestion(self):
        manager, _calls = self._manager()
        manager.add_frame(
            7,
            Image.new("RGB", (4, 4), color=(255, 255, 255)),
            1_000,
        )
        query_started = threading.Event()
        release_query = threading.Event()
        append_completed = threading.Event()
        original_query = ProbeBuffer.query

        def blocked_query(buffer, *args, **kwargs):
            query_started.set()
            release_query.wait(timeout=2.0)
            return original_query(buffer, *args, **kwargs)

        with patch.object(ProbeBuffer, "query", blocked_query):
            query_thread = threading.Thread(
                target=lambda: manager.query(
                    7,
                    ["bright foreground"],
                    ["dark background"],
                    0.1,
                    0.0,
                    1,
                    roi_norm=(0.0, 0.0, 1.0, 1.0),
                ),
                daemon=True,
            )
            query_thread.start()
            self.assertTrue(query_started.wait(timeout=0.5))
            append_thread = threading.Thread(
                target=lambda: (
                    manager.add_frame(
                        7,
                        Image.new("RGB", (4, 4), color=(0, 0, 0)),
                        2_000,
                    ),
                    append_completed.set(),
                ),
                daemon=True,
            )
            append_thread.start()
            self.assertTrue(append_completed.wait(timeout=0.5))
            release_query.set()
            query_thread.join(timeout=1.0)
            append_thread.join(timeout=1.0)

        self.assertEqual(manager.status(7)["frames"], 2)

    def test_score_frames_honors_roi_for_current_frame(self):
        def embed_image(image):
            level = float(image.getpixel((0, 0))[0]) / 255.0
            return np.asarray([level, 1.0 - level], dtype=np.float32)

        def embed_text(text):
            return (
                np.asarray([1.0, 0.0], dtype=np.float32)
                if "bright" in text.casefold()
                else np.asarray([0.0, 1.0], dtype=np.float32)
            )

        def encode(image, **_kwargs):
            output = BytesIO()
            image.save(output, format="PNG")
            return base64.b64encode(output.getvalue()).decode("ascii")

        manager = ProbeManager(
            embed_image_fn=embed_image,
            embed_text_fn=embed_text,
            jpeg_encoder=encode,
        )
        frame = Image.new("RGB", (8, 4), color=(0, 0, 0))
        for x in range(4, 8):
            for y in range(4):
                frame.putpixel((x, y), (255, 255, 255))
        with patch.object(ProbeBuffer, "_rebuild_index", return_value=None):
            manager.add_frame(7, frame, 1_000)

        full = manager.score_frames(7, ["bright"], ["dark"])["results"][0]
        cropped = manager.score_frames(
            7,
            ["bright"],
            ["dark"],
            min_ts_ms=1_000,
            max_ts_ms=1_000,
            roi_norm=(0.5, 0.0, 0.5, 1.0),
            roi_padding=0.0,
        )["results"][0]
        self.assertLess(full["margin"], -0.9)
        self.assertGreater(cropped["margin"], 0.9)


if __name__ == "__main__":
    unittest.main()
