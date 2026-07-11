import base64
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

import oldapp
from oldapp import app, config
from probe_manager import ProbeBuffer, ProbeManager


def _jpeg_b64() -> str:
    img = Image.new("RGB", (8, 8), color=(128, 32, 16))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class EmbeddingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self._original = {
            "EXPERIMENTAL_EMBEDDERS_ENABLED": getattr(config, "EXPERIMENTAL_EMBEDDERS_ENABLED", False),
            "PRODUCTION_CLIP_MODEL": getattr(config, "PRODUCTION_CLIP_MODEL", "ViT-B/32"),
            "EMBEDDER": config.EMBEDDER,
            "CLIP_MODEL": config.CLIP_MODEL,
            "INDEX_MODE": config.INDEX_MODE,
            "FUSION_ENABLED": config.FUSION_ENABLED,
            "DINO_SEGMENTS_ENABLED": config.DINO_SEGMENTS_ENABLED,
            "AUTH_ENABLED": config.AUTH_ENABLED,
            "SETTINGS_LOCAL_ONLY": config.SETTINGS_LOCAL_ONLY,
        }
        config.AUTH_ENABLED = False
        config.SETTINGS_LOCAL_ONLY = True

    def tearDown(self) -> None:
        for key, value in self._original.items():
            setattr(config, key, value)

    def test_settings_payload_locks_experimental_embedding_modes_by_default(self) -> None:
        config.EXPERIMENTAL_EMBEDDERS_ENABLED = False
        config.PRODUCTION_CLIP_MODEL = "ViT-B/32"
        config.EMBEDDER = "fusion"
        config.CLIP_MODEL = "google/siglip2-base-patch16-224"
        config.INDEX_MODE = "dual"
        config.FUSION_ENABLED = True
        config.DINO_SEGMENTS_ENABLED = True

        resp = self.client.get("/settings")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertTrue(payload["success"])
        settings = payload["settings"]

        self.assertFalse(settings["experimentalEmbeddersEnabled"])
        self.assertEqual(settings["embedder"], "clip")
        self.assertEqual(settings["clipModel"], "ViT-B/32")
        self.assertEqual(settings["indexMode"], "clip")
        self.assertFalse(settings["fusionEnabled"])
        self.assertFalse(settings["segmentsEnabled"])

    def test_policy_normalizers_allow_experimental_models_only_when_enabled(self) -> None:
        config.EXPERIMENTAL_EMBEDDERS_ENABLED = False
        config.PRODUCTION_CLIP_MODEL = "ViT-B/32"
        self.assertEqual(oldapp._normalize_embedder_for_policy("dino", True), "clip")
        self.assertEqual(
            oldapp._normalize_clip_model_for_policy("google/siglip2-base-patch16-224"),
            "ViT-B/32",
        )

        config.EXPERIMENTAL_EMBEDDERS_ENABLED = True
        self.assertEqual(oldapp._normalize_embedder_for_policy("dino", True), "dino")
        self.assertEqual(
            oldapp._normalize_clip_model_for_policy("google/siglip2-base-patch16-224"),
            "google/siglip2-base-patch16-224",
        )

    def test_index_metadata_rejects_mismatched_clip_model_in_production(self) -> None:
        config.EXPERIMENTAL_EMBEDDERS_ENABLED = False
        config.PRODUCTION_CLIP_MODEL = "ViT-B/32"
        config.CLIP_MODEL = "ViT-B/32"

        self.assertTrue(
            oldapp._index_metadata_compatible(
                "clip",
                {"model": "ViT-B/32", "backend": "openai_clip"},
            )
        )
        self.assertFalse(
            oldapp._index_metadata_compatible(
                "clip",
                {"model": "google/siglip2-base-patch16-224", "backend": "siglip2"},
            )
        )


class ProbeVectorGuardTests(unittest.TestCase):
    def test_live_buffer_preserves_capture_apex_provenance(self) -> None:
        manager = ProbeManager(
            embed_image_fn=lambda _img: np.ones(4, dtype=np.float32),
            embed_text_fn=lambda _text: np.ones(4, dtype=np.float32),
            jpeg_encoder=lambda *_args, **_kwargs: "thumb",
        )
        image = Image.new("RGB", (8, 8), color="white")

        manager.add_frame(
            7,
            image,
            100500,
            provenance={
                "selection_source": "capture_cv_frame_delta",
                "selected_source_frame_index": 2,
                "selected_frame_hash": "frame-hash",
                "source_frame_indices": [1, 2, 3],
                "fallback_reason": "",
                "ignored_secret": "must-not-be-stored",
            },
        )

        stored = manager.buffers[7].meta[0]["selection_provenance"]
        self.assertEqual(stored["selection_source"], "capture_cv_frame_delta")
        self.assertEqual(stored["selected_source_frame_index"], 2)
        self.assertEqual(stored["source_frame_indices"], [1, 2, 3])
        self.assertNotIn("ignored_secret", stored)
        result = manager.query(
            7,
            positives=["motion"],
            negatives=[],
            pos_floor=0.1,
            margin_thr=0.0,
            top_k=1,
        )
        self.assertEqual(
            result["results"][0]["selection_provenance"]["selected_frame_hash"],
            "frame-hash",
        )

    def test_text_and_image_probe_dimension_mismatch_returns_error(self) -> None:
        manager = ProbeManager(
            embed_image_fn=lambda _img: np.ones(4, dtype=np.float32),
            embed_text_fn=lambda _text: np.ones(3, dtype=np.float32),
            jpeg_encoder=lambda *_args, **_kwargs: "",
        )

        result = manager.query(
            1,
            positives=["person"],
            negatives=[],
            pos_floor=0.1,
            margin_thr=0.0,
            top_k=3,
            image_probe={"data": _jpeg_b64(), "enabled": True},
        )

        self.assertIn("error", result)
        self.assertIn("different dimensions", result["error"])

    def test_live_buffer_dimension_mismatch_is_cleared(self) -> None:
        manager = ProbeManager(
            embed_image_fn=lambda _img: np.ones(4, dtype=np.float32),
            embed_text_fn=lambda _text: np.ones(4, dtype=np.float32),
            jpeg_encoder=lambda *_args, **_kwargs: "",
        )
        buffer = ProbeBuffer(max_frames=10, thumb_edge=64)
        buffer.embeddings = [np.ones(3, dtype=np.float32)]
        buffer.meta = [
            {
                "uid": 1,
                "timestamp_ms": 1000,
                "channel_id": 7,
                "thumb": "",
            }
        ]
        manager.buffers[7] = buffer

        result = manager.query(
            7,
            positives=["person"],
            negatives=[],
            pos_floor=0.1,
            margin_thr=0.0,
            top_k=3,
        )

        self.assertIn("error", result)
        self.assertEqual(manager.status(7)["frames"], 0)


if __name__ == "__main__":
    unittest.main()
