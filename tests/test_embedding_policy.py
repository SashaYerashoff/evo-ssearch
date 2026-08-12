import base64
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
            "EMBEDDER_FALLBACK_ENABLED": getattr(
                config,
                "EMBEDDER_FALLBACK_ENABLED",
                False,
            ),
            "CLIP_DEVICE": getattr(config, "CLIP_DEVICE", "auto"),
            "CLIP_MODEL": config.CLIP_MODEL,
            "CLIP_MODEL_REVISION": getattr(config, "CLIP_MODEL_REVISION", ""),
            "INDEX_MODE": config.INDEX_MODE,
            "FUSION_ENABLED": config.FUSION_ENABLED,
            "DINO_SEGMENTS_ENABLED": config.DINO_SEGMENTS_ENABLED,
            "AUTH_ENABLED": config.AUTH_ENABLED,
            "SETTINGS_LOCAL_ONLY": config.SETTINGS_LOCAL_ONLY,
            "OFFLINE_MODE": config.OFFLINE_MODE,
            "OPENAI_CLIP_CACHE_DIR": config.OPENAI_CLIP_CACHE_DIR,
            "MODEL_CACHE_DIR": config.MODEL_CACHE_DIR,
        }
        config.AUTH_ENABLED = False
        config.SETTINGS_LOCAL_ONLY = True
        config.EMBEDDER_FALLBACK_ENABLED = False
        config.CLIP_DEVICE = "auto"

    def tearDown(self) -> None:
        oldapp.reset_embedder_runtime_state()
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

    def test_settings_read_defers_expensive_archive_storage_scan(self) -> None:
        with patch("oldapp._archive_storage_summary") as storage_summary:
            resp = self.client.get("/settings")

        self.assertEqual(resp.status_code, 200)
        storage_summary.assert_not_called()
        archive_summary = resp.get_json()["settings"]["archiveStorageSummary"]
        self.assertFalse(archive_summary["available"])
        self.assertTrue(archive_summary["deferred"])

    def test_archive_capacity_can_skip_or_request_current_storage_scan(self) -> None:
        current = {"available": True, "row_count": 73}
        with patch("oldapp._archive_storage_summary", return_value=current) as storage_summary:
            deferred = self.client.get("/settings/archive_capacity?include_current=false")
            scanned = self.client.get("/settings/archive_capacity?include_current=true")

        self.assertEqual(deferred.status_code, 200)
        self.assertTrue(deferred.get_json()["current"]["deferred"])
        self.assertEqual(scanned.status_code, 200)
        self.assertEqual(scanned.get_json()["current"]["row_count"], 73)
        storage_summary.assert_called_once_with()

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

    def test_openai_clip_fails_closed_when_offline_artifact_is_missing(self) -> None:
        config.OFFLINE_MODE = True
        with tempfile.TemporaryDirectory() as temp_dir:
            config.OPENAI_CLIP_CACHE_DIR = Path(temp_dir)
            clip_module = SimpleNamespace(
                _MODELS={
                    "ViT-B/32": "https://models.example.invalid/clip.pt",
                },
                load=Mock(),
            )
            with patch(
                "oldapp._get_clip_module",
                return_value=clip_module,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "not present in offline cache",
                ):
                    oldapp._load_openai_clip_model("ViT-B/32", "cpu")

            clip_module.load.assert_not_called()

    def test_openai_clip_uses_legacy_offline_cache_without_public_registry(self) -> None:
        config.OFFLINE_MODE = True
        with tempfile.TemporaryDirectory() as temp_dir:
            config.OPENAI_CLIP_CACHE_DIR = Path(temp_dir)
            cached_model = Path(temp_dir) / "ViT-B-32.pt"
            cached_model.write_bytes(b"cached model")
            model = Mock()
            clip_module = SimpleNamespace(
                clip=SimpleNamespace(
                    _MODELS={
                        "ViT-B/32": "https://models.example.invalid/ViT-B-32.pt",
                    }
                ),
                load=Mock(return_value=(model, Mock())),
            )
            with patch("oldapp._get_clip_module", return_value=clip_module):
                loaded, _preprocess = oldapp._load_openai_clip_model(
                    "ViT-B/32",
                    "cpu",
                )

            self.assertIs(loaded, model)
            clip_module.load.assert_called_once_with(
                str(cached_model),
                device="cpu",
                download_root=temp_dir,
            )

    def test_siglip_loader_forces_local_cache_in_offline_mode(self) -> None:
        config.OFFLINE_MODE = True
        config.MODEL_CACHE_DIR = Path("/var/lib/eva-ai/models/huggingface")
        model = Mock()
        model.to.return_value = model
        with (
            patch(
                "oldapp.AutoModel.from_pretrained",
                return_value=model,
            ) as load_model,
            patch(
                "oldapp.AutoProcessor.from_pretrained",
                return_value=Mock(),
            ) as load_processor,
        ):
            oldapp._load_siglip2_clip_model("local/siglip", "cpu")

        self.assertTrue(load_model.call_args.kwargs["local_files_only"])
        self.assertEqual(
            load_model.call_args.kwargs["cache_dir"],
            "/var/lib/eva-ai/models/huggingface",
        )
        self.assertTrue(load_processor.call_args.kwargs["local_files_only"])
        self.assertEqual(load_processor.call_args.kwargs["backend"], "torchvision")

    def test_siglip_transformers_five_pooler_contract_and_dimension(self) -> None:
        pooled = oldapp.torch.ones((2, 768), dtype=oldapp.torch.float32)
        output = SimpleNamespace(pooler_output=pooled)
        self.assertIs(oldapp._siglip_feature_tensor(output), pooled)

        model = SimpleNamespace(
            config=SimpleNamespace(
                projection_dim=None,
                text_config=SimpleNamespace(projection_size=768),
                vision_config=SimpleNamespace(projection_size=768),
            )
        )
        with patch.object(oldapp, "clip_model", model):
            self.assertEqual(oldapp._siglip_projection_dimension(), 768)

    def test_siglip_init_fails_closed_instead_of_changing_embedding_space(self) -> None:
        config.EXPERIMENTAL_EMBEDDERS_ENABLED = True
        config.CLIP_MODEL = "google/siglip2-base-patch16-224"
        config.EMBEDDER_FALLBACK_ENABLED = False
        oldapp.reset_embedder_runtime_state()

        with (
            patch(
                "oldapp._load_siglip2_clip_model",
                side_effect=RuntimeError("missing local artifact"),
            ),
            patch("oldapp._load_openai_clip_model") as load_clip,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "embedding fallback is disabled",
            ):
                oldapp.init_clip()

        load_clip.assert_not_called()

    def test_siglip_cold_start_is_single_flight(self) -> None:
        config.EXPERIMENTAL_EMBEDDERS_ENABLED = True
        config.CLIP_MODEL = "google/siglip2-base-patch16-224"
        config.EMBEDDER_FALLBACK_ENABLED = False
        oldapp.reset_embedder_runtime_state()
        model = Mock()
        model.config = SimpleNamespace(_commit_hash="revision")
        processor = Mock()

        def slow_load(*_args, **_kwargs):
            time.sleep(0.05)
            return model, processor

        with patch(
            "oldapp._load_siglip2_clip_model",
            side_effect=slow_load,
        ) as load_siglip:
            threads = [threading.Thread(target=oldapp.init_clip) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

        self.assertEqual(load_siglip.call_count, 1)

    def test_siglip_init_uses_clip_fallback_only_when_explicitly_enabled(self) -> None:
        config.EXPERIMENTAL_EMBEDDERS_ENABLED = True
        config.CLIP_MODEL = "google/siglip2-base-patch16-224"
        config.EMBEDDER_FALLBACK_ENABLED = True
        oldapp.reset_embedder_runtime_state()
        fallback_model = Mock()
        fallback_preprocess = Mock()

        with (
            patch(
                "oldapp._load_siglip2_clip_model",
                side_effect=RuntimeError("missing local artifact"),
            ),
            patch(
                "oldapp._load_openai_clip_model",
                return_value=(fallback_model, fallback_preprocess),
            ) as load_clip,
        ):
            oldapp.init_clip()

        load_clip.assert_called_once_with("ViT-B/32", oldapp.device)
        self.assertIs(oldapp.clip_model, fallback_model)
        self.assertEqual(oldapp.clip_backend_kind, "openai_clip")
        self.assertEqual(oldapp.clip_runtime_model, "ViT-B/32")


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


class ProbeCaptureWarmupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_warmup = config.PROBE_CAPTURE_WARMUP_SEC

    def tearDown(self) -> None:
        config.PROBE_CAPTURE_WARMUP_SEC = self.original_warmup

    def test_synthetic_benchmark_cannot_block_active_live_capture(self) -> None:
        live_session = SimpleNamespace()
        with (
            patch.object(oldapp.luxriot_manager, "sessions", {112: live_session}),
            patch.object(oldapp.luxriot_manager, "probe_sessions", {118: live_session}),
            patch("oldapp.init_clip") as init_clip,
            app.test_request_context("/probes/bench"),
        ):
            response, status = oldapp.probes_bench()

        payload = response.get_json()
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "benchmark_blocked_by_live_capture")
        self.assertEqual(payload["active_channel_ids"], [112, 118])
        init_clip.assert_not_called()

    def test_empty_query_starts_capture_and_retries_after_first_frame(self) -> None:
        config.PROBE_CAPTURE_WARMUP_SEC = 0.1
        query = Mock(
            side_effect=[
                {"results": [], "frames_indexed": 0},
                {
                    "results": [{"margin": 0.7}],
                    "frames_indexed": 1,
                    "status": {"frames": 1},
                },
            ]
        )
        with (
            patch(
                "oldapp.luxriot_manager.is_probe_capture_paused",
                return_value=False,
            ),
            patch(
                "oldapp.luxriot_manager.start_probe_capture",
                return_value={"running": True, "channel_id": 7},
            ) as start_capture,
            patch(
                "oldapp.probe_manager.status",
                side_effect=[{"frames": 0}, {"frames": 1}],
            ),
        ):
            result = oldapp._query_probe_with_capture_warmup(
                channel_id=7,
                fps=2.0,
                query=query,
            )

        self.assertEqual(query.call_count, 2)
        start_capture.assert_called_once_with(
            7,
            fps=2.0,
            clear_pause=False,
        )
        self.assertTrue(result["capture_warmup_retry"])
        self.assertFalse(result["capture_warming_up"])
        self.assertEqual(result["frames_indexed"], 1)

    def test_empty_query_respects_operator_capture_pause(self) -> None:
        query = Mock(return_value={"results": [], "frames_indexed": 0})
        with (
            patch(
                "oldapp.luxriot_manager.is_probe_capture_paused",
                return_value=True,
            ),
            patch(
                "oldapp.luxriot_manager.start_probe_capture",
            ) as start_capture,
        ):
            result = oldapp._query_probe_with_capture_warmup(
                channel_id=7,
                query=query,
            )

        start_capture.assert_not_called()
        self.assertEqual(result["capture_state"], "paused")
        self.assertFalse(result["capture_warming_up"])

    def test_new_probe_defaults_are_conservative_and_centralized(self) -> None:
        embedding_space = {
            "backend": "siglip2",
            "model": "google/siglip2-base-patch16-224",
            "revision": "test-revision",
        }
        with patch(
            "oldapp.get_probe_embedding_space",
            return_value=embedding_space,
        ):
            payload = oldapp._build_probe_payload(
                {
                    "channel_id": 7,
                    "name": "person",
                    "positives": ["a person"],
                }
            )

        self.assertEqual(payload["pos_floor"], config.PROBE_POS_FLOOR_DEFAULT)
        self.assertEqual(payload["margin"], config.PROBE_MARGIN_DEFAULT)
        self.assertEqual(payload["embedding_space"], embedding_space)


if __name__ == "__main__":
    unittest.main()
