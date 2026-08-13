from __future__ import annotations

from unittest.mock import patch
import base64
from io import BytesIO

import numpy as np
from PIL import Image

from probe_manager import ProbeBuffer, ProbeManager
from semantic_presence import SemanticPresenceTracker, normalize_presence_labels
from semantic_patch_attention import build_patch_affinity_payload
from luxriot_connector import LuxriotManager


def test_presence_homeostasis_warms_then_exposes_bounded_deviation():
    tracker = SemanticPresenceTracker(
        ("person",),
        warmup_samples=3,
        history_size=10,
        alpha=0.1,
        noise_floor=0.01,
    )
    for timestamp in (1_000, 2_000, 3_000):
        tracker.update(7, timestamp, {"person": 0.1})

    status = tracker.update(7, 4_000, {"person": 0.2})
    person = status["classes"][0]

    assert status["state"] == "ready"
    assert person["state"] == "above_baseline"
    assert person["score"] == 0.2
    assert 0.1 < person["baseline"] < 0.12
    assert person["delta"] > 0.08
    assert len(person["history"]) == 4


def test_presence_registry_is_canonical_bounded_and_keeps_core_labels():
    tracker = SemanticPresenceTracker(
        ("person", "vehicle"),
        maximum_classes=4,
    )
    labels = tracker.set_channel_labels(
        112,
        (" Person ", "trolleybus", "animal", "smoke", "ignored"),
    )

    assert labels == ("person", "vehicle", "trolleybus", "animal")
    assert normalize_presence_labels((" Cat ", "cat", "", "DOG")) == ("cat", "dog")


def test_probe_manager_scores_presence_from_the_existing_image_vector():
    image_calls = []
    text_calls = []

    def embed_image(_image):
        image_calls.append(True)
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_texts(texts):
        text_calls.append(list(texts))
        return np.asarray(
            [
                [1.0, 0.0] if "person" in text or "people" in text else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )

    manager = ProbeManager(
        embed_image_fn=embed_image,
        embed_text_fn=lambda _text: np.asarray([0.0, 1.0], dtype=np.float32),
        embed_texts_fn=embed_texts,
        jpeg_encoder=lambda _image, **_kwargs: "jpeg",
        semantic_presence_enabled=True,
        semantic_presence_classes=("person", "vehicle"),
    )
    phrases = [
        prompt
        for _label, prompts in manager.semantic_presence_tracker.prompt_plan(7)
        for prompt in prompts
    ]
    manager.prewarm_texts(phrases)

    with patch.object(ProbeBuffer, "_rebuild_index", return_value=None):
        result = manager.add_frame(
            7,
            Image.new("RGB", (4, 4), color=(255, 255, 255)),
            1_000,
        )

    assert len(image_calls) == 1
    assert len(text_calls) == 1
    presence = result["semantic_presence"]
    scores = {item["label"]: item.get("score") for item in presence["classes"]}
    assert scores["person"] == 1.0
    assert scores["vehicle"] == 0.0
    assert manager.status(7)["semantic_presence"]["semantics"].endswith(
        "not_object_detection"
    )


def test_same_forward_patch_shadow_is_not_part_of_embedding_space():
    def embed_with_metadata(_image):
        return np.asarray([1.0, 0.0], dtype=np.float32), {
            "backend": "siglip2",
            "model": "test",
            "dimension": 2,
            "_semantic_patch_presence_v1": {
                "semantics": "same_forward_top_patch_text_affinity_shadow_v1",
                "classes": {
                    "person": {"score": 0.12, "contrast": 0.04},
                    "vehicle": {"score": 0.31, "contrast": 0.09},
                },
            },
        }

    manager = ProbeManager(
        embed_image_fn=lambda _image: np.asarray([1.0, 0.0], dtype=np.float32),
        embed_image_with_metadata_fn=embed_with_metadata,
        embed_text_fn=lambda _text: np.asarray([1.0, 0.0], dtype=np.float32),
        embed_texts_fn=lambda texts: np.asarray(
            [[1.0, 0.0] for _text in texts],
            dtype=np.float32,
        ),
        jpeg_encoder=lambda _image, **_kwargs: "jpeg",
        semantic_presence_enabled=True,
        semantic_presence_classes=("person", "vehicle"),
    )
    phrases = [
        prompt
        for _label, prompts in manager.semantic_presence_tracker.prompt_plan(7)
        for prompt in prompts
    ]
    manager.prewarm_texts(phrases)

    with patch.object(ProbeBuffer, "_rebuild_index", return_value=None):
        result = manager.add_frame(
            7,
            Image.new("RGB", (4, 4), color=(255, 255, 255)),
            1_000,
        )

    assert "_semantic_patch_presence_v1" not in result["embedding_space"]
    by_label = {
        item["label"]: item
        for item in result["semantic_presence"]["classes"]
    }
    assert by_label["person"]["spatial_score"] == 0.12
    assert by_label["vehicle"]["spatial_score"] == 0.31
    assert by_label["vehicle"]["spatial_contrast"] == 0.09
    status = manager.status(7)["semantic_presence"]
    assert status["spatial_semantics"] == (
        "same_forward_top_patch_text_affinity_shadow_v1"
    )


def test_presence_is_archived_compactly_but_not_injected_into_vlm_prompt():
    raw = {
        "version": 1,
        "channel_id": 7,
        "semantic_presence": {
            "enabled": True,
            "shadow": True,
            "state": "ready",
            "timestamp_ms": 4_000,
            "classes": [
                {
                    "label": "person",
                    "score": 0.2,
                    "baseline": 0.1,
                    "delta": 0.1,
                    "z": 10.0,
                    "samples": 40,
                    "state": "above_baseline",
                    "history": [{"timestamp_ms": 4_000, "score": 0.2}],
                }
            ],
        },
        "motion_intervals": [
            {
                "started_at_ms": 3_000,
                "ended_at_ms": 4_000,
                "state": "quiet",
                "sample_count": 1,
                "motion_max": 0.0,
            }
        ],
    }

    compact = LuxriotManager._compact_vector_signal(raw)
    prompt = LuxriotManager._vector_signal_prompt_view(raw)

    assert compact["semantic_presence"]["classes"][0]["label"] == "person"
    assert "history" not in compact["semantic_presence"]["classes"][0]
    assert "semantic_presence" not in prompt


def test_patch_affinity_is_relative_bounded_and_returns_only_a_roi_hint():
    patches = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    payload = build_patch_affinity_payload(
        patches,
        np.asarray([1.0, 0.0], dtype=np.float32),
        rows=2,
        cols=2,
        minimum_contrast=0.0,
    )

    assert payload["grid"] == {"rows": 2, "cols": 2}
    assert len(payload["heatmap"]) == 4
    assert all(0.0 <= value <= 1.0 for value in payload["heatmap"])
    assert payload["semantics"].endswith("not_detection")
    assert payload["suggested_roi"] is not None
    assert "patch_embeddings" not in payload


def test_patch_attention_runs_only_on_explicit_exact_frame_request():
    image_calls = []
    patch_calls = []

    def encode_jpeg(image, **_kwargs):
        stream = BytesIO()
        image.save(stream, format="JPEG")
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def embed_image(_image):
        image_calls.append(True)
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_texts(texts):
        return np.asarray(
            [
                [1.0, 0.0] if text == "a person" else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )

    def inspect(image, text_vector, prompt):
        patch_calls.append((image.size, text_vector.copy(), prompt))
        return {
            "semantics": "experimental_relative_patch_text_affinity_not_detection",
            "grid": {"rows": 1, "cols": 1},
            "heatmap": [1.0],
            "suggested_roi": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        }

    manager = ProbeManager(
        embed_image_fn=embed_image,
        embed_text_fn=lambda _text: np.asarray([0.0, 1.0], dtype=np.float32),
        embed_texts_fn=embed_texts,
        jpeg_encoder=encode_jpeg,
        semantic_presence_classes=("person",),
        patch_attention_fn=inspect,
        patch_attention_enabled=True,
    )
    manager.patch_attention_min_interval_sec = 0.0
    frame = Image.new("RGB", (8, 6), color=(240, 240, 240))
    added = manager.add_frame(7, frame, 1_000)

    assert len(image_calls) == 1
    assert patch_calls == []
    result = manager.patch_attention(7, 1_000, "person")

    assert len(image_calls) == 1
    assert len(patch_calls) == 1
    assert patch_calls[0][0] == (8, 6)
    assert patch_calls[0][2] == "a person"
    assert result["frame_url"] == "/probes/signal_frame/7/1000"
    assert result["ephemeral"] is True
    assert result["class_key"] == "person"
    assert manager.patch_attention(7, 1_000, "not-registered")["error_code"] == (
        "patch_attention_class_invalid"
    )
