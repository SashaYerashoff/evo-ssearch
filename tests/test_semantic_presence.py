from __future__ import annotations

from unittest.mock import patch

import numpy as np
from PIL import Image

from probe_manager import ProbeBuffer, ProbeManager
from semantic_presence import SemanticPresenceTracker, normalize_presence_labels
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
