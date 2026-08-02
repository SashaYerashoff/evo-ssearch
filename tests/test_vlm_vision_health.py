from __future__ import annotations

import io
import json
from unittest.mock import patch

from scripts import vlm_vision_watchdog
from vlm_vision_health import (
    VisionCanaryResult,
    build_control_png,
    probe_vision,
    read_health_state,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_control_png_is_valid_and_changes_with_visual_facts():
    first = build_control_png("7391", ("RED", "GREEN", "BLUE"))
    second = build_control_png("1937", ("BLUE", "RED", "GREEN"))

    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert second.startswith(b"\x89PNG\r\n\x1a\n")
    assert first != second


def test_probe_accepts_ordered_visual_facts_without_echo_prefix():
    response = {
        "choices": [{"message": {"content": "4552 red green blue"}}]
    }

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://vlm.local/v1/chat/completions"
        assert timeout == 12
        payload = json.loads(request.data)
        assert payload["model"] == "vlm-test"
        assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        return FakeResponse(json.dumps(response).encode("utf-8"))

    with patch("vlm_vision_health.urllib.request.urlopen", fake_urlopen):
        result = probe_vision(
            "http://vlm.local/v1",
            "vlm-test",
            timeout_sec=12,
            seed=7391,
        )

    assert result.ok is True
    assert result.expected == "VISION_OK 4552 RED GREEN BLUE"


def test_probe_tolerates_one_ocr_digit_but_not_stale_visual_facts():
    one_digit_error = {
        "choices": [{"message": {"content": "VISION_OK 4592 red green blue"}}]
    }
    stale = {
        "choices": [{"message": {"content": "VISION_OK 7391 blue red green"}}]
    }

    with patch(
        "vlm_vision_health.urllib.request.urlopen",
        return_value=FakeResponse(json.dumps(one_digit_error).encode("utf-8")),
    ):
        assert probe_vision("http://vlm.local/v1", "vlm-test", seed=7391).ok is True
    with patch(
        "vlm_vision_health.urllib.request.urlopen",
        return_value=FakeResponse(json.dumps(stale).encode("utf-8")),
    ):
        assert probe_vision("http://vlm.local/v1", "vlm-test", seed=7391).ok is False


def test_watchdog_quarantines_first_mismatch_and_escalates_second(tmp_path):
    state_file = tmp_path / "vision-health.json"
    failed = VisionCanaryResult(
        ok=False,
        expected="VISION_OK 7391 RED GREEN BLUE",
        observed="glass containers",
        latency_ms=10.0,
        error="visual_control_mismatch",
    )
    argv = [
        "--base-url",
        "http://vlm.local/v1",
        "--model",
        "vlm-test",
        "--state-file",
        str(state_file),
        "--failure-threshold",
        "2",
    ]

    with patch.object(vlm_vision_watchdog, "probe_vision", return_value=failed):
        assert vlm_vision_watchdog.main(argv) == 0
        assert read_health_state(state_file)["status"] == "suspect"
        assert vlm_vision_watchdog.main(argv) == 2

    state = read_health_state(state_file)
    assert state["status"] == "degraded"
    assert state["consecutive_failures"] == 2
