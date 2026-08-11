from __future__ import annotations

import io
import json
from unittest.mock import patch

from scripts import vlm_vision_watchdog
from vlm_vision_health import (
    EndpointLivenessResult,
    EndpointWorkloadResult,
    VisionCanaryResult,
    build_control_png,
    probe_openai_workload,
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


def test_probe_treats_ocr_as_advisory_but_rejects_stale_colour_order():
    digit_errors = {
        "choices": [{"message": {"content": "VISION_OK 2024 red green blue"}}]
    }
    stale = {
        "choices": [{"message": {"content": "VISION_OK 7391 blue red green"}}]
    }

    with patch(
        "vlm_vision_health.urllib.request.urlopen",
        return_value=FakeResponse(json.dumps(digit_errors).encode("utf-8")),
    ):
        assert probe_vision("http://vlm.local/v1", "vlm-test", seed=7391).ok is True
    with patch(
        "vlm_vision_health.urllib.request.urlopen",
        return_value=FakeResponse(json.dumps(stale).encode("utf-8")),
    ):
        assert probe_vision("http://vlm.local/v1", "vlm-test", seed=7391).ok is False


def test_workload_probe_treats_a_processing_slot_as_busy_without_a_queue():
    metrics = b"""
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
"""

    with patch(
        "vlm_vision_health.urllib.request.urlopen",
        return_value=FakeResponse(metrics),
    ):
        result = probe_openai_workload("http://vlm.local/v1")

    assert result.known is True
    assert result.busy is True
    assert result.processing == 1
    assert result.deferred == 0


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

    idle = EndpointWorkloadResult(True, False, 0, 0, 1.0)
    live = EndpointLivenessResult(True, 1.0)
    with (
        patch.object(vlm_vision_watchdog, "probe_vision", return_value=failed),
        patch.object(vlm_vision_watchdog, "probe_openai_workload", return_value=idle),
        patch.object(vlm_vision_watchdog, "probe_openai_liveness", return_value=live),
    ):
        assert vlm_vision_watchdog.main(argv) == 0
        assert read_health_state(state_file)["status"] == "suspect"
        assert vlm_vision_watchdog.main(argv) == 2

    state = read_health_state(state_file)
    assert state["status"] == "degraded"
    assert state["consecutive_failures"] == 2


def test_watchdog_treats_timed_out_canary_as_busy_when_api_is_alive(tmp_path):
    state_file = tmp_path / "vision-health.json"
    state_file.write_text(
        json.dumps(
            {
                "status": "healthy",
                "ok": True,
                "consecutive_failures": 0,
                "last_success_at": "2026-08-11T06:00:00+00:00",
            }
        )
    )
    timed_out = VisionCanaryResult(
        ok=False,
        expected="VISION_OK 7391 RED GREEN BLUE",
        observed="",
        latency_ms=30000.0,
        error="TimeoutError: timed out",
    )
    live = EndpointLivenessResult(ok=True, latency_ms=2.5)
    idle = EndpointWorkloadResult(True, False, 1, 0, 1.0)
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

    with (
        patch.object(vlm_vision_watchdog, "probe_vision", return_value=timed_out),
        patch.object(
            vlm_vision_watchdog,
            "probe_openai_workload",
            return_value=idle,
        ),
        patch.object(
            vlm_vision_watchdog,
            "probe_openai_liveness",
            return_value=live,
        ),
    ):
        assert vlm_vision_watchdog.main(argv) == 0

    state = read_health_state(state_file)
    assert state["status"] == "busy"
    assert state["ok"] is True
    assert state["vision_ok"] is False
    assert state["consecutive_failures"] == 0
    assert state["endpoint_liveness_ok"] is True


def test_watchdog_skips_canary_when_metrics_report_a_deferred_queue(tmp_path):
    state_file = tmp_path / "vision-health.json"
    state_file.write_text(
        json.dumps({"status": "degraded", "consecutive_failures": 3})
    )
    live = EndpointLivenessResult(ok=True, latency_ms=1.5)
    busy = EndpointWorkloadResult(
        known=True,
        busy=True,
        processing=1,
        deferred=2,
        latency_ms=2.0,
    )
    argv = [
        "--base-url",
        "http://vlm.local/v1",
        "--model",
        "vlm-test",
        "--state-file",
        str(state_file),
        "--failure-threshold",
        "3",
    ]

    with (
        patch.object(vlm_vision_watchdog, "probe_openai_liveness", return_value=live),
        patch.object(vlm_vision_watchdog, "probe_openai_workload", return_value=busy),
        patch.object(vlm_vision_watchdog, "probe_vision") as vision,
    ):
        assert vlm_vision_watchdog.main(argv) == 0

    vision.assert_not_called()
    state = read_health_state(state_file)
    assert state["status"] == "busy"
    assert state["consecutive_failures"] == 0
    assert state["workload_deferred"] == 2
