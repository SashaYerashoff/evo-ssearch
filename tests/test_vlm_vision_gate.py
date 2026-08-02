from __future__ import annotations

import time
from contextlib import nullcontext
from datetime import datetime, timezone
from unittest.mock import Mock
from unittest.mock import patch

import pytest

import oldapp
from vlm_vision_health import write_health_state


def _profile():
    return {
        "id": "vlm",
        "kind": "vlm",
        "base_url": "http://vlm.local/v1",
        "model": "vlm-test",
        "api_key": "",
        "timeout": 30,
    }


def test_content_health_gate_accepts_fresh_matching_profile(tmp_path, monkeypatch):
    state_file = tmp_path / "vision-health.json"
    write_health_state(
        state_file,
        {
            "status": "healthy",
            "ok": True,
            "checked_at_epoch": time.time(),
            "base_url": "http://vlm.local/v1",
            "model": "vlm-test",
        },
    )
    monkeypatch.setattr(oldapp.config, "LM_VISION_HEALTH_STATE_FILE", str(state_file))
    monkeypatch.setattr(oldapp.config, "LM_VISION_HEALTH_MAX_AGE_SEC", 180.0)

    health = oldapp._check_vlm_vision_health(profile=_profile())

    assert health["ok"] is True
    assert health["status"] == "healthy"


def test_content_health_gate_allows_one_suspect_after_recent_success(tmp_path, monkeypatch):
    state_file = tmp_path / "vision-health.json"
    write_health_state(
        state_file,
        {
            "status": "suspect",
            "ok": False,
            "checked_at_epoch": time.time(),
            "base_url": "http://vlm.local/v1",
            "model": "vlm-test",
            "consecutive_failures": 1,
            "failure_threshold": 2,
            "last_success_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    monkeypatch.setattr(oldapp.config, "LM_VISION_HEALTH_STATE_FILE", str(state_file))
    monkeypatch.setattr(oldapp.config, "LM_VISION_HEALTH_MAX_AGE_SEC", 180.0)

    health = oldapp._check_vlm_vision_health(profile=_profile())

    assert health["ok"] is True
    assert health["status"] == "suspect"
    assert health["suspect_grace"] is True


def test_content_health_gate_blocks_suspect_without_prior_success(tmp_path, monkeypatch):
    state_file = tmp_path / "vision-health.json"
    write_health_state(
        state_file,
        {
            "status": "suspect",
            "ok": False,
            "checked_at_epoch": time.time(),
            "base_url": "http://vlm.local/v1",
            "model": "vlm-test",
            "consecutive_failures": 1,
            "failure_threshold": 2,
        },
    )
    monkeypatch.setattr(oldapp.config, "LM_VISION_HEALTH_STATE_FILE", str(state_file))
    monkeypatch.setattr(oldapp.config, "LM_VISION_HEALTH_MAX_AGE_SEC", 180.0)

    with (
        patch("oldapp._resolve_lm_profile", return_value=_profile()),
        patch("oldapp.requests.post") as post,
        pytest.raises(RuntimeError, match="quarantined"),
    ):
        oldapp._call_video_understanding([])

    post.assert_not_called()


def test_content_health_gate_blocks_result_when_watchdog_changes_midflight():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "stale visual description"},
            }
        ]
    }
    healthy = {"required": True, "ok": True, "status": "healthy"}
    suspect = {"required": True, "ok": False, "status": "suspect"}

    with (
        patch("oldapp._resolve_lm_profile", return_value=_profile()),
        patch("oldapp._check_vlm_vision_health", side_effect=[healthy, suspect]),
        patch.object(
            oldapp._lm_admission_controller,
            "admission",
            return_value=nullcontext(),
        ),
        patch("oldapp.requests.post", return_value=response),
        pytest.raises(RuntimeError, match="phase=postflight"),
    ):
        oldapp._call_video_understanding([])
