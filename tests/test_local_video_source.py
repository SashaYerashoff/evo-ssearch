from __future__ import annotations

import subprocess
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

import local_video_source
from local_video_source import (
    LocalVideoClient,
    LocalVideoSource,
    LocalVideoSourceRegistry,
    parse_local_video_sources,
)
from luxriot_connector import LuxriotManager


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), (12, 34, 56)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _manager() -> LuxriotManager:
    config = SimpleNamespace(
        LUXRIOT_BASE_URL="http://evo.invalid",
        LUXRIOT_USERNAME="",
        LUXRIOT_PASSWORD="",
        LOCAL_VIDEO_SOURCES=(
            {"id": 900001, "title": "USB camera", "device": "/dev/video0"},
        ),
    )
    return LuxriotManager(
        config,
        lambda _messages, _model: "summary",
        lambda *_args: [],
        lambda _image, **_kwargs: "jpeg",
    )


def test_parse_local_video_sources_returns_bounded_device_configuration():
    rows = parse_local_video_sources(
        '[{"id":900001,"title":"USB camera","device":"/dev/video0",'
        '"input_format":"mjpeg","width":1280,"height":720,"fps":15,"preview_fps":8}]'
    )

    assert rows == (
        {
            "id": 900001,
            "title": "USB camera",
            "device": "/dev/video0",
            "input_format": "mjpeg",
            "width": 1280,
            "height": 720,
            "fps": 15.0,
            "preview_fps": 8.0,
        },
    )


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '[{"id":1,"device":"rtsp://camera.invalid/live"}]',
        '[{"id":1,"device":"/dev/video0"},{"id":1,"device":"/dev/video1"}]',
        '[{"id":1,"device":"/dev/video0","input_format":"unknown"}]',
    ],
)
def test_parse_local_video_sources_rejects_unsafe_or_ambiguous_configuration(raw):
    with pytest.raises(ValueError):
        parse_local_video_sources(raw)


def test_registry_exposes_local_channel_provenance_and_no_archive():
    registry = LocalVideoSourceRegistry(
        ({"id": 900001, "title": "USB camera", "device": "/dev/video0"},)
    )

    assert registry.channels() == [
        {
            "id": 900001,
            "guid": "local-v4l2:/dev/video0",
            "title": "USB camera",
            "server": "local-v4l2",
            "ptzCapabilities": None,
            "source": "local_v4l2",
            "device": "/dev/video0",
            "archive_available": False,
        }
    ]
    assert registry.has_channel(900001) is True
    assert isinstance(registry.client_for(900001), LocalVideoClient)
    assert registry.client_for(7) is None


def test_manager_keeps_local_channel_available_when_evo_inventory_is_offline(monkeypatch):
    manager = _manager()

    class OfflineEvoClient:
        channel_inventory_meta = {"error": "offline"}

        def get_channels(self):
            raise RuntimeError("Evo is offline")

    monkeypatch.setattr(manager, "build_client", lambda: OfflineEvoClient())

    channels = manager.get_channels(force=True)

    assert [channel["id"] for channel in channels] == [900001]
    assert channels[0]["source"] == "local_v4l2"
    assert manager.channel_inventory_status()["stale"] is True
    assert "offline" in str(manager.channel_inventory_status()["last_error"]).lower()


def test_manager_routes_only_configured_local_id_around_evo(monkeypatch):
    manager = _manager()
    evo_client = object()
    monkeypatch.setattr(manager, "build_client", lambda: evo_client)

    assert isinstance(manager.build_capture_client(900001), LocalVideoClient)
    assert manager.build_capture_client(112) is evo_client


def test_local_alert_is_retained_without_attempting_evo_bookmark(monkeypatch):
    manager = _manager()
    manager.default_bookmark_enabled = True
    manager.alert_parser = lambda _text, _channel_id, _ts: [
        {"title": "Visible event", "severity": "high", "timestamp_ms": 1234}
    ]
    monkeypatch.setattr(
        manager,
        "build_client",
        lambda: (_ for _ in ()).throw(AssertionError("Evo must not be called")),
    )

    result = manager.process_summary_alerts(
        900001,
        'ALERTS_JSON:\n{"alerts":[{"title":"Visible event"}]}',
        default_ts_ms=1234,
    )

    assert int(result) == 0
    assert result.failed == 0
    assert result.alert_events[0]["delivery_status"] == "local_source_no_recorder"


def test_snapshot_uses_bounded_offline_ffmpeg_capture(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=_jpeg_bytes(), stderr=b"")

    monkeypatch.setattr(local_video_source, "_ffmpeg_binary", lambda: "/bundle/ffmpeg")
    monkeypatch.setattr(subprocess, "run", fake_run)
    client = LocalVideoClient(
        LocalVideoSource(900001, "USB camera", "/dev/video0", width=1280, height=720, fps=15)
    )

    image = client.get_snapshot(900001, timeout=3.5)

    assert image.mode == "RGB"
    assert image.size == (32, 24)
    command, kwargs = calls[0]
    assert command[0] == "/bundle/ffmpeg"
    assert command[command.index("-i") + 1] == "/dev/video0"
    assert "1280x720" in command
    assert kwargs["timeout"] == 3.5


def test_snapshot_timeout_is_reported_as_source_timeout(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 2)

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = LocalVideoClient(LocalVideoSource(900001, "USB camera", "/dev/video0"))

    with pytest.raises(TimeoutError, match="snapshot timed out"):
        client.get_snapshot(900001, timeout=2)
