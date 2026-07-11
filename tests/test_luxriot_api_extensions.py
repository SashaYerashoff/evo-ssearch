from __future__ import annotations

import json
from io import BytesIO

import pytest
import requests
from PIL import Image

from luxriot_connector import LuxriotClient
from road_events import capture_luxriot_archive_mp4_segment, iter_luxriot_archive_snapshots


def _jpeg_bytes(width: int = 32, height: int = 24) -> bytes:
    image = Image.new("RGB", (width, height), color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, *, payload=None, text="", content=b"", headers=None, lines=None):
        self._payload = payload
        self.text = text
        self.content = content
        self.headers = dict(headers or {})
        self.lines = list(lines or [])
        self.status_code = 200
        self.closed = False

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, _chunk_size=None, decode_unicode=False, **_kwargs):
        if self.lines:
            for line in self.lines:
                if decode_unicode and isinstance(line, bytes):
                    yield line.decode("utf-8")
                else:
                    yield line
            return
        yield self.content

    def iter_lines(self, decode_unicode=False):
        for line in self.lines:
            if decode_unicode and isinstance(line, bytes):
                yield line.decode("utf-8")
            else:
                yield line

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.auth = None

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        return self.responses.pop(0)


def test_luxriot_client_merges_fragmented_channel_initial_state_and_deltas():
    response = FakeResponse(
        lines=[
            json.dumps(
                {
                    "type": "initial",
                    "hasMore": True,
                    "added": {"channels": [{"id": 7, "title": "North", "server": 1}]},
                }
            ),
            json.dumps(
                {
                    "type": "initial",
                    "hasMore": True,
                    "added": {"channels": [{"id": 8, "title": "South", "server": 1}]},
                }
            ),
            json.dumps(
                {
                    "type": "initial_complete",
                    "updated": {"channels": [{"id": 7, "title": "North gate"}]},
                    "removed": {"channels": [8]},
                    "added": {"channels": [{"id": 9, "title": "West", "server": 2}]},
                }
            ),
        ]
    )
    client = LuxriotClient("http://luxriot.test", "user", "pass")
    client.session = FakeSession([response])
    client.CHANNEL_STREAM_SETTLE_SEC = 0.01

    channels = client.get_channels()

    assert channels == [
        {"id": 7, "guid": None, "title": "North gate", "server": 1, "ptzCapabilities": None},
        {"id": 9, "guid": None, "title": "West", "server": 2, "ptzCapabilities": None},
    ]
    assert client.channel_inventory_meta["complete"] is True
    assert client.channel_inventory_meta["completion"] == "explicit"
    assert client.channel_inventory_meta["payload_count"] == 3
    assert response.closed is True


def test_luxriot_client_applies_unmarked_resource_deltas_in_initial_burst():
    response = FakeResponse(
        lines=[
            json.dumps({"added": {"channels": [{"id": 7, "title": "Original"}]}}),
            json.dumps(
                {
                    "updated": {"channels": [{"id": 7, "title": "Updated"}]},
                    "added": {"channels": [{"id": 8, "title": "Temporary"}]},
                }
            ),
            json.dumps({"removed": {"channels": [{"id": 8}]}}),
        ]
    )
    client = LuxriotClient("http://luxriot.test", "user", "pass")
    client.session = FakeSession([response])
    client.CHANNEL_STREAM_SETTLE_SEC = 0.01

    channels = client.get_channels()

    assert [channel["id"] for channel in channels] == [7]
    assert channels[0]["title"] == "Updated"
    assert client.channel_inventory_meta["completion"] == "eof"
    assert client.channel_inventory_meta["payload_count"] == 3


def test_luxriot_client_rejects_explicitly_incomplete_channel_initial_state():
    response = FakeResponse(
        lines=[
            json.dumps(
                {
                    "type": "initial",
                    "hasMore": True,
                    "added": {"channels": [{"id": 7, "title": "Only first fragment"}]},
                }
            )
        ]
    )
    client = LuxriotClient("http://luxriot.test", "user", "pass")
    client.session = FakeSession([response])
    client.CHANNEL_STREAM_SETTLE_SEC = 0.01

    with pytest.raises(RuntimeError, match="before initial state completed"):
        client.get_channels()

    assert client.channel_inventory_meta["complete"] is False
    assert response.closed is True


def test_luxriot_client_redacts_url_credentials_from_request_errors():
    class FailingSession:
        auth = None

        def request(self, *_args, **_kwargs):
            raise requests.ConnectionError(
                "failed http://viewer:sample-password@camera.invalid/live?token=sample-token"
            )

    client = LuxriotClient("http://viewer:sample-password@luxriot.test", "user", "pass")
    client.session = FailingSession()

    with pytest.raises(RuntimeError) as exc_info:
        client._request("GET", "/channels?token=sample-token")

    message = str(exc_info.value)
    assert "sample-password" not in message
    assert "sample-token" not in message
    assert "<redacted>" in message


def test_luxriot_client_exposes_archive_snapshot_and_timeline_methods():
    client = LuxriotClient("http://luxriot.test", "user", "pass")
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "main": {"from": "1000", "to": 9000},
                    "use": {"from": 1000, "to": 9000},
                    "sub": {"from": 0, "to": 0},
                }
            ),
            FakeResponse(payload=[[1000, 2000], ["3000", "4000"], ["bad"]]),
            FakeResponse(text="3000"),
            FakeResponse(content=_jpeg_bytes(), headers={"Content-Type": "image/jpeg"}),
            FakeResponse(),
            FakeResponse(content=b"\x00\x00\x00\x18ftypisom", headers={"Content-Type": "video/mp4"}),
        ]
    )
    client.session = session

    boundaries = client.get_archive_boundaries(7)
    timeline = client.get_archive_timeline(7, 1000, 9000, interval_ms=5000)
    next_frame = client.get_next_archive_frame_time(7, 2500)
    snapshot = client.get_archive_snapshot(7, 3000)
    stream = client.open_live_stream(7, timeout=(1.25, 2.5))

    assert boundaries["main"] == {"from": 1000, "to": 9000}
    assert timeline == [(1000, 2000), (3000, 4000)]
    assert next_frame == 3000
    assert snapshot.size == (32, 24)
    assert stream.headers["Content-Type"] == "video/mp4"
    assert session.requests[0][1] == "http://luxriot.test/archive/7/boundaries"
    assert session.requests[0][2]["params"] == {"streamType": "mainStream"}
    assert session.requests[3][1] == "http://luxriot.test/archive/7/snapshot"
    assert session.requests[3][2]["params"] == {"time": 3000, "type": "video1"}
    assert session.requests[4][1] == "http://luxriot.test/live/7/addStreamToken"
    assert session.requests[4][2]["params"]["stream"] == "mainStream"
    assert session.requests[4][2]["timeout"] == (1.25, 2.5)
    live_token = session.requests[4][2]["params"]["token"]
    assert live_token
    assert session.requests[5][1] == "http://luxriot.test/retrieveLiveStreamByToken"
    assert session.requests[5][2]["params"] == {"token": live_token}
    assert session.requests[5][2]["stream"] is True
    assert getattr(stream, "_eva_live_transport") == "token"


def test_archive_snapshot_uses_documented_video_type_with_legacy_fallback():
    class LegacyArchiveSession:
        def __init__(self):
            self.auth = None
            self.requests = []

        def request(self, method, url, **kwargs):
            self.requests.append((method, url, kwargs))
            if len(self.requests) == 1:
                raise requests.HTTPError("documented archive snapshot selector unsupported")
            return FakeResponse(content=_jpeg_bytes(), headers={"Content-Type": "image/jpeg"})

    client = LuxriotClient("http://luxriot.test", "user", "pass")
    session = LegacyArchiveSession()
    client.session = session

    image = client.get_archive_snapshot(7, 3000, stream_type="subStream")

    assert image.size == (32, 24)
    assert session.requests[0][2]["params"] == {"time": 3000, "type": "video2"}
    assert session.requests[1][2]["params"] == {"time": 3000, "streamType": "subStream"}


def test_luxriot_live_stream_falls_back_to_direct_digest_without_exposing_token():
    class TokenUnsupportedSession:
        def __init__(self):
            self.auth = None
            self.requests = []

        def request(self, method, url, **kwargs):
            self.requests.append((method, url, kwargs))
            if len(self.requests) == 1:
                token = kwargs.get("params", {}).get("token")
                raise requests.HTTPError(f"token endpoint unavailable?token={token}")
            return FakeResponse(
                content=b"\x00\x00\x00\x18ftypisom",
                headers={"Content-Type": "video/mp4"},
            )

    client = LuxriotClient("http://luxriot.test", "user", "pass")
    session = TokenUnsupportedSession()
    client.session = session

    response = client.open_live_stream(7)

    assert session.requests[-1][1] == "http://luxriot.test/live/7/mainStream"
    assert getattr(response, "_eva_live_transport") == "digest_direct_fallback"
    assert getattr(response, "_eva_live_transport_fallback_reason") == "RuntimeError"


class ArchiveSnapshotClient:
    def __init__(self):
        self.frame_times = [1000, 2300, 4100]
        self.snapshot_times = []

    def get_next_archive_frame_time(self, _channel_id, cursor, *, stream_type="mainStream"):
        for frame_time in self.frame_times:
            if frame_time >= cursor:
                return frame_time
        return None

    def get_archive_snapshot(self, _channel_id, time_ms, *, stream_type="mainStream"):
        self.snapshot_times.append((time_ms, stream_type))
        return Image.new("RGB", (20, 10), color=(time_ms % 255, 0, 0))


def test_iter_luxriot_archive_snapshots_aligns_to_recorded_frames():
    client = ArchiveSnapshotClient()

    frames = list(
        iter_luxriot_archive_snapshots(
            client,
            7,
            start_ms=900,
            end_ms=5000,
            interval_ms=1000,
            stream_type="mainStream",
            max_frames=3,
        )
    )

    assert [frame.timestamp_ms for frame in frames] == [1000, 2300, 4100]
    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert frames[0].image.size == (20, 10)
    assert client.snapshot_times == [
        (1000, "mainStream"),
        (2300, "mainStream"),
        (4100, "mainStream"),
    ]


class ArchiveStreamClient:
    def open_archive_stream(self, channel_id, time_ms, *, stream_type="mainStream"):
        assert channel_id == 7
        assert time_ms == 2000
        assert stream_type == "mainStream"
        return FakeResponse(
            content=b"0123456789",
            headers={
                "Content-Type": "video/mp4",
                "X-Stream-Start-Time": "1900",
                "X-Stream-End-Time": "2900",
            },
        )


def test_capture_luxriot_archive_mp4_segment_records_headers_and_bytes(tmp_path):
    target = tmp_path / "segment.mp4"

    segment = capture_luxriot_archive_mp4_segment(
        ArchiveStreamClient(),
        7,
        2000,
        target,
        max_bytes=6,
    )

    assert target.read_bytes() == b"012345"
    assert segment.bytes_written == 6
    assert segment.content_type == "video/mp4"
    assert segment.source_start_ms == 1900
    assert segment.source_end_ms == 2900
