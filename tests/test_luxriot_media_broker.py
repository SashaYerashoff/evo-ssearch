from __future__ import annotations

import base64

import requests
import pytest

import oldapp


MP4_BYTES = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"


class FakeUpstream:
    def __init__(
        self,
        chunks=None,
        *,
        content=b"",
        headers=None,
        status_code=200,
        text="",
    ):
        self.chunks = list(chunks if chunks is not None else ([content] if content else []))
        self.content = content
        self.headers = dict(headers or {})
        self.status_code = status_code
        self.text = text
        self.closed = False

    def iter_content(self, chunk_size=None):
        assert chunk_size == oldapp._LUXRIOT_MEDIA_CHUNK_BYTES
        yield from self.chunks

    def close(self):
        self.closed = True


class FakeLuxriotClient:
    def __init__(self, response=None, error=None):
        self.responses = list(response) if isinstance(response, (list, tuple)) else [response]
        self.error = error
        self.calls = []

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError(f"unexpected upstream request: {method} {path}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def open_live_stream(self, channel_id, *, stream="mainStream", headers=None, timeout=None):
        self.calls.append(
            (
                "OPEN_LIVE_TOKEN",
                f"/live/{int(channel_id)}/{stream}",
                {"headers": dict(headers or {}), "timeout": timeout, "stream": True},
            )
        )
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError(f"unexpected live upstream request for channel {channel_id}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        setattr(response, "_eva_live_transport", "token")
        return response


@pytest.fixture()
def app_client(monkeypatch):
    monkeypatch.setattr(oldapp.config, "AUTH_ENABLED", False)
    oldapp.app.config.update(TESTING=True)
    with oldapp.app.test_client() as client:
        yield client


def _install_client(monkeypatch, client):
    monkeypatch.setattr(oldapp.luxriot_manager, "build_client", lambda: client)


def test_media_limits_are_backed_by_deployment_config(monkeypatch):
    monkeypatch.setattr(oldapp.config, "LUXRIOT_MEDIA_CONNECT_TIMEOUT_SEC", 1.25)
    monkeypatch.setattr(oldapp.config, "LUXRIOT_MEDIA_READ_TIMEOUT_SEC", 2.5)
    monkeypatch.setattr(oldapp.config, "LUXRIOT_LIVE_MEDIA_MAX_SECONDS", 12.0)
    monkeypatch.setattr(oldapp.config, "LUXRIOT_LIVE_MEDIA_MAX_BYTES", 4096)
    monkeypatch.setattr(oldapp.config, "LUXRIOT_ARCHIVE_MEDIA_MAX_SECONDS", 24.0)
    monkeypatch.setattr(oldapp.config, "LUXRIOT_ARCHIVE_MEDIA_MAX_BYTES", 8192)

    assert oldapp._luxriot_media_limits("live") == ((1.25, 2.5), 12.0, 4096)
    assert oldapp._luxriot_media_limits("archive") == ((1.25, 2.5), 24.0, 8192)


@pytest.mark.parametrize(
    ("lease_seconds", "renew_after_ms"),
    [(1.0, 750), (12.0, 9000), (30.0, 22500)],
)
def test_live_media_renewal_precedes_bounded_response_cutoff(
    lease_seconds,
    renew_after_ms,
):
    assert oldapp._luxriot_media_renew_after_ms(lease_seconds) == renew_after_ms


def test_runtime_javascript_does_not_embed_luxriot_credentials(app_client, monkeypatch):
    monkeypatch.setattr(
        oldapp.config,
        "LUXRIOT_BASE_URL",
        "http://evo-user:evo-secret@recorder.invalid:8080",
    )

    response = app_client.get("/js/app.js")

    assert response.status_code == 200
    assert b"evo-user" not in response.data
    assert b"evo-secret" not in response.data
    assert b"recorder.invalid" not in response.data


def test_live_media_broker_sniffs_octet_stream_mp4_without_exposing_credentials(
    app_client,
    monkeypatch,
):
    upstream = FakeUpstream(
        [MP4_BYTES, b"media-tail"],
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(MP4_BYTES) + len(b"media-tail")),
            "X-Private-Token": "must-not-leak",
        },
    )
    fake = FakeLuxriotClient(upstream)
    _install_client(monkeypatch, fake)

    response = app_client.get("/luxriot/media/live/7?stream=mainStream")

    assert response.status_code == 200
    assert response.data == MP4_BYTES + b"media-tail"
    assert response.headers["Content-Type"] == "video/mp4"
    assert response.headers["X-EVA-Media-Kind"] == "video"
    assert response.headers["X-EVA-Live-Transport"] == "token"
    assert response.headers["Cache-Control"] == "no-store, private, max-age=0"
    assert "X-Private-Token" not in response.headers
    assert "password" not in response.get_data(as_text=True).lower()
    assert upstream.closed is True
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("OPEN_LIVE_TOKEN", "/live/7/mainStream")
    assert kwargs["stream"] is True
    assert kwargs["headers"]["Accept-Encoding"] == "identity"
    assert "Authorization" not in kwargs["headers"]
    assert isinstance(kwargs["timeout"], tuple)


def test_live_media_broker_exposes_lease_and_does_not_forward_unsafe_length(
    app_client,
    monkeypatch,
):
    monkeypatch.setattr(oldapp.config, "LUXRIOT_LIVE_MEDIA_MAX_SECONDS", 12.0)
    upstream = FakeUpstream(
        [MP4_BYTES, b"tail"],
        headers={
            "Content-Type": "video/mp4",
            # The time-bounded broker cannot promise that this upstream length
            # will arrive before it closes the response.
            "Content-Length": str(4 * 1024 * 1024),
        },
    )
    _install_client(monkeypatch, FakeLuxriotClient(upstream))

    response = app_client.get("/luxriot/media/live/7?stream=mainStream")

    assert response.status_code == 200
    assert response.headers["X-EVA-Media-Lease-Seconds"] == "12"
    assert response.headers["X-EVA-Media-Renew-After-Ms"] == "9000"
    assert "Content-Length" not in response.headers
    assert response.data == MP4_BYTES + b"tail"


def test_live_media_broker_cuts_at_lease_and_closes_upstream(
    app_client,
    monkeypatch,
):
    monkeypatch.setattr(oldapp.config, "LUXRIOT_LIVE_MEDIA_MAX_SECONDS", 12.0)
    ticks = iter((100.0, 101.0, 112.1))
    monkeypatch.setattr(oldapp.time, "monotonic", lambda: next(ticks))
    upstream = FakeUpstream(
        [MP4_BYTES, b"inside-lease", b"after-lease"],
        headers={"Content-Type": "video/mp4"},
    )
    _install_client(monkeypatch, FakeLuxriotClient(upstream))

    response = app_client.get("/luxriot/media/live/7?stream=mainStream")

    assert response.data == MP4_BYTES + b"inside-lease"
    assert upstream.closed is True


def test_attention_stream_reuses_selected_apex_without_opening_evo(
    app_client,
    monkeypatch,
):
    jpeg = b"\xff\xd8\xffmodel-visible-apex\xff\xd9"
    frame = {
        "thumbnail": base64.b64encode(jpeg).decode("ascii"),
        "captured_at": 1700000000.125,
        "frame_hash": "apex-hash",
        "capture_selection": {"selected_frame_hash": "apex-hash"},
    }
    monkeypatch.setattr(oldapp, "_luxriot_recent_frame_item", lambda *_args, **_kwargs: dict(frame))
    monkeypatch.setattr(
        oldapp.luxriot_manager,
        "build_client",
        lambda: (_ for _ in ()).throw(AssertionError("attention preview must not open Evo")),
    )

    head = app_client.head("/luxriot/attention_stream/7")
    response = app_client.get("/luxriot/attention_stream/7", buffered=False)
    first_part = next(response.response)
    response.close()

    assert head.status_code == 200
    assert head.headers["X-EVA-Attention-Preview"] == "1"
    assert head.headers["X-EVA-Media-Renew-After-Ms"] == "90000"
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
    assert b"X-EVA-Frame-Timestamp-Ms: 1700000000125" in first_part
    assert jpeg in first_part


def test_attention_stream_reemits_static_apex_as_keepalive(
    app_client,
    monkeypatch,
):
    jpeg = b"\xff\xd8\xffstatic-model-visible-apex\xff\xd9"
    frame = {
        "thumbnail": base64.b64encode(jpeg).decode("ascii"),
        "captured_at": 1700000000.125,
        "frame_hash": "static-apex-hash",
    }
    clock = [100.0]

    monkeypatch.setattr(oldapp, "_luxriot_recent_frame_item", lambda *_args, **_kwargs: dict(frame))
    monkeypatch.setattr(oldapp, "_luxriot_media_limits", lambda _kind: ((1, 1), 12.0, 1))
    monkeypatch.setattr(oldapp.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(oldapp.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    response = app_client.get("/luxriot/attention_stream/7", buffered=False)
    iterator = iter(response.response)
    started_at = clock[0]
    parts = [next(iterator), next(iterator)]
    elapsed = clock[0] - started_at
    response.close()

    assert response.status_code == 200
    assert elapsed <= 11.0
    assert parts[0] == parts[1]
    assert all(jpeg in part for part in parts)


def test_media_negotiation_recovers_mjpeg_boundary_from_octet_stream():
    first_chunk = (
        b"--eva-boundary\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        b"\xff\xd8\xffjpeg\r\n"
    )

    kind, content_type, error = oldapp._luxriot_media_negotiation(
        "application/octet-stream",
        first_chunk,
        range_header=None,
    )

    assert kind == "mjpeg"
    assert content_type == "multipart/x-mixed-replace; boundary=eva-boundary"
    assert error == ""


def test_media_negotiation_rejects_jpeg_bytes_mislabeled_as_video():
    kind, content_type, error = oldapp._luxriot_media_negotiation(
        "video/mp4",
        b"\xff\xd8\xffjpeg",
        range_header=None,
    )

    assert kind is None
    assert content_type is None
    assert error == "snapshot_only"


def test_archive_media_broker_forwards_single_range_and_response_range_headers(
    app_client,
    monkeypatch,
):
    frame_time = FakeUpstream(text="1700000000000")
    upstream = FakeUpstream(
        [MP4_BYTES],
        status_code=206,
        headers={
            "Content-Type": "video/mp4",
            "Content-Range": "bytes 0-23/100",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(MP4_BYTES)),
            "X-Stream-Start-Time": "1700000000000",
            "X-Stream-End-Time": "1700000001000",
            "X-Stream-Last-Sample-Timestamp": "1700000000900",
        },
    )
    fake = FakeLuxriotClient([frame_time, upstream])
    _install_client(monkeypatch, fake)

    response = app_client.get(
        "/luxriot/media/archive/9?time_ms=1700000000000&stream=subStream&duration_sec=15",
        headers={"Range": "bytes=0-23"},
    )

    assert response.status_code == 206
    assert response.headers["Content-Range"] == "bytes 0-23/100"
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.headers["X-Stream-Start-Time"] == "1700000000000"
    assert response.headers["X-Stream-Last-Sample-Timestamp"] == "1700000000900"
    assert response.headers["X-EVA-Archive-Resolved-Time-Ms"] == "1700000000000"
    assert response.headers["X-EVA-Archive-Duration-Seconds"] == "15"
    assert response.headers["X-EVA-Archive-Frame-Alignment"] == "next_frame_time"
    assert response.headers["X-EVA-HTML5-Compatible"] == "requested"
    frame_method, frame_path, frame_kwargs = fake.calls[0]
    assert (frame_method, frame_path) == ("GET", "/archive/9/nextFrameTime")
    assert frame_kwargs["params"] == {
        "time": 1700000000000,
        "streamType": "subStream",
    }
    method, path, kwargs = fake.calls[1]
    assert (method, path) == ("GET", "/archive/9/stream")
    assert kwargs["params"] == {
        "time": 1700000000000,
        "streamType": "subStream",
        "duration": 15,
        "html5compatible": "true",
    }
    assert kwargs["headers"]["Range"] == "bytes=0-23"
    assert kwargs["headers"]["Streaming-Web-Ver"] == "1.3.0"
    assert kwargs["timeout"][1] >= 23.0


def test_archive_media_broker_falls_back_when_html5_query_is_rejected(
    app_client,
    monkeypatch,
):
    rejected = requests.HTTPError("unsupported query")
    rejected.response = requests.Response()
    rejected.response.status_code = 400
    media = FakeUpstream([MP4_BYTES], headers={"Content-Type": "video/mp4"})
    fake = FakeLuxriotClient([
        FakeUpstream(text="1700000000200"),
        rejected,
        media,
    ])
    _install_client(monkeypatch, fake)

    response = app_client.head(
        "/luxriot/media/archive/9?time_ms=1700000000000&stream=mainStream"
    )

    assert response.status_code == 200
    assert response.headers["X-EVA-HTML5-Compatible"] == "unsupported_fallback"
    assert response.headers["X-EVA-Archive-Resolved-Time-Ms"] == "1700000000200"
    assert fake.calls[1][2]["params"]["html5compatible"] == "true"
    assert fake.calls[2][2]["params"] == {
        "time": 1700000000200,
        "streamType": "mainStream",
    }


def test_archive_media_broker_reports_gap_when_next_frame_time_is_empty(
    app_client,
    monkeypatch,
):
    fake = FakeLuxriotClient(FakeUpstream(text="0"))
    _install_client(monkeypatch, fake)

    response = app_client.get(
        "/luxriot/media/archive/9?time_ms=1700000000000&stream=mainStream"
    )

    assert response.status_code == 409
    assert response.get_json()["error_code"] == "archive_gap"
    assert len(fake.calls) == 1


def test_archive_media_broker_rejects_duration_beyond_bounded_proxy_limit(
    app_client,
    monkeypatch,
):
    monkeypatch.setattr(
        oldapp.luxriot_manager,
        "build_client",
        lambda: (_ for _ in ()).throw(AssertionError("upstream must not be called")),
    )

    response = app_client.get(
        "/luxriot/media/archive/9?time_ms=1700000000000&stream=mainStream&duration_sec=999"
    )

    assert response.status_code == 400
    assert response.get_json()["error_code"] == "invalid_archive_duration"


def test_media_broker_rejects_malformed_or_multiple_ranges_before_upstream(
    app_client,
    monkeypatch,
):
    monkeypatch.setattr(
        oldapp.luxriot_manager,
        "build_client",
        lambda: (_ for _ in ()).throw(AssertionError("upstream must not be called")),
    )

    response = app_client.get(
        "/luxriot/media/live/7",
        headers={"Range": "bytes=0-10,20-30"},
    )

    assert response.status_code == 416
    assert response.get_json()["error_code"] == "invalid_range"
    assert response.headers["Content-Range"] == "bytes */*"
    assert response.headers["Cache-Control"] == "no-store, private, max-age=0"


def test_media_broker_preserves_upstream_range_not_satisfiable_as_416(
    app_client,
    monkeypatch,
):
    rejected = requests.HTTPError("range rejected")
    rejected.response = requests.Response()
    rejected.response.status_code = 416
    fake = FakeLuxriotClient(error=rejected)
    _install_client(monkeypatch, fake)

    response = app_client.get(
        "/luxriot/media/live/7",
        headers={"Range": "bytes=999999-"},
    )

    assert response.status_code == 416
    assert response.get_json()["error_code"] == "range_not_satisfiable"
    assert response.headers["Content-Range"] == "bytes */*"
    assert fake.calls[0][2]["headers"]["Range"] == "bytes=999999-"


def test_media_broker_does_not_describe_a_snapshot_response_as_video(
    app_client,
    monkeypatch,
):
    upstream = FakeUpstream(
        [b"\xff\xd8\xffjpeg"],
        headers={"Content-Type": "image/jpeg"},
    )
    fake = FakeLuxriotClient(upstream)
    _install_client(monkeypatch, fake)

    response = app_client.head("/luxriot/media/live/7?stream=mainStream")

    assert response.status_code == 415
    assert response.data == b""
    assert response.headers["X-EVA-Media-Error"] == "snapshot_only"
    assert response.headers["X-EVA-Media-Fallback"].startswith("/luxriot/snapshot/7?")
    assert upstream.closed is True


def test_media_broker_returns_generic_timeout_without_echoing_upstream_secret(
    app_client,
    monkeypatch,
):
    try:
        raise requests.ReadTimeout("http://evo-user:evo-secret@recorder.invalid/live")
    except requests.ReadTimeout as cause:
        wrapped = RuntimeError("upstream request failed")
        wrapped.__cause__ = cause
    fake = FakeLuxriotClient(error=wrapped)
    _install_client(monkeypatch, fake)

    response = app_client.get("/luxriot/media/live/7")

    assert response.status_code == 504
    assert response.get_json()["error_code"] == "media_timeout"
    rendered = response.get_data(as_text=True)
    assert "evo-user" not in rendered
    assert "evo-secret" not in rendered
    assert "recorder.invalid" not in rendered


def test_archive_snapshot_fallback_is_explicitly_degraded_and_not_cacheable(
    app_client,
    monkeypatch,
):
    jpeg = b"\xff\xd8\xffarchived-jpeg"
    upstream = FakeUpstream(
        content=jpeg,
        headers={"Content-Type": "image/jpeg"},
    )
    fake = FakeLuxriotClient(upstream)
    _install_client(monkeypatch, fake)

    response = app_client.get(
        "/luxriot/archive_snapshot/11?time_ms=1700000000123&stream=mainStream"
    )

    assert response.status_code == 200
    assert response.data == jpeg
    assert response.headers["Content-Type"] == "image/jpeg"
    assert response.headers["X-EVA-Media-State"] == "degraded"
    assert response.headers["X-EVA-Media-Kind"] == "static_frame"
    assert response.headers["X-EVA-Archive-Time-Ms"] == "1700000000123"
    assert response.headers["Cache-Control"] == "no-store, private, max-age=0"
    method, path, kwargs = fake.calls[0]
    assert (method, path) == ("GET", "/archive/11/snapshot")
    assert kwargs["params"] == {
        "time": 1700000000123,
        "type": "video1",
    }
