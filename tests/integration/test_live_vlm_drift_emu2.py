"""Live emu-2 VLM drift alert smoke (OPT-IN).

This test verifies the real integration path:
  Luxriot emu-2 loop -> video-summary capture -> VLM alert JSON -> archive
  `vlm_alert` evidence record with an alert-anchor frame.

It is intentionally live-gated and slow. Run only on the dev/demo machine that
has the 10-second drift loop on emu-2/channel 120.

    EVA_LIVE_BASE_URL=https://127.0.0.1:5443 \
    EVA_LIVE_USER=admin EVA_LIVE_PASSWORD='Luxriot2026!?' \
    EVA_LIVE_INCLUDE=drift \
    .venv/bin/pytest -q tests/integration/test_live_vlm_drift_emu2.py -s
"""
from __future__ import annotations

import os
import re
import time
import unittest
from typing import Any, Dict, Iterable, List, Mapping, Optional

from tests.integration.eva_client import EvaSession


_BASE = os.getenv("EVA_LIVE_BASE_URL", "").strip()
_USER = os.getenv("EVA_LIVE_USER", "").strip()
_PASSWORD = os.getenv("EVA_LIVE_PASSWORD", "")
_CSRF = os.getenv("EVA_LIVE_CSRF_COOKIE", "eva_csrf").strip() or "eva_csrf"
_VERIFY_TLS = os.getenv("EVA_LIVE_VERIFY_TLS", "").strip().lower() in {"1", "true", "yes", "on"}
_INCLUDE = {t.strip() for t in os.getenv("EVA_LIVE_INCLUDE", "").split(",") if t.strip()}

_CHANNEL_ID = int(os.getenv("EVA_LIVE_DRIFT_CHANNEL_ID", "120") or "120")
_WAIT_SEC = float(os.getenv("EVA_LIVE_DRIFT_WAIT_SEC", "600") or "600")
_POLL_SEC = float(os.getenv("EVA_LIVE_DRIFT_POLL_SEC", "20") or "20")

_DRIFT_RE = re.compile(
    r"\b(?:drift(?:ing)?|donut|burnout|skid(?:ding)?|sideways|spin(?:ning)?|"
    r"tire smoke|tyre smoke|smoke emission|loss of traction)\b",
    flags=re.IGNORECASE,
)
_VEHICLE_RE = re.compile(r"\b(?:vehicle|car|sedan|bmw|auto)\b", flags=re.IGNORECASE)

_DRIFT_ALERT_POLICY = """Traffic drift/burnout watch for this channel:
- Emit an alert when a visible vehicle drifts, slides sideways, performs a donut or burnout, loses traction, creates tire smoke, or makes an aggressive high-angle maneuver inconsistent with normal lane following.
- Treat a BMW, sedan, or dark/silver car sliding in the scene as alert-worthy even if the generated test loop contains minor visual artifacts.
- Use severity high for active drifting/burnout/tire smoke; normal for ambiguous skidding.
- In each alert description include the clearest snapshot number or range, preferring the apex moment where the vehicle is most sideways or tire smoke is most visible.
"""


def _json(resp: Any) -> Dict[str, Any]:
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise AssertionError(f"Expected JSON object, got {type(data).__name__}")
    return data


def _csrf_headers(session: EvaSession) -> Dict[str, str]:
    token = session.http.cookies.get(_CSRF) or ""
    return {"X-CSRF-Token": token} if token else {}


def _drift_text(value: Any) -> str:
    if isinstance(value, Mapping):
        parts: List[str] = []
        for key in ("title", "description", "summary", "probe_name", "severity", "state"):
            if value.get(key):
                parts.append(str(value.get(key)))
        alert_event = value.get("alert_event")
        if isinstance(alert_event, Mapping):
            parts.append(_drift_text(alert_event))
        payload = value.get("payload")
        if isinstance(payload, Mapping):
            parts.append(_drift_text(payload))
        return " ".join(parts)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_drift_text(item) for item in value)
    return str(value or "")


def _is_drift_signal(value: Any) -> bool:
    text = _drift_text(value)
    return bool(_DRIFT_RE.search(text) and _VEHICLE_RE.search(text))


def _latest_drift_log(logs: Iterable[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for log in reversed([item for item in logs if isinstance(item, Mapping)]):
        alert_events = log.get("alert_events")
        if isinstance(alert_events, list) and any(_is_drift_signal(event) for event in alert_events):
            return log
        if _is_drift_signal(log.get("summary")):
            return log
    return None


def _latest_drift_detection(detections: Iterable[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for detection in detections:
        if _is_drift_signal(detection):
            return detection
    return None


@unittest.skipUnless(_BASE and "drift" in _INCLUDE, "set EVA_LIVE_BASE_URL and EVA_LIVE_INCLUDE=drift")
class LiveVlmDriftEmu2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = EvaSession(_BASE, csrf_cookie=_CSRF, verify_tls=_VERIFY_TLS, timeout=180.0)
        if _USER:
            cls.session.login(_USER, _PASSWORD)

    def _get(self, path: str, **params: Any) -> Dict[str, Any]:
        resp = self.session.http.get(f"{_BASE.rstrip('/')}{path}", params=params, timeout=180.0)
        return _json(resp)

    def _post(self, path: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        resp = self.session.http.post(
            f"{_BASE.rstrip('/')}{path}",
            json=dict(body),
            headers=_csrf_headers(self.session),
            timeout=180.0,
        )
        return _json(resp)

    def test_emu2_drift_loop_generates_vlm_alert_with_apex_anchor(self) -> None:
        started_ms = int((time.time() - 5.0) * 1000)

        self._post(
            "/luxriot/prompt_settings",
            {
                "channel_id": _CHANNEL_ID,
                "alert_policy_prompt": _DRIFT_ALERT_POLICY,
                "bookmark_enabled": True,
                "bookmark_cooldown_sec": 60,
            },
        )
        self._post(
            "/luxriot/start_capture",
            {
                "channel_id": _CHANNEL_ID,
                "batch_size": 12,
                "snapshot_interval_sec": 1,
            },
        )

        deadline = time.monotonic() + max(30.0, _WAIT_SEC)
        last_status: Dict[str, Any] = {}
        last_logs: List[Mapping[str, Any]] = []
        last_detections: List[Mapping[str, Any]] = []
        drift_log: Optional[Mapping[str, Any]] = None
        drift_detection: Optional[Mapping[str, Any]] = None

        while time.monotonic() < deadline:
            last_status = self._get("/luxriot/streams")
            session_status = self._get("/luxriot/session", channel_id=_CHANNEL_ID, limit=6)
            logs_raw = session_status.get("logs") or []
            last_logs = [item for item in logs_raw if isinstance(item, Mapping)]
            drift_log = _latest_drift_log(last_logs)

            detections_payload = self._get(
                "/detections/list",
                channel_id=_CHANNEL_ID,
                source="vlm_alert",
                since_ms=started_ms,
                limit=20,
            )
            det_raw = detections_payload.get("detections") or []
            last_detections = [item for item in det_raw if isinstance(item, Mapping)]
            drift_detection = _latest_drift_detection(last_detections)
            if drift_log and drift_detection:
                payload = drift_detection.get("payload") if isinstance(drift_detection, Mapping) else {}
                if isinstance(payload, Mapping) and payload.get("anchor_role") == "alert_anchor":
                    frame_index = int(payload.get("anchor_frame_index") or 0)
                    if frame_index >= 5 and (
                        payload.get("anchor_selection") in {"alert_snapshot_reference", "summary_snapshot_reference"}
                        or payload.get("anchor_snapshot_hint")
                    ):
                        break
            time.sleep(max(1.0, _POLL_SEC))

        self.assertIsNotNone(drift_log, f"No drift VLM log found. Last logs: {last_logs!r}")
        self.assertGreaterEqual(int(drift_log.get("json_alert_count") or 0), 1, drift_log)
        self.assertGreaterEqual(int(drift_log.get("parser_alert_count") or 0), 1, drift_log)
        self.assertTrue(any(_is_drift_signal(event) for event in (drift_log.get("alert_events") or [])), drift_log)

        self.assertIsNotNone(
            drift_detection,
            f"No drift vlm_alert archive row found. Last detections: {last_detections!r}; streams: {last_status!r}",
        )
        payload = drift_detection.get("payload")
        self.assertIsInstance(payload, dict, drift_detection)
        assert isinstance(payload, dict)
        self.assertEqual(payload.get("source"), "vlm_alert", payload)
        self.assertEqual(payload.get("anchor_role"), "alert_anchor", payload)
        self.assertIn(payload.get("anchor_selection"), {"alert_snapshot_reference", "summary_snapshot_reference"}, payload)
        self.assertGreaterEqual(int(payload.get("anchor_frame_index") or 0), 5, payload)
        self.assertTrue(drift_detection.get("thumbnail"), "vlm_alert archive row must carry thumbnail evidence")

        detection_id = drift_detection.get("id")
        self.assertIsNotNone(detection_id, drift_detection)
        thumb_resp = self.session.http.get(
            f"{_BASE.rstrip('/')}/detections/thumbnail/{int(detection_id)}",
            timeout=60.0,
        )
        self.assertEqual(thumb_resp.status_code, 200, thumb_resp.text[:200])
        self.assertIn(thumb_resp.headers.get("Content-Type", "").split(";")[0], {"image/jpeg", "image/png"})


if __name__ == "__main__":
    unittest.main()
