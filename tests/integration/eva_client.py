"""Thin client for live integration testing of the EVA agent over its real HTTP/SSE contract.

Grounded on the actual endpoints (oldapp.py):
  - POST /auth/login            {username, password} -> sets session + CSRF cookies
  - POST /agent/chat            {message, session_id?, image_b64?} -> SSE text/event-stream
  - POST /agent/action-plans/<plan_id>/execute  {session_id?} -> JSON {success, result}  (the UI "Apply")

SSE event types yielded by the agent include tool_call, tool_result,
tool_progress, tool_budget, context_budget, context_metrics, text, session,
done, and error.

This is an *acceptance smoke* client: assert structure (tool calls/results,
safe_to_apply, restricted_matches, delivery_status, action receipts), not LLM prose.
No new dependency: uses `requests` (already required) + stdlib.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


class SseDeadlineExceeded(TimeoutError):
    """Wall-clock deadline hit while heartbeats kept an SSE socket alive."""

    def __init__(self, deadline_sec: float, events: List[Dict[str, Any]]) -> None:
        self.deadline_sec = float(deadline_sec)
        self.events = [dict(event) for event in events]
        super().__init__(f"SSE wall-clock deadline exceeded after {deadline_sec:.3f}s")


def parse_sse_events(
    lines: Any,
    *,
    elapsed_fn: Optional[Any] = None,
    deadline_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Parse an SSE byte/line stream into a list of decoded `data:` JSON events.

    Tolerant: ignores comment/heartbeat lines and non-JSON data payloads.
    Supports both our normal one-line JSON events and standards-compliant
    multi-line `data:` frames.
    """
    events: List[Dict[str, Any]] = []
    data_parts: List[str] = []

    def flush() -> None:
        if not data_parts:
            return
        parts = [part for part in data_parts]
        data_parts.clear()
        payloads = ["\n".join(parts).strip()]
        if len(parts) > 1:
            payloads.extend(part.strip() for part in parts)
        for payload in payloads:
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if callable(elapsed_fn):
                    try:
                        obj["_received_at_sec"] = round(
                            max(0.0, float(elapsed_fn())),
                            6,
                        )
                    except Exception:
                        pass
                events.append(obj)
                if payload == payloads[0]:
                    return

    def enforce_deadline() -> None:
        if deadline_sec is None or not callable(elapsed_fn):
            return
        try:
            elapsed = max(0.0, float(elapsed_fn()))
        except Exception:
            return
        if elapsed < float(deadline_sec):
            return
        flush()
        raise SseDeadlineExceeded(float(deadline_sec), events)

    for raw in lines:
        enforce_deadline()
        if raw is None:
            continue
        line = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        line = line.rstrip("\r\n")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data_parts.append(line[len("data:"):].lstrip())
    enforce_deadline()
    flush()
    return events


@dataclass
class Transcript:
    """Structured view over one agent turn's SSE events (assert on this, not prose)."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    telemetry_samples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def tool_calls(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            (str(e.get("name") or ""), dict(e.get("args") or {}))
            for e in self.events
            if e.get("type") == "tool_call"
        ]

    @property
    def tool_results(self) -> List[Tuple[str, Any, Optional[str]]]:
        out: List[Tuple[str, Any, Optional[str]]] = []
        for e in self.events:
            if e.get("type") == "tool_result":
                out.append((str(e.get("name") or ""), e.get("result"), e.get("error")))
        return out

    @property
    def text(self) -> str:
        return "".join(str(e.get("content") or "") for e in self.events if e.get("type") == "text")

    @property
    def session_id(self) -> Optional[str]:
        for e in self.events:
            if e.get("type") == "session" and e.get("session_id"):
                return str(e.get("session_id"))
            if e.get("type") == "done" and e.get("session_id"):
                return str(e.get("session_id"))
        return None

    @property
    def errored(self) -> bool:
        return any(e.get("type") == "error" for e in self.events)

    @property
    def finished(self) -> bool:
        return any(e.get("type") == "done" for e in self.events)

    @property
    def tool_call_count(self) -> int:
        return sum(1 for event in self.events if event.get("type") == "tool_call")

    @property
    def ui_effects(self) -> List[Dict[str, Any]]:
        effects: List[Dict[str, Any]] = []
        for event in self.events:
            if event.get("type") != "tool_result":
                continue
            for effect in event.get("ui_effects") or []:
                if isinstance(effect, dict):
                    effects.append(dict(effect))
        return effects

    @property
    def budget_stops(self) -> List[Dict[str, Any]]:
        stops: List[Dict[str, Any]] = []
        for event in self.events:
            if event.get("type") == "tool_budget":
                stops.append(dict(event))
            elif event.get("type") == "context_budget" and event.get("status") == "hard_stop":
                stops.append(dict(event))
        return stops

    @property
    def context_metrics(self) -> List[Dict[str, Any]]:
        return [
            dict(event)
            for event in self.events
            if event.get("type") == "context_metrics"
        ]

    @property
    def tool_trace(self) -> List[Dict[str, Any]]:
        results_by_call_id = {
            str(event.get("call_id")): event.get("result")
            for event in self.events
            if event.get("type") == "tool_result" and event.get("call_id")
        }
        trace: List[Dict[str, Any]] = []
        for event in self.events:
            if event.get("type") != "tool_call":
                continue
            call_id = str(event.get("call_id") or "")
            result = results_by_call_id.get(call_id)
            try:
                result_chars = len(json.dumps(result, ensure_ascii=False, default=str))
            except Exception:
                result_chars = len(str(result or ""))
            trace.append({
                "name": str(event.get("name") or ""),
                "args": dict(event.get("args") or {}),
                "result_chars": result_chars,
            })
        return trace

    @property
    def tool_timings(self) -> List[Dict[str, Any]]:
        """Pair streamed calls/results and expose client-observed wall time.

        This is deliberately transport-level timing. It includes gateway and
        tool execution time, but excludes the model's decision before the call.
        """

        result_events_by_id = {
            str(event.get("call_id")): event
            for event in self.events
            if event.get("type") == "tool_result" and event.get("call_id")
        }
        result_events_without_id: Dict[str, List[Dict[str, Any]]] = {}
        for event in self.events:
            if event.get("type") != "tool_result" or event.get("call_id"):
                continue
            result_events_without_id.setdefault(
                str(event.get("name") or ""),
                [],
            ).append(event)

        rows: List[Dict[str, Any]] = []
        used_without_id: Counter[str] = Counter()
        for event in self.events:
            if event.get("type") != "tool_call":
                continue
            call_id = str(event.get("call_id") or "")
            name = str(event.get("name") or "")
            result_event = result_events_by_id.get(call_id) if call_id else None
            if result_event is None and not call_id:
                candidates = result_events_without_id.get(name) or []
                index = used_without_id[name]
                if index < len(candidates):
                    result_event = candidates[index]
                    used_without_id[name] += 1
            started = _finite_number(event.get("_received_at_sec"))
            completed = _finite_number(
                result_event.get("_received_at_sec")
                if isinstance(result_event, dict)
                else None
            )
            row: Dict[str, Any] = {
                "call_id": call_id or None,
                "name": name,
                "started_at_sec": started,
                "completed_at_sec": completed,
                "error": bool(result_event and result_event.get("error")),
            }
            if started is not None and completed is not None:
                row["duration_ms"] = round(
                    max(0.0, completed - started) * 1000.0,
                    3,
                )
            rows.append(row)
        return rows

    @property
    def exact_duplicate_tool_calls(self) -> List[Dict[str, Any]]:
        seen: Counter[str] = Counter()
        duplicates: List[Dict[str, Any]] = []
        for name, args in self.tool_calls:
            try:
                key = json.dumps(
                    [name, args],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            except Exception:
                key = f"{name}:{args!r}"
            seen[key] += 1
            if seen[key] > 1:
                duplicates.append({"name": name, "args": dict(args)})
        return duplicates

    @property
    def admission_metrics(self) -> Dict[str, Any]:
        """Summarize the LM admission queue sampled during this turn."""

        samples: List[Dict[str, Any]] = []
        errors = 0
        for sample in self.telemetry_samples:
            if sample.get("path") != "/lm/admission":
                continue
            payload = sample.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("resources"), list):
                samples.append(sample)
            else:
                errors += 1

        # Heartbeats carry the exact agent resource even when the authenticated
        # polling endpoint is unavailable. Fold them into the same format.
        for event in self.events:
            admission = event.get("lm_admission")
            if event.get("type") != "heartbeat" or not isinstance(admission, dict):
                continue
            samples.append({
                "path": "/lm/admission",
                "at_sec": event.get("_received_at_sec"),
                "payload": {"resources": [dict(admission)]},
                "source": "sse_heartbeat",
            })

        resources: Dict[str, Dict[str, Any]] = {}
        for sample in samples:
            payload = sample.get("payload") or {}
            for raw in payload.get("resources") or []:
                if not isinstance(raw, dict):
                    continue
                resource = str(raw.get("resource") or "unknown")
                row = resources.setdefault(resource, {
                    "resource": resource,
                    "sample_count": 0,
                    "max_active": 0,
                    "max_queued": 0,
                    "max_oldest_queue_age_sec": 0.0,
                    "first_counters": None,
                    "last_counters": None,
                    "first_average_wait_ms": 0.0,
                    "last_average_wait_ms": 0.0,
                })
                row["sample_count"] += 1
                row["max_active"] = max(
                    int(row["max_active"]),
                    int(_finite_number(raw.get("active")) or 0),
                )
                row["max_queued"] = max(
                    int(row["max_queued"]),
                    int(_finite_number(raw.get("queued")) or 0),
                )
                row["max_oldest_queue_age_sec"] = round(max(
                    float(row["max_oldest_queue_age_sec"]),
                    float(_finite_number(raw.get("oldest_queue_age_sec")) or 0.0),
                ), 3)
                counters = raw.get("counters")
                if isinstance(counters, dict):
                    normalized = {
                        str(key): int(value)
                        for key, value in counters.items()
                        if _finite_number(value) is not None
                    }
                    if row["first_counters"] is None:
                        row["first_counters"] = normalized
                        row["first_average_wait_ms"] = float(
                            _finite_number(raw.get("average_wait_ms")) or 0.0
                        )
                    row["last_counters"] = normalized
                    row["last_average_wait_ms"] = float(
                        _finite_number(raw.get("average_wait_ms")) or 0.0
                    )

        totals = {
            "agent_admissions": 0,
            "agent_queued": 0,
            "agent_completed": 0,
            "agent_failed": 0,
            "agent_wait_ms_estimate": 0.0,
            "max_active": 0,
            "max_queued": 0,
            "max_oldest_queue_age_sec": 0.0,
        }
        compact_resources: List[Dict[str, Any]] = []
        for row in resources.values():
            first = row.pop("first_counters") or {}
            last = row.pop("last_counters") or {}
            deltas = {
                key: max(0, int(last.get(key, 0)) - int(first.get(key, 0)))
                for key in set(first) | set(last)
            }
            admitted = deltas.get("admitted_agent", 0)
            first_admitted = int(first.get("admitted_total", 0))
            last_admitted = int(last.get("admitted_total", 0))
            first_wait_total = float(row.pop("first_average_wait_ms")) * first_admitted
            last_wait_total = float(row.pop("last_average_wait_ms")) * last_admitted
            wait_delta = max(0.0, last_wait_total - first_wait_total)
            agent_wait_estimate = (
                wait_delta * admitted / max(1, deltas.get("admitted_total", 0))
                if admitted
                else 0.0
            )
            row["counter_delta"] = {
                key: value for key, value in sorted(deltas.items()) if value
            }
            row["agent_wait_ms_estimate"] = round(agent_wait_estimate, 3)
            totals["agent_admissions"] += admitted
            totals["agent_queued"] += deltas.get("queued_agent", 0)
            totals["agent_completed"] += deltas.get("completed_agent", 0)
            totals["agent_failed"] += deltas.get("failed_agent", 0)
            totals["agent_wait_ms_estimate"] += agent_wait_estimate
            totals["max_active"] = max(totals["max_active"], int(row["max_active"]))
            totals["max_queued"] = max(totals["max_queued"], int(row["max_queued"]))
            totals["max_oldest_queue_age_sec"] = max(
                totals["max_oldest_queue_age_sec"],
                float(row["max_oldest_queue_age_sec"]),
            )
            compact_resources.append(row)
        totals["agent_wait_ms_estimate"] = round(
            float(totals["agent_wait_ms_estimate"]),
            3,
        )
        totals["max_oldest_queue_age_sec"] = round(
            float(totals["max_oldest_queue_age_sec"]),
            3,
        )
        return {
            "sample_count": len(samples),
            "sample_errors": errors,
            **totals,
            "resources": sorted(compact_resources, key=lambda item: item["resource"]),
        }

    @property
    def performance_metrics(self) -> Dict[str, Any]:
        timings = self.tool_timings
        timed_tool_ms = sum(
            float(row.get("duration_ms") or 0.0)
            for row in timings
        )
        received = [
            float(value)
            for value in (
                _finite_number(event.get("_received_at_sec"))
                for event in self.events
            )
            if value is not None
        ]

        def first_at(event_type: str) -> Optional[float]:
            for event in self.events:
                if event.get("type") != event_type:
                    continue
                value = _finite_number(event.get("_received_at_sec"))
                if value is not None:
                    return round(float(value), 6)
            return None

        return {
            "elapsed_seconds": round(float(self.elapsed_seconds), 6),
            "first_event_seconds": round(min(received), 6) if received else None,
            "first_tool_call_seconds": first_at("tool_call"),
            "first_text_seconds": first_at("text"),
            "done_seconds": first_at("done"),
            "tool_wall_ms": round(timed_tool_ms, 3),
            "non_tool_wall_ms": round(
                max(0.0, float(self.elapsed_seconds) * 1000.0 - timed_tool_ms),
                3,
            ),
            "tool_timings": timings,
            "lm_admission": self.admission_metrics,
        }

    @property
    def dangling_tool_calls(self) -> List[str]:
        """Return calls that never received a matching tool_result event."""

        call_ids = [
            str(event.get("call_id"))
            for event in self.events
            if event.get("type") == "tool_call" and event.get("call_id")
        ]
        result_ids = {
            str(event.get("call_id"))
            for event in self.events
            if event.get("type") == "tool_result" and event.get("call_id")
        }
        dangling = [call_id for call_id in call_ids if call_id not in result_ids]

        # Recorded/legacy fixtures may omit call_id. Match those by tool name.
        calls_without_ids = Counter(
            str(event.get("name") or "")
            for event in self.events
            if event.get("type") == "tool_call" and not event.get("call_id")
        )
        results_without_ids = Counter(
            str(event.get("name") or "")
            for event in self.events
            if event.get("type") == "tool_result" and not event.get("call_id")
        )
        for name, count in calls_without_ids.items():
            dangling.extend([name] * max(0, count - results_without_ids[name]))
        return dangling

    def called(self, name: str) -> bool:
        return any(n == name for n, _ in self.tool_calls)

    def calls_of(self, name: str) -> List[Dict[str, Any]]:
        return [args for n, args in self.tool_calls if n == name]

    def result_of(self, name: str) -> Optional[Any]:
        for n, result, _err in self.tool_results:
            if n == name:
                return result
        return None

    def results_of(self, name: str) -> List[Any]:
        return [result for n, result, _err in self.tool_results if n == name]

    def errors_of(self, name: str) -> List[str]:
        return [str(error) for n, _result, error in self.tool_results if n == name and error]

    def approval_plan_ids(self) -> List[str]:
        ids: List[str] = []
        for _n, result, _err in self.tool_results:
            if isinstance(result, dict):
                for key in ("approval", "action_plan"):
                    approval = result.get(key)
                    if isinstance(approval, dict) and approval.get("plan_id"):
                        plan_id = str(approval["plan_id"])
                        if plan_id not in ids:
                            ids.append(plan_id)
        return ids

    def approval_plan_ids_for(self, tool_name: str) -> List[str]:
        """Return approval IDs produced by one exact tool."""

        ids: List[str] = []
        for name, result, _err in self.tool_results:
            if name != tool_name or not isinstance(result, dict):
                continue
            for key in ("approval", "action_plan"):
                approval = result.get(key)
                if not isinstance(approval, dict) or not approval.get("plan_id"):
                    continue
                plan_id = str(approval["plan_id"])
                if plan_id not in ids:
                    ids.append(plan_id)
        return ids

    def prose_has(self, pattern: str) -> bool:
        return re.search(pattern, self.text, flags=re.IGNORECASE) is not None


def combine_transcripts(transcripts: List[Transcript]) -> Transcript:
    """Combine an explicit multi-turn scenario without mixing setup prose.

    Tool and budget events from setup turns remain observable, while prose and
    terminal events come only from the final turn. This lets checks cover an
    intentional workflow without warnings matching a clarification in turn one.
    """

    if not transcripts:
        return Transcript()
    events: List[Dict[str, Any]] = []
    telemetry: List[Dict[str, Any]] = []
    offset = 0.0
    for index, transcript in enumerate(transcripts):
        is_last = index == len(transcripts) - 1
        for raw_event in transcript.events:
            if not is_last and raw_event.get("type") in {"text", "session", "done"}:
                continue
            event = dict(raw_event)
            received = _finite_number(event.get("_received_at_sec"))
            if received is not None:
                event["_received_at_sec"] = round(offset + received, 6)
            events.append(event)
        for raw_sample in transcript.telemetry_samples:
            sample = dict(raw_sample)
            sampled_at = _finite_number(sample.get("at_sec"))
            if sampled_at is not None:
                sample["at_sec"] = round(offset + sampled_at, 6)
            telemetry.append(sample)
        offset += float(transcript.elapsed_seconds)
    return Transcript(
        events=events,
        elapsed_seconds=offset,
        telemetry_samples=telemetry,
    )


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) == float("inf"):
        return None
    return number


class EvaSession:
    """Authenticated live session against a running EVA AI instance."""

    def __init__(
        self,
        base_url: str,
        *,
        csrf_cookie: str = "eva_csrf",
        verify_tls: bool = False,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.csrf_cookie = csrf_cookie
        self.verify_tls = verify_tls
        self.timeout = timeout
        self.http = requests.Session()
        self.http.verify = verify_tls
        if not verify_tls:
            try:  # silence expected self-signed dev cert warnings only for this mode
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:  # pragma: no cover
                pass

    def _csrf_headers(self) -> Dict[str, str]:
        token = self.http.cookies.get(self.csrf_cookie) or ""
        return {"X-CSRF-Token": token} if token else {}

    def login(self, username: str, password: str) -> Dict[str, Any]:
        resp = self.http.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        # Site-specific deployments deliberately rename the CSRF cookie.  The
        # authenticated contract tells clients the effective name; learning it
        # here keeps the harness aligned with the same runtime ground truth as
        # the React client instead of assuming the development default.
        if isinstance(payload, dict):
            csrf_cookie = str(payload.get("csrfCookie") or "").strip()
            if csrf_cookie:
                self.csrf_cookie = csrf_cookie
        return payload

    def whoami(self) -> Dict[str, Any]:
        resp = self.http.get(f"{self.base_url}/auth/me", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_json(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """GET one authenticated frontend endpoint and require an object response."""

        normalized = "/" + str(path or "").lstrip("/")
        resp = self.http.get(
            f"{self.base_url}{normalized}",
            params=params or None,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise TypeError(f"{normalized} returned {type(payload).__name__}, expected object")
        return payload

    def post_json(
        self,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST one authenticated frontend endpoint and require an object response."""

        normalized = "/" + str(path or "").lstrip("/")
        resp = self.http.post(
            f"{self.base_url}{normalized}",
            json=body or {},
            headers=self._csrf_headers(),
            timeout=self.timeout,
        )
        try:
            payload = resp.json()
        except ValueError:
            payload = {"raw": resp.text[:500]}
        if not isinstance(payload, dict):
            raise TypeError(f"{normalized} returned {type(payload).__name__}, expected object")
        return {"status": resp.status_code, **payload}

    def ask(
        self,
        message: str,
        session_id: Optional[str] = None,
        image_b64: Optional[str] = None,
        *,
        operator_mode: bool = False,
        console_context: Optional[Dict[str, Any]] = None,
        telemetry_interval_sec: float = 0.25,
    ) -> Transcript:
        body: Dict[str, Any] = {"message": message}
        if session_id:
            body["session_id"] = session_id
        if image_b64:
            body["image_b64"] = image_b64
        if operator_mode:
            body["operator_mode"] = True
        if isinstance(console_context, dict):
            body["console_context"] = dict(console_context)
        started = time.monotonic()
        telemetry_samples: List[Dict[str, Any]] = []
        telemetry_stop = threading.Event()
        telemetry_interval = max(0.1, min(5.0, float(telemetry_interval_sec)))

        def sample_admission() -> None:
            http = requests.Session()
            http.verify = self.verify_tls
            http.cookies.update(self.http.cookies)
            try:
                while not telemetry_stop.is_set():
                    sampled_at = max(0.0, time.monotonic() - started)
                    try:
                        response = http.get(
                            f"{self.base_url}/lm/admission",
                            timeout=min(self.timeout, 5.0),
                        )
                        payload = response.json()
                        telemetry_samples.append({
                            "path": "/lm/admission",
                            "at_sec": round(sampled_at, 6),
                            "status": response.status_code,
                            "payload": payload if isinstance(payload, dict) else {},
                        })
                    except Exception as exc:
                        telemetry_samples.append({
                            "path": "/lm/admission",
                            "at_sec": round(sampled_at, 6),
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                        })
                    telemetry_stop.wait(telemetry_interval)
            finally:
                http.close()

        sampler = threading.Thread(
            target=sample_admission,
            name="eva-live-agent-admission-sampler",
            daemon=True,
        )
        sampler.start()
        resp = self.http.post(
            f"{self.base_url}/agent/chat",
            json=body,
            headers={**self._csrf_headers(), "Accept": "text/event-stream"},
            stream=True,
            timeout=self.timeout,
        )
        try:
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                detail = resp.text.strip()[:500]
                raise requests.HTTPError(
                    f"{exc}; response={detail}",
                    response=resp,
                ) from exc
            events = parse_sse_events(
                resp.iter_lines(),
                elapsed_fn=lambda: time.monotonic() - started,
                deadline_sec=self.timeout,
            )
            elapsed = time.monotonic() - started
        except SseDeadlineExceeded as exc:
            elapsed = time.monotonic() - started
            events = list(exc.events)
            events.append({
                "type": "error",
                "error": "wall_clock_deadline_exceeded",
                "message": str(exc),
                "deadline_seconds": round(float(exc.deadline_sec), 3),
                "_received_at_sec": round(float(elapsed), 6),
            })
        finally:
            resp.close()
            telemetry_stop.set()
            sampler.join(timeout=6.0)
        return Transcript(
            events=events,
            elapsed_seconds=elapsed,
            telemetry_samples=telemetry_samples,
        )

    def apply_plan(self, plan_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Simulate the UI 'Apply' button: commit a previewed action plan."""
        resp = self.http.post(
            f"{self.base_url}/agent/action-plans/{plan_id}/execute",
            json={"session_id": session_id} if session_id else {},
            headers=self._csrf_headers(),
            timeout=self.timeout,
        )
        # may legitimately return 4xx (e.g. approval not enabled); return the body either way
        try:
            return {"status": resp.status_code, **resp.json()}
        except ValueError:
            return {"status": resp.status_code, "raw": resp.text[:500]}
