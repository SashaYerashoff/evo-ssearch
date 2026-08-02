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
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


def parse_sse_events(lines: Any) -> List[Dict[str, Any]]:
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
                events.append(obj)
                if payload == payloads[0]:
                    return

    for raw in lines:
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
    flush()
    return events


@dataclass
class Transcript:
    """Structured view over one agent turn's SSE events (assert on this, not prose)."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

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
                approval = result.get("approval")
                if isinstance(approval, dict) and approval.get("plan_id"):
                    ids.append(str(approval["plan_id"]))
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
    for transcript in transcripts[:-1]:
        events.extend(
            event
            for event in transcript.events
            if event.get("type") not in {"text", "session", "done"}
        )
    events.extend(transcripts[-1].events)
    return Transcript(
        events=events,
        elapsed_seconds=sum(item.elapsed_seconds for item in transcripts),
    )


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
        return resp.json()

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

    def ask(self, message: str, session_id: Optional[str] = None, image_b64: Optional[str] = None) -> Transcript:
        body: Dict[str, Any] = {"message": message}
        if session_id:
            body["session_id"] = session_id
        if image_b64:
            body["image_b64"] = image_b64
        started = time.monotonic()
        resp = self.http.post(
            f"{self.base_url}/agent/chat",
            json=body,
            headers={**self._csrf_headers(), "Accept": "text/event-stream"},
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return Transcript(
            events=parse_sse_events(resp.iter_lines()),
            elapsed_seconds=time.monotonic() - started,
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
