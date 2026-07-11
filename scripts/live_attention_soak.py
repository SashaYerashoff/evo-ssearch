#!/usr/bin/env python3
"""Read-only randomized EVA agent/VLM contention soak.

Credentials are read from environment, never command-line arguments:
  EVA_LIVE_PASSWORD=... scripts/live_attention_soak.py --from-ts ... --to-ts ...
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.integration.eva_client import EvaSession, Transcript


@dataclass
class SoakResult:
    started_at: float
    finished_at: float = 0.0
    seed: int = 0
    worker_count: int = 0
    turns: int = 0
    sse_errors: int = 0
    request_errors: List[str] = field(default_factory=list)
    invariant_errors: List[str] = field(default_factory=list)
    desired_snapshots: List[List[int]] = field(default_factory=list)
    lm_queue_peak: int = 0
    lm_active_peak: int = 0
    tool_calls: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": max(0.0, self.finished_at - self.started_at),
            "seed": self.seed,
            "worker_count": self.worker_count,
            "turns": self.turns,
            "sse_errors": self.sse_errors,
            "request_errors": self.request_errors,
            "invariant_errors": self.invariant_errors,
            "desired_snapshots": self.desired_snapshots,
            "lm_queue_peak": self.lm_queue_peak,
            "lm_active_peak": self.lm_active_peak,
            "tool_calls": dict(sorted(self.tool_calls.items())),
            "ok": not self.sse_errors and not self.request_errors and not self.invariant_errors,
        }


def _session(base_url: str, username: str, password: str, verify_tls: bool) -> EvaSession:
    session = EvaSession(base_url, verify_tls=verify_tls, timeout=900)
    session.login(username, password)
    whoami = session.whoami().get("user") or {}
    if "*" not in {str(item) for item in whoami.get("allowedChannelIds") or []}:
        raise PermissionError("soak requires a named all-channel admin session")
    return session


def _record_transcript(
    transcript: Transcript,
    *,
    expected_channel: Optional[int],
    from_ts: float,
    to_ts: float,
    events: "queue.Queue[Dict[str, Any]]",
) -> None:
    payload: Dict[str, Any] = {
        "kind": "turn",
        "errored": transcript.errored,
        "session_id": transcript.session_id,
        "tool_calls": [name for name, _args in transcript.tool_calls],
        "errors": [error for _name, _result, error in transcript.tool_results if error],
        "invariants": [],
    }
    for name, args in transcript.tool_calls:
        if expected_channel is not None and name in {
            "get_video_summaries",
            "get_detections",
            "search_archive",
            "generate_report",
        }:
            raw_channel = args.get("channel_id")
            if raw_channel is not None and int(raw_channel) != expected_channel:
                payload["invariants"].append(
                    f"{name} drifted to channel {raw_channel}, expected {expected_channel}"
                )
        raw_from = args.get("from_ts")
        raw_to = args.get("to_ts")
        if raw_from is not None and abs(float(raw_from) - from_ts) > 0.001:
            payload["invariants"].append(f"{name} changed from_ts to {raw_from}")
        if raw_to is not None and abs(float(raw_to) - to_ts) > 0.001:
            payload["invariants"].append(f"{name} changed to_ts to {raw_to}")
    events.put(payload)


def run_soak(
    *,
    base_url: str,
    username: str,
    password: str,
    channels: List[int],
    from_ts: float,
    to_ts: float,
    duration_sec: float,
    seed: int,
    worker_count: int,
    poll_interval_sec: float,
    verify_tls: bool,
) -> SoakResult:
    if not channels:
        raise ValueError("at least one channel is required")
    stop = threading.Event()
    events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    result = SoakResult(started_at=time.time(), seed=seed, worker_count=worker_count)
    observer = _session(base_url, username, password, verify_tls)
    baseline_streams = observer.get_json("/luxriot/streams")
    baseline_desired = sorted(int(item) for item in baseline_streams.get("desired_video_channels") or [])

    prompts = (
        lambda rng: (
            "Use list_video_summary_channels for exactly channel IDs "
            f"{channels} from_ts={from_ts} to_ts={to_ts}. Preserve the exact scope and report unchecked IDs."
        ),
        lambda rng: (
            "Use get_video_summaries with evidence frames for numeric "
            f"channel_id={rng.choice(channels)}, depth=L0, from_ts={from_ts}, to_ts={to_ts}."
        ),
        lambda rng: (
            "Generate a read-only video-description report for numeric "
            f"channel_id={rng.choice(channels)}, from_ts={from_ts}, to_ts={to_ts}; state coverage."
        ),
        lambda rng: (
            "Search the archive read-only for visible vehicle drifting on numeric "
            f"channel_id={rng.choice(channels)}, from_ts={from_ts}, to_ts={to_ts}; state search coverage."
        ),
    )

    def worker(index: int) -> None:
        rng = random.Random(seed + index * 10_007)
        try:
            session = _session(base_url, username, password, verify_tls)
        except Exception as exc:
            events.put({"kind": "request_error", "error": f"worker login: {type(exc).__name__}: {exc}"})
            return
        session_id: Optional[str] = None
        while not stop.is_set():
            prompt = prompts[rng.randrange(len(prompts))](rng)
            expected_channel = None
            marker = "channel_id="
            if marker in prompt:
                try:
                    expected_channel = int(prompt.split(marker, 1)[1].split(",", 1)[0])
                except Exception:
                    expected_channel = None
            try:
                transcript = session.ask(prompt, session_id=session_id)
                session_id = transcript.session_id or session_id
                _record_transcript(
                    transcript,
                    expected_channel=expected_channel,
                    from_ts=from_ts,
                    to_ts=to_ts,
                    events=events,
                )
            except Exception as exc:
                events.put({"kind": "request_error", "error": f"agent turn: {type(exc).__name__}: {exc}"})
            stop.wait(rng.uniform(0.2, 1.0))

    def poller() -> None:
        while not stop.is_set():
            try:
                streams = observer.get_json("/luxriot/streams")
                desired = sorted(int(item) for item in streams.get("desired_video_channels") or [])
                admission = observer.get_json("/lm/admission")
                events.put(
                    {
                        "kind": "poll",
                        "desired": desired,
                        "baseline_desired": baseline_desired,
                        "lm_queued": int(admission.get("queued") or 0),
                        "lm_active": int(admission.get("active") or 0),
                    }
                )
            except Exception as exc:
                events.put({"kind": "request_error", "error": f"poll: {type(exc).__name__}: {exc}"})
            stop.wait(max(0.5, poll_interval_sec))

    threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(worker_count)]
    threads.append(threading.Thread(target=poller, daemon=True))
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + max(1.0, duration_sec)
    try:
        while time.monotonic() < deadline:
            try:
                event = events.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if event["kind"] == "turn":
                result.turns += 1
                result.sse_errors += int(bool(event.get("errored")))
                for name in event.get("tool_calls") or []:
                    result.tool_calls[name] = int(result.tool_calls.get(name) or 0) + 1
                result.request_errors.extend(str(item) for item in event.get("errors") or [])
                result.invariant_errors.extend(str(item) for item in event.get("invariants") or [])
            elif event["kind"] == "poll":
                desired = list(event.get("desired") or [])
                result.desired_snapshots.append(desired)
                if desired != event.get("baseline_desired"):
                    result.invariant_errors.append(
                        f"desired stream set changed: {event.get('baseline_desired')} -> {desired}"
                    )
                result.lm_queue_peak = max(result.lm_queue_peak, int(event.get("lm_queued") or 0))
                result.lm_active_peak = max(result.lm_active_peak, int(event.get("lm_active") or 0))
            else:
                result.request_errors.append(str(event.get("error") or "unknown request error"))
            print(json.dumps(event, ensure_ascii=False, default=str), flush=True)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
        result.finished_at = time.time()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("EVA_LIVE_BASE_URL", "https://127.0.0.1:5443"))
    parser.add_argument("--user", default=os.getenv("EVA_LIVE_USER", "admin"))
    parser.add_argument("--channels", default=os.getenv("EVA_LIVE_CHANNEL_IDS", "112,118,120"))
    parser.add_argument("--from-ts", type=float, required=True)
    parser.add_argument("--to-ts", type=float, required=True)
    parser.add_argument("--duration", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--poll", type=float, default=3.0)
    parser.add_argument("--verify-tls", action="store_true")
    args = parser.parse_args()
    password = os.getenv("EVA_LIVE_PASSWORD", "")
    if not password:
        parser.error("EVA_LIVE_PASSWORD must be set in the environment")
    channels = [int(item.strip()) for item in args.channels.split(",") if item.strip()]
    report = run_soak(
        base_url=args.base_url,
        username=args.user,
        password=password,
        channels=channels,
        from_ts=args.from_ts,
        to_ts=args.to_ts,
        duration_sec=args.duration,
        seed=args.seed,
        worker_count=max(1, args.workers),
        poll_interval_sec=args.poll,
        verify_tls=args.verify_tls,
    )
    print(json.dumps({"kind": "final", **report.as_dict()}, ensure_ascii=False, indent=2), flush=True)
    return 0 if report.as_dict()["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
