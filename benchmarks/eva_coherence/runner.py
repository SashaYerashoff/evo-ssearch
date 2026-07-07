"""Deterministic coherence checks for the EVA agent tool chain.

This harness intentionally avoids a live LLM. It executes scripted tool plans
against fake stores/managers and validates the tool-contract invariants that a
real agent answer must respect.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import (  # noqa: E402
    AgentTools,
    _apply_turn_tool_context,
    _compact_tool_result_for_model,
    _remember_turn_tool_result,
    _seed_turn_tool_context,
)


class EmptyProbeStore:
    def list_probes(self) -> List[Dict[str, Any]]:
        return []


def _as_dict(value: Any) -> Dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    data: Dict[str, Any] = {}
    for key in (
        "id",
        "title",
        "prompt",
        "tool_plan",
        "checks",
        "notes",
        "expected_answer",
    ):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    return data


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    return str(scenario.get("id") or scenario.get("slug") or "").strip()


def _scenario_time_args(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    if scenario.get("from_ts") is not None:
        args["from_ts"] = scenario.get("from_ts")
    if scenario.get("to_ts") is not None:
        args["to_ts"] = scenario.get("to_ts")
    return args


def _scenario_ms_args(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    if scenario.get("from_ts") is not None:
        args["since_ms"] = int(float(scenario["from_ts"]) * 1000)
    if scenario.get("to_ts") is not None:
        args["until_ms"] = int(float(scenario["to_ts"]) * 1000)
    return args


def _default_tool_plan(scenario: Mapping[str, Any]) -> List[Dict[str, Any]]:
    slug = _scenario_id(scenario)
    channel_ids = list(scenario.get("channel_ids") or [])
    primary_channel = int(channel_ids[0]) if channel_ids else None
    time_args = _scenario_time_args(scenario)
    ms_args = _scenario_ms_args(scenario)
    if slug == "zenbook_orlandina_transition":
        return [
            {"name": "get_video_summaries", "args": {"channel_id": primary_channel, "depth": "L2", "limit": 8, **time_args}},
            {"name": "get_video_summaries", "args": {"channel_id": primary_channel, "depth": "L1", "limit": 8, **time_args}},
            {
                "name": "get_video_summaries",
                "args": {
                    "channel_id": primary_channel,
                    "depth": "live",
                    "limit": 12,
                    "include_evidence_frames": True,
                    "evidence_frame_limit": 6,
                    **time_args,
                },
            },
            {
                "name": "get_visual_window_signals",
                "args": {
                    "channel_id": primary_channel,
                    "positive_query": "Orlandina visible at the desk",
                    "negative_query": "empty desk with no cat visible",
                    "limit_per_source": 4,
                    **ms_args,
                },
            },
            {"name": "describe_frame", "args": {"detection_id": 112003, "prompt": "Describe whether Orlandina is visible."}},
        ]
    if slug == "multi_channel_event_sweep":
        return [
            {"name": "list_video_summary_channels", "args": {"depth": "L1", "limit": 12, **time_args}},
        ]
    if slug == "dog_no_visible_ear_tag":
        return [
            {"name": "get_video_summaries", "args": {"channel_id": primary_channel, "depth": "L2", "limit": 4, **time_args}},
            {"name": "get_video_summaries", "args": {"channel_id": primary_channel, "depth": "L1", "limit": 4, **time_args}},
            {
                "name": "get_visual_window_signals",
                "args": {
                    "channel_id": primary_channel,
                    "positive_query": "dog without visible ear tag",
                    "negative_query": "dog with visible ear tag",
                    "limit_per_source": 4,
                    **ms_args,
                },
            },
            {
                "name": "get_video_summaries",
                "args": {
                    "channel_id": primary_channel,
                    "depth": "live",
                    "limit": 6,
                    "include_evidence_frames": True,
                    "evidence_frame_limit": 4,
                    **time_args,
                },
            },
            {"name": "describe_frame", "args": {"detection_id": 305002, "prompt": "Describe visible dog markers only."}},
        ]
    return []


def _default_checks(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    slug = _scenario_id(scenario)
    if slug == "zenbook_orlandina_transition":
        return {
            "required_tools": [
                "get_video_summaries",
                "get_visual_window_signals",
                "describe_frame",
            ],
            "required_order": ["get_video_summaries", "get_visual_window_signals", "describe_frame"],
            "require_coverage_contract": True,
            "require_vlm_evidence_sources": True,
            "require_describe_frame_for_visual_confirmation": True,
            "pnm": {"min_margin": 0.02},
        }
    if slug == "multi_channel_event_sweep":
        return {
            "required_tools": ["list_video_summary_channels"],
            "require_confirmation_for_many_channels": True,
            "min_active_channels": 9,
        }
    if slug == "dog_no_visible_ear_tag":
        return {
            "required_tools": [
                "get_video_summaries",
                "get_visual_window_signals",
                "describe_frame",
            ],
            "require_coverage_contract": True,
            "require_vlm_evidence_sources": True,
            "require_describe_frame_for_visual_confirmation": True,
            "pnm": {"min_margin": 0.02},
            "required_answer_phrases": ["no visible ear tag", "cannot be determined"],
            "forbidden_answer_phrases": ["is unvaccinated", "was unvaccinated", "confirmed unvaccinated"],
        }
    return {}


def _default_expected_answer(scenario: Mapping[str, Any]) -> str:
    slug = _scenario_id(scenario)
    if slug == "dog_no_visible_ear_tag":
        return (
            "Candidate finding: a dog is visible near the clinic side door with no visible ear tag. "
            "Vaccination status cannot be determined from the image."
        )
    if slug == "zenbook_orlandina_transition":
        return (
            "Coverage: checked the requested two-hour window. Orlandina is reported absent before "
            "about 07:05 UTC and present afterward; visual confirmation comes only from returned "
            "VLM evidence frames and describe_frame."
        )
    return ""


def _normalize_scenario(raw: Any) -> Dict[str, Any]:
    scenario = _as_dict(raw)
    if "id" not in scenario and scenario.get("slug"):
        scenario["id"] = scenario.get("slug")
    scenario.setdefault("tool_plan", _default_tool_plan(scenario))
    scenario.setdefault("checks", _default_checks(scenario))
    if not scenario.get("expected_answer"):
        scenario["expected_answer"] = _default_expected_answer(scenario)
    return scenario


def _row_ts(row: Mapping[str, Any]) -> int:
    for key in ("timestamp_ms", "event_timestamp_ms", "recorded_at_ms"):
        try:
            raw = row.get(key)
            if raw is not None:
                return int(raw)
        except Exception:
            continue
    return 0


def _score(row: Mapping[str, Any]) -> float:
    for key in ("similarity", "score", "clip_similarity"):
        try:
            raw = row.get(key)
            if raw is not None:
                return float(raw)
        except Exception:
            continue
    return 0.0


def _build_search_fn(detections_store: Any):
    def search_detections(
        *,
        query: str,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        source: Optional[str] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 12,
        sort_by: str = "similarity",
        candidate_limit: int = 20000,
        mode: str = "clip",
    ) -> List[Dict[str, Any]]:
        if hasattr(detections_store, "search_detections"):
            return detections_store.search_detections(
                query=query,
                probe_id=probe_id,
                channel_id=channel_id,
                source=source,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                sort_by=sort_by,
                candidate_limit=candidate_limit,
                mode=mode,
            )
        rows, _total = detections_store.list_detections(
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=min(max(int(candidate_limit or limit), int(limit)), 5000),
            offset=0,
        )
        out = [dict(row) for row in rows]
        if sort_by == "time":
            out.sort(key=_row_ts, reverse=True)
        else:
            out.sort(key=_score, reverse=True)
        return out[:limit]

    return search_detections


def _make_tools(env: Mapping[str, Any]) -> AgentTools:
    detections_store = env["detections_store"]
    luxriot_manager = env["luxriot_manager"]
    probes_store = env.get("probes_store") or EmptyProbeStore()
    return AgentTools(
        detections_store=detections_store,
        probes_store=probes_store,
        luxriot_manager=luxriot_manager,
        embed_text_fn=lambda _text: None,
        embed_image_fn=lambda _image: None,
        call_lm_fn=lambda *_args, **_kwargs: "scripted visual description",
        encode_jpeg_fn=lambda *_args, **_kwargs: "",
        search_indexed_folder_fn=lambda **_kwargs: [],
        search_detections_fn=_build_search_fn(detections_store),
    )


def _load_scenarios() -> Tuple[List[Dict[str, Any]], Mapping[str, Any]]:
    try:
        from benchmarks.eva_coherence.fakes import build_environment
        from benchmarks.eva_coherence.scenarios import SCENARIOS
    except Exception as exc:  # pragma: no cover - CLI failure path
        raise SystemExit(f"Could not import EVA coherence fixtures: {exc}") from exc
    raw_scenarios: Iterable[Any]
    if isinstance(SCENARIOS, Mapping):
        raw_scenarios = SCENARIOS.values()
    else:
        raw_scenarios = SCENARIOS
    return [_normalize_scenario(item) for item in raw_scenarios], build_environment()


def _select_scenarios(scenarios: Sequence[Dict[str, Any]], selector: str) -> List[Dict[str, Any]]:
    if selector == "all":
        return list(scenarios)
    selected = [scenario for scenario in scenarios if str(scenario.get("id")) == selector]
    if not selected:
        known = ", ".join(str(item.get("id")) for item in scenarios)
        raise SystemExit(f"Unknown scenario {selector!r}. Known: {known}")
    return selected


def _execute_plan(
    tools: AgentTools,
    scenario: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    context = _seed_turn_tool_context(scenario.get("prompt") or "")
    transcript: List[Dict[str, Any]] = []
    for index, raw_step in enumerate(scenario.get("tool_plan") or []):
        step = _as_dict(raw_step)
        name = str(step.get("name") or "").strip()
        args = dict(step.get("args") or {})
        if not name:
            transcript.append({
                "index": index,
                "ok": False,
                "error": "missing tool name",
            })
            continue
        prepared = _apply_turn_tool_context(name, args, context)
        try:
            result = tools.execute(name, prepared)
            compact = _compact_tool_result_for_model(name, result)
            _remember_turn_tool_result(name, result, context)
            transcript.append(
                {
                    "index": index,
                    "name": name,
                    "args": prepared,
                    "ok": True,
                    "result": result,
                    "compact": compact,
                }
            )
        except Exception as exc:
            transcript.append(
                {
                    "index": index,
                    "name": name,
                    "args": prepared,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return transcript, context


def _all_tool_names(transcript: Sequence[Mapping[str, Any]]) -> List[str]:
    return [str(step.get("name")) for step in transcript if step.get("ok")]


def _tool_results(transcript: Sequence[Mapping[str, Any]], name: str) -> List[Mapping[str, Any]]:
    return [
        step.get("result") or {}
        for step in transcript
        if step.get("ok") and step.get("name") == name and isinstance(step.get("result"), Mapping)
    ]


def _flatten_detections(result: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    rows = result.get("detections")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    rows = result.get("evidence_frames")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    rows = result.get("candidate_frames")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def _answer_text(scenario: Mapping[str, Any]) -> str:
    return str(scenario.get("expected_answer") or scenario.get("answer") or "")


def _check_required_tools(
    checks: Mapping[str, Any],
    transcript: Sequence[Mapping[str, Any]],
) -> List[str]:
    failures: List[str] = []
    names = _all_tool_names(transcript)
    for tool_name in checks.get("required_tools") or []:
        if tool_name not in names:
            failures.append(f"required tool was not called: {tool_name}")
    ordered = checks.get("required_order") or []
    if ordered:
        cursor = 0
        for tool_name in names:
            if cursor < len(ordered) and tool_name == ordered[cursor]:
                cursor += 1
        if cursor < len(ordered):
            failures.append(f"required tool order not satisfied: {ordered}")
    return failures


def _check_video_coverage(
    checks: Mapping[str, Any],
    transcript: Sequence[Mapping[str, Any]],
) -> List[str]:
    if not checks.get("require_coverage_contract"):
        return []
    failures: List[str] = []
    results = _tool_results(transcript, "get_video_summaries")
    if not results:
        return ["coverage contract required but get_video_summaries was not called"]
    for result in results:
        coverage = result.get("coverage")
        if not isinstance(coverage, Mapping):
            failures.append("get_video_summaries result lacks coverage")
            continue
        if not coverage.get("must_state_coverage"):
            failures.append("coverage.must_state_coverage is not true")
        if coverage.get("status") not in {"covered", "partial", "truncated", "no_data"}:
            failures.append(f"unexpected coverage status: {coverage.get('status')!r}")
    return failures


def _check_evidence_grounding(
    checks: Mapping[str, Any],
    scenario: Mapping[str, Any],
    transcript: Sequence[Mapping[str, Any]],
) -> List[str]:
    failures: List[str] = []
    if checks.get("require_vlm_evidence_sources"):
        allowed = {"vlm_summary", "vlm_alert"}
        rows: List[Mapping[str, Any]] = []
        for result in _tool_results(transcript, "get_video_summaries"):
            rows.extend(_flatten_detections(result))
        for result in _tool_results(transcript, "get_detections"):
            rows.extend(_flatten_detections(result))
        if not rows:
            failures.append("VLM evidence frames required but no frame rows were returned")
        for row in rows:
            source = str(row.get("source") or "").strip()
            if source and source not in allowed:
                failures.append(f"non-VLM evidence source returned: {source}")
            if source in allowed and not row.get("image_url"):
                failures.append(f"VLM evidence row lacks image_url: {row.get('id') or row.get('detection_id')}")
    if checks.get("require_describe_frame_for_visual_confirmation"):
        text = _answer_text(scenario).lower()
        says_confirmed = "confirmed visually" in text or "visual confirmation" in text
        if says_confirmed and "describe_frame" not in _all_tool_names(transcript):
            failures.append("answer claims visual confirmation without describe_frame")
    return failures


def _check_multi_channel_confirmation(
    checks: Mapping[str, Any],
    transcript: Sequence[Mapping[str, Any]],
) -> List[str]:
    if not checks.get("require_confirmation_for_many_channels"):
        return []
    results = _tool_results(transcript, "list_video_summary_channels")
    if not results:
        return ["multi-channel confirmation required but list_video_summary_channels was not called"]
    result = results[-1]
    failures: List[str] = []
    active_count = result.get("active_count")
    min_active = checks.get("min_active_channels")
    if min_active is not None and (active_count is None or int(active_count) < int(min_active)):
        failures.append(f"active channel count below expected fixture size: {active_count} < {min_active}")
    if not result.get("requires_confirmation"):
        failures.append("broad active channel set did not require confirmation/chunking")
    if int(result.get("returned") or 0) > int(result.get("per_turn_channel_limit") or 8):
        failures.append("candidate channel list exceeded per-turn channel limit")
    return failures


def _check_pnm(
    checks: Mapping[str, Any],
    transcript: Sequence[Mapping[str, Any]],
) -> List[str]:
    expected = checks.get("pnm")
    if not isinstance(expected, Mapping):
        return []
    failures: List[str] = []
    results = _tool_results(transcript, "get_visual_window_signals")
    if not results:
        return ["P/N/M expectation exists but get_visual_window_signals was not called"]
    pnm = results[-1].get("pnm") if isinstance(results[-1].get("pnm"), Mapping) else {}
    min_margin = expected.get("min_margin")
    max_margin = expected.get("max_margin")
    state = expected.get("state")
    margin = pnm.get("m")
    if min_margin is not None and (margin is None or float(margin) < float(min_margin)):
        failures.append(f"P/N/M margin below expected minimum: {margin} < {min_margin}")
    if max_margin is not None and (margin is None or float(margin) > float(max_margin)):
        failures.append(f"P/N/M margin above expected maximum: {margin} > {max_margin}")
    if state is not None and pnm.get("state") != state:
        failures.append(f"P/N/M state mismatch: {pnm.get('state')!r} != {state!r}")
    if pnm.get("score_semantics") != "clip_retrieval_signal_not_proof":
        failures.append("P/N/M score_semantics must be clip_retrieval_signal_not_proof")
    return failures


def _check_answer_constraints(scenario: Mapping[str, Any]) -> List[str]:
    answer = _answer_text(scenario)
    if not answer:
        return []
    checks = scenario.get("checks") if isinstance(scenario.get("checks"), Mapping) else {}
    failures: List[str] = []
    lowered = answer.lower()
    for phrase in checks.get("required_answer_phrases") or []:
        if str(phrase).lower() not in lowered:
            failures.append(f"answer missing required phrase: {phrase!r}")
    for phrase in checks.get("forbidden_answer_phrases") or []:
        if str(phrase).lower() in lowered:
            failures.append(f"answer contains forbidden phrase: {phrase!r}")
    return failures


def _evaluate_scenario(
    scenario: Mapping[str, Any],
    transcript: Sequence[Mapping[str, Any]],
) -> List[str]:
    checks = scenario.get("checks") if isinstance(scenario.get("checks"), Mapping) else {}
    failures: List[str] = []
    failures.extend(step.get("error", "tool call failed") for step in transcript if not step.get("ok"))
    failures.extend(_check_required_tools(checks, transcript))
    failures.extend(_check_video_coverage(checks, transcript))
    failures.extend(_check_evidence_grounding(checks, scenario, transcript))
    failures.extend(_check_multi_channel_confirmation(checks, transcript))
    failures.extend(_check_pnm(checks, transcript))
    failures.extend(_check_answer_constraints(scenario))
    return failures


def _result_summary(transcript: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "index": step.get("index"),
            "name": step.get("name"),
            "ok": bool(step.get("ok")),
            "args": step.get("args"),
            "compact": step.get("compact") if step.get("ok") else None,
            "error": step.get("error"),
        }
        for step in transcript
    ]


def run_scenario(scenario: Mapping[str, Any], env: Mapping[str, Any]) -> Dict[str, Any]:
    tools = _make_tools(env)
    transcript, context = _execute_plan(tools, scenario)
    failures = _evaluate_scenario(scenario, transcript)
    return {
        "id": scenario.get("id"),
        "title": scenario.get("title"),
        "ok": not failures,
        "failures": failures,
        "tool_calls": _result_summary(transcript),
        "context": context,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run EVA agent coherence scenarios.")
    parser.add_argument("--scenario", default="all", help="Scenario id or 'all'.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON report.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first failed scenario.")
    args = parser.parse_args(argv)

    scenarios, env = _load_scenarios()
    selected = _select_scenarios(scenarios, str(args.scenario))
    results: List[Dict[str, Any]] = []
    for scenario in selected:
        result = run_scenario(scenario, env)
        results.append(result)
        if args.fail_fast and not result["ok"]:
            break

    report = {
        "ok": all(result["ok"] for result in results),
        "count": len(results),
        "failed": [result["id"] for result in results if not result["ok"]],
        "results": results,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        status = "OK" if report["ok"] else "FAIL"
        print(f"EVA coherence: {status} ({len(results)} scenario(s))")
        for result in results:
            marker = "ok" if result["ok"] else "FAIL"
            print(f"- {marker}: {result.get('id')} — {result.get('title') or ''}".rstrip())
            for failure in result["failures"]:
                print(f"  - {failure}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
