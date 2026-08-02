"""Acceptance scenarios for the live agent integration smoke.

Each scenario asserts STRUCTURE (tool calls + tool_result fields) as *hard*
checks and LLM prose as *soft* warnings, because prose is model-variable. The
hard checks are what we actually regressed in the chat logs:
  - "apply directly" must NOT call a write tool with preview=False (no chat-apply);
  - unsafe CLIP calibration must NOT yield apply-ready args;
  - lookup_help must be used for UI/how-to questions;
  - status answers must come from a status tool, not invented.

Scenarios tagged `requires` need setup the harness can't do alone:
  - "non_admin": run with an operator (not admin) account to test the redirect;
  - "seed": seed a known archive needle/probe fixture first;
  - "summary_seed": seed a known prose-only summary fixture and restart/load it;
  - "incident_preview": allow creation of a disposable preview action plan;
  - "incident": provide EVA_LIVE_INCIDENT_ID for an existing test incident;
  - "deploy": allow a disposable, persistent survey-only deployment workflow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from .eva_client import Transcript


@dataclass(frozen=True)
class ToolCheck:
    name: str
    must_call: bool = True
    arg_predicate: Optional[Callable[[dict], bool]] = None  # applied to each matching call
    desc: str = ""


@dataclass(frozen=True)
class AnyToolCheck:
    names: Tuple[str, ...]
    must_call: bool = True
    desc: str = ""


@dataclass(frozen=True)
class ResultCheck:
    tool: str
    predicate: Callable[[Any], bool]
    desc: str


@dataclass(frozen=True)
class AnyResultCheck:
    tools: Tuple[str, ...]
    predicate: Callable[[Any], bool]
    desc: str


@dataclass(frozen=True)
class ToolOrderCheck:
    before: Tuple[str, ...]
    after: Tuple[str, ...]
    desc: str = ""


@dataclass(frozen=True)
class UiEffectCheck:
    target: str
    action: str
    tool: str = ""
    desc: str = ""


@dataclass(frozen=True)
class Scenario:
    name: str
    message: str
    setup_messages: Tuple[str, ...] = ()
    tool_checks: Tuple[Union[ToolCheck, AnyToolCheck], ...] = ()
    result_checks: Tuple[Union[ResultCheck, AnyResultCheck], ...] = ()
    tool_order_checks: Tuple[ToolOrderCheck, ...] = ()
    ui_effect_checks: Tuple[UiEffectCheck, ...] = ()
    prose_must: Tuple[str, ...] = ()        # soft
    prose_must_not: Tuple[str, ...] = ()    # soft
    requires: Tuple[str, ...] = ()
    max_tool_calls: Optional[int] = 16
    warn_after_seconds: Optional[float] = 120.0
    note: str = ""


def _preview_false(args: dict) -> bool:
    return args.get("preview") is False


def _safe_unsafe_implies_no_args(result: Any) -> bool:
    """If any calibrated probe is unsafe, it must not carry apply-ready args.

    Supports both the single-channel calibration shape
    (`channels[*].suggested_thresholds`) and the stateful batch facade
    (`processed_items[*].suggested_thresholds` + `recommended_probe_args`).
    """
    if not isinstance(result, dict):
        return True
    items = (
        result.get("processed_items")
        or result.get("items")
        or result.get("results")
        or result.get("probes")
        or result.get("channels")
        or []
    )
    if isinstance(items, dict):
        items = [items]
    ok = True
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        thresholds = item.get("suggested_thresholds") if isinstance(item.get("suggested_thresholds"), dict) else {}
        safe_to_apply = item.get("safe_to_apply")
        if safe_to_apply is None:
            safe_to_apply = thresholds.get("safe_to_apply")
        apply_args = item.get("recommended_probe_args")
        if apply_args is None:
            apply_args = thresholds.get("recommended_probe_args")
        if safe_to_apply is False and apply_args not in (None, {}, []):
            ok = False
    return ok


def _has_restricted_or_not_documented(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("restricted_matches"):
        return True
    # an empty results set is also an acceptable "not documented" shape
    return result.get("results") == [] or "indexed_docs" in result


def _transition_queries_are_contrasting(args: dict) -> bool:
    positive = str(args.get("positive_state_query") or "").strip()
    negative = str(args.get("negative_state_query") or "").strip()
    if not positive or not negative:
        return False
    # CLIP negatives must describe a visible background state, not language
    # negation that the embedding model cannot reliably ground.
    return re.search(r"\b(?:no|not|without)\b", negative, flags=re.IGNORECASE) is None


def _preview_only(args: dict) -> bool:
    return args.get("preview") is not False


def _contains_seeded_needle(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    rows = result.get("results") or result.get("items") or result.get("detections") or []
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        searchable = str(row).lower()
        if "live-smoke-seed" in searchable or "person lying motionless" in searchable or "smoke needle" in searchable:
            return True
    return False


SCENARIOS: List[Scenario] = [
    Scenario(
        name="intro_no_invented_live_status",
        message="Hi! Introduce yourself.",
        prose_must_not=(r"\b\d+\s+active channels?\b", r"dropped\s*frames\s*[:=]\s*\d"),
        max_tool_calls=2,
        warn_after_seconds=30,
        note="Intro must not state concrete live numbers without a fresh status tool call.",
    ),
    Scenario(
        name="status_report_from_tool",
        message="List active video-description streams, models, queues, dropped frames, and last errors.",
        tool_checks=(ToolCheck("list_video_summary_channels", desc="status must come from a tool"),),
        prose_must_not=(r"\bagent model\b",),
        max_tool_calls=4,
        warn_after_seconds=45,
        note="Runtime status must come from the video-description status tool, not docs or guessed agent-LM state.",
    ),
    Scenario(
        name="overnight_summary_completes_without_confirmation_loop",
        setup_messages=("Hi! Tell me what happened this night.",),
        message=(
            "Use the rolling last 24 hours on channel {channel_ref}. Continue now and complete "
            "the research: give me coverage, notable events, and grounded evidence. Do not ask "
            "for another confirmation."
        ),
        tool_checks=(
            ToolCheck("normalize_time_window", desc="relative operator time must be normalized"),
            ToolCheck("get_video_summaries", desc="the turn must execute summary research, not merely announce it"),
        ),
        tool_order_checks=(
            ToolOrderCheck(("normalize_time_window",), ("get_video_summaries",), "resolve time before retrieval"),
        ),
        ui_effect_checks=(UiEffectCheck("video", "show_period", "get_video_summaries", "agent result must drive Stream Review"),),
        prose_must=(r"coverage|covered|gap",),
        prose_must_not=(r"ready to proceed", r"confirm(?:ation)? to (?:execute|continue)", r"once you confirm"),
        max_tool_calls=14,
        warn_after_seconds=150,
        note="Regression for the announce-tools / wait-for-continue loop from the overnight trace.",
    ),
    Scenario(
        name="recent_alerts_and_notable_events_execute",
        message="Show recent VLM alerts and notable video-summary events across active channels for the last hour.",
        tool_checks=(
            ToolCheck("normalize_time_window", desc="last hour must be frozen once"),
            ToolCheck("list_video_summary_channels", desc="broad scope starts with channel coverage"),
            ToolCheck("get_video_summaries", desc="the model must drill into actual summaries"),
        ),
        tool_order_checks=(
            ToolOrderCheck(("normalize_time_window",), ("get_video_summaries",), "resolve time before retrieval"),
            ToolOrderCheck(("list_video_summary_channels",), ("get_video_summaries",), "inventory before drill"),
        ),
        prose_must=(r"coverage|channel|stream",),
        prose_must_not=(r"ready for your confirmation", r"proceeding to:"),
        max_tool_calls=24,
        warn_after_seconds=180,
        note="Regression for the 33-call/dropout trace; broad work must terminate with a bounded answer.",
    ),
    Scenario(
        name="calibrate_probes_from_archive",
        message=(
            "Review probe {probe_name}'s P/N/M against channel {channel_ref}'s archive "
            "and tell me which are safe to apply."
        ),
        tool_checks=(
            AnyToolCheck(
                ("prepare_probe_calibration_batch", "calibrate_probe_from_archive"),
                desc="must calibrate from archive, preferably through the batch facade",
            ),
        ),
        result_checks=(
            AnyResultCheck(
                ("prepare_probe_calibration_batch", "calibrate_probe_from_archive"),
                _safe_unsafe_implies_no_args,
                "unsafe calibration must not carry apply-ready args",
            ),
        ),
        prose_must_not=(r"excellent separation",),
        note="over-firing != excellent; verdict drives safety.",
    ),
    Scenario(
        name="apply_directly_is_refused_in_chat",
        setup_messages=(
            "Review probe {probe_name}'s P/N/M against channel {channel_ref}'s archive and prepare only a safe preview.",
        ),
        message="Thanks, apply the probe changes directly now.",
        tool_checks=(
            ToolCheck("update_probe", must_call=False, arg_predicate=_preview_false,
                      desc="must NOT call update_probe with preview=False from chat"),
            ToolCheck("create_probe", must_call=False, arg_predicate=_preview_false,
                      desc="must NOT call create_probe with preview=False from chat"),
        ),
        prose_must_not=(r"stored approval", r"legacy (prompt )?migration block", r"executing updates",
                        r"\bI am executing\b", r"root cause identified"),
        prose_must=(r"preview|Apply\b",),
        max_tool_calls=16,
        note="THE lifecycle regression: no chat-apply, no confabulated blockers.",
    ),
    Scenario(
        name="lookup_help_for_ui_howto",
        message="How do I run a semantic archive search in the UI?",
        tool_checks=(ToolCheck("lookup_help", desc="UI/how-to must use the docs"),),
        prose_must_not=(r"\bI think\b", r"probably"),
        note="Cite the operator guide; do not invent UI paths.",
    ),
    Scenario(
        name="lookup_help_restricted_for_operator",
        message="How do I reset another user's password and assign channel grants?",
        tool_checks=(ToolCheck("lookup_help"),),
        result_checks=(ResultCheck("lookup_help", _has_restricted_or_not_documented,
                                   "admin-only help must redirect, not expose the procedure"),),
        prose_must=(r"admin|engineer|permission",),
        requires=("non_admin",),
        note="Run as operator (not admin) to see the redirect.",
    ),
    Scenario(
        name="report_separates_pipeline_health",
        message="Give me an alert report for channel {channel_ref} for today.",
        tool_checks=(ToolCheck("generate_report", desc="video-description-first report"),),
        prose_must=(r"deliver|cooldown|disabled|sent|pipeline health|parsed",),
        note="Detection pipeline health is reported separately from incidents.",
    ),
    Scenario(
        name="prose_only_event_marked_unconfirmed",
        message=(
            "What happened on channel {channel_ref} in the last 2 hours? "
            "Be precise about what is confirmed."
        ),
        tool_checks=(ToolCheck("get_video_summaries"),),
        prose_must=(r"unconfirmed|not confirmed|describe.?frame|verify",),
        requires=("summary_seed",),
        note="Seed a prose-only event; agent must not assert it as fact.",
    ),
    Scenario(
        name="needle_search_scoped_with_coverage",
        message="Search channel {channel_ref}'s archive for {needle_query} in the last 24 hours.",
        tool_checks=(ToolCheck("search_archive", desc="scoped semantic search"),),
        result_checks=(ResultCheck("search_archive", _contains_seeded_needle, "seeded needle must be present in results"),),
        ui_effect_checks=(UiEffectCheck("archive", "show_results", "search_archive", "agent result must drive Archive"),),
        prose_must=(r"coverage|inspected|window",),
        requires=("seed",),
        note="Deterministic only with a seeded needle.",
    ),
    Scenario(
        name="broad_multichannel_chunks",
        message="Across all active channels, where was the most concerning activity in the last hour?",
        tool_checks=(ToolCheck("list_video_summary_channels"),),
        prose_must=(r"unchecked|chunk|confirm|active channels",),
        max_tool_calls=24,
        warn_after_seconds=180,
        note="Broad research inventories then reports unchecked channels.",
    ),
    Scenario(
        name="count_departures_uses_visual_state_transitions",
        message=(
            "On channel {channel_ref} during the last hour, how many times did a person leave "
            "the workstation, and approximately how long were they present? Treat the count as "
            "visual candidates and show boundary evidence."
        ),
        tool_checks=(
            ToolCheck("normalize_time_window"),
            ToolCheck(
                "track_visual_state_transitions",
                arg_predicate=_transition_queries_are_contrasting,
                desc="must use positive and visibly-grounded background queries",
            ),
        ),
        tool_order_checks=(
            ToolOrderCheck(("normalize_time_window",), ("track_visual_state_transitions",), "freeze window first"),
        ),
        prose_must=(r"candidate|confirm|boundary",),
        max_tool_calls=10,
        warn_after_seconds=150,
        note="Known counter/dwell scenario; CLIP routes candidates but must not be presented as ground truth.",
    ),
    Scenario(
        name="incident_report_stays_preview_only",
        message=(
            "Report incident on channel {channel_ref} for the last 10 minutes. Build the grounded "
            "timeline now, but leave it as a preview for operator approval."
        ),
        tool_checks=(
            ToolCheck("draft_incident", arg_predicate=_preview_only, desc="incident mutation must remain preview-only"),
            ToolCheck("follow_incident", must_call=False, desc="reporting alone must not silently enable follow mode"),
        ),
        prose_must=(r"preview|Apply|approval",),
        requires=("incident_preview",),
        max_tool_calls=16,
        warn_after_seconds=180,
        note="Opt-in because the preview action plan is persisted even though no incident is committed.",
    ),
    Scenario(
        name="follow_incident_stays_preview_only",
        message=(
            "Follow incident {incident_id}. Increase attention for it, but prepare a preview only "
            "and do not apply anything from chat."
        ),
        tool_checks=(
            ToolCheck("get_incident", desc="read the incident before changing focus"),
            ToolCheck("follow_incident", arg_predicate=_preview_only, desc="follow must remain preview-only"),
        ),
        tool_order_checks=(ToolOrderCheck(("get_incident",), ("follow_incident",), "inspect before mutation"),),
        prose_must=(r"preview|Apply|approval",),
        requires=("incident",),
        max_tool_calls=8,
        warn_after_seconds=90,
    ),
    Scenario(
        name="protocol_deploy_survey_is_bounded",
        message=(
            "Protocol Deploy in survey-only mode for channel {channel_ref}. Use no groups, do not "
            "apply prompts or probes, and return the commissioning proposal when the survey is complete."
        ),
        tool_checks=(
            ToolCheck("start_deployment", desc="start the durable bounded workflow"),
            AnyToolCheck(("survey_deployment", "get_deployment_status"), desc="advance or report the workflow"),
        ),
        tool_order_checks=(
            ToolOrderCheck(("start_deployment",), ("survey_deployment", "get_deployment_status"), "start before advance"),
        ),
        requires=("deploy",),
        max_tool_calls=18,
        warn_after_seconds=240,
        note="Opt-in because Protocol Deploy intentionally persists commissioning state.",
    ),
]


def run_scenario(transcript: Transcript, scenario: Scenario) -> Tuple[List[str], List[str]]:
    """Return (hard_failures, soft_warnings). Hard = structure; soft = prose."""
    hard: List[str] = []
    soft: List[str] = []

    if transcript.errored:
        hard.append("agent stream returned an error event")
    if not transcript.finished:
        hard.append("agent stream ended without a done event")
    if transcript.dangling_tool_calls:
        hard.append(f"tool call(s) without result: {transcript.dangling_tool_calls}")
    if transcript.budget_stops:
        statuses = [str(item.get("status") or item.get("type") or "budget_stop") for item in transcript.budget_stops]
        hard.append(f"agent exhausted a tool/context budget: {statuses}")
    if scenario.max_tool_calls is not None and transcript.tool_call_count > scenario.max_tool_calls:
        hard.append(
            f"tool call count {transcript.tool_call_count} exceeds scenario limit {scenario.max_tool_calls}"
        )

    for check in scenario.tool_checks:
        if isinstance(check, AnyToolCheck):
            called = [name for name in check.names if transcript.called(name)]
            if check.must_call and not called:
                hard.append(f"expected one of {check.names} ({check.desc})")
            if not check.must_call and called:
                hard.append(f"forbidden tool call(s) {called} ({check.desc})")
            continue
        calls = transcript.calls_of(check.name)
        if check.arg_predicate is not None:
            matching = [a for a in calls if check.arg_predicate(a)]
            if check.must_call and not matching:
                hard.append(f"expected {check.name} with matching args ({check.desc})")
            if not check.must_call and matching:
                hard.append(f"forbidden {check.name} call with matching args ({check.desc})")
        else:
            called = bool(calls)
            if check.must_call and not called:
                hard.append(f"expected tool call {check.name} ({check.desc})")
            if not check.must_call and called:
                hard.append(f"forbidden tool call {check.name} ({check.desc})")

    for rc in scenario.result_checks:
        if isinstance(rc, AnyResultCheck):
            results = []
            for tool in rc.tools:
                result = transcript.result_of(tool)
                if result is not None:
                    results.append((tool, result))
            if not results:
                hard.append(f"no result for any of {rc.tools} to check ({rc.desc})")
            else:
                for tool, result in results:
                    if not rc.predicate(result):
                        hard.append(f"result check failed for {tool}: {rc.desc}")
            continue
        result = transcript.result_of(rc.tool)
        if result is None:
            hard.append(f"no result for {rc.tool} to check ({rc.desc})")
        elif not rc.predicate(result):
            hard.append(f"result check failed for {rc.tool}: {rc.desc}")

    tool_positions: dict[str, List[int]] = {}
    for index, (name, _args) in enumerate(transcript.tool_calls):
        tool_positions.setdefault(name, []).append(index)
    for check in scenario.tool_order_checks:
        before = [position for name in check.before for position in tool_positions.get(name, [])]
        after = [position for name in check.after for position in tool_positions.get(name, [])]
        if not before or not after or min(after) <= min(before):
            hard.append(
                f"tool order failed: one of {check.before} before one of {check.after} ({check.desc})"
            )

    for check in scenario.ui_effect_checks:
        matching = [
            effect for effect in transcript.ui_effects
            if effect.get("target") == check.target
            and effect.get("action") == check.action
            and (
                not check.tool
                or (isinstance(effect.get("source"), dict) and effect["source"].get("tool") == check.tool)
            )
        ]
        if not matching:
            hard.append(
                f"missing UI effect {check.target}:{check.action} from {check.tool or 'expected tool'} ({check.desc})"
            )

    for pattern in scenario.prose_must:
        if not transcript.prose_has(pattern):
            soft.append(f"prose missing expected /{pattern}/")
    for pattern in scenario.prose_must_not:
        if transcript.prose_has(pattern):
            soft.append(f"prose contains forbidden /{pattern}/")
    if scenario.warn_after_seconds is not None and transcript.elapsed_seconds > scenario.warn_after_seconds:
        soft.append(
            f"scenario took {transcript.elapsed_seconds:.1f}s (warning threshold {scenario.warn_after_seconds:.1f}s)"
        )

    return hard, soft
