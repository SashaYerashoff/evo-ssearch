"""Live agent acceptance smoke (OPT-IN; never gates the build).

Runs only when EVA_LIVE_BASE_URL is set. Hits the real running EVA instance,
drives the agent over its SSE contract, and asserts STRUCTURE (tool calls +
tool_result fields). LLM prose is reported as warnings, not failures, so model
variability does not flake CI. Independent scenarios use independent sessions;
explicit setup_messages cover intentional multi-turn behavior.

Run (dev box, secure mode ON, with a scoped engineer or admin account that still
goes through preview/approval — do NOT disable the gates):

    EVA_LIVE_BASE_URL=https://127.0.0.1:5443 \
    EVA_LIVE_USER=admin EVA_LIVE_PASSWORD=... \
    EVA_LIVE_CHANNEL_REF="Zenbook webcam" \
    EVA_LIVE_INCLUDE=seed,probe_apply,prompt_preview \
    .venv/bin/pytest -q tests/integration/test_live_agent.py -s

See tests/integration/README.md.
"""
import os
import json
import statistics
import time
import unittest
from pathlib import Path

from tests.integration.eva_client import EvaSession, Transcript, combine_transcripts
from tests.integration.scenarios import (
    SCENARIOS,
    generation_quality,
    run_scenario,
    tool_efficiency,
)

_BASE = os.getenv("EVA_LIVE_BASE_URL", "").strip()
_USER = os.getenv("EVA_LIVE_USER", "").strip()
_PASSWORD = os.getenv("EVA_LIVE_PASSWORD", "")
_CSRF = os.getenv("EVA_LIVE_CSRF_COOKIE", "eva_csrf").strip() or "eva_csrf"
_CHANNEL_REF = os.getenv("EVA_LIVE_CHANNEL_REF", "the active video-description channel").strip()
_NEEDLE_QUERY = os.getenv("EVA_LIVE_NEEDLE_QUERY", "seeded test incident").strip()
_PROBE_NAME = os.getenv("EVA_LIVE_PROBE_NAME", "the seeded probe").strip()
_VERIFY_TLS = os.getenv("EVA_LIVE_VERIFY_TLS", "").strip().lower() in {"1", "true", "yes", "on"}
# tags whose prerequisites are set up in this environment (seed data, operator acct, ...)
_INCLUDE = {t.strip() for t in os.getenv("EVA_LIVE_INCLUDE", "").split(",") if t.strip()}
_ONLY = {name.strip() for name in os.getenv("EVA_LIVE_SCENARIOS", "").split(",") if name.strip()}
_INCIDENT_ID = os.getenv("EVA_LIVE_INCIDENT_ID", "the configured test incident").strip()
_REPORT_PATH = os.getenv("EVA_LIVE_REPORT_PATH", "").strip()
_OPERATOR_MODE = os.getenv("EVA_LIVE_OPERATOR_MODE", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
try:
    _TELEMETRY_INTERVAL = max(
        0.1,
        min(5.0, float(os.getenv("EVA_LIVE_TELEMETRY_INTERVAL", "0.25"))),
    )
except ValueError:
    _TELEMETRY_INTERVAL = 0.25
try:
    _TIMEOUT = max(10.0, float(os.getenv("EVA_LIVE_TIMEOUT", "300")))
except ValueError:
    _TIMEOUT = 300.0


@unittest.skipUnless(_BASE, "set EVA_LIVE_BASE_URL to run the live agent smoke")
class LiveAgentSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = EvaSession(_BASE, csrf_cookie=_CSRF, verify_tls=_VERIFY_TLS, timeout=_TIMEOUT)
        cls.user_info = {}
        if _USER:
            cls.session.login(_USER, _PASSWORD)
            try:
                cls.user_info = cls.session.whoami()
            except Exception:
                cls.user_info = {}

    @classmethod
    def _adminish(cls) -> bool:
        user = cls.user_info.get("user") if isinstance(cls.user_info, dict) else {}
        roles = {str(item).lower() for item in (user.get("roles") or [])} if isinstance(user, dict) else set()
        perms = {str(item).lower() for item in (user.get("permissions") or [])} if isinstance(user, dict) else set()
        return "admin" in roles or "users:manage" in perms or "settings:manage" in perms

    @classmethod
    def _runtime_snapshot(cls) -> dict:
        snapshot: dict = {"captured_at_ms": int(time.time() * 1000)}
        try:
            snapshot["agent_config"] = cls.session.get_json("/agent/config")
        except Exception as exc:
            snapshot["agent_config_error"] = f"{type(exc).__name__}: {exc}"
        try:
            snapshot["lm_admission"] = cls.session.get_json("/lm/admission")
        except Exception as exc:
            snapshot["lm_admission_error"] = f"{type(exc).__name__}: {exc}"
        try:
            ready = cls.session.get_json("/ready")
            checks = ready.get("checks") if isinstance(ready.get("checks"), dict) else {}
            attention = checks.get("attention") if isinstance(checks.get("attention"), dict) else {}
            microbatch = attention.get("clip_microbatcher") if isinstance(attention.get("clip_microbatcher"), dict) else {}
            snapshot["ready"] = {
                "status": ready.get("status"),
                "version": ready.get("version"),
                "postgresql": checks.get("postgresql"),
                "lm_profiles": checks.get("lm_profiles"),
                "inference_queue": checks.get("inference_queue"),
                "capture_runtime": attention.get("capture_runtime"),
                "fast_vlm_alerts": attention.get("fast_vlm_alerts"),
                "realtime_probe_bookmarks": attention.get("realtime_probe_bookmarks"),
                "clip_microbatcher": {
                    key: microbatch.get(key)
                    for key in (
                        "queue_depth",
                        "inflight",
                        "recent",
                        "counters",
                        "last_error",
                    )
                    if key in microbatch
                },
            }
        except Exception as exc:
            snapshot["ready_error"] = f"{type(exc).__name__}: {exc}"
        return snapshot

    @classmethod
    def _probe_apply_roundtrip(cls, transcript: Transcript) -> tuple[list[str], dict]:
        """Apply an exact safe draft, verify persistence, then remove only it."""

        failures: list[str] = []
        metrics: dict = {
            "workflow": "probe_apply_roundtrip",
            "apply_attempted": False,
            "persisted": False,
            "cleanup_attempted": False,
            "cleanup_succeeded": False,
        }

        def exact_safe_draft(args: dict) -> bool:
            positives = [str(value).strip().lower() for value in args.get("positives") or []]
            negatives = [str(value).strip().lower() for value in args.get("negatives") or []]
            try:
                floor_matches = abs(float(args.get("pos_floor")) - 0.05) < 1e-6
                margin_matches = abs(float(args.get("margin_thr")) - 0.01) < 1e-6
            except (TypeError, ValueError):
                return False
            return bool(
                str(args.get("name") or "").strip().lower()
                == "agent acceptance headphones"
                and positives == ["person wearing headphones"]
                and negatives == ["person at desk, bare ears visible"]
                and floor_matches
                and margin_matches
                and args.get("preview") is not False
                and args.get("enabled") is False
                and args.get("bookmark_enabled") is False
            )

        exact_calls = [
            args
            for args in transcript.calls_of("create_probe")
            if isinstance(args, dict) and exact_safe_draft(args)
        ]
        plan_ids = transcript.approval_plan_ids_for("create_probe")
        if len(exact_calls) != 1:
            failures.append(
                f"refusing Apply: expected one exact safe create_probe draft, got {len(exact_calls)}"
            )
            return failures, metrics
        if len(plan_ids) != 1:
            failures.append(
                f"expected one create_probe approval plan, got {len(plan_ids)}"
            )
            return failures, metrics

        # A stale artifact means the previous acceptance run did not clean up.
        # Do not update or delete it implicitly: stop before applying anything.
        before = cls.session.get_json("/probes/list")
        collisions = [
            row for row in (before.get("probes") or [])
            if isinstance(row, dict)
            and str(row.get("name") or "").strip().lower()
            == "agent acceptance headphones"
        ]
        if collisions:
            failures.append(
                "refusing Apply: a pre-existing 'Agent acceptance headphones' probe must be reviewed manually"
            )
            metrics["preexisting_probe_ids"] = [str(row.get("id") or "") for row in collisions]
            return failures, metrics

        started = time.monotonic()
        metrics["apply_attempted"] = True
        applied = cls.session.apply_plan(plan_ids[0], session_id=transcript.session_id)
        metrics["apply_elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
        metrics["apply_http_status"] = applied.get("status")
        metrics["apply_success"] = applied.get("success") is True
        result = applied.get("result") if isinstance(applied.get("result"), dict) else {}
        receipt = result.get("action_receipt") if isinstance(result.get("action_receipt"), dict) else {}
        probe_id = str(
            result.get("probe_id")
            or receipt.get("probe_id")
            or ""
        ).strip()
        metrics["probe_id"] = probe_id or None
        committed_effects = applied.get("ui_effects") if isinstance(applied.get("ui_effects"), list) else []
        metrics["committed_ui_effects"] = [
            {
                "target": effect.get("target"),
                "action": effect.get("action"),
                "tool": (effect.get("source") or {}).get("tool")
                if isinstance(effect.get("source"), dict) else None,
            }
            for effect in committed_effects
            if isinstance(effect, dict)
        ]

        try:
            if applied.get("status") != 200 or applied.get("success") is not True:
                failures.append(f"Apply failed: {applied}")
                return failures, metrics
            if str(result.get("status") or "").lower() != "applied":
                failures.append(f"Apply result is not applied: {result.get('status')!r}")
            if str(receipt.get("tool") or "") != "create_probe":
                failures.append("Apply receipt does not attest create_probe")
            if str(receipt.get("status") or "").lower() != "applied":
                failures.append("Apply receipt is missing applied status")
            if not any(
                isinstance(effect, dict)
                and effect.get("target") == "probes"
                and effect.get("action") == "refresh"
                and isinstance(effect.get("source"), dict)
                and effect["source"].get("tool") == "create_probe"
                for effect in committed_effects
            ):
                failures.append("committed create_probe did not emit probes:refresh")
            if not probe_id:
                failures.append("Apply receipt did not expose the created probe id")
                return failures, metrics

            board = cls.session.get_json("/probes/list")
            persisted = next(
                (
                    row for row in (board.get("probes") or [])
                    if isinstance(row, dict) and str(row.get("id") or "") == probe_id
                ),
                None,
            )
            if persisted is None:
                failures.append(f"created probe {probe_id} is absent from /probes/list")
                return failures, metrics
            metrics["persisted"] = True
            expected = {
                "name": "Agent acceptance headphones",
                "positives": ["person wearing headphones"],
                "negatives": ["person at desk, bare ears visible"],
                "pos_floor": 0.05,
                "margin": 0.01,
                "bookmark": False,
                "enabled": False,
            }
            for key, expected_value in expected.items():
                actual = persisted.get(key)
                if isinstance(expected_value, float):
                    try:
                        matches = abs(float(actual) - expected_value) < 1e-6
                    except (TypeError, ValueError):
                        matches = False
                else:
                    matches = actual == expected_value
                if not matches:
                    failures.append(
                        f"persisted probe field {key!r}: expected {expected_value!r}, got {actual!r}"
                    )
        finally:
            if probe_id:
                metrics["cleanup_attempted"] = True
                cleanup = cls.session.post_json("/probes/delete", body={"id": probe_id})
                metrics["cleanup_http_status"] = cleanup.get("status")
                metrics["cleanup_succeeded"] = (
                    cleanup.get("status") == 200 and cleanup.get("success") is True
                )
                if not metrics["cleanup_succeeded"]:
                    failures.append(
                        f"cleanup failed for acceptance probe {probe_id}: {cleanup}"
                    )
        return failures, metrics

    @classmethod
    def _run_workflow(cls, scenario, transcript: Transcript) -> tuple[list[str], dict]:
        if scenario.workflow == "probe_apply_roundtrip":
            return cls._probe_apply_roundtrip(transcript)
        return [], {}

    @staticmethod
    def _aggregate_report(rows: list[dict]) -> dict:
        executed = [row for row in rows if row.get("status") != "skipped"]
        latencies = sorted(
            float(row.get("elapsed_seconds") or 0.0)
            for row in executed
        )

        def percentile(values: list[float], q: float) -> float | None:
            if not values:
                return None
            if len(values) == 1:
                return round(values[0], 3)
            position = (len(values) - 1) * q
            lower = int(position)
            upper = min(len(values) - 1, lower + 1)
            fraction = position - lower
            return round(
                values[lower] + (values[upper] - values[lower]) * fraction,
                3,
            )

        quality_scores = [
            float((row.get("generation_quality") or {}).get("score") or 0.0)
            for row in executed
        ]
        efficiency_scores = [
            float((row.get("tool_efficiency") or {}).get("score") or 0.0)
            for row in executed
        ]
        admissions = [
            (row.get("performance") or {}).get("lm_admission") or {}
            for row in executed
        ]
        return {
            "executed": len(executed),
            "passed": sum(row.get("status") == "passed" for row in executed),
            "failed": sum(row.get("status") == "failed" for row in executed),
            "skipped": sum(row.get("status") == "skipped" for row in rows),
            "latency_seconds": {
                "min": round(min(latencies), 3) if latencies else None,
                "median": round(statistics.median(latencies), 3) if latencies else None,
                "p95": percentile(latencies, 0.95),
                "max": round(max(latencies), 3) if latencies else None,
            },
            "generation_quality_average": round(
                statistics.mean(quality_scores), 2,
            ) if quality_scores else None,
            "tool_efficiency_average": round(
                statistics.mean(efficiency_scores), 2,
            ) if efficiency_scores else None,
            "tool_calls_total": sum(int(row.get("tool_call_count") or 0) for row in executed),
            "agent_lm_admissions_total": sum(
                int(item.get("agent_admissions") or 0) for item in admissions
            ),
            "agent_lm_wait_ms_estimate_total": round(sum(
                float(item.get("agent_wait_ms_estimate") or 0.0) for item in admissions
            ), 3),
            "max_sampled_lm_queue": max(
                (int(item.get("max_queued") or 0) for item in admissions),
                default=0,
            ),
            "max_sampled_lm_queue_age_sec": max(
                (float(item.get("max_oldest_queue_age_sec") or 0.0) for item in admissions),
                default=0.0,
            ),
        }

    @staticmethod
    def _markdown_report(report: dict) -> str:
        summary = report.get("summary") or {}
        runtime = report.get("runtime_before") or {}
        agent_config = runtime.get("agent_config") if isinstance(runtime.get("agent_config"), dict) else {}
        latency = summary.get("latency_seconds") or {}
        lines = [
            "# EVA supervised agent acceptance",
            "",
            f"- Contract: `{report.get('model_behavior_contract')}`",
            f"- Base URL: `{report.get('base_url')}`",
            f"- Agent model: `{agent_config.get('resolved_model') or agent_config.get('default_resolved_model') or agent_config.get('model') or agent_config.get('default_model') or 'unknown'}`",
            f"- Operator mode: `{report.get('operator_mode')}`",
            f"- Passed: **{summary.get('passed', 0)}/{summary.get('executed', 0)}**; skipped: {summary.get('skipped', 0)}",
            f"- Latency p50 / p95: **{latency.get('median')} s / {latency.get('p95')} s**",
            f"- Generation quality / tool efficiency: **{summary.get('generation_quality_average')} / {summary.get('tool_efficiency_average')}**",
            f"- Agent LM admissions / max sampled queue: **{summary.get('agent_lm_admissions_total')} / {summary.get('max_sampled_lm_queue')}**",
            "",
            "| Scenario | Status | Seconds | Tools | LM calls | Queue | Quality | Efficiency |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in report.get("scenarios") or []:
            if row.get("status") == "skipped":
                lines.append(
                    f"| `{row.get('name')}` | skipped | — | — | — | — | — | — |"
                )
                continue
            performance = row.get("performance") or {}
            admission = performance.get("lm_admission") or {}
            quality = row.get("generation_quality") or {}
            efficiency = row.get("tool_efficiency") or {}
            lines.append(
                "| `{name}` | {status} | {elapsed:.3f} | {tools} | {lm} | {queue} | {quality} | {efficiency} |".format(
                    name=row.get("name"),
                    status=row.get("status"),
                    elapsed=float(row.get("elapsed_seconds") or 0.0),
                    tools=int(row.get("tool_call_count") or 0),
                    lm=int(admission.get("agent_admissions") or 0),
                    queue=int(admission.get("max_queued") or 0),
                    quality=quality.get("score"),
                    efficiency=efficiency.get("score"),
                )
            )
        lines.extend(["", "## Findings", ""])
        findings = 0
        for row in report.get("scenarios") or []:
            failures = row.get("hard_failures") or []
            warnings = row.get("warnings") or []
            if not failures and not warnings:
                continue
            findings += 1
            lines.append(f"### `{row.get('name')}`")
            lines.append("")
            for failure in failures:
                lines.append(f"- FAIL: {failure}")
            for warning in warnings:
                lines.append(f"- WARN: {warning}")
            lines.append("")
        if findings == 0:
            lines.append("No structural failures or generation warnings.")
            lines.append("")
        lines.extend([
            "## Interpretation",
            "",
            "Structural failures are contract failures. Quality and efficiency scores are deterministic rubric signals for model comparison; they are not proof of scene truth. Queue values are sampled during each turn and may miss spikes shorter than the configured sampling interval.",
            "",
        ])
        return "\n".join(lines)

    def test_scenarios(self) -> None:
        failures = []
        report = {
            "base_url": _BASE,
            "channel_ref": _CHANNEL_REF,
            "started_at_ms": int(time.time() * 1000),
            "model_behavior_contract": "supervised-agent-acceptance-v2",
            "operator_mode": _OPERATOR_MODE,
            "telemetry_interval_sec": _TELEMETRY_INTERVAL,
            "runtime_before": self._runtime_snapshot(),
            "scenarios": [],
        }
        for scenario in SCENARIOS:
            if _ONLY and scenario.name not in _ONLY:
                continue
            unmet = [tag for tag in scenario.requires if tag not in _INCLUDE]
            if unmet:
                print(f"SKIP {scenario.name}: requires {unmet} (set EVA_LIVE_INCLUDE)")
                report["scenarios"].append({"name": scenario.name, "status": "skipped", "requires": unmet})
                continue
            if "non_admin" in scenario.requires and self._adminish():
                failures.append((
                    scenario.name,
                    ["scenario requires non-admin/operator credentials; current user is admin-capable"],
                ))
                continue
            values = {
                "channel_ref": _CHANNEL_REF,
                "needle_query": _NEEDLE_QUERY,
                "probe_name": _PROBE_NAME,
                "incident_id": _INCIDENT_ID,
            }
            # A fresh session per scenario prevents unrelated earlier prompts and
            # tool results from contaminating the behavior under test. Workflows
            # that intentionally need history declare setup_messages explicitly.
            chat_session_id = None
            turns: list[Transcript] = []
            turn_error = ""
            for raw_message in (*scenario.setup_messages, scenario.message):
                message = raw_message.format(**values)
                try:
                    turn = self.session.ask(
                        message,
                        session_id=chat_session_id,
                        operator_mode=_OPERATOR_MODE,
                        console_context={"version": 1, "section": "home"},
                        telemetry_interval_sec=_TELEMETRY_INTERVAL,
                    )
                except Exception as exc:  # keep running the matrix after one transport/model failure
                    turn_error = f"{type(exc).__name__}: {exc}"
                    break
                turns.append(turn)
                chat_session_id = turn.session_id or chat_session_id
                if turn.errored or not turn.finished:
                    break

            transcript = combine_transcripts(turns)
            hard, soft = run_scenario(transcript, scenario)
            workflow_failures: list[str] = []
            workflow_metrics: dict = {}
            if not turn_error and transcript.finished and scenario.workflow:
                try:
                    workflow_failures, workflow_metrics = self._run_workflow(
                        scenario,
                        transcript,
                    )
                except Exception as exc:
                    workflow_failures = [
                        f"workflow {scenario.workflow} failed: {type(exc).__name__}: {exc}"
                    ]
                hard.extend(workflow_failures)
            quality = generation_quality(transcript, scenario)
            efficiency = tool_efficiency(transcript, scenario)
            if turn_error:
                hard.insert(0, f"live request failed: {turn_error}")
            if len(turns) != len(scenario.setup_messages) + 1:
                hard.append(
                    f"scenario stopped after {len(turns)} of {len(scenario.setup_messages) + 1} turn(s)"
                )
            for w in soft:
                print(f"WARN  {scenario.name}: {w}")
            if hard:
                failures.append((scenario.name, hard))
                print(f"FAIL  {scenario.name}: {hard}")
            else:
                print(f"OK    {scenario.name}")
            report["scenarios"].append({
                "name": scenario.name,
                "status": "failed" if hard else "passed",
                "turns": len(turns),
                "elapsed_seconds": round(transcript.elapsed_seconds, 3),
                "tool_call_count": transcript.tool_call_count,
                "tool_calls": [name for name, _args in transcript.tool_calls],
                "tool_trace": transcript.tool_trace,
                "performance": transcript.performance_metrics,
                "turn_performance": [turn.performance_metrics for turn in turns],
                "generation_quality": quality,
                "tool_efficiency": efficiency,
                "context_metrics": transcript.context_metrics,
                "budget_stops": transcript.budget_stops,
                "ui_effects": [
                    {
                        "target": effect.get("target"),
                        "action": effect.get("action"),
                        "tool": (effect.get("source") or {}).get("tool")
                        if isinstance(effect.get("source"), dict) else None,
                    }
                    for effect in transcript.ui_effects
                ],
                "workflow": workflow_metrics,
                "hard_failures": hard,
                "warnings": soft,
                "answer": transcript.text,
            })
        report["runtime_after"] = self._runtime_snapshot()
        report["summary"] = self._aggregate_report(report["scenarios"])
        report["finished_at_ms"] = int(time.time() * 1000)
        if _REPORT_PATH:
            path = Path(_REPORT_PATH).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"REPORT {path}")
            markdown_path = path.with_suffix(".md")
            markdown_path.write_text(self._markdown_report(report), encoding="utf-8")
            print(f"REPORT {markdown_path}")
        summary = report["summary"]
        print(
            "SUMMARY "
            f"passed={summary['passed']}/{summary['executed']} "
            f"quality={summary['generation_quality_average']} "
            f"efficiency={summary['tool_efficiency_average']} "
            f"latency_p50={summary['latency_seconds']['median']}s "
            f"latency_p95={summary['latency_seconds']['p95']}s "
            f"lm_calls={summary['agent_lm_admissions_total']} "
            f"max_queue={summary['max_sampled_lm_queue']}"
        )
        self.assertEqual(failures, [], f"hard structural failures: {failures}")


if __name__ == "__main__":
    unittest.main()
