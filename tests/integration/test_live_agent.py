"""Live agent acceptance smoke (OPT-IN; never gates the build).

Runs only when EVA_LIVE_BASE_URL is set. Hits the real running EVA instance,
drives the agent over its SSE contract, and asserts STRUCTURE (tool calls +
tool_result fields). LLM prose is reported as warnings, not failures, so model
variability does not flake CI. Independent scenarios use independent sessions;
explicit setup_messages cover intentional multi-turn behavior.

Run (dev box, secure mode ON, with an admin account that still goes through
preview/approval — do NOT disable the gates):

    EVA_LIVE_BASE_URL=https://127.0.0.1:5443 \
    EVA_LIVE_USER=admin EVA_LIVE_PASSWORD=... \
    EVA_LIVE_CHANNEL_REF="Zenbook webcam" \
    EVA_LIVE_INCLUDE=seed \
    .venv/bin/pytest -q tests/integration/test_live_agent.py -s

See tests/integration/README.md.
"""
import os
import json
import unittest
from pathlib import Path

from tests.integration.eva_client import EvaSession, Transcript, combine_transcripts
from tests.integration.scenarios import SCENARIOS, run_scenario

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

    def test_scenarios(self) -> None:
        failures = []
        report = {
            "base_url": _BASE,
            "channel_ref": _CHANNEL_REF,
            "model_behavior_contract": "structure-v1",
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
                    turn = self.session.ask(message, session_id=chat_session_id)
                except Exception as exc:  # keep running the matrix after one transport/model failure
                    turn_error = f"{type(exc).__name__}: {exc}"
                    break
                turns.append(turn)
                chat_session_id = turn.session_id or chat_session_id
                if turn.errored or not turn.finished:
                    break

            transcript = combine_transcripts(turns)
            hard, soft = run_scenario(transcript, scenario)
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
                "hard_failures": hard,
                "warnings": soft,
                "answer": transcript.text,
            })
        if _REPORT_PATH:
            path = Path(_REPORT_PATH).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"REPORT {path}")
        self.assertEqual(failures, [], f"hard structural failures: {failures}")


if __name__ == "__main__":
    unittest.main()
