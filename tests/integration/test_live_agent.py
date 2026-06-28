"""Live agent acceptance smoke (OPT-IN; never gates the build).

Runs only when EVA_LIVE_BASE_URL is set. Hits the real running EVA instance,
drives the agent over its SSE contract, and asserts STRUCTURE (tool calls +
tool_result fields). LLM prose is reported as warnings, not failures, so model
variability does not flake CI.

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
import unittest

from tests.integration.eva_client import EvaSession
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


@unittest.skipUnless(_BASE, "set EVA_LIVE_BASE_URL to run the live agent smoke")
class LiveAgentSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = EvaSession(_BASE, csrf_cookie=_CSRF, verify_tls=_VERIFY_TLS)
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
        chat_session_id = None
        failures = []
        for scenario in SCENARIOS:
            unmet = [tag for tag in scenario.requires if tag not in _INCLUDE]
            if unmet:
                print(f"SKIP {scenario.name}: requires {unmet} (set EVA_LIVE_INCLUDE)")
                continue
            if "non_admin" in scenario.requires and self._adminish():
                failures.append((
                    scenario.name,
                    ["scenario requires non-admin/operator credentials; current user is admin-capable"],
                ))
                continue
            message = scenario.message.format(
                channel_ref=_CHANNEL_REF,
                needle_query=_NEEDLE_QUERY,
                probe_name=_PROBE_NAME,
            )
            transcript = self.session.ask(message, session_id=chat_session_id)
            chat_session_id = transcript.session_id or chat_session_id
            hard, soft = run_scenario(transcript, scenario)
            for w in soft:
                print(f"WARN  {scenario.name}: {w}")
            if hard:
                failures.append((scenario.name, hard))
                print(f"FAIL  {scenario.name}: {hard}")
            else:
                print(f"OK    {scenario.name}")
        self.assertEqual(failures, [], f"hard structural failures: {failures}")


if __name__ == "__main__":
    unittest.main()
