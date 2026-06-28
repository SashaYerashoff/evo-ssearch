# Live agent integration smoke

Acceptance smoke that drives the **real running agent** over its HTTP/SSE
contract and asserts **structure** (tool calls + `tool_result` fields), not LLM
prose. Golden/unit tests gate the build; this is acceptance — opt-in, never in
the default suite so model variability can't flake CI.

## Principle
- **Don't disable the secure gates.** Use an **admin** account, but keep
  `EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true`, preview-only, and the
  approval/UI-Apply route enabled — those gates are *under test* (the chat logs
  failed precisely there). Admin removes only *channel-scope* friction, not
  *action* gates.
- **Assert structure, not mood.** Tool-call sequence, `safe_to_apply`,
  `calibration_status`, `recommended_probe_args`, `restricted_matches`,
  `delivery_status`, and the action receipt are deterministic given the tools.
  Prose checks are *soft warnings*.

## Setup (dev box)
1. Service running (e.g. `https://127.0.0.1:5443`), secure mode on.
2. Admin user: `python scripts/bootstrap_admin.py` (see admin_guide).
3. For the redirect scenario, also create an **operator** (non-admin) account:
   ```bash
   EVA_LIVE_OPERATOR_PASSWORD='...' \
   .venv/bin/python scripts/bootstrap_live_smoke_operator.py --channel-id 112 --set-password
   ```
4. For deterministic needle/contamination scenarios, **seed** known
   summary/archive/probe fixtures first (a known prose-only event; a probe with
   known archive frames; a planted searchable incident).

## Run
```bash
EVA_LIVE_BASE_URL=https://127.0.0.1:5443 \
EVA_LIVE_USER=admin EVA_LIVE_PASSWORD='...' \
EVA_LIVE_CHANNEL_REF='Zenbook webcam' \
EVA_LIVE_NEEDLE_QUERY='person lying on the ground at night' \
EVA_LIVE_PROBE_NAME='smoke: thumbs up gesture' \
EVA_LIVE_INCLUDE=seed \
.venv/bin/pytest -q tests/integration/test_live_agent.py -s
```
- `EVA_LIVE_INCLUDE` lists prerequisite tags you've set up (`seed`, `non_admin`).
  Scenarios needing an unmet tag are skipped (not failed).
- Do not combine `EVA_LIVE_USER=admin` with `EVA_LIVE_INCLUDE=non_admin`.
  Run restricted-help coverage as a separate pass with an operator account:
  ```bash
  EVA_LIVE_BASE_URL=https://127.0.0.1:5443 \
  EVA_LIVE_USER=operator EVA_LIVE_PASSWORD='...' \
  EVA_LIVE_CHANNEL_REF='Zenbook webcam' \
  EVA_LIVE_INCLUDE=non_admin \
  .venv/bin/pytest -q tests/integration/test_live_agent.py -s
  ```
- `EVA_LIVE_CHANNEL_REF` is the channel name or ID used in channel-scoped prompts
  (default: `the active video-description channel`, which relies on the agent to resolve it).
- `EVA_LIVE_NEEDLE_QUERY` is the seeded archive-search query.
- `EVA_LIVE_PROBE_NAME` is the seeded/configured probe used by the calibration scenario.
- `EVA_LIVE_CSRF_COOKIE` overrides the CSRF cookie name (default `eva_csrf`).
- `EVA_LIVE_VERIFY_TLS=1` enables TLS verification. By default the smoke accepts
  the local self-signed dev certificate.

## Lifecycle (preview → Apply → receipt)
The UI "Apply" is `POST /agent/action-plans/<plan_id>/execute`. A preview
`tool_result` carries `approval.plan_id`. To test the full commit loop from a
script:
```python
t = session.ask("prepare probe preview ...")
plan_id = t.approval_plan_ids()[0]
receipt = session.apply_plan(plan_id, session_id=t.session_id)   # simulates UI Apply
t2 = session.ask("did that apply?", session_id=t.session_id)     # must report applied (with receipt)
```

## What is / isn't covered here
- **Covered (live):** tool wiring, SSE streaming, auth/CSRF, the structural
  contracts above, the preview/apply lifecycle, broad-channel chunking.
- **Not here (build-gating golden/unit):** calibration verdict math, negation
  rejection, provenance/`delivery_status`, transition debounce, the status
  digest — those are deterministic and live in the normal pytest suite.
- **Browser-only (manual / Playwright later):** rendered cards, role-based UI
  hiding, the physical Apply button. The Apply *receipt* itself is API-testable
  (above), so the browser layer is a thin visual smoke.

`test_sse_parser.py` here is deterministic and **does** run in the normal suite —
it regression-covers the harness (parser + scenario checker) without a live service.
