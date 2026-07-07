# EVA AI β 0.8.2 - office manual test triage, 2026-06-29

Source artifact: `tests/Copy of eva-ai-manual-test-082.pdf` from the office
demo manual run by the field intern on 2026-06-28. The PDF is an input artifact
and should not be committed.

## Executive Result

The office deployment itself is healthy:

- `/health` returned `β 0.8.2`.
- `/ready` returned `status=ready`.
- PostgreSQL schema head is `20260614_0006`.
- Luxriot is reachable.
- Agent and VLM LM Studio profiles are reachable.
- Four video-description streams were restored.

The remaining issues are mostly test clarity and operational configuration, not
release-blocking code failures.

## Real Findings

### HTTP vs HTTPS confusion

The office service listens on `http://0.0.0.0:5000`, while the first runbook and
manual test prompt still used the local dev URL `https://127.0.0.1:5443`.

Impact: repeated WARN notes like "one error of UI on top of http".

Status: runbook and manual scenario were updated to default to
`http://127.0.0.1:5000` for office demo, while keeping TLS/5443 as a dev/client
variant.

Client note: behind TLS, set `EVOSSEARCH_AUTH_COOKIE_SECURE=true`.

### Evidence is sometimes textual only

Manual test 4.1 recorded that channel 135 had textual evidence but weak image
legitimacy.

Impact: acceptable WARN if the agent clearly says images are unavailable; FAIL
only if it claims visual confirmation without frames.

Follow-up: during the next office pass, capture one concrete channel/time where
visual evidence is missing and save the agent trace. This may be either normal
coverage limitation or a thumbnail/evidence-link gap.

### Report pipeline health wording is unclear

Manual test 5.1 was PASS/WARN with note `parser/delivery health - ?`.

Impact: report content appears present but the operator may not understand what
parser/delivery health means.

Follow-up: consider adding a short UI/report label explanation:

- detected = VLM/structured event found;
- parsed = event extracted from model output;
- delivered = bookmark/event sent;
- cooldown/disabled/failed = event detected but not delivered as a bookmark.

### Probe control was safer than the final summary suggests

The PDF final summary marks `7.1-7.4` as FAIL. The screenshots suggest a more
specific interpretation:

- `7.1` returned "no secondary CLIP probe recommended" for a routine police
  vehicle alert. That is acceptable if no distinct actionable class exists.
- `7.2` returned `NOT SAFE TO APPLY`, `weak_separation`, raised threshold, and
  recommended manual frame review. That matches the desired safety behavior.
- `7.3` rejected `negative: no weapon`, explained that CLIP negatives must be
  visible alternatives, and asked for visible negative prompts. That is PASS,
  not FAIL.
- `7.4` asked whether a probe was applied after an unsafe/rejected preview path.
  The agent correctly answered that no change was applied and that UI Apply is
  required. The test should be N/A unless a safe preview exists.

Status: manual scenario was updated to make these pass conditions explicit.

Remaining product risk: operators still need clearer wording that "not safe to
apply" is a successful safety outcome, not a failure of the system.

### RBAC result needs one clean rerun

The RBAC screenshot appears to redirect instead of exposing admin steps, which is
the expected behavior for an operator. The tester note says a previous run gave
a "completely negative answer".

Impact: not a confirmed bug. It may be role/context dependent.

Follow-up: rerun once with a known non-admin operator account and once with an
admin/engineer account. Expected:

- operator: redirect and required permission, no procedure steps;
- admin/engineer: sanitized procedure help is allowed.

## Current Priority

For the Sunday/client-prep pass:

1. Repeat evidence test 4.1 on a channel/time with known VLM alert frames.
2. Repeat RBAC with explicit account roles.
3. Repeat probe lifecycle with a safe preview, then UI Apply, then receipt check.
4. Do not treat "probe rejected as unsafe" as a failure; treat it as PASS/WARN.

## Not Release Blocking

- `pytest` absent in office production venv. The service starts via
  `.venv/bin/gunicorn`; runtime `py_compile`, `/health`, and `/ready` passed.
- HTTP on office demo. This is a deployment-mode WARN, not app failure.
- Inference slowness. 5-10 minute complex agent turns remain expected on demo
  hardware.
