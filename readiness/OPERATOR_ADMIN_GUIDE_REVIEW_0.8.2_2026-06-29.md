# EVA AI β 0.8.2 - operator/admin guide review, 2026-06-29

Source artifact: `docs/EVAAI_Operator_guide.pdf`, a 0.8.1-era Google Docs export
with field notes from the intern. The PDF is an input artifact and should not be
committed.

## Summary

The current Markdown docs are already ahead of the PDF in the important 0.8.2
areas:

- video-description-first workflow;
- `alert_policy_prompt` separated from `stream_system_prompt`;
- `json_alert_prompt` described as a parser contract, not an operator policy
  field;
- probes/CLIP documented as secondary candidate signals;
- preview-only prompt/probe edits with UI Apply receipts.

The PDF still contained several 0.8.1-era or draft notes that could confuse
operators if copied forward.

## PDF Notes Interpreted

### "dont touch json"

Correct intent. In 0.8.2 wording this should be:

`ALERTS_JSON` / `json_alert_prompt` is the machine-readable output contract. Do
not edit it unless an engineer is intentionally changing parser/schema behavior.
Operator watch criteria belong in **Alert Criteria** (`alert_policy_prompt`).

### "Channel quiet vs blind"

Correct concern. Operators must check stream health/coverage gaps, not infer
"nothing happened" from silence. This is already in operator/observability docs.

### "SESSIONS - new"

The UI has persistent agent sessions. Operator docs now explain when to open a
new session, when to reuse an old one, and how to re-ground a drifting thread.

### "negative example means not having X"

Incorrect and dangerous. CLIP negative prompts must be visible contrast states,
not absence phrases. Operator/admin docs now explicitly say that refusal of
`negative: no weapon` is correct safety behavior.

### Admin section mixed into operator guide

The PDF blended operator and admin material. Current canonical docs keep:

- operator workflow in `docs/operator/operator_guide.md`;
- user/grant/audit/retention/settings administration in
  `docs/admin/admin_guide.md`;
- exact invariants in `docs/00_CANON/`.

## Changes Applied

- `docs/operator/operator_guide.md`
  - added office HTTP vs client HTTPS sign-in note;
  - documented Start/Stop summaries persistence;
  - documented Summary Lens;
  - documented Agent Sessions;
  - clarified Alert Criteria vs L0 prompt vs `ALERTS_JSON`;
  - clarified visible negative prompts and safe refusal;
  - clarified that `NOT SAFE TO APPLY` can be the correct probe-calibration
    result.

- `docs/admin/admin_guide.md`
  - added practical Settings → Users path;
  - clarified operator/admin behavior for password/grant help;
  - added Settings → Audit pointer;
  - added desired-session restore note;
  - added visible-negative prompt rule for probes;
  - added TLS cookie setting note for client deployments.

## Remaining Verification

- Confirm exact admin UI paths for audit and CLI user-management subcommands.
- If a polished shareable PDF is needed, regenerate from current Markdown after
  the Sunday client-prep pass rather than editing the 0.8.1 PDF.
