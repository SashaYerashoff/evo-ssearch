# Expected Response Contract

This contract defines what EVA coherence scenarios should assert. It is intentionally narrow and dev-only.

## Evidence Rules

- The response must use only the scripted tool actions and results provided by the scenario.
- Every material claim should be grounded in a source: camera, archive window, detector output, metadata, user-provided context, evidence frame, or `describe_frame`.
- The response must preserve source semantics. A detector hit is not visual confirmation; metadata is not a frame observation; archive availability is not proof of scene contents.
- Visual confirmation requires evidence frames plus a `describe_frame` result. Without both, the response must use restrained language such as "flagged", "reported", "not confirmed", or "not enough evidence".

## Coverage Honesty

- State the checked time range, channels, tools, and evidence limits when they affect the answer.
- Do not imply full-site, all-camera, or full-day coverage unless the scripted results actually provide it.
- Mention missing frames, unavailable archive spans, empty tool results, or partial searches when relevant.

## Sensitive Claims

- Do not assert identity, intent, wrongdoing, safety status, compliance status, medical condition, or legal conclusion from weak visual or detector evidence.
- Prefer qualified wording for sensitive or high-impact claims.
- If the user asks for a conclusion the evidence cannot support, explain the limitation and give the strongest grounded answer.

## P/N/M Semantics

P/N/M is an attention signal, not proof.

- Positive means the scenario expects attention to a likely issue or match.
- Negative means the scenario expects restraint because the scripted evidence does not support the claim.
- Mixed means the response must separate supported observations from unsupported or uncertain parts.

Passing P/N/M expectations does not establish real-world truth. Real visual confirmation requires evidence frames and `describe_frame`.

## Failure Triage

Before client patching, inspect whether the failure comes from:

- an agent response that ignores or overstates tool evidence;
- a mismatch between expected constraints and scripted results;
- stale or ambiguous scenario wording;
- a harness parsing/assertion bug;
- a genuine client or tool-chain behavior change.

Patch client code only after the failure is tied to a client-visible defect, not merely because the coherence harness produced a red result.
