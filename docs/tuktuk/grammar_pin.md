# tuktuk Grammar Pin

Status: pinned design gate for EVA agent tool and skill changes.
Pinned on: 2026-07-19.

The tuktuk grammar is the current reflex contract for EVA AI. Any change to the
agent tool surface, compact result surface, intent exposure, write/approval flow,
or `skills/*/SKILL.md` must be checked against this contract before it is treated
as done.

## Pinned Source

Pinned source commit:
`0c97a830f06bfee766bf69cc3327222758786be1`

The pinned artifacts are checked into this branch under `docs/tuktuk/`.

| Artifact | sha256 |
|---|---|
| `docs/tuktuk/grammar.yaml` | `c165f597bb8f30af8cefd7c2f6456b25ac16a1d878e25aba117bae7d7c069b73` |
| `docs/tuktuk/grammar.md` | `a4c44f4046b157dfc24d545abe94eb95b96f46982bda3a22ffc2f448a0932193` |
| `docs/tuktuk/homeostate.yaml` | `5c53bdeda5806acd5d03da421fba5a4726bf635724d8e4475b9e9d473fa628e1` |
| `docs/tuktuk/homeostatic_doc.md` | `4d65142e9a72187e6ed994c7c4de8eaefa3f0c64cb45ca980b893e4f656cc41a` |
| `docs/tuktuk/tool_surface.json` | `c9afe58795adcbae0c30c0c0e8569b3db7718089077a940256c81184a4728e33` |
| `docs/tuktuk/tool_inventory.md` | `dee0b3747d050e5791c0d809daf37ace2152acff2f44f997b7577935d69d3081` |

## Design Gate

When adding or changing a tool or skill, answer these checks explicitly:

1. Which grammar block owns the change?
   `W`, `C`, `MAP`, `RANK`, `DRILL`, `AGG`, `CAL`, `MUT`, or `TERM`.
2. Are model-emitted arguments copied only from one of the three legal sources:
   operator text, scratchpad/prior compact result, or a closed enum?
3. Does the compact result expose only stable keys that tuktuk can learn from?
   If a devalidation relies on a symptom, that symptom must be visible in the
   compact result or in the homeostatic document.
4. Does the change preserve the harness-owned boundary?
   Window resolution, channel resolution, continuation/chunking, result envelope,
   alias normalization, and homeostatic context selection stay outside the model.
5. Does the change affect a gate?
   Visual confirmation, coverage honesty, trusted receipts, hidden-state
   phrasing, source honesty, or low-confidence escalation must remain mechanically
   checkable.
6. Does the change introduce a new devalidation or alter an existing d01-d15
   correction path?

## Conflict Policy

If a tool or skill does not fit the pinned grammar, raise it out loud in the
work log/review summary. Do not silently adjust the grammar, tool surface, or
skill text to make the conflict disappear.

Record unresolved conflicts in `docs/tuktuk/grammar_review_questions.md`.
Grammar revisions should update this pin, the artifact hashes, and the gym
corpus imports together.

## Current Known Conflict

None. The previously recorded write-tool conflict was an artifact-extraction
bug, resolved 2026-07-19 — see `grammar_review_questions.md` (Resolved) for
the actual `MUT` envelope (`diff` + `action_plan.plan_id`, raw `approval`
stripped) and the intended representation of d15.
