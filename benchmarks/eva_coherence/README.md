# EVA Coherence Benchmark

This is a dev-only benchmark harness for checking whether EVA agent responses stay coherent with the tool chain. It is not a customer demo, product certification, or proof of detection quality.

The harness exists to exercise:

- tool-chain coherence: responses must match the scripted tool calls and results;
- coverage honesty: the agent must state what was and was not checked;
- evidence grounding: claims must be tied to available evidence, timestamps, and sources;
- source semantics: live camera, archive, detector, metadata, and manual context must not be treated as interchangeable;
- sensitive-claim restraint: identity, intent, compliance, safety, medical, legal, or security claims must be qualified unless directly evidenced.

## Scenario Pattern

Each scenario should contain three parts:

1. User prompt: the question or instruction given to the agent.
2. Scripted tool actions/results: deterministic tool calls and returned observations used as the only evidence base.
3. Expected constraints: response requirements, forbidden claims, required caveats, and source/evidence references.

Scenarios should test the answer contract rather than the client UI. Keep fixtures small and explicit so a failure can be traced to one missing or incorrect reasoning step.

## Run

Run all scenarios from the repository root:

```bash
python -m benchmarks.eva_coherence.runner --scenario all
```

## Interpreting Failures

Treat a failure as an attention signal for the agent/tool contract before patching any client code. First inspect the scripted transcript, tool results, expected constraints, and prompt wording. A failing scenario may indicate a real agent issue, an outdated fixture, ambiguous expected wording, or a harness bug.

Do not use this harness result alone as proof that a visual event happened or did not happen. P/N/M labels are attention signals, not proof. Visual confirmation requires evidence frames and an explicit `describe_frame` result grounded in those frames.
