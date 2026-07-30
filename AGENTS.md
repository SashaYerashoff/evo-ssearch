# Agent Working Notes

## tuktuk Grammar Pin

Before changing agent tools, tool result compaction, intent groups, or
`skills/*/SKILL.md`, read [docs/tuktuk/grammar_pin.md](docs/tuktuk/grammar_pin.md).
The tuktuk grammar is the pinned reflex contract for the EVA agent surface.

Treat these changes as grammar-affecting by default:

- adding, removing, renaming, or changing a tool schema in `agent.py`;
- changing `AgentTools.execute` dispatch behavior;
- changing `_TOOL_INTENT_GROUPS`, security exposure, or progressive disclosure;
- changing `_compact_tool_result_for_model` keys or result envelope shape;
- adding or materially changing a skill/runbook under `skills/`;
- adding UI-driving agent actions or write/approval flows.

For any such change, check whether it still fits the pinned block grammar:
`W`, `C`, `MAP`, `RANK`, `DRILL`, `AGG`, `CAL`, `MUT`, `TERM`; the global gates;
and the extractability law: model arguments come only from operator text,
scratchpad/prior result, or closed enums.

If the change conflicts with the pin, do not silently bend the implementation or
the grammar. Raise the conflict explicitly in the work log or review summary and
add/update a question in `docs/tuktuk/grammar_review_questions.md` until Sasha
accepts a grammar revision.
