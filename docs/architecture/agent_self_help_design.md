# Design: Agent Self-Help over Operator Docs (BM25)

Goal: let the agent answer "how do I do X in the UI?" / "what does this scenario
mean?" by retrieving from EVA's **own vetted operator docs**, so an operator (or
the intern) can ask the agent how to push the right button to get things working —
not just click and observe.

Status: design proposal for implementation (codex). Connects to the
[cognitive_architecture](cognitive_architecture.md) idea of the operator guide as
another signal source the agent can bring into context.

## Why this is low-risk (no catastrophic avoidance needed)

This is **read-only retrieval over a small, curated, first-party corpus** —
markdown we wrote and control. It is not web access, not user-generated content,
not an action. The realistic failure mode is "unhelpful answer," not "harmful
action." Concretely:

- No tool side effects: retrieval only, returns text passages.
- Corpus is our own docs → no new injection surface beyond content we author.
- Behind the existing tool gateway (authz, rate limit, audit) like any tool.
- Output is labelled as **operator-guide help**, kept separate from incident data
  so the agent never confuses "how-to" text with facts about a scene.

The one real precaution: **do not index secret/field-specific docs.** Index only
the sanitized operator docs; exclude `install/field_rollout_demo.md`, `.env`,
anything with `[FIELD]` credentials/topology.

## Simplest safe solution: BM25 over chunked markdown

BM25 (lexical) is ideal here: the corpus is tiny, queries are keyword-ish ("where
is the min match slider", "how to start a video-description channel"), and it needs
no embedding model or DB. CLIP stays for images; this is plain text.

### Corpus (index these)
- `docs/operator/*.md` (operator_guide, operator_scenarios, agent_capabilities, demo_runbook)
- `docs/00_CANON/glossary.md`
- (optionally) `docs/admin/observability.md` for operator-relevant health checks

### Chunking
- Split each doc by `##` / `###` headings → records of
  `{doc_path, section_heading, text}`. Headings make great citations and keep
  chunks focused.

### Index
- `rank_bm25` `BM25Okapi` over tokenized chunks, built **in-memory at startup**.
- Rebuild when a source file's mtime changes (cheap; corpus is small).
- No persistence/DB needed.

### Tool
- New read-only agent tool: `lookup_help(query: str, top_k: int = 3)`.
- Returns: list of `{doc, section, snippet, score}` (snippet trimmed to a budget).
- Register in `agent_security/eva_adapter.py`: permission `agent:use` (any agent
  user), modest rate limit, no channel scope, audited like other tools.

### Prompt rule (add to `build_system_prompt`)
> For UI / how-to / "where is the button" / scenario-meaning questions, call
> `lookup_help` first and answer from the returned passages, citing the doc and
> section. Do not invent UI steps or menu names; if `lookup_help` returns nothing
> relevant, say so and suggest the closest documented action.

### Result handling
- Label results clearly as **operator-guide help** in the UI tool-result card.
- The agent paraphrases + cites (e.g., "Operator Guide § Video tab"), so the
  operator can open the doc.

## Integration checklist (codex)
1. Add a small `agent_help_index.py` (chunk + BM25 build + query); pure-python dep
   `rank_bm25` (or a trivial TF-IDF if avoiding a dep).
2. Build the index at app startup from the corpus list; mtime-based refresh.
3. Add `lookup_help` tool + schema in `agent.py`, dispatch entry, and a compact
   UI result formatter.
4. Register authz/limits in `eva_adapter.py`.
5. Add the prompt rule; add 1–2 tests (query → expected section retrieved;
   excluded-doc never returned).

## Consequence: docs become load-bearing
Once the agent answers from these docs, **doc accuracy is operational.** This
raises the value of the canon + anti-drift discipline
([config_reference](../00_CANON/config_reference.md), facts, and the planned CI
grep-guard): a wrong instruction in a doc becomes a wrong answer from the agent.
Keep the operator docs current as part of the release checklist.

## Optional later upgrade
If lexical recall proves weak on paraphrased questions, add CLIP/text-embedding
reranking of the BM25 top-N (reuse the existing embedder). Not needed for the
pilot — BM25 over a curated corpus is enough and simpler/safer.
