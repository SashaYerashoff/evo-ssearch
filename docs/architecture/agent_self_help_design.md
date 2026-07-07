# Design: Agent Self-Help over Operator/Admin Docs (BM25)

Goal: let the agent answer "how do I do X in the UI / what does this scenario
mean / what's the product status capability" by retrieving from EVA's **own
vetted, role-appropriate docs** — so an operator can ask the agent how to push the
right button to get things working.

Status: **finalized spec** (incorporates codex review). Ready for implementation.
Read-only, low-risk: retrieval over a first-party allowlisted corpus, behind the
existing agent gateway. No web, no actions, no new dependency.

## 1. Corpus — allowlist only

**Index (allowlist):**
- `docs/operator/*.md` — audience: `operator`
- `docs/00_CANON/glossary.md` — audience: `operator`
- `docs/known_limitations.md` — audience: `operator`
- `docs/admin/observability.md` — audience: `operator` (monitoring/health is
  operator-relevant, not privileged). Section-level overrides are a possible
  future extension, not implemented.
- `docs/admin/admin_guide.md` — audience: `admin`, required_permission `users:manage`
- `docs/admin/backup_recovery.md` — audience: `admin`, required_permission `settings:manage`
- `docs/install/deployment_guide.md`, `docs/install/inference_topology.md` —
  audience: `engineer`, required_permission `settings:manage`

**Never index (exclude):**
- `readiness/history/`, `docs/gtm/`, `docs/legal/`
- `install/field_rollout_demo.md`, `.env`, anything with `[FIELD]` values, real
  IPs, credentials, or secrets.

The allowlist is an explicit config (file path → default audience + optional
section overrides), reviewed in code — not "scan everything under docs/".

## 2. Chunking & metadata

Split each allowlisted doc by `##`/`###` headings. Each chunk record:

```
{
  doc_path,            # e.g. docs/admin/admin_guide.md
  heading,             # the section heading
  section,             # short path/label for citation
  audience,            # operator | engineer | admin  (deterministic, from allowlist + overrides)
  required_permission, # optional, from the Permission enum where known
  text                 # chunk body, markdown-stripped for scoring
}
```

`audience` and `required_permission` are assigned deterministically per file from
the allowlist (section-level overrides are a possible future extension, not yet
implemented). `required_permission` is None for operator docs and, for
admin/engineer docs, references the real `Permission` gate
(`users:manage`, `settings:manage`).

Permissions are **never** read from model/tool arguments. The secure adapter sets
them on a trusted execution-context channel (`_set_trusted_permissions` from
`ToolExecutionContext.permissions`) and strips any model-supplied
`_granted_permissions`; the legacy/non-secure path defaults to operator-only.

## 3. In-repo BM25 (no new dependency)

Implement a tiny Okapi BM25 in `agent_help_index.py` — no pip dependency (closed
network + license-audited; ~40 lines beats a wheelhouse update):

- **Tokenize:** strip code fences and markdown markers; lowercase; `[a-z0-9]+`;
  drop a small stopword list.
- **Index:** per-chunk term frequencies; corpus document frequencies; average doc
  length. Built **at startup (deterministic rebuild)**; optional mtime-based
  refresh is an optimization, not required.
- **Score (Okapi BM25, k1=1.5, b=0.75):**
  `idf(t) = ln(1 + (N - df + 0.5)/(df + 0.5))`,
  `score = Σ idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))`.

## 4. Tool contract

`lookup_help(query: str, top_k: int = 3)` →

```
{
  "results": [            # allowed for the caller's role
    {doc, section, heading, snippet, score}
  ],
  "restricted_matches": [ # admin/engineer-only hits, NO detailed procedure
    {section, heading, required_permission}
  ],
  "indexed_docs": [...],  # optional
  "refreshed_at": ...     # optional
}
```

- **Snippets bounded** (~400–600 chars); never return whole large markdown blocks.
- `restricted_matches` lets the agent acknowledge an admin answer exists and
  redirect, without exposing steps.

## 5. Authorization & gateway

- Read-only tool behind the existing agent gateway: permission `agent:use`,
  **audited**, **rate-limited**, no side effects, no channel scope.
- Role/permission filtering uses the caller's role/permissions from
  `tool_context`: a chunk is in `results` only if the caller's audience/permission
  covers it; otherwise it goes to `restricted_matches`.
- **The security boundary is the gateway/RBAC, not the corpus.** Knowing a
  procedure ≠ being able to run it (admin tools/routes stay gated independently).
  Corpus role-gating is for accuracy + defensibility, not secrecy.

## 6. Redirect behavior

If the best match for a non-privileged caller is admin/engineer-only, the agent
must **not** recite steps. It answers: *"That's an admin/engineer action — ask a
user with permission `<X>`."* (Name the role/permission from
`required_permission`/audience.)

## 7. Prompt rule (add to `build_system_prompt`)

> For UI / how-to / product-status / scenario-meaning questions, call `lookup_help`
> first and answer from the returned passages, citing doc + section. Do not mix
> help-doc passages with incident evidence. If a match is in `restricted_matches`,
> tell the operator it's an admin/engineer action and name the required permission
> — do not invent the steps. If `lookup_help` returns nothing relevant, say it's
> **not documented** rather than inventing UI paths.

## 8. Tests (required)

- Operator query → expected operator section returned.
- Admin query (admin caller) → expected admin section returned.
- **Operator query hitting an admin section → `restricted_matches` redirect, not
  the procedure.**
- Excluded docs (`field_rollout_demo`, `docs/gtm`, `docs/legal`, history) **never**
  appear in any result.
- Deterministic startup rebuild produces a stable index (and mtime refresh if
  implemented).
- Gateway path: `agent:use` required, audited, rate-limited; no side effects.

## 9. Anti-drift tie-in

The corpus is now agent-load-bearing — a wrong instruction becomes a wrong agent
answer. Add the allowlist to the release/doc checklist
([../maintenance.md](../maintenance.md)); optionally extend
`scripts/check_docs_drift.sh` to assert every allowlisted file exists.

## Implementation checklist (codex)
1. `agent_help_index.py` — allowlist config + chunker + in-repo BM25 + role filter.
2. Build index at startup; expose `query(text, top_k, caller_audience/perms)`.
3. `lookup_help` tool + schema in `agent.py`; dispatch entry; compact UI formatter.
4. Register in `agent_security/eva_adapter.py`: `agent:use`, rate limit, audit.
5. Prompt rule in `build_system_prompt`.
6. Tests per §8.
