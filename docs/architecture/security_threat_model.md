# Security & Threat Model

Security posture of the EVA AI pilot, including the **honest limits of the AI
agent**. Pairs with [data_retention_privacy](data_retention_privacy.md) and the
[cognitive_architecture](cognitive_architecture.md) (why the agent needs
human-in-the-loop). `[NEEDS LEGAL]` marks items for PM/legal sign-off.

## Assets

- Live and archived imagery of public spaces and the people in them.
- Derived data: descriptions, alerts, CLIP vectors, bookmarks.
- Credentials/secrets (Luxriot, DB DSNs, LM API keys).
- IAM data, audit log, agent session history.

## Actors & roles (RBAC)

- **admin** — user/role/grant management, settings, audit.
- **engineer** — configuration, prompt/probe tuning.
- **operator** — monitoring, agent use, scoped channels.
- **viewer** — read-only, scoped channels.
- **External/attacker** — not authenticated; or an authenticated user acting
  beyond scope; or a hostile *input* reaching the agent.

## Trust boundaries & controls

| Boundary | Control |
|---|---|
| Browser ↔ app | TLS at browser-facing boundary; secure cookies; CSRF token; session TTL |
| User ↔ data | Named auth; RBAC; **per-channel grants**; RLS forced in DB |
| App ↔ PostgreSQL | Separate runtime roles (API/audit/worker); RLS tenant isolation |
| App ↔ Luxriot / LM hosts | Closed network; credentials in `.env` (`0600`), never in code/docs |
| Sensitive ops | Audited; mutating/sensitive routes centrally guarded |
| Disabled features | Offline-video / probe-snap / indexed-folder return 404 server-side, not just hidden |

Gunicorn's default production service is internal HTTP (`EVOSSEARCH_PORT=5000`).
TLS is normally supplied by Nginx or the site's TLS boundary. If a temporary
office/lab deployment is HTTP-only, `EVOSSEARCH_AUTH_COOKIE_SECURE=false` may be
needed for browser login, and the deployment must be treated as non-client-facing
until HTTPS is restored.

## The AI agent as a threat surface (important)

The agent reasons over representations: every signal (operator text, tool
results, retrieved descriptions) reaches it through one channel. This yields a
structural property, not a patchable bug:

- **Prompt injection = a hostile driver-signal.** Content the agent reads
  (a crafted description, a planted note, an uploaded image's text) can attempt to
  steer it exactly like a legitimate instruction. Nothing reaching the agent is
  *incorrigible*.

Therefore the controls are architectural, not just filtering:

1. **No autonomous sensitive actions.** Prompt/probe changes are **preview-only**
   (diff shown, applied only on explicit human confirmation). Bookmark creation is
   **approval-gated**.
2. **Authorization is enforced at the tool gateway**, independent of what the
   agent "decides" — per-tool permission, channel scope, rate/row limits, audit.
   A compromised prompt cannot exceed the user's grants.
3. **Conclusions are evidence-cited candidates**, not verdicts — operators confirm
   on the boundary/described frames.
4. **Control vs content separation** — urgency/severity/budgets modulate priority,
   they are not authorities the agent can be talked into overriding.

Operational rule for the client: **the agent advises; a human acts** on anything
that leaves the system (bookmarks/exports) or changes its behavior (prompts/probes).

## Residual risks (disclose, don't hide)

- VLM model errors: missed or false alerts; mitigated by parallel watch-list
  probes, severity grading, and human confirmation.
- Coverage gaps under load: surfaced via coverage contracts and stream-health,
  not silent.
- Bookmark delivery failures: counted/logged (`bookmark_failed_count`), in-EVA
  evidence retained even if Luxriot rejects.
- Single-worker availability: a crash stops capture until restart; desired
  sessions auto-restore.
- Closed-network assumption: the model relies on network isolation; exposure to
  untrusted networks is out of scope for the pilot. `[NEEDS LEGAL]`/infra review.

## Out of scope (pilot)

- Multi-tenant hostile co-tenancy, public internet exposure, container hardening,
  HA/failover, and formal pen-test sign-off — `[NEEDS LEGAL]` / future work.

## Secrets handling

- All secrets in on-host `.env` (`0600`); never committed; never in shareable
  docs. `.env` writes are atomic and quoted. Rotation via admin tooling +
  service restart.
