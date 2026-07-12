# Data Retention, Privacy & EU AI Act

Technical data-governance reference for a government pilot monitoring public
spaces. The **technical facts** here are authoritative; **legal classification and
conclusions are `[NEEDS LEGAL]`** for PM/legal sign-off. Pairs with
[security_threat_model](security_threat_model.md).

## Data collected

| Category | Contents | Store |
|---|---|---|
| Frames | Sampled thumbnails from public-space cameras | `archive.detections` (Postgres) |
| Embeddings | CLIP vectors of frames (no raw biometric templates) | `archive.detections` |
| Descriptions | VLM natural-language summaries of batches | `archive.runtime_state` (history) |
| Alerts/bookmarks | Structured events + severity | history + Luxriot Evo |
| Probe definitions | Operator/agent search criteria | `archive.probes` |
| IAM | User accounts, roles, channel grants | IAM schema |
| Audit | Sensitive actions, agent tool calls | audit schema |

EVA stores **descriptions and sampled frames**, not a continuous video copy
(Luxriot Evo holds the recordings). No face-recognition/identity matching is
performed; CLIP vectors are semantic, not identity templates. `[NEEDS LEGAL]`
confirm this characterization for the jurisdiction.

## Retention (configurable — see [config_reference](../00_CANON/config_reference.md))

| Data | Control | Default |
|---|---|---|
| Archive rows | `EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS` | 90 (pilot: set to demo window) |
| Thumbnails | `EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS` | 14 |
| Row cap | `EVOSSEARCH_ARCHIVE_MAX_RECORDS` | 5,000,000 (raise for multi-week) |
| Summary history | `EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS` | 7 |
| Semantic L1–L3 history | `EVOSSEARCH_LUXRIOT_ROLLUP_RETENTION_DAYS` | archive row retention (normally 90) |

Retention is enforced by a scheduled prune. **Data minimization:** keep windows
no longer than the pilot/operational need; default to the shortest window that
supports the use case. `[NEEDS LEGAL]` set the binding retention policy.

## Access control & isolation

- Named auth, RBAC, **per-channel grants** — users see only granted channels.
- PostgreSQL **row-level security forced** on archive/agent/audit/IAM, tenant-scoped.
- Separate DB runtime roles (API/audit/worker).
- All sensitive access audited (who/what/when).

## Deletion / erasure

- Time-based retention prunes rows + thumbnails automatically.
- Targeted deletion (by channel/time) is possible via DB tooling; document an
  erasure SOP. `[NEEDS LEGAL]` define request handling + timelines.

## EU AI Act — technical mapping (draft, `[NEEDS LEGAL]` for classification)

Public-space monitoring AI is likely in scope as high-risk; **classification and
permitted-use conclusions require legal review.** Below is the technical evidence
the system already produces toward common high-risk obligations:

| Obligation (indicative) | What EVA provides technically |
|---|---|
| **Human oversight** | Agent advises, human acts: preview-only changes, approval-gated bookmarks, evidence-cited candidate conclusions. See security model. |
| **Transparency** | Outputs labelled as model-generated; CLIP transitions labelled "candidates, not ground truth"; coverage contracts state what was/ wasn't inspected. |
| **Logging & traceability** | Audit of sensitive actions and agent tool calls; alerts/bookmarks carry timestamps and channel; frame evidence retained. |
| **Accuracy / robustness** | Known limitations documented (VLM miss/false rate, recall bounds); parallel watch-list detector; severity grading; no autonomous action on uncertain signals. |
| **Data governance** | Retention controls, RLS isolation, channel scoping, minimization defaults, no identity matching. |
| **Technical documentation** | This docs set: architecture, config, security, retention, operator/admin guides. |
| **Record-keeping** | Audit log + immutable release notes/changelog + canonical facts. |

Gaps to close for a formal posture (`[NEEDS LEGAL]` + engineering):
- Written DPIA / conformity assessment, permitted-purpose and legal-basis
  statement, and authority notifications — **legal/PM owns**.
- Bias/accuracy evaluation report for the deployed models on the client's scenes.
- Formal human-oversight procedure signed by the operating authority.

## What we tell the client (honest framing)

EVA is a **decision-support** tool for operators, not an autonomous decision
system: it surfaces and explains candidate public-order events with evidence and
explicit coverage, a human confirms and acts, and every sensitive action is
logged. It stores descriptions and sampled evidence under bounded retention, with
per-channel access control — not a parallel face-recognition database.
