# Known Limitations ("do not promise")

Honest limits of the pilot, so operators and sales do not overpromise — important
for a government buyer. Each item has the mitigation in place. Pairs with
[security_threat_model](architecture/security_threat_model.md) and
[cognitive_architecture](architecture/cognitive_architecture.md).

## Detection & accuracy

- **The VLM can miss or mislabel.** Small model, batched frames; alert emission is
  not guaranteed. *Mitigation:* parallel watch-list probes, severity grading,
  always human-confirmed on evidence frames. Do not promise exhaustive detection.
- **CLIP state-transition counts are candidates, not ground truth.** *Mitigation:*
  boundary-frame evidence + "describe to confirm". Do not present counts as facts.
- **Archive probe calibration is evidence-based, not labeled ground truth.** It
  estimates thresholds from stored CLIP vectors and can inherit archive/source
  bias. *Mitigation:* representative frames + preview-only probe changes.
- **CLIP does not understand negation** ("no vehicle"). *Mitigation:* visible-
  background contrast phrasing ("empty gate").

## Search & memory

- **Broad search recall is bounded by the candidate window.** An unscoped
  "search the last two weeks" only inspects the most recent slice. *Mitigation:*
  scope by channel + time; coverage contract surfaces truncation. Do not promise
  full-archive recall without scoping.
- **Coverage gaps under load.** Under heavy channel counts, description batches can
  drop. *Mitigation:* coverage contracts + stream-health visibility make gaps
  explicit (a quiet channel is distinguishable from a blind one).

## Agent

- **No autonomous sensitive actions.** Prompt/probe edits are preview-only;
  bookmark creation is approval-gated. The agent advises; a human acts.
- **Prompt-injection is a structural surface.** Inputs the agent reads can attempt
  to steer it. *Mitigation:* gateway authz independent of the agent, evidence-cited
  candidate conclusions, human-in-the-loop. Do not let the agent action anything
  irreversible unattended.
- **Granular continuity.** The agent reconstructs its working state each turn; long
  threads can drift. *Mitigation:* restate channel/period to re-ground.
- **No export formats** beyond structured chat reports (no PDF/CSV/email/async
  queues). Do not promise these.

## Delivery & runtime

- **Bookmark delivery can fail** at the Luxriot API. *Mitigation:* counted/logged;
  in-EVA evidence retained regardless. Do not promise guaranteed Luxriot delivery
  without monitoring.
- **Single Gunicorn worker.** A crash stops capture until restart. *Mitigation:*
  desired sessions auto-restore; graceful-restart flush. No HA in the pilot.
- **Static VLM balancer, no failover.** A dead VLM host stops its channels until
  reassignment. *Mitigation:* coverage visibility.
- **Hard kill (SIGKILL)** can lose up to the persist-debounce interval of summary
  history. Graceful restarts flush.

## Scaling

- The pilot is sized for ~50 channels with the chosen cadence. **10k streams needs
  the durable queue + worker pool, pgvector ANN recall, and time-partitioned
  retention** — see [system_architecture](architecture/system_architecture.md).
  Do not quote 10k on the current single-node, brute-force-recall build.

## Compliance / commercial (open)

- EU AI Act classification, DPIA, EULA, and model/third-party notices are **not
  finalized** — `[NEEDS LEGAL]` / PM. See
  [data_retention_privacy](architecture/data_retention_privacy.md) and
  [legal/dependency_license_audit](legal/dependency_license_audit.md).
