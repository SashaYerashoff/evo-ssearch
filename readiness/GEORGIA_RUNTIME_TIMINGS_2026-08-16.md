# Georgia β 0.8.1 → β 0.8.7 latency snapshot

Collected: `2026-08-16T08:56:53.273654+00:00`  
Source commit: `07abb370ccbf7602a096a5dd8f0c328a9238f6c8`  
Live status: `ready`; schema: `20260805_0013`
Current window: `6.0` h; validation history: `168.0` h
Database sample cap: latest `1024` rows per source

## Processing cost

| Stage | Samples | p50, ms | p95, ms | What is measured |
|---|---:|---:|---:|---|
| CV apex | 512 | 5.2 | 12.3 | gray-160 + frame delta + edge variance, CPU |
| SigLIP batch compute | 256 | 70.5 | 380.1 | live recent worker window |
| SigLIP queue wait | 256 | 11.4 | 94.6 | live recent worker window |
| Fast VLM inference | 0 | — | — | model execution only |
| Fast VLM event → processed | 0 | — | — | batching/roll + queue + inference + parse |
| Full L0 VLM inference | 99 | 9651.0 | 17718.0 | model execution only |
| Full L0 VLM event → processed | 99 | 26731.0 | 45541.0 | batching/roll + queue + inference + parse |

## Bookmark acknowledgement in Evo

| Pipeline | Sent samples | Event → Evo ack p50, ms | p95, ms | EVA → Evo delivery p50, ms |
|---|---:|---:|---:|---:|
| Operator probe, direct lane | 0 | — | — | — |
| Probe retrospective fallback | 0 | — | — | — |
| Fast VLM alert | 0 | — | — | — |
| Full L0 VLM alert | 26 | 25065.0 | 37668.0 | 45.0 |

### Validation history

This second table retains controlled probe/VLM checks which may not have a fresh hit in the current window.

| Pipeline | Sent samples | Event → Evo ack p50, ms | p95, ms | EVA → Evo delivery p50, ms |
|---|---:|---:|---:|---:|
| Operator probe, direct lane | 84 | 889.0 | 2843.0 | 62.0 |
| Probe retrospective fallback | 83 | 3618.0 | 43250.0 | 75.0 |
| Fast VLM alert | 5 | 12976.0 | 18328.0 | 87.0 |
| Full L0 VLM alert | 282 | 30987.0 | 50239.0 | 75.0 |

## Interpretation

- CV and SigLIP values are compute/queue costs, not camera-to-operator latency.
- A direct operator probe still waits for the configured embedding cadence and hit confirmation; its model score itself is a millisecond-scale operation.
- Full L0 VLM includes the batch observation window. Fast VLM includes post-roll and admission. Evo delivery after EVA decides is normally only tens of milliseconds.
- Only rows with an actual `sent` acknowledgement enter the bookmark table. Cooldown/deduplicated alerts are intentionally excluded.
- Small sample counts are shown rather than hidden. Re-run immediately after the controlled thumbs-up tests for release acceptance numbers.
- Database reads are deliberately capped per source so a timing preflight does not decompress the full historical evidence archive.
- Semantic presence is an affinity/homeostasis signal, not object detection. CV apex is an attention selector, not event truth.
