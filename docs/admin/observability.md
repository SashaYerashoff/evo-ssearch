# Observability & Health

What to watch so a "quiet" system is actually quiet, not blind — during the demo
and the data-collection week. Invariants: [facts](../00_CANON/facts.md).

## Endpoints

- `GET /health` — liveness. Should be ok whenever the service is up.
- `GET /ready` — component readiness: auth, PostgreSQL, Luxriot, embedder,
  deployment security, LM profiles, inference queue. Use after every restart/patch.
  A `not_ready` names the failing component.

## Video-description coverage (the key signal)

Per channel, watch (Video tab stream-health + agent `list_video_summary_channels`):
- **running** — is the channel actively producing descriptions?
- **coverage gaps / dropped batches** — `dropped_frames`, `queue_dropped_batches`,
  `last_error` on the capture session. Non-zero and rising = the channel is
  partially blind. Investigate (VLM saturation, Luxriot snapshot errors).
- **first/latest description time** — if `latest` lags well behind now, the
  channel stopped producing (a drop).

Rule: a channel reporting **0 alerts with no gaps** is genuinely quiet; **0 alerts
with gaps** means you lost coverage — don't trust the silence.

## Alert & bookmark delivery

- `alerts_parsed` — alerts extracted from descriptions.
- `bookmark_failed_count` / `bookmark_last_error` — Luxriot bookmark delivery
  failures. If failures climb while alerts are parsed, alerts are being detected
  but not reaching Luxriot (check Luxriot API/credentials/severity mapping). The
  in-EVA `vlm_alert` evidence is retained regardless.

## Inference & queue

- If the durable queue is enabled: `queue_depth`, `oldest_age_seconds`,
  `dropped_count`, `workers_alive`. Rising depth/age = VLM can't keep up.
- Default (queue off): dispatch is synchronous; watch capture `last_error` and
  per-channel cadence instead.

## During the demo

- Pre-flight: `/ready` green; demo channels running with no gaps; recent alerts
  present; bookmark failures flat.
- Live: if the agent slows, check VLM host load; avoid bursts of fresh
  `describe_frame`.

## During the collection week

- Daily: coverage gaps per channel, archive growth vs. cap
  (`ARCHIVE_MAX_RECORDS`), summary-history growth, bookmark failure trend.
- Watch disk (thumbnails) and DB size against the [config_reference](../00_CANON/config_reference.md)
  retention settings.

## Logs

- Service logs (systemd journal for `eva-ai` / `eva-ai-local-5443`).
- Audit log (admin/diagnostics reader) for sensitive actions and agent tool calls.
