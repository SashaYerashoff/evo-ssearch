# Luxriot EVA AI β 0.8.0 Release Notes

Release date: 2026-06-14  
Previous public baseline: `α 0.4.2`  
Release class: production-pilot beta for controlled on-prem deployment

## Summary

`β 0.8.0` moves EVA AI from an alpha PoC into a controlled production-pilot build. The main shift is not one feature: it is the addition of named users, role-based access, PostgreSQL-backed runtime state, audited agent tools, a searchable frame archive, VLM/probe retention controls, distributed inference profiles, and an Ubuntu deployment path.

This is suitable for a supervised client pilot in a closed network. It is not yet a GA/1.0 release.

## Headline Changes Since α 0.4.2

### Users, Roles, And Audit

- Added named-user authentication backed by PostgreSQL.
- Added role-based authorization for admin, engineer, operator, and viewer workflows.
- Added channel-scoped access control, including support for non-admin users with access to all channels.
- Added admin APIs and UI for user creation, password reset, session revoke, enable/disable, and channel grants.
- Added durable login throttling and session inventory.
- Added audit recording for sensitive endpoints and agent tool execution.
- Added protected audit event reader for admin/diagnostics workflows.
- Added fail-closed behavior when channel ownership metadata is missing.

### PostgreSQL Control Plane

- Added PostgreSQL control-plane foundation and Alembic migrations.
- Added strict runtime-role readiness checks for secure deployments.
- Added separate DSN paths for API, audit, worker, and migration roles.
- Added migration `20260612_0005_archive_runtime` for archive runtime storage.
- Added migration `20260614_0006_iam_all_channel_access` for all-channel IAM grants.
- Added Ubuntu LTS deployment guide with systemd, env, Postgres, auth, and retention setup.

### Frame Archive And Search

- Added PostgreSQL-backed frame archive for probe hits and VLM-sampled frames.
- Added archive source semantics:
  - `probe`: actual probe hit / detection.
  - `vlm_summary`: sampled frame saved from a video-description batch.
  - `vlm_alert`: frame anchored to a VLM alert.
- Added archive previews/thumbnails for VLM and probe frames.
- Added archive search source filtering, custom time range filtering, and source labels in UI.
- Added match-threshold slider behavior for archive search results.
- Added probe score details in archive results: positive score, negative score, and margin.
- Added archive retention policy controls and storage/capacity estimates.
- Simplified archive UI for deployment by hiding indexed-folder controls from the main workflow.
- Default production embedding policy now favors CLIP; experimental DINO/fusion paths are disabled unless explicitly enabled.

### Luxriot Video Understanding

- Added live VLM profile routing for distributed inference.
- Added named LM profiles for agent and VLM backends.
- Added optional static VLM balancer over configured model endpoints.
- Added per-channel live capture cadence control.
- Added stream context panel with channel, preview, cadence, batch, model, queue, and probe state.
- Added L0-L3 video-summary rollup handling with retention settings.
- Added VLM alert indicators on collapsed summary rows.
- Added VLM summary archive callback so selected frames are searchable in Archive Research.
- Added support for offline image/video upload analysis, not only server-path video analysis.
- Improved Luxriot readiness checks and runtime state reporting.

### Probes And Monitoring

- Added WSGI-started probe daemon so saved probes run under production gunicorn.
- Saved probes and live summaries continue after admin/operator logout.
- Added per-probe bookmark gate settings and stronger dedupe/cooldown controls.
- Added ROI-aware probe capture and probe editor improvements.
- Added "cast probe" workflow to apply one probe to multiple channels.
- Added probe threshold/margin visibility needed for tuning noisy probes.
- Added safer probe tuning semantics in agent instructions.
- Fixed invalid probe margin handling.

### EVA Agent

- Added secured agent tool gateway with per-tool authorization, channel scope, rate limits, row limits, and audit.
- Added durable action approval flow for sensitive agent operations.
- Increased per-turn tool budget to 64 calls.
- Added tool progress streaming and improved browser streaming resilience.
- Added agent model controls independent of VLM model controls.
- Added video-summary review workflow for period-based reports.
- Added broad-channel behavior: if the operator asks about many active channels, the agent inventories candidates and asks for confirmation before full multi-turn research.
- Added archive source semantics to agent prompts so it does not confuse probe detections with VLM-sampled frames.
- Added time-window normalization rules for period-based archive and video-summary queries.
- Added playbooks for probe tuning and deployment/protocol workflows.

### UI/UX

- Refactored the monolithic page into template, CSS, and JS assets.
- Added a cleaner three-tab workspace for Archive, Video, Monitoring, and Agent flows.
- Improved Settings modal layout with compact tabs and fixed footer behavior.
- Added admin channel picker instead of manual channel-ID entry only.
- Hid inaccessible tabs and controls according to user permissions.
- Improved Chrome chat continuation behavior after agent responses.
- Improved panel scrollbars, stream preview layout, and right-side video-analysis panel sizing.
- Added image/video upload controls for offline video analysis.

### Deployment And Readiness

- Added `/health` liveness endpoint.
- Added `/ready` component readiness endpoint.
- Added secure-deployment readiness gate for auth, Postgres, Luxriot, embedder, deployment security, LM profiles, and inference queue.
- Added support for control-plane deployments without a local vision stack.
- Added inference queue foundation and PostgreSQL inference queue adapter.
- Added distributed model profile configuration for separate agent/VLM endpoints.
- Added deployment runbook for Ubuntu LTS.
- Added route security smoke coverage and broader auth/archive/agent tests.
- Updated visible product version to `β 0.8.0`.

## Upgrade Notes

### Required Database Migration

Run migrations before starting the beta service:

```bash
set -a
. /etc/eva-ai/eva-ai.env
set +a
alembic upgrade head
alembic current
```

Expected head:

```text
20260614_0006
```

### Required Deployment Settings

For a secure pilot deployment, use named DB roles and enable strict checks:

```env
EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true
EVOSSEARCH_AUTH_ENABLED=true
EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true
EVOSSEARCH_ARCHIVE_STORE=postgres
EVOSSEARCH_EMBEDDER=clip
EVOSSEARCH_DINO_SEGMENTS_ENABLED=false
EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED=false
EVOSSEARCH_GUNICORN_WORKERS=1
```

If HTTPS/TLS is enabled at the app or reverse proxy boundary:

```env
EVOSSEARCH_AUTH_COOKIE_SECURE=true
```

If the deployment env overrides the app version, update or remove it:

```env
EVOSSEARCH_APP_VERSION="β 0.8.0"
```

### Inference Profiles

For a deployment with app/CLIP on one server and model inference on another, configure explicit LM profiles:

```env
EVOSSEARCH_LM_PROFILES=agent,vlm-rtx6000
EVOSSEARCH_LM_AGENT_PROFILE_ID=agent
EVOSSEARCH_LM_VLM_PROFILE_ID=vlm-rtx6000
EVOSSEARCH_LM_VLM_BALANCER_ENABLED=false

EVOSSEARCH_LM_PROFILE_AGENT_KIND=agent
EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=http://<model-host>:1234/v1
EVOSSEARCH_LM_PROFILE_AGENT_MODEL=<agent-model-id>
EVOSSEARCH_LM_PROFILE_AGENT_TIMEOUT=600

EVOSSEARCH_LM_PROFILE_VLM_RTX6000_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_RTX6000_BASE_URL=http://<model-host>:1234/v1
EVOSSEARCH_LM_PROFILE_VLM_RTX6000_MODEL=<vlm-model-id>
EVOSSEARCH_LM_PROFILE_VLM_RTX6000_TIMEOUT=600
```

Use the VLM balancer only when multiple VLM endpoints are available:

```env
EVOSSEARCH_LM_VLM_BALANCER_ENABLED=true
EVOSSEARCH_LM_VLM_BALANCER_PROFILES=vlm-1,vlm-2,vlm-3,vlm-4
```

The current balancer is static channel-to-profile routing, not health-aware failover.

## Verification Run

The beta commit was verified with:

```text
129 tests OK, 5 skipped
py_compile OK
alembic current: 20260614_0006 (head)
```

Known test-log noise:

- Some tests intentionally log mocked archive/image-serving failures.
- The unittest process may print DB worker shutdown warnings after successful assertions.

## Known Limitations

- Live Luxriot capture and saved-probe runtime are still in-process. Logout is safe, but service restart stops active sessions until they are started again.
- Gunicorn worker count must stay at `1` for this process model. Multiple workers can duplicate probe daemons and split runtime capture state.
- VLM profile balancing is static. It does not yet monitor GPU utilization or fail over unhealthy endpoints.
- Agent broad research over many channels is chunked through turns, not a durable background job queue.
- Container deployment is not finalized in this release.
- Legal/commercial documents, model notices, and customer EULA still need finalization before GA.

## Recommended Pilot Checklist

1. Deploy parallel to the old PoC on a new port/service name.
2. Apply migrations to head.
3. Create the first admin user and disable shared-token workflows.
4. Verify `/health` and `/ready`.
5. Verify Luxriot channel list and one preview snapshot.
6. Start one VLM summary channel with a conservative cadence.
7. Create a noisy probe, confirm probe hits in Archive Research, then tune thresholds.
8. Confirm VLM summary frames appear in Archive Research as `Video-description frame`.
9. Create operator/viewer accounts and verify UI hiding/channel scope.
10. Only then scale toward the target channel count.
