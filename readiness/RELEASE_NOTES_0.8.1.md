# Luxriot EVA AI β 0.8.1 Release Notes

Release type: production-pilot stabilization patch  
Previous baseline: `β 0.8.0`

`β 0.8.1` hardens the 50-channel pilot build after live workflow testing. The focus is agent reliability, visual evidence handling, read-only summary inspection, and deployment runbooks for offline/client environments.

## Highlights

- Added `track_visual_state_transitions` agent tool for visual appear/disappear, open/close, and leave/return workflows over archived CLIP-scored VLM frames.
- Added CLIP-safe negative-state handling: negated phrases like `no vehicle` are converted into visible-background contrast phrases such as `empty gate`.
- Added top candidate frame evidence even when no stable transition is confirmed.
- Added automatic evidence thumbnails for agent answers that mention detection/frame IDs.
- Made agent video-summary reads use read-only rollup mode so investigation tools do not trigger LLM rollup synthesis.
- Added coverage and truncation contracts for video-summary and visual-state workflows.
- Removed demo-specific prompt/schema examples from agent-visible tool contracts.
- Improved VLM feed JSON handling, UI layout stability, stream health visibility, and archive/agent evidence cards.
- Extended PostgreSQL/archive/security test coverage and kept SQLite legacy tests out of the production path.

## Operational Changes

- Runtime version is now `β 0.8.1`.
- `VERSION` remains the source of truth unless `EVOSSEARCH_APP_VERSION` overrides it.
- Existing `β 0.8.0` deployments can be upgraded without a new database migration.
- For client-side patching, use the offline patch scripts and restart `eva-ai`/`eva-ai-local-5443` after applying files.

## Verification

Last local verification before commit:

- `python -m py_compile agent.py luxriot_connector.py agent_security/eva_adapter.py`
- `node --check static/js/app.js`
- `python -m pytest -q`
- `git diff --check`

