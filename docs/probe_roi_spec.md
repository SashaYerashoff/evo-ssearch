# Probe ROI Spec (Design, No Code)

Last updated: 2026-02-24
Branch: `vu-improvements`

## Objective
Add a simple, reliable ROI (Region of Interest) mode for probes so probe scoring focuses on a user-selected area while detections archive still keeps full-frame context for later research.

## Why ROI
- Improve probe precision by reducing background noise.
- Improve operator trust: probe intent is visible and explicit.
- Keep post-hoc analysis quality by preserving full snapshots.

## Critique / Risks
- ROI does not materially reduce CLIP compute cost (model input is resized); quality benefit is the primary gain.
- Per-probe ROI can reduce embedding reuse when many probes run on one channel.
- Biggest source of bugs is coordinate mismatch between UI draw box and source frame (scaling/letterboxing).
- Fixed ROI can degrade over time with PTZ/moving camera scenes.

## Design Principles
- Keep v1 minimal: one rectangle ROI per probe.
- Full-frame archive remains primary truth.
- Probe decision path and archive/search path are intentionally separated.
- Fallback-safe: invalid ROI never blocks probe execution.

## UX Scope (Probe Settings Modal)
Add ROI controls to live preview section:
- `ROI OFF/ON` toggle.
- `Draw ROI` action when ON.
- `Clear ROI` action.
- Overlay rectangle rendered on live preview.

Behavior:
- ON + Draw: user drags one bbox.
- Re-draw replaces prior bbox.
- If no bbox exists while ON, probe behaves as full-frame and shows "ROI not set" state.

Operator affordances:
- Show normalized area summary (e.g. `x=0.18 y=0.24 w=0.31 h=0.42`).
- Show ROI enabled badge on probe card.

## Data Model Changes (Probe)
Add optional fields in probe payload/store:
- `roi_enabled: bool`
- `roi_norm: { x: float, y: float, w: float, h: float } | null`  # normalized [0..1]
- `roi_updated_at: epoch_ms | null`

Validation:
- `0 <= x,y,w,h <= 1`
- `w > 0`, `h > 0`
- clamp to frame bounds when denormalizing.

## Runtime Inference Flow
Per frame, per probe:
1. Decode/capture full frame as today.
2. If `roi_enabled && roi_norm`:
- Convert normalized ROI to pixel box on current frame.
- Apply small padding (configurable, default 5%).
- Crop ROI patch.
- Score probe using ROI crop embedding.
3. Else score using full frame embedding.
4. For persisted hit: always keep full-frame snapshot in archive.

## Archive / Search Behavior
Required:
- Persist full snapshot as today.
- Persist ROI metadata with hit row for audit/explainability.

Optional for UX:
- Persist tiny ROI thumbnail for quick preview.

Important:
- Do not require dual embeddings per frame in v1.
- If later needed, store full-frame embedding lazily only for persisted hits.

## API / Payload Adjustments
Endpoints impacted:
- `POST /probes/save` (or equivalent save path)
- `GET /probes/list`
- `POST /probes/run`

Probe payload additions:
```json
{
  "roi_enabled": true,
  "roi_norm": { "x": 0.18, "y": 0.24, "w": 0.31, "h": 0.42 }
}
```

Detection row additions (response payload):
```json
{
  "roi_enabled": true,
  "roi_norm": { "x": 0.18, "y": 0.24, "w": 0.31, "h": 0.42 },
  "roi_px": { "x": 224, "y": 161, "w": 386, "h": 282 }
}
```

## Performance Notes
- Expected quality gain > throughput gain.
- Throughput impact depends on probe count/channel:
  - low probe count: negligible
  - many distinct ROIs: extra crop+embed overhead
- Keep v1 simple; optimize only if measured bottleneck appears.

## Edge Cases
- ROI outside frame after resolution changes: clamp and continue.
- Tiny ROI (`< min pixels`): reject during validation or fallback to full frame.
- Rotated/letterboxed UI preview: map pointer to true content rect before normalization.
- Missing ROI while enabled: non-fatal warning + full-frame fallback.

## Rollout Plan
1. Schema/payload support (no inference changes yet).
2. UI draw + save/load ROI.
3. Runtime crop scoring path.
4. Detection metadata surfacing in archive card/expanded view.
5. Optional ROI thumbnail enhancement.

## Test Plan
Functional:
- Save probe with ROI, reload page, ROI persists and draws correctly.
- ROI ON with bbox changes probe hit behavior relative to OFF.
- Archive hit still opens full-frame image.

Correctness:
- Pointer-to-frame mapping matches expected pixels for non-square preview.
- ROI clamping works at borders.

Regression:
- Existing probes without ROI behave exactly as before.
- Search in archive unaffected for non-ROI probes.

## Out of Scope (v1)
- Polygon/freeform ROI.
- Auto-tracking/dynamic ROI.
- Multi-ROI per probe.
- ROI-specific FAISS/index partitioning.
