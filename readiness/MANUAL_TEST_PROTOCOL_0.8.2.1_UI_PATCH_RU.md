# EVA AI β 0.8.2.1 - manual UI patch test protocol

Audience: PM, field intern, engineer/operator tester.  
Build under test: `β 0.8.2.1`.  
Scope: UI fixes after the β 0.8.2 office-demo manual pass.

## 0. Rules

- Test in Chrome/Firefox **and** in the Luxriot EVO Monitor embedded web tile if
  available.
- Before testing, hard-refresh the browser or reload the EVO web tile. This patch
  changes `app.js`, `app.css`, and `templates/index.html`.
- Complex agent calls may take several minutes. Do not mark a long answer as
  failed until the stream finishes or a server error is shown.
- Save screenshots for every WARN/FAIL.

Use:

- **PASS** - behavior matches expected result.
- **WARN** - usable, but wording/layout is confusing or slow.
- **FAIL** - misleading, broken, clipped, non-clickable action hidden, or visual
  evidence claimed without an image.

## 1. Version And Service

1. Open EVA AI.
2. Check `/health` or UI version.

Expected:

- Version is `β 0.8.2.1`.
- Login/session works.
- Main tabs load.

Result: PASS / WARN / FAIL  
Notes:

## 2. Agent Probe Preview/Apply Is Outside Research Trace

Goal: the operator must not need to expand Research trace to apply a probe
preview.

Prompt example:

```text
Review probe P/N/M for one existing CLIP probe on channel <channel>, suggest a safer threshold update, and create preview only.
```

Steps:

1. Wait for the agent to finish.
2. Find the probe preview card.
3. Confirm where the Apply button is rendered.
4. Click Apply if the preview is safe and intentional.

Expected:

- The probe preview appears as a standalone approval card in the assistant
  message body.
- The card is **outside** the collapsible Research trace.
- Research trace may contain research/tool steps, but it does not hide the Apply
  button.
- Clicking Apply changes the card/receipt state and does not require another
  chat message.
- The agent does not claim the change was applied before the UI Apply receipt.

FAIL if:

- Apply is only visible inside Research trace.
- Research trace is collapsed and hides the only Apply button.
- The agent says the probe was applied without a receipt.
- The tool calls `create_probe` or `update_probe` with direct apply semantics
  instead of preview/action-plan approval.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 3. Video Summary Machine JSON Labels

Goal: machine blocks should be understandable at a glance.

Steps:

1. Open Video descriptions / VLM feed.
2. Select a channel with recent L0 summaries.
3. Find entries with collapsed machine JSON blocks.

Expected:

- Short non-alert JSON is labeled `System message`, not generic `Machine JSON`.
- Alert JSON is labeled by the alert title or a clear alert event label.
- Memory/homeostasis/baseline messages are labeled `Memory/homeostasis`.
- Raw JSON remains accessible for admin/engineer users.
- The label does not cover or truncate the normal summary text.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 4. Monitor Probe Board And Selected Probe Panel

Goal: probe cards should be stable under resize and the old selected-probe
filmstrip should be gone.

Steps:

1. Open Monitor / CLIP Probes.
2. Resize the browser/window across narrow and wide widths.
3. Select several probe cards.
4. Inspect the right panel.

Expected:

- Probe cards do not overlap vertically.
- Play/stop and edit controls stay aligned at the top-right of each card.
- Delete stays in the bottom/right action area and is not clipped.
- Long probe names wrap/clamp cleanly.
- The selected-probe panel shows current state, scores, bookmark gate, text
  pairs, image-probe status, and direct action buttons.
- The old `Latest CLIP Hits` filmstrip is not visible in this monitor inspector.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 5. Agent Search Results With Missing Images

Goal: metadata-only rows must not look like visual evidence.

Prompt example:

```text
Search archive for a visual event that returns mixed probe/VLM detections and provide candidate frames.
```

Steps:

1. Run an archive/probe search that returns detection cards.
2. Inspect thumbnails.
3. Click available thumbnails.

Expected:

- Rows with real images show thumbnails and can open in the lightbox.
- Rows without image data show `No image #<id>` or a clear non-image tile.
- Missing-image tiles are not visually presented as confirmed frame evidence.
- Broken image icons should not remain visible after load failure.
- If the agent tries to describe a metadata-only detection, it should report that
  no image was available instead of claiming visual confirmation.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 6. Archive Review Modal Smoke

Goal: the previous archive modal fixes still hold.

Steps:

1. Open Archive search.
2. Open a result with a valid image.
3. Use Review / Describe frame / Find similar / Open VLM feed.

Expected:

- Modal uses the current shared styling.
- Frame is fully visible and not vertically cut in half.
- Summary pane scrolls with styled scrollbar.
- Open VLM feed jumps near the selected timestamp.
- Buttons do not overlap or leave the modal bounds.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 7. Final Summary

| Area | Result | Notes |
|---|---|---|
| Version/service |  |  |
| Agent probe approval outside trace |  |  |
| Machine JSON labels |  |  |
| Monitor probe board |  |  |
| Missing-image handling |  |  |
| Archive review modal |  |  |

Overall: PASS / WARN / FAIL

