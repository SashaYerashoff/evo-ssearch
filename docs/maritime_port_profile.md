# Maritime port profile

The maritime profile is an EVA Protocol Deploy extension for up to eight
homeostatically regulated port-gate, fairway, and coastline channels. It
narrows operator attention; it is not vessel registration, legal evidence, or
a replacement for radar/AIS and collision-avoidance systems.

## Runtime path

1. Dense CV continues to decode the configured 4–6 frames/s and computes
   lightweight motion/quality signals.
2. Global optical flow is split into translation and radial zoom components.
   EVA classifies the camera as steady, pan, tilt, zoom, preset cut, or
   settling.
3. Stable views receive compact illumination-tolerant fingerprints. Repeated
   views become per-channel `preset_id` values; a transition increments the
   `scene_epoch`.
4. While a PTZ camera moves, settles, or enters a new unconfirmed view, road-CV
   cues and scene-specific SigLIP2 probes cannot regulate attention. The one-Hz
   semantic snapshot archive continues independently.
5. `VECTOR_SIGNALS_JSON.camera_scene` reaches L0 as non-evidentiary routing
   metadata. L0 must verify current images and report uncovered areas as not
   observed.

## Deployment roles

- `maritime_gate`: passages, stopped/lingering vessels in gate or fairway,
  close approaches, and small craft near large vessels.
- `maritime_coast`: shore contacts, nearshore loitering, visible distress, and
  unexpected coastline activity.
- `maritime_mixed_ptz`: a PTZ tour that may contain several port/coast views.

Protocol Deploy stores an optional operator location card, but never guesses
the port from image overlays. Role starter policies install at most four
probes per channel, with `attention_only=true`, bookmarks off, and conservative
P/N/M defaults. The first commissioning pass uses independent semantic
snapshots and produces review-only threshold/cadence proposals.

## Prompt hierarchy

- L0 establishes camera coverage first, then coarse visible vessel class and
  episode continuity. Camera motion is never vessel motion.
- L1 preserves short maritime episodes and separates event time from camera
  motion/unavailable time.
- L2 distinguishes routine traffic from material deviations without erasing
  coverage gaps.
- L3 produces a chronological eight-hour operational account and audits
  operator feedback/probe drift as proposals only.

## UI localization

React UI chrome supports English and Latvian from Appearance. The setting is
stored per browser and changes stable operator-control labels only. EVA does
not machine-translate evidence, channel names, VLM summaries, agent messages,
incident reports, or operator text. Internal prompt semantics remain English
for the deployed 4B head.

## Agent grammar ownership

No new agent tool was added. The existing composite Protocol Deploy tools own
the extension:

- `C`: the closed `general|maritime` profile, operator-selected channel IDs,
  and closed maritime role enums;
- `CAL`: the bounded channel survey and later independent shadow calibration;
- `MUT`: the composite preview/apply receipt for prompts, probes, counters,
  groups, and quiet window.

The model still receives no authority to invent channel IDs, arbitrary time
windows, thresholds, or direct live mutations. The extension therefore does
not conflict with the pinned Tuktuk grammar.
