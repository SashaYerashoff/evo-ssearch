# Probe Tuning

Goal: tune probes using evidence from short-term, medium-term, and broader history.

Default order:
1. Read `get_detection_summary` for the last 24h and for the target channel if known.
2. Build a representative sample with `build_research_batch`.
3. Compare recent, medium, and longer windows instead of tuning from a single moment.
4. Inspect representative detections with `get_detections` and `describe_frame`.
5. Compare the probe behavior to `get_video_summaries` if LLM summaries exist for that period.
6. Only then propose `update_probe` with `preview=true`.
7. Apply `update_probe` with `preview=false` only after explicit operator confirmation.

Tuning heuristics:
- For false positives on humans, inspect `positives`, `negatives`, and real detections before changing thresholds.
- Look for min/max ranges of `pos_score` and `margin` across the sampled windows.
- Evaluate text pairs over three horizons when possible:
  - recent: hours
  - medium: days
  - broad: the largest safe historical window available
- If data is too thin or contradictory, say so and ask the operator for a target scenario or more time range context.
