# Head Harness Guide

The head harness isolates heavy segmentation heads (Mask2Former) so we can
validate resource requirements before wiring them into `oldapp.py`.

## Quick Start

```bash
python tools/head_harness.py \
  --model facebook/mask2former-swin-base-ade-semantic \
  --image tests/data/sample.jpg \
  --summary /tmp/m2f_profile.json
```

Outputs a JSON summary detailing:
- load latency (`load_time_s`)
- CPU and GPU memory snapshots before/after model load and inference
- tensor shapes for class and mask logits to confirm dimensionality

Use `--skip-run` to measure load costs without an actual forward pass.

## Target Envelope

- GPU memory after load: ≤ 6 GB on CUDA test rig
- CPU RSS after load: ≤ 8 GB
- Forward pass (batch=1, 1024×1024) under 1.5 s

Runs exceeding these thresholds should be flagged before integration.

## Extending

- Swap `--model` to evaluate alternative heads/quantized checkpoints.
- Provide multiple `--image` paths to test batch behaviour.
- For automated QA, point `--summary` to a writable path and parse the JSON.

## In-App Integration

- `EVOSSEARCH_M2F_ENABLED=true` enables the Mask2Former refinement head exposed by this harness.
- The UI now offers a "Region threshold" slider (40–99%) to interactively tune the DINO heatmap quantile before refinement.
- When refinement is active, the results panel overlays Mask2Former masks in cyan alongside the heatmap (amber) so you can compare coarse vs. refined coverage.
