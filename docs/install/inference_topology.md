# Inference Topology & Sizing

How the models are placed and how to size them for the channel count. Site
specifics are `[FIELD]`. Variables: [config_reference](../00_CANON/config_reference.md).

## Roles

| Workload | Model | Where | Why separate |
|---|---|---|---|
| Video-description (VLM) | `qwen/qwen3-vl-4b` | Dedicated vLLM host(s) `[FIELD]` | The firehose: many batches/min across channels |
| Agent LM | `qwen3.5-9b` class | Separate endpoint `[FIELD]` | Must stay responsive during demo, not compete with the VLM firehose |
| CLIP embedding | `ViT-B/32` | App host (in-process) | Embeds every frame for search/probes; keep off the VLM GPUs |

Keep the **agent endpoint physically separate** from the VLM endpoints — otherwise
operator questions compete with the description firehose and the agent stalls
during the demo.

## Profiles

Configure explicit LM profiles (see config_reference `EVOSSEARCH_LM_*`):

```env
EVOSSEARCH_LM_PROFILES=agent,vlm
EVOSSEARCH_LM_AGENT_PROFILE_ID=agent
EVOSSEARCH_LM_VLM_PROFILE_ID=vlm
EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=<agent-host>/v1
EVOSSEARCH_LM_PROFILE_AGENT_MODEL=<agent-model>
EVOSSEARCH_LM_PROFILE_VLM_BASE_URL=<vlm-host>/v1
EVOSSEARCH_LM_PROFILE_VLM_MODEL=qwen/qwen3-vl-4b
```

## Multiple VLM hosts (balancer)

With more than one VLM host, enable the static balancer to spread channels:

```env
EVOSSEARCH_LM_VLM_BALANCER_ENABLED=true
EVOSSEARCH_LM_VLM_BALANCER_PROFILES=vlm-1,vlm-2[,...]
```

The balancer is **static channel→profile routing**, not health-aware failover. If
a VLM host dies, its channels stop being described until reassignment — watch
coverage (see [observability](../admin/observability.md)).

## Sizing the VLM load

Approximate description batch rate:

```
batches_per_min ≈ channels × 60 / (snapshot_interval_s × batch_size)
```

Each batch is `batch_size` images to the VLM. Example: 50 channels, interval 5 s,
batch 12 → ~50 batches/min. Lowering the interval multiplies the load fast
(interval 1 s → ~250 batches/min) — confirm the VLM hosts sustain the chosen rate
before scaling, or coverage gaps appear (dropped batches).

Levers when the VLM can't keep up:
- Increase `snapshot_interval` (less blind only if the host was saturated).
- Add VLM hosts + balancer profiles.
- Enable the durable inference queue + bounded worker pool (off by default;
  validate on a stand first).

## Notes

- Synchronous dispatch (default) ties each channel's capture loop to its VLM call;
  under saturation, batches are retried and may drop. The durable queue decouples
  this — see [system_architecture](../architecture/system_architecture.md).
- CLIP load scales with frame rate × channels on the app host; keep it off the VLM
  GPUs. `[FIELD]` confirm the app host has adequate CLIP throughput for the
  channel count.
