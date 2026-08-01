# Inference Topology & Sizing

How the models are placed and how to size them for the channel count. Site
specifics are `[FIELD]`. Variables: [config_reference](../00_CANON/config_reference.md).

## Roles

| Workload | Model | Where | Why separate |
|---|---|---|---|
| Video-description (VLM) | `qwen/qwen3-vl-4b` | Dedicated vLLM host(s) `[FIELD]` | The firehose: many batches/min across channels |
| Agent LM | `qwen3-vl-4b` in the constrained eight-channel profile; optional 9B-class scale-out | Shared protected vLLM profile or separate endpoint `[FIELD]` | Small-head tool use stays bounded by composite workflows and admission |
| Deep L3 review | optional 9B-class model | Separate endpoint, often CPU/offloaded `[FIELD]` | Proposal-only consolidation inside an operator-defined quiet window |
| Semantic embedding | `google/siglip2-base-patch16-224` (CLIP retained for A/B) | App process, GPU in the single-4070S profile | Embeds every archived cadence frame for search/probes; exact model revision defines an isolated vector epoch |

For the single-4070S eight-channel appliance, agent and VLM may intentionally
share the same Qwen3-VL-4B endpoint. EVA's admission controller reserves a
protected agent/alert/rollup slot; L0 may borrow it only while protected work is
not waiting. At larger scale, separate the agent endpoint so operator questions
do not compete with the description firehose.

The appliance starts vLLM at `gpu_memory_utilization=0.75`, leaving headroom for
the in-process FP16 SigLIP2 base encoder. CPU SigLIP2 is a degraded fallback and
cannot sustain eight channels at the default one embedding per second cadence.

## Profiles

Configure explicit LM profiles (see config_reference `EVOSSEARCH_LM_*`):

```env
EVOSSEARCH_LM_PROFILES=agent,vlm
EVOSSEARCH_LM_AGENT_PROFILE_ID=agent
EVOSSEARCH_LM_VLM_PROFILE_ID=vlm
EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=<agent-host>/v1
EVOSSEARCH_LM_PROFILE_AGENT_MODEL=qwen/qwen3-vl-4b
EVOSSEARCH_LM_PROFILE_VLM_BASE_URL=<vlm-host>/v1
EVOSSEARCH_LM_PROFILE_VLM_MODEL=qwen/qwen3-vl-4b
EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS=65536
```

For the full configured budget, the agent inference server must expose at least
the same 65,536-token context. For llama.cpp use `-c 65536`; for vLLM set the
equivalent max model length and confirm `/v1/models` reports `meta.n_ctx` or
`max_model_len` at or above 65536. When `/v1/models` reports a smaller
`max_model_len`, EVA automatically lowers its warning, compaction, and hard-stop
budgets to that served limit. This keeps long tool workflows safe but does not
create additional server context; changing only the EVA environment never
enlarges the model server context.

Optional 9B deep L3 is configured through
`EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_*`; it is not the live agent fallback and
does not run outside the persisted quiet-window and attention gates.

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
