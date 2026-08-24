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

The appliance starts vLLM at `gpu_memory_utilization=0.85` with four concurrent
sequences and a hard limit of eight images per request. EVA's L0 attention
selector submits at most eight representative images even when a legacy or
operator-selected source batch is larger. A same-second sharper companion is
attached only when an image slot remains. EVA also counts the fully constructed
request and rejects it locally if it exceeds eight images, so the endpoint never
has to truncate it silently. On the measured 16 GiB RTX 5060 Ti profile this exposed
121,088 FP8 KV-cache tokens (3.7 full 32k sequences) while leaving about 2.3 GiB
outside vLLM for the in-process SigLIP2 encoder and CUDA/runtime variance. The
same bounded profile is intended for 12+ GiB cards, but site acceptance must
still verify the exact driver/vLLM combination. CPU SigLIP2 is a degraded
fallback and cannot sustain eight channels at the default one embedding per
second cadence. EVA admission remains lower than the vLLM sequence limit so
agent and rollup work cannot be buried by one synchronized L0 wave.

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

For the local appliance profile, `eva-vlm-vision-watchdog.timer` sends a fresh
control image every minute. Unlike `/health`, it verifies that the multimodal
encoder perceives new ordered visual facts. The first mismatch quarantines L0
only when no recent successful canary exists; otherwise it remains a visible
`suspect` warning. A second consecutive mismatch quarantines L0 and triggers a
bounded `eva-vllm` restart. Stale watchdog state also fails closed.
The appliance runs the Qwen3-VL vision encoder with `FLASH_ATTN` and disables
the multimodal processor cache because live video frames are unique.

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

## Multiple VLM hosts (assignment planner)

With more than one VLM host, enable Auto routing across the declared profiles:

```env
EVOSSEARCH_LM_VLM_BALANCER_ENABLED=true
EVOSSEARCH_LM_VLM_BALANCER_PROFILES=vlm-1,vlm-2[,...]
EVOSSEARCH_LM_PROFILE_VLM_1_MAX_INFLIGHT=4
EVOSSEARCH_LM_PROFILE_VLM_2_MAX_INFLIGHT=1
```

Auto is a **session assignment**, not per-request round robin. At stream start
EVA compares the profiles' configured/served capacity, the steady demand of
already assigned streams (`~1 / snapshot_interval_s` images/s), current
admission `active + queued`, and cached `/models` health. Equal projected loads
use a deterministic channel/profile tie-break. The chosen profile is persisted
and remains stable so one channel's L0 continuity does not bounce between model
servers.

The Stream settings UI shows the selector separately from the actual assigned
profile. Choosing a profile creates a durable manual pin. Choosing Auto permits
EVA to re-plan that channel on a later restart. Desired sessions written by an
older EVA do not contain the original selector, so the upgrade treats their
existing profile as `legacy pinned` rather than silently redistributing a site.
An operator must explicitly change those channels to Auto.

This is not in-flight failover: if a host dies after a session was assigned, its
channels stop being described until restart/reassignment. Watch coverage (see
[observability](../admin/observability.md)).

## Sizing the VLM load

Approximate description batch rate:

```
batches_per_min ≈ channels × 60 / (snapshot_interval_s × batch_size)
```

The capture window contains `batch_size` saved frames. The VLM sees at most
between one and `EVOSSEARCH_LUXRIOT_VLM_MAX_IMAGES_PER_REQUEST`
chronologically ordered evidence images. CV/SigLIP still process the wider
upstream capture window; the L0 record retains the source/selected/omitted
counts and reserves a slot for a useful stable companion of a CV apex, while the independent
per-second semantic snapshot archive retains its own configured cadence. When
12/16-frame windows are compressed, both the prompt and UI explicitly say that
VLM visual coverage is partial.

The fresh-install default is a 12-sample temporal window at 1 Hz: a non-empty
packet seals by about 12 s without waiting to fill the image budget. A quiet
heartbeat may therefore contain only one or two images, while a busy event may
use all eight. The absolute 50-channel ceiling is about 250 routine packets/min
before event coalescing; image and token cost varies with evidence density.
Confirm the four Georgia VLM endpoints sustain the measured p95 service time;
otherwise use per-channel policy and additional hosts rather than accepting
growing queues or coverage gaps.

Levers when the VLM can't keep up:
- Increase `snapshot_interval` (less blind only if the host was saturated).
- Use batch 12/16 only as an explicitly partial, attention-compressed window;
  it is not exhaustive VLM inspection of every saved frame.
- Add VLM hosts + Auto-routing profiles with truthful per-profile capacity.
- Enable the durable inference queue + bounded worker pool (off by default;
  validate on a stand first).

## Notes

- Synchronous dispatch (default) ties each channel's capture loop to its VLM call;
  under saturation, batches are retried and may drop. The durable queue decouples
  this — see [system_architecture](../architecture/system_architecture.md).
- CLIP load scales with frame rate × channels on the app host; keep it off the VLM
  GPUs. `[FIELD]` confirm the app host has adequate CLIP throughput for the
  channel count.
