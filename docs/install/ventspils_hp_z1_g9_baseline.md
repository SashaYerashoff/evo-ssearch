# Ventspils maritime appliance baseline

This document is the sanitized, reproducible baseline for the Ventspils pilot
appliance. It intentionally excludes host identifiers, serial numbers, network
addresses, credentials, and operator data. Record a new dated measurement here
before changing the inference topology or replacing hardware.

Measured: 2026-08-04, while two live EVA channels and the production VLM were
running.

## Hardware and operating system

| Component | Measured configuration |
| --- | --- |
| Workstation | HP Z1 G9 Tower Desktop PC |
| Mainboard | HP 8AC1, KBC 12.03.27 |
| Firmware | HP U01 02.16.01, 2024-10-30 |
| CPU | Intel Core i9-14900, 24 cores / 32 threads, up to 5.8 GHz, AVX2 + AVX-VNNI, one NUMA node |
| System memory | 64 GiB installed; 61.5 GiB visible; 8 GiB swap |
| Discrete GPU | NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB, PCIe 4.0 x16, 220 W limit |
| Integrated GPU | Intel Raptor Lake-S GT1 / UHD Graphics 770 (`8086:a780`), `i915`, `/dev/dri/renderD128` |
| Storage | WD PC SN560 1 TB NVMe, ext4 root |
| Network | Intel I219-LM, `e1000e` |
| OS | Ubuntu 26.04 LTS (Resolute) |
| Kernel | Linux 7.0.0-28-generic, x86-64 |
| NVIDIA driver | 595.84 |
| FFmpeg | 8.0.1-3ubuntu2 |
| Intel media driver | Intel iHD 26.1.2, VA-API 1.23 |

## Frozen inference topology

### Live VLM on RTX 4070 SUPER

The live visual model is Qwen3-VL-4B AWQ (about 4.2 GiB on disk) served by vLLM:

```text
model alias: qwen/qwen3-vl-4b
context: 32768
GPU memory utilization: 0.72
parallel sequences: 4
max batched tokens: 4096
KV cache: FP8
execution: eager
attention backend: TRITON_ATTN
images per prompt: 16
video inputs: disabled
tool parser: hermes
```

Measured with two active channels, vLLM used about 9.5 GiB VRAM. The EVA app
embedding process used about 1.0 GiB VRAM, leaving limited but intentional
headroom on the 12 GiB card. The measured decode rate remained approximately
50 tokens/s; request queue time was effectively zero. Moving camera decode to
the Intel media engine does not make CUDA generation faster. It protects CPU
and GPU headroom as the channel count grows.

### Camera ingest on Intel media

The handoff profile must use:

```text
EVOSSEARCH_LUXRIOT_CAPTURE_SOURCE=live_segment
EVOSSEARCH_LUXRIOT_FFMPEG_HWACCEL=auto
EVOSSEARCH_LUXRIOT_FFMPEG_INTEL_DEVICE=/dev/dri/renderD128
```

On this Ubuntu build, direct QSV initialization fails with oneVPL error
`MFX session -9`. EVA therefore selects Intel VAAPI. The verified live command
uses hardware H.264 decode and `scale_vaapi` before downloading 800 px frames
at 3 fps. Per-channel hardware failure is retried once in software and is
reported in stream runtime state.

Two-channel soak result:

- both streams reported `decoder=intel_vaapi`;
- four complete live windows per stream had zero hardware fallbacks and zero
  live-segment failures;
- DRM `drm-engine-video` counters increased for both FFmpeg clients;
- combined media-engine utilization was about 1.5%, too low to be visible in
  Mission Center's rounded graph;
- FFmpeg CPU use was about 2.5-3% per active channel, compared with roughly 6%
  on the previous software path;
- NVIDIA Video Decode remained at 0%.

### Background Qwen3.5-9B-MTP on CPU

The background consolidation model is
`Qwen3.5-9B-Q4_K_M.gguf` (5,868,826,976 bytes) served by the CPU llama.cpp
build:

```text
alias: qwen3.5-9b-mtp
context: 65536
GPU layers: 0
flash attention: on
KV cache: Q8_0 K and V
slots: 1
continuous batching: on
decode threads: 12
batch threads: 16
MTP draft: enabled, maximum 4 tokens
```

The idle service reserves about 10.0-10.6 GiB RSS. Benchmarks were taken while
EVA, two Intel-media camera streams, and the RTX VLM remained active:

| Workload | Prompt | Output | Prompt rate | Generation rate | MTP acceptance | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm concise request | 40 tokens | 11 tokens | 39.22 tok/s | 9.81 tok/s | 8/16 (50.0%) | 2.14 s |
| Representative consolidation | 3,235 tokens | 128 tokens | 39.74 tok/s | 10.95 tok/s | 93/135 (68.9%) | 93.09 s |
| Long consolidation, stopped | 8,192 processed | not reached | 29.50 tok/s at stop | not measured | not measured | more than 277.66 s of prefill |

Operational conclusion: the CPU 9B profile is useful for bounded background
L2/L3 consolidation, operator-feedback review, and probe tuning. It is not an
interactive agent at long context. Run it only through one-slot admission,
prefer an operator-configured quiet window, allow cancellation/preemption, and
compact evidence before sending it. A quiet window is a budget preference, not
a security blind spot: urgent live VLM work remains on the RTX model.

## Eight-channel focus contract

- Target eight independently regulated channels, not constant full-rate VLM
  processing on every stream.
- Preserve one semantic archive embedding per configured second independent of
  alerts.
- Decode and resize on Intel media; keep CV motion/quality/homeostasis on CPU.
- Build L0 batches from CV apexes and control frames, at most 16 frames and at
  most 60 seconds, with faster closure during meaningful activity.
- Keep the live VLM admission queue bounded and preserve per-channel fairness.
- Use probes for low-latency focused bookmarks; use VLM alerts for richer
  episode understanding.
- Keep L1/L2 on the interactive agent profile under admission control. Route
  heavy L3/operator-feedback/probe consolidation to CPU 9B only when its quiet
  window and resource budget allow.
- Treat PTZ movement as a scene-transition/perception reset signal, not as a
  security event by itself.
- SigLIP2 remains a separately budgeted embedding decision. Re-measure its VRAM
  and archive epoch before changing the production embedder.

## First diagnostics when the pilot degrades

1. Confirm `eva-ai`, `eva-vllm`, `eva-deep-review`, and PostgreSQL are active.
2. Inspect `/luxriot/streams`: decoder, fallback count, failed windows, pending
   frames, summary age, and queue depth per channel.
3. Check FFmpeg arguments for `/dev/dri/renderD128`, `-hwaccel vaapi`, and
   `scale_vaapi`; check `/proc/<ffmpeg-pid>/fdinfo/*` for increasing
   `drm-engine-video` counters.
4. Check vLLM running/waiting requests, TTFT, generation rate, and VRAM before
   changing batching.
5. Check the CPU 9B slot before starting L3. A busy 9B request can consume about
   14 CPU threads during long prefill.
6. Compare the source revision and active environment with this baseline before
   tuning thresholds or changing models.

