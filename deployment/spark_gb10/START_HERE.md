# EVA AI for NVIDIA GB10 / ARM64

This package targets an Ubuntu 24.04 ARM64 Spark-class appliance whose NVIDIA
driver, Docker engine and NVIDIA Container Toolkit are supplied by the machine
vendor.

That is the only factory prerequisite. No Python ML environment, inference
image, model cache, PostgreSQL database or EVA service is assumed to exist.
The installer stops before application mutation if the vendor GPU/container
bridge is absent; it never replaces the GB10 kernel or NVIDIA driver.

The installer preserves that vendor stack. It does not install an Ubuntu HWE
kernel or a desktop NVIDIA driver. EVA and local SigLIP2 run in a separate
container based on the immutable ARM64 image
`eva-ai/spark-runtime:0.8.7-arm64`, derived from the pinned NVIDIA
`nvcr.io/nvidia/vllm:26.07-py3` base with the offline ffmpeg runtime added. The
VLM endpoint remains a separate service.
The USB always carries and verifies that image offline. Factory acceptance is
performed with no EVA image or model preloaded in Docker; an exact matching
image may only be reused during an idempotent field repair.

Before installation, the launcher verifies:

- `aarch64` / `arm64` and Ubuntu 24.04;
- the offline ARM64 APT repository and Python wheelhouse;
- the bundled ARM64 runtime archive and its exact pinned image identity;
- CUDA-enabled `torch`, `torchvision`, OpenCV and `ffmpeg` inside that image;
- a visible NVIDIA device through `docker run --gpus all`;
- the selected VLM model and a real image-understanding smoke test.

Start with:

```bash
sudo ./START_EVA_AI.sh
```

For a non-interactive factory installation using the bundled local VLM:

```bash
sudo ./START_EVA_AI.sh --mode install -- \
  --non-interactive \
  --evo-url http://EVO-IP:PORT \
  --evo-username admin \
  --evo-password-file /root/evo-password \
  --no-deep-review \
  --admin-username admin \
  --admin-password-file /root/eva-admin-password
```

The installer never installs Python ML packages into the host interpreter. It
loads the pinned image, creates its own `eva-vllm` inference service from the
bundled Qwen weights, then creates a separate `eva-ai-app` container. Those
services share only immutable image layers and the GB10 GPU.

The password files must contain only the corresponding password and should be
mode `0600`.

For release acceptance on a clean or reused laboratory machine, follow
`repo/deployment/spark_gb10/FACTORY_ACCEPTANCE.md`.
