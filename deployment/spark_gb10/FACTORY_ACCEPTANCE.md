# Spark factory acceptance

Use a fresh vendor Ubuntu 24.04 ARM64 installation. The machine may provide its
factory NVIDIA driver, Docker engine and NVIDIA Container Toolkit, but it must
not provide any EVA image, Qwen model, Python environment, database or service.

The release is not accepted by testing an operator-supplied inference endpoint.
The USB installer must load its own pinned `eva-ai/spark-runtime:0.8.7-arm64`
image, copy its own Qwen3-VL-4B and SigLIP2 weights, migrate a fresh local
PostgreSQL database, install systemd units and start both `eva-vllm` and
`eva-ai`.

For a reused laboratory Spark, stop the old inference harness before the test.
Prefer a clean Docker data root or a fresh machine. At minimum, confirm that
neither the derived EVA runtime image nor its Qwen destination exists before
starting. Do not remove vendor images from a production machine merely to run
this rehearsal.

An already loaded exact EVA image is an idempotent repair path, not a factory
acceptance pass. The acceptance log must show `docker load --input` reading the
runtime archive from this USB bundle.

Acceptance gates:

1. The installer completes with the network disconnected except for the Evo
   LAN endpoint.
2. The pinned OCI archive loads from the USB; the runtime and application
   canaries collectively prove CUDA, torchvision, OpenCV, an executable FP32
   SigLIP2 patch convolution and H.264 decoding through the isolated ffmpeg.
3. `eva-vllm.service` starts the bundled Qwen model on `127.0.0.1:1234`, and
   the installer's synthetic image smoke test returns the expected code and
   colors.
4. `eva-ai.service` reaches `/ready`, SigLIP2 loads on CUDA, and a live probe
   produces a scored frame and pulse.
5. The database is at schema `20260805_0013`, the administrator can log in,
   and both services survive a reboot without the USB attached.

The exact image ID, model revision and all critical payload checksums are bound
into `manifest.json`. A different preloaded image cannot satisfy these gates.
