# EVA AI offline appliance compatibility contract

This document defines the host contract for the EVA AI 0.8.7 offline bundles.
The installer is not expected to support arbitrary Linux, Python, Docker, GPU,
or locally modified service layouts. A host is supported when it matches a
listed platform and passes the installer's capability canaries.

## Supported fresh-install targets

| Component | x64 appliance | NVIDIA GB10 / Spark-class ARM64 appliance |
| --- | --- | --- |
| Architecture | `amd64` / `x86_64` | `arm64` / `aarch64` |
| Operating system | Ubuntu 24.04 LTS or Ubuntu 26.04 LTS; use the x64 bundle built for the exact host release | Ubuntu 24.04 LTS |
| Host Python | Ubuntu system Python 3.12 on 24.04 or Python 3.14 on 26.04; no host ML packages required | Ubuntu 24.04 system Python 3.12; no host PyTorch, vLLM, or Transformers required |
| GPU | Modern NVIDIA GPU with at least 12 GiB VRAM | NVIDIA GB10 or an OEM Spark-class GB10 equivalent |
| Memory | Workload-dependent; 32 GiB RAM recommended | 120 GiB unified memory class |
| Free disk before install | At least 48 GiB; installer may require more from the actual payload size | At least 70 GiB; more is recommended for diagnostics, backups, and archive growth |
| Local VLM | Bundled Python runtime and Qwen3-VL-4B AWQ/vLLM, or an operator-selected external endpoint | Bundled pinned NVIDIA runtime container and Qwen3-VL-4B FP8/vLLM, or an operator-selected external endpoint |
| Semantic probes | Local SigLIP2 on CUDA | Local SigLIP2 FP32 on CUDA inside the pinned container runtime |

Fresh installation on other Linux distributions, Ubuntu interim releases,
WSL, Docker Desktop, rootless Docker, generic ARM servers without GB10, or
unlisted CPU architectures is unsupported.

The x64 Ubuntu 24.04 and 26.04 bundles are separate release artifacts because
their offline APT closures are distribution-specific. They contain the same
EVA release and update compatibility, but they are not interchangeable for a
fresh installation. The launcher verifies this before changing the host.

## ARM64 Docker and NVIDIA contract

The ARM64 installer relies on capabilities, not on one accidental Docker patch
version. The supported daemon must provide all of the following:

- a system-wide rootful Docker Engine controlled by `docker.service`;
- `docker load`, `docker image inspect`, bind mounts, host networking, host IPC,
  custom entrypoints, and `--gpus all`;
- NVIDIA Container Toolkit integration and a GB10 device visible inside a
  container;
- support for the bundle's zstd-compressed OCI image archive;
- either the classic Docker image store, where `.Id` is the image-config digest,
  or the containerd image store, where `.Id` may be the OCI-manifest digest.

Release acceptance covers Docker Engine 27 through 29 when these capability
checks pass. Docker 29.2.1 with `io.containerd.snapshotter.v1` is an explicit
test case. A newer major Docker release is not automatically certified: add it
to the acceptance matrix after the same load, identity, CUDA, media, and reboot
canaries pass.

Do not disable the containerd image store or edit `/etc/docker/daemon.json` to
make EVA install. EVA must adapt to either legitimate image-ID representation.
Do not replace the vendor kernel, NVIDIA driver, Docker engine, or NVIDIA
Container Toolkit during installation.

The ARM runtime canary must prove, from the pinned image that will actually be
used by systemd:

1. the image architecture is ARM64 and its identity matches the release's
   pinned config or manifest digest;
2. CUDA is visible and reports an NVIDIA GB10 device;
3. PyTorch, torchvision, vLLM, and the pinned NumPy runtime import;
4. an FP32 SigLIP2-shaped CUDA convolution executes;
5. the isolated FFmpeg runtime has an H.264 decoder.

## Update compatibility

The x64 0.8.7 universal bundle can update supported EVA installations on Ubuntu
24.04 or 26.04 whose existing EVA environment uses CPython 3.12, 3.13, or 3.14.
An update is authorized by the updater's read-only preflight, database
visibility guard, backup, schema path, service identity, and post-update report;
it is not authorized merely because the host has a compatible Python version.

The ARM64 resume hotfix for bundle `9063d2f` is not a general updater. It is
only for a fresh installation whose journal stopped at
`spark_container_runtime` after loading the exact pinned ARM image.

## Network and service prerequisites

- Luxriot Evo is reachable from the EVA host and its HTTP API credentials are
  known. A bare address uses port 8080.
- DNS is required only when names rather than IP addresses are entered.
- Correct system time, timezone data, and active time synchronization are
  required for TLS, audit timestamps, and incident chronology.
- Port 443 must be available for the operator HTTPS endpoint.
- EVA binds its application service to loopback port 5000 behind nginx.
- A bundled local VLM binds to loopback port 1234.
- The local PostgreSQL service and Unix socket must be available.
- External inference endpoints, when selected, must be reachable from the EVA
  host and expose an OpenAI-compatible `/v1` API. The VLM must accept up to
  eight images per request.

## Host-state prerequisites and exclusions

- Run the installer with `sudo` from the complete extracted bundle directory.
- Do not extract the bundle onto a filesystem that destroys symlinks or Unix
  executable bits.
- A fresh host must not already contain an EVA database, EVA systemd service,
  EVA application tree, or conflicting partial deployment. A recognized
  installer journal may resume the exact interrupted bundle.
- Existing unrelated Docker images and containers are permitted, but the host
  must have enough disk for the 45.7 GB ARM runtime image and the remaining EVA
  payload.
- Existing nginx, PostgreSQL, firewall, proxy, hardening, or endpoint-security
  policies must be reviewed for conflicts. The installer does not promise to
  merge arbitrary local service customizations.
- SELinux-enforcing hosts, nonstandard AppArmor confinement, read-only system
  paths, custom Docker data roots with insufficient space, and corporate TLS
  interception are outside the tested factory profile unless separately
  accepted.

## Acceptance boundary

An installation is complete only when the deployment report confirms the EVA
service, React UI, schema revision, Luxriot reachability, inference readiness,
and semantic runtime. ARM acceptance additionally requires one real VLM vision
result, one live SigLIP2 probe with a scored frame, and a reboot test without
the USB media attached.

If a capability fails, preserve the installer journal and generate the bug
report package. Do not edit generated environment files, Docker daemon settings,
database roles, or systemd units as an undocumented workaround.
