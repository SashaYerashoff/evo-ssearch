ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.07-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="EVA AI Spark ARM64 runtime" \
      org.opencontainers.image.version="0.8.7" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/vllm:26.07-py3" \
      com.eva-ai.runtime.profile="spark-gb10-arm64"

# rootfs carries an isolated Ubuntu 24.04 ARM64 ffmpeg dependency closure and a
# pinned launcher. It does not reuse the NGC image's pruned package inventory or
# replace its vendor CUDA/Python libraries. No foreign-architecture code runs
# at image build time.
COPY rootfs/ /
