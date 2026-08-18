ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.07-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="EVA AI Spark ARM64 runtime" \
      org.opencontainers.image.version="0.8.7" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/vllm:26.07-py3" \
      com.eva-ai.runtime.profile="spark-gb10-arm64"

# rootfs is an offline ARM64 ffmpeg runtime closure resolved against the exact
# NGC base image's dpkg status. No foreign-architecture code runs at build time.
COPY rootfs/ /
