#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -o pipefail
#
# Final base-AMI smoke test: GPU passthrough into a Docker container.
#
# Catches driver / nvidia-container-toolkit / Docker misconfiguration
# at AMI-build time instead of at customer instance launch. If this
# step fails, the AMI is not produced.

# CUDA-13 / Blackwell driver floor (mirrors install-base.sh).
DRIVER_FLOOR=580

echo "=== kumo-base smoke: nvidia-smi inside container ==="
# Check the host driver before Docker GPU passthrough.
uname -r
modinfo nvidia            # fails if no module exists for the booted kernel
sudo modprobe nvidia
nvidia-smi                # host-level driver check (not just in-container)

# Enforce the CUDA-13 / Blackwell driver floor. The build box may be a T4
# (Turing), where an old driver (e.g. 575) still inits and nvidia-smi
# succeeds – but 575 fails GSP init on Blackwell (sm_120) and silently falls
# back to CPU. Assert the driver major here so a bad DLAMI snapshot fails the
# build instead of shipping a CPU-only-on-Blackwell image.
drv_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1 | tr -d '[:space:]')
if ! printf '%s' "${drv_major}" | grep -qE '^[0-9]+$'; then
    echo "FATAL: could not read a numeric NVIDIA driver version (got '${drv_major}'); refusing to build."
    exit 1
fi
if [ "${drv_major}" -lt "${DRIVER_FLOOR}" ]; then
    echo "FATAL: host NVIDIA driver ${drv_major}.x is below the CUDA-13 floor (>= ${DRIVER_FLOOR}); refusing to build."
    exit 1
fi
echo "driver floor OK: ${drv_major}.x >= ${DRIVER_FLOOR}"
# Record the full driver version baked into this AMI for audit/rollback
# (the kept DLAMI driver varies per snapshot under most_recent).
sudo mkdir -p /etc/kumo
nvidia-smi --query-gpu=driver_version --format=csv,noheader \
  | head -1 | sudo tee /etc/kumo/nvidia-driver-version
sudo docker run --rm --gpus all \
    nvidia/cuda:13.0.2-base-ubuntu22.04 \
    nvidia-smi
echo "=== smoke OK ==="

# Pull is large-ish; rmi so it doesn't bloat the AMI snapshot.
sudo docker rmi nvidia/cuda:13.0.2-base-ubuntu22.04
