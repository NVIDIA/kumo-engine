#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${FAISS_GPU_AMD64_OUTPUT_DIR:-$REPO_ROOT/3rdparty/kumo-wheels}"
CONTAINER_CLI="${CONTAINER_CLI:-docker}"
BUILDER_DIR="$REPO_ROOT/tools/faiss-gpu-amd64"
# TODO: Key cached wheels by host architecture before sharing this output
# directory between x86_64 and ARM64 builders; the current manylinux glob can
# reuse a wheel produced for the other architecture.
WHEEL_GLOB='faiss_gpu-1.14.3+cu130sm120-cp314-cp314-manylinux*.whl'

mkdir -p "$OUTPUT_DIR"
if find "$OUTPUT_DIR" -maxdepth 1 -name "$WHEEL_GLOB" -print -quit | grep -q .; then
    echo "Reusing CUDA 13 sm_120 Faiss wheel from $OUTPUT_DIR"
    exit 0
fi

CUDA_ARCHES="${FAISS_GPU_AMD64_ARCHES:-}"
if [ -z "$CUDA_ARCHES" ]; then
    # The public PyPI faiss-gpu wheel ships cubins for sm_80/86/89/90/100 only.
    # A GPU outside that set - Tesla T4 (sm_75) or workstation Blackwell (sm_120) -
    # needs our source-built override, or faiss GPU calls fail with CUDA error 209
    # ("no kernel image is available"). GPUs the public wheel already covers skip
    # the build. Capture-then-match (no `... | grep -q`) to stay correct under
    # `set -o pipefail`, where grep -q closing the pipe early can fail the pipeline.
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
    # CUDA 13 requires Turing or newer (compute cap >= 7.5). Refuse pre-Turing
    # (e.g. V100 sm_70) rather than build a wheel that can't run on the GPU.
    if [ -n "$cc" ] && [ "$cc" -lt 75 ] 2>/dev/null; then
        echo "ERROR: CUDA 13 requires a Turing-or-newer GPU (compute cap >= 7.5); sm_${cc} is unsupported." >&2
        exit 1
    fi
    case " 80 86 89 90 100 " in
        *" ${cc} "*)
            echo "Public faiss-gpu already covers sm_${cc}; no override needed."
            exit 0 ;;
    esac
    # T4, workstation Blackwell, or an undetected GPU: build the full fleet arch set
    # so one wheel covers every GPU we run (native cubins sm_75 through sm_120).
    CUDA_ARCHES="75;80;86;89;90;100;120"
fi

if ! command -v "$CONTAINER_CLI" >/dev/null 2>&1; then
    echo "ERROR: $CONTAINER_CLI is required to build the CUDA 13 sm_120 Faiss wheel" >&2
    exit 1
fi

if [ ! -f "$BUILDER_DIR/build.Dockerfile" ]; then
    echo "ERROR: $BUILDER_DIR does not contain the CUDA 13 Faiss builder" >&2
    exit 1
fi

IMAGE="kumo/faiss-cu130-builder:local"
"$CONTAINER_CLI" build -t "$IMAGE" -f "$BUILDER_DIR/build.Dockerfile" "$BUILDER_DIR"

VOLUME_SUFFIX=""
if command -v getenforce >/dev/null 2>&1 && [ "$(getenforce 2>/dev/null)" = "Enforcing" ]; then
    VOLUME_SUFFIX=":Z"
fi

echo "Building CUDA 13 Faiss for architectures: $CUDA_ARCHES"
"$CONTAINER_CLI" run --rm \
    -e CUDA_ARCHES="$CUDA_ARCHES" \
    -v "$BUILDER_DIR:/workspace/pkg:ro${VOLUME_SUFFIX}" \
    -v "$OUTPUT_DIR:/out${VOLUME_SUFFIX}" \
    "$IMAGE" /workspace/pkg/build_wheel.sh

find "$OUTPUT_DIR" -maxdepth 1 -name "$WHEEL_GLOB" -print -quit | grep -q . || {
    echo "ERROR: CUDA 13 Faiss build completed without producing the expected wheel" >&2
    exit 1
}
