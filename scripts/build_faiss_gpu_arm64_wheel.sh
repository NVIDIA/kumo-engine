#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build_faiss_gpu_arm64_wheel.sh [--python <python>] [--cuda <cu126|cu130>]

Build a local faiss-gpu wheel for Linux ARM GPU development. Official
faiss-gpu PyPI wheels are x86_64-only, so local ARM GPU setups need a compatible wheel in a
find-links directory before resolving Kumo's Python dependencies.
Set FAISS_CUDA_ARCHES to a Torch-style list such as 8.7;12.1 when building
for GPUs other than the locally visible device.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CUDA="${CUDA:-cu130}"
FAISS_GPU_VERSION="${FAISS_GPU_VERSION:-1.14.3}"
FAISS_REPO_URL="${FAISS_REPO_URL:-https://github.com/facebookresearch/faiss.git}"
FAISS_TAG="${FAISS_TAG:-v${FAISS_GPU_VERSION}}"

while [ $# -gt 0 ]; do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --cuda)
            CUDA="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

log() {
    echo "$@" >&2
}

is_linux_arm64() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    [ "$os" = "Linux" ] && { [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; }
}

if ! is_linux_arm64; then
    echo "ERROR: local faiss-gpu wheel builds are only configured for Linux ARM64" >&2
    exit 1
fi

if [ "$CUDA" = "cpu" ]; then
    echo "ERROR: faiss-gpu wheel build requires a CUDA selector, got CUDA=cpu" >&2
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi
PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required to fetch FAISS source" >&2
    exit 1
fi

CUDA_ARCHES="${FAISS_CUDA_ARCHES:-}"
if [ -z "$CUDA_ARCHES" ]; then
    CUDA_ARCHES="$("$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; cannot build faiss-gpu")

major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
fi

CUDA_ARCHES="${CUDA_ARCHES//,/ }"
CUDA_ARCHES="${CUDA_ARCHES//;/ }"
read -r -a CUDA_ARCH_VALUES <<< "$CUDA_ARCHES"
CMAKE_CUDA_ARCHES=""
for cuda_arch in "${CUDA_ARCH_VALUES[@]}"; do
    cuda_arch="${cuda_arch%+PTX}"
    cuda_arch="${cuda_arch//./}"
    if [[ ! "$cuda_arch" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid CUDA architecture '$cuda_arch'" >&2
        exit 2
    fi
    case ";${CMAKE_CUDA_ARCHES};" in
        *";${cuda_arch};"*) continue ;;
    esac
    CMAKE_CUDA_ARCHES="${CMAKE_CUDA_ARCHES:+${CMAKE_CUDA_ARCHES};}${cuda_arch}"
done
if [ -z "$CMAKE_CUDA_ARCHES" ]; then
    echo "ERROR: at least one CUDA architecture is required" >&2
    exit 2
fi

KEY_ARCHES="${CMAKE_CUDA_ARCHES//;/_}"
CACHE_ROOT="${FAISS_GPU_CACHE_DIR:-${REPO_ROOT}/.cache/kumo/faiss-gpu}"
# TODO: Add host architecture to wheel and source cache keys before CI shares
# this cache between x86_64 and ARM64 builders.
WHEEL_DIR="${CACHE_ROOT}/wheels/${FAISS_GPU_VERSION}-${CUDA}/arches-${KEY_ARCHES}"
SRC_DIR="${CACHE_ROOT}/src/faiss-${FAISS_GPU_VERSION}-${CUDA}/arches-${KEY_ARCHES}"
mkdir -p "$WHEEL_DIR" "$(dirname "$SRC_DIR")"

if ls "$WHEEL_DIR"/faiss_gpu-"${FAISS_GPU_VERSION}"-*.whl >/dev/null 2>&1; then
    log "Reusing local faiss-gpu ${FAISS_GPU_VERSION} wheel in ${WHEEL_DIR}"
    printf '%s\n' "$WHEEL_DIR"
    exit 0
fi

log "Building faiss-gpu ${FAISS_GPU_VERSION} for Linux ARM64 CUDA=${CUDA} architectures=${CMAKE_CUDA_ARCHES}"
rm -rf "$SRC_DIR"
git clone --depth 1 --branch "$FAISS_TAG" "$FAISS_REPO_URL" "$SRC_DIR" >&2

cp "$SRC_DIR/pyproject-gpu.toml" "$SRC_DIR/pyproject.toml"

"$PYTHON_BIN" - "$SRC_DIR" "$CUDA" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
cuda = sys.argv[2]

pyproject = src / "pyproject.toml"
text = pyproject.read_text()
if cuda == "cu130":
    text = text.replace(
        '"Environment :: GPU :: NVIDIA CUDA :: 12"',
        '"Environment :: GPU :: NVIDIA CUDA :: 13"',
    )
    text = text.replace(
        '"nvidia-cuda-runtime-cu12>=12.6,<13"',
        '"nvidia-cuda-runtime>=13,<14"',
    )
    text = text.replace(
        '"nvidia-cublas-cu12>=12.6,<13"',
        '"nvidia-cublas>=13,<14"',
    )
    text = text.replace(
        '"nvidia-curand-cu12>=10.3.7,<11"',
        '"nvidia-curand>=10.4,<11"',
    )
pyproject.write_text(text)

init_py = src / "faiss" / "python" / "__init__.py"
text = init_py.read_text()
start = text.index("def _preload_gpu_libs():")
end = text.index("\n\n_preload_gpu_libs()", start)
replacement = r'''def _preload_gpu_libs():
    """Pre-load CUDA shared libraries from NVIDIA pip packages.

    GPU wheels dlopen libfaiss.so, which has CUDA undefined symbols. The CUDA
    runtime wheels are not on the default dynamic loader path, so load them with
    RTLD_GLOBAL before importing the SWIG extension.
    """
    try:
        from . import _gpu_build  # noqa: F401
    except ImportError:
        return

    import ctypes
    import os

    candidate_groups = []

    try:
        import nvidia
        cu13_lib = os.path.join(nvidia.__path__[0], "cu13", "lib")
        candidate_groups.append([
            os.path.join(cu13_lib, "libcudart.so.13"),
            os.path.join(cu13_lib, "libcublas.so.13"),
            os.path.join(cu13_lib, "libcublasLt.so.13"),
            os.path.join(cu13_lib, "libcurand.so.10"),
        ])
    except ImportError:
        pass

    try:
        import nvidia.cuda_runtime
        import nvidia.cublas
        import nvidia.curand
        candidate_groups.append([
            os.path.join(nvidia.cuda_runtime.__path__[0], "lib", "libcudart.so.12"),
            os.path.join(nvidia.cublas.__path__[0], "lib", "libcublas.so.12"),
            os.path.join(nvidia.cublas.__path__[0], "lib", "libcublasLt.so.12"),
            os.path.join(nvidia.curand.__path__[0], "lib", "libcurand.so.10"),
        ])
    except ImportError:
        pass

    errors = []
    for libs in candidate_groups:
        try:
            for so_path in libs:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError as e:
            errors.append(str(e))

    raise RuntimeError(
        "faiss-gpu installed but CUDA runtime libraries could not be loaded. "
        "Install matching NVIDIA CUDA runtime, cuBLAS, and cuRAND wheels. "
        f"Load errors: {errors}"
    )'''
init_py.write_text(text[:start] + replacement + text[end:])
PY

uv pip install --python "$PYTHON_BIN" build >&2

(
    cd "$SRC_DIR"
    CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
    "$PYTHON_BIN" -m build --wheel --outdir "$WHEEL_DIR" \
        -C "cmake.define.CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHES}" >&2
)

printf '%s\n' "$WHEEL_DIR"
