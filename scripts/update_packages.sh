#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

if ! command -v uv >/dev/null 2>&1; then
    cat << EOF
uv is not available.

Install project toolchains with mise, then retry:

  mise install
EOF
    exit 1
fi

VENV_PYTHON=${VENV_PYTHON:-./.venv/bin/python}
if [ ! -x "${VENV_PYTHON}" ]; then
    cat << EOF
Python virtual environment not found at ${VENV_PYTHON}.

Create it first:

  ./scripts/install_venv.sh
EOF
    exit 1
fi

# Guard against a stale-interpreter venv (brownfield cutover hazard): a reused
# venv on an older Python silently receiving this branch's targeted stack.
# Run BEFORE any install/CUDA-detect so a mismatch never partially mutates
# the venv. Read the required major.minor from .python-version (uv-managed,
# not mise); if it can't be read, skip the guard rather than break
# environments that don't pin one.
REQUIRED_PYTHON=""
if [ -f .python-version ]; then
    REQUIRED_PYTHON=$(sed -n 's/^[[:space:]]*\([0-9][0-9]*\.[0-9][0-9]*\)[0-9.]*[[:space:]]*$/\1/p' .python-version | head -n1)
fi

if [ -z "${REQUIRED_PYTHON}" ]; then
    echo "WARNING: could not parse required Python version from .python-version; skipping interpreter guard." >&2
else
    # Capture without aborting under 'set -e' if the venv python is broken.
    VENV_PY_VER=$("${VENV_PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || VENV_PY_VER=""
    if [ -z "${VENV_PY_VER}" ]; then
        echo "WARNING: could not determine venv Python version from ${VENV_PYTHON}; skipping interpreter guard." >&2
    elif [ "${VENV_PY_VER}" != "${REQUIRED_PYTHON}" ]; then
        MSG="virtualenv Python is ${VENV_PY_VER} but this branch requires ${REQUIRED_PYTHON} (.python-version). update-packages does NOT recreate the venv. Recreate it on the right interpreter: 'make install-venv' (or 'PYTHON_BIN=<py3.14> ./scripts/install_venv.sh'), then re-run. Set KUMO_ALLOW_PYTHON_MISMATCH=1 to override."
        if [ "${KUMO_ALLOW_PYTHON_MISMATCH:-}" = "1" ]; then
            echo "WARNING (KUMO_ALLOW_PYTHON_MISMATCH=1): ${MSG}" >&2
        else
            echo "ERROR: ${MSG}" >&2
            exit 1
        fi
    fi
fi

# Keep these instructions in sync with the runtime container configuration.

# Auto-detect CUDA: GPU development uses the CUDA 13 / Blackwell wheel line.
# Set CUDA=cpu to install the CPU stack instead. CUDA selects the torch build via `uv sync --extra ${CUDA}`.
if [ -z "${CUDA+x}" ]; then
    if command -v nvidia-smi &> /dev/null; then
        CUDA=cu130
    else
        CUDA=cpu
    fi
fi
case "${CUDA}" in
    cpu|cu130) ;;
    *)
        echo "ERROR: unsupported CUDA selector '${CUDA}'; use CUDA=cu130 for GPU development or CUDA=cpu" >&2
        exit 1
        ;;
esac
export CUDA
echo "CUDA=$CUDA"

is_linux_arm64() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    [ "$os" = "Linux" ] && { [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; }
}

is_linux_x86_64() {
    [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]
}

if [ "${CUDA}" = "cu130" ] && ! is_linux_arm64; then
    ./scripts/build_faiss_gpu_amd64_wheel.sh
fi

FAISS_FIND_LINKS=()
if is_linux_arm64 && [ "${CUDA}" != "cpu" ]; then
    FAISS_GPU_WHEEL_DIR="$(./scripts/build_faiss_gpu_arm64_wheel.sh --python "${VENV_PYTHON}" --cuda "${CUDA}")"
    FAISS_FIND_LINKS=(-f "${FAISS_GPU_WHEEL_DIR}")
fi

if ! is_linux_x86_64 && ! is_linux_arm64; then
    echo "WARNING: unrecognized platform ($(uname -s) $(uname -m)); using the default uv sync path." >&2
fi

# `uv sync --python` only selects the interpreter used to resolve/create the
# project environment; it does NOT change where uv installs to (that's
# UV_PROJECT_ENVIRONMENT, default ./.venv). Derive it from VENV_PYTHON so a
# non-default override is actually honored as the install destination.
export UV_PROJECT_ENVIRONMENT="$(cd "$(dirname "${VENV_PYTHON}")/.." && pwd)"

uv sync --python "${VENV_PYTHON}" --extra full --extra dev --extra "${CUDA}" "${FAISS_FIND_LINKS[@]}"
