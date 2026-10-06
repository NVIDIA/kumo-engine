#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build a CUDA 13 faiss-gpu wheel from source.
#
# Designed to run inside the container defined by build.Dockerfile (NVIDIA
# CUDA devel image on Rocky 8 + gcc-toolset-13). Clones the upstream faiss
# repository at the pinned revision, builds the C++ library and SWIG bindings
# for the requested CUDA architectures, then produces a manylinux_2_28
# wheel via setuptools + auditwheel.
#
# The wheel keeps upstream's package name, `faiss-gpu`, matching the official
# Meta-published PyPI package it overrides for architectures the public
# wheel doesn't cover (see deps.json).
#
# Usage (inside the container):
#   /workspace/pkg/build_wheel.sh
#
# Environment variables:
#   FAISS_VERSION  upstream tag without the "v" prefix     (default: 1.14.3)
#   FAISS_REVISION immutable upstream Git commit
#   PACKAGE_VERSION package version compatible with deps   (default: 1.14.3)
#   CUDA_ARCHES    CMAKE_CUDA_ARCHITECTURES list           (default: 75;80;86;89;90;100;120)
#   CUDA_TAG       PEP 440 local version segment for wheel (default: cu130sm120)
#   PYTHON         absolute path to python binary          (default: /usr/local/bin/python3.14)
#   PLAT           auditwheel --plat target                (default: manylinux_2_28_x86_64)
#   OUT_DIR        host-mounted output directory           (default: /out)
set -euo pipefail

# Pinned to the facebookresearch/faiss v1.14.3 tag (the same version the
# official faiss-gpu==1.14.3 PyPI wheel is built from), so this override's
# version matches kumo-core's dependency pin exactly instead of drifting to
# a different PEP 440 version.
FAISS_VERSION="${FAISS_VERSION:-1.14.3}"
FAISS_REVISION="${FAISS_REVISION:-0ca9df4792b173d573044ee14ca0704780176e82}"
PACKAGE_VERSION="${PACKAGE_VERSION:-1.14.3}"
CUDA_ARCHES="${CUDA_ARCHES:-75;80;86;89;90;100;120}"
CUDA_TAG="${CUDA_TAG:-cu130sm120}"
PYTHON="${PYTHON:-/usr/local/bin/python3.14}"
PLAT="${PLAT:-manylinux_2_28_x86_64}"
OUT_DIR="${OUT_DIR:-/out}"

mkdir -p "${OUT_DIR}"
cd /workspace

if [ ! -d faiss ]; then
    git init -q faiss
    git -C faiss remote add origin https://github.com/facebookresearch/faiss.git
    git -C faiss fetch -q --depth 1 origin "${FAISS_REVISION}"
    git -C faiss checkout -q --detach FETCH_HEAD
fi
cd faiss
if [ "$(git rev-parse HEAD)" != "${FAISS_REVISION}" ]; then
    echo "ERROR: expected Faiss revision ${FAISS_REVISION}, found $(git rev-parse HEAD)" >&2
    exit 1
fi

# Normalize timestamps used by the compiler and wheel tooling so retry
# comparisons are not invalidated by build time alone. Both wheel and
# auditwheel honor SOURCE_DATE_EPOCH when writing ZIP entries.
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
PYTHONHASHSEED=0
TZ=UTC
export SOURCE_DATE_EPOCH PYTHONHASHSEED TZ

# Rename upstream package "faiss" -> "faiss-gpu" so the wheel matches the
# name we already use in deps.json. Restricting to the name= line keeps the
# rest of setup.py (which references the import path "faiss/") untouched.
sed -i 's/^    name="faiss",$/    name="faiss-gpu",/' faiss/python/setup.py

# Tag the version with a PEP 440 local segment so we never collide with the
# public CUDA 13 wheel while identifying the sm_120-enabled build.
LOCAL_VERSION="${PACKAGE_VERSION}+${CUDA_TAG}"
sed -i "s/^    version=\"${FAISS_VERSION}\",$/    version=\"${LOCAL_VERSION}\",/" \
    faiss/python/setup.py

cmake -B build . \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_C_API=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=avx2 \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHES}" \
    -DPython_EXECUTABLE="${PYTHON}"

# C++ library: generic + AVX2. faiss/loader.py picks the most optimized
# variant at import time based on the host CPU, so we ship both.
make -C build -j"$(nproc)" faiss faiss_avx2

# SWIG Python bindings linked against the C++ libraries above.
make -C build -j"$(nproc)" swigfaiss swigfaiss_avx2

cd build/faiss/python
"${PYTHON}" setup.py bdist_wheel

# auditwheel needs libfaiss.so on LD_LIBRARY_PATH to detect and bundle it.
# CUDA libs are already on LD_LIBRARY_PATH thanks to the base image.
export LD_LIBRARY_PATH="/workspace/faiss/build/faiss:${LD_LIBRARY_PATH:-}"

# Bundle native deps such as libfaiss, libgomp, and OpenBLAS. CUDA runtime
# libraries come from the CUDA 13 PyTorch/image stack; excluding them avoids
# a ~400 MB duplicate and prevents a second CUDA runtime from taking precedence.
# libcuda is the userspace driver shim and must always come from the host.
"${PYTHON}" -m auditwheel repair dist/*.whl \
    --exclude libcuda.so.1 \
    --exclude libcuda.so \
    --exclude libcudart.so.13 \
    --exclude libcublas.so.13 \
    --exclude libcublasLt.so.13 \
    --plat "${PLAT}" \
    -w "${OUT_DIR}"

# faiss's setup.py ships the SWIG extensions as package *data* rather than
# as Extension modules, so bdist_wheel mislabels the wheel "py3-none"
# (pure-Python, ABI-agnostic). It is not: the bundled _swigfaiss*.so are
# built against this exact CPython ABI. Retag to cpXY-cpXY so pip refuses
# to install it on the wrong interpreter (where import would abort).
PYTAG="cp$(${PYTHON} -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
"${PYTHON}" -m wheel tags --remove \
    --python-tag "${PYTAG}" \
    --abi-tag "${PYTAG}" \
    "${OUT_DIR}"/*.whl

echo ""
echo "Built wheels:"
ls -lh "${OUT_DIR}"
