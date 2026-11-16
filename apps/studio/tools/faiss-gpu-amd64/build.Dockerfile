# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build environment for CUDA 13 faiss-gpu wheels.
#
# Uses NVIDIA's official CUDA devel image on Rocky Linux 8 because:
#   - It is on Docker Hub (no cross-org auth needed)
#   - Rocky 8 ships glibc 2.28, matching the manylinux_2_28 wheel tag
#   - CUDA 13.0 is preinstalled at /usr/local/cuda-13.0
#
# CUDA 13 is the required runtime and compiles sm_120 (Blackwell RTX PRO / RTX 50xx).
# It supports GCC 13, so we layer gcc-toolset-13 on
# top of the base (Rocky 8's stock gcc is 8.5, too old for faiss's C++20).
ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-rockylinux8@sha256:7692d63946b3eafbd54268b0a4eca0ae3c13d3df4891670ad81cb1bb7ec16f4b
FROM ${BASE_IMAGE}

# Major.minor used for the interpreter binary name and the wheel ABI tag.
ARG PYTHON_VERSION=3.14
# Full CPython version built from source (see below). Must agree with
# PYTHON_VERSION's major.minor. Rocky 8's AppStream skips 3.10 entirely
# (it does not ship 3.14), so CPython has to be built from source.
ARG PYTHON_FULL_VERSION=3.14.6
ARG PYTHON_SOURCE_SHA256=74d0d71d0600e477651a077101d6e62d1e2e69b8e992ba18c993dd643b7ba222

# EPEL gives us patchelf; PowerTools / "powertools" provides openblas-devel.
# The *-devel packages (openssl, libffi, bzip2, xz, zlib) are CPython build
# dependencies — without them the interpreter loses ssl/ctypes/etc. and pip
# can't fetch wheels.
RUN dnf -y install dnf-plugins-core epel-release \
    && dnf config-manager --set-enabled powertools \
    && dnf -y install \
        bzip2-devel \
        cmake \
        gcc-toolset-13 \
        git \
        libffi-devel \
        make \
        openblas-devel \
        openssl-devel \
        pcre2-devel \
        wget \
        xz-devel \
        zlib-devel \
    && dnf clean all

# Build CPython from source. A plain build (no PGO) is enough to run
# setup.py/pip and to provide Python.h for the swigfaiss extension.
# --enable-shared produces libpython3.x.so so faiss's CMake FindPython can
# link it cleanly; the -rpath plus ldconfig let the interpreter locate it.
RUN cd /tmp \
    && wget -q "https://www.python.org/ftp/python/${PYTHON_FULL_VERSION}/Python-${PYTHON_FULL_VERSION}.tgz" \
    && echo "${PYTHON_SOURCE_SHA256}  Python-${PYTHON_FULL_VERSION}.tgz" | sha256sum -c - \
    && tar xf "Python-${PYTHON_FULL_VERSION}.tgz" \
    && cd "Python-${PYTHON_FULL_VERSION}" \
    && ./configure --prefix=/usr/local --enable-shared --with-ensurepip=install \
        LDFLAGS="-Wl,-rpath,/usr/local/lib" \
    && make -j"$(nproc)" \
    && make altinstall \
    && cd / && rm -rf "/tmp/Python-${PYTHON_FULL_VERSION}" "/tmp/Python-${PYTHON_FULL_VERSION}.tgz" \
    && echo /usr/local/lib > /etc/ld.so.conf.d/local-python.conf && ldconfig

# faiss's CMake passes `-doxygen` to swig, which requires SWIG >= 4.0.
# Rocky 8 ships SWIG 3, so build SWIG from source. We pull the official
# release tarball from SourceForge (the GitHub tag tarball only has
# parser.y, which would require bison 3.5+ to regenerate; Rocky 8 ships
# bison 3.0.4. SourceForge releases ship a pre-generated parser.c and
# a pre-generated configure, so neither bison nor autoconf is needed).
ARG SWIG_VERSION=4.2.1
ARG SWIG_SOURCE_SHA256=fa045354e2d048b2cddc69579e4256245d4676894858fcf0bab2290ecf59b7d8
RUN cd /tmp \
    && wget -q "https://downloads.sourceforge.net/project/swig/swig/swig-${SWIG_VERSION}/swig-${SWIG_VERSION}.tar.gz" -O swig.tar.gz \
    && echo "${SWIG_SOURCE_SHA256}  swig.tar.gz" | sha256sum -c - \
    && tar xf swig.tar.gz \
    && cd "swig-${SWIG_VERSION}" \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && cd / && rm -rf "/tmp/swig-${SWIG_VERSION}" /tmp/swig.tar.gz

# gcc-toolset-13 ships under /opt/rh — wire it into PATH/LD_LIBRARY_PATH so
# cmake picks gcc-13 by default.
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:/opt/rh/gcc-toolset-13/root/usr/lib:${LD_LIBRARY_PATH}

ENV PYTHON=/usr/local/bin/python${PYTHON_VERSION}
# Rocky 8's patchelf is 0.12, but auditwheel >= 6 requires patchelf >= 0.14.
# The PyPI `patchelf` package bundles a recent binary (>= 0.17).
ARG BUILD_REQUIREMENTS="auditwheel==6.7.0 build==1.5.0 numpy==2.5.0 packaging==26.2 patchelf==0.17.2.4 pyelftools==0.33 pyproject-hooks==1.2.0 setuptools==82.0.1 wheel==0.47.0"
RUN ${PYTHON} -m pip install --no-cache-dir --no-deps ${BUILD_REQUIREMENTS}

WORKDIR /workspace
