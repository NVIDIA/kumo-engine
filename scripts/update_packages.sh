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
# Run BEFORE any uninstall/install/CUDA-detect so a mismatch never partially
# mutates the venv. Parse the required major.minor from .mise.toml; if it can't
# be read, skip the guard rather than break non-mise environments.
REQUIRED_PYTHON=""
if [ -f .mise.toml ]; then
    # Matches: python = "3.14"  or  python = "3.14.4"  (quotes optional), and
    # the list form  python = ["3.14", ...] — take the first version, trim to
    # major.minor.
    REQUIRED_PYTHON=$(sed -n 's/^[[:space:]]*python[[:space:]]*=[[:space:]]*\[\{0,1\}[[:space:]]*["'\'']\{0,1\}\([0-9][0-9]*\.[0-9][0-9]*\)[0-9.]*["'\'']\{0,1\}.*/\1/p' .mise.toml | head -n1)
fi

if [ -z "${REQUIRED_PYTHON}" ]; then
    echo "WARNING: could not parse required Python version from .mise.toml; skipping interpreter guard." >&2
else
    # Capture without aborting under 'set -e' if the venv python is broken.
    VENV_PY_VER=$("${VENV_PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || VENV_PY_VER=""
    if [ -z "${VENV_PY_VER}" ]; then
        echo "WARNING: could not determine venv Python version from ${VENV_PYTHON}; skipping interpreter guard." >&2
    elif [ "${VENV_PY_VER}" != "${REQUIRED_PYTHON}" ]; then
        MSG="virtualenv Python is ${VENV_PY_VER} but this branch requires ${REQUIRED_PYTHON} (.mise.toml). update-packages does NOT recreate the venv. Recreate it on the right interpreter: 'make install-venv' (or 'PYTHON_BIN=<py3.14> ./scripts/install_venv.sh'), then re-run. Set KUMO_ALLOW_PYTHON_MISMATCH=1 to override."
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
# CUDA 12 is no longer supported by this local installer. Set CUDA=cpu to
# explicitly install the CPU stack instead.
if [ -z "${CUDA+x}" ]; then
    if command -v nvidia-smi &> /dev/null; then
        CUDA=cu130
    else
        CUDA=cpu
    fi
fi
if [ "${CUDA}" != "cpu" ] && [ "${CUDA}" != "cu130" ]; then
    echo "ERROR: unsupported CUDA selector '${CUDA}'; use CUDA=cu130 for GPU development or CUDA=cpu" >&2
    exit 1
fi
export CUDA
echo "CUDA=$CUDA"

is_linux_arm64() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    [ "$os" = "Linux" ] && { [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; }
}

echo "Uninstall existing packages"
uv pip uninstall --python "${VENV_PYTHON}" scikit-learn || true
uv pip uninstall --python "${VENV_PYTHON}" pandas || true
uv pip uninstall --python "${VENV_PYTHON}" pytorch-frame || true
uv pip uninstall --python "${VENV_PYTHON}" kumo-api || true
uv pip uninstall --python "${VENV_PYTHON}" kumo-pql || true

echo "Install Kumo (along with PyG and other dependencies)"
PYPI_INDEX_URL=https://pypi.org/simple
SETUPTOOLS_VERSION_SPEC='setuptools>=80.10.2,<82'

github_token="${GITHUB_TOKEN:-}"
if [ -z "${github_token}" ] && command -v gh >/dev/null 2>&1; then
    github_token="$(gh auth token 2>/dev/null || true)"
fi

uv_pip_install_kumo() {
    if [ -z "${github_token}" ]; then
        uv pip install "$@"
        return
    fi

    # Keep the token command-scoped: private HTTPS Git dependencies can use
    # the existing gh login without writing credentials into global Git config.
    local github_basic_auth git_config_count i key_var value_var
    local existing_key existing_value
    github_basic_auth="$(printf 'x-access-token:%s' "${github_token}" | base64 | tr -d '\n')"
    git_config_count="${GIT_CONFIG_COUNT:-0}"
    if ! [[ "${git_config_count}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: GIT_CONFIG_COUNT must be a non-negative integer." >&2
        return 1
    fi
    for ((i = 0; i < git_config_count; i++)); do
        key_var="GIT_CONFIG_KEY_${i}"
        value_var="GIT_CONFIG_VALUE_${i}"
        existing_key="$(printf '%s' "${!key_var:-}" | tr '[:upper:]' '[:lower:]')"
        existing_value="$(printf '%s' "${!value_var:-}" | tr '[:upper:]' '[:lower:]')"
        if { [ "${existing_key}" = "http.https://github.com/.extraheader" ] ||
             [ "${existing_key}" = "http.extraheader" ]; } &&
           [[ "${existing_value}" == authorization:* ]]; then
            uv pip install "$@"
            return
        fi
    done
    env \
      "GIT_CONFIG_COUNT=$((git_config_count + 1))" \
      "GIT_CONFIG_KEY_${git_config_count}=http.https://github.com/.extraheader" \
      "GIT_CONFIG_VALUE_${git_config_count}=AUTHORIZATION: basic ${github_basic_auth}" \
      uv pip install "$@"
}

if [ "${CUDA}" = "cu130" ] && ! is_linux_arm64; then
    ./scripts/build_faiss_gpu_amd64_wheel.sh
fi

# Some runtime deps still import pkg_resources (for example Snowpark).
# Pin >=80.10.2 for vendored CVE fixes, <82 because 82 removed pkg_resources.
uv pip install --python "${VENV_PYTHON}" --index-url "${PYPI_INDEX_URL}" "${SETUPTOOLS_VERSION_SPEC}"

FAISS_FIND_LINKS=()
if is_linux_arm64 && [ "${CUDA}" != "cpu" ]; then
  FAISS_GPU_WHEEL_DIR="$(./scripts/build_faiss_gpu_arm64_wheel.sh --python "${VENV_PYTHON}" --cuda "${CUDA}")"
  FAISS_FIND_LINKS=(-f "${FAISS_GPU_WHEEL_DIR}")
fi

# DiskGraph is not on a package index. Install it before the root
# resolve: `--no-deps` leaves kumo-core's dependency metadata in place.
./scripts/diskgraph install --python "${VENV_PYTHON}"

# The full Kumo checkout consumes the same portable implementation shipped as
# the standalone wheels, plus the source-reader packages the warehouse
# connectors import lazily. Install first-party packages from the monorepo
# before the root editable package so local contract changes are exercised
# directly; `--no-deps` keeps their `kumo-core` pin from starting a second
# resolve.
uv pip install --python "${VENV_PYTHON}" --no-deps \
  -e packages/kumo-api \
  -e packages/kumo-pql \
  -e packages/kumo-core \
  -e packages/kumo-snowflake \
  -e packages/kumo-bigquery \
  -e packages/kumo-databricks \
  -e packages/kumo-s3 \
  -e packages/kumo-gcs

# Install project/build dependencies.
uv_pip_install_kumo --python "${VENV_PYTHON}" --index-url "${PYPI_INDEX_URL}" --torch-backend="${CUDA}" -e ".[full,dev]" \
  -f 3rdparty/kumo-wheels/ \
  "${FAISS_FIND_LINKS[@]}"

# `install` above verified a pre-resolve venv; re-check that the resolve
# left the pinned wheel in place.
./scripts/diskgraph verify --python "${VENV_PYTHON}"

# Snowpark needs to be patched after running the above step.
echo "Update packages post Kumo installation."
uv pip uninstall --python "${VENV_PYTHON}" snowflake-connector-python || true
uv pip uninstall --python "${VENV_PYTHON}" snowflake-snowpark-python || true
uv pip uninstall --python "${VENV_PYTHON}" pyspark || true
uv pip uninstall --python "${VENV_PYTHON}" databricks-connect || true
# the following installs the snowflake-connector-python as well.
# Floor >=1.49: first release admitting Python 3.14 (1.11.0/1.24 cap at <3.12,
# so on 3.14 the resolver must pick >=1.49). Keep in sync with setup.py.
uv pip install --python "${VENV_PYTHON}" "snowflake-snowpark-python>=1.49"
