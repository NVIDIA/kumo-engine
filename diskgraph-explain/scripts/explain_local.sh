#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run diskgraph-explain on a local parquet + config.

ULIMIT_NOFILE="${ULIMIT_NOFILE:-100000}"
if ! ulimit -n "${ULIMIT_NOFILE}" 2>/dev/null; then
  echo "warning: failed to raise ulimit -n to ${ULIMIT_NOFILE}; current limit may be too low" >&2
fi

CONFIG_PATH="${CONFIG_PATH:-${1:-}}"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "usage: CONFIG_PATH=/path/to/config.json $0" >&2
  exit 1
fi

cargo run -p diskgraph-explain --bin explain --release -- "${CONFIG_PATH}"
