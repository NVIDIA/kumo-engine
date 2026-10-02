#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Interactive inspector for diskgraph-explain outputs.

ULIMIT_NOFILE="${ULIMIT_NOFILE:-100000}"
if ! ulimit -n "${ULIMIT_NOFILE}" 2>/dev/null; then
  echo "warning: failed to raise ulimit -n to ${ULIMIT_NOFILE}; current limit may be too low" >&2
fi

EXPLAIN_PATH="${EXPLAIN_PATH:-${1:-}}"
TRENDS_PATH="${TRENDS_PATH:-${2:-}}"
if [[ -z "${EXPLAIN_PATH}" || -z "${TRENDS_PATH}" ]]; then
  echo "usage: $0 /path/to/explanations.parquet /path/to/global_trends.parquet" >&2
  exit 1
fi

cargo run -p diskgraph-explain --bin explain_cli --release -- "${EXPLAIN_PATH}" "${TRENDS_PATH}"
