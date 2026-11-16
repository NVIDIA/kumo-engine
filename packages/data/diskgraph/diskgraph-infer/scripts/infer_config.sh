#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

INPUT="${1:-${SOURCE_ROOT:-}}"
OUTPUT="${2:-${OUTPUT_PATH:-}}"

if [[ -z "${INPUT}" ]]; then
  echo "usage: $0 <input_dir_or_url> [output_path]" >&2
  exit 1
fi

CMD=(cargo run -p diskgraph-infer --bin infer_config --release -- --input "${INPUT}")
if [[ -n "${OUTPUT}" ]]; then
  CMD+=(--output "${OUTPUT}")
fi

"${CMD[@]}"
