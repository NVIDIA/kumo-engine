#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config.json>"
  exit 1
fi

CONFIG_PATH="$1"
CONFIG_NAME="$(basename "$CONFIG_PATH")"
DATASET_NAME="${CONFIG_NAME%.json}"

if [[ -n "${DISKGRAPH_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${DISKGRAPH_DATA_ROOT}"
elif [[ -d /mnt/localssd && -w /mnt/localssd ]]; then
  DATA_ROOT="/mnt/localssd"
else
  DATA_ROOT="/tmp"
fi
OUT_ROOT="${DATA_ROOT%/}/datasets/${DATASET_NAME}"

if [[ -d "$OUT_ROOT" ]]; then
  rm -rf "$OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"

if cargo run -p diskgraph-datagenerator --bin diskgraph-datagenerator --release -- \
  --config "$CONFIG_PATH" \
  --output "$OUT_ROOT"; then
  echo "Next steps:"
  echo "  ./diskgraph-infer/scripts/infer_config.sh \"$OUT_ROOT\""
  echo "  ./diskgraph-ingest/scripts/ingest_local.sh \"$OUT_ROOT\""
  echo "  ./diskgraph-index/scripts/explore_local.sh ${DATA_ROOT%/}/diskgraph-index/${DATASET_NAME}"
fi
