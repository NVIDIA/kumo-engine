#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config.json>"
  exit 1
fi

CONFIG_PATH="$1"
CONFIG_NAME="$(basename "$CONFIG_PATH")"
DATASET_NAME="${CONFIG_NAME%.json}"
OUT_ROOT="/mnt/localssd/datasets/${DATASET_NAME}"

if [[ -d "$OUT_ROOT" ]]; then
  rm -rf "$OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"

if cargo run -p diskgraph-datagenerator --bin diskgraph-datagenerator --release -- \
  --config "$CONFIG_PATH" \
  --output "$OUT_ROOT"; then
  echo "Next steps:"
  echo "  ./diskgraph-infer/scripts/infer_source.sh \"$OUT_ROOT\""
  echo "  ./diskgraph-ingest/scripts/ingest_local.sh \"$OUT_ROOT\""
  echo "  ./diskgraph-index/scripts/explore_local.sh /mnt/localssd/diskgraph-index/${DATASET_NAME}"
fi
