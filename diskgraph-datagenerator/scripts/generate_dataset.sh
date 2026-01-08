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

mkdir -p "$OUT_ROOT"

exec cargo run -p diskgraph-datagenerator --release -- \
  --config "$CONFIG_PATH" \
  --output "$OUT_ROOT"
