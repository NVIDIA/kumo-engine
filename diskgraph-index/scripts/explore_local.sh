#!/usr/bin/env bash
set -euo pipefail

# Explore a local shard directory (must contain offsets.bin, edges.bin, metadata.json).

SHARD_PATH="${SHARD_PATH:-${1:-}}"
if [[ -z "${SHARD_PATH}" ]]; then
  echo "usage: SHARD_PATH=/path/to/shard $0"
  exit 1
fi

echo "Exploring local shard at ${SHARD_PATH}"
cargo run -p diskgraph-index --bin explore -- "${SHARD_PATH}"
