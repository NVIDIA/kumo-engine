#!/usr/bin/env bash
set -euo pipefail

# Local multi-table data generation for benchmarking diskgraph ingest.
# Adjust the constants below or override via env vars when calling the script.

NODES="${1:-${NODES:-500m}}"
EDGES_PER_REL="${2:-${EDGES_PER_REL:-500m}}"
NODE_DATA_BYTES="${3:-${NODE_DATA_BYTES:-128b}}"
NUM_SHARDS="${NUM_SHARDS:-5}"
EDGE_BATCH="${EDGE_BATCH:-1m}"
ROW_GROUP="${ROW_GROUP:-4m}"
SEED="${SEED:-42}"
NODE_DATA="${NODE_DATA:-true}"
NODE_DATA_BATCH="${NODE_DATA_BATCH:-1m}"
NODE_DATA_FLAG=""
if [[ "${NODE_DATA}" == "false" ]]; then
  NODE_DATA_FLAG="--no-node-data"
fi

if [[ -d /mnt/localssd ]]; then
  DEFAULT_OUT_ROOT="/mnt/localssd/diskgraph-multi"
  if ! mkdir -p "${DEFAULT_OUT_ROOT}" 2>/dev/null; then
    echo "ERROR: ${DEFAULT_OUT_ROOT} is not writable. Run with sudo or chown /mnt/localssd." >&2
    exit 1
  fi
else
  DEFAULT_OUT_ROOT="/tmp/diskgraph-multi"
fi
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
RUN_ID="${RUN_ID:-nodes-${NODES}-edges-${EDGES_PER_REL}-bytes-${NODE_DATA_BYTES}}"
OUTPUT_DIR="${OUT_ROOT}/${RUN_ID}"

mkdir -p "${OUTPUT_DIR}"

echo "Generating multi-table dataset to ${OUTPUT_DIR}"
cargo run -p diskgraph-datagenerator --bin generate-multi --release -- \
  --output "${OUTPUT_DIR}" \
  --nodes "${NODES}" \
  --edges-per-relation "${EDGES_PER_REL}" \
  --num-shards "${NUM_SHARDS}" \
  --edge-batch-size "${EDGE_BATCH}" \
  --row-group-size "${ROW_GROUP}" \
  --seed "${SEED}" \
  ${NODE_DATA_FLAG} \
  --node-data-bytes "${NODE_DATA_BYTES}" \
  --node-data-batch-size "${NODE_DATA_BATCH}"

echo "Done. Dataset available at ${OUTPUT_DIR}"
