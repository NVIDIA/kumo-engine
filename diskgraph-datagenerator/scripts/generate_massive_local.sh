#!/usr/bin/env bash
set -euo pipefail

# Local massive data generation for benchmarking diskgraph ingest.
# Adjust the constants below or override via env vars when calling the script.

NODES="${NODES:-500000000}"
EDGES_PER_REL="${EDGES_PER_REL:-500000000}"
MAX_DST_ID="${MAX_DST_ID:-100000000000}"
REL0_NAME="${REL0_NAME:-rel_a}"
REL1_NAME="${REL1_NAME:-rel_b}"
REL0_SORT_BITS="${REL0_SORT_BITS:-0}"   # 0 disables dest_sort_key
REL1_SORT_BITS="${REL1_SORT_BITS:-0}"
NODE_TYPE="${NODE_TYPE:-Node}"
EDGE_BATCH="${EDGE_BATCH:-1000000}"
ROW_GROUP="${ROW_GROUP:-4000000}"
SEED="${SEED:-42}"
NODE_DATA="${NODE_DATA:-true}"
NODE_DATA_BYTES="${NODE_DATA_BYTES:-128}"
NODE_DATA_BATCH="${NODE_DATA_BATCH:-1000000}"
NODE_DATA_FLAG=""
if [[ "${NODE_DATA}" == "false" ]]; then
  NODE_DATA_FLAG="--no-node-data"
fi

if [[ -d /mnt/localssd ]]; then
  DEFAULT_OUT_ROOT="/mnt/localssd/diskgraph-massive"
  if ! mkdir -p "${DEFAULT_OUT_ROOT}" 2>/dev/null; then
    echo "ERROR: ${DEFAULT_OUT_ROOT} is not writable. Run with sudo or chown /mnt/localssd." >&2
    exit 1
  fi
else
  DEFAULT_OUT_ROOT="/tmp/diskgraph-massive"
fi
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
RUN_ID="${RUN_ID:-run-$(date +%s)-$RANDOM}"
OUTPUT_DIR="${OUT_ROOT}/${RUN_ID}"

mkdir -p "${OUTPUT_DIR}"

echo "Generating massive dataset to ${OUTPUT_DIR}"
cargo run -p diskgraph-datagenerator --bin massive --release -- \
  --output "${OUTPUT_DIR}" \
  --nodes "${NODES}" \
  --edges-per-relation "${EDGES_PER_REL}" \
  --max-dst-id "${MAX_DST_ID}" \
  --edge-batch-size "${EDGE_BATCH}" \
  --row-group-size "${ROW_GROUP}" \
  --seed "${SEED}" \
  --node-type-name "${NODE_TYPE}" \
  --relation0-name "${REL0_NAME}" \
  --relation1-name "${REL1_NAME}" \
  --relation0-sort-bits "${REL0_SORT_BITS}" \
  --relation1-sort-bits "${REL1_SORT_BITS}" \
  ${NODE_DATA_FLAG} \
  --node-data-bytes "${NODE_DATA_BYTES}" \
  --node-data-batch-size "${NODE_DATA_BATCH}"

echo "Done. Dataset available at ${OUTPUT_DIR}"
