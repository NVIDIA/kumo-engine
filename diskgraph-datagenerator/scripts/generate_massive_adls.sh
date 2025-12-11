#!/usr/bin/env bash
set -euo pipefail

# Generate a massive dataset locally and push it to Azure Blob/ADLS via azcopy.
# Edit the constants below or override via env vars.

NODES="${NODES:-500000000}"
EDGES_PER_REL="${EDGES_PER_REL:-500000000}"
MAX_DST_ID="${MAX_DST_ID:-100000000000}"
REL0_NAME="${REL0_NAME:-rel_a}"
REL1_NAME="${REL1_NAME:-rel_b}"
REL0_SORT_BITS="${REL0_SORT_BITS:-0}"
REL1_SORT_BITS="${REL1_SORT_BITS:-0}"
NODE_TYPE="${NODE_TYPE:-Node}"
EDGE_BATCH="${EDGE_BATCH:-1000000}"
NODE_BATCH="${NODE_BATCH:-5000000}"
ROW_GROUP="${ROW_GROUP:-1000000}"
SEED="${SEED:-42}"

LOCAL_ROOT="${LOCAL_ROOT:-/tmp/diskgraph-massive}"
RUN_ID="${RUN_ID:-run-$(date +%s)-$RANDOM}"
OUTPUT_DIR="${LOCAL_ROOT}/${RUN_ID}"

# Target URL prefix, e.g. https://ACCOUNT.blob.core.windows.net/CONTAINER/diskgraph-datagenerator
AZURE_URL="${AZURE_URL:-https://kumo.blob.core.windows.net/kumo-unit-test/diskgraph-datagenerator}"
DEST="${AZURE_URL%/}/${RUN_ID}"

echo "Using Azure destination: ${DEST}"
echo "Ensure azcopy is logged in (SAS/account key/identity)."

mkdir -p "${OUTPUT_DIR}"

echo "Generating massive dataset to ${OUTPUT_DIR}"
cargo run -p diskgraph-datagenerator --bin massive --release -- \
  --output "${OUTPUT_DIR}" \
  --nodes "${NODES}" \
  --edges-per-relation "${EDGES_PER_REL}" \
  --max-dst-id "${MAX_DST_ID}" \
  --edge-batch-size "${EDGE_BATCH}" \
  --node-batch-size "${NODE_BATCH}" \
  --row-group-size "${ROW_GROUP}" \
  --seed "${SEED}" \
  --node-type-name "${NODE_TYPE}" \
  --relation0-name "${REL0_NAME}" \
  --relation1-name "${REL1_NAME}" \
  --relation0-sort-bits "${REL0_SORT_BITS}" \
  --relation1-sort-bits "${REL1_SORT_BITS}"

echo "Uploading dataset to ${DEST}"
azcopy copy "${OUTPUT_DIR}" "${DEST}" --recursive

echo "Done. Dataset uploaded to ${DEST}"
