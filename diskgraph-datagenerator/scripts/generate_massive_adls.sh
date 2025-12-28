#!/usr/bin/env bash
set -euo pipefail

# Generate a massive dataset and stream it directly to Azure Blob/ADLS.
# Edit the constants below or override via env vars.

NODES="${1:-${NODES:-500m}}"
EDGES_PER_REL="${2:-${EDGES_PER_REL:-500m}}"
NODE_DATA_BYTES="${3:-${NODE_DATA_BYTES:-128b}}"
MAX_DST_ID="${MAX_DST_ID:-100b}"
REL0_NAME="${REL0_NAME:-rel_a}"
REL1_NAME="${REL1_NAME:-rel_b}"
REL0_SORT_BITS="${REL0_SORT_BITS:-0}"
REL1_SORT_BITS="${REL1_SORT_BITS:-0}"
NODE_TYPE="${NODE_TYPE:-Node}"
EDGE_BATCH="${EDGE_BATCH:-1m}"
ROW_GROUP="${ROW_GROUP:-1m}"
SEED="${SEED:-42}"
NODE_DATA="${NODE_DATA:-true}"
NODE_DATA_BATCH="${NODE_DATA_BATCH:-1m}"
NODE_DATA_FLAG=""
if [[ "${NODE_DATA}" == "false" ]]; then
  NODE_DATA_FLAG="--no-node-data"
fi

# Target URL prefix, e.g. az://ACCOUNT/CONTAINER/diskgraph-datagenerator
AZURE_URL="${AZURE_URL:-az://kumo/kumo-unit-test/diskgraph-datagenerator}"
RUN_ID="${RUN_ID:-nodes-${NODES}-edges-${EDGES_PER_REL}-bytes-${NODE_DATA_BYTES}}"
DEST="${AZURE_URL%/}/${RUN_ID}"

echo "Using Azure destination: ${DEST}"
echo "Ensure azcopy is logged in (SAS/account key/identity)."

echo "Generating massive dataset directly to ${DEST}"
cargo run -p diskgraph-datagenerator --bin massive --release -- \
  --dest adls \
  --output "${DEST}" \
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

echo "Done. Dataset written to ${DEST}"
