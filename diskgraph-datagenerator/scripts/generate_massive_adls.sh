#!/usr/bin/env bash
set -euo pipefail

# Generate a massive dataset and stream it directly to Azure Blob/ADLS.
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
ROW_GROUP="${ROW_GROUP:-1000000}"
SEED="${SEED:-42}"

# Target URL prefix, e.g. az://ACCOUNT/CONTAINER/diskgraph-datagenerator
AZURE_URL="${AZURE_URL:-az://kumo/kumo-unit-test/diskgraph-datagenerator}"
RUN_ID="${RUN_ID:-run-$(date +%s)-$RANDOM}"
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
  --relation1-sort-bits "${REL1_SORT_BITS}"

echo "Done. Dataset written to ${DEST}"
