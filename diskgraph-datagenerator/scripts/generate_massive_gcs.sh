#!/usr/bin/env bash
set -euo pipefail

# Generate a massive dataset and stream it directly to GCS.
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
NODE_DATA="${NODE_DATA:-true}"
NODE_DATA_BYTES="${NODE_DATA_BYTES:-128}"
NODE_DATA_BATCH="${NODE_DATA_BATCH:-1000000}"
NODE_DATA_FLAG=""
if [[ "${NODE_DATA}" == "false" ]]; then
  NODE_DATA_FLAG="--no-node-data"
fi

# Target prefix in GCS, e.g. gs://bucket/diskgraph-datagenerator
GCS_PREFIX="${GCS_PREFIX:-gs://kumo-unit-test/diskgraph-datagenerator}"
RUN_ID="${RUN_ID:-run-$(date +%s)-$RANDOM}"
DEST="${GCS_PREFIX%/}/${RUN_ID}"

echo "Using GCS destination: ${DEST}"
echo "Checking gcloud/ADC access with a dummy upload..."
TMPFILE="$(mktemp)"
echo "diskgraph massive generator access check" > "${TMPFILE}"
gsutil cp "${TMPFILE}" "${DEST}/_access_check.txt" >/dev/null
rm -f "${TMPFILE}"

echo "Generating massive dataset directly to ${DEST}"
cargo run -p diskgraph-datagenerator --bin massive --release -- \
  --dest gcs \
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
