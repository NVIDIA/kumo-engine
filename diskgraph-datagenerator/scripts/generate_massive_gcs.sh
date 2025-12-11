#!/usr/bin/env bash
set -euo pipefail

# Generate a massive dataset locally and push it to GCS.
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

# Target prefix in GCS, e.g. gs://bucket/diskgraph-datagenerator
GCS_PREFIX="${GCS_PREFIX:-gs://kumo-unit-test/diskgraph-datagenerator}"
DEST="${GCS_PREFIX%/}/${RUN_ID}"

echo "Using GCS destination: ${DEST}"
echo "Checking gcloud/ADC access with a dummy upload..."
TMPFILE="$(mktemp)"
echo "diskgraph massive generator access check" > "${TMPFILE}"
gsutil cp "${TMPFILE}" "${DEST}/_access_check.txt" >/dev/null
rm -f "${TMPFILE}"

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
gsutil -m cp -r "${OUTPUT_DIR}"/* "${DEST}/"

echo "Done. Dataset uploaded to ${DEST}"
