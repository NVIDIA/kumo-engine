#!/usr/bin/env bash
set -euo pipefail

# Generate raw CSV/Parquet source data directly to GCS.

NODES="${1:-${NODES:-1m}}"
EDGE_ROWS="${2:-${EDGE_ROWS:-500k}}"
FORMAT="${FORMAT:-parquet}"
SEED="${SEED:-42}"

GCS_PREFIX="${GCS_PREFIX:-gs://kumo-unit-test/diskgraph-source}"
RUN_ID="${RUN_ID:-nodes-${NODES}-edges-${EDGE_ROWS}-${FORMAT}}"
DEST="${GCS_PREFIX%/}/${RUN_ID}"

echo "Generating source dataset to ${DEST}"
cargo run -p diskgraph-datagenerator --bin generate-source --release -- \
  --dest gcs \
  --output "${DEST}" \
  --nodes "${NODES}" \
  --edge-rows "${EDGE_ROWS}" \
  --format "${FORMAT}" \
  --seed "${SEED}"

echo "Done. Dataset written to ${DEST}"
