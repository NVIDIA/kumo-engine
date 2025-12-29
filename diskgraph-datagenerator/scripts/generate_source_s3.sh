#!/usr/bin/env bash
set -euo pipefail

# Generate raw CSV/Parquet source data directly to S3.

NODES="${1:-${NODES:-1m}}"
EDGE_ROWS="${2:-${EDGE_ROWS:-500k}}"
FORMAT="${FORMAT:-parquet}"
SEED="${SEED:-42}"

S3_PREFIX="${S3_PREFIX:-s3://kumo-unit-test/diskgraph-source}"
RUN_ID="${RUN_ID:-nodes-${NODES}-edges-${EDGE_ROWS}-${FORMAT}}"
DEST="${S3_PREFIX%/}/${RUN_ID}"

export AWS_EC2_METADATA_DISABLED=true
export AWS_SDK_LOAD_CONFIG=1

echo "Generating source dataset to ${DEST}"
cargo run -p diskgraph-datagenerator --bin generate-source --release -- \
  --dest s3 \
  --output "${DEST}" \
  --nodes "${NODES}" \
  --edge-rows "${EDGE_ROWS}" \
  --format "${FORMAT}" \
  --seed "${SEED}"

echo "Done. Dataset written to ${DEST}"
