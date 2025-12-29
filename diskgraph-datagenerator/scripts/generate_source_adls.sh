#!/usr/bin/env bash
set -euo pipefail

# Generate raw CSV/Parquet source data directly to Azure Blob/ADLS.

NODES="${1:-${NODES:-1m}}"
EDGE_ROWS="${2:-${EDGE_ROWS:-500k}}"
FORMAT="${FORMAT:-parquet}"
SEED="${SEED:-42}"

AZURE_URL="${AZURE_URL:-az://kumo/kumo-unit-test/diskgraph-source}"
RUN_ID="${RUN_ID:-nodes-${NODES}-edges-${EDGE_ROWS}-${FORMAT}}"
DEST="${AZURE_URL%/}/${RUN_ID}"

echo "Generating source dataset to ${DEST}"
cargo run -p diskgraph-datagenerator --bin generate-source --release -- \
  --dest adls \
  --output "${DEST}" \
  --nodes "${NODES}" \
  --edge-rows "${EDGE_ROWS}" \
  --format "${FORMAT}" \
  --seed "${SEED}"

echo "Done. Dataset written to ${DEST}"
