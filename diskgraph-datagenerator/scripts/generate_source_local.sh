#!/usr/bin/env bash
set -euo pipefail

# Generate raw CSV/Parquet source data plus source_metadata.json locally.

NODES="${1:-${NODES:-1m}}"
EDGE_ROWS="${2:-${EDGE_ROWS:-500k}}"
FORMAT="${FORMAT:-parquet}"
SEED="${SEED:-42}"

if [[ -d /mnt/localssd ]]; then
  DEFAULT_OUT_ROOT="/mnt/localssd/diskgraph-source"
  if ! mkdir -p "${DEFAULT_OUT_ROOT}" 2>/dev/null; then
    echo "ERROR: ${DEFAULT_OUT_ROOT} is not writable. Run with sudo or chown /mnt/localssd." >&2
    exit 1
  fi
else
  DEFAULT_OUT_ROOT="/tmp/diskgraph-source"
fi
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
RUN_ID="${RUN_ID:-nodes-${NODES}-edges-${EDGE_ROWS}-${FORMAT}}"
OUTPUT_DIR="${OUT_ROOT}/${RUN_ID}"

mkdir -p "${OUTPUT_DIR}"

echo "Generating source dataset to ${OUTPUT_DIR}"
cargo run -p diskgraph-datagenerator --bin generate-source --release -- \
  --output "${OUTPUT_DIR}" \
  --nodes "${NODES}" \
  --edge-rows "${EDGE_ROWS}" \
  --format "${FORMAT}" \
  --seed "${SEED}"

echo "Done. Dataset available at ${OUTPUT_DIR}"
