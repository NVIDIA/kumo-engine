#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${DISKGRAPH_DATA_ROOT:-}" ]]; then
  DEFAULT_ROOT="${DISKGRAPH_DATA_ROOT}"
elif [[ -d /mnt/localssd && -w /mnt/localssd ]]; then
  DEFAULT_ROOT="/mnt/localssd"
else
  DEFAULT_ROOT="/tmp"
fi

DATASET_ROOT="${1:-${DEFAULT_ROOT%/}/datasets/hnm}"
INDEX_ROOT="${2:-${DEFAULT_ROOT%/}/diskgraph-index/hnm}"
OUTPUT_PARQUET="${3:-${DEFAULT_ROOT%/}/train-tables/hnm_temporal.parquet}"
CONFIG_PATH="${DATASET_ROOT%/}/train_table_job.json"
INFER_CONFIG_PATH="${DATASET_ROOT%/}/diskgraph_config.json"

mkdir -p "$(dirname "${OUTPUT_PARQUET}")"

if [[ ! -f "${DATASET_ROOT%/}/articles.parquet" || ! -f "${DATASET_ROOT%/}/customers.parquet" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "error: aws CLI not found (required for hnm download)" >&2
    exit 1
  fi
  ./diskgraph-datagenerator/scripts/hnm100_download.sh "${DATASET_ROOT}"
else
  echo "dataset already present; skipping download"
fi

if [[ ! -f "${INFER_CONFIG_PATH}" ]]; then
  ./diskgraph-infer/scripts/infer_config.sh "${DATASET_ROOT}"
else
  echo "diskgraph_config.json already present; skipping infer"
fi

if [[ ! -f "${INDEX_ROOT%/}/index_metadata.json" ]]; then
  OUTPUT="${INDEX_ROOT}" ./diskgraph-ingest/scripts/ingest_local.sh "${DATASET_ROOT}" "${INDEX_ROOT}"
else
  echo "index already present; skipping ingest"
fi

cat <<JSON > "${CONFIG_PATH}"
{
  "index_path": "${INDEX_ROOT}",
  "query": "PREDICT SUM(transactions.price WHERE transactions.sales_channel_id = 1, 0, 30, days) FOR EACH customers.customer_id",
  "output_parquet": "${OUTPUT_PARQUET}",
  "split": {
    "type": "random",
    "fractions": [0.8, 0.1, 0.1],
    "seed": 123
  },
  "options": {
    "temporal_sampling": "timeframe"
  }
}
JSON

cargo run -p diskgraph-train-table --bin generate --release -- --config "${CONFIG_PATH}"

echo "train table written to ${OUTPUT_PARQUET}"
