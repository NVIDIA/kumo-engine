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
QUERY_FILTER="${QUERY_FILTER:-all}"

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

REL_CUSTOMER_TXN=$(python3 - <<PY
import json, os, sys
path = os.path.join("${INDEX_ROOT}", "index_metadata.json")
try:
    with open(path, "r") as f:
        meta = json.load(f)
    for rel in meta.get("edge_relations", []):
        if rel.get("src_type") == "customers" and rel.get("dst_type") == "transactions":
            print(rel.get("name", ""))
            sys.exit(0)
except Exception:
    pass
print("")
PY
)
if [[ -z "${REL_CUSTOMER_TXN}" ]]; then
  REL_CUSTOMER_TXN="transactions"
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

queries=(
  "PREDICT COUNT(${REL_CUSTOMER_TXN}.*, 0, 7, days) FOR EACH customers.customer_id ASSUMING COUNT(${REL_CUSTOMER_TXN}.*, -30, 0, days) > 0"
  "PREDICT SUM(transactions.price WHERE transactions.sales_channel_id = 1, 0, 30, days) FOR EACH customers.customer_id"
  "PREDICT AVG(transactions.price, 0, 30, days) FOR EACH customers.customer_id"
  "PREDICT LIST_DISTINCT(transactions.article_id, 0, 7, days) FOR EACH customers.customer_id"
  "PREDICT LIST_DISTINCT(transactions.article_id) FOR EACH transactions.customer_id"
)

run_query() {
  local idx="$1"
  local query="$2"
  echo "running query ${idx}: ${query}"
  ./diskgraph-train-table/scripts/ttui.sh "${INDEX_ROOT}" -- -C "${query}"
}

if [[ "${QUERY_FILTER}" == "all" ]]; then
  for i in "${!queries[@]}"; do
    run_query "$((i + 1))" "${queries[$i]}"
  done
else
  IFS=',' read -r -a selected <<< "${QUERY_FILTER}"
  for sel in "${selected[@]}"; do
    idx=$((sel - 1))
    if [[ "${idx}" -ge 0 && "${idx}" -lt "${#queries[@]}" ]]; then
      run_query "${sel}" "${queries[$idx]}"
    else
      echo "skipping invalid query index: ${sel}" >&2
    fi
  done
fi
