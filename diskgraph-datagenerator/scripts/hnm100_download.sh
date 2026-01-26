#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${DISKGRAPH_DATA_ROOT:-}" ]]; then
  DEFAULT_ROOT="${DISKGRAPH_DATA_ROOT}"
elif [[ -d /mnt/localssd && -w /mnt/localssd ]]; then
  DEFAULT_ROOT="/mnt/localssd"
else
  DEFAULT_ROOT="/tmp"
fi
DEST_ROOT="${1:-${DEFAULT_ROOT%/}/datasets/hnm}"
TXN_SUBDIR="${2:-transactions}"
DUP_COUNT="${3:-100}"

S3_BASE="s3://kumo-public-datasets/Kaggle_H&M/parquet"

rm -rf "${DEST_ROOT}"
mkdir -p "${DEST_ROOT}" "${DEST_ROOT}/${TXN_SUBDIR}"

aws s3 cp "${S3_BASE}/articles.parquet" "${DEST_ROOT}/"
aws s3 cp "${S3_BASE}/customers.parquet" "${DEST_ROOT}/"

TXN_SOURCE="${DEST_ROOT}/${TXN_SUBDIR}/transactions_train.parquet"
aws s3 cp "${S3_BASE}/transactions_train.parquet" "${TXN_SOURCE}"

for i in $(seq -w 1 "${DUP_COUNT}"); do
  cp -f "${TXN_SOURCE}" "${DEST_ROOT}/${TXN_SUBDIR}/transactions_train_${i}.parquet"
done

echo "Next: ./diskgraph-infer/scripts/infer_config.sh ${DEST_ROOT}"
