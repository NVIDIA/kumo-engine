#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="${1:-/mnt/localssd/datasets/mimic}"
S3_PREFIX="s3://kumo-public-datasets/MIMIC_IV_demo"

rm -rf "${DEST_ROOT}"
mkdir -p "${DEST_ROOT}"

aws s3 sync "${S3_PREFIX}" "${DEST_ROOT}"

echo "Next: ./diskgraph-infer/scripts/infer_source.sh ${DEST_ROOT}"
