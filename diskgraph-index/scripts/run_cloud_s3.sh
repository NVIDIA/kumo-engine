#!/usr/bin/env bash
set -euo pipefail

# Run cloud_ingest S3 test against a real S3-compatible endpoint using IAM/role or supplied creds.

if [[ "${RUN_CLOUD_TESTS:-}" != "1" ]]; then
  export RUN_CLOUD_TESTS=1
fi

: "${S3_BUCKET:?Set S3_BUCKET to your test bucket}"
S3_PREFIX_BASE=${S3_PREFIX_BASE:-"diskgraph-index/s3"}
RUN_ID="run-$(date +%s)-$RANDOM"
export S3_PREFIX=${S3_PREFIX:-"${S3_PREFIX_BASE}/${RUN_ID}"}

echo "Using S3 bucket=${S3_BUCKET}, prefix=${S3_PREFIX}"
echo "Running cloud ingest test (S3)..."
cargo test -p diskgraph-index --test cloud_ingest -- --ignored s3_ingest_round_trip
