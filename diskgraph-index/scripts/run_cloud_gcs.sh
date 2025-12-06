#!/usr/bin/env bash
set -euo pipefail

# Run cloud_ingest GCS test against real GCS (or another https-compatible endpoint).

if [[ "${RUN_CLOUD_TESTS:-}" != "1" ]]; then
  export RUN_CLOUD_TESTS=1
fi

: "${GCS_BUCKET:?Set GCS_BUCKET to your test bucket}"
: "${GCS_ENDPOINT:?Set GCS_ENDPOINT=https://storage.googleapis.com or your endpoint (must be https)}"
GCS_PREFIX_BASE=${GCS_PREFIX_BASE:-"diskgraph-index/gcs"}
RUN_ID="run-$(date +%s)-$RANDOM"
export GCS_PREFIX=${GCS_PREFIX:-"${GCS_PREFIX_BASE}/${RUN_ID}"}

echo "Using GCS bucket=${GCS_BUCKET}, prefix=${GCS_PREFIX}"
echo "Running cloud ingest test (GCS)..."
cargo test -p diskgraph-index --test cloud_ingest -- --ignored gcs_ingest_round_trip
