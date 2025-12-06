#!/usr/bin/env bash
set -euo pipefail

# Run cloud_ingest S3 test against a real S3-compatible endpoint using IAM/role or supplied creds.

if [[ "${RUN_CLOUD_TESTS:-}" != "1" ]]; then
  export RUN_CLOUD_TESTS=1
fi

: "${S3_PREFIX:=s3://kumo-unit-test/diskgraph-index}"
RUN_ID="run-$(date +%s)-$RANDOM"
export S3_URL="${S3_PREFIX%/}/${RUN_ID}"

echo "Using S3 URL=${S3_URL}"
echo "Checking access by uploading a dummy object..."
TMPFILE="$(mktemp)"
echo "diskgraph cloud ingest access check" > "${TMPFILE}"
if ! aws s3 cp "${TMPFILE}" "${S3_URL}/_access_check.txt" >/dev/null 2>&1; then
  echo "ERROR: Unable to write to ${S3_URL}. Ensure your AWS creds/role are configured (e.g., run 'aws sso login' on a laptop or attach the proper IAM role on a VM)." >&2
  rm -f "${TMPFILE}"
  exit 1
fi
rm -f "${TMPFILE}"
echo "Running cloud ingest test (S3)..."
cargo test -p diskgraph-index --test cloud_ingest -- --ignored s3_ingest_round_trip
