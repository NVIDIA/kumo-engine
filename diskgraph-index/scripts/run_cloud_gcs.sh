#!/usr/bin/env bash
set -euo pipefail

# Run cloud_ingest GCS test against real GCS (or another https-compatible endpoint).

if [[ "${RUN_CLOUD_TESTS:-}" != "1" ]]; then
  export RUN_CLOUD_TESTS=1
fi

: "${GCS_PREFIX:=gs://kumo-unit-test/diskgraph-index}"
RUN_ID="run-$(date +%s)-$RANDOM"
export GCS_URL="${GCS_PREFIX%/}/${RUN_ID}"

if [[ -d /mnt/localssd && -w /mnt/localssd ]]; then
  TMPDIR="/mnt/localssd/tmp"
  mkdir -p "${TMPDIR}"
fi

echo "Using GCS URL=${GCS_URL}"
echo "Checking access by uploading a dummy object..."
TMPFILE="$(mktemp -p "${TMPDIR:-/tmp}")"
echo "diskgraph cloud ingest access check" > "${TMPFILE}"
if ! gsutil cp "${TMPFILE}" "${GCS_URL}/_access_check.txt" >/dev/null 2>&1; then
  echo "ERROR: Unable to write to ${GCS_URL}. Ensure your GCP creds are configured (e.g., run 'gcloud auth login' or activate the correct service account/Workload Identity)." >&2
  rm -f "${TMPFILE}"
  exit 1
fi
rm -f "${TMPFILE}"
echo "Running cloud ingest test (GCS)..."
cargo test -p diskgraph-index --test cloud_ingest -- gcs_ingest_round_trip
