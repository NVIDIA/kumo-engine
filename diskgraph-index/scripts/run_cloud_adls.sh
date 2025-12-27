#!/usr/bin/env bash
set -euo pipefail

# Run cloud_ingest ADLS test against real Azure Blob/ADLS v2 (MSI/role or supplied creds).

export RUN_CLOUD_TESTS=1

: "${AZURE_PREFIX:=az://kumo-unit-test/diskgraph-index}"
RUN_ID="run-$(date +%s)-$RANDOM"
export AZURE_URL="${AZURE_PREFIX%/}/${RUN_ID}"

if [[ -d /mnt/localssd && -w /mnt/localssd ]]; then
  TMPDIR="/mnt/localssd/tmp"
  mkdir -p "${TMPDIR}"
fi

echo "Using Azure URL=${AZURE_URL}"
echo "Checking access by uploading a dummy blob..."
TMPFILE="$(mktemp -p "${TMPDIR:-/tmp}")"
echo "diskgraph cloud ingest access check" > "${TMPFILE}"
AZ_ACCOUNT="$(echo "${AZURE_URL}" | cut -d/ -f3)"
AZ_CONTAINER="$(echo "${AZURE_URL}" | cut -d/ -f4)"
AZ_BLOB_PATH="$(echo "${AZURE_URL}" | cut -d/ -f5-)/_access_check.txt"
if ! az storage blob upload --overwrite --account-name "${AZ_ACCOUNT}" --container-name "${AZ_CONTAINER}" --name "${AZ_BLOB_PATH}" --file "${TMPFILE}" >/dev/null 2>&1; then
  echo "ERROR: Unable to write to ${AZURE_URL}. Ensure Azure auth is configured (e.g., 'az login' or MSI/role) and the container exists." >&2
  rm -f "${TMPFILE}"
  exit 1
fi
rm -f "${TMPFILE}"
echo "Running cloud ingest test (ADLS)..."
cargo test -p diskgraph-index --test cloud_ingest -- adls_ingest_round_trip
