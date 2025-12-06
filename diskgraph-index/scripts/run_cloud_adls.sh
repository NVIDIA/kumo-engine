#!/usr/bin/env bash
set -euo pipefail

# Run cloud_ingest ADLS test against real Azure Blob/ADLS v2 (MSI/role or supplied creds).

if [[ "${RUN_CLOUD_TESTS:-}" != "1" ]]; then
  export RUN_CLOUD_TESTS=1
fi

: "${AZURE_PREFIX:=az://kumo-unit-test/diskgraph-index}"
RUN_ID="run-$(date +%s)-$RANDOM"
export AZURE_URL="${AZURE_PREFIX%/}/${RUN_ID}"

echo "Using Azure URL=${AZURE_URL}"
echo "Checking access to ${AZURE_URL}..."
if ! az storage blob list --container-name "$(echo "${AZURE_URL}" | cut -d/ -f4)" --account-name "$(echo "${AZURE_URL}" | cut -d/ -f3)" >/dev/null 2>&1; then
  echo "ERROR: Unable to access ${AZURE_URL}. Ensure Azure auth is configured (e.g., 'az login' or MSI/role) and the container exists." >&2
  exit 1
fi
echo "Running cloud ingest test (ADLS)..."
cargo test -p diskgraph-index --test cloud_ingest -- --ignored adls_ingest_round_trip
