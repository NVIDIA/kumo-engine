#!/usr/bin/env bash
set -euo pipefail

# Run cloud_ingest ADLS test against real Azure Blob/ADLS v2 (MSI/role or supplied creds).

if [[ "${RUN_CLOUD_TESTS:-}" != "1" ]]; then
  export RUN_CLOUD_TESTS=1
fi

: "${AZURE_ACCOUNT:?Set AZURE_ACCOUNT}"
: "${AZURE_CONTAINER:?Set AZURE_CONTAINER}"
AZURE_PREFIX_BASE=${AZURE_PREFIX_BASE:-"diskgraph-index/azure"}
RUN_ID="run-$(date +%s)-$RANDOM"
export AZURE_PREFIX=${AZURE_PREFIX:-"${AZURE_PREFIX_BASE}/${RUN_ID}"}

echo "Using Azure account=${AZURE_ACCOUNT}, container=${AZURE_CONTAINER}, prefix=${AZURE_PREFIX}"
echo "Running cloud ingest test (ADLS)..."
cargo test -p diskgraph-index --test cloud_ingest -- --ignored adls_ingest_round_trip
