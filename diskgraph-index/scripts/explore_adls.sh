#!/usr/bin/env bash
set -euo pipefail

# Explore a shard stored on Azure Blob/ADLS (expects metadata.json, offsets.bin, edges.bin).
# Set AZURE_URL=az://account/container/prefix or pass as first arg.

AZURE_URL="${AZURE_URL:-${1:-}}"
if [[ -z "${AZURE_URL}" ]]; then
  echo "usage: AZURE_URL=az://account/container/prefix $0"
  exit 1
fi

echo "Exploring Azure shard at ${AZURE_URL}"
echo "Ensure AZURE_ACCOUNT/AZURE_CONTAINER/AZURE_ACCESS_KEY or SAS/MSI env vars are set."
cargo run -p diskgraph-index --bin explore -- "${AZURE_URL}"
