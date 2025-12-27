#!/usr/bin/env bash
set -euo pipefail

# Explore a shard stored on Azure Blob/ADLS (expects metadata.json, offsets.bin, edges.bin).
# Pass az://account/container/prefix as the first arg.

AZURE_URL="${1:-}"
if [[ -z "${AZURE_URL}" ]]; then
  echo "usage: $0 az://account/container/prefix"
  exit 1
fi

echo "Exploring Azure shard at ${AZURE_URL}"
echo "Ensure AZURE_ACCOUNT/AZURE_CONTAINER/AZURE_ACCESS_KEY or SAS/MSI env vars are set."
cargo run -p diskgraph-index --bin explore -- "${AZURE_URL}"
