#!/usr/bin/env bash
set -euo pipefail

# Explore a shard stored on Azure Blob/ADLS (expects index_metadata.json, offsets.bin, edges.bin).
# Pass az://account/container/prefix as the first arg.

PROFILE_ENABLED="false"
while getopts "p" opt; do
  case "${opt}" in
    p)
      PROFILE_ENABLED="true"
      ;;
    *)
      echo "usage: $0 [-p] az://account/container/prefix" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

AZURE_URL="${1:-}"
if [[ -z "${AZURE_URL}" ]]; then
  echo "usage: $0 [-p] az://account/container/prefix" >&2
  exit 1
fi

echo "Exploring Azure shard at ${AZURE_URL}"
echo "Ensure AZURE_ACCOUNT/AZURE_CONTAINER/AZURE_ACCESS_KEY or SAS/MSI env vars are set."
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  cargo run -p diskgraph-index --bin explore --release -- --profile "explore-profile" "${AZURE_URL}"
else
  cargo run -p diskgraph-index --bin explore --release -- "${AZURE_URL}"
fi
