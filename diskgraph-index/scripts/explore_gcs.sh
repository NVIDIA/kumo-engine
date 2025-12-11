#!/usr/bin/env bash
set -euo pipefail

# Explore a shard stored on GCS (expects metadata.json, offsets.bin, edges.bin under the prefix).
# Set GCS_URL=gs://bucket/prefix or pass as first arg. Uses ADC/gcloud auth.

GCS_URL="${GCS_URL:-${1:-}}"
if [[ -z "${GCS_URL}" ]]; then
  echo "usage: GCS_URL=gs://bucket/prefix $0"
  exit 1
fi

echo "Exploring GCS shard at ${GCS_URL}"
gsutil ls "${GCS_URL}" >/dev/null 2>&1 || echo "Warning: gsutil check failed; ensure ADC creds are set."
cargo run -p diskgraph-index --bin explore -- "${GCS_URL}"
