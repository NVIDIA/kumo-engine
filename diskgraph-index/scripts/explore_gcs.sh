#!/usr/bin/env bash
set -euo pipefail

# Explore a shard stored on GCS (expects metadata.json, offsets.bin, edges.bin under the prefix).
# Pass gs://bucket/prefix as the first arg. Uses ADC/gcloud auth.

GCS_URL="${1:-}"
if [[ -z "${GCS_URL}" ]]; then
  echo "usage: $0 gs://bucket/prefix"
  exit 1
fi

echo "Exploring GCS shard at ${GCS_URL}"
gsutil ls "${GCS_URL}" >/dev/null 2>&1 || echo "Warning: gsutil check failed; ensure ADC creds are set."
cargo run -p diskgraph-index --bin explore -- "${GCS_URL}"
