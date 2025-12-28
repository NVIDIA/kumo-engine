#!/usr/bin/env bash
set -euo pipefail

# Explore a shard stored on GCS (expects metadata.json, offsets.bin, edges.bin under the prefix).
# Pass gs://bucket/prefix as the first arg. Uses ADC/gcloud auth.

PROFILE_ENABLED="false"
while getopts "p" opt; do
  case "${opt}" in
    p)
      PROFILE_ENABLED="true"
      ;;
    *)
      echo "usage: $0 [-p] gs://bucket/prefix" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

GCS_URL="${1:-}"
if [[ -z "${GCS_URL}" ]]; then
  echo "usage: $0 [-p] gs://bucket/prefix" >&2
  exit 1
fi

echo "Exploring GCS shard at ${GCS_URL}"
gsutil ls "${GCS_URL}" >/dev/null 2>&1 || echo "Warning: gsutil check failed; ensure ADC creds are set."
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  cargo run -p diskgraph-index --bin explore --release -- --profile "explore-profile" "${GCS_URL}"
else
  cargo run -p diskgraph-index --bin explore --release -- "${GCS_URL}"
fi
