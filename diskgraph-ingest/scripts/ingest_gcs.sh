#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest to GCS output (source paths can be local or gs://).

PROFILE_ENABLED="false"
POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    -p)
      PROFILE_ENABLED="true"
      ;;
    *)
      POSITIONAL+=("${arg}")
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

SOURCE="${SOURCE:-${1:-}}"
OUTPUT="${OUTPUT:-${2:-}}"
MAPPING="${MAPPING:-hash}"

if [[ -z "${SOURCE}" || -z "${OUTPUT}" ]]; then
  echo "Usage: SOURCE=path OUTPUT=gs://bucket/prefix [MAPPING=hash|sorted] $0 [-p]" >&2
  exit 1
fi

if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  cargo run -p diskgraph-ingest --bin ingest --release -- \
    --profile \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest gcs \
    --mapping "${MAPPING}"
else
  cargo run -p diskgraph-ingest --bin ingest --release -- \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest gcs \
    --mapping "${MAPPING}"
fi
