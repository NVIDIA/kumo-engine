#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest to GCS output (source paths can be local or gs://).

SOURCE="${SOURCE:-${1:-}}"
OUTPUT="${OUTPUT:-${2:-}}"
MAPPING="${MAPPING:-hash}"

if [[ -z "${SOURCE}" || -z "${OUTPUT}" ]]; then
  echo "Usage: SOURCE=path OUTPUT=gs://bucket/prefix [MAPPING=hash|sorted] $0" >&2
  exit 1
fi

cargo run -p diskgraph-ingest --bin ingest --release -- \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --dest gcs \
  --mapping "${MAPPING}"
