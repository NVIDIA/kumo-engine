#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest to Azure output (source paths can be local or az://).

SOURCE="${SOURCE:-${1:-}}"
OUTPUT="${OUTPUT:-${2:-}}"
MAPPING="${MAPPING:-hash}"

if [[ -z "${SOURCE}" || -z "${OUTPUT}" ]]; then
  echo "Usage: SOURCE=path OUTPUT=az://account/container/prefix [MAPPING=hash|sorted] $0" >&2
  exit 1
fi

cargo run -p diskgraph-ingest --bin ingest --release -- \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --dest adls \
  --mapping "${MAPPING}"
