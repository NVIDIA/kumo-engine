#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest to S3 output (source paths can be local or s3://).

SOURCE="${SOURCE:-${1:-}}"
OUTPUT="${OUTPUT:-${2:-}}"
MAPPING="${MAPPING:-hash}"

if [[ -z "${SOURCE}" || -z "${OUTPUT}" ]]; then
  echo "Usage: SOURCE=path OUTPUT=s3://bucket/prefix [MAPPING=hash|sorted] $0" >&2
  exit 1
fi

export AWS_EC2_METADATA_DISABLED=true
export AWS_SDK_LOAD_CONFIG=1

cargo run -p diskgraph-ingest --bin ingest --release -- \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --dest s3 \
  --mapping "${MAPPING}"
