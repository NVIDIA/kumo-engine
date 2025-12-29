#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest to S3 output (source paths can be local or s3://).

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
  echo "Usage: SOURCE=path OUTPUT=s3://bucket/prefix [MAPPING=hash|sorted] $0 [-p]" >&2
  exit 1
fi

export AWS_EC2_METADATA_DISABLED=true
export AWS_SDK_LOAD_CONFIG=1

if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  cargo run -p diskgraph-ingest --bin ingest --release -- \
    --profile \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest s3 \
    --mapping "${MAPPING}"
else
  cargo run -p diskgraph-ingest --bin ingest --release -- \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest s3 \
    --mapping "${MAPPING}"
fi
