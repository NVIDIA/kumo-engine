#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest to GCS output (source paths can be local or gs://).

PROFILE_ENABLED="false"
DEBUGINFO_ENABLED="false"
POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    -p)
      PROFILE_ENABLED="true"
      ;;
    -d)
      DEBUGINFO_ENABLED="true"
      ;;
    *)
      POSITIONAL+=("${arg}")
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

SOURCE="${SOURCE:-${1:-}}"
OUTPUT="${OUTPUT:-${2:-}}"

if [[ -z "${SOURCE}" || -z "${OUTPUT}" ]]; then
  echo "Usage: SOURCE=path OUTPUT=gs://bucket/prefix $0 [-p] [-d]" >&2
  exit 1
fi

env_args=()
if [[ "${DEBUGINFO_ENABLED}" == "true" ]]; then
  env_args+=(CARGO_PROFILE_RELEASE_DEBUG=1)
  env_args+=(RUSTFLAGS="${RUSTFLAGS:-} -C debuginfo=1 -C force-frame-pointers=yes")
fi

if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  env "${env_args[@]}" cargo run -p diskgraph-ingest --bin ingest --release -- \
    --profile \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest gcs
else
  env "${env_args[@]}" cargo run -p diskgraph-ingest --bin ingest --release -- \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest gcs
fi
