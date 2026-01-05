#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest to Azure output (source paths can be local or az://).

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

if [[ -z "${SOURCE}" || -z "${OUTPUT}" ]]; then
  echo "Usage: SOURCE=path OUTPUT=az://account/container/prefix $0 [-p]" >&2
  exit 1
fi

if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  cargo run -p diskgraph-ingest --bin ingest --release -- \
    --profile \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest adls
else
  cargo run -p diskgraph-ingest --bin ingest --release -- \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest adls
fi
