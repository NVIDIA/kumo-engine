#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest on a local source directory or source_metadata.json.

PROFILE_ENABLED="false"
PARALLEL_ENABLED="${PARALLEL_ENABLED:-true}"
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

if [[ -z "${SOURCE}" ]]; then
  echo "Usage: SOURCE=/path/to/source_dir_or_metadata [OUTPUT=path] $0 [-p]" >&2
  exit 1
fi

if [[ -z "${OUTPUT}" ]]; then
  if [[ -d /mnt/localssd ]]; then
    DEFAULT_OUT_ROOT="/mnt/localssd/diskgraph-index"
  else
    DEFAULT_OUT_ROOT="/tmp/diskgraph-index"
  fi
  if [[ -d "${SOURCE}" ]]; then
    INPUT_NAME="$(basename "${SOURCE}")"
  else
    INPUT_NAME="$(basename "$(dirname "${SOURCE}")")"
  fi
  OUTPUT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}/${INPUT_NAME}"
fi

PARALLEL_FLAG=()
if [[ "${PARALLEL_ENABLED}" == "true" ]]; then
  PARALLEL_FLAG=(--parallel)
fi

if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  cargo run -p diskgraph-ingest --bin ingest --release -- \
    --profile \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest local \
    "${PARALLEL_FLAG[@]}"
else
  cargo run -p diskgraph-ingest --bin ingest --release -- \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest local \
    "${PARALLEL_FLAG[@]}"
fi
