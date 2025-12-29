#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest on a local source directory or source_metadata.json.

SOURCE="${SOURCE:-${1:-}}"
OUTPUT="${OUTPUT:-${2:-}}"
MAPPING="${MAPPING:-hash}"

if [[ -z "${SOURCE}" ]]; then
  echo "Usage: SOURCE=/path/to/source_dir_or_metadata [OUTPUT=path] [MAPPING=hash|sorted] $0" >&2
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

cargo run -p diskgraph-ingest --bin ingest --release -- \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --dest local \
  --mapping "${MAPPING}"
