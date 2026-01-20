#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest from local or s3:// source to local output.

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
OUTPUT="${OUT_ROOT:-${2:-}}"

if [[ -z "${SOURCE}" ]]; then
  echo "Usage: $0 <path_or_s3> [output_dir] [-p] [-d]" >&2
  exit 1
fi

if [[ -z "${OUTPUT}" ]]; then
  if [[ -n "${DISKGRAPH_DATA_ROOT:-}" ]]; then
    DEFAULT_OUT_ROOT="${DISKGRAPH_DATA_ROOT%/}/diskgraph-index"
  elif [[ -d /mnt/localssd ]]; then
    DEFAULT_OUT_ROOT="/mnt/localssd/diskgraph-index"
  else
    DEFAULT_OUT_ROOT="/tmp/diskgraph-index"
  fi
  if [[ -d "${SOURCE}" ]]; then
    INPUT_NAME="$(basename "${SOURCE}")"
  else
    INPUT_NAME="$(basename "$(dirname "${SOURCE}")")"
  fi
  OUTPUT="${DEFAULT_OUT_ROOT}/${INPUT_NAME}"
fi

CMD=(cargo run -p diskgraph-ingest --bin ingest --release -- --source "${SOURCE}" --output "${OUTPUT}" --dest local)
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  CMD+=(--profile)
fi

env_args=()
if [[ "${DEBUGINFO_ENABLED}" == "true" ]]; then
  env_args+=(CARGO_PROFILE_RELEASE_DEBUG=1)
  env_args+=(RUSTFLAGS="${RUSTFLAGS:-} -C debuginfo=1 -C force-frame-pointers=yes")
fi

env "${env_args[@]}" "${CMD[@]}"
