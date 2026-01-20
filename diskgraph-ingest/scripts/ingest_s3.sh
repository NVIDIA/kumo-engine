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

if [[ -z "${SOURCE}" ]]; then
  echo "Usage: $0 <path_or_s3> [-p] [-d]" >&2
  exit 1
fi

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
OUTPUT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}/${INPUT_NAME}"

export AWS_EC2_METADATA_DISABLED=true
export AWS_SDK_LOAD_CONFIG=1

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
    --dest local
else
  env "${env_args[@]}" cargo run -p diskgraph-ingest --bin ingest --release -- \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --dest local
fi
