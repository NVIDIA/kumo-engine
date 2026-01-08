#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest on a local source directory or source_metadata.json.

ULIMIT_NOFILE="${ULIMIT_NOFILE:-100000}"
if ! ulimit -n "${ULIMIT_NOFILE}" 2>/dev/null; then
  echo "warning: failed to raise ulimit -n to ${ULIMIT_NOFILE}; current limit may be too low" >&2
fi

PROFILE_ENABLED="false"
DEBUGINFO_ENABLED="false"
PARALLEL_ENABLED="${PARALLEL_ENABLED:-true}"
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

if [[ -z "${SOURCE}" ]]; then
  echo "Usage: SOURCE=/path/to/source_dir_or_metadata [OUTPUT=path] $0 [-p] [-d]" >&2
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

env_args=()
if [[ "${DEBUGINFO_ENABLED}" == "true" ]]; then
  env_args+=(CARGO_PROFILE_RELEASE_DEBUG=1)
  env_args+=(RUSTFLAGS="${RUSTFLAGS:-} -C debuginfo=1 -C force-frame-pointers=yes")
fi

args=(cargo run -p diskgraph-ingest --bin ingest --release -- \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --dest local)
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  args+=(--profile)
fi
if [[ "${PARALLEL_ENABLED}" == "true" ]]; then
  args+=(--parallel)
fi
env "${env_args[@]}" "${args[@]}"
