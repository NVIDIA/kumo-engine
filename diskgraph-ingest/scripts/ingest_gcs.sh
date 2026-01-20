#!/usr/bin/env bash
set -euo pipefail

# Run diskgraph-ingest on a gs:// source prefix or diskgraph_config.json URL.
#
# Quick usage:
#   ./diskgraph-ingest/scripts/ingest_gcs.sh gs://bucket/prefix
#   ./diskgraph-ingest/scripts/ingest_gcs.sh -p gs://bucket/prefix
#   ./diskgraph-ingest/scripts/ingest_gcs.sh -d gs://bucket/prefix
#   ./diskgraph-ingest/scripts/ingest_gcs.sh -m gs://bucket/prefix
#
# Flags:
#   -p  Enable pprof profiling output.
#   -d  Build with debuginfo + frame pointers for stack dumps.
#   -m  Enable jemalloc heap dump watchdog.

ULIMIT_NOFILE="${ULIMIT_NOFILE:-100000}"
if ! ulimit -n "${ULIMIT_NOFILE}" 2>/dev/null; then
  echo "warning: failed to raise ulimit -n to ${ULIMIT_NOFILE}; current limit may be too low" >&2
fi

PROFILE_ENABLED="false"
DEBUGINFO_ENABLED="false"
MEM_PROFILE_ENABLED="false"
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
    -m)
      MEM_PROFILE_ENABLED="true"
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
  echo "Usage: SOURCE=gs://bucket/prefix [OUTPUT=path] $0 [-p] [-d] [-m]" >&2
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
  INPUT_NAME="$(basename "$(dirname "${SOURCE}")")"
  OUTPUT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}/${INPUT_NAME}"
fi

env_args=()
if [[ "${DEBUGINFO_ENABLED}" == "true" ]]; then
  env_args+=(CARGO_PROFILE_RELEASE_DEBUG=1)
  env_args+=(RUSTFLAGS="${RUSTFLAGS:-} -C debuginfo=1 -C force-frame-pointers=yes")
fi
base_env=(PATH="${PATH}" HOME="${HOME}" USER="${USER}")

USE_CARGO_RUN="true"
if [[ "${MEM_PROFILE_ENABLED}" == "true" ]]; then
  if [[ "${DEBUGINFO_ENABLED}" == "false" ]]; then
    env_args+=(CARGO_PROFILE_RELEASE_DEBUG=1)
    env_args+=(RUSTFLAGS="${RUSTFLAGS:-} -C debuginfo=1 -C force-frame-pointers=yes")
  fi
  env -i "${base_env[@]}" \
    MALLOC_CONF= _RJEM_MALLOC_CONF= _rjem_MALLOC_CONF= \
    JEMALLOC_CONF= JEMALLOC_SYS_WITH_MALLOC_CONF= JEMALLOC_OPTIONS= \
    "${env_args[@]}" \
    cargo build -p diskgraph-ingest --bin ingest --release --features mem-profile
  env_args+=(MALLOC_CONF= _rjem_MALLOC_CONF=)
  env_args+=(_RJEM_MALLOC_CONF="${_RJEM_MALLOC_CONF:-prof:true}")
  env_args+=(JEMALLOC_CONF= JEMALLOC_SYS_WITH_MALLOC_CONF= JEMALLOC_OPTIONS=)
  args=(./target/release/ingest)
  USE_CARGO_RUN="false"
elif [[ -n "${MALLOC_CONF:-}" || -n "${_RJEM_MALLOC_CONF:-}" || -n "${_rjem_MALLOC_CONF:-}" || -n "${JEMALLOC_CONF:-}" || -n "${JEMALLOC_SYS_WITH_MALLOC_CONF:-}" || -n "${JEMALLOC_OPTIONS:-}" ]]; then
  env_args+=(MALLOC_CONF= _RJEM_MALLOC_CONF= _rjem_MALLOC_CONF=)
  env_args+=(JEMALLOC_CONF= JEMALLOC_SYS_WITH_MALLOC_CONF= JEMALLOC_OPTIONS=)
  args=(cargo run -p diskgraph-ingest --bin ingest --release)
else
  args=(cargo run -p diskgraph-ingest --bin ingest --release)
fi
if [[ "${USE_CARGO_RUN}" == "true" ]]; then
  args+=(-- --source "${SOURCE}" --output "${OUTPUT}" --dest local)
else
  args+=(--source "${SOURCE}" --output "${OUTPUT}" --dest local)
fi
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  args+=(--profile)
fi
if [[ "${MEM_PROFILE_ENABLED}" == "true" ]]; then
  args+=(--mem-profile)
fi
if [[ "${PARALLEL_ENABLED}" == "true" ]]; then
  args+=(--parallel)
fi
env ${env_args[@]+"${env_args[@]}"} "${args[@]}"
