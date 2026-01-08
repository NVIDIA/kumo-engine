#!/usr/bin/env bash
set -euo pipefail

# Explore a local index directory (must contain index_metadata.json).

ULIMIT_NOFILE="${ULIMIT_NOFILE:-100000}"
if ! ulimit -n "${ULIMIT_NOFILE}" 2>/dev/null; then
  echo "warning: failed to raise ulimit -n to ${ULIMIT_NOFILE}; current limit may be too low" >&2
fi

SHARD_PATH="${SHARD_PATH:-${1:-}}"
if [[ -z "${SHARD_PATH}" ]]; then
  echo "usage: $0 /path/to/index" >&2
  exit 1
fi

echo "Exploring local index at ${SHARD_PATH}"
cargo run -p diskgraph-index --bin explore --release -- "${SHARD_PATH}"
