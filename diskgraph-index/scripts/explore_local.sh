#!/usr/bin/env bash
set -euo pipefail

# Explore a local shard directory (must contain offsets.bin, edges.bin, metadata.json).

PROFILE_ENABLED="false"
while getopts "p" opt; do
  case "${opt}" in
    p)
      PROFILE_ENABLED="true"
      ;;
    *)
      echo "usage: $0 [-p] /path/to/shard" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

SHARD_PATH="${SHARD_PATH:-${1:-}}"
if [[ -z "${SHARD_PATH}" ]]; then
  echo "usage: $0 [-p] /path/to/shard" >&2
  exit 1
fi

echo "Exploring local shard at ${SHARD_PATH}"
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  cargo run -p diskgraph-index --bin explore --release -- --profile "explore-profile" "${SHARD_PATH}"
else
  cargo run -p diskgraph-index --bin explore --release -- "${SHARD_PATH}"
fi
