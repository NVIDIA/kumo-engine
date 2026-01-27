#!/usr/bin/env bash
set -euo pipefail

INDEX_PATH="${1:-}"
shift || true

if [[ -z "${INDEX_PATH}" ]]; then
  echo "usage: ttui.sh <index_path> [-- -C \"PREDICT ...\" | -o output_dir]" >&2
  exit 2
fi

cargo run -p diskgraph-train-table --bin ttui -- "${INDEX_PATH}" "$@"
