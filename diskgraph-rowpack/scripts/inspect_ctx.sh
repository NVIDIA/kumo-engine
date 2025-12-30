#!/usr/bin/env bash
set -euo pipefail

CTX_PATH="${CTX_PATH:-${1:-}}"
if [[ -z "${CTX_PATH}" ]]; then
  echo "usage: $0 /path/to/context.ctx" >&2
  exit 1
fi

cargo run -p diskgraph-rowpack --bin rowpack_ctx_inspect --release -- "${CTX_PATH}"
