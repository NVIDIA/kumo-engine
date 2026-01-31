#!/usr/bin/env bash
set -euo pipefail

# Open the ingest performance TUI against the stats bucket.
#
# Quick usage:
#   ./diskgraph-ingest/scripts/perf_explorer.sh
#   ./diskgraph-ingest/scripts/perf_explorer.sh hnm
#
# Env:
#   DISKGRAPH_STATS_DEST   Stats root (s3://, gs://, az://).

DATASET="${1:-${DISKGRAPH_STATS_DATASET:-}}"

args=()
if [[ -n "${DATASET}" ]]; then
  args+=(--dataset "${DATASET}")
fi

env DISKGRAPH_STATS_DEST="${DISKGRAPH_STATS_DEST:-s3://kumo-unit-test/diskgraph-profiles}" \
  cargo run -p diskgraph-ingest --bin ingest_tui -- "${args[@]}"
