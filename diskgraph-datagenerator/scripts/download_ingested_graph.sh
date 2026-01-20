#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the Rust downloader.
#
# Usage:
#   ./diskgraph-datagenerator/scripts/download_ingested_graph.sh <customer_name> <job_id>
#   ./diskgraph-datagenerator/scripts/download_ingested_graph.sh <job_url>
#
# Optional flags (passed through):
#   --out-root /mnt/localssd/datasets
#   DISKGRAPH_DATA_ROOT=/custom/root (overrides /mnt/localssd when --out-root not set)
#   --dry-run

out_root="${DISKGRAPH_DATA_ROOT:-/mnt/localssd}/datasets"
dry_run=0
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      out_root="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --*)
      shift
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

exec cargo run -p diskgraph-datagenerator --bin download-ingested-graph --release -- "${positional[@]}"
