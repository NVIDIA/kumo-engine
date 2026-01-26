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

if [[ -n "${DISKGRAPH_DATA_ROOT:-}" ]]; then
  default_root="${DISKGRAPH_DATA_ROOT}"
elif [[ -d /mnt/localssd && -w /mnt/localssd ]]; then
  default_root="/mnt/localssd"
else
  default_root="/tmp"
fi

out_root="${default_root%/}/datasets"
out_root_set=0
extra_args=()
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      out_root="$2"
      out_root_set=1
      extra_args+=("$1" "$2")
      shift 2
      ;;
    --dry-run)
      extra_args+=("$1")
      shift
      ;;
    --*)
      extra_args+=("$1")
      shift
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if [[ $out_root_set -eq 0 ]]; then
  extra_args+=("--out-root" "$out_root")
fi

exec cargo run -p diskgraph-datagenerator --bin download-ingested-graph --release -- \
  "${extra_args[@]}" \
  "${positional[@]}"
