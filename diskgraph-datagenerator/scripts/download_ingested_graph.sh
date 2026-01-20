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

cargo run -p diskgraph-datagenerator --bin download-ingested-graph --release -- "${positional[@]}"

if [[ "$dry_run" -eq 1 ]]; then
  exit 0
fi

customer=""
job_id=""
if [[ "${#positional[@]}" -eq 1 && "${positional[0]}" == http* ]]; then
  url="${positional[0]}"
  customer="$(echo "$url" | awk -F[/:] '{print $4}' | cut -d. -f1)"
  job_id="$(echo "$url" | sed -n 's#.*/\\(bp-job-[^/?]*\\).*#\\1#p')"
elif [[ "${#positional[@]}" -ge 2 ]]; then
  customer="${positional[0]}"
  job_id="${positional[1]}"
fi

if [[ -n "$customer" && -n "$job_id" ]]; then
  out_dir="${out_root}/${customer}_${job_id}"
  echo "Next steps:"
  echo "  ./diskgraph-infer/scripts/infer_source.sh \"${out_dir}\""
else
  echo "Next steps:"
  echo "  ./diskgraph-infer/scripts/infer_source.sh <downloaded_path>"
fi
