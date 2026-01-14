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
#   --max-requests 64
#   --part-size-mb 64
#   --multipart-threshold-mb 128
#   --part-concurrency 8
#   --dry-run

exec cargo run -p diskgraph-datagenerator --bin download-ingested-graph --release -- "$@"
