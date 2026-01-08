#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the Rust downloader.
#
# Usage:
#   ./scripts/download_ingested_graph.sh <customer_name> <job_id>
#   ./scripts/download_ingested_graph.sh <job_url>
#
# Optional flags (passed through):
#   --out-root /mnt/localssd/datasets
#   --max-requests 64
#   --part-size-mb 64
#   --multipart-threshold-mb 128
#   --part-concurrency 8
#   --dry-run

exec cargo run -p diskgraph-datagenerator --bin download-ingested-graph --release -- "$@"
