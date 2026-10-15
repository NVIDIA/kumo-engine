#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Run DiskGraph validation on customer jobs
#
# Usage:
#   ./scripts/run_validation.sh [OPTIONS] <job1> [job2] ...
#
# Options:
#   --bucket <name>           Required S3 bucket
#   --customer-prefix <name>  Storage prefix, if different from the customer name
#   --region <region>         AWS region (uses the AWS configuration by default)
#   --output <dir>            Output directory (default: temp_data/index)
#   --clean                   Remove validation_report.json before running
#   --help                    Show this help
#
# Example:
#   ./scripts/run_validation.sh --bucket example-bucket \
#     "example trainingjob-abc" "example bp-job-xyz"

set -e

# Defaults
BUCKET=""
STORAGE_ARGS=()
REGION=""
OUTPUT_ROOT="temp_data/index"
DOWNLOAD_ROOT="temp_data/raw"
CLEAN=false
JOBS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --customer-prefix)
            STORAGE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --output)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            sed -n '2,/^set -e/{ /^set -e/d; p; }' "$0" | sed 's/^#//' | sed 's/^ //'
            exit 0
            ;;
        *)
            JOBS+=("$1")
            shift
            ;;
    esac
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "Error: No jobs specified"
    echo "Usage: $0 [OPTIONS] <job1> [job2] ..."
    echo "Run '$0 --help' for more information"
    exit 1
fi

if [ -z "$BUCKET" ]; then
    echo "Error: --bucket is required" >&2
    exit 1
fi

# Set region environment variable
if [ -n "$REGION" ]; then
    export AWS_REGION="$REGION"
    export AWS_DEFAULT_REGION="$REGION"
fi

echo "=== DiskGraph Validation ==="
echo "Bucket: $BUCKET"
echo "Region: ${AWS_REGION:-default}"
echo "Output: $OUTPUT_ROOT"
echo "Download: $DOWNLOAD_ROOT"
echo "Jobs: ${JOBS[*]}"
echo ""

# Build
echo ">>> Building release binary..."
cargo build --release --bin validate-s3-ingest

# Create directories
mkdir -p "$OUTPUT_ROOT" "$DOWNLOAD_ROOT"

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo ">>> Removing old validation_report.json"
    rm -f validation_report.json
fi

# Run validation
echo ""
echo ">>> Running validation..."
cargo run --release --bin validate-s3-ingest -- \
    --bucket "$BUCKET" "${STORAGE_ARGS[@]}" \
    --output-root "$OUTPUT_ROOT" \
    --download-root "$DOWNLOAD_ROOT" \
    --keep --resume \
    "${JOBS[@]}"

echo ""
echo "=== Done ==="
echo "Results: validation_report.json"
