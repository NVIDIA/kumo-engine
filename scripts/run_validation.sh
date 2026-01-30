#!/bin/bash
#
# Run DiskGraph validation on customer jobs
#
# Usage:
#   ./scripts/run_validation.sh [OPTIONS] <job1> [job2] ...
#
# Options:
#   --bucket <name>     S3 bucket shortcut or full name
#                       Shortcuts: "prod-ue1" -> kumo-prod-ue1-dataplane-bucket (us-east-1)
#                                  "prod-uw2" -> kumo-prod-dataplane-bucket (us-west-2, default)
#   --region <region>   AWS region (auto-set based on bucket shortcut)
#   --output <dir>      Output directory for indices (default: temp_data/index)
#   --clean             Remove validation_report.json before running
#   --help              Show this help
#
# Examples:
#   # US-East-1 bucket
#   ./scripts/run_validation.sh --bucket prod-ue1 "allstate trainingjob-abc123"
#
#   # US-West-2 bucket (default)
#   ./scripts/run_validation.sh --bucket prod-uw2 "allianz trainingjob-xyz789"
#
#   # Multiple jobs
#   ./scripts/run_validation.sh --bucket prod-ue1 \
#     "allstate trainingjob-abc" \
#     "expedia bp-job-xyz"

set -e

# Defaults
BUCKET="kumo-prod-dataplane-bucket"
REGION=""
OUTPUT_ROOT="temp_data/index"
DOWNLOAD_ROOT="temp_data/raw"
CLEAN=false
JOBS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bucket)
            case $2 in
                prod-ue1)
                    BUCKET="kumo-prod-ue1-dataplane-bucket"
                    REGION="${REGION:-us-east-1}"
                    ;;
                prod-uw2)
                    BUCKET="kumo-prod-dataplane-bucket"
                    REGION="${REGION:-us-west-2}"
                    ;;
                *)
                    BUCKET="$2"
                    ;;
            esac
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
            head -30 "$0" | tail -n +2 | sed 's/^#//' | sed 's/^ //'
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
    --bucket "$BUCKET" \
    --output-root "$OUTPUT_ROOT" \
    --download-root "$DOWNLOAD_ROOT" \
    --keep --resume \
    "${JOBS[@]}"

echo ""
echo "=== Done ==="
echo "Results: validation_report.json"
