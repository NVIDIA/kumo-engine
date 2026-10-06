#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# validate_ttgen.sh — Production TT-gen validation pipeline
#
# Converts Kumo config to DiskGraph config, ingests data (from S3 directly),
# downloads Kumo's train_table output, runs our TT-gen executor, and compares.
#
# Usage:
#   ./tools/validate_ttgen.sh <customer> <job_id> [options]
#
# Options:
#   --bucket <name>       S3 bucket (default: kumo-prod-dataplane-bucket)
#   --index-path <path>   Reuse existing DiskGraph index instead of building one
#   --keep                Keep downloaded data after validation
#   --skip-download       Skip S3 downloads (reuse existing WORK_DIR)
#   --plan-only           Only compile and print the plan, don't execute
#
# Examples:
#   # Full E2E from S3:
#   ./tools/validate_ttgen.sh akihironitta-staging trainingjob-32611f77... --keep
#
#   # Reuse existing index:
#   ./tools/validate_ttgen.sh faire trainingjob-026fd28d... \
#       --index-path /mnt/localssd/faire_data/index --keep
#
# Prerequisites:
#   - AWS credentials configured (aws sso login --profile normal-profile)
#   - diskgraph pybind wheel installed (scripts/build_python_wheel.sh + pip install)
#   - kumoapi on PYTHONPATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── Defaults ────────────────────────────────────────────────────────────────

BUCKET="kumo-prod-dataplane-bucket"
INDEX_PATH=""
KEEP=false
SKIP_DOWNLOAD=false
PLAN_ONLY=false

# ─── Parse args ──────────────────────────────────────────────────────────────

if [ $# -lt 2 ]; then
    cat <<'EOF'
Usage: validate_ttgen.sh <customer> <job_id> [options]

Options:
  --bucket <name>       S3 bucket (default: kumo-prod-dataplane-bucket)
  --index-path <path>   Reuse existing DiskGraph index
  --keep                Keep downloaded data after validation
  --skip-download       Skip S3 downloads
  --plan-only           Only compile and print the plan
EOF
    exit 1
fi

CUSTOMER="$1"; shift
JOB_ID="$1"; shift

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket)        BUCKET="$2"; shift 2 ;;
        --index-path)    INDEX_PATH="$2"; shift 2 ;;
        --keep)          KEEP=true; shift ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --plan-only)     PLAN_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Setup ───────────────────────────────────────────────────────────────────

JOB_SHORT="${JOB_ID:0:20}"
WORK_DIR="/mnt/localssd/ttgen_validate/${CUSTOMER}_${JOB_SHORT}"
S3_PREFIX="s3://${BUCKET}/${CUSTOMER}/prod"
S3_JOB="${S3_PREFIX}/jobs/${JOB_ID}"

export AWS_PROFILE="${AWS_PROFILE:-normal-profile}"
export S3_PROFILE="${S3_PROFILE:-$AWS_PROFILE}"

TOTAL_START=$(date +%s%N)

# Map bucket to region for convert_kumo_config
if [ "$BUCKET" = "kumo-prod-ue1-dataplane-bucket" ]; then
    REGION="ue1"
else
    REGION="uw2"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  TT-gen Validation: ${CUSTOMER} / ${JOB_SHORT}"
echo "  Bucket: ${BUCKET} (region: ${REGION})"
echo "  Work dir: ${WORK_DIR}"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p "$WORK_DIR"

elapsed_ms() {
    local start_ns=$1 end_ns=$(date +%s%N)
    echo $(( (end_ns - start_ns) / 1000000 ))
}

# ─── Step 1: Download training_job_input.json ────────────────────────────────

JOB_CONFIG="${WORK_DIR}/training_job_input.json"

if [ "$SKIP_DOWNLOAD" = false ] || [ ! -f "$JOB_CONFIG" ]; then
    echo ""
    echo "▶ Step 1: Downloading training_job_input.json..."
    STEP_START=$(date +%s%N)
    aws s3 cp "${S3_JOB}/inputs/training_job_input.json" "$JOB_CONFIG" --quiet
    echo "  ⏱  $(elapsed_ms $STEP_START) ms"
else
    echo ""
    echo "▶ Step 1: Using existing job config at ${JOB_CONFIG}"
fi

# ─── Step 2: Download Kumo's training table output ───────────────────────────

KUMO_TT_DIR="${WORK_DIR}/kumo_train_table"

if [ "$SKIP_DOWNLOAD" = false ] || [ ! -d "$KUMO_TT_DIR" ] || [ -z "$(find "$KUMO_TT_DIR" -name '*.parquet' 2>/dev/null | head -1)" ]; then
    echo ""
    echo "▶ Step 2: Downloading Kumo's training table output..."
    STEP_START=$(date +%s%N)

    mkdir -p "$KUMO_TT_DIR"

    # Download from train_table/NA/train_table/ (Spark-generated, always exists)
    TT_S3="${S3_JOB}/train_table/NA/train_table/"
    echo "  Downloading from ${TT_S3} ..."
    aws s3 sync "$TT_S3" "$KUMO_TT_DIR/" --quiet || true

    TT_COUNT=$(find "$KUMO_TT_DIR" -name "*.parquet" 2>/dev/null | wc -l)
    echo "  Downloaded: ${TT_COUNT} train_table parquet files"
    if [ "$TT_COUNT" -eq 0 ]; then
        echo "  ERROR: No parquet files found in Kumo train_table output"
        exit 1
    fi
    echo "  ⏱  $(elapsed_ms $STEP_START) ms"
else
    TT_COUNT=$(find "$KUMO_TT_DIR" -name "*.parquet" 2>/dev/null | wc -l)
    echo ""
    echo "▶ Step 2: Using existing Kumo train_table (${TT_COUNT} files)"
fi

# ─── Step 3: Convert config + build DiskGraph index ──────────────────────────

if [ -z "$INDEX_PATH" ]; then
    INDEX_PATH="${WORK_DIR}/index"
fi

DISKGRAPH_CONFIG="${WORK_DIR}/diskgraph_config.json"

if [ -f "$INDEX_PATH/index_metadata.json" ]; then
    echo ""
    echo "▶ Step 3: Using existing index at ${INDEX_PATH}"
else
    # Step 3a: Convert Kumo config to DiskGraph config
    echo ""
    echo "▶ Step 3a: Converting Kumo config to DiskGraph config..."
    STEP_START=$(date +%s%N)

    cd "$REPO_DIR"
    cargo run --release -p diskgraph-ingest --bin convert_kumo_config -- \
        --customer "$CUSTOMER" \
        --job "$JOB_ID" \
        --region "$REGION" \
        --profile "$AWS_PROFILE" \
        --output "$DISKGRAPH_CONFIG"

    echo "  ⏱  $(elapsed_ms $STEP_START) ms"

    # Step 3b: Build DiskGraph index (reads from S3 directly)
    echo ""
    echo "▶ Step 3b: Building DiskGraph index (ingesting from S3)..."
    STEP_START=$(date +%s%N)

    cargo run --release -p diskgraph-ingest --bin ingest -- \
        --source "$DISKGRAPH_CONFIG" \
        --output "$INDEX_PATH" \
        --col-stats \
        --cat-count-strategy hybrid

    echo "  ⏱  $(elapsed_ms $STEP_START) ms"
fi

# ─── Step 4: Check pybind ────────────────────────────────────────────────────

echo ""
echo "▶ Step 4: Checking pybind..."
if python3 -c "import diskgraph._c" 2>/dev/null; then
    echo "  pybind OK"
else
    echo "  ERROR: diskgraph pybind not installed."
    echo "  Run: python3 -m pip install ."
    echo "  See diskgraph-pybind/README.md for wheel build options."
    exit 1
fi

# ─── Step 5: Run TT-gen + comparison ────────────────────────────────────────

PLAN_ONLY_FLAG=""
if [ "$PLAN_ONLY" = true ]; then
    PLAN_ONLY_FLAG="--plan-only"
fi

echo ""
echo "▶ Step 5: Running TT-gen comparison..."
STEP_START=$(date +%s%N)
PYTHONPATH="/home/vedaant/kumo-api:${PYTHONPATH:-}" \
python3 "$SCRIPT_DIR/ttgen_compare.py" \
    --job-config "$JOB_CONFIG" \
    --index-path "$INDEX_PATH" \
    --kumo-tt-dir "$KUMO_TT_DIR" \
    $PLAN_ONLY_FLAG
echo "  ⏱  $(elapsed_ms $STEP_START) ms"

# ─── Cleanup ────────────────────────────────────────────────────────────────

if [ "$KEEP" = false ]; then
    echo ""
    echo "▶ Cleaning up ${WORK_DIR}..."
    rm -rf "$WORK_DIR"
fi

TOTAL_ELAPSED=$(( ($(date +%s%N) - TOTAL_START) / 1000000 ))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done. Total time: ${TOTAL_ELAPSED} ms"
echo "═══════════════════════════════════════════════════════════════"
