#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SDK Direct Upload Integration Test (MVP)
# Export the required values before running.
#   chmod +x run_sdk_direct_upload_integ_test.sh
#   ./run_sdk_direct_upload_integ_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# ============================
# REQUIRED
# ============================
export RUN_KUMO_INTEG="${RUN_KUMO_INTEG:-1}"
export KUMO_API_URL="${KUMO_API_URL:?Set KUMO_API_URL to the API endpoint}"
export KUMO_API_TOKEN="${KUMO_API_TOKEN:?Set KUMO_API_TOKEN before running}"

# Choose ONE cloud path for KUMO_DEST (uncomment & edit):
# --- AWS S3 ---
# export KUMO_DEST="s3://kumo-public-datasets/sdk_upload_integ_test/"
# --- Google Cloud Storage ---
# export KUMO_DEST="gs://bigquery-connector-testing/sdk_upload_integ_test/"
# --- Azure Data Lake Storage Gen2 (ABFS/Secure) ---
# export KUMO_DEST="abfs://container@account.dfs.core.windows.net/kumo-integ"
# export KUMO_DEST="abfss://container@account.dfs.core.windows.net/kumo-integ"
# Default if not overridden:
export KUMO_DEST="${KUMO_DEST:-s3://your-bucket/kumo-integ}"

# ============================
# OPTIONAL TOGGLES
# ============================
export KUMO_KEEP="${KUMO_KEEP:-0}"             # 1 = keep generated data (no cleanup)
export KUMO_HEAVY="${KUMO_HEAVY:-0}"           # 1 = run big parquet test (many shards)
export KUMO_INCLUDE_GZ="${KUMO_INCLUDE_GZ:-0}" # 1 = include .csv.gz failure test

# ============================
# OPTIONAL AUTH EXAMPLES
# Uncomment the block for your platform if your environment
# is not already authenticated via default mechanisms.
# ============================

# --- AWS S3 Auth (typical env vars or AWS CLI profile) ---
# export AWS_ACCESS_KEY_ID="AKIA..."
# export AWS_SECRET_ACCESS_KEY="..."
# export AWS_SESSION_TOKEN="..."          # (if using temporary creds)
# export AWS_DEFAULT_REGION="us-west-2"
# Or use an AWS profile:
# export AWS_PROFILE="my-profile"

# --- GCS Auth (Application Default Credentials or SA JSON) ---
# gcloud auth application-default login
# Or point to a service account key:
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"

# --- Azure ADLS Gen2 Auth (ABFS/ABFSS) ---
# Using account key:
# export AZURE_STORAGE_ACCOUNT="myaccount"
# export AZURE_STORAGE_KEY="myaccountkey=="
# Using SAS (for abfss):
# export AZURE_STORAGE_SAS_TOKEN="?sv=...&sig=..."
# Or interactive login (Azure CLI):
# az login

echo "=== SDK Direct Upload Integration Test ==="
echo "RUN_KUMO_INTEG=$RUN_KUMO_INTEG"
echo "KUMO_API_URL=$KUMO_API_URL"
echo "KUMO_DEST=$KUMO_DEST"
echo "KUMO_KEEP=$KUMO_KEEP  KUMO_HEAVY=$KUMO_HEAVY  KUMO_INCLUDE_GZ=$KUMO_INCLUDE_GZ"
echo "=========================================="

"$KUMO_SDK_PYTHON" -m pytest -s -vv \
  test/connector/test_kumo_integration_e2e.py
