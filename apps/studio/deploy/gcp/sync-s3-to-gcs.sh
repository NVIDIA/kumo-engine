#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sync a dataset from S3 to GCS via a local temp directory.
#
# Short form  (bare names, no cloud prefix):
#   ./sync-s3-to-gcs.sh <dataset> <instance-id>
#   ./sync-s3-to-gcs.sh hnm_tiny elvin-test
#     -> s3://kumo-public-datasets/hnm_tiny/  to  gs://kumo-elvin-test/datasets/hnm_tiny/
#
# Full form  (explicit cloud URIs override both defaults):
#   ./sync-s3-to-gcs.sh s3://my-bucket/foo/ gs://other-bucket/bar/
#
# Mixed forms also work — prefix detection is per-argument.
#
# Prerequisites:
#   aws configure (or AWS_PROFILE / AWS_ACCESS_KEY_ID set)
#   gcloud auth login --update-adc
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset|s3-uri> <instance-id|gs-uri>"
    echo "  Short: $0 hnm_tiny elvin-test"
    echo "  Full:  $0 s3://kumo-public-datasets/hnm_tiny/ gs://elvin-test/datasets/hnm_tiny/"
    exit 1
fi

ARG1="$1"
ARG2="$2"

# Resolve S3 source
if [[ "$ARG1" == s3://* ]]; then
    S3_SOURCE="$ARG1"
    DATASET_NAME=$(echo "$S3_SOURCE" | sed 's|/*$||; s|.*/||')
else
    DATASET_NAME="$ARG1"
    S3_SOURCE="s3://kumo-public-datasets/${DATASET_NAME}/"
fi

# Resolve GCS dest
if [[ "$ARG2" == gs://* ]]; then
    GCS_DEST="$ARG2"
else
    GCS_DEST="gs://kumo-${ARG2}/datasets/${DATASET_NAME}/"
fi

TMPDIR=$(mktemp -d)
LOCAL_PATH="${TMPDIR}/${DATASET_NAME}"
trap 'echo "Cleaning up ${TMPDIR}..."; rm -rf "${TMPDIR}"' EXIT

# --- preflight checks ---
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "ERROR: AWS credentials not found. Configure via aws configure or AWS_PROFILE."
    exit 1
fi

if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "ERROR: GCP credentials not found. Run: gcloud auth login --update-adc"
    exit 1
fi

echo "S3 source:  ${S3_SOURCE}"
echo "GCS dest:   ${GCS_DEST}"
echo "Local tmp:  ${LOCAL_PATH}"
echo ""

# --- download from S3 ---
echo "==> Downloading from S3..."
aws s3 sync "${S3_SOURCE}" "${LOCAL_PATH}/"

# --- upload to GCS ---
echo ""
echo "==> Uploading to GCS..."
gsutil -m rsync -r "${LOCAL_PATH}/" "${GCS_DEST}"

echo ""
echo "Done. Files at ${GCS_DEST}"
