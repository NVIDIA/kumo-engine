#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Delete a Kumo singlehost GCP instance.
# Usage: ./teardown-instance-gcp.sh <instance-name>
#
# The boot disk and any attached local SSDs are deleted with the instance.
# Set FORCE=1 to skip the confirmation prompt.
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project <project-id>
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <instance-name>"
    exit 1
fi

INSTANCE_NAME="$1"

if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "ERROR: gcloud user credentials not found. Run: gcloud auth login"
    exit 1
fi

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}"
if [ -z "$GCP_PROJECT" ]; then
    echo "ERROR: GCP project not set. Either:"
    echo "  export GCP_PROJECT=<project-id>"
    echo "  gcloud config set project <project-id>"
    exit 1
fi

GCP_ZONE="${GCP_ZONE:-us-west1-b}"
FORCE="${FORCE:-0}"

# Verify the instance exists before prompting
if ! gcloud compute instances describe "${INSTANCE_NAME}" \
        --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" >/dev/null 2>&1; then
    echo "ERROR: Instance '${INSTANCE_NAME}' not found in project ${GCP_PROJECT} / zone ${GCP_ZONE}"
    exit 1
fi

echo "Instance:   ${INSTANCE_NAME}"
echo "Project:    ${GCP_PROJECT}"
echo "Zone:       ${GCP_ZONE}"
echo ""
echo "WARNING: This will permanently delete the instance and its boot disk."
echo "         Local SSDs (if attached) are also destroyed."

if [ "${FORCE}" != "1" ]; then
    read -r -p "Type the instance name to confirm: " confirm
    if [ "${confirm}" != "${INSTANCE_NAME}" ]; then
        echo "Aborted."
        exit 1
    fi
fi

gcloud compute instances delete "${INSTANCE_NAME}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --quiet

echo "Instance '${INSTANCE_NAME}' deleted."
