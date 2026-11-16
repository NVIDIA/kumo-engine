#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helper script to run packer build with AWS SSO credentials.
# Usage: ./packer-build.sh <kumo_image_tag>
#
# Packer does not support AWS SSO profiles natively. This script
# extracts creds from the AWS CLI SSO cache, exports them as env vars
# for Packer's own API calls, and passes them as -var flags for the
# build instance's ECR docker pull.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <image-tag>"
    echo "  <image-tag>: commit SHA (full 40-char) or version tag (e.g. v2.20.3)"
    exit 1
fi

KUMO_IMAGE_TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Verify AWS SSO session is valid
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS SSO session expired. Run: aws sso login"
    exit 1
fi

# Extract creds from SSO cache
eval "$(python3 -c "
import json, glob, os
for f in sorted(glob.glob(os.path.expanduser('~/.aws/cli/cache/*.json')), key=os.path.getmtime, reverse=True):
    d = json.load(open(f))
    if 'Credentials' in d:
        c = d['Credentials']
        print(f\"export AWS_ACCESS_KEY_ID={c['AccessKeyId']}\")
        print(f\"export AWS_SECRET_ACCESS_KEY={c['SecretAccessKey']}\")
        print(f\"export AWS_SESSION_TOKEN={c['SessionToken']}\")
        break
")"

if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "ERROR: Could not extract credentials from ~/.aws/cli/cache/"
    echo "Try: aws sso login && aws sts get-caller-identity"
    exit 1
fi

echo "AWS creds extracted (key=${AWS_ACCESS_KEY_ID:0:8}...)"
echo "Building AMI for tag: ${KUMO_IMAGE_TAG}"

cd "$SCRIPT_DIR"
packer init kumo-singlehost.pkr.hcl

exec packer build \
    -var "aws_access_key_id=${AWS_ACCESS_KEY_ID}" \
    -var "aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}" \
    -var "aws_session_token=${AWS_SESSION_TOKEN}" \
    -var "kumo_image_tag=${KUMO_IMAGE_TAG}" \
    kumo-singlehost.pkr.hcl
