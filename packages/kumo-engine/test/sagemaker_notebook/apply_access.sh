#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

command -v aws >/dev/null
command -v jq >/dev/null
access_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
access_account=$(aws sts get-caller-identity --query Account --output text)
if [ "$access_account" != 926922431314 ]; then
  echo 'Refusing IAM changes outside AWS account 926922431314.' >&2
  exit 1
fi
access_tmp=$(mktemp -d)
trap 'rm -rf "$access_tmp"' EXIT HUP INT TERM

aws iam get-role --role-name gha-kumo-workflow-tests \
  --query Role.AssumeRolePolicyDocument --output json > "$access_tmp/before.json"
jq --slurpfile addition "$access_dir/runner-trust-addition.json" '
  .Statement |= (if type == "array" then . else [.] end) |
  if any(.Statement[]; . == $addition[0]) then .
  else .Statement += $addition end
' "$access_tmp/before.json" > "$access_tmp/after.json"

aws iam get-role --role-name gha-kumo-workflow-tests \
  --query Role.AssumeRolePolicyDocument --output json > "$access_tmp/current.json"
jq -S . "$access_tmp/before.json" > "$access_tmp/before-sorted.json"
jq -S . "$access_tmp/current.json" > "$access_tmp/current-sorted.json"
if ! cmp -s "$access_tmp/before-sorted.json" "$access_tmp/current-sorted.json"; then
  echo 'Runner trust changed during preparation; retry after reviewing the change.' >&2
  exit 1
fi

aws iam update-assume-role-policy --role-name gha-kumo-workflow-tests \
  --policy-document "file://$access_tmp/after.json"
aws iam put-role-policy --role-name gha-kumo-workflow-tests \
  --policy-name kumo-engine-sagemaker-notebook \
  --policy-document "file://$access_dir/runner-permissions.json"
aws iam put-role-policy --role-name kumo-mlops-validation-sagemaker \
  --policy-name kumo-engine-sagemaker-notebook \
  --policy-document "file://$access_dir/execution-permissions.json"
echo 'Applied SageMaker notebook permissions to the existing test roles.'
