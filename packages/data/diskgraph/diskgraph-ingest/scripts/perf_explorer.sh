#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Open the ingest performance TUI against the stats bucket.
#
# Quick usage:
#   ./diskgraph-ingest/scripts/perf_explorer.sh
#   ./diskgraph-ingest/scripts/perf_explorer.sh hnm
#
# Env:
#   DISKGRAPH_STATS_DEST   Stats root (s3://, gs://, az://).

DATASET="${1:-${DISKGRAPH_STATS_DATASET:-}}"
STATS_DEST="${DISKGRAPH_STATS_DEST:-}"
METADATA_DEST=""

USE_AWS="false"
if [[ -z "${METADATA_DEST}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    if curl -fs --connect-timeout 0.2 -m 0.5 \
      -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/id" >/dev/null 2>&1; then
      METADATA_DEST="gs://kumo-unit-test/diskgraph-profiles"
    elif curl -fs --connect-timeout 0.2 -m 0.5 \
      -H "Metadata-Flavor: Google" \
      "http://169.254.169.254/computeMetadata/v1/instance/id" >/dev/null 2>&1; then
      METADATA_DEST="gs://kumo-unit-test/diskgraph-profiles"
    fi
  fi
fi

if [[ -z "${METADATA_DEST}" ]]; then
  if [[ -z "${AWS_EC2_METADATA_DISABLED:-}" ]]; then
    if command -v curl >/dev/null 2>&1; then
      AWS_IMDS_TOKEN="$(curl -s --connect-timeout 0.2 -m 0.5 -X PUT \
        "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)"
      if [[ -n "${AWS_IMDS_TOKEN}" ]]; then
        if curl -s --connect-timeout 0.2 -m 0.5 \
          -H "X-aws-ec2-metadata-token: ${AWS_IMDS_TOKEN}" \
          "http://169.254.169.254/latest/meta-data/instance-id" >/dev/null 2>&1; then
          METADATA_DEST="s3://kumo-unit-test/diskgraph-profiles"
          USE_AWS="true"
        fi
      fi
    fi
  fi
fi

if [[ -z "${METADATA_DEST}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    if curl -s --connect-timeout 0.2 -m 0.5 \
      -H "Metadata: true" \
      "http://169.254.169.254/metadata/instance?api-version=2021-02-01" >/dev/null 2>&1; then
      AZ_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-${AZURE_ACCOUNT:-account}}"
      AZ_CONTAINER="${AZURE_STORAGE_CONTAINER:-${AZURE_CONTAINER:-container}}"
      METADATA_DEST="az://${AZ_ACCOUNT}/${AZ_CONTAINER}/diskgraph-profiles"
    fi
  fi
fi

if [[ -n "${METADATA_DEST}" ]]; then
  STATS_DEST="${METADATA_DEST}"
fi

if [[ -z "${STATS_DEST}" ]]; then
  # Prefer SSO/profile creds if provided; otherwise rely on default chain (with SSO cache).
  export AWS_SDK_LOAD_CONFIG=1
  if command -v aws >/dev/null 2>&1; then
    AWS_PROFILE_TO_USE="${S3_PROFILE:-${AWS_PROFILE:-}}"
    if [[ -n "${AWS_PROFILE_TO_USE}" ]]; then
      export AWS_PROFILE="${AWS_PROFILE_TO_USE}"
    else
      # If no profile specified and there's exactly one non-default profile, use it.
      AWS_PROFILES_RAW="$(aws configure list-profiles 2>/dev/null | grep -v '^default$' || true)"
      # shellcheck disable=SC2086 # intentional word splitting into positional params
      set -- ${AWS_PROFILES_RAW}
      if [[ "$#" -eq 1 ]]; then
        export AWS_PROFILE="$1"
        echo "No AWS_PROFILE provided; auto-selecting profile '${AWS_PROFILE}'."
      fi
    fi
    if aws sts get-caller-identity >/dev/null 2>&1; then
      STATS_DEST="s3://kumo-unit-test/diskgraph-profiles"
      USE_AWS="true"
    fi
  fi
fi

if [[ "${STATS_DEST}" == s3://* ]]; then
  USE_AWS="true"
fi

args=()
if [[ -n "${DATASET}" ]]; then
  args+=(--dataset "${DATASET}")
fi

stats_env=()
if [[ -n "${STATS_DEST}" ]]; then
  stats_env+=(DISKGRAPH_STATS_DEST="${STATS_DEST}")
fi
env ${stats_env[@]+"${stats_env[@]}"} cargo run -p diskgraph-ingest --bin ingest_tui --features ingest-tui -- "${args[@]}"
