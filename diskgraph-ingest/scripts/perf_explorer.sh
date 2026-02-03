#!/usr/bin/env bash
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

USE_AWS="false"
if [[ "${STATS_DEST}" == s3://* ]]; then
  USE_AWS="true"
elif [[ -z "${STATS_DEST}" ]]; then
  if [[ -n "${AWS_REGION:-}" || -n "${AWS_DEFAULT_REGION:-}" || -n "${AWS_PROFILE:-}" || -n "${S3_PROFILE:-}" ]]; then
    USE_AWS="true"
  fi
fi

if [[ "${USE_AWS}" == "true" ]]; then
  # Prefer SSO/profile creds if provided; otherwise rely on default chain (with SSO cache).
  export AWS_SDK_LOAD_CONFIG=1
  IMDS_AVAILABLE="false"
  if [[ -z "${AWS_EC2_METADATA_DISABLED:-}" ]]; then
    if command -v curl >/dev/null 2>&1; then
      IMDS_TOKEN="$(curl -s --connect-timeout 0.2 -m 0.5 -X PUT \
        "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)"
      if [[ -n "${IMDS_TOKEN}" ]]; then
        if curl -s --connect-timeout 0.2 -m 0.5 \
          -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" \
          "http://169.254.169.254/latest/meta-data/instance-id" >/dev/null 2>&1; then
          IMDS_AVAILABLE="true"
        fi
      fi
    fi
    if [[ "${IMDS_AVAILABLE}" == "true" ]]; then
      export AWS_EC2_METADATA_DISABLED=false
    else
      export AWS_EC2_METADATA_DISABLED=true
    fi
  fi

  AWS_PROFILE_TO_USE="${S3_PROFILE:-${AWS_PROFILE:-}}"
  if [[ -n "${AWS_PROFILE_TO_USE}" ]]; then
    export AWS_PROFILE="${AWS_PROFILE_TO_USE}"
  elif [[ "${IMDS_AVAILABLE}" != "true" ]]; then
    # If no profile specified and there's exactly one non-default profile, use it.
    AWS_PROFILES_RAW="$(aws configure list-profiles 2>/dev/null | grep -v '^default$' || true)"
    # shellcheck disable=SC2086 # intentional word splitting into positional params
    set -- ${AWS_PROFILES_RAW}
    if [[ "$#" -eq 1 ]]; then
      export AWS_PROFILE="$1"
      echo "No AWS_PROFILE provided; auto-selecting profile '${AWS_PROFILE}'."
    fi
  fi
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "ERROR: Unable to call sts:GetCallerIdentity. Ensure AWS creds/role are configured (e.g., run 'aws sso login' or set AWS_PROFILE/S3_PROFILE to an SSO-enabled profile). On EC2, ensure IMDS is reachable and AWS_EC2_METADATA_DISABLED is not set to true." >&2
    exit 1
  fi
fi

args=()
if [[ -n "${DATASET}" ]]; then
  args+=(--dataset "${DATASET}")
fi

stats_env=()
if [[ -n "${STATS_DEST}" ]]; then
  stats_env+=(DISKGRAPH_STATS_DEST="${STATS_DEST}")
fi
env ${stats_env[@]+"${stats_env[@]}"} cargo run -p diskgraph-ingest --bin ingest_tui -- "${args[@]}"
