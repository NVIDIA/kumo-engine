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

# Prefer SSO/profile creds if provided; otherwise rely on default chain (with SSO cache).
export AWS_EC2_METADATA_DISABLED=true
export AWS_SDK_LOAD_CONFIG=1
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
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: Unable to call sts:GetCallerIdentity. Ensure AWS creds/role are configured (e.g., run 'aws sso login' or set AWS_PROFILE/S3_PROFILE to an SSO-enabled profile)." >&2
  exit 1
fi

args=()
if [[ -n "${DATASET}" ]]; then
  args+=(--dataset "${DATASET}")
fi

env DISKGRAPH_STATS_DEST="${DISKGRAPH_STATS_DEST:-s3://kumo-unit-test/diskgraph-profiles}" \
  cargo run -p diskgraph-ingest --bin ingest_tui -- "${args[@]}"
