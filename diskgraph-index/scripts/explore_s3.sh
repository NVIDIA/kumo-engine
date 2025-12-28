#!/usr/bin/env bash
set -euo pipefail

# Explore a shard stored on S3 (expects metadata.json, offsets.bin, edges.bin under the prefix).
# Pass s3://bucket/prefix as the first arg.

PROFILE_ENABLED="false"
while getopts "p" opt; do
  case "${opt}" in
    p)
      PROFILE_ENABLED="true"
      ;;
    *)
      echo "usage: $0 [-p] s3://bucket/prefix" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

S3_URL="${1:-}"
if [[ -z "${S3_URL}" ]]; then
  echo "usage: $0 [-p] s3://bucket/prefix" >&2
  exit 1
fi

export AWS_EC2_METADATA_DISABLED=true
export AWS_SDK_LOAD_CONFIG=1
AWS_PROFILE_TO_USE="${S3_PROFILE:-${AWS_PROFILE:-}}"
if [[ -n "${AWS_PROFILE_TO_USE}" ]]; then
  export AWS_PROFILE="${AWS_PROFILE_TO_USE}"
else
  AWS_PROFILES_RAW="$(aws configure list-profiles 2>/dev/null | grep -v '^default$' || true)"
  # shellcheck disable=SC2086
  set -- ${AWS_PROFILES_RAW}
  if [[ "$#" -eq 1 ]]; then
    export AWS_PROFILE="$1"
    echo "No AWS_PROFILE provided; auto-selecting profile '${AWS_PROFILE}'."
  fi
fi

echo "Exploring S3 shard at ${S3_URL}"
aws sts get-caller-identity >/dev/null
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
  AWS_ALLOW_HTTP=0 AWS_EC2_METADATA_DISABLED=1 cargo run -p diskgraph-index --bin explore --release -- --profile "explore-profile" "${S3_URL}"
else
  AWS_ALLOW_HTTP=0 AWS_EC2_METADATA_DISABLED=1 cargo run -p diskgraph-index --bin explore --release -- "${S3_URL}"
fi
