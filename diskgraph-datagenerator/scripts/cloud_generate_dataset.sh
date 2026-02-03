#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config.json>"
  exit 1
fi

CONFIG_PATH="$1"
CONFIG_NAME="$(basename "$CONFIG_PATH")"
DATASET_NAME="${CONFIG_NAME%.json}"

BUCKET_NAME="${DISKGRAPH_CLOUD_BUCKET:-kumo-unit-test}"
PREFIX_BASE="${DISKGRAPH_CLOUD_PREFIX:-diskgraph-datagenerator}"

detect_target() {
  if is_gcp_vm; then
    echo "gs://${BUCKET_NAME}/${PREFIX_BASE}"
    return 0
  fi
  if is_aws_vm; then
    echo "s3://${BUCKET_NAME}/${PREFIX_BASE}"
    return 0
  fi
  if is_azure_vm; then
    local account="${AZURE_STORAGE_ACCOUNT:-${AZURE_ACCOUNT_NAME:-}}"
    local container="${DISKGRAPH_CLOUD_CONTAINER:-${BUCKET_NAME}}"
    if [[ -z "${account}" ]]; then
      echo "AZURE_STORAGE_ACCOUNT or AZURE_ACCOUNT_NAME must be set for az:// output." >&2
      exit 1
    fi
    echo "az://${account}/${container}/${PREFIX_BASE}"
    return 0
  fi
  if [[ -n "${AWS_ACCESS_KEY_ID:-}" || -n "${AWS_PROFILE:-}" || -n "${S3_REGION:-}" || -n "${S3_ENDPOINT:-}" ]]; then
    echo "s3://${BUCKET_NAME}/${PREFIX_BASE}"
    return 0
  fi
  if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" || -n "${GOOGLE_CLOUD_PROJECT:-}" || -n "${GCP_PROJECT:-}" ]]; then
    echo "gs://${BUCKET_NAME}/${PREFIX_BASE}"
    return 0
  fi
  if [[ -n "${AZURE_STORAGE_ACCOUNT:-}" || -n "${AZURE_ACCOUNT_NAME:-}" || -n "${AZURE_ACCESS_KEY:-}" || -n "${AZURE_SAS_KEY:-}" || -n "${AZURE_TENANT_ID:-}" ]]; then
    local account="${AZURE_STORAGE_ACCOUNT:-${AZURE_ACCOUNT_NAME:-}}"
    local container="${DISKGRAPH_CLOUD_CONTAINER:-${BUCKET_NAME}}"
    if [[ -z "${account}" ]]; then
      echo "AZURE_STORAGE_ACCOUNT or AZURE_ACCOUNT_NAME must be set for az:// output." >&2
      exit 1
    fi
    echo "az://${account}/${container}/${PREFIX_BASE}"
    return 0
  fi
  echo "Unable to detect cloud environment. Set AWS_*, GOOGLE_*, or AZURE_* env vars." >&2
  exit 1
}

curl_meta() {
  curl -sSf --connect-timeout 1 --max-time 2 --noproxy "*" "$@" || true
}

is_aws_vm() {
  local token
  token="$(curl_meta -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" http://169.254.169.254/latest/api/token)"
  if [[ -n "${token}" ]]; then
    curl_meta -H "X-aws-ec2-metadata-token: ${token}" http://169.254.169.254/latest/meta-data/ >/dev/null
    return $?
  fi
  curl_meta http://169.254.169.254/latest/meta-data/ >/dev/null
}

is_gcp_vm() {
  curl_meta -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/ >/dev/null
}

is_azure_vm() {
  curl_meta -H "Metadata: true" "http://169.254.169.254/metadata/instance?api-version=2021-02-01" >/dev/null
}

OUT_ROOT="$(detect_target)/${DATASET_NAME}"

echo "Output target: ${OUT_ROOT}"

if cargo run -p diskgraph-datagenerator --bin diskgraph-datagenerator --release -- \
  --config "$CONFIG_PATH" \
  --output "$OUT_ROOT"; then
  echo "Output: ${OUT_ROOT}"
  echo "Next steps:"
  echo "  ./diskgraph-infer/scripts/infer_config.sh \"${OUT_ROOT}\""
fi
