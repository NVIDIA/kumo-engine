#!/usr/bin/env bash
set -euo pipefail

# Generate a multi-table dataset and stream it directly to S3.
# Edit the constants below or override via env vars.

NODES="${1:-${NODES:-500m}}"
EDGES_PER_REL="${2:-${EDGES_PER_REL:-500m}}"
NODE_DATA_BYTES="${3:-${NODE_DATA_BYTES:-128b}}"
NUM_SHARDS="${NUM_SHARDS:-5}"
EDGE_BATCH="${EDGE_BATCH:-1m}"
ROW_GROUP="${ROW_GROUP:-1m}"
SEED="${SEED:-42}"
NODE_DATA="${NODE_DATA:-true}"
NODE_DATA_BATCH="${NODE_DATA_BATCH:-1m}"
NODE_DATA_FLAG=""
if [[ "${NODE_DATA}" == "false" ]]; then
  NODE_DATA_FLAG="--no-node-data"
fi

# Target prefix (bucket or path) in S3, e.g. s3://bucket/diskgraph-datagenerator
S3_PREFIX="${S3_PREFIX:-s3://kumo-unit-test/diskgraph-datagenerator}"
RUN_ID="${RUN_ID:-nodes-${NODES}-edges-${EDGES_PER_REL}-bytes-${NODE_DATA_BYTES}}"
DEST="${S3_PREFIX%/}/${RUN_ID}"

export AWS_EC2_METADATA_DISABLED=true
export AWS_SDK_LOAD_CONFIG=1

# Profile detection similar to diskgraph-index cloud tests.
AWS_PROFILE_TO_USE="${S3_PROFILE:-${AWS_PROFILE:-}}"
if [[ -n "${AWS_PROFILE_TO_USE}" ]]; then
  export AWS_PROFILE="${AWS_PROFILE_TO_USE}"
else
  AWS_PROFILES_RAW="$(aws configure list-profiles 2>/dev/null | grep -v '^default$' || true)"
  # shellcheck disable=SC2086 # intentional splitting
  set -- ${AWS_PROFILES_RAW}
  if [[ "$#" -eq 1 ]]; then
    export AWS_PROFILE="$1"
    echo "No AWS_PROFILE provided; auto-selecting profile '${AWS_PROFILE}'."
  fi
fi

echo "Using S3 destination: ${DEST}"
echo "Checking AWS credentials (sts:GetCallerIdentity)..."
aws sts get-caller-identity >/dev/null

echo "Generating multi-table dataset directly to ${DEST}"
cargo run -p diskgraph-datagenerator --bin generate-multi --release -- \
  --dest s3 \
  --output "${DEST}" \
  --nodes "${NODES}" \
  --edges-per-relation "${EDGES_PER_REL}" \
  --num-shards "${NUM_SHARDS}" \
  --edge-batch-size "${EDGE_BATCH}" \
  --row-group-size "${ROW_GROUP}" \
  --seed "${SEED}" \
  ${NODE_DATA_FLAG} \
  --node-data-bytes "${NODE_DATA_BYTES}" \
  --node-data-batch-size "${NODE_DATA_BATCH}"

echo "Done. Dataset written to ${DEST}"
