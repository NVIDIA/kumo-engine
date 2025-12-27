#!/usr/bin/env bash
set -euo pipefail

# Generate a massive dataset locally and push it to S3.
# Edit the constants below or override via env vars.

NODES="${NODES:-500000000}"
EDGES_PER_REL="${EDGES_PER_REL:-500000000}"
MAX_DST_ID="${MAX_DST_ID:-100000000000}"
REL0_NAME="${REL0_NAME:-rel_a}"
REL1_NAME="${REL1_NAME:-rel_b}"
REL0_SORT_BITS="${REL0_SORT_BITS:-0}"
REL1_SORT_BITS="${REL1_SORT_BITS:-0}"
NODE_TYPE="${NODE_TYPE:-Node}"
EDGE_BATCH="${EDGE_BATCH:-1000000}"
ROW_GROUP="${ROW_GROUP:-1000000}"
SEED="${SEED:-42}"

# Target prefix (bucket or path) in S3, e.g. s3://bucket/diskgraph-datagenerator
S3_PREFIX="${S3_PREFIX:-s3://kumo-unit-test/diskgraph-datagenerator}"
RUN_ID="${RUN_ID:-run-$(date +%s)-$RANDOM}"
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

echo "Generating massive dataset directly to ${DEST}"
cargo run -p diskgraph-datagenerator --bin massive --release -- \
  --dest s3 \
  --output "${DEST}" \
  --nodes "${NODES}" \
  --edges-per-relation "${EDGES_PER_REL}" \
  --max-dst-id "${MAX_DST_ID}" \
  --edge-batch-size "${EDGE_BATCH}" \
  --row-group-size "${ROW_GROUP}" \
  --seed "${SEED}" \
  --node-type-name "${NODE_TYPE}" \
  --relation0-name "${REL0_NAME}" \
  --relation1-name "${REL1_NAME}" \
  --relation0-sort-bits "${REL0_SORT_BITS}" \
  --relation1-sort-bits "${REL1_SORT_BITS}"

echo "Done. Dataset written to ${DEST}"
