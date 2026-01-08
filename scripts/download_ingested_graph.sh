#!/usr/bin/env bash
set -euo pipefail

# Download ingested graph parquet files for a training or bp job.
#
# Usage:
#   ./scripts/download_ingested_graph.sh <customer_name> <job_id>
#
# Example:
#   ./scripts/download_ingested_graph.sh reddit trainingjob-693c503ad95b47e2ada9b18bd693961a
#
# Output:
#   /mnt/localssd/<customer_name>_<job_id>/<table_name>/*.parquet
#
# Notes:
# - Requires AWS CLI credentials with access to kumo-prod-dataplane-bucket.
# - Errors if the raw ingest data is empty (likely GC'd).

usage() {
  echo "usage: $0 <customer_name> <job_id>" >&2
  echo "       $0 <job_url>" >&2
  echo "example: $0 reddit trainingjob-693c503ad95b47e2ada9b18bd693961a" >&2
  echo "example: $0 https://reddit.kumoai.cloud/predictions/prediction/bp-job-ac18520f009847378ca48903e620b349?page=1&job_type=BATCH_PREDICTION_JOB" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 && $# -ne 2 ]]; then
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  read -r CUSTOMER_NAME JOB_ID < <(
    python3 - <<'PY'
import re
import sys
from urllib.parse import urlparse

url = sys.argv[1]
parsed = urlparse(url)
host = parsed.hostname or ""
customer = host.split(".")[0] if host else ""
match = re.search(r"(trainingjob-[A-Za-z0-9]+|bp-job-[A-Za-z0-9]+)", parsed.path)
job_id = match.group(1) if match else ""
print(customer, job_id)
PY
    "$1"
  )
else
  CUSTOMER_NAME="$1"
  JOB_ID="$2"
fi
OUT_ROOT="${OUT_ROOT:-/mnt/localssd}"

if ! [[ "${CUSTOMER_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: customer_name must be alnum/._- only" >&2
  exit 1
fi
if ! [[ "${JOB_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: job_id must be alnum/._- only" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI not found in PATH" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found in PATH" >&2
  exit 1
fi

if [[ ! -d "${OUT_ROOT}" ]]; then
  echo "error: OUT_ROOT does not exist: ${OUT_ROOT}" >&2
  exit 1
fi
if [[ ! -w "${OUT_ROOT}" ]]; then
  echo "error: OUT_ROOT is not writable: ${OUT_ROOT}" >&2
  exit 1
fi

BUCKET="s3://kumo-prod-dataplane-bucket"
GRAPHS_PREFIX="${BUCKET}/${CUSTOMER_NAME}/prod/jobs/${JOB_ID}/graphs/"
OUT_DIR="${OUT_ROOT}/${CUSTOMER_NAME}_${JOB_ID}"

echo "Listing tables at ${GRAPHS_PREFIX}"
TABLE_PREFIXES=()
while read -r prefix; do
  [[ -z "${prefix}" ]] && continue
  TABLE_PREFIXES+=("${prefix}")
done < <(aws s3 ls "${GRAPHS_PREFIX}" | awk '/PRE / {print $2}')

if [[ ${#TABLE_PREFIXES[@]} -eq 0 ]]; then
  echo "error: no table directories found under ${GRAPHS_PREFIX}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

for table_prefix in "${TABLE_PREFIXES[@]}"; do
  attr_path="${GRAPHS_PREFIX}${table_prefix}attributes.json"
  echo "Reading ${attr_path}"
  attrs_json="$(aws s3 cp "${attr_path}" -)"
  read -r table_name source_name data_source data_version < <(
    python3 - <<'PY'
import json, sys
doc = json.load(sys.stdin)
tm = doc.get("table_metadata_config") or {}
fields = [
    tm.get("table_name"),
    tm.get("source_name"),
    tm.get("data_source"),
    tm.get("data_version"),
]
print(" ".join("" if v is None else str(v) for v in fields))
PY
  <<<"${attrs_json}"
  )

  if [[ -z "${table_name}" || -z "${source_name}" || -z "${data_source}" || -z "${data_version}" ]]; then
    echo "error: missing metadata fields in ${attr_path}" >&2
    exit 1
  fi
  if ! [[ "${table_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: unsupported table_name '${table_name}' from ${attr_path}" >&2
    exit 1
  fi

  raw_prefix="${BUCKET}/${CUSTOMER_NAME}/prod/data-persist-dir/ingest/fpv3.${data_source}.${source_name}/${data_version}/"
  if ! aws s3 ls "${raw_prefix}" | awk 'NF{found=1} END{exit !found}'; then
    echo "error: raw ingest data not found (likely GC'd): ${raw_prefix}" >&2
    exit 1
  fi

  table_out="${OUT_DIR}/${table_name}"
  mkdir -p "${table_out}"
  echo "Downloading ${raw_prefix} -> ${table_out}"
  aws s3 cp "${raw_prefix}" "${table_out}/" --recursive --exclude "*" --include "*.parquet"
done

echo "Done. Data at ${OUT_DIR}"
