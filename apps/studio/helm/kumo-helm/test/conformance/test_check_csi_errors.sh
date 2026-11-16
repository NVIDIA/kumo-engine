#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-csi-errors.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

BIN_DIR="${TMP_DIR}/bin"
KUBECTL_LOG="${TMP_DIR}/kubectl.log"
mkdir -p "$BIN_DIR"

cat >"${TMP_DIR}/values.yaml" <<'YAML'
kumoConfig:
  storage:
    dataBucket: s3://no-network
  diskgraph:
    storageCapacityTracking: required
YAML

cat >"${TMP_DIR}/advisory-values.yaml" <<'YAML'
kumoConfig:
  storage:
    dataBucket: s3://no-network
  diskgraph:
    storageCapacityTracking: advisory
YAML

cat >"${BIN_DIR}/kubectl" <<'FAKE_KUBECTL'
#!/usr/bin/env bash
set -u

printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"

if [ "${1:-}" = version ]; then
  printf '%s\n' '{"serverVersion":{"major":"1","minor":"28"}}'
  exit 0
fi

if [ "${1:-}" != get ]; then
  exit 1
fi

case "${2:-}" in
  sc)
    if [ "${FAKE_KUBECTL_CASE}" = storageclass ]; then
      echo "forced StorageClass query failure" >&2
      exit 41
    fi
    if [ "${FAKE_KUBECTL_CASE}" = storageclass_missing ]; then
      exit 0
    fi
    printf '%s\n' \
      '{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"WaitForFirstConsumer","reclaimPolicy":"Delete"}'
    ;;
  csidriver)
    if [ "${FAKE_KUBECTL_CASE}" = csidriver ]; then
      echo "forced CSIDriver query failure" >&2
      exit 42
    fi
    if [ "${FAKE_KUBECTL_CASE}" = advisory_driver ] ||
       [ "${FAKE_KUBECTL_CASE}" = required_driver_without_capacity ]; then
      printf '%s\n' '{"spec":{"storageCapacity":false}}'
    else
      printf '%s\n' '{"spec":{"storageCapacity":true}}'
    fi
    ;;
  csistoragecapacities.storage.k8s.io)
    if [ "${FAKE_KUBECTL_CASE}" = capacity ]; then
      echo "forced CSIStorageCapacity query failure" >&2
      exit 43
    fi
    if [ "${FAKE_KUBECTL_CASE}" = capacity_published ]; then
      printf '%s\n' \
        '{"items":[{"storageClassName":"ebs-gp3-xfs","capacity":"857280Mi"}]}'
    else
      printf '%s\n' '{"items":[]}'
    fi
    ;;
  *)
    exit 1
    ;;
esac
FAKE_KUBECTL
chmod +x "${BIN_DIR}/kubectl"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

strip_ansi() {
  sed $'s/\033\\[[0-9;]*m//g'
}

assert_contains() {
  local output="$1" expected="$2" message="$3"
  [[ "$output" == *"$expected"* ]] ||
    fail "$message (missing: $expected)"
}

assert_not_contains() {
  local output="$1" unexpected="$2" message="$3"
  [[ "$output" != *"$unexpected"* ]] ||
    fail "$message (unexpected: $unexpected)"
}

assert_status() {
  local output="$1" status="$2" id="$3"
  printf '%s\n' "$output" |
    grep -Eq "^[[:space:]]*${status}[[:space:]]+${id}([[:space:]]|$)" ||
    fail "missing ${status} ${id} result"
}

assert_single_csi_result() {
  local output="$1" count
  count=$(printf '%s\n' "$output" |
    awk '$1 ~ /^(PASS|FAIL|SKIP|BLOCK|ADV)$/ && $2 == "CSI" { n++ }
         END { print n + 0 }')
  [ "$count" -eq 1 ] || fail "expected one CSI result, got $count"
}

run_case() {
  local scenario="$1" values_file="${2:-${TMP_DIR}/values.yaml}" output
  : >"$KUBECTL_LOG"
  output=$(FAKE_KUBECTL_CASE="$scenario" \
    FAKE_KUBECTL_LOG="$KUBECTL_LOG" \
    PATH="${BIN_DIR}:${PATH}" \
    bash "${ROOT}/conformance/check.sh" \
      --chart "$ROOT" \
      -f "${ROOT}/test/manifests/values-aws-minimal.yaml" \
      -f "$values_file" \
      -n csi-errors prereq 2>&1) || :
  printf '%s\n' "$output" | strip_ansi
}

storageclass_output=$(run_case storageclass)
assert_status "$storageclass_output" FAIL CSI
assert_contains "$storageclass_output" \
  "could not query configured StorageClass 'ebs-gp3-xfs'" \
  "StorageClass API failure was not diagnosed as a query error"
assert_contains "$storageclass_output" "Kubernetes API access/RBAC" \
  "StorageClass query diagnostic omitted API/RBAC guidance"
assert_not_contains "$storageclass_output" "not found" \
  "StorageClass query failure was misdiagnosed as absence"
assert_single_csi_result "$storageclass_output"
assert_status "$storageclass_output" ADV CSI_NODE_COVERAGE
assert_contains "$storageclass_output" "needs CSI" \
  "node coverage did not identify its failed StorageClass-query dependency"
storageclass_calls=$(<"$KUBECTL_LOG")
assert_contains "$storageclass_calls" \
  "get sc ebs-gp3-xfs --ignore-not-found -o json" \
  "StorageClass read did not use --ignore-not-found"

storageclass_missing_output=$(run_case storageclass_missing)
assert_status "$storageclass_missing_output" FAIL CSI
assert_contains "$storageclass_missing_output" \
  "configured StorageClass 'ebs-gp3-xfs' not found" \
  "missing StorageClass did not produce an actionable failure"
assert_status "$storageclass_missing_output" ADV CSI_NODE_COVERAGE
assert_contains "$storageclass_missing_output" "needs CSI" \
  "node coverage did not identify its failed StorageClass dependency"

csidriver_output=$(run_case csidriver)
assert_status "$csidriver_output" FAIL CSI
assert_contains "$csidriver_output" \
  "could not query CSIDriver 'ebs.csi.aws.com'" \
  "CSIDriver API failure was not diagnosed as a query error"
assert_contains "$csidriver_output" "Kubernetes API access/RBAC" \
  "CSIDriver query diagnostic omitted API/RBAC guidance"
assert_not_contains "$csidriver_output" "CSIDriver ebs.csi.aws.com not found" \
  "CSIDriver query failure was misdiagnosed as absence"
assert_single_csi_result "$csidriver_output"
assert_status "$csidriver_output" ADV CSI_NODE_COVERAGE
assert_contains "$csidriver_output" "needs CSI" \
  "node coverage did not identify its failed CSIDriver-query dependency"
csidriver_calls=$(<"$KUBECTL_LOG")
assert_contains "$csidriver_calls" \
  "get csidriver ebs.csi.aws.com --ignore-not-found -o json" \
  "CSIDriver read did not use --ignore-not-found"

capacity_output=$(run_case capacity)
assert_status "$capacity_output" ADV CSI_CAPACITY
assert_contains "$capacity_output" "capacity unknown/unqueryable" \
  "capacity query failure did not report unknown capacity"
assert_contains "$capacity_output" "RBAC/API availability" \
  "capacity query diagnostic omitted RBAC/API guidance"
assert_not_contains "$capacity_output" "no nonzero CSIStorageCapacity" \
  "capacity query failure claimed confirmed absence"
assert_not_contains "$capacity_output" "scaled to zero" \
  "capacity query failure claimed scale-to-zero"

capacity_published_output=$(run_case capacity_published)
assert_status "$capacity_published_output" PASS CSI_CAPACITY
assert_contains "$capacity_published_output" \
  "nonzero CSIStorageCapacity published for 'ebs-gp3-xfs'" \
  "published capacity was not reported as PASS"
assert_not_contains "$capacity_published_output" "no nonzero CSIStorageCapacity" \
  "published capacity was misreported as absent"

capacity_empty_output=$(run_case capacity_empty)
assert_status "$capacity_empty_output" ADV CSI_CAPACITY
assert_contains "$capacity_empty_output" "no nonzero CSIStorageCapacity" \
  "empty capacity list was not reported as absence"
assert_contains "$capacity_empty_output" "scaled to zero" \
  "empty capacity list advisory omitted scale-to-zero context"

required_driver_output=$(run_case required_driver_without_capacity)
assert_status "$required_driver_output" FAIL CSI
assert_contains "$required_driver_output" \
  "does not satisfy storageCapacityTracking=required" \
  "required policy accepted a driver without capacity tracking"

advisory_driver_output=$(run_case advisory_driver "${TMP_DIR}/advisory-values.yaml")
assert_status "$advisory_driver_output" PASS CSI
assert_contains "$advisory_driver_output" "capacity tracking is advisory" \
  "advisory policy did not accept the late-binding driver"
assert_status "$advisory_driver_output" ADV CSI_CAPACITY
assert_contains "$advisory_driver_output" \
  "does not advertise storageCapacity; allowed by the advisory policy" \
  "advisory driver did not explain why capacity publication is optional"
advisory_driver_calls=$(<"$KUBECTL_LOG")
assert_not_contains "$advisory_driver_calls" \
  "get csistoragecapacities.storage.k8s.io" \
  "advisory driver queried capacity objects it declares it will not publish"

echo "CSI query error-path tests: passed"
