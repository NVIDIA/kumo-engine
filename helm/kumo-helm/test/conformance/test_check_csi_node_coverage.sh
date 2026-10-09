#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-csi-node-coverage.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

BIN_DIR="${TMP_DIR}/bin"
KUBECTL_LOG="${TMP_DIR}/kubectl.log"
mkdir -p "$BIN_DIR"

cat >"${TMP_DIR}/values.yaml" <<'YAML'
kumoConfig:
  storage:
    dataBucket: s3://no-network
  diskgraph:
    nodeTypes:
      test-gpu: {ssd_gib: 1000, gpu_count: 3}
      test-gpu-duplicate: {ssd_gib: 900, gpu_count: 2}
  extra:
    data_compute:
      k8s_pool:
        pool_diskgraph_node_types:
          test-gpu: {ssd_gib: "+900", gpu_count: "+2"}
          extra-gpu: {ssd_gib: 1600, gpu_count: 4}
YAML

cat >"${TMP_DIR}/normalized-tenant.yaml" <<'YAML'
global:
  tenant: Acme_Prod
YAML

cat >"${BIN_DIR}/kubectl" <<'FAKE_KUBECTL'
#!/usr/bin/env bash
set -u

printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"

if [ "${1:-}" = version ]; then
  printf '%s\n' '{"serverVersion":{"major":"1","minor":"28"}}'
  exit 0
fi
[ "${1:-}" = get ] || exit 1

READY='{"type":"Ready","status":"True"}'
NOT_READY='{"type":"Ready","status":"False"}'
node() { printf '{"metadata":{"name":"%s"},"status":{"conditions":[%s]}}' "$1" "$2"; }
tainted_node() { printf '{"metadata":{"name":"%s"},"spec":{"taints":[{"key":"%s","value":"%s","effect":"%s"}]},"status":{"conditions":[%s]}}' "$1" "$2" "$3" "$4" "$5"; }
cordoned() { printf '{"metadata":{"name":"%s"},"spec":{"unschedulable":true},"status":{"conditions":[%s]}}' "$1" "$READY"; }
csinode() { printf '{"metadata":{"name":"%s"},"spec":{"drivers":[{"name":"%s"}]}}' "$1" "$2"; }

case "${2:-}" in
  sc)
    printf '%s\n' '{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"WaitForFirstConsumer","reclaimPolicy":"Delete"}'
    ;;
  csidriver)
    printf '%s\n' '{"spec":{"storageCapacity":true}}'
    ;;
  csistoragecapacities.storage.k8s.io)
    # Capacity publication is deliberately absent. Valid CSINode registration
    # must still make CSI_NODE_COVERAGE pass.
    printf '%s\n' '{"items":[]}'
    ;;
  csinodes.storage.k8s.io)
    if [ "${FAKE_KUBECTL_CASE}" = csinodes_unqueryable ]; then
      echo "forced CSINode list failure" >&2
      exit 44
    fi
    case "${FAKE_KUBECTL_CASE}" in
      uncovered)
        printf '{"items":[%s,%s]}\n' \
          "$(csinode cache-900 ebs.csi.aws.com)" "$(csinode cache-1600 other.csi)"
        ;;
      startup_tainted)
        printf '{"items":[%s]}\n' "$(csinode cache-1600 ebs.csi.aws.com)"
        ;;
      *)
        printf '{"items":[%s,%s]}\n' \
          "$(csinode cache-900 ebs.csi.aws.com)" "$(csinode cache-1600 ebs.csi.aws.com)"
        ;;
    esac
    ;;
  nodepools.karpenter.sh)
    printf '%s\n' '{"items":[]}'
    ;;
  nodes)
    case "$*" in
      *"-o jsonpath="*) printf '%s\n' '1'; exit 0 ;;
      *"--no-headers"*) printf '%s\n' 'node Ready 1d'; exit 0 ;;
    esac
    case "$*" in
      *"kumo.ai/diskgraph-pool-type=cache,kumo.ai/diskgraph-ssd-size=900,kumo.ai/diskgraph-gpu-count=2 -o json"*)
        if [ "${FAKE_KUBECTL_CASE}" = ineligible ]; then
          printf '{"items":[%s]}\n' "$(cordoned cache-900-cordoned)"
        elif [ "${FAKE_KUBECTL_CASE}" = scale_to_zero ]; then
          printf '%s\n' '{"items":[]}'
        elif [ "${FAKE_KUBECTL_CASE}" = startup_tainted ]; then
          printf '{"items":[%s]}\n' \
            "$(tainted_node cache-900-initializing storage.example/init true NoSchedule "$READY")"
        elif [ "${FAKE_KUBECTL_CASE}" = normalized_tenant ]; then
          printf '{"items":[%s]}\n' \
            "$(tainted_node cache-900 dedicated diskgraph-acme-prod NoSchedule "$READY")"
        else
          printf '{"items":[%s]}\n' "$(node cache-900 "$READY")"
        fi
        ;;
      *"kumo.ai/diskgraph-pool-type=cache,kumo.ai/diskgraph-ssd-size=1600,kumo.ai/diskgraph-gpu-count=4 -o json"*)
        if [ "${FAKE_KUBECTL_CASE}" = nodes_unqueryable ]; then
          echo "forced tier node list failure" >&2
          exit 45
        elif [ "${FAKE_KUBECTL_CASE}" = ineligible ] ||
             [ "${FAKE_KUBECTL_CASE}" = mixed_ineligible ]; then
          printf '{"items":[%s]}\n' "$(node cache-1600-not-ready "$NOT_READY")"
        elif [ "${FAKE_KUBECTL_CASE}" = scale_to_zero ]; then
          printf '%s\n' '{"items":[]}'
        elif [ "${FAKE_KUBECTL_CASE}" = startup_tainted ]; then
          printf '%s\n' '{"items":[]}'
        else
          printf '{"items":[%s]}\n' "$(node cache-1600 "$READY")"
        fi
        ;;
      *) printf '%s\n' '{"items":[]}' ;;
    esac
    ;;
  *) exit 1 ;;
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
  [[ "$output" == *"$expected"* ]] || fail "$message (missing: $expected)"
}

assert_not_contains() {
  local output="$1" unexpected="$2" message="$3"
  [[ "$output" != *"$unexpected"* ]] || fail "$message (unexpected: $unexpected)"
}

assert_count() {
  local output="$1" expected="$2" count="$3" message="$4" actual
  actual=$(printf '%s\n' "$output" | grep -Fc "$expected" || true)
  [ "$actual" -eq "$count" ] ||
    fail "$message (expected $count, got $actual: $expected)"
}

assert_status() {
  local output="$1" status="$2" id="$3"
  printf '%s\n' "$output" |
    grep -Eq "^[[:space:]]*${status}[[:space:]]+${id}([[:space:]]|$)" ||
    fail "missing ${status} ${id} result"
}

run_case() {
  local scenario="$1" output
  local -a tenant_args=()
  case "$scenario" in
    normalized_tenant) tenant_args=(-f "${TMP_DIR}/normalized-tenant.yaml") ;;
  esac
  : >"$KUBECTL_LOG"
  output=$(FAKE_KUBECTL_CASE="$scenario" \
    FAKE_KUBECTL_LOG="$KUBECTL_LOG" \
    PATH="${BIN_DIR}:${PATH}" \
    bash "${ROOT}/conformance/check.sh" \
      --chart "$ROOT" \
      -f "${ROOT}/test/manifests/values-aws-minimal.yaml" \
      -f "${TMP_DIR}/values.yaml" \
      "${tenant_args[@]}" \
      -n csi-coverage prereq 2>&1) || :
  printf '%s\n' "$output" | strip_ansi
}

# Probe tiers must come from the same fully merged JSON delivered to the app.
rendered=$(helm template kumo "$ROOT" \
  -f "${ROOT}/test/manifests/values-aws-minimal.yaml" \
  -f "${TMP_DIR}/values.yaml" \
  --show-only templates/tests/probe-inputs.yaml)
inputs=$(printf '%s\n' "$rendered" | sed -n -E \
  's/^[[:space:]]*inputs\.json:[[:space:]]*//p' | jq -c fromjson)
printf '%s' "$inputs" | jq -e '
  [.diskgraphTiers[] | [(.ssd | tostring), (.gpu | tostring)]] | sort ==
  [["+900","+2"],["1600","4"],["900","2"]]
' >/dev/null || fail "probe ignored effective tiers added/replaced through kumoConfig.extra"

covered_output=$(run_case covered)
covered_calls=$(<"$KUBECTL_LOG")
assert_status "$covered_output" PASS CSI_NODE_COVERAGE
assert_status "$covered_output" ADV CSI_CAPACITY
assert_contains "$covered_output" "registered on all 2 eligible cache node(s)" \
  "covered cache nodes were not reported"
assert_contains "$covered_calls" \
  "get nodes -l kumo.ai/diskgraph-pool-type=cache,kumo.ai/diskgraph-ssd-size=900,kumo.ai/diskgraph-gpu-count=2 -o json" \
  "effective 900Gi/2gpu selector was not queried exactly"
assert_count "$covered_calls" \
  "get nodes -l kumo.ai/diskgraph-pool-type=cache,kumo.ai/diskgraph-ssd-size=900,kumo.ai/diskgraph-gpu-count=2 -o json" \
  1 "integer-coerced duplicate tier was queried more than once"
assert_contains "$covered_calls" \
  "get nodes -l kumo.ai/diskgraph-pool-type=cache,kumo.ai/diskgraph-ssd-size=1600,kumo.ai/diskgraph-gpu-count=4 -o json" \
  "effective 1600Gi/4gpu selector was not queried exactly"
assert_not_contains "$covered_calls" \
  "kumo.ai/diskgraph-pool-type=cache,kumo.ai/diskgraph-ssd-size=800,kumo.ai/diskgraph-gpu-count=1 -o json" \
  "pre-override tier was queried"

uncovered_output=$(run_case uncovered)
assert_status "$uncovered_output" FAIL CSI_NODE_COVERAGE
assert_contains "$uncovered_output" "cache-1600 (1600Gi/4gpu)" \
  "uncovered live node was not named with its effective tier"
assert_contains "$uncovered_output" "CSI node DaemonSet's scheduling rules" \
  "failure omitted actionable CSI DaemonSet guidance"
assert_contains "$uncovered_output" "each node's CSINode" \
  "failure omitted registration verification guidance"

ineligible_output=$(run_case ineligible)
assert_status "$ineligible_output" ADV CSI_NODE_COVERAGE
assert_contains "$ineligible_output" \
  "matching cache nodes exist for tier(s) [1600Gi/4gpu, 900Gi/2gpu]" \
  "existing ineligible nodes were not distinguished from an empty pool"
assert_contains "$ineligible_output" \
  "none are currently eligible (Ready, non-cordoned, and compatible with the cache pod's hard-taint tolerations)" \
  "ineligible-node diagnostic omitted node eligibility state"
assert_not_contains "$ineligible_output" "scale-to-zero" \
  "existing ineligible nodes were misclassified as scale-to-zero"
assert_not_contains "$ineligible_output" "cache-900-cordoned" \
  "cordoned node was judged"
assert_not_contains "$ineligible_output" "cache-1600-not-ready" \
  "not-Ready node was judged"

mixed_ineligible_output=$(run_case mixed_ineligible)
assert_status "$mixed_ineligible_output" ADV CSI_NODE_COVERAGE
assert_contains "$mixed_ineligible_output" \
  "tier(s) [1600Gi/4gpu] have matching nodes but none currently eligible (Ready, non-cordoned, and compatible with the cache pod's hard-taint tolerations)" \
  "mixed-tier diagnostic omitted the ineligible tier"
assert_not_contains "$mixed_ineligible_output" "scale-to-zero" \
  "mixed-tier ineligible nodes were misclassified as scale-to-zero"

scale_to_zero_output=$(run_case scale_to_zero)
assert_status "$scale_to_zero_output" SKIP CSI_NODE_COVERAGE
assert_contains "$scale_to_zero_output" "no matching cache nodes" \
  "empty matching node list was not diagnosed"
assert_contains "$scale_to_zero_output" "scale-to-zero" \
  "empty matching node list lost scale-to-zero semantics"

startup_tainted_output=$(run_case startup_tainted)
assert_status "$startup_tainted_output" ADV CSI_NODE_COVERAGE
assert_contains "$startup_tainted_output" "hard-taint tolerations" \
  "startup-tainted node result did not explain its temporary ineligibility"
assert_not_contains "$startup_tainted_output" "FAIL  CSI_NODE_COVERAGE" \
  "startup-tainted node produced a false failure before a cache pod could schedule"
assert_not_contains "$startup_tainted_output" "cache-900-initializing" \
  "startup-tainted node was judged for CSI registration"

normalized_tenant_output=$(run_case normalized_tenant)
assert_status "$normalized_tenant_output" PASS CSI_NODE_COVERAGE
assert_contains "$normalized_tenant_output" "registered on all 2 eligible cache node(s)" \
  "pool-controller tenant normalization was not applied to the cache toleration"

nodes_error_output=$(run_case nodes_unqueryable)
assert_status "$nodes_error_output" ADV CSI_NODE_COVERAGE
assert_contains "$nodes_error_output" \
  "could not list nodes for {kumo.ai/diskgraph-pool-type=cache,kumo.ai/diskgraph-ssd-size=1600,kumo.ai/diskgraph-gpu-count=4}" \
  "tier node-list failure was not diagnosed accurately"
assert_not_contains "$nodes_error_output" "PASS  CSI_NODE_COVERAGE" \
  "node-list failure produced a false PASS"

csinodes_error_output=$(run_case csinodes_unqueryable)
assert_status "$csinodes_error_output" ADV CSI_NODE_COVERAGE
assert_contains "$csinodes_error_output" "could not list CSINodes for driver 'ebs.csi.aws.com'" \
  "CSINode query failure was not diagnosed accurately"
assert_not_contains "$csinodes_error_output" "PASS  CSI_NODE_COVERAGE" \
  "CSINode query failure produced a false PASS"

echo "CSI node coverage tests: passed"
