#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "${ROOT}/conformance/csi_contract.sh"

VALID_SC='{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"WaitForFirstConsumer","reclaimPolicy":"Delete"}'
IMMEDIATE_SC='{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"Immediate","reclaimPolicy":"Delete"}'
RETAIN_SC='{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"WaitForFirstConsumer","reclaimPolicy":"Retain"}'
VALID_DRIVER='{"spec":{"storageCapacity":true}}'
DISABLED_DRIVER='{"spec":{"storageCapacity":false}}'
CAPACITIES='{"items":[{"storageClassName":"other","capacity":"900Gi"},{"storageClassName":"ebs-gp3-xfs","capacity":"800Gi"}]}'
ZERO_CAPACITIES='{"items":[{"storageClassName":"ebs-gp3-xfs","capacity":"0"},{"storageClassName":"ebs-gp3-xfs","capacity":null}]}'

fail() { echo "FAIL: $*" >&2; exit 1; }

csi_validate_config "$VALID_SC" "$VALID_DRIVER" ||
  fail "valid capacity-aware configuration was rejected"
csi_validate_config "$VALID_SC" "$DISABLED_DRIVER" advisory ||
  fail "advisory capacity policy rejected a valid late-binding CSI driver"

if csi_validate_config "$IMMEDIATE_SC" "$VALID_DRIVER" >/dev/null 2>&1; then
  fail "Immediate binding was accepted"
fi
if csi_validate_config "$RETAIN_SC" "$VALID_DRIVER" >/dev/null 2>&1; then
  fail "Retain reclaim policy was accepted"
fi
if csi_validate_config "$VALID_SC" "$DISABLED_DRIVER" >/dev/null 2>&1; then
  fail "storageCapacity=false was accepted by the required policy"
fi
if csi_validate_config "$VALID_SC" '' >/dev/null 2>&1; then
  fail "missing CSIDriver was accepted"
fi
if csi_validate_config "$VALID_SC" "$VALID_DRIVER" optional >/dev/null 2>&1; then
  fail "unknown storage capacity policy was accepted"
fi

csi_has_published_capacity ebs-gp3-xfs "$CAPACITIES" ||
  fail "matching nonzero capacity was not detected"
if csi_has_published_capacity ebs-gp3-xfs "$ZERO_CAPACITIES"; then
  fail "zero or unset capacity was accepted"
fi
if csi_has_published_capacity missing "$CAPACITIES"; then
  fail "capacity for a different StorageClass was accepted"
fi

READY='[{"type":"Ready","status":"True"}]'
NOT_READY='[{"type":"Ready","status":"False"}]'
NODE_A='{"metadata":{"name":"node-a"},"status":{"conditions":'"$READY"'}}'
NODE_B='{"metadata":{"name":"node-b"},"status":{"conditions":'"$READY"'}}'
NODE_CORDONED='{"metadata":{"name":"node-cordoned"},"spec":{"unschedulable":true},"status":{"conditions":'"$READY"'}}'
NODE_NOT_READY='{"metadata":{"name":"node-not-ready"},"status":{"conditions":'"$NOT_READY"'}}'
NODE_CACHE_TAINT='{"metadata":{"name":"node-cache-taint"},"spec":{"taints":[{"key":"dedicated","value":"diskgraph-acme","effect":"NoSchedule"}]},"status":{"conditions":'"$READY"'}}'
NODE_STARTUP_TAINT='{"metadata":{"name":"node-startup-taint"},"spec":{"taints":[{"key":"storage.example/init","value":"true","effect":"NoSchedule"}]},"status":{"conditions":'"$READY"'}}'
NODE_PREFERRED_TAINT='{"metadata":{"name":"node-preferred-taint"},"spec":{"taints":[{"key":"spot","value":"true","effect":"PreferNoSchedule"}]},"status":{"conditions":'"$READY"'}}'
CSI_A='{"metadata":{"name":"node-a"},"spec":{"drivers":[{"name":"ebs.csi.aws.com"}]}}'
CSI_B_WRONG='{"metadata":{"name":"node-b"},"spec":{"drivers":[{"name":"other.csi"}]}}'
CSI_CACHE_TAINT='{"metadata":{"name":"node-cache-taint"},"spec":{"drivers":[{"name":"ebs.csi.aws.com"}]}}'
CSI_PREFERRED_TAINT='{"metadata":{"name":"node-preferred-taint"},"spec":{"drivers":[{"name":"ebs.csi.aws.com"}]}}'

coverage() {
  printf '{"nodes":{"items":[%s]},"csinodes":{"items":[%s]}}' "$1" "$2" |
    csi_node_coverage ebs.csi.aws.com dedicated diskgraph-acme "${3:-[]}"
}

[ "$(coverage "$NODE_A" "$CSI_A")" = '{"matched":1,"eligible":1,"uncovered":[]}' ] ||
  fail "registered CSINode driver was not accepted"
[ "$(coverage "$NODE_A,$NODE_B" "$CSI_A,$CSI_B_WRONG")" = \
  '{"matched":2,"eligible":2,"uncovered":["node-b"]}' ] ||
  fail "missing established driver registration was not reported"
[ "$(coverage "$NODE_A,$NODE_CORDONED,$NODE_NOT_READY" "$CSI_A")" = \
  '{"matched":3,"eligible":1,"uncovered":[]}' ] ||
  fail "cordoned or not-Ready nodes were judged"
[ "$(coverage "$NODE_CACHE_TAINT,$NODE_STARTUP_TAINT,$NODE_PREFERRED_TAINT" \
  "$CSI_CACHE_TAINT,$CSI_PREFERRED_TAINT")" = \
  '{"matched":3,"eligible":2,"uncovered":[]}' ] ||
  fail "cache-pod taint compatibility was not applied to eligible nodes"
[ "$(coverage "" "")" = '{"matched":0,"eligible":0,"uncovered":[]}' ] ||
  fail "empty matching node list was not distinguished"
if printf '%s' '{"nodes":{},"csinodes":{"items":[]}}' |
  csi_node_coverage ebs.csi.aws.com dedicated diskgraph-acme >/dev/null 2>&1; then
  fail "malformed node list was accepted"
fi

EXTRA_TOLERATIONS='[{"key":"storage.example/init","operator":"Exists","effect":"NoSchedule"}]'
[ "$(coverage "$NODE_STARTUP_TAINT" '' "$EXTRA_TOLERATIONS")" = \
  '{"matched":1,"eligible":1,"uncovered":["node-startup-taint"]}' ] ||
  fail "an extra-tolerated node was excluded from CSI coverage"
[ "$(coverage "$NODE_STARTUP_TAINT" '' '[{"key":"storage.example/init","value":"other"}]')" = \
  '{"matched":1,"eligible":0,"uncovered":[]}' ] ||
  fail "an unmatched Equal toleration made a node eligible"
[ "$(coverage "$NODE_STARTUP_TAINT" '' '[{"operator":"Exists"}]')" = \
  '{"matched":1,"eligible":1,"uncovered":["node-startup-taint"]}' ] ||
  fail "a standard wildcard toleration did not cover its node"
[ "$(coverage "$NODE_STARTUP_TAINT" '' '[{"key":"storage.example/init","operator":"Exists","effect":"NoExecute"}]')" = \
  '{"matched":1,"eligible":0,"uncovered":[]}' ] ||
  fail "an extra toleration ignored a non-matching effect"

echo "CSI contract tests: passed"
