#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Provision a dev tenant for the diskgraph pool+k8s devserver via kubectl.
#
# Bypasses the normal ephemeral-environment flow entirely — for OPTION B
# (local controller + workers in cluster) we only need 4 cluster
# resources, all of which SSO PowerUserAccess can create directly:
#
#   - Namespace                       kumo-dataplane-<TENANT>
#   - ServiceAccount + IRSA           kumo-temporal in that namespace
#   - NodePool (cache, Karpenter)     diskgraph-<TENANT>
#   - NodePool (eph, Karpenter)       diskgraph-eph-<TENANT>
#
# Cluster-shared dependencies (we reference, never create):
#   - EC2NodeClass diskgraph-cache, diskgraph-eph
#   - IAM role     testing-uw4-test-kumo-dataplane-...
#
# Idempotent (kubectl apply is upsert). TENANT is fully user-chosen,
# subject only to the controller's actual constraint: RFC 1123 (lowercase
# alphanumeric + '-', no underscores) and ≤ 48 chars.

set -euo pipefail

EKS_CLUSTER='testing-uw4'
EKS_REGION='us-west-2'
# Terraform names the dataplane IRSA role with a generated 20-digit
# suffix that's stable per role lifetime but rotates if infra
# recreates it. Discover by prefix instead of pinning the suffix,
# so we don't silently bind the SA to a stale ARN after a re-create.
IRSA_ROLE_NAME_PREFIX="${EKS_CLUSTER}-test-kumo-dataplane-"

resolve_irsa_role_arn() {
    local arns
    arns=$(aws iam list-roles --region "$EKS_REGION" --output text \
        --query "Roles[?starts_with(RoleName, '${IRSA_ROLE_NAME_PREFIX}')].Arn")
    local count
    count=$(printf '%s\n' "$arns" | grep -c . || true)
    if [[ "$count" -ne 1 ]]; then
        echo "Error: expected exactly 1 IAM role with prefix" \
             "'${IRSA_ROLE_NAME_PREFIX}', found $count:" >&2
        printf '  %s\n' $arns >&2
        exit 1
    fi
    printf '%s' "$arns"
}

usage() {
    cat <<EOF >&2
Usage: $0 <TENANT>
       TENANT=<id> $0

TENANT format: lowercase alphanumeric + '-', max 48 chars,
starts with alphanumeric (RFC 1123). Example: siyang-dev

Reads TENANT from \$1 if provided, else from env var TENANT.
EOF
    exit 1
}

TENANT="${1:-${TENANT:-}}"
[[ -n "$TENANT" ]] || usage

if ! [[ "$TENANT" =~ ^[a-z0-9][a-z0-9-]{0,47}$ ]]; then
    echo "Error: TENANT '$TENANT' is invalid." >&2
    echo "  Must match ^[a-z0-9][a-z0-9-]{0,47}\$ (lowercase alphanumeric + '-', no underscores)." >&2
    exit 2
fi

NAMESPACE="kumo-dataplane-${TENANT}"
CACHE_POOL="diskgraph-${TENANT}"
EPH_POOL="diskgraph-eph-${TENANT}"
IRSA_ROLE_ARN=$(resolve_irsa_role_arn)

printf '[provision] Cluster:    %s\n' "$EKS_CLUSTER"
printf '[provision] TENANT: %s\n' "$TENANT"
printf '[provision] Namespace:  %s\n' "$NAMESPACE"
printf '[provision] NodePools:  %s + %s\n' "$CACHE_POOL" "$EPH_POOL"
printf '[provision] IRSA role:  %s\n' "$IRSA_ROLE_ARN"
printf '\n'

kubectl --context "$EKS_CLUSTER" apply -f - <<EOF
---
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
  labels:
    kumo.ai/managed-by: devserver-pool-k8s
    kumo.ai/tenant: ${TENANT}
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kumo-temporal
  namespace: ${NAMESPACE}
  annotations:
    eks.amazonaws.com/role-arn: ${IRSA_ROLE_ARN}
  labels:
    kumo.ai/managed-by: devserver-pool-k8s
---
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: ${CACHE_POOL}
  labels:
    kumo.ai/managed-by: devserver-pool-k8s
    kumo.ai/tenant: ${TENANT}
spec:
  disruption:
    budgets:
    - nodes: "100%"
      reasons:
      - Empty
    consolidateAfter: 1s
    consolidationPolicy: WhenEmpty
  template:
    metadata:
      labels:
        kumo.ai/instance-type: diskgraph-cache
        kumo.ai/tenant: ${TENANT}
        pool-type: cache
    spec:
      expireAfter: 720h
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: diskgraph-cache
      requirements:
      - key: node.kubernetes.io/instance-type
        operator: In
        values: [g6e.16xlarge]
      - key: karpenter.sh/capacity-type
        operator: In
        values: [on-demand]
      - key: kubernetes.io/arch
        operator: In
        values: [amd64]
      - key: kubernetes.io/os
        operator: In
        values: [linux]
      taints:
      - effect: NoSchedule
        key: dedicated
        value: ${CACHE_POOL}
  weight: 30
---
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: ${EPH_POOL}
  labels:
    kumo.ai/managed-by: devserver-pool-k8s
    kumo.ai/tenant: ${TENANT}
spec:
  disruption:
    budgets:
    - nodes: "100%"
      reasons:
      - Empty
    consolidateAfter: 1s
    consolidationPolicy: WhenEmpty
  template:
    metadata:
      labels:
        kumo.ai/instance-type: diskgraph-eph
        kumo.ai/tenant: ${TENANT}
        pool-type: ephemeral
    spec:
      expireAfter: 720h
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: diskgraph-eph
      requirements:
      - key: node.kubernetes.io/instance-type
        operator: In
        values: [g6e.16xlarge]
      - key: karpenter.sh/capacity-type
        operator: In
        values: [on-demand]
      - key: kubernetes.io/arch
        operator: In
        values: [amd64]
      - key: kubernetes.io/os
        operator: In
        values: [linux]
      taints:
      - effect: NoSchedule
        key: dedicated
        value: ${EPH_POOL}
  weight: 30
EOF

printf '\n[provision] Done.\n'
