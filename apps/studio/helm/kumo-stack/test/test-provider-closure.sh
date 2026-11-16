#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STACK_ROOT="${ROOT}/kumo-stack"
TRACE_FILE=$(mktemp "${TMPDIR:-/tmp}/kumo-provider-closure.XXXXXX")
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-provider-work.XXXXXX")
trap 'rm -f "${TRACE_FILE}"; rm -rf "${WORK_DIR}"' EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

die() {
  fail "$*"
}

helm_upgrade() {
  printf 'helm %s\n' "$*" >>"${TRACE_FILE}"
}

ensure_helm_repo() {
  printf 'repo %s\n' "$*" >>"${TRACE_FILE}"
}

ensure_namespace() {
  printf 'namespace %s\n' "$*" >>"${TRACE_FILE}"
}

kubectl_cmd() {
  printf 'kubectl %s\n' "$*" >>"${TRACE_FILE}"
  if [[ "$*" == *" create secret tls "* ]]; then
    printf '%s\n' 'apiVersion: v1' 'kind: Secret'
  fi
}

mock_helm() {
  printf 'helm-bin %s\n' "$*" >>"${TRACE_FILE}"
  if [[ "$*" == "show crds"* ]]; then
    printf '%s\n' 'apiVersion: apiextensions.k8s.io/v1' 'kind: CustomResourceDefinition'
  fi
}

HELM=mock_helm
JQ=jq
CONTEXT=test
TIMEOUT=20m
PROVIDER_CLUSTER=test-eks
PROVIDER_REGION=us-west-2
EKS_VPC_ID=vpc-0123456789abcdef0
EKS_KARPENTER_CONTROLLER_ROLE_ARN=arn:aws:iam::000000000000:role/karpenter
EKS_LOAD_BALANCER_CONTROLLER_ROLE_ARN=arn:aws:iam::000000000000:role/lbc
EKS_EDGE_CERTIFICATE_ARN=arn:aws:acm:us-west-2:000000000000:certificate/test
EKS_EDGE_SCHEME=internal
KARPENTER_IMAGE_REPOSITORY=private.example/karpenter/controller
KARPENTER_IMAGE_TAG=1.12.0
KARPENTER_IMAGE_DIGEST=sha256:8128bc8cdd0cc8ca2d2cf68ec61033eceb8cfb24834a08ab278784924013cfd0
AWS_LOAD_BALANCER_CONTROLLER_IMAGE_REPOSITORY=private.example/eks/aws-load-balancer-controller
AWS_LOAD_BALANCER_CONTROLLER_IMAGE_TAG=v2.14.0@sha256:58d83491b3eb25080e478bdd9a245b773fd9ceb29d9182fa4b198dee5ff31d22

# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/providers/common.sh"
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/providers/eks.sh"

install_karpenter
install_aws_load_balancer_controller

for expected in \
  "oci://public.ecr.aws/karpenter/karpenter-crd --namespace kube-system --version 1.12.0" \
  "oci://public.ecr.aws/karpenter/karpenter --namespace kube-system --version 1.12.0" \
  "eks/aws-load-balancer-controller --namespace kube-system --version 1.14.0" \
  "controller.image.repository=private.example/karpenter/controller" \
  "controller.image.digest=sha256:8128bc8cdd0cc8ca2d2cf68ec61033eceb8cfb24834a08ab278784924013cfd0" \
  "image.repository=private.example/eks/aws-load-balancer-controller" \
  "image.tag=v2.14.0@sha256:58d83491b3eb25080e478bdd9a245b773fd9ceb29d9182fa4b198dee5ff31d22"; do
  grep -Fq "${expected}" "${TRACE_FILE}" ||
    fail "missing provider install: ${expected}"
done

: >"${TRACE_FILE}"
PROVIDER_RESOURCE_GROUP=test-rg
AKS_APPLICATION_GATEWAY_ID=/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-rg/providers/Microsoft.Network/applicationGateways/test-private
TENANT_NAMESPACE=kumo-acme
EDGE_TLS_SECRET=kumo-edge-tls
EDGE_TLS_CERTIFICATE_FILE="${WORK_DIR}/tls.crt"
EDGE_TLS_PRIVATE_KEY_FILE="${WORK_DIR}/tls.key"
printf certificate >"${EDGE_TLS_CERTIFICATE_FILE}"
printf key >"${EDGE_TLS_PRIVATE_KEY_FILE}"

az() {
  printf 'az %s\n' "$*" >>"${TRACE_FILE}"
  if [[ "$*" == "aks show"* ]]; then
    if [[ "${AKS_ADDON_ENABLED}" == true ]]; then
      printf '%s\n' '{"addonProfiles":{"ingressApplicationGateway":{"enabled":true,"config":{"effectiveApplicationGatewayId":"/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-rg/providers/Microsoft.Network/applicationGateways/test-private"},"identity":{"objectId":"11111111-1111-1111-1111-111111111111"}}}}'
    else
      printf '{"addonProfiles":{"ingressApplicationGateway":{"enabled":false}}}\n'
    fi
  elif [[ "$*" == "aks enable-addons"* ]]; then
    AKS_ADDON_ENABLED=true
  elif [[ "$*" == "aks check-acr"* ]]; then
    printf 'Validating image pull permission: SUCCEEDED\n'
  elif [[ "$*" == "network application-gateway show"* ]]; then
    printf '%s\n' '{"provisioningState":"Succeeded","operationalState":"Running","sku":{"tier":"Standard_v2"},"frontendIPConfigurations":[{"privateIPAddress":"10.0.0.10"}],"gatewayIPConfigurations":[{"subnet":{"id":"/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-rg/providers/Microsoft.Network/virtualNetworks/test-vnet/subnets/appgw"}}]}'
  fi
}

kubectl_cmd() {
  printf 'kubectl %s\n' "$*" >>"${TRACE_FILE}"
  if [[ "$*" == *" create secret tls "* ]]; then
    printf '%s\n' 'apiVersion: v1' 'kind: Secret'
  fi
}

# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/providers/aks.sh"
AKS_ADDON_ENABLED=false
KUMO_PULL_SECRET=
KUMO_CPU_IMAGE=test.azurecr.io/kumo/kumo-backend-cpu:test
verify_aks_application_gateway
verify_aks_acr_access
ensure_aks_application_gateway_addon
ensure_aks_edge_tls_secret

grep -Fq \
  'az aks enable-addons --resource-group test-rg --name test-eks --addons ingress-appgw --appgw-id /subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-rg/providers/Microsoft.Network/applicationGateways/test-private' \
  "${TRACE_FILE}" ||
  fail "AKS installer did not enable managed AGIC against the selected gateway"
grep -Fq \
  'kubectl -n kumo-acme create secret tls kumo-edge-tls' \
  "${TRACE_FILE}" ||
  fail "AKS installer did not create the tenant TLS Secret"
grep -Fq \
  'az aks check-acr --resource-group test-rg --name test-eks --acr test.azurecr.io' \
  "${TRACE_FILE}" ||
  fail "AKS installer did not verify kubelet access to ACR"
grep -Fq \
  'az role assignment create --assignee-object-id 11111111-1111-1111-1111-111111111111 --assignee-principal-type ServicePrincipal --role Contributor --scope /subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-rg/providers/Microsoft.Network/applicationGateways/test-private' \
  "${TRACE_FILE}" ||
  fail "AKS installer did not grant AGIC access to the Application Gateway"
grep -Fq \
  'az role assignment create --assignee-object-id 11111111-1111-1111-1111-111111111111 --assignee-principal-type ServicePrincipal --role Network Contributor --scope /subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-rg/providers/Microsoft.Network/virtualNetworks/test-vnet/subnets/appgw' \
  "${TRACE_FILE}" ||
  fail "AKS installer did not grant AGIC access to the gateway subnet"

# Cloud preflight owns the only early proof of the workload identity that the
# Temporal bootstrap Job logs in as. Without it a mistyped role ARN or client ID
# renders, passes `check cloud`, and first fails after the installer has created
# the Temporal password Secret, as a generic in-Job bootstrap error.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/database.sh"
TEMPORAL_ENABLED=true
POSTGRES_BOOTSTRAP=true
TEMPORAL_NAMESPACE=temporal
POSTGRES_ADMIN_ANNOTATION_KEY=eks.amazonaws.com/role-arn
POSTGRES_ADMIN_ANNOTATION_VALUE=arn:aws:iam::000000000000:role/kumo-postgres-admin
eks_cluster_json='{"identity":{"oidc":{"issuer":"https://oidc.eks.us-west-2.amazonaws.com/id/EXAMPLE"}}}'
trust_condition=StringEquals
trust_subject="system:serviceaccount:temporal:kumo-temporal-bootstrap"
trust_provider="arn:aws:iam::000000000000:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/EXAMPLE"
trust_audience="sts.amazonaws.com"
role_exists=true
role_arn="${POSTGRES_ADMIN_ANNOTATION_VALUE}"

aws() {
  [[ "$*" == "iam get-role"* ]] || fail "unexpected aws call: $*"
  [[ "${role_exists}" == true ]] || return 1
  jq -nc \
    --arg arn "${role_arn}" \
    --arg condition "${trust_condition}" \
    --arg provider "${trust_provider}" \
    --arg audience "${trust_audience}" \
    --arg subject "${trust_subject}" '{
    Arn: $arn,
    AssumeRolePolicyDocument: {
      Statement: [
        {Effect: "Allow", Action: "sts:TagSession"},
        {
          Effect: "Allow",
          Action: ["sts:AssumeRoleWithWebIdentity"],
          Principal: {Federated: $provider},
          Condition: {($condition): {
            "oidc.eks.us-west-2.amazonaws.com/id/EXAMPLE:aud": $audience,
            "oidc.eks.us-west-2.amazonaws.com/id/EXAMPLE:sub": $subject
          }}
        }
      ]
    }
  }'
}

verify_eks_postgres_admin_identity "${eks_cluster_json}"
# A role name carries no path, so the lookup can answer for a different role.
role_arn="arn:aws:iam::000000000000:role/other/kumo-postgres-admin"
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight accepted a role whose ARN is not the configured one"
fi
role_arn="${POSTGRES_ADMIN_ANNOTATION_VALUE}"
# A trust policy copied from another cluster keeps that cluster's provider.
trust_provider="arn:aws:iam::000000000000:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/OTHER"
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight accepted a role federated with another OIDC provider"
fi
# A provider whose URL merely ends with this cluster's issuer is another
# provider, and cannot authenticate this cluster's tokens.
trust_provider="arn:aws:iam::000000000000:oidc-provider/mirror.example/oidc.eks.us-west-2.amazonaws.com/id/EXAMPLE"
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight accepted a provider that only ends with the issuer"
fi
trust_provider="arn:aws:iam::000000000000:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/EXAMPLE"
trust_audience="api://other"
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight accepted a role that demands another token audience"
fi
# An audience condition is read whichever operator states it.
trust_condition=StringLike
trust_audience="other-*"
trust_subject="system:serviceaccount:temporal:kumo-temporal-bootstrap"
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight ignored a StringLike audience condition"
fi
trust_audience="sts.amazonaws.com"
verify_eks_postgres_admin_identity "${eks_cluster_json}"
trust_condition=StringEquals
# An IAM StringLike condition is a glob, and a legitimate way to trust every
# ServiceAccount in the Temporal namespace.
trust_condition=StringLike
trust_subject="system:serviceaccount:temporal:*"
verify_eks_postgres_admin_identity "${eks_cluster_json}"
# IAM's StringLike knows only * and ?; bash pattern syntax must not widen it.
trust_subject="system:serviceaccount:temporal:[k]umo-temporal-bootstrap"
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight matched a StringLike subject as a bash character class"
fi
# StringEquals is literal, so a wildcard spelled there trusts nothing.
trust_condition=StringEquals
trust_subject="system:serviceaccount:temporal:*"
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight read a literal StringEquals subject as a wildcard"
fi
trust_subject="system:serviceaccount:kumo-acme:kumo-control"
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight accepted a role that trusts a different ServiceAccount"
fi
trust_subject="system:serviceaccount:temporal:kumo-temporal-bootstrap"
role_exists=false
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight accepted a PostgreSQL admin role it cannot read"
fi
role_exists=true
POSTGRES_ADMIN_ANNOTATION_VALUE=kumo-postgres-admin
if (verify_eks_postgres_admin_identity "${eks_cluster_json}") 2>/dev/null; then
  fail "EKS preflight accepted an admin annotation that is not an IAM role ARN"
fi
unset -f aws

POSTGRES_ADMIN_ANNOTATION_KEY=azure.workload.identity/client-id
POSTGRES_ADMIN_ANNOTATION_VALUE=11111111-2222-3333-4444-555555555555
admin_identities=$(jq -nc --arg client_id "${POSTGRES_ADMIN_ANNOTATION_VALUE}" '[{
  clientId: $client_id,
  name: "kumo-postgres-admin",
  resourceGroup: "test-rg",
  principalId: "66666666-7777-8888-9999-000000000000"
}]')
federated_subject="system:serviceaccount:temporal:kumo-temporal-bootstrap"
az() {
  case "$*" in
    "identity federated-credential list"*)
      jq -nc --arg subject "${federated_subject}" \
        '[{subject: $subject, audiences: ["api://AzureADTokenExchange"]}]'
      ;;
    "role assignment list"*)
      fail "the bootstrap admin identity was proved against a storage scope"
      ;;
    *) fail "unexpected az call: $*" ;;
  esac
}

# The admin identity reaches PostgreSQL alone: its federated ServiceAccount is
# proved, and no ADLS role assignment is demanded of it.
verify_aks_identity "${POSTGRES_ADMIN_ANNOTATION_VALUE}" \
  "$(temporal_database_identity_subject)" "${admin_identities}" ""
federated_subject="system:serviceaccount:kumo-acme:kumo-control"
if (
  verify_aks_identity "${POSTGRES_ADMIN_ANNOTATION_VALUE}" \
    "$(temporal_database_identity_subject)" "${admin_identities}" ""
) 2>/dev/null; then
  fail "AKS preflight accepted an identity federated with another ServiceAccount"
fi
verify_azure_workload_identity_id "${POSTGRES_ADMIN_ANNOTATION_VALUE}" admin
if (verify_azure_workload_identity_id kumo-postgres-admin admin) 2>/dev/null; then
  fail "preflight accepted a managed-identity client ID that is not a UUID"
fi

# Only the bootstrap needs PostgreSQL 15: it reassigns each Temporal database's
# public schema, which 14 reserves for the owning superuser.
POSTGRES_HOST=kumo-test.postgres.database.azure.com
POSTGRES_RESOURCE_GROUP=test-rg
azure_server_version=15
az() {
  case "$*" in
    "postgres flexible-server show"*)
      jq -nc --arg version "${azure_server_version}" '{
        state: "Ready",
        authConfig: {
          activeDirectoryAuth: "Enabled",
          passwordAuth: "Enabled"
        },
        administratorLogin: "entra_admin",
        version: $version
      }'
      ;;
    "postgres flexible-server parameter show"*) printf 'btree_gin,pgaudit\n' ;;
    *) fail "unexpected az call: $*" ;;
  esac
}
verify_azure_postgres
azure_server_version=14
if (verify_azure_postgres) 2>/dev/null; then
  fail "preflight accepted PostgreSQL 14 for the Temporal bootstrap"
fi
POSTGRES_BOOTSTRAP=false
verify_azure_postgres
unset -f az

echo "Kumo provider dependency-closure and preflight tests: passed"
