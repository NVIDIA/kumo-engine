#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HELM_BIN="${HELM_BIN:-helm}"
KUMO_RELEASE_TAG=$(awk '
  $1 == "kumo:" { in_kumo = 1; next }
  in_kumo && $1 == "tag:" { gsub(/"/, "", $2); print $2; exit }
' "${ROOT}/kumo-stack/renderer/values.yaml")
[[ "${KUMO_RELEASE_TAG}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "FAIL: default Kumo image tag is invalid" >&2
  exit 1
}
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/kumo-stack-render-test.XXXXXX")
trap 'rm -rf "${TMP_ROOT}"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/cloud.sh"

assert_file_contains() {
  local file="$1" value="$2"
  grep -Fq -- "${value}" "${file}" ||
    fail "${file} does not contain ${value}"
}

assert_absent() {
  if grep -Fq -- "$2" "$1"; then
    fail "$1 unexpectedly contains $2"
  fi
}

# Extract one exact manifest document. A missing document is an error so absence
# assertions cannot pass on empty output.
manifest_doc() {
  local file="$1" kind="$2" name="$3" output="$4"
  awk -v kind="kind: ${kind}" -v name="  name: ${name}" '
    found { next }
    $0 == "---" {
      if (isKind && isNamed) { printf "%s", doc; found = 1; next }
      doc = ""; isKind = 0; isNamed = 0; next
    }
    { doc = doc $0 "\n" }
    $0 == kind { isKind = 1 }
    $0 == name { isNamed = 1 }
    END { if (!found && isKind && isNamed) printf "%s", doc }
  ' "${file}" >"${output}"
  [[ -s "${output}" ]] || fail "${file}: no ${kind} named ${name}"
}

assert_pool_contains() {
  local file="$1" pool="$2" value="$3"
  manifest_doc "${file}" NodePool "kumo-${pool}" "${TMP_ROOT}/doc.yaml"
  grep -Fq -- "${value}" "${TMP_ROOT}/doc.yaml" ||
    fail "${file}: NodePool kumo-${pool} does not contain ${value}"
}

# Match the whole line because kumo-gpu prefixes kumo-gpu-ephemeral.
assert_pool_class() {
  local file="$1" pool="$2" class="$3"
  manifest_doc "${file}" NodePool "kumo-${pool}" "${TMP_ROOT}/doc.yaml"
  grep -Fxq -- "        name: ${class}" "${TMP_ROOT}/doc.yaml" ||
    fail "${file}: NodePool kumo-${pool} does not reference EC2NodeClass ${class}"
}

assert_class_contains() {
  local file="$1" class="$2" value="$3"
  manifest_doc "${file}" EC2NodeClass "${class}" "${TMP_ROOT}/doc.yaml"
  grep -Fq -- "${value}" "${TMP_ROOT}/doc.yaml" ||
    fail "${file}: EC2NodeClass ${class} does not contain ${value}"
}

assert_class_absent() {
  local file="$1" class="$2" value="$3"
  manifest_doc "${file}" EC2NodeClass "${class}" "${TMP_ROOT}/doc.yaml"
  if grep -Fq -- "${value}" "${TMP_ROOT}/doc.yaml"; then
    fail "${file}: EC2NodeClass ${class} unexpectedly contains ${value}"
  fi
}

assert_images_exact() {
  local file="$1"
  shift
  local actual expected
  actual=$(
    sed -nE \
      's/^[[:space:]]+image:[[:space:]]*"?([^"[:space:]]+)"?.*$/\1/p' \
      "${file}" | sort -u
  )
  expected=$(printf '%s\n' "$@" | sed '/^$/d' | sort -u)
  [[ "${actual}" == "${expected}" ]] || {
    printf 'Unexpected image inventory in %s\nExpected:\n%s\nActual:\n%s\n' \
      "${file}" "${expected:-<none>}" "${actual:-<none>}" >&2
    fail "rendered images must be limited to Kumo, Temporal, NVIDIA, and provider-managed operands"
  }
  validate_rendered_image_policy test "${file}" "$@"
}

# Renders a platform reference with an overlay and fails if Helm accepts it.
assert_rejected() {
  local platform="$1" description="$2" overlay="$3"
  if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
    -f "${ROOT}/kumo-stack/examples/values-${platform}.yaml" -f "${overlay}" \
    >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
    fail "${description}"
  fi
}

assert_eks_rejected() {
  assert_rejected eks "$1" "$2"
}

# Extracts one rendered release file from the renderer's ConfigMaps the way
# bin/kumo does, so an overlay case can assert on the same per-release files the
# installer feeds to Helm.
render_configmap() {
  local template="$1" key="$2" output="$3"
  shift 3
  "${HELM_BIN}" template kumo-stack "${ROOT}/kumo-stack/renderer" "$@" \
    --show-only "templates/${template}" |
    awk -v marker="  ${key}: |-" '
      $0 == marker { found = 1; next }
      found && /^    / { sub(/^    /, ""); print; next }
      found && $0 == "" { print; next }
      found { exit }
      END { if (!found) exit 2 }
    ' >"${output}"
}

# The `key: value` pairs Kumo's core pods select CPU capacity with.
core_node_selector() {
  awk '
    $0 == "nodeTargeting:" { targeting = 1; next }
    targeting && $0 == "  nodeSelector:" { selector = 1; next }
    selector && $0 == "    {}" { exit }
    selector && /^    [^ ]/ { sub(/^    /, ""); print; next }
    selector { exit }
  ' "$1"
}

# The `key: value` pairs the EKS CPU NodePool stamps on the nodes it launches.
eks_cpu_pool_labels() {
  awk '
    $0 == "  name: kumo-cpu" { pool = 1; next }
    pool && $0 == "---" { exit }
    pool && $0 == "      labels:" { labels = 1; next }
    labels && /^        [^ ]/ { sub(/^        /, ""); print; next }
    labels { exit }
  ' "$1"
}

# The `key: value` pairs the AKS provider stamps on the CPU agent pool. It
# passes CPU_NODE_SELECTOR straight to `az aks nodepool` and compares the live
# pool against it, so the rendered value is the label set.
aks_cpu_pool_labels() {
  sed -n 's/^CPU_NODE_SELECTOR=//p' "$1" | tr ' ' '\n' | sed 's/=/: /'
}

# Capacity that does not carry every label core selects leaves core pods
# Pending with no other symptom, so rendering the selector is not enough.
assert_selector_satisfied() {
  local description="$1" selector_file="$2" labels_file="$3" pair
  [[ -s "${selector_file}" ]] ||
    fail "${description} rendered no CPU node selector"
  [[ -s "${labels_file}" ]] ||
    fail "${description} rendered no CPU capacity labels"
  while IFS= read -r pair; do
    grep -Fxq -- "${pair}" "${labels_file}" ||
      fail "${description} CPU capacity does not carry ${pair}"
  done <"${selector_file}"
}

for provider in eks aks aro; do
  output="${TMP_ROOT}/${provider}"
  HELM_BIN="${HELM_BIN}" "${ROOT}/kumo-stack/bin/kumo" render \
    --values "${ROOT}/kumo-stack/examples/values-${provider}.yaml" \
    --output-dir "${output}" >/dev/null
  assert_file_contains "${output}/core-values.yaml" "enabled: true"
  assert_file_contains "${output}/core-values.yaml" "method: password"
  assert_file_contains "${output}/runtime.env" "AUTH_ENABLED=true"
  assert_file_contains "${output}/runtime.env" "AUTH_METHOD=password"
  assert_file_contains "${output}/runtime.env" \
    "AUTH_SECRET=kumo-bootstrap-password"

  platform_manifest="${TMP_ROOT}/${provider}-platform.yaml"
  "${HELM_BIN}" lint "${ROOT}/kumo-platform-${provider}" \
    -f "${output}/platform-values.yaml" >/dev/null
  "${HELM_BIN}" template "kumo-platform-${provider}" \
    "${ROOT}/kumo-platform-${provider}" \
    -f "${output}/platform-values.yaml" >"${platform_manifest}"

  core_node_selector "${output}/core-values.yaml" \
    >"${output}/core-node-selector.txt"

  if [[ "${provider}" == eks ]]; then
    profile=aws
    registry=nvcr.io
    temporal_registry=docker.io/temporalio
    nvidia_image=nvcr.io/nvidia/k8s-device-plugin:v0.19.0@sha256:7bf6ab18378be099493c9fe50f9cd2d559e0d81f17742d7ac188365408579d5d
    assert_file_contains "${output}/platform-values.yaml" "g4dn.xlarge"
    assert_file_contains "${output}/platform-values.yaml" "g4dn.2xlarge"
    assert_file_contains "${output}/platform-values.yaml" "g4dn.8xlarge"
    # The CPU pool has to hold the whole control plane — 5.6 CPU and 17.4 GiB
    # across six Deployments — which a 4-vCPU node cannot do at any ceiling.
    assert_file_contains "${output}/platform-values.yaml" "m7a.2xlarge"
    assert_pool_contains "${platform_manifest}" cpu "- m7a.2xlarge"
    assert_pool_contains "${platform_manifest}" cpu 'cpu: "32"'
    assert_absent "${platform_manifest}" "- m7a.xlarge"
    assert_file_contains "${output}/core-values.yaml" "authMode: aws_iam"
    assert_file_contains "${output}/runtime.env" \
      "EKS_KARPENTER_CONTROLLER_ROLE_ARN=arn:aws:iam::000000000000:role/KarpenterController-test-eks"
    assert_file_contains "${output}/edge-values.yaml" "mode: alb-acm"
    assert_file_contains "${output}/edge-values.yaml" \
      "certificateArn: \"arn:aws:acm:us-west-2:000000000000:certificate/00000000-0000-0000-0000-000000000000\""
    assert_file_contains "${output}/runtime.env" \
      "KARPENTER_IMAGE=public.ecr.aws/karpenter/controller:1.12.0@sha256:8128bc8cdd0cc8ca2d2cf68ec61033eceb8cfb24834a08ab278784924013cfd0"
    assert_file_contains "${output}/runtime.env" \
      "AWS_LOAD_BALANCER_CONTROLLER_IMAGE=public.ecr.aws/eks/aws-load-balancer-controller:v2.14.0@sha256:58d83491b3eb25080e478bdd9a245b773fd9ceb29d9182fa4b198dee5ff31d22"
    assert_images_exact "${platform_manifest}" \
      "${nvidia_image}"
    # Unset, the storage knobs resolve to the class EKS already named.
    assert_file_contains "${output}/core-values.yaml" \
      'storageClass: "ebs-gp3-xfs"'
    assert_file_contains "${output}/runtime.env" "STORAGE_CLASS=ebs-gp3-xfs"
    assert_file_contains "${output}/platform-values.yaml" 'name: "ebs-gp3-xfs"'
    "${HELM_BIN}" template kumo-platform-eks "${ROOT}/kumo-platform-eks" \
      -f "${output}/platform-values.yaml" \
      --show-only templates/nodepools.yaml >"${output}/nodepools.yaml"
    eks_cpu_pool_labels "${output}/nodepools.yaml" \
      >"${output}/cpu-pool-labels.txt"
    assert_selector_satisfied "the EKS reference" \
      "${output}/core-node-selector.txt" "${output}/cpu-pool-labels.txt"
    # The discriminator is the CPU pool's alone, or it separates nothing.
    [[ "$(grep -c 'kumo.ai/node-pool' "${output}/nodepools.yaml")" == 1 ]] ||
      fail "kumo.ai/node-pool is not exclusive to the EKS CPU NodePool"
  elif [[ "${provider}" == aks ]]; then
    profile=azure
    registry=example.azurecr.io
    temporal_registry=docker.io/temporalio
    nvidia_image=nvcr.io/nvidia/k8s-device-plugin:v0.19.0@sha256:7bf6ab18378be099493c9fe50f9cd2d559e0d81f17742d7ac188365408579d5d
    assert_file_contains "${output}/platform-values.yaml" \
      "repository: nvcr.io/nvidia/k8s-device-plugin"
    assert_file_contains "${output}/core-values.yaml" "authMode: azure_entra"
    assert_file_contains "${output}/edge-values.yaml" "mode: azure-appgw"
    assert_file_contains "${output}/runtime.env" \
      "AKS_APPLICATION_GATEWAY_ID=/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-aks-rg/providers/Microsoft.Network/applicationGateways/test-aks-private"
    assert_images_exact "${platform_manifest}" \
      "${nvidia_image}"
    aks_cpu_pool_labels "${output}/runtime.env" >"${output}/cpu-pool-labels.txt"
    assert_selector_satisfied "the AKS reference" \
      "${output}/core-node-selector.txt" "${output}/cpu-pool-labels.txt"
  else
    profile=azure
    registry=nvcr.io
    temporal_registry=docker.io/temporalio
    nvidia_image=nvcr.io/nvidia/k8s-device-plugin:v0.19.0@sha256:7bf6ab18378be099493c9fe50f9cd2d559e0d81f17742d7ac188365408579d5d
    assert_file_contains "${output}/platform-values.yaml" \
      "cacheMachineSets:"
    assert_file_contains "${output}/core-values.yaml" "authMode: azure_entra"
    assert_file_contains "${output}/edge-values.yaml" "mode: openshift"
    assert_images_exact "${platform_manifest}"
    grep -Fxq \
      "CPU_NODE_SELECTOR=kumo.ai/node-pool=cpu" \
      "${output}/runtime.env" ||
      fail "the ARO reference does not select its labeled CPU capacity"
  fi
  if [[ "${provider}" != eks ]]; then
    assert_absent "${output}/runtime.env" "KARPENTER_IMAGE"
    assert_absent "${output}/runtime.env" \
      "AWS_LOAD_BALANCER_CONTROLLER_IMAGE"
    assert_file_contains "${output}/core-values.yaml" \
      'storageClass: "managed-csi-xfs"'
    assert_file_contains "${output}/platform-values.yaml" \
      'name: "managed-csi-xfs"'
  fi
  # Every reference keeps the platform-created cloud-disk class and advisory
  # tracking; only the knobs are new.
  assert_file_contains "${output}/core-values.yaml" \
    "storageCapacityTracking: advisory"
  assert_file_contains "${output}/runtime.env" "STORAGE_CLASS_CREATE=true"
  assert_file_contains "${output}/platform-values.yaml" "create: true"

  temporal_args=(-f "${ROOT}/kumo-temporal/profiles/${profile}.yaml")
  if [[ "${provider}" == aro ]]; then
    temporal_args+=(-f "${ROOT}/kumo-temporal/profiles/openshift.yaml")
  fi
  temporal_args+=(-f "${output}/temporal-values.yaml")
  temporal_manifest="${TMP_ROOT}/${provider}-temporal.yaml"
  "${HELM_BIN}" lint "${ROOT}/kumo-temporal" \
    "${temporal_args[@]}" >/dev/null
  "${HELM_BIN}" template temporal "${ROOT}/kumo-temporal" \
    --namespace temporal \
    "${temporal_args[@]}" >"${temporal_manifest}"
  assert_images_exact "${temporal_manifest}" \
    "${temporal_registry}/admin-tools:1.31.2@sha256:7e5820112475b3490f011b28d86ca9fd8348f1640b8dd04c62adb906e6b28cb2" \
    "${temporal_registry}/server:1.31.2@sha256:6b02e5176631c8d28f010735a3a73c69423a9adbcfad7e5dfe23836c904b7e26"
  assert_file_contains "${output}/temporal-values.yaml" \
    "kumo.ai/node-pool: cpu"
  if [[ "${provider}" == aro ]]; then
    assert_absent "${temporal_manifest}" "runAsUser: 1000"
    assert_absent "${temporal_manifest}" "fsGroup: 1000"
  else
    assert_file_contains "${temporal_manifest}" "runAsUser: 1000"
    assert_file_contains "${temporal_manifest}" "fsGroup: 1000"
  fi

  core_args=(-f "${ROOT}/kumo-helm/profiles/${profile}.yaml")
  if [[ "${provider}" == aro ]]; then
    core_args+=(-f "${ROOT}/kumo-helm/profiles/openshift.yaml")
  fi
  core_args+=(-f "${output}/core-values.yaml")
  core_manifest="${TMP_ROOT}/${provider}-core.yaml"
  "${HELM_BIN}" lint "${ROOT}/kumo-helm" "${core_args[@]}" >/dev/null
  "${HELM_BIN}" template acme "${ROOT}/kumo-helm" \
    --namespace kumo-acme "${core_args[@]}" >"${core_manifest}"

  # The edge is its own release; kumo-helm must render none of it.
  edge_manifest="${TMP_ROOT}/${provider}-edge.yaml"
  "${HELM_BIN}" lint "${ROOT}/kumo-edge" \
    -f "${output}/edge-values.yaml" >/dev/null
  "${HELM_BIN}" template acme-edge "${ROOT}/kumo-edge" \
    --namespace kumo-acme -f "${output}/edge-values.yaml" >"${edge_manifest}"
  for kind in Ingress Route Gateway VirtualService PeerAuthentication \
    AuthorizationPolicy NetworkPolicy; do
    assert_absent "${core_manifest}" "kind: ${kind}"
  done
  [[ $(grep -c "maxSurge: 0" "${core_manifest}") -eq 6 &&
     $(grep -c "maxUnavailable: 1" "${core_manifest}") -eq 6 ]] ||
    fail "every Kumo deployment must roll without surge capacity"
  assert_images_exact "${core_manifest}" \
    "${registry}/1000161941370113/kumo-3/kumo-backend-cpu:${KUMO_RELEASE_TAG}" \
    "${registry}/1000161941370113/kumo-3/kumo-ui:${KUMO_RELEASE_TAG}"
  assert_file_contains "${core_manifest}" 'DEPLOY_STAGE: "prod"'
  assert_file_contains "${core_manifest}" 'DISABLE_AUTH: "false"'
  if [[ "${provider}" == eks ]]; then
    assert_file_contains "${edge_manifest}" "kind: Ingress"
    assert_file_contains "${edge_manifest}" \
      "alb.ingress.kubernetes.io/certificate-arn"
    assert_absent "${edge_manifest}" "kind: VirtualService"
    assert_absent "${edge_manifest}" "kind: Gateway"
  elif [[ "${provider}" == aro ]]; then
    assert_file_contains "${edge_manifest}" "kind: Route"
    assert_file_contains "${edge_manifest}" "name: kumo-ui"
    assert_file_contains "${edge_manifest}" "targetPort: http"
    assert_absent "${edge_manifest}" "kind: HTTPRoute"
    assert_absent "${edge_manifest}" "kind: Gateway"
  else
    assert_file_contains "${output}/runtime.env" \
      "AKS_CPU_POOL_INSTANCE_TYPE=Standard_D8as_v5"
    assert_file_contains "${edge_manifest}" "kind: Ingress"
    assert_file_contains "${edge_manifest}" \
      "ingressClassName: azure-application-gateway"
    assert_absent "${edge_manifest}" \
      "kubernetes.io/ingress.class"
    assert_file_contains "${edge_manifest}" \
      "appgw.ingress.kubernetes.io/use-private-ip: \"true\""
    assert_file_contains "${edge_manifest}" \
      "secretName: kumo-edge-tls"
    assert_absent "${edge_manifest}" "kind: VirtualService"
    assert_absent "${edge_manifest}" "kind: Gateway"
  fi

  # Every reference keeps the index on a dynamically provisioned cloud disk,
  # whose size is independent of the VM.
  assert_file_contains "${output}/core-values.yaml" "ssd_gib: 256"
  assert_file_contains "${output}/core-values.yaml" "ssd_gib: 512"
  assert_file_contains "${output}/core-values.yaml" "ssd_gib: 1024"
  assert_file_contains "${output}/core-values.yaml" \
    "temporal-frontend.temporal.svc:7233"
  assert_file_contains "${output}/temporal-values.yaml" \
    'existingSecret: "temporal-postgres"'
  assert_file_contains "${output}/runtime.env" \
    "TEMPORAL_SERVER_IMAGE=${temporal_registry}/server:1.31.2@sha256:6b02e5176631c8d28f010735a3a73c69423a9adbcfad7e5dfe23836c904b7e26"
  assert_file_contains "${output}/runtime.env" \
    "TEMPORAL_ADMIN_TOOLS_IMAGE=${temporal_registry}/admin-tools:1.31.2@sha256:7e5820112475b3490f011b28d86ca9fd8348f1640b8dd04c62adb906e6b28cb2"
  assert_file_contains "${output}/runtime.env" \
    "NVIDIA_DEVICE_PLUGIN_IMAGE=${nvidia_image}"
  assert_file_contains "${output}/runtime.env" \
    "/1000161941370113/kumo-3/kumo-backend-cpu:${KUMO_RELEASE_TAG}"
  assert_file_contains "${output}/runtime.env" \
    "/1000161941370113/kumo-3/kumo-backend-gpu:${KUMO_RELEASE_TAG}"
  assert_file_contains "${output}/runtime.env" \
    "/1000161941370113/kumo-3/kumo-ui:${KUMO_RELEASE_TAG}"
  assert_file_contains "${output}/runtime.env" \
    "KUMO_IMAGE_TAG=${KUMO_RELEASE_TAG}"
  assert_file_contains "${output}/runtime.env" \
    "TENANT_SECRET_KEY=key"
  assert_file_contains "${output}/core-values.yaml" \
    "POOL_INIT_CANARY_ENABLED: \"false\""
  grep -A1 -F 'name: POOL_WORKER_TENANT' "${core_manifest}" |
    grep -Fq 'value: "acme"' ||
    fail "pool worker tenant does not match the stack tenant"
  assert_file_contains "${output}/runtime.env" "OBJECT_STORAGE_URI="
  assert_file_contains "${output}/runtime.env" "STORAGE_CLASS="
  assert_file_contains "${output}/namespace-labels.txt" \
    "kumo.ai/temporal-client=true"
  if grep -Fq "istio.io/" "${output}/namespace-labels.txt"; then
    fail "${provider} namespaces unexpectedly require Istio"
  fi
  if grep -Eiq '(^|[[:space:]])password:[[:space:]]+[^[:space:]]' \
    "${output}/temporal-values.yaml"; then
    fail "rendered Temporal values contain a literal password"
  fi
done

override_manifest="${TMP_ROOT}/release-override.yaml"
"${HELM_BIN}" template release-override "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-aks.yaml" \
  --set-string images.kumo.tag=customer-selected-tag \
  --show-only templates/runtime.txt >"${override_manifest}"
assert_file_contains "${override_manifest}" \
  "KUMO_IMAGE_TAG=customer-selected-tag"
assert_absent "${override_manifest}" "KUMO_IMAGE_TAG=${KUMO_RELEASE_TAG}"

# Sizing the CPU pool is done through the stack, so an override has to reach
# the NodePool Karpenter actually acts on rather than stopping at the renderer.
cpu_values="${TMP_ROOT}/values-eks-cpu.yaml"
sed 's/^    eks:$/&\n      cpuNodePool: {instanceTypes: [m7a.4xlarge, m7i.4xlarge], maxCpu: "64"}/' \
  "${ROOT}/kumo-stack/examples/values-eks.yaml" >"${cpu_values}"
assert_file_contains "${cpu_values}" "cpuNodePool:"
cpu_output="${TMP_ROOT}/eks-cpu"
HELM_BIN="${HELM_BIN}" "${ROOT}/kumo-stack/bin/kumo" render \
  --values "${cpu_values}" --output-dir "${cpu_output}" >/dev/null
cpu_manifest="${TMP_ROOT}/eks-cpu-platform.yaml"
"${HELM_BIN}" template kumo-platform-eks "${ROOT}/kumo-platform-eks" \
  -f "${cpu_output}/platform-values.yaml" >"${cpu_manifest}"
assert_pool_contains "${cpu_manifest}" cpu "- m7a.4xlarge"
assert_pool_contains "${cpu_manifest}" cpu "- m7i.4xlarge"
assert_pool_contains "${cpu_manifest}" cpu 'cpu: "64"'
assert_absent "${cpu_manifest}" "- m7a.2xlarge"

# Renderer failures must happen before any provider or cluster operation.
# The platform chart requires a string ceiling, so a bare number must be
# refused here rather than surfacing as a schema error two stages later.
if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  --set cluster.providers.eks.cpuNodePool.maxCpu=64 \
  >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
  fail "numeric CPU pool ceiling unexpectedly rendered"
fi
# The cache tiers next door take a singular instanceType, so the plural here is
# easy to miss. Dropping the misspelling in silence would leave the pool at a
# size the operator believes they changed.
if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  --set 'cluster.providers.eks.cpuNodePool.instanceType={m7a.4xlarge}' \
  >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
  fail "misspelled CPU pool instance types unexpectedly rendered"
fi
if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  --set-string cluster.providers.eks.clusterName= \
  >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
  fail "missing EKS clusterName unexpectedly rendered"
fi
if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-aks.yaml" \
  --set 'cluster.providers.aks.cachePoolNames={onlyone}' \
  >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
  fail "mismatched AKS cache pools unexpectedly rendered"
fi
if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-aks.yaml" \
  --set-string cluster.providers.aks.managedIstioRevision=asm-1-30 \
  >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
  fail "removed AKS managed-Istio setting unexpectedly rendered"
fi
if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-aks.yaml" \
  --set-string 'cluster.meshNamespaceLabels.istio\.io/rev=asm-1-30' \
  >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
  fail "removed stack mesh enrollment unexpectedly rendered"
fi
if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  --set-string tenant.extraValues.postgres.host=other.example.com \
  >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
  fail "tenant.extraValues overrode stack-managed PostgreSQL"
fi
if "${HELM_BIN}" template invalid "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  --set-string tenant.extraValues.workloads.poolcontroller.extraEnv.POOL_INIT_CANARY_ENABLED=true \
  >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
  fail "tenant.extraValues overrode the staged GPU lifecycle"
fi
# extraEnv is appended after the chart's own environment, so a duplicate name
# would win in the container and defeat the values the stack verified.
cat >"${TMP_ROOT}/reserved-env.yaml" <<'EOF'
tenant:
  extraValues:
    workloads:
      poolcontroller:
        extraEnv: {POOL_STORAGE_CLASS_NAME: other}
EOF
assert_eks_rejected "tenant.extraValues overrode POOL_STORAGE_CLASS_NAME" \
  "${TMP_ROOT}/reserved-env.yaml"

# Every configurable platform value: the reference default above, an override
# that is honoured, and an invalid setting that is refused before any cluster
# operation.

# GPU SKUs are resolved from a table, so both supported families render their
# real vCPU ceiling and anything unresolvable is refused. The reference is T4;
# these are the L40S sizes.
cat >"${TMP_ROOT}/l40s-tiers.yaml" <<'EOF'
gpu:
  cacheTiers:
    - {name: small, instanceType: g6e.2xlarge, ssdGiB: 256, gpuCount: 1}
    - {name: large, instanceType: g6e.16xlarge, ssdGiB: 1024, gpuCount: 1}
  ephemeral: {instanceType: g6e.xlarge}
EOF
l40s_output="${TMP_ROOT}/eks-l40s.yaml"
"${HELM_BIN}" template l40s "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  -f "${TMP_ROOT}/l40s-tiers.yaml" >"${l40s_output}"
assert_file_contains "${l40s_output}" "instanceTypes: [g6e.16xlarge]"
assert_file_contains "${l40s_output}" 'maxCpu: "64"'
assert_file_contains "${l40s_output}" 'maxCpu: "4"'

cat >"${TMP_ROOT}/unknown-sku.yaml" <<'EOF'
gpu:
  ephemeral: {instanceType: p5.48xlarge}
EOF
assert_eks_rejected "unknown EKS GPU SKU rendered" "${TMP_ROOT}/unknown-sku.yaml"

cat >"${TMP_ROOT}/wrong-gpu-count.yaml" <<'EOF'
gpu:
  cacheTiers:
    - {name: small, instanceType: g6e.12xlarge, ssdGiB: 256, gpuCount: 1}
    - {name: large, instanceType: g6e.16xlarge, ssdGiB: 1024, gpuCount: 1}
EOF
assert_eks_rejected "cache tier mislabelled a multi-GPU SKU" \
  "${TMP_ROOT}/wrong-gpu-count.yaml"

cat >"${TMP_ROOT}/duplicate-sku.yaml" <<'EOF'
gpu:
  cacheTiers:
    - {name: small, instanceType: g6e.2xlarge, ssdGiB: 256, gpuCount: 1}
    - {name: medium, instanceType: g6e.2xlarge, ssdGiB: 512, gpuCount: 1}
EOF
assert_eks_rejected "cache tiers collided on one nodeTypes key" \
  "${TMP_ROOT}/duplicate-sku.yaml"

# Azure has no SKU table to cross-check a GPU count against, so its tiers stay
# pinned to one GPU.
cat >"${TMP_ROOT}/azure-multi-gpu.yaml" <<'EOF'
gpu:
  cacheTiers:
    - {name: small, instanceType: Standard_NC4as_T4_v3, ssdGiB: 256, gpuCount: 2}
    - {name: large, instanceType: Standard_NC16as_T4_v3, ssdGiB: 1024, gpuCount: 1}
EOF
for azure_platform in aks aro; do
  assert_rejected "${azure_platform}" \
    "${azure_platform} accepted a multi-GPU cache tier" \
    "${TMP_ROOT}/azure-multi-gpu.yaml"
done

# A local-volume cache keeps its NVMe raw for the volume manager.
cat >"${TMP_ROOT}/adopted-class.yaml" <<'EOF'
cluster:
  providers:
    eks: {cacheGpuInstanceStorePolicy: ""}
storage:
  diskgraph:
    storageClass: {name: topolvm-ssd-xfs, create: false}
    capacityTracking: required
EOF
adopted_dir="${TMP_ROOT}/eks-adopted"
mkdir -p "${adopted_dir}"
adopted_values=(
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml"
  -f "${TMP_ROOT}/adopted-class.yaml"
)
render_configmap core-values.yaml values.yaml \
  "${adopted_dir}/core-values.yaml" "${adopted_values[@]}"
render_configmap platform-values.yaml values.yaml \
  "${adopted_dir}/platform-values.yaml" "${adopted_values[@]}"
render_configmap runtime.txt runtime.txt \
  "${adopted_dir}/runtime.env" "${adopted_values[@]}"
assert_file_contains "${adopted_dir}/core-values.yaml" \
  'storageClass: "topolvm-ssd-xfs"'
assert_file_contains "${adopted_dir}/core-values.yaml" \
  "storageCapacityTracking: required"
assert_file_contains "${adopted_dir}/runtime.env" \
  "STORAGE_CLASS=topolvm-ssd-xfs"
assert_file_contains "${adopted_dir}/runtime.env" "STORAGE_CLASS_CREATE=false"
assert_file_contains "${adopted_dir}/platform-values.yaml" "create: false"
# The platform release leaves the cache instance store raw.
adopted_manifest="${TMP_ROOT}/eks-adopted-platform.yaml"
"${HELM_BIN}" template eks "${ROOT}/kumo-platform-eks" \
  -f "${adopted_dir}/platform-values.yaml" >"${adopted_manifest}"
assert_absent "${adopted_manifest}" "kind: StorageClass"
assert_class_absent "${adopted_manifest}" kumo-gpu "instanceStorePolicy"
# Overflow remains kubelet-backed regardless of the cache policy.
assert_class_contains "${adopted_manifest}" kumo-gpu-ephemeral \
  "instanceStorePolicy: RAID0"
assert_pool_class "${adopted_manifest}" ephemeral kumo-gpu-ephemeral
# Cache pools retain their existing class name.
for adopted_tier in small medium large; do
  assert_pool_class "${adopted_manifest}" "cache-${adopted_tier}" kumo-gpu
done
# The reference, which sets none of this, still creates its cloud-disk class.
reference_manifest="${TMP_ROOT}/eks-reference-platform.yaml"
"${HELM_BIN}" template eks "${ROOT}/kumo-platform-eks" \
  -f "${TMP_ROOT}/eks/platform-values.yaml" >"${reference_manifest}"
assert_file_contains "${reference_manifest}" "kind: StorageClass"
assert_class_contains "${reference_manifest}" kumo-gpu \
  "instanceStorePolicy: RAID0"
assert_class_contains "${reference_manifest}" kumo-gpu-ephemeral \
  "instanceStorePolicy: RAID0"

# Cache root sizing is independent of the instance-store policy.
for root_manifest in "${adopted_manifest}" "${reference_manifest}"; do
  assert_class_contains "${root_manifest}" kumo-gpu "volumeSize: 200Gi"
  assert_class_contains "${root_manifest}" kumo-gpu "deviceName: /dev/xvda"
  # Ephemeral and CPU nodes can use the default root volume.
  assert_class_absent "${root_manifest}" kumo-gpu-ephemeral "blockDeviceMappings"
  assert_class_absent "${root_manifest}" kumo-cpu "blockDeviceMappings"
done
sized_dir="${TMP_ROOT}/eks-sized-root"
mkdir -p "${sized_dir}"
cat >"${TMP_ROOT}/sized-root.yaml" <<'EOF'
cluster:
  providers:
    eks: {cacheGpuRootVolumeGiB: 250}
EOF
render_configmap platform-values.yaml values.yaml \
  "${sized_dir}/platform-values.yaml" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  -f "${TMP_ROOT}/sized-root.yaml"
sized_manifest="${TMP_ROOT}/eks-sized-platform.yaml"
"${HELM_BIN}" template eks "${ROOT}/kumo-platform-eks" \
  -f "${sized_dir}/platform-values.yaml" >"${sized_manifest}"
assert_class_contains "${sized_manifest}" kumo-gpu "volumeSize: 250Gi"
# Reject the 20 GiB AMI default that caused DiskPressure.
cat >"${TMP_ROOT}/tiny-root.yaml" <<'EOF'
cluster:
  providers:
    eks: {cacheGpuRootVolumeGiB: 20}
EOF
assert_eks_rejected "cache root volume below the GPU image floor rendered" \
  "${TMP_ROOT}/tiny-root.yaml"

cat >"${TMP_ROOT}/bad-cache-instance-store.yaml" <<'EOF'
cluster:
  providers:
    eks: {cacheGpuInstanceStorePolicy: RAID5}
EOF
assert_eks_rejected "invalid cache instance-store policy rendered" \
  "${TMP_ROOT}/bad-cache-instance-store.yaml"
# A bare YAML null removes the required key.
cat >"${TMP_ROOT}/null-cache-instance-store.yaml" <<'EOF'
cluster:
  providers:
    eks:
      cacheGpuInstanceStorePolicy:
EOF
assert_eks_rejected "bare (null) cache instance-store policy rendered" \
  "${TMP_ROOT}/null-cache-instance-store.yaml"
# Misspellings are rejected as unknown provider keys.
cat >"${TMP_ROOT}/misspelled-cache-instance-store.yaml" <<'EOF'
cluster:
  providers:
    eks: {cacheGpuInstanceStoragePolicy: ""}
EOF
assert_eks_rejected "misspelled cache instance-store policy rendered" \
  "${TMP_ROOT}/misspelled-cache-instance-store.yaml"

# Several schema-valid class names are YAML 1.1 scalars: unquoted, `null` and
# friends reach kumo-helm as a null and the pool controller as an empty name.
cat >"${TMP_ROOT}/retyped-class.yaml" <<'EOF'
storage:
  diskgraph:
    storageClass: {name: "null", create: true}
    capacityTracking: advisory
EOF
retyped_output="${TMP_ROOT}/eks-retyped.yaml"
"${HELM_BIN}" template retyped "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  -f "${TMP_ROOT}/retyped-class.yaml" >"${retyped_output}"
assert_file_contains "${retyped_output}" 'storageClass: "null"'
assert_absent "${retyped_output}" 'storageClass: null'
assert_file_contains "${retyped_output}" 'name: "null"'
assert_file_contains "${retyped_output}" 'STORAGE_CLASS=null'

cat >"${TMP_ROOT}/bad-tracking.yaml" <<'EOF'
storage:
  diskgraph: {capacityTracking: maybe}
EOF
assert_eks_rejected "invalid storageCapacityTracking rendered" \
  "${TMP_ROOT}/bad-tracking.yaml"
# No cloud CSI driver publishes CSIStorageCapacity, so a class the stack
# creates can never satisfy `required`.
cat >"${TMP_ROOT}/created-required.yaml" <<'EOF'
storage:
  diskgraph:
    storageClass: {create: true}
    capacityTracking: required
EOF
for tracking_platform in eks aks aro; do
  assert_rejected "${tracking_platform}" \
    "${tracking_platform} asked a stack-created cloud class for capacity tracking" \
    "${TMP_ROOT}/created-required.yaml"
done

# CPU node targeting is the operator's, but EKS/AKS must name it: their GPU
# pools are labelled and not tainted, so an unselected Kumo pod can pin one up.
# A selector that only reaches the core release is worse than none — the
# capacity the stack creates would not carry it and core would sit Pending — so
# it has to arrive at the EKS NodePool and the AKS agent pool too.
cat >"${TMP_ROOT}/cpu-selector.yaml" <<'EOF'
cluster:
  cpuNodeSelector: {kumo.ai/node-pool: cpu, kumo.ai/instance-type: kumo-default}
EOF
for selector_platform in eks aks; do
  selector_output="${TMP_ROOT}/${selector_platform}-selector"
  selector_values=(
    -f "${ROOT}/kumo-stack/examples/values-${selector_platform}.yaml"
    -f "${TMP_ROOT}/cpu-selector.yaml"
  )
  mkdir -p "${selector_output}"
  render_configmap core-values.yaml values.yaml \
    "${selector_output}/core-values.yaml" "${selector_values[@]}"
  core_node_selector "${selector_output}/core-values.yaml" \
    >"${selector_output}/core-node-selector.txt"
  assert_file_contains "${selector_output}/core-node-selector.txt" \
    "kumo.ai/instance-type: kumo-default"
  if [[ "${selector_platform}" == eks ]]; then
    render_configmap platform-values.yaml values.yaml \
      "${selector_output}/platform-values.yaml" "${selector_values[@]}"
    "${HELM_BIN}" template kumo-platform-eks "${ROOT}/kumo-platform-eks" \
      -f "${selector_output}/platform-values.yaml" \
      --show-only templates/nodepools.yaml \
      >"${selector_output}/nodepools.yaml"
    eks_cpu_pool_labels "${selector_output}/nodepools.yaml" \
      >"${selector_output}/cpu-pool-labels.txt"
    # The whole point of the required discriminator is that GPU pools never
    # carry it, so it must appear on the CPU pool and nowhere else.
    [[ "$(grep -c 'kumo.ai/node-pool' \
      "${selector_output}/nodepools.yaml")" == 1 ]] ||
      fail "kumo.ai/node-pool is not exclusive to the EKS CPU NodePool"
  else
    render_configmap runtime.txt runtime.txt \
      "${selector_output}/runtime.env" "${selector_values[@]}"
    aks_cpu_pool_labels "${selector_output}/runtime.env" \
      >"${selector_output}/cpu-pool-labels.txt"
  fi
  assert_selector_satisfied "a custom ${selector_platform} CPU selector" \
    "${selector_output}/core-node-selector.txt" \
    "${selector_output}/cpu-pool-labels.txt"
done
cat >"${TMP_ROOT}/no-selector.yaml" <<'EOF'
cluster:
  cpuNodeSelector: null
EOF
assert_eks_rejected "EKS rendered without CPU node targeting" \
  "${TMP_ROOT}/no-selector.yaml"
# Blocking known-bad keys cannot prove the rest are good: every EKS GPU
# NodePool is on-demand too, so this selector would reach GPU capacity. Where
# the stack builds the CPU pool, the selector has to keep the one label no GPU
# pool carries.
cat >"${TMP_ROOT}/no-discriminator.yaml" <<'EOF'
cluster:
  cpuNodeSelector: {karpenter.sh/capacity-type: on-demand}
EOF
for discriminator_platform in eks aks; do
  assert_rejected "${discriminator_platform}" \
    "${discriminator_platform} accepted a CPU selector its GPU pools satisfy" \
    "${TMP_ROOT}/no-discriminator.yaml"
done
# ARO builds no CPU pool, so it has no such label to require; `check cloud`
# proves the same thing against the MachineSets it adopts. Its reference sets
# no selector at all, so this overlay needs no key removed.
cat >"${TMP_ROOT}/aro-plain-selector.yaml" <<'EOF'
cluster:
  cpuNodeSelector: {node-role.kubernetes.io/worker: ""}
EOF
"${HELM_BIN}" template discriminator "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-aro.yaml" \
  -f "${TMP_ROOT}/aro-plain-selector.yaml" >/dev/null ||
  fail "ARO was made to require a CPU pool label the stack never creates"
cat >"${TMP_ROOT}/wrong-discriminator.yaml" <<'EOF'
cluster:
  cpuNodeSelector: {kumo.ai/node-pool: gpu}
EOF
assert_eks_rejected "EKS accepted a CPU selector naming another pool" \
  "${TMP_ROOT}/wrong-discriminator.yaml"
# A label key Kubernetes refuses would otherwise reach a Karpenter NodePool or
# `az aks nodepool --labels` and fail there. The prefix is a DNS subdomain, so
# a malformed segment, and a prefix past the 253-character subdomain limit, are
# as invalid as a space.
dns_label=$(printf 'a%.0s' $(seq 1 63))
long_prefix="${dns_label}.${dns_label}.${dns_label}.${dns_label}"
cat >"${TMP_ROOT}/longest-selector.yaml" <<EOF
cluster:
  cpuNodeSelector: {kumo.ai/node-pool: cpu, "${long_prefix:0:253}/key": cpu}
EOF
"${HELM_BIN}" template longest "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  -f "${TMP_ROOT}/longest-selector.yaml" >/dev/null ||
  fail "the longest valid CPU node selector prefix was rejected"
for bad_key in "not a label key" "a..b/key" "a-.b/key" "${long_prefix}/key"; do
  cat >"${TMP_ROOT}/bad-selector.yaml" <<EOF
cluster:
  cpuNodeSelector: {kumo.ai/node-pool: cpu, "${bad_key}": cpu}
EOF
  assert_eks_rejected "CPU node selector ${bad_key} rendered" \
    "${TMP_ROOT}/bad-selector.yaml"
  # The leaf chart is usable without the renderer, so it validates the same
  # keys rather than stamping whatever it is handed onto a NodePool.
  cat >"${TMP_ROOT}/bad-pool-label.yaml" <<EOF
nodePools:
  cpu:
    labels: {"${bad_key}": cpu}
EOF
  if "${HELM_BIN}" template invalid "${ROOT}/kumo-platform-eks" \
    -f "${ROOT}/kumo-platform-eks/test/values.yaml" \
    -f "${TMP_ROOT}/bad-pool-label.yaml" \
    >"${TMP_ROOT}/invalid.yaml" 2>"${TMP_ROOT}/invalid.log"; then
    fail "kumo-platform-eks rendered CPU node-pool label ${bad_key}"
  fi
done

# These keys mark GPU capacity on every platform. Selecting one would put core
# on a scale-to-zero GPU node, and on EKS and AKS the stack would stamp the
# marker onto the CPU pool it creates as well.
# kubelet sets the kubernetes.io/{os,arch} pair on every node, GPU ones
# included, so a selector built from them tells no capacity apart.
for gpu_key in kumo.ai/gpu-node kumo.ai/diskgraph-pool-type \
  kumo.ai/diskgraph-ssd-size kumo.ai/diskgraph-anything \
  kubernetes.io/os kubernetes.io/arch \
  beta.kubernetes.io/os beta.kubernetes.io/arch \
  topology.kubernetes.io/region topology.kubernetes.io/zone \
  failure-domain.beta.kubernetes.io/region \
  failure-domain.beta.kubernetes.io/zone; do
  cat >"${TMP_ROOT}/gpu-selector.yaml" <<EOF
cluster:
  cpuNodeSelector: {kumo.ai/node-pool: cpu, ${gpu_key}: "true"}
EOF
  for gpu_platform in eks aks aro; do
    assert_rejected "${gpu_platform}" \
      "${gpu_platform} CPU node selector claimed GPU capacity with ${gpu_key}" \
      "${TMP_ROOT}/gpu-selector.yaml"
  done
done

# Explicit block mappings replace AMI defaults; /dev/xvda is valid only for
# AL2023.
if "${HELM_BIN}" template eks "${ROOT}/kumo-platform-eks" \
  -f "${ROOT}/kumo-platform-eks/test/values.yaml" \
  --set-string karpenter.amiAlias=bottlerocket@latest >/dev/null 2>&1; then
  fail "kumo-platform-eks accepted a non-AL2023 amiAlias"
fi
"${HELM_BIN}" template eks "${ROOT}/kumo-platform-eks" \
  -f "${ROOT}/kumo-platform-eks/test/values.yaml" \
  --set-string karpenter.amiAlias=al2023@v20240807 >/dev/null

# The discriminator is the chart's own, not the operator's: a conflicting value
# in nodePools.cpu.labels must not be able to displace it.
cat >"${TMP_ROOT}/conflicting-pool-label.yaml" <<'EOF'
nodePools:
  cpu:
    labels: {kumo.ai/node-pool: custom, kumo.ai/instance-type: kumo-default}
EOF
conflicting_manifest="${TMP_ROOT}/eks-conflicting-labels.yaml"
"${HELM_BIN}" template eks "${ROOT}/kumo-platform-eks" \
  -f "${ROOT}/kumo-platform-eks/test/values.yaml" \
  -f "${TMP_ROOT}/conflicting-pool-label.yaml" \
  --show-only templates/nodepools.yaml >"${conflicting_manifest}"
assert_file_contains "${conflicting_manifest}" "kumo.ai/node-pool: cpu"
assert_absent "${conflicting_manifest}" "kumo.ai/node-pool: custom"
assert_file_contains "${conflicting_manifest}" "kumo.ai/instance-type: kumo-default"

# `check cloud` decides whether a StorageClass is stack-created by reading
# storageClass.create and storageClass.name out of the deployed platform
# release. That is only sound while the charts render the class from exactly
# those two values, so pin the contract for all three.
for provider in eks aks aro; do
  contract_args=()
  if [[ -r "${ROOT}/kumo-platform-${provider}/test/values.yaml" ]]; then
    contract_args=(-f "${ROOT}/kumo-platform-${provider}/test/values.yaml")
  fi
  "${HELM_BIN}" template contract "${ROOT}/kumo-platform-${provider}" \
    "${contract_args[@]}" \
    --set storageClass.create=true \
    --set-string storageClass.name=contract-class \
    --show-only templates/storageclass.yaml \
    >"${TMP_ROOT}/contract-${provider}.yaml"
  assert_file_contains "${TMP_ROOT}/contract-${provider}.yaml" \
    "kind: StorageClass"
  assert_file_contains "${TMP_ROOT}/contract-${provider}.yaml" \
    'name: "contract-class"'
  "${HELM_BIN}" template contract "${ROOT}/kumo-platform-${provider}" \
    "${contract_args[@]}" --set storageClass.create=false \
    >"${TMP_ROOT}/contract-${provider}-none.yaml"
  assert_absent "${TMP_ROOT}/contract-${provider}-none.yaml" "kind: StorageClass"
done

# External Temporal and pre-created Secret modes do not need database-admin
# credentials. Keep those paths renderable without dummy values.
"${HELM_BIN}" template external "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-eks.yaml" \
  --set temporal.enabled=false \
  --set-string temporal.externalAddress=temporal.customer.example:7233 \
  --set cloud.postgres.bootstrapTemporal=false \
  --set-string cloud.postgres.admin.mode= \
  --set-string cloud.postgres.admin.username= \
  >"${TMP_ROOT}/external-temporal.yaml"
assert_file_contains "${TMP_ROOT}/external-temporal.yaml" \
  "registerNamespace: false"
"${HELM_BIN}" template existing-secret "${ROOT}/kumo-stack/renderer" \
  -f "${ROOT}/kumo-stack/examples/values-aks.yaml" \
  --set cloud.postgres.bootstrapTemporal=false \
  --set-string cloud.postgres.admin.mode= \
  --set-string cloud.postgres.admin.username= >/dev/null

eks_manifest="${TMP_ROOT}/eks-platform.yaml"
"${HELM_BIN}" template eks "${ROOT}/kumo-platform-eks" \
  -f "${ROOT}/kumo-platform-eks/test/values.yaml" >"${eks_manifest}"
assert_file_contains "${eks_manifest}" "reclaimPolicy: Delete"
# The chart's own defaults, with no stack values in front of them.
assert_class_contains "${eks_manifest}" kumo-gpu "instanceStorePolicy: RAID0"
assert_class_contains "${eks_manifest}" kumo-gpu-ephemeral \
  "instanceStorePolicy: RAID0"
assert_class_absent "${eks_manifest}" kumo-cpu "instanceStorePolicy"

aks_manifest="${TMP_ROOT}/aks-platform.yaml"
"${HELM_BIN}" template aks "${ROOT}/kumo-platform-aks" >"${aks_manifest}"
assert_file_contains "${aks_manifest}" "reclaimPolicy: Delete"

aro_manifest="${TMP_ROOT}/aro-platform.yaml"
aro_existing_manifest="${TMP_ROOT}/aro-platform-existing.yaml"
"${HELM_BIN}" template aro "${ROOT}/kumo-platform-aro" \
  -f "${ROOT}/kumo-platform-aro/test/values.yaml" >"${aro_existing_manifest}"
assert_absent "${aro_existing_manifest}" "kind: Subscription"
assert_absent "${aro_existing_manifest}" "kind: NodeFeatureDiscovery"
assert_absent "${aro_existing_manifest}" "kind: ClusterPolicy"
"${HELM_BIN}" template aro "${ROOT}/kumo-platform-aro" \
  -f "${ROOT}/kumo-platform-aro/test/values.yaml" \
  --set operators.mode=dedicated \
  --set nfd.create=true \
  --set clusterPolicy.create=true >"${aro_manifest}"
assert_images_exact "${aro_manifest}" \
  "registry.redhat.io/openshift4/ose-node-feature-discovery-rhel9:v4.20"
assert_file_contains "${aro_manifest}" "kind: NodeFeatureDiscovery"
assert_absent "${aro_manifest}" "topologyupdater"
assert_absent "${aro_manifest}" "topologyUpdater"
assert_file_contains "${aro_manifest}" \
  "registry.redhat.io/openshift4/ose-node-feature-discovery-rhel9:v4.20"
assert_file_contains "${aro_manifest}" "channel: v26.3"
assert_file_contains "${aro_manifest}" "kind: ClusterPolicy"
for cluster_policy_field in daemonsets dcgm dcgmExporter devicePlugin driver \
  gfd nodeStatusExporter operator toolkit; do
  assert_file_contains "${aro_manifest}" "  ${cluster_policy_field}:"
done
assert_file_contains "${aro_manifest}" "use_ocp_driver_toolkit: true"
assert_absent "${aro_manifest}" "servicemeshoperator"
assert_absent "${aro_manifest}" "kind: Istio"
assert_absent "${aro_manifest}" "kind: ZTunnel"

# The installer must reject unexpected rendered images before it reaches a
# Helm install or upgrade. These representative helpers are explicitly not
# part of the supported stack.
for forbidden_image in \
  quay.io/jetstack/cert-manager-controller:v1.21.0 \
  registry.k8s.io/ingress-nginx/controller:v1.15.1 \
  ghcr.io/external-secrets/external-secrets:v1.0.0 \
  quay.io/topolvm/topolvm-with-sidecar:0.36.0; do
  printf '      image: %s\n' "${forbidden_image}" \
    >"${TMP_ROOT}/forbidden-image.yaml"
  if (
    validate_rendered_image_policy core "${TMP_ROOT}/forbidden-image.yaml" \
      "example.azurecr.io/1000161941370113/kumo-3/kumo-backend-cpu:test"
  ) 2>/dev/null; then
    fail "installer accepted non-permitted image ${forbidden_image}"
  fi
done

# The release readers run against whatever Helm the operator has, and a flag
# one major version dropped would otherwise surface only against a live
# cluster: `--all` was exactly that, removed from `helm list` in Helm 4. An
# unreachable cluster here is expected and ignored; a rejected flag is not.
assert_helm_accepts_flags() {
  local output
  output=$("${HELM_BIN}" "$@" 2>&1) || true
  if grep -Eq 'unknown (flag|shorthand flag)|flag provided but not defined' \
    <<<"${output}"; then
    fail "this Helm rejects the flags behind '$*': ${output}"
  fi
}
helm_list_args=()
[[ "$("${HELM_BIN}" version --short)" != v3.* ]] || helm_list_args=(--all)
assert_helm_accepts_flags list --namespace kumo-system \
  "${helm_list_args[@]}" --filter '^kumo-platform-eks$' --output json
assert_helm_accepts_flags get values kumo-platform-eks \
  --namespace kumo-system --all --output json

bash -n "${ROOT}/kumo-stack/bin/kumo"
for provider in "${ROOT}"/kumo-stack/providers/*.sh; do
  bash -n "${provider}"
done
for stage in "${ROOT}"/kumo-stack/stages/*.sh; do
  bash -n "${stage}"
done
# The Job assets run only inside the cluster, mounted from a ConfigMap, so a
# syntax error in them would otherwise surface no earlier than an install.
sh -n "${ROOT}/kumo-stack/files/temporal-database.sh"
PYTHONPYCACHEPREFIX="${TMP_ROOT}" python3 -m py_compile \
  "${ROOT}/kumo-stack/files/postgres-admin-token.py"

help_output=$("${ROOT}/kumo-stack/bin/kumo" render --help)
for command in \
  "check cloud" "install [platform" "bootstrap database" \
  "verify platform" upgrade conformance decommission; do
  grep -Fq "kumo ${command}" <<<"${help_output}" ||
    fail "installer help does not expose ${command}"
done
for provider in eks aks aro; do
  grep -Fq "provider_decommission()" \
    "${ROOT}/kumo-stack/providers/${provider}.sh" ||
    fail "${provider} provider has no decommission implementation"
done
if grep -Fq -- "--enable-managed-gpu" \
  "${ROOT}/kumo-stack/providers/aks.sh"; then
  fail "AKS provider uses the preview managed-GPU profile"
fi

echo "Kumo stack render tests: passed"
