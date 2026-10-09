#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

container_image_is_permitted() {
  local image="$1"
  case "${image}" in
    "${KUMO_CPU_IMAGE}"|"${KUMO_GPU_IMAGE}"|"${KUMO_UI_IMAGE}"|\
    "${TEMPORAL_SERVER_IMAGE}"|"${TEMPORAL_ADMIN_TOOLS_IMAGE}"|\
    "${POSTGRES_CLIENT_IMAGE}"|\
    "${NVIDIA_DEVICE_PLUGIN_IMAGE}"|"${KARPENTER_IMAGE}"|\
    "${AWS_LOAD_BALANCER_CONTROLLER_IMAGE}")
      return 0
      ;;
  esac
  case "${PLATFORM}:${image}" in
    eks:public.ecr.aws/*|\
    eks:*.dkr.ecr.*.amazonaws.com/*)
      return 0
      ;;
    aks:mcr.microsoft.com/*)
      return 0
      ;;
    aro:registry.redhat.io/*|\
    aro:registry.access.redhat.com/*|\
    aro:quay.io/openshift-release-dev/*|\
    aro:quay.io/openshift/*|\
    aro:nvcr.io/nvidia/*)
      return 0
      ;;
  esac
  return 1
}

stage_container_policy_verify() {
  local pods image failed=false
  pods=$(kubectl_cmd get pods --all-namespaces -o json)
  while IFS= read -r image; do
    [[ -n "${image}" ]] || continue
    if ! container_image_is_permitted "${image}"; then
      printf 'non-permitted cluster image: %s\n' "${image}" >&2
      failed=true
    fi
  done < <("${JQ}" -r '
    .items[] |
    (
      .spec.initContainers[]?,
      .spec.containers[]?,
      .spec.ephemeralContainers[]?
    ) | .image
  ' <<<"${pods}" | sort -u)
  [[ "${failed}" == false ]] ||
    die "fresh cluster contains an image outside the Kumo deployment policy"
}

require_namespaced_secret() {
  local namespace="$1" name="$2" description="$3"
  [[ -z "${name}" ]] ||
    kubectl_cmd -n "${namespace}" get secret "${name}" >/dev/null 2>&1 ||
    die "missing ${description} Secret ${namespace}/${name}"
}

ensure_tenant_secret() {
  local secret_file="${WORK_DIR}/tenant-secret-key"
  if kubectl_cmd -n "${TENANT_NAMESPACE}" get secret \
    "${TENANT_SECRET}" >/dev/null 2>&1; then
    kubectl_cmd -n "${TENANT_NAMESPACE}" get secret \
      "${TENANT_SECRET}" -o json |
      "${JQ}" -e --arg key "${TENANT_SECRET_KEY}" \
        '.data[$key] | type == "string" and length > 0' >/dev/null ||
      die "Secret ${TENANT_NAMESPACE}/${TENANT_SECRET} lacks key ${TENANT_SECRET_KEY}"
    return
  fi

  "${OPENSSL}" rand 32 |
    "${OPENSSL}" base64 -A |
    tr '+/' '-_' >"${secret_file}"
  kubectl_cmd -n "${TENANT_NAMESPACE}" create secret generic \
    "${TENANT_SECRET}" \
    "--from-file=${TENANT_SECRET_KEY}=${secret_file}" >/dev/null
}

show_core_failure() {
  printf 'Kumo core diagnostics:\n' >&2
  kubectl_cmd -n "${TENANT_NAMESPACE}" get pods,jobs,deployments >&2 || true
  kubectl_cmd -n "${TENANT_NAMESPACE}" get events \
    --sort-by=.metadata.creationTimestamp 2>/dev/null |
    tail -80 >&2 || true
}

stage_core_apply() {
  ensure_namespace "${TENANT_NAMESPACE}"
  label_namespace "${TENANT_NAMESPACE}" true
  ensure_tenant_secret
  require_namespaced_secret "${TENANT_NAMESPACE}" "${KUMO_PULL_SECRET}" \
    "image pull"
  if [[ "${AUTH_ENABLED}" == true ]]; then
    require_namespaced_secret "${TENANT_NAMESPACE}" "${AUTH_SECRET}" \
      "${AUTH_METHOD} authentication"
  fi

  core_profile_args
  if ! helm_upgrade "${TENANT_RELEASE}" "${REPO_ROOT}/kumo-helm" \
    --namespace "${TENANT_NAMESPACE}" \
    --create-namespace \
    --atomic --wait --timeout "${TIMEOUT}" \
    "${CORE_PROFILE_ARGS[@]}"; then
    show_core_failure
    return 1
  fi

  # The provider edge is a separate release: kumo-helm does not own ingress,
  # but a turnkey install still needs a front door.
  edge_profile_args
  if ! helm_upgrade "${TENANT_RELEASE}-edge" "${REPO_ROOT}/kumo-edge" \
    --namespace "${TENANT_NAMESPACE}" \
    --atomic --wait --timeout "${TIMEOUT}" \
    "${EDGE_PROFILE_ARGS[@]}"; then
    show_core_failure
    return 1
  fi
}

stage_core_verify() {
  helm_release_is_deployed "${TENANT_RELEASE}" "${TENANT_NAMESPACE}"
  kubectl_cmd -n "${TENANT_NAMESPACE}" rollout status deployment \
    --selector "tenant=${TENANT}" --timeout="${TIMEOUT}"
  local pods failed pending unready
  pods=$(kubectl_cmd -n "${TENANT_NAMESPACE}" get pods -o json)
  failed=$("${JQ}" '[.items[] | select(
      any(.status.containerStatuses[]?;
        ((.state.terminated.exitCode? // 0) != 0))
    )] | length' <<<"${pods}")
  pending=$("${JQ}" '[.items[] | select(
      .status.phase != "Running" and .status.phase != "Succeeded"
    )] | length' <<<"${pods}")
  unready=$("${JQ}" '[.items[] | select(
      .status.phase == "Running" and
      any(.status.containerStatuses[]?; .ready != true)
    )] | length' <<<"${pods}")
  [[ "${failed}" == 0 && "${pending}" == 0 &&
     "${unready}" == 0 ]] || {
    show_core_failure
    die "Kumo core has failed, pending, or unready pods"
  }

  if [[ "${PLATFORM}" == eks || "${PLATFORM}" == aks ]]; then
    kubectl_cmd -n "${TENANT_NAMESPACE}" wait \
      --for=jsonpath='{.status.loadBalancer.ingress[0]}' \
      "ingress/${TENANT_RELEASE}-edge" --timeout="${VERIFY_TIMEOUT}"
  elif [[ "${PLATFORM}" == aro ]]; then
    kubectl_cmd -n "${TENANT_NAMESPACE}" get \
      "route/${TENANT_RELEASE}-edge" >/dev/null
  else
    die "unsupported platform edge verification: ${PLATFORM}"
  fi
}
