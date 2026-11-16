#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

stage_platform_apply() {
  load_provider
  ensure_namespace "${PLATFORM_NAMESPACE}"
  ensure_namespace "${TENANT_NAMESPACE}"
  label_namespace "${TENANT_NAMESPACE}" true
  if [[ "${TEMPORAL_ENABLED}" == true ]]; then
    ensure_namespace "${TEMPORAL_NAMESPACE}"
    label_namespace "${TEMPORAL_NAMESPACE}" false
  fi
  provider_prepare_cloud

  helm_upgrade "kumo-platform-${PLATFORM}" "$(platform_chart)" \
    --namespace "${PLATFORM_NAMESPACE}" \
    --create-namespace \
    --atomic --wait --timeout "${TIMEOUT}" \
    -f "${WORK_DIR}/platform-values.yaml"

  if [[ "${PLATFORM}" == aro &&
        "${ARO_OPERATOR_MODE}" == dedicated ]]; then
    kubectl_cmd wait --for=create \
      crd/nodefeaturediscoveries.nfd.openshift.io --timeout="${TIMEOUT}"
    kubectl_cmd wait --for=create \
      crd/clusterpolicies.nvidia.com --timeout="${TIMEOUT}"
    kubectl_cmd wait --for=condition=Established \
      crd/nodefeaturediscoveries.nfd.openshift.io --timeout="${TIMEOUT}"
    kubectl_cmd wait --for=condition=Established \
      crd/clusterpolicies.nvidia.com --timeout="${TIMEOUT}"
    helm_upgrade "kumo-platform-${PLATFORM}" "$(platform_chart)" \
      --namespace "${PLATFORM_NAMESPACE}" \
      --atomic --wait --timeout "${TIMEOUT}" \
      -f "${WORK_DIR}/platform-values.yaml" \
      --set nfd.create=true \
      --set clusterPolicy.create=true
  fi
}

stage_platform_verify() {
  helm_release_is_deployed "kumo-platform-${PLATFORM}" \
    "${PLATFORM_NAMESPACE}"
  load_provider
  provider_verify_platform

  local storage_json
  storage_json=$(read_storage_class) ||
    die "cannot read StorageClass ${STORAGE_CLASS}"
  [[ -n "${storage_json}" ]] ||
    die "StorageClass ${STORAGE_CLASS} does not exist"
  verify_storage_class_contract "${storage_json}"
}
