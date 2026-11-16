#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

show_temporal_failure() {
  printf 'Temporal schema/server diagnostics:\n' >&2
  kubectl_cmd -n "${TEMPORAL_NAMESPACE}" get pods,jobs \
    -l app.kubernetes.io/instance="${TEMPORAL_RELEASE}" >&2 || true
  local pod container
  pod=$(kubectl_cmd -n "${TEMPORAL_NAMESPACE}" get pods \
    -l app.kubernetes.io/component=database \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true)
  if [[ -n "${pod}" ]]; then
    for container in \
      manage-schema-default-store manage-schema-visibility-store; do
      printf '\n%s/%s:\n' "${pod}" "${container}" >&2
      kubectl_cmd -n "${TEMPORAL_NAMESPACE}" logs "${pod}" \
        -c "${container}" --tail=120 >&2 || true
    done
  fi
}

stage_temporal_apply() {
  [[ "${TEMPORAL_ENABLED}" == true ]] || return 0
  temporal_profile_args
  if ! helm_upgrade "${TEMPORAL_RELEASE}" "${REPO_ROOT}/kumo-temporal" \
    --namespace "${TEMPORAL_NAMESPACE}" \
    --create-namespace \
    --atomic --wait --timeout "${TEMPORAL_TIMEOUT}" \
    "${TEMPORAL_PROFILE_ARGS[@]}"; then
    show_temporal_failure
    return 1
  fi
}

stage_temporal_verify() {
  [[ "${TEMPORAL_ENABLED}" == true ]] || return 0
  helm_release_is_deployed "${TEMPORAL_RELEASE}" "${TEMPORAL_NAMESPACE}"
  local component
  for component in frontend history matching worker; do
    kubectl_cmd -n "${TEMPORAL_NAMESPACE}" rollout status \
      "deployment/temporal-${component}" --timeout="${VERIFY_TIMEOUT}"
  done
  local endpoint
  endpoint=$(kubectl_cmd -n "${TEMPORAL_NAMESPACE}" get endpoints \
    temporal-frontend \
    -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || true)
  [[ -n "${endpoint}" ]] ||
    die "Temporal frontend Service has no ready endpoint"
}
