#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

wait_for_gpu_scale_zero() {
  local deadline count
  deadline=$((SECONDS + $(duration_seconds "${GPU_VERIFY_TIMEOUT}")))
  while (( SECONDS < deadline )); do
    count=$(kubectl_cmd get nodes -l kumo.ai/gpu-node=true -o json |
      "${JQ}" '.items | length')
    [[ "${count}" == 0 ]] && return 0
    sleep 10
  done
  return 1
}

verify_canary_storage_deleted() {
  local count
  count=$(kubectl_cmd -n "${TENANT_NAMESPACE}" get pods,pvc \
    -l app=dg-pool-canary -o json |
    "${JQ}" '.items | length')
  [[ "${count}" == 0 ]] ||
    die "DiskGraph GPU canary pods or PVCs remain in ${TENANT_NAMESPACE}"

  count=$(kubectl_cmd get pv -o json |
    "${JQ}" --arg namespace "${TENANT_NAMESPACE}" '
      [.items[] | select(
        .spec.claimRef.namespace == $namespace and
        (.spec.claimRef.name // "" |
          startswith("dg-pool-canary-pvc-"))
      )] | length
    ')
  [[ "${count}" == 0 ]] ||
    die "DiskGraph GPU canary PVs have not been deleted"
}

# Best-effort EKS capacity diagnostic scoped to Kumo GPU claims. It must never
# abort the retry it annotates.
gpu_capacity_shortfall() {
  [[ "${PLATFORM}" == eks ]] || return 0
  kubectl_cmd get nodeclaims -l kumo.ai/gpu-node=true -o json 2>/dev/null |
    "${JQ}" -r '
      [.items[]
        | select(.status.conditions[]?
            | select(.type == "Launched" and .status == "False"))
        | "\(.metadata.labels["karpenter.sh/nodepool"] // "?"): \(
            [.status.conditions[] | select(.type == "Launched")][0].message // "")"]
      | unique | join("; ")' 2>/dev/null || return 0
}

# Run one canary sweep in this shell because helm_upgrade mutates shell state.
gpu_canary_sweep() {
  helm_upgrade "${TENANT_RELEASE}" "${REPO_ROOT}/kumo-helm" \
    --namespace "${TENANT_NAMESPACE}" \
    --atomic --wait --timeout "$1" \
    "${CORE_PROFILE_ARGS[@]}" \
    --set-string \
      workloads.poolcontroller.extraEnv.POOL_INIT_CANARY_ENABLED=true \
    --set-string \
      workloads.poolcontroller.extraEnv.POOL_CANARY_RESULT_TTL_HOURS=0
}

# The pools the controller recorded as validated; empty until every one of them passed.
gpu_canary_validation() {
  kubectl_cmd -n "${TENANT_NAMESPACE}" get deployment kumo-poolcontroller \
    -o jsonpath='{.metadata.annotations.kumo\.ai/diskgraph-pools-validated}'
}

stage_gpu_lifecycle_verify() {
  stage_core_verify
  core_profile_args

  # Fail closed so validation from an earlier run cannot satisfy this gate.
  kubectl_cmd -n "${TENANT_NAMESPACE}" annotate deployment \
    kumo-poolcontroller kumo.ai/diskgraph-pools-validated- >/dev/null ||
    die "cannot clear the previous DiskGraph validation annotation on kumo-poolcontroller in ${TENANT_NAMESPACE}; this run's canary result would be indistinguishable from an earlier install's"

  # Retry transient GPU capacity failures with bounded attempts and backoff.
  local deadline delay max_delay attempt_cap remaining attempt validation shortfall
  deadline=$((SECONDS + $(duration_seconds "${GPU_VERIFY_TIMEOUT}")))
  delay=$(duration_seconds "${GPU_CANARY_RETRY_DELAY}")
  max_delay=$(duration_seconds "${GPU_CANARY_RETRY_MAX_DELAY}")
  attempt_cap=$(duration_seconds "${GPU_CANARY_ATTEMPT_TIMEOUT}")
  attempt=0
  validation=""

  while :; do
    attempt=$((attempt + 1))
    remaining=$((deadline - SECONDS))
    (( remaining > 0 )) || break
    (( remaining < attempt_cap )) && attempt_cap="${remaining}"

    validation=""
    if gpu_canary_sweep "${attempt_cap}s"; then
      # An unreadable annotation is another failed attempt.
      validation=$(gpu_canary_validation) || validation=""
      [[ -n "${validation}" ]] && break
    fi

    remaining=$((deadline - SECONDS))
    (( remaining > delay )) || break

    shortfall=$(gpu_capacity_shortfall)
    printf 'GPU canary attempt %d did not validate every pool%s; retrying in %ds (%ds of %s budget left).\n' \
      "${attempt}" \
      "${shortfall:+ (${shortfall})}" \
      "${delay}" "${remaining}" "${GPU_VERIFY_TIMEOUT}" >&2
    sleep "${delay}"
    delay=$(( delay * 2 ))
    (( delay > max_delay )) && delay="${max_delay}"
  done

  # Restore normal mode while retaining successful validation evidence.
  stage_core_apply

  if [[ -z "${validation}" ]]; then
    shortfall=$(gpu_capacity_shortfall)
    kubectl_cmd -n "${TENANT_NAMESPACE}" logs \
      deployment/kumo-poolcontroller --tail=200 >&2 || true
    show_core_failure
    die "the GPU canary did not record validation of every cache and ephemeral pool after ${attempt} attempt(s) within ${GPU_VERIFY_TIMEOUT}${shortfall:+ — last capacity error: ${shortfall}}"
  fi
  verify_canary_storage_deleted
  wait_for_gpu_scale_zero ||
    die "GPU nodes did not scale back to zero within ${GPU_VERIFY_TIMEOUT}"
  printf 'GPU cache tiers, ephemeral pool, device plugin, PVC deletion, and scale-to-zero lifecycle passed.\n'
}
