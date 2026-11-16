#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Define GPU labels once for both reconciliation and selector preflight.
aro_gpu_labels() {
  local tier="$1" tier_name instance_type ssd_gib gpu_count
  if [[ -z "${tier}" ]]; then
    "${JQ}" -nc '{
      "kumo.ai/gpu-node": "true",
      "kumo.ai/diskgraph-pool-type": "ephemeral",
      "cluster-api/accelerator": "nvidia-t4"
    }'
    return
  fi
  IFS=',' read -r tier_name instance_type ssd_gib gpu_count <<<"${tier}"
  "${JQ}" -nc --arg ssd "${ssd_gib}" --arg gpu "${gpu_count}" '{
    "kumo.ai/gpu-node": "true",
    "kumo.ai/diskgraph-pool-type": "cache",
    "kumo.ai/diskgraph-ssd-size": $ssd,
    "kumo.ai/diskgraph-gpu-count": $gpu,
    "cluster-api/accelerator": "nvidia-t4"
  }'
}

# Add invariant kubelet labels so preflight evaluates the labels a provisioned
# GPU node is known to carry, not only the MachineSet's explicit labels.
aro_gpu_node_labels() {
  local tier="$1" instance_type="$2"
  "${JQ}" -nc \
    --argjson stamped "$(aro_gpu_labels "${tier}")" \
    --arg instance_type "${instance_type}" '
      $stamped + {
        "kubernetes.io/os": "linux",
        "kubernetes.io/arch": "amd64",
        "node.openshift.io/os_id": "rhcos",
        "node.kubernetes.io/instance-type": $instance_type
      }
    '
}

# Reject CPU selectors matched by adopted GPU MachineSets. This checks only
# configured MachineSets and labels known before node startup.
verify_aro_gpu_exclusion() {
  local machine_set tier machine_set_json instance_type stamped labels index=0
  local -a machine_sets tiers
  [[ -n "${CPU_NODE_SELECTOR}" ]] || return 0
  IFS=',' read -r -a machine_sets <<<"${ARO_CACHE_MACHINESETS}"
  IFS=';' read -r -a tiers <<<"${GPU_CACHE_TIERS}"
  for machine_set in "${machine_sets[@]}" "${ARO_EPHEMERAL_MACHINESET}"; do
    # One tier per cache MachineSet; the trailing entry is the ephemeral pool.
    tier=""
    if (( index < ${#machine_sets[@]} )); then
      tier="${tiers[$index]}"
    fi
    machine_set_json=$(kubectl_cmd -n openshift-machine-api get machineset \
      "${machine_set}" -o json) ||
      die "cannot read MachineSet ${machine_set}"
    instance_type=$("${JQ}" -r \
      '.spec.template.spec.providerSpec.value.vmSize // empty' \
      <<<"${machine_set_json}")
    stamped=$(aro_gpu_node_labels "${tier}" "${instance_type}")
    labels=$("${JQ}" -c --argjson stamped "${stamped}" \
      '(.spec.template.spec.metadata.labels // {}) + $stamped' \
      <<<"${machine_set_json}")
    verify_cpu_selector_excludes "MachineSet ${machine_set}" "${labels}"
    index=$((index + 1))
  done
}

provider_preflight() {
  command -v az >/dev/null
  command -v jq >/dev/null
  local cluster_version
  cluster_version=$(kubectl_cmd get clusterversion version -o json)
  "${JQ}" -e '
    (.status.desired.version | split(".")[1] | tonumber) >= 20
  ' <<<"${cluster_version}" >/dev/null ||
    die "ARO requires OpenShift 4.20"
  # Only a class this release creates is Azure Disk-backed. An adopted one
  # proves its own driver in verify_storage_class_ownership.
  if [[ "${STORAGE_CLASS_CREATE}" == true ]]; then
    verify_csi_driver disk.csi.azure.com
  fi
  kubectl_cmd get clusterautoscaler default >/dev/null
  kubectl_cmd -n openshift-cloud-credential-operator \
    get deployment pod-identity-webhook -o json |
    "${JQ}" -e '
      (.status.availableReplicas // 0) > 0
    ' >/dev/null ||
    die "ARO requires an available pod-identity-webhook replica"

  verify_storage_class_ownership kumo-platform-aro
  verify_cpu_node_selector
  verify_aro_gpu_exclusion
  verify_azure_postgres
  if temporal_database_bootstrap_enabled; then
    # ARO cannot inspect workload-identity federation through the installer login;
    # validate its shape here and let the Job prove authentication.
    verify_azure_workload_identity_id "${POSTGRES_ADMIN_ANNOTATION_VALUE}" \
      "cloud.postgres.admin.annotations[${POSTGRES_ADMIN_ANNOTATION_KEY}]"
  fi
  if [[ "${ARO_OPERATOR_MODE}" == existing ]]; then
    kubectl_cmd get crd nodefeaturediscoveries.nfd.openshift.io >/dev/null
    kubectl_cmd get crd clusterpolicies.nvidia.com >/dev/null
    kubectl_cmd get nodefeaturediscovery -A -o json |
      "${JQ}" -e '.items | length > 0' >/dev/null ||
      die "existing ARO operator mode requires a NodeFeatureDiscovery"
    kubectl_cmd get clusterpolicy -o json |
      "${JQ}" -e '.items | length > 0' >/dev/null ||
      die "existing ARO operator mode requires a ClusterPolicy"
  fi
}

provider_prepare_cloud() {
  local network_json
  network_json=$(kubectl_cmd get network.operator.openshift.io cluster -o json)
  if ! "${JQ}" -e '
    .spec.defaultNetwork.ovnKubernetesConfig.gatewayConfig.routingViaHost == true
  ' <<<"${network_json}" >/dev/null; then
    kubectl_cmd patch network.operator.openshift.io cluster \
      --type merge \
      -p '{"spec":{"defaultNetwork":{"ovnKubernetesConfig":{"gatewayConfig":{"routingViaHost":true}}}}}' \
      >/dev/null
  fi

  # MachineSets are intentionally explicit inputs. They retain the target
  # cluster's provider spec, zones, subnet, image, and credentials; this module
  # owns their autoscaling and GPU/operator contract without guessing Azure
  # infrastructure fields.
  local machine_set tier tier_name instance_type ssd_gib gpu_count index=0
  local actual_type patch
  local -a cache_machine_sets tiers
  IFS=',' read -r -a cache_machine_sets <<<"${ARO_CACHE_MACHINESETS}"
  IFS=';' read -r -a tiers <<<"${GPU_CACHE_TIERS}"
  for machine_set in "${cache_machine_sets[@]}"; do
    kubectl_cmd -n openshift-machine-api get machineset \
      "${machine_set}" >/dev/null
    tier="${tiers[$index]}"
    IFS=',' read -r tier_name instance_type ssd_gib gpu_count <<<"${tier}"
    actual_type=$(kubectl_cmd -n openshift-machine-api get machineset \
      "${machine_set}" -o jsonpath='{.spec.template.spec.providerSpec.value.vmSize}')
    if [[ "${actual_type}" != "${instance_type}" ]]; then
      echo "MachineSet ${machine_set} uses ${actual_type}, want ${instance_type}" >&2
      return 1
    fi
    patch=$("${JQ}" -nc --argjson labels "$(aro_gpu_labels "${tier}")" \
      '{spec:{template:{spec:{metadata:{labels:$labels}}}}}')
    kubectl_cmd -n openshift-machine-api patch machineset \
      "${machine_set}" --type merge -p "${patch}" >/dev/null
    index=$((index + 1))
  done
  kubectl_cmd -n openshift-machine-api get machineset \
    "${ARO_EPHEMERAL_MACHINESET}" >/dev/null
  actual_type=$(kubectl_cmd -n openshift-machine-api get machineset \
    "${ARO_EPHEMERAL_MACHINESET}" \
    -o jsonpath='{.spec.template.spec.providerSpec.value.vmSize}')
  if [[ "${actual_type}" != "${GPU_EPHEMERAL_INSTANCE_TYPE}" ]]; then
    echo "MachineSet ${ARO_EPHEMERAL_MACHINESET} uses ${actual_type}, want ${GPU_EPHEMERAL_INSTANCE_TYPE}" >&2
    return 1
  fi
  patch=$("${JQ}" -nc --argjson labels "$(aro_gpu_labels "")" \
    '{spec:{template:{spec:{metadata:{labels:$labels}}}}}')
  kubectl_cmd -n openshift-machine-api patch machineset \
    "${ARO_EPHEMERAL_MACHINESET}" --type merge -p "${patch}" >/dev/null
}

provider_verify_platform() {
  local machine_set
  local -a cache_machine_sets
  IFS=',' read -r -a cache_machine_sets <<<"${ARO_CACHE_MACHINESETS}"
  for machine_set in \
    "${cache_machine_sets[@]}" "${ARO_EPHEMERAL_MACHINESET}"; do
    kubectl_cmd -n openshift-machine-api get machineautoscaler \
      "${machine_set}" -o json |
      "${JQ}" -e '.spec.minReplicas == 0' >/dev/null ||
      die "MachineAutoscaler ${machine_set} does not scale to zero"
  done
  if [[ "${ARO_OPERATOR_MODE}" == existing ]]; then
    kubectl_cmd get nodefeaturediscovery -A -o json |
      "${JQ}" -e '.items | length > 0' >/dev/null ||
      die "existing ARO operator mode has no NodeFeatureDiscovery"
    kubectl_cmd get clusterpolicy -o json |
      "${JQ}" -e '.items | length > 0' >/dev/null ||
      die "existing ARO operator mode has no ClusterPolicy"
  fi
  verify_no_gpu_nodes
}

provider_decommission() {
  local machine_set
  local -a cache_machine_sets
  IFS=',' read -r -a cache_machine_sets <<<"${ARO_CACHE_MACHINESETS}"
  for machine_set in \
    "${cache_machine_sets[@]}" "${ARO_EPHEMERAL_MACHINESET}"; do
    kubectl_cmd -n openshift-machine-api scale machineset \
      "${machine_set}" --replicas=0
  done
}
