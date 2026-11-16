#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ensure_helm_repo() {
  local name="$1" url="$2"
  "${HELM}" repo add "${name}" "${url}" --force-update >/dev/null
}

wait_for_deployment() {
  local namespace="$1" deployment="$2"
  kubectl_cmd -n "${namespace}" rollout status \
    "deployment/${deployment}" --timeout="${TIMEOUT}"
}

# Return the StorageClass JSON or empty on NotFound; propagate unreadable APIs.
read_storage_class() {
  kubectl_cmd get storageclass "${STORAGE_CLASS}" -o json --ignore-not-found
}

# Validate the scheduling, filesystem, and lifecycle contract independently of
# which CSI driver provides the class.
verify_storage_class_contract() {
  local storage_json="$1" provisioner driver_json
  "${JQ}" -e '
    .volumeBindingMode == "WaitForFirstConsumer" and
    .reclaimPolicy == "Delete" and
    (
      .parameters.fsType == "xfs" or
      .parameters["csi.storage.k8s.io/fstype"] == "xfs"
    )
  ' <<<"${storage_json}" >/dev/null ||
    die "StorageClass ${STORAGE_CLASS} must bind WaitForFirstConsumer, reclaim Delete, and format xfs"
  provisioner=$("${JQ}" -r '.provisioner // empty' <<<"${storage_json}")
  [[ -n "${provisioner}" ]] ||
    die "StorageClass ${STORAGE_CLASS} names no provisioner"
  verify_csi_driver "${provisioner}"
}

# Validate a class's driver, or a cloud driver before its class exists.
verify_csi_driver() {
  local provisioner="$1" driver_json
  driver_json=$(kubectl_cmd get csidriver "${provisioner}" \
    -o json --ignore-not-found) ||
    die "cannot read CSI driver ${provisioner}"
  [[ -n "${driver_json}" ]] ||
    die "StorageClass ${STORAGE_CLASS} needs CSI driver ${provisioner}, which is not registered"
  # `required` makes the pool controller wait for CSIStorageCapacity objects. A
  # driver that never publishes them turns that into a silent scheduling hang.
  [[ "${STORAGE_CAPACITY_TRACKING}" == required ]] || return 0
  "${JQ}" -e '.spec.storageCapacity == true' <<<"${driver_json}" >/dev/null ||
    die "storage.diskgraph.capacityTracking=required needs CSI driver ${provisioner} to publish storage capacity"
}

# Determine ownership from a deployed release's stored values, not mutable
# object annotations or rendered YAML. Return false for an absent/different
# class; unreadable or interrupted release state is fatal.
release_creates_storage_class() {
  local release="$1" status values creates
  # Catch failure outside the substitution so unreadable history is not absence.
  status=$(helm_release_status "${release}" "${PLATFORM_NAMESPACE}") ||
    die "cannot tell whether the ${release} release exists"
  [[ -n "${status}" ]] || return 1
  # Only a deployed release has a definite owned-resource set.
  [[ "${status}" == deployed ]] ||
    die "the ${release} release is ${status}, so which StorageClass it owns is indeterminate; settle that release before installing"
  values=$(helm_release_values "${release}" "${PLATFORM_NAMESPACE}") ||
    die "cannot read the ${release} release values"
  # Reject empty, malformed, or multiple values documents; jq accepts empty input.
  creates=0
  "${JQ}" -se --arg name "${STORAGE_CLASS}" '
    if length == 1 then
      .[0].storageClass.create == true and .[0].storageClass.name == $name
    else
      error("release values are not one document")
    end
  ' <<<"${values}" >/dev/null || creates=$?
  (( creates <= 1 )) ||
    die "cannot read the ${release} release values as a single document"
  return "${creates}"
}

verify_storage_class_ownership() {
  local release="$1" storage_json
  storage_json=$(read_storage_class) ||
    die "cannot read StorageClass ${STORAGE_CLASS}"
  if [[ -z "${storage_json}" ]]; then
    # An adopted class is an external prerequisite and must already exist.
    [[ "${STORAGE_CLASS_CREATE}" == true ]] ||
      die "adopted StorageClass ${STORAGE_CLASS} does not exist; create it or set storage.diskgraph.storageClass.create=true"
    return
  fi
  # Adopted classes must not remain owned or rendered by this release.
  if [[ "${STORAGE_CLASS_CREATE}" != true ]]; then
    "${JQ}" -e \
      --arg release "${release}" \
      --arg namespace "${PLATFORM_NAMESPACE}" '
        .metadata.annotations["meta.helm.sh/release-name"] != $release or
        .metadata.annotations["meta.helm.sh/release-namespace"] != $namespace
      ' <<<"${storage_json}" >/dev/null ||
      die "StorageClass ${STORAGE_CLASS} belongs to the ${release} release and would be removed by the next upgrade; adopt a class that release does not own"
    # Mutable annotations cannot make a release-rendered class safe to adopt.
    ! release_creates_storage_class "${release}" ||
      die "StorageClass ${STORAGE_CLASS} is created by the ${release} release and would be removed by the next upgrade; adopt a class that release does not own"
    verify_storage_class_contract "${storage_json}"
    return
  fi
  "${JQ}" -e \
    --arg release "${release}" \
    --arg namespace "${PLATFORM_NAMESPACE}" '
      .metadata.annotations["meta.helm.sh/release-name"] == $release and
      .metadata.annotations["meta.helm.sh/release-namespace"] == $namespace
    ' <<<"${storage_json}" >/dev/null ||
    die "StorageClass ${STORAGE_CLASS} already exists outside the ${release} release; adopt it with storage.diskgraph.storageClass.create=false"
  release_creates_storage_class "${release}" ||
    die "StorageClass ${STORAGE_CLASS} already exists but the ${release} release does not create it; adopt it with storage.diskgraph.storageClass.create=false"
}


# Nothing in this stack creates or labels the CPU capacity ARO runs Kumo on, so
# a selector no node carries can only surface later as core pods stuck Pending.
verify_cpu_node_selector() {
  local nodes
  [[ -n "${CPU_NODE_SELECTOR}" ]] || return 0
  nodes=$(kubectl_cmd get nodes -l "${CPU_NODE_SELECTOR// /,}" -o json) ||
    die "cannot list the nodes cluster.cpuNodeSelector selects"
  "${JQ}" -e '.items | length > 0' <<<"${nodes}" >/dev/null ||
    die "no node carries cluster.cpuNodeSelector ${CPU_NODE_SELECTOR}; label the CPU capacity Kumo runs on, or point the selector at labels those nodes already have"
}

# Reject selectors matched by known GPU labels. A pass cannot account for
# labels added after node startup.
verify_cpu_selector_excludes() {
  local description="$1" labels_json="$2" pair key value matched
  local -a pairs
  [[ -n "${CPU_NODE_SELECTOR}" ]] || return 0
  read -r -a pairs <<<"${CPU_NODE_SELECTOR}"
  for pair in "${pairs[@]}"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    matched=0
    "${JQ}" -e --arg key "${key}" --arg value "${value}" \
      '.[$key] == $value' <<<"${labels_json}" >/dev/null || matched=$?
    # jq status 1 means no match; larger statuses are evaluation failures.
    (( matched <= 1 )) ||
      die "cannot compare cluster.cpuNodeSelector against ${description}"
    (( matched == 0 )) || return 0
  done
  die "cluster.cpuNodeSelector ${CPU_NODE_SELECTOR} also selects ${description}; Kumo's CPU workloads would stay free to run on GPU capacity"
}

verify_no_gpu_nodes() {
  kubectl_cmd get nodes -l kumo.ai/gpu-node=true -o json |
    "${JQ}" -e '.items | length == 0' >/dev/null ||
    die "GPU nodes are running before a GPU workload was requested"
}

# A mistyped client ID is still a non-empty string: it renders, passes every
# check the stack makes on it, and only fails inside the bootstrap Job as a
# generic login error. Managed-identity client IDs are always UUIDs.
verify_azure_workload_identity_id() {
  local client_id="$1" label="$2"
  [[ "${client_id}" =~ ^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$ ]] ||
    die "${label} must be a managed-identity client ID (a UUID), not ${client_id}"
}

# The ServiceAccount the Temporal database Jobs run as. Cloud preflight proves
# the admin workload identity is federated with exactly this subject.
temporal_database_identity_subject() {
  printf 'system:serviceaccount:%s:%s' \
    "${TEMPORAL_NAMESPACE}" "${TEMPORAL_DATABASE_SERVICE_ACCOUNT}"
}

temporal_database_bootstrap_enabled() {
  [[ "${TEMPORAL_ENABLED}" == true && "${POSTGRES_BOOTSTRAP}" == true ]]
}

verify_azure_postgres() {
  local server_name postgres_json extensions server_version
  local connection_timeout="${AZURE_CORE_CONNECTION_TIMEOUT:-10}"
  local read_timeout="${AZURE_CORE_READ_TIMEOUT:-30}"
  server_name="${POSTGRES_HOST%%.*}"
  postgres_json=$(AZURE_CORE_CONNECTION_TIMEOUT="${connection_timeout}" \
    AZURE_CORE_READ_TIMEOUT="${read_timeout}" \
    az postgres flexible-server show \
    --resource-group "${POSTGRES_RESOURCE_GROUP}" \
    --name "${server_name}" --output json)
  "${JQ}" -e '
    .state == "Ready" and
    .authConfig.activeDirectoryAuth == "Enabled"
  ' <<<"${postgres_json}" >/dev/null ||
    die "Azure PostgreSQL must be Ready with Entra authentication enabled"

  [[ "${TEMPORAL_ENABLED}" == true ]] || return 0
  "${JQ}" -e '
    .authConfig.passwordAuth == "Enabled" and
    (.administratorLogin | type == "string" and length > 0)
  ' <<<"${postgres_json}" >/dev/null ||
    die "bundled Temporal requires Azure PostgreSQL password auth and a local administrator"
  extensions=$(AZURE_CORE_CONNECTION_TIMEOUT="${connection_timeout}" \
    AZURE_CORE_READ_TIMEOUT="${read_timeout}" \
    az postgres flexible-server parameter show \
    --resource-group "${POSTGRES_RESOURCE_GROUP}" \
    --server-name "${server_name}" \
    --name azure.extensions --query value --output tsv)
  tr ',' '\n' <<<"${extensions}" | grep -Fxiq btree_gin ||
    die "bundled Temporal requires btree_gin in Azure PostgreSQL azure.extensions"

  # Only the bootstrap needs 15: it reassigns each Temporal database's public
  # schema, which earlier releases allow the owning superuser alone. A
  # customer-managed Temporal database on 14 is still supported.
  temporal_database_bootstrap_enabled || return 0
  server_version=$("${JQ}" -r '.version // empty' <<<"${postgres_json}")
  if [[ ! "${server_version}" =~ ^[0-9]+$ ]] || (( server_version < 15 )); then
    die "cloud.postgres.bootstrapTemporal requires PostgreSQL 15 or newer; ${server_name} reports ${server_version:-no version}"
  fi
}
