#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

aks_pool_exists() {
  az aks nodepool show \
    --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --cluster-name "${PROVIDER_CLUSTER}" \
    --name "$1" >/dev/null 2>&1
}

verify_aks_application_gateway() {
  local gateway_json
  gateway_json=$(az network application-gateway show \
    --ids "${AKS_APPLICATION_GATEWAY_ID}" --output json) ||
    die "Azure Application Gateway ${AKS_APPLICATION_GATEWAY_ID} was not found"
  "${JQ}" -e '
    .provisioningState == "Succeeded" and
    .operationalState == "Running" and
    (.sku.tier == "Standard_v2" or .sku.tier == "WAF_v2") and
    any(.frontendIPConfigurations[]?;
      (.privateIPAddress // "") != "")
  ' <<<"${gateway_json}" >/dev/null ||
    die "AKS ingress requires a running Standard_v2 or WAF_v2 Application Gateway with a private frontend IP"
}

verify_aks_application_gateway_addon() {
  local aks_json
  aks_json=$(az aks show \
    --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --name "${PROVIDER_CLUSTER}" --output json)
  "${JQ}" -e --arg id "${AKS_APPLICATION_GATEWAY_ID}" '
    .addonProfiles.ingressApplicationGateway.enabled == true and
    (
      .addonProfiles.ingressApplicationGateway.config.effectiveApplicationGatewayId
      | ascii_downcase
    ) == ($id | ascii_downcase)
  ' <<<"${aks_json}" >/dev/null
}

verify_aks_acr_access() {
  [[ -n "${KUMO_PULL_SECRET}" ]] && return
  local registry="${KUMO_CPU_IMAGE%%/*}" result
  result=$(az aks check-acr \
    --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --name "${PROVIDER_CLUSTER}" \
    --acr "${registry}" 2>&1)
  if [[ "${result}" != *"Validating image pull permission: SUCCEEDED"* ]]; then
    printf '%s\n' "${result}" >&2
    die "AKS kubelet cannot pull images from ${registry}"
  fi
}

ensure_aks_application_gateway_addon() {
  local aks_json gateway_json current_id expected_id principal_id subnet_id
  aks_json=$(az aks show \
    --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --name "${PROVIDER_CLUSTER}" --output json)
  if "${JQ}" -e '
    .addonProfiles.ingressApplicationGateway.enabled == true
  ' <<<"${aks_json}" >/dev/null; then
    current_id=$("${JQ}" -r '
      .addonProfiles.ingressApplicationGateway.config.effectiveApplicationGatewayId
      // ""
    ' <<<"${aks_json}")
    current_id=$(tr '[:upper:]' '[:lower:]' <<<"${current_id}")
    expected_id=$(tr '[:upper:]' '[:lower:]' \
      <<<"${AKS_APPLICATION_GATEWAY_ID}")
    if [[ "${current_id}" != "${expected_id}" ]]; then
      die "AKS AGIC add-on already targets ${current_id}; expected ${AKS_APPLICATION_GATEWAY_ID}"
    fi
  else
    az aks enable-addons \
      --resource-group "${PROVIDER_RESOURCE_GROUP}" \
      --name "${PROVIDER_CLUSTER}" \
      --addons ingress-appgw \
      --appgw-id "${AKS_APPLICATION_GATEWAY_ID}" >/dev/null
    aks_json=$(az aks show \
      --resource-group "${PROVIDER_RESOURCE_GROUP}" \
      --name "${PROVIDER_CLUSTER}" --output json)
  fi
  principal_id=$("${JQ}" -r '
    .addonProfiles.ingressApplicationGateway.identity.objectId // empty
  ' <<<"${aks_json}")
  [[ -n "${principal_id}" ]] ||
    die "AKS managed AGIC add-on has no managed identity"
  gateway_json=$(az network application-gateway show \
    --ids "${AKS_APPLICATION_GATEWAY_ID}" --output json)
  subnet_id=$("${JQ}" -r '
    .gatewayIPConfigurations[0].subnet.id // empty
  ' <<<"${gateway_json}")
  [[ -n "${subnet_id}" ]] ||
    die "Azure Application Gateway ${AKS_APPLICATION_GATEWAY_ID} has no subnet"
  az role assignment create \
    --assignee-object-id "${principal_id}" \
    --assignee-principal-type ServicePrincipal \
    --role Contributor \
    --scope "${AKS_APPLICATION_GATEWAY_ID}" >/dev/null
  az role assignment create \
    --assignee-object-id "${principal_id}" \
    --assignee-principal-type ServicePrincipal \
    --role "Network Contributor" \
    --scope "${subnet_id}" >/dev/null
}

ensure_aks_edge_tls_secret() {
  [[ -r "${EDGE_TLS_CERTIFICATE_FILE}" ]] ||
    die "AKS edge certificate file is not readable: ${EDGE_TLS_CERTIFICATE_FILE}"
  [[ -r "${EDGE_TLS_PRIVATE_KEY_FILE}" ]] ||
    die "AKS edge private-key file is not readable: ${EDGE_TLS_PRIVATE_KEY_FILE}"
  kubectl_cmd -n "${TENANT_NAMESPACE}" create secret tls \
    "${EDGE_TLS_SECRET}" \
    "--cert=${EDGE_TLS_CERTIFICATE_FILE}" \
    "--key=${EDGE_TLS_PRIVATE_KEY_FILE}" \
    --dry-run=client -o yaml |
    kubectl_cmd apply -f - >/dev/null
}

aks_pool_matches() {
  local pool_json="$1" size="$2" min_count="$3" max_count="$4"
  local labels="$5" gpu="$6"
  "${JQ}" -e \
    --arg size "${size}" \
    --argjson min "${min_count}" \
    --argjson max "${max_count}" \
    --arg labels "${labels}" \
    --argjson gpu "${gpu}" '
      .vmSize == $size and
      .enableAutoScaling == true and
      .minCount == $min and
      .maxCount == $max and
      .scaleDownMode == "Delete" and
      (if $gpu then
        (.gpuProfile.nvidia.managementMode // "") != "Managed"
      else true end) and
      .nodeLabels == (
        $labels | split(" ") | map(
          split("=") | {key: .[0], value: .[1]}
        ) | from_entries
      )
    ' <<<"${pool_json}" >/dev/null
}

ensure_aks_pool() {
  local name="$1"
  local size="$2"
  local min_count="$3"
  local max_count="$4"
  local labels="$5"
  local gpu="$6"

  local -a autoscaler_args label_args
  read -r -a label_args <<<"${labels}"

  local pool_json
  if pool_json=$(az aks nodepool show \
    --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --cluster-name "${PROVIDER_CLUSTER}" \
    --name "${name}" --output json 2>/dev/null); then
    local actual_size autoscaling_enabled gpu_management_mode
    actual_size=$("${JQ}" -r '.vmSize' <<<"${pool_json}")
    if [[ "${actual_size}" != "${size}" ]]; then
      echo "AKS pool ${name} uses ${actual_size}, want ${size}; recreate it" >&2
      return 1
    fi
    if [[ "${gpu}" == true ]]; then
      gpu_management_mode=$("${JQ}" -r \
        '.gpuProfile.nvidia.managementMode // empty' <<<"${pool_json}")
      if [[ "${gpu_management_mode}" == Managed ]]; then
        echo "AKS pool ${name} uses the fully managed GPU profile, which does not support cluster autoscaling; recreate it with the driver-only profile" >&2
        return 1
      fi
    fi
    autoscaling_enabled=$("${JQ}" -r '.enableAutoScaling' <<<"${pool_json}")
    if aks_pool_matches "${pool_json}" "${size}" "${min_count}" \
      "${max_count}" "${labels}" "${gpu}"; then
      return
    fi
    if [[ "${autoscaling_enabled}" == true ]]; then
      autoscaler_args=(--update-cluster-autoscaler)
    else
      autoscaler_args=(--enable-cluster-autoscaler)
    fi
    az aks nodepool update \
      --resource-group "${PROVIDER_RESOURCE_GROUP}" \
      --cluster-name "${PROVIDER_CLUSTER}" \
      --name "${name}" \
      "${autoscaler_args[@]}" \
      --min-count "${min_count}" \
      --max-count "${max_count}" \
      --scale-down-mode Delete \
      --labels "${label_args[@]}" >/dev/null
    return
  fi

  local args=(
    aks nodepool add
    --resource-group "${PROVIDER_RESOURCE_GROUP}"
    --cluster-name "${PROVIDER_CLUSTER}"
    --name "${name}"
    --mode User
    --node-vm-size "${size}"
    --os-sku AzureLinux3
    --enable-cluster-autoscaler
    --min-count "${min_count}"
    --max-count "${max_count}"
    --node-count "${min_count}"
    --scale-down-mode Delete
    --labels "${label_args[@]}"
  )
  # AKS installs the NVIDIA driver by default. The Helm leaf supplies the stock
  # device plugin, so this path does not use the preview managed-GPU profile.
  az "${args[@]}" >/dev/null
}

add_aks_pool_quota_requirement() {
  local requirements="$1" skus="$2" pools="$3"
  local instance_type="$4" pool_name="$5" max_count="$6"
  local sku family vcpus current_count desired
  sku=$("${JQ}" -c --arg name "${instance_type}" '
    [.[] | select(
      .name == $name and .resourceType == "virtualMachines"
    )][0] // null
  ' <<<"${skus}")
  [[ "${sku}" != null ]] ||
    die "Azure VM SKU ${instance_type} is not offered in the AKS region"
  "${JQ}" -e '
    all(.restrictions[]?;
      .reasonCode != "NotAvailableForSubscription")
  ' <<<"${sku}" >/dev/null ||
    die "Azure VM SKU ${instance_type} is restricted for this subscription/region"
  family=$("${JQ}" -r '.family' <<<"${sku}")
  vcpus=$("${JQ}" -r '
    [.capabilities[] | select(.name == "vCPUs")][0].value // empty
  ' <<<"${sku}")
  [[ "${vcpus}" =~ ^[0-9]+$ ]] ||
    die "Azure VM SKU ${instance_type} does not report a vCPU count"
  current_count=$("${JQ}" -r \
    --arg name "${pool_name}" --arg size "${instance_type}" '
    [.[] | select(.name == $name and .vmSize == $size)][0].count // 0
  ' <<<"${pools}")
  desired=$((vcpus * max_count))
  "${JQ}" -c \
    --arg family "${family}" \
    --argjson desired "${desired}" \
    --argjson current "$((vcpus * current_count))" '
      .[$family].desired = ((.[$family].desired // 0) + $desired) |
      .[$family].current = ((.[$family].current // 0) + $current)
    ' <<<"${requirements}"
}

verify_aks_pool_quota() {
  local location="$1" skus usage pools requirements='{}'
  local tier tier_name instance_type ssd_gib gpu_count pool_name index=0
  local family desired stack_current quota current base
  local -a tiers pool_names

  # One regional SKU inventory covers every configured tier. This is slower
  # than guessing family names from SKU strings, but remains correct for
  # customer-selected Azure sizes.
  skus=$(az vm list-skus --location "${location}" --all --output json)
  usage=$(az vm list-usage --location "${location}" --output json)
  pools=$(az aks nodepool list \
    --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --cluster-name "${PROVIDER_CLUSTER}" --output json)

  requirements=$(add_aks_pool_quota_requirement \
    "${requirements}" "${skus}" "${pools}" \
    "${AKS_CPU_POOL_INSTANCE_TYPE}" "${AKS_CPU_POOL_NAME}" 3)

  IFS=';' read -r -a tiers <<<"${GPU_CACHE_TIERS}"
  IFS=',' read -r -a pool_names <<<"${AKS_CACHE_POOL_NAMES}"
  for tier in "${tiers[@]}"; do
    IFS=',' read -r tier_name instance_type ssd_gib gpu_count <<<"${tier}"
    pool_name="${pool_names[$index]}"
    requirements=$(add_aks_pool_quota_requirement \
      "${requirements}" "${skus}" "${pools}" \
      "${instance_type}" "${pool_name}" "${GPU_MAX_NODES}")
    index=$((index + 1))
  done
  requirements=$(add_aks_pool_quota_requirement \
    "${requirements}" "${skus}" "${pools}" \
    "${GPU_EPHEMERAL_INSTANCE_TYPE}" "${AKS_EPHEMERAL_POOL_NAME}" \
    "${GPU_MAX_NODES}")

  while IFS=$'\t' read -r family desired stack_current; do
    quota=$("${JQ}" -c --arg family "${family}" '
      [.[] | select(
        .name.value == $family or
        .name.localizedValue == $family or
        (.name.localizedValue | startswith($family + " "))
      )][0] // null
    ' <<<"${usage}")
    [[ "${quota}" != null ]] ||
      die "Azure quota ${family} is not visible in ${location}"
    current=$("${JQ}" -r '.currentValue' <<<"${quota}")
    quota=$("${JQ}" -r '.limit' <<<"${quota}")
    base=$((current - stack_current))
    (( base < 0 )) && base=0
    (( base + desired <= quota )) ||
      die "Azure ${family} quota in ${location} is ${quota} vCPUs: non-stack use is ${base}, but configured Kumo pool maxima need ${desired}; raise the quota to at least $((base + desired))"
  done < <("${JQ}" -r '
    to_entries[] |
    [.key, (.value.desired | tostring), (.value.current | tostring)] |
    @tsv
  ' <<<"${requirements}")
}

provider_preflight() {
  command -v az >/dev/null
  local aks_json
  local identities storage_accounts storage_account storage_id
  aks_json=$(az aks show \
    --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --name "${PROVIDER_CLUSTER}" --output json)
  "${JQ}" -e '
    .provisioningState == "Succeeded" and
    .oidcIssuerProfile.enabled == true and
    .securityProfile.workloadIdentity.enabled == true and
    .networkProfile.networkPlugin == "azure" and
    ((.kubernetesVersion | split(".")[1] | tonumber) >= 33)
  ' <<<"${aks_json}" >/dev/null ||
    die "AKS must be Succeeded on Kubernetes 1.33+ with Azure CNI, OIDC, and workload identity enabled"
  # Only a class this release creates is Azure Disk-backed. An adopted one
  # proves its own driver in verify_storage_class_ownership.
  if [[ "${STORAGE_CLASS_CREATE}" == true ]]; then
    verify_csi_driver disk.csi.azure.com
  fi
  verify_aks_acr_access

  verify_storage_class_ownership kumo-platform-aks
  verify_azure_postgres

  storage_account="${OBJECT_STORAGE_URI#*@}"
  storage_account="${storage_account%%.*}"
  storage_accounts=$(az storage account list --output json)
  storage_id=$("${JQ}" -r --arg name "${storage_account}" \
    '.[] | select(.name == $name) | .id' <<<"${storage_accounts}")
  [[ -n "${storage_id}" ]] ||
    die "ADLS account ${storage_account} was not found"

  identities=$(az identity list --output json)
  verify_aks_identity "${STATE_CONTROL_CLIENT_ID}" \
    "system:serviceaccount:${TENANT_NAMESPACE}:kumo-control" \
    "${identities}" "${storage_id}"
  verify_aks_identity "${STATE_WORKER_CLIENT_ID}" \
    "system:serviceaccount:${TENANT_NAMESPACE}:kumo-worker" \
    "${identities}" "${storage_id}"
  verify_aks_identity "${STATE_CONTROLLER_CLIENT_ID}" \
    "system:serviceaccount:${TENANT_NAMESPACE}:kumo-controller" \
    "${identities}" "${storage_id}"
  if temporal_database_bootstrap_enabled; then
    # The Temporal bootstrap Job logs in as the PostgreSQL administrator with
    # this identity. It reaches PostgreSQL alone, so it is proved against its
    # federated ServiceAccount and deliberately against no storage scope.
    verify_azure_workload_identity_id "${POSTGRES_ADMIN_ANNOTATION_VALUE}" \
      "cloud.postgres.admin.annotations[${POSTGRES_ADMIN_ANNOTATION_KEY}]"
    verify_aks_identity "${POSTGRES_ADMIN_ANNOTATION_VALUE}" \
      "$(temporal_database_identity_subject)" "${identities}" ""
  fi
  verify_aks_pool_quota "$("${JQ}" -r '.location' <<<"${aks_json}")"
  verify_aks_application_gateway
}

# An empty storage scope means the identity is not expected to reach ADLS at
# all; every caller states the scope it wants proved.
verify_aks_identity() {
  local client_id="$1" subject="$2" identities="$3" storage_id="$4"
  local identity name resource_group principal_id credentials assignments
  identity=$("${JQ}" -c --arg client_id "${client_id}" \
    '.[] | select(.clientId == $client_id)' <<<"${identities}")
  [[ -n "${identity}" ]] ||
    die "Azure managed identity ${client_id} was not found"
  name=$("${JQ}" -r '.name' <<<"${identity}")
  resource_group=$("${JQ}" -r '.resourceGroup' <<<"${identity}")
  principal_id=$("${JQ}" -r '.principalId' <<<"${identity}")

  credentials=$(az identity federated-credential list \
    --identity-name "${name}" --resource-group "${resource_group}" \
    --output json)
  "${JQ}" -e --arg subject "${subject}" '
    any(.[]; .subject == $subject and
      (.audiences | index("api://AzureADTokenExchange") != null))
  ' <<<"${credentials}" >/dev/null ||
    die "Azure identity ${name} has no federated credential for ${subject}"

  [[ -n "${storage_id}" ]] || return 0
  assignments=$(az role assignment list \
    --assignee-object-id "${principal_id}" \
    --scope "${storage_id}" --output json)
  "${JQ}" -e '
    any(.[]; .roleDefinitionName == "Storage Blob Data Contributor" or
      .roleDefinitionName == "Storage Blob Data Owner")
  ' <<<"${assignments}" >/dev/null ||
    die "Azure identity ${name} lacks ADLS data access on ${storage_id}"
}

provider_prepare_cloud() {
  ensure_aks_application_gateway_addon
  if [[ $(az aks show --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --name "${PROVIDER_CLUSTER}" \
    --query autoScalerProfile.skipNodesWithSystemPods --output tsv) != false ]]; then
    az aks update --resource-group "${PROVIDER_RESOURCE_GROUP}" \
      --name "${PROVIDER_CLUSTER}" \
      --cluster-autoscaler-profile skip-nodes-with-system-pods=false \
      >/dev/null
  fi
  ensure_aks_edge_tls_secret

  # The CPU pool is the capacity core Kumo is targeted at, so it carries exactly
  # the selector core is rendered with.
  ensure_aks_pool "${AKS_CPU_POOL_NAME}" "${AKS_CPU_POOL_INSTANCE_TYPE}" 1 3 \
    "${CPU_NODE_SELECTOR}" false

  local tier pool_name tier_name instance_type ssd_gib gpu_count index=0
  local -a tiers pool_names
  IFS=';' read -r -a tiers <<<"${GPU_CACHE_TIERS}"
  IFS=',' read -r -a pool_names <<<"${AKS_CACHE_POOL_NAMES}"
  for tier in "${tiers[@]}"; do
    IFS=',' read -r tier_name instance_type ssd_gib gpu_count <<<"${tier}"
    pool_name="${pool_names[$index]}"
    ensure_aks_pool "${pool_name}" "${instance_type}" 0 "${GPU_MAX_NODES}" \
      "kumo.ai/gpu-node=true kumo.ai/diskgraph-pool-type=cache kumo.ai/diskgraph-ssd-size=${ssd_gib} kumo.ai/diskgraph-gpu-count=${gpu_count}" true
    index=$((index + 1))
  done

  ensure_aks_pool "${AKS_EPHEMERAL_POOL_NAME}" \
    "${GPU_EPHEMERAL_INSTANCE_TYPE}" 0 "${GPU_MAX_NODES}" \
    "kumo.ai/gpu-node=true kumo.ai/diskgraph-pool-type=ephemeral" true
}

verify_aks_pool() {
  local name="$1" size="$2" min_count="$3" max_count="$4"
  local labels="$5" gpu="$6" pool_json
  pool_json=$(az aks nodepool show \
    --resource-group "${PROVIDER_RESOURCE_GROUP}" \
    --cluster-name "${PROVIDER_CLUSTER}" \
    --name "${name}" --output json) ||
    die "AKS pool ${name} does not exist"
  aks_pool_matches "${pool_json}" "${size}" "${min_count}" \
    "${max_count}" "${labels}" "${gpu}" ||
    die "AKS pool ${name} does not match the rendered size, labels, or autoscaler contract"
}

provider_verify_platform() {
  verify_aks_application_gateway
  verify_aks_application_gateway_addon ||
    die "AKS managed AGIC add-on is not targeting ${AKS_APPLICATION_GATEWAY_ID}"
  kubectl_cmd get ingressclass azure-application-gateway >/dev/null

  verify_aks_pool "${AKS_CPU_POOL_NAME}" "${AKS_CPU_POOL_INSTANCE_TYPE}" 1 3 \
    "${CPU_NODE_SELECTOR}" false

  local tier pool_name tier_name instance_type ssd_gib gpu_count index=0
  local -a tiers pool_names
  IFS=';' read -r -a tiers <<<"${GPU_CACHE_TIERS}"
  IFS=',' read -r -a pool_names <<<"${AKS_CACHE_POOL_NAMES}"
  for tier in "${tiers[@]}"; do
    IFS=',' read -r tier_name instance_type ssd_gib gpu_count <<<"${tier}"
    pool_name="${pool_names[$index]}"
    verify_aks_pool "${pool_name}" "${instance_type}" 0 "${GPU_MAX_NODES}" \
      "kumo.ai/gpu-node=true kumo.ai/diskgraph-pool-type=cache kumo.ai/diskgraph-ssd-size=${ssd_gib} kumo.ai/diskgraph-gpu-count=${gpu_count}" true
    index=$((index + 1))
  done
  verify_aks_pool "${AKS_EPHEMERAL_POOL_NAME}" \
    "${GPU_EPHEMERAL_INSTANCE_TYPE}" 0 "${GPU_MAX_NODES}" \
    "kumo.ai/gpu-node=true kumo.ai/diskgraph-pool-type=ephemeral" true

  verify_no_gpu_nodes
}

delete_aks_pool() {
  local name="$1"
  if aks_pool_exists "${name}"; then
    az aks nodepool delete \
      --resource-group "${PROVIDER_RESOURCE_GROUP}" \
      --cluster-name "${PROVIDER_CLUSTER}" \
      --name "${name}"
  fi
}

provider_decommission() {
  local pool_name
  local -a pool_names
  require_command az
  IFS=',' read -r -a pool_names <<<"${AKS_CACHE_POOL_NAMES}"
  for pool_name in "${pool_names[@]}" "${AKS_EPHEMERAL_POOL_NAME}"; do
    delete_aks_pool "${pool_name}"
  done
  delete_aks_pool "${AKS_CPU_POOL_NAME}"
}
