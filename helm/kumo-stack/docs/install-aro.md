# Install Kumo on ARO

> Reference deployment example, outside the supported Kumo application-chart
> boundary. Operators own the surrounding infrastructure. See the
> [bundle guide](../README.md) for scope and prerequisites.


This recipe installs one Kumo environment on a fresh Azure Red Hat OpenShift
(ARO) cluster. The built-in OpenShift router terminates HTTPS and routes
directly to the Kumo UI, and Kumo initially uses email/password login. No
Istio or other service mesh is installed.

## Prerequisites

Prepare all of the following before starting:

- An Azure subscription where you can create resources, managed identities,
  federated credentials, and role assignments.
- A [Red Hat pull secret](https://console.redhat.com/openshift/install/pull-secret)
  saved as a local JSON file. Passing it during ARO creation enables the
  Red Hat and certified operator catalogs used for GPU support.
- Regional quota for the reference ARO cluster and Kumo GPU pools. A fresh
  reference deployment needs 44 `standardDSv5Family` vCPUs during cluster
  creation, 32 `NCASv3_T4` vCPUs for the four scale-to-zero GPU pools, and 68
  total regional vCPUs at maximum steady-state scale.
- An installer host with private network and DNS connectivity to the target
  virtual network. The host must be able to reach the private ARO API and
  private PostgreSQL endpoint.
- Azure Database for PostgreSQL Flexible Server 14 or newer with private
  networking, private DNS, Microsoft Entra authentication, password
  authentication, and `btree_gin` listed in the `azure.extensions` parameter.
  Version 15 or newer is required when `cloud.postgres.bootstrapTemporal` is
  true; preflight and the bootstrap both refuse to run against 14.
- A Microsoft Entra PostgreSQL administrator mapped to a user-assigned managed
  identity, so the in-cluster bootstrap Job can log in as it. Keep the server's
  local administrator for infrastructure recovery; Kumo does not consume that
  password.
- One user-assigned managed identity shared by the Kumo `control`, `worker`,
  and `controller` Kubernetes ServiceAccounts.
- A private Azure Data Lake Storage Gen2 container and prefix for
  Kumo-managed data.
- Access to an approved registry containing all three compatible Kumo release
  images. Public artifact locations remain a release prerequisite.
- Node network paths to that registry and Docker Hub, or private mirrors for
  both.
- Azure CLI 2.84.0 or newer, `oc` and `kubectl` from the same OpenShift 4.20
  client archive, Helm 3, `jq`, `openssl`, `psql` 14 or newer, and `curl`.
  `psql` is used once to create the Kumo application-metadata database; the
  Temporal database bootstrap runs inside the cluster.
- A versioned editable bundle compatible with the selected application release;
  see the artifact caveat in the [bundle README](../README.md).

## End-to-end network architecture

```text
Customer private network
  User browser
      |
      | HTTPS:443; private DNS resolves the Kumo hostname
      | ExpressRoute, VPN, or VNet peering
      v
Azure virtual network
  ARO internal Azure Load Balancer  [Layer 4 forwarding]
      |
      v
  ARO cluster
    Built-in OpenShift ingress router  [TLS termination]
      |
      | OpenShift Route; HTTP:80
      v
    kumo-ui Service -> UI pod and built-in API reverse proxy
                           |
                           +--> Kumo REST/API and control pods
                                   |
                                   +--> Temporal frontend
                                   |
                                   +--> CPU/GPU worker pods

  Private managed services
    Kumo pods ----- TLS + Entra identity --------> PostgreSQL metadata DB
    Temporal ------ TLS + Kubernetes password ---> PostgreSQL Temporal DBs
    Kumo workers -- HTTPS + Entra identity ------> ADLS Gen2
```

For inbound application traffic, client TLS terminates at the OpenShift router
and the private router-to-UI hop uses HTTP. Connections to PostgreSQL and ADLS
use their own TLS. Traffic behind the router stays on private addresses;
operators must supply Kumo namespace NetworkPolicies to restrict it. Bundled
Temporal installs its own ingress policy.

ARO intentionally keeps its built-in router instead of adding Azure
Application Gateway. Every supported ARO cluster already runs and depends on
the OpenShift ingress controller for platform routes. Application Gateway
would add another cloud resource and controller without removing that router,
so it would increase rather than simplify the default deployment.

## Prepare the ARO environment

This reference creates the Azure and OpenShift resources required by the
installation: a private managed-identity ARO cluster, private PostgreSQL and
ADLS, Kumo workload identities, three T4 cache tiers, one ephemeral T4 pool,
and cluster-wide autoscaling limits.

Run these commands in one Bash session as a Microsoft Entra user that can
create resources and role assignments in the target subscription. The host
must already have private network and DNS connectivity to the new virtual
network; this recipe does not create an enterprise VPN, ExpressRoute circuit,
DNS forwarder, or installer VM. It also does not obtain the Red Hat pull
secret, Kumo registry credential, or additional quota.

Microsoft's
[managed-identity ARO procedure](https://learn.microsoft.com/azure/openshift/howto-create-openshift-cluster)
is the source for the platform identities and role assignments below.

### 1. Set the deployment names

Change only the values in the first block. `UNIQUE_SUFFIX` must contain only
lowercase letters and numbers and keep `kumo${UNIQUE_SUFFIX}` at 24 characters
or fewer. Select an exact ARO 4.20 version returned by the readiness command in
the next step.

```bash
set -euo pipefail
cleanup_kumo_install_secrets() {
  unset POSTGRES_ADMIN_TOKEN POSTGRES_LOCAL_ADMIN_PASSWORD
}
trap cleanup_kumo_install_secrets EXIT

SUBSCRIPTION_ID="<subscription-id>"
LOCATION="westus2"
PREFIX="acme-kumo"
UNIQUE_SUFFIX="acme1234"
TENANT_NAME="acme"
ARO_VERSION="<supported-4.20-version>"
RED_HAT_PULL_SECRET_FILE="/secure/path/pull-secret.txt"

RESOURCE_GROUP="${PREFIX}-rg"
VNET_NAME="${PREFIX}-vnet"
ARO_CLUSTER="${PREFIX}-aro"
STORAGE_ACCOUNT="kumo${UNIQUE_SUFFIX}"
POSTGRES_SERVER="${PREFIX}-${UNIQUE_SUFFIX}"
WORKLOAD_IDENTITY="${PREFIX}-workload"
POSTGRES_ADMIN_IDENTITY="${PREFIX}-postgres-admin"
TENANT_NAMESPACE="kumo-${TENANT_NAME}"
TEMPORAL_NAMESPACE="temporal"
POSTGRES_DNS_ZONE="${PREFIX}.private.postgres.database.azure.com"
ADLS_FILESYSTEM="kumo"
POSTGRES_LOCAL_ADMIN="local_recovery_admin"
POSTGRES_STATE_DATABASE="kumo"
POSTGRES_STATE_USER="kumo_workload"
GPU_SMALL_MACHINESET="${ARO_CLUSTER}-gpu-small"
GPU_MEDIUM_MACHINESET="${ARO_CLUSTER}-gpu-medium"
GPU_LARGE_MACHINESET="${ARO_CLUSTER}-gpu-large"
GPU_EPHEMERAL_MACHINESET="${ARO_CLUSTER}-gpu-ephemeral"

az account set --subscription "${SUBSCRIPTION_ID}"
ENTRA_ADMIN_OBJECT_ID=$(
  az ad signed-in-user show --query id --output tsv
)
ENTRA_ADMIN_NAME=$(
  az ad signed-in-user show --query userPrincipalName --output tsv
)
```

The signed-in user is the PostgreSQL Microsoft Entra administrator used for
the one-time Kumo application-metadata database setup. A separate managed
identity becomes an additional PostgreSQL administrator for the in-cluster
Temporal bootstrap.

### 2. Check subscription and installer readiness

Verify the client versions and the Red Hat pull secret before creating
resources:

```bash
az version --query '"azure-cli"' --output tsv
oc version --client
kubectl version --client
helm version --short
psql --version
jq -e '
  .auths["quay.io"] and
  .auths["registry.connect.redhat.com"] and
  .auths["registry.redhat.io"]
' "${RED_HAT_PULL_SECRET_FILE}" >/dev/null
```

Azure CLI must be 2.84.0 or newer. Obtain `oc` and `kubectl` from the same
OpenShift 4.20 client archive. All providers in this read-only query must
report `Registered`; ask the subscription owner to register any missing
provider before continuing.

```bash
az provider list \
  --query "[?namespace=='Microsoft.RedHatOpenShift' ||
    namespace=='Microsoft.Compute' ||
    namespace=='Microsoft.Network' ||
    namespace=='Microsoft.Storage' ||
    namespace=='Microsoft.ManagedIdentity' ||
    namespace=='Microsoft.DBforPostgreSQL' ||
    namespace=='Microsoft.Quota'].{
      provider:namespace,
      state:registrationState
    }" \
  --output table

az aro get-versions --location "${LOCATION}" --output table
```

Confirm that `ARO_VERSION` appears in the version list and that the fresh
subscription has at least 44 DSv5-family vCPUs, 32 T4-family vCPUs, and 68
total regional vCPUs available:

```bash
az vm list-usage \
  --location "${LOCATION}" \
  --query "[?name.value=='cores' ||
    name.value=='standardDSv5Family' ||
    contains(name.localizedValue, 'NCASv3_T4')].{
      quota:name.localizedValue,
      used:currentValue,
      limit:limit
    }" \
  --output table
```

Stop and have the subscription owner raise quota if any row lacks that
headroom. ARO temporarily uses 44 DSv5-family vCPUs while creating the
bootstrap, control-plane, and three worker nodes; the bootstrap node is
removed after creation.

### 3. Create the virtual network

The reference CIDRs are non-overlapping with ARO's default pod and service
CIDRs. Replace the entire address plan if any range conflicts with a connected
network. ARO requires empty master and worker subnets without an attached NSG.

```bash
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}"

az network vnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${VNET_NAME}" \
  --location "${LOCATION}" \
  --address-prefixes 10.50.0.0/16

az network vnet subnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name master \
  --address-prefixes 10.50.0.0/23

az network vnet subnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name worker \
  --address-prefixes 10.50.2.0/23

az network vnet subnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name postgres \
  --address-prefixes 10.50.4.0/24 \
  --delegations Microsoft.DBforPostgreSQL/flexibleServers

az network vnet subnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name private-endpoints \
  --address-prefixes 10.50.5.0/24 \
  --private-endpoint-network-policies Disabled

VNET_ID=$(
  az network vnet show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VNET_NAME}" \
    --query id \
    --output tsv
)
MASTER_SUBNET_ID=$(
  az network vnet subnet show \
    --resource-group "${RESOURCE_GROUP}" \
    --vnet-name "${VNET_NAME}" \
    --name master \
    --query id \
    --output tsv
)
WORKER_SUBNET_ID=$(
  az network vnet subnet show \
    --resource-group "${RESOURCE_GROUP}" \
    --vnet-name "${VNET_NAME}" \
    --name worker \
    --query id \
    --output tsv
)
POSTGRES_SUBNET_ID=$(
  az network vnet subnet show \
    --resource-group "${RESOURCE_GROUP}" \
    --vnet-name "${VNET_NAME}" \
    --name postgres \
    --query id \
    --output tsv
)
```

### 4. Create the ARO platform identities

ARO uses one cluster identity and eight operator identities. The role IDs are
ARO-specific built-in roles from Microsoft's managed-identity installation
procedure.

```bash
ARO_CLUSTER_IDENTITY="${PREFIX}-aro-cluster"
ARO_PLATFORM_IDENTITIES=(
  cloud-controller-manager
  ingress
  machine-api
  disk-csi-driver
  cloud-network-config
  image-registry
  file-csi-driver
  aro-operator
)

for identity_name in \
  "${ARO_CLUSTER_IDENTITY}" "${ARO_PLATFORM_IDENTITIES[@]}"; do
  az identity create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${identity_name}" \
    --location "${LOCATION}" \
    --output none
done

identity_id() {
  az identity show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "$1" \
    --query id \
    --output tsv
}

identity_principal_id() {
  az identity show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "$1" \
    --query principalId \
    --output tsv
}

assign_aro_role() {
  az role assignment create \
    --assignee-object-id "$1" \
    --assignee-principal-type ServicePrincipal \
    --role \
      "/subscriptions/${SUBSCRIPTION_ID}/providers/Microsoft.Authorization/roleDefinitions/$2" \
    --scope "$3" \
    --output none
}

ARO_CLUSTER_PRINCIPAL_ID=$(identity_principal_id "${ARO_CLUSTER_IDENTITY}")
for identity_name in "${ARO_PLATFORM_IDENTITIES[@]}"; do
  assign_aro_role \
    "${ARO_CLUSTER_PRINCIPAL_ID}" \
    ef318e2a-8334-4a05-9e4a-295a196c6a6e \
    "$(identity_id "${identity_name}")"
done

for subnet_id in "${MASTER_SUBNET_ID}" "${WORKER_SUBNET_ID}"; do
  assign_aro_role \
    "$(identity_principal_id cloud-controller-manager)" \
    a1f96423-95ce-4224-ab27-4e3dc72facd4 \
    "${subnet_id}"
  assign_aro_role \
    "$(identity_principal_id ingress)" \
    0336e1d3-7a87-462b-b6db-342b63f7802c \
    "${subnet_id}"
  assign_aro_role \
    "$(identity_principal_id machine-api)" \
    0358943c-7e01-48ba-8889-02cc51d78637 \
    "${subnet_id}"
  assign_aro_role \
    "$(identity_principal_id aro-operator)" \
    4436bae4-7702-4c84-919b-c4069ff25ee2 \
    "${subnet_id}"
done

assign_aro_role \
  "$(identity_principal_id cloud-network-config)" \
  be7a6435-15ae-4171-8f30-4a343eff9e8f \
  "${VNET_ID}"
assign_aro_role \
  "$(identity_principal_id file-csi-driver)" \
  0d7aedc0-15fd-4a67-a412-efad370c947e \
  "${VNET_ID}"
assign_aro_role \
  "$(identity_principal_id image-registry)" \
  8b32b316-c2f5-4ddf-b05b-83dacd2d08b5 \
  "${VNET_ID}"

ARO_RP_PRINCIPAL_ID=$(
  az ad sp list \
    --display-name "Azure Red Hat OpenShift RP" \
    --query '[0].id' \
    --output tsv
)
assign_aro_role \
  "${ARO_RP_PRINCIPAL_ID}" \
  42f3c60f-e7b1-46d7-ba56-6de681664342 \
  "${VNET_ID}"
```

The disk CSI identity needs the cluster-identity assignment created by the
first loop; it does not need an additional VNet role assignment.

### 5. Create the private ARO cluster

Pass the Red Hat pull secret during cluster creation. Without it, the
`redhat-operators` and `certified-operators` catalogs needed by the Kumo GPU
setup are unavailable. The command creates a private API, private ingress,
three 4-vCPU/16-GiB general workers, and no public management endpoint.

```bash
az aro create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ARO_CLUSTER}" \
  --location "${LOCATION}" \
  --vnet "${VNET_NAME}" \
  --master-subnet master \
  --worker-subnet worker \
  --version "${ARO_VERSION}" \
  --master-vm-size Standard_D8s_v5 \
  --worker-vm-size Standard_D4s_v5 \
  --worker-count 3 \
  --apiserver-visibility Private \
  --ingress-visibility Private \
  --pull-secret "@${RED_HAT_PULL_SECRET_FILE}" \
  --enable-managed-identity \
  --assign-cluster-identity "$(identity_id "${ARO_CLUSTER_IDENTITY}")" \
  --assign-platform-workload-identity \
    cloud-controller-manager "$(identity_id cloud-controller-manager)" \
  --assign-platform-workload-identity \
    ingress "$(identity_id ingress)" \
  --assign-platform-workload-identity \
    machine-api "$(identity_id machine-api)" \
  --assign-platform-workload-identity \
    disk-csi-driver "$(identity_id disk-csi-driver)" \
  --assign-platform-workload-identity \
    cloud-network-config "$(identity_id cloud-network-config)" \
  --assign-platform-workload-identity \
    image-registry "$(identity_id image-registry)" \
  --assign-platform-workload-identity \
    file-csi-driver "$(identity_id file-csi-driver)" \
  --assign-platform-workload-identity \
    aro-operator "$(identity_id aro-operator)"

export KUBECONFIG="${PWD}/kubeconfig-${ARO_CLUSTER}"
az aro get-admin-kubeconfig \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ARO_CLUSTER}" \
  --file "${KUBECONFIG}"
chmod 600 "${KUBECONFIG}"

ARO_CONTEXT=$(kubectl config current-context)
oc get nodes
oc --namespace openshift-cloud-credential-operator \
  rollout status deployment/pod-identity-webhook --timeout=10m

OIDC_ISSUER=$(
  az aro show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${ARO_CLUSTER}" \
    --query clusterProfile.oidcIssuer \
    --output tsv
)
ARO_CONSOLE_URL=$(
  az aro show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${ARO_CLUSTER}" \
    --query consoleProfile.url \
    --output tsv
)
ARO_APPS_DOMAIN=${ARO_CONSOLE_URL#https://console-openshift-console.}
ARO_APPS_DOMAIN=${ARO_APPS_DOMAIN%/}
EDGE_HOST="kumo.${ARO_APPS_DOMAIN}"
```

ARO owns the wildcard DNS and wildcard ingress certificate for
`*.${ARO_APPS_DOMAIN}`. Do not create an Azure Private DNS zone with that
managed suffix; a customer-owned zone with the same name can shadow ARO's
records.

### 6. Create private ADLS and PostgreSQL

Create the storage account and an ADLS Gen2 filesystem for Kumo-managed data:

```bash
STORAGE_ID=$(
  az storage account create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${STORAGE_ACCOUNT}" \
    --location "${LOCATION}" \
    --kind StorageV2 \
    --sku Standard_LRS \
    --enable-hierarchical-namespace true \
    --allow-blob-public-access false \
    --min-tls-version TLS1_2 \
    --public-network-access Disabled \
    --query id \
    --output tsv
)

az storage container-rm create \
  --resource-group "${RESOURCE_GROUP}" \
  --storage-account "${STORAGE_ID}" \
  --name "${ADLS_FILESYSTEM}" \
  --public-access off

for subresource in blob dfs; do
  zone_name="privatelink.${subresource}.core.windows.net"
  endpoint_name="${PREFIX}-${subresource}-pe"

  az network private-dns zone create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${zone_name}"

  az network private-dns link vnet create \
    --resource-group "${RESOURCE_GROUP}" \
    --zone-name "${zone_name}" \
    --name "${subresource}-vnet-link" \
    --virtual-network "${VNET_ID}" \
    --registration-enabled false

  az network private-endpoint create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${endpoint_name}" \
    --location "${LOCATION}" \
    --vnet-name "${VNET_NAME}" \
    --subnet private-endpoints \
    --private-connection-resource-id "${STORAGE_ID}" \
    --group-id "${subresource}" \
    --connection-name "${endpoint_name}"

  zone_id=$(
    az network private-dns zone show \
      --resource-group "${RESOURCE_GROUP}" \
      --name "${zone_name}" \
      --query id \
      --output tsv
  )

  az network private-endpoint dns-zone-group create \
    --resource-group "${RESOURCE_GROUP}" \
    --endpoint-name "${endpoint_name}" \
    --name default \
    --zone-name "${subresource}" \
    --private-dns-zone "${zone_id}"
done
```

The database called **StateDB** in Kumo configuration stores Kumo application
metadata. Temporal uses two separate databases on the same server. The local
password below is retained only for infrastructure recovery; Kumo does not
consume it. Azure CLI accepts it as a command argument, so run this step on a
dedicated administrator host where process listings and debug output are not
shared.

```bash
az network private-dns zone create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${POSTGRES_DNS_ZONE}"

az network private-dns link vnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --zone-name "${POSTGRES_DNS_ZONE}" \
  --name postgres-vnet-link \
  --virtual-network "${VNET_ID}" \
  --registration-enabled false

POSTGRES_DNS_ZONE_ID=$(
  az network private-dns zone show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${POSTGRES_DNS_ZONE}" \
    --query id \
    --output tsv
)

read -r -s -p "PostgreSQL local recovery password: " \
  POSTGRES_LOCAL_ADMIN_PASSWORD
printf '\n'

az postgres flexible-server create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${POSTGRES_SERVER}" \
  --location "${LOCATION}" \
  --version 16 \
  --tier Burstable \
  --sku-name Standard_B2ms \
  --storage-size 32 \
  --storage-auto-grow Enabled \
  --subnet "${POSTGRES_SUBNET_ID}" \
  --private-dns-zone "${POSTGRES_DNS_ZONE_ID}" \
  --microsoft-entra-auth Enabled \
  --password-auth Enabled \
  --admin-user "${POSTGRES_LOCAL_ADMIN}" \
  --admin-password "${POSTGRES_LOCAL_ADMIN_PASSWORD}" \
  --admin-object-id "${ENTRA_ADMIN_OBJECT_ID}" \
  --admin-display-name "${ENTRA_ADMIN_NAME}" \
  --admin-type User \
  --yes
unset POSTGRES_LOCAL_ADMIN_PASSWORD

az postgres flexible-server wait \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${POSTGRES_SERVER}" \
  --custom "state=='Ready'"

az postgres flexible-server parameter set \
  --resource-group "${RESOURCE_GROUP}" \
  --server-name "${POSTGRES_SERVER}" \
  --name azure.extensions \
  --value btree_gin
```

ADLS Gen2 requires both `blob` and `dfs` private endpoints. PostgreSQL and
ADLS names must resolve to private addresses from the installer host and ARO
worker subnet before continuing.

### 7. Configure Kumo and Temporal workload identity

The Kumo identity obtains short-lived Azure tokens for StateDB and ADLS from
the ARO OIDC issuer. The PostgreSQL administrator identity is used only by the
in-cluster Temporal bootstrap Job.

```bash
WORKLOAD_CLIENT_ID=$(
  az identity create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${WORKLOAD_IDENTITY}" \
    --location "${LOCATION}" \
    --query clientId \
    --output tsv
)
WORKLOAD_PRINCIPAL_ID=$(
  az identity show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${WORKLOAD_IDENTITY}" \
    --query principalId \
    --output tsv
)

POSTGRES_ADMIN_CLIENT_ID=$(
  az identity create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${POSTGRES_ADMIN_IDENTITY}" \
    --location "${LOCATION}" \
    --query clientId \
    --output tsv
)
POSTGRES_ADMIN_PRINCIPAL_ID=$(
  az identity show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${POSTGRES_ADMIN_IDENTITY}" \
    --query principalId \
    --output tsv
)

for service_account in kumo-control kumo-worker kumo-controller; do
  az identity federated-credential create \
    --resource-group "${RESOURCE_GROUP}" \
    --identity-name "${WORKLOAD_IDENTITY}" \
    --name "${service_account}" \
    --issuer "${OIDC_ISSUER}" \
    --subject \
      "system:serviceaccount:${TENANT_NAMESPACE}:${service_account}" \
    --audiences api://AzureADTokenExchange
done

az identity federated-credential create \
  --resource-group "${RESOURCE_GROUP}" \
  --identity-name "${POSTGRES_ADMIN_IDENTITY}" \
  --name kumo-temporal-bootstrap \
  --issuer "${OIDC_ISSUER}" \
  --subject \
    "system:serviceaccount:${TEMPORAL_NAMESPACE}:kumo-temporal-bootstrap" \
  --audiences api://AzureADTokenExchange

az role assignment create \
  --assignee-object-id "${WORKLOAD_PRINCIPAL_ID}" \
  --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" \
  --scope "${STORAGE_ID}"

az postgres flexible-server microsoft-entra-admin create \
  --resource-group "${RESOURCE_GROUP}" \
  --server-name "${POSTGRES_SERVER}" \
  --display-name "${POSTGRES_ADMIN_IDENTITY}" \
  --object-id "${POSTGRES_ADMIN_PRINCIPAL_ID}" \
  --type ServicePrincipal
```

Create StateDB and map the Kumo identity to its login. The command uses the
signed-in user's short-lived PostgreSQL token, verifies the server certificate
against the operating-system CA bundle, and is compatible with `psql` 14.

```bash
for candidate in \
  /etc/ssl/cert.pem \
  /etc/ssl/certs/ca-certificates.crt \
  /etc/pki/tls/certs/ca-bundle.crt; do
  if [[ -r "${candidate}" ]]; then
    POSTGRES_SSL_ROOT="${candidate}"
    break
  fi
done
: "${POSTGRES_SSL_ROOT:?No operating-system CA bundle found}"

[[ "${POSTGRES_STATE_USER}" =~ ^[a-z_][a-z0-9_]*$ ]]
[[ "${POSTGRES_STATE_DATABASE}" =~ ^[a-z_][a-z0-9_]*$ ]]

POSTGRES_ADMIN_TOKEN=$(
  az account get-access-token \
    --resource-type oss-rdbms \
    --query accessToken \
    --output tsv
)

PGPASSWORD="${POSTGRES_ADMIN_TOKEN}" \
PGSSLMODE=verify-full \
PGSSLROOTCERT="${POSTGRES_SSL_ROOT}" \
psql \
  --host "${POSTGRES_SERVER}.postgres.database.azure.com" \
  --port 5432 \
  --username "${ENTRA_ADMIN_NAME}" \
  --dbname postgres \
  --set=workload_principal_id="${WORKLOAD_PRINCIPAL_ID}" \
  --set=state_user="${POSTGRES_STATE_USER}" \
  --set=state_database="${POSTGRES_STATE_DATABASE}" <<'SQL'
SELECT * FROM pg_catalog.pgaadauth_create_principal_with_oid(
  :'state_user',
  :'workload_principal_id',
  'service',
  false,
  false
);
CREATE DATABASE :"state_database" OWNER :"state_user";
SQL

PGPASSWORD="${POSTGRES_ADMIN_TOKEN}" \
PGSSLMODE=verify-full \
PGSSLROOTCERT="${POSTGRES_SSL_ROOT}" \
psql \
  --host "${POSTGRES_SERVER}.postgres.database.azure.com" \
  --port 5432 \
  --username "${ENTRA_ADMIN_NAME}" \
  --dbname "${POSTGRES_STATE_DATABASE}" \
  --set=state_user="${POSTGRES_STATE_USER}" <<'SQL'
ALTER SCHEMA public OWNER TO :"state_user";
SQL
unset POSTGRES_ADMIN_TOKEN
```

Kumo runs its own StateDB schema migrations with this identity. The Temporal
bootstrap Job uses the separate managed administrator to create a password
login and the `temporal` and `temporal_visibility` databases; no Temporal
database password is handled on the installer host.

Microsoft documents the identity mapping functions in
[Manage Microsoft Entra roles in Azure Database for PostgreSQL](https://learn.microsoft.com/azure/postgresql/flexible-server/how-to-manage-azure-ad-users).

### 8. Prepare CPU and GPU MachineSets

Kumo schedules ordinary services only on nodes labeled
`kumo.ai/node-pool=cpu`. Stamp both the current general workers and their
MachineSet templates so replacement nodes keep the label.

```bash
MACHINE_NAMESPACE=openshift-machine-api
read -r -a GENERAL_MACHINESETS <<<"$(
  oc --namespace "${MACHINE_NAMESPACE}" get machineset \
    -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}'
)"

for machine_set in "${GENERAL_MACHINESETS[@]}"; do
  oc --namespace "${MACHINE_NAMESPACE}" patch machineset "${machine_set}" \
    --type merge \
    --patch \
      '{"spec":{"template":{"spec":{"metadata":{"labels":{"kumo.ai/node-pool":"cpu"}}}}}}'

  read -r -a machine_names <<<"$(
    oc --namespace "${MACHINE_NAMESPACE}" get machine \
      --selector \
        "machine.openshift.io/cluster-api-machineset=${machine_set}" \
      -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}'
  )"
  for machine_name in "${machine_names[@]}"; do
    node_name=$(
      oc --namespace "${MACHINE_NAMESPACE}" get machine "${machine_name}" \
        -o jsonpath='{.status.nodeRef.name}'
    )
    if [[ -n "${node_name}" ]]; then
      oc label node "${node_name}" kumo.ai/node-pool=cpu --overwrite
    fi
  done
done
```

Create the four scale-to-zero GPU MachineSets by cloning the Azure provider
specification from a general MachineSet. The `cluster-api/accelerator` label
must exactly match the GPU type in `ClusterAutoscaler/default`.

```bash
BASE_MACHINESET="${GENERAL_MACHINESETS[0]}"
BASE_MACHINESET_JSON=$(
  oc --namespace "${MACHINE_NAMESPACE}" \
    get machineset "${BASE_MACHINESET}" -o json
)

create_gpu_machine_set() {
  local name="$1"
  local vm_size="$2"

  jq \
    --arg name "${name}" \
    --arg vm_size "${vm_size}" '
      del(
        .metadata.annotations,
        .metadata.creationTimestamp,
        .metadata.generation,
        .metadata.managedFields,
        .metadata.ownerReferences,
        .metadata.resourceVersion,
        .metadata.uid,
        .status,
        .spec.template.spec.metadata.labels["kumo.ai/node-pool"]
      )
      | .metadata.name = $name
      | .spec.replicas = 0
      | .spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"] = $name
      | .spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"] = $name
      | .spec.template.spec.metadata.labels["cluster-api/accelerator"] = "nvidia-t4"
      | .spec.template.spec.providerSpec.value.vmSize = $vm_size
    ' <<<"${BASE_MACHINESET_JSON}" |
    oc apply -f -
}

create_gpu_machine_set \
  "${GPU_SMALL_MACHINESET}" Standard_NC4as_T4_v3
create_gpu_machine_set \
  "${GPU_MEDIUM_MACHINESET}" Standard_NC8as_T4_v3
create_gpu_machine_set \
  "${GPU_LARGE_MACHINESET}" Standard_NC16as_T4_v3
create_gpu_machine_set \
  "${GPU_EPHEMERAL_MACHINESET}" Standard_NC4as_T4_v3

oc apply -f - <<'YAML'
apiVersion: autoscaling.openshift.io/v1
kind: ClusterAutoscaler
metadata:
  name: default
spec:
  resourceLimits:
    maxNodesTotal: 10
    cores:
      min: 0
      max: 72
    memory:
      min: 0
      max: 400
    gpus:
      - type: nvidia-t4
        min: 0
        max: 4
  scaleDown:
    enabled: true
    delayAfterAdd: 10m
    delayAfterDelete: 5m
    delayAfterFailure: 30s
    unneededTime: 5m
    utilizationThreshold: "0.4"
YAML
```

OpenShift documents the exact accelerator-label matching requirement in
[Applying autoscaling to an OpenShift Container Platform cluster](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html/machine_management/applying-autoscaling).

The ARO environment is now ready for the Kumo installer. Use the following
values in `config.yaml`:

```text
cluster.context:                         ${ARO_CONTEXT}
cluster.cpuNodeSelector:
  kumo.ai/node-pool:                     cpu
cluster.providers.aro.clusterName:       ${ARO_CLUSTER}
cluster.providers.aro.resourceGroup:     ${RESOURCE_GROUP}
cluster.providers.aro.cacheMachineSets:
  -                                      ${GPU_SMALL_MACHINESET}
  -                                      ${GPU_MEDIUM_MACHINESET}
  -                                      ${GPU_LARGE_MACHINESET}
cluster.providers.aro.ephemeralMachineSet:
                                         ${GPU_EPHEMERAL_MACHINESET}
cloud.postgres.host:                     ${POSTGRES_SERVER}.postgres.database.azure.com
cloud.postgres.resourceGroup:            ${RESOURCE_GROUP}
cloud.postgres.admin.mode:               azure_entra
cloud.postgres.admin.username:           ${POSTGRES_ADMIN_IDENTITY}
cloud.postgres.admin.annotations:
  azure.workload.identity/client-id:     ${POSTGRES_ADMIN_CLIENT_ID}
cloud.postgres.stateDatabase:            ${POSTGRES_STATE_DATABASE}
cloud.postgres.stateUser:                ${POSTGRES_STATE_USER}
cloud.objectStorage.uri:
  abfss://${ADLS_FILESYSTEM}@${STORAGE_ACCOUNT}.dfs.core.windows.net/disk-graph
images.kumo.registry:                    <your-registry>
images.kumo.repoPath:                    <your-repository-prefix>
images.kumo.tag:                         <compatible-release-tag>
images.kumo.pullSecretName:              kumo-registry
gpu.cacheTiers:
  - name:                                small
    instanceType:                        Standard_NC4as_T4_v3
    ssdGiB:                              256
    gpuCount:                            1
  - name:                                medium
    instanceType:                        Standard_NC8as_T4_v3
    ssdGiB:                              512
    gpuCount:                            1
  - name:                                large
    instanceType:                        Standard_NC16as_T4_v3
    ssdGiB:                              1024
    gpuCount:                            1
gpu.ephemeral.instanceType:              Standard_NC4as_T4_v3
gpu.maxNodes:                            1
stateDbIdentity.controlAnnotations:
  azure.workload.identity/client-id:     ${WORKLOAD_CLIENT_ID}
stateDbIdentity.workerAnnotations:
  azure.workload.identity/client-id:     ${WORKLOAD_CLIENT_ID}
stateDbIdentity.controllerAnnotations:
  azure.workload.identity/client-id:     ${WORKLOAD_CLIENT_ID}
tenant.name:                             ${TENANT_NAME}
tenant.exposureHost:                     ${EDGE_HOST}
```

Grant `Storage Blob Data Reader` to the Kumo identity on any additional
source-data containers that Kumo will ingest. The storage-account-level
`Storage Blob Data Contributor` assignment above covers Kumo-managed data.

## Container ingredients

Set `images.kumo.registry`, `images.kumo.repoPath`, and a compatible
`images.kumo.tag` explicitly. The current source defaults reference a restricted
registry. The placeholders below are not public artifact locations. If
third-party images are mirrored, keep the listed versions and digests.

| Source repository and tag field | Purpose |
|---|---|
| `<registry>/<repoPath>/kumo-backend-cpu` + `images.kumo.tag` | Kumo API, workers, and controllers |
| `<registry>/<repoPath>/kumo-backend-gpu` + `images.kumo.tag` | Kumo GPU work |
| `<registry>/<repoPath>/kumo-ui` + `images.kumo.tag` | Kumo web UI |
| `docker.io/temporalio/server:1.31.2@sha256:6b02e5176631c8d28f010735a3a73c69423a9adbcfad7e5dfe23836c904b7e26` | Temporal server |
| `docker.io/temporalio/admin-tools:1.31.2@sha256:7e5820112475b3490f011b28d86ca9fd8348f1640b8dd04c62adb906e6b28cb2` | Temporal schema and administration jobs |
| `docker.io/postgres:17.10-alpine@sha256:af194ccf3e2d7fe367012c7b88ce8b816c5c889b18a5b316799a1f0d7eac746a` | psql for the Temporal database bootstrap Job |

ARO resolves these platform images through approved OpenShift catalogs:

| Catalog dependency | Selection |
|---|---|
| Node Feature Discovery operand | `registry.redhat.io/openshift4/ose-node-feature-discovery-rhel9:v4.20` |
| Node Feature Discovery operator | `nfd` from `redhat-operators`, `stable` channel |
| NVIDIA GPU operator and related operands | `gpu-operator-certified` from `certified-operators`, `v26.3` channel |

OpenShift resolves operator bundle and `relatedImages` references from the
customer's catalogs. In a disconnected environment, mirror those catalog
channels using the standard OpenShift catalog workflow before running this
recipe. The built-in OpenShift router is part of ARO and adds no Kumo-selected
image.

This installation does not require Red Hat Service Mesh, ingress-nginx,
cert-manager, ExternalDNS, External Secrets, Secrets Store CSI, TopoLVM,
MinIO, an in-cluster PostgreSQL server, or bundled observability.

## Installation

Run all commands from the root of the extracted bundle directory after
logging in to the cluster with `oc`.

### 1. Verify the prepared cluster

```bash
VALUES_FILE="$PWD/config.yaml"
KUMO_STACK="$PWD/bin/kumo"

az account set --subscription "${SUBSCRIPTION_ID}"
oc --context "${ARO_CONTEXT}" get nodes
oc --context "${ARO_CONTEXT}" \
  --namespace openshift-cloud-credential-operator \
  get deployment pod-identity-webhook
```

`VALUES_FILE` and `KUMO_STACK` are shell conveniences used repeatedly below;
they do not configure the installer.

### 2. Edit the customer configuration

```bash
${EDITOR:-vi} "$VALUES_FILE"
```

Replace every example cluster name, Kubernetes context, MachineSet, database
host and login, storage account, managed-identity client ID, registry, tenant
name, hostname, and administrator email.

The shipped file selects EKS. Change `platform` and replace its provider block:

```yaml
platform: aro
cluster:
  context: <aro-context-name>
  cpuNodeSelector:
    kumo.ai/node-pool: cpu
  providers:
    aro:
      clusterName: <aro-cluster-name>
      resourceGroup: <aro-resource-group>
      operatorMode: dedicated
      cacheMachineSets:
        - <small-gpu-machine-set>
        - <medium-gpu-machine-set>
        - <large-gpu-machine-set>
      ephemeralMachineSet: <ephemeral-gpu-machine-set>
gpu:
  cacheTiers:
    - name: small
      instanceType: Standard_NC4as_T4_v3
      ssdGiB: 256
      gpuCount: 1
    - name: medium
      instanceType: Standard_NC8as_T4_v3
      ssdGiB: 512
      gpuCount: 1
    - name: large
      instanceType: Standard_NC16as_T4_v3
      ssdGiB: 1024
      gpuCount: 1
  ephemeral:
    instanceType: Standard_NC4as_T4_v3
  maxNodes: 1
```

Set the registry, repository path, tag, and pull Secret to match the chosen
Kumo image source. If Temporal was mirrored, replace its two repository fields.
This recipe does not install a service mesh. Do not place
the Kumo login password, registry password, or Temporal PostgreSQL password in
this file.

### 3. Render and inspect the complete configuration

```bash
bash "$KUMO_STACK" render \
  --values "$VALUES_FILE" \
  --output-dir ./rendered-aro

runtime_value() {
  sed -n "s/^${1}=//p" ./rendered-aro/runtime.env
}
CONTEXT=$(runtime_value CONTEXT)
TENANT_NAMESPACE=$(runtime_value TENANT_NAMESPACE)
TENANT_RELEASE=$(runtime_value TENANT_RELEASE)
EDGE_HOST=$(runtime_value EDGE_HOST)
TEMPORAL_NAMESPACE=$(runtime_value TEMPORAL_NAMESPACE)
```

Inspect `rendered-aro/runtime.env` and the four rendered Helm values files
(platform, Temporal, core, and edge).
They contain image references and Kubernetes Secret names, but no Secret
values. The assignments above read the non-secret names used below without
executing the file as a shell script.

### 4. Create the registry and initial-login Secrets

```bash
read -r -p "Registry hostname (match images.kumo.registry): " KUMO_REGISTRY
read -r -p "Registry username: " KUMO_REGISTRY_USER
read -r -s -p "Registry password or pull token: " KUMO_REGISTRY_PASSWORD
printf '\n'

for namespace in "$TENANT_NAMESPACE" "$TEMPORAL_NAMESPACE"; do
  oc --context "$CONTEXT" create namespace "$namespace" \
    --dry-run=client -o yaml |
    oc --context "$CONTEXT" apply -f -
  oc --context "$CONTEXT" --namespace "$namespace" \
    create secret docker-registry kumo-registry \
    --docker-server="$KUMO_REGISTRY" \
    --docker-username="$KUMO_REGISTRY_USER" \
    --docker-password="$KUMO_REGISTRY_PASSWORD" \
    --dry-run=client -o yaml |
    oc --context "$CONTEXT" --namespace "$namespace" apply -f -
done
unset KUMO_REGISTRY_PASSWORD

oc --context "$CONTEXT" create namespace "$TENANT_NAMESPACE" \
  --dry-run=client -o yaml |
  oc --context "$CONTEXT" apply -f -

read -r -s -p "Temporary Kumo admin password: " KUMO_BOOTSTRAP_PASSWORD
printf '\n'
printf '%s' "$KUMO_BOOTSTRAP_PASSWORD" |
  oc --context "$CONTEXT" \
    --namespace "$TENANT_NAMESPACE" \
    create secret generic kumo-bootstrap-password \
    --from-file=password=/dev/stdin \
    --dry-run=client -o yaml |
  oc --context "$CONTEXT" \
    --namespace "$TENANT_NAMESPACE" apply -f -
unset KUMO_BOOTSTRAP_PASSWORD
```

The commands above cover a registry requiring credentials with bundled
Temporal. If its credentials are already in the OpenShift global pull Secret,
skip the registry-secret loop and set `images.kumo.pullSecretName: ""`. With
external Temporal, create the registry Secret only in `TENANT_NAMESPACE`.
For a private mirror, use its registry and pull-only credential instead.

Before installation, supply Kumo namespace network policies appropriate to
this cluster's CNI and ingress path. The application chart does not install
them. To start from the bundled backend-ingress example:

```bash
cp charts/kumo-helm/examples/networkpolicy.yaml ./kumo-networkpolicy.yaml
${EDITOR:-vi} ./kumo-networkpolicy.yaml
# Replace <release> and <ns> with TENANT_RELEASE and TENANT_NAMESPACE.
# Review selectors, permitted traffic, and CNI enforcement before applying.
oc --context "$CONTEXT" --namespace "$TENANT_NAMESPACE" \
  apply -f ./kumo-networkpolicy.yaml
```

This example covers backend ingress only; operators own the complete network
policy for their deployment.

### 5. Install

```bash
bash install.sh
```

The installer installs the Node Feature Discovery and NVIDIA GPU operators
when `operatorMode: dedicated`, reconciles the GPU MachineSets and
autoscalers, sets OVN-Kubernetes `routingViaHost: true`, installs Temporal and
Kumo, and generates the Temporal database password directly into a Kubernetes
Secret.

### 6. Verify the private OpenShift route

```bash
getent hosts "${EDGE_HOST}"
curl --fail --show-error --head "https://${EDGE_HOST}/"
```

The hostname uses ARO's managed `apps.*.aroapp.io` wildcard DNS and ingress
certificate. If `getent` cannot resolve it from the installer host, fix the
corporate DNS forwarding path to the private ARO environment; do not create a
customer-owned zone that shadows the managed suffix.

### 7. Run the acceptance checks

```bash
bash "$KUMO_STACK" conformance --values "$VALUES_FILE"
bash "$KUMO_STACK" verify gpu-lifecycle --values "$VALUES_FILE"
bash "$KUMO_STACK" verify container-policy --values "$VALUES_FILE"
bash "$KUMO_STACK" status --values "$VALUES_FILE"
```

The GPU check creates each configured T4 tier and returns it to zero. The
container-policy check examines every namespace, so run it only on the fresh
acceptance cluster described by this recipe.

Log in at `https://$EDGE_HOST` with the configured administrator email and
temporary password, then replace that password through the Kumo UI.

For command and lifecycle details, see the
[shared installer reference](../README.md). For failures, use
[troubleshooting by stage](troubleshooting.md).
