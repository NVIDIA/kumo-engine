# Install Kumo on AKS

> Reference deployment example, outside the supported Kumo application-chart
> boundary. Operators own the surrounding infrastructure. See the
> [bundle guide](../README.md) for scope and prerequisites.


This recipe installs one Kumo environment on a fresh AKS cluster. An
Azure Application Gateway v2 terminates HTTPS, and the AKS-managed Application
Gateway Ingress Controller (AGIC) routes requests to Kumo. Kumo initially uses
email/password login. Operators supply Kumo namespace NetworkPolicies; the
bundled Temporal chart installs its own ingress policy. No service mesh is installed.

## Prerequisites

Either supply existing resources that meet the following requirements or
create the reference environment in
[Prepare the AKS environment](#prepare-the-aks-environment):

- An amd64 AKS cluster on Kubernetes 1.33 or newer using Azure CNI or Azure
  CNI Overlay, with OIDC issuer, workload identity, Azure Disk CSI, and a
  network-policy implementation enabled.
- At least one 4-vCPU/16-GiB system node such as `Standard_D4as_v5`.
- Regional quota for three 8-vCPU/32-GiB Kumo CPU nodes such as
  `Standard_D8as_v5` (24 `standardDASv5Family` vCPUs), in addition to the
  system pool and unrelated usage. One node runs and upgrades the default
  Kumo release; the remaining quota permits the documented autoscaling ceiling
  for workload growth.
- Regional quota for the configured T4 pool maxima. With `gpu.maxNodes: 1`,
  the small, medium, large, and ephemeral defaults require 32 `NCASv3_T4`
  vCPUs in addition to unrelated usage.
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
- The subscription feature
  `Microsoft.Network/EnableApplicationGatewayNetworkIsolation` in
  `Registered` state.
- A dedicated `/24` subnet delegated to
  `Microsoft.Network/applicationGateways`, and an Application Gateway
  Standard_v2 or WAF_v2 with a private frontend IP in that subnet. With Azure
  CNI Overlay, the gateway and AKS cluster must use the same virtual network.
  Record the gateway's Azure resource ID.
- A certificate and private-key file for the Kumo hostname and an Azure
  Private DNS zone containing that hostname.
- An Azure Container Registry containing the three Kumo images. Grant the AKS
  kubelet identity `AcrPull` on that registry.
- Node network paths to Azure Container Registry, Docker Hub, and NVIDIA NGC,
  or private mirrors for all three.
- Azure CLI, Helm 3, `kubectl`, `jq`, `openssl`, `psql`, and `curl` on
  an installer host that can reach the private AKS API. `psql` and a reachable
  PostgreSQL endpoint are needed only for the one-time state-database step under
  "PostgreSQL access" below; every database step the stack itself runs happens
  inside the cluster.
- Access to the three compatible Kumo release images for import into ACR.
  Public artifact locations remain a release prerequisite.
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
  Private Application Gateway v2  [TLS termination]
      |
      | HTTP:80 to AKS pod addresses
      v
  AKS cluster
    AGIC add-on  - - programs gateway from the Kumo Ingress - -^

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

For inbound application traffic, client TLS terminates at Application Gateway
and the private gateway-to-UI hop uses HTTP. Connections to PostgreSQL and
ADLS use their own TLS. Traffic behind the gateway stays on private addresses
and requires operator-provided Kumo NetworkPolicies for traffic restrictions.
AGIC is a control-plane
component: requests pass through Application Gateway directly to Kumo pods,
not through the AGIC pod.

## Prepare the AKS environment

This recipe creates a private AKS environment with one system node pool,
PostgreSQL, ADLS, Application Gateway, ACR, and separate managed identities
for Kumo and Temporal database setup. The installer adds the application CPU
pool and scale-to-zero T4 pools.

Run these commands in one Bash session as a Microsoft Entra user that can
create resources and role assignments in the target subscription. The host
must already have private network and DNS connectivity to the new virtual
network; this recipe does not create an enterprise VPN, ExpressRoute circuit,
DNS forwarder, or installer VM. It also does not obtain the Kumo TLS
certificate, copy Kumo images, or request quota.

### 1. Set the deployment names

Change only the values in the first block. `UNIQUE_SUFFIX` must be lowercase
letters and numbers and short enough that `kumo${UNIQUE_SUFFIX}` is no more
than 24 characters.

```bash
set -euo pipefail
cleanup_kumo_install_secrets() {
  unset KUMO_REGISTRY_PASSWORD POSTGRES_ADMIN_TOKEN POSTGRES_LOCAL_ADMIN_PASSWORD \
    KUMO_BOOTSTRAP_PASSWORD
}
trap cleanup_kumo_install_secrets EXIT

SUBSCRIPTION_ID="<subscription-id>"
LOCATION="westus2"
PREFIX="acme-kumo"
UNIQUE_SUFFIX="acme1234"
TENANT_NAME="acme"
EDGE_DNS_ZONE="example.com"
EDGE_RECORD="kumo"

RESOURCE_GROUP="${PREFIX}-rg"
VNET_NAME="${PREFIX}-vnet"
AKS_CLUSTER="${PREFIX}-aks"
ACR_NAME="kumo${UNIQUE_SUFFIX}"
STORAGE_ACCOUNT="kumo${UNIQUE_SUFFIX}"
POSTGRES_SERVER="${PREFIX}-${UNIQUE_SUFFIX}"
APP_GATEWAY="${PREFIX}-appgw"
WORKLOAD_IDENTITY="${PREFIX}-workload"
POSTGRES_ADMIN_IDENTITY="${PREFIX}-postgres-admin"
TENANT_NAMESPACE="kumo-${TENANT_NAME}"
TEMPORAL_NAMESPACE="temporal"
EDGE_HOST="${EDGE_RECORD}.${EDGE_DNS_ZONE}"
POSTGRES_DNS_ZONE="${PREFIX}.private.postgres.database.azure.com"
ADLS_FILESYSTEM="kumo"
APP_GATEWAY_PRIVATE_IP="10.42.4.10"
POSTGRES_LOCAL_ADMIN="local_recovery_admin"
POSTGRES_STATE_DATABASE="kumo"
POSTGRES_STATE_USER="kumo_workload"

az account set --subscription "${SUBSCRIPTION_ID}"
ENTRA_ADMIN_OBJECT_ID=$(
  az ad signed-in-user show --query id --output tsv
)
ENTRA_ADMIN_NAME=$(
  az ad signed-in-user show --query userPrincipalName --output tsv
)
```

The signed-in user creates Kumo's application-metadata database. A separate
managed identity administers the Temporal databases.

### 2. Check subscription readiness

All providers in this read-only query must report `Registered`. Ask the
subscription owner to register any missing provider before continuing.

```bash
az provider list \
  --query "[?namespace=='Microsoft.ContainerService' ||
    namespace=='Microsoft.Compute' ||
    namespace=='Microsoft.Network' ||
    namespace=='Microsoft.Storage' ||
    namespace=='Microsoft.ManagedIdentity' ||
    namespace=='Microsoft.DBforPostgreSQL' ||
    namespace=='Microsoft.ContainerRegistry' ||
    namespace=='Microsoft.Quota'].{
      provider:namespace,
      state:registrationState
    }" \
  --output table
```

The following read-only quota query is sufficient for the default pool
topology:

```bash
az vm list-usage \
  --location "${LOCATION}" \
  --query "[?name.value=='cores' ||
    name.value=='standardDASv5Family' ||
    contains(name.localizedValue, 'NCASv3_T4')].{
      quota:name.localizedValue,
      used:currentValue,
      limit:limit
    }" \
  --output table
```

Before creating this fresh reference environment, `limit - used` must be at
least 32 for `Standard NCASv3_T4 Family vCPUs`, 28 for
`Standard DASv5 Family vCPUs`, and 60 for `Total Regional vCPUs`. These values
cover the four default T4 pool maxima, the three Kumo CPU nodes, and the
system node. Stop and have the subscription owner raise quota if any row has
less headroom; this recipe does not request or reserve quota.

A private-only Application Gateway also requires the subscription feature
below to report `Registered`:

```bash
az feature show \
  --namespace Microsoft.Network \
  --name EnableApplicationGatewayNetworkIsolation \
  --query properties.state \
  --output tsv
```

### 3. Create the virtual network

The reference CIDRs are non-overlapping with the AKS pod and service CIDRs.
Replace the entire address plan if any range conflicts with a connected
network.

```bash
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}"

az network vnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${VNET_NAME}" \
  --location "${LOCATION}" \
  --address-prefixes 10.42.0.0/16

az network vnet subnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name aks-nodes \
  --address-prefixes 10.42.0.0/22

az network vnet subnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name appgw \
  --address-prefixes 10.42.4.0/24 \
  --delegations Microsoft.Network/applicationGateways

az network vnet subnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name postgres \
  --address-prefixes 10.42.5.0/24 \
  --delegations Microsoft.DBforPostgreSQL/flexibleServers

az network vnet subnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name private-endpoints \
  --address-prefixes 10.42.7.0/24 \
  --private-endpoint-network-policies Disabled

VNET_ID=$(
  az network vnet show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VNET_NAME}" \
    --query id \
    --output tsv
)
AKS_SUBNET_ID=$(
  az network vnet subnet show \
    --resource-group "${RESOURCE_GROUP}" \
    --vnet-name "${VNET_NAME}" \
    --name aks-nodes \
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

Confirm that the fresh Application Gateway subnet is `/24`, delegated to
`Microsoft.Network/applicationGateways`, and has no attached resources:

```bash
az network vnet subnet show \
  --resource-group "${RESOURCE_GROUP}" \
  --vnet-name "${VNET_NAME}" \
  --name appgw \
  --query '{
    prefix:addressPrefix,
    delegation:delegations[0].serviceName,
    attachedResources:length(ipConfigurations || `[]`)
  }' \
  --output table
```

### 4. Create ACR and private ADLS

ACR uses its normal TLS endpoint for the reference deployment. AKS nodes have
outbound access but no public IPs or inbound management ports. Enterprises
may add ACR Private Link without changing the Helm values.

```bash
ACR_ID=$(
  az acr create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${ACR_NAME}" \
    --location "${LOCATION}" \
    --sku Basic \
    --admin-enabled false \
    --query id \
    --output tsv
)

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
```

ADLS Gen2 requires both `blob` and `dfs` private endpoints. The DNS zone
groups below create and maintain the corresponding private A records.

```bash
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

### 5. Create PostgreSQL and Application Gateway

The PostgreSQL size below is the reference configuration. Size it and
configure availability according to the customer's production requirements.
The local password is retained only as a recovery credential; Kumo uses the
Microsoft Entra administrator and workload identity. Azure CLI accepts the
recovery password only as a command argument, so run this step on a dedicated
administrator host where process listings and debug output are not shared.

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

az postgres flexible-server microsoft-entra-admin create \
  --resource-group "${RESOURCE_GROUP}" \
  --server-name "${POSTGRES_SERVER}" \
  --display-name "${ENTRA_ADMIN_NAME}" \
  --object-id "${ENTRA_ADMIN_OBJECT_ID}" \
  --type User

az network application-gateway create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_GATEWAY}" \
  --location "${LOCATION}" \
  --sku Standard_v2 \
  --capacity 1 \
  --vnet-name "${VNET_NAME}" \
  --subnet appgw \
  --private-ip-address "${APP_GATEWAY_PRIVATE_IP}" \
  --frontend-port 80 \
  --http-settings-port 80 \
  --http-settings-protocol Http \
  --priority 100

APPLICATION_GATEWAY_ID=$(
  az network application-gateway show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${APP_GATEWAY}" \
    --query id \
    --output tsv
)

az network private-dns zone create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${EDGE_DNS_ZONE}"

az network private-dns link vnet create \
  --resource-group "${RESOURCE_GROUP}" \
  --zone-name "${EDGE_DNS_ZONE}" \
  --name edge-vnet-link \
  --virtual-network "${VNET_ID}" \
  --registration-enabled false

az network private-dns record-set a add-record \
  --resource-group "${RESOURCE_GROUP}" \
  --zone-name "${EDGE_DNS_ZONE}" \
  --record-set-name "${EDGE_RECORD}" \
  --ipv4-address "${APP_GATEWAY_PRIVATE_IP}"
```

If the organization owns the application hostname in central private DNS,
skip the three edge-zone commands and create the equivalent private A record
there.

### 6. Create AKS and its workload identity

This creates only the inexpensive 4-vCPU/16-GiB system pool. Current AKS
enables Azure Disk CSI by default.

```bash
az aks create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${AKS_CLUSTER}" \
  --location "${LOCATION}" \
  --nodepool-name system \
  --node-count 1 \
  --node-vm-size Standard_D4as_v5 \
  --os-sku AzureLinux3 \
  --vnet-subnet-id "${AKS_SUBNET_ID}" \
  --network-plugin azure \
  --network-plugin-mode overlay \
  --network-dataplane azure \
  --network-policy azure \
  --pod-cidr 10.244.0.0/16 \
  --service-cidr 10.0.0.0/16 \
  --dns-service-ip 10.0.0.10 \
  --load-balancer-sku standard \
  --enable-managed-identity \
  --enable-private-cluster \
  --disable-public-fqdn \
  --enable-oidc-issuer \
  --enable-workload-identity \
  --attach-acr "${ACR_ID}" \
  --no-ssh-key

OIDC_ISSUER=$(
  az aks show \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${AKS_CLUSTER}" \
    --query oidcIssuerProfile.issuerUrl \
    --output tsv
)

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

Next, create the PostgreSQL role described in
[PostgreSQL access](#postgresql-access) and import the images in
[Container ingredients](#container-ingredients). Then update `config.yaml`
with:

```text
cluster.context:                         ${AKS_CLUSTER}
cluster.providers.aks.clusterName:       ${AKS_CLUSTER}
cluster.providers.aks.resourceGroup:     ${RESOURCE_GROUP}
cluster.providers.aks.applicationGatewayId:
                                         ${APPLICATION_GATEWAY_ID}
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
images.kumo.registry:                    ${ACR_NAME}.azurecr.io
images.kumo.repoPath:                    <destination-repository-prefix>
images.kumo.tag:                         <compatible-release-tag>
stateDbIdentity.*Annotations:
  azure.workload.identity/client-id:     ${WORKLOAD_CLIENT_ID}
tenant.name:                             ${TENANT_NAME}
tenant.releaseName:                      ${TENANT_NAME}
tenant.namespace:                        ${TENANT_NAMESPACE}
tenant.exposureHost:                     ${EDGE_HOST}
```

## Application Gateway access

The installer enables the AKS-managed `ingress-appgw` add-on and points it at
`cluster.providers.aks.applicationGatewayId`. If the add-on is already
enabled, it must already target that same gateway; the installer will not
retarget it because doing so could disrupt another application.

The installer identity needs `Contributor` on the AKS cluster and Application
Gateway, `Network Contributor` on the AKS node and Application Gateway
subnets, and `Role Based Access Control Administrator` on the gateway and its
subnet. The installer reconciles the managed add-on identity's `Contributor`
access to the gateway and `Network Contributor` access to its subnet. Custom
roles may provide the same actions.

The gateway needs network reachability to AKS pod addresses. It may also have
a public frontend, but this recipe neither creates a public listener nor
publishes public DNS.

Verify the gateway before installation:

```bash
az network application-gateway show \
  --ids "${APPLICATION_GATEWAY_ID}" \
  --query '{
    provisioningState:provisioningState,
    operationalState:operationalState,
    tier:sku.tier,
    privateIPs:frontendIPConfigurations[].privateIPAddress
  }'
```

Both states must report `Succeeded`/`Running`, the tier must be `Standard_v2`
or `WAF_v2`, and `privateIPs` must contain an address.

## Workload identity

The customer-values `stateDbIdentity` block configures the shared managed
identity that Kumo uses for PostgreSQL and Azure Data Lake Storage. For that
identity, create three federated credentials with audience
`api://AzureADTokenExchange`, one for each subject:

- `system:serviceaccount:<tenant-namespace>:kumo-control`
- `system:serviceaccount:<tenant-namespace>:kumo-worker`
- `system:serviceaccount:<tenant-namespace>:kumo-controller`

Put the managed identity's client ID in all three annotations in
`config.yaml`.

## PostgreSQL access

Kumo stores application metadata in a PostgreSQL database. The configuration
and diagnostics call this database **StateDB**. Connect to the `postgres`
database as the Microsoft Entra PostgreSQL administrator and:

1. Map the managed identity to the PostgreSQL login named by
   `cloud.postgres.stateUser`.
2. Create the database named by `cloud.postgres.stateDatabase`.
3. Make that login the owner of the database and its `public` schema. Kumo
   runs its own schema migrations with this identity.

Run the StateDB bootstrap with the same Entra identity used by Azure CLI:

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
\connect :state_database
ALTER SCHEMA public OWNER TO :"state_user";
SQL
unset POSTGRES_ADMIN_TOKEN
```

A Job in the `temporal` namespace creates a separate password login and the
`temporal` and `temporal_visibility` databases for the stock Temporal server.
The Temporal bootstrap Job uses the Microsoft Entra PostgreSQL administrator
configured above to create its role and databases. It authenticates through
Azure workload identity, so Temporal setup stores no PostgreSQL administrator
secret.

PostgreSQL 15 or newer is required when `bootstrapTemporal` is true: the
bootstrap reassigns each Temporal database's `public` schema, which earlier
releases allow only for a superuser.

Microsoft documents the identity mapping functions in
[Manage Microsoft Entra roles in Azure Database for PostgreSQL](https://learn.microsoft.com/azure/postgresql/security/security-manage-entra-users).

## Azure Data Lake Storage access

Grant the Kumo managed identity `Storage Blob Data Contributor` on the
storage account configured in `cloud.objectStorage.uri`. This provides the
read, write, list, and delete operations used below the configured prefix.
The shared identity also needs `Storage Blob Data Reader` on every additional
source-data container that Kumo may ingest.

On first startup, Kumo registers the AKS-injected workload identity as its
credential-free ADLS identity and assigns it to the initial `Default` group.
An ADLS connector can therefore omit `identity_id`; no Azure token, client
secret, or post-install SDK identity setup is needed. An administrator may
later replace that group assignment with narrower Kumo groups, and subsequent
startups will not restore the removed `Default` assignment. Azure role
assignments remain the source of storage authorization.

Azure documents narrower container scopes in
[Assign an Azure role for blob data access](https://learn.microsoft.com/azure/storage/blobs/assign-azure-role-data-access).
The current installer validates the storage-account-level assignment shown
in the environment-preparation recipe.

## Container ingredients

Use one compatible Kumo release tag for all three images. Set
`images.kumo.registry`, `images.kumo.repoPath`, and `images.kumo.tag` to their
ACR destination. The current source defaults reference a restricted registry;
the placeholders below are not public artifact locations. If third-party
images are mirrored, keep the listed versions and digests.

| Source repository and tag field | Purpose |
|---|---|
| `<registry>/<repoPath>/kumo-backend-cpu` + `images.kumo.tag` | Kumo API, workers, and controllers |
| `<registry>/<repoPath>/kumo-backend-gpu` + `images.kumo.tag` | Kumo GPU work |
| `<registry>/<repoPath>/kumo-ui` + `images.kumo.tag` | Kumo web UI |
| `docker.io/temporalio/server:1.31.2@sha256:6b02e5176631c8d28f010735a3a73c69423a9adbcfad7e5dfe23836c904b7e26` | Temporal server |
| `docker.io/temporalio/admin-tools:1.31.2@sha256:7e5820112475b3490f011b28d86ca9fd8348f1640b8dd04c62adb906e6b28cb2` | Temporal schema and administration jobs |
| `docker.io/postgres:17.10-alpine@sha256:af194ccf3e2d7fe367012c7b88ce8b816c5c889b18a5b316799a1f0d7eac746a` | psql for the Temporal database bootstrap Job |
| `nvcr.io/nvidia/k8s-device-plugin:v0.19.0@sha256:7bf6ab18378be099493c9fe50f9cd2d559e0d81f17742d7ac188365408579d5d` | NVIDIA device discovery |

Import the Kumo images from a registry you can access into the ACR created
above. This example uses registry username/token authentication; adjust the
import authentication for your registry. Use the destination prefix and tag
below in `config.yaml` as well.

```bash
read -r -p "Source registry hostname: " KUMO_SOURCE_REGISTRY
read -r -p "Source repository prefix: " KUMO_SOURCE_REPO_PATH
read -r -p "Destination repository prefix: " KUMO_DEST_REPO_PATH
read -r -p "Compatible Kumo release tag: " KUMO_RELEASE_TAG
read -r -p "Source registry username: " KUMO_REGISTRY_USER
read -r -s -p "Source registry password or pull token: " KUMO_REGISTRY_PASSWORD
printf '\n'

for image_name in kumo-backend-cpu kumo-backend-gpu kumo-ui; do
  az acr import \
    --name "${ACR_NAME}" \
    --source "${KUMO_SOURCE_REGISTRY}/${KUMO_SOURCE_REPO_PATH}/${image_name}:${KUMO_RELEASE_TAG}" \
    --image "${KUMO_DEST_REPO_PATH}/${image_name}:${KUMO_RELEASE_TAG}" \
    --username "${KUMO_REGISTRY_USER}" \
    --password "${KUMO_REGISTRY_PASSWORD}" \
    --force
done
unset KUMO_REGISTRY_PASSWORD
```

AKS supplies the control-plane, node, Azure Disk CSI, workload-identity, and
managed AGIC add-on images. Microsoft selects those image names from the AKS
and managed add-on revisions; Kumo neither selects nor mirrors them.

This installation does not require Istio, ingress-nginx, cert-manager,
ExternalDNS, External Secrets, Secrets Store CSI, TopoLVM, MinIO, an
in-cluster PostgreSQL server, or bundled observability.

## Installation

Run all commands from the root of the extracted bundle directory.

### 1. Connect to the cluster

```bash
VALUES_FILE="$PWD/config.yaml"
KUMO_STACK="$PWD/bin/kumo"

az account set --subscription "${SUBSCRIPTION_ID}"
az aks get-credentials \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${AKS_CLUSTER}" \
  --overwrite-existing

kubectl --context "${AKS_CLUSTER}" get nodes
```

`VALUES_FILE` and `KUMO_STACK` are shell conveniences used repeatedly below;
they do not configure the installer.

### 2. Edit the customer configuration

```bash
${EDITOR:-vi} "$VALUES_FILE"
```

Replace every example cluster name, Kubernetes context, Application Gateway
resource ID, database host and login, storage account, managed-identity client
ID, Azure Container Registry, certificate path, tenant name, hostname, and
administrator email.

The shipped file selects EKS. Change `platform`, replace its provider block
and AWS GPU tiers, and set the GPU node limit used in the quota estimate:

```yaml
platform: aks
cluster:
  providers:
    aks:
      clusterName: <aks-cluster-name>
      resourceGroup: <aks-resource-group>
      applicationGatewayId: /subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.Network/applicationGateways/<gateway-name>
gpu:
  cacheTiers:
    - {name: small, instanceType: Standard_NC4as_T4_v3, ssdGiB: 256, gpuCount: 1}
    - {name: medium, instanceType: Standard_NC8as_T4_v3, ssdGiB: 512, gpuCount: 1}
    - {name: large, instanceType: Standard_NC16as_T4_v3, ssdGiB: 1024, gpuCount: 1}
  ephemeral: {instanceType: Standard_NC4as_T4_v3}
  maxNodes: 1
```

Set the Kumo registry to the destination containing all three images, and set
`images.kumo.repoPath` and `images.kumo.tag` to the `KUMO_DEST_REPO_PATH` and
`KUMO_RELEASE_TAG` used during import. If third-party images were mirrored,
replace their repository fields.

The installer creates the certificate and private key as a TLS Secret in the
Kumo namespace. AGIC securely synchronizes that certificate to Application
Gateway for HTTPS termination. Do not place either the Kumo login password or
Temporal PostgreSQL password in this file.

### 3. Render and inspect the complete configuration

```bash
bash "$KUMO_STACK" render \
  --values "$VALUES_FILE" \
  --output-dir ./rendered-aks

runtime_value() {
  sed -n "s/^${1}=//p" ./rendered-aks/runtime.env
}
CONTEXT=$(runtime_value CONTEXT)
TENANT_NAMESPACE=$(runtime_value TENANT_NAMESPACE)
TENANT_RELEASE=$(runtime_value TENANT_RELEASE)
EDGE_HOST=$(runtime_value EDGE_HOST)
```

Inspect `rendered-aks/runtime.env` and the four rendered Helm values files
(platform, Temporal, core, and edge).
They contain image references and Kubernetes Secret names, but no Secret
values. The assignments above read the non-secret names used below without
executing the file as a shell script.

### 4. Create the initial Kumo login

```bash
kubectl --context "$CONTEXT" create namespace "$TENANT_NAMESPACE" \
  --dry-run=client -o yaml |
  kubectl --context "$CONTEXT" apply -f -

read -r -s -p "Temporary Kumo admin password: " KUMO_BOOTSTRAP_PASSWORD
printf '\n'
printf '%s' "$KUMO_BOOTSTRAP_PASSWORD" |
  kubectl --context "$CONTEXT" \
    --namespace "$TENANT_NAMESPACE" \
    create secret generic kumo-bootstrap-password \
    --from-file=password=/dev/stdin \
    --dry-run=client -o yaml |
  kubectl --context "$CONTEXT" \
    --namespace "$TENANT_NAMESPACE" apply -f -
unset KUMO_BOOTSTRAP_PASSWORD
```

Before installation, supply Kumo namespace network policies appropriate to
this cluster's CNI and ingress path. The application chart does not install
them. To start from the bundled backend-ingress example:

```bash
cp charts/kumo-helm/examples/networkpolicy.yaml ./kumo-networkpolicy.yaml
${EDITOR:-vi} ./kumo-networkpolicy.yaml
# Replace <release> and <ns> with TENANT_RELEASE and TENANT_NAMESPACE.
# Review selectors, permitted traffic, and CNI enforcement before applying.
kubectl --context "$CONTEXT" --namespace "$TENANT_NAMESPACE" \
  apply -f ./kumo-networkpolicy.yaml
```

This example covers backend ingress only; operators own the complete network
policy for their deployment.

### 5. Install

```bash
bash install.sh
```

The installer enables the AKS-managed AGIC add-on against the selected
Application Gateway, creates the edge TLS Secret, creates the CPU and T4
pools, installs the NVIDIA device plugin, Temporal, and Kumo, and generates
the Temporal database password directly into a Kubernetes Secret.

### 6. Verify private routing

```bash
kubectl --context "$CONTEXT" \
  --namespace "$TENANT_NAMESPACE" \
  wait --for=jsonpath='{.status.loadBalancer.ingress[0]}' \
  "ingress/${TENANT_RELEASE}-edge" --timeout=5m
for attempt in {1..30}; do
  if curl --fail --show-error --head "https://${EDGE_HOST}/"; then
    break
  fi
  (( attempt == 30 )) && exit 1
  sleep 10
done
```

The DNS record created during environment preparation must resolve
`EDGE_HOST` to the private Application Gateway frontend IP from the installer
host and corporate client network.

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
