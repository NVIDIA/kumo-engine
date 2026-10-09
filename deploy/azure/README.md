# Kumo single-host on Azure

This directory builds and provisions the Azure single-host path with a
user-assigned managed identity, ADLS Gen2 dataplane and artifact roots, and
private storage endpoints by default. Do not put storage keys,
service-principal secrets, Direct Connect secrets, or SAS tokens in images,
Terraform state, VM user data, or deploy configuration.

## Prerequisites

- Azure CLI, Packer, and Terraform authenticated to the selected Azure cloud.
- An image resource group and ACR containing the release images used by the
  AWS/GCP single-host builds. Set `AZURE_SUBSCRIPTION_ID`,
  `AZURE_IMAGE_RESOURCE_GROUP`, and `AZURE_ACR_NAME`.
- A builder identity with `AcrPull`; set
  `AZURE_BUILDER_IDENTITY_ID` and `AZURE_BUILDER_IDENTITY_CLIENT_ID`.
  Grant it `Storage Blob Data Reader` on the immutable GloVe source and set
  `AZURE_GLOVE_SOURCE_URL` to the HTTPS directory URL without a SAS token.
- A versioned Key Vault secret for the Direct Connect OIDC client secret, or
  the temporary password when using password mode. Terraform receives only
  the secret ID.
- For the default private topology, an apply agent and operator/client network
  routed to the generated VNet with Azure private-DNS resolution. An
  internet-only runner cannot create the HNS filesystems after private
  endpoints are enabled.

For sovereign deployments, set `AZURE_ENVIRONMENT` and
`azure_environment` to `usgovernment` or `china`. This selects the CLI cloud,
identity authority, ADLS suffix, private DNS zones, and preflight together;
unsupported suffixes fail closed.

## Build and deploy

Run the commands below from `deploy/azure`. Copy
[`terraform/terraform.tfvars.example`](terraform/terraform.tfvars.example) to
`terraform/terraform.tfvars` and complete its values, then run:

```bash
./packer-build-azure.sh <release-image-tag>
./launch-instance-azure.sh terraform.tfvars
```

Set `key_vault_id` and exactly the applicable versioned secret URI:
`oidc_client_secret_uri` or `direct_connect_admin_temp_password_uri`. See
[`terraform/variables.tf`](terraform/variables.tf) for all supported values and
defaults. First boot retrieves the secret with managed identity into Kumo's
runtime-owned secret file (mode `0600`); it does not write the value to
structured config.

Terraform grants the runtime identity only:

- `Storage Blob Data Contributor` on the dataplane filesystem;
- `Storage Blob Data Reader` on the artifact filesystem;
- `Storage Blob Delegator` on the account for signed downloads; and
- `Key Vault Secrets User` on the configured vault, when used.

The runtime cannot publish artifacts. Populate the immutable artifact
filesystem with a separate publisher identity scoped as
`Storage Blob Data Contributor`, and record artifact hashes/version metadata.
The default artifact root is
`abfss://artifacts@<account>.<sovereign-dfs-suffix>/kumo`.

## Storage, network, and validation

The shared [cloud object-storage contract](../../docs/cloud-storage/README.md)
defines lifecycle scope, readiness, recovery, and release evidence.

Public VM and storage access are disabled by default. Terraform creates Blob
and DFS private endpoints and private DNS zones, but not customer peering,
VPN/ExpressRoute, Bastion, or private DNS for `public_url`. Operators and API
clients must supply that routed access. Public access requires the explicit
public-IP, storage-access, and ingress variables; keep ingress CIDRs empty
unless public access is intentional.

First boot runs
[`cloud-storage-preflight.sh`](../shared/scripts/cloud-storage-preflight.sh)
against the dataplane and artifact roots; any failure stops startup. Diagnose
with `/var/log/cloud-init-output.log`, `kumo-health`, `kumo-diag`, and the
Terraform serial-console output.

## Upgrade, cleanup, and limitations

Build an immutable managed image for an upgrade; do not mutate cached
application images in place. Before replacing the VM through Terraform, back
up and rehearse restoration of the bundled PostgreSQL data and
`/opt/kumo/secrets`, including encryption keys. They live on the VM's OS disk;
Terraform replacement does not migrate them to the new VM. An external
PostgreSQL deployment still requires preserving the matching secrets.

The teardown command destroys the Terraform-managed VM, storage account and
its filesystems, identity, and networking. Retain or export required data
before running it:

```bash
./teardown-instance-azure.sh terraform.tfvars
```

The image build currently caches Prometheus as `kumo/prometheus:latest`,
while the shared Compose default requests `prom/prometheus:v2.53.4`. Ensure
that requested image is available to the VM, or prepare an explicit
`PROMETHEUS_IMAGE` override. A build alone does not prove offline image closure.

Checked-in tests do not prove a real image build or apply, metadata-service or
managed-identity behavior, RBAC propagation, private routing/DNS, lifecycle
expiration, restart/upgrade, or teardown. These remain release acceptance
steps. Azure Serial Console with managed boot diagnostics is the break-glass
path when guest networking is unavailable.
