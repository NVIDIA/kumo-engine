# Azure Data Lake Storage Gen2

Kumo supports hierarchical-namespace ADLS Gen2 under the shared
[cloud object-storage contract](README.md). Use Microsoft Entra workload or
managed identity. Storage keys, stored client secrets, and user-supplied SAS
tokens are not connector identity modes.

## Identity and RBAC

Project connectors use workload identity federation. Configure the tenant ID,
client ID, Azure cloud, projected token file (`AZURE_FEDERATED_TOKEN_FILE`),
issuer, subject, and audience; Kumo validates a storage token before use. The
Azure single-host deployment attaches a user-assigned managed identity. Never
put credentials in images, Terraform state, user data, or connector config.

| Purpose                    | Minimum role                    | Scope                       |
| -------------------------- | ------------------------------- | --------------------------- |
| Read/list sources          | `Storage Blob Data Reader`      | Source filesystem           |
| Rooted output or dataplane | `Storage Blob Data Contributor` | Output/dataplane filesystem |
| User-delegation SAS        | `Storage Blob Delegator`        | Storage account             |
| Immutable artifacts        | `Storage Blob Data Reader`      | Artifact filesystem         |

`Storage Blob Delegator` at account scope supplies delegation-key permission
when data access is only filesystem-scoped; an existing account-scoped role
may already grant it. See [delegation-key permissions](https://learn.microsoft.com/en-us/rest/api/storageservices/get-user-delegation-key).
For ACL-based access, grant execute on ancestors and the required target
permissions. ACLs cannot restrict access already granted by sufficient RBAC
permissions; see [Azure permission evaluation](https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-access-control-model#how-permissions-are-evaluated). The
[Azure deployment guide](../../deploy/azure/README.md) defines the separate
builder, runtime, publisher, and optional Key Vault scopes.

## ADLS-specific URI and transfer behavior

| Cloud         | Identity setting | Required DFS suffix          |
| ------------- | ---------------- | ---------------------------- |
| Public        | `public`         | `dfs.core.windows.net`       |
| US Government | `us_government`  | `dfs.core.usgovcloudapi.net` |
| China         | `china`          | `dfs.core.chinacloudapi.cn`  |

Use `abfss://<filesystem>@<account>.<suffix>/<path>`. Source-table `abfs://`,
Blob endpoints, custom or mixed-cloud suffixes, and account-wide discovery are
unsupported. An any-URI connector may read an exact authorized URI across
accounts; it still follows the shared source-only contract.

ADLS browser uploads use signed staged blocks. Downloads use read-only
user-delegation SAS or ZIP for multi-file results. ADLS has no requester-pays
option.

## ADLS dataplane operations

Use a root such as
`abfss://dataplane@<account>.<suffix>/kumo/dataplane`. The VM identity needs
contributor on that filesystem and reader on the separate artifact filesystem.
First boot accepts `abfs://` only for this dataplane input and rewrites it to
validated `abfss://`; connector and artifact URIs must already use `abfss://`.

In addition to the shared dataplane checks, first boot verifies artifact
read/list access. For failures, inspect Azure Activity and storage logs and
verify cloud selection, both DFS and Blob private DNS, RBAC propagation, HNS
ACLs, and Delegator scope.

See the [Azure deployment guide](../../deploy/azure/README.md) for build,
networking, preflight, replacement upgrades, and teardown. Live identity,
private networking, RBAC/ACL propagation, user-delegation signing, and HNS
concurrency remain release checks. Sovereign behavior remains offline-only
unless a real sovereign environment is available.
