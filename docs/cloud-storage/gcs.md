# Google Cloud Storage

Kumo supports GCS under the shared [cloud object-storage contract](README.md).
Use workload identity and service-account impersonation for new deployments.
Stored service-account keys remain supported for project connectors when
federation is unavailable; protect and rotate them through the identity
workflow.

## Identity and IAM

For workload identity, configure the deployment issuer/ADC, enable GCP
workload exchange, and grant the workload principal
`roles/iam.serviceAccountTokenCreator` on the target service account. Grant
storage access to that account, not individual Kumo users. Developer ADC and
static source credentials are rejected for workload exchange.

| Purpose                    | Minimum predefined role                   | Scope                   |
| -------------------------- | ----------------------------------------- | ----------------------- |
| Read/list sources          | `roles/storage.objectViewer`              | Source bucket           |
| Rooted output or dataplane | `roles/storage.objectAdmin`               | Output/dataplane bucket |
| Keyless URL signing        | `roles/iam.serviceAccountTokenCreator`    | Signing service account |
| Requester-pays billing     | `roles/serviceusage.serviceUsageConsumer` | Billing project         |

Narrower custom roles may provide equivalent permissions. Bucket creation,
IAM mutation, and account-wide listing are not runtime requirements.

For requester-pays buckets, set the connector billing project and grant
`serviceusage.services.use` there. This routes billing but does not grant
storage access; the target service account still needs bucket permissions.

GCS browser uploads use signed multipart composition; downloads use signed
URLs or ZIP for multi-file results. Non-GCS endpoints are unsupported.

## GCS dataplane operations

Use a root such as `gs://<bucket>/kumo/dataplane` with object-admin access for
the runtime service account. See the [GCP deployment assets](../../deploy/gcp/)
for VM, identity, bucket, private access, and first-boot configuration, and the
shared contract for lifecycle, preflight, recovery, and release evidence.

For failures, inspect Cloud Audit Logs and verify connector mode/root,
requester-pays project, IAM, and keyless signing. Live federation,
metadata-service behavior, private routing, and signing remain release checks.
