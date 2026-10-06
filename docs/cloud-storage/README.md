# Cloud object storage

Kumo supports AWS S3, Google Cloud Storage, and Azure Data Lake Storage Gen2
for CSV/Parquet sources, rooted outputs, and provider-native dataplanes.

| Provider | Canonical URI | Provider and deployment guidance |
| -------- | ------------- | -------------------------------- |
| AWS S3 | `s3://bucket/path` | [AWS single-host guide](../../deploy/ami/USER_GUIDE.md) |
| GCS | `gs://bucket/path` | [GCS deltas](gcs.md) and [deployment assets](../../deploy/gcp/) |
| ADLS Gen2 | `abfss://filesystem@account.<dfs-suffix>/path` | [ADLS deltas](adls.md) and [Azure single-host guide](../../deploy/azure/README.md) |

See also the [development storage matrix](../devserver.md).

## Connector contract

A rooted connector fixes a provider authority and optional prefix. Tables use
relative paths; a canonical full URI beneath the same root may normalize to
the same relative coordinate.

An any-URI connector requires one exact canonical URI per source. It never
enumerates all storage visible to an identity and grants no access beyond that
identity. Any-URI connectors are source-only, including for custom prediction
input. Prediction, embedding, training-table, model, and other outputs require
a project-owned rooted connector.

Flat CSV datasets and nested or partitioned Parquet datasets are supported.
Recursive S3 and GCS scans consume file/directory types from provider listings
instead of issuing a separate provider directory probe for each listed entry;
other storage backends retain per-entry directory checks. If a provider lists
the same path as both an object and a directory prefix, directory semantics
take precedence.
Queries, fragments, traversal, encoded separators, missing paths, and
provider-mismatched schemes are rejected.

Source-column and sample-row workflows share a 120-second Temporal execution
deadline by default, including asynchronous sample-row reads. Set
`INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS` on the REST workload to change
the deadline. The workflow execution deadline returns sanitized, nonretryable
guidance; activity, transport, and unrelated failures retain the generic
sanitized failure. For synchronous HTTP reads, keep the fronting proxy read
timeout greater than the connector deadline.

## Dataplane operator contract

Dataplane roots are Kumo-managed and separate from project connector roots.
Lifecycle deletion is limited to `tmp_payloads/`,
`temporary-uploaded-datasets/`, and `tmp_archives/`. Set lifecycle readiness
only after those rules exist, keep permanent state outside those prefixes, and
retain temporary data longer than the workflow, retry, and recovery window.

Startup must pass identity, network/DNS, scoped list/write/read/delete/sign,
and provider-specific artifact checks. On failure, use `kumo-health` and
`kumo-diag` and inspect cloud-init, service, and provider audit logs. Do not
delete temporary objects belonging to an active or recoverable workflow.

Offline tests do not prove live identity, private routing, lifecycle
enforcement, image provisioning, restart/upgrade, or teardown. Before release,
run the affected connector/dataplane smoke with the intended identity and
record the environment, commit, commands, hashes, results, and limitations.
