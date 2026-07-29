# diskgraph-fs

Utility crate for cloud storage access in the diskgraph pipeline. It wraps
`object_store` client setup and provides a pooled, bounded transfer engine for
multipart uploads and downloads.

## Key features

- **Client builders** for S3, GCS, and ADLS/Azure Blob using env-aware config.
- **CloudTransfer** for multipart uploads and downloads with shared limits.
- **BufferPool** to reuse large file buffers and avoid repeated allocations.
- **Async metadata helpers** for listing prefixes and objects, plus head/get.

## Quick start

```rust
use diskgraph_fs::{build_s3_client, CloudTransfer, CloudTransferConfig, S3Config};
use object_store::path::Path as ObjectPath;

let s3 = build_s3_client(S3Config {
    bucket: "my-bucket".to_string(),
    region: None,
    endpoint: None,
    allow_http: None,
    virtual_hosted: None,
    profile: None,
    access_key_id: None,
    secret_access_key: None,
    session_token: None,
})?;

let transfer = CloudTransfer::new(s3.store, CloudTransferConfig::from_env());
let mut buffer = transfer.acquire_buffer();
buffer.write_all(b"hello")?;
transfer
    .submit(ObjectPath::from("example/hello.txt"), buffer.into_inner())?
    .wait()?;
```

## CloudTransfer design

`CloudTransfer` is a pooled, bounded transfer engine built to move many large
objects efficiently while keeping memory stable.

### Core ideas

- **Pooled memory**: A fixed-size `BufferPool` hands out `PooledBuffer` values.
  These buffers are reused across uploads and downloads to avoid repeated heap
  allocations. Drop returns the buffer to the pool.
- **Zero-copy uploads**: Multipart uploads use a shared `Arc<Vec<u8>>` and send
  part payloads as slices into that buffer. Parts do not copy into temporary
  buffers before IO.
- **In-place downloads**: Multipart downloads allocate a single buffer sized to
  the object and fill ranges in-place. Part tasks write directly into the
  shared buffer, avoiding per-part allocations.
- **Global concurrency control**: A bounded queue limits file submissions, and
  a shared semaphore caps total part requests across all files. Uploads and
  downloads share the same limits, so the network stays saturated without
  exploding memory or threads.
- **File-level retries**: If a part fails, the file is retried as a whole.
  This keeps the logic simple and ensures that each completed file is fully
  consistent.

### High-throughput path

For a workload of many large files, the system maintains:

- A steady number of in-flight files (`max_files_in_flight`).
- A bounded number of in-flight part requests (`max_part_requests`).
- A stable memory footprint set by the buffer pool size and current file sizes.

This keeps the download pipeline CPU-light, IO-heavy, and predictable.

### Transfer limits

Configured via `CloudTransferConfig::from_env()`:

- `DISKGRAPH_CLOUD_PART_SIZE_MB` (default 64)
- `DISKGRAPH_CLOUD_DOWNLOAD_PART_MB` (default to upload part size)
- `DISKGRAPH_CLOUD_QUEUE_MAX` (default `ncpu/8 + 1`)
- `DISKGRAPH_CLOUD_FILES_IN_FLIGHT` (default `ncpu`)
- `DISKGRAPH_CLOUD_MAX_REQUESTS` (default `ncpu`)
- `DISKGRAPH_CLOUD_FILE_RETRY_MAX` (default 2)
- `DISKGRAPH_CLOUD_FILE_BUFFER_COUNT` (default `2 * ncpu`)

### Metadata helpers

`CloudTransfer` exposes simple object_store wrappers:

- `list_prefixes(prefix: &ObjectPath)`
- `list_objects(prefix: &ObjectPath)`
- `head(path: &ObjectPath)`
- `get_bytes(path: &ObjectPath)`
- `get_string(path: &ObjectPath)`

These keep the downloader code portable and avoid AWS SDK usage.

## Client builders

### S3

`build_s3_client` combines explicit `S3Config` values with env vars:

- `S3_ALLOW_HTTP` / `AWS_ALLOW_HTTP` -> `allow_http`
- `S3_ENDPOINT` -> `endpoint`
- `S3_PROFILE` / `AWS_PROFILE`
- `S3_REGION` -> region (fallback to SDK default or `us-east-1`)
- `S3_VIRTUAL_HOSTED`
- `S3_ACCESS_KEY` / `AWS_ACCESS_KEY_ID`
- `S3_SECRET_KEY` / `AWS_SECRET_ACCESS_KEY`
- `S3_SESSION_TOKEN` / `AWS_SESSION_TOKEN`

For async contexts, use `build_s3_client_async` to avoid nested runtimes.

### GCS

`build_gcs_client` uses `GoogleCloudStorageBuilder::from_env()` with standard
GCP env configuration (for example, `GOOGLE_APPLICATION_CREDENTIALS`).

### ADLS / Azure Blob

`build_adls_client` maps config and env vars to the Azure builder:

- `AZURE_ALLOW_HTTP`
- `AZURE_ENDPOINT`
- `AZURE_ACCESS_KEY`, `AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_STORAGE_ACCESS_KEY`
- `AZURE_SAS_KEY`
- `AZURE_BEARER_TOKEN`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_TENANT_ID`
- `AZURE_FEDERATED_TOKEN_FILE` (env only)
- `AZURE_AUTHORITY_HOST` (env only)

Only these are read; the builder is not populated from the wider environment.
Credential selection is `object_store`'s: a bearer token wins, then an account
key, then workload identity (`AZURE_CLIENT_ID` + `AZURE_TENANT_ID` +
`AZURE_FEDERATED_TOKEN_FILE`), then a client secret, then SAS. With none of
those it falls back to IMDS managed identity, which requires the pod to reach
the instance metadata endpoint — unavailable on clusters such as ARO.

## Notes

- `CloudTransfer` is designed for high-throughput uploads and downloads on
  large files. It favors fewer large buffers and a fixed concurrency budget.
- Use `CloudTransferStatsHandle` to read queue depth, in-flight counts, and
  the last error message for telemetry and progress reporting.
