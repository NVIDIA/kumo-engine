# diskgraph-fs

Helpers for building `object_store` clients used by `diskgraph-infer` and other crates. This crate only handles client configuration and credential resolution; it does not perform any file IO itself.

## Supported backends

- **S3** (AWS or S3-compatible)
- **GCS**
- **ADLS / Azure Blob Storage**

Each builder returns an `ObjectStore` plus the resolved configuration it ended up using.

## Usage

```rust
use diskgraph_fs::{build_s3_client, S3Config};

let client = build_s3_client(S3Config {
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

let store = client.store;
```

## Configuration details

### S3

`build_s3_client` combines explicit `S3Config` values with environment variables and AWS SDK defaults:

- `S3_ALLOW_HTTP` / `AWS_ALLOW_HTTP` → `allow_http`
- `S3_ENDPOINT` → `endpoint` (useful for MinIO or custom endpoints)
- `S3_PROFILE` / `AWS_PROFILE` → AWS profile selection
- `S3_REGION` → region (falls back to AWS SDK default or `us-east-1`)
- `S3_VIRTUAL_HOSTED` → virtual hosted style requests
- `S3_ACCESS_KEY` / `AWS_ACCESS_KEY_ID`
- `S3_SECRET_KEY` / `AWS_SECRET_ACCESS_KEY`
- `S3_SESSION_TOKEN` / `AWS_SESSION_TOKEN`

Credentials are loaded using the AWS SDK's default credential provider chain.

### GCS

`build_gcs_client` uses `GoogleCloudStorageBuilder::from_env()` and relies on standard GCP environment configuration (e.g., `GOOGLE_APPLICATION_CREDENTIALS`).

### ADLS / Azure Blob

`build_adls_client` maps config and environment variables to the Azure builder:

- `AZURE_ALLOW_HTTP`
- `AZURE_ENDPOINT`
- `AZURE_ACCESS_KEY`
- `AZURE_SAS_KEY`
- `AZURE_BEARER_TOKEN`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_TENANT_ID`

These fields align with the `object_store` Azure backend expectations.
