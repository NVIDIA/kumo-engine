//! Integration tests for S3/GCS/ADLS ingest. These rely on external endpoints
//! (either real clouds or emulators) provided via environment variables.
//! Tests are skipped unless RUN_CLOUD_TESTS=1 is set and the necessary
//! per-backend env vars are present.

use std::sync::Arc;

use diskgraph_index::builder::{ShardBuildParams, ShardBuilder};
use diskgraph_index::reader::ShardReader;
use diskgraph_index::streaming::{
    AdlsLocation, EdgeShardInput, GcsLocation, NodeShardInput, ParquetSource, S3Location,
};
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use tempfile::tempdir;
use tokio::runtime::Runtime;

#[derive(Debug, Clone)]
struct EdgeRow {
    src: i64,
    dst: i64,
    dest_sort_key: Option<i64>,
}

#[derive(Debug, Clone)]
struct S3Config {
    bucket: String,
    endpoint: Option<String>,
    region: Option<String>,
    access_key: Option<String>,
    secret_key: Option<String>,
    session_token: Option<String>,
    virtual_hosted: bool,
    allow_http: bool,
}

#[derive(Debug, Clone)]
struct GcsConfig {
    bucket: String,
    endpoint: Option<String>,
    service_account_path: Option<String>,
    service_account_key: Option<String>,
    application_credentials_path: Option<String>,
}

#[derive(Debug, Clone)]
struct AdlsConfig {
    account: String,
    container: String,
    endpoint: Option<String>,
    access_key: Option<String>,
    sas_key: Option<String>,
    bearer_token: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    tenant_id: Option<String>,
    use_emulator: bool,
}

fn node_meta(name: &str, max_id: u64) -> diskgraph_core::metadata::NodeTypeMetadata {
    diskgraph_core::metadata::NodeTypeMetadata {
        name: name.into(),
        max_id,
        shards: vec![],
    }
}

fn relation_meta(
    name: &str,
    src: &str,
    dst: &str,
    dest_sort_key_bits: Option<u8>,
) -> diskgraph_core::metadata::EdgeRelationMetadata {
    diskgraph_core::metadata::EdgeRelationMetadata {
        name: name.into(),
        src_type: src.into(),
        dst_type: dst.into(),
        shards: vec![],
        dest_sort_key_bits,
    }
}

fn write_nodes(path: &std::path::Path, ids: &[i64]) -> anyhow::Result<()> {
    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;

    let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let batch = arrow_array::RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef],
    )?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_edges(
    path: &std::path::Path,
    edges: &[EdgeRow],
    with_sort_key: bool,
) -> anyhow::Result<()> {
    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;

    let mut fields = vec![
        Field::new("src_id", DataType::Int64, false),
        Field::new("dst_id", DataType::Int64, false),
    ];
    if with_sort_key {
        fields.push(Field::new("dest_sort_key", DataType::Int64, false));
    }
    let schema = Schema::new(fields);
    let src: Vec<i64> = edges.iter().map(|e| e.src).collect();
    let dst: Vec<i64> = edges.iter().map(|e| e.dst).collect();
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(src)) as ArrayRef,
        Arc::new(Int64Array::from(dst)) as ArrayRef,
    ];
    if with_sort_key {
        let dsk: Vec<i64> = edges
            .iter()
            .map(|e| e.dest_sort_key.unwrap_or_default())
            .collect();
        cols.push(Arc::new(Int64Array::from(dsk)) as ArrayRef);
    }
    let batch = arrow_array::RecordBatch::try_new(Arc::new(schema.clone()), cols)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn decode_relation(
    reader: &ShardReader,
    node_idx: u64,
    relation_idx: usize,
) -> Vec<(u64, Option<u64>)> {
    let view = reader.edges_for_node(node_idx).unwrap();
    let segments = view.relation_segments().unwrap();
    let seg = &segments[relation_idx];
    let mut out = Vec::new();
    if seg.count == 0 {
        return out;
    }
    let mut buf = [0u8; 16];
    for chunk in seg.data.chunks(seg.layout.entry_bytes) {
        buf[..seg.layout.entry_bytes].copy_from_slice(chunk);
        let val = u128::from_le_bytes(buf);
        let dst = (val & seg.layout.dst_mask as u128) as u64;
        let sort = seg
            .layout
            .dest_sort_key_bits
            .map(|bits| (val >> seg.layout.dst_bits as u128) & ((1u128 << bits) - 1))
            .map(|v| v as u64);
        out.push((dst, sort));
        buf.fill(0);
    }
    out
}

/// Helper to upload a local file to an ObjectStore and return the (store, location).
fn upload_to_store(
    rt: &Runtime,
    store: Arc<dyn ObjectStore>,
    bucket_key: &str,
    local_path: &std::path::Path,
) -> (Arc<dyn ObjectStore>, ObjectPath) {
    let location = ObjectPath::from(bucket_key);
    let bytes = std::fs::read(local_path).expect("read local parquet");
    rt.block_on(async {
        store
            .put(&location, bytes.into())
            .await
            .expect("upload to store (ensure bucket/container exists and credentials are valid)");
    });
    (store, location)
}

/// Helper to upload to a fake GCS emulator using the JSON API (uploadType=media).
fn upload_to_gcs_emulator(
    rt: &Runtime,
    endpoint: &str,
    bucket: &str,
    key: &str,
    local_path: &std::path::Path,
) {
    let bytes = std::fs::read(local_path).expect("read local parquet");
    let base = endpoint.trim_end_matches('/');
    let url = format!(
        "{}/upload/storage/v1/b/{}/o?uploadType=media&name={}",
        base, bucket, key
    );
    rt.block_on(async {
        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .expect("build reqwest client");
        client
            .post(&url)
            .header("Content-Type", "application/octet-stream")
            .body(bytes)
            .send()
            .await
            .expect("upload to fake GCS (ensure bucket exists and endpoint reachable)")
            .error_for_status()
            .expect("fake GCS upload returned error");
    });
}

fn s3_config_from_env() -> Option<S3Config> {
    if std::env::var("RUN_CLOUD_TESTS").ok()?.as_str() != "1" {
        return None;
    }
    let bucket = std::env::var("S3_BUCKET").ok()?;
    let allow_http = std::env::var("S3_ALLOW_HTTP")
        .or_else(|_| std::env::var("AWS_ALLOW_HTTP"))
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    Some(S3Config {
        bucket,
        endpoint: std::env::var("S3_ENDPOINT").ok(),
        region: std::env::var("S3_REGION").ok(),
        access_key: std::env::var("S3_ACCESS_KEY").ok(),
        secret_key: std::env::var("S3_SECRET_KEY").ok(),
        session_token: std::env::var("S3_SESSION_TOKEN").ok(),
        virtual_hosted: std::env::var("S3_VIRTUAL_HOSTED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false),
        allow_http,
    })
}

fn gcs_config_from_env() -> Option<GcsConfig> {
    if std::env::var("RUN_CLOUD_TESTS").ok()?.as_str() != "1" {
        return None;
    }
    let bucket = std::env::var("GCS_BUCKET").ok()?;
    Some(GcsConfig {
        bucket,
        endpoint: std::env::var("GCS_ENDPOINT").ok(),
        service_account_path: std::env::var("GCS_SERVICE_ACCOUNT_PATH").ok(),
        service_account_key: std::env::var("GCS_SERVICE_ACCOUNT_KEY").ok(),
        application_credentials_path: std::env::var("GCS_APPLICATION_CREDENTIALS_PATH").ok(),
    })
}

fn adls_config_from_env() -> Option<AdlsConfig> {
    if std::env::var("RUN_CLOUD_TESTS").ok()?.as_str() != "1" {
        return None;
    }
    let account = std::env::var("AZURE_ACCOUNT").ok()?;
    let container = std::env::var("AZURE_CONTAINER").ok()?;
    Some(AdlsConfig {
        account,
        container,
        endpoint: std::env::var("AZURE_ENDPOINT").ok(),
        access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
        sas_key: std::env::var("AZURE_SAS_KEY").ok(),
        bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
        client_id: std::env::var("AZURE_CLIENT_ID").ok(),
        client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
        tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
        use_emulator: std::env::var("AZURE_USE_EMULATOR")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false),
    })
}

#[test]
fn s3_ingest_round_trip() {
    let Some(cfg) = s3_config_from_env() else {
        if std::env::var("RUN_CLOUD_TESTS").ok().as_deref() == Some("1") {
            panic!("RUN_CLOUD_TESTS=1 but S3_* env vars not set; set at least S3_BUCKET (and S3_ENDPOINT/creds for emulator or cloud)");
        } else {
            eprintln!("skipping s3_ingest_round_trip; set RUN_CLOUD_TESTS=1 and S3_* env vars");
            return;
        }
    };
    let rt = Runtime::new().unwrap();
    let mut builder = object_store::aws::AmazonS3Builder::new()
        .with_bucket_name(&cfg.bucket)
        .with_virtual_hosted_style_request(cfg.virtual_hosted)
        .with_allow_http(cfg.allow_http);
    if let Some(ep) = &cfg.endpoint {
        builder = builder.with_endpoint(ep);
    }
    if let Some(region) = &cfg.region {
        builder = builder.with_region(region);
    }
    if let Some(ak) = &cfg.access_key {
        builder = builder.with_access_key_id(ak);
    }
    if let Some(sk) = &cfg.secret_key {
        builder = builder.with_secret_access_key(sk);
    }
    if let Some(token) = &cfg.session_token {
        builder = builder.with_token(token);
    }
    let store: Arc<dyn ObjectStore> = Arc::new(
        builder
            .build()
            .expect("failed to build S3 object store (check endpoint/creds)"),
    );

    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0, 1]).unwrap();
    write_edges(
        &edge_path,
        &[
            EdgeRow {
                src: 0,
                dst: 1,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 1,
                dst: 0,
                dest_sort_key: None,
            },
        ],
        false,
    )
    .unwrap();

    let (_s, node_loc) = upload_to_store(&rt, store.clone(), "nodes.parquet", &node_path);
    let (_s2, edge_loc) = upload_to_store(&rt, store.clone(), "edges.parquet", &edge_path);

    let node_meta = node_meta("User", 10);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::S3(S3Location {
                    bucket: cfg.bucket.clone(),
                    key: node_loc.to_string(),
                    region: cfg.region.clone(),
                    endpoint: cfg.endpoint.clone(),
                    access_key_id: cfg.access_key.clone(),
                    secret_access_key: cfg.secret_key.clone(),
                    session_token: cfg.session_token.clone(),
                    virtual_hosted_style_request: Some(cfg.virtual_hosted),
                }),
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::S3(S3Location {
                    bucket: cfg.bucket.clone(),
                    key: edge_loc.to_string(),
                    region: cfg.region.clone(),
                    endpoint: cfg.endpoint.clone(),
                    access_key_id: cfg.access_key.clone(),
                    secret_access_key: cfg.secret_key.clone(),
                    session_token: cfg.session_token.clone(),
                    virtual_hosted_style_request: Some(cfg.virtual_hosted),
                }),
                has_dest_sort_key: false,
            }],
        )
        .expect("S3 build");

    let reader = ShardReader::open(&out, built.layouts.clone()).unwrap();
    let decoded = decode_relation(&reader, 0, 0);
    assert_eq!(decoded, vec![(1, None)]);
}

#[test]
fn gcs_ingest_round_trip() {
    // If RUN_CLOUD_TESTS=1 is set but GCS_* vars are missing, fail fast
    let Some(cfg) = gcs_config_from_env() else {
        if std::env::var("RUN_CLOUD_TESTS").ok().as_deref() == Some("1") {
            panic!("RUN_CLOUD_TESTS=1 but GCS_* env vars not set; set at least GCS_BUCKET (and optionally GCS_ENDPOINT/creds)");
        } else {
            eprintln!("skipping gcs_ingest_round_trip; set RUN_CLOUD_TESTS=1 and GCS_* env vars");
            return;
        }
    };
    let rt = Runtime::new().unwrap();
    let mut builder =
        object_store::gcp::GoogleCloudStorageBuilder::new().with_bucket_name(&cfg.bucket);
    if let Some(sa_path) = &cfg.service_account_path {
        builder = builder.with_service_account_path(sa_path);
    } else if let Some(sa_key) = &cfg.service_account_key {
        builder = builder.with_service_account_key(sa_key);
    } else if let Some(app_creds) = &cfg.application_credentials_path {
        builder = builder.with_application_credentials(app_creds);
    } else if let Some(ep) = &cfg.endpoint {
        // For emulator: use a minimal service account blob that disables OAuth and points to the custom base URL.
        let sa = format!(
            r#"{{"private_key":"private_key","private_key_id":"fake","client_email":"fake@example.com","disable_oauth":true,"gcs_base_url":"{ep}"}}"#
        );
        builder = builder.with_service_account_key(sa);
    } else {
        panic!("GCS_ENDPOINT must be set for gcs_ingest_round_trip (emulator or real endpoint)");
    }
    let _store: Arc<dyn ObjectStore> = Arc::new(
        builder
            .build()
            .expect("failed to build GCS object store (check endpoint/creds)"),
    );

    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0]).unwrap();
    write_edges(
        &edge_path,
        &[EdgeRow {
            src: 0,
            dst: 42,
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();

    let node_key = "nodes.parquet".to_string();
    let edge_key = "edges.parquet".to_string();
    let ep = cfg
        .endpoint
        .as_ref()
        .expect("GCS_ENDPOINT must be set for emulator uploads");
    upload_to_gcs_emulator(&rt, ep, &cfg.bucket, &node_key, &node_path);
    upload_to_gcs_emulator(&rt, ep, &cfg.bucket, &edge_key, &edge_path);
    let node_loc = ObjectPath::from(node_key);
    let edge_loc = ObjectPath::from(edge_key);
    let node_meta = node_meta("User", 100);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::Gcs(GcsLocation {
                    bucket: cfg.bucket.clone(),
                    key: node_loc.to_string(),
                    service_account_path: cfg.service_account_path.clone(),
                    service_account_key: cfg.service_account_key.clone(),
                    application_credentials_path: cfg.application_credentials_path.clone(),
                    endpoint: cfg.endpoint.clone(),
                }),
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::Gcs(GcsLocation {
                    bucket: cfg.bucket.clone(),
                    key: edge_loc.to_string(),
                    service_account_path: cfg.service_account_path.clone(),
                    service_account_key: cfg.service_account_key.clone(),
                    application_credentials_path: cfg.application_credentials_path.clone(),
                    endpoint: cfg.endpoint.clone(),
                }),
                has_dest_sort_key: false,
            }],
        )
        .expect("GCS build");

    let reader = ShardReader::open(&out, built.layouts.clone()).unwrap();
    let decoded = decode_relation(&reader, 0, 0);
    assert_eq!(decoded, vec![(42, None)]);
}

#[test]
fn adls_ingest_round_trip() {
    let Some(cfg) = adls_config_from_env() else {
        if std::env::var("RUN_CLOUD_TESTS").ok().as_deref() == Some("1") {
            panic!("RUN_CLOUD_TESTS=1 but AZURE_* env vars not set; set AZURE_ACCOUNT, AZURE_CONTAINER, and either AZURE_ACCESS_KEY/SAS or emulator vars");
        } else {
            eprintln!("skipping adls_ingest_round_trip; set RUN_CLOUD_TESTS=1 and AZURE_* env vars");
            return;
        }
    };
    let rt = Runtime::new().unwrap();
    let allow_http = cfg
        .endpoint
        .as_deref()
        .map(|e| e.starts_with("http://"))
        .unwrap_or(cfg.use_emulator);
    let mut builder = object_store::azure::MicrosoftAzureBuilder::new()
        .with_account(&cfg.account)
        .with_container_name(&cfg.container)
        .with_use_emulator(cfg.use_emulator)
        .with_allow_http(allow_http);
    if let Some(ep) = &cfg.endpoint {
        builder = builder.with_endpoint(ep.to_string());
    }
    if let Some(access_key) = &cfg.access_key {
        builder = builder.with_access_key(access_key);
    }
    if let Some(sas) = &cfg.sas_key {
        builder = builder.with_config(object_store::azure::AzureConfigKey::SasKey, sas);
    }
    if let Some(bearer) = &cfg.bearer_token {
        builder = builder.with_bearer_token_authorization(bearer);
    }
    if let Some(client_id) = &cfg.client_id {
        builder = builder.with_client_id(client_id);
    }
    if let Some(client_secret) = &cfg.client_secret {
        builder = builder.with_client_secret(client_secret);
    }
    if let Some(tenant_id) = &cfg.tenant_id {
        builder = builder.with_tenant_id(tenant_id);
    }
    let store: Arc<dyn ObjectStore> = Arc::new(
        builder
            .build()
            .expect("failed to build Azure object store (check endpoint/creds)"),
    );

    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0, 1]).unwrap();
    write_edges(
        &edge_path,
        &[EdgeRow {
            src: 1,
            dst: 7,
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();

    let (_s, node_loc) = upload_to_store(&rt, store.clone(), "nodes.parquet", &node_path);
    let (_s2, edge_loc) = upload_to_store(&rt, store.clone(), "edges.parquet", &edge_path);

    let node_meta = node_meta("User", 10);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::Adls(AdlsLocation {
                    account: cfg.account.clone(),
                    container: cfg.container.clone(),
                    key: node_loc.to_string(),
                    access_key: cfg.access_key.clone(),
                    sas_key: cfg.sas_key.clone(),
                    bearer_token: cfg.bearer_token.clone(),
                    client_id: cfg.client_id.clone(),
                    client_secret: cfg.client_secret.clone(),
                    tenant_id: cfg.tenant_id.clone(),
                    endpoint: cfg.endpoint.clone(),
                    use_emulator: Some(cfg.use_emulator),
                }),
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::Adls(AdlsLocation {
                    account: cfg.account.clone(),
                    container: cfg.container.clone(),
                    key: edge_loc.to_string(),
                    access_key: cfg.access_key.clone(),
                    sas_key: cfg.sas_key.clone(),
                    bearer_token: cfg.bearer_token.clone(),
                    client_id: cfg.client_id.clone(),
                    client_secret: cfg.client_secret.clone(),
                    tenant_id: cfg.tenant_id.clone(),
                    endpoint: cfg.endpoint.clone(),
                    use_emulator: Some(cfg.use_emulator),
                }),
                has_dest_sort_key: false,
            }],
        )
        .expect("ADLS build");

    let reader = ShardReader::open(&out, built.layouts.clone()).unwrap();
    let decoded = decode_relation(&reader, 1, 0);
    assert_eq!(decoded, vec![(7, None)]);
}
