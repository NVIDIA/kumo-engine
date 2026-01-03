//! Integration tests for S3/GCS/ADLS ingest. These rely on external endpoints
//! provided via environment variables.
//! Tests are skipped unless RUN_CLOUD_TESTS=1 is set and the necessary
//! per-backend env vars are present.

use std::sync::Arc;

use diskgraph_index::builder::{ShardBuildParams, ShardBuilder};
use diskgraph_index::reader::ShardReader;
use diskgraph_index::streaming::{AdlsLocation, EdgeShardInput, GcsLocation, ParquetSource, S3Location};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig as FsAdlsConfig,
    GcsConfig as FsGcsConfig, S3Config as FsS3Config,
};
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use tempfile::tempdir;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::runtime::Runtime;
use url::Url;

#[derive(Debug, Clone)]
struct EdgeRow {
    src: i64,
    dst: i64,
    dest_sort_key: Option<i64>,
}

#[derive(Debug, Clone)]
struct S3Config {
    bucket: String,
    prefix: Option<String>,
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
    prefix: Option<String>,
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
    prefix: Option<String>,
}

fn node_meta(name: &str, max_id: u64) -> diskgraph_core::metadata::NodeTypeMetadata {
    diskgraph_core::metadata::NodeTypeMetadata {
        name: name.into(),
        max_id,
        has_node_data: false,
        node_data_compression: None,
        primary_key_map: None,
        shards: vec![diskgraph_core::metadata::NodeShardMetadata {
            min_id: 0,
            max_id,
            node_data_path: None,
        }],
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
        format: diskgraph_core::metadata::EdgeStorageFormat::Parquet,
    }
}

#[allow(dead_code)]
fn write_edges(
    path: &std::path::Path,
    edges: &[EdgeRow],
    with_sort_key: bool,
) -> anyhow::Result<()> {
    write_edges_batched(path, edges, with_sort_key, edges.len())
}

fn write_edges_batched(
    path: &std::path::Path,
    edges: &[EdgeRow],
    with_sort_key: bool,
    batch_size: usize,
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
    let schema = Arc::new(Schema::new(fields));
    let props = WriterProperties::builder()
        .set_max_row_group_size(batch_size.max(1))
        .build();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
    for chunk in edges.chunks(batch_size.max(1)) {
        let src: Vec<i64> = chunk.iter().map(|e| e.src).collect();
        let dst: Vec<i64> = chunk.iter().map(|e| e.dst).collect();
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(src)) as ArrayRef,
            Arc::new(Int64Array::from(dst)) as ArrayRef,
        ];
        if with_sort_key {
            let dsk: Vec<i64> = chunk
                .iter()
                .map(|e| e.dest_sort_key.unwrap_or_default())
                .collect();
            cols.push(Arc::new(Int64Array::from(dsk)) as ArrayRef);
        }
        let batch = arrow_array::RecordBatch::try_new(schema.clone(), cols)?;
        writer.write(&batch)?;
    }
    writer.close()?;
    Ok(())
}

fn assert_multiple_row_groups(path: &std::path::Path) {
    let file = std::fs::File::open(path).expect("open parquet to inspect row groups");
    let reader = SerializedFileReader::new(file).expect("read parquet metadata");
    let rg = reader.num_row_groups();
    assert!(
        rg > 1,
        "expected parquet at {} to have multiple row groups, saw {}",
        path.display(),
        rg
    );
}

fn complex_dataset() -> (Vec<i64>, Vec<EdgeRow>, Vec<EdgeRow>) {
    // Nodes span > DELTA_CHUNK (64) to force multiple offset chunks.
    let nodes: Vec<i64> = (0..=130).collect();

    // Relation 0: has dest_sort_key, two edges for most nodes, none for tail nodes to cover empty lists.
    let mut follows = Vec::new();
    for src in 0..=110 {
        let high_key = 2000 - (src as i64);
        let low_key = 1000 - (src as i64);
        follows.push(EdgeRow {
            src,
            dst: ((src + 1) % 131) as i64,
            dest_sort_key: Some(high_key),
        });
        follows.push(EdgeRow {
            src,
            dst: ((src + 2) % 131) as i64,
            dest_sort_key: Some(low_key),
        });
    }

    // Relation 1: no dest_sort_key, sparse edges.
    let mut blocks = Vec::new();
    for src in (5..=125).step_by(3) {
        blocks.push(EdgeRow {
            src,
            dst: (200 + src as i64) % 500,
            dest_sort_key: None,
        });
    }
    (nodes, follows, blocks)
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
        // Verify the object is visible (helps catch auth/path issues early).
        store
            .head(&location)
            .await
            .expect("uploaded object not visible; check permissions/prefix");
    });
    (store, location)
}

fn prefix_or_generated(env_prefix: &Option<String>, default_base: &str) -> String {
    if let Some(p) = env_prefix {
        return p.clone();
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}/run-{}", default_base, nanos)
}

fn join_prefix(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), name)
    }
}

fn s3_config_from_env() -> Option<S3Config> {
    if std::env::var("RUN_CLOUD_TESTS").ok()?.as_str() != "1" {
        return None;
    }
    let mut bucket = None;
    let mut prefix = None;
    if let Ok(url) = std::env::var("S3_URL") {
        if let Ok(parsed) = Url::parse(&url) {
            if parsed.scheme() == "s3" {
                bucket = parsed.host_str().map(|s| s.to_string());
                let path = parsed.path().trim_start_matches('/');
                if !path.is_empty() {
                    prefix = Some(path.to_string());
                }
            }
        }
    }
    if bucket.is_none() {
        bucket = std::env::var("S3_BUCKET").ok();
    }
    let bucket = bucket?;
    let allow_http = std::env::var("S3_ALLOW_HTTP")
        .or_else(|_| std::env::var("AWS_ALLOW_HTTP"))
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let endpoint = std::env::var("S3_ENDPOINT").ok();
    let virtual_hosted = std::env::var("S3_VIRTUAL_HOSTED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        // Default to virtual-hosted for real S3; path-style for custom endpoints unless explicitly set.
        .unwrap_or_else(|_| endpoint.as_ref().map_or(true, |ep| ep.contains("amazonaws.com")));
    Some(S3Config {
        bucket,
        prefix,
        endpoint,
        region: std::env::var("S3_REGION").ok(),
        access_key: std::env::var("S3_ACCESS_KEY").ok(),
        secret_key: std::env::var("S3_SECRET_KEY").ok(),
        session_token: std::env::var("S3_SESSION_TOKEN").ok(),
        virtual_hosted,
        allow_http,
    })
}

fn gcs_config_from_env() -> Option<GcsConfig> {
    if std::env::var("RUN_CLOUD_TESTS").ok()?.as_str() != "1" {
        return None;
    }
    let mut bucket = None;
    let mut prefix = None;
    if let Ok(url) = std::env::var("GCS_URL") {
        if let Ok(parsed) = Url::parse(&url) {
            if parsed.scheme() == "gs" {
                bucket = parsed.host_str().map(|s| s.to_string());
                let path = parsed.path().trim_start_matches('/');
                if !path.is_empty() {
                    prefix = Some(path.to_string());
                }
            }
        }
    }
    if bucket.is_none() {
        bucket = std::env::var("GCS_BUCKET").ok();
    }
    let bucket = bucket?;
    Some(GcsConfig {
        bucket,
        prefix,
    })
}

fn adls_config_from_env() -> Option<AdlsConfig> {
    if std::env::var("RUN_CLOUD_TESTS").ok()?.as_str() != "1" {
        return None;
    }
    let mut account = None;
    let mut container = None;
    let mut prefix = None;
    if let Ok(url) = std::env::var("AZURE_URL") {
        if let Ok(parsed) = Url::parse(&url) {
            if parsed.scheme() == "az" {
                account = parsed.host_str().map(|s| s.to_string());
                let mut segments = parsed
                    .path()
                    .trim_start_matches('/')
                    .split('/')
                    .filter(|s| !s.is_empty());
                if let Some(c) = segments.next() {
                    container = Some(c.to_string());
                }
                let rest: String = segments.collect::<Vec<_>>().join("/");
                if !rest.is_empty() {
                    prefix = Some(rest);
                }
            }
        }
    }
    if account.is_none() {
        account = std::env::var("AZURE_ACCOUNT").ok();
    }
    if container.is_none() {
        container = std::env::var("AZURE_CONTAINER").ok();
    }
    let account = account?;
    let container = container?;
    Some(AdlsConfig {
        account,
        container,
        prefix,
        endpoint: std::env::var("AZURE_ENDPOINT").ok(),
        access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
        sas_key: std::env::var("AZURE_SAS_KEY").ok(),
        bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
        client_id: std::env::var("AZURE_CLIENT_ID").ok(),
        client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
        tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
    })
}

#[test]
fn s3_ingest_round_trip() {
    let Some(cfg) = s3_config_from_env() else {
        if std::env::var("RUN_CLOUD_TESTS").ok().as_deref() == Some("1") {
            panic!("RUN_CLOUD_TESTS=1 but S3_* env vars not set; set at least S3_BUCKET (and S3_ENDPOINT/creds for custom endpoints if needed)");
        } else {
            eprintln!("skipping s3_ingest_round_trip; set RUN_CLOUD_TESTS=1 and S3_* env vars");
            return;
        }
    };
    let rt = Runtime::new().unwrap();
    let s3 = build_s3_client(FsS3Config {
        bucket: cfg.bucket.clone(),
        region: cfg.region.clone(),
        endpoint: cfg.endpoint.clone(),
        allow_http: Some(cfg.allow_http),
        virtual_hosted: Some(cfg.virtual_hosted),
        profile: std::env::var("S3_PROFILE")
            .ok()
            .or_else(|| std::env::var("AWS_PROFILE").ok()),
        access_key_id: cfg.access_key.clone(),
        secret_access_key: cfg.secret_key.clone(),
        session_token: cfg.session_token.clone(),
    })
    .expect("failed to build S3 client");
    let store: Arc<dyn ObjectStore> = s3.store.clone();
    let resolved = s3.resolved.clone();

    let dir = tempdir().unwrap();
    let follows_path = dir.path().join("follows.parquet");
    let blocks_path = dir.path().join("blocks.parquet");
    let (_nodes, follows, blocks) = complex_dataset();
    write_edges_batched(&follows_path, &follows, true, 50).unwrap();
    write_edges_batched(&blocks_path, &blocks, false, 20).unwrap();
    assert_multiple_row_groups(&follows_path);
    assert_multiple_row_groups(&blocks_path);

    let prefix = prefix_or_generated(&cfg.prefix, "diskgraph-index/s3");
    let follows_key = join_prefix(&prefix, "follows.parquet");
    let blocks_key = join_prefix(&prefix, "blocks.parquet");
    let (_s2, follows_loc) = upload_to_store(&rt, store.clone(), &follows_key, &follows_path);
    let (_s3, blocks_loc) = upload_to_store(&rt, store.clone(), &blocks_key, &blocks_path);

    let node_meta = node_meta("User", 1000);
    let follows_meta = relation_meta("Follows", "User", "User", Some(16));
    let blocks_meta = relation_meta("Blocks", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![follows_meta.clone(), blocks_meta.clone()],
        shard_index: 0,
        node_data: None,
    });
    let built = builder
        .build(&[
            EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::S3(S3Location {
                    bucket: cfg.bucket.clone(),
                    key: follows_loc.to_string(),
                    region: Some(resolved.region.clone()),
                    endpoint: resolved.endpoint.clone(),
                    access_key_id: Some(resolved.access_key_id.clone()),
                    secret_access_key: Some(resolved.secret_access_key.clone()),
                    session_token: resolved.session_token.clone(),
                    virtual_hosted_style_request: Some(resolved.virtual_hosted),
                }),
                has_dest_sort_key: true,
                format: diskgraph_core::metadata::EdgeStorageFormat::Parquet,
            },
            EdgeShardInput {
                relation: "Blocks".into(),
                source_shard_index: 0,
                parquet: ParquetSource::S3(S3Location {
                    bucket: cfg.bucket.clone(),
                    key: blocks_loc.to_string(),
                    region: Some(resolved.region.clone()),
                    endpoint: resolved.endpoint.clone(),
                    access_key_id: Some(resolved.access_key_id.clone()),
                    secret_access_key: Some(resolved.secret_access_key.clone()),
                    session_token: resolved.session_token.clone(),
                    virtual_hosted_style_request: Some(resolved.virtual_hosted),
                }),
                has_dest_sort_key: false,
                format: diskgraph_core::metadata::EdgeStorageFormat::Parquet,
            },
        ])
        .expect("S3 build");

    let reader = ShardReader::open(&out, built.layouts.clone(), false).unwrap();
    let decoded_follows = decode_relation(&reader, 0, 0);
    assert_eq!(
        decoded_follows,
        vec![(1, Some(2000)), (2, Some(1000))]
    );
    let decoded_blocks = decode_relation(&reader, 8, 1);
    assert_eq!(decoded_blocks, vec![(208, None)]);
    // Tail node has no edges in either relation.
    assert!(decode_relation(&reader, 128, 0).is_empty());
    assert!(decode_relation(&reader, 128, 1).is_empty());
}

#[test]
fn gcs_ingest_round_trip() {
    // If RUN_CLOUD_TESTS=1 is set but GCS_* vars are missing, fail fast
    let Some(cfg) = gcs_config_from_env() else {
        if std::env::var("RUN_CLOUD_TESTS").ok().as_deref() == Some("1") {
            panic!("RUN_CLOUD_TESTS=1 but GCS_* env vars not set; set at least GCS_BUCKET");
        } else {
            eprintln!("skipping gcs_ingest_round_trip; set RUN_CLOUD_TESTS=1 and GCS_* env vars");
            return;
        }
    };
    let rt = Runtime::new().unwrap();
    let gcs = build_gcs_client(FsGcsConfig {
        bucket: cfg.bucket.clone(),
    })
    .expect("failed to build GCS client");
    let store: Arc<dyn ObjectStore> = gcs.store.clone();

    let dir = tempdir().unwrap();
    let follows_path = dir.path().join("follows.parquet");
    let blocks_path = dir.path().join("blocks.parquet");
    let (_nodes, follows, blocks) = complex_dataset();
    write_edges_batched(&follows_path, &follows, true, 50).unwrap();
    write_edges_batched(&blocks_path, &blocks, false, 20).unwrap();
    assert_multiple_row_groups(&follows_path);
    assert_multiple_row_groups(&blocks_path);

    let prefix = prefix_or_generated(&cfg.prefix, "diskgraph-index/gcs");
    let follows_key = join_prefix(&prefix, "follows.parquet");
    let blocks_key = join_prefix(&prefix, "blocks.parquet");
    let (_, follows_loc) = upload_to_store(&rt, store.clone(), &follows_key, &follows_path);
    let (_, blocks_loc) = upload_to_store(&rt, store.clone(), &blocks_key, &blocks_path);
    let node_meta = node_meta("User", 1000);
    let relation_follows = relation_meta("Follows", "User", "User", Some(16));
    let relation_blocks = relation_meta("Blocks", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_follows.clone(), relation_blocks.clone()],
        shard_index: 0,
        node_data: None,
    });
    let built = builder
        .build(&[
            EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::Gcs(GcsLocation {
                    bucket: cfg.bucket.clone(),
                    key: follows_loc.to_string(),
                    service_account_path: None,
                    service_account_key: None,
                    application_credentials_path: None,
                }),
                has_dest_sort_key: true,
                format: diskgraph_core::metadata::EdgeStorageFormat::Parquet,
            },
            EdgeShardInput {
                relation: "Blocks".into(),
                source_shard_index: 0,
                parquet: ParquetSource::Gcs(GcsLocation {
                    bucket: cfg.bucket.clone(),
                    key: blocks_loc.to_string(),
                    service_account_path: None,
                    service_account_key: None,
                    application_credentials_path: None,
                }),
                has_dest_sort_key: false,
                format: diskgraph_core::metadata::EdgeStorageFormat::Parquet,
            },
        ])
        .expect("GCS build");

    let reader = ShardReader::open(&out, built.layouts.clone(), false).unwrap();
    let decoded_follows = decode_relation(&reader, 5, 0);
    assert_eq!(
        decoded_follows,
        vec![(6, Some(1995)), (7, Some(995))]
    );
    let decoded_blocks = decode_relation(&reader, 14, 1);
    assert_eq!(decoded_blocks, vec![(214, None)]);
    assert!(decode_relation(&reader, 128, 0).is_empty());
    assert!(decode_relation(&reader, 128, 1).is_empty());
}

#[test]
fn adls_ingest_round_trip() {
    let Some(cfg) = adls_config_from_env() else {
        if std::env::var("RUN_CLOUD_TESTS").ok().as_deref() == Some("1") {
            panic!("RUN_CLOUD_TESTS=1 but AZURE_* env vars not set; set AZURE_ACCOUNT, AZURE_CONTAINER, and either AZURE_ACCESS_KEY/SAS or other auth vars");
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
        .unwrap_or(false);
    let adls = build_adls_client(FsAdlsConfig {
        account: cfg.account.clone(),
        container: cfg.container.clone(),
        endpoint: cfg.endpoint.clone(),
        allow_http: Some(allow_http),
        access_key: cfg.access_key.clone(),
        sas_key: cfg.sas_key.clone(),
        bearer_token: cfg.bearer_token.clone(),
        client_id: cfg.client_id.clone(),
        client_secret: cfg.client_secret.clone(),
        tenant_id: cfg.tenant_id.clone(),
    })
    .expect("failed to build Azure client");
    let store: Arc<dyn ObjectStore> = adls.store.clone();
    let resolved = adls.resolved.clone();

    let dir = tempdir().unwrap();
    let follows_path = dir.path().join("follows.parquet");
    let blocks_path = dir.path().join("blocks.parquet");
    let (_nodes, follows, blocks) = complex_dataset();
    write_edges_batched(&follows_path, &follows, true, 50).unwrap();
    write_edges_batched(&blocks_path, &blocks, false, 20).unwrap();
    assert_multiple_row_groups(&follows_path);
    assert_multiple_row_groups(&blocks_path);

    let prefix = prefix_or_generated(&cfg.prefix, "diskgraph-index/azure");
    let follows_key = join_prefix(&prefix, "follows.parquet");
    let blocks_key = join_prefix(&prefix, "blocks.parquet");
    let (_s2, follows_loc) = upload_to_store(&rt, store.clone(), &follows_key, &follows_path);
    let (_s3, blocks_loc) = upload_to_store(&rt, store.clone(), &blocks_key, &blocks_path);

    let node_meta = node_meta("User", 1000);
    let follows_meta = relation_meta("Follows", "User", "User", Some(16));
    let blocks_meta = relation_meta("Blocks", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![follows_meta.clone(), blocks_meta.clone()],
        shard_index: 0,
        node_data: None,
    });
    let built = builder
        .build(&[
            EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::Adls(AdlsLocation {
                    account: resolved.account.clone(),
                    container: resolved.container.clone(),
                    key: follows_loc.to_string(),
                    access_key: resolved.access_key.clone(),
                    sas_key: resolved.sas_key.clone(),
                    bearer_token: resolved.bearer_token.clone(),
                    client_id: resolved.client_id.clone(),
                    client_secret: resolved.client_secret.clone(),
                    tenant_id: resolved.tenant_id.clone(),
                    endpoint: resolved.endpoint.clone(),
                }),
                has_dest_sort_key: true,
                format: diskgraph_core::metadata::EdgeStorageFormat::Parquet,
            },
            EdgeShardInput {
                relation: "Blocks".into(),
                source_shard_index: 0,
                parquet: ParquetSource::Adls(AdlsLocation {
                    account: resolved.account.clone(),
                    container: resolved.container.clone(),
                    key: blocks_loc.to_string(),
                    access_key: resolved.access_key.clone(),
                    sas_key: resolved.sas_key.clone(),
                    bearer_token: resolved.bearer_token.clone(),
                    client_id: resolved.client_id.clone(),
                    client_secret: resolved.client_secret.clone(),
                    tenant_id: resolved.tenant_id.clone(),
                    endpoint: resolved.endpoint.clone(),
                }),
                has_dest_sort_key: false,
                format: diskgraph_core::metadata::EdgeStorageFormat::Parquet,
            },
        ])
        .expect("ADLS build");

    let reader = ShardReader::open(&out, built.layouts.clone(), false).unwrap();
    let decoded_follows = decode_relation(&reader, 2, 0);
    assert_eq!(
        decoded_follows,
        vec![(3, Some(1998)), (4, Some(998))]
    );
    let decoded_blocks = decode_relation(&reader, 11, 1);
    assert_eq!(decoded_blocks, vec![(211, None)]);
    assert!(decode_relation(&reader, 128, 0).is_empty());
    assert!(decode_relation(&reader, 128, 1).is_empty());
}
