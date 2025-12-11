//! Interactive shard explorer for diskgraph-index.
//!
//! Supports local shards or remote shards on S3/GCS/ADLS by streaming Parquet
//! shards through `ShardBuilder` into a temp cache (derived from the remote
//! path) and opening a `ShardReader`. Provides a simple REPL for inspecting
//! edge lists by node id.

use anyhow::{anyhow, Context, Result};
use aws_config;
use aws_credential_types::provider::ProvideCredentials;
use aws_types::region::Region;
use diskgraph_core::metadata::{DatasetMetadata, EdgeRelationMetadata, NodeTypeMetadata};
use diskgraph_index::builder::{compute_relation_layouts, ShardBuildParams, ShardBuilder};
use diskgraph_index::reader::ShardReader;
use diskgraph_index::streaming::{EdgeShardInput, NodeShardInput, ParquetSource};
use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::{path::Path as ObjectPath, ObjectStore};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[derive(Clone, Copy, Debug)]
enum RemoteKind {
    S3,
    Gcs,
    Adls,
}

#[derive(Clone, Debug)]
enum Source {
    Local(PathBuf),
    Remote {
        kind: RemoteKind,
        prefix: String, // key prefix without scheme/bucket
    },
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let source_str = args
        .next()
        .ok_or_else(|| anyhow!("usage: explore <source> (local path or s3:// gs:// az://)"))?;
    let source = parse_source(&source_str)?;
    let rt = Runtime::new().context("init tokio runtime")?;
    let tmp = tempfile::tempdir().context("create tempdir")?;
    let (metadata, shard_dir) = match &source {
        Source::Local(p) => {
            let metadata: DatasetMetadata = serde_json::from_slice(
                &fs::read(p.join("metadata.json")).context("read metadata.json")?,
            )
            .context("parse metadata.json")?;
            (metadata, p.clone())
        }
        Source::Remote { kind, prefix } => {
            let store = build_store(*kind)?;
            let (metadata, shard_dir) =
                build_from_remote(&rt, *kind, prefix, &store, tmp.path())?;
            (metadata, shard_dir)
        }
    };
    let (layouts, _node_types) = relation_layouts(&metadata)?;
    let reader = ShardReader::open(&shard_dir, layouts.clone()).context("open shard reader")?;

    println!("Loaded shard from {}", source_str);
    print_header(&metadata, &reader);
    repl(&metadata.edge_relations, &reader)?;
    Ok(())
}

fn relation_layouts(
    metadata: &DatasetMetadata,
) -> Result<(Arc<Vec<diskgraph_index::layout::RelationLayout>>, Vec<NodeTypeMetadata>)> {
    let layouts = compute_relation_layouts(&metadata.edge_relations, &metadata.node_types)
        .context("compute relation layouts")?;
    Ok((Arc::new(layouts), metadata.node_types.clone()))
}

fn build_from_remote(
    rt: &Runtime,
    kind: RemoteKind,
    prefix: &str,
    store: &Arc<dyn ObjectStore>,
    cache_root: &Path,
) -> Result<(DatasetMetadata, PathBuf)> {
    // Cache path derived from prefix.
    let safe_prefix = prefix.replace(['/', ':'], "_");
    let cache_dir = cache_root.join(safe_prefix);
    let offsets = cache_dir.join("offsets.bin");
    let edges = cache_dir.join("edges.bin");
    let meta_path = cache_dir.join("metadata.json");
    if offsets.exists() && edges.exists() && meta_path.exists() {
        let metadata: DatasetMetadata = serde_json::from_slice(
            &fs::read(&meta_path).context("read cached metadata")?,
        )
        .context("parse cached metadata")?;
        return Ok((metadata, cache_dir));
    }
    fs::create_dir_all(&cache_dir)?;
    // Download metadata, then build using remote Parquet sources.
    let loc = if prefix.is_empty() {
        ObjectPath::from("metadata.json")
    } else {
        ObjectPath::from(format!("{}/{}", prefix.trim_end_matches('/'), "metadata.json"))
    };
    let meta_obj = rt
        .block_on(store.get(&loc))
        .with_context(|| format!("download metadata.json from {prefix}"))?;
    let meta_bytes = rt
        .block_on(meta_obj.bytes())
        .context("buffer metadata.json")?;
    fs::write(&meta_path, &meta_bytes)?;
    let metadata: DatasetMetadata =
        serde_json::from_slice(&meta_bytes).context("parse metadata.json")?;

    // Build using streaming Parquet sources (first shard per type/relation).
    let node_inputs: Vec<NodeShardInput> = metadata
        .node_types
        .iter()
        .map(|nt| {
            let shard = nt
                .shards
                .get(0)
                .ok_or_else(|| anyhow!("no node shard found for {}", nt.name))?;
            let key = format!("{}/{}", prefix.trim_end_matches('/'), shard.path);
            let parquet = match kind {
                RemoteKind::S3 => {
                    let auth = s3_auth()?;
                    ParquetSource::S3(diskgraph_index::streaming::S3Location {
                        bucket: auth.bucket,
                        key,
                        region: Some(auth.region),
                        endpoint: auth.endpoint,
                        access_key_id: Some(auth.access_key_id),
                        secret_access_key: Some(auth.secret_access_key),
                        session_token: auth.session_token,
                        virtual_hosted_style_request: Some(auth.virtual_hosted),
                    })
                }
                RemoteKind::Gcs => ParquetSource::Gcs(diskgraph_index::streaming::GcsLocation {
                    bucket: std::env::var("GCS_BUCKET").unwrap_or_default(),
                    key,
                    service_account_path: None,
                    service_account_key: None,
                    application_credentials_path: None,
                    endpoint: None,
                }),
                RemoteKind::Adls => ParquetSource::Adls(diskgraph_index::streaming::AdlsLocation {
                    account: std::env::var("AZURE_ACCOUNT").unwrap_or_default(),
                    container: std::env::var("AZURE_CONTAINER").unwrap_or_default(),
                    key,
                    access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
                    sas_key: std::env::var("AZURE_SAS_KEY").ok(),
                    bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
                    client_id: std::env::var("AZURE_CLIENT_ID").ok(),
                    client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
                    tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
                    endpoint: std::env::var("AZURE_ENDPOINT").ok(),
                    use_emulator: None,
                }),
            };
            Ok(NodeShardInput {
                node_type: nt.name.clone(),
                shard_index: 0,
                parquet,
            })
        })
        .collect::<Result<_>>()?;

    let edge_inputs: Vec<EdgeShardInput> = metadata
        .edge_relations
        .iter()
        .map(|rel| {
            let shard = rel
                .shards
                .get(0)
                .ok_or_else(|| anyhow!("no edge shard found for {}", rel.name))?;
            let key = format!("{}/{}", prefix.trim_end_matches('/'), shard.path);
            let parquet = match kind {
                RemoteKind::S3 => {
                    let auth = s3_auth()?;
                    ParquetSource::S3(diskgraph_index::streaming::S3Location {
                        bucket: auth.bucket,
                        key,
                        region: Some(auth.region),
                        endpoint: auth.endpoint,
                        access_key_id: Some(auth.access_key_id),
                        secret_access_key: Some(auth.secret_access_key),
                        session_token: auth.session_token,
                        virtual_hosted_style_request: Some(auth.virtual_hosted),
                    })
                }
                RemoteKind::Gcs => ParquetSource::Gcs(diskgraph_index::streaming::GcsLocation {
                    bucket: std::env::var("GCS_BUCKET").unwrap_or_default(),
                    key,
                    service_account_path: None,
                    service_account_key: None,
                    application_credentials_path: None,
                    endpoint: None,
                }),
                RemoteKind::Adls => ParquetSource::Adls(diskgraph_index::streaming::AdlsLocation {
                    account: std::env::var("AZURE_ACCOUNT").unwrap_or_default(),
                    container: std::env::var("AZURE_CONTAINER").unwrap_or_default(),
                    key,
                    access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
                    sas_key: std::env::var("AZURE_SAS_KEY").ok(),
                    bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
                    client_id: std::env::var("AZURE_CLIENT_ID").ok(),
                    client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
                    tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
                    endpoint: std::env::var("AZURE_ENDPOINT").ok(),
                    use_emulator: None,
                }),
            };
            Ok(EdgeShardInput {
                relation: rel.name.clone(),
                source_shard_index: 0,
                parquet,
                has_dest_sort_key: rel.dest_sort_key_bits.is_some(),
            })
        })
        .collect::<Result<_>>()?;
    let shard_dir = cache_dir.clone();
    let params = ShardBuildParams {
        output_dir: shard_dir.clone(),
        node_meta: metadata.node_types[0].clone(), // assumes single shard/node_type
        node_types: metadata.node_types.clone(),
        relation_metas: metadata.edge_relations.clone(),
        shard_index: 0,
    };
    let builder = ShardBuilder::new(params);
    builder
        .build(&node_inputs[0], &edge_inputs)
        .context("build shard from parquet")?;
    fs::write(&meta_path, serde_json::to_vec_pretty(&metadata)?)?;
    Ok((metadata, shard_dir))
}

fn repl(relations: &[EdgeRelationMetadata], reader: &ShardReader) -> Result<()> {
    let mut rl = DefaultEditor::new().context("init editor")?;
    println!("Commands: info | node <id> | quit");
    loop {
        match rl.readline("explore> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                rl.add_history_entry(trimmed).ok();
                if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
                    break;
                }
                if trimmed.eq_ignore_ascii_case("info") {
                    print_info(relations, reader);
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("node ") {
                    match rest.trim().parse::<u64>() {
                        Ok(id) => show_node(relations, reader, id)?,
                        Err(_) => println!("Invalid node id: {rest}"),
                    }
                    continue;
                }
                println!("Unknown command. Use: info | node <id> | quit");
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

fn print_header(metadata: &DatasetMetadata, reader: &ShardReader) {
    println!(
        "Node types: {}; Relations: {}; Nodes in shard: {}",
        metadata.node_types.len(),
        metadata.edge_relations.len(),
        reader.header().nodes
    );
    for (idx, rel) in metadata.edge_relations.iter().enumerate() {
        let sort = rel
            .dest_sort_key_bits
            .map(|b| format!("sort_bits={b}"))
            .unwrap_or_else(|| "no sort key".into());
        println!("  [{}] {} ({} -> {}), {}", idx, rel.name, rel.src_type, rel.dst_type, sort);
    }
}

fn print_info(relations: &[EdgeRelationMetadata], reader: &ShardReader) {
    println!(
        "Nodes: {}, relations: {}",
        reader.header().nodes,
        relations.len()
    );
    print_header(
        &DatasetMetadata {
            dataset_name: "".into(),
            node_types: vec![],
            edge_relations: relations.to_vec(),
        },
        reader,
    );
}

fn show_node(relations: &[EdgeRelationMetadata], reader: &ShardReader, node: u64) -> Result<()> {
    let view = match reader.edges_for_node(node) {
        Ok(v) => v,
        Err(e) => {
            println!("Error reading node {}: {}", node, e);
            return Ok(());
        }
    };
    let segments = match view.relation_segments() {
        Ok(s) => s,
        Err(e) => {
            println!("Error decoding relations: {}", e);
            return Ok(());
        }
    };
    println!("Node {node}:");
    for (idx, seg) in segments.iter().enumerate() {
        let rel = &relations[idx];
        println!(
            "  [{}] {}: count={}",
            idx,
            rel.name,
            seg.count
        );
        if seg.count > 0 {
            let mut buf = [0u8; 16];
            let mut rows = Vec::new();
            for chunk in seg.data.chunks(seg.layout.entry_bytes).take(10) {
                buf[..seg.layout.entry_bytes].copy_from_slice(chunk);
                let val = u128::from_le_bytes(buf);
                let dst = (val & seg.layout.dst_mask as u128) as u64;
                let sort = seg
                    .layout
                    .dest_sort_key_bits
                    .map(|bits| (val >> seg.layout.dst_bits as u128) & ((1u128 << bits) - 1))
                    .map(|v| v as u64);
                rows.push((dst, sort));
                buf.fill(0);
            }
            println!("    first {} edges: {:?}", rows.len(), rows);
        }
    }
    Ok(())
}

fn parse_source(raw: &str) -> Result<Source> {
    if raw.starts_with("s3://") {
        let (bucket, key) = parse_s3_url(raw)?;
        std::env::set_var("S3_BUCKET", &bucket);
        Ok(Source::Remote {
            kind: RemoteKind::S3,
            prefix: key,
        })
    } else if raw.starts_with("gs://") {
        let (bucket, key) = parse_gs_url(raw)?;
        std::env::set_var("GCS_BUCKET", &bucket);
        Ok(Source::Remote {
            kind: RemoteKind::Gcs,
            prefix: key,
        })
    } else if raw.starts_with("az://") {
        let (account, container, key) = parse_az_url(raw)?;
        std::env::set_var("AZURE_ACCOUNT", &account);
        std::env::set_var("AZURE_CONTAINER", &container);
        Ok(Source::Remote {
            kind: RemoteKind::Adls,
            prefix: key,
        })
    } else {
        Ok(Source::Local(PathBuf::from(raw)))
    }
}

fn build_store(kind: RemoteKind) -> Result<Arc<dyn ObjectStore>> {
    match kind {
        RemoteKind::S3 => {
            let auth = s3_auth()?;
            let mut builder = AmazonS3Builder::new()
                .with_bucket_name(&auth.bucket)
                .with_region(&auth.region)
                .with_virtual_hosted_style_request(auth.virtual_hosted)
                .with_allow_http(auth.allow_http);
            if let Some(endpoint) = &auth.endpoint {
                builder = builder.with_endpoint(endpoint);
            }
            builder = builder
                .with_access_key_id(auth.access_key_id.clone())
                .with_secret_access_key(auth.secret_access_key.clone());
            if let Some(token) = &auth.session_token {
                builder = builder.with_token(token);
            }
            Ok(Arc::new(builder.build()?))
        }
        RemoteKind::Gcs => {
            let bucket = std::env::var("GCS_BUCKET").context("set GCS_BUCKET or include bucket in url")?;
            let builder = GoogleCloudStorageBuilder::from_env().with_bucket_name(&bucket);
            Ok(Arc::new(builder.build()?))
        }
        RemoteKind::Adls => {
            let account =
                std::env::var("AZURE_ACCOUNT").context("set AZURE_ACCOUNT for az:// sources")?;
            let container =
                std::env::var("AZURE_CONTAINER").context("set AZURE_CONTAINER for az:// sources")?;
            let allow_http = std::env::var("AZURE_ALLOW_HTTP")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let mut builder = MicrosoftAzureBuilder::new()
                .with_account(&account)
                .with_container_name(&container)
                .with_allow_http(allow_http);
            if let Ok(ep) = std::env::var("AZURE_ENDPOINT") {
                builder = builder.with_endpoint(ep);
            }
            if let Ok(access_key) = std::env::var("AZURE_ACCESS_KEY") {
                builder = builder.with_access_key(access_key);
            }
            if let Ok(sas) = std::env::var("AZURE_SAS_KEY") {
                builder = builder.with_config(object_store::azure::AzureConfigKey::SasKey, sas);
            }
            if let Ok(bearer) = std::env::var("AZURE_BEARER_TOKEN") {
                builder = builder.with_bearer_token_authorization(bearer);
            }
            if let Ok(client_id) = std::env::var("AZURE_CLIENT_ID") {
                builder = builder.with_client_id(client_id);
            }
            if let Ok(client_secret) = std::env::var("AZURE_CLIENT_SECRET") {
                builder = builder.with_client_secret(client_secret);
            }
            if let Ok(tenant_id) = std::env::var("AZURE_TENANT_ID") {
                builder = builder.with_tenant_id(tenant_id);
            }
            Ok(Arc::new(builder.build()?))
        }
    }
}

fn parse_s3_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw)
        .map_err(|e| anyhow::anyhow!("invalid s3 url {raw}: {e}"))?;
    if url.scheme() != "s3" {
        anyhow::bail!("expected s3:// url, got {raw}");
    }
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket in s3 url"))?
        .to_string();
    let mut key = url.path().trim_start_matches('/').to_string();
    if key.ends_with('/') {
        key.pop();
    }
    Ok((bucket, key))
}

fn parse_gs_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw)
        .map_err(|e| anyhow::anyhow!("invalid gs url {raw}: {e}"))?;
    if url.scheme() != "gs" {
        anyhow::bail!("expected gs:// url, got {raw}");
    }
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket in gs url"))?
        .to_string();
    let mut key = url.path().trim_start_matches('/').to_string();
    if key.ends_with('/') {
        key.pop();
    }
    Ok((bucket, key))
}

fn parse_az_url(raw: &str) -> Result<(String, String, String)> {
    let url = url::Url::parse(raw)
        .map_err(|e| anyhow::anyhow!("invalid az url {raw}: {e}"))?;
    if url.scheme() != "az" {
        anyhow::bail!("expected az:// url, got {raw}");
    }
    let account = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("missing account in az url"))?
        .to_string();
    let mut segments = url
        .path()
        .trim_start_matches('/')
        .split('/')
        .filter(|s| !s.is_empty());
    let container = segments
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing container in az url"))?
        .to_string();
    let mut key = segments.collect::<Vec<_>>().join("/");
    if key.ends_with('/') {
        key.pop();
    }
    Ok((account, container, key))
}

struct S3Auth {
    bucket: String,
    region: String,
    endpoint: Option<String>,
    allow_http: bool,
    virtual_hosted: bool,
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

fn s3_auth() -> Result<S3Auth> {
    let bucket = std::env::var("S3_BUCKET").context("set S3_BUCKET or include bucket in url")?;
    let allow_http = std::env::var("S3_ALLOW_HTTP")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let endpoint = std::env::var("S3_ENDPOINT").ok();
    let rt = Runtime::new()?;
    let profile = std::env::var("S3_PROFILE")
        .ok()
        .or_else(|| std::env::var("AWS_PROFILE").ok());
    let aws_conf = rt.block_on(async {
        let mut loader = aws_config::from_env();
        if let Some(p) = profile.clone() {
            loader = loader.profile_name(p);
        }
        if let Ok(region) = std::env::var("S3_REGION") {
            loader = loader.region(Region::new(region));
        }
        loader.load().await
    });
    let creds_provider = aws_conf
        .credentials_provider()
        .ok_or_else(|| anyhow!("No AWS credentials provider found; ensure AWS_PROFILE/SSO is set"))?;
    let creds = rt
        .block_on(async {
            creds_provider
                .provide_credentials()
                .await
                .map_err(|e| anyhow!("Failed to load AWS credentials: {e}"))
        })?;
    let region = std::env::var("S3_REGION")
        .ok()
        .or_else(|| aws_conf.region().map(|r| r.as_ref().to_string()))
        .unwrap_or_else(|| "us-east-1".to_string());
    let virtual_hosted = std::env::var("S3_VIRTUAL_HOSTED")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or_else(|| endpoint.as_ref().map_or(true, |ep| ep.contains("amazonaws.com")));
    let access_key_id = std::env::var("S3_ACCESS_KEY")
        .ok()
        .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
        .unwrap_or_else(|| creds.access_key_id().to_string());
    let secret_access_key = std::env::var("S3_SECRET_KEY")
        .ok()
        .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
        .unwrap_or_else(|| creds.secret_access_key().to_string());
    let session_token = std::env::var("S3_SESSION_TOKEN")
        .ok()
        .or_else(|| creds.session_token().map(|t| t.to_string()));
    Ok(S3Auth {
        bucket,
        region,
        endpoint,
        allow_http,
        virtual_hosted,
        access_key_id,
        secret_access_key,
        session_token,
    })
}
