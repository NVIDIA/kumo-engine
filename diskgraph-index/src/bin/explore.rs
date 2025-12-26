//! Interactive shard explorer for diskgraph-index.
//!
//! Supports local shards or remote shards on S3/GCS/ADLS by streaming Parquet
//! shards through `ShardBuilder` into a temp cache (derived from the remote
//! path) and opening a `ShardReader`. Provides a simple REPL for inspecting
//! edge lists by node id.

use anyhow::{anyhow, Context, Result};
use diskgraph_core::metadata::{DatasetMetadata, EdgeRelationMetadata, NodeTypeMetadata};
use diskgraph_index::builder::{compute_relation_layouts, ShardBuildParams, ShardBuilder};
use diskgraph_index::reader::ShardReader;
use diskgraph_index::streaming::{EdgeShardInput, NodeShardInput, ParquetSource};
use diskgraph_index::layout::{OFFSETS_FILENAME, EDGES_FILENAME};
use object_store::ObjectStore;
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig, S3Config,
};
use indicatif::{ProgressBar, ProgressStyle};
use object_store::path::Path as ObjectPath;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[derive(Clone, Debug)]
enum Source {
    Local(PathBuf),
    S3 { bucket: String, prefix: String },
    Gcs { bucket: String, prefix: String },
    Adls {
        account: String,
        container: String,
        prefix: String,
    },
}

enum RemoteStore {
    S3 {
        bucket: String,
        prefix: String,
        client: diskgraph_fs::S3Client,
    },
    Gcs {
        bucket: String,
        prefix: String,
        client: diskgraph_fs::GcsClient,
    },
    Adls {
        account: String,
        container: String,
        prefix: String,
        client: diskgraph_fs::AdlsClient,
    },
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let source_str = args
        .next()
        .ok_or_else(|| anyhow!("usage: explore <source> (local path or s3:// gs:// az://)"))?;
    let source = parse_source(&source_str)?;
    let rt = Runtime::new().context("init tokio runtime")?;
    let tmp = cache_tempdir().context("create cache tempdir")?;
    let load_start = std::time::Instant::now();
    let (metadata, shard_dir) = match &source {
        Source::Local(p) => {
            if is_indexed_shard(p) {
                let metadata: DatasetMetadata = serde_json::from_slice(
                    &fs::read(p.join("metadata.json")).context("read metadata.json")?,
                )
                .context("parse metadata.json")?;
                (metadata, p.clone())
            } else {
                build_from_local_parquet(p, tmp.path())?
            }
        }
        Source::S3 { bucket, prefix } => {
            let profile = std::env::var("S3_PROFILE")
                .ok()
                .or_else(|| std::env::var("AWS_PROFILE").ok());
            let s3 = build_s3_client(S3Config {
                bucket: bucket.clone(),
                region: std::env::var("S3_REGION").ok(),
                endpoint: std::env::var("S3_ENDPOINT").ok(),
                allow_http: env_flag("S3_ALLOW_HTTP").or_else(|| env_flag("AWS_ALLOW_HTTP")),
                virtual_hosted: env_flag("S3_VIRTUAL_HOSTED"),
                profile,
                access_key_id: None,
                secret_access_key: None,
                session_token: None,
            })?;
            let (metadata, dir) = build_from_remote(
                &rt,
                RemoteStore::S3 {
                    bucket: bucket.clone(),
                    prefix: prefix.clone(),
                    client: s3,
                },
                tmp.path(),
            )?;
            (metadata, dir)
        }
        Source::Gcs { bucket, prefix } => {
            let gcs = build_gcs_client(GcsConfig {
                bucket: bucket.clone(),
            })?;
            let (metadata, dir) = build_from_remote(
                &rt,
                RemoteStore::Gcs {
                    bucket: bucket.clone(),
                    prefix: prefix.clone(),
                    client: gcs,
                },
                tmp.path(),
            )?;
            (metadata, dir)
        }
        Source::Adls {
            account,
            container,
            prefix,
        } => {
            let adls = build_adls_client(AdlsConfig {
                account: account.clone(),
                container: container.clone(),
                endpoint: std::env::var("AZURE_ENDPOINT").ok(),
                allow_http: env_flag("AZURE_ALLOW_HTTP"),
                access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
                sas_key: std::env::var("AZURE_SAS_KEY").ok(),
                bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
                client_id: std::env::var("AZURE_CLIENT_ID").ok(),
                client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
                tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
            })?;
            let (metadata, dir) = build_from_remote(
                &rt,
                RemoteStore::Adls {
                    account: account.clone(),
                    container: container.clone(),
                    prefix: prefix.clone(),
                    client: adls,
                },
                tmp.path(),
            )?;
            (metadata, dir)
        }
    };
    let elapsed = load_start.elapsed().as_secs_f64();
    let (layouts, _node_types) = relation_layouts(&metadata)?;
    let reader = ShardReader::open(&shard_dir, layouts.clone()).context("open shard reader")?;

    println!("Loaded shard from {}", source_str);
    if elapsed > 0.0 {
        let offsets_size = fs::metadata(shard_dir.join(OFFSETS_FILENAME))
            .map(|m| m.len())
            .unwrap_or(0);
        let edges_size = fs::metadata(shard_dir.join(EDGES_FILENAME))
            .map(|m| m.len())
            .unwrap_or(0);
        let bytes_total = offsets_size + edges_size;
        let nodes = reader.header().nodes;
        let edges = reader.edge_count();
        let bps = bytes_total as f64 / elapsed;
        let nps = nodes as f64 / elapsed;
        let eps = edges as f64 / elapsed;
        println!(
            "Load stats: {:.2}s, {:.2} MB, {:.2} MB/s, {:.0} nodes/s, {:.0} edges/s, {:.0} (nodes+edges)/s",
            elapsed,
            bytes_total as f64 / (1024.0 * 1024.0),
            bps / (1024.0 * 1024.0),
            nps,
            eps,
            (nodes + edges) as f64 / elapsed
        );
    }
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
    remote: RemoteStore,
    cache_root: &Path,
) -> Result<(DatasetMetadata, PathBuf)> {
    let cache_spinner = start_spinner("Checking cache");
    let (safe_prefix, store_arc) = {
        match &remote {
            RemoteStore::S3 { bucket, prefix, client } => (
                format!("s3_{}_{}", bucket, prefix).replace(['/', ':'], "_"),
                client.store.clone(),
            ),
            RemoteStore::Gcs { bucket, prefix, client } => (
                format!("gcs_{}_{}", bucket, prefix).replace(['/', ':'], "_"),
                client.store.clone(),
            ),
            RemoteStore::Adls {
                account,
                container,
                prefix,
                client,
            } => (
                format!("az_{}_{}_{}", account, container, prefix).replace(['/', ':'], "_"),
                client.store.clone(),
            ),
        }
    };
    let cache_dir = cache_root.join(safe_prefix);
    let offsets = cache_dir.join("offsets.bin");
    let edges = cache_dir.join("edges.bin");
    let meta_path = cache_dir.join("metadata.json");
    if offsets.exists() && edges.exists() && meta_path.exists() {
        cache_spinner.finish_with_message("Using cached shard");
        let metadata: DatasetMetadata = serde_json::from_slice(
            &fs::read(&meta_path).context("read cached metadata")?,
        )
        .context("parse cached metadata")?;
        return Ok((metadata, cache_dir));
    }
    cache_spinner.finish_with_message("Cache miss");
    fs::create_dir_all(&cache_dir)?;
    let prefix = match &remote {
        RemoteStore::S3 { prefix, .. }
        | RemoteStore::Gcs { prefix, .. }
        | RemoteStore::Adls { prefix, .. } => prefix.trim_end_matches('/').to_string(),
    };
    // Download metadata, then build using remote Parquet sources.
    let loc = if prefix.is_empty() {
        ObjectPath::from("metadata.json")
    } else {
        ObjectPath::from(format!("{}/{}", prefix, "metadata.json"))
    };
    let meta_spinner = start_spinner("Downloading metadata.json");
    let meta_obj = rt
        .block_on(store_arc.get(&loc))
        .with_context(|| format!("download metadata.json from {prefix}"))?;
    let meta_bytes = rt
        .block_on(meta_obj.bytes())
        .context("buffer metadata.json")?;
    fs::write(&meta_path, &meta_bytes)?;
    meta_spinner.finish_with_message("Downloaded metadata.json");
    let metadata: DatasetMetadata =
        serde_json::from_slice(&meta_bytes).context("parse metadata.json")?;

    let make_source = |key: String| -> Result<ParquetSource> {
        match &remote {
            RemoteStore::S3 { bucket, client, .. } => Ok(ParquetSource::S3(
                diskgraph_index::streaming::S3Location {
                    bucket: bucket.clone(),
                    key,
                    region: Some(client.resolved.region.clone()),
                    endpoint: client.resolved.endpoint.clone(),
                    access_key_id: Some(client.resolved.access_key_id.clone()),
                    secret_access_key: Some(client.resolved.secret_access_key.clone()),
                    session_token: client.resolved.session_token.clone(),
                    virtual_hosted_style_request: Some(client.resolved.virtual_hosted),
                },
            )),
            RemoteStore::Gcs { bucket, .. } => Ok(ParquetSource::Gcs(
                diskgraph_index::streaming::GcsLocation {
                    bucket: bucket.clone(),
                    key,
                    service_account_path: None,
                    service_account_key: None,
                    application_credentials_path: None,
                    endpoint: None,
                },
            )),
            RemoteStore::Adls {
                account,
                container,
                client,
                ..
            } => Ok(ParquetSource::Adls(
                diskgraph_index::streaming::AdlsLocation {
                    account: account.clone(),
                    container: container.clone(),
                    key,
                    access_key: client.resolved.access_key.clone(),
                    sas_key: client.resolved.sas_key.clone(),
                    bearer_token: client.resolved.bearer_token.clone(),
                    client_id: client.resolved.client_id.clone(),
                    client_secret: client.resolved.client_secret.clone(),
                    tenant_id: client.resolved.tenant_id.clone(),
                    endpoint: client.resolved.endpoint.clone(),
                    use_emulator: None,
                },
            )),
        }
    };

    // Build using streaming Parquet sources (first shard per type/relation).
    let node_inputs: Vec<NodeShardInput> = metadata
        .node_types
        .iter()
        .map(|nt| {
            let shard = nt
                .shards
                .get(0)
                .ok_or_else(|| anyhow!("no node shard found for {}", nt.name))?;
            let key = if prefix.is_empty() {
                shard.path.clone()
            } else {
                format!("{}/{}", prefix.trim_end_matches('/'), shard.path)
            };
            let parquet = make_source(key)?;
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
            let key = if prefix.is_empty() {
                shard.path.clone()
            } else {
                format!("{}/{}", prefix.trim_end_matches('/'), shard.path)
            };
            let parquet = make_source(key)?;
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
    let total_nodes = shard_node_count(&metadata);
    let (build_progress, is_bar) =
        start_build_progress(total_nodes, "Building shard from parquet");
    let mut on_progress = |seen: u64, total: u64| {
        if is_bar {
            let len = build_progress.length().unwrap_or(total);
            build_progress.set_position(seen.min(len));
        } else {
            build_progress.tick();
        }
    };
    builder
        .build_with_progress(&node_inputs[0], &edge_inputs, total_nodes, Some(&mut on_progress))
        .context("build shard from parquet")?;
    build_progress.finish_with_message("Built shard");
    fs::write(&meta_path, serde_json::to_vec_pretty(&metadata)?)?;
    Ok((metadata, shard_dir))
}

fn cache_tempdir() -> Result<tempfile::TempDir> {
    if let Some(root) = cache_root_dir() {
        tempfile::Builder::new()
            .prefix("diskgraph-explore-")
            .tempdir_in(root)
            .context("create tempdir in /mnt/localssd")
    } else {
        tempfile::Builder::new()
            .prefix("diskgraph-explore-")
            .tempdir()
            .context("create tempdir in system temp")
    }
}

fn cache_root_dir() -> Option<PathBuf> {
    let root = PathBuf::from("/mnt/localssd/diskgraph-index-cache");
    if root.exists() {
        return Some(root);
    }
    if let Some(parent) = root.parent() {
        if parent.exists() && fs::create_dir_all(&root).is_ok() {
            return Some(root);
        }
    }
    None
}

fn shard_node_count(metadata: &DatasetMetadata) -> Option<u64> {
    metadata
        .node_types
        .get(0)
        .and_then(|nt| nt.shards.get(0))
        .map(|s| s.max_id.saturating_sub(s.min_id).saturating_add(1))
}

fn start_build_progress(total: Option<u64>, message: &str) -> (ProgressBar, bool) {
    let total = total.unwrap_or(0);
    if total > 0 {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {pos}/{len} ({eta})")
                .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb.set_message(message.to_string());
        (pb, true)
    } else {
        (start_spinner(message), false)
    }
}

fn start_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(120));
    pb
}

fn is_indexed_shard(dir: &Path) -> bool {
    dir.join(OFFSETS_FILENAME).exists()
        && dir.join(EDGES_FILENAME).exists()
        && dir.join("metadata.json").exists()
}

fn sanitize_cache_name(input: &str) -> String {
    input
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn build_from_local_parquet(
    source: &Path,
    cache_root: &Path,
) -> Result<(DatasetMetadata, PathBuf)> {
    let meta_path = source.join("metadata.json");
    let meta_spinner = start_spinner("Loading metadata.json");
    let meta_bytes = fs::read(&meta_path).context("read metadata.json")?;
    meta_spinner.finish_with_message("Loaded metadata.json");
    let metadata: DatasetMetadata =
        serde_json::from_slice(&meta_bytes).context("parse metadata.json")?;

    let safe_prefix = format!("local_{}", sanitize_cache_name(&source.display().to_string()));
    let cache_dir = cache_root.join(safe_prefix);
    let offsets = cache_dir.join(OFFSETS_FILENAME);
    let edges = cache_dir.join(EDGES_FILENAME);
    let cached_meta = cache_dir.join("metadata.json");
    if offsets.exists() && edges.exists() && cached_meta.exists() {
        return Ok((metadata, cache_dir));
    }
    fs::create_dir_all(&cache_dir)?;

    let node_inputs: Vec<NodeShardInput> = metadata
        .node_types
        .iter()
        .map(|nt| {
            let shard = nt
                .shards
                .get(0)
                .ok_or_else(|| anyhow!("no node shard found for {}", nt.name))?;
            let parquet = ParquetSource::LocalFile(source.join(&shard.path));
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
            let parquet = ParquetSource::LocalFile(source.join(&shard.path));
            Ok(EdgeShardInput {
                relation: rel.name.clone(),
                source_shard_index: 0,
                parquet,
                has_dest_sort_key: rel.dest_sort_key_bits.is_some(),
            })
        })
        .collect::<Result<_>>()?;

    let params = ShardBuildParams {
        output_dir: cache_dir.clone(),
        node_meta: metadata.node_types[0].clone(),
        node_types: metadata.node_types.clone(),
        relation_metas: metadata.edge_relations.clone(),
        shard_index: 0,
    };
    let builder = ShardBuilder::new(params);
    let total_nodes = shard_node_count(&metadata);
    let (build_progress, is_bar) =
        start_build_progress(total_nodes, "Building shard from local parquet");
    let mut on_progress = |seen: u64, total: u64| {
        if is_bar {
            let len = build_progress.length().unwrap_or(total);
            build_progress.set_position(seen.min(len));
        } else {
            build_progress.tick();
        }
    };
    builder
        .build_with_progress(&node_inputs[0], &edge_inputs, total_nodes, Some(&mut on_progress))
        .context("build shard from parquet")?;
    build_progress.finish_with_message("Built shard");
    fs::write(&cached_meta, serde_json::to_vec_pretty(&metadata)?)?;
    Ok((metadata, cache_dir))
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
        Ok(Source::S3 { bucket, prefix: key })
    } else if raw.starts_with("gs://") {
        let (bucket, key) = parse_gs_url(raw)?;
        Ok(Source::Gcs { bucket, prefix: key })
    } else if raw.starts_with("az://") {
        let (account, container, key) = parse_az_url(raw)?;
        Ok(Source::Adls {
            account,
            container,
            prefix: key,
        })
    } else {
        Ok(Source::Local(PathBuf::from(raw)))
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

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}
