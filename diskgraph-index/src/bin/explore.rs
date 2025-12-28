//! Interactive shard explorer for diskgraph-index.
//!
//! Supports local shards or remote shards on S3/GCS/ADLS by streaming Parquet
//! shards through `ShardBuilder` into a temp cache (derived from the remote
//! path) and opening a `ShardReader`. Provides a simple REPL for inspecting
//! edge lists by node id.

use anyhow::{anyhow, Context, Result};
use diskgraph_core::metadata::{DatasetMetadata, EdgeRelationMetadata, NodeTypeMetadata};
use diskgraph_index::builder::{
    BuildProgress, BuildStage, ProgressReporter, compute_relation_layouts, IndexBuildError,
    ShardBuildParams, ShardBuilder,
};
use diskgraph_index::reader::ShardReader;
use diskgraph_index::streaming::{EdgeShardInput, ParquetSource};
use diskgraph_index::layout::{OFFSETS_FILENAME, EDGES_FILENAME};
use object_store::ObjectStore;
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig, S3Config,
};
use indicatif::{ProgressBar, ProgressStyle};
use object_store::path::Path as ObjectPath;
use pprof::protos::Message;
use pprof::ProfilerGuard;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::io::Write;
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
    let mut args = std::env::args().skip(1).peekable();
    let mut profile_out_base: Option<PathBuf> = None;
    let mut source_str: Option<String> = None;
    while let Some(arg) = args.next() {
        if arg == "--profile" {
            let base = match args.peek() {
                Some(next) if !next.starts_with("--") => args.next().unwrap(),
                _ => "explore-profile".to_string(),
            };
            profile_out_base = Some(PathBuf::from(base));
            continue;
        }
        if arg.starts_with("--") {
            return Err(anyhow!("unknown flag: {arg}"));
        }
        if source_str.is_none() {
            source_str = Some(arg);
        }
    }
    let source_str = source_str.ok_or_else(|| {
        anyhow!(
            "usage: explore [--profile [base]] <source> (local path or s3:// gs:// az://)"
        )
    })?;
    let profile_enabled = profile_out_base.is_some();
    if profile_enabled && cfg!(debug_assertions) && cfg!(target_arch = "aarch64") {
        return Err(anyhow!(
            "pprof profiling in debug builds can panic on aarch64; rerun with `--release`"
        ));
    }
    let _guard = profile_out_base
        .as_ref()
        .map(|_| ProfilerGuard::new(500))
        .transpose()
        .context("start profiler")?;
    let run_result = run_explore(&source_str, profile_enabled);
    if let Some(guard) = _guard.as_ref() {
        let report = guard.report().build().context("build profile report")?;
        if let Some(base) = profile_out_base.as_ref() {
            let svg_path = PathBuf::from(format!("{}.svg", base.display()));
            let pb_path = PathBuf::from(format!("{}.pb", base.display()));
            let file = fs::File::create(&svg_path).context("create profile output")?;
            report
                .flamegraph(file)
                .context("write flamegraph profile")?;
            println!("Wrote profile to {}", svg_path.display());
            let profile = report.pprof().context("build pprof profile")?;
            let mut file = fs::File::create(&pb_path).context("create pprof output")?;
            let bytes = profile.encode_to_vec();
            file.write_all(&bytes).context("write pprof profile")?;
            println!("Wrote pprof profile to {}", pb_path.display());
        }
    }
    match run_result {
        Ok(()) => Ok(()),
        Err(err) => {
            let aborted = err
                .downcast_ref::<IndexBuildError>()
                .map(|e| matches!(e, IndexBuildError::Aborted))
                .unwrap_or(false);
            if aborted {
                println!("Profile aborted.");
                Ok(())
            } else {
                Err(err)
            }
        }
    }
}

fn run_explore(source_str: &str, profile_enabled: bool) -> Result<()> {
    let source = parse_source(source_str)?;
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
    let reader = ShardReader::open(
        &shard_dir,
        layouts.clone(),
        metadata.node_types.first().map(|n| n.has_node_data).unwrap_or(false),
    )
    .context("open shard reader")?;

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
        let duration_display = format_duration_seconds(elapsed);
        println!(
            "Load stats: {}, {}, {}/s, {} nodes/s, {} edges/s, {} (nodes+edges)/s",
            duration_display,
            format_bytes(bytes_total),
            format_bytes(bps.round() as u64),
            format_number(nps.round() as u64),
            format_number(eps.round() as u64),
            format_number(((nodes + edges) as f64 / elapsed).round() as u64),
        );
    }
    let reader = Arc::new(reader);
    print_header(&metadata, &reader);
    if profile_enabled {
        let offsets_size = fs::metadata(shard_dir.join(OFFSETS_FILENAME))
            .map(|m| m.len())
            .unwrap_or(0);
        let edges_size = fs::metadata(shard_dir.join(EDGES_FILENAME))
            .map(|m| m.len())
            .unwrap_or(0);
        let bytes_total = offsets_size + edges_size;
        let nodes = reader.header().nodes;
        let edges = reader.edge_count();
        let bps = bytes_total as f64 / elapsed.max(0.0001);
        let nps = nodes as f64 / elapsed.max(0.0001);
        let eps = edges as f64 / elapsed.max(0.0001);
        println!("Profile mode summary:");
        println!("  Nodes: {}", format_number(nodes));
        println!("  Edges: {}", format_number(edges));
        println!("  Time: {:.2}s", elapsed);
        println!("  Data: {:.2} GB", bytes_total as f64 / (1024.0 * 1024.0 * 1024.0));
        println!("  Throughput: {:.2} MB/s", bps / (1024.0 * 1024.0));
        println!("  Nodes/s: {}", format_number(nps.round() as u64));
        println!("  Edges/s: {}", format_number(eps.round() as u64));
        return Ok(());
    }
    repl(&metadata.edge_relations, reader)?;
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
                },
            )),
        }
    };

    let node_data_source = if metadata
        .node_types
        .first()
        .map(|n| n.has_node_data)
        .unwrap_or(false)
    {
        let shard = metadata
            .node_types
            .first()
            .and_then(|n| n.shards.get(0))
            .ok_or_else(|| anyhow!("no node shard found for node data"))?;
        let path = shard
            .node_data_path
            .clone()
            .ok_or_else(|| anyhow!("node data path missing in metadata"))?;
        let key = if prefix.is_empty() {
            path
        } else {
            format!("{}/{}", prefix.trim_end_matches('/'), path)
        };
        Some(make_source(key)?)
    } else {
        None
    };

    // Build using streaming Parquet sources (first shard per relation).
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
        node_data: node_data_source,
    };
    let builder = ShardBuilder::new(params);
    let total_nodes = shard_node_count(&metadata);
    let (build_progress, progress_reporter) = make_progress_handler(total_nodes);
    builder
        .build_with_progress(
            &edge_inputs,
            total_nodes,
            Some(&progress_reporter),
            None,
        )
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum ProgressStageKey {
    Download {
        file: String,
        index: usize,
        total: usize,
    },
    BuildIndex,
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {prefix} ({eta})")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

fn make_progress_handler(
    total_nodes: Option<u64>,
) -> (ProgressBar, ProgressReporter<Box<dyn FnMut(BuildProgress)>>) {
    let pb = ProgressBar::new(0);
    pb.set_style(spinner_style());
    let pb_handle = pb.clone();
    let mut last_stage: Option<ProgressStageKey> = None;
    let mut last_total: Option<u64> = None;
    let handler = move |progress: BuildProgress| {
        match progress.stage {
            BuildStage::Download { file, index, total } => {
                let stage_key = ProgressStageKey::Download {
                    file: file.to_string(),
                    index,
                    total,
                };
                if last_stage.as_ref() != Some(&stage_key) {
                    pb_handle.set_style(bar_style());
                    pb_handle.set_position(0);
                    pb_handle.set_message(format!("Downloading {} ({}/{})", file, index, total));
                    last_stage = Some(stage_key);
                    last_total = None;
                }
                if let Some(total) = progress.total {
                    if last_total != Some(total) {
                        pb_handle.set_length(total);
                        last_total = Some(total);
                    }
                    pb_handle.set_position(progress.current.min(total));
                    pb_handle.set_prefix(format!(
                        "{} / {}",
                        format_bytes(progress.current),
                        format_bytes(total),
                    ));
                } else {
                    pb_handle.tick();
                }
            }
            BuildStage::BuildIndex => {
                let stage_key = ProgressStageKey::BuildIndex;
                if last_stage.as_ref() != Some(&stage_key) {
                    pb_handle.set_message("Building shard from parquet".to_string());
                    if let Some(total) = progress.total.or(total_nodes) {
                        pb_handle.set_style(bar_style());
                        pb_handle.set_length(total);
                        pb_handle.set_position(0);
                        last_total = Some(total);
                    } else {
                        pb_handle.set_style(spinner_style());
                        last_total = None;
                    }
                    last_stage = Some(stage_key);
                }
                if let Some(total) = progress.total.or(total_nodes) {
                    if last_total != Some(total) {
                        pb_handle.set_length(total);
                        last_total = Some(total);
                    }
                    pb_handle.set_position(progress.current.min(total));
                    pb_handle.set_prefix(format!(
                        "{} / {}",
                        format_count(progress.current),
                        format_count(total),
                    ));
                } else {
                    pb_handle.tick();
                }
            }
        }
    };
    (pb, ProgressReporter::new(Box::new(handler)))
}

fn format_bytes(value: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let v = value as f64;
    if v >= GB {
        format!("{:.1} GB", v / GB)
    } else if v >= MB {
        format!("{:.1} MB", v / MB)
    } else if v >= KB {
        format!("{:.1} KB", v / KB)
    } else {
        format!("{} B", value)
    }
}

fn format_count(value: u64) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;
    const B: f64 = 1_000_000_000.0;
    let v = value as f64;
    if v >= B {
        format!("{:.2}B", v / B)
    } else if v >= M {
        format!("{:.2}M", v / M)
    } else if v >= K {
        format!("{:.1}K", v / K)
    } else {
        value.to_string()
    }
}

fn format_number(value: u64) -> String {
    let s = value.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let mut count = 0usize;
    for ch in s.chars().rev() {
        if count != 0 && count % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
        count += 1;
    }
    out.chars().rev().collect()
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

    let node_data_source = if metadata
        .node_types
        .first()
        .map(|n| n.has_node_data)
        .unwrap_or(false)
    {
        let shard = metadata
            .node_types
            .first()
            .and_then(|n| n.shards.get(0))
            .ok_or_else(|| anyhow!("no node shard found for node data"))?;
        let path = shard
            .node_data_path
            .clone()
            .ok_or_else(|| anyhow!("node data path missing in metadata"))?;
        Some(ParquetSource::LocalFile(source.join(&path)))
    } else {
        None
    };

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
        node_data: node_data_source,
    };
    let builder = ShardBuilder::new(params);
    let total_nodes = shard_node_count(&metadata);
    let (build_progress, progress_reporter) = make_progress_handler(total_nodes);
    builder
        .build_with_progress(
            &edge_inputs,
            total_nodes,
            Some(&progress_reporter),
            None,
        )
        .context("build shard from parquet")?;
    build_progress.finish_with_message("Built shard");
    fs::write(&cached_meta, serde_json::to_vec_pretty(&metadata)?)?;
    Ok((metadata, cache_dir))
}

fn repl(relations: &[EdgeRelationMetadata], reader: Arc<ShardReader>) -> Result<()> {
    let mut rl = DefaultEditor::new().context("init editor")?;
    println!("Commands: info | node <id> | benchmark <threads> [duration] | quit");
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
                    print_info(relations, &reader);
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("node ") {
                    match rest.trim().parse::<u64>() {
                        Ok(id) => show_node(relations, &reader, id)?,
                        Err(_) => println!("Invalid node id: {rest}"),
                    }
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("benchmark ") {
                    let mut parts = rest.split_whitespace();
                    let Some(thread_part) = parts.next() else {
                        println!("Invalid benchmark command: {rest}");
                        continue;
                    };
                    let duration_part = parts.next();
                    let extra = parts.next();
                    if extra.is_some() {
                        println!("Invalid benchmark command: {rest}");
                        continue;
                    }
                    let threads = match thread_part.parse::<usize>() {
                        Ok(threads) if threads > 0 => threads,
                        _ => {
                            println!("Invalid thread count: {thread_part}");
                            continue;
                        }
                    };
                    let duration = if let Some(raw) = duration_part {
                        match parse_duration(raw) {
                            Some(d) => d,
                            None => {
                                println!("Invalid duration: {raw} (expected 30s or 5m)");
                                continue;
                            }
                        }
                    } else {
                        std::time::Duration::from_secs(20)
                    };
                    run_benchmark(reader.clone(), threads, duration)?;
                    continue;
                }
                println!(
                    "Unknown command. Use: info | node <id> | benchmark <threads> [duration] | quit"
                );
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
    let start = std::time::Instant::now();
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
    match view.node_data() {
        Ok(Some(bytes)) => {
            let escaped = escape_ascii(bytes);
            println!("  node_data: \"{}\"", escaped);
        }
        Ok(None) => {}
        Err(e) => {
            println!("  node_data: <error: {}>", e);
        }
    }
    let elapsed = start.elapsed();
    println!("  lookup: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    Ok(())
}

fn run_benchmark(reader: Arc<ShardReader>, threads: usize, duration: std::time::Duration) -> Result<()> {
    let nodes = reader.header().nodes;
    if nodes == 0 {
        println!("Benchmark skipped: no nodes in shard.");
        return Ok(());
    }
    println!(
        "Benchmark: threads={}, duration={}s, nodes={}",
        threads, duration.as_secs(), nodes
    );
    let start_time = std::time::Instant::now();
    let end_time = start_time + duration;

    const BUCKETS_US: [u64; 18] = [
        1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
        100_000, 200_000, 500_000,
    ];

    let mut handles = Vec::with_capacity(threads);
    for idx in 0..threads {
        let reader = reader.clone();
        handles.push(std::thread::spawn(move || {
            let mut state = (idx as u64 + 1) * 0x9E37_79B9_7F4A_7C15;
            let mut ops = 0u64;
            let mut errors = 0u64;
            let mut total_ns = 0u128;
            let mut min_ns = u128::MAX;
            let mut max_ns = 0u128;
            let mut total_edges = 0u64;
            let mut total_edge_bytes = 0u64;
            let mut total_node_data = 0u64;
            let mut checksum = 0u64;
            let mut hist = vec![0u64; BUCKETS_US.len() + 1];
            while std::time::Instant::now() < end_time {
                // LCG for speed; take high bits for distribution.
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let node = ((state >> 32) % nodes) as u64;
                let start = std::time::Instant::now();
                let view = reader.edges_for_node(node);
                let elapsed_ns = start.elapsed().as_nanos();
                if let Ok(view) = view {
                    if let Ok(segments) = view.relation_segments() {
                        let mut edges_here = 0u64;
                        let mut edge_bytes_here = 0u64;
                        for seg in &segments {
                            edges_here += seg.count as u64;
                            edge_bytes_here += seg.data.len() as u64;
                            for &b in seg.data {
                                checksum = checksum.wrapping_add(b as u64);
                            }
                        }
                        total_edges += edges_here;
                        total_edge_bytes += edge_bytes_here;
                    } else {
                        errors += 1;
                    }
                    if let Ok(node_data) = view.node_data() {
                        if let Some(bytes) = node_data {
                            total_node_data += bytes.len() as u64;
                        }
                    } else {
                        errors += 1;
                    }
                } else {
                    errors += 1;
                }
                ops += 1;
                total_ns += elapsed_ns;
                if elapsed_ns < min_ns {
                    min_ns = elapsed_ns;
                }
                if elapsed_ns > max_ns {
                    max_ns = elapsed_ns;
                }
                let us = (elapsed_ns / 1_000) as u64;
                let mut bucket = BUCKETS_US.len();
                for (i, bound) in BUCKETS_US.iter().enumerate() {
                    if us <= *bound {
                        bucket = i;
                        break;
                    }
                }
                hist[bucket] += 1;
            }
            (
                ops,
                errors,
                total_ns,
                min_ns,
                max_ns,
                total_edges,
                total_edge_bytes,
                total_node_data,
                checksum,
                hist,
            )
        }));
    }

    let length = duration.as_secs().max(1);
    let pb = ProgressBar::new(length);
    pb.set_style(
        ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {pos}/{len}s ({eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb.set_message("Benchmark");
    while std::time::Instant::now() < end_time {
        let elapsed = std::time::Instant::now().duration_since(start_time);
        pb.set_position(elapsed.as_secs().min(length));
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    pb.finish_with_message("Benchmark complete");

    let mut total_ops = 0u64;
    let mut total_errors = 0u64;
    let mut total_ns = 0u128;
    let mut min_ns = u128::MAX;
    let mut max_ns = 0u128;
    let mut total_edges = 0u64;
    let mut total_edge_bytes = 0u64;
    let mut total_node_data = 0u64;
    let mut checksum = 0u64;
    let mut hist = vec![0u64; BUCKETS_US.len() + 1];
    for handle in handles {
        let (ops, errors, ns, min, max, edges, edge_bytes, node_bytes, chk, h) =
            handle.join().unwrap();
        total_ops += ops;
        total_errors += errors;
        total_ns += ns;
        if min < min_ns {
            min_ns = min;
        }
        if max > max_ns {
            max_ns = max;
        }
        total_edges += edges;
        total_edge_bytes += edge_bytes;
        total_node_data += node_bytes;
        checksum = checksum.wrapping_add(chk);
        for (i, v) in h.into_iter().enumerate() {
            hist[i] += v;
        }
    }

    let elapsed = duration.as_secs_f64();
    let ops_per_sec = (total_ops as f64 / elapsed).round() as u64;
    let avg_ns = if total_ops > 0 {
        (total_ns / total_ops as u128) as u64
    } else {
        0
    };
    println!("Results:");
    println!("  ops: {}", format_number(total_ops));
    println!("  ops/s: {}", format_number(ops_per_sec));
    println!("  errors: {}", format_number(total_errors));
    println!("  edges read: {}", format_number(total_edges));
    println!(
        "  edges/s: {}",
        format_number((total_edges as f64 / elapsed).round() as u64)
    );
    println!("  edge bytes: {}", format_bytes(total_edge_bytes));
    let edge_bps = (total_edge_bytes as f64 / elapsed).round() as u64;
    println!("  edge bytes/s: {}/s", format_bytes(edge_bps));
    println!("  node_data bytes: {}", format_bytes(total_node_data));
    let node_data_bps = (total_node_data as f64 / elapsed).round() as u64;
    println!(
        "  node_data bytes/s: {}/s",
        format_bytes(node_data_bps)
    );
    println!("  checksum: {}", checksum);
    println!("  avg: {}", format_duration_ns(avg_ns as u128));
    if min_ns != u128::MAX {
        println!("  min: {}", format_duration_ns(min_ns));
    }
    println!("  max: {}", format_duration_ns(max_ns));
    println!("  p50: {}", format_duration_ns(percentile_ns(&hist, &BUCKETS_US, 0.50)));
    println!("  p90: {}", format_duration_ns(percentile_ns(&hist, &BUCKETS_US, 0.90)));
    println!("  p99: {}", format_duration_ns(percentile_ns(&hist, &BUCKETS_US, 0.99)));
    println!("Latency histogram:");
    for (i, count) in hist.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        if i < BUCKETS_US.len() {
            let bound_ns = (BUCKETS_US[i] as u128) * 1_000;
            println!(
                "  <= {:>9}: {}",
                format_duration_ns(bound_ns),
                format_number(*count)
            );
        } else {
            let bound_ns = (BUCKETS_US[BUCKETS_US.len() - 1] as u128) * 1_000;
            println!(
                "  >  {:>9}: {}",
                format_duration_ns(bound_ns),
                format_number(*count)
            );
        }
    }
    Ok(())
}

fn parse_duration(raw: &str) -> Option<std::time::Duration> {
    if raw.len() < 2 {
        return None;
    }
    let (value, suffix) = raw.split_at(raw.len() - 1);
    let amount: u64 = value.parse().ok()?;
    match suffix {
        "s" => Some(std::time::Duration::from_secs(amount)),
        "m" => Some(std::time::Duration::from_secs(amount.saturating_mul(60))),
        _ => None,
    }
}

fn percentile_ns(hist: &[u64], buckets_us: &[u64], pct: f64) -> u128 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0;
    }
    let target = (total as f64 * pct).ceil() as u64;
    let mut cumulative = 0u64;
    for (i, count) in hist.iter().enumerate() {
        cumulative += *count;
        if cumulative >= target {
            if i < buckets_us.len() {
                return (buckets_us[i] as u128) * 1_000;
            }
            return (buckets_us[buckets_us.len() - 1] as u128) * 1_000;
        }
    }
    (buckets_us[buckets_us.len() - 1] as u128) * 1_000
}

fn format_duration_ns(ns: u128) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2} s", ns as f64 / 1_000_000_000.0)
    } else if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.2} us", ns as f64 / 1_000.0)
    } else {
        format!("{} ns", ns)
    }
}

fn format_duration_seconds(seconds: f64) -> String {
    if seconds < 60.0 {
        return format!("{:.2}s", seconds);
    }
    let total_secs = seconds.round() as u64;
    let minutes = total_secs / 60;
    let secs = total_secs % 60;
    format!("{}m {}s", minutes, secs)
}

fn escape_ascii(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7E => out.push(b as char),
            _ => out.push_str(&format!("\\x{:02x}", b)),
        }
    }
    out
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
