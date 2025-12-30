//! Interactive shard explorer for diskgraph-index.
//!
//! Supports local graphs or remote graphs on S3/GCS/ADLS by streaming Parquet
//! shards through `GraphIndexBuilder` into a temp cache (derived from the remote
//! path) and opening a `GraphIndexReader`. Provides a simple REPL for inspecting
//! edge lists by node id.

use anyhow::{anyhow, Context, Result};
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_index::builder::IndexBuildError;
use diskgraph_index::graph::{
    GraphBuildProgress, GraphBuildStage, GraphDataSource, GraphIndexBuilder, GraphIndexReader,
    GraphProgressSink, TableReader,
};
use diskgraph_index::layout::{EDGES_FILENAME, OFFSETS_FILENAME};
use diskgraph_rowpack::{ColumnEncoding, DecimalStorage, FloatStorage, RowPackContext, StringEncoding};
use object_store::{Error as ObjectStoreError, ObjectStore};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig, S3Config,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use object_store::path::Path as ObjectPath;
use pprof::protos::Message;
use pprof::ProfilerGuard;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write, IsTerminal};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
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

#[derive(Debug, Deserialize)]
struct IngestReport {
    stats: IngestStatsReport,
    source: SourceMetadataReport,
}

#[derive(Debug, Deserialize)]
struct IngestStatsReport {
    key_parse_failures: u64,
    key_lookup_failures: u64,
    key_failure_samples: Vec<String>,
    key_failures_by_context: HashMap<String, KeyFailureCountsReport>,
    pk_duplicate_counts: HashMap<String, u64>,
    pk_duplicate_total: u64,
    pk_duplicate_samples: HashMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct KeyFailureCountsReport {
    parse_failures: u64,
    lookup_failures: u64,
}

#[derive(Debug, Deserialize)]
struct SourceMetadataReport {
    dataset_name: String,
    tables: Vec<TableSpecReport>,
    #[serde(default)]
    relationships: Vec<RelationshipSpecReport>,
}

#[derive(Debug, Deserialize)]
struct TableSpecReport {
    name: String,
    columns: Vec<ColumnSpecReport>,
    #[serde(default)]
    primary_key: Option<String>,
    #[serde(default)]
    edge_table: Option<EdgeTableSpecReport>,
}

#[derive(Debug, Deserialize)]
struct ColumnSpecReport {
    name: String,
    dtype: String,
    #[serde(default)]
    fixed_len: Option<usize>,
    #[serde(default)]
    key_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EdgeTableSpecReport {
    src_fk: ForeignKeySpecReport,
    dst_fk: ForeignKeySpecReport,
}

#[derive(Debug, Deserialize)]
struct ForeignKeySpecReport {
    column: String,
    ref_table: String,
    ref_column: String,
}

#[derive(Debug, Deserialize)]
struct RelationshipSpecReport {
    #[serde(default)]
    name: Option<String>,
    src_table: String,
    src_column: String,
    dst_table: String,
    dst_column: String,
}

#[derive(Debug, Deserialize)]
struct BuildReport {
    build_stats: BuildStatsReport,
}

#[derive(Debug, Deserialize)]
struct BuildStatsReport {
    nodes_ingested: u64,
    edges_ingested: u64,
    truncated_nodes: u64,
    truncated_edges_total: u64,
    truncated_edges_per_relation: Vec<u64>,
    capped_edge_lists: u64,
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
    let (metadata, index_root) = match &source {
        Source::Local(p) => {
            if p.is_file() && p.file_name().map(|n| n == "index_metadata.json").unwrap_or(false) {
                let root = p.parent().unwrap_or(p).to_path_buf();
                if is_indexed_graph(&root) {
                    let metadata: DatasetMetadata =
                        serde_json::from_slice(&fs::read(p).context("read index_metadata.json")?)
                            .context("parse index_metadata.json")?;
                    (metadata, root)
                } else {
                    build_from_local_dataset(&root, tmp.path())?
                }
            } else if is_indexed_graph(p) {
                let metadata: DatasetMetadata = serde_json::from_slice(
                    &fs::read(p.join("index_metadata.json")).context("read index_metadata.json")?,
                )
                .context("parse index_metadata.json")?;
                (metadata, p.clone())
            } else {
                build_from_local_dataset(p, tmp.path())?
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
    let reader = GraphIndexReader::open(&index_root.join("index_metadata.json"), &index_root)
        .context("open graph reader")?;

    println!("Loaded graph from {}", source_str);
    if elapsed > 0.0 {
        let mut offsets_size = 0u64;
        let mut edges_size = 0u64;
        let mut nodes = 0u64;
        let mut edges = 0u64;
        for table in reader.tables() {
            for idx in 0..table.shard_count() {
                if let Some(shard) = table.shard(idx) {
                    offsets_size += shard.offsets_len();
                    edges_size += shard.edges_len();
                    nodes += shard.header().nodes;
                    edges += shard.edge_count();
                }
            }
        }
        let bytes_total = offsets_size + edges_size;
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
    if metadata.node_types.is_empty() {
        println!("No tables found in metadata.");
        return Ok(());
    }
    let default_table = metadata
        .node_types
        .first()
        .map(|n| n.name.clone())
        .unwrap_or_else(|| "unknown".to_string());
    print_header(&metadata, &reader, &default_table);
    if profile_enabled {
        let mut offsets_size = 0u64;
        let mut edges_size = 0u64;
        let mut nodes = 0u64;
        let mut edges = 0u64;
        for table in reader.tables() {
            for idx in 0..table.shard_count() {
                if let Some(shard) = table.shard(idx) {
                    offsets_size += shard.offsets_len();
                    edges_size += shard.edges_len();
                    nodes += shard.header().nodes;
                    edges += shard.edge_count();
                }
            }
        }
        let bytes_total = offsets_size + edges_size;
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
    repl(reader, default_table, index_root)?;
    Ok(())
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
    let meta_path = cache_dir.join("index_metadata.json");
    let prefix = match &remote {
        RemoteStore::S3 { prefix, .. }
        | RemoteStore::Gcs { prefix, .. }
        | RemoteStore::Adls { prefix, .. } => prefix.trim_end_matches('/').to_string(),
    };
    if cache_dir.join("nodes").exists() && meta_path.exists() {
        cache_spinner.finish_with_message("Cache hit");
        let metadata: DatasetMetadata = serde_json::from_slice(
            &fs::read(&meta_path).context("read cached metadata")?,
        )
        .context("parse cached metadata")?;
        download_rowpack_contexts(rt, store_arc.as_ref(), &prefix, &cache_dir, &metadata)
            .context("download rowpack contexts")?;
        download_optional_object(
            rt,
            store_arc.as_ref(),
            &prefix,
            "ingest_stats.json",
            &cache_dir,
        )
        .context("download ingest_stats.json")?;
        return Ok((metadata, cache_dir));
    }
    cache_spinner.finish_with_message("Cache miss");
    fs::create_dir_all(&cache_dir)?;
    // Download metadata, then build using remote Parquet sources.
    let loc = if prefix.is_empty() {
        ObjectPath::from("index_metadata.json")
    } else {
        ObjectPath::from(format!("{}/{}", prefix, "index_metadata.json"))
    };
    let meta_spinner = start_spinner("Downloading index_metadata.json");
    let meta_obj = rt
        .block_on(store_arc.get(&loc))
        .with_context(|| format!("download index_metadata.json from {prefix}"))?;
    let meta_bytes = rt
        .block_on(meta_obj.bytes())
        .context("buffer index_metadata.json")?;
    fs::write(&meta_path, &meta_bytes)?;
    meta_spinner.finish_with_message("Downloaded index_metadata.json");
    let metadata: DatasetMetadata =
        serde_json::from_slice(&meta_bytes).context("parse index_metadata.json")?;
    download_rowpack_contexts(rt, store_arc.as_ref(), &prefix, &cache_dir, &metadata)
        .context("download rowpack contexts")?;
    download_optional_object(
        rt,
        store_arc.as_ref(),
        &prefix,
        "ingest_stats.json",
        &cache_dir,
    )
    .context("download ingest_stats.json")?;

    let data_source = match &remote {
        RemoteStore::S3 { bucket, client, prefix } => GraphDataSource::S3 {
            bucket: bucket.clone(),
            prefix: prefix.clone(),
            region: Some(client.resolved.region.clone()),
            endpoint: client.resolved.endpoint.clone(),
            access_key_id: Some(client.resolved.access_key_id.clone()),
            secret_access_key: Some(client.resolved.secret_access_key.clone()),
            session_token: client.resolved.session_token.clone(),
            virtual_hosted_style_request: Some(client.resolved.virtual_hosted),
        },
        RemoteStore::Gcs { bucket, prefix, .. } => GraphDataSource::Gcs {
            bucket: bucket.clone(),
            prefix: prefix.clone(),
        },
        RemoteStore::Adls {
            account,
            container,
            prefix,
            client,
        } => GraphDataSource::Adls {
            account: account.clone(),
            container: container.clone(),
            prefix: prefix.clone(),
            access_key: client.resolved.access_key.clone(),
            sas_key: client.resolved.sas_key.clone(),
            bearer_token: client.resolved.bearer_token.clone(),
            client_id: client.resolved.client_id.clone(),
            client_secret: client.resolved.client_secret.clone(),
            tenant_id: client.resolved.tenant_id.clone(),
            endpoint: client.resolved.endpoint.clone(),
        },
    };

    let builder = GraphIndexBuilder::from_metadata(metadata.clone(), data_source, &cache_dir);
    let total_nodes = builder
        .metadata()
        .node_types
        .iter()
        .flat_map(|t| t.shards.iter())
        .map(|s| s.max_id.saturating_sub(s.min_id).saturating_add(1))
        .sum::<u64>();
    let (_mp, progress_reporter) = make_graph_progress_handler(total_nodes);
    builder
        .build(Some(&progress_reporter))
        .context("build graph from parquet")?;
    Ok((metadata, cache_dir))
}

fn copy_rowpack_contexts_from_local(
    metadata: &DatasetMetadata,
    source_root: &Path,
    cache_root: &Path,
) -> Result<()> {
    for node in &metadata.node_types {
        let compression = match &node.node_data_compression {
            Some(compression) if compression.codec == "rowpack" => compression,
            _ => continue,
        };
        if compression.context_path.is_empty() {
            return Err(anyhow!("rowpack context_path is empty for {}", node.name));
        }
        let ctx_path = Path::new(&compression.context_path);
        let src = source_root.join(ctx_path);
        let dst = cache_root.join(ctx_path);
        if dst.exists() {
            continue;
        }
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dst)
            .with_context(|| format!("copy rowpack context {}", src.display()))?;
    }
    Ok(())
}

fn download_rowpack_contexts(
    rt: &Runtime,
    store: &dyn ObjectStore,
    prefix: &str,
    cache_root: &Path,
    metadata: &DatasetMetadata,
) -> Result<()> {
    for node in &metadata.node_types {
        let compression = match &node.node_data_compression {
            Some(compression) if compression.codec == "rowpack" => compression,
            _ => continue,
        };
        if compression.context_path.is_empty() {
            return Err(anyhow!("rowpack context_path is empty for {}", node.name));
        }
        let key = if prefix.is_empty() {
            compression.context_path.clone()
        } else {
            format!("{}/{}", prefix, compression.context_path)
        };
        let dst = cache_root.join(&compression.context_path);
        if dst.exists() {
            continue;
        }
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        let obj = rt
            .block_on(store.get(&ObjectPath::from(key.clone())))
            .with_context(|| format!("download rowpack context {}", key))?;
        let bytes = rt.block_on(obj.bytes()).context("buffer rowpack context")?;
        fs::write(&dst, &bytes)?;
    }
    Ok(())
}

fn copy_optional_local_file(source_root: &Path, filename: &str, cache_root: &Path) -> Result<bool> {
    let src = source_root.join(filename);
    if !src.exists() {
        return Ok(false);
    }
    let dst = cache_root.join(filename);
    if dst.exists() {
        return Ok(true);
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&src, &dst)
        .with_context(|| format!("copy {}", src.display()))?;
    Ok(true)
}

fn download_optional_object(
    rt: &Runtime,
    store: &dyn ObjectStore,
    prefix: &str,
    filename: &str,
    cache_root: &Path,
) -> Result<bool> {
    let key = if prefix.is_empty() {
        filename.to_string()
    } else {
        format!("{}/{}", prefix, filename)
    };
    let dst = cache_root.join(filename);
    if dst.exists() {
        return Ok(true);
    }
    let obj = match rt.block_on(store.get(&ObjectPath::from(key.clone()))) {
        Ok(obj) => obj,
        Err(ObjectStoreError::NotFound { .. }) => return Ok(false),
        Err(err) => return Err(anyhow!("download {}: {}", key, err)),
    };
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = rt.block_on(obj.bytes()).context("buffer object")?;
    fs::write(&dst, &bytes)?;
    Ok(true)
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

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {prefix} ({eta})")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

struct GraphProgressReporter {
    multi: Arc<MultiProgress>,
    bars: Mutex<HashMap<String, ProgressBar>>,
    global: ProgressBar,
    total_nodes: u64,
    shard_progress: Mutex<HashMap<String, u64>>,
}

impl GraphProgressReporter {
    fn new(multi: Arc<MultiProgress>, total_nodes: u64) -> Self {
        let global = multi.add(ProgressBar::new(total_nodes));
        global.set_style(bar_style());
        global.set_message("Overall nodes");
        global.set_prefix(format!("0 / {}", format_count(total_nodes)));
        Self {
            multi,
            bars: Mutex::new(HashMap::new()),
            global,
            total_nodes,
            shard_progress: Mutex::new(HashMap::new()),
        }
    }

    fn bar_for(&self, key: &str) -> ProgressBar {
        let mut bars = self.bars.lock().unwrap();
        if let Some(pb) = bars.get(key) {
            return pb.clone();
        }
        let pb = self.multi.insert_before(&self.global, ProgressBar::new(0));
        pb.set_style(spinner_style());
        pb.set_message(format!("Building {key}"));
        bars.insert(key.to_string(), pb.clone());
        pb
    }

    fn finish_bar(&self, key: &str) {
        let mut bars = self.bars.lock().unwrap();
        if let Some(pb) = bars.remove(key) {
            pb.finish_with_message(format!("Built {key}"));
        }
    }
}

impl GraphProgressSink for GraphProgressReporter {
    fn report(&self, update: GraphBuildProgress) {
        match update.stage {
            GraphBuildStage::Download {
                shard,
                file,
                index,
                total,
            } => {
                let pb = self.bar_for(shard.as_ref());
                pb.set_style(bar_style());
                pb.set_message(format!("Downloading {file} ({index}/{total})"));
                if let Some(total) = update.total {
                    pb.set_length(total);
                    pb.set_position(update.current.min(total));
                    pb.set_prefix(format!(
                        "{} / {}",
                        format_bytes(update.current),
                        format_bytes(total),
                    ));
                } else {
                    pb.tick();
                }
            }
            GraphBuildStage::BuildShard { shard } => {
                let pb = self.bar_for(shard.as_ref());
                pb.set_message(format!("Building {shard}"));
                if let Some(total) = update.total {
                    pb.set_style(bar_style());
                    pb.set_length(total);
                    pb.set_position(update.current.min(total));
                    pb.set_prefix(format!(
                        "{} / {}",
                        format_count(update.current),
                        format_count(total),
                    ));
                } else {
                    pb.set_style(spinner_style());
                    pb.tick();
                }
                let mut shard_progress = self.shard_progress.lock().unwrap();
                shard_progress.insert(shard.as_ref().to_string(), update.current);
                let total_done: u64 = shard_progress.values().sum();
                self.global.set_position(total_done.min(self.total_nodes));
                self.global.set_prefix(format!(
                    "{} / {}",
                    format_count(total_done),
                    format_count(self.total_nodes),
                ));
            }
            GraphBuildStage::Complete { shard } => {
                if let Some(total) = update.total {
                    let mut shard_progress = self.shard_progress.lock().unwrap();
                    shard_progress.insert(shard.as_ref().to_string(), total);
                }
                self.finish_bar(shard.as_ref());
            }
        }
    }
}

fn make_graph_progress_handler(total_nodes: u64) -> (Arc<MultiProgress>, GraphProgressReporter) {
    let multi = Arc::new(MultiProgress::new());
    let reporter = GraphProgressReporter::new(Arc::clone(&multi), total_nodes);
    (multi, reporter)
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

fn is_indexed_graph(dir: &Path) -> bool {
    let meta_path = dir.join("index_metadata.json");
    if !meta_path.exists() || !dir.join("nodes").exists() {
        return false;
    }
    let bytes = match fs::read(&meta_path) {
        Ok(bytes) => bytes,
        Err(_) => return false,
    };
    let metadata: DatasetMetadata = match serde_json::from_slice(&bytes) {
        Ok(meta) => meta,
        Err(_) => return false,
    };
    for node_type in &metadata.node_types {
        for (idx, _shard) in node_type.shards.iter().enumerate() {
            let shard_dir = dir
                .join("nodes")
                .join(&node_type.name)
                .join(format!("shard-{:05}", idx));
            if !shard_dir.join(OFFSETS_FILENAME).exists()
                || !shard_dir.join(EDGES_FILENAME).exists()
            {
                return false;
            }
        }
    }
    true
}

fn sanitize_cache_name(input: &str) -> String {
    input
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn build_from_local_dataset(
    source: &Path,
    cache_root: &Path,
) -> Result<(DatasetMetadata, PathBuf)> {
    let meta_path = source.join("index_metadata.json");
    let meta_spinner = start_spinner("Loading index_metadata.json");
    let meta_bytes = fs::read(&meta_path).context("read index_metadata.json")?;
    meta_spinner.finish_with_message("Loaded index_metadata.json");
    let metadata: DatasetMetadata =
        serde_json::from_slice(&meta_bytes).context("parse index_metadata.json")?;

    let safe_prefix = format!("local_{}", sanitize_cache_name(&source.display().to_string()));
    let cache_dir = cache_root.join(safe_prefix);
    let cached_meta = cache_dir.join("index_metadata.json");
    if cache_dir.join("nodes").exists() && cached_meta.exists() {
        copy_rowpack_contexts_from_local(&metadata, source, &cache_dir)
            .context("copy rowpack contexts")?;
        copy_optional_local_file(source, "ingest_stats.json", &cache_dir)
            .context("copy ingest_stats.json")?;
        return Ok((metadata, cache_dir));
    }
    fs::create_dir_all(&cache_dir)?;

    let builder = GraphIndexBuilder::from_metadata_path(&meta_path, &cache_dir)?;
    let total_nodes = builder
        .metadata()
        .node_types
        .iter()
        .flat_map(|t| t.shards.iter())
        .map(|s| s.max_id.saturating_sub(s.min_id).saturating_add(1))
        .sum::<u64>();
    let (_mp, progress_reporter) = make_graph_progress_handler(total_nodes);
    builder
        .build(Some(&progress_reporter))
        .context("build graph from parquet")?;
    copy_rowpack_contexts_from_local(&metadata, source, &cache_dir)
        .context("copy rowpack contexts")?;
    copy_optional_local_file(source, "ingest_stats.json", &cache_dir)
        .context("copy ingest_stats.json")?;
    fs::write(&cached_meta, serde_json::to_vec_pretty(&metadata)?)?;
    Ok((metadata, cache_dir))
}

fn repl(reader: Arc<GraphIndexReader>, default_table: String, index_root: PathBuf) -> Result<()> {
    let mut rl = DefaultEditor::new().context("init editor")?;
    let mut current_table = default_table;
    println!("Commands: tables | use <table> | info | node <id> | formats | benchmark <threads> [duration] | benchmarkall <threads> [duration] | quit");
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
                if trimmed.eq_ignore_ascii_case("tables") {
                    print_tables(&reader);
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("use ") {
                    let name = rest.trim();
                    if reader.table(name).is_some() {
                        current_table = name.to_string();
                        println!("Using table {}", current_table);
                    } else {
                        println!("Unknown table: {name}");
                    }
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("info") {
                    print_info(&reader, &current_table);
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("formats") {
                    print_formats(reader.metadata(), &index_root)?;
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("node ") {
                    match rest.trim().parse::<u64>() {
                        Ok(id) => show_node(&reader, &current_table, id)?,
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
                    let table = match reader.table(&current_table) {
                        Some(table) => table,
                        None => {
                            println!("Unknown table: {}", current_table);
                            continue;
                        }
                    };
                    run_benchmark(table, threads, duration)?;
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("benchmarkall ") {
                    let mut parts = rest.split_whitespace();
                    let Some(thread_part) = parts.next() else {
                        println!("Invalid benchmarkall command: {rest}");
                        continue;
                    };
                    let duration_part = parts.next();
                    let extra = parts.next();
                    if extra.is_some() {
                        println!("Invalid benchmarkall command: {rest}");
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
                    run_benchmark_all(reader.clone(), threads, duration)?;
                    continue;
                }
                println!(
                    "Unknown command. Use: tables | use <table> | info | node <id> | formats | benchmark <threads> [duration] | benchmarkall <threads> [duration] | quit"
                );
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

fn print_header(metadata: &DatasetMetadata, reader: &GraphIndexReader, current_table: &str) {
    println!(
        "Node types: {}; Relations: {}; Current table: {}",
        metadata.node_types.len(),
        metadata.edge_relations.len(),
        current_table
    );
    print_tables(reader);
}

fn print_tables(reader: &GraphIndexReader) {
    println!("Tables:");
    for table in reader.tables() {
        println!(
            "  {} (shards={}, nodes={})",
            table.node_type(),
            table.shard_count(),
            format_number(table.node_count())
        );
        for rel in table.relations() {
            let sort_label = rel
                .dest_sort_key_bits
                .map(|bits| format!("{bits}b"))
                .unwrap_or_else(|| "none".to_string());
            println!(
                "    -> {} (dst={}, sort_key={})",
                rel.name, rel.dst_type, sort_label
            );
        }
    }
}

fn print_formats(metadata: &DatasetMetadata, root: &Path) -> Result<()> {
    println!("Formats:");
    for node in &metadata.node_types {
        println!();
        println!("Table: {}", node.name);
        if let Some(pk) = &node.primary_key_map {
            let key_mode = pk.key_mode.as_deref().unwrap_or("var");
            let fixed = pk
                .fixed_len
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "  Primary key: {} (dtype={}, mode={}, fixed_len={}, mapping={})",
                pk.column, pk.dtype, key_mode, fixed, pk.mapping_kind
            );
        } else {
            println!("  Primary key: none");
        }
        if let Some(compression) = &node.node_data_compression {
            println!(
                "  Node data: {} (ctx={})",
                compression.codec, compression.context_path
            );
            if compression.codec == "rowpack" {
                let ctx_path = root.join(&compression.context_path);
                if let Ok(bytes) = fs::read(&ctx_path) {
                    if let Ok(ctx) = RowPackContext::from_bytes(&bytes) {
                        print_rowpack_summary(&ctx);
                    } else {
                        println!("    Rowpack: failed to parse {}", ctx_path.display());
                    }
                } else {
                    println!("    Rowpack: missing {}", ctx_path.display());
                }
            }
        } else {
            println!("  Node data: none");
        }
    }

    if let Some(report) = read_ingest_report(root)? {
        print_ingest_report(&report);
    } else {
        println!();
        println!("Ingest stats: ingest_stats.json not found");
    }

    if let Some(report) = read_build_report(root)? {
        print_build_report(&report);
    } else {
        println!();
        println!("Build stats: build_stats.json not found");
    }
    Ok(())
}

fn read_ingest_report(root: &Path) -> Result<Option<IngestReport>> {
    let path = root.join("ingest_stats.json");
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let report: IngestReport = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(Some(report))
}

fn read_build_report(root: &Path) -> Result<Option<BuildReport>> {
    let path = root.join("build_stats.json");
    if !path.exists() {
        return read_build_reports_recursive(root);
    }
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let report: BuildReport =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    Ok(Some(report))
}

fn read_build_reports_recursive(root: &Path) -> Result<Option<BuildReport>> {
    let nodes_root = root.join("nodes");
    if !nodes_root.exists() {
        return Ok(None);
    }
    let mut stack = vec![nodes_root];
    let mut agg = BuildStatsReport {
        nodes_ingested: 0,
        edges_ingested: 0,
        truncated_nodes: 0,
        truncated_edges_total: 0,
        truncated_edges_per_relation: Vec::new(),
        capped_edge_lists: 0,
    };
    let mut found = false;
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.file_name().and_then(|n| n.to_str()) != Some("build_stats.json") {
                continue;
            }
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let report: BuildReport =
                serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
            found = true;
            agg.nodes_ingested = agg.nodes_ingested.saturating_add(report.build_stats.nodes_ingested);
            agg.edges_ingested = agg.edges_ingested.saturating_add(report.build_stats.edges_ingested);
            agg.truncated_nodes =
                agg.truncated_nodes.saturating_add(report.build_stats.truncated_nodes);
            agg.truncated_edges_total =
                agg.truncated_edges_total.saturating_add(report.build_stats.truncated_edges_total);
            agg.capped_edge_lists =
                agg.capped_edge_lists.saturating_add(report.build_stats.capped_edge_lists);
        }
    }
    if !found {
        return Ok(None);
    }
    Ok(Some(BuildReport { build_stats: agg }))
}

fn print_ingest_report(report: &IngestReport) {
    println!();
    println!("Ingest stats:");
    println!(
        "  Key failures: parse={}, lookup={}",
        report.stats.key_parse_failures, report.stats.key_lookup_failures
    );
    if !report.stats.key_failures_by_context.is_empty() {
        let mut entries: Vec<_> = report.stats.key_failures_by_context.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        println!("  Key failures by context:");
        for (ctx, counts) in entries {
            if counts.parse_failures == 0 && counts.lookup_failures == 0 {
                continue;
            }
            println!(
                "    {}: parse {}, lookup {}",
                ctx, counts.parse_failures, counts.lookup_failures
            );
        }
    }
    if !report.stats.key_failure_samples.is_empty() {
        println!("  Key failure samples:");
        for sample in &report.stats.key_failure_samples {
            println!("    {}", sample);
        }
    }
    if report.stats.pk_duplicate_total > 0 {
        println!("  Duplicate primary keys: {}", report.stats.pk_duplicate_total);
        let mut entries: Vec<_> = report.stats.pk_duplicate_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (table, count) in entries {
            println!("    {}: {}", table, count);
            if let Some(samples) = report.stats.pk_duplicate_samples.get(table) {
                if !samples.is_empty() {
                    println!("      samples: {}", samples.join(", "));
                }
            }
        }
    }
    print_source_schema(&report.source);
}

fn print_source_schema(source: &SourceMetadataReport) {
    println!();
    println!("Source schema: {}", source.dataset_name);
    let mut tables: Vec<&TableSpecReport> = source.tables.iter().collect();
    tables.sort_by(|a, b| a.name.cmp(&b.name));
    let mut table_map: HashMap<&str, &TableSpecReport> = HashMap::new();
    for table in &source.tables {
        table_map.insert(table.name.as_str(), table);
    }
    for table in tables {
        println!("  Table {}:", table.name);
        if let Some(pk) = &table.primary_key {
            if let Some(col) = table.columns.iter().find(|c| &c.name == pk) {
                println!("    Primary key: {} ({})", pk, format_column(col));
            } else {
                println!("    Primary key: {}", pk);
            }
        } else {
            println!("    Primary key: none");
        }
        if let Some(edge) = &table.edge_table {
            println!(
                "    Edge table: {} -> {}",
                format_fk(&table, &edge.src_fk),
                format_fk(&table, &edge.dst_fk)
            );
        }
    }
    if !source.relationships.is_empty() {
        println!("  Relationships:");
        for rel in &source.relationships {
            let name = rel
                .name
                .as_deref()
                .unwrap_or("unnamed");
            let src = table_map
                .get(rel.src_table.as_str())
                .and_then(|t| t.columns.iter().find(|c| c.name == rel.src_column));
            let dst = table_map
                .get(rel.dst_table.as_str())
                .and_then(|t| t.columns.iter().find(|c| c.name == rel.dst_column));
            let src_desc = src.map(format_column).unwrap_or_else(|| "unknown".to_string());
            let dst_desc = dst.map(format_column).unwrap_or_else(|| "unknown".to_string());
            println!(
                "    {}: {}.{} ({}) -> {}.{} ({})",
                name,
                rel.src_table,
                rel.src_column,
                src_desc,
                rel.dst_table,
                rel.dst_column,
                dst_desc
            );
        }
    }
}

fn format_fk(table: &TableSpecReport, fk: &ForeignKeySpecReport) -> String {
    let col = table.columns.iter().find(|c| c.name == fk.column);
    match col {
        Some(col) => format!(
            "{} ({} -> {}.{})",
            fk.column,
            format_column(col),
            fk.ref_table,
            fk.ref_column
        ),
        None => format!("{} -> {}.{}", fk.column, fk.ref_table, fk.ref_column),
    }
}

fn format_column(col: &ColumnSpecReport) -> String {
    let mut out = col.dtype.clone();
    if let Some(mode) = &col.key_mode {
        out.push_str(&format!(", mode={}", mode));
    } else {
        out.push_str(", mode=var");
    }
    if let Some(len) = col.fixed_len {
        out.push_str(&format!(", fixed_len={}", len));
    }
    out
}

fn print_build_report(report: &BuildReport) {
    let stats = &report.build_stats;
    println!();
    println!("Build stats:");
    println!("  Nodes ingested: {}", stats.nodes_ingested);
    println!("  Edges ingested: {}", stats.edges_ingested);
    if stats.truncated_nodes > 0 || stats.truncated_edges_total > 0 {
        println!("  Truncated nodes: {}", stats.truncated_nodes);
        println!("  Truncated edges: {}", stats.truncated_edges_total);
    }
    if stats.capped_edge_lists > 0 {
        println!("  Capped edge lists: {}", stats.capped_edge_lists);
    }
    if !stats.truncated_edges_per_relation.is_empty() {
        let total: u64 = stats.truncated_edges_per_relation.iter().sum();
        if total > 0 {
            println!("  Truncated edges by relation: {}", format_count(total));
        }
    }
}

fn print_rowpack_summary(ctx: &RowPackContext) {
    println!("    Rowpack: v{}, estimated_row_bytes={}", ctx.version, ctx.estimated_row_bytes);
    println!("    Columns:");
    for (idx, col) in ctx.columns.iter().enumerate() {
        let encoding = format_rowpack_encoding(&col.encoding);
        println!("      [{}] {} ({:?}) {}", idx, col.name, col.logical_type, encoding);
    }
}

fn format_rowpack_encoding(encoding: &ColumnEncoding) -> String {
    match encoding {
        ColumnEncoding::Int { signed, varint, bits } => {
            format!("int(signed={}, varint={}, bits={})", signed, varint, bits)
        }
        ColumnEncoding::Float { storage } => format!("float({})", format_float_storage(*storage)),
        ColumnEncoding::Bool => "bool".to_string(),
        ColumnEncoding::String { encoding } => format_string_encoding(encoding),
        ColumnEncoding::Bytes { fixed_len } => match fixed_len {
            Some(len) => format!("bytes(fixed_len={})", len),
            None => "bytes".to_string(),
        },
        ColumnEncoding::Date { varint, bits } => {
            format!("date(varint={}, bits={})", varint, bits)
        }
        ColumnEncoding::Timestamp => "timestamp".to_string(),
        ColumnEncoding::Decimal {
            precision,
            scale,
            bytes,
            storage,
        } => format!(
            "decimal(precision={}, scale={}, bytes={}, storage={})",
            precision,
            scale,
            bytes,
            format_decimal_storage(storage)
        ),
        ColumnEncoding::FloatArray { storage } => {
            format!("float_array({})", format_float_storage(*storage))
        }
    }
}

fn format_string_encoding(encoding: &StringEncoding) -> String {
    match encoding {
        StringEncoding::Raw => "string(raw)".to_string(),
        StringEncoding::Dict { values } => format!("string(dict entries={})", values.len()),
        StringEncoding::Alphabet { alphabet } => {
            format!("string(alphabet len={})", alphabet.len())
        }
        StringEncoding::Bpe { dict } => format!("string(bpe entries={})", dict.len()),
        StringEncoding::NumericInt => "string(numeric_int)".to_string(),
        StringEncoding::NumericScaled => "string(numeric_scaled)".to_string(),
        StringEncoding::Hex { len, dashed, upper } => {
            format!("string(hex len={} dashed={} upper={})", len, dashed, upper)
        }
        StringEncoding::IsoTimestamp { fmt } => format!("string(iso_timestamp {:?})", fmt),
        StringEncoding::Snappy => "string(snappy)".to_string(),
    }
}

fn format_float_storage(storage: FloatStorage) -> &'static str {
    match storage {
        FloatStorage::F16 => "f16",
        FloatStorage::F32 => "f32",
        FloatStorage::F64 => "f64",
    }
}

fn format_decimal_storage(storage: &DecimalStorage) -> String {
    match storage {
        DecimalStorage::Raw => "raw".to_string(),
        DecimalStorage::Int { varint, bits } => format!("int(varint={}, bits={})", varint, bits),
        DecimalStorage::Float { storage } => format!("float({})", format_float_storage(*storage)),
    }
}

fn print_info(reader: &GraphIndexReader, table_name: &str) {
    let table = match reader.table(table_name) {
        Some(table) => table,
        None => {
            println!("Unknown table: {table_name}");
            return;
        }
    };
    println!(
        "Table: {} (nodes={}, relations={})",
        table.node_type(),
        format_number(table.node_count()),
        table.relations().len()
    );
    for (idx, rel) in table.relations().iter().enumerate() {
        let sort = rel
            .dest_sort_key_bits
            .map(|b| format!("sort_bits={b}"))
            .unwrap_or_else(|| "no sort key".into());
        println!("  [{}] {} ({} -> {}), {}", idx, rel.name, rel.src_type, rel.dst_type, sort);
    }
}

fn show_node(reader: &GraphIndexReader, table_name: &str, node: u64) -> Result<()> {
    let start = std::time::Instant::now();
    let table = match reader.table(table_name) {
        Some(table) => table,
        None => {
            println!("Unknown table: {table_name}");
            return Ok(());
        }
    };
    let view = match table.edges_for_node(node) {
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
        let rel = &table.relations()[idx];
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
        Ok(_) => match table.decode_node_data_json(&view) {
            Ok(Some(value)) => match serde_json::to_string_pretty(&value) {
                Ok(pretty) => println!("  node_data:\n{pretty}"),
                Err(_) => {
                    println!("  node_data: <error: json encode>");
                }
            },
            Ok(None) => match table.decode_node_data(&view) {
                Ok(Some(bytes)) => {
                    let escaped = escape_ascii(&bytes);
                    println!("  node_data: \"{}\"", escaped);
                }
                Ok(None) => {}
                Err(e) => {
                    println!("  node_data: <error: {}>", e);
                }
            },
            Err(e) => {
                println!("  node_data: <error: {}>", e);
            }
        },
        Err(e) => {
            println!("  node_data: <error: {}>", e);
        }
    }
    let elapsed = start.elapsed();
    println!("  lookup: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    Ok(())
}

fn run_benchmark(
    table: Arc<TableReader>,
    threads: usize,
    duration: std::time::Duration,
) -> Result<()> {
    run_benchmark_tables(vec![table], false, threads, duration)
}

fn run_benchmark_all(
    reader: Arc<GraphIndexReader>,
    threads: usize,
    duration: std::time::Duration,
) -> Result<()> {
    let tables: Vec<Arc<TableReader>> = reader
        .tables()
        .into_iter()
        .filter(|t| t.node_count() > 0)
        .collect();
    if tables.is_empty() {
        println!("Benchmark skipped: no nodes across tables.");
        return Ok(());
    }
    run_benchmark_tables(tables, true, threads, duration)
}

fn run_benchmark_tables(
    tables: Vec<Arc<TableReader>>,
    round_robin: bool,
    threads: usize,
    duration: std::time::Duration,
) -> Result<()> {
    let table_count = tables.len();
    if table_count == 1 && tables[0].node_count() == 0 {
        println!("Benchmark skipped: no nodes in table.");
        return Ok(());
    }
    let total_nodes: u64 = tables.iter().map(|t| t.node_count()).sum();
    let label = if round_robin { "Benchmark all" } else { "Benchmark" };
    println!(
        "{}: threads={}, duration={}s, tables={}, nodes={}",
        label,
        threads,
        duration.as_secs(),
        table_count,
        format_number(total_nodes)
    );
    let start_time = std::time::Instant::now();
    let end_time = start_time + duration;
    let mut ops_counters = Vec::with_capacity(threads);
    for _ in 0..threads {
        ops_counters.push(Arc::new(AtomicU64::new(0)));
    }

    const BUCKETS_US: [u64; 18] = [
        1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
        100_000, 200_000, 500_000,
    ];

    let mut handles = Vec::with_capacity(threads);
    for idx in 0..threads {
        let tables = tables.clone();
        let table_count = table_count;
        let ops_counter = ops_counters[idx].clone();
        let round_robin = round_robin;
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
                let table_idx = if round_robin {
                    (ops as usize) % table_count
                } else {
                    0
                };
                let table = &tables[table_idx];
                let node_count = table.node_count();
                if node_count == 0 {
                    errors += 1;
                    ops += 1;
                    ops_counter.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let offset = ((state >> 32) % node_count) as u64;
                let Some(node) = table.node_id_from_offset(offset) else {
                    errors += 1;
                    ops += 1;
                    ops_counter.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                let start = std::time::Instant::now();
                let view = table.edges_for_node(node);
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
                ops_counter.fetch_add(1, Ordering::Relaxed);
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

    if io::stdout().is_terminal() {
        let mut samples = Vec::new();
        let mut scale_max = 1u64;
        let mut last_bucket_time = start_time;
        let mut last_ops = 0u64;
        while std::time::Instant::now() < end_time {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(start_time);
            let total_ops: u64 = ops_counters
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .sum();
            if now.duration_since(last_bucket_time) >= std::time::Duration::from_secs(1) {
                let bucket_secs = now.duration_since(last_bucket_time).as_secs_f64().max(0.001);
                let delta = total_ops.saturating_sub(last_ops);
                let throughput = (delta as f64 / bucket_secs).round() as u64;
                samples.push(throughput);
                last_ops = total_ops;
                last_bucket_time = now;
            }
            render_benchmark_display(elapsed, duration, total_ops, &samples, &mut scale_max)?;
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        let total_ops: u64 = ops_counters
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();
        let bucket_secs = std::time::Instant::now()
            .duration_since(last_bucket_time)
            .as_secs_f64()
            .max(0.001);
        let delta = total_ops.saturating_sub(last_ops);
        let throughput = (delta as f64 / bucket_secs).round() as u64;
        samples.push(throughput);
        render_benchmark_display(duration, duration, total_ops, &samples, &mut scale_max)?;
        clear_benchmark_display()?;
    } else {
        while std::time::Instant::now() < end_time {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }

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
    let (offsets_bytes, edges_bytes) = total_table_bytes(&tables);
    let data_bytes = offsets_bytes.saturating_add(edges_bytes);
    println!("  offsets bytes: {}", format_bytes(offsets_bytes));
    println!("  edge bytes: {}", format_bytes(edges_bytes));
    if let Some(available) = available_memory_bytes() {
        println!("  available ram: {}", format_bytes(available));
        let load_factor = if available > 0 {
            data_bytes as f64 / available as f64
        } else {
            0.0
        };
        println!("  load factor: {:.2}", load_factor);
    } else {
        println!("  available ram: n/a");
        println!("  load factor: n/a");
    }
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

fn render_benchmark_display(
    elapsed: std::time::Duration,
    duration: std::time::Duration,
    total_ops: u64,
    samples: &[u64],
    scale_max: &mut u64,
) -> Result<()> {
    let width = terminal_width().unwrap_or(80);
    let chart_height = 10usize;
    let mut out = String::new();
    out.push_str("\x1b[2J\x1b[H");
    out.push_str(&format!(
        "Benchmark running: {} / {}\n",
        format_duration_short(elapsed),
        format_duration_short(duration)
    ));
    out.push_str(&format!("Ops so far: {}\n", format_number(total_ops)));
    let current_ops = samples.last().copied().unwrap_or(0);
    out.push_str(&format!(
        "Current ops/s: {}\n",
        format_number(current_ops)
    ));
    out.push_str(&render_progress_bar(width, elapsed, duration));
    out.push('\n');
    out.push_str(&render_ops_chart(samples, width, chart_height, scale_max));
    let mut stdout = io::stdout();
    stdout.write_all(out.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

fn clear_benchmark_display() -> Result<()> {
    let mut stdout = io::stdout();
    stdout.write_all(b"\x1b[2J\x1b[H")?;
    stdout.flush()?;
    Ok(())
}

fn render_progress_bar(width: usize, elapsed: std::time::Duration, duration: std::time::Duration) -> String {
    let bar_width = width.saturating_sub(20).max(10);
    let frac = (elapsed.as_secs_f64() / duration.as_secs_f64()).min(1.0).max(0.0);
    let filled = (bar_width as f64 * frac).round() as usize;
    let empty = bar_width.saturating_sub(filled);
    let percent = (frac * 100.0).round() as u64;
    format!(
        "[{}{}] {:>3}%",
        format!("\x1b[46m{}\x1b[0m", " ".repeat(filled)),
        "-".repeat(empty),
        percent
    )
}

fn render_ops_chart(samples: &[u64], width: usize, height: usize, scale_max: &mut u64) -> String {
    if height == 0 || width == 0 {
        return String::new();
    }
    let label_width = 10usize;
    let chart_width = width.saturating_sub(label_width + 3).max(10);
    let start = samples.len().saturating_sub(chart_width);
    let window = &samples[start..];
    let max_val = window.iter().copied().max().unwrap_or(0);
    let avg_val = if window.is_empty() {
        0.0
    } else {
        window.iter().map(|v| *v as f64).sum::<f64>() / window.len() as f64
    };
    let target_max = (avg_val * 2.0).ceil() as u64;
    let desired_scale = if max_val == 0 { 1 } else { max_val.max(1) };
    if desired_scale > *scale_max {
        *scale_max = desired_scale;
    } else if target_max > 0 {
        let min_increase = ((*scale_max as f64) * 1.5).ceil() as u64;
        if target_max >= min_increase {
            *scale_max = target_max;
        }
    }
    let scale_max = *scale_max;
    let mut out = String::new();
    out.push('\n');
    let row_mid = (height + 1) / 2;
    for row in (1..=height).rev() {
        let label = if row == height {
            format_count(scale_max)
        } else if row == row_mid {
            format_count(scale_max / 2)
        } else if row == 1 {
            "0".to_string()
        } else {
            String::new()
        };
        out.push_str(&format!("{:>width$} |", label, width = label_width));
        for &val in window {
            let bar_height = ((val as u128 * height as u128) + scale_max as u128 - 1)
                / scale_max as u128;
            if bar_height >= row as u128 {
                let color = color_for_row(row, height);
                out.push_str(color);
                out.push('⣿');
                out.push_str("\x1b[0m");
            } else {
                out.push(' ');
            }
        }
        out.push('\n');
    }
    out.push_str(&format!(
        "{:>width$} +{}\n",
        "",
        "-".repeat(chart_width),
        width = label_width
    ));
    let mut axis = vec![' '; chart_width];
    let total_secs = samples.len() as u64;
    let window_start = total_secs.saturating_sub(window.len() as u64);
    for i in 0..window.len() {
        let t = window_start + i as u64 + 1;
        if t % 10 == 0 {
            let label = format_duration_short(std::time::Duration::from_secs(t));
            let pos = i.min(chart_width.saturating_sub(1));
            let end = (pos + label.len()).min(axis.len());
            for (idx, ch) in label.chars().enumerate() {
                if pos + idx >= end {
                    break;
                }
                axis[pos + idx] = ch;
            }
        }
    }
    out.push_str(&format!(
        "{:>width$}  {}\n",
        "",
        axis.into_iter().collect::<String>(),
        width = label_width
    ));
    out
}

fn terminal_width() -> Option<usize> {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
            return Some(ws.ws_col as usize);
        }
    }
    std::env::var("COLUMNS").ok().and_then(|v| v.parse().ok())
}

fn format_duration_short(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        return format!("{}s", secs);
    }
    let minutes = secs / 60;
    let rem = secs % 60;
    format!("{}m {}s", minutes, rem)
}

fn color_for_row(row: usize, height: usize) -> &'static str {
    if height == 0 {
        return "\x1b[37m";
    }
    let ratio = (row as f64 / height as f64).min(1.0);
    match ratio {
        r if r >= 0.85 => "\x1b[31m", // red
        r if r >= 0.70 => "\x1b[33m", // yellow
        r if r >= 0.55 => "\x1b[32m", // green
        r if r >= 0.40 => "\x1b[36m", // cyan
        r if r >= 0.25 => "\x1b[34m", // blue
        r if r >= 0.10 => "\x1b[35m", // magenta
        _ => "\x1b[37m",              // white
    }
}

fn available_memory_bytes() -> Option<u64> {
    let contents = fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let parts: Vec<&str> = rest.trim().split_whitespace().collect();
            if let Some(kb) = parts.first() {
                if let Ok(value) = kb.parse::<u64>() {
                    return Some(value.saturating_mul(1024));
                }
            }
        }
    }
    None
}

fn total_table_bytes(tables: &[Arc<TableReader>]) -> (u64, u64) {
    let mut offsets = 0u64;
    let mut edges = 0u64;
    for table in tables {
        for idx in 0..table.shard_count() {
            if let Some(shard) = table.shard(idx) {
                offsets = offsets.saturating_add(shard.offsets_len());
                edges = edges.saturating_add(shard.edges_len());
            }
        }
    }
    (offsets, edges)
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
