//! Generate multi-table synthetic dataset for diskgraph ingest.
//!
//! This generator emits multiple node tables and bidirectional edge relations
//! across them. It favors throughput: no joins or sorting passes beyond
//! streaming ordered writes.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use arrow_array::builder::BinaryBuilder;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, EdgeStorageFormat, NodeShardMetadata,
    NodeTypeMetadata,
};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig, S3Config,
};
use object_store::WriteMultipart;
use object_store::{path::Path as ObjectPath, ObjectStore};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokio::runtime::Runtime;
use tracing::info;
use std::env;
use futures_util::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate multi-table Parquet shards for benchmarking diskgraph ingest.")]
struct MultiArgs {
    /// Output target; local path or s3 URL (e.g., s3://bucket/prefix).
    #[arg(long, value_name = "PATH_OR_URL")]
    output: String,
    /// Destination kind.
    #[arg(long, value_enum, default_value = "local")]
    dest: Dest,
    /// Total nodes across all tables (supports 10k, 5m, 2b).
    #[arg(long, default_value = "500000000", value_parser = parse_human_u64)]
    nodes: u64,
    /// Edges per relation (applied to each relation).
    #[arg(long, default_value = "50000000", value_parser = parse_human_u64)]
    edges_per_relation: u64,
    /// Rows per RecordBatch written to Parquet for edges (supports 10k, 5m).
    #[arg(long, default_value = "1000000", value_parser = parse_human_usize)]
    edge_batch_size: usize,
    /// Parquet row group size (rows); keep aligned with batch sizes for streaming writes.
    #[arg(long, default_value = "1000000", value_parser = parse_human_usize)]
    row_group_size: usize,
    /// RNG seed (per relation a different salt is applied).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of shards per node table.
    #[arg(long, default_value_t = 5)]
    num_shards: usize,
    /// Worker threads used for parallel generation.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
    /// Enable node data Parquet generation.
    #[arg(long, default_value_t = true)]
    node_data: bool,
    /// Disable node data Parquet generation.
    #[arg(long, default_value_t = false)]
    no_node_data: bool,
    /// Bytes per node in the node data column (supports 12mb, 256k).
    #[arg(long, default_value = "128b", value_parser = parse_human_usize_bytes)]
    node_data_bytes: usize,
    /// Rows per RecordBatch written to Parquet for node data.
    #[arg(long, default_value = "1000000", value_parser = parse_human_usize)]
    node_data_batch_size: usize,
    /// S3 region override (dest=s3).
    #[arg(long)]
    s3_region: Option<String>,
    /// S3 endpoint override (dest=s3, for custom/minio).
    #[arg(long)]
    s3_endpoint: Option<String>,
    /// S3 multipart part size in bytes (supports 64mb, 128mb).
    #[arg(long, default_value = "64mb", value_parser = parse_human_usize_bytes)]
    s3_part_size: usize,
    /// Max concurrent multipart parts in flight (for all cloud backends).
    #[arg(long, default_value_t = 10)]
    multipart_concurrency: usize,
    /// Allow http for S3 endpoints (e.g., local test endpoints).
    #[arg(long)]
    s3_allow_http: Option<bool>,
    /// Force virtual-hosted style for S3 (defaults to true for AWS endpoints, false for custom unless set).
    #[arg(long)]
    s3_virtual_hosted: Option<bool>,
    /// Azure endpoint override (dest=adls, for custom endpoints).
    #[arg(long)]
    azure_endpoint: Option<String>,
    /// Allow http for Azure endpoints (e.g., local test endpoints).
    #[arg(long)]
    azure_allow_http: Option<bool>,
    /// Azure SAS key.
    #[arg(long)]
    azure_sas_key: Option<String>,
    /// Azure access key.
    #[arg(long)]
    azure_access_key: Option<String>,
    /// Azure bearer token.
    #[arg(long)]
    azure_bearer_token: Option<String>,
    /// Azure client id.
    #[arg(long)]
    azure_client_id: Option<String>,
    /// Azure client secret.
    #[arg(long)]
    azure_client_secret: Option<String>,
    /// Azure tenant id.
    #[arg(long)]
    azure_tenant_id: Option<String>,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Dest {
    Local,
    S3,
    Gcs,
    Adls,
}

#[derive(Clone)]
struct NodeTypeDef {
    name: &'static str,
    weight: u32,
}

#[derive(Clone)]
struct RelationDef {
    name: &'static str,
    src: &'static str,
    dst: &'static str,
    dest_sort_key_bits: Option<u8>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let mut args = MultiArgs::parse();
    if args.no_node_data {
        args.node_data = false;
    }
    let start = std::time::Instant::now();
    let target = OutputTarget::parse(&args)?;
    target.prepare()?;

    let node_defs = vec![
        NodeTypeDef { name: "User", weight: 35 },
        NodeTypeDef { name: "Product", weight: 25 },
        NodeTypeDef { name: "Order", weight: 15 },
        NodeTypeDef { name: "Session", weight: 10 },
        NodeTypeDef { name: "Merchant", weight: 10 },
        NodeTypeDef { name: "Category", weight: 5 },
    ];

    let relation_defs = vec![
        RelationDef { name: "Views", src: "User", dst: "Product", dest_sort_key_bits: Some(32) },
        RelationDef { name: "ViewedBy", src: "Product", dst: "User", dest_sort_key_bits: Some(32) },
        RelationDef { name: "Purchases", src: "User", dst: "Product", dest_sort_key_bits: Some(16) },
        RelationDef { name: "PurchasedBy", src: "Product", dst: "User", dest_sort_key_bits: Some(16) },
        RelationDef { name: "FollowsMerchant", src: "User", dst: "Merchant", dest_sort_key_bits: None },
        RelationDef { name: "Follower", src: "Merchant", dst: "User", dest_sort_key_bits: None },
        RelationDef { name: "InCategory", src: "Product", dst: "Category", dest_sort_key_bits: None },
        RelationDef { name: "HasProduct", src: "Category", dst: "Product", dest_sort_key_bits: None },
        RelationDef { name: "HasSession", src: "User", dst: "Session", dest_sort_key_bits: Some(32) },
        RelationDef { name: "SessionFor", src: "Session", dst: "User", dest_sort_key_bits: Some(32) },
        RelationDef { name: "OrderContains", src: "Order", dst: "Product", dest_sort_key_bits: None },
        RelationDef { name: "OrderedIn", src: "Product", dst: "Order", dest_sort_key_bits: None },
        RelationDef { name: "PlacedOrder", src: "User", dst: "Order", dest_sort_key_bits: None },
        RelationDef { name: "PlacedBy", src: "Order", dst: "User", dest_sort_key_bits: None },
    ];

    let node_counts = distribute_counts(args.nodes, &node_defs);
    let node_map: std::collections::HashMap<&'static str, u64> = node_defs
        .iter()
        .zip(node_counts.iter())
        .map(|(d, c)| (d.name, *c))
        .collect();

    info!(
        "Generating multi dataset: nodes={}, edges_per_relation={}, node_data={}, node_data_bytes={}",
        format_count_with_raw(args.nodes),
        format_count_with_raw(args.edges_per_relation),
        args.node_data,
        format_bytes_with_raw(args.node_data_bytes as u64),
    );

    let mut node_types = Vec::new();
    let mut shard_ranges_by_type: std::collections::HashMap<String, Vec<NodeShardMetadata>> =
        std::collections::HashMap::new();
    for (def, count) in node_defs.iter().zip(node_counts.iter()) {
        let mut shards = shard_plan(*count, args.num_shards);
        if args.node_data {
            for (idx, shard) in shards.iter_mut().enumerate() {
                let rel_path = format!(
                    "nodes/{}/part-{:05}.parquet",
                    def.name.to_lowercase(),
                    idx
                );
                shard.node_data_paths = Some(vec![rel_path]);
            }
        }
        shard_ranges_by_type.insert(def.name.to_string(), shards.clone());
        let max_id = count.saturating_sub(1);
        node_types.push(NodeTypeMetadata {
            name: def.name.to_string(),
            max_id,
            has_node_data: args.node_data,
            shards,
            node_data_compression: None,
            primary_key_map: None,
        });
    }

    let mut tasks = Vec::new();
    if args.node_data {
        plan_node_data(&target, &node_types, &mut tasks)?;
    }

    let mut edge_relations = Vec::new();
    for rel in &relation_defs {
        let src_total = *node_map.get(rel.src).unwrap_or(&0);
        let dst_total = *node_map.get(rel.dst).unwrap_or(&0);
        let src_shards = shard_ranges_by_type
            .get(rel.src)
            .cloned()
            .unwrap_or_default();
        let edge_meta = plan_edges_for_relation(
            &target,
            rel,
            &args,
            src_total,
            dst_total,
            &src_shards,
            &mut tasks,
        )?;
        edge_relations.push(EdgeRelationMetadata {
            name: rel.name.to_string(),
            src_type: rel.src.to_string(),
            dst_type: rel.dst.to_string(),
            shards: edge_meta,
            dest_sort_key_bits: rel.dest_sort_key_bits,
            format: EdgeStorageFormat::Parquet,
        });
    }

    let workers = args.threads.max(1);
    print_plan(&node_types, &edge_relations, tasks.len(), workers, args.node_data);
    info!("Generating shards with {workers} worker threads.");
    run_tasks(tasks, workers, target.clone(), &args)?;

    let metadata = DatasetMetadata {
        dataset_name: target.dataset_name(),
        node_types,
        edge_relations,
    };
    let json = serde_json::to_string_pretty(&metadata)?;
    target.write_bytes("index_metadata.json", json.into_bytes())?;

    print_stats(&args, &metadata, &target, start.elapsed());
    info!("Done. Output written to {}", target.display());
    Ok(())
}

fn distribute_counts(total: u64, defs: &[NodeTypeDef]) -> Vec<u64> {
    let total_weight: u64 = defs.iter().map(|d| d.weight as u64).sum();
    let mut counts = Vec::with_capacity(defs.len());
    let mut remaining = total;
    for (idx, def) in defs.iter().enumerate() {
        if idx == defs.len() - 1 {
            counts.push(remaining);
        } else {
            let share = total.saturating_mul(def.weight as u64) / total_weight.max(1);
            let take = share.min(remaining);
            counts.push(take);
            remaining = remaining.saturating_sub(take);
        }
    }
    if counts.iter().all(|&c| c == 0) && !counts.is_empty() {
        counts[0] = 1;
    }
    counts
}

fn shard_plan(total_nodes: u64, shards: usize) -> Vec<NodeShardMetadata> {
    let shard_count = shards.max(1) as u64;
    if total_nodes == 0 {
        return vec![
            NodeShardMetadata {
                min_id: 0,
                max_id: 0,
                node_data_paths: None,
            };
            shard_count as usize
        ];
    }
    let per_shard = ((total_nodes + shard_count - 1) / shard_count).max(1);
    let mut remaining = total_nodes;
    let mut start = 0u64;
    let mut shards_out = Vec::with_capacity(shard_count as usize);
    for _ in 0..shard_count {
        if remaining == 0 {
            shards_out.push(NodeShardMetadata {
                min_id: 0,
                max_id: 0,
                node_data_paths: None,
            });
            continue;
        }
        let len = per_shard.min(remaining);
        let end = start + len - 1;
        shards_out.push(NodeShardMetadata {
            min_id: start,
            max_id: end,
            node_data_paths: None,
        });
        remaining -= len;
        start = end + 1;
    }
    shards_out
}

fn plan_node_data(
    target: &OutputTarget,
    node_types: &[NodeTypeMetadata],
    tasks: &mut Vec<Task>,
) -> Result<()> {
    for node_type in node_types {
        for (idx, shard) in node_type.shards.iter().enumerate() {
            let rows = shard.max_id.saturating_sub(shard.min_id).saturating_add(1);
            let rel_path = shard
                .node_data_paths
                .as_ref()
                .and_then(|paths| paths.first())
                .cloned()
                .unwrap_or_else(|| {
                    format!("nodes/{}/part-{:05}.parquet", node_type.name.to_lowercase(), idx)
                });
            let path = target.join(&rel_path);
            if let OutputTarget::Local { .. } = target {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
            }
            tasks.push(Task {
                label: format!("Nodes {} shard {}", node_type.name, idx),
                total: rows,
                kind: TaskKind::NodeData {
                    path,
                    start_id: shard.min_id,
                    rows,
                },
            });
        }
    }
    Ok(())
}

fn plan_edges_for_relation(
    target: &OutputTarget,
    rel: &RelationDef,
    args: &MultiArgs,
    src_total: u64,
    dst_total: u64,
    src_shards: &[NodeShardMetadata],
    tasks: &mut Vec<Task>,
) -> Result<Vec<EdgeShardMetadata>> {
    let mut out = Vec::with_capacity(src_shards.len());
    let rel_dir = format!("edges/{}", rel.name.to_lowercase());
    for (idx, shard) in src_shards.iter().enumerate() {
        let rel_path = format!("{}/part-{:05}.parquet", rel_dir, idx);
        let path = target.join(&rel_path);
        if let OutputTarget::Local { .. } = target {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
        }
        let shard_len = shard
            .max_id
            .saturating_sub(shard.min_id)
            .saturating_add(1);
        tasks.push(Task {
            label: format!("Edges {} shard {}", rel.name, idx),
            total: shard_len,
            kind: TaskKind::EdgeShard {
                path,
                dest_sort_key_bits: rel.dest_sort_key_bits,
                src_min: shard.min_id,
                src_max: shard.max_id,
                src_total,
                dst_total,
                seed: args.seed ^ ((idx as u64) << 16) ^ (rel.name.len() as u64),
            },
        });
        out.push(EdgeShardMetadata {
            path: rel_path,
            src_min_id: shard.min_id,
            src_max_id: shard.max_id,
        });
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct TaskConfig {
    edge_batch_size: usize,
    row_group_size: usize,
    node_data_batch_size: usize,
    node_data_bytes: usize,
    edges_per_relation: u64,
}

impl TaskConfig {
    fn new(args: &MultiArgs) -> Self {
        Self {
            edge_batch_size: args.edge_batch_size,
            row_group_size: args.row_group_size,
            node_data_batch_size: args.node_data_batch_size,
            node_data_bytes: args.node_data_bytes,
            edges_per_relation: args.edges_per_relation,
        }
    }
}

struct Task {
    label: String,
    total: u64,
    kind: TaskKind,
}

enum TaskKind {
    NodeData {
        path: PathBuf,
        start_id: u64,
        rows: u64,
    },
    EdgeShard {
        path: PathBuf,
        dest_sort_key_bits: Option<u8>,
        src_min: u64,
        src_max: u64,
        src_total: u64,
        dst_total: u64,
        seed: u64,
    },
}

fn run_tasks(tasks: Vec<Task>, threads: usize, target: OutputTarget, args: &MultiArgs) -> Result<()> {
    if tasks.is_empty() {
        return Ok(());
    }
    let config = TaskConfig::new(args);
    let target = Arc::new(target);
    let multi = Arc::new(MultiProgress::new());
    let total_tasks = tasks.len() as u64;
    let global = multi.add(ProgressBar::new(total_tasks));
    global.set_style(bar_style());
    global.set_message("Overall");
    global.set_prefix(format!("0 / {}", format_count(total_tasks)));
    let (tx, rx) = mpsc::channel::<Task>();
    for task in tasks {
        tx.send(task).map_err(|e| anyhow::anyhow!("task queue error: {e}"))?;
    }
    drop(tx);

    let rx = Arc::new(Mutex::new(rx));
    let err: Arc<Mutex<Option<anyhow::Error>>> = Arc::new(Mutex::new(None));
    let mut handles = Vec::with_capacity(threads);

    for _ in 0..threads {
        let rx = Arc::clone(&rx);
        let err = Arc::clone(&err);
        let multi = Arc::clone(&multi);
        let target = Arc::clone(&target);
        let global_bar = global.clone();
        let config = config;
        handles.push(thread::spawn(move || loop {
            if err.lock().unwrap().is_some() {
                break;
            }
            let task = {
                let guard = rx.lock().unwrap();
                guard.recv()
            };
            let task = match task {
                Ok(task) => task,
                Err(_) => break,
            };
            let pb = multi.insert_before(&global_bar, ProgressBar::new(task.total));
            pb.set_style(bar_style());
            pb.set_message(task.label.clone());
            pb.set_prefix(format!("0 / {}", format_count(task.total)));
            let result = match task.kind {
                TaskKind::NodeData {
                    path,
                    start_id,
                    rows,
                } => write_node_data_parquet(
                    &target,
                    &path,
                    start_id,
                    rows,
                    config.node_data_bytes,
                    config,
                    &pb,
                ),
                TaskKind::EdgeShard {
                    path,
                    dest_sort_key_bits,
                    src_min,
                    src_max,
                    src_total,
                    dst_total,
                    seed,
                } => write_edge_shard(
                    &target,
                    &path,
                    dest_sort_key_bits,
                    config,
                    src_min,
                    src_max,
                    src_total,
                    dst_total,
                    seed,
                    &pb,
                )
                .map(|_| ()),
            };
            match result {
                Ok(()) => {
                    pb.finish_and_clear();
                    global_bar.inc(1);
                    let done = global_bar.position();
                    global_bar.set_prefix(format!("{} / {}", format_count(done), format_count(total_tasks)));
                }
                Err(e) => {
                    pb.finish_with_message(format!("Failed: {}", task.label));
                    *err.lock().unwrap() = Some(e);
                }
            }
        }));
    }

    for handle in handles {
        let _ = handle.join();
    }
    let mut err = err.lock().unwrap();
    if let Some(e) = err.take() {
        global.finish_with_message("Overall failed");
        Err(e)
    } else {
        global.finish_with_message("Overall done");
        Ok(())
    }
}

fn write_edge_shard(
    target: &OutputTarget,
    path: &Path,
    dest_sort_key_bits: Option<u8>,
    config: TaskConfig,
    src_min: u64,
    src_max: u64,
    src_total: u64,
    dst_total: u64,
    seed: u64,
    progress: &ProgressBar,
) -> Result<(u64, u64)> {
    let schema = edge_schema(dest_sort_key_bits.is_some());
    let props = WriterProperties::builder()
        .set_max_row_group_size(config.row_group_size)
        .build();
    let sink = target.sink(path)?;
    let mut writer = ArrowWriter::try_new(sink, Arc::new(schema.clone()), Some(props))?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut src_buf: Vec<i64> = Vec::with_capacity(config.edge_batch_size);
    let mut dst_buf: Vec<i64> = Vec::with_capacity(config.edge_batch_size);
    let mut dsk_buf: Vec<i64> = Vec::with_capacity(config.edge_batch_size);

    if src_total == 0 || dst_total == 0 || config.edges_per_relation == 0 {
        let sink = writer.into_inner()?;
        sink.finish()?;
        return Ok((src_min, src_max));
    }

    let base_edges = config.edges_per_relation / src_total.max(1);
    let extra_edges = config.edges_per_relation % src_total.max(1);
    let uniform_dst = rand::distributions::Uniform::new_inclusive(0, (dst_total - 1) as i64);

    let mut processed = 0u64;
    for src_id in src_min..=src_max {
        let global_idx = src_id;
        let edges_here = base_edges + u64::from(global_idx < extra_edges);
        if edges_here == 0 {
            processed += 1;
            continue;
        }
        for idx in 0..edges_here {
            src_buf.push(src_id as i64);
            dst_buf.push(rng.sample(uniform_dst));
            if dest_sort_key_bits.is_some() {
                dsk_buf.push((edges_here - idx) as i64);
            }
            if src_buf.len() == config.edge_batch_size {
                flush_edge_batch(&schema, &mut writer, &mut src_buf, &mut dst_buf, &mut dsk_buf)?;
            }
        }
        processed += 1;
        update_progress(progress, processed);
    }
    flush_edge_batch(&schema, &mut writer, &mut src_buf, &mut dst_buf, &mut dsk_buf)?;
    let sink = writer.into_inner()?;
    sink.finish()?;
    Ok((src_min, src_max))
}

fn flush_edge_batch<W: Write + Send>(
    schema: &Schema,
    writer: &mut ArrowWriter<W>,
    src: &mut Vec<i64>,
    dst: &mut Vec<i64>,
    dsk: &mut Vec<i64>,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(if schema.fields().len() == 3 { 3 } else { 2 });
    cols.push(Arc::new(Int64Array::from(std::mem::take(src))) as ArrayRef);
    cols.push(Arc::new(Int64Array::from(std::mem::take(dst))) as ArrayRef);
    if schema.field_with_name("dest_sort_key").is_ok() {
        cols.push(Arc::new(Int64Array::from(std::mem::take(dsk))) as ArrayRef);
    } else {
        dsk.clear();
    }
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), cols)?;
    writer.write(&batch)?;
    Ok(())
}

fn edge_schema(with_sort_key: bool) -> Schema {
    let mut fields = vec![
        Field::new("src_id", DataType::Int64, false),
        Field::new("dst_id", DataType::Int64, false),
    ];
    if with_sort_key {
        fields.push(Field::new("dest_sort_key", DataType::Int64, false));
    }
    Schema::new(fields)
}

fn write_node_data_parquet(
    target: &OutputTarget,
    path: &Path,
    start_id: u64,
    rows: u64,
    bytes_per_node: usize,
    config: TaskConfig,
    progress: &ProgressBar,
) -> Result<()> {
    let schema = Schema::new(vec![Field::new("data", DataType::Binary, false)]);
    let props = WriterProperties::builder()
        .set_max_row_group_size(config.row_group_size)
        .build();
    let sink = target.sink(path)?;
    let mut writer = ArrowWriter::try_new(sink, Arc::new(schema.clone()), Some(props))?;

    let mut remaining = rows;
    let mut next_id = start_id;
    let mut processed = 0u64;
    let mut next_update = PROGRESS_STEP;
    while remaining > 0 {
        let take = remaining.min(config.node_data_batch_size as u64) as usize;
        let mut builder = BinaryBuilder::new();
        for i in 0..take {
            let id = next_id + i as u64;
            let bytes = node_data_bytes(id, bytes_per_node);
            builder.append_value(&bytes);
            processed += 1;
            if processed >= next_update {
                update_progress(progress, processed);
                next_update = next_update.saturating_add(PROGRESS_STEP);
            }
        }
        let array = builder.finish();
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)])?;
        writer.write(&batch)?;
        remaining -= take as u64;
        next_id += take as u64;
        update_progress(progress, processed);
    }
    let sink = writer.into_inner()?;
    sink.finish()?;
    Ok(())
}

fn node_data_bytes(node_id: u64, bytes_per_node: usize) -> Vec<u8> {
    if bytes_per_node == 0 {
        return Vec::new();
    }
    let text = format!("node:{node_id} fake data bytes_per_node={bytes_per_node}");
    let mut out = text.into_bytes();
    if out.len() > bytes_per_node {
        out.truncate(bytes_per_node);
    } else if out.len() < bytes_per_node {
        out.resize(bytes_per_node, b' ');
    }
    out
}

fn print_stats(
    args: &MultiArgs,
    metadata: &DatasetMetadata,
    target: &OutputTarget,
    elapsed: std::time::Duration,
) {
    println!("Stats:");
    println!("  nodes: {}", format_count_with_raw(args.nodes));
    println!(
        "  edges_per_relation: {}",
        format_count_with_raw(args.edges_per_relation)
    );
    println!("  node_types: {}", metadata.node_types.len());
    println!("  relations: {}", metadata.edge_relations.len());
    println!(
        "  node_data_bytes: {}",
        format_bytes_with_raw(args.node_data_bytes as u64)
    );
    println!("  time: {:.2}s", elapsed.as_secs_f64());
    match list_output_objects(target) {
        Ok(mut entries) => {
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let total: u64 = entries.iter().map(|(_, size)| *size).sum();
            if elapsed.as_secs_f64() > 0.0 {
                let rate = (total as f64 / elapsed.as_secs_f64()).round() as u64;
                println!("  throughput: {}/s", format_bytes(rate));
            }
            println!("  total_bytes: {}", format_bytes_with_raw(total));
            println!("  files:");
            for (name, size) in entries {
                println!("    {:<32} {}", name, format_bytes_with_raw(size));
            }
        }
        Err(err) => {
            println!("  total_bytes: n/a");
            println!("  files: n/a ({err})");
        }
    }
}

#[derive(Clone)]
enum OutputTarget {
    Local { root: PathBuf },
    S3 {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        display: String,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
    Gcs {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        display: String,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
    Adls {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        display: String,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
}

impl OutputTarget {
    fn parse(args: &MultiArgs) -> Result<Self> {
        match args.dest {
            Dest::Local => Ok(OutputTarget::Local {
                root: PathBuf::from(&args.output),
            }),
            Dest::S3 => {
                let (bucket, prefix) = parse_s3_url(&args.output)?;
                let part_concurrency = env::var("MULTIPART_CONCURRENCY")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.multipart_concurrency);
                let allow_http = args
                    .s3_allow_http
                    .or_else(|| env_flag("S3_ALLOW_HTTP"))
                    .or_else(|| env_flag("AWS_ALLOW_HTTP"))
                    .unwrap_or(false);
                let virtual_hosted = args
                    .s3_virtual_hosted
                    .or_else(|| env_flag("S3_VIRTUAL_HOSTED"))
                    .unwrap_or_else(|| {
                        args.s3_endpoint
                            .as_ref()
                            .map_or(true, |ep| ep.contains("amazonaws.com"))
                    });
                let profile = env::var("S3_PROFILE")
                    .ok()
                    .or_else(|| env::var("AWS_PROFILE").ok());
                let s3 = build_s3_client(S3Config {
                    bucket: bucket.clone(),
                    region: args.s3_region.clone(),
                    endpoint: args.s3_endpoint.clone(),
                    allow_http: Some(allow_http),
                    virtual_hosted: Some(virtual_hosted),
                    profile,
                    access_key_id: None,
                    secret_access_key: None,
                    session_token: None,
                })?;
                Ok(OutputTarget::S3 {
                    prefix,
                    store: s3.store,
                    display: args.output.clone(),
                    rt: Arc::new(Runtime::new()?),
                    part_size: args.s3_part_size,
                    part_concurrency,
                })
            }
            Dest::Gcs => {
                let (bucket, prefix) = parse_gs_url(&args.output)?;
                let part_concurrency = env::var("MULTIPART_CONCURRENCY")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.multipart_concurrency);
                let gcs = build_gcs_client(GcsConfig { bucket: bucket.clone() })?;
                Ok(OutputTarget::Gcs {
                    prefix,
                    store: gcs.store,
                    display: args.output.clone(),
                    rt: Arc::new(Runtime::new()?),
                    part_size: args.s3_part_size,
                    part_concurrency,
                })
            }
            Dest::Adls => {
                let (account, container, prefix) = parse_az_url(&args.output)?;
                let part_concurrency = env::var("MULTIPART_CONCURRENCY")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.multipart_concurrency);
                let allow_http = args
                    .azure_allow_http
                    .or_else(|| args.azure_endpoint.as_ref().map(|e| e.starts_with("http://")))
                    .unwrap_or(false);
                let adls = build_adls_client(AdlsConfig {
                    account: account.clone(),
                    container: container.clone(),
                    endpoint: args.azure_endpoint.clone(),
                    allow_http: Some(allow_http),
                    access_key: args
                        .azure_access_key
                        .clone()
                        .or_else(|| env::var("AZURE_ACCESS_KEY").ok()),
                    sas_key: args
                        .azure_sas_key
                        .clone()
                        .or_else(|| env::var("AZURE_SAS_KEY").ok()),
                    bearer_token: args
                        .azure_bearer_token
                        .clone()
                        .or_else(|| env::var("AZURE_BEARER_TOKEN").ok()),
                    client_id: args
                        .azure_client_id
                        .clone()
                        .or_else(|| env::var("AZURE_CLIENT_ID").ok()),
                    client_secret: args
                        .azure_client_secret
                        .clone()
                        .or_else(|| env::var("AZURE_CLIENT_SECRET").ok()),
                    tenant_id: args
                        .azure_tenant_id
                        .clone()
                        .or_else(|| env::var("AZURE_TENANT_ID").ok()),
                })?;
                Ok(OutputTarget::Adls {
                    prefix,
                    store: adls.store,
                    display: args.output.clone(),
                    rt: Arc::new(Runtime::new()?),
                    part_size: args.s3_part_size,
                    part_concurrency,
                })
            }
        }
    }

    fn prepare(&self) -> Result<()> {
        match self {
            OutputTarget::Local { root } => {
                fs::create_dir_all(root)?;
            }
            OutputTarget::S3 { .. } | OutputTarget::Gcs { .. } | OutputTarget::Adls { .. } => {}
        }
        Ok(())
    }

    fn join(&self, name: &str) -> PathBuf {
        match self {
            OutputTarget::Local { root } => root.join(name),
            OutputTarget::S3 { prefix, .. }
            | OutputTarget::Gcs { prefix, .. }
            | OutputTarget::Adls { prefix, .. } => {
                let p = if prefix.is_empty() {
                    name.to_string()
                } else {
                    format!("{}/{}", prefix.trim_end_matches('/'), name)
                };
                PathBuf::from(p)
            }
        }
    }

    fn sink(&self, path: &Path) -> Result<OutputSink> {
        match self {
            OutputTarget::Local { .. } => {
                let file = fs::File::create(path)?;
                Ok(OutputSink::Local(file))
            }
            OutputTarget::S3 {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            }
            | OutputTarget::Gcs {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            }
            | OutputTarget::Adls {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            } => {
                let obj = ObjectPath::from(path.to_string_lossy().to_string());
                let upload = rt.block_on(store.put_multipart(&obj)).map_err(to_io_err)?;
                Ok(OutputSink::Multipart(MultipartSink {
                    writer: Some(WriteMultipart::new_with_chunk_size(upload, *part_size)),
                    rt: rt.clone(),
                    max_concurrency: *part_concurrency,
                    handle: rt.handle().clone(),
                }))
            }
        }
    }

    fn write_bytes(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
        match self {
            OutputTarget::Local { root } => {
                fs::write(root.join(name), bytes)?;
                Ok(())
            }
            OutputTarget::S3 {
                store,
                rt,
                prefix,
                ..
            }
            | OutputTarget::Gcs {
                store,
                rt,
                prefix,
                ..
            }
            | OutputTarget::Adls {
                store,
                rt,
                prefix,
                ..
            } => {
                let obj = if prefix.is_empty() {
                    ObjectPath::from(name)
                } else {
                    ObjectPath::from(format!("{}/{}", prefix.trim_end_matches('/'), name))
                };
                rt.block_on(store.put(&obj, Bytes::from(bytes).into()))
                    .map_err(anyhow::Error::from)?;
                Ok(())
            }
        }
    }

    fn display(&self) -> String {
        match self {
            OutputTarget::Local { root } => root.display().to_string(),
            OutputTarget::S3 { display, .. }
            | OutputTarget::Gcs { display, .. }
            | OutputTarget::Adls { display, .. } => display.clone(),
        }
    }

    fn dataset_name(&self) -> String {
        match self {
            OutputTarget::Local { root } => root
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "multi".into()),
            OutputTarget::S3 { prefix, .. }
            | OutputTarget::Gcs { prefix, .. }
            | OutputTarget::Adls { prefix, .. } => prefix.clone(),
        }
    }
}

enum OutputSink {
    Local(fs::File),
    Multipart(MultipartSink),
}

impl OutputSink {
    fn finish(self) -> io::Result<()> {
        match self {
            OutputSink::Local(mut file) => file.flush(),
            OutputSink::Multipart(sink) => sink.finish(),
        }
    }
}

impl Write for OutputSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        match self {
            OutputSink::Local(file) => file.write(data),
            OutputSink::Multipart(sink) => sink.write(data),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            OutputSink::Local(file) => file.flush(),
            OutputSink::Multipart(sink) => sink.flush(),
        }
    }
}

struct MultipartSink {
    writer: Option<WriteMultipart>,
    rt: Arc<Runtime>,
    max_concurrency: usize,
    handle: tokio::runtime::Handle,
}

impl MultipartSink {
    fn finish(mut self) -> io::Result<()> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> io::Result<()> {
        let mut writer = match self.writer.take() {
            Some(writer) => writer,
            None => return Ok(()),
        };
        let rt = self.rt.clone();
        Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(0)))?;
        self.rt.block_on(writer.finish()).map_err(to_io_err)?;
        Ok(())
    }

    fn retry_with_backoff<T>(mut op: impl FnMut() -> Result<T, object_store::Error>) -> io::Result<T> {
        const RETRIES: usize = 3;
        const BASE_DELAY_MS: u64 = 250;
        let mut last_err: Option<object_store::Error> = None;
        for attempt in 0..=RETRIES {
            match op() {
                Ok(val) => return Ok(val),
                Err(err) => {
                    last_err = Some(err);
                    if attempt == RETRIES {
                        break;
                    }
                    let delay = BASE_DELAY_MS.saturating_mul(1 << attempt);
                    std::thread::sleep(Duration::from_millis(delay));
                }
            }
        }
        if let Some(err) = last_err {
            Err(to_io_err(err))
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "unknown multipart error",
            ))
        }
    }
}

impl Write for MultipartSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if let Some(writer) = self.writer.as_mut() {
            let rt = self.rt.clone();
            let max_concurrency = self.max_concurrency;
            Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(max_concurrency)))?;
            let _guard = self.handle.enter();
            writer.write(data);
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            let rt = self.rt.clone();
            Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(0)))?;
        }
        Ok(())
    }
}

impl Drop for MultipartSink {
    fn drop(&mut self) {
        if let Err(err) = self.finish_inner() {
            eprintln!("MultipartSink finish error on drop: {err}");
        }
    }
}

fn list_output_objects(target: &OutputTarget) -> Result<Vec<(String, u64)>> {
    let (store, rt, prefix) = match target {
        OutputTarget::S3 { store, rt, prefix, .. }
        | OutputTarget::Gcs { store, rt, prefix, .. }
        | OutputTarget::Adls { store, rt, prefix, .. } => (store, rt, prefix),
        OutputTarget::Local { root } => {
            return dir_files(root);
        }
    };
    let root = if prefix.is_empty() {
        None
    } else {
        Some(ObjectPath::from(prefix.clone()))
    };
    rt.block_on(async {
        let mut stream = store.list(root.as_ref());
        let mut entries = Vec::new();
        while let Some(res) = stream.next().await {
            let meta = res?;
            entries.push((meta.location.to_string(), meta.size as u64));
        }
        Ok(entries)
    })
}

fn dir_files(path: &Path) -> Result<Vec<(String, u64)>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        let entry_path = entry.path();
        if meta.is_dir() {
            let sub = dir_files(&entry_path)?;
            for (name, size) in sub {
                let rel = entry_path
                    .strip_prefix(path)
                    .unwrap_or(&entry_path)
                    .join(name);
                entries.push((rel.to_string_lossy().to_string(), size));
            }
        } else {
            let rel = entry_path.strip_prefix(path).unwrap_or(&entry_path);
            entries.push((rel.to_string_lossy().to_string(), meta.len()));
        }
    }
    Ok(entries)
}

fn to_io_err(err: object_store::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err)
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

fn env_flag(key: &str) -> Option<bool> {
    env::var(key)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
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

fn format_count_with_raw(value: u64) -> String {
    format!("{} ({})", value, format_count(value))
}

fn format_bytes_with_raw(value: u64) -> String {
    format!("{} ({})", value, format_bytes(value))
}

const PROGRESS_STEP: u64 = 100_000;

fn print_plan(
    node_types: &[NodeTypeMetadata],
    edge_relations: &[EdgeRelationMetadata],
    task_count: usize,
    threads: usize,
    node_data: bool,
) {
    let table_names = node_types
        .iter()
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let table_counts = node_types
        .iter()
        .map(|t| {
            let count = t.max_id.saturating_add(1);
            format!("{}={}", t.name, format_count_with_raw(count))
        })
        .collect::<Vec<_>>()
        .join(", ");
    let shard_counts = node_types
        .iter()
        .map(|t| format!("{}:{}", t.name, t.shards.len()))
        .collect::<Vec<_>>()
        .join(", ");
    let node_shards: usize = node_types.iter().map(|t| t.shards.len()).sum();
    let edge_shards: usize = edge_relations.iter().map(|r| r.shards.len()).sum();

    println!("Plan:");
    println!("  tables: {} ({})", node_types.len(), table_names);
    println!("  nodes: {}", table_counts);
    println!("  node shards: {} ({})", node_shards, shard_counts);
    let relation_counts = edge_relations
        .iter()
        .map(|r| format!("{}:{}", r.name, r.shards.len()))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "  relations: {} ({}), edge shards: {}",
        edge_relations.len(),
        relation_counts,
        edge_shards
    );
    println!(
        "  node data: {}, tasks: {}, workers: {}",
        node_data, task_count, threads
    );
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {prefix} ({eta})")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
}

fn update_progress(pb: &ProgressBar, position: u64) {
    if let Some(total) = pb.length() {
        if position % PROGRESS_STEP != 0 && position != total {
            return;
        }
        pb.set_position(position.min(total));
        pb.set_prefix(format!(
            "{} / {}",
            format_count(position),
            format_count(total),
        ));
    } else {
        if position % PROGRESS_STEP != 0 {
            return;
        }
        pb.set_position(position);
    }
}

fn parse_human_usize(raw: &str) -> Result<usize, String> {
    let value = parse_human_u64(raw)?;
    usize::try_from(value).map_err(|_| format!("value out of range: {raw}"))
}

fn parse_human_usize_bytes(raw: &str) -> Result<usize, String> {
    let value = parse_human_bytes(raw)?;
    usize::try_from(value).map_err(|_| format!("value out of range: {raw}"))
}

fn parse_human_u64(raw: &str) -> Result<u64, String> {
    let (number, suffix) = split_number_suffix(raw)?;
    let multiplier = match suffix.as_str() {
        "" => 1u64,
        "k" => 1_000u64,
        "m" => 1_000_000u64,
        "b" => 1_000_000_000u64,
        "t" => 1_000_000_000_000u64,
        _ => return Err(format!("invalid suffix: {raw}")),
    };
    number
        .checked_mul(multiplier)
        .ok_or_else(|| format!("value out of range: {raw}"))
}

fn parse_human_bytes(raw: &str) -> Result<u64, String> {
    let (number, suffix) = split_number_suffix(raw)?;
    let multiplier = match suffix.as_str() {
        "" | "b" => 1u64,
        "k" | "kb" => 1024u64,
        "m" | "mb" => 1024u64 * 1024u64,
        "g" | "gb" => 1024u64 * 1024u64 * 1024u64,
        "t" | "tb" => 1024u64 * 1024u64 * 1024u64 * 1024u64,
        _ => return Err(format!("invalid suffix: {raw}")),
    };
    number
        .checked_mul(multiplier)
        .ok_or_else(|| format!("value out of range: {raw}"))
}

fn split_number_suffix(raw: &str) -> Result<(u64, String), String> {
    let cleaned = raw.trim().replace('_', "");
    if cleaned.is_empty() {
        return Err("empty value".into());
    }
    let lower = cleaned.to_ascii_lowercase();
    let mut idx = 0;
    for (i, ch) in lower.char_indices() {
        if !ch.is_ascii_digit() {
            idx = i;
            break;
        }
        idx = i + ch.len_utf8();
    }
    let (num_part, suffix_part) = lower.split_at(idx);
    if num_part.is_empty() {
        return Err(format!("missing number: {raw}"));
    }
    let number: u64 = num_part
        .parse()
        .map_err(|_| format!("invalid number: {raw}"))?;
    Ok((number, suffix_part.to_string()))
}
