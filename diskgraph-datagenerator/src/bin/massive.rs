//! Generate massive synthetic shards for benchmarking.
//!
//! This binary emits two edge shards (relations) plus node shard metadata and a
//! metadata.json describing the dataset. It streams Parquet row groups in
//! batches to keep memory usage low even for very large datasets (hundreds of
//! millions of nodes / billions of edges).

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
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, NodeShardMetadata, NodeTypeMetadata,
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

#[path = "../progress.rs"]
mod progress;

use progress::ProgressState;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate massive Parquet shards for benchmarking diskgraph ingest.")]
struct MassiveArgs {
    /// Output target; local path or s3 URL (e.g., s3://bucket/prefix).
    #[arg(long, value_name = "PATH_OR_URL")]
    output: String,
    /// Destination kind.
    #[arg(long, value_enum, default_value = "local")]
    dest: Dest,
    /// Number of nodes to generate.
    #[arg(long, default_value_t = 500_000_000)]
    nodes: u64,
    /// Number of edges to generate per relation (two relations are emitted).
    #[arg(long, default_value_t = 500_000_000)]
    edges_per_relation: u64,
    /// Maximum destination id (dst_id sampled uniformly from [0, max_dst_id]).
    #[arg(long, default_value_t = 100_000_000_000u64)]
    max_dst_id: u64,
    /// Rows per RecordBatch written to Parquet for edges (controls memory usage).
    #[arg(long, default_value_t = 1_000_000)]
    edge_batch_size: usize,
    /// Parquet row group size (rows); keep aligned with batch sizes for streaming writes.
    #[arg(long, default_value_t = 1_000_000)]
    row_group_size: usize,
    /// RNG seed (per relation a different salt is applied).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Node type name used in metadata.
    #[arg(long, default_value = "Node")]
    node_type_name: String,
    /// First relation name.
    #[arg(long, default_value = "rel_a")]
    relation0_name: String,
    /// Second relation name.
    #[arg(long, default_value = "rel_b")]
    relation1_name: String,
    /// Optional dest_sort_key bits for relation 0 (omit or set to 0 to disable).
    #[arg(long)]
    relation0_sort_bits: Option<u8>,
    /// Optional dest_sort_key bits for relation 1 (omit or set to 0 to disable).
    #[arg(long)]
    relation1_sort_bits: Option<u8>,
    /// S3 region override (dest=s3).
    #[arg(long)]
    s3_region: Option<String>,
    /// S3 endpoint override (dest=s3, for custom/minio).
    #[arg(long)]
    s3_endpoint: Option<String>,
    /// S3 multipart part size in bytes (default 64 MiB).
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
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
    /// Enable node data Parquet generation.
    #[arg(long, default_value_t = true)]
    node_data: bool,
    /// Disable node data Parquet generation.
    #[arg(long, default_value_t = false)]
    no_node_data: bool,
    /// Bytes per node in the node data column.
    #[arg(long, default_value_t = 128)]
    node_data_bytes: usize,
    /// Rows per RecordBatch written to Parquet for node data.
    #[arg(long, default_value_t = 1_000_000)]
    node_data_batch_size: usize,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Dest {
    Local,
    S3,
    Gcs,
    Adls,
}

#[derive(Clone)]
struct RelationConfig {
    name: String,
    dest_sort_key_bits: Option<u8>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let mut args = MassiveArgs::parse();
    if args.no_node_data {
        args.node_data = false;
    }
    let mut progress = ProgressState::new();
    let start = std::time::Instant::now();
    let target = OutputTarget::parse(&args)?;
    target.prepare()?;

    let edge0_path = target.join(&format!("edges_{}.parquet", args.relation0_name));
    let edge1_path = target.join(&format!("edges_{}.parquet", args.relation1_name));

    info!(
        "Generating massive dataset: nodes={}, edges_per_relation={}, max_dst_id={}, node_data={}, node_data_bytes={}",
        args.nodes, args.edges_per_relation, args.max_dst_id, args.node_data, args.node_data_bytes
    );
    let mut node_meta = NodeShardMetadata {
        min_id: 0,
        max_id: args.nodes.saturating_sub(1),
        node_data_path: None,
    };

    let rel0_cfg = RelationConfig {
        name: args.relation0_name.clone(),
        dest_sort_key_bits: normalize_bits(args.relation0_sort_bits),
    };
    let rel1_cfg = RelationConfig {
        name: args.relation1_name.clone(),
        dest_sort_key_bits: normalize_bits(args.relation1_sort_bits),
    };

    let edge0_meta = generate_edges(
        &target,
        &edge0_path,
        &rel0_cfg,
        &args,
        args.seed ^ 0xA5A5_0001,
        Some(&mut progress),
    )?;
    let edge1_meta = generate_edges(
        &target,
        &edge1_path,
        &rel1_cfg,
        &args,
        args.seed ^ 0xA5A5_0002,
        Some(&mut progress),
    )?;

    if args.node_data {
        let rel = "nodes.parquet".to_string();
        generate_node_data(&target, &rel, &args, Some(&mut progress))?;
        node_meta.node_data_path = Some(rel);
    }

    progress.start_stage("Writing metadata", None);
    write_metadata(
        &target,
        &args.node_type_name,
        node_meta,
        args.node_data,
        args.max_dst_id,
        rel0_cfg,
        edge0_meta,
        rel1_cfg,
        edge1_meta,
    )?;

    progress.finish("Done");
    print_stats(&args, &target, start.elapsed());
    info!("Done. Output written to {}", target.display());
    Ok(())
}

fn normalize_bits(bits: Option<u8>) -> Option<u8> {
    match bits {
        Some(0) => None,
        other => other,
    }
}

fn generate_edges(
    target: &OutputTarget,
    path: &Path,
    rel: &RelationConfig,
    args: &MassiveArgs,
    seed: u64,
    mut progress: Option<&mut ProgressState>,
) -> Result<EdgeShardMetadata> {
    let schema = edge_schema(rel.dest_sort_key_bits.is_some());
    let props = WriterProperties::builder()
        .set_max_row_group_size(args.row_group_size)
        .build();
    let sink = target.sink(path)?;
    let mut writer = ArrowWriter::try_new(sink, Arc::new(schema.clone()), Some(props))?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut src_buf: Vec<i64> = Vec::with_capacity(args.edge_batch_size);
    let mut dst_buf: Vec<i64> = Vec::with_capacity(args.edge_batch_size);
    let mut dsk_buf: Vec<i64> = Vec::with_capacity(args.edge_batch_size);

    let uniform_dst = rand::distributions::Uniform::new_inclusive(0, args.max_dst_id as i64);
    let base_edges = args.edges_per_relation / args.nodes.max(1);
    let extra_edges = args.edges_per_relation % args.nodes.max(1);
    let mut remaining = args.edges_per_relation;
    let total_nodes = if args.edges_per_relation == 0 {
        0
    } else {
        args.nodes.min(args.edges_per_relation)
    };
    if let Some(progress) = progress.as_deref_mut() {
        progress.start_stage(format!("Generating edges {}", rel.name), Some(total_nodes));
        progress.set_position(0);
    }
    let mut processed_nodes = 0u64;
    let mut next_report = 1_000_000u64;

    for src in 0..args.nodes {
        if remaining == 0 {
            break;
        }
        let mut edges_here = base_edges + u64::from(src < extra_edges);
        edges_here = edges_here.min(remaining);
        if edges_here == 0 {
            continue;
        }
        for idx in 0..edges_here {
            src_buf.push(src as i64);
            dst_buf.push(rng.sample(uniform_dst));
            if rel.dest_sort_key_bits.is_some() {
                // Descending per src_id.
                dsk_buf.push((edges_here - idx) as i64);
            }
            if src_buf.len() == args.edge_batch_size {
                flush_edge_batch(&schema, &mut writer, &mut src_buf, &mut dst_buf, &mut dsk_buf)?;
            }
        }
        remaining -= edges_here;
        processed_nodes += 1;
        if let Some(progress) = progress.as_deref_mut() {
            if processed_nodes >= next_report {
                progress.set_position(processed_nodes);
                next_report += 1_000_000;
            }
        }
    }
    flush_edge_batch(&schema, &mut writer, &mut src_buf, &mut dst_buf, &mut dsk_buf)?;
    let sink = writer.into_inner()?;
    sink.finish()?;
    if let Some(progress) = progress.as_deref_mut() {
        progress.set_position(processed_nodes);
    }

    Ok(EdgeShardMetadata {
        path: format!("edges_{}.parquet", rel.name),
        src_min_id: 0,
        src_max_id: args.nodes.saturating_sub(1),
    })
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

fn write_metadata(
    root: &OutputTarget,
    node_type_name: &str,
    node: NodeShardMetadata,
    has_node_data: bool,
    max_dst_id: u64,
    rel0: RelationConfig,
    edge0: EdgeShardMetadata,
    rel1: RelationConfig,
    edge1: EdgeShardMetadata,
) -> Result<()> {
    let node_meta = NodeTypeMetadata {
        name: node_type_name.into(),
        max_id: node.max_id.max(max_dst_id),
        has_node_data,
        shards: vec![node],
    };
    let edge_relations = vec![
        EdgeRelationMetadata {
            name: rel0.name.clone(),
            src_type: node_type_name.into(),
            dst_type: node_type_name.into(),
            shards: vec![edge0],
            dest_sort_key_bits: rel0.dest_sort_key_bits,
        },
        EdgeRelationMetadata {
            name: rel1.name.clone(),
            src_type: node_type_name.into(),
            dst_type: node_type_name.into(),
            shards: vec![edge1],
            dest_sort_key_bits: rel1.dest_sort_key_bits,
        },
    ];
    let metadata = DatasetMetadata {
        dataset_name: root.dataset_name(),
        node_types: vec![node_meta],
        edge_relations,
    };
    let json = serde_json::to_string_pretty(&metadata)?;
    root.write_bytes("metadata.json", json.into_bytes())
}

fn generate_node_data(
    target: &OutputTarget,
    rel_path: &str,
    args: &MassiveArgs,
    mut progress: Option<&mut ProgressState>,
) -> Result<()> {
    let path = target.join(rel_path);
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }

    let schema = Schema::new(vec![Field::new("data", DataType::Binary, false)]);
    let props = WriterProperties::builder()
        .set_max_row_group_size(args.row_group_size)
        .build();
    let sink = target.sink(&path)?;
    let mut writer = ArrowWriter::try_new(sink, Arc::new(schema.clone()), Some(props))?;

    if let Some(progress) = progress.as_deref_mut() {
        progress.start_stage("Generating node data", Some(args.nodes));
        progress.set_position(0);
    }
    let mut remaining = args.nodes;
    let mut next_id = 0u64;
    let mut processed = 0u64;
    while remaining > 0 {
        let take = remaining.min(args.node_data_batch_size as u64) as usize;
        let mut builder = BinaryBuilder::new();
        for i in 0..take {
            let id = next_id + i as u64;
            let bytes = node_data_bytes(id, args.node_data_bytes);
            builder.append_value(&bytes);
        }
        let array = builder.finish();
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)])?;
        writer.write(&batch)?;
        remaining -= take as u64;
        next_id += take as u64;
        processed += take as u64;
        if let Some(progress) = progress.as_deref_mut() {
            progress.set_position(processed);
        }
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

fn print_stats(args: &MassiveArgs, target: &OutputTarget, elapsed: std::time::Duration) {
    println!("Stats:");
    println!("  nodes: {}", format_count(args.nodes));
    println!("  edges_per_relation: {}", format_count(args.edges_per_relation));
    println!("  relations: 2");
    println!("  node_data_bytes: {}", format_bytes(args.node_data_bytes as u64));
    println!("  time: {:.2}s", elapsed.as_secs_f64());
    if let OutputTarget::Local { root } = target {
        let files = [
            root.join(format!("edges_{}.parquet", args.relation0_name)),
            root.join(format!("edges_{}.parquet", args.relation1_name)),
            root.join("metadata.json"),
            root.join("nodes.parquet"),
        ];
        let mut entries = Vec::new();
        let mut total = 0u64;
        for path in files {
            if let Ok(meta) = fs::metadata(&path) {
                let size = meta.len();
                total += size;
                let name = path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.display().to_string());
                entries.push((name, size));
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        println!("  total_bytes: {}", format_bytes(total));
        println!("  files:");
        for (name, size) in entries {
            println!("    {:<24} {}", name, format_bytes(size));
        }
    } else {
        println!("  total_bytes: n/a");
        println!("  files: n/a (remote)");
    }
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
    fn parse(args: &MassiveArgs) -> Result<Self> {
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
                let store: Arc<dyn ObjectStore> = s3.store;
                Ok(OutputTarget::S3 {
                    prefix,
                    store,
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
                let store: Arc<dyn ObjectStore> = gcs.store;
                let rt = Arc::new(Runtime::new()?);
                Ok(OutputTarget::Gcs {
                    prefix,
                    store,
                    display: args.output.clone(),
                    rt,
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
                let store: Arc<dyn ObjectStore> = adls.store;
                let rt = Arc::new(Runtime::new()?);
                Ok(OutputTarget::Adls {
                    prefix,
                    store,
                    display: args.output.clone(),
                    rt,
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
                let upload = rt
                    .block_on(store.put_multipart(&obj))
                    .map_err(to_io_err)?;
                Ok(OutputSink::Multipart(MultipartSink {
                    writer: Some(WriteMultipart::new_with_chunk_size(
                        upload,
                        *part_size,
                    )),
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
                .unwrap_or_else(|| "massive".into()),
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
        // Ensure in-flight parts have progressed; does not force a partial chunk upload.
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
