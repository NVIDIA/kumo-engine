//! Diskgraph data generator.
//!
//! Generates synthetic datasets in the on-disk layout expected by the
//! diskgraph ingest pipeline: a dataset root containing `index_metadata.json` and
//! Parquet shards for edge tables. Node shard ranges are recorded in metadata.
//! Scenarios cover combinations of
//! optional dest_sort_key fields and single vs multi-table layouts.
//! All shards are sorted: edges by (src_id, dest_sort_key desc when present).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use arrow_array::builder::BinaryBuilder;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, ValueEnum};
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, EdgeStorageFormat, NodeShardMetadata,
    NodeTypeMetadata,
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::distributions::Uniform;
use rand::prelude::*;
use tracing::info;

#[derive(Clone, Debug, ValueEnum)]
enum ScenarioKind {
    /// Single node type and relation, no optional fields.
    SimpleNoOptional,
    /// Temporal dest_sort_key (sorted shards).
    TemporalPresorted,
    /// Temporal weighted sampling mode (sorted shards).
    TemporalWeighted,
    /// Weighted-only dest_sort_key (no temporal filter).
    WeightedOnly,
    /// Multi-table mix of relations with and without optional dest_sort_key.
    MixedMultitable,
    /// Generate all scenarios above under one output directory.
    Coverage,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Output directory where dataset directories will be written.
    #[arg(short, long, value_name = "PATH")]
    output: PathBuf,
    /// Which scenario to generate.
    #[arg(short, long, value_enum, default_value = "coverage")]
    scenario: ScenarioKind,
    /// Total nodes per node type (used by scenarios that support override).
    #[arg(long, default_value_t = 10_000)]
    nodes: u64,
    /// Number of node shard ranges (recorded in metadata) and edge shards.
    #[arg(long, default_value_t = 4)]
    node_shards: usize,
    /// Total edges per relation (used by scenarios that support override).
    #[arg(long, default_value_t = 50_000)]
    edges: u64,
    /// Enable node data Parquet generation.
    #[arg(long, default_value_t = true)]
    node_data: bool,
    /// Disable node data Parquet generation.
    #[arg(long, default_value_t = false)]
    no_node_data: bool,
    /// Bytes per node in the node data column.
    #[arg(long, default_value_t = 128)]
    node_data_bytes: usize,
}

const SHARD_GAP: u64 = 7;

#[derive(Clone, Debug)]
struct ShardRange {
    start: u64,
    end: u64,
}

#[derive(Clone, Debug)]
struct ShardPlan {
    ranges: Vec<Option<ShardRange>>,
    max_id: u64,
}

type EdgeCountMap = HashMap<String, u64>;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let mut cli = Cli::parse();
    if cli.no_node_data {
        cli.node_data = false;
    }
    fs::create_dir_all(&cli.output)?;

    match cli.scenario.clone() {
        ScenarioKind::Coverage => {
            for scenario in [
                ScenarioKind::SimpleNoOptional,
                ScenarioKind::TemporalPresorted,
                ScenarioKind::TemporalWeighted,
                ScenarioKind::WeightedOnly,
                ScenarioKind::MixedMultitable,
            ] {
                generate_dataset(&cli, scenario)?;
            }
        }
        other => generate_dataset(&cli, other)?,
    }

    Ok(())
}

fn generate_dataset(cli: &Cli, scenario: ScenarioKind) -> anyhow::Result<()> {
    let start = std::time::Instant::now();
    let name = scenario_dir_name(&scenario);
    let root = cli.output.join(name);
    fs::create_dir_all(&root)?;
    info!(
        "generating scenario {:?} under {}",
        scenario,
        root.display()
    );

    let (mut node_types, mut edge_relations, edge_counts) = match scenario {
        ScenarioKind::SimpleNoOptional => build_simple(cli),
        ScenarioKind::TemporalPresorted => build_temporal_presorted(cli),
        ScenarioKind::TemporalWeighted => build_temporal_weighted(cli),
        ScenarioKind::WeightedOnly => build_weighted_only(cli),
        ScenarioKind::MixedMultitable => build_mixed(cli),
        ScenarioKind::Coverage => unreachable!("handled earlier"),
    };

    let shard_plan = shard_plan(cli.nodes, cli.node_shards, SHARD_GAP);
    let mut shard_ranges_by_type: HashMap<String, Vec<Option<ShardRange>>> = HashMap::new();
    for nt in node_types.iter_mut() {
        nt.max_id = shard_plan.max_id;
        shard_ranges_by_type.insert(nt.name.clone(), shard_plan.ranges.clone());
    }

    assign_node_shards(&mut node_types, &shard_ranges_by_type);
    if cli.node_data {
        write_node_data(&root, &mut node_types, cli.node_data_bytes)?;
    }
    let edge_rows = build_edge_shards(&edge_relations, &edge_counts, &shard_ranges_by_type);
    write_edges(&root, &mut edge_relations, edge_rows, cli.node_shards)?;
    write_metadata(&root, &node_types, &edge_relations)?;
    print_stats(
        &root,
        &scenario,
        &node_types,
        &edge_counts,
        cli.node_data_bytes,
        start.elapsed(),
    );
    Ok(())
}

/// Scenario: single node type, single relation, no optional fields.
fn build_simple(
    cli: &Cli,
) -> (
    Vec<NodeTypeMetadata>,
    Vec<EdgeRelationMetadata>,
    EdgeCountMap,
) {
    let node_types = vec![NodeTypeMetadata {
        name: "User".into(),
        max_id: 0,
        has_node_data: cli.node_data,
        shards: Vec::new(),
        node_data_compression: None,
        primary_key_map: None,
    }];
    let mut edge_counts = HashMap::new();
    edge_counts.insert("Follows".into(), cli.edges);
    let relations = vec![EdgeRelationMetadata {
        name: "Follows".into(),
        src_type: "User".into(),
        dst_type: "User".into(),
        shards: Vec::new(),
        dest_sort_key_bits: None,
        format: EdgeStorageFormat::Parquet,
    }];
    (node_types, relations, edge_counts)
}

/// Scenario: dest_sort_key present, shards presorted by (src_id, dest_sort_key desc).
fn build_temporal_presorted(
    cli: &Cli,
) -> (
    Vec<NodeTypeMetadata>,
    Vec<EdgeRelationMetadata>,
    EdgeCountMap,
) {
    let node_types = vec![NodeTypeMetadata {
        name: "Account".into(),
        max_id: 0,
        has_node_data: cli.node_data,
        shards: Vec::new(),
        node_data_compression: None,
        primary_key_map: None,
    }];
    let mut edge_counts = HashMap::new();
    edge_counts.insert("Transfers".into(), cli.edges);
    let relations = vec![EdgeRelationMetadata {
        name: "Transfers".into(),
        src_type: "Account".into(),
        dst_type: "Account".into(),
        shards: Vec::new(),
        dest_sort_key_bits: Some(32),
        format: EdgeStorageFormat::Parquet,
    }];
    (node_types, relations, edge_counts)
}

/// Scenario: dest_sort_key present for temporal weighted sampling.
fn build_temporal_weighted(
    cli: &Cli,
) -> (
    Vec<NodeTypeMetadata>,
    Vec<EdgeRelationMetadata>,
    EdgeCountMap,
) {
    let node_types = vec![NodeTypeMetadata {
        name: "Device".into(),
        max_id: 0,
        has_node_data: cli.node_data,
        shards: Vec::new(),
        node_data_compression: None,
        primary_key_map: None,
    }];
    let mut edge_counts = HashMap::new();
    edge_counts.insert("Interactions".into(), cli.edges);
    let relations = vec![EdgeRelationMetadata {
        name: "Interactions".into(),
        src_type: "Device".into(),
        dst_type: "Device".into(),
        shards: Vec::new(),
        dest_sort_key_bits: Some(16),
        format: EdgeStorageFormat::Parquet,
    }];
    (node_types, relations, edge_counts)
}

/// Scenario: dest_sort_key used purely as weight, no edge_id.
fn build_weighted_only(
    cli: &Cli,
) -> (
    Vec<NodeTypeMetadata>,
    Vec<EdgeRelationMetadata>,
    EdgeCountMap,
) {
    let node_types = vec![NodeTypeMetadata {
        name: "Item".into(),
        max_id: 0,
        has_node_data: cli.node_data,
        shards: Vec::new(),
        node_data_compression: None,
        primary_key_map: None,
    }];
    let mut edge_counts = HashMap::new();
    edge_counts.insert("SimilarTo".into(), cli.edges);
    let relations = vec![EdgeRelationMetadata {
        name: "SimilarTo".into(),
        src_type: "Item".into(),
        dst_type: "Item".into(),
        shards: Vec::new(),
        dest_sort_key_bits: Some(8),
        format: EdgeStorageFormat::Parquet,
    }];
    (node_types, relations, edge_counts)
}

/// Scenario: multi-table with mixed optional fields.
fn build_mixed(
    cli: &Cli,
) -> (
    Vec<NodeTypeMetadata>,
    Vec<EdgeRelationMetadata>,
    EdgeCountMap,
) {
    let node_types = vec![
        NodeTypeMetadata {
            name: "User".into(),
            max_id: 0,
            has_node_data: cli.node_data,
            shards: Vec::new(),
            node_data_compression: None,
            primary_key_map: None,
        },
        NodeTypeMetadata {
            name: "Ip".into(),
            max_id: 0,
            has_node_data: cli.node_data,
            shards: Vec::new(),
            node_data_compression: None,
            primary_key_map: None,
        },
        NodeTypeMetadata {
            name: "Merchant".into(),
            max_id: 0,
            has_node_data: cli.node_data,
            shards: Vec::new(),
            node_data_compression: None,
            primary_key_map: None,
        },
    ];

    let mut edge_counts = HashMap::new();
    edge_counts.insert("UserIp".into(), cli.edges / 2);
    edge_counts.insert("Purchases".into(), cli.edges / 2);

    let relations = vec![
        EdgeRelationMetadata {
            name: "UserIp".into(),
            src_type: "User".into(),
            dst_type: "Ip".into(),
            shards: Vec::new(),
            dest_sort_key_bits: Some(32),
            format: EdgeStorageFormat::Parquet,
        },
        EdgeRelationMetadata {
            name: "Purchases".into(),
            src_type: "User".into(),
            dst_type: "Merchant".into(),
            shards: Vec::new(),
            dest_sort_key_bits: None,
            format: EdgeStorageFormat::Parquet,
        },
    ];
    (node_types, relations, edge_counts)
}

#[derive(Clone)]
struct EdgeRow {
    src: i64,
    dst: i64,
    dest_sort_key: Option<i64>,
}

#[derive(Clone)]
struct EdgeShardData {
    rows: Vec<EdgeRow>,
    src_min: u64,
    src_max: u64,
}

fn shard_plan(total_nodes: u64, shards: usize, gap: u64) -> ShardPlan {
    let shard_count = shards.max(1) as u64;
    if total_nodes == 0 {
        return ShardPlan {
            ranges: vec![None; shard_count as usize],
            max_id: 0,
        };
    }
    let per_shard = ((total_nodes + shard_count - 1) / shard_count).max(1);
    let mut remaining = total_nodes;
    let mut start = 0u64;
    let mut ranges = Vec::with_capacity(shard_count as usize);
    let mut max_id = 0;
    for _ in 0..shard_count {
        if remaining == 0 {
            ranges.push(None);
            continue;
        }
        let len = per_shard.min(remaining);
        let end = start + len - 1;
        ranges.push(Some(ShardRange { start, end }));
        max_id = end;
        remaining -= len;
        start = end + 1 + gap;
    }
    // Ensure we produce exactly shard_count entries.
    while ranges.len() < shard_count as usize {
        ranges.push(None);
    }
    ShardPlan { ranges, max_id }
}

fn build_edge_shards(
    relations: &[EdgeRelationMetadata],
    edge_counts: &EdgeCountMap,
    shard_ranges_by_type: &HashMap<String, Vec<Option<ShardRange>>>,
) -> HashMap<String, Vec<EdgeShardData>> {
    let mut shards = HashMap::new();
    for rel in relations {
        let count = *edge_counts.get(&rel.name).unwrap_or(&0);
        let src_ranges = shard_ranges_by_type
            .get(&rel.src_type)
            .cloned()
            .unwrap_or_default();
        let dst_ranges: Vec<ShardRange> = shard_ranges_by_type
            .get(&rel.dst_type)
            .into_iter()
            .flat_map(|ranges| ranges.iter())
            .filter_map(|r| r.clone())
            .collect::<Vec<_>>();
        let rows = generate_edges_for_relation(rel, count, &src_ranges, &dst_ranges);
        shards.insert(rel.name.clone(), rows);
    }
    shards
}

fn generate_edges_for_relation(
    relation: &EdgeRelationMetadata,
    count: u64,
    src_ranges: &[Option<ShardRange>],
    dst_ranges: &[ShardRange],
) -> Vec<EdgeShardData> {
    let mut rng = StdRng::seed_from_u64(hash_seed(
        &relation.name,
        &relation.src_type,
        &relation.dst_type,
    ));
    if count == 0 {
        return vec![
            EdgeShardData {
                rows: Vec::new(),
                src_min: 0,
                src_max: 0,
            };
            src_ranges.len().max(1)
        ];
    }

    let active_shards: Vec<(usize, &ShardRange)> = src_ranges
        .iter()
        .enumerate()
        .filter_map(|(idx, r)| r.as_ref().map(|range| (idx, range)))
        .collect();
    if active_shards.is_empty() {
        return vec![
            EdgeShardData {
                rows: Vec::new(),
                src_min: 0,
                src_max: 0,
            };
            src_ranges.len().max(1)
        ];
    }

    let per_shard = ((count + active_shards.len() as u64 - 1) / active_shards.len() as u64).max(1);
    let mut remaining_edges = count;
    let mut shards_rows: Vec<EdgeShardData> = vec![
        EdgeShardData {
            rows: Vec::new(),
            src_min: 0,
            src_max: 0
        };
        src_ranges.len().max(1)
    ];

    for (idx, range) in active_shards {
        if remaining_edges == 0 {
            break;
        }
        let edges_here = per_shard.min(remaining_edges);
        let src_range = Uniform::new_inclusive(range.start as i64, range.end as i64);
        let mut shard_rows = Vec::new();
        for i in 0..edges_here {
            let src = rng.sample(src_range);
            let dst = sample_from_ranges(&mut rng, dst_ranges).unwrap_or(0);
            let dest_sort_key = relation.dest_sort_key_bits.map(|_| {
                let base = (edges_here * (idx as u64 + 1) - i) as i64;
                base + rng.gen_range(0..10)
            });
            shard_rows.push(EdgeRow {
                src,
                dst,
                dest_sort_key,
            });
        }
        shard_rows.sort_by(|a, b| {
            a.src
                .cmp(&b.src)
                .then_with(|| b.dest_sort_key.cmp(&a.dest_sort_key))
        });
        shards_rows[idx] = EdgeShardData {
            rows: shard_rows,
            src_min: range.start,
            src_max: range.end,
        };
        remaining_edges -= edges_here;
    }
    shards_rows
}

fn sample_from_ranges(rng: &mut StdRng, ranges: &[ShardRange]) -> Option<i64> {
    if ranges.is_empty() {
        return None;
    }
    let total: u64 = ranges.iter().map(|r| r.end - r.start + 1).sum();
    if total == 0 {
        return None;
    }
    let offset = rng.gen_range(0..total);
    let mut acc = offset;
    for range in ranges {
        let len = range.end - range.start + 1;
        if acc < len {
            return Some((range.start + acc) as i64);
        }
        acc -= len;
    }
    None
}

fn assign_node_shards(
    node_types: &mut [NodeTypeMetadata],
    shard_ranges: &HashMap<String, Vec<Option<ShardRange>>>,
) {
    for nt in node_types.iter_mut() {
        let ranges = shard_ranges
            .get(&nt.name)
            .cloned()
            .unwrap_or_else(|| vec![None; 1]);
        nt.shards.clear();
        for (_idx, maybe_range) in ranges.iter().enumerate() {
            if maybe_range.is_none() {
                nt.shards.push(NodeShardMetadata {
                    min_id: 0,
                    max_id: 0,
                    node_data_path: None,
                    node_data_paths: None,
                });
                continue;
            }
            let range = maybe_range.as_ref().unwrap();
            nt.shards.push(NodeShardMetadata {
                min_id: range.start,
                max_id: range.end,
                node_data_path: None,
                node_data_paths: None,
            });
        }
    }
}

fn write_node_data(
    root: &Path,
    node_types: &mut [NodeTypeMetadata],
    bytes_per_node: usize,
) -> anyhow::Result<()> {
    for nt in node_types.iter_mut() {
        if !nt.has_node_data {
            continue;
        }
        for (idx, shard) in nt.shards.iter_mut().enumerate() {
            let rows = shard
                .max_id
                .saturating_sub(shard.min_id)
                .saturating_add(1);
            let path = root.join(format!("nodes-{idx:05}.parquet"));
            write_node_data_parquet(&path, shard.min_id, rows, bytes_per_node)?;
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            shard.node_data_path = Some(rel);
        }
    }
    Ok(())
}

fn write_node_data_parquet(
    path: &Path,
    start_id: u64,
    rows: u64,
    bytes_per_node: usize,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let schema = Schema::new(vec![Field::new("data", DataType::Binary, false)]);
    let props = WriterProperties::builder().build();
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
    let mut remaining = rows;
    let mut next_id = start_id;
    let batch_size = 10_000u64;
    while remaining > 0 {
        let take = remaining.min(batch_size) as usize;
        let mut builder = BinaryBuilder::new();
        for i in 0..take {
            let id = next_id + i as u64;
            let bytes = node_data_bytes(id, bytes_per_node);
            builder.append_value(&bytes);
        }
        let array = builder.finish();
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)])?;
        writer.write(&batch)?;
        remaining -= take as u64;
        next_id += take as u64;
    }
    writer.close()?;
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
    root: &Path,
    scenario: &ScenarioKind,
    node_types: &[NodeTypeMetadata],
    edge_counts: &EdgeCountMap,
    node_data_bytes: usize,
    elapsed: std::time::Duration,
) {
    let total_nodes: u64 = node_types
        .iter()
        .map(|nt| nt.max_id.saturating_add(1))
        .sum();
    let total_edges: u64 = edge_counts.values().copied().sum();
    println!("Stats ({:?}):", scenario);
    println!("  nodes: {}", format_count(total_nodes));
    println!("  edges: {}", format_count(total_edges));
    println!("  relations: {}", edge_counts.len());
    println!("  node_data_bytes: {}", format_bytes(node_data_bytes as u64));
    println!("  time: {:.2}s", elapsed.as_secs_f64());
    if let Ok(bytes) = dir_size(root) {
        println!("  total_bytes: {}", format_bytes(bytes));
    }
    if let Ok(mut files) = dir_files(root) {
        files.sort_by(|a, b| a.0.cmp(&b.0));
        println!("  files:");
        for (name, size) in files {
            println!("    {:<32} {}", name, format_bytes(size));
        }
    }
}

fn dir_size(path: &Path) -> anyhow::Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

fn dir_files(path: &Path) -> anyhow::Result<Vec<(String, u64)>> {
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

fn write_edges(
    root: &Path,
    relations: &mut [EdgeRelationMetadata],
    shards: HashMap<String, Vec<EdgeShardData>>,
    shard_count: usize,
) -> anyhow::Result<()> {
    let base = root.join("edges");
    fs::create_dir_all(&base)?;

    for rel in relations.iter_mut() {
        let rel_dir = base.join(rel.name.to_lowercase());
        fs::create_dir_all(&rel_dir)?;
        let mut rel_rows = shards.get(&rel.name).cloned().unwrap_or_else(|| {
            vec![
                EdgeShardData {
                    rows: Vec::new(),
                    src_min: 0,
                    src_max: 0
                };
                shard_count
            ]
        });
        if rel_rows.len() < shard_count {
            rel_rows.resize_with(shard_count, || EdgeShardData {
                rows: Vec::new(),
                src_min: 0,
                src_max: 0,
            });
        }
        rel.shards.clear();
        for (idx, shard) in rel_rows.into_iter().enumerate() {
            let path = rel_dir.join(format!("part-{idx:05}.parquet"));
            let schema = edge_schema(rel);
            let batch = edge_batch(&schema, &shard.rows)?;
            write_parquet(&path, &schema, &[batch])?;
            let (src_min, src_max) = if shard.src_min > 0 || shard.src_max > 0 {
                (shard.src_min, shard.src_max)
            } else {
                (0, 0)
            };
            rel.shards.push(EdgeShardMetadata {
                path: format!("edges/{}/part-{idx:05}.parquet", rel.name.to_lowercase()),
                src_min_id: src_min,
                src_max_id: src_max,
            });
        }
    }
    Ok(())
}

fn edge_schema(rel: &EdgeRelationMetadata) -> Schema {
    let mut fields = vec![
        Field::new("src_id", DataType::Int64, false),
        Field::new("dst_id", DataType::Int64, false),
    ];
    if rel.dest_sort_key_bits.is_some() {
        fields.push(Field::new("dest_sort_key", DataType::Int64, false));
    }
    Schema::new(fields)
}

fn edge_batch(schema: &Schema, rows: &[EdgeRow]) -> anyhow::Result<RecordBatch> {
    let len = rows.len();
    let make_array = |vals: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(vals)) as ArrayRef };
    let src_vals: Vec<i64> = rows.iter().map(|r| r.src).collect();
    let dst_vals: Vec<i64> = rows.iter().map(|r| r.dst).collect();
    let mut columns: Vec<ArrayRef> = vec![make_array(src_vals), make_array(dst_vals)];

    if schema.field_with_name("dest_sort_key").is_ok() {
        let dsk_vals: Vec<i64> = rows
            .iter()
            .map(|r| r.dest_sort_key.unwrap_or_default())
            .collect();
        columns.push(make_array(dsk_vals));
    }
    // Ensure non-empty columns for empty shards by using length hints.
    let batch = if len == 0 {
        let empty_cols: Vec<ArrayRef> = columns
            .iter()
            .map(|_| {
                let array = Int64Array::from(Vec::<i64>::new());
                Arc::new(array) as ArrayRef
            })
            .collect();
        RecordBatch::try_new(Arc::new(schema.clone()), empty_cols)?
    } else {
        RecordBatch::try_new(Arc::new(schema.clone()), columns)?
    };
    Ok(batch)
}

fn write_parquet(path: &Path, schema: &Schema, batches: &[RecordBatch]) -> anyhow::Result<()> {
    let file = fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(())
}

fn write_metadata(
    root: &Path,
    node_types: &[NodeTypeMetadata],
    edge_relations: &[EdgeRelationMetadata],
) -> anyhow::Result<()> {
    let metadata = DatasetMetadata {
        dataset_name: root
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "synthetic".into()),
        node_types: node_types.to_vec(),
        edge_relations: edge_relations.to_vec(),
    };
    let path = root.join("index_metadata.json");
    let json = serde_json::to_string_pretty(&metadata)?;
    fs::write(path, json)?;
    Ok(())
}

fn scenario_dir_name(scenario: &ScenarioKind) -> String {
    match scenario {
        ScenarioKind::SimpleNoOptional => "simple_no_optional",
        ScenarioKind::TemporalPresorted => "temporal_presorted",
        ScenarioKind::TemporalWeighted => "temporal_weighted",
        ScenarioKind::WeightedOnly => "weighted_only",
        ScenarioKind::MixedMultitable => "mixed_multitable",
        ScenarioKind::Coverage => "coverage",
    }
    .into()
}

fn hash_seed(a: &str, b: &str, c: &str) -> u64 {
    let mut hasher = seahash::SeaHasher::new();
    use std::hash::Hash;
    (a, b, c).hash(&mut hasher);
    hasher.finish()
}

// Required for SeaHasher.
use seahash;
use std::hash::Hasher;
use std::sync::Arc;
