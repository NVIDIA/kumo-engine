//! Diskgraph data generator.
//!
//! Generates synthetic datasets in the on-disk layout expected by the
//! diskgraph ingest pipeline: a dataset root containing `metadata.json` and
//! Parquet shards for node and edge tables. Scenarios cover combinations of
//! optional dest_sort_key fields and single vs multi-table layouts.
//! All shards are sorted: nodes by id, edges by (src_id, dest_sort_key desc when present).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, ValueEnum};
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, NodeShardMetadata, NodeTypeMetadata,
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
    /// Number of shards to split node tables into.
    #[arg(long, default_value_t = 4)]
    node_shards: usize,
    /// Total edges per relation (used by scenarios that support override).
    #[arg(long, default_value_t = 50_000)]
    edges: u64,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let cli = Cli::parse();
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
    let name = scenario_dir_name(&scenario);
    let root = cli.output.join(name);
    fs::create_dir_all(&root)?;
    info!("generating scenario {:?} under {}", scenario, root.display());

    let (mut node_types, mut edge_relations, edge_rows) = match scenario {
        ScenarioKind::SimpleNoOptional => build_simple(cli),
        ScenarioKind::TemporalPresorted => build_temporal_presorted(cli),
        ScenarioKind::TemporalWeighted => build_temporal_weighted(cli),
        ScenarioKind::WeightedOnly => build_weighted_only(cli),
        ScenarioKind::MixedMultitable => build_mixed(cli),
        ScenarioKind::Coverage => unreachable!("handled earlier"),
    };

    write_nodes(&root, &mut node_types, cli.node_shards)?;
    write_edges(&root, &mut edge_relations, edge_rows, cli.node_shards)?;
    write_metadata(&root, &node_types, &edge_relations)?;
    Ok(())
}

/// Scenario: single node type, single relation, no optional fields.
fn build_simple(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, HashMap<String, Vec<EdgeShardData>>) {
    let node_types = vec![NodeTypeMetadata {
        name: "User".into(),
        max_id: cli.nodes - 1,
        shards: Vec::new(),
    }];
    let edges = generate_edges(
        "Follows",
        "User",
        "User",
        cli.edges,
        cli.nodes - 1,
        cli.nodes - 1,
        None,
        cli.node_shards,
    );
    let relations = vec![EdgeRelationMetadata {
        name: "Follows".into(),
        src_type: "User".into(),
        dst_type: "User".into(),
        shards: Vec::new(),
        dest_sort_key_bits: None,
    }];
    (node_types, relations, edges)
}

/// Scenario: dest_sort_key present, shards presorted by (src_id, dest_sort_key desc).
fn build_temporal_presorted(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, HashMap<String, Vec<EdgeShardData>>) {
    let node_types = vec![NodeTypeMetadata {
        name: "Account".into(),
        max_id: cli.nodes - 1,
        shards: Vec::new(),
    }];
    let edges = generate_edges(
        "Transfers",
        "Account",
        "Account",
        cli.edges,
        cli.nodes - 1,
        cli.nodes - 1,
        Some(32),
        cli.node_shards,
    );
    let relations = vec![EdgeRelationMetadata {
        name: "Transfers".into(),
        src_type: "Account".into(),
        dst_type: "Account".into(),
        shards: Vec::new(),
        dest_sort_key_bits: Some(32),
    }];
    (node_types, relations, edges)
}

/// Scenario: dest_sort_key present for temporal weighted sampling.
fn build_temporal_weighted(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, HashMap<String, Vec<EdgeShardData>>) {
    let node_types = vec![NodeTypeMetadata {
        name: "Device".into(),
        max_id: cli.nodes - 1,
        shards: Vec::new(),
    }];
    let edges = generate_edges(
        "Interactions",
        "Device",
        "Device",
        cli.edges,
        cli.nodes - 1,
        cli.nodes - 1,
        Some(16),
        cli.node_shards,
    );
    let relations = vec![EdgeRelationMetadata {
        name: "Interactions".into(),
        src_type: "Device".into(),
        dst_type: "Device".into(),
        shards: Vec::new(),
        dest_sort_key_bits: Some(16),
    }];
    (node_types, relations, edges)
}

/// Scenario: dest_sort_key used purely as weight, no edge_id.
fn build_weighted_only(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, HashMap<String, Vec<EdgeShardData>>) {
    let node_types = vec![NodeTypeMetadata {
        name: "Item".into(),
        max_id: cli.nodes - 1,
        shards: Vec::new(),
    }];
    let edges = generate_edges(
        "SimilarTo",
        "Item",
        "Item",
        cli.edges,
        cli.nodes - 1,
        cli.nodes - 1,
        Some(8),
        cli.node_shards,
    );
    let relations = vec![EdgeRelationMetadata {
        name: "SimilarTo".into(),
        src_type: "Item".into(),
        dst_type: "Item".into(),
        shards: Vec::new(),
        dest_sort_key_bits: Some(8),
    }];
    (node_types, relations, edges)
}

/// Scenario: multi-table with mixed optional fields.
fn build_mixed(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, HashMap<String, Vec<EdgeShardData>>) {
    let node_types = vec![
        NodeTypeMetadata {
            name: "User".into(),
            max_id: cli.nodes - 1,
            shards: Vec::new(),
        },
        NodeTypeMetadata {
            name: "Ip".into(),
            max_id: cli.nodes - 1,
            shards: Vec::new(),
        },
        NodeTypeMetadata {
            name: "Merchant".into(),
            max_id: cli.nodes - 1,
            shards: Vec::new(),
        },
    ];

    let mut edges = HashMap::new();
    edges.extend(generate_edges(
        "UserIp",
        "User",
        "Ip",
        cli.edges / 2,
        cli.nodes - 1,
        cli.nodes - 1,
        Some(32),
        cli.node_shards,
    ));
    edges.extend(generate_edges(
        "Purchases",
        "User",
        "Merchant",
        cli.edges / 2,
        cli.nodes - 1,
        cli.nodes - 1,
        None,
        cli.node_shards,
    ));

    let relations = vec![
        EdgeRelationMetadata {
            name: "UserIp".into(),
            src_type: "User".into(),
            dst_type: "Ip".into(),
            shards: Vec::new(),
            dest_sort_key_bits: Some(32),
        },
        EdgeRelationMetadata {
            name: "Purchases".into(),
            src_type: "User".into(),
            dst_type: "Merchant".into(),
            shards: Vec::new(),
            dest_sort_key_bits: None,
        },
    ];
    (node_types, relations, edges)
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

fn generate_edges(
    relation: &str,
    src_type: &str,
    dst_type: &str,
    count: u64,
    src_max: u64,
    dst_max: u64,
    dest_sort_key_bits: Option<u8>,
    shards: usize,
) -> HashMap<String, Vec<EdgeShardData>> {
    let mut rng = StdRng::seed_from_u64(hash_seed(relation, src_type, dst_type));
    let ranges = shard_ranges(src_max, shards);
    let dst_range = Uniform::new_inclusive(0_i64, dst_max as i64);
    if count == 0 {
        let mut map = HashMap::new();
        map.insert(relation.to_string(), vec![EdgeShardData { rows: Vec::new(), src_min: 0, src_max: 0 }; shards.max(1)]);
        return map;
    }
    let per_shard = ((count + shards as u64 - 1) / shards as u64).max(1);
    let mut shards_rows: Vec<EdgeShardData> = Vec::with_capacity(shards);
    for (idx, (start, end)) in ranges.iter().enumerate() {
        if *end < *start {
            shards_rows.push(EdgeShardData {
                rows: Vec::new(),
                src_min: 0,
                src_max: 0,
            });
            continue;
        }
        let src_range = Uniform::new_inclusive(*start, *end);
        let mut shard_rows = Vec::new();
        for i in 0..per_shard {
            let src = rng.sample(src_range);
            let dst = rng.sample(dst_range);
            let dest_sort_key = dest_sort_key_bits.map(|_| {
                let base = (per_shard * (idx as u64 + 1) - i) as i64;
                base + rng.gen_range(0..10)
            });
            shard_rows.push(EdgeRow { src, dst, dest_sort_key });
        }
        // Sort shard by src, then dest_sort_key descending if present.
        shard_rows.sort_by(|a, b| {
            a.src
                .cmp(&b.src)
                .then_with(|| b.dest_sort_key.cmp(&a.dest_sort_key))
        });
        shards_rows.push(EdgeShardData {
            rows: shard_rows,
            src_min: *start as u64,
            src_max: *end as u64,
        });
    }
    let mut map = HashMap::new();
    map.insert(relation.to_string(), shards_rows);
    map
}

fn shard_ranges(max_id: u64, shards: usize) -> Vec<(i64, i64)> {
    let shard_count = shards.max(1) as u64;
    let per_shard = ((max_id + 1) + shard_count - 1) / shard_count; // ceil div
    let mut ranges = Vec::with_capacity(shard_count as usize);
    let mut start = 0_i64;
    for _ in 0..shard_count {
        let end = (start as u64 + per_shard).min(max_id + 1) as i64 - 1;
        ranges.push((start, end.max(start)));
        start = end + 1;
        if start as u64 > max_id {
            break;
        }
    }
    // Ensure we produce exactly shard_count entries.
    while ranges.len() < shard_count as usize {
        ranges.push((0, -1)); // empty range for padding shards if max_id < shard count
    }
    ranges
}

fn write_nodes(root: &Path, node_types: &mut [NodeTypeMetadata], shards: usize) -> anyhow::Result<()> {
    let base = root.join("nodes");
    fs::create_dir_all(&base)?;
    for nt in node_types.iter_mut() {
        let nt_dir = base.join(nt.name.to_lowercase());
        fs::create_dir_all(&nt_dir)?;
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let ranges = shard_ranges(nt.max_id, shards);
        nt.shards.clear();
        for (idx, (start, end)) in ranges.iter().enumerate() {
            if *end < *start {
                // Empty shard placeholder.
                let empty = Int64Array::from(Vec::<i64>::new());
                let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(empty) as ArrayRef])?;
                let path = nt_dir.join(format!("part-{idx:05}.parquet"));
                write_parquet(&path, &schema, &[batch])?;
                nt.shards.push(NodeShardMetadata {
                    path: format!("nodes/{}/part-{idx:05}.parquet", nt.name.to_lowercase()),
                    min_id: 0,
                    max_id: 0,
                });
                continue;
            }
            let ids: Int64Array = (*start..=*end).collect();
            let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(ids) as ArrayRef])?;
            let path = nt_dir.join(format!("part-{idx:05}.parquet"));
            write_parquet(&path, &schema, &[batch])?;
            nt.shards.push(NodeShardMetadata {
                path: format!("nodes/{}/part-{idx:05}.parquet", nt.name.to_lowercase()),
                min_id: *start as u64,
                max_id: *end as u64,
            });
        }
    }
    Ok(())
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
        let mut rel_rows = shards
            .get(&rel.name)
            .cloned()
            .unwrap_or_else(|| vec![EdgeShardData { rows: Vec::new(), src_min: 0, src_max: 0 }; shard_count]);
        if rel_rows.len() < shard_count {
            rel_rows.resize_with(
                shard_count,
                || EdgeShardData {
                    rows: Vec::new(),
                    src_min: 0,
                    src_max: 0,
                },
            );
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
    let path = root.join("metadata.json");
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
use std::hash::Hasher;
use seahash;
use std::sync::Arc;
