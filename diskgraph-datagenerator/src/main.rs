//! Diskgraph data generator.
//!
//! Generates synthetic datasets in the on-disk layout expected by the
//! diskgraph ingest pipeline: a dataset root containing `metadata.json` and
//! Parquet shards for node and edge tables. Scenarios cover combinations of
//! optional fields (dest_sort_key, edge_id), single vs multi-table layouts, and
//! presorted vs unsorted input shards.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, ValueEnum};
use diskgraph_core::metadata::{BitWidth, DatasetMetadata, EdgeRelationMetadata, NodeTypeMetadata};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::distributions::Uniform;
use rand::prelude::*;
use tracing::info;

#[derive(Clone, Debug, ValueEnum)]
enum ScenarioKind {
    /// Single node type and relation, no optional fields.
    SimpleNoOptional,
    /// Temporal dest_sort_key with presorted shards and edge_id.
    TemporalPresortedWithEdgeId,
    /// Temporal weighted sampling mode; shards unsorted to test robustness.
    TemporalWeightedUnsorted,
    /// Weighted-only dest_sort_key (no temporal filter) with edge_id absent.
    WeightedOnly,
    /// Multi-table mix of relations with and without optional fields.
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
    /// Total edges per relation (used by scenarios that support override).
    #[arg(long, default_value_t = 50_000)]
    edges: u64,
    /// Number of shards to split edge tables into.
    #[arg(long, default_value_t = 4)]
    edge_shards: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output)?;

    match cli.scenario.clone() {
        ScenarioKind::Coverage => {
            for scenario in [
                ScenarioKind::SimpleNoOptional,
                ScenarioKind::TemporalPresortedWithEdgeId,
                ScenarioKind::TemporalWeightedUnsorted,
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

    let (node_types, edge_relations, edge_rows) = match scenario {
        ScenarioKind::SimpleNoOptional => build_simple(cli),
        ScenarioKind::TemporalPresortedWithEdgeId => build_temporal_presorted(cli),
        ScenarioKind::TemporalWeightedUnsorted => build_temporal_weighted(cli),
        ScenarioKind::WeightedOnly => build_weighted_only(cli),
        ScenarioKind::MixedMultitable => build_mixed(cli),
        ScenarioKind::Coverage => unreachable!("handled earlier"),
    };

    write_nodes(&root, &node_types)?;
    write_edges(&root, &edge_relations, edge_rows, cli.edge_shards)?;
    write_metadata(&root, &node_types, &edge_relations)?;
    Ok(())
}

/// Scenario: single node type, single relation, no optional fields.
fn build_simple(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, Vec<EdgeRow>) {
    let node_types = vec![NodeTypeMetadata {
        name: "User".into(),
        max_id: cli.nodes - 1,
        estimated_count: Some(cli.nodes),
        id_width: BitWidth::Bits64,
    }];
    let edges = generate_edges(
        "Follows",
        "User",
        "User",
        cli.edges,
        cli.nodes - 1,
        cli.nodes - 1,
        None,
        None,
        true,
        cli.edge_shards,
    );
    let relations = vec![EdgeRelationMetadata {
        name: "Follows".into(),
        src_type: "User".into(),
        dst_type: "User".into(),
        shards: shard_paths("edges/follows", cli.edge_shards),
        dest_sort_key_bits: None,
        edge_id_bits: None,
        presorted_by_src_and_sort_key: false,
        estimated_count: Some(cli.edges),
    }];
    (node_types, relations, edges)
}

/// Scenario: dest_sort_key present, shards presorted by (src_id, dest_sort_key desc), edge_id present.
fn build_temporal_presorted(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, Vec<EdgeRow>) {
    let node_types = vec![NodeTypeMetadata {
        name: "Account".into(),
        max_id: cli.nodes - 1,
        estimated_count: Some(cli.nodes),
        id_width: BitWidth::Bits64,
    }];
    let edges = generate_edges(
        "Transfers",
        "Account",
        "Account",
        cli.edges,
        cli.nodes - 1,
        cli.nodes - 1,
        Some(BitWidth::Bits32),
        Some(BitWidth::Bits64),
        true,
        cli.edge_shards,
    );
    let relations = vec![EdgeRelationMetadata {
        name: "Transfers".into(),
        src_type: "Account".into(),
        dst_type: "Account".into(),
        shards: shard_paths("edges/transfers", cli.edge_shards),
        dest_sort_key_bits: Some(BitWidth::Bits32),
        edge_id_bits: Some(BitWidth::Bits64),
        presorted_by_src_and_sort_key: true,
        estimated_count: Some(cli.edges),
    }];
    (node_types, relations, edges)
}

/// Scenario: dest_sort_key present for temporal weighted sampling, shards unsorted, edge_id present.
fn build_temporal_weighted(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, Vec<EdgeRow>) {
    let node_types = vec![NodeTypeMetadata {
        name: "Device".into(),
        max_id: cli.nodes - 1,
        estimated_count: Some(cli.nodes),
        id_width: BitWidth::Bits64,
    }];
    let edges = generate_edges(
        "Interactions",
        "Device",
        "Device",
        cli.edges,
        cli.nodes - 1,
        cli.nodes - 1,
        Some(BitWidth::Bits16),
        Some(BitWidth::Bits32),
        false,
        cli.edge_shards,
    );
    let relations = vec![EdgeRelationMetadata {
        name: "Interactions".into(),
        src_type: "Device".into(),
        dst_type: "Device".into(),
        shards: shard_paths("edges/interactions", cli.edge_shards),
        dest_sort_key_bits: Some(BitWidth::Bits16),
        edge_id_bits: Some(BitWidth::Bits32),
        presorted_by_src_and_sort_key: false,
        estimated_count: Some(cli.edges),
    }];
    (node_types, relations, edges)
}

/// Scenario: dest_sort_key used purely as weight, no edge_id.
fn build_weighted_only(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, Vec<EdgeRow>) {
    let node_types = vec![NodeTypeMetadata {
        name: "Item".into(),
        max_id: cli.nodes - 1,
        estimated_count: Some(cli.nodes),
        id_width: BitWidth::Bits64,
    }];
    let edges = generate_edges(
        "SimilarTo",
        "Item",
        "Item",
        cli.edges,
        cli.nodes - 1,
        cli.nodes - 1,
        Some(BitWidth::Bits8),
        None,
        false,
        cli.edge_shards,
    );
    let relations = vec![EdgeRelationMetadata {
        name: "SimilarTo".into(),
        src_type: "Item".into(),
        dst_type: "Item".into(),
        shards: shard_paths("edges/similar_to", cli.edge_shards),
        dest_sort_key_bits: Some(BitWidth::Bits8),
        edge_id_bits: None,
        presorted_by_src_and_sort_key: false,
        estimated_count: Some(cli.edges),
    }];
    (node_types, relations, edges)
}

/// Scenario: multi-table with mixed optional fields.
fn build_mixed(cli: &Cli) -> (Vec<NodeTypeMetadata>, Vec<EdgeRelationMetadata>, Vec<EdgeRow>) {
    let node_types = vec![
        NodeTypeMetadata {
            name: "User".into(),
            max_id: cli.nodes - 1,
            estimated_count: Some(cli.nodes),
            id_width: BitWidth::Bits64,
        },
        NodeTypeMetadata {
            name: "Ip".into(),
            max_id: cli.nodes - 1,
            estimated_count: Some(cli.nodes),
            id_width: BitWidth::Bits32,
        },
        NodeTypeMetadata {
            name: "Merchant".into(),
            max_id: cli.nodes - 1,
            estimated_count: Some(cli.nodes),
            id_width: BitWidth::Bits64,
        },
    ];

    let mut edges = Vec::new();
    edges.extend(generate_edges(
        "UserIp",
        "User",
        "Ip",
        cli.edges / 2,
        cli.nodes - 1,
        cli.nodes - 1,
        Some(BitWidth::Bits32),
        None,
        true,
        cli.edge_shards,
    ));
    edges.extend(generate_edges(
        "Purchases",
        "User",
        "Merchant",
        cli.edges / 2,
        cli.nodes - 1,
        cli.nodes - 1,
        None,
        Some(BitWidth::Bits64),
        false,
        cli.edge_shards,
    ));

    let relations = vec![
        EdgeRelationMetadata {
            name: "UserIp".into(),
            src_type: "User".into(),
            dst_type: "Ip".into(),
            shards: shard_paths("edges/user_ip", cli.edge_shards),
            dest_sort_key_bits: Some(BitWidth::Bits32),
            edge_id_bits: None,
            presorted_by_src_and_sort_key: true,
            estimated_count: Some(cli.edges / 2),
        },
        EdgeRelationMetadata {
            name: "Purchases".into(),
            src_type: "User".into(),
            dst_type: "Merchant".into(),
            shards: shard_paths("edges/purchases", cli.edge_shards),
            dest_sort_key_bits: None,
            edge_id_bits: Some(BitWidth::Bits64),
            presorted_by_src_and_sort_key: false,
            estimated_count: Some(cli.edges / 2),
        },
    ];
    (node_types, relations, edges)
}

#[derive(Clone)]
struct EdgeRow {
    relation: String,
    src: i64,
    dst: i64,
    dest_sort_key: Option<i64>,
    edge_id: Option<i64>,
}

fn generate_edges(
    relation: &str,
    src_type: &str,
    dst_type: &str,
    count: u64,
    src_max: u64,
    dst_max: u64,
    dest_sort_key_bits: Option<BitWidth>,
    edge_id_bits: Option<BitWidth>,
    presorted: bool,
    shards: usize,
) -> Vec<EdgeRow> {
    let mut rng = StdRng::seed_from_u64(hash_seed(relation, src_type, dst_type));
    let src_range = Uniform::new_inclusive(0_i64, src_max as i64);
    let dst_range = Uniform::new_inclusive(0_i64, dst_max as i64);
    let mut rows: Vec<EdgeRow> = (0..count)
        .map(|i| {
            let src = rng.sample(src_range);
            let dst = rng.sample(dst_range);
            let dest_sort_key = dest_sort_key_bits.map(|_| {
                // For temporal scenarios, generate decreasing keys to allow presort.
                let base = (count - i) as i64;
                base + rng.gen_range(0..10)
            });
            let edge_id = edge_id_bits.map(|_| rng.gen_range(1..=i64::MAX));
            EdgeRow {
                relation: relation.to_string(),
                src,
                dst,
                dest_sort_key,
                edge_id,
            }
        })
        .collect();

    if presorted {
        rows.sort_by(|a, b| {
            a.src
                .cmp(&b.src)
                .then_with(|| b.dest_sort_key.cmp(&a.dest_sort_key))
        });
    } else {
        rows.shuffle(&mut rng);
    }

    // Trim or pad shards by duplicating minimal data to ensure every shard exists.
    if shards > 0 {
        let mut sharded_rows = Vec::new();
        let per_shard = (rows.len() / shards).max(1);
        for chunk in rows.chunks(per_shard) {
            sharded_rows.extend_from_slice(chunk);
        }
        rows = sharded_rows;
    }
    rows
}

fn shard_paths(base: &str, shards: usize) -> Vec<String> {
    (0..shards)
        .map(|i| format!("{base}/part-{i:05}.parquet"))
        .collect()
}

fn write_nodes(root: &Path, node_types: &[NodeTypeMetadata]) -> anyhow::Result<()> {
    let base = root.join("nodes");
    fs::create_dir_all(&base)?;
    for nt in node_types {
        let path = base.join(format!("{}_nodes.parquet", nt.name.to_lowercase()));
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let ids: Int64Array = (0..=nt.max_id as i64).collect();
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(ids) as ArrayRef])?;
        write_parquet(&path, &schema, &[batch])?;
    }
    Ok(())
}

fn write_edges(
    root: &Path,
    relations: &[EdgeRelationMetadata],
    rows: Vec<EdgeRow>,
    shards: usize,
) -> anyhow::Result<()> {
    let base = root.join("edges");
    fs::create_dir_all(&base)?;
    let mut by_relation: HashMap<String, Vec<EdgeRow>> = HashMap::new();
    for row in rows {
        by_relation.entry(row.relation.clone()).or_default().push(row);
    }

    for rel in relations {
        let rel_dir = base.join(rel.name.to_lowercase());
        fs::create_dir_all(&rel_dir)?;
        let rel_rows = by_relation.remove(&rel.name).unwrap_or_default();
        let per_shard = (rel_rows.len() / shards.max(1)).max(1);
        for idx in 0..shards.max(1) {
            let start = idx * per_shard;
            let end = ((idx + 1) * per_shard).min(rel_rows.len());
            let chunk = if start < rel_rows.len() {
                &rel_rows[start..end]
            } else {
                &[]
            };
            let path = rel_dir.join(format!("part-{idx:05}.parquet"));
            let schema = edge_schema(rel);
            let batch = edge_batch(&schema, chunk)?;
            write_parquet(&path, &schema, &[batch])?;
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
    if rel.edge_id_bits.is_some() {
        fields.push(Field::new("edge_id", DataType::Int64, false));
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
    if schema.field_with_name("edge_id").is_ok() {
        let eid_vals: Vec<i64> = rows.iter().map(|r| r.edge_id.unwrap_or_default()).collect();
        columns.push(make_array(eid_vals));
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
        dest_sort_key_name: Some("dest_sort_key".into()),
        edge_id_name: Some("edge_id".into()),
        source_parquet_bytes: None,
    };
    let path = root.join("metadata.json");
    let json = serde_json::to_string_pretty(&metadata)?;
    fs::write(path, json)?;
    Ok(())
}

fn scenario_dir_name(scenario: &ScenarioKind) -> String {
    match scenario {
        ScenarioKind::SimpleNoOptional => "simple_no_optional",
        ScenarioKind::TemporalPresortedWithEdgeId => "temporal_presorted_with_edge_id",
        ScenarioKind::TemporalWeightedUnsorted => "temporal_weighted_unsorted",
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
