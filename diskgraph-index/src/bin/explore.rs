//! Interactive shard explorer for diskgraph-index.
//!
//! Supports local graphs by opening a pre-built index and providing a simple
//! REPL for inspecting edge lists by node id.

use anyhow::{anyhow, Context, Result};
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeSortKeyTimestampKind, EdgeSortKeyWeightKind,
};
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_index::layout::{EDGES_FILENAME, OFFSETS_FILENAME};
use diskgraph_rowpack::{ColumnEncoding, DecimalStorage, FloatStorage, RowPackContext, StringEncoding};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write, IsTerminal};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use half::f16;

#[derive(Clone, Debug)]
enum Source {
    Local(PathBuf),
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
    #[serde(default)]
    edge_sort_keys: Option<EdgeSortKeysReport>,
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
    source_dtype: ColumnTypeReport,
    ingested_dtype: ColumnTypeReport,
    #[serde(default)]
    fixed_len: Option<usize>,
    #[serde(default)]
    key_mode: Option<KeyModeReport>,
    #[serde(default)]
    timestamp_format: Option<String>,
    #[serde(default)]
    timestamp_unit: Option<TimestampUnitReport>,
    #[serde(default)]
    precision: Option<u32>,
    #[serde(default)]
    scale: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct EdgeSortKeysReport {
    #[serde(default)]
    relationships: HashMap<String, RelationshipSortKeysReport>,
    #[serde(default)]
    edge_tables: HashMap<String, EdgeTableSortKeysReport>,
}

#[derive(Debug, Deserialize)]
struct RelationshipSortKeysReport {
    #[serde(default)]
    forward: Option<SortKeyReport>,
    #[serde(default)]
    reverse: Option<SortKeyReport>,
}

#[derive(Debug, Deserialize)]
struct EdgeTableSortKeysReport {
    #[serde(default)]
    src_to_dst: Option<SortKeyReport>,
    #[serde(default)]
    dst_to_src: Option<SortKeyReport>,
}

#[derive(Debug, Deserialize)]
struct SortKeyReport {
    #[serde(default)]
    timestamp: Option<SortKeyTimestampReport>,
    #[serde(default)]
    weight: Option<SortKeyWeightReport>,
}

#[derive(Debug, Deserialize)]
struct SortKeyTimestampReport {
    table: String,
    column: String,
    encoding: SortKeyTimestampEncodingReport,
}

#[derive(Debug, Deserialize)]
struct SortKeyTimestampEncodingReport {
    kind: SortKeyTimestampKindReport,
    anchor: i64,
    step: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SortKeyTimestampKindReport {
    Int16,
    Int32,
}

#[derive(Debug, Deserialize)]
struct SortKeyWeightReport {
    table: String,
    expr: String,
    encoding: SortKeyWeightEncodingReport,
}

#[derive(Debug, Deserialize)]
struct SortKeyWeightEncodingReport {
    dtype: SortKeyWeightKindReport,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SortKeyWeightKindReport {
    Byte,
    UInt16,
    UInt32,
    Float16,
    Float32,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ColumnTypeReport {
    Int8,
    Int16,
    Int64,
    Int32,
    UInt8,
    UInt16,
    UInt64,
    UInt32,
    Float64,
    Float32,
    Bool,
    String,
    Bytes,
    Date,
    Timestamp,
    Decimal128,
    Decimal256,
    Float32Array,
    Float64Array,
}

impl ColumnTypeReport {
    fn as_str(&self) -> &'static str {
        match self {
            ColumnTypeReport::Int8 => "int8",
            ColumnTypeReport::Int16 => "int16",
            ColumnTypeReport::Int64 => "int64",
            ColumnTypeReport::Int32 => "int32",
            ColumnTypeReport::UInt8 => "uint8",
            ColumnTypeReport::UInt16 => "uint16",
            ColumnTypeReport::UInt64 => "uint64",
            ColumnTypeReport::UInt32 => "uint32",
            ColumnTypeReport::Float64 => "float64",
            ColumnTypeReport::Float32 => "float32",
            ColumnTypeReport::Bool => "bool",
            ColumnTypeReport::String => "string",
            ColumnTypeReport::Bytes => "bytes",
            ColumnTypeReport::Date => "date",
            ColumnTypeReport::Timestamp => "timestamp",
            ColumnTypeReport::Decimal128 => "decimal128",
            ColumnTypeReport::Decimal256 => "decimal256",
            ColumnTypeReport::Float32Array => "float32_array",
            ColumnTypeReport::Float64Array => "float64_array",
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TimestampUnitReport {
    Seconds,
    Millis,
    Micros,
    Nanos,
}

impl TimestampUnitReport {
    fn as_str(&self) -> &'static str {
        match self {
            TimestampUnitReport::Seconds => "seconds",
            TimestampUnitReport::Millis => "millis",
            TimestampUnitReport::Micros => "micros",
            TimestampUnitReport::Nanos => "nanos",
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum KeyModeReport {
    Fixed,
    Uuid,
}

impl KeyModeReport {
    fn as_str(&self) -> &'static str {
        match self {
            KeyModeReport::Fixed => "fixed",
            KeyModeReport::Uuid => "uuid",
        }
    }
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
    #[serde(default)]
    node_data_trimmed: u64,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let mut source_str: Option<String> = None;
    while let Some(arg) = args.next() {
        if arg.starts_with("--") {
            return Err(anyhow!("unknown flag: {arg}"));
        }
        if source_str.is_none() {
            source_str = Some(arg);
        }
    }
    let source_str = source_str.ok_or_else(|| {
        anyhow!("usage: explore <source> (local path)")
    })?;
    run_explore(&source_str)
}

fn run_explore(source_str: &str) -> Result<()> {
    let source = parse_source(source_str)?;
    let load_start = std::time::Instant::now();
    let (metadata, index_root) = match &source {
        Source::Local(p) => {
            let (root, meta_path) = if p.is_file()
                && p.file_name().map(|n| n == "index_metadata.json").unwrap_or(false)
            {
                let root = p.parent().unwrap_or(p).to_path_buf();
                (root, p.to_path_buf())
            } else {
                (p.to_path_buf(), p.join("index_metadata.json"))
            };
            if !is_indexed_graph(&root) {
                return Err(anyhow!("expected a pre-built index at {}", root.display()));
            }
            let metadata: DatasetMetadata =
                serde_json::from_slice(&fs::read(&meta_path).context("read index_metadata.json")?)
                    .context("parse index_metadata.json")?;
            (metadata, root)
        }
    };
    let elapsed = load_start.elapsed().as_secs_f64();
    let reader = GraphIndexReader::open(&index_root.join("index_metadata.json"), &index_root)
        .context("open graph reader")?;

    println!("Loaded graph from {}", source_str);
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
    let duration_display = format_duration_seconds(elapsed);
    println!(
        "Load stats: {}, {} bytes, {} nodes, {} edges",
        duration_display,
        format_bytes(bytes_total),
        format_number(nodes),
        format_number(edges),
    );
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
    repl(reader, default_table, index_root)?;
    Ok(())
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

fn repl(reader: Arc<GraphIndexReader>, default_table: String, index_root: PathBuf) -> Result<()> {
    let mut rl = DefaultEditor::new().context("init editor")?;
    let mut current_table = default_table;
    println!("Commands: tables | use <table> | info | node <id> | pkey <value> | formats | ingest | schema | benchmark <threads> [duration] | benchmarkall <threads> [duration] | quit");
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
                if trimmed.eq_ignore_ascii_case("ingest") {
                    print_ingest_stats(&index_root)?;
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("schema") {
                    print_source_schema_only(&index_root)?;
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("node ") {
                    match rest.trim().parse::<u64>() {
                        Ok(id) => show_node(&reader, &current_table, id)?,
                        Err(_) => println!("Invalid node id: {rest}"),
                    }
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("pkey ") {
                    let value = rest.trim();
                    if value.is_empty() {
                        println!("Missing primary key value.");
                    } else {
                        show_node_by_pk(&reader, &current_table, value)?;
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
                    "Unknown command. Use: tables | use <table> | info | node <id> | pkey <value> | formats | ingest | schema | benchmark <threads> [duration] | benchmarkall <threads> [duration] | quit"
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
            let sort_label = format_sort_key_summary(rel);
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
    Ok(())
}

fn print_ingest_stats(root: &Path) -> Result<()> {
    if let Some(report) = read_ingest_report(root)? {
        print_ingest_report(&report);
    } else {
        println!("Ingest stats: ingest_stats.json not found");
    }
    if let Some(report) = read_build_report(root)? {
        print_build_report(&report);
    } else {
        println!("Build stats: build_stats.json not found");
    }
    Ok(())
}

fn print_source_schema_only(root: &Path) -> Result<()> {
    if let Some(report) = read_ingest_report(root)? {
        print_source_schema(&report.source);
    } else {
        println!("Source schema: ingest_stats.json not found");
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
        node_data_trimmed: 0,
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
            agg.node_data_trimmed =
                agg.node_data_trimmed.saturating_add(report.build_stats.node_data_trimmed);
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
    let edge_tables: Vec<_> = source
        .tables
        .iter()
        .filter_map(|table| table.edge_table.as_ref().map(|edge| (table, edge)))
        .collect();
    if !edge_tables.is_empty() {
        println!("  Edge table relationships:");
        for (table, edge) in edge_tables {
            let src_col = table.columns.iter().find(|c| c.name == edge.src_fk.column);
            let dst_col = table.columns.iter().find(|c| c.name == edge.dst_fk.column);
            let src_desc = src_col.map(format_column).unwrap_or_else(|| "unknown".to_string());
            let dst_desc = dst_col.map(format_column).unwrap_or_else(|| "unknown".to_string());
            let src_ref_col = source
                .tables
                .iter()
                .find(|t| t.name == edge.src_fk.ref_table)
                .and_then(|t| t.columns.iter().find(|c| c.name == edge.src_fk.ref_column));
            let dst_ref_col = source
                .tables
                .iter()
                .find(|t| t.name == edge.dst_fk.ref_table)
                .and_then(|t| t.columns.iter().find(|c| c.name == edge.dst_fk.ref_column));
            let src_ref_desc =
                src_ref_col.map(format_column).unwrap_or_else(|| "unknown".to_string());
            let dst_ref_desc =
                dst_ref_col.map(format_column).unwrap_or_else(|| "unknown".to_string());
            println!(
                "    {}: src {} ({}) -> {}.{} ({}); dst {} ({}) -> {}.{} ({})",
                table.name,
                edge.src_fk.column,
                src_desc,
                edge.src_fk.ref_table,
                edge.src_fk.ref_column,
                src_ref_desc,
                edge.dst_fk.column,
                dst_desc,
                edge.dst_fk.ref_table,
                edge.dst_fk.ref_column,
                dst_ref_desc
            );
        }
    }
    if let Some(sort_keys) = &source.edge_sort_keys {
        if !sort_keys.relationships.is_empty() {
            println!("  Relationship sort keys:");
            let mut names: Vec<_> = sort_keys.relationships.keys().collect();
            names.sort();
            for name in names {
                if let Some(cfg) = sort_keys.relationships.get(name) {
                    if let Some(spec) = &cfg.forward {
                        println!(
                            "    {} forward: {}",
                            name,
                            format_sort_key_report(spec)
                        );
                    }
                    if let Some(spec) = &cfg.reverse {
                        println!(
                            "    {} reverse: {}",
                            name,
                            format_sort_key_report(spec)
                        );
                    }
                }
            }
        }
        if !sort_keys.edge_tables.is_empty() {
            println!("  Edge table sort keys:");
            let mut names: Vec<_> = sort_keys.edge_tables.keys().collect();
            names.sort();
            for name in names {
                if let Some(cfg) = sort_keys.edge_tables.get(name) {
                    if let Some(spec) = &cfg.src_to_dst {
                        println!(
                            "    {} src_to_dst: {}",
                            name,
                            format_sort_key_report(spec)
                        );
                    }
                    if let Some(spec) = &cfg.dst_to_src {
                        println!(
                            "    {} dst_to_src: {}",
                            name,
                            format_sort_key_report(spec)
                        );
                    }
                }
            }
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
    let mut out = col.ingested_dtype.as_str().to_string();
    if col.source_dtype != col.ingested_dtype {
        out.push_str(&format!(", source={}", col.source_dtype.as_str()));
    }
    if let Some(mode) = &col.key_mode {
        out.push_str(&format!(", mode={}", mode.as_str()));
    } else {
        out.push_str(", mode=var");
    }
    if let Some(len) = col.fixed_len {
        out.push_str(&format!(", fixed_len={}", len));
    }
    if let Some(fmt) = &col.timestamp_format {
        out.push_str(&format!(", format={}", fmt));
    }
    if let Some(unit) = &col.timestamp_unit {
        out.push_str(&format!(", unit={}", unit.as_str()));
    }
    if let Some(precision) = col.precision {
        out.push_str(&format!(", precision={}", precision));
    }
    if let Some(scale) = col.scale {
        out.push_str(&format!(", scale={}", scale));
    }
    out
}

fn print_build_report(report: &BuildReport) {
    let stats = &report.build_stats;
    println!();
    println!("Build stats:");
    println!("  Nodes ingested: {}", format_count(stats.nodes_ingested));
    println!("  Edges ingested: {}", format_count(stats.edges_ingested));
    if stats.truncated_nodes > 0 || stats.truncated_edges_total > 0 {
        println!("  Truncated nodes: {}", format_count(stats.truncated_nodes));
        println!(
            "  Truncated edges: {}",
            format_count(stats.truncated_edges_total)
        );
    }
    if stats.capped_edge_lists > 0 {
        println!(
            "  Capped edge lists: {}",
            format_count(stats.capped_edge_lists)
        );
    }
    if stats.node_data_trimmed > 0 {
        println!(
            "  Dropped oversized node data: {}",
            format_count(stats.node_data_trimmed)
        );
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
        if let ColumnEncoding::String { encoding } = &col.encoding {
            print_rowpack_string_details(encoding);
        }
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
        StringEncoding::Lz4 => "string(lz4)".to_string(),
    }
}

fn print_rowpack_string_details(encoding: &StringEncoding) {
    match encoding {
        StringEncoding::Dict { values } => {
            let preview = values.iter().take(8);
            for (idx, value) in preview.enumerate() {
                println!("        dict[{}]={}", idx, format_rowpack_bytes(value));
            }
        }
        StringEncoding::Alphabet { alphabet } => {
            println!("        alphabet={}", format_rowpack_bytes(alphabet));
        }
        StringEncoding::Bpe { dict } => {
            let preview = dict.iter().take(8);
            for (idx, value) in preview.enumerate() {
                println!("        bpe[{}]={}", idx, format_rowpack_bytes(value));
            }
        }
        _ => {}
    }
}

fn format_rowpack_bytes(bytes: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(bytes) {
        if text.chars().all(|ch| ch.is_ascii_graphic() || ch == ' ') {
            return format!("{:?}", text);
        }
    }
    let mut out = String::with_capacity(bytes.len() * 2 + 2);
    out.push_str("0x");
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn format_anchor_time(anchor: i64) -> String {
    use chrono::{DateTime, Utc};
    if let Some(dt) = DateTime::<Utc>::from_timestamp(anchor, 0) {
        if anchor.rem_euclid(86_400) == 0 {
            return dt.format("%Y-%m-%d").to_string();
        }
        return dt.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    }
    anchor.to_string()
}

fn format_interval(step: i64) -> String {
    if step == 0 {
        return "0s".to_string();
    }
    let neg = step < 0;
    let mut remaining = step.abs();
    let units = [
        (86_400_i64, "d"),
        (3_600_i64, "h"),
        (60_i64, "m"),
        (1_i64, "s"),
    ];
    let mut parts = Vec::new();
    for (unit, suffix) in units {
        if remaining >= unit {
            let count = remaining / unit;
            remaining %= unit;
            parts.push(format!("{count}{suffix}"));
        }
    }
    if parts.is_empty() {
        parts.push("0s".to_string());
    }
    if neg {
        format!("-{}", parts.join(""))
    } else {
        parts.join("")
    }
}

fn format_sort_key_report(spec: &SortKeyReport) -> String {
    let mut parts = Vec::new();
    if let Some(ts) = &spec.timestamp {
        let kind = match ts.encoding.kind {
            SortKeyTimestampKindReport::Int16 => "int16",
            SortKeyTimestampKindReport::Int32 => "int32",
        };
        let anchor = format_anchor_time(ts.encoding.anchor);
        let step = format_interval(ts.encoding.step);
        parts.push(format!(
            "timestamp(table={}, column={}, kind={}, anchor={}, step={})",
            ts.table, ts.column, kind, anchor, step
        ));
    }
    if let Some(weight) = &spec.weight {
        let kind = match weight.encoding.dtype {
            SortKeyWeightKindReport::Byte => "byte",
            SortKeyWeightKindReport::UInt16 => "uint16",
            SortKeyWeightKindReport::UInt32 => "uint32",
            SortKeyWeightKindReport::Float16 => "float16",
            SortKeyWeightKindReport::Float32 => "float32",
        };
        parts.push(format!(
            "weight(table={}, expr={}, dtype={})",
            weight.table, weight.expr, kind
        ));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(" ")
    }
}

fn format_sort_key_summary(rel: &EdgeRelationMetadata) -> String {
    let meta = match &rel.dest_sort_key {
        Some(meta) => meta,
        None => {
            return rel
                .dest_sort_key_bits
                .map(|bits| format!("{bits}b"))
                .unwrap_or_else(|| "none".to_string())
        }
    };
    let mut parts = Vec::new();
    if let Some(ts) = &meta.timestamp {
        let kind = match ts.encoding.kind {
            EdgeSortKeyTimestampKind::Int8 => "int8",
            EdgeSortKeyTimestampKind::Int16 => "int16",
            EdgeSortKeyTimestampKind::Int32 => "int32",
        };
        parts.push(format!("ts({kind})"));
    }
    if let Some(weight) = &meta.weight {
        let kind = match weight.encoding.dtype {
            EdgeSortKeyWeightKind::Byte => "byte",
            EdgeSortKeyWeightKind::UInt16 => "u16",
            EdgeSortKeyWeightKind::UInt32 => "u32",
            EdgeSortKeyWeightKind::Float16 => "f16",
            EdgeSortKeyWeightKind::Float32 => "f32",
        };
        parts.push(format!("w({kind})"));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        format!("{}b {}", meta.width_bits, parts.join("+"))
    }
}

fn format_sort_key_value(rel: &EdgeRelationMetadata, raw: u64) -> String {
    let meta = match &rel.dest_sort_key {
        Some(meta) => meta,
        None => return format!("{raw}"),
    };
    let weight_bits = meta.weight.as_ref().map(weight_bits_for_meta).unwrap_or(0);
    let weight_mask = if weight_bits > 0 {
        (1u64 << weight_bits) - 1
    } else {
        0
    };
    let weight_raw = if weight_bits > 0 { raw & weight_mask } else { 0 };
    let ts_raw = if weight_bits > 0 { raw >> weight_bits } else { raw };
    let mut parts = Vec::new();
    if let Some(ts) = &meta.timestamp {
        let bits = timestamp_bits_for_meta(ts);
        let delta = decode_ordered_signed(ts_raw, bits);
        if delta == min_signed_for_bits(bits) {
            parts.push("ts=NaT".to_string());
        } else {
            let secs = delta
                .saturating_mul(ts.encoding.step)
                .saturating_add(ts.encoding.anchor);
            parts.push(format!("ts={}", format_epoch_seconds(secs)));
        }
    }
    if let Some(weight) = &meta.weight {
        let value = decode_weight_value(weight, weight_raw);
        parts.push(value);
    }
    if parts.is_empty() {
        format!("{raw}")
    } else {
        parts.join(",")
    }
}

fn format_epoch_seconds(secs: i64) -> String {
    use chrono::{DateTime, Utc};
    if let Some(dt) = DateTime::<Utc>::from_timestamp(secs, 0) {
        if secs.rem_euclid(86_400) == 0 {
            return dt.format("%Y-%m-%d").to_string();
        }
        return dt.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    }
    secs.to_string()
}

fn timestamp_bits_for_meta(meta: &diskgraph_core::metadata::EdgeSortKeyTimestampMetadata) -> u8 {
    match meta.encoding.kind {
        EdgeSortKeyTimestampKind::Int8 => 8,
        EdgeSortKeyTimestampKind::Int16 => 16,
        EdgeSortKeyTimestampKind::Int32 => 32,
    }
}

fn weight_bits_for_meta(meta: &diskgraph_core::metadata::EdgeSortKeyWeightMetadata) -> u8 {
    match meta.encoding.dtype {
        EdgeSortKeyWeightKind::Byte => 8,
        EdgeSortKeyWeightKind::UInt16 => 16,
        EdgeSortKeyWeightKind::UInt32 => 32,
        EdgeSortKeyWeightKind::Float16 => 16,
        EdgeSortKeyWeightKind::Float32 => 32,
    }
}

fn decode_ordered_signed(raw: u64, bits: u8) -> i64 {
    let mask = 1u64 << (bits - 1);
    let decoded = raw ^ mask;
    match bits {
        8 => (decoded as u8 as i8) as i64,
        16 => (decoded as u16 as i16) as i64,
        32 => (decoded as u32 as i32) as i64,
        64 => decoded as i64,
        _ => decoded as i64,
    }
}

fn min_signed_for_bits(bits: u8) -> i64 {
    if bits == 0 || bits >= 63 {
        return i64::MIN;
    }
    -(1i64 << (bits - 1))
}

fn ordered_f32_from_bits(raw: u32) -> f32 {
    let bits = if raw & 0x8000_0000 != 0 {
        raw ^ 0x8000_0000
    } else {
        !raw
    };
    f32::from_bits(bits)
}

fn ordered_f16_from_bits(raw: u16) -> f32 {
    let bits = if raw & 0x8000 != 0 { raw ^ 0x8000 } else { !raw };
    f16::from_bits(bits).to_f32()
}

fn decode_weight_value(
    meta: &diskgraph_core::metadata::EdgeSortKeyWeightMetadata,
    raw: u64,
) -> String {
    match meta.encoding.dtype {
        EdgeSortKeyWeightKind::Byte => format!("weight={raw}"),
        EdgeSortKeyWeightKind::UInt16 => format!("weight={raw}"),
        EdgeSortKeyWeightKind::UInt32 => format!("weight={raw}"),
        EdgeSortKeyWeightKind::Float16 => {
            let val = ordered_f16_from_bits(raw as u16);
            format!("weight={val}")
        }
        EdgeSortKeyWeightKind::Float32 => {
            let val = ordered_f32_from_bits(raw as u32);
            format!("weight={val}")
        }
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
        let sort = format_sort_key_summary(rel);
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
                let sort_display = sort
                    .map(|raw| format_sort_key_value(rel, raw))
                    .unwrap_or_else(|| "none".to_string());
                rows.push((dst, sort_display));
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

fn show_node_by_pk(reader: &GraphIndexReader, table_name: &str, key: &str) -> Result<()> {
    let table = match reader.table(table_name) {
        Some(table) => table,
        None => {
            println!("Unknown table: {table_name}");
            return Ok(());
        }
    };
    if table.primary_key_map().is_none() {
        println!("Table {table_name} has no primary key map");
        return Ok(());
    }
    let key = match table.parse_pk_key(key) {
        Ok(key) => key,
        Err(err) => {
            println!("Invalid primary key: {err}");
            return Ok(());
        }
    };
    let node_id = match table.node_id_from_pk(&key) {
        Ok(Some(node_id)) => node_id,
        Ok(None) => {
            println!("Primary key not found");
            return Ok(());
        }
        Err(err) => {
            println!("Primary key lookup error: {err}");
            return Ok(());
        }
    };
    show_node(reader, table_name, node_id)
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
    Ok(Source::Local(PathBuf::from(raw)))
}
