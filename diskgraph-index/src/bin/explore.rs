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
use diskgraph_rowpack::{
    ColumnEncoding, DecimalStorage, FloatStorage, RowPackContext, RowPackDecodeScratch,
    StringEncoding, ValueRef,
};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{Context as ReadlineContext, Editor, Helper};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write, IsTerminal};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use half::f16;

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

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
    #[serde(default)]
    edge_counts: HashMap<String, u64>,
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

#[derive(Debug, Deserialize, Clone)]
struct TableSpecReport {
    name: String,
    #[serde(default)]
    format: Option<TableFormatReport>,
    #[serde(default)]
    paths: Vec<String>,
    columns: Vec<ColumnSpecReport>,
    #[serde(default)]
    primary_key: Option<String>,
    #[serde(default)]
    edge_table: Option<EdgeTableSpecReport>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
enum TableFormatReport {
    Csv,
    Parquet,
}

#[derive(Debug, Deserialize, Clone)]
struct ColumnSpecReport {
    name: String,
    ingested_dtype: ColumnTypeReport,
    #[serde(default)]
    fixed_len: Option<usize>,
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

#[derive(Debug, Deserialize, PartialEq, Eq, Clone)]
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
    Uuid,
    Uuid256,
    Ip,
    Date,
    Timestamp,
    Decimal128,
    Decimal256,
    Float16Array,
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
            ColumnTypeReport::Uuid => "uuid",
            ColumnTypeReport::Uuid256 => "uuid256",
            ColumnTypeReport::Ip => "ip",
            ColumnTypeReport::Date => "date",
            ColumnTypeReport::Timestamp => "timestamp",
            ColumnTypeReport::Decimal128 => "decimal128",
            ColumnTypeReport::Decimal256 => "decimal256",
            ColumnTypeReport::Float16Array => "float16_array",
            ColumnTypeReport::Float32Array => "float32_array",
            ColumnTypeReport::Float64Array => "float64_array",
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq, Clone)]
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

#[derive(Debug, Deserialize, Clone)]
struct EdgeTableSpecReport {
    src_fk: ForeignKeySpecReport,
    dst_fk: ForeignKeySpecReport,
}

#[derive(Debug, Deserialize, Clone)]
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
        "{} {}, {} {}, {} {}, {} {}",
        label("Load stats:"),
        duration_display,
        label("bytes"),
        ok(&format_bytes(bytes_total)),
        label("nodes"),
        ok(&format_number(nodes)),
        label("edges"),
        ok(&format_number(edges)),
    );
    let reader = Arc::new(reader);
    if metadata.node_types.is_empty() {
        println!("{}", warn("No tables found in metadata."));
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

struct ExploreHelper {
    commands: Vec<String>,
    tables: Vec<String>,
}

impl ExploreHelper {
    fn new(reader: &GraphIndexReader) -> Self {
        let mut tables: Vec<String> = reader
            .tables()
            .iter()
            .map(|t| t.node_type().to_string())
            .collect();
        tables.sort();
        let commands = vec![
            "tables",
            "use",
            "info",
            "node",
            "pkey",
            "formats",
            "ingest",
            "schema",
            "benchmark",
            "benchmarkall",
            "quit",
            "exit",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        Self { commands, tables }
    }
}

impl Completer for ExploreHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &ReadlineContext<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        let input = &line[..pos];
        if let Some(rest) = input.strip_prefix("use ") {
            let start = input.len() - rest.len();
            let prefix = rest;
            let matches = self
                .tables
                .iter()
                .filter(|name| name.starts_with(prefix))
                .map(|name| Pair {
                    display: name.clone(),
                    replacement: name.clone(),
                })
                .collect();
            return Ok((start, matches));
        }

        if !input.contains(' ') {
            let start = input
                .rfind(|ch: char| ch.is_whitespace())
                .map(|idx| idx + 1)
                .unwrap_or(0);
            let prefix = &input[start..];
            let matches = self
                .commands
                .iter()
                .filter(|cmd| cmd.starts_with(prefix))
                .map(|cmd| Pair {
                    display: cmd.clone(),
                    replacement: cmd.clone(),
                })
                .collect();
            return Ok((start, matches));
        }

        Ok((pos, Vec::new()))
    }
}

impl Hinter for ExploreHelper {
    type Hint = String;
}

impl Highlighter for ExploreHelper {}
impl Validator for ExploreHelper {}
impl Helper for ExploreHelper {}

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

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const BLUE: &str = "\x1b[34m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";

fn colorize(value: &str, color: &str) -> String {
    format!("{color}{value}{RESET}")
}

fn dim(value: &str) -> String {
    format!("{DIM}{value}{RESET}")
}

fn ok(value: &str) -> String {
    colorize(value, GREEN)
}

fn warn(value: &str) -> String {
    colorize(value, YELLOW)
}

fn err(value: &str) -> String {
    colorize(value, RED)
}

fn label(value: &str) -> String {
    colorize(value, CYAN)
}

fn name(value: &str) -> String {
    colorize(value, BLUE)
}

fn highlight(value: &str) -> String {
    colorize(value, MAGENTA)
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
    let mut rl = Editor::<ExploreHelper, DefaultHistory>::new().context("init editor")?;
    rl.set_helper(Some(ExploreHelper::new(&reader)));
    let mut current_table = default_table;
    println!(
        "{} {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
        label("Commands:"),
        name("tables"),
        name("use <table>"),
        name("info"),
        name("node <id>"),
        name("pkey <value>"),
        name("formats"),
        name("ingest"),
        name("schema"),
        name("benchmark <threads> [duration]"),
        name("benchmarkall <threads> [duration]"),
        name("quit")
    );
    loop {
        let prompt = format!("{BOLD}{CYAN}explore>{RESET} ");
        match rl.readline(&prompt) {
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
                    let table_name = rest.trim();
                    if reader.table(table_name).is_some() {
                        current_table = table_name.to_string();
                        println!("{} {}", label("Using table"), name(&current_table));
                    } else {
                        println!("{} {}", err("Unknown table:"), name(table_name));
                    }
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("info") {
                    print_info(&reader, &current_table, &index_root);
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
                        Err(_) => println!("{} {rest}", err("Invalid node id:")),
                    }
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("pkey ") {
                    let value = rest.trim();
                    if value.is_empty() {
                        println!("{}", warn("Missing primary key value."));
                    } else {
                        show_node_by_pk(&reader, &current_table, value)?;
                    }
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("benchmark ") {
                    let mut parts = rest.split_whitespace();
                    let Some(thread_part) = parts.next() else {
                        println!("{} {rest}", err("Invalid benchmark command:"));
                        continue;
                    };
                    let duration_part = parts.next();
                    let extra = parts.next();
                    if extra.is_some() {
                        println!("{} {rest}", err("Invalid benchmark command:"));
                        continue;
                    }
                    let threads = match thread_part.parse::<usize>() {
                        Ok(threads) if threads > 0 => threads,
                        _ => {
                            println!("{} {thread_part}", err("Invalid thread count:"));
                            continue;
                        }
                    };
                    let duration = if let Some(raw) = duration_part {
                        match parse_duration(raw) {
                            Some(d) => d,
                            None => {
                                println!(
                                    "{} {raw} {}",
                                    err("Invalid duration:"),
                                    dim("(expected 30s or 5m)")
                                );
                                continue;
                            }
                        }
                    } else {
                        std::time::Duration::from_secs(20)
                    };
                    let table = match reader.table(&current_table) {
                        Some(table) => table,
                        None => {
                            println!("{} {}", err("Unknown table:"), name(&current_table));
                            continue;
                        }
                    };
                    run_benchmark(table, threads, duration)?;
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("benchmarkall ") {
                    let mut parts = rest.split_whitespace();
                    let Some(thread_part) = parts.next() else {
                        println!("{} {rest}", err("Invalid benchmarkall command:"));
                        continue;
                    };
                    let duration_part = parts.next();
                    let extra = parts.next();
                    if extra.is_some() {
                        println!("{} {rest}", err("Invalid benchmarkall command:"));
                        continue;
                    }
                    let threads = match thread_part.parse::<usize>() {
                        Ok(threads) if threads > 0 => threads,
                        _ => {
                            println!("{} {thread_part}", err("Invalid thread count:"));
                            continue;
                        }
                    };
                    let duration = if let Some(raw) = duration_part {
                        match parse_duration(raw) {
                            Some(d) => d,
                            None => {
                                println!(
                                    "{} {raw} {}",
                                    err("Invalid duration:"),
                                    dim("(expected 30s or 5m)")
                                );
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
                    "{} {}",
                    err("Unknown command."),
                    dim("Use: tables | use <table> | info | node <id> | pkey <value> | formats | ingest | schema | benchmark <threads> [duration] | benchmarkall <threads> [duration] | quit")
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
        "{} {} {} {} {} {}",
        label("Node types:"),
        ok(&metadata.node_types.len().to_string()),
        label("Relations:"),
        ok(&metadata.edge_relations.len().to_string()),
        label("Current table:"),
        name(current_table)
    );
    print_tables(reader);
}

fn print_tables(reader: &GraphIndexReader) {
    println!("{}", label("Tables:"));
    for table in reader.tables() {
        println!(
            "  {} ({}={}, {}={})",
            highlight(table.node_type()),
            label("shards"),
            ok(&table.shard_count().to_string()),
            label("nodes"),
            ok(&format_number(table.node_count()))
        );
        for rel in table.relations() {
            let sort_label = format_sort_key_summary(rel);
            println!(
                "    {} {} ({}={}, {}={})",
                dim("->"),
                name(&rel.name),
                label("dst"),
                highlight(&rel.dst_type),
                label("sort_key"),
                if sort_label == "none" {
                    dim(&sort_label)
                } else {
                    highlight(&sort_label)
                }
            );
        }
    }
}

fn print_formats(metadata: &DatasetMetadata, root: &Path) -> Result<()> {
    println!("{}", label("Formats:"));
    for node in &metadata.node_types {
        println!();
        println!("{} {}", label("Table:"), name(&node.name));
        if let Some(pk) = &node.primary_key_map {
            let fixed = pk
                .fixed_len
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "  {} {} ({}={}, {}={}, {}={})",
                label("Primary key:"),
                name(&pk.column),
                label("dtype"),
                highlight(&pk.dtype),
                label("fixed_len"),
                if fixed == "n/a" { dim(&fixed) } else { ok(&fixed) },
                label("mapping"),
                highlight(&pk.mapping_kind)
            );
        } else {
            println!("  {} {}", label("Primary key:"), dim("none"));
        }
        if let Some(compression) = &node.node_data_compression {
            println!(
                "  {} {} ({}={})",
                label("Node data:"),
                highlight(&compression.codec),
                label("ctx"),
                dim(&compression.context_path)
            );
            if compression.codec == "rowpack" {
                let ctx_path = root.join(&compression.context_path);
                if let Ok(bytes) = fs::read(&ctx_path) {
                    if let Ok(ctx) = RowPackContext::from_bytes(&bytes) {
                        print_rowpack_summary(&ctx);
                    } else {
                        println!(
                            "    {} {}",
                            err("Rowpack: failed to parse"),
                            dim(&ctx_path.display().to_string())
                        );
                    }
                } else {
                    println!(
                        "    {} {}",
                        warn("Rowpack: missing"),
                        dim(&ctx_path.display().to_string())
                    );
                }
            }
        } else {
            println!("  {} {}", label("Node data:"), dim("none"));
        }
    }
    Ok(())
}

fn print_ingest_stats(root: &Path) -> Result<()> {
    if let Some(report) = read_ingest_report(root)? {
        print_ingest_report(&report, root);
    } else {
        println!(
            "{} {}",
            warn("Ingest stats missing:"),
            dim("ingest_stats.json not found")
        );
    }
    if let Some(report) = read_build_report(root)? {
        print_build_report(&report);
    } else {
        println!(
            "{} {}",
            warn("Build stats missing:"),
            dim("build_stats.json not found")
        );
    }
    Ok(())
}

fn print_source_schema_only(root: &Path) -> Result<()> {
    if let Some(report) = read_ingest_report(root)? {
        print_source_schema(&report.source);
    } else {
        println!(
            "{} {}",
            warn("Source schema missing:"),
            dim("ingest_stats.json not found")
        );
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

fn print_edge_stats(stats: &IngestStatsReport, relations: &[&EdgeRelationMetadata]) {
    for rel in relations {
        let success = stats.edge_counts.get(&rel.name).copied().unwrap_or(0);
        let ctx_src = format!("src_lookup:{}", rel.name);
        let ctx_dst = format!("dst_lookup:{}", rel.name);
        let src_counts = stats.key_failures_by_context.get(&ctx_src);
        let dst_counts = stats.key_failures_by_context.get(&ctx_dst);
        let parse = src_counts
            .map(|c| c.parse_failures)
            .unwrap_or(0)
            .saturating_add(dst_counts.map(|c| c.parse_failures).unwrap_or(0));
        let lookup = src_counts
            .map(|c| c.lookup_failures)
            .unwrap_or(0)
            .saturating_add(dst_counts.map(|c| c.lookup_failures).unwrap_or(0));
        let rate = if success == 0 && lookup == 0 {
            dim("n/a")
        } else {
            let denom = success.saturating_add(lookup);
            let pct = (success as f64 * 100.0) / denom as f64;
            if pct >= 99.0 {
                ok(&format!("{pct:.2}%"))
            } else if pct >= 90.0 {
                warn(&format!("{pct:.2}%"))
            } else {
                err(&format!("{pct:.2}%"))
            }
        };
        println!(
            "    {}: {} {} {} {} {} {} {} {}",
            name(&rel.name),
            label("success"),
            ok(&format_number(success)),
            label("lookup"),
            if lookup == 0 {
                ok(&format_number(lookup))
            } else {
                warn(&format_number(lookup))
            },
            label("parse"),
            if parse == 0 {
                ok(&format_number(parse))
            } else {
                warn(&format_number(parse))
            },
            label("match"),
            rate
        );
    }
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

fn sum_path_bytes(path: &Path) -> io::Result<u64> {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let meta = fs::metadata(&current)?;
        if meta.is_file() {
            total = total.saturating_add(meta.len());
            continue;
        }
        if meta.is_dir() {
            for entry in fs::read_dir(&current)? {
                let entry = entry?;
                stack.push(entry.path());
            }
        }
    }
    Ok(total)
}

fn sum_table_paths(root: &Path, paths: &[String]) -> Option<u64> {
    if paths.is_empty() {
        return None;
    }
    let mut total = 0u64;
    for path in paths {
        let raw = Path::new(path);
        let resolved = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            root.join(raw)
        };
        let bytes = sum_path_bytes(&resolved).ok()?;
        total = total.saturating_add(bytes);
    }
    Some(total)
}

fn print_ingest_report(report: &IngestReport, root: &Path) {
    println!();
    println!("{}", label("Ingest stats:"));
    if !report.source.tables.is_empty() {
        let mut table_sizes = Vec::new();
        let mut total = 0u64;
        let mut has_total = true;
        for table in &report.source.tables {
            let size = sum_table_paths(root, &table.paths);
            if let Some(bytes) = size {
                total = total.saturating_add(bytes);
            } else {
                has_total = false;
            }
            table_sizes.push((table.name.as_str(), size, table.format.as_ref()));
        }
        println!(
            "  {} {}",
            label("Data size:"),
            if has_total {
                ok(&format_bytes(total))
            } else {
                warn("partial")
            }
        );
        for (table_name, size, format) in table_sizes {
            let format_label = match format {
                Some(TableFormatReport::Csv) => "csv",
                Some(TableFormatReport::Parquet) => "parquet",
                None => "unknown",
            };
            println!(
                "    {} {} {} {}",
                name(table_name),
                dim(format_label),
                label("size"),
                match size {
                    Some(bytes) => ok(&format_bytes(bytes)),
                    None => warn("n/a"),
                }
            );
        }
    }
    println!(
        "  {} {}={} {}={}",
        label("Key failures:"),
        label("parse"),
        err(&format_number(report.stats.key_parse_failures)),
        label("lookup"),
        err(&format_number(report.stats.key_lookup_failures))
    );
    if !report.stats.key_failures_by_context.is_empty() {
        let mut entries: Vec<_> = report.stats.key_failures_by_context.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        println!("  {}", label("Key failures by context:"));
        for (ctx, counts) in entries {
            if counts.parse_failures == 0 && counts.lookup_failures == 0 {
                continue;
            }
            println!(
                "    {}: {} {} {} {}",
                name(ctx),
                label("parse"),
                err(&format_number(counts.parse_failures)),
                label("lookup"),
                err(&format_number(counts.lookup_failures))
            );
        }
    }
    if !report.stats.key_failure_samples.is_empty() {
        println!("  {}", label("Key failure samples:"));
        for sample in &report.stats.key_failure_samples {
            println!("    {}", dim(sample));
        }
    }
    if report.stats.pk_duplicate_total > 0 {
        println!(
            "  {} {}",
            label("Duplicate primary keys:"),
            warn(&format_number(report.stats.pk_duplicate_total))
        );
        let mut entries: Vec<_> = report.stats.pk_duplicate_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (table, count) in entries {
            println!(
                "    {}: {}",
                name(table),
                warn(&format_number(*count))
            );
            if let Some(samples) = report.stats.pk_duplicate_samples.get(table) {
                if !samples.is_empty() {
                    println!("      {} {}", label("samples:"), dim(&samples.join(", ")));
                }
            }
        }
    }
    print_source_schema(&report.source);
}

fn print_source_schema(source: &SourceMetadataReport) {
    println!();
    println!("{} {}", label("Source schema:"), name(&source.dataset_name));
    let mut tables: Vec<&TableSpecReport> = source.tables.iter().collect();
    tables.sort_by(|a, b| a.name.cmp(&b.name));
    let mut table_map: HashMap<&str, &TableSpecReport> = HashMap::new();
    for table in &source.tables {
        table_map.insert(table.name.as_str(), table);
    }
    for table in tables {
        println!("  {} {}:", label("Table"), name(&table.name));
        if let Some(pk) = &table.primary_key {
            if let Some(col) = table.columns.iter().find(|c| &c.name == pk) {
                println!(
                    "    {} {} ({})",
                    label("Primary key:"),
                    name(pk),
                    highlight(&format_column(col))
                );
            } else {
                println!("    {} {}", label("Primary key:"), name(pk));
            }
        } else {
            println!("    {} {}", label("Primary key:"), dim("none"));
        }
        if let Some(edge) = &table.edge_table {
            println!(
                "    {} {} {} {}",
                label("Edge table:"),
                highlight(&format_fk(&table, &edge.src_fk)),
                dim("->"),
                highlight(&format_fk(&table, &edge.dst_fk))
            );
        }
    }
    if !source.relationships.is_empty() {
        println!("  {}", label("Relationships:"));
        for rel in &source.relationships {
            let rel_name = rel
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
                "    {}: {}.{} ({}) {} {}.{} ({})",
                highlight(rel_name),
                name(&rel.src_table),
                name(&rel.src_column),
                highlight(&src_desc),
                dim("->"),
                name(&rel.dst_table),
                name(&rel.dst_column),
                highlight(&dst_desc)
            );
        }
    }
    let edge_tables: Vec<_> = source
        .tables
        .iter()
        .filter_map(|table| table.edge_table.as_ref().map(|edge| (table, edge)))
        .collect();
    if !edge_tables.is_empty() {
        println!("  {}", label("Edge table relationships:"));
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
                "    {}: {} {} ({}) {} {}.{} ({}); {} {} ({}) {} {}.{} ({})",
                name(&table.name),
                label("src"),
                name(&edge.src_fk.column),
                highlight(&src_desc),
                dim("->"),
                name(&edge.src_fk.ref_table),
                name(&edge.src_fk.ref_column),
                highlight(&src_ref_desc),
                label("dst"),
                name(&edge.dst_fk.column),
                highlight(&dst_desc),
                dim("->"),
                name(&edge.dst_fk.ref_table),
                name(&edge.dst_fk.ref_column),
                highlight(&dst_ref_desc)
            );
        }
    }
    if let Some(sort_keys) = &source.edge_sort_keys {
        if !sort_keys.relationships.is_empty() {
            println!("  {}", label("Relationship sort keys:"));
            let mut names: Vec<_> = sort_keys.relationships.keys().collect();
            names.sort();
            for rel_key in names {
                if let Some(cfg) = sort_keys.relationships.get(rel_key) {
                    if let Some(spec) = &cfg.forward {
                        println!(
                            "    {} {} {}",
                            name(rel_key),
                            label("forward:"),
                            highlight(&format_sort_key_report(spec))
                        );
                    }
                    if let Some(spec) = &cfg.reverse {
                        println!(
                            "    {} {} {}",
                            name(rel_key),
                            label("reverse:"),
                            highlight(&format_sort_key_report(spec))
                        );
                    }
                }
            }
        }
        if !sort_keys.edge_tables.is_empty() {
            println!("  {}", label("Edge table sort keys:"));
            let mut names: Vec<_> = sort_keys.edge_tables.keys().collect();
            names.sort();
            for rel_key in names {
                if let Some(cfg) = sort_keys.edge_tables.get(rel_key) {
                    if let Some(spec) = &cfg.src_to_dst {
                        println!(
                            "    {} {} {}",
                            name(rel_key),
                            label("src_to_dst:"),
                            highlight(&format_sort_key_report(spec))
                        );
                    }
                    if let Some(spec) = &cfg.dst_to_src {
                        println!(
                            "    {} {} {}",
                            name(rel_key),
                            label("dst_to_src:"),
                            highlight(&format_sort_key_report(spec))
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
    println!("{}", label("Build stats:"));
    println!(
        "  {} {}",
        label("Nodes ingested:"),
        ok(&format_count(stats.nodes_ingested))
    );
    println!(
        "  {} {}",
        label("Edges ingested:"),
        ok(&format_count(stats.edges_ingested))
    );
    if stats.truncated_nodes > 0 || stats.truncated_edges_total > 0 {
        println!(
            "  {} {}",
            label("Truncated nodes:"),
            warn(&format_count(stats.truncated_nodes))
        );
        println!(
            "  {} {}",
            label("Truncated edges:"),
            warn(&format_count(stats.truncated_edges_total))
        );
    }
    if stats.capped_edge_lists > 0 {
        println!(
            "  {} {}",
            label("Capped edge lists:"),
            warn(&format_count(stats.capped_edge_lists))
        );
    }
    if stats.node_data_trimmed > 0 {
        println!(
            "  {} {}",
            label("Dropped oversized node data:"),
            warn(&format_count(stats.node_data_trimmed))
        );
    }
    if !stats.truncated_edges_per_relation.is_empty() {
        let total: u64 = stats.truncated_edges_per_relation.iter().sum();
        if total > 0 {
            println!(
                "  {} {}",
                label("Truncated edges by relation:"),
                warn(&format_count(total))
            );
        }
    }
}

fn print_rowpack_summary(ctx: &RowPackContext) {
    println!(
        "    {} {} {}={}",
        label("Rowpack:"),
        ok(&format!("v{}", ctx.version)),
        label("estimated_row_bytes"),
        ok(&ctx.estimated_row_bytes.to_string())
    );
    println!("    {}", label("Columns:"));
    for (idx, col) in ctx.columns.iter().enumerate() {
        let encoding = format_rowpack_encoding(&col.encoding);
        println!(
            "      [{}] {} ({}) {}",
            idx,
            name(&col.name),
            highlight(&format!("{:?}", col.logical_type)),
            highlight(&encoding)
        );
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
        ColumnEncoding::Uuid => "uuid".to_string(),
        ColumnEncoding::Uuid256 => "uuid256".to_string(),
        ColumnEncoding::Ip => "ip".to_string(),
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
                println!(
                    "        {}[{}]={}",
                    label("dict"),
                    ok(&idx.to_string()),
                    dim(&format_rowpack_bytes(value))
                );
            }
        }
        StringEncoding::Alphabet { alphabet } => {
            println!(
                "        {}={}",
                label("alphabet"),
                dim(&format_rowpack_bytes(alphabet))
            );
        }
        StringEncoding::Bpe { dict } => {
            let preview = dict.iter().take(8);
            for (idx, value) in preview.enumerate() {
                println!(
                    "        {}[{}]={}",
                    label("bpe"),
                    ok(&idx.to_string()),
                    dim(&format_rowpack_bytes(value))
                );
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
        DecimalStorage::Varint => "varint".to_string(),
    }
}

fn print_info(reader: &GraphIndexReader, table_name: &str, root: &Path) {
    let table = match reader.table(table_name) {
        Some(table) => table,
        None => {
            println!("{} {}", err("Unknown table:"), name(table_name));
            return;
        }
    };
    let node_meta = reader
        .metadata()
        .node_types
        .iter()
        .find(|n| n.name == table.node_type());
    let shard_count = table.shard_count();
    let rel_count = table.relations().len();
    println!("{} {}", label("Table:"), name(table.node_type()));
    println!(
        "  {} {} ({}={})",
        label("Nodes:"),
        ok(&format_number(table.node_count())),
        label("shards"),
        ok(&shard_count.to_string())
    );
    println!("  {} {}", label("Relations:"), ok(&rel_count.to_string()));
    println!(
        "  {} {}",
        label("Node data:"),
        node_meta
            .map(|n| if n.has_node_data { ok("enabled") } else { dim("disabled") })
            .unwrap_or_else(|| dim("unknown"))
    );

    let mut total_edges = 0u64;
    let mut total_offsets = 0u64;
    let mut total_edge_bytes = 0u64;
    if let Some(node_meta) = node_meta {
        println!("  {}", label("Shards:"));
        for (idx, shard) in node_meta.shards.iter().enumerate() {
            let node_count = if shard.max_id >= shard.min_id {
                shard.max_id - shard.min_id + 1
            } else {
                0
            };
            if let Some(reader_shard) = table.shard(idx) {
                let header = reader_shard.header();
                let node_data_paths = shard
                    .node_data_paths
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0);
                total_edges = total_edges.saturating_add(reader_shard.edge_count());
                total_offsets = total_offsets.saturating_add(reader_shard.offsets_len());
                total_edge_bytes = total_edge_bytes.saturating_add(reader_shard.edges_len());
                let offsets = match header.delta_bytes {
                    2 => "16bit",
                    3 => "24bit",
                    4 => "32bit",
                    _ => "unknown",
                };
                println!(
                    "    {} {}: {} {}..{} ({}={}) {}={} {}={} {}={} {}={} {}={}",
                    dim("-"),
                    label(&format!("shard {:05}", idx)),
                    label("ids"),
                    name(&shard.min_id.to_string()),
                    name(&shard.max_id.to_string()),
                    label("nodes"),
                    ok(&format_number(node_count)),
                    label("offsets"),
                    ok(&format_bytes(reader_shard.offsets_len())),
                    label("data"),
                    ok(&format_bytes(reader_shard.edges_len())),
                    label("offsets"),
                    ok(offsets),
                    label("edge_count"),
                    ok(&format_number(reader_shard.edge_count())),
                    label("node_data_parts"),
                    ok(&node_data_paths.to_string())
                );
            } else {
                println!(
                    "    {} {}: {} {}..{} ({}={})",
                    dim("-"),
                    label(&format!("shard {:05}", idx)),
                    label("ids"),
                    name(&shard.min_id.to_string()),
                    name(&shard.max_id.to_string()),
                    label("nodes"),
                    ok(&format_number(node_count))
                );
            }
        }
    }
    println!(
        "  {} {}={} {}={} {}={}",
        label("Storage:"),
        label("data"),
        ok(&format_bytes(total_edge_bytes)),
        label("offsets"),
        ok(&format_bytes(total_offsets)),
        label("total_edges"),
        ok(&format_number(total_edges))
    );

    println!("  {}", label("Relation layouts:"));
    let layouts = table.layouts();
    for (idx, rel) in table.relations().iter().enumerate() {
        let sort = format_sort_key_summary(rel);
        let layout = layouts.get(idx);
        let (dst_bits, entry_bytes, sort_bits) = layout
            .map(|l| (l.dst_bits, l.entry_bytes, l.dest_sort_key_bits))
            .unwrap_or((0, 0, None));
        println!(
            "    {} [{}] {} ({} {} {}), {} {}={} {}={} {}={} {}={}",
            dim("-"),
            idx,
            name(&rel.name),
            label("src"),
            name(&rel.src_type),
            dim("->"),
            name(&rel.dst_type),
            label("sort_key"),
            if sort == "none" { dim(&sort) } else { highlight(&sort) },
            label("dst_bits"),
            ok(&dst_bits.to_string()),
            label("entry_bytes"),
            ok(&entry_bytes.to_string()),
            label("sort_bits"),
            sort_bits
                .map(|v| ok(&v.to_string()))
                .unwrap_or_else(|| dim("none"))
        );
    }

    let mut source_schema: Option<TableSpecReport> = None;
    if let Ok(Some(report)) = read_ingest_report(root) {
        source_schema = report
            .source
            .tables
            .iter()
            .find(|t| t.name == table.node_type())
            .map(|t| TableSpecReport {
                name: t.name.clone(),
                format: t.format.clone(),
                paths: t.paths.clone(),
                columns: t.columns.clone(),
                primary_key: t.primary_key.clone(),
                edge_table: t.edge_table.clone(),
            });
        let incoming: Vec<_> = reader
            .metadata()
            .edge_relations
            .iter()
            .filter(|rel| rel.dst_type == table.node_type())
            .collect();
        let outgoing: Vec<_> = reader
            .metadata()
            .edge_relations
            .iter()
            .filter(|rel| rel.src_type == table.node_type())
            .collect();
        if !incoming.is_empty() {
            println!("  {}", label("Incoming edge stats:"));
            print_edge_stats(&report.stats, &incoming);
        }
        if !outgoing.is_empty() {
            println!("  {}", label("Outgoing edge stats:"));
            print_edge_stats(&report.stats, &outgoing);
        }

    }

    if let Ok(Some(report)) = read_build_report(root) {
        println!(
            "  {} {}={} {}={} {}={} {}={} {}={} {}={}",
            label("Build stats (dataset-wide):"),
            label("nodes_ingested"),
            ok(&format_number(report.build_stats.nodes_ingested)),
            label("edges_ingested"),
            ok(&format_number(report.build_stats.edges_ingested)),
            label("truncated_nodes"),
            warn(&format_number(report.build_stats.truncated_nodes)),
            label("truncated_edges"),
            warn(&format_number(report.build_stats.truncated_edges_total)),
            label("capped_lists"),
            warn(&format_number(report.build_stats.capped_edge_lists)),
            label("node_data_trimmed"),
            warn(&format_number(report.build_stats.node_data_trimmed))
        );
        if !report.build_stats.truncated_edges_per_relation.is_empty() {
            println!("  {}", label("Truncated edges per relation:"));
            for (idx, count) in report
                .build_stats
                .truncated_edges_per_relation
                .iter()
                .enumerate()
            {
                let rel_name = table
                    .relations()
                    .get(idx)
                    .map(|rel| rel.name.as_str())
                    .unwrap_or("unknown");
                println!(
                    "    {} [{}] {}: {}",
                    dim("-"),
                    idx,
                    name(rel_name),
                    warn(&format_number(*count))
                );
            }
        }
    }

    if let Some(table_spec) = source_schema.as_ref() {
        println!("  {}", label("Source schema:"));
        if let Some(pk) = &table_spec.primary_key {
            println!("    {} {}", label("Primary key:"), name(pk));
        } else {
            println!("    {} {}", label("Primary key:"), dim("none"));
        }
        if let Some(edge) = &table_spec.edge_table {
            println!(
                "    {} {} {} {}",
                label("Edge table:"),
                highlight(&format_fk(table_spec, &edge.src_fk)),
                dim("->"),
                highlight(&format_fk(table_spec, &edge.dst_fk))
            );
        }
        println!("    {}", label("Columns:"));
        for col in &table_spec.columns {
            println!(
                "      {} {}: {}",
                dim("-"),
                name(&col.name),
                highlight(&format_column(col))
            );
        }
    }
}

fn show_node(reader: &GraphIndexReader, table_name: &str, node: u64) -> Result<()> {
    let start = std::time::Instant::now();
    let table = match reader.table(table_name) {
        Some(table) => table,
        None => {
            println!("{} {}", err("Unknown table:"), name(table_name));
            return Ok(());
        }
    };
    let view = match table.edges_for_node(node) {
        Ok(v) => v,
        Err(e) => {
            println!(
                "{} {}: {}",
                err("Error reading node"),
                name(&node.to_string()),
                err(&e.to_string())
            );
            return Ok(());
        }
    };
    let segments = match view.relation_segments() {
        Ok(s) => s,
        Err(e) => {
            println!("{} {}", err("Error decoding relations:"), err(&e.to_string()));
            return Ok(());
        }
    };
    println!("{} {}", label("Node:"), name(&node.to_string()));
    for (idx, seg) in segments.iter().enumerate() {
        let rel = &table.relations()[idx];
        println!(
            "  [{}] {} {}={}",
            idx,
            name(&rel.name),
            label("count"),
            ok(&seg.count.to_string())
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
            println!(
                "    {} {} {}",
                label("first"),
                ok(&rows.len().to_string()),
                dim(&format!("edges: {:?}", rows))
            );
        }
    }
    match view.node_data() {
        Ok(_) => match table.decode_node_data_json(&view) {
            Ok(Some(value)) => match serde_json::to_string_pretty(&value) {
                Ok(pretty) => println!("  {}{}\n{pretty}", label("node_data"), dim(":")),
                Err(_) => {
                    println!("  {} {}", label("node_data:"), err("<error: json encode>"));
                }
            },
            Ok(None) => match table.decode_node_data(&view) {
                Ok(Some(bytes)) => {
                    let escaped = escape_ascii(&bytes);
                    println!("  {} \"{}\"", label("node_data:"), dim(&escaped));
                }
                Ok(None) => {}
                Err(e) => {
                    println!("  {} {}", label("node_data:"), err(&format!("<error: {e}>")));
                }
            },
            Err(e) => {
                println!("  {} {}", label("node_data:"), err(&format!("<error: {e}>")));
            }
        },
        Err(e) => {
            println!("  {} {}", label("node_data:"), err(&format!("<error: {e}>")));
        }
    }
    let elapsed = start.elapsed();
    println!(
        "  {} {}",
        label("lookup:"),
        ok(&format!("{:.3} ms", elapsed.as_secs_f64() * 1000.0))
    );
    Ok(())
}

fn show_node_by_pk(reader: &GraphIndexReader, table_name: &str, key: &str) -> Result<()> {
    let table = match reader.table(table_name) {
        Some(table) => table,
        None => {
            println!("{} {}", err("Unknown table:"), name(table_name));
            return Ok(());
        }
    };
    if table.primary_key_map().is_none() {
        println!(
            "{} {}",
            warn("Table has no primary key map:"),
            name(table_name)
        );
        return Ok(());
    }
    let key = match table.parse_pk_key(key) {
        Ok(key) => key,
        Err(parse_err) => {
            println!(
                "{} {}",
                err("Invalid primary key:"),
                err(&parse_err.to_string())
            );
            return Ok(());
        }
    };
    println!(
        "{} {:?} {}",
        dim("pkey parsed:"),
        key.key_type(),
        dim(&format_pk_key_debug(&key))
    );
    let node_id = match table.node_id_from_pk(&key) {
        Ok(Some(node_id)) => node_id,
        Ok(None) => {
            if let Some(pk_map) = table.pk_map() {
                let shard_idx = diskgraph_index::pkmap::shard_for_key(&key, pk_map.shard_count());
                println!(
                    "{} {}",
                    dim("pkey shard:"),
                    dim(&format!("{shard_idx}"))
                );
                if let Some((found_shard, found_local)) =
                    debug_scan_pk_all_shards(table.as_ref(), &key)
                {
                    println!(
                        "{} shard={} local_id={}",
                        warn("pkey found in alternate shard:"),
                        found_shard,
                        found_local
                    );
                }
            }
            println!("{}", warn("Primary key not found."));
            return Ok(());
        }
        Err(lookup_err) => {
            println!(
                "{} {}",
                err("Primary key lookup error:"),
                err(&lookup_err.to_string())
            );
            return Ok(());
        }
    };
    if !pk_matches_node(&table, node_id, &key)? {
        println!("{}", warn("Primary key not found."));
        return Ok(());
    }
    show_node(reader, table_name, node_id)
}

fn debug_scan_pk_all_shards(
    table: &diskgraph_index::graph::TableReader,
    key: &diskgraph_index::pkmap::Key,
) -> Option<(usize, u32)> {
    let pk_map = table.pk_map()?;
    for shard_idx in 0..pk_map.shard_count() {
        let shard = pk_map.shard(shard_idx)?;
        if let Some(local) = shard.lookup(key) {
            return Some((shard_idx, local));
        }
    }
    None
}

fn format_pk_key_debug(key: &diskgraph_index::pkmap::Key) -> String {
    match key {
        diskgraph_index::pkmap::Key::Int(v) => format!("int={v}"),
        diskgraph_index::pkmap::Key::Uuid(v) => format!("uuid={}", hex_bytes(v)),
        diskgraph_index::pkmap::Key::Uuid256(v) => format!("uuid256={}", hex_bytes(v)),
        diskgraph_index::pkmap::Key::Ip(v) => format!("ip={}", hex_bytes(v)),
        diskgraph_index::pkmap::Key::Bytes(v) => {
            let ascii = escape_ascii(v);
            format!("bytes_len={} ascii=\"{}\" hex={}", v.len(), ascii, hex_bytes(v))
        }
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX_CHARS[(b >> 4) as usize] as char);
        out.push(HEX_CHARS[(b & 0x0f) as usize] as char);
    }
    out
}

fn pk_matches_node(table: &diskgraph_index::graph::TableReader, node_id: u64, key: &diskgraph_index::pkmap::Key) -> Result<bool> {
    let pk = match table.primary_key_map() {
        Some(pk) => pk,
        None => return Ok(true),
    };
    let view = table
        .edges_for_node(node_id)
        .map_err(|e| anyhow!("node lookup failed: {e}"))?;
    let raw = match view.node_data()? {
        Some(raw) => raw,
        None => return Ok(true),
    };
    let compression = table
        .node_data_compression()
        .ok_or_else(|| anyhow!("node data compression missing"))?;
    if compression.codec != "rowpack" {
        return Ok(true);
    }
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("rowpack ctx missing for {}", table.node_type()))?;
    let ctx = decoder.context();
    let col_idx = match pk.pk_column_index.and_then(|idx| {
        ctx.columns
            .get(idx)
            .filter(|col| col.name == pk.column)
            .map(|_| idx)
    }) {
        Some(idx) => idx,
        None => match ctx.columns.iter().position(|col| col.name == pk.column) {
            Some(idx) => idx,
            None => return Ok(true),
        },
    };
    let mut scratch = RowPackDecodeScratch::default();
    let value = match decoder.decode_column_value(raw, col_idx, &mut scratch) {
        Ok(value) => value,
        Err(_) => return Ok(true),
    };
    Ok(pk_value_matches(key, &value))
}

fn pk_value_matches(key: &diskgraph_index::pkmap::Key, value: &ValueRef<'_>) -> bool {
    match (key, value) {
        (diskgraph_index::pkmap::Key::Int(k), ValueRef::Int64(v)) => *v >= 0 && *k == *v as u64,
        (diskgraph_index::pkmap::Key::Int(k), ValueRef::Int32(v)) => *v >= 0 && *k == *v as u64,
        (diskgraph_index::pkmap::Key::Int(k), ValueRef::UInt64(v)) => *k == *v,
        (diskgraph_index::pkmap::Key::Int(k), ValueRef::UInt32(v)) => *k == *v as u64,
        (diskgraph_index::pkmap::Key::Bytes(k), ValueRef::Bytes(v)) => k.as_slice() == *v,
        (diskgraph_index::pkmap::Key::Bytes(k), ValueRef::String(v)) => k.as_slice() == v.as_bytes(),
        (diskgraph_index::pkmap::Key::Uuid(k), ValueRef::Uuid(v)) => k == v,
        (diskgraph_index::pkmap::Key::Uuid256(k), ValueRef::Uuid256(v)) => k == v,
        (diskgraph_index::pkmap::Key::Ip(k), ValueRef::Ip(v)) => k == v,
        _ => false,
    }
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
        println!("{}", warn("Benchmark skipped: no nodes across tables."));
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
        println!("{}", warn("Benchmark skipped: no nodes in table."));
        return Ok(());
    }
    let total_nodes: u64 = tables.iter().map(|t| t.node_count()).sum();
    let heading = if round_robin { "Benchmark all" } else { "Benchmark" };
    println!(
        "{} {}={} {}={} {}={} {}={}",
        label(heading),
        label("threads"),
        ok(&threads.to_string()),
        label("duration"),
        ok(&format!("{}s", duration.as_secs())),
        label("tables"),
        ok(&table_count.to_string()),
        label("nodes"),
        ok(&format_number(total_nodes))
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
    println!("{}", label("Results:"));
    let (offsets_bytes, edges_bytes) = total_table_bytes(&tables);
    let data_bytes = offsets_bytes.saturating_add(edges_bytes);
    println!(
        "  {} {}",
        label("offsets bytes:"),
        ok(&format_bytes(offsets_bytes))
    );
    println!(
        "  {} {}",
        label("edge bytes:"),
        ok(&format_bytes(edges_bytes))
    );
    if let Some(available) = available_memory_bytes() {
        println!(
            "  {} {}",
            label("available ram:"),
            ok(&format_bytes(available))
        );
        let load_factor = if available > 0 {
            data_bytes as f64 / available as f64
        } else {
            0.0
        };
        println!("  {} {}", label("load factor:"), ok(&format!("{load_factor:.2}")));
    } else {
        println!("  {} {}", label("available ram:"), dim("n/a"));
        println!("  {} {}", label("load factor:"), dim("n/a"));
    }
    println!("  {} {}", label("ops:"), ok(&format_number(total_ops)));
    println!("  {} {}", label("ops/s:"), ok(&format_number(ops_per_sec)));
    println!(
        "  {} {}",
        label("errors:"),
        if total_errors == 0 {
            ok(&format_number(total_errors))
        } else {
            warn(&format_number(total_errors))
        }
    );
    println!(
        "  {} {}",
        label("edges read:"),
        ok(&format_number(total_edges))
    );
    println!(
        "  {} {}",
        label("edges/s:"),
        ok(&format_number((total_edges as f64 / elapsed).round() as u64))
    );
    println!(
        "  {} {}",
        label("edge bytes:"),
        ok(&format_bytes(total_edge_bytes))
    );
    let edge_bps = (total_edge_bytes as f64 / elapsed).round() as u64;
    println!(
        "  {} {}/s",
        label("edge bytes/s:"),
        ok(&format_bytes(edge_bps))
    );
    println!(
        "  {} {}",
        label("node_data bytes:"),
        ok(&format_bytes(total_node_data))
    );
    let node_data_bps = (total_node_data as f64 / elapsed).round() as u64;
    println!(
        "  {} {}/s",
        label("node_data bytes/s:"),
        ok(&format_bytes(node_data_bps))
    );
    println!("  {} {}", label("checksum:"), dim(&checksum.to_string()));
    println!(
        "  {} {}",
        label("avg:"),
        ok(&format_duration_ns(avg_ns as u128))
    );
    if min_ns != u128::MAX {
        println!(
            "  {} {}",
            label("min:"),
            ok(&format_duration_ns(min_ns))
        );
    }
    println!("  {} {}", label("max:"), ok(&format_duration_ns(max_ns)));
    println!(
        "  {} {}",
        label("p50:"),
        ok(&format_duration_ns(percentile_ns(&hist, &BUCKETS_US, 0.50)))
    );
    println!(
        "  {} {}",
        label("p90:"),
        warn(&format_duration_ns(percentile_ns(&hist, &BUCKETS_US, 0.90)))
    );
    println!(
        "  {} {}",
        label("p99:"),
        warn(&format_duration_ns(percentile_ns(&hist, &BUCKETS_US, 0.99)))
    );
    println!("{}", label("Latency histogram:"));
    for (i, count) in hist.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        if i < BUCKETS_US.len() {
            let bound_ns = (BUCKETS_US[i] as u128) * 1_000;
            println!(
                "  {} {} {}",
                dim("<="),
                highlight(&format_duration_ns(bound_ns)),
                ok(&format_number(*count))
            );
        } else {
            let bound_ns = (BUCKETS_US[BUCKETS_US.len() - 1] as u128) * 1_000;
            println!(
                "  {} {} {}",
                dim(">"),
                highlight(&format_duration_ns(bound_ns)),
                ok(&format_number(*count))
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
        "{} {} {} {}\n",
        label("Benchmark running:"),
        ok(&format_duration_short(elapsed)),
        dim("/"),
        ok(&format_duration_short(duration))
    ));
    out.push_str(&format!(
        "{} {}\n",
        label("Ops so far:"),
        ok(&format_number(total_ops))
    ));
    let current_ops = samples.last().copied().unwrap_or(0);
    out.push_str(&format!(
        "{} {}\n",
        label("Current ops/s:"),
        highlight(&format_number(current_ops))
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
