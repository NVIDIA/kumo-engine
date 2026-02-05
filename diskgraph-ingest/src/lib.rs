//! Diskgraph ingest: convert raw CSV/Parquet sources into index-ready artifacts.

mod mapping;
pub mod graph_config;
pub mod config_converter;
pub mod utils;
pub mod progress;
pub mod config;
pub mod errors;
pub mod stats;
pub mod stats_publish;
pub mod scratch;
pub mod consts;
pub mod io;
pub mod coercion;
#[cfg(test)]
mod tests_unit;

pub use crate::config::IngestOptions;
use crate::config::PeakMemStats;
pub use crate::io::Dest;
use crate::consts::*;
pub use crate::errors::{CoercionError, CoercionWarning, ConvertError, KeyParseError, KeyParseResult};
use crate::mapping::{
    write_pk_data_entry, write_pk_data_header, Key, KeyType, PkDataAdvice, PkHashShard,
    PkHashStore, PkPtrHash,
};
pub use crate::progress::{
    format_bytes, format_duration, IndexProgressAdapter, ProgressCallback, ProgressEvent,
    ProgressKind, ProgressPhase, TableIndexTotals,
};
pub use crate::scratch::{
    estimate_pk_map_bytes, estimate_varbytes_avg_len_multi, new_pk_map, reserve_map,
    EdgeMapScratch, PkMapHashMap, PkMapHasher, PkMapScratch,
};
pub use crate::stats::{
    IngestReport, IngestStats, KeyFailureCounts, KeyFailureState, PerfStats,
};
use crate::graph_config::{
    ColumnSpec, DirectionSpec, IngestedType, TableFormat, TableSpec, TimeSpec, WeightSpec,
};
use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, EdgeSortKeyMetadata,
    EdgeSortKeyTimestampEncoding, EdgeSortKeyTimestampKind, EdgeSortKeyTimestampMetadata,
    EdgeSortKeyWeightEncoding, EdgeSortKeyWeightKind, EdgeSortKeyWeightMetadata, EdgeStorageFormat,
    NodeDataCompression, NodeShardMetadata, NodeTypeMetadata, PrimaryKeyMap,
};
use diskgraph_fs::CloudTransfer;
use crate::io::{
    build_input_transfer, download_cloud_bytes, is_cloud_path, resolve_local_paths, OutputTarget, OutputSink, require_local_output,
};
pub use crate::io::{
    build_transfer_for_cloud_path, object_path_from_cloud_url,
};
use diskgraph_rowpack::{
    IngestedType as RowpackType, RowAccess, RowAccessHints, RowAccessStats, RowPackConfig,
    RowPackContextBuilder, RowPackEncoder, RowPackRowLens,
};
use diskgraph_index::builder::{EdgeBufferPool as IndexEdgeBufferPool, MemoryBudget};
use diskgraph_index::graph::{
    GraphDataSource, GraphIndexBuilder, GraphProgressSink,
};
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use ptr_hash::PtrHashScratch;
use fasteval::{eval_compiled_ref, Compiler, Evaler, Parser, Slab};
use half::f16;
use memmap2::Mmap;
use num_traits::{Signed, ToPrimitive};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pprof::protos::Message;
use pprof::ProfilerGuard;
use serde_json::Value as JsonValue;
use gxhash::GxHasher;
use epserde::prelude::Serialize as EpSerialize;
use ptr_hash::DefaultPtrHash;
use std::fs;
use std::io::{self as std_io, BufWriter, Cursor, Read, Write};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::process::{Command, Stdio};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use uuid_simd::{parse_hyphenated, parse_simple, AsOut};

pub use graph_config::GraphConfig;




const DEFAULT_INDEX_MEM_BUDGET_FRAC: f64 = 0.7;

pub fn ingest(
    graph_config_path: &Path,
    output: &str,
    dest: Dest,
    options: IngestOptions,
) -> Result<DatasetMetadata> {
    let (metadata, _) = ingest_with_stats(graph_config_path, output, dest, options)?;
    Ok(metadata)
}

pub fn ingest_with_stats(
    graph_config_path: &Path,
    output: &str,
    dest: Dest,
    options: IngestOptions,
) -> Result<(DatasetMetadata, IngestStats)> {
    let metadata_bytes = fs::read(graph_config_path)
        .with_context(|| format!("read {}", graph_config_path.display()))?;
    let source: GraphConfig = serde_json::from_slice(&metadata_bytes)?;
    let base_dir = graph_config_path
        .parent()
        .ok_or_else(|| anyhow!("missing parent for {}", graph_config_path.display()))?;
    ingest_with_stats_from_source(source, Some(base_dir), output, dest, options)
}

pub fn ingest_with_stats_from_source(
    mut source: GraphConfig,
    base_dir: Option<&Path>,
    output: &str,
    dest: Dest,
    options: IngestOptions,
) -> Result<(DatasetMetadata, IngestStats)> {
    let wall_start = std::time::Instant::now();
    if let Some(base_dir) = base_dir {
        resolve_local_paths(&mut source, base_dir);
    }
    source.validate()?;

    // Resolve FK column types to match their referenced PK types.
    // This enables FK→PK joins across different source types (e.g., String FK → Int64 PK).
    let source = source.resolve_fk_types();

    let target = std::sync::Arc::new(OutputTarget::parse(output, dest)?);
    if !matches!(&*target, OutputTarget::Local { .. }) {
        return Err(anyhow!("index build requires local output"));
    }
    target.prepare()?;
    if let OutputTarget::Local { root } = &*target {
        if root.exists() {
            fs::remove_dir_all(root)?;
            fs::create_dir_all(root)?;
        }
    }

    // Save the resolved config (with FK types updated to match PKs)
    target.write_bytes("config_resolved.json", serde_json::to_vec_pretty(&source)?)?;

    let mut ctx = IngestContext::new(source, target, options, output)?;
    let profile_dir = ctx.profile_dir.clone();
    if let Some(profile_dir) = &profile_dir {
        fs::create_dir_all(profile_dir)?;
    }
    ctx.set_peak_stage(Some(0));
    let context_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "context", || ctx.build_node_contexts())?;
    } else {
        ctx.build_node_contexts()?;
    }
    let context_wall = context_start.elapsed();
    eprintln!(
        "context stage complete elapsed={}",
        format_duration(context_wall.as_millis() as u64)
    );
    ctx.set_peak_stage(Some(1));
    let encode_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "encode", || ctx.build_temp_shards())?;
    } else {
        ctx.build_temp_shards()?;
    }
    let encode_wall = encode_start.elapsed();
    eprintln!(
        "encode stage complete elapsed={}",
        format_duration(encode_wall.as_millis() as u64)
    );
    ctx.set_peak_stage(Some(2));
    let pk_map_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "pk_map", || ctx.build_sharded_pk_maps())?;
    } else {
        ctx.build_sharded_pk_maps()?;
    }
    let pk_map_wall = pk_map_start.elapsed();
    eprintln!(
        "pk_map stage complete elapsed={}",
        format_duration(pk_map_wall.as_millis() as u64)
    );
    let edge_ctx = ctx.edge_build_context()?;
    ctx.set_peak_stage(Some(3));
    let edges_map_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "edges_map", || ctx.build_edges_mapped(&edge_ctx))?;
    } else {
        ctx.build_edges_mapped(&edge_ctx)?;
    }
    let edges_map_wall = edges_map_start.elapsed();
    eprintln!(
        "edges_map stage complete elapsed={}",
        format_duration(edges_map_wall.as_millis() as u64)
    );
    drop(edge_ctx);
    ctx.set_peak_stage(Some(4));
    let index_start = std::time::Instant::now();
    let index_metadata = ctx.build_index_metadata()?;
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "index", || ctx.build_index(index_metadata.clone()))?;
    } else {
        ctx.build_index(index_metadata.clone())?;
    }
    let index_wall = index_start.elapsed();
    eprintln!(
        "index stage complete elapsed={}",
        format_duration(index_wall.as_millis() as u64)
    );
    ctx.set_peak_stage(Some(5));
    let finalize_start = std::time::Instant::now();
    let index_metadata = ctx.finalize_metadata(index_metadata)?;
    if let OutputTarget::Local { root } = &*ctx.target {
        let nodes_root = root.join(TEMP_DIR_NAME).join("nodes");
        if nodes_root.exists() {
            ctx.temp_node_data_bytes = sum_dir_bytes(&nodes_root).unwrap_or(0);
        }
        let edges_root = root.join(TEMP_DIR_NAME).join("edges_map");
        if edges_root.exists() {
            ctx.temp_edges_map_bytes = sum_dir_bytes(&edges_root).unwrap_or(0);
        }
    }
    ctx.cleanup_temp()?;
    let finalize_wall = finalize_start.elapsed();
    ctx.set_peak_stage(None);
    eprintln!(
        "finalize stage complete elapsed={}",
        format_duration(finalize_wall.as_millis() as u64)
    );
    let wall_time = wall_start.elapsed();
    let bytes_read = sum_input_bytes(&ctx.source);
    let bytes_written = match &*ctx.target {
        OutputTarget::Local { root } => sum_dir_bytes(root).unwrap_or(0),
        _ => 0,
    };
    let perf = PerfStats {
        wall_time_ms: wall_time.as_millis() as u64,
        context_wall_ms: context_wall.as_millis() as u64,
        encode_wall_ms: encode_wall.as_millis() as u64,
        pk_map_wall_ms: pk_map_wall.as_millis() as u64,
        edges_map_wall_ms: edges_map_wall.as_millis() as u64,
        index_wall_ms: index_wall.as_millis() as u64,
        finalize_wall_ms: finalize_wall.as_millis() as u64,
        bytes_read,
        bytes_written,
        temp_edges_src_bytes: ctx.temp_edges_src_bytes,
        temp_fkeys_bytes: ctx.temp_fkeys_bytes,
        temp_edges_map_bytes: ctx.temp_edges_map_bytes,
        temp_node_data_bytes: ctx.temp_node_data_bytes,
        temp_bytes_total: ctx
            .temp_edges_src_bytes
            .saturating_add(ctx.temp_fkeys_bytes)
            .saturating_add(ctx.temp_edges_map_bytes)
            .saturating_add(ctx.temp_node_data_bytes),
        peak_rss_anon_bytes: ctx
            .peak_mem_stats
            .as_ref()
            .map(|v| v.overall_max())
            .filter(|v| *v > 0),
        peak_context_rss_anon_bytes: ctx
            .peak_mem_stats
            .as_ref()
            .map(|v| v.stage_max(0))
            .filter(|v| *v > 0),
        peak_encode_rss_anon_bytes: ctx
            .peak_mem_stats
            .as_ref()
            .map(|v| v.stage_max(1))
            .filter(|v| *v > 0),
        peak_pk_map_rss_anon_bytes: ctx
            .peak_mem_stats
            .as_ref()
            .map(|v| v.stage_max(2))
            .filter(|v| *v > 0),
        peak_edges_map_rss_anon_bytes: ctx
            .peak_mem_stats
            .as_ref()
            .map(|v| v.stage_max(3))
            .filter(|v| *v > 0),
        peak_index_rss_anon_bytes: ctx
            .peak_mem_stats
            .as_ref()
            .map(|v| v.stage_max(4))
            .filter(|v| *v > 0),
        peak_finalize_rss_anon_bytes: ctx
            .peak_mem_stats
            .as_ref()
            .map(|v| v.stage_max(5))
            .filter(|v| *v > 0),
    };
    let stats = ctx.build_stats(&index_metadata, perf);
    let mut stats = stats;
    stats.num_shards = Some(ctx.num_shards);
    let report = IngestReport {
        stats: stats.clone(),
        source: ctx.source.clone(),
    };
    ctx.target
        .write_bytes("ingest_stats.json", serde_json::to_vec_pretty(&report)?)?;
    Ok((index_metadata, stats))
}

#[derive(Clone)]

struct IngestContext {
    source: GraphConfig,
    target: std::sync::Arc<OutputTarget>,
    profile_dir: Option<PathBuf>,
    parallel: bool,
    index_mem_budget: Option<std::sync::Arc<MemoryBudget>>,
    num_shards: usize,
    input_transfer: Option<CloudTransfer>,
    table_rows: HashMap<String, u64>,
    table_contexts: HashMap<String, String>,
    shard_rows: HashMap<String, Vec<u64>>,
    shard_offsets: HashMap<String, Vec<u64>>,
    pk_hash_paths: HashMap<String, Vec<PkHashPaths>>,
    relation_defs: Vec<RelationDef>,
    relationships: Vec<RelationshipRuntime>,
    edge_counts: HashMap<String, u64>,
    pk_duplicate_counts: HashMap<String, u64>,
    pk_duplicate_drops: HashMap<String, u64>,
    pk_null_timestamp_drops: HashMap<String, u64>,
    edge_null_timestamp_drops: HashMap<String, u64>,
    pk_duplicate_samples: HashMap<String, Vec<String>>,
    dropped_rows_counts: HashMap<String, u64>,
    key_parse_failures: u64,
    key_lookup_failures: u64,
    key_failure_samples: Vec<String>,
    key_failures_by_context: HashMap<String, KeyFailureCounts>,
    output: String,
    trimmed_strings: std::sync::Arc<AtomicU64>,
    progress: Option<ProgressCallback>,
    temp_edges_src_bytes: u64,
    temp_fkeys_bytes: u64,
    temp_edges_map_bytes: u64,
    temp_node_data_bytes: u64,
    peak_mem_stats: Option<std::sync::Arc<PeakMemStats>>,
}

struct EdgeBuildContext {
    root: PathBuf,
    rel_sort_bits: HashMap<String, Option<u8>>,
    rel_sort_lookups: std::sync::Arc<HashMap<String, Option<std::sync::Arc<SortKeyLookup>>>>,
    max_ids: HashMap<String, u64>,
    pk_hash_store: Arc<PkHashStore>,
    pk_hash_total_bytes: u64,
    sort_key_total_bytes: u64,
    sort_key_edge_bytes: u64,
    sort_key_dest_bytes: u64,
}

struct SortKeyComponentStore {
    width_bytes: usize,
    data: Mmap,
    is_edge: bool,
}

enum SortKeyComponentSource {
    DestTable { shards: Vec<SortKeyComponentStore> },
    EdgeTable { shards: Vec<SortKeyComponentStore> },
}

struct SortKeyLookup {
    weight_bits: u8,
    timestamp: Option<SortKeyComponentSource>,
    weight: Option<SortKeyComponentSource>,
}

impl SortKeyComponentStore {
    fn get(&self, idx: u64) -> Result<u64> {
        let offset = (idx as usize)
            .checked_mul(self.width_bytes)
            .ok_or_else(|| anyhow!("sort key offset overflow"))?;
        let end = offset + self.width_bytes;
        if end > self.data.len() {
            return Err(anyhow!("sort key offset out of range"));
        }
        let mut buf = [0u8; 8];
        buf[..self.width_bytes].copy_from_slice(&self.data[offset..end]);
        Ok(u64::from_le_bytes(buf))
    }

    fn total_bytes(&self) -> u64 {
        self.data.len() as u64
    }

    fn edge_bytes(&self) -> u64 {
        if self.is_edge {
            self.total_bytes()
        } else {
            0
        }
    }

    fn dest_bytes(&self) -> u64 {
        if self.is_edge {
            0
        } else {
            self.total_bytes()
        }
    }
}

impl SortKeyComponentSource {
    fn get_value(
        &self,
        dest_shard: usize,
        dest_local: u64,
        edge_shard: usize,
        edge_row: u64,
    ) -> Result<u64> {
        match self {
            SortKeyComponentSource::DestTable { shards } => shards
                .get(dest_shard)
                .ok_or_else(|| anyhow!("sort key dest shard missing"))?
                .get(dest_local),
            SortKeyComponentSource::EdgeTable { shards } => shards
                .get(edge_shard)
                .ok_or_else(|| anyhow!("sort key edge shard missing"))?
                .get(edge_row),
        }
    }

    fn total_bytes(&self) -> u64 {
        match self {
            SortKeyComponentSource::DestTable { shards }
            | SortKeyComponentSource::EdgeTable { shards } => {
                shards.iter().map(|shard| shard.total_bytes()).sum()
            }
        }
    }

    fn edge_bytes(&self) -> u64 {
        match self {
            SortKeyComponentSource::DestTable { shards }
            | SortKeyComponentSource::EdgeTable { shards } => {
                shards.iter().map(|shard| shard.edge_bytes()).sum()
            }
        }
    }

    fn dest_bytes(&self) -> u64 {
        match self {
            SortKeyComponentSource::DestTable { shards }
            | SortKeyComponentSource::EdgeTable { shards } => {
                shards.iter().map(|shard| shard.dest_bytes()).sum()
            }
        }
    }
}

impl SortKeyLookup {
    fn sort_key(
        &self,
        dest_shard: usize,
        dest_local: u64,
        edge_shard: usize,
        edge_row: u64,
    ) -> Result<u64> {
        let ts_val = if let Some(ts) = &self.timestamp {
            ts.get_value(dest_shard, dest_local, edge_shard, edge_row)?
        } else {
            0
        };
        let weight_val = if let Some(weight) = &self.weight {
            weight.get_value(dest_shard, dest_local, edge_shard, edge_row)?
        } else {
            0
        };
        if self.weight_bits > 0 {
            Ok((ts_val << self.weight_bits) | weight_val)
        } else {
            Ok(ts_val)
        }
    }

    fn total_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Some(ts) = &self.timestamp {
            total = total.saturating_add(ts.total_bytes());
        }
        if let Some(weight) = &self.weight {
            total = total.saturating_add(weight.total_bytes());
        }
        total
    }

    fn edge_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Some(ts) = &self.timestamp {
            total = total.saturating_add(ts.edge_bytes());
        }
        if let Some(weight) = &self.weight {
            total = total.saturating_add(weight.edge_bytes());
        }
        total
    }

    fn dest_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Some(ts) = &self.timestamp {
            total = total.saturating_add(ts.dest_bytes());
        }
        if let Some(weight) = &self.weight {
            total = total.saturating_add(weight.dest_bytes());
        }
        total
    }
}

#[derive(Clone)]
struct PkHashPaths {
    hash_path: String,
    data_path: String,
}

struct TempRowpackMeta {
    row_count: u64,
    #[allow(dead_code)]
    max_row_len: u32,
}

struct TempRowWriter {
    writer: FrameEncoder<OutputSink>,
    row_count: u64,
}

struct TempNodeFileResult {
    total_rows: u64,
    node_data_dropped: u64,
}

fn spawn_background_delete(path: PathBuf) {
    if !path.exists() {
        return;
    }
    if let Err(err) = spawn_delete_process(&path) {
        tracing::warn!("delete spawn failed for {}: {}", path.display(), err);
    }
}

fn spawn_delete_process(path: &Path) -> std_io::Result<()> {
    let mut cmd = if command_exists("ionice") {
        let mut cmd = Command::new("ionice");
        cmd.args([
            "-c3",
            "-n7",
            "nice",
            "-n",
            "19",
            "rm",
            "-rf",
            "--",
        ]);
        cmd.arg(path);
        cmd
    } else if command_exists("nice") {
        let mut cmd = Command::new("nice");
        cmd.args(["-n", "19", "rm", "-rf", "--"]);
        cmd.arg(path);
        cmd
    } else {
        let mut cmd = Command::new("rm");
        cmd.args(["-rf", "--"]);
        cmd.arg(path);
        cmd
    };

    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    detach_command(&mut cmd);
    let _child = cmd.spawn()?;
    Ok(())
}

fn command_exists(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|mut child| {
            let _ = child.wait();
            true
        })
        .unwrap_or(false)
}

#[cfg(unix)]
fn detach_command(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn detach_command(_cmd: &mut Command) {}

#[derive(Clone)]
struct RelationDef {
    name: String,
    src_table: String,
    dst_table: String,
    sort_key: Option<RelationSortKeyConfig>,
}

#[derive(Clone)]
struct RelationshipRuntime {
    name_forward: String,
    name_reverse: String,
    src_table: String,
    src_column: String,
    dst_table: String,
}

// Internal runtime types for encoding specs (moved from graph_config.rs)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimestampEncodingKind {
    Int8,
    Int16,
    Int32,
}

#[derive(Clone, Debug)]
pub(crate) struct TimestampEncodingSpec {
    pub(crate) kind: TimestampEncodingKind,
    pub(crate) anchor: i64,
    pub(crate) step: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WeightEncodingKind {
    Byte,
    UInt16,
    UInt32,
    Float16,
    Float32,
}

#[derive(Clone, Debug)]
struct WeightEncodingSpec {
    dtype: WeightEncodingKind,
}

/// Convert TimeSpec from config to internal encoding spec.
fn time_spec_to_encoding(spec: &TimeSpec) -> TimestampEncodingSpec {
    let kind = match spec.bitwidth {
        Some(8) => TimestampEncodingKind::Int8,
        Some(16) => TimestampEncodingKind::Int16,
        Some(32) | None => TimestampEncodingKind::Int32, // default to 32-bit
        Some(other) => {
            eprintln!("warning: unsupported timestamp bitwidth {other}, using 32-bit");
            TimestampEncodingKind::Int32
        }
    };
    let step = match spec.unit.as_str() {
        "s" | "second" | "seconds" => 1,
        "ms" | "milli" | "millis" | "millisecond" | "milliseconds" => 1,
        "us" | "micro" | "micros" | "microsecond" | "microseconds" => 1,
        "ns" | "nano" | "nanos" | "nanosecond" | "nanoseconds" => 1,
        "minute" | "minutes" | "min" => 60,
        "hour" | "hours" | "h" => 3600,
        "day" | "days" | "d" => 86400,
        "week" | "weeks" | "w" => 604800,
        other => {
            eprintln!("warning: unsupported timestamp unit '{other}', using seconds");
            1
        }
    };
    TimestampEncodingSpec {
        kind,
        anchor: spec.anchor_time.unwrap_or(0),
        step,
    }
}

/// Convert WeightSpec from config to internal encoding spec.
fn weight_spec_to_encoding(spec: &WeightSpec) -> WeightEncodingSpec {
    let dtype = match spec.dtype.as_str() {
        "byte" | "u8" | "uint8" => WeightEncodingKind::Byte,
        "uint16" | "u16" => WeightEncodingKind::UInt16,
        "uint32" | "u32" => WeightEncodingKind::UInt32,
        "float16" | "f16" => WeightEncodingKind::Float16,
        "float32" | "f32" => WeightEncodingKind::Float32,
        other => {
            eprintln!("warning: unsupported weight dtype '{other}', using float32");
            WeightEncodingKind::Float32
        }
    };
    WeightEncodingSpec { dtype }
}

#[derive(Clone)]
struct RelationSortKeyConfig {
    timestamp: Option<SortKeyTimestampConfig>,
    weight: Option<SortKeyWeightConfig>,
    width_bits: u8,
}

#[derive(Clone)]
struct SortKeyTimestampConfig {
    table: String,
    column: String,
    encoding: TimestampEncodingSpec,
}

#[derive(Clone)]
struct SortKeyWeightConfig {
    table: String,
    expr: String,
    encoding: WeightEncodingSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SortKeyComponentKind {
    Timestamp,
    Weight,
}

#[derive(Clone)]
struct SortKeyComponentSpec {
    kind: SortKeyComponentKind,
    table: String,
    width_bits: u8,
}

#[derive(Clone)]
struct RelationSortKeyInfo {
    timestamp: Option<SortKeyComponentInfo>,
    weight: Option<SortKeyComponentInfo>,
}

#[derive(Clone)]
struct SortKeyComponentInfo {
    spec: SortKeyComponentSpec,
}

enum SortKeyComponentEncoder {
    Timestamp {
        column_idx: usize,
        encoding: TimestampEncodingSpec,
    },
    Weight {
        expr: WeightExprRuntime,
        encoding: WeightEncodingSpec,
    },
}

struct WeightExprRuntime {
    slab: Slab,
    compiled: fasteval::compiler::Instruction,
    vars: Vec<WeightExprVar>,
    var_values: std::collections::BTreeMap<String, f64>,
    simple_idx: Option<usize>,
    simple_dtype: Option<IngestedType>,
    simple_scale: Option<i32>,
}

struct WeightExprVar {
    name: String,
    idx: usize,
    dtype: IngestedType,
    scale: Option<i32>,
}

fn timestamp_width_bits(kind: &TimestampEncodingKind) -> u8 {
    match kind {
        TimestampEncodingKind::Int8 => 8,
        TimestampEncodingKind::Int16 => 16,
        TimestampEncodingKind::Int32 => 32,
    }
}

fn weight_width_bits(kind: &WeightEncodingKind) -> u8 {
    match kind {
        WeightEncodingKind::Byte => 8,
        WeightEncodingKind::UInt16 => 16,
        WeightEncodingKind::UInt32 => 32,
        WeightEncodingKind::Float16 => 16,
        WeightEncodingKind::Float32 => 32,
    }
}

fn build_relation_sort_key_config(spec: &DirectionSpec) -> Option<RelationSortKeyConfig> {
    let mut width_bits: u16 = 0;
    let timestamp = spec.time.as_ref().map(|ts| {
        let encoding = time_spec_to_encoding(ts);
        width_bits += timestamp_width_bits(&encoding.kind) as u16;
        SortKeyTimestampConfig {
            table: ts.table.clone(),
            column: ts.column.clone(),
            encoding,
        }
    });
    let weight = spec.weight.as_ref().map(|w| {
        let encoding = weight_spec_to_encoding(w);
        width_bits += weight_width_bits(&encoding.dtype) as u16;
        SortKeyWeightConfig {
            table: w.table.clone(),
            expr: w.expr.clone(),
            encoding,
        }
    });
    if width_bits == 0 {
        None
    } else {
        Some(RelationSortKeyConfig {
            timestamp,
            weight,
            width_bits: width_bits as u8,
        })
    }
}

fn build_sort_key_metadata(cfg: &RelationSortKeyConfig) -> EdgeSortKeyMetadata {
    EdgeSortKeyMetadata {
        width_bits: cfg.width_bits,
        timestamp: cfg.timestamp.as_ref().map(|ts| EdgeSortKeyTimestampMetadata {
            table: ts.table.clone(),
            column: ts.column.clone(),
            encoding: EdgeSortKeyTimestampEncoding {
                kind: match ts.encoding.kind {
                    TimestampEncodingKind::Int8 => EdgeSortKeyTimestampKind::Int8,
                    TimestampEncodingKind::Int16 => EdgeSortKeyTimestampKind::Int16,
                    TimestampEncodingKind::Int32 => EdgeSortKeyTimestampKind::Int32,
                },
                anchor: ts.encoding.anchor,
                step: ts.encoding.step,
            },
        }),
        weight: cfg.weight.as_ref().map(|w| EdgeSortKeyWeightMetadata {
            table: w.table.clone(),
            expr: w.expr.clone(),
            encoding: EdgeSortKeyWeightEncoding {
                dtype: match w.encoding.dtype {
                    WeightEncodingKind::Byte => EdgeSortKeyWeightKind::Byte,
                    WeightEncodingKind::UInt16 => EdgeSortKeyWeightKind::UInt16,
                    WeightEncodingKind::UInt32 => EdgeSortKeyWeightKind::UInt32,
                    WeightEncodingKind::Float16 => EdgeSortKeyWeightKind::Float16,
                    WeightEncodingKind::Float32 => EdgeSortKeyWeightKind::Float32,
                },
            },
        }),
    }
}

fn rewrite_weight_expr(expr: &str) -> Result<(String, std::collections::HashMap<String, String>)> {
    let mut out = String::new();
    let mut map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut chars = expr.chars().peekable();
    let mut idx = 0usize;
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut name = String::new();
            let mut escaped = false;
            while let Some(c) = chars.next() {
                if escaped {
                    match c {
                        'n' => name.push('\n'),
                        'r' => name.push('\r'),
                        't' => name.push('\t'),
                        '\\' => name.push('\\'),
                        '`' => name.push('`'),
                        other => name.push(other),
                    }
                    escaped = false;
                    continue;
                }
                if c == '\\' {
                    escaped = true;
                    continue;
                }
                if c == '`' {
                    break;
                }
                name.push(c);
            }
            if escaped {
                return Err(anyhow!("unterminated escape in `{expr}`"));
            }
            if name.is_empty() {
                return Err(anyhow!("empty backtick identifier"));
            }
            let var = format!("__c{idx}");
            idx += 1;
            out.push_str(&var);
            map.insert(var, name);
        } else {
            out.push(ch);
        }
    }
    Ok((out, map))
}

fn compile_weight_expr(expr: &str, table: &TableSpec) -> Result<WeightExprRuntime> {
    let (rewritten, var_map) = rewrite_weight_expr(expr)?;
    let parser = Parser::new();
    let mut slab = Slab::new();
    let expr_i = parser
        .parse(&rewritten, &mut slab.ps)
        .map_err(|e| anyhow!("parse error: {e}"))?;
    let expr_ref = expr_i.from(&slab.ps);
    let vars = expr_ref.var_names(&slab);
    if vars.is_empty() {
        return Err(anyhow!("expression has no column references"));
    }
    let mut var_values = std::collections::BTreeMap::new();
    let mut runtime_vars = Vec::with_capacity(vars.len());
    let mut simple_idx = None;
    let mut simple_dtype = None;
    let mut simple_scale = None;
    for var in vars {
        let column = var_map.get(&var).cloned().unwrap_or_else(|| var.clone());
        let idx = column_index(table, &column)?;
        let col = column_spec(table, &column)?;
        var_values.insert(var.clone(), 0.0);
        runtime_vars.push(WeightExprVar {
            name: var.clone(),
            idx,
            dtype: col.ingested_dtype.clone(),
            scale: col.scale,
        });
        if rewritten.trim() == var {
            simple_idx = Some(idx);
            simple_dtype = Some(col.ingested_dtype.clone());
            simple_scale = col.scale;
        }
    }
    let compiled = expr_ref.compile(&slab.ps, &mut slab.cs);
    Ok(WeightExprRuntime {
        slab,
        compiled,
        vars: runtime_vars,
        var_values,
        simple_idx,
        simple_dtype,
        simple_scale,
    })
}

fn access_value_to_f64(
    access: &mut RowAccess<'_>,
    idx: usize,
    dtype: &IngestedType,
    scale: Option<i32>,
) -> Option<f64> {
    match dtype {
        IngestedType::Int64 => access.get_int64(idx).ok().flatten().map(|v| v as f64),
        IngestedType::Int32 | IngestedType::Int16 | IngestedType::Int8 => {
            access.get_int32(idx).ok().flatten().map(|v| v as f64)
        }
        IngestedType::UInt64 => access.get_uint64(idx).ok().flatten().map(|v| v as f64),
        IngestedType::UInt32 | IngestedType::UInt16 | IngestedType::UInt8 => {
            access.get_uint32(idx).ok().flatten().map(|v| v as f64)
        }
        IngestedType::Float64 => access.get_float64(idx).ok().flatten(),
        IngestedType::Float32 => access.get_float32(idx).ok().flatten().map(|v| v as f64),
        IngestedType::Decimal128 => {
            let scale = scale.unwrap_or(0);
            let denom = 10_f64.powi(scale);
            access
                .get_decimal128(idx)
                .ok()
                .flatten()
                .map(|v| (v as f64) / denom)
        }
        IngestedType::Decimal256 => {
            let scale = scale.unwrap_or(0);
            access
                .get_decimal256(idx)
                .ok()
                .flatten()
                .and_then(|v| {
                    let num = num_bigint::BigInt::from_signed_bytes_le(&v);
                    num.to_f64().map(|f| f / 10_f64.powi(scale))
                })
        }
        _ => None,
    }
}

pub(crate) fn encode_timestamp_value(
    value: Option<i64>,
    encoding: &TimestampEncodingSpec,
) -> u64 {
    let step = encoding.step.max(1);
    let raw = value.unwrap_or(i64::MIN);
    let delta = raw.saturating_sub(encoding.anchor).div_euclid(step);
    match encoding.kind {
        TimestampEncodingKind::Int8 => {
            let clamped = delta.clamp(i8::MIN as i64, i8::MAX as i64) as i8;
            ordered_signed_to_u64(clamped as i64, 8)
        }
        TimestampEncodingKind::Int16 => {
            let clamped = delta.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
            ordered_signed_to_u64(clamped as i64, 16)
        }
        TimestampEncodingKind::Int32 => {
            let clamped = delta.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            ordered_signed_to_u64(clamped as i64, 32)
        }
    }
}

pub(crate) fn ordered_signed_to_u64(value: i64, bits: u8) -> u64 {
    let mask = 1u64 << (bits - 1);
    let raw = match bits {
        8 => (value as i8 as u8) as u64,
        16 => (value as i16 as u16) as u64,
        32 => (value as i32 as u32) as u64,
        _ => value as u64,
    };
    raw ^ mask
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

fn ordered_f16_bits(value: f32) -> u16 {
    let bits = f16::from_f32(value).to_bits();
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits ^ 0x8000
    }
}

fn encode_weight_value(value: Option<f64>, encoding: &WeightEncodingSpec) -> u64 {
    let val = value.unwrap_or(0.0);
    if !val.is_finite() {
        return 0;
    }
    match encoding.dtype {
        WeightEncodingKind::Byte => val.clamp(0.0, u8::MAX as f64) as u64,
        WeightEncodingKind::UInt16 => val.clamp(0.0, u16::MAX as f64) as u64,
        WeightEncodingKind::UInt32 => val.clamp(0.0, u32::MAX as f64) as u64,
        WeightEncodingKind::Float16 => ordered_f16_bits(val as f32) as u64,
        WeightEncodingKind::Float32 => ordered_f32_bits(val as f32) as u64,
    }
}

fn eval_weight_expr(expr: &mut WeightExprRuntime, access: &mut RowAccess<'_>) -> Result<Option<f64>> {
    if let Some(idx) = expr.simple_idx {
        return Ok(access_value_to_f64(
            access,
            idx,
            expr.simple_dtype.as_ref().unwrap(),
            expr.simple_scale,
        ));
    }
    for var in &expr.vars {
        let num = access_value_to_f64(access, var.idx, &var.dtype, var.scale);
        if let Some(slot) = expr.var_values.get_mut(&var.name) {
            if let Some(num) = num {
                *slot = num;
            } else {
                return Ok(None);
            }
        }
    }
    let val = eval_compiled_ref!(&expr.compiled, &expr.slab, &mut expr.var_values);
    Ok(Some(val))
}

fn build_sort_key_batchers_for_node_table(
    table: &TableSpec,
    relation_defs: &[RelationDef],
    num_shards: usize,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>>,
) -> Result<Vec<SortKeyComponentBatcher>> {
    let mut batchers = Vec::new();
    for rel in relation_defs {
        let cfg = match &rel.sort_key {
            Some(cfg) => cfg,
            None => continue,
        };
        if let Some(ts) = &cfg.timestamp {
            if ts.table == table.name {
                let idx = column_index(table, &ts.column)?;
                let width_bytes = (timestamp_width_bits(&ts.encoding.kind) / 8) as u8;
                batchers.push(SortKeyComponentBatcher::new(
                    std::sync::Arc::<str>::from(rel.name.clone()),
                    SortKeyComponentKind::Timestamp,
                    width_bytes,
                    SortKeyComponentEncoder::Timestamp {
                        column_idx: idx,
                        encoding: ts.encoding.clone(),
                    },
                    num_shards,
                    buffer_limit,
                    std::sync::Arc::clone(&senders),
                ));
            }
        }
        if let Some(weight) = &cfg.weight {
            if weight.table == table.name {
                let expr = compile_weight_expr(&weight.expr, table)?;
                let width_bytes = (weight_width_bits(&weight.encoding.dtype) / 8) as u8;
                batchers.push(SortKeyComponentBatcher::new(
                    std::sync::Arc::<str>::from(rel.name.clone()),
                    SortKeyComponentKind::Weight,
                    width_bytes,
                    SortKeyComponentEncoder::Weight {
                        expr,
                        encoding: weight.encoding.clone(),
                    },
                    num_shards,
                    buffer_limit,
                    std::sync::Arc::clone(&senders),
                ));
            }
        }
    }
    Ok(batchers)
}

fn build_sort_key_batchers_for_edge_table(
    table: &TableSpec,
    relation_defs: &[RelationDef],
    num_shards: usize,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>>,
) -> Result<Vec<SortKeyComponentBatcher>> {
    let mut batchers = Vec::new();
    for rel in relation_defs {
        let cfg = match &rel.sort_key {
            Some(cfg) => cfg,
            None => continue,
        };
        if let Some(ts) = &cfg.timestamp {
            if ts.table == table.name {
                let idx = column_index(table, &ts.column)?;
                let width_bytes = (timestamp_width_bits(&ts.encoding.kind) / 8) as u8;
                batchers.push(SortKeyComponentBatcher::new(
                    std::sync::Arc::<str>::from(rel.name.clone()),
                    SortKeyComponentKind::Timestamp,
                    width_bytes,
                    SortKeyComponentEncoder::Timestamp {
                        column_idx: idx,
                        encoding: ts.encoding.clone(),
                    },
                    num_shards,
                    buffer_limit,
                    std::sync::Arc::clone(&senders),
                ));
            }
        }
        if let Some(weight) = &cfg.weight {
            if weight.table == table.name {
                let expr = compile_weight_expr(&weight.expr, table)?;
                let width_bytes = (weight_width_bits(&weight.encoding.dtype) / 8) as u8;
                batchers.push(SortKeyComponentBatcher::new(
                    std::sync::Arc::<str>::from(rel.name.clone()),
                    SortKeyComponentKind::Weight,
                    width_bytes,
                    SortKeyComponentEncoder::Weight {
                        expr,
                        encoding: weight.encoding.clone(),
                    },
                    num_shards,
                    buffer_limit,
                    std::sync::Arc::clone(&senders),
                ));
            }
        }
    }
    Ok(batchers)
}

impl IngestContext {
    fn make_pool<S: Default + Send + 'static>(&self) -> TaskPool<S> {
        TaskPool::new(if self.parallel {
            default_worker_threads()
        } else {
            1
        })
    }

    fn emit_progress(&self, event: ProgressEvent) {
        if let Some(cb) = &self.progress {
            cb(event);
        }
    }

    fn merge_key_failures(&mut self, state: KeyFailureState) {
        self.key_parse_failures = self
            .key_parse_failures
            .saturating_add(state.parse_failures);
        self.key_lookup_failures = self
            .key_lookup_failures
            .saturating_add(state.lookup_failures);
        for (context, counts) in state.by_context {
            let entry = self.key_failures_by_context.entry(context).or_default();
            entry.parse_failures = entry.parse_failures.saturating_add(counts.parse_failures);
            entry.lookup_failures = entry.lookup_failures.saturating_add(counts.lookup_failures);
        }
        for sample in state.samples {
            if self.key_failure_samples.len() >= 10 {
                break;
            }
            self.key_failure_samples.push(sample);
        }
    }
    fn new(
        source: GraphConfig,
        target: std::sync::Arc<OutputTarget>,
        options: IngestOptions,
        output: &str,
    ) -> Result<Self> {
        let index_mem_budget_bytes = options
            .index_mem_budget_bytes
            .or_else(default_index_mem_budget_bytes);
        let index_mem_budget = index_mem_budget_bytes
            .map(|bytes| std::sync::Arc::new(MemoryBudget::new(bytes)));
        let num_shards = source.num_shards.unwrap_or(1).max(1);
        let mut relation_defs = Vec::new();
        let mut relationships = Vec::new();

        // Build table name -> primary key map for looking up dst_column
        let table_pk_map: HashMap<&str, &str> = source
            .tables
            .iter()
            .filter_map(|t| t.primary_key.as_deref().map(|pk| (t.name.as_str(), pk)))
            .collect();

        // Build relationships from fkeys embedded in tables
        for table in &source.tables {
            for fkey in &table.fkeys {
                // Get the destination table's primary key
                let dst_pk = table_pk_map.get(fkey.dst_table.as_str()).copied().unwrap_or("id");

                // Build relationship name: {src_table}_{src_column}_to_{dst_table}_{dst_pk}
                let base = format!(
                    "{}_{}_to_{}_{}",
                    table.name, fkey.src_column, fkey.dst_table, dst_pk
                );

                // Build sort key configs from DirectionSpec
                let dst_sort = fkey
                    .forward
                    .as_ref()
                    .and_then(build_relation_sort_key_config);
                let src_sort = fkey
                    .reverse
                    .as_ref()
                    .and_then(build_relation_sort_key_config);

                relation_defs.push(RelationDef {
                    name: base.clone(),
                    src_table: table.name.clone(),
                    dst_table: fkey.dst_table.clone(),
                    sort_key: dst_sort,
                });
                relation_defs.push(RelationDef {
                    name: format!("{base}_reverse"),
                    src_table: fkey.dst_table.clone(),
                    dst_table: table.name.clone(),
                    sort_key: src_sort,
                });
                relationships.push(RelationshipRuntime {
                    name_forward: base.clone(),
                    name_reverse: format!("{base}_reverse"),
                    src_table: table.name.clone(),
                    src_column: fkey.src_column.clone(),
                    dst_table: fkey.dst_table.clone(),
                });
            }

            // Handle edge tables
            if let Some(edge) = &table.edge_table {
                let base = format!("edge_{}", table.name);
                // Use forward/reverse from EdgeTableSpec directly
                let src_rel = RelationDef {
                    name: format!("{base}_src_to_dst"),
                    src_table: edge.src_fk.ref_table.clone(),
                    dst_table: edge.dst_fk.ref_table.clone(),
                    sort_key: edge
                        .forward
                        .as_ref()
                        .and_then(build_relation_sort_key_config),
                };
                let dst_rel = RelationDef {
                    name: format!("{base}_dst_to_src"),
                    src_table: edge.dst_fk.ref_table.clone(),
                    dst_table: edge.src_fk.ref_table.clone(),
                    sort_key: edge
                        .reverse
                        .as_ref()
                        .and_then(build_relation_sort_key_config),
                };
                relation_defs.push(src_rel);
                relation_defs.push(dst_rel);
            }
        }
        let input_transfer = build_input_transfer(&source)?;
        Ok(Self {
            source,
            target,
            profile_dir: options.profile_dir,
            parallel: options.parallel,
            index_mem_budget,
            num_shards,
            input_transfer,
            table_rows: HashMap::new(),
            table_contexts: HashMap::new(),
            shard_rows: HashMap::new(),
            shard_offsets: HashMap::new(),
            pk_hash_paths: HashMap::new(),
            relation_defs,
            relationships,
            edge_counts: HashMap::new(),
            pk_duplicate_counts: HashMap::new(),
            pk_duplicate_drops: HashMap::new(),
            pk_null_timestamp_drops: HashMap::new(),
            edge_null_timestamp_drops: HashMap::new(),
            pk_duplicate_samples: HashMap::new(),
            dropped_rows_counts: HashMap::new(),
            key_parse_failures: 0,
            key_lookup_failures: 0,
            key_failure_samples: Vec::new(),
            key_failures_by_context: HashMap::new(),
            output: output.to_string(),
            trimmed_strings: std::sync::Arc::new(AtomicU64::new(0)),
            progress: options.progress,
            temp_edges_src_bytes: 0,
            temp_fkeys_bytes: 0,
            temp_edges_map_bytes: 0,
            temp_node_data_bytes: 0,
            peak_mem_stats: options.peak_mem_stats,
        })
    }

    fn set_peak_stage(&self, idx: Option<usize>) {
        if let Some(stats) = &self.peak_mem_stats {
            stats.set_stage(idx);
        }
    }

    fn build_node_contexts(&mut self) -> Result<()> {
        let tables = self.source.tables.clone();
        let (tx, rx) = std::sync::mpsc::channel::<Result<(String, String)>>();
        let mut tasks = Vec::new();
        let source_rowpack = self.source.rowpack.clone();
        let input_transfer = self.input_transfer.clone();
        for table in &tables {
            if table.edge_table.is_some() {
                continue;
            }
            let table = table.clone();
            let target = std::sync::Arc::clone(&self.target);
            let trimmed_strings = std::sync::Arc::clone(&self.trimmed_strings);
            let progress = self.progress.clone();
            let tx = tx.clone();
            let rowpack_cfg = source_rowpack.clone();
            let input_transfer = input_transfer.clone();
            tasks.push(WorkItem {
                bucket: None,
                run: Box::new(move |_scratch| {
                    if let Some(cb) = &progress {
                        cb(ProgressEvent {
                            phase: ProgressPhase::Context,
                            table: table.name.clone(),
                            shard: None,
                            kind: ProgressKind::Start,
                            delta: 0,
                            total: Some(table.get_paths().len() as u64),
                            status: None,
});
                    }
                    let rowpack_config = rowpack_config_for_table(rowpack_cfg.as_ref(), &table);
                    let node_columns = rowpack_columns(&table)?;
                    let mut node_builder =
                        RowPackContextBuilder::new(node_columns, rowpack_config.clone());
                    let mut sampled = 0usize;
                    let sample_limit = rowpack_config.sample_rows().min(SAMPLE_ROW_LIMIT);
                    let row_access_hints = build_row_access_hints(&table);
                    let stats = RowAccessStats::new(Some(trimmed_strings.as_ref()));
                    let mut access = RowAccess::new(row_access_hints, stats);
                    let mut reached_limit = false;
                    let mut processed_paths = 0u64;
                    if table.get_paths().is_empty() {
                        return Err(anyhow!("no input files for table {}", table.name));
                    }
                    'file_loop: for path in table.get_paths() {
                        if !is_cloud_path(path) && !Path::new(path).exists() {
                            return Err(anyhow!("input path not found {}", path));
                        }
                        let pooled = if let Some(transfer) = &input_transfer {
                            if is_cloud_path(path) {
                                Some(download_cloud_bytes(transfer, path)?)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        match table.format() {
                            TableFormat::Parquet => {
                                if let Some(pooled) = pooled {
                                    let bytes = pooled.bytes();
                                    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
                                    let reader =
                                        builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                                    for batch in reader {
                                        let batch = batch?;
                                        let mut cols = Vec::with_capacity(table.columns.len());
                                        for col in &table.columns {
                                            let idx = batch
                                                .schema()
                                                .index_of(&col.name)
                                                .context("missing column in batch")?;
                                            cols.push(batch.column(idx).clone());
                                        }
                                        access.set_cols(&cols);
                                        for row in 0..batch.num_rows() {
                                            if sampled >= sample_limit {
                                                reached_limit = true;
                                                break;
                                            }
                                            access.set_row(row);
                                            node_builder.observe_row_from_access(&mut access)?;
                                            sampled = sampled.saturating_add(1);
                                        }
                                        if reached_limit {
                                            break;
                                        }
                                    }
                                    drop(pooled);
                                } else {
                                    let file = fs::File::open(path)
                                        .with_context(|| format!("open {}", path))?;
                                    advise_file_willneed(&file);
                                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                                    let reader =
                                        builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                                    for batch in reader {
                                        let batch = batch?;
                                        let mut cols = Vec::with_capacity(table.columns.len());
                                        for col in &table.columns {
                                            let idx = batch
                                                .schema()
                                                .index_of(&col.name)
                                                .context("missing column in batch")?;
                                            cols.push(batch.column(idx).clone());
                                        }
                                        access.set_cols(&cols);
                                        for row in 0..batch.num_rows() {
                                            if sampled >= sample_limit {
                                                reached_limit = true;
                                                break;
                                            }
                                            access.set_row(row);
                                            node_builder.observe_row_from_access(&mut access)?;
                                            sampled = sampled.saturating_add(1);
                                        }
                                        if reached_limit {
                                            break;
                                        }
                                    }
                                }
                            }
                            TableFormat::Csv => {
                                let schema = table_schema(&table)?;
                                if let Some(pooled) = pooled {
                                    let bytes = pooled.bytes();
                                    let cursor = Cursor::new(bytes);
                                    let builder = arrow_csv::ReaderBuilder::new(Arc::new(schema))
                                        .with_header(true);
                                    let reader = builder.build(cursor)?;
                                    for batch in reader {
                                        let batch = batch?;
                                        let mut cols = Vec::with_capacity(table.columns.len());
                                        for col in &table.columns {
                                            let idx = batch
                                                .schema()
                                                .index_of(&col.name)
                                                .context("missing column in batch")?;
                                            cols.push(batch.column(idx).clone());
                                        }
                                        access.set_cols(&cols);
                                        for row in 0..batch.num_rows() {
                                            if sampled >= sample_limit {
                                                reached_limit = true;
                                                break;
                                            }
                                            access.set_row(row);
                                            node_builder.observe_row_from_access(&mut access)?;
                                            sampled = sampled.saturating_add(1);
                                        }
                                        if reached_limit {
                                            break;
                                        }
                                    }
                                    drop(pooled);
                                } else {
                                    let file = fs::File::open(path)
                                        .with_context(|| format!("open {}", path))?;
                                    advise_file_willneed(&file);
                                    let builder = arrow_csv::ReaderBuilder::new(Arc::new(schema))
                                        .with_header(true);
                                    let reader = builder.build(file)?;
                                    for batch in reader {
                                        let batch = batch?;
                                        let mut cols = Vec::with_capacity(table.columns.len());
                                        for col in &table.columns {
                                            let idx = batch
                                                .schema()
                                                .index_of(&col.name)
                                                .context("missing column in batch")?;
                                            cols.push(batch.column(idx).clone());
                                        }
                                        access.set_cols(&cols);
                                        for row in 0..batch.num_rows() {
                                            if sampled >= sample_limit {
                                                reached_limit = true;
                                                break;
                                            }
                                            access.set_row(row);
                                            node_builder.observe_row_from_access(&mut access)?;
                                            sampled = sampled.saturating_add(1);
                                        }
                                        if reached_limit {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        processed_paths = processed_paths.saturating_add(1);
                        if let Some(cb) = &progress {
                            cb(ProgressEvent {
                                phase: ProgressPhase::Context,
                                table: table.name.clone(),
                                shard: None,
                                kind: ProgressKind::Advance,
                                delta: 1,
                                total: Some(table.get_paths().len() as u64),
                                status: None,
});
                        }
                        if reached_limit {
                            break 'file_loop;
                        }
                    }
                    let node_context = node_builder.finish()?;
                    let ctx_bytes = node_context.to_bytes()?;
                    target.write_bytes(&temp_node_context_path(&table.name), ctx_bytes.clone())?;
                    target.write_bytes(&node_context_output_path(&table.name), ctx_bytes)?;
                    if let Some(cb) = &progress {
                        cb(ProgressEvent {
                            phase: ProgressPhase::Context,
                            table: table.name.clone(),
                            shard: None,
                            kind: ProgressKind::Finish,
                            delta: 0,
                            total: Some(processed_paths.max(1)),
                            status: None,
});
                    }
                    tx.send(Ok((
                        table.name.clone(),
                        format!("nodes/{}/node_data.ctx", table.name.to_lowercase()),
                    )))
                    .map_err(|e| anyhow!("context output error: {e}"))?;
                    Ok(())
                }),
            });
        }
        drop(tx);
        let task_count = tasks.len();
        let pool = self.make_pool::<()>();
        run_tasks(tasks, &pool)?;
        for _ in 0..task_count {
            let result = rx
                .recv()
                .map_err(|e| anyhow!("context output error: {e}"))??;
            let (table_name, ctx_path) = result;
            self.table_contexts.insert(table_name.clone(), ctx_path);
        }
        Ok(())
    }


    fn build_temp_shards(&mut self) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        let temp_root = root.join(TEMP_DIR_NAME);
        fs::create_dir_all(&temp_root)?;
        let mut table_rows: HashMap<String, u64> = HashMap::new();
        let tables = self.source.tables.clone();
        #[derive(Default)]
        struct EncodeResult {
            table: String,
            total_rows: u64,
            node_data_dropped: u64,
            key_failures: KeyFailureState,
        }

        let (tx, rx) = std::sync::mpsc::channel::<Result<EncodeResult>>();
        let mut tasks = Vec::new();
        let relationships = self.relationships.clone();
        let relation_defs = std::sync::Arc::new(self.relation_defs.clone());
        let num_shards = self.num_shards;
        let target = std::sync::Arc::clone(&self.target);
        let node_pool = NodeWritePool::new(std::sync::Arc::clone(&target), edge_writer_workers());
        let edge_pool = EdgeSrcWritePool::new(std::sync::Arc::clone(&target), edge_writer_workers());
        let dest_key_pool = DestKeyWritePool::new(std::sync::Arc::clone(&target), edge_writer_workers());
        let sort_key_pool =
            SortKeyWritePool::new(std::sync::Arc::clone(&target), edge_writer_workers());
        let node_senders = node_pool.senders();
        let edge_senders = edge_pool.senders();
        let edge_errors = edge_pool.error();
        let dest_key_senders = dest_key_pool.senders();
        let dest_key_errors = dest_key_pool.error();
        let sort_key_senders = sort_key_pool.senders();
        let input_transfer = self.input_transfer.clone();
        let trimmed_strings = std::sync::Arc::clone(&self.trimmed_strings);
        let progress = self.progress.clone();
        for table in &tables {
            if let Some(cb) = &self.progress {
                cb(ProgressEvent {
                    phase: ProgressPhase::Encode,
                    table: table.name.clone(),
                    shard: None,
                    kind: ProgressKind::Start,
                    delta: 0,
                    total: Some(table.get_paths().len() as u64),
                    status: None,
});
            }
            let table = table.clone();
            let relationships = relationships.clone();
            let target = std::sync::Arc::clone(&target);
            let node_senders = std::sync::Arc::clone(&node_senders);
            let edge_senders = std::sync::Arc::clone(&edge_senders);
            let sort_key_senders = std::sync::Arc::clone(&sort_key_senders);
            let trimmed_strings = std::sync::Arc::clone(&trimmed_strings);
            let progress = progress.clone();
            let root = root.clone();
            let relation_defs = std::sync::Arc::clone(&relation_defs);
            let file_count = table.get_paths().len();
            for file_idx in 0..file_count {
                let table = table.clone();
                let relationships = relationships.clone();
                let target = std::sync::Arc::clone(&target);
                let node_senders = std::sync::Arc::clone(&node_senders);
                let edge_senders = std::sync::Arc::clone(&edge_senders);
                let dest_key_senders = std::sync::Arc::clone(&dest_key_senders);
                let dest_key_errors = std::sync::Arc::clone(&dest_key_errors);
                let sort_key_senders = std::sync::Arc::clone(&sort_key_senders);
                let trimmed_strings = std::sync::Arc::clone(&trimmed_strings);
                let progress = progress.clone();
                let root = root.clone();
                let relation_defs = std::sync::Arc::clone(&relation_defs);
                let input_transfer = input_transfer.clone();
                let tx = tx.clone();
                let edge_errors = std::sync::Arc::clone(&edge_errors);
                tasks.push(WorkItem {
                    bucket: None,
                    run: Box::new(move |_scratch| {
                        let mut key_failures = KeyFailureState::default();
                        if table.edge_table.is_some() {
                            let node_data_dropped = Self::build_temp_edge_file(
                                &table,
                                file_idx,
                                num_shards,
                                &target,
                                input_transfer.clone(),
                                &trimmed_strings,
                                &progress,
                                &mut key_failures,
                                &edge_senders,
                                &edge_errors,
                                &dest_key_senders,
                                &dest_key_errors,
                                &sort_key_senders,
                                relation_defs.as_ref(),
                            )?;
                            tx.send(Ok(EncodeResult {
                                table: table.name.clone(),
                                total_rows: 0,
                                node_data_dropped,
                                key_failures,
                            }))
                            .map_err(|e| anyhow!("encode output error: {e}"))?;
                            return Ok(());
                        }
                        let node = Self::build_temp_node_file(
                            &table,
                            &relationships,
                            &relation_defs,
                            file_idx,
                            num_shards,
                            &target,
                            &root,
                            input_transfer.clone(),
                            &trimmed_strings,
                            &progress,
                            &mut key_failures,
                            &node_senders,
                            &dest_key_senders,
                            &dest_key_errors,
                            &sort_key_senders,
                        )?;
                        tx.send(Ok(EncodeResult {
                            table: table.name.clone(),
                            total_rows: node.total_rows,
                            node_data_dropped: node.node_data_dropped,
                            key_failures,
                        }))
                        .map_err(|e| anyhow!("encode output error: {e}"))?;
                        Ok(())
                    }),
                });
            }
        }
        drop(tx);
        let task_count = tasks.len();
        let pool = self.make_pool::<()>();
        run_tasks(tasks, &pool)?;
        for _ in 0..task_count {
            let result = rx
                .recv()
                .map_err(|e| anyhow!("encode output error: {e}"))??;
            if let Some(total) = table_rows.get_mut(&result.table) {
                *total = total.saturating_add(result.total_rows);
            } else if result.total_rows > 0 {
                table_rows.insert(result.table.clone(), result.total_rows);
            }
            if result.node_data_dropped > 0 {
                let entry = self.dropped_rows_counts.entry(result.table.clone()).or_insert(0);
                *entry = entry.saturating_add(result.node_data_dropped);
            }
            self.merge_key_failures(result.key_failures);
            if let Some(cb) = &self.progress {
                let expected = tables
                    .iter()
                    .find(|t| t.name == result.table)
                    .map(|t| t.get_paths().len() as u64)
                    .unwrap_or(0);
                cb(ProgressEvent {
                    phase: ProgressPhase::Encode,
                    table: result.table.clone(),
                    shard: None,
                    kind: ProgressKind::Finish,
                    delta: 0,
                    total: Some(expected),
                    status: None,
});
            }
        }
        drop(node_senders);
        drop(edge_senders);
        drop(dest_key_senders);
        drop(sort_key_senders);
        let node_stats = node_pool.finish()?;
        let edge_stats = edge_pool.finish()?;
        let dest_key_stats = dest_key_pool.finish()?;
        let _sort_key_stats = sort_key_pool.finish()?;
        self.write_consolidated_meta(&node_stats, &edge_stats, &dest_key_stats)?;
        self.refresh_shard_rows_from_stats(&node_stats);
        self.table_rows = table_rows;
        Ok(())
    }


    fn refresh_shard_rows_from_stats(&mut self, stats: &NodeWriterStats) {
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            let mut shard_rows = vec![0u64; self.num_shards];
            for shard in 0..self.num_shards {
                let key = (table.name.clone(), shard);
                if let Some(count) = stats.node_rows.get(&key) {
                    shard_rows[shard] = *count;
                }
            }
            self.shard_rows.insert(table.name.clone(), shard_rows);
        }
    }

    fn write_consolidated_meta(
        &self,
        node_stats: &NodeWriterStats,
        edge_stats: &EdgeSrcWriterStats,
        dest_key_stats: &DestKeyWriterStats,
    ) -> Result<()> {
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            for shard in 0..self.num_shards {
                let node_key = (table.name.clone(), shard);
                let node_rows = node_stats.node_rows.get(&node_key).copied().unwrap_or(0);
                if node_rows == 0 {
                    ensure_temp_file(&self.target, &temp_node_data_path(&table.name, shard))?;
                }
                let max_len = node_stats
                    .node_data_max_len
                    .get(&(table.name.clone(), shard))
                    .copied()
                    .unwrap_or(0);
                write_temp_meta(
                    &self.target,
                    &temp_node_data_meta_path(&table.name, shard),
                    node_rows,
                    max_len,
                )?;
                if table.primary_key.is_some() {
                    let pkey_rows = node_stats.pkey_rows.get(&node_key).copied().unwrap_or(0);
                    if pkey_rows == 0 {
                        ensure_temp_file(&self.target, &temp_pkey_path(&table.name, shard))?;
                    }
                    write_temp_meta(
                        &self.target,
                        &temp_pkey_meta_path(&table.name, shard),
                        pkey_rows,
                        0,
                    )?;
                }
                for rel in self
                    .relationships
                    .iter()
                    .filter(|rel| rel.src_table == table.name)
                {
                    let rel_key = (table.name.clone(), rel.name_forward.clone(), shard);
                    let rows = node_stats.fkey_rows.get(&rel_key).copied().unwrap_or(0);
                    if rows == 0 {
                        ensure_temp_file(
                            &self.target,
                            &temp_fkey_path(&table.name, &rel.name_forward, shard),
                        )?;
                        ensure_fkey_bitmap_file(
                            &self.target,
                            &temp_fkey_bitmap_path(&table.name, &rel.name_forward, shard),
                        )?;
                    }
                    write_temp_meta(
                        &self.target,
                        &temp_fkey_meta_path(&table.name, &rel.name_forward, shard),
                        rows,
                        0,
                    )?;
                    let dst_key = (table.name.clone(), rel.name_forward.clone(), shard);
                    let dst_rows = dest_key_stats
                        .node_fkey_rows
                        .get(&dst_key)
                        .copied()
                        .unwrap_or(0);
                    if dst_rows == 0 {
                        ensure_temp_file(
                            &self.target,
                            &temp_fkey_dst_path(&table.name, &rel.name_forward, shard),
                        )?;
                    }
                    write_temp_meta(
                        &self.target,
                        &temp_fkey_dst_meta_path(&table.name, &rel.name_forward, shard),
                        dst_rows,
                        0,
                    )?;
                }
            }
        }

        for table in self.source.tables.iter().filter(|t| t.edge_table.is_some()) {
            let forward_name = format!("edge_{}_src_to_dst", table.name);
            for shard in 0..self.num_shards {
                let rows = edge_stats
                    .rows
                    .get(&(forward_name.clone(), shard))
                    .copied()
                    .unwrap_or(0);
                if rows == 0 {
                    ensure_temp_file(&self.target, &temp_edge_path(&forward_name, shard))?;
                }
                write_temp_meta(
                    &self.target,
                    &temp_edge_meta_path(&forward_name, shard),
                    rows,
                    0,
                )?;
                for kind in [EdgeKeyKind::Src, EdgeKeyKind::Dst] {
                    let key_rows = dest_key_stats
                        .edge_key_rows
                        .get(&(table.name.clone(), kind, shard))
                        .copied()
                        .unwrap_or(0);
                    if key_rows == 0 {
                        ensure_temp_file(
                            &self.target,
                            &temp_edge_key_path(&table.name, kind, shard),
                        )?;
                    }
                    write_temp_meta(
                        &self.target,
                        &temp_edge_key_meta_path(&table.name, kind, shard),
                        key_rows,
                        0,
                    )?;
                }
            }
        }
        for rel in &self.relation_defs {
            if let Some(cfg) = &rel.sort_key {
                let kinds = [
                    (SortKeyComponentKind::Timestamp, cfg.timestamp.is_some()),
                    (SortKeyComponentKind::Weight, cfg.weight.is_some()),
                ];
                for (kind, enabled) in kinds {
                    if !enabled {
                        continue;
                    }
                    for shard in 0..self.num_shards {
                        let path = temp_sort_key_component_path(&rel.name, kind, shard);
                        ensure_sort_key_file(&self.target, &path)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn build_temp_node_file(
        table: &TableSpec,
        relationships: &[RelationshipRuntime],
        relation_defs: &[RelationDef],
        file_idx: usize,
        num_shards: usize,
        _target: &std::sync::Arc<OutputTarget>,
        root: &Path,
        input_transfer: Option<CloudTransfer>,
        trimmed_strings: &std::sync::Arc<AtomicU64>,
        progress: &Option<ProgressCallback>,
        key_failures: &mut KeyFailureState,
        senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>>,
        dest_key_senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
        dest_key_errors: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        sort_key_senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>>,
    ) -> Result<TempNodeFileResult> {
        let pk_col = table.primary_key.as_ref().and_then(|pk| column_spec(table, pk).ok());
        let pk_idx = table
            .primary_key
            .as_ref()
            .and_then(|pk| column_index(table, pk).ok());
        let relations: Vec<RelationshipRuntime> = relationships
            .iter()
            .filter(|rel| rel.src_table == table.name)
            .cloned()
            .collect();
        let mut rel_names: Vec<std::sync::Arc<str>> = Vec::new();
        let mut rel_specs = Vec::new();
        for rel in &relations {
            let idx = column_index(table, &rel.src_column)?;
            let rel_name = std::sync::Arc::<str>::from(rel.name_forward.clone());
            let rel_pos = rel_names.len();
            rel_names.push(rel_name.clone());
            rel_specs.push((rel_name, idx, rel_pos));
        }
        let mut total_rows = 0u64;
        let row_access_hints = build_row_access_hints(table);
        let stats = RowAccessStats::new(Some(trimmed_strings.as_ref()));
        let mut access = RowAccess::new(row_access_hints, stats);
        let ctx_path = root.join(temp_node_context_path(&table.name));
        let ctx_bytes =
            fs::read(&ctx_path).with_context(|| format!("read {}", ctx_path.display()))?;
        let node_context = diskgraph_rowpack::RowPackContext::from_bytes(&ctx_bytes)?;
        let node_encoder = Some(RowPackEncoder::new(node_context)?);
        let mut key_bufs: Vec<Vec<u8>> = vec![Vec::new(); table.columns.len()];
        let mut f16_buf: Vec<f16> = Vec::new();
        let mut node_data_dropped = 0u64;
        let has_pkey = pk_col.is_some() && pk_idx.is_some();
        let rel_names = std::sync::Arc::new(rel_names);
        let buffer_limit = edge_batch_buffer_limit(num_shards);
        let mut batcher = NodeBatcher::new(
            std::sync::Arc::<str>::from(table.name.clone()),
            std::sync::Arc::clone(&rel_names),
            has_pkey,
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(senders),
        );
        let mut dest_fkey_batcher = DestFkeyBatcher::new(
            std::sync::Arc::<str>::from(table.name.clone()),
            std::sync::Arc::clone(&rel_names),
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(dest_key_senders),
            std::sync::Arc::clone(dest_key_errors),
        );
        let mut sort_key_batchers = build_sort_key_batchers_for_node_table(
            table,
            relation_defs,
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(sort_key_senders),
        )?;

        let create_time_info = table.create_time.as_ref().and_then(|ct| {
            let idx = column_index(table, ct).ok()?;
            let spec = column_spec(table, ct).ok()?;
            Some((idx, spec.ingested_dtype.clone()))
        });

        let path = table
            .get_paths()
            .get(file_idx)
            .ok_or_else(|| anyhow!("missing input file {}", file_idx))?;
        if !is_cloud_path(path) && !Path::new(path).exists() {
            return Err(anyhow!("input path not found {}", path));
        }
        let mut row_id = 0u64;

        macro_rules! process_node_batch {
            ($batch:expr) => {
                let timestamp_idx = create_time_info.as_ref().map(|(idx, _)| *idx);
                let validity_array = timestamp_idx.map(|idx| access.array(idx).unwrap().clone());
                let pk_array = pk_idx.and_then(|idx| access.array(idx).ok());
                let precomputed = precompute_shards(pk_array, pk_col, num_shards);

                let mut fkeys_pre = Vec::with_capacity(rel_specs.len());
                for (_, idx, _) in &rel_specs {
                    let col_spec = &table.columns[*idx];
                    let array = access.array(*idx).unwrap();
                    let kind = edge_fast_path_for_array(col_spec, array);
                    let should_optimize = match kind {
                        Some(EdgeKeyFastKind::Int64ToInt32)
                        | Some(EdgeKeyFastKind::StringToInt64)
                        | Some(EdgeKeyFastKind::StringUuid)
                        | Some(EdgeKeyFastKind::StringUuid256)
                        | Some(EdgeKeyFastKind::StringIp) => true,
                        _ => false,
                    };
                    if should_optimize {
                        fkeys_pre.push(extract_batch_keys(array, kind.unwrap()).ok());
                    } else {
                        fkeys_pre.push(None);
                    }
                }

                for row in 0..$batch.num_rows() {
                    access.set_row(row);
                    total_rows += 1;
                    let timestamp_valid = validity_array.as_ref().map(|a| a.is_valid(row)).unwrap_or(true);
                    if !timestamp_valid && pk_idx.is_none() {
                        node_data_dropped += 1;
                    }
                    let shard_override = precomputed.as_ref().and_then(|p| p.shards[row]);
                    let pk_pre = precomputed
                        .as_ref()
                        .and_then(|p| p.keys.as_ref())
                        .and_then(|k| k[row].as_ref());

                    Self::write_node_row_from_access(
                        table,
                        pk_idx,
                        &rel_specs,
                        &fkeys_pre,
                        row,
                        &mut access,
                        &node_encoder,
                        &mut batcher,
                        &mut dest_fkey_batcher,
                        &mut sort_key_batchers,
                        &mut row_id,
                        pk_col,
                        pk_pre,
                        &mut key_bufs,
                        &mut f16_buf,
                        &mut node_data_dropped,
                        num_shards,
                        key_failures,
                        timestamp_valid,
                        shard_override,
                    )?;
                }
            };
        }

        match table.format() {
            TableFormat::Parquet => {
                if let Some(transfer) = &input_transfer {
                    if is_cloud_path(path) {
                        let pooled = download_cloud_bytes(transfer, path)?;
                        let bytes = pooled.bytes();
                        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
                        let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                        for batch in reader {
                            let batch = batch?;
                            let mut cols = Vec::with_capacity(table.columns.len());
                            for col in &table.columns {
                                let idx = batch
                                    .schema()
                                    .index_of(&col.name)
                                    .context("missing column in batch")?;
                                cols.push(batch.column(idx).clone());
                            }
                            access.set_cols(&cols);
                            process_node_batch!(batch);
                        }
                        drop(pooled);
                    } else {
                        let file = fs::File::open(path)
                            .with_context(|| format!("open {}", path))?;
                        advise_file_willneed(&file);
                        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                        let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                        for batch in reader {
                            let batch = batch?;
                            let mut cols = Vec::with_capacity(table.columns.len());
                            for col in &table.columns {
                                let idx = batch
                                    .schema()
                                    .index_of(&col.name)
                                    .context("missing column in batch")?;
                                cols.push(batch.column(idx).clone());
                            }
                            access.set_cols(&cols);
                            process_node_batch!(batch);
                        }
                    }
                } else {
                    let file = fs::File::open(path)
                        .with_context(|| format!("open {}", path))?;
                    advise_file_willneed(&file);
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                    for batch in reader {
                        let batch = batch?;
                        let mut cols = Vec::with_capacity(table.columns.len());
                        for col in &table.columns {
                            let idx = batch
                                .schema()
                                .index_of(&col.name)
                                .context("missing column in batch")?;
                            cols.push(batch.column(idx).clone());
                        }
                        access.set_cols(&cols);
                        process_node_batch!(batch);
                    }
                }
            }
            TableFormat::Csv => {
                let schema = table_schema(table)?;
                if let Some(transfer) = &input_transfer {
                    if is_cloud_path(path) {
                        let pooled = download_cloud_bytes(transfer, path)?;
                        let bytes = pooled.bytes();
                        let cursor = Cursor::new(bytes);
                        let builder =
                            arrow_csv::ReaderBuilder::new(Arc::new(schema)).with_header(true);
                        let reader = builder.build(cursor)?;
                        for batch in reader {
                            let batch = batch?;
                            let mut cols = Vec::with_capacity(table.columns.len());
                            for col in &table.columns {
                                let idx = batch
                                    .schema()
                                    .index_of(&col.name)
                                    .context("missing column in batch")?;
                                cols.push(batch.column(idx).clone());
                            }
                            access.set_cols(&cols);
                            process_node_batch!(batch);
                        }
                        drop(pooled);
                    } else {
                        let file = fs::File::open(path)
                            .with_context(|| format!("open {}", path))?;
                        advise_file_willneed(&file);
                        let builder = arrow_csv::ReaderBuilder::new(Arc::new(schema))
                            .with_header(true);
                        let reader = builder.build(file)?;
                        for batch in reader {
                            let batch = batch?;
                            let mut cols = Vec::with_capacity(table.columns.len());
                            for col in &table.columns {
                                let idx = batch
                                    .schema()
                                    .index_of(&col.name)
                                    .context("missing column in batch")?;
                                cols.push(batch.column(idx).clone());
                            }
                            access.set_cols(&cols);
                            process_node_batch!(batch);
                        }
                    }
                } else {
                    let file = fs::File::open(path)
                        .with_context(|| format!("open {}", path))?;
                    advise_file_willneed(&file);
                    let builder =
                        arrow_csv::ReaderBuilder::new(Arc::new(schema)).with_header(true);
                    let reader = builder.build(file)?;
                    for batch in reader {
                        let batch = batch?;
                        let mut cols = Vec::with_capacity(table.columns.len());
                        for col in &table.columns {
                            let idx = batch
                                .schema()
                                .index_of(&col.name)
                                .context("missing column in batch")?;
                            cols.push(batch.column(idx).clone());
                        }
                        access.set_cols(&cols);
                        process_node_batch!(batch);
                    }
                }
            }
        }
        batcher.flush_all()?;
        dest_fkey_batcher.flush_all()?;
        for batcher in &mut sort_key_batchers {
            batcher.flush_all()?;
        }
        if let Some(cb) = progress {
            cb(ProgressEvent {
                phase: ProgressPhase::Encode,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Advance,
                delta: 1,
                total: Some(table.get_paths().len() as u64),
                status: None,
});
        }
        Ok(TempNodeFileResult {
            total_rows,
            node_data_dropped,
        })
    }

    fn write_node_row_from_access(
        table: &TableSpec,
        pk_idx: Option<usize>,
        rel_specs: &[(std::sync::Arc<str>, usize, usize)],
        fkeys_pre: &[Option<Vec<Option<Key>>>],
        row: usize,
        access: &mut RowAccess<'_>,
        node_encoder: &Option<RowPackEncoder>,
        batcher: &mut NodeBatcher,
        dest_fkey_batcher: &mut DestFkeyBatcher,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
        row_id: &mut u64,
        pk_col: Option<&ColumnSpec>,
        pk_pre: Option<&Key>,
        key_bufs: &mut Vec<Vec<u8>>,
        f16_buf: &mut Vec<f16>,
        node_data_dropped: &mut u64,
        num_shards: usize,
        key_failures: &mut KeyFailureState,
        timestamp_valid: bool,
        shard_override: Option<usize>,
    ) -> Result<()> {
        for buf in key_bufs.iter_mut() {
            buf.clear();
        }
        let mut has_pkey = false;
        let shard = if let (Some(idx), Some(pk_col)) = (pk_idx, pk_col) {
            let buf = key_bufs
                .get_mut(idx)
                .ok_or_else(|| anyhow!("missing key buffer"))?;
            let res = if let Some(key) = pk_pre {
                write_key_to_buf(buf, key).map_err(|_| KeyParseError::TypeMismatch)
            } else {
                key_from_access_into_buf(access, idx, pk_col, buf)
            };

            match res {
                Ok(_) => {
                    has_pkey = true;
                    shard_override.unwrap_or_else(|| shard_for_key(buf, num_shards))
                }
                Err(_) => {
                    let context = format!("pkey:{}.{}", table.name, pk_col.name);
                    record_key_parse_failure_sample(
                        &mut key_failures.parse_failures,
                        &mut key_failures.samples,
                        &mut key_failures.by_context,
                        &context,
                        sample_from_access(access, idx),
                    );
                    return Ok(());
                }
            }
        } else {
            (*row_id as usize) % num_shards
        };
        for (i, (_rel_name, rel_idx, _rel_pos)) in rel_specs.iter().enumerate() {
            let col = table
                .columns
                .get(*rel_idx)
                .ok_or_else(|| anyhow!("missing fk column"))?;
            let buf = key_bufs
                .get_mut(*rel_idx)
                .ok_or_else(|| anyhow!("missing key buffer"))?;

            let res = if let Some(keys) = fkeys_pre.get(i).and_then(|k| k.as_ref()) {
                if let Some(Some(key)) = keys.get(row) {
                    write_key_to_buf(buf, key).map_err(|_| KeyParseError::TypeMismatch)
                } else {
                    // Fallback to RowAccess for None values to ensure correct error reporting
                    // (extract_batch_keys uses None for both nulls and parse errors).
                    key_from_access_into_buf(access, *rel_idx, col, buf)
                }
            } else {
                key_from_access_into_buf(access, *rel_idx, col, buf)
            };

            if res.is_err() {
                let context = format!("fkey:{}.{}", table.name, col.name);
                record_key_parse_failure_sample(
                    &mut key_failures.parse_failures,
                    &mut key_failures.samples,
                    &mut key_failures.by_context,
                    &context,
                    sample_from_access(access, *rel_idx),
                );
                buf.clear();
                continue;
            }
        }
        let encoder = node_encoder
            .as_ref()
            .ok_or_else(|| anyhow!("missing node encoder"))?;
        let node_tag_start;
        let node_bool_start;
        let node_data_start;
        {
            let sink = batcher.node_sink(shard)?;
            node_tag_start = sink.tags.len();
            node_bool_start = sink.bools.len();
            node_data_start = sink.data.len();
            let lens = encoder.encode_row_from_access_into_buffers(
                access,
                sink.tags,
                sink.bools,
                sink.data,
                f16_buf,
                key_bufs,
            )?;
            let row_len = lens
                .tag_bytes
                .saturating_add(lens.bool_bytes)
                .saturating_add(lens.data_bytes);
            if row_len > MAX_NODE_DATA_LEN {
                sink.tags.truncate(node_tag_start);
                sink.bools.truncate(node_bool_start);
                sink.data.truncate(node_data_start);
                *node_data_dropped = node_data_dropped.saturating_add(1);
                batcher.push_empty_node(shard)?;
            } else {
                batcher.push_node_row(shard, lens)?;
            }
        }
        if has_pkey {
            let buf = key_bufs
                .get(pk_idx.unwrap())
                .ok_or_else(|| anyhow!("missing pkey buffer"))?;
            batcher.push_pkey(shard, buf, timestamp_valid)?;
        }
        for sort_key in sort_key_batchers.iter_mut() {
            sort_key.push_row(shard, access)?;
        }
        for (_rel_name, rel_idx, rel_pos) in rel_specs {
            if let Some(buf) = key_bufs.get(*rel_idx) {
                if buf.is_empty() {
                    batcher.push_fkey_empty(shard, *rel_pos)?;
                } else {
                    batcher.push_fkey(shard, *rel_pos, buf, timestamp_valid)?;
                    let dst_shard = shard_for_key(buf, num_shards);
                    dest_fkey_batcher.push(dst_shard, *rel_pos, buf)?;
                }
            }
        }
        batcher.maybe_flush(shard)?;
        *row_id = row_id.saturating_add(1);
        Ok(())
    }

    fn build_temp_edge_file(
        table: &TableSpec,
        file_idx: usize,
        num_shards: usize,
        _target: &std::sync::Arc<OutputTarget>,
        input_transfer: Option<CloudTransfer>,
        trimmed_strings: &std::sync::Arc<AtomicU64>,
        progress: &Option<ProgressCallback>,
        key_failures: &mut KeyFailureState,
        senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>>,
        error: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        dest_key_senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
        dest_key_errors: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        sort_key_senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>>,
        relation_defs: &[RelationDef],
    ) -> Result<u64> {
        let edge = table
            .edge_table
            .as_ref()
            .ok_or_else(|| anyhow!("missing edge table spec for {}", table.name))?;
        let src_fk_idx = column_index(table, &edge.src_fk.edge_column)?;
        let dst_fk_idx = column_index(table, &edge.dst_fk.edge_column)?;
        let src_fk_col = column_spec(table, &edge.src_fk.edge_column)?;
        let dst_fk_col = column_spec(table, &edge.dst_fk.edge_column)?;
        let src_context = format!("edge_fk:{}.{}", table.name, edge.src_fk.edge_column);
        let dst_context = format!("edge_fk:{}.{}", table.name, edge.dst_fk.edge_column);
        let schema = table_schema(table)?;
        let rel_name = std::sync::Arc::<str>::from(format!("edge_{}_src_to_dst", table.name));
        let buffer_limit = edge_batch_buffer_limit(num_shards);
        let mut batcher = EdgeSrcBatcher::new(
            std::sync::Arc::clone(&rel_name),
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(senders),
            std::sync::Arc::clone(error),
        );
        let mut edge_key_batcher = EdgeKeyBatcher::new(
            std::sync::Arc::<str>::from(table.name.clone()),
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(dest_key_senders),
            std::sync::Arc::clone(dest_key_errors),
        );
        let mut sort_key_batchers = build_sort_key_batchers_for_edge_table(
            table,
            relation_defs,
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(sort_key_senders),
        )?;

        let create_time_info = table.create_time.as_ref().and_then(|ct| {
            let idx = column_index(table, ct).ok()?;
            let spec = column_spec(table, ct).ok()?;
            Some((idx, spec.ingested_dtype.clone()))
        });

        let path = table
            .get_paths()
            .get(file_idx)
            .ok_or_else(|| anyhow!("missing input file {}", file_idx))?;
        if !is_cloud_path(path) && !Path::new(path).exists() {
            return Err(anyhow!("input path not found {}", path));
        }
        let row_access_hints = build_row_access_hints(table);
        let stats = RowAccessStats::new(Some(trimmed_strings.as_ref()));
        let mut access = RowAccess::new(row_access_hints, stats);

        macro_rules! process_edge_batch {
            ($batch:expr, $cols:expr, $src_fast:expr, $dst_fast:expr, $src_key_buf:expr, $dst_key_buf:expr, $forward_row:expr) => {
                let timestamp_idx = create_time_info.as_ref().map(|(idx, _)| *idx);
                let validity_array = timestamp_idx.map(|idx| access.array(idx).unwrap().clone());

                let src_keys_pre = $src_fast
                    .map(|kind| extract_batch_keys(&$cols[src_fk_idx], kind))
                    .transpose()?;
                let dst_keys_pre = $dst_fast
                    .map(|kind| extract_batch_keys(&$cols[dst_fk_idx], kind))
                    .transpose()?;

                for row in 0..$batch.num_rows() {
                    access.set_row(row);
                    let timestamp_valid = validity_array.as_ref().map(|a| a.is_valid(row)).unwrap_or(true);
                    let src_key_pre = src_keys_pre.as_ref().and_then(|v| v[row].as_ref());
                    let dst_key_pre = dst_keys_pre.as_ref().and_then(|v| v[row].as_ref());

                    Self::write_edge_row_from_access(
                        src_fk_idx,
                        dst_fk_idx,
                        src_fk_col,
                        dst_fk_col,
                        &src_context,
                        &dst_context,
                        $cols,
                        row,
                        src_key_pre,
                        dst_key_pre,
                        &mut access,
                        $src_key_buf,
                        $dst_key_buf,
                        $forward_row,
                        &mut batcher,
                        &mut edge_key_batcher,
                        &mut sort_key_batchers,
                        num_shards,
                        key_failures,
                        timestamp_valid,
                    )?;
                }
            };
        }

        match table.format() {
            TableFormat::Parquet => {
                if let Some(transfer) = &input_transfer {
                    if is_cloud_path(path) {
                        let pooled = download_cloud_bytes(transfer, path)?;
                        let bytes = pooled.bytes();
                        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
                        let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                        for batch in reader {
                            let batch = batch?;
                            let mut cols = Vec::with_capacity(table.columns.len());
                            for col in &table.columns {
                                let idx = batch
                                    .schema()
                                    .index_of(&col.name)
                                    .context("missing column in batch")?;
                                cols.push(batch.column(idx).clone());
                            }
                            access.set_cols(&cols);
                            let src_fast = edge_fast_path_for_array(src_fk_col, &cols[src_fk_idx]);
                            let dst_fast = edge_fast_path_for_array(dst_fk_col, &cols[dst_fk_idx]);
                            let mut src_key_buf = Vec::new();
                            let mut dst_key_buf = Vec::new();
                            let mut forward_row = Vec::new();
                            process_edge_batch!(
                                batch,
                                &cols,
                                src_fast,
                                dst_fast,
                                &mut src_key_buf,
                                &mut dst_key_buf,
                                &mut forward_row
                            );
                        }
                        drop(pooled);
                    } else {
                        let file = fs::File::open(path)
                            .with_context(|| format!("open {}", path))?;
                        advise_file_willneed(&file);
                        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                        let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                        for batch in reader {
                            let batch = batch?;
                            let mut cols = Vec::with_capacity(table.columns.len());
                            for col in &table.columns {
                                let idx = batch
                                    .schema()
                                    .index_of(&col.name)
                                    .context("missing column in batch")?;
                                cols.push(batch.column(idx).clone());
                            }
                            access.set_cols(&cols);
                            let src_fast = edge_fast_path_for_array(src_fk_col, &cols[src_fk_idx]);
                            let dst_fast = edge_fast_path_for_array(dst_fk_col, &cols[dst_fk_idx]);
                            let mut src_key_buf = Vec::new();
                            let mut dst_key_buf = Vec::new();
                            let mut forward_row = Vec::new();
                            process_edge_batch!(
                                batch,
                                &cols,
                                src_fast,
                                dst_fast,
                                &mut src_key_buf,
                                &mut dst_key_buf,
                                &mut forward_row
                            );
                        }
                    }
                } else {
                    let file = fs::File::open(path)
                        .with_context(|| format!("open {}", path))?;
                    advise_file_willneed(&file);
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                    for batch in reader {
                        let batch = batch?;
                        let mut cols = Vec::with_capacity(table.columns.len());
                        for col in &table.columns {
                            let idx = batch
                                .schema()
                                .index_of(&col.name)
                                .context("missing column in batch")?;
                            cols.push(batch.column(idx).clone());
                        }
                        access.set_cols(&cols);
                        let src_fast = edge_fast_path_for_array(src_fk_col, &cols[src_fk_idx]);
                        let dst_fast = edge_fast_path_for_array(dst_fk_col, &cols[dst_fk_idx]);
                            let mut src_key_buf = Vec::new();
                            let mut dst_key_buf = Vec::new();
                            let mut forward_row = Vec::new();
                            process_edge_batch!(
                                batch,
                                &cols,
                                src_fast,
                                dst_fast,
                                &mut src_key_buf,
                                &mut dst_key_buf,
                                &mut forward_row
                            );
                        }
                    }
            },
            TableFormat::Csv => {
                let builder = arrow_csv::ReaderBuilder::new(Arc::new(schema.clone()))
                    .with_header(true);
                if let Some(transfer) = &input_transfer {
                    if is_cloud_path(path) {
                        let pooled = download_cloud_bytes(transfer, path)?;
                        let bytes = pooled.bytes();
                        let cursor = Cursor::new(bytes);
                        let reader = builder.build(cursor)?;
                        for batch in reader {
                            let batch = batch?;
                            let mut cols = Vec::with_capacity(table.columns.len());
                            for col in &table.columns {
                                let idx = batch
                                    .schema()
                                    .index_of(&col.name)
                                    .context("missing column in batch")?;
                                cols.push(batch.column(idx).clone());
                            }
                            access.set_cols(&cols);
                            let src_fast = edge_fast_path_for_array(src_fk_col, &cols[src_fk_idx]);
                            let dst_fast = edge_fast_path_for_array(dst_fk_col, &cols[dst_fk_idx]);
                            let mut src_key_buf = Vec::new();
                            let mut dst_key_buf = Vec::new();
                            let mut forward_row = Vec::new();
                            process_edge_batch!(
                                batch,
                                &cols,
                                src_fast,
                                dst_fast,
                                &mut src_key_buf,
                                &mut dst_key_buf,
                                &mut forward_row
                            );
                        }
                        drop(pooled);
                    } else {
                        let file = fs::File::open(path)
                            .with_context(|| format!("open {}", path))?;
                        advise_file_willneed(&file);
                        let reader = builder.build(file)?;
                        for batch in reader {
                            let batch = batch?;
                            let mut cols = Vec::with_capacity(table.columns.len());
                            for col in &table.columns {
                                let idx = batch
                                    .schema()
                                    .index_of(&col.name)
                                    .context("missing column in batch")?;
                                cols.push(batch.column(idx).clone());
                            }
                            access.set_cols(&cols);
                            let src_fast = edge_fast_path_for_array(src_fk_col, &cols[src_fk_idx]);
                            let dst_fast = edge_fast_path_for_array(dst_fk_col, &cols[dst_fk_idx]);
                            let mut src_key_buf = Vec::new();
                            let mut dst_key_buf = Vec::new();
                            let mut forward_row = Vec::new();
                            process_edge_batch!(
                                batch,
                                &cols,
                                src_fast,
                                dst_fast,
                                &mut src_key_buf,
                                &mut dst_key_buf,
                                &mut forward_row
                            );
                        }
                    }
                } else {
                    let file = fs::File::open(path)
                        .with_context(|| format!("open {}", path))?;
                    advise_file_willneed(&file);
                    let reader = builder.build(file)?;
                    for batch in reader {
                        let batch = batch?;
                        let mut cols = Vec::with_capacity(table.columns.len());
                        for col in &table.columns {
                            let idx = batch
                                .schema()
                                .index_of(&col.name)
                                .context("missing column in batch")?;
                            cols.push(batch.column(idx).clone());
                        }
                        access.set_cols(&cols);
                        let src_fast = edge_fast_path_for_array(src_fk_col, &cols[src_fk_idx]);
                        let dst_fast = edge_fast_path_for_array(dst_fk_col, &cols[dst_fk_idx]);
                        let mut src_key_buf = Vec::new();
                        let mut dst_key_buf = Vec::new();
                        let mut forward_row = Vec::new();
                        process_edge_batch!(
                            batch,
                            &cols,
                            src_fast,
                            dst_fast,
                            &mut src_key_buf,
                            &mut dst_key_buf,
                            &mut forward_row
                        );
                    }
                }
            }
        }
        batcher.flush_all()?;
        edge_key_batcher.flush_all()?;
        for batcher in &mut sort_key_batchers {
            batcher.flush_all()?;
        }
        if let Some(cb) = progress {
            cb(ProgressEvent {
                phase: ProgressPhase::Encode,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Advance,
                delta: 1,
                total: Some(table.get_paths().len() as u64),
                status: None,
});
        }
        Ok(0)
    }


    fn write_edge_row_from_access(
        src_idx: usize,
        dst_idx: usize,
        src_col: &ColumnSpec,
        dst_col: &ColumnSpec,
        src_context: &str,
        dst_context: &str,
        _cols: &[ArrayRef],
        _row: usize,
        src_key_pre: Option<&Key>,
        dst_key_pre: Option<&Key>,
        access: &mut RowAccess<'_>,
        src_key_buf: &mut Vec<u8>,
        dst_key_buf: &mut Vec<u8>,
        forward_row: &mut Vec<u8>,
        batcher: &mut EdgeSrcBatcher,
        edge_key_batcher: &mut EdgeKeyBatcher,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
        num_shards: usize,
        key_failures: &mut KeyFailureState,
        timestamp_valid: bool,
    ) -> Result<()> {
        src_key_buf.clear();
        dst_key_buf.clear();
        if let Some(key) = src_key_pre {
            if write_key_to_buf(src_key_buf, key).is_err() {
                record_key_parse_failure_sample(
                    &mut key_failures.parse_failures,
                    &mut key_failures.samples,
                    &mut key_failures.by_context,
                    src_context,
                    sample_from_access(access, src_idx),
                );
                return Ok(());
            }
        } else if key_from_access_into_buf(access, src_idx, src_col, src_key_buf).is_err() {
            record_key_parse_failure_sample(
                &mut key_failures.parse_failures,
                &mut key_failures.samples,
                &mut key_failures.by_context,
                src_context,
                sample_from_access(access, src_idx),
            );
            return Ok(());
        }

        if let Some(key) = dst_key_pre {
            if write_key_to_buf(dst_key_buf, key).is_err() {
                record_key_parse_failure_sample(
                    &mut key_failures.parse_failures,
                    &mut key_failures.samples,
                    &mut key_failures.by_context,
                    dst_context,
                    sample_from_access(access, dst_idx),
                );
                return Ok(());
            }
        } else if key_from_access_into_buf(access, dst_idx, dst_col, dst_key_buf).is_err() {
            record_key_parse_failure_sample(
                &mut key_failures.parse_failures,
                &mut key_failures.samples,
                &mut key_failures.by_context,
                dst_context,
                sample_from_access(access, dst_idx),
            );
            return Ok(());
        }
        let forward_shard = shard_for_key(src_key_buf, num_shards);
        let dst_shard = shard_for_key(dst_key_buf, num_shards);
        forward_row.clear();
        forward_row.reserve(src_key_buf.len().saturating_add(dst_key_buf.len()).saturating_add(4));
        append_key_bytes_with_len_prefix(forward_row, src_key_buf, src_col)?;
        append_key_bytes_with_len_prefix(forward_row, dst_key_buf, dst_col)?;
        batcher.push(forward_shard, forward_row, timestamp_valid)?;
        edge_key_batcher.push_src(forward_shard, src_key_buf)?;
        edge_key_batcher.push_dst(dst_shard, dst_key_buf)?;
        for batcher in sort_key_batchers.iter_mut() {
            batcher.push_row(forward_shard, access)?;
        }
        Ok(())
    }

    fn build_sharded_pk_maps(&mut self) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        let pk_map_budget = total_memory_bytes().map(|total| {
            let budget = (total as f64 * PK_MAP_MEM_BUDGET_FRAC) as u64;
            std::sync::Arc::new(MemoryBudget::new(budget))
        });
        let duplicate_counts = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let duplicate_samples = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let pk_null_ts_counts = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let key_failures = std::sync::Arc::new(std::sync::Mutex::new(KeyFailureState::default()));
        let relationships = self.relationships.clone();
        let source = self.source.clone();
        let mut tasks = Vec::new();
        let mut pk_hash_handles: HashMap<
            String,
            std::sync::Arc<std::sync::Mutex<Vec<Option<PkHashPaths>>>>,
        > =
            HashMap::new();
        for table in &self.source.tables {
            if table.edge_table.is_some() || table.primary_key.is_none() {
                continue;
            }
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::PkMap,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Start,
                delta: 0,
                total: Some(self.num_shards as u64),
                status: None,
            });
            let pk_name = table.primary_key.clone().unwrap();
            let pk_col = column_spec(table, &pk_name)?;
            let pk_hash_paths = std::sync::Arc::new(std::sync::Mutex::new(vec![
                None;
                self.num_shards
            ]));
            pk_hash_handles.insert(table.name.clone(), std::sync::Arc::clone(&pk_hash_paths));
            let num_shards = self.num_shards as u64;
            for shard in 0..self.num_shards {
                let target = std::sync::Arc::clone(&self.target);
                let table_name = table.name.clone();
                let pk_col = pk_col.clone();
                let progress = self.progress.clone();
                let pk_hash_paths = std::sync::Arc::clone(&pk_hash_paths);
                let duplicate_counts = std::sync::Arc::clone(&duplicate_counts);
                let duplicate_samples = std::sync::Arc::clone(&duplicate_samples);
                let pk_null_ts_counts = std::sync::Arc::clone(&pk_null_ts_counts);
                let key_failures_shared = std::sync::Arc::clone(&key_failures);
                let root = root.clone();
                let pk_map_budget = pk_map_budget.clone();
                let relationships = relationships.clone();
                let source = source.clone();
                tasks.push(WorkItem {
                    bucket: None,
                    run: Box::new(move |scratch: &mut PkMapScratch| {
                        let key_type = key_type_for_column(&pk_col)?;
                        let meta_path = root.join(temp_pkey_meta_path(&table_name, shard));
                        let meta = read_temp_meta(&meta_path)?;
                        let total_rows = meta.row_count;
                        let reserve_rows = total_rows
                            .saturating_mul(110)
                            .saturating_add(99)
                            / 100;
                        let reserve_rows_usize =
                            usize::try_from(reserve_rows).unwrap_or(usize::MAX);
                        let data_path = root.join(temp_pkey_path(&table_name, shard));
                        let mut fkey_inputs: Vec<(PathBuf, PathBuf, ColumnSpec)> = Vec::new();
                        let mut edge_inputs: Vec<(PathBuf, PathBuf, ColumnSpec)> = Vec::new();
                        for rel in relationships.iter().filter(|rel| rel.dst_table == table_name) {
                            let src_table = table_by_name(&source, &rel.src_table)?;
                            let src_col = column_spec(src_table, &rel.src_column)?;
                            fkey_inputs.push((
                                root.join(temp_fkey_dst_path(&rel.src_table, &rel.name_forward, shard)),
                                root.join(temp_fkey_dst_meta_path(&rel.src_table, &rel.name_forward, shard)),
                                src_col.clone(),
                            ));
                        }
                        for table in source.tables.iter().filter(|t| t.edge_table.is_some()) {
                            let edge = table.edge_table.as_ref().unwrap();
                            if edge.src_fk.ref_table == table_name {
                                let src_col = column_spec(table, &edge.src_fk.edge_column)?;
                                edge_inputs.push((
                                    root.join(temp_edge_key_path(&table.name, EdgeKeyKind::Src, shard)),
                                    root.join(temp_edge_key_meta_path(&table.name, EdgeKeyKind::Src, shard)),
                                    src_col.clone(),
                                ));
                            }
                            if edge.dst_fk.ref_table == table_name {
                                let dst_col = column_spec(table, &edge.dst_fk.edge_column)?;
                                edge_inputs.push((
                                    root.join(temp_edge_key_path(&table.name, EdgeKeyKind::Dst, shard)),
                                    root.join(temp_edge_key_meta_path(&table.name, EdgeKeyKind::Dst, shard)),
                                    dst_col.clone(),
                                ));
                            }
                        }
                        let mut union_rows = total_rows;
                        let mut var_paths = Vec::new();
                        if matches!(key_type, KeyType::VarBytes) {
                            var_paths.push(data_path.clone());
                        }
                        for (data_path, meta_path, _) in fkey_inputs.iter().chain(edge_inputs.iter()) {
                            let meta = read_temp_meta(meta_path)?;
                            union_rows = union_rows.saturating_add(meta.row_count);
                            if matches!(key_type, KeyType::VarBytes) {
                                var_paths.push(data_path.clone());
                            }
                        }
                        let avg_var_bytes = if matches!(key_type, KeyType::VarBytes) {
                            estimate_varbytes_avg_len_multi(union_rows, meta.max_row_len, &var_paths)
                        } else {
                            0
                        };
                        let estimate = estimate_pk_map_bytes(union_rows, key_type, avg_var_bytes);
                        let _budget_guard = if let Some(budget) = pk_map_budget.as_ref() {
                            let budget_total = budget.snapshot().total;
                            if budget_total == 0 || estimate == 0 {
                                None
                            } else {
                                let request = estimate.min(budget_total);
                                Some(budget.acquire(request)?)
                            }
                        } else {
                            None
                        };
                        let mut fkey_bitmaps: Vec<memmap2::MmapMut> = Vec::new();
                        for rel in relationships.iter().filter(|r| r.src_table == table_name) {
                            let path = root.join(temp_fkey_bitmap_path(&table_name, &rel.name_forward, shard));
                            if path.exists() {
                                let file = std::fs::OpenOptions::new()
                                    .read(true)
                                    .write(true)
                                    .open(&path)?;
                                fkey_bitmaps.push(unsafe { memmap2::MmapMut::map_mut(&file)? });
                            }
                        }

                        let mut duplicate_count = 0u64;
                        let mut null_timestamp_drops = 0u64;
                        let mut local_key_failures = KeyFailureState::default();
                        let mut duplicate_samples_local: Vec<Key> = Vec::new();
                        let mut row_id = 0u64;
                        let (hash_rel, data_rel) =
                            match key_type {
                            KeyType::Int64 => {
                                {
                                    let pkmap = scratch;
                                    let mut ptrhash =
                                        std::mem::replace(&mut pkmap.ptrhash, PtrHashScratch::new());
                                    pkmap.clear_others_for(KeyType::Int64);
                                    pkmap.seen_u64.clear();
                                    reserve_map(&mut pkmap.seen_u64, reserve_rows_usize);
                                    duplicate_samples_local.clear();
                                    {
                                        let seen = &mut pkmap.seen_u64;
                                        read_temp_rows(&data_path, |row| {
                                            let timestamp_valid = if let Some(last) = row.last() {
                                                *last != 0
                                            } else {
                                                true
                                            };
                                            let key_len = row.len().saturating_sub(1);
                                            let mut data = &row[..key_len];
                                            let key = read_key_from_buf(&mut data, &pk_col)?;
                                            if let Key::Int(v) = key {
                                                if !timestamp_valid {
                                                    null_timestamp_drops = null_timestamp_drops.saturating_add(1);
                                                    let context = format!("node_data_dropped:{}", table_name);
                                                    record_key_parse_failure_sample(
                                                        &mut local_key_failures.parse_failures,
                                                        &mut local_key_failures.samples,
                                                        &mut local_key_failures.by_context,
                                                        &context,
                                                        Some(v.to_string()),
                                                    );
                                                    for bitmap in fkey_bitmaps.iter_mut() {
                                                        let byte_idx = (row_id / 8) as usize;
                                                        if byte_idx < bitmap.len() {
                                                            let bit = (row_id % 8) as u8;
                                                            bitmap[byte_idx] &= !(1u8 << bit);
                                                        }
                                                    }
                                                } else {
                                                    let local_id = u32::try_from(row_id)
                                                        .map_err(|_| anyhow!("pk local id overflow"))?;
                                                    match seen.entry(v) {
                                                        Entry::Vacant(entry) => {
                                                            entry.insert(local_id);
                                                        }
                                                        Entry::Occupied(_) => {
                                                            duplicate_count =
                                                                duplicate_count.saturating_add(1);
                                                            for bitmap in fkey_bitmaps.iter_mut() {
                                                                let byte_idx = (row_id / 8) as usize;
                                                                if byte_idx < bitmap.len() {
                                                                    let bit = (row_id % 8) as u8;
                                                                    bitmap[byte_idx] &= !(1u8 << bit);
                                                                }
                                                            }
                                                            if duplicate_samples_local.len() < 10 {
                                                                duplicate_samples_local.push(Key::Int(v));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            row_id += 1;
                                            Ok(())
                                        })?;
                                        for (path, _meta_path, col) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf(&mut data, col)?;
                                                if let Key::Int(v) = key {
                                                    seen.entry(v).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptr_hash::PtrHashParams::default();
                                    let seen = std::mem::take(&mut pkmap.seen_u64);
                                    let mphf =
                                        DefaultPtrHash::<PkPtrHash, u64>::try_new_from_source_with_scratch(
                                            &seen,
                                            params,
                                            &mut ptrhash,
                                        )
                                        .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                    let entries = {
                                        let slots = mphf.max_index();
                                        let mut entries = std::mem::take(&mut pkmap.entries);
                                        entries.resize(slots, (0u64, 0u32));
                                        for (key, value) in seen.iter() {
                                            let idx = mphf.index_no_remap(key);
                                            entries[idx] = (0, *value);
                                        }
                                        pkmap.entries = entries;
                                        pkmap.seen_u64 = seen;
                                        pkmap.entries.as_slice()
                                    };
                                    pkmap.ptrhash = ptrhash;
                                    let hash_rel = format!(
                                        "nodes/{}/pk_hash-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    let data_rel = format!(
                                        "nodes/{}/pk_data-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    IngestContext::write_pk_hash_and_data(
                                        &target,
                                        &hash_rel,
                                        &data_rel,
                                        key_type,
                                        &mphf,
                                        entries,
                                    )?;
                                    Ok::<_, anyhow::Error>((hash_rel, data_rel))
                                }
                            }
                            KeyType::VarBytes => {
                                {
                                    let pkmap = scratch;
                                    let mut ptrhash =
                                        std::mem::replace(&mut pkmap.ptrhash, PtrHashScratch::new());
                                    pkmap.clear_others_for(KeyType::VarBytes);
                                    pkmap.seen_bytes.clear();
                                    reserve_map(&mut pkmap.seen_bytes, reserve_rows_usize);
                                    duplicate_samples_local.clear();
                                    {
                                        let seen = &mut pkmap.seen_bytes;
                                        read_temp_rows(&data_path, |row| {
                                            let timestamp_valid = if let Some(last) = row.last() {
                                                *last != 0
                                            } else {
                                                true
                                            };
                                            let key_len = row.len().saturating_sub(1);
                                            let mut data = &row[..key_len];
                                            let key = read_key_from_buf(&mut data, &pk_col)?;
                                            if !timestamp_valid {
                                                null_timestamp_drops = null_timestamp_drops.saturating_add(1);
                                                let context = format!("node_data_dropped:{}", table_name);
                                                record_key_parse_failure_sample(
                                                    &mut local_key_failures.parse_failures,
                                                    &mut local_key_failures.samples,
                                                    &mut local_key_failures.by_context,
                                                    &context,
                                                    sample_key(&key),
                                                );
                                                for bitmap in fkey_bitmaps.iter_mut() {
                                                    let byte_idx = (row_id / 8) as usize;
                                                    if byte_idx < bitmap.len() {
                                                        let bit = (row_id % 8) as u8;
                                                        bitmap[byte_idx] &= !(1u8 << bit);
                                                    }
                                                }
                                            } else {
                                                let bytes = match key {
                                                    Key::Bytes(bytes) => bytes,
                                                    _ => {
                                                        row_id += 1;
                                                        return Ok(());
                                                    }
                                                };
                                                let local_id = u32::try_from(row_id)
                                                    .map_err(|_| anyhow!("pk local id overflow"))?;
                                                match seen.entry(bytes) {
                                                    Entry::Vacant(entry) => {
                                                        entry.insert(local_id);
                                                    }
                                                    Entry::Occupied(_) => {
                                                        duplicate_count =
                                                            duplicate_count.saturating_add(1);
                                                        for bitmap in fkey_bitmaps.iter_mut() {
                                                            let byte_idx = (row_id / 8) as usize;
                                                            if byte_idx < bitmap.len() {
                                                                let bit = (row_id % 8) as u8;
                                                                bitmap[byte_idx] &= !(1u8 << bit);
                                                            }
                                                        }
                                                        // Note: We don't sample VarBytes duplicates for now as they are large
                                                    }
                                                }
                                            }
                                            row_id += 1;
                                            Ok(())
                                        })?;
                                        for (path, _meta_path, col) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf(&mut data, col)?;
                                                if let Key::Bytes(bytes) = key {
                                                    seen.entry(bytes).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptr_hash::PtrHashParams::default();
                                    let seen = std::mem::take(&mut pkmap.seen_bytes);
                                    let mphf =
                                        DefaultPtrHash::<PkPtrHash, Vec<u8>>::try_new_from_source_with_scratch(
                                            &seen,
                                            params,
                                            &mut ptrhash,
                                        )
                                        .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                    let entries = {
                                        let slots = mphf.max_index();
                                        let mut entries = std::mem::take(&mut pkmap.entries);
                                        entries.resize(slots, (0u64, 0u32));
                                        for (key, value) in seen.iter() {
                                            let idx = mphf.index_no_remap(key);
                                            entries[idx] = (0, *value);
                                        }
                                        pkmap.entries = entries;
                                        pkmap.seen_bytes = seen;
                                        pkmap.entries.as_slice()
                                    };
                                    pkmap.ptrhash = ptrhash;
                                    let hash_rel = format!(
                                        "nodes/{}/pk_hash-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    let data_rel = format!(
                                        "nodes/{}/pk_data-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    IngestContext::write_pk_hash_and_data(
                                        &target,
                                        &hash_rel,
                                        &data_rel,
                                        key_type,
                                        &mphf,
                                        entries,
                                    )?;
                                    Ok::<_, anyhow::Error>((hash_rel, data_rel))
                                }
                            }
                            KeyType::Uuid => {
                                {
                                    let pkmap = scratch;
                                    let mut ptrhash =
                                        std::mem::replace(&mut pkmap.ptrhash, PtrHashScratch::new());
                                    pkmap.clear_others_for(KeyType::Uuid);
                                    pkmap.seen_uuid.clear();
                                    reserve_map(&mut pkmap.seen_uuid, reserve_rows_usize);
                                    duplicate_samples_local.clear();
                                    {
                                        let seen = &mut pkmap.seen_uuid;
                                        read_temp_rows(&data_path, |row| {
                                            let timestamp_valid = if let Some(last) = row.last() {
                                                *last != 0
                                            } else {
                                                true
                                            };
                                            let key_len = row.len().saturating_sub(1);
                                            let mut data = &row[..key_len];
                                            let key = read_key_from_buf(&mut data, &pk_col)?;
                                            if let Key::Uuid(uuid) = key {
                                                if !timestamp_valid {
                                                    null_timestamp_drops = null_timestamp_drops.saturating_add(1);
                                                    let context = format!("node_data_dropped:{}", table_name);
                                                    record_key_parse_failure_sample(
                                                        &mut local_key_failures.parse_failures,
                                                        &mut local_key_failures.samples,
                                                        &mut local_key_failures.by_context,
                                                        &context,
                                                        sample_key(&key),
                                                    );
                                                    for bitmap in fkey_bitmaps.iter_mut() {
                                                        let byte_idx = (row_id / 8) as usize;
                                                        if byte_idx < bitmap.len() {
                                                            let bit = (row_id % 8) as u8;
                                                            bitmap[byte_idx] &= !(1u8 << bit);
                                                        }
                                                    }
                                                } else {
                                                    let local_id = u32::try_from(row_id)
                                                        .map_err(|_| anyhow!("pk local id overflow"))?;
                                                    match seen.entry(uuid) {
                                                        Entry::Vacant(entry) => {
                                                            entry.insert(local_id);
                                                        }
                                                        Entry::Occupied(_) => {
                                                            duplicate_count =
                                                                duplicate_count.saturating_add(1);
                                                            for bitmap in fkey_bitmaps.iter_mut() {
                                                                let byte_idx = (row_id / 8) as usize;
                                                                if byte_idx < bitmap.len() {
                                                                    let bit = (row_id % 8) as u8;
                                                                    bitmap[byte_idx] &= !(1u8 << bit);
                                                                }
                                                            }
                                                            if duplicate_samples_local.len() < 10 {
                                                                duplicate_samples_local
                                                                    .push(Key::Uuid(uuid));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            row_id += 1;
                                            Ok(())
                                        })?;
                                        for (path, _meta_path, col) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf(&mut data, col)?;
                                                if let Key::Uuid(uuid) = key {
                                                    seen.entry(uuid).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptr_hash::PtrHashParams::default();
                                    let seen = std::mem::take(&mut pkmap.seen_uuid);
                                    let mphf =
                                        DefaultPtrHash::<PkPtrHash, [u8; 16]>::try_new_from_source_with_scratch(
                                            &seen,
                                            params,
                                            &mut ptrhash,
                                        )
                                        .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                    let entries = {
                                        let slots = mphf.max_index();
                                        let mut entries = std::mem::take(&mut pkmap.entries);
                                        entries.resize(slots, (0u64, 0u32));
                                        for (key, value) in seen.iter() {
                                            let idx = mphf.index_no_remap(key);
                                            entries[idx] = (0, *value);
                                        }
                                        pkmap.entries = entries;
                                        pkmap.seen_uuid = seen;
                                        pkmap.entries.as_slice()
                                    };
                                    pkmap.ptrhash = ptrhash;
                                    let hash_rel = format!(
                                        "nodes/{}/pk_hash-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    let data_rel = format!(
                                        "nodes/{}/pk_data-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    IngestContext::write_pk_hash_and_data(
                                        &target,
                                        &hash_rel,
                                        &data_rel,
                                        key_type,
                                        &mphf,
                                        entries,
                                    )?;
                                    Ok::<_, anyhow::Error>((hash_rel, data_rel))
                                }
                            }
                            KeyType::Uuid256 => {
                                {
                                    let pkmap = scratch;
                                    let mut ptrhash =
                                        std::mem::replace(&mut pkmap.ptrhash, PtrHashScratch::new());
                                    pkmap.clear_others_for(KeyType::Uuid256);
                                    pkmap.seen_uuid256.clear();
                                    reserve_map(&mut pkmap.seen_uuid256, reserve_rows_usize);
                                    duplicate_samples_local.clear();
                                    {
                                        let seen = &mut pkmap.seen_uuid256;
                                        read_temp_rows(&data_path, |row| {
                                            let timestamp_valid = if let Some(last) = row.last() {
                                                *last != 0
                                            } else {
                                                true
                                            };
                                            let key_len = row.len().saturating_sub(1);
                                            let mut data = &row[..key_len];
                                            let key = read_key_from_buf(&mut data, &pk_col)?;
                                            if let Key::Uuid256(uuid) = key {
                                                if !timestamp_valid {
                                                    null_timestamp_drops = null_timestamp_drops.saturating_add(1);
                                                    let context = format!("node_data_dropped:{}", table_name);
                                                    record_key_parse_failure_sample(
                                                        &mut local_key_failures.parse_failures,
                                                        &mut local_key_failures.samples,
                                                        &mut local_key_failures.by_context,
                                                        &context,
                                                        sample_key(&key),
                                                    );
                                                    for bitmap in fkey_bitmaps.iter_mut() {
                                                        let byte_idx = (row_id / 8) as usize;
                                                        if byte_idx < bitmap.len() {
                                                            let bit = (row_id % 8) as u8;
                                                            bitmap[byte_idx] &= !(1u8 << bit);
                                                        }
                                                    }
                                                } else {
                                                    let local_id = u32::try_from(row_id)
                                                        .map_err(|_| anyhow!("pk local id overflow"))?;
                                                    match seen.entry(uuid) {
                                                        Entry::Vacant(entry) => {
                                                            entry.insert(local_id);
                                                        }
                                                        Entry::Occupied(_) => {
                                                            duplicate_count =
                                                                duplicate_count.saturating_add(1);
                                                            for bitmap in fkey_bitmaps.iter_mut() {
                                                                let byte_idx = (row_id / 8) as usize;
                                                                if byte_idx < bitmap.len() {
                                                                    let bit = (row_id % 8) as u8;
                                                                    bitmap[byte_idx] &= !(1u8 << bit);
                                                                }
                                                            }
                                                            if duplicate_samples_local.len() < 10 {
                                                                duplicate_samples_local
                                                                    .push(Key::Uuid256(uuid));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            row_id += 1;
                                            Ok(())
                                        })?;
                                        for (path, _meta_path, col) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf(&mut data, col)?;
                                                if let Key::Uuid256(uuid) = key {
                                                    seen.entry(uuid).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptr_hash::PtrHashParams::default();
                                    let seen = std::mem::take(&mut pkmap.seen_uuid256);
                                    let mphf =
                                        DefaultPtrHash::<PkPtrHash, [u8; 32]>::try_new_from_source_with_scratch(
                                            &seen,
                                            params,
                                            &mut ptrhash,
                                        )
                                        .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                    let entries = {
                                        let slots = mphf.max_index();
                                        let mut entries = std::mem::take(&mut pkmap.entries);
                                        entries.resize(slots, (0u64, 0u32));
                                        for (key, value) in seen.iter() {
                                            let idx = mphf.index_no_remap(key);
                                            entries[idx] = (0, *value);
                                        }
                                        pkmap.entries = entries;
                                        pkmap.seen_uuid256 = seen;
                                        pkmap.entries.as_slice()
                                    };
                                    pkmap.ptrhash = ptrhash;
                                    let hash_rel = format!(
                                        "nodes/{}/pk_hash-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    let data_rel = format!(
                                        "nodes/{}/pk_data-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    IngestContext::write_pk_hash_and_data(
                                        &target,
                                        &hash_rel,
                                        &data_rel,
                                        key_type,
                                        &mphf,
                                        entries,
                                    )?;
                                    Ok::<_, anyhow::Error>((hash_rel, data_rel))
                                }
                            }
                            KeyType::Ip => {
                                {
                                    let pkmap = scratch;
                                    let mut ptrhash =
                                        std::mem::replace(&mut pkmap.ptrhash, PtrHashScratch::new());
                                    pkmap.clear_others_for(KeyType::Ip);
                                    pkmap.seen_ip.clear();
                                    reserve_map(&mut pkmap.seen_ip, reserve_rows_usize);
                                    duplicate_samples_local.clear();
                                    {
                                        let seen = &mut pkmap.seen_ip;
                                        read_temp_rows(&data_path, |row| {
                                            let timestamp_valid = if let Some(last) = row.last() {
                                                *last != 0
                                            } else {
                                                true
                                            };
                                            let key_len = row.len().saturating_sub(1);
                                            let mut data = &row[..key_len];
                                            let key = read_key_from_buf(&mut data, &pk_col)?;
                                            if let Key::Ip(ip) = key {
                                                if !timestamp_valid {
                                                    null_timestamp_drops = null_timestamp_drops.saturating_add(1);
                                                    let context = format!("node_data_dropped:{}", table_name);
                                                    record_key_parse_failure_sample(
                                                        &mut local_key_failures.parse_failures,
                                                        &mut local_key_failures.samples,
                                                        &mut local_key_failures.by_context,
                                                        &context,
                                                        sample_key(&key),
                                                    );
                                                    for bitmap in fkey_bitmaps.iter_mut() {
                                                        let byte_idx = (row_id / 8) as usize;
                                                        if byte_idx < bitmap.len() {
                                                            let bit = (row_id % 8) as u8;
                                                            bitmap[byte_idx] &= !(1u8 << bit);
                                                        }
                                                    }
                                                } else {
                                                    let local_id = u32::try_from(row_id)
                                                        .map_err(|_| anyhow!("pk local id overflow"))?;
                                                    match seen.entry(ip) {
                                                        Entry::Vacant(entry) => {
                                                            entry.insert(local_id);
                                                        }
                                                        Entry::Occupied(_) => {
                                                            duplicate_count =
                                                                duplicate_count.saturating_add(1);
                                                            for bitmap in fkey_bitmaps.iter_mut() {
                                                                let byte_idx = (row_id / 8) as usize;
                                                                if byte_idx < bitmap.len() {
                                                                    let bit = (row_id % 8) as u8;
                                                                    bitmap[byte_idx] &= !(1u8 << bit);
                                                                }
                                                            }
                                                            if duplicate_samples_local.len() < 10 {
                                                                duplicate_samples_local
                                                                    .push(Key::Ip(ip));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            row_id += 1;
                                            Ok(())
                                        })?;
                                        for (path, _meta_path, col) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf(&mut data, col)?;
                                                if let Key::Ip(ip) = key {
                                                    seen.entry(ip).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptr_hash::PtrHashParams::default();
                                    let seen = std::mem::take(&mut pkmap.seen_ip);
                                    let mphf =
                                        DefaultPtrHash::<PkPtrHash, [u8; 16]>::try_new_from_source_with_scratch(
                                            &seen,
                                            params,
                                            &mut ptrhash,
                                        )
                                        .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                    let entries = {
                                        let slots = mphf.max_index();
                                        let mut entries = std::mem::take(&mut pkmap.entries);
                                        entries.resize(slots, (0u64, 0u32));
                                        for (key, value) in seen.iter() {
                                            let idx = mphf.index_no_remap(key);
                                            entries[idx] = (0, *value);
                                        }
                                        pkmap.entries = entries;
                                        pkmap.seen_ip = seen;
                                        pkmap.entries.as_slice()
                                    };
                                    pkmap.ptrhash = ptrhash;
                                    let hash_rel = format!(
                                        "nodes/{}/pk_hash-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    let data_rel = format!(
                                        "nodes/{}/pk_data-{:05}.bin",
                                        table_name.to_lowercase(),
                                        shard
                                    );
                                    IngestContext::write_pk_hash_and_data(
                                        &target,
                                        &hash_rel,
                                        &data_rel,
                                        key_type,
                                        &mphf,
                                        entries,
                                    )?;
                                    Ok::<_, anyhow::Error>((hash_rel, data_rel))
                                }
                            }
                        }?;
                        if duplicate_count > 0 {
                            let key = format!("{}:shard-{:05}", table_name, shard);
                            duplicate_counts
                                .lock()
                                .unwrap()
                                .insert(key.clone(), duplicate_count);
                            let mut samples = Vec::new();
                            for key in &duplicate_samples_local {
                                if let Some(sample) = sample_key(key) {
                                    samples.push(sample);
                                }
                            }
                            if !samples.is_empty() {
                                duplicate_samples.lock().unwrap().insert(key, samples);
                            }
                        }
                        if null_timestamp_drops > 0 {
                            pk_null_ts_counts
                                .lock()
                                .unwrap()
                                .insert(format!("{}:shard-{:05}", table_name, shard), null_timestamp_drops);
                        }
                        key_failures_shared.lock().unwrap().merge(local_key_failures);

                        let dup_meta = format!(
                            "nodes/{}/pk_hash-{:05}.meta.json",
                            table_name.to_lowercase(),
                            shard
                        );
                        let mut samples = Vec::new();
                        for key in &duplicate_samples_local {
                            if let Some(sample) = sample_key(key) {
                                samples.push(sample);
                            }
                        }
                        let meta = serde_json::json!({
                            "duplicate_count": duplicate_count,
                            "duplicate_samples": samples,
                        });
                        {
                            target.write_bytes(&dup_meta, serde_json::to_vec_pretty(&meta)?)?;
                        }
                        let mut guard = pk_hash_paths.lock().unwrap();
                        guard[shard] = Some(PkHashPaths {
                            hash_path: hash_rel,
                            data_path: data_rel,
                        });
                        if let Some(cb) = &progress {
                            cb(ProgressEvent {
                                phase: ProgressPhase::PkMap,
                                table: table_name.clone(),
                                shard: None,
                                kind: ProgressKind::Advance,
                                delta: 1,
                                total: Some(num_shards),
                                status: None,
                            });
                        }
                        Ok(())
                    }),
                });
            }
        }
        let pool = self.make_pool::<PkMapScratch>();
        let run_result = run_tasks(tasks.drain(..).collect(), &pool);
        run_result?;
        for table in &self.source.tables {
            if table.edge_table.is_some() || table.primary_key.is_none() {
                continue;
            }
            let pk_hash_paths = pk_hash_handles
                .remove(&table.name)
                .ok_or_else(|| anyhow!("missing pk map paths for {}", table.name))?;
            let paths = std::sync::Arc::try_unwrap(pk_hash_paths)
                .map_err(|_| anyhow!("pk map paths still in use"))?
                .into_inner()
                .map_err(|_| anyhow!("pk map paths locked"))?
                .into_iter()
                .map(|p| p.ok_or_else(|| anyhow!("missing pk map path for {}", table.name)))
                .collect::<Result<Vec<_>>>()?;
            self.pk_hash_paths.insert(table.name.clone(), paths);
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::PkMap,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Finish,
                delta: 0,
                total: Some(self.num_shards as u64),
                status: None,
            });
        }
        if let Ok(counts) = std::sync::Arc::try_unwrap(duplicate_counts) {
            let pk_counts = counts.into_inner().unwrap();
            for (k, v) in pk_counts {
                let entry = self.pk_duplicate_counts.entry(k).or_insert(0);
                *entry = entry.saturating_add(v);
            }
        }
        if let Ok(samples) = std::sync::Arc::try_unwrap(duplicate_samples) {
            self.pk_duplicate_samples = samples.into_inner().unwrap();
        }
        if let Ok(counts) = std::sync::Arc::try_unwrap(pk_null_ts_counts) {
            let pk_counts = counts.into_inner().unwrap();
            for (k, v) in pk_counts {
                let entry = self.pk_null_timestamp_drops.entry(k).or_insert(0);
                *entry = entry.saturating_add(v);
            }
        }
        if let Ok(failures) = std::sync::Arc::try_unwrap(key_failures) {
            self.merge_key_failures(failures.into_inner().unwrap());
        }
        Ok(())
    }

    fn write_pk_hash_and_data<H: EpSerialize>(
        target: &OutputTarget,
        hash_rel: &str,
        data_rel: &str,
        key_type: KeyType,
        mphf: &H,
        entries: &[(u64, u32)],
    ) -> Result<()> {
        let hash_path = target.join(hash_rel);
        let data_path = target.join(data_rel);
        if let OutputTarget::Local { .. } = target {
            if let Some(parent) = hash_path.parent() {
                fs::create_dir_all(parent)?;
            }
            if let Some(parent) = data_path.parent() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut hash_writer =
            BufWriter::with_capacity(4 * 1024 * 1024, target.sink(&hash_path)?);
        mphf.serialize(&mut hash_writer)?;
        hash_writer.flush()?;
        let hash_sink = hash_writer
            .into_inner()
            .map_err(|_| anyhow!("pk hash flush failed"))?;
        hash_sink.finish()?;

        let mut data_writer =
            BufWriter::with_capacity(4 * 1024 * 1024, target.sink(&data_path)?);
        write_pk_data_header(&mut data_writer, key_type, entries.len() as u64)?;
        for (_key_hash, value) in entries {
            write_pk_data_entry(&mut data_writer, *value)?;
        }
        data_writer.flush()?;
        let data_sink = data_writer
            .into_inner()
            .map_err(|_| anyhow!("pk data flush failed"))?;
        data_sink.finish()?;
        Ok(())
    }

    fn edge_build_context(&mut self) -> Result<EdgeBuildContext> {
        let root = require_local_output(&self.target)?.to_path_buf();
        self.build_shard_offsets();
        let mut rel_sort_bits = HashMap::new();
        let mut rel_sort_keys = HashMap::new();
        for rel in &self.relation_defs {
            let sort_bits = rel.sort_key.as_ref().map(|cfg| cfg.width_bits);
            rel_sort_bits.insert(rel.name.clone(), sort_bits);
            if let Some(cfg) = &rel.sort_key {
                let timestamp = cfg.timestamp.as_ref().map(|ts| SortKeyComponentInfo {
                    spec: SortKeyComponentSpec {
                        kind: SortKeyComponentKind::Timestamp,
                        table: ts.table.clone(),
                        width_bits: timestamp_width_bits(&ts.encoding.kind),
                    },
                });
                let weight = cfg.weight.as_ref().map(|w| SortKeyComponentInfo {
                    spec: SortKeyComponentSpec {
                        kind: SortKeyComponentKind::Weight,
                        table: w.table.clone(),
                        width_bits: weight_width_bits(&w.encoding.dtype),
                    },
                });
                rel_sort_keys.insert(
                    rel.name.clone(),
                    RelationSortKeyInfo {
                        timestamp,
                        weight,
                    },
                );
            }
        }
        let mut max_ids = HashMap::new();
        for (table, counts) in &self.shard_rows {
            let total: u64 = counts.iter().sum();
            let max_id = total.saturating_sub(1);
            max_ids.insert(table.clone(), max_id);
        }
        let mut edge_tables = HashSet::new();
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                edge_tables.insert(table.name.clone());
            }
        }
        let rel_sort_lookups =
            self.build_sort_key_lookups_parallel(&root, &edge_tables, &rel_sort_keys)?;
        let mut sort_key_total_bytes = 0u64;
        let mut sort_key_edge_bytes = 0u64;
        let mut sort_key_dest_bytes = 0u64;
        for lookup in rel_sort_lookups.values().filter_map(|lookup| lookup.as_ref()) {
            sort_key_total_bytes = sort_key_total_bytes.saturating_add(lookup.total_bytes());
            sort_key_edge_bytes = sort_key_edge_bytes.saturating_add(lookup.edge_bytes());
            sort_key_dest_bytes = sort_key_dest_bytes.saturating_add(lookup.dest_bytes());
        }
        let (pk_hash_store, pk_hash_total_bytes) = self.build_pk_hash_store_parallel(&root)?;
        let pk_hash_store = Arc::new(pk_hash_store);
        Ok(EdgeBuildContext {
            root,
            rel_sort_bits,
            rel_sort_lookups: std::sync::Arc::new(rel_sort_lookups),
            max_ids,
            pk_hash_store,
            pk_hash_total_bytes,
            sort_key_total_bytes,
            sort_key_edge_bytes,
            sort_key_dest_bytes,
        })
    }
    fn build_pk_hash_store_parallel(&self, root: &Path) -> Result<(PkHashStore, u64)> {
        let mut shard_jobs: Vec<(String, usize, KeyType, PathBuf, PathBuf)> = Vec::new();
        let mut table_shard_counts = HashMap::new();
        for table in &self.source.tables {
            if table.edge_table.is_some() || table.primary_key.is_none() {
                continue;
            }
            let pk_name = table.primary_key.as_ref().unwrap();
            let pk_col = column_spec(table, pk_name)?;
            let key_type = key_type_for_column(pk_col)?;
            let paths = self
                .pk_hash_paths
                .get(&table.name)
                .ok_or_else(|| anyhow!("missing pk hash paths for {}", table.name))?;
            table_shard_counts.insert(table.name.clone(), paths.len());
            for (shard_idx, shard_paths) in paths.iter().enumerate() {
                let hash_path = root.join(&shard_paths.hash_path);
                let data_path = root.join(&shard_paths.data_path);
                shard_jobs.push((
                    table.name.clone(),
                    shard_idx,
                    key_type,
                    hash_path,
                    data_path,
                ));
            }
        }
        if shard_jobs.is_empty() {
            return Ok((PkHashStore::new(HashMap::new()), 0));
        }
        let mut total_bytes = 0u64;
        for (_, _, _, hash_path, data_path) in &shard_jobs {
            let hash_len = fs::metadata(hash_path)
                .with_context(|| format!("stat pk hash {}", hash_path.display()))?
                .len();
            let data_len = fs::metadata(data_path)
                .with_context(|| format!("stat pk data {}", data_path.display()))?
                .len();
            total_bytes = total_bytes.saturating_add(hash_len);
            total_bytes = total_bytes.saturating_add(data_len);
        }
        let data_advice = match total_memory_bytes() {
            Some(total_mem) if total_bytes <= total_mem.saturating_mul(75) / 100 => {
                PkDataAdvice::WillNeed
            }
            _ => PkDataAdvice::Random,
        };
        if let Some(total_mem) = total_memory_bytes() {
            println!(
                "pk_hash mmap hint: total_bytes={} total_mem={} advice={:?}",
                format_bytes(total_bytes),
                format_bytes(total_mem),
                data_advice
            );
        } else {
            println!(
                "pk_hash mmap hint: total_bytes={} total_mem=unknown advice={:?}",
                format_bytes(total_bytes),
                data_advice
            );
        }
        let (tx, rx) = std::sync::mpsc::channel::<Result<(String, usize, PkHashShard)>>();
        let mut tasks = Vec::with_capacity(shard_jobs.len());
        for (table_name, shard_idx, key_type, hash_path, data_path) in shard_jobs {
            let tx = tx.clone();
            let data_advice = data_advice;
            tasks.push(WorkItem {
                bucket: None,
                run: Box::new(move |_scratch| {
                    let shard = PkHashShard::load(&hash_path, &data_path, key_type, data_advice)?;
                    let _ = tx.send(Ok((table_name, shard_idx, shard)));
                    Ok(())
                }),
            });
        }
        drop(tx);
        let pool = self.make_pool::<()>();
        run_tasks(tasks, &pool)?;
        let mut tables: HashMap<String, Vec<Option<PkHashShard>>> = HashMap::new();
        for (table_name, shard_count) in table_shard_counts {
            let shards = (0..shard_count).map(|_| None).collect();
            tables.insert(table_name, shards);
        }
        for _ in 0..tables.values().map(|v| v.len()).sum::<usize>() {
            match rx.recv().map_err(|e| anyhow!("pk hash output error: {e}"))? {
                Ok((table_name, shard_idx, shard)) => {
                    let entry = tables
                        .get_mut(&table_name)
                        .ok_or_else(|| anyhow!("pk hash table missing {}", table_name))?;
                    if shard_idx >= entry.len() {
                        return Err(anyhow!("pk hash shard out of range {}:{}", table_name, shard_idx));
                    }
                    entry[shard_idx] = Some(shard);
                }
                Err(err) => return Err(err),
            }
        }
        let mut final_tables = HashMap::new();
        for (table_name, shards) in tables {
            let mut resolved = Vec::with_capacity(shards.len());
            for shard in shards {
                resolved.push(
                    shard.ok_or_else(|| anyhow!("missing pk hash shard {}", table_name))?,
                );
            }
            final_tables.insert(table_name, resolved);
        }
        Ok((PkHashStore::new(final_tables), total_bytes))
    }

    fn build_sort_key_lookups_parallel(
        &self,
        root: &Path,
        edge_tables: &HashSet<String>,
        rel_sort_keys: &HashMap<String, RelationSortKeyInfo>,
    ) -> Result<HashMap<String, Option<std::sync::Arc<SortKeyLookup>>>> {
        let edge_tables = std::sync::Arc::new(edge_tables.clone());
        let rel_sort_keys = std::sync::Arc::new(rel_sort_keys.clone());
        let (tx, rx) = std::sync::mpsc::channel::<Result<(String, Option<std::sync::Arc<SortKeyLookup>>)>>();
        let mut tasks = Vec::with_capacity(self.relation_defs.len());
        for rel in &self.relation_defs {
            let rel_name = rel.name.clone();
            let root = root.to_path_buf();
            let edge_tables = std::sync::Arc::clone(&edge_tables);
            let rel_sort_keys = std::sync::Arc::clone(&rel_sort_keys);
            let tx = tx.clone();
            let num_shards = self.num_shards;
            tasks.push(WorkItem {
                bucket: None,
                run: Box::new(move |_scratch| {
                    let lookup = build_sort_key_lookup(
                        &root,
                        edge_tables.as_ref(),
                        rel_sort_keys.as_ref(),
                        &rel_name,
                        num_shards,
                    )?;
                    let lookup = lookup.map(std::sync::Arc::new);
                    let _ = tx.send(Ok((rel_name, lookup)));
                    Ok(())
                }),
            });
        }
        drop(tx);
        let pool = self.make_pool::<()>();
        run_tasks(tasks, &pool)?;
        let mut lookups = HashMap::with_capacity(self.relation_defs.len());
        for _ in 0..self.relation_defs.len() {
            match rx.recv().map_err(|e| anyhow!("sort key output error: {e}"))? {
                Ok((rel_name, lookup)) => {
                    lookups.insert(rel_name, lookup);
                }
                Err(err) => return Err(err),
            }
        }
        Ok(lookups)
    }


    fn build_edges_mapped(&mut self, edge_ctx: &EdgeBuildContext) -> Result<()> {
        let root = &edge_ctx.root;
        self.edge_counts.clear();

        let pool = self.make_pool::<EdgeMapScratch>();
        {
            #[derive(Default)]
            struct EdgeMapResult {
                relation: String,
                key_failures: KeyFailureState,
                edge_counts: HashMap<String, u64>,
                pk_duplicate_drops: u64,
                null_timestamp_drops: u64,
            }
            if let Some(total_mem) = total_memory_bytes() {
                println!(
                    "edges_map mmap sizes: pk_hash_bytes={} sort_key_bytes={} (edge={} dest={}) total_mem={}",
                    format_bytes(edge_ctx.pk_hash_total_bytes),
                    format_bytes(edge_ctx.sort_key_total_bytes),
                    format_bytes(edge_ctx.sort_key_edge_bytes),
                    format_bytes(edge_ctx.sort_key_dest_bytes),
                    format_bytes(total_mem)
                );
            } else {
                println!(
                    "edges_map mmap sizes: pk_hash_bytes={} sort_key_bytes={} (edge={} dest={}) total_mem=unknown",
                    format_bytes(edge_ctx.pk_hash_total_bytes),
                    format_bytes(edge_ctx.sort_key_total_bytes),
                    format_bytes(edge_ctx.sort_key_edge_bytes),
                    format_bytes(edge_ctx.sort_key_dest_bytes)
                );
            }
            let progress = self.progress.clone();
            let target = std::sync::Arc::clone(&self.target);
            let buffer_pool = std::sync::Arc::new(EdgeBufferPool::new());
            let writer_pool = EdgeWriterPool::new(
                std::sync::Arc::clone(&target),
                edge_writer_workers(),
                std::sync::Arc::clone(&buffer_pool),
            );
            let writer_senders = writer_pool.senders();
            let shard_offsets = self.shard_offsets.clone();
            let pk_hash_store = edge_ctx.pk_hash_store.clone();
            let max_ids = edge_ctx.max_ids.clone();
            let rel_sort_bits = edge_ctx.rel_sort_bits.clone();
            let rel_sort_lookups = std::sync::Arc::clone(&edge_ctx.rel_sort_lookups);
            let source = self.source.clone();
            let num_shards = self.num_shards;
            let (tx, rx) = std::sync::mpsc::channel::<Result<EdgeMapResult>>();
            let mut tasks = Vec::new();
            let mut expected = HashMap::new();
            let mut totals = HashMap::new();

            for rel in &self.relationships {
                let rel = rel.clone();
                let progress = progress.clone();
                let shard_offsets = shard_offsets.clone();
                let pk_hash_store = pk_hash_store.clone();
                let max_ids = max_ids.clone();
                let rel_sort_bits = rel_sort_bits.clone();
                let root = root.clone();
                let source = source.clone();
                let tx = tx.clone();
                let src_table = table_by_name(&source, &rel.src_table)?;
                let src_fk_col = column_spec(src_table, &rel.src_column)?;
                let mut total_rows = 0u64;
                for shard in 0..num_shards {
                    let fkey_meta = root.join(temp_fkey_meta_path(
                        &rel.src_table,
                        &rel.name_forward,
                        shard,
                    ));
                    let meta = read_temp_meta(&fkey_meta)?;
                    total_rows = total_rows.saturating_add(meta.row_count);
                }
                if let Some(cb) = &progress {
                    cb(ProgressEvent {
                        phase: ProgressPhase::EdgesPassA,
                        table: rel.name_forward.clone(),
                        shard: None,
                        kind: ProgressKind::Start,
                        delta: 0,
                        total: Some(total_rows),
                        status: None,
                    });
                }
                expected.insert(rel.name_forward.clone(), num_shards as u64);
                totals.insert(rel.name_forward.clone(), total_rows);
                for shard in 0..num_shards {
                    let rel = rel.clone();
                    let progress = progress.clone();
                    let shard_offsets = shard_offsets.clone();
                    let pk_hash_store = pk_hash_store.clone();
                    let max_ids = max_ids.clone();
                    let rel_sort_bits = rel_sort_bits.clone();
                    let rel_sort_lookups = std::sync::Arc::clone(&rel_sort_lookups);
                    let root = root.clone();
                    let writer_senders = std::sync::Arc::clone(&writer_senders);
                    let buffer_pool = std::sync::Arc::clone(&buffer_pool);
                    let tx = tx.clone();
                    let src_fk_col = src_fk_col.clone();
                    tasks.push(WorkItem {
                        bucket: None,
                        run: Box::new(move |scratch: &mut EdgeMapScratch| {
                            let failure_context =
                                format!("fkey:{}.{}", rel.src_table, rel.src_column);
                            let lookup_context = format!("dst_lookup:{}", rel.name_forward);
                            let mut result = EdgeMapResult::default();
                            result.relation = rel.name_forward.clone();
                            let src_offsets = shard_offsets
                                .get(&rel.src_table)
                                .cloned()
                                .unwrap_or_else(|| vec![0; num_shards]);
                            let dst_offsets = shard_offsets
                                .get(&rel.dst_table)
                                .cloned()
                                .unwrap_or_else(|| vec![0; num_shards]);
                            let forward_sort_bits = rel_sort_bits
                                .get(&rel.name_forward)
                                .copied()
                                .flatten();
                            let reverse_sort_bits = rel_sort_bits
                                .get(&rel.name_reverse)
                                .copied()
                                .flatten();
                            let forward_lookup = rel_sort_lookups
                                .get(&rel.name_forward)
                                .and_then(|lookup| lookup.as_ref().map(std::sync::Arc::clone));
                            let reverse_lookup = rel_sort_lookups
                                .get(&rel.name_reverse)
                                .and_then(|lookup| lookup.as_ref().map(std::sync::Arc::clone));
                            let forward_spec = edge_binary_spec(
                                max_ids.get(&rel.src_table).copied().unwrap_or(0),
                                max_ids.get(&rel.dst_table).copied().unwrap_or(0),
                                forward_sort_bits,
                            );
                            let reverse_spec = edge_binary_spec(
                                max_ids.get(&rel.dst_table).copied().unwrap_or(0),
                                max_ids.get(&rel.src_table).copied().unwrap_or(0),
                                reverse_sort_bits,
                            );
                            let buffer_limit = edge_batch_buffer_limit(num_shards);
                            let mut forward_batcher = EdgeBatcher::new(
                                std::sync::Arc::<str>::from(rel.name_forward.clone()),
                                forward_spec,
                                num_shards,
                                buffer_limit,
                                std::sync::Arc::clone(&writer_senders),
                                std::sync::Arc::clone(&buffer_pool),
                            );
                            let mut reverse_batcher = EdgeBatcher::new(
                                std::sync::Arc::<str>::from(rel.name_reverse.clone()),
                                reverse_spec,
                                num_shards,
                                buffer_limit,
                                std::sync::Arc::clone(&writer_senders),
                                std::sync::Arc::clone(&buffer_pool),
                            );
                            let fkey_meta = root.join(temp_fkey_meta_path(
                                &rel.src_table,
                                &rel.name_forward,
                                shard,
                            ));
                            let meta = read_temp_meta(&fkey_meta)?;
                            let mut dst_maps = Vec::with_capacity(num_shards);
                            for dst_shard in 0..num_shards {
                                dst_maps.push(pk_hash_store.shard(&rel.dst_table, dst_shard)?);
                            }
                            let mut pending = 0u64;
                            let mut local_count = 0u64;
                            let mut forward_count = 0u64;
                            let mut reverse_count = 0u64;
                            let mut pk_duplicate_drops = 0u64;
                            let fkey_path = root.join(temp_fkey_path(
                                &rel.src_table,
                                &rel.name_forward,
                                shard,
                            ));
                            let bitmap_path = root.join(temp_fkey_bitmap_path(
                                &rel.src_table,
                                &rel.name_forward,
                                shard,
                            ));
                            {
                                let edge_scratch = scratch;
                                edge_scratch.key_buf.clear();
                                let fkey_bitmap = read_fkey_bitmap_mmap(&bitmap_path, meta.row_count)?;
                                edge_scratch.fkey_bitmap = Some(fkey_bitmap);
                                let fkey_bitmap = edge_scratch.fkey_bitmap.as_ref().unwrap();
                                let key_buf = &mut edge_scratch.key_buf;
                                let result = read_temp_rows(&fkey_path, |row| {
                                    pending = pending.saturating_add(1);
                                    if pending >= PROGRESS_ROW_BATCH {
                                        if let Some(cb) = &progress {
                                            cb(ProgressEvent {
                                                phase: ProgressPhase::EdgesPassA,
                                                table: rel.name_forward.clone(),
                                                shard: None,
                                                kind: ProgressKind::Advance,
                                                delta: pending,
                                                total: Some(total_rows),
                                                status: None,
});
                                        }
                                        pending = 0;
                                    }
                                    let src_local = local_count;
                                    let src_id = src_offsets
                                        .get(shard)
                                        .copied()
                                        .unwrap_or(0)
                                        .saturating_add(local_count);
                                    local_count = local_count.saturating_add(1);
                                    let byte_idx = (src_local / 8) as usize;
                                    if byte_idx >= fkey_bitmap.len() {
                                        return Err(anyhow!("fkey bitmap truncated"));
                                    }
                                    let bit = (src_local % 8) as u8;
                                    if (fkey_bitmap[byte_idx] >> bit) & 1u8 == 0u8 {
                                        pk_duplicate_drops = pk_duplicate_drops.saturating_add(1);
                                        return Ok(());
                                    }
                                    let mut data = row;
                                    let dst_key = match read_key_from_buf(&mut data, &src_fk_col) {
                                        Ok(key) => key,
                                        Err(_) => {
                                            record_key_parse_failure(
                                                &mut result.key_failures.parse_failures,
                                                &mut result.key_failures.samples,
                                                &mut result.key_failures.by_context,
                                                &failure_context,
                                                row,
                                            );
                                            return Ok(());
                                        }
                                    };
                                    let dst_shard =
                                        shard_for_key_from_key(&dst_key, key_buf, num_shards)?;
                                    let dst_local = match dst_maps
                                        .get(dst_shard)
                                        .and_then(|map| map.lookup(&dst_key))
                                    {
                                        Some(id) => u64::from(id),
                                        None => {
                                            let sample = if result.key_failures.samples.len() < 10 {
                                                sample_key(&dst_key)
                                            } else {
                                                None
                                            };
                                            record_key_lookup_failure(
                                                &mut result.key_failures.lookup_failures,
                                                &mut result.key_failures.samples,
                                                &mut result.key_failures.by_context,
                                                &lookup_context,
                                                sample,
                                            );
                                            return Ok(());
                                        }
                                    };
                                    let dst_id = dst_offsets
                                        .get(dst_shard)
                                        .copied()
                                        .unwrap_or(0)
                                        .saturating_add(dst_local);
                                    let forward_sort_key = match &forward_lookup {
                                        Some(lookup) => lookup.sort_key(
                                            dst_shard,
                                            dst_local,
                                            shard,
                                            0,
                                        )?,
                                        None => 0,
                                    };
                                    let reverse_sort_key = match &reverse_lookup {
                                        Some(lookup) => lookup.sort_key(
                                            shard,
                                            src_local,
                                            shard,
                                            0,
                                        )?,
                                        None => 0,
                                    };
                                    forward_batcher.push(shard, src_id, dst_id, forward_sort_key)?;
                                    reverse_batcher.push(dst_shard, dst_id, src_id, reverse_sort_key)?;
                                    forward_count = forward_count.saturating_add(1);
                                    reverse_count = reverse_count.saturating_add(1);
                                    Ok(())
                                });
                                result?;
                            }
                            forward_batcher.flush_all()?;
                            reverse_batcher.flush_all()?;
                            if pending > 0 {
                                if let Some(cb) = &progress {
                                    cb(ProgressEvent {
                                        phase: ProgressPhase::EdgesPassA,
                                        table: rel.name_forward.clone(),
                                        shard: None,
                                        kind: ProgressKind::Advance,
                                        delta: pending,
                                        total: Some(total_rows),
                                        status: None,
});
                                }
                            }
                            result
                                .edge_counts
                                .insert(rel.name_forward.clone(), forward_count);
                            result
                                .edge_counts
                                .insert(rel.name_reverse.clone(), reverse_count);
                            result.pk_duplicate_drops = pk_duplicate_drops;
                            tx.send(Ok(result))
                                .map_err(|e| anyhow!("edges_map output error: {e}"))?;
                            Ok(())
                        }),
                    });
                }
            }

            for table in self.source.tables.iter().filter(|t| t.edge_table.is_some()) {
                let table = table.clone();
                let progress = progress.clone();
                let shard_offsets = shard_offsets.clone();
                let pk_hash_store = pk_hash_store.clone();
                let max_ids = max_ids.clone();
                let rel_sort_bits = rel_sort_bits.clone();
                let root = root.clone();
                let tx = tx.clone();
                let edge = table
                    .edge_table
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing edge table spec for {}", table.name))?
                    .clone();
                let forward_name = format!("edge_{}_src_to_dst", table.name);
                let reverse_name = format!("edge_{}_dst_to_src", table.name);
                let mut total_rows = 0u64;
                for shard in 0..num_shards {
                    let meta =
                        read_temp_meta(&root.join(temp_edge_meta_path(&forward_name, shard)))?;
                    total_rows = total_rows.saturating_add(meta.row_count);
                }
                if let Some(cb) = &progress {
                    cb(ProgressEvent {
                        phase: ProgressPhase::EdgesPassA,
                        table: forward_name.clone(),
                        shard: None,
                        kind: ProgressKind::Start,
                        delta: 0,
                        total: Some(total_rows),
                        status: None,
                    });
                }
                expected.insert(forward_name.clone(), num_shards as u64);
                totals.insert(forward_name.clone(), total_rows);
                let src_table = table_by_name(&source, &edge.src_fk.ref_table)?;
                let dst_table = table_by_name(&source, &edge.dst_fk.ref_table)?;
                let src_pk = src_table
                    .primary_key
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing pk for {}", edge.src_fk.ref_table))?;
                let dst_pk = dst_table
                    .primary_key
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing pk for {}", edge.dst_fk.ref_table))?;
                let src_pk_col = column_spec(src_table, src_pk)?;
                let dst_pk_col = column_spec(dst_table, dst_pk)?;
                for shard in 0..num_shards {
                    let progress = progress.clone();
                    let shard_offsets = shard_offsets.clone();
                    let pk_hash_store = pk_hash_store.clone();
                    let max_ids = max_ids.clone();
                    let rel_sort_bits = rel_sort_bits.clone();
                    let rel_sort_lookups = std::sync::Arc::clone(&rel_sort_lookups);
                    let root = root.clone();
                    let tx = tx.clone();
                    let writer_senders = std::sync::Arc::clone(&writer_senders);
                    let buffer_pool = std::sync::Arc::clone(&buffer_pool);
                    let edge = edge.clone();
                    let forward_name = forward_name.clone();
                    let reverse_name = reverse_name.clone();
                    let src_pk_col = src_pk_col.clone();
                    let dst_pk_col = dst_pk_col.clone();
                    let table_name = table.name.clone();
                    tasks.push(WorkItem {
                        bucket: None,
                        run: Box::new(move |scratch: &mut EdgeMapScratch| {
                            let mut result = EdgeMapResult::default();
                            result.relation = forward_name.clone();
                            let src_lookup_context = format!("src_lookup:{}", forward_name);
                            let dst_lookup_context = format!("dst_lookup:{}", forward_name);
                            let src_offsets = shard_offsets
                                .get(&edge.src_fk.ref_table)
                                .cloned()
                                .unwrap_or_else(|| vec![0; num_shards]);
                            let dst_offsets = shard_offsets
                                .get(&edge.dst_fk.ref_table)
                                .cloned()
                                .unwrap_or_else(|| vec![0; num_shards]);
                            let forward_sort_bits = rel_sort_bits
                                .get(&forward_name)
                                .copied()
                                .flatten();
                            let reverse_sort_bits = rel_sort_bits
                                .get(&reverse_name)
                                .copied()
                                .flatten();
                            let forward_lookup = rel_sort_lookups
                                .get(&forward_name)
                                .and_then(|lookup| lookup.as_ref().map(std::sync::Arc::clone));
                            let reverse_lookup = rel_sort_lookups
                                .get(&reverse_name)
                                .and_then(|lookup| lookup.as_ref().map(std::sync::Arc::clone));
                            let forward_spec = edge_binary_spec(
                                max_ids
                                    .get(&edge.src_fk.ref_table)
                                    .copied()
                                    .unwrap_or(0),
                                max_ids
                                    .get(&edge.dst_fk.ref_table)
                                    .copied()
                                    .unwrap_or(0),
                                forward_sort_bits,
                            );
                            let reverse_spec = edge_binary_spec(
                                max_ids
                                    .get(&edge.dst_fk.ref_table)
                                    .copied()
                                    .unwrap_or(0),
                                max_ids
                                    .get(&edge.src_fk.ref_table)
                                    .copied()
                                    .unwrap_or(0),
                                reverse_sort_bits,
                            );
                            let buffer_limit = edge_batch_buffer_limit(num_shards);
                            let mut forward_batcher = EdgeBatcher::new(
                                std::sync::Arc::<str>::from(forward_name.clone()),
                                forward_spec,
                                num_shards,
                                buffer_limit,
                                std::sync::Arc::clone(&writer_senders),
                                std::sync::Arc::clone(&buffer_pool),
                            );
                            let mut reverse_batcher = EdgeBatcher::new(
                                std::sync::Arc::<str>::from(reverse_name.clone()),
                                reverse_spec,
                                num_shards,
                                buffer_limit,
                                std::sync::Arc::clone(&writer_senders),
                                std::sync::Arc::clone(&buffer_pool),
                            );
                            let mut pending = 0u64;
                            let mut forward_count = 0u64;
                            let mut reverse_count = 0u64;
                            let mut null_timestamp_drops = 0u64;
                            let forward_path = root.join(temp_edge_path(&forward_name, shard));
                            let mut row_idx = 0u64;
                            {
                                let edge_scratch = scratch;
                                edge_scratch.key_buf.clear();
                                let key_buf = &mut edge_scratch.key_buf;
                                let result = read_temp_rows(&forward_path, |row| {
                                    pending = pending.saturating_add(1);
                                    if pending >= PROGRESS_ROW_BATCH {
                                        if let Some(cb) = &progress {
                                        cb(ProgressEvent {
                                            phase: ProgressPhase::EdgesPassA,
                                            table: forward_name.clone(),
                                            shard: None,
                                            kind: ProgressKind::Advance,
                                            delta: pending,
                                            total: Some(total_rows),
                                            status: None,
});
                                    }
                                    pending = 0;
                                }
                                let mut data = row;
                                let src_key = read_key_from_buf_len_prefixed(&mut data, &src_pk_col)
                                    .map_err(|_| anyhow!("src key parse"))?;
                                let dst_key = read_key_from_buf_len_prefixed(&mut data, &dst_pk_col)
                                    .map_err(|_| anyhow!("dst key parse"))?;
                                let timestamp_valid = if !data.is_empty() {
                                    data[0] != 0
                                } else {
                                    true
                                };
                                if !timestamp_valid {
                                    null_timestamp_drops = null_timestamp_drops.saturating_add(1);
                                    let context = format!("node_data_dropped:{}", table_name);
                                    record_key_parse_failure_sample(
                                        &mut result.key_failures.parse_failures,
                                        &mut result.key_failures.samples,
                                        &mut result.key_failures.by_context,
                                        &context,
                                        sample_key(&src_key),
                                    );
                                    row_idx = row_idx.saturating_add(1);
                                    return Ok(());
                                }
                                let src_shard =
                                    shard_for_key_from_key(&src_key, key_buf, num_shards)?;
                                let dst_shard =
                                    shard_for_key_from_key(&dst_key, key_buf, num_shards)?;
                                let src_local = match pk_hash_store
                                    .shard(&edge.src_fk.ref_table, src_shard)?
                                    .lookup(&src_key)
                                {
                                    Some(id) => u64::from(id),
                                    None => {
                                        let sample = if result.key_failures.samples.len() < 10 {
                                            sample_key(&src_key)
                                        } else {
                                            None
                                        };
                                        record_key_lookup_failure(
                                            &mut result.key_failures.lookup_failures,
                                            &mut result.key_failures.samples,
                                            &mut result.key_failures.by_context,
                                            &src_lookup_context,
                                            sample,
                                        );
                                        return Ok(());
                                    }
                                };
                                let dst_local = match pk_hash_store
                                    .shard(&edge.dst_fk.ref_table, dst_shard)?
                                    .lookup(&dst_key)
                                {
                                    Some(id) => u64::from(id),
                                    None => {
                                        let sample = if result.key_failures.samples.len() < 10 {
                                            sample_key(&dst_key)
                                        } else {
                                            None
                                        };
                                        record_key_lookup_failure(
                                            &mut result.key_failures.lookup_failures,
                                            &mut result.key_failures.samples,
                                            &mut result.key_failures.by_context,
                                            &dst_lookup_context,
                                            sample,
                                        );
                                        return Ok(());
                                    }
                                };
                                let src_id = src_offsets
                                    .get(src_shard)
                                    .copied()
                                    .unwrap_or(0)
                                    .saturating_add(src_local);
                                let dst_id = dst_offsets
                                    .get(dst_shard)
                                    .copied()
                                    .unwrap_or(0)
                                    .saturating_add(dst_local);
                                let forward_sort_key = match &forward_lookup {
                                    Some(lookup) => lookup.sort_key(
                                        dst_shard,
                                        dst_local,
                                        shard,
                                        row_idx,
                                    )?,
                                    None => 0,
                                };
                                let reverse_sort_key = match &reverse_lookup {
                                    Some(lookup) => lookup.sort_key(
                                        src_shard,
                                        src_local,
                                        shard,
                                        row_idx,
                                    )?,
                                    None => 0,
                                };
                                forward_batcher.push(shard, src_id, dst_id, forward_sort_key)?;
                                reverse_batcher.push(dst_shard, dst_id, src_id, reverse_sort_key)?;
                                forward_count = forward_count.saturating_add(1);
                                reverse_count = reverse_count.saturating_add(1);
                                row_idx = row_idx.saturating_add(1);
                                Ok(())
                                });
                                result?;
                            }
                            forward_batcher.flush_all()?;
                            reverse_batcher.flush_all()?;
                            if pending > 0 {
                                if let Some(cb) = &progress {
                                    cb(ProgressEvent {
                                        phase: ProgressPhase::EdgesPassA,
                                        table: forward_name.clone(),
                                        shard: None,
                                        kind: ProgressKind::Advance,
                                        delta: pending,
                                        total: Some(total_rows),
                                        status: None,
});
                                }
                            }
                            result.edge_counts.insert(forward_name, forward_count);
                            result.edge_counts.insert(reverse_name, reverse_count);
                            result.null_timestamp_drops = null_timestamp_drops;
                            tx.send(Ok(result))
                                .map_err(|e| anyhow!("edges_map output error: {e}"))?;
                            Ok(())
                        }),
                    });
                }
            }

            drop(tx);
            let task_count = tasks.len();
            run_tasks(tasks, &pool)?;
            let mut done = HashMap::new();
            for _ in 0..task_count {
                let result = rx
                    .recv()
                    .map_err(|e| anyhow!("edges_map output error: {e}"))??;
                self.merge_key_failures(result.key_failures);
                for (rel, count) in result.edge_counts {
                    let entry = self.edge_counts.entry(rel).or_insert(0);
                    *entry = entry.saturating_add(count);
                }
                if result.pk_duplicate_drops > 0 {
                    let entry = self
                        .pk_duplicate_drops
                        .entry(result.relation.clone())
                        .or_insert(0);
                    *entry = entry.saturating_add(result.pk_duplicate_drops);
                }
                if result.null_timestamp_drops > 0 {
                    let entry = self
                        .edge_null_timestamp_drops
                        .entry(result.relation.clone())
                        .or_insert(0);
                    *entry = entry.saturating_add(result.null_timestamp_drops);
                }
                let entry = done.entry(result.relation.clone()).or_insert(0u64);
                *entry = entry.saturating_add(1);
                if let Some(expected) = expected.get(&result.relation) {
                    if *entry == *expected {
                        let total = totals.get(&result.relation).copied();
                        if let Some(cb) = &self.progress {
                            cb(ProgressEvent {
                                phase: ProgressPhase::EdgesPassA,
                                table: result.relation.clone(),
                                shard: None,
                                kind: ProgressKind::Finish,
                                delta: 0,
                                total,
                                status: None,
});
                        }
                    }
                }
            }
            drop(writer_senders);
            writer_pool.finish()?;
            self.ensure_edge_map_shards(edge_ctx)?;
            self.temp_edges_src_bytes = self
                .temp_edges_src_bytes
                .saturating_add(self.record_edge_stage_bytes("edges_src")?);
            self.temp_fkeys_bytes = self
                .temp_fkeys_bytes
                .saturating_add(self.record_edge_stage_bytes("fkeys")?)
                .saturating_add(self.record_edge_stage_bytes("fkeys_dst")?)
                .saturating_add(self.record_edge_stage_bytes("edge_keys")?);
            self.temp_edges_map_bytes = self
                .temp_edges_map_bytes
                .saturating_add(self.record_edge_stage_bytes("edges_map")?);
            self.cleanup_edge_stage("edges_src")?;
            self.cleanup_edge_stage("fkeys")?;
            self.cleanup_edge_stage("fkeys_dst")?;
            self.cleanup_edge_stage("edge_keys")?;
            return Ok(());
        }
    }

    fn ensure_edge_map_shards(&self, edge_ctx: &EdgeBuildContext) -> Result<()> {
        if !matches!(&*self.target, OutputTarget::Local { .. }) {
            return Ok(());
        }
        let root = &edge_ctx.root;
        for rel in &self.relation_defs {
            let sort_bits = edge_ctx.rel_sort_bits.get(&rel.name).copied().flatten();
            let src_max = edge_ctx.max_ids.get(&rel.src_table).copied().unwrap_or(0);
            let dst_max = edge_ctx.max_ids.get(&rel.dst_table).copied().unwrap_or(0);
            let spec = edge_binary_spec(src_max, dst_max, sort_bits);
            for shard in 0..self.num_shards {
                let path = root.join(temp_edge_map_part_path(&rel.name, shard));
                if path.exists() {
                    continue;
                }
                let writer = EdgePartWriter::open(&self.target, &path, spec)?;
                writer.finish()?;
            }
        }
        Ok(())
    }

    fn build_shard_offsets(&mut self) {
        let mut offsets = HashMap::new();
        for (table, counts) in &self.shard_rows {
            let mut shard_offsets = Vec::with_capacity(counts.len());
            let mut acc = 0u64;
            for count in counts {
                shard_offsets.push(acc);
                acc = acc.saturating_add(*count);
            }
            offsets.insert(table.clone(), shard_offsets);
        }
        self.shard_offsets = offsets;
    }

    fn cleanup_temp(&self) -> Result<()> {
        if let OutputTarget::Local { root } = &*self.target {
            let temp_root = root.join(TEMP_DIR_NAME);
            if temp_root.exists() {
                spawn_background_delete(temp_root);
            }
        }
        Ok(())
    }

    fn cleanup_edge_stage(&self, dir: &str) -> Result<()> {
        if let OutputTarget::Local { root } = &*self.target {
            let stage_root = root.join(TEMP_DIR_NAME).join(dir);
            if stage_root.exists() {
                spawn_background_delete(stage_root);
            }
        }
        Ok(())
    }

    fn record_edge_stage_bytes(&mut self, dir: &str) -> Result<u64> {
        if let OutputTarget::Local { root } = &*self.target {
            let stage_root = root.join(TEMP_DIR_NAME).join(dir);
            if stage_root.exists() {
                let bytes = sum_dir_bytes(&stage_root).unwrap_or(0);
                return Ok(bytes);
            }
        }
        Ok(0)
    }

    fn build_index(&mut self, metadata: DatasetMetadata) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        let builder = GraphIndexBuilder::from_metadata(
            metadata,
            GraphDataSource::Local { root: root.clone() },
            &root,
        );
        let table_totals = self.index_table_totals();
        let mem_budget = self.index_mem_budget.clone();
        let edge_buffer_pool = mem_budget
            .as_ref()
            .map(|_| std::sync::Arc::new(IndexEdgeBufferPool::new(mem_budget.clone())));
        fs::create_dir_all(&root)?;
        let metadata_path = root.join("index_metadata.json");
        let metadata_file = fs::File::create(&metadata_path)?;
        let mut metadata_writer = BufWriter::new(metadata_file);
        serde_json::to_writer_pretty(&mut metadata_writer, builder.metadata())?;
        metadata_writer.flush()?;
        let jobs = builder.prepare_jobs_with_pool(edge_buffer_pool)?;
        let adapter = self
            .progress
            .clone()
            .map(|cb| {
                IndexProgressAdapter::new(cb, table_totals.clone())
            });
        let adapter = adapter.map(std::sync::Arc::new);
        let mut tasks = Vec::with_capacity(jobs.len());
        for job in jobs {
            let adapter = adapter.clone();
            let mem_budget = mem_budget.clone();
            tasks.push(WorkItem {
                bucket: None,
                run: Box::new(move |_scratch| {
                    let progress = adapter
                        .as_ref()
                        .map(|value| value.as_ref() as &dyn GraphProgressSink);
                    let result = job.run_with_budget(progress, mem_budget).map(|_| ());
                    result
                }),
            });
        }
        let pool = self.make_pool::<()>();
        run_tasks(tasks, &pool)?;
        Ok(())
    }


    fn index_table_totals(&self) -> HashMap<String, TableIndexTotals> {
        let mut totals = HashMap::new();
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            let shard_rows = self
                .shard_rows
                .get(&table.name)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let total_nodes = shard_rows.iter().copied().sum::<u64>();
            totals.insert(
                table.name.clone(),
                TableIndexTotals {
                    total_nodes,
                    total_shards: shard_rows.len(),
                },
            );
        }
        totals
    }

    fn build_index_metadata(&self) -> Result<DatasetMetadata> {
        let mut node_types = Vec::new();
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            let shard_rows = self
                .shard_rows
                .get(&table.name)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let shard_offsets = self
                .shard_offsets
                .get(&table.name)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let mut shards = Vec::new();
            for shard in 0..self.num_shards {
                let count = shard_rows.get(shard).copied().unwrap_or(0);
                let min_id = shard_offsets.get(shard).copied().unwrap_or(0);
                let (min_id, max_id) = shard_range(min_id, count);
                let node_data_paths = vec![temp_node_data_path(&table.name, shard)];
                shards.push(NodeShardMetadata {
                    min_id,
                    max_id,
                    node_data_paths: Some(node_data_paths),
                });
            }
            let context_path = self.table_contexts.get(&table.name).cloned();
            let compression = context_path.map(|path| NodeDataCompression {
                codec: "rowpack".to_string(),
                context_path: path,
            });
            let pk_map = table.primary_key.as_ref().map(|pk| PrimaryKeyMap {
                column: pk.clone(),
                dtype: table
                    .columns
                    .iter()
                    .find(|c| c.name == *pk)
                    .map(|c| format!("{:?}", c.ingested_dtype).to_lowercase())
                    .unwrap_or_else(|| "unknown".to_string()),
                mapping_path: format!("nodes/{}/pk_data-{{shard}}.bin", table.name.to_lowercase()),
                hash_path: Some(format!(
                    "nodes/{}/pk_hash-{{shard}}.bin",
                    table.name.to_lowercase()
                )),
                mapping_kind: "ptrhash".to_string(),
                fixed_len: table
                    .columns
                    .iter()
                    .find(|c| c.name == *pk)
                    .and_then(|c| c.fixed_len),
                pk_column_index: table.columns.iter().position(|c| c.name == *pk),
            });
            let max_id = shard_rows.iter().sum::<u64>().saturating_sub(1);
            node_types.push(NodeTypeMetadata {
                name: table.name.clone(),
                max_id,
                has_node_data: true,
                node_data_compression: compression,
                primary_key_map: pk_map,
                shards,
            });
        }

        let mut edge_relations = Vec::new();
        for rel in &self.relation_defs {
            let src_shards = self
                .shard_offsets
                .get(&rel.src_table)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let src_counts = self
                .shard_rows
                .get(&rel.src_table)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let mut shards = Vec::new();
            for shard in 0..self.num_shards {
                let min_id = src_shards.get(shard).copied().unwrap_or(0);
                let count = src_counts.get(shard).copied().unwrap_or(0);
                let (min_id, max_id) = shard_range(min_id, count);
                shards.push(EdgeShardMetadata {
                    path: temp_edge_map_shard_dir(&rel.name, shard),
                    src_min_id: min_id,
                    src_max_id: max_id,
                });
            }
            edge_relations.push(EdgeRelationMetadata {
                name: rel.name.clone(),
                src_type: rel.src_table.clone(),
                dst_type: rel.dst_table.clone(),
                shards,
                dest_sort_key_bits: rel.sort_key.as_ref().map(|cfg| cfg.width_bits),
                dest_sort_key: rel.sort_key.as_ref().map(build_sort_key_metadata),
                format: EdgeStorageFormat::Binary,
            });
        }

        let index_metadata = DatasetMetadata {
            dataset_name: self.source.dataset_name.clone(),
            node_types,
            edge_relations,
        };
        Ok(index_metadata)
    }

    fn finalize_metadata(&self, mut metadata: DatasetMetadata) -> Result<DatasetMetadata> {
        for node in &mut metadata.node_types {
            for shard in &mut node.shards {
                shard.node_data_paths = None;
            }
        }
        for rel in &mut metadata.edge_relations {
            for shard in &mut rel.shards {
                shard.path.clear();
            }
        }
        let json = serde_json::to_vec_pretty(&metadata)?;
        self.target.write_bytes("index_metadata.json", json)?;
        Ok(metadata)
    }

    fn build_stats(&self, metadata: &DatasetMetadata, perf: PerfStats) -> IngestStats {
        let mut node_counts = HashMap::new();
        for node in metadata.node_types.iter() {
            let count = node.max_id.saturating_add(1);
            node_counts.insert(node.name.clone(), count);
        }
        let total_nodes = node_counts.values().sum();
        let total_edges = self.edge_counts.values().sum();
        let pk_duplicate_total: u64 = self.pk_duplicate_counts.values().sum();
        let trimmed_strings = self.trimmed_strings.load(Ordering::Relaxed);

        let mut dropped_rows_counts = self.dropped_rows_counts.clone();
        for (key, count) in &self.pk_null_timestamp_drops {
            if let Some((table, _)) = key.split_once(":shard-") {
                let entry = dropped_rows_counts.entry(table.to_string()).or_insert(0);
                *entry = entry.saturating_add(*count);
            }
        }
        for (rel, count) in &self.edge_null_timestamp_drops {
            if rel.starts_with("edge_") && rel.ends_with("_src_to_dst") {
                let table = &rel[5..rel.len() - 11];
                let entry = dropped_rows_counts.entry(table.to_string()).or_insert(0);
                *entry = entry.saturating_add(*count);
            }
        }

        IngestStats {
            output: self.output.clone(),
            node_tables: metadata.node_types.len(),
            edge_relations: metadata.edge_relations.len(),
            total_nodes,
            total_edges,
            node_counts,
            edge_counts: self.edge_counts.clone(),
            key_parse_failures: self.key_parse_failures,
            key_lookup_failures: self.key_lookup_failures,
            key_failure_samples: self.key_failure_samples.clone(),
            key_failures_by_context: self.key_failures_by_context.clone(),
            pk_duplicate_counts: self.pk_duplicate_counts.clone(),
            pk_duplicate_drops: self.pk_duplicate_drops.clone(),
            pk_null_timestamp_drops: self.pk_null_timestamp_drops.clone(),
            edge_null_timestamp_drops: self.edge_null_timestamp_drops.clone(),
            pk_duplicate_total,
            pk_duplicate_samples: self.pk_duplicate_samples.clone(),
            dropped_rows_counts,
            trimmed_strings: if trimmed_strings > 0 {
                Some(trimmed_strings)
            } else {
                None
            },
            perf,
            num_shards: Some(self.num_shards),
        }
    }

}

fn profile_phase<T>(dir: &Path, name: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let guard = ProfilerGuard::new(500).context("start profiler")?;
    let result = f();
    let report = guard.report().build().context("build profile report")?;
    let svg_path = dir.join(format!("{name}-profile.svg"));
    let pb_path = dir.join(format!("{name}-profile.pb"));
    let file = fs::File::create(&svg_path).context("create profile output")?;
    report
        .flamegraph(file)
        .context("write flamegraph profile")?;
    let profile = report.pprof().context("build pprof profile")?;
    let mut file = fs::File::create(&pb_path).context("create pprof output")?;
    let bytes = profile.encode_to_vec();
    file.write_all(&bytes).context("write pprof profile")?;
    match result {
        Ok(value) => Ok(value),
        Err(err) => Err(err),
    }
}

fn edge_id_width(max_id: u64) -> u8 {
    let bits = 64 - max_id.leading_zeros();
    let bytes = ((bits + 7) / 8).max(1) as u8;
    bytes.min(8)
}

fn shard_range(min_id: u64, count: u64) -> (u64, u64) {
    if count == 0 {
        return (1, 0);
    }
    let max_id = min_id.saturating_add(count.saturating_sub(1));
    (min_id, max_id)
}

fn edge_binary_spec(
    src_max_id: u64,
    dst_max_id: u64,
    sort_bits: Option<u8>,
) -> EdgeBinarySpec {
    let sort_width = sort_bits.map(|b| ((b as u16 + 7) / 8) as u8).unwrap_or(0);
    EdgeBinarySpec {
        src_width: edge_id_width(src_max_id),
        dst_width: edge_id_width(dst_max_id),
        sort_width,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EdgeBinarySpec {
    src_width: u8,
    dst_width: u8,
    sort_width: u8,
}

fn temp_edge_map_shard_dir(name: &str, shard: usize) -> String {
    format!(
        "{}/edges_map/{}/shard-{:05}",
        TEMP_DIR_NAME,
        name.to_lowercase(),
        shard
    )
}

fn temp_edge_map_part_path(name: &str, shard: usize) -> String {
    format!("{}/part-00000.bin", temp_edge_map_shard_dir(name, shard))
}

fn temp_node_shard_dir(name: &str, shard: usize) -> String {
    format!(
        "{}/nodes/{}/shard-{:05}",
        TEMP_DIR_NAME,
        name.to_lowercase(),
        shard
    )
}

fn temp_node_data_path(name: &str, shard: usize) -> String {
    format!("{}/node_data.bin", temp_node_shard_dir(name, shard))
}

fn temp_node_data_meta_path(name: &str, shard: usize) -> String {
    format!("{}/node_data.meta", temp_node_shard_dir(name, shard))
}

fn temp_node_context_path(name: &str) -> String {
    format!("{}/nodes/{}/node_data.ctx", TEMP_DIR_NAME, name.to_lowercase())
}

fn node_context_output_path(name: &str) -> String {
    format!("nodes/{}/node_data.ctx", name.to_lowercase())
}

fn temp_pkey_path(name: &str, shard: usize) -> String {
    format!("{}/pkey.bin", temp_node_shard_dir(name, shard))
}

fn temp_pkey_meta_path(name: &str, shard: usize) -> String {
    format!("{}/pkey.meta", temp_node_shard_dir(name, shard))
}

fn temp_fkey_shard_dir(table: &str, rel: &str, shard: usize) -> String {
    format!(
        "{}/fkeys/{}/{}/shard-{:05}",
        TEMP_DIR_NAME,
        table.to_lowercase(),
        rel.to_lowercase(),
        shard
    )
}

fn temp_fkey_path(table: &str, rel: &str, shard: usize) -> String {
    format!("{}/rows.bin", temp_fkey_shard_dir(table, rel, shard))
}

fn temp_fkey_meta_path(table: &str, rel: &str, shard: usize) -> String {
    format!("{}/rows.meta", temp_fkey_shard_dir(table, rel, shard))
}

fn temp_fkey_bitmap_path(table: &str, rel: &str, shard: usize) -> String {
    format!("{}/bitmap.bin", temp_fkey_shard_dir(table, rel, shard))
}

fn temp_fkey_dst_shard_dir(table: &str, rel: &str, shard: usize) -> String {
    format!(
        "{}/fkeys_dst/{}/{}/shard-{:05}",
        TEMP_DIR_NAME,
        table.to_lowercase(),
        rel.to_lowercase(),
        shard
    )
}

fn temp_fkey_dst_path(table: &str, rel: &str, shard: usize) -> String {
    format!("{}/rows.bin", temp_fkey_dst_shard_dir(table, rel, shard))
}

fn temp_fkey_dst_meta_path(table: &str, rel: &str, shard: usize) -> String {
    format!("{}/rows.meta", temp_fkey_dst_shard_dir(table, rel, shard))
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum EdgeKeyKind {
    Src,
    Dst,
}

fn edge_key_kind_name(kind: EdgeKeyKind) -> &'static str {
    match kind {
        EdgeKeyKind::Src => "src",
        EdgeKeyKind::Dst => "dst",
    }
}

fn temp_edge_key_shard_dir(edge_table: &str, kind: EdgeKeyKind, shard: usize) -> String {
    format!(
        "{}/edge_keys/{}/{}/shard-{:05}",
        TEMP_DIR_NAME,
        edge_table.to_lowercase(),
        edge_key_kind_name(kind),
        shard
    )
}

fn temp_edge_key_path(edge_table: &str, kind: EdgeKeyKind, shard: usize) -> String {
    format!("{}/rows.bin", temp_edge_key_shard_dir(edge_table, kind, shard))
}

fn temp_edge_key_meta_path(edge_table: &str, kind: EdgeKeyKind, shard: usize) -> String {
    format!("{}/rows.meta", temp_edge_key_shard_dir(edge_table, kind, shard))
}

fn temp_edge_shard_dir(rel: &str, shard: usize) -> String {
    format!(
        "{}/edges_src/{}/shard-{:05}",
        TEMP_DIR_NAME,
        rel.to_lowercase(),
        shard
    )
}

fn temp_edge_path(rel: &str, shard: usize) -> String {
    format!("{}/rows.bin", temp_edge_shard_dir(rel, shard))
}

fn temp_edge_meta_path(rel: &str, shard: usize) -> String {
    format!("{}/rows.meta", temp_edge_shard_dir(rel, shard))
}

fn sort_key_component_name(kind: SortKeyComponentKind) -> &'static str {
    match kind {
        SortKeyComponentKind::Timestamp => "timestamp",
        SortKeyComponentKind::Weight => "weight",
    }
}

fn temp_sort_key_component_shard_dir(rel: &str, kind: SortKeyComponentKind, shard: usize) -> String {
    format!(
        "{}/sort_keys/{}/{}/shard-{:05}",
        TEMP_DIR_NAME,
        rel.to_lowercase(),
        sort_key_component_name(kind),
        shard
    )
}

fn temp_sort_key_component_path(rel: &str, kind: SortKeyComponentKind, shard: usize) -> String {
    format!(
        "{}/values.bin",
        temp_sort_key_component_shard_dir(rel, kind, shard)
    )
}

fn write_temp_meta(
    target: &OutputTarget,
    rel_path: &str,
    row_count: u64,
    max_row_len: u32,
) -> Result<()> {
    let mut payload = Vec::with_capacity(24);
    payload.extend_from_slice(&TEMP_META_MAGIC);
    payload.push(TEMP_META_VERSION);
    payload.push(0u8);
    payload.extend_from_slice(&row_count.to_le_bytes());
    payload.extend_from_slice(&max_row_len.to_le_bytes());
    let ctx_len = 0u32;
    payload.extend_from_slice(&ctx_len.to_le_bytes());
    target.write_bytes(rel_path, payload)
}

fn read_temp_meta(path: &Path) -> Result<TempRowpackMeta> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 4 + 1 + 1 + 8 + 4 + 4 {
        return Err(anyhow!("temp meta truncated"));
    }
    if bytes[0..4] != TEMP_META_MAGIC {
        return Err(anyhow!("temp meta magic mismatch"));
    }
    if bytes[4] != TEMP_META_VERSION {
        return Err(anyhow!("temp meta version mismatch"));
    }
    let row_count = u64::from_le_bytes(bytes[6..14].try_into().unwrap());
    let max_row_len = u32::from_le_bytes(bytes[14..18].try_into().unwrap());
    let ctx_len = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
    if bytes.len() < 22 + ctx_len {
        return Err(anyhow!("temp meta context truncated"));
    }
    Ok(TempRowpackMeta {
        row_count,
        max_row_len,
    })
}

fn open_temp_row_writer(target: &OutputTarget, rel_path: &str) -> Result<TempRowWriter> {
    let path = target.join(rel_path);
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    let sink = target.sink(&path)?;
    Ok(TempRowWriter {
        writer: FrameEncoder::new(sink),
        row_count: 0,
    })
}

fn open_sort_key_writer(target: &OutputTarget, rel_path: &str) -> Result<BufWriter<OutputSink>> {
    let path = target.join(rel_path);
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(BufWriter::with_capacity(4 * 1024 * 1024, target.sink(&path)?))
}

fn ensure_sort_key_file(target: &OutputTarget, rel_path: &str) -> Result<()> {
    if let OutputTarget::Local { .. } = target {
        let path = target.join(rel_path);
        if path.exists() {
            return Ok(());
        }
    }
    let mut writer = open_sort_key_writer(target, rel_path)?;
    writer.flush()?;
    let sink = writer
        .into_inner()
        .map_err(|_| anyhow!("sort key flush failed"))?;
    sink.finish()?;
    Ok(())
}

fn ensure_temp_file(target: &OutputTarget, rel_path: &str) -> Result<()> {
    if let OutputTarget::Local { .. } = target {
        let path = target.join(rel_path);
        if path.exists() {
            return Ok(());
        }
    }
    let writer = open_temp_row_writer(target, rel_path)?;
    finish_temp_writer(writer)?;
    Ok(())
}

fn open_fkey_bitmap_writer(target: &OutputTarget, rel_path: &str) -> Result<BufWriter<OutputSink>> {
    let path = target.join(rel_path);
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(BufWriter::with_capacity(4 * 1024 * 1024, target.sink(&path)?))
}

fn ensure_fkey_bitmap_file(target: &OutputTarget, rel_path: &str) -> Result<()> {
    if let OutputTarget::Local { .. } = target {
        let path = target.join(rel_path);
        if path.exists() {
            return Ok(());
        }
    }
    let mut writer = open_fkey_bitmap_writer(target, rel_path)?;
    writer.flush()?;
    let sink = writer
        .into_inner()
        .map_err(|_| anyhow!("fkey bitmap flush failed"))?;
    sink.finish()?;
    Ok(())
}

fn write_temp_rows_batch(writer: &mut TempRowWriter, data: &[u8], rows: u64) -> Result<()> {
    if !data.is_empty() {
        writer.writer.write_all(data)?;
    }
    writer.row_count = writer.row_count.saturating_add(rows);
    Ok(())
}

fn write_temp_rows_segments(
    writer: &mut TempRowWriter,
    batch: &RowPackSegmentBatch,
) -> Result<u32> {
    let mut tag_offset = 0usize;
    let mut bool_offset = 0usize;
    let mut data_offset = 0usize;
    let mut max_row_len = 0u32;
    for lens in &batch.rows {
        let row_len = lens.tag_bytes + lens.bool_bytes + lens.data_bytes;
        max_row_len = max_row_len.max(row_len as u32);
        let len = u32::try_from(row_len).map_err(|_| anyhow!("temp row too large"))?;
        writer.writer.write_all(&len.to_le_bytes())?;
        if lens.tag_bytes > 0 {
            writer.writer.write_all(
                &batch.tags[tag_offset..tag_offset.saturating_add(lens.tag_bytes)],
            )?;
        }
        if lens.bool_bytes > 0 {
            writer.writer.write_all(
                &batch.bools[bool_offset..bool_offset.saturating_add(lens.bool_bytes)],
            )?;
        }
        if lens.data_bytes > 0 {
            writer.writer.write_all(
                &batch.data[data_offset..data_offset.saturating_add(lens.data_bytes)],
            )?;
        }
        tag_offset = tag_offset.saturating_add(lens.tag_bytes);
        bool_offset = bool_offset.saturating_add(lens.bool_bytes);
        data_offset = data_offset.saturating_add(lens.data_bytes);
    }
    writer.row_count = writer.row_count.saturating_add(batch.rows.len() as u64);
    Ok(max_row_len)
}

fn finish_temp_writer(writer: TempRowWriter) -> Result<u64> {
    let sink = writer
        .writer
        .finish()
        .map_err(|e| anyhow!("temp writer finish failed: {e}"))?;
    sink.finish()?;
    Ok(writer.row_count)
}

fn append_temp_row(buf: &mut Vec<u8>, row: &[u8]) -> Result<()> {
    let len = row.len() as u32;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(row);
    Ok(())
}

fn append_fkey_bit(bits: &mut Vec<u8>, row_idx: u64, present: bool) -> usize {
    let byte_idx = (row_idx / 8) as usize;
    let bit = (row_idx % 8) as u8;
    let mut added = 0usize;
    if bits.len() <= byte_idx {
        bits.push(0);
        added = 1;
    }
    if present {
        bits[byte_idx] |= 1u8 << bit;
    }
    added
}

fn read_temp_rows(path: &Path, mut f: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    advise_file_willneed(&file);
    let mut file = FrameDecoder::new(file);
    let mut len_buf = [0u8; 4];
    let mut data = Vec::new();
    loop {
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(err) if err.kind() == std_io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(err) => return Err(err.into()),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        if data.len() < len {
            data.resize(len, 0u8);
        }
        let slice = &mut data[..len];
        file.read_exact(slice)?;
        f(slice)?;
    }
}

fn read_fkey_bitmap_mmap(path: &Path, row_count: u64) -> Result<Mmap> {
    let expected = ((row_count + 7) / 8) as usize;
    let file = fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let map = unsafe { Mmap::map(&file)? };
    if map.len() < expected {
        return Err(anyhow!(
            "fkey bitmap length mismatch: expected {expected}, got {}",
            map.len()
        ));
    }
    advise_sequential(&map);
    Ok(map)
}

fn load_sort_key_component_shards(
    root: &Path,
    rel_name: &str,
    kind: SortKeyComponentKind,
    num_shards: usize,
    width_bytes: usize,
    sequential: bool,
) -> Result<Vec<SortKeyComponentStore>> {
    let mut shards = Vec::with_capacity(num_shards);
    for shard in 0..num_shards {
        let path = root.join(temp_sort_key_component_path(rel_name, kind, shard));
        let file = fs::File::open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        let data = unsafe { Mmap::map(&file)? };
        if sequential {
            advise_sequential(&data);
        } else {
            advise_willneed(&data);
        }
        shards.push(SortKeyComponentStore {
            width_bytes,
            data,
            is_edge: sequential,
        });
    }
    Ok(shards)
}

fn advise_willneed(map: &Mmap) {
    let ret = unsafe {
        libc::madvise(
            map.as_ptr() as *mut libc::c_void,
            map.len(),
            libc::MADV_WILLNEED,
        )
    };
    if ret != 0 {
        let _ = std_io::Error::last_os_error();
    }
}

fn advise_sequential(map: &Mmap) {
    let ret = unsafe {
        libc::madvise(
            map.as_ptr() as *mut libc::c_void,
            map.len(),
            libc::MADV_SEQUENTIAL,
        )
    };
    if ret != 0 {
        let _ = std_io::Error::last_os_error();
    }
}

#[cfg(target_os = "linux")]
fn advise_file_willneed(file: &fs::File) {
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_file_willneed(_file: &fs::File) {}

fn build_sort_key_lookup(
    root: &Path,
    edge_tables: &HashSet<String>,
    rel_sort_keys: &HashMap<String, RelationSortKeyInfo>,
    rel_name: &str,
    num_shards: usize,
) -> Result<Option<SortKeyLookup>> {
    let info = match rel_sort_keys.get(rel_name) {
        Some(info) => info,
        None => return Ok(None),
    };
    let mut timestamp = None;
    let mut weight = None;
    if let Some(ts) = &info.timestamp {
        let width_bytes = (ts.spec.width_bits / 8) as usize;
        let sequential = edge_tables.contains(&ts.spec.table);
        let shards = load_sort_key_component_shards(
            root,
            rel_name,
            ts.spec.kind,
            num_shards,
            width_bytes,
            sequential,
        )?;
        let source = if sequential {
            SortKeyComponentSource::EdgeTable { shards }
        } else {
            SortKeyComponentSource::DestTable { shards }
        };
        timestamp = Some(source);
    }
    if let Some(w) = &info.weight {
        let width_bytes = (w.spec.width_bits / 8) as usize;
        let sequential = edge_tables.contains(&w.spec.table);
        let shards = load_sort_key_component_shards(
            root,
            rel_name,
            w.spec.kind,
            num_shards,
            width_bytes,
            sequential,
        )?;
        let source = if sequential {
            SortKeyComponentSource::EdgeTable { shards }
        } else {
            SortKeyComponentSource::DestTable { shards }
        };
        weight = Some(source);
    }
    Ok(Some(SortKeyLookup {
        weight_bits: info
            .weight
            .as_ref()
            .map(|w| w.spec.width_bits)
            .unwrap_or(0),
        timestamp,
        weight,
    }))
}

fn write_edge_bin_header(writer: &mut BufWriter<OutputSink>, spec: EdgeBinarySpec) -> Result<()> {
    writer.write_all(&[spec.src_width, spec.dst_width, spec.sort_width])?;
    Ok(())
}

fn push_u64_width(buf: &mut Vec<u8>, value: u64, width: u8) {
    let bytes = value.to_le_bytes();
    buf.extend_from_slice(&bytes[..width as usize]);
}

fn append_edge_record(
    buf: &mut Vec<u8>,
    spec: EdgeBinarySpec,
    src_id: u64,
    dst_id: u64,
    sort_key: u64,
) {
    push_u64_width(buf, src_id, spec.src_width);
    push_u64_width(buf, dst_id, spec.dst_width);
    if spec.sort_width > 0 {
        push_u64_width(buf, sort_key, spec.sort_width);
    }
}

struct EdgePartWriter {
    writer: BufWriter<OutputSink>,
    spec: EdgeBinarySpec,
    row_count: u64,
}

impl EdgePartWriter {
    fn open(target: &OutputTarget, path: &Path, spec: EdgeBinarySpec) -> Result<Self> {
        if let OutputTarget::Local { .. } = target {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, target.sink(path)?);
        write_edge_bin_header(&mut writer, spec)?;
        Ok(Self {
            writer,
            spec,
            row_count: 0,
        })
    }

    fn write_batch(&mut self, data: &[u8], rows: u64) -> Result<()> {
        self.writer.write_all(data)?;
        self.row_count = self.row_count.saturating_add(rows);
        Ok(())
    }

    fn finish(mut self) -> Result<u64> {
        self.writer.flush()?;
        let sink = self
            .writer
            .into_inner()
            .map_err(|_| anyhow!("edge part flush failed"))?;
        sink.finish()?;
        Ok(self.row_count)
    }
}

const EDGE_WRITE_QUEUE_DEPTH: usize = 128;
const EDGE_BATCH_MIN_BYTES: usize = 64 * 1024;
const EDGE_BATCH_TARGET_BYTES: usize = 8 * 1024 * 1024;
const EDGE_BATCH_MAX_BYTES: usize = 2 * 1024 * 1024;

fn edge_batch_buffer_limit(num_shards: usize) -> usize {
    let per = EDGE_BATCH_TARGET_BYTES / num_shards.max(1);
    let per = per.max(EDGE_BATCH_MIN_BYTES);
    per.min(EDGE_BATCH_MAX_BYTES)
}

fn edge_writer_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn edge_writer_index(relation_hash: u64, shard: usize, workers: usize) -> usize {
    let mixed = relation_hash ^ (shard as u64);
    (mixed % workers as u64) as usize
}

#[derive(Clone)]
struct EdgeWriteBatch {
    relation: std::sync::Arc<str>,
    shard: usize,
    spec: EdgeBinarySpec,
    data: Vec<u8>,
    row_count: u64,
}

enum EdgeWriteMessage {
    Batch(EdgeWriteBatch),
}

struct EdgeWriterPool {
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeWriteMessage>>>,
    handles: Vec<std::thread::JoinHandle<Result<()>>>,
}

impl EdgeWriterPool {
    fn new(
        target: std::sync::Arc<OutputTarget>,
        workers: usize,
        buffer_pool: std::sync::Arc<EdgeBufferPool>,
    ) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) = std::sync::mpsc::sync_channel::<EdgeWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            let buffer_pool = std::sync::Arc::clone(&buffer_pool);
            handles.push(std::thread::spawn(move || edge_writer_worker(target, buffer_pool, rx)));
            senders.push(tx);
        }
        Self {
            senders: std::sync::Arc::new(senders),
            handles,
        }
    }

    fn senders(&self) -> std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeWriteMessage>>> {
        std::sync::Arc::clone(&self.senders)
    }

    fn finish(self) -> Result<()> {
        drop(self.senders);
        for handle in self.handles {
            match handle.join() {
                Ok(result) => result?,
                Err(_) => return Err(anyhow!("edge writer thread panicked")),
            }
        }
        Ok(())
    }
}

fn edge_writer_worker(
    target: std::sync::Arc<OutputTarget>,
    buffer_pool: std::sync::Arc<EdgeBufferPool>,
    rx: std::sync::mpsc::Receiver<EdgeWriteMessage>,
) -> Result<()> {
    let mut writers: HashMap<(std::sync::Arc<str>, usize), EdgePartWriter> = HashMap::new();
    while let Ok(msg) = rx.recv() {
        match msg {
            EdgeWriteMessage::Batch(batch) => {
                let key = (std::sync::Arc::clone(&batch.relation), batch.shard);
                if let Some(writer) = writers.get_mut(&key) {
                    if writer.spec != batch.spec {
                        return Err(anyhow!(
                            "edge spec mismatch for {} shard {}",
                            batch.relation,
                            batch.shard
                        ));
                    }
                    writer.write_batch(&batch.data, batch.row_count)?;
                    buffer_pool.release(batch.data);
                    continue;
                }
                let path = target.join(&temp_edge_map_part_path(&batch.relation, batch.shard));
                let mut writer = EdgePartWriter::open(&target, &path, batch.spec)?;
                writer.write_batch(&batch.data, batch.row_count)?;
                buffer_pool.release(batch.data);
                writers.insert(key, writer);
            }
        }
    }
    for (_, writer) in writers {
        writer.finish()?;
    }
    Ok(())
}

struct EdgeBuffer {
    data: Vec<u8>,
    row_count: u64,
}

struct EdgeBufferPool {
    buffers: std::sync::Mutex<Vec<Vec<u8>>>,
}

impl EdgeBufferPool {
    fn new() -> Self {
        Self {
            buffers: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn take_with_capacity(&self, min_capacity: usize) -> Vec<u8> {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(mut buf) = buffers.pop() {
            if buf.capacity() < min_capacity {
                buf.reserve(min_capacity - buf.capacity());
            }
            buf.clear();
            return buf;
        }
        Vec::with_capacity(min_capacity)
    }

    fn release(&self, mut buf: Vec<u8>) {
        buf.clear();
        let mut buffers = self.buffers.lock().unwrap();
        buffers.push(buf);
    }
}

struct EdgeBatcher {
    relation: std::sync::Arc<str>,
    relation_hash: u64,
    spec: EdgeBinarySpec,
    buffers: Vec<EdgeBuffer>,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeWriteMessage>>>,
    buffer_pool: std::sync::Arc<EdgeBufferPool>,
}

impl EdgeBatcher {
    fn new(
        relation: std::sync::Arc<str>,
        spec: EdgeBinarySpec,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeWriteMessage>>>,
        buffer_pool: std::sync::Arc<EdgeBufferPool>,
    ) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        relation.hash(&mut hasher);
        let relation_hash = hasher.finish();
        let buffers = (0..num_shards)
            .map(|_| EdgeBuffer {
                data: buffer_pool.take_with_capacity(buffer_limit),
                row_count: 0,
            })
            .collect();
        Self {
            relation,
            relation_hash,
            spec,
            buffers,
            buffer_limit,
            senders,
            buffer_pool,
        }
    }

    fn push(&mut self, shard: usize, src_id: u64, dst_id: u64, sort_key: u64) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("edge buffer shard {} out of bounds", shard))?;
        append_edge_record(&mut buffer.data, self.spec, src_id, dst_id, sort_key);
        buffer.row_count = buffer.row_count.saturating_add(1);
        if buffer.data.len() >= self.buffer_limit {
            self.flush_shard(shard)?;
        }
        Ok(())
    }

    fn flush_shard(&mut self, shard: usize) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("edge buffer shard {} out of bounds", shard))?;
        if buffer.row_count == 0 {
            return Ok(());
        }
        let mut data = self
            .buffer_pool
            .take_with_capacity(buffer.data.capacity().max(self.buffer_limit));
        std::mem::swap(&mut data, &mut buffer.data);
        let batch = EdgeWriteBatch {
            relation: std::sync::Arc::clone(&self.relation),
            shard,
            spec: self.spec,
            data,
            row_count: buffer.row_count,
        };
        buffer.row_count = 0;
        let idx = edge_writer_index(self.relation_hash, shard, self.senders.len());
        self.senders[idx]
            .send(EdgeWriteMessage::Batch(batch))
            .map_err(|e| anyhow!("edge writer channel error: {e}"))?;
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for shard in 0..self.buffers.len() {
            self.flush_shard(shard)?;
        }
        Ok(())
    }
}

impl Drop for EdgeBatcher {
    fn drop(&mut self) {
        for buffer in &mut self.buffers {
            if !buffer.data.is_empty() {
                buffer.data.clear();
            }
            let data = std::mem::take(&mut buffer.data);
            self.buffer_pool.release(data);
        }
    }
}

fn hash_name(name: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

struct NodeWriteBatch {
    table: std::sync::Arc<str>,
    rel_names: std::sync::Arc<Vec<std::sync::Arc<str>>>,
    shard: usize,
    node_data: RowPackSegmentBatch,
    pkey_data: Option<Vec<u8>>,
    pkey_rows: u64,
    fkey_data: Vec<Vec<u8>>,
    fkey_bits: Vec<Vec<u8>>,
    fkey_rows: Vec<u64>,
}

struct RowPackSegmentBatch {
    tags: Vec<u8>,
    bools: Vec<u8>,
    data: Vec<u8>,
    rows: Vec<RowPackRowLens>,
}

enum NodeWriteMessage {
    Batch(NodeWriteBatch),
}

#[derive(Default)]
struct NodeWriterStats {
    node_rows: HashMap<(String, usize), u64>,
    pkey_rows: HashMap<(String, usize), u64>,
    fkey_rows: HashMap<(String, String, usize), u64>,
    node_data_max_len: HashMap<(String, usize), u32>,
}

impl NodeWriterStats {
    fn merge(&mut self, other: NodeWriterStats) {
        for (key, value) in other.node_rows {
            let entry = self.node_rows.entry(key).or_insert(0);
            *entry = entry.saturating_add(value);
        }
        for (key, value) in other.pkey_rows {
            let entry = self.pkey_rows.entry(key).or_insert(0);
            *entry = entry.saturating_add(value);
        }
        for (key, value) in other.fkey_rows {
            let entry = self.fkey_rows.entry(key).or_insert(0);
            *entry = entry.saturating_add(value);
        }
        for (key, value) in other.node_data_max_len {
            let entry = self.node_data_max_len.entry(key).or_insert(0);
            *entry = (*entry).max(value);
        }
    }
}

struct BitWriter {
    writer: BufWriter<OutputSink>,
    buffer: u8,
    bits_used: u8,
}

impl BitWriter {
    fn new(writer: BufWriter<OutputSink>) -> Self {
        Self {
            writer,
            buffer: 0,
            bits_used: 0,
        }
    }

    fn append_bits(&mut self, data: &[u8], count: u64) -> Result<()> {
        let mut bits_remaining = count;
        let mut byte_idx = 0;
        while bits_remaining > 0 {
            let byte = data[byte_idx];
            byte_idx += 1;
            let take = bits_remaining.min(8) as u8;

            // If we are byte-aligned and taking a full byte, fast path
            if self.bits_used == 0 && take == 8 {
                self.writer.write_all(&[byte])?;
                bits_remaining -= 8;
                continue;
            }

            // Append 'take' bits from 'byte' to self.buffer
            // Bits in 'byte' are filled from LSB (bit 0) to MSB (bit 7).
            // We want to preserve this order.
            // Existing buffer has 'bits_used' bits.
            // We append starting at 'bits_used'.

            // Mask for the bits we are taking from the input byte
            let mask = if take == 8 { 0xFF } else { (1u8 << take) - 1 };
            let val = byte & mask;

            self.buffer |= val << self.bits_used;
            let space = 8 - self.bits_used;

            if take >= space {
                // Buffer is full
                self.writer.write_all(&[self.buffer])?;
                // Remaining bits from 'val' that didn't fit
                let written = space;
                let leftover = take - written;
                if leftover > 0 {
                    self.buffer = val >> written;
                    self.bits_used = leftover;
                } else {
                    self.buffer = 0;
                    self.bits_used = 0;
                }
            } else {
                self.bits_used += take;
            }
            bits_remaining -= take as u64;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if self.bits_used > 0 {
            self.writer.write_all(&[self.buffer])?;
        }
        self.writer.flush()?;
        let sink = self.writer.into_inner().map_err(|_| anyhow!("bit writer flush failed"))?;
        sink.finish()?;
        Ok(())
    }
}

struct NodeShardWriter {
    node: TempRowWriter,
    pkey: Option<TempRowWriter>,
    fkeys: Vec<TempRowWriter>,
    fkey_bits: Vec<BitWriter>,
    rel_names: std::sync::Arc<Vec<std::sync::Arc<str>>>,
}

struct NodeWritePool {
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>>,
    handles: Vec<std::thread::JoinHandle<Result<NodeWriterStats>>>,
}

impl NodeWritePool {
    fn new(target: std::sync::Arc<OutputTarget>, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) = std::sync::mpsc::sync_channel::<NodeWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            handles.push(std::thread::spawn(move || node_writer_worker(target, rx)));
            senders.push(tx);
        }
        Self {
            senders: std::sync::Arc::new(senders),
            handles,
        }
    }

    fn senders(&self) -> std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>> {
        std::sync::Arc::clone(&self.senders)
    }

    fn finish(self) -> Result<NodeWriterStats> {
        drop(self.senders);
        let mut stats = NodeWriterStats::default();
        for handle in self.handles {
            match handle.join() {
                Ok(result) => {
                    stats.merge(result?);
                }
                Err(_) => return Err(anyhow!("node writer thread panicked")),
            }
        }
        Ok(stats)
    }
}

fn node_writer_worker(
    target: std::sync::Arc<OutputTarget>,
    rx: std::sync::mpsc::Receiver<NodeWriteMessage>,
) -> Result<NodeWriterStats> {
    let mut writers: HashMap<(std::sync::Arc<str>, usize), NodeShardWriter> = HashMap::new();
    let mut stats = NodeWriterStats::default();
    while let Ok(msg) = rx.recv() {
        match msg {
            NodeWriteMessage::Batch(batch) => {
                let key = (std::sync::Arc::clone(&batch.table), batch.shard);
                let writer = match writers.get_mut(&key) {
                    Some(writer) => writer,
                    None => {
                        let node =
                            open_temp_row_writer(&target, &temp_node_data_path(&batch.table, batch.shard))?;
                        let pkey = if batch.pkey_data.is_some() {
                            Some(open_temp_row_writer(
                                &target,
                                &temp_pkey_path(&batch.table, batch.shard),
                            )?)
                        } else {
                            None
                        };
                        let mut fkeys = Vec::with_capacity(batch.rel_names.len());
                        let mut fkey_bits = Vec::with_capacity(batch.rel_names.len());
                        for rel in batch.rel_names.iter() {
                            fkeys.push(open_temp_row_writer(
                                &target,
                                &temp_fkey_path(&batch.table, rel, batch.shard),
                            )?);
                            fkey_bits.push(BitWriter::new(open_fkey_bitmap_writer(
                                &target,
                                &temp_fkey_bitmap_path(&batch.table, rel, batch.shard),
                            )?));
                        }
                        writers.insert(
                            key.clone(),
                            NodeShardWriter {
                                node,
                                pkey,
                                fkeys,
                                fkey_bits,
                                rel_names: std::sync::Arc::clone(&batch.rel_names),
                            },
                        );
                        writers.get_mut(&key).unwrap()
                    }
                };
                if writer.rel_names.len() != batch.rel_names.len() {
                    return Err(anyhow!(
                        "node writer relation count mismatch for {} shard {}",
                        batch.table,
                        batch.shard
                    ));
                }
                let batch_max_len = write_temp_rows_segments(&mut writer.node, &batch.node_data)?;
                let node_key = (batch.table.to_string(), batch.shard);
                let entry = stats.node_rows.entry(node_key).or_insert(0);
                *entry = entry.saturating_add(batch.node_data.rows.len() as u64);
                let max_key = (batch.table.to_string(), batch.shard);
                let entry = stats.node_data_max_len.entry(max_key).or_insert(0);
                *entry = (*entry).max(batch_max_len);
                if let Some(data) = batch.pkey_data {
                    if writer.pkey.is_none() {
                        writer.pkey = Some(open_temp_row_writer(
                            &target,
                            &temp_pkey_path(&batch.table, batch.shard),
                        )?);
                    }
                    if let Some(pkey_writer) = writer.pkey.as_mut() {
                        write_temp_rows_batch(pkey_writer, &data, batch.pkey_rows)?;
                    }
                    let pkey_key = (batch.table.to_string(), batch.shard);
                    let entry = stats.pkey_rows.entry(pkey_key).or_insert(0);
                    *entry = entry.saturating_add(batch.pkey_rows);
                }
                for (idx, fkey_bytes) in batch.fkey_data.into_iter().enumerate() {
                    if let Some(writer_fkey) = writer.fkeys.get_mut(idx) {
                        let rows = batch.fkey_rows.get(idx).copied().unwrap_or(0);
                        write_temp_rows_batch(writer_fkey, &fkey_bytes, rows)?;
                        let rel = batch
                            .rel_names
                            .get(idx)
                            .map(|name| name.to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        let key = (batch.table.to_string(), rel, batch.shard);
                        let entry = stats.fkey_rows.entry(key).or_insert(0);
                        *entry = entry.saturating_add(rows);
                    }
                }
                for (idx, bits) in batch.fkey_bits.into_iter().enumerate() {
                    if let Some(writer_bits) = writer.fkey_bits.get_mut(idx) {
                        let rows = batch.fkey_rows.get(idx).copied().unwrap_or(0);
                        writer_bits.append_bits(&bits, rows)?;
                    }
                }
            }
        }
    }
    for (_, mut writer) in writers {
        finish_temp_writer(writer.node)?;
        if let Some(pkey) = writer.pkey.take() {
            finish_temp_writer(pkey)?;
        }
        for fkey in writer.fkeys {
            finish_temp_writer(fkey)?;
        }
        for bits in writer.fkey_bits {
            bits.finish()?;
        }
    }
    Ok(stats)
}

struct NodeShardBuffer {
    node: RowPackSegmentBatch,
    pkey: Option<Vec<u8>>,
    pkey_rows: u64,
    fkeys: Vec<Vec<u8>>,
    fkey_bits: Vec<Vec<u8>>,
    fkey_rows: Vec<u64>,
    bytes: usize,
}

struct NodeBatcher {
    table: std::sync::Arc<str>,
    table_hash: u64,
    rel_names: std::sync::Arc<Vec<std::sync::Arc<str>>>,
    buffers: Vec<NodeShardBuffer>,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>>,
}

struct NodeRowSink<'a> {
    tags: &'a mut Vec<u8>,
    bools: &'a mut Vec<u8>,
    data: &'a mut Vec<u8>,
}

impl NodeBatcher {
    fn new(
        table: std::sync::Arc<str>,
        rel_names: std::sync::Arc<Vec<std::sync::Arc<str>>>,
        has_pkey: bool,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>>,
    ) -> Self {
        let table_hash = hash_name(&table);
        let buffers = (0..num_shards)
            .map(|_| NodeShardBuffer {
                node: RowPackSegmentBatch {
                    tags: Vec::new(),
                    bools: Vec::new(),
                    data: Vec::new(),
                    rows: Vec::new(),
                },
                pkey: if has_pkey {
                    Some(Vec::new())
                } else {
                    None
                },
                pkey_rows: 0,
                fkeys: vec![Vec::new(); rel_names.len()],
                fkey_bits: vec![Vec::new(); rel_names.len()],
                fkey_rows: vec![0; rel_names.len()],
                bytes: 0,
            })
            .collect();
        Self {
            table,
            table_hash,
            rel_names,
            buffers,
            buffer_limit,
            senders,
        }
    }

    fn node_sink(&mut self, shard: usize) -> Result<NodeRowSink<'_>> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        Ok(NodeRowSink {
            tags: &mut buffer.node.tags,
            bools: &mut buffer.node.bools,
            data: &mut buffer.node.data,
        })
    }

    fn push_node_row(&mut self, shard: usize, lens: RowPackRowLens) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        let row_len = lens
            .tag_bytes
            .saturating_add(lens.bool_bytes)
            .saturating_add(lens.data_bytes);
        let _ = u16::try_from(row_len).map_err(|_| anyhow!("temp row too large"))?;
        buffer.node.rows.push(lens);
        buffer.bytes = buffer.bytes.saturating_add(2 + row_len);
        Ok(())
    }

    fn push_empty_node(&mut self, shard: usize) -> Result<()> {
        self.push_node_row(
            shard,
            RowPackRowLens {
                tag_bytes: 0,
                bool_bytes: 0,
                data_bytes: 0,
            },
        )
    }

    fn push_pkey(&mut self, shard: usize, row: &[u8], timestamp_valid: bool) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("pkey buffer shard {} out of bounds", shard))?;
        if let Some(pkey) = buffer.pkey.as_mut() {
            let total_len = (row.len() + 1) as u32;
            pkey.extend_from_slice(&total_len.to_le_bytes());
            pkey.extend_from_slice(row);
            pkey.push(if timestamp_valid { 1 } else { 0 });
            buffer.pkey_rows = buffer.pkey_rows.saturating_add(1);
            buffer.bytes = buffer.bytes.saturating_add(4 + row.len() + 1);
        }
        Ok(())
    }

    fn push_fkey(&mut self, shard: usize, rel_idx: usize, row: &[u8], timestamp_valid: bool) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("fkey buffer shard {} out of bounds", shard))?;
        if let Some(fkey) = buffer.fkeys.get_mut(rel_idx) {
            let row_idx = buffer.fkey_rows.get(rel_idx).copied().unwrap_or(0);
            append_temp_row(fkey, row)?;
            if let Some(count) = buffer.fkey_rows.get_mut(rel_idx) {
                *count = count.saturating_add(1);
            }
            if let Some(bits) = buffer.fkey_bits.get_mut(rel_idx) {
                let added = append_fkey_bit(bits, row_idx, timestamp_valid);
                buffer.bytes = buffer.bytes.saturating_add(added);
            }
            buffer.bytes = buffer.bytes.saturating_add(4 + row.len());
        }
        Ok(())
    }

    fn push_fkey_empty(&mut self, shard: usize, rel_idx: usize) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("fkey buffer shard {} out of bounds", shard))?;
        if let Some(fkey) = buffer.fkeys.get_mut(rel_idx) {
            let row_idx = buffer.fkey_rows.get(rel_idx).copied().unwrap_or(0);
            append_temp_row(fkey, &[])?;
            if let Some(count) = buffer.fkey_rows.get_mut(rel_idx) {
                *count = count.saturating_add(1);
            }
            if let Some(bits) = buffer.fkey_bits.get_mut(rel_idx) {
                let added = append_fkey_bit(bits, row_idx, false);
                buffer.bytes = buffer.bytes.saturating_add(added);
            }
            buffer.bytes = buffer.bytes.saturating_add(4);
        }
        Ok(())
    }

    fn maybe_flush(&mut self, shard: usize) -> Result<()> {
        let buffer = self
            .buffers
            .get(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        if buffer.bytes >= self.buffer_limit {
            self.flush_shard(shard)?;
        }
        Ok(())
    }

    fn flush_shard(&mut self, shard: usize) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        if buffer.node.rows.is_empty()
            && buffer.pkey_rows == 0
            && buffer.fkey_rows.iter().all(|r| *r == 0)
        {
            return Ok(());
        }
        let mut node = RowPackSegmentBatch {
            tags: Vec::with_capacity(buffer.node.tags.capacity()),
            bools: Vec::with_capacity(buffer.node.bools.capacity()),
            data: Vec::with_capacity(buffer.node.data.capacity()),
            rows: Vec::with_capacity(buffer.node.rows.capacity()),
        };
        std::mem::swap(&mut node, &mut buffer.node);
        let pkey = buffer.pkey.as_mut().map(|src| {
            let mut data = Vec::with_capacity(src.capacity());
            std::mem::swap(&mut data, src);
            data
        });
        let pkey_rows = buffer.pkey_rows;
        let mut fkeys = Vec::with_capacity(buffer.fkeys.len());
        for mut entry in buffer.fkeys.drain(..) {
            let mut data = Vec::with_capacity(entry.capacity());
            std::mem::swap(&mut data, &mut entry);
            fkeys.push(data);
        }
        let mut fkey_bits = Vec::with_capacity(buffer.fkey_bits.len());
        for mut entry in buffer.fkey_bits.drain(..) {
            let mut data = Vec::with_capacity(entry.capacity());
            std::mem::swap(&mut data, &mut entry);
            fkey_bits.push(data);
        }
        let fkey_rows = buffer.fkey_rows.clone();
        let batch = NodeWriteBatch {
            table: std::sync::Arc::clone(&self.table),
            rel_names: std::sync::Arc::clone(&self.rel_names),
            shard,
            node_data: node,
            pkey_data: pkey,
            pkey_rows,
            fkey_data: fkeys,
            fkey_bits,
            fkey_rows,
        };
        buffer.node.rows.clear();
        buffer.pkey_rows = 0;
        buffer.fkey_rows = vec![0; self.rel_names.len()];
        buffer.fkeys = vec![Vec::new(); self.rel_names.len()];
        buffer.fkey_bits = vec![Vec::new(); self.rel_names.len()];
        buffer.bytes = 0;
        let idx = edge_writer_index(self.table_hash, shard, self.senders.len());
        self.senders[idx]
            .send(NodeWriteMessage::Batch(batch))
            .map_err(|e| anyhow!("node writer channel error: {e}"))?;
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for shard in 0..self.buffers.len() {
            self.flush_shard(shard)?;
        }
        Ok(())
    }
}

struct SortKeyWriteBatch {
    rel: std::sync::Arc<str>,
    kind: SortKeyComponentKind,
    shard: usize,
    data: Vec<u8>,
    row_count: u64,
    width_bytes: usize,
}

enum SortKeyWriteMessage {
    Batch(SortKeyWriteBatch),
}

#[derive(Default)]
struct SortKeyWriterStats {
    rows: HashMap<(String, SortKeyComponentKind, usize), u64>,
}

impl SortKeyWriterStats {
    fn merge(&mut self, other: SortKeyWriterStats) {
        for (key, value) in other.rows {
            let entry = self.rows.entry(key).or_insert(0);
            *entry = entry.saturating_add(value);
        }
    }
}

struct SortKeyShardWriter {
    writer: BufWriter<OutputSink>,
    width_bytes: usize,
}

struct SortKeyWritePool {
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>>,
    handles: Vec<std::thread::JoinHandle<Result<SortKeyWriterStats>>>,
}

impl SortKeyWritePool {
    fn new(target: std::sync::Arc<OutputTarget>, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) =
                std::sync::mpsc::sync_channel::<SortKeyWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            handles.push(std::thread::spawn(move || sort_key_writer_worker(target, rx)));
            senders.push(tx);
        }
        Self {
            senders: std::sync::Arc::new(senders),
            handles,
        }
    }

    fn senders(&self) -> std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>> {
        std::sync::Arc::clone(&self.senders)
    }

    fn finish(self) -> Result<SortKeyWriterStats> {
        drop(self.senders);
        let mut stats = SortKeyWriterStats::default();
        for handle in self.handles {
            match handle.join() {
                Ok(result) => {
                    stats.merge(result?);
                }
                Err(_) => return Err(anyhow!("sort key writer thread panicked")),
            }
        }
        Ok(stats)
    }
}

fn sort_key_writer_worker(
    target: std::sync::Arc<OutputTarget>,
    rx: std::sync::mpsc::Receiver<SortKeyWriteMessage>,
) -> Result<SortKeyWriterStats> {
    let mut writers: HashMap<(std::sync::Arc<str>, SortKeyComponentKind, usize), SortKeyShardWriter> =
        HashMap::new();
    let mut stats = SortKeyWriterStats::default();
    while let Ok(msg) = rx.recv() {
        match msg {
            SortKeyWriteMessage::Batch(batch) => {
                let key = (
                    std::sync::Arc::clone(&batch.rel),
                    batch.kind,
                    batch.shard,
                );
                let writer = match writers.get_mut(&key) {
                    Some(writer) => writer,
                    None => {
                        let path = temp_sort_key_component_path(&batch.rel, batch.kind, batch.shard);
                        let writer = open_sort_key_writer(&target, &path)?;
                        writers.insert(
                            key.clone(),
                            SortKeyShardWriter {
                                writer,
                                width_bytes: batch.width_bytes,
                            },
                        );
                        writers.get_mut(&key).unwrap()
                    }
                };
                if writer.width_bytes != batch.width_bytes {
                    return Err(anyhow!(
                        "sort key width mismatch for {} {:?} shard {}",
                        batch.rel,
                        batch.kind,
                        batch.shard
                    ));
                }
                writer.writer.write_all(&batch.data)?;
                let entry = stats
                    .rows
                    .entry((batch.rel.to_string(), batch.kind, batch.shard))
                    .or_insert(0);
                *entry = entry.saturating_add(batch.row_count);
            }
        }
    }
    for (_, mut writer) in writers {
        writer.writer.flush()?;
        let sink = writer
            .writer
            .into_inner()
            .map_err(|_| anyhow!("sort key flush failed"))?;
        sink.finish()?;
    }
    Ok(stats)
}

struct SortKeyComponentBatcher {
    rel: std::sync::Arc<str>,
    rel_hash: u64,
    kind: SortKeyComponentKind,
    width_bytes: u8,
    encoder: SortKeyComponentEncoder,
    buffers: Vec<Vec<u8>>,
    rows: Vec<u64>,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>>,
}

impl SortKeyComponentBatcher {
    fn new(
        rel: std::sync::Arc<str>,
        kind: SortKeyComponentKind,
        width_bytes: u8,
        encoder: SortKeyComponentEncoder,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>>,
    ) -> Self {
        let rel_hash = hash_name(&rel);
        Self {
            rel,
            rel_hash,
            kind,
            width_bytes,
            encoder,
            buffers: vec![Vec::new(); num_shards],
            rows: vec![0; num_shards],
            buffer_limit,
            senders,
        }
    }

    fn push_row(&mut self, shard: usize, access: &mut RowAccess<'_>) -> Result<()> {
        let value = match &mut self.encoder {
            SortKeyComponentEncoder::Timestamp { column_idx, encoding } => {
                let ts_value = if let Some((secs, _)) =
                    access.get_timestamp(*column_idx).ok().flatten()
                {
                    Some(secs)
                } else {
                    access
                        .get_date(*column_idx)
                        .ok()
                        .flatten()
                        .map(|days| i64::from(days) * 86_400)
                };
                encode_timestamp_value(ts_value, encoding)
            }
            SortKeyComponentEncoder::Weight { expr, encoding } => {
                let weight = eval_weight_expr(expr, access)?;
                encode_weight_value(weight, encoding)
            }
        };
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("sort key buffer shard {} out of bounds", shard))?;
        push_u64_width(buffer, value, self.width_bytes);
        if let Some(row_count) = self.rows.get_mut(shard) {
            *row_count = row_count.saturating_add(1);
        }
        if buffer.len() >= self.buffer_limit {
            self.flush_shard(shard)?;
        }
        Ok(())
    }

    fn flush_shard(&mut self, shard: usize) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("sort key buffer shard {} out of bounds", shard))?;
        let rows = self.rows.get_mut(shard).ok_or_else(|| anyhow!("sort key row count missing"))?;
        if *rows == 0 {
            return Ok(());
        }
        let mut data = Vec::with_capacity(buffer.capacity());
        std::mem::swap(&mut data, buffer);
        let batch = SortKeyWriteBatch {
            rel: std::sync::Arc::clone(&self.rel),
            kind: self.kind,
            shard,
            data,
            row_count: *rows,
            width_bytes: self.width_bytes as usize,
        };
        *rows = 0;
        let idx = edge_writer_index(self.rel_hash, shard, self.senders.len());
        self.senders[idx]
            .send(SortKeyWriteMessage::Batch(batch))
            .map_err(|e| anyhow!("sort key writer channel error: {e}"))?;
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for shard in 0..self.buffers.len() {
            self.flush_shard(shard)?;
        }
        Ok(())
    }
}

struct EdgeSrcWriteBatch {
    rel: std::sync::Arc<str>,
    shard: usize,
    data: Vec<u8>,
    row_count: u64,
}

enum EdgeSrcWriteMessage {
    Batch(EdgeSrcWriteBatch),
}

#[derive(Default)]
struct EdgeSrcWriterStats {
    rows: HashMap<(String, usize), u64>,
}

impl EdgeSrcWriterStats {
    fn merge(&mut self, other: EdgeSrcWriterStats) {
        for (key, value) in other.rows {
            let entry = self.rows.entry(key).or_insert(0);
            *entry = entry.saturating_add(value);
        }
    }
}

struct EdgeSrcWritePool {
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>>,
    handles: Vec<std::thread::JoinHandle<Result<EdgeSrcWriterStats>>>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl EdgeSrcWritePool {
    fn new(target: std::sync::Arc<OutputTarget>, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        let error = std::sync::Arc::new(std::sync::Mutex::new(None));
        for _ in 0..workers {
            let (tx, rx) =
                std::sync::mpsc::sync_channel::<EdgeSrcWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            let error = std::sync::Arc::clone(&error);
            handles.push(std::thread::spawn(move || {
                match edge_src_writer_worker(target, rx) {
                    Ok(stats) => Ok(stats),
                    Err(err) => {
                        let mut guard = error.lock().unwrap_or_else(|e| e.into_inner());
                        if guard.is_none() {
                            *guard = Some(format!("{err:#}"));
                        }
                        Err(err)
                    }
                }
            }));
            senders.push(tx);
        }
        Self {
            senders: std::sync::Arc::new(senders),
            handles,
            error,
        }
    }

    fn senders(&self) -> std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>> {
        std::sync::Arc::clone(&self.senders)
    }

    fn error(&self) -> std::sync::Arc<std::sync::Mutex<Option<String>>> {
        std::sync::Arc::clone(&self.error)
    }

    fn finish(self) -> Result<EdgeSrcWriterStats> {
        drop(self.senders);
        let mut stats = EdgeSrcWriterStats::default();
        for handle in self.handles {
            match handle.join() {
                Ok(result) => {
                    stats.merge(result?);
                }
                Err(_) => return Err(anyhow!("edge src writer thread panicked")),
            }
        }
        if let Ok(guard) = self.error.lock() {
            if let Some(message) = guard.as_ref() {
                return Err(anyhow!("edge src writer failed: {message}"));
            }
        }
        Ok(stats)
    }
}

fn edge_src_writer_worker(
    target: std::sync::Arc<OutputTarget>,
    rx: std::sync::mpsc::Receiver<EdgeSrcWriteMessage>,
) -> Result<EdgeSrcWriterStats> {
    let mut writers: HashMap<(std::sync::Arc<str>, usize), TempRowWriter> = HashMap::new();
    let mut stats = EdgeSrcWriterStats::default();
    while let Ok(msg) = rx.recv() {
        match msg {
            EdgeSrcWriteMessage::Batch(batch) => {
                let key = (std::sync::Arc::clone(&batch.rel), batch.shard);
                let writer = match writers.get_mut(&key) {
                    Some(writer) => writer,
                    None => {
                        writers.insert(
                            key.clone(),
                            open_temp_row_writer(&target, &temp_edge_path(&batch.rel, batch.shard))?,
                        );
                        writers.get_mut(&key).unwrap()
                    }
                };
                write_temp_rows_batch(writer, &batch.data, batch.row_count)?;
                let entry = stats
                    .rows
                    .entry((batch.rel.to_string(), batch.shard))
                    .or_insert(0);
                *entry = entry.saturating_add(batch.row_count);
            }
        }
    }
    for (_, writer) in writers {
        finish_temp_writer(writer)?;
    }
    Ok(stats)
}

enum DestKeyWriteMessage {
    NodeFkey {
        table: std::sync::Arc<str>,
        rel: std::sync::Arc<str>,
        shard: usize,
        data: Vec<u8>,
        row_count: u64,
    },
    EdgeKey {
        edge_table: std::sync::Arc<str>,
        kind: EdgeKeyKind,
        shard: usize,
        data: Vec<u8>,
        row_count: u64,
    },
}

#[derive(Default)]
struct DestKeyWriterStats {
    node_fkey_rows: HashMap<(String, String, usize), u64>,
    edge_key_rows: HashMap<(String, EdgeKeyKind, usize), u64>,
}

impl DestKeyWriterStats {
    fn merge(&mut self, other: DestKeyWriterStats) {
        for (key, value) in other.node_fkey_rows {
            let entry = self.node_fkey_rows.entry(key).or_insert(0);
            *entry = entry.saturating_add(value);
        }
        for (key, value) in other.edge_key_rows {
            let entry = self.edge_key_rows.entry(key).or_insert(0);
            *entry = entry.saturating_add(value);
        }
    }
}

struct DestKeyWritePool {
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
    handles: Vec<std::thread::JoinHandle<Result<DestKeyWriterStats>>>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl DestKeyWritePool {
    fn new(target: std::sync::Arc<OutputTarget>, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        let error = std::sync::Arc::new(std::sync::Mutex::new(None));
        for _ in 0..workers {
            let (tx, rx) =
                std::sync::mpsc::sync_channel::<DestKeyWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            let error = std::sync::Arc::clone(&error);
            handles.push(std::thread::spawn(move || {
                match dest_key_writer_worker(target, rx) {
                    Ok(stats) => Ok(stats),
                    Err(err) => {
                        let mut guard = error.lock().unwrap_or_else(|e| e.into_inner());
                        if guard.is_none() {
                            *guard = Some(format!("{err:#}"));
                        }
                        Err(err)
                    }
                }
            }));
            senders.push(tx);
        }
        Self {
            senders: std::sync::Arc::new(senders),
            handles,
            error,
        }
    }

    fn senders(&self) -> std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>> {
        std::sync::Arc::clone(&self.senders)
    }

    fn error(&self) -> std::sync::Arc<std::sync::Mutex<Option<String>>> {
        std::sync::Arc::clone(&self.error)
    }

    fn finish(self) -> Result<DestKeyWriterStats> {
        drop(self.senders);
        let mut stats = DestKeyWriterStats::default();
        for handle in self.handles {
            match handle.join() {
                Ok(result) => {
                    stats.merge(result?);
                }
                Err(_) => return Err(anyhow!("dest key writer thread panicked")),
            }
        }
        if let Ok(guard) = self.error.lock() {
            if let Some(message) = guard.as_ref() {
                return Err(anyhow!("dest key writer failed: {message}"));
            }
        }
        Ok(stats)
    }
}

fn dest_key_writer_worker(
    target: std::sync::Arc<OutputTarget>,
    rx: std::sync::mpsc::Receiver<DestKeyWriteMessage>,
) -> Result<DestKeyWriterStats> {
    let mut node_writers: HashMap<(std::sync::Arc<str>, std::sync::Arc<str>, usize), TempRowWriter> =
        HashMap::new();
    let mut edge_writers: HashMap<(std::sync::Arc<str>, EdgeKeyKind, usize), TempRowWriter> =
        HashMap::new();
    let mut stats = DestKeyWriterStats::default();
    while let Ok(msg) = rx.recv() {
        match msg {
            DestKeyWriteMessage::NodeFkey {
                table,
                rel,
                shard,
                data,
                row_count,
            } => {
                let key = (std::sync::Arc::clone(&table), std::sync::Arc::clone(&rel), shard);
                let writer = match node_writers.get_mut(&key) {
                    Some(writer) => writer,
                    None => {
                        node_writers.insert(
                            key.clone(),
                            open_temp_row_writer(&target, &temp_fkey_dst_path(&table, &rel, shard))?,
                        );
                        node_writers.get_mut(&key).unwrap()
                    }
                };
                write_temp_rows_batch(writer, &data, row_count)?;
                let entry = stats
                    .node_fkey_rows
                    .entry((table.to_string(), rel.to_string(), shard))
                    .or_insert(0);
                *entry = entry.saturating_add(row_count);
            }
            DestKeyWriteMessage::EdgeKey {
                edge_table,
                kind,
                shard,
                data,
                row_count,
            } => {
                let key = (std::sync::Arc::clone(&edge_table), kind, shard);
                let writer = match edge_writers.get_mut(&key) {
                    Some(writer) => writer,
                    None => {
                        edge_writers.insert(
                            key.clone(),
                            open_temp_row_writer(
                                &target,
                                &temp_edge_key_path(&edge_table, kind, shard),
                            )?,
                        );
                        edge_writers.get_mut(&key).unwrap()
                    }
                };
                write_temp_rows_batch(writer, &data, row_count)?;
                let entry = stats
                    .edge_key_rows
                    .entry((edge_table.to_string(), kind, shard))
                    .or_insert(0);
                *entry = entry.saturating_add(row_count);
            }
        }
    }
    for (_, writer) in node_writers {
        finish_temp_writer(writer)?;
    }
    for (_, writer) in edge_writers {
        finish_temp_writer(writer)?;
    }
    Ok(stats)
}

struct EdgeSrcBuffer {
    data: Vec<u8>,
    row_count: u64,
}

struct EdgeSrcBatcher {
    rel: std::sync::Arc<str>,
    rel_hash: u64,
    buffers: Vec<EdgeSrcBuffer>,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl EdgeSrcBatcher {
    fn new(
        rel: std::sync::Arc<str>,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>>,
        error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    ) -> Self {
        let rel_hash = hash_name(&rel);
        let buffers = (0..num_shards)
            .map(|_| EdgeSrcBuffer {
                data: Vec::new(),
                row_count: 0,
            })
            .collect();
        Self {
            rel,
            rel_hash,
            buffers,
            buffer_limit,
            senders,
            error,
        }
    }

    fn push(&mut self, shard: usize, row: &[u8], timestamp_valid: bool) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("edge src buffer shard {} out of bounds", shard))?;
        let total_len = (row.len() + 1) as u32;
        buffer.data.extend_from_slice(&total_len.to_le_bytes());
        buffer.data.extend_from_slice(row);
        buffer.data.push(if timestamp_valid { 1 } else { 0 });
        buffer.row_count = buffer.row_count.saturating_add(1);
        if buffer.data.len() >= self.buffer_limit {
            self.flush_shard(shard)?;
        }
        Ok(())
    }

    fn flush_shard(&mut self, shard: usize) -> Result<()> {
        if let Ok(guard) = self.error.lock() {
            if let Some(message) = guard.as_ref() {
                return Err(anyhow!("edge src writer failed: {message}"));
            }
        }
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("edge src buffer shard {} out of bounds", shard))?;
        if buffer.row_count == 0 {
            return Ok(());
        }
        let mut data = Vec::with_capacity(buffer.data.capacity());
        std::mem::swap(&mut data, &mut buffer.data);
        let batch = EdgeSrcWriteBatch {
            rel: std::sync::Arc::clone(&self.rel),
            shard,
            data,
            row_count: buffer.row_count,
        };
        buffer.row_count = 0;
        let idx = edge_writer_index(self.rel_hash, shard, self.senders.len());
        self.senders[idx]
            .send(EdgeSrcWriteMessage::Batch(batch))
            .map_err(|e| {
                if let Ok(guard) = self.error.lock() {
                    if let Some(message) = guard.as_ref() {
                        return anyhow!("edge src writer failed: {message}");
                    }
                }
                anyhow!("edge src writer channel error: {e}")
            })?;
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for shard in 0..self.buffers.len() {
            self.flush_shard(shard)?;
        }
        Ok(())
    }
}

struct DestFkeyBuffer {
    data: Vec<u8>,
    row_count: u64,
}

struct DestFkeyShardBuffer {
    fkeys: Vec<DestFkeyBuffer>,
    bytes: usize,
}

struct DestFkeyBatcher {
    table: std::sync::Arc<str>,
    table_hash: u64,
    rel_names: std::sync::Arc<Vec<std::sync::Arc<str>>>,
    buffers: Vec<DestFkeyShardBuffer>,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl DestFkeyBatcher {
    fn new(
        table: std::sync::Arc<str>,
        rel_names: std::sync::Arc<Vec<std::sync::Arc<str>>>,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
        error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    ) -> Self {
        let table_hash = hash_name(&table);
        let buffers = (0..num_shards)
            .map(|_| DestFkeyShardBuffer {
                fkeys: rel_names
                    .iter()
                    .map(|_| DestFkeyBuffer {
                        data: Vec::new(),
                        row_count: 0,
                    })
                    .collect(),
                bytes: 0,
            })
            .collect();
        Self {
            table,
            table_hash,
            rel_names,
            buffers,
            buffer_limit,
            senders,
            error,
        }
    }

    fn push(&mut self, shard: usize, rel_idx: usize, row: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("dest fkey buffer shard {} out of bounds", shard))?;
        let fkey = buffer
            .fkeys
            .get_mut(rel_idx)
            .ok_or_else(|| anyhow!("dest fkey rel {} out of bounds", rel_idx))?;
        append_temp_row(&mut fkey.data, row)?;
        fkey.row_count = fkey.row_count.saturating_add(1);
        buffer.bytes = buffer.bytes.saturating_add(4 + row.len());
        if buffer.bytes >= self.buffer_limit {
            self.flush_shard(shard)?;
        }
        Ok(())
    }

    fn flush_shard(&mut self, shard: usize) -> Result<()> {
        if let Ok(guard) = self.error.lock() {
            if let Some(message) = guard.as_ref() {
                return Err(anyhow!("dest key writer failed: {message}"));
            }
        }
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("dest fkey buffer shard {} out of bounds", shard))?;
        if buffer.bytes == 0 {
            return Ok(());
        }
        for (rel_idx, entry) in buffer.fkeys.iter_mut().enumerate() {
            if entry.row_count == 0 {
                continue;
            }
            let mut data = Vec::with_capacity(entry.data.capacity());
            std::mem::swap(&mut data, &mut entry.data);
            let rel = self
                .rel_names
                .get(rel_idx)
                .cloned()
                .ok_or_else(|| anyhow!("dest fkey rel {} missing", rel_idx))?;
            let batch = DestKeyWriteMessage::NodeFkey {
                table: std::sync::Arc::clone(&self.table),
                rel,
                shard,
                data,
                row_count: entry.row_count,
            };
            entry.row_count = 0;
            let idx = edge_writer_index(self.table_hash, shard, self.senders.len());
            self.senders[idx]
                .send(batch)
                .map_err(|e| anyhow!("dest key writer channel error: {e}"))?;
        }
        buffer.bytes = 0;
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for shard in 0..self.buffers.len() {
            self.flush_shard(shard)?;
        }
        Ok(())
    }
}

struct EdgeKeyBuffer {
    data: Vec<u8>,
    row_count: u64,
}

struct EdgeKeyBatcher {
    edge_table: std::sync::Arc<str>,
    edge_hash: u64,
    buffers_src: Vec<EdgeKeyBuffer>,
    buffers_dst: Vec<EdgeKeyBuffer>,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl EdgeKeyBatcher {
    fn new(
        edge_table: std::sync::Arc<str>,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
        error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    ) -> Self {
        let edge_hash = hash_name(&edge_table);
        let buffers_src = (0..num_shards)
            .map(|_| EdgeKeyBuffer {
                data: Vec::new(),
                row_count: 0,
            })
            .collect();
        let buffers_dst = (0..num_shards)
            .map(|_| EdgeKeyBuffer {
                data: Vec::new(),
                row_count: 0,
            })
            .collect();
        Self {
            edge_table,
            edge_hash,
            buffers_src,
            buffers_dst,
            buffer_limit,
            senders,
            error,
        }
    }

    fn push_src(&mut self, shard: usize, row: &[u8]) -> Result<()> {
        self.push_kind(EdgeKeyKind::Src, shard, row)
    }

    fn push_dst(&mut self, shard: usize, row: &[u8]) -> Result<()> {
        self.push_kind(EdgeKeyKind::Dst, shard, row)
    }

    fn push_kind(&mut self, kind: EdgeKeyKind, shard: usize, row: &[u8]) -> Result<()> {
        let buffers = match kind {
            EdgeKeyKind::Src => &mut self.buffers_src,
            EdgeKeyKind::Dst => &mut self.buffers_dst,
        };
        let buffer = buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("edge key buffer shard {} out of bounds", shard))?;
        append_temp_row(&mut buffer.data, row)?;
        buffer.row_count = buffer.row_count.saturating_add(1);
        if buffer.data.len() >= self.buffer_limit {
            self.flush_shard(kind, shard)?;
        }
        Ok(())
    }

    fn flush_shard(&mut self, kind: EdgeKeyKind, shard: usize) -> Result<()> {
        if let Ok(guard) = self.error.lock() {
            if let Some(message) = guard.as_ref() {
                return Err(anyhow!("dest key writer failed: {message}"));
            }
        }
        let buffers = match kind {
            EdgeKeyKind::Src => &mut self.buffers_src,
            EdgeKeyKind::Dst => &mut self.buffers_dst,
        };
        let buffer = buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("edge key buffer shard {} out of bounds", shard))?;
        if buffer.row_count == 0 {
            return Ok(());
        }
        let mut data = Vec::with_capacity(buffer.data.capacity());
        std::mem::swap(&mut data, &mut buffer.data);
        let batch = DestKeyWriteMessage::EdgeKey {
            edge_table: std::sync::Arc::clone(&self.edge_table),
            kind,
            shard,
            data,
            row_count: buffer.row_count,
        };
        buffer.row_count = 0;
        let idx = edge_writer_index(self.edge_hash, shard, self.senders.len());
        self.senders[idx]
            .send(batch)
            .map_err(|e| anyhow!("dest key writer channel error: {e}"))?;
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for shard in 0..self.buffers_src.len() {
            self.flush_shard(EdgeKeyKind::Src, shard)?;
            self.flush_shard(EdgeKeyKind::Dst, shard)?;
        }
        Ok(())
    }
}

fn column_spec<'a>(table: &'a TableSpec, name: &str) -> Result<&'a ColumnSpec> {
    table
        .columns
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| anyhow!("column {} not found in {}", name, table.name))
}

fn table_by_name<'a>(source: &'a GraphConfig, name: &str) -> Result<&'a TableSpec> {
    source
        .tables
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("table {} not found", name))
}

fn sum_input_bytes(source: &GraphConfig) -> u64 {
    let mut total = 0u64;
    for table in &source.tables {
        for path in table.get_paths() {
            if is_cloud_path(path) {
                continue;
            }
            if let Ok(meta) = fs::metadata(path) {
                total = total.saturating_add(meta.len());
            }
        }
    }
    total
}

fn sum_dir_bytes(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    if path.is_file() {
        return Ok(fs::metadata(path)?.len());
    }
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                total = total.saturating_add(sum_dir_bytes(&entry.path())?);
            } else {
                total = total.saturating_add(meta.len());
            }
        }
    }
    Ok(total)
}

fn default_index_mem_budget_bytes() -> Option<u64> {
    let total = total_memory_bytes()?;
    Some((total as f64 * DEFAULT_INDEX_MEM_BUDGET_FRAC) as u64)
}

#[cfg(target_os = "linux")]
fn total_memory_bytes() -> Option<u64> {
    let mem_total = read_meminfo_total().unwrap_or(0);
    let cgroup_limit = read_cgroup_limit().unwrap_or(u64::MAX);
    let total = mem_total.min(cgroup_limit);
    if total == 0 || total == u64::MAX {
        None
    } else {
        Some(total)
    }
}

#[cfg(not(target_os = "linux"))]
fn total_memory_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn read_meminfo_total() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(value) = line.strip_prefix("MemTotal:") {
            return parse_kb_line(value);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn read_cgroup_limit() -> Option<u64> {
    if let Ok(limit) = fs::read_to_string("/sys/fs/cgroup/memory.max") {
        let trimmed = limit.trim();
        if trimmed != "max" {
            if let Ok(bytes) = trimmed.parse::<u64>() {
                return Some(bytes);
            }
        }
    }
    if let Ok(limit) = fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        let trimmed = limit.trim();
        if let Ok(bytes) = trimmed.parse::<u64>() {
            return Some(bytes);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn parse_kb_line(value: &str) -> Option<u64> {
    let mut iter = value.split_whitespace();
    let num = iter.next()?.parse::<u64>().ok()?;
    Some(num.saturating_mul(1024))
}

struct WorkItem<S> {
    run: Box<dyn FnOnce(&mut S) -> Result<()> + Send>,
    bucket: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
}

struct PoolTask<S> {
    item: WorkItem<S>,
    done: std::sync::mpsc::Sender<Result<()>>,
}

enum PoolMessage<S> {
    Task(PoolTask<S>),
    Stop,
}

struct TaskPool<S> {
    sender: std::sync::mpsc::Sender<PoolMessage<S>>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl<S: Default + Send + 'static> TaskPool<S> {
    fn new(threads: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<PoolMessage<S>>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let rx = std::sync::Arc::clone(&rx);
            handles.push(std::thread::spawn(move || {
                let mut scratch = S::default();
                loop {
                    let msg = {
                        let guard = rx.lock().unwrap();
                        guard.recv()
                    };
                    let msg = match msg {
                        Ok(msg) => msg,
                        Err(_) => break,
                    };
                    match msg {
                        PoolMessage::Task(task) => {
                            let start = std::time::Instant::now();
                            let result = (task.item.run)(&mut scratch);
                            let elapsed = start.elapsed().as_nanos() as u64;
                            if let Some(bucket) = &task.item.bucket {
                                bucket.fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);
                            }
                            let _ = task.done.send(result);
                        }
                        PoolMessage::Stop => break,
                    }
                }
            }));
        }
        Self {
            sender: tx,
            handles,
        }
    }

    fn run_tasks(&self, tasks: Vec<WorkItem<S>>) -> Result<()> {
        if tasks.is_empty() {
            return Ok(());
        }
        let mut receivers = Vec::with_capacity(tasks.len());
        for task in tasks {
            let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
            self.sender
                .send(PoolMessage::Task(PoolTask { item: task, done: tx }))
                .map_err(|e| anyhow!("task queue error: {e}"))?;
            receivers.push(rx);
        }
        for rx in receivers {
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(err),
                Err(err) => return Err(anyhow!("task result error: {err}")),
            }
        }
        Ok(())
    }
}

impl<S> Drop for TaskPool<S> {
    fn drop(&mut self) {
        for _ in 0..self.handles.len() {
            let _ = self.sender.send(PoolMessage::Stop);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .saturating_mul(105)
        .saturating_add(99)
        / 100
}

fn run_tasks<S: Default + Send + 'static>(
    tasks: Vec<WorkItem<S>>,
    pool: &TaskPool<S>,
) -> Result<()> {
    pool.run_tasks(tasks)
}


fn key_type_for_column(col: &ColumnSpec) -> Result<KeyType> {
    match col.ingested_dtype {
        IngestedType::Int64
        | IngestedType::Int32
        | IngestedType::Int16
        | IngestedType::Int8
        | IngestedType::UInt64
        | IngestedType::UInt32
        | IngestedType::UInt16
        | IngestedType::UInt8 => Ok(KeyType::Int64),
        IngestedType::Float32
        | IngestedType::Float64
        | IngestedType::Decimal128 => Ok(KeyType::Int64),
        IngestedType::Decimal256 => Ok(KeyType::VarBytes),
        IngestedType::String | IngestedType::Bytes => Ok(KeyType::VarBytes),
        IngestedType::Uuid => Ok(KeyType::Uuid),
        IngestedType::Uuid256 => Ok(KeyType::Uuid256),
        IngestedType::Ip => Ok(KeyType::Ip),
        _ => Err(anyhow!("unsupported primary key type")),
    }
}

fn column_index(table: &TableSpec, name: &str) -> Result<usize> {
    table
        .columns
        .iter()
        .position(|col| col.name == name)
        .ok_or_else(|| anyhow!("missing column {}", name))
}

fn key_from_access_into_buf(
    access: &mut RowAccess<'_>,
    idx: usize,
    col: &ColumnSpec,
    out: &mut Vec<u8>,
) -> KeyParseResult<()> {
    let key_type = key_type_for_column(col).map_err(|_| KeyParseError::UnsupportedType)?;
    out.clear();
    match key_type {
        KeyType::Int64 => {
            let value = match col.ingested_dtype {
                IngestedType::Int64 => access
                    .get_int64(idx)
                    .map_err(|_| KeyParseError::TypeMismatch)?
                    .ok_or(KeyParseError::TypeMismatch)?,
                IngestedType::Int32 | IngestedType::Int16 | IngestedType::Int8 => {
                    let value = access
                        .get_int32(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    value as i64
                }
                IngestedType::UInt64 => {
                    let value = access
                        .get_uint64(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    out.extend_from_slice(&value.to_le_bytes());
                    return Ok(());
                }
                IngestedType::UInt32 | IngestedType::UInt16 | IngestedType::UInt8 => {
                    let value = access
                        .get_uint32(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    out.extend_from_slice(&(value as u64).to_le_bytes());
                    return Ok(());
                }
                IngestedType::Decimal128 => {
                    let mut value = access
                        .get_decimal128(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    if let Some(scale) = col.scale {
                        if scale > 0 {
                            if let Some(divisor) = 10i128.checked_pow(scale as u32) {
                                value /= divisor;
                            }
                        }
                    }
                    let value = i64::try_from(value).map_err(|_| KeyParseError::TooLarge)?;
                    // Negative values are cast to u64 (two's complement)
                    out.extend_from_slice(&(value as u64).to_le_bytes());
                    return Ok(());
                }
                IngestedType::String | IngestedType::Bytes => {
                    let array = access
                        .array(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    let bytes = read_bytes(array, access.row())
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    let value = parse_base10_bytes(bytes)?;
                    out.extend_from_slice(&value.to_le_bytes());
                    return Ok(());
                }
                _ => return Err(KeyParseError::TypeMismatch),
            };
            // Negative values are cast to u64 (two's complement)
            out.extend_from_slice(&(value as u64).to_le_bytes());
            Ok(())
        }
        KeyType::VarBytes => {
            let mut write_bytes = |bytes: &[u8]| -> KeyParseResult<()> {
                let start = out.len();
                let len = write_normalized_numeric_bytes(out, bytes, col)?;
                if len > MAX_VAR_KEY_LEN {
                    out.truncate(start);
                    return Err(KeyParseError::TooLarge);
                }
                Ok(())
            };
            match col.ingested_dtype {
                IngestedType::Bytes => {
                    let array = access
                        .array(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    let bytes = read_bytes(array, access.row())
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    write_bytes(bytes)?;
                }
                IngestedType::String | IngestedType::Decimal256 => {
                    access
                        .with_string(idx, |text| {
                            write_bytes(text.as_bytes()).map_err(|_| anyhow!("key parse failed"))?;
                            Ok(())
                        })
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                }
                IngestedType::Int64 => {
                    let value = access
                        .get_int64(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    let start = out.len();
                    let len = write_display_to_vec(out, value);
                    if len > MAX_VAR_KEY_LEN {
                        out.truncate(start);
                        return Err(KeyParseError::TooLarge);
                    }
                }
                IngestedType::Int32 | IngestedType::Int16 | IngestedType::Int8 => {
                    let value = access
                        .get_int32(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    let start = out.len();
                    let len = write_display_to_vec(out, value);
                    if len > MAX_VAR_KEY_LEN {
                        out.truncate(start);
                        return Err(KeyParseError::TooLarge);
                    }
                }
                IngestedType::UInt64 => {
                    let value = access
                        .get_uint64(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    let start = out.len();
                    let len = write_display_to_vec(out, value);
                    if len > MAX_VAR_KEY_LEN {
                        out.truncate(start);
                        return Err(KeyParseError::TooLarge);
                    }
                }
                IngestedType::UInt32 | IngestedType::UInt16 | IngestedType::UInt8 => {
                    let value = access
                        .get_uint32(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    let start = out.len();
                    let len = write_display_to_vec(out, value);
                    if len > MAX_VAR_KEY_LEN {
                        out.truncate(start);
                        return Err(KeyParseError::TooLarge);
                    }
                }
                _ => return Err(KeyParseError::TypeMismatch),
            };
            Ok(())
        }
        KeyType::Uuid => {
            let mut buf = [0u8; 16];
            let uuid = match col.ingested_dtype {
                IngestedType::Uuid => {
                    let value = access
                        .get_uuid(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    buf.copy_from_slice(&value);
                    buf
                }
                IngestedType::String | IngestedType::Bytes => {
                    let array = access
                        .array(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    if let Ok(text) = read_utf8(array, access.row()) {
                        parse_uuid_string(text.as_bytes())?
                    } else {
                        let data = read_bytes(array, access.row())
                            .map_err(|_| KeyParseError::TypeMismatch)?;
                        if data.len() != 16 {
                            return Err(KeyParseError::LengthMismatch);
                        }
                        buf.copy_from_slice(data);
                        buf
                    }
                }
                _ => return Err(KeyParseError::TypeMismatch),
            };
            out.extend_from_slice(&uuid);
            Ok(())
        }
        KeyType::Uuid256 => {
            let mut buf = [0u8; 32];
            let uuid = match col.ingested_dtype {
                IngestedType::Uuid256 => {
                    let value = access
                        .get_uuid256(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    buf.copy_from_slice(&value);
                    buf
                }
                IngestedType::String | IngestedType::Bytes => {
                    let array = access
                        .array(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    if let Ok(text) = read_utf8(array, access.row()) {
                        parse_uuid256_string(text.as_bytes())?
                    } else {
                        let data = read_bytes(array, access.row())
                            .map_err(|_| KeyParseError::TypeMismatch)?;
                        if data.len() != 32 {
                            return Err(KeyParseError::LengthMismatch);
                        }
                        buf.copy_from_slice(data);
                        buf
                    }
                }
                _ => return Err(KeyParseError::TypeMismatch),
            };
            out.extend_from_slice(&uuid);
            Ok(())
        }
        KeyType::Ip => {
            let ip = match col.ingested_dtype {
                IngestedType::Ip => {
                    let value = access
                        .get_ip(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    value
                }
                IngestedType::String | IngestedType::Bytes => {
                    let array = access
                        .array(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    if let Ok(text) = read_utf8(array, access.row()) {
                        parse_ip_string(text.as_bytes())?
                    } else {
                        let data = read_bytes(array, access.row())
                            .map_err(|_| KeyParseError::TypeMismatch)?;
                        parse_ip_bytes(data)?
                    }
                }
                _ => return Err(KeyParseError::TypeMismatch),
            };
            out.extend_from_slice(&ip);
            Ok(())
        }
    }
}

fn shard_for_key_from_key(key: &Key, buf: &mut Vec<u8>, num_shards: usize) -> Result<usize> {
    buf.clear();
    write_key_to_buf(buf, key)?;
    Ok(shard_for_key(buf, num_shards))
}

fn write_key_to_buf(buf: &mut Vec<u8>, key: &Key) -> Result<()> {
    match key {
        Key::Int(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Key::Uuid(v) => buf.extend_from_slice(v),
        Key::Uuid256(v) => buf.extend_from_slice(v),
        Key::Ip(v) => buf.extend_from_slice(v),
        Key::Bytes(v) => {
            buf.extend_from_slice(v);
        }
    }
    Ok(())
}

fn append_key_bytes_with_len_prefix(
    row: &mut Vec<u8>,
    key_bytes: &[u8],
    col: &ColumnSpec,
) -> Result<()> {
    match key_type_for_column(col)? {
        KeyType::VarBytes => {
            let len = u16::try_from(key_bytes.len())
                .map_err(|_| anyhow!("temp key too large"))?;
            row.extend_from_slice(&len.to_le_bytes());
            row.extend_from_slice(key_bytes);
        }
        _ => {
            row.extend_from_slice(key_bytes);
        }
    }
    Ok(())
}

fn read_key_from_buf(data: &mut &[u8], col: &ColumnSpec) -> Result<Key> {
    let key_type = key_type_for_column(col)?;
    match key_type {
        KeyType::Int64 => {
            if data.len() < 8 {
                return Err(anyhow!("temp key truncated"));
            }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&data[..8]);
            *data = &data[8..];
            Ok(Key::Int(u64::from_le_bytes(buf)))
        }
        KeyType::Uuid => {
            if data.len() < 16 {
                return Err(anyhow!("temp key truncated"));
            }
            let mut buf = [0u8; 16];
            buf.copy_from_slice(&data[..16]);
            *data = &data[16..];
            Ok(Key::Uuid(buf))
        }
        KeyType::Uuid256 => {
            if data.len() < 32 {
                return Err(anyhow!("temp key truncated"));
            }
            let mut buf = [0u8; 32];
            buf.copy_from_slice(&data[..32]);
            *data = &data[32..];
            Ok(Key::Uuid256(buf))
        }
        KeyType::Ip => {
            if data.len() < 16 {
                return Err(anyhow!("temp key truncated"));
            }
            let mut buf = [0u8; 16];
            buf.copy_from_slice(&data[..16]);
            *data = &data[16..];
            Ok(Key::Ip(buf))
        }
        KeyType::VarBytes => {
            if data.len() > MAX_VAR_KEY_LEN {
                return Err(anyhow!("temp key too large"));
            }
            let bytes = data.to_vec();
            *data = &[];
            Ok(Key::Bytes(bytes))
        }
    }
}

fn read_key_from_buf_len_prefixed(data: &mut &[u8], col: &ColumnSpec) -> Result<Key> {
    let key_type = key_type_for_column(col)?;
    match key_type {
        KeyType::Int64 => {
            if data.len() < 8 {
                return Err(anyhow!("temp key truncated"));
            }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&data[..8]);
            *data = &data[8..];
            Ok(Key::Int(u64::from_le_bytes(buf)))
        }
        KeyType::Uuid => {
            if data.len() < 16 {
                return Err(anyhow!("temp key truncated"));
            }
            let mut buf = [0u8; 16];
            buf.copy_from_slice(&data[..16]);
            *data = &data[16..];
            Ok(Key::Uuid(buf))
        }
        KeyType::Uuid256 => {
            if data.len() < 32 {
                return Err(anyhow!("temp key truncated"));
            }
            let mut buf = [0u8; 32];
            buf.copy_from_slice(&data[..32]);
            *data = &data[32..];
            Ok(Key::Uuid256(buf))
        }
        KeyType::Ip => {
            if data.len() < 16 {
                return Err(anyhow!("temp key truncated"));
            }
            let mut buf = [0u8; 16];
            buf.copy_from_slice(&data[..16]);
            *data = &data[16..];
            Ok(Key::Ip(buf))
        }
        KeyType::VarBytes => {
            if data.len() < 2 {
                return Err(anyhow!("temp key truncated"));
            }
            let len = u16::from_le_bytes(data[..2].try_into().unwrap()) as usize;
            *data = &data[2..];
            if data.len() < len {
                return Err(anyhow!("temp key truncated"));
            }
            if len > MAX_VAR_KEY_LEN {
                return Err(anyhow!("temp key too large"));
            }
            let bytes = data[..len].to_vec();
            *data = &data[len..];
            Ok(Key::Bytes(bytes))
        }
    }
}
fn parse_base10_bytes(bytes: &[u8]) -> KeyParseResult<u64> {
    if bytes.is_empty() {
        return Err(KeyParseError::Empty);
    }
    if bytes.len() > 20 {
        return Err(KeyParseError::TooLarge);
    }
    if bytes.contains(&b'.') {
        return Err(KeyParseError::NotInteger);
    }
    let mut value: u64 = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return Err(KeyParseError::NotBase10);
        }
        let digit = (b - b'0') as u64;
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(digit))
            .ok_or(KeyParseError::TooLarge)?;
    }
    Ok(value)
}



pub(crate) fn parse_uuid_string(bytes: &[u8]) -> KeyParseResult<[u8; 16]> {
    let mut out = [0u8; 16];
    if bytes.len() == 32 {
        parse_simple(bytes, out.as_out()).map_err(|_| KeyParseError::InvalidUuid)?;
        return Ok(out);
    }
    if bytes.len() == 36 {
        parse_hyphenated(bytes, out.as_out()).map_err(|_| KeyParseError::InvalidUuid)?;
        return Ok(out);
    }
    Err(KeyParseError::InvalidUuid)
}

pub(crate) fn parse_uuid256_string(bytes: &[u8]) -> KeyParseResult<[u8; 32]> {
    if bytes.len() != 64 {
        return Err(KeyParseError::InvalidUuid256);
    }
    let mut out = [0u8; 32];
    // Safety: `out` is 32 bytes, so the two 16-byte regions are in-bounds and non-overlapping.
    let out_ptr = out.as_mut_ptr();
    let hi = unsafe { &mut *(out_ptr as *mut [u8; 16]) };
    let lo = unsafe { &mut *(out_ptr.add(16) as *mut [u8; 16]) };
    parse_simple(&bytes[..32], hi.as_out()).map_err(|_| KeyParseError::InvalidUuid256)?;
    parse_simple(&bytes[32..], lo.as_out()).map_err(|_| KeyParseError::InvalidUuid256)?;
    Ok(out)
}

pub(crate) fn parse_ip_string(bytes: &[u8]) -> KeyParseResult<[u8; 16]> {
    let s = std::str::from_utf8(bytes).map_err(|_| KeyParseError::NotUtf8)?;
    let ip: std::net::IpAddr = s.parse().map_err(|_| KeyParseError::InvalidIp)?;
    Ok(ip_to_ipv6_bytes(ip))
}

pub(crate) fn parse_ip_bytes(bytes: &[u8]) -> KeyParseResult<[u8; 16]> {
    match bytes.len() {
        4 => {
            let ip = std::net::IpAddr::V4(std::net::Ipv4Addr::new(
                bytes[0], bytes[1], bytes[2], bytes[3],
            ));
            Ok(ip_to_ipv6_bytes(ip))
        }
        16 => {
            let mut out = [0u8; 16];
            out.copy_from_slice(bytes);
            Ok(out)
        }
        _ => Err(KeyParseError::LengthMismatch),
    }
}

fn ip_to_ipv6_bytes(ip: std::net::IpAddr) -> [u8; 16] {
    match ip {
        std::net::IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
        std::net::IpAddr::V6(v6) => v6.octets(),
    }
}

fn write_normalized_numeric_bytes(
    out: &mut Vec<u8>,
    bytes: &[u8],
    target_col: &ColumnSpec,
) -> KeyParseResult<usize> {
    if matches!(
        target_col.ingested_dtype,
        IngestedType::Float32
            | IngestedType::Float64
            | IngestedType::Decimal128
            | IngestedType::Decimal256
    ) {
        return write_normalized_base10_bytes(out, bytes);
    }
    out.extend_from_slice(bytes);
    Ok(bytes.len())
}

fn write_normalized_base10_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> KeyParseResult<usize> {
    let s = std::str::from_utf8(bytes).map_err(|_| KeyParseError::NotUtf8)?;
    if s.is_empty() {
        return Err(KeyParseError::Empty);
    }
    let mut parts = s.split('.');
    let int_part = parts.next().unwrap();
    if int_part.is_empty() {
        return Err(KeyParseError::NotBase10);
    }
    let frac_part = parts.next();
    if parts.next().is_some() {
        return Err(KeyParseError::NotBase10);
    }
    if !int_part.bytes().all(|b| b.is_ascii_digit()) {
        return Err(KeyParseError::NotBase10);
    }
    if let Some(frac) = frac_part {
        if !frac.bytes().all(|b| b.is_ascii_digit()) {
            return Err(KeyParseError::NotBase10);
        }
    }
    let int_trimmed = int_part.trim_start_matches('0');
    let int_out = if int_trimmed.is_empty() { "0" } else { int_trimmed };
    let frac_out = frac_part.unwrap_or("").trim_end_matches('0');
    out.extend_from_slice(int_out.as_bytes());
    if frac_out.is_empty() {
        return Ok(int_out.len());
    }
    out.push(b'.');
    out.extend_from_slice(frac_out.as_bytes());
    Ok(int_out.len() + 1 + frac_out.len())
}

struct VecWrite<'a> {
    buf: &'a mut Vec<u8>,
}

impl<'a> std::fmt::Write for VecWrite<'a> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.buf.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

fn write_display_to_vec<T: std::fmt::Display>(out: &mut Vec<u8>, value: T) -> usize {
    let start = out.len();
    let mut writer = VecWrite { buf: out };
    let _ = std::fmt::write(&mut writer, format_args!("{}", value));
    out.len() - start
}

fn sample_key(key: &Key) -> Option<String> {
    match key {
        Key::Int(v) => Some(v.to_string()),
        Key::Bytes(bytes) => Some(format_bytes_sample(bytes)),
        Key::Uuid(bytes) => Some(hex_encode(bytes)),
        Key::Uuid256(bytes) => Some(hex_encode(bytes)),
        Key::Ip(bytes) => Some(hex_encode(bytes)),
    }
}

fn format_bytes_sample(bytes: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(bytes) {
        if text.chars().all(|ch| ch.is_ascii_graphic() || ch == ' ') {
            return truncate_sample(text);
        }
    }
    format!("0x{}", hex_encode(bytes))
}

fn format_bytes_sample_prefix(bytes: &[u8]) -> String {
    let limit = bytes.len().min(MAX_KEY_SAMPLE_LEN);
    let sample = format_bytes_sample(&bytes[..limit]);
    if bytes.len() > limit {
        format!("{sample}...")
    } else {
        sample
    }
}

fn truncate_sample(input: &str) -> String {
    if input.len() <= MAX_KEY_SAMPLE_LEN {
        return input.to_string();
    }
    let mut out: String = input.chars().take(MAX_KEY_SAMPLE_LEN).collect();
    out.push_str("...");
    out
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(hex_char(b >> 4));
        out.push(hex_char(b & 0x0f));
    }
    out
}

fn hex_char(v: u8) -> char {
    match v {
        0..=9 => (b'0' + v) as char,
        _ => (b'a' + (v - 10)) as char,
    }
}

fn record_key_parse_failure(
    total: &mut u64,
    samples: &mut Vec<String>,
    by_context: &mut HashMap<String, KeyFailureCounts>,
    context: &str,
    row: &[u8],
) {
    *total = total.saturating_add(1);
    let entry = by_context.entry(context.to_string()).or_default();
    entry.parse_failures = entry.parse_failures.saturating_add(1);
    if samples.len() < 10 {
        let sample = format_bytes_sample_prefix(row);
        samples.push(format!("{context}: len={} {sample}", row.len()));
    }
}

fn sample_from_access(access: &RowAccess<'_>, idx: usize) -> Option<String> {
    let array = access.array(idx).ok()?;
    let value = value_from_array(array, access.row()).ok()?;
    match value {
        JsonValue::Null => None,
        JsonValue::String(s) => Some(truncate_sample(&s)),
        JsonValue::Number(n) => Some(n.to_string()),
        JsonValue::Bool(v) => Some(v.to_string()),
        other => Some(other.to_string()),
    }
}

fn record_key_parse_failure_sample(
    total: &mut u64,
    samples: &mut Vec<String>,
    by_context: &mut HashMap<String, KeyFailureCounts>,
    context: &str,
    sample: Option<String>,
) {
    *total = total.saturating_add(1);
    let entry = by_context.entry(context.to_string()).or_default();
    entry.parse_failures = entry.parse_failures.saturating_add(1);
    if samples.len() < 10 {
        if let Some(sample) = sample {
            samples.push(format!("{context}: {sample}"));
        }
    }
}

fn record_key_lookup_failure(
    total: &mut u64,
    samples: &mut Vec<String>,
    by_context: &mut HashMap<String, KeyFailureCounts>,
    context: &str,
    sample: Option<String>,
) {
    *total = total.saturating_add(1);
    let entry = by_context.entry(context.to_string()).or_default();
    entry.lookup_failures = entry.lookup_failures.saturating_add(1);
    if samples.len() < 10 {
        if let Some(sample) = sample {
            samples.push(format!("{context}: {sample}"));
        }
    }
}



fn table_schema(table: &TableSpec) -> Result<Schema> {
    let mut fields = Vec::with_capacity(table.columns.len());
    for col in &table.columns {
        fields.push(Field::new(&col.name, DataType::Utf8, true));
    }
    Ok(Schema::new(fields))
}

fn rowpack_config_for_table(
    source_cfg: Option<&RowPackConfig>,
    table: &TableSpec,
) -> RowPackConfig {
    RowPackConfig::merge(source_cfg, table.rowpack.as_ref())
}

pub(crate) fn build_row_access_hints(table: &TableSpec) -> Vec<RowAccessHints> {
    table
        .columns
        .iter()
        .map(|col| RowAccessHints {
            timestamp_format: col.timestamp_format.clone(),
            timestamp_unit: to_arrow_time_unit(col.timestamp_unit.as_ref()),
            decimal_scale: col.scale,
        })
        .collect()
}

fn to_arrow_time_unit(
    unit: Option<&crate::graph_config::TimestampUnit>,
) -> Option<arrow_schema::TimeUnit> {
    match unit? {
        crate::graph_config::TimestampUnit::Seconds => Some(arrow_schema::TimeUnit::Second),
        crate::graph_config::TimestampUnit::Millis => Some(arrow_schema::TimeUnit::Millisecond),
        crate::graph_config::TimestampUnit::Micros => Some(arrow_schema::TimeUnit::Microsecond),
        crate::graph_config::TimestampUnit::Nanos => Some(arrow_schema::TimeUnit::Nanosecond),
    }
}

fn rowpack_columns(table: &TableSpec) -> Result<Vec<diskgraph_rowpack::RowPackColumn>> {
    let mut columns = Vec::with_capacity(table.columns.len());
    for col in &table.columns {
        let dtype = match col.ingested_dtype {
            IngestedType::Int8 => RowpackType::Int8,
            IngestedType::Int16 => RowpackType::Int16,
            IngestedType::Int64 => RowpackType::Int64,
            IngestedType::Int32 => RowpackType::Int32,
            IngestedType::UInt8 => RowpackType::UInt8,
            IngestedType::UInt16 => RowpackType::UInt16,
            IngestedType::UInt64 => RowpackType::UInt64,
            IngestedType::UInt32 => RowpackType::UInt32,
            IngestedType::Float64 => RowpackType::Float64,
            IngestedType::Float32 => RowpackType::Float32,
            IngestedType::Bool => RowpackType::Bool,
            IngestedType::String => RowpackType::String,
            IngestedType::Bytes => RowpackType::Bytes,
            IngestedType::Uuid => RowpackType::Uuid,
            IngestedType::Uuid256 => RowpackType::Uuid256,
            IngestedType::Ip => RowpackType::Ip,
            IngestedType::Date => RowpackType::Date,
            IngestedType::Timestamp => RowpackType::Timestamp,
            IngestedType::Decimal128 => RowpackType::Decimal128,
            IngestedType::Decimal256 => RowpackType::Decimal256,
            IngestedType::Float32Array => RowpackType::Float32Array,
            IngestedType::Float64Array => RowpackType::Float64Array,
            IngestedType::Float16Array => RowpackType::Float16Array,
        };
        let fixed_len = match col.ingested_dtype {
            IngestedType::Uuid | IngestedType::Uuid256 | IngestedType::Ip => None,
            _ => col.fixed_len,
        };
        columns.push(diskgraph_rowpack::RowPackColumn {
            name: col.name.clone(),
            logical_type: dtype,
            fixed_len,
            precision: col.precision,
            scale: col.scale,
        });
    }
    Ok(columns)
}

struct PrecomputedKeys {
    shards: Vec<Option<usize>>,
    keys: Option<Vec<Option<Key>>>,
}

fn precompute_shards(
    pk_array: Option<&ArrayRef>,
    pk_col: Option<&ColumnSpec>,
    num_shards: usize,
) -> Option<PrecomputedKeys> {
    if num_shards <= 1 {
        return None;
    }
    let array = pk_array?;
    let pk_col = pk_col?;
    let len = array.len();

    match (array.data_type(), &pk_col.ingested_dtype) {
        (DataType::Int64, IngestedType::Int64) => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
            let mut shards = Vec::with_capacity(len);
            for i in 0..len {
                if arr.is_null(i) {
                    shards.push(None);
                } else {
                    let val = arr.value(i) as u64;
                    let mut hasher = GxHasher::with_seed(GXHASH_SHARD_SEED as i64);
                    hasher.write(&val.to_le_bytes());
                    shards.push(Some((hasher.finish() as usize) % num_shards));
                }
            }
            Some(PrecomputedKeys { shards, keys: None })
        }
        (DataType::Int64, IngestedType::Int32) => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
            let mut shards = Vec::with_capacity(len);
            let mut keys = Vec::with_capacity(len);
            for i in 0..len {
                if arr.is_null(i) {
                    shards.push(None);
                    keys.push(None);
                } else {
                    let val = arr.value(i) as i32 as u64;
                    let mut hasher = GxHasher::with_seed(GXHASH_SHARD_SEED as i64);
                    hasher.write(&val.to_le_bytes());
                    shards.push(Some((hasher.finish() as usize) % num_shards));
                    keys.push(Some(Key::Int(val)));
                }
            }
            Some(PrecomputedKeys { shards, keys: Some(keys) })
        }
        (DataType::Utf8 | DataType::LargeUtf8, IngestedType::Uuid) => {
            let mut shards = Vec::with_capacity(len);
            let mut keys = Vec::with_capacity(len);
            for i in 0..len {
                if array.is_null(i) {
                    shards.push(None);
                    keys.push(None);
                } else {
                    let text = read_utf8(array, i).ok()?;
                    let uuid = parse_uuid_string(text.as_bytes()).ok()?;
                    let mut hasher = GxHasher::with_seed(GXHASH_SHARD_SEED as i64);
                    hasher.write(&uuid);
                    shards.push(Some((hasher.finish() as usize) % num_shards));
                    keys.push(Some(Key::Uuid(uuid)));
                }
            }
            Some(PrecomputedKeys { shards, keys: Some(keys) })
        }
        (DataType::FixedSizeBinary(16), IngestedType::Uuid | IngestedType::Ip) => {
            let arr = array.as_any().downcast_ref::<arrow_array::FixedSizeBinaryArray>().unwrap();
            let mut shards = Vec::with_capacity(len);
            for i in 0..len {
                if arr.is_null(i) {
                    shards.push(None);
                } else {
                    let val = arr.value(i);
                    let mut hasher = GxHasher::with_seed(GXHASH_SHARD_SEED as i64);
                    hasher.write(val);
                    shards.push(Some((hasher.finish() as usize) % num_shards));
                }
            }
            Some(PrecomputedKeys { shards, keys: None })
        }
        (DataType::FixedSizeBinary(32), IngestedType::Uuid256) => {
            let arr = array.as_any().downcast_ref::<arrow_array::FixedSizeBinaryArray>().unwrap();
            let mut shards = Vec::with_capacity(len);
            for i in 0..len {
                if arr.is_null(i) {
                    shards.push(None);
                } else {
                    let val = arr.value(i);
                    let mut hasher = GxHasher::with_seed(GXHASH_SHARD_SEED as i64);
                    hasher.write(val);
                    shards.push(Some((hasher.finish() as usize) % num_shards));
                }
            }
            Some(PrecomputedKeys { shards, keys: None })
        }
        _ => None,
    }
}

fn shard_for_key(bytes: &[u8], num_shards: usize) -> usize {
    if num_shards <= 1 {
        return 0;
    }
    let mut hasher = GxHasher::with_seed(GXHASH_SHARD_SEED as i64);
    hasher.write(bytes);
    (hasher.finish() as usize) % num_shards
}

#[derive(Clone, Copy)]
enum EdgeKeyFastKind {
    Int64,
    Int32,
    Int16,
    Int8,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
    Int64ToInt32,
    StringToInt64,
    StringVar,
    BytesVar,
    StringUuid,
    BytesUuid,
    StringUuid256,
    BytesUuid256,
    StringIp,
    BytesIp,
}

fn edge_fast_path_for_array(col: &ColumnSpec, array: &ArrayRef) -> Option<EdgeKeyFastKind> {
    match (array.data_type(), &col.ingested_dtype) {
        (DataType::Int64, IngestedType::Int64) => Some(EdgeKeyFastKind::Int64),
        (DataType::Int32, IngestedType::Int32) => Some(EdgeKeyFastKind::Int32),
        (DataType::Int16, IngestedType::Int16) => Some(EdgeKeyFastKind::Int16),
        (DataType::Int8, IngestedType::Int8) => Some(EdgeKeyFastKind::Int8),
        (DataType::UInt64, IngestedType::UInt64) => Some(EdgeKeyFastKind::UInt64),
        (DataType::UInt32, IngestedType::UInt32) => Some(EdgeKeyFastKind::UInt32),
        (DataType::UInt16, IngestedType::UInt16) => Some(EdgeKeyFastKind::UInt16),
        (DataType::UInt8, IngestedType::UInt8) => Some(EdgeKeyFastKind::UInt8),
        (DataType::Int64, IngestedType::Int32) => Some(EdgeKeyFastKind::Int64ToInt32),
        (DataType::Utf8 | DataType::LargeUtf8, IngestedType::Int64) => {
            Some(EdgeKeyFastKind::StringToInt64)
        }
        (DataType::Utf8 | DataType::LargeUtf8, IngestedType::String) => {
            Some(EdgeKeyFastKind::StringVar)
        }
        (
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_),
            IngestedType::Bytes,
        ) => Some(EdgeKeyFastKind::BytesVar),
        (DataType::Utf8 | DataType::LargeUtf8, IngestedType::Uuid) => {
            Some(EdgeKeyFastKind::StringUuid)
        }
        (
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_),
            IngestedType::Uuid,
        ) => Some(EdgeKeyFastKind::BytesUuid),
        (DataType::Utf8 | DataType::LargeUtf8, IngestedType::Uuid256) => {
            Some(EdgeKeyFastKind::StringUuid256)
        }
        (
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_),
            IngestedType::Uuid256,
        ) => Some(EdgeKeyFastKind::BytesUuid256),
        (DataType::Utf8 | DataType::LargeUtf8, IngestedType::Ip) => {
            Some(EdgeKeyFastKind::StringIp)
        }
        (
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_),
            IngestedType::Ip,
        ) => Some(EdgeKeyFastKind::BytesIp),
        _ => None,
    }
}

fn extract_batch_keys(
    array: &ArrayRef,
    kind: EdgeKeyFastKind,
) -> Result<Vec<Option<Key>>> {
    let len = array.len();
    let mut out = Vec::with_capacity(len);
    match kind {
        EdgeKeyFastKind::Int64 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i) as u64))); }
            }
        }
        EdgeKeyFastKind::Int32 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i) as u64))); }
            }
        }
        EdgeKeyFastKind::Int16 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int16Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i) as u64))); }
            }
        }
        EdgeKeyFastKind::Int8 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int8Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i) as u64))); }
            }
        }
        EdgeKeyFastKind::UInt64 => {
            let arr = array.as_any().downcast_ref::<arrow_array::UInt64Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i)))); }
            }
        }
        EdgeKeyFastKind::UInt32 => {
            let arr = array.as_any().downcast_ref::<arrow_array::UInt32Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i) as u64))); }
            }
        }
        EdgeKeyFastKind::UInt16 => {
            let arr = array.as_any().downcast_ref::<arrow_array::UInt16Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i) as u64))); }
            }
        }
        EdgeKeyFastKind::UInt8 => {
            let arr = array.as_any().downcast_ref::<arrow_array::UInt8Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i) as u64))); }
            }
        }
        EdgeKeyFastKind::Int64ToInt32 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
            for i in 0..len {
                if arr.is_null(i) { out.push(None); }
                else { out.push(Some(Key::Int(arr.value(i) as i32 as u64))); }
            }
        }
        EdgeKeyFastKind::StringToInt64 => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<u64> {
                        let text = fast_string_value(array, i)?;
                        let v = parse_base10_bytes(text.as_bytes())?;
                        Ok(v)
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Int(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
        EdgeKeyFastKind::StringVar => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<Vec<u8>> {
                        let data = fast_string_value(array, i)?.as_bytes();
                        if data.len() > MAX_VAR_KEY_LEN { return Err(anyhow!("key too large")); }
                        Ok(data.to_vec())
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Bytes(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
        EdgeKeyFastKind::BytesVar => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<Vec<u8>> {
                        let data = fast_bytes_value(array, i)?;
                        if data.len() > MAX_VAR_KEY_LEN { return Err(anyhow!("key too large")); }
                        Ok(data.to_vec())
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Bytes(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
        EdgeKeyFastKind::StringUuid => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<[u8; 16]> {
                        let text = fast_string_value(array, i)?;
                        let uuid = parse_uuid_string(text.as_bytes())?;
                        Ok(uuid)
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Uuid(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
        EdgeKeyFastKind::BytesUuid => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<[u8; 16]> {
                        let data = fast_bytes_value(array, i)?;
                        if data.len() != 16 { return Err(anyhow!("length mismatch")); }
                        let mut buf = [0u8; 16];
                        buf.copy_from_slice(data);
                        Ok(buf)
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Uuid(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
        EdgeKeyFastKind::StringUuid256 => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<[u8; 32]> {
                        let text = fast_string_value(array, i)?;
                        let uuid = parse_uuid256_string(text.as_bytes())?;
                        Ok(uuid)
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Uuid256(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
        EdgeKeyFastKind::BytesUuid256 => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<[u8; 32]> {
                        let data = fast_bytes_value(array, i)?;
                        if data.len() != 32 { return Err(anyhow!("length mismatch")); }
                        let mut buf = [0u8; 32];
                        buf.copy_from_slice(data);
                        Ok(buf)
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Uuid256(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
        EdgeKeyFastKind::StringIp => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<[u8; 16]> {
                        let text = fast_string_value(array, i)?;
                        let ip = parse_ip_string(text.as_bytes())?;
                        Ok(ip)
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Ip(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
        EdgeKeyFastKind::BytesIp => {
            for i in 0..len {
                if array.is_null(i) { out.push(None); }
                else {
                    let res = (|| -> Result<[u8; 16]> {
                        let data = fast_bytes_value(array, i)?;
                        let ip = parse_ip_bytes(data)?;
                        Ok(ip)
                    })();
                    match res {
                        Ok(v) => out.push(Some(Key::Ip(v))),
                        Err(_) => out.push(None),
                    }
                }
            }
        }
    }
    Ok(out)
}

fn fast_string_value<'a>(
    array: &'a ArrayRef,
    row: usize,
) -> Result<&'a str, KeyParseError> {
    if let Some(arr) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
        Ok(arr.value(row))
    } else if let Some(arr) = array
        .as_any()
        .downcast_ref::<arrow_array::LargeStringArray>()
    {
        Ok(arr.value(row))
    } else {
        Err(KeyParseError::TypeMismatch)
    }
}

fn fast_bytes_value<'a>(
    array: &'a ArrayRef,
    row: usize,
) -> Result<&'a [u8], KeyParseError> {
    if let Some(arr) = array.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        Ok(arr.value(row))
    } else if let Some(arr) = array
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        Ok(arr.value(row))
    } else if let Some(arr) = array
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
    {
        Ok(arr.value(row))
    } else {
        Err(KeyParseError::TypeMismatch)
    }
}

fn read_utf8<'a>(

array: &'a ArrayRef, row: usize) -> Result<&'a str> {
    match array.data_type() {
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string cast"))?;
            Ok(arr.value(row))
        }
        DataType::LargeUtf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| anyhow!("large string cast"))?;
            Ok(arr.value(row))
        }
        DataType::Binary => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| anyhow!("binary cast"))?;
            std::str::from_utf8(arr.value(row)).map_err(|_| anyhow!("invalid utf8"))
        }
        DataType::LargeBinary => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .ok_or_else(|| anyhow!("large binary cast"))?;
            std::str::from_utf8(arr.value(row)).map_err(|_| anyhow!("invalid utf8"))
        }
        DataType::FixedSizeBinary(_) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                .ok_or_else(|| anyhow!("fixed binary cast"))?;
            std::str::from_utf8(arr.value(row)).map_err(|_| anyhow!("invalid utf8"))
        }
        _ => Err(anyhow!("unsupported utf8 source")),
    }
}

fn read_bytes<'a>(array: &'a ArrayRef, row: usize) -> Result<&'a [u8]> {
    match array.data_type() {
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string cast"))?;
            Ok(arr.value(row).as_bytes())
        }
        DataType::LargeUtf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| anyhow!("large string cast"))?;
            Ok(arr.value(row).as_bytes())
        }
        DataType::Binary => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| anyhow!("binary cast"))?;
            Ok(arr.value(row))
        }
        DataType::LargeBinary => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .ok_or_else(|| anyhow!("large binary cast"))?;
            Ok(arr.value(row))
        }
        DataType::FixedSizeBinary(_) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                .ok_or_else(|| anyhow!("fixed binary cast"))?;
            Ok(arr.value(row))
        }
        _ => Err(anyhow!("unsupported bytes source")),
    }
}

 fn value_from_array(array: &ArrayRef, row: usize) -> Result<JsonValue> {
    if array.is_null(row) {
        return Ok(JsonValue::Null);
    }
    match array.data_type() {
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .ok_or_else(|| anyhow!("int64 cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .ok_or_else(|| anyhow!("int32 cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .ok_or_else(|| anyhow!("uint64 cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .ok_or_else(|| anyhow!("uint32 cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Float64Array>()
                .ok_or_else(|| anyhow!("float64 cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or_else(|| anyhow!("float32 cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::BooleanArray>()
                .ok_or_else(|| anyhow!("bool cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::Binary => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| anyhow!("binary cast"))?;
            Ok(JsonValue::from(BASE64.encode(arr.value(row))))
        }
        DataType::Date32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Date32Array>()
                .ok_or_else(|| anyhow!("date cast"))?;
            Ok(JsonValue::from(arr.value(row)))
        }
        DataType::Timestamp(_, _) => {
            let (secs, _) = timestamp_parts_from_array(array, row)?;
            Ok(JsonValue::from(secs))
        }
        DataType::Decimal128(_, scale) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Decimal128Array>()
                .ok_or_else(|| anyhow!("decimal128 cast"))?;
            Ok(JsonValue::from(decimal_to_string_i128(
                arr.value(row),
                i32::from(*scale),
            )))
        }
        DataType::Decimal256(_, scale) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Decimal256Array>()
                .ok_or_else(|| anyhow!("decimal256 cast"))?;
            Ok(JsonValue::from(decimal_to_string_i256(
                arr.value(row),
                i32::from(*scale),
            )))
        }
        DataType::List(_) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::ListArray>()
                .ok_or_else(|| anyhow!("list cast"))?;
            let values = arr.value(row);
            let mut out = Vec::new();
            for i in 0..values.len() {
                let v = value_from_array(&values, i)?;
                out.push(v);
            }
            Ok(JsonValue::Array(out))
        }
        _ => Err(anyhow!("unsupported array type")),
    }
}

fn timestamp_parts_from_array(array: &ArrayRef, row: usize) -> Result<(i64, u32)> {
    let unit = match array.data_type() {
        DataType::Timestamp(unit, _) => *unit,
        _ => return Err(anyhow!("timestamp type mismatch")),
    };
    match unit {
        arrow_schema::TimeUnit::Second => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampSecondArray>()
                .ok_or_else(|| anyhow!("timestamp second cast"))?;
            Ok((arr.value(row), 0))
        }
        arrow_schema::TimeUnit::Millisecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                .ok_or_else(|| anyhow!("timestamp millisecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000);
            let millis = value.rem_euclid(1_000) as u32;
            Ok((secs, millis * 1_000_000))
        }
        arrow_schema::TimeUnit::Microsecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                .ok_or_else(|| anyhow!("timestamp microsecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000_000);
            let micros = value.rem_euclid(1_000_000) as u32;
            Ok((secs, micros * 1_000))
        }
        arrow_schema::TimeUnit::Nanosecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                .ok_or_else(|| anyhow!("timestamp nanosecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000_000_000);
            let nanos = value.rem_euclid(1_000_000_000) as u32;
            Ok((secs, nanos))
        }
    }
}

fn decimal_to_string_i128(value: i128, scale: i32) -> String {
    let big = num_bigint::BigInt::from(value);
    format_decimal_from_bigint(&big, scale)
}

fn decimal_to_string_i256(value: arrow_buffer::i256, scale: i32) -> String {
    let bytes = value.to_le_bytes();
    let big = num_bigint::BigInt::from_signed_bytes_le(&bytes);
    format_decimal_from_bigint(&big, scale)
}

fn format_decimal_from_bigint(value: &num_bigint::BigInt, scale: i32) -> String {
    let negative = value.is_negative();
    let mut digits = value.abs().to_str_radix(10);
    if scale <= 0 {
        let zeros = (-scale) as usize;
        if zeros > 0 {
            digits.push_str(&"0".repeat(zeros));
        }
        return if negative {
            format!("-{}", digits)
        } else {
            digits
        };
    }
    let scale = scale as usize;
    if digits.len() <= scale {
        let pad = scale - digits.len();
        let mut out = String::with_capacity(scale + 2);
        if negative {
            out.push('-');
        }
        out.push_str("0.");
        out.push_str(&"0".repeat(pad));
        out.push_str(&digits);
        return out;
    }
    let split = digits.len() - scale;
    let mut out = String::with_capacity(digits.len() + 1);
    if negative {
        out.push('-');
    }
    out.push_str(&digits[..split]);
    out.push('.');
    out.push_str(&digits[split..]);
    out
}
