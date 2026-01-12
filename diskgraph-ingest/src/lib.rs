//! Diskgraph ingest: convert raw CSV/Parquet sources into index-ready artifacts.

mod mapping;
pub mod source_metadata;

use crate::mapping::{
    fingerprint_seed_for, write_pk_data_entry, write_pk_data_header, GxPtrHash64, Key, KeyType,
    PkHashShard, PkHashStore, PkGxHash,
};
use crate::source_metadata::{
    relationship_name, ColumnSpec, IngestedType, SortKeySpec, TableFormat, TableSpec,
    TimestampEncodingKind, TimestampEncodingSpec, WeightEncodingKind, WeightEncodingSpec,
};
use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, EdgeSortKeyMetadata,
    EdgeSortKeyTimestampEncoding, EdgeSortKeyTimestampKind, EdgeSortKeyTimestampMetadata,
    EdgeSortKeyWeightEncoding, EdgeSortKeyWeightKind, EdgeSortKeyWeightMetadata, EdgeStorageFormat,
    NodeDataCompression, NodeShardMetadata, NodeTypeMetadata, PrimaryKeyMap,
};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig, S3Config,
};
use diskgraph_rowpack::{
    IngestedType as RowpackType, RowAccess, RowAccessHints, RowAccessStats, RowPackConfig,
    RowPackContextBuilder, RowPackEncoder, RowPackRowLens,
};
use diskgraph_index::builder::MemoryBudget;
use diskgraph_index::graph::{
    GraphBuildProgress, GraphBuildStage, GraphDataSource, GraphIndexBuilder, GraphProgressSink,
};
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use fasteval::{eval_compiled_ref, Compiler, Evaler, Parser, Slab};
use half::f16;
use memmap2::Mmap;
use num_traits::{Signed, ToPrimitive};
use object_store::{path::Path as ObjectPath, ObjectStore, WriteMultipart};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pprof::protos::Message;
use pprof::ProfilerGuard;
use serde::Serialize;
use serde_json::Value as JsonValue;
use gxhash::GxHasher;
use epserde::prelude::Serialize as EpSerialize;
use ptr_hash::DefaultPtrHash;
use std::fs;
use std::io::{self, BufWriter, Read, Write};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use std::process::{Command, Stdio};
use tokio::runtime::Runtime;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use uuid_simd::{parse_hyphenated, parse_simple, AsOut};

pub use source_metadata::SourceMetadata;

const NODE_BATCH_SIZE: usize = 100_000;
const MAX_KEY_SAMPLE_LEN: usize = 64;
const TEMP_DIR_NAME: &str = "_tmp";
const TEMP_META_MAGIC: [u8; 4] = *b"DGTP";
const TEMP_META_VERSION: u8 = 2;
const SAMPLE_ROW_LIMIT: usize = 100_000;
const MAX_NODE_DATA_LEN: usize = 64 * 1024 * 1024;
const PROGRESS_ROW_BATCH: u64 = 1_000_000;
const MAX_VAR_KEY_LEN: usize = u16::MAX as usize;
const GXHASH_SHARD_SEED: u64 = 0x6a09e667f3bcc909;

#[derive(Clone, Copy, Debug)]
enum KeyParseError {
    TypeMismatch,
    UnsupportedType,
    LengthMismatch,
    Empty,
    TooLarge,
    NotInteger,
    NotBase10,
    NotUtf8,
    Negative,
    InvalidUuid,
    InvalidUuid256,
    InvalidIp,
}

type KeyParseResult<T> = std::result::Result<T, KeyParseError>;

const DEFAULT_INDEX_MEM_BUDGET_FRAC: f64 = 0.9;

#[derive(Clone, Debug, Serialize)]
pub struct IngestReport {
    pub stats: IngestStats,
    pub source: SourceMetadata,
}

#[derive(Clone)]
pub struct IngestOptions {
    pub profile_dir: Option<PathBuf>,
    pub progress: Option<ProgressCallback>,
    pub parallel: bool,
    pub index_mem_budget_bytes: Option<u64>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProgressPhase {
    Context,
    Encode,
    PkMap,
    EdgesPassA,
    EdgesPassB,
    EdgesPassC,
    Index,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressKind {
    Start,
    Advance,
    Finish,
}

#[derive(Clone, Debug)]
pub struct ProgressEvent {
    pub phase: ProgressPhase,
    pub table: String,
    pub shard: Option<usize>,
    pub kind: ProgressKind,
    pub delta: u64,
    pub total: Option<u64>,
}

pub type ProgressCallback = std::sync::Arc<dyn Fn(ProgressEvent) + Send + Sync>;

struct IndexProgressAdapter {
    cb: ProgressCallback,
    state: Mutex<IndexProgressTracker>,
}

struct IndexProgressTracker {
    shard_states: HashMap<(String, Option<usize>), IndexProgressState>,
    table_states: HashMap<String, TableProgressState>,
}

#[derive(Clone, Copy)]
struct IndexProgressState {
    current: u64,
}

#[derive(Clone, Copy)]
struct TableProgressState {
    current: u64,
    total: Option<u64>,
    remaining_shards: Option<usize>,
    started: bool,
    finished: bool,
}

#[derive(Clone, Copy)]
struct TableIndexTotals {
    total_nodes: u64,
    total_shards: usize,
}

impl IndexProgressAdapter {
    fn new(cb: ProgressCallback, table_totals: HashMap<String, TableIndexTotals>) -> Self {
        let mut table_states = HashMap::new();
        for (table, totals) in table_totals {
            table_states.insert(
                table,
                TableProgressState {
                    current: 0,
                    total: Some(totals.total_nodes),
                    remaining_shards: Some(totals.total_shards),
                    started: false,
                    finished: false,
                },
            );
        }
        Self {
            cb,
            state: Mutex::new(IndexProgressTracker {
                shard_states: HashMap::new(),
                table_states,
            }),
        }
    }

    fn shard_key(shard: &str) -> (String, Option<usize>) {
        if let Some((table, rest)) = shard.split_once("/shard-") {
            if let Ok(idx) = rest.parse::<usize>() {
                return (table.to_string(), Some(idx));
            }
        }
        (shard.to_string(), None)
    }
}

impl GraphProgressSink for IndexProgressAdapter {
    fn report(&self, update: GraphBuildProgress) {
        let (table, shard) = match &update.stage {
            GraphBuildStage::BuildShard { shard }
            | GraphBuildStage::Complete { shard } => Self::shard_key(shard.as_ref()),
        };
        let key = (table.clone(), shard);
        let mut state = self.state.lock().unwrap();
        match update.stage {
            GraphBuildStage::BuildShard { .. } => {
                let delta = match state.shard_states.entry(key) {
                    Entry::Vacant(v) => {
                        v.insert(IndexProgressState {
                            current: update.current,
                        });
                        update.current
                    }
                    Entry::Occupied(mut o) => {
                        let prev = o.get().current;
                        if update.current > prev {
                            let delta = update.current.saturating_sub(prev);
                            o.insert(IndexProgressState {
                                current: update.current,
                            });
                            delta
                        } else {
                            0
                        }
                    }
                };
                let table_state = state.table_states.entry(table.clone()).or_insert(
                    TableProgressState {
                        current: 0,
                        total: None,
                        remaining_shards: None,
                        started: false,
                        finished: false,
                    },
                );
                if !table_state.started && !table_state.finished {
                    table_state.started = true;
                    (self.cb)(ProgressEvent {
                        phase: ProgressPhase::Index,
                        table: table.clone(),
                        shard: None,
                        kind: ProgressKind::Start,
                        delta: 0,
                        total: table_state.total,
                    });
                }
                if delta > 0 {
                    table_state.current = table_state.current.saturating_add(delta);
                    (self.cb)(ProgressEvent {
                        phase: ProgressPhase::Index,
                        table,
                        shard: None,
                        kind: ProgressKind::Advance,
                        delta,
                        total: table_state.total,
                    });
                }
            }
            GraphBuildStage::Complete { .. } => {
                state.shard_states.remove(&key);
                let table_state = state.table_states.entry(table.clone()).or_insert(
                    TableProgressState {
                        current: 0,
                        total: None,
                        remaining_shards: None,
                        started: false,
                        finished: false,
                    },
                );
                if let Some(remaining) = table_state.remaining_shards.as_mut() {
                    if *remaining > 0 {
                        *remaining = remaining.saturating_sub(1);
                    }
                }
                let should_finish = match table_state.remaining_shards {
                    Some(remaining) => remaining == 0,
                    None => true,
                };
                if should_finish && !table_state.finished {
                    table_state.finished = true;
                    (self.cb)(ProgressEvent {
                        phase: ProgressPhase::Index,
                        table,
                        shard: None,
                        kind: ProgressKind::Finish,
                        delta: 0,
                        total: table_state.total.or(Some(table_state.current)),
                    });
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct IngestStats {
    pub output: String,
    pub node_tables: usize,
    pub edge_relations: usize,
    pub total_nodes: u64,
    pub total_edges: u64,
    pub node_counts: HashMap<String, u64>,
    pub edge_counts: HashMap<String, u64>,
    pub key_parse_failures: u64,
    pub key_lookup_failures: u64,
    pub key_failure_samples: Vec<String>,
    pub key_failures_by_context: HashMap<String, KeyFailureCounts>,
    pub pk_duplicate_counts: HashMap<String, u64>,
    pub pk_duplicate_total: u64,
    pub pk_duplicate_samples: HashMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trimmed_strings: Option<u64>,
    pub perf: PerfStats,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct KeyFailureCounts {
    pub parse_failures: u64,
    pub lookup_failures: u64,
}

#[derive(Default)]
struct KeyFailureState {
    parse_failures: u64,
    lookup_failures: u64,
    samples: Vec<String>,
    by_context: HashMap<String, KeyFailureCounts>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PerfStats {
    pub wall_time_ms: u64,
    pub context_wall_ms: u64,
    pub encode_wall_ms: u64,
    pub pk_map_wall_ms: u64,
    pub edges_map_wall_ms: u64,
    pub index_wall_ms: u64,
    pub finalize_wall_ms: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub temp_edges_src_bytes: u64,
    pub temp_fkeys_bytes: u64,
    pub temp_edges_map_bytes: u64,
    pub temp_node_data_bytes: u64,
    pub temp_bytes_total: u64,
}

pub fn ingest(
    source_metadata_path: &Path,
    output: &str,
    dest: Dest,
    options: IngestOptions,
) -> Result<DatasetMetadata> {
    let (metadata, _) = ingest_with_stats(source_metadata_path, output, dest, options)?;
    Ok(metadata)
}

pub fn ingest_with_stats(
    source_metadata_path: &Path,
    output: &str,
    dest: Dest,
    options: IngestOptions,
) -> Result<(DatasetMetadata, IngestStats)> {
    let wall_start = std::time::Instant::now();
    let metadata_bytes = fs::read(source_metadata_path)
        .with_context(|| format!("read {}", source_metadata_path.display()))?;
    let mut source: SourceMetadata = serde_json::from_slice(&metadata_bytes)?;
    let base_dir = source_metadata_path
        .parent()
        .ok_or_else(|| anyhow!("missing parent for {}", source_metadata_path.display()))?;
    resolve_local_paths(&mut source, base_dir);
    source.validate()?;

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

    let mut ctx = IngestContext::new(source, target, options, output)?;
    let profile_dir = ctx.profile_dir.clone();
    if let Some(profile_dir) = &profile_dir {
        fs::create_dir_all(profile_dir)?;
    }
    let context_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "context", || ctx.build_node_contexts())?;
    } else {
        ctx.build_node_contexts()?;
    }
    let context_wall = context_start.elapsed();
    let encode_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "encode", || ctx.build_temp_shards())?;
    } else {
        ctx.build_temp_shards()?;
    }
    let encode_wall = encode_start.elapsed();
    let pk_map_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "pk_map", || ctx.build_sharded_pk_maps())?;
    } else {
        ctx.build_sharded_pk_maps()?;
    }
    let pk_map_wall = pk_map_start.elapsed();
    let edge_ctx = ctx.edge_build_context()?;
    let edges_map_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "edges_map", || ctx.build_edges_mapped(&edge_ctx))?;
    } else {
        ctx.build_edges_mapped(&edge_ctx)?;
    }
    let edges_map_wall = edges_map_start.elapsed();
    let index_start = std::time::Instant::now();
    let index_metadata = ctx.build_index_metadata()?;
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "index", || ctx.build_index(index_metadata.clone()))?;
    } else {
        ctx.build_index(index_metadata.clone())?;
    }
    let index_wall = index_start.elapsed();
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
    };
    let stats = ctx.build_stats(&index_metadata, perf);
    let report = IngestReport {
        stats: stats.clone(),
        source: ctx.source.clone(),
    };
    ctx.target
        .write_bytes("ingest_stats.json", serde_json::to_vec_pretty(&report)?)?;
    Ok((index_metadata, stats))
}

#[derive(Clone, Copy, Debug)]
pub enum Dest {
    Local,
    S3,
    Gcs,
    Adls,
}

struct IngestContext {
    source: SourceMetadata,
    target: std::sync::Arc<OutputTarget>,
    profile_dir: Option<PathBuf>,
    pool: std::sync::Arc<TaskPool>,
    index_mem_budget: Option<std::sync::Arc<MemoryBudget>>,
    num_shards: usize,
    table_rows: HashMap<String, u64>,
    table_contexts: HashMap<String, String>,
    shard_rows: HashMap<String, Vec<u64>>,
    shard_offsets: HashMap<String, Vec<u64>>,
    pk_hash_paths: HashMap<String, Vec<PkHashPaths>>,
    relation_defs: Vec<RelationDef>,
    relationships: Vec<RelationshipRuntime>,
    edge_counts: HashMap<String, u64>,
    pk_duplicate_counts: HashMap<String, u64>,
    pk_duplicate_samples: HashMap<String, Vec<String>>,
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
}

struct EdgeBuildContext {
    root: PathBuf,
    rel_sort_bits: HashMap<String, Option<u8>>,
    rel_sort_keys: HashMap<String, RelationSortKeyInfo>,
    max_ids: HashMap<String, u64>,
    pk_hash_store: Arc<PkHashStore>,
    edge_tables: HashSet<String>,
}

struct SortKeyComponentStore {
    width_bytes: usize,
    data: Mmap,
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

fn spawn_delete_process(path: &Path) -> io::Result<()> {
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

fn build_relation_sort_key_config(spec: &SortKeySpec) -> Option<RelationSortKeyConfig> {
    let mut width_bits: u16 = 0;
    let timestamp = spec.timestamp.as_ref().map(|ts| {
        width_bits += timestamp_width_bits(&ts.encoding.kind) as u16;
        SortKeyTimestampConfig {
            table: ts.table.clone(),
            column: ts.column.clone(),
            encoding: ts.encoding.clone(),
        }
    });
    let weight = spec.weight.as_ref().map(|w| {
        width_bits += weight_width_bits(&w.encoding.dtype) as u16;
        SortKeyWeightConfig {
            table: w.table.clone(),
            expr: w.expr.clone(),
            encoding: w.encoding.clone(),
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

fn encode_timestamp_value(
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

fn ordered_signed_to_u64(value: i64, bits: u8) -> u64 {
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
        source: SourceMetadata,
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
        let sort_keys = source.edge_sort_keys.clone().unwrap_or_default();
        for rel in &source.relationships {
            let base = relationship_name(rel);
            let rel_sort = sort_keys.relationships.get(&base);
            let dst_sort = rel_sort
                .and_then(|cfg| cfg.forward.as_ref())
                .and_then(build_relation_sort_key_config);
            let src_sort = rel_sort
                .and_then(|cfg| cfg.reverse.as_ref())
                .and_then(build_relation_sort_key_config);
            relation_defs.push(RelationDef {
                name: base.clone(),
                src_table: rel.src_table.clone(),
                dst_table: rel.dst_table.clone(),
                sort_key: dst_sort,
            });
            relation_defs.push(RelationDef {
                name: format!("{base}_reverse"),
                src_table: rel.dst_table.clone(),
                dst_table: rel.src_table.clone(),
                sort_key: src_sort,
            });
            relationships.push(RelationshipRuntime {
                name_forward: base.clone(),
                name_reverse: format!("{base}_reverse"),
                src_table: rel.src_table.clone(),
                src_column: rel.src_column.clone(),
                dst_table: rel.dst_table.clone(),
            });
        }
        for table in &source.tables {
            if let Some(edge) = &table.edge_table {
                let base = format!("edge_{}", table.name);
                let edge_sort = sort_keys.edge_tables.get(&table.name);
                let src_rel = RelationDef {
                    name: format!("{base}_src_to_dst"),
                    src_table: edge.src_fk.ref_table.clone(),
                    dst_table: edge.dst_fk.ref_table.clone(),
                    sort_key: edge_sort
                        .and_then(|cfg| cfg.src_to_dst.as_ref())
                        .and_then(build_relation_sort_key_config),
                };
                let dst_rel = RelationDef {
                    name: format!("{base}_dst_to_src"),
                    src_table: edge.dst_fk.ref_table.clone(),
                    dst_table: edge.src_fk.ref_table.clone(),
                    sort_key: edge_sort
                        .and_then(|cfg| cfg.dst_to_src.as_ref())
                        .and_then(build_relation_sort_key_config),
                };
                relation_defs.push(src_rel);
                relation_defs.push(dst_rel);
            }
        }
        Ok(Self {
            source,
            target,
            profile_dir: options.profile_dir,
            pool: std::sync::Arc::new(TaskPool::new(if options.parallel {
                default_worker_threads()
            } else {
                1
            })),
            index_mem_budget,
            num_shards,
            table_rows: HashMap::new(),
            table_contexts: HashMap::new(),
            shard_rows: HashMap::new(),
            shard_offsets: HashMap::new(),
            pk_hash_paths: HashMap::new(),
            relation_defs,
            relationships,
            edge_counts: HashMap::new(),
            pk_duplicate_counts: HashMap::new(),
            pk_duplicate_samples: HashMap::new(),
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
        })
    }

    fn build_node_contexts(&mut self) -> Result<()> {
        let tables = self.source.tables.clone();
        let (tx, rx) = std::sync::mpsc::channel::<Result<(String, String)>>();
        let mut tasks = Vec::new();
        let source_rowpack = self.source.rowpack.clone();
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
            tasks.push(WorkItem {
                bucket: None,
                run: Box::new(move || {
                    if let Some(cb) = &progress {
                        cb(ProgressEvent {
                            phase: ProgressPhase::Context,
                            table: table.name.clone(),
                            shard: None,
                            kind: ProgressKind::Start,
                            delta: 0,
                            total: Some(table.paths.len() as u64),
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
                    if table.paths.is_empty() {
                        return Err(anyhow!("no input files for table {}", table.name));
                    }
                    'file_loop: for path in &table.paths {
                        if !is_cloud_path(path) && !Path::new(path).exists() {
                            return Err(anyhow!("input path not found {}", path));
                        }
                        let local = localize_input(path)?;
                        match table.format {
                            TableFormat::Parquet => {
                                let file = fs::File::open(&local.path)
                                    .with_context(|| format!("open {}", local.path.display()))?;
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
                            TableFormat::Csv => {
                                let file = fs::File::open(&local.path)
                                    .with_context(|| format!("open {}", local.path.display()))?;
                                let schema = table_schema(&table)?;
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
                        processed_paths = processed_paths.saturating_add(1);
                        if let Some(cb) = &progress {
                            cb(ProgressEvent {
                                phase: ProgressPhase::Context,
                                table: table.name.clone(),
                                shard: None,
                                kind: ProgressKind::Advance,
                                delta: 1,
                                total: Some(table.paths.len() as u64),
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
        run_tasks(tasks, &self.pool)?;
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
        let sort_key_pool =
            SortKeyWritePool::new(std::sync::Arc::clone(&target), edge_writer_workers());
        let node_senders = node_pool.senders();
        let edge_senders = edge_pool.senders();
        let sort_key_senders = sort_key_pool.senders();
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
                    total: Some(table.paths.len() as u64),
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
            let file_count = table.paths.len();
            for file_idx in 0..file_count {
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
                let tx = tx.clone();
                tasks.push(WorkItem {
                    bucket: None,
                    run: Box::new(move || {
                        let mut key_failures = KeyFailureState::default();
                        if table.edge_table.is_some() {
                            Self::build_temp_edge_file(
                                &table,
                                file_idx,
                                num_shards,
                                &target,
                                &trimmed_strings,
                                &progress,
                                &mut key_failures,
                                &edge_senders,
                                &sort_key_senders,
                                relation_defs.as_ref(),
                            )?;
                            tx.send(Ok(EncodeResult {
                                table: table.name.clone(),
                                total_rows: 0,
                                node_data_dropped: 0,
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
                            &trimmed_strings,
                            &progress,
                            &mut key_failures,
                            &node_senders,
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
        run_tasks(tasks, &self.pool)?;
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
                let key = format!("node_data_dropped:{}", result.table);
                let entry = self.pk_duplicate_counts.entry(key).or_insert(0);
                *entry = entry.saturating_add(result.node_data_dropped);
            }
            self.merge_key_failures(result.key_failures);
            if let Some(cb) = &self.progress {
                let expected = tables
                    .iter()
                    .find(|t| t.name == result.table)
                    .map(|t| t.paths.len() as u64)
                    .unwrap_or(0);
                cb(ProgressEvent {
                    phase: ProgressPhase::Encode,
                    table: result.table.clone(),
                    shard: None,
                    kind: ProgressKind::Finish,
                    delta: 0,
                    total: Some(expected),
                });
            }
        }
        drop(node_senders);
        drop(edge_senders);
        drop(sort_key_senders);
        let node_stats = node_pool.finish()?;
        let edge_stats = edge_pool.finish()?;
        let _sort_key_stats = sort_key_pool.finish()?;
        self.write_consolidated_meta(&node_stats, &edge_stats)?;
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
        trimmed_strings: &std::sync::Arc<AtomicU64>,
        progress: &Option<ProgressCallback>,
        key_failures: &mut KeyFailureState,
        senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>>,
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
        let mut sort_key_batchers = build_sort_key_batchers_for_node_table(
            table,
            relation_defs,
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(sort_key_senders),
        )?;

        let path = table
            .paths
            .get(file_idx)
            .ok_or_else(|| anyhow!("missing input file {}", file_idx))?;
        if !is_cloud_path(path) && !Path::new(path).exists() {
            return Err(anyhow!("input path not found {}", path));
        }
        let local = localize_input(path)?;
        let mut row_id = 0u64;
        match table.format {
            TableFormat::Parquet => {
                let file = fs::File::open(&local.path)
                    .with_context(|| format!("open {}", local.path.display()))?;
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
                    for row in 0..batch.num_rows() {
                        total_rows += 1;
                        access.set_row(row);
                        Self::write_node_row_from_access(
                            table,
                            pk_idx,
                            &rel_specs,
                            &mut access,
                            &node_encoder,
                            &mut batcher,
                            &mut sort_key_batchers,
                            &mut row_id,
                            pk_col,
                            &mut key_bufs,
                            &mut node_data_dropped,
                            num_shards,
                            key_failures,
                        )?;
                    }
                }
            }
            TableFormat::Csv => {
                let file = fs::File::open(&local.path)
                    .with_context(|| format!("open {}", local.path.display()))?;
                let schema = table_schema(table)?;
                let builder = arrow_csv::ReaderBuilder::new(Arc::new(schema)).with_header(true);
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
                        total_rows += 1;
                        access.set_row(row);
                        Self::write_node_row_from_access(
                            table,
                            pk_idx,
                            &rel_specs,
                            &mut access,
                            &node_encoder,
                            &mut batcher,
                            &mut sort_key_batchers,
                            &mut row_id,
                            pk_col,
                            &mut key_bufs,
                            &mut node_data_dropped,
                            num_shards,
                            key_failures,
                        )?;
                    }
                }
            }
        }
        batcher.flush_all()?;
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
                total: Some(table.paths.len() as u64),
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
        access: &mut RowAccess<'_>,
        node_encoder: &Option<RowPackEncoder>,
        batcher: &mut NodeBatcher,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
        row_id: &mut u64,
        pk_col: Option<&ColumnSpec>,
        key_bufs: &mut Vec<Vec<u8>>,
        node_data_dropped: &mut u64,
        num_shards: usize,
        key_failures: &mut KeyFailureState,
    ) -> Result<()> {
        for buf in key_bufs.iter_mut() {
            buf.clear();
        }
        let mut has_pkey = false;
        let shard = if let (Some(idx), Some(pk_col)) = (pk_idx, pk_col) {
            let buf = key_bufs
                .get_mut(idx)
                .ok_or_else(|| anyhow!("missing key buffer"))?;
            match key_from_access_into_buf(access, idx, pk_col, buf) {
                Ok(_) => {}
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
            };
            has_pkey = true;
            shard_for_key(buf, num_shards)
        } else {
            (*row_id as usize) % num_shards
        };
        for (_rel_name, rel_idx, _rel_pos) in rel_specs {
            let col = table
                .columns
                .get(*rel_idx)
                .ok_or_else(|| anyhow!("missing fk column"))?;
            let buf = key_bufs
                .get_mut(*rel_idx)
                .ok_or_else(|| anyhow!("missing key buffer"))?;
            if key_from_access_into_buf(access, *rel_idx, col, buf).is_err() {
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
            batcher.push_pkey(shard, buf)?;
        }
        for sort_key in sort_key_batchers.iter_mut() {
            sort_key.push_row(shard, access)?;
        }
        for (_rel_name, rel_idx, rel_pos) in rel_specs {
            if let Some(buf) = key_bufs.get(*rel_idx) {
                if buf.is_empty() {
                    batcher.push_fkey_empty(shard, *rel_pos)?;
                } else {
                    batcher.push_fkey(shard, *rel_pos, buf)?;
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
        trimmed_strings: &std::sync::Arc<AtomicU64>,
        progress: &Option<ProgressCallback>,
        key_failures: &mut KeyFailureState,
        senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>>,
        sort_key_senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<SortKeyWriteMessage>>>,
        relation_defs: &[RelationDef],
    ) -> Result<()> {
        let edge = table
            .edge_table
            .as_ref()
            .ok_or_else(|| anyhow!("missing edge table spec for {}", table.name))?;
        let src_fk_idx = column_index(table, &edge.src_fk.column)?;
        let dst_fk_idx = column_index(table, &edge.dst_fk.column)?;
        let src_fk_col = column_spec(table, &edge.src_fk.column)?;
        let dst_fk_col = column_spec(table, &edge.dst_fk.column)?;
        let src_context = format!("edge_fk:{}.{}", table.name, edge.src_fk.column);
        let dst_context = format!("edge_fk:{}.{}", table.name, edge.dst_fk.column);
        let schema = table_schema(table)?;
        let rel_name = std::sync::Arc::<str>::from(format!("edge_{}_src_to_dst", table.name));
        let buffer_limit = edge_batch_buffer_limit(num_shards);
        let mut batcher = EdgeSrcBatcher::new(
            std::sync::Arc::clone(&rel_name),
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(senders),
        );
        let mut sort_key_batchers = build_sort_key_batchers_for_edge_table(
            table,
            relation_defs,
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(sort_key_senders),
        )?;

        let path = table
            .paths
            .get(file_idx)
            .ok_or_else(|| anyhow!("missing input file {}", file_idx))?;
        if !is_cloud_path(path) && !Path::new(path).exists() {
            return Err(anyhow!("input path not found {}", path));
        }
        let local = localize_input(path)?;
        let row_access_hints = build_row_access_hints(table);
        let stats = RowAccessStats::new(Some(trimmed_strings.as_ref()));
        let mut access = RowAccess::new(row_access_hints, stats);
        match table.format {
            TableFormat::Parquet => {
                let file = fs::File::open(&local.path)
                    .with_context(|| format!("open {}", local.path.display()))?;
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
                    for row in 0..batch.num_rows() {
                        access.set_row(row);
                        Self::write_edge_row_from_access(
                            src_fk_idx,
                            dst_fk_idx,
                            src_fk_col,
                            dst_fk_col,
                            &src_context,
                            &dst_context,
                            &cols,
                            row,
                            src_fast,
                            dst_fast,
                            &mut access,
                            &mut src_key_buf,
                            &mut dst_key_buf,
                            &mut forward_row,
                            &mut batcher,
                            &mut sort_key_batchers,
                            num_shards,
                            key_failures,
                        )?;
                    }
                }
            }
            TableFormat::Csv => {
                let file = fs::File::open(&local.path)
                    .with_context(|| format!("open {}", local.path.display()))?;
                let builder = arrow_csv::ReaderBuilder::new(Arc::new(schema.clone()))
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
                    let src_fast = edge_fast_path_for_array(src_fk_col, &cols[src_fk_idx]);
                    let dst_fast = edge_fast_path_for_array(dst_fk_col, &cols[dst_fk_idx]);
                    let mut src_key_buf = Vec::new();
                    let mut dst_key_buf = Vec::new();
                    let mut forward_row = Vec::new();
                    for row in 0..batch.num_rows() {
                        access.set_row(row);
                        Self::write_edge_row_from_access(
                            src_fk_idx,
                            dst_fk_idx,
                            src_fk_col,
                            dst_fk_col,
                            &src_context,
                            &dst_context,
                            &cols,
                            row,
                            src_fast,
                            dst_fast,
                            &mut access,
                            &mut src_key_buf,
                            &mut dst_key_buf,
                            &mut forward_row,
                            &mut batcher,
                            &mut sort_key_batchers,
                            num_shards,
                            key_failures,
                        )?;
                    }
                }
            }
        }
        batcher.flush_all()?;
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
                total: Some(table.paths.len() as u64),
            });
        }
        Ok(())
    }


    fn write_edge_row_from_access(
        src_idx: usize,
        dst_idx: usize,
        src_col: &ColumnSpec,
        dst_col: &ColumnSpec,
        src_context: &str,
        dst_context: &str,
        cols: &[ArrayRef],
        row: usize,
        src_fast: Option<EdgeKeyFastKind>,
        dst_fast: Option<EdgeKeyFastKind>,
        access: &mut RowAccess<'_>,
        src_key_buf: &mut Vec<u8>,
        dst_key_buf: &mut Vec<u8>,
        forward_row: &mut Vec<u8>,
        batcher: &mut EdgeSrcBatcher,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
        num_shards: usize,
        key_failures: &mut KeyFailureState,
    ) -> Result<()> {
        src_key_buf.clear();
        dst_key_buf.clear();
        match src_fast {
            Some(kind) => match fast_key_from_array(cols, row, src_idx, kind) {
                Ok(key) => {
                    if write_key_to_buf(src_key_buf, &key).is_err() {
                        record_key_parse_failure_sample(
                            &mut key_failures.parse_failures,
                            &mut key_failures.samples,
                            &mut key_failures.by_context,
                            src_context,
                            sample_from_access(access, src_idx),
                        );
                        return Ok(());
                    }
                }
                Err(_) => {
                    record_key_parse_failure_sample(
                        &mut key_failures.parse_failures,
                        &mut key_failures.samples,
                        &mut key_failures.by_context,
                        src_context,
                        sample_from_access(access, src_idx),
                    );
                    return Ok(());
                }
            },
            None => {
                if key_from_access_into_buf(access, src_idx, src_col, src_key_buf).is_err() {
                    record_key_parse_failure_sample(
                        &mut key_failures.parse_failures,
                        &mut key_failures.samples,
                        &mut key_failures.by_context,
                        src_context,
                        sample_from_access(access, src_idx),
                    );
                    return Ok(());
                }
            }
        }
        match dst_fast {
            Some(kind) => match fast_key_from_array(cols, row, dst_idx, kind) {
                Ok(key) => {
                    if write_key_to_buf(dst_key_buf, &key).is_err() {
                        record_key_parse_failure_sample(
                            &mut key_failures.parse_failures,
                            &mut key_failures.samples,
                            &mut key_failures.by_context,
                            dst_context,
                            sample_from_access(access, dst_idx),
                        );
                        return Ok(());
                    }
                }
                Err(_) => {
                    record_key_parse_failure_sample(
                        &mut key_failures.parse_failures,
                        &mut key_failures.samples,
                        &mut key_failures.by_context,
                        dst_context,
                        sample_from_access(access, dst_idx),
                    );
                    return Ok(());
                }
            },
            None => {
                if key_from_access_into_buf(access, dst_idx, dst_col, dst_key_buf).is_err() {
                    record_key_parse_failure_sample(
                        &mut key_failures.parse_failures,
                        &mut key_failures.samples,
                        &mut key_failures.by_context,
                        dst_context,
                        sample_from_access(access, dst_idx),
                    );
                    return Ok(());
                }
            }
        }
        let forward_shard = shard_for_key(src_key_buf, num_shards);
        forward_row.clear();
        forward_row.reserve(src_key_buf.len().saturating_add(dst_key_buf.len()));
        forward_row.extend_from_slice(src_key_buf);
        forward_row.extend_from_slice(dst_key_buf);
        batcher.push(forward_shard, forward_row)?;
        for batcher in sort_key_batchers.iter_mut() {
            batcher.push_row(forward_shard, access)?;
        }
        Ok(())
    }

    fn build_sharded_pk_maps(&mut self) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        let duplicate_counts = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let duplicate_samples = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
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
                let root = root.clone();
                tasks.push(WorkItem {
                    bucket: None,
                    run: Box::new(move || {
                        let key_type = key_type_for_column(&pk_col)?;
                        let meta_path = root.join(temp_pkey_meta_path(&table_name, shard));
                        let meta = read_temp_meta(&meta_path)?;
                        let total_rows = meta.row_count;
                        let mut duplicate_count = 0u64;
                        let mut duplicate_samples_local: Vec<Key> = Vec::new();
                        let mut row_id = 0u64;
                        let fingerprint_seed = fingerprint_seed_for(&table_name, shard);
                        let (hash_rel, data_rel) = match key_type {
                            KeyType::Int64 => {
                                let mut seen: std::collections::HashMap<u64, u32> =
                                    std::collections::HashMap::with_capacity(total_rows as usize);
                                let data_path = root.join(temp_pkey_path(&table_name, shard));
                                read_temp_rows(&data_path, |row| {
                                    let mut data = row;
                                    let key = read_key_from_buf(&mut data, &pk_col)?;
                                    if let Key::Int(v) = key {
                                        let local_id = u32::try_from(row_id)
                                            .map_err(|_| anyhow!("pk local id overflow"))?;
                                        match seen.entry(v) {
                                            Entry::Vacant(entry) => {
                                                entry.insert(local_id);
                                            }
                                            Entry::Occupied(_) => {
                                                duplicate_count =
                                                    duplicate_count.saturating_add(1);
                                                if duplicate_samples_local.len() < 10 {
                                                    duplicate_samples_local.push(Key::Int(v));
                                                }
                                            }
                                        }
                                    }
                                    row_id += 1;
                                    Ok(())
                                })?;
                                let mut keys = Vec::with_capacity(seen.len());
                                let mut values = Vec::with_capacity(seen.len());
                                for (key, value) in seen.into_iter() {
                                    keys.push(key);
                                    values.push(value);
                                }
                                let params = ptr_hash::PtrHashParams::default();
                                let mphf = DefaultPtrHash::<GxPtrHash64, u64>::try_new(
                                    &keys,
                                    params,
                                )
                                .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                let slots = mphf.max_index();
                                let mut entries = vec![(0u64, 0u32); slots];
                                for (key, value) in keys.iter().zip(values.iter()) {
                                    let idx = mphf.index_no_remap(key);
                                    let fp = PkGxHash::hash(key, fingerprint_seed);
                                    entries[idx] = (fp, *value);
                                }
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
                                    fingerprint_seed,
                                    &mphf,
                                    &entries,
                                )?;
                                (hash_rel, data_rel)
                            }
                            KeyType::VarBytes => {
                                let mut seen: std::collections::HashMap<Vec<u8>, u32> =
                                    std::collections::HashMap::with_capacity(total_rows as usize);
                                let data_path = root.join(temp_pkey_path(&table_name, shard));
                                read_temp_rows(&data_path, |row| {
                                    let mut data = row;
                                    let key = read_key_from_buf(&mut data, &pk_col)?;
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
                                        Entry::Occupied(entry) => {
                                            duplicate_count = duplicate_count.saturating_add(1);
                                            if duplicate_samples_local.len() < 10 {
                                                let sample = entry.key().clone();
                                                duplicate_samples_local.push(Key::Bytes(sample));
                                            }
                                        }
                                    }
                                    row_id += 1;
                                    Ok(())
                                })?;
                                let mut keys = Vec::with_capacity(seen.len());
                                let mut values = Vec::with_capacity(seen.len());
                                for (key, value) in seen.into_iter() {
                                    keys.push(key);
                                    values.push(value);
                                }
                                let params = ptr_hash::PtrHashParams::default();
                                let mphf = DefaultPtrHash::<GxPtrHash64, Vec<u8>>::try_new(
                                    &keys,
                                    params,
                                )
                                .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                let slots = mphf.max_index();
                                let mut entries = vec![(0u64, 0u32); slots];
                                for (key, value) in keys.iter().zip(values.iter()) {
                                    let idx = mphf.index_no_remap(key);
                                    let fp = PkGxHash::hash(key, fingerprint_seed);
                                    entries[idx] = (fp, *value);
                                }
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
                                    fingerprint_seed,
                                    &mphf,
                                    &entries,
                                )?;
                                (hash_rel, data_rel)
                            }
                            KeyType::Uuid => {
                                let mut seen: std::collections::HashMap<[u8; 16], u32> =
                                    std::collections::HashMap::with_capacity(total_rows as usize);
                                let data_path = root.join(temp_pkey_path(&table_name, shard));
                                read_temp_rows(&data_path, |row| {
                                    let mut data = row;
                                    let key = read_key_from_buf(&mut data, &pk_col)?;
                                    if let Key::Uuid(uuid) = key {
                                        let local_id = u32::try_from(row_id)
                                            .map_err(|_| anyhow!("pk local id overflow"))?;
                                        match seen.entry(uuid) {
                                            Entry::Vacant(entry) => {
                                                entry.insert(local_id);
                                            }
                                            Entry::Occupied(_) => {
                                                duplicate_count =
                                                    duplicate_count.saturating_add(1);
                                                if duplicate_samples_local.len() < 10 {
                                                    duplicate_samples_local.push(Key::Uuid(uuid));
                                                }
                                            }
                                        }
                                    }
                                    row_id += 1;
                                    Ok(())
                                })?;
                                let mut keys = Vec::with_capacity(seen.len());
                                let mut values = Vec::with_capacity(seen.len());
                                for (key, value) in seen.into_iter() {
                                    keys.push(key);
                                    values.push(value);
                                }
                                let params = ptr_hash::PtrHashParams::default();
                                let mphf = DefaultPtrHash::<GxPtrHash64, [u8; 16]>::try_new(
                                    &keys,
                                    params,
                                )
                                .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                let slots = mphf.max_index();
                                let mut entries = vec![(0u64, 0u32); slots];
                                for (key, value) in keys.iter().zip(values.iter()) {
                                    let idx = mphf.index_no_remap(key);
                                    let fp = PkGxHash::hash(key, fingerprint_seed);
                                    entries[idx] = (fp, *value);
                                }
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
                                    fingerprint_seed,
                                    &mphf,
                                    &entries,
                                )?;
                                (hash_rel, data_rel)
                            }
                            KeyType::Uuid256 => {
                                let mut seen: std::collections::HashMap<[u8; 32], u32> =
                                    std::collections::HashMap::with_capacity(total_rows as usize);
                                let data_path = root.join(temp_pkey_path(&table_name, shard));
                                read_temp_rows(&data_path, |row| {
                                    let mut data = row;
                                    let key = read_key_from_buf(&mut data, &pk_col)?;
                                    if let Key::Uuid256(uuid) = key {
                                        let local_id = u32::try_from(row_id)
                                            .map_err(|_| anyhow!("pk local id overflow"))?;
                                        match seen.entry(uuid) {
                                            Entry::Vacant(entry) => {
                                                entry.insert(local_id);
                                            }
                                            Entry::Occupied(_) => {
                                                duplicate_count =
                                                    duplicate_count.saturating_add(1);
                                                if duplicate_samples_local.len() < 10 {
                                                    duplicate_samples_local
                                                        .push(Key::Uuid256(uuid));
                                                }
                                            }
                                        }
                                    }
                                    row_id += 1;
                                    Ok(())
                                })?;
                                let mut keys = Vec::with_capacity(seen.len());
                                let mut values = Vec::with_capacity(seen.len());
                                for (key, value) in seen.into_iter() {
                                    keys.push(key);
                                    values.push(value);
                                }
                                let params = ptr_hash::PtrHashParams::default();
                                let mphf = DefaultPtrHash::<GxPtrHash64, [u8; 32]>::try_new(
                                    &keys,
                                    params,
                                )
                                .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                let slots = mphf.max_index();
                                let mut entries = vec![(0u64, 0u32); slots];
                                for (key, value) in keys.iter().zip(values.iter()) {
                                    let idx = mphf.index_no_remap(key);
                                    let fp = PkGxHash::hash(key, fingerprint_seed);
                                    entries[idx] = (fp, *value);
                                }
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
                                    fingerprint_seed,
                                    &mphf,
                                    &entries,
                                )?;
                                (hash_rel, data_rel)
                            }
                            KeyType::Ip => {
                                let mut seen: std::collections::HashMap<[u8; 16], u32> =
                                    std::collections::HashMap::with_capacity(total_rows as usize);
                                let data_path = root.join(temp_pkey_path(&table_name, shard));
                                read_temp_rows(&data_path, |row| {
                                    let mut data = row;
                                    let key = read_key_from_buf(&mut data, &pk_col)?;
                                    if let Key::Ip(ip) = key {
                                        let local_id = u32::try_from(row_id)
                                            .map_err(|_| anyhow!("pk local id overflow"))?;
                                        match seen.entry(ip) {
                                            Entry::Vacant(entry) => {
                                                entry.insert(local_id);
                                            }
                                            Entry::Occupied(_) => {
                                                duplicate_count =
                                                    duplicate_count.saturating_add(1);
                                                if duplicate_samples_local.len() < 10 {
                                                    duplicate_samples_local.push(Key::Ip(ip));
                                                }
                                            }
                                        }
                                    }
                                    row_id += 1;
                                    Ok(())
                                })?;
                                let mut keys = Vec::with_capacity(seen.len());
                                let mut values = Vec::with_capacity(seen.len());
                                for (key, value) in seen.into_iter() {
                                    keys.push(key);
                                    values.push(value);
                                }
                                let params = ptr_hash::PtrHashParams::default();
                                let mphf = DefaultPtrHash::<GxPtrHash64, [u8; 16]>::try_new(
                                    &keys,
                                    params,
                                )
                                .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                let slots = mphf.max_index();
                                let mut entries = vec![(0u64, 0u32); slots];
                                for (key, value) in keys.iter().zip(values.iter()) {
                                    let idx = mphf.index_no_remap(key);
                                    let fp = PkGxHash::hash(key, fingerprint_seed);
                                    entries[idx] = (fp, *value);
                                }
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
                                    fingerprint_seed,
                                    &mphf,
                                    &entries,
                                )?;
                                (hash_rel, data_rel)
                            }
                        };
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
                        target.write_bytes(&dup_meta, serde_json::to_vec_pretty(&meta)?)?;
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
                            });
                        }
                        Ok(())
                    }),
                });
            }
        }
        run_tasks(tasks.drain(..).collect(), &self.pool)?;
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
            });
        }
        if let Ok(counts) = std::sync::Arc::try_unwrap(duplicate_counts) {
            self.pk_duplicate_counts = counts.into_inner().unwrap();
        }
        if let Ok(samples) = std::sync::Arc::try_unwrap(duplicate_samples) {
            self.pk_duplicate_samples = samples.into_inner().unwrap();
        }
        Ok(())
    }

    fn write_pk_hash_and_data<H: EpSerialize>(
        target: &OutputTarget,
        hash_rel: &str,
        data_rel: &str,
        key_type: KeyType,
        fingerprint_seed: u64,
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
        let mut hash_writer = BufWriter::with_capacity(1024 * 1024, target.sink(&hash_path)?);
        mphf.serialize(&mut hash_writer)?;
        hash_writer.flush()?;
        let hash_sink = hash_writer
            .into_inner()
            .map_err(|_| anyhow!("pk hash flush failed"))?;
        hash_sink.finish()?;

        let mut data_writer = BufWriter::with_capacity(1024 * 1024, target.sink(&data_path)?);
        write_pk_data_header(
            &mut data_writer,
            key_type,
            entries.len() as u64,
            fingerprint_seed,
        )?;
        for (fingerprint, value) in entries {
            write_pk_data_entry(&mut data_writer, *fingerprint, *value)?;
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
        let pk_hash_store = Arc::new(self.build_pk_hash_store(&root)?);
        Ok(EdgeBuildContext {
            root,
            rel_sort_bits,
            rel_sort_keys,
            max_ids,
            pk_hash_store,
            edge_tables,
        })
    }
    fn build_pk_hash_store(&self, root: &Path) -> Result<PkHashStore> {
        let mut tables = std::collections::HashMap::new();
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
            let mut shards = Vec::with_capacity(paths.len());
            for shard_paths in paths {
                let hash_path = root.join(&shard_paths.hash_path);
                let data_path = root.join(&shard_paths.data_path);
                let shard = PkHashShard::load(&hash_path, &data_path, key_type)?;
                shards.push(shard);
            }
            tables.insert(table.name.clone(), shards);
        }
        Ok(PkHashStore::new(tables))
    }


    fn build_edges_mapped(&mut self, edge_ctx: &EdgeBuildContext) -> Result<()> {
        let root = &edge_ctx.root;
        self.edge_counts.clear();

        let pool = &self.pool;
        {
            #[derive(Default)]
            struct EdgeMapResult {
                relation: String,
                key_failures: KeyFailureState,
                edge_counts: HashMap<String, u64>,
            }
            let progress = self.progress.clone();
            let target = std::sync::Arc::clone(&self.target);
            let writer_pool = EdgeWriterPool::new(std::sync::Arc::clone(&target), edge_writer_workers());
            let writer_senders = writer_pool.senders();
            let shard_offsets = self.shard_offsets.clone();
            let pk_hash_store = edge_ctx.pk_hash_store.clone();
            let max_ids = edge_ctx.max_ids.clone();
            let rel_sort_bits = edge_ctx.rel_sort_bits.clone();
            let rel_sort_keys = edge_ctx.rel_sort_keys.clone();
            let edge_tables = edge_ctx.edge_tables.clone();
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
                    let rel_sort_keys = rel_sort_keys.clone();
                    let edge_tables = edge_tables.clone();
                    let root = root.clone();
                    let writer_senders = std::sync::Arc::clone(&writer_senders);
                    let tx = tx.clone();
                    let src_fk_col = src_fk_col.clone();
                    tasks.push(WorkItem {
                        bucket: None,
                        run: Box::new(move || {
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
                            let forward_lookup = build_sort_key_lookup(
                                &root,
                                &edge_tables,
                                &rel_sort_keys,
                                &rel.name_forward,
                                num_shards,
                            )?;
                            let reverse_lookup = build_sort_key_lookup(
                                &root,
                                &edge_tables,
                                &rel_sort_keys,
                                &rel.name_reverse,
                                num_shards,
                            )?;
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
                            );
                            let mut reverse_batcher = EdgeBatcher::new(
                                std::sync::Arc::<str>::from(rel.name_reverse.clone()),
                                reverse_spec,
                                num_shards,
                                buffer_limit,
                                std::sync::Arc::clone(&writer_senders),
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
                            let fkey_bitmap = read_fkey_bitmap(&bitmap_path, meta.row_count)?;
                            let mut key_buf = Vec::new();
                            read_temp_rows(&fkey_path, |row| {
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
                                    shard_for_key_from_key(&dst_key, &mut key_buf, num_shards)?;
                                let dst_local = match dst_maps
                                    .get(dst_shard)
                                    .and_then(|map| map.lookup(&dst_key))
                                {
                                    Some(id) => u64::from(id),
                                    None => {
                                        let sample = sample_key(&dst_key);
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
                            })?;
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
                                    });
                                }
                            }
                            result
                                .edge_counts
                                .insert(rel.name_forward.clone(), forward_count);
                            result
                                .edge_counts
                                .insert(rel.name_reverse.clone(), reverse_count);
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
                    let rel_sort_keys = rel_sort_keys.clone();
                    let edge_tables = edge_tables.clone();
                    let root = root.clone();
                    let tx = tx.clone();
                    let writer_senders = std::sync::Arc::clone(&writer_senders);
                    let edge = edge.clone();
                    let forward_name = forward_name.clone();
                    let reverse_name = reverse_name.clone();
                    let src_pk_col = src_pk_col.clone();
                    let dst_pk_col = dst_pk_col.clone();
                    tasks.push(WorkItem {
                        bucket: None,
                        run: Box::new(move || {
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
                            let forward_lookup = build_sort_key_lookup(
                                &root,
                                &edge_tables,
                                &rel_sort_keys,
                                &forward_name,
                                num_shards,
                            )?;
                            let reverse_lookup = build_sort_key_lookup(
                                &root,
                                &edge_tables,
                                &rel_sort_keys,
                                &reverse_name,
                                num_shards,
                            )?;
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
                            );
                            let mut reverse_batcher = EdgeBatcher::new(
                                std::sync::Arc::<str>::from(reverse_name.clone()),
                                reverse_spec,
                                num_shards,
                                buffer_limit,
                                std::sync::Arc::clone(&writer_senders),
                            );
                            let mut pending = 0u64;
                            let mut forward_count = 0u64;
                            let mut reverse_count = 0u64;
                            let forward_path = root.join(temp_edge_path(&forward_name, shard));
                            let mut row_idx = 0u64;
                            let mut key_buf = Vec::new();
                            read_temp_rows(&forward_path, |row| {
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
                                        });
                                    }
                                    pending = 0;
                                }
                                let mut data = row;
                                let src_key = read_key_from_buf(&mut data, &src_pk_col)
                                    .map_err(|_| anyhow!("src key parse"))?;
                                let dst_key = read_key_from_buf(&mut data, &dst_pk_col)
                                    .map_err(|_| anyhow!("dst key parse"))?;
                                let src_shard =
                                    shard_for_key_from_key(&src_key, &mut key_buf, num_shards)?;
                                let dst_shard =
                                    shard_for_key_from_key(&dst_key, &mut key_buf, num_shards)?;
                                let src_local = match pk_hash_store
                                    .shard(&edge.src_fk.ref_table, src_shard)?
                                    .lookup(&src_key)
                                {
                                    Some(id) => u64::from(id),
                                    None => {
                                        let sample = sample_key(&src_key);
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
                                        let sample = sample_key(&dst_key);
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
                            })?;
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
                                    });
                                }
                            }
                            result.edge_counts.insert(forward_name, forward_count);
                            result.edge_counts.insert(reverse_name, reverse_count);
                            tx.send(Ok(result))
                                .map_err(|e| anyhow!("edges_map output error: {e}"))?;
                            Ok(())
                        }),
                    });
                }
            }

            drop(tx);
            let task_count = tasks.len();
            run_tasks(tasks, pool)?;
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
                .saturating_add(self.record_edge_stage_bytes("fkeys")?);
            self.temp_edges_map_bytes = self
                .temp_edges_map_bytes
                .saturating_add(self.record_edge_stage_bytes("edges_map")?);
            self.cleanup_edge_stage("edges_src")?;
            self.cleanup_edge_stage("fkeys")?;
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
        fs::create_dir_all(&root)?;
        let metadata_path = root.join("index_metadata.json");
        fs::write(&metadata_path, serde_json::to_vec_pretty(builder.metadata())?)?;
        let jobs = builder.prepare_jobs()?;
        let adapter = self
            .progress
            .clone()
            .map(|cb| IndexProgressAdapter::new(cb, table_totals.clone()));
        let adapter = adapter.map(std::sync::Arc::new);
        let mut tasks = Vec::with_capacity(jobs.len());
        for job in jobs {
            let adapter = adapter.clone();
            let mem_budget = mem_budget.clone();
            tasks.push(WorkItem {
                bucket: None,
                run: Box::new(move || {
                    let progress = adapter
                        .as_ref()
                        .map(|value| value.as_ref() as &dyn GraphProgressSink);
                    job.run_with_budget(progress, mem_budget).map(|_| ())
                }),
            });
        }
        run_tasks(tasks, &self.pool)?;
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
            pk_duplicate_total,
            pk_duplicate_samples: self.pk_duplicate_samples.clone(),
            trimmed_strings: if trimmed_strings > 0 {
                Some(trimmed_strings)
            } else {
                None
            },
            perf,
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
    Ok(BufWriter::with_capacity(1024 * 1024, target.sink(&path)?))
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
    Ok(BufWriter::with_capacity(1024 * 1024, target.sink(&path)?))
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
    let mut file = FrameDecoder::new(file);
    let mut len_buf = [0u8; 4];
    let mut data = Vec::new();
    loop {
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
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

fn read_fkey_bitmap(path: &Path, row_count: u64) -> Result<Vec<u8>> {
    let expected = ((row_count + 7) / 8) as usize;
    let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if data.len() < expected {
        return Err(anyhow!(
            "fkey bitmap length mismatch: expected {expected}, got {}",
            data.len()
        ));
    }
    if data.len() == expected {
        return Ok(data);
    }
    Ok(data[..expected].to_vec())
}

fn load_sort_key_component_shards(
    root: &Path,
    rel_name: &str,
    kind: SortKeyComponentKind,
    num_shards: usize,
    width_bytes: usize,
) -> Result<Vec<SortKeyComponentStore>> {
    let mut shards = Vec::with_capacity(num_shards);
    for shard in 0..num_shards {
        let path = root.join(temp_sort_key_component_path(rel_name, kind, shard));
        let file = fs::File::open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        let data = unsafe { Mmap::map(&file)? };
        shards.push(SortKeyComponentStore { width_bytes, data });
    }
    Ok(shards)
}

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
        let shards = load_sort_key_component_shards(root, rel_name, ts.spec.kind, num_shards, width_bytes)?;
        let source = if edge_tables.contains(&ts.spec.table) {
            SortKeyComponentSource::EdgeTable { shards }
        } else {
            SortKeyComponentSource::DestTable { shards }
        };
        timestamp = Some(source);
    }
    if let Some(w) = &info.weight {
        let width_bytes = (w.spec.width_bits / 8) as usize;
        let shards = load_sort_key_component_shards(root, rel_name, w.spec.kind, num_shards, width_bytes)?;
        let source = if edge_tables.contains(&w.spec.table) {
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
        let mut writer = BufWriter::with_capacity(1024 * 1024, target.sink(path)?);
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
    fn new(target: std::sync::Arc<OutputTarget>, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) = std::sync::mpsc::sync_channel::<EdgeWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            handles.push(std::thread::spawn(move || edge_writer_worker(target, rx)));
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
                    continue;
                }
                let path = target.join(&temp_edge_map_part_path(&batch.relation, batch.shard));
                let mut writer = EdgePartWriter::open(&target, &path, batch.spec)?;
                writer.write_batch(&batch.data, batch.row_count)?;
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

struct EdgeBatcher {
    relation: std::sync::Arc<str>,
    relation_hash: u64,
    spec: EdgeBinarySpec,
    buffers: Vec<EdgeBuffer>,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeWriteMessage>>>,
}

impl EdgeBatcher {
    fn new(
        relation: std::sync::Arc<str>,
        spec: EdgeBinarySpec,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeWriteMessage>>>,
    ) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        relation.hash(&mut hasher);
        let relation_hash = hasher.finish();
        let buffers = (0..num_shards)
            .map(|_| EdgeBuffer {
                data: Vec::new(),
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
        let mut data = Vec::with_capacity(buffer.data.capacity());
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

struct NodeShardWriter {
    node: TempRowWriter,
    pkey: Option<TempRowWriter>,
    fkeys: Vec<TempRowWriter>,
    fkey_bits: Vec<BufWriter<OutputSink>>,
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
                            fkey_bits.push(open_fkey_bitmap_writer(
                                &target,
                                &temp_fkey_bitmap_path(&batch.table, rel, batch.shard),
                            )?);
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
                        writer_bits.write_all(&bits)?;
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
            let mut bits = bits;
            bits.flush()?;
            let sink = bits
                .into_inner()
                .map_err(|_| anyhow!("fkey bitmap flush failed"))?;
            sink.finish()?;
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

    fn push_pkey(&mut self, shard: usize, row: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("pkey buffer shard {} out of bounds", shard))?;
        if let Some(pkey) = buffer.pkey.as_mut() {
            append_temp_row(pkey, row)?;
            buffer.pkey_rows = buffer.pkey_rows.saturating_add(1);
            buffer.bytes = buffer.bytes.saturating_add(4 + row.len());
        }
        Ok(())
    }

    fn push_fkey(&mut self, shard: usize, rel_idx: usize, row: &[u8]) -> Result<()> {
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
                let added = append_fkey_bit(bits, row_idx, true);
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
}

impl EdgeSrcWritePool {
    fn new(target: std::sync::Arc<OutputTarget>, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) =
                std::sync::mpsc::sync_channel::<EdgeSrcWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            handles.push(std::thread::spawn(move || edge_src_writer_worker(target, rx)));
            senders.push(tx);
        }
        Self {
            senders: std::sync::Arc::new(senders),
            handles,
        }
    }

    fn senders(&self) -> std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>> {
        std::sync::Arc::clone(&self.senders)
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
}

impl EdgeSrcBatcher {
    fn new(
        rel: std::sync::Arc<str>,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>>,
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
        }
    }

    fn push(&mut self, shard: usize, row: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("edge src buffer shard {} out of bounds", shard))?;
        append_temp_row(&mut buffer.data, row)?;
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
            .map_err(|e| anyhow!("edge src writer channel error: {e}"))?;
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for shard in 0..self.buffers.len() {
            self.flush_shard(shard)?;
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

fn table_by_name<'a>(source: &'a SourceMetadata, name: &str) -> Result<&'a TableSpec> {
    source
        .tables
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("table {} not found", name))
}

fn sum_input_bytes(source: &SourceMetadata) -> u64 {
    let mut total = 0u64;
    for table in &source.tables {
        for path in &table.paths {
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

struct WorkItem {
    run: Box<dyn FnOnce() -> Result<()> + Send>,
    bucket: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
}

struct PoolTask {
    item: WorkItem,
    done: std::sync::mpsc::Sender<Result<()>>,
}

enum PoolMessage {
    Task(PoolTask),
    Stop,
}

struct TaskPool {
    sender: std::sync::mpsc::Sender<PoolMessage>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl TaskPool {
    fn new(threads: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<PoolMessage>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let rx = std::sync::Arc::clone(&rx);
            handles.push(std::thread::spawn(move || loop {
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
                        let result = (task.item.run)();
                        let elapsed = start.elapsed().as_nanos() as u64;
                        if let Some(bucket) = &task.item.bucket {
                            bucket.fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);
                        }
                        let _ = task.done.send(result);
                    }
                    PoolMessage::Stop => break,
                }
            }));
        }
        Self {
            sender: tx,
            handles,
        }
    }

    fn run_tasks(&self, tasks: Vec<WorkItem>) -> Result<()> {
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

impl Drop for TaskPool {
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

fn run_tasks(tasks: Vec<WorkItem>, pool: &TaskPool) -> Result<()> {
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
        | IngestedType::Decimal128
        | IngestedType::Decimal256 => Ok(KeyType::Int64),
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
            if value < 0 {
                return Err(KeyParseError::Negative);
            }
            out.extend_from_slice(&(value as u64).to_le_bytes());
            Ok(())
        }
        KeyType::VarBytes => {
            let mut write_len_prefixed = |bytes: &[u8]| -> KeyParseResult<()> {
                let start = out.len();
                out.extend_from_slice(&[0u8; 2]);
                let len = write_normalized_numeric_bytes(out, bytes, col)?;
                let len_u16 = u16::try_from(len).map_err(|_| KeyParseError::TooLarge)?;
                out[start..start + 2].copy_from_slice(&len_u16.to_le_bytes());
                Ok(())
            };
            match col.ingested_dtype {
                IngestedType::Bytes => {
                    let array = access
                        .array(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    let bytes = read_bytes(array, access.row())
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    write_len_prefixed(bytes)?;
                }
                IngestedType::String => {
                    let array = access
                        .array(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    let text = read_utf8(array, access.row())
                        .map_err(|_| KeyParseError::TypeMismatch)?;
                    write_len_prefixed(text.as_bytes())?;
                }
                IngestedType::Int64 => {
                    let value = access
                        .get_int64(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    if value < 0 {
                        return Err(KeyParseError::Negative);
                    }
                    let start = out.len();
                    out.extend_from_slice(&[0u8; 2]);
                    let len = write_display_to_vec(out, value);
                    let len_u16 = u16::try_from(len).map_err(|_| KeyParseError::TooLarge)?;
                    out[start..start + 2].copy_from_slice(&len_u16.to_le_bytes());
                }
                IngestedType::Int32 | IngestedType::Int16 | IngestedType::Int8 => {
                    let value = access
                        .get_int32(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    if value < 0 {
                        return Err(KeyParseError::Negative);
                    }
                    let start = out.len();
                    out.extend_from_slice(&[0u8; 2]);
                    let len = write_display_to_vec(out, value);
                    let len_u16 = u16::try_from(len).map_err(|_| KeyParseError::TooLarge)?;
                    out[start..start + 2].copy_from_slice(&len_u16.to_le_bytes());
                }
                IngestedType::UInt64 => {
                    let value = access
                        .get_uint64(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    let start = out.len();
                    out.extend_from_slice(&[0u8; 2]);
                    let len = write_display_to_vec(out, value);
                    let len_u16 = u16::try_from(len).map_err(|_| KeyParseError::TooLarge)?;
                    out[start..start + 2].copy_from_slice(&len_u16.to_le_bytes());
                }
                IngestedType::UInt32 | IngestedType::UInt16 | IngestedType::UInt8 => {
                    let value = access
                        .get_uint32(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    let start = out.len();
                    out.extend_from_slice(&[0u8; 2]);
                    let len = write_display_to_vec(out, value);
                    let len_u16 = u16::try_from(len).map_err(|_| KeyParseError::TooLarge)?;
                    out[start..start + 2].copy_from_slice(&len_u16.to_le_bytes());
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
            let len = u16::try_from(v.len())
                .map_err(|_| anyhow!("temp key too large"))?;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(v);
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

#[cfg(test)]
fn parse_uuid_bytes(bytes: &[u8]) -> KeyParseResult<[u8; 16]> {
    if bytes.len() != 16 {
        return Err(KeyParseError::InvalidUuid);
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(bytes);
    Ok(out)
}

fn parse_uuid_string(bytes: &[u8]) -> KeyParseResult<[u8; 16]> {
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

fn parse_uuid256_string(bytes: &[u8]) -> KeyParseResult<[u8; 32]> {
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

fn parse_ip_string(bytes: &[u8]) -> KeyParseResult<[u8; 16]> {
    let s = std::str::from_utf8(bytes).map_err(|_| KeyParseError::NotUtf8)?;
    let ip: std::net::IpAddr = s.parse().map_err(|_| KeyParseError::InvalidIp)?;
    Ok(ip_to_ipv6_bytes(ip))
}

fn parse_ip_bytes(bytes: &[u8]) -> KeyParseResult<[u8; 16]> {
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_metadata::TimestampUnit;
    use chrono::NaiveDate;
    use arrow_array::{BinaryArray, Int64Array, StringArray};

    #[test]
    fn parse_uuid_string_formats() {
        let raw = "00112233445566778899aabbccddeeff";
        let dashed = "00112233-4455-6677-8899-aabbccddeeff";
        let expected = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff,
        ];
        assert_eq!(parse_uuid_string(raw.as_bytes()).unwrap(), expected);
        assert_eq!(parse_uuid_string(dashed.as_bytes()).unwrap(), expected);
    }

    #[test]
    fn parse_uuid256_string_formats() {
        let raw = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
        let expected: [u8; 32] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];
        assert_eq!(parse_uuid256_string(raw.as_bytes()).unwrap(), expected);
    }

    #[test]
    fn parse_ip_string_formats() {
        let v4 = "192.168.1.2";
        let v6 = "2001:db8::1";
        let v4_expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 1, 2,
        ];
        let v6_expected = [
            0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        assert_eq!(parse_ip_string(v4.as_bytes()).unwrap(), v4_expected);
        assert_eq!(parse_ip_string(v6.as_bytes()).unwrap(), v6_expected);
    }

    #[test]
    fn parse_uuid_bytes_requires_len_16() {
        assert!(parse_uuid_bytes(&[0u8; 15]).is_err());
        assert!(parse_uuid_bytes(&[0u8; 17]).is_err());
        assert!(parse_uuid_bytes(&[0u8; 16]).is_ok());
    }

    fn single_col_table(ingested: IngestedType) -> TableSpec {
        TableSpec {
            name: "sample".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                ingested_dtype: ingested,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        }
    }

    fn with_access(
        table: &TableSpec,
        cols: Vec<ArrayRef>,
        f: impl FnOnce(&mut RowAccess<'_>),
    ) {
        let trimmed = std::sync::Arc::new(AtomicU64::new(0));
        let hints = build_row_access_hints(table);
        let stats = RowAccessStats::new(Some(trimmed.as_ref()));
        let mut access = RowAccess::new(hints, stats);
        access.set_cols(&cols);
        access.set_row(0);
        f(&mut access);
    }

    fn days_from_secs(secs: i64) -> i32 {
        secs.div_euclid(86_400) as i32
    }

    #[test]
    fn access_uuid_from_uuid_string_source() {
        let table = single_col_table(IngestedType::Uuid);
        let text = "00112233445566778899aabbccddeeff";
        let cols: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text]))];
        let expected = parse_uuid_string(text.as_bytes()).unwrap();
        with_access(&table, cols, |access| {
            let uuid = access.get_uuid(0).unwrap().unwrap();
            assert_eq!(uuid, expected);
        });
    }

    #[test]
    fn access_uuid_from_string_to_uuid() {
        let table = single_col_table(IngestedType::Uuid);
        let text = "00112233445566778899aabbccddeeff";
        let cols: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text]))];
        let expected = parse_uuid_string(text.as_bytes()).unwrap();
        with_access(&table, cols, |access| {
            let uuid = access.get_uuid(0).unwrap().unwrap();
            assert_eq!(uuid, expected);
        });
    }

    #[test]
    fn access_uuid256_from_string_to_uuid256() {
        let table = single_col_table(IngestedType::Uuid256);
        let text = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
        let cols: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text]))];
        let expected = parse_uuid256_string(text.as_bytes()).unwrap();
        with_access(&table, cols, |access| {
            let uuid = access.get_uuid256(0).unwrap().unwrap();
            assert_eq!(uuid, expected);
        });
    }

    #[test]
    fn access_ip_from_ip_string_source() {
        let table = single_col_table(IngestedType::Ip);
        let text = "192.168.1.2";
        let cols: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text]))];
        let expected = parse_ip_string(text.as_bytes()).unwrap();
        with_access(&table, cols, |access| {
            let ip = access.get_ip(0).unwrap().unwrap();
            assert_eq!(ip, expected);
        });
    }

    #[test]
    fn convert_string_timestamp_to_rowpack_timestamp() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_at".to_string(),
                ingested_dtype: IngestedType::Timestamp,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        };
        let arr = StringArray::from(vec!["2020-01-02T03:04:05Z"]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        with_access(&table, cols, |access| {
            let (secs, nanos) = access.get_timestamp(0).unwrap().unwrap();
            assert_eq!(nanos, 0);
            assert_eq!(secs, 1_577_934_245);
        });
    }

    #[test]
    fn convert_int_timestamp_with_unit() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_at".to_string(),
                ingested_dtype: IngestedType::Timestamp,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: Some(TimestampUnit::Millis),
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        };
        let arr = Int64Array::from(vec![1_500_000_i64]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        with_access(&table, cols, |access| {
            let (secs, nanos) = access.get_timestamp(0).unwrap().unwrap();
            assert_eq!(secs, 1500);
            assert_eq!(nanos, 0);
        });
    }

    #[test]
    fn encode_timestamp_int8() {
        let encoding = TimestampEncodingSpec {
            kind: TimestampEncodingKind::Int8,
            anchor: 0,
            step: 1,
        };
        let encoded = encode_timestamp_value(Some(10), &encoding);
        assert_eq!(encoded, ordered_signed_to_u64(10, 8));
        let encoded_neg = encode_timestamp_value(Some(-1), &encoding);
        assert_eq!(encoded_neg, ordered_signed_to_u64(-1, 8));
        let encoded_clamp = encode_timestamp_value(Some(1000), &encoding);
        assert_eq!(encoded_clamp, ordered_signed_to_u64(i8::MAX as i64, 8));
    }

    #[test]
    fn convert_string_date_to_rowpack_date() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_date".to_string(),
                ingested_dtype: IngestedType::Date,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: Some("%Y-%m-%d".to_string()),
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        };
        let arr = StringArray::from(vec!["2024-02-03"]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        with_access(&table, cols, |access| {
            let days = access.get_date(0).unwrap().unwrap();
            let expected = days_from_secs(
                NaiveDate::from_ymd_opt(2024, 2, 3)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
                    .timestamp(),
            );
            assert_eq!(days, expected);
        });
    }

    #[test]
    fn convert_numeric_to_string_for_rowpack() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        };
        let arr = Int64Array::from(vec![42_i64]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        with_access(&table, cols, |access| {
            let value = access
                .with_string(0, |v| Ok(v.to_string()))
                .unwrap()
                .unwrap();
            assert_eq!(value, "42");
        });
    }

    #[test]
    fn convert_bytes_to_int_for_rowpack() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                ingested_dtype: IngestedType::Int32,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        };
        let arr = BinaryArray::from(vec![b"123".as_slice()]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        with_access(&table, cols, |access| {
            let value = access.get_int32(0).unwrap().unwrap();
            assert_eq!(value, 123);
        });
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

fn build_row_access_hints(table: &TableSpec) -> Vec<RowAccessHints> {
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
    unit: Option<&crate::source_metadata::TimestampUnit>,
) -> Option<arrow_schema::TimeUnit> {
    match unit? {
        crate::source_metadata::TimestampUnit::Seconds => Some(arrow_schema::TimeUnit::Second),
        crate::source_metadata::TimestampUnit::Millis => Some(arrow_schema::TimeUnit::Millisecond),
        crate::source_metadata::TimestampUnit::Micros => Some(arrow_schema::TimeUnit::Microsecond),
        crate::source_metadata::TimestampUnit::Nanos => Some(arrow_schema::TimeUnit::Nanosecond),
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

fn fast_key_from_array(
    cols: &[ArrayRef],
    row: usize,
    col_idx: usize,
    kind: EdgeKeyFastKind,
) -> KeyParseResult<Key> {
    let array = cols.get(col_idx).ok_or(KeyParseError::TypeMismatch)?;
    if array.is_null(row) {
        return Err(KeyParseError::TypeMismatch);
    }
    match kind {
        EdgeKeyFastKind::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let v = arr.value(row);
            if v < 0 {
                Err(KeyParseError::Negative)
            } else {
                Ok(Key::Int(v as u64))
            }
        }
        EdgeKeyFastKind::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let v = arr.value(row);
            if v < 0 {
                Err(KeyParseError::Negative)
            } else {
                Ok(Key::Int(v as u64))
            }
        }
        EdgeKeyFastKind::Int16 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int16Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let v = arr.value(row);
            if v < 0 {
                Err(KeyParseError::Negative)
            } else {
                Ok(Key::Int(v as u64))
            }
        }
        EdgeKeyFastKind::Int8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int8Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let v = arr.value(row);
            if v < 0 {
                Err(KeyParseError::Negative)
            } else {
                Ok(Key::Int(v as u64))
            }
        }
        EdgeKeyFastKind::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            Ok(Key::Int(arr.value(row)))
        }
        EdgeKeyFastKind::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            Ok(Key::Int(arr.value(row) as u64))
        }
        EdgeKeyFastKind::UInt16 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt16Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            Ok(Key::Int(arr.value(row) as u64))
        }
        EdgeKeyFastKind::UInt8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt8Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            Ok(Key::Int(arr.value(row) as u64))
        }
        EdgeKeyFastKind::StringUuid => {
            let text = fast_string_value(array, row)?;
            let uuid = parse_uuid_string(text.as_bytes())?;
            Ok(Key::Uuid(uuid))
        }
        EdgeKeyFastKind::BytesUuid => {
            let data = fast_bytes_value(array, row)?;
            if data.len() != 16 {
                return Err(KeyParseError::LengthMismatch);
            }
            let mut buf = [0u8; 16];
            buf.copy_from_slice(data);
            Ok(Key::Uuid(buf))
        }
        EdgeKeyFastKind::StringVar => {
            let data = fast_string_value(array, row)?.as_bytes();
            if data.len() > MAX_VAR_KEY_LEN {
                return Err(KeyParseError::TooLarge);
            }
            Ok(Key::Bytes(data.to_vec()))
        }
        EdgeKeyFastKind::BytesVar => {
            let data = fast_bytes_value(array, row)?;
            if data.len() > MAX_VAR_KEY_LEN {
                return Err(KeyParseError::TooLarge);
            }
            Ok(Key::Bytes(data.to_vec()))
        }
        EdgeKeyFastKind::StringUuid256 => {
            let text = fast_string_value(array, row)?;
            let uuid = parse_uuid256_string(text.as_bytes())?;
            Ok(Key::Uuid256(uuid))
        }
        EdgeKeyFastKind::BytesUuid256 => {
            let data = fast_bytes_value(array, row)?;
            if data.len() != 32 {
                return Err(KeyParseError::LengthMismatch);
            }
            let mut buf = [0u8; 32];
            buf.copy_from_slice(data);
            Ok(Key::Uuid256(buf))
        }
        EdgeKeyFastKind::StringIp => {
            let text = fast_string_value(array, row)?;
            let ip = parse_ip_string(text.as_bytes())?;
            Ok(Key::Ip(ip))
        }
        EdgeKeyFastKind::BytesIp => {
            let data = fast_bytes_value(array, row)?;
            let ip = parse_ip_bytes(data)?;
            Ok(Key::Ip(ip))
        }
    }
}


fn read_utf8<'a>(array: &'a ArrayRef, row: usize) -> Result<&'a str> {
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


struct LocalizedInput {
    path: PathBuf,
    _guard: Option<tempfile::TempPath>,
}

fn localize_input(path: &str) -> Result<LocalizedInput> {
    if path.starts_with("s3://") || path.starts_with("gs://") || path.starts_with("az://") {
        let scratch = scratch_root()?;
        fs::create_dir_all(&scratch)?;
        let tmp = tempfile::Builder::new()
            .prefix("ingest-source-")
            .tempfile_in(&scratch)?;
        let (store, object_path) = build_store_for_path(path)?;
        let rt = Runtime::new()?;
        let obj = rt.block_on(store.get(&object_path))?;
        let bytes = rt.block_on(obj.bytes())?;
        fs::write(tmp.path(), &bytes)?;
        Ok(LocalizedInput {
            path: tmp.path().to_path_buf(),
            _guard: Some(tmp.into_temp_path()),
        })
    } else {
        Ok(LocalizedInput {
            path: PathBuf::from(path),
            _guard: None,
        })
    }
}

fn scratch_root() -> Result<PathBuf> {
    let ssd = PathBuf::from("/mnt/localssd/diskgraph-ingest-tmp");
    if ssd.exists() {
        return Ok(ssd);
    }
    if ssd.parent().map(|p| p.exists()).unwrap_or(false) {
        fs::create_dir_all(&ssd)?;
        return Ok(ssd);
    }
    Ok(PathBuf::from("/tmp/diskgraph-ingest-tmp"))
}

fn build_store_for_path(path: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath)> {
    if path.starts_with("s3://") {
        let (bucket, key) = parse_s3_url(path)?;
        let s3 = build_s3_client(S3Config {
            bucket: bucket.clone(),
            region: std::env::var("S3_REGION").ok(),
            endpoint: std::env::var("S3_ENDPOINT").ok(),
            allow_http: None,
            virtual_hosted: None,
            profile: std::env::var("S3_PROFILE").ok(),
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
        })?;
        Ok((s3.store, ObjectPath::from(key)))
    } else if path.starts_with("gs://") {
        let (bucket, key) = parse_gs_url(path)?;
        let gcs = build_gcs_client(GcsConfig { bucket: bucket.clone() })?;
        Ok((gcs.store, ObjectPath::from(key)))
    } else if path.starts_with("az://") {
        let (account, container, key) = parse_az_url(path)?;
        let adls = build_adls_client(AdlsConfig {
            account: account.clone(),
            container: container.clone(),
            endpoint: std::env::var("AZURE_ENDPOINT").ok(),
            allow_http: None,
            access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
            sas_key: std::env::var("AZURE_SAS_KEY").ok(),
            bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
            client_id: std::env::var("AZURE_CLIENT_ID").ok(),
            client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
            tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
        })?;
        Ok((adls.store, ObjectPath::from(key)))
    } else {
        Err(anyhow!("unsupported source path {}", path))
    }
}

fn parse_s3_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw).map_err(|e| anyhow!("invalid s3 url {raw}: {e}"))?;
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow!("missing bucket in s3 url"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    Ok((bucket, key))
}

fn parse_gs_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw).map_err(|e| anyhow!("invalid gs url {raw}: {e}"))?;
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow!("missing bucket in gs url"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    Ok((bucket, key))
}

fn parse_az_url(raw: &str) -> Result<(String, String, String)> {
    let url = url::Url::parse(raw).map_err(|e| anyhow!("invalid az url {raw}: {e}"))?;
    let account = url
        .host_str()
        .ok_or_else(|| anyhow!("missing account in az url"))?
        .to_string();
    let mut segments = url
        .path()
        .trim_start_matches('/')
        .split('/')
        .filter(|s| !s.is_empty());
    let container = segments
        .next()
        .ok_or_else(|| anyhow!("missing container in az url"))?
        .to_string();
    let key = segments.collect::<Vec<_>>().join("/");
    Ok((account, container, key))
}

fn resolve_local_paths(source: &mut SourceMetadata, base_dir: &Path) {
    for table in &mut source.tables {
        for path in &mut table.paths {
            if is_cloud_path(path) {
                continue;
            }
            let path_buf = PathBuf::from(path.as_str());
            if path_buf.is_absolute() {
                continue;
            }
            let joined = base_dir.join(path_buf);
            *path = joined.to_string_lossy().to_string();
        }
    }
}

fn is_cloud_path(path: &str) -> bool {
    path.starts_with("s3://") || path.starts_with("gs://") || path.starts_with("az://")
}

fn require_local_output(target: &OutputTarget) -> Result<&Path> {
    match target {
        OutputTarget::Local { root } => Ok(root.as_path()),
        _ => Err(anyhow!("sharded ingest requires a local output path")),
    }
}

enum OutputTarget {
    Local { root: PathBuf },
    S3 {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
    Gcs {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
    Adls {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
}

impl OutputTarget {
    fn parse(output: &str, dest: Dest) -> Result<Self> {
        match dest {
            Dest::Local => Ok(OutputTarget::Local {
                root: PathBuf::from(output),
            }),
            Dest::S3 => {
                let (bucket, prefix) = parse_s3_url(output)?;
                let s3 = build_s3_client(S3Config {
                    bucket: bucket.clone(),
                    region: None,
                    endpoint: None,
                    allow_http: None,
                    virtual_hosted: None,
                    profile: std::env::var("S3_PROFILE").ok(),
                    access_key_id: None,
                    secret_access_key: None,
                    session_token: None,
                })?;
                Ok(OutputTarget::S3 {
                    prefix,
                    store: s3.store,
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
            Dest::Gcs => {
                let (bucket, prefix) = parse_gs_url(output)?;
                let gcs = build_gcs_client(GcsConfig { bucket })?;
                Ok(OutputTarget::Gcs {
                    prefix,
                    store: gcs.store,
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
            Dest::Adls => {
                let (account, container, prefix) = parse_az_url(output)?;
                let adls = build_adls_client(AdlsConfig {
                    account,
                    container,
                    endpoint: std::env::var("AZURE_ENDPOINT").ok(),
                    allow_http: None,
                    access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
                    sas_key: std::env::var("AZURE_SAS_KEY").ok(),
                    bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
                    client_id: std::env::var("AZURE_CLIENT_ID").ok(),
                    client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
                    tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
                })?;
                Ok(OutputTarget::Adls {
                    prefix,
                    store: adls.store,
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
        }
    }

    fn prepare(&self) -> Result<()> {
        if let OutputTarget::Local { root } = self {
            fs::create_dir_all(root)?;
        }
        Ok(())
    }

    fn join(&self, name: &str) -> PathBuf {
        match self {
            OutputTarget::Local { root } => root.join(name),
            OutputTarget::S3 { prefix, .. }
            | OutputTarget::Gcs { prefix, .. }
            | OutputTarget::Adls { prefix, .. } => {
                if prefix.is_empty() {
                    PathBuf::from(name)
                } else {
                    PathBuf::from(format!("{}/{}", prefix.trim_end_matches('/'), name))
                }
            }
        }
    }

    fn write_bytes(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
        match self {
            OutputTarget::Local { root } => {
                let path = root.join(name);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&path, bytes)
                    .with_context(|| format!("write {}", path.display()))?;
            }
            OutputTarget::S3 { store, rt, prefix, .. }
            | OutputTarget::Gcs { store, rt, prefix, .. }
            | OutputTarget::Adls { store, rt, prefix, .. } => {
                let obj = if prefix.is_empty() {
                    ObjectPath::from(name)
                } else {
                    ObjectPath::from(format!("{}/{}", prefix.trim_end_matches('/'), name))
                };
                rt.block_on(store.put(&obj, Bytes::from(bytes).into()))
                    .map_err(anyhow::Error::from)?;
            }
        }
        Ok(())
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

fn to_io_err(err: object_store::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err)
}
