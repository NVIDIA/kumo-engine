//! Diskgraph ingest: convert raw CSV/Parquet sources into index-ready artifacts.

mod mapping;
pub mod source_metadata;

use crate::mapping::{Key, KeyType, PkMap};
use crate::source_metadata::{
    ColumnSpec, ColumnType, KeyMode, TableFormat, TableSpec, TimestampUnit,
};
use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use chrono::{NaiveDate, NaiveDateTime};
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, EdgeStorageFormat, NodeDataCompression,
    NodeShardMetadata, NodeTypeMetadata, PrimaryKeyMap,
};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig, S3Config,
};
use diskgraph_rowpack::{
    LogicalType as RowpackType, RowPackConfig, RowPackContextBuilder, RowPackDecodeScratch,
    RowPackDecoder, RowPackEncoder, RowPackScratch, ValueRef,
};
use diskgraph_index::graph::{
    GraphBuildProgress, GraphBuildStage, GraphDataSource, GraphIndexBuilder, GraphProgressSink,
};
use num_traits::Signed;
use object_store::{path::Path as ObjectPath, ObjectStore, WriteMultipart};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pprof::protos::Message;
use pprof::ProfilerGuard;
use serde::Serialize;
use serde_json::Value as JsonValue;
use ahash::AHashMap as HashMap;
use std::fs;
use std::io::{self, BufWriter, Read, Write};
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::hash_map::Entry;
use std::time::Duration;
use twox_hash::XxHash64;
use tokio::runtime::Runtime;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

pub use mapping::MappingKind;
pub use source_metadata::SourceMetadata;

const NODE_BATCH_SIZE: usize = 100_000;
const MAX_KEY_SAMPLE_LEN: usize = 64;
const MAX_INPUT_STRING_BYTES: usize = 32 * 1024;
const TEMP_DIR_NAME: &str = "_tmp";
const TEMP_META_MAGIC: [u8; 4] = *b"DGTP";
const TEMP_META_VERSION: u8 = 1;
const SAMPLE_ROW_LIMIT: usize = 100_000;
const SAMPLE_ROW_TARGET: usize = 10_000;
const PK_MAP_MEMORY_BUDGET: usize = 1024 * 1024 * 1024;
const MAX_NODE_DATA_LEN: usize = 60 * 1024;
const PROGRESS_ROW_BATCH: u64 = 100_000;
const MAX_VAR_KEY_LEN: usize = u16::MAX as usize;

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
}

type KeyParseResult<T> = std::result::Result<T, KeyParseError>;

#[derive(Clone, Debug, Serialize)]
pub struct IngestReport {
    pub stats: IngestStats,
    pub source: SourceMetadata,
}

#[derive(Clone)]
pub struct IngestOptions {
    pub profile_dir: Option<PathBuf>,
    pub progress: Option<ProgressCallback>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            profile_dir: None,
            progress: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProgressPhase {
    Context,
    Shuffle,
    PkMap,
    Edges,
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
    last: Mutex<HashMap<(String, Option<usize>), u64>>,
}

impl IndexProgressAdapter {
    fn new(cb: ProgressCallback) -> Self {
        Self {
            cb,
            last: Mutex::new(HashMap::new()),
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
        let mut last = self.last.lock().unwrap();
        match update.stage {
            GraphBuildStage::BuildShard { .. } => {
                match last.entry(key.clone()) {
                    Entry::Vacant(v) => {
                        (self.cb)(ProgressEvent {
                            phase: ProgressPhase::Index,
                            table: table.clone(),
                            shard,
                            kind: ProgressKind::Start,
                            delta: 0,
                            total: update.total,
                        });
                        v.insert(update.current);
                        if update.current > 0 {
                            (self.cb)(ProgressEvent {
                                phase: ProgressPhase::Index,
                                table,
                                shard,
                                kind: ProgressKind::Advance,
                                delta: update.current,
                                total: update.total,
                            });
                        }
                    }
                    Entry::Occupied(mut o) => {
                        let prev = *o.get();
                        if update.current > prev {
                            let delta = update.current.saturating_sub(prev);
                            o.insert(update.current);
                            (self.cb)(ProgressEvent {
                                phase: ProgressPhase::Index,
                                table,
                                shard,
                                kind: ProgressKind::Advance,
                                delta,
                                total: update.total,
                            });
                        }
                    }
                }
            }
            GraphBuildStage::Complete { .. } => {
                let prev = last.get(&key).copied().unwrap_or(0);
                (self.cb)(ProgressEvent {
                    phase: ProgressPhase::Index,
                    table,
                    shard,
                    kind: ProgressKind::Finish,
                    delta: 0,
                    total: update.total.or(Some(prev)),
                });
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

#[derive(Clone, Debug, Default, Serialize)]
pub struct PerfStats {
    pub wall_time_ms: u64,
    pub context_wall_ms: u64,
    pub shuffle_wall_ms: u64,
    pub pk_map_wall_ms: u64,
    pub build_wall_ms: u64,
    pub index_wall_ms: u64,
    pub finalize_wall_ms: u64,
    pub context_task_ms: u64,
    pub shuffle_task_ms: u64,
    pub pk_map_task_ms: u64,
    pub build_task_ms: u64,
    pub edge_task_ms: u64,
    pub index_task_ms: u64,
    pub context_tasks: u64,
    pub shuffle_tasks: u64,
    pub pk_map_tasks: u64,
    pub build_tasks: u64,
    pub edge_tasks: u64,
    pub index_tasks: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
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
    let shuffle_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "shuffle", || ctx.build_temp_shards())?;
    } else {
        ctx.build_temp_shards()?;
    }
    let shuffle_wall = shuffle_start.elapsed();
    let pk_map_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "pk_map", || ctx.build_sharded_pk_maps())?;
    } else {
        ctx.build_sharded_pk_maps()?;
    }
    let pk_map_wall = pk_map_start.elapsed();
    let final_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "edges", || ctx.build_final_edges())?;
    } else {
        ctx.build_final_edges()?;
    }
    let build_wall = final_start.elapsed();
    let index_start = std::time::Instant::now();
    let index_metadata = ctx.build_index_metadata()?;
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "index", || ctx.build_index(index_metadata.clone()))?;
    } else {
        ctx.build_index(index_metadata.clone())?;
    }
    ctx.cleanup_temp_inputs()?;
    let index_wall = index_start.elapsed();
    let finalize_start = std::time::Instant::now();
    let index_metadata = ctx.finalize_metadata(index_metadata)?;
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
        shuffle_wall_ms: shuffle_wall.as_millis() as u64,
        pk_map_wall_ms: pk_map_wall.as_millis() as u64,
        build_wall_ms: build_wall.as_millis() as u64,
        index_wall_ms: index_wall.as_millis() as u64,
        finalize_wall_ms: finalize_wall.as_millis() as u64,
        context_task_ms: ctx.context_task_ms,
        shuffle_task_ms: ctx.shuffle_task_ms,
        pk_map_task_ms: ctx.pk_map_task_ms,
        build_task_ms: ctx.build_task_ms,
        edge_task_ms: ctx.edge_task_ms,
        index_task_ms: ctx.index_task_ms,
        context_tasks: ctx.context_tasks,
        shuffle_tasks: ctx.shuffle_tasks,
        pk_map_tasks: ctx.pk_map_tasks,
        build_tasks: ctx.build_tasks,
        edge_tasks: ctx.edge_tasks,
        index_tasks: ctx.index_tasks,
        bytes_read,
        bytes_written,
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
    num_shards: usize,
    table_rows: HashMap<String, u64>,
    table_contexts: HashMap<String, String>,
    shard_rows: HashMap<String, Vec<u64>>,
    shard_offsets: HashMap<String, Vec<u64>>,
    pk_map_paths: HashMap<String, Vec<String>>,
    relation_defs: Vec<RelationDef>,
    relationships: Vec<RelationshipRuntime>,
    edge_counts: HashMap<String, u64>,
    pk_duplicate_counts: HashMap<String, u64>,
    pk_duplicate_samples: HashMap<String, Vec<String>>,
    output: String,
    trimmed_strings: std::sync::Arc<AtomicU64>,
    progress: Option<ProgressCallback>,
    shuffle_task_ms: u64,
    pk_map_task_ms: u64,
    build_task_ms: u64,
    edge_task_ms: u64,
    index_task_ms: u64,
    context_task_ms: u64,
    shuffle_tasks: u64,
    pk_map_tasks: u64,
    build_tasks: u64,
    edge_tasks: u64,
    index_tasks: u64,
    context_tasks: u64,
}

struct TempRowpackMeta {
    row_count: u64,
}

struct TempRowWriter {
    writer: BufWriter<OutputSink>,
    row_count: u64,
}

#[derive(Clone)]
enum OwnedValue {
    Null,
    Int64(i64),
    Int32(i32),
    UInt64(u64),
    UInt32(u32),
    Float64(f64),
    Float32(f32),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    Date(i32),
    Timestamp { secs: i64, nanos: u32 },
    Decimal128(i128),
    Decimal256([u8; 32]),
    Float32Array(Vec<f32>),
    Float64Array(Vec<f64>),
}

#[derive(Clone)]
struct OwnedRow {
    values: Vec<OwnedValue>,
}

impl OwnedValue {
    fn as_value_ref(&self) -> ValueRef<'_> {
        match self {
            OwnedValue::Null => ValueRef::Null,
            OwnedValue::Int64(v) => ValueRef::Int64(*v),
            OwnedValue::Int32(v) => ValueRef::Int32(*v),
            OwnedValue::UInt64(v) => ValueRef::UInt64(*v),
            OwnedValue::UInt32(v) => ValueRef::UInt32(*v),
            OwnedValue::Float64(v) => ValueRef::Float64(*v),
            OwnedValue::Float32(v) => ValueRef::Float32(*v),
            OwnedValue::Bool(v) => ValueRef::Bool(*v),
            OwnedValue::String(v) => ValueRef::String(v.as_str()),
            OwnedValue::Bytes(v) => ValueRef::Bytes(v.as_slice()),
            OwnedValue::Date(v) => ValueRef::Date(*v),
            OwnedValue::Timestamp { secs, nanos } => ValueRef::Timestamp {
                secs: *secs,
                nanos: *nanos,
            },
            OwnedValue::Decimal128(v) => ValueRef::Decimal128(*v),
            OwnedValue::Decimal256(v) => ValueRef::Decimal256(*v),
            OwnedValue::Float32Array(v) => ValueRef::Float32Array(v.as_slice()),
            OwnedValue::Float64Array(v) => ValueRef::Float64Array(v.as_slice()),
        }
    }
}

impl OwnedRow {
    fn as_value_refs<'a>(&'a self, out: &mut Vec<ValueRef<'a>>) {
        out.clear();
        out.reserve(self.values.len());
        for value in &self.values {
            out.push(value.as_value_ref());
        }
    }

    fn as_value_refs_for_indices<'a>(&'a self, indices: &[usize], out: &mut Vec<ValueRef<'a>>) {
        out.clear();
        out.reserve(indices.len());
        for idx in indices {
            if let Some(value) = self.values.get(*idx) {
                out.push(value.as_value_ref());
            }
        }
    }
}


#[derive(Clone)]
struct RelationDef {
    name: String,
    src_table: String,
    dst_table: String,
    sort_key: SortKeyMode,
}

#[derive(Clone)]
struct RelationshipRuntime {
    name_forward: String,
    name_reverse: String,
    src_table: String,
    src_column: String,
    dst_table: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortKeyMode {
    None,
    Weight,
    CreateTime,
}

impl IngestContext {
    fn emit_progress(&self, event: ProgressEvent) {
        if let Some(cb) = &self.progress {
            cb(event);
        }
    }
    fn new(
        source: SourceMetadata,
        target: std::sync::Arc<OutputTarget>,
        options: IngestOptions,
        output: &str,
    ) -> Result<Self> {
        let num_shards = source.num_shards.unwrap_or(1).max(1);
        let mut relation_defs = Vec::new();
        let mut relationships = Vec::new();
        for rel in &source.relationships {
            let base = rel.name.clone().unwrap_or_else(|| {
                format!(
                    "{}_{}_to_{}_{}",
                    rel.src_table, rel.src_column, rel.dst_table, rel.dst_column
                )
            });
            let dst_table = source
                .tables
                .iter()
                .find(|t| t.name == rel.dst_table)
                .ok_or_else(|| anyhow!("missing table {}", rel.dst_table))?;
            let src_table = source
                .tables
                .iter()
                .find(|t| t.name == rel.src_table)
                .ok_or_else(|| anyhow!("missing table {}", rel.src_table))?;
            let dst_sort = if dst_table.weight.is_some() {
                SortKeyMode::Weight
            } else if dst_table.create_time.is_some() {
                SortKeyMode::CreateTime
            } else {
                SortKeyMode::None
            };
            let src_sort = if src_table.weight.is_some() {
                SortKeyMode::Weight
            } else if src_table.create_time.is_some() {
                SortKeyMode::CreateTime
            } else {
                SortKeyMode::None
            };
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
                let src_rel = RelationDef {
                    name: format!("{base}_src_to_dst"),
                    src_table: edge.src_fk.ref_table.clone(),
                    dst_table: edge.dst_fk.ref_table.clone(),
                    sort_key: if table.weight.is_some() {
                        SortKeyMode::Weight
                    } else if table.create_time.is_some() {
                        SortKeyMode::CreateTime
                    } else {
                        SortKeyMode::None
                    },
                };
                let dst_rel = RelationDef {
                    name: format!("{base}_dst_to_src"),
                    src_table: edge.dst_fk.ref_table.clone(),
                    dst_table: edge.src_fk.ref_table.clone(),
                    sort_key: src_rel.sort_key,
                };
                relation_defs.push(src_rel);
                relation_defs.push(dst_rel);
            }
        }
        Ok(Self {
            source,
            target,
            profile_dir: options.profile_dir,
            num_shards,
            table_rows: HashMap::new(),
            table_contexts: HashMap::new(),
            shard_rows: HashMap::new(),
            shard_offsets: HashMap::new(),
            pk_map_paths: HashMap::new(),
            relation_defs,
            relationships,
            edge_counts: HashMap::new(),
            pk_duplicate_counts: HashMap::new(),
            pk_duplicate_samples: HashMap::new(),
            output: output.to_string(),
            trimmed_strings: std::sync::Arc::new(AtomicU64::new(0)),
            progress: options.progress,
            shuffle_task_ms: 0,
            pk_map_task_ms: 0,
            build_task_ms: 0,
            edge_task_ms: 0,
            index_task_ms: 0,
            context_task_ms: 0,
            shuffle_tasks: 0,
            pk_map_tasks: 0,
            build_tasks: 0,
            edge_tasks: 0,
            index_tasks: 0,
            context_tasks: 0,
        })
    }

    fn build_node_contexts(&mut self) -> Result<()> {
        let phase_start = std::time::Instant::now();
        let tables = self.source.tables.clone();
        for table in &tables {
            if table.edge_table.is_some() {
                continue;
            }
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Context,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Start,
                delta: 0,
                total: Some(table.paths.len() as u64),
            });
            let pk_col = table.primary_key.as_ref().and_then(|pk| column_spec(table, pk).ok());
            let pk_idx = table
                .primary_key
                .as_ref()
                .and_then(|pk| column_index(table, pk).ok());
            let rowpack_config = rowpack_config_for_table(self.source.rowpack.as_ref(), table);
            let mut buffered_rows: Vec<OwnedRow> = Vec::new();
            let mut buffered_total = 0usize;
            let mut reached_limit = false;
            if table.paths.is_empty() {
                return Err(anyhow!("no input files for table {}", table.name));
            }
            'file_loop: for (_file_idx, path) in table.paths.iter().enumerate() {
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
                            let mut values = Vec::with_capacity(table.columns.len());
                            let mut arrays_f32: Vec<Vec<f32>> = Vec::new();
                            let mut arrays_f64: Vec<Vec<f64>> = Vec::new();
                            let mut strings: Vec<String> = Vec::new();
                            let mut bytes: Vec<Vec<u8>> = Vec::new();
                            for row in 0..batch.num_rows() {
                                if buffered_total >= SAMPLE_ROW_LIMIT {
                                    reached_limit = true;
                                    break;
                                }
                                let row_view = RowView {
                                    row,
                                    cols: &cols,
                                };
                                let owned = owned_row_from_row(
                                    &row_view,
                                    table,
                                    &mut values,
                                    &mut arrays_f32,
                                    &mut arrays_f64,
                                    &mut strings,
                                    &mut bytes,
                                    &self.trimmed_strings,
                                )?;
                                buffered_rows.push(owned);
                                buffered_total = buffered_total.saturating_add(1);
                            }
                            if reached_limit {
                                break;
                            }
                        }
                    }
                    TableFormat::Csv => {
                        let file = fs::File::open(&local.path)
                            .with_context(|| format!("open {}", local.path.display()))?;
                        let schema = table_schema(table)?;
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
                            let mut values = Vec::with_capacity(table.columns.len());
                            let mut arrays_f32: Vec<Vec<f32>> = Vec::new();
                            let mut arrays_f64: Vec<Vec<f64>> = Vec::new();
                            let mut strings: Vec<String> = Vec::new();
                            let mut bytes: Vec<Vec<u8>> = Vec::new();
                            for row in 0..batch.num_rows() {
                                if buffered_total >= SAMPLE_ROW_LIMIT {
                                    reached_limit = true;
                                    break;
                                }
                                let row_view = RowView {
                                    row,
                                    cols: &cols,
                                };
                                let owned = owned_row_from_row(
                                    &row_view,
                                    table,
                                    &mut values,
                                    &mut arrays_f32,
                                    &mut arrays_f64,
                                    &mut strings,
                                    &mut bytes,
                                    &self.trimmed_strings,
                                )?;
                                buffered_rows.push(owned);
                                buffered_total = buffered_total.saturating_add(1);
                            }
                            if reached_limit {
                                break;
                            }
                        }
                    }
                }
                self.emit_progress(ProgressEvent {
                    phase: ProgressPhase::Context,
                    table: table.name.clone(),
                    shard: None,
                    kind: ProgressKind::Advance,
                    delta: 1,
                    total: Some(table.paths.len() as u64),
                });
                if reached_limit {
                    break 'file_loop;
                }
            }
            let mut node_context = None;
            let mut pkey_context = None;
            self.build_node_contexts_from_samples(
                table,
                pk_col,
                pk_idx,
                &rowpack_config,
                &buffered_rows,
                &mut node_context,
                &mut pkey_context,
            )?;
            let ctx_bytes = node_context
                .as_ref()
                .ok_or_else(|| anyhow!("missing node context"))?
                .to_bytes()?;
            self.target
                .write_bytes(&temp_node_context_path(&table.name), ctx_bytes.clone())?;
            self.target
                .write_bytes(&node_context_output_path(&table.name), ctx_bytes)?;
            if let Some(ctx) = pkey_context.as_ref() {
                let bytes = ctx.to_bytes()?;
                self.target
                    .write_bytes(&temp_pkey_context_path(&table.name), bytes)?;
            }
            self.table_contexts.insert(
                table.name.clone(),
                format!("nodes/{}/node_data.ctx", table.name.to_lowercase()),
            );
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Context,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Finish,
                delta: 0,
                total: Some(table.paths.len() as u64),
            });
        }
        let elapsed = phase_start.elapsed();
        self.context_task_ms = self
            .context_task_ms
            .saturating_add(elapsed.as_millis() as u64);
        self.context_tasks = self.context_tasks.saturating_add(1);
        Ok(())
    }

    fn build_temp_shards(&mut self) -> Result<()> {
        let phase_start = std::time::Instant::now();
        let root = require_local_output(&self.target)?.to_path_buf();
        let temp_root = root.join(TEMP_DIR_NAME);
        fs::create_dir_all(&temp_root)?;
        let mut table_rows = HashMap::new();
        let tables = self.source.tables.clone();
        for table in &tables {
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Shuffle,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Start,
                delta: 0,
                total: Some(table.paths.len() as u64),
            });
            if table.edge_table.is_some() {
                self.build_temp_edge_table(table)?;
            } else {
                let rows = self.build_temp_node_table(table)?;
                table_rows.insert(table.name.clone(), rows);
            }
            let total = Some(table.paths.len() as u64);
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Shuffle,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Finish,
                delta: 0,
                total,
            });
        }
        self.table_rows = table_rows;
        let elapsed = phase_start.elapsed();
        self.shuffle_task_ms = self
            .shuffle_task_ms
            .saturating_add(elapsed.as_millis() as u64);
        self.shuffle_tasks = self.shuffle_tasks.saturating_add(1);
        Ok(())
    }

    fn build_temp_node_table(&mut self, table: &TableSpec) -> Result<u64> {
        let pk_col = table.primary_key.as_ref().and_then(|pk| column_spec(table, pk).ok());
        let pk_idx = table
            .primary_key
            .as_ref()
            .and_then(|pk| column_index(table, pk).ok());
        let relations: Vec<RelationshipRuntime> = self
            .relationships
            .iter()
            .filter(|rel| rel.src_table == table.name)
            .cloned()
            .collect();
        let mut rel_specs = Vec::new();
        for rel in &relations {
            let idx = column_index(table, &rel.src_column)?;
            rel_specs.push((rel.name_forward.clone(), idx));
        }
        let mut total_rows = 0u64;
        let root = require_local_output(&self.target)?.to_path_buf();
        let ctx_path = root.join(temp_node_context_path(&table.name));
        let ctx_bytes =
            fs::read(&ctx_path).with_context(|| format!("read {}", ctx_path.display()))?;
        let node_context = diskgraph_rowpack::RowPackContext::from_bytes(&ctx_bytes)?;
        let node_encoder = Some(RowPackEncoder::new(node_context)?);
        let mut node_scratch = Some(RowPackScratch::new(
            node_encoder.as_ref().unwrap().context(),
        ));
        let mut pkey_encoder: Option<RowPackEncoder> = None;
        let mut pkey_scratch: Option<RowPackScratch> = None;
        let mut node_out: Vec<u8> = Vec::new();
        let mut pkey_out: Vec<u8> = Vec::new();
        let mut row_id = 0u64;
        let mut key_buf: Vec<u8> = Vec::new();
        let mut node_data_dropped = 0u64;
        let has_pkey = pk_col.is_some() && pk_idx.is_some();
        let mut shard_rows = vec![0u64; self.num_shards];
        if has_pkey {
            let pkey_ctx_path = root.join(temp_pkey_context_path(&table.name));
            let ctx_bytes = fs::read(&pkey_ctx_path)
                .with_context(|| format!("read {}", pkey_ctx_path.display()))?;
            let pkey_context = diskgraph_rowpack::RowPackContext::from_bytes(&ctx_bytes)?;
            pkey_encoder = Some(RowPackEncoder::new(pkey_context)?);
            pkey_scratch = Some(RowPackScratch::new(
                pkey_encoder.as_ref().unwrap().context(),
            ));
        }

        struct NodeTempWriters {
            node: Vec<TempRowWriter>,
            pkey: Option<Vec<TempRowWriter>>,
            fkey: HashMap<String, Vec<TempRowWriter>>,
        }

        let mut finish_writers = |file_idx: usize, mut writers: NodeTempWriters| -> Result<()> {
            for shard in 0..self.num_shards {
                let rows = finish_temp_writer(writers.node.remove(0))?;
                shard_rows[shard] = shard_rows[shard].saturating_add(rows);
                let meta_path = temp_node_data_meta_path(&table.name, shard, file_idx);
                write_temp_meta(&self.target, &meta_path, rows)?;
            }
            if let Some(mut writers) = writers.pkey.take() {
                for shard in 0..self.num_shards {
                    let rows = finish_temp_writer(writers.remove(0))?;
                    let meta_path = temp_pkey_meta_path(&table.name, shard, file_idx);
                    write_temp_meta(&self.target, &meta_path, rows)?;
                }
            }
            for (rel, mut writers) in writers.fkey {
                for shard in 0..self.num_shards {
                    let rows = finish_temp_writer(writers.remove(0))?;
                    let meta_path = temp_fkey_meta_path(&table.name, &rel, shard, file_idx);
                    write_temp_meta(&self.target, &meta_path, rows)?;
                }
            }
            Ok(())
        };

        for (file_idx, path) in table.paths.iter().enumerate() {
            if !is_cloud_path(path) && !Path::new(path).exists() {
                return Err(anyhow!("input path not found {}", path));
            }
            let local = localize_input(path)?;
            let mut writers = NodeTempWriters {
                node: Vec::new(),
                pkey: None,
                fkey: HashMap::new(),
            };
            self.init_node_temp_writers(
                table,
                file_idx,
                &mut writers.node,
                &mut writers.pkey,
                &mut writers.fkey,
                &rel_specs,
                has_pkey,
            )?;
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
                        let mut access = RowAccess::new(table, &cols, &self.trimmed_strings);
                        for row in 0..batch.num_rows() {
                            total_rows += 1;
                            access.reset(row);
                            self.write_node_row_from_access(
                                table,
                                pk_idx,
                                &rel_specs,
                                &mut access,
                                &node_encoder,
                                &pkey_encoder,
                                &mut node_scratch,
                                &mut pkey_scratch,
                                &mut node_out,
                                &mut pkey_out,
                                &mut writers.node,
                                &mut writers.pkey,
                                &mut writers.fkey,
                                &mut row_id,
                                pk_col,
                                &mut key_buf,
                                &mut node_data_dropped,
                            )?;
                        }
                    }
                }
                TableFormat::Csv => {
                    let file = fs::File::open(&local.path)
                        .with_context(|| format!("open {}", local.path.display()))?;
                    let schema = table_schema(table)?;
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
                        let mut access = RowAccess::new(table, &cols, &self.trimmed_strings);
                        for row in 0..batch.num_rows() {
                            total_rows += 1;
                            access.reset(row);
                            self.write_node_row_from_access(
                                table,
                                pk_idx,
                                &rel_specs,
                                &mut access,
                                &node_encoder,
                                &pkey_encoder,
                                &mut node_scratch,
                                &mut pkey_scratch,
                                &mut node_out,
                                &mut pkey_out,
                                &mut writers.node,
                                &mut writers.pkey,
                                &mut writers.fkey,
                                &mut row_id,
                                pk_col,
                                &mut key_buf,
                                &mut node_data_dropped,
                            )?;
                        }
                    }
                }
            }
            finish_writers(file_idx, writers)?;
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Shuffle,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Advance,
                delta: 1,
                total: Some(table.paths.len() as u64),
            });
        }

        if node_data_dropped > 0 {
            self.pk_duplicate_counts.insert(
                format!("node_data_dropped:{}", table.name),
                node_data_dropped,
            );
        }
        self.shard_rows.insert(table.name.clone(), shard_rows);
        Ok(total_rows)
    }

    fn build_node_contexts_from_samples(
        &self,
        table: &TableSpec,
        pk_col: Option<&ColumnSpec>,
        pk_idx: Option<usize>,
        rowpack_config: &RowPackConfig,
        rows: &[OwnedRow],
        node_context: &mut Option<diskgraph_rowpack::RowPackContext>,
        pkey_context: &mut Option<diskgraph_rowpack::RowPackContext>,
    ) -> Result<()> {
        let sample_idxs = sample_indices(rows.len(), SAMPLE_ROW_TARGET);
        let mut value_refs = Vec::new();
        let node_columns = rowpack_columns(table)?;
        let mut node_builder = RowPackContextBuilder::new(node_columns, rowpack_config.clone());
        for idx in &sample_idxs {
            rows[*idx].as_value_refs(&mut value_refs);
            node_builder.observe_row(&value_refs)?;
        }
        *node_context = Some(node_builder.finish()?);

        if let (Some(pk_col), Some(pk_idx)) = (pk_col, pk_idx) {
            let columns = rowpack_columns_for_specs(&[pk_col.clone()])?;
            let mut pk_builder = RowPackContextBuilder::new(columns, rowpack_config.clone());
            for idx in &sample_idxs {
                rows[*idx].as_value_refs_for_indices(&[pk_idx], &mut value_refs);
                pk_builder.observe_row(&value_refs)?;
            }
            *pkey_context = Some(pk_builder.finish()?);
        }

        Ok(())
    }

    fn init_node_temp_writers(
        &self,
        table: &TableSpec,
        file_idx: usize,
        node_writers: &mut Vec<TempRowWriter>,
        pkey_writers: &mut Option<Vec<TempRowWriter>>,
        fkey_writers: &mut HashMap<String, Vec<TempRowWriter>>,
        rel_specs: &[(String, usize)],
        has_pkey: bool,
    ) -> Result<()> {
        for shard in 0..self.num_shards {
            node_writers.push(open_temp_row_writer(
                &self.target,
                &temp_node_data_path(&table.name, shard, file_idx),
            )?);
        }
        if has_pkey {
            let mut writers = Vec::with_capacity(self.num_shards);
            for shard in 0..self.num_shards {
                writers.push(open_temp_row_writer(
                    &self.target,
                    &temp_pkey_path(&table.name, shard, file_idx),
                )?);
            }
            *pkey_writers = Some(writers);
        }
        for (rel, _) in rel_specs {
            let mut writers = Vec::with_capacity(self.num_shards);
            for shard in 0..self.num_shards {
                writers.push(open_temp_row_writer(
                    &self.target,
                    &temp_fkey_path(&table.name, rel, shard, file_idx),
                )?);
            }
            fkey_writers.insert(rel.clone(), writers);
        }
        Ok(())
    }

    fn write_node_row_from_access(
        &self,
        table: &TableSpec,
        pk_idx: Option<usize>,
        rel_specs: &[(String, usize)],
        access: &mut RowAccess<'_>,
        node_encoder: &Option<RowPackEncoder>,
        pkey_encoder: &Option<RowPackEncoder>,
        node_scratch: &mut Option<RowPackScratch>,
        pkey_scratch: &mut Option<RowPackScratch>,
        node_out: &mut Vec<u8>,
        pkey_out: &mut Vec<u8>,
        node_writers: &mut Vec<TempRowWriter>,
        pkey_writers: &mut Option<Vec<TempRowWriter>>,
        fkey_writers: &mut HashMap<String, Vec<TempRowWriter>>,
        row_id: &mut u64,
        pk_col: Option<&ColumnSpec>,
        key_buf: &mut Vec<u8>,
        node_data_dropped: &mut u64,
    ) -> Result<()> {
        let encoder = node_encoder
            .as_ref()
            .ok_or_else(|| anyhow!("missing node encoder"))?;
        let scratch = node_scratch
            .as_mut()
            .ok_or_else(|| anyhow!("missing node scratch"))?;
        encoder.encode_row_from_access(access, node_out, scratch)?;
        let shard = if let (Some(idx), Some(pkey_encoder), Some(pk_col)) =
            (pk_idx, pkey_encoder.as_ref(), pk_col)
        {
            let pk_value = access.value_ref(idx)?;
            let key = key_from_value_ref(&pk_value, pk_col).map_err(|_| anyhow!("pk parse"))?;
            let mut scratch = [0u8; 16];
            let shard = shard_for_key(key_hash_bytes(&key, &mut scratch), self.num_shards);
            let pkey_scratch = pkey_scratch
                .as_mut()
                .ok_or_else(|| anyhow!("missing pkey scratch"))?;
            pkey_encoder.encode_row_into(std::slice::from_ref(&pk_value), pkey_out, pkey_scratch)?;
            if let Some(writers) = pkey_writers.as_mut() {
                write_temp_row(&mut writers[shard], pkey_out)?;
            }
            shard
        } else {
            (*row_id as usize) % self.num_shards
        };
        if node_out.len() > MAX_NODE_DATA_LEN {
            *node_data_dropped = node_data_dropped.saturating_add(1);
            write_temp_row_u16(&mut node_writers[shard], &[])?;
        } else {
            write_temp_row_u16(&mut node_writers[shard], node_out)?;
        }
        for (rel_name, rel_idx) in rel_specs {
            if let Some(writers) = fkey_writers.get_mut(rel_name) {
                let col = table
                    .columns
                    .get(*rel_idx)
                    .ok_or_else(|| anyhow!("missing fk column"))?;
                let value = access.value_ref(*rel_idx)?;
                let key = key_from_value_ref(&value, col).map_err(|_| anyhow!("fk parse"))?;
                key_buf.clear();
                write_key_to_buf(key_buf, &key)?;
                write_temp_row(&mut writers[shard], key_buf)?;
            }
        }
        *row_id = row_id.saturating_add(1);
        Ok(())
    }

    fn build_temp_edge_table(&mut self, table: &TableSpec) -> Result<()> {
        let edge = table
            .edge_table
            .as_ref()
            .ok_or_else(|| anyhow!("missing edge table spec for {}", table.name))?;
        let src_fk_idx = column_index(table, &edge.src_fk.column)?;
        let dst_fk_idx = column_index(table, &edge.dst_fk.column)?;
        let src_fk_col = column_spec(table, &edge.src_fk.column)?;
        let dst_fk_col = column_spec(table, &edge.dst_fk.column)?;
        let schema = table_schema(table)?;

        for (file_idx, path) in table.paths.iter().enumerate() {
            if !is_cloud_path(path) && !Path::new(path).exists() {
                return Err(anyhow!("input path not found {}", path));
            }
            let local = localize_input(path)?;
            let mut forward_writers: Vec<TempRowWriter> = Vec::new();
            let mut reverse_writers: Vec<TempRowWriter> = Vec::new();
            self.init_edge_temp_writers(
                &table.name,
                file_idx,
                &mut forward_writers,
                &mut reverse_writers,
            )?;
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
                        let mut values = Vec::with_capacity(table.columns.len());
                        let mut arrays_f32: Vec<Vec<f32>> = Vec::new();
                        let mut arrays_f64: Vec<Vec<f64>> = Vec::new();
                        let mut strings: Vec<String> = Vec::new();
                        let mut bytes: Vec<Vec<u8>> = Vec::new();
                        for row in 0..batch.num_rows() {
                            let row_view = RowView {
                                row,
                                cols: &cols,
                            };
                            let owned = owned_row_from_row(
                                &row_view,
                                table,
                                &mut values,
                                &mut arrays_f32,
                                &mut arrays_f64,
                                &mut strings,
                                &mut bytes,
                                &self.trimmed_strings,
                            )?;
                            self.write_edge_row(
                                src_fk_idx,
                                dst_fk_idx,
                                src_fk_col,
                                dst_fk_col,
                                owned,
                                &mut forward_writers,
                                &mut reverse_writers,
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
                        let mut values = Vec::with_capacity(table.columns.len());
                        let mut arrays_f32: Vec<Vec<f32>> = Vec::new();
                        let mut arrays_f64: Vec<Vec<f64>> = Vec::new();
                        let mut strings: Vec<String> = Vec::new();
                        let mut bytes: Vec<Vec<u8>> = Vec::new();
                        for row in 0..batch.num_rows() {
                            let row_view = RowView {
                                row,
                                cols: &cols,
                            };
                            let owned = owned_row_from_row(
                                &row_view,
                                table,
                                &mut values,
                                &mut arrays_f32,
                                &mut arrays_f64,
                                &mut strings,
                                &mut bytes,
                                &self.trimmed_strings,
                            )?;
                            self.write_edge_row(
                                src_fk_idx,
                                dst_fk_idx,
                                src_fk_col,
                                dst_fk_col,
                                owned,
                                &mut forward_writers,
                                &mut reverse_writers,
                            )?;
                        }
                    }
                }
            }
            for shard in 0..self.num_shards {
                let rows = finish_temp_writer(forward_writers.remove(0))?;
                let meta_path = temp_edge_meta_path(
                    &format!("edge_{}_src_to_dst", table.name),
                    shard,
                    file_idx,
                );
                write_temp_meta(&self.target, &meta_path, rows)?;
            }
            for shard in 0..self.num_shards {
                let rows = finish_temp_writer(reverse_writers.remove(0))?;
                let meta_path = temp_edge_meta_path(
                    &format!("edge_{}_dst_to_src", table.name),
                    shard,
                    file_idx,
                );
                write_temp_meta(&self.target, &meta_path, rows)?;
            }
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Shuffle,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Advance,
                delta: 1,
                total: Some(table.paths.len() as u64),
            });
        }
        Ok(())
    }

    fn init_edge_temp_writers(
        &self,
        table_name: &str,
        file_idx: usize,
        forward_writers: &mut Vec<TempRowWriter>,
        reverse_writers: &mut Vec<TempRowWriter>,
    ) -> Result<()> {
        let forward_name = format!("edge_{}_src_to_dst", table_name);
        let reverse_name = format!("edge_{}_dst_to_src", table_name);
        for shard in 0..self.num_shards {
            forward_writers.push(open_temp_row_writer(
                &self.target,
                &temp_edge_path(&forward_name, shard, file_idx),
            )?);
            reverse_writers.push(open_temp_row_writer(
                &self.target,
                &temp_edge_path(&reverse_name, shard, file_idx),
            )?);
        }
        Ok(())
    }

    fn write_edge_row(
        &self,
        src_idx: usize,
        dst_idx: usize,
        src_col: &ColumnSpec,
        dst_col: &ColumnSpec,
        row: OwnedRow,
        forward_writers: &mut Vec<TempRowWriter>,
        reverse_writers: &mut Vec<TempRowWriter>,
    ) -> Result<()> {
        let mut value_refs = Vec::new();
        row.as_value_refs_for_indices(&[src_idx], &mut value_refs);
        let src_key = key_from_value_ref(&value_refs[0], src_col)
            .map_err(|_| anyhow!("src key parse"))?;
        row.as_value_refs_for_indices(&[dst_idx], &mut value_refs);
        let dst_key = key_from_value_ref(&value_refs[0], dst_col)
            .map_err(|_| anyhow!("dst key parse"))?;
        let mut scratch = [0u8; 16];
        let forward_shard = shard_for_key(key_hash_bytes(&src_key, &mut scratch), self.num_shards);
        let reverse_shard =
            shard_for_key(key_hash_bytes(&dst_key, &mut scratch), self.num_shards);
        let mut forward_row = Vec::new();
        write_key_to_buf(&mut forward_row, &src_key)?;
        write_key_to_buf(&mut forward_row, &dst_key)?;
        write_temp_row(&mut forward_writers[forward_shard], &forward_row)?;
        let mut reverse_row = Vec::new();
        write_key_to_buf(&mut reverse_row, &dst_key)?;
        write_key_to_buf(&mut reverse_row, &src_key)?;
        write_temp_row(&mut reverse_writers[reverse_shard], &reverse_row)?;
        Ok(())
    }

    fn build_sharded_pk_maps(&mut self) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        let limiter = Arc::new(MemoryLimiter::new(PK_MAP_MEMORY_BUDGET));
        let pk_map_time = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let duplicate_counts = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let duplicate_samples = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let mut tasks = Vec::new();
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
            let shard_rows = self
                .shard_rows
                .get(&table.name)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let pk_map_paths = std::sync::Arc::new(std::sync::Mutex::new(vec![
                None;
                self.num_shards
            ]));
            let file_count = table.paths.len();
            for shard in 0..self.num_shards {
                let limiter = Arc::clone(&limiter);
                let target = std::sync::Arc::clone(&self.target);
                let table_name = table.name.clone();
                let pk_col = pk_col.clone();
                let progress = self.progress.clone();
                let pk_map_paths = std::sync::Arc::clone(&pk_map_paths);
                let duplicate_counts = std::sync::Arc::clone(&duplicate_counts);
                let duplicate_samples = std::sync::Arc::clone(&duplicate_samples);
                let estimate = estimate_pk_map_bytes(&pk_col, shard_rows[shard] as usize);
                let root = root.clone();
                let pkey_ctx_rel = temp_pkey_context_path(&table.name);
                tasks.push(WorkItem {
                    bucket: Some(pk_map_time.clone()),
                    run: Box::new(move || {
                        let _permit = limiter.acquire(estimate);
                        let ctx_path = root.join(&pkey_ctx_rel);
                        let ctx_bytes = fs::read(&ctx_path)
                            .with_context(|| format!("read {}", ctx_path.display()))?;
                        let ctx = diskgraph_rowpack::RowPackContext::from_bytes(&ctx_bytes)?;
                        let decoder = RowPackDecoder::new(ctx)?;
                        let key_type = key_type_for_column(&pk_col)?;
                        let mapping_kind = match key_type {
                            KeyType::VarBytes => MappingKind::Hash,
                            _ => MappingKind::Sorted,
                        };
                        let mut total_rows = 0u64;
                        for file_idx in 0..file_count {
                            let meta_path =
                                root.join(temp_pkey_meta_path(&table_name, shard, file_idx));
                            let meta = read_temp_meta(&meta_path)?;
                            total_rows = total_rows.saturating_add(meta.row_count);
                        }
                        let mut pk_map =
                            PkMap::new(mapping_kind, key_type, total_rows as usize);
                        let mut row_id = 0u64;
                        let mut scratch = RowPackDecodeScratch::default();
                        for file_idx in 0..file_count {
                            let data_path =
                                root.join(temp_pkey_path(&table_name, shard, file_idx));
                            read_temp_rows(&data_path, |row| {
                                let value = decoder.decode_column_value(row, 0, &mut scratch)?;
                                if let Ok(key) = key_from_value_ref(&value, &pk_col) {
                                    pk_map.insert(key, row_id)?;
                                }
                                row_id += 1;
                                Ok(())
                            })?;
                        }
                        pk_map.finalize();
                        if pk_map.duplicate_count() > 0 {
                            let key = format!("{}:shard-{:05}", table_name, shard);
                            duplicate_counts
                                .lock()
                                .unwrap()
                                .insert(key.clone(), pk_map.duplicate_count());
                            let mut samples = Vec::new();
                            for key in pk_map.duplicate_samples() {
                                if let Some(sample) = sample_key(key) {
                                    samples.push(sample);
                                }
                            }
                            if !samples.is_empty() {
                                duplicate_samples.lock().unwrap().insert(key, samples);
                            }
                        }
                        let map_rel = format!(
                            "nodes/{}/pk_map-{:05}.bin",
                            table_name.to_lowercase(),
                            shard
                        );
                        let out_path = target.join(&map_rel);
                        if let OutputTarget::Local { .. } = &*target {
                            if let Some(parent) = out_path.parent() {
                                fs::create_dir_all(parent)?;
                            }
                        }
                        let mut writer =
                            BufWriter::with_capacity(1024 * 1024, target.sink(&out_path)?);
                        pk_map.write_to(&mut writer)?;
                        writer.flush()?;
                        let sink = writer.into_inner().map_err(|_| anyhow!("pk map flush failed"))?;
                        sink.finish()?;
                        let dup_meta = format!(
                            "nodes/{}/pk_map-{:05}.meta.json",
                            table_name.to_lowercase(),
                            shard
                        );
                        let mut samples = Vec::new();
                        for key in pk_map.duplicate_samples() {
                            if let Some(sample) = sample_key(key) {
                                samples.push(sample);
                            }
                        }
                        let meta = serde_json::json!({
                            "duplicate_count": pk_map.duplicate_count(),
                            "duplicate_samples": samples,
                        });
                        target.write_bytes(&dup_meta, serde_json::to_vec_pretty(&meta)?)?;
                        let mut guard = pk_map_paths.lock().unwrap();
                        guard[shard] = Some(map_rel);
                        if let Some(cb) = &progress {
                            cb(ProgressEvent {
                                phase: ProgressPhase::PkMap,
                                table: table_name.clone(),
                                shard: Some(shard),
                                kind: ProgressKind::Advance,
                                delta: 1,
                                total: Some(1),
                            });
                        }
                        Ok(())
                    }),
                });
            }
            run_tasks(tasks.drain(..).collect(), default_threads(), Some(pk_map_time.clone()))?;
            let paths = std::sync::Arc::try_unwrap(pk_map_paths)
                .unwrap()
                .into_inner()
                .unwrap()
                .into_iter()
                .map(|p| p.ok_or_else(|| anyhow!("missing pk map path for {}", table.name)))
                .collect::<Result<Vec<_>>>()?;
            self.pk_map_paths.insert(table.name.clone(), paths);
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::PkMap,
                table: table.name.clone(),
                shard: None,
                kind: ProgressKind::Finish,
                delta: 0,
                total: Some(self.num_shards as u64),
            });
        }
        self.pk_map_task_ms = pk_map_time.load(Ordering::Relaxed) / 1_000_000;
        self.pk_map_tasks = self.pk_map_paths.values().map(|v| v.len() as u64).sum();
        if let Ok(counts) = std::sync::Arc::try_unwrap(duplicate_counts) {
            self.pk_duplicate_counts = counts.into_inner().unwrap();
        }
        if let Ok(samples) = std::sync::Arc::try_unwrap(duplicate_samples) {
            self.pk_duplicate_samples = samples.into_inner().unwrap();
        }
        Ok(())
    }

    fn build_final_edges(&mut self) -> Result<()> {
        let phase_start = std::time::Instant::now();
        let root = require_local_output(&self.target)?.to_path_buf();
        self.build_shard_offsets();
        let mut rel_sort_bits = HashMap::new();
        for rel in &self.relation_defs {
            let sort_bits = match rel.sort_key {
                SortKeyMode::None => None,
                _ => Some(32),
            };
            rel_sort_bits.insert(rel.name.clone(), sort_bits);
        }
        let mut edge_counts = HashMap::new();
        let mut pk_maps_by_table: HashMap<String, Vec<PkMap>> = HashMap::new();
        for (table, paths) in &self.pk_map_paths {
            let mut maps = Vec::with_capacity(paths.len());
            for path in paths {
                let full = root.join(path);
                let data = fs::read(&full).with_context(|| format!("read {}", full.display()))?;
                maps.push(PkMap::read_from(&data)?);
            }
            pk_maps_by_table.insert(table.clone(), maps);
        }
        let mut max_ids = HashMap::new();
        for (table, counts) in &self.shard_rows {
            let total: u64 = counts.iter().sum();
            let max_id = total.saturating_sub(1);
            max_ids.insert(table.clone(), max_id);
        }
        for rel in &self.relationships {
            let file_count = table_by_name(&self.source, &rel.src_table)?.paths.len();
            let mut rel_total_rows = 0u64;
            for shard in 0..self.num_shards {
                for file_idx in 0..file_count {
                    let fkey_meta = root.join(temp_fkey_meta_path(
                        &rel.src_table,
                        &rel.name_forward,
                        shard,
                        file_idx,
                    ));
                    let meta = read_temp_meta(&fkey_meta)?;
                    rel_total_rows = rel_total_rows.saturating_add(meta.row_count);
                }
            }
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Edges,
                table: rel.name_forward.clone(),
                shard: None,
                kind: ProgressKind::Start,
                delta: 0,
                total: Some(rel_total_rows),
            });
            let src_table = table_by_name(&self.source, &rel.src_table)?;
            let src_fk_col = column_spec(src_table, &rel.src_column)?;
            let src_offsets = self
                .shard_offsets
                .get(&rel.src_table)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let dst_offsets = self
                .shard_offsets
                .get(&rel.dst_table)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let dst_maps = pk_maps_by_table
                .get(&rel.dst_table)
                .ok_or_else(|| anyhow!("missing pk maps for {}", rel.dst_table))?;
            for shard in 0..self.num_shards {
                let mut total_rows = 0u64;
                for file_idx in 0..file_count {
                    let fkey_meta = root.join(temp_fkey_meta_path(
                        &rel.src_table,
                        &rel.name_forward,
                        shard,
                        file_idx,
                    ));
                    let meta = read_temp_meta(&fkey_meta)?;
                    total_rows = total_rows.saturating_add(meta.row_count);
                }
                let forward_rel = temp_edge_shard_output_path(&rel.name_forward, shard);
                let reverse_rel = temp_edge_shard_output_path(&rel.name_reverse, shard);
                let forward_path = self.target.join(&forward_rel);
                let reverse_path = self.target.join(&reverse_rel);
                if let OutputTarget::Local { .. } = &*self.target {
                    if let Some(parent) = forward_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    if let Some(parent) = reverse_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                }
                let forward_sort_bits = rel_sort_bits.get(&rel.name_forward).copied().flatten();
                let reverse_sort_bits = rel_sort_bits.get(&rel.name_reverse).copied().flatten();
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
                let mut forward_edges = Vec::new();
                let mut reverse_edges = Vec::new();
                if let Ok(capacity) = usize::try_from(total_rows) {
                    forward_edges.reserve(capacity);
                    reverse_edges.reserve(capacity);
                }
                let mut forward_sorted = true;
                let mut reverse_sorted = true;
                let mut forward_prev_src: Option<u64> = None;
                let mut reverse_prev_src: Option<u64> = None;
                let mut count = 0u64;
                let mut scratch = [0u8; 16];
                let mut pending = 0u64;
                for file_idx in 0..file_count {
                    let fkey_path = root.join(temp_fkey_path(
                        &rel.src_table,
                        &rel.name_forward,
                        shard,
                        file_idx,
                    ));
                    read_temp_rows(&fkey_path, |row| {
                        pending = pending.saturating_add(1);
                        if pending >= PROGRESS_ROW_BATCH {
                            self.emit_progress(ProgressEvent {
                                phase: ProgressPhase::Edges,
                                table: rel.name_forward.clone(),
                                shard: None,
                                kind: ProgressKind::Advance,
                                delta: pending,
                                total: Some(rel_total_rows),
                            });
                            pending = 0;
                        }
                        let mut data = row;
                        let dst_key = read_key_from_buf(&mut data, src_fk_col)
                            .map_err(|_| anyhow!("fk parse"))?;
                        let dst_shard =
                            shard_for_key(key_hash_bytes(&dst_key, &mut scratch), self.num_shards);
                        let src_id = src_offsets[shard].saturating_add(count);
                        let dst_local = dst_maps
                            .get(dst_shard)
                            .and_then(|map| map.lookup(&dst_key))
                            .ok_or_else(|| anyhow!("missing dst key"))?;
                        let dst_id = dst_offsets
                            .get(dst_shard)
                            .copied()
                            .unwrap_or(0)
                            .saturating_add(dst_local);
                        if let Some(prev) = forward_prev_src {
                            if src_id < prev {
                                forward_sorted = false;
                            }
                        }
                        forward_prev_src = Some(src_id);
                        forward_edges.push(EdgeRecord {
                            src_id,
                            dst_id,
                            sort_key: 0,
                        });
                        if let Some(prev) = reverse_prev_src {
                            if dst_id < prev {
                                reverse_sorted = false;
                            }
                        }
                        reverse_prev_src = Some(dst_id);
                        reverse_edges.push(EdgeRecord {
                            src_id: dst_id,
                            dst_id: src_id,
                            sort_key: 0,
                        });
                        count += 1;
                        Ok(())
                    })?;
                }
                if pending > 0 {
                    self.emit_progress(ProgressEvent {
                        phase: ProgressPhase::Edges,
                        table: rel.name_forward.clone(),
                        shard: None,
                        kind: ProgressKind::Advance,
                        delta: pending,
                        total: Some(rel_total_rows),
                    });
                }
                write_edge_records(
                    &self.target,
                    &forward_path,
                    forward_spec,
                    forward_edges,
                    forward_sorted,
                )?;
                write_edge_records(
                    &self.target,
                    &reverse_path,
                    reverse_spec,
                    reverse_edges,
                    reverse_sorted,
                )?;
                edge_counts.insert(rel.name_forward.clone(), count);
                edge_counts.insert(rel.name_reverse.clone(), count);
            }
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Edges,
                table: rel.name_forward.clone(),
                shard: None,
                kind: ProgressKind::Finish,
                delta: 0,
                total: Some(rel_total_rows),
            });
        }

        for table in self.source.tables.iter().filter(|t| t.edge_table.is_some()) {
            let edge = table
                .edge_table
                .as_ref()
                .ok_or_else(|| anyhow!("missing edge table spec for {}", table.name))?;
            let forward_name = format!("edge_{}_src_to_dst", table.name);
            let reverse_name = format!("edge_{}_dst_to_src", table.name);
            let file_count = table.paths.len();
            let mut forward_total = 0u64;
            let mut reverse_total = 0u64;
            for shard in 0..self.num_shards {
                for file_idx in 0..file_count {
                    let meta = read_temp_meta(
                        &root.join(temp_edge_meta_path(&forward_name, shard, file_idx)),
                    )?;
                    forward_total = forward_total.saturating_add(meta.row_count);
                    let meta = read_temp_meta(
                        &root.join(temp_edge_meta_path(&reverse_name, shard, file_idx)),
                    )?;
                    reverse_total = reverse_total.saturating_add(meta.row_count);
                }
            }
            let total_rows = forward_total.saturating_add(reverse_total);
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Edges,
                table: forward_name.clone(),
                shard: None,
                kind: ProgressKind::Start,
                delta: 0,
                total: Some(total_rows),
            });
            let src_table = table_by_name(&self.source, &edge.src_fk.ref_table)?;
            let dst_table = table_by_name(&self.source, &edge.dst_fk.ref_table)?;
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
            let src_maps = pk_maps_by_table
                .get(&edge.src_fk.ref_table)
                .ok_or_else(|| anyhow!("missing pk maps for {}", edge.src_fk.ref_table))?;
            let dst_maps = pk_maps_by_table
                .get(&edge.dst_fk.ref_table)
                .ok_or_else(|| anyhow!("missing pk maps for {}", edge.dst_fk.ref_table))?;
            let src_offsets = self
                .shard_offsets
                .get(&edge.src_fk.ref_table)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            let dst_offsets = self
                .shard_offsets
                .get(&edge.dst_fk.ref_table)
                .cloned()
                .unwrap_or_else(|| vec![0; self.num_shards]);
            for shard in 0..self.num_shards {
                let mut forward_rows = 0u64;
                for file_idx in 0..file_count {
                    let meta = read_temp_meta(
                        &root.join(temp_edge_meta_path(&forward_name, shard, file_idx)),
                    )?;
                    forward_rows = forward_rows.saturating_add(meta.row_count);
                }
                let out_rel = temp_edge_shard_output_path(&forward_name, shard);
                let out_path = self.target.join(&out_rel);
                if let OutputTarget::Local { .. } = &*self.target {
                    if let Some(parent) = out_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                }
                let forward_sort_bits = rel_sort_bits.get(&forward_name).copied().flatten();
                let spec = edge_binary_spec(
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
                let mut forward_edges = Vec::new();
                if let Ok(capacity) = usize::try_from(forward_rows) {
                    forward_edges.reserve(capacity);
                }
                let mut forward_sorted = true;
                let mut forward_prev_src: Option<u64> = None;
                let mut count = 0u64;
                let mut scratch = [0u8; 16];
                let mut pending = 0u64;
                for file_idx in 0..file_count {
                    let forward_path = root.join(temp_edge_path(&forward_name, shard, file_idx));
                    read_temp_rows(&forward_path, |row| {
                        pending = pending.saturating_add(1);
                        if pending >= PROGRESS_ROW_BATCH {
                            self.emit_progress(ProgressEvent {
                                phase: ProgressPhase::Edges,
                                table: forward_name.clone(),
                                shard: None,
                                kind: ProgressKind::Advance,
                                delta: pending,
                                total: Some(total_rows),
                            });
                            pending = 0;
                        }
                        let mut data = row;
                        let src_key = read_key_from_buf(&mut data, src_pk_col)
                            .map_err(|_| anyhow!("src key parse"))?;
                        let dst_key = read_key_from_buf(&mut data, dst_pk_col)
                            .map_err(|_| anyhow!("dst key parse"))?;
                        let src_shard =
                            shard_for_key(key_hash_bytes(&src_key, &mut scratch), self.num_shards);
                        let dst_shard =
                            shard_for_key(key_hash_bytes(&dst_key, &mut scratch), self.num_shards);
                        let src_local = src_maps
                            .get(src_shard)
                            .and_then(|map| map.lookup(&src_key))
                            .ok_or_else(|| anyhow!("missing src key"))?;
                        let dst_local = dst_maps
                            .get(dst_shard)
                            .and_then(|map| map.lookup(&dst_key))
                            .ok_or_else(|| anyhow!("missing dst key"))?;
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
                        if let Some(prev) = forward_prev_src {
                            if src_id < prev {
                                forward_sorted = false;
                            }
                        }
                        forward_prev_src = Some(src_id);
                        forward_edges.push(EdgeRecord {
                            src_id,
                            dst_id,
                            sort_key: 0,
                        });
                        count += 1;
                        Ok(())
                    })?;
                }
                if pending > 0 {
                    self.emit_progress(ProgressEvent {
                        phase: ProgressPhase::Edges,
                        table: forward_name.clone(),
                        shard: None,
                        kind: ProgressKind::Advance,
                        delta: pending,
                        total: Some(total_rows),
                    });
                }
                write_edge_records(
                    &self.target,
                    &out_path,
                    spec,
                    forward_edges,
                    forward_sorted,
                )?;
                edge_counts.insert(forward_name.clone(), count);

                let mut reverse_rows = 0u64;
                for file_idx in 0..file_count {
                    let meta = read_temp_meta(
                        &root.join(temp_edge_meta_path(&reverse_name, shard, file_idx)),
                    )?;
                    reverse_rows = reverse_rows.saturating_add(meta.row_count);
                }
                let out_rel = temp_edge_shard_output_path(&reverse_name, shard);
                let out_path = self.target.join(&out_rel);
                if let OutputTarget::Local { .. } = &*self.target {
                    if let Some(parent) = out_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                }
                let reverse_sort_bits = rel_sort_bits.get(&reverse_name).copied().flatten();
                let spec = edge_binary_spec(
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
                let mut reverse_edges = Vec::new();
                if let Ok(capacity) = usize::try_from(reverse_rows) {
                    reverse_edges.reserve(capacity);
                }
                let mut reverse_sorted = true;
                let mut reverse_prev_src: Option<u64> = None;
                let mut count = 0u64;
                let mut scratch = [0u8; 16];
                let mut pending = 0u64;
                for file_idx in 0..file_count {
                    let reverse_path = root.join(temp_edge_path(&reverse_name, shard, file_idx));
                    read_temp_rows(&reverse_path, |row| {
                        pending = pending.saturating_add(1);
                        if pending >= PROGRESS_ROW_BATCH {
                            self.emit_progress(ProgressEvent {
                                phase: ProgressPhase::Edges,
                                table: forward_name.clone(),
                                shard: None,
                                kind: ProgressKind::Advance,
                                delta: pending,
                                total: Some(total_rows),
                            });
                            pending = 0;
                        }
                        let mut data = row;
                        let dst_key = read_key_from_buf(&mut data, dst_pk_col)
                            .map_err(|_| anyhow!("dst key parse"))?;
                        let src_key = read_key_from_buf(&mut data, src_pk_col)
                            .map_err(|_| anyhow!("src key parse"))?;
                        let dst_shard =
                            shard_for_key(key_hash_bytes(&dst_key, &mut scratch), self.num_shards);
                        let src_shard =
                            shard_for_key(key_hash_bytes(&src_key, &mut scratch), self.num_shards);
                        let dst_local = dst_maps
                            .get(dst_shard)
                            .and_then(|map| map.lookup(&dst_key))
                            .ok_or_else(|| anyhow!("missing dst key"))?;
                        let src_local = src_maps
                            .get(src_shard)
                            .and_then(|map| map.lookup(&src_key))
                            .ok_or_else(|| anyhow!("missing src key"))?;
                        let dst_id = dst_offsets
                            .get(dst_shard)
                            .copied()
                            .unwrap_or(0)
                            .saturating_add(dst_local);
                        let src_id = src_offsets
                            .get(src_shard)
                            .copied()
                            .unwrap_or(0)
                            .saturating_add(src_local);
                        if let Some(prev) = reverse_prev_src {
                            if dst_id < prev {
                                reverse_sorted = false;
                            }
                        }
                        reverse_prev_src = Some(dst_id);
                        reverse_edges.push(EdgeRecord {
                            src_id: dst_id,
                            dst_id: src_id,
                            sort_key: 0,
                        });
                        count += 1;
                        Ok(())
                    })?;
                }
                if pending > 0 {
                    self.emit_progress(ProgressEvent {
                        phase: ProgressPhase::Edges,
                        table: forward_name.clone(),
                        shard: None,
                        kind: ProgressKind::Advance,
                        delta: pending,
                        total: Some(total_rows),
                    });
                }
                write_edge_records(
                    &self.target,
                    &out_path,
                    spec,
                    reverse_edges,
                    reverse_sorted,
                )?;
                edge_counts.insert(reverse_name.clone(), count);
            }
            self.emit_progress(ProgressEvent {
                phase: ProgressPhase::Edges,
                table: forward_name.clone(),
                shard: None,
                kind: ProgressKind::Finish,
                delta: 0,
                total: Some(total_rows),
            });
        }
        self.edge_counts = edge_counts;
        let elapsed = phase_start.elapsed();
        self.edge_task_ms = self
            .edge_task_ms
            .saturating_add(elapsed.as_millis() as u64);
        self.build_task_ms = self
            .build_task_ms
            .saturating_add(elapsed.as_millis() as u64);
        self.edge_tasks = self.edge_tasks.saturating_add(1);
        self.build_tasks = self.build_tasks.saturating_add(1);
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
                fs::remove_dir_all(temp_root)?;
            }
        }
        Ok(())
    }

    fn cleanup_temp_inputs(&self) -> Result<()> {
        fn purge_dir(dir: &Path) -> Result<()> {
            let mut entries = fs::read_dir(dir)?;
            while let Some(entry) = entries.next() {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    purge_dir(&path)?;
                    if fs::read_dir(&path)?.next().is_none() {
                        fs::remove_dir(&path)?;
                    }
                    continue;
                }
                let name = match path.file_name().and_then(|n| n.to_str()) {
                    Some(name) => name,
                    None => continue,
                };
                match name {
                    "node_data.bin"
                    | "node_data.ctx"
                    | "node_data.meta"
                    | "pkey.bin"
                    | "pkey.ctx"
                    | "pkey.meta"
                    | "rows.bin"
                    | "rows.meta" => {
                        let _ = fs::remove_file(&path);
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        if let OutputTarget::Local { root } = &*self.target {
            let temp_root = root.join(TEMP_DIR_NAME);
            if temp_root.exists() {
                purge_dir(&temp_root)?;
            }
        }
        Ok(())
    }

    fn build_index(&mut self, metadata: DatasetMetadata) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        let builder = GraphIndexBuilder::from_metadata(
            metadata,
            GraphDataSource::Local { root: root.clone() },
            &root,
        );
        if let Some(progress) = self.progress.clone() {
            let adapter = IndexProgressAdapter::new(progress);
            let output = builder.build(Some(&adapter))?;
            self.index_task_ms = output.total_shard_ms;
            self.index_tasks = output.total_shards;
        } else {
            let output = builder.build(None)?;
            self.index_task_ms = output.total_shard_ms;
            self.index_tasks = output.total_shards;
        }
        Ok(())
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
            let file_count = table.paths.len();
            let mut shards = Vec::new();
            for shard in 0..self.num_shards {
                let count = shard_rows.get(shard).copied().unwrap_or(0);
                let min_id = shard_offsets.get(shard).copied().unwrap_or(0);
                let max_id = if count == 0 {
                    min_id.saturating_sub(1)
                } else {
                    min_id.saturating_add(count.saturating_sub(1))
                };
                let mut node_data_paths = Vec::with_capacity(file_count);
                for file_idx in 0..file_count {
                    node_data_paths.push(temp_node_data_path(&table.name, shard, file_idx));
                }
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
                mapping_path: format!("nodes/{}/pk_map-{{shard}}.bin", table.name.to_lowercase()),
                mapping_kind: "sharded".to_string(),
                fixed_len: table
                    .columns
                    .iter()
                    .find(|c| c.name == *pk)
                    .and_then(|c| c.fixed_len),
                key_mode: table
                    .columns
                    .iter()
                    .find(|c| c.name == *pk)
                    .and_then(|c| c.key_mode.as_ref())
                    .and_then(|mode| match mode {
                        KeyMode::Uuid => Some("uuid".to_string()),
                        KeyMode::Fixed => None,
                    }),
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
                let max_id = if count == 0 {
                    min_id.saturating_sub(1)
                } else {
                    min_id.saturating_add(count.saturating_sub(1))
                };
                shards.push(EdgeShardMetadata {
                    path: temp_edge_shard_output_path(&rel.name, shard),
                    src_min_id: min_id,
                    src_max_id: max_id,
                });
            }
            edge_relations.push(EdgeRelationMetadata {
                name: rel.name.clone(),
                src_type: rel.src_table.clone(),
                dst_type: rel.dst_table.clone(),
                shards,
                dest_sort_key_bits: match rel.sort_key {
                    SortKeyMode::None => None,
                    _ => Some(32),
                },
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

    fn build_stats(&self, metadata: &DatasetMetadata, mut perf: PerfStats) -> IngestStats {
        perf.context_task_ms = self.context_task_ms;
        perf.shuffle_task_ms = self.shuffle_task_ms;
        perf.edge_task_ms = self.edge_task_ms;
        perf.index_task_ms = self.index_task_ms;
        perf.context_tasks = self.context_tasks;
        perf.shuffle_tasks = self.shuffle_tasks;
        perf.edge_tasks = self.edge_tasks;
        perf.index_tasks = self.index_tasks;
        let mut node_counts = HashMap::new();
        for node in metadata.node_types.iter() {
            let count = node.max_id.saturating_add(1);
            node_counts.insert(node.name.clone(), count);
        }
        let total_nodes = node_counts.values().sum();
        let total_edges = self.edge_counts.values().sum();
        let key_failure_samples = Vec::new();
        let key_failures_by_context = HashMap::new();
        let key_parse_failures = 0u64;
        let key_lookup_failures = 0u64;
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
            key_parse_failures,
            key_lookup_failures,
            key_failure_samples,
            key_failures_by_context,
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

#[derive(Clone, Copy, Debug)]
struct EdgeBinarySpec {
    src_width: u8,
    dst_width: u8,
    sort_width: u8,
}

#[derive(Clone, Copy, Debug)]
struct EdgeRecord {
    src_id: u64,
    dst_id: u64,
    sort_key: u64,
}

fn temp_edge_shard_output_path(name: &str, shard: usize) -> String {
    format!(
        "{}/edges/{}/shard-{:05}/part-00000.bin",
        TEMP_DIR_NAME,
        name.to_lowercase(),
        shard
    )
}

fn temp_node_shard_dir(name: &str, shard: usize) -> String {
    format!(
        "{}/nodes/{}/shard-{:05}",
        TEMP_DIR_NAME,
        name.to_lowercase(),
        shard
    )
}

fn temp_node_data_path(name: &str, shard: usize, file_idx: usize) -> String {
    format!(
        "{}/file-{:05}/node_data.bin",
        temp_node_shard_dir(name, shard),
        file_idx
    )
}

fn temp_node_data_meta_path(name: &str, shard: usize, file_idx: usize) -> String {
    format!(
        "{}/file-{:05}/node_data.meta",
        temp_node_shard_dir(name, shard),
        file_idx
    )
}

fn temp_node_context_path(name: &str) -> String {
    format!("{}/nodes/{}/node_data.ctx", TEMP_DIR_NAME, name.to_lowercase())
}

fn node_context_output_path(name: &str) -> String {
    format!("nodes/{}/node_data.ctx", name.to_lowercase())
}

fn temp_pkey_path(name: &str, shard: usize, file_idx: usize) -> String {
    format!(
        "{}/file-{:05}/pkey.bin",
        temp_node_shard_dir(name, shard),
        file_idx
    )
}

fn temp_pkey_meta_path(name: &str, shard: usize, file_idx: usize) -> String {
    format!(
        "{}/file-{:05}/pkey.meta",
        temp_node_shard_dir(name, shard),
        file_idx
    )
}

fn temp_pkey_context_path(name: &str) -> String {
    format!("{}/nodes/{}/pkey.ctx", TEMP_DIR_NAME, name.to_lowercase())
}

fn temp_fkey_shard_dir(table: &str, rel: &str, shard: usize) -> String {
    format!(
        "{}/nodes/{}/fkeys/{}/shard-{:05}",
        TEMP_DIR_NAME,
        table.to_lowercase(),
        rel.to_lowercase(),
        shard
    )
}

fn temp_fkey_path(table: &str, rel: &str, shard: usize, file_idx: usize) -> String {
    format!(
        "{}/file-{:05}/rows.bin",
        temp_fkey_shard_dir(table, rel, shard),
        file_idx
    )
}

fn temp_fkey_meta_path(table: &str, rel: &str, shard: usize, file_idx: usize) -> String {
    format!(
        "{}/file-{:05}/rows.meta",
        temp_fkey_shard_dir(table, rel, shard),
        file_idx
    )
}

fn temp_edge_shard_dir(rel: &str, shard: usize) -> String {
    format!(
        "{}/edges/{}/shard-{:05}",
        TEMP_DIR_NAME,
        rel.to_lowercase(),
        shard
    )
}

fn temp_edge_path(rel: &str, shard: usize, file_idx: usize) -> String {
    format!(
        "{}/file-{:05}/rows.bin",
        temp_edge_shard_dir(rel, shard),
        file_idx
    )
}

fn temp_edge_meta_path(rel: &str, shard: usize, file_idx: usize) -> String {
    format!(
        "{}/file-{:05}/rows.meta",
        temp_edge_shard_dir(rel, shard),
        file_idx
    )
}

fn write_temp_meta(target: &OutputTarget, rel_path: &str, row_count: u64) -> Result<()> {
    let mut payload = Vec::with_capacity(16);
    payload.extend_from_slice(&TEMP_META_MAGIC);
    payload.push(TEMP_META_VERSION);
    payload.push(0u8);
    payload.extend_from_slice(&row_count.to_le_bytes());
    let ctx_len = 0u32;
    payload.extend_from_slice(&ctx_len.to_le_bytes());
    target.write_bytes(rel_path, payload)
}

fn read_temp_meta(path: &Path) -> Result<TempRowpackMeta> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 4 + 1 + 1 + 8 + 4 {
        return Err(anyhow!("temp meta truncated"));
    }
    if bytes[0..4] != TEMP_META_MAGIC {
        return Err(anyhow!("temp meta magic mismatch"));
    }
    if bytes[4] != TEMP_META_VERSION {
        return Err(anyhow!("temp meta version mismatch"));
    }
    let row_count = u64::from_le_bytes(bytes[6..14].try_into().unwrap());
    let ctx_len = u32::from_le_bytes(bytes[14..18].try_into().unwrap()) as usize;
    if bytes.len() < 18 + ctx_len {
        return Err(anyhow!("temp meta context truncated"));
    }
    Ok(TempRowpackMeta { row_count })
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
        writer: BufWriter::with_capacity(1024 * 1024, sink),
        row_count: 0,
    })
}

fn write_temp_row(writer: &mut TempRowWriter, row: &[u8]) -> Result<()> {
    let len = row.len() as u32;
    writer.writer.write_all(&len.to_le_bytes())?;
    writer.writer.write_all(row)?;
    writer.row_count = writer.row_count.saturating_add(1);
    Ok(())
}

fn write_temp_row_u16(writer: &mut TempRowWriter, row: &[u8]) -> Result<()> {
    let len = u16::try_from(row.len()).map_err(|_| anyhow!("temp row too large"))?;
    writer.writer.write_all(&len.to_le_bytes())?;
    writer.writer.write_all(row)?;
    writer.row_count = writer.row_count.saturating_add(1);
    Ok(())
}

fn finish_temp_writer(writer: TempRowWriter) -> Result<u64> {
    let mut writer = writer;
    writer.writer.flush()?;
    let sink = writer
        .writer
        .into_inner()
        .map_err(|_| anyhow!("temp writer flush failed"))?;
    sink.finish()?;
    Ok(writer.row_count)
}

fn read_temp_rows(path: &Path, mut f: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut file = io::BufReader::with_capacity(1024 * 1024, file);
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

fn write_edge_bin_header(writer: &mut BufWriter<OutputSink>, spec: EdgeBinarySpec) -> Result<()> {
    writer.write_all(&[spec.src_width, spec.dst_width, spec.sort_width])?;
    Ok(())
}

fn write_u64_width(writer: &mut BufWriter<OutputSink>, value: u64, width: u8) -> Result<()> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes[..width as usize])?;
    Ok(())
}

fn write_edge_records(
    target: &OutputTarget,
    path: &Path,
    spec: EdgeBinarySpec,
    mut records: Vec<EdgeRecord>,
    already_sorted: bool,
) -> Result<()> {
    if !already_sorted {
        radix_sort_edge_records(&mut records, spec.src_width);
        if spec.sort_width > 0 {
            sort_edge_ranges_by_sort_key(&mut records);
        }
    }
    let mut writer = BufWriter::with_capacity(1024 * 1024, target.sink(path)?);
    write_edge_bin_header(&mut writer, spec)?;
    for record in records {
        write_u64_width(&mut writer, record.src_id, spec.src_width)?;
        write_u64_width(&mut writer, record.dst_id, spec.dst_width)?;
        if spec.sort_width > 0 {
            write_u64_width(&mut writer, record.sort_key, spec.sort_width)?;
        }
    }
    writer.flush()?;
    let sink = writer
        .into_inner()
        .map_err(|_| anyhow!("edge flush failed"))?;
    sink.finish()?;
    Ok(())
}

fn sort_edge_ranges_by_sort_key(records: &mut [EdgeRecord]) {
    if records.len() <= 1 {
        return;
    }
    let mut start = 0usize;
    while start < records.len() {
        let src_id = records[start].src_id;
        let mut end = start + 1;
        while end < records.len() && records[end].src_id == src_id {
            end += 1;
        }
        if end - start > 1 {
            records[start..end].sort_unstable_by(|a, b| b.sort_key.cmp(&a.sort_key));
        }
        start = end;
    }
}

fn radix_sort_edge_records(records: &mut [EdgeRecord], src_width: u8) {
    if records.len() <= 1 {
        return;
    }
    let mut or_src = 0u64;
    for rec in records.iter() {
        or_src |= rec.src_id;
    }
    let mut passes = ((64 - or_src.leading_zeros() + 7) / 8) as usize;
    passes = passes.max(1).min(src_width as usize).min(8);
    if passes == 0 {
        return;
    }
    let mut tmp = vec![
        EdgeRecord {
            src_id: 0,
            dst_id: 0,
            sort_key: 0,
        };
        records.len()
    ];
    let mut swapped = false;
    {
        let mut from: &mut [EdgeRecord] = records;
        let mut to: &mut [EdgeRecord] = tmp.as_mut_slice();
        for pass in 0..passes {
            let shift = pass * 8;
            let mut counts = [0usize; 256];
            for rec in from.iter() {
                let byte = ((rec.src_id >> shift) & 0xFF) as usize;
                counts[byte] += 1;
            }
            let mut sum = 0usize;
            for count in counts.iter_mut() {
                let current = *count;
                *count = sum;
                sum = sum.saturating_add(current);
            }
            for rec in from.iter() {
                let byte = ((rec.src_id >> shift) & 0xFF) as usize;
                let idx = counts[byte];
                to[idx] = *rec;
                counts[byte] = idx + 1;
            }
            std::mem::swap(&mut from, &mut to);
            swapped = !swapped;
        }
        if swapped {
            to.copy_from_slice(from);
        }
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

struct WorkItem {
    run: Box<dyn FnOnce() -> Result<()> + Send>,
    bucket: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
}

struct MemoryLimiter {
    available: Mutex<usize>,
    cvar: Condvar,
}

struct MemoryPermit {
    limiter: Arc<MemoryLimiter>,
    amount: usize,
}


impl MemoryLimiter {
    fn new(total: usize) -> Self {
        Self {
            available: Mutex::new(total),
            cvar: Condvar::new(),
        }
    }

    fn acquire(self: &Arc<Self>, amount: usize) -> MemoryPermit {
        let mut available = self.available.lock().unwrap();
        while *available < amount {
            available = self.cvar.wait(available).unwrap();
        }
        *available -= amount;
        MemoryPermit {
            limiter: Arc::clone(self),
            amount,
        }
    }

    fn release(&self, amount: usize) {
        let mut available = self.available.lock().unwrap();
        *available = available.saturating_add(amount);
        self.cvar.notify_all();
    }
}

impl Drop for MemoryPermit {
    fn drop(&mut self) {
        self.limiter.release(self.amount);
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn run_tasks(
    tasks: Vec<WorkItem>,
    threads: usize,
    task_time: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
) -> Result<()> {
    if tasks.is_empty() {
        return Ok(());
    }
    let (tx, rx) = std::sync::mpsc::channel::<WorkItem>();
    for task in tasks {
        tx.send(task).map_err(|e| anyhow!("task queue error: {e}"))?;
    }
    drop(tx);
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    let err: std::sync::Arc<std::sync::Mutex<Option<anyhow::Error>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let rx = std::sync::Arc::clone(&rx);
        let err = std::sync::Arc::clone(&err);
        let task_time = task_time.clone();
        handles.push(std::thread::spawn(move || loop {
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
            let start = std::time::Instant::now();
            let result = (task.run)();
            let elapsed = start.elapsed().as_nanos() as u64;
            if let Some(task_time) = &task_time {
                task_time.fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(bucket) = task.bucket {
                bucket.fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);
            }
            if let Err(e) = result {
                *err.lock().unwrap() = Some(e);
                break;
            }
        }));
    }
    for handle in handles {
        let _ = handle.join();
    }
    let mut err = err.lock().unwrap();
    if let Some(e) = err.take() {
        Err(e)
    } else {
        Ok(())
    }
}

fn key_type_for_column(col: &ColumnSpec) -> Result<KeyType> {
    match col.ingested_dtype {
        ColumnType::Int64
        | ColumnType::Int32
        | ColumnType::Int16
        | ColumnType::Int8
        | ColumnType::UInt64
        | ColumnType::UInt32
        | ColumnType::UInt16
        | ColumnType::UInt8 => Ok(KeyType::Int64),
        ColumnType::Float32
        | ColumnType::Float64
        | ColumnType::Decimal128
        | ColumnType::Decimal256 => Ok(KeyType::Int64),
        ColumnType::String | ColumnType::Bytes => {
            if col.key_mode == Some(KeyMode::Uuid) {
                return Ok(KeyType::Uuid);
            }
            if col.key_mode == Some(KeyMode::Fixed) || col.fixed_len.is_some() {
                let len = col.fixed_len.ok_or_else(|| anyhow!("fixed_len required"))?;
                return Ok(KeyType::FixedBytes { len });
            }
            Ok(KeyType::VarBytes)
        }
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

fn key_from_value_ref(value: &ValueRef<'_>, col: &ColumnSpec) -> KeyParseResult<Key> {
    let key_type = key_type_for_column(col).map_err(|_| KeyParseError::UnsupportedType)?;
    match key_type {
        KeyType::Int64 => match value {
            ValueRef::Int64(v) => {
                if *v < 0 {
                    Err(KeyParseError::Negative)
                } else {
                    Ok(Key::Int(*v as u64))
                }
            }
            ValueRef::Int32(v) => {
                if *v < 0 {
                    Err(KeyParseError::Negative)
                } else {
                    Ok(Key::Int(*v as u64))
                }
            }
            ValueRef::UInt64(v) => Ok(Key::Int(*v)),
            ValueRef::UInt32(v) => Ok(Key::Int(*v as u64)),
            ValueRef::String(s) => {
                if col.key_mode == Some(KeyMode::Uuid)
                    || col.key_mode == Some(KeyMode::Fixed)
                    || col.fixed_len.is_some()
                {
                    return Err(KeyParseError::TypeMismatch);
                }
                parse_base10_bytes(s.as_bytes()).map(Key::Int)
            }
            ValueRef::Bytes(b) => parse_base10_bytes(b).map(Key::Int),
            _ => Err(KeyParseError::TypeMismatch),
        },
        KeyType::FixedBytes { len } => match value {
            ValueRef::Bytes(b) => {
                if col.key_mode == Some(KeyMode::Uuid) {
                    return Err(KeyParseError::TypeMismatch);
                }
                if b.len() != len {
                    return Err(KeyParseError::LengthMismatch);
                }
                Ok(Key::FixedBytes(b.to_vec()))
            }
            ValueRef::String(s) => {
                if col.key_mode == Some(KeyMode::Uuid) {
                    return Err(KeyParseError::TypeMismatch);
                }
                let bytes = normalize_numeric_bytes(s.as_bytes(), col)?;
                if bytes.len() != len {
                    return Err(KeyParseError::LengthMismatch);
                }
                Ok(Key::FixedBytes(bytes))
            }
            _ => Err(KeyParseError::TypeMismatch),
        },
        KeyType::VarBytes => match value {
            ValueRef::Bytes(b) => {
                if col.key_mode == Some(KeyMode::Uuid)
                    || col.key_mode == Some(KeyMode::Fixed)
                    || col.fixed_len.is_some()
                {
                    return Err(KeyParseError::TypeMismatch);
                }
                if b.len() > MAX_VAR_KEY_LEN {
                    return Err(KeyParseError::TooLarge);
                }
                Ok(Key::Bytes(b.to_vec()))
            }
            ValueRef::String(s) => {
                if col.key_mode == Some(KeyMode::Uuid)
                    || col.key_mode == Some(KeyMode::Fixed)
                    || col.fixed_len.is_some()
                {
                    return Err(KeyParseError::TypeMismatch);
                }
                let bytes = normalize_numeric_bytes(s.as_bytes(), col)?;
                if bytes.len() > MAX_VAR_KEY_LEN {
                    return Err(KeyParseError::TooLarge);
                }
                Ok(Key::Bytes(bytes))
            }
            ValueRef::Int64(v) => {
                if *v < 0 {
                    return Err(KeyParseError::Negative);
                }
                let bytes = v.to_string().into_bytes();
                if bytes.len() > MAX_VAR_KEY_LEN {
                    return Err(KeyParseError::TooLarge);
                }
                Ok(Key::Bytes(bytes))
            }
            ValueRef::Int32(v) => {
                if *v < 0 {
                    return Err(KeyParseError::Negative);
                }
                let bytes = v.to_string().into_bytes();
                if bytes.len() > MAX_VAR_KEY_LEN {
                    return Err(KeyParseError::TooLarge);
                }
                Ok(Key::Bytes(bytes))
            }
            ValueRef::UInt64(v) => {
                let bytes = v.to_string().into_bytes();
                if bytes.len() > MAX_VAR_KEY_LEN {
                    return Err(KeyParseError::TooLarge);
                }
                Ok(Key::Bytes(bytes))
            }
            ValueRef::UInt32(v) => {
                let bytes = v.to_string().into_bytes();
                if bytes.len() > MAX_VAR_KEY_LEN {
                    return Err(KeyParseError::TooLarge);
                }
                Ok(Key::Bytes(bytes))
            }
            _ => Err(KeyParseError::TypeMismatch),
        },
        KeyType::Uuid => match value {
            ValueRef::String(s) => {
                if col.key_mode != Some(KeyMode::Uuid) {
                    return Err(KeyParseError::TypeMismatch);
                }
                let uuid = parse_uuid_string(s.as_bytes())?;
                Ok(Key::Uuid(uuid))
            }
            ValueRef::Bytes(b) => {
                if col.key_mode != Some(KeyMode::Uuid) {
                    return Err(KeyParseError::TypeMismatch);
                }
                if b.len() != 16 {
                    return Err(KeyParseError::LengthMismatch);
                }
                let mut buf = [0u8; 16];
                buf.copy_from_slice(b);
                Ok(Key::Uuid(buf))
            }
            _ => Err(KeyParseError::TypeMismatch),
        },
    }
}

fn key_hash_bytes<'a>(key: &'a Key, scratch: &'a mut [u8; 16]) -> &'a [u8] {
    match key {
        Key::Int(v) => {
            scratch[..8].copy_from_slice(&v.to_le_bytes());
            &scratch[..8]
        }
        Key::Uuid(v) => v.as_slice(),
        Key::FixedBytes(v) => v.as_slice(),
        Key::Bytes(v) => v.as_slice(),
    }
}

fn write_key_to_buf(buf: &mut Vec<u8>, key: &Key) -> Result<()> {
    match key {
        Key::Int(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Key::Uuid(v) => buf.extend_from_slice(v),
        Key::FixedBytes(v) => buf.extend_from_slice(v),
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
        KeyType::FixedBytes { len } => {
            if data.len() < len {
                return Err(anyhow!("temp key truncated"));
            }
            let bytes = data[..len].to_vec();
            *data = &data[len..];
            Ok(Key::FixedBytes(bytes))
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
    let mut input = bytes;
    if starts_with_case_insensitive(input, b"urn:uuid:") {
        input = &input[9..];
    }
    if input.len() >= 2 {
        let first = input[0];
        let last = input[input.len() - 1];
        if (first == b'{' && last == b'}') || (first == b'(' && last == b')') {
            input = &input[1..input.len() - 1];
        }
    }
    let mut out = [0u8; 16];
    let mut idx = 0usize;
    let mut high = true;
    for &b in input {
        if b == b'-' {
            continue;
        }
        let val = hex_value(b).ok_or(KeyParseError::InvalidUuid)?;
        if idx >= 16 {
            return Err(KeyParseError::InvalidUuid);
        }
        if high {
            out[idx] = val << 4;
            high = false;
        } else {
            out[idx] |= val;
            idx += 1;
            high = true;
        }
    }
    if idx != 16 || !high {
        return Err(KeyParseError::InvalidUuid);
    }
    Ok(out)
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn starts_with_case_insensitive(haystack: &[u8], needle: &[u8]) -> bool {
    if haystack.len() < needle.len() {
        return false;
    }
    haystack
        .iter()
        .zip(needle.iter())
        .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
}

fn normalize_numeric_bytes(bytes: &[u8], target_col: &ColumnSpec) -> KeyParseResult<Vec<u8>> {
    if matches!(
        target_col.ingested_dtype,
        ColumnType::Float32
            | ColumnType::Float64
            | ColumnType::Decimal128
            | ColumnType::Decimal256
    ) {
        return normalize_base10_bytes(bytes);
    }
    Ok(bytes.to_vec())
}

fn normalize_base10_bytes(bytes: &[u8]) -> KeyParseResult<Vec<u8>> {
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
    let frac_out = frac_part
        .unwrap_or("")
        .trim_end_matches('0');
    if frac_out.is_empty() {
        return Ok(int_out.as_bytes().to_vec());
    }
    let mut out = Vec::with_capacity(int_out.len() + 1 + frac_out.len());
    out.extend_from_slice(int_out.as_bytes());
    out.push(b'.');
    out.extend_from_slice(frac_out.as_bytes());
    Ok(out)
}

fn sample_key(key: &Key) -> Option<String> {
    match key {
        Key::Int(v) => Some(v.to_string()),
        Key::FixedBytes(bytes) | Key::Bytes(bytes) => Some(format_bytes_sample(bytes)),
        Key::Uuid(bytes) => Some(hex_encode(bytes)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BinaryArray, Int64Array, StringArray};

    #[test]
    fn parse_uuid_string_formats() {
        let raw = "00112233445566778899aabbccddeeff";
        let dashed = "00112233-4455-6677-8899-aabbccddeeff";
        let upper = "00112233-4455-6677-8899-AABBCCDDEEFF";
        let urn = "urn:uuid:00112233-4455-6677-8899-aabbccddeeff";
        let braced = "{00112233-4455-6677-8899-aabbccddeeff}";
        let expected = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff,
        ];
        assert_eq!(parse_uuid_string(raw.as_bytes()).unwrap(), expected);
        assert_eq!(parse_uuid_string(dashed.as_bytes()).unwrap(), expected);
        assert_eq!(parse_uuid_string(upper.as_bytes()).unwrap(), expected);
        assert_eq!(parse_uuid_string(urn.as_bytes()).unwrap(), expected);
        assert_eq!(parse_uuid_string(braced.as_bytes()).unwrap(), expected);
    }

    #[test]
    fn parse_uuid_bytes_requires_len_16() {
        assert!(parse_uuid_bytes(&[0u8; 15]).is_err());
        assert!(parse_uuid_bytes(&[0u8; 17]).is_err());
        assert!(parse_uuid_bytes(&[0u8; 16]).is_ok());
    }

    #[test]
    fn convert_string_timestamp_to_rowpack_timestamp() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_at".to_string(),
                source_dtype: ColumnType::String,
                ingested_dtype: ColumnType::Timestamp,
                fixed_len: None,
                timestamp_format: Some("rfc3339".to_string()),
                timestamp_unit: None,
                key_mode: None,
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
        let row = RowView {
            row: 0,
            cols: &cols,
        };
        let mut values = Vec::new();
        let mut arrays_f32 = Vec::new();
        let mut arrays_f64 = Vec::new();
        let mut strings = Vec::new();
        let mut bytes = Vec::new();
        row_values_from_row(
            &row,
            &table,
            &mut values,
            &mut arrays_f32,
            &mut arrays_f64,
            &mut strings,
            &mut bytes,
            &std::sync::Arc::new(AtomicU64::new(0)),
        )
        .unwrap();
        match values[0] {
            ValueRef::Timestamp { secs, nanos } => {
                assert_eq!(nanos, 0);
                assert_eq!(secs, 1_577_934_245);
            }
            _ => panic!("expected timestamp"),
        }
    }

    #[test]
    fn convert_int_timestamp_with_unit() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_at".to_string(),
                source_dtype: ColumnType::Int64,
                ingested_dtype: ColumnType::Timestamp,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: Some(TimestampUnit::Millis),
                key_mode: None,
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
        let row = RowView {
            row: 0,
            cols: &cols,
        };
        let mut values = Vec::new();
        let mut arrays_f32 = Vec::new();
        let mut arrays_f64 = Vec::new();
        let mut strings = Vec::new();
        let mut bytes = Vec::new();
        row_values_from_row(
            &row,
            &table,
            &mut values,
            &mut arrays_f32,
            &mut arrays_f64,
            &mut strings,
            &mut bytes,
            &std::sync::Arc::new(AtomicU64::new(0)),
        )
        .unwrap();
        match values[0] {
            ValueRef::Timestamp { secs, nanos } => {
                assert_eq!(secs, 1500);
                assert_eq!(nanos, 0);
            }
            _ => panic!("expected timestamp"),
        }
    }

    #[test]
    fn convert_string_date_to_rowpack_date() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_date".to_string(),
                source_dtype: ColumnType::String,
                ingested_dtype: ColumnType::Date,
                fixed_len: None,
                timestamp_format: Some("%Y-%m-%d".to_string()),
                timestamp_unit: None,
                key_mode: None,
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
        let row = RowView {
            row: 0,
            cols: &cols,
        };
        let mut values = Vec::new();
        let mut arrays_f32 = Vec::new();
        let mut arrays_f64 = Vec::new();
        let mut strings = Vec::new();
        let mut bytes = Vec::new();
        row_values_from_row(
            &row,
            &table,
            &mut values,
            &mut arrays_f32,
            &mut arrays_f64,
            &mut strings,
            &mut bytes,
            &std::sync::Arc::new(AtomicU64::new(0)),
        )
        .unwrap();
        match values[0] {
            ValueRef::Date(days) => {
                let expected = days_from_secs(
                    NaiveDate::from_ymd_opt(2024, 2, 3)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc()
                        .timestamp(),
                );
                assert_eq!(days, expected);
            }
            _ => panic!("expected date"),
        }
    }

    #[test]
    fn convert_numeric_to_string_for_rowpack() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                source_dtype: ColumnType::Int64,
                ingested_dtype: ColumnType::String,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                key_mode: None,
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
        let row = RowView {
            row: 0,
            cols: &cols,
        };
        let mut values = Vec::new();
        let mut arrays_f32 = Vec::new();
        let mut arrays_f64 = Vec::new();
        let mut strings = Vec::new();
        let mut bytes = Vec::new();
        row_values_from_row(
            &row,
            &table,
            &mut values,
            &mut arrays_f32,
            &mut arrays_f64,
            &mut strings,
            &mut bytes,
            &std::sync::Arc::new(AtomicU64::new(0)),
        )
        .unwrap();
        match values[0] {
            ValueRef::String(value) => assert_eq!(value, "42"),
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn convert_bytes_to_int_for_rowpack() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                source_dtype: ColumnType::Bytes,
                ingested_dtype: ColumnType::Int32,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                key_mode: None,
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
        let row = RowView {
            row: 0,
            cols: &cols,
        };
        let mut values = Vec::new();
        let mut arrays_f32 = Vec::new();
        let mut arrays_f64 = Vec::new();
        let mut strings = Vec::new();
        let mut bytes = Vec::new();
        row_values_from_row(
            &row,
            &table,
            &mut values,
            &mut arrays_f32,
            &mut arrays_f64,
            &mut strings,
            &mut bytes,
            &std::sync::Arc::new(AtomicU64::new(0)),
        )
        .unwrap();
        match values[0] {
            ValueRef::Int32(value) => assert_eq!(value, 123),
            _ => panic!("expected int32"),
        }
    }

 
}

fn table_schema(table: &TableSpec) -> Result<Schema> {
    let mut fields = Vec::with_capacity(table.columns.len());
    for col in &table.columns {
        let dtype = match col.source_dtype {
            ColumnType::Int8 => DataType::Int8,
            ColumnType::Int16 => DataType::Int16,
            ColumnType::Int64 => DataType::Int64,
            ColumnType::Int32 => DataType::Int32,
            ColumnType::UInt8 => DataType::UInt8,
            ColumnType::UInt16 => DataType::UInt16,
            ColumnType::UInt64 => DataType::UInt64,
            ColumnType::UInt32 => DataType::UInt32,
            ColumnType::Float64 => DataType::Float64,
            ColumnType::Float32 => DataType::Float32,
            ColumnType::Bool => DataType::Boolean,
            ColumnType::String => DataType::Utf8,
            ColumnType::Bytes => DataType::Binary,
            ColumnType::Date => DataType::Date32,
            ColumnType::Timestamp => DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            ColumnType::Decimal128 => {
                let precision = col
                    .precision
                    .ok_or_else(|| anyhow!("decimal128 column {} missing precision", col.name))?;
                let scale = col
                    .scale
                    .ok_or_else(|| anyhow!("decimal128 column {} missing scale", col.name))?;
                let precision = u8::try_from(precision)
                    .map_err(|_| anyhow!("decimal128 column {} precision out of range", col.name))?;
                let scale =
                    i8::try_from(scale).map_err(|_| anyhow!("decimal128 column {} scale out of range", col.name))?;
                DataType::Decimal128(precision, scale)
            }
            ColumnType::Decimal256 => {
                let precision = col
                    .precision
                    .ok_or_else(|| anyhow!("decimal256 column {} missing precision", col.name))?;
                let scale = col
                    .scale
                    .ok_or_else(|| anyhow!("decimal256 column {} missing scale", col.name))?;
                let precision = u8::try_from(precision)
                    .map_err(|_| anyhow!("decimal256 column {} precision out of range", col.name))?;
                let scale =
                    i8::try_from(scale).map_err(|_| anyhow!("decimal256 column {} scale out of range", col.name))?;
                DataType::Decimal256(precision, scale)
            }
            ColumnType::Float32Array => DataType::List(Arc::new(Field::new(
                "item",
                DataType::Float32,
                true,
            ))),
            ColumnType::Float64Array => DataType::List(Arc::new(Field::new(
                "item",
                DataType::Float64,
                true,
            ))),
        };
        fields.push(Field::new(&col.name, dtype, true));
    }
    Ok(Schema::new(fields))
}

fn rowpack_config_for_table(
    source_cfg: Option<&RowPackConfig>,
    table: &TableSpec,
) -> RowPackConfig {
    RowPackConfig::merge(source_cfg, table.rowpack.as_ref())
}

fn rowpack_columns(table: &TableSpec) -> Result<Vec<diskgraph_rowpack::RowPackColumn>> {
    let mut columns = Vec::with_capacity(table.columns.len());
    for col in &table.columns {
        let dtype = match col.ingested_dtype {
            ColumnType::Int8 => RowpackType::Int8,
            ColumnType::Int16 => RowpackType::Int16,
            ColumnType::Int64 => RowpackType::Int64,
            ColumnType::Int32 => RowpackType::Int32,
            ColumnType::UInt8 => RowpackType::UInt8,
            ColumnType::UInt16 => RowpackType::UInt16,
            ColumnType::UInt64 => RowpackType::UInt64,
            ColumnType::UInt32 => RowpackType::UInt32,
            ColumnType::Float64 => RowpackType::Float64,
            ColumnType::Float32 => RowpackType::Float32,
            ColumnType::Bool => RowpackType::Bool,
            ColumnType::String => RowpackType::String,
            ColumnType::Bytes => RowpackType::Bytes,
            ColumnType::Date => RowpackType::Date,
            ColumnType::Timestamp => RowpackType::Timestamp,
            ColumnType::Decimal128 => RowpackType::Decimal128,
            ColumnType::Decimal256 => RowpackType::Decimal256,
            ColumnType::Float32Array => RowpackType::Float32Array,
            ColumnType::Float64Array => RowpackType::Float64Array,
        };
        columns.push(diskgraph_rowpack::RowPackColumn {
            name: col.name.clone(),
            logical_type: dtype,
            fixed_len: col.fixed_len,
            precision: col.precision,
            scale: col.scale,
        });
    }
    Ok(columns)
}

fn rowpack_columns_for_specs(
    cols: &[ColumnSpec],
) -> Result<Vec<diskgraph_rowpack::RowPackColumn>> {
    let mut columns = Vec::with_capacity(cols.len());
    for col in cols {
        let dtype = match col.ingested_dtype {
            ColumnType::Int8 => RowpackType::Int8,
            ColumnType::Int16 => RowpackType::Int16,
            ColumnType::Int64 => RowpackType::Int64,
            ColumnType::Int32 => RowpackType::Int32,
            ColumnType::UInt8 => RowpackType::UInt8,
            ColumnType::UInt16 => RowpackType::UInt16,
            ColumnType::UInt64 => RowpackType::UInt64,
            ColumnType::UInt32 => RowpackType::UInt32,
            ColumnType::Float64 => RowpackType::Float64,
            ColumnType::Float32 => RowpackType::Float32,
            ColumnType::Bool => RowpackType::Bool,
            ColumnType::String => RowpackType::String,
            ColumnType::Bytes => RowpackType::Bytes,
            ColumnType::Date => RowpackType::Date,
            ColumnType::Timestamp => RowpackType::Timestamp,
            ColumnType::Decimal128 => RowpackType::Decimal128,
            ColumnType::Decimal256 => RowpackType::Decimal256,
            ColumnType::Float32Array => RowpackType::Float32Array,
            ColumnType::Float64Array => RowpackType::Float64Array,
        };
        columns.push(diskgraph_rowpack::RowPackColumn {
            name: col.name.clone(),
            logical_type: dtype,
            fixed_len: col.fixed_len,
            precision: col.precision,
            scale: col.scale,
        });
    }
    Ok(columns)
}

fn sample_indices(total: usize, target: usize) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    if total <= target {
        return (0..total).collect();
    }
    let stride = (total as f64 / target as f64).ceil() as usize;
    (0..total).step_by(stride).take(target).collect()
}

fn shard_for_key(bytes: &[u8], num_shards: usize) -> usize {
    if num_shards <= 1 {
        return 0;
    }
    let mut hasher = XxHash64::with_seed(0);
    hasher.write(bytes);
    (hasher.finish() as usize) % num_shards
}

fn estimate_pk_map_bytes(col: &ColumnSpec, rows: usize) -> usize {
    let key_type = key_type_for_column(col).unwrap_or(KeyType::VarBytes);
    let key_len = match key_type {
        KeyType::Int64 => 8,
        KeyType::FixedBytes { len } => len,
        KeyType::Uuid => 16,
        KeyType::VarBytes => col.fixed_len.unwrap_or(32),
    };
    rows.saturating_mul(key_len + 16)
}

struct RowView<'a> {
    row: usize,
    cols: &'a [ArrayRef],
}

struct RowAccess<'a> {
    row: usize,
    cols: &'a [ArrayRef],
    table: &'a TableSpec,
    pools: RowValuePools,
    trimmed_strings: &'a Arc<AtomicU64>,
}

impl<'a> RowAccess<'a> {
    fn new(
        table: &'a TableSpec,
        cols: &'a [ArrayRef],
        trimmed_strings: &'a Arc<AtomicU64>,
    ) -> Self {
        Self {
            row: 0,
            cols,
            table,
            pools: RowValuePools::new(),
            trimmed_strings,
        }
    }

    fn reset(&mut self, row: usize) {
        self.row = row;
        self.pools.reset();
    }

    fn pool_str(&mut self, value: String) -> &'a str {
        let slice = self.pools.take_string_owned(value);
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    }

    fn pool_bytes(&mut self, value: Vec<u8>) -> &'a [u8] {
        let slice = self.pools.take_bytes_owned(value);
        unsafe { std::mem::transmute::<&[u8], &'a [u8]>(slice) }
    }

    fn value_ref(&mut self, idx: usize) -> Result<ValueRef<'a>> {
        let col = self
            .table
            .columns
            .get(idx)
            .ok_or_else(|| anyhow!("missing column {}", idx))?;
        let array = self
            .cols
            .get(idx)
            .ok_or_else(|| anyhow!("missing column {}", col.name))?;
        if array.is_null(self.row) {
            return Ok(ValueRef::Null);
        }
        match (col.source_dtype.clone(), col.ingested_dtype.clone()) {
            (source, ingested) if source == ingested => match source {
                ColumnType::Int64 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .ok_or_else(|| anyhow!("int64 cast"))?;
                    Ok(ValueRef::Int64(arr.value(self.row)))
                }
                ColumnType::Int8 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int8Array>()
                        .ok_or_else(|| anyhow!("int8 cast"))?;
                    Ok(ValueRef::Int32(arr.value(self.row) as i32))
                }
                ColumnType::Int16 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int16Array>()
                        .ok_or_else(|| anyhow!("int16 cast"))?;
                    Ok(ValueRef::Int32(arr.value(self.row) as i32))
                }
                ColumnType::Int32 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int32Array>()
                        .ok_or_else(|| anyhow!("int32 cast"))?;
                    Ok(ValueRef::Int32(arr.value(self.row)))
                }
                ColumnType::UInt64 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt64Array>()
                        .ok_or_else(|| anyhow!("uint64 cast"))?;
                    Ok(ValueRef::UInt64(arr.value(self.row)))
                }
                ColumnType::UInt8 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt8Array>()
                        .ok_or_else(|| anyhow!("uint8 cast"))?;
                    Ok(ValueRef::UInt32(arr.value(self.row) as u32))
                }
                ColumnType::UInt16 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt16Array>()
                        .ok_or_else(|| anyhow!("uint16 cast"))?;
                    Ok(ValueRef::UInt32(arr.value(self.row) as u32))
                }
                ColumnType::UInt32 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt32Array>()
                        .ok_or_else(|| anyhow!("uint32 cast"))?;
                    Ok(ValueRef::UInt32(arr.value(self.row)))
                }
                ColumnType::Float64 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Float64Array>()
                        .ok_or_else(|| anyhow!("float64 cast"))?;
                    Ok(ValueRef::Float64(arr.value(self.row)))
                }
                ColumnType::Float32 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                        .ok_or_else(|| anyhow!("float32 cast"))?;
                    Ok(ValueRef::Float32(arr.value(self.row)))
                }
                ColumnType::Bool => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::BooleanArray>()
                        .ok_or_else(|| anyhow!("bool cast"))?;
                    Ok(ValueRef::Bool(arr.value(self.row)))
                }
                ColumnType::String => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?;
                    let text = arr.value(self.row);
                    if text.len() > MAX_INPUT_STRING_BYTES {
                        let trimmed = trim_str_bytes(text, MAX_INPUT_STRING_BYTES).to_string();
                        self.trimmed_strings.fetch_add(1, Ordering::Relaxed);
                        Ok(ValueRef::String(self.pool_str(trimmed)))
                    } else {
                        Ok(ValueRef::String(text))
                    }
                }
                ColumnType::Bytes => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .ok_or_else(|| anyhow!("binary cast"))?;
                    let data = arr.value(self.row);
                    if data.len() > MAX_INPUT_STRING_BYTES {
                        self.trimmed_strings.fetch_add(1, Ordering::Relaxed);
                        Ok(ValueRef::Bytes(self.pool_bytes(
                            data[..MAX_INPUT_STRING_BYTES].to_vec(),
                        )))
                    } else {
                        Ok(ValueRef::Bytes(data))
                    }
                }
                ColumnType::Date => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Date32Array>()
                        .ok_or_else(|| anyhow!("date cast"))?;
                    Ok(ValueRef::Date(arr.value(self.row)))
                }
                ColumnType::Timestamp => {
                    let (secs, nanos) = timestamp_parts_from_array(array, self.row)?;
                    Ok(ValueRef::Timestamp { secs, nanos })
                }
                ColumnType::Decimal128 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Decimal128Array>()
                        .ok_or_else(|| anyhow!("decimal128 cast"))?;
                    Ok(ValueRef::Decimal128(arr.value(self.row)))
                }
                ColumnType::Decimal256 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Decimal256Array>()
                        .ok_or_else(|| anyhow!("decimal256 cast"))?;
                    Ok(ValueRef::Decimal256(arr.value(self.row).to_le_bytes()))
                }
                ColumnType::Float32Array => {
                    let list = array
                        .as_any()
                        .downcast_ref::<arrow_array::ListArray>()
                        .ok_or_else(|| anyhow!("list cast"))?;
                    let values_arr = list
                        .values()
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                        .ok_or_else(|| anyhow!("float32 list cast"))?;
                    let offsets = list.value_offsets();
                    let start = offsets[self.row] as usize;
                    let end = offsets[self.row + 1] as usize;
                    let vec = self.pools.take_f32_vec(end.saturating_sub(start));
                    for idx in start..end {
                        if values_arr.is_null(idx) {
                            vec.push(0.0);
                        } else {
                            vec.push(values_arr.value(idx));
                        }
                    }
                    Ok(ValueRef::Float32Array(unsafe {
                        std::mem::transmute::<&[f32], &'a [f32]>(vec.as_slice())
                    }))
                }
                ColumnType::Float64Array => {
                    let list = array
                        .as_any()
                        .downcast_ref::<arrow_array::ListArray>()
                        .ok_or_else(|| anyhow!("list cast"))?;
                    let values_arr = list
                        .values()
                        .as_any()
                        .downcast_ref::<arrow_array::Float64Array>()
                        .ok_or_else(|| anyhow!("float64 list cast"))?;
                    let offsets = list.value_offsets();
                    let start = offsets[self.row] as usize;
                    let end = offsets[self.row + 1] as usize;
                    let vec = self.pools.take_f64_vec(end.saturating_sub(start));
                    for idx in start..end {
                        if values_arr.is_null(idx) {
                            vec.push(0.0);
                        } else {
                            vec.push(values_arr.value(idx));
                        }
                    }
                    Ok(ValueRef::Float64Array(unsafe {
                        std::mem::transmute::<&[f64], &'a [f64]>(vec.as_slice())
                    }))
                }
            },
            (ColumnType::String, ColumnType::Timestamp)
            | (ColumnType::Bytes, ColumnType::Timestamp) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some((secs, nanos)) = parse_timestamp_text(text, col.timestamp_format.as_deref()) {
                    Ok(ValueRef::Timestamp { secs, nanos })
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::Date)
            | (ColumnType::Bytes, ColumnType::Date) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(days) = parse_date_text(text, col.timestamp_format.as_deref()) {
                    Ok(ValueRef::Date(days))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::Int8, ColumnType::Timestamp)
            | (ColumnType::Int16, ColumnType::Timestamp)
            | (ColumnType::Int32, ColumnType::Timestamp)
            | (ColumnType::Int64, ColumnType::Timestamp)
            | (ColumnType::UInt8, ColumnType::Timestamp)
            | (ColumnType::UInt16, ColumnType::Timestamp)
            | (ColumnType::UInt32, ColumnType::Timestamp)
            | (ColumnType::UInt64, ColumnType::Timestamp) => {
                let value = read_i64(array, self.row, &col.source_dtype)?;
                let unit = col
                    .timestamp_unit
                    .clone()
                    .unwrap_or(TimestampUnit::Seconds);
                let (secs, nanos) = timestamp_from_int(value, unit);
                Ok(ValueRef::Timestamp { secs, nanos })
            }
            (ColumnType::Int8, ColumnType::Date)
            | (ColumnType::Int16, ColumnType::Date)
            | (ColumnType::Int32, ColumnType::Date)
            | (ColumnType::Int64, ColumnType::Date)
            | (ColumnType::UInt8, ColumnType::Date)
            | (ColumnType::UInt16, ColumnType::Date)
            | (ColumnType::UInt32, ColumnType::Date)
            | (ColumnType::UInt64, ColumnType::Date) => {
                let value = read_i64(array, self.row, &col.source_dtype)?;
                let days = if let Some(unit) = col.timestamp_unit.as_ref() {
                    let (secs, _) = timestamp_from_int(value, unit.clone());
                    days_from_secs(secs)
                } else {
                    value as i32
                };
                Ok(ValueRef::Date(days))
            }
            (ColumnType::String, ColumnType::Int64)
            | (ColumnType::Bytes, ColumnType::Int64) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_i64(text) {
                    Ok(ValueRef::Int64(v))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::Int32)
            | (ColumnType::Bytes, ColumnType::Int32) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_i32(text) {
                    Ok(ValueRef::Int32(v))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::UInt64)
            | (ColumnType::Bytes, ColumnType::UInt64) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_u64(text) {
                    Ok(ValueRef::UInt64(v))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::UInt32)
            | (ColumnType::Bytes, ColumnType::UInt32) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_u32(text) {
                    Ok(ValueRef::UInt32(v))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::Float64)
            | (ColumnType::Bytes, ColumnType::Float64) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_f64(text) {
                    Ok(ValueRef::Float64(v))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::Float32)
            | (ColumnType::Bytes, ColumnType::Float32) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_f32(text) {
                    Ok(ValueRef::Float32(v))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::Bool)
            | (ColumnType::Bytes, ColumnType::Bool) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_bool(text) {
                    Ok(ValueRef::Bool(v))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::Decimal128)
            | (ColumnType::Bytes, ColumnType::Decimal128) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_decimal_i128(text, col.scale.unwrap_or(0)) {
                    Ok(ValueRef::Decimal128(v))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::String, ColumnType::Decimal256)
            | (ColumnType::Bytes, ColumnType::Decimal256) => {
                let text = read_utf8(array, self.row, &col.source_dtype)?;
                if let Some(v) = parse_decimal_i256(text, col.scale.unwrap_or(0)) {
                    Ok(ValueRef::Decimal256(v.to_le_bytes()))
                } else {
                    Ok(ValueRef::Null)
                }
            }
            (ColumnType::Int64, ColumnType::String)
            | (ColumnType::Int32, ColumnType::String)
            | (ColumnType::Int16, ColumnType::String)
            | (ColumnType::Int8, ColumnType::String)
            | (ColumnType::UInt64, ColumnType::String)
            | (ColumnType::UInt32, ColumnType::String)
            | (ColumnType::UInt16, ColumnType::String)
            | (ColumnType::UInt8, ColumnType::String)
            | (ColumnType::Float64, ColumnType::String)
            | (ColumnType::Float32, ColumnType::String)
            | (ColumnType::Bool, ColumnType::String)
            | (ColumnType::Date, ColumnType::String)
            | (ColumnType::Timestamp, ColumnType::String)
            | (ColumnType::Decimal128, ColumnType::String)
            | (ColumnType::Decimal256, ColumnType::String) => {
                let value = value_from_array(array, self.row)?;
                let text = match value {
                    JsonValue::Null => return Ok(ValueRef::Null),
                    JsonValue::String(s) => s,
                    JsonValue::Number(n) => n.to_string(),
                    JsonValue::Bool(v) => v.to_string(),
                    _ => return Ok(ValueRef::Null),
                };
                Ok(ValueRef::String(self.pool_str(text)))
            }
            (ColumnType::String, ColumnType::Bytes) | (ColumnType::Bytes, ColumnType::Bytes) => {
                let data = read_bytes(array, self.row, &col.source_dtype)?;
                if data.len() > MAX_INPUT_STRING_BYTES {
                    self.trimmed_strings.fetch_add(1, Ordering::Relaxed);
                    Ok(ValueRef::Bytes(self.pool_bytes(
                        data[..MAX_INPUT_STRING_BYTES].to_vec(),
                    )))
                } else {
                    Ok(ValueRef::Bytes(data))
                }
            }
            _ => Ok(ValueRef::Null),
        }
    }
}

impl<'a> diskgraph_rowpack::RowValueAccess for RowAccess<'a> {
    fn get_int64(&mut self, idx: usize) -> Result<Option<i64>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Int64(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack int64 mismatch")),
        }
    }

    fn get_int32(&mut self, idx: usize) -> Result<Option<i32>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Int32(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack int32 mismatch")),
        }
    }

    fn get_uint64(&mut self, idx: usize) -> Result<Option<u64>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::UInt64(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack uint64 mismatch")),
        }
    }

    fn get_uint32(&mut self, idx: usize) -> Result<Option<u32>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::UInt32(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack uint32 mismatch")),
        }
    }

    fn get_float64(&mut self, idx: usize) -> Result<Option<f64>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Float64(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack float64 mismatch")),
        }
    }

    fn get_float32(&mut self, idx: usize) -> Result<Option<f32>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Float32(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack float32 mismatch")),
        }
    }

    fn get_bool(&mut self, idx: usize) -> Result<Option<bool>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Bool(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack bool mismatch")),
        }
    }

    fn get_string(&mut self, idx: usize) -> Result<Option<&str>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::String(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack string mismatch")),
        }
    }

    fn get_bytes(&mut self, idx: usize) -> Result<Option<&[u8]>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Bytes(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack bytes mismatch")),
        }
    }

    fn get_date(&mut self, idx: usize) -> Result<Option<i32>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Date(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack date mismatch")),
        }
    }

    fn get_timestamp(&mut self, idx: usize) -> Result<Option<(i64, u32)>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Timestamp { secs, nanos } => Ok(Some((secs, nanos))),
            _ => Err(anyhow!("rowpack timestamp mismatch")),
        }
    }

    fn get_decimal128(&mut self, idx: usize) -> Result<Option<i128>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Decimal128(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack decimal128 mismatch")),
        }
    }

    fn get_decimal256(&mut self, idx: usize) -> Result<Option<[u8; 32]>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Decimal256(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack decimal256 mismatch")),
        }
    }

    fn get_float32_array(&mut self, idx: usize) -> Result<Option<&[f32]>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Float32Array(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack float32 array mismatch")),
        }
    }

    fn get_float64_array(&mut self, idx: usize) -> Result<Option<&[f64]>> {
        match self.value_ref(idx)? {
            ValueRef::Null => Ok(None),
            ValueRef::Float64Array(v) => Ok(Some(v)),
            _ => Err(anyhow!("rowpack float64 array mismatch")),
        }
    }
}
struct RowValuePools {
    strings: Vec<String>,
    bytes: Vec<Vec<u8>>,
    arrays_f32: Vec<Vec<f32>>,
    arrays_f64: Vec<Vec<f64>>,
    strings_len: usize,
    bytes_len: usize,
    arrays_f32_len: usize,
    arrays_f64_len: usize,
}

impl RowValuePools {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            bytes: Vec::new(),
            arrays_f32: Vec::new(),
            arrays_f64: Vec::new(),
            strings_len: 0,
            bytes_len: 0,
            arrays_f32_len: 0,
            arrays_f64_len: 0,
        }
    }

    fn reset(&mut self) {
        self.strings_len = 0;
        self.bytes_len = 0;
        self.arrays_f32_len = 0;
        self.arrays_f64_len = 0;
    }

    fn take_string_owned(&mut self, value: String) -> &str {
        if self.strings_len < self.strings.len() {
            self.strings[self.strings_len] = value;
            let out = self.strings[self.strings_len].as_str();
            self.strings_len += 1;
            out
        } else {
            self.strings.push(value);
            self.strings_len += 1;
            self.strings.last().unwrap().as_str()
        }
    }

    fn take_bytes_owned(&mut self, value: Vec<u8>) -> &[u8] {
        if self.bytes_len < self.bytes.len() {
            self.bytes[self.bytes_len] = value;
            let out = self.bytes[self.bytes_len].as_slice();
            self.bytes_len += 1;
            out
        } else {
            self.bytes.push(value);
            self.bytes_len += 1;
            self.bytes.last().unwrap().as_slice()
        }
    }

    fn take_f32_vec(&mut self, capacity: usize) -> &mut Vec<f32> {
        if self.arrays_f32_len < self.arrays_f32.len() {
            let slot = &mut self.arrays_f32[self.arrays_f32_len];
            slot.clear();
            slot.reserve(capacity);
            self.arrays_f32_len += 1;
            slot
        } else {
            self.arrays_f32.push(Vec::with_capacity(capacity));
            self.arrays_f32_len += 1;
            self.arrays_f32.last_mut().unwrap()
        }
    }

    fn take_f64_vec(&mut self, capacity: usize) -> &mut Vec<f64> {
        if self.arrays_f64_len < self.arrays_f64.len() {
            let slot = &mut self.arrays_f64[self.arrays_f64_len];
            slot.clear();
            slot.reserve(capacity);
            self.arrays_f64_len += 1;
            slot
        } else {
            self.arrays_f64.push(Vec::with_capacity(capacity));
            self.arrays_f64_len += 1;
            self.arrays_f64.last_mut().unwrap()
        }
    }
}

 
fn row_values_from_row<'a>(
    row: &RowView<'a>,
    table: &TableSpec,
    values: &mut Vec<ValueRef<'a>>,
    arrays_f32: &mut Vec<Vec<f32>>,
    arrays_f64: &mut Vec<Vec<f64>>,
    strings: &mut Vec<String>,
    bytes: &mut Vec<Vec<u8>>,
    trimmed_strings: &std::sync::Arc<AtomicU64>,
) -> Result<()> {
    values.clear();
    arrays_f32.clear();
    arrays_f64.clear();
    strings.clear();
    bytes.clear();
    let reserve = table.columns.len();
    strings.reserve(reserve);
    bytes.reserve(reserve);
    arrays_f32.reserve(reserve);
    arrays_f64.reserve(reserve);
    for (idx, col) in table.columns.iter().enumerate() {
        let array = row
            .cols
            .get(idx)
            .ok_or_else(|| anyhow!("missing column {}", col.name))?;
        if array.is_null(row.row) {
            values.push(ValueRef::Null);
            continue;
        }
        match (col.source_dtype.clone(), col.ingested_dtype.clone()) {
            (source, ingested) if source == ingested => match source {
                ColumnType::Int64 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .ok_or_else(|| anyhow!("int64 cast"))?;
                    values.push(ValueRef::Int64(arr.value(row.row)));
                }
                ColumnType::Int8 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int8Array>()
                        .ok_or_else(|| anyhow!("int8 cast"))?;
                    values.push(ValueRef::Int32(arr.value(row.row) as i32));
                }
                ColumnType::Int16 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int16Array>()
                        .ok_or_else(|| anyhow!("int16 cast"))?;
                    values.push(ValueRef::Int32(arr.value(row.row) as i32));
                }
                ColumnType::Int32 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int32Array>()
                        .ok_or_else(|| anyhow!("int32 cast"))?;
                    values.push(ValueRef::Int32(arr.value(row.row)));
                }
                ColumnType::UInt64 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt64Array>()
                        .ok_or_else(|| anyhow!("uint64 cast"))?;
                    values.push(ValueRef::UInt64(arr.value(row.row)));
                }
                ColumnType::UInt8 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt8Array>()
                        .ok_or_else(|| anyhow!("uint8 cast"))?;
                    values.push(ValueRef::UInt32(arr.value(row.row) as u32));
                }
                ColumnType::UInt16 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt16Array>()
                        .ok_or_else(|| anyhow!("uint16 cast"))?;
                    values.push(ValueRef::UInt32(arr.value(row.row) as u32));
                }
                ColumnType::UInt32 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt32Array>()
                        .ok_or_else(|| anyhow!("uint32 cast"))?;
                    values.push(ValueRef::UInt32(arr.value(row.row)));
                }
                ColumnType::Float64 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Float64Array>()
                        .ok_or_else(|| anyhow!("float64 cast"))?;
                    values.push(ValueRef::Float64(arr.value(row.row)));
                }
                ColumnType::Float32 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                        .ok_or_else(|| anyhow!("float32 cast"))?;
                    values.push(ValueRef::Float32(arr.value(row.row)));
                }
                ColumnType::Bool => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::BooleanArray>()
                        .ok_or_else(|| anyhow!("bool cast"))?;
                    values.push(ValueRef::Bool(arr.value(row.row)));
                }
                ColumnType::String => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?;
                    let text = arr.value(row.row);
                    if text.len() > MAX_INPUT_STRING_BYTES {
                        let trimmed = trim_str_bytes(text, MAX_INPUT_STRING_BYTES).to_string();
                        trimmed_strings.fetch_add(1, Ordering::Relaxed);
                        strings.push(trimmed);
                        let slice_ptr = strings.last().unwrap().as_str() as *const str;
                        let slice = unsafe { &*slice_ptr };
                        values.push(ValueRef::String(slice));
                    } else {
                        values.push(ValueRef::String(text));
                    }
                }
                ColumnType::Bytes => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .ok_or_else(|| anyhow!("binary cast"))?;
                    let data = arr.value(row.row);
                    if data.len() > MAX_INPUT_STRING_BYTES {
                        trimmed_strings.fetch_add(1, Ordering::Relaxed);
                        bytes.push(data[..MAX_INPUT_STRING_BYTES].to_vec());
                        let slice_ptr = bytes.last().unwrap().as_slice() as *const [u8];
                        let slice = unsafe { &*slice_ptr };
                        values.push(ValueRef::Bytes(slice));
                    } else {
                        values.push(ValueRef::Bytes(data));
                    }
                }
                ColumnType::Date => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Date32Array>()
                        .ok_or_else(|| anyhow!("date cast"))?;
                    values.push(ValueRef::Date(arr.value(row.row)));
                }
                ColumnType::Timestamp => {
                    let (secs, nanos) = timestamp_parts_from_array(array, row.row)?;
                    values.push(ValueRef::Timestamp { secs, nanos });
                }
                ColumnType::Decimal128 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Decimal128Array>()
                        .ok_or_else(|| anyhow!("decimal128 cast"))?;
                    values.push(ValueRef::Decimal128(arr.value(row.row)));
                }
                ColumnType::Decimal256 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::Decimal256Array>()
                        .ok_or_else(|| anyhow!("decimal256 cast"))?;
                    values.push(ValueRef::Decimal256(arr.value(row.row).to_le_bytes()));
                }
                ColumnType::Float32Array => {
                    let list = array
                        .as_any()
                        .downcast_ref::<arrow_array::ListArray>()
                        .ok_or_else(|| anyhow!("list cast"))?;
                    let values_arr = list
                        .values()
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                        .ok_or_else(|| anyhow!("float32 list cast"))?;
                    let offsets = list.value_offsets();
                    let start = offsets[row.row] as usize;
                    let end = offsets[row.row + 1] as usize;
                    let mut vec = Vec::with_capacity(end.saturating_sub(start));
                    for idx in start..end {
                        if values_arr.is_null(idx) {
                            vec.push(0.0);
                        } else {
                            vec.push(values_arr.value(idx));
                        }
                    }
                    arrays_f32.push(vec);
                    let slice_ptr = arrays_f32.last().unwrap().as_slice() as *const [f32];
                    let slice = unsafe { &*slice_ptr };
                    values.push(ValueRef::Float32Array(slice));
                }
                ColumnType::Float64Array => {
                    let list = array
                        .as_any()
                        .downcast_ref::<arrow_array::ListArray>()
                        .ok_or_else(|| anyhow!("list cast"))?;
                    let values_arr = list
                        .values()
                        .as_any()
                        .downcast_ref::<arrow_array::Float64Array>()
                        .ok_or_else(|| anyhow!("float64 list cast"))?;
                    let offsets = list.value_offsets();
                    let start = offsets[row.row] as usize;
                    let end = offsets[row.row + 1] as usize;
                    let mut vec = Vec::with_capacity(end.saturating_sub(start));
                    for idx in start..end {
                        if values_arr.is_null(idx) {
                            vec.push(0.0);
                        } else {
                            vec.push(values_arr.value(idx));
                        }
                    }
                    arrays_f64.push(vec);
                    let slice_ptr = arrays_f64.last().unwrap().as_slice() as *const [f64];
                    let slice = unsafe { &*slice_ptr };
                    values.push(ValueRef::Float64Array(slice));
                }
            },
            (ColumnType::String, ColumnType::Timestamp)
            | (ColumnType::Bytes, ColumnType::Timestamp) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some((secs, nanos)) = parse_timestamp_text(text, col.timestamp_format.as_deref()) {
                    values.push(ValueRef::Timestamp { secs, nanos });
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::Date)
            | (ColumnType::Bytes, ColumnType::Date) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(days) = parse_date_text(text, col.timestamp_format.as_deref()) {
                    values.push(ValueRef::Date(days));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::Int8, ColumnType::Timestamp)
            | (ColumnType::Int16, ColumnType::Timestamp)
            | (ColumnType::Int32, ColumnType::Timestamp)
            | (ColumnType::Int64, ColumnType::Timestamp)
            | (ColumnType::UInt8, ColumnType::Timestamp)
            | (ColumnType::UInt16, ColumnType::Timestamp)
            | (ColumnType::UInt32, ColumnType::Timestamp)
            | (ColumnType::UInt64, ColumnType::Timestamp) => {
                let value = read_i64(array, row.row, &col.source_dtype)?;
                let unit = col
                    .timestamp_unit
                    .clone()
                    .unwrap_or(TimestampUnit::Seconds);
                let (secs, nanos) = timestamp_from_int(value, unit);
                values.push(ValueRef::Timestamp { secs, nanos });
            }
            (ColumnType::Int8, ColumnType::Date)
            | (ColumnType::Int16, ColumnType::Date)
            | (ColumnType::Int32, ColumnType::Date)
            | (ColumnType::Int64, ColumnType::Date)
            | (ColumnType::UInt8, ColumnType::Date)
            | (ColumnType::UInt16, ColumnType::Date)
            | (ColumnType::UInt32, ColumnType::Date)
            | (ColumnType::UInt64, ColumnType::Date) => {
                let value = read_i64(array, row.row, &col.source_dtype)?;
                let days = if let Some(unit) = col.timestamp_unit.as_ref() {
                    let (secs, _) = timestamp_from_int(value, unit.clone());
                    days_from_secs(secs)
                } else {
                    value as i32
                };
                values.push(ValueRef::Date(days));
            }
            (ColumnType::String, ColumnType::Int64)
            | (ColumnType::Bytes, ColumnType::Int64) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_i64(text) {
                    values.push(ValueRef::Int64(v));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::Int32)
            | (ColumnType::Bytes, ColumnType::Int32) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_i32(text) {
                    values.push(ValueRef::Int32(v));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::UInt64)
            | (ColumnType::Bytes, ColumnType::UInt64) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_u64(text) {
                    values.push(ValueRef::UInt64(v));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::UInt32)
            | (ColumnType::Bytes, ColumnType::UInt32) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_u32(text) {
                    values.push(ValueRef::UInt32(v));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::Float64)
            | (ColumnType::Bytes, ColumnType::Float64) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_f64(text) {
                    values.push(ValueRef::Float64(v));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::Float32)
            | (ColumnType::Bytes, ColumnType::Float32) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_f32(text) {
                    values.push(ValueRef::Float32(v));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::Bool)
            | (ColumnType::Bytes, ColumnType::Bool) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_bool(text) {
                    values.push(ValueRef::Bool(v));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::Decimal128)
            | (ColumnType::Bytes, ColumnType::Decimal128) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_decimal_i128(text, col.scale.unwrap_or(0)) {
                    values.push(ValueRef::Decimal128(v));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::String, ColumnType::Decimal256)
            | (ColumnType::Bytes, ColumnType::Decimal256) => {
                let text = read_utf8(array, row.row, &col.source_dtype)?;
                if let Some(v) = parse_decimal_i256(text, col.scale.unwrap_or(0)) {
                    values.push(ValueRef::Decimal256(v.to_le_bytes()));
                } else {
                    values.push(ValueRef::Null);
                }
            }
            (ColumnType::Int64, ColumnType::String)
            | (ColumnType::Int32, ColumnType::String)
            | (ColumnType::Int16, ColumnType::String)
            | (ColumnType::Int8, ColumnType::String)
            | (ColumnType::UInt64, ColumnType::String)
            | (ColumnType::UInt32, ColumnType::String)
            | (ColumnType::UInt16, ColumnType::String)
            | (ColumnType::UInt8, ColumnType::String)
            | (ColumnType::Float64, ColumnType::String)
            | (ColumnType::Float32, ColumnType::String)
            | (ColumnType::Bool, ColumnType::String)
            | (ColumnType::Date, ColumnType::String)
            | (ColumnType::Timestamp, ColumnType::String)
            | (ColumnType::Decimal128, ColumnType::String)
            | (ColumnType::Decimal256, ColumnType::String) => {
                let value = value_from_array(array, row.row)?;
                let text = match value {
                    JsonValue::Null => {
                        values.push(ValueRef::Null);
                        continue;
                    }
                    JsonValue::String(s) => s,
                    JsonValue::Number(n) => n.to_string(),
                    JsonValue::Bool(v) => v.to_string(),
                    _ => {
                        values.push(ValueRef::Null);
                        continue;
                    }
                };
                strings.push(text);
                let slice_ptr = strings.last().unwrap().as_str() as *const str;
                let slice = unsafe { &*slice_ptr };
                values.push(ValueRef::String(slice));
            }
            (ColumnType::String, ColumnType::Bytes) | (ColumnType::Bytes, ColumnType::Bytes) => {
                let data = read_bytes(array, row.row, &col.source_dtype)?;
                if data.len() > MAX_INPUT_STRING_BYTES {
                    trimmed_strings.fetch_add(1, Ordering::Relaxed);
                    bytes.push(data[..MAX_INPUT_STRING_BYTES].to_vec());
                    let slice_ptr = bytes.last().unwrap().as_slice() as *const [u8];
                    let slice = unsafe { &*slice_ptr };
                    values.push(ValueRef::Bytes(slice));
                } else {
                    values.push(ValueRef::Bytes(data));
                }
            }
            _ => {
                values.push(ValueRef::Null);
            }
        }
    }
    Ok(())
}

fn owned_row_from_refs(
    values: &[ValueRef<'_>],
) -> OwnedRow {
    let mut owned = Vec::with_capacity(values.len());
    for value in values {
        let out = match value {
            ValueRef::Null => OwnedValue::Null,
            ValueRef::Int64(v) => OwnedValue::Int64(*v),
            ValueRef::Int32(v) => OwnedValue::Int32(*v),
            ValueRef::UInt64(v) => OwnedValue::UInt64(*v),
            ValueRef::UInt32(v) => OwnedValue::UInt32(*v),
            ValueRef::Float64(v) => OwnedValue::Float64(*v),
            ValueRef::Float32(v) => OwnedValue::Float32(*v),
            ValueRef::Bool(v) => OwnedValue::Bool(*v),
            ValueRef::String(v) => {
                OwnedValue::String((*v).to_string())
            }
            ValueRef::Bytes(v) => {
                OwnedValue::Bytes((*v).to_vec())
            }
            ValueRef::Date(v) => OwnedValue::Date(*v),
            ValueRef::Timestamp { secs, nanos } => OwnedValue::Timestamp {
                secs: *secs,
                nanos: *nanos,
            },
            ValueRef::Decimal128(v) => OwnedValue::Decimal128(*v),
            ValueRef::Decimal256(v) => OwnedValue::Decimal256(*v),
            ValueRef::Float32Array(v) => OwnedValue::Float32Array((*v).to_vec()),
            ValueRef::Float64Array(v) => OwnedValue::Float64Array((*v).to_vec()),
        };
        owned.push(out);
    }
    OwnedRow { values: owned }
}

fn owned_row_from_row<'a>(
    row: &RowView<'a>,
    table: &TableSpec,
    values: &mut Vec<ValueRef<'a>>,
    arrays_f32: &mut Vec<Vec<f32>>,
    arrays_f64: &mut Vec<Vec<f64>>,
    strings: &mut Vec<String>,
    bytes: &mut Vec<Vec<u8>>,
    trimmed_strings: &std::sync::Arc<AtomicU64>,
) -> Result<OwnedRow> {
    row_values_from_row(
        row,
        table,
        values,
        arrays_f32,
        arrays_f64,
        strings,
        bytes,
        trimmed_strings,
    )?;
    Ok(owned_row_from_refs(values))
}

fn read_utf8<'a>(array: &'a ArrayRef, row: usize, dtype: &ColumnType) -> Result<&'a str> {
    match dtype {
        ColumnType::String => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string cast"))?;
            Ok(arr.value(row))
        }
        ColumnType::Bytes => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| anyhow!("binary cast"))?;
            std::str::from_utf8(arr.value(row)).map_err(|_| anyhow!("invalid utf8"))
        }
        _ => Err(anyhow!("unsupported utf8 source")),
    }
}

fn read_bytes<'a>(array: &'a ArrayRef, row: usize, dtype: &ColumnType) -> Result<&'a [u8]> {
    match dtype {
        ColumnType::String => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string cast"))?;
            Ok(arr.value(row).as_bytes())
        }
        ColumnType::Bytes => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| anyhow!("binary cast"))?;
            Ok(arr.value(row))
        }
        _ => Err(anyhow!("unsupported bytes source")),
    }
}

fn read_i64(array: &ArrayRef, row: usize, dtype: &ColumnType) -> Result<i64> {
    match dtype {
        ColumnType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .ok_or_else(|| anyhow!("int64 cast"))?;
            Ok(arr.value(row))
        }
        ColumnType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .ok_or_else(|| anyhow!("int32 cast"))?;
            Ok(arr.value(row) as i64)
        }
        ColumnType::Int16 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int16Array>()
                .ok_or_else(|| anyhow!("int16 cast"))?;
            Ok(arr.value(row) as i64)
        }
        ColumnType::Int8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int8Array>()
                .ok_or_else(|| anyhow!("int8 cast"))?;
            Ok(arr.value(row) as i64)
        }
        ColumnType::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .ok_or_else(|| anyhow!("uint64 cast"))?;
            Ok(arr.value(row) as i64)
        }
        ColumnType::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .ok_or_else(|| anyhow!("uint32 cast"))?;
            Ok(arr.value(row) as i64)
        }
        ColumnType::UInt16 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt16Array>()
                .ok_or_else(|| anyhow!("uint16 cast"))?;
            Ok(arr.value(row) as i64)
        }
        ColumnType::UInt8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt8Array>()
                .ok_or_else(|| anyhow!("uint8 cast"))?;
            Ok(arr.value(row) as i64)
        }
        _ => Err(anyhow!("unsupported int source")),
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

fn parse_i64(value: &str) -> Option<i64> {
    value.trim().parse::<i64>().ok()
}

fn parse_i32(value: &str) -> Option<i32> {
    value.trim().parse::<i32>().ok()
}

fn parse_u64(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

fn parse_u32(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok()
}

fn parse_f64(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok()
}

fn parse_f32(value: &str) -> Option<f32> {
    value.trim().parse::<f32>().ok()
}

fn trim_str_bytes(value: &str, max: usize) -> &str {
    if value.len() <= max {
        return value;
    }
    let mut end = max;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn parse_timestamp_text(value: &str, format: Option<&str>) -> Option<(i64, u32)> {
    let ts = if let Some(fmt) = format {
        if fmt == "rfc3339" {
            chrono::DateTime::parse_from_rfc3339(value)
                .ok()
                .map(|v| v.naive_utc())
        } else {
            NaiveDateTime::parse_from_str(value, fmt).ok().or_else(|| {
                NaiveDate::parse_from_str(value, fmt)
                    .ok()
                    .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
            })
        }
    } else {
        chrono::DateTime::parse_from_rfc3339(value)
            .ok()
            .map(|v| v.naive_utc())
            .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").ok())
            .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f").ok())
            .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M").ok())
            .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S").ok())
            .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").ok())
            .or_else(|| NaiveDate::parse_from_str(value, "%Y-%m-%d").ok().map(|d| d.and_hms_opt(0, 0, 0).unwrap()))
    }?;
    let ts = ts.and_utc();
    Some((ts.timestamp(), ts.timestamp_subsec_nanos()))
}

fn parse_date_text(value: &str, format: Option<&str>) -> Option<i32> {
    let dt = if let Some(fmt) = format {
        NaiveDate::parse_from_str(value, fmt)
            .ok()
            .or_else(|| NaiveDateTime::parse_from_str(value, fmt).ok().map(|v| v.date()))
    } else {
        NaiveDate::parse_from_str(value, "%Y-%m-%d")
            .ok()
            .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").ok().map(|v| v.date()))
            .or_else(|| chrono::DateTime::parse_from_rfc3339(value).ok().map(|v| v.date_naive()))
    }?;
    let ts = dt.and_hms_opt(0, 0, 0)?.and_utc().timestamp();
    Some(days_from_secs(ts))
}

fn timestamp_from_int(value: i64, unit: TimestampUnit) -> (i64, u32) {
    match unit {
        TimestampUnit::Seconds => (value, 0),
        TimestampUnit::Millis => {
            let secs = value.div_euclid(1_000);
            let millis = value.rem_euclid(1_000) as u32;
            (secs, millis * 1_000_000)
        }
        TimestampUnit::Micros => {
            let secs = value.div_euclid(1_000_000);
            let micros = value.rem_euclid(1_000_000) as u32;
            (secs, micros * 1_000)
        }
        TimestampUnit::Nanos => {
            let secs = value.div_euclid(1_000_000_000);
            let nanos = value.rem_euclid(1_000_000_000) as u32;
            (secs, nanos)
        }
    }
}

fn days_from_secs(secs: i64) -> i32 {
    secs.div_euclid(86_400) as i32
}

fn parse_decimal_i128(value: &str, scale: i32) -> Option<i128> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let digits = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    let mut parts = digits.split('.');
    let whole = parts.next().unwrap_or("");
    let frac = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return None;
    }
    let scale = scale.max(0) as usize;
    if frac.len() > scale {
        return None;
    }
    let mut buf = String::with_capacity(whole.len() + scale);
    buf.push_str(whole);
    buf.push_str(frac);
    for _ in frac.len()..scale {
        buf.push('0');
    }
    let parsed = buf.parse::<i128>().ok()?;
    if negative {
        Some(-parsed)
    } else {
        Some(parsed)
    }
}

fn parse_decimal_i256(value: &str, scale: i32) -> Option<arrow_buffer::i256> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let digits = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    let mut parts = digits.split('.');
    let whole = parts.next().unwrap_or("");
    let frac = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return None;
    }
    let scale = scale.max(0) as usize;
    if frac.len() > scale {
        return None;
    }
    let mut buf = String::with_capacity(whole.len() + scale);
    buf.push_str(whole);
    buf.push_str(frac);
    for _ in frac.len()..scale {
        buf.push('0');
    }
    let mut parsed = arrow_buffer::i256::from_string(&buf)?;
    if negative {
        parsed = parsed.checked_neg()?;
    }
    Some(parsed)
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
