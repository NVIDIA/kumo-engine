//! Diskgraph ingest: convert raw CSV/Parquet sources into index-ready artifacts.

mod mapping;
pub mod source_metadata;

use crate::mapping::{Key, KeyType, PkMap};
use crate::source_metadata::{
    ColumnSpec, ColumnType, KeyMode, TableFormat, TableSpec, TimestampUnit,
};
use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_array::builder::ArrayBuilder;
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
    LogicalType as RowpackType, RowPackConfig, RowPackContextBuilder, RowPackEncoder,
    RowValueAccess, ValueRef,
};
use num_traits::Signed;
use object_store::{path::Path as ObjectPath, ObjectStore, WriteMultipart};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::FileReader;
use pprof::protos::Message;
use pprof::ProfilerGuard;
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

pub use mapping::MappingKind;
pub use source_metadata::SourceMetadata;

const NODE_BATCH_SIZE: usize = 100_000;
const EDGE_BATCH_SIZE: usize = 400_000;
const PROGRESS_STEP: u64 = 10_000;
const MAX_KEY_FAILURE_SAMPLES: u64 = 10;
const MAX_KEY_SAMPLE_LEN: usize = 64;
const EDGE_BIN_MAGIC: [u8; 4] = *b"DGED";
const EDGE_BIN_VERSION: u8 = 1;
const EDGE_BIN_FLAG_SORT: u8 = 1;

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
    NotFinite,
    Negative,
    Null,
    InvalidUuid,
}

impl KeyParseError {
    fn reason(self) -> &'static str {
        match self {
            KeyParseError::TypeMismatch => "type mismatch",
            KeyParseError::UnsupportedType => "unsupported type",
            KeyParseError::LengthMismatch => "length mismatch",
            KeyParseError::Empty => "empty",
            KeyParseError::TooLarge => "too large",
            KeyParseError::NotInteger => "not an integer",
            KeyParseError::NotBase10 => "not base10",
            KeyParseError::NotUtf8 => "invalid utf8",
            KeyParseError::NotFinite => "not finite",
            KeyParseError::Negative => "negative",
            KeyParseError::Null => "null",
            KeyParseError::InvalidUuid => "invalid uuid",
        }
    }
}

type KeyParseResult<T> = std::result::Result<T, KeyParseError>;

#[derive(Clone, Debug, Serialize)]
pub struct IngestReport {
    pub stats: IngestStats,
    pub source: SourceMetadata,
}

#[derive(Clone)]
pub struct IngestOptions {
    pub mapping_kind: MappingKind,
    pub progress: Option<ProgressCallback>,
    pub profile_dir: Option<PathBuf>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            mapping_kind: MappingKind::Hash,
            progress: None,
            profile_dir: None,
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
    pub pk_map_wall_ms: u64,
    pub build_wall_ms: u64,
    pub finalize_wall_ms: u64,
    pub pk_map_task_ms: u64,
    pub build_task_ms: u64,
    pub node_task_ms: u64,
    pub edge_task_ms: u64,
    pub pk_map_tasks: u64,
    pub build_tasks: u64,
    pub node_tasks: u64,
    pub edge_tasks: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

#[derive(Clone, Debug)]
pub enum ProgressEvent {
    Start { key: String, total: Option<u64> },
    Advance { key: String, delta: u64 },
    Finish { key: String, total: Option<u64> },
}

pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

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
    target.prepare()?;

    let mut ctx = IngestContext::new(source, target, options, output)?;
    let profile_dir = ctx.profile_dir.clone();
    if let Some(profile_dir) = &profile_dir {
        fs::create_dir_all(profile_dir)?;
    }
    let pk_map_start = std::time::Instant::now();
    if let Some(profile_dir) = &profile_dir {
        profile_phase(profile_dir, "pk_map", || ctx.build_pk_maps())?;
    } else {
        ctx.build_pk_maps()?;
    }
    let pk_map_wall = pk_map_start.elapsed();
    let mut build_wall = std::time::Duration::from_millis(0);
    if let Some(profile_dir) = &profile_dir {
        let node_start = std::time::Instant::now();
        profile_phase(profile_dir, "nodes", || ctx.build_node_data_and_edges(BuildPhase::Nodes))?;
        build_wall += node_start.elapsed();
        let node_task_ms = ctx.node_task_ms;
        let node_tasks = ctx.node_tasks;
        let build_task_ms = ctx.build_task_ms;
        let build_tasks = ctx.build_tasks;
        let edge_start = std::time::Instant::now();
        profile_phase(profile_dir, "edges", || ctx.build_node_data_and_edges(BuildPhase::Edges))?;
        build_wall += edge_start.elapsed();
        ctx.node_task_ms = node_task_ms;
        ctx.node_tasks = node_tasks;
        ctx.build_task_ms = build_task_ms + ctx.build_task_ms;
        ctx.build_tasks = build_tasks + ctx.build_tasks;
    } else {
        let build_start = std::time::Instant::now();
        ctx.build_node_data_and_edges(BuildPhase::All)?;
        build_wall = build_start.elapsed();
    }
    let finalize_start = std::time::Instant::now();
    let index_metadata = ctx.finalize_metadata()?;
    let finalize_wall = finalize_start.elapsed();
    let wall_time = wall_start.elapsed();
    let bytes_read = sum_input_bytes(&ctx.source);
    let bytes_written = match &*ctx.target {
        OutputTarget::Local { root } => sum_dir_bytes(root).unwrap_or(0),
        _ => 0,
    };
    let perf = PerfStats {
        wall_time_ms: wall_time.as_millis() as u64,
        pk_map_wall_ms: pk_map_wall.as_millis() as u64,
        build_wall_ms: build_wall.as_millis() as u64,
        finalize_wall_ms: finalize_wall.as_millis() as u64,
        pk_map_task_ms: ctx.pk_map_task_ms,
        build_task_ms: ctx.build_task_ms,
        node_task_ms: ctx.node_task_ms,
        edge_task_ms: ctx.edge_task_ms,
        pk_map_tasks: ctx.pk_map_tasks,
        build_tasks: ctx.build_tasks,
        node_tasks: ctx.node_tasks,
        edge_tasks: ctx.edge_tasks,
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
    mapping_kind: MappingKind,
    progress: Option<ProgressCallback>,
    profile_dir: Option<PathBuf>,
    pk_maps: HashMap<String, TableMapState>,
    table_rows: HashMap<String, u64>,
    table_contexts: HashMap<String, String>,
    relation_defs: Vec<RelationDef>,
    relationships: Vec<RelationshipRuntime>,
    edge_counts: HashMap<String, u64>,
    pk_duplicate_counts: HashMap<String, u64>,
    pk_duplicate_samples: HashMap<String, Vec<String>>,
    output: String,
    key_failures: std::sync::Arc<KeyFailureCollector>,
    pk_map_task_ms: u64,
    build_task_ms: u64,
    node_task_ms: u64,
    edge_task_ms: u64,
    pk_map_tasks: u64,
    build_tasks: u64,
    node_tasks: u64,
    edge_tasks: u64,
}

struct TableMapState {
    key_type: KeyType,
    pk_map: PkMap,
    sort_map: Option<PkMap>,
    pk_column: String,
    pk_dtype: ColumnType,
    fixed_len: Option<usize>,
    key_mode: Option<KeyMode>,
    rowpack_builder: Option<diskgraph_rowpack::RowPackContextBuilder>,
}

#[derive(Default)]
struct KeyFailureContext {
    parse_failures: u64,
    lookup_failures: u64,
}

struct KeyFailureCollector {
    contexts: std::sync::Mutex<HashMap<String, KeyFailureContext>>,
    samples: std::sync::Mutex<Vec<String>>,
}

#[derive(Default)]
struct KeyFailureLocal {
    contexts: HashMap<String, KeyFailureContext>,
    samples: Vec<String>,
}

impl KeyFailureCollector {
    fn new() -> Self {
        Self {
            contexts: std::sync::Mutex::new(HashMap::new()),
            samples: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn merge(&self, local: KeyFailureLocal) {
        if !local.contexts.is_empty() {
            let mut contexts = self.contexts.lock().unwrap();
            for (name, counts) in local.contexts {
                let entry = contexts.entry(name).or_default();
                entry.parse_failures = entry.parse_failures.saturating_add(counts.parse_failures);
                entry.lookup_failures =
                    entry.lookup_failures.saturating_add(counts.lookup_failures);
            }
        }
        if !local.samples.is_empty() {
            let mut samples = self.samples.lock().unwrap();
            if samples.len() < MAX_KEY_FAILURE_SAMPLES as usize {
                for sample in local.samples {
                    if samples.len() >= MAX_KEY_FAILURE_SAMPLES as usize {
                        break;
                    }
                    samples.push(sample);
                }
            }
        }
    }
}

impl KeyFailureLocal {
    fn record_parse_failure<F>(&mut self, context: &str, err: KeyParseError, sample: F)
    where
        F: FnOnce() -> Option<String>,
    {
        let entry = self.contexts.entry(context.to_string()).or_default();
        entry.parse_failures = entry.parse_failures.saturating_add(1);
        if self.samples.len() < MAX_KEY_FAILURE_SAMPLES as usize {
            if let Some(value) = sample() {
                self.samples
                    .push(format!("{context}: {} ({})", value, err.reason()));
            }
        }
    }

    fn record_lookup_failure<F>(&mut self, context: &str, sample: F)
    where
        F: FnOnce() -> Option<String>,
    {
        let entry = self.contexts.entry(context.to_string()).or_default();
        entry.lookup_failures = entry.lookup_failures.saturating_add(1);
        if self.samples.len() < MAX_KEY_FAILURE_SAMPLES as usize {
            if let Some(value) = sample() {
                self.samples
                    .push(format!("{context}: {} (missing key)", value));
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
    forward_sort: SortKeyMode,
    reverse_sort: SortKeyMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortKeyMode {
    None,
    Weight,
    CreateTime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuildPhase {
    All,
    Nodes,
    Edges,
}

impl IngestContext {
    fn new(
        source: SourceMetadata,
        target: std::sync::Arc<OutputTarget>,
        options: IngestOptions,
        output: &str,
    ) -> Result<Self> {
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
                forward_sort: dst_sort,
                reverse_sort: src_sort,
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
            mapping_kind: options.mapping_kind,
            progress: options.progress,
            profile_dir: options.profile_dir,
            pk_maps: HashMap::new(),
            table_rows: HashMap::new(),
            table_contexts: HashMap::new(),
            relation_defs,
            relationships,
            edge_counts: HashMap::new(),
            pk_duplicate_counts: HashMap::new(),
            pk_duplicate_samples: HashMap::new(),
            output: output.to_string(),
            key_failures: std::sync::Arc::new(KeyFailureCollector::new()),
            pk_map_task_ms: 0,
            build_task_ms: 0,
            node_task_ms: 0,
            edge_task_ms: 0,
            pk_map_tasks: 0,
            build_tasks: 0,
            node_tasks: 0,
            edge_tasks: 0,
        })
    }

    fn build_pk_maps(&mut self) -> Result<()> {
        let mapping_kind = self.mapping_kind;
        let progress = self.progress.clone();
        let tables: Vec<TableSpec> = self
            .source
            .tables
            .iter()
            .filter(|t| t.edge_table.is_none() && t.primary_key.is_some())
            .cloned()
            .collect();
        let pk_maps = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let table_rows = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let pk_duplicate_counts = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let pk_duplicate_samples = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let task_time = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let key_failures = std::sync::Arc::clone(&self.key_failures);
        let mut tasks = Vec::new();
        let source_rowpack = self.source.rowpack.clone();
        for table in tables {
            let pk_maps = std::sync::Arc::clone(&pk_maps);
            let table_rows = std::sync::Arc::clone(&table_rows);
            let pk_duplicate_counts = std::sync::Arc::clone(&pk_duplicate_counts);
            let pk_duplicate_samples = std::sync::Arc::clone(&pk_duplicate_samples);
            let progress = progress.clone();
            let key_failures = std::sync::Arc::clone(&key_failures);
            let source_rowpack = source_rowpack.clone();
            tasks.push(WorkItem {
                bucket: None,
                run: Box::new(move || {
                    let pk = table.primary_key.clone().unwrap_or_default();
                    let pk_col = column_spec(&table, &pk)?;
                    let key_type = key_type_for_column(&pk_col)?;
                    let pk_idx = column_index(&table, &pk)?;
                    let total_hint = row_count_hint(&table).unwrap_or(0);
                    let capacity_hint = estimate_rows(&table).max(total_hint as usize);
                    let mut pk_map = PkMap::new(mapping_kind, key_type, capacity_hint);
                    let mut sort_map = None;
                    let rowpack_config = rowpack_config_for_table(source_rowpack.as_ref(), &table);
                    let columns = rowpack_columns(&table)?;
                    let mut rowpack_builder =
                        RowPackContextBuilder::new(columns, rowpack_config);
                    let mut row_id = 0u64;
                    let progress_key = format!("pk_map:{}", table.name);
                    let total_opt = if total_hint > 0 { Some(total_hint) } else { None };
                    if let Some(cb) = &progress {
                        cb(ProgressEvent::Start {
                            key: progress_key.clone(),
                            total: total_opt,
                        });
                    }
                    let mut next_update = PROGRESS_STEP;
                    let mut arrays_f32: Vec<Vec<f32>> = Vec::new();
                    let mut arrays_f64: Vec<Vec<f64>> = Vec::new();
                    let mut strings: Vec<String> = Vec::new();
                    let mut bytes: Vec<Vec<u8>> = Vec::new();
                    let mut key_failures_local = KeyFailureLocal::default();
                    let parse_context = format!("pk_map:{}:{}", table.name, pk);
                    for_each_row(&table, |row| {
                        let array = row
                            .cols
                            .get(pk_idx)
                            .ok_or_else(|| anyhow!("missing column {}", pk))?;
                        let key = match key_from_array_for_target(
                            array,
                            row.row,
                            &pk_col,
                            &pk_col,
                            key_type,
                        ) {
                            Ok(key) => Some(key),
                            Err(err) => {
                                if matches!(
                                    err,
                                    KeyParseError::TypeMismatch | KeyParseError::UnsupportedType
                                ) {
                                    return Err(anyhow!("{} key parse error: {}", pk, err.reason()));
                                }
                                key_failures_local.record_parse_failure(&parse_context, err, || {
                                    sample_key_value(array, row.row, &pk_col)
                                });
                                None
                            }
                        };
                        if let (Some(key), Some(sort_val)) = (key.as_ref(), row.sort_key(&table)?) {
                            let sort_key = sort_key_to_u32(&sort_val)?;
                            let sort_map = sort_map
                                .get_or_insert_with(|| PkMap::new(mapping_kind, key_type, capacity_hint));
                            sort_map.insert(key.clone(), sort_key as u64)?;
                        }
                        if let Some(key) = key {
                            pk_map.insert(key, row_id)?;
                        }
                        arrays_f32.clear();
                        arrays_f64.clear();
                        strings.clear();
                        bytes.clear();
                        let mut values = Vec::with_capacity(table.columns.len());
                        row_values_from_row(
                            row,
                            &table,
                            &mut values,
                            &mut arrays_f32,
                            &mut arrays_f64,
                            &mut strings,
                            &mut bytes,
                        )?;
                        rowpack_builder.observe_row(&values)?;
                        row_id += 1;
                        if row_id == next_update {
                            if let Some(cb) = &progress {
                                cb(ProgressEvent::Advance {
                                    key: progress_key.clone(),
                                    delta: PROGRESS_STEP,
                                });
                            }
                            next_update = next_update.saturating_add(PROGRESS_STEP);
                        }
                        Ok(())
                    })?;
                    if row_id > 0 && row_id < next_update {
                        let delta = row_id.saturating_sub(next_update - PROGRESS_STEP);
                        if delta > 0 {
                            if let Some(cb) = &progress {
                                cb(ProgressEvent::Advance {
                                    key: progress_key.clone(),
                                    delta,
                                });
                            }
                        }
                    }
                    if let Some(cb) = &progress {
                        cb(ProgressEvent::Finish {
                            key: progress_key,
                            total: Some(row_id),
                        });
                    }
                    key_failures.merge(key_failures_local);
                    pk_map.finalize();
                    if let Some(sort) = sort_map.as_mut() {
                        sort.finalize();
                    }
                    let duplicate_count = pk_map.duplicate_count();
                    if duplicate_count > 0 {
                        pk_duplicate_counts
                            .lock()
                            .unwrap()
                            .insert(table.name.clone(), duplicate_count);
                        let mut samples = Vec::new();
                        for key in pk_map.duplicate_samples() {
                            if samples.len() >= 10 {
                                break;
                            }
                            if let Some(sample) = sample_key(key) {
                                samples.push(sample);
                            }
                        }
                        if !samples.is_empty() {
                            pk_duplicate_samples
                                .lock()
                                .unwrap()
                                .insert(table.name.clone(), samples);
                        }
                    }
                    table_rows.lock().unwrap().insert(table.name.clone(), row_id);
                    pk_maps.lock().unwrap().insert(
                        table.name.clone(),
                        TableMapState {
                            key_type,
                            pk_map,
                            sort_map,
                            pk_column: pk.clone(),
                            pk_dtype: pk_col.ingested_dtype.clone(),
                            fixed_len: pk_col.fixed_len,
                            key_mode: pk_col.key_mode.clone(),
                            rowpack_builder: Some(rowpack_builder),
                        },
                    );
                    Ok(())
                }),
            });
        }
        self.pk_map_tasks = tasks.len() as u64;
        run_tasks(tasks, default_threads(), Some(task_time.clone()))?;
        self.pk_map_task_ms = task_time.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000;
        let pk_maps = std::sync::Arc::try_unwrap(pk_maps)
            .map_err(|_| anyhow!("pk map tasks still running"))?
            .into_inner()
            .unwrap();
        let table_rows = std::sync::Arc::try_unwrap(table_rows)
            .map_err(|_| anyhow!("pk map row tasks still running"))?
            .into_inner()
            .unwrap();
        let pk_duplicate_counts = std::sync::Arc::try_unwrap(pk_duplicate_counts)
            .map_err(|_| anyhow!("pk map duplicate tasks still running"))?
            .into_inner()
            .unwrap();
        let pk_duplicate_samples = std::sync::Arc::try_unwrap(pk_duplicate_samples)
            .map_err(|_| anyhow!("pk map duplicate sample tasks still running"))?
            .into_inner()
            .unwrap();
        for (table_name, state) in &pk_maps {
            let map_path = format!("nodes/{}/pk_map.bin", table_name.to_lowercase());
            let target = self.target.join(&map_path);
            if let OutputTarget::Local { .. } = self.target.as_ref() {
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent)?;
                }
            }
            let sink = self.target.sink(&target)?;
            let mut writer = io::BufWriter::new(sink);
            state.pk_map.write_to(&mut writer)?;
            writer.flush()?;
            let sink = match writer.into_inner() {
                Ok(sink) => sink,
                Err(err) => {
                    let message = err.error().to_string();
                    let _ = err.into_inner().flush();
                    return Err(anyhow!(message));
                }
            };
            sink.finish()?;
        }
        self.pk_maps = pk_maps;
        self.table_rows = table_rows;
        self.pk_duplicate_counts = pk_duplicate_counts;
        self.pk_duplicate_samples = pk_duplicate_samples;
        Ok(())
    }

    fn build_node_data_and_edges(&mut self, phase: BuildPhase) -> Result<()> {
        let progress = self.progress.clone();
        let target = std::sync::Arc::clone(&self.target);
        let source = std::sync::Arc::new(self.source.clone());
        let pk_maps = std::sync::Arc::new(std::mem::take(&mut self.pk_maps));
        let source_rowpack = source.rowpack.clone();
        let table_rows_snapshot = std::sync::Arc::new(self.table_rows.clone());
        let table_rows = std::sync::Arc::new(std::sync::Mutex::new(self.table_rows.clone()));
        let table_contexts =
            std::sync::Arc::new(std::sync::Mutex::new(self.table_contexts.clone()));
        let edge_counts = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        let relationships = self.relationships.clone();
        let relationships_len = relationships.len();
        let task_time = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let node_task_time = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let edge_task_time = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let key_failures = std::sync::Arc::clone(&self.key_failures);

        let mut tasks = Vec::new();

        let include_nodes = matches!(phase, BuildPhase::All | BuildPhase::Nodes);
        let include_edges = matches!(phase, BuildPhase::All | BuildPhase::Edges);

        if include_nodes {
            for table in source.tables.iter().filter(|t| t.edge_table.is_none()) {
                let table = table.clone();
                let pk_maps = std::sync::Arc::clone(&pk_maps);
                let target = std::sync::Arc::clone(&target);
                let progress = progress.clone();
                let table_rows_snapshot = std::sync::Arc::clone(&table_rows_snapshot);
                let table_rows = std::sync::Arc::clone(&table_rows);
                let table_contexts = std::sync::Arc::clone(&table_contexts);
                let source_rowpack = source_rowpack.clone();
                let node_task_time = std::sync::Arc::clone(&node_task_time);
                tasks.push(WorkItem {
                    bucket: Some(node_task_time),
                    run: Box::new(move || {
                        let rowpack_builder =
                            pk_maps.get(table.name.as_str()).and_then(|s| s.rowpack_builder.clone());
                        let context = if let Some(builder) = rowpack_builder {
                            builder.finish()?
                        } else {
                            sample_rowpack_context(source_rowpack.as_ref(), &table)?
                        };
                        let context_path =
                            format!("nodes/{}/node_data.ctx", table.name.to_lowercase());
                        target.write_bytes(&context_path, context.to_bytes()?)?;
                        table_contexts
                            .lock()
                            .unwrap()
                            .insert(table.name.clone(), context_path.clone());
                        let encoder = RowPackEncoder::new(context)?;
                        let progress_key = format!("nodes:{}", table.name);
                        let total = table_rows_snapshot
                            .get(table.name.as_str())
                            .copied()
                            .or_else(|| row_count_hint(&table));
                        let progress_enabled = total.is_some();
                        if progress_enabled {
                            emit_progress_start(&progress, &progress_key, total);
                        }
                        let row_count = write_node_data_to(
                            &target,
                            &table,
                            &encoder,
                            progress.as_ref(),
                            progress_enabled.then_some(progress_key.as_str()),
                        )?;
                        if progress_enabled {
                            emit_progress_finish(&progress, &progress_key, Some(row_count));
                        }
                        table_rows.lock().unwrap().insert(table.name.clone(), row_count);
                        Ok(())
                    }),
                });
            }
        }

        if include_edges {
            for rel in relationships {
                let source = std::sync::Arc::clone(&source);
                let pk_maps = std::sync::Arc::clone(&pk_maps);
                let target = std::sync::Arc::clone(&target);
                let progress = progress.clone();
                let table_rows_snapshot = std::sync::Arc::clone(&table_rows_snapshot);
                let edge_counts = std::sync::Arc::clone(&edge_counts);
                let edge_task_time = std::sync::Arc::clone(&edge_task_time);
                let key_failures = std::sync::Arc::clone(&key_failures);
                tasks.push(WorkItem {
                    bucket: Some(edge_task_time),
                    run: Box::new(move || {
                    let src_table = table_by_name(&source, &rel.src_table)?;
                    let dst_map = pk_maps
                        .get(rel.dst_table.as_str())
                        .ok_or_else(|| anyhow!("missing pk map for {}", rel.dst_table))?;
                    let dst_pk = dst_map.pk_column.clone();
                    let dst_pk_col =
                        column_spec(table_by_name(&source, &rel.dst_table)?, &dst_pk)?;
                    let dst_key_type = key_type_for_column(&dst_pk_col)?;
                    if dst_key_type != dst_map.key_type {
                        return Err(anyhow!("pk map key type mismatch for {}", rel.dst_table));
                    }
                    let src_fk_idx = column_index(src_table, &rel.src_column)?;
                    let src_fk_col = &src_table.columns[src_fk_idx];
                    let dst_sort_map = dst_map.sort_map.as_ref();
                    if rel.forward_sort != SortKeyMode::None && dst_sort_map.is_none() {
                        return Err(anyhow!("missing sort key map for {}", rel.dst_table));
                    }
                    let forward_key = format!("edges:{}", rel.name_forward);
                    let reverse_key = format!("edges:{}", rel.name_reverse);
                    let total = table_rows_snapshot.get(rel.src_table.as_str()).copied();
                    let progress_enabled = total.is_some();
                    if progress_enabled {
                        emit_progress_start(&progress, &forward_key, total);
                        emit_progress_start(&progress, &reverse_key, total);
                    }
                    let edge_format = edge_output_format(&source);
                    let mut forward_writer = create_edge_writer(
                        &target,
                        &rel.name_forward,
                        rel.forward_sort != SortKeyMode::None,
                        edge_format,
                    )?;
                    let mut forward_edges =
                        EdgeAccumulator::new(rel.forward_sort != SortKeyMode::None);
                    let mut reverse_records: Vec<EdgeRecord> = Vec::new();
                    let mut forward_count = 0u64;
                    let mut reverse_count = 0u64;
                    let mut seen = 0u64;
                    let mut next_update = PROGRESS_STEP;
                    let parse_context = format!("rel:{}:{}", rel.name_forward, rel.src_column);
                    let mut key_failures_local = KeyFailureLocal::default();
                    for_each_row(src_table, |row| {
                        let src_id = row.row_id;
                        let array = row
                            .cols
                            .get(src_fk_idx)
                            .ok_or_else(|| anyhow!("missing column {}", rel.src_column))?;
                        seen += 1;
                        if progress_enabled && seen == next_update {
                            emit_progress_advance(&progress, &forward_key, PROGRESS_STEP);
                            emit_progress_advance(&progress, &reverse_key, PROGRESS_STEP);
                            next_update = next_update.saturating_add(PROGRESS_STEP);
                        }
                        let fk_key = match key_from_array_for_target(
                            array,
                            row.row,
                            src_fk_col,
                            dst_pk_col,
                            dst_key_type,
                        ) {
                            Ok(key) => key,
                            Err(err) => {
                                key_failures_local.record_parse_failure(&parse_context, err, || {
                                    sample_key_value(array, row.row, src_fk_col)
                                });
                                return Ok(());
                            }
                        };
                        let dst_id = match dst_map.pk_map.lookup(&fk_key) {
                            Some(id) => id,
                            None => {
                                key_failures_local.record_lookup_failure(&parse_context, || {
                                    sample_key(&fk_key)
                                });
                                return Ok(());
                            }
                        };
                        let forward_sort_key = match rel.forward_sort {
                            SortKeyMode::Weight | SortKeyMode::CreateTime => dst_sort_map
                                .and_then(|m| m.lookup(&fk_key))
                                .map(|v| v as u32),
                            SortKeyMode::None => None,
                        };
                        forward_edges.push(src_id, dst_id, forward_sort_key);
                        let reverse_sort_key = match rel.reverse_sort {
                            SortKeyMode::Weight | SortKeyMode::CreateTime => row
                                .sort_key(src_table)?
                                .map(|v| sort_key_to_u32(&v).unwrap_or(0)),
                            SortKeyMode::None => None,
                        };
                        reverse_records.push(EdgeRecord::new(dst_id, src_id, reverse_sort_key));
                        forward_count += 1;
                        reverse_count += 1;
                        if forward_edges.len() >= EDGE_BATCH_SIZE {
                            flush_edge_batch(&mut forward_writer, &forward_edges)?;
                            forward_edges.clear();
                        }
                        Ok(())
                    })?;
                    if !forward_edges.is_empty() {
                        flush_edge_batch(&mut forward_writer, &forward_edges)?;
                    }
                    finish_edge_writer(forward_writer)?;
                    if !edge_records_sorted(&reverse_records) {
                        reverse_records.sort_unstable_by(edge_record_cmp);
                    }
                    write_edge_records_to(
                        &target,
                        &rel.name_reverse,
                        rel.reverse_sort != SortKeyMode::None,
                        edge_format,
                        &reverse_records,
                    )?;
                    if progress_enabled {
                        emit_progress_finish(&progress, &forward_key, Some(seen));
                        emit_progress_finish(&progress, &reverse_key, Some(seen));
                    }
                    edge_counts
                        .lock()
                        .unwrap()
                        .insert(rel.name_forward.clone(), forward_count);
                    edge_counts
                        .lock()
                        .unwrap()
                        .insert(rel.name_reverse.clone(), reverse_count);
                    key_failures.merge(key_failures_local);
                    Ok(())
                    }),
                });
            }

            for table in source.tables.iter().filter(|t| t.edge_table.is_some()) {
                let table = table.clone();
                let source = std::sync::Arc::clone(&source);
                let pk_maps = std::sync::Arc::clone(&pk_maps);
                let target = std::sync::Arc::clone(&target);
                let progress = progress.clone();
                let edge_counts = std::sync::Arc::clone(&edge_counts);
                let edge_task_time = std::sync::Arc::clone(&edge_task_time);
                let key_failures = std::sync::Arc::clone(&key_failures);
                tasks.push(WorkItem {
                    bucket: Some(edge_task_time),
                    run: Box::new(move || {
                    let edge = table
                        .edge_table
                        .as_ref()
                        .ok_or_else(|| anyhow!("missing edge table spec for {}", table.name))?;
                    let src_map = pk_maps
                        .get(edge.src_fk.ref_table.as_str())
                        .ok_or_else(|| anyhow!("missing pk map for {}", edge.src_fk.ref_table))?;
                    let dst_map = pk_maps
                        .get(edge.dst_fk.ref_table.as_str())
                        .ok_or_else(|| anyhow!("missing pk map for {}", edge.dst_fk.ref_table))?;
                    let _src_pk_col = column_spec(
                        table_by_name(&source, &edge.src_fk.ref_table)?,
                        &edge.src_fk.ref_column,
                    )?;
                    let _dst_pk_col = column_spec(
                        table_by_name(&source, &edge.dst_fk.ref_table)?,
                        &edge.dst_fk.ref_column,
                    )?;
                    let src_fk_idx = column_index(&table, &edge.src_fk.column)?;
                    let dst_fk_idx = column_index(&table, &edge.dst_fk.column)?;
                    let src_fk_col = &table.columns[src_fk_idx];
                    let dst_fk_col = &table.columns[dst_fk_idx];
                    let src_key_type = src_map.key_type;
                    let dst_key_type = dst_map.key_type;
                    let sort_mode = if table.weight.is_some() {
                        SortKeyMode::Weight
                    } else if table.create_time.is_some() {
                        SortKeyMode::CreateTime
                    } else {
                        SortKeyMode::None
                    };
                    let total_hint = row_count_hint(&table).unwrap_or(0);
                    let total_opt = if total_hint > 0 { Some(total_hint) } else { None };
                    let progress_key = format!("edge_table:{}", table.name);
                    let progress_enabled = total_opt.is_some();
                    if progress_enabled {
                        emit_progress_start(&progress, &progress_key, total_opt);
                    }
                    let mut next_update = PROGRESS_STEP;
                    let mut seen = 0u64;
                    let parse_context_src =
                        format!("edge:{}:{}", table.name, edge.src_fk.column);
                    let parse_context_dst =
                        format!("edge:{}:{}", table.name, edge.dst_fk.column);
                    let mut key_failures_local = KeyFailureLocal::default();
                    let mut forward_records: Vec<EdgeRecord> = Vec::new();
                    let mut reverse_records: Vec<EdgeRecord> = Vec::new();
                    for_each_row(&table, |row| {
                    let src_array = row
                        .cols
                        .get(src_fk_idx)
                        .ok_or_else(|| anyhow!("missing column {}", edge.src_fk.column))?;
                    let dst_array = row
                        .cols
                        .get(dst_fk_idx)
                        .ok_or_else(|| anyhow!("missing column {}", edge.dst_fk.column))?;
                        seen += 1;
                        if progress_enabled && seen == next_update {
                            emit_progress_advance(&progress, &progress_key, PROGRESS_STEP);
                            next_update = next_update.saturating_add(PROGRESS_STEP);
                        }
                        let src_key = match key_from_array_for_target(
                            src_array,
                            row.row,
                            src_fk_col,
                            _src_pk_col,
                            src_key_type,
                        ) {
                            Ok(key) => key,
                            Err(err) => {
                                key_failures_local.record_parse_failure(&parse_context_src, err, || {
                                    sample_key_value(src_array, row.row, src_fk_col)
                                });
                                return Ok(());
                            }
                        };
                        let dst_key = match key_from_array_for_target(
                            dst_array,
                            row.row,
                            dst_fk_col,
                            _dst_pk_col,
                            dst_key_type,
                        ) {
                            Ok(key) => key,
                            Err(err) => {
                                key_failures_local.record_parse_failure(&parse_context_dst, err, || {
                                    sample_key_value(dst_array, row.row, dst_fk_col)
                                });
                                return Ok(());
                            }
                        };
                        let src_id = match src_map.pk_map.lookup(&src_key) {
                            Some(id) => id,
                            None => {
                                key_failures_local.record_lookup_failure(&parse_context_src, || {
                                    sample_key(&src_key)
                                });
                                return Ok(());
                            }
                        };
                        let dst_id = match dst_map.pk_map.lookup(&dst_key) {
                            Some(id) => id,
                            None => {
                                key_failures_local.record_lookup_failure(&parse_context_dst, || {
                                    sample_key(&dst_key)
                                });
                                return Ok(());
                            }
                        };
                        let sort_key = match sort_mode {
                            SortKeyMode::Weight | SortKeyMode::CreateTime => {
                                row.sort_key(&table)?.map(|v| sort_key_to_u32(&v).unwrap_or(0))
                            }
                            SortKeyMode::None => None,
                        };
                        forward_records.push(EdgeRecord::new(src_id, dst_id, sort_key));
                        reverse_records.push(EdgeRecord::new(dst_id, src_id, sort_key));
                        Ok(())
                    })?;
                    if progress_enabled {
                        let delta = seen.saturating_sub(next_update - PROGRESS_STEP);
                        if delta > 0 {
                            emit_progress_advance(&progress, &progress_key, delta);
                        }
                    }
                    let edge_format = edge_output_format(&source);
                    if !edge_records_sorted(&forward_records) {
                        forward_records.sort_unstable_by(edge_record_cmp);
                    }
                    if !edge_records_sorted(&reverse_records) {
                        reverse_records.sort_unstable_by(edge_record_cmp);
                    }
                    write_edge_records_to(
                        &target,
                        &format!("edge_{}_src_to_dst", table.name),
                        sort_mode != SortKeyMode::None,
                        edge_format,
                        &forward_records,
                    )?;
                    write_edge_records_to(
                        &target,
                        &format!("edge_{}_dst_to_src", table.name),
                        sort_mode != SortKeyMode::None,
                        edge_format,
                        &reverse_records,
                    )?;
                    if progress_enabled {
                        emit_progress_finish(&progress, &progress_key, Some(seen));
                    }
                    edge_counts.lock().unwrap().insert(
                        format!("edge_{}_src_to_dst", table.name),
                        forward_records.len() as u64,
                    );
                    edge_counts.lock().unwrap().insert(
                        format!("edge_{}_dst_to_src", table.name),
                        reverse_records.len() as u64,
                    );
                    key_failures.merge(key_failures_local);
                    Ok(())
                    }),
                });
            }
        }

        self.node_tasks = if include_nodes {
            source.tables.iter().filter(|t| t.edge_table.is_none()).count() as u64
        } else {
            0
        };
        self.edge_tasks = if include_edges {
            relationships_len as u64
                + source.tables.iter().filter(|t| t.edge_table.is_some()).count() as u64
        } else {
            0
        };
        self.build_tasks = tasks.len() as u64;
        run_tasks(tasks, default_threads(), Some(task_time.clone()))?;
        self.build_task_ms = task_time.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000;
        self.node_task_ms = node_task_time.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000;
        self.edge_task_ms = edge_task_time.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000;
        self.table_rows = std::sync::Arc::try_unwrap(table_rows)
            .map_err(|_| anyhow!("node task row refs still running"))?
            .into_inner()
            .unwrap();
        self.edge_counts = std::sync::Arc::try_unwrap(edge_counts)
            .map_err(|_| anyhow!("edge task refs still running"))?
            .into_inner()
            .unwrap();
        self.table_contexts = std::sync::Arc::try_unwrap(table_contexts)
            .map_err(|_| anyhow!("node task context refs still running"))?
            .into_inner()
            .unwrap();
        self.pk_maps = std::sync::Arc::try_unwrap(pk_maps)
            .map_err(|_| anyhow!("pk map refs still running"))?;
        Ok(())
    }

    fn finalize_metadata(&self) -> Result<DatasetMetadata> {
        let mut node_types = Vec::new();
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            let map_state = self.pk_maps.get(table.name.as_str());
            let row_count = self
                .table_rows
                .get(table.name.as_str())
                .copied()
                .unwrap_or(0);
            let shard = NodeShardMetadata {
                min_id: 0,
                max_id: row_count.saturating_sub(1),
                node_data_path: Some(format!(
                    "nodes/{}/part-00000.parquet",
                    table.name.to_lowercase()
                )),
            };
            let (compression, pk_map) = if let Some(state) = map_state {
                let context_path = self.table_contexts.get(&table.name).cloned();
                let compression = context_path.map(|path| NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: path,
                });
                let fixed_len = if state.key_mode == Some(KeyMode::Uuid) {
                    Some(16)
                } else {
                    state.fixed_len
                };
                let pk_map = Some(PrimaryKeyMap {
                    column: state.pk_column.clone(),
                    dtype: format!("{:?}", state.pk_dtype).to_lowercase(),
                    mapping_path: format!(
                        "nodes/{}/pk_map.bin",
                        table.name.to_lowercase()
                    ),
                    mapping_kind: match self.mapping_kind {
                        MappingKind::Hash => "hash".to_string(),
                        MappingKind::Sorted => "sorted".to_string(),
                    },
                    fixed_len,
                    key_mode: state
                        .key_mode
                        .as_ref()
                        .and_then(|mode| match mode {
                            KeyMode::Uuid => Some("uuid".to_string()),
                            KeyMode::Fixed => None,
                        }),
                });
                (compression, pk_map)
            } else {
                let context_path = self.table_contexts.get(&table.name).cloned();
                let compression = context_path.map(|path| NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: path,
                });
                (compression, None)
            };
            node_types.push(NodeTypeMetadata {
                name: table.name.clone(),
                max_id: shard.max_id,
                has_node_data: true,
                node_data_compression: compression,
                primary_key_map: pk_map,
                shards: vec![shard],
            });
        }

        let mut edge_relations = Vec::new();
        let edge_format = edge_output_format(&self.source);
        for rel in &self.relation_defs {
            let rel_path = edge_shard_path(&rel.name, edge_format);
            edge_relations.push(EdgeRelationMetadata {
                name: rel.name.clone(),
                src_type: rel.src_table.clone(),
                dst_type: rel.dst_table.clone(),
                shards: vec![EdgeShardMetadata {
                    path: rel_path,
                    src_min_id: 0,
                    src_max_id: table_rows(self, &rel.src_table).saturating_sub(1),
                }],
                dest_sort_key_bits: match rel.sort_key {
                    SortKeyMode::None => None,
                    _ => Some(32),
                },
                format: edge_format,
            });
        }

        let index_metadata = DatasetMetadata {
            dataset_name: self.source.dataset_name.clone(),
            node_types,
            edge_relations,
        };
        let json = serde_json::to_vec_pretty(&index_metadata)?;
        self.target.write_bytes("index_metadata.json", json)?;

        Ok(index_metadata)
    }

    fn build_stats(&self, metadata: &DatasetMetadata, mut perf: PerfStats) -> IngestStats {
        perf.node_task_ms = self.node_task_ms;
        perf.edge_task_ms = self.edge_task_ms;
        perf.node_tasks = self.node_tasks;
        perf.edge_tasks = self.edge_tasks;
        let mut node_counts = HashMap::new();
        for node in metadata.node_types.iter() {
            let count = node.max_id.saturating_add(1);
            node_counts.insert(node.name.clone(), count);
        }
        let total_nodes = node_counts.values().sum();
        let total_edges = self.edge_counts.values().sum();
        let key_failure_samples = self.key_failures.samples.lock().unwrap().clone();
        let contexts = self.key_failures.contexts.lock().unwrap();
        let mut key_failures_by_context = HashMap::new();
        let mut key_parse_failures = 0u64;
        let mut key_lookup_failures = 0u64;
        for (name, counts) in contexts.iter() {
            key_parse_failures = key_parse_failures.saturating_add(counts.parse_failures);
            key_lookup_failures = key_lookup_failures.saturating_add(counts.lookup_failures);
            key_failures_by_context.insert(
                name.clone(),
                KeyFailureCounts {
                    parse_failures: counts.parse_failures,
                    lookup_failures: counts.lookup_failures,
                },
            );
        }
        let pk_duplicate_total: u64 = self.pk_duplicate_counts.values().sum();
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
            perf,
        }
    }

}

fn emit_progress_start(progress: &Option<ProgressCallback>, key: &str, total: Option<u64>) {
    if let Some(progress) = progress {
        progress(ProgressEvent::Start {
            key: key.to_string(),
            total,
        });
    }
}

fn emit_progress_advance(progress: &Option<ProgressCallback>, key: &str, delta: u64) {
    if let Some(progress) = progress {
        progress(ProgressEvent::Advance {
            key: key.to_string(),
            delta,
        });
    }
}

fn emit_progress_finish(progress: &Option<ProgressCallback>, key: &str, total: Option<u64>) {
    if let Some(progress) = progress {
        progress(ProgressEvent::Finish {
            key: key.to_string(),
            total,
        });
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

fn write_node_data_to(
    target: &OutputTarget,
    table: &TableSpec,
    encoder: &RowPackEncoder,
    progress: Option<&ProgressCallback>,
    progress_key: Option<&str>,
) -> Result<u64> {
    let rel_path = format!("nodes/{}/part-00000.parquet", table.name.to_lowercase());
    let path = target.join(&rel_path);
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    let schema = Schema::new(vec![Field::new("data", DataType::Binary, false)]);
    let props = WriterProperties::builder()
        .set_max_row_group_size(NODE_BATCH_SIZE)
        .build();
    let sink = target.sink(&path)?;
    let mut writer = ArrowWriter::try_new(sink, Arc::new(schema.clone()), Some(props))?;
    let mut rows = 0u64;
    let mut next_update = PROGRESS_STEP;
    let mut scratch = diskgraph_rowpack::RowPackScratch::new(encoder.context());
    let mut encoded = Vec::with_capacity(encoder.context().estimated_row_bytes as usize);
    let mut builder = arrow_array::builder::BinaryBuilder::new();
    let mut arrays_f32: Vec<Vec<f32>> = Vec::new();
    let mut arrays_f64: Vec<Vec<f64>> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    let mut bytes: Vec<Vec<u8>> = Vec::new();
    for_each_batch(table, |cols, batch_rows, _row_base| {
        let accessors = build_accessors(cols, table)?;
        for row in 0..batch_rows {
            arrays_f32.clear();
            arrays_f64.clear();
            strings.clear();
            bytes.clear();
            let mut access = RowAccessor {
                row,
                accessors: &accessors,
                columns: &table.columns,
                strings: &mut strings,
                bytes: &mut bytes,
                arrays_f32: &mut arrays_f32,
                arrays_f64: &mut arrays_f64,
            };
            encoder.encode_row_from_access(&mut access, &mut encoded, &mut scratch)?;
            builder.append_value(&encoded);
            rows += 1;
            if rows == next_update {
                if let (Some(progress), Some(key)) = (progress, progress_key) {
                    progress(ProgressEvent::Advance {
                        key: key.to_string(),
                        delta: PROGRESS_STEP,
                    });
                }
                next_update = next_update.saturating_add(PROGRESS_STEP);
            }
            if builder.len() >= NODE_BATCH_SIZE {
                flush_node_batch(&schema, &mut writer, &mut builder)?;
                builder = arrow_array::builder::BinaryBuilder::new();
            }
        }
        Ok(())
    })?;
    if let (Some(progress), Some(key)) = (progress, progress_key) {
        let delta = rows.saturating_sub(next_update - PROGRESS_STEP);
        if delta > 0 {
            progress(ProgressEvent::Advance {
                key: key.to_string(),
                delta,
            });
        }
    }
    flush_node_batch(&schema, &mut writer, &mut builder)?;
    let sink = writer.into_inner()?;
    sink.finish()?;
    Ok(rows)
}

enum EdgeWriter {
    Parquet { writer: ArrowWriter<OutputSink>, schema: Schema },
    Binary { writer: BufWriter<OutputSink>, with_sort: bool },
}

fn edge_output_format(source: &SourceMetadata) -> EdgeStorageFormat {
    source.edge_format.unwrap_or(EdgeStorageFormat::Binary)
}

fn edge_shard_path(name: &str, format: EdgeStorageFormat) -> String {
    match format {
        EdgeStorageFormat::Parquet => {
            format!("edges/{}/part-00000.parquet", name.to_lowercase())
        }
        EdgeStorageFormat::Binary => format!("edges/{}/part-00000.bin", name.to_lowercase()),
    }
}

fn create_edge_writer(
    target: &OutputTarget,
    name: &str,
    with_sort: bool,
    format: EdgeStorageFormat,
) -> Result<EdgeWriter> {
    let path = edge_shard_path(name, format);
    let out_path = target.join(&path);
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    match format {
        EdgeStorageFormat::Parquet => {
            let schema = edge_schema(with_sort);
            let props = WriterProperties::builder()
                .set_max_row_group_size(EDGE_BATCH_SIZE)
                .build();
            let sink = target.sink(&out_path)?;
            let writer = ArrowWriter::try_new(sink, Arc::new(schema.clone()), Some(props))?;
            Ok(EdgeWriter::Parquet { writer, schema })
        }
        EdgeStorageFormat::Binary => {
            let sink = target.sink(&out_path)?;
            let mut writer = BufWriter::new(sink);
            write_edge_bin_header(&mut writer, with_sort)?;
            Ok(EdgeWriter::Binary { writer, with_sort })
        }
    }
}

fn write_edge_records_to(
    target: &OutputTarget,
    name: &str,
    with_sort: bool,
    format: EdgeStorageFormat,
    records: &[EdgeRecord],
) -> Result<()> {
    let mut writer = create_edge_writer(target, name, with_sort, format)?;
    let mut batch = EdgeAccumulator::new(with_sort);
    for record in records {
        batch.push(record.src_id, record.dst_id, record.sort_key);
        if batch.len() >= EDGE_BATCH_SIZE {
            flush_edge_batch(&mut writer, &batch)?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        flush_edge_batch(&mut writer, &batch)?;
    }
    finish_edge_writer(writer)?;
    Ok(())
}

struct EdgeAccumulator {
    src: Vec<i64>,
    dst: Vec<i64>,
    sort: Vec<i64>,
    with_sort: bool,
}

impl EdgeAccumulator {
    fn new(with_sort: bool) -> Self {
        Self {
            src: Vec::with_capacity(EDGE_BATCH_SIZE),
            dst: Vec::with_capacity(EDGE_BATCH_SIZE),
            sort: Vec::with_capacity(EDGE_BATCH_SIZE),
            with_sort,
        }
    }

    fn push(&mut self, src: u64, dst: u64, sort: Option<u32>) {
        self.src.push(src as i64);
        self.dst.push(dst as i64);
        if self.with_sort {
            self.sort.push(sort.unwrap_or(0) as i64);
        }
    }

    fn len(&self) -> usize {
        self.src.len()
    }

    fn is_empty(&self) -> bool {
        self.src.is_empty()
    }

    fn clear(&mut self) {
        self.src.clear();
        self.dst.clear();
        self.sort.clear();
    }
}

#[derive(Clone)]
struct EdgeRecord {
    src_id: u64,
    dst_id: u64,
    sort_key: Option<u32>,
}

impl EdgeRecord {
    fn new(src_id: u64, dst_id: u64, sort_key: Option<u32>) -> Self {
        Self {
            src_id,
            dst_id,
            sort_key,
        }
    }
}

fn edge_record_cmp(a: &EdgeRecord, b: &EdgeRecord) -> std::cmp::Ordering {
    match a.src_id.cmp(&b.src_id) {
        std::cmp::Ordering::Equal => match (a.sort_key, b.sort_key) {
            (Some(a), Some(b)) => b.cmp(&a),
            (None, None) => a.dst_id.cmp(&b.dst_id),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
        },
        other => other,
    }
}

fn edge_records_sorted(records: &[EdgeRecord]) -> bool {
    records
        .windows(2)
        .all(|w| edge_record_cmp(&w[0], &w[1]) != std::cmp::Ordering::Greater)
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

fn flush_edge_batch(writer: &mut EdgeWriter, data: &EdgeAccumulator) -> Result<()> {
    match writer {
        EdgeWriter::Parquet { writer, schema } => flush_edge_batch_parquet(schema, writer, data),
        EdgeWriter::Binary { writer, with_sort } => {
            write_edge_batch_binary(writer, *with_sort, data)
        }
    }
}

fn flush_edge_batch_parquet(
    schema: &Schema,
    writer: &mut ArrowWriter<OutputSink>,
    data: &EdgeAccumulator,
) -> Result<()> {
    let mut cols: Vec<ArrayRef> =
        Vec::with_capacity(if schema.fields().len() == 3 { 3 } else { 2 });
    cols.push(Arc::new(arrow_array::Int64Array::from(data.src.clone())) as ArrayRef);
    cols.push(Arc::new(arrow_array::Int64Array::from(data.dst.clone())) as ArrayRef);
    if schema.field_with_name("dest_sort_key").is_ok() {
        cols.push(Arc::new(arrow_array::Int64Array::from(data.sort.clone())) as ArrayRef);
    }
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), cols)?;
    writer.write(&batch)?;
    Ok(())
}

fn write_edge_bin_header(writer: &mut BufWriter<OutputSink>, with_sort: bool) -> Result<()> {
    writer.write_all(&EDGE_BIN_MAGIC)?;
    writer.write_all(&[EDGE_BIN_VERSION])?;
    let flags = if with_sort { EDGE_BIN_FLAG_SORT } else { 0 };
    writer.write_all(&[flags])?;
    writer.write_all(&[0u8, 0u8])?;
    Ok(())
}

fn write_edge_batch_binary(
    writer: &mut BufWriter<OutputSink>,
    with_sort: bool,
    data: &EdgeAccumulator,
) -> Result<()> {
    if with_sort && data.sort.len() != data.src.len() {
        return Err(anyhow!("edge sort buffer length mismatch"));
    }
    for idx in 0..data.src.len() {
        writer.write_all(&data.src[idx].to_le_bytes())?;
        writer.write_all(&data.dst[idx].to_le_bytes())?;
        if with_sort {
            let sort = data.sort[idx] as u32;
            writer.write_all(&sort.to_le_bytes())?;
        }
    }
    Ok(())
}

fn finish_edge_writer(writer: EdgeWriter) -> Result<()> {
    match writer {
        EdgeWriter::Parquet { writer, .. } => {
            let sink = writer.into_inner()?;
            sink.finish()?;
        }
        EdgeWriter::Binary { mut writer, .. } => {
            writer.flush()?;
            let sink = writer.into_inner().map_err(|_| anyhow!("edge sink flush failed"))?;
            sink.finish()?;
        }
    }
    Ok(())
}

fn flush_node_batch(
    schema: &Schema,
    writer: &mut ArrowWriter<OutputSink>,
    builder: &mut arrow_array::builder::BinaryBuilder,
) -> Result<()> {
    if builder.len() == 0 {
        return Ok(());
    }
    let array = builder.finish();
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)])?;
    writer.write(&batch)?;
    Ok(())
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

fn estimate_rows(table: &TableSpec) -> usize {
    table.paths.len().saturating_mul(1_000_000)
}

fn row_count_hint(table: &TableSpec) -> Option<u64> {
    if !matches!(table.format, TableFormat::Parquet) {
        return None;
    }
    let mut total = 0u64;
    for path in &table.paths {
        if is_cloud_path(path) {
            return None;
        }
        let file = fs::File::open(path).ok()?;
        let reader = parquet::file::serialized_reader::SerializedFileReader::new(file).ok()?;
        total = total.saturating_add(reader.metadata().file_metadata().num_rows() as u64);
    }
    if total == 0 {
        None
    } else {
        Some(total)
    }
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

fn key_from_array(array: &ArrayRef, row: usize, col: &ColumnSpec) -> KeyParseResult<Key> {
    if array.is_null(row) {
        return Err(KeyParseError::Null);
    }
    match col.source_dtype {
        ColumnType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let v = arr.value(row);
            if v < 0 {
                return Err(KeyParseError::Negative);
            }
            Ok(Key::Int(v as u64))
        }
        ColumnType::Int16 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int16Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let v = arr.value(row);
            if v < 0 {
                return Err(KeyParseError::Negative);
            }
            Ok(Key::Int(v as u64))
        }
        ColumnType::Int8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int8Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let v = arr.value(row);
            if v < 0 {
                return Err(KeyParseError::Negative);
            }
            Ok(Key::Int(v as u64))
        }
        ColumnType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let v = arr.value(row);
            if v < 0 {
                return Err(KeyParseError::Negative);
            }
            Ok(Key::Int(v as u64))
        }
        ColumnType::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            Ok(Key::Int(arr.value(row)))
        }
        ColumnType::UInt16 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt16Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            Ok(Key::Int(arr.value(row) as u64))
        }
        ColumnType::UInt8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt8Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            Ok(Key::Int(arr.value(row) as u64))
        }
        ColumnType::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .ok_or(KeyParseError::TypeMismatch)?;
            Ok(Key::Int(arr.value(row) as u64))
        }
        ColumnType::String => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let value = arr.value(row).as_bytes();
            if col.key_mode == Some(KeyMode::Uuid) {
                let uuid = parse_uuid_string(value)?;
                return Ok(Key::Uuid(uuid));
            }
            let bytes = value.to_vec();
            if col.key_mode == Some(KeyMode::Fixed) || col.fixed_len.is_some() {
                let len = col.fixed_len.unwrap_or(bytes.len());
                if bytes.len() != len {
                    return Err(KeyParseError::LengthMismatch);
                }
                return Ok(Key::FixedBytes(bytes));
            }
            Ok(Key::Bytes(bytes))
        }
        ColumnType::Bytes => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or(KeyParseError::TypeMismatch)?;
            let value = arr.value(row);
            if col.key_mode == Some(KeyMode::Uuid) {
                let uuid = parse_uuid_bytes(value)?;
                return Ok(Key::Uuid(uuid));
            }
            let bytes = value.to_vec();
            if col.key_mode == Some(KeyMode::Fixed) || col.fixed_len.is_some() {
                let len = col.fixed_len.unwrap_or(bytes.len());
                if bytes.len() != len {
                    return Err(KeyParseError::LengthMismatch);
                }
                return Ok(Key::FixedBytes(bytes));
            }
            Ok(Key::Bytes(bytes))
        }
        _ => Err(KeyParseError::UnsupportedType),
    }
}

fn key_from_array_for_target(
    array: &ArrayRef,
    row: usize,
    col: &ColumnSpec,
    target_col: &ColumnSpec,
    target: KeyType,
) -> KeyParseResult<Key> {
    if array.is_null(row) {
        return Err(KeyParseError::Null);
    }
    match target {
        KeyType::Int64 => match col.source_dtype {
            ColumnType::Int64
            | ColumnType::Int32
            | ColumnType::Int16
            | ColumnType::Int8
            | ColumnType::UInt64
            | ColumnType::UInt32
            | ColumnType::UInt16
            | ColumnType::UInt8 => key_from_array(array, row, col),
            ColumnType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let v = arr.value(row) as f64;
                let s = format_float_base10(v, 7)?;
                parse_base10_bytes(s.as_bytes()).map(Key::Int)
            }
            ColumnType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let v = arr.value(row);
                let s = format_float_base10(v, 15)?;
                parse_base10_bytes(s.as_bytes()).map(Key::Int)
            }
            ColumnType::Decimal128 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let scale = target_col.scale.unwrap_or(0);
                let s = decimal_to_string_i128(arr.value(row), scale);
                parse_base10_bytes(s.as_bytes()).map(Key::Int)
            }
            ColumnType::Decimal256 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let scale = target_col.scale.unwrap_or(0);
                let s = decimal_to_string_i256(arr.value(row), scale);
                parse_base10_bytes(s.as_bytes()).map(Key::Int)
            }
            ColumnType::String => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                if col.key_mode == Some(KeyMode::Uuid)
                    || col.key_mode == Some(KeyMode::Fixed)
                    || col.fixed_len.is_some()
                {
                    return Err(KeyParseError::TypeMismatch);
                }
                let bytes = arr.value(row).as_bytes();
                let v = parse_base10_bytes(bytes)?;
                Ok(Key::Int(v))
            }
            ColumnType::Bytes => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                if col.key_mode == Some(KeyMode::Uuid)
                    || col.key_mode == Some(KeyMode::Fixed)
                    || col.fixed_len.is_some()
                {
                    return Err(KeyParseError::TypeMismatch);
                }
                let bytes = arr.value(row);
                let v = parse_base10_bytes(bytes)?;
                Ok(Key::Int(v))
            }
            _ => Err(KeyParseError::UnsupportedType),
        },
        KeyType::FixedBytes { len } => match col.source_dtype {
            ColumnType::String => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                if col.key_mode == Some(KeyMode::Uuid) {
                    return Err(KeyParseError::TypeMismatch);
                }
                if col.key_mode.is_none() && col.fixed_len.is_none() {
                    return Err(KeyParseError::TypeMismatch);
                }
                let bytes = normalize_numeric_bytes(arr.value(row).as_bytes(), target_col)?;
                if bytes.len() != len {
                    return Err(KeyParseError::LengthMismatch);
                }
                Ok(Key::FixedBytes(bytes))
            }
            ColumnType::Bytes => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                if col.key_mode == Some(KeyMode::Uuid) {
                    return Err(KeyParseError::TypeMismatch);
                }
                if col.key_mode.is_none() && col.fixed_len.is_none() {
                    return Err(KeyParseError::TypeMismatch);
                }
                let bytes = normalize_numeric_bytes(arr.value(row), target_col)?;
                if bytes.len() != len {
                    return Err(KeyParseError::LengthMismatch);
                }
                Ok(Key::FixedBytes(bytes))
            }
            _ => Err(KeyParseError::UnsupportedType),
        },
        KeyType::VarBytes => match col.source_dtype {
            ColumnType::String => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                if col.key_mode == Some(KeyMode::Uuid)
                    || col.key_mode == Some(KeyMode::Fixed)
                    || col.fixed_len.is_some()
                {
                    return Err(KeyParseError::TypeMismatch);
                }
                Ok(Key::Bytes(arr.value(row).as_bytes().to_vec()))
            }
            ColumnType::Bytes => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                if col.key_mode == Some(KeyMode::Uuid)
                    || col.key_mode == Some(KeyMode::Fixed)
                    || col.fixed_len.is_some()
                {
                    return Err(KeyParseError::TypeMismatch);
                }
                Ok(Key::Bytes(arr.value(row).to_vec()))
            }
            ColumnType::Int64
            | ColumnType::Int32
            | ColumnType::Int16
            | ColumnType::Int8
            | ColumnType::UInt64
            | ColumnType::UInt32
            | ColumnType::UInt16
            | ColumnType::UInt8 => {
                let key = key_from_array(array, row, col)?;
                let Key::Int(v) = key else {
                    return Err(KeyParseError::TypeMismatch);
                };
                Ok(Key::Bytes(v.to_string().into_bytes()))
            }
            ColumnType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let s = format_float_base10(arr.value(row) as f64, 7)?;
                let bytes = normalize_numeric_bytes(s.as_bytes(), target_col)?;
                Ok(Key::Bytes(bytes))
            }
            ColumnType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let s = format_float_base10(arr.value(row), 15)?;
                let bytes = normalize_numeric_bytes(s.as_bytes(), target_col)?;
                Ok(Key::Bytes(bytes))
            }
            ColumnType::Decimal128 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let scale = target_col.scale.unwrap_or(0);
                let s = decimal_to_string_i128(arr.value(row), scale);
                let bytes = normalize_numeric_bytes(s.as_bytes(), target_col)?;
                Ok(Key::Bytes(bytes))
            }
            ColumnType::Decimal256 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let scale = target_col.scale.unwrap_or(0);
                let s = decimal_to_string_i256(arr.value(row), scale);
                let bytes = normalize_numeric_bytes(s.as_bytes(), target_col)?;
                Ok(Key::Bytes(bytes))
            }
            _ => Err(KeyParseError::UnsupportedType),
        },
        KeyType::Uuid => match col.source_dtype {
            ColumnType::String => {
                if col.key_mode != Some(KeyMode::Uuid) {
                    return Err(KeyParseError::TypeMismatch);
                }
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let uuid = parse_uuid_string(arr.value(row).as_bytes())?;
                Ok(Key::Uuid(uuid))
            }
            ColumnType::Bytes => {
                if col.key_mode != Some(KeyMode::Uuid) {
                    return Err(KeyParseError::TypeMismatch);
                }
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or(KeyParseError::TypeMismatch)?;
                let uuid = parse_uuid_bytes(arr.value(row))?;
                Ok(Key::Uuid(uuid))
            }
            _ => Err(KeyParseError::UnsupportedType),
        },
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

fn format_float_base10(value: f64, precision: usize) -> KeyParseResult<String> {
    if !value.is_finite() {
        return Err(KeyParseError::NotFinite);
    }
    if value < 0.0 {
        return Err(KeyParseError::Negative);
    }
    let mut s = format!("{:.*}", precision, value);
    if let Some(dot) = s.find('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.len() == dot + 1 {
            s.pop();
        }
    }
    Ok(s)
}

fn sample_key_value(array: &ArrayRef, row: usize, col: &ColumnSpec) -> Option<String> {
    if array.is_null(row) {
        return Some("<null>".to_string());
    }
    match col.source_dtype {
        ColumnType::String => {
            let arr = array.as_any().downcast_ref::<arrow_array::StringArray>()?;
            Some(format_bytes_sample(arr.value(row).as_bytes()))
        }
        ColumnType::Bytes => {
            let arr = array.as_any().downcast_ref::<arrow_array::BinaryArray>()?;
            Some(format_bytes_sample(arr.value(row)))
        }
        ColumnType::Int64 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int64Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::Int32 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int32Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::Int16 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int16Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::Int8 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Int8Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::UInt64 => {
            let arr = array.as_any().downcast_ref::<arrow_array::UInt64Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::UInt32 => {
            let arr = array.as_any().downcast_ref::<arrow_array::UInt32Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::UInt16 => {
            let arr = array.as_any().downcast_ref::<arrow_array::UInt16Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::UInt8 => {
            let arr = array.as_any().downcast_ref::<arrow_array::UInt8Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::Float32 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Float32Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::Float64 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Float64Array>()?;
            Some(arr.value(row).to_string())
        }
        ColumnType::Decimal128 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Decimal128Array>()?;
            let scale = col.scale.unwrap_or(0);
            Some(decimal_to_string_i128(arr.value(row), scale))
        }
        ColumnType::Decimal256 => {
            let arr = array.as_any().downcast_ref::<arrow_array::Decimal256Array>()?;
            let scale = col.scale.unwrap_or(0);
            Some(decimal_to_string_i256(arr.value(row), scale))
        }
        _ => None,
    }
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
    use std::collections::HashMap;

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
        let mut name_to_index = HashMap::new();
        name_to_index.insert("created_at".to_string(), 0);
        let row = RowView {
            row: 0,
            row_id: 0,
            cols: &cols,
            name_to_index: &name_to_index,
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
        let mut name_to_index = HashMap::new();
        name_to_index.insert("created_at".to_string(), 0);
        let row = RowView {
            row: 0,
            row_id: 0,
            cols: &cols,
            name_to_index: &name_to_index,
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
        let mut name_to_index = HashMap::new();
        name_to_index.insert("created_date".to_string(), 0);
        let row = RowView {
            row: 0,
            row_id: 0,
            cols: &cols,
            name_to_index: &name_to_index,
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
        let mut name_to_index = HashMap::new();
        name_to_index.insert("id".to_string(), 0);
        let row = RowView {
            row: 0,
            row_id: 0,
            cols: &cols,
            name_to_index: &name_to_index,
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
        let mut name_to_index = HashMap::new();
        name_to_index.insert("id".to_string(), 0);
        let row = RowView {
            row: 0,
            row_id: 0,
            cols: &cols,
            name_to_index: &name_to_index,
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
        )
        .unwrap();
        match values[0] {
            ValueRef::Int32(value) => assert_eq!(value, 123),
            _ => panic!("expected int32"),
        }
    }

    #[test]
    fn convert_string_timestamp_for_sort_key() {
        let col = ColumnSpec {
            name: "created_at".to_string(),
            source_dtype: ColumnType::String,
            ingested_dtype: ColumnType::Timestamp,
            fixed_len: None,
            timestamp_format: Some("rfc3339".to_string()),
            timestamp_unit: None,
            key_mode: None,
            precision: None,
            scale: None,
        };
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["2020-01-02T03:04:05Z"]));
        let value = value_from_array_with_spec(&arr, 0, &col).unwrap();
        assert_eq!(value.as_i64().unwrap(), 1_577_934_245);
    }
}

fn sort_key_to_u32(value: &JsonValue) -> Result<u32> {
    if let Some(v) = value.as_i64() {
        return Ok(v.max(0) as u32);
    }
    if let Some(v) = value.as_u64() {
        return Ok(v.min(u32::MAX as u64) as u32);
    }
    if let Some(v) = value.as_f64() {
        let v = v as f32;
        let bits = v.to_bits();
        let mapped = if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        };
        return Ok(mapped);
    }
    Err(anyhow!("unsupported sort key type"))
}

fn for_each_row(table: &TableSpec, mut f: impl FnMut(&RowView) -> Result<()>) -> Result<()> {
    let mut name_to_index = HashMap::with_capacity(table.columns.len());
    for (idx, col) in table.columns.iter().enumerate() {
        name_to_index.insert(col.name.clone(), idx);
    }
    let schema = table_schema(table)?;
    let mut global_row = 0u64;
    for path in &table.paths {
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
                    for row in 0..batch.num_rows() {
                        let row_view = RowView {
                            row,
                            row_id: global_row,
                            cols: &cols,
                            name_to_index: &name_to_index,
                        };
                        f(&row_view)?;
                        global_row += 1;
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
                    for row in 0..batch.num_rows() {
                        let row_view = RowView {
                            row,
                            row_id: global_row,
                            cols: &cols,
                            name_to_index: &name_to_index,
                        };
                        f(&row_view)?;
                        global_row += 1;
                    }
                }
            }
        }
    }
    Ok(())
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

fn sample_rowpack_context(
    source_cfg: Option<&RowPackConfig>,
    table: &TableSpec,
) -> Result<diskgraph_rowpack::RowPackContext> {
    let config = rowpack_config_for_table(source_cfg, table);
    let columns = rowpack_columns(table)?;
    let mut builder = RowPackContextBuilder::new(columns, config.clone());
    let limit = config.sample_rows();
    let mut arrays_f32: Vec<Vec<f32>> = Vec::new();
    let mut arrays_f64: Vec<Vec<f64>> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    let mut bytes: Vec<Vec<u8>> = Vec::new();
    for_each_row_with_limit(table, limit, |row| {
        arrays_f32.clear();
        arrays_f64.clear();
        strings.clear();
        bytes.clear();
        let mut values = Vec::with_capacity(table.columns.len());
        row_values_from_row(
            row,
            table,
            &mut values,
            &mut arrays_f32,
            &mut arrays_f64,
            &mut strings,
            &mut bytes,
        )?;
        builder.observe_row(&values)?;
        Ok(())
    })?;
    builder.finish()
}

fn for_each_row_with_limit(
    table: &TableSpec,
    limit: usize,
    mut f: impl FnMut(&RowView) -> Result<()>,
) -> Result<u64> {
    if limit == 0 {
        return Ok(0);
    }
    let mut name_to_index = HashMap::with_capacity(table.columns.len());
    for (idx, col) in table.columns.iter().enumerate() {
        name_to_index.insert(col.name.clone(), idx);
    }
    let schema = table_schema(table)?;
    let mut global_row = 0u64;
    'paths: for path in &table.paths {
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
                    for row in 0..batch.num_rows() {
                        let row_view = RowView {
                            row,
                            row_id: global_row,
                            cols: &cols,
                            name_to_index: &name_to_index,
                        };
                        f(&row_view)?;
                        global_row += 1;
                        if global_row as usize >= limit {
                            break 'paths;
                        }
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
                    for row in 0..batch.num_rows() {
                        let row_view = RowView {
                            row,
                            row_id: global_row,
                            cols: &cols,
                            name_to_index: &name_to_index,
                        };
                        f(&row_view)?;
                        global_row += 1;
                        if global_row as usize >= limit {
                            break 'paths;
                        }
                    }
                }
            }
        }
    }
    Ok(global_row)
}

fn for_each_batch(
    table: &TableSpec,
    mut f: impl FnMut(&[ArrayRef], usize, u64) -> Result<()>,
) -> Result<u64> {
    let schema = table_schema(table)?;
    let mut global_row = 0u64;
    for path in &table.paths {
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
                    let num_rows = batch.num_rows();
                    f(&cols, num_rows, global_row)?;
                    global_row += num_rows as u64;
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
                    let num_rows = batch.num_rows();
                    f(&cols, num_rows, global_row)?;
                    global_row += num_rows as u64;
                }
            }
        }
    }
    Ok(global_row)
}

struct RowView<'a> {
    row: usize,
    row_id: u64,
    cols: &'a [ArrayRef],
    name_to_index: &'a HashMap<String, usize>,
}

impl<'a> RowView<'a> {
    fn get_value(&self, col: &ColumnSpec) -> Result<JsonValue> {
        let idx = self
            .name_to_index
            .get(&col.name)
            .ok_or_else(|| anyhow!("missing column {}", col.name))?;
        let array = self
            .cols
            .get(*idx)
            .ok_or_else(|| anyhow!("missing column {}", col.name))?;
        value_from_array_with_spec(array, self.row, col)
    }

    fn sort_key(&self, table: &TableSpec) -> Result<Option<JsonValue>> {
        if let Some(weight) = &table.weight {
            let col = column_spec(table, weight)?;
            let value = self.get_value(col)?;
            if value.is_null() {
                return Ok(None);
            }
            return Ok(Some(value));
        }
        if let Some(ts) = &table.create_time {
            let col = column_spec(table, ts)?;
            let value = self.get_value(col)?;
            if value.is_null() {
                return Ok(None);
            }
            return Ok(Some(value));
        }
        Ok(None)
    }

}

enum TimestampAccessor<'a> {
    Seconds(&'a arrow_array::TimestampSecondArray),
    Millis(&'a arrow_array::TimestampMillisecondArray),
    Micros(&'a arrow_array::TimestampMicrosecondArray),
    Nanos(&'a arrow_array::TimestampNanosecondArray),
}

impl<'a> TimestampAccessor<'a> {
    fn value(&self, row: usize) -> (i64, u32) {
        match self {
            TimestampAccessor::Seconds(arr) => (arr.value(row), 0),
            TimestampAccessor::Millis(arr) => {
                let value = arr.value(row);
                let secs = value.div_euclid(1_000);
                let millis = value.rem_euclid(1_000) as u32;
                (secs, millis * 1_000_000)
            }
            TimestampAccessor::Micros(arr) => {
                let value = arr.value(row);
                let secs = value.div_euclid(1_000_000);
                let micros = value.rem_euclid(1_000_000) as u32;
                (secs, micros * 1_000)
            }
            TimestampAccessor::Nanos(arr) => {
                let value = arr.value(row);
                let secs = value.div_euclid(1_000_000_000);
                let nanos = value.rem_euclid(1_000_000_000) as u32;
                (secs, nanos)
            }
        }
    }
}

enum ColumnAccessor<'a> {
    Int64(&'a arrow_array::Int64Array),
    Int32(&'a arrow_array::Int32Array),
    Int16(&'a arrow_array::Int16Array),
    Int8(&'a arrow_array::Int8Array),
    UInt64(&'a arrow_array::UInt64Array),
    UInt32(&'a arrow_array::UInt32Array),
    UInt16(&'a arrow_array::UInt16Array),
    UInt8(&'a arrow_array::UInt8Array),
    Float64(&'a arrow_array::Float64Array),
    Float32(&'a arrow_array::Float32Array),
    Bool(&'a arrow_array::BooleanArray),
    String(&'a arrow_array::StringArray),
    Bytes(&'a arrow_array::BinaryArray),
    Date(&'a arrow_array::Date32Array),
    Timestamp(TimestampAccessor<'a>),
    Decimal128(&'a arrow_array::Decimal128Array),
    Decimal256(&'a arrow_array::Decimal256Array),
    Float32List {
        list: &'a arrow_array::ListArray,
        values: &'a arrow_array::Float32Array,
        values_has_nulls: bool,
    },
    Float64List {
        list: &'a arrow_array::ListArray,
        values: &'a arrow_array::Float64Array,
        values_has_nulls: bool,
    },
}

struct RowAccessor<'a> {
    row: usize,
    accessors: &'a [ColumnAccessor<'a>],
    columns: &'a [ColumnSpec],
    strings: &'a mut Vec<String>,
    bytes: &'a mut Vec<Vec<u8>>,
    arrays_f32: &'a mut Vec<Vec<f32>>,
    arrays_f64: &'a mut Vec<Vec<f64>>,
}

impl<'a> RowValueAccess for RowAccessor<'a> {
    fn get_int64(&mut self, idx: usize) -> Result<Option<i64>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Int64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_i64(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_i64(text))
                }
            }
            Some(ColumnAccessor::Float64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as i64))
                }
            }
            Some(ColumnAccessor::Float32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as i64))
                }
            }
            Some(ColumnAccessor::Decimal128(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as i64))
                }
            }
            Some(ColumnAccessor::Decimal256(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let value = arr.value(self.row);
                    Ok(value.to_i128().map(|v| v as i64))
                }
            }
            _ => Err(anyhow!("rowpack int64 accessor mismatch")),
        }
    }

    fn get_int32(&mut self, idx: usize) -> Result<Option<i32>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Int32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Int16(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as i32))
                }
            }
            Some(ColumnAccessor::Int8(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as i32))
                }
            }
            Some(ColumnAccessor::Int64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as i32))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_i32(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_i32(text))
                }
            }
            _ => Err(anyhow!("rowpack int32 accessor mismatch")),
        }
    }

    fn get_uint64(&mut self, idx: usize) -> Result<Option<u64>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::UInt64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::UInt32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as u64))
                }
            }
            Some(ColumnAccessor::UInt16(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as u64))
                }
            }
            Some(ColumnAccessor::UInt8(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as u64))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_u64(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_u64(text))
                }
            }
            _ => Err(anyhow!("rowpack uint64 accessor mismatch")),
        }
    }

    fn get_uint32(&mut self, idx: usize) -> Result<Option<u32>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::UInt32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::UInt16(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as u32))
                }
            }
            Some(ColumnAccessor::UInt8(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as u32))
                }
            }
            Some(ColumnAccessor::UInt64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as u32))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_u32(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_u32(text))
                }
            }
            _ => Err(anyhow!("rowpack uint32 accessor mismatch")),
        }
    }

    fn get_float64(&mut self, idx: usize) -> Result<Option<f64>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Float64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Float32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as f64))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_f64(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_f64(text))
                }
            }
            _ => Err(anyhow!("rowpack float64 accessor mismatch")),
        }
    }

    fn get_float32(&mut self, idx: usize) -> Result<Option<f32>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Float32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Float64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row) as f32))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_f32(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_f32(text))
                }
            }
            _ => Err(anyhow!("rowpack float32 accessor mismatch")),
        }
    }

    fn get_bool(&mut self, idx: usize) -> Result<Option<bool>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Bool(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_bool(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_bool(text))
                }
            }
            _ => Err(anyhow!("rowpack bool accessor mismatch")),
        }
    }

    fn get_string(&mut self, idx: usize) -> Result<Option<&str>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(Some(text))
                }
            }
            Some(ColumnAccessor::Int64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.strings.push(arr.value(self.row).to_string());
                    Ok(self.strings.last().map(|s| s.as_str()))
                }
            }
            Some(ColumnAccessor::Int32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.strings.push(arr.value(self.row).to_string());
                    Ok(self.strings.last().map(|s| s.as_str()))
                }
            }
            Some(ColumnAccessor::UInt64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.strings.push(arr.value(self.row).to_string());
                    Ok(self.strings.last().map(|s| s.as_str()))
                }
            }
            Some(ColumnAccessor::UInt32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.strings.push(arr.value(self.row).to_string());
                    Ok(self.strings.last().map(|s| s.as_str()))
                }
            }
            Some(ColumnAccessor::Float64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.strings.push(arr.value(self.row).to_string());
                    Ok(self.strings.last().map(|s| s.as_str()))
                }
            }
            Some(ColumnAccessor::Float32(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.strings.push(arr.value(self.row).to_string());
                    Ok(self.strings.last().map(|s| s.as_str()))
                }
            }
            _ => Err(anyhow!("rowpack string accessor mismatch")),
        }
    }

    fn get_bytes(&mut self, idx: usize) -> Result<Option<&[u8]>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.bytes.push(arr.value(self.row).as_bytes().to_vec());
                    Ok(self.bytes.last().map(|v| v.as_slice()))
                }
            }
            Some(ColumnAccessor::Int64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.bytes.push(arr.value(self.row).to_string().into_bytes());
                    Ok(self.bytes.last().map(|v| v.as_slice()))
                }
            }
            Some(ColumnAccessor::UInt64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    self.bytes.push(arr.value(self.row).to_string().into_bytes());
                    Ok(self.bytes.last().map(|v| v.as_slice()))
                }
            }
            _ => Err(anyhow!("rowpack bytes accessor mismatch")),
        }
    }

    fn get_date(&mut self, idx: usize) -> Result<Option<i32>> {
        let col = self
            .columns
            .get(idx)
            .ok_or_else(|| anyhow!("rowpack date missing column"))?;
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Date(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::Timestamp(accessor)) => {
                if accessor_is_null(accessor, self.row) {
                    Ok(None)
                } else {
                    let (secs, _) = accessor.value(self.row);
                    Ok(Some(days_from_secs(secs)))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_date_text(arr.value(self.row), col.timestamp_format.as_deref()))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_date_text(text, col.timestamp_format.as_deref()))
                }
            }
            Some(ColumnAccessor::Int64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let value = arr.value(self.row);
                    let days = if let Some(unit) = col.timestamp_unit.as_ref() {
                        let (secs, _) = timestamp_from_int(value, unit.clone());
                        days_from_secs(secs)
                    } else {
                        value as i32
                    };
                    Ok(Some(days))
                }
            }
            _ => Err(anyhow!("rowpack date accessor mismatch")),
        }
    }

    fn get_timestamp(&mut self, idx: usize) -> Result<Option<(i64, u32)>> {
        let col = self
            .columns
            .get(idx)
            .ok_or_else(|| anyhow!("rowpack timestamp missing column"))?;
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Timestamp(accessor)) => {
                if accessor_is_null(accessor, self.row) {
                    Ok(None)
                } else {
                    Ok(Some(accessor.value(self.row)))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_timestamp_text(
                        arr.value(self.row),
                        col.timestamp_format.as_deref(),
                    ))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_timestamp_text(text, col.timestamp_format.as_deref()))
                }
            }
            Some(ColumnAccessor::Int64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let unit = col
                        .timestamp_unit
                        .clone()
                        .unwrap_or(TimestampUnit::Seconds);
                    Ok(Some(timestamp_from_int(arr.value(self.row), unit)))
                }
            }
            Some(ColumnAccessor::UInt64(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let unit = col
                        .timestamp_unit
                        .clone()
                        .unwrap_or(TimestampUnit::Seconds);
                    Ok(Some(timestamp_from_int(arr.value(self.row) as i64, unit)))
                }
            }
            _ => Err(anyhow!("rowpack timestamp accessor mismatch")),
        }
    }

    fn get_decimal128(&mut self, idx: usize) -> Result<Option<i128>> {
        let col = self
            .columns
            .get(idx)
            .ok_or_else(|| anyhow!("rowpack decimal128 missing column"))?;
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Decimal128(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row)))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_decimal_i128(
                        arr.value(self.row),
                        col.scale.unwrap_or(0),
                    ))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_decimal_i128(text, col.scale.unwrap_or(0)))
                }
            }
            _ => Err(anyhow!("rowpack decimal128 accessor mismatch")),
        }
    }

    fn get_decimal256(&mut self, idx: usize) -> Result<Option<[u8; 32]>> {
        let col = self
            .columns
            .get(idx)
            .ok_or_else(|| anyhow!("rowpack decimal256 missing column"))?;
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Decimal256(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(self.row).to_le_bytes()))
                }
            }
            Some(ColumnAccessor::String(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    Ok(parse_decimal_i256(
                        arr.value(self.row),
                        col.scale.unwrap_or(0),
                    )
                    .map(|v| v.to_le_bytes()))
                }
            }
            Some(ColumnAccessor::Bytes(arr)) => {
                if arr.is_null(self.row) {
                    Ok(None)
                } else {
                    let text = match std::str::from_utf8(arr.value(self.row)) {
                        Ok(text) => text,
                        Err(_) => return Ok(None),
                    };
                    Ok(parse_decimal_i256(text, col.scale.unwrap_or(0)).map(|v| v.to_le_bytes()))
                }
            }
            _ => Err(anyhow!("rowpack decimal256 accessor mismatch")),
        }
    }

    fn get_float32_array(&mut self, idx: usize) -> Result<Option<&[f32]>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Float32List {
                list,
                values,
                values_has_nulls,
            }) => {
                if list.is_null(self.row) {
                    return Ok(None);
                }
                let offsets = list.value_offsets();
                let start = offsets[self.row] as usize;
                let end = offsets[self.row + 1] as usize;
                if !*values_has_nulls {
                    let slice = &values.values().as_ref()[start..end];
                    Ok(Some(slice))
                } else {
                    let mut vec = Vec::with_capacity(end.saturating_sub(start));
                    for idx in start..end {
                        if values.is_null(idx) {
                            vec.push(0.0);
                        } else {
                            vec.push(values.value(idx));
                        }
                    }
                    self.arrays_f32.push(vec);
                    let slice_ptr = self.arrays_f32.last().unwrap().as_slice() as *const [f32];
                    let slice = unsafe { &*slice_ptr };
                    Ok(Some(slice))
                }
            }
            _ => Err(anyhow!("rowpack float32 array accessor mismatch")),
        }
    }

    fn get_float64_array(&mut self, idx: usize) -> Result<Option<&[f64]>> {
        match self.accessors.get(idx) {
            Some(ColumnAccessor::Float64List {
                list,
                values,
                values_has_nulls,
            }) => {
                if list.is_null(self.row) {
                    return Ok(None);
                }
                let offsets = list.value_offsets();
                let start = offsets[self.row] as usize;
                let end = offsets[self.row + 1] as usize;
                if !*values_has_nulls {
                    let slice = &values.values().as_ref()[start..end];
                    Ok(Some(slice))
                } else {
                    let mut vec = Vec::with_capacity(end.saturating_sub(start));
                    for idx in start..end {
                        if values.is_null(idx) {
                            vec.push(0.0);
                        } else {
                            vec.push(values.value(idx));
                        }
                    }
                    self.arrays_f64.push(vec);
                    let slice_ptr = self.arrays_f64.last().unwrap().as_slice() as *const [f64];
                    let slice = unsafe { &*slice_ptr };
                    Ok(Some(slice))
                }
            }
            _ => Err(anyhow!("rowpack float64 array accessor mismatch")),
        }
    }
}

fn build_accessors<'a>(cols: &'a [ArrayRef], table: &TableSpec) -> Result<Vec<ColumnAccessor<'a>>> {
    if cols.len() != table.columns.len() {
        return Err(anyhow!("column count mismatch"));
    }
    let mut accessors = Vec::with_capacity(cols.len());
    for (idx, col) in table.columns.iter().enumerate() {
        let array = cols
            .get(idx)
            .ok_or_else(|| anyhow!("missing column {}", col.name))?;
        let accessor = match col.source_dtype {
            ColumnType::Int64 => ColumnAccessor::Int64(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?,
            ),
            ColumnType::Int32 => ColumnAccessor::Int32(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?,
            ),
            ColumnType::Int16 => ColumnAccessor::Int16(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?,
            ),
            ColumnType::Int8 => ColumnAccessor::Int8(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?,
            ),
            ColumnType::UInt64 => ColumnAccessor::UInt64(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?,
            ),
            ColumnType::UInt32 => ColumnAccessor::UInt32(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?,
            ),
            ColumnType::UInt16 => ColumnAccessor::UInt16(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?,
            ),
            ColumnType::UInt8 => ColumnAccessor::UInt8(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?,
            ),
            ColumnType::Float64 => ColumnAccessor::Float64(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?,
            ),
            ColumnType::Float32 => ColumnAccessor::Float32(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?,
            ),
            ColumnType::Bool => ColumnAccessor::Bool(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?,
            ),
            ColumnType::String => ColumnAccessor::String(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string cast"))?,
            ),
            ColumnType::Bytes => ColumnAccessor::Bytes(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or_else(|| anyhow!("binary cast"))?,
            ),
            ColumnType::Date => ColumnAccessor::Date(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Date32Array>()
                    .ok_or_else(|| anyhow!("date cast"))?,
            ),
            ColumnType::Timestamp => {
                let accessor = match array.data_type() {
                    DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampSecondArray>()
                            .ok_or_else(|| anyhow!("timestamp second cast"))?;
                        ColumnAccessor::Timestamp(TimestampAccessor::Seconds(arr))
                    }
                    DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                            .ok_or_else(|| anyhow!("timestamp millisecond cast"))?;
                        ColumnAccessor::Timestamp(TimestampAccessor::Millis(arr))
                    }
                    DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                            .ok_or_else(|| anyhow!("timestamp microsecond cast"))?;
                        ColumnAccessor::Timestamp(TimestampAccessor::Micros(arr))
                    }
                    DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                            .ok_or_else(|| anyhow!("timestamp nanosecond cast"))?;
                        ColumnAccessor::Timestamp(TimestampAccessor::Nanos(arr))
                    }
                    _ => return Err(anyhow!("timestamp type mismatch")),
                };
                accessor
            }
            ColumnType::Decimal128 => ColumnAccessor::Decimal128(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?,
            ),
            ColumnType::Decimal256 => ColumnAccessor::Decimal256(
                array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?,
            ),
            ColumnType::Float32Array => {
                let list = array
                    .as_any()
                    .downcast_ref::<arrow_array::ListArray>()
                    .ok_or_else(|| anyhow!("list cast"))?;
                let values = list
                    .values()
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 list cast"))?;
                ColumnAccessor::Float32List {
                    list,
                    values,
                    values_has_nulls: values.null_count() > 0,
                }
            }
            ColumnType::Float64Array => {
                let list = array
                    .as_any()
                    .downcast_ref::<arrow_array::ListArray>()
                    .ok_or_else(|| anyhow!("list cast"))?;
                let values = list
                    .values()
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 list cast"))?;
                ColumnAccessor::Float64List {
                    list,
                    values,
                    values_has_nulls: values.null_count() > 0,
                }
            }
        };
        accessors.push(accessor);
    }
    Ok(accessors)
}

fn accessor_is_null(accessor: &TimestampAccessor<'_>, row: usize) -> bool {
    match accessor {
        TimestampAccessor::Seconds(arr) => arr.is_null(row),
        TimestampAccessor::Millis(arr) => arr.is_null(row),
        TimestampAccessor::Micros(arr) => arr.is_null(row),
        TimestampAccessor::Nanos(arr) => arr.is_null(row),
    }
}

fn row_values_from_row<'a>(
    row: &'a RowView<'a>,
    table: &'a TableSpec,
    values: &mut Vec<ValueRef<'a>>,
    arrays_f32: &'a mut Vec<Vec<f32>>,
    arrays_f64: &'a mut Vec<Vec<f64>>,
    strings: &'a mut Vec<String>,
    bytes: &'a mut Vec<Vec<u8>>,
) -> Result<()> {
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
                    values.push(ValueRef::String(arr.value(row.row)));
                }
                ColumnType::Bytes => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .ok_or_else(|| anyhow!("binary cast"))?;
                    values.push(ValueRef::Bytes(arr.value(row.row)));
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
                    if values_arr.null_count() == 0 {
                        let slice = &values_arr.values().as_ref()[start..end];
                        values.push(ValueRef::Float32Array(slice));
                    } else {
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
                    if values_arr.null_count() == 0 {
                        let slice = &values_arr.values().as_ref()[start..end];
                        values.push(ValueRef::Float64Array(slice));
                    } else {
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
                bytes.push(data.to_vec());
                let slice_ptr = bytes.last().unwrap().as_slice() as *const [u8];
                let slice = unsafe { &*slice_ptr };
                values.push(ValueRef::Bytes(slice));
            }
            _ => {
                values.push(ValueRef::Null);
            }
        }
    }
    Ok(())
}

fn value_from_array_with_spec(
    array: &ArrayRef,
    row: usize,
    col: &ColumnSpec,
) -> Result<JsonValue> {
    if array.is_null(row) {
        return Ok(JsonValue::Null);
    }
    if col.source_dtype == col.ingested_dtype {
        return value_from_array(array, row);
    }
    match (col.source_dtype.clone(), col.ingested_dtype.clone()) {
        (ColumnType::String, ColumnType::Timestamp)
        | (ColumnType::Bytes, ColumnType::Timestamp) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            if let Some((secs, _)) = parse_timestamp_text(text, col.timestamp_format.as_deref()) {
                Ok(JsonValue::from(secs))
            } else {
                Ok(JsonValue::Null)
            }
        }
        (ColumnType::String, ColumnType::Date) | (ColumnType::Bytes, ColumnType::Date) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            if let Some(days) = parse_date_text(text, col.timestamp_format.as_deref()) {
                Ok(JsonValue::from(days))
            } else {
                Ok(JsonValue::Null)
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
            let value = read_i64(array, row, &col.source_dtype)?;
            let unit = col
                .timestamp_unit
                .clone()
                .unwrap_or(TimestampUnit::Seconds);
            let (secs, _) = timestamp_from_int(value, unit);
            Ok(JsonValue::from(secs))
        }
        (ColumnType::Int8, ColumnType::Date)
        | (ColumnType::Int16, ColumnType::Date)
        | (ColumnType::Int32, ColumnType::Date)
        | (ColumnType::Int64, ColumnType::Date)
        | (ColumnType::UInt8, ColumnType::Date)
        | (ColumnType::UInt16, ColumnType::Date)
        | (ColumnType::UInt32, ColumnType::Date)
        | (ColumnType::UInt64, ColumnType::Date) => {
            let value = read_i64(array, row, &col.source_dtype)?;
            let days = if let Some(unit) = col.timestamp_unit.as_ref() {
                let (secs, _) = timestamp_from_int(value, unit.clone());
                days_from_secs(secs)
            } else {
                value as i32
            };
            Ok(JsonValue::from(days))
        }
        (ColumnType::String, ColumnType::Int64) | (ColumnType::Bytes, ColumnType::Int64) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_i64(text).map(JsonValue::from).unwrap_or(JsonValue::Null))
        }
        (ColumnType::String, ColumnType::Int32) | (ColumnType::Bytes, ColumnType::Int32) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_i32(text).map(JsonValue::from).unwrap_or(JsonValue::Null))
        }
        (ColumnType::String, ColumnType::UInt64) | (ColumnType::Bytes, ColumnType::UInt64) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_u64(text).map(JsonValue::from).unwrap_or(JsonValue::Null))
        }
        (ColumnType::String, ColumnType::UInt32) | (ColumnType::Bytes, ColumnType::UInt32) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_u32(text).map(JsonValue::from).unwrap_or(JsonValue::Null))
        }
        (ColumnType::String, ColumnType::Float64) | (ColumnType::Bytes, ColumnType::Float64) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_f64(text).map(JsonValue::from).unwrap_or(JsonValue::Null))
        }
        (ColumnType::String, ColumnType::Float32) | (ColumnType::Bytes, ColumnType::Float32) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_f32(text).map(JsonValue::from).unwrap_or(JsonValue::Null))
        }
        (ColumnType::String, ColumnType::Bool) | (ColumnType::Bytes, ColumnType::Bool) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_bool(text).map(JsonValue::from).unwrap_or(JsonValue::Null))
        }
        (ColumnType::String, ColumnType::Decimal128)
        | (ColumnType::Bytes, ColumnType::Decimal128) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_decimal_i128(text, col.scale.unwrap_or(0))
                .map(|v| JsonValue::from(v.to_string()))
                .unwrap_or(JsonValue::Null))
        }
        (ColumnType::String, ColumnType::Decimal256)
        | (ColumnType::Bytes, ColumnType::Decimal256) => {
            let text = read_utf8(array, row, &col.source_dtype)?;
            Ok(parse_decimal_i256(text, col.scale.unwrap_or(0))
                .map(|v| JsonValue::from(v.to_string()))
                .unwrap_or(JsonValue::Null))
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
            let value = value_from_array(array, row)?;
            match value {
                JsonValue::Null => Ok(JsonValue::Null),
                JsonValue::String(s) => Ok(JsonValue::from(s)),
                JsonValue::Number(n) => Ok(JsonValue::from(n.to_string())),
                JsonValue::Bool(v) => Ok(JsonValue::from(v.to_string())),
                other => Ok(JsonValue::from(other.to_string())),
            }
        }
        _ => value_from_array(array, row),
    }
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


fn table_rows(ctx: &IngestContext, name: &str) -> u64 {
    ctx.table_rows.get(name).copied().unwrap_or(0)
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
