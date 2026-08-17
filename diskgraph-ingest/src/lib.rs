//! Diskgraph ingest: convert raw CSV/Parquet sources into index-ready artifacts.

pub mod binary_label;
pub mod coercion;
pub mod col_stats;
pub mod col_stats_export;
pub mod column_inference;
pub mod config;
pub mod config_converter;
pub mod consts;
pub mod databricks;
pub mod disk_manager;
pub mod errors;
pub mod graph_config;
pub mod index_metadata;
pub mod io;
pub mod mapping;
pub mod progress;
pub mod sample_weight;
pub mod scratch;
pub mod snapshot_mirror;
mod source_memory;
pub mod stage_extension;
pub mod stats;
pub mod stats_publish;
pub mod success_marker;
#[cfg(test)]
mod tests_unit;
pub mod ts_grouping;
pub mod utils;

pub use crate::column_inference::infer_stage_columns;
use crate::config::PeakMemStats;
pub use crate::config::{IngestOptions, IngestPhase, ResolveColumnSpec, ResolveKind, ALL_PHASES};
use crate::consts::*;
pub use crate::errors::{
    CoercionError, CoercionWarning, ConvertError, KeyParseError, KeyParseResult,
};
use crate::graph_config::{
    ColumnSpec, DirectionSpec, IngestedType, ObjectStoreObject, SemanticType, TableFormat,
    TableSpec, TimeSpec, WeightSpec,
};
pub use crate::io::Dest;
use crate::io::{
    build_input_transfer_for_table, download_cloud_bytes_condition, is_cloud_path,
    require_local_output, resolve_local_paths, OutputSink, OutputTarget,
};
pub use crate::io::{build_transfer_for_cloud_path, object_path_from_cloud_url};
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
pub use crate::stage_extension::{
    ingest_stage_extension_with_resolution, ingest_stage_extension_with_resolution_and_sidecars,
    ResolveSpec, StageExtensionResult, StageSidecarColumns, StageSpec,
};
pub use crate::stats::{
    IngestReport, IngestSourceConfig, IngestStats, KeyFailureCounts, KeyFailureState, PerfStats,
};
use anyhow::{anyhow, ensure, Context, Result};
use arrow_array::{make_array, Array, ArrayRef, RecordBatch};
use arrow_data::transform::MutableArrayData;
use arrow_schema::{DataType, Field, Schema};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::Bytes;
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, EdgeSortKeyMetadata,
    EdgeSortKeyTimestampEncoding, EdgeSortKeyTimestampKind, EdgeSortKeyTimestampMetadata,
    EdgeSortKeyWeightEncoding, EdgeSortKeyWeightKind, EdgeSortKeyWeightMetadata, EdgeStorageFormat,
    NodeDataCompression, NodeShardMetadata, NodeTypeMetadata, PrimaryKeyMap,
};
use diskgraph_fs::{CloudObjectCondition, CloudTransfer, PooledBytes};
use diskgraph_index::builder::{
    EdgeBufferPool as IndexEdgeBufferPool, MemoryBudget, MemoryBudgetGuard,
};
use diskgraph_index::graph::{
    committed_shard_expected_nodes, index_shard_commit_path, shard_output_is_committed,
    GraphBuildProgress, GraphBuildStage, GraphDataSource, GraphIndexBuilder, GraphProgressSink,
};
use diskgraph_rowpack::{
    BinaryStringHint, IngestedType as RowpackType, RowAccess, RowAccessHints, RowAccessStats,
    RowPackConfig, RowPackContextBuilder, RowPackEncoder, RowPackRowLens,
};
use epserde::prelude::Serialize as EpSerialize;
use fasteval::{eval_compiled_ref, Compiler, Evaler, Parser, Slab};
use gxhash::GxHasher;
use half::f16;
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use memmap2::Mmap;
use num_traits::{Signed, ToPrimitive};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use parquet::basic::{Encoding, PageType, Type as PhysicalType};
use parquet::column::page::{Page, PageReader};
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::reader::{ChunkReader, Length, SerializedPageReader};
use pprof::protos::Message;
use pprof::ProfilerGuard;
use ptr_hash::DefaultPtrHash;
use ptr_hash::PtrHashScratch;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self as std_io, BufWriter, Read, Write};
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use uuid_simd::{parse_hyphenated, parse_simple, AsOut};

pub use graph_config::GraphConfig;

struct PhaseMarkerChain {
    first_incomplete: IngestPhase,
    encode_complete: bool,
}

fn phase_marker_chain(root: &Path) -> Result<PhaseMarkerChain> {
    let mut first_missing: Option<IngestPhase> = None;
    let mut encode_complete = false;
    for phase in ALL_PHASES {
        let marker = root
            .join("_tmp")
            .join(format!("_phase_complete_{}", phase.name()));
        if marker.exists() {
            if let Some(missing) = first_missing {
                anyhow::bail!(
                    "invalid phase marker chain: '{}' exists after missing '{}'",
                    phase.name(),
                    missing.name(),
                );
            }
            encode_complete |= phase == IngestPhase::Encode;
        } else if first_missing.is_none() {
            first_missing = Some(phase);
        }
    }
    Ok(PhaseMarkerChain {
        first_incomplete: first_missing.unwrap_or(IngestPhase::Index),
        encode_complete,
    })
}

pub fn first_incomplete_phase(root: &Path) -> Result<IngestPhase> {
    Ok(phase_marker_chain(root)?.first_incomplete)
}

/// Return the safe automatic resume point for an unpublished local ingest.
pub fn initial_resume_phase(root: &Path) -> Result<Option<IngestPhase>> {
    if root.join(success_marker::SUCCESS_MARKER_FILENAME).exists() {
        return Ok(None);
    }
    let chain = phase_marker_chain(root)?;
    Ok(chain.encode_complete.then_some(chain.first_incomplete))
}

/// Return validated durable Index commits when an unpublished ingest is
/// eligible for a bounded Index-resume retry.
pub fn validated_index_resume_progress(root: &Path) -> Result<Option<Vec<String>>> {
    if initial_resume_phase(root)? != Some(IngestPhase::Index)
        || phase_marker_exists(root, IngestPhase::Index.name())
    {
        return Ok(None);
    }
    let commits = diskgraph_index::graph::validated_index_shard_commits(root)?;
    Ok((!commits.is_empty()).then_some(commits))
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct PhysicalFieldSignature {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct SourceSchemaSignature {
    pub tables: BTreeMap<String, Vec<PhysicalFieldSignature>>,
}

fn remove_temp_entry(path: &Path) -> std::io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

#[cfg(test)]
mod cleanup_temp_tests {
    use super::remove_temp_entry;
    use std::fs;

    #[test]
    fn removes_both_files_and_directories() {
        let root = tempfile::tempdir().unwrap();
        let marker = root.path().join("_phase_complete_encode");
        let stage = root.path().join("encode");
        fs::write(&marker, b"").unwrap();
        fs::create_dir(&stage).unwrap();
        fs::write(stage.join("part.bin"), b"data").unwrap();

        remove_temp_entry(&marker).unwrap();
        remove_temp_entry(&stage).unwrap();

        assert!(!marker.exists());
        assert!(!stage.exists());
    }
}

fn physical_schema_signature(schema: &Schema) -> Vec<PhysicalFieldSignature> {
    schema
        .fields()
        .iter()
        .map(|field| PhysicalFieldSignature {
            name: field.name().clone(),
            data_type: format!("{:?}", field.data_type()),
            nullable: field.is_nullable(),
        })
        .collect()
}

fn start_databricks_helper(
    source: &GraphConfig,
) -> Result<Option<crate::databricks::DatabricksHelper>> {
    if !source
        .tables
        .iter()
        .any(|table| matches!(table.source, graph_config::DataSource::Databricks { .. }))
    {
        return Ok(None);
    }
    let mut helper = crate::databricks::DatabricksHelper::spawn()?;
    let hello = helper.hello()?;
    if hello.protocol != 1 || hello.connector != "4.3.0" || hello.thrift != "0.22.0" {
        return Err(anyhow!("Databricks helper protocol version mismatch"));
    }
    Ok(Some(helper))
}

struct ParquetInput {
    path: String,
    condition: Option<CloudObjectCondition>,
}

/// Rows for the encoding context, taken as a stride across the table's files.
///
/// Every encoding decision is made from this sample, and value vocabularies are
/// partitioned across files, so the head of one file shows a fraction of a
/// column's distinct values however many rows it reads. On doordash `orders`,
/// 10k head rows find 1,029 of 9,279 `ALL_TAGS` values; the same budget spread
/// over six files finds 2,085.
///
/// This reads through its own readers and consumes nothing from the encode
/// stream, so encode still reads every file whole. These rows are therefore
/// read twice, which for `sample_limit` rows out of a whole table is cheaper
/// than any coupling that would avoid it. Loading without a resident budget
/// takes the ranged-read path rather than downloading whole cloud objects.
/// Longest value in a variable-width column. An upper bound on the key
/// `pk_map` hashes, since `parse_bytes_key` only ever shrinks its input.
fn max_binary_value_len(array: &ArrayRef) -> usize {
    fn from_offsets<O: arrow_array::OffsetSizeTrait>(offsets: &[O]) -> usize {
        offsets
            .windows(2)
            .map(|w| (w[1] - w[0]).as_usize())
            .max()
            .unwrap_or(0)
    }
    match array.data_type() {
        DataType::Utf8 => from_offsets(
            array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .map(|a| a.value_offsets())
                .unwrap_or(&[]),
        ),
        DataType::LargeUtf8 => from_offsets(
            array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .map(|a| a.value_offsets())
                .unwrap_or(&[]),
        ),
        DataType::Binary => from_offsets(
            array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .map(|a| a.value_offsets())
                .unwrap_or(&[]),
        ),
        DataType::LargeBinary => from_offsets(
            array
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .map(|a| a.value_offsets())
                .unwrap_or(&[]),
        ),
        _ => 0,
    }
}

fn sample_context_batches(
    table: &TableSpec,
    transfer: Option<&CloudTransfer>,
    source_budget: &Arc<crate::source_memory::ByteBudget>,
    sample_limit: usize,
) -> Result<Vec<RecordBatch>> {
    if !matches!(
        table.source,
        graph_config::DataSource::Parquet { .. }
            | graph_config::DataSource::ObjectStoreParquet { .. }
    ) {
        return Ok(Vec::new());
    }
    let inputs = parquet_inputs(table);
    if inputs.is_empty() || sample_limit == 0 {
        return Ok(Vec::new());
    }
    let files = inputs.len().min(CONTEXT_STRIDE_PARTITIONS);
    let per_file = sample_limit.div_ceil(files).max(1);
    // Floor, with `take(files)` below: `step_by` visits `ceil(N/step)` files, so
    // a ceiling step visits fewer than `files` and the sample comes up short --
    // 200 files at a 100k budget collected 78k rows.
    let step = (inputs.len() / files).max(1);

    let mut batches = Vec::new();
    let mut rows = 0usize;
    for input in inputs.iter().step_by(step).take(files) {
        if rows >= sample_limit {
            break;
        }
        let resident = ResidentFile::load(
            &input.path,
            TableFormat::Parquet,
            transfer,
            None,
            input.condition.clone(),
        )?;
        let Some(row_groups) = plan_resident_parquet_partitions(&resident)?
            .into_iter()
            .next()
        else {
            continue;
        };
        let resident = Arc::new(resident);
        let (plans, _) = resident_partition_memory_plans(
            &resident,
            &row_groups,
            source_budget.max_input_chunk_bytes(),
            source_budget,
        )?;
        let mut reader = BoundedParquetBatchReader::partition(resident, plans);
        let mut taken = 0usize;
        let want = per_file.min(sample_limit - rows);
        while taken < want {
            let Some(batch) = reader.next() else {
                break;
            };
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let take = (want - taken).min(batch.num_rows());
            // A slice is a view that pins the parent's full column buffers, and
            // these are held until the context and text stats are built.
            batches.push(compact_record_batch_prefix(&batch, take)?);
            taken += take;
        }
        rows += taken;
    }
    Ok(batches)
}

fn parquet_inputs(table: &TableSpec) -> Vec<ParquetInput> {
    table
        .source_objects()
        .map(|object| ParquetInput {
            path: object.path.to_owned(),
            condition: object.condition(),
        })
        .collect()
}

/// Preserve the legacy `parquet.paths` cloud contract without preserving a
/// second Parquet reader. Capture one exact identity per path, in caller order,
/// then route schema inspection and Encode through the manifest-aware resident
/// loader. The normalized source also makes retry markers detect mutations.
fn promote_legacy_cloud_parquet_sources(source: &mut GraphConfig) -> Result<()> {
    for table in &mut source.tables {
        let should_promote = matches!(
            &table.source,
            graph_config::DataSource::Parquet { paths }
                if !paths.is_empty() && paths.iter().all(|path| is_cloud_path(path))
        );
        if !should_promote {
            continue;
        }
        let transfer = build_input_transfer_for_table(table)?
            .ok_or_else(|| anyhow!("cloud input transfer unavailable for table {}", table.name))?;
        promote_legacy_cloud_parquet_table(table, &transfer)?;
    }
    Ok(())
}

fn promote_legacy_cloud_parquet_table(
    table: &mut TableSpec,
    transfer: &CloudTransfer,
) -> Result<()> {
    let graph_config::DataSource::Parquet { paths } = &table.source else {
        return Ok(());
    };
    let mut objects = Vec::with_capacity(paths.len());
    let mut source_size = 0u64;
    for path in paths {
        let metadata = transfer
            .head_blocking(object_path_from_cloud_url(path)?)
            .with_context(|| format!("pin legacy cloud Parquet object {path}"))?;
        ensure!(
            metadata.size > 0,
            "cloud Parquet object {path} has zero size"
        );
        ensure!(
            metadata.e_tag.is_some() || metadata.version.is_some(),
            "cloud Parquet object {path} has no immutable identity"
        );
        source_size = source_size
            .checked_add(metadata.size)
            .ok_or_else(|| anyhow!("table {} source size overflows u64", table.name))?;
        objects.push(ObjectStoreObject {
            path: path.clone(),
            size: metadata.size,
            e_tag: metadata.e_tag,
            version: metadata.version,
        });
    }
    table.source = graph_config::DataSource::ObjectStoreParquet {
        objects,
        auth_profile: "ambient".to_string(),
        authority: None,
    };
    table.source_size = Some(source_size);
    Ok(())
}

fn source_schema_threads(file_count: usize) -> usize {
    effective_parallelism()
        .saturating_mul(2)
        .clamp(2, 32)
        .min(file_count.max(1))
}

fn load_parquet_source_schemas(
    table: &TableSpec,
    inputs: Vec<ParquetInput>,
) -> Result<Vec<Arc<Schema>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let transfer = build_input_transfer_for_table(table)?;
    let slots = Arc::new(std::sync::Mutex::new(vec![None; inputs.len()]));
    let tasks = inputs
        .into_iter()
        .enumerate()
        .map(|(index, input)| {
            let transfer = transfer.clone();
            let slots = Arc::clone(&slots);
            let table_name = table.name.clone();
            WorkItem {
                bucket: None,
                run: Box::new(move |_| {
                    let resident = ResidentFile::load(
                        &input.path,
                        TableFormat::Parquet,
                        transfer.as_ref(),
                        None,
                        input.condition,
                    )
                    .map_err(|error| {
                        if error
                            .downcast_ref::<InvalidParquetMetadataError>()
                            .is_some()
                        {
                            anyhow::Error::new(PublicIngestError::new(format!(
                                "table {} Parquet input {} has an invalid footer or metadata; regenerate the source file",
                                table_name,
                                index + 1,
                            )))
                        } else {
                            error.context(format!(
                                "prepare Parquet schema for {}",
                                input.path
                            ))
                        }
                    })?;
                    let schema = Arc::clone(
                        resident
                            .parquet_metadata
                            .as_ref()
                            .ok_or_else(|| {
                                anyhow!("missing prepared Parquet metadata for {}", input.path)
                            })?
                            .schema(),
                    );
                    slots
                        .lock()
                        .map_err(|_| anyhow!("Parquet schema result lock poisoned"))?[index] =
                        Some(schema);
                    Ok(())
                }),
            }
        })
        .collect::<Vec<_>>();
    let pool = TaskPool::<()>::new(source_schema_threads(tasks.len()));
    pool.run_tasks(tasks)?;
    drop(pool);
    Arc::try_unwrap(slots)
        .map_err(|_| anyhow!("Parquet schema results remained shared"))?
        .into_inner()
        .map_err(|_| anyhow!("Parquet schema result lock poisoned"))?
        .into_iter()
        .enumerate()
        .map(|(index, schema)| {
            schema.ok_or_else(|| anyhow!("missing prepared Parquet schema at index {index}"))
        })
        .collect()
}

fn visit_source_schemas(
    source: &GraphConfig,
    databricks_config_file: Option<&Path>,
    helper: &mut Option<crate::databricks::DatabricksHelper>,
    mut visit: impl FnMut(&TableSpec, &Schema) -> Result<()>,
) -> Result<()> {
    for table in &source.tables {
        match &table.source {
            graph_config::DataSource::Parquet { .. }
            | graph_config::DataSource::ObjectStoreParquet { .. } => {
                for schema in load_parquet_source_schemas(table, parquet_inputs(table))? {
                    visit(table, schema.as_ref())?;
                }
            }
            graph_config::DataSource::Csv { .. } => visit(table, &table_schema(table)?)?,
            graph_config::DataSource::Databricks { .. } => {
                let schema = helper
                    .as_mut()
                    .ok_or_else(|| anyhow!("Databricks helper was not initialized"))?
                    .schema(&table.source, databricks_config_file)?;
                visit(table, schema.as_ref())?;
            }
            _ => return Err(anyhow!("table {} source is not implemented", table.name)),
        }
    }
    Ok(())
}

/// Resolve the complete physical source schema without scanning source rows.
/// The hardening benchmark uses this to compare the legacy materialized path
/// with direct Databricks input for field order, type, and nullability.
pub fn inspect_source_schema(
    mut source: GraphConfig,
    base_dir: Option<&Path>,
    databricks_config_file: Option<&Path>,
) -> Result<SourceSchemaSignature> {
    if let Some(base_dir) = base_dir {
        resolve_local_paths(&mut source, base_dir);
    }
    source.normalize_multicat_pseudo_list();
    promote_legacy_cloud_parquet_sources(&mut source)?;
    let mut helper = start_databricks_helper(&source)?;
    let mut tables = BTreeMap::new();
    for table in &source.tables {
        if tables.insert(table.name.clone(), None).is_some() {
            return Err(anyhow!("duplicate table {}", table.name));
        }
    }
    visit_source_schemas(
        &source,
        databricks_config_file,
        &mut helper,
        |table, schema| {
            let current = physical_schema_signature(schema);
            let expected = tables.get_mut(&table.name).expect("validated table name");
            if expected.as_ref().is_some_and(|prior| prior != &current) {
                return Err(anyhow!(
                    "table {} has inconsistent physical schemas across source files",
                    table.name
                ));
            }
            *expected = Some(current);
            Ok(())
        },
    )?;
    let tables = tables
        .into_iter()
        .map(|(table, signature)| {
            signature
                .map(|signature| (table.clone(), signature))
                .ok_or_else(|| anyhow!("table {table} has no source files"))
        })
        .collect::<Result<_>>()?;
    Ok(SourceSchemaSignature { tables })
}

/// Target type information for FK columns during edge key extraction.
///
/// This is separate from `ingested_dtype` (used for rowpack storage).
/// Every FK column gets an FkEdgeTarget entry, even when FK and PK have the same type.
/// This enables the edge building code to extract keys correctly without modifying
/// the storage type.
#[derive(Debug, Clone, PartialEq)]
pub struct FkEdgeTarget {
    /// FK column's original type (same as ColumnSpec.ingested_dtype).
    pub ingested_type: IngestedType,
    /// PK column's type (determines Key variant for edge lookup).
    pub target_type: IngestedType,
    /// PK's decimal scale (for Decimal128/256 key extraction).
    pub scale: Option<i32>,
    /// PK's decimal precision (for Decimal128/256).
    pub precision: Option<u32>,
}

const DEFAULT_INDEX_MEM_BUDGET_FRAC: f64 = 0.7;
const EPHEMERAL_DATABRICKS_CONFIG_ENV: &str = "DISKGRAPH_EPHEMERAL_DATABRICKS_CONFIG";

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
    ingest_with_stats_from_source(source, Some(base_dir), output, dest, options, None, None)
}

fn choose_ingest_num_shards(pinned: Option<usize>, memory_floor: usize) -> usize {
    pinned.unwrap_or(0).max(memory_floor).max(1)
}

fn table_sized_bytes(declared: Option<u64>, compute_local_bytes: impl FnOnce() -> u64) -> u64 {
    match declared {
        Some(bytes) if bytes > 0 => bytes,
        _ => compute_local_bytes(),
    }
}

/// Run ingest end-to-end.
///
/// `output_dataplane_dir` (optional) is the fully-resolved dataplane
/// destination — cloud (`s3://…`) or local (e.g., `/mnt/ebs/…`). When
/// set, every output file is streamed to the dataplane during ingest
/// via [`SnapshotMirror`], `_MANIFEST.json` is written + uploaded once
/// every other file is durable, then `_SUCCESS` is written + uploaded
/// strictly last. The path layout is owned by the caller (the kumo
/// Python `DiskgraphService._dataplane_path`); Rust treats it as
/// opaque.
///
/// `success_marker_content` (optional) is the JSON payload Python's
/// `success_marker.write_diskgraph_success_marker` builds (the
/// `DiskgraphIngestResult` shape). When set, ingest writes it as the
/// final step both locally (atomic via tmp+rename) and on the
/// dataplane (if mirroring). When `None`, ingest skips the marker
/// write — preserving the legacy behavior where the caller writes the
/// marker out-of-band via `success_marker::write_success_marker` after
/// ingest returns.
pub fn ingest_with_stats_from_source(
    source: GraphConfig,
    base_dir: Option<&Path>,
    output: &str,
    dest: Dest,
    options: IngestOptions,
    output_dataplane_dir: Option<&str>,
    success_marker_content: Option<&str>,
) -> Result<(DatasetMetadata, IngestStats)> {
    ingest_with_stats_from_source_and_sidecars(
        source,
        base_dir,
        output,
        dest,
        options,
        output_dataplane_dir,
        success_marker_content,
        HashMap::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ingest_with_stats_from_source_and_sidecars(
    mut source: GraphConfig,
    base_dir: Option<&Path>,
    output: &str,
    dest: Dest,
    options: IngestOptions,
    output_dataplane_dir: Option<&str>,
    success_marker_content: Option<&str>,
    stage_sidecars: HashMap<String, StageSidecarColumns>,
) -> Result<(DatasetMetadata, IngestStats)> {
    let wall_start = std::time::Instant::now();
    // Reject the markerless-mirror config at the API boundary.
    // `output_dataplane_dir.is_some() + success_marker_content.is_none()`
    // would produce a dataplane copy with data files but no
    // `_MANIFEST.json` and no `_SUCCESS` (those are written by
    // `finalize_snapshot`, which only runs when the marker payload is
    // supplied). The result is unverifiable by `verify_snapshot_integrity`
    // and unrestorable by the kumo-side restore path that keys off
    // `_SUCCESS`. There is no production caller for this shape — both
    // pybind kwargs are independent only for legacy compatibility.
    // Failing loud at the API boundary turns "silently produces a
    // useless dataplane copy" into "ingest aborts with a clear
    // message" before any S3 bytes are written.
    if output_dataplane_dir.is_some() && success_marker_content.is_none() {
        return Err(anyhow!(
            "output_dataplane_dir set without success_marker_content: \
             this would write data files to the dataplane but skip \
             `_MANIFEST.json` and `_SUCCESS`, producing a copy that \
             cannot be verified or restored. Either supply \
             `success_marker_content` (atomic-marker mode) or drop \
             `output_dataplane_dir` (legacy local-only ingest)."
        ));
    }
    tracing::debug!(
        dataset_name = %source.dataset_name,
        tables = source.tables.len(),
        output,
        dest = ?dest,
        has_base_dir = base_dir.is_some(),
        "Ingest run started"
    );
    if let Some(base_dir) = base_dir {
        resolve_local_paths(&mut source, base_dir);
    }
    // Universal entry-point pre-pass: collapse multicategorical "pseudo list"
    // columns (delimited string + delimiter) into the native list on-disk
    // shape. Applies to every config that flows into ingest, regardless of
    // whether it came from `convert_kumo_to_diskgraph` (which calls the
    // same pass) or was loaded directly from JSON. Idempotent. See
    // kumo-ai/kumo#28478.
    source.normalize_multicat_pseudo_list();
    source.validate()?;
    promote_legacy_cloud_parquet_sources(&mut source)?;
    // Size num_shards in the process that spends the per-shard PK-map/MPHF
    // memory. An upstream conversion may have pinned a host-sized value in a
    // different cgroup, so the ingest process takes the maximum of that pin
    // and its own memory-derived floor.
    let sized_bytes: u64 = source
        .tables
        .iter()
        .map(|table| {
            table_sized_bytes(table.source_size, || {
                table
                    .source_paths()
                    .filter(|path| !path.contains("://"))
                    .map(crate::config_converter::inference::compute_local_dir_bytes)
                    .sum()
            })
        })
        .sum();
    let memory_floor = if sized_bytes > 0 {
        crate::config_converter::inference::estimate_num_shards(sized_bytes)
    } else {
        tracing::warn!(
            pinned = source.num_shards.unwrap_or(0),
            "cannot derive ingest-cgroup shard floor: no source_size or resolved local bytes"
        );
        1
    };
    let pinned = source.num_shards;
    let chosen = choose_ingest_num_shards(pinned, memory_floor);
    tracing::info!(
        pinned = pinned.unwrap_or(0),
        memory_floor,
        num_shards = chosen,
        sized_bytes,
        cgroup_mem_bytes = total_memory_bytes().unwrap_or(0),
        worker_threads = default_worker_threads(),
        "resolved num_shards to ingest cgroup budget"
    );
    if pinned.is_some() && pinned != Some(chosen) {
        tracing::warn!(
            pinned = pinned.unwrap_or(0),
            chosen,
            "overriding pinned num_shards: too few for this ingest cgroup's memory budget"
        );
    }
    source.num_shards = Some(chosen);
    let encode_marker_payload = serde_json::to_vec(&source)?;

    // NOTE: We no longer call resolve_fk_types() here.
    // FK columns keep their original ingested_dtype for rowpack storage.
    // FkEdgeTarget (built in IngestContext::new) handles edge key extraction.

    let target = std::sync::Arc::new(OutputTarget::parse_with_dataplane(
        output,
        dest,
        output_dataplane_dir,
    )?);
    if !matches!(&*target, OutputTarget::Local { .. }) {
        return Err(anyhow!("index build requires local output"));
    }
    // Invalidation belongs at the reusable ingest boundary, not only in the
    // CLI supervisor: pybind and direct Rust callers can also retry/resume.
    // Complete this synchronously before `prepare`, a fresh local wipe, phase
    // cleanup, or any upload can mutate objects below a published marker.
    target.invalidate_publication()?;
    target.prepare()?;
    let resume_from_phase = options.resume_from_phase;
    if let OutputTarget::Local { root, .. } = &*target {
        // Log disk space before ingest as a pre-flight warning
        log_disk_space(root);

        // Fresh ingest semantics: wipe everything on entry. In resume mode the
        // caller is asserting the on-disk state is intentional state from a
        // prior partial run and we keep it intact.
        if resume_from_phase.is_none() && root.exists() {
            fs::remove_dir_all(root)?;
            fs::create_dir_all(root)?;
        } else if !root.exists() {
            fs::create_dir_all(root)?;
        }

        // Validate predecessor markers before doing any work — fail fast if
        // the caller asked for a resume from phase P but earlier phases did
        // not complete in a previous attempt.
        if let Some(rfp) = resume_from_phase {
            for pred in rfp.predecessors() {
                if !phase_marker_exists(root, pred.name()) {
                    return Err(anyhow!(
                        "resume_from_phase={:?}: predecessor phase '{}' has no completion marker at {} \
                         — cannot resume from {:?} without prior phases done",
                        rfp,
                        pred.name(),
                        phase_marker_path(root, pred.name()).display(),
                        rfp,
                    ));
                }
                if pred == IngestPhase::Encode {
                    let marker = fs::read(phase_marker_path(root, pred.name()))?;
                    if marker != encode_marker_payload {
                        return Err(anyhow!(
                            "resume source identity does not match the completed Encode marker"
                        ));
                    }
                }
            }
        }
    }

    // CleanupGuard: in fresh-ingest mode, if ingest fails or panics, remove
    // partial output. In resume mode we keep partial state on failure so the
    // next attempt can resume from where this one died.
    let cleanup_guard = if resume_from_phase.is_none() {
        if let OutputTarget::Local { root, .. } = &*target {
            Some(CleanupGuard::new(root.clone()))
        } else {
            None
        }
    } else {
        None
    };

    // Keep a clone of the target Arc so the post-ingest finalize step
    // can reach the manifest/mirror after `ctx` has consumed its own
    // clone. Cloning an `Arc` is cheap (refcount bump).
    let target_for_finalize = std::sync::Arc::clone(&target);
    let mut ctx = IngestContext::new(source, target, options, output, stage_sidecars)?;
    let profile_dir = ctx.profile_dir.clone();
    if let Some(profile_dir) = &profile_dir {
        fs::create_dir_all(profile_dir)?;
    }
    let local_root = match &*ctx.target {
        OutputTarget::Local { root, .. } => root.clone(),
        _ => unreachable!("validated above: ingest requires local output"),
    };
    let should_run = |p: IngestPhase| -> bool {
        match resume_from_phase {
            None => true,
            Some(rfp) => p >= rfp,
        }
    };
    // Only wipe at phase entry for the resume target — that's the one
    // that may have partial output from a prior crash. Fresh runs already
    // wiped the entire output_dir at function entry; phases that succeed
    // and trigger downstream work don't need their successors wiped.
    let wipe_on_entry = |p: IngestPhase| -> bool { Some(p) == resume_from_phase };

    // Repopulate in-memory state that skipped predecessor phases would
    // normally have produced. Order matches phase order so each helper
    // sees its own predecessors' output.
    if let Some(rfp) = resume_from_phase {
        if rfp > IngestPhase::Encode {
            ctx.restore_encode_state()?;
            ctx.repopulate_table_contexts_from_source()?;
            ctx.repopulate_node_stats_from_meta()?;
        }
        if rfp > IngestPhase::PkMap {
            ctx.restore_resume_stats_state(PK_MAP_STATE_PATH)?;
            ctx.repopulate_pk_hash_paths_from_disk()?;
        }
        if rfp > IngestPhase::EdgesMap {
            ctx.restore_resume_stats_state(EDGES_MAP_STATE_PATH)?;
        }
    }

    // Wall-time durations for each phase. Skipped phases stay at zero —
    // they're consumed by PerfStats below regardless of which phases ran.
    let context_wall = std::time::Duration::ZERO;
    let mut encode_wall = std::time::Duration::ZERO;
    let mut pk_map_wall = std::time::Duration::ZERO;
    let mut edges_map_wall = std::time::Duration::ZERO;
    let mut index_wall = std::time::Duration::ZERO;

    ctx.set_peak_stage(Some(0));
    if should_run(IngestPhase::Encode) {
        let wipe_encode_on_entry = wipe_on_entry(IngestPhase::Encode);
        if wipe_encode_on_entry {
            // An Encode retry starts from offset zero. Discard provisional
            // output before even the schema query so a failed preflight can
            // never leave prior partial output looking resumable.
            wipe_phase_outputs(&local_root, "encode")?;
        }
        ctx.prepare_input_schemas()?;
        let encode_start = std::time::Instant::now();
        run_phase_with_marker_payload(
            &local_root,
            "encode",
            false,
            profile_dir.as_deref(),
            &encode_marker_payload,
            || {
                ctx.set_peak_stage(Some(1));
                ctx.build_temp_shards()
            },
        )?;
        encode_wall = encode_start.elapsed();
        tracing::info!(
            stage = "encode",
            elapsed_ms = encode_wall.as_millis() as u64,
            elapsed = %format_duration(encode_wall.as_millis() as u64),
            context_ms = context_wall.as_millis() as u64,
            "encode stage complete"
        );
    } else {
        tracing::info!(stage = "encode", "resume: skipped (marker present)");
    }
    ctx.set_peak_stage(Some(2));
    if should_run(IngestPhase::PkMap) {
        let pk_map_start = std::time::Instant::now();
        let rss_before_pk_map = rss_anon_diag_bytes();
        update_atomic_max(&ctx.pk_map_peak_rss, rss_before_pk_map);
        // Phase-pristine COW for fkey bitmaps so pk_map cannot leak partial
        // mutations into encode's outputs across a resume boundary.
        ctx.pk_map_pre()?;
        run_phase_with_marker(
            &local_root,
            "pk_map",
            wipe_on_entry(IngestPhase::PkMap),
            profile_dir.as_deref(),
            || {
                ctx.build_sharded_pk_maps()?;
                ctx.persist_resume_stats_state(PK_MAP_STATE_PATH)
            },
        )?;
        // Marker is now written; a crash from this line on does not invalidate
        // pk_map. Drop the pristines + free pkey.bin temp files we no longer
        // need (saves ~160 GB on DoorDash-shaped data).
        ctx.pk_map_post()?;
        let pkey_freed = ctx.cleanup_pkey_temp_files().unwrap_or(0);
        if pkey_freed > 0 {
            tracing::info!(
                pkey_bytes_freed = pkey_freed,
                pkey_gb_freed = pkey_freed / 1_073_741_824,
                "post-pk_map cleanup: freed encode pkey temp files"
            );
        }
        pk_map_wall = pk_map_start.elapsed();
        let estimate_sum = ctx
            .pk_map_estimate_sum
            .load(std::sync::atomic::Ordering::Relaxed);
        let observed_peak = ctx
            .pk_map_peak_rss
            .load(std::sync::atomic::Ordering::Relaxed);
        let rss_after_pk_map = rss_anon_diag_bytes();
        let ratio_pct = if estimate_sum > 0 {
            (observed_peak.saturating_sub(rss_before_pk_map) as f64 / estimate_sum as f64 * 100.0)
                as u64
        } else {
            0
        };
        tracing::info!(
            stage = "pk_map",
            elapsed_ms = pk_map_wall.as_millis() as u64,
            elapsed = %format_duration(pk_map_wall.as_millis() as u64),
            estimate_sum_bytes = estimate_sum,
            estimate_sum_gb = estimate_sum / 1_073_741_824,
            rss_before_bytes = rss_before_pk_map,
            rss_peak_bytes = observed_peak,
            rss_peak_gb = observed_peak / 1_073_741_824,
            rss_after_bytes = rss_after_pk_map,
            actual_over_estimate_pct = ratio_pct,
            "pk_map stage complete"
        );
    } else {
        tracing::info!(stage = "pk_map", "resume: skipped (marker present)");
    }
    // Shard offsets are needed by Index metadata even when EdgesMap is
    // skipped during resume. They are independent of reclaimable edge inputs.
    ctx.build_shard_offsets();
    ctx.set_peak_stage(Some(3));
    if should_run(IngestPhase::EdgesMap) {
        // Build the sort-key/FK lookups only when EdgesMap will actually run.
        // Index resume intentionally permits those EdgesMap inputs to have
        // been reclaimed after its durable predecessor marker was written.
        let edge_ctx = ctx.edge_build_context()?;
        let edges_map_start = std::time::Instant::now();
        run_phase_with_marker(
            &local_root,
            "edges_map",
            wipe_on_entry(IngestPhase::EdgesMap),
            profile_dir.as_deref(),
            || {
                ctx.build_edges_mapped(&edge_ctx)?;
                ctx.record_index_input_bytes();
                ctx.persist_resume_stats_state(EDGES_MAP_STATE_PATH)
            },
        )?;
        drop(edge_ctx);
        edges_map_wall = edges_map_start.elapsed();
        tracing::info!(
            stage = "edges_map",
            elapsed_ms = edges_map_wall.as_millis() as u64,
            elapsed = %format_duration(edges_map_wall.as_millis() as u64),
            "edges_map stage complete"
        );
        // Post-edges_map best-effort cleanup of inputs that no future phase
        // needs. Marker is already on disk so a crash here doesn't break
        // resume — `index` doesn't read these dirs, and `cleanup_temp` at
        // pipeline end will remove `_tmp/` wholesale on full success.
    } else {
        tracing::info!(stage = "edges_map", "resume: skipped (marker present)");
    }
    for stage in ["edges_src", "fkeys", "fkeys_dst", "edge_keys", "sort_keys"] {
        let _ = ctx.cleanup_edge_stage(stage);
    }
    ctx.set_peak_stage(Some(4));
    ctx.record_index_input_bytes_if_missing();
    let index_metadata = ctx.build_index_metadata()?;
    if should_run(IngestPhase::Index) {
        let index_start = std::time::Instant::now();
        let index_metadata_for_phase = index_metadata.clone();
        run_phase_with_marker(
            &local_root,
            "index",
            wipe_on_entry(IngestPhase::Index),
            profile_dir.as_deref(),
            || ctx.build_index(index_metadata_for_phase),
        )?;
        index_wall = index_start.elapsed();
        tracing::info!(
            stage = "index",
            elapsed_ms = index_wall.as_millis() as u64,
            elapsed = %format_duration(index_wall.as_millis() as u64),
            "index stage complete"
        );
        // Keep edges_map until publication. If finalization OOMs, the
        // supervisor reruns Index from this durable predecessor without
        // reopening Encode or rebuilding EdgesMap.
    } else {
        tracing::info!(stage = "index", "resume: skipped (marker present)");
    }
    ctx.set_peak_stage(Some(5));
    let finalize_start = std::time::Instant::now();
    let (index_metadata, source_config) = ctx.finalize_metadata(index_metadata)?;
    // TT-NG timestamp grouping: merge the per-shard spills into per-stage
    // `bucket_index.bin` (final node-ids grouped by TIMESTAMP). Runs after
    // `finalize_metadata` (shard offsets / min-ids known) and BEFORE
    // final publication (spills and resumable phase state live under `_tmp/`).
    // No-op when grouping is off.
    let ts_groupings = ctx.run_ts_grouping()?;
    let sample_weight_indexes = ctx.run_sample_weight_indexes()?;
    let binary_label_indexes = ctx.run_binary_label_indexes()?;
    let finalize_wall = finalize_start.elapsed();
    ctx.set_peak_stage(None);
    tracing::info!(
        stage = "finalize",
        elapsed_ms = finalize_wall.as_millis() as u64,
        elapsed = %format_duration(finalize_wall.as_millis() as u64),
        "finalize stage complete"
    );
    let wall_time = wall_start.elapsed();
    // Preserve the existing Kumo SOURCE_SIZE contract: logical Arrow bytes
    // observed by the one shared consumer, not compressed files, IPC framing,
    // or Cloud Fetch wire bytes.
    let source_total_bytes = Some(ctx.logical_arrow_source_bytes);
    let bytes_read = ctx.logical_arrow_source_bytes;
    let bytes_written = match &*ctx.target {
        OutputTarget::Local { root, .. } => sum_published_dir_bytes(root).unwrap_or(0),
        _ => 0,
    };
    let perf = PerfStats {
        wall_time_ms: wall_time.as_millis() as u64,
        encode_wall_nanos: encode_wall.as_nanos().min(u64::MAX as u128) as u64,
        source_input_wait_nanos: ctx.source_input_wait_nanos,
        context_wall_ms: context_wall.as_millis() as u64,
        encode_wall_ms: encode_wall.as_millis() as u64,
        pk_map_wall_ms: pk_map_wall.as_millis() as u64,
        edges_map_wall_ms: edges_map_wall.as_millis() as u64,
        index_wall_ms: index_wall.as_millis() as u64,
        finalize_wall_ms: finalize_wall.as_millis() as u64,
        bytes_read,
        bytes_written,
        source_total_bytes,
        databricks_wire_bytes: ctx.databricks_wire_bytes,
        databricks_decoded_ipc_bytes: ctx.databricks_decoded_ipc_bytes,
        databricks_query_startup_nanos: ctx.databricks_query_startup_nanos,
        encode_memory_high_water_bytes: ctx.encode_memory_high_water_bytes,
        source_rows: ctx.logical_source_rows,
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
    stats.ts_groupings = ts_groupings;
    stats.sample_weight_indexes = sample_weight_indexes;
    stats.binary_label_indexes = binary_label_indexes;
    let report = IngestReport::compact(stats.clone(), source_config);
    ctx.target
        .write_bytes("ingest_stats.json", serde_json::to_vec_pretty(&report)?)?;

    // Write kumo-format edge stats if kumo metadata is available.
    if let Some(edge_stats_bytes) = ctx.build_edge_stats_json(&index_metadata)? {
        ctx.target
            .write_bytes(col_stats_export::EDGE_STATS_FILENAME, edge_stats_bytes)?;
        tracing::info!("wrote {}", col_stats_export::EDGE_STATS_FILENAME);
    }

    let total_pk_drops: u64 = ctx.pk_duplicate_drops.values().sum();
    let total_null_ts_drops: u64 = ctx.pk_null_timestamp_drops.values().sum::<u64>()
        + ctx.edge_null_timestamp_drops.values().sum::<u64>();
    tracing::info!(
        total_pk_duplicates_dropped = total_pk_drops,
        total_null_timestamps_dropped = total_null_ts_drops,
        "Ingestion data quality summary"
    );

    tracing::debug!(
        elapsed_ms = wall_time.as_millis() as u64,
        node_types = index_metadata.node_types.len(),
        edge_relations = index_metadata.edge_relations.len(),
        bytes_read,
        bytes_written,
        num_shards = ctx.num_shards,
        "Ingest run finished"
    );

    // Ingest succeeded — drain pending dataplane uploads (if any) and
    // write the manifest + success markers. This must happen BEFORE the
    // cleanup guard is disarmed so any failure here triggers cleanup of
    // the partial output (no half-finalized snapshot lingers locally).
    //
    // The "wipe L1 on L2 upload failure" behavior is INTENTIONAL. The
    // diskgraph pool's storage model requires the invariant ``L1 ⇒ L2``:
    // an L1-only state would force every downstream consumer (eviction,
    // restore, pool-state model) to track per-snapshot "L2 status" as an
    // extra axis, fragmenting the storage layer. CloudTransfer's per-file
    // retry absorbs transient blips; sustained S3 unavailability failing
    // the whole ingest is correct — the pool cannot function without
    // dataplane backup regardless. See ``snapshot_mirror::finalize_snapshot``
    // for the full rationale.
    if let Some(content) = success_marker_content {
        // Pull in any files that were written outside `OutputTarget`
        // (the per-shard `offsets.bin` / `edges.bin` / `build_stats.json`
        // emitted by `diskgraph-index::builder`). Without this the
        // manifest is missing those entries and the dataplane mirror
        // never receives them, so a successful ingest produces a
        // snapshot that's openable on L1 but un-openable on L2 after
        // restore — the failure mode singlehost hit on AMI stop/start.
        // Mirror submissions queued here are drained by
        // `finalize_snapshot::wait_all_pending` below before the
        // manifest + `_SUCCESS` writes, preserving `_SUCCESS`-last
        // atomicity on both backends.
        target_for_finalize
            .reconcile_unrecorded_files()
            .context("reconcile post-build files into manifest + mirror")?;
        let root = require_local_output(&target_for_finalize)?.to_path_buf();
        let manifest = target_for_finalize
            .manifest()
            .ok_or_else(|| anyhow!("local output target missing manifest accumulator"))?;
        let mirror = target_for_finalize.mirror();
        crate::snapshot_mirror::finalize_snapshot(
            &root,
            mirror.map(|m| m.as_ref()),
            manifest,
            content,
        )
        .context("finalize_snapshot at end of ingest")?;
    }
    // No `else if mirror without marker` branch: the markerless-mirror
    // combo is rejected at the API boundary above, so the only way to
    // reach here without a marker is also without a mirror — nothing
    // to drain.

    // Phase markers and EncodeState must survive every operation that can
    // still fail publication. Once `_SUCCESS` is durable they are no longer
    // needed. Cleanup is best-effort after publication: a temp-file deletion
    // failure must not turn a valid published snapshot into a reported ingest
    // failure (or arm CleanupGuard to erase it).
    if success_marker_content.is_some() {
        if let Err(error) = ctx.cleanup_temp() {
            tracing::warn!("published snapshot retained temporary files: {error:#}");
        }
    } else {
        ctx.cleanup_temp()?;
    }

    if let Some(guard) = cleanup_guard {
        guard.disarm();
    }

    Ok((index_metadata, stats))
}

/// Guard that removes a directory on drop (cleanup partial output on failure).
/// Call `disarm()` on success to prevent cleanup.
struct CleanupGuard {
    path: PathBuf,
    armed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CleanupGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            armed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    fn disarm(self) {
        self.armed.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        if self.armed.load(std::sync::atomic::Ordering::SeqCst) {
            if phase_marker_exists(&self.path, IngestPhase::Encode.name()) {
                tracing::warn!(
                    path = %self.path.display(),
                    "preserving resumable ingest output after completed Encode"
                );
                return;
            }
            tracing::warn!(
                path = %self.path.display(),
                "cleaning up partial ingest output"
            );
            if let Err(e) = fs::remove_dir_all(&self.path) {
                tracing::error!(
                    path = %self.path.display(),
                    error = %e,
                    "failed to clean up partial ingest output"
                );
            }
        }
    }
}

/// Check if an IO error indicates disk space exhaustion (ENOSPC or EDQUOT).
pub fn is_disk_space_error(err: &anyhow::Error) -> bool {
    for cause in err.chain() {
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>() {
            if let Some(code) = io_err.raw_os_error() {
                // ENOSPC = 28, EDQUOT = 122
                if code == 28 || code == 122 {
                    return true;
                }
            }
        }
    }
    // Fallback: check stringified error for disk-space keywords
    // (catches cases where io::Error was stringified by intermediate error types)
    let msg = format!("{err:?}");
    msg.contains("No space left on device")
        || msg.contains("os error 28")
        || msg.contains("Disk quota exceeded")
        || msg.contains("os error 122")
}

#[cfg(test)]
mod tests_resume_phase {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ingest_phase_ordering() {
        assert!(IngestPhase::Encode < IngestPhase::PkMap);
        assert!(IngestPhase::PkMap < IngestPhase::EdgesMap);
        assert!(IngestPhase::EdgesMap < IngestPhase::Index);
    }

    #[test]
    fn ingest_phase_predecessors() {
        let preds: Vec<_> = IngestPhase::Index.predecessors().collect();
        assert_eq!(
            preds,
            vec![
                IngestPhase::Encode,
                IngestPhase::PkMap,
                IngestPhase::EdgesMap,
            ]
        );
        let preds: Vec<_> = IngestPhase::PkMap.predecessors().collect();
        assert_eq!(preds, vec![IngestPhase::Encode]);
        let preds: Vec<_> = IngestPhase::Encode.predecessors().collect();
        assert!(preds.is_empty());
    }

    #[test]
    fn ingest_phase_name_roundtrip() {
        for p in ALL_PHASES.iter().copied() {
            assert_eq!(IngestPhase::from_name(p.name()), Some(p));
        }
        assert!(IngestPhase::from_name("nope").is_none());
    }

    #[test]
    fn validated_index_progress_requires_index_phase_and_a_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        assert_eq!(validated_index_resume_progress(root).unwrap(), None);

        write_phase_marker(root, IngestPhase::Encode.name()).unwrap();
        assert_eq!(validated_index_resume_progress(root).unwrap(), None);
        write_phase_marker(root, IngestPhase::PkMap.name()).unwrap();
        write_phase_marker(root, IngestPhase::EdgesMap.name()).unwrap();
        assert_eq!(first_incomplete_phase(root).unwrap(), IngestPhase::Index);
        assert_eq!(validated_index_resume_progress(root).unwrap(), None);

        write_phase_marker(root, IngestPhase::Index.name()).unwrap();
        assert_eq!(validated_index_resume_progress(root).unwrap(), None);
    }

    #[test]
    fn validated_index_progress_returns_only_fully_validated_commits() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for phase in [
            IngestPhase::Encode,
            IngestPhase::PkMap,
            IngestPhase::EdgesMap,
        ] {
            write_phase_marker(root, phase.name()).unwrap();
        }

        let output = root.join("nodes/users/shard-00000");
        fs::create_dir_all(&output).unwrap();
        let offsets = bincode::serialize(
            &diskgraph_index::layout::OffsetFileHeader::new(0, 1),
        )
        .unwrap();
        fs::write(output.join("offsets.bin"), &offsets).unwrap();
        fs::write(output.join("edges.bin"), b"").unwrap();
        fs::write(output.join("build_stats.json"), b"{}").unwrap();
        let marker = diskgraph_index::graph::index_shard_commit_path(
            root,
            "users",
            "shard-00000",
        );
        fs::create_dir_all(marker.parent().unwrap()).unwrap();
        fs::write(
            &marker,
            serde_json::to_vec(&serde_json::json!({
                "version": 1,
                "shard_key": "Users/shard-00000",
                "expected_nodes": 0,
                "offsets_len": offsets.len(),
                "edges_len": 0,
                "build_stats_len": 2,
            }))
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            validated_index_resume_progress(root).unwrap(),
            Some(vec!["Users/shard-00000".to_string()])
        );
        fs::write(output.join("edges.bin"), b"corrupt").unwrap();
        assert!(validated_index_resume_progress(root).is_err());
    }

    #[test]
    fn write_and_check_phase_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        assert!(!phase_marker_exists(root, "encode"));
        write_phase_marker(root, "encode").unwrap();
        assert!(phase_marker_exists(root, "encode"));
        // marker lives under _tmp/ so cleanup_temp will sweep it
        let path = phase_marker_path(root, "encode");
        assert!(path.starts_with(root.join(TEMP_DIR_NAME)));
    }

    #[test]
    fn ephemeral_databricks_config_removal_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("databricks.cfg");
        fs::write(&path, b"[profile]\ntoken = secret\n").unwrap();

        remove_ephemeral_databricks_config_path(&path).unwrap();
        assert!(!path.exists());
        remove_ephemeral_databricks_config_path(&path).unwrap();
    }

    #[test]
    fn wipe_phase_outputs_removes_marker_and_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // arrange: marker present + pk_map outputs alongside context's
        // node_data.ctx and an index shard-NNNNN/ dir that must survive.
        write_phase_marker(root, "pk_map").unwrap();
        let table_dir = root.join("nodes/users");
        fs::create_dir_all(&table_dir).unwrap();
        fs::write(table_dir.join("pk_hash-00000.bin"), b"stale").unwrap();
        fs::write(table_dir.join("pk_data-00000.bin"), b"stale").unwrap();
        fs::write(table_dir.join("pk_hash-00000.meta.json"), b"{}").unwrap();
        fs::write(table_dir.join("node_data.ctx"), b"keep").unwrap();
        let shard = table_dir.join("shard-00000");
        fs::create_dir_all(&shard).unwrap();
        fs::write(shard.join("offsets.bin"), b"keep").unwrap();
        // act
        wipe_phase_outputs(root, "pk_map").unwrap();
        // assert
        assert!(!phase_marker_exists(root, "pk_map"));
        assert!(!table_dir.join("pk_hash-00000.bin").exists());
        assert!(!table_dir.join("pk_data-00000.bin").exists());
        assert!(!table_dir.join("pk_hash-00000.meta.json").exists());
        assert!(table_dir.join("node_data.ctx").exists());
        assert!(shard.join("offsets.bin").exists());
    }

    #[test]
    fn wipe_phase_outputs_index_preserves_pk_map_and_context() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_phase_marker(root, "index").unwrap();
        let table_dir = root.join("nodes/users");
        fs::create_dir_all(&table_dir).unwrap();
        fs::write(table_dir.join("node_data.ctx"), b"keep").unwrap();
        fs::write(table_dir.join("pk_hash-00000.bin"), b"keep").unwrap();
        fs::write(table_dir.join("pk_data-00000.bin"), b"keep").unwrap();
        let shard = table_dir.join("shard-00000");
        fs::create_dir_all(&shard).unwrap();
        fs::write(shard.join("offsets.bin"), b"stale").unwrap();
        fs::write(shard.join("edges.bin"), b"stale").unwrap();
        fs::write(shard.join("build_stats.json"), b"{}").unwrap();
        fs::write(shard.join("keep.bin"), b"keep").unwrap();
        fs::write(root.join("index_metadata.json"), b"{}").unwrap();
        wipe_phase_outputs(root, "index").unwrap();
        assert!(!phase_marker_exists(root, "index"));
        assert!(!root.join("index_metadata.json").exists());
        assert!(!shard.join("offsets.bin").exists());
        assert!(!shard.join("edges.bin").exists());
        assert!(!shard.join("build_stats.json").exists());
        assert!(shard.join("keep.bin").exists());
        assert!(table_dir.join("node_data.ctx").exists());
        assert!(table_dir.join("pk_hash-00000.bin").exists());
        assert!(table_dir.join("pk_data-00000.bin").exists());
    }

    #[test]
    fn wipe_index_preserves_committed_shard_and_rejects_corruption() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let shard = root.join("nodes/users/shard-00000");
        fs::create_dir_all(&shard).unwrap();
        let header = diskgraph_index::layout::OffsetFileHeader::new(0, 1);
        let offsets = bincode::serialize(&header).unwrap();
        fs::write(shard.join("offsets.bin"), &offsets).unwrap();
        fs::write(shard.join("edges.bin"), b"").unwrap();
        fs::write(shard.join("build_stats.json"), b"{}").unwrap();
        let commit = index_shard_commit_path(root, "users", "shard-00000");
        fs::create_dir_all(commit.parent().unwrap()).unwrap();
        fs::write(
            &commit,
            serde_json::to_vec(&serde_json::json!({
                "version": 1,
                "shard_key": "Users/shard-00000",
                "expected_nodes": 0,
                "offsets_len": offsets.len(),
                "edges_len": 0,
                "build_stats_len": 2,
            }))
            .unwrap(),
        )
        .unwrap();

        wipe_phase_outputs(root, "index").unwrap();
        assert!(shard.join("offsets.bin").exists());
        assert!(shard.join("edges.bin").exists());
        assert!(shard.join("build_stats.json").exists());

        fs::write(shard.join("edges.bin"), b"corrupt").unwrap();
        assert!(wipe_phase_outputs(root, "index").is_err());
        assert!(commit.exists(), "an invalid commit must not be discarded");
    }

    #[test]
    fn wipe_phase_outputs_edges_map_only_touches_tmp() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_phase_marker(root, "edges_map").unwrap();
        let edges = root.join("_tmp/edges_map/rel/shard-00000");
        fs::create_dir_all(&edges).unwrap();
        fs::write(edges.join("part-0.bin"), b"stale").unwrap();
        let table_dir = root.join("nodes/users");
        fs::create_dir_all(&table_dir).unwrap();
        fs::write(table_dir.join("node_data.ctx"), b"keep").unwrap();
        fs::write(table_dir.join("pk_hash-00000.bin"), b"keep").unwrap();
        wipe_phase_outputs(root, "edges_map").unwrap();
        assert!(!phase_marker_exists(root, "edges_map"));
        assert!(!root.join("_tmp/edges_map").exists());
        assert!(table_dir.join("node_data.ctx").exists());
        assert!(table_dir.join("pk_hash-00000.bin").exists());
    }

    #[test]
    fn run_phase_with_marker_writes_marker_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let called = std::cell::Cell::new(false);
        run_phase_with_marker(root, "encode", false, None, || {
            called.set(true);
            Ok(())
        })
        .unwrap();
        assert!(called.get());
        assert!(phase_marker_exists(root, "encode"));
    }

    #[test]
    fn run_phase_with_marker_no_marker_on_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let result = run_phase_with_marker(root, "encode", false, None, || {
            Err(anyhow!("synthetic boom"))
        });
        assert!(result.is_err());
        assert!(!phase_marker_exists(root, "encode"));
    }

    #[test]
    fn run_phase_with_marker_wipes_partial_outputs_first() {
        // Simulate a previous crashed pk_map run that left partial pk_hash
        // files behind — wipe must clear them before the phase body runs.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let stale = root.join("nodes/orders/pk_hash-00007.bin");
        fs::create_dir_all(stale.parent().unwrap()).unwrap();
        fs::write(&stale, b"partial garbage").unwrap();
        run_phase_with_marker(root, "pk_map", true, None, || {
            // body sees a clean output area
            assert!(!stale.exists());
            Ok(())
        })
        .unwrap();
        assert!(phase_marker_exists(root, "pk_map"));
    }

    #[test]
    fn predecessor_check_passes_when_all_markers_present() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_phase_marker(root, "encode").unwrap();
        // emulate the same predecessor check that runs in
        // ingest_with_stats_from_source
        for pred in IngestPhase::PkMap.predecessors() {
            assert!(phase_marker_exists(root, pred.name()), "{}", pred.name());
        }
    }

    #[test]
    fn predecessor_check_detects_missing_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Encode marker missing — pk_map should not be considered resumable.
        let missing: HashSet<_> = IngestPhase::PkMap
            .predecessors()
            .filter(|p| !phase_marker_exists(root, p.name()))
            .map(|p| p.name())
            .collect();
        assert_eq!(missing, ["encode"].iter().copied().collect());
    }
}

#[cfg(test)]
mod tests_repopulate_helpers {
    //! Tests for `IngestContext::repopulate_*` helpers and the
    //! `wipe_phase_outputs` rescoping done in commit ac25fd5. These exercise
    //! the resume-from-phase path where in-memory state must be rebuilt from
    //! whatever is already on disk.
    use super::*;
    use crate::graph_config::{
        ColumnSpec, DataSource, EdgeTableSpec, ForeignKeySpec, GraphConfig, IngestedType, TableSpec,
    };

    // -- minimal fixtures --------------------------------------------------

    pub(super) fn node_table(name: &str, primary_key: Option<&str>) -> TableSpec {
        TableSpec {
            name: name.to_string(),
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            }],
            primary_key: primary_key.map(str::to_string),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        }
    }

    fn edge_table(name: &str, src_ref: &str, dst_ref: &str) -> TableSpec {
        TableSpec {
            name: name.to_string(),
            columns: vec![
                ColumnSpec {
                    name: "src_id".to_string(),
                    ingested_dtype: IngestedType::Int64,
                    source_dtype: None,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    precision: None,
                    scale: None,
                    semantic_type: None,
                    multicategorical_delimiter: None,
                },
                ColumnSpec {
                    name: "dst_id".to_string(),
                    ingested_dtype: IngestedType::Int64,
                    source_dtype: None,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    precision: None,
                    scale: None,
                    semantic_type: None,
                    multicategorical_delimiter: None,
                },
            ],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: Some(EdgeTableSpec {
                src_fk: ForeignKeySpec {
                    edge_column: "src_id".to_string(),
                    ref_table: src_ref.to_string(),
                },
                dst_fk: ForeignKeySpec {
                    edge_column: "dst_id".to_string(),
                    ref_table: dst_ref.to_string(),
                },
                forward: None,
                reverse: None,
            }),
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        }
    }

    fn graph_config(tables: Vec<TableSpec>) -> GraphConfig {
        GraphConfig {
            dataset_name: "test_dataset".to_string(),
            tables,
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: Default::default(),
        }
    }

    /// Build a minimal `IngestContext` directly (bypassing `IngestContext::new`,
    /// which requires a fully-loaded `IngestOptions` and has unrelated
    /// validation logic). Only the fields the repopulate helpers read /
    /// write are populated meaningfully; everything else gets a default.
    fn make_ctx(
        source: GraphConfig,
        target: Arc<OutputTarget>,
        num_shards: usize,
    ) -> IngestContext {
        IngestContext {
            source,
            target,
            profile_dir: None,
            databricks_config_file: None,
            databricks_helper: None,
            parallel: false,
            source_loader_threads: None,
            source_encoder_threads: None,
            pk_map_threads: None,
            edges_map_threads: None,
            index_threads: None,
            pk_map_estimate_sum: Arc::new(AtomicU64::new(0)),
            pk_map_peak_rss: Arc::new(AtomicU64::new(0)),
            index_mem_budget: None,
            num_shards,
            table_rows: HashMap::new(),
            table_contexts: HashMap::new(),
            shard_rows: HashMap::new(),
            shard_offsets: HashMap::new(),
            stage_sidecars: HashMap::new(),
            binary_label_counts: HashMap::new(),
            sample_weight_counts: HashMap::new(),
            pk_hash_paths: HashMap::new(),
            relation_defs: Vec::new(),
            relationships: Vec::new(),
            fk_edge_targets: HashMap::new(),
            edge_counts: HashMap::new(),
            edge_dst_nonzero_counts: HashMap::new(),
            pk_duplicate_counts: HashMap::new(),
            pk_duplicate_drops: HashMap::new(),
            pk_null_timestamp_drops: HashMap::new(),
            edge_null_timestamp_drops: HashMap::new(),
            pk_duplicate_samples: HashMap::new(),
            dropped_rows_counts: HashMap::new(),
            pk_sampled_max_key_len: HashMap::new(),
            key_parse_failures: 0,
            key_lookup_failures: 0,
            key_membership_failures: 0,
            key_failure_samples: Vec::new(),
            key_failures_by_context: HashMap::new(),
            output: String::new(),
            trimmed_strings: Arc::new(AtomicU64::new(0)),
            progress: None,
            temp_edges_src_bytes: 0,
            temp_fkeys_bytes: 0,
            temp_edges_map_bytes: 0,
            temp_node_data_bytes: 0,
            databricks_wire_bytes: 0,
            databricks_decoded_ipc_bytes: 0,
            databricks_query_startup_nanos: 0,
            logical_arrow_source_bytes: 0,
            logical_arrow_source_bytes_by_table: HashMap::new(),
            source_input_wait_nanos: 0,
            encode_memory_high_water_bytes: 0,
            logical_source_rows: 0,
            logical_source_rows_by_table: HashMap::new(),
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: crate::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
            text_sample_stats: HashMap::new(),
            parent_index_path: None,
            resolve_columns: Vec::new(),
        }
    }

    fn local_target(root: &Path) -> Arc<OutputTarget> {
        Arc::new(OutputTarget::Local {
            root: root.to_path_buf(),
            mirror: None,
            manifest: Arc::new(crate::snapshot_mirror::ManifestAccumulator::new()),
        })
    }

    // -- A. repopulate_node_stats_from_meta --------------------------------

    #[test]
    fn repopulate_node_stats_happy_path() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let tables = vec![
            node_table("Users", Some("id")),
            node_table("Items", Some("id")),
            node_table("Reviews", Some("id")),
        ];
        let num_shards = 4;
        // Per-table per-shard row counts, deterministic and distinct.
        let counts: HashMap<&str, [u64; 4]> = [
            ("Users", [10u64, 20, 30, 40]),
            ("Items", [1, 2, 3, 4]),
            ("Reviews", [100, 0, 50, 7]),
        ]
        .into_iter()
        .collect();
        for table in &tables {
            for (shard, &n) in counts[table.name.as_str()]
                .iter()
                .enumerate()
                .take(num_shards)
            {
                write_temp_meta(&target, &temp_node_data_meta_path(&table.name, shard), n, 0)
                    .unwrap();
            }
        }
        let mut ctx = make_ctx(graph_config(tables.clone()), target, num_shards);
        ctx.repopulate_node_stats_from_meta().unwrap();
        for table in &tables {
            let expected: Vec<u64> = counts[table.name.as_str()].to_vec();
            assert_eq!(
                ctx.shard_rows.get(&table.name),
                Some(&expected),
                "shard_rows mismatch for {}",
                table.name
            );
            let expected_total: u64 = expected.iter().sum();
            assert_eq!(
                ctx.table_rows.get(&table.name),
                Some(&expected_total),
                "table_rows mismatch for {}",
                table.name
            );
        }
    }

    #[test]
    fn repopulate_node_stats_overwrites_garbage() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let table = node_table("Users", Some("id"));
        let num_shards = 2;
        for shard in 0..num_shards {
            write_temp_meta(&target, &temp_node_data_meta_path(&table.name, shard), 7, 0).unwrap();
        }
        let mut ctx = make_ctx(graph_config(vec![table.clone()]), target, num_shards);
        // Pre-populate with garbage that doesn't match what's on disk.
        ctx.shard_rows
            .insert("Users".to_string(), vec![999, 999, 999, 999]);
        ctx.table_rows.insert("Users".to_string(), 3996);
        ctx.repopulate_node_stats_from_meta().unwrap();
        assert_eq!(
            ctx.shard_rows.get("Users"),
            Some(&vec![7u64, 7]),
            "garbage shard_rows not overwritten"
        );
        assert_eq!(
            ctx.table_rows.get("Users"),
            Some(&14u64),
            "garbage table_rows not overwritten"
        );
    }

    #[test]
    fn repopulate_node_stats_missing_file_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let table = node_table("Users", Some("id"));
        let num_shards = 3;
        // Write meta for shards 0 and 2; leave shard 1 absent.
        write_temp_meta(&target, &temp_node_data_meta_path(&table.name, 0), 5, 0).unwrap();
        write_temp_meta(&target, &temp_node_data_meta_path(&table.name, 2), 11, 0).unwrap();
        let mut ctx = make_ctx(graph_config(vec![table]), target, num_shards);
        ctx.repopulate_node_stats_from_meta()
            .expect("missing meta should be skipped, not fail");
        assert_eq!(
            ctx.shard_rows.get("Users"),
            Some(&vec![5u64, 0, 11]),
            "missing-shard slot should be 0"
        );
        assert_eq!(ctx.table_rows.get("Users"), Some(&16u64));
    }

    #[test]
    fn repopulate_node_stats_uses_validated_commit_after_meta_reclaim() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let target = local_target(root);
        let table = node_table("Users", Some("id"));
        let num_shards = 3;
        write_temp_meta(&target, &temp_node_data_meta_path(&table.name, 0), 5, 0).unwrap();
        write_temp_meta(&target, &temp_node_data_meta_path(&table.name, 2), 11, 0).unwrap();

        let shard_name = "shard-00001";
        let output_dir = root.join("nodes/users").join(shard_name);
        fs::create_dir_all(&output_dir).unwrap();
        let offsets = bincode::serialize(&diskgraph_index::layout::OffsetFileHeader::new(7, 8))
            .unwrap();
        fs::write(output_dir.join("offsets.bin"), &offsets).unwrap();
        fs::write(output_dir.join("edges.bin"), b"").unwrap();
        fs::write(output_dir.join("build_stats.json"), b"{}").unwrap();
        let commit = index_shard_commit_path(root, "users", shard_name);
        fs::create_dir_all(commit.parent().unwrap()).unwrap();
        fs::write(
            &commit,
            serde_json::to_vec(&serde_json::json!({
                "version": 1,
                "shard_key": "Users/shard-00001",
                "expected_nodes": 7,
                "offsets_len": offsets.len(),
                "edges_len": 0,
                "build_stats_len": 2,
            }))
            .unwrap(),
        )
        .unwrap();

        let mut ctx = make_ctx(graph_config(vec![table]), target, num_shards);
        ctx.repopulate_node_stats_from_meta().unwrap();
        assert_eq!(ctx.shard_rows.get("Users"), Some(&vec![5u64, 7, 11]));
        assert_eq!(ctx.table_rows.get("Users"), Some(&23u64));
    }

    #[test]
    fn binary_label_index_fails_when_counted_spill_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let table = node_table("train_seeds", None);
        let mut ctx = make_ctx(graph_config(vec![table]), target, 1);
        ctx.stage_sidecars.insert(
            "train_seeds".to_string(),
            StageSidecarColumns {
                binary_label_column: Some("TARGET1".to_string()),
                sample_weight_column: None,
            },
        );
        ctx.shard_offsets
            .insert("train_seeds".to_string(), vec![100]);
        ctx.binary_label_counts.insert(
            ("train_seeds".to_string(), 0),
            crate::binary_label::LabelCounts {
                false_count: 1,
                true_count: 1,
                invalid_count: 0,
            },
        );

        let err = ctx.run_binary_label_indexes().unwrap_err();
        assert!(
            err.to_string().contains("missing binary-label spill"),
            "{err:#}"
        );
    }

    #[test]
    fn sample_weight_index_fails_when_counted_spill_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let table = node_table("train_seeds", None);
        let mut ctx = make_ctx(graph_config(vec![table]), target, 1);
        ctx.stage_sidecars.insert(
            "train_seeds".to_string(),
            StageSidecarColumns {
                binary_label_column: None,
                sample_weight_column: Some("WEIGHT".to_string()),
            },
        );
        ctx.shard_offsets
            .insert("train_seeds".to_string(), vec![100]);
        ctx.sample_weight_counts.insert(
            ("train_seeds".to_string(), 0),
            crate::sample_weight::WeightCounts {
                row_count: 2,
                valid_count: 2,
                invalid_count: 0,
                total_weight: 3.0,
            },
        );

        let err = ctx.run_sample_weight_indexes().unwrap_err();
        assert!(
            err.to_string().contains("missing sample-weight spill"),
            "{err:#}"
        );
    }

    #[test]
    fn repopulate_node_stats_skips_edge_tables() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let users = node_table("Users", Some("id"));
        let edges = edge_table("Friendship", "Users", "Users");
        let num_shards = 1;
        write_temp_meta(&target, &temp_node_data_meta_path(&users.name, 0), 42, 0).unwrap();
        // Even if an edge table somehow had a meta on disk, the helper
        // must not surface it under shard_rows.
        write_temp_meta(&target, &temp_node_data_meta_path(&edges.name, 0), 99, 0).unwrap();
        let mut ctx = make_ctx(graph_config(vec![users, edges]), target, num_shards);
        ctx.repopulate_node_stats_from_meta().unwrap();
        assert!(ctx.shard_rows.contains_key("Users"));
        assert!(
            !ctx.shard_rows.contains_key("Friendship"),
            "edge table should not appear in shard_rows"
        );
        assert!(
            !ctx.table_rows.contains_key("Friendship"),
            "edge table should not appear in table_rows"
        );
    }

    // -- B. repopulate_pk_hash_paths_from_disk -----------------------------

    fn touch_pk_files(root: &Path, table: &str, num_shards: usize) {
        for shard in 0..num_shards {
            let dir = root.join(format!("nodes/{}", table.to_lowercase()));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(format!("pk_hash-{:05}.bin", shard)), b"h").unwrap();
            fs::write(dir.join(format!("pk_data-{:05}.bin", shard)), b"d").unwrap();
        }
    }

    #[test]
    fn repopulate_pk_hash_paths_happy_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let target = local_target(root);
        let tables = vec![
            node_table("Users", Some("id")),
            node_table("Items", Some("id")),
        ];
        let num_shards = 3;
        for t in &tables {
            touch_pk_files(root, &t.name, num_shards);
        }
        let mut ctx = make_ctx(graph_config(tables.clone()), target, num_shards);
        ctx.repopulate_pk_hash_paths_from_disk().unwrap();
        for t in &tables {
            let paths = ctx
                .pk_hash_paths
                .get(&t.name)
                .unwrap_or_else(|| panic!("missing pk_hash_paths for {}", t.name));
            assert_eq!(paths.len(), num_shards, "wrong shard count for {}", t.name);
            for (shard, p) in paths.iter().enumerate() {
                let expected_hash =
                    format!("nodes/{}/pk_hash-{:05}.bin", t.name.to_lowercase(), shard);
                let expected_data =
                    format!("nodes/{}/pk_data-{:05}.bin", t.name.to_lowercase(), shard);
                assert_eq!(p.hash_path, expected_hash, "hash_path mismatch");
                assert_eq!(p.data_path, expected_data, "data_path mismatch");
            }
        }
    }

    #[test]
    fn repopulate_pk_hash_paths_missing_file_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let target = local_target(root);
        let table = node_table("Users", Some("id"));
        let num_shards = 2;
        touch_pk_files(root, &table.name, num_shards);
        // Remove one pk_hash file to simulate a broken / partially-cleaned
        // pk_map output dir.
        let missing = root.join("nodes/users/pk_hash-00001.bin");
        fs::remove_file(&missing).unwrap();
        let mut ctx = make_ctx(graph_config(vec![table]), target, num_shards);
        let err = ctx
            .repopulate_pk_hash_paths_from_disk()
            .expect_err("expected error for missing pk_hash file");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("pk_hash-00001.bin"),
            "error message should name the missing file, got: {msg}"
        );
    }

    #[test]
    fn repopulate_pk_hash_paths_skips_edge_and_pkless_tables() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let target = local_target(root);
        let users = node_table("Users", Some("id"));
        let pkless = node_table("Logs", None);
        let edges = edge_table("Friendship", "Users", "Users");
        let num_shards = 1;
        touch_pk_files(root, &users.name, num_shards);
        // Lay down stray files for ineligible tables — the helper must
        // not try to read them and must not surface them in the map.
        touch_pk_files(root, &pkless.name, num_shards);
        touch_pk_files(root, &edges.name, num_shards);
        let mut ctx = make_ctx(graph_config(vec![users, pkless, edges]), target, num_shards);
        ctx.repopulate_pk_hash_paths_from_disk().unwrap();
        assert!(ctx.pk_hash_paths.contains_key("Users"));
        assert!(
            !ctx.pk_hash_paths.contains_key("Logs"),
            "PK-less table should be excluded"
        );
        assert!(
            !ctx.pk_hash_paths.contains_key("Friendship"),
            "edge table should be excluded"
        );
    }

    #[test]
    fn repopulate_pk_hash_paths_overwrites_garbage() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let target = local_target(root);
        let table = node_table("Users", Some("id"));
        let num_shards = 2;
        touch_pk_files(root, &table.name, num_shards);
        let mut ctx = make_ctx(graph_config(vec![table]), target, num_shards);
        ctx.pk_hash_paths.insert(
            "Users".to_string(),
            vec![PkHashPaths {
                hash_path: "GARBAGE".to_string(),
                data_path: "GARBAGE".to_string(),
            }],
        );
        ctx.repopulate_pk_hash_paths_from_disk().unwrap();
        let paths = ctx.pk_hash_paths.get("Users").unwrap();
        assert_eq!(paths.len(), num_shards, "garbage entry not replaced");
        assert_eq!(paths[0].hash_path, "nodes/users/pk_hash-00000.bin");
    }

    // -- C. repopulate_table_contexts_from_source --------------------------

    fn write_ctx_files(root: &Path, tables: &[TableSpec]) {
        for t in tables {
            if t.edge_table.is_some() {
                continue;
            }
            let dir = root.join(format!("nodes/{}", t.name.to_lowercase()));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("node_data.ctx"), b"ctx").unwrap();
        }
    }

    #[test]
    fn repopulate_table_contexts_happy_path() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let tables = vec![
            node_table("Users", Some("id")),
            node_table("Items", None),
            node_table("Reviews", Some("id")),
        ];
        write_ctx_files(tmp.path(), &tables);
        let mut ctx = make_ctx(graph_config(tables.clone()), target, 1);
        ctx.repopulate_table_contexts_from_source().unwrap();
        assert_eq!(ctx.table_contexts.len(), 3);
        for t in &tables {
            let expected = format!("nodes/{}/node_data.ctx", t.name.to_lowercase());
            assert_eq!(
                ctx.table_contexts.get(&t.name),
                Some(&expected),
                "ctx path mismatch for {}",
                t.name
            );
        }
    }

    #[test]
    fn repopulate_table_contexts_overwrites_garbage() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let table = node_table("Users", Some("id"));
        let tables = vec![table];
        write_ctx_files(tmp.path(), &tables);
        let mut ctx = make_ctx(graph_config(tables), target, 1);
        ctx.table_contexts
            .insert("Users".to_string(), "GARBAGE/path.ctx".to_string());
        ctx.repopulate_table_contexts_from_source().unwrap();
        assert_eq!(
            ctx.table_contexts.get("Users"),
            Some(&"nodes/users/node_data.ctx".to_string())
        );
    }

    #[test]
    fn repopulate_table_contexts_skips_edge_tables() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let users = node_table("Users", Some("id"));
        let edges = edge_table("Friendship", "Users", "Users");
        let tables = vec![users, edges];
        write_ctx_files(tmp.path(), &tables);
        let mut ctx = make_ctx(graph_config(tables), target, 1);
        ctx.repopulate_table_contexts_from_source().unwrap();
        assert!(ctx.table_contexts.contains_key("Users"));
        assert!(
            !ctx.table_contexts.contains_key("Friendship"),
            "edge table should not get a context entry"
        );
    }

    #[test]
    fn repopulate_table_contexts_errors_on_missing_ctx_file() {
        let tmp = tempfile::tempdir().unwrap();
        let target = local_target(tmp.path());
        let tables = vec![node_table("Users", Some("id"))];
        let mut ctx = make_ctx(graph_config(tables), target, 1);
        let err = ctx
            .repopulate_table_contexts_from_source()
            .expect_err("should error when ctx file missing");
        let msg = format!("{err}");
        assert!(msg.contains("Users"), "error must name the table: {msg}");
        assert!(
            msg.contains("node_data.ctx"),
            "error must name the path: {msg}"
        );
    }

    // -- D. wipe_phase_outputs rescoping coverage --------------------------
    //
    // The existing `wipe_phase_outputs_*` tests in `tests_resume_phase` cover
    // the broad happy path. These tests pin down a few invariants that the
    // rescoping in ac25fd5 was specifically meant to enforce.

    #[test]
    fn wipe_pk_map_preserves_ctx_and_shard_dirs() {
        // Defensive check: even if a node_data.bin somehow ended up under a
        // shard-NNNNN/ subdir of nodes/<t>/ (it normally lives in _tmp), the
        // pk_map wipe must not touch it because it doesn't have one of the
        // pk_hash- / pk_data- prefixes.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let table_dir = root.join("nodes/users");
        fs::create_dir_all(&table_dir).unwrap();
        fs::write(table_dir.join("node_data.ctx"), b"keep").unwrap();
        fs::write(table_dir.join("pk_hash-00000.bin"), b"stale").unwrap();
        let shard = table_dir.join("shard-00000");
        fs::create_dir_all(&shard).unwrap();
        fs::write(shard.join("node_data.bin"), b"keep").unwrap();
        wipe_phase_outputs(root, "pk_map").unwrap();
        assert!(
            table_dir.join("node_data.ctx").exists(),
            "node_data.ctx must survive pk_map wipe"
        );
        assert!(
            shard.join("node_data.bin").exists(),
            "shard-NNNNN/* must survive pk_map wipe"
        );
        assert!(
            !table_dir.join("pk_hash-00000.bin").exists(),
            "pk_hash file should be removed"
        );
    }

    #[test]
    fn wipe_index_preserves_pk_map_and_ctx_outputs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let table_dir = root.join("nodes/users");
        fs::create_dir_all(&table_dir).unwrap();
        fs::write(table_dir.join("node_data.ctx"), b"keep").unwrap();
        fs::write(table_dir.join("pk_hash-00000.bin"), b"keep").unwrap();
        fs::write(table_dir.join("pk_data-00000.bin"), b"keep").unwrap();
        let shard0 = table_dir.join("shard-00000");
        let shard1 = table_dir.join("shard-00001");
        fs::create_dir_all(&shard0).unwrap();
        fs::create_dir_all(&shard1).unwrap();
        fs::write(shard0.join("offsets.bin"), b"stale").unwrap();
        fs::write(shard0.join("edges.bin"), b"stale").unwrap();
        fs::write(shard0.join("build_stats.json"), b"{}").unwrap();
        fs::write(shard1.join("offsets.bin"), b"stale").unwrap();
        fs::write(root.join("index_metadata.json"), b"{}").unwrap();
        wipe_phase_outputs(root, "index").unwrap();
        // index outputs must be gone
        assert!(!root.join("index_metadata.json").exists());
        assert!(!shard0.join("offsets.bin").exists());
        assert!(!shard0.join("edges.bin").exists());
        assert!(!shard0.join("build_stats.json").exists());
        assert!(!shard1.join("offsets.bin").exists());
        // pk_map + context outputs must survive
        assert!(table_dir.join("node_data.ctx").exists());
        assert!(table_dir.join("pk_hash-00000.bin").exists());
        assert!(table_dir.join("pk_data-00000.bin").exists());
    }

    #[test]
    fn wipe_edges_map_only_touches_tmp_edges_map() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let edges_part = root.join("_tmp/edges_map/r/shard-00000/part-00000.bin");
        fs::create_dir_all(edges_part.parent().unwrap()).unwrap();
        fs::write(&edges_part, b"stale").unwrap();
        let table_dir = root.join("nodes/users");
        fs::create_dir_all(&table_dir).unwrap();
        fs::write(table_dir.join("pk_hash-00000.bin"), b"keep").unwrap();
        // Sibling _tmp dirs that belong to other phases must survive — only
        // _tmp/edges_map should be touched.
        let sibling = root.join("_tmp/fkeys/u/shard-00000");
        fs::create_dir_all(&sibling).unwrap();
        fs::write(sibling.join("rows.bin"), b"keep").unwrap();
        wipe_phase_outputs(root, "edges_map").unwrap();
        assert!(
            !root.join("_tmp/edges_map").exists(),
            "_tmp/edges_map should be removed"
        );
        assert!(
            table_dir.join("pk_hash-00000.bin").exists(),
            "pk_hash-* must survive edges_map wipe"
        );
        assert!(
            sibling.join("rows.bin").exists(),
            "other _tmp/* dirs must survive edges_map wipe"
        );
    }
}

#[cfg(test)]
mod tests_disk_space {
    use super::*;

    #[test]
    fn test_is_disk_space_error_enospc() {
        let io_err = std::io::Error::from_raw_os_error(28); // ENOSPC
        let err = anyhow::Error::new(io_err).context("write failed");
        assert!(is_disk_space_error(&err));
    }

    #[test]
    fn test_is_disk_space_error_edquot() {
        let io_err = std::io::Error::from_raw_os_error(122); // EDQUOT
        let err = anyhow::Error::new(io_err);
        assert!(is_disk_space_error(&err));
    }

    #[test]
    fn test_is_disk_space_error_other() {
        let io_err = std::io::Error::from_raw_os_error(13); // EACCES
        let err = anyhow::Error::new(io_err);
        assert!(!is_disk_space_error(&err));
    }

    #[test]
    fn test_is_disk_space_error_non_io() {
        let err = anyhow::anyhow!("some other error");
        assert!(!is_disk_space_error(&err));
    }

    #[test]
    fn test_is_disk_space_error_stringified_enospc() {
        // Simulates ParquetError::General(String) that stringifies the inner io::Error
        let err = anyhow::anyhow!("General error: No space left on device (os error 28)");
        assert!(is_disk_space_error(&err));
    }

    #[test]
    fn test_is_disk_space_error_stringified_edquot() {
        let err = anyhow::anyhow!("General error: Disk quota exceeded (os error 122)");
        assert!(is_disk_space_error(&err));
    }

    #[test]
    fn test_cleanup_guard_armed_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("partial_output");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("data.bin"), b"test data").unwrap();
        assert!(target.exists());

        {
            let _guard = CleanupGuard::new(target.clone());
            // guard drops here, should remove target
        }
        assert!(!target.exists());
    }

    #[test]
    fn test_cleanup_guard_disarmed_keeps() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("good_output");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("data.bin"), b"test data").unwrap();

        {
            let guard = CleanupGuard::new(target.clone());
            guard.disarm();
            // guard drops here, should NOT remove target
        }
        assert!(target.exists());
    }

    #[test]
    fn test_cleanup_guard_preserves_completed_encode_for_resume() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("resumable_output");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("source-data.bin"), b"durable source data").unwrap();
        write_phase_marker(&target, IngestPhase::Encode.name()).unwrap();

        {
            let _guard = CleanupGuard::new(target.clone());
        }

        assert!(target.join("source-data.bin").exists());
        assert!(phase_marker_exists(&target, IngestPhase::Encode.name()));
    }
}

/// Log available disk space as a pre-flight check. Non-blocking — just a warning.
fn log_disk_space(output_path: &Path) {
    let parent = output_path.parent().unwrap_or(output_path);
    match disk_space_bytes(parent) {
        Some((total, available)) => {
            let pct_used = if total > 0 {
                ((total - available) as f64 / total as f64 * 100.0) as u64
            } else {
                0
            };
            if pct_used > 90 {
                tracing::warn!(
                    available_gb = available / 1_073_741_824,
                    total_gb = total / 1_073_741_824,
                    pct_used,
                    "disk space is low before ingest"
                );
            } else {
                tracing::info!(
                    available_gb = available / 1_073_741_824,
                    total_gb = total / 1_073_741_824,
                    pct_used,
                    "pre-ingest disk space check"
                );
            }
        }
        None => {
            tracing::debug!("could not determine disk space for {:?}", parent);
        }
    }
}

/// Get disk space via disk_manager. Returns (total, available) in bytes.
fn disk_space_bytes(path: &Path) -> Option<(u64, u64)> {
    let available = disk_manager::available_space(path).ok()?;
    let total = disk_manager::total_space(path).ok()?;
    Some((total, available))
}

struct BudgetedDstDegreeBitmap {
    bits: Vec<u8>,
    _budget_guard: Option<MemoryBudgetGuard>,
}

fn ensure_dst_degree_bitmap_budget(bitmap_bytes: u64, budget_bytes: u64) -> Result<()> {
    ensure!(
        bitmap_bytes.saturating_mul(2) <= budget_bytes,
        "destination-degree bitmap requires {bitmap_bytes} B, but edges_map needs room for both \
         an accumulator and a worker bitmap within its {budget_bytes} B budget"
    );
    Ok(())
}

fn merge_dst_degree_bitmap(
    merged: &mut HashMap<String, BudgetedDstDegreeBitmap>,
    relation: String,
    bitmap: BudgetedDstDegreeBitmap,
) {
    match merged.entry(relation) {
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(bitmap);
        }
        std::collections::hash_map::Entry::Occupied(mut entry) => {
            let target = &mut entry.get_mut().bits;
            for (index, byte) in bitmap.bits.iter().enumerate() {
                if index < target.len() {
                    target[index] |= byte;
                }
            }
        }
    }
}

#[cfg(test)]
mod dst_degree_bitmap_budget_tests {
    use super::*;

    #[test]
    fn budget_must_hold_an_accumulator_and_a_worker_bitmap() {
        assert!(ensure_dst_degree_bitmap_budget(50, 100).is_ok());
        assert!(ensure_dst_degree_bitmap_budget(51, 100).is_err());
    }

    #[test]
    fn merged_bitmap_stays_charged_until_relation_finalization() {
        let budget = Arc::new(MemoryBudget::new(100));
        let mut merged = HashMap::new();
        merge_dst_degree_bitmap(
            &mut merged,
            "rel".to_string(),
            BudgetedDstDegreeBitmap {
                bits: vec![0b0001],
                _budget_guard: Some(budget.acquire(10).unwrap()),
            },
        );
        merge_dst_degree_bitmap(
            &mut merged,
            "rel".to_string(),
            BudgetedDstDegreeBitmap {
                bits: vec![0b0100],
                _budget_guard: Some(budget.acquire(10).unwrap()),
            },
        );

        assert_eq!(merged["rel"].bits, vec![0b0101]);
        assert_eq!(budget.snapshot().in_use, 10);
        merged.remove("rel");
        assert_eq!(budget.snapshot().in_use, 0);
    }
}

struct IngestContext {
    source: GraphConfig,
    target: std::sync::Arc<OutputTarget>,
    profile_dir: Option<PathBuf>,
    databricks_config_file: Option<PathBuf>,
    databricks_helper: Option<crate::databricks::DatabricksHelper>,
    parallel: bool,
    source_loader_threads: Option<usize>,
    source_encoder_threads: Option<usize>,
    pk_map_threads: Option<usize>,
    edges_map_threads: Option<usize>,
    index_threads: Option<usize>,
    /// Sum of per-shard `estimate_pk_map_bytes` requests across the pk_map
    /// phase. Compared against observed peak RSS at end of the phase to
    /// validate that the budget estimate matches reality.
    pk_map_estimate_sum: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// High-water mark of RssAnon observed at task milestones during pk_map.
    pk_map_peak_rss: std::sync::Arc<std::sync::atomic::AtomicU64>,
    index_mem_budget: Option<std::sync::Arc<MemoryBudget>>,
    num_shards: usize,
    table_rows: HashMap<String, u64>,
    table_contexts: HashMap<String, String>,
    shard_rows: HashMap<String, Vec<u64>>,
    shard_offsets: HashMap<String, Vec<u64>>,
    /// TT-NG seed-sampling sidecar requests keyed by stage table name.
    /// Empty for ordinary graph ingest.
    stage_sidecars: HashMap<String, StageSidecarColumns>,
    /// TT-NG binary-label sidecars: per-(table,shard) label counts captured by
    /// the node writer. Empty unless requested in `stage_sidecars`.
    binary_label_counts: HashMap<(String, usize), crate::binary_label::LabelCounts>,
    /// TT-NG row-weight sidecars: per-(table,shard) weight counts captured by
    /// the node writer. Empty unless requested in `stage_sidecars`.
    sample_weight_counts: HashMap<(String, usize), crate::sample_weight::WeightCounts>,
    pk_hash_paths: HashMap<String, Vec<PkHashPaths>>,
    relation_defs: Vec<RelationDef>,
    relationships: Vec<RelationshipRuntime>,
    /// FK column target types for edge key extraction.
    /// Key format: "table_name.column_name" -> FkEdgeTarget
    fk_edge_targets: HashMap<String, FkEdgeTarget>,
    edge_counts: HashMap<String, u64>,
    /// Scalar result retained after EdgesMap so resume state never serializes
    /// destination-sized bitmaps.
    edge_dst_nonzero_counts: HashMap<String, u64>,
    pk_duplicate_counts: HashMap<String, u64>,
    pk_duplicate_drops: HashMap<String, u64>,
    pk_null_timestamp_drops: HashMap<String, u64>,
    edge_null_timestamp_drops: HashMap<String, u64>,
    pk_duplicate_samples: HashMap<String, Vec<String>>,
    dropped_rows_counts: HashMap<String, u64>,
    pk_sampled_max_key_len: HashMap<String, usize>,
    key_parse_failures: u64,
    key_lookup_failures: u64,
    key_membership_failures: u64,
    key_failure_samples: Vec<String>,
    key_failures_by_context: HashMap<String, KeyFailureCounts>,
    output: String,
    trimmed_strings: std::sync::Arc<AtomicU64>,
    progress: Option<ProgressCallback>,
    temp_edges_src_bytes: u64,
    temp_fkeys_bytes: u64,
    temp_edges_map_bytes: u64,
    temp_node_data_bytes: u64,
    databricks_wire_bytes: u64,
    databricks_decoded_ipc_bytes: u64,
    databricks_query_startup_nanos: u64,
    logical_arrow_source_bytes: u64,
    logical_arrow_source_bytes_by_table: HashMap<String, u64>,
    source_input_wait_nanos: u64,
    encode_memory_high_water_bytes: u64,
    logical_source_rows: u64,
    logical_source_rows_by_table: HashMap<String, u64>,
    peak_mem_stats: Option<std::sync::Arc<PeakMemStats>>,
    col_stats: bool,
    cat_count_strategy: crate::col_stats::CategoryCountStrategy,
    row_subsample: bool,
    /// Text sample stats computed during the context phase (table -> col -> stats).
    text_sample_stats: HashMap<String, HashMap<String, crate::col_stats::TextSampleStats>>,
    /// Parent DG index whose pkmap shards back the resolution of stage-table
    /// reference columns to node-ids (see `resolve_columns`). `None` for an
    /// ordinary base-graph ingest.
    parent_index_path: Option<PathBuf>,
    /// FK-free LP resolution columns (see [`crate::config::ResolveColumnSpec`]).
    /// When non-empty the encode phase resolves the listed columns' PK
    /// reference value(s) to graph node-ids and writes them as rowpack columns
    /// (no FK edge). Empty (default) preserves the edge-materializing path.
    resolve_columns: Vec<crate::config::ResolveColumnSpec>,
}

const ENCODE_STATE_VERSION: u32 = 2;
const ENCODE_STATE_PATH: &str = "_tmp/encode_state.json";
const RESUME_STATS_STATE_VERSION: u32 = 1;
const PK_MAP_STATE_PATH: &str = "_tmp/pk_map_state.json";
const EDGES_MAP_STATE_PATH: &str = "_tmp/edges_map_state.json";

#[derive(serde::Serialize, serde::Deserialize)]
struct EncodeLabelCounts {
    table: String,
    shard: usize,
    counts: crate::binary_label::LabelCounts,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct EncodeWeightCounts {
    table: String,
    shard: usize,
    counts: crate::sample_weight::WeightCounts,
}

/// Everything produced only in memory by Encode and needed by a later-phase
/// resume. Encoded rows and contexts remain in their native files; this small
/// manifest prevents inferred schema and counters from silently reverting.
#[derive(serde::Serialize, serde::Deserialize)]
struct EncodeState {
    version: u32,
    source: GraphConfig,
    binary_label_counts: Vec<EncodeLabelCounts>,
    sample_weight_counts: Vec<EncodeWeightCounts>,
    dropped_rows_counts: HashMap<String, u64>,
    pk_sampled_max_key_len: HashMap<String, usize>,
    key_parse_failures: u64,
    key_lookup_failures: u64,
    key_membership_failures: u64,
    key_failure_samples: Vec<String>,
    key_failures_by_context: HashMap<String, KeyFailureCounts>,
    trimmed_strings: u64,
    temp_edges_src_bytes: u64,
    temp_fkeys_bytes: u64,
    temp_node_data_bytes: u64,
    databricks_wire_bytes: u64,
    databricks_decoded_ipc_bytes: u64,
    databricks_query_startup_nanos: u64,
    logical_arrow_source_bytes: u64,
    logical_arrow_source_bytes_by_table: HashMap<String, u64>,
    #[serde(default)]
    source_input_wait_nanos: u64,
    encode_memory_high_water_bytes: u64,
    logical_source_rows: u64,
    logical_source_rows_by_table: HashMap<String, u64>,
}

/// In-memory counters produced after Encode and required to make a later
/// phase resume observationally equivalent to an uninterrupted ingest.
/// Each completed phase writes a full snapshot so restore never has to
/// combine deltas from multiple files.
#[derive(serde::Serialize, serde::Deserialize)]
struct ResumeStatsState {
    version: u32,
    edge_counts: HashMap<String, u64>,
    edge_dst_nonzero_counts: HashMap<String, u64>,
    pk_duplicate_counts: HashMap<String, u64>,
    pk_duplicate_drops: HashMap<String, u64>,
    pk_null_timestamp_drops: HashMap<String, u64>,
    edge_null_timestamp_drops: HashMap<String, u64>,
    pk_duplicate_samples: HashMap<String, Vec<String>>,
    key_parse_failures: u64,
    key_lookup_failures: u64,
    key_membership_failures: u64,
    key_failure_samples: Vec<String>,
    key_failures_by_context: HashMap<String, KeyFailureCounts>,
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

/// Safe validation error whose outcome cannot change across worker retries.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct PublicIngestError {
    message: String,
}

const PUBLIC_INGEST_ERROR_CODE: &str = "diskgraph_ingest_validation";
const PUBLIC_INGEST_ERROR_MESSAGE_MAX_BYTES: usize = 2 * 1024;

#[derive(serde::Serialize)]
struct PublicIngestErrorEnvelope<'a> {
    version: u8,
    code: &'static str,
    public_message: &'a str,
    native_type: &'static str,
    non_retryable: bool,
}

impl PublicIngestError {
    fn new(message: String) -> Self {
        Self { message }
    }

    pub fn int64_decimal_out_of_range(
        table_name: Option<&str>,
        column_name: &str,
    ) -> Self {
        let location = table_name.map_or_else(
            || format!("column '{column_name}'"),
            |table_name| format!("table {table_name} column '{column_name}'"),
        );
        Self::new(format!(
            "{location} contains a scale-zero decimal value outside Kumo's signed 64-bit integer \
             range; cast the column to DOUBLE or STRING in Custom SQL, or keep its values within \
             the signed 64-bit range"
        ))
    }
}

pub fn public_ingest_error_message(error: &anyhow::Error) -> Option<String> {
    error
        .downcast_ref::<PublicIngestError>()
        .map(ToString::to_string)
}

/// Serialize the allowlisted worker error contract consumed by supervisors.
pub fn public_ingest_error_envelope(error: &anyhow::Error) -> Option<String> {
    let error = error.downcast_ref::<PublicIngestError>()?;
    let mut message_end = error
        .message
        .len()
        .min(PUBLIC_INGEST_ERROR_MESSAGE_MAX_BYTES);
    while !error.message.is_char_boundary(message_end) {
        message_end -= 1;
    }
    serde_json::to_string(&PublicIngestErrorEnvelope {
        version: 1,
        code: PUBLIC_INGEST_ERROR_CODE,
        public_message: &error.message[..message_end],
        native_type: "PublicIngestError",
        non_retryable: true,
    })
    .ok()
}

pub fn is_non_retryable_ingest_error(error: &anyhow::Error) -> bool {
    error.downcast_ref::<PublicIngestError>().is_some()
}

enum SourceBatchRetention {
    Budget {
        _permit: crate::source_memory::BytePermit,
    },
    Databricks {
        _lease: crate::databricks::DatabricksBatchLease,
    },
    ResidentParquet {
        _permit: Arc<crate::source_memory::BytePermit>,
    },
}

struct SourceBatch {
    batch: RecordBatch,
    _retention: SourceBatchRetention,
    observed: bool,
}

impl SourceBatch {
    fn budgeted(batch: RecordBatch, permit: crate::source_memory::BytePermit) -> Self {
        Self {
            batch,
            _retention: SourceBatchRetention::Budget { _permit: permit },
            observed: false,
        }
    }

    fn databricks(batch: crate::databricks::DatabricksBatch) -> Self {
        let (batch, lease) = batch.into_parts();
        Self {
            batch,
            _retention: SourceBatchRetention::Databricks { _lease: lease },
            observed: false,
        }
    }

    fn resident_parquet(batch: RecordBatch, permit: Arc<crate::source_memory::BytePermit>) -> Self {
        Self {
            batch,
            _retention: SourceBatchRetention::ResidentParquet { _permit: permit },
            observed: false,
        }
    }

    fn observed(mut self) -> Self {
        self.observed = true;
        self
    }

    fn slice(self, offset: usize, length: usize) -> Self {
        Self {
            batch: self.batch.slice(offset, length),
            _retention: self._retention,
            observed: self.observed,
        }
    }
}

struct SourceValidationState {
    table: TableSpec,
    synthetic_columns: HashSet<String>,
    schema_validated: std::sync::Mutex<bool>,
    multicat_validator: std::sync::Mutex<
        Option<crate::config_converter::inference::ExplicitMulticatDelimiterValidator>,
    >,
    rows_seen: AtomicU64,
    logical_arrow_bytes: AtomicU64,
}

impl SourceValidationState {
    fn new(table: &TableSpec, synthetic_columns: HashSet<String>) -> Arc<Self> {
        Arc::new(Self {
            table: table.clone(),
            synthetic_columns,
            schema_validated: std::sync::Mutex::new(false),
            multicat_validator: std::sync::Mutex::new(Some(
                crate::config_converter::inference::ExplicitMulticatDelimiterValidator::new(table),
            )),
            rows_seen: AtomicU64::new(0),
            logical_arrow_bytes: AtomicU64::new(0),
        })
    }

    fn observe(&self, batch: &RecordBatch) -> Result<()> {
        {
            let mut validated = self
                .schema_validated
                .lock()
                .map_err(|_| anyhow!("source schema validation lock was poisoned"))?;
            if !*validated {
                validate_source_schema(
                    &self.table,
                    batch.schema().as_ref(),
                    &self.synthetic_columns,
                )?;
                *validated = true;
            }
        }
        validate_source_batch_values(&self.table, batch, &self.synthetic_columns)?;
        self.rows_seen
            .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
        let batch_bytes = batch
            .columns()
            .iter()
            .try_fold(0u64, |total, array| {
                array
                    .to_data()
                    .get_slice_memory_size()
                    .ok()
                    .and_then(|bytes| total.checked_add(bytes as u64))
            })
            .unwrap_or(u64::MAX);
        self.logical_arrow_bytes
            .fetch_add(batch_bytes, Ordering::Relaxed);
        let mut validator = self
            .multicat_validator
            .lock()
            .map_err(|_| anyhow!("multicategorical validation lock was poisoned"))?;
        if let Some(validator) = validator.as_mut() {
            validator.observe(batch);
        }
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        let validator = self
            .multicat_validator
            .lock()
            .map_err(|_| anyhow!("multicategorical validation lock was poisoned"))?
            .take();
        if let Some(validator) = validator {
            validator.finish().map_err(|error| anyhow!("{error}"))?;
        }
        Ok(())
    }

    fn rows_seen(&self) -> u64 {
        self.rows_seen.load(Ordering::Relaxed)
    }

    fn logical_arrow_bytes(&self) -> u64 {
        self.logical_arrow_bytes.load(Ordering::Relaxed)
    }
}

enum SourceWork {
    Batch(SourceBatch),
    Parquet(ParquetPartition),
    ParquetReader {
        reader: BoundedParquetBatchReader,
        permit: Arc<crate::source_memory::BytePermit>,
        first_batch: Option<SourceBatch>,
    },
}

struct EncodeBatchIter {
    receiver: Arc<std::sync::Mutex<std::sync::mpsc::Receiver<SourceWork>>>,
    source_budget: Arc<crate::source_memory::ByteBudget>,
    validation: Arc<SourceValidationState>,
    active_retention: Option<SourceBatchRetention>,
    active_parquet: Option<(
        BoundedParquetBatchReader,
        Arc<crate::source_memory::BytePermit>,
        Option<SourceBatch>,
    )>,
}

impl Iterator for EncodeBatchIter {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.active_retention = None;
            if let Some((reader, permit, first_batch)) = &mut self.active_parquet {
                if let Some(SourceBatch {
                    batch,
                    _retention,
                    observed,
                }) = first_batch.take()
                {
                    if !observed {
                        if let Err(error) = self.validation.observe(&batch) {
                            return Some(Err(error));
                        }
                    }
                    self.active_retention = Some(_retention);
                    return Some(Ok(batch));
                }
                match reader.next() {
                    Some(Ok(batch)) => {
                        if let Err(error) = self.validation.observe(&batch) {
                            return Some(Err(error));
                        }
                        self.active_retention = Some(SourceBatchRetention::ResidentParquet {
                            _permit: Arc::clone(permit),
                        });
                        return Some(Ok(batch));
                    }
                    Some(Err(error)) => return Some(Err(error)),
                    None => self.active_parquet = None,
                }
            }
            let work = match self.receiver.lock() {
                Ok(receiver) => receiver.recv().ok(),
                Err(_) => return Some(Err(anyhow!("Encode work queue lock was poisoned"))),
            }?;
            match work {
                SourceWork::Batch(SourceBatch {
                    batch,
                    _retention,
                    observed,
                }) => {
                    if !observed {
                        if let Err(error) = self.validation.observe(&batch) {
                            return Some(Err(error));
                        }
                    }
                    self.active_retention = Some(_retention);
                    return Some(Ok(batch));
                }
                SourceWork::Parquet(partition) => {
                    let permit = match self
                        .source_budget
                        .acquire_input_chunk(partition.max_decode_bytes)
                    {
                        Ok(permit) => permit,
                        Err(error) => return Some(Err(error)),
                    };
                    let permit = Arc::new(permit);
                    self.active_parquet = Some((
                        BoundedParquetBatchReader::partition(
                            partition.resident,
                            partition.row_groups,
                        ),
                        permit,
                        None,
                    ));
                }
                SourceWork::ParquetReader {
                    reader,
                    permit,
                    first_batch,
                } => {
                    self.active_parquet = Some((reader, permit, first_batch));
                }
            }
        }
    }
}

fn run_encode_batch_workers<I, R, F>(
    works: I,
    workers: usize,
    source_budget: Arc<crate::source_memory::ByteBudget>,
    validation: Arc<SourceValidationState>,
    encode: F,
) -> Result<Vec<R>>
where
    I: IntoIterator<Item = Result<SourceWork>>,
    R: Send,
    F: Fn(EncodeBatchIter) -> Result<R> + Sync,
{
    let mut works = works.into_iter();
    // Reserve the first source work before workers allocate retained scratch.
    // This guarantees progress when the shared working-memory budget is tight.
    let first_work = works.next().transpose()?;
    let workers = if first_work.is_some() {
        workers.max(1)
    } else {
        1
    };
    std::thread::scope(|scope| {
        let (sender, receiver) =
            std::sync::mpsc::sync_channel::<SourceWork>(workers.saturating_mul(2).max(1));
        let receiver = Arc::new(std::sync::Mutex::new(receiver));
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let receiver = Arc::clone(&receiver);
            let source_budget = Arc::clone(&source_budget);
            let validation = Arc::clone(&validation);
            let encode = &encode;
            handles.push(scope.spawn(move || {
                encode(EncodeBatchIter {
                    receiver,
                    source_budget,
                    validation,
                    active_retention: None,
                    active_parquet: None,
                })
            }));
        }
        drop(receiver);

        let mut source_error = None;
        if let Some(work) = first_work {
            if sender.send(work).is_err() {
                source_error = Some(anyhow!("all Encode workers stopped"));
            }
        }
        for work in works {
            if source_error.is_some() {
                break;
            }
            match work {
                Ok(work) => {
                    if sender.send(work).is_err() {
                        source_error = Some(anyhow!("all Encode workers stopped"));
                        break;
                    }
                }
                Err(error) => {
                    source_error = Some(error);
                    break;
                }
            }
        }
        drop(sender);

        let mut results = Vec::with_capacity(workers);
        let mut worker_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(error)) => {
                    if worker_error.is_none() {
                        worker_error = Some(error);
                    }
                }
                Err(_) => {
                    if worker_error.is_none() {
                        worker_error = Some(anyhow!("Encode worker panicked"));
                    }
                }
            }
        }
        if let Some(error) = worker_error {
            return Err(error);
        }
        if let Some(error) = source_error {
            return Err(error);
        }
        Ok(results)
    })
}

#[cfg(test)]
#[test]
fn encode_batch_dispatcher_runs_workers_in_parallel() {
    let workers = 4;
    let budget = crate::source_memory::ByteBudget::new(1024).unwrap();
    let works = (0..workers)
        .map(|_| {
            let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
            Ok(SourceWork::Batch(SourceBatch::budgeted(
                batch,
                budget.acquire(1).unwrap(),
            )))
        })
        .collect::<Vec<_>>();
    let table: TableSpec = serde_json::from_value(serde_json::json!({
        "name": "dispatcher", "columns": [], "source": {"type": "parquet", "paths": []}
    }))
    .unwrap();
    let validation = SourceValidationState::new(&table, HashSet::new());
    let rendezvous = Arc::new(std::sync::Barrier::new(workers));
    let results = run_encode_batch_workers(
        works,
        workers,
        Arc::clone(&budget),
        validation,
        |mut batches| {
            let batch = batches.next().expect("worker did not receive a batch")?;
            rendezvous.wait();
            Ok(batch.num_rows())
        },
    )
    .unwrap();
    assert_eq!(results, vec![0; workers]);
}

#[cfg(test)]
mod resident_parquet_tests {
    use super::*;
    use arrow_array::Int64Array;
    use diskgraph_fs::CloudTransferConfig;
    use object_store::{memory::InMemory, ObjectStoreExt, PutPayload};
    use parquet::arrow::arrow_writer::ArrowWriter;

    pub(super) fn int_batch(name: &str, values: impl IntoIterator<Item = i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(values))]).unwrap()
    }

    pub(super) fn write_parquet(
        path: &Path,
        batch: &RecordBatch,
        properties: Option<parquet::file::properties::WriterProperties>,
    ) {
        let mut writer =
            ArrowWriter::try_new(fs::File::create(path).unwrap(), batch.schema(), properties)
                .unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
    }

    pub(super) fn resident(path: &Path) -> Arc<ResidentFile> {
        Arc::new(
            ResidentFile::load(
                path.to_str().unwrap(),
                TableFormat::Parquet,
                None,
                None,
                None,
            )
            .unwrap(),
        )
    }

    fn cloud_parquet_fixture(
        object_path: &object_store::path::Path,
        values: [i64; 3],
    ) -> (Arc<InMemory>, object_store::ObjectMeta) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fixture.parquet");
        write_parquet(&path, &int_batch("id", values), None);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let store = Arc::new(InMemory::new());
        let payload = PutPayload::from(Bytes::from(fs::read(path).unwrap()));
        runtime.block_on(store.put(object_path, payload)).unwrap();
        let metadata = runtime.block_on(store.head(object_path)).unwrap();
        (store, metadata)
    }

    fn test_cloud_transfer(store: Arc<InMemory>) -> CloudTransfer {
        let max_bytes = 64 * 1024 * 1024;
        let mut config = CloudTransferConfig::from_env();
        config.max_bytes = max_bytes;
        config.buffer_count = 2;
        CloudTransfer::new(store, config)
    }

    fn read_resident_cloud_values(table: &TableSpec, transfer: &CloudTransfer) -> Result<Vec<i64>> {
        let max_bytes = 64 * 1024 * 1024;
        let budget = crate::source_memory::ByteBudget::new(max_bytes)?;
        let mut stream =
            ResidentParquetWorkStream::open(table, Some(transfer), Some(Arc::clone(&budget)), 2)?;
        let mut values = Vec::new();
        while let Some(partition) = stream.next_partition(&budget, true) {
            let partition = partition?;
            for batch in
                BoundedParquetBatchReader::partition(partition.resident, partition.row_groups)
            {
                let batch = batch?;
                values.extend(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values(),
                );
            }
        }
        Ok(values)
    }

    #[test]
    fn parallel_parquet_schema_preparation_preserves_source_order() {
        let directory = tempfile::tempdir().unwrap();
        let names = ["first", "second", "third"];
        let paths = names.map(|name| {
            let path = directory.path().join(format!("{name}.parquet"));
            write_parquet(&path, &int_batch(name, []), None);
            path.to_string_lossy().into_owned()
        });
        let mut table = super::tests_repopulate_helpers::node_table("ordered-schemas", None);
        table.source = graph_config::DataSource::Parquet {
            paths: paths.into(),
        };
        let schemas = load_parquet_source_schemas(&table, parquet_inputs(&table)).unwrap();
        assert!(schemas
            .iter()
            .zip(names)
            .all(|(schema, name)| schema.field(0).name() == name));
    }

    #[test]
    fn corrupt_local_footer_is_public_without_exposing_the_path() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("customer-secret.parquet");
        fs::write(&path, b"not a parquet file").unwrap();
        let mut table = super::tests_repopulate_helpers::node_table("Nodes", None);
        table.source = graph_config::DataSource::Parquet {
            paths: vec![path.to_string_lossy().into_owned()],
        };

        let error = load_parquet_source_schemas(&table, parquet_inputs(&table)).unwrap_err();

        assert_eq!(
            public_ingest_error_message(&error).as_deref(),
            Some(
                "table Nodes Parquet input 1 has an invalid footer or metadata; regenerate the source file"
            )
        );
        assert!(is_non_retryable_ingest_error(&error));
        assert!(!error.to_string().contains("customer-secret"));
    }

    #[test]
    fn corrupt_local_thrift_metadata_is_public() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metadata.parquet");
        write_parquet(&path, &int_batch("id", [1, 2, 3]), None);
        let mut bytes = fs::read(&path).unwrap();
        let metadata_len = u32::from_le_bytes(
            bytes[bytes.len() - 8..bytes.len() - 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let metadata_start = bytes.len() - 8 - metadata_len;
        bytes[metadata_start..metadata_start + metadata_len.min(8)].fill(0xff);
        fs::write(&path, bytes).unwrap();
        let mut table = super::tests_repopulate_helpers::node_table("Nodes", None);
        table.source = graph_config::DataSource::Parquet {
            paths: vec![path.to_string_lossy().into_owned()],
        };

        let error = load_parquet_source_schemas(&table, parquet_inputs(&table)).unwrap_err();

        assert!(is_non_retryable_ingest_error(&error));
        assert_eq!(
            public_ingest_error_message(&error).as_deref(),
            Some(
                "table Nodes Parquet input 1 has an invalid footer or metadata; regenerate the source file"
            )
        );
    }

    #[test]
    fn corrupt_pinned_remote_metadata_is_structural() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let object_path = object_store::path::Path::from("data/corrupt.parquet");
        let store = Arc::new(InMemory::new());
        runtime
            .block_on(store.put(
                &object_path,
                PutPayload::from(Bytes::from_static(b"not a parquet file")),
            ))
            .unwrap();
        let metadata = runtime.block_on(store.head(&object_path)).unwrap();
        let transfer = test_cloud_transfer(store);
        let error = ResidentFile::load(
            "s3://bucket/data/corrupt.parquet",
            TableFormat::Parquet,
            Some(&transfer),
            None,
            Some(CloudObjectCondition {
                size: metadata.size,
                e_tag: metadata.e_tag,
                version: metadata.version,
            }),
        )
        .err()
        .expect("corrupt remote metadata should fail");

        assert!(error
            .downcast_ref::<InvalidParquetMetadataError>()
            .is_some());
        assert!(!is_non_retryable_ingest_error(&error));
    }

    #[test]
    fn changed_pinned_remote_object_preserves_retryable_read_source() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let object_path = object_store::path::Path::from("data/changing.parquet");
        let (store, metadata) = cloud_parquet_fixture(&object_path, [1, 2, 3]);
        let condition = CloudObjectCondition {
            size: metadata.size,
            e_tag: metadata.e_tag,
            version: metadata.version,
        };
        assert!(condition.e_tag.is_some() || condition.version.is_some());
        runtime
            .block_on(store.put(
                &object_path,
                PutPayload::from(Bytes::from(vec![0u8; condition.size as usize])),
            ))
            .unwrap();
        let transfer = test_cloud_transfer(store);

        let error = ResidentFile::load(
            "s3://bucket/data/changing.parquet",
            TableFormat::Parquet,
            Some(&transfer),
            None,
            Some(condition),
        )
        .err()
        .expect("changed remote object should fail");

        assert!(error
            .chain()
            .any(|source| source.to_string().contains("precondition failure")),
            "unexpected error chain: {error:#}"
        );
        assert!(error
            .downcast_ref::<InvalidParquetMetadataError>()
            .is_none());
        assert!(public_ingest_error_message(&error).is_none());
    }

    #[test]
    fn local_open_failure_remains_retryable() {
        let error = ResidentFile::load(
            "/definitely/missing/source.parquet",
            TableFormat::Parquet,
            None,
            None,
            None,
        )
        .err()
        .expect("missing local input should fail");

        assert!(error
            .downcast_ref::<InvalidParquetMetadataError>()
            .is_none());
        assert!(public_ingest_error_message(&error).is_none());
    }

    struct ShortReadFixture;

    impl Length for ShortReadFixture {
        fn len(&self) -> u64 {
            128
        }
    }

    impl ChunkReader for ShortReadFixture {
        type T = std_io::Cursor<Vec<u8>>;

        fn get_read(&self, _start: u64) -> parquet::errors::Result<Self::T> {
            Err(parquet::errors::ParquetError::External(Box::new(
                std_io::Error::new(std_io::ErrorKind::TimedOut, "injected read timeout"),
            )))
        }

        fn get_bytes(&self, _start: u64, _length: usize) -> parquet::errors::Result<Bytes> {
            Err(parquet::errors::ParquetError::EOF(
                "injected short read".to_string(),
            ))
        }
    }

    #[test]
    fn adapter_short_read_is_not_structural() {
        let error = load_parquet_metadata(ShortReadFixture).unwrap_err();

        assert!(error
            .downcast_ref::<InvalidParquetMetadataError>()
            .is_none());
        assert!(
            error
                .chain()
                .any(|source| source.to_string().contains("injected")),
            "unexpected error chain: {error:#}"
        );
        assert!(public_ingest_error_message(&error).is_none());
    }

    struct SuccessfulShortReadFixture;

    impl Length for SuccessfulShortReadFixture {
        fn len(&self) -> u64 {
            128
        }
    }

    impl ChunkReader for SuccessfulShortReadFixture {
        type T = std_io::Cursor<Vec<u8>>;

        fn get_read(&self, _start: u64) -> parquet::errors::Result<Self::T> {
            Ok(std_io::Cursor::new(vec![0; 4]))
        }

        fn get_bytes(&self, _start: u64, length: usize) -> parquet::errors::Result<Bytes> {
            Ok(Bytes::from(vec![0; length.min(4)]))
        }
    }

    #[test]
    fn successful_get_bytes_short_read_is_not_structural() {
        let error = load_parquet_metadata(SuccessfulShortReadFixture).unwrap_err();

        assert!(error
            .downcast_ref::<InvalidParquetMetadataError>()
            .is_none());
        assert!(public_ingest_error_message(&error).is_none());
    }

    #[test]
    fn successful_reader_eof_before_captured_length_is_tracked() {
        let reader = TrackedChunkReader::new(SuccessfulShortReadFixture);
        let mut returned_reader = reader.get_read(0).unwrap();
        let mut bytes = Vec::new();

        returned_reader.read_to_end(&mut bytes).unwrap();

        assert_eq!(bytes.len(), 4);
        assert!(reader.read_failed());
    }

    #[test]
    fn local_truncate_after_captured_length_is_not_structural() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("truncated-after-stat.parquet");
        write_parquet(&path, &int_batch("id", [1, 2, 3]), None);
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let captured_size = file.metadata().unwrap().len();
        assert!(captured_size > 8);
        file.set_len(captured_size - 4).unwrap();

        let error = load_parquet_metadata(ResidentReader::Local {
            file,
            size: captured_size,
        })
        .unwrap_err();

        assert!(error
            .downcast_ref::<InvalidParquetMetadataError>()
            .is_none());
        assert!(public_ingest_error_message(&error).is_none());
    }

    #[test]
    fn inspect_source_schema_accepts_a_partial_config() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("partial.parquet");
        write_parquet(&path, &int_batch("physical_id", [1, 2, 3]), None);
        let source = serde_json::from_value(serde_json::json!({
            "dataset_name": "partial",
            "tables": [{
                "name": "nodes",
                "columns": [],
                "source": {
                    "type": "parquet",
                    "paths": [path.to_string_lossy()]
                }
            }]
        }))
        .unwrap();
        let signature = inspect_source_schema(source, None, None).unwrap();
        assert_eq!(signature.tables["nodes"][0].name, "physical_id");
    }

    #[test]
    #[ignore = "requires AWS credentials for kumo-public-datasets"]
    fn legacy_cloud_schema_inspection_uses_the_public_small_dataset() {
        let source = serde_json::from_value(serde_json::json!({
            "dataset_name": "hnm_tiny_legacy",
            "tables": [{
                "name": "articles",
                "columns": [],
                "source": {
                    "type": "parquet",
                    "paths": [
                        "s3://kumo-public-datasets/hnm_tiny/articles/part-00000-0005aa7e-18d0-4272-b52b-76a64f3e5a57-c000.snappy.parquet"
                    ]
                }
            }]
        }))
        .unwrap();
        let signature = inspect_source_schema(source, None, None).unwrap();
        assert!(!signature.tables["articles"].is_empty());
    }

    #[test]
    #[ignore = "requires AWS credentials for kumo-public-datasets"]
    fn source_authority_reads_the_public_small_dataset() {
        let source = serde_json::from_value(serde_json::json!({
            "dataset_name": "hnm_tiny_authority",
            "tables": [{
                "name": "articles",
                "columns": [],
                "source": {
                    "type": "object_store_parquet",
                    "objects": [{
                        "path": "s3://kumo-public-datasets/hnm_tiny/articles/part-00000-0005aa7e-18d0-4272-b52b-76a64f3e5a57-c000.snappy.parquet",
                        "size": 65701,
                        "e_tag": "\"557711dc7d7a4500a8c2ec030570dd6e-1\"",
                        "version": null
                    }],
                    "auth_profile": "ambient",
                    "authority": {
                        "type": "s3",
                        "region": "us-west-2"
                    }
                },
                "source_size": 65701
            }]
        }))
        .unwrap();
        let signature = inspect_source_schema(source, None, None).unwrap();
        assert!(!signature.tables["articles"].is_empty());
    }

    #[test]
    fn resident_parquet_partitions_decode_inside_encode_worker() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("parallel.parquet");
        write_parquet(&path, &int_batch("id", 0..4), None);
        let resident = resident(&path);
        let plan = parquet_row_group_memory_plan(
            resident.parquet_metadata.as_ref().unwrap(),
            0,
            4 * 1024 * 1024,
            None,
        )
        .unwrap();
        let works = [Ok(SourceWork::Parquet(ParquetPartition {
            resident,
            max_decode_bytes: plan.total_bytes,
            row_groups: vec![plan],
        }))];
        let table = super::tests_repopulate_helpers::node_table("parallel", Some("id"));
        let budget = crate::source_memory::ByteBudget::new(4 * 1024 * 1024).unwrap();
        let validation = SourceValidationState::new(&table, HashSet::new());
        let rows =
            run_encode_batch_workers(works, 1, budget, Arc::clone(&validation), |mut batches| {
                batches.try_fold(0, |rows, batch| Ok(rows + batch?.num_rows()))
            })
            .unwrap();
        assert_eq!(rows, [4]);
        assert_eq!(validation.rows_seen(), 4);
    }

    #[test]
    fn object_store_manifest_drives_the_resident_parquet_loader() {
        let object_path = object_store::path::Path::from("data/manifest.parquet");
        let (store, metadata) = cloud_parquet_fixture(&object_path, [1, 2, 3]);
        let mut table = super::tests_repopulate_helpers::node_table("manifest", Some("id"));
        table.source = graph_config::DataSource::ObjectStoreParquet {
            objects: vec![graph_config::ObjectStoreObject {
                path: "s3://bucket/data/manifest.parquet".to_string(),
                size: metadata.size,
                e_tag: metadata.e_tag,
                version: metadata.version,
            }],
            auth_profile: "ambient".to_string(),
            authority: None,
        };
        let transfer = test_cloud_transfer(store);
        assert_eq!(
            read_resident_cloud_values(&table, &transfer).unwrap(),
            [1, 2, 3]
        );
    }

    #[test]
    fn legacy_cloud_parquet_paths_are_promoted_into_the_resident_loader() {
        let object_path = object_store::path::Path::from("data/legacy.parquet");
        let (store, metadata) = cloud_parquet_fixture(&object_path, [4, 5, 6]);
        let transfer = test_cloud_transfer(Arc::clone(&store));
        let mut table = super::tests_repopulate_helpers::node_table("legacy", Some("id"));
        table.source = graph_config::DataSource::Parquet {
            paths: vec!["s3://bucket/data/legacy.parquet".to_string()],
        };
        promote_legacy_cloud_parquet_table(&mut table, &transfer).unwrap();
        let graph_config::DataSource::ObjectStoreParquet { objects, .. } = &table.source else {
            panic!("legacy source was not promoted");
        };
        assert_eq!(objects[0].size, metadata.size);
        assert_eq!(objects[0].e_tag, metadata.e_tag);
        assert_eq!(objects[0].version, metadata.version);
        assert_eq!(
            read_resident_cloud_values(&table, &transfer).unwrap(),
            [4, 5, 6]
        );

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(store.put(
                &object_path,
                PutPayload::from(Bytes::from_static(b"mutated")),
            ))
            .unwrap();
        assert!(read_resident_cloud_values(&table, &transfer).is_err());
    }
}

#[cfg(test)]
#[test]
fn only_explicit_public_ingest_errors_are_publishable() {
    let public_error =
        anyhow::Error::new(PublicIngestError::new("safe validation detail".to_string()))
            .context("internal worker context");
    assert_eq!(
        public_ingest_error_message(&public_error).as_deref(),
        Some("safe validation detail")
    );
    assert!(is_non_retryable_ingest_error(&public_error));
    let envelope: JsonValue =
        serde_json::from_str(&public_ingest_error_envelope(&public_error).unwrap()).unwrap();
    assert_eq!(
        envelope,
        serde_json::json!({
            "version": 1,
            "code": "diskgraph_ingest_validation",
            "public_message": "safe validation detail",
            "native_type": "PublicIngestError",
            "non_retryable": true,
        })
    );
    assert!(public_ingest_error_message(&anyhow!("provider secret detail")).is_none());
    assert!(public_ingest_error_envelope(&anyhow!("provider secret detail")).is_none());
    assert!(!is_non_retryable_ingest_error(&anyhow!(
        "provider secret detail"
    )));
}

#[cfg(test)]
#[test]
fn missing_source_columns_are_public_and_non_retryable() {
    let table: TableSpec = serde_json::from_value(serde_json::json!({
        "name": "TRANS",
        "columns": [
            {"name": "id", "ingested_dtype": "int64"},
            {"name": "K_SYMBOL", "ingested_dtype": "utf8"}
        ],
        "primary_key": "id",
        "source": {"type": "parquet", "paths": ["unused.parquet"]}
    }))
    .unwrap();
    let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);

    let error = validate_source_schema(&table, &schema, &HashSet::new()).unwrap_err();

    assert_eq!(
        public_ingest_error_message(&error).as_deref(),
        Some("table TRANS source schema is missing column 'K_SYMBOL'")
    );
    assert!(is_non_retryable_ingest_error(&error));
}

const FILE_DECODE_READ_BUFFER_BYTES: usize = 64 * 1024;

/// Drives Arrow's push CSV decoder with a hard byte boundary. Reaching the
/// boundary at a record boundary yields a smaller batch; reaching it inside a
/// record rejects before the decoder can grow again.
struct BoundedCsvBatchReader {
    reader: Box<dyn Read + Send>,
    decoder: arrow_csv::reader::Decoder,
    buffer: Vec<u8>,
    buffer_start: usize,
    buffer_end: usize,
    decoded_since_flush: u64,
    max_raw_batch_bytes: u64,
    eof: bool,
    finished: bool,
}

impl BoundedCsvBatchReader {
    fn new(reader: Box<dyn Read + Send>, schema: Schema, max_decode_bytes: u64) -> Result<Self> {
        // The decoder retains raw fields while Arrow arrays are built. Limit
        // raw CSV to one quarter of the permit, leaving space for the copied
        // Arrow values, offsets/validity, and parser/read buffers.
        let max_raw_batch_bytes = max_decode_bytes / 4;
        if max_raw_batch_bytes < 1024 {
            return Err(anyhow!("Encode CSV decode budget is too small"));
        }
        let columns = schema.fields().len().max(1) as u64;
        let estimated_per_row = columns
            .checked_mul(32)
            .ok_or_else(|| anyhow!("CSV decode row estimate overflowed"))?;
        let batch_rows =
            usize::try_from((max_decode_bytes / 16 / estimated_per_row).clamp(1, 1024))
                .unwrap_or(1);
        let buffer_bytes = FILE_DECODE_READ_BUFFER_BYTES
            .min(usize::try_from((max_decode_bytes / 16).max(1024)).unwrap_or(usize::MAX));
        Ok(Self {
            reader,
            decoder: arrow_csv::ReaderBuilder::new(Arc::new(schema))
                .with_header(true)
                .with_batch_size(batch_rows)
                .build_decoder(),
            buffer: vec![0; buffer_bytes],
            buffer_start: 0,
            buffer_end: 0,
            decoded_since_flush: 0,
            max_raw_batch_bytes,
            eof: false,
            finished: false,
        })
    }

    fn flush(&mut self) -> Option<Result<RecordBatch>> {
        match self.decoder.flush() {
            Ok(Some(batch)) => {
                self.decoded_since_flush = 0;
                Some(Ok(batch))
            }
            Ok(None) => {
                self.decoded_since_flush = 0;
                None
            }
            Err(error) => {
                self.finished = true;
                Some(Err(anyhow::Error::from(error)))
            }
        }
    }
}

impl Iterator for BoundedCsvBatchReader {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            if self.decoder.capacity() == 0 {
                if let Some(batch) = self.flush() {
                    return Some(batch);
                }
            }
            if self.decoded_since_flush >= self.max_raw_batch_bytes {
                self.finished = true;
                return Some(Err(anyhow!(
                    "CSV record batch exceeds the bounded Encode decode chunk"
                )));
            }
            if self.buffer_start == self.buffer_end && !self.eof {
                match self.reader.read(&mut self.buffer) {
                    Ok(0) => self.eof = true,
                    Ok(read) => {
                        self.buffer_start = 0;
                        self.buffer_end = read;
                    }
                    Err(error) => {
                        self.finished = true;
                        return Some(Err(anyhow::Error::from(error)));
                    }
                }
            }
            if self.buffer_start == self.buffer_end && self.eof {
                if let Err(error) = self.decoder.decode(&[]) {
                    self.finished = true;
                    return Some(Err(anyhow::Error::from(error)));
                }
                self.finished = true;
                return self.flush();
            }
            let remaining = self
                .max_raw_batch_bytes
                .saturating_sub(self.decoded_since_flush);
            let available = self.buffer_end.saturating_sub(self.buffer_start);
            let offered = available.min(usize::try_from(remaining).unwrap_or(usize::MAX));
            match self
                .decoder
                .decode(&self.buffer[self.buffer_start..self.buffer_start + offered])
            {
                Ok(0) => {
                    if let Some(batch) = self.flush() {
                        return Some(batch);
                    }
                }
                Ok(decoded) => {
                    self.buffer_start += decoded;
                    self.decoded_since_flush =
                        self.decoded_since_flush.saturating_add(decoded as u64);
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(anyhow::Error::from(error)));
                }
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct MemoryTerms {
    constant_bytes: u64,
    bytes_per_row: u64,
}

impl MemoryTerms {
    fn add_constant(&mut self, bytes: u64, kind: &str) -> Result<()> {
        self.constant_bytes = self
            .constant_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("Parquet {kind} bound overflowed"))?;
        Ok(())
    }

    fn add_per_row(&mut self, bytes: u64, kind: &str) -> Result<()> {
        self.bytes_per_row = self
            .bytes_per_row
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("Parquet {kind} bound overflowed"))?;
        Ok(())
    }

    fn double(&mut self, kind: &str) -> Result<()> {
        self.constant_bytes = self
            .constant_bytes
            .checked_mul(2)
            .ok_or_else(|| anyhow!("Parquet {kind} bound overflowed"))?;
        self.bytes_per_row = self
            .bytes_per_row
            .checked_mul(2)
            .ok_or_else(|| anyhow!("Parquet {kind} bound overflowed"))?;
        Ok(())
    }

    fn bound(self, rows: u64, kind: &str) -> Result<u64> {
        self.bytes_per_row
            .checked_mul(rows)
            .and_then(|bytes| self.constant_bytes.checked_add(bytes))
            .ok_or_else(|| anyhow!("Parquet {kind} bound overflowed"))
    }
}

#[derive(Clone, Copy)]
enum ArrowLeafValue {
    Fixed(u64),
    Variable(u64),
}

#[derive(Clone, Copy)]
struct ArrowLeafShape {
    value: ArrowLeafValue,
    wrapper_bytes: u64,
    terminal_bytes: u64,
    fixed_multiplicity: u64,
    variable_ancestor: bool,
}

fn fixed_arrow_leaf_bytes(data_type: &DataType) -> Option<u64> {
    Some(match data_type {
        DataType::Null => 0,
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 2,
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => 3,
        DataType::Int32
        | DataType::UInt32
        | DataType::Float32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Interval(arrow_schema::IntervalUnit::YearMonth)
        | DataType::Decimal32(_, _) => 5,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Interval(arrow_schema::IntervalUnit::DayTime)
        | DataType::Decimal64(_, _) => 9,
        DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano)
        | DataType::Decimal128(_, _) => 17,
        DataType::Decimal256(_, _) => 33,
        DataType::FixedSizeBinary(width) if *width >= 0 => (*width as u64).checked_add(1)?,
        _ => return None,
    })
}

fn variable_arrow_slot_bytes(data_type: &DataType) -> Option<u64> {
    match data_type {
        DataType::Utf8 | DataType::Binary => Some(5),
        DataType::LargeUtf8 | DataType::LargeBinary => Some(9),
        DataType::Utf8View | DataType::BinaryView => Some(17),
        _ => None,
    }
}

fn collect_arrow_leaf_shapes(
    data_type: &DataType,
    wrapper_bytes: u64,
    terminal_bytes: u64,
    fixed_multiplicity: u64,
    variable_ancestor: bool,
    leaves: &mut Vec<ArrowLeafShape>,
) -> Option<()> {
    let recurse = |field: &Field,
                   wrapper_delta,
                   terminal_delta,
                   fixed_factor,
                   has_variable_ancestor,
                   leaves: &mut Vec<_>| {
        collect_arrow_leaf_shapes(
            field.data_type(),
            wrapper_bytes.checked_add(wrapper_delta)?,
            terminal_bytes.checked_add(terminal_delta)?,
            fixed_multiplicity.checked_mul(fixed_factor)?,
            variable_ancestor || has_variable_ancestor,
            leaves,
        )
    };
    match data_type {
        DataType::Struct(fields) => {
            for field in fields {
                recurse(field, 1, 0, 1, false, leaves)?;
            }
        }
        DataType::List(field) => recurse(field, 5, 4, 1, true, leaves)?,
        DataType::LargeList(field) => recurse(field, 9, 8, 1, true, leaves)?,
        DataType::ListView(field) => recurse(field, 9, 0, 1, true, leaves)?,
        DataType::LargeListView(field) => recurse(field, 17, 0, 1, true, leaves)?,
        DataType::FixedSizeList(field, count) if *count > 0 => {
            recurse(field, 1, 0, *count as u64, false, leaves)?
        }
        DataType::Map(field, _) => recurse(field, 5, 4, 1, true, leaves)?,
        DataType::Dictionary(keys, values) => collect_arrow_leaf_shapes(
            values,
            wrapper_bytes.checked_add(fixed_arrow_leaf_bytes(keys)?)?,
            terminal_bytes,
            fixed_multiplicity,
            variable_ancestor,
            leaves,
        )?,
        _ => {
            let value = fixed_arrow_leaf_bytes(data_type)
                .map(ArrowLeafValue::Fixed)
                .or_else(|| variable_arrow_slot_bytes(data_type).map(ArrowLeafValue::Variable))?;
            leaves.push(ArrowLeafShape {
                value,
                wrapper_bytes,
                terminal_bytes,
                fixed_multiplicity,
                variable_ancestor,
            });
        }
    }
    Some(())
}

fn max_fixed_multiplicity(data_type: &DataType, multiplicity: u64) -> Option<u64> {
    match data_type {
        DataType::Struct(fields) => fields.iter().try_fold(0u64, |max, field| {
            max_fixed_multiplicity(field.data_type(), multiplicity).map(|value| max.max(value))
        }),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => max_fixed_multiplicity(field.data_type(), multiplicity),
        DataType::FixedSizeList(field, count) if *count > 0 => {
            max_fixed_multiplicity(field.data_type(), multiplicity.checked_mul(*count as u64)?)
        }
        DataType::Dictionary(_, values) => max_fixed_multiplicity(values, multiplicity),
        _ => Some(multiplicity),
    }
}

fn is_parquet_dictionary_encoded(column: &ColumnChunkMetaData) -> bool {
    column.dictionary_page_offset().is_some()
        || column.encodings().any(|encoding| {
            matches!(
                encoding,
                Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY
            )
        })
}

fn max_plain_byte_array_entry(buf: &[u8], entries: usize) -> Result<u64> {
    let mut offset = 0usize;
    let mut widest = 0u64;
    for _ in 0..entries {
        let length_end = offset
            .checked_add(4)
            .ok_or_else(|| anyhow!("Parquet dictionary page offset overflowed"))?;
        let length = u32::from_le_bytes(
            buf.get(offset..length_end)
                .ok_or_else(|| anyhow!("truncated Parquet dictionary page"))?
                .try_into()
                .map_err(|_| anyhow!("invalid Parquet dictionary entry length"))?,
        ) as usize;
        offset = length_end
            .checked_add(length)
            .ok_or_else(|| anyhow!("Parquet dictionary page offset overflowed"))?;
        if offset > buf.len() {
            return Err(anyhow!("truncated Parquet dictionary page"));
        }
        widest = widest.max(length as u64);
    }
    Ok(widest)
}

struct ParquetDictionaryPages<'a> {
    resident: &'a ResidentFile,
    budget: &'a Arc<crate::source_memory::ByteBudget>,
}

impl ParquetDictionaryPages<'_> {
    fn max_entry_bytes(&self, column: &ColumnChunkMetaData) -> Result<u64> {
        if column.column_type() != PhysicalType::BYTE_ARRAY {
            return Err(anyhow!(
                "dictionary-encoded variable Parquet column is not BYTE_ARRAY"
            ));
        }
        if column.dictionary_page_offset().is_none() {
            return Err(anyhow!(
                "dictionary-encoded Parquet column has no dictionary page"
            ));
        }
        let compressed = u64::try_from(column.compressed_size())
            .map_err(|_| anyhow!("Parquet column has a negative compressed size"))?;
        let uncompressed = u64::try_from(column.uncompressed_size())
            .map_err(|_| anyhow!("Parquet column has a negative decoded size"))?;
        let remote_buffer = if matches!(&self.resident.source, ResidentSource::Remote(_)) {
            REMOTE_READ_CHUNK_BYTES
        } else {
            0
        };
        // Page decode may retain both the compressed input and decoded buffer.
        let scratch_bytes = compressed
            .checked_add(uncompressed)
            .and_then(|bytes| bytes.checked_add(remote_buffer))
            .ok_or_else(|| anyhow!("Parquet dictionary planning bound overflowed"))?;
        let _permit = self.budget.acquire_input_chunk(scratch_bytes)?;
        let reader = Arc::new(self.resident.source.reader(false)?);
        let total_values = usize::try_from(column.num_values())
            .map_err(|_| anyhow!("Parquet column value count does not fit in memory"))?;
        let mut pages = SerializedPageReader::new(reader, column, total_values, None)?;
        match pages.get_next_page()? {
            Some(Page::DictionaryPage {
                buf,
                num_values,
                encoding: Encoding::PLAIN | Encoding::PLAIN_DICTIONARY,
                ..
            }) => max_plain_byte_array_entry(&buf, num_values as usize),
            Some(Page::DictionaryPage { encoding, .. }) => Err(anyhow!(
                "unsupported Parquet dictionary page encoding {encoding}"
            )),
            Some(_) => Err(anyhow!("Parquet dictionary page is missing")),
            None => Err(anyhow!("Parquet dictionary column contains no pages")),
        }
    }
}

fn ceil_div(value: u64, divisor: u64) -> Result<u64> {
    if divisor == 0 {
        return Err(anyhow!("cannot bound Parquet values across zero rows"));
    }
    Ok(value / divisor + u64::from(!value.is_multiple_of(divisor)))
}

fn column_num_values(column: &ColumnChunkMetaData) -> Result<u64> {
    u64::try_from(column.num_values())
        .map_err(|_| anyhow!("Parquet column has a negative value count"))
}

fn add_variable_payload_bound(
    terms: &mut MemoryTerms,
    column: &ColumnChunkMetaData,
    rows: u64,
    shape: ArrowLeafShape,
    dictionary_pages: Option<&ParquetDictionaryPages<'_>>,
) -> Result<()> {
    if let Some(bytes) = column.unencoded_byte_array_data_bytes() {
        return terms.add_constant(
            u64::try_from(bytes)
                .map_err(|_| anyhow!("Parquet variable payload has a negative size"))?,
            "Arrow output",
        );
    }
    let chunk_bytes = u64::try_from(column.uncompressed_size())
        .map_err(|_| anyhow!("Parquet column has a negative decoded size"))?;
    if !is_parquet_dictionary_encoded(column) {
        return terms.add_constant(chunk_bytes, "Arrow output");
    }
    let widest = dictionary_pages
        .ok_or_else(|| anyhow!("Parquet dictionary page reader is unavailable"))?
        .max_entry_bytes(column)?;
    let values = column_num_values(column)?;
    let encoded_dictionary_payload = widest
        .checked_mul(values)
        .ok_or_else(|| anyhow!("Parquet Arrow output bound overflowed"))?;
    let fixed_dictionary_payload = encoded_dictionary_payload
        .checked_mul(shape.fixed_multiplicity)
        .ok_or_else(|| anyhow!("Parquet Arrow output bound overflowed"))?;
    if shape.variable_ancestor || rows == 0 {
        terms.add_constant(fixed_dictionary_payload, "Arrow output")?;
    } else {
        let fixed_bytes_per_row = widest
            .checked_mul(shape.fixed_multiplicity)
            .ok_or_else(|| anyhow!("Parquet Arrow output bound overflowed"))?;
        terms.add_per_row(
            fixed_bytes_per_row.max(ceil_div(encoded_dictionary_payload, rows)?),
            "Arrow output",
        )?;
    }
    let has_plain_data = column.page_encoding_stats().is_none_or(|stats| {
        stats.iter().any(|page| {
            page.page_type != PageType::DICTIONARY_PAGE
                && !matches!(
                    page.encoding,
                    Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY
                )
        })
    });
    if has_plain_data {
        terms.add_constant(chunk_bytes, "Arrow output")?;
    }
    Ok(())
}

fn add_arrow_leaf_bound(
    terms: &mut MemoryTerms,
    shape: ArrowLeafShape,
    column: &ColumnChunkMetaData,
    rows: u64,
    dictionary_pages: Option<&ParquetDictionaryPages<'_>>,
) -> Result<()> {
    let values = column_num_values(column)?;
    let value_bytes = match shape.value {
        ArrowLeafValue::Fixed(bytes) | ArrowLeafValue::Variable(bytes) => bytes,
    };
    let slot_bytes = value_bytes
        .checked_add(shape.wrapper_bytes)
        .ok_or_else(|| anyhow!("Parquet Arrow output bound overflowed"))?;
    // Leaf output scales with the leaf values in a batch. Under a variable-length
    // ancestor the values-per-row is an average; a row that packs far more than
    // its share is caught by the decoded-batch bound in BoundedParquetBatchReader.
    let values_per_row = if shape.variable_ancestor {
        values.div_ceil(rows.max(1))
    } else {
        1
    };
    terms.add_per_row(
        slot_bytes
            .checked_mul(values_per_row)
            .and_then(|bytes| bytes.checked_mul(shape.fixed_multiplicity))
            .ok_or_else(|| anyhow!("Parquet Arrow output bound overflowed"))?,
        "Arrow output",
    )?;
    terms.add_constant(shape.terminal_bytes, "Arrow output")?;
    if matches!(shape.value, ArrowLeafValue::Variable(_)) {
        if column.column_type() != PhysicalType::BYTE_ARRAY {
            return Err(anyhow!(
                "variable Arrow column does not map to a BYTE_ARRAY Parquet leaf"
            ));
        }
        add_variable_payload_bound(terms, column, rows, shape, dictionary_pages)?;
    }
    terms.add_constant(256, "Arrow output")
}

fn add_conservative_arrow_leaf_bound(
    terms: &mut MemoryTerms,
    column: &ColumnChunkMetaData,
    fixed_multiplicity: u64,
) -> Result<()> {
    let values = column_num_values(column)?;
    let chunk_bytes = u64::try_from(column.uncompressed_size())
        .map_err(|_| anyhow!("Parquet column has a negative decoded size"))?;
    terms.add_constant(
        values
            .checked_mul(chunk_bytes.max(1))
            .and_then(|bytes| bytes.checked_mul(fixed_multiplicity))
            .ok_or_else(|| anyhow!("Parquet Arrow output bound overflowed"))?,
        "Arrow output",
    )?;
    terms.add_constant(256, "Arrow output")
}

fn parquet_arrow_output_terms(
    metadata: &ArrowReaderMetadata,
    row_group_index: usize,
    dictionary_pages: Option<&ParquetDictionaryPages<'_>>,
) -> Result<MemoryTerms> {
    let row_group = metadata.metadata().row_group(row_group_index);
    let rows = u64::try_from(row_group.num_rows())
        .map_err(|_| anyhow!("Parquet row group has a negative row count"))?;
    let schema = metadata.metadata().file_metadata().schema_descr();
    if schema.num_columns() != row_group.num_columns() {
        return Err(anyhow!(
            "Parquet row group leaf count does not match its schema"
        ));
    }
    let fields = metadata.schema().fields();
    if fields.len() != schema.root_schema().get_fields().len() {
        return Err(anyhow!("Arrow and Parquet root column counts do not match"));
    }

    let mut terms = MemoryTerms::default();
    let mut leaf_index = 0usize;
    for (root_index, field) in fields.iter().enumerate() {
        let start = leaf_index;
        while leaf_index < schema.num_columns()
            && schema.get_column_root_idx(leaf_index) == root_index
        {
            leaf_index += 1;
        }
        if start == leaf_index {
            return Err(anyhow!("Parquet root column contains no leaves"));
        }
        let mut shapes = Vec::new();
        let mapped = collect_arrow_leaf_shapes(field.data_type(), 0, 0, 1, false, &mut shapes)
            .is_some()
            && shapes.len() == leaf_index - start;
        let fallback_multiplicity = max_fixed_multiplicity(field.data_type(), 1)
            .ok_or_else(|| anyhow!("Parquet Arrow output bound overflowed"))?;
        for (offset, column) in row_group.columns()[start..leaf_index].iter().enumerate() {
            if mapped {
                add_arrow_leaf_bound(&mut terms, shapes[offset], column, rows, dictionary_pages)?;
            } else {
                add_conservative_arrow_leaf_bound(&mut terms, column, fallback_multiplicity)?;
            }
        }
    }
    if leaf_index != schema.num_columns() {
        return Err(anyhow!(
            "Parquet leaf ownership does not match its root columns"
        ));
    }
    terms.double("Arrow output")?;
    Ok(terms)
}

#[derive(Clone, Debug)]
struct ParquetRowGroupMemoryPlan {
    row_group_index: usize,
    total_bytes: u64,
    batch_size: usize,
    arrow_output_bytes: u64,
}

fn parquet_row_group_memory_plan(
    metadata: &ArrowReaderMetadata,
    row_group_index: usize,
    max_decode_bytes: u64,
    dictionary_pages: Option<&ParquetDictionaryPages<'_>>,
) -> Result<ParquetRowGroupMemoryPlan> {
    let row_group = metadata.metadata().row_group(row_group_index);
    let compressed = u64::try_from(row_group.compressed_size())
        .map_err(|_| anyhow!("Parquet row group has a negative compressed size"))?;
    let rows = u64::try_from(row_group.num_rows())
        .map_err(|_| anyhow!("Parquet row group has a negative row count"))?;
    // Only the compressed column chunks are retained across the whole row group.
    // Decoded values are materialised one batch at a time and charged, batch
    // scaled, by the Arrow output terms below -- not held all at once.
    let decoder_constant = compressed;
    let mut decoder_terms = MemoryTerms::default();
    // Level scratch scales with the leaf values in a batch, so charge each column
    // its per-row leaf count. A single row holding the whole row group's values
    // still trips the batch=1 bound below, preserving rejection of oversized rows.
    for column in row_group.columns() {
        let leaves_per_row = column_num_values(column)?.div_ceil(rows.max(1));
        decoder_terms.add_per_row(
            leaves_per_row
                .checked_mul(16)
                .ok_or_else(|| anyhow!("Parquet row group structural size overflowed"))?,
            "row group structural size",
        )?;
    }
    let minimum_decode = decoder_constant
        .checked_add(decoder_terms.bound(1, "row group decode")?)
        .ok_or_else(|| anyhow!("Parquet row group decode bound overflowed"))?;
    if minimum_decode > max_decode_bytes {
        return Err(anyhow!(
            "Parquet row group memory bound {minimum_decode} exceeds the Encode oversized chunk limit {max_decode_bytes}"
        ));
    }

    let output_terms = parquet_arrow_output_terms(metadata, row_group_index, dictionary_pages)?;
    let constant_bytes = decoder_constant
        .checked_add(decoder_terms.constant_bytes)
        .and_then(|bytes| bytes.checked_add(output_terms.constant_bytes))
        .ok_or_else(|| anyhow!("Parquet row group memory bound overflowed"))?;
    let bytes_per_row = decoder_terms
        .bytes_per_row
        .checked_add(output_terms.bytes_per_row)
        .ok_or_else(|| anyhow!("Parquet row group memory bound overflowed"))?
        .max(1);
    let batch_cap = usize::try_from(rows.min(NODE_BATCH_SIZE as u64))
        .unwrap_or(NODE_BATCH_SIZE)
        .max(1);
    let batch_size = usize::try_from(
        (max_decode_bytes.saturating_sub(constant_bytes) / bytes_per_row)
            .min(batch_cap as u64)
            .max(1),
    )
    .unwrap_or(batch_cap);
    let total_bytes = constant_bytes
        .checked_add(
            bytes_per_row
                .checked_mul(batch_size as u64)
                .ok_or_else(|| anyhow!("Parquet row group memory bound overflowed"))?,
        )
        .ok_or_else(|| anyhow!("Parquet row group memory bound overflowed"))?;
    if total_bytes > max_decode_bytes {
        return Err(anyhow!(
            "Parquet row group memory bound {total_bytes} exceeds the Encode oversized chunk limit {max_decode_bytes}"
        ));
    }
    Ok(ParquetRowGroupMemoryPlan {
        row_group_index,
        total_bytes,
        batch_size,
        arrow_output_bytes: output_terms.bound(batch_size as u64, "Arrow output")?,
    })
}

struct BoundedParquetBatchReader {
    resident: Arc<ResidentFile>,
    row_groups: VecDeque<ParquetRowGroupMemoryPlan>,
    current: Option<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>>,
    current_output_bound: u64,
}

impl BoundedParquetBatchReader {
    fn partition(resident: Arc<ResidentFile>, row_groups: Vec<ParquetRowGroupMemoryPlan>) -> Self {
        Self {
            resident,
            row_groups: row_groups.into(),
            current: None,
            current_output_bound: 0,
        }
    }

    fn open_next_row_group(&mut self) -> Result<bool> {
        let Some(memory) = self.row_groups.pop_front() else {
            return Ok(false);
        };
        let metadata = self
            .resident
            .parquet_metadata
            .as_ref()
            .ok_or_else(|| anyhow!("missing Parquet metadata"))?;
        let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(
            self.resident.source.reader(false)?,
            metadata.clone(),
        )
        .with_row_groups(vec![memory.row_group_index])
        .with_batch_size(memory.batch_size)
        .build()?
        .map(|batch| batch.map_err(anyhow::Error::from));
        self.current_output_bound = memory.arrow_output_bytes;
        self.current = Some(Box::new(reader));
        Ok(true)
    }
}

impl Iterator for BoundedParquetBatchReader {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(reader) = &mut self.current {
                if let Some(batch) = reader.next() {
                    return Some(batch.and_then(|batch| {
                        let output_bytes = batch.get_array_memory_size() as u64;
                        if output_bytes > self.current_output_bound {
                            return Err(anyhow!(
                                "Parquet Arrow output {output_bytes} exceeds its planned bound {}",
                                self.current_output_bound
                            ));
                        }
                        Ok(batch)
                    }));
                }
                self.current = None;
            }
            match self.open_next_row_group() {
                Ok(true) => {}
                Ok(false) => return None,
                Err(error) => return Some(Err(error)),
            }
        }
    }
}

struct CsvBatchStream {
    schema: Schema,
    paths: VecDeque<String>,
    transfer: Option<CloudTransfer>,
    max_decode_bytes: u64,
    current: Option<BoundedCsvBatchReader>,
    source_budget: Arc<crate::source_memory::ByteBudget>,
    active_decode_permit: Option<crate::source_memory::BytePermit>,
}

impl CsvBatchStream {
    fn next_record_batch(&mut self) -> Option<Result<RecordBatch>> {
        loop {
            if let Some(reader) = &mut self.current {
                if let Some(batch) = reader.next() {
                    return Some(batch);
                }
                self.current = None;
            }
            let path = self.paths.pop_front()?;
            match open_resident_csv_batch_reader(
                &path,
                self.transfer.as_ref(),
                self.schema.clone(),
                self.max_decode_bytes,
            ) {
                Ok(reader) => self.current = Some(reader),
                Err(error) => {
                    self.paths.clear();
                    return Some(Err(error));
                }
            }
        }
    }
}

impl Iterator for CsvBatchStream {
    type Item = Result<SourceBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // CSV decoders retain input/parser buffers across batches. Keep one
        // full single-chunk permit for the decoder's lifetime; the bounded CSV
        // adapter rejects before it can exceed this ceiling. Parquet decoding
        // is performed by Encode workers through resident partition work.
        if self.active_decode_permit.is_none() {
            match self
                .source_budget
                .acquire_input_chunk(self.source_budget.max_input_chunk_bytes())
            {
                Ok(permit) => self.active_decode_permit = Some(permit),
                Err(error) => return Some(Err(error)),
            }
        }
        let next = self.next_record_batch().map(|result| {
            result.and_then(|batch| {
                let batch_bytes = batch.get_array_memory_size() as u64;
                if batch_bytes > self.source_budget.max_input_chunk_bytes() {
                    return Err(anyhow!(
                        "Encode input chunk exceeds the oversized chunk limit"
                    ));
                }
                let permit = if batch_bytes <= self.source_budget.normal_limit_bytes() {
                    self.source_budget.acquire_dispatched_batch(batch_bytes)?
                } else {
                    self.active_decode_permit.take().ok_or_else(|| {
                        anyhow!("Encode oversized CSV batch is missing its input reservation")
                    })?
                };
                Ok(SourceBatch::budgeted(batch, permit))
            })
        });
        if next.is_none() {
            self.active_decode_permit = None;
        }
        next
    }
}

const REMOTE_READ_CHUNK_BYTES: u64 = 8 * 1024 * 1024;
const REMOTE_METADATA_RANGE_MAX_BYTES: usize = 256 * 1024 * 1024;

/// A local file, a complete pooled cloud download, or a metadata-only remote
/// object. Parquet Encode uses only `Local` and `Cloud`; `Remote` is retained
/// for schema inspection and the independently supported CSV reader.
enum ResidentSource {
    Local(PathBuf),
    Cloud {
        bytes: PooledBytes,
        _permit: crate::source_memory::BytePermit,
    },
    Remote(Arc<RemoteObject>),
}

enum ResidentReader {
    Local { file: fs::File, size: u64 },
    Cloud(Bytes),
    Remote(Arc<RemoteObject>),
}

/// Records failures at the source adapter boundary. Parquet metadata decoding
/// errors are deterministic only when every requested source byte was read
/// successfully; provider, open, range, and short-read failures stay retryable.
struct TrackedChunkReader<R> {
    inner: R,
    read_failed: Arc<AtomicBool>,
}

impl<R> TrackedChunkReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            read_failed: Arc::new(AtomicBool::new(false)),
        }
    }

    fn read_failed(&self) -> bool {
        self.read_failed.load(Ordering::Relaxed)
    }
}

struct TrackedRead<R> {
    inner: R,
    read_failed: Arc<AtomicBool>,
    remaining: u64,
}

impl<R: Read> Read for TrackedRead<R> {
    fn read(&mut self, output: &mut [u8]) -> std_io::Result<usize> {
        match self.inner.read(output) {
            Ok(0) if !output.is_empty() && self.remaining > 0 => {
                self.read_failed.store(true, Ordering::Relaxed);
                Ok(0)
            }
            Ok(read) => {
                self.remaining = self.remaining.saturating_sub(read as u64);
                Ok(read)
            }
            Err(error) => {
                self.read_failed.store(true, Ordering::Relaxed);
                Err(error)
            }
        }
    }
}

impl<R: Length> Length for TrackedChunkReader<R> {
    fn len(&self) -> u64 {
        self.inner.len()
    }
}

impl<R: ChunkReader> ChunkReader for TrackedChunkReader<R> {
    type T = TrackedRead<R::T>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        let remaining = self.len().saturating_sub(start);
        self.inner
            .get_read(start)
            .map(|inner| TrackedRead {
                inner,
                read_failed: Arc::clone(&self.read_failed),
                remaining,
            })
            .inspect_err(|_| {
                self.read_failed.store(true, Ordering::Relaxed);
            })
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        match self.inner.get_bytes(start, length) {
            Ok(bytes) => {
                if bytes.len() != length {
                    self.read_failed.store(true, Ordering::Relaxed);
                }
                Ok(bytes)
            }
            Err(error) => {
                self.read_failed.store(true, Ordering::Relaxed);
                Err(error)
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Parquet footer or metadata is invalid")]
struct InvalidParquetMetadataError {
    #[source]
    source: parquet::errors::ParquetError,
}

struct RemoteObject {
    transfer: CloudTransfer,
    path: object_store::path::Path,
    size: u64,
    condition: Option<CloudObjectCondition>,
}

impl RemoteObject {
    fn read_range(&self, start: u64, end: u64) -> Result<Bytes> {
        if start > end || end > self.size {
            return Err(anyhow!("remote object range is outside the object"));
        }
        let pooled = self
            .transfer
            .get_range_blocking_condition(self.path.clone(), start..end, self.condition.clone())
            .context("remote object range read failed")?;
        Ok(pooled.bytes())
    }
}

struct RemoteRangeReader {
    object: Arc<RemoteObject>,
    position: u64,
    buffered_start: u64,
    buffered: Bytes,
}

impl RemoteRangeReader {
    fn new(object: Arc<RemoteObject>, position: u64) -> Self {
        Self {
            object,
            position,
            buffered_start: position,
            buffered: Bytes::new(),
        }
    }

    fn refill(&mut self) -> std_io::Result<()> {
        if self.position >= self.object.size {
            self.buffered = Bytes::new();
            return Ok(());
        }
        let end = self
            .position
            .saturating_add(REMOTE_READ_CHUNK_BYTES)
            .min(self.object.size);
        // Release the previous pooled range before waiting for the next one.
        // This keeps a one-buffer CloudTransfer configuration deadlock-free.
        self.buffered = Bytes::new();
        self.buffered = self
            .object
            .read_range(self.position, end)
            .map_err(|error| std_io::Error::other(error.into_boxed_dyn_error()))?;
        self.buffered_start = self.position;
        Ok(())
    }
}

impl Read for RemoteRangeReader {
    fn read(&mut self, output: &mut [u8]) -> std_io::Result<usize> {
        if output.is_empty() || self.position >= self.object.size {
            return Ok(0);
        }
        let buffered_end = self
            .buffered_start
            .saturating_add(self.buffered.len() as u64);
        if self.position < self.buffered_start || self.position >= buffered_end {
            self.refill()?;
        }
        let offset = usize::try_from(self.position.saturating_sub(self.buffered_start))
            .map_err(|_| std_io::Error::other("remote object offset overflowed"))?;
        let available = self.buffered.len().saturating_sub(offset);
        let count = available.min(output.len());
        output[..count].copy_from_slice(&self.buffered[offset..offset + count]);
        self.position = self.position.saturating_add(count as u64);
        Ok(count)
    }
}

#[cfg(test)]
mod remote_range_reader_tests {
    use super::*;
    use diskgraph_fs::CloudTransferConfig;
    use object_store::memory::InMemory;
    use object_store::{ObjectStoreExt, PutPayload};

    #[test]
    fn remote_reader_supports_bounded_forward_and_random_ranges() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let path = object_store::path::Path::from("fixture.bin");
        let payload = Bytes::from(
            (0u8..=255)
                .cycle()
                .take(REMOTE_READ_CHUNK_BYTES as usize + 1024)
                .collect::<Vec<_>>(),
        );
        let store = Arc::new(InMemory::new());
        runtime
            .block_on(store.put(&path, PutPayload::from(payload.clone())))
            .unwrap();
        let mut config = CloudTransferConfig::from_env();
        config.buffer_count = 1;
        let transfer = CloudTransfer::new(store, config);
        let object = Arc::new(RemoteObject {
            transfer,
            path,
            size: payload.len() as u64,
            condition: None,
        });
        let mut forward = RemoteRangeReader::new(Arc::clone(&object), 17);
        let mut decoded = Vec::new();
        forward.read_to_end(&mut decoded).unwrap();
        assert_eq!(decoded, payload.slice(17..));
        drop(forward);

        let reader = ResidentReader::Remote(object);
        assert_eq!(reader.get_bytes(100, 33).unwrap(), payload.slice(100..133));
    }
}

impl Length for ResidentReader {
    fn len(&self) -> u64 {
        match self {
            Self::Local { size, .. } => *size,
            Self::Cloud(bytes) => bytes.len() as u64,
            Self::Remote(object) => object.size,
        }
    }
}

impl ChunkReader for ResidentReader {
    type T = Box<dyn Read + Send>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        match self {
            Self::Local { file, .. } => Ok(Box::new(ChunkReader::get_read(file, start)?)),
            Self::Cloud(bytes) => Ok(Box::new(ChunkReader::get_read(bytes, start)?)),
            Self::Remote(object) => {
                if start > object.size {
                    return Err(parquet::errors::ParquetError::General(
                        "remote Parquet read starts beyond the object".to_string(),
                    ));
                }
                Ok(Box::new(RemoteRangeReader::new(Arc::clone(object), start)))
            }
        }
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        match self {
            Self::Local { file, .. } => ChunkReader::get_bytes(file, start, length),
            Self::Cloud(bytes) => ChunkReader::get_bytes(bytes, start, length),
            Self::Remote(object) => {
                if length > REMOTE_METADATA_RANGE_MAX_BYTES {
                    return Err(parquet::errors::ParquetError::General(
                        "remote Parquet metadata range exceeds the limit".to_string(),
                    ));
                }
                let end = start.checked_add(length as u64).ok_or_else(|| {
                    parquet::errors::ParquetError::General(
                        "remote Parquet range overflowed".to_string(),
                    )
                })?;
                object.read_range(start, end).map_err(|source| {
                    parquet::errors::ParquetError::External(source.into_boxed_dyn_error())
                })
            }
        }
    }
}

fn load_parquet_metadata<R: ChunkReader>(reader: R) -> Result<ArrowReaderMetadata> {
    let reader = TrackedChunkReader::new(reader);
    match ArrowReaderMetadata::load(&reader, Default::default()) {
        Ok(metadata) => Ok(metadata),
        Err(error) if reader.read_failed() => Err(error.into()),
        Err(source) => Err(InvalidParquetMetadataError { source }.into()),
    }
}

impl ResidentSource {
    fn reader(&self, willneed: bool) -> Result<ResidentReader> {
        match self {
            Self::Local(path) => {
                let file = fs::File::open(path)
                    .with_context(|| format!("open resident source {}", path.display()))?;
                let size = file
                    .metadata()
                    .with_context(|| format!("stat resident source {}", path.display()))?
                    .len();
                if willneed {
                    advise_file_willneed(&file);
                }
                Ok(ResidentReader::Local { file, size })
            }
            Self::Cloud { bytes, .. } => Ok(ResidentReader::Cloud(bytes.bytes())),
            Self::Remote(object) => Ok(ResidentReader::Remote(Arc::clone(object))),
        }
    }
}

struct ResidentFile {
    source: ResidentSource,
    parquet_metadata: Option<ArrowReaderMetadata>,
}

impl ResidentFile {
    fn load(
        path: &str,
        format: TableFormat,
        transfer: Option<&CloudTransfer>,
        resident_budget: Option<&Arc<crate::source_memory::ByteBudget>>,
        condition: Option<CloudObjectCondition>,
    ) -> Result<Self> {
        let source = if is_cloud_path(path) {
            let transfer =
                transfer.ok_or_else(|| anyhow!("cloud input transfer unavailable for {path}"))?;
            let object_path = object_path_from_cloud_url(path)?;
            ensure!(
                format != TableFormat::Parquet || condition.is_some(),
                "cloud Parquet object {path} was not pinned before loading"
            );
            let size = match condition
                .as_ref()
                .and_then(CloudObjectCondition::pinned_size)
            {
                Some(size) => size,
                None => transfer.head_blocking(object_path.clone())?.size,
            };
            match (&format, resident_budget) {
                (TableFormat::Parquet, Some(resident_budget)) => {
                    let permit = resident_budget.acquire_resident_object(size)?;
                    let bytes =
                        download_cloud_bytes_condition(transfer, path, condition.clone())
                            .with_context(|| format!("download resident Parquet object {path}"))?;
                    let received_size = bytes.bytes().len() as u64;
                    if received_size != size {
                        return Err(anyhow!(
                            "resident Parquet object {path} changed size while downloading: expected {size} bytes, received {received_size}"
                        ));
                    }
                    ResidentSource::Cloud {
                        bytes,
                        _permit: permit,
                    }
                }
                _ => ResidentSource::Remote(Arc::new(RemoteObject {
                    transfer: transfer.clone(),
                    path: object_path,
                    size,
                    condition,
                })),
            }
        } else {
            fs::metadata(path).with_context(|| format!("stat {path}"))?;
            ResidentSource::Local(PathBuf::from(path))
        };
        let parquet_metadata = if format == TableFormat::Parquet {
            // Advise the local file once while loading metadata. Partition
            // readers must not repeatedly prefetch the complete file.
            let reader = source.reader(true)?;
            Some(load_parquet_metadata(reader)?)
        } else {
            None
        };
        Ok(Self {
            source,
            parquet_metadata,
        })
    }
}

struct ParquetPartition {
    resident: Arc<ResidentFile>,
    row_groups: Vec<ParquetRowGroupMemoryPlan>,
    max_decode_bytes: u64,
}

struct PendingParquetPartition {
    resident: Arc<ResidentFile>,
    row_groups: Vec<usize>,
}

const PARQUET_PARTITION_TARGET_BYTES: u64 = 256 * 1024 * 1024;

fn plan_resident_parquet_partitions(resident: &ResidentFile) -> Result<Vec<Vec<usize>>> {
    let metadata = resident
        .parquet_metadata
        .as_ref()
        .ok_or_else(|| anyhow!("missing resident Parquet metadata"))?;
    let row_groups = metadata.metadata().row_groups();
    if row_groups.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    let mut partitions = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0u64;
    for (index, row_group) in row_groups.iter().enumerate() {
        current.push(index);
        current_bytes = current_bytes.saturating_add(row_group.compressed_size().max(0) as u64);
        if current_bytes >= PARQUET_PARTITION_TARGET_BYTES {
            partitions.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
    }
    if !current.is_empty() {
        partitions.push(current);
    }
    Ok(partitions)
}

fn resident_partition_memory_plans(
    resident: &ResidentFile,
    row_groups: &[usize],
    max_decode_bytes: u64,
    source_budget: &Arc<crate::source_memory::ByteBudget>,
) -> Result<(Vec<ParquetRowGroupMemoryPlan>, u64)> {
    let metadata = resident
        .parquet_metadata
        .as_ref()
        .ok_or_else(|| anyhow!("missing resident Parquet metadata"))?;
    let dictionary_pages = ParquetDictionaryPages {
        resident,
        budget: source_budget,
    };
    let mut plans = Vec::with_capacity(row_groups.len());
    let mut bound = 0u64;
    for row_group in row_groups {
        let plan = parquet_row_group_memory_plan(
            metadata,
            *row_group,
            max_decode_bytes,
            Some(&dictionary_pages),
        )?;
        bound = bound.max(plan.total_bytes);
        plans.push(plan);
    }
    Ok((plans, bound))
}

fn catch_source_worker_panic<T>(run: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|payload| {
        let detail = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic");
        Err(anyhow!("resident Parquet loader panicked: {detail}"))
    })
}

struct ResidentParquetWorkStream {
    inputs: VecDeque<ParquetInput>,
    transfer: Option<CloudTransfer>,
    resident_budget: Option<Arc<crate::source_memory::ByteBudget>>,
    receiver: Option<std::sync::mpsc::Receiver<Result<VecDeque<PendingParquetPartition>>>>,
    active_file: Option<Result<VecDeque<PendingParquetPartition>>>,
    loaders: Vec<std::thread::JoinHandle<()>>,
    loader_threads: usize,
    failed: bool,
}

impl ResidentParquetWorkStream {
    fn open(
        table: &TableSpec,
        transfer: Option<&CloudTransfer>,
        resident_budget: Option<Arc<crate::source_memory::ByteBudget>>,
        loader_threads: usize,
    ) -> Result<Self> {
        let inputs = VecDeque::from(parquet_inputs(table));
        if inputs.is_empty() {
            return Err(anyhow!("no input files for table {}", table.name));
        }
        Ok(Self {
            loader_threads: loader_threads.max(1).min(inputs.len()),
            inputs,
            transfer: transfer.cloned(),
            resident_budget,
            receiver: None,
            active_file: None,
            loaders: Vec::new(),
            failed: false,
        })
    }

    fn load(
        input: ParquetInput,
        transfer: Option<&CloudTransfer>,
        resident_budget: Option<&Arc<crate::source_memory::ByteBudget>>,
    ) -> Result<VecDeque<PendingParquetPartition>> {
        let resident = Arc::new(ResidentFile::load(
            &input.path,
            TableFormat::Parquet,
            transfer,
            resident_budget,
            input.condition,
        )?);
        Ok(plan_resident_parquet_partitions(&resident)?
            .into_iter()
            .map(|row_groups| PendingParquetPartition {
                resident: Arc::clone(&resident),
                row_groups,
            })
            .collect())
    }

    fn enable_unordered(&mut self) -> Result<()> {
        if self.receiver.is_some() || self.inputs.is_empty() {
            return Ok(());
        }
        let worker_count = self.loader_threads.min(self.inputs.len());
        let inputs = Arc::new(std::sync::Mutex::new(std::mem::take(&mut self.inputs)));
        // A rendezvous keeps at most one loaded file per worker outside the
        // consumer, bounding metadata and resident objects by loader threads.
        let (sender, receiver) = std::sync::mpsc::sync_channel(0);
        for worker in 0..worker_count {
            let inputs = Arc::clone(&inputs);
            let transfer = self.transfer.clone();
            let resident_budget = self.resident_budget.clone();
            let worker_sender = sender.clone();
            let loader = std::thread::Builder::new()
                .name(format!("resident-parquet-loader-{worker}"))
                .spawn(move || loop {
                    let input = match inputs.lock() {
                        Ok(mut inputs) => inputs.pop_front(),
                        Err(_) => return,
                    };
                    let Some(input) = input else {
                        return;
                    };
                    let loaded = catch_source_worker_panic(|| {
                        Self::load(input, transfer.as_ref(), resident_budget.as_ref())
                    });
                    let failed = loaded.is_err();
                    if worker_sender.send(loaded).is_err() || failed {
                        return;
                    }
                })
                .with_context(|| format!("start resident Parquet loader {worker}"));
            match loader {
                Ok(loader) => self.loaders.push(loader),
                Err(error) => {
                    drop(receiver);
                    drop(sender);
                    for loader in self.loaders.drain(..) {
                        let _ = loader.join();
                    }
                    return Err(error);
                }
            }
        }
        drop(sender);
        self.receiver = Some(receiver);
        Ok(())
    }

    fn finish_loaders(&mut self) -> Result<()> {
        self.receiver = None;
        let mut panicked = false;
        for loader in self.loaders.drain(..) {
            panicked |= loader.join().is_err();
        }
        if panicked {
            Err(anyhow!("resident Parquet loader panicked"))
        } else {
            Ok(())
        }
    }

    fn next_partition(
        &mut self,
        source_budget: &Arc<crate::source_memory::ByteBudget>,
        ordered: bool,
    ) -> Option<Result<ParquetPartition>> {
        if self.failed {
            return None;
        }
        loop {
            if let Some(active) = &mut self.active_file {
                let partition = match active {
                    Ok(partitions) => partitions.pop_front().map(Ok),
                    Err(_) => None,
                };
                let partition = match partition {
                    Some(partition) => partition,
                    None => {
                        let pending = self.active_file.take().expect("active resident file");
                        match pending {
                            Ok(_) => continue,
                            Err(error) => {
                                self.failed = true;
                                return Some(Err(error));
                            }
                        }
                    }
                };
                let partition = match partition {
                    Ok(partition) => partition,
                    Err(error) => return Some(Err(error)),
                };
                let max_decode_bytes = source_budget.max_input_chunk_bytes();
                let (row_groups, required_bytes) = match resident_partition_memory_plans(
                    &partition.resident,
                    &partition.row_groups,
                    max_decode_bytes,
                    source_budget,
                ) {
                    Ok(plans) => plans,
                    Err(error) => return Some(Err(error)),
                };
                return Some(Ok(ParquetPartition {
                    resident: partition.resident,
                    row_groups,
                    max_decode_bytes: required_bytes,
                }));
            }

            if ordered && self.receiver.is_none() {
                let input = self.inputs.pop_front()?;
                let loaded = catch_source_worker_panic(|| {
                    Self::load(input, self.transfer.as_ref(), self.resident_budget.as_ref())
                });
                self.active_file = Some(loaded);
                continue;
            }
            if let Err(error) = self.enable_unordered() {
                self.failed = true;
                return Some(Err(error));
            }
            let receiver = self.receiver.as_ref()?;
            let event = match receiver.recv() {
                Ok(event) => event,
                Err(_) => {
                    if let Err(error) = self.finish_loaders() {
                        self.failed = true;
                        return Some(Err(error));
                    }
                    return None;
                }
            };
            self.active_file = Some(event);
        }
    }
}

impl Drop for ResidentParquetWorkStream {
    fn drop(&mut self) {
        self.receiver = None;
        self.active_file = None;
        let _ = self.finish_loaders();
    }
}

enum TableSourceStreamInner<'a> {
    Csv(Box<CsvBatchStream>),
    Databricks(Box<crate::databricks::DatabricksQueryStream<'a>>),
    Parquet(ResidentParquetWorkStream),
}

struct TableSourceStream<'a> {
    inner: TableSourceStreamInner<'a>,
    source_budget: Arc<crate::source_memory::ByteBudget>,
    validation: Arc<SourceValidationState>,
    active_context_reader: Option<(
        BoundedParquetBatchReader,
        Arc<crate::source_memory::BytePermit>,
        Option<SourceBatch>,
    )>,
    input_wait_nanos: u64,
}

impl<'a> TableSourceStream<'a> {
    fn new(
        inner: TableSourceStreamInner<'a>,
        source_budget: Arc<crate::source_memory::ByteBudget>,
        validation: Arc<SourceValidationState>,
    ) -> Self {
        Self {
            inner,
            source_budget,
            validation,
            active_context_reader: None,
            input_wait_nanos: 0,
        }
    }

    fn next_batch(&mut self) -> Option<Result<SourceBatch>> {
        let input_start = std::time::Instant::now();
        let next = match &mut self.inner {
            TableSourceStreamInner::Csv(stream) => stream.next(),
            TableSourceStreamInner::Databricks(stream) => stream
                .next()
                .map(|result| result.map(SourceBatch::databricks)),
            TableSourceStreamInner::Parquet(_) => {
                return Some(Err(anyhow!(
                    "Parquet source requested as a materialized batch"
                )))
            }
        };
        self.input_wait_nanos = self
            .input_wait_nanos
            .saturating_add(input_start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        next
    }

    fn next_parquet_partition(&mut self, ordered: bool) -> Option<Result<ParquetPartition>> {
        let TableSourceStreamInner::Parquet(stream) = &mut self.inner else {
            return Some(Err(anyhow!(
                "non-Parquet source requested as a resident partition"
            )));
        };
        let input_start = std::time::Instant::now();
        let next = stream.next_partition(&self.source_budget, ordered);
        self.input_wait_nanos = self
            .input_wait_nanos
            .saturating_add(input_start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        next
    }

    fn next_context_batch(&mut self) -> Option<Result<SourceBatch>> {
        loop {
            if let Some((reader, permit, _)) = &mut self.active_context_reader {
                match reader.next() {
                    Some(Ok(batch)) => {
                        if let Err(error) = self.validation.observe(&batch) {
                            return Some(Err(error));
                        }
                        return Some(Ok(
                            SourceBatch::resident_parquet(batch, Arc::clone(permit)).observed()
                        ));
                    }
                    Some(Err(error)) => return Some(Err(error)),
                    None => self.active_context_reader = None,
                }
            }
            if !matches!(&self.inner, TableSourceStreamInner::Parquet(_)) {
                return self.next_batch().map(|result| {
                    result.and_then(|batch| {
                        self.validation.observe(&batch.batch)?;
                        Ok(batch.observed())
                    })
                });
            }
            let partition = match self.next_parquet_partition(true)? {
                Ok(partition) => partition,
                Err(error) => return Some(Err(error)),
            };
            let permit = match self
                .source_budget
                .acquire_input_chunk(partition.max_decode_bytes)
            {
                Ok(permit) => permit,
                Err(error) => return Some(Err(error)),
            };
            let permit = Arc::new(permit);
            self.active_context_reader = Some((
                BoundedParquetBatchReader::partition(partition.resident, partition.row_groups),
                permit,
                None,
            ));
        }
    }

    fn defer_context_remainder(&mut self, batch: SourceBatch) -> Option<SourceBatch> {
        match &mut self.active_context_reader {
            Some((_, _, remainder)) => {
                debug_assert!(remainder.is_none());
                *remainder = Some(batch);
                None
            }
            None => Some(batch),
        }
    }

    fn databricks_metrics(&self) -> Option<crate::databricks::QueryMetrics> {
        match &self.inner {
            TableSourceStreamInner::Databricks(stream) => Some(stream.metrics()),
            TableSourceStreamInner::Csv(_) | TableSourceStreamInner::Parquet(_) => None,
        }
    }

    fn input_wait_nanos(&self) -> u64 {
        self.input_wait_nanos
    }
}

impl Iterator for TableSourceStream<'_> {
    type Item = Result<SourceWork>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((reader, permit, first_batch)) = self.active_context_reader.take() {
            return Some(Ok(SourceWork::ParquetReader {
                reader,
                permit,
                first_batch,
            }));
        }
        if !matches!(&self.inner, TableSourceStreamInner::Parquet(_)) {
            return self
                .next_batch()
                .map(|result| result.map(SourceWork::Batch));
        }
        self.next_parquet_partition(false)
            .map(|result| result.map(SourceWork::Parquet))
    }
}

fn open_csv_batch_stream(
    table: &TableSpec,
    transfer: Option<&CloudTransfer>,
    source_budget: Arc<crate::source_memory::ByteBudget>,
) -> Result<CsvBatchStream> {
    if table.get_paths().is_empty() {
        return Err(anyhow!("no input files for table {}", table.name));
    }
    let max_decode_bytes = source_budget.max_input_chunk_bytes();
    Ok(CsvBatchStream {
        schema: table_schema(table)?,
        paths: table.get_paths().iter().cloned().collect(),
        transfer: transfer.cloned(),
        max_decode_bytes,
        current: None,
        source_budget,
        active_decode_permit: None,
    })
}

fn open_resident_csv_batch_reader(
    path: &str,
    transfer: Option<&CloudTransfer>,
    schema: Schema,
    max_decode_bytes: u64,
) -> Result<BoundedCsvBatchReader> {
    let resident = ResidentFile::load(path, TableFormat::Csv, transfer, None, None)?;
    BoundedCsvBatchReader::new(
        resident.source.reader(true)?.get_read(0)?,
        schema,
        max_decode_bytes,
    )
}

#[cfg(test)]
mod bounded_file_decoder_tests {
    use super::*;
    use arrow_array::builder::{FixedSizeListBuilder, Float64Builder, Int64Builder, ListBuilder};
    use arrow_array::{BooleanArray, FixedSizeBinaryArray, StringArray};
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use std::io::Cursor;

    fn resident_parquet(
        batch: &RecordBatch,
        properties: Option<WriterProperties>,
    ) -> (tempfile::TempDir, Arc<ResidentFile>) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input.parquet");
        super::resident_parquet_tests::write_parquet(&path, batch, properties);
        (directory, super::resident_parquet_tests::resident(&path))
    }

    fn read_parquet(reader: BoundedParquetBatchReader, max_batch_bytes: u64) -> (usize, usize) {
        reader.fold((0, 0), |(rows, batches), batch| {
            let batch = batch.unwrap();
            assert!(batch.get_array_memory_size() as u64 <= max_batch_bytes);
            (rows + batch.num_rows(), batches + 1)
        })
    }

    fn parquet_plan(
        resident: &Arc<ResidentFile>,
        max_decode_bytes: u64,
    ) -> Result<ParquetRowGroupMemoryPlan> {
        let budget = crate::source_memory::ByteBudget::with_oversized_limit(
            max_decode_bytes,
            max_decode_bytes,
        )?;
        let pages = ParquetDictionaryPages {
            resident,
            budget: &budget,
        };
        parquet_row_group_memory_plan(
            resident.parquet_metadata.as_ref().unwrap(),
            0,
            max_decode_bytes,
            Some(&pages),
        )
    }

    fn planned_reader(
        resident: Arc<ResidentFile>,
        max_decode_bytes: u64,
    ) -> Result<(ParquetRowGroupMemoryPlan, BoundedParquetBatchReader)> {
        let plan = parquet_plan(&resident, max_decode_bytes)?;
        let reader = BoundedParquetBatchReader::partition(resident, vec![plan.clone()]);
        Ok((plan, reader))
    }

    #[test]
    fn csv_rejects_a_record_before_crossing_the_decode_ceiling() {
        let max_decode_bytes = 256 * 1024;
        let mut csv = b"value\n".to_vec();
        csv.extend(std::iter::repeat_n(b'x', max_decode_bytes as usize / 4 + 1));
        csv.push(b'\n');
        let schema = Schema::new(vec![Field::new("value", DataType::Utf8, true)]);
        let mut reader =
            BoundedCsvBatchReader::new(Box::new(Cursor::new(csv)), schema, max_decode_bytes)
                .unwrap();
        let error = reader.next().unwrap().unwrap_err();
        assert!(error
            .to_string()
            .contains("CSV record batch exceeds the bounded Encode decode chunk"));
        assert!(reader.next().is_none());
    }

    #[test]
    fn parquet_rejects_an_unadmitted_row_group_before_arrow_decode() {
        let batch = super::resident_parquet_tests::int_batch("id", 0..1024);
        let (_directory, resident) = resident_parquet(&batch, None);
        let error = parquet_plan(&resident, 1024).unwrap_err();
        assert!(error.to_string().contains("Parquet row group memory bound"));
    }

    #[test]
    fn parquet_accounts_for_dictionary_expanded_fixed_width_output() {
        let value = vec![7u8; 3000];
        let values = std::iter::repeat_n(value.as_slice(), 100);
        let array = FixedSizeBinaryArray::try_from_iter(values).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::FixedSizeBinary(3000),
            false,
        )]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap();
        let (_directory, resident) = resident_parquet(&batch, None);
        let memory_plan = parquet_plan(&resident, 1024 * 1024).unwrap();
        assert!(memory_plan.total_bytes > batch.get_array_memory_size() as u64);
        let max_decode_bytes = memory_plan.total_bytes;
        let reader = BoundedParquetBatchReader::partition(resident, vec![memory_plan]);
        let (rows, _batches) = read_parquet(reader, max_decode_bytes);
        assert_eq!(rows, 100);
    }

    #[test]
    fn parquet_admits_uniform_fixed_width_lists_by_batch_scaling() {
        // Many rows, each a moderate uniform list: the leaf-value count is huge
        // but no single row is large. Charging num_values as a row-group constant
        // rejected this; batch-scaling the per-row leaf count admits it.
        let rows = 10_000usize;
        let per_row = 200usize;
        let mut builder = ListBuilder::new(Int64Builder::new());
        for _ in 0..rows {
            for _ in 0..per_row {
                builder.values().append_value(0);
            }
            builder.append(true);
        }
        let array = builder.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            array.data_type().clone(),
            false,
        )]));
        let source = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        let (_directory, resident) = resident_parquet(&source, None);
        let max_decode_bytes = 8 * 1024 * 1024;
        assert!((rows as u64 * per_row as u64) * 16 > max_decode_bytes);
        let plan = parquet_plan(&resident, max_decode_bytes).unwrap();
        let reader = BoundedParquetBatchReader::partition(resident, vec![plan]);
        let (decoded, _batches) = read_parquet(reader, max_decode_bytes);
        assert_eq!(decoded, rows);
    }

    #[test]
    fn parquet_keeps_plain_variable_width_batches_large() {
        let array = StringArray::from_iter_values((0..100_000).map(|value| value.to_string()));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            false,
        )]));
        let source = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap();
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::None)
            .build();
        let (_directory, resident) = resident_parquet(&source, Some(properties));
        let metadata = resident.parquet_metadata.as_ref().unwrap();
        assert!(metadata
            .metadata()
            .row_group(0)
            .column(0)
            .unencoded_byte_array_data_bytes()
            .is_none());
        let memory_plan = parquet_plan(&resident, 16 * 1024 * 1024).unwrap();
        assert_eq!(memory_plan.batch_size, NODE_BATCH_SIZE);
    }

    #[test]
    fn parquet_bounds_skewed_plain_variable_width_output() {
        let large = "x".repeat(2 * 1024 * 1024);
        let array = StringArray::from_iter_values(
            std::iter::repeat_n("", NODE_BATCH_SIZE).chain(std::iter::once(large.as_str())),
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            false,
        )]));
        let source = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap();
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::None)
            .build();
        let (_directory, resident) = resident_parquet(&source, Some(properties));
        let max_decode_bytes = 32 * 1024 * 1024;
        let memory_plan = parquet_plan(&resident, max_decode_bytes).unwrap();
        assert_eq!(memory_plan.batch_size, NODE_BATCH_SIZE);

        let reader = BoundedParquetBatchReader::partition(resident, vec![memory_plan.clone()]);
        let (rows, batches) = read_parquet(reader, memory_plan.total_bytes);
        assert_eq!(rows, source.num_rows());
        assert_eq!(batches, 2);
    }

    #[test]
    fn parquet_limits_dictionary_expanded_variable_width_batches() {
        let value = "x".repeat(16 * 1024);
        let array = StringArray::from_iter_values(std::iter::repeat_n(value.as_str(), 1_000));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            false,
        )]));
        let source = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap();
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(true)
            .set_statistics_enabled(EnabledStatistics::None)
            .build();
        let (_directory, resident) = resident_parquet(&source, Some(properties));
        let max_decode_bytes = 64 * 1024 * 1024;
        let memory_plan = parquet_plan(&resident, max_decode_bytes).unwrap();
        assert_eq!(memory_plan.batch_size, source.num_rows());
        assert!(memory_plan.total_bytes <= max_decode_bytes);

        let reader = BoundedParquetBatchReader::partition(resident, vec![memory_plan.clone()]);
        let (rows, batches) = read_parquet(reader, memory_plan.total_bytes);
        assert_eq!(rows, source.num_rows());
        assert_eq!(batches, 1);
    }

    #[test]
    fn parquet_admits_bounded_fixed_width_lists_without_leaf_count_product() {
        let mut values = ListBuilder::new(Float64Builder::new());
        for row in 0..1024 {
            for offset in 0..6 {
                values.values().append_value((row + offset) as f64);
            }
            values.append(true);
        }
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        )]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values.finish())]).unwrap();
        let (_directory, resident) = resident_parquet(&batch, None);
        let max_decode_bytes = 1024 * 1024;
        let memory_plan = parquet_plan(&resident, max_decode_bytes).unwrap();
        assert!(memory_plan.total_bytes <= max_decode_bytes);

        let reader = BoundedParquetBatchReader::partition(resident, vec![memory_plan]);
        let (rows, _) = read_parquet(reader, max_decode_bytes);
        assert_eq!(rows, 1024);
    }

    #[test]
    fn parquet_batches_wide_scalar_row_groups_under_the_decode_ceiling() {
        let rows = 20_000;
        let columns = 20;
        let fields = (0..columns)
            .map(|index| Field::new(format!("flag_{index}"), DataType::Boolean, false))
            .collect::<Vec<_>>();
        let arrays = (0..columns)
            .map(|index| {
                Arc::new(BooleanArray::from(
                    (0..rows)
                        .map(|row| (row + index) % 2 == 0)
                        .collect::<Vec<_>>(),
                )) as ArrayRef
            })
            .collect();
        let source = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        let (_directory, resident) = resident_parquet(&source, None);
        let max_decode_bytes = 2 * 1024 * 1024;
        let old_structural = rows as u64 * columns as u64 * 16;
        assert!(old_structural > max_decode_bytes);

        let (plan, reader) = planned_reader(resident, max_decode_bytes).unwrap();
        assert!(plan.batch_size < rows);
        let (decoded_rows, batches) = read_parquet(reader, plan.arrow_output_bytes);
        assert_eq!(decoded_rows, rows);
        assert!(batches > 1);
    }

    #[test]
    fn parquet_bounds_nested_lists_per_leaf_value() {
        let rows = 1_000;
        let mut outer = ListBuilder::new(ListBuilder::new(Int64Builder::new()));
        for row in 0..rows {
            for inner in 0..10 {
                for value in 0..10 {
                    outer
                        .values()
                        .values()
                        .append_value((row + inner + value) as i64);
                }
                outer.values().append(true);
            }
            outer.append(true);
        }
        let values = outer.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            values.data_type().clone(),
            false,
        )]));
        let source = RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap();
        let (_directory, resident) = resident_parquet(&source, None);
        let max_decode_bytes = 8 * 1024 * 1024;
        let row_group = resident
            .parquet_metadata
            .as_ref()
            .unwrap()
            .metadata()
            .row_group(0);
        let leaf_values = column_num_values(row_group.column(0)).unwrap();
        let old_fallback = u64::try_from(row_group.total_byte_size())
            .unwrap()
            .checked_mul(leaf_values)
            .and_then(|bytes| bytes.checked_mul(2))
            .unwrap();
        assert!(old_fallback > max_decode_bytes);

        let (plan, reader) = planned_reader(resident, max_decode_bytes).unwrap();
        let (decoded_rows, _) = read_parquet(reader, plan.arrow_output_bytes);
        assert_eq!(decoded_rows, rows);
    }

    #[test]
    fn parquet_bounds_null_nested_fixed_size_list_padding() {
        const ROWS: usize = 32;
        const OUTER: i32 = 64;
        const INNER: i32 = 64;
        let inner = FixedSizeListBuilder::new(Int64Builder::new(), INNER);
        let mut outer = FixedSizeListBuilder::new(inner, OUTER);
        for _ in 0..ROWS {
            for _ in 0..OUTER {
                for _ in 0..INNER {
                    outer.values().values().append_null();
                }
                outer.values().append(false);
            }
            outer.append(false);
        }
        let values = outer.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            values.data_type().clone(),
            true,
        )]));
        let source = RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap();
        let (_directory, resident) = resident_parquet(&source, None);
        let leaf_values = column_num_values(
            resident
                .parquet_metadata
                .as_ref()
                .unwrap()
                .metadata()
                .row_group(0)
                .column(0),
        )
        .unwrap();
        let fixed_slots = ROWS as u64 * OUTER as u64 * INNER as u64;
        assert!(leaf_values < fixed_slots);

        let max_decode_bytes = 4 * 1024 * 1024;
        let (plan, reader) = planned_reader(resident, max_decode_bytes).unwrap();
        let (decoded_rows, _) = read_parquet(reader, plan.arrow_output_bytes);
        assert_eq!(decoded_rows, ROWS);
    }

    #[test]
    fn fixed_width_list_bound_reserves_arrow_buffer_headroom() {
        let mut values = ListBuilder::new(Float64Builder::new());
        for row in 0..1024 {
            for offset in 0..6 {
                values.values().append_value((row + offset) as f64);
            }
            values.append(true);
        }
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        )]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values.finish())]).unwrap();
        let (_directory, resident) = resident_parquet(&batch, None);
        let memory_plan = parquet_plan(&resident, 16 * 1024 * 1024).unwrap();
        assert!(memory_plan.total_bytes >= 2 * 1024 * 6 * 8);
        assert!(memory_plan.total_bytes >= batch.get_array_memory_size() as u64 * 2);
    }

    #[test]
    fn parquet_bounds_legacy_dictionary_strings_next_to_repeated_values() {
        let rows = 1_000;
        let mut values = ListBuilder::new(Float64Builder::new());
        for row in 0..rows {
            for offset in 0..6 {
                values.values().append_value((row + offset) as f64);
            }
            values.append(true);
        }
        let labels = (0..rows)
            .map(|row| format!("label-{:03}-{}", row % 16, "x".repeat(128)))
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                false,
            ),
            Field::new("label", DataType::Utf8, false),
        ]));
        let source = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(values.finish()),
                Arc::new(StringArray::from_iter_values(labels.iter())),
            ],
        )
        .unwrap();
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(true)
            .set_statistics_enabled(EnabledStatistics::None)
            .build();
        let (_directory, resident) = resident_parquet(&source, Some(properties));
        let row_group = resident
            .parquet_metadata
            .as_ref()
            .unwrap()
            .metadata()
            .row_group(0);
        let label = row_group.column(1);
        assert!(is_parquet_dictionary_encoded(label));
        assert!(label.unencoded_byte_array_data_bytes().is_none());

        let max_decode_bytes = 4 * 1024 * 1024;
        let (plan, reader) = planned_reader(resident, max_decode_bytes).unwrap();
        let (decoded_rows, _) = read_parquet(reader, plan.arrow_output_bytes);
        assert_eq!(decoded_rows, rows);
    }

    #[test]
    fn parquet_bounds_plain_pages_after_dictionary_fallback() {
        let long = "x".repeat(400);
        let values = (0..4_000)
            .map(|entry| format!("{entry:08}"))
            .chain(std::iter::repeat_n(long, 20_000))
            .collect::<Vec<_>>();
        let decoded_payload = values.iter().map(|value| value.len() as u64).sum::<u64>();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "label",
            DataType::Utf8,
            false,
        )]));
        let source =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values.clone()))])
                .unwrap();
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(true)
            .set_statistics_enabled(EnabledStatistics::None)
            .set_dictionary_page_size_limit(4 * 1024)
            .set_data_page_size_limit(4 * 1024)
            .build();
        let (_directory, resident) = resident_parquet(&source, Some(properties));
        let chunk = resident
            .parquet_metadata
            .as_ref()
            .unwrap()
            .metadata()
            .row_group(0)
            .column(0);
        assert!(chunk.page_encoding_stats().unwrap().iter().any(|page| {
            page.page_type != PageType::DICTIONARY_PAGE && page.encoding == Encoding::PLAIN
        }));

        let max_decode_bytes = 64 * 1024 * 1024;
        let (plan, reader) = planned_reader(resident, max_decode_bytes).unwrap();
        assert!(plan.arrow_output_bytes >= decoded_payload);
        let (decoded_rows, _) = read_parquet(reader, plan.arrow_output_bytes);
        assert_eq!(decoded_rows, values.len());
    }

    #[test]
    fn parquet_rejects_a_single_oversized_repeated_row() {
        let mut values = ListBuilder::new(Int64Builder::new());
        for _ in 0..200_000 {
            values.values().append_value(7);
        }
        values.append(true);
        let values = values.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            values.data_type().clone(),
            false,
        )]));
        let source = RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap();
        let (_directory, resident) = resident_parquet(&source, None);
        let max_decode_bytes = 1024 * 1024;
        let row_group = resident
            .parquet_metadata
            .as_ref()
            .unwrap()
            .metadata()
            .row_group(0);
        assert!(u64::try_from(row_group.total_byte_size()).unwrap() < max_decode_bytes);

        let error = parquet_plan(&resident, max_decode_bytes).unwrap_err();
        assert!(error.to_string().contains("Parquet row group memory bound"));
    }
}

#[derive(Clone)]
struct RelationDef {
    name: String,
    src_table: String,
    dst_table: String,
    sort_key: Option<RelationSortKeyConfig>,
    reverse_of: Option<String>,
}

#[derive(Clone)]
struct RelationshipRuntime {
    name_forward: String,
    name_reverse: String,
    src_table: String,
    src_column: String,
    dst_table: String,
    /// True when both forward and reverse edges should be emitted. False
    /// emits forward only — skips reverse RelationDef, reverse_batcher
    /// pushes, and DestFkeyBatcher pushes for this FK.
    bidirectional: bool,
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
            tracing::warn!("unsupported timestamp bitwidth {other}, using 32-bit");
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
            tracing::warn!("unsupported timestamp unit '{other}', using seconds");
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
            tracing::warn!("unsupported weight dtype '{other}', using float32");
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

#[allow(clippy::large_enum_variant)]
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
        timestamp: cfg
            .timestamp
            .as_ref()
            .map(|ts| EdgeSortKeyTimestampMetadata {
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
    let mut map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut chars = expr.chars().peekable();
    let mut idx = 0usize;
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut name = String::new();
            let mut escaped = false;
            for c in chars.by_ref() {
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
            access.get_decimal256(idx).ok().flatten().and_then(|v| {
                let num = num_bigint::BigInt::from_signed_bytes_le(&v);
                num.to_f64().map(|f| f / 10_f64.powi(scale))
            })
        }
        _ => None,
    }
}

pub(crate) fn encode_timestamp_value(value: Option<i64>, encoding: &TimestampEncodingSpec) -> u64 {
    let step = encoding.step.max(1);
    let raw = value.unwrap_or(i64::MIN);
    let diff = raw.saturating_sub(encoding.anchor);
    let d = diff.div_euclid(step);
    let r = diff.rem_euclid(step);
    let delta = if r > 0 { d + 1 } else { d };
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

fn eval_weight_expr(
    expr: &mut WeightExprRuntime,
    access: &mut RowAccess<'_>,
) -> Result<Option<f64>> {
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
                let width_bytes = timestamp_width_bits(&ts.encoding.kind) / 8;
                batchers.push(SortKeyComponentBatcher::new(
                    std::sync::Arc::<str>::from(rel.name.clone()),
                    SortKeyComponentKind::Timestamp,
                    width_bytes,
                    SortKeyComponentEncoder::Timestamp {
                        column_idx: idx,
                        encoding: ts.encoding.clone(),
                    },
                    num_shards,
                ));
            }
        }
        if let Some(weight) = &cfg.weight {
            if weight.table == table.name {
                let expr = compile_weight_expr(&weight.expr, table)?;
                let width_bytes = weight_width_bits(&weight.encoding.dtype) / 8;
                batchers.push(SortKeyComponentBatcher::new(
                    std::sync::Arc::<str>::from(rel.name.clone()),
                    SortKeyComponentKind::Weight,
                    width_bytes,
                    SortKeyComponentEncoder::Weight {
                        expr,
                        encoding: weight.encoding.clone(),
                    },
                    num_shards,
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
                let width_bytes = timestamp_width_bits(&ts.encoding.kind) / 8;
                batchers.push(SortKeyComponentBatcher::new(
                    std::sync::Arc::<str>::from(rel.name.clone()),
                    SortKeyComponentKind::Timestamp,
                    width_bytes,
                    SortKeyComponentEncoder::Timestamp {
                        column_idx: idx,
                        encoding: ts.encoding.clone(),
                    },
                    num_shards,
                ));
            }
        }
        if let Some(weight) = &cfg.weight {
            if weight.table == table.name {
                let expr = compile_weight_expr(&weight.expr, table)?;
                let width_bytes = weight_width_bits(&weight.encoding.dtype) / 8;
                batchers.push(SortKeyComponentBatcher::new(
                    std::sync::Arc::<str>::from(rel.name.clone()),
                    SortKeyComponentKind::Weight,
                    width_bytes,
                    SortKeyComponentEncoder::Weight {
                        expr,
                        encoding: weight.encoding.clone(),
                    },
                    num_shards,
                ));
            }
        }
    }
    Ok(batchers)
}

const BASE_ENCODE_MEMORY_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_DATABRICKS_ENCODE_MEMORY_BUDGET_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const MAX_RESIDENT_PARQUET_ENCODE_MEMORY_BUDGET_BYTES: u64 = 16 * 1024 * 1024 * 1024;

fn encode_memory_budget_bytes(
    has_databricks: bool,
    has_resident_parquet: bool,
    total_memory: Option<u64>,
) -> u64 {
    let mut budget = BASE_ENCODE_MEMORY_BUDGET_BYTES;
    if has_databricks {
        budget = budget.max(
            total_memory
                .map(|bytes| bytes / 64)
                .unwrap_or(BASE_ENCODE_MEMORY_BUDGET_BYTES)
                .clamp(
                    BASE_ENCODE_MEMORY_BUDGET_BYTES,
                    MAX_DATABRICKS_ENCODE_MEMORY_BUDGET_BYTES,
                ),
        );
    }
    if has_resident_parquet {
        // Resident Parquet decodes row groups in the shared Encode pool. Give
        // each CPU enough bounded decoder/output memory while CloudTransfer
        // independently caps complete compressed objects at one quarter of RAM.
        budget = budget.max(
            total_memory
                .map(|bytes| bytes / 16)
                .unwrap_or(BASE_ENCODE_MEMORY_BUDGET_BYTES)
                .clamp(
                    BASE_ENCODE_MEMORY_BUDGET_BYTES,
                    MAX_RESIDENT_PARQUET_ENCODE_MEMORY_BUDGET_BYTES,
                ),
        );
    }
    budget
}

#[cfg(test)]
mod encode_memory_budget_tests {
    use super::{
        encode_memory_budget_bytes, BASE_ENCODE_MEMORY_BUDGET_BYTES,
        MAX_DATABRICKS_ENCODE_MEMORY_BUDGET_BYTES, MAX_RESIDENT_PARQUET_ENCODE_MEMORY_BUDGET_BYTES,
    };

    #[test]
    fn non_databricks_budget_does_not_scale() {
        assert_eq!(
            encode_memory_budget_bytes(false, false, Some(1024 * 1024 * 1024 * 1024)),
            BASE_ENCODE_MEMORY_BUDGET_BYTES
        );
    }

    #[test]
    fn resident_parquet_budget_scales_for_parallel_row_group_decode() {
        assert_eq!(
            encode_memory_budget_bytes(false, true, Some(64 * 1024 * 1024 * 1024)),
            4 * 1024 * 1024 * 1024
        );
        assert_eq!(
            encode_memory_budget_bytes(false, true, Some(512 * 1024 * 1024 * 1024)),
            MAX_RESIDENT_PARQUET_ENCODE_MEMORY_BUDGET_BYTES
        );
    }

    #[test]
    fn databricks_budget_scales_between_laptop_floor_and_vm_cap() {
        assert_eq!(
            encode_memory_budget_bytes(true, false, Some(24 * 1024 * 1024 * 1024)),
            BASE_ENCODE_MEMORY_BUDGET_BYTES
        );
        assert_eq!(
            encode_memory_budget_bytes(true, false, Some(512 * 1024 * 1024 * 1024)),
            8 * 1024 * 1024 * 1024
        );
        assert_eq!(
            encode_memory_budget_bytes(true, false, Some(4 * 1024 * 1024 * 1024 * 1024)),
            MAX_DATABRICKS_ENCODE_MEMORY_BUDGET_BYTES
        );
    }
}

impl IngestContext {
    fn make_pool<S: Default + Send + 'static>(&self, threads: Option<usize>) -> TaskPool<S> {
        if !self.parallel {
            return TaskPool::new(1);
        }
        let n = threads.unwrap_or_else(default_worker_threads);
        let n = n.max(1);
        TaskPool::new(n)
    }

    fn emit_progress(&self, event: ProgressEvent) {
        if let Some(cb) = &self.progress {
            cb(event);
        }
    }

    fn merge_key_failures(&mut self, state: KeyFailureState) {
        self.key_parse_failures = self.key_parse_failures.saturating_add(state.parse_failures);
        self.key_lookup_failures = self
            .key_lookup_failures
            .saturating_add(state.lookup_failures);
        self.key_membership_failures = self
            .key_membership_failures
            .saturating_add(state.membership_failures);
        for (context, counts) in state.by_context {
            let entry = self.key_failures_by_context.entry(context).or_default();
            entry.parse_failures = entry.parse_failures.saturating_add(counts.parse_failures);
            entry.lookup_failures = entry.lookup_failures.saturating_add(counts.lookup_failures);
            entry.membership_failures = entry
                .membership_failures
                .saturating_add(counts.membership_failures);
        }
        for sample in state.samples {
            if self.key_failure_samples.len() >= 10 {
                break;
            }
            self.key_failure_samples.push(sample);
        }
    }

    fn binary_label_column_for(&self, table_name: &str) -> Option<&str> {
        self.stage_sidecars
            .get(table_name)
            .and_then(|sidecars| sidecars.binary_label_column.as_deref())
    }

    fn sample_weight_column_for(&self, table_name: &str) -> Option<&str> {
        self.stage_sidecars
            .get(table_name)
            .and_then(|sidecars| sidecars.sample_weight_column.as_deref())
    }

    fn new(
        source: GraphConfig,
        target: std::sync::Arc<OutputTarget>,
        options: IngestOptions,
        output: &str,
        stage_sidecars: HashMap<String, StageSidecarColumns>,
    ) -> Result<Self> {
        let index_mem_budget_bytes = options
            .index_mem_budget_bytes
            .or_else(default_index_mem_budget_bytes);
        let index_mem_budget =
            index_mem_budget_bytes.map(|bytes| std::sync::Arc::new(MemoryBudget::new(bytes)));
        let num_shards = source.num_shards.unwrap_or(1).max(1);
        let mut relation_defs = Vec::new();
        let mut relationships = Vec::new();

        // Build table name -> primary key map for looking up dst_column
        let table_pk_map: HashMap<&str, &str> = source
            .tables
            .iter()
            .filter_map(|t| t.primary_key.as_deref().map(|pk| (t.name.as_str(), pk)))
            .collect();

        // Build table name -> TableSpec map for column lookups
        let table_map: HashMap<&str, &TableSpec> =
            source.tables.iter().map(|t| (t.name.as_str(), t)).collect();

        // Build FK edge targets for all FK columns
        // Key: "table_name.column_name" -> FkEdgeTarget
        let mut fk_edge_targets: HashMap<String, FkEdgeTarget> = HashMap::new();

        // Process embedded fkeys
        for table in &source.tables {
            for fkey in &table.fkeys {
                let fk_col = table.columns.iter().find(|c| c.name == fkey.src_column);
                let ref_table = table_map.get(fkey.dst_table.as_str());
                if let (Some(fk_col), Some(ref_table)) = (fk_col, ref_table) {
                    let pk_name = table_pk_map
                        .get(fkey.dst_table.as_str())
                        .copied()
                        .unwrap_or("id");
                    let pk_col = ref_table.columns.iter().find(|c| c.name == pk_name);
                    if let Some(pk_col) = pk_col {
                        let key = format!("{}.{}", table.name, fkey.src_column);
                        fk_edge_targets.insert(
                            key,
                            FkEdgeTarget {
                                ingested_type: fk_col.ingested_dtype.clone(),
                                target_type: pk_col.ingested_dtype.clone(),
                                scale: pk_col.scale,
                                precision: pk_col.precision,
                            },
                        );
                    }
                }
            }

            // Process edge tables
            if let Some(edge) = &table.edge_table {
                // src_fk column
                let fk_col = table
                    .columns
                    .iter()
                    .find(|c| c.name == edge.src_fk.edge_column);
                let ref_table = table_map.get(edge.src_fk.ref_table.as_str());
                if let (Some(fk_col), Some(ref_table)) = (fk_col, ref_table) {
                    let pk_name = table_pk_map
                        .get(edge.src_fk.ref_table.as_str())
                        .copied()
                        .unwrap_or("id");
                    let pk_col = ref_table.columns.iter().find(|c| c.name == pk_name);
                    if let Some(pk_col) = pk_col {
                        let key = format!("{}.{}", table.name, edge.src_fk.edge_column);
                        fk_edge_targets.insert(
                            key,
                            FkEdgeTarget {
                                ingested_type: fk_col.ingested_dtype.clone(),
                                target_type: pk_col.ingested_dtype.clone(),
                                scale: pk_col.scale,
                                precision: pk_col.precision,
                            },
                        );
                    }
                }

                // dst_fk column
                let fk_col = table
                    .columns
                    .iter()
                    .find(|c| c.name == edge.dst_fk.edge_column);
                let ref_table = table_map.get(edge.dst_fk.ref_table.as_str());
                if let (Some(fk_col), Some(ref_table)) = (fk_col, ref_table) {
                    let pk_name = table_pk_map
                        .get(edge.dst_fk.ref_table.as_str())
                        .copied()
                        .unwrap_or("id");
                    let pk_col = ref_table.columns.iter().find(|c| c.name == pk_name);
                    if let Some(pk_col) = pk_col {
                        let key = format!("{}.{}", table.name, edge.dst_fk.edge_column);
                        fk_edge_targets.insert(
                            key,
                            FkEdgeTarget {
                                ingested_type: fk_col.ingested_dtype.clone(),
                                target_type: pk_col.ingested_dtype.clone(),
                                scale: pk_col.scale,
                                precision: pk_col.precision,
                            },
                        );
                    }
                }
            }
        }

        // Build relationships from fkeys embedded in tables
        for table in &source.tables {
            for fkey in &table.fkeys {
                // Get the destination table's primary key
                let dst_pk = table_pk_map
                    .get(fkey.dst_table.as_str())
                    .copied()
                    .unwrap_or("id");

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

                let rev_name = format!("{base}_reverse");
                // Forward relation: paired with reverse only when bidirectional.
                relation_defs.push(RelationDef {
                    name: base.clone(),
                    src_table: table.name.clone(),
                    dst_table: fkey.dst_table.clone(),
                    sort_key: dst_sort,
                    reverse_of: if fkey.bidirectional {
                        Some(rev_name.clone())
                    } else {
                        None
                    },
                });
                // Reverse relation: skipped entirely when bidirectional=false.
                if fkey.bidirectional {
                    relation_defs.push(RelationDef {
                        name: rev_name.clone(),
                        src_table: fkey.dst_table.clone(),
                        dst_table: table.name.clone(),
                        sort_key: src_sort,
                        reverse_of: Some(base.clone()),
                    });
                }
                relationships.push(RelationshipRuntime {
                    name_forward: base.clone(),
                    name_reverse: rev_name,
                    src_table: table.name.clone(),
                    src_column: fkey.src_column.clone(),
                    dst_table: fkey.dst_table.clone(),
                    bidirectional: fkey.bidirectional,
                });
            }

            // Handle edge tables
            if let Some(edge) = &table.edge_table {
                let base = format!("edge_{}", table.name);
                // Use forward/reverse from EdgeTableSpec directly
                let fwd_name = format!("{base}_src_to_dst");
                let rev_name = format!("{base}_dst_to_src");
                let src_rel = RelationDef {
                    name: fwd_name.clone(),
                    src_table: edge.src_fk.ref_table.clone(),
                    dst_table: edge.dst_fk.ref_table.clone(),
                    sort_key: edge
                        .forward
                        .as_ref()
                        .and_then(build_relation_sort_key_config),
                    reverse_of: Some(rev_name.clone()),
                };
                let dst_rel = RelationDef {
                    name: rev_name,
                    src_table: edge.dst_fk.ref_table.clone(),
                    dst_table: edge.src_fk.ref_table.clone(),
                    sort_key: edge
                        .reverse
                        .as_ref()
                        .and_then(build_relation_sort_key_config),
                    reverse_of: Some(fwd_name),
                };
                relation_defs.push(src_rel);
                relation_defs.push(dst_rel);
            }
        }
        Ok(Self {
            source,
            target,
            profile_dir: options.profile_dir,
            databricks_config_file: options.databricks_config_file,
            databricks_helper: None,
            parallel: options.parallel,
            source_loader_threads: options.source_loader_threads,
            source_encoder_threads: options.source_encoder_threads,
            pk_map_threads: options.pk_map_threads,
            edges_map_threads: options.edges_map_threads,
            index_threads: options.index_threads,
            pk_map_estimate_sum: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            pk_map_peak_rss: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            index_mem_budget,
            num_shards,
            table_rows: HashMap::new(),
            table_contexts: HashMap::new(),
            shard_rows: HashMap::new(),
            shard_offsets: HashMap::new(),
            stage_sidecars,
            binary_label_counts: HashMap::new(),
            sample_weight_counts: HashMap::new(),
            pk_hash_paths: HashMap::new(),
            relation_defs,
            relationships,
            fk_edge_targets,
            edge_counts: HashMap::new(),
            edge_dst_nonzero_counts: HashMap::new(),
            pk_duplicate_counts: HashMap::new(),
            pk_duplicate_drops: HashMap::new(),
            pk_null_timestamp_drops: HashMap::new(),
            edge_null_timestamp_drops: HashMap::new(),
            pk_duplicate_samples: HashMap::new(),
            dropped_rows_counts: HashMap::new(),
            pk_sampled_max_key_len: HashMap::new(),
            key_parse_failures: 0,
            key_lookup_failures: 0,
            key_membership_failures: 0,
            key_failure_samples: Vec::new(),
            key_failures_by_context: HashMap::new(),
            output: output.to_string(),
            trimmed_strings: std::sync::Arc::new(AtomicU64::new(0)),
            progress: options.progress,
            temp_edges_src_bytes: 0,
            temp_fkeys_bytes: 0,
            temp_edges_map_bytes: 0,
            temp_node_data_bytes: 0,
            databricks_wire_bytes: 0,
            databricks_decoded_ipc_bytes: 0,
            databricks_query_startup_nanos: 0,
            logical_arrow_source_bytes: 0,
            logical_arrow_source_bytes_by_table: HashMap::new(),
            source_input_wait_nanos: 0,
            encode_memory_high_water_bytes: 0,
            logical_source_rows: 0,
            logical_source_rows_by_table: HashMap::new(),
            peak_mem_stats: options.peak_mem_stats,
            col_stats: options.col_stats,
            cat_count_strategy: options.cat_count_strategy,
            row_subsample: options.row_subsample,
            text_sample_stats: HashMap::new(),
            parent_index_path: options.parent_index_path,
            resolve_columns: options.resolve_columns,
        })
    }

    fn set_peak_stage(&self, idx: Option<usize>) {
        if let Some(stats) = &self.peak_mem_stats {
            stats.set_stage(idx);
        }
    }

    /// Resolve every physical schema before the Encode marker/timer begins.
    /// Databricks uses a LIMIT 0 metadata query on the same reusable helper
    /// connection, so preparation does not add a full source scan.
    fn prepare_input_schemas(&mut self) -> Result<()> {
        let mut helper = start_databricks_helper(&self.source)?;
        visit_source_schemas(
            &self.source,
            self.databricks_config_file.as_deref(),
            &mut helper,
            |table, schema| {
                let synthetic_columns = self
                    .resolve_columns
                    .iter()
                    .filter(|resolution| resolution.table == table.name)
                    .map(|resolution| resolution.node_id_col.clone())
                    .collect::<HashSet<_>>();
                validate_source_schema(table, schema, &synthetic_columns)
            },
        )?;
        self.databricks_helper = helper;
        Ok(())
    }

    fn build_temp_shards(&mut self) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        let temp_root = root.join(TEMP_DIR_NAME);
        fs::create_dir_all(&temp_root)?;
        let mut table_rows: HashMap<String, u64> = HashMap::new();
        const ENCODE_MAX_OVERSIZED_CHUNK_BYTES: u64 = 2 * 1024 * 1024 * 1024;
        let mut tables = self.source.tables.clone();
        let encode_memory_budget_bytes = encode_memory_budget_bytes(
            tables
                .iter()
                .any(|table| matches!(table.source, graph_config::DataSource::Databricks { .. })),
            tables.iter().any(|table| {
                matches!(
                    table.source,
                    graph_config::DataSource::Parquet { .. }
                        | graph_config::DataSource::ObjectStoreParquet { .. }
                )
            }),
            total_memory_bytes(),
        );
        let source_rowpack = self.source.rowpack.clone();
        let relationships = Arc::new(self.relationships.clone());
        let relation_defs = std::sync::Arc::new(self.relation_defs.clone());
        let fk_edge_targets = std::sync::Arc::new(self.fk_edge_targets.clone());
        let num_shards = self.num_shards;
        let category_memory_bytes = if self.col_stats
            && self.cat_count_strategy != crate::col_stats::CategoryCountStrategy::None
        {
            diskgraph_core::accumulators::CATEGORY_MEMORY_BUDGET_BYTES
        } else {
            0
        };
        let source_working_budget = encode_memory_budget_bytes
            .checked_sub(category_memory_bytes)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| {
                anyhow!(
                    "Encode category state requires {} bytes, exhausting the {} byte working budget",
                    category_memory_bytes,
                    encode_memory_budget_bytes
                )
            })?;
        let category_memory_budget = (category_memory_bytes > 0).then(|| {
            Arc::new(crate::col_stats::CategoryStringBudget::new(
                category_memory_bytes,
            ))
        });
        let source_budget = crate::source_memory::ByteBudget::with_oversized_limit(
            source_working_budget,
            ENCODE_MAX_OVERSIZED_CHUNK_BYTES,
        )?;
        let target = std::sync::Arc::clone(&self.target);
        let source_encoder_workers = if self.parallel {
            self.source_encoder_threads
                .unwrap_or_else(edge_writer_workers)
                .max(1)
        } else {
            1
        };
        let node_pool = NodeWritePool::new(std::sync::Arc::clone(&target), edge_writer_workers());
        let edge_pool =
            EdgeSrcWritePool::new(std::sync::Arc::clone(&target), edge_writer_workers());
        let dest_key_pool =
            DestKeyWritePool::new(std::sync::Arc::clone(&target), edge_writer_workers());
        let node_senders = node_pool.senders();
        let node_errors = node_pool.error();
        let edge_senders = edge_pool.senders();
        let edge_errors = edge_pool.error();
        let dest_key_senders = dest_key_pool.senders();
        let dest_key_errors = dest_key_pool.error();
        let trimmed_strings = std::sync::Arc::clone(&self.trimmed_strings);
        let col_stats_enabled = self.col_stats;
        let cat_count_strategy = self.cat_count_strategy;
        let row_subsample = self.row_subsample;

        let mut table_cat_counts: HashMap<
            String,
            std::sync::Arc<crate::col_stats::TableCategoryCounts>,
        > = HashMap::new();
        let mut table_col_stats: HashMap<String, crate::col_stats::FileColStats> = HashMap::new();
        // FK-free LP resolution mode: load the parent pkmap(s) ONCE here, before
        // the parallel encode pass, and share read-only via Arc so each encode
        // worker can resolve PK reference columns to graph node-ids inline (no
        // FK edge). `PkHashShard::lookup` is `&self`, so concurrent reads are
        // safe. Empty `resolve_columns` (the default) → `None`, zero overhead.
        let resolve_columns = std::sync::Arc::new(self.resolve_columns.clone());
        let ref_resolver: Option<std::sync::Arc<crate::mapping::RefResolver>> =
            if resolve_columns.is_empty() {
                None
            } else {
                let parent_root = self.parent_index_path.as_ref().ok_or_else(|| {
                    anyhow!(
                        "resolution mode requires IngestOptions.parent_index_path \
                         (the parent index whose pkmap resolves the reference columns)"
                    )
                })?;
                let dst_tables: Vec<String> = resolve_columns
                    .iter()
                    .map(|r| r.dst_table.clone())
                    .collect();
                Some(std::sync::Arc::new(crate::mapping::RefResolver::load(
                    parent_root,
                    &dst_tables,
                )?))
            };

        // Every transport enters the same batch dispatcher and parallel
        // node/edge consumers. Context sampling remains single-threaded;
        // replay and all later batches may be encoded by any worker.
        {
            let mut helper = self.databricks_helper.take();
            for table in &mut tables {
                let table_encode_start = std::time::Instant::now();
                tracing::info!(phase = "encode", table = %table.name, "table started");
                let synthetic_columns = resolve_columns
                    .iter()
                    .filter(|resolution| resolution.table == table.name)
                    .map(|resolution| resolution.node_id_col.clone())
                    .collect::<HashSet<_>>();
                if let Some(callback) = &self.progress {
                    callback(ProgressEvent {
                        phase: ProgressPhase::Encode,
                        table: table.name.clone(),
                        shard: None,
                        kind: ProgressKind::Start,
                        delta: 0,
                        total: Some(1),
                        status: None,
                    });
                }
                let sidecars = self
                    .stage_sidecars
                    .get(&table.name)
                    .cloned()
                    .unwrap_or_default();
                let validation = SourceValidationState::new(table, synthetic_columns.clone());
                let input_transfer = build_input_transfer_for_table(table)?;
                let stream_inner = match &table.source {
                    graph_config::DataSource::Parquet { .. }
                    | graph_config::DataSource::ObjectStoreParquet { .. } => {
                        let loader_threads = if self.parallel {
                            self.source_loader_threads.unwrap_or_else(|| {
                                effective_parallelism().saturating_mul(2).clamp(2, 64)
                            })
                        } else {
                            1
                        };
                        let resident_budget = input_transfer
                            .as_ref()
                            .map(|transfer| {
                                crate::source_memory::ByteBudget::new(
                                    transfer.stats_handle().snapshot().budget_max_bytes,
                                )
                            })
                            .transpose()?;
                        TableSourceStreamInner::Parquet(ResidentParquetWorkStream::open(
                            table,
                            input_transfer.as_ref(),
                            resident_budget,
                            loader_threads,
                        )?)
                    }
                    graph_config::DataSource::Csv { .. } => {
                        TableSourceStreamInner::Csv(Box::new(open_csv_batch_stream(
                            table,
                            input_transfer.as_ref(),
                            Arc::clone(&source_budget),
                        )?))
                    }
                    graph_config::DataSource::Databricks { .. } => {
                        let mut query_options = crate::databricks::QueryStreamOptions::default();
                        if let Some(loaders) = self.source_loader_threads {
                            query_options.max_concurrent_downloads = loaders.max(1);
                        }
                        if std::env::var_os("DISKGRAPH_DATABRICKS_BENCHMARK_ALLOW_HTTP_LOOPBACK")
                            .is_some_and(|value| value == "1")
                        {
                            query_options.allow_loopback_http_for_benchmark();
                        }
                        if std::env::var_os("DISKGRAPH_DATABRICKS_BENCHMARK_ISOLATE_CONSUMER")
                            .is_some_and(|value| value == "1")
                        {
                            query_options.max_concurrent_downloads = 1;
                            query_options.pause_download_while_batch_active();
                        }
                        TableSourceStreamInner::Databricks(Box::new(
                            crate::databricks::DatabricksQueryStream::open_with_shared_budget(
                                helper.as_mut().ok_or_else(|| {
                                    anyhow!("Databricks helper was not initialized")
                                })?,
                                &table.source,
                                self.databricks_config_file.as_deref(),
                                query_options,
                                Arc::clone(&source_budget),
                            )?,
                        ))
                    }
                    _ => return Err(anyhow!("table {} source is not implemented", table.name)),
                };
                let mut stream = TableSourceStream::new(
                    stream_inner,
                    Arc::clone(&source_budget),
                    Arc::clone(&validation),
                );
                if table.edge_table.is_some() {
                    let encode_table: &TableSpec = table;
                    let results = run_encode_batch_workers(
                        &mut stream,
                        source_encoder_workers,
                        Arc::clone(&source_budget),
                        Arc::clone(&validation),
                        |batches| {
                            let mut key_failures = KeyFailureState::default();
                            let dropped = Self::build_temp_edge_batches(
                                encode_table,
                                batches,
                                num_shards,
                                &trimmed_strings,
                                &mut key_failures,
                                &edge_senders,
                                &edge_errors,
                                &dest_key_senders,
                                &dest_key_errors,
                                &relation_defs,
                                &fk_edge_targets,
                            )?;
                            Ok((dropped, key_failures))
                        },
                    )?;
                    let mut dropped = 0u64;
                    for (worker_dropped, key_failures) in results {
                        dropped = dropped.saturating_add(worker_dropped);
                        self.merge_key_failures(key_failures);
                    }
                    if dropped > 0 {
                        self.dropped_rows_counts.insert(table.name.clone(), dropped);
                    }
                } else {
                    let rowpack_config = rowpack_config_for_table(source_rowpack.as_ref(), table);
                    let sample_limit = rowpack_config.sample_rows().min(SAMPLE_ROW_LIMIT);
                    let mut context_builder =
                        RowPackContextBuilder::new(rowpack_columns(table)?, rowpack_config);
                    let resolve_node_ids: HashSet<usize> = resolve_columns
                        .iter()
                        .filter(|resolution| resolution.table == table.name)
                        .filter_map(|resolution| column_index(table, &resolution.node_id_col).ok())
                        .collect();
                    let pk_col_index = table
                        .primary_key
                        .as_ref()
                        .and_then(|pk| column_index(table, pk).ok());
                    let mut pk_max_key_len = 0usize;
                    let row_access_hints = build_row_access_hints(table);
                    let stats = RowAccessStats::new(Some(trimmed_strings.as_ref()));
                    let mut access = RowAccess::new(row_access_hints, stats);
                    let mut prefix = Vec::new();
                    let mut prefix_rows = 0usize;
                    let mut prefix_bytes = 0usize;
                    let mut remainder = None;

                    // Parquet builds the context from its own strided pass and
                    // leaves the encode stream untouched. Other sources consume
                    // their input destructively, so they still learn from the
                    // prefix they are about to encode.
                    let strided = sample_context_batches(
                        table,
                        input_transfer.as_ref(),
                        &source_budget,
                        sample_limit,
                    )?;
                    for batch in &strided {
                        if let Some(col) = pk_col_index {
                            pk_max_key_len =
                                pk_max_key_len.max(max_binary_value_len(batch.column(col)));
                        }
                        let cols = build_access_cols(batch, table, &resolve_node_ids)?;
                        access.set_cols(&cols);
                        for row in 0..batch.num_rows() {
                            access.set_row(row);
                            context_builder.observe_row_from_access(&mut access)?;
                        }
                        access.set_cols(&[]);
                    }

                    while strided.is_empty()
                        && prefix_rows < sample_limit
                        && prefix_bytes < SOURCE_PREFIX_MAX_BYTES
                    {
                        let batch = match stream.next_context_batch() {
                            Some(batch) => batch?,
                            None => break,
                        };
                        if batch.batch.num_rows() == 0 {
                            continue;
                        }
                        let remaining_prefix_bytes = SOURCE_PREFIX_MAX_BYTES - prefix_bytes;
                        let mut prefix_permit =
                            source_budget.acquire(remaining_prefix_bytes as u64)?;
                        let compact = compact_context_prefix_with_budget(
                            &table.name,
                            &batch.batch,
                            sample_limit - prefix_rows,
                            remaining_prefix_bytes,
                            prefix_rows,
                            prefix_bytes,
                        )?;
                        let Some(compact) = compact else {
                            remainder = stream.defer_context_remainder(batch);
                            break;
                        };
                        let sampled_rows = compact.num_rows();
                        prefix_permit.shrink_to(compact.get_array_memory_size() as u64);
                        if let Some(col) = pk_col_index {
                            pk_max_key_len =
                                pk_max_key_len.max(max_binary_value_len(compact.column(col)));
                        }
                        let cols = build_access_cols(&compact, table, &resolve_node_ids)?;
                        access.set_cols(&cols);
                        for row in 0..sampled_rows {
                            access.set_row(row);
                            context_builder.observe_row_from_access(&mut access)?;
                        }
                        access.set_cols(&[]);
                        prefix_rows += sampled_rows;
                        prefix_bytes = prefix_bytes.saturating_add(compact.get_array_memory_size());
                        prefix.push(SourceBatch::budgeted(compact, prefix_permit).observed());
                        if sampled_rows < batch.batch.num_rows() {
                            let remaining_rows =
                                batch.batch.num_rows().saturating_sub(sampled_rows);
                            remainder = stream
                                .defer_context_remainder(batch.slice(sampled_rows, remaining_rows));
                            break;
                        }
                    }

                    let unique_counts = context_builder.sample_unique_counts();
                    let fk_columns: HashSet<&str> = table
                        .fkeys
                        .iter()
                        .map(|foreign_key| foreign_key.src_column.as_str())
                        .collect();
                    let semantic_types: Vec<SemanticType> = table
                        .columns
                        .iter()
                        .zip(unique_counts.iter())
                        .map(|(column, &(num_unique, count))| {
                            if let Some(semantic_type) = column.semantic_type {
                                return semantic_type;
                            }
                            let role = graph_config::ColumnRole {
                                is_pk: table
                                    .primary_key
                                    .as_deref()
                                    .is_some_and(|primary_key| primary_key == column.name),
                                is_fk: fk_columns.contains(column.name.as_str()),
                                name: &column.name,
                            };
                            let sample = (num_unique > 0 || count > 0)
                                .then_some(graph_config::ColumnSampleStats { num_unique, count });
                            graph_config::infer_semantic_type(column, &role, sample.as_ref())
                        })
                        .collect();
                    if col_stats_enabled {
                        // Whichever pass produced the sample feeds the text
                        // stats; with striding `prefix` is empty by design.
                        let text_sample: Vec<RecordBatch> = if strided.is_empty() {
                            prefix.iter().map(|batch| batch.batch.clone()).collect()
                        } else {
                            strided.clone()
                        };
                        let text_stats =
                            text_stats_from_prefix(table, &semantic_types, &text_sample)?;
                        if !text_stats.is_empty() {
                            self.text_sample_stats
                                .insert(table.name.clone(), text_stats);
                        }
                    }
                    for (column, semantic_type) in
                        table.columns.iter_mut().zip(semantic_types.iter().copied())
                    {
                        column.semantic_type = Some(semantic_type);
                    }
                    if let Some(source_table) = self
                        .source
                        .tables
                        .iter_mut()
                        .find(|source_table| source_table.name == table.name)
                    {
                        source_table.columns.clone_from(&table.columns);
                    }

                    if pk_col_index.is_some() {
                        self.pk_sampled_max_key_len
                            .insert(table.name.clone(), pk_max_key_len);
                    }
                    let node_context = context_builder.finish()?;
                    let context_bytes = node_context.to_bytes()?;
                    self.target
                        .write_bytes(&temp_node_context_path(&table.name), context_bytes.clone())?;
                    self.target
                        .write_bytes(&node_context_output_path(&table.name), context_bytes)?;
                    self.table_contexts.insert(
                        table.name.clone(),
                        format!("nodes/{}/node_data.ctx", table.name.to_lowercase()),
                    );
                    let node_encoder = RowPackEncoder::new(node_context)?;

                    if cat_count_strategy != crate::col_stats::CategoryCountStrategy::None
                        && col_stats_enabled
                    {
                        let stypes = table
                            .columns
                            .iter()
                            .map(|column| column.semantic_type)
                            .collect::<Vec<_>>();
                        table_cat_counts.insert(
                            table.name.clone(),
                            Arc::new(
                                if let Some(category_memory_budget) = &category_memory_budget {
                                    crate::col_stats::TableCategoryCounts::new_with_string_budget(
                                        cat_count_strategy,
                                        &stypes,
                                        row_subsample,
                                        Arc::clone(category_memory_budget),
                                    )
                                } else {
                                    crate::col_stats::TableCategoryCounts::new(
                                        cat_count_strategy,
                                        &stypes,
                                        row_subsample,
                                    )
                                },
                            ),
                        );
                    }
                    let replay = prefix
                        .into_iter()
                        .map(|batch| Ok(SourceWork::Batch(batch)))
                        .chain(
                            remainder
                                .into_iter()
                                .map(|batch| Ok(SourceWork::Batch(batch))),
                        )
                        .chain(&mut stream);
                    let table_cat = table_cat_counts.get(&table.name).cloned();
                    let encode_table: &TableSpec = table;
                    let results = run_encode_batch_workers(
                        replay,
                        source_encoder_workers,
                        Arc::clone(&source_budget),
                        Arc::clone(&validation),
                        |batches| {
                            let mut key_failures = KeyFailureState::default();
                            if col_stats_enabled {
                                let mut observer = if let Some(categories) = &table_cat {
                                    crate::col_stats::FileColStats::new_with_categories(
                                        &encode_table.columns,
                                        Arc::clone(categories),
                                    )
                                } else {
                                    crate::col_stats::FileColStats::new(&encode_table.columns)
                                };
                                let node = Self::build_temp_node_batches(
                                    encode_table,
                                    batches,
                                    &relationships,
                                    &relation_defs,
                                    num_shards,
                                    &node_encoder,
                                    &trimmed_strings,
                                    &mut key_failures,
                                    &node_senders,
                                    &node_errors,
                                    &dest_key_senders,
                                    &dest_key_errors,
                                    &fk_edge_targets,
                                    &resolve_columns,
                                    ref_resolver.as_deref(),
                                    &sidecars,
                                    &mut observer,
                                )?;
                                observer.flush_categories();
                                Ok((node, key_failures, Some(observer)))
                            } else {
                                let node = Self::build_temp_node_batches(
                                    encode_table,
                                    batches,
                                    &relationships,
                                    &relation_defs,
                                    num_shards,
                                    &node_encoder,
                                    &trimmed_strings,
                                    &mut key_failures,
                                    &node_senders,
                                    &node_errors,
                                    &dest_key_senders,
                                    &dest_key_errors,
                                    &fk_edge_targets,
                                    &resolve_columns,
                                    ref_resolver.as_deref(),
                                    &sidecars,
                                    &mut diskgraph_rowpack::NoOpObserver,
                                )?;
                                Ok((node, key_failures, None))
                            }
                        },
                    )?;
                    let mut total_rows = 0u64;
                    let mut node_data_dropped = 0u64;
                    let mut merged_stats: Option<crate::col_stats::FileColStats> = None;
                    for (node, key_failures, file_stats) in results {
                        total_rows = total_rows.saturating_add(node.total_rows);
                        node_data_dropped =
                            node_data_dropped.saturating_add(node.node_data_dropped);
                        self.merge_key_failures(key_failures);
                        if let Some(mut file_stats) = file_stats {
                            if let Some(stats) = &mut merged_stats {
                                stats.merge(&mut file_stats);
                            } else {
                                merged_stats = Some(file_stats);
                            }
                        }
                    }
                    table_rows.insert(table.name.clone(), total_rows);
                    if node_data_dropped > 0 {
                        self.dropped_rows_counts
                            .insert(table.name.clone(), node_data_dropped);
                    }
                    if let Some(stats) = merged_stats {
                        table_col_stats.insert(table.name.clone(), stats);
                    }
                }
                validation.finish()?;
                let source_rows = validation.rows_seen();
                if source_rows == 0 && self.parent_index_path.is_none() {
                    return Err(anyhow!(
                        "graph source returned zero rows for table {}",
                        table.name
                    ));
                }
                let source_bytes = validation.logical_arrow_bytes();
                self.logical_source_rows = self.logical_source_rows.saturating_add(source_rows);
                self.logical_source_rows_by_table
                    .insert(table.name.clone(), source_rows);
                self.source_input_wait_nanos = self
                    .source_input_wait_nanos
                    .saturating_add(stream.input_wait_nanos());
                self.logical_arrow_source_bytes =
                    self.logical_arrow_source_bytes.saturating_add(source_bytes);
                self.logical_arrow_source_bytes_by_table
                    .insert(table.name.clone(), source_bytes);
                tracing::info!(
                    phase = "encode",
                    table = %table.name,
                    rows = source_rows,
                    elapsed_ms = table_encode_start.elapsed().as_millis(),
                    "table producers complete"
                );
                if let Some(metrics) = stream.databricks_metrics() {
                    self.databricks_wire_bytes = self
                        .databricks_wire_bytes
                        .saturating_add(metrics.wire_bytes);
                    self.databricks_decoded_ipc_bytes = self
                        .databricks_decoded_ipc_bytes
                        .saturating_add(metrics.decoded_ipc_bytes);
                    self.databricks_query_startup_nanos = self
                        .databricks_query_startup_nanos
                        .saturating_add(metrics.query_startup_nanos);
                }
                if let Some(callback) = &self.progress {
                    callback(ProgressEvent {
                        phase: ProgressPhase::Encode,
                        table: table.name.clone(),
                        shard: None,
                        kind: ProgressKind::Advance,
                        delta: 1,
                        total: Some(1),
                        status: None,
                    });
                }
            }
        }
        if let Some(cb) = &self.progress {
            for table in &tables {
                cb(ProgressEvent {
                    phase: ProgressPhase::Encode,
                    table: table.name.clone(),
                    shard: None,
                    kind: ProgressKind::Finish,
                    delta: 0,
                    total: Some(1),
                    status: None,
                });
            }
        }
        // Write col_stats.json if enabled
        if col_stats_enabled && !table_col_stats.is_empty() {
            // Check for timestamp parse failures before writing stats
            for table in &tables {
                if let Some(stats) = table_col_stats.get(&table.name) {
                    let col_names: Vec<String> =
                        table.columns.iter().map(|c| c.name.clone()).collect();
                    stats.warn_parse_failures(&col_names, &table.name);
                }
            }
            let mut root_map = serde_json::Map::new();
            // All col-stats writes go through `self.target.write_bytes` so the
            // dataplane mirror picks them up (each call records a manifest
            // entry and submits the upload). The rel paths are forward-slash
            // composed under ``kumo_col_stats/col_stats_{full,min}/`` to
            // match the layout consumers expect.
            let kumo_stats_full_rel = format!(
                "{}/{}",
                col_stats_export::COL_STATS_DIR,
                col_stats_export::COL_STATS_FULL,
            );
            let kumo_stats_min_rel = format!(
                "{}/{}",
                col_stats_export::COL_STATS_DIR,
                col_stats_export::COL_STATS_MIN,
            );
            for table in &tables {
                if let Some(mut stats) = table_col_stats.remove(&table.name) {
                    let col_names: Vec<String> =
                        table.columns.iter().map(|c| c.name.clone()).collect();
                    let ingested_types: Vec<_> = table
                        .columns
                        .iter()
                        .map(|c| c.ingested_dtype.clone())
                        .collect();
                    let tc = table_cat_counts.get(&table.name).map(|a| a.as_ref());
                    let tss = self.text_sample_stats.get(&table.name);
                    root_map.insert(
                        table.name.clone(),
                        stats.to_json(&col_names, &ingested_types, tc, tss),
                    );
                    // Write per-table kumo-format stats (replaces Python bridge)
                    if let Some(dv) = &table.data_version {
                        let kumo_map = stats.to_kumo_json(&col_names, &ingested_types, tc, tss, dv);
                        let kumo_json = serde_json::to_vec(&kumo_map)?;
                        self.target.write_bytes(
                            &format!("{}/{}.json", kumo_stats_full_rel, table.name),
                            kumo_json,
                        )?;
                        let min_map = crate::col_stats::minify_kumo_stats(&kumo_map, 20);
                        let min_json = serde_json::to_vec(&min_map)?;
                        self.target.write_bytes(
                            &format!("{}/{}.json", kumo_stats_min_rel, table.name),
                            min_json,
                        )?;
                    }
                }
            }
            let json_bytes = serde_json::to_vec_pretty(&serde_json::Value::Object(root_map))?;
            self.target.write_bytes("col_stats.json", json_bytes)?;
            eprintln!(
                "wrote col_stats.json (and kumo_col_stats/) under {}",
                root.display()
            );
        }
        drop(node_senders);
        drop(edge_senders);
        drop(dest_key_senders);
        tracing::info!(phase = "encode", "output writer drain started");
        let node_stats = node_pool.finish()?;
        let edge_stats = edge_pool.finish()?;
        let dest_key_stats = dest_key_pool.finish()?;
        tracing::info!(phase = "encode", "output writer drain complete");
        self.write_consolidated_meta(&node_stats, &edge_stats, &dest_key_stats)?;
        self.refresh_shard_rows_from_stats(&node_stats);
        self.table_rows = table_rows;
        self.encode_memory_high_water_bytes = source_budget.high_water_bytes().saturating_add(
            category_memory_budget
                .as_ref()
                .map_or(0, |budget| budget.used_bytes()),
        );
        self.persist_encode_state()?;
        Ok(())
    }

    fn persist_encode_state(&self) -> Result<()> {
        let state = EncodeState {
            version: ENCODE_STATE_VERSION,
            source: self.source.clone(),
            pk_sampled_max_key_len: self.pk_sampled_max_key_len.clone(),
            binary_label_counts: self
                .binary_label_counts
                .iter()
                .map(|((table, shard), counts)| EncodeLabelCounts {
                    table: table.clone(),
                    shard: *shard,
                    counts: counts.clone(),
                })
                .collect(),
            sample_weight_counts: self
                .sample_weight_counts
                .iter()
                .map(|((table, shard), counts)| EncodeWeightCounts {
                    table: table.clone(),
                    shard: *shard,
                    counts: counts.clone(),
                })
                .collect(),
            dropped_rows_counts: self.dropped_rows_counts.clone(),
            key_parse_failures: self.key_parse_failures,
            key_lookup_failures: self.key_lookup_failures,
            key_membership_failures: self.key_membership_failures,
            key_failure_samples: self.key_failure_samples.clone(),
            key_failures_by_context: self.key_failures_by_context.clone(),
            trimmed_strings: self.trimmed_strings.load(Ordering::Relaxed),
            temp_edges_src_bytes: self.temp_edges_src_bytes,
            temp_fkeys_bytes: self.temp_fkeys_bytes,
            temp_node_data_bytes: self.temp_node_data_bytes,
            databricks_wire_bytes: self.databricks_wire_bytes,
            databricks_decoded_ipc_bytes: self.databricks_decoded_ipc_bytes,
            databricks_query_startup_nanos: self.databricks_query_startup_nanos,
            logical_arrow_source_bytes: self.logical_arrow_source_bytes,
            logical_arrow_source_bytes_by_table: self.logical_arrow_source_bytes_by_table.clone(),
            source_input_wait_nanos: self.source_input_wait_nanos,
            encode_memory_high_water_bytes: self.encode_memory_high_water_bytes,
            logical_source_rows: self.logical_source_rows,
            logical_source_rows_by_table: self.logical_source_rows_by_table.clone(),
        };
        self.target
            .write_bytes(ENCODE_STATE_PATH, serde_json::to_vec_pretty(&state)?)
    }

    fn restore_encode_state(&mut self) -> Result<()> {
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root,
            _ => return Ok(()),
        };
        let path = root.join(ENCODE_STATE_PATH);
        let bytes = fs::read(&path)
            .with_context(|| format!("read completed Encode state: {}", path.display()))?;
        let state: EncodeState = serde_json::from_slice(&bytes)
            .with_context(|| format!("decode completed Encode state: {}", path.display()))?;
        if state.version != ENCODE_STATE_VERSION {
            return Err(anyhow!(
                "unsupported Encode state version {} at {}",
                state.version,
                path.display()
            ));
        }
        self.source = state.source;
        self.binary_label_counts = state
            .binary_label_counts
            .into_iter()
            .map(|entry| ((entry.table, entry.shard), entry.counts))
            .collect();
        self.sample_weight_counts = state
            .sample_weight_counts
            .into_iter()
            .map(|entry| ((entry.table, entry.shard), entry.counts))
            .collect();
        self.dropped_rows_counts = state.dropped_rows_counts;
        self.pk_sampled_max_key_len = state.pk_sampled_max_key_len;
        self.key_parse_failures = state.key_parse_failures;
        self.key_lookup_failures = state.key_lookup_failures;
        self.key_membership_failures = state.key_membership_failures;
        self.key_failure_samples = state.key_failure_samples;
        self.key_failures_by_context = state.key_failures_by_context;
        self.trimmed_strings
            .store(state.trimmed_strings, Ordering::Relaxed);
        self.temp_edges_src_bytes = state.temp_edges_src_bytes;
        self.temp_fkeys_bytes = state.temp_fkeys_bytes;
        self.temp_node_data_bytes = state.temp_node_data_bytes;
        self.databricks_wire_bytes = state.databricks_wire_bytes;
        self.databricks_decoded_ipc_bytes = state.databricks_decoded_ipc_bytes;
        self.databricks_query_startup_nanos = state.databricks_query_startup_nanos;
        self.logical_arrow_source_bytes = state.logical_arrow_source_bytes;
        self.logical_arrow_source_bytes_by_table = state.logical_arrow_source_bytes_by_table;
        self.source_input_wait_nanos = state.source_input_wait_nanos;
        self.encode_memory_high_water_bytes = state.encode_memory_high_water_bytes;
        self.logical_source_rows = state.logical_source_rows;
        self.logical_source_rows_by_table = state.logical_source_rows_by_table;
        Ok(())
    }

    fn persist_resume_stats_state(&self, relative_path: &str) -> Result<()> {
        let root = require_local_output(&self.target)?;
        let state = ResumeStatsState {
            version: RESUME_STATS_STATE_VERSION,
            edge_counts: self.edge_counts.clone(),
            edge_dst_nonzero_counts: self.edge_dst_nonzero_counts.clone(),
            pk_duplicate_counts: self.pk_duplicate_counts.clone(),
            pk_duplicate_drops: self.pk_duplicate_drops.clone(),
            pk_null_timestamp_drops: self.pk_null_timestamp_drops.clone(),
            edge_null_timestamp_drops: self.edge_null_timestamp_drops.clone(),
            pk_duplicate_samples: self.pk_duplicate_samples.clone(),
            key_parse_failures: self.key_parse_failures,
            key_lookup_failures: self.key_lookup_failures,
            key_membership_failures: self.key_membership_failures,
            key_failure_samples: self.key_failure_samples.clone(),
            key_failures_by_context: self.key_failures_by_context.clone(),
        };
        write_durable_file(
            &root.join(relative_path),
            &serde_json::to_vec_pretty(&state)?,
            "resume statistics state",
        )
    }

    fn restore_resume_stats_state(&mut self, relative_path: &str) -> Result<()> {
        let root = require_local_output(&self.target)?;
        let path = root.join(relative_path);
        let bytes = fs::read(&path)
            .with_context(|| format!("read completed phase state: {}", path.display()))?;
        let state: ResumeStatsState = serde_json::from_slice(&bytes)
            .with_context(|| format!("decode completed phase state: {}", path.display()))?;
        if state.version != RESUME_STATS_STATE_VERSION {
            return Err(anyhow!(
                "unsupported resume statistics state version {} at {}",
                state.version,
                path.display()
            ));
        }
        self.edge_counts = state.edge_counts;
        self.edge_dst_nonzero_counts = state.edge_dst_nonzero_counts;
        self.pk_duplicate_counts = state.pk_duplicate_counts;
        self.pk_duplicate_drops = state.pk_duplicate_drops;
        self.pk_null_timestamp_drops = state.pk_null_timestamp_drops;
        self.edge_null_timestamp_drops = state.edge_null_timestamp_drops;
        self.pk_duplicate_samples = state.pk_duplicate_samples;
        self.key_parse_failures = state.key_parse_failures;
        self.key_lookup_failures = state.key_lookup_failures;
        self.key_membership_failures = state.key_membership_failures;
        self.key_failure_samples = state.key_failure_samples;
        self.key_failures_by_context = state.key_failures_by_context;
        Ok(())
    }

    /// Re-derive `shard_rows` and `table_rows` from encode's on-disk
    /// `_tmp/nodes/<t>/shard-N/node_data.meta` files. Used when an ingest
    /// invocation skips the encode phase (resume_from_phase > encode) — encode
    /// would normally have populated these via `refresh_shard_rows_from_stats`,
    /// but the in-memory state is fresh on a resumed invocation. Both
    /// `edge_build_context` and `build_index_metadata` read `shard_rows`, so
    /// later phases would otherwise see empty maps and fail with
    /// "missing pk hash paths for X" or wrong shard offsets.
    fn repopulate_node_stats_from_meta(&mut self) -> Result<()> {
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root.clone(),
            _ => return Ok(()),
        };
        self.shard_rows.clear();
        self.table_rows.clear();
        let mut recovered_committed_shards = 0usize;
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            let mut shard_rows = vec![0u64; self.num_shards];
            let mut total = 0u64;
            for (shard, slot) in shard_rows.iter_mut().enumerate() {
                let meta_path = root.join(temp_node_data_meta_path(&table.name, shard));
                let row_count = if meta_path.exists() {
                    read_temp_meta(&meta_path)?.row_count
                } else {
                    // A committed Index shard is allowed to reclaim its
                    // node_data.meta. Recover the durable count from the
                    // validated commit marker instead of treating the shard
                    // as empty on resume.
                    let shard_name = format!("shard-{shard:05}");
                    let output_dir = root
                        .join("nodes")
                        .join(table.name.to_lowercase())
                        .join(&shard_name);
                    let commit_path = index_shard_commit_path(
                        &root,
                        &table.name.to_lowercase(),
                        &shard_name,
                    );
                    let shard_key = format!("{}/{shard_name}", table.name);
                    match committed_shard_expected_nodes(
                        &output_dir,
                        &commit_path,
                        Some(&shard_key),
                    )? {
                        Some(count) => {
                            recovered_committed_shards += 1;
                            count
                        }
                        None => 0,
                    }
                };
                *slot = row_count;
                total = total.saturating_add(row_count);
            }
            self.shard_rows.insert(table.name.clone(), shard_rows);
            self.table_rows.insert(table.name.clone(), total);
        }
        tracing::info!(
            tables = self.shard_rows.len(),
            recovered_committed_shards,
            "repopulated shard_rows + table_rows from encode meta files"
        );
        Ok(())
    }

    /// Re-derive `pk_hash_paths` from pk_map's deterministic on-disk layout.
    /// Used on resume when pk_map was skipped; edges_map reads `pk_hash_paths`
    /// to build the PkHashStore and would otherwise see an empty map.
    fn repopulate_pk_hash_paths_from_disk(&mut self) -> Result<()> {
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root.clone(),
            _ => return Ok(()),
        };
        self.pk_hash_paths.clear();
        for table in &self.source.tables {
            if table.edge_table.is_some() || table.primary_key.is_none() {
                continue;
            }
            let mut paths = Vec::with_capacity(self.num_shards);
            for shard in 0..self.num_shards {
                let hash_path = format!(
                    "nodes/{}/pk_hash-{:05}.bin",
                    table.name.to_lowercase(),
                    shard
                );
                let data_path = format!(
                    "nodes/{}/pk_data-{:05}.bin",
                    table.name.to_lowercase(),
                    shard
                );
                let abs_hash = root.join(&hash_path);
                let abs_data = root.join(&data_path);
                if !abs_hash.exists() {
                    return Err(anyhow!(
                        "repopulate_pk_hash_paths_from_disk: missing pk hash file {} \
                         for table {} shard {} — pk_map marker present but output is gone",
                        abs_hash.display(),
                        table.name,
                        shard
                    ));
                }
                if !abs_data.exists() {
                    return Err(anyhow!(
                        "repopulate_pk_hash_paths_from_disk: missing pk data file {} \
                         for table {} shard {} — pk_map marker present but output is gone",
                        abs_data.display(),
                        table.name,
                        shard
                    ));
                }
                paths.push(PkHashPaths {
                    hash_path,
                    data_path,
                });
            }
            self.pk_hash_paths.insert(table.name.clone(), paths);
        }
        tracing::info!(
            tables = self.pk_hash_paths.len(),
            "repopulated pk_hash_paths from on-disk pk_map outputs"
        );
        Ok(())
    }

    /// Re-derive `table_contexts` from the deterministic context-phase layout.
    /// Used on resume when context was skipped; `build_index_metadata` reads
    /// this map to point each table's index entry at its `node_data.ctx`.
    fn repopulate_table_contexts_from_source(&mut self) -> Result<()> {
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root.clone(),
            _ => return Ok(()),
        };
        self.table_contexts.clear();
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            let ctx_path = format!("nodes/{}/node_data.ctx", table.name.to_lowercase());
            let abs = root.join(&ctx_path);
            if !abs.exists() {
                return Err(anyhow!(
                    "repopulate_table_contexts_from_source: missing {} for table {} \
                     (context predecessor marker should have prevented this)",
                    abs.display(),
                    table.name,
                ));
            }
            self.table_contexts.insert(table.name.clone(), ctx_path);
        }
        tracing::info!(
            tables = self.table_contexts.len(),
            "repopulated table_contexts from source schema"
        );
        Ok(())
    }

    fn refresh_shard_rows_from_stats(&mut self, stats: &NodeWriterStats) {
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            let mut shard_rows = vec![0u64; self.num_shards];
            #[allow(clippy::needless_range_loop)]
            for shard in 0..self.num_shards {
                let key = (table.name.clone(), shard);
                if let Some(count) = stats.node_rows.get(&key) {
                    shard_rows[shard] = *count;
                }
            }
            self.shard_rows.insert(table.name.clone(), shard_rows);
        }
        if !stats.binary_label_counts.is_empty() {
            self.binary_label_counts = stats.binary_label_counts.clone();
        }
        if !stats.sample_weight_counts.is_empty() {
            self.sample_weight_counts = stats.sample_weight_counts.clone();
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

    #[allow(clippy::too_many_arguments)]
    fn build_temp_node_batches<O, I>(
        table: &TableSpec,
        batches: I,
        relationships: &[RelationshipRuntime],
        relation_defs: &[RelationDef],
        num_shards: usize,
        node_encoder: &RowPackEncoder,
        trimmed_strings: &std::sync::Arc<AtomicU64>,
        key_failures: &mut KeyFailureState,
        senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>>,
        node_errors: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        dest_key_senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
        dest_key_errors: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        fk_edge_targets: &HashMap<String, FkEdgeTarget>,
        resolve_columns: &[crate::config::ResolveColumnSpec],
        ref_resolver: Option<&crate::mapping::RefResolver>,
        sidecars: &StageSidecarColumns,
        observer: &mut O,
    ) -> Result<TempNodeFileResult>
    where
        O: diskgraph_rowpack::ColumnValueObserver,
        I: IntoIterator<Item = Result<arrow_array::RecordBatch>>,
    {
        // FK-free LP resolution mode: build the per-table runtime resolve specs
        // (only those targeting THIS table). When non-empty, the node-write pass
        // resolves the source PK reference column(s) to graph node-id(s) and
        // feeds them into the encoder via the override channel (no FK edge).
        let resolve_runtime: Vec<ResolveColRuntime> = if resolve_columns.is_empty() {
            Vec::new()
        } else {
            build_resolve_runtime(table, resolve_columns)?
        };
        let pk_col = table
            .primary_key
            .as_ref()
            .and_then(|pk| column_spec(table, pk).ok());
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
            // Pre-resolve target type from fk_edge_targets (avoids per-row HashMap lookup)
            let fk_key = format!("{}.{}", table.name, rel.src_column);
            let target_type = fk_edge_targets
                .get(&fk_key)
                .map(|t| t.target_type.clone())
                .unwrap_or_else(|| table.columns[idx].ingested_dtype.clone());
            rel_specs.push((rel_name, idx, rel_pos, rel.src_column.clone(), target_type));
        }
        let mut total_rows = 0u64;
        let row_access_hints = build_row_access_hints(table);
        let stats = RowAccessStats::new(Some(trimmed_strings.as_ref()));
        let mut access = RowAccess::new(row_access_hints, stats);
        let mut key_bufs: Vec<Vec<u8>> = vec![Vec::new(); table.columns.len()];
        let mut f16_buf: Vec<f16> = Vec::new();
        let mut node_tags = Vec::new();
        let mut node_bools = Vec::new();
        let mut node_data = Vec::new();
        // FK-free LP resolution-mode scratch + synthetic-column index set.
        let resolve_node_ids = resolve_node_id_indices(&resolve_runtime);
        let mut resolve_scratch: Vec<u8> = Vec::new();
        let mut resolve_shard_buf: Vec<u8> = Vec::new();
        let num_columns = table.columns.len();
        let mut node_data_dropped = 0u64;
        let has_pkey = pk_col.is_some() && pk_idx.is_some();
        let rel_names = std::sync::Arc::new(rel_names);
        let buffer_limit = edge_batch_buffer_limit(num_shards);
        let binary_label_idx = sidecars
            .binary_label_column
            .as_deref()
            .and_then(|col| column_index(table, col).ok());
        let sample_weight_idx = sidecars
            .sample_weight_column
            .as_deref()
            .and_then(|col| column_index(table, col).ok());
        let mut batcher = NodeBatcher::new(
            std::sync::Arc::<str>::from(table.name.clone()),
            std::sync::Arc::clone(&rel_names),
            has_pkey,
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(senders),
            std::sync::Arc::clone(node_errors),
            table.group_by_timestamp,
            binary_label_idx.is_some(),
            sample_weight_idx.is_some(),
        );
        let mut dest_fkey_batcher = DestFkeyBatcher::new(
            std::sync::Arc::<str>::from(table.name.clone()),
            std::sync::Arc::clone(&rel_names),
            num_shards,
            buffer_limit,
            std::sync::Arc::clone(dest_key_senders),
            std::sync::Arc::clone(dest_key_errors),
        );
        let mut sort_key_batchers =
            build_sort_key_batchers_for_node_table(table, relation_defs, num_shards)?;

        let create_time_info = table.create_time.as_ref().and_then(|ct| {
            let idx = column_index(table, ct).ok()?;
            let spec = column_spec(table, ct).ok()?;
            Some((idx, spec.ingested_dtype.clone()))
        });

        // TT-NG timestamp grouping: index of the `TIMESTAMP` column whose value
        // is spilled per node. Only resolved when grouping is enabled for this
        // table; a missing column is a no-op (no spill / no bucket file).
        let group_ts_idx = if table.group_by_timestamp {
            column_index(table, TIMESTAMP_GROUP_COL).ok()
        } else {
            None
        };

        let mut row_id = 0u64;

        macro_rules! process_node_batch {
            ($batch:expr) => {
                let timestamp_idx = create_time_info.as_ref().map(|(idx, _)| *idx);
                let validity_array = timestamp_idx.map(|idx| access.array(idx).unwrap().clone());

                for row in 0..$batch.num_rows() {
                    access.set_row(row);
                    total_rows += 1;
                    let timestamp_valid = validity_array
                        .as_ref()
                        .map(|a| a.is_valid(row))
                        .unwrap_or(true);
                    // Skip rows with null time_col — matches Spark's drop_na
                    // behavior during load_table_data materialization.
                    if !timestamp_valid {
                        node_data_dropped += 1;
                        continue;
                    }
                    // FK-free LP resolution: resolve the source PK reference
                    // column(s) to graph node-id(s) for this row, producing a
                    // per-column override vector. Empty when not in resolve mode.
                    let resolve_overrides = if resolve_runtime.is_empty() {
                        None
                    } else {
                        let resolver = ref_resolver.ok_or_else(|| {
                            anyhow!("resolution mode: RefResolver not provided to encode worker")
                        })?;
                        let (overrides, drop_row) = resolve_overrides_for_row(
                            &mut access,
                            table,
                            &resolve_runtime,
                            resolver,
                            num_columns,
                            &mut resolve_scratch,
                            &mut resolve_shard_buf,
                            key_failures,
                        )?;
                        // Encode-time orphan drop: skip the write so the row
                        // never enters the shuffle/batch pool.
                        if drop_row {
                            node_data_dropped += 1;
                            continue;
                        }
                        Some(overrides)
                    };

                    Self::write_node_row_from_access(
                        table,
                        pk_idx,
                        &rel_specs,
                        row,
                        &mut access,
                        node_encoder,
                        &mut batcher,
                        &mut dest_fkey_batcher,
                        &mut sort_key_batchers,
                        &mut row_id,
                        pk_col,
                        &mut key_bufs,
                        &mut f16_buf,
                        &mut node_tags,
                        &mut node_bools,
                        &mut node_data,
                        &mut node_data_dropped,
                        num_shards,
                        key_failures,
                        timestamp_valid,
                        resolve_overrides.as_deref(),
                        observer,
                        create_time_info.as_ref().map(|(idx, _)| *idx),
                        group_ts_idx,
                        binary_label_idx,
                        sample_weight_idx,
                    )?;
                }
            };
        }

        for batch in batches {
            let batch = batch?;
            let cols = build_access_cols(&batch, table, &resolve_node_ids)?;
            access.set_cols(&cols);
            process_node_batch!(batch);
            access.set_cols(&[]);
        }
        batcher.flush_all(&mut sort_key_batchers)?;
        dest_fkey_batcher.flush_all()?;
        Ok(TempNodeFileResult {
            total_rows,
            node_data_dropped,
        })
    }

    fn binary_label_byte_from_access(access: &RowAccess<'_>, idx: usize) -> Result<u8> {
        let array = access.array(idx)?;
        let row = access.row();
        if array.is_null(row) {
            return Ok(crate::binary_label::LABEL_INVALID);
        }
        macro_rules! int_boolish {
            ($array_ty:ty) => {
                if let Some(values) = array.as_any().downcast_ref::<$array_ty>() {
                    return Ok(match values.value(row) {
                        0 => crate::binary_label::LABEL_FALSE,
                        1 => crate::binary_label::LABEL_TRUE,
                        _ => crate::binary_label::LABEL_INVALID,
                    });
                }
            };
        }
        match array.data_type() {
            DataType::Boolean => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?;
                Ok(if values.value(row) {
                    crate::binary_label::LABEL_TRUE
                } else {
                    crate::binary_label::LABEL_FALSE
                })
            }
            DataType::Int8 => {
                int_boolish!(arrow_array::Int8Array);
                Ok(crate::binary_label::LABEL_INVALID)
            }
            DataType::Int16 => {
                int_boolish!(arrow_array::Int16Array);
                Ok(crate::binary_label::LABEL_INVALID)
            }
            DataType::Int32 => {
                int_boolish!(arrow_array::Int32Array);
                Ok(crate::binary_label::LABEL_INVALID)
            }
            DataType::Int64 => {
                int_boolish!(arrow_array::Int64Array);
                Ok(crate::binary_label::LABEL_INVALID)
            }
            DataType::UInt8 => {
                int_boolish!(arrow_array::UInt8Array);
                Ok(crate::binary_label::LABEL_INVALID)
            }
            DataType::UInt16 => {
                int_boolish!(arrow_array::UInt16Array);
                Ok(crate::binary_label::LABEL_INVALID)
            }
            DataType::UInt32 => {
                int_boolish!(arrow_array::UInt32Array);
                Ok(crate::binary_label::LABEL_INVALID)
            }
            DataType::UInt64 => {
                int_boolish!(arrow_array::UInt64Array);
                Ok(crate::binary_label::LABEL_INVALID)
            }
            DataType::Float32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                Ok(match values.value(row) {
                    0.0 => crate::binary_label::LABEL_FALSE,
                    1.0 => crate::binary_label::LABEL_TRUE,
                    _ => crate::binary_label::LABEL_INVALID,
                })
            }
            DataType::Float64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                Ok(match values.value(row) {
                    0.0 => crate::binary_label::LABEL_FALSE,
                    1.0 => crate::binary_label::LABEL_TRUE,
                    _ => crate::binary_label::LABEL_INVALID,
                })
            }
            _ => Ok(crate::binary_label::LABEL_INVALID),
        }
    }

    fn sample_weight_from_access(access: &RowAccess<'_>, idx: usize) -> Result<f64> {
        let array = access.array(idx)?;
        let row = access.row();
        if array.is_null(row) {
            return Ok(f64::NAN);
        }
        macro_rules! numeric_weight {
            ($array_ty:ty) => {
                if let Some(values) = array.as_any().downcast_ref::<$array_ty>() {
                    return Ok(values.value(row) as f64);
                }
            };
        }
        match array.data_type() {
            DataType::Float64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                Ok(values.value(row))
            }
            DataType::Float32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                Ok(values.value(row) as f64)
            }
            DataType::Int8 => {
                numeric_weight!(arrow_array::Int8Array);
                Ok(f64::NAN)
            }
            DataType::Int16 => {
                numeric_weight!(arrow_array::Int16Array);
                Ok(f64::NAN)
            }
            DataType::Int32 => {
                numeric_weight!(arrow_array::Int32Array);
                Ok(f64::NAN)
            }
            DataType::Int64 => {
                numeric_weight!(arrow_array::Int64Array);
                Ok(f64::NAN)
            }
            DataType::UInt8 => {
                numeric_weight!(arrow_array::UInt8Array);
                Ok(f64::NAN)
            }
            DataType::UInt16 => {
                numeric_weight!(arrow_array::UInt16Array);
                Ok(f64::NAN)
            }
            DataType::UInt32 => {
                numeric_weight!(arrow_array::UInt32Array);
                Ok(f64::NAN)
            }
            DataType::UInt64 => {
                numeric_weight!(arrow_array::UInt64Array);
                Ok(f64::NAN)
            }
            _ => Ok(f64::NAN),
        }
    }

    // rel_specs is `&[(name, rel_id, fkey_idx, fkey_col, key_type)]` — the
    // per-relation spec table threaded through ingest's per-row FK-resolution
    // path. Kept as a tuple to avoid allocating per call.
    #[allow(clippy::too_many_arguments, clippy::ptr_arg, clippy::type_complexity)]
    fn write_node_row_from_access<O: diskgraph_rowpack::ColumnValueObserver>(
        table: &TableSpec,
        pk_idx: Option<usize>,
        rel_specs: &[(std::sync::Arc<str>, usize, usize, String, IngestedType)],
        row: usize,
        access: &mut RowAccess<'_>,
        node_encoder: &RowPackEncoder,
        batcher: &mut NodeBatcher,
        dest_fkey_batcher: &mut DestFkeyBatcher,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
        row_id: &mut u64,
        pk_col: Option<&ColumnSpec>,
        key_bufs: &mut Vec<Vec<u8>>,
        f16_buf: &mut Vec<f16>,
        node_tags: &mut Vec<u8>,
        node_bools: &mut Vec<u8>,
        node_data: &mut Vec<u8>,
        node_data_dropped: &mut u64,
        num_shards: usize,
        key_failures: &mut KeyFailureState,
        timestamp_valid: bool,
        // FK-free LP resolution overrides for this row (per-column), or `None`
        // when not in resolution mode. Forwarded to `encode_row_observed`.
        resolve_overrides: Option<&[diskgraph_rowpack::ResolveOverride]>,
        observer: &mut O,
        create_time_col_idx: Option<usize>,
        // TT-NG timestamp grouping: index of the `TIMESTAMP` column to spill per
        // node, or `None` when grouping is disabled for this table.
        group_ts_idx: Option<usize>,
        binary_label_idx: Option<usize>,
        sample_weight_idx: Option<usize>,
    ) -> Result<()> {
        for buf in key_bufs.iter_mut() {
            buf.clear();
        }
        let mut has_pkey = false;
        let shard = if let (Some(idx), Some(pk_col)) = (pk_idx, pk_col) {
            let buf = key_bufs
                .get_mut(idx)
                .ok_or_else(|| anyhow!("missing key buffer"))?;
            let res = key_from_access_into_buf(access, idx, pk_col, &pk_col.ingested_dtype, buf);

            match res {
                Ok(_) => {
                    has_pkey = true;
                    shard_for_key(buf, num_shards)
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
        for (_rel_name, rel_idx, _rel_pos, _fk_col_name, target_type) in rel_specs {
            let col = table
                .columns
                .get(*rel_idx)
                .ok_or_else(|| anyhow!("missing fk column"))?;
            let buf = key_bufs
                .get_mut(*rel_idx)
                .ok_or_else(|| anyhow!("missing key buffer"))?;

            let res = key_from_access_into_buf(access, *rel_idx, col, target_type, buf);

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
        // TT-NG timestamp grouping: read this row's TIMESTAMP before the encoder
        // borrow. Stage seed tables for temporal LP always carry a valid
        // TIMESTAMP, so a null/undecodable value here is a hard error — silently
        // bucketing it under ts=0 would create a phantom bucket and corrupt the
        // grouping. `get_timestamp` is only called on the grouping-enabled path
        // (`group_ts_idx.is_some()`), so the non-grouping path is unchanged.
        let group_ts = match group_ts_idx {
            Some(idx) => Some(
                access
                    .get_timestamp(idx)
                    .with_context(|| {
                        format!(
                            "TT-NG ts-grouping: failed to decode TIMESTAMP for table `{}` row {}",
                            table.name, row
                        )
                    })?
                    .ok_or_else(|| {
                        anyhow!(
                            "TT-NG ts-grouping: null TIMESTAMP for table `{}` row {}; \
                             temporal-LP seed tables must carry a valid TIMESTAMP",
                            table.name,
                            row
                        )
                    })?,
            ),
            None => None,
        };
        let binary_label = match binary_label_idx {
            Some(idx) => Some(Self::binary_label_byte_from_access(access, idx)?),
            None => None,
        };
        let sample_weight = match sample_weight_idx {
            Some(idx) => Some(Self::sample_weight_from_access(access, idx)?),
            None => None,
        };
        node_tags.clear();
        node_bools.clear();
        node_data.clear();
        let lens = node_encoder.encode_row_observed(
            access,
            node_tags,
            node_bools,
            node_data,
            f16_buf,
            key_bufs,
            resolve_overrides,
            observer,
        )?;
        {
            // Drop row if time_col was encoded as null (tag bits == 0).
            // This catches: Arrow null that slipped past Phase 2, string parse
            // failures, and out-of-range dates (Change A in access.rs).
            if let Some(ts_idx) = create_time_col_idx {
                let bit_offset = ts_idx * 2; // 2 tag bits per column
                let byte_idx = bit_offset / 8;
                let bit_pos = (bit_offset % 8) as u8;
                if byte_idx < node_tags.len() {
                    let tag_val = (node_tags[byte_idx] >> bit_pos) & 0b11;
                    if tag_val == 0 {
                        *node_data_dropped = node_data_dropped.saturating_add(1);
                        *row_id = row_id.saturating_add(1);
                        return Ok(());
                    }
                }
            }
            let row_len = lens
                .tag_bytes
                .saturating_add(lens.bool_bytes)
                .saturating_add(lens.data_bytes);
            if row_len > MAX_NODE_DATA_LEN {
                *node_data_dropped = node_data_dropped.saturating_add(1);
                batcher.push_empty_node(shard)?;
            } else {
                batcher.push_encoded_node(shard, lens, node_tags, node_bools, node_data)?;
            }
            // Record the grouping timestamp 1:1 with the node just pushed (both
            // the normal and oversized-empty paths emit a node). The two
            // drop-paths above return early and never reach here, keeping the
            // spill aligned with surviving nodes.
            if let Some(ts) = group_ts {
                batcher.push_group_ts(shard, ts)?;
            }
            if let Some(label) = binary_label {
                batcher.push_binary_label(shard, label)?;
            }
            if let Some(weight) = sample_weight {
                batcher.push_sample_weight(shard, weight)?;
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
        for (_rel_name, rel_idx, rel_pos, _fk_col_name, _) in rel_specs {
            if let Some(buf) = key_bufs.get(*rel_idx) {
                if buf.is_empty() {
                    batcher.push_fkey_empty(shard, *rel_pos)?;
                } else {
                    batcher.push_fkey(shard, *rel_pos, buf, timestamp_valid)?;
                    // dest_fkey_batcher writes per-dst-shard accumulator data
                    // used to build the reverse (dst→src) sort order.
                    let dst_shard = shard_for_key(buf, num_shards);
                    dest_fkey_batcher.push(dst_shard, *rel_pos, buf)?;
                }
            }
        }
        batcher.maybe_flush(shard, sort_key_batchers)?;
        *row_id = row_id.saturating_add(1);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_temp_edge_batches<I>(
        table: &TableSpec,
        batches: I,
        num_shards: usize,
        trimmed_strings: &std::sync::Arc<AtomicU64>,
        key_failures: &mut KeyFailureState,
        senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<EdgeSrcWriteMessage>>>,
        error: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        dest_key_senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
        dest_key_errors: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        relation_defs: &[RelationDef],
        fk_edge_targets: &HashMap<String, FkEdgeTarget>,
    ) -> Result<u64>
    where
        I: IntoIterator<Item = Result<arrow_array::RecordBatch>>,
    {
        let edge = table
            .edge_table
            .as_ref()
            .ok_or_else(|| anyhow!("missing edge table spec for {}", table.name))?;
        let src_fk_idx = column_index(table, &edge.src_fk.edge_column)?;
        let dst_fk_idx = column_index(table, &edge.dst_fk.edge_column)?;
        let src_fk_col = column_spec(table, &edge.src_fk.edge_column)?;
        let dst_fk_col = column_spec(table, &edge.dst_fk.edge_column)?;
        // Pre-resolve target types for src and dst FK columns
        let src_target_type = fk_edge_targets
            .get(&format!("{}.{}", table.name, edge.src_fk.edge_column))
            .map(|t| &t.target_type)
            .unwrap_or(&src_fk_col.ingested_dtype);
        let dst_target_type = fk_edge_targets
            .get(&format!("{}.{}", table.name, edge.dst_fk.edge_column))
            .map(|t| &t.target_type)
            .unwrap_or(&dst_fk_col.ingested_dtype);
        let src_context = format!("edge_fk:{}.{}", table.name, edge.src_fk.edge_column);
        let dst_context = format!("edge_fk:{}.{}", table.name, edge.dst_fk.edge_column);
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
        let mut sort_key_batchers =
            build_sort_key_batchers_for_edge_table(table, relation_defs, num_shards)?;

        let create_time_info = table.create_time.as_ref().and_then(|ct| {
            let idx = column_index(table, ct).ok()?;
            let spec = column_spec(table, ct).ok()?;
            Some((idx, spec.ingested_dtype.clone()))
        });

        let mut edge_dropped = 0u64;

        let row_access_hints = build_row_access_hints(table);
        let stats = RowAccessStats::new(Some(trimmed_strings.as_ref()));
        let mut access = RowAccess::new(row_access_hints, stats);
        macro_rules! process_edge_batch {
            ($batch:expr, $src_key_buf:expr, $dst_key_buf:expr, $forward_row:expr) => {
                let timestamp_idx = create_time_info.as_ref().map(|(idx, _)| *idx);
                let validity_array = timestamp_idx.map(|idx| access.array(idx).unwrap().clone());

                let create_time_col_idx = create_time_info.as_ref().map(|(idx, _)| *idx);
                for row in 0..$batch.num_rows() {
                    access.set_row(row);
                    let mut timestamp_valid = validity_array
                        .as_ref()
                        .map(|a| a.is_valid(row))
                        .unwrap_or(true);
                    // Also check for out-of-range / parse failures on time_col.
                    // get_timestamp() returns Err for out-of-range Date32 (Change A).
                    if timestamp_valid {
                        if let Some(ts_idx) = create_time_col_idx {
                            match access.get_timestamp(ts_idx) {
                                Ok(Some(_)) => {} // valid
                                _ => {
                                    timestamp_valid = false;
                                }
                            }
                        }
                    }
                    if !timestamp_valid {
                        edge_dropped += 1;
                        continue;
                    }
                    Self::write_edge_row_from_access(
                        src_fk_idx,
                        dst_fk_idx,
                        src_fk_col,
                        dst_fk_col,
                        src_target_type,
                        dst_target_type,
                        &src_context,
                        &dst_context,
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

        macro_rules! consume_edge_reader {
            ($reader:expr) => {
                for batch in $reader {
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
                    let mut src_key_buf = Vec::new();
                    let mut dst_key_buf = Vec::new();
                    let mut forward_row = Vec::new();
                    process_edge_batch!(
                        batch,
                        &mut src_key_buf,
                        &mut dst_key_buf,
                        &mut forward_row
                    );
                    access.set_cols(&[]);
                }
            };
        }

        consume_edge_reader!(batches);
        batcher.flush_all(&mut sort_key_batchers)?;
        edge_key_batcher.flush_all()?;
        Ok(edge_dropped)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_edge_row_from_access(
        src_idx: usize,
        dst_idx: usize,
        src_col: &ColumnSpec,
        dst_col: &ColumnSpec,
        src_target_type: &IngestedType,
        dst_target_type: &IngestedType,
        src_context: &str,
        dst_context: &str,
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
        if key_from_access_into_buf(access, src_idx, src_col, src_target_type, src_key_buf).is_err()
        {
            record_key_parse_failure_sample(
                &mut key_failures.parse_failures,
                &mut key_failures.samples,
                &mut key_failures.by_context,
                src_context,
                sample_from_access(access, src_idx),
            );
            return Ok(());
        }

        if key_from_access_into_buf(access, dst_idx, dst_col, dst_target_type, dst_key_buf).is_err()
        {
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
        forward_row.reserve(
            src_key_buf
                .len()
                .saturating_add(dst_key_buf.len())
                .saturating_add(4),
        );
        append_key_bytes_with_len_prefix(forward_row, src_key_buf, src_col, src_target_type)?;
        append_key_bytes_with_len_prefix(forward_row, dst_key_buf, dst_col, dst_target_type)?;
        for sk in sort_key_batchers.iter_mut() {
            sk.push_row(forward_shard, access)?;
        }
        batcher.push(
            forward_shard,
            forward_row,
            timestamp_valid,
            sort_key_batchers,
        )?;
        edge_key_batcher.push_src(forward_shard, src_key_buf)?;
        edge_key_batcher.push_dst(dst_shard, dst_key_buf)?;
        Ok(())
    }

    fn build_sharded_pk_maps(&mut self) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        self.pk_duplicate_counts.clear();
        self.pk_duplicate_samples.clear();
        self.pk_null_timestamp_drops.clear();
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
        let fk_edge_targets = self.fk_edge_targets.clone();
        let mut tasks = Vec::new();
        let mut pk_hash_handles: HashMap<
            String,
            std::sync::Arc<std::sync::Mutex<Vec<Option<PkHashPaths>>>>,
        > = HashMap::new();
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
            let max_key_len = match key_type_for_column(pk_col)? {
                KeyType::VarBytes => *self
                    .pk_sampled_max_key_len
                    .get(&table.name)
                    .ok_or_else(|| anyhow!("table {} was not sampled", table.name))?,
                KeyType::Int64 => 8,
                KeyType::Uuid | KeyType::Ip => 16,
                KeyType::Uuid256 => 32,
            };
            let ptrhash_params = ptr_hash::PtrHashParams {
                prefer_fallback_hash: max_key_len > ptr_hash::MAX_GXHASH_INJECTIVE_LEN,
                ..ptr_hash::PtrHashParams::default()
            };
            let pk_hash_paths =
                std::sync::Arc::new(std::sync::Mutex::new(vec![None; self.num_shards]));
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
                let fk_edge_targets = fk_edge_targets.clone();
                let estimate_sum = std::sync::Arc::clone(&self.pk_map_estimate_sum);
                let peak_rss = std::sync::Arc::clone(&self.pk_map_peak_rss);
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
                        // Store (path, meta_path, target_type) - use target type for reading FK temp files
                        let mut fkey_inputs: Vec<(PathBuf, PathBuf, IngestedType)> = Vec::new();
                        let mut edge_inputs: Vec<(PathBuf, PathBuf, IngestedType)> = Vec::new();
                        for rel in relationships.iter().filter(|rel| rel.dst_table == table_name) {
                            // Look up the target type from fk_edge_targets (FK keys were written using target type)
                            let fk_key = format!("{}.{}", rel.src_table, rel.src_column);
                            let target_type = fk_edge_targets
                                .get(&fk_key)
                                .map(|t| t.target_type.clone())
                                .unwrap_or_else(|| {
                                    // Fallback to PK column type (this table's PK)
                                    pk_col.ingested_dtype.clone()
                                });
                            fkey_inputs.push((
                                root.join(temp_fkey_dst_path(&rel.src_table, &rel.name_forward, shard)),
                                root.join(temp_fkey_dst_meta_path(&rel.src_table, &rel.name_forward, shard)),
                                target_type,
                            ));
                        }
                        for table in source.tables.iter().filter(|t| t.edge_table.is_some()) {
                            let edge = table.edge_table.as_ref().unwrap();
                            if edge.src_fk.ref_table == table_name {
                                // Look up target type for edge table src FK
                                let fk_key = format!("{}.{}", table.name, edge.src_fk.edge_column);
                                let target_type = fk_edge_targets
                                    .get(&fk_key)
                                    .map(|t| t.target_type.clone())
                                    .unwrap_or_else(|| pk_col.ingested_dtype.clone());
                                edge_inputs.push((
                                    root.join(temp_edge_key_path(&table.name, EdgeKeyKind::Src, shard)),
                                    root.join(temp_edge_key_meta_path(&table.name, EdgeKeyKind::Src, shard)),
                                    target_type,
                                ));
                            }
                            if edge.dst_fk.ref_table == table_name {
                                // Look up target type for edge table dst FK
                                let fk_key = format!("{}.{}", table.name, edge.dst_fk.edge_column);
                                let target_type = fk_edge_targets
                                    .get(&fk_key)
                                    .map(|t| t.target_type.clone())
                                    .unwrap_or_else(|| pk_col.ingested_dtype.clone());
                                edge_inputs.push((
                                    root.join(temp_edge_key_path(&table.name, EdgeKeyKind::Dst, shard)),
                                    root.join(temp_edge_key_meta_path(&table.name, EdgeKeyKind::Dst, shard)),
                                    target_type,
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
                        // Track this task's estimate so we can compare the
                        // sum-of-estimates vs the actual peak RSS at end of pk_map.
                        estimate_sum.fetch_add(estimate, std::sync::atomic::Ordering::Relaxed);
                        // Sample current RSS at task entry — gives a cheap
                        // proxy for the high-water mark across the phase.
                        update_atomic_max(&peak_rss, rss_anon_diag_bytes());
                        let _budget_guard = if let Some(budget) = pk_map_budget.as_ref() {
                            let budget_total = budget.snapshot().total;
                            if budget_total == 0 || estimate == 0 {
                                None
                            } else {
                                // A single shard whose PK-map estimate exceeds the whole
                                // budget cannot be made to fit by halving threads (threads
                                // don't shrink a shard), so retrying the same shard is
                                // futile. Surface a permanent, actionable anyhow error
                                // (rc=1 → ingest_runner classifies as "fatal"/
                                // non-retryable) instead of letting `acquire` fail via
                                // `panic!` (rc=134), which the runner would treat as
                                // OOM-class and retry with halved threads to exhaustion.
                                // With cgroup-aware shard sizing this should be rare.
                                if estimate > budget_total {
                                    anyhow::bail!(
                                        "single-shard PK-map estimate {estimate} B exceeds \
                                         memory budget {budget_total} B; increase num_shards \
                                         (re-run infer or raise num_shards) — halving threads \
                                         cannot reduce a single shard"
                                    );
                                }
                                // Normal case (fits): keep the existing per-shard acquire
                                // semantics, which block until the budget frees up.
                                Some(budget.acquire(estimate)?)
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
                                        for (path, _meta_path, target_type) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf_for_type(&mut data, target_type)?;
                                                if let Key::Int(v) = key {
                                                    seen.entry(v).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptrhash_params;
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
                                    let pre_clear_cap = pkmap.seen_bytes.capacity();
                                    pkmap.seen_bytes.clear();
                                    reserve_map(&mut pkmap.seen_bytes, reserve_rows_usize);
                                    tracing::info!(
                                        target: "diskgraph::pk_map_diag",
                                        table = %table_name, shard, phase = "task_start",
                                        seen_len = 0u64,
                                        seen_cap = pkmap.seen_bytes.capacity() as u64,
                                        prev_seen_cap = pre_clear_cap as u64,
                                        reserve_rows = reserve_rows_usize as u64,
                                        union_rows = union_rows,
                                        avg_var_bytes = avg_var_bytes,
                                        rss_anon = rss_anon_diag_bytes(),
                                        "pk_map: starting VarBytes shard task"
                                    );
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
                                        tracing::info!(
                                            target: "diskgraph::pk_map_diag",
                                            table = %table_name, shard, phase = "after_pk_ingest",
                                            seen_len = seen.len() as u64,
                                            seen_cap = seen.capacity() as u64,
                                            row_id,
                                            rss_anon = rss_anon_diag_bytes(),
                                            "pk_map: PK rows ingested"
                                        );
                                        for (idx, (path, _meta_path, target_type)) in
                                            fkey_inputs.iter().chain(edge_inputs.iter()).enumerate()
                                        {
                                            let len_before = seen.len();
                                            let cap_before = seen.capacity();
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf_for_type(&mut data, target_type)?;
                                                if let Key::Bytes(bytes) = key {
                                                    seen.entry(bytes).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                            tracing::info!(
                                                target: "diskgraph::pk_map_diag",
                                                table = %table_name, shard, phase = "after_fk_file",
                                                fk_index = idx,
                                                fk_path = %path.display(),
                                                seen_len_before = len_before as u64,
                                                seen_len_after = seen.len() as u64,
                                                seen_cap_before = cap_before as u64,
                                                seen_cap_after = seen.capacity() as u64,
                                                rss_anon = rss_anon_diag_bytes(),
                                                "pk_map: FK file merged"
                                            );
                                        }
                                    }
                                    let params = ptrhash_params;
                                    let seen = std::mem::take(&mut pkmap.seen_bytes);
                                    tracing::info!(
                                        target: "diskgraph::pk_map_diag",
                                        table = %table_name, shard, phase = "before_mphf",
                                        seen_len = seen.len() as u64,
                                        seen_cap = seen.capacity() as u64,
                                        rss_anon = rss_anon_diag_bytes(),
                                        "pk_map: building MPHF"
                                    );
                                    let mphf =
                                        DefaultPtrHash::<PkPtrHash, Vec<u8>>::try_new_from_source_with_scratch(
                                            &seen,
                                            params,
                                            &mut ptrhash,
                                        )
                                        .ok_or_else(|| anyhow!("ptrhash construction failed"))?;
                                    tracing::info!(
                                        target: "diskgraph::pk_map_diag",
                                        table = %table_name, shard, phase = "after_mphf",
                                        seen_len = seen.len() as u64,
                                        mphf_slots = mphf.max_index() as u64,
                                        rss_anon = rss_anon_diag_bytes(),
                                        "pk_map: MPHF built"
                                    );
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
                                    tracing::info!(
                                        target: "diskgraph::pk_map_diag",
                                        table = %table_name, shard, phase = "task_done",
                                        rss_anon = rss_anon_diag_bytes(),
                                        "pk_map: VarBytes shard task complete"
                                    );
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
                                        for (path, _meta_path, target_type) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf_for_type(&mut data, target_type)?;
                                                if let Key::Uuid(uuid) = key {
                                                    seen.entry(uuid).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptrhash_params;
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
                                        for (path, _meta_path, target_type) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf_for_type(&mut data, target_type)?;
                                                if let Key::Uuid256(uuid) = key {
                                                    seen.entry(uuid).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptrhash_params;
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
                                        for (path, _meta_path, target_type) in
                                            fkey_inputs.iter().chain(edge_inputs.iter())
                                        {
                                            read_temp_rows(path, |row| {
                                                let mut data = row;
                                                let key = read_key_from_buf_for_type(&mut data, target_type)?;
                                                if let Key::Ip(ip) = key {
                                                    seen.entry(ip).or_insert(INVALID_LOCAL_ID);
                                                }
                                                Ok(())
                                            })?;
                                        }
                                    }
                                    let params = ptrhash_params;
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
        let pool = self.make_pool::<PkMapScratch>(self.pk_map_threads);
        let run_result = run_tasks(std::mem::take(&mut tasks), &pool);
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
        let mut hash_writer = BufWriter::with_capacity(4 * 1024 * 1024, target.sink(&hash_path)?);
        mphf.serialize(&mut hash_writer)?;
        hash_writer.flush()?;
        let hash_sink = hash_writer
            .into_inner()
            .map_err(|_| anyhow!("pk hash flush failed"))?;
        hash_sink.finish()?;

        let mut data_writer = BufWriter::with_capacity(4 * 1024 * 1024, target.sink(&data_path)?);
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
                rel_sort_keys.insert(rel.name.clone(), RelationSortKeyInfo { timestamp, weight });
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
        for lookup in rel_sort_lookups
            .values()
            .filter_map(|lookup| lookup.as_ref())
        {
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
            tracing::info!(
                total_bytes = %format_bytes(total_bytes),
                total_mem = %format_bytes(total_mem),
                advice = ?data_advice,
                "pk_hash mmap hint"
            );
        } else {
            tracing::info!(
                total_bytes = %format_bytes(total_bytes),
                total_mem = "unknown",
                advice = ?data_advice,
                "pk_hash mmap hint"
            );
        }
        let (tx, rx) = std::sync::mpsc::channel::<Result<(String, usize, PkHashShard)>>();
        let mut tasks = Vec::with_capacity(shard_jobs.len());
        for (table_name, shard_idx, key_type, hash_path, data_path) in shard_jobs {
            let tx = tx.clone();
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
        let pool = self.make_pool::<()>(self.edges_map_threads);
        run_tasks(tasks, &pool)?;
        let mut tables: HashMap<String, Vec<Option<PkHashShard>>> = HashMap::new();
        for (table_name, shard_count) in table_shard_counts {
            let shards = (0..shard_count).map(|_| None).collect();
            tables.insert(table_name, shards);
        }
        for _ in 0..tables.values().map(|v| v.len()).sum::<usize>() {
            match rx
                .recv()
                .map_err(|e| anyhow!("pk hash output error: {e}"))?
            {
                Ok((table_name, shard_idx, shard)) => {
                    let entry = tables
                        .get_mut(&table_name)
                        .ok_or_else(|| anyhow!("pk hash table missing {}", table_name))?;
                    if shard_idx >= entry.len() {
                        return Err(anyhow!(
                            "pk hash shard out of range {}:{}",
                            table_name,
                            shard_idx
                        ));
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
                resolved
                    .push(shard.ok_or_else(|| anyhow!("missing pk hash shard {}", table_name))?);
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
        let (tx, rx) =
            std::sync::mpsc::channel::<Result<(String, Option<std::sync::Arc<SortKeyLookup>>)>>();
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
        let pool = self.make_pool::<()>(self.edges_map_threads);
        run_tasks(tasks, &pool)?;
        let mut lookups = HashMap::with_capacity(self.relation_defs.len());
        for _ in 0..self.relation_defs.len() {
            match rx
                .recv()
                .map_err(|e| anyhow!("sort key output error: {e}"))?
            {
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
        self.edge_dst_nonzero_counts.clear();
        self.pk_duplicate_drops.clear();
        self.edge_null_timestamp_drops.clear();

        let pool = self.make_pool::<EdgeMapScratch>(self.edges_map_threads);
        {
            #[derive(Default)]
            struct EdgeMapResult {
                relation: String,
                key_failures: KeyFailureState,
                edge_counts: HashMap<String, u64>,
                pk_duplicate_drops: u64,
                null_timestamp_drops: u64,
                /// Forward relation name → bitvec tracking unique dst node IDs.
                dst_degree_bits: HashMap<String, BudgetedDstDegreeBitmap>,
            }
            if let Some(total_mem) = total_memory_bytes() {
                tracing::info!(
                    pk_hash_bytes = %format_bytes(edge_ctx.pk_hash_total_bytes),
                    sort_key_bytes = %format_bytes(edge_ctx.sort_key_total_bytes),
                    sort_key_edge_bytes = %format_bytes(edge_ctx.sort_key_edge_bytes),
                    sort_key_dest_bytes = %format_bytes(edge_ctx.sort_key_dest_bytes),
                    total_mem = %format_bytes(total_mem),
                    "edges_map mmap sizes"
                );
            } else {
                tracing::info!(
                    pk_hash_bytes = %format_bytes(edge_ctx.pk_hash_total_bytes),
                    sort_key_bytes = %format_bytes(edge_ctx.sort_key_total_bytes),
                    sort_key_edge_bytes = %format_bytes(edge_ctx.sort_key_edge_bytes),
                    sort_key_dest_bytes = %format_bytes(edge_ctx.sort_key_dest_bytes),
                    total_mem = "unknown",
                    "edges_map mmap sizes"
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
            // Destination-degree bitmaps are the largest anonymous allocation in
            // this phase. Reserve at most 20% of process memory for all running
            // and queued shard bitmaps, leaving headroom for PK maps, sort keys,
            // writer buffers, and the rest of the service stack.
            let dst_degree_bitmap_budget = total_memory_bytes().map(|total| {
                std::sync::Arc::new(MemoryBudget::new(total.saturating_mul(20) / 100))
            });
            // A result can contain a destination bitmap proportional to the full
            // destination table. Keep at most one completed result queued while
            // the coordinator merges it into the relation-level bitmap.
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<EdgeMapResult>>(1);
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
                // Look up target_type for this FK (keys were written using target_type in encode)
                let fk_key = format!("{}.{}", rel.src_table, rel.src_column);
                let fk_target_type = self
                    .fk_edge_targets
                    .get(&fk_key)
                    .map(|t| t.target_type.clone())
                    .unwrap_or_else(|| src_fk_col.ingested_dtype.clone());
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
                    let fk_target_type = fk_target_type.clone();
                    let dst_degree_bitmap_budget = dst_degree_bitmap_budget.clone();
                    tasks.push(WorkItem {
                        bucket: None,
                        run: Box::new(move |scratch: &mut EdgeMapScratch| {
                            let failure_context =
                                format!("fkey:{}.{}", rel.src_table, rel.src_column);
                            let lookup_context = format!("dst_lookup:{}", rel.name_forward);
                            let mut result = EdgeMapResult {
                                relation: rel.name_forward.clone(),
                                ..Default::default()
                            };
                            let src_offsets = shard_offsets
                                .get(&rel.src_table)
                                .cloned()
                                .unwrap_or_else(|| vec![0; num_shards]);
                            let dst_offsets = shard_offsets
                                .get(&rel.dst_table)
                                .cloned()
                                .unwrap_or_else(|| vec![0; num_shards]);
                            let forward_sort_bits =
                                rel_sort_bits.get(&rel.name_forward).copied().flatten();
                            let reverse_sort_bits =
                                rel_sort_bits.get(&rel.name_reverse).copied().flatten();
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
                            // Snapshot per-relationship flags up front so the
                            // hot per-row loop sees a `bool` local (perfect
                            // branch prediction) rather than chasing through
                            // `rel.*` field accesses every iteration.
                            let bidirectional = rel.bidirectional;
                            let mut forward_batcher = EdgeBatcher::new(
                                std::sync::Arc::<str>::from(rel.name_forward.clone()),
                                forward_spec,
                                num_shards,
                                buffer_limit,
                                std::sync::Arc::clone(&writer_senders),
                                std::sync::Arc::clone(&buffer_pool),
                            );
                            // Only construct the reverse batcher when needed —
                            // saves an Arc clone + buffer allocations per shard
                            // for the seed-table case (bidirectional=false).
                            let mut reverse_batcher = if bidirectional {
                                Some(EdgeBatcher::new(
                                    std::sync::Arc::<str>::from(rel.name_reverse.clone()),
                                    reverse_spec,
                                    num_shards,
                                    buffer_limit,
                                    std::sync::Arc::clone(&writer_senders),
                                    std::sync::Arc::clone(&buffer_pool),
                                ))
                            } else {
                                let _ = reverse_spec; // silence unused warning
                                None
                            };
                            let fkey_meta = root.join(temp_fkey_meta_path(
                                &rel.src_table,
                                &rel.name_forward,
                                shard,
                            ));
                            let meta = read_temp_meta(&fkey_meta)?;
                            // Route dst keys and load dst_maps with the dst
                            // table's actual pkmap shard count.
                            let dst_num_shards = pk_hash_store
                                .table_shard_count(&rel.dst_table)
                                .unwrap_or(num_shards);
                            let mut dst_maps = Vec::with_capacity(dst_num_shards);
                            for dst_shard in 0..dst_num_shards {
                                dst_maps.push(pk_hash_store.shard(&rel.dst_table, dst_shard)?);
                            }
                            let mut pending = 0u64;
                            let mut local_count = 0u64;
                            let mut forward_count = 0u64;
                            let mut reverse_count = 0u64;
                            let mut pk_duplicate_drops = 0u64;
                            // dst_degree_bits is a per-task bitvec sized for the dst
                            // table's max_id; it tracks which dst rows received an FK
                            // edge so kumo can compute per-rel uniqueness stats.
                            let dst_max = max_ids.get(&rel.dst_table).copied().unwrap_or(0);
                            let dst_bitmap_bytes = (dst_max + 8) / 8;
                            let dst_degree_budget_guard =
                                if let Some(budget) = &dst_degree_bitmap_budget {
                                    let budget_total = budget.snapshot().total;
                                    ensure_dst_degree_bitmap_budget(
                                        dst_bitmap_bytes,
                                        budget_total,
                                    )
                                    .with_context(|| {
                                        format!("relation {}", rel.name_forward)
                                    })?;
                                    Some(budget.acquire(dst_bitmap_bytes)?)
                                } else {
                                    None
                                };
                            let dst_bitmap_len = usize::try_from(dst_bitmap_bytes)
                                .map_err(|_| anyhow!("destination-degree bitmap size overflow"))?;
                            let mut dst_bits = vec![0u8; dst_bitmap_len];
                            let fkey_path =
                                root.join(temp_fkey_path(&rel.src_table, &rel.name_forward, shard));
                            let bitmap_path = root.join(temp_fkey_bitmap_path(
                                &rel.src_table,
                                &rel.name_forward,
                                shard,
                            ));
                            {
                                let edge_scratch = scratch;
                                edge_scratch.key_buf.clear();
                                let fkey_bitmap =
                                    read_fkey_bitmap_mmap(&bitmap_path, meta.row_count)?;
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
                                    let dst_key = match read_key_from_buf_for_type(
                                        &mut data,
                                        &fk_target_type,
                                    ) {
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
                                        shard_for_key_from_key(&dst_key, key_buf, dst_num_shards)?;
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
                                        Some(lookup) => {
                                            lookup.sort_key(dst_shard, dst_local, shard, 0)?
                                        }
                                        None => 0,
                                    };
                                    let reverse_sort_key = if bidirectional {
                                        match &reverse_lookup {
                                            Some(lookup) => {
                                                lookup.sort_key(shard, src_local, shard, 0)?
                                            }
                                            None => 0,
                                        }
                                    } else {
                                        0
                                    };
                                    forward_batcher.push(
                                        shard,
                                        src_id,
                                        dst_id,
                                        forward_sort_key,
                                    )?;
                                    if let Some(rb) = reverse_batcher.as_mut() {
                                        rb.push(dst_shard, dst_id, src_id, reverse_sort_key)?;
                                        reverse_count = reverse_count.saturating_add(1);
                                    }
                                    let bit_idx = dst_id as usize;
                                    if bit_idx / 8 < dst_bits.len() {
                                        dst_bits[bit_idx / 8] |= 1u8 << (bit_idx % 8) as u8;
                                    }
                                    forward_count = forward_count.saturating_add(1);
                                    Ok(())
                                });
                                result?;
                            }
                            forward_batcher.flush_all()?;
                            if let Some(rb) = reverse_batcher.as_mut() {
                                rb.flush_all()?;
                            }
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
                            if bidirectional {
                                result
                                    .edge_counts
                                    .insert(rel.name_reverse.clone(), reverse_count);
                            }
                            result
                                .dst_degree_bits
                                .insert(
                                    rel.name_forward.clone(),
                                    BudgetedDstDegreeBitmap {
                                        bits: dst_bits,
                                        _budget_guard: dst_degree_budget_guard,
                                    },
                                );
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
                            let mut result = EdgeMapResult {
                                relation: forward_name.clone(),
                                ..Default::default()
                            };
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
                            let forward_sort_bits =
                                rel_sort_bits.get(&forward_name).copied().flatten();
                            let reverse_sort_bits =
                                rel_sort_bits.get(&reverse_name).copied().flatten();
                            let forward_lookup = rel_sort_lookups
                                .get(&forward_name)
                                .and_then(|lookup| lookup.as_ref().map(std::sync::Arc::clone));
                            let reverse_lookup = rel_sort_lookups
                                .get(&reverse_name)
                                .and_then(|lookup| lookup.as_ref().map(std::sync::Arc::clone));
                            let forward_spec = edge_binary_spec(
                                max_ids.get(&edge.src_fk.ref_table).copied().unwrap_or(0),
                                max_ids.get(&edge.dst_fk.ref_table).copied().unwrap_or(0),
                                forward_sort_bits,
                            );
                            let reverse_spec = edge_binary_spec(
                                max_ids.get(&edge.dst_fk.ref_table).copied().unwrap_or(0),
                                max_ids.get(&edge.src_fk.ref_table).copied().unwrap_or(0),
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
                                    let src_key =
                                        read_key_from_buf_len_prefixed(&mut data, &src_pk_col)?;
                                    let dst_key =
                                        read_key_from_buf_len_prefixed(&mut data, &dst_pk_col)?;
                                    let timestamp_valid =
                                        if !data.is_empty() { data[0] != 0 } else { true };
                                    if !timestamp_valid {
                                        null_timestamp_drops =
                                            null_timestamp_drops.saturating_add(1);
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
                                        Some(lookup) => {
                                            lookup.sort_key(dst_shard, dst_local, shard, row_idx)?
                                        }
                                        None => 0,
                                    };
                                    let reverse_sort_key = match &reverse_lookup {
                                        Some(lookup) => {
                                            lookup.sort_key(src_shard, src_local, shard, row_idx)?
                                        }
                                        None => 0,
                                    };
                                    forward_batcher.push(
                                        shard,
                                        src_id,
                                        dst_id,
                                        forward_sort_key,
                                    )?;
                                    reverse_batcher.push(
                                        dst_shard,
                                        dst_id,
                                        src_id,
                                        reverse_sort_key,
                                    )?;
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
            let mut done = HashMap::new();
            let mut merged_dst_degree_bits: HashMap<String, BudgetedDstDegreeBitmap> =
                HashMap::new();
            // Workers must run concurrently with result consumption. Waiting for
            // every task first retains num_shards full-destination bitmaps per
            // relation in this channel (O(num_shards * destination_nodes)).
            // Streaming the merge bounds retained result memory to the running
            // workers plus the channel's single completed result.
            std::thread::scope(|scope| -> Result<()> {
                let task_runner = scope.spawn(|| run_tasks(tasks, &pool));
                let mut output_error = None;
                for _ in 0..task_count {
                    let result = match rx.recv() {
                        Ok(Ok(result)) => result,
                        Ok(Err(error)) => {
                            if output_error.is_none() {
                                output_error = Some(error);
                            }
                            continue;
                        }
                        Err(error) => {
                            if output_error.is_none() {
                                output_error =
                                    Some(anyhow!("edges_map output error: {error}"));
                            }
                            break;
                        }
                    };
                    self.merge_key_failures(result.key_failures);
                    for (rel, count) in result.edge_counts {
                        let entry = self.edge_counts.entry(rel).or_insert(0);
                        *entry = entry.saturating_add(count);
                    }
                    for (rel, bitmap) in result.dst_degree_bits {
                        merge_dst_degree_bitmap(&mut merged_dst_degree_bits, rel, bitmap);
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
                            if let Some(bitmap) =
                                merged_dst_degree_bits.remove(&result.relation)
                            {
                                let count = bitmap
                                    .bits
                                    .iter()
                                    .map(|byte| byte.count_ones() as u64)
                                    .sum();
                                self.edge_dst_nonzero_counts
                                    .insert(result.relation.clone(), count);
                            }
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
                let task_result = task_runner
                    .join()
                    .map_err(|_| anyhow!("edges_map task runner panicked"))?;
                task_result?;
                if let Some(error) = output_error {
                    return Err(error);
                }
                Ok(())
            })?;
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
            for (relation, bitmap) in merged_dst_degree_bits {
                let count = bitmap
                    .bits
                    .iter()
                    .map(|byte| byte.count_ones() as u64)
                    .sum();
                self.edge_dst_nonzero_counts.insert(relation, count);
            }
            // Post-edges_map cleanup of inputs (edges_src/fkeys/fkeys_dst/edge_keys)
            // is now driven by `ingest_with_stats_from_source` after the edges_map
            // marker is written, so a crash mid-cleanup cannot leave the pipeline
            // in a state that breaks resume.
            Ok(())
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

    /// TT-NG timestamp grouping merge. For each table with
    /// `group_by_timestamp = true`, combine its per-shard timestamp spills (in
    /// node-write order) with the shard `min_id`s into a single
    /// `<table>/bucket_index.bin` of final node-ids grouped by TIMESTAMP, and
    /// return the per-table bucket lists. No-op (empty map) when no table opted
    /// in. Requires local output + already-built `shard_offsets`.
    fn run_ts_grouping(&self) -> Result<HashMap<String, crate::ts_grouping::StageGrouping>> {
        let mut out = HashMap::new();
        if !self
            .source
            .tables
            .iter()
            .any(|table| table.group_by_timestamp)
        {
            return Ok(out);
        }
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root.clone(),
            _ => return Err(anyhow!("ts grouping requires local output")),
        };
        for table in &self.source.tables {
            if !table.group_by_timestamp {
                continue;
            }
            let offsets = self.shard_offsets.get(&table.name).ok_or_else(|| {
                anyhow!(
                    "ts grouping: missing shard offsets for table {}",
                    table.name
                )
            })?;
            // Build the per-shard spill inputs. A shard with no spill file
            // (zero nodes routed to it) contributes nothing.
            let mut shards = Vec::with_capacity(self.num_shards);
            for shard in 0..self.num_shards {
                let spill_path = root.join(crate::ts_grouping::spill_rel_path(&table.name, shard));
                if !spill_path.exists() {
                    continue;
                }
                shards.push(crate::ts_grouping::ShardSpill {
                    min_id: offsets.get(shard).copied().unwrap_or(0),
                    spill_path,
                });
            }
            if shards.is_empty() {
                continue;
            }
            let bucket_rel = format!(
                "{}/{}",
                table.name,
                crate::ts_grouping::BUCKET_INDEX_FILENAME
            );
            let bucket_abs = root.join(&bucket_rel);
            let grouping = crate::ts_grouping::merge_stage(&shards, &bucket_abs, bucket_rel)?;
            out.insert(table.name.clone(), grouping);
        }
        Ok(out)
    }

    fn run_binary_label_indexes(
        &self,
    ) -> Result<HashMap<String, crate::binary_label::BinaryLabelIndex>> {
        let mut out = HashMap::new();
        if self.binary_label_counts.is_empty() {
            return Ok(out);
        }
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root.clone(),
            _ => return Err(anyhow!("binary label sidecars require local output")),
        };
        for table in &self.source.tables {
            if self.binary_label_column_for(&table.name).is_none() {
                continue;
            }
            let offsets = self.shard_offsets.get(&table.name).ok_or_else(|| {
                anyhow!(
                    "binary label sidecars: missing shard offsets for table {}",
                    table.name
                )
            })?;
            let mut shards = Vec::with_capacity(self.num_shards);
            for shard in 0..self.num_shards {
                let key = (table.name.clone(), shard);
                let Some(counts) = self.binary_label_counts.get(&key) else {
                    continue;
                };
                let spill_path = root.join(crate::binary_label::spill_rel_path(&table.name, shard));
                if !spill_path.exists() {
                    return Err(anyhow!(
                        "missing binary-label spill for table {} shard {} at {}",
                        table.name,
                        shard,
                        spill_path.display()
                    ));
                }
                shards.push(crate::binary_label::ShardSpill {
                    min_id: offsets.get(shard).copied().ok_or_else(|| {
                        anyhow!(
                            "binary label sidecars: missing shard offset for table {} shard {}",
                            table.name,
                            shard
                        )
                    })?,
                    counts: counts.clone(),
                    spill_path,
                });
            }
            if shards.is_empty() {
                continue;
            }
            let (false_rel, true_rel) = crate::binary_label::output_rel_paths(&table.name);
            if let Some(index) = crate::binary_label::merge_stage(
                &shards,
                &root.join(&false_rel),
                false_rel,
                &root.join(&true_rel),
                true_rel,
            )? {
                out.insert(table.name.clone(), index);
            }
        }
        Ok(out)
    }

    fn run_sample_weight_indexes(
        &self,
    ) -> Result<HashMap<String, crate::sample_weight::SampleWeightIndex>> {
        let mut out = HashMap::new();
        if self.sample_weight_counts.is_empty() {
            return Ok(out);
        }
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root.clone(),
            _ => return Err(anyhow!("sample-weight sidecars require local output")),
        };
        for table in &self.source.tables {
            if self.sample_weight_column_for(&table.name).is_none() {
                continue;
            }
            let offsets = self.shard_offsets.get(&table.name).ok_or_else(|| {
                anyhow!(
                    "sample-weight sidecars: missing shard offsets for table {}",
                    table.name
                )
            })?;
            let mut shards = Vec::with_capacity(self.num_shards);
            for shard in 0..self.num_shards {
                let key = (table.name.clone(), shard);
                let Some(counts) = self.sample_weight_counts.get(&key) else {
                    continue;
                };
                let spill_path =
                    root.join(crate::sample_weight::spill_rel_path(&table.name, shard));
                if !spill_path.exists() {
                    return Err(anyhow!(
                        "missing sample-weight spill for table {} shard {} at {}",
                        table.name,
                        shard,
                        spill_path.display()
                    ));
                }
                let label_spill_path =
                    root.join(crate::binary_label::spill_rel_path(&table.name, shard));
                if self.binary_label_counts.contains_key(&key) && !label_spill_path.exists() {
                    return Err(anyhow!(
                        "missing binary-label spill for sample-weight table {} shard {} at {}",
                        table.name,
                        shard,
                        label_spill_path.display()
                    ));
                }
                shards.push(crate::sample_weight::ShardSpill {
                    min_id: offsets.get(shard).copied().ok_or_else(|| {
                        anyhow!(
                            "sample-weight sidecars: missing shard offset for table {} shard {}",
                            table.name,
                            shard
                        )
                    })?,
                    counts: counts.clone(),
                    spill_path,
                    label_spill_path: label_spill_path.exists().then_some(label_spill_path),
                    label_counts: self.binary_label_counts.get(&key).cloned(),
                });
            }
            if shards.is_empty() {
                continue;
            }
            let [(all_ids_rel, all_cdf_rel), (false_ids_rel, false_cdf_rel), (true_ids_rel, true_cdf_rel)] =
                crate::sample_weight::output_rel_paths(&table.name);
            if let Some(index) = crate::sample_weight::merge_stage(
                &shards,
                &root.join(&all_ids_rel),
                all_ids_rel,
                &root.join(&all_cdf_rel),
                all_cdf_rel,
                &root.join(&false_ids_rel),
                false_ids_rel,
                &root.join(&false_cdf_rel),
                false_cdf_rel,
                &root.join(&true_ids_rel),
                true_ids_rel,
                &root.join(&true_cdf_rel),
                true_cdf_rel,
            )? {
                out.insert(table.name.clone(), index);
            }
        }
        Ok(out)
    }

    fn cleanup_temp(&self) -> Result<()> {
        if let OutputTarget::Local { root, .. } = &*self.target {
            let temp_root = root.join(TEMP_DIR_NAME);
            if temp_root.exists() {
                // Synchronous parallel delete: collect top-level subdirs and
                // delete them in parallel threads, then remove the (now empty)
                // temp root.  This avoids the race condition where a background
                // delete process from a previous attempt interferes with a
                // Temporal retry's fresh writes.
                let entries: Vec<_> = std::fs::read_dir(&temp_root)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .collect();
                if !entries.is_empty() {
                    let start = std::time::Instant::now();
                    std::thread::scope(|s| {
                        for p in &entries {
                            let p = p.clone();
                            s.spawn(move || {
                                if let Err(e) = remove_temp_entry(&p) {
                                    tracing::warn!(
                                        "cleanup_temp: failed to delete {}: {}",
                                        p.display(),
                                        e
                                    );
                                }
                            });
                        }
                    });
                    tracing::info!(
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        entries = entries.len(),
                        "cleanup_temp: parallel delete complete",
                    );
                }
                // Remove the now-empty temp root
                let _ = std::fs::remove_dir(&temp_root);
            }
        }
        Ok(())
    }

    fn cleanup_edge_stage(&self, dir: &str) -> Result<()> {
        if let OutputTarget::Local { root, .. } = &*self.target {
            let stage_root = root.join(TEMP_DIR_NAME).join(dir);
            if stage_root.exists() {
                std::fs::remove_dir_all(&stage_root)
                    .with_context(|| format!("cleanup_edge_stage: {}", stage_root.display()))?;
            }
        }
        Ok(())
    }

    fn record_index_input_bytes(&mut self) {
        let OutputTarget::Local { root, .. } = &*self.target else {
            return;
        };
        let nodes_root = root.join(TEMP_DIR_NAME).join("nodes");
        if nodes_root.exists() {
            self.temp_node_data_bytes = self
                .temp_node_data_bytes
                .max(sum_dir_bytes(&nodes_root).unwrap_or(0));
        }
        let edges_root = root.join(TEMP_DIR_NAME).join("edges_map");
        if edges_root.exists() {
            self.temp_edges_map_bytes = self
                .temp_edges_map_bytes
                .max(sum_dir_bytes(&edges_root).unwrap_or(0));
        }
    }

    fn record_index_input_bytes_if_missing(&mut self) {
        if self.temp_node_data_bytes == 0 || self.temp_edges_map_bytes == 0 {
            self.record_index_input_bytes();
        }
    }

    fn record_edge_stage_bytes(&mut self, dir: &str) -> Result<u64> {
        if let OutputTarget::Local { root, .. } = &*self.target {
            let stage_root = root.join(TEMP_DIR_NAME).join(dir);
            if stage_root.exists() {
                let bytes = sum_dir_bytes(&stage_root).unwrap_or(0);
                return Ok(bytes);
            }
        }
        Ok(0)
    }

    /// Walk `_tmp/fkeys/**/bitmap.bin` and apply `f` to each path found.
    /// Helper for the pk_map pristine-COW pre/post hooks.
    fn for_each_fkey_bitmap<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&Path) -> Result<()>,
    {
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root.clone(),
            _ => return Ok(()),
        };
        let fkeys_root = root.join(TEMP_DIR_NAME).join("fkeys");
        if !fkeys_root.exists() {
            return Ok(());
        }
        // _tmp/fkeys/<src_table>/<rel>/shard-NNNNN/bitmap.bin
        for src_entry in fs::read_dir(&fkeys_root)? {
            let src = src_entry?.path();
            if !src.is_dir() {
                continue;
            }
            for rel_entry in fs::read_dir(&src)? {
                let rel = rel_entry?.path();
                if !rel.is_dir() {
                    continue;
                }
                for shard_entry in fs::read_dir(&rel)? {
                    let shard = shard_entry?.path();
                    if !shard.is_dir() {
                        continue;
                    }
                    let bitmap = shard.join("bitmap.bin");
                    if bitmap.is_file() {
                        f(&bitmap)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Pre-pk_map hook: phase-pristine COW for fkey bitmaps.
    ///
    /// pk_map mutates `_tmp/fkeys/.../bitmap.bin` in place during the phase
    /// (clearing bits for null timestamps and duplicate PKs). Those bitmap
    /// files are encode's outputs; pk_map mutating them violates the phase
    /// atomicity invariant ("a phase writes only to its own outputs").
    ///
    /// Fix: at pk_map start, snapshot each `bitmap.bin` to `bitmap.bin.pristine`
    /// (or restore the canonical from a pristine that already exists, which
    /// signals we're resuming after a previous pk_map crash). pk_map_post
    /// deletes the pristine on success so it cannot poison a future re-encode.
    fn pk_map_pre(&self) -> Result<()> {
        if !matches!(&*self.target, OutputTarget::Local { .. }) {
            return Ok(());
        }
        let mut snapshot_count = 0u64;
        let mut restore_count = 0u64;
        self.for_each_fkey_bitmap(|bitmap| {
            let pristine = bitmap.with_extension("bin.pristine");
            if !pristine.exists() {
                fs::copy(bitmap, &pristine)
                    .with_context(|| format!("pk_map_pre: snapshot {}", bitmap.display()))?;
                snapshot_count += 1;
            } else {
                fs::copy(&pristine, bitmap)
                    .with_context(|| format!("pk_map_pre: restore {}", bitmap.display()))?;
                restore_count += 1;
            }
            Ok(())
        })?;
        tracing::info!(
            stage = "pk_map",
            snapshot_count,
            restore_count,
            "pk_map_pre: bitmap pristine COW prepared"
        );
        Ok(())
    }

    /// Post-pk_map hook: drop pristines now that bitmap.bin is the canonical
    /// post-pk_map state. This must run only after the pk_map marker is
    /// written so a crash here does not leave the pipeline in a state where
    /// a future resume of pk_map cannot restore.
    fn pk_map_post(&self) -> Result<()> {
        if !matches!(&*self.target, OutputTarget::Local { .. }) {
            return Ok(());
        }
        let mut removed = 0u64;
        self.for_each_fkey_bitmap(|bitmap| {
            let pristine = bitmap.with_extension("bin.pristine");
            if pristine.exists() {
                fs::remove_file(&pristine)
                    .with_context(|| format!("pk_map_post: remove {}", pristine.display()))?;
                removed += 1;
            }
            Ok(())
        })?;
        tracing::info!(
            stage = "pk_map",
            removed_pristines = removed,
            "pk_map_post: dropped bitmap pristines"
        );
        Ok(())
    }

    /// Post-pk_map cleanup: free encode's `pkey.bin`/`pkey.meta` files.
    /// pk_map is the only consumer of those files; once pk_map's marker is
    /// written, holding them through the rest of the pipeline costs ~160 GB
    /// for DoorDash-shaped data and gains nothing.
    /// `node_data.bin` stays — `build_index_metadata` reads it.
    fn cleanup_pkey_temp_files(&self) -> Result<u64> {
        let root = match &*self.target {
            OutputTarget::Local { root, .. } => root.clone(),
            _ => return Ok(0),
        };
        let nodes_root = root.join(TEMP_DIR_NAME).join("nodes");
        if !nodes_root.exists() {
            return Ok(0);
        }
        let mut removed_bytes = 0u64;
        for table_entry in fs::read_dir(&nodes_root)? {
            let table_dir = table_entry?.path();
            if !table_dir.is_dir() {
                continue;
            }
            for shard_entry in fs::read_dir(&table_dir)? {
                let shard_dir = shard_entry?.path();
                if !shard_dir.is_dir() {
                    continue;
                }
                for name in &["pkey.bin", "pkey.meta"] {
                    let p = shard_dir.join(name);
                    if p.is_file() {
                        if let Ok(meta) = p.metadata() {
                            removed_bytes = removed_bytes.saturating_add(meta.len());
                        }
                        let _ = fs::remove_file(&p);
                    }
                }
            }
        }
        Ok(removed_bytes)
    }

    fn build_index(&mut self, metadata: DatasetMetadata) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        // Coordinated layout fix (write side): if this is a sub-index ingest
        // (parent_index_path was set), thread the parent's node_types into
        // the builder so cross-boundary FK edges get encoded with the real
        // parent `max_id` instead of the `dst_bits=64` fallback. Persisted
        // metadata stays clean — externals only live in the in-memory build
        // scope. See `GraphIndexBuilder::with_external_node_types`.
        let external_node_types = self
            .parent_index_path
            .as_ref()
            .map(|p| -> Result<Vec<_>> {
                let parent_meta_path = p.join("index_metadata.json");
                let bytes = fs::read(&parent_meta_path).with_context(|| {
                    format!(
                        "coordinated layout fix: read parent metadata {}",
                        parent_meta_path.display()
                    )
                })?;
                let parent_meta: DatasetMetadata =
                    serde_json::from_slice(&bytes).with_context(|| {
                        format!(
                            "coordinated layout fix: parse parent metadata {}",
                            parent_meta_path.display()
                        )
                    })?;
                Ok(parent_meta.node_types)
            })
            .transpose()?
            .unwrap_or_default();
        let builder = GraphIndexBuilder::from_metadata(
            metadata,
            GraphDataSource::Local { root: root.clone() },
            &root,
        )
        .with_external_node_types(external_node_types);
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
        let prepared = builder.prepare_resumable_jobs_with_pool(edge_buffer_pool)?;
        let adapter = self
            .progress
            .clone()
            .map(|cb| IndexProgressAdapter::new(cb, table_totals.clone()));
        let adapter = adapter.map(std::sync::Arc::new);
        if let Some(progress) = adapter.as_ref() {
            for resumed in prepared.resumed_shards {
                progress.report(GraphBuildProgress {
                    stage: GraphBuildStage::BuildShard {
                        shard: std::sync::Arc::clone(&resumed.shard_key),
                    },
                    current: resumed.expected_nodes,
                    total: Some(resumed.expected_nodes),
                });
                progress.report(GraphBuildProgress {
                    stage: GraphBuildStage::Complete {
                        shard: resumed.shard_key,
                    },
                    current: 0,
                    total: None,
                });
            }
        }
        let mut tasks = Vec::with_capacity(prepared.jobs.len());
        for job in prepared.jobs {
            let adapter = adapter.clone();
            let mem_budget = mem_budget.clone();
            tasks.push(WorkItem {
                bucket: None,
                run: Box::new(move |_scratch| {
                    let progress = adapter
                        .as_ref()
                        .map(|value| value.as_ref() as &dyn GraphProgressSink);

                    job.run_with_budget(progress, mem_budget)?;
                    let reclaimed = job.cleanup_consumed_inputs()?;
                    if reclaimed > 0 {
                        tracing::info!(
                            shard = job.shard_key(),
                            reclaimed_bytes = reclaimed,
                            "index shard complete: removed consumed scratch"
                        );
                    }
                    Ok(())
                }),
            });
        }
        let pool = self.make_pool::<()>(self.index_threads);
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
                create_time_col: table.create_time.clone(),
                end_time_col: table.end_time.clone(),
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
                reverse_of: rel.reverse_of.clone(),
            });
        }

        let index_metadata = DatasetMetadata {
            timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
            dataset_name: self.source.dataset_name.clone(),
            node_types,
            edge_relations,
        };
        Ok(index_metadata)
    }

    fn finalize_metadata(
        &self,
        mut metadata: DatasetMetadata,
    ) -> Result<(DatasetMetadata, IngestSourceConfig)> {
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

        // Persist the resolved graph config so downstream consumers (TT-gen,
        // BP) can recover per-column semantic_type without re-supplying it
        // via plan JSON each query (kumo#28400 follow-up). `self.source` has
        // semantic_type populated for every column by the context-build
        // stage (see populate_table_contexts at lib.rs:1998).
        let config_json = serde_json::to_vec_pretty(&self.source)?;
        let config_sha256 = format!("{:x}", Sha256::digest(&config_json));
        self.target
            .write_bytes(graph_config::DISKGRAPH_CONFIG_FILENAME, config_json)?;
        Ok((
            metadata,
            IngestSourceConfig {
                path: graph_config::DISKGRAPH_CONFIG_FILENAME.to_string(),
                sha256: config_sha256,
                representative_data_path: stats::representative_data_path(&self.source),
            },
        ))
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
            key_membership_failures: self.key_membership_failures,
            key_failure_samples: self.key_failure_samples.clone(),
            key_failures_by_context: self.key_failures_by_context.clone(),
            pk_duplicate_counts: self.pk_duplicate_counts.clone(),
            pk_duplicate_drops: self.pk_duplicate_drops.clone(),
            pk_null_timestamp_drops: self.pk_null_timestamp_drops.clone(),
            edge_null_timestamp_drops: self.edge_null_timestamp_drops.clone(),
            pk_duplicate_total,
            pk_duplicate_samples: self.pk_duplicate_samples.clone(),
            dropped_rows_counts,
            source_rows_by_table: self.logical_source_rows_by_table.clone(),
            source_bytes_by_table: self.logical_arrow_source_bytes_by_table.clone(),
            trimmed_strings: if trimmed_strings > 0 {
                Some(trimmed_strings)
            } else {
                None
            },
            perf,
            num_shards: Some(self.num_shards),
            // Populated by the main ingest flow after post-finalize sidecars.
            ts_groupings: HashMap::new(),
            binary_label_indexes: HashMap::new(),
            sample_weight_indexes: HashMap::new(),
        }
    }

    /// Build kumo-format edge_stats.json content.
    ///
    /// Returns `None` if kumo metadata (connector_id, source_table_name) is not
    /// available on the tables — i.e., the config was not produced by
    /// `convert_kumo_to_diskgraph` and we're running outside the kumo pipeline.
    fn build_edge_stats_json(
        &self,
        metadata: &diskgraph_core::metadata::DatasetMetadata,
    ) -> Result<Option<Vec<u8>>> {
        use serde_json::{json, Map, Value};

        // Build table name → TableSpec lookup
        let table_map: HashMap<&str, &crate::graph_config::TableSpec> = self
            .source
            .tables
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        // Build table name → node count from metadata
        let mut node_counts: HashMap<&str, u64> = HashMap::new();
        for nt in &metadata.node_types {
            node_counts.insert(
                // Find matching table name (metadata node type name matches table name)
                table_map
                    .keys()
                    .find(|k| **k == nt.name)
                    .copied()
                    .unwrap_or(nt.name.as_str()),
                nt.max_id.saturating_add(1),
            );
        }

        // Check that kumo metadata is available on at least one table
        let has_kumo_metadata = self
            .source
            .tables
            .iter()
            .any(|t| t.kumo_connector_id.is_some());
        if !has_kumo_metadata {
            return Ok(None);
        }

        let mut stats_map = Map::new();

        for rel in &self.relationships {
            let src_table = match table_map.get(rel.src_table.as_str()) {
                Some(t) => t,
                None => continue,
            };
            let dst_table = match table_map.get(rel.dst_table.as_str()) {
                Some(t) => t,
                None => continue,
            };

            let src_connector = src_table.kumo_connector_id.as_deref().unwrap_or("");
            let src_source_name = src_table
                .kumo_source_table_name
                .as_deref()
                .unwrap_or(&src_table.name);
            let dst_connector = dst_table.kumo_connector_id.as_deref().unwrap_or("");
            let dst_source_name = dst_table
                .kumo_source_table_name
                .as_deref()
                .unwrap_or(&dst_table.name);
            let fk_col = &rel.src_column;
            let pk_col = match &dst_table.primary_key {
                Some(pk) => pk.as_str(),
                None => continue,
            };
            let data_version = src_table.data_version.as_deref().unwrap_or("");

            let edge_count = self
                .edge_counts
                .get(&rel.name_forward)
                .copied()
                .unwrap_or(0);
            let num_src_nodes = node_counts
                .get(rel.src_table.as_str())
                .copied()
                .unwrap_or(0);
            let num_dst_nodes = node_counts
                .get(rel.dst_table.as_str())
                .copied()
                .unwrap_or(0);

            // For FK→PK, forward_count = unique src nodes (each src row → 0 or 1 edge)
            let num_src_nonzero = edge_count;

            // Unique destinations reduced from the EdgesMap bitvec before the
            // phase snapshot is serialized.
            let num_dst_nonzero = self
                .edge_dst_nonzero_counts
                .get(&rel.name_forward)
                .copied()
                .unwrap_or(0);

            let frac_src = if num_src_nodes > 0 {
                num_src_nonzero as f64 / num_src_nodes as f64
            } else {
                0.0
            };
            let frac_dst = if num_dst_nodes > 0 {
                num_dst_nonzero as f64 / num_dst_nodes as f64
            } else {
                0.0
            };

            // Build key prefix: src_table__src_connector__dst_table__dst_connector__fk__pk
            let key_prefix = format!(
                "{}__{}__{}__{}__{}__{}",
                src_source_name, src_connector, dst_source_name, dst_connector, fk_col, pk_col,
            );

            let stat_value = |val: Value| -> Value {
                json!({
                    "value": val,
                    "approx": false,
                    "data_version": data_version,
                    "num_entries": null,
                    "timestamp": null,
                })
            };

            stats_map.insert(
                format!("{}__EDGE_COUNT", key_prefix),
                stat_value(json!(edge_count)),
            );
            stats_map.insert(
                format!("{}__NUM_SRC_NODES", key_prefix),
                stat_value(json!(num_src_nodes)),
            );
            stats_map.insert(
                format!("{}__NUM_DST_NODES", key_prefix),
                stat_value(json!(num_dst_nodes)),
            );
            stats_map.insert(
                format!("{}__NUM_SRC_NONZERO_DEGREE", key_prefix),
                stat_value(json!(num_src_nonzero)),
            );
            stats_map.insert(
                format!("{}__NUM_DST_NONZERO_DEGREE", key_prefix),
                stat_value(json!(num_dst_nonzero)),
            );
            stats_map.insert(
                format!("{}__FRAC_SRC_NONZERO_DEGREE", key_prefix),
                stat_value(json!(frac_src)),
            );
            stats_map.insert(
                format!("{}__FRAC_DST_NONZERO_DEGREE", key_prefix),
                stat_value(json!(frac_dst)),
            );
        }

        if stats_map.is_empty() {
            return Ok(None);
        }

        let json_bytes = serde_json::to_vec_pretty(&Value::Object(stats_map))?;
        Ok(Some(json_bytes))
    }
}

/// Compute text sample stats during the context phase.
/// Reads up to TEXT_SAMPLE_LIMIT rows from the first parquet file(s) for text columns.
const TEXT_SAMPLE_LIMIT: usize = 10_000;
const SOURCE_PREFIX_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Copy a bounded row prefix into compact Arrow buffers. `RecordBatch::slice`
/// alone retains the complete source batch and defeats Encode-phase memory
/// bounds when the sampled prefix is small.
fn compact_record_batch_prefix(batch: &RecordBatch, rows: usize) -> Result<RecordBatch> {
    if rows > batch.num_rows() {
        return Err(anyhow!("source prefix row count exceeds its batch"));
    }
    let arrays = batch
        .columns()
        .iter()
        .map(|array| {
            let data = array.to_data();
            let mut compact = MutableArrayData::new(vec![&data], false, rows);
            compact.extend(0, 0, rows);
            make_array(compact.freeze())
        })
        .collect();
    Ok(RecordBatch::try_new(batch.schema(), arrays)?)
}

/// The failure when the context prefix cannot take one more row.
///
/// The condition is a budget already spent by earlier batches, not an
/// oversized row, and the message says so: a row that genuinely exceeded
/// `SOURCE_PREFIX_MAX_BYTES` would be reported by the same numbers, and every
/// other case names the real remaining budget so the reader is not sent
/// looking for a pathological row that does not exist.
fn context_prefix_budget_error(
    table: &str,
    remaining_bytes: usize,
    row_bytes: usize,
    sampled_rows: usize,
    sampled_bytes: usize,
) -> anyhow::Error {
    anyhow!(
        "table {table}: the context prefix budget has {remaining_bytes} bytes left, which cannot \
         hold one more source row of {row_bytes} bytes; {sampled_rows} rows totalling \
         {sampled_bytes} bytes were already sampled against the {SOURCE_PREFIX_MAX_BYTES} byte \
         budget"
    )
}

/// The cost of the prefix `compact_record_batch_prefix` would produce, measured
/// by producing it.
///
/// The obvious cheaper measurement is wrong for nested data.
/// `ArrayData::slice` is a zero-copy view, and it propagates the new offset
/// into child data only for `Struct`; every other type keeps its `child_data`
/// whole. `get_slice_memory_size` then recurses into that unsliced child, so a
/// `ListArray` prefix is charged for the entire child buffer no matter how few
/// rows are taken. The measurement carries a constant term that no row count
/// can trim, which is the opposite of what a row-count search needs.
/// `compact_record_batch_prefix` copies through `MutableArrayData`, which does
/// narrow children to the rows taken, so measuring with it cannot disagree
/// with what is finally kept.
fn compact_prefix_bytes(batch: &RecordBatch, rows: usize) -> Result<usize> {
    Ok(compact_record_batch_prefix(batch, rows)?.get_array_memory_size())
}

fn compact_prefix_with_budget(
    batch: &RecordBatch,
    max_rows: usize,
    max_bytes: usize,
) -> Result<Option<RecordBatch>> {
    let max_rows = max_rows.min(batch.num_rows());
    if max_rows == 0 || max_bytes == 0 {
        return Ok(None);
    }
    // Grow the candidate from below instead of bisecting the whole range from
    // its midpoint. Every probe now materialises the prefix it measures, and a
    // midpoint probe would allocate far past the budget this call was given a
    // permit for. If a probe exceeds the budget, doubling makes it the smallest
    // rejected row count, no larger than the source batch already resident in
    // memory, and bisection then stays below that count.
    //
    // The result fits the budget but is not necessarily the largest row count
    // that would: `MutableArrayData` sizes by capacity and capacity doubles, so
    // prefix cost is not monotone in row count and a bisection can settle below
    // a larger prefix that also fits. The bound that holds is that the returned
    // prefix fits; sampling reads it as a statistics sample, which does not
    // require maximality.
    let mut largest_fitting = None;
    let mut smallest_rejected = None;
    let mut rows = 1usize;
    loop {
        if compact_prefix_bytes(batch, rows)? <= max_bytes {
            largest_fitting = Some(rows);
            if rows == max_rows {
                break;
            }
            rows = rows.saturating_mul(2).min(max_rows);
        } else {
            smallest_rejected = Some(rows);
            break;
        }
    }
    if let (Some(fitting), Some(rejected)) = (largest_fitting, smallest_rejected) {
        let mut fitting = fitting;
        let mut rejected = rejected;
        while rejected - fitting > 1 {
            let candidate = fitting + (rejected - fitting) / 2;
            if compact_prefix_bytes(batch, candidate)? <= max_bytes {
                fitting = candidate;
            } else {
                rejected = candidate;
            }
        }
        largest_fitting = Some(fitting);
    }
    let Some(rows) = largest_fitting else {
        return Ok(None);
    };
    // No re-check against `max_bytes` here: the search accepted `rows` using
    // this very function, so the result is the measurement.
    Ok(Some(compact_record_batch_prefix(batch, rows)?))
}

#[cfg(test)]
mod context_prefix_tests {
    use super::*;
    use arrow_array::builder::{Float64Builder, ListBuilder, StringBuilder};
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};

    /// A feature table with embedding columns: the shape that exposed this.
    /// Tubi's `programs` is 3,401 rows with two 1,020-wide `float64` lists.
    fn embedding_batch(rows: usize, width: usize) -> RecordBatch {
        let mut embedding = ListBuilder::new(Float64Builder::new());
        for row in 0..rows {
            for element in 0..width {
                embedding.values().append_value((row + element) as f64);
            }
            embedding.append(true);
        }
        let embedding = embedding.finish();
        let mut label = StringBuilder::new();
        for row in 0..rows {
            label.append_value(format!("label-{row}"));
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
            Field::new("embedding", embedding.data_type().clone(), false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..rows as i64)),
                Arc::new(label.finish()),
                Arc::new(embedding),
            ],
        )
        .unwrap()
    }

    fn flat_batch(rows: usize) -> RecordBatch {
        let mut label = StringBuilder::new();
        for row in 0..rows {
            label.append_value(format!("label-{row}"));
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..rows as i64)),
                Arc::new(label.finish()),
            ],
        )
        .unwrap()
    }

    fn sliced_measure(batch: &RecordBatch, rows: usize) -> usize {
        batch
            .columns()
            .iter()
            .map(|array| {
                array
                    .slice(0, rows)
                    .to_data()
                    .get_slice_memory_size()
                    .unwrap()
            })
            .sum()
    }

    /// The defect, stated as an invariant of the two measurements. Slicing a
    /// `ListArray` does not narrow its child, so the old measurement is almost
    /// flat in the row count while the real cost is linear. Any search that
    /// trusts the first cannot converge on the second.
    #[test]
    fn slicing_a_list_column_does_not_shrink_its_measured_size() {
        let batch = embedding_batch(2_000, 256);
        let sliced_one = sliced_measure(&batch, 1);
        let sliced_all = sliced_measure(&batch, batch.num_rows());
        let compact_one = compact_prefix_bytes(&batch, 1).unwrap();
        let compact_all = compact_prefix_bytes(&batch, batch.num_rows()).unwrap();

        // One row measures within a few percent of the whole batch under the
        // old measurement: the child buffer is carried whole either way.
        assert!(
            sliced_one * 100 > sliced_all * 95,
            "slice measure for 1 row is {sliced_one}, for {} rows is {sliced_all}",
            batch.num_rows()
        );
        // Measured by the function that materialises it, one row is three
        // orders of magnitude smaller than the batch.
        assert!(
            compact_one * 1_000 < compact_all,
            "compact measure for 1 row is {compact_one}, for {} rows is {compact_all}",
            batch.num_rows()
        );
    }

    /// The regression: a budget that comfortably holds hundreds of rows but not
    /// the whole batch. The old search measured every candidate at roughly the
    /// whole child buffer, found nothing that fit, and the caller aborted the
    /// ingest.
    #[test]
    fn a_partially_spent_budget_still_admits_rows_of_a_list_column() {
        let batch = embedding_batch(2_000, 256);
        let row_bytes = compact_prefix_bytes(&batch, 1).unwrap();
        let budget = row_bytes * 500;
        // Precondition: this is exactly the situation that used to fail.
        assert!(
            sliced_measure(&batch, 1) > budget,
            "the old measurement must reject even one row for this to be the regression"
        );

        let compact = compact_prefix_with_budget(&batch, SAMPLE_ROW_LIMIT, budget)
            .unwrap()
            .expect("a budget worth ~500 rows must admit rows, not abort the ingest");
        assert!(compact.num_rows() > 1);
        assert!(compact.get_array_memory_size() <= budget);
    }

    /// Replay the context-prefix loop across source parts. The first part uses
    /// most of the byte budget; the remainder is smaller than the second
    /// part's retained list child but large enough for a materialised prefix.
    /// Sampling must advance into part two instead of aborting the table.
    #[test]
    fn a_partially_spent_budget_advances_into_the_next_list_part() {
        let first_part = embedding_batch(400, 256);
        let second_part = embedding_batch(200, 256);
        let first_part_bytes = compact_prefix_bytes(&first_part, first_part.num_rows()).unwrap();
        let second_prefix_rows = 50;
        let second_prefix_bytes = compact_prefix_bytes(&second_part, second_prefix_rows).unwrap();
        let total_budget = first_part_bytes + second_prefix_bytes;
        let old_one_row_measure = sliced_measure(&second_part, 1);

        assert!(
            old_one_row_measure > second_prefix_bytes,
            "the old measurement must reject part two for this to cover the regression"
        );

        let mut sampled_rows = 0;
        let mut sampled_bytes = 0;
        for part in [&first_part, &second_part] {
            let remaining = total_budget - sampled_bytes;
            let compact =
                compact_prefix_with_budget(part, SAMPLE_ROW_LIMIT - sampled_rows, remaining)
                    .unwrap()
                    .expect("the remaining budget must admit a materialised list prefix");
            sampled_rows += compact.num_rows();
            sampled_bytes += compact.get_array_memory_size();
            if compact.num_rows() < part.num_rows() {
                break;
            }
        }

        assert_eq!(sampled_rows, first_part.num_rows() + second_prefix_rows);
        assert_eq!(sampled_bytes, total_budget);
    }

    /// The admitted prefix fits the budget. That is the property the search
    /// guarantees for every input.
    ///
    /// Maximality is not guaranteed and is deliberately not asserted here:
    /// prefix cost is not monotone in row count, because `MutableArrayData`
    /// sizes by capacity and capacity doubles, so a bisection can settle below
    /// a larger prefix that also fits. On this fixture the result does happen
    /// to be maximal, which
    /// `the_next_row_does_not_fit_on_an_evenly_sized_fixture` records as an
    /// observation about the fixture rather than as a general claim.
    #[test]
    fn the_admitted_prefix_fits_the_budget() {
        let batch = embedding_batch(2_000, 256);
        let budget = compact_prefix_bytes(&batch, 1).unwrap() * 500;
        let compact = compact_prefix_with_budget(&batch, SAMPLE_ROW_LIMIT, budget)
            .unwrap()
            .unwrap();
        let rows = compact.num_rows();
        assert!(rows > 1);
        assert!(compact_prefix_bytes(&batch, rows).unwrap() <= budget);
        assert!(compact.get_array_memory_size() <= budget);
    }

    /// Records that on a fixture whose rows are all the same size the search
    /// does reach the largest fitting count. This is a statement about this
    /// fixture, not about the search: see
    /// `the_admitted_prefix_fits_the_budget` for why maximality is not general.
    #[test]
    fn the_next_row_does_not_fit_on_an_evenly_sized_fixture() {
        let batch = embedding_batch(2_000, 256);
        let budget = compact_prefix_bytes(&batch, 1).unwrap() * 500;
        let compact = compact_prefix_with_budget(&batch, SAMPLE_ROW_LIMIT, budget)
            .unwrap()
            .unwrap();
        let rows = compact.num_rows();
        if rows < batch.num_rows() {
            assert!(
                compact_prefix_bytes(&batch, rows + 1).unwrap() > budget,
                "{rows} rows fit but the search stopped short of the budget"
            );
        }
    }

    /// A budget that genuinely cannot hold one row still refuses, rather than
    /// admitting a prefix over budget. That case is real and must stay an
    /// error; only its explanation changes.
    #[test]
    fn a_budget_below_one_row_still_refuses() {
        let batch = embedding_batch(64, 256);
        let row_bytes = compact_prefix_bytes(&batch, 1).unwrap();
        let half_a_row = compact_prefix_with_budget(&batch, SAMPLE_ROW_LIMIT, row_bytes / 2);
        assert!(half_a_row.unwrap().is_none());
        let no_budget = compact_prefix_with_budget(&batch, SAMPLE_ROW_LIMIT, 0);
        assert!(no_budget.unwrap().is_none());
        let no_rows = compact_prefix_with_budget(&batch, 0, usize::MAX);
        assert!(no_rows.unwrap().is_none());
    }

    /// Batches with no nested columns were already measured correctly. The new
    /// search must return the same answer, so this is not a behaviour change
    /// for the common case.
    #[test]
    fn flat_batches_keep_their_previous_row_counts() {
        let batch = flat_batch(5_000);
        for budget in [1_024usize, 16_384, 1 << 20, SOURCE_PREFIX_MAX_BYTES] {
            let old_rows = {
                let (mut low, mut high, mut best) = (1usize, batch.num_rows(), None);
                while low <= high {
                    let rows = low + (high - low) / 2;
                    if sliced_measure(&batch, rows) <= budget {
                        best = Some(rows);
                        low = rows + 1;
                    } else {
                        high = rows - 1;
                    }
                }
                best
            };
            let new_rows = compact_prefix_with_budget(&batch, SAMPLE_ROW_LIMIT, budget)
                .unwrap()
                .map(|compact| compact.num_rows());
            // The old measurement undercounts a compacted Utf8 array's
            // allocated capacity, so it can admit one or two rows more than
            // actually fit. The new count is never larger, and never differs by
            // a factor.
            match (old_rows, new_rows) {
                (Some(old), Some(new)) => {
                    assert!(new <= old, "budget {budget}: old {old} rows, new {new}");
                    assert!(
                        new * 2 >= old,
                        "budget {budget}: old {old} rows, new {new} is not within a factor of two"
                    );
                }
                (None, None) => {}
                other => panic!("budget {budget}: disagreement on admissibility: {other:?}"),
            }
        }
    }

    /// The message must describe the condition that actually held. It reports
    /// the remaining budget and the measured per-row cost, and never claims a
    /// row is larger than the budget.
    #[test]
    fn the_budget_error_names_the_remaining_budget_not_an_oversized_row() {
        let message = context_prefix_budget_error("programs", 8_533_676, 17_228, 3_401, 58_575_188)
            .to_string();
        assert!(message.contains("programs"));
        assert!(message.contains("8533676"));
        assert!(message.contains("17228"));
        assert!(message.contains("3401"));
        assert!(message.contains("58575188"));
        assert!(
            !message.contains("larger than the context prefix budget"),
            "{message}"
        );
    }
}

fn compact_context_prefix_with_budget(
    table_name: &str,
    batch: &RecordBatch,
    max_rows: usize,
    max_bytes: usize,
    sampled_rows: usize,
    sampled_bytes: usize,
) -> Result<Option<RecordBatch>> {
    let compact = compact_prefix_with_budget(batch, max_rows, max_bytes)?;
    if compact.is_none() && sampled_rows == 0 {
        let row_bytes = compact_prefix_bytes(batch, 1)?;
        return Err(context_prefix_budget_error(
            table_name,
            max_bytes,
            row_bytes,
            sampled_rows,
            sampled_bytes,
        ));
    }
    Ok(compact)
}

#[cfg(test)]
mod source_prefix_tests {
    use super::*;
    use arrow_array::Int64Array;
    use arrow_schema::{Field, Schema};

    fn batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![1]))],
        )
        .unwrap()
    }

    #[test]
    fn residual_context_prefix_budget_defers_the_next_batch() {
        let compact =
            compact_context_prefix_with_budget("nodes", &batch(), 1, 1, 10, 100).unwrap();
        assert!(compact.is_none());
    }

    #[test]
    fn first_source_row_must_fit_the_full_context_prefix_budget() {
        let error = compact_context_prefix_with_budget("nodes", &batch(), 1, 1, 0, 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("table nodes: the context prefix budget has 1 bytes left"));
        assert!(error.contains("cannot hold one more source row"));
    }
}

fn text_stats_from_prefix(
    table: &TableSpec,
    semantic_types: &[SemanticType],
    batches: &[RecordBatch],
) -> Result<HashMap<String, crate::col_stats::TextSampleStats>> {
    let text_cols: Vec<(usize, &str)> = table
        .columns
        .iter()
        .zip(semantic_types)
        .enumerate()
        .filter(|(_, (_, semantic_type))| **semantic_type == SemanticType::Text)
        .map(|(index, (column, _))| (index, column.name.as_str()))
        .collect();
    let mut samples: Vec<Vec<String>> = vec![Vec::new(); text_cols.len()];
    for batch in batches {
        for (sample_index, &(column_index, _)) in text_cols.iter().enumerate() {
            if samples[sample_index].len() >= TEXT_SAMPLE_LIMIT {
                continue;
            }
            let source_index = batch
                .schema()
                .index_of(&table.columns[column_index].name)
                .context("missing text column in Encode prefix")?;
            let array = batch.column(source_index);
            for row in 0..batch.num_rows() {
                if samples[sample_index].len() >= TEXT_SAMPLE_LIMIT {
                    break;
                }
                if array.is_null(row) {
                    continue;
                }
                if let Some(values) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
                    samples[sample_index].push(values.value(row).to_owned());
                } else if let Some(values) = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                {
                    samples[sample_index].push(values.value(row).to_owned());
                }
            }
        }
    }
    let mut result = HashMap::new();
    for (sample_index, &(_, column_name)) in text_cols.iter().enumerate() {
        let values: Vec<&str> = samples[sample_index].iter().map(String::as_str).collect();
        if !values.is_empty() {
            result.insert(
                column_name.to_owned(),
                crate::col_stats::compute_text_sample_stats(&values),
            );
        }
    }
    Ok(result)
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
    result
}

/// Path of the marker file that proves a phase ran to completion.
/// Lives under `_tmp/` so the existing `cleanup_temp()` removes markers on
/// pipeline success without any extra logic.
fn phase_marker_path(root: &Path, name: &str) -> PathBuf {
    root.join(TEMP_DIR_NAME)
        .join(format!("_phase_complete_{name}"))
}

fn phase_marker_exists(root: &Path, name: &str) -> bool {
    phase_marker_path(root, name).exists()
}

/// Write the phase marker atomically after its payload is durable.
#[cfg(test)]
fn write_phase_marker(root: &Path, name: &str) -> Result<()> {
    write_phase_marker_payload(root, name, b"")
}

fn write_phase_marker_payload(root: &Path, name: &str, payload: &[u8]) -> Result<()> {
    let final_path = phase_marker_path(root, name);
    write_durable_file(&final_path, payload, "phase marker")
}

fn write_durable_file(final_path: &Path, payload: &[u8], description: &str) -> Result<()> {
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = final_path.with_extension("inprogress");
    let mut marker = fs::File::create(&tmp_path)
        .with_context(|| format!("create {description} tmp: {}", tmp_path.display()))?;
    marker
        .write_all(payload)
        .with_context(|| format!("write {description} tmp: {}", tmp_path.display()))?;
    marker
        .sync_all()
        .with_context(|| format!("sync {description} tmp: {}", tmp_path.display()))?;
    drop(marker);
    fs::rename(&tmp_path, final_path)
        .with_context(|| format!("rename {description}: {}", final_path.display()))?;
    if let Some(parent) = final_path.parent() {
        let _ = fs::File::open(parent).and_then(|f| f.sync_all());
    }
    Ok(())
}

/// Wipe the outputs (and stale marker) of a single phase before re-running it.
/// Called only when about to *re-run* a phase that may have partial state from
/// a prior crash — i.e. on the resume-target phase. Fresh ingests rely on the
/// top-level output_dir wipe and never call this.
///
fn wipe_phase_outputs(root: &Path, phase: &str) -> Result<()> {
    // Remove stale marker if present (we are re-running this phase).
    let _ = fs::remove_file(phase_marker_path(root, phase));

    match phase {
        // Encode owns both the context and encoded outputs. A retry starts
        // from offset zero, so no provisional context, encoded file, stats, or
        // later-phase artifact may survive.
        "encode" => {
            for relative in [TEMP_DIR_NAME, "nodes", "kumo_col_stats"] {
                let path = root.join(relative);
                if path.exists() {
                    fs::remove_dir_all(&path)
                        .with_context(|| format!("wipe encode output: {}", path.display()))?;
                }
            }
            for relative in [
                "col_stats.json",
                crate::col_stats_export::EDGE_STATS_FILENAME,
                "ingest_stats.json",
                "index_metadata.json",
                "diskgraph_config.json",
                crate::snapshot_mirror::MANIFEST_FILENAME,
                crate::success_marker::SUCCESS_MARKER_FILENAME,
            ] {
                let path = root.join(relative);
                if path.exists() {
                    fs::remove_file(&path)
                        .with_context(|| format!("wipe encode output: {}", path.display()))?;
                }
            }
            fs::create_dir_all(root.join(TEMP_DIR_NAME))?;
            Ok(())
        }

        // pk_map writes pk_hash-NNNNN.bin / pk_data-NNNNN.bin /
        // pk_hash-NNNNN.meta.json into nodes/<table>/. Preserve siblings
        // (node_data.ctx from context, shard-NNNNN/ from index).
        "pk_map" => {
            for relative in [PK_MAP_STATE_PATH, EDGES_MAP_STATE_PATH] {
                let state = root.join(relative);
                if state.exists() {
                    fs::remove_file(&state)
                        .with_context(|| format!("remove {}", state.display()))?;
                }
            }
            let nodes = root.join("nodes");
            if nodes.is_dir() {
                for entry in
                    fs::read_dir(&nodes).with_context(|| format!("read_dir {}", nodes.display()))?
                {
                    let entry = entry?;
                    let table_dir = entry.path();
                    if !entry.file_type()?.is_dir() {
                        continue;
                    }
                    wipe_files_with_prefixes(&table_dir, &["pk_hash-", "pk_data-"])?;
                }
            }
            Ok(())
        }

        "edges_map" => {
            let state = root.join(EDGES_MAP_STATE_PATH);
            if state.exists() {
                fs::remove_file(&state).with_context(|| format!("remove {}", state.display()))?;
            }
            let p = root.join("_tmp/edges_map");
            if p.exists() {
                fs::remove_dir_all(&p)
                    .with_context(|| format!("wipe edges_map output: {}", p.display()))?;
            }
            Ok(())
        }

        // index writes index_metadata.json at the root and
        // offsets.bin / edges.bin / build_stats.json into each
        // nodes/<table>/shard-NNNNN/. Preserve everything else.
        "index" => {
            let meta = root.join("index_metadata.json");
            if meta.exists() {
                fs::remove_file(&meta).with_context(|| format!("remove {}", meta.display()))?;
            }
            let nodes = root.join("nodes");
            if nodes.is_dir() {
                for table_entry in
                    fs::read_dir(&nodes).with_context(|| format!("read_dir {}", nodes.display()))?
                {
                    let table_entry = table_entry?;
                    if !table_entry.file_type()?.is_dir() {
                        continue;
                    }
                    let table_dir = table_entry.path();
                    for shard_entry in fs::read_dir(&table_dir)
                        .with_context(|| format!("read_dir {}", table_dir.display()))?
                    {
                        let shard_entry = shard_entry?;
                        let shard_dir = shard_entry.path();
                        if !shard_entry.file_type()?.is_dir() {
                            continue;
                        }
                        let name = match shard_dir.file_name().and_then(|n| n.to_str()) {
                            Some(n) => n.to_owned(),
                            None => continue,
                        };
                        if !name.starts_with("shard-") {
                            continue;
                        }
                        let table_name = match table_dir.file_name().and_then(|n| n.to_str()) {
                            Some(name) => name,
                            None => continue,
                        };
                        let commit_path = index_shard_commit_path(root, table_name, name.as_str());
                        if shard_output_is_committed(&shard_dir, &commit_path, None, None)? {
                            continue;
                        }
                        for f in ["offsets.bin", "edges.bin", "build_stats.json"] {
                            let p = shard_dir.join(f);
                            if p.exists() {
                                fs::remove_file(&p)
                                    .with_context(|| format!("remove {}", p.display()))?;
                            }
                        }
                        let stale_commit_tmp = commit_path.with_extension("json.tmp");
                        if stale_commit_tmp.exists() {
                            fs::remove_file(&stale_commit_tmp).with_context(|| {
                                format!("remove {}", stale_commit_tmp.display())
                            })?;
                        }
                    }
                }
            }
            Ok(())
        }

        _ => Ok(()),
    }
}

/// Remove every provisional artifact owned by an incomplete Encode phase.
/// The supervisor uses this after its final failed Encode attempt so a large
/// partial ingest does not remain stranded without a resumable phase marker.
pub fn discard_provisional_encode_outputs(root: &Path) -> Result<()> {
    wipe_phase_outputs(root, "encode")
}

/// Remove regular files in `dir` whose names start with any of the given
/// prefixes. Non-recursive; subdirs and non-matching files are left intact.
fn wipe_files_with_prefixes(dir: &Path, prefixes: &[&str]) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if !ft.is_file() {
            continue;
        }
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if prefixes.iter().any(|p| name.starts_with(p)) {
            let path = entry.path();
            fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        }
    }
    Ok(())
}

/// Run a single pipeline phase with marker semantics.
///
/// `wipe_on_entry` should be true only when re-running a phase that may
/// have partial output from a prior crash (i.e. the resume-target phase).
/// Fresh runs pass false because the top-level output_dir wipe already
/// produced a clean slate.
fn run_phase_with_marker(
    root: &Path,
    name: &str,
    wipe_on_entry: bool,
    profile_dir: Option<&Path>,
    f: impl FnOnce() -> Result<()>,
) -> Result<()> {
    run_phase_with_marker_payload(root, name, wipe_on_entry, profile_dir, b"", f)
}

fn run_phase_with_marker_payload(
    root: &Path,
    name: &str,
    wipe_on_entry: bool,
    profile_dir: Option<&Path>,
    marker_payload: &[u8],
    f: impl FnOnce() -> Result<()>,
) -> Result<()> {
    // Any phase rerun invalidates publication first. This also protects
    // explicit later-phase resumes from leaving a stale success marker while
    // index artifacts are being replaced.
    for relative in [
        crate::success_marker::SUCCESS_MARKER_FILENAME,
        crate::snapshot_mirror::MANIFEST_FILENAME,
    ] {
        let path = root.join(relative);
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("invalidate publication marker {}", path.display()))?;
        }
    }
    if wipe_on_entry {
        wipe_phase_outputs(root, name)?;
    }
    if let Some(dir) = profile_dir {
        profile_phase(dir, name, f)?;
    } else {
        f()?;
    }
    write_phase_marker_payload(root, name, marker_payload)?;
    if name == "encode" {
        remove_ephemeral_databricks_config()?;
    }
    Ok(())
}

fn remove_ephemeral_databricks_config() -> Result<()> {
    let Some(path) = std::env::var_os(EPHEMERAL_DATABRICKS_CONFIG_ENV) else {
        return Ok(());
    };
    remove_ephemeral_databricks_config_path(&PathBuf::from(path))
}

fn remove_ephemeral_databricks_config_path(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("remove ephemeral Databricks config: {}", path.display())),
    }
}

fn edge_id_width(max_id: u64) -> u8 {
    let bits = 64 - max_id.leading_zeros();
    let bytes = bits.div_ceil(8).max(1) as u8;
    bytes.min(8)
}

fn shard_range(min_id: u64, count: u64) -> (u64, u64) {
    if count == 0 {
        return (1, 0);
    }
    let max_id = min_id.saturating_add(count.saturating_sub(1));
    (min_id, max_id)
}

fn edge_binary_spec(src_max_id: u64, dst_max_id: u64, sort_bits: Option<u8>) -> EdgeBinarySpec {
    let sort_width = sort_bits.map(|b| (b as u16).div_ceil(8) as u8).unwrap_or(0);
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
    format!(
        "{}/nodes/{}/node_data.ctx",
        TEMP_DIR_NAME,
        name.to_lowercase()
    )
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
    format!(
        "{}/rows.bin",
        temp_edge_key_shard_dir(edge_table, kind, shard)
    )
}

fn temp_edge_key_meta_path(edge_table: &str, kind: EdgeKeyKind, shard: usize) -> String {
    format!(
        "{}/rows.meta",
        temp_edge_key_shard_dir(edge_table, kind, shard)
    )
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

fn temp_sort_key_component_shard_dir(
    rel: &str,
    kind: SortKeyComponentKind,
    shard: usize,
) -> String {
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
    Ok(BufWriter::with_capacity(
        4 * 1024 * 1024,
        target.sink(&path)?,
    ))
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

fn open_fkey_bitmap_writer(target: &OutputTarget, rel_path: &str) -> Result<OutputSink> {
    let path = target.join(rel_path);
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    target.sink(&path)
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
    writer.finish()?;
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
            writer
                .writer
                .write_all(&batch.tags[tag_offset..tag_offset.saturating_add(lens.tag_bytes)])?;
        }
        if lens.bool_bytes > 0 {
            writer.writer.write_all(
                &batch.bools[bool_offset..bool_offset.saturating_add(lens.bool_bytes)],
            )?;
        }
        if lens.data_bytes > 0 {
            writer
                .writer
                .write_all(&batch.data[data_offset..data_offset.saturating_add(lens.data_bytes)])?;
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

fn read_temp_rows(path: &Path, mut f: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    advise_file_willneed(&file);
    let mut file = FrameDecoder::new(file);
    let mut len_buf = [0u8; 4];
    let mut data = Vec::new();
    let mut row_count: u64 = 0;
    loop {
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(err) if err.kind() == std_io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "read length prefix at row {} in {} (file_len={})",
                        row_count,
                        path.display(),
                        file_len,
                    )
                })
            }
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        if data.len() < len {
            data.resize(len, 0u8);
        }
        let slice = &mut data[..len];
        file.read_exact(slice).with_context(|| {
            format!(
                "read {} bytes at row {} in {} (file_len={})",
                len,
                row_count,
                path.display(),
                file_len,
            )
        })?;
        f(slice)?;
        row_count += 1;
    }
}

fn read_fkey_bitmap_mmap(path: &Path, row_count: u64) -> Result<Mmap> {
    let expected = row_count.div_ceil(8) as usize;
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
        let file = fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
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
        weight_bits: info.weight.as_ref().map(|w| w.spec.width_bits).unwrap_or(0),
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
    // Pools consume CPU, so unlike the memory-driven shard floor they must
    // honor the process's cpu.max quota.
    effective_parallelism()
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
            let (tx, rx) =
                std::sync::mpsc::sync_channel::<EdgeWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            let buffer_pool = std::sync::Arc::clone(&buffer_pool);
            handles.push(std::thread::spawn(move || {
                edge_writer_worker(target, buffer_pool, rx)
            }));
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

fn take_with_capacity<T>(source: &mut Vec<T>) -> Vec<T> {
    let mut data = Vec::with_capacity(source.capacity());
    std::mem::swap(&mut data, source);
    data
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
    sort_keys: Vec<SortKeyWriteBatch>,
    // TT-NG timestamp grouping: per-node TIMESTAMP values in node-write order,
    // 1:1 with `node_data.rows`. `None` when grouping is disabled.
    group_ts: Option<Vec<i64>>,
    // TT-NG binary-label sidecars: per-node label bytes in node-write order.
    binary_labels: Option<Vec<u8>>,
    // TT-NG row-weight sidecars: per-node weights in node-write order.
    sample_weights: Option<Vec<f64>>,
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
    // TT-NG binary-label sidecars: per-(table,shard) false/true/invalid counts.
    binary_label_counts: HashMap<(String, usize), crate::binary_label::LabelCounts>,
    // TT-NG row-weight sidecars: per-(table,shard) valid/invalid counts.
    sample_weight_counts: HashMap<(String, usize), crate::sample_weight::WeightCounts>,
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
        for (key, counts) in other.binary_label_counts {
            let entry = self.binary_label_counts.entry(key).or_default();
            entry.false_count = entry.false_count.saturating_add(counts.false_count);
            entry.true_count = entry.true_count.saturating_add(counts.true_count);
            entry.invalid_count = entry.invalid_count.saturating_add(counts.invalid_count);
        }
        for (key, counts) in other.sample_weight_counts {
            let entry = self.sample_weight_counts.entry(key).or_default();
            entry.row_count = entry.row_count.saturating_add(counts.row_count);
            entry.valid_count = entry.valid_count.saturating_add(counts.valid_count);
            entry.invalid_count = entry.invalid_count.saturating_add(counts.invalid_count);
            entry.total_weight += counts.total_weight;
        }
    }
}

struct BitWriter {
    writer: OutputSink,
    buffer: u8,
    bits_used: u8,
}

impl BitWriter {
    fn new(writer: OutputSink) -> Self {
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
            if self.bits_used == 0 && bits_remaining >= 8 {
                let full_bytes = usize::try_from(bits_remaining / 8)
                    .unwrap_or(usize::MAX)
                    .min(data.len().saturating_sub(byte_idx));
                self.writer
                    .write_all(&data[byte_idx..byte_idx + full_bytes])?;
                byte_idx += full_bytes;
                bits_remaining -= (full_bytes as u64) * 8;
                continue;
            }
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
        self.writer.finish()?;
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
    // First worker error, mirrored into producers: when a worker dies its
    // channel closes, and the producer's send error ("sending on a closed
    // channel") would otherwise mask the root cause (e.g. EMFILE).
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl NodeWritePool {
    fn new(target: std::sync::Arc<OutputTarget>, workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        let error = std::sync::Arc::new(std::sync::Mutex::new(None));
        for _ in 0..workers {
            let (tx, rx) =
                std::sync::mpsc::sync_channel::<NodeWriteMessage>(EDGE_WRITE_QUEUE_DEPTH);
            let target = std::sync::Arc::clone(&target);
            let error = std::sync::Arc::clone(&error);
            handles.push(std::thread::spawn(move || {
                match node_writer_worker(target, rx) {
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

    fn senders(&self) -> std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>> {
        std::sync::Arc::clone(&self.senders)
    }

    fn error(&self) -> std::sync::Arc<std::sync::Mutex<Option<String>>> {
        std::sync::Arc::clone(&self.error)
    }

    fn finish(mut self) -> Result<NodeWriterStats> {
        let senders = std::mem::replace(&mut self.senders, Arc::new(Vec::new()));
        drop(senders);
        let mut stats = NodeWriterStats::default();
        for handle in self.handles.drain(..) {
            match handle.join() {
                Ok(result) => {
                    stats.merge(result?);
                }
                Err(_) => return Err(anyhow!("node writer thread panicked")),
            }
        }
        if let Ok(guard) = self.error.lock() {
            if let Some(message) = guard.as_ref() {
                return Err(anyhow!("node writer failed: {message}"));
            }
        }
        Ok(stats)
    }
}

impl Drop for NodeWritePool {
    fn drop(&mut self) {
        let senders = std::mem::replace(&mut self.senders, Arc::new(Vec::new()));
        drop(senders);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn node_writer_worker(
    target: std::sync::Arc<OutputTarget>,
    rx: std::sync::mpsc::Receiver<NodeWriteMessage>,
) -> Result<NodeWriterStats> {
    let mut writers: HashMap<(std::sync::Arc<str>, usize), NodeShardWriter> = HashMap::new();
    let mut sort_key_writers: HashMap<
        (std::sync::Arc<str>, SortKeyComponentKind, usize),
        SortKeyShardWriter,
    > = HashMap::new();
    // TT-NG timestamp grouping: per-(table,shard) append-only spill FD. Opened
    // lazily on the first grouped batch. Empty unless grouping is enabled.
    let mut group_spills: HashMap<(std::sync::Arc<str>, usize), std::fs::File> = HashMap::new();
    let mut binary_label_spills: HashMap<(std::sync::Arc<str>, usize), std::fs::File> =
        HashMap::new();
    let mut sample_weight_spills: HashMap<(std::sync::Arc<str>, usize), std::fs::File> =
        HashMap::new();
    let mut stats = NodeWriterStats::default();
    while let Ok(msg) = rx.recv() {
        match msg {
            NodeWriteMessage::Batch(batch) => {
                let key = (std::sync::Arc::clone(&batch.table), batch.shard);
                let writer = match writers.get_mut(&key) {
                    Some(writer) => writer,
                    None => {
                        let node = open_temp_row_writer(
                            &target,
                            &temp_node_data_path(&batch.table, batch.shard),
                        )?;
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
                // TT-NG timestamp grouping: append this batch's per-node
                // timestamps (node-write order) to the shard spill. Counts are
                // derived later in a bounded post-Encode pass. The spill's
                // record order IS the node order, so the k-th record's node-id
                // is min_id + k.
                if let Some(group_ts) = &batch.group_ts {
                    debug_assert_eq!(
                        group_ts.len(),
                        batch.node_data.rows.len(),
                        "group_ts must be 1:1 with node rows"
                    );
                    let spill = match group_spills.get_mut(&key) {
                        Some(f) => f,
                        None => {
                            let rel = crate::ts_grouping::spill_rel_path(&batch.table, batch.shard);
                            let path = target.join(&rel);
                            if let Some(parent) = path.parent() {
                                std::fs::create_dir_all(parent)?;
                            }
                            let f = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(&path)
                                .with_context(|| {
                                    format!("open ts-group spill {}", path.display())
                                })?;
                            group_spills.entry(key.clone()).or_insert(f)
                        }
                    };
                    crate::ts_grouping::append_spill(spill, group_ts)?;
                }
                if let Some(labels) = &batch.binary_labels {
                    debug_assert_eq!(
                        labels.len(),
                        batch.node_data.rows.len(),
                        "binary labels must be 1:1 with node rows"
                    );
                    let spill = match binary_label_spills.get_mut(&key) {
                        Some(f) => f,
                        None => {
                            let rel =
                                crate::binary_label::spill_rel_path(&batch.table, batch.shard);
                            let path = target.join(&rel);
                            if let Some(parent) = path.parent() {
                                std::fs::create_dir_all(parent)?;
                            }
                            let f = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(&path)
                                .with_context(|| {
                                    format!("open binary-label spill {}", path.display())
                                })?;
                            binary_label_spills.entry(key.clone()).or_insert(f)
                        }
                    };
                    crate::binary_label::append_spill(spill, labels)?;
                    let counts = stats
                        .binary_label_counts
                        .entry((batch.table.to_string(), batch.shard))
                        .or_default();
                    for label in labels {
                        counts.add(*label);
                    }
                }
                if let Some(weights) = &batch.sample_weights {
                    debug_assert_eq!(
                        weights.len(),
                        batch.node_data.rows.len(),
                        "sample weights must be 1:1 with node rows"
                    );
                    let spill = match sample_weight_spills.get_mut(&key) {
                        Some(f) => f,
                        None => {
                            let rel =
                                crate::sample_weight::spill_rel_path(&batch.table, batch.shard);
                            let path = target.join(&rel);
                            if let Some(parent) = path.parent() {
                                std::fs::create_dir_all(parent)?;
                            }
                            let f = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(&path)
                                .with_context(|| {
                                    format!("open sample-weight spill {}", path.display())
                                })?;
                            sample_weight_spills.entry(key.clone()).or_insert(f)
                        }
                    };
                    crate::sample_weight::append_spill(spill, weights)?;
                    let counts = stats
                        .sample_weight_counts
                        .entry((batch.table.to_string(), batch.shard))
                        .or_default();
                    for weight in weights {
                        counts.add(*weight);
                    }
                }
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
                // Write bundled sort key data.
                for sk_batch in batch.sort_keys {
                    write_sort_key_batch(&target, &mut sort_key_writers, sk_batch)?;
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
    for (_, mut writer) in sort_key_writers {
        writer.writer.flush()?;
        let sink = writer
            .writer
            .into_inner()
            .map_err(|_| anyhow!("sort key flush failed"))?;
        sink.finish()?;
    }
    for (_, mut spill) in group_spills {
        spill.flush()?;
    }
    for (_, mut spill) in binary_label_spills {
        spill.flush()?;
    }
    for (_, mut spill) in sample_weight_spills {
        spill.flush()?;
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
    // TT-NG timestamp grouping: when `Some`, holds this shard's per-node
    // TIMESTAMP value, pushed 1:1 with `push_node_row`/`push_empty_node`, in
    // node-write order. `None` (default) = grouping disabled, zero overhead.
    group_ts: Option<Vec<i64>>,
    // TT-NG binary label sidecars: one label byte per node row in write order.
    binary_labels: Option<Vec<u8>>,
    // TT-NG row-weight sidecars: one f64 per node row in write order.
    sample_weights: Option<Vec<f64>>,
}

struct NodeBatcher {
    table: std::sync::Arc<str>,
    table_hash: u64,
    rel_names: std::sync::Arc<Vec<std::sync::Arc<str>>>,
    buffers: Vec<NodeShardBuffer>,
    buffer_limit: usize,
    senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>>,
    error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    // TT-NG timestamp grouping enabled for this table (see `group_by_timestamp`).
    group_by_timestamp: bool,
    binary_label_enabled: bool,
    sample_weight_enabled: bool,
}

impl NodeBatcher {
    #[allow(clippy::too_many_arguments)]
    fn new(
        table: std::sync::Arc<str>,
        rel_names: std::sync::Arc<Vec<std::sync::Arc<str>>>,
        has_pkey: bool,
        num_shards: usize,
        buffer_limit: usize,
        senders: std::sync::Arc<Vec<std::sync::mpsc::SyncSender<NodeWriteMessage>>>,
        error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
        group_by_timestamp: bool,
        binary_label_enabled: bool,
        sample_weight_enabled: bool,
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
                pkey: if has_pkey { Some(Vec::new()) } else { None },
                pkey_rows: 0,
                fkeys: vec![Vec::new(); rel_names.len()],
                fkey_bits: vec![Vec::new(); rel_names.len()],
                fkey_rows: vec![0; rel_names.len()],
                bytes: 0,
                group_ts: if group_by_timestamp {
                    Some(Vec::new())
                } else {
                    None
                },
                binary_labels: if binary_label_enabled {
                    Some(Vec::new())
                } else {
                    None
                },
                sample_weights: if sample_weight_enabled {
                    Some(Vec::new())
                } else {
                    None
                },
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
            group_by_timestamp,
            binary_label_enabled,
            sample_weight_enabled,
        }
    }

    /// Record a node's TIMESTAMP for grouping. MUST be called exactly once per
    /// `push_node_row`/`push_empty_node`, in the same order, so the k-th ts in a
    /// shard buffer pairs with the k-th node written to that shard. No-op when
    /// grouping is disabled.
    fn push_group_ts(&mut self, shard: usize, ts: i64) -> Result<()> {
        if !self.group_by_timestamp {
            return Ok(());
        }
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        if let Some(group_ts) = buffer.group_ts.as_mut() {
            group_ts.push(ts);
        }
        Ok(())
    }

    fn push_binary_label(&mut self, shard: usize, label: u8) -> Result<()> {
        if !self.binary_label_enabled {
            return Ok(());
        }
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        if let Some(labels) = buffer.binary_labels.as_mut() {
            labels.push(label);
        }
        Ok(())
    }

    fn push_sample_weight(&mut self, shard: usize, weight: f64) -> Result<()> {
        if !self.sample_weight_enabled {
            return Ok(());
        }
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        if let Some(weights) = buffer.sample_weights.as_mut() {
            weights.push(weight);
        }
        Ok(())
    }

    fn push_encoded_node(
        &mut self,
        shard: usize,
        lens: RowPackRowLens,
        tags: &[u8],
        bools: &[u8],
        data: &[u8],
    ) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        let row_len = lens
            .tag_bytes
            .saturating_add(lens.bool_bytes)
            .saturating_add(lens.data_bytes);
        let _ = u16::try_from(row_len).map_err(|_| anyhow!("temp row too large"))?;
        if tags.len() != lens.tag_bytes
            || bools.len() != lens.bool_bytes
            || data.len() != lens.data_bytes
        {
            return Err(anyhow!(
                "rowpack encoded buffer lengths do not match row metadata"
            ));
        }
        buffer.node.tags.extend_from_slice(tags);
        buffer.node.bools.extend_from_slice(bools);
        buffer.node.data.extend_from_slice(data);
        buffer.node.rows.push(lens);
        buffer.bytes = buffer.bytes.saturating_add(2 + row_len);
        Ok(())
    }

    fn push_empty_node(&mut self, shard: usize) -> Result<()> {
        self.push_encoded_node(
            shard,
            RowPackRowLens {
                tag_bytes: 0,
                bool_bytes: 0,
                data_bytes: 0,
            },
            &[],
            &[],
            &[],
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

    fn push_fkey(
        &mut self,
        shard: usize,
        rel_idx: usize,
        row: &[u8],
        timestamp_valid: bool,
    ) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("fkey buffer shard {} out of bounds", shard))?;
        if let Some(fkey) = buffer.fkeys.get_mut(rel_idx) {
            let row_idx = buffer.fkey_rows.get(rel_idx).copied().unwrap_or(0);
            let len = u32::try_from(row.len()).map_err(|_| anyhow!("temp row too large"))?;
            fkey.extend_from_slice(&len.to_le_bytes());
            fkey.extend_from_slice(row);
            if let Some(count) = buffer.fkey_rows.get_mut(rel_idx) {
                *count = count.saturating_add(1);
            }
            if let Some(bits) = buffer.fkey_bits.get_mut(rel_idx) {
                let byte_idx = (row_idx / 8) as usize;
                let bit = (row_idx % 8) as u8;
                let mut added = 0usize;
                if bits.len() <= byte_idx {
                    bits.push(0);
                    added = 1;
                }
                if timestamp_valid {
                    *bits
                        .get_mut(byte_idx)
                        .ok_or_else(|| anyhow!("missing fkey validity byte"))? |= 1u8 << bit;
                }
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
            fkey.extend_from_slice(&0u32.to_le_bytes());
            if let Some(count) = buffer.fkey_rows.get_mut(rel_idx) {
                *count = count.saturating_add(1);
            }
            if let Some(bits) = buffer.fkey_bits.get_mut(rel_idx) {
                let byte_idx = (row_idx / 8) as usize;
                let mut added = 0usize;
                if bits.len() <= byte_idx {
                    bits.push(0);
                    added = 1;
                }
                buffer.bytes = buffer.bytes.saturating_add(added);
            }
            buffer.bytes = buffer.bytes.saturating_add(4);
        }
        Ok(())
    }

    fn maybe_flush(
        &mut self,
        shard: usize,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
    ) -> Result<()> {
        let buffer = self
            .buffers
            .get(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        let sort_key_bytes = sort_key_batchers
            .iter()
            .map(|batcher| batcher.buffered_bytes(shard))
            .sum::<usize>();
        if buffer.bytes.saturating_add(sort_key_bytes) >= self.buffer_limit {
            self.flush_shard(shard, sort_key_batchers)?;
        }
        Ok(())
    }

    fn flush_shard(
        &mut self,
        shard: usize,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
    ) -> Result<()> {
        if let Ok(guard) = self.error.lock() {
            if let Some(message) = guard.as_ref() {
                return Err(anyhow!("node writer failed: {message}"));
            }
        }
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
        let node = RowPackSegmentBatch {
            tags: take_with_capacity(&mut buffer.node.tags),
            bools: take_with_capacity(&mut buffer.node.bools),
            data: take_with_capacity(&mut buffer.node.data),
            rows: take_with_capacity(&mut buffer.node.rows),
        };
        let pkey = buffer.pkey.as_mut().map(take_with_capacity);
        let pkey_rows = buffer.pkey_rows;
        let fkeys = buffer.fkeys.iter_mut().map(take_with_capacity).collect();
        let fkey_bits = buffer
            .fkey_bits
            .iter_mut()
            .map(take_with_capacity)
            .collect();
        let fkey_rows = buffer.fkey_rows.clone();
        // Drain this shard's grouping timestamps (node-write order) into the
        // batch, leaving an empty Vec to keep accumulating. `None` when
        // grouping is disabled.
        let group_ts = buffer.group_ts.as_mut().map(take_with_capacity);
        let binary_labels = buffer.binary_labels.as_mut().map(take_with_capacity);
        let sample_weights = buffer.sample_weights.as_mut().map(take_with_capacity);
        // Drain sort key data for this shard and bundle into the batch.
        let mut sort_keys = Vec::new();
        for sk in sort_key_batchers.iter_mut() {
            if let Some(sk_batch) = sk.drain_shard(shard)? {
                sort_keys.push(sk_batch);
            }
        }
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
            sort_keys,
            group_ts,
            binary_labels,
            sample_weights,
        };
        buffer.pkey_rows = 0;
        buffer.fkey_rows.fill(0);
        buffer.bytes = 0;
        let idx = edge_writer_index(self.table_hash, shard, self.senders.len());
        self.senders[idx]
            .send(NodeWriteMessage::Batch(batch))
            .map_err(|e| {
                if let Ok(guard) = self.error.lock() {
                    if let Some(message) = guard.as_ref() {
                        return anyhow!("node writer failed: {message}");
                    }
                }
                anyhow!("node writer channel error: {e}")
            })?;
        Ok(())
    }

    fn flush_all(&mut self, sort_key_batchers: &mut [SortKeyComponentBatcher]) -> Result<()> {
        for shard in 0..self.buffers.len() {
            self.flush_shard(shard, sort_key_batchers)?;
        }
        Ok(())
    }
}

#[allow(dead_code)]
struct SortKeyWriteBatch {
    rel: std::sync::Arc<str>,
    kind: SortKeyComponentKind,
    shard: usize,
    data: Vec<u8>,
    row_count: u64,
    width_bytes: usize,
}

/// Write a single sort key batch to the appropriate shard writer, creating it
/// on first access.  Shared by node_writer_worker and edge_src_writer_worker.
fn write_sort_key_batch(
    target: &OutputTarget,
    writers: &mut HashMap<(std::sync::Arc<str>, SortKeyComponentKind, usize), SortKeyShardWriter>,
    batch: SortKeyWriteBatch,
) -> Result<()> {
    let key = (std::sync::Arc::clone(&batch.rel), batch.kind, batch.shard);
    let writer = match writers.get_mut(&key) {
        Some(writer) => writer,
        None => {
            let path = temp_sort_key_component_path(&batch.rel, batch.kind, batch.shard);
            let w = open_sort_key_writer(target, &path)?;
            writers.insert(
                key.clone(),
                SortKeyShardWriter {
                    writer: w,
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
    Ok(())
}

struct SortKeyShardWriter {
    writer: BufWriter<OutputSink>,
    width_bytes: usize,
}

struct SortKeyComponentBatcher {
    rel: std::sync::Arc<str>,
    kind: SortKeyComponentKind,
    width_bytes: u8,
    encoder: SortKeyComponentEncoder,
    buffers: Vec<Vec<u8>>,
    rows: Vec<u64>,
}

impl SortKeyComponentBatcher {
    fn new(
        rel: std::sync::Arc<str>,
        kind: SortKeyComponentKind,
        width_bytes: u8,
        encoder: SortKeyComponentEncoder,
        num_shards: usize,
    ) -> Self {
        Self {
            rel,
            kind,
            width_bytes,
            encoder,
            buffers: vec![Vec::new(); num_shards],
            rows: vec![0; num_shards],
        }
    }

    fn push_row(&mut self, shard: usize, access: &mut RowAccess<'_>) -> Result<()> {
        let value = match &mut self.encoder {
            SortKeyComponentEncoder::Timestamp {
                column_idx,
                encoding,
            } => {
                let ts_value = if let Some(secs) = access.get_timestamp(*column_idx).ok().flatten()
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
        let bytes = value.to_le_bytes();
        buffer.extend_from_slice(&bytes[..self.width_bytes as usize]);
        if let Some(row_count) = self.rows.get_mut(shard) {
            *row_count = row_count.saturating_add(1);
        }
        Ok(())
    }

    fn buffered_bytes(&self, shard: usize) -> usize {
        self.buffers.get(shard).map(Vec::len).unwrap_or(0)
    }

    /// Drain the buffer for a shard, returning a `SortKeyWriteBatch` if there
    /// is data. Does not send to any channel — the caller bundles the result
    /// into a `NodeWriteBatch` or `EdgeSrcWriteBatch`.
    fn drain_shard(&mut self, shard: usize) -> Result<Option<SortKeyWriteBatch>> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("sort key buffer shard {} out of bounds", shard))?;
        let rows = self
            .rows
            .get_mut(shard)
            .ok_or_else(|| anyhow!("sort key row count missing"))?;
        if *rows == 0 {
            return Ok(None);
        }
        let batch = SortKeyWriteBatch {
            rel: std::sync::Arc::clone(&self.rel),
            kind: self.kind,
            shard,
            data: take_with_capacity(buffer),
            row_count: *rows,
            width_bytes: self.width_bytes as usize,
        };
        *rows = 0;
        Ok(Some(batch))
    }
}

struct EdgeSrcWriteBatch {
    rel: std::sync::Arc<str>,
    shard: usize,
    data: Vec<u8>,
    row_count: u64,
    sort_keys: Vec<SortKeyWriteBatch>,
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

    fn finish(mut self) -> Result<EdgeSrcWriterStats> {
        let senders = std::mem::replace(&mut self.senders, Arc::new(Vec::new()));
        drop(senders);
        let mut stats = EdgeSrcWriterStats::default();
        for handle in self.handles.drain(..) {
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

impl Drop for EdgeSrcWritePool {
    fn drop(&mut self) {
        let senders = std::mem::replace(&mut self.senders, Arc::new(Vec::new()));
        drop(senders);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn edge_src_writer_worker(
    target: std::sync::Arc<OutputTarget>,
    rx: std::sync::mpsc::Receiver<EdgeSrcWriteMessage>,
) -> Result<EdgeSrcWriterStats> {
    let mut writers: HashMap<(std::sync::Arc<str>, usize), TempRowWriter> = HashMap::new();
    let mut sort_key_writers: HashMap<
        (std::sync::Arc<str>, SortKeyComponentKind, usize),
        SortKeyShardWriter,
    > = HashMap::new();
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
                            open_temp_row_writer(
                                &target,
                                &temp_edge_path(&batch.rel, batch.shard),
                            )?,
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
                // Write bundled sort key data.
                for sk_batch in batch.sort_keys {
                    write_sort_key_batch(&target, &mut sort_key_writers, sk_batch)?;
                }
            }
        }
    }
    for (_, writer) in writers {
        finish_temp_writer(writer)?;
    }
    for (_, mut writer) in sort_key_writers {
        writer.writer.flush()?;
        let sink = writer
            .writer
            .into_inner()
            .map_err(|_| anyhow!("sort key flush failed"))?;
        sink.finish()?;
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

    fn finish(mut self) -> Result<DestKeyWriterStats> {
        let senders = std::mem::replace(&mut self.senders, Arc::new(Vec::new()));
        drop(senders);
        let mut stats = DestKeyWriterStats::default();
        for handle in self.handles.drain(..) {
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

impl Drop for DestKeyWritePool {
    fn drop(&mut self) {
        let senders = std::mem::replace(&mut self.senders, Arc::new(Vec::new()));
        drop(senders);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn dest_key_writer_worker(
    target: std::sync::Arc<OutputTarget>,
    rx: std::sync::mpsc::Receiver<DestKeyWriteMessage>,
) -> Result<DestKeyWriterStats> {
    let mut node_writers: HashMap<
        (std::sync::Arc<str>, std::sync::Arc<str>, usize),
        TempRowWriter,
    > = HashMap::new();
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
                let key = (
                    std::sync::Arc::clone(&table),
                    std::sync::Arc::clone(&rel),
                    shard,
                );
                let writer = match node_writers.get_mut(&key) {
                    Some(writer) => writer,
                    None => {
                        node_writers.insert(
                            key.clone(),
                            open_temp_row_writer(
                                &target,
                                &temp_fkey_dst_path(&table, &rel, shard),
                            )?,
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

    fn push(
        &mut self,
        shard: usize,
        row: &[u8],
        timestamp_valid: bool,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
    ) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(shard)
            .ok_or_else(|| anyhow!("edge src buffer shard {} out of bounds", shard))?;
        let total_len = (row.len() + 1) as u32;
        buffer.data.extend_from_slice(&total_len.to_le_bytes());
        buffer.data.extend_from_slice(row);
        buffer.data.push(if timestamp_valid { 1 } else { 0 });
        buffer.row_count = buffer.row_count.saturating_add(1);
        let sort_key_bytes = sort_key_batchers
            .iter()
            .map(|batcher| batcher.buffered_bytes(shard))
            .sum::<usize>();
        if buffer.data.len().saturating_add(sort_key_bytes) >= self.buffer_limit {
            self.flush_shard(shard, sort_key_batchers)?;
        }
        Ok(())
    }

    fn flush_shard(
        &mut self,
        shard: usize,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
    ) -> Result<()> {
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
        let data = take_with_capacity(&mut buffer.data);
        // Drain sort key data for this shard and bundle into the batch.
        let mut sort_keys = Vec::new();
        for sk in sort_key_batchers.iter_mut() {
            if let Some(sk_batch) = sk.drain_shard(shard)? {
                sort_keys.push(sk_batch);
            }
        }
        let batch = EdgeSrcWriteBatch {
            rel: std::sync::Arc::clone(&self.rel),
            shard,
            data,
            row_count: buffer.row_count,
            sort_keys,
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

    fn flush_all(&mut self, sort_key_batchers: &mut [SortKeyComponentBatcher]) -> Result<()> {
        for shard in 0..self.buffers.len() {
            self.flush_shard(shard, sort_key_batchers)?;
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
        let len = u32::try_from(row.len()).map_err(|_| anyhow!("temp row too large"))?;
        fkey.data.extend_from_slice(&len.to_le_bytes());
        fkey.data.extend_from_slice(row);
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
            let data = take_with_capacity(&mut entry.data);
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
        let len = u32::try_from(row.len()).map_err(|_| anyhow!("temp row too large"))?;
        buffer.data.extend_from_slice(&len.to_le_bytes());
        buffer.data.extend_from_slice(row);
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
        let data = take_with_capacity(&mut buffer.data);
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

fn sum_published_dir_bytes(root: &Path) -> Result<u64> {
    let mut total = 0u64;
    if !root.is_dir() {
        return sum_dir_bytes(root);
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_name() == TEMP_DIR_NAME {
            continue;
        }
        total = total.saturating_add(sum_dir_bytes(&entry.path())?);
    }
    Ok(total)
}

fn default_index_mem_budget_bytes() -> Option<u64> {
    let total = total_memory_bytes()?;
    Some((total as f64 * DEFAULT_INDEX_MEM_BUDGET_FRAC) as u64)
}

#[cfg(target_os = "linux")]
pub(crate) fn total_memory_bytes() -> Option<u64> {
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
pub(crate) fn total_memory_bytes() -> Option<u64> {
    None
}

/// CAS-loop atomic max-update. Used by pk_map peak-RSS sampling.
fn update_atomic_max(target: &std::sync::atomic::AtomicU64, value: u64) {
    use std::sync::atomic::Ordering;
    let mut cur = target.load(Ordering::Relaxed);
    while value > cur {
        match target.compare_exchange_weak(cur, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(prev) => cur = prev,
        }
    }
}

/// Best-effort read of process anonymous-RSS in bytes for diagnostic logs.
/// Returns 0 if unavailable so call sites can format unconditionally.
#[cfg(target_os = "linux")]
fn rss_anon_diag_bytes() -> u64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("RssAnon:") {
            return parse_kb_line(value).unwrap_or(0);
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn rss_anon_diag_bytes() -> u64 {
    0
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

/// The cgroup-v2 sub-path this process belongs to, from `/proc/self/cgroup`
/// (the `0::<path>` line). Returns "" for the root. `None` if unreadable.
///
/// When the limit is set on a NESTED cgroup (e.g. a `systemd-run --scope` slice, or a
/// k8s pod slice) the enforced `memory.max` lives at
/// `/sys/fs/cgroup/<this-path>/memory.max`, NOT at the mount root. Reading only the root
/// misses a nested limit entirely. (Inside a container the cgroup namespace makes the root
/// == the container's own cgroup, so the old root-only read happened to work there.)
#[cfg(target_os = "linux")]
fn cgroup_v2_self_path() -> Option<String> {
    let contents = fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("0::") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

/// Walk the process's own cgroup-v2 subtree from leaf to root, reading `memory.max` at each
/// level, and return the TIGHTEST (minimum) numeric limit found. Falls back to the mount
/// root and the v1 hierarchy. Returns None when unlimited/unreadable.
#[cfg(target_os = "linux")]
fn read_cgroup_limit() -> Option<u64> {
    let read = |p: &std::path::Path| -> Option<String> { fs::read_to_string(p).ok() };
    resolve_tightest_mem_limit(cgroup_v2_self_path().as_deref(), &read)
}

/// Pure leaf→root memory-limit resolution, parameterized on the file reader and the
/// process's cgroup-v2 self path. Also compiled in tests on non-Linux hosts.
#[cfg(any(target_os = "linux", test))]
fn resolve_tightest_mem_limit(
    self_cgroup: Option<&str>,
    read: &impl Fn(&std::path::Path) -> Option<String>,
) -> Option<u64> {
    let parse = |s: &str| -> Option<u64> {
        let t = s.trim();
        if t == "max" {
            None
        } else {
            t.parse::<u64>().ok()
        }
    };
    let mut tightest: Option<u64> = None;
    // cgroup v2: walk this process's own path up to the root.
    if let Some(self_path) = self_cgroup {
        let mut cur = std::path::PathBuf::from("/sys/fs/cgroup");
        cur.push(self_path.trim_start_matches('/'));
        loop {
            if let Some(s) = read(&cur.join("memory.max")) {
                if let Some(bytes) = parse(&s) {
                    tightest = Some(tightest.map_or(bytes, |t| t.min(bytes)));
                }
            }
            if cur == std::path::Path::new("/sys/fs/cgroup") {
                break;
            }
            if !cur.pop() {
                break;
            }
        }
    }
    // v2 root fallback (also covers the container-namespaced case).
    if let Some(s) = read(std::path::Path::new("/sys/fs/cgroup/memory.max")) {
        if let Some(bytes) = parse(&s) {
            tightest = Some(tightest.map_or(bytes, |t| t.min(bytes)));
        }
    }
    if tightest.is_some() {
        return tightest;
    }
    // cgroup v1 fallback.
    if let Some(limit) = read(std::path::Path::new(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )) {
        if let Ok(bytes) = limit.trim().parse::<u64>() {
            if bytes < (u64::MAX / 2) {
                return Some(bytes);
            }
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

/// Read the cgroup CPU quota as a whole-core count (rounded up, floored at 1), taking the
/// TIGHTEST quota across this process's own cgroup-v2 path (leaf → root).
/// v2: `cpu.max` = "<quota> <period>"; v1: `cpu.cfs_quota_us` / `cpu.cfs_period_us`.
/// Returns None when unlimited or unreadable. Linux only.
#[cfg(target_os = "linux")]
pub(crate) fn read_cgroup_cpu_quota() -> Option<usize> {
    let read = |p: &std::path::Path| -> Option<String> { fs::read_to_string(p).ok() };
    resolve_tightest_cpu_quota(cgroup_v2_self_path().as_deref(), &read)
}

/// Pure leaf→root cpu-quota resolution, parameterized on the file reader and the process's
/// cgroup-v2 self path. Also compiled in tests on non-Linux hosts.
#[cfg(any(target_os = "linux", test))]
fn resolve_tightest_cpu_quota(
    self_cgroup: Option<&str>,
    read: &impl Fn(&std::path::Path) -> Option<String>,
) -> Option<usize> {
    let quota_cores = |quota: i64, period: i64| -> Option<usize> {
        if quota <= 0 || period <= 0 {
            return None;
        }
        Some(((quota as f64 / period as f64).ceil() as usize).max(1))
    };
    let parse_cpu_max = |contents: &str| -> Option<usize> {
        let mut parts = contents.split_whitespace();
        let (q, p) = (parts.next()?, parts.next()?);
        if q == "max" {
            return None;
        }
        quota_cores(q.parse::<i64>().ok()?, p.parse::<i64>().ok()?)
    };
    let mut tightest: Option<usize> = None;
    if let Some(self_path) = self_cgroup {
        let mut cur = std::path::PathBuf::from("/sys/fs/cgroup");
        cur.push(self_path.trim_start_matches('/'));
        loop {
            if let Some(s) = read(&cur.join("cpu.max")) {
                if let Some(c) = parse_cpu_max(&s) {
                    tightest = Some(tightest.map_or(c, |t| t.min(c)));
                }
            }
            if cur == std::path::Path::new("/sys/fs/cgroup") {
                break;
            }
            if !cur.pop() {
                break;
            }
        }
    }
    if let Some(s) = read(std::path::Path::new("/sys/fs/cgroup/cpu.max")) {
        if let Some(c) = parse_cpu_max(&s) {
            tightest = Some(tightest.map_or(c, |t| t.min(c)));
        }
    }
    if tightest.is_some() {
        return tightest;
    }
    let q = read(std::path::Path::new("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"));
    let p = read(std::path::Path::new("/sys/fs/cgroup/cpu/cpu.cfs_period_us"));
    if let (Some(q), Some(p)) = (q, p) {
        if let (Ok(q), Ok(p)) = (q.trim().parse::<i64>(), p.trim().parse::<i64>()) {
            return quota_cores(q, p);
        }
    }
    None
}

/// CPU capacity available through the process affinity/cpuset mask, without
/// applying a cgroup CPU-bandwidth quota (`cpu.max`).
///
/// Linux's `available_parallelism()` accounts for both affinity and
/// `cpu.max`. Shard sizing needs the former but deliberately not the latter:
/// lowering worker bandwidth must not produce fewer, larger shards.
#[cfg(target_os = "linux")]
pub fn cpu_capacity_parallelism() -> usize {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let status =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    if status == 0 {
        let count = (0..libc::CPU_SETSIZE as usize)
            .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &set) })
            .count();
        if count > 0 {
            return count;
        }
    }
    // `available_parallelism()` is not a safe fallback here because Linux's
    // implementation also applies cpu.max. Online host CPUs may overestimate
    // a cpuset but remain memory-safe by producing smaller shards.
    let online = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if online > 0 {
        online as usize
    } else {
        1
    }
}

#[cfg(not(target_os = "linux"))]
pub fn cpu_capacity_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Effective CPU parallelism for worker pools.
///
/// `available_parallelism()` honors affinity/cpuset and, on supported Linux
/// versions, cgroup CPU quota. The explicit quota minimum also covers runtimes
/// where the standard library does not discover the quota.
pub fn effective_parallelism() -> usize {
    let host = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    #[cfg(target_os = "linux")]
    {
        if let Some(q) = read_cgroup_cpu_quota() {
            return host.min(q).max(1);
        }
    }
    host
}

#[allow(clippy::type_complexity)]
struct WorkItem<S> {
    run: Box<dyn FnOnce(&mut S) -> Result<()> + Send>,
    bucket: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
}

struct PoolTask<S> {
    item: WorkItem<S>,
    done: Option<std::sync::mpsc::Sender<Result<()>>>,
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
                            if let Some(done) = task.done {
                                let _ = done.send(result);
                            }
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
                .send(PoolMessage::Task(PoolTask {
                    item: task,
                    done: Some(tx),
                }))
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

pub fn default_worker_threads() -> usize {
    effective_parallelism()
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

fn key_type_for_dtype(dtype: &IngestedType) -> Result<KeyType> {
    match dtype {
        IngestedType::Int64
        | IngestedType::Int32
        | IngestedType::Int16
        | IngestedType::Int8
        | IngestedType::UInt64
        | IngestedType::UInt32
        | IngestedType::UInt16
        | IngestedType::UInt8
        | IngestedType::Float32
        | IngestedType::Float64
        | IngestedType::Decimal128 => Ok(KeyType::Int64),
        IngestedType::Decimal256 => Ok(KeyType::VarBytes),
        IngestedType::String | IngestedType::Bytes => Ok(KeyType::VarBytes),
        IngestedType::Uuid => Ok(KeyType::Uuid),
        IngestedType::Uuid256 => Ok(KeyType::Uuid256),
        IngestedType::Ip => Ok(KeyType::Ip),
        _ => Err(anyhow!("unsupported key type: {:?}", dtype)),
    }
}

fn key_type_for_column(col: &ColumnSpec) -> Result<KeyType> {
    key_type_for_dtype(&col.ingested_dtype)
}

fn column_index(table: &TableSpec, name: &str) -> Result<usize> {
    table
        .columns
        .iter()
        .position(|col| col.name == name)
        .ok_or_else(|| anyhow!("missing column {}", name))
}

// ─── FK-free LP resolution-mode helpers (S3) ────────────────────────────────
//
// Runtime spec for one resolution column, built at the start of
// `build_temp_node_batches` from `IngestContext.resolve_columns`. The Encode
// consumer resolves the source column's PK reference value(s) to graph node-ids
// (via the shared `RefResolver`) and feeds the result into the encoder through
// a dedicated override channel — NO FK edge / no temp-edge push.
/// Resolution column role, driving the encode-time orphan-drop rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolveRole {
    /// `entity_node_id` — unresolved anchor drops the row.
    Anchor,
    /// `target_node_ids` — an all-orphan list drops the row.
    Positives,
    /// `exclude_node_ids` — empty is fine, never drops.
    Exclude,
}

struct ResolveColRuntime {
    /// Column index of the NEW node-id column in `table.columns`.
    node_id_idx: usize,
    /// Column index of the source PK reference column in `table.columns`.
    src_idx: usize,
    /// Parent node table whose pkmap resolves the references.
    dst_table: String,
    /// Scalar vs list arity.
    kind: crate::config::ResolveKind,
    /// Drop-rule role.
    role: ResolveRole,
    /// Element/scalar key type used to read the source value (Int64 / VarBytes /
    /// Uuid / …), derived from the source column's ingested dtype.
    src_type: IngestedType,
}

/// Element (scalar) PK type for a list-bearing source column. Mirrors the
/// list-FK `list_fk_element_type` port; identity for already-scalar types.
fn resolve_list_element_type(container: &IngestedType) -> Result<IngestedType> {
    Ok(match container {
        IngestedType::StringArray => IngestedType::String,
        IngestedType::Int64Array => IngestedType::Int64,
        IngestedType::Int32Array => IngestedType::Int32,
        IngestedType::Int16Array => IngestedType::Int16,
        IngestedType::Int8Array => IngestedType::Int8,
        IngestedType::String | IngestedType::Int64 | IngestedType::Int32 => container.clone(),
        other => {
            return Err(anyhow!(
                "unsupported resolution list element type: {other:?}"
            ))
        }
    })
}

/// Encode one list child element at `pos` into `out` using `elem_type`, matching
/// the scalar `key_from_access_into_buf` byte layout (Int64 → 8 LE bytes;
/// String → raw bytes). Ported from the list-FK worktree (`encode_list_element_key`).
fn resolve_encode_list_element_key(
    child: &dyn arrow_array::Array,
    pos: usize,
    elem_type: &IngestedType,
    out: &mut Vec<u8>,
) -> KeyParseResult<()> {
    out.clear();
    match elem_type {
        IngestedType::String => {
            let arr = child
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or(KeyParseError::TypeMismatch)?;
            out.extend_from_slice(arr.value(pos).as_bytes());
            Ok(())
        }
        IngestedType::Int64 | IngestedType::Int32 | IngestedType::Int16 | IngestedType::Int8 => {
            let v: i64 = if let Some(a) = child.as_any().downcast_ref::<arrow_array::Int64Array>() {
                a.value(pos)
            } else if let Some(a) = child.as_any().downcast_ref::<arrow_array::Int32Array>() {
                a.value(pos) as i64
            } else if let Some(a) = child.as_any().downcast_ref::<arrow_array::Int16Array>() {
                a.value(pos) as i64
            } else if let Some(a) = child.as_any().downcast_ref::<arrow_array::Int8Array>() {
                a.value(pos) as i64
            } else {
                return Err(KeyParseError::TypeMismatch);
            };
            out.extend_from_slice(&(v as u64).to_le_bytes());
            Ok(())
        }
        _ => Err(KeyParseError::TypeMismatch),
    }
}

/// Build the per-element `Key`s for a list-bearing source column at the current
/// row. Null/absent cells and unparseable elements are skipped (mask-don't-fail,
/// mirroring the scalar path). Ported/adapted from the list-FK
/// `list_keys_from_access_into_buf`, but yields `Key`s for `RefResolver::lookup`
/// instead of a framed temp-edge buffer.
fn resolve_list_keys_from_access(
    access: &mut RowAccess<'_>,
    idx: usize,
    container_type: &IngestedType,
    scratch: &mut Vec<u8>,
) -> Result<Vec<Key>> {
    let elem_type = resolve_list_element_type(container_type)?;
    let array = access.array(idx)?;
    let row = access.row();
    let (child, start, end): (&ArrayRef, usize, usize) =
        if let Some(list) = array.as_any().downcast_ref::<arrow_array::ListArray>() {
            if list.is_null(row) {
                return Ok(Vec::new());
            }
            let offs = list.value_offsets();
            (list.values(), offs[row] as usize, offs[row + 1] as usize)
        } else if let Some(list) = array.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            if list.is_null(row) {
                return Ok(Vec::new());
            }
            let offs = list.value_offsets();
            (list.values(), offs[row] as usize, offs[row + 1] as usize)
        } else {
            return Ok(Vec::new());
        };
    let mut keys = Vec::with_capacity(end.saturating_sub(start));
    for pos in start..end {
        if child.is_null(pos) {
            continue;
        }
        if resolve_encode_list_element_key(child.as_ref(), pos, &elem_type, scratch).is_err() {
            continue;
        }
        let mut data: &[u8] = scratch.as_slice();
        match read_key_from_buf_for_type(&mut data, &elem_type) {
            Ok(key) => keys.push(key),
            Err(_) => continue,
        }
    }
    Ok(keys)
}

/// Build the per-table runtime resolve specs (column indices + key types) from
/// the global [`crate::config::ResolveColumnSpec`] list, keeping only those that
/// target `table`. Errs if a referenced source / node-id column is missing.
fn build_resolve_runtime(
    table: &TableSpec,
    resolve_columns: &[crate::config::ResolveColumnSpec],
) -> Result<Vec<ResolveColRuntime>> {
    let mut out = Vec::new();
    for spec in resolve_columns {
        if spec.table != table.name {
            continue;
        }
        let src_idx = column_index(table, &spec.src_col).with_context(|| {
            format!(
                "resolution mode: source column `{}` not found in table `{}`",
                spec.src_col, table.name
            )
        })?;
        let node_id_idx = column_index(table, &spec.node_id_col).with_context(|| {
            format!(
                "resolution mode: node-id column `{}` not found in table `{}` \
                 (append it to TableSpec.columns)",
                spec.node_id_col, table.name
            )
        })?;
        // Validate the target column's logical type matches the arity.
        let node_col = &table.columns[node_id_idx];
        match spec.kind {
            crate::config::ResolveKind::Scalar => {
                if node_col.ingested_dtype != IngestedType::UInt64 {
                    return Err(anyhow!(
                        "resolution scalar column `{}` must be UInt64, got {:?}",
                        spec.node_id_col,
                        node_col.ingested_dtype
                    ));
                }
            }
            crate::config::ResolveKind::List => {
                if node_col.ingested_dtype != IngestedType::Int64Array {
                    return Err(anyhow!(
                        "resolution list column `{}` must be Int64Array, got {:?}",
                        spec.node_id_col,
                        node_col.ingested_dtype
                    ));
                }
            }
        }
        // Scalar = anchor; list `target_node_ids` = positives, else exclude.
        let role = match spec.kind {
            crate::config::ResolveKind::Scalar => ResolveRole::Anchor,
            crate::config::ResolveKind::List => {
                if spec.node_id_col == diskgraph_core::tt_builder::TARGET_NODE_IDS_COL {
                    ResolveRole::Positives
                } else {
                    ResolveRole::Exclude
                }
            }
        };
        let src_type = table.columns[src_idx].ingested_dtype.clone();
        out.push(ResolveColRuntime {
            node_id_idx,
            src_idx,
            dst_table: spec.dst_table.clone(),
            kind: spec.kind,
            role,
            src_type,
        });
    }
    Ok(out)
}

/// Indices of all resolution NODE-ID columns for `table` — these are synthetic
/// (not present in the source parquet) and must be filled with a length-correct
/// placeholder array when building `RowAccess` columns.
fn resolve_node_id_indices(runtime: &[ResolveColRuntime]) -> std::collections::HashSet<usize> {
    runtime.iter().map(|r| r.node_id_idx).collect()
}

/// Build the per-column `ArrayRef` list for `RowAccess`, fetching real columns
/// from `batch` by name and substituting a length-correct null placeholder for
/// synthetic resolution node-id columns (whose values come from the override
/// channel, not the parquet).
fn build_access_cols(
    batch: &arrow_array::RecordBatch,
    table: &TableSpec,
    resolve_node_ids: &std::collections::HashSet<usize>,
) -> Result<Vec<ArrayRef>> {
    let n = batch.num_rows();
    let mut cols = Vec::with_capacity(table.columns.len());
    for (col_idx, col) in table.columns.iter().enumerate() {
        if resolve_node_ids.contains(&col_idx) {
            // Synthetic resolution column — not in the parquet. The encoder
            // reads it from the override channel, so a null placeholder of the
            // correct length is sufficient (the array is never inspected).
            cols.push(arrow_array::new_null_array(&DataType::Null, n));
            continue;
        }
        let idx = batch
            .schema()
            .index_of(&col.name)
            .context("missing column in batch")?;
        cols.push(batch.column(idx).clone());
    }
    Ok(cols)
}

/// Compute the per-row resolution overrides for `row`, resolving the source PK
/// reference value(s) of each runtime spec to graph node-id(s) via `resolver`.
/// Returns a per-column override vector (length == `table.columns.len()`) with
/// `ResolveOverride::None` for every non-resolution column.
///
/// Resolution is membership-verified: a key absent from the parent table —
/// whether a genuine pkmap miss or an MPH false positive rejected by the
/// verifier — leaves a scalar column `None` (encoded as null) / drops the list
/// element, and is counted into `key_failures` (misses as lookup failures,
/// false positives as membership failures). Null / unparsable source values
/// are left null silently, as before — they are legitimate input, not orphans.
#[allow(clippy::too_many_arguments)]
fn resolve_overrides_for_row(
    access: &mut RowAccess<'_>,
    table: &TableSpec,
    runtime: &[ResolveColRuntime],
    resolver: &crate::mapping::RefResolver,
    num_columns: usize,
    scratch: &mut Vec<u8>,
    shard_buf: &mut Vec<u8>,
    key_failures: &mut KeyFailureState,
) -> Result<(Vec<diskgraph_rowpack::ResolveOverride>, bool)> {
    use crate::mapping::ResolveVerified;
    let mut overrides = vec![diskgraph_rowpack::ResolveOverride::None; num_columns];
    // Row has no trainable anchor / positives → caller skips the write.
    let mut drop_row = false;
    for spec in runtime {
        let shard_count = resolver.shard_count(&spec.dst_table).ok_or_else(|| {
            anyhow!(
                "resolution: pkmap for dst_table `{}` not loaded",
                spec.dst_table
            )
        })?;
        let record_failure =
            |key_failures: &mut KeyFailureState, outcome: ResolveVerified, key: &Key| {
                let context = format!(
                    "resolve:{}.{}_to_{}",
                    table.name, table.columns[spec.src_idx].name, spec.dst_table
                );
                let sample = if key_failures.samples.len() < 10 {
                    sample_key(key)
                } else {
                    None
                };
                match outcome {
                    ResolveVerified::Miss => record_key_lookup_failure(
                        &mut key_failures.lookup_failures,
                        &mut key_failures.samples,
                        &mut key_failures.by_context,
                        &context,
                        sample,
                    ),
                    ResolveVerified::Orphan => record_key_membership_failure(
                        &mut key_failures.membership_failures,
                        &mut key_failures.samples,
                        &mut key_failures.by_context,
                        &context,
                        sample,
                    ),
                    ResolveVerified::Resolved(_) => {}
                }
            };
        match spec.kind {
            crate::config::ResolveKind::Scalar => {
                let col = &table.columns[spec.src_idx];
                scratch.clear();
                let mut resolved = false;
                if key_from_access_into_buf(access, spec.src_idx, col, &spec.src_type, scratch)
                    .is_ok()
                {
                    let mut data: &[u8] = scratch.as_slice();
                    if let Ok(key) = read_key_from_buf_for_type(&mut data, &spec.src_type) {
                        let shard = shard_for_key_from_key(&key, shard_buf, shard_count)?;
                        match resolver.resolve_verified_in_shard(&spec.dst_table, shard, &key)? {
                            ResolveVerified::Resolved(node_id) => {
                                overrides[spec.node_id_idx] =
                                    diskgraph_rowpack::ResolveOverride::Scalar(node_id);
                                resolved = true;
                            }
                            // orphan → count; anchor row dropped below
                            outcome => record_failure(key_failures, outcome, &key),
                        }
                    }
                }
                // Anchor with no resolved node (orphan/null/unparsable) → drop row.
                if !resolved && spec.role == ResolveRole::Anchor {
                    drop_row = true;
                }
            }
            crate::config::ResolveKind::List => {
                let keys =
                    resolve_list_keys_from_access(access, spec.src_idx, &spec.src_type, scratch)?;
                let mut ids: Vec<i64> = Vec::with_capacity(keys.len());
                for key in &keys {
                    let shard = shard_for_key_from_key(key, shard_buf, shard_count)?;
                    match resolver.resolve_verified_in_shard(&spec.dst_table, shard, key)? {
                        ResolveVerified::Resolved(node_id) => ids.push(node_id as i64),
                        // orphan element → drop (mirrors null-element drop) + count
                        outcome => record_failure(key_failures, outcome, key),
                    }
                }
                // All positives orphaned (had keys, none resolved) → drop the row.
                // A source-empty list is kept (not an orphan); exclude never drops.
                if !keys.is_empty() && ids.is_empty() && spec.role == ResolveRole::Positives {
                    drop_row = true;
                }
                overrides[spec.node_id_idx] = diskgraph_rowpack::ResolveOverride::List(ids);
            }
        }
    }
    Ok((overrides, drop_row))
}

/// Coerce Uuid/Uuid256/Ip source data to VarBytes (String or Bytes target).
/// Separated from the hot path to avoid icache pollution.
#[cold]
fn key_from_access_into_buf(
    access: &mut RowAccess<'_>,
    idx: usize,
    col: &ColumnSpec,
    target_type: &IngestedType,
    out: &mut Vec<u8>,
) -> KeyParseResult<()> {
    let key_type = key_type_for_dtype(target_type).map_err(|_| KeyParseError::UnsupportedType)?;
    out.clear();
    match key_type {
        KeyType::Int64 => {
            let value = match *target_type {
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
                    let array = access.array(idx).map_err(|_| KeyParseError::TypeMismatch)?;
                    let bytes =
                        read_bytes(array, access.row()).map_err(|_| KeyParseError::TypeMismatch)?;
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
            match *target_type {
                IngestedType::Bytes => {
                    let array = access.array(idx).map_err(|_| KeyParseError::TypeMismatch)?;
                    let bytes =
                        read_bytes(array, access.row()).map_err(|_| KeyParseError::TypeMismatch)?;
                    write_bytes(bytes)?;
                }
                IngestedType::String | IngestedType::Decimal256 => {
                    access
                        .with_string(idx, |text| {
                            write_bytes(text.as_bytes())
                                .map_err(|_| anyhow!("key parse failed"))?;
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
            let uuid = match *target_type {
                IngestedType::Uuid => {
                    let value = access
                        .get_uuid(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    buf.copy_from_slice(&value);
                    buf
                }
                IngestedType::String | IngestedType::Bytes => {
                    let array = access.array(idx).map_err(|_| KeyParseError::TypeMismatch)?;
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
            let uuid = match *target_type {
                IngestedType::Uuid256 => {
                    let value = access
                        .get_uuid256(idx)
                        .map_err(|_| KeyParseError::TypeMismatch)?
                        .ok_or(KeyParseError::TypeMismatch)?;
                    buf.copy_from_slice(&value);
                    buf
                }
                IngestedType::String | IngestedType::Bytes => {
                    let array = access.array(idx).map_err(|_| KeyParseError::TypeMismatch)?;
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
            let ip = match *target_type {
                IngestedType::Ip => access
                    .get_ip(idx)
                    .map_err(|_| KeyParseError::TypeMismatch)?
                    .ok_or(KeyParseError::TypeMismatch)?,
                IngestedType::String | IngestedType::Bytes => {
                    let array = access.array(idx).map_err(|_| KeyParseError::TypeMismatch)?;
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
    _col: &ColumnSpec,
    target_type: &IngestedType,
) -> Result<()> {
    match key_type_for_dtype(target_type)? {
        KeyType::VarBytes => {
            let len = u16::try_from(key_bytes.len()).map_err(|_| anyhow!("temp key too large"))?;
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

/// Read a key from buffer using the target type directly.
/// Used in edges_map when FK keys were written using target_type (from FK→PK coercion).
fn read_key_from_buf_for_type(data: &mut &[u8], target_type: &IngestedType) -> Result<Key> {
    let key_type = match target_type {
        IngestedType::Int64
        | IngestedType::Int32
        | IngestedType::Int16
        | IngestedType::Int8
        | IngestedType::UInt64
        | IngestedType::UInt32
        | IngestedType::UInt16
        | IngestedType::UInt8
        | IngestedType::Float32
        | IngestedType::Float64
        | IngestedType::Decimal128 => KeyType::Int64,
        IngestedType::Decimal256 => KeyType::VarBytes,
        IngestedType::String | IngestedType::Bytes => KeyType::VarBytes,
        IngestedType::Uuid => KeyType::Uuid,
        IngestedType::Uuid256 => KeyType::Uuid256,
        IngestedType::Ip => KeyType::Ip,
        _ => {
            return Err(anyhow!(
                "unsupported key type for target: {:?}",
                target_type
            ))
        }
    };
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
    if bytes.len() > 21 {
        // +1 for potential sign
        return Err(KeyParseError::TooLarge);
    }
    if bytes.contains(&b'.') {
        return Err(KeyParseError::NotInteger);
    }

    // Handle negative sign
    let (negative, digits) = if bytes.first() == Some(&b'-') {
        (true, &bytes[1..])
    } else {
        (false, bytes)
    };

    if digits.is_empty() {
        // Just "-" is invalid
        return Err(KeyParseError::Empty);
    }

    let mut value: u64 = 0;
    for &b in digits {
        if !b.is_ascii_digit() {
            return Err(KeyParseError::NotBase10);
        }
        let digit = (b - b'0') as u64;
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(digit))
            .ok_or(KeyParseError::TooLarge)?;
    }

    if negative {
        // Convert to two's complement representation
        // This matches how negative i64 values are stored as u64 in PK keys
        // For example: -1 becomes 0xFFFFFFFFFFFFFFFF, -123 becomes (-123i64) as u64
        if value > (i64::MAX as u64) + 1 {
            // Magnitude too large for i64 (can't represent values < i64::MIN)
            return Err(KeyParseError::TooLarge);
        }
        // Negate using two's complement: -(value as i64) then cast back to u64
        // Special case: value == (i64::MAX as u64) + 1 means i64::MIN
        let signed = if value == (i64::MAX as u64) + 1 {
            i64::MIN
        } else {
            -(value as i64)
        };
        Ok(signed as u64)
    } else {
        Ok(value)
    }
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
    let int_out = if int_trimmed.is_empty() {
        "0"
    } else {
        int_trimmed
    };
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

fn record_key_membership_failure(
    total: &mut u64,
    samples: &mut Vec<String>,
    by_context: &mut HashMap<String, KeyFailureCounts>,
    context: &str,
    sample: Option<String>,
) {
    *total = total.saturating_add(1);
    let entry = by_context.entry(context.to_string()).or_default();
    entry.membership_failures = entry.membership_failures.saturating_add(1);
    if samples.len() < 10 {
        if let Some(sample) = sample {
            samples.push(format!("{context} (membership): {sample}"));
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

fn validate_source_schema(
    table: &TableSpec,
    schema: &Schema,
    synthetic_columns: &HashSet<String>,
) -> Result<()> {
    for column in &table.columns {
        if synthetic_columns.contains(&column.name) {
            continue;
        }
        let field_index = schema.index_of(&column.name).map_err(|_| {
            PublicIngestError::new(format!(
                "table {} source schema is missing column '{}'",
                table.name, column.name
            ))
        })?;
        let actual = schema.field(field_index).data_type();
        if *actual == DataType::Null {
            return Err(PublicIngestError::new(format!(
                "table {} column '{}' has Null data type in the source schema",
                table.name, column.name
            ))
            .into());
        }
    }
    Ok(())
}

fn validate_source_batch_values(
    table: &TableSpec,
    batch: &RecordBatch,
    synthetic_columns: &HashSet<String>,
) -> Result<()> {
    for column in &table.columns {
        if synthetic_columns.contains(&column.name) || column.ingested_dtype != IngestedType::Int64
        {
            continue;
        }
        let field_index = batch.schema().index_of(&column.name).map_err(|_| {
            PublicIngestError::new(format!(
                "table {} source batch is missing column '{}'",
                table.name, column.name
            ))
        })?;
        let array = batch.column(field_index);
        match array.data_type() {
            DataType::Decimal128(_, scale) if *scale == 0 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 array cast failed"))?;
                for row in 0..values.len() {
                    if !values.is_null(row) && i64::try_from(values.value(row)).is_err() {
                        return Err(PublicIngestError::int64_decimal_out_of_range(
                            Some(&table.name),
                            &column.name,
                        )
                        .into());
                    }
                }
            }
            DataType::Decimal256(_, scale) if *scale == 0 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 array cast failed"))?;
                for row in 0..values.len() {
                    if !values.is_null(row) && values.value(row).to_i64().is_none() {
                        return Err(PublicIngestError::int64_decimal_out_of_range(
                            Some(&table.name),
                            &column.name,
                        )
                        .into());
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
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
            timestamp_unit: col.timestamp_unit.map(|u| u.to_arrow()),
            decimal_scale: col.scale,
            binary_string_hint: match col.ingested_dtype {
                IngestedType::Uuid => BinaryStringHint::Uuid,
                IngestedType::Uuid256 => BinaryStringHint::Uuid256,
                IngestedType::Ip => BinaryStringHint::Ip,
                _ => BinaryStringHint::Utf8,
            },
            multicategorical_delimiter: col.multicategorical_delimiter.clone(),
            column_name: Some(col.name.clone()),
        })
        .collect()
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
            IngestedType::Int8Array => RowpackType::Int8Array,
            IngestedType::Int16Array => RowpackType::Int16Array,
            IngestedType::Int32Array => RowpackType::Int32Array,
            IngestedType::Int64Array => RowpackType::Int64Array,
            IngestedType::StringArray => RowpackType::StringArray,
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

fn read_utf8(array: &ArrayRef, row: usize) -> Result<&str> {
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

fn read_bytes(array: &ArrayRef, row: usize) -> Result<&[u8]> {
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
            let secs = timestamp_secs_from_array(array, row)?;
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

fn timestamp_secs_from_array(array: &ArrayRef, row: usize) -> Result<i64> {
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
            Ok(arr.value(row))
        }
        arrow_schema::TimeUnit::Millisecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                .ok_or_else(|| anyhow!("timestamp millisecond cast"))?;
            Ok(arr.value(row).div_euclid(1_000))
        }
        arrow_schema::TimeUnit::Microsecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                .ok_or_else(|| anyhow!("timestamp microsecond cast"))?;
            Ok(arr.value(row).div_euclid(1_000_000))
        }
        arrow_schema::TimeUnit::Nanosecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                .ok_or_else(|| anyhow!("timestamp nanosecond cast"))?;
            Ok(arr.value(row).div_euclid(1_000_000_000))
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

#[cfg(test)]
mod tests_cgroup {
    use super::{
        choose_ingest_num_shards, resolve_tightest_cpu_quota, resolve_tightest_mem_limit,
        table_sized_bytes,
    };
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    fn fake_reader(files: &[(&str, &str)]) -> impl Fn(&Path) -> Option<String> {
        let map: HashMap<PathBuf, String> = files
            .iter()
            .map(|(p, v)| (PathBuf::from(p), v.to_string()))
            .collect();
        move |p: &Path| map.get(p).cloned()
    }

    // The v2 self path is joined onto "/sys/fs/cgroup", so a self path of "/slice/leaf"
    // yields the level dirs .../slice/leaf, .../slice, then the mount root.

    #[test]
    fn mem_leaf_tighter_than_parent() {
        let read = fake_reader(&[
            ("/sys/fs/cgroup/slice/leaf/memory.max", "1000"),
            ("/sys/fs/cgroup/slice/memory.max", "5000"),
        ]);
        assert_eq!(
            resolve_tightest_mem_limit(Some("/slice/leaf"), &read),
            Some(1000)
        );
    }

    #[test]
    fn mem_parent_tighter_than_leaf() {
        let read = fake_reader(&[
            ("/sys/fs/cgroup/slice/leaf/memory.max", "5000"),
            ("/sys/fs/cgroup/slice/memory.max", "1000"),
        ]);
        assert_eq!(
            resolve_tightest_mem_limit(Some("/slice/leaf"), &read),
            Some(1000)
        );
    }

    #[test]
    fn mem_all_max_is_none() {
        let read = fake_reader(&[
            ("/sys/fs/cgroup/slice/leaf/memory.max", "max"),
            ("/sys/fs/cgroup/slice/memory.max", "max"),
            ("/sys/fs/cgroup/memory.max", "max"),
        ]);
        assert_eq!(resolve_tightest_mem_limit(Some("/slice/leaf"), &read), None);
    }

    #[test]
    fn mem_only_v2_root_set() {
        let read = fake_reader(&[("/sys/fs/cgroup/memory.max", "2048")]);
        assert_eq!(
            resolve_tightest_mem_limit(Some("/slice/leaf"), &read),
            Some(2048)
        );
    }

    #[test]
    fn mem_v1_fallback() {
        let read = fake_reader(&[("/sys/fs/cgroup/memory/memory.limit_in_bytes", "4096")]);
        assert_eq!(resolve_tightest_mem_limit(None, &read), Some(4096));
    }

    #[test]
    fn mem_unlimited_everywhere_is_none() {
        let read = fake_reader(&[]);
        assert_eq!(resolve_tightest_mem_limit(Some("/slice/leaf"), &read), None);
    }

    #[test]
    fn cpu_four_cores() {
        let read = fake_reader(&[("/sys/fs/cgroup/leaf/cpu.max", "400000 100000")]);
        assert_eq!(resolve_tightest_cpu_quota(Some("/leaf"), &read), Some(4));
    }

    #[test]
    fn cpu_max_quota_is_none() {
        let read = fake_reader(&[("/sys/fs/cgroup/leaf/cpu.max", "max 100000")]);
        assert_eq!(resolve_tightest_cpu_quota(Some("/leaf"), &read), None);
    }

    #[test]
    fn cpu_leaf_tighter_than_parent() {
        let read = fake_reader(&[
            ("/sys/fs/cgroup/slice/leaf/cpu.max", "200000 100000"),
            ("/sys/fs/cgroup/slice/cpu.max", "400000 100000"),
        ]);
        assert_eq!(
            resolve_tightest_cpu_quota(Some("/slice/leaf"), &read),
            Some(2)
        );
    }

    #[test]
    fn cpu_v1_fallback() {
        let read = fake_reader(&[
            ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "300000"),
            ("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "100000"),
        ]);
        assert_eq!(resolve_tightest_cpu_quota(None, &read), Some(3));
    }

    #[test]
    fn pinned_shard_count_cannot_weaken_ingest_memory_floor() {
        assert_eq!(choose_ingest_num_shards(None, 165), 165);
        assert_eq!(choose_ingest_num_shards(Some(43), 165), 165);
        assert_eq!(choose_ingest_num_shards(Some(256), 165), 256);
        assert_eq!(choose_ingest_num_shards(Some(0), 0), 1);
    }

    #[test]
    fn mixed_declared_and_local_sources_contribute_to_shard_sizing() {
        let declared = table_sized_bytes(Some(10), || 100);
        let local = table_sized_bytes(None, || 20);
        let zero_declared = table_sized_bytes(Some(0), || 30);
        assert_eq!(declared + local + zero_declared, 60);
    }
}
