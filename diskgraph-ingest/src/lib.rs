//! Diskgraph ingest: convert raw CSV/Parquet sources into index-ready artifacts.

pub mod binary_label;
pub mod coercion;
pub mod col_stats;
pub mod col_stats_export;
pub mod column_inference;
pub mod config;
pub mod config_converter;
pub mod consts;
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
    ColumnSpec, DirectionSpec, IngestedType, SemanticType, TableFormat, TableSpec, TimeSpec,
    WeightSpec,
};
pub use crate::io::Dest;
use crate::io::{
    build_input_transfer, download_cloud_bytes, is_cloud_path, require_local_output,
    resolve_local_paths, OutputSink, OutputTarget,
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
pub use crate::stats::{IngestReport, IngestStats, KeyFailureCounts, KeyFailureState, PerfStats};
use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, Schema};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, EdgeSortKeyMetadata,
    EdgeSortKeyTimestampEncoding, EdgeSortKeyTimestampKind, EdgeSortKeyTimestampMetadata,
    EdgeSortKeyWeightEncoding, EdgeSortKeyWeightKind, EdgeSortKeyWeightMetadata, EdgeStorageFormat,
    NodeDataCompression, NodeShardMetadata, NodeTypeMetadata, PrimaryKeyMap,
};
use diskgraph_fs::CloudTransfer;
use diskgraph_index::builder::{EdgeBufferPool as IndexEdgeBufferPool, MemoryBudget};
use diskgraph_index::graph::{GraphDataSource, GraphIndexBuilder, GraphProgressSink};
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
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pprof::protos::Message;
use pprof::ProfilerGuard;
use ptr_hash::DefaultPtrHash;
use ptr_hash::PtrHashScratch;
use serde_json::Value as JsonValue;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self as std_io, BufWriter, Cursor, Read, Write};
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid_simd::{parse_hyphenated, parse_simple, AsOut};

pub use graph_config::GraphConfig;

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
                    .get_paths()
                    .iter()
                    .filter(|path| !path.contains("://"))
                    .map(|path| crate::config_converter::inference::compute_local_dir_bytes(path))
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
    // Validate explicit multicategorical delimiters even on direct DiskGraph
    // configs (not just kumo→diskgraph converted ones). Pre-#28478 a wrong
    // delimiter would silently produce one-token tokenization downstream;
    // post-flip the same mistake bakes into the on-disk rowpack and is
    // irreversible without re-ingest. Convert the error into an anyhow form
    // here since `ingest_with_stats_from_source` already propagates anyhow.
    let transfer = match crate::io::build_input_transfer(&source) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                "could not build input transfer for delim validation: {e}"
            );
            None
        }
    };
    crate::config_converter::inference::validate_explicit_multicat_delimiters(
        &source, &transfer,
    )
    .map_err(|e| anyhow!("{e}"))?;
    source.validate()?;

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
        if rfp > IngestPhase::Context {
            ctx.repopulate_table_contexts_from_source()?;
        }
        if rfp > IngestPhase::Encode {
            ctx.repopulate_node_stats_from_meta()?;
        }
        if rfp > IngestPhase::PkMap {
            ctx.repopulate_pk_hash_paths_from_disk()?;
        }
    }

    // Wall-time durations for each phase. Skipped phases stay at zero —
    // they're consumed by PerfStats below regardless of which phases ran.
    let mut context_wall = std::time::Duration::ZERO;
    let mut encode_wall = std::time::Duration::ZERO;
    let mut pk_map_wall = std::time::Duration::ZERO;
    let mut edges_map_wall = std::time::Duration::ZERO;
    let mut index_wall = std::time::Duration::ZERO;

    ctx.set_peak_stage(Some(0));
    if should_run(IngestPhase::Context) {
        let context_start = std::time::Instant::now();
        run_phase_with_marker(
            &local_root,
            "context",
            wipe_on_entry(IngestPhase::Context),
            profile_dir.as_deref(),
            || ctx.build_node_contexts(),
        )?;
        context_wall = context_start.elapsed();
        tracing::info!(
            stage = "context",
            elapsed_ms = context_wall.as_millis() as u64,
            elapsed = %format_duration(context_wall.as_millis() as u64),
            "context stage complete"
        );
    } else {
        tracing::info!(stage = "context", "resume: skipped (marker present)");
    }
    ctx.set_peak_stage(Some(1));
    if should_run(IngestPhase::Encode) {
        let encode_start = std::time::Instant::now();
        run_phase_with_marker(
            &local_root,
            "encode",
            wipe_on_entry(IngestPhase::Encode),
            profile_dir.as_deref(),
            || ctx.build_temp_shards(),
        )?;
        encode_wall = encode_start.elapsed();
        tracing::info!(
            stage = "encode",
            elapsed_ms = encode_wall.as_millis() as u64,
            elapsed = %format_duration(encode_wall.as_millis() as u64),
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
            || ctx.build_sharded_pk_maps(),
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
    let edge_ctx = ctx.edge_build_context()?;
    ctx.set_peak_stage(Some(3));
    if should_run(IngestPhase::EdgesMap) {
        let edges_map_start = std::time::Instant::now();
        run_phase_with_marker(
            &local_root,
            "edges_map",
            wipe_on_entry(IngestPhase::EdgesMap),
            profile_dir.as_deref(),
            || ctx.build_edges_mapped(&edge_ctx),
        )?;
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
        let _ = ctx.cleanup_edge_stage("edges_src");
        let _ = ctx.cleanup_edge_stage("fkeys");
        let _ = ctx.cleanup_edge_stage("fkeys_dst");
        let _ = ctx.cleanup_edge_stage("edge_keys");
    } else {
        tracing::info!(stage = "edges_map", "resume: skipped (marker present)");
    }
    drop(edge_ctx);
    ctx.set_peak_stage(Some(4));
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
        // Marker is on disk. edges_map intermediate is no longer needed by any
        // phase; free it so disk pressure during finalize stays low.
        let _ = ctx.cleanup_edge_stage("edges_map");
    } else {
        tracing::info!(stage = "index", "resume: skipped (marker present)");
    }
    ctx.set_peak_stage(Some(5));
    let finalize_start = std::time::Instant::now();
    let index_metadata = ctx.finalize_metadata(index_metadata)?;
    if let OutputTarget::Local { root, .. } = &*ctx.target {
        let nodes_root = root.join(TEMP_DIR_NAME).join("nodes");
        if nodes_root.exists() {
            ctx.temp_node_data_bytes = sum_dir_bytes(&nodes_root).unwrap_or(0);
        }
        let edges_root = root.join(TEMP_DIR_NAME).join("edges_map");
        if edges_root.exists() {
            ctx.temp_edges_map_bytes = sum_dir_bytes(&edges_root).unwrap_or(0);
        }
    }
    // TT-NG timestamp grouping: merge the per-shard spills into per-stage
    // `bucket_index.bin` (final node-ids grouped by TIMESTAMP). Runs after
    // `finalize_metadata` (shard offsets / min-ids known) and BEFORE
    // `cleanup_temp` (spills live under `_tmp/`). No-op when grouping is off.
    let ts_groupings = ctx.run_ts_grouping()?;
    let sample_weight_indexes = ctx.run_sample_weight_indexes()?;
    let binary_label_indexes = ctx.run_binary_label_indexes()?;
    ctx.cleanup_temp()?;
    let finalize_wall = finalize_start.elapsed();
    ctx.set_peak_stage(None);
    tracing::info!(
        stage = "finalize",
        elapsed_ms = finalize_wall.as_millis() as u64,
        elapsed = %format_duration(finalize_wall.as_millis() as u64),
        "finalize stage complete"
    );
    let wall_time = wall_start.elapsed();
    let bytes_read = sum_input_bytes(&ctx.source);
    let bytes_written = match &*ctx.target {
        OutputTarget::Local { root, .. } => sum_dir_bytes(root).unwrap_or(0),
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
        source_total_bytes: Some(bytes_read),
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
    let report = IngestReport {
        stats: stats.clone(),
        source: ctx.source.clone(),
    };
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
        assert!(IngestPhase::Context < IngestPhase::Encode);
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
                IngestPhase::Context,
                IngestPhase::Encode,
                IngestPhase::PkMap,
                IngestPhase::EdgesMap,
            ]
        );
        let preds: Vec<_> = IngestPhase::PkMap.predecessors().collect();
        assert_eq!(preds, vec![IngestPhase::Context, IngestPhase::Encode]);
        let preds: Vec<_> = IngestPhase::Context.predecessors().collect();
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
        run_phase_with_marker(root, "context", false, None, || {
            called.set(true);
            Ok(())
        })
        .unwrap();
        assert!(called.get());
        assert!(phase_marker_exists(root, "context"));
    }

    #[test]
    fn run_phase_with_marker_no_marker_on_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let result = run_phase_with_marker(root, "context", false, None, || {
            Err(anyhow!("synthetic boom"))
        });
        assert!(result.is_err());
        assert!(!phase_marker_exists(root, "context"));
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
        write_phase_marker(root, "context").unwrap();
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
        write_phase_marker(root, "context").unwrap();
        // encode marker missing — pk_map should not be considered resumable
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

    fn node_table(name: &str, primary_key: Option<&str>) -> TableSpec {
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
            parallel: false,
            context_threads: None,
            encode_threads: None,
            pk_map_threads: None,
            edges_map_threads: None,
            index_threads: None,
            pk_map_estimate_sum: Arc::new(AtomicU64::new(0)),
            pk_map_peak_rss: Arc::new(AtomicU64::new(0)),
            index_mem_budget: None,
            num_shards,
            input_transfer: None,
            table_rows: HashMap::new(),
            table_contexts: HashMap::new(),
            shard_rows: HashMap::new(),
            shard_offsets: HashMap::new(),
            group_ts_counts: HashMap::new(),
            stage_sidecars: HashMap::new(),
            binary_label_counts: HashMap::new(),
            sample_weight_counts: HashMap::new(),
            pk_hash_paths: HashMap::new(),
            relation_defs: Vec::new(),
            relationships: Vec::new(),
            fk_edge_targets: HashMap::new(),
            edge_counts: HashMap::new(),
            edge_dst_degree_bits: HashMap::new(),
            pk_duplicate_counts: HashMap::new(),
            pk_duplicate_drops: HashMap::new(),
            pk_null_timestamp_drops: HashMap::new(),
            edge_null_timestamp_drops: HashMap::new(),
            pk_duplicate_samples: HashMap::new(),
            dropped_rows_counts: HashMap::new(),
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
            for (shard, &n) in counts[table.name.as_str()].iter().enumerate().take(num_shards) {
                write_temp_meta(
                    &target,
                    &temp_node_data_meta_path(&table.name, shard),
                    n,
                    0,
                )
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

#[derive(Clone)]

struct IngestContext {
    source: GraphConfig,
    target: std::sync::Arc<OutputTarget>,
    profile_dir: Option<PathBuf>,
    parallel: bool,
    /// Per-phase thread-count overrides. `None` falls back to
    /// `default_worker_threads()`.
    context_threads: Option<usize>,
    encode_threads: Option<usize>,
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
    input_transfer: Option<CloudTransfer>,
    table_rows: HashMap<String, u64>,
    table_contexts: HashMap<String, String>,
    shard_rows: HashMap<String, Vec<u64>>,
    shard_offsets: HashMap<String, Vec<u64>>,
    /// TT-NG timestamp grouping: per-(table,shard) distinct-ts → count, captured
    /// during encode from the node writer. Empty unless grouping is enabled.
    /// Consumed by the post-build merge to size buckets without a count pass.
    group_ts_counts: HashMap<(String, usize), HashMap<i64, u64>>,
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
    /// Per forward-relation bitvec of dst node IDs with nonzero degree.
    /// Used to compute NUM_DST_NONZERO_DEGREE for kumo edge stats.
    edge_dst_degree_bits: HashMap<String, Vec<u8>>,
    pk_duplicate_counts: HashMap<String, u64>,
    pk_duplicate_drops: HashMap<String, u64>,
    pk_null_timestamp_drops: HashMap<String, u64>,
    edge_null_timestamp_drops: HashMap<String, u64>,
    pk_duplicate_samples: HashMap<String, Vec<String>>,
    dropped_rows_counts: HashMap<String, u64>,
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

impl IngestContext {
    fn make_pool<S: Default + Send + 'static>(&self, phase: IngestPhase) -> TaskPool<S> {
        if !self.parallel {
            return TaskPool::new(1);
        }
        let n = match phase {
            IngestPhase::Context => self.context_threads,
            IngestPhase::Encode => self.encode_threads,
            IngestPhase::PkMap => self.pk_map_threads,
            IngestPhase::EdgesMap => self.edges_map_threads,
            IngestPhase::Index => self.index_threads,
        }
        .unwrap_or_else(default_worker_threads);
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
        let input_transfer = build_input_transfer(&source)?;
        Ok(Self {
            source,
            target,
            profile_dir: options.profile_dir,
            parallel: options.parallel,
            context_threads: options.context_threads,
            encode_threads: options.encode_threads,
            pk_map_threads: options.pk_map_threads,
            edges_map_threads: options.edges_map_threads,
            index_threads: options.index_threads,
            pk_map_estimate_sum: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            pk_map_peak_rss: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            index_mem_budget,
            num_shards,
            input_transfer,
            table_rows: HashMap::new(),
            table_contexts: HashMap::new(),
            shard_rows: HashMap::new(),
            shard_offsets: HashMap::new(),
            group_ts_counts: HashMap::new(),
            stage_sidecars,
            binary_label_counts: HashMap::new(),
            sample_weight_counts: HashMap::new(),
            pk_hash_paths: HashMap::new(),
            relation_defs,
            relationships,
            fk_edge_targets,
            edge_counts: HashMap::new(),
            edge_dst_degree_bits: HashMap::new(),
            pk_duplicate_counts: HashMap::new(),
            pk_duplicate_drops: HashMap::new(),
            pk_null_timestamp_drops: HashMap::new(),
            edge_null_timestamp_drops: HashMap::new(),
            pk_duplicate_samples: HashMap::new(),
            dropped_rows_counts: HashMap::new(),
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

    fn build_node_contexts(&mut self) -> Result<()> {
        type ContextResult = (
            String,
            String,
            Vec<SemanticType>,
            HashMap<String, crate::col_stats::TextSampleStats>,
        );
        let tables = self.source.tables.clone();
        let (tx, rx) = std::sync::mpsc::channel::<Result<ContextResult>>();
        let mut tasks = Vec::new();
        let source_rowpack = self.source.rowpack.clone();
        let input_transfer = self.input_transfer.clone();
        let col_stats_enabled = self.col_stats;
        let resolve_columns_all = self.resolve_columns.clone();
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
            // Synthetic resolution node-id column indices for THIS table (empty
            // unless resolution mode). Used to inject placeholder arrays for the
            // columns that are absent from the source parquet.
            let resolve_node_ids: std::collections::HashSet<usize> = resolve_columns_all
                .iter()
                .filter(|r| r.table == table.name)
                .filter_map(|r| column_index(&table, &r.node_id_col).ok())
                .collect();
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
                                    validate_parquet_column_types(
                                        builder.schema(),
                                        &table.columns,
                                        path,
                                    )?;
                                    let reader =
                                        builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                                    for batch in reader {
                                        let batch = batch?;
                                        let cols =
                                            build_access_cols(&batch, &table, &resolve_node_ids)?;
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
                                    validate_parquet_column_types(
                                        builder.schema(),
                                        &table.columns,
                                        path,
                                    )?;
                                    let reader =
                                        builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                                    for batch in reader {
                                        let batch = batch?;
                                        let cols =
                                            build_access_cols(&batch, &table, &resolve_node_ids)?;
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
                                        let cols =
                                            build_access_cols(&batch, &table, &resolve_node_ids)?;
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
                                        let cols =
                                            build_access_cols(&batch, &table, &resolve_node_ids)?;
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
                    // Infer semantic types from sample stats before finish() consumes the builder.
                    let unique_counts = node_builder.sample_unique_counts();
                    let fk_columns: HashSet<&str> = table
                        .fkeys
                        .iter()
                        .map(|fk| fk.src_column.as_str())
                        .collect();
                    let semantic_types: Vec<SemanticType> = table
                        .columns
                        .iter()
                        .zip(unique_counts.iter())
                        .map(|(col, &(num_unique, count))| {
                            // Respect explicit semantic_type from config.
                            if let Some(stype) = col.semantic_type {
                                return stype;
                            }
                            use crate::graph_config::{
                                infer_semantic_type, ColumnRole, ColumnSampleStats,
                            };
                            let role = ColumnRole {
                                is_pk: table
                                    .primary_key
                                    .as_deref()
                                    .is_some_and(|pk| pk == col.name),
                                is_fk: fk_columns.contains(col.name.as_str()),
                                name: &col.name,
                            };
                            let sample = if num_unique > 0 || count > 0 {
                                Some(ColumnSampleStats { num_unique, count })
                            } else {
                                None
                            };
                            infer_semantic_type(col, &role, sample.as_ref())
                        })
                        .collect();

                    // Compute text sample stats from context-phase sample rows.
                    // Read first parquet file(s) for text columns, up to 10K strings each.
                    let text_stats_map = if col_stats_enabled {
                        compute_context_text_stats(&table, &semantic_types, &input_transfer)?
                    } else {
                        HashMap::new()
                    };

                    let node_context = node_builder.finish()?;
                    let ctx_bytes = node_context.to_bytes()?;
                    tracing::debug!(
                        table = %table.name,
                        ctx_bytes_len = ctx_bytes.len(),
                        temp_path = %temp_node_context_path(&table.name),
                        "writing node_data.ctx",
                    );
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
                        semantic_types,
                        text_stats_map,
                    )))
                    .map_err(|e| anyhow!("context output error: {e}"))?;
                    Ok(())
                }),
            });
        }
        drop(tx);
        let task_count = tasks.len();
        let pool = self.make_pool::<()>(IngestPhase::Context);
        run_tasks(tasks, &pool)?;
        for _ in 0..task_count {
            let result = rx
                .recv()
                .map_err(|e| anyhow!("context output error: {e}"))??;
            let (table_name, ctx_path, semantic_types, text_stats) = result;
            self.table_contexts.insert(table_name.clone(), ctx_path);
            if !text_stats.is_empty() {
                self.text_sample_stats
                    .insert(table_name.clone(), text_stats);
            }
            // Populate semantic_type on each column.
            if let Some(table) = self.source.tables.iter_mut().find(|t| t.name == table_name) {
                for (col, stype) in table.columns.iter_mut().zip(semantic_types) {
                    col.semantic_type = Some(stype);
                }
            }
        }
        Ok(())
    }

    fn build_temp_shards(&mut self) -> Result<()> {
        let root = require_local_output(&self.target)?.to_path_buf();
        let temp_root = root.join(TEMP_DIR_NAME);
        fs::create_dir_all(&temp_root)?;
        let mut table_rows: HashMap<String, u64> = HashMap::new();
        let tables = self.source.tables.clone();
        struct EncodeResult {
            table: String,
            total_rows: u64,
            node_data_dropped: u64,
            key_failures: KeyFailureState,
            file_col_stats: Option<crate::col_stats::FileColStats>,
        }

        let (tx, rx) = std::sync::mpsc::channel::<Result<EncodeResult>>();
        let mut tasks = Vec::new();
        let relationships = self.relationships.clone();
        let relation_defs = std::sync::Arc::new(self.relation_defs.clone());
        let fk_edge_targets = std::sync::Arc::new(self.fk_edge_targets.clone());
        let num_shards = self.num_shards;
        let target = std::sync::Arc::clone(&self.target);
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
        let input_transfer = self.input_transfer.clone();
        let trimmed_strings = std::sync::Arc::clone(&self.trimmed_strings);
        let progress = self.progress.clone();
        let col_stats_enabled = self.col_stats;
        let cat_count_strategy = self.cat_count_strategy;
        let row_subsample = self.row_subsample;

        // Create shared category count state per table (if enabled).
        let table_cat_counts: HashMap<
            String,
            std::sync::Arc<crate::col_stats::TableCategoryCounts>,
        > = if cat_count_strategy != crate::col_stats::CategoryCountStrategy::None
            && col_stats_enabled
        {
            tables
                .iter()
                .filter(|t| t.edge_table.is_none()) // Only node tables
                .map(|t| {
                    let stypes: Vec<_> = t.columns.iter().map(|c| c.semantic_type).collect();
                    let cat = crate::col_stats::TableCategoryCounts::new(
                        cat_count_strategy,
                        &stypes,
                        row_subsample,
                    );
                    (t.name.clone(), std::sync::Arc::new(cat))
                })
                .collect()
        } else {
            HashMap::new()
        };

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
            let trimmed_strings = std::sync::Arc::clone(&trimmed_strings);
            let progress = progress.clone();
            let root = root.clone();
            let relation_defs = std::sync::Arc::clone(&relation_defs);
            let fk_edge_targets = std::sync::Arc::clone(&fk_edge_targets);
            let table_cat = table_cat_counts.get(&table.name).cloned();
            let resolve_columns = std::sync::Arc::clone(&resolve_columns);
            let ref_resolver = ref_resolver.clone();
            let sidecars = self
                .stage_sidecars
                .get(&table.name)
                .cloned()
                .unwrap_or_default();
            let file_count = table.get_paths().len();
            for file_idx in 0..file_count {
                let table = table.clone();
                let sidecars = sidecars.clone();
                let relationships = relationships.clone();
                let target = std::sync::Arc::clone(&target);
                let node_senders = std::sync::Arc::clone(&node_senders);
                let node_errors = std::sync::Arc::clone(&node_errors);
                let edge_senders = std::sync::Arc::clone(&edge_senders);
                let dest_key_senders = std::sync::Arc::clone(&dest_key_senders);
                let dest_key_errors = std::sync::Arc::clone(&dest_key_errors);
                let trimmed_strings = std::sync::Arc::clone(&trimmed_strings);
                let progress = progress.clone();
                let root = root.clone();
                let relation_defs = std::sync::Arc::clone(&relation_defs);
                let fk_edge_targets = std::sync::Arc::clone(&fk_edge_targets);
                let input_transfer = input_transfer.clone();
                let tx = tx.clone();
                let edge_errors = std::sync::Arc::clone(&edge_errors);
                let table_cat = table_cat.clone();
                let resolve_columns = std::sync::Arc::clone(&resolve_columns);
                let ref_resolver = ref_resolver.clone();
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
                                relation_defs.as_ref(),
                                &fk_edge_targets,
                            )?;
                            tx.send(Ok(EncodeResult {
                                table: table.name.clone(),
                                total_rows: 0,
                                node_data_dropped,
                                key_failures,
                                file_col_stats: None,
                            }))
                            .map_err(|e| anyhow!("encode output error: {e}"))?;
                            return Ok(());
                        }
                        let (node, file_stats) = if col_stats_enabled {
                            let mut obs = if let Some(ref tc) = table_cat {
                                crate::col_stats::FileColStats::new_with_categories(
                                    &table.columns,
                                    std::sync::Arc::clone(tc),
                                )
                            } else {
                                crate::col_stats::FileColStats::new(&table.columns)
                            };
                            let result = Self::build_temp_node_file(
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
                                &node_errors,
                                &dest_key_senders,
                                &dest_key_errors,
                                &fk_edge_targets,
                                &resolve_columns,
                                ref_resolver.as_deref(),
                                &sidecars,
                                &mut obs,
                            )?;
                            // Flush local category counts to shared state (Approach 3)
                            obs.flush_categories();
                            (result, Some(obs))
                        } else {
                            let result = Self::build_temp_node_file(
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
                                &node_errors,
                                &dest_key_senders,
                                &dest_key_errors,
                                &fk_edge_targets,
                                &resolve_columns,
                                ref_resolver.as_deref(),
                                &sidecars,
                                &mut diskgraph_rowpack::NoOpObserver,
                            )?;
                            (result, None)
                        };
                        tx.send(Ok(EncodeResult {
                            table: table.name.clone(),
                            total_rows: node.total_rows,
                            node_data_dropped: node.node_data_dropped,
                            key_failures,
                            file_col_stats: file_stats,
                        }))
                        .map_err(|e| anyhow!("encode output error: {e}"))?;
                        Ok(())
                    }),
                });
            }
        }
        drop(tx);
        let task_count = tasks.len();
        let pool = self.make_pool::<()>(IngestPhase::Encode);
        run_tasks(tasks, &pool)?;
        // Per-table col stats accumulator (Option C: merge on receive side)
        let mut table_col_stats: HashMap<String, crate::col_stats::FileColStats> = HashMap::new();
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
                let entry = self
                    .dropped_rows_counts
                    .entry(result.table.clone())
                    .or_insert(0);
                *entry = entry.saturating_add(result.node_data_dropped);
            }
            self.merge_key_failures(result.key_failures);
            // Merge per-file col stats into per-table accumulator
            if let Some(mut file_stats) = result.file_col_stats {
                table_col_stats
                    .entry(result.table.clone())
                    .and_modify(|acc| acc.merge(&mut file_stats))
                    .or_insert(file_stats);
            }
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
        let node_stats = node_pool.finish()?;
        let edge_stats = edge_pool.finish()?;
        let dest_key_stats = dest_key_pool.finish()?;
        self.write_consolidated_meta(&node_stats, &edge_stats, &dest_key_stats)?;
        self.refresh_shard_rows_from_stats(&node_stats);
        self.table_rows = table_rows;
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
        for table in &self.source.tables {
            if table.edge_table.is_some() {
                continue;
            }
            let mut shard_rows = vec![0u64; self.num_shards];
            let mut total = 0u64;
            for (shard, slot) in shard_rows.iter_mut().enumerate() {
                let meta_path = root.join(temp_node_data_meta_path(&table.name, shard));
                if !meta_path.exists() {
                    continue;
                }
                let meta = read_temp_meta(&meta_path)?;
                *slot = meta.row_count;
                total = total.saturating_add(meta.row_count);
            }
            self.shard_rows.insert(table.name.clone(), shard_rows);
            self.table_rows.insert(table.name.clone(), total);
        }
        tracing::info!(
            tables = self.shard_rows.len(),
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
        // TT-NG timestamp grouping: carry the per-(table,shard) count maps from
        // the writer into the context so the post-build merge can size buckets.
        // Empty unless grouping is enabled (no allocation in the common path).
        if !stats.group_ts_counts.is_empty() {
            self.group_ts_counts = stats.group_ts_counts.clone();
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
    fn build_temp_node_file<O: diskgraph_rowpack::ColumnValueObserver>(
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
        node_errors: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        dest_key_senders: &std::sync::Arc<Vec<std::sync::mpsc::SyncSender<DestKeyWriteMessage>>>,
        dest_key_errors: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
        fk_edge_targets: &HashMap<String, FkEdgeTarget>,
        resolve_columns: &[crate::config::ResolveColumnSpec],
        ref_resolver: Option<&crate::mapping::RefResolver>,
        sidecars: &StageSidecarColumns,
        observer: &mut O,
    ) -> Result<TempNodeFileResult> {
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
        let ctx_path = root.join(temp_node_context_path(&table.name));
        let ctx_file_len = fs::metadata(&ctx_path).map(|m| m.len()).unwrap_or(0);
        tracing::debug!(
            table = %table.name,
            path = %ctx_path.display(),
            file_len = ctx_file_len,
            "reading node_data.ctx",
        );
        let ctx_bytes = fs::read(&ctx_path)
            .with_context(|| format!("read {} (file_len={})", ctx_path.display(), ctx_file_len,))?;
        let node_context = diskgraph_rowpack::RowPackContext::from_bytes(&ctx_bytes)?;
        let node_encoder = Some(RowPackEncoder::new(node_context)?);
        let mut key_bufs: Vec<Vec<u8>> = vec![Vec::new(); table.columns.len()];
        let mut f16_buf: Vec<f16> = Vec::new();
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
                for (_, idx, _, _, target_type) in &rel_specs {
                    let array = access.array(*idx).unwrap();
                    let kind = edge_fast_path_for_array(array, target_type);
                    let should_optimize = match kind {
                        Some(EdgeKeyFastKind::Int64ToInt32)
                        | Some(EdgeKeyFastKind::StringToInt64)
                        | Some(EdgeKeyFastKind::StringUuid)
                        | Some(EdgeKeyFastKind::StringUuid256)
                        | Some(EdgeKeyFastKind::StringIp) => true,
                        _ => false,
                    };
                    if should_optimize {
                        let keys_result = extract_batch_keys(array, kind.unwrap());
                        fkeys_pre.push(keys_result.ok());
                    } else {
                        fkeys_pre.push(None);
                    }
                }

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
                    let shard_override = precomputed.as_ref().and_then(|p| p.shards[row]);
                    let pk_pre = precomputed
                        .as_ref()
                        .and_then(|p| p.keys.as_ref())
                        .and_then(|k| k[row].as_ref());

                    // FK-free LP resolution: resolve the source PK reference
                    // column(s) to graph node-id(s) for this row, producing a
                    // per-column override vector. Empty when not in resolve mode.
                    let resolve_overrides = if resolve_runtime.is_empty() {
                        None
                    } else {
                        let resolver = ref_resolver.ok_or_else(|| {
                            anyhow!("resolution mode: RefResolver not provided to encode worker")
                        })?;
                        Some(resolve_overrides_for_row(
                            &mut access,
                            table,
                            &resolve_runtime,
                            resolver,
                            num_columns,
                            &mut resolve_scratch,
                            &mut resolve_shard_buf,
                            key_failures,
                        )?)
                    };

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

        match table.format() {
            TableFormat::Parquet => {
                if let Some(transfer) = &input_transfer {
                    if is_cloud_path(path) {
                        let pooled = download_cloud_bytes(transfer, path)?;
                        let bytes = pooled.bytes();
                        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
                        validate_parquet_column_types(builder.schema(), &table.columns, path)?;
                        let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                        for batch in reader {
                            let batch = batch?;
                            let cols = build_access_cols(&batch, table, &resolve_node_ids)?;
                            access.set_cols(&cols);
                            process_node_batch!(batch);
                        }
                        drop(pooled);
                    } else {
                        let file =
                            fs::File::open(path).with_context(|| format!("open {}", path))?;
                        advise_file_willneed(&file);
                        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                        validate_parquet_column_types(builder.schema(), &table.columns, path)?;
                        let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                        for batch in reader {
                            let batch = batch?;
                            let cols = build_access_cols(&batch, table, &resolve_node_ids)?;
                            access.set_cols(&cols);
                            process_node_batch!(batch);
                        }
                    }
                } else {
                    let file = fs::File::open(path).with_context(|| format!("open {}", path))?;
                    advise_file_willneed(&file);
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    validate_parquet_column_types(builder.schema(), &table.columns, path)?;
                    let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                    for batch in reader {
                        let batch = batch?;
                        let cols = build_access_cols(&batch, table, &resolve_node_ids)?;
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
                            let cols = build_access_cols(&batch, table, &resolve_node_ids)?;
                            access.set_cols(&cols);
                            process_node_batch!(batch);
                        }
                        drop(pooled);
                    } else {
                        let file =
                            fs::File::open(path).with_context(|| format!("open {}", path))?;
                        advise_file_willneed(&file);
                        let builder =
                            arrow_csv::ReaderBuilder::new(Arc::new(schema)).with_header(true);
                        let reader = builder.build(file)?;
                        for batch in reader {
                            let batch = batch?;
                            let cols = build_access_cols(&batch, table, &resolve_node_ids)?;
                            access.set_cols(&cols);
                            process_node_batch!(batch);
                        }
                    }
                } else {
                    let file = fs::File::open(path).with_context(|| format!("open {}", path))?;
                    advise_file_willneed(&file);
                    let builder = arrow_csv::ReaderBuilder::new(Arc::new(schema)).with_header(true);
                    let reader = builder.build(file)?;
                    for batch in reader {
                        let batch = batch?;
                        let cols = build_access_cols(&batch, table, &resolve_node_ids)?;
                        access.set_cols(&cols);
                        process_node_batch!(batch);
                    }
                }
            }
        }
        batcher.flush_all(&mut sort_key_batchers)?;
        dest_fkey_batcher.flush_all()?;
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
            let res = if let Some(key) = pk_pre {
                write_key_to_buf(buf, key).map_err(|_| KeyParseError::TypeMismatch)
            } else {
                key_from_access_into_buf(access, idx, pk_col, &pk_col.ingested_dtype, buf)
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
        for (i, (_rel_name, rel_idx, _rel_pos, _fk_col_name, target_type)) in
            rel_specs.iter().enumerate()
        {
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
                    key_from_access_into_buf(access, *rel_idx, col, target_type, buf)
                }
            } else {
                key_from_access_into_buf(access, *rel_idx, col, target_type, buf)
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
        let node_tag_start;
        let node_bool_start;
        let node_data_start;
        {
            let sink = batcher.node_sink(shard)?;
            node_tag_start = sink.tags.len();
            node_bool_start = sink.bools.len();
            node_data_start = sink.data.len();
            let lens = encoder.encode_row_observed(
                access,
                sink.tags,
                sink.bools,
                sink.data,
                f16_buf,
                key_bufs,
                resolve_overrides,
                observer,
            )?;
            // Drop row if time_col was encoded as null (tag bits == 0).
            // This catches: Arrow null that slipped past Phase 2, string parse
            // failures, and out-of-range dates (Change A in access.rs).
            if let Some(ts_idx) = create_time_col_idx {
                let bit_offset = ts_idx * 2; // 2 tag bits per column
                let byte_idx = node_tag_start + (bit_offset / 8);
                let bit_pos = (bit_offset % 8) as u8;
                if byte_idx < sink.tags.len() {
                    let tag_val = (sink.tags[byte_idx] >> bit_pos) & 0b11;
                    if tag_val == 0 {
                        sink.tags.truncate(node_tag_start);
                        sink.bools.truncate(node_bool_start);
                        sink.data.truncate(node_data_start);
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
                sink.tags.truncate(node_tag_start);
                sink.bools.truncate(node_bool_start);
                sink.data.truncate(node_data_start);
                *node_data_dropped = node_data_dropped.saturating_add(1);
                batcher.push_empty_node(shard)?;
            } else {
                batcher.push_node_row(shard, lens)?;
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
        relation_defs: &[RelationDef],
        fk_edge_targets: &HashMap<String, FkEdgeTarget>,
    ) -> Result<u64> {
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
        let mut sort_key_batchers =
            build_sort_key_batchers_for_edge_table(table, relation_defs, num_shards)?;

        let create_time_info = table.create_time.as_ref().and_then(|ct| {
            let idx = column_index(table, ct).ok()?;
            let spec = column_spec(table, ct).ok()?;
            Some((idx, spec.ingested_dtype.clone()))
        });

        let mut edge_dropped = 0u64;

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
                    let src_key_pre = src_keys_pre.as_ref().and_then(|v| v[row].as_ref());
                    let dst_key_pre = dst_keys_pre.as_ref().and_then(|v| v[row].as_ref());

                    Self::write_edge_row_from_access(
                        src_fk_idx,
                        dst_fk_idx,
                        src_fk_col,
                        dst_fk_col,
                        src_target_type,
                        dst_target_type,
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
                        validate_parquet_column_types(builder.schema(), &table.columns, path)?;
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
                            let src_fast =
                                edge_fast_path_for_array(&cols[src_fk_idx], src_target_type);
                            let dst_fast =
                                edge_fast_path_for_array(&cols[dst_fk_idx], dst_target_type);
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
                        let file =
                            fs::File::open(path).with_context(|| format!("open {}", path))?;
                        advise_file_willneed(&file);
                        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                        validate_parquet_column_types(builder.schema(), &table.columns, path)?;
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
                            let src_fast =
                                edge_fast_path_for_array(&cols[src_fk_idx], src_target_type);
                            let dst_fast =
                                edge_fast_path_for_array(&cols[dst_fk_idx], dst_target_type);
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
                    let file = fs::File::open(path).with_context(|| format!("open {}", path))?;
                    advise_file_willneed(&file);
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    validate_parquet_column_types(builder.schema(), &table.columns, path)?;
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
                        let src_fast = edge_fast_path_for_array(&cols[src_fk_idx], src_target_type);
                        let dst_fast = edge_fast_path_for_array(&cols[dst_fk_idx], dst_target_type);
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
            TableFormat::Csv => {
                let builder =
                    arrow_csv::ReaderBuilder::new(Arc::new(schema.clone())).with_header(true);
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
                            let src_fast =
                                edge_fast_path_for_array(&cols[src_fk_idx], src_target_type);
                            let dst_fast =
                                edge_fast_path_for_array(&cols[dst_fk_idx], dst_target_type);
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
                        let file =
                            fs::File::open(path).with_context(|| format!("open {}", path))?;
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
                            let src_fast =
                                edge_fast_path_for_array(&cols[src_fk_idx], src_target_type);
                            let dst_fast =
                                edge_fast_path_for_array(&cols[dst_fk_idx], dst_target_type);
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
                    let file = fs::File::open(path).with_context(|| format!("open {}", path))?;
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
                        let src_fast = edge_fast_path_for_array(&cols[src_fk_idx], src_target_type);
                        let dst_fast = edge_fast_path_for_array(&cols[dst_fk_idx], dst_target_type);
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
        batcher.flush_all(&mut sort_key_batchers)?;
        edge_key_batcher.flush_all()?;
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
        } else if key_from_access_into_buf(access, src_idx, src_col, src_target_type, src_key_buf)
            .is_err()
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
        } else if key_from_access_into_buf(access, dst_idx, dst_col, dst_target_type, dst_key_buf)
            .is_err()
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
                                    let params = ptr_hash::PtrHashParams::default();
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
        let pool = self.make_pool::<PkMapScratch>(IngestPhase::PkMap);
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
        let pool = self.make_pool::<()>(IngestPhase::EdgesMap);
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
        let pool = self.make_pool::<()>(IngestPhase::EdgesMap);
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

        let pool = self.make_pool::<EdgeMapScratch>(IngestPhase::EdgesMap);
        {
            #[derive(Default)]
            struct EdgeMapResult {
                relation: String,
                key_failures: KeyFailureState,
                edge_counts: HashMap<String, u64>,
                pk_duplicate_drops: u64,
                null_timestamp_drops: u64,
                /// Forward relation name → bitvec tracking unique dst node IDs.
                dst_degree_bits: HashMap<String, Vec<u8>>,
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
                            let mut dst_bits = vec![0u8; ((dst_max + 8) / 8) as usize];
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
                                .insert(rel.name_forward.clone(), dst_bits);
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
                for (rel, bits) in result.dst_degree_bits {
                    let merged = self.edge_dst_degree_bits.entry(rel).or_default();
                    if merged.is_empty() {
                        *merged = bits;
                    } else {
                        for (i, b) in bits.iter().enumerate() {
                            if i < merged.len() {
                                merged[i] |= b;
                            }
                        }
                    }
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
        if self.group_ts_counts.is_empty() {
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
                let key = (table.name.clone(), shard);
                let Some(count_map) = self.group_ts_counts.get(&key) else {
                    continue;
                };
                let spill_path = root.join(crate::ts_grouping::spill_rel_path(&table.name, shard));
                if !spill_path.exists() {
                    continue;
                }
                shards.push(crate::ts_grouping::ShardSpill {
                    min_id: offsets.get(shard).copied().unwrap_or(0),
                    count_map: count_map.clone(),
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
                                if let Err(e) = std::fs::remove_dir_all(&p) {
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
        let jobs = builder.prepare_jobs_with_pool(edge_buffer_pool)?;
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
                run: Box::new(move |_scratch| {
                    let progress = adapter
                        .as_ref()
                        .map(|value| value.as_ref() as &dyn GraphProgressSink);

                    job.run_with_budget(progress, mem_budget).map(|_| ())
                }),
            });
        }
        let pool = self.make_pool::<()>(IngestPhase::Index);
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

        // Persist the resolved graph config so downstream consumers (TT-gen,
        // BP) can recover per-column semantic_type without re-supplying it
        // via plan JSON each query (kumo#28400 follow-up). `self.source` has
        // semantic_type populated for every column by the context-build
        // stage (see populate_table_contexts at lib.rs:1998).
        let config_json = serde_json::to_vec_pretty(&self.source)?;
        self.target
            .write_bytes(graph_config::DISKGRAPH_CONFIG_FILENAME, config_json)?;
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

            // Unique dst from bitvec popcount
            let num_dst_nonzero = self
                .edge_dst_degree_bits
                .get(&rel.name_forward)
                .map(|bits| bits.iter().map(|b| b.count_ones() as u64).sum::<u64>())
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

fn compute_context_text_stats(
    table: &crate::graph_config::TableSpec,
    semantic_types: &[SemanticType],
    input_transfer: &Option<CloudTransfer>,
) -> Result<HashMap<String, crate::col_stats::TextSampleStats>> {
    // Find text column indices and names.
    let text_cols: Vec<(usize, &str)> = table
        .columns
        .iter()
        .zip(semantic_types.iter())
        .enumerate()
        .filter(|(_, (_, stype))| **stype == SemanticType::Text)
        .map(|(i, (col, _))| (i, col.name.as_str()))
        .collect();

    if text_cols.is_empty() {
        return Ok(HashMap::new());
    }

    // Buffer: col_idx -> Vec<String>
    let mut buffers: Vec<Vec<String>> = vec![Vec::new(); text_cols.len()];

    // Decide how many files to read and rows per file:
    //   1 file  → 10K from it
    //   2 files → 5K each
    //   3+ files → 4K, 3K, 3K
    let all_paths = table.get_paths();
    let num_files = all_paths.len();
    let file_limits: Vec<(&str, usize)> = if num_files == 1 {
        vec![(all_paths[0].as_str(), TEXT_SAMPLE_LIMIT)]
    } else if num_files == 2 {
        vec![
            (all_paths[0].as_str(), 5_000),
            (all_paths[1].as_str(), 5_000),
        ]
    } else {
        vec![
            (all_paths[0].as_str(), 4_000),
            (all_paths[1].as_str(), 3_000),
            (all_paths[2].as_str(), 3_000),
        ]
    };

    for (path, row_limit) in &file_limits {
        let mut rows_read = 0usize;
        match table.format() {
            TableFormat::Parquet => {
                let pooled = if let Some(transfer) = input_transfer {
                    if is_cloud_path(path) {
                        Some(download_cloud_bytes(transfer, path)?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                macro_rules! read_batches {
                    ($reader:expr) => {
                        for batch in $reader {
                            let batch = batch?;
                            for (buf_idx, &(col_idx, _)) in text_cols.iter().enumerate() {
                                let col_name = &table.columns[col_idx].name;
                                let arr_idx = batch
                                    .schema()
                                    .index_of(col_name)
                                    .context("missing column in batch")?;
                                let array = batch.column(arr_idx);
                                for row in 0..batch.num_rows() {
                                    if rows_read + row >= *row_limit {
                                        break;
                                    }
                                    if array.is_null(row) {
                                        continue;
                                    }
                                    if let Some(s) =
                                        array.as_any().downcast_ref::<arrow_array::StringArray>()
                                    {
                                        buffers[buf_idx].push(s.value(row).to_string());
                                    } else if let Some(s) = array
                                        .as_any()
                                        .downcast_ref::<arrow_array::LargeStringArray>(
                                    ) {
                                        buffers[buf_idx].push(s.value(row).to_string());
                                    }
                                }
                            }
                            rows_read += batch.num_rows();
                            if rows_read >= *row_limit {
                                break;
                            }
                        }
                    };
                }

                if let Some(pooled) = pooled {
                    let bytes = pooled.bytes();
                    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
                    let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                    read_batches!(reader);
                } else {
                    let file = fs::File::open(path).with_context(|| format!("open {}", path))?;
                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                    let reader = builder.with_batch_size(NODE_BATCH_SIZE).build()?;
                    read_batches!(reader);
                }
            }
            TableFormat::Csv => {
                break;
            }
        }
    }

    // Compute text sample stats for each text column.
    let mut result = HashMap::new();
    for (buf_idx, &(_, col_name)) in text_cols.iter().enumerate() {
        let refs: Vec<&str> = buffers[buf_idx].iter().map(|s| s.as_str()).collect();
        if !refs.is_empty() {
            let stats = crate::col_stats::compute_text_sample_stats(&refs);
            result.insert(col_name.to_string(), stats);
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

/// Write the phase marker atomically: tmp file + rename + best-effort dir fsync.
fn write_phase_marker(root: &Path, name: &str) -> Result<()> {
    let final_path = phase_marker_path(root, name);
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = final_path.with_extension("inprogress");
    fs::write(&tmp_path, b"")
        .with_context(|| format!("write phase marker tmp: {}", tmp_path.display()))?;
    fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("rename phase marker: {}", final_path.display()))?;
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
/// `encode` is special-cased: it shares `_tmp/nodes/` with `context` (context
/// writes `.ctx` files there). Encode's wipe must preserve `.ctx` files but
/// remove all other files inside `_tmp/nodes/` (encode's `.bin`/`.meta`).
fn wipe_phase_outputs(root: &Path, phase: &str) -> Result<()> {
    // Remove stale marker if present (we are re-running this phase).
    let _ = fs::remove_file(phase_marker_path(root, phase));

    match phase {
        // Context has no exclusive outputs — its `.ctx` files live in
        // `_tmp/nodes/<table>/shard-N/` which is shared with encode.
        // If we ever resume "from context" we're effectively running
        // from scratch; the top-level wipe (which fired because
        // resume_from_phase == Context implies nothing valid on disk
        // yet) is sufficient.
        "context" => Ok(()),

        // Encode shares `_tmp/nodes/` with context. Wipe all of encode's
        // siblings wholesale; in `_tmp/nodes/`, remove everything except
        // `*.ctx` (which belongs to context).
        "encode" => {
            for d in [
                "_tmp/fkeys",
                "_tmp/fkeys_dst",
                "_tmp/edge_keys",
                "_tmp/edges_src",
                "_tmp/sort_keys",
            ] {
                let p = root.join(d);
                if p.exists() {
                    fs::remove_dir_all(&p)
                        .with_context(|| format!("wipe encode output: {}", p.display()))?;
                }
            }
            let nodes_root = root.join(TEMP_DIR_NAME).join("nodes");
            if nodes_root.exists() {
                wipe_dir_keep_ext(&nodes_root, "ctx")?;
            }
            Ok(())
        }

        // pk_map writes pk_hash-NNNNN.bin / pk_data-NNNNN.bin /
        // pk_hash-NNNNN.meta.json into nodes/<table>/. Preserve siblings
        // (node_data.ctx from context, shard-NNNNN/ from index).
        "pk_map" => {
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
                        for f in ["offsets.bin", "edges.bin", "build_stats.json"] {
                            let p = shard_dir.join(f);
                            if p.exists() {
                                fs::remove_file(&p)
                                    .with_context(|| format!("remove {}", p.display()))?;
                            }
                        }
                    }
                }
            }
            Ok(())
        }

        _ => Ok(()),
    }
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

/// Recursively walk `root`, removing every regular file whose extension is
/// not `keep_ext`. Empty directories are left in place (they're cheap and
/// the next phase recreates them on demand).
fn wipe_dir_keep_ext(root: &Path, keep_ext: &str) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root).with_context(|| format!("read_dir {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            wipe_dir_keep_ext(&path, keep_ext)?;
        } else if ft.is_file() || ft.is_symlink() {
            let keep = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == keep_ext)
                .unwrap_or(false);
            if !keep {
                fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
            }
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
    if wipe_on_entry {
        wipe_phase_outputs(root, name)?;
    }
    if let Some(dir) = profile_dir {
        profile_phase(dir, name, f)?;
    } else {
        f()?;
    }
    write_phase_marker(root, name)?;
    Ok(())
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

fn open_fkey_bitmap_writer(target: &OutputTarget, rel_path: &str) -> Result<BufWriter<OutputSink>> {
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
    // TT-NG timestamp grouping: per-(table,shard) distinct-timestamp → count.
    // O(D) per shard. Empty unless grouping is enabled for the table. Used by
    // the post-build merge to size buckets without a count pass.
    group_ts_counts: HashMap<(String, usize), HashMap<i64, u64>>,
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
        for (key, counts) in other.group_ts_counts {
            let entry = self.group_ts_counts.entry(key).or_default();
            for (ts, c) in counts {
                let slot = entry.entry(ts).or_insert(0);
                *slot = slot.saturating_add(c);
            }
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
        let sink = self
            .writer
            .into_inner()
            .map_err(|_| anyhow!("bit writer flush failed"))?;
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
        if let Ok(guard) = self.error.lock() {
            if let Some(message) = guard.as_ref() {
                return Err(anyhow!("node writer failed: {message}"));
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
                // timestamps (node-write order) to the shard spill, and fold
                // them into the per-shard count map. The spill's record order IS
                // the node order, so the k-th record's node-id is min_id + k.
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
                    let counts = stats
                        .group_ts_counts
                        .entry((batch.table.to_string(), batch.shard))
                        .or_default();
                    for ts in group_ts {
                        *counts.entry(*ts).or_insert(0) += 1;
                    }
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

struct NodeRowSink<'a> {
    tags: &'a mut Vec<u8>,
    bools: &'a mut Vec<u8>,
    data: &'a mut Vec<u8>,
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

    fn maybe_flush(
        &mut self,
        shard: usize,
        sort_key_batchers: &mut [SortKeyComponentBatcher],
    ) -> Result<()> {
        let buffer = self
            .buffers
            .get(shard)
            .ok_or_else(|| anyhow!("node buffer shard {} out of bounds", shard))?;
        if buffer.bytes >= self.buffer_limit {
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
        // Drain this shard's grouping timestamps (node-write order) into the
        // batch, leaving an empty Vec to keep accumulating. `None` when
        // grouping is disabled.
        let group_ts = buffer.group_ts.as_mut().map(std::mem::take);
        let binary_labels = buffer.binary_labels.as_mut().map(std::mem::take);
        let sample_weights = buffer.sample_weights.as_mut().map(std::mem::take);
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
        buffer.node.rows.clear();
        buffer.pkey_rows = 0;
        buffer.fkey_rows = vec![0; self.rel_names.len()];
        buffer.fkeys = vec![Vec::new(); self.rel_names.len()];
        buffer.fkey_bits = vec![Vec::new(); self.rel_names.len()];
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
        push_u64_width(buffer, value, self.width_bytes);
        if let Some(row_count) = self.rows.get_mut(shard) {
            *row_count = row_count.saturating_add(1);
        }
        Ok(())
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
        if buffer.data.len() >= self.buffer_limit {
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
        let mut data = Vec::with_capacity(buffer.data.capacity());
        std::mem::swap(&mut data, &mut buffer.data);
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
/// process's cgroup-v2 self path. Not cfg-gated so it is unit-testable on any OS.
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
/// cgroup-v2 self path. Not cfg-gated so it is unit-testable on any OS.
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
                .send(PoolMessage::Task(PoolTask {
                    item: task,
                    done: tx,
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
// Per-table runtime spec for one resolution column, built once in
// `build_temp_node_file` from `IngestContext.resolve_columns`. The encode
// worker resolves the source column's PK reference value(s) to graph node-ids
// (via the shared `RefResolver`) and feeds the result into the encoder through
// a dedicated override channel — NO FK edge / no temp-edge push.
struct ResolveColRuntime {
    /// Column index of the NEW node-id column in `table.columns`.
    node_id_idx: usize,
    /// Column index of the source PK reference column in `table.columns`.
    src_idx: usize,
    /// Parent node table whose pkmap resolves the references.
    dst_table: String,
    /// Scalar vs list arity.
    kind: crate::config::ResolveKind,
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
        let src_type = table.columns[src_idx].ingested_dtype.clone();
        out.push(ResolveColRuntime {
            node_id_idx,
            src_idx,
            dst_table: spec.dst_table.clone(),
            kind: spec.kind,
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
) -> Result<Vec<diskgraph_rowpack::ResolveOverride>> {
    use crate::mapping::ResolveVerified;
    let mut overrides = vec![diskgraph_rowpack::ResolveOverride::None; num_columns];
    for spec in runtime {
        let shard_count = resolver.shard_count(&spec.dst_table).ok_or_else(|| {
            anyhow!(
                "resolution: pkmap for dst_table `{}` not loaded",
                spec.dst_table
            )
        })?;
        let record_failure = |key_failures: &mut KeyFailureState,
                                  outcome: ResolveVerified,
                                  key: &Key| {
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
                if key_from_access_into_buf(access, spec.src_idx, col, &spec.src_type, scratch)
                    .is_err()
                {
                    continue; // null / parse failure → leave column null
                }
                let mut data: &[u8] = scratch.as_slice();
                let key = match read_key_from_buf_for_type(&mut data, &spec.src_type) {
                    Ok(k) => k,
                    Err(_) => continue,
                };
                let shard = shard_for_key_from_key(&key, shard_buf, shard_count)?;
                match resolver.resolve_verified_in_shard(&spec.dst_table, shard, &key)? {
                    ResolveVerified::Resolved(node_id) => {
                        overrides[spec.node_id_idx] =
                            diskgraph_rowpack::ResolveOverride::Scalar(node_id);
                    }
                    // orphan → leave None (null) + count
                    outcome => record_failure(key_failures, outcome, &key),
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
                overrides[spec.node_id_idx] = diskgraph_rowpack::ResolveOverride::List(ids);
            }
        }
    }
    Ok(overrides)
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

fn validate_parquet_column_types(
    schema: &Schema,
    columns: &[ColumnSpec],
    file_path: &str,
) -> Result<()> {
    for col in columns {
        if let Ok(idx) = schema.index_of(&col.name) {
            if *schema.field(idx).data_type() == DataType::Null {
                anyhow::bail!(
                    "column '{}' in '{}' has Null data type (entire column is null). \
                     This typically happens when Pandas writes a shard where the column \
                     is entirely empty. Ensure all shards have properly typed columns.",
                    col.name,
                    file_path,
                );
            }
        }
    }
    Ok(())
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
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .unwrap();
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
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .unwrap();
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
            Some(PrecomputedKeys {
                shards,
                keys: Some(keys),
            })
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
            Some(PrecomputedKeys {
                shards,
                keys: Some(keys),
            })
        }
        (DataType::FixedSizeBinary(16), IngestedType::Uuid | IngestedType::Ip) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                .unwrap();
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
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                .unwrap();
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

fn edge_fast_path_for_array(
    array: &ArrayRef,
    target_type: &IngestedType,
) -> Option<EdgeKeyFastKind> {
    match (array.data_type(), target_type) {
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
        (DataType::Utf8 | DataType::LargeUtf8, IngestedType::Ip) => Some(EdgeKeyFastKind::StringIp),
        (
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_),
            IngestedType::Ip,
        ) => Some(EdgeKeyFastKind::BytesIp),
        _ => None,
    }
}

fn extract_batch_keys(array: &ArrayRef, kind: EdgeKeyFastKind) -> Result<Vec<Option<Key>>> {
    let len = array.len();
    let mut out = Vec::with_capacity(len);
    match kind {
        EdgeKeyFastKind::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i) as u64)));
                }
            }
        }
        EdgeKeyFastKind::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i) as u64)));
                }
            }
        }
        EdgeKeyFastKind::Int16 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int16Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i) as u64)));
                }
            }
        }
        EdgeKeyFastKind::Int8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int8Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i) as u64)));
                }
            }
        }
        EdgeKeyFastKind::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i))));
                }
            }
        }
        EdgeKeyFastKind::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i) as u64)));
                }
            }
        }
        EdgeKeyFastKind::UInt16 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt16Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i) as u64)));
                }
            }
        }
        EdgeKeyFastKind::UInt8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::UInt8Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i) as u64)));
                }
            }
        }
        EdgeKeyFastKind::Int64ToInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .unwrap();
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(Key::Int(arr.value(i) as i32 as u64)));
                }
            }
        }
        EdgeKeyFastKind::StringToInt64 => {
            for i in 0..len {
                if array.is_null(i) {
                    out.push(None);
                } else {
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
                if array.is_null(i) {
                    out.push(None);
                } else {
                    let res = (|| -> Result<Vec<u8>> {
                        let data = fast_string_value(array, i)?.as_bytes();
                        if data.len() > MAX_VAR_KEY_LEN {
                            return Err(anyhow!("key too large"));
                        }
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
                if array.is_null(i) {
                    out.push(None);
                } else {
                    let res = (|| -> Result<Vec<u8>> {
                        let data = fast_bytes_value(array, i)?;
                        if data.len() > MAX_VAR_KEY_LEN {
                            return Err(anyhow!("key too large"));
                        }
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
                if array.is_null(i) {
                    out.push(None);
                } else {
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
                if array.is_null(i) {
                    out.push(None);
                } else {
                    let res = (|| -> Result<[u8; 16]> {
                        let data = fast_bytes_value(array, i)?;
                        if data.len() != 16 {
                            return Err(anyhow!("length mismatch"));
                        }
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
                if array.is_null(i) {
                    out.push(None);
                } else {
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
                if array.is_null(i) {
                    out.push(None);
                } else {
                    let res = (|| -> Result<[u8; 32]> {
                        let data = fast_bytes_value(array, i)?;
                        if data.len() != 32 {
                            return Err(anyhow!("length mismatch"));
                        }
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
                if array.is_null(i) {
                    out.push(None);
                } else {
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
                if array.is_null(i) {
                    out.push(None);
                } else {
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

fn fast_string_value(array: &ArrayRef, row: usize) -> Result<&str, KeyParseError> {
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

fn fast_bytes_value(array: &ArrayRef, row: usize) -> Result<&[u8], KeyParseError> {
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
