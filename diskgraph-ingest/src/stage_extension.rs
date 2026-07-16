//! Source-agnostic stage-extension ingest.
//!
//! A "stage extension" is a DG index extension containing one or more
//! pre-materialized *stage* tables (e.g. `train_seeds`/`val_seeds`/`test_seeds`
//! for the training path, or `predict` for the batch-prediction path) whose
//! rows reference a parent DG index. Seed→graph is a reference, not topology:
//! at ingest the reference PK column(s) are resolved against the parent's
//! primary-key map into node-id rowpack columns (`entity_node_id`,
//! `target_node_ids`) — NO seed FK edges / `seed.edges.bin` are produced.
//!
//! This module factors that ingest out of the TT-NG-specific pybind layer into
//! a plain primitive: callers supply the stage parquet paths, the shared column
//! schema, and the resolution specs; the primitive assembles a [`GraphConfig`]
//! (shard count size-derived or caller-overridden), runs
//! [`ingest_with_stats_from_source`], and reports the emitted row count per
//! stage plus the orphan skip counters — see the contract on
//! [`ingest_stage_extension_with_resolution`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{ResolveColumnSpec, ResolveKind};
use crate::graph_config::{ColumnSpec, DataSource, FKeySpec, GraphConfig, IngestedType, TableSpec};
use crate::{ingest_with_stats_from_source_and_sidecars, Dest, IngestOptions};

/// One stage table to ingest into a stage extension.
pub struct StageSpec {
    /// On-disk table name for this stage (e.g. `train_seeds`, `predict`).
    pub stage_table_name: String,
    /// Path(s) to the pre-materialized stage parquet part(s). A stage may be a
    /// single file (predict / no-split path) or multiple `part-*.parquet` files
    /// (split path). All parts share the same schema; the per-file work pool
    /// ingests them in parallel and the resulting sub-index is identical to
    /// ingesting one concatenated parquet (union of all part rows).
    pub parquet_paths: Vec<PathBuf>,
}

/// Optional seed-sampling sidecars to emit for a stage table.
///
/// Kept separate from [`TableSpec`] and [`StageSpec`] so ordinary graph/table
/// config and ordinary stage ingestion do not need to mention weighted
/// training internals.
#[derive(Clone, Debug, Default)]
pub struct StageSidecarColumns {
    /// Binary target column for replacement-sampling class buckets.
    /// Non-binary values omit the sidecar instead of failing ingest.
    pub binary_label_column: Option<String>,
    /// Numeric row-weight column for replacement-sampling CDF buckets.
    /// Invalid values omit the sidecar instead of failing ingest.
    pub sample_weight_column: Option<String>,
}

/// Result of [`ingest_stage_extension_with_resolution`]: per-stage `(table_name, node_count)`
/// plus, when `group_by_timestamp` was set, the per-stage-table timestamp
/// grouping (`<table>/bucket_index.bin` + bucket list). `ts_groupings` is empty
/// when grouping was disabled. `key_failures` carries the orphan-FK skip
/// counters (see the contract on [`ingest_stage_extension_with_resolution`]).
pub struct StageExtensionResult {
    pub counts: Vec<(String, u64)>,
    pub ts_groupings:
        std::collections::HashMap<String, crate::ts_grouping::StageGrouping>,
    pub binary_label_indexes:
        std::collections::HashMap<String, crate::binary_label::BinaryLabelIndex>,
    pub sample_weight_indexes:
        std::collections::HashMap<String, crate::sample_weight::SampleWeightIndex>,
    pub key_failures: crate::stats::KeyFailureState,
}

/// Lift the key-failure counters out of a full [`crate::stats::IngestStats`].
fn key_failures_from_stats(stats: &crate::stats::IngestStats) -> crate::stats::KeyFailureState {
    crate::stats::KeyFailureState {
        parse_failures: stats.key_parse_failures,
        lookup_failures: stats.key_lookup_failures,
        membership_failures: stats.key_membership_failures,
        samples: stats.key_failure_samples.clone(),
        by_context: stats.key_failures_by_context.clone(),
    }
}

/// A single FK-free LP resolution column for [`ingest_stage_extension_with_resolution`]:
/// resolve `src_col`'s PK reference value(s) against the parent table `dst_table`
/// to graph node-id(s), and write them into a NEW rowpack column `node_id_col`
/// (a `UInt64` scalar or `Int64Array` list) with NO FK edge.
pub struct ResolveSpec {
    pub src_col: String,
    pub dst_table: String,
    pub kind: ResolveKind,
    pub node_id_col: String,
}

/// Helper used by the synthetic-column appender to build a NEW node-id column.
fn node_id_column(name: &str, dtype: IngestedType) -> ColumnSpec {
    ColumnSpec {
        name: name.to_string(),
        source_dtype: None,
        ingested_dtype: dtype,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    }
}

/// Resolution-mode stage-extension ingest: appends a node-id column
/// (`UInt64` for [`ResolveKind::Scalar`], `Int64Array` for [`ResolveKind::List`])
/// per resolve spec and fills it inline from the parent pkmap. No FK edge is
/// emitted for these columns — the engine reads the node-id columns directly.
#[allow(clippy::too_many_arguments)]
pub fn ingest_stage_extension_with_resolution(
    dataset_name: &str,
    extension_output_root: &Path,
    parent_index_path: &Path,
    stages: &[StageSpec],
    columns: Vec<ColumnSpec>,
    fkey_specs: Vec<FKeySpec>,
    resolve_specs: &[ResolveSpec],
    num_shards: Option<usize>,
    col_stats: bool,
    // Emit per-stage `bucket_index` (temporal LP only).
    group_by_timestamp: bool,
) -> Result<StageExtensionResult> {
    ingest_stage_extension_with_resolution_and_sidecars(
        dataset_name,
        extension_output_root,
        parent_index_path,
        stages,
        HashMap::new(),
        columns,
        fkey_specs,
        resolve_specs,
        num_shards,
        col_stats,
        group_by_timestamp,
    )
}

/// Identical to [`ingest_stage_extension_with_resolution`], plus optional
/// per-stage sidecar requests for TT-NG replacement seed sampling.
#[allow(clippy::too_many_arguments)]
pub fn ingest_stage_extension_with_resolution_and_sidecars(
    dataset_name: &str,
    extension_output_root: &Path,
    parent_index_path: &Path,
    stages: &[StageSpec],
    stage_sidecars: HashMap<String, StageSidecarColumns>,
    columns: Vec<ColumnSpec>,
    fkey_specs: Vec<FKeySpec>,
    resolve_specs: &[ResolveSpec],
    num_shards: Option<usize>,
    col_stats: bool,
    // Emit per-stage `bucket_index` (temporal LP only).
    group_by_timestamp: bool,
) -> Result<StageExtensionResult> {
    // Append the synthetic resolved node-id columns to the shared column list.
    let mut columns = columns;
    for spec in resolve_specs {
        let dtype = match spec.kind {
            ResolveKind::Scalar => IngestedType::UInt64,
            ResolveKind::List => IngestedType::Int64Array,
        };
        columns.push(node_id_column(&spec.node_id_col, dtype));
    }

    let tables: Vec<TableSpec> = stages
        .iter()
        .map(|stage| TableSpec {
            name: stage.stage_table_name.clone(),
            columns: columns.clone(),
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: stage
                    .parquet_paths
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect(),
            },
            source_size: None,
            // Only caller-supplied fkeys are materialized as edges. The
            // resolved reference columns are NOT in this list → no edge.
            fkeys: fkey_specs.clone(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp,
        })
        .collect();

    // Per-stage resolve column specs (one set per stage table, all sharing the
    // same resolution mapping).
    let resolve_columns: Vec<ResolveColumnSpec> = stages
        .iter()
        .flat_map(|stage| {
            resolve_specs.iter().map(move |spec| ResolveColumnSpec {
                table: stage.stage_table_name.clone(),
                src_col: spec.src_col.clone(),
                dst_table: spec.dst_table.clone(),
                kind: spec.kind,
                node_id_col: spec.node_id_col.clone(),
            })
        })
        .collect();

    let resolved_num_shards = match num_shards {
        Some(n) => n.max(1),
        None => {
            let total_bytes: u64 = stages
                .iter()
                .flat_map(|stage| stage.parquet_paths.iter())
                .map(|path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
                .sum();
            crate::config_converter::inference::estimate_num_shards(total_bytes).max(1)
        }
    };

    let graph_config = GraphConfig {
        dataset_name: dataset_name.to_string(),
        tables,
        rowpack: None,
        num_shards: Some(resolved_num_shards),
        edge_format: None,
        coercion_mode: Default::default(),
    };

    let options = IngestOptions {
        col_stats,
        parent_index_path: Some(parent_index_path.to_path_buf()),
        resolve_columns,
        ..Default::default()
    };

    let (_, stats) = ingest_with_stats_from_source_and_sidecars(
        graph_config,
        None,
        &extension_output_root.to_string_lossy(),
        Dest::Local,
        options,
        None,
        None,
        stage_sidecars,
    )
    .with_context(|| {
        format!(
            "resolution-mode stage-extension ingest failed for dataset `{dataset_name}` at {}",
            extension_output_root.display()
        )
    })?;

    let metadata_path = extension_output_root.join("index_metadata.json");
    let reader =
        diskgraph_index::graph::GraphIndexReader::open(&metadata_path, extension_output_root)
            .with_context(|| {
                format!(
                    "failed to open produced stage extension at {}",
                    extension_output_root.display()
                )
            })?;

    let mut counts = Vec::with_capacity(stages.len());
    for stage in stages {
        let node_count = reader
            .table(&stage.stage_table_name)
            .with_context(|| {
                format!(
                    "stage table `{}` missing from produced extension at {}",
                    stage.stage_table_name,
                    extension_output_root.display()
                )
            })?
            .node_count();
        counts.push((stage.stage_table_name.clone(), node_count));
    }
    let key_failures = key_failures_from_stats(&stats);
    Ok(StageExtensionResult {
        counts,
        ts_groupings: stats.ts_groupings,
        binary_label_indexes: stats.binary_label_indexes,
        sample_weight_indexes: stats.sample_weight_indexes,
        key_failures,
    })
}
