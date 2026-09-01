//! Source-agnostic TT stage-extension ingest orchestration.
//!
//! After a TT pass produces `tt.parquet` and (for split plans) the per-stage
//! seed parquets, this module ingests the stages as a samplable DG extension
//! via [`diskgraph_ingest::ingest_stage_extension_with_resolution`], then
//! publishes the extension under the parent index (`train_tables/<id>/`). Every
//! stage resolves its reference PK column(s) to node-id rowpack columns — there
//! is no FK-edge mode.
//!
//! Two shapes:
//!   * **Split** (training): per-stage seed parts present → ingest
//!     `train_seeds` / `val_seeds` / `test_seeds` stages.
//!   * **No split** (prediction / PassThrough): no seed parts → the ordered
//!     `tt.parquet` parts form a single `predict` stage.
//!
//! This mirrors the PQL training path in
//! `diskgraph-pybind/src/tt_executor.rs::execute_tt_plan`
//! (`demux_tt_parquet_by_split` → `ingest_stage_parquets_into_extension` →
//! `publish_tt_extension`) so the CUSTOM (pre-materialized parquet) path
//! produces an orphan-correct, samplable extension identical in shape.

use std::path::{Path, PathBuf};

use diskgraph_core::time::Seconds;
use diskgraph_core::tt_builder::{stage_resolution_specs, DiskGraphTTPlan, RefResolveKind};

use diskgraph_ingest::{
    infer_stage_columns, ingest_stage_extension_with_resolution, ResolveSpec, StageSpec,
};

use crate::errors::{TTGenError, TTGenResult};
use crate::ttgen_executor::TTPassOutput;

/// Outcome of ingesting + publishing a TT extension.
#[derive(Debug)]
pub struct StageIngestOutput {
    /// Published extension dir (the `tt_root` passed in).
    pub extension_path: PathBuf,
    /// `(stage_table_name, emitted_row_count)` per ingested stage.
    pub stage_counts: Vec<(String, u64)>,
    /// Orphan-FK skip counters from the stage ingest (see the contract on
    /// `diskgraph_ingest::ingest_stage_extension`).
    pub key_failures: diskgraph_ingest::KeyFailureState,
}

/// This plan's resolution specs in the `diskgraph-ingest` type (see
/// [`stage_resolution_specs`]).
fn resolve_specs_from_plan(plan: &DiskGraphTTPlan<Seconds>) -> TTGenResult<Vec<ResolveSpec>> {
    let specs = stage_resolution_specs(&plan.lp_resolution, &plan.fkey_mapping, &plan.entity.table)
        .map_err(TTGenError::InvalidPlan)?;
    Ok(specs
        .iter()
        .map(|r| ResolveSpec {
            src_col: r.src_col.clone(),
            dst_table: r.dst_table.clone(),
            kind: match r.kind {
                RefResolveKind::Scalar => diskgraph_ingest::ResolveKind::Scalar,
                RefResolveKind::List => diskgraph_ingest::ResolveKind::List,
            },
            node_id_col: r.node_id_col.clone(),
        })
        .collect())
}

fn ingest_stage_set(
    dataset_name: &str,
    ingest_root: &Path,
    parent_index_path: &Path,
    stages: &[StageSpec],
    columns: Vec<diskgraph_ingest::graph_config::ColumnSpec>,
    plan: &DiskGraphTTPlan<Seconds>,
) -> TTGenResult<(Vec<(String, u64)>, diskgraph_ingest::KeyFailureState)> {
    // Per-stage `bucket_index` only for LP with a temporal RHS.
    let group_by_timestamp = plan.lp_resolution.is_some() && plan.lp_rhs_is_temporal;
    let resolve_specs = resolve_specs_from_plan(plan)?;

    ingest_stage_extension_with_resolution(
        dataset_name,
        ingest_root,
        parent_index_path,
        stages,
        columns,
        Vec::new(),
        &resolve_specs,
        None,
        true,
        group_by_timestamp,
    )
    .map(|r| (r.counts, r.key_failures))
    .map_err(|e| {
        TTGenError::SystemError(format!(
            "TT resolution-mode stage-table ingest failed: {e:#}"
        ))
    })
}

/// Demux (when a split.bin is present) + ingest + publish a TT extension for
/// any source. The PQL path and the custom (pre-materialized) path both feed
/// this with a finished [`TTPassOutput`].
///
/// * `tt_root` is the final published subdir (e.g.
///   `<index>/train_tables/<id>`); `staging_root`/`ingest_root` are scratch
///   dirs that are renamed/removed.
/// * Split plans (split.bin present) ingest `train_seeds` / `val_seeds` /
///   `test_seeds`. No-split plans ingest a single `predict` stage from the
///   whole `tt.parquet` dataset (no demux, no split.bin).
pub fn ingest_and_publish_tt_extension(
    dataset_name: &str,
    pass_output: &TTPassOutput,
    plan: &DiskGraphTTPlan<Seconds>,
    parent_index_path: &Path,
    // Scratch dir retained for signature stability; gen now writes the per-stage
    // parquets directly (no demux scratch needed).
    _staging_root: &Path,
    ingest_root: &Path,
    tt_root: &Path,
) -> TTGenResult<StageIngestOutput> {
    let (stage_counts, key_failures) = if let Some(infos) = pass_output.stage_gen_infos.as_ref() {
        // Split path: gen already wrote the per-stage seed parquets next to
        // tt.parquet (no split.bin, no demux). Ingest them directly.
        let gen_dir = pass_output.tt_parquet_path.parent().ok_or_else(|| {
            TTGenError::SystemError(
                "tt_stage_ingest: tt_parquet_path has no parent dir".to_string(),
            )
        })?;
        let stages: Vec<StageSpec> = infos
            .iter()
            .map(|i| StageSpec {
                stage_table_name: i.table_name.clone(),
                // A stage is one or more `<stage_table>/part-NNNNN.parquet`
                // files; gen records each part path relative to the staging
                // (gen) dir. Resolve them all so the per-file work pool fans
                // out over the full set.
                parquet_paths: i.parts.iter().map(|p| gen_dir.join(p)).collect(),
            })
            .collect();
        let columns = infer_stage_columns(&stages[0].parquet_paths[0])
            .map_err(|e| TTGenError::SchemaError(format!("TT stage column inference: {e:#}")))?;
        ingest_stage_set(
            dataset_name,
            ingest_root,
            parent_index_path,
            &stages,
            columns,
            plan,
        )?
    } else {
        // No-split path: all tt.parquet parts form one `predict` stage.
        let parquet_paths =
            crate::parquet_dataset::ordered_tt_dataset_parts(&pass_output.tt_parquet_path)?;
        let first_part = parquet_paths.first().ok_or_else(|| {
            TTGenError::InvalidPlan("no-split training table dataset is empty".to_string())
        })?;
        let columns = infer_stage_columns(first_part)
            .map_err(|e| TTGenError::SchemaError(format!("TT stage column inference: {e:#}")))?;
        let stages = [StageSpec {
            stage_table_name: "predict".to_string(),
            parquet_paths,
        }];
        ingest_stage_set(
            dataset_name,
            ingest_root,
            parent_index_path,
            &stages,
            columns,
            plan,
        )?
    };

    // Publish: rename the ingested extension into place.
    if tt_root.exists() {
        std::fs::remove_dir_all(tt_root).map_err(TTGenError::from)?;
    }
    if let Some(parent) = tt_root.parent() {
        std::fs::create_dir_all(parent).map_err(TTGenError::from)?;
    }
    std::fs::rename(ingest_root, tt_root).map_err(TTGenError::from)?;

    Ok(StageIngestOutput {
        extension_path: tt_root.to_path_buf(),
        stage_counts,
        key_failures,
    })
}
