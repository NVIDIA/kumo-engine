//! PyO3 bindings for the TT-gen executor.
//!
//! Surface area:
//!
//! * [`execute_tt_plan`] — run a plan against an index and write
//!   `tt.parquet`, `split.bin`, and `manifest.json` to the output dir.
//! * [`read_tt_parquet_batches`] — open a TT parquet and yield
//!   `pyarrow.RecordBatch` objects. Bridge for `diskgraph_batch_pred.py`
//!   (kumo) until that path is ported to Rust.
//! * [`build_prediction_table_edge_bundle_artifact`] — JSON-only artifact
//!   builder (file metadata + checksums) used by the prediction-table
//!   workflow in `prediction_artifacts.py`.
//! * [`persist_prediction_artifact_bundle`] — manifest writer for the
//!   prediction-table workflow.
//!
//! The legacy `execute_tt_plan_materialize_with_split` path and the entire
//! TT artifact-bundle Python wrapper layer were deleted when this crate
//! took over TT-gen; their helpers, downsampling decision tree, pre-DS
//! scratch I/O, and post-DS col-stats recompute have been removed.
#![allow(clippy::too_many_arguments)]

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::{Component, Path};

use arrow::pyarrow::ToPyArrow;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use diskgraph_core::time::Seconds;
use diskgraph_core::tt_builder::{DiskGraphTTPlan, RefResolveKind, RefResolveSpec, TTFKeySpec};
use diskgraph_train_table::ttgen_executor::{
    execute_tt_pass, execute_tt_pass_with_source, CustomTtSource, StageGenInfo, TTPassOutput,
    STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION,
};
use diskgraph_train_table::ttgen_output::{PRIMARY_ENTITY_COL, PRIMARY_TARGET_COL, TIMESTAMP_COL};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use serde_json::{json, Value};
use tracing::info;

use crate::artifact_helpers::{
    artifact_ref, ensure_artifacts_dir, merge_prefixed_checksums, sha256_hex_bytes,
    sha256_hex_path, write_artifact_file, write_manifest_json,
};

// ─── cross-job ingest spec ──────────────────────────────────────────────────

/// Schema version stamped into `tt_ingest_spec.json`. Bump when the on-disk
/// contract changes incompatibly.
pub(crate) const TT_INGEST_SPEC_SCHEMA_VERSION: &str = "diskgraph.tt_ingest_spec.v3";

/// DG's own cross-job hand-off contract, written by [`execute_tt_gen`] into the
/// staging dir (sibling of `tt.parquet`) and read back by [`execute_tt_ingest`].
///
/// It carries EVERYTHING the ingest/publish phase needs from the in-memory
/// `plan` that is no longer available in the separate ingest job:
///
/// * `fkey_mapping` — the exact `plan.fkey_mapping`, serialized as the real
///   [`TTFKeySpec`] type (NOT a hand-rolled field list) so that when a later
///   branch adds e.g. `list_fk: bool #[serde(default)]` to `TTFKeySpec`, it
///   threads through this spec automatically with no change here.
/// * `entity_table` — `plan.entity.table`, the fallback `dst_table` for the
///   primary-entity FK used by `ingest_stage_parquets_into_extension` when
///   `fkey_mapping` is `None`.
///
/// The returned task-dataset dict's column names (`primary_entity`,
/// `primary_target`, `timestamp`) and stage table names are compile-time
/// constants (`PRIMARY_ENTITY_COL` etc.), not plan-derived, so they do not
/// need to be carried in the spec.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct TtIngestSpec {
    pub schema_version: String,
    pub tt_id: String,
    #[serde(default)]
    pub fkey_mapping: Option<Vec<TTFKeySpec>>,
    pub entity_table: String,
    /// LP resolution specs (`ENTITY1`/`TARGET1` → node-id rowpack columns).
    /// `None` for non-LP; ingest then synthesizes the scalar `ENTITY1` spec.
    #[serde(default)]
    pub lp_resolution: Option<Vec<RefResolveSpec>>,
    /// `plan.split.is_temporal`, carried so the ingest-output manifest is
    /// self-describing. `#[serde(default)]` → old specs deserialize as `false`.
    #[serde(default)]
    pub is_temporal: bool,
    /// `plan.lp_rhs_is_temporal` (kumo-computed data-condition for grouping):
    /// `true` iff LP whose RHS/target table has a create-time or end-time
    /// column. The split ingest path gates `bucket_index` on
    /// `lp_resolution.is_some() && lp_rhs_is_temporal`. `#[serde(default)]` →
    /// old specs deserialize as `false` (flat).
    #[serde(default)]
    pub lp_rhs_is_temporal: bool,
    /// Per-stage seed parquets gen produced, with row counts + seed-times.
    /// `Some` on the split path; `None` on the no-split (prediction) path.
    /// Set by `execute_tt_gen` from the pass output, not derivable from the plan.
    #[serde(default)]
    pub split_stages: Option<Vec<StageGenInfo>>,
}

impl TtIngestSpec {
    pub(crate) fn from_plan(tt_id: &str, plan: &DiskGraphTTPlan<Seconds>) -> Self {
        Self {
            schema_version: TT_INGEST_SPEC_SCHEMA_VERSION.to_string(),
            tt_id: tt_id.to_string(),
            fkey_mapping: plan.fkey_mapping.clone(),
            entity_table: plan.entity.table.clone(),
            lp_resolution: plan.lp_resolution.clone(),
            is_temporal: plan
                .split
                .as_ref()
                .and_then(|s| s.is_temporal)
                .unwrap_or(false),
            lp_rhs_is_temporal: plan.lp_rhs_is_temporal,
            // Filled by `execute_tt_gen` from the pass output.
            split_stages: None,
        }
    }
}

/// Derive the FK-free LP RHS (target) node type from a resolution plan: it is
/// the `dst_table` of the single `List`-kind reference (`TARGET1` → RHS table).
/// `None` for non-LP plans (no resolution, or only the `Scalar` entity ref).
pub(crate) fn rhs_node_type_from_resolution(
    lp_resolution: &Option<Vec<RefResolveSpec>>,
) -> Option<String> {
    lp_resolution.as_ref().and_then(|specs| {
        specs
            .iter()
            .find(|s| s.kind == RefResolveKind::List)
            .map(|s| s.dst_table.clone())
    })
}

// ─── execute_tt_plan ─────────────────────────────────────────────────────────

/// Run a TT-gen plan against a DiskGraph index. Writes TT artifacts and
/// ingested stage tables under `index_path/train_tables/<tt_id>/`. Returns the
/// native DiskGraph TT execute result for Kumo's runtime task-dataset manifest
/// writer. This new wheel intentionally does not return the old Kumo artifact
/// bundle shape.
#[pyfunction]
#[tracing::instrument(
    level = "info",
    skip_all,
    fields(index_path = %index_path, tt_id = %tt_id)
)]
pub fn execute_tt_plan(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
    tt_id: &str,
) -> PyResult<Py<PyAny>> {
    validate_tt_id(tt_id)?;
    info!(
        "execute_tt_plan started (TT-NG end-state path) tt_id={}",
        tt_id
    );
    let plan: DiskGraphTTPlan<Seconds> = serde_json::from_str(plan_json)
        .map_err(|e| crate::DiskgraphConfigError::new_err(format!("invalid plan JSON: {e}")))?;

    let idx = index_path.to_owned();
    let tt_id_owned = tt_id.to_owned();

    let (
        manifest_stages,
        tt_extension_path,
        unpublished_extension_path,
        key_failures,
        input_row_count,
    ): (
        Vec<ManifestStage>,
        String,
        String,
        diskgraph_ingest::KeyFailureState,
        u64,
    ) = py
        .detach(|| -> diskgraph_train_table::errors::TTGenResult<_> {
            let index_root = Path::new(&idx);
            let tt_root = index_root.join("train_tables").join(&tt_id_owned);
            let staging_root = staging_root_for(index_root, &tt_id_owned);
            let ingest_root = ingest_root_for(index_root, &tt_id_owned);
            for tmp in [&staging_root, &ingest_root] {
                if tmp.exists() {
                    std::fs::remove_dir_all(tmp)?;
                }
            }
            std::fs::create_dir_all(&staging_root)?;

            // Gen inline (NOT via execute_tt_gen — that's a separate Python
            // entry point), then drive ingest/publish through the SAME shared
            // helpers the split execute_tt_ingest uses.
            let pass_output = execute_tt_pass(&plan, index_root, &staging_root)?;
            let artifacts = StagedTtArtifacts::from_pass_output(&pass_output)?;
            let stages = resolve_ingest_stages(
                &staging_root,
                &pass_output.tt_parquet_path,
                DEFAULT_PREDICT_STAGE,
                pass_output.stage_gen_infos.as_deref(),
            )?;
            // Build the per-stage `bucket_index` only when single-timestamp
            // grouping can actually be used: resolution-mode LP whose RHS/target
            // table is temporal (kumo-computed `lp_rhs_is_temporal`). This
            // mirrors the kumo LOAD gate and legacy's `TimestampSampler` RHS
            // check — an atemporal RHS (e.g. amazon `products`) builds no bucket
            // index and the LOAD gate stays flat. Byte-identical otherwise.
            let group_by_timestamp = plan.lp_resolution.is_some() && plan.lp_rhs_is_temporal;
            let resolve_specs = diskgraph_core::tt_builder::stage_resolution_specs(
                &plan.lp_resolution,
                &plan.fkey_mapping,
                &plan.entity.table,
            )
            .map_err(diskgraph_train_table::errors::TTGenError::InvalidPlan)?;
            let (manifest_stages, key_failures, input_row_count) = ingest_resolved_stages(
                &stages,
                &tt_id_owned,
                &ingest_root,
                index_root,
                &resolve_specs,
                &artifacts,
                group_by_timestamp,
            )?;
            let _ = std::fs::remove_dir_all(&staging_root);
            Ok((
                manifest_stages,
                tt_root.to_string_lossy().to_string(),
                ingest_root.to_string_lossy().to_string(),
                key_failures,
                input_row_count,
            ))
        })
        .map_err(crate::map_ttgen_err)?;

    // Self-describing manifest fields, derived from the plan (the fused path
    // has the plan in hand; the split ingest path recovers them from the spec).
    let is_temporal = plan
        .split
        .as_ref()
        .and_then(|s| s.is_temporal)
        .unwrap_or(false);
    let rhs_node_type = rhs_node_type_from_resolution(&plan.lp_resolution);
    write_tt_manifest_and_build_result(
        py,
        index_path,
        tt_id,
        &tt_extension_path,
        &unpublished_extension_path,
        &manifest_stages,
        is_temporal,
        rhs_node_type.as_deref(),
        &key_failures,
        input_row_count,
    )
}

/// Staging-dir convention for the split gen/ingest hand-off: a hidden sibling
/// of the published `<index>/train_tables/<tt_id>/` extension. This is the SAME
/// dir the fused `execute_tt_plan` uses for its pre-publish artifacts.
pub(crate) fn staging_root_for(index_path: &Path, tt_id: &str) -> std::path::PathBuf {
    index_path
        .join("train_tables")
        .join(format!(".{}.staging", tt_id))
}

/// Hidden sibling holding the unpublished ingested extension before the atomic
/// rename into `<index>/train_tables/<tt_id>/`.
pub(crate) fn ingest_root_for(index_path: &Path, tt_id: &str) -> std::path::PathBuf {
    index_path
        .join("train_tables")
        .join(format!(".{}.ingested", tt_id))
}

/// Filename of DG's cross-job ingest spec, written by [`execute_tt_gen`] as a
/// sibling of `tt.parquet` and read back by [`execute_tt_ingest`].
pub(crate) const TT_INGEST_SPEC_FILENAME: &str = "tt_ingest_spec.json";

/// One ingest stage: manifest key, seed-table name, parquet part(s), and
/// seed-time `(min,max)`. The predict path leaves `seed_time` `None` with a
/// ordered parts under `tt.parquet`; the split path may carry several
/// `part-*.parquet` files into one stage table. Row count is read back
/// post-ingest as the surviving `node_count` (gen-time counts go stale after
/// orphan drops).
#[derive(Debug, Clone)]
pub(crate) struct StageSpec {
    pub key: String,
    pub table_name: String,
    pub parquet_paths: Vec<std::path::PathBuf>,
    pub seed_time: Option<[i64; 2]>,
    pub binary_label_column: Option<String>,
    pub sample_weight_column: Option<String>,
}

fn invalid_stage(
    info: &StageGenInfo,
    message: impl std::fmt::Display,
) -> diskgraph_train_table::errors::TTGenError {
    diskgraph_train_table::errors::TTGenError::InvalidPlan(format!(
        "stage {} artifact contract is invalid: {message}",
        info.key
    ))
}

fn invalid_staging(message: impl std::fmt::Display) -> diskgraph_train_table::errors::TTGenError {
    diskgraph_train_table::errors::TTGenError::InvalidPlan(format!(
        "TT staging artifact contract is invalid: {message}"
    ))
}

fn require_regular_staging_file(
    staging_dir: &Path,
    relative: &Path,
) -> diskgraph_train_table::errors::TTGenResult<std::fs::Metadata> {
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid_staging(format!(
            "unsafe relative artifact path {:?}",
            relative
        )));
    }

    let mut current = staging_dir.to_path_buf();
    let component_count = relative.components().count();
    for (index, component) in relative.components().enumerate() {
        current.push(component.as_os_str());
        let metadata = std::fs::symlink_metadata(&current).map_err(|error| {
            invalid_staging(format!("cannot inspect {}: {error}", current.display()))
        })?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(invalid_staging(format!(
                "{} traverses a symlink",
                current.display()
            )));
        }
        let is_leaf = index + 1 == component_count;
        if (is_leaf && !file_type.is_file()) || (!is_leaf && !file_type.is_dir()) {
            return Err(invalid_staging(format!(
                "{} is not a {}",
                current.display(),
                if is_leaf { "regular file" } else { "directory" }
            )));
        }
        if is_leaf {
            return Ok(metadata);
        }
    }
    unreachable!("an empty relative path was rejected above")
}

fn validate_staging_artifact_types(
    staging_dir: &Path,
) -> diskgraph_train_table::errors::TTGenResult<()> {
    let root_metadata = std::fs::symlink_metadata(staging_dir).map_err(|error| {
        invalid_staging(format!(
            "cannot inspect staging root {}: {error}",
            staging_dir.display()
        ))
    })?;
    if root_metadata.file_type().is_symlink() || !root_metadata.file_type().is_dir() {
        return Err(invalid_staging(format!(
            "staging root {} must be a non-symlink directory",
            staging_dir.display()
        )));
    }
    for artifact in [TT_INGEST_SPEC_FILENAME, "manifest.json"] {
        require_regular_staging_file(staging_dir, Path::new(artifact))?;
    }
    diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(
        &staging_dir.join(diskgraph_train_table::parquet_dataset::TT_DATASET_DIR),
    )?;
    Ok(())
}

fn weight_col_is_per_element(parts: &[std::path::PathBuf]) -> bool {
    let Some(first) = parts.first() else {
        return false;
    };
    let Ok(file) = std::fs::File::open(first) else {
        return false;
    };
    let Ok(builder) = ParquetRecordBatchReaderBuilder::try_new(file) else {
        return false;
    };
    let schema = builder.schema();
    schema
        .column_with_name(diskgraph_train_table::ttgen_output::WEIGHT_COL)
        .is_some_and(|(_, f)| matches!(f.data_type(), arrow_schema::DataType::List(_)))
}

fn validate_train_table_inventory(
    staging_dir: &Path,
) -> diskgraph_train_table::errors::TTGenResult<Vec<std::path::PathBuf>> {
    let manifest =
        diskgraph_train_table::manifest::read_manifest(&staging_dir.join("manifest.json"))?;
    if manifest.train_table.path != diskgraph_train_table::parquet_dataset::TT_DATASET_DIR {
        return Err(invalid_staging(format!(
            "manifest train_table.path must be {:?}, got {:?}",
            diskgraph_train_table::parquet_dataset::TT_DATASET_DIR,
            manifest.train_table.path
        )));
    }
    if manifest.train_table.format != diskgraph_train_table::manifest::TT_PARQUET_FORMAT {
        return Err(invalid_staging(format!(
            "manifest train_table.format must be {:?}, got {:?}",
            diskgraph_train_table::manifest::TT_PARQUET_FORMAT,
            manifest.train_table.format
        )));
    }
    let dataset_path = staging_dir.join(&manifest.train_table.path);
    let paths = diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(&dataset_path)?;
    if paths.len() != manifest.train_table.parts.len() {
        return Err(invalid_staging("manifest train-table part count mismatch"));
    }
    let mut rows = 0u64;
    let mut bytes = 0u64;
    for (path, declared) in paths.iter().zip(&manifest.train_table.parts) {
        let actual_relative = Path::new(&manifest.train_table.path)
            .join(path.file_name().expect("dataset part has a filename"));
        if actual_relative != Path::new(&declared.path) {
            return Err(invalid_staging(format!(
                "manifest train-table part {:?} does not match {:?}",
                declared.path, actual_relative
            )));
        }
        let size = std::fs::metadata(path)?.len();
        if size != declared.size_bytes {
            return Err(invalid_staging(format!(
                "train-table part size mismatch for {}: expected={} actual={size}",
                path.display(),
                declared.size_bytes
            )));
        }
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
        let part_rows = u64::try_from(builder.metadata().file_metadata().num_rows())
            .map_err(|_| invalid_staging(format!("negative row count in {}", path.display())))?;
        if part_rows != declared.row_count {
            return Err(invalid_staging(format!(
                "train-table part row mismatch for {}: expected={} actual={part_rows}",
                path.display(),
                declared.row_count
            )));
        }
        rows = rows
            .checked_add(part_rows)
            .ok_or_else(|| invalid_staging("row count overflow"))?;
        bytes = bytes
            .checked_add(size)
            .ok_or_else(|| invalid_staging("byte count overflow"))?;
    }
    if rows != manifest.train_table.row_count
        || rows != manifest.row_count
        || bytes != manifest.train_table.size_bytes
    {
        return Err(invalid_staging(format!(
            "train-table aggregate mismatch: rows={rows} manifest_rows={} root_rows={} bytes={bytes} manifest_bytes={}",
            manifest.train_table.row_count,
            manifest.row_count,
            manifest.train_table.size_bytes
        )));
    }
    Ok(paths)
}

fn validate_stage_parts(
    staging_dir: &Path,
    info: &StageGenInfo,
) -> diskgraph_train_table::errors::TTGenResult<Vec<std::path::PathBuf>> {
    if info.parts.is_empty() {
        return Err(invalid_stage(info, "parts must not be empty"));
    }

    let mut seen = HashSet::with_capacity(info.parts.len());
    let mut paths = Vec::with_capacity(info.parts.len());
    for part in &info.parts {
        let relative = Path::new(part);
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(invalid_stage(
                info,
                format!("unsafe relative path {part:?}"),
            ));
        }
        if !seen.insert(part.as_str()) {
            return Err(invalid_stage(info, format!("duplicate part path {part:?}")));
        }
        paths.push(staging_dir.join(relative));
    }

    let inventory = info
        .artifact_inventory
        .as_ref()
        .ok_or_else(|| invalid_stage(info, "missing artifact inventory"))?;
    if inventory.schema_version != STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION {
        return Err(invalid_stage(
            info,
            format!(
                "unsupported inventory schema {:?}",
                inventory.schema_version
            ),
        ));
    }
    if inventory.parts.len() != info.parts.len() {
        return Err(invalid_stage(
            info,
            "inventory part count does not match parts",
        ));
    }
    let declared = &inventory.parts;

    let mut observed_stage_rows = 0u64;
    for (part_index, path) in paths.iter().enumerate() {
        let declared_part = &declared[part_index];
        if declared_part.path != info.parts[part_index] {
            return Err(invalid_stage(
                info,
                format!(
                    "inventory path {:?} does not match {:?}",
                    declared_part.path, info.parts[part_index]
                ),
            ));
        }

        let size_bytes =
            require_regular_staging_file(staging_dir, Path::new(&info.parts[part_index]))
            .map_err(|error| invalid_stage(info, error))?
            .len();
        if size_bytes != declared_part.size_bytes {
            return Err(invalid_stage(
                info,
                format!(
                    "size mismatch for {}: expected={} actual={size_bytes}",
                    path.display(),
                    declared_part.size_bytes
                ),
            ));
        }

        let file = File::open(path).map_err(|error| {
            invalid_stage(info, format!("cannot open {}: {error}", path.display()))
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
            invalid_stage(
                info,
                format!(
                    "cannot read parquet metadata for {}: {error}",
                    path.display()
                ),
            )
        })?;
        let metadata = builder.metadata();
        if metadata.file_metadata().schema_descr().num_columns() == 0
            || builder.schema().fields().is_empty()
        {
            return Err(invalid_stage(
                info,
                format!("{} has an empty parquet schema", path.display()),
            ));
        }
        let footer_rows = u64::try_from(metadata.file_metadata().num_rows()).map_err(|_| {
            invalid_stage(info, format!("{} has a negative row count", path.display()))
        })?;
        let mut row_group_rows = 0u64;
        for row_group in metadata.row_groups() {
            let rows = u64::try_from(row_group.num_rows()).map_err(|_| {
                invalid_stage(
                    info,
                    format!("{} has a negative row-group count", path.display()),
                )
            })?;
            row_group_rows = row_group_rows.checked_add(rows).ok_or_else(|| {
                invalid_stage(info, format!("{} row count overflow", path.display()))
            })?;
            if row_group.num_columns() != metadata.file_metadata().schema_descr().num_columns() {
                return Err(invalid_stage(
                    info,
                    format!("{} row-group/schema mismatch", path.display()),
                ));
            }
        }
        if row_group_rows != footer_rows {
            return Err(invalid_stage(
                info,
                format!(
                    "{} row groups sum to {row_group_rows} but footer declares {footer_rows}",
                    path.display()
                ),
            ));
        }
        if footer_rows != declared_part.row_count {
            return Err(invalid_stage(
                info,
                format!(
                    "row_count mismatch for {}: expected={} actual={footer_rows}",
                    path.display(),
                    declared_part.row_count
                ),
            ));
        }
        observed_stage_rows = observed_stage_rows
            .checked_add(footer_rows)
            .ok_or_else(|| invalid_stage(info, "stage row count overflow"))?;
    }

    if observed_stage_rows != info.row_count {
        return Err(invalid_stage(
            info,
            format!(
                "row_count mismatch: expected={} actual={observed_stage_rows}",
                info.row_count
            ),
        ));
    }
    Ok(paths)
}

fn validate_ingest_spec(spec: &TtIngestSpec) -> diskgraph_train_table::errors::TTGenResult<()> {
    if spec.schema_version != TT_INGEST_SPEC_SCHEMA_VERSION {
        return Err(diskgraph_train_table::errors::TTGenError::InvalidPlan(
            format!(
            "unsupported tt_ingest_spec schema_version {:?}",
            spec.schema_version
            ),
        ));
    }
    if spec.split_stages.as_ref().is_some_and(|stages| {
        stages
            .iter()
            .any(|stage| stage.artifact_inventory.is_none())
    }) {
        return Err(diskgraph_train_table::errors::TTGenError::InvalidPlan(
            "tt_ingest_spec.v3 split stage is missing its artifact inventory".to_string(),
        ));
    }
    Ok(())
}

/// THE single split/predict fork. With per-stage seed parquets
/// (`stage_gen_infos`), build a [`StageSpec`] per stage from its
/// `part-*.parquet` files under `<staging_dir>`; otherwise (no-split predict)
/// emit one `<predict_key>_seeds` stage covering the whole `tt.parquet` dataset.
pub(crate) fn resolve_ingest_stages(
    staging_dir: &Path,
    tt_parquet: &Path,
    predict_key: &str,
    stage_gen_infos: Option<&[StageGenInfo]>,
) -> diskgraph_train_table::errors::TTGenResult<Vec<StageSpec>> {
    match stage_gen_infos {
        Some(infos) => {
            let mut seen_parts = HashSet::new();
            for info in infos {
                for part in &info.parts {
                    if !seen_parts.insert(part.as_str()) {
                        return Err(invalid_stage(
                            info,
                            format!("part path {part:?} is declared by multiple stages"),
                        ));
                    }
                }
            }
            infos
                .iter()
                .map(|info| {
                    let parquet_paths = validate_stage_parts(staging_dir, info)?;
                    let per_element_weight =
                        info.key == "train" && weight_col_is_per_element(&parquet_paths);
                    Ok(StageSpec {
                        key: info.key.clone(),
                        table_name: info.table_name.clone(),
                        parquet_paths,
                        seed_time: info.seed_time,
                        binary_label_column: (info.key == "train")
                            .then(|| PRIMARY_TARGET_COL.to_string()),
                        // Seed sampling weights whole rows; a per-element
                        // weight has no per-row value to sample by.
                        sample_weight_column: (info.key == "train" && !per_element_weight)
                            .then(|| diskgraph_train_table::ttgen_output::WEIGHT_COL.to_string()),
                    })
                })
                .collect()
        }
        None => Ok(vec![StageSpec {
            key: predict_key.to_string(),
            table_name: format!("{predict_key}_seeds"),
            // No-split (predict) path: the ordered dataset parts form one stage.
            parquet_paths: diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(
                tt_parquet,
            )?,
            seed_time: None,
            binary_label_column: None,
            sample_weight_column: None,
        }]),
    }
}

// ─── execute_tt_gen ──────────────────────────────────────────────────────────

/// Shared gen-only core for BOTH TT producers — the PQL path
/// ([`execute_tt_gen`], `source=None`, plan compiled from the predictive
/// query) and the custom-TT path (`custom_tt_entry`, `source=Some`, minimal
/// empty-stage plan reading every output column from the customer parquet).
/// Runs the ttgen pass into `out_dir` and writes the cross-job
/// `tt_ingest_spec.json` beside `tt.parquet`, producing the durable staging
/// bundle the separate [`execute_tt_ingest`] job consumes. The producers
/// differ only in how the plan is built, where the bundle lives, and their
/// result-dict shapes — everything bundle-defining is here.
pub(crate) fn run_gen_bundle(
    plan: &DiskGraphTTPlan<Seconds>,
    index_root: &Path,
    out_dir: &Path,
    tt_id: &str,
    source: Option<CustomTtSource>,
) -> diskgraph_train_table::errors::TTGenResult<(TTPassOutput, std::path::PathBuf)> {
    let pass_output = execute_tt_pass_with_source(plan, index_root, out_dir, source)?;

    // Cross-job contract: serialize the plan fields the ingest phase consumes
    // plus the per-stage seed parquets gen produced, so the separate ingest
    // job needs neither the in-memory plan nor a demux.
    let mut spec = TtIngestSpec::from_plan(tt_id, plan);
    spec.split_stages = pass_output.stage_gen_infos.clone();
    let spec_path = out_dir.join(TT_INGEST_SPEC_FILENAME);
    let spec_bytes = serde_json::to_vec_pretty(&spec).map_err(|e| {
        diskgraph_train_table::errors::TTGenError::SystemError(format!(
            "failed to encode tt_ingest_spec.json: {e}"
        ))
    })?;
    std::fs::write(&spec_path, spec_bytes)?;
    Ok((pass_output, spec_path))
}

/// Run ONLY the ttgen phase of a TT plan (no demux, no edge ingest, no
/// publish). Writes `tt.parquet`, `split.bin`, `manifest.json`, and the
/// kumo-contract stats sidecars into a single staging dir
/// (`<index>/train_tables/.<tt_id>.staging/`), plus a NEW `tt_ingest_spec.json`
/// that carries everything the separate `execute_tt_ingest` job will need from
/// the (then-unavailable) in-memory plan.
///
/// Returns a dict describing the gen outputs:
/// `{schema_version, tt_id, staging_dir, tt_parquet_path, split_bin_path,
///   ingest_spec_path, row_count}`.
#[pyfunction]
#[tracing::instrument(
    level = "info",
    skip_all,
    fields(index_path = %index_path, tt_id = %tt_id)
)]
pub fn execute_tt_gen(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
    tt_id: &str,
) -> PyResult<Py<PyAny>> {
    validate_tt_id(tt_id)?;
    info!("execute_tt_gen started (ttgen-only) tt_id={}", tt_id);
    let plan: DiskGraphTTPlan<Seconds> = serde_json::from_str(plan_json)
        .map_err(|e| crate::DiskgraphConfigError::new_err(format!("invalid plan JSON: {e}")))?;

    let idx = index_path.to_owned();
    let tt_id_owned = tt_id.to_owned();

    let (staging_dir, tt_parquet_path, ingest_spec_path, row_count): (String, String, String, u64) =
        py.detach(|| -> diskgraph_train_table::errors::TTGenResult<_> {
            let index_root = Path::new(&idx);
            let staging_root = staging_root_for(index_root, &tt_id_owned);
            if staging_root.exists() {
                std::fs::remove_dir_all(&staging_root)?;
            }
            std::fs::create_dir_all(&staging_root)?;

            let (pass_output, spec_path) =
                run_gen_bundle(&plan, index_root, &staging_root, &tt_id_owned, None)?;

            Ok((
                staging_root.to_string_lossy().to_string(),
                pass_output.tt_parquet_path.to_string_lossy().to_string(),
                spec_path.to_string_lossy().to_string(),
                pass_output.rows_written,
            ))
        })
        .map_err(crate::map_ttgen_err)?;

    let dict = PyDict::new(py);
    dict.set_item("schema_version", "diskgraph.tt_gen_result.v1")?;
    dict.set_item("tt_id", tt_id)?;
    dict.set_item("staging_dir", &staging_dir)?;
    dict.set_item("tt_parquet_path", &tt_parquet_path)?;
    dict.set_item("ingest_spec_path", &ingest_spec_path)?;
    dict.set_item("row_count", row_count)?;
    Ok(dict.into())
}

// ─── execute_tt_ingest ───────────────────────────────────────────────────────

/// Run ONLY the ingest/publish phase of a TT plan, consuming the artifacts that
/// a prior [`execute_tt_gen`] job wrote into the staging dir. No plan / fkey
/// args: the staging dir is `parent_dir(tt_parquet_path)`, and `split.bin` +
/// `tt_ingest_spec.json` are read as SIBLINGS of the parquet.
///
/// Runs `resolve_ingest_stages` → `ingest_resolved_stages` (fkey specs
/// reconstructed from `tt_ingest_spec.json`; ingest → `copy_stage_artifacts` →
/// `publish_tt_extension` to `<index>/train_tables/<tt_id>/`) → writes
/// `diskgraph_manifest.json`.
///
/// Returns the SAME dict shape as [`execute_tt_plan`].
///
/// `tt_id` names the TARGET extension; predict may pass a target ≠ the bundle's
/// `spec.tt_id` (per-BP-job fan-out); train/custom passes the same id.
/// Default single-stage key/table-name used by the no-split (prediction) ingest
/// path when the caller passes no `predict_stage_name`.
pub(crate) const DEFAULT_PREDICT_STAGE: &str = "predict";

#[pyfunction]
#[pyo3(signature = (staging_dir, index_path, tt_id, predict_stage_name = None))]
#[tracing::instrument(
    level = "info",
    skip_all,
    fields(index_path = %index_path, tt_id = %tt_id)
)]
pub fn execute_tt_ingest(
    py: Python<'_>,
    // The staging dir produced by `execute_tt_gen` (holds `tt.parquet`,
    // `tt_ingest_spec.json`, and the per-stage seed parquets). The directory
    // IS the gen→ingest exchange unit.
    staging_dir: &str,
    index_path: &str,
    tt_id: &str,
    // Stage name used ONLY on the no-split (prediction) path: the whole
    // The tt.parquet parts are ingested as one stage whose manifest key + on-disk
    // seed-table name derive from this. Ignored on the split path (which always
    // emits train/val/test). Defaults to "predict".
    predict_stage_name: Option<String>,
) -> PyResult<Py<PyAny>> {
    validate_tt_id(tt_id)?;
    info!("execute_tt_ingest started (ingest-only) tt_id={}", tt_id);

    let idx = index_path.to_owned();
    let tt_id_owned = tt_id.to_owned();
    let staging_dir_owned = staging_dir.to_owned();
    let predict_stage_key = predict_stage_name.unwrap_or_else(|| DEFAULT_PREDICT_STAGE.to_string());

    let (
        manifest_stages,
        tt_extension_path,
        unpublished_extension_path,
        is_temporal,
        rhs_node_type,
        key_failures,
        input_row_count,
    ): (
        Vec<ManifestStage>,
        String,
        String,
        bool,
        Option<String>,
        diskgraph_ingest::KeyFailureState,
        u64,
    ) = py
        .detach(|| -> diskgraph_train_table::errors::TTGenResult<_> {
            let index_root = Path::new(&idx);
            let staging_dir = Path::new(&staging_dir_owned).to_path_buf();
            validate_staging_artifact_types(&staging_dir)?;
            validate_train_table_inventory(&staging_dir)?;
            // The well-known `tt.parquet` inside the staging dir.
            let tt_parquet = staging_dir.join("tt.parquet");
            let tt_parquet = tt_parquet.as_path();

            // Read DG's cross-job spec (sibling of tt.parquet) to reconstruct
            // the fkey specs + entity-table fallback the ingest phase needs.
            let spec_path = staging_dir.join(TT_INGEST_SPEC_FILENAME);
            let spec_bytes = std::fs::read(&spec_path).map_err(|e| {
                diskgraph_train_table::errors::TTGenError::InvalidPlan(format!(
                    "execute_tt_ingest: cannot read {}: {e}",
                    spec_path.display()
                ))
            })?;
            let spec: TtIngestSpec = serde_json::from_slice(&spec_bytes).map_err(|e| {
                diskgraph_train_table::errors::TTGenError::InvalidPlan(format!(
                    "execute_tt_ingest: invalid {}: {e}",
                    spec_path.display()
                ))
            })?;
            validate_ingest_spec(&spec)?;

            let tt_root = index_root.join("train_tables").join(&tt_id_owned);
            let ingest_root = ingest_root_for(index_root, &tt_id_owned);
            if ingest_root.exists() {
                std::fs::remove_dir_all(&ingest_root)?;
            }
            let artifacts = StagedTtArtifacts::from_staging_dir(&staging_dir);

            // Both paths feed the SAME ingest: resolution-mode predict seeds are
            // topologically identical to resolution-mode train seeds.
            let stages = resolve_ingest_stages(
                &staging_dir,
                tt_parquet,
                &predict_stage_key,
                spec.split_stages.as_deref(),
            )?;
            // Build the per-stage `bucket_index` only when single-timestamp
            // grouping can actually be used: resolution-mode LP whose RHS/target
            // table is temporal (kumo-computed `lp_rhs_is_temporal`). This
            // mirrors the kumo LOAD gate and legacy's `TimestampSampler` RHS
            // check. Atemporal-RHS / non-LP ingest is byte-identical to before.
            let group_by_timestamp = spec.lp_resolution.is_some() && spec.lp_rhs_is_temporal;
            let resolve_specs = diskgraph_core::tt_builder::stage_resolution_specs(
                &spec.lp_resolution,
                &spec.fkey_mapping,
                &spec.entity_table,
            )
            .map_err(diskgraph_train_table::errors::TTGenError::InvalidPlan)?;
            let (manifest_stages, key_failures, input_row_count) = ingest_resolved_stages(
                &stages,
                &tt_id_owned,
                &ingest_root,
                index_root,
                &resolve_specs,
                &artifacts,
                group_by_timestamp,
            )?;
            // Self-describing manifest fields, recovered from the gen-time spec.
            let rhs_node_type = rhs_node_type_from_resolution(&spec.lp_resolution);
            Ok((
                manifest_stages,
                tt_root.to_string_lossy().to_string(),
                ingest_root.to_string_lossy().to_string(),
                spec.is_temporal,
                rhs_node_type,
                key_failures,
                input_row_count,
            ))
        })
        .map_err(crate::map_ttgen_err)?;

    write_tt_manifest_and_build_result(
        py,
        index_path,
        tt_id,
        &tt_extension_path,
        &unpublished_extension_path,
        &manifest_stages,
        is_temporal,
        rhs_node_type.as_deref(),
        &key_failures,
        input_row_count,
    )
}

/// Read back a stage table's node_count from a freshly-ingested extension's
/// `index_metadata.json`. Used by the no-split path to populate the predict
/// stage's manifest `row_count` (the split path gets counts from the demux).
fn stage_node_count(
    ingest_root: &Path,
    table_name: &str,
) -> diskgraph_train_table::errors::TTGenResult<u64> {
    let metadata_path = ingest_root.join("index_metadata.json");
    let reader = diskgraph_index::graph::GraphIndexReader::open(&metadata_path, ingest_root)
        .map_err(|e| {
            diskgraph_train_table::errors::TTGenError::SystemError(format!(
                "execute_tt_ingest: cannot open published predict extension metadata {}: {e:#}",
                metadata_path.display()
            ))
        })?;
    reader
        .table(table_name)
        .map(|t| t.node_count())
        .ok_or_else(|| {
            diskgraph_train_table::errors::TTGenError::SystemError(format!(
                "execute_tt_ingest: predict stage `{table_name}` missing from published extension"
            ))
        })
}

/// Enforce the link-prediction VAL/TEST single-timeframe contract for a stage
/// whose timestamp grouping was built: `num_buckets` is the distinct-timestamp
/// count, which must be `<= 1` for the `val`/`test` stages (other stage keys are
/// unconstrained). LP val/test eval assumes one anchor seed time per stage, so a
/// multi-timeframe val/test would silently mis-evaluate; fail loud instead.
fn enforce_single_timestamp_eval_stage(
    stage_key: &str,
    num_buckets: usize,
) -> diskgraph_train_table::errors::TTGenResult<()> {
    if (stage_key == "val" || stage_key == "test") && num_buckets > 1 {
        return Err(diskgraph_train_table::errors::TTGenError::InvalidPlan(
            format!(
                "Link prediction requires exactly one timeframe per {stage_key} \
                 set, but found {num_buckets}. Adjust the temporal split to \
                 produce a single timeframe in {stage_key}, or filter the \
                 prediction time range to a single timestamp."
            ),
        ));
    }
    Ok(())
}

/// Ingest a resolved stage list and copy artifacts into an unpublished
/// attempt directory, returning the manifest stages. The ONLY caller-visible
/// difference between the split (3
/// stage) and predict (1 stage) paths is the `stages` list and `split_bin`
/// (`Some` copies split.bin into the extension). Predict-stage counts (those
/// `StageSpec`s with `count = None`) are read back from the freshly-ingested
/// extension before publish.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ingest_resolved_stages(
    stages: &[StageSpec],
    tt_id: &str,
    ingest_root: &Path,
    index_root: &Path,
    resolve_specs: &[RefResolveSpec],
    artifacts: &StagedTtArtifacts,
    // TT-NG temporal-LP timestamp grouping: emit per-stage `bucket_index.bin`.
    // Set only for the temporal-LP seed stages.
    group_by_timestamp: bool,
) -> diskgraph_train_table::errors::TTGenResult<(
    Vec<ManifestStage>,
    diskgraph_ingest::KeyFailureState,
    u64,
)> {
    let stage_ingest = ingest_stage_parquets_into_extension(
        tt_id,
        ingest_root,
        stages,
        index_root,
        resolve_specs,
        group_by_timestamp,
    )?;
    // LP VAL/TEST single-timeframe contract: when ts grouping is built (i.e.
    // temporal RHS), each val/test stage must collapse to a single timestamp
    // bucket. Enforced here (was a kumo-side Python guard) since the bucket-list
    // length is exactly the distinct-ts count, already computed by ingest.
    for s in stages {
        if let Some(grouping) = stage_ingest.ts_groupings.get(&s.table_name) {
            enforce_single_timestamp_eval_stage(&s.key, grouping.buckets.len())?;
        }
    }
    copy_stage_artifacts(artifacts, stages, ingest_root)?;
    // Per-stage row count = surviving node_count (post-drop), split + predict
    // alike. Pre-drop total is `task_dataset.input_row_count`.
    let manifest_stages = stages
        .iter()
        .map(|s| {
            let count = stage_node_count(ingest_root, &s.table_name)?;
            Ok((
                s.key.clone(),
                s.table_name.clone(),
                count,
                s.parquet_paths.len(),
                s.seed_time,
                stage_ingest.ts_groupings.get(&s.table_name).cloned(),
                stage_ingest
                    .binary_label_indexes
                    .get(&s.table_name)
                    .cloned(),
                stage_ingest
                    .sample_weight_indexes
                    .get(&s.table_name)
                    .cloned(),
            ))
        })
        .collect::<diskgraph_train_table::errors::TTGenResult<Vec<_>>>()?;
    Ok((
        manifest_stages,
        stage_ingest.key_failures,
        stage_ingest.input_row_count,
    ))
}

/// One published stage in the manifest `stages` map: `(stage_key, table_name,
/// row_count, part_count, seed_time, ts_grouping)`. `part_count` is how many
/// `<table_name>/part-NNNNN.parquet` files `copy_stage_artifacts` published; the
/// manifest's `parts` list enumerates exactly those. `ts_grouping` is the
/// per-stage TT-NG timestamp grouping (`bucket_index.bin` + bucket list),
/// `None` unless temporal-LP grouping was enabled for this stage.
pub(crate) type ManifestStage = (
    String,
    String,
    u64,
    usize,
    Option<[i64; 2]>,
    Option<diskgraph_ingest::ts_grouping::StageGrouping>,
    Option<diskgraph_ingest::binary_label::BinaryLabelIndex>,
    Option<diskgraph_ingest::sample_weight::SampleWeightIndex>,
);

fn weighted_bucket_to_pydict<'py>(
    py: Python<'py>,
    bucket: &diskgraph_ingest::sample_weight::WeightedBucketIndex,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("node_ids", &bucket.node_ids)?;
    out.set_item("cdf", &bucket.cdf)?;
    out.set_item("count", bucket.count)?;
    out.set_item("total_weight", bucket.total_weight)?;
    Ok(out)
}

fn sample_weight_index_to_pydict<'py>(
    py: Python<'py>,
    index: &diskgraph_ingest::sample_weight::SampleWeightIndex,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("all", weighted_bucket_to_pydict(py, &index.all)?)?;
    if let Some(bucket) = &index.false_bucket {
        out.set_item("false_bucket", weighted_bucket_to_pydict(py, bucket)?)?;
    }
    if let Some(bucket) = &index.true_bucket {
        out.set_item("true_bucket", weighted_bucket_to_pydict(py, bucket)?)?;
    }
    Ok(out)
}

/// Write `<tt_extension>/diskgraph_manifest.json` and return the Python result
/// dict. Shared by [`execute_tt_plan`] and [`execute_tt_ingest`] so both emit
/// byte-identical manifests and dict shapes. `stages` is a variable list of
/// `(stage_key, table_name, row_count)`.
pub(crate) fn write_tt_manifest_and_build_result(
    py: Python<'_>,
    index_path: &str,
    tt_id: &str,
    tt_extension_path: &str,
    unpublished_extension_path: &str,
    stages: &[ManifestStage],
    // Self-describing task-level metadata. `rhs_node_type` is the FK-free LP RHS
    // table (`List`-resolution `dst_table`), `None` for node-prediction.
    is_temporal: bool,
    rhs_node_type: Option<&str>,
    // Orphan-FK skip counters from the stage ingest (see the contract on
    // `diskgraph_ingest::ingest_stage_extension`). Surfaced in both the
    // manifest and the result dict so kumo can warn on cross-graph TT reuse.
    key_failures: &diskgraph_ingest::KeyFailureState,
    // Pre-drop seed total (kept + dropped) — kumo's orphan-% denominator.
    input_row_count: u64,
) -> PyResult<Py<PyAny>> {
    let diskgraph_manifest_path = Path::new(tt_extension_path)
        .join("diskgraph_manifest.json")
        .to_string_lossy()
        .to_string();
    // Drives `task_dataset.columns.weight`, which kumo reads as `has_weight`.
    let has_weight = stages_have_weight_column(unpublished_extension_path, stages);
    let weight_col_json = if has_weight {
        Value::String(diskgraph_train_table::ttgen_output::WEIGHT_COL.to_string())
    } else {
        Value::Null
    };
    // Enumerate the on-disk parts published by `copy_stage_artifacts`.
    let stage_parts = |table_name: &str, part_count: usize| -> Vec<String> {
        (0..part_count)
            .map(|i| stage_part_rel_path(table_name, i))
            .collect()
    };
    let mut stages_json = serde_json::Map::new();
    for (
        stage_key,
        table_name,
        row_count,
        part_count,
        seed_time,
        ts_grouping,
        binary_index,
        sample_weight_index,
    ) in stages
    {
        // TT-NG temporal-LP timestamp grouping: `<table>/bucket_index.bin`
        // (final node-ids grouped by TIMESTAMP) + the bucket list. Present only
        // for grouped stages; omitted entirely otherwise.
        let bucket_index_json = ts_grouping.as_ref().map(|g| {
            json!({
                "path": g.bucket_index_rel_path,
                "buckets": g
                    .buckets
                    .iter()
                    .map(|b| json!({ "ts": b.ts, "offset": b.offset, "count": b.count }))
                    .collect::<Vec<_>>(),
            })
        });
        let mut entry = json!({
            "table_name": table_name,
            "index_path": tt_extension_path,
            "row_count": row_count,
            "parts": stage_parts(table_name, *part_count),
            // Per-stage seed-time (min,max) ns; `null` for the predict stage
            // and non-temporal plans. Kumo reads this for temporal-LP RHS.
            "seed_time": seed_time,
        });
        if let Some(bi) = bucket_index_json {
            entry["bucket_index"] = bi;
        }
        if let Some(index) = binary_index {
            entry["binary_label_index"] = json!({
                "false_node_ids": &index.false_node_ids,
                "false_count": index.false_count,
                "true_node_ids": &index.true_node_ids,
                "true_count": index.true_count,
            });
        }
        if let Some(index) = sample_weight_index {
            entry["sample_weight_index"] = serde_json::to_value(index).map_err(|e| {
                crate::DiskgraphSystemError::new_err(format!(
                    "failed to encode sample_weight_index: {e}"
                ))
            })?;
        }
        stages_json.insert(stage_key.to_string(), entry);
    }
    let key_failures_json = serde_json::to_value(key_failures).map_err(|e| {
        crate::DiskgraphSystemError::new_err(format!(
            "failed to encode ingest key-failure counters: {e}"
        ))
    })?;
    let manifest_json = json!({
        "schema_version": "diskgraph.tt_execute_result.v1",
        "tt_id": tt_id,
        "graph_index_path": index_path,
        "tt_extension_path": tt_extension_path,
        "diskgraph_manifest_path": diskgraph_manifest_path,
        "key_failures": key_failures_json,
        "task_dataset": {
            "schema_version": "diskgraph.task_dataset.v1",
            "graph_index_path": index_path,
            "tt_extension_path": tt_extension_path,
            "is_temporal": is_temporal,
            "rhs_node_type": rhs_node_type,
            "input_row_count": input_row_count,
            "stages": Value::Object(stages_json),
            "columns": {
                "primary_entity": PRIMARY_ENTITY_COL,
                "primary_target": PRIMARY_TARGET_COL,
                "timestamp": TIMESTAMP_COL,
                "weight": weight_col_json,
            },
        },
    });
    let unpublished_manifest_path =
        Path::new(unpublished_extension_path).join("diskgraph_manifest.json");
    write_unpublished_tt_manifest(
        &unpublished_manifest_path,
        &serde_json::to_vec_pretty(&manifest_json).map_err(|e| {
            crate::DiskgraphSystemError::new_err(format!(
                "failed to encode diskgraph TT manifest: {e}"
            ))
        })?,
    )
    .map_err(|e| {
        crate::DiskgraphSystemError::new_err(format!(
            "failed to write unpublished diskgraph TT manifest {}: {e}",
            unpublished_manifest_path.display()
        ))
    })?;
    // The final extension must never be observable without the manifest that
    // describes its files. Publish the complete attempt directory only after
    // the manifest has been fully written and closed in that directory.
    publish_tt_extension(
        Path::new(unpublished_extension_path),
        Path::new(tt_extension_path),
    )
    .map_err(crate::map_ttgen_err)?;

    let dict = PyDict::new(py);
    dict.set_item("schema_version", "diskgraph.tt_execute_result.v1")?;
    dict.set_item("tt_id", tt_id)?;
    dict.set_item("graph_index_path", index_path)?;
    dict.set_item("tt_extension_path", tt_extension_path)?;
    dict.set_item("diskgraph_manifest_path", &diskgraph_manifest_path)?;

    let kf = PyDict::new(py);
    kf.set_item("parse_failures", key_failures.parse_failures)?;
    kf.set_item("lookup_failures", key_failures.lookup_failures)?;
    kf.set_item("membership_failures", key_failures.membership_failures)?;
    kf.set_item("samples", key_failures.samples.clone())?;
    let by_context = PyDict::new(py);
    for (context, counts) in &key_failures.by_context {
        let entry = PyDict::new(py);
        entry.set_item("parse_failures", counts.parse_failures)?;
        entry.set_item("lookup_failures", counts.lookup_failures)?;
        entry.set_item("membership_failures", counts.membership_failures)?;
        by_context.set_item(context, entry)?;
    }
    kf.set_item("by_context", by_context)?;
    dict.set_item("key_failures", kf)?;

    let task_dataset = PyDict::new(py);
    task_dataset.set_item("schema_version", "diskgraph.task_dataset.v1")?;
    task_dataset.set_item("graph_index_path", index_path)?;
    task_dataset.set_item("tt_extension_path", tt_extension_path)?;
    task_dataset.set_item("is_temporal", is_temporal)?;
    task_dataset.set_item("rhs_node_type", rhs_node_type)?;
    task_dataset.set_item("input_row_count", input_row_count)?;

    let stages_dict = PyDict::new(py);
    for (
        stage_key,
        table_name,
        row_count,
        part_count,
        seed_time,
        ts_grouping,
        binary_index,
        sample_weight_index,
    ) in stages
    {
        let stage_dict = PyDict::new(py);
        stage_dict.set_item("table_name", table_name)?;
        stage_dict.set_item("index_path", tt_extension_path)?;
        stage_dict.set_item("row_count", *row_count)?;
        stage_dict.set_item("parts", stage_parts(table_name, *part_count))?;
        stage_dict.set_item("seed_time", seed_time.map(|st| st.to_vec()))?;
        // TT-NG temporal-LP timestamp grouping: bucket-index path + buckets,
        // present only for grouped stages.
        if let Some(g) = ts_grouping {
            let bi = PyDict::new(py);
            bi.set_item("path", &g.bucket_index_rel_path)?;
            let buckets: Vec<(i64, u64, u64)> = g
                .buckets
                .iter()
                .map(|b| (b.ts, b.offset, b.count))
                .collect();
            bi.set_item("buckets", buckets)?;
            stage_dict.set_item("bucket_index", bi)?;
        }
        if let Some(index) = binary_index {
            let bi = PyDict::new(py);
            bi.set_item("false_node_ids", &index.false_node_ids)?;
            bi.set_item("false_count", index.false_count)?;
            bi.set_item("true_node_ids", &index.true_node_ids)?;
            bi.set_item("true_count", index.true_count)?;
            stage_dict.set_item("binary_label_index", bi)?;
        }
        if let Some(index) = sample_weight_index {
            stage_dict.set_item(
                "sample_weight_index",
                sample_weight_index_to_pydict(py, index)?,
            )?;
        }
        stages_dict.set_item(stage_key, stage_dict)?;
    }
    task_dataset.set_item("stages", stages_dict)?;

    let columns = PyDict::new(py);
    columns.set_item("primary_entity", PRIMARY_ENTITY_COL)?;
    columns.set_item("primary_target", PRIMARY_TARGET_COL)?;
    columns.set_item("timestamp", TIMESTAMP_COL)?;
    if has_weight {
        columns.set_item("weight", diskgraph_train_table::ttgen_output::WEIGHT_COL)?;
    } else {
        columns.set_item("weight", py.None())?;
    }
    task_dataset.set_item("columns", columns)?;
    dict.set_item("task_dataset", task_dataset)?;
    Ok(dict.into())
}

pub(crate) fn validate_tt_id(tt_id: &str) -> PyResult<()> {
    if tt_id.is_empty() {
        return Err(crate::DiskgraphConfigError::new_err(
            "execute_tt_plan: tt_id is required and must be non-empty",
        ));
    }
    if tt_id == "."
        || tt_id == ".."
        || tt_id.contains('/')
        || tt_id.contains('\\')
        || tt_id.contains('\0')
    {
        return Err(crate::DiskgraphConfigError::new_err(format!(
            "execute_tt_plan: tt_id must be filename-safe, got {tt_id:?}"
        )));
    }
    Ok(())
}

#[pyfunction]
pub fn drop_tt(index_path: &str, tt_id: &str) -> PyResult<()> {
    validate_tt_id(tt_id)?;
    let path = Path::new(index_path).join("train_tables").join(tt_id);
    match std::fs::remove_dir_all(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(crate::DiskgraphSystemError::new_err(format!(
            "drop_tt failed for {}: {e}",
            path.display()
        ))),
    }
}

/// Run a batch-prediction TT-gen plan (no split) and write the prediction
/// table to `output_path`. Runs one `execute_tt_pass` and moves its
/// `tt.parquet` into place; the compiler emits the BP shape natively (entities
/// only, no target), so no rewrite is needed. Returns the row count. Writes
/// locally; the caller stages to remote storage if the destination is remote.
#[pyfunction]
#[tracing::instrument(
    level = "info",
    skip_all,
    fields(index_path = %index_path, output_path = %output_path)
)]
pub fn execute_bp_plan(
    py: Python<'_>,
    plan_json: &str,
    index_path: &str,
    output_path: &str,
) -> PyResult<u64> {
    let plan: DiskGraphTTPlan<Seconds> = serde_json::from_str(plan_json)
        .map_err(|e| crate::DiskgraphConfigError::new_err(format!("invalid plan JSON: {e}")))?;

    let idx = index_path.to_owned();
    let out = output_path.to_owned();

    let rows = py
        .detach(|| -> diskgraph_train_table::errors::TTGenResult<u64> {
            let index_root = Path::new(&idx);
            let output = Path::new(&out);
            let output_parent = output.parent().ok_or_else(|| {
                diskgraph_train_table::errors::TTGenError::InvalidPlan(format!(
                    "execute_bp_plan: output_path has no parent dir: {out}"
                ))
            })?;
            std::fs::create_dir_all(output_parent)?;

            // Stage the pass in a hidden sibling dir, then move the tt.parquet
            // dataset into place (the rest of the pass output is dropped).
            let stem = output
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("prediction");
            let staging_root = output_parent.join(format!(".{stem}.bp_staging"));
            if staging_root.exists() {
                std::fs::remove_dir_all(&staging_root)?;
            }
            std::fs::create_dir_all(&staging_root)?;

            let pass_output = execute_tt_pass(&plan, index_root, &staging_root)?;
            std::fs::rename(&pass_output.tt_parquet_path, output)?;
            let _ = std::fs::remove_dir_all(&staging_root);
            Ok(pass_output.rows_written)
        })
        .map_err(crate::map_ttgen_err)?;

    Ok(rows)
}

/// Stage-table ingest over a variable list of stages (split: `train/val/test_seeds`;
/// predict: one stage). Columns inferred from the first stage's parquet (shared schema).
pub(crate) fn ingest_stage_parquets_into_extension(
    tt_id: &str,
    tt_root: &Path,
    stages_in: &[StageSpec],
    graph_index_path: &Path,
    // From `stage_resolution_specs` (at least the scalar `ENTITY1`).
    resolve_specs: &[RefResolveSpec],
    // TT-NG temporal-LP timestamp grouping: when true, every stage table emits a
    // `<table>/bucket_index.bin` (final node-ids grouped by TIMESTAMP) + bucket
    // list. Set only for the temporal-LP seed stages. Returns the per-stage
    // groupings (empty when disabled).
    group_by_timestamp: bool,
) -> diskgraph_train_table::errors::TTGenResult<diskgraph_ingest::StageExtensionResult> {
    let first_stage = stages_in.first().ok_or_else(|| {
        diskgraph_train_table::errors::TTGenError::InvalidPlan(
            "ingest_stage_parquets_into_extension: no stages supplied".to_string(),
        )
    })?;
    // All stages (and all parts within a stage) share one schema; infer from the
    // first part of the first stage.
    let column_source = first_stage.parquet_paths.first().ok_or_else(|| {
        diskgraph_train_table::errors::TTGenError::InvalidPlan(
            "ingest_stage_parquets_into_extension: first stage has no parquet parts".to_string(),
        )
    })?;
    let columns = diskgraph_ingest::infer_stage_columns(column_source).map_err(|e| {
        diskgraph_train_table::errors::TTGenError::SchemaError(format!(
            "TT stage column inference: {e:#}"
        ))
    })?;

    let stages: Vec<diskgraph_ingest::StageSpec> = stages_in
        .iter()
        .map(|s| diskgraph_ingest::StageSpec {
            stage_table_name: s.table_name.clone(),
            parquet_paths: s.parquet_paths.clone(),
        })
        .collect();
    let stage_sidecars: std::collections::HashMap<String, diskgraph_ingest::StageSidecarColumns> =
        stages_in
        .iter()
        .filter_map(|s| {
            if s.binary_label_column.is_none() && s.sample_weight_column.is_none() {
                return None;
            }
            Some((
                s.table_name.clone(),
                diskgraph_ingest::StageSidecarColumns {
                    binary_label_column: s.binary_label_column.clone(),
                    sample_weight_column: s.sample_weight_column.clone(),
                },
            ))
        })
        .collect();

    // Resolve reference columns to node-id rowpack columns (no FK edge).
    let resolve_specs: Vec<diskgraph_ingest::ResolveSpec> = resolve_specs
        .iter()
        .map(|r| diskgraph_ingest::ResolveSpec {
            src_col: r.src_col.clone(),
            dst_table: r.dst_table.clone(),
            kind: match r.kind {
                RefResolveKind::Scalar => diskgraph_ingest::ResolveKind::Scalar,
                RefResolveKind::List => diskgraph_ingest::ResolveKind::List,
            },
            node_id_col: r.node_id_col.clone(),
        })
        .collect();
    diskgraph_ingest::ingest_stage_extension_with_resolution_and_sidecars(
        &format!("tt_seeds_{tt_id}"),
        tt_root,
        graph_index_path,
        &stages,
        stage_sidecars,
        columns,
        Vec::new(), // no FK edges
        &resolve_specs,
        None,
        true,
        group_by_timestamp,
    )
    .map_err(|e| {
        diskgraph_train_table::errors::TTGenError::SystemError(format!(
            "TT resolution-mode stage-table ingest failed: {e:#}"
        ))
    })
}

pub(crate) fn publish_tt_extension(
    ingest_root: &Path,
    tt_root: &Path,
) -> diskgraph_train_table::errors::TTGenResult<()> {
    publish_tt_extension_with_rename(ingest_root, tt_root, |src, dst| std::fs::rename(src, dst))
}

fn publish_tt_extension_with_rename<F>(
    ingest_root: &Path,
    tt_root: &Path,
    mut rename: F,
) -> diskgraph_train_table::errors::TTGenResult<()>
where
    F: FnMut(&Path, &Path) -> std::io::Result<()>,
{
    recover_tt_extension_publication(tt_root, &mut rename)?;

    let manifest = ingest_root.join("diskgraph_manifest.json");
    if !is_complete_tt_extension_for(ingest_root, tt_root) {
        return Err(diskgraph_train_table::errors::TTGenError::SystemError(
            format!(
                "refusing to publish incomplete or target-mismatched TT extension at {} (manifest {})",
                ingest_root.display(),
                manifest.display()
            ),
        ));
    }

    let previous_root = tt_extension_backup_path(tt_root)?;
    let had_previous = tt_root.exists();
    if had_previous {
        rename(tt_root, &previous_root).map_err(diskgraph_train_table::errors::TTGenError::from)?;
        sync_tt_extension_parent(tt_root)?;
    }
    if let Err(publish_error) = rename(ingest_root, tt_root) {
        if had_previous {
            if let Err(rollback_error) = rename(&previous_root, tt_root) {
                return Err(diskgraph_train_table::errors::TTGenError::SystemError(
                    format!(
                        "failed to publish TT extension: {publish_error}; \
                         failed to restore prior extension: {rollback_error}"
                    ),
                ));
            }
            sync_tt_extension_parent(tt_root)?;
        }
        return Err(diskgraph_train_table::errors::TTGenError::from(
            publish_error,
        ));
    }
    sync_tt_extension_parent(tt_root)?;
    if had_previous {
        std::fs::remove_dir_all(&previous_root)
            .map_err(diskgraph_train_table::errors::TTGenError::from)?;
        sync_tt_extension_parent(tt_root)?;
    }
    Ok(())
}

fn tt_extension_backup_path(
    tt_root: &Path,
) -> diskgraph_train_table::errors::TTGenResult<std::path::PathBuf> {
    let name = tt_root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            diskgraph_train_table::errors::TTGenError::SystemError(format!(
                "TT extension path has no UTF-8 file name: {}",
                tt_root.display()
            ))
        })?;
    Ok(tt_root.with_file_name(format!(".{name}.previous")))
}

fn is_complete_tt_extension_for(path: &Path, tt_root: &Path) -> bool {
    let Ok(root_metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !root_metadata.file_type().is_dir() || root_metadata.file_type().is_symlink() {
        return false;
    }
    let manifest_path = path.join("diskgraph_manifest.json");
    let Ok(manifest_metadata) = std::fs::symlink_metadata(&manifest_path) else {
        return false;
    };
    if !manifest_metadata.file_type().is_file() || manifest_metadata.file_type().is_symlink() {
        return false;
    }
    let Ok(manifest) = std::fs::File::open(&manifest_path) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_reader::<_, Value>(manifest) else {
        return false;
    };
    let expected_tt_id = tt_root.file_name().and_then(|name| name.to_str());
    let Some(train_tables_root) = tt_root.parent() else {
        return false;
    };
    if train_tables_root.file_name().and_then(|name| name.to_str()) != Some("train_tables") {
        return false;
    }
    let Some(graph_index_path) = train_tables_root.parent() else {
        return false;
    };
    let expected_manifest_path = tt_root.join("diskgraph_manifest.json");
    if manifest.get("schema_version").and_then(Value::as_str)
        != Some("diskgraph.tt_execute_result.v1")
        || manifest.get("tt_id").and_then(Value::as_str) != expected_tt_id
        || !manifest_path_matches(&manifest, "graph_index_path", graph_index_path)
        || !manifest_path_matches(&manifest, "tt_extension_path", tt_root)
        || !manifest_path_matches(
            &manifest,
            "diskgraph_manifest_path",
            &expected_manifest_path,
        )
    {
        return false;
    }

    let Some(task_dataset) = manifest.get("task_dataset").and_then(Value::as_object) else {
        return false;
    };
    if task_dataset.get("schema_version").and_then(Value::as_str)
        != Some("diskgraph.task_dataset.v1")
        || !object_path_matches(task_dataset, "graph_index_path", graph_index_path)
        || !object_path_matches(task_dataset, "tt_extension_path", tt_root)
        || !task_dataset
            .get("is_temporal")
            .is_some_and(Value::is_boolean)
        || !task_dataset
            .get("input_row_count")
            .is_some_and(Value::is_u64)
        || !valid_task_dataset_columns(task_dataset.get("columns"))
    {
        return false;
    }
    let Some(stages) = task_dataset.get("stages").and_then(Value::as_object) else {
        return false;
    };
    !stages.is_empty()
        && stages
            .values()
            .all(|stage| valid_manifest_stage(path, tt_root, stage))
}

fn manifest_path_matches(manifest: &Value, key: &str, expected: &Path) -> bool {
    manifest
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|actual| Path::new(actual) == expected)
}

fn object_path_matches(
    object: &serde_json::Map<String, Value>,
    key: &str,
    expected: &Path,
) -> bool {
    object
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|actual| Path::new(actual) == expected)
}

fn valid_task_dataset_columns(columns: Option<&Value>) -> bool {
    let Some(columns) = columns.and_then(Value::as_object) else {
        return false;
    };
    columns.get("primary_entity").and_then(Value::as_str) == Some(PRIMARY_ENTITY_COL)
        && columns.get("primary_target").and_then(Value::as_str) == Some(PRIMARY_TARGET_COL)
        && columns.get("timestamp").and_then(Value::as_str) == Some(TIMESTAMP_COL)
        && columns
            .get("weight")
            .is_some_and(|weight| weight.is_null() || weight.is_string())
}

fn valid_manifest_stage(extension_root: &Path, tt_root: &Path, stage: &Value) -> bool {
    let Some(stage) = stage.as_object() else {
        return false;
    };
    if stage
        .get("table_name")
        .and_then(Value::as_str)
        .is_none_or(|name| name.is_empty())
        || !object_path_matches(stage, "index_path", tt_root)
        || !stage.get("row_count").is_some_and(Value::is_u64)
        || !(stage.get("seed_time").is_some_and(Value::is_null)
            || stage.get("seed_time").is_some_and(Value::is_array))
    {
        return false;
    }
    let Some(parts) = stage.get("parts").and_then(Value::as_array) else {
        return false;
    };
    if parts.is_empty()
        || !parts.iter().all(|part| {
            part.as_str()
                .and_then(|path| owned_regular_file_len(extension_root, path))
                .is_some_and(|size| size > 0)
        })
    {
        return false;
    }
    stage
        .get("bucket_index")
        .is_none_or(|index| valid_bucket_index(extension_root, index))
        && stage
            .get("binary_label_index")
            .is_none_or(|index| valid_binary_label_index(extension_root, index))
        && stage
            .get("sample_weight_index")
            .is_none_or(|index| valid_sample_weight_index(extension_root, index))
}

fn valid_bucket_index(extension_root: &Path, index: &Value) -> bool {
    let Some(index) = index.as_object() else {
        return false;
    };
    let Some(path) = index.get("path").and_then(Value::as_str) else {
        return false;
    };
    let Some(size) = owned_regular_file_len(extension_root, path) else {
        return false;
    };
    let Some(buckets) = index.get("buckets").and_then(Value::as_array) else {
        return false;
    };
    let mut expected_offset = 0_u64;
    for bucket in buckets {
        let Some(bucket) = bucket.as_object() else {
            return false;
        };
        let Some(offset) = bucket.get("offset").and_then(Value::as_u64) else {
            return false;
        };
        let Some(count) = bucket.get("count").and_then(Value::as_u64) else {
            return false;
        };
        if !bucket.get("ts").is_some_and(Value::is_i64) || offset != expected_offset {
            return false;
        }
        let Some(next_offset) = offset.checked_add(count) else {
            return false;
        };
        expected_offset = next_offset;
    }
    expected_offset
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .is_some_and(|expected_size| expected_size == size)
}

fn valid_binary_label_index(extension_root: &Path, index: &Value) -> bool {
    let Some(index) = index.as_object() else {
        return false;
    };
    [
        ("false_node_ids", "false_count"),
        ("true_node_ids", "true_count"),
    ]
    .into_iter()
    .all(|(path_key, count_key)| {
        let Some(path) = index.get(path_key).and_then(Value::as_str) else {
            return false;
        };
        let Some(count) = index.get(count_key).and_then(Value::as_u64) else {
            return false;
        };
        owned_regular_file_len(extension_root, path)
            == count.checked_mul(std::mem::size_of::<u64>() as u64)
    })
}

fn valid_sample_weight_index(extension_root: &Path, index: &Value) -> bool {
    let Some(index) = index.as_object() else {
        return false;
    };
    index
        .get("all")
        .is_some_and(|bucket| valid_weighted_bucket(extension_root, bucket))
        && index
            .get("false_bucket")
            .is_none_or(|bucket| valid_weighted_bucket(extension_root, bucket))
        && index
            .get("true_bucket")
            .is_none_or(|bucket| valid_weighted_bucket(extension_root, bucket))
}

fn valid_weighted_bucket(extension_root: &Path, bucket: &Value) -> bool {
    let Some(bucket) = bucket.as_object() else {
        return false;
    };
    let Some(count) = bucket.get("count").and_then(Value::as_u64) else {
        return false;
    };
    if !bucket
        .get("total_weight")
        .and_then(Value::as_f64)
        .is_some_and(f64::is_finite)
    {
        return false;
    }
    let Some(expected_size) = count.checked_mul(std::mem::size_of::<u64>() as u64) else {
        return false;
    };
    ["node_ids", "cdf"].into_iter().all(|key| {
        bucket
            .get(key)
            .and_then(Value::as_str)
            .and_then(|path| owned_regular_file_len(extension_root, path))
            == Some(expected_size)
    })
}

fn owned_regular_file_len(root: &Path, relative_path: &str) -> Option<u64> {
    let relative_path = Path::new(relative_path);
    if relative_path.as_os_str().is_empty()
        || !relative_path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        return None;
    }
    let components = relative_path.components().collect::<Vec<_>>();
    let mut current = root.to_path_buf();
    for (index, component) in components.iter().enumerate() {
        current.push(component.as_os_str());
        let metadata = std::fs::symlink_metadata(&current).ok()?;
        if metadata.file_type().is_symlink() {
            return None;
        }
        if index + 1 == components.len() {
            return metadata.file_type().is_file().then_some(metadata.len());
        }
        if !metadata.file_type().is_dir() {
            return None;
        }
    }
    None
}

fn write_unpublished_tt_manifest(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let mut manifest = std::fs::File::create(path)?;
    manifest.write_all(contents)?;
    manifest.sync_all()?;
    drop(manifest);
    sync_directory_if_supported(path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("TT manifest path has no parent: {}", path.display()),
        )
    })?)
}

fn remove_tt_extension_path(path: &Path) -> diskgraph_train_table::errors::TTGenResult<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
    .map_err(diskgraph_train_table::errors::TTGenError::from)
}

fn sync_tt_extension_parent(tt_root: &Path) -> diskgraph_train_table::errors::TTGenResult<()> {
    let parent = tt_root.parent().ok_or_else(|| {
        diskgraph_train_table::errors::TTGenError::SystemError(format!(
            "TT extension path has no parent: {}",
            tt_root.display()
        ))
    })?;
    sync_directory_if_supported(parent).map_err(diskgraph_train_table::errors::TTGenError::from)
}

fn sync_directory_if_supported(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)?.sync_all()?;
    }
    Ok(())
}

fn recover_tt_extension_publication<F>(
    tt_root: &Path,
    rename: &mut F,
) -> diskgraph_train_table::errors::TTGenResult<()>
where
    F: FnMut(&Path, &Path) -> std::io::Result<()>,
{
    let parent = tt_root.parent().ok_or_else(|| {
        diskgraph_train_table::errors::TTGenError::SystemError(format!(
            "TT extension path has no parent: {}",
            tt_root.display()
        ))
    })?;
    let expected_backup = tt_extension_backup_path(tt_root)?;
    let backup_prefix = expected_backup
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            diskgraph_train_table::errors::TTGenError::SystemError(format!(
                "TT extension backup path has no UTF-8 file name: {}",
                expected_backup.display()
            ))
        })?;
    let entries = std::fs::read_dir(parent)
        .map_err(diskgraph_train_table::errors::TTGenError::from)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(diskgraph_train_table::errors::TTGenError::from)?;
    let backups = entries
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(backup_prefix))
        })
        .collect::<Vec<_>>();

    if backups.len() > 1 || backups.first().is_some_and(|path| path != &expected_backup) {
        return Err(diskgraph_train_table::errors::TTGenError::SystemError(
            format!(
                "ambiguous TT extension publication backups for {}: {:?}",
                tt_root.display(),
                backups
            ),
        ));
    }

    let backup_exists = expected_backup.exists();
    let final_exists = tt_root.exists();
    match (final_exists, backup_exists) {
        (false, false) => Ok(()),
        (false, true) => {
            if !is_complete_tt_extension_for(&expected_backup, tt_root) {
                return Err(diskgraph_train_table::errors::TTGenError::SystemError(
                    format!(
                        "cannot recover incomplete TT extension backup {}",
                        expected_backup.display()
                    ),
                ));
            }
            rename(&expected_backup, tt_root)
                .map_err(diskgraph_train_table::errors::TTGenError::from)?;
            sync_tt_extension_parent(tt_root)
        }
        (true, false) => {
            if is_complete_tt_extension_for(tt_root, tt_root) {
                Ok(())
            } else {
                Err(diskgraph_train_table::errors::TTGenError::SystemError(
                    format!(
                        "incomplete TT extension has no recoverable backup: {}",
                        tt_root.display()
                    ),
                ))
            }
        }
        (true, true) => {
            if !is_complete_tt_extension_for(&expected_backup, tt_root) {
                return Err(diskgraph_train_table::errors::TTGenError::SystemError(
                    format!(
                        "cannot recover with incomplete TT extension backup {}",
                        expected_backup.display()
                    ),
                ));
            }
            if is_complete_tt_extension_for(tt_root, tt_root) {
                remove_tt_extension_path(&expected_backup)?;
                sync_tt_extension_parent(tt_root)
            } else {
                remove_tt_extension_path(tt_root)?;
                sync_tt_extension_parent(tt_root)?;
                rename(&expected_backup, tt_root)
                    .map_err(diskgraph_train_table::errors::TTGenError::from)?;
                sync_tt_extension_parent(tt_root)
            }
        }
    }
}

/// Source paths for the TT artifacts that were written by the gen phase into
/// the staging dir. Decouples [`copy_stage_artifacts`] from the in-memory
/// [`TTPassOutput`] so the ingest job (which has no `TTPassOutput`, only the
/// on-disk staging dir) can drive the copy from sibling files.
pub(crate) struct StagedTtArtifacts {
    /// Directory holding `tt.parquet`, `manifest.json`, the per-stage seed
    /// parquets, and the kumo-contract stats sidecars.
    pub staging_dir: std::path::PathBuf,
    pub tt_parquet_path: std::path::PathBuf,
    pub manifest_path: std::path::PathBuf,
}

impl StagedTtArtifacts {
    /// Build from a freshly-run `TTPassOutput` (fused / gen phase). The
    /// staging dir is `manifest_path`'s parent (where the executor wrote the
    /// stats sidecars + per-stage seed parquets).
    pub(crate) fn from_pass_output(
        pass_output: &TTPassOutput,
    ) -> diskgraph_train_table::errors::TTGenResult<Self> {
        let staging_dir = pass_output
            .manifest_path
            .parent()
            .ok_or_else(|| {
                diskgraph_train_table::errors::TTGenError::InvalidPlan(
                    "pass_output.manifest_path has no parent dir".to_string(),
                )
            })?
            .to_path_buf();
        Ok(Self {
            staging_dir,
            tt_parquet_path: pass_output.tt_parquet_path.clone(),
            manifest_path: pass_output.manifest_path.clone(),
        })
    }

    /// Build from an on-disk staging dir (ingest phase). All artifacts are
    /// siblings of `tt.parquet`.
    fn from_staging_dir(staging_dir: &Path) -> Self {
        Self {
            staging_dir: staging_dir.to_path_buf(),
            tt_parquet_path: staging_dir.join("tt.parquet"),
            manifest_path: staging_dir.join("manifest.json"),
        }
    }
}

/// Relative published path of a stage's `part_idx`-th seed parquet:
/// `<table_name>/part-NNNNN.parquet`. The on-disk layout and the manifest's
/// per-stage `parts` list MUST stay in lockstep — both go through here.
fn stage_part_rel_path(table_name: &str, part_idx: usize) -> String {
    format!("{table_name}/part-{part_idx:05}.parquet")
}

/// Whether any published stage carries the reserved `WEIGHT` column. Probes
/// one part per stage (schema is identical across a stage's parts); returns
/// `false` if no part is readable. Drives the manifest `weight` column.
fn stages_have_weight_column(tt_extension_path: &str, stages: &[ManifestStage]) -> bool {
    let root = Path::new(tt_extension_path);
    for (
        _stage_key,
        table_name,
        _row_count,
        part_count,
        _seed_time,
        _ts_grouping,
        _binary_index,
        _sample_weight_index,
    ) in stages
    {
        for part_idx in 0..*part_count {
            let part_path = root.join(stage_part_rel_path(table_name, part_idx));
            let Ok(file) = std::fs::File::open(&part_path) else {
                continue;
            };
            let Ok(builder) = ParquetRecordBatchReaderBuilder::try_new(file) else {
                continue;
            };
            if builder
                .schema()
                .fields()
                .iter()
                .any(|f| f.name() == diskgraph_train_table::ttgen_output::WEIGHT_COL)
            {
                return true;
            }
            // Schema is identical across parts of a stage; one probe is enough.
            break;
        }
    }
    false
}

/// Copy the published TT artifacts into the (unpublished) extension: tt.parquet parts,
/// manifest, each stage's seed parquet part(s) under `<table_name>/part-*.parquet`,
/// and the kumo-contract stats sidecars.
fn copy_stage_artifacts(
    artifacts: &StagedTtArtifacts,
    stages: &[StageSpec],
    tt_root: &Path,
) -> diskgraph_train_table::errors::TTGenResult<()> {
    let published_dataset = tt_root.join("tt.parquet");
    std::fs::create_dir(&published_dataset)?;
    for part in diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(
        &artifacts.tt_parquet_path,
    )? {
        std::fs::copy(
            &part,
            published_dataset.join(part.file_name().expect("dataset part has a filename")),
        )?;
    }
    std::fs::copy(&artifacts.manifest_path, tt_root.join("manifest.json"))?;
    for stage in stages {
        let stage_dir = tt_root.join(&stage.table_name);
        std::fs::create_dir_all(&stage_dir)?;
        for (part_idx, part) in stage.parquet_paths.iter().enumerate() {
            std::fs::copy(
                part,
                tt_root.join(stage_part_rel_path(&stage.table_name, part_idx)),
            )?;
        }
    }
    for sidecar in [
        "stats.json",
        "split_stats.json",
        "train_table_col_stats.json",
    ] {
        let src = artifacts.staging_dir.join(sidecar);
        if src.exists() {
            std::fs::copy(&src, tt_root.join(sidecar))?;
        }
    }
    Ok(())
}

// ─── read_tt_parquet_batches ─────────────────────────────────────────────────

/// Iterator over `RecordBatch`es of a TT parquet directory dataset. Yields
/// `pyarrow.RecordBatch` objects. Created via [`read_tt_parquet_batches`].
///
/// The reader is opened lazily on the first `__next__` call so that
/// constructing the iterator in Python is cheap. The underlying file is
/// closed when the iterator is exhausted or dropped.
///
/// `unsendable` because `parquet::arrow::arrow_reader::ParquetRecordBatchReader`
/// holds non-`Sync` `Box<dyn ArrayReader>` state. The iterator is owned by
/// one Python thread for its lifetime — sharing across threads is meaningless
/// for a single-pass iterator anyway.
#[pyclass(unsendable)]
pub struct TTParquetBatchIterator {
    dataset_path: String,
    columns: Option<Vec<String>>,
    parquet_paths: Option<Vec<std::path::PathBuf>>,
    next_part: usize,
    reader: Option<ParquetRecordBatchReader>,
    schema: Option<SchemaRef>,
}

impl TTParquetBatchIterator {
    fn ensure_reader(&mut self) -> PyResult<bool> {
        if self.reader.is_some() {
            return Ok(true);
        }
        if self.parquet_paths.is_none() {
            self.parquet_paths = Some(
                diskgraph_train_table::parquet_dataset::ordered_tt_dataset_parts(Path::new(
                    &self.dataset_path,
                ))
                .map_err(crate::map_ttgen_err)?,
            );
        }
        let paths = self
            .parquet_paths
            .as_ref()
            .expect("dataset parts initialized");
        let Some(path) = paths.get(self.next_part) else {
            return Ok(false);
        };
        self.next_part += 1;
        let file = File::open(path).map_err(|e| {
            crate::DiskgraphSystemError::new_err(format!(
                "cannot open {} for read: {e}",
                path.display()
            ))
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            crate::DiskgraphSystemError::new_err(format!(
                "cannot read parquet metadata for {}: {e}",
                path.display()
            ))
        })?;
        let builder = if let Some(cols) = &self.columns {
            let parquet_schema = builder.parquet_schema();
            let mut indices = Vec::with_capacity(cols.len());
            let arrow_schema = builder.schema();
            for col in cols {
                let idx = arrow_schema.index_of(col).map_err(|_| {
                    crate::DiskgraphDataError::new_err(format!(
                        "column not found in {}: {col}",
                        path.display()
                    ))
                })?;
                indices.push(idx);
            }
            let mask = ProjectionMask::roots(parquet_schema, indices);
            builder.with_projection(mask)
        } else {
            builder
        };
        let reader = builder.build().map_err(|e| {
            crate::DiskgraphSystemError::new_err(format!(
                "cannot build parquet reader for {}: {e}",
                path.display()
            ))
        })?;
        let schema = reader.schema().clone();
        if self
            .schema
            .as_ref()
            .is_some_and(|expected| expected != &schema)
        {
            return Err(crate::DiskgraphDataError::new_err(format!(
                "training table part schema mismatch: {}",
                path.display()
            )));
        }
        self.reader = Some(reader);
        self.schema = Some(schema);
        Ok(true)
    }
}

#[pymethods]
impl TTParquetBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        loop {
            if !self.ensure_reader()? {
                return Ok(None);
            }
            let next = self.reader.as_mut().expect("reader initialized").next();
            match next {
                None => self.reader = None,
            Some(Ok(batch)) => {
                let py_obj = batch.to_pyarrow(py)?;
                    return Ok(Some(py_obj.into()));
            }
                Some(Err(e)) => {
                    return Err(crate::DiskgraphSystemError::new_err(format!(
                "parquet read error: {e}"
                    )))
                }
            }
        }
    }

    /// Pyarrow schema of the underlying file (after column projection if
    /// supplied at construction). Forces opening the file.
    fn schema(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if !self.ensure_reader()? {
            return Err(crate::DiskgraphDataError::new_err(format!(
                "training table dataset is empty: {}",
                self.dataset_path
            )));
        }
        let schema = self
            .schema
            .as_ref()
            .ok_or_else(|| crate::DiskgraphSystemError::new_err("schema unavailable"))?
            .clone();
        let empty_batch = RecordBatch::new_empty(schema);
        let py_batch = empty_batch.to_pyarrow(py)?;
        let py_schema = py_batch.getattr("schema")?;
        Ok(py_schema.unbind())
    }
}

/// Open `parquet_path` and return an iterator over its `RecordBatch`es.
///
/// `columns` optionally projects to a subset of column names (in the
/// returned order matching the file's column-definition order). Missing
/// column names raise a `DiskgraphDataError`.
///
/// This is the BP / batch-prediction bridge for kumo's
/// `diskgraph_batch_pred.py` — caller iterates batches without materializing
/// the whole table.
#[pyfunction]
#[pyo3(signature = (parquet_path, columns = None))]
pub fn read_tt_parquet_batches(
    parquet_path: &str,
    columns: Option<Vec<String>>,
) -> PyResult<TTParquetBatchIterator> {
    Ok(TTParquetBatchIterator {
        dataset_path: parquet_path.to_owned(),
        columns,
        parquet_paths: None,
        next_part: 0,
        reader: None,
        schema: None,
    })
}

// ─── Prediction artifact bundle helpers ──────────────────────────────────────
//
// These build JSON blobs describing per-table (fkey + reverse-edge + mask)
// artifacts and a manifest for the prediction-table workflow. Used by the
// Python `prediction_artifacts` module — kept after the legacy TT-gen path was removed because they
// are NOT part of the deleted materialize-with-split path.

fn build_table_edges_bundle_value(
    output_dir: &Path,
    table_name: &str,
    mask_present: bool,
    mask_absent_reason: Option<&str>,
    edge_attrs_path: Option<&str>,
    artifact_kind: &str,
    artifact_format: &str,
    artifact_label: &str,
) -> anyhow::Result<Value> {
    let fkey_path = output_dir.join(format!("{}_fkey_cols.parquet", table_name));
    if !fkey_path.exists() {
        return Err(anyhow::anyhow!(
            "fkey_cols artifact missing for {artifact_label} bundle: {}",
            fkey_path.display()
        ));
    }

    let mut reverse_edges = serde_json::Map::new();
    let mut reverse_checksums = serde_json::Map::new();
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.contains("__rev_") || !name.ends_with(&format!("__{table_name}")) {
            continue;
        }
        let edge_path = path.join("edge.parquet");
        if !edge_path.exists() {
            continue;
        }
        reverse_edges.insert(
            name.to_string(),
            json!({"path": edge_path.to_string_lossy(), "format": "edge_parquet"}),
        );
        reverse_checksums.insert(name.to_string(), json!(sha256_hex_path(&edge_path)?));
    }
    if reverse_edges.is_empty() {
        return Err(anyhow::anyhow!(
            "no reverse edge artifacts discovered for {artifact_label} bundle in {}",
            output_dir.display()
        ));
    }

    let mask_value = if mask_present {
        let mask_path = output_dir.join(format!("{}_mask.parquet", table_name));
        if !mask_path.exists() {
            return Err(anyhow::anyhow!(
                "mask artifact missing for {artifact_label} bundle: {}",
                mask_path.display()
            ));
        }
        json!({
            "present": true,
            "path": mask_path.to_string_lossy(),
            "format": "mask_parquet",
        })
    } else {
        json!({
            "present": false,
            "reason": mask_absent_reason.unwrap_or("mask_not_required"),
        })
    };

    let mut checksums = serde_json::Map::new();
    checksums.insert(
        "fkey_cols_sha256".to_string(),
        json!(sha256_hex_path(&fkey_path)?),
    );
    for (k, v) in &reverse_checksums {
        checksums.insert(format!("{}_sha256", k), v.clone());
    }
    if mask_present {
        let mask_path = output_dir.join(format!("{}_mask.parquet", table_name));
        checksums.insert(
            "mask_sha256".to_string(),
            json!(sha256_hex_path(&mask_path)?),
        );
    }

    let edge_attrs_value = if let Some(path) = edge_attrs_path {
        let p = Path::new(path);
        if !p.exists() {
            return Err(anyhow::anyhow!(
                "edge_attrs artifact missing for {artifact_label} bundle: {}",
                p.display()
            ));
        }
        checksums.insert("edge_attrs_sha256".to_string(), json!(sha256_hex_path(p)?));
        Some(json!({"path": path, "format": "edge_attrs_json"}))
    } else {
        None
    };

    let mut root = serde_json::Map::new();
    root.insert("version".to_string(), json!(1));
    root.insert("kind".to_string(), json!(artifact_kind));
    root.insert("format".to_string(), json!(artifact_format));
    root.insert("reverse_edges".to_string(), Value::Object(reverse_edges));
    root.insert(
        "fkey_cols".to_string(),
        json!({"path": fkey_path.to_string_lossy(), "format": "fkey_cols_parquet"}),
    );
    root.insert("mask".to_string(), mask_value);
    let mut provided_fields = vec!["reverse_edges", "fkey_cols", "mask"];
    if let Some(v) = edge_attrs_value {
        root.insert("edge_attrs".to_string(), v);
        provided_fields.push("edge_attrs");
    }
    root.insert("checksums".to_string(), Value::Object(checksums));
    root.insert("provided_fields".to_string(), json!(provided_fields));
    Ok(Value::Object(root))
}

fn build_prediction_table_edges_bundle_value(
    output_dir: &Path,
    prediction_table_name: &str,
    mask_present: bool,
    mask_absent_reason: Option<&str>,
    edge_attrs_path: Option<&str>,
) -> anyhow::Result<Value> {
    build_table_edges_bundle_value(
        output_dir,
        prediction_table_name,
        mask_present,
        mask_absent_reason,
        edge_attrs_path,
        "tt_prediction_table_edges",
        "tt_pred_edge_bundle_v1",
        "prediction_table_edges",
    )
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(output_dir, prediction_table_name="custom_prediction_table", mask_present=true, mask_absent_reason=None, edge_attrs_path=None))]
pub fn build_prediction_table_edge_bundle_artifact(
    py: Python<'_>,
    output_dir: &str,
    prediction_table_name: &str,
    mask_present: bool,
    mask_absent_reason: Option<&str>,
    edge_attrs_path: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let out_dir = output_dir.to_owned();
    let prediction_table_name = prediction_table_name.to_owned();
    let edge_attrs_path = edge_attrs_path.map(|s| s.to_owned());
    let mask_absent_reason = mask_absent_reason.map(|s| s.to_owned());

    let payload = py
        .detach(|| -> anyhow::Result<Value> {
            build_prediction_table_edges_bundle_value(
                Path::new(&out_dir),
                &prediction_table_name,
                mask_present,
                mask_absent_reason.as_deref(),
                edge_attrs_path.as_deref(),
            )
        })
        .map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "build_prediction_table_edge_bundle_artifact failed: {e:#}"
            ))
        })?;

    let out_dict = PyDict::new(py);
    out_dict.set_item(
        "payload_json",
        serde_json::to_string(&payload)
            .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))?,
    )?;
    Ok(out_dict.into())
}

#[pyfunction]
#[tracing::instrument(level = "info", skip_all)]
#[pyo3(signature=(output_path, row_count, prediction_table_edges_json, prediction_table_col_stats_json=None, producer_summary_json=None))]
pub fn persist_prediction_artifact_bundle(
    py: Python<'_>,
    output_path: &str,
    row_count: u64,
    prediction_table_edges_json: &str,
    prediction_table_col_stats_json: Option<&str>,
    producer_summary_json: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let out = output_path.to_owned();
    let prediction_table_edges_payload = prediction_table_edges_json.to_string();
    let prediction_table_edges_value: Value =
        serde_json::from_str::<Value>(&prediction_table_edges_payload).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "invalid prediction_table_edges_json: {e}"
            ))
        })?;

    let kind_ok = prediction_table_edges_value
        .get("kind")
        .and_then(|v| v.as_str())
        == Some("tt_prediction_table_edges");
    if !kind_ok {
        return Err(crate::DiskgraphConfigError::new_err(
            "prediction_table_edges_json.kind must be tt_prediction_table_edges",
        ));
    }

    let format_ok = prediction_table_edges_value
        .get("format")
        .and_then(|v| v.as_str())
        == Some("tt_pred_edge_bundle_v1");
    if !format_ok {
        return Err(crate::DiskgraphConfigError::new_err(
            "prediction_table_edges_json.format must be tt_pred_edge_bundle_v1",
        ));
    }

    let prediction_col_stats_payload = prediction_table_col_stats_json.map(|s| s.to_string());
    let prediction_col_stats_value: Option<Value> = match prediction_col_stats_payload.as_ref() {
        Some(s) => Some(serde_json::from_str::<Value>(s).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "invalid prediction_table_col_stats_json: {e}"
            ))
        })?),
        None => None,
    };
    if let Some(v) = prediction_col_stats_value.as_ref() {
        let kind_ok =
            v.get("kind").and_then(|x| x.as_str()) == Some("tt_prediction_table_col_stats");
        if !kind_ok {
            return Err(crate::DiskgraphConfigError::new_err(
                "prediction_table_col_stats_json.kind must be tt_prediction_table_col_stats",
            ));
        }
        let format_ok = v.get("stats_format").and_then(|x| x.as_str()) == Some("kumo_col_stats_v1");
        if !format_ok {
            return Err(crate::DiskgraphConfigError::new_err(
                "prediction_table_col_stats_json.stats_format must be kumo_col_stats_v1",
            ));
        }
    }

    let producer_summary_value: Value = producer_summary_json
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or(Value::Null);

    let (manifest_path, artifacts_dir) = py
        .detach(|| -> anyhow::Result<(String, String)> {
            let artifacts_dir = format!("{}.artifacts", out);
            ensure_artifacts_dir(&artifacts_dir)?;
            let manifest_path = format!("{}.prediction.manifest.json", out);
            let edge_bundle_path = format!("{}/prediction_table_edges.bundle.json", artifacts_dir);
            let col_stats_path = format!("{}/stats.prediction_table_col_stats.json", artifacts_dir);

            write_artifact_file(&edge_bundle_path, prediction_table_edges_payload.as_bytes())?;
            if let Some(raw) = prediction_col_stats_payload.as_ref() {
                write_artifact_file(&col_stats_path, raw.as_bytes())?;
            }

            let mut checksums = serde_json::Map::new();
            checksums.insert(
                "prediction_table_sha256".to_string(),
                json!(sha256_hex_path(Path::new(&out))?),
            );
            checksums.insert(
                "prediction_table_edges_sha256".to_string(),
                json!(sha256_hex_bytes(prediction_table_edges_payload.as_bytes())),
            );
            merge_prefixed_checksums(
                &mut checksums,
                &prediction_table_edges_value,
                "prediction_table_edges",
            );
            if let Some(raw) = prediction_col_stats_payload.as_ref() {
                checksums.insert(
                    "prediction_table_col_stats_sha256".to_string(),
                    json!(sha256_hex_bytes(raw.as_bytes())),
                );
                if let Some(stats_value) = prediction_col_stats_value.as_ref() {
                    merge_prefixed_checksums(
                        &mut checksums,
                        stats_value,
                        "prediction_table_col_stats",
                    );
                }
            }

            let mut manifest = serde_json::Map::new();
            manifest.insert("version".to_string(), json!(1));
            manifest.insert("row_count".to_string(), json!(row_count));
            manifest.insert(
                "prediction_table".to_string(),
                artifact_ref(&out, "parquet_with_arange"),
            );
            manifest.insert(
                "prediction_table_edges".to_string(),
                prediction_table_edges_value.clone(),
            );
            if let Some(stats_value) = prediction_col_stats_value.as_ref() {
                manifest.insert(
                    "prediction_table_col_stats".to_string(),
                    stats_value.clone(),
                );
            }

            let mut prediction_artifacts = serde_json::Map::new();
            prediction_artifacts.insert(
                "prediction_table_edges".to_string(),
                artifact_ref(&edge_bundle_path, "tt_prediction_table_edges_json"),
            );
            if prediction_col_stats_payload.is_some() {
                prediction_artifacts.insert(
                    "prediction_table_col_stats".to_string(),
                    artifact_ref(&col_stats_path, "tt_prediction_table_col_stats_json"),
                );
            }
            manifest.insert(
                "prediction_artifacts".to_string(),
                Value::Object(prediction_artifacts),
            );
            manifest.insert("checksums".to_string(), Value::Object(checksums));
            if !producer_summary_value.is_null() {
                manifest.insert("producer_summary".to_string(), producer_summary_value);
            }

            let manifest = Value::Object(manifest);
            write_manifest_json(&manifest_path, &manifest)?;
            Ok((manifest_path, artifacts_dir))
        })
        .map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!(
                "persist_prediction_artifact_bundle failed: {e:#}"
            ))
        })?;

    let out_dict = PyDict::new(py);
    out_dict.set_item("manifest_path", manifest_path)?;
    out_dict.set_item("artifacts_dir", artifacts_dir)?;
    Ok(out_dict.into())
}

#[cfg(test)]
#[allow(deprecated)]
mod tests;
