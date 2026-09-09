//! PyO3 wrappers for the custom-TT artifact rehydration entry points.
//!
//! Provides `_build_train_table_artifacts_for_existing_table` (TRAIN) and
//! `_build_prediction_table_artifacts_for_existing_table` (PREDICT), the
//! Rust-side counterparts to kumo's
//! `kumo/engine/diskgraph_native_tt_artifact_rehydration.py`. Both build a
//! minimal `DiskGraphTTPlan` (empty stages) plus a `CustomTtSource` and dispatch
//! into `execute_tt_pass_with_source` via the shared
//! [`build_bundle_for_existing_table`] impl. The TRAIN entry consumes the
//! customer TARGET (`is_prediction=false`, split honored); the PREDICT entry
//! takes a no-target parquet (`is_prediction=true`, split forced off → a single
//! `predict` stage).
//!
//! GEN-ONLY (TT-NG durable bundle): this entry point runs the SAME shared gen
//! core as `execute_tt_gen` (`tt_executor::run_gen_bundle`), just source-driven
//! and into a KEPT bundle dir (`<base>.artifacts/`): the ttgen pass plus a
//! self-describing `tt_ingest_spec.json` beside `tt.parquet` + the per-stage seed
//! parquets, and STOPS. It does NOT ingest, does NOT publish a samplable
//! `train_tables/<tt_id>/` extension, and does NOT delete the bundle. The
//! separate, pod-independent `execute_tt_ingest` job (run later by kumo) consumes
//! this bundle to publish the extension — so a snapshot restore can always
//! re-ingest from the durable bundle.
//!
//! Returns a dict describing the bundle: `staging_dir`, `tt_parquet_path`,
//! `ingest_spec_path`, `row_count`, `output_path`, `artifacts_dir`,
//! `split_counts: dict[int,int]`, `split_meta: {family, authoritative, reason}`.
//!
//! Design invariants (do not relitigate):
//!
//! 1. **Never `PassThrough`**. PassThrough drops TARGET1; Node/Link + empty
//!    stages preserve customer-supplied targets.
//! 2. **Empty `stages`** — no plan compilation; every output column comes
//!    from the customer parquet directly.
//! 3. **`weight_col`** propagated as-is from the kumo caller.
//! 4. **`target_stype`** required and threaded through.
//!
//! The Python wrapper layer (`tt_artifacts.py`) marshals the SplitSpec
//! object and the kwargs into the JSON-shaped inputs this function takes.

use std::path::{Path, PathBuf};

use diskgraph_core::time::{Seconds, TimeUnit};
use diskgraph_core::tt_builder::{
    DiskGraphTTPlan, EntitySpec, PredictionType, RefResolveKind, RefResolveSpec,
    SplitSpec as PlanSplitSpec,
};
use diskgraph_train_table::ttgen_executor::CustomTtSource;
use diskgraph_train_table::route::EXCLUDE_TARGET_KEYS_COL;
use diskgraph_train_table::ttgen_output::{PRIMARY_ENTITY_COL, PRIMARY_TARGET_COL};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::tt_executor::{run_gen_bundle, validate_tt_id};

fn custom_tt_artifact_base(parquet_path: &Path) -> PathBuf {
    // Multi-part contract (kumo Phase 2): kumo passes the train-table
    // DIRECTORY (containing one or more `part-*.parquet`). The bundle root IS
    // the directory itself — sidecars (`.artifacts`/`.manifest.json`/
    // `.dg_extension`) hang off the dir.
    if parquet_path.is_dir() {
        return parquet_path.to_path_buf();
    }

    let file_name = parquet_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if file_name == "part-00000.parquet" {
        if let Some(parent) = parquet_path.parent() {
            if !parent.as_os_str().is_empty() {
                return parent.to_path_buf();
            }
        }
    }

    if let Some(stripped) = file_name.strip_suffix(".parquet") {
        if !stripped.is_empty() {
            return parquet_path.with_file_name(stripped);
        }
    }

    parquet_path.to_path_buf()
}

fn path_with_suffix(base: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{}", base.display(), suffix))
}

/// Build TT artifacts (manifest + sidecars) for an already-materialized
/// TRAINING-table parquet (consumes the customer TARGET).
///
/// Args mirror what kumo's
/// `ensure_diskgraph_native_artifacts_for_train_table` calls into. The
/// `split_spec_json` is a JSON-serialized `SplitSpec<Seconds>` shape (the
/// pure-Rust plan SplitSpec, NOT the Python SplitSpec dataclass — the
/// Python wrapper does the conversion).
///
/// `tt_id` is carried into the durable `tt_ingest_spec.json` so the separate
/// `execute_tt_ingest` job can publish `<index_path>/train_tables/<tt_id>/`
/// later. Kumo passes its `ttng_tt_id`. This gen-only entry does NOT publish an
/// extension itself.
///
/// Returns a dict describing the durable bundle: `staging_dir` (the kept bundle
/// dir), `tt_parquet_path`, `ingest_spec_path`, `row_count`, `output_path`,
/// `artifacts_dir`, `split_counts: dict[int,int]`,
/// `split_meta: {family, authoritative, reason}`.
///
/// Thin wrapper over [`build_bundle_for_existing_table`] with
/// `is_prediction=false` — behavior is byte-identical to the pre-refactor entry.
#[pyfunction]
#[pyo3(signature = (
    output_path,
    index_path,
    tt_id,
    entity_table_name,
    entity_pk_column,
    target_stype,
    input_timestamp_unit,
    split_spec_json = None,
    weight_col = None,
    target_entity_table = None,
    task_type = None,
    lp_rhs_is_temporal = false,
    allow_new_target_entities = false,
))]
pub fn build_train_table_artifacts_for_existing_table(
    py: Python<'_>,
    output_path: &str,
    index_path: &str,
    tt_id: &str,
    entity_table_name: &str,
    entity_pk_column: &str,
    target_stype: &str,
    input_timestamp_unit: &str,
    split_spec_json: Option<&str>,
    weight_col: Option<&str>,
    target_entity_table: Option<&str>,
    task_type: Option<&str>,
    lp_rhs_is_temporal: bool,
    allow_new_target_entities: bool,
) -> PyResult<Py<PyAny>> {
    build_bundle_for_existing_table(
        py,
        output_path,
        index_path,
        tt_id,
        entity_table_name,
        entity_pk_column,
        target_stype,
        input_timestamp_unit,
        split_spec_json,
        weight_col,
        target_entity_table,
        task_type,
        lp_rhs_is_temporal,
        allow_new_target_entities,
        /* is_prediction */ false,
    )
}

/// Build TT artifacts (manifest + sidecars) for an already-materialized
/// PREDICTION-table parquet (ENTITY + TIMESTAMP, NO target).
///
/// Mirrors [`build_train_table_artifacts_for_existing_table`] but runs the
/// gen pass with `is_prediction=true` and NO split → `SplitMode::None`, so the
/// bundle holds a single `predict` stage (`split_stages=None`, no target
/// consumed). The result-dict shape is identical to the train entry.
///
/// `tt_id` is the gen-side id carried into `tt_ingest_spec.json`. Predict fans
/// out: the separate `execute_tt_ingest` job may publish the bundle under a
/// per-BP-job TARGET id that differs from this gen `tt_id` (see the note on
/// `execute_tt_ingest`). This gen-only entry does NOT publish an extension.
///
/// Thin wrapper over [`build_bundle_for_existing_table`] with
/// `is_prediction=true` (and split forced off inside the shared impl).
#[pyfunction]
#[pyo3(signature = (
    output_path,
    index_path,
    tt_id,
    entity_table_name,
    entity_pk_column,
    target_stype,
    input_timestamp_unit,
    weight_col = None,
    target_entity_table = None,
    task_type = None,
))]
pub fn build_prediction_table_artifacts_for_existing_table(
    py: Python<'_>,
    output_path: &str,
    index_path: &str,
    tt_id: &str,
    entity_table_name: &str,
    entity_pk_column: &str,
    target_stype: &str,
    input_timestamp_unit: &str,
    weight_col: Option<&str>,
    target_entity_table: Option<&str>,
    task_type: Option<&str>,
) -> PyResult<Py<PyAny>> {
    build_bundle_for_existing_table(
        py,
        output_path,
        index_path,
        tt_id,
        entity_table_name,
        entity_pk_column,
        target_stype,
        input_timestamp_unit,
        // Predict has no split: force SplitMode::None → single `predict` stage.
        /* split_spec_json */
        None,
        weight_col,
        target_entity_table,
        task_type,
        // Predict has no split/grouping, so RHS temporality is irrelevant here.
        /* lp_rhs_is_temporal */
        false,
        /* allow_new_target_entities */
        false,
        /* is_prediction */ true,
    )
}

/// Build the plan's `fkey_mapping`: always the LHS `ENTITY1 → entity_table`
/// entry, plus a `TARGET1 → target_table` entry for LP TRAIN unless Kumo's
/// model plan explicitly allows targets created after the anchor.
///
/// A PREDICTION seed table has NO TARGET1 column (just LHS entities +
/// TIMESTAMP), so the TARGET1 entry is suppressed for predict — otherwise
/// ingest would error looking up a nonexistent `TARGET1` source column.
/// With `allow_new_target_entities`, the gen-time TARGET1 mapping is omitted so
/// the custom chunk processor does not apply anchor-time validity. The
/// independent `lp_resolution` mapping remains present and validates every PK
/// during ingest. The PREDICT case likewise suppresses TARGET1 because that
/// source column is absent; its RHS type is carried by the task-dataset spec.
fn build_fkey_mapping(
    entity_table_name: &str,
    target_entity_table: Option<&str>,
    is_prediction: bool,
    allow_new_target_entities: bool,
) -> Vec<diskgraph_core::tt_builder::TTFKeySpec> {
    let mut mapping = vec![diskgraph_core::tt_builder::TTFKeySpec {
        src_col: PRIMARY_ENTITY_COL.to_string(),
        dst_table: entity_table_name.to_string(),
    }];
    if let Some(target_table) = target_entity_table {
        if !is_prediction && !allow_new_target_entities {
            mapping.push(diskgraph_core::tt_builder::TTFKeySpec {
                src_col: PRIMARY_TARGET_COL.to_string(),
                dst_table: target_table.to_string(),
            });
        }
    }
    mapping
}

/// Build the plan's FK-free `lp_resolution` (`None` for non-LP). LP always
/// resolves the LHS `ENTITY1 → entity_node_id`; LP TRAIN additionally resolves
/// `TARGET1 → target_node_ids`. Predict suppresses the TARGET1 spec (no such
/// column in a prediction seed table) — see [`build_fkey_mapping`].
///
/// `per_link_excludes` mirrors the native PQL path's third spec: a per-link
/// split appends `exclude_target_keys` (prior-stage RHS keys on val/test rows)
/// and the engine hydrates the past-visit exclude set from `exclude_node_ids`.
/// Without the spec the column is never resolved, the engine's lookup misses,
/// and evaluation silently ranks against already-visited targets. It must track
/// `per_link_output_schema` exactly: a declared-but-absent source column is a
/// hard ingest error, not a skip.
fn build_lp_resolution(
    entity_table_name: &str,
    target_entity_table: Option<&str>,
    is_prediction: bool,
    per_link_excludes: bool,
) -> Option<Vec<RefResolveSpec>> {
    target_entity_table.map(|target_table| {
        let mut specs = vec![RefResolveSpec {
            src_col: PRIMARY_ENTITY_COL.to_string(),
            dst_table: entity_table_name.to_string(),
            kind: RefResolveKind::Scalar,
            node_id_col: "entity_node_id".to_string(),
        }];
        if !is_prediction {
            specs.push(RefResolveSpec {
                src_col: PRIMARY_TARGET_COL.to_string(),
                dst_table: target_table.to_string(),
                kind: RefResolveKind::List,
                node_id_col: "target_node_ids".to_string(),
            });
            if per_link_excludes {
                specs.push(RefResolveSpec {
                    src_col: EXCLUDE_TARGET_KEYS_COL.to_string(),
                    dst_table: target_table.to_string(),
                    kind: RefResolveKind::List,
                    node_id_col: "exclude_node_ids".to_string(),
                });
            }
        }
        specs
    })
}

/// Gen appends `exclude_target_keys` only on the fused per-link pass-B path.
fn emits_per_link_excludes(split_kind: Option<&str>) -> bool {
    matches!(split_kind, Some("RandomLinkSplit") | Some("CustomLinkSplit"))
}

/// Shared gen-only bundle builder for both the TRAIN and PREDICT custom-TT
/// entries. `is_prediction` toggles the plan flag (predict skips the customer
/// TARGET); the PREDICT wrapper additionally passes `split_spec_json=None` so
/// the executor takes the `SplitMode::None` single-`predict`-stage path.
fn build_bundle_for_existing_table(
    py: Python<'_>,
    output_path: &str,
    index_path: &str,
    tt_id: &str,
    entity_table_name: &str,
    entity_pk_column: &str,
    target_stype: &str,
    input_timestamp_unit: &str,
    split_spec_json: Option<&str>,
    weight_col: Option<&str>,
    target_entity_table: Option<&str>,
    task_type: Option<&str>,
    lp_rhs_is_temporal: bool,
    allow_new_target_entities: bool,
    is_prediction: bool,
) -> PyResult<Py<PyAny>> {
    validate_tt_id(tt_id)?;
    // The bridge MUST declare the unit for any Int64 TIMESTAMP column in
    // the customer parquet — DG's typed Arrow path self-describes, but
    // the kumo flow currently writes int64 ns (Spark `LongType`). No
    // default: callers pass a `TimeUnit` name string ("nanoseconds",
    // "microseconds", ...) parsed by `TimeUnit::from_name`.
    let parsed_unit: TimeUnit = TimeUnit::from_name(input_timestamp_unit).ok_or_else(|| {
        crate::DiskgraphConfigError::new_err(format!(
            "build_train_table_artifacts_for_existing_table: \
             input_timestamp_unit={input_timestamp_unit:?} is not a valid \
             TimeUnit name (expected one of: nanoseconds, microseconds, \
             milliseconds, seconds)"
        ))
    })?;
    // Parse the optional split spec JSON. When None, the plan has no split and
    // the executor takes the SplitMode::None path. A custom split (e.g.
    // `ManualSplit` on a customer-provided split column) flows through here as a
    // SplitSpec and routes via the column deciders — there is no split.bin.
    let split: Option<PlanSplitSpec<Seconds>> = match split_spec_json {
        Some(s) if !s.is_empty() => Some(serde_json::from_str(s).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!("invalid split_spec_json: {e}"))
        })?),
        _ => None,
    };

    let split_kind_for_meta = split.as_ref().and_then(|s| s.kind.clone());

    let is_link_prediction = target_entity_table.is_some();
    let is_forecasting = task_type
        .map(|value| value.eq_ignore_ascii_case("forecasting"))
        .unwrap_or(false);

    // Construct a minimal plan. Empty stages: every output column
    // (ENTITYk, TIMESTAMP, TARGETk) is read directly from the customer
    // parquet by the chunk processor. LP CTT uses LinkArray output plus
    // the same FK-free resolution contract as PQL TT-NG LP.
    let plan: DiskGraphTTPlan<Seconds> = DiskGraphTTPlan {
        entity: EntitySpec {
            table: entity_table_name.to_string(),
            column: entity_pk_column.to_string(),
            output_time_col_name: None,
            default_time_value: None,
            time_col: None,
            valid_to_col: None,
            filter: None,
        },
        temporal: None,
        prediction_type: if is_link_prediction {
            PredictionType::Link {
                include_frequency: true,
            }
        } else if is_forecasting {
            PredictionType::Forecasting
        } else {
            PredictionType::Node
        },
        is_prediction,
        split,
        stages: vec![],
        target_stype: Some(target_stype.to_string()),
        weight_col: weight_col.map(String::from),
        fkey_mapping: Some(build_fkey_mapping(
            entity_table_name,
            target_entity_table,
            is_prediction,
            allow_new_target_entities,
        )),
        lp_resolution: build_lp_resolution(
            entity_table_name,
            target_entity_table,
            is_prediction,
            emits_per_link_excludes(split_kind_for_meta.as_deref()),
        ),
        // Same grouping gate as the native PQL path: Kumo computes RHS
        // temporality from graph metadata and passes it through this bridge.
        lp_rhs_is_temporal,
    };

    // Output dirs derive from the train-table bundle root, not a physical
    // parquet part. kumo passes the train-table DIRECTORY (one or more
    // `part-*.parquet`) → the bundle root IS that dir. Legacy single-file
    // callers (`<custom_train_table>/part-00000.parquet` or
    // `<name>.parquet`) collapse to the parent / file stem. The
    // `custom_source.parquet_path` (dir or file) flows into Pass A, which
    // enumerates all parts via `enumerate_custom_tt_parts`.
    let parquet_path = Path::new(output_path);
    let artifact_base = custom_tt_artifact_base(parquet_path);
    let artifacts_dir_path = path_with_suffix(&artifact_base, ".artifacts")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&artifacts_dir_path).map_err(|e| {
        crate::DiskgraphSystemError::new_err(format!(
            "failed to create artifacts dir {artifacts_dir_path}: {e}"
        ))
    })?;

    let custom_source = CustomTtSource {
        parquet_path: parquet_path.to_path_buf(),
        input_timestamp_unit: parsed_unit,
    };

    let idx = index_path.to_owned();
    let out_dir = artifacts_dir_path.clone();
    let tt_id_owned = tt_id.to_owned();

    // GEN-ONLY: run the shared gen core (`run_gen_bundle`, the same one
    // `execute_tt_gen` uses) with the custom source, into the KEPT bundle
    // dir. Source-mode gen reads every output column directly from the
    // customer parquet (empty plan stages) and writes per-stage seed parquets
    // (split) or a single `tt.parquet` (no split / predict), stats sidecars,
    // and `weight.bin` into the bundle dir — the `weight.bin` travels WITH
    // the bundle so re-ingest can carry it. No ingest, no extension publish,
    // no bundle delete — the separate `execute_tt_ingest` job consumes this
    // durable bundle later.
    let (tt_parquet_path, ingest_spec_path, pass_output_rows, split_counts_vec): (
        String,
        String,
        u64,
        Vec<(String, u64)>,
    ) = py
        .detach(|| -> diskgraph_train_table::errors::TTGenResult<_> {
            let index_root = Path::new(&idx);
            let out_dir_path = Path::new(&out_dir);

            let (pass_output, spec_path) = run_gen_bundle(
                &plan,
                index_root,
                out_dir_path,
                &tt_id_owned,
                Some(custom_source),
            )?;

            // split_counts source: gen's per-stage seed infos (split path).
            let split_counts_vec: Vec<(String, u64)> = pass_output
                .stage_gen_infos
                .as_ref()
                .map(|infos| infos.iter().map(|i| (i.key.clone(), i.row_count)).collect())
                .unwrap_or_default();
            Ok((
                pass_output.tt_parquet_path.to_string_lossy().into_owned(),
                spec_path.to_string_lossy().into_owned(),
                pass_output.rows_written,
                split_counts_vec,
            ))
        })
        .map_err(crate::map_ttgen_err)?;

    let has_stage_gen_infos = !split_counts_vec.is_empty();

    // ── Result dict: describes the durable gen bundle for the kumo caller ──
    let result = PyDict::new(py);
    // `staging_dir` == the kept bundle dir the separate `execute_tt_ingest`
    // consumes; kumo reads this to persist + re-ingest.
    result.set_item("staging_dir", &out_dir)?;
    result.set_item("tt_parquet_path", &tt_parquet_path)?;
    result.set_item("ingest_spec_path", &ingest_spec_path)?;
    result.set_item("row_count", pass_output_rows)?;
    result.set_item("output_path", output_path)?;
    result.set_item("artifacts_dir", &artifacts_dir_path)?;

    // split_counts: dict[int,int] mapping stage byte -> count. Derived from the
    // gen-produced per-stage seed parquets (0=train, 1=val, 2=test).
    let split_counts = PyDict::new(py);
    for (key, count) in &split_counts_vec {
        let byte = match key.as_str() {
            "train" => 0i64,
            "val" => 1,
            "test" => 2,
            _ => 255,
        };
        split_counts.set_item(byte, *count as i64)?;
    }
    result.set_item("split_counts", split_counts)?;

    // split_meta: {family, authoritative, reason}. Family comes from the
    // SplitSpec.kind. Authoritative=true for compute-from-scratch; reason
    // describes the path.
    let split_meta = PyDict::new(py);
    split_meta.set_item("family", split_kind_for_meta.as_deref())?;
    split_meta.set_item("authoritative", has_stage_gen_infos)?;
    let reason = if has_stage_gen_infos {
        "diskgraph_executor_compute"
    } else {
        "no_split"
    };
    split_meta.set_item("reason", reason)?;
    result.set_item("split_meta", split_meta)?;

    Ok(result.into())
}

#[cfg(test)]
mod tests {
    use super::{
        build_fkey_mapping, build_lp_resolution, custom_tt_artifact_base,
        emits_per_link_excludes, path_with_suffix,
    };
    use diskgraph_core::tt_builder::RefResolveKind;
    use diskgraph_train_table::route::EXCLUDE_TARGET_KEYS_COL;
    use diskgraph_train_table::ttgen_output::{PRIMARY_ENTITY_COL, PRIMARY_TARGET_COL};
    use std::path::{Path, PathBuf};

    #[test]
    fn lp_train_plan_resolves_entity_and_target() {
        // TRAIN LP: both ENTITY1 (scalar) and TARGET1 (list) resolve, and both
        // appear in fkey_mapping unless new targets are explicitly allowed.
        let fkeys = build_fkey_mapping(
            "users",
            Some("products"),
            /* is_prediction */ false,
            /* allow_new_target_entities */ false,
        );
        assert_eq!(fkeys.len(), 2);
        assert_eq!(fkeys[0].src_col, PRIMARY_ENTITY_COL);
        assert_eq!(fkeys[1].src_col, PRIMARY_TARGET_COL);
        assert_eq!(fkeys[1].dst_table, "products");

        let res = build_lp_resolution("users", Some("products"), false, false)
            .expect("LP plan must have resolution");
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].src_col, PRIMARY_ENTITY_COL);
        assert_eq!(res[0].kind, RefResolveKind::Scalar);
        assert_eq!(res[1].src_col, PRIMARY_TARGET_COL);
        assert_eq!(res[1].kind, RefResolveKind::List);
        assert_eq!(res[1].dst_table, "products");

        let allowed_fkeys = build_fkey_mapping(
            "users",
            Some("products"),
            /* is_prediction */ false,
            /* allow_new_target_entities */ true,
        );
        assert_eq!(allowed_fkeys.len(), 1);
        assert_eq!(allowed_fkeys[0].src_col, PRIMARY_ENTITY_COL);
    }

    #[test]
    fn lp_predict_plan_resolves_entity_only_no_target() {
        // PREDICT LP: a prediction seed table has NO TARGET1 column, so neither
        // fkey_mapping nor lp_resolution may reference TARGET1 (ingest would
        // otherwise fail resolving a nonexistent column). Only ENTITY1 remains.
        let fkeys = build_fkey_mapping(
            "users",
            Some("products"),
            /* is_prediction */ true,
            /* allow_new_target_entities */ false,
        );
        assert_eq!(fkeys.len(), 1);
        assert_eq!(fkeys[0].src_col, PRIMARY_ENTITY_COL);
        assert!(fkeys.iter().all(|f| f.src_col != PRIMARY_TARGET_COL));

        let res = build_lp_resolution("users", Some("products"), true, true)
            .expect("LP plan must still resolve the entity");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].src_col, PRIMARY_ENTITY_COL);
        assert_eq!(res[0].kind, RefResolveKind::Scalar);
        assert!(res.iter().all(|r| r.src_col != PRIMARY_TARGET_COL));
    }

    #[test]
    fn per_link_lp_train_plan_resolves_past_visit_excludes() {
        // A per-link split appends `exclude_target_keys` at gen; without this
        // third spec the column is never resolved into `exclude_node_ids`, the
        // engine's lookup misses, and eval ranks against already-visited
        // targets -- silently, since nothing on that path warns.
        let res = build_lp_resolution("users", Some("products"), false, true)
            .expect("LP plan must have resolution");
        assert_eq!(res.len(), 3);
        assert_eq!(res[2].src_col, EXCLUDE_TARGET_KEYS_COL);
        assert_eq!(res[2].node_id_col, "exclude_node_ids");
        assert_eq!(res[2].kind, RefResolveKind::List);
        // Resolved against the SAME RHS table as the positives, or the ids
        // would land in a different node space than the score matrix.
        assert_eq!(res[2].dst_table, res[1].dst_table);
    }

    #[test]
    fn exclude_spec_tracks_the_split_kinds_that_emit_the_column() {
        // A declared-but-absent src_col is a hard ingest error, so this gate
        // must match `per_link_output_schema` exactly.
        assert!(emits_per_link_excludes(Some("RandomLinkSplit")));
        assert!(emits_per_link_excludes(Some("CustomLinkSplit")));
        for kind in [None, Some("RandomSplit"), Some("TimeRangeSplit"), Some("ManualSplit")] {
            assert!(!emits_per_link_excludes(kind), "{kind:?} does not emit the column");
        }
    }

    #[test]
    fn non_lp_plan_has_no_resolution() {
        // Node/forecasting (no target_entity_table): entity-only fkey, no
        // lp_resolution, for both train and predict.
        for is_prediction in [false, true] {
            let fkeys = build_fkey_mapping("users", None, is_prediction, false);
            assert_eq!(fkeys.len(), 1);
            assert_eq!(fkeys[0].src_col, PRIMARY_ENTITY_COL);
            assert!(build_lp_resolution("users", None, is_prediction, true).is_none());
        }
    }

    #[test]
    fn custom_tt_artifacts_use_bundle_root_for_part_file() {
        let parquet = Path::new("/tmp/job/custom_train_table/part-00000.parquet");

        let base = custom_tt_artifact_base(parquet);

        assert_eq!(base, PathBuf::from("/tmp/job/custom_train_table"));
        assert_eq!(
            path_with_suffix(&base, ".artifacts"),
            PathBuf::from("/tmp/job/custom_train_table.artifacts")
        );
        assert_eq!(
            path_with_suffix(&base, ".manifest.json"),
            PathBuf::from("/tmp/job/custom_train_table.manifest.json")
        );
    }

    #[test]
    fn custom_tt_artifacts_strip_parquet_extension_for_single_file() {
        let parquet = Path::new("/tmp/job/truncated_tt_ttng.parquet");

        let base = custom_tt_artifact_base(parquet);

        assert_eq!(base, PathBuf::from("/tmp/job/truncated_tt_ttng"));
        assert_eq!(
            path_with_suffix(&base, ".manifest.json"),
            PathBuf::from("/tmp/job/truncated_tt_ttng.manifest.json")
        );
    }

    #[test]
    fn custom_tt_artifacts_use_dir_itself_as_bundle_root() {
        // Multi-part contract: kumo passes the train-table DIRECTORY. The
        // bundle root is the dir itself; sidecars hang off it.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("custom_train_table");
        std::fs::create_dir_all(&dir).unwrap();
        // Touch two parts so the path is unambiguously a multi-part dir.
        std::fs::write(dir.join("part-00000.parquet"), b"").unwrap();
        std::fs::write(dir.join("part-00001.parquet"), b"").unwrap();

        let base = custom_tt_artifact_base(&dir);

        assert_eq!(base, dir);
        assert_eq!(
            path_with_suffix(&base, ".artifacts"),
            PathBuf::from(format!("{}.artifacts", dir.display()))
        );
        assert_eq!(
            path_with_suffix(&base, ".manifest.json"),
            PathBuf::from(format!("{}.manifest.json", dir.display()))
        );
    }

    #[test]
    fn bare_part_file_falls_back_to_file_stem() {
        let parquet = Path::new("part-00000.parquet");

        let base = custom_tt_artifact_base(parquet);

        assert_eq!(base, PathBuf::from("part-00000"));
    }
}
