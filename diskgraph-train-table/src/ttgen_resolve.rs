//! Plan preprocessing: resolve string references to index handles and indices.
//!
//! Runs once on the main thread before the parallel entity loop.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::errors::{TTGenError, TTGenResult};
use arrow_schema::{DataType, Field};
use diskgraph_core::sortkey::SortKeyEncoder;
use diskgraph_core::time::{Abs, Dur, UnitSystem};
use diskgraph_core::tt_builder::*;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_index::layout::RelationLayout;
use diskgraph_rowpack::{IngestedType, RowPackContext};

// ─── Resolved plan types ─────────────────────────────────────────────────────

/// A fully resolved plan ready for parallel execution.
pub struct ResolvedPlan<U: UnitSystem> {
    // Entity
    pub entity_table_spec: EntitySpec<U>,
    pub entity_table: Arc<TableReader>,
    pub entity_time_col_idx: Option<usize>,
    pub entity_valid_to_col_idx: Option<usize>,
    pub entity_filter: Option<ResolvedRowFilter>,
    /// Column indices into entity rowpack for columns referenced as EntityColumn
    /// in any filter RHS. Maps column name → rowpack column index.
    pub entity_col_indices: Vec<(String, usize)>,

    // Temporal
    pub timestamps: Vec<Abs<U>>,
    pub has_temporal: bool,
    pub lag_timesteps: i64,

    // Stages
    pub stages: Vec<ResolvedStage<U>>,

    // Aliases
    pub num_aliases: usize,
    pub alias_to_idx: HashMap<String, usize>,
    pub target_alias_idx: usize,
    pub autoregression_alias_idx: Option<usize>,
    pub yoy_alias_idx: Option<usize>,

    // Entity PK
    pub entity_pk_col_idx: usize,
    pub entity_dtype: DataType,

    // Prediction type
    pub prediction_type: PredictionType,

    // Target dtype: Bool for Compare, else inferred from ingested column type
    pub target_dtype: DataType,
    pub autoregression_dtype: Option<DataType>,
    pub yoy_dtype: Option<DataType>,

    /// Semantic type of the target column ("numerical", "categorical", etc.).
    pub target_stype: String,
    pub split_projection: Option<ResolvedSplitProjection>,
}

#[derive(Clone)]
struct SplitProjectionRequest {
    kind: SplitProjectionKind,
    source_table: String,
    source_column: String,
    output_name: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SplitProjectionKind {
    ScalarEntity,
    LinkArray,
}

pub struct ResolvedStage<U: UnitSystem> {
    pub ops: Vec<ResolvedOp<U>>,
    pub filter: Option<ResolvedStageFilter>,
}

pub enum ResolvedOp<U: UnitSystem> {
    Aggregate(ResolvedAggregate<U>),
    Column(ResolvedColumn),
    Compare(ResolvedCompare),
    LogicalOp(ResolvedLogicalOp),
    Autoregression(ResolvedAutoregression<U>),
    HasEdges(ResolvedHasEdges<U>),
}

pub struct ResolvedHasEdges<U: UnitSystem> {
    pub alias_idx: usize,
    pub path: Vec<ResolvedEdge>,
    pub window: Option<ResolvedWindow<U>>,
    pub row_filter: Option<ResolvedRowFilter>,
}

pub struct ResolvedAutoregression<U: UnitSystem> {
    pub alias_idx: usize,
    pub target_op: Box<ResolvedOp<U>>,
    pub lag_offsets: Vec<Dur<U>>,
}

impl<U: UnitSystem> ResolvedOp<U> {
    pub fn alias_idx(&self) -> usize {
        match self {
            ResolvedOp::Aggregate(a) => a.alias_idx,
            ResolvedOp::HasEdges(a) => a.alias_idx,
            ResolvedOp::Column(c) => c.alias_idx,
            ResolvedOp::Compare(c) => c.alias_idx,
            ResolvedOp::LogicalOp(l) => l.alias_idx,
            ResolvedOp::Autoregression(a) => a.alias_idx,
        }
    }
}

pub struct ResolvedAggregate<U: UnitSystem> {
    pub alias_idx: usize,
    pub path: Vec<ResolvedEdge>,
    pub func: AggFunc,
    pub target_col_idx: Option<usize>,
    pub window: Option<ResolvedWindow<U>>,
    pub row_filter: Option<ResolvedRowFilter>,
    pub limit: Option<u32>,
    /// For LP ListDistinct: the table whose PK the target column references.
    /// Used to filter out orphaned FK values that don't exist in the
    /// destination table (e.g., article_ids in transactions that have no
    /// matching row in the articles table).
    pub target_pk_table: Option<Arc<TableReader>>,
    /// Pre-resolved rowpack column indices for the target entity table's
    /// `create_time_col` and `end_time_col`. Used for temporal validity
    /// filtering of LP targets (e.g., exclude a FIDO whose ADDED_DATE is
    /// after the anchor timestamp).
    pub target_temporal_create_col_idx: Option<usize>,
    pub target_temporal_end_col_idx: Option<usize>,
    pub link_split_col_idx: Option<usize>,
    /// Source-column semantic type. Used at the per-edge consume callback to
    /// dispatch on `multicategorical` for per-token observe; all other values
    /// take the legacy whole-cell path.
    ///
    /// Post-#28478: ingest's `normalize_multicat_pseudo_list` flips delimited
    /// multicat scalars to `IngestedType::StringArray` at convert time. The
    /// executor never sees a multicat-with-delimiter scalar source; no
    /// delimiter field is plumbed through resolution.
    pub target_semantic_type: Option<String>,
}

pub struct ResolvedColumn {
    pub alias_idx: usize,
    pub table: Arc<TableReader>,
    pub col_idx: usize,
    /// See `ResolvedAggregate::target_semantic_type`. Drives dispatch at the
    /// `ResolvedOp::Column` direct-target read site (Route A).
    pub target_semantic_type: Option<String>,
}

pub struct ResolvedSplitProjection {
    pub output_name: String,
    pub entity_col_idx: Option<usize>,
    pub dtype: DataType,
}

pub struct ResolvedCompare {
    pub alias_idx: usize,
    pub lhs_alias_idx: usize,
    pub op: CompareOp,
    pub rhs: Value,
}

pub struct ResolvedLogicalOp {
    pub alias_idx: usize,
    pub op: LogicalOpKind,
    pub operand_indices: Vec<usize>,
}

/// A resolved edge hop: relation index + sort key encoder + destination table.
pub struct ResolvedEdge {
    pub src_table: Arc<TableReader>,
    pub relation_idx: usize,
    pub layout: RelationLayout,
    pub encoder: Option<SortKeyEncoder>,
    pub dst_table: Arc<TableReader>,
}

/// Window bounds for temporal aggregation.
///
/// `Fixed` is used for units with a constant duration (seconds through weeks).
/// `Calendar` is used for months/years which require calendar-aware arithmetic
/// to match Kumo's `pd.DateOffset` behavior (e.g., 3 months from Jul 1 = Oct 1,
/// not Sep 29).
pub enum ResolvedWindow<U: UnitSystem> {
    Fixed {
        offset_lo: Dur<U>,
        offset_hi: Dur<U>,
        encoder: SortKeyEncoder,
    },
    Calendar {
        lo_months: i32,
        hi_months: i32,
        encoder: SortKeyEncoder,
    },
}

// ─── Resolved filter types ───────────────────────────────────────────────────

pub enum ResolvedRowFilter {
    Compare {
        lhs: ResolvedRowFilterLhs,
        op: CompareOp,
        rhs: ResolvedRowFilterRhs,
    },
    And(Vec<ResolvedRowFilter>),
    Or(Vec<ResolvedRowFilter>),
    Not(Box<ResolvedRowFilter>),
}

pub enum ResolvedRowFilterLhs {
    /// Column index on the current destination rowpack.
    Column(usize),
    /// Sub-traversal path + column index on final destination.
    Remote {
        path: Vec<ResolvedEdge>,
        col_idx: usize,
    },
}

pub enum ResolvedRowFilterRhs {
    Constant(Value),
    /// Index into the pre-resolved entity columns array.
    EntityColumn(usize),
    Remote {
        path: Vec<ResolvedEdge>,
        col_idx: usize,
    },
}

pub enum ResolvedStageFilter {
    Compare {
        alias_idx: usize,
        op: CompareOp,
        rhs: ResolvedStageFilterRhs,
    },
    And(Vec<ResolvedStageFilter>),
    Or(Vec<ResolvedStageFilter>),
    Not(Box<ResolvedStageFilter>),
}

pub enum ResolvedStageFilterRhs {
    Constant(Value),
    EntityColumn(usize),
    Remote {
        path: Vec<ResolvedEdge>,
        col_idx: usize,
    },
    Alias(usize),
}

// ─── Resolution logic ────────────────────────────────────────────────────────

/// Preprocess a plan: resolve all string references to index handles.
pub fn resolve_plan<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
    reader: &GraphIndexReader,
) -> TTGenResult<ResolvedPlan<U>> {
    // 1. Resolve entity table
    let entity_table = reader
        .table(&plan.entity.table)
        .ok_or_else(|| TTGenError::SchemaError(format!("entity table '{}' not found in index", plan.entity.table)))?;

    let entity_ctx = entity_table
        .node_data_decoder()
        .map(|d| d.context().clone());

    // 2. Resolve entity column indices
    // Fall back to index metadata create_time_col when the plan doesn't specify
    // a time_col. This ensures temporal queries filter out (entity, timestamp)
    // pairs where timestamp < entity.create_time even when the compiler omits it.
    let effective_time_col: Option<&str> = plan
        .entity
        .time_col
        .as_deref()
        .or_else(|| entity_table.create_time_col());

    let entity_time_col_idx = effective_time_col
        .map(|col| find_col_idx_in_context(&entity_ctx, col))
        .transpose().map_err(|e| TTGenError::SchemaError(format!("resolving entity time_col: {}", e)))?;

    // Same fallback for valid_to_col: use index metadata end_time_col.
    let effective_valid_to_col: Option<&str> = plan
        .entity
        .valid_to_col
        .as_deref()
        .or_else(|| entity_table.end_time_col());

    let entity_valid_to_col_idx = effective_valid_to_col
        .map(|col| find_col_idx_in_context(&entity_ctx, col))
        .transpose().map_err(|e| TTGenError::SchemaError(format!("resolving entity valid_to_col: {}", e)))?;

    // 3. Resolve entity PK column index and Arrow type
    let entity_pk_col_idx = find_col_idx_in_context(&entity_ctx, &plan.entity.column).map_err(|e| TTGenError::SchemaError(format!("resolving entity PK column: {}", e)))?;
    let entity_dtype = {
        let ctx = entity_ctx
            .as_ref()
            .expect("entity_ctx must be Some if find_col_idx_in_context succeeded");
        ingested_type_to_arrow(&ctx.columns[entity_pk_col_idx].logical_type)
    };

    // 4. Assign alias indices
    let mut alias_names = Vec::new();
    for stage in &plan.stages {
        for op in &stage.ops {
            match op {
                Op::Aggregate { alias, .. } |
                Op::HasEdges { alias, .. } |
                Op::Column { alias, .. } |
                Op::Compare { alias, .. } |
                Op::LogicalOp { alias, .. } => {
                    alias_names.push(alias.clone());
                }
                Op::Autoregression { alias, target_op, .. } => {
                    // Extract the inner op's alias first so it gets a valid index
                    let inner_alias = match &**target_op {
                        Op::Aggregate { alias: inner, .. } => inner,
                        _ => panic!("Autoregression target must be an Aggregate"),
                    };
                    alias_names.push(inner_alias.clone());
                    alias_names.push(alias.clone());
                }
            };
        }
    }
    let alias_to_idx: HashMap<String, usize> = alias_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();
    let num_aliases = alias_names.len();

    // 5. Collect all entity column references for pre-resolution
    let entity_col_indices = collect_entity_col_refs(plan, &entity_ctx)?;
    let split_request = split_projection_request(plan)?;

    // 6. Resolve stages
    let mut resolved_stages = Vec::new();
    for stage in &plan.stages {
        let resolved = resolve_stage(
            stage,
            reader,
            &entity_table,
            &entity_ctx,
            &alias_to_idx,
            &entity_col_indices,
            split_request.as_ref(),
        )?;
        resolved_stages.push(resolved);
    }
    validate_link_split_projection_attached(split_request.as_ref(), &resolved_stages)?;

    // 7. Determine target alias and dtype
    let target_alias_idx = if num_aliases > 0 {
        // The last op's alias is typically the target
        num_aliases - 1
    } else {
        if matches!(plan.prediction_type, PredictionType::PassThrough) {
            0 // Dummy value, safely ignored by executor
        } else {
            return Err(TTGenError::InvalidPlan("plan has no ops".to_string()));
        }
    };

    // Infer target dtype from the last op:
    // - Compare → Boolean
    // - LP aggregate (ListDistinct) → dtype of the target column (FK column type)
    // - Other aggregates → Float64
    let target_dtype = infer_target_dtype(plan, &resolved_stages);

    // 7.5 Resolve forecasting-specific aliases
    let mut autoregression_alias_idx = None;
    let mut autoregression_dtype = None;
    let mut yoy_alias_idx = None;
    let mut yoy_dtype = None;

    if matches!(plan.prediction_type, PredictionType::Forecasting) {
        if let Some(idx) = alias_to_idx.get("AUTOREGRESSION_COL") {
            autoregression_alias_idx = Some(*idx);
            autoregression_dtype = Some(infer_op_dtype(plan.stages.iter().flat_map(|s| &s.ops).find(|op| match op {
                Op::Autoregression { alias, .. } => alias == "AUTOREGRESSION_COL",
                _ => false,
            }).unwrap(), &resolved_stages));
        }
        if let Some(idx) = alias_to_idx.get("YOY_COL") {
            yoy_alias_idx = Some(*idx);
            yoy_dtype = Some(infer_op_dtype(plan.stages.iter().flat_map(|s| &s.ops).find(|op| match op {
                Op::Autoregression { alias, .. } => alias == "YOY_COL",
                _ => false,
            }).unwrap(), &resolved_stages));
        }
    }

    // 8. Timestamps – the executor uses partition_point (binary search)
    //    which requires ascending order. Sort defensively since callers
    //    may pass timestamps in arbitrary order.
    let mut timestamps = plan
        .temporal
        .as_ref()
        .map(|t| t.timestamps.clone())
        .unwrap_or_default();
    timestamps.sort_unstable_by_key(|t| t.raw());

    // 9. Entity filter
    let entity_filter = plan
        .entity
        .filter
        .as_ref()
        .map(|f| resolve_row_filter(f, reader, &entity_table, &entity_ctx, &entity_col_indices))
        .transpose().map_err(|e| TTGenError::SchemaError(format!("resolving entity filter: {}", e)))?;

    let lag_timesteps = plan
        .temporal
        .as_ref()
        .and_then(|t| t.lag_timesteps)
        .unwrap_or(0);
    let split_projection = resolve_split_projection(plan, &entity_ctx, split_request.as_ref())?;

    Ok(ResolvedPlan {
        entity_table_spec: plan.entity.clone(),
        entity_table,
        entity_time_col_idx,
        entity_valid_to_col_idx,
        entity_filter,
        entity_col_indices,
        timestamps,
        has_temporal: plan.temporal.is_some(),
        lag_timesteps,
        stages: resolved_stages,
        num_aliases,
        alias_to_idx,
        target_alias_idx,
        autoregression_alias_idx,
        yoy_alias_idx,
        entity_pk_col_idx,
        entity_dtype,
        prediction_type: plan.prediction_type.clone(),
        target_dtype,
        autoregression_dtype,
        yoy_dtype,
        target_stype: plan.target_stype.clone().ok_or_else(|| {
            TTGenError::InvalidPlan("target_stype is required".to_string())
        })?,
        split_projection,
    })
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn split_projection_request<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
) -> TTGenResult<Option<SplitProjectionRequest>> {
    let Some(split) = &plan.split else {
        return Ok(None);
    };
    let (kind, split_kind) = match split.kind.as_deref() {
        Some("ManualSplit") => (SplitProjectionKind::ScalarEntity, "ManualSplit"),
        Some("CustomSplit") => (SplitProjectionKind::ScalarEntity, "CustomSplit"),
        Some("CustomLinkSplit") => (SplitProjectionKind::LinkArray, "CustomLinkSplit"),
        _ => return Ok(None),
    };
    let split_column = split
        .split_column
        .as_deref()
        .ok_or_else(|| TTGenError::InvalidPlan(format!("{split_kind} requires split_column")))?;
    let (table_name, column_name) = split_column_table_and_column(split_column, &plan.entity.table);
    Ok(Some(SplitProjectionRequest {
        kind,
        source_table: table_name,
        source_column: column_name,
        output_name: split
            .split_output_column
            .clone()
            .unwrap_or_else(|| split_column.to_string()),
    }))
}

fn resolve_split_projection<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
    entity_ctx: &Option<RowPackContext>,
    request: Option<&SplitProjectionRequest>,
) -> TTGenResult<Option<ResolvedSplitProjection>> {
    let Some(request) = request else {
        return Ok(None);
    };
    if request.kind == SplitProjectionKind::LinkArray {
        return Ok(Some(ResolvedSplitProjection {
            output_name: request.output_name.clone(),
            entity_col_idx: None,
            dtype: DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        }));
    }

    let split_kind = plan
        .split
        .as_ref()
        .and_then(|s| s.kind.as_deref())
        .unwrap_or("ManualSplit");
    let split_column = plan
        .split
        .as_ref()
        .and_then(|s| s.split_column.as_deref())
        .unwrap_or(&request.source_column);
    if request.source_table != plan.entity.table {
        return Err(TTGenError::InvalidPlan(format!(
            "{split_kind} split_column must reference entity table '{}' for scalar tt-gen projection, got '{}'",
            plan.entity.table, split_column
        )));
    }

    let col_idx = find_col_idx_in_context(entity_ctx, &request.source_column)
        .map_err(|e| TTGenError::SchemaError(format!("resolving {split_kind} split_column: {}", e)))?;
    Ok(Some(ResolvedSplitProjection {
        output_name: request.output_name.clone(),
        entity_col_idx: Some(col_idx),
        dtype: DataType::Utf8,
    }))
}

fn validate_link_split_projection_attached<U: UnitSystem>(
    request: Option<&SplitProjectionRequest>,
    stages: &[ResolvedStage<U>],
) -> TTGenResult<()> {
    let Some(request) = request else {
        return Ok(());
    };
    if request.kind != SplitProjectionKind::LinkArray {
        return Ok(());
    }

    let mut has_attached_projection = false;
    let mut traversed_tables = BTreeSet::new();
    for stage in stages {
        for op in &stage.ops {
            collect_list_distinct_split_projection_state(
                op,
                &mut has_attached_projection,
                &mut traversed_tables,
            );
        }
    }
    if has_attached_projection {
        return Ok(());
    }

    let traversed = if traversed_tables.is_empty() {
        "none".to_string()
    } else {
        traversed_tables.into_iter().collect::<Vec<_>>().join(", ")
    };
    Err(TTGenError::InvalidPlan(format!(
        "CustomLinkSplit split_column '{}.{}' references table '{}', but no ListDistinct aggregate in the plan traverses that table; ListDistinct traversed table(s): {}",
        request.source_table, request.source_column, request.source_table, traversed
    )))
}

fn collect_list_distinct_split_projection_state<U: UnitSystem>(
    op: &ResolvedOp<U>,
    has_attached_projection: &mut bool,
    traversed_tables: &mut BTreeSet<String>,
) {
    match op {
        ResolvedOp::Aggregate(agg) if agg.func == AggFunc::ListDistinct => {
            if let Some(edge) = agg.path.last() {
                traversed_tables.insert(edge.dst_table.node_type().to_string());
            }
            if agg.link_split_col_idx.is_some() {
                *has_attached_projection = true;
            }
        }
        ResolvedOp::Autoregression(agg) => collect_list_distinct_split_projection_state(
            &agg.target_op,
            has_attached_projection,
            traversed_tables,
        ),
        _ => {}
    }
}

fn split_column_table_and_column<'a>(
    split_column: &'a str,
    default_table: &'a str,
) -> (String, String) {
    match split_column.rsplit_once('.') {
        Some((table, column)) if !table.is_empty() && !column.is_empty() => {
            (table.to_string(), column.to_string())
        }
        _ => (default_table.to_string(), split_column.to_string()),
    }
}

fn find_col_idx_in_context(
    ctx: &Option<RowPackContext>,
    col_name: &str,
) -> TTGenResult<usize> {
    let ctx = ctx
        .as_ref()
        .ok_or_else(|| TTGenError::SystemError("no rowpack context available".to_string()))?;
    ctx.columns
        .iter()
        .position(|c| c.name == col_name)
        .ok_or_else(|| TTGenError::SchemaError(format!("column '{}' not found in rowpack context", col_name)))
}

fn collect_entity_col_refs<U: UnitSystem>(
    plan: &DiskGraphTTPlan<U>,
    entity_ctx: &Option<RowPackContext>,
) -> TTGenResult<Vec<(String, usize)>> {
    let mut refs = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Helper closure to collect EntityColumn refs from filters
    fn collect_from_row_filter(f: &RowFilter, seen: &mut std::collections::HashSet<String>) {
        match f {
            RowFilter::Compare { rhs, .. } => {
                if let RowFilterRhs::EntityColumn(col) = rhs {
                    seen.insert(col.clone());
                }
            }
            RowFilter::And(filters) | RowFilter::Or(filters) => {
                for f in filters {
                    collect_from_row_filter(f, seen);
                }
            }
            RowFilter::Not(f) => collect_from_row_filter(f, seen),
        }
    }

    fn collect_from_stage_filter(f: &StageFilter, seen: &mut std::collections::HashSet<String>) {
        match f {
            StageFilter::Compare { rhs, .. } => {
                if let StageFilterRhs::EntityColumn(col) = rhs {
                    seen.insert(col.clone());
                }
            }
            StageFilter::And(filters) | StageFilter::Or(filters) => {
                for f in filters {
                    collect_from_stage_filter(f, seen);
                }
            }
            StageFilter::Not(f) => collect_from_stage_filter(f, seen),
        }
    }

    // Entity filter
    if let Some(f) = &plan.entity.filter {
        collect_from_row_filter(f, &mut seen);
    }

    // Stage filters and row filters
    for stage in &plan.stages {
        for op in &stage.ops {
            if let Op::Aggregate { row_filter: Some(f), .. } = op {
                collect_from_row_filter(f, &mut seen);
            }
            if let Op::HasEdges { row_filter: Some(f), .. } = op {
                collect_from_row_filter(f, &mut seen);
            }
        }
        if let Some(f) = &stage.filter {
            collect_from_stage_filter(f, &mut seen);
        }
    }

    let mut names: Vec<_> = seen.into_iter().collect();
    names.sort();
    for col_name in names {
        let idx = find_col_idx_in_context(entity_ctx, &col_name)?;
        refs.push((col_name, idx));
    }

    Ok(refs)
}

fn resolve_stage<U: UnitSystem>(
    stage: &Stage<U>,
    reader: &GraphIndexReader,
    entity_table: &Arc<TableReader>,
    entity_ctx: &Option<RowPackContext>,
    alias_to_idx: &HashMap<String, usize>,
    entity_col_indices: &[(String, usize)],
    split_request: Option<&SplitProjectionRequest>,
) -> TTGenResult<ResolvedStage<U>> {
    let mut ops = Vec::new();
    for op in &stage.ops {
        let resolved = resolve_op(
            op,
            reader,
            entity_table,
            entity_ctx,
            alias_to_idx,
            entity_col_indices,
            split_request,
        )?;
        ops.push(resolved);
    }

    let filter = stage
        .filter
        .as_ref()
        .map(|f| resolve_stage_filter(f, reader, entity_table, entity_ctx, alias_to_idx, entity_col_indices))
        .transpose().map_err(|e| TTGenError::SchemaError(format!("resolving stage filter: {}", e)))?;

    Ok(ResolvedStage { ops, filter })
}

fn resolve_op<U: UnitSystem>(
    op: &Op<U>,
    reader: &GraphIndexReader,
    _entity_table: &Arc<TableReader>,
    _entity_ctx: &Option<RowPackContext>,
    alias_to_idx: &HashMap<String, usize>,
    entity_col_indices: &[(String, usize)],
    split_request: Option<&SplitProjectionRequest>,
) -> TTGenResult<ResolvedOp<U>> {
    match op {
        Op::Aggregate {
            alias,
            path,
            func,
            target_col,
            time_col: _,
            window,
            window_unit,
            row_filter,
            limit,
            target_entity_table,
        } => {
            let alias_idx = alias_to_idx[alias];
            let resolved_path = resolve_path(path, reader)?;

            // Resolve target column on the destination table
            let dst_table = &resolved_path.last().unwrap().dst_table;
            let dst_ctx = dst_table.node_data_decoder().map(|d| d.context().clone());
            let target_col_idx = target_col
                .as_ref()
                .map(|col| find_col_idx_in_context(&dst_ctx, col))
                .transpose().map_err(|e| TTGenError::SchemaError(format!("resolving aggregate target column: {}", e)))?;

            // Resolve window — applies to the FIRST hop (reverse FK with sort keys),
            // matching Kumo's temporal join semantics.
            let resolved_window: Option<ResolvedWindow<U>> = match window {
                Some((lo, hi)) => {
                    let first_edge = resolved_path.first().unwrap();
                    let encoder = first_edge.encoder.clone().ok_or_else(|| {
                        TTGenError::InvalidPlan("cannot apply time window on edge relation without sort key timestamp".to_string())
                    })?;
                    let unit = window_unit.as_ref().unwrap_or(&TimeUnit::Seconds);
                    match unit {
                        TimeUnit::Months => Some(ResolvedWindow::Calendar {
                            lo_months: *lo as i32,
                            hi_months: *hi as i32,
                            encoder,
                        }),
                        TimeUnit::Years => Some(ResolvedWindow::Calendar {
                            lo_months: (*lo as i32) * 12,
                            hi_months: (*hi as i32) * 12,
                            encoder,
                        }),
                        _ => {
                            let unit_secs = time_unit_to_secs(unit);
                            Some(ResolvedWindow::Fixed {
                                offset_lo: Dur::<U>::from_secs(lo * unit_secs),
                                offset_hi: Dur::<U>::from_secs(hi * unit_secs),
                                encoder,
                            })
                        }
                    }
                }
                None => None,
            };

            // Use plan-level entity_col_indices so positions match the pre-resolved
            // entity_cols array built in execute_entity.
            let resolved_filter = row_filter
                .as_ref()
                .map(|f| resolve_row_filter(f, reader, dst_table, &dst_ctx, entity_col_indices))
                .transpose().map_err(|e| TTGenError::SchemaError(format!("resolving aggregate row filter: {}", e)))?;

            // For LP `ListDistinct`: look up the target entity table whose PK
            // map should be consulted for orphan filtering. The plan's
            // `target_entity_table` is authoritative — the caller (kumo)
            // resolved it from the graph schema. `None` means no filter.
            let target_pk_table = match (func, target_entity_table.as_deref()) {
                (AggFunc::ListDistinct, Some(name)) => {
                    let table = reader.table(name).ok_or_else(|| {
                        TTGenError::SchemaError(format!(
                            "target_entity_table '{name}' not found in index"
                        ))
                    })?;
                    if table.pk_map().is_none() {
                        return Err(TTGenError::SchemaError(format!(
                            "target_entity_table '{name}' has no primary key map"
                        )));
                    }
                    tracing::info!(
                        target_entity_table = name,
                        pk_map_shards = table.pk_map().map(|m| m.shard_count()).unwrap_or(0),
                        create_time_col = ?table.create_time_col(),
                        end_time_col = ?table.end_time_col(),
                        alias = %alias,
                        "[orphan-filter] ACTIVATED for ListDistinct"
                    );
                    Some(table)
                }
                _ => None,
            };

            // Pre-resolve temporal column indices for the target entity table.
            let (target_temporal_create_col_idx, target_temporal_end_col_idx) =
                target_pk_table
                    .as_ref()
                    .map(|t| t.temporal_col_indices())
                    .unwrap_or((None, None));

            if target_temporal_create_col_idx.is_some() || target_temporal_end_col_idx.is_some() {
                tracing::info!(
                    create_col_idx = ?target_temporal_create_col_idx,
                    end_col_idx = ?target_temporal_end_col_idx,
                    alias = %alias,
                    "[orphan-filter] temporal validity check enabled"
                );
            }

            let link_split_col_idx = match (func, split_request) {
                (AggFunc::ListDistinct, Some(req))
                    if req.kind == SplitProjectionKind::LinkArray
                        && req.source_table == dst_table.node_type() =>
                {
                    Some(find_col_idx_in_context(&dst_ctx, &req.source_column).map_err(|e| {
                        TTGenError::SchemaError(format!(
                            "resolving CustomLinkSplit split_column: {}",
                            e
                        ))
                    })?)
                }
                _ => None,
            };

            // Resolve the source-column stype from the persisted graph config
            // (`{index_root}/diskgraph_config.json`, populated at ingest time).
            // This replaces the per-query `column_metadata` plumbing on the
            // kumo→diskgraph compile boundary — stype is a one-time fact
            // declared at ingest, not a per-query parameter.
            let resolved_stype = target_col_idx.and_then(|col_idx| {
                let col_name = &dst_table
                    .node_data_decoder()?
                    .context()
                    .columns[col_idx]
                    .name;
                reader
                    .column_semantic_type(dst_table.node_type(), col_name)
                    .map(String::from)
            });

            // Reject (Multicategorical, Float*Array) at plan resolution: float
            // -keyed labels are not a real workload (precision drift, NaN,
            // signed zero, infinities). Rejection is a single-point gate
            // shared by every read site (kumo#28400 §4.2 Step 1.6).
            //
            // Inverse gate: a native-list source column read without a
            // multicat declaration on disk would silently produce
            // `Value::Int(0)` via `read_column_value`'s unhandled-array
            // short-circuit. Reject loudly so a missing semantic_type on the
            // ingested column fails plan compilation instead of corrupting
            // downstream stats and TT output.
            if let Some(col_idx) = target_col_idx {
                let dst_col = &dst_table
                    .node_data_decoder()
                    .ok_or_else(|| TTGenError::SchemaError(
                        "destination table missing node-data decoder".to_string()
                    ))?
                    .context()
                    .columns[col_idx];
                let is_multicat = resolved_stype.as_deref()
                    == Some("multicategorical");
                match dst_col.logical_type {
                    IngestedType::Float16Array
                    | IngestedType::Float32Array
                    | IngestedType::Float64Array
                        if is_multicat => {
                        return Err(TTGenError::InvalidPlan(format!(
                            "multicategorical source column '{}' has unsupported \
                             float-list ingested type {:?}; use intlist or stringlist",
                            dst_col.name, dst_col.logical_type
                        )));
                    }
                    IngestedType::StringArray | IngestedType::Int64Array
                        if !is_multicat =>
                    {
                        return Err(TTGenError::InvalidPlan(format!(
                            "native-list source column '{}.{}' (ingested type {:?}) \
                             has no multicategorical semantic_type in the \
                             persisted diskgraph_config.json; got {:?}. \
                             Re-ingest with stype=multicategorical declared on \
                             the column, or change the column declaration.",
                            dst_table.node_type(),
                            dst_col.name,
                            dst_col.logical_type,
                            resolved_stype
                        )));
                    }
                    // List-shape WITH multicat declaration: the supported path.
                    // Falls through to the OK case below; listed explicitly so
                    // the scalar-multicat catch-all does not swallow it.
                    IngestedType::StringArray | IngestedType::Int64Array => {}
                    _ if is_multicat => {
                        // Defensive net for legacy indexes built before
                        // graph_config::validate rejected scalar-multicat. A
                        // scalar dtype paired with multicat is degenerate: a
                        // single-value-per-row column is `categorical`, not
                        // multicategorical. Fail loudly rather than silently
                        // observe whole-cell.
                        return Err(TTGenError::InvalidPlan(format!(
                            "multicategorical source column '{}.{}' has scalar \
                             ingested type {:?}; multicategorical requires a \
                             native list shape (string_array / int64_array). \
                             Re-ingest with stype=categorical for single-value \
                             columns, or with a list-typed source dtype.",
                            dst_table.node_type(),
                            dst_col.name,
                            dst_col.logical_type,
                        )));
                    }
                    _ => {}
                }
            }

            Ok(ResolvedOp::Aggregate(ResolvedAggregate {
                alias_idx,
                path: resolved_path,
                func: *func,
                target_col_idx,
                window: resolved_window,
                row_filter: resolved_filter,
                limit: *limit,
                target_pk_table,
                target_temporal_create_col_idx,
                target_temporal_end_col_idx,
                link_split_col_idx,
                target_semantic_type: resolved_stype,
            }))
        }
        Op::HasEdges {
            alias,
            path,
            time_col: _,
            window,
            window_unit,
            row_filter,
        } => {
            let alias_idx = alias_to_idx[alias];
            let resolved_path = resolve_path(path, reader)?;
            let dst_table = &resolved_path.last().unwrap().dst_table;
            let dst_ctx = dst_table.node_data_decoder().map(|d| d.context().clone());

            let resolved_window: Option<ResolvedWindow<U>> = match window {
                Some((lo, hi)) => {
                    let first_edge = resolved_path.first().unwrap();
                    let encoder = first_edge.encoder.clone().ok_or_else(|| {
                        TTGenError::InvalidPlan("cannot apply time window on edge relation without sort key timestamp".to_string())
                    })?;
                    let unit = window_unit.as_ref().unwrap_or(&TimeUnit::Seconds);
                    match unit {
                        TimeUnit::Months => Some(ResolvedWindow::Calendar {
                            lo_months: *lo as i32,
                            hi_months: *hi as i32,
                            encoder,
                        }),
                        TimeUnit::Years => Some(ResolvedWindow::Calendar {
                            lo_months: (*lo as i32) * 12,
                            hi_months: (*hi as i32) * 12,
                            encoder,
                        }),
                        _ => {
                            let unit_secs = time_unit_to_secs(unit);
                            Some(ResolvedWindow::Fixed {
                                offset_lo: Dur::<U>::from_secs(lo * unit_secs),
                                offset_hi: Dur::<U>::from_secs(hi * unit_secs),
                                encoder,
                            })
                        }
                    }
                }
                None => None,
            };
            let resolved_filter = row_filter
                .as_ref()
                .map(|f| resolve_row_filter(f, reader, dst_table, &dst_ctx, entity_col_indices))
                .transpose().map_err(|e| TTGenError::SchemaError(format!("resolving has_edges row filter: {}", e)))?;

            Ok(ResolvedOp::HasEdges(ResolvedHasEdges {
                alias_idx,
                path: resolved_path,
                window: resolved_window,
                row_filter: resolved_filter,
            }))
        }
        Op::Column {
            alias,
            table,
            column,
        } => {
            let alias_idx = alias_to_idx[alias];
            let table_reader = reader
                .table(table)
                .ok_or_else(|| TTGenError::SchemaError(format!("table '{}' not found", table)))?;
            let ctx = table_reader.node_data_decoder().map(|d| d.context().clone());
            let col_idx = find_col_idx_in_context(&ctx, column)?;

            // Resolve stype from the persisted graph config. Same source as
            // the Aggregate arm — see comment there for rationale.
            let resolved_stype = reader
                .column_semantic_type(table, column)
                .map(String::from);

            // Mirror of the Aggregate-arm reject gates: Route A direct-target
            // reads share the same Float*Array reject and the inverse
            // native-list-without-multicat reject (kumo#28400 §4.2 Step 1.6
            // + codex review follow-up).
            {
                let col_meta = &table_reader
                    .node_data_decoder()
                    .ok_or_else(|| TTGenError::SchemaError(
                        "table missing node-data decoder".to_string()
                    ))?
                    .context()
                    .columns[col_idx];
                let is_multicat = resolved_stype.as_deref()
                    == Some("multicategorical");
                match col_meta.logical_type {
                    IngestedType::Float16Array
                    | IngestedType::Float32Array
                    | IngestedType::Float64Array
                        if is_multicat => {
                        return Err(TTGenError::InvalidPlan(format!(
                            "multicategorical source column '{}' has unsupported \
                             float-list ingested type {:?}; use intlist or stringlist",
                            col_meta.name, col_meta.logical_type
                        )));
                    }
                    IngestedType::StringArray | IngestedType::Int64Array
                        if !is_multicat =>
                    {
                        return Err(TTGenError::InvalidPlan(format!(
                            "native-list source column '{}.{}' (ingested type {:?}) \
                             has no multicategorical semantic_type in the \
                             persisted diskgraph_config.json; got {:?}. \
                             Re-ingest with stype=multicategorical declared on \
                             the column, or change the column declaration.",
                            table, col_meta.name, col_meta.logical_type, resolved_stype
                        )));
                    }
                    // List-shape WITH multicat declaration: the supported path.
                    // Falls through to the OK case below; listed explicitly so
                    // the scalar-multicat catch-all does not swallow it.
                    IngestedType::StringArray | IngestedType::Int64Array => {}
                    _ if is_multicat => {
                        // Defensive net for legacy indexes built before
                        // graph_config::validate rejected scalar-multicat. See
                        // the matching gate in the Aggregate arm above for
                        // rationale.
                        return Err(TTGenError::InvalidPlan(format!(
                            "multicategorical source column '{}.{}' has scalar \
                             ingested type {:?}; multicategorical requires a \
                             native list shape (string_array / int64_array). \
                             Re-ingest with stype=categorical for single-value \
                             columns, or with a list-typed source dtype.",
                            table, col_meta.name, col_meta.logical_type,
                        )));
                    }
                    _ => {}
                }
            }

            Ok(ResolvedOp::Column(ResolvedColumn {
                alias_idx,
                table: table_reader,
                col_idx,
                target_semantic_type: resolved_stype,
            }))
        }
        Op::Compare { alias, lhs, op, rhs } => {
            let alias_idx = alias_to_idx[alias];
            let lhs_alias_idx = alias_to_idx
                .get(lhs)
                .copied()
                .ok_or_else(|| TTGenError::SchemaError(format!("alias '{}' not found", lhs)))?;
            Ok(ResolvedOp::Compare(ResolvedCompare {
                alias_idx,
                lhs_alias_idx,
                op: *op,
                rhs: rhs.clone(),
            }))
        }
        Op::LogicalOp { alias, op, operands } => {
            let alias_idx = alias_to_idx[alias];
            let operand_indices = operands
                .iter()
                .map(|name| {
                    alias_to_idx
                        .get(name)
                        .copied()
                        .ok_or_else(|| TTGenError::SchemaError(format!("alias '{}' not found", name)))
                })
                .collect::<TTGenResult<Vec<_>>>()?;
            Ok(ResolvedOp::LogicalOp(ResolvedLogicalOp {
                alias_idx,
                op: *op,
                operand_indices,
            }))
        }
        Op::Autoregression { alias, target_op, lag_offsets } => {
            let alias_idx = alias_to_idx[alias];
            let resolved_target = resolve_op(
                target_op,
                reader,
                _entity_table,
                _entity_ctx,
                alias_to_idx,
                entity_col_indices,
                split_request,
            )?;
            Ok(ResolvedOp::Autoregression(ResolvedAutoregression {
                alias_idx,
                target_op: Box::new(resolved_target),
                lag_offsets: lag_offsets.clone(),
            }))
        }
    }
}

fn resolve_path(
    path: &[EdgeRef],
    reader: &GraphIndexReader,
) -> TTGenResult<Vec<ResolvedEdge>> {
    let mut resolved = Vec::with_capacity(path.len());
    for edge_ref in path {
        let src_table = reader
            .table(&edge_ref.src_table)
            .ok_or_else(|| TTGenError::SchemaError(format!("src table '{}' not found", edge_ref.src_table)))?;
        let dst_table = reader
            .table(&edge_ref.dst_table)
            .ok_or_else(|| TTGenError::SchemaError(format!("dst table '{}' not found", edge_ref.dst_table)))?;

        // Find the relation in src_table's relations.
        // Multiple relations can share the same (src_type, dst_type) — e.g.,
        // orders.buyer_id→users and orders.seller_id→users. We disambiguate by
        // matching the relation name against the EdgeRef column names.
        let relations = src_table.relations();
        let (relation_idx, relation_meta) = find_relation(relations, edge_ref)?;

        let layouts = src_table.layouts();
        let layout = layouts
            .get(relation_idx)
            .cloned()
            .ok_or_else(|| TTGenError::SystemError(format!("layout not found for relation index {}", relation_idx)))?;

        let encoder = relation_meta
            .dest_sort_key
            .as_ref()
            .map(SortKeyEncoder::new);

        resolved.push(ResolvedEdge {
            src_table,
            relation_idx,
            layout,
            encoder,
            dst_table,
        });
    }
    Ok(resolved)
}

#[allow(clippy::only_used_in_recursion)]
fn resolve_row_filter(
    filter: &RowFilter,
    reader: &GraphIndexReader,
    current_table: &Arc<TableReader>,
    current_ctx: &Option<RowPackContext>,
    entity_col_indices: &[(String, usize)],
) -> TTGenResult<ResolvedRowFilter> {
    match filter {
        RowFilter::Compare { lhs, op, rhs } => {
            let resolved_lhs = match lhs {
                RowFilterLhs::Column(col) => {
                    let idx = find_col_idx_in_context(current_ctx, col)?;
                    ResolvedRowFilterLhs::Column(idx)
                }
                RowFilterLhs::Remote { path, column } => {
                    let resolved_path = resolve_path(path, reader)?;
                    let dst_table = &resolved_path.last().unwrap().dst_table;
                    let dst_ctx = dst_table.node_data_decoder().map(|d| d.context().clone());
                    let col_idx = find_col_idx_in_context(&dst_ctx, column)?;
                    ResolvedRowFilterLhs::Remote {
                        path: resolved_path,
                        col_idx,
                    }
                }
            };

            let resolved_rhs = match rhs {
                RowFilterRhs::Constant(v) => ResolvedRowFilterRhs::Constant(v.clone()),
                RowFilterRhs::EntityColumn(col) => {
                    let idx = entity_col_indices
                        .iter()
                        .position(|(name, _)| name == col)
                        .ok_or_else(|| TTGenError::SchemaError(format!("entity column '{}' not found in refs", col)))?;
                    ResolvedRowFilterRhs::EntityColumn(idx)
                }
                RowFilterRhs::Remote { path, column } => {
                    let resolved_path = resolve_path(path, reader)?;
                    let dst_table = &resolved_path.last().unwrap().dst_table;
                    let dst_ctx = dst_table.node_data_decoder().map(|d| d.context().clone());
                    let col_idx = find_col_idx_in_context(&dst_ctx, column)?;
                    ResolvedRowFilterRhs::Remote {
                        path: resolved_path,
                        col_idx,
                    }
                }
            };

            Ok(ResolvedRowFilter::Compare {
                lhs: resolved_lhs,
                op: *op,
                rhs: resolved_rhs,
            })
        }
        RowFilter::And(filters) => {
            let resolved = filters
                .iter()
                .map(|f| resolve_row_filter(f, reader, current_table, current_ctx, entity_col_indices))
                .collect::<TTGenResult<Vec<_>>>()?;
            Ok(ResolvedRowFilter::And(resolved))
        }
        RowFilter::Or(filters) => {
            let resolved = filters
                .iter()
                .map(|f| resolve_row_filter(f, reader, current_table, current_ctx, entity_col_indices))
                .collect::<TTGenResult<Vec<_>>>()?;
            Ok(ResolvedRowFilter::Or(resolved))
        }
        RowFilter::Not(f) => {
            let resolved = resolve_row_filter(f, reader, current_table, current_ctx, entity_col_indices)?;
            Ok(ResolvedRowFilter::Not(Box::new(resolved)))
        }
    }
}

#[allow(clippy::only_used_in_recursion)]
fn resolve_stage_filter(
    filter: &StageFilter,
    reader: &GraphIndexReader,
    entity_table: &Arc<TableReader>,
    entity_ctx: &Option<RowPackContext>,
    alias_to_idx: &HashMap<String, usize>,
    entity_col_indices: &[(String, usize)],
) -> TTGenResult<ResolvedStageFilter> {
    match filter {
        StageFilter::Compare { alias, op, rhs } => {
            let alias_idx = alias_to_idx
                .get(alias)
                .copied()
                .ok_or_else(|| TTGenError::SchemaError(format!("alias '{}' not found", alias)))?;

            let resolved_rhs = match rhs {
                StageFilterRhs::Constant(v) => ResolvedStageFilterRhs::Constant(v.clone()),
                StageFilterRhs::EntityColumn(col) => {
                    // Use position in entity_col_indices (not rowpack column index!)
                    // so it matches the pre-resolved entity_cols array.
                    let idx = entity_col_indices
                        .iter()
                        .position(|(name, _)| name == col)
                        .ok_or_else(|| TTGenError::SchemaError(format!("entity column '{}' not found in refs", col)))?;
                    ResolvedStageFilterRhs::EntityColumn(idx)
                }
                StageFilterRhs::Remote { path, column } => {
                    let resolved_path = resolve_path(path, reader)?;
                    let dst_table = &resolved_path.last().unwrap().dst_table;
                    let dst_ctx = dst_table.node_data_decoder().map(|d| d.context().clone());
                    let col_idx = find_col_idx_in_context(&dst_ctx, column)?;
                    ResolvedStageFilterRhs::Remote {
                        path: resolved_path,
                        col_idx,
                    }
                }
                StageFilterRhs::Alias(a) => {
                    let idx = alias_to_idx
                        .get(a)
                        .copied()
                        .ok_or_else(|| TTGenError::SchemaError(format!("alias '{}' not found", a)))?;
                    ResolvedStageFilterRhs::Alias(idx)
                }
            };

            Ok(ResolvedStageFilter::Compare {
                alias_idx,
                op: *op,
                rhs: resolved_rhs,
            })
        }
        StageFilter::And(filters) => {
            let resolved = filters
                .iter()
                .map(|f| resolve_stage_filter(f, reader, entity_table, entity_ctx, alias_to_idx, entity_col_indices))
                .collect::<TTGenResult<Vec<_>>>()?;
            Ok(ResolvedStageFilter::And(resolved))
        }
        StageFilter::Or(filters) => {
            let resolved = filters
                .iter()
                .map(|f| resolve_stage_filter(f, reader, entity_table, entity_ctx, alias_to_idx, entity_col_indices))
                .collect::<TTGenResult<Vec<_>>>()?;
            Ok(ResolvedStageFilter::Or(resolved))
        }
        StageFilter::Not(f) => {
            let resolved = resolve_stage_filter(f, reader, entity_table, entity_ctx, alias_to_idx, entity_col_indices)?;
            Ok(ResolvedStageFilter::Not(Box::new(resolved)))
        }
    }
}

/// Find the correct relation for an EdgeRef, disambiguating when multiple
/// relations share the same dst_type.
///
/// Matching strategy:
/// 1. Construct the expected relation name from EdgeRef columns
///    (FK forward: `{src}_{src_col}_to_{dst}_{dst_col}`,
///    FK reverse: `...._reverse`,
///    edge table: `edge_{table}_src_to_dst` / `edge_{table}_dst_to_src`).
/// 2. Try exact name match.
/// 3. Fall back to matching on dst_type alone (requires exactly one match).
fn find_relation<'a>(
    relations: &'a [diskgraph_core::metadata::EdgeRelationMetadata],
    edge_ref: &EdgeRef,
) -> TTGenResult<(usize, &'a diskgraph_core::metadata::EdgeRelationMetadata)> {
    // Candidates matching dst_type
    let candidates: Vec<(usize, &diskgraph_core::metadata::EdgeRelationMetadata)> = relations
        .iter()
        .enumerate()
        .filter(|(_, r)| r.dst_type == edge_ref.dst_table)
        .collect();

    if candidates.is_empty() {
        return Err(TTGenError::InvalidPlan(format!(
            "edge relation from '{}' to '{}' not found",
            edge_ref.src_table, edge_ref.dst_table,
        )));
    }

    // If only one candidate, no ambiguity
    if candidates.len() == 1 {
        return Ok(candidates[0]);
    }

    // Multiple candidates — disambiguate by relation name.
    // Try: {src}_{src_col}_to_{dst}_{dst_col} (FK forward)
    let expected_fwd = format!(
        "{}_{}_to_{}_{}",
        edge_ref.src_table, edge_ref.src_col, edge_ref.dst_table, edge_ref.dst_col,
    );
    // Try: ...._reverse (FK reverse has this suffix)
    let expected_rev = format!("{}_reverse", expected_fwd);

    for &(idx, rel) in &candidates {
        if rel.name == expected_fwd || rel.name == expected_rev {
            return Ok((idx, rel));
        }
    }

    // Try: relation name contains both src_col and dst_col
    for &(idx, rel) in &candidates {
        if rel.name.contains(&edge_ref.src_col) && rel.name.contains(&edge_ref.dst_col) {
            return Ok((idx, rel));
        }
    }

    // Last resort: relation name contains src_col
    for &(idx, rel) in &candidates {
        if rel.name.contains(&edge_ref.src_col) {
            return Ok((idx, rel));
        }
    }

    Err(TTGenError::InvalidPlan(format!(
        "ambiguous: {} relations from '{}' to '{}', none match columns '{}.{}'. \
         Relation names: [{}]",
        candidates.len(),
        edge_ref.src_table,
        edge_ref.dst_table,
        edge_ref.src_col,
        edge_ref.dst_col,
        candidates.iter().map(|(_, r)| r.name.as_str()).collect::<Vec<_>>().join(", "),
    )))
}

/// Infer the Arrow DataType for the TARGET column.
fn infer_target_dtype<U: UnitSystem>(plan: &DiskGraphTTPlan<U>, stages: &[ResolvedStage<U>]) -> DataType {
    let last_plan_op = plan.stages.iter().rev()
        .flat_map(|s| s.ops.iter().rev())
        .next();
    match last_plan_op {
        // ListDistinct + Link prediction: source column dtype (exploded/array rows).
        Some(Op::Aggregate { func: AggFunc::ListDistinct, .. })
            if matches!(plan.prediction_type, PredictionType::Link { .. }) =>
        {
            lookup_last_agg_col_dtype(stages).unwrap_or(DataType::Float64)
        }
        // ListDistinct + Node prediction (CLASSIFY): pipe-separated string.
        Some(Op::Aggregate { func: AggFunc::ListDistinct, .. }) => DataType::Utf8,
        Some(op) => infer_op_dtype(op, stages),
        None => {
            // No ops (e.g. LP with implicit ListDistinct): look up ingested type.
            lookup_last_agg_col_dtype(stages).unwrap_or(DataType::Float64)
        }
    }
}

/// Infer Arrow DataType for a specific Op.
fn infer_op_dtype<U: UnitSystem>(op: &Op<U>, stages: &[ResolvedStage<U>]) -> DataType {
    match op {
        Op::Compare { .. } | Op::LogicalOp { .. } => DataType::Boolean,
        // COUNT/COUNT_DISTINCT always produce integer counts.
        Op::Aggregate {
            func: AggFunc::Count | AggFunc::CountDistinct,
            ..
        } =>
        {
            DataType::Int64
        }
        Op::HasEdges { .. } => DataType::Boolean,
        Op::Autoregression { target_op, .. } => {
            // Autoregression yields a List of whatever the inner target_op yields
            let inner_dtype = infer_op_dtype(target_op, stages);
            DataType::List(Arc::new(Field::new("item", inner_dtype, true)))
        },

        // FIRST/LAST preserve the source column's dtype (e.g. FIRST of string → Utf8).
        Op::Aggregate {
            func: AggFunc::First | AggFunc::Last,
            ..
        } =>
        {
            lookup_last_agg_col_dtype(stages).unwrap_or(DataType::Float64)
        }
        // Direct column prediction: look up the column's ingested type.
        Op::Column { .. } => {
            let last_col = stages.iter().rev()
                .flat_map(|s| s.ops.iter().rev())
                .find_map(|op| {
                    if let ResolvedOp::Column(c) = op { Some(c) } else { None }
                });
            if let Some(col) = last_col {
                if let Some(decoder) = col.table.node_data_decoder() {
                    let c = &decoder.context().columns[col.col_idx];
                    return ingested_type_to_arrow(&c.logical_type);
                }
            }
            DataType::Float64
        }
        _ => DataType::Float64,
    }
}

/// Look up the ingested column type from the last resolved aggregate.
fn lookup_last_agg_col_dtype<U: UnitSystem>(stages: &[ResolvedStage<U>]) -> Option<DataType> {
    let last_agg = stages.iter().rev()
        .flat_map(|s| s.ops.iter().rev())
        .find_map(|op| {
            if let ResolvedOp::Aggregate(a) = op { Some(a) } else { None }
        })?;
    let col_idx = last_agg.target_col_idx?;
    let dst_table = &last_agg.path.last()?.dst_table;
    let decoder = dst_table.node_data_decoder()?;
    let col = &decoder.context().columns[col_idx];
    Some(ingested_type_to_arrow(&col.logical_type))
}

/// Map an ingested column type to the Arrow DataType used for the ENTITY column.
/// All integer types widen to Int64 (matching our Value::Int(i64) representation).
fn ingested_type_to_arrow(it: &IngestedType) -> DataType {
    match it {
        IngestedType::Int8
        | IngestedType::Int16
        | IngestedType::Int32
        | IngestedType::Int64
        | IngestedType::UInt8
        | IngestedType::UInt16
        | IngestedType::UInt32
        | IngestedType::UInt64 => DataType::Int64,
        IngestedType::Float32 | IngestedType::Float64 => DataType::Float64,
        IngestedType::Bool => DataType::Boolean,
        // String, Bytes, Uuid, Ip, Date, Timestamp, Decimal, arrays → Utf8
        _ => DataType::Utf8,
    }
}

fn time_unit_to_secs(unit: &TimeUnit) -> i64 {
    match unit {
        TimeUnit::Seconds => 1,
        TimeUnit::Minutes => 60,
        TimeUnit::Hours => 3600,
        TimeUnit::Days => 86400,
        TimeUnit::Weeks => 604800,
        TimeUnit::Milliseconds | TimeUnit::Microseconds | TimeUnit::Nanoseconds => {
            unreachable!("Sub-second units are not supported for PQL window aggregation")
        }
        // Months and Years are handled as Calendar windows, not fixed seconds.
        TimeUnit::Months | TimeUnit::Years => {
            unreachable!("Months/Years should use ResolvedWindow::Calendar")
        }
    }
}
