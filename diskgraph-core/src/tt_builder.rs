//! Training-table execution plan types.
//!
//! `DiskGraphTTPlan` is the contract between the Python-side
//! `DiskGraphTTBuilder` (PyO3, in `diskgraph-pybind`) and the Rust executor.
//! Python walks the PQL AST, calls builder methods, and `build()` produces a
//! `DiskGraphTTPlan`. The executor consumes it.
//!
//! ## Builder vs Plan
//!
//! - **`DiskGraphTTBuilder`** (Python): mutable, accumulates ops via method
//!   calls (`for_each`, `aggregate`, `filter`, …).
//! - **`DiskGraphTTPlan`** (Rust): immutable output of `build()`. Consumed by
//!   the executor; never exposed to Python directly.
//!
//! ## Staged execution
//!
//! A plan is a flat list of [`Stage`]s. Each stage has ops and an optional
//! [`StageFilter`]. If the filter fails for an entity, all subsequent stages
//! are skipped (short-circuit).
//!
//! ## Two filter types
//!
//! - [`RowFilter`]: evaluated per-row during edge traversal. LHS is a column
//!   value (local or remote via sub-traversal). RHS is a constant, entity
//!   column, or remote column.
//! - [`StageFilter`]: evaluated per-entity between stages. LHS is a computed
//!   alias. RHS is a constant, entity column, remote column, or another alias.

pub use crate::time::TimeUnit;

// ─── Reserved column names ────────────────────────────────────────────────────

/// Primary entity column emitted by ttgen (`ENTITY1`).
pub const PRIMARY_ENTITY_COL: &str = "ENTITY1";
/// Temporal anchor column emitted by ttgen (`TIMESTAMP`).
pub const TIMESTAMP_COL: &str = "TIMESTAMP";
/// Primary target column emitted by ttgen (`TARGET1`).
pub const PRIMARY_TARGET_COL: &str = "TARGET1";
/// Per-row training-weight column. Rides each seed row from gen (when
/// `weight_col` is set) through pass-B routing into the stage parquet and
/// is ingested as a numeric rowpack column. The engine reads it into
/// `task.weight` and MUST skip it in `project_seed_rowpack` so it is never
/// encoded as a model feature.
pub const WEIGHT_COL: &str = "WEIGHT";
/// Resolved LHS entity node-id (`u64` scalar); the engine seeds off this column.
pub const ENTITY_NODE_ID_COL: &str = "entity_node_id";
/// Resolved LP RHS positives (`list<i64>` node-ids).
pub const TARGET_NODE_IDS_COL: &str = "target_node_ids";

use crate::time::{Abs, Dur, UnitSystem};
use serde::{Deserialize, Serialize};

// ─── Top-level plan ──────────────────────────────────────────────────────────

/// A fully-specified training-table execution plan, parameterised by
/// the index's timestamp unit system `U`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(bound = "")]
pub struct DiskGraphTTPlan<U: UnitSystem> {
    pub entity: EntitySpec<U>,
    pub temporal: Option<TemporalSpec<U>>,
    pub prediction_type: PredictionType,
    #[serde(default)]
    pub is_prediction: bool,
    #[serde(default)]
    pub split: Option<SplitSpec<U>>,
    pub stages: Vec<Stage<U>>,
    /// Semantic type of the target column (e.g. "numerical", "categorical",
    /// "boolean", "ID", "multicategorical"). Drives the choice of inline
    /// col-stats accumulators (Welford/TDigest vs category counts).
    #[serde(default)]
    pub target_stype: Option<String>,
    /// Optional source column (in the customer parquet for custom-TT path)
    /// to read per-row weights from. When `Some(col_name)`, the executor
    /// emits `weight.bin` next to `tt.parquet` with one f32 per emitted
    /// output row (NOT per input row — orphan rows whose entity PK isn't
    /// in the index are silently skipped, see
    /// `custom_tt_chunk_processor`). When `None`, no `weight.bin` is
    /// emitted and the manifest's `weight_sidecar` field is `None`.
    ///
    /// This field is set on custom-TT plans before serialization. The PQ
    /// path leaves it `None` (no source column, no weight emission).
    #[serde(default)]
    pub weight_col: Option<String>,

    /// Explicit mapping of columns in the seed table to their destination entity tables.
    #[serde(default)]
    pub fkey_mapping: Option<Vec<TTFKeySpec>>,

    /// LP resolution specs (`ENTITY1`/`TARGET1` → node-id rowpack columns).
    /// `None` for non-LP; ingest then synthesizes the scalar `ENTITY1` spec (see
    /// [`stage_resolution_specs`]).
    #[serde(default)]
    pub lp_resolution: Option<Vec<RefResolveSpec>>,

    /// Data-condition for TT-NG single-timestamp grouping, computed kumo-side:
    /// `true` iff this is an LP plan whose RHS/target node table has a
    /// create-time or end-time column. The `bucket_index` (and hence grouping)
    /// is built only when `lp_resolution.is_some() && lp_rhs_is_temporal`,
    /// mirroring legacy's `TimestampSampler` RHS check. `#[serde(default)]` →
    /// old plans (and non-LP plans) deserialize as `false` (flat).
    #[serde(default)]
    pub lp_rhs_is_temporal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(bound = "")]
pub struct SplitSpec<U: UnitSystem> {
    pub ratio: Option<Vec<f64>>,
    pub kind: Option<String>,
    pub rnd_seed: Option<u64>,
    pub is_temporal: Option<bool>,
    pub offsets: Option<Vec<Dur<U>>>,
    #[serde(default)]
    pub offset_unit: Option<String>,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub timeframe: Option<i64>,
    #[serde(default)]
    pub anchor_start_offset: Option<i64>,
    pub ranges: Option<Vec<(Abs<U>, Abs<U>)>>,
    pub backtesting_timestamp: Option<Abs<U>>,
    pub split_column: Option<String>,
    #[serde(default, alias = "split_projection")]
    pub split_output_column: Option<String>,
    pub manual_train_values: Option<Vec<String>>,
    pub manual_val_values: Option<Vec<String>>,
    pub manual_test_values: Option<Vec<String>>,
    /// Phase 13: calendar-month offsets for `DateOffsetSplit`. When present,
    /// these are passed through as `OffsetUnit::Months` (calendar arithmetic).
    /// Mutually exclusive with `offsets` for the same kind. `Some(vec![-3, -1])`
    /// means train ends 3 months before max_ts, val ends 1 month before. Year
    /// offsets should be expressed as months (year * 12) at construction time.
    #[serde(default)]
    pub month_offsets: Option<Vec<i32>>,
    /// Temporal-split boundary adjustment (kumo `flexible_boundaries`, prod
    /// default `True`). DateOffsetSplit uses this to snap each split start back
    /// to the sampling grid; TimeRangeSplit normally arrives with this disabled
    /// after kumo resolves the query-specific ranges.
    #[serde(default)]
    pub flexible_boundaries: Option<bool>,
    /// Temporal sampling step (kumo `timeframe_step`). DateOffsetSplit uses it
    /// as the grid spacing for flexible-boundary snapping.
    #[serde(default)]
    pub timeframe_step: Option<i64>,
}

// ─── Entity (FOR EACH) ──────────────────────────────────────────────────────


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TTFKeySpec {
    pub src_col: String,
    pub dst_table: String,
}

/// Arity of a [`RefResolveSpec`] reference column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefResolveKind {
    /// A single PK value → one `u64` node-id (e.g. the LHS `ENTITY1` anchor).
    Scalar,
    /// A list of PK values → a `list<i64>` of node-ids (e.g. the LP `TARGET1`
    /// positives).
    List,
}

/// One column in the FK-free LP "resolution mode": `src_col` holds PK
/// value(s) referencing `dst_table`'s primary key; the special tt-ingest
/// resolves them to graph node-ids via `dst_table`'s pkmap and writes the
/// result into the rowpack column named `node_id_col`. NO FK edge is emitted
/// for these columns (the engine reads the column directly), so there is no
/// `seed.edges.bin` and no message-passing leak.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RefResolveSpec {
    /// Source column in the stage parquet holding the PK reference value(s)
    /// (e.g. `ENTITY1`, `TARGET1`).
    pub src_col: String,
    /// Parent node table whose pkmap resolves `src_col`'s values to node-ids.
    pub dst_table: String,
    /// Scalar vs list arity of `src_col`.
    pub kind: RefResolveKind,
    /// Name of the NEW rowpack column the resolved node-id(s) are written to
    /// (e.g. `entity_node_id`, `target_node_ids`).
    pub node_id_col: String,
}

/// Resolution specs for stage ingest (seed→graph is a reference, never an FK
/// edge). LP plans carry them in `lp_resolution`; non-LP plans carry none, so
/// synthesize the scalar `ENTITY1 → entity_node_id` from `entity_table`. A
/// `fkey_mapping` naming `TARGET1` without `lp_resolution` is the removed
/// exploded-LP shape and is rejected.
pub fn stage_resolution_specs(
    lp_resolution: &Option<Vec<RefResolveSpec>>,
    fkey_mapping: &Option<Vec<TTFKeySpec>>,
    entity_table: &str,
) -> Result<Vec<RefResolveSpec>, String> {
    if let Some(specs) = lp_resolution {
        return Ok(specs.clone());
    }
    if let Some(mapping) = fkey_mapping {
        if mapping.iter().any(|f| f.src_col == PRIMARY_TARGET_COL) {
            return Err(format!(
                "stage ingest: fkey_mapping references `{PRIMARY_TARGET_COL}` with no \
                 `lp_resolution` — the legacy exploded-LP FK-edge shape is unsupported; \
                 an LP plan must declare `lp_resolution`"
            ));
        }
    }
    Ok(vec![RefResolveSpec {
        src_col: PRIMARY_ENTITY_COL.to_string(),
        dst_table: entity_table.to_string(),
        kind: RefResolveKind::Scalar,
        node_id_col: ENTITY_NODE_ID_COL.to_string(),
    }])
}

/// Which table/column to iterate over.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(bound = "")]
pub struct EntitySpec<U: UnitSystem> {
    pub table: String,
    pub column: String,
    /// Overrides the name of the output time column (defaults to "TIMESTAMP").
    #[serde(default)]
    pub output_time_col_name: Option<String>,
    /// The default timestamp value to emit for static queries (defaults to 0).
    #[serde(default)]
    pub default_time_value: Option<Abs<U>>,
    /// Entity create-time column for temporal filtering.
    pub time_col: Option<String>,
    /// Entity valid-to column for temporal filtering (seed entities need both
    /// `time_col` and `valid_to_col` read from rowpack since they are not
    /// visited via edges).
    pub valid_to_col: Option<String>,
    /// Row-level filter on entity table columns.
    pub filter: Option<RowFilter>,
}

// ─── Temporal config ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(bound = "")]
pub struct TemporalSpec<U: UnitSystem> {
    /// Anchor timestamps, already converted to the index unit system.
    pub timestamps: Vec<Abs<U>>,
    pub forecast_length: i64,
    pub lag_timesteps: Option<i64>,
}

// ─── Prediction type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionType {
    Node,
    Link {
        /// When `true`, include a FREQUENCY column counting occurrences of each
        /// distinct value. Set for RANK problems; unset for CLASSIFY.
        include_frequency: bool,
    },
    /// Emits only (ENTITY, TIMESTAMP) columns. Used for RHS candidate generation.
    PassThrough,
    /// Forecasting task: emits autoregression_labels and optionally year_over_year_labels.
    Forecasting,
}

// ─── Staged execution ────────────────────────────────────────────────────────

/// Ops + optional short-circuit filter. If filter fails, skip remaining stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(bound = "")]
pub struct Stage<U: UnitSystem> {
    pub ops: Vec<Op<U>>,
    pub filter: Option<StageFilter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(bound = "")]
pub enum Op<U: UnitSystem> {
    /// Traverse edges, apply filters, accumulate.
    Aggregate {
        alias: String,
        /// Single-hop: one `EdgeRef`. Multi-hop: `[A→B, B→C]`.
        path: Vec<EdgeRef>,
        func: AggFunc,
        /// Column to aggregate. `None` = `COUNT(*)`.
        target_col: Option<String>,
        /// Column for temporal window filtering on the far-end table.
        time_col: Option<String>,
        /// `(lo, hi)` in `window_unit`. Resolved to `Dur<U>` during plan resolution.
        window: Option<(i64, i64)>,
        window_unit: Option<TimeUnit>,
        row_filter: Option<RowFilter>,
        /// Early-termination limit (PQL `TOP k`). The executor may stop
        /// accumulating after `limit` items, primarily useful for `ListDistinct`.
        limit: Option<u32>,
        /// For `ListDistinct` in link prediction: the *target entity*
        /// table — the table whose rows enumerate the items being
        /// predicted. Used by the ttgen executor to drop values not
        /// present in that table's PK map (orphan filter, matching
        /// GraphEngine's inner-join label semantics).
        ///
        /// When `Some`, filter is enabled. When `None`, no filtering is
        /// applied. Ignored for every other aggregation func.
        ///
        /// Resolution rules (performed by the caller, *not* inferred):
        ///   * If the leaf column is a foreign key → the FK destination.
        ///   * If the leaf column is the primary key of its own table →
        ///     that same table.
        ///   * Otherwise → `None`.
        target_entity_table: Option<String>,
    },
    /// Direct column lookup, optionally after traversing a validated relationship path.
    Column {
        alias: String,
        table: String,
        column: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        path: Vec<EdgeRef>,
    },
    /// Boolean comparison on a previously computed alias.
    Compare {
        alias: String,
        lhs: String,
        op: CompareOp,
        rhs: Value,
    },
    /// Logical combination of previously computed boolean aliases.
    LogicalOp {
        alias: String,
        op: LogicalOpKind,
        /// For `Not`: single operand. For `And`/`Or`: two operands.
        operands: Vec<String>,
    },
    /// Evaluate a target operation repeatedly across historical lag offsets.
    Autoregression {
        alias: String,
        /// The inner operation to evaluate (typically an Aggregate).
        target_op: Box<Op<U>>,
        /// Lag offsets applied to the anchor timestamp.
        lag_offsets: Vec<Dur<U>>,
    },
    /// Evaluate supervised forecasting labels across horizon offsets.
    FutureLabels {
        alias: String,
        target_op: Box<Op<U>>,
        future_offsets: Vec<Dur<U>>,
    },
    /// Fast boolean check: Returns true if any edge satisfies the path and filters.
    HasEdges {
        alias: String,
        path: Vec<EdgeRef>,
        time_col: Option<String>,
        /// `(lo, hi)` in `window_unit`. Resolved to `Dur<U>` during plan resolution.
        window: Option<(i64, i64)>,
        window_unit: Option<TimeUnit>,
        row_filter: Option<RowFilter>,
    },
}

/// Kind of logical operation on boolean values.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogicalOpKind {
    And,
    Or,
    Not,
}

/// Directed foreign-key reference between two tables.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct EdgeRef {
    pub src_table: String,
    pub src_col: String,
    pub dst_table: String,
    pub dst_col: String,
}

// ─── Enums ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    CountDistinct,
    ListDistinct,
    First,
    Last,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    IsNull,
    IsNotNull,
    StartsWith,
    EndsWith,
    Contains,
    NotContains,
    In,
}

/// Dynamically-typed scalar for filter comparisons.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    List(Vec<Value>),
}

// ─── Filter LHS / RHS types ─────────────────────────────────────────────────

/// Where the left-hand value comes from in a [`RowFilter`] comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowFilterLhs {
    /// Column on the table currently being traversed.
    Column(String),
    /// Column on a table reachable via sub-traversal from the current row
    /// (multi-hop filtering).
    Remote { path: Vec<EdgeRef>, column: String },
}

/// Right-hand side of a [`RowFilter`] comparison.
///
/// Does not include `Alias` — aliases are computed by ops and do not exist
/// during per-row traversal. If a comparison involves a computed alias, it
/// belongs in a [`StageFilter`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowFilterRhs {
    /// Literal constant.
    Constant(Value),
    /// Column on the entity table (resolved per-entity before traversal).
    EntityColumn(String),
    /// Column on a table reachable via sub-traversal from the current row.
    Remote { path: Vec<EdgeRef>, column: String },
}

/// Right-hand side of a [`StageFilter`] comparison.
///
/// Extends [`RowFilterRhs`] with `Alias` for comparing two computed values
/// (e.g. `WHERE SUM(...) > AVG(...)`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageFilterRhs {
    /// Literal constant.
    Constant(Value),
    /// Column on the entity table (resolved per-entity at runtime).
    EntityColumn(String),
    /// Column on a table reachable via sub-traversal from the entity.
    Remote { path: Vec<EdgeRef>, column: String },
    /// Alias of a previously computed op in a prior stage.
    Alias(String),
}

// ─── Filters ─────────────────────────────────────────────────────────────────

/// Row-level filter evaluated per-row during edge traversal.
///
/// LHS is parameterised by [`RowFilterLhs`] (local column or remote
/// sub-traversal). RHS is parameterised by [`RowFilterRhs`] (constant,
/// entity column, or remote sub-traversal).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowFilter {
    Compare {
        lhs: RowFilterLhs,
        op: CompareOp,
        rhs: RowFilterRhs,
    },
    And(Vec<RowFilter>),
    Or(Vec<RowFilter>),
    Not(Box<RowFilter>),
}

/// Post-stage filter evaluated per-entity between stages.
///
/// LHS is always a computed alias (from a prior op). RHS is parameterised
/// by [`StageFilterRhs`] (constant, entity column, remote, or another alias).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageFilter {
    Compare {
        alias: String,
        op: CompareOp,
        rhs: StageFilterRhs,
    },
    /// Require a computed alias to be a present, non-empty [`Value::List`].
    NonEmptyList {
        alias: String,
    },
    And(Vec<StageFilter>),
    Or(Vec<StageFilter>),
    Not(Box<StageFilter>),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Seconds;

    /// Seals the kumo<->DG contract: the EXACT JSON `_apply_lp_resolution_plan`
    /// emits for `plan['lp_resolution']` must deserialize into the Rust
    /// `Vec<RefResolveSpec>` (field names + the snake_case Scalar/List enum).
    #[test]
    fn lp_resolution_round_trips_kumo_emitted_json() {
        let json = r#"[
            {"src_col":"ENTITY1","dst_table":"users","kind":"scalar","node_id_col":"entity_node_id"},
            {"src_col":"TARGET1","dst_table":"products","kind":"list","node_id_col":"target_node_ids"}
        ]"#;
        let specs: Vec<RefResolveSpec> =
            serde_json::from_str(json).expect("kumo lp_resolution must deserialize into RefResolveSpec");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].src_col, "ENTITY1");
        assert_eq!(specs[0].dst_table, "users");
        assert_eq!(specs[0].kind, RefResolveKind::Scalar);
        assert_eq!(specs[0].node_id_col, "entity_node_id");
        assert_eq!(specs[1].src_col, "TARGET1");
        assert_eq!(specs[1].kind, RefResolveKind::List);
        assert_eq!(specs[1].node_id_col, "target_node_ids");
    }

    #[test]
    fn non_empty_list_stage_filter_round_trips_expected_json() {
        let filter = StageFilter::NonEmptyList {
            alias: "labels".to_string(),
        };
        let json = serde_json::to_value(&filter).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"non_empty_list": {"alias": "labels"}})
        );

        let decoded: StageFilter = serde_json::from_value(json).unwrap();
        assert!(matches!(
            decoded,
            StageFilter::NonEmptyList { alias } if alias == "labels"
        ));
    }

    /// Verify that a `DiskGraphTTPlan<Seconds>` serialized as JSON produces
    /// the same bytes as the old non-generic plan (plain i64 values).
    #[test]
    fn plan_serde_backward_compat() {
        let plan = DiskGraphTTPlan::<Seconds> {
            entity: EntitySpec {
                table: "USERS".into(),
                column: "USER_ID".into(),
                output_time_col_name: None,
                default_time_value: Some(Abs::from_secs(0)),
                time_col: None,
                valid_to_col: None,
                filter: None,
            },
            temporal: Some(TemporalSpec {
                timestamps: vec![
                    Abs::from_secs(86400),
                    Abs::from_secs(172800),
                ],
                forecast_length: 1,
                lag_timesteps: None,
            }),
            prediction_type: PredictionType::Node,
            is_prediction: false,
            split: Some(SplitSpec {
                ratio: Some(vec![0.8, 0.2]),
                kind: None,
                rnd_seed: None,
                is_temporal: Some(true),
                offsets: Some(vec![Dur::from_days(-7), Dur::from_days(-3)]),
                offset_unit: None,
                unit: None,
                timeframe: None,
                anchor_start_offset: None,
                ranges: None,
                backtesting_timestamp: None,
                split_column: None,
                split_output_column: None,
                manual_train_values: None,
                manual_val_values: None,
                manual_test_values: None,
                month_offsets: None,
                flexible_boundaries: None,
                timeframe_step: None,
            }),
            stages: vec![],
            target_stype: None,
            weight_col: None,
            fkey_mapping: None,
            lp_resolution: None,
            lp_rhs_is_temporal: false,
        };

        let json = serde_json::to_string(&plan).unwrap();

        // Deserialize back
        let back: DiskGraphTTPlan<Seconds> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.temporal.as_ref().unwrap().timestamps.len(), 2);
        assert_eq!(back.temporal.as_ref().unwrap().timestamps[0].raw(), 86400);
        assert_eq!(back.entity.default_time_value.unwrap().raw(), 0);
        assert_eq!(
            back.split.as_ref().unwrap().offsets.as_ref().unwrap()[0].raw(),
            -7 * 86400
        );
    }

    /// Verify that JSON with raw i64 timestamps (from older serializations)
    /// deserializes into the generic types correctly.
    #[test]
    fn plan_deserialize_from_raw_i64_json() {
        let json = r#"{
            "entity": {"table": "T", "column": "C"},
            "temporal": {"timestamps": [100, 200], "forecast_length": 1},
            "prediction_type": "node",
            "stages": []
        }"#;

        let plan: DiskGraphTTPlan<Seconds> = serde_json::from_str(json).unwrap();
        assert_eq!(plan.temporal.as_ref().unwrap().timestamps[0].raw(), 100);
        assert_eq!(plan.temporal.as_ref().unwrap().timestamps[1].raw(), 200);
    }

    /// Verify autoregression with typed Dur offsets.
    #[test]
    fn op_autoregression_serde() {
        let op = Op::<Seconds>::Autoregression {
            alias: "ar".into(),
            target_op: Box::new(Op::Column {
                alias: "col".into(),
                table: "T".into(),
                column: "C".into(),
                path: vec![],
            }),
            lag_offsets: vec![Dur::from_secs(-86400), Dur::from_secs(-172800)],
        };

        let json = serde_json::to_string(&op).unwrap();
        let back: Op<Seconds> = serde_json::from_str(&json).unwrap();
        if let Op::Autoregression { lag_offsets, .. } = back {
            assert_eq!(lag_offsets[0].raw(), -86400);
            assert_eq!(lag_offsets[1].raw(), -172800);
        } else {
            panic!("expected Autoregression");
        }
    }

    #[test]
    fn op_future_labels_serde() {
        let op = Op::<Seconds>::FutureLabels {
            alias: "future".into(),
            target_op: Box::new(Op::Column {
                alias: "col".into(),
                table: "T".into(),
                column: "C".into(),
                path: vec![],
            }),
            future_offsets: vec![Dur::from_secs(0), Dur::from_secs(86400)],
        };

        let json = serde_json::to_string(&op).unwrap();
        let back: Op<Seconds> = serde_json::from_str(&json).unwrap();
        if let Op::FutureLabels { future_offsets, .. } = back {
            assert_eq!(future_offsets[0].raw(), 0);
            assert_eq!(future_offsets[1].raw(), 86400);
        } else {
            panic!("expected FutureLabels");
        }
    }
}
