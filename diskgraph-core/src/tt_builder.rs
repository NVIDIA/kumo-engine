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

use serde::{Deserialize, Serialize};

// ─── Top-level plan ──────────────────────────────────────────────────────────

/// A fully-specified training-table execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskGraphTTPlan {
    pub entity: EntitySpec,
    pub temporal: Option<TemporalSpec>,
    pub prediction_type: PredictionType,
    pub stages: Vec<Stage>,
}

// ─── Entity (FOR EACH) ──────────────────────────────────────────────────────

/// Which table/column to iterate over.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySpec {
    pub table: String,
    pub column: String,
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
pub struct TemporalSpec {
    /// Epoch microseconds for each anchor time.
    pub timestamps: Vec<i64>,
    pub forecast_length: i64,
    pub lag_timesteps: Option<i64>,
}

// ─── Prediction type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionType {
    Node,
    Link {
        /// When `true`, each list element becomes a separate output row.
        explode: bool,
    },
}

// ─── Staged execution ────────────────────────────────────────────────────────

/// Ops + optional short-circuit filter. If filter fails, skip remaining stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    pub ops: Vec<Op>,
    pub filter: Option<StageFilter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
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
        /// `(lo, hi)` relative to the current timestamp.
        window: Option<(i64, i64)>,
        window_unit: Option<TimeUnit>,
        row_filter: Option<RowFilter>,
        /// Early-termination limit (PQL `TOP k`). The executor may stop
        /// accumulating after `limit` items, primarily useful for `ListDistinct`.
        limit: Option<u32>,
    },
    /// Direct column lookup (no traversal).
    Column {
        alias: String,
        table: String,
        column: String,
    },
    /// Boolean comparison on a previously computed alias.
    Compare {
        alias: String,
        lhs: String,
        op: CompareOp,
        rhs: Value,
    },
}

/// Directed foreign-key reference between two tables.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EdgeRef {
    pub src_table: String,
    pub src_col: String,
    pub dst_table: String,
    pub dst_col: String,
}

// ─── Enums ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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
pub enum CompareOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

/// Dynamically-typed scalar for filter comparisons.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

// ─── Filter LHS / RHS types ─────────────────────────────────────────────────

/// Where the left-hand value comes from in a [`RowFilter`] comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RowFilterLhs {
    /// Column on the table currently being traversed.
    Column(String),
    /// Column on a table reachable via sub-traversal from the current row
    /// (multi-hop filtering).
    Remote {
        path: Vec<EdgeRef>,
        column: String,
    },
}

/// Right-hand side of a [`RowFilter`] comparison.
///
/// Does not include `Alias` — aliases are computed by ops and do not exist
/// during per-row traversal. If a comparison involves a computed alias, it
/// belongs in a [`StageFilter`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RowFilterRhs {
    /// Literal constant.
    Constant(Value),
    /// Column on the entity table (resolved per-entity before traversal).
    EntityColumn(String),
    /// Column on a table reachable via sub-traversal from the current row.
    Remote {
        path: Vec<EdgeRef>,
        column: String,
    },
}

/// Right-hand side of a [`StageFilter`] comparison.
///
/// Extends [`RowFilterRhs`] with `Alias` for comparing two computed values
/// (e.g. `WHERE SUM(...) > AVG(...)`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageFilterRhs {
    /// Literal constant.
    Constant(Value),
    /// Column on the entity table (resolved per-entity at runtime).
    EntityColumn(String),
    /// Column on a table reachable via sub-traversal from the entity.
    Remote {
        path: Vec<EdgeRef>,
        column: String,
    },
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
pub enum StageFilter {
    Compare {
        alias: String,
        op: CompareOp,
        rhs: StageFilterRhs,
    },
    And(Vec<StageFilter>),
    Or(Vec<StageFilter>),
    Not(Box<StageFilter>),
}
