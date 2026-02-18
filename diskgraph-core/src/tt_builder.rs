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
//! - [`RowFilter`]: evaluated during edge traversal on table columns (SQL `WHERE`).
//! - [`StageFilter`]: evaluated after ops complete on computed aliases (SQL `HAVING`).

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
    /// Entity-level temporal filtering (skip entities where `time_col > timestamp`).
    pub time_col: Option<String>,
    /// Row-level filter on entity table columns.
    pub filter: Option<RowFilter>,
}

// ─── Temporal config ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSpec {
    pub timestamps: Vec<TimestampEntry>,
    pub forecast_length: i64,
    pub lag_timesteps: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampEntry {
    /// Epoch microseconds, inclusive.
    pub start: i64,
    /// Epoch microseconds, exclusive.
    pub end: i64,
    pub split: SplitLabel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SplitLabel {
    Train,
    Val,
    Test,
}

// ─── Prediction type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionType {
    Node,
    Link {
        target_entity_table: String,
        target_entity_col: String,
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

// ─── Filters ─────────────────────────────────────────────────────────────────

/// Post-stage filter on computed aliases (SQL `HAVING`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageFilter {
    Compare {
        alias: String,
        op: CompareOp,
        value: Value,
    },
    And(Vec<StageFilter>),
    Or(Vec<StageFilter>),
    Not(Box<StageFilter>),
}

/// Row-level filter on table columns during edge traversal (SQL `WHERE`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RowFilter {
    Compare {
        column: String,
        op: CompareOp,
        value: Value,
    },
    And(Vec<RowFilter>),
    Or(Vec<RowFilter>),
    Not(Box<RowFilter>),
}
