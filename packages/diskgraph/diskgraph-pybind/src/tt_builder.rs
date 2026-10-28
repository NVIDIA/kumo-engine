// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for [`DiskGraphTTBuilder`].
//!
//! ```python
//! from diskgraph_pybind import DiskGraphTTBuilder, EdgeRef, RowFilter, StageFilter
//!
//! edge = EdgeRef("users", "user_id", "transactions", "user_id")
//! b = DiskGraphTTBuilder()
//! b.for_each("users", "user_id")
//! b.aggregate("target", edge, "sum", target_col="amount")
//! plan_json = b.build()
//! ```

use pyo3::prelude::*;
use pyo3::types::PyList;

use diskgraph_core::time::{
    abs_from_unit, ceil_parts, convert_trunc, dur_from_unit, split_parts, Abs, Nanoseconds,
    TimeUnit,
};
use diskgraph_core::tt_builder as tt;
use diskgraph_core::with_index_clock;

// ─── EdgeRef ─────────────────────────────────────────────────────────────────

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct EdgeRef {
    pub(crate) inner: tt::EdgeRef,
}

#[pymethods]
impl EdgeRef {
    #[new]
    fn new(src_table: String, src_col: String, dst_table: String, dst_col: String) -> Self {
        Self {
            inner: tt::EdgeRef {
                src_table,
                src_col,
                dst_table,
                dst_col,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EdgeRef({}.{} -> {}.{})",
            self.inner.src_table, self.inner.src_col, self.inner.dst_table, self.inner.dst_col,
        )
    }
}

// ─── RowFilter ───────────────────────────────────────────────────────────────

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct RowFilter {
    pub(crate) inner: tt::RowFilter,
}

#[pymethods]
impl RowFilter {
    /// Column vs constant (today's common case).
    ///
    /// ```python
    /// RowFilter.compare("status", "==", "paid")
    /// ```
    #[staticmethod]
    #[pyo3(signature = (column, op, value, timestamp_unit="seconds"))]
    fn compare(
        column: String,
        op: &str,
        value: &Bound<'_, PyAny>,
        timestamp_unit: &str,
    ) -> PyResult<Self> {
        let clock = parse_index_clock(timestamp_unit)?;
        Ok(Self {
            inner: tt::RowFilter::Compare {
                lhs: tt::RowFilterLhs::Column(column),
                op: parse_compare_op(op)?,
                rhs: tt::RowFilterRhs::Constant(extract_value(value, clock)?),
            },
        })
    }

    /// Column vs entity column (entity=target comparison).
    ///
    /// ```python
    /// RowFilter.compare_to_entity("currency", "==", "pref_currency")
    /// ```
    #[staticmethod]
    fn compare_to_entity(column: String, op: &str, entity_column: String) -> PyResult<Self> {
        Ok(Self {
            inner: tt::RowFilter::Compare {
                lhs: tt::RowFilterLhs::Column(column),
                op: parse_compare_op(op)?,
                rhs: tt::RowFilterRhs::EntityColumn(entity_column),
            },
        })
    }

    /// Remote column (via sub-traversal) vs constant (multi-hop filtering).
    ///
    /// ```python
    /// RowFilter.compare_remote_lhs(order_to_merchant, "country", "==", "US")
    /// ```
    #[staticmethod]
    #[pyo3(signature = (path, column, op, value, timestamp_unit="seconds"))]
    fn compare_remote_lhs(
        path: &Bound<'_, PyAny>,
        column: String,
        op: &str,
        value: &Bound<'_, PyAny>,
        timestamp_unit: &str,
    ) -> PyResult<Self> {
        let clock = parse_index_clock(timestamp_unit)?;
        Ok(Self {
            inner: tt::RowFilter::Compare {
                lhs: tt::RowFilterLhs::Remote {
                    path: extract_edge_path(path)?,
                    column,
                },
                op: parse_compare_op(op)?,
                rhs: tt::RowFilterRhs::Constant(extract_value(value, clock)?),
            },
        })
    }

    /// Column vs remote column (sub-traversal on RHS).
    ///
    /// ```python
    /// RowFilter.compare_remote_rhs("currency", "==", order_to_merchant, "default_currency")
    /// ```
    #[staticmethod]
    fn compare_remote_rhs(
        column: String,
        op: &str,
        rhs_path: &Bound<'_, PyAny>,
        rhs_column: String,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: tt::RowFilter::Compare {
                lhs: tt::RowFilterLhs::Column(column),
                op: parse_compare_op(op)?,
                rhs: tt::RowFilterRhs::Remote {
                    path: extract_edge_path(rhs_path)?,
                    column: rhs_column,
                },
            },
        })
    }

    /// Remote column vs entity column.
    ///
    /// ```python
    /// RowFilter.compare_remote_to_entity(
    ///     order_to_merchant, "country", "==", "country",
    /// )
    /// ```
    #[staticmethod]
    fn compare_remote_to_entity(
        path: &Bound<'_, PyAny>,
        column: String,
        op: &str,
        entity_column: String,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: tt::RowFilter::Compare {
                lhs: tt::RowFilterLhs::Remote {
                    path: extract_edge_path(path)?,
                    column,
                },
                op: parse_compare_op(op)?,
                rhs: tt::RowFilterRhs::EntityColumn(entity_column),
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (*filters))]
    fn and_(filters: Vec<RowFilter>) -> Self {
        Self {
            inner: tt::RowFilter::And(filters.into_iter().map(|f| f.inner).collect()),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*filters))]
    fn or_(filters: Vec<RowFilter>) -> Self {
        Self {
            inner: tt::RowFilter::Or(filters.into_iter().map(|f| f.inner).collect()),
        }
    }

    #[staticmethod]
    fn not_(filter: RowFilter) -> Self {
        Self {
            inner: tt::RowFilter::Not(Box::new(filter.inner)),
        }
    }
}

// ─── StageFilter ─────────────────────────────────────────────────────────────

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct StageFilter {
    pub(crate) inner: tt::StageFilter,
}

#[pymethods]
impl StageFilter {
    /// Alias vs constant (today's common case).
    ///
    /// ```python
    /// StageFilter.compare("_sum", ">", 0)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (alias, op, value, timestamp_unit="seconds"))]
    fn compare(
        alias: String,
        op: &str,
        value: &Bound<'_, PyAny>,
        timestamp_unit: &str,
    ) -> PyResult<Self> {
        let clock = parse_index_clock(timestamp_unit)?;
        Ok(Self {
            inner: tt::StageFilter::Compare {
                alias,
                op: parse_compare_op(op)?,
                rhs: tt::StageFilterRhs::Constant(extract_value(value, clock)?),
            },
        })
    }

    /// Alias vs entity column.
    ///
    /// ```python
    /// StageFilter.compare_to_entity("_sum", ">", "credit_limit")
    /// ```
    #[staticmethod]
    fn compare_to_entity(alias: String, op: &str, entity_column: String) -> PyResult<Self> {
        Ok(Self {
            inner: tt::StageFilter::Compare {
                alias,
                op: parse_compare_op(op)?,
                rhs: tt::StageFilterRhs::EntityColumn(entity_column),
            },
        })
    }

    /// Alias vs remote column (sub-traversal from entity).
    ///
    /// ```python
    /// StageFilter.compare_to_remote("_sum", ">", user_to_company, "spending_limit")
    /// ```
    #[staticmethod]
    fn compare_to_remote(
        alias: String,
        op: &str,
        path: &Bound<'_, PyAny>,
        column: String,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: tt::StageFilter::Compare {
                alias,
                op: parse_compare_op(op)?,
                rhs: tt::StageFilterRhs::Remote {
                    path: extract_edge_path(path)?,
                    column,
                },
            },
        })
    }

    /// Alias vs another alias.
    ///
    /// ```python
    /// StageFilter.compare_to_alias("_sum", ">", "_avg")
    /// ```
    #[staticmethod]
    fn compare_to_alias(alias: String, op: &str, rhs_alias: String) -> PyResult<Self> {
        Ok(Self {
            inner: tt::StageFilter::Compare {
                alias,
                op: parse_compare_op(op)?,
                rhs: tt::StageFilterRhs::Alias(rhs_alias),
            },
        })
    }

    /// Require an alias to contain a non-empty list.
    ///
    /// ```python
    /// StageFilter.non_empty_list("_labels")
    /// ```
    #[staticmethod]
    fn non_empty_list(alias: String) -> Self {
        Self {
            inner: tt::StageFilter::NonEmptyList { alias },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*filters))]
    fn and_(filters: Vec<StageFilter>) -> Self {
        Self {
            inner: tt::StageFilter::And(filters.into_iter().map(|f| f.inner).collect()),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*filters))]
    fn or_(filters: Vec<StageFilter>) -> Self {
        Self {
            inner: tt::StageFilter::Or(filters.into_iter().map(|f| f.inner).collect()),
        }
    }

    #[staticmethod]
    fn not_(filter: StageFilter) -> Self {
        Self {
            inner: tt::StageFilter::Not(Box::new(filter.inner)),
        }
    }
}

// ─── DiskGraphTTBuilder ──────────────────────────────────────────────────────

/// Holds plan times in nanoseconds; `build()` writes the plan in `timestamp_unit`.
#[pyclass]
pub struct DiskGraphTTBuilder {
    timestamp_unit: TimeUnit,
    entity: Option<tt::EntitySpec<Nanoseconds>>,
    default_time_secs: Option<i64>,
    temporal: Option<tt::TemporalSpec<Nanoseconds>>,
    prediction_type: tt::PredictionType,
    is_prediction: bool,
    split: Option<tt::SplitSpec<Nanoseconds>>,
    stages: Vec<tt::Stage<Nanoseconds>>,
    current_ops: Vec<tt::Op<Nanoseconds>>,
    target_stype: Option<String>,
    fkey_mapping: Option<Vec<tt::TTFKeySpec>>,
}

#[pymethods]
impl DiskGraphTTBuilder {
    #[new]
    #[pyo3(signature = (timestamp_unit="seconds"))]
    fn new(timestamp_unit: &str) -> PyResult<Self> {
        Ok(Self {
            timestamp_unit: parse_index_clock(timestamp_unit)?,
            entity: None,
            default_time_secs: None,
            temporal: None,
            prediction_type: tt::PredictionType::Node,
            is_prediction: false,
            split: None,
            stages: Vec::new(),
            current_ops: Vec::new(),
            target_stype: None,
            fkey_mapping: None,
        })
    }

    /// `FOR EACH {table}.{column}`; `default_time_value` is in epoch seconds.
    #[pyo3(signature = (table, column, output_time_col_name=None, default_time_value=None))]
    fn for_each(
        &mut self,
        table: String,
        column: String,
        output_time_col_name: Option<String>,
        default_time_value: Option<i64>,
    ) {
        self.entity = Some(tt::EntitySpec {
            table,
            column,
            output_time_col_name,
            default_time_value: None,
            time_col: None,
            valid_to_col: None,
            filter: None,
        });
        self.default_time_secs = default_time_value;
    }

    /// Entity-level create-time column for temporal filtering.
    fn entity_time_col(&mut self, column: String) -> PyResult<()> {
        match &mut self.entity {
            Some(e) => {
                e.time_col = Some(column);
                Ok(())
            }
            None => Err(crate::DiskgraphConfigError::new_err(
                "call for_each() before entity_time_col()",
            )),
        }
    }

    /// Entity-level valid-to column for temporal filtering.
    fn entity_valid_to_col(&mut self, column: String) -> PyResult<()> {
        match &mut self.entity {
            Some(e) => {
                e.valid_to_col = Some(column);
                Ok(())
            }
            None => Err(crate::DiskgraphConfigError::new_err(
                "call for_each() before entity_valid_to_col()",
            )),
        }
    }

    /// Row-level filter on entity table.
    fn entity_filter(&mut self, f: RowFilter) -> PyResult<()> {
        match &mut self.entity {
            Some(e) => {
                e.filter = Some(f.inner);
                Ok(())
            }
            None => Err(crate::DiskgraphConfigError::new_err(
                "call for_each() before entity_filter()",
            )),
        }
    }

    /// Set anchor timestamps. `unit` declares the source unit of the raw i64
    /// values (e.g. "seconds", "nanoseconds"). Rust converts to the index
    /// unit system internally.
    #[pyo3(signature = (timestamps, forecast_length, lag_timesteps=None, unit="seconds"))]
    fn temporal(
        &mut self,
        timestamps: Vec<i64>,
        forecast_length: i64,
        lag_timesteps: Option<i64>,
        unit: &str,
    ) -> PyResult<()> {
        let time_unit = parse_time_unit(unit)?;
        self.temporal = Some(tt::TemporalSpec {
            timestamps: timestamps
                .into_iter()
                .map(|v| abs_from_unit::<Nanoseconds>(v, time_unit))
                .collect(),
            forecast_length,
            lag_timesteps,
        });
        Ok(())
    }

    /// Node-level prediction (default).
    fn node_prediction(&mut self) {
        self.prediction_type = tt::PredictionType::Node;
    }

    /// Set the plan's split from a Python `SplitSpec` (the dataclass in
    /// `diskgraph/split_contract.py`, or anything carrying the same
    /// attributes).
    ///
    /// The unit rules live in [`make_split_spec`], shared with
    /// [`build_split_spec_json`] -- see there for why they are asymmetric.
    fn split_ratio(&mut self, spec: &Bound<'_, PyAny>) -> PyResult<()> {
        self.split = Some(make_split_spec(extract_split_spec(spec)?)?);
        Ok(())
    }

    /// Link prediction. FK-free array LP is the only Link output mode.
    #[pyo3(signature = (include_frequency=false))]
    fn link_prediction(&mut self, include_frequency: bool) {
        self.prediction_type = tt::PredictionType::Link { include_frequency };
    }

    /// Pass-through prediction (LHS/RHS extraction without target ops).
    fn pass_through_prediction(&mut self) {
        self.prediction_type = tt::PredictionType::PassThrough;
    }

    /// Forecasting prediction (outputs Autoregression and YoY columns).
    fn forecasting_prediction(&mut self) {
        self.prediction_type = tt::PredictionType::Forecasting;
    }

    /// Set whether this is a prediction job (Batch Prediction) or training job (TT generation).
    fn set_is_prediction(&mut self, is_prediction: bool) {
        self.is_prediction = is_prediction;
    }

    /// Set the semantic type of the target column (e.g. "numerical",
    /// "categorical", "boolean", "ID", "multicategorical").
    fn set_target_stype(&mut self, stype: String) {
        self.target_stype = Some(stype);
    }

    fn set_fkey_mapping(&mut self, mapping: Vec<(String, String)>) {
        self.fkey_mapping = Some(
            mapping
                .into_iter()
                .map(|(src_col, dst_table)| tt::TTFKeySpec {
                    src_col,
                    dst_table,
                })
                .collect(),
        );
    }

    /// Add aggregate op. `path`: single `EdgeRef` or list for multi-hop.
    /// `limit`: early-termination hint (PQL `TOP k`), primarily for `ListDistinct`.
    /// `target_entity_table`: for LP `ListDistinct`, the target entity table
    /// whose PK map is checked for orphan filtering. Resolved by the caller
    /// using the graph schema. `None` disables the filter.
    #[pyo3(signature = (alias, path, func, target_col=None, time_col=None, window=None, window_unit=None, row_filter=None, limit=None, target_entity_table=None))]
    #[allow(clippy::too_many_arguments)]
    fn aggregate(
        &mut self,
        alias: String,
        path: &Bound<'_, PyAny>,
        func: &str,
        target_col: Option<String>,
        time_col: Option<String>,
        window: Option<(i64, i64)>,
        window_unit: Option<&str>,
        row_filter: Option<RowFilter>,
        limit: Option<u32>,
        target_entity_table: Option<String>,
    ) -> PyResult<()> {
        let edge_path = extract_edge_path(path)?;
        let agg_func = parse_agg_func(func)?;
        let unit = window_unit.map(parse_time_unit).transpose()?;

        self.current_ops.push(tt::Op::Aggregate {
            alias,
            path: edge_path,
            func: agg_func,
            target_col,
            time_col,
            window,
            window_unit: unit,
            row_filter: row_filter.map(|f| f.inner),
            limit,
            target_entity_table,
        });
        Ok(())
    }

    /// Direct column lookup, optionally after traversing a validated relationship path.
    #[pyo3(signature = (alias, table, column, path=None))]
    fn column(
        &mut self,
        alias: String,
        table: String,
        column: String,
        path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let path = path.map(extract_edge_path).transpose()?.unwrap_or_default();
        self.current_ops.push(tt::Op::Column {
            alias,
            table,
            column,
            path,
        });
        Ok(())
    }

    /// Boolean comparison on a prior alias.
    fn compare(
        &mut self,
        alias: String,
        lhs_alias: String,
        op: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.current_ops.push(tt::Op::Compare {
            alias,
            lhs: lhs_alias,
            op: parse_compare_op(op)?,
            rhs: extract_value(value, self.timestamp_unit)?,
        });
        Ok(())
    }

    /// Logical combination of previously computed boolean aliases.
    pub fn logical_op(&mut self, alias: String, op: String, operands: Vec<String>) -> PyResult<()> {
        let op_kind = match op.as_str() {
            "and" => tt::LogicalOpKind::And,
            "or" => tt::LogicalOpKind::Or,
            "not" => tt::LogicalOpKind::Not,
            _ => {
                return Err(crate::DiskgraphConfigError::new_err(format!(
                    "Invalid logical op: {}",
                    op
                )))
            }
        };

        self.current_ops.push(tt::Op::LogicalOp {
            alias,
            op: op_kind,
            operands,
        });
        Ok(())
    }

    /// `lag_offset_unit` declares the unit for `lag_offsets` (default "seconds").
    #[pyo3(signature = (alias, target_op_json, lag_offsets, lag_offset_unit="seconds"))]
    pub fn autoregression(
        &mut self,
        alias: String,
        target_op_json: String,
        lag_offsets: Vec<i64>,
        lag_offset_unit: &str,
    ) -> PyResult<()> {
        let target_op: tt::Op<Nanoseconds> =
            serde_json::from_str(&target_op_json).map_err(|e| {
                crate::DiskgraphConfigError::new_err(format!("Invalid target op JSON: {}", e))
            })?;

        let unit = parse_time_unit(lag_offset_unit)?;
        self.current_ops.push(tt::Op::Autoregression {
            alias,
            target_op: Box::new(target_op),
            lag_offsets: lag_offsets
                .into_iter()
                .map(|v| dur_from_unit::<Nanoseconds>(v, unit))
                .collect(),
        });
        Ok(())
    }

    #[pyo3(signature = (alias, target_op_json, future_offsets, future_offset_unit="seconds"))]
    pub fn future_labels(
        &mut self,
        alias: String,
        target_op_json: String,
        future_offsets: Vec<i64>,
        future_offset_unit: &str,
    ) -> PyResult<()> {
        let target_op: tt::Op<Nanoseconds> =
            serde_json::from_str(&target_op_json).map_err(|e| {
                crate::DiskgraphConfigError::new_err(format!("Invalid target op JSON: {}", e))
            })?;

        let unit = parse_time_unit(future_offset_unit)?;
        self.current_ops.push(tt::Op::FutureLabels {
            alias,
            target_op: Box::new(target_op),
            future_offsets: future_offsets
                .into_iter()
                .map(|v| dur_from_unit::<Nanoseconds>(v, unit))
                .collect(),
        });
        Ok(())
    }

    #[pyo3(signature = (alias, path, time_col=None, window=None, window_unit=None, row_filter=None))]
    pub fn has_edges(
        &mut self,
        alias: String,
        path: &Bound<'_, PyAny>,
        time_col: Option<String>,
        window: Option<(i64, i64)>,
        window_unit: Option<&str>,
        row_filter: Option<RowFilter>,
    ) -> PyResult<()> {
        let path = extract_edge_path(path)?;
        let unit = window_unit.map(parse_time_unit).transpose()?;
        self.current_ops.push(tt::Op::HasEdges {
            alias,
            path,
            time_col,
            window,
            window_unit: unit,
            row_filter: row_filter.map(|f| f.inner.clone()),
        });
        Ok(())
    }

    /// Pops the last operation off the current stage and returns it as a JSON string.
    /// This is useful for wrapping existing operations (like Aggregates) inside other operations (like Autoregression).
    pub fn pop_last_op_json(&mut self) -> PyResult<String> {
        let op = self
            .current_ops
            .pop()
            .ok_or_else(|| crate::DiskgraphConfigError::new_err("No operations to pop"))?;
        serde_json::to_string(&op).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!("Failed to serialize op: {}", e))
        })
    }

    /// Close current stage with a short-circuit filter, start new stage.
    fn filter(&mut self, expr: StageFilter) {
        let ops = std::mem::take(&mut self.current_ops);
        self.stages.push(tt::Stage {
            ops,
            filter: Some(expr.inner),
        });
    }

    /// Finalise and return `DiskGraphTTPlan` as JSON.
    fn build(&mut self) -> PyResult<String> {
        if !self.current_ops.is_empty() {
            let ops = std::mem::take(&mut self.current_ops);
            self.stages.push(tt::Stage { ops, filter: None });
        }

        let entity = self.entity.take().ok_or_else(|| {
            crate::DiskgraphConfigError::new_err("call for_each() before build()")
        })?;

        let plan: tt::DiskGraphTTPlan<Nanoseconds> = tt::DiskGraphTTPlan {
            entity,
            temporal: self.temporal.take(),
            prediction_type: std::mem::replace(&mut self.prediction_type, tt::PredictionType::Node),
            is_prediction: self.is_prediction,
            split: self.split.take(),
            stages: std::mem::take(&mut self.stages),
            target_stype: self.target_stype.take(),
            // weight_col is set on custom-TT plans only, via the new
            // `build_train_table_artifacts_for_existing_table` entry
            // point in `custom_tt_entry.rs`. PQ-built plans always emit
            // None.
            weight_col: None,
            fkey_mapping: None,
            lp_resolution: None,
            lp_rhs_is_temporal: false,
            fact_pseudo_time: None,
        };

        let default_time_secs = self.default_time_secs.take();
        with_index_clock!(self.timestamp_unit, U => {
            plan.to_clock::<U>().and_then(|mut plan| {
                plan.entity.default_time_value = default_time_secs.map(Abs::from_secs);
                plan.to_json()
            })
        })
        .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn parse_compare_op(op: &str) -> PyResult<tt::CompareOp> {
    let normalized = op
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_");

    match normalized.as_str() {
        "==" | "eq" => Ok(tt::CompareOp::Eq),
        "!=" | "neq" => Ok(tt::CompareOp::Neq),
        ">" | "gt" => Ok(tt::CompareOp::Gt),
        ">=" | "gte" => Ok(tt::CompareOp::Gte),
        "<" | "lt" => Ok(tt::CompareOp::Lt),
        "<=" | "lte" => Ok(tt::CompareOp::Lte),
        "isnull" | "is_null" => Ok(tt::CompareOp::IsNull),
        "isnotnull" | "is_not_null" => Ok(tt::CompareOp::IsNotNull),
        "startswith" | "starts_with" => Ok(tt::CompareOp::StartsWith),
        "endswith" | "ends_with" => Ok(tt::CompareOp::EndsWith),
        "contains" => Ok(tt::CompareOp::Contains),
        "notcontains" | "not_contains" => Ok(tt::CompareOp::NotContains),
        "in" | "is_in" => Ok(tt::CompareOp::In),
        _ => Err(crate::DiskgraphConfigError::new_err(format!(
            "unknown compare op: {op}"
        ))),
    }
}

fn parse_agg_func(func: &str) -> PyResult<tt::AggFunc> {
    match func.to_lowercase().as_str() {
        "count" => Ok(tt::AggFunc::Count),
        "sum" => Ok(tt::AggFunc::Sum),
        "avg" | "mean" => Ok(tt::AggFunc::Avg),
        "min" => Ok(tt::AggFunc::Min),
        "max" => Ok(tt::AggFunc::Max),
        "count_distinct" | "countdistinct" => Ok(tt::AggFunc::CountDistinct),
        "list_distinct" | "listdistinct" => Ok(tt::AggFunc::ListDistinct),
        "first" => Ok(tt::AggFunc::First),
        "last" => Ok(tt::AggFunc::Last),
        _ => Err(crate::DiskgraphConfigError::new_err(format!(
            "unknown agg func: {func}"
        ))),
    }
}

// ─── Standalone SplitSpec marshalling ────────────────────────────────────────

/// The Python `SplitSpec` dataclass, extracted field by field.
///
/// `#[derive(FromPyObject)]` on a named-field struct does one `getattr` per
/// field, so this binds directly to `diskgraph/split_contract.py`'s
/// `SplitSpec` -- or to anything else carrying the same attributes; the
/// extraction is structural, not nominal, which keeps tests free of the
/// dataclass.
///
/// Field names must match the dataclass exactly. `timestamp_unit` is the
/// dataclass's name for what the plan calls `range_unit`; the rename happens in
/// [`make_split_spec`], which is also where the `days` / `days` / `seconds` / `0`
/// defaults are applied, so a caller that omits them cannot get them wrong.
#[derive(FromPyObject)]
pub(crate) struct SplitSpecArgs {
    pub kind: Option<String>,
    pub ratio: Option<Vec<f64>>,
    pub rnd_seed: Option<u64>,
    pub is_temporal: Option<bool>,
    pub offsets: Option<Vec<i64>>,
    pub ranges: Option<Vec<[i64; 2]>>,
    pub backtesting_timestamp: Option<i64>,
    pub split_column: Option<String>,
    pub split_output_column: Option<String>,
    pub manual_train_values: Option<Vec<String>>,
    pub manual_val_values: Option<Vec<String>>,
    pub manual_test_values: Option<Vec<String>>,
    pub timeframe: Option<i64>,
    pub unit: Option<String>,
    pub offset_unit: Option<String>,
    pub anchor_start_offset: Option<i64>,
    pub flexible_boundaries: Option<bool>,
    pub timeframe_step: Option<i64>,
    pub timestamp_unit: Option<String>,
}

/// Extract a [`SplitSpecArgs`], naming the argument when the object is wrong.
///
/// The derived `FromPyObject` reports a bare `getattr` failure -- "'dict' object
/// has no attribute 'kind'" -- which says neither what was being extracted nor
/// what the caller should have passed. A duck-typed object missing exactly one
/// of the nineteen fields fails the same opaque way.
fn extract_split_spec(spec: &Bound<'_, PyAny>) -> PyResult<SplitSpecArgs> {
    spec.extract::<SplitSpecArgs>().map_err(|e| {
        crate::DiskgraphConfigError::new_err(format!(
            "split spec must be a SplitSpec, or an object carrying all of its \
             fields: {e}"
        ))
    })
}

/// Resolve a unit name, treating `""` and `None` alike.
///
/// The Python callers used to apply these defaults with `spec.unit or "days"`,
/// where `or` catches the empty string too -- and `normalize_split_spec`
/// propagates `unit=""` verbatim when a config carries one.
fn unit_or<'a>(value: &'a Option<String>, default: &'a str) -> &'a str {
    match value.as_deref() {
        Some(s) if !s.is_empty() => s,
        _ => default,
    }
}

/// Parse the unit of `ranges` / `backtesting_timestamp`.
///
/// Stricter than [`parse_time_unit`] because these are *absolute* timestamps:
/// `abs_from_unit` (`diskgraph-core/src/time.rs`) panics on the five units it
/// cannot anchor to the epoch. Rejecting them here turns an unsupported
/// `timestamp_unit` into a config error at the boundary instead of a Rust panic
/// crossing into Python as a `PanicException`.
fn parse_timestamp_unit(unit: &str) -> PyResult<tt::TimeUnit> {
    let parsed = parse_time_unit(unit)?;
    match parsed {
        tt::TimeUnit::Seconds
        | tt::TimeUnit::Milliseconds
        | tt::TimeUnit::Microseconds
        | tt::TimeUnit::Nanoseconds
        | tt::TimeUnit::Days => Ok(parsed),
        other => Err(crate::DiskgraphConfigError::new_err(format!(
            "split_config.timestamp_unit={} is not supported for `ranges` / \
             `backtesting_timestamp`; use seconds, milliseconds, microseconds, \
             nanoseconds, or days",
            time_unit_name(other)
        ))),
    }
}

/// The single implementation of the kumo split config → plan `SplitSpec`
/// contract.
///
/// Both routes into the executor land here: the PQ path via
/// [`DiskGraphTTBuilder::split_ratio`], which folds the result into the plan it
/// is building, and the custom-TT path via [`build_split_spec_json`], which
/// serializes it standalone because
/// `build_train_table_artifacts_for_existing_table` takes the split as its own
/// argument.
///
/// One implementation is what keeps those two agreeing. The unit handling is
/// asymmetric enough to invite drift: `offsets` must be *pre-scaled* into base
/// units here, because `ttgen_executor` reads them as `Dur<U>` and never
/// rescales (`offset_unit` survives only as a label — no consumer reads it),
/// whereas `timeframe` / `anchor_start_offset` / `timeframe_step` must cross
/// *unscaled*, because the executor rescales them itself with
/// `dur_from_unit(v, unit)` — and for `TimeRangeSplit` that unit may be
/// calendar (`months` / `years`), which has no fixed-duration form.
fn make_split_spec(args: SplitSpecArgs) -> PyResult<tt::SplitSpec<Nanoseconds>> {
    let SplitSpecArgs {
        kind,
        ratio,
        rnd_seed,
        is_temporal,
        offsets,
        ranges,
        backtesting_timestamp,
        split_column,
        split_output_column,
        manual_train_values,
        manual_val_values,
        manual_test_values,
        timeframe,
        unit,
        offset_unit,
        anchor_start_offset,
        flexible_boundaries,
        timeframe_step,
        timestamp_unit,
    } = args;
    if let Some(ratio_vals) = ratio.as_ref() {
        if ratio_vals.len() < 2 || ratio_vals.len() > 3 {
            return Err(crate::DiskgraphConfigError::new_err(
                "split ratio must contain 2 or 3 entries",
            ));
        }
        if ratio_vals.iter().any(|v| *v < 0.0) {
            return Err(crate::DiskgraphConfigError::new_err(
                "split ratio cannot contain negative values",
            ));
        }
    }
    // Defaults live here rather than at the call sites: they are part of the
    // contract, and a caller that has to remember them is a caller that can
    // forget them.
    let off_unit = parse_time_unit(unit_or(&offset_unit, "days"))?;
    let split_unit = parse_time_unit(unit_or(&unit, "days"))?;
    let rng_unit = parse_timestamp_unit(unit_or(&timestamp_unit, "seconds"))?;
    let anchor_start_offset = anchor_start_offset.unwrap_or(0);
    let is_date_offset_split = kind.as_deref() == Some("DateOffsetSplit");
    let has_temporal_boundary_semantics = matches!(
        kind.as_deref(),
        Some("DateOffsetSplit") | Some("TimeRangeSplit")
    );
    if is_date_offset_split {
        reject_calendar_date_offset_unit("unit", split_unit)?;
    }
    // When `offset_unit` is "months" or "years", route the int
    // values through `month_offsets` (calendar arithmetic) rather than
    // `offsets` (typed Dur<U>). Years are normalized to months at the
    // boundary so the executor only has to dispatch on one dimension.
    let (offsets_dur, month_offsets) = match (off_unit, offsets) {
        (tt::TimeUnit::Months, Some(v)) => (
            None,
            Some(v.into_iter().map(|o| o as i32).collect::<Vec<i32>>()),
        ),
        (tt::TimeUnit::Years, Some(v)) => (
            None,
            Some(
                v.into_iter()
                    .map(|o| (o as i32).saturating_mul(12))
                    .collect::<Vec<i32>>(),
            ),
        ),
        (_, Some(v)) => (
            Some(
                v.into_iter()
                    .map(|o| dur_from_unit::<Nanoseconds>(o, off_unit))
                    .collect::<Vec<_>>(),
            ),
            None,
        ),
        (_, None) => (None, None),
    };
    Ok(tt::SplitSpec {
        ratio,
        kind,
        rnd_seed,
        is_temporal,
        offsets: offsets_dur,
        offset_unit: is_date_offset_split.then(|| time_unit_name(off_unit).to_string()),
        unit: has_temporal_boundary_semantics.then(|| time_unit_name(split_unit).to_string()),
        timeframe: if has_temporal_boundary_semantics {
            timeframe
        } else {
            None
        },
        anchor_start_offset: if has_temporal_boundary_semantics {
            Some(anchor_start_offset)
        } else {
            None
        },
        ranges: ranges.map(|v| {
            v.into_iter()
                .map(|[lo, hi]| {
                    (
                        abs_from_unit::<Nanoseconds>(lo, rng_unit),
                        abs_from_unit::<Nanoseconds>(hi, rng_unit),
                    )
                })
                .collect()
        }),
        backtesting_timestamp: backtesting_timestamp
            .map(|v| abs_from_unit::<Nanoseconds>(v, rng_unit)),
        split_column,
        split_output_column,
        manual_train_values,
        manual_val_values,
        manual_test_values,
        month_offsets,
        flexible_boundaries: if has_temporal_boundary_semantics {
            flexible_boundaries
        } else {
            None
        },
        timeframe_step: if has_temporal_boundary_semantics {
            timeframe_step
        } else {
            None
        },
    })
}

/// Marshal a `SplitSpec` into the standalone JSON that
/// `build_train_table_artifacts_for_existing_table` accepts as
/// `split_spec_json`: a serialized `SplitSpec` in `timestamp_unit`, which
/// `custom_tt_entry` parses straight back with `serde_json::from_str`. Serialization and
/// deserialization are the derived impls on one struct, so the round-trip is
/// symmetric by construction.
///
/// Takes the same single argument as [`DiskGraphTTBuilder::split_ratio`], so
/// the generated and custom-TT paths share not just the marshalling rules but
/// the call itself -- two hand-maintained argument lists were the weaker
/// version of the same failure mode this exists to prevent.
#[pyfunction]
#[pyo3(signature = (spec, timestamp_unit="seconds"))]
pub fn build_split_spec_json(spec: &Bound<'_, PyAny>, timestamp_unit: &str) -> PyResult<String> {
    let split = make_split_spec(extract_split_spec(spec)?)?;
    with_index_clock!(parse_index_clock(timestamp_unit)?, U => {
        split.to_clock::<U>().and_then(|split| serde_json::to_string(&split))
    })
    .map_err(|e| {
        crate::DiskgraphConfigError::new_err(format!("failed to serialize split spec: {e}"))
    })
}

pub(crate) fn parse_index_clock(unit: &str) -> PyResult<TimeUnit> {
    TimeUnit::from_name(unit)
        .filter(|unit| unit.is_index_clock())
        .ok_or_else(|| {
            crate::DiskgraphConfigError::new_err(format!("unsupported timestamp_unit {unit:?}"))
        })
}

fn parse_time_unit(unit: &str) -> PyResult<tt::TimeUnit> {
    match unit.to_lowercase().as_str() {
        "seconds" | "second" | "s" => Ok(tt::TimeUnit::Seconds),
        "milliseconds" | "millisecond" | "ms" => Ok(tt::TimeUnit::Milliseconds),
        "microseconds" | "microsecond" | "us" => Ok(tt::TimeUnit::Microseconds),
        "nanoseconds" | "nanosecond" | "ns" => Ok(tt::TimeUnit::Nanoseconds),
        "minutes" | "minute" | "min" => Ok(tt::TimeUnit::Minutes),
        "hours" | "hour" | "h" => Ok(tt::TimeUnit::Hours),
        "days" | "day" | "d" => Ok(tt::TimeUnit::Days),
        "weeks" | "week" | "w" => Ok(tt::TimeUnit::Weeks),
        "months" | "month" | "mo" => Ok(tt::TimeUnit::Months),
        "years" | "year" | "y" => Ok(tt::TimeUnit::Years),
        _ => Err(crate::DiskgraphConfigError::new_err(format!(
            "unknown time unit: {unit}"
        ))),
    }
}

fn time_unit_name(unit: tt::TimeUnit) -> &'static str {
    match unit {
        tt::TimeUnit::Nanoseconds => "nanoseconds",
        tt::TimeUnit::Microseconds => "microseconds",
        tt::TimeUnit::Milliseconds => "milliseconds",
        tt::TimeUnit::Seconds => "seconds",
        tt::TimeUnit::Minutes => "minutes",
        tt::TimeUnit::Hours => "hours",
        tt::TimeUnit::Days => "days",
        tt::TimeUnit::Weeks => "weeks",
        tt::TimeUnit::Months => "months",
        tt::TimeUnit::Years => "years",
    }
}

fn reject_calendar_date_offset_unit(field: &str, unit: tt::TimeUnit) -> PyResult<()> {
    if matches!(unit, tt::TimeUnit::Months | tt::TimeUnit::Years) {
        return Err(crate::DiskgraphConfigError::new_err(format!(
            "DateOffsetSplit {field}={} is not supported for DiskGraph summary derivation",
            time_unit_name(unit)
        )));
    }
    Ok(())
}

fn extract_value(value: &Bound<'_, PyAny>, clock: TimeUnit) -> PyResult<tt::Value> {
    // Check bool before int — Python `bool` is a subclass of `int`.
    if let Ok(b) = value.extract::<bool>() {
        Ok(tt::Value::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(tt::Value::Int(i))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(tt::Value::Float(f))
    } else if is_datetime(value) {
        // datetime.datetime (pd.Timestamp is a subclass, handled here too).
        // Round up to the index clock to match rowpack's div_ceil storage semantics.
        let (secs, nanos) = datetime_parts(value)?;
        Ok(tt::Value::Int(ceil_parts(secs, nanos, clock)))
    } else if is_date(value) {
        // datetime.date — convert to midnight UTC epoch seconds.
        // date.toordinal() gives days since 0001-01-01; Unix epoch is ordinal 719163.
        let ordinal: i64 = value
            .call_method0("toordinal")
            .map_err(|e| {
                crate::DiskgraphConfigError::new_err(format!("date.toordinal() failed: {e}"))
            })?
            .extract::<i64>()
            .map_err(|e| {
                crate::DiskgraphConfigError::new_err(format!(
                    "date.toordinal() did not return int: {e}"
                ))
            })?;
        let days_since_epoch = ordinal - 719_163;
        Ok(tt::Value::Int(convert_trunc(
            days_since_epoch,
            TimeUnit::Days,
            clock,
        )))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(tt::Value::Str(s))
    } else if let Ok(values) = value.cast::<PyList>() {
        // Validated PQuery constants reject NULL list members, and the TT IR
        // has no NULL Value variant. Recurse so direct builder calls fail loud
        // instead of silently dropping a NULL member.
        let values = values
            .iter()
            .map(|item| extract_value(&item, clock))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(tt::Value::List(values))
    } else {
        Err(crate::DiskgraphConfigError::new_err(
            "value must be bool, int, float, str, datetime.datetime, datetime.date, or list",
        ))
    }
}

/// Exact epoch seconds and nanoseconds with `.timestamp()`'s time-zone rules
/// (a naive `datetime` is local time, a naive `pd.Timestamp` is UTC). The float
/// `.timestamp()` alone loses a `pd.Timestamp`'s nanoseconds.
fn datetime_parts(value: &Bound<'_, PyAny>) -> PyResult<(i64, u32)> {
    if value.hasattr("nanosecond")? {
        let nanos: i64 = value.getattr("value")?.extract()?;
        return Ok(split_parts(nanos, TimeUnit::Nanoseconds));
    }
    let epoch: f64 = value.call_method0("timestamp")?.extract()?;
    let micros: u32 = value.getattr("microsecond")?.extract()?;
    let secs = (epoch - f64::from(micros) / 1e6).round() as i64;
    Ok((secs, micros * 1_000))
}

/// Check if a Python object is an instance of `datetime.datetime`.
/// Must be checked before `datetime.date` since datetime is a subclass of date.
fn is_datetime(value: &Bound<'_, PyAny>) -> bool {
    value
        .py()
        .import("datetime")
        .and_then(|m| m.getattr("datetime"))
        .and_then(|cls| value.is_instance(&cls))
        .unwrap_or(false)
}

/// Check if a Python object is an instance of `datetime.date` (but not datetime).
fn is_date(value: &Bound<'_, PyAny>) -> bool {
    if is_datetime(value) {
        return false;
    }
    value
        .py()
        .import("datetime")
        .and_then(|m| m.getattr("date"))
        .and_then(|cls| value.is_instance(&cls))
        .unwrap_or(false)
}

fn extract_edge_path(path: &Bound<'_, PyAny>) -> PyResult<Vec<tt::EdgeRef>> {
    if let Ok(edge) = path.extract::<EdgeRef>() {
        Ok(vec![edge.inner])
    } else if let Ok(edges) = path.extract::<Vec<EdgeRef>>() {
        if edges.is_empty() {
            return Err(crate::DiskgraphConfigError::new_err(
                "path must contain at least one EdgeRef",
            ));
        }
        Ok(edges.into_iter().map(|e| e.inner).collect())
    } else {
        Err(crate::DiskgraphConfigError::new_err(
            "path must be an EdgeRef or a list of EdgeRef",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_in_compare_ops() {
        assert_eq!(parse_compare_op("in").unwrap(), tt::CompareOp::In);
        assert_eq!(parse_compare_op("IS IN").unwrap(), tt::CompareOp::In);
    }

    #[test]
    fn time_range_split_preserves_temporal_boundary_semantics() {
        // `split_ratio` now takes a Python object, so exercise the shared
        // implementation directly -- it is what carries the semantics under
        // test, and it needs no interpreter.
        let split = make_split_spec(SplitSpecArgs {
            kind: Some("TimeRangeSplit".to_string()),
            ratio: None,
            rnd_seed: None,
            is_temporal: Some(true),
            offsets: None,
            ranges: Some(vec![[0, 20], [20, 30], [30, 40]]),
            backtesting_timestamp: None,
            split_column: None,
            split_output_column: None,
            manual_train_values: None,
            manual_val_values: None,
            manual_test_values: None,
            timeframe: Some(1),
            unit: Some("days".to_string()),
            offset_unit: Some("days".to_string()),
            anchor_start_offset: Some(2),
            flexible_boundaries: Some(false),
            timeframe_step: Some(1),
            timestamp_unit: Some("seconds".to_string()),
        })
        .expect("spec should build");

        assert_eq!(split.unit.as_deref(), Some("days"));
        assert_eq!(split.timeframe, Some(1));
        assert_eq!(split.anchor_start_offset, Some(2));
        assert_eq!(split.flexible_boundaries, Some(false));
        assert_eq!(split.timeframe_step, Some(1));
    }
}
