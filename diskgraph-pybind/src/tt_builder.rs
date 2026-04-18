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

use diskgraph_core::tt_builder as tt;

// ─── EdgeRef ─────────────────────────────────────────────────────────────────

#[pyclass]
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

#[pyclass]
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
    fn compare(column: String, op: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: tt::RowFilter::Compare {
                lhs: tt::RowFilterLhs::Column(column),
                op: parse_compare_op(op)?,
                rhs: tt::RowFilterRhs::Constant(extract_value(value)?),
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
    fn compare_remote_lhs(
        path: &Bound<'_, PyAny>,
        column: String,
        op: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: tt::RowFilter::Compare {
                lhs: tt::RowFilterLhs::Remote {
                    path: extract_edge_path(path)?,
                    column,
                },
                op: parse_compare_op(op)?,
                rhs: tt::RowFilterRhs::Constant(extract_value(value)?),
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

#[pyclass]
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
    fn compare(alias: String, op: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: tt::StageFilter::Compare {
                alias,
                op: parse_compare_op(op)?,
                rhs: tt::StageFilterRhs::Constant(extract_value(value)?),
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

#[pyclass]
pub struct DiskGraphTTBuilder {
    entity: Option<tt::EntitySpec>,
    temporal: Option<tt::TemporalSpec>,
    prediction_type: tt::PredictionType,
    split: Option<tt::SplitSpec>,
    stages: Vec<tt::Stage>,
    current_ops: Vec<tt::Op>,
}

#[pymethods]
impl DiskGraphTTBuilder {
    #[new]
    fn new() -> Self {
        Self {
            entity: None,
            temporal: None,
            prediction_type: tt::PredictionType::Node,
            split: None,
            stages: Vec::new(),
            current_ops: Vec::new(),
        }
    }

    /// `FOR EACH {table}.{column}`.
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
            default_time_value,
            time_col: None,
            valid_to_col: None,
            filter: None,
        });
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

    /// Timestamp windows. `timestamps`: list of epoch microseconds.
    #[pyo3(signature = (timestamps, forecast_length, lag_timesteps=None))]
    fn temporal(
        &mut self,
        timestamps: Vec<i64>,
        forecast_length: i64,
        lag_timesteps: Option<i64>,
    ) -> PyResult<()> {
        self.temporal = Some(tt::TemporalSpec {
            timestamps,
            forecast_length,
            lag_timesteps,
        });
        Ok(())
    }

    /// Node-level prediction (default).
    fn node_prediction(&mut self) {
        self.prediction_type = tt::PredictionType::Node;
    }

    /// Optional split metadata for native split payload derivation.
    #[pyo3(signature = (
        ratio,
        kind=None,
        rnd_seed=None,
        is_temporal=None,
        offsets=None,
        ranges=None,
        backtesting_timestamp=None,
        split_column=None,
        manual_train_values=None,
        manual_val_values=None,
        manual_test_values=None
    ))]
    fn split_ratio(
        &mut self,
        ratio: Option<Vec<f64>>,
        kind: Option<String>,
        rnd_seed: Option<u64>,
        is_temporal: Option<bool>,
        offsets: Option<Vec<i64>>,
        ranges: Option<Vec<(i64, i64)>>,
        backtesting_timestamp: Option<i64>,
        split_column: Option<String>,
        manual_train_values: Option<Vec<String>>,
        manual_val_values: Option<Vec<String>>,
        manual_test_values: Option<Vec<String>>,
    ) -> PyResult<()> {
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
        self.split = Some(tt::SplitSpec {
            ratio,
            kind,
            rnd_seed,
            is_temporal,
            offsets,
            ranges,
            backtesting_timestamp,
            split_column,
            manual_train_values,
            manual_val_values,
            manual_test_values,
        });
        Ok(())
    }

    /// Link prediction.
    #[pyo3(signature = (explode=false, include_frequency=false))]
    fn link_prediction(&mut self, explode: bool, include_frequency: bool) {
        self.prediction_type = tt::PredictionType::Link {
            explode,
            include_frequency,
        };
    }

    /// Pass-through prediction (LHS/RHS extraction without target ops).
    fn pass_through_prediction(&mut self) {
        self.prediction_type = tt::PredictionType::PassThrough;
    }

    /// Forecasting prediction (outputs Autoregression and YoY columns).
    fn forecasting_prediction(&mut self) {
        self.prediction_type = tt::PredictionType::Forecasting;
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

    /// Direct column lookup (no traversal).
    fn column(&mut self, alias: String, table: String, column: String) {
        self.current_ops.push(tt::Op::Column {
            alias,
            table,
            column,
        });
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
            rhs: extract_value(value)?,
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

    #[pyo3(signature = (alias, target_op_json, lag_offsets))]
    pub fn autoregression(
        &mut self,
        alias: String,
        target_op_json: String,
        lag_offsets: Vec<i64>,
    ) -> PyResult<()> {
        let target_op: tt::Op = serde_json::from_str(&target_op_json).map_err(|e| {
            crate::DiskgraphConfigError::new_err(format!("Invalid target op JSON: {}", e))
        })?;

        self.current_ops.push(tt::Op::Autoregression {
            alias,
            target_op: Box::new(target_op),
            lag_offsets,
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

        let plan = tt::DiskGraphTTPlan {
            entity,
            temporal: self.temporal.take(),
            prediction_type: std::mem::replace(&mut self.prediction_type, tt::PredictionType::Node),
            split: self.split.take(),
            stages: std::mem::take(&mut self.stages),
        };

        serde_json::to_string(&plan)
            .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn parse_compare_op(op: &str) -> PyResult<tt::CompareOp> {
    match op {
        "==" | "eq" => Ok(tt::CompareOp::Eq),
        "!=" | "neq" => Ok(tt::CompareOp::Neq),
        ">" | "gt" => Ok(tt::CompareOp::Gt),
        ">=" | "gte" => Ok(tt::CompareOp::Gte),
        "<" | "lt" => Ok(tt::CompareOp::Lt),
        "<=" | "lte" => Ok(tt::CompareOp::Lte),
        "isnull" | "is_null" => Ok(tt::CompareOp::IsNull),
        "isnotnull" | "is_not_null" => Ok(tt::CompareOp::IsNotNull),
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

fn parse_time_unit(unit: &str) -> PyResult<tt::TimeUnit> {
    match unit.to_lowercase().as_str() {
        "seconds" | "second" | "s" => Ok(tt::TimeUnit::Seconds),
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

fn extract_value(value: &Bound<'_, PyAny>) -> PyResult<tt::Value> {
    // Check bool before int — Python `bool` is a subclass of `int`.
    if let Ok(b) = value.extract::<bool>() {
        Ok(tt::Value::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(tt::Value::Int(i))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(tt::Value::Float(f))
    } else if is_datetime(value) {
        // datetime.datetime (pd.Timestamp is a subclass, handled here too).
        // Convert to epoch seconds via Python's .timestamp() method, then
        // ceiling to match rowpack's div_ceil storage semantics.
        let epoch_f: f64 = value
            .call_method0("timestamp")
            .map_err(|e| {
                crate::DiskgraphConfigError::new_err(format!(
                    "datetime.timestamp() failed: {e}"
                ))
            })?
            .extract::<f64>()
            .map_err(|e| {
                crate::DiskgraphConfigError::new_err(format!(
                    "datetime.timestamp() did not return float: {e}"
                ))
            })?;
        Ok(tt::Value::Int(epoch_f.ceil() as i64))
    } else if is_date(value) {
        // datetime.date — convert to midnight UTC epoch seconds.
        // date.toordinal() gives days since 0001-01-01; Unix epoch is ordinal 719163.
        let ordinal: i64 = value
            .call_method0("toordinal")
            .map_err(|e| {
                crate::DiskgraphConfigError::new_err(format!(
                    "date.toordinal() failed: {e}"
                ))
            })?
            .extract::<i64>()
            .map_err(|e| {
                crate::DiskgraphConfigError::new_err(format!(
                    "date.toordinal() did not return int: {e}"
                ))
            })?;
        let days_since_epoch = ordinal - 719_163;
        Ok(tt::Value::Int(days_since_epoch * 86_400))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(tt::Value::Str(s))
    } else {
        Err(crate::DiskgraphConfigError::new_err(
            "value must be bool, int, float, str, datetime.datetime, or datetime.date",
        ))
    }
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
