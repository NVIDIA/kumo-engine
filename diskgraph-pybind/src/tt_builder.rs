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

use pyo3::exceptions::PyValueError;
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
    #[staticmethod]
    fn compare(column: String, op: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: tt::RowFilter::Compare {
                column,
                op: parse_compare_op(op)?,
                value: extract_value(value)?,
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
    #[staticmethod]
    fn compare(alias: String, op: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: tt::StageFilter::Compare {
                alias,
                op: parse_compare_op(op)?,
                value: extract_value(value)?,
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
            stages: Vec::new(),
            current_ops: Vec::new(),
        }
    }

    /// `FOR EACH {table}.{column}`.
    fn for_each(&mut self, table: String, column: String) {
        self.entity = Some(tt::EntitySpec {
            table,
            column,
            time_col: None,
            filter: None,
        });
    }

    /// Entity-level time column for temporal filtering.
    fn entity_time_col(&mut self, column: String) -> PyResult<()> {
        match &mut self.entity {
            Some(e) => {
                e.time_col = Some(column);
                Ok(())
            }
            None => Err(PyValueError::new_err(
                "call for_each() before entity_time_col()",
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
            None => Err(PyValueError::new_err(
                "call for_each() before entity_filter()",
            )),
        }
    }

    /// Timestamp windows. `timestamps`: list of `(start_us, end_us, "train"|"val"|"test")`.
    #[pyo3(signature = (timestamps, forecast_length, lag_timesteps=None))]
    fn temporal(
        &mut self,
        timestamps: Vec<(i64, i64, String)>,
        forecast_length: i64,
        lag_timesteps: Option<i64>,
    ) -> PyResult<()> {
        let entries = timestamps
            .into_iter()
            .map(|(start, end, split)| {
                let label = match split.as_str() {
                    "train" => tt::SplitLabel::Train,
                    "val" => tt::SplitLabel::Val,
                    "test" => tt::SplitLabel::Test,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "invalid split label: {split} (expected train/val/test)"
                        )))
                    }
                };
                Ok(tt::TimestampEntry {
                    start,
                    end,
                    split: label,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        self.temporal = Some(tt::TemporalSpec {
            timestamps: entries,
            forecast_length,
            lag_timesteps,
        });
        Ok(())
    }

    /// Node-level prediction (default).
    fn node_prediction(&mut self) {
        self.prediction_type = tt::PredictionType::Node;
    }

    /// Link prediction.
    #[pyo3(signature = (target_table, target_col, explode=false))]
    fn link_prediction(&mut self, target_table: String, target_col: String, explode: bool) {
        self.prediction_type = tt::PredictionType::Link {
            target_entity_table: target_table,
            target_entity_col: target_col,
            explode,
        };
    }

    /// Add aggregate op. `path`: single `EdgeRef` or list for multi-hop.
    #[pyo3(signature = (alias, path, func, target_col=None, time_col=None, window=None, window_unit=None, row_filter=None))]
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

        let entity = self
            .entity
            .take()
            .ok_or_else(|| PyValueError::new_err("call for_each() before build()"))?;

        let plan = tt::DiskGraphTTPlan {
            entity,
            temporal: self.temporal.take(),
            prediction_type: std::mem::replace(&mut self.prediction_type, tt::PredictionType::Node),
            stages: std::mem::take(&mut self.stages),
        };

        serde_json::to_string(&plan).map_err(|e| PyValueError::new_err(e.to_string()))
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
        _ => Err(PyValueError::new_err(format!("unknown compare op: {op}"))),
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
        _ => Err(PyValueError::new_err(format!("unknown agg func: {func}"))),
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
        _ => Err(PyValueError::new_err(format!("unknown time unit: {unit}"))),
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
    } else if let Ok(s) = value.extract::<String>() {
        Ok(tt::Value::Str(s))
    } else {
        Err(PyValueError::new_err(
            "value must be bool, int, float, or str",
        ))
    }
}

fn extract_edge_path(path: &Bound<'_, PyAny>) -> PyResult<Vec<tt::EdgeRef>> {
    if let Ok(edge) = path.extract::<EdgeRef>() {
        Ok(vec![edge.inner])
    } else if let Ok(edges) = path.extract::<Vec<EdgeRef>>() {
        if edges.is_empty() {
            return Err(PyValueError::new_err(
                "path must contain at least one EdgeRef",
            ));
        }
        Ok(edges.into_iter().map(|e| e.inner).collect())
    } else {
        Err(PyValueError::new_err(
            "path must be an EdgeRef or a list of EdgeRef",
        ))
    }
}
