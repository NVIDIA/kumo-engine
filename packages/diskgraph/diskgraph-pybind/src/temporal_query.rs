// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small, batch-oriented temporal row query used by storage adapters.
//!
//! This is deliberately below task construction and model encoding.  A caller
//! supplies dense roots, per-root cutoffs, exact relation paths, and the row
//! projections/reductions it needs.  Each distinct row frame is traversed and
//! decoded once, then shared by sequence and aggregate consumers.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, NullArray, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::pyarrow::ToPyArrow;
use arrow_array::RecordBatch;
use diskgraph_core::metadata::EdgeSortKeyMetadata;
use diskgraph_core::sortkey::SortKeyDecoder;
use diskgraph_engine::edge_unpacker::EdgeUnpacker;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use serde::Deserialize;

use crate::DiskgraphConfigError;

const REDUCTIONS: &[&str] = &[
    "count",
    "count_distinct",
    "sum",
    "mean",
    "min",
    "max",
    "std_population",
    "std_sample",
    "var_population",
    "var_sample",
    "recency",
    "mode",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryPlan {
    frames: Vec<FramePlan>,
    #[serde(default)]
    sequences: Vec<SequencePlan>,
    #[serde(default)]
    aggregates: Vec<AggregatePlan>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FramePlan {
    name: String,
    root_type: String,
    path: Vec<PathStep>,
    time_column: Option<String>,
    start_seconds: Option<i64>,
    stop_seconds: Option<i64>,
    distinct: bool,
    columns: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PathStep {
    relation: String,
    destination_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SequencePlan {
    name: String,
    frame: String,
    columns: Vec<String>,
    order: SequenceOrder,
    limit: Option<usize>,
    #[serde(default)]
    output_order: Option<SequenceOrder>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SequenceOrder {
    Ascending,
    Descending,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AggregatePlan {
    name: String,
    frame: String,
    #[serde(default)]
    groups: Vec<GroupPlan>,
    operation: String,
    column: Option<String>,
    sum_type: Option<SumType>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupPlan {
    keys: Vec<String>,
    values: Vec<NamedReduction>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedReduction {
    name: String,
    operation: String,
    column: Option<String>,
    sum_type: Option<SumType>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SumType {
    Int64,
    Uint64,
}

#[derive(Clone, Debug)]
enum ScalarValue {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    String(String),
}

/// Python `repr(float)`: Rust's shortest round-trip digits and exponent cutoffs, Python spelling.
fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    let text = format!("{value:?}");
    match text.split_once('e') {
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(digits) => ('-', digits),
                None => ('+', exponent),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => text,
    }
}

/// Characters Python's `str.isprintable()` rejects, apart from ASCII space. Surrogates (Cs)
/// cannot occur in a Rust `str`.
static NON_PRINTABLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{Cc}\p{Cf}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]").expect("valid class")
});

/// Python `repr(str)`.
fn python_str_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    let mut buffer = [0; 4];
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ' '..='~' => out.push(ch),
            _ if NON_PRINTABLE.is_match(ch.encode_utf8(&mut buffer)) => {
                let code = u32::from(ch);
                out.push_str(&match code {
                    0..=0xff => format!("\\x{code:02x}"),
                    0x100..=0xffff => format!("\\u{code:04x}"),
                    _ => format!("\\U{code:08x}"),
                });
            }
            _ => out.push(ch),
        }
    }
    out.push(quote);
    out
}

impl ScalarValue {
    fn numeric(&self) -> Result<Option<f64>, String> {
        match self {
            Self::Null => Ok(None),
            Self::I64(value) => Ok(Some(*value as f64)),
            Self::U64(value) => Ok(Some(*value as f64)),
            Self::F64(value) => Ok(Some(*value)),
            Self::Bool(_) | Self::String(_) => Err("reduction requires a numeric field".into()),
        }
    }

    fn epoch_seconds(&self) -> Result<Option<i64>, String> {
        match self {
            Self::Null => Ok(None),
            Self::I64(value) => Ok(Some(*value)),
            Self::U64(value) => i64::try_from(*value)
                .map(Some)
                .map_err(|_| "timestamp exceeds int64 range".to_string()),
            _ => Err("time field must be an integer or timestamp".to_string()),
        }
    }

    fn equality_key(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Bool(value) => format!("bool:{value}"),
            Self::I64(value) => format!("int:{value}"),
            Self::U64(value) => format!("int:{value}"),
            Self::F64(value) => format!("float:{:016x}", value.to_bits()),
            Self::String(value) => format!("str:{value}"),
        }
    }

    /// Mirrors RDL's stable `(type(value).__name__, repr(value))` tie break.
    fn stable_sort_key(&self) -> (String, String) {
        match self {
            Self::Null => ("NoneType".into(), "None".into()),
            Self::Bool(value) => ("bool".into(), if *value { "True" } else { "False" }.into()),
            Self::I64(value) => ("int".into(), value.to_string()),
            Self::U64(value) => ("int".into(), value.to_string()),
            Self::F64(value) => ("float".into(), python_float_repr(*value)),
            Self::String(value) => ("str".into(), python_str_repr(value)),
        }
    }

    fn compare(&self, other: &Self) -> Result<Ordering, String> {
        match (self, other) {
            (Self::I64(left), Self::I64(right)) => Ok(left.cmp(right)),
            (Self::U64(left), Self::U64(right)) => Ok(left.cmp(right)),
            (Self::F64(left), Self::F64(right)) => left
                .partial_cmp(right)
                .ok_or_else(|| "min/max cannot order NaN".to_string()),
            (Self::I64(left), Self::U64(right)) => Ok(i128::from(*left).cmp(&i128::from(*right))),
            (Self::U64(left), Self::I64(right)) => Ok(i128::from(*left).cmp(&i128::from(*right))),
            (Self::String(left), Self::String(right)) => Ok(left.cmp(right)),
            (Self::Bool(left), Self::Bool(right)) => Ok(left.cmp(right)),
            _ => Err("min/max values have incompatible types".to_string()),
        }
    }

    fn append_to(&self, list: &Bound<'_, PyList>) -> PyResult<()> {
        match self {
            Self::Null => list.append(list.py().None()),
            Self::Bool(value) => list.append(*value),
            Self::I64(value) => list.append(*value),
            Self::U64(value) => list.append(*value),
            Self::F64(value) => list.append(*value),
            Self::String(value) => list.append(value),
        }
    }
}

fn scalar_values_to_array(
    values: &[ScalarValue],
    sum_type: Option<SumType>,
) -> Result<ArrayRef, String> {
    let first = values
        .iter()
        .find(|value| !matches!(value, ScalarValue::Null))
        .cloned()
        .or_else(|| {
            sum_type.map(|kind| match kind {
                SumType::Int64 => ScalarValue::I64(0),
                SumType::Uint64 => ScalarValue::U64(0),
            })
        });
    match first.as_ref() {
        None => Ok(Arc::new(NullArray::new(values.len()))),
        Some(ScalarValue::Bool(_)) => Ok(Arc::new(BooleanArray::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Null => Ok(None),
                    ScalarValue::Bool(value) => Ok(Some(*value)),
                    _ => Err("aggregate values have inconsistent physical types".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        Some(ScalarValue::I64(_)) => Ok(Arc::new(Int64Array::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Null => Ok(None),
                    ScalarValue::I64(value) => Ok(Some(*value)),
                    _ => Err("aggregate values have inconsistent physical types".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        Some(ScalarValue::U64(_)) => Ok(Arc::new(UInt64Array::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Null => Ok(None),
                    ScalarValue::U64(value) => Ok(Some(*value)),
                    _ => Err("aggregate values have inconsistent physical types".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        Some(ScalarValue::F64(_)) => Ok(Arc::new(Float64Array::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Null => Ok(None),
                    ScalarValue::F64(value) => Ok(Some(*value)),
                    _ => Err("aggregate values have inconsistent physical types".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        Some(ScalarValue::String(_)) => Ok(Arc::new(StringArray::from(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Null => Ok(None),
                    ScalarValue::String(value) => Ok(Some(value.as_str())),
                    _ => Err("aggregate values have inconsistent physical types".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        Some(ScalarValue::Null) => unreachable!("nulls were filtered above"),
    }
}

struct MaterializedFrame {
    name: String,
    destination: String,
    deferred_columns: HashMap<String, usize>,
    positions: HashMap<String, usize>,
    batch: RecordBatch,
    node_ids: Vec<u64>,
    timestamps: Vec<Option<i64>>,
    offsets: Vec<usize>,
}

struct SequenceResult {
    name: String,
    offsets: Vec<usize>,
    node_ids: Vec<u64>,
    timestamps: Vec<Option<i64>>,
    batch: RecordBatch,
}

struct AggregateResult {
    name: String,
    values: Vec<ScalarValue>,
    sum_type: Option<SumType>,
}

#[derive(Default)]
struct QueryTimings {
    traversal: u64,
    decode_and_select: u64,
    sequence_assembly: u64,
    aggregation: u64,
    total: u64,
}

#[derive(Default)]
struct QueryStats {
    frames_materialized: usize,
    rows_materialized: usize,
    rows_decoded: usize,
    values_decoded: usize,
    edges_visited: u64,
}

#[derive(Default)]
struct LengthTimings {
    traversal: u64,
    time_filter: u64,
    total: u64,
}

#[derive(Default)]
struct LengthStats {
    frames_counted: usize,
    rows_considered: usize,
    edges_visited: u64,
    sort_key_timestamps: usize,
    rowpack_timestamps: usize,
}

struct SequenceLengths {
    name: String,
    lengths: Vec<usize>,
}

pub(crate) struct LengthResult {
    sequences: Vec<SequenceLengths>,
    timings: LengthTimings,
    stats: LengthStats,
}

pub(crate) struct QueryResult {
    sequences: Vec<SequenceResult>,
    aggregates: Vec<AggregateResult>,
    timings: QueryTimings,
    stats: QueryStats,
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn projection(table: &TableReader, names: &[String]) -> Result<Vec<usize>, String> {
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| format!("table {:?} has no rowpack data", table.node_type()))?;
    names
        .iter()
        .map(|name| {
            decoder
                .context()
                .columns
                .iter()
                .position(|column| &column.name == name)
                .ok_or_else(|| format!("table {:?} has no column {name:?}", table.node_type()))
        })
        .collect()
}

fn table_rows(
    table: &TableReader,
    node_ids: &[u64],
    projection_indices: &[usize],
) -> Result<RecordBatch, String> {
    let decoder = table
        .node_data_decoder()
        .ok_or_else(|| format!("table {:?} has no rowpack data", table.node_type()))?;
    let mut rows = Vec::with_capacity(node_ids.len());
    for &node_id in node_ids {
        let view = table
            .edges_for_node(node_id)
            .map_err(|error| error.to_string())?;
        rows.push(
            view.node_data()
                .map_err(|error| error.to_string())?
                .ok_or_else(|| format!("node {node_id} has no rowpack data"))?,
        );
    }
    decoder
        .decode_batch_to_arrow(&rows, projection_indices)
        .map_err(|error| error.to_string())
}

fn scalar_at(array: &dyn Array, row: usize) -> Result<ScalarValue, String> {
    if array.is_null(row) {
        return Ok(ScalarValue::Null);
    }
    macro_rules! signed {
        ($type:ty) => {
            if let Some(values) = array.as_any().downcast_ref::<$type>() {
                return Ok(ScalarValue::I64(values.value(row) as i64));
            }
        };
    }
    signed!(Int8Array);
    signed!(Int16Array);
    signed!(Int32Array);
    signed!(Int64Array);
    macro_rules! unsigned {
        ($type:ty) => {
            if let Some(values) = array.as_any().downcast_ref::<$type>() {
                return Ok(ScalarValue::U64(values.value(row) as u64));
            }
        };
    }
    unsigned!(UInt8Array);
    unsigned!(UInt16Array);
    unsigned!(UInt32Array);
    unsigned!(UInt64Array);
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(ScalarValue::F64(f64::from(values.value(row))));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(ScalarValue::F64(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(ScalarValue::Bool(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(ScalarValue::String(values.value(row).to_string()));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(ScalarValue::String(values.value(row).to_string()));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        return Ok(ScalarValue::I64(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return Ok(ScalarValue::I64(values.value(row).div_euclid(1_000)));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Ok(ScalarValue::I64(values.value(row).div_euclid(1_000_000)));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok(ScalarValue::I64(
            values.value(row).div_euclid(1_000_000_000),
        ));
    }
    if let Some(values) = array.as_any().downcast_ref::<Date32Array>() {
        return Ok(ScalarValue::I64(i64::from(values.value(row)) * 86_400));
    }
    if let Some(values) = array.as_any().downcast_ref::<Date64Array>() {
        return Ok(ScalarValue::I64(values.value(row).div_euclid(1_000)));
    }
    Err(format!(
        "unsupported scalar Arrow type {}",
        array.data_type()
    ))
}

fn relation_columns(
    table: &TableReader,
    relation_name: &str,
    node_id: u64,
    include_sort_keys: bool,
    window: Option<(&SortKeyDecoder, Option<i64>, Option<i64>, &str)>,
    edges_visited: &mut u64,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let relation_id = table
        .relations()
        .iter()
        .position(|relation| relation.name == relation_name)
        .ok_or_else(|| {
            format!(
                "relation {relation_name:?} is not stored on source table {:?}",
                table.node_type()
            )
        })?;
    let view = table
        .edges_for_node(node_id)
        .map_err(|error| error.to_string())?;
    let segments = view
        .relation_segments()
        .map_err(|error| error.to_string())?;
    let segment = segments
        .iter()
        .find(|segment| segment.layout.relation_id as usize == relation_id)
        .ok_or_else(|| format!("relation {relation_name:?} has no physical segment"))?;
    let mut unpacker = EdgeUnpacker::new(segment.data, &segment.layout);
    let mut destinations = Vec::with_capacity(segment.count as usize);
    let mut sort_keys = if include_sort_keys {
        Vec::with_capacity(segment.count as usize)
    } else {
        Vec::new()
    };
    let mut destination_buffer = [0u64; 256];
    let mut sort_key_buffer = [0u64; 256];
    loop {
        let count = unpacker.unpack_batch(&mut destination_buffer, &mut sort_key_buffer);
        if count == 0 {
            break;
        }
        if let Some((decoder, lower, upper, _)) = window {
            for (&destination, &key) in destination_buffer[..count]
                .iter()
                .zip(&sort_key_buffer[..count])
            {
                // A clipped code is not the row's time; the rowpack check decides it.
                let admitted = exact_key_time(decoder, key).is_none_or(|timestamp| {
                    lower.is_none_or(|bound| timestamp > bound)
                        && upper.is_none_or(|bound| timestamp <= bound)
                });
                if admitted {
                    destinations.push(destination);
                    if include_sort_keys {
                        sort_keys.push(key);
                    }
                }
            }
        } else {
            destinations.extend_from_slice(&destination_buffer[..count]);
            if include_sort_keys {
                sort_keys.extend_from_slice(&sort_key_buffer[..count]);
            }
        }
        *edges_visited = edges_visited.saturating_add(count as u64);
    }
    Ok((destinations, sort_keys))
}

fn relation_destinations(
    table: &TableReader,
    relation_name: &str,
    node_id: u64,
    window: Option<(&SortKeyDecoder, Option<i64>, Option<i64>, &str)>,
    sequence: Option<&SequencePlan>,
    distinct: bool,
    stats: &mut QueryStats,
) -> Result<Vec<u64>, String> {
    let (mut destinations, keys) = relation_columns(
        table,
        relation_name,
        node_id,
        sequence.is_some(),
        window,
        &mut stats.edges_visited,
    )?;
    if let (Some(sequence), Some((decoder, ..))) = (sequence, window) {
        if let Some(limit) = sequence.limit.filter(|&n| n > 0 && n < destinations.len()) {
            // All-one bits decode to the maximum ordered timestamp, regardless
            // of timestamp width or weight bits. Ingestion may clamp later rows
            // there: they must reach the existing rowpack window check before
            // they can count toward the limit.
            if keys.iter().any(|&key| exact_key_time(decoder, key).is_none()) {
                return Ok(destinations);
            }
            let mut seen = HashSet::new();
            let mut timestamps = keys
                .iter()
                .zip(&destinations)
                .filter(|(_, node)| !distinct || seen.insert(**node))
                .map(|(&key, _)| decoder.decode(key).0)
                .collect::<Vec<_>>();
            if timestamps.len() <= limit {
                return Ok(destinations);
            }
            let rank = match sequence.order {
                SequenceOrder::Ascending => limit - 1,
                SequenceOrder::Descending => timestamps.len() - limit,
            };
            let boundary = *timestamps.select_nth_unstable(rank).1;
            let mut index = 0;
            // Keep the entire boundary tie in encounter order. The existing
            // rowpack primary-key sort and sequence limit make the final choice.
            destinations.retain(|_| {
                let timestamp = decoder.decode(keys[index]).0;
                index += 1;
                match sequence.order {
                    SequenceOrder::Ascending => timestamp <= boundary,
                    SequenceOrder::Descending => timestamp >= boundary,
                }
            });
        }
    }
    Ok(destinations)
}

/// The time a sort key stores exactly. Ingestion clips out-of-range times to the
/// ceiling code or to the lowest code, which also means NULL; neither is exact.
fn exact_key_time(decoder: &SortKeyDecoder, key: u64) -> Option<i64> {
    let timestamp = decoder.decode(key).0?;
    (Some(timestamp) != decoder.decode(u64::MAX).0).then_some(timestamp)
}

fn matching_time_decoder(
    metadata: Option<&EdgeSortKeyMetadata>,
    destination: &str,
    clock: &str,
) -> Option<SortKeyDecoder> {
    metadata.and_then(|metadata| {
        metadata.timestamp.as_ref().and_then(|timestamp| {
            (timestamp.table == destination
                && timestamp.column == clock
                && timestamp.encoding.step == 1)
                .then(|| SortKeyDecoder::new(metadata))
        })
    })
}

fn validate_frame_plan(plan: &FramePlan, root_type: &str) -> Result<(), String> {
    if plan.name.is_empty() {
        return Err("temporal frame names must be non-empty".into());
    }
    if plan.root_type != root_type {
        return Err(format!(
            "frame {:?} starts from {:?}, but roots are {:?}",
            plan.name, plan.root_type, root_type
        ));
    }
    if plan
        .start_seconds
        .zip(plan.stop_seconds)
        .is_some_and(|(start, stop)| start >= stop)
    {
        return Err(format!("frame {:?} must start before it stops", plan.name));
    }
    if plan.time_column.is_none() && (plan.start_seconds.is_some() || plan.stop_seconds.is_some()) {
        return Err(format!(
            "frame {:?} cannot set time bounds without a time column",
            plan.name
        ));
    }
    Ok(())
}

fn take_batch(batch: &RecordBatch, rows: &[u64], columns: &[usize]) -> Result<RecordBatch, String> {
    let indices = UInt64Array::from(rows.to_vec());
    let arrays = columns
        .iter()
        .map(|&column| arrow::compute::take(batch.column(column), &indices, None))
        .collect::<Result<Vec<ArrayRef>, _>>()
        .map_err(|error| error.to_string())?;
    let fields = columns
        .iter()
        .map(|&column| batch.schema().field(column).clone())
        .collect::<Vec<_>>();
    RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), arrays)
        .map_err(|error| error.to_string())
}

fn sort_frame_rows(rows: &mut [(usize, Option<i64>, ScalarValue)]) {
    rows.sort_by_key(|row| row.1);
    let mut start = 0;
    while start < rows.len() {
        let mut stop = start + 1;
        while stop < rows.len() && rows[stop].1 == rows[start].1 {
            stop += 1;
        }
        // Format the existing primary-key tie breaker only when time ties.
        if stop - start > 1 {
            rows[start..stop].sort_by_cached_key(|row| row.2.stable_sort_key());
        }
        start = stop;
    }
}

fn materialize_frame(
    reader: &GraphIndexReader,
    plan: &FramePlan,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    sequence: Option<&SequencePlan>,
    full_columns: Option<(&HashSet<&str>, usize)>,
    timings: &mut QueryTimings,
    stats: &mut QueryStats,
) -> Result<MaterializedFrame, String> {
    validate_frame_plan(plan, root_type)?;

    let traversal_started = Instant::now();
    let mut rows_by_anchor = Vec::with_capacity(roots.len());
    let mut destination = root_type.to_string();
    for (anchor, &root) in roots.iter().enumerate() {
        let mut frontier = vec![root];
        destination = root_type.to_string();
        for (hop, step) in plan.path.iter().enumerate() {
            let relation = reader
                .merged_edge_relations()
                .iter()
                .find(|relation| relation.name == step.relation)
                .ok_or_else(|| format!("unknown temporal relation {:?}", step.relation))?;
            if relation.src_type != destination || relation.dst_type != step.destination_type {
                return Err(format!(
                    "relation {:?} is {:?} -> {:?}, not {:?} -> {:?}",
                    step.relation,
                    relation.src_type,
                    relation.dst_type,
                    destination,
                    step.destination_type
                ));
            }
            let table = reader
                .table(&destination)
                .ok_or_else(|| format!("unknown source table {destination:?}"))?;
            let decoder = if hop + 1 == plan.path.len()
                && (plan.start_seconds.is_some() || plan.stop_seconds.is_some())
            {
                plan.time_column.as_deref().and_then(|clock| {
                    matching_time_decoder(
                        relation.dest_sort_key.as_ref(),
                        &step.destination_type,
                        clock,
                    )
                })
            } else {
                None
            };
            let window = decoder.as_ref().map(|decoder| {
                (
                    decoder,
                    plan.start_seconds
                        .map(|seconds| cutoffs[anchor].saturating_add(seconds)),
                    plan.stop_seconds
                        .map(|seconds| cutoffs[anchor].saturating_add(seconds)),
                    plan.name.as_str(),
                )
            });
            let mut next = Vec::new();
            for node_id in frontier {
                next.extend(relation_destinations(
                    table.as_ref(),
                    &step.relation,
                    node_id,
                    window,
                    sequence.filter(|_| plan.path.len() == 1 && window.is_some()),
                    plan.distinct,
                    stats,
                )?);
            }
            frontier = next;
            destination = step.destination_type.clone();
        }
        if plan.distinct {
            let mut seen = HashSet::new();
            frontier.retain(|node_id| seen.insert(*node_id));
        }
        rows_by_anchor.push(frontier);
    }
    timings.traversal = timings
        .traversal
        .saturating_add(elapsed_ns(traversal_started));

    let decode_started = Instant::now();
    let table = reader
        .table(&destination)
        .ok_or_else(|| format!("unknown destination table {destination:?}"))?;
    let primary_key = table
        .primary_key_map()
        .ok_or_else(|| format!("table {destination:?} has no primary key metadata"))?
        .column
        .clone();
    let mut columns = plan.columns.clone();
    if let Some(time_column) = &plan.time_column {
        if !columns.contains(time_column) {
            columns.push(time_column.clone());
        }
    }
    if !columns.contains(&primary_key) {
        columns.push(primary_key.clone());
    }
    let projection_indices = projection(table.as_ref(), &columns)?;
    let mut deferred_columns = HashMap::new();
    // A second rowpack pass loses when most rows are returned. Conservatively
    // defer only if at most a quarter of candidates need sequence-only fields.
    // This changes execution cost, never which rows an output receives.
    let full_columns = full_columns.filter(|(_, limit)| {
        let total = rows_by_anchor.iter().map(Vec::len).sum::<usize>();
        let selected = rows_by_anchor
            .iter()
            .map(|rows| rows.len().min(*limit))
            .sum::<usize>();
        total > 0 && selected <= total / 4
    });
    let (columns, projection_indices) = if let Some((required, _)) = full_columns {
        columns
            .into_iter()
            .zip(projection_indices)
            .filter(|(name, index)| {
                let keep = required.contains(name.as_str())
                    || name == &primary_key
                    || plan.time_column.as_ref() == Some(name);
                if !keep {
                    deferred_columns.insert(name.clone(), *index);
                }
                keep
            })
            .unzip()
    } else {
        (columns, projection_indices)
    };
    let positions = columns
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect::<HashMap<_, _>>();
    let flat_nodes = rows_by_anchor.iter().flatten().copied().collect::<Vec<_>>();
    stats.rows_decoded += flat_nodes.len();
    stats.values_decoded += flat_nodes.len() * projection_indices.len();
    let decoded = table_rows(table.as_ref(), &flat_nodes, &projection_indices)?;

    let mut input_offsets = Vec::with_capacity(rows_by_anchor.len() + 1);
    input_offsets.push(0usize);
    for rows in &rows_by_anchor {
        input_offsets.push(input_offsets.last().copied().unwrap_or(0) + rows.len());
    }
    let primary_key_position = positions[&primary_key];
    let time_position = plan.time_column.as_ref().map(|column| positions[column]);
    let mut selected_rows = Vec::<u64>::new();
    let mut node_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut offsets = Vec::with_capacity(roots.len() + 1);
    offsets.push(0);
    for anchor in 0..roots.len() {
        let lower = plan
            .start_seconds
            .map(|value| cutoffs[anchor].saturating_add(value));
        let upper = plan
            .stop_seconds
            .map(|value| cutoffs[anchor].saturating_add(value));
        let mut anchor_rows = Vec::<(usize, Option<i64>, ScalarValue)>::new();
        for row in input_offsets[anchor]..input_offsets[anchor + 1] {
            let timestamp = match time_position {
                Some(position) => Some(
                    scalar_at(decoded.column(position).as_ref(), row)?
                        .epoch_seconds()?
                        .ok_or_else(|| {
                            format!("frame {:?} contains a null timestamp", plan.name)
                        })?,
                ),
                None => None,
            };
            if timestamp.is_some_and(|value| {
                lower.is_some_and(|bound| value <= bound)
                    || upper.is_some_and(|bound| value > bound)
            }) {
                continue;
            }
            let key = scalar_at(decoded.column(primary_key_position).as_ref(), row)?;
            anchor_rows.push((row, timestamp, key));
        }
        sort_frame_rows(&mut anchor_rows);
        for (row, timestamp, _) in anchor_rows {
            selected_rows.push(row as u64);
            node_ids.push(flat_nodes[row]);
            timestamps.push(timestamp);
        }
        offsets.push(selected_rows.len());
    }
    let column_indices = (0..columns.len()).collect::<Vec<_>>();
    let batch = take_batch(&decoded, &selected_rows, &column_indices)?;
    timings.decode_and_select = timings
        .decode_and_select
        .saturating_add(elapsed_ns(decode_started));
    stats.frames_materialized += 1;
    stats.rows_materialized += batch.num_rows();
    Ok(MaterializedFrame {
        name: plan.name.clone(),
        destination,
        deferred_columns,
        positions,
        batch,
        node_ids,
        timestamps,
        offsets,
    })
}

fn sequence_result(
    reader: &GraphIndexReader,
    frame: &MaterializedFrame,
    plan: &SequencePlan,
    stats: &mut QueryStats,
) -> Result<SequenceResult, String> {
    if plan.name.is_empty() {
        return Err("sequence names must be non-empty".into());
    }
    if plan.limit == Some(0) {
        return Err(format!("sequence {:?} limit must be positive", plan.name));
    }
    let mut selected = Vec::<u64>::new();
    let mut node_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut offsets = Vec::with_capacity(frame.offsets.len());
    offsets.push(0);
    for anchor in 0..frame.offsets.len() - 1 {
        let start = frame.offsets[anchor];
        let stop = frame.offsets[anchor + 1];
        let mut rows = (start..stop).collect::<Vec<_>>();
        if matches!(plan.order, SequenceOrder::Descending) {
            rows.reverse();
        }
        if let Some(limit) = plan.limit {
            rows.truncate(limit);
        }
        if plan.output_order.is_some_and(|order| order != plan.order) {
            rows.reverse();
        }
        for row in rows {
            selected.push(row as u64);
            node_ids.push(frame.node_ids[row]);
            timestamps.push(frame.timestamps[row]);
        }
        offsets.push(selected.len());
    }
    let batch = if frame.deferred_columns.is_empty()
        || plan
            .columns
            .iter()
            .all(|name| frame.positions.contains_key(name))
    {
        let columns = plan
            .columns
            .iter()
            .map(|name| {
                frame.positions.get(name).copied().ok_or_else(|| {
                    format!("frame {:?} did not project column {name:?}", frame.name)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        take_batch(&frame.batch, &selected, &columns)?
    } else {
        let missing = plan
            .columns
            .iter()
            .filter_map(|name| frame.deferred_columns.get(name).copied())
            .collect::<Vec<_>>();
        let table = reader
            .table(&frame.destination)
            .ok_or_else(|| format!("unknown destination table {:?}", frame.destination))?;
        let extra = table_rows(table.as_ref(), &node_ids, &missing)?;
        stats.rows_decoded += node_ids.len();
        stats.values_decoded += node_ids.len() * missing.len();
        let indices = UInt64Array::from(selected);
        let mut fields = Vec::with_capacity(plan.columns.len());
        let mut arrays = Vec::with_capacity(plan.columns.len());
        let mut next = 0;
        for name in &plan.columns {
            if let Some(&column) = frame.positions.get(name) {
                fields.push(frame.batch.schema().field(column).clone());
                arrays.push(
                    arrow::compute::take(frame.batch.column(column), &indices, None)
                        .map_err(|error| error.to_string())?,
                );
            } else if frame.deferred_columns.contains_key(name) {
                fields.push(extra.schema().field(next).clone());
                arrays.push(extra.column(next).clone());
                next += 1;
            } else {
                return Err(format!(
                    "frame {:?} did not project column {name:?}",
                    frame.name
                ));
            }
        }
        RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), arrays)
            .map_err(|error| error.to_string())?
    };
    Ok(SequenceResult {
        name: plan.name.clone(),
        offsets,
        node_ids,
        timestamps,
        batch,
    })
}

#[derive(Clone)]
struct OwnedRow {
    values: BTreeMap<String, ScalarValue>,
}

enum RowsView<'a> {
    Frame {
        frame: &'a MaterializedFrame,
        start: usize,
        stop: usize,
    },
    Owned(&'a [OwnedRow]),
}

impl RowsView<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Frame { start, stop, .. } => stop - start,
            Self::Owned(rows) => rows.len(),
        }
    }

    fn values(&self, column: &str) -> Result<Vec<ScalarValue>, String> {
        match self {
            Self::Frame { frame, start, stop } => {
                let position = frame.positions.get(column).copied().ok_or_else(|| {
                    format!("frame {:?} did not project column {column:?}", frame.name)
                })?;
                (*start..*stop)
                    .map(|row| scalar_at(frame.batch.column(position).as_ref(), row))
                    .collect()
            }
            Self::Owned(rows) => rows
                .iter()
                .map(|row| {
                    row.values
                        .get(column)
                        .cloned()
                        .ok_or_else(|| format!("grouped row has no column {column:?}"))
                })
                .collect(),
        }
    }
}

fn validate_reduction(operation: &str, column: Option<&str>) -> Result<(), String> {
    if !REDUCTIONS.contains(&operation) {
        return Err(format!("unsupported temporal reduction {operation:?}"));
    }
    if operation != "count" && column.is_none() {
        return Err(format!("{operation} requires a column"));
    }
    Ok(())
}

fn reduce(
    rows: RowsView<'_>,
    operation: &str,
    column: Option<&str>,
    cutoff: i64,
    sum_type: Option<SumType>,
) -> Result<ScalarValue, String> {
    validate_reduction(operation, column)?;
    if sum_type.is_some() && operation != "sum" {
        return Err("sum_type is only valid for sum".into());
    }
    if operation == "count" {
        return i64::try_from(rows.len())
            .map(ScalarValue::I64)
            .map_err(|_| "count exceeds int64 range".to_string());
    }
    let values = rows
        .values(column.expect("validated reduction column"))?
        .into_iter()
        .filter(|value| !matches!(value, ScalarValue::Null))
        .collect::<Vec<_>>();
    if operation == "count_distinct" {
        return i64::try_from(
            values
                .iter()
                .map(ScalarValue::equality_key)
                .collect::<HashSet<_>>()
                .len(),
        )
        .map(ScalarValue::I64)
        .map_err(|_| "count_distinct exceeds int64 range".to_string());
    }
    if operation == "sum" && rows.len() == 0 {
        return Ok(match sum_type {
            Some(SumType::Int64) => ScalarValue::I64(0),
            Some(SumType::Uint64) => ScalarValue::U64(0),
            None => ScalarValue::F64(0.0),
        });
    }
    if values.is_empty() {
        return Ok(ScalarValue::Null);
    }
    if operation == "sum" {
        if let Some(sum_type) = sum_type {
            let total = values.iter().try_fold(0i128, |total, value| {
                let value = match (sum_type, value) {
                    (SumType::Int64, ScalarValue::I64(value)) => i128::from(*value),
                    (SumType::Uint64, ScalarValue::U64(value)) => i128::from(*value),
                    _ => return Err("integer sum requires matching integer operands".into()),
                };
                total
                    .checked_add(value)
                    .ok_or_else(|| "integer sum exceeds accumulator range".to_string())
            })?;
            return match sum_type {
                SumType::Int64 => i64::try_from(total)
                    .map(ScalarValue::I64)
                    .map_err(|_| "integer sum exceeds int64 range".to_string()),
                SumType::Uint64 => u64::try_from(total)
                    .map(ScalarValue::U64)
                    .map_err(|_| "integer sum exceeds uint64 range".to_string()),
            };
        }
    }
    if operation == "recency" {
        let latest = values
            .iter()
            .map(ScalarValue::epoch_seconds)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .max()
            .expect("non-null timestamp values");
        return Ok(ScalarValue::F64(cutoff.saturating_sub(latest) as f64));
    }
    if operation == "mode" {
        let mut counts = HashMap::<String, (usize, ScalarValue)>::new();
        for value in values {
            let entry = counts
                .entry(value.equality_key())
                .or_insert_with(|| (0, value.clone()));
            entry.0 += 1;
        }
        return Ok(counts
            .into_values()
            .max_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| right.1.stable_sort_key().cmp(&left.1.stable_sort_key()))
            })
            .expect("non-empty mode")
            .1);
    }
    if operation == "min" || operation == "max" {
        let mut best = values[0].clone();
        for value in &values[1..] {
            let ordering = value.compare(&best)?;
            if (operation == "min" && ordering == Ordering::Less)
                || (operation == "max" && ordering == Ordering::Greater)
            {
                best = value.clone();
            }
        }
        return Ok(best);
    }

    let numbers = values
        .iter()
        .map(ScalarValue::numeric)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut count = 0u64;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut sum = 0.0f64;
    for value in numbers {
        count += 1;
        sum += value;
        let delta = value - mean;
        mean += delta / count as f64;
        m2 += delta * (value - mean);
    }
    Ok(match operation {
        "sum" => ScalarValue::F64(sum),
        "mean" => ScalarValue::F64(mean),
        "var_population" => ScalarValue::F64(m2 / count as f64),
        "std_population" => ScalarValue::F64((m2 / count as f64).sqrt()),
        "var_sample" if count >= 2 => ScalarValue::F64(m2 / (count - 1) as f64),
        "std_sample" if count >= 2 => ScalarValue::F64((m2 / (count - 1) as f64).sqrt()),
        "var_sample" | "std_sample" => ScalarValue::Null,
        _ => return Err(format!("unsupported numeric reduction {operation:?}")),
    })
}

fn group_rows(rows: RowsView<'_>, plan: &GroupPlan, cutoff: i64) -> Result<Vec<OwnedRow>, String> {
    if plan.keys.is_empty() || plan.values.is_empty() {
        return Err("group_by requires keys and values".into());
    }
    let key_columns = plan
        .keys
        .iter()
        .map(|key| rows.values(key))
        .collect::<Result<Vec<_>, _>>()?;
    let mut groups = BTreeMap::<Vec<(String, String)>, Vec<usize>>::new();
    for row in 0..rows.len() {
        let key = key_columns
            .iter()
            .map(|values| values[row].stable_sort_key())
            .collect::<Vec<_>>();
        groups.entry(key).or_default().push(row);
    }
    let value_columns = plan
        .values
        .iter()
        .map(|value| {
            validate_reduction(&value.operation, value.column.as_deref())?;
            if value.sum_type.is_some() && value.operation != "sum" {
                return Err("sum_type is only valid for sum".to_string());
            }
            Ok(match &value.column {
                Some(column) => Some(rows.values(column)?),
                None => None,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut output = Vec::with_capacity(groups.len());
    for indices in groups.into_values() {
        let mut values = BTreeMap::new();
        for (key_index, key) in plan.keys.iter().enumerate() {
            values.insert(key.clone(), key_columns[key_index][indices[0]].clone());
        }
        for (reduction_index, reduction) in plan.values.iter().enumerate() {
            let selected = value_columns[reduction_index]
                .as_ref()
                .map(|column| {
                    indices
                        .iter()
                        .map(|index| OwnedRow {
                            values: BTreeMap::from([(
                                reduction.column.clone().expect("column present"),
                                column[*index].clone(),
                            )]),
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| {
                    vec![
                        OwnedRow {
                            values: BTreeMap::new()
                        };
                        indices.len()
                    ]
                });
            values.insert(
                reduction.name.clone(),
                reduce(
                    RowsView::Owned(&selected),
                    &reduction.operation,
                    reduction.column.as_deref(),
                    cutoff,
                    reduction.sum_type,
                )?,
            );
        }
        output.push(OwnedRow { values });
    }
    Ok(output)
}

fn aggregate_result(
    frame: &MaterializedFrame,
    plan: &AggregatePlan,
    cutoffs: &[i64],
) -> Result<AggregateResult, String> {
    validate_reduction(&plan.operation, plan.column.as_deref())?;
    let mut values = Vec::with_capacity(cutoffs.len());
    for (anchor, cutoff) in cutoffs.iter().copied().enumerate() {
        let frame_view = RowsView::Frame {
            frame,
            start: frame.offsets[anchor],
            stop: frame.offsets[anchor + 1],
        };
        let value = if let Some((first, remaining)) = plan.groups.split_first() {
            let mut grouped = group_rows(frame_view, first, cutoff)?;
            for group in remaining {
                grouped = group_rows(RowsView::Owned(&grouped), group, cutoff)?;
            }
            reduce(
                RowsView::Owned(&grouped),
                &plan.operation,
                plan.column.as_deref(),
                cutoff,
                plan.sum_type,
            )?
        } else {
            reduce(
                frame_view,
                &plan.operation,
                plan.column.as_deref(),
                cutoff,
                plan.sum_type,
            )?
        };
        values.push(value);
    }
    Ok(AggregateResult {
        name: plan.name.clone(),
        values,
        sum_type: plan.sum_type,
    })
}

fn count_frame(
    reader: &GraphIndexReader,
    plan: &FramePlan,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    timings: &mut LengthTimings,
    stats: &mut LengthStats,
) -> Result<Vec<usize>, String> {
    validate_frame_plan(plan, root_type)?;

    let traversal_started = Instant::now();
    let mut rows_by_anchor = Vec::with_capacity(roots.len());
    let mut destination = root_type.to_string();
    let mut final_sort_key = None::<EdgeSortKeyMetadata>;
    for &root in roots {
        let mut frontier = vec![(root, 0)];
        destination = root_type.to_string();
        for step in &plan.path {
            let relation = reader
                .merged_edge_relations()
                .iter()
                .find(|relation| relation.name == step.relation)
                .ok_or_else(|| format!("unknown temporal relation {:?}", step.relation))?;
            if relation.src_type != destination || relation.dst_type != step.destination_type {
                return Err(format!(
                    "relation {:?} is {:?} -> {:?}, not {:?} -> {:?}",
                    step.relation,
                    relation.src_type,
                    relation.dst_type,
                    destination,
                    step.destination_type
                ));
            }
            let table = reader
                .table(&destination)
                .ok_or_else(|| format!("unknown source table {destination:?}"))?;
            let mut next = Vec::new();
            for (node_id, _) in frontier {
                let (destinations, sort_keys) = relation_columns(
                    table.as_ref(),
                    &step.relation,
                    node_id,
                    true,
                    None,
                    &mut stats.edges_visited,
                )?;
                next.extend(destinations.into_iter().zip(sort_keys));
            }
            frontier = next;
            destination = step.destination_type.clone();
            final_sort_key = relation.dest_sort_key.clone();
        }
        if plan.distinct {
            let mut seen = HashSet::new();
            frontier.retain(|(node_id, _)| seen.insert(*node_id));
        }
        stats.rows_considered = stats.rows_considered.saturating_add(frontier.len());
        rows_by_anchor.push(frontier);
    }
    timings.traversal = timings
        .traversal
        .saturating_add(elapsed_ns(traversal_started));

    let filter_started = Instant::now();
    let has_bounds = plan.start_seconds.is_some() || plan.stop_seconds.is_some();
    let mut lengths = rows_by_anchor.iter().map(Vec::len).collect::<Vec<_>>();
    if has_bounds {
        let time_column = plan
            .time_column
            .as_ref()
            .expect("frame validation requires a time column for bounds");
        let sort_decoder =
            matching_time_decoder(final_sort_key.as_ref(), &destination, time_column)
                .filter(|decoder| {
                    rows_by_anchor
                        .iter()
                        .flatten()
                        .all(|&(_, key)| exact_key_time(decoder, key).is_some())
                });
        if let Some(decoder) = sort_decoder {
            for (anchor, rows) in rows_by_anchor.iter().enumerate() {
                let lower = plan
                    .start_seconds
                    .map(|value| cutoffs[anchor].saturating_add(value));
                let upper = plan
                    .stop_seconds
                    .map(|value| cutoffs[anchor].saturating_add(value));
                let mut selected = 0;
                for &(_, raw_sort_key) in rows {
                    let timestamp = exact_key_time(&decoder, raw_sort_key)
                        .expect("ambiguous sort keys select the rowpack path");
                    stats.sort_key_timestamps += 1;
                    if lower.is_none_or(|bound| timestamp > bound)
                        && upper.is_none_or(|bound| timestamp <= bound)
                    {
                        selected += 1;
                    }
                }
                lengths[anchor] = selected;
            }
        } else {
            let table = reader
                .table(&destination)
                .ok_or_else(|| format!("unknown destination table {destination:?}"))?;
            let projection_indices = projection(table.as_ref(), std::slice::from_ref(time_column))?;
            let flat_nodes = rows_by_anchor
                .iter()
                .flatten()
                .map(|(node_id, _)| *node_id)
                .collect::<Vec<_>>();
            let decoded = table_rows(table.as_ref(), &flat_nodes, &projection_indices)?;
            let mut row = 0;
            for (anchor, rows) in rows_by_anchor.iter().enumerate() {
                let lower = plan
                    .start_seconds
                    .map(|value| cutoffs[anchor].saturating_add(value));
                let upper = plan
                    .stop_seconds
                    .map(|value| cutoffs[anchor].saturating_add(value));
                let mut selected = 0;
                for _ in rows {
                    let timestamp = scalar_at(decoded.column(0).as_ref(), row)?
                        .epoch_seconds()?
                        .ok_or_else(|| {
                            format!("frame {:?} contains a null timestamp", plan.name)
                        })?;
                    row += 1;
                    stats.rowpack_timestamps += 1;
                    if lower.is_none_or(|bound| timestamp > bound)
                        && upper.is_none_or(|bound| timestamp <= bound)
                    {
                        selected += 1;
                    }
                }
                lengths[anchor] = selected;
            }
        }
    }
    timings.time_filter = timings
        .time_filter
        .saturating_add(elapsed_ns(filter_started));
    stats.frames_counted += 1;
    Ok(lengths)
}

pub(crate) fn execute_lengths(
    reader: &GraphIndexReader,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    plan_json: &str,
) -> Result<LengthResult, String> {
    let total_started = Instant::now();
    if roots.len() != cutoffs.len() {
        return Err(format!(
            "root IDs and cutoffs must have equal length: {} != {}",
            roots.len(),
            cutoffs.len()
        ));
    }
    let plan: QueryPlan = serde_json::from_str(plan_json)
        .map_err(|error| format!("invalid temporal query JSON: {error}"))?;
    let mut frame_plans = HashMap::new();
    for frame in &plan.frames {
        if frame_plans.insert(frame.name.as_str(), frame).is_some() {
            return Err(format!("duplicate temporal frame {:?}", frame.name));
        }
    }

    let mut timings = LengthTimings::default();
    let mut stats = LengthStats::default();
    let mut counted_frames = HashMap::<&str, Vec<usize>>::new();
    for sequence in &plan.sequences {
        if sequence.name.is_empty() {
            return Err("sequence names must be non-empty".into());
        }
        if sequence.limit == Some(0) {
            return Err(format!(
                "sequence {:?} limit must be positive",
                sequence.name
            ));
        }
        if !counted_frames.contains_key(sequence.frame.as_str()) {
            let frame = frame_plans.get(sequence.frame.as_str()).ok_or_else(|| {
                format!(
                    "sequence {:?} has unknown frame {:?}",
                    sequence.name, sequence.frame
                )
            })?;
            let lengths = count_frame(
                reader,
                frame,
                root_type,
                roots,
                cutoffs,
                &mut timings,
                &mut stats,
            )?;
            counted_frames.insert(sequence.frame.as_str(), lengths);
        }
    }
    let sequences = plan
        .sequences
        .iter()
        .map(|sequence| SequenceLengths {
            name: sequence.name.clone(),
            lengths: counted_frames[sequence.frame.as_str()]
                .iter()
                .map(|&length| sequence.limit.map_or(length, |limit| length.min(limit)))
                .collect(),
        })
        .collect();
    timings.total = elapsed_ns(total_started);
    Ok(LengthResult {
        sequences,
        timings,
        stats,
    })
}

pub(crate) fn execute(
    reader: &GraphIndexReader,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    plan_json: &str,
) -> Result<QueryResult, String> {
    let total_started = Instant::now();
    if roots.len() != cutoffs.len() {
        return Err(format!(
            "root IDs and cutoffs must have equal length: {} != {}",
            roots.len(),
            cutoffs.len()
        ));
    }
    let plan: QueryPlan = serde_json::from_str(plan_json)
        .map_err(|error| format!("invalid temporal query JSON: {error}"))?;
    let mut timings = QueryTimings::default();
    let mut stats = QueryStats::default();
    let mut frames = HashMap::new();
    for frame_plan in &plan.frames {
        if frames.contains_key(&frame_plan.name) {
            return Err(format!("duplicate temporal frame {:?}", frame_plan.name));
        }
        let consumers = plan
            .sequences
            .iter()
            .filter(|sequence| sequence.frame == frame_plan.name)
            .collect::<Vec<_>>();
        // Aggregates need the full frame; multiple consumers may need opposite
        // ends or different limits. Keep their full row selection unchanged.
        let sequence = match consumers.as_slice() {
            [sequence]
                if !plan.aggregates.iter().any(|a| a.frame == frame_plan.name)
                    && sequence.limit.is_some() =>
            {
                Some(*sequence)
            }
            _ => None,
        };
        // Keep the aggregate's complete, identically ordered inputs, but defer
        // sequence-only columns until its final rows have been selected.
        let full_columns = (consumers.len() == 1
            && consumers[0].limit.is_some()
            && plan.aggregates.iter().any(|a| a.frame == frame_plan.name))
        .then(|| {
            let mut columns = HashSet::new();
            for aggregate in plan
                .aggregates
                .iter()
                .filter(|a| a.frame == frame_plan.name)
            {
                if let Some(group) = aggregate.groups.first() {
                    columns.extend(group.keys.iter().map(String::as_str));
                    columns.extend(group.values.iter().filter_map(|v| v.column.as_deref()));
                } else {
                    columns.extend(aggregate.column.as_deref());
                }
            }
            columns
        });
        let frame = materialize_frame(
            reader,
            frame_plan,
            root_type,
            roots,
            cutoffs,
            sequence,
            full_columns
                .as_ref()
                .zip(consumers.first().and_then(|sequence| sequence.limit)),
            &mut timings,
            &mut stats,
        )?;
        frames.insert(frame_plan.name.clone(), frame);
    }

    let sequence_started = Instant::now();
    let sequences = plan
        .sequences
        .iter()
        .map(|sequence| {
            let frame = frames.get(&sequence.frame).ok_or_else(|| {
                format!(
                    "sequence {:?} has unknown frame {:?}",
                    sequence.name, sequence.frame
                )
            })?;
            sequence_result(reader, frame, sequence, &mut stats)
        })
        .collect::<Result<Vec<_>, String>>()?;
    timings.sequence_assembly = elapsed_ns(sequence_started);

    let aggregate_started = Instant::now();
    let aggregates = plan
        .aggregates
        .iter()
        .map(|aggregate| {
            let frame = frames.get(&aggregate.frame).ok_or_else(|| {
                format!(
                    "aggregate {:?} has unknown frame {:?}",
                    aggregate.name, aggregate.frame
                )
            })?;
            aggregate_result(frame, aggregate, cutoffs)
        })
        .collect::<Result<Vec<_>, String>>()?;
    timings.aggregation = elapsed_ns(aggregate_started);
    timings.total = elapsed_ns(total_started);
    Ok(QueryResult {
        sequences,
        aggregates,
        timings,
        stats,
    })
}

impl QueryResult {
    pub(crate) fn into_python(self, py: Python<'_>, model_format: bool) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        let sequences = PyList::empty(py);
        for sequence in self.sequences {
            let item = PyDict::new(py);
            item.set_item("name", sequence.name)?;
            if model_format {
                let offsets = UInt64Array::from(
                    sequence
                        .offsets
                        .into_iter()
                        .map(|value| value as u64)
                        .collect::<Vec<_>>(),
                );
                item.set_item("offsets", offsets.to_data().to_pyarrow(py)?)?;
                item.set_item(
                    "node_ids",
                    UInt64Array::from(sequence.node_ids)
                        .to_data()
                        .to_pyarrow(py)?,
                )?;
                item.set_item(
                    "timestamps",
                    Int64Array::from(sequence.timestamps)
                        .to_data()
                        .to_pyarrow(py)?,
                )?;
            } else {
                item.set_item("offsets", sequence.offsets)?;
                item.set_item("node_ids", sequence.node_ids)?;
                item.set_item("timestamps", sequence.timestamps)?;
            }
            item.set_item("batch", sequence.batch.to_pyarrow(py)?)?;
            sequences.append(item)?;
        }
        output.set_item("sequences", sequences)?;

        let aggregates = PyList::empty(py);
        for aggregate in self.aggregates {
            let item = PyDict::new(py);
            item.set_item("name", aggregate.name)?;
            if model_format {
                let values = scalar_values_to_array(&aggregate.values, aggregate.sum_type)
                    .map_err(DiskgraphConfigError::new_err)?;
                item.set_item("values", values.to_data().to_pyarrow(py)?)?;
            } else {
                let values = PyList::empty(py);
                for value in aggregate.values {
                    value.append_to(&values)?;
                }
                item.set_item("values", values)?;
            }
            aggregates.append(item)?;
        }
        output.set_item("aggregates", aggregates)?;

        let timings = PyDict::new(py);
        timings.set_item("traversal", self.timings.traversal)?;
        timings.set_item("decode_and_select", self.timings.decode_and_select)?;
        timings.set_item("sequence_assembly", self.timings.sequence_assembly)?;
        timings.set_item("aggregation", self.timings.aggregation)?;
        timings.set_item("total", self.timings.total)?;
        output.set_item("timings_ns", timings)?;

        let stats = PyDict::new(py);
        stats.set_item("frames_materialized", self.stats.frames_materialized)?;
        stats.set_item("rows_materialized", self.stats.rows_materialized)?;
        stats.set_item("rows_decoded", self.stats.rows_decoded)?;
        stats.set_item("values_decoded", self.stats.values_decoded)?;
        stats.set_item("edges_visited", self.stats.edges_visited)?;
        output.set_item("stats", stats)?;
        Ok(output.unbind())
    }
}

impl LengthResult {
    pub(crate) fn into_python(self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        let sequences = PyList::empty(py);
        for sequence in self.sequences {
            let item = PyDict::new(py);
            item.set_item("name", sequence.name)?;
            let lengths = UInt64Array::from(
                sequence
                    .lengths
                    .into_iter()
                    .map(|value| value as u64)
                    .collect::<Vec<_>>(),
            );
            item.set_item("lengths", lengths.to_data().to_pyarrow(py)?)?;
            sequences.append(item)?;
        }
        output.set_item("sequences", sequences)?;

        let timings = PyDict::new(py);
        timings.set_item("traversal", self.timings.traversal)?;
        timings.set_item("time_filter", self.timings.time_filter)?;
        timings.set_item("total", self.timings.total)?;
        output.set_item("timings_ns", timings)?;

        let stats = PyDict::new(py);
        stats.set_item("frames_counted", self.stats.frames_counted)?;
        stats.set_item("rows_considered", self.stats.rows_considered)?;
        stats.set_item("edges_visited", self.stats.edges_visited)?;
        stats.set_item("sort_key_timestamps", self.stats.sort_key_timestamps)?;
        stats.set_item("rowpack_timestamps", self.stats.rowpack_timestamps)?;
        output.set_item("stats", stats)?;
        Ok(output.unbind())
    }
}

pub(crate) fn config_error(error: String) -> PyErr {
    DiskgraphConfigError::new_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_sort_keeps_existing_tie_order_for_every_scalar_kind() {
        use rand::{seq::SliceRandom, SeedableRng};
        let keys = [
            ScalarValue::Null,
            ScalarValue::Bool(true),
            ScalarValue::Bool(false),
            ScalarValue::I64(-20),
            ScalarValue::I64(2),
            ScalarValue::I64(10),
            ScalarValue::U64(u64::MAX),
            ScalarValue::F64(-0.0),
            ScalarValue::F64(f64::NAN),
            ScalarValue::F64(f64::INFINITY),
            ScalarValue::String("café".into()),
            ScalarValue::String("\"a".into()),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(93);
        for clock in 0..3 {
            for _ in 0..100 {
                let mut actual = (0..60)
                    .map(|row| {
                        (
                            row,
                            match clock {
                                0 => None,
                                1 => Some(0),
                                _ => Some((row % 7) as i64),
                            },
                            keys[row % keys.len()].clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                actual.shuffle(&mut rng);
                let mut expected = actual
                    .iter()
                    .map(|(row, time, key)| (*row, *time, key.stable_sort_key()))
                    .collect::<Vec<_>>();
                expected
                    .sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.2.cmp(&right.2)));
                sort_frame_rows(&mut actual);
                assert_eq!(
                    actual.iter().map(|row| row.0).collect::<Vec<_>>(),
                    expected.iter().map(|row| row.0).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn stable_sort_key_spells_values_like_python_repr() {
        for (value, expected) in [
            (1e16, "1e+16"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (123.0, "123.0"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (1e300, "1e+300"),
            (5e-324, "5e-324"),
            (0.0001, "0.0001"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e-4 * 0.99, "9.900000000000001e-05"),
            (f64::MAX, "1.7976931348623157e+308"),
            (0.1, "0.1"),
        ] {
            assert_eq!(python_float_repr(value), expected);
        }
        for (value, expected) in [
            ("a", "'a'"),
            ("it's", "\"it's\""),
            ("say \"hi\"", "'say \"hi\"'"),
            ("both ' and \"", "'both \\' and \"'"),
            ("back\\slash", "'back\\\\slash'"),
            ("tab\t", "'tab\\t'"),
            ("nl\n", "'nl\\n'"),
            ("cr\r", "'cr\\r'"),
            ("\x7f", "'\\x7f'"),
            ("\u{80}", "'\\x80'"),
            ("\u{a0}", "'\\xa0'"),
            ("\u{ad}", "'\\xad'"),
            ("é", "'é'"),
            ("\u{200b}", "'\\u200b'"),
            ("\u{2028}", "'\\u2028'"),
            ("\u{3000}", "'\\u3000'"),
            ("\u{e0001}", "'\\U000e0001'"),
            ("\u{10ffff}", "'\\U0010ffff'"),
            ("😀", "'😀'"),
            ("", "''"),
            (" ", "' '"),
        ] {
            assert_eq!(python_str_repr(value), expected);
        }
    }

    fn row(value: Option<ScalarValue>) -> OwnedRow {
        OwnedRow {
            values: BTreeMap::from([("value".to_string(), value.unwrap_or(ScalarValue::Null))]),
        }
    }

    #[test]
    fn reduction_empty_and_null_semantics_are_distinct() {
        let empty = Vec::<OwnedRow>::new();
        assert!(matches!(
            reduce(RowsView::Owned(&empty), "sum", Some("value"), 0, None).unwrap(),
            ScalarValue::F64(0.0)
        ));
        assert!(matches!(
            reduce(
                RowsView::Owned(&empty),
                "sum",
                Some("value"),
                0,
                Some(SumType::Int64)
            )
            .unwrap(),
            ScalarValue::I64(0)
        ));
        assert!(matches!(
            reduce(
                RowsView::Owned(&[row(None)]),
                "sum",
                Some("value"),
                0,
                Some(SumType::Int64)
            )
            .unwrap(),
            ScalarValue::Null
        ));
        assert!(matches!(
            reduce(
                RowsView::Owned(&empty),
                "count_distinct",
                Some("value"),
                0,
                None
            )
            .unwrap(),
            ScalarValue::I64(0)
        ));
    }

    #[test]
    fn integer_sum_preserves_large_values_and_checks_overflow() {
        let rows = [
            row(Some(ScalarValue::I64(1 << 53))),
            row(Some(ScalarValue::I64(1))),
        ];
        assert!(matches!(
            reduce(RowsView::Owned(&rows), "sum", Some("value"), 0, Some(SumType::Int64)).unwrap(),
            ScalarValue::I64(value) if value == (1 << 53) + 1
        ));
        let output = scalar_values_to_array(
            &[ScalarValue::I64(0), ScalarValue::I64((1 << 53) + 1)],
            Some(SumType::Int64),
        )
        .unwrap();
        assert_eq!(output.data_type(), &arrow_schema::DataType::Int64);
        assert_eq!(
            output
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(1),
            (1 << 53) + 1
        );
        for kind in [SumType::Int64, SumType::Uint64] {
            let output = scalar_values_to_array(&[ScalarValue::Null], Some(kind)).unwrap();
            assert_eq!(output.null_count(), 1);
            assert_eq!(
                output.data_type(),
                &match kind {
                    SumType::Int64 => arrow_schema::DataType::Int64,
                    SumType::Uint64 => arrow_schema::DataType::UInt64,
                }
            );
        }
        let overflowing = [
            row(Some(ScalarValue::I64(i64::MAX))),
            row(Some(ScalarValue::I64(1))),
        ];
        assert!(reduce(
            RowsView::Owned(&overflowing),
            "sum",
            Some("value"),
            0,
            Some(SumType::Int64)
        )
        .is_err());
        let unsigned = [row(Some(ScalarValue::U64(u64::MAX)))];
        assert!(matches!(
            reduce(
                RowsView::Owned(&unsigned),
                "sum",
                Some("value"),
                0,
                Some(SumType::Uint64)
            )
            .unwrap(),
            ScalarValue::U64(u64::MAX)
        ));
        let floats = [
            row(Some(ScalarValue::F64(1.25))),
            row(Some(ScalarValue::F64(2.5))),
        ];
        assert!(matches!(
            reduce(RowsView::Owned(&floats), "sum", Some("value"), 0, None).unwrap(),
            ScalarValue::F64(3.75)
        ));
    }

    #[test]
    fn welford_statistics_and_mode_match_contract() {
        let rows = vec![
            row(Some(ScalarValue::F64(10.0))),
            row(Some(ScalarValue::F64(20.0))),
            row(Some(ScalarValue::F64(30.0))),
        ];
        assert!(matches!(
            reduce(RowsView::Owned(&rows), "var_population", Some("value"), 0, None).unwrap(),
            ScalarValue::F64(value) if (value - 66.66666666666667).abs() < 1e-10
        ));
        assert!(matches!(
            reduce(RowsView::Owned(&rows), "std_sample", Some("value"), 0, None).unwrap(),
            ScalarValue::F64(value) if (value - 10.0).abs() < 1e-10
        ));
        let tied = vec![
            row(Some(ScalarValue::String("b".into()))),
            row(Some(ScalarValue::String("a".into()))),
        ];
        assert!(matches!(
            reduce(RowsView::Owned(&tied), "mode", Some("value"), 0, None).unwrap(),
            ScalarValue::String(value) if value == "a"
        ));
    }
}
