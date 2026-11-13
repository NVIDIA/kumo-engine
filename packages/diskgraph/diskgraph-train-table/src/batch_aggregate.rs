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
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use crate::ttgen_executor::{execute_numeric_group, NumericAggregatePolicy, NumericRows};
use crate::ttgen_resolve::{resolve_aggregate_input, resolve_named_path, ResolvedEdge};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, NullArray, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_array::{RecordBatch, RecordBatchOptions};
use diskgraph_core::metadata::EdgeSortKeyMetadata;
use diskgraph_core::sortkey::SortKeyDecoder;
use diskgraph_core::time::{
    add_calendar_months, convert_floor, convert_trunc, Abs, Seconds, TimeUnit,
};
use diskgraph_core::tt_builder::{
    AggFunc, AggregateFrame as FramePlan, AggregateGroup as GroupPlan, AggregateSumType as SumType,
    ScalarAggregate as AggregatePlan,
};
use diskgraph_engine::edge_unpacker::EdgeUnpacker;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use rayon::prelude::*;
use regex::Regex;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryPlan {
    frames: Vec<FramePlan>,
    #[serde(default)]
    sequences: Vec<SequencePlan>,
    #[serde(default)]
    aggregates: Vec<AggregatePlan>,
    #[serde(skip)]
    prepared_frames: HashMap<String, PreparedFrame>,
    #[serde(skip)]
    window_groups: Option<Vec<(FramePlan, Vec<FramePlan>)>>,
    #[serde(skip)]
    numeric: HashMap<String, Vec<crate::ttgen_resolve::ResolvedAggregate<Seconds>>>,
    #[serde(skip)]
    selection: Option<PreparedRowSelection>,
}

struct PreparedFrame {
    path: FramePath,
    table: Arc<TableReader>,
    primary_key: Option<String>,
    columns: Vec<String>,
    projection: Vec<usize>,
}

impl PreparedFrame {
    fn new(reader: &GraphIndexReader, plan: &FramePlan, root: &str) -> Result<Self, String> {
        validate_frame_plan(plan, root)?;
        let path = frame_path(reader, plan, root)?;
        let table = reader.table(&path.1).ok_or("unknown frame destination")?;
        let primary_key = table.primary_key_map().map(|m| m.column.clone());
        let mut columns = plan.columns.clone();
        for column in plan.time_column.iter().chain(primary_key.iter()) {
            if !columns.contains(column) {
                columns.push(column.clone());
            }
        }
        let projection = projection(&table, &columns)?;
        Ok(Self {
            path,
            table,
            primary_key,
            columns,
            projection,
        })
    }
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

fn sum_zero(value: SumType) -> ScalarValue {
    match value {
        SumType::Int64 => ScalarValue::I64(0),
        SumType::Uint64 => ScalarValue::U64(0),
    }
}

#[derive(Clone, Debug)]
pub enum ScalarValue {
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
}

/// `fallback` fixes the output type when no value is non-null.
pub fn scalar_values_to_array(
    values: &[ScalarValue],
    fallback: Option<ScalarValue>,
) -> Result<ArrayRef, String> {
    let first = values
        .iter()
        .find(|value| !matches!(value, ScalarValue::Null))
        .cloned()
        .or(fallback);
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
    clock: TimeUnit,
    name: String,
    destination: String,
    deferred_columns: HashMap<String, usize>,
    positions: HashMap<String, usize>,
    batch: RecordBatch,
    node_ids: Vec<u64>,
    timestamps: Vec<Option<i64>>,
    offsets: Vec<usize>,
}

pub struct SequenceResult {
    pub clock: TimeUnit,
    pub name: String,
    pub offsets: Vec<usize>,
    pub node_ids: Vec<u64>,
    pub timestamps: Vec<Option<i64>>,
    pub batch: RecordBatch,
}

pub struct AggregateResult {
    pub name: String,
    pub values: Vec<ScalarValue>,
    pub kind: Option<ScalarValue>,
}

#[derive(Default)]
pub struct QueryTimings {
    pub traversal: u64,
    pub decode_and_select: u64,
    pub sequence_assembly: u64,
    pub aggregation: u64,
    pub total: u64,
}

#[derive(Default)]
pub struct QueryStats {
    pub frames_materialized: usize,
    pub rows_materialized: usize,
    pub rows_decoded: usize,
    pub values_decoded: usize,
    pub edges_visited: u64,
}

#[derive(Default)]
pub struct LengthTimings {
    pub traversal: u64,
    pub time_filter: u64,
    pub total: u64,
}

#[derive(Default)]
pub struct LengthStats {
    pub frames_counted: usize,
    pub rows_considered: usize,
    pub edges_visited: u64,
    pub sort_key_timestamps: usize,
    pub rowpack_timestamps: usize,
}

pub struct SequenceLengths {
    pub name: String,
    pub lengths: Vec<usize>,
}

pub struct LengthResult {
    pub sequences: Vec<SequenceLengths>,
    pub timings: LengthTimings,
    pub stats: LengthStats,
}

pub struct QueryResult {
    pub sequences: Vec<SequenceResult>,
    pub aggregates: Vec<AggregateResult>,
    pub timings: QueryTimings,
    pub stats: QueryStats,
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
    if projection_indices.is_empty() {
        return RecordBatch::try_new_with_options(
            Arc::new(arrow_schema::Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(node_ids.len())),
        )
        .map_err(|error| error.to_string());
    }
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
            if keys
                .iter()
                .any(|&key| exact_key_time(decoder, key).is_none())
            {
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
/// Bounds are compared in the index clock, never against unconverted edge ticks.
fn frame_bounds(
    plan: &FramePlan,
    cutoff_seconds: i64,
    clock: TimeUnit,
) -> Result<(Option<i64>, Option<i64>), String> {
    let unit = plan.window_unit.unwrap_or(TimeUnit::Seconds);
    let bound = |offset: Option<i64>| -> Result<Option<i64>, String> {
        let Some(offset) = offset else {
            return Ok(None);
        };
        if offset == crate::ttgen_resolve::UNBOUNDED_LOWER_OFFSET {
            return Ok(None);
        }
        if matches!(unit, TimeUnit::Months | TimeUnit::Years) {
            let months = if unit == TimeUnit::Years {
                offset.checked_mul(12)
            } else {
                Some(offset)
            }
            .and_then(|v| i32::try_from(v).ok())
            .ok_or("calendar window exceeds supported range")?;
            let seconds = add_calendar_months(Abs::<Seconds>::from_raw(cutoff_seconds), months)
                .ok_or("calendar window exceeds supported clock")?
                .raw();
            Ok(Some(convert_floor(seconds, TimeUnit::Seconds, clock)))
        } else {
            let cutoff = convert_trunc(cutoff_seconds, TimeUnit::Seconds, clock);
            let delta = convert_floor(offset, unit, clock);
            Ok(Some(cutoff.saturating_add(delta)))
        }
    };
    Ok((bound(plan.start_seconds)?, bound(plan.stop_seconds)?))
}

fn row_clock_time(array: &dyn Array, row: usize, clock: TimeUnit) -> Result<Option<i64>, String> {
    if array.is_null(row) {
        return Ok(None);
    }
    macro_rules! time {
        ($ty:ty,$unit:expr) => {
            if let Some(a) = array.as_any().downcast_ref::<$ty>() {
                return Ok(Some(convert_floor(a.value(row) as i64, $unit, clock)));
            }
        };
    }
    time!(TimestampSecondArray, TimeUnit::Seconds);
    time!(TimestampMillisecondArray, TimeUnit::Milliseconds);
    time!(TimestampMicrosecondArray, TimeUnit::Microseconds);
    time!(TimestampNanosecondArray, TimeUnit::Nanoseconds);
    time!(Date32Array, TimeUnit::Days);
    time!(Date64Array, TimeUnit::Milliseconds);
    scalar_at(array, row)?.epoch_seconds()
}

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

type FramePath = (Vec<ResolvedEdge>, String);

fn frame_path(
    reader: &GraphIndexReader,
    plan: &FramePlan,
    root_type: &str,
) -> Result<FramePath, String> {
    if !plan.column_path.is_empty() {
        let path = crate::ttgen_resolve::resolve_path(&plan.column_path, reader)
            .map_err(|e| e.to_string())?;
        let destination = plan.column_path.last().unwrap().dst_table.clone();
        return Ok((path, destination));
    }

    let names = plan
        .path
        .iter()
        .map(|step| (step.relation.as_str(), step.destination_type.as_str()))
        .collect::<Vec<_>>();
    let path = resolve_named_path(reader, root_type, &names).map_err(|e| e.to_string())?;
    let destination = path
        .last()
        .map(|edge| edge.dst_table.node_type())
        .unwrap_or(root_type)
        .to_string();
    Ok((path, destination))
}

fn validate_sequence_plan(plan: &SequencePlan) -> Result<(), String> {
    if plan.name.is_empty() {
        return Err("sequence names must be non-empty".into());
    }
    if plan.limit == Some(0) {
        return Err(format!("sequence {:?} limit must be positive", plan.name));
    }
    if plan.columns.is_empty() {
        return Err(format!(
            "sequence {:?} must project at least one column",
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
    RecordBatch::try_new_with_options(
        Arc::new(arrow_schema::Schema::new(fields)),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(rows.len())),
    )
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

#[allow(clippy::too_many_arguments)]
fn materialize_frame(
    reader: &GraphIndexReader,
    plan: &FramePlan,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    sequence: Option<&SequencePlan>,
    full_columns: Option<(&HashSet<&str>, usize)>,
    prepared: Option<&PreparedFrame>,
    timings: &mut QueryTimings,
    stats: &mut QueryStats,
) -> Result<MaterializedFrame, String> {
    let resolved;
    let prepared = match prepared {
        Some(prepared) => prepared,
        None => {
            resolved = PreparedFrame::new(reader, plan, root_type)?;
            &resolved
        }
    };
    let (hops, destination) = &prepared.path;

    let traversal_started = Instant::now();
    let mut rows_by_anchor = Vec::with_capacity(roots.len());
    for (anchor, &root) in roots.iter().enumerate() {
        let mut frontier = vec![root];
        for (hop, edge) in hops.iter().enumerate() {
            let table = &edge.src_table;
            let relation = &table.relations()[edge.relation_idx];
            let decoder = if hop + 1 == hops.len()
                && (plan.start_seconds.is_some() || plan.stop_seconds.is_some())
            {
                plan.time_column.as_deref().and_then(|clock| {
                    matching_time_decoder(
                        relation.dest_sort_key.as_ref(),
                        edge.dst_table.node_type(),
                        clock,
                    )
                })
            } else {
                None
            };
            let (lower, upper) =
                frame_bounds(plan, cutoffs[anchor], reader.metadata().timestamp_unit)?;
            let window = decoder
                .as_ref()
                .map(|decoder| (decoder, lower, upper, plan.name.as_str()));
            let mut next = Vec::new();
            for node_id in frontier {
                next.extend(relation_destinations(
                    table.as_ref(),
                    &table.relations()[edge.relation_idx].name,
                    node_id,
                    window,
                    sequence.filter(|_| hops.len() == 1 && window.is_some()),
                    plan.distinct,
                    stats,
                )?);
            }
            frontier = next;
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
    let table = &prepared.table;
    let primary_key = &prepared.primary_key;
    let columns = prepared.columns.clone();
    let projection_indices = prepared.projection.clone();
    let mut deferred_columns = HashMap::new();
    // A capped history-only frame must defer payload until selection, even
    // without a matching edge clock. Retain the selectivity heuristic for
    // frames whose aggregates also require a complete input projection.
    let full_columns = full_columns.filter(|(_, limit)| {
        let total = rows_by_anchor.iter().map(Vec::len).sum::<usize>();
        let selected = rows_by_anchor
            .iter()
            .map(|rows| rows.len().min(*limit))
            .sum::<usize>();
        total > 0 && selected < total && (sequence.is_some() || selected <= total / 4)
    });
    let (columns, projection_indices) = if let Some((required, _)) = full_columns {
        columns
            .into_iter()
            .zip(projection_indices)
            .filter(|(name, index)| {
                let keep = required.contains(name.as_str())
                    || primary_key.as_ref() == Some(name)
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
    if !projection_indices.is_empty() {
        stats.rows_decoded += flat_nodes.len();
    }
    stats.values_decoded += flat_nodes.len() * projection_indices.len();
    let decoded = table_rows(table.as_ref(), &flat_nodes, &projection_indices)?;

    let mut input_offsets = Vec::with_capacity(rows_by_anchor.len() + 1);
    input_offsets.push(0usize);
    for rows in &rows_by_anchor {
        input_offsets.push(input_offsets.last().copied().unwrap_or(0) + rows.len());
    }
    let primary_key_position = primary_key.as_ref().map(|name| positions[name]);
    let time_position = plan.time_column.as_ref().map(|column| positions[column]);
    let mut selected_rows = Vec::<u64>::new();
    let mut node_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut offsets = Vec::with_capacity(roots.len() + 1);
    offsets.push(0);
    for anchor in 0..roots.len() {
        let (lower, upper) = frame_bounds(plan, cutoffs[anchor], reader.metadata().timestamp_unit)?;
        let mut anchor_rows = Vec::<(usize, Option<i64>, ScalarValue)>::new();
        for (row, &node_id) in flat_nodes
            .iter()
            .enumerate()
            .take(input_offsets[anchor + 1])
            .skip(input_offsets[anchor])
        {
            let timestamp = match time_position {
                Some(position) => Some(
                    row_clock_time(
                        decoded.column(position).as_ref(),
                        row,
                        reader.metadata().timestamp_unit,
                    )?
                    .ok_or_else(|| format!("frame {:?} contains a null timestamp", plan.name))?,
                ),
                None => None,
            };
            if timestamp.is_some_and(|value| {
                lower.is_some_and(|bound| value <= bound)
                    || upper.is_some_and(|bound| value > bound)
            }) {
                continue;
            }
            let key = match primary_key_position {
                Some(position) => scalar_at(decoded.column(position).as_ref(), row)?,
                None => ScalarValue::U64(node_id),
            };
            anchor_rows.push((row, timestamp, key));
        }
        if !plan.preserve_traversal_order {
            sort_frame_rows(&mut anchor_rows);
        }
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
        clock: reader.metadata().timestamp_unit,
        name: plan.name.clone(),
        destination: destination.clone(),
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
        clock: frame.clock,
        name: plan.name.clone(),
        offsets,
        node_ids,
        timestamps,
        batch,
    })
}

type OwnedRow = crate::ttgen_row_aggregate::AggregateRow<ScalarValue>;

enum RowsView<'a> {
    Frame {
        frame: &'a MaterializedFrame,
        start: usize,
        stop: usize,
    },
    Owned(&'a [OwnedRow]),
    Selected { frame: &'a MaterializedFrame, rows: Vec<usize> },
}

impl RowsView<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Frame { start, stop, .. } => stop - start,
            Self::Owned(rows) => rows.len(),
            Self::Selected { rows, .. } => rows.len(),
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
            Self::Selected { frame, rows } => {
                let mut values = Vec::with_capacity(rows.len());
                for &row in rows {
                    values.extend(RowsView::Frame { frame, start: row, stop: row + 1 }.values(column)?);
                }
                Ok(values)
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

impl crate::ttgen_row_aggregate::AggregateRows<ScalarValue> for RowsView<'_> {
    fn len(&self) -> usize {
        RowsView::len(self)
    }
    fn values(&self, column: &str) -> Result<Vec<ScalarValue>, String> {
        RowsView::values(self, column)
    }
}
impl crate::ttgen_row_aggregate::AggregateScalar for ScalarValue {
    fn null() -> Self {
        Self::Null
    }
    fn from_i64(v: i64) -> Self {
        Self::I64(v)
    }
    fn from_u64(v: u64) -> Self {
        Self::U64(v)
    }
    fn from_f64(v: f64) -> Self {
        Self::F64(v)
    }
    fn floating(&self) -> Option<f64> {
        match self { Self::F64(value) => Some(*value), _ => None }
    }
    fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    fn integer(&self, unsigned: bool) -> Result<i128, String> {
        match (unsigned, self) {
            (true, Self::U64(v)) => Ok(i128::from(*v)),
            (false, Self::I64(v)) => Ok(i128::from(*v)),
            _ => Err("integer sum requires matching integer operands".into()),
        }
    }
    fn numeric(&self) -> Result<Option<f64>, String> {
        ScalarValue::numeric(self)
    }
    fn epoch_seconds(&self) -> Result<Option<i64>, String> {
        ScalarValue::epoch_seconds(self)
    }
    fn equality_key(&self) -> String {
        ScalarValue::equality_key(self)
    }
    fn stable_sort_key(&self) -> (String, String) {
        ScalarValue::stable_sort_key(self)
    }
    fn compare(&self, other: &Self) -> Result<Ordering, String> {
        ScalarValue::compare(self, other)
    }
}

fn validate_reduction(operation: &str, column: Option<&str>) -> Result<(), String> {
    crate::ttgen_row_aggregate::validate_scalar_reduction(operation, column)
}

fn reduce(
    rows: RowsView<'_>,
    operation: &str,
    column: Option<&str>,
    cutoff: i64,
    sum_type: Option<SumType>,
    semantics: diskgraph_core::tt_builder::ReductionSemantics,
) -> Result<ScalarValue, String> {
    crate::ttgen_row_aggregate::reduce_rows(
        &rows,
        &AggregatePlan {
            name: String::new(),
            frame: String::new(),
            groups: Vec::new(),
            operation: operation.into(),
            column: column.map(str::to_string),
            sum_type,
            semantics,
            row_filter: None,
            limit: None,
            target_entity_table: None,
        },
        cutoff,
    )
}

fn group_rows(rows: RowsView<'_>, plan: &GroupPlan, cutoff: i64) -> Result<Vec<OwnedRow>, String> {
    crate::ttgen_row_aggregate::group_rows(&rows, plan, cutoff)
}

fn aggregate_result(
    frame: &MaterializedFrame,
    plan: &AggregatePlan,
    cutoffs: &[i64],
    roots: &[u64],
    selection: Option<&PreparedRowSelection>,
) -> Result<AggregateResult, String> {
    validate_reduction(&plan.operation, plan.column.as_deref())?;
    let mut values = Vec::with_capacity(cutoffs.len());
    for (anchor, cutoff) in cutoffs.iter().copied().enumerate() {
        let frame_view = if let Some((selection, operation)) = selection.and_then(|selection| {
            selection.operations.iter().find(|(name, _)| name == &plan.name)
                .map(|(_, operation)| (selection, operation))
        }) {
            let columns = selection.entity_col_indices.iter().map(|(_, column)| {
                crate::ttgen_traversal::read_column_value(&selection.entity_table, roots[anchor], *column)
            }).collect::<Vec<_>>();
            let table = &operation.path.last().ok_or("empty scalar selection path")?.dst_table;
            let rows = (frame.offsets[anchor]..frame.offsets[anchor + 1]).filter(|&row| {
                operation.row_filter.as_ref().is_none_or(|filter| {
                    crate::ttgen_filter::eval_row_filter(filter, frame.node_ids[row], table, &columns)
                })
            }).collect();
            RowsView::Selected { frame, rows }
        } else {
            RowsView::Frame { frame, start: frame.offsets[anchor], stop: frame.offsets[anchor + 1] }
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
                plan.semantics,
            )?
        } else {
            reduce(
                frame_view,
                &plan.operation,
                plan.column.as_deref(),
                cutoff,
                plan.sum_type,
                plan.semantics,
            )?
        };
        values.push(value);
    }
    Ok(AggregateResult {
        name: plan.name.clone(),
        values,
        kind: aggregate_kind(frame, plan),
    })
}

/// The scalar kind an aggregate produces, independent of its values.
fn aggregate_kind(frame: &MaterializedFrame, plan: &AggregatePlan) -> Option<ScalarValue> {
    use arrow_schema::DataType;
    match plan.operation.as_str() {
        "min" | "max" if plan.semantics.all_null_sum_is_zero && !plan.semantics.keep_first_equal => {
            Some(ScalarValue::F64(0.0))
        }
        "count" | "count_distinct" => Some(ScalarValue::I64(0)),
        "sum" => Some(plan.sum_type.map_or(ScalarValue::F64(0.0), sum_zero)),
        // Grouped inputs are reduction outputs, not frame columns.
        "min" | "max" | "mode" if plan.groups.is_empty() => {
            let column = frame.positions.get(plan.column.as_deref()?)?;
            match frame.batch.schema().field(*column).data_type() {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Timestamp(..)
                | DataType::Date32
                | DataType::Date64 => Some(ScalarValue::I64(0)),
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                    Some(ScalarValue::U64(0))
                }
                DataType::Float32 | DataType::Float64 => Some(ScalarValue::F64(0.0)),
                DataType::Boolean => Some(ScalarValue::Bool(false)),
                DataType::Utf8 | DataType::LargeUtf8 => Some(ScalarValue::String(String::new())),
                _ => None,
            }
        }
        "min" | "max" | "mode" => None,
        _ => Some(ScalarValue::F64(0.0)),
    }
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
    let (hops, destination) = frame_path(reader, plan, root_type)?;
    let final_sort_key = hops.last().and_then(|edge| {
        edge.src_table.relations()[edge.relation_idx]
            .dest_sort_key
            .clone()
    });

    let traversal_started = Instant::now();
    let mut rows_by_anchor = Vec::with_capacity(roots.len());
    for &root in roots {
        let mut frontier = vec![(root, 0)];
        for edge in &hops {
            let table = &edge.src_table;
            let mut next = Vec::new();
            for (node_id, _) in frontier {
                let (destinations, sort_keys) = relation_columns(
                    table.as_ref(),
                    &table.relations()[edge.relation_idx].name,
                    node_id,
                    true,
                    None,
                    &mut stats.edges_visited,
                )?;
                next.extend(destinations.into_iter().zip(sort_keys));
            }
            frontier = next;
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
            matching_time_decoder(final_sort_key.as_ref(), &destination, time_column).filter(
                |decoder| {
                    rows_by_anchor
                        .iter()
                        .flatten()
                        .all(|&(_, key)| exact_key_time(decoder, key).is_some())
                },
            );
        if let Some(decoder) = sort_decoder {
            for (anchor, rows) in rows_by_anchor.iter().enumerate() {
                let (lower, upper) =
                    frame_bounds(plan, cutoffs[anchor], reader.metadata().timestamp_unit)?;
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
                let (lower, upper) =
                    frame_bounds(plan, cutoffs[anchor], reader.metadata().timestamp_unit)?;
                let mut selected = 0;
                for _ in rows {
                    let timestamp = row_clock_time(
                        decoded.column(0).as_ref(),
                        row,
                        reader.metadata().timestamp_unit,
                    )?
                    .ok_or_else(|| format!("frame {:?} contains a null timestamp", plan.name))?;
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

pub fn execute_lengths_parallel(
    reader: &GraphIndexReader,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    plan_json: &str,
    pool: &rayon::ThreadPool,
) -> Result<LengthResult, String> {
    if roots.len() != cutoffs.len() || roots.len() < 64 || pool.current_num_threads() == 1 {
        return execute_lengths(reader, root_type, roots, cutoffs, plan_json);
    }
    let started = Instant::now();
    let chunks = roots.len().div_ceil(64).min(pool.current_num_threads() * 2);
    let size = roots.len().div_ceil(chunks);
    let results = pool.install(|| {
        roots
            .par_chunks(size)
            .zip(cutoffs.par_chunks(size))
            .map(|(roots, cutoffs)| execute_lengths(reader, root_type, roots, cutoffs, plan_json))
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut results = results.into_iter();
    let mut output = results.next().expect("nonempty anchor chunks");
    for result in results {
        for (sequence, next) in output.sequences.iter_mut().zip(result.sequences) {
            sequence.lengths.extend(next.lengths);
        }
        output.stats.frames_counted = output.stats.frames_counted.max(result.stats.frames_counted);
        output.stats.rows_considered += result.stats.rows_considered;
        output.stats.edges_visited += result.stats.edges_visited;
        output.stats.sort_key_timestamps += result.stats.sort_key_timestamps;
        output.stats.rowpack_timestamps += result.stats.rowpack_timestamps;
        output.timings.traversal += result.timings.traversal;
        output.timings.time_filter += result.timings.time_filter;
    }
    output.timings.total = elapsed_ns(started);
    Ok(output)
}

pub fn execute_lengths(
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
        validate_sequence_plan(sequence)?;
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

fn prepare_numeric_frame(
    reader: &GraphIndexReader,
    frame: &FramePlan,
    root_type: &str,
    aggregates: &[&AggregatePlan],
) -> Result<Option<Vec<crate::ttgen_resolve::ResolvedAggregate<Seconds>>>, String> {
    if frame.path.len().max(frame.column_path.len()) != 1
        || aggregates.is_empty()
        || reader.metadata().timestamp_unit != TimeUnit::Seconds
        || frame.window_unit.is_some_and(|u| u != TimeUnit::Seconds)
        || frame.start_seconds.is_some() && frame.stop_seconds.is_none()
    {
        return Ok(None);
    }
    validate_frame_plan(frame, root_type)?;
    let (_, destination) = frame_path(reader, frame, root_type)?;
    let table = reader
        .table(&destination)
        .ok_or_else(|| format!("unknown destination table {destination:?}"))?;
    let Some(primary_key) = table.primary_key_map() else {
        return Ok(None);
    };
    let mut requested = frame.columns.clone();
    requested.push(primary_key.column.clone());
    requested.extend(frame.time_column.iter().cloned());
    projection(table.as_ref(), &requested)?;
    let mut operations = Vec::new();
    for (index, aggregate) in aggregates.iter().enumerate() {
        let function = match aggregate.operation.as_str() {
            "count" => AggFunc::Count,
            "sum" => AggFunc::Sum,
            "mean" => AggFunc::Avg,
            "min" => AggFunc::Min,
            "max" => AggFunc::Max,
            _ => return Ok(None),
        };
        if aggregate.row_filter.is_some() || aggregate.limit.is_some() || aggregate.target_entity_table.is_some()
            || !aggregate.groups.is_empty()
            || aggregate.sum_type.is_some()
            || aggregate
                .column
                .as_ref()
                .is_some_and(|column| function != AggFunc::Count && !frame.columns.contains(column))
        {
            return Ok(None);
        }
        let (path, _) = frame_path(reader, frame, root_type)?;
        let window = frame.stop_seconds.map(|hi| {
            (
                frame
                    .start_seconds
                    .unwrap_or(crate::ttgen_resolve::UNBOUNDED_LOWER_OFFSET),
                hi,
            )
        });
        let operation = match resolve_aggregate_input::<Seconds>(
            index,
            path,
            function,
            aggregate
                .column
                .as_deref()
                .filter(|_| function != AggFunc::Count),
            window.as_ref(),
            Some(&TimeUnit::Seconds),
        ) {
            Ok(operation) => operation,
            Err(_) => return Ok(None),
        };
        operations.push(operation);
    }
    Ok(Some(operations))
}

/// Lower the existing fluent query to the shared post-resolution PQL IR.
/// The row selector remains the established exact-time/PK-ordered policy.
#[allow(clippy::too_many_arguments)]
fn shared_numeric_frame(
    reader: &GraphIndexReader,
    frame: &FramePlan,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    aggregates: &[&AggregatePlan],
    timings: &mut QueryTimings,
    stats: &mut QueryStats,
    retained_frame: &mut Option<MaterializedFrame>,
    prepared: &QueryPlan,
    predecoded: Option<MaterializedFrame>,
) -> Result<Option<Vec<AggregateResult>>, String> {
    *retained_frame = predecoded;
    let resolved;
    let operations = if let Some(operations) = prepared.numeric.get(&frame.name) {
        operations
    } else if prepared.prepared_frames.contains_key(&frame.name) {
        return Ok(None);
    } else {
        resolved = prepare_numeric_frame(reader, frame, root_type, aggregates)?;
        let Some(operations) = &resolved else {
            return Ok(None);
        };
        operations
    };
    let references = operations.iter().collect::<Vec<_>>();
    let timestamps = cutoffs
        .iter()
        .copied()
        .map(Abs::from_raw)
        .collect::<Vec<_>>();
    let mut selection_times = QueryTimings::default();
    let mut selection_stats = QueryStats::default();
    let mut selected: Option<MaterializedFrame> = retained_frame.take();
    let mut anchor = 0;
    let started = Instant::now();
    let result = execute_numeric_group(
        &references,
        roots,
        &timestamps,
        if aggregates
            .iter()
            .all(|a| a.semantics.all_null_sum_is_zero && !a.semantics.keep_first_equal)
        {
            NumericAggregatePolicy::Pql
        } else {
            NumericAggregatePolicy::TemporalQuery
        },
        |_, _, _| {
            if selected.is_none() {
                selected = Some(
                    materialize_frame(
                        reader,
                        frame,
                        root_type,
                        roots,
                        cutoffs,
                        None,
                        None,
                        prepared.prepared_frames.get(&frame.name),
                        &mut selection_times,
                        &mut selection_stats,
                    )
                    .map_err(crate::errors::TTGenError::InvalidPlan)?,
                );
            }
            let frame = selected.as_ref().unwrap();
            let mut columns = Vec::new();
            for (operation, aggregate) in operations.iter().zip(aggregates) {
                if let Some(column) = operation.target_col_idx {
                    if !columns.iter().any(|(index, _)| *index == column) {
                        let name = aggregate.column.as_ref().unwrap();
                        columns.push((column, frame.batch.column(frame.positions[name]).clone()));
                    }
                }
            }
            let rows = frame.offsets[anchor]..frame.offsets[anchor + 1];
            anchor += 1;
            Ok(NumericRows::Projected { columns, rows })
        },
    )
    .map_err(|error| error.to_string())?;
    timings.traversal = timings.traversal.saturating_add(selection_times.traversal);
    timings.decode_and_select = timings
        .decode_and_select
        .saturating_add(selection_times.decode_and_select);
    timings.aggregation = timings.aggregation.saturating_add(
        elapsed_ns(started).saturating_sub(
            selection_times
                .traversal
                .saturating_add(selection_times.decode_and_select),
        ),
    );
    stats.frames_materialized += selection_stats.frames_materialized;
    stats.rows_materialized += selection_stats.rows_materialized;
    stats.rows_decoded += selection_stats.rows_decoded;
    stats.values_decoded += selection_stats.values_decoded;
    stats.edges_visited += selection_stats.edges_visited;
    let Some(values) = result else {
        *retained_frame = selected;
        return Ok(None);
    };
    let output = aggregates
        .iter()
        .zip(operations.iter())
        .zip(values)
        .map(|((aggregate, operation), values)| {
            let kind = if operation.func == AggFunc::Count {
                ScalarValue::I64(0)
            } else {
                ScalarValue::F64(0.0)
            };
            let values = values
                .into_iter()
                .map(|value| match value {
                    Some(diskgraph_core::tt_builder::Value::Int(value)) => ScalarValue::I64(value),
                    Some(diskgraph_core::tt_builder::Value::Float(value)) => {
                        ScalarValue::F64(value)
                    }
                    None => ScalarValue::Null,
                    _ => unreachable!("shared numeric evaluator returned a nonnumeric value"),
                })
                .collect();
            AggregateResult {
                name: aggregate.name.clone(),
                values,
                kind: Some(kind),
            }
        })
        .collect();
    Ok(Some(output))
}

fn window_groups(plan: &QueryPlan) -> Vec<(FramePlan, Vec<FramePlan>)> {
    use crate::ttgen_row_aggregate::merge_window_bounds;
    let mut pending = plan
        .frames
        .iter()
        .filter(|frame| {
            !plan
                .sequences
                .iter()
                .any(|sequence| sequence.frame == frame.name)
        })
        .collect::<Vec<_>>();
    let mut groups = Vec::new();
    while !pending.is_empty() {
        let first = pending.remove(0);
        let mut union = first.clone();
        let mut group = vec![first];
        let mut position = 0;
        while position < pending.len() {
            let frame = pending[position];
            let bounds = merge_window_bounds(
                (union.start_seconds, union.stop_seconds),
                (frame.start_seconds, frame.stop_seconds),
            );
            if let Some((lower, upper)) = bounds.filter(|_| frame.root_type == union.root_type
                && frame.path == union.path
                && frame.column_path == union.column_path
                && frame.window_unit == union.window_unit
                && frame.preserve_traversal_order == union.preserve_traversal_order
                && frame.time_column == union.time_column
                && frame.distinct == union.distinct)
            {
                union.start_seconds = lower;
                union.stop_seconds = upper;
                for column in &frame.columns {
                    if !union.columns.contains(column) {
                        union.columns.push(column.clone());
                    }
                }
                group.push(pending.remove(position));
                position = 0; // an extended union can now overlap an earlier frame
            } else {
                position += 1;
            }
        }
        if group.len() == 1 {
            continue;
        }
        union.name = format!("__shared_union_{}", groups.len());
        while plan.frames.iter().any(|f| f.name == union.name) {
            union.name.push('_');
        }
        groups.push((union, group.into_iter().cloned().collect()));
    }
    groups
}

/// Reuse one traversal/projection for compatible overlapping aggregate windows.
/// Selection and reducers retain original exact timestamp and chronological/PK
/// policies. Distinctness and relation identity must match before sharing.
fn shared_window_frames(
    reader: &GraphIndexReader,
    plan: &QueryPlan,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    timings: &mut QueryTimings,
    stats: &mut QueryStats,
) -> Result<HashMap<String, MaterializedFrame>, String> {
    let computed;
    let groups = if let Some(groups) = &plan.window_groups {
        groups
    } else {
        computed = window_groups(plan);
        &computed
    };
    let mut output = HashMap::new();
    for (union, group) in groups {
        let frame = materialize_frame(
            reader,
            union,
            root_type,
            roots,
            cutoffs,
            None,
            None,
            plan.prepared_frames.get(&union.name),
            timings,
            stats,
        )?;
        for original in group {
            let mut selected = Vec::new();
            let mut offsets = vec![0];
            for (anchor, &cutoff) in cutoffs.iter().enumerate() {
                let (lower, upper) =
                    frame_bounds(original, cutoff, reader.metadata().timestamp_unit)?;
                for row in frame.offsets[anchor]..frame.offsets[anchor + 1] {
                    let timestamp = frame.timestamps[row];
                    if lower.is_none_or(|bound| timestamp.is_some_and(|t| t > bound))
                        && upper.is_none_or(|bound| timestamp.is_some_and(|t| t <= bound))
                    {
                        selected.push(row);
                    }
                }
                offsets.push(selected.len());
            }
            let indices =
                UInt64Array::from(selected.iter().map(|&row| row as u64).collect::<Vec<_>>());
            let arrays = frame
                .batch
                .columns()
                .iter()
                .map(|column| {
                    arrow::compute::take(column, &indices, None).map_err(|e| e.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let batch = RecordBatch::try_new_with_options(
                frame.batch.schema(),
                arrays,
                &RecordBatchOptions::new().with_row_count(Some(selected.len())),
            )
            .map_err(|e| e.to_string())?;
            output.insert(
                original.name.clone(),
                MaterializedFrame {
                    clock: frame.clock,
                    name: original.name.clone(),
                    destination: frame.destination.clone(),
                    deferred_columns: HashMap::new(),
                    positions: frame.positions.clone(),
                    batch,
                    node_ids: selected.iter().map(|&row| frame.node_ids[row]).collect(),
                    timestamps: selected.iter().map(|&row| frame.timestamps[row]).collect(),
                    offsets,
                },
            );
        }
    }
    Ok(output)
}

/// Execute the common TT plan's explicit-anchor scalar fragment in bounded
/// chunks. The existing temporal executor supplies exact relation/time/order
/// selection; its numeric groups reuse the PQL executor and accumulators.
/// Job-local prepared scalar operations. It owns no anchor or result buffers.
pub struct PreparedAggregatePlan {
    root_type: String,
    query: QueryPlan,
    output_positions: HashMap<String, usize>,
}

struct PreparedRowSelection {
    entity_table: Arc<TableReader>,
    entity_col_indices: Vec<(String, usize)>,
    operations: Vec<(String, crate::ttgen_resolve::ResolvedAggregate<Seconds>)>,
}

impl PreparedAggregatePlan {
    pub fn prepare(reader: &GraphIndexReader, root: &str, json: &str) -> Result<Self, String> {
        use diskgraph_core::tt_builder::{DiskGraphTTPlan, Op};
        let mut plan: DiskGraphTTPlan<Seconds> =
            serde_json::from_str(json).map_err(|e| format!("invalid DGTTPlan: {e}"))?;
        plan.normalize_scalar_ops()?;
        if plan.entity.table != root || plan.temporal.is_some() {
            return Err("explicit-anchor scalar plans require matching entity table and no entity enumeration".into());
        }
        let mut selected = plan.clone();
        let mut query = QueryPlan {
            frames: vec![],
            sequences: vec![],
            aggregates: vec![],
            prepared_frames: HashMap::new(),
            window_groups: None,
            numeric: HashMap::new(),
            selection: None,
        };
        let mut names = HashSet::new();
        let mut output_positions = HashMap::new();
        for stage in &mut selected.stages {
            if stage.filter.is_some() {
                return Err("scalar materialization does not accept stage side effects".into());
            }
            let mut pql = Vec::new();
            for op in stage.ops.drain(..) {
                let Op::RowAggregate { rows, reduction } = &op else {
                    return Err("scalar materialization requires row aggregate ops".into());
                };
                if !names.insert(reduction.name.clone()) {
                    return Err("duplicate scalar output name".into());
                }
                output_positions.insert(reduction.name.clone(), output_positions.len());
                let selected_rows = reduction.row_filter.is_some()
                    || reduction.limit.is_some()
                    || reduction.target_entity_table.is_some();
                if selected_rows && (!reduction.semantics.all_null_sum_is_zero
                    || reduction.semantics.keep_first_equal)
                {
                    return Err("PQL selection policies require PQL reduction semantics".into());
                }
                if selected_rows && (reader.metadata().timestamp_unit != TimeUnit::Seconds
                    || diskgraph_core::tt_builder::plan_timestamp_unit(json)
                        .map_err(|e| e.to_string())? != TimeUnit::Seconds)
                {
                    return Err("PQL row filters/limits require matching seconds plan, cutoff and index clocks".into());
                }
                if let Some(old) = query.frames.iter().find(|f| f.name == rows.name) {
                    if old != rows {
                        return Err("conflicting shared row-source declarations".into());
                    }
                } else {
                    query.frames.push(rows.clone());
                }
                query.aggregates.push(reduction.clone());
                if reduction.row_filter.is_some() || reduction.limit.is_some() || reduction.target_entity_table.is_some() {
                    pql.push(op);
                }
            }
            stage.ops = pql;
        }
        query.selection = if selected.stages.iter().any(|s| !s.ops.is_empty()) {
            let entity_table = reader.table(root).ok_or("unknown entity table")?;
            let context = entity_table
                .node_data_decoder()
                .map(|d| d.context().clone());
            let columns = crate::ttgen_resolve::collect_entity_col_refs(&selected, &context)
                .map_err(|e| e.to_string())?;
            let aliases = selected
                .stages
                .iter()
                .flat_map(|s| &s.ops)
                .enumerate()
                .map(|(i, op)| {
                    let Op::RowAggregate { reduction, .. } = op else {
                        unreachable!()
                    };
                    (reduction.name.clone(), i)
                })
                .collect::<HashMap<_, _>>();
            let mut operations = Vec::new();
            for stage in &selected.stages {
                for op in &stage.ops {
                    let resolved = crate::ttgen_resolve::resolve_op(
                        op,
                        reader,
                        &entity_table,
                        &context,
                        &aliases,
                        &columns,
                        None,
                    )
                    .map_err(|e| e.to_string())?;
                    let crate::ttgen_resolve::ResolvedOp::Aggregate(aggregate) = resolved else {
                        unreachable!()
                    };
                    let Op::RowAggregate { reduction, .. } = op else {
                        unreachable!()
                    };
                    operations.push((reduction.name.clone(), aggregate));
                }
            }
            Some(PreparedRowSelection {
                entity_table,
                entity_col_indices: columns,
                operations,
            })
        } else {
            None
        };
        let windows = window_groups(&query);
        for frame in query
            .frames
            .iter()
            .chain(windows.iter().map(|(union, _)| union))
        {
            query
                .prepared_frames
                .insert(frame.name.clone(), PreparedFrame::new(reader, frame, root)?);
        }
        for frame in &query.frames {
            let consumers = query
                .aggregates
                .iter()
                .filter(|a| a.frame == frame.name)
                .collect::<Vec<_>>();
            if let Some(operations) = prepare_numeric_frame(reader, frame, root, &consumers)? {
                query.numeric.insert(frame.name.clone(), operations);
            }
        }
        query.window_groups = Some(windows);
        Ok(Self {
            root_type: root.into(),
            query,
            output_positions,
        })
    }

    pub fn execute(
        &self,
        reader: &GraphIndexReader,
        roots: &[u64],
        cutoffs: &[i64],
        anchor_batch_size: usize,
        pool: &rayon::ThreadPool,
    ) -> Result<QueryResult, String> {
        if anchor_batch_size == 0 {
            return Err("anchor_batch_size must be positive".into());
        }
        if roots.len() != cutoffs.len() {
            return Err("root IDs and cutoffs must have equal length".into());
        }
        let mut merged: Option<QueryResult> = None;
        let ranges = if roots.is_empty() {
            std::iter::once(0..0).collect::<Vec<_>>()
        } else {
            (0..roots.len())
                .step_by(anchor_batch_size)
                .map(|i| i..(i + anchor_batch_size).min(roots.len()))
                .collect()
        };
        for range in ranges {
            let roots = &roots[range.clone()];
            let cutoffs = &cutoffs[range];
            let chunks = roots.len().div_ceil(64).max(1).min(pool.current_num_threads());
            let size = roots.len().div_ceil(chunks).max(1);
            let parts = if roots.is_empty() {
                vec![self.execute_chunk(reader, roots, cutoffs)?]
            } else {
                pool.install(|| {
                    roots
                        .par_chunks(size)
                        .zip(cutoffs.par_chunks(size))
                        .map(|(roots, cutoffs)| self.execute_chunk(reader, roots, cutoffs))
                        .collect::<Result<Vec<_>, _>>()
                })?
            };
            for result in parts {
                if let Some(output) = &mut merged {
                    for (target, source) in output.aggregates.iter_mut().zip(result.aggregates) {
                        target.values.extend(source.values);
                        if target.kind.is_none() {
                            target.kind = source.kind;
                        }
                    }
                    output.timings.traversal += result.timings.traversal;
                    output.timings.decode_and_select += result.timings.decode_and_select;
                    output.timings.aggregation += result.timings.aggregation;
                    output.timings.total += result.timings.total;
                    output.stats.frames_materialized += result.stats.frames_materialized;
                    output.stats.rows_materialized += result.stats.rows_materialized;
                    output.stats.rows_decoded += result.stats.rows_decoded;
                    output.stats.values_decoded += result.stats.values_decoded;
                    output.stats.edges_visited += result.stats.edges_visited;
                } else {
                    merged = Some(result);
                }
            }
        }
        Ok(merged.expect("empty requests execute one empty chunk"))
    }

    fn execute_chunk(
        &self,
        reader: &GraphIndexReader,
        roots: &[u64],
        cutoffs: &[i64],
    ) -> Result<QueryResult, String> {
        let started = Instant::now();
        let mut result = execute_query(reader, &self.root_type, roots, cutoffs, &self.query)?;
        result
            .aggregates
            .sort_by_key(|aggregate| self.output_positions[&aggregate.name]);
        result.timings.total = elapsed_ns(started);
        Ok(result)
    }
}

pub fn execute_aggregate_plan(
    reader: &GraphIndexReader,
    root: &str,
    roots: &[u64],
    cutoffs: &[i64],
    json: &str,
    anchor_batch_size: usize,
    pool: &rayon::ThreadPool,
) -> Result<QueryResult, String> {
    PreparedAggregatePlan::prepare(reader, root, json)?.execute(
        reader,
        roots,
        cutoffs,
        anchor_batch_size,
        pool,
    )
}

/// A temporal plan resolved once for all batches of an iterator.
pub struct PreparedTemporalPlan {
    root_type: String,
    query: QueryPlan,
}

impl PreparedTemporalPlan {
    pub fn prepare(reader: &GraphIndexReader, root: &str, json: &str) -> Result<Self, String> {
        let query: QueryPlan = serde_json::from_str(json)
            .map_err(|e| format!("invalid temporal query JSON: {e}"))?;
        Self::from_query(reader, root, query)
    }

    /// Runtime seed timestamps are index ticks, unlike standalone query cutoffs
    /// whose omitted unit retains its historical seconds default.
    pub fn prepare_history(reader: &GraphIndexReader, root: &str, json: &str) -> Result<Self, String> {
        if reader.metadata().timestamp_unit != TimeUnit::Seconds {
            return Err("history iteration requires seconds until explicit index-clock cutoffs are enabled".into());
        }
        Self::prepare(reader, root, json)
    }

    fn from_query(reader: &GraphIndexReader, root: &str, mut query: QueryPlan) -> Result<Self, String> {
        for sequence in &query.sequences { validate_sequence_plan(sequence)?; }
        let windows = window_groups(&query);
        for frame in query.frames.iter().chain(windows.iter().map(|(union, _)| union)) {
            query.prepared_frames.insert(frame.name.clone(), PreparedFrame::new(reader, frame, root)?);
        }
        for frame in &query.frames {
            let consumers = query.aggregates.iter().filter(|a| a.frame == frame.name).collect::<Vec<_>>();
            if let Some(operations) = prepare_numeric_frame(reader, frame, root, &consumers)? {
                query.numeric.insert(frame.name.clone(), operations);
            }
        }
        query.window_groups = Some(windows);
        Ok(Self { root_type: root.into(), query })
    }

    pub fn require_timed_sequence(&self, name: &str) -> Result<(), String> {
        let sequence = self.query.sequences.iter().find(|s| s.name == name)
            .ok_or_else(|| format!("unknown merge sequence {name:?}"))?;
        if self.query.frames.iter().find(|f| f.name == sequence.frame)
            .is_none_or(|f| f.time_column.is_none()) {
            return Err("native chronological history merging requires timed frames".into());
        }
        Ok(())
    }

    pub fn execute(&self, reader: &GraphIndexReader, roots: &[u64], cutoffs: &[i64], pool: &rayon::ThreadPool) -> Result<QueryResult, String> {
        execute_prepared_parallel(reader, &self.root_type, roots, cutoffs, &self.query, pool)
    }
}

/// Run independent anchors with one shared, prepared temporal plan.
pub fn execute_parallel(
    reader: &GraphIndexReader,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    plan_json: &str,
    pool: &rayon::ThreadPool,
) -> Result<QueryResult, String> {
    PreparedTemporalPlan::prepare(reader, root_type, plan_json)?.execute(reader, roots, cutoffs, pool)
}

fn execute_prepared_parallel(
    reader: &GraphIndexReader,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    plan: &QueryPlan,
    pool: &rayon::ThreadPool,
) -> Result<QueryResult, String> {
    if roots.len() != cutoffs.len() || roots.len() < 64 || pool.current_num_threads() == 1 {
        if roots.len() != cutoffs.len() {
            return Err("root IDs and cutoffs must have equal length".into());
        }
        return execute_query(reader, root_type, roots, cutoffs, plan);
    }
    let started = Instant::now();
    let chunks = roots.len().div_ceil(64).min(pool.current_num_threads() * 2);
    let size = roots.len().div_ceil(chunks);
    let results = pool.install(|| {
        roots
            .par_chunks(size)
            .zip(cutoffs.par_chunks(size))
            .map(|(roots, cutoffs)| execute_query(reader, root_type, roots, cutoffs, plan))
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut results = results.into_iter();
    let mut output = results.next().expect("nonempty anchor chunks");
    let mut batches = output
        .sequences
        .iter()
        .map(|sequence| vec![sequence.batch.clone()])
        .collect::<Vec<_>>();
    for result in results {
        for (index, (sequence, next)) in output
            .sequences
            .iter_mut()
            .zip(result.sequences)
            .enumerate()
        {
            let base = sequence.node_ids.len();
            sequence
                .offsets
                .extend(next.offsets.into_iter().skip(1).map(|v| base + v));
            sequence.node_ids.extend(next.node_ids);
            sequence.timestamps.extend(next.timestamps);
            batches[index].push(next.batch);
        }
        for (aggregate, next) in output.aggregates.iter_mut().zip(result.aggregates) {
            aggregate.values.extend(next.values);
        }
        output.stats.frames_materialized = output
            .stats
            .frames_materialized
            .max(result.stats.frames_materialized);
        output.stats.rows_materialized += result.stats.rows_materialized;
        output.stats.rows_decoded += result.stats.rows_decoded;
        output.stats.values_decoded += result.stats.values_decoded;
        output.stats.edges_visited += result.stats.edges_visited;
        output.timings.traversal += result.timings.traversal;
        output.timings.decode_and_select += result.timings.decode_and_select;
        output.timings.sequence_assembly += result.timings.sequence_assembly;
        output.timings.aggregation += result.timings.aggregation;
    }
    for (sequence, batches) in output.sequences.iter_mut().zip(batches) {
        sequence.batch = arrow::compute::concat_batches(&sequence.batch.schema(), &batches)
            .map_err(|error| error.to_string())?;
    }
    output.timings.total = elapsed_ns(started);
    Ok(output)
}

pub fn execute(
    reader: &GraphIndexReader,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    plan_json: &str,
) -> Result<QueryResult, String> {
    if roots.len() != cutoffs.len() {
        return Err(format!(
            "root IDs and cutoffs must have equal length: {} != {}",
            roots.len(),
            cutoffs.len()
        ));
    }
    let plan: QueryPlan = serde_json::from_str(plan_json)
        .map_err(|error| format!("invalid temporal query JSON: {error}"))?;
    execute_query(reader, root_type, roots, cutoffs, &plan)
}

fn execute_query(
    reader: &GraphIndexReader,
    root_type: &str,
    roots: &[u64],
    cutoffs: &[i64],
    plan: &QueryPlan,
) -> Result<QueryResult, String> {
    let total_started = Instant::now();
    for sequence in &plan.sequences {
        validate_sequence_plan(sequence)?;
    }
    let mut timings = QueryTimings::default();
    let mut stats = QueryStats::default();
    let mut frames = HashMap::new();
    let mut seen_frames = HashSet::new();
    let mut shared_results = HashMap::new();
    let mut window_frames = shared_window_frames(
        reader,
        plan,
        root_type,
        roots,
        cutoffs,
        &mut timings,
        &mut stats,
    )?;
    for frame_plan in &plan.frames {
        if !seen_frames.insert(frame_plan.name.clone()) {
            return Err(format!("duplicate temporal frame {:?}", frame_plan.name));
        }
        let consumers = plan
            .sequences
            .iter()
            .filter(|sequence| sequence.frame == frame_plan.name)
            .collect::<Vec<_>>();
        if consumers.is_empty() {
            let consumers = plan
                .aggregates
                .iter()
                .enumerate()
                .filter(|(_, aggregate)| aggregate.frame == frame_plan.name)
                .collect::<Vec<_>>();
            let aggregates = consumers
                .iter()
                .map(|(_, aggregate)| *aggregate)
                .collect::<Vec<_>>();
            let mut retained_frame = None;
            if let Some(values) = shared_numeric_frame(
                reader,
                frame_plan,
                root_type,
                roots,
                cutoffs,
                &aggregates,
                &mut timings,
                &mut stats,
                &mut retained_frame,
                plan,
                window_frames.remove(&frame_plan.name),
            )? {
                for ((index, _), value) in consumers.into_iter().zip(values) {
                    shared_results.insert(index, value);
                }
                continue;
            }
            if let Some(frame) = retained_frame {
                frames.insert(frame_plan.name.clone(), frame);
                continue;
            }
        }
        // A history frame may serve several projections/output orders, but
        // preselection is safe only when they select the same capped rows.
        let sequence = consumers.first().copied().filter(|first| {
            first.limit.is_some()
                && !plan.aggregates.iter().any(|a| a.frame == frame_plan.name)
                && consumers.iter().all(|s| s.limit == first.limit && s.order == first.order)
        });
        // Keep aggregate inputs complete; decode history-only payload only
        // after its exact timestamp/PK selection, including fallback clocks.
        let full_columns = (sequence.is_some()
            || consumers.len() == 1 && consumers[0].limit.is_some()).then(|| {
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
            plan.prepared_frames.get(&frame_plan.name),
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
        .enumerate()
        .map(|(index, aggregate)| {
            if let Some(result) = shared_results.remove(&index) {
                return Ok(result);
            }
            let frame = frames.get(&aggregate.frame).ok_or_else(|| {
                format!(
                    "aggregate {:?} has unknown frame {:?}",
                    aggregate.name, aggregate.frame
                )
            })?;
            aggregate_result(frame, aggregate, cutoffs, roots, plan.selection.as_ref())
        })
        .collect::<Result<Vec<_>, String>>()?;
    timings.aggregation = timings
        .aggregation
        .saturating_add(elapsed_ns(aggregate_started));
    timings.total = elapsed_ns(total_started);
    Ok(QueryResult {
        sequences,
        aggregates,
        timings,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn reduce(
        rows: RowsView<'_>,
        operation: &str,
        column: Option<&str>,
        cutoff: i64,
        sum_type: Option<SumType>,
    ) -> Result<ScalarValue, String> {
        super::reduce(rows, operation, column, cutoff, sum_type, Default::default())
    }

    #[test]
    fn frame_sort_keeps_existing_tie_order_for_every_scalar_kind() {
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
        for clock in 0..3 {
            for rotation in 0..100 {
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
                actual.rotate_left(rotation % 60);
                if rotation % 2 == 0 {
                    actual.reverse();
                }
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
            Some(ScalarValue::I64(0)),
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
            let output = scalar_values_to_array(&[ScalarValue::Null], Some(match kind { SumType::Int64 => ScalarValue::I64(0), SumType::Uint64 => ScalarValue::U64(0) })).unwrap();
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
