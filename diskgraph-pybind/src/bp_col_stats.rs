//! Compute Kumo-format statistics over completed prediction parquet files.
//!
//! Columns are resolved by name. Scalar and category statistics use the first
//! pass; exact numerical and timestamp distributions require a second pass.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    PrimitiveArray, RecordBatch, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_schema::{DataType, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;
use serde_json::{json, Map as JsonMap, Value as JsonValue};

use diskgraph_core::accumulators::{
    kumo_entry, kumo_key, CategoryCountStrategy, WelfordAccumulator,
};
use diskgraph_ingest::col_stats::{minify_kumo_stats, FileColStats, TableCategoryCounts};
use diskgraph_ingest::graph_config::{ColumnSpec, IngestedType, SemanticType, TimestampUnit};
use diskgraph_rowpack::ColumnValueObserver;

/// CATEGORY_COUNTS truncation limit for the "min" variant (matches ingest).
const MIN_COL_STATS_CATEGORY_LIMIT: usize = 20;
const NUMERICAL_HISTOGRAM_BINS: usize = 20;
// Match Kumo's signed-nanosecond timestamp histogram domain.
const FIRST_HALF_MONTH_ID: i32 = 1677 * 24 + 17;
const LAST_HALF_MONTH_ID: i32 = 2262 * 24 + 6;
const TIMESTAMP_HISTOGRAM_BINS: usize = (LAST_HALF_MONTH_ID - FIRST_HALF_MONTH_ID + 1) as usize;

#[derive(Clone, Copy)]
struct FloatingHistogramSpec {
    spec_idx: usize,
    min: f64,
    max: f64,
}

#[derive(Clone, Copy)]
struct IntegerHistogramSpec {
    spec_idx: usize,
    min: i64,
    bin_width: u128,
    num_bins: usize,
}

#[derive(Clone, Copy)]
struct TimestampHistogramSpec {
    spec_idx: usize,
}

#[derive(Clone, Copy, Default)]
struct IntegerRange {
    min: Option<i64>,
    max: Option<i64>,
}

impl IntegerRange {
    fn observe(&mut self, value: i64) {
        self.min = Some(self.min.map_or(value, |min| min.min(value)));
        self.max = Some(self.max.map_or(value, |max| max.max(value)));
    }

    fn merge(&mut self, other: Self) {
        if let Some(min) = other.min {
            self.observe(min);
        }
        if let Some(max) = other.max {
            self.observe(max);
        }
    }
}

/// Whether the stats paths support this physical/logical type pair.
fn arrow_type_matches_dtype(arrow: &DataType, dtype: &IngestedType) -> bool {
    use IngestedType as T;
    match dtype {
        // Integer / unsigned observers widen any signed/unsigned physical width
        // to i64 (kumo maps int32/int16/byte to narrow physical types).
        T::Int8 | T::Int16 | T::Int32 | T::Int64 | T::UInt8 | T::UInt16 | T::UInt32 | T::UInt64 => {
            matches!(
                arrow,
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
            )
        }
        // Float observers downcast `Float64Array` or `Float32Array`.
        T::Float64 | T::Float32 => matches!(arrow, DataType::Float64 | DataType::Float32),
        // Bool observer downcasts `BooleanArray`.
        T::Bool => matches!(arrow, DataType::Boolean),
        // String observer downcasts `StringArray` (Utf8).
        T::String => matches!(arrow, DataType::Utf8),
        // Timestamp observer accepts any Arrow `TimeUnit` (Second/Milli/Micro/
        // Nano, all normalized to seconds) or an Int64 epoch column.
        T::Timestamp => matches!(arrow, DataType::Timestamp(_, _) | DataType::Int64),
        // Not stats-tracked by BP col-stats (see the `_ => {}` arm of
        // `drive_from_arrow_column`): any physical type is acceptable.
        T::Bytes
        | T::Uuid
        | T::Uuid256
        | T::Ip
        | T::Date
        | T::Decimal128
        | T::Decimal256
        | T::Float16Array
        | T::Float32Array
        | T::Float64Array
        | T::Int8Array
        | T::Int16Array
        | T::Int32Array
        | T::Int64Array
        | T::StringArray => true,
    }
}

/// Map spec positions to parquet columns by name and validate their types.
fn resolve_parquet_indices(columns: &[ColumnSpec], schema: &Schema) -> Result<Vec<usize>> {
    columns
        .iter()
        .map(|col| {
            let idx = schema.index_of(&col.name).map_err(|_| {
                anyhow::anyhow!(
                    "schema error: column '{}' (from column specs) is absent from the \
                     parquet schema; parquet columns are {:?}",
                    col.name,
                    schema
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect::<Vec<_>>()
                )
            })?;
            let arrow = schema.field(idx).data_type();
            if !arrow_type_matches_dtype(arrow, &col.ingested_dtype) {
                anyhow::bail!(
                    "schema error: column '{}' has parquet Arrow type {arrow:?}, incompatible \
                     with its declared ingested dtype {:?} (type mismatch)",
                    col.name,
                    col.ingested_dtype,
                );
            }
            Ok(idx)
        })
        .collect()
}

fn parquet_arrow_schema(path: &Path) -> Result<Schema> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder.schema().as_ref().clone())
}

struct DistributionPartial {
    floating: Vec<[u64; NUMERICAL_HISTOGRAM_BINS]>,
    integer: Vec<Vec<u64>>,
    timestamps: Vec<TimestampHistogram>,
}

impl DistributionPartial {
    fn new(floating: usize, integer_specs: &[IntegerHistogramSpec], timestamps: usize) -> Self {
        Self {
            floating: vec![[0; NUMERICAL_HISTOGRAM_BINS]; floating],
            integer: integer_specs
                .iter()
                .map(|spec| vec![0; spec.num_bins])
                .collect(),
            timestamps: (0..timestamps).map(|_| TimestampHistogram::new()).collect(),
        }
    }

    fn merge(&mut self, other: Self) {
        for (total, counts) in self.floating.iter_mut().zip(other.floating) {
            for (total, count) in total.iter_mut().zip(counts) {
                *total += count;
            }
        }
        for (total, counts) in self.integer.iter_mut().zip(other.integer) {
            for (total, count) in total.iter_mut().zip(counts) {
                *total += count;
            }
        }
        for (total, counts) in self.timestamps.iter_mut().zip(other.timestamps) {
            total.merge(counts);
        }
    }
}

struct TimestampHistogram {
    counts: Box<[u64]>,
    first: usize,
    last: usize,
    total: u64,
}

impl TimestampHistogram {
    fn new() -> Self {
        Self {
            counts: vec![0; TIMESTAMP_HISTOGRAM_BINS].into_boxed_slice(),
            first: TIMESTAMP_HISTOGRAM_BINS,
            last: 0,
            total: 0,
        }
    }

    fn observe(&mut self, seconds: i64) {
        let Some(idx) = half_month_index(seconds) else {
            return;
        };
        self.counts[idx] += 1;
        self.first = self.first.min(idx);
        self.last = self.last.max(idx);
        self.total += 1;
    }

    fn merge(&mut self, other: Self) {
        if other.total == 0 {
            return;
        }
        for idx in other.first..=other.last {
            self.counts[idx] += other.counts[idx];
        }
        self.first = self.first.min(other.first);
        self.last = self.last.max(other.last);
        self.total += other.total;
    }
}

fn timestamp_seconds_in_signed_ns_domain(raw: i64, unit: TimestampUnit) -> Option<i64> {
    let nanos_per_unit: i128 = match unit {
        TimestampUnit::Seconds => 1_000_000_000,
        TimestampUnit::Millis => 1_000_000,
        TimestampUnit::Micros => 1_000,
        TimestampUnit::Nanos => 1,
    };
    let nanos = i128::from(raw) * nanos_per_unit;
    // Pandas reserves i64::MIN as NaT; valid signed-nanosecond timestamps are
    // the remaining values through i64::MAX, inclusive.
    if nanos <= i128::from(i64::MIN) || nanos > i128::from(i64::MAX) {
        return None;
    }
    Some(unit.scale_to_secs(raw))
}

fn timestamp_seconds(array: &dyn Array, row: usize, col: &ColumnSpec) -> Option<i64> {
    let any = array.as_any();
    if let Some(a) = any.downcast_ref::<TimestampSecondArray>() {
        return (!a.is_null(row))
            .then(|| timestamp_seconds_in_signed_ns_domain(a.value(row), TimestampUnit::Seconds))
            .flatten();
    }
    if let Some(a) = any.downcast_ref::<TimestampMillisecondArray>() {
        return (!a.is_null(row))
            .then(|| timestamp_seconds_in_signed_ns_domain(a.value(row), TimestampUnit::Millis))
            .flatten();
    }
    if let Some(a) = any.downcast_ref::<TimestampMicrosecondArray>() {
        return (!a.is_null(row))
            .then(|| timestamp_seconds_in_signed_ns_domain(a.value(row), TimestampUnit::Micros))
            .flatten();
    }
    if let Some(a) = any.downcast_ref::<TimestampNanosecondArray>() {
        return (!a.is_null(row))
            .then(|| timestamp_seconds_in_signed_ns_domain(a.value(row), TimestampUnit::Nanos))
            .flatten();
    }
    let a = any.downcast_ref::<Int64Array>()?;
    if a.is_null(row) {
        return None;
    }
    let raw = a.value(row);
    let unit = col
        .timestamp_unit
        .or_else(|| TimestampUnit::from_epoch_magnitude(raw))
        .unwrap_or(TimestampUnit::Seconds);
    timestamp_seconds_in_signed_ns_domain(raw, unit)
}

fn half_month_index(seconds: i64) -> Option<usize> {
    use chrono::{Datelike, Utc};

    let date = chrono::DateTime::<Utc>::from_timestamp(seconds, 0)?.date_naive();
    let half = i32::from(date.day() > 15);
    let id = date.year() * 24 + (date.month() as i32 - 1) * 2 + half;
    usize::try_from(id - FIRST_HALF_MONTH_ID)
        .ok()
        .filter(|idx| *idx < TIMESTAMP_HISTOGRAM_BINS)
}

fn half_month_bucket(idx: usize) -> String {
    let id = FIRST_HALF_MONTH_ID + idx as i32;
    let year = id / 24;
    let month_half = id % 24;
    let month = month_half / 2 + 1;
    let day = if month_half % 2 == 0 { 1 } else { 16 };
    format!("kumo_timestamp_{year:04}-{month:02}-{day:02} 00:00:00")
}

fn histogram_bin(value: f64, spec: FloatingHistogramSpec) -> usize {
    if spec.max <= spec.min {
        return 0;
    }
    let ratio = if spec.min < 0.0 && spec.max > 0.0 {
        // Avoid overflowing ranges that span zero.
        let scale = spec.min.abs().max(spec.max.abs());
        let min = spec.min / scale;
        let max = spec.max / scale;
        (value / scale - min) / (max - min)
    } else {
        (value - spec.min) / (spec.max - spec.min)
    };
    ((ratio.clamp(0.0, 1.0) * NUMERICAL_HISTOGRAM_BINS as f64).floor() as usize)
        .min(NUMERICAL_HISTOGRAM_BINS - 1)
}

fn histogram_left_edge(spec: FloatingHistogramSpec, idx: usize) -> f64 {
    let fraction = idx as f64 / NUMERICAL_HISTOGRAM_BINS as f64;
    if spec.min < 0.0 && spec.max > 0.0 {
        (1.0 - fraction) * spec.min + fraction * spec.max
    } else {
        spec.min + fraction * (spec.max - spec.min)
    }
}

fn observe_int_col<T, F, O>(col_idx: usize, a: &PrimitiveArray<T>, to_i64: F, observer: &mut O)
where
    T: ArrowPrimitiveType,
    F: Fn(T::Native) -> i64,
    O: ColumnValueObserver,
{
    for i in 0..a.len() {
        if a.is_null(i) {
            observer.observe_null(col_idx);
        } else {
            observer.observe_i64(col_idx, to_i64(a.value(i)));
        }
    }
}

/// Normalize timestamps with the same ceil rounding used by ingest.
fn observe_ts_col<T, O>(
    col_idx: usize,
    a: &PrimitiveArray<T>,
    unit: TimestampUnit,
    observer: &mut O,
) where
    T: ArrowPrimitiveType<Native = i64>,
    O: ColumnValueObserver,
{
    for i in 0..a.len() {
        if a.is_null(i) {
            observer.observe_null(col_idx);
        } else {
            observer.observe_timestamp(col_idx, unit.scale_to_secs(a.value(i)));
        }
    }
}

fn bin_numeric_col<T, F>(
    a: &PrimitiveArray<T>,
    spec: FloatingHistogramSpec,
    counts: &mut [u64; NUMERICAL_HISTOGRAM_BINS],
    to_f64: F,
) where
    T: ArrowPrimitiveType,
    F: Fn(T::Native) -> f64,
{
    for i in 0..a.len() {
        if a.is_null(i) {
            continue;
        }
        let value = to_f64(a.value(i));
        if value.is_finite() {
            counts[histogram_bin(value, spec)] += 1;
        }
    }
}

fn observe_integer_range<T, F>(a: &PrimitiveArray<T>, range: &mut IntegerRange, to_i64: F)
where
    T: ArrowPrimitiveType,
    F: Fn(T::Native) -> Option<i64>,
{
    for i in 0..a.len() {
        if !a.is_null(i) {
            if let Some(value) = to_i64(a.value(i)) {
                range.observe(value);
            }
        }
    }
}

fn integer_range(array: &dyn Array) -> IntegerRange {
    let mut range = IntegerRange::default();
    let any = array.as_any();
    if let Some(a) = any.downcast_ref::<Int64Array>() {
        observe_integer_range(a, &mut range, Some);
    } else if let Some(a) = any.downcast_ref::<Int32Array>() {
        observe_integer_range(a, &mut range, |v| Some(i64::from(v)));
    } else if let Some(a) = any.downcast_ref::<Int16Array>() {
        observe_integer_range(a, &mut range, |v| Some(i64::from(v)));
    } else if let Some(a) = any.downcast_ref::<Int8Array>() {
        observe_integer_range(a, &mut range, |v| Some(i64::from(v)));
    } else if let Some(a) = any.downcast_ref::<UInt64Array>() {
        observe_integer_range(a, &mut range, |v| i64::try_from(v).ok());
    } else if let Some(a) = any.downcast_ref::<UInt32Array>() {
        observe_integer_range(a, &mut range, |v| Some(i64::from(v)));
    } else if let Some(a) = any.downcast_ref::<UInt16Array>() {
        observe_integer_range(a, &mut range, |v| Some(i64::from(v)));
    } else if let Some(a) = any.downcast_ref::<UInt8Array>() {
        observe_integer_range(a, &mut range, |v| Some(i64::from(v)));
    }
    range
}

fn integer_histogram_bin(value: i64, spec: IntegerHistogramSpec) -> usize {
    let offset = (i128::from(value) - i128::from(spec.min)) as u128;
    usize::try_from(offset / spec.bin_width)
        .unwrap_or(usize::MAX)
        .min(spec.num_bins - 1)
}

fn bin_integer_col<T, F>(
    a: &PrimitiveArray<T>,
    spec: IntegerHistogramSpec,
    counts: &mut [u64],
    to_i64: F,
) where
    T: ArrowPrimitiveType,
    F: Fn(T::Native) -> Option<i64>,
{
    for i in 0..a.len() {
        if !a.is_null(i) {
            if let Some(value) = to_i64(a.value(i)) {
                counts[integer_histogram_bin(value, spec)] += 1;
            }
        }
    }
}

fn is_integer_dtype(dtype: &IngestedType) -> bool {
    matches!(
        dtype,
        IngestedType::Int8
            | IngestedType::Int16
            | IngestedType::Int32
            | IngestedType::Int64
            | IngestedType::UInt8
            | IngestedType::UInt16
            | IngestedType::UInt32
            | IngestedType::UInt64
    )
}

fn accumulate_distributions_for_row_group(
    path: &Path,
    row_group: usize,
    columns: &[ColumnSpec],
    parquet_indices: &[usize],
    floating_specs: &[FloatingHistogramSpec],
    integer_specs: &[IntegerHistogramSpec],
    timestamp_specs: &[TimestampHistogramSpec],
    out: &mut DistributionPartial,
) -> Result<()> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut projected_indices: Vec<usize> = floating_specs
        .iter()
        .map(|spec| parquet_indices[spec.spec_idx])
        .chain(
            integer_specs
                .iter()
                .map(|spec| parquet_indices[spec.spec_idx]),
        )
        .chain(
            timestamp_specs
                .iter()
                .map(|spec| parquet_indices[spec.spec_idx]),
        )
        .collect();
    projected_indices.sort_unstable();
    projected_indices.dedup();
    let projection = ProjectionMask::roots(builder.parquet_schema(), projected_indices);
    let reader = builder
        .with_row_groups(vec![row_group])
        .with_projection(projection)
        .build()?;
    for batch_res in reader {
        let batch = batch_res?;
        let schema = batch.schema();
        for (hist_idx, spec) in floating_specs.iter().enumerate() {
            let array = batch.column(schema.index_of(&columns[spec.spec_idx].name)?);
            let counts = &mut out.floating[hist_idx];
            let any = array.as_any();
            // Floats plus any signed/unsigned integer width — the same widened
            // numeric set the observer pass consumes, so a narrow-int numerical
            // column gets a histogram too (not just scalar stats).
            if let Some(a) = any.downcast_ref::<Float64Array>() {
                bin_numeric_col(a, *spec, counts, |v| v);
            } else if let Some(a) = any.downcast_ref::<Float32Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            } else if let Some(a) = any.downcast_ref::<Int64Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            } else if let Some(a) = any.downcast_ref::<Int32Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            } else if let Some(a) = any.downcast_ref::<Int16Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            } else if let Some(a) = any.downcast_ref::<Int8Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            } else if let Some(a) = any.downcast_ref::<UInt64Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            } else if let Some(a) = any.downcast_ref::<UInt32Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            } else if let Some(a) = any.downcast_ref::<UInt16Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            } else if let Some(a) = any.downcast_ref::<UInt8Array>() {
                bin_numeric_col(a, *spec, counts, |v| v as f64);
            }
        }
        for (hist_idx, spec) in integer_specs.iter().enumerate() {
            let array = batch.column(schema.index_of(&columns[spec.spec_idx].name)?);
            let counts = &mut out.integer[hist_idx];
            let any = array.as_any();
            if let Some(a) = any.downcast_ref::<Int64Array>() {
                bin_integer_col(a, *spec, counts, Some);
            } else if let Some(a) = any.downcast_ref::<Int32Array>() {
                bin_integer_col(a, *spec, counts, |v| Some(i64::from(v)));
            } else if let Some(a) = any.downcast_ref::<Int16Array>() {
                bin_integer_col(a, *spec, counts, |v| Some(i64::from(v)));
            } else if let Some(a) = any.downcast_ref::<Int8Array>() {
                bin_integer_col(a, *spec, counts, |v| Some(i64::from(v)));
            } else if let Some(a) = any.downcast_ref::<UInt64Array>() {
                bin_integer_col(a, *spec, counts, |v| i64::try_from(v).ok());
            } else if let Some(a) = any.downcast_ref::<UInt32Array>() {
                bin_integer_col(a, *spec, counts, |v| Some(i64::from(v)));
            } else if let Some(a) = any.downcast_ref::<UInt16Array>() {
                bin_integer_col(a, *spec, counts, |v| Some(i64::from(v)));
            } else if let Some(a) = any.downcast_ref::<UInt8Array>() {
                bin_integer_col(a, *spec, counts, |v| Some(i64::from(v)));
            }
        }
        for (hist_idx, spec) in timestamp_specs.iter().enumerate() {
            let array = batch.column(schema.index_of(&columns[spec.spec_idx].name)?);
            for row in 0..array.len() {
                if let Some(seconds) =
                    timestamp_seconds(array.as_ref(), row, &columns[spec.spec_idx])
                {
                    out.timestamps[hist_idx].observe(seconds);
                }
            }
        }
    }
    Ok(())
}

/// Feed one name-resolved Arrow column into a stats observer.
fn drive_from_arrow_column<O: ColumnValueObserver>(
    col_idx: usize,
    array: &dyn Array,
    col: &ColumnSpec,
    observer: &mut O,
) -> Result<()> {
    match &col.ingested_dtype {
        IngestedType::Int64
        | IngestedType::Int32
        | IngestedType::Int16
        | IngestedType::Int8
        | IngestedType::UInt64
        | IngestedType::UInt32
        | IngestedType::UInt16
        | IngestedType::UInt8 => {
            let any = array.as_any();
            if let Some(a) = any.downcast_ref::<Int64Array>() {
                observe_int_col(col_idx, a, |v| v, observer);
            } else if let Some(a) = any.downcast_ref::<Int32Array>() {
                observe_int_col(col_idx, a, |v| v as i64, observer);
            } else if let Some(a) = any.downcast_ref::<Int16Array>() {
                observe_int_col(col_idx, a, |v| v as i64, observer);
            } else if let Some(a) = any.downcast_ref::<Int8Array>() {
                observe_int_col(col_idx, a, |v| v as i64, observer);
            } else if let Some(a) = any.downcast_ref::<UInt64Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        let value = a.value(i);
                        let value = i64::try_from(value).map_err(|_| {
                            anyhow::anyhow!(
                                "type mismatch: column '{}' contains UInt64 value {value}, \
                                 outside the supported i64 range for column statistics",
                                col.name
                            )
                        })?;
                        observer.observe_i64(col_idx, value);
                    }
                }
            } else if let Some(a) = any.downcast_ref::<UInt32Array>() {
                observe_int_col(col_idx, a, |v| v as i64, observer);
            } else if let Some(a) = any.downcast_ref::<UInt16Array>() {
                observe_int_col(col_idx, a, |v| v as i64, observer);
            } else if let Some(a) = any.downcast_ref::<UInt8Array>() {
                observe_int_col(col_idx, a, |v| v as i64, observer);
            }
        }
        IngestedType::Float64 | IngestedType::Float32 => {
            if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_f64(col_idx, a.value(i));
                    }
                }
            } else if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_f64(col_idx, a.value(i) as f64);
                    }
                }
            }
        }
        IngestedType::Bool => {
            if let Some(a) = array.as_any().downcast_ref::<BooleanArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_i64(col_idx, i64::from(a.value(i)));
                    }
                }
            }
        }
        IngestedType::String => {
            if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        observer.observe_string(col_idx, a.value(i));
                    }
                }
            }
        }
        IngestedType::Timestamp => {
            let any = array.as_any();
            if let Some(a) = any.downcast_ref::<TimestampSecondArray>() {
                observe_ts_col(col_idx, a, TimestampUnit::Seconds, observer);
            } else if let Some(a) = any.downcast_ref::<TimestampMillisecondArray>() {
                observe_ts_col(col_idx, a, TimestampUnit::Millis, observer);
            } else if let Some(a) = any.downcast_ref::<TimestampMicrosecondArray>() {
                observe_ts_col(col_idx, a, TimestampUnit::Micros, observer);
            } else if let Some(a) = any.downcast_ref::<TimestampNanosecondArray>() {
                observe_ts_col(col_idx, a, TimestampUnit::Nanos, observer);
            } else if let Some(a) = any.downcast_ref::<Int64Array>() {
                // Explicit units avoid ambiguous epoch-magnitude inference.
                for i in 0..a.len() {
                    if a.is_null(i) {
                        observer.observe_null(col_idx);
                    } else {
                        let raw = a.value(i);
                        let secs = match col
                            .timestamp_unit
                            .or_else(|| TimestampUnit::from_epoch_magnitude(raw))
                        {
                            Some(unit) => unit.scale_to_secs(raw),
                            None => raw,
                        };
                        observer.observe_timestamp(col_idx, secs);
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

type ByClassStats = HashMap<String, WelfordAccumulator>;

fn find_score_class(columns: &[ColumnSpec]) -> Option<(usize, usize)> {
    let score = columns
        .iter()
        .position(|c| c.name == "SCORE" && c.semantic_type == Some(SemanticType::Numerical))?;
    let class = columns
        .iter()
        .position(|c| c.name == "CLASS" && c.semantic_type == Some(SemanticType::Categorical))?;
    Some((score, class))
}

fn accumulate_by_class(
    batch: &RecordBatch,
    score_idx: usize,
    class_idx: usize,
    out: &mut ByClassStats,
) {
    let classes = match batch
        .column(class_idx)
        .as_any()
        .downcast_ref::<StringArray>()
    {
        Some(c) => c,
        None => return,
    };
    let score_col = batch.column(score_idx);
    let scores_f64 = score_col.as_any().downcast_ref::<Float64Array>();
    let scores_f32 = score_col.as_any().downcast_ref::<Float32Array>();
    if scores_f64.is_none() && scores_f32.is_none() {
        return;
    }
    for i in 0..classes.len() {
        if classes.is_null(i) {
            continue;
        }
        let score = if let Some(a) = scores_f64 {
            if a.is_null(i) {
                continue;
            }
            a.value(i)
        } else {
            let a = scores_f32.expect("checked non-none above");
            if a.is_null(i) {
                continue;
            }
            a.value(i) as f64
        };
        out.entry(classes.value(i).to_string())
            .or_default()
            .observe(score);
    }
}

fn stats_for_row_group(
    path: &Path,
    row_group: usize,
    columns: &[ColumnSpec],
    parquet_indices: &[usize],
    integer_spec_indices: &[usize],
    table_cat: &Arc<TableCategoryCounts>,
    score_class: Option<(usize, usize)>,
) -> Result<(FileColStats, ByClassStats, Vec<IntegerRange>)> {
    let mut fcs = FileColStats::new_with_categories(columns, Arc::clone(table_cat));
    let mut by_class = ByClassStats::new();
    let mut integer_ranges = vec![IntegerRange::default(); integer_spec_indices.len()];
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_row_groups(vec![row_group])
        .build()?;
    for batch_res in reader {
        let batch = batch_res?;
        // Stats keys use spec order; reads use physical parquet order.
        for (i, col) in columns.iter().enumerate() {
            drive_from_arrow_column(i, batch.column(parquet_indices[i]).as_ref(), col, &mut fcs)?;
        }
        for (range, &spec_idx) in integer_ranges.iter_mut().zip(integer_spec_indices) {
            range.merge(integer_range(
                batch.column(parquet_indices[spec_idx]).as_ref(),
            ));
        }
        if let Some((si, ci)) = score_class {
            accumulate_by_class(&batch, si, ci, &mut by_class);
        }
    }
    fcs.flush_categories();
    Ok((fcs, by_class, integer_ranges))
}

/// Compute full and min Kumo-format statistics for prediction parquet files.
pub fn compute_parquet_col_stats(
    parquet_paths: &[&Path],
    columns: &[ColumnSpec],
    data_version: &str,
) -> Result<(String, String)> {
    let names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
    let types: Vec<IngestedType> = columns.iter().map(|c| c.ingested_dtype.clone()).collect();
    let stypes: Vec<_> = columns.iter().map(|c| c.semantic_type).collect();

    // Hybrid counting bounds memory for high-cardinality columns.
    let table_cat = Arc::new(TableCategoryCounts::new(
        CategoryCountStrategy::Hybrid,
        &stypes,
        /*row_subsample=*/ false,
    ));

    let mut work: Vec<(usize, usize)> = Vec::new();
    for (pi, path) in parquet_paths.iter().enumerate() {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let num_row_groups = builder.metadata().num_row_groups();
        for rg in 0..num_row_groups {
            work.push((pi, rg));
        }
    }

    let parquet_indices: Vec<Vec<usize>> = parquet_paths
        .iter()
        .map(|path| resolve_parquet_indices(columns, &parquet_arrow_schema(path)?))
        .collect::<Result<_>>()?;

    let integer_spec_indices: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter_map(|(idx, col)| {
            (col.semantic_type == Some(SemanticType::Numerical)
                && is_integer_dtype(&col.ingested_dtype))
            .then_some(idx)
        })
        .collect();

    let score_class = find_score_class(columns);

    let (mut acc, by_class, integer_ranges) = if work.is_empty() {
        let mut fcs = FileColStats::new_with_categories(columns, Arc::clone(&table_cat));
        fcs.flush_categories();
        (
            fcs,
            ByClassStats::new(),
            vec![IntegerRange::default(); integer_spec_indices.len()],
        )
    } else {
        work.par_iter()
            .map(|&(pi, rg)| {
                let score_class_pq =
                    score_class.map(|(s, c)| (parquet_indices[pi][s], parquet_indices[pi][c]));
                stats_for_row_group(
                    parquet_paths[pi],
                    rg,
                    columns,
                    &parquet_indices[pi],
                    &integer_spec_indices,
                    &table_cat,
                    score_class_pq,
                )
            })
            .try_reduce(
                || {
                    (
                        FileColStats::new_with_categories(columns, Arc::clone(&table_cat)),
                        ByClassStats::new(),
                        vec![IntegerRange::default(); integer_spec_indices.len()],
                    )
                },
                |(mut acc, mut by_class, mut integer_ranges),
                 (mut partial, partial_by_class, partial_ranges)| {
                    acc.merge(&mut partial);
                    for (class, stats) in partial_by_class {
                        by_class.entry(class).or_default().merge(&stats);
                    }
                    for (total, range) in integer_ranges.iter_mut().zip(partial_ranges) {
                        total.merge(range);
                    }
                    Ok((acc, by_class, integer_ranges))
                },
            )?
    };

    // Serialization drains category state, so derive min from the full map.
    let mut full = acc.to_kumo_json(&names, &types, Some(&table_cat), None, data_version);

    // FileColStats uses floating accumulators for every numerical dtype. Keep
    // its mean/variance/quantiles, but restore exact integer endpoints before
    // serializing transformed prediction columns above f64's exact range.
    for (&spec_idx, range) in integer_spec_indices.iter().zip(&integer_ranges) {
        let name = &columns[spec_idx].name;
        for (stat, value) in [("MIN", range.min), ("MAX", range.max)] {
            if let (Some(entry), Some(value)) = (
                full.get_mut(&kumo_key(name, "kumo.numerical", stat, false)),
                value,
            ) {
                if let Some(entry) = entry.as_object_mut() {
                    entry.insert("value".to_string(), json!(value));
                }
            }
        }
    }

    let floating_specs: Vec<FloatingHistogramSpec> = columns
        .iter()
        .enumerate()
        .filter_map(|(col_idx, col)| {
            if col.semantic_type != Some(SemanticType::Numerical)
                || is_integer_dtype(&col.ingested_dtype)
            {
                return None;
            }
            let stype = "kumo.numerical";
            let min = full
                .get(&kumo_key(&col.name, stype, "MIN", false))?
                .get("value")?
                .as_f64()?;
            let max = full
                .get(&kumo_key(&col.name, stype, "MAX", false))?
                .get("value")?
                .as_f64()?;
            (min.is_finite() && max.is_finite() && max > min).then_some(FloatingHistogramSpec {
                spec_idx: col_idx,
                min,
                max,
            })
        })
        .collect();
    let integer_specs: Vec<IntegerHistogramSpec> = integer_spec_indices
        .iter()
        .zip(integer_ranges)
        .filter_map(|(&spec_idx, range)| {
            let (min, max) = (range.min?, range.max?);
            let value_range = (i128::from(max) - i128::from(min)) as u128;
            if value_range == 0 {
                return None;
            }
            let bin_width = value_range.div_ceil(NUMERICAL_HISTOGRAM_BINS as u128);
            let num_bins = usize::try_from(value_range.div_ceil(bin_width)).ok()?;
            Some(IntegerHistogramSpec {
                spec_idx,
                min,
                bin_width,
                num_bins,
            })
        })
        .collect();
    let timestamp_specs: Vec<TimestampHistogramSpec> = columns
        .iter()
        .enumerate()
        .filter_map(|(idx, col)| {
            (col.semantic_type == Some(SemanticType::Timestamp))
                .then_some(TimestampHistogramSpec { spec_idx: idx })
        })
        .collect();

    if !work.is_empty()
        && (!floating_specs.is_empty() || !integer_specs.is_empty() || !timestamp_specs.is_empty())
    {
        let distributions = work
            .par_iter()
            .try_fold(
                || {
                    DistributionPartial::new(
                        floating_specs.len(),
                        &integer_specs,
                        timestamp_specs.len(),
                    )
                },
                |mut out, &(pi, rg)| {
                    accumulate_distributions_for_row_group(
                        parquet_paths[pi],
                        rg,
                        columns,
                        &parquet_indices[pi],
                        &floating_specs,
                        &integer_specs,
                        &timestamp_specs,
                        &mut out,
                    )?;
                    Ok::<_, anyhow::Error>(out)
                },
            )
            .try_reduce(
                || {
                    DistributionPartial::new(
                        floating_specs.len(),
                        &integer_specs,
                        timestamp_specs.len(),
                    )
                },
                |mut total, partial| {
                    total.merge(partial);
                    Ok(total)
                },
            )?;

        for (spec, counts) in floating_specs.iter().zip(distributions.floating) {
            let total: u64 = counts.iter().sum();
            if total == 0 {
                continue;
            }
            let histogram = JsonValue::Array(
                counts
                    .iter()
                    .enumerate()
                    .map(|(idx, count)| {
                        json!([
                            histogram_left_edge(*spec, idx),
                            *count as f64 / total as f64 * 100.0
                        ])
                    })
                    .collect(),
            );
            let name = &columns[spec.spec_idx].name;
            full.insert(
                kumo_key(name, "kumo.numerical", "HISTOGRAM", false),
                kumo_entry(histogram, false, data_version, total),
            );
        }
        for (spec, counts) in integer_specs.iter().zip(distributions.integer) {
            let total: u64 = counts.iter().sum();
            if total == 0 {
                continue;
            }
            let histogram = JsonValue::Array(
                counts
                    .iter()
                    .enumerate()
                    .map(|(idx, count)| {
                        let left = i128::from(spec.min) + idx as i128 * spec.bin_width as i128;
                        json!([
                            i64::try_from(left).expect("integer histogram edge fits i64"),
                            *count as f64 / total as f64 * 100.0
                        ])
                    })
                    .collect(),
            );
            full.insert(
                kumo_key(
                    &columns[spec.spec_idx].name,
                    "kumo.numerical",
                    "HISTOGRAM",
                    false,
                ),
                kumo_entry(histogram, false, data_version, total),
            );
        }
        for (spec, histogram) in timestamp_specs.iter().zip(distributions.timestamps) {
            if histogram.total == 0 {
                continue;
            }
            full.insert(
                kumo_key(
                    &columns[spec.spec_idx].name,
                    "kumo.timestamp",
                    "HISTOGRAM_HALF_MONTH",
                    false,
                ),
                kumo_entry(
                    JsonValue::Array(
                        (histogram.first..=histogram.last)
                            .map(|idx| json!([half_month_bucket(idx), histogram.counts[idx]]))
                            .collect(),
                    ),
                    false,
                    data_version,
                    histogram.total,
                ),
            );
        }
    }

    let mut min = minify_kumo_stats(&full, MIN_COL_STATS_CATEGORY_LIMIT);
    // The cache fallback reads this stat from the min variant.
    if let Some((key, entry)) =
        build_summary_stats_by_class(columns, score_class, &by_class, data_version)
    {
        full.insert(key.clone(), entry.clone());
        min.insert(key, entry);
    }
    Ok((to_json_string(&full)?, to_json_string(&min)?))
}

fn build_summary_stats_by_class(
    columns: &[ColumnSpec],
    score_class: Option<(usize, usize)>,
    by_class: &ByClassStats,
    data_version: &str,
) -> Option<(String, JsonValue)> {
    let (score_idx, _class_idx) = score_class?;
    let mut total: u64 = 0;
    let mut classes = JsonMap::new();
    for (cls, w) in by_class {
        if w.count == 0 {
            continue;
        }
        total += w.count;
        classes.insert(
            cls.clone(),
            json!({"min": w.min, "max": w.max, "mean": w.mean()}),
        );
    }
    if classes.is_empty() {
        return None;
    }
    let score_name = &columns[score_idx].name;
    Some((
        kumo_key(
            score_name,
            "kumo.numerical",
            "SUMMARY_STATS_BY_CLASS",
            false,
        ),
        kumo_entry(JsonValue::Object(classes), false, data_version, total),
    ))
}

fn to_json_string(map: &JsonMap<String, JsonValue>) -> Result<String> {
    Ok(serde_json::to_string(map)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Arc as StdArc;

    use arrow_array::{
        ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, TimestampSecondArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use diskgraph_ingest::graph_config::SemanticType;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use tempfile::tempdir;

    /// Write a parquet with an explicit max rows-per-row-group so the test
    /// exercises the multi-row-group parallel path on a single file.
    fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>, max_rows_per_rg: usize) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let file = fs::File::create(path).unwrap();
        let batch = RecordBatch::try_new(StdArc::new(schema), columns).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_size(max_rows_per_rg)
            .build();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn col(name: &str, ingested: IngestedType, stype: SemanticType) -> ColumnSpec {
        ColumnSpec {
            name: name.to_string(),
            source_dtype: None,
            ingested_dtype: ingested,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: Some(stype),
            multicategorical_delimiter: None,
        }
    }

    fn val(full: &str, key: &str) -> JsonValue {
        let m: JsonMap<String, JsonValue> = serde_json::from_str(full).unwrap();
        m[key]["value"].clone()
    }

    /// Multi-row-group single file → parallel workers + merge must equal truth.
    #[test]
    fn parallel_row_groups_match_truth() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let n = 1000usize;
        let schema = Schema::new(vec![
            Field::new("entity", DataType::Utf8, false),
            Field::new("SCORE", DataType::Float64, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        let entity: Vec<String> = (0..n).map(|i| format!("u{i}")).collect();
        let score: Vec<Option<f64>> = (0..n).map(|i| Some(i as f64 / 1000.0)).collect();
        let class: Vec<Option<&str>> = (0..n).map(|i| Some(["A", "B", "C"][i % 3])).collect();
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(StringArray::from(entity)) as ArrayRef,
                StdArc::new(Float64Array::from(score)) as ArrayRef,
                StdArc::new(StringArray::from(class)) as ArrayRef,
            ],
            // 128 rows/group → ~8 row groups → real parallelism.
            128,
        );

        let columns = vec![
            col("entity", IngestedType::String, SemanticType::Id),
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let (full, min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();

        // COUNT == n across columns (merge across row groups correct).
        assert_eq!(val(&full, "entity__kumo.ID__COUNT__False"), n as u64);
        assert_eq!(val(&full, "SCORE__kumo.numerical__COUNT__False"), n as u64);
        assert_eq!(
            val(&full, "CLASS__kumo.categorical__COUNT__False"),
            n as u64
        );

        // SCORE min/max exact across the whole file.
        let mn = val(&full, "SCORE__kumo.numerical__MIN__False")
            .as_f64()
            .unwrap();
        let mx = val(&full, "SCORE__kumo.numerical__MAX__False")
            .as_f64()
            .unwrap();
        assert!((mn - 0.0).abs() < 1e-9, "min={mn}");
        assert!((mx - 0.999).abs() < 1e-9, "max={mx}");
        let histogram = val(&full, "SCORE__kumo.numerical__HISTOGRAM__False");
        let histogram = histogram.as_array().unwrap();
        assert_eq!(histogram.len(), 20);
        let percentage_sum: f64 = histogram.iter().map(|bin| bin[1].as_f64().unwrap()).sum();
        assert!((percentage_sum - 100.0).abs() < 1e-9);

        // CLASS num_unique == 3 EXACT (Hybrid map complete, overrides HLL).
        assert_eq!(
            val(&full, "CLASS__kumo.categorical__NUM_UNIQUE__False"),
            3u64
        );

        // entity num_unique == n EXACT (1000 distinct, well under the 1M cap).
        assert_eq!(val(&full, "entity__kumo.ID__NUM_UNIQUE__False"), n as u64);

        // CLASS category counts populated (new_with_categories).
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(mfull.contains_key("CLASS__kumo.categorical__CATEGORY_COUNTS__False"));

        // SUMMARY_STATS_BY_CLASS: SCORE grouped by CLASS, merged correctly
        // across the ~8 row groups. Class A = {i : i%3==0}, score=i/1000.
        let bc = val(
            &full,
            "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False",
        );
        assert!((bc["A"]["min"].as_f64().unwrap() - 0.0).abs() < 1e-9);
        assert!((bc["A"]["max"].as_f64().unwrap() - 0.999).abs() < 1e-9);
        assert!((bc["A"]["mean"].as_f64().unwrap() - 0.4995).abs() < 1e-6);
        // Three classes, no more.
        assert_eq!(bc.as_object().unwrap().len(), 3);

        // min variant parses and keeps the scalar COUNT.
        let mmin: JsonMap<String, JsonValue> = serde_json::from_str(&min).unwrap();
        assert_eq!(
            mmin["CLASS__kumo.categorical__COUNT__False"]["value"],
            1000u64
        );
        // by-class lives in BOTH variants: BP_MULTICLASS_CLASSIFICATION maps to
        // the MIN refresh stage, so the cache-fallback read path reads min.
        assert!(mmin.contains_key("SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False"));
        assert_eq!(
            mmin["SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False"]["value"],
            mfull["SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False"]["value"],
        );
    }

    /// Float32 SCORE must still produce numerical + by-class stats (model
    /// scores are frequently f32 — must not silently zero out).
    #[test]
    fn float32_score_by_class() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("SCORE", DataType::Float32, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(arrow_array::Float32Array::from(vec![
                    Some(1.0f32),
                    Some(3.0),
                    Some(5.0),
                ])) as ArrayRef,
                StdArc::new(StringArray::from(vec![Some("A"), Some("A"), Some("B")])) as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("SCORE", IngestedType::Float32, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        // Numerical scalar stats populated (not empty-Welford defaults).
        let mx = val(&full, "SCORE__kumo.numerical__MAX__False")
            .as_f64()
            .unwrap();
        assert!((mx - 5.0).abs() < 1e-6, "max={mx}");
        let bc = val(
            &full,
            "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False",
        );
        assert!((bc["A"]["mean"].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((bc["B"]["min"].as_f64().unwrap() - 5.0).abs() < 1e-6);
    }

    /// Null SCORE / null CLASS rows are skipped; remaining valid pairs still
    /// produce correct per-class stats and `num_entries`.
    #[test]
    fn by_class_skips_nulls() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("SCORE", DataType::Float64, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                // (A,2.0) (A,null-score→skip) (null-class,9.0→skip) (B,4.0)
                StdArc::new(Float64Array::from(vec![
                    Some(2.0),
                    None,
                    Some(9.0),
                    Some(4.0),
                ])) as ArrayRef,
                StdArc::new(StringArray::from(vec![
                    Some("A"),
                    Some("A"),
                    None,
                    Some("B"),
                ])) as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        let bc = val(
            &full,
            "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False",
        );
        assert!((bc["A"]["mean"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((bc["B"]["mean"].as_f64().unwrap() - 4.0).abs() < 1e-9);
        // null-CLASS row (score 9.0) contributed to no bucket.
        assert_eq!(bc.as_object().unwrap().len(), 2);
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert_eq!(
            mfull["SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False"]["num_entries"],
            2u64,
        );
    }

    /// Exact per-class min/max/mean for a hand-checked tiny input.
    #[test]
    fn summary_stats_by_class_exact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("SCORE", DataType::Float64, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(Float64Array::from(vec![Some(1.0), Some(3.0), Some(5.0)])) as ArrayRef,
                StdArc::new(StringArray::from(vec![Some("A"), Some("A"), Some("B")])) as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        let bc = val(
            &full,
            "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False",
        );
        // A: [1,3] -> min 1, max 3, mean 2 ; B: [5] -> min/max/mean 5.
        assert!((bc["A"]["min"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert!((bc["A"]["max"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert!((bc["A"]["mean"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((bc["B"]["min"].as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert!((bc["B"]["max"].as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert!((bc["B"]["mean"].as_f64().unwrap() - 5.0).abs() < 1e-9);
    }

    /// No CLASS column (e.g. regression/binary BP) → no by-class stat emitted.
    #[test]
    fn no_summary_stats_by_class_without_class_column() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![Field::new("TARGET_PRED", DataType::Float64, true)]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])) as ArrayRef],
            1024,
        );
        let columns = vec![col(
            "TARGET_PRED",
            IngestedType::Float64,
            SemanticType::Numerical,
        )];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(!mfull.keys().any(|k| k.contains("SUMMARY_STATS_BY_CLASS")));
    }

    /// Multiple part files merge correctly.
    #[test]
    fn merges_multiple_files() {
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("part-0.parquet");
        let p2 = dir.path().join("part-1.parquet");
        let schema = || Schema::new(vec![Field::new("v", DataType::Int64, false)]);
        write_parquet(
            &p1,
            schema(),
            vec![StdArc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
            2,
        );
        write_parquet(
            &p2,
            schema(),
            vec![StdArc::new(Int64Array::from(vec![4, 5, 6])) as ArrayRef],
            2,
        );
        let columns = vec![col("v", IngestedType::Int64, SemanticType::Numerical)];
        let (full, _min) =
            compute_parquet_col_stats(&[p1.as_path(), p2.as_path()], &columns, "v1").unwrap();
        assert_eq!(val(&full, "v__kumo.numerical__COUNT__False"), 6u64);
        let mx = val(&full, "v__kumo.numerical__MAX__False")
            .as_f64()
            .unwrap();
        assert!((mx - 6.0).abs() < 1e-9);
    }

    #[test]
    fn resolves_each_parquet_file_schema_by_name() {
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("part-1.parquet");
        let p2 = dir.path().join("part-2.parquet");
        write_parquet(
            &p1,
            Schema::new(vec![
                Field::new("score", DataType::Float64, true),
                Field::new("entity", DataType::Utf8, true),
            ]),
            vec![
                StdArc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
                StdArc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
            ],
            1,
        );
        write_parquet(
            &p2,
            Schema::new(vec![
                Field::new("entity", DataType::Utf8, true),
                Field::new("score", DataType::Float64, true),
            ]),
            vec![
                StdArc::new(StringArray::from(vec!["c", "d"])) as ArrayRef,
                StdArc::new(Float64Array::from(vec![3.0, 4.0])) as ArrayRef,
            ],
            1,
        );

        let columns = vec![
            col("entity", IngestedType::String, SemanticType::Id),
            col("score", IngestedType::Float64, SemanticType::Numerical),
        ];
        let (full, _) =
            compute_parquet_col_stats(&[p1.as_path(), p2.as_path()], &columns, "v1").unwrap();
        assert_eq!(val(&full, "entity__kumo.ID__COUNT__False"), 4);
        assert_eq!(val(&full, "score__kumo.numerical__MIN__False"), 1.0);
        assert_eq!(val(&full, "score__kumo.numerical__MAX__False"), 4.0);
        let histogram = val(&full, "score__kumo.numerical__HISTOGRAM__False");
        let total: f64 = histogram
            .as_array()
            .unwrap()
            .iter()
            .map(|bin| bin[1].as_f64().unwrap())
            .sum();
        assert!((total - 100.0).abs() < 1e-9);
    }

    #[test]
    fn validates_every_parquet_file_schema() {
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("part-1.parquet");
        let p2 = dir.path().join("part-2.parquet");
        write_parquet(
            &p1,
            Schema::new(vec![Field::new("score", DataType::Float64, true)]),
            vec![StdArc::new(Float64Array::from(vec![1.0])) as ArrayRef],
            1,
        );
        write_parquet(
            &p2,
            Schema::new(vec![Field::new("other", DataType::Float64, true)]),
            vec![StdArc::new(Float64Array::from(vec![2.0])) as ArrayRef],
            1,
        );
        let err = compute_parquet_col_stats(
            &[p1.as_path(), p2.as_path()],
            &[col("score", IngestedType::Float64, SemanticType::Numerical)],
            "v1",
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("column 'score'"));
    }

    /// Build a ColumnSpec with an explicit `timestamp_unit`.
    fn ts_col(name: &str, unit: Option<TimestampUnit>) -> ColumnSpec {
        ColumnSpec {
            timestamp_unit: unit,
            ..col(name, IngestedType::Timestamp, SemanticType::Timestamp)
        }
    }

    fn epoch_seconds(year: i32, month: u32, day: u32) -> i64 {
        use chrono::TimeZone;

        chrono::Utc
            .with_ymd_and_hms(year, month, day, 0, 0, 0)
            .single()
            .unwrap()
            .timestamp()
    }

    #[test]
    fn timestamp_histograms_are_dense_and_merge_independently() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new(
                "created_at",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                true,
            ),
            Field::new(
                "updated_at",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                true,
            ),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(TimestampSecondArray::from(vec![
                    Some(epoch_seconds(2020, 1, 15)),
                    Some(epoch_seconds(2020, 1, 16)),
                    Some(epoch_seconds(2020, 3, 20)),
                    None,
                    Some(epoch_seconds(2020, 1, 1)),
                    Some(epoch_seconds(2020, 3, 16)),
                ])) as ArrayRef,
                StdArc::new(TimestampSecondArray::from(vec![
                    Some(epoch_seconds(2020, 2, 1)),
                    None,
                    Some(epoch_seconds(2020, 2, 16)),
                    Some(epoch_seconds(2020, 4, 1)),
                    Some(epoch_seconds(2020, 2, 1)),
                    Some(epoch_seconds(2020, 4, 15)),
                ])) as ArrayRef,
            ],
            1,
        );

        let columns = vec![ts_col("created_at", None), ts_col("updated_at", None)];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        assert_eq!(
            val(
                &full,
                "created_at__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"
            ),
            json!([
                ["kumo_timestamp_2020-01-01 00:00:00", 2],
                ["kumo_timestamp_2020-01-16 00:00:00", 1],
                ["kumo_timestamp_2020-02-01 00:00:00", 0],
                ["kumo_timestamp_2020-02-16 00:00:00", 0],
                ["kumo_timestamp_2020-03-01 00:00:00", 0],
                ["kumo_timestamp_2020-03-16 00:00:00", 2],
            ])
        );
        assert_eq!(
            val(
                &full,
                "updated_at__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"
            ),
            json!([
                ["kumo_timestamp_2020-02-01 00:00:00", 2],
                ["kumo_timestamp_2020-02-16 00:00:00", 1],
                ["kumo_timestamp_2020-03-01 00:00:00", 0],
                ["kumo_timestamp_2020-03-16 00:00:00", 0],
                ["kumo_timestamp_2020-04-01 00:00:00", 2],
            ])
        );
    }

    #[test]
    fn timestamp_histogram_is_bounded_by_the_pandas_domain() {
        const MIN_SIGNED_NS_SECONDS: i64 = -9_223_372_036;
        const MAX_SIGNED_NS_SECONDS: i64 = 9_223_372_036;
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            true,
        )]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(TimestampSecondArray::from(vec![
                Some(MIN_SIGNED_NS_SECONDS - 1),
                Some(MIN_SIGNED_NS_SECONDS),
                None,
                Some(MAX_SIGNED_NS_SECONDS),
                Some(MAX_SIGNED_NS_SECONDS + 1),
            ])) as ArrayRef],
            1,
        );

        let (full, _min) =
            compute_parquet_col_stats(&[path.as_path()], &[ts_col("ts", None)], "v1").unwrap();
        let histogram = val(&full, "ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False");
        let histogram = histogram.as_array().unwrap();
        assert_eq!(histogram.len(), TIMESTAMP_HISTOGRAM_BINS);
        assert_eq!(
            histogram.first().unwrap(),
            &json!(["kumo_timestamp_1677-09-16 00:00:00", 1])
        );
        assert_eq!(
            histogram.last().unwrap(),
            &json!(["kumo_timestamp_2262-04-01 00:00:00", 1])
        );
        assert_eq!(
            histogram
                .iter()
                .map(|bucket| bucket[1].as_u64().unwrap())
                .sum::<u64>(),
            2
        );
    }

    #[test]
    fn nanosecond_histogram_includes_valid_endpoints_and_excludes_nat() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
            true,
        )]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(TimestampNanosecondArray::from(vec![
                i64::MIN,
                i64::MIN + 1,
                i64::MAX,
            ])) as ArrayRef],
            1,
        );

        let (full, _) =
            compute_parquet_col_stats(&[path.as_path()], &[ts_col("ts", None)], "v1").unwrap();
        let histogram = val(&full, "ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False");
        let histogram = histogram.as_array().unwrap();
        assert_eq!(histogram.len(), TIMESTAMP_HISTOGRAM_BINS);
        assert_eq!(histogram.first().unwrap()[1], 1);
        assert_eq!(histogram.last().unwrap()[1], 1);
        assert_eq!(
            histogram
                .iter()
                .map(|bucket| bucket[1].as_u64().unwrap())
                .sum::<u64>(),
            2
        );
    }

    #[test]
    fn no_in_range_timestamps_omit_only_the_timestamp_histogram() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                true,
            ),
            Field::new(
                "null_ts",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                true,
            ),
            Field::new("score", DataType::Float64, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(TimestampSecondArray::from(vec![
                    Some(-9_223_372_037),
                    Some(9_223_372_037),
                    None,
                ])) as ArrayRef,
                StdArc::new(TimestampSecondArray::from(vec![None, None, None])) as ArrayRef,
                StdArc::new(Float64Array::from(vec![Some(0.0), Some(0.5), Some(1.0)])) as ArrayRef,
            ],
            1,
        );

        let columns = vec![
            ts_col("ts", None),
            ts_col("null_ts", None),
            col("score", IngestedType::Float64, SemanticType::Numerical),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        let stats: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(!stats.contains_key("ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"));
        assert!(!stats.contains_key("null_ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"));
        assert!(stats.contains_key("ts__kumo.timestamp__MIN__False"));
        assert!(stats.contains_key("ts__kumo.timestamp__MAX__False"));
        assert_eq!(
            stats["score__kumo.numerical__HISTOGRAM__False"]["value"]
                .as_array()
                .unwrap()
                .len(),
            NUMERICAL_HISTOGRAM_BINS
        );
    }

    /// Int64 epoch-ms timestamp scales to seconds (inferred and explicit unit).
    #[test]
    fn int64_epoch_ms_timestamp_scales_to_correct_year() {
        let dir = tempdir().unwrap();
        // Two equal-instant rows so MIN == MAX and the assertion is exact.
        let raw_ms = vec![1_659_304_801_388_i64, 1_659_304_801_388_i64];
        let expected = "kumo_timestamp_2022-07-31 22:00:02";

        for (label, unit) in [
            ("inferred", None),
            ("explicit", Some(TimestampUnit::Millis)),
        ] {
            let path = dir.path().join(format!("preds_{label}.parquet"));
            let schema = Schema::new(vec![Field::new("ts", DataType::Int64, true)]);
            write_parquet(
                &path,
                schema,
                vec![StdArc::new(Int64Array::from(raw_ms.clone())) as ArrayRef],
                64,
            );
            let columns = vec![ts_col("ts", unit)];
            let (full, _min) =
                compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
            assert_eq!(
                val(&full, "ts__kumo.timestamp__MIN__False"),
                expected,
                "MIN wrong for {label} unit"
            );
            assert_eq!(
                val(&full, "ts__kumo.timestamp__MAX__False"),
                expected,
                "MAX wrong for {label} unit"
            );
            assert_eq!(
                val(&full, "ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"),
                json!([["kumo_timestamp_2022-07-16 00:00:00", 2]])
            );
        }
    }

    /// Int64 epoch-seconds (no explicit unit) is left as-is, not over-scaled.
    #[test]
    fn int64_epoch_seconds_timestamp_unaffected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("preds.parquet");
        let raw_s = vec![1_659_304_801_i64, 1_659_304_801_i64];
        let schema = Schema::new(vec![Field::new("ts", DataType::Int64, true)]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(Int64Array::from(raw_s)) as ArrayRef],
            64,
        );
        let columns = vec![ts_col("ts", None)];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        assert_eq!(
            val(&full, "ts__kumo.timestamp__MIN__False"),
            "kumo_timestamp_2022-07-31 22:00:01"
        );
    }

    /// Arrow Timestamp(ns) stays on the /1e9 path, not the Int64 unit logic.
    #[test]
    fn arrow_nanosecond_timestamp_unaffected() {
        use arrow_array::TimestampNanosecondArray;
        let dir = tempdir().unwrap();
        let path = dir.path().join("preds.parquet");
        // 1659304801 s expressed in ns.
        let raw_ns = vec![1_659_304_801_000_000_000_i64, 1_659_304_801_000_000_000_i64];
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
            true,
        )]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(TimestampNanosecondArray::from(raw_ns)) as ArrayRef],
            64,
        );
        let columns = vec![ts_col("ts", None)];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        assert_eq!(
            val(&full, "ts__kumo.timestamp__MIN__False"),
            "kumo_timestamp_2022-07-31 22:00:01"
        );
    }

    #[test]
    fn permuted_spec_order_aligns_by_name() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let n = 300usize;
        let schema = Schema::new(vec![
            Field::new("entity", DataType::Utf8, false),
            Field::new("SCORE", DataType::Float64, true),
            Field::new("CLASS", DataType::Utf8, true),
        ]);
        let entity: Vec<String> = (0..n).map(|i| format!("u{i}")).collect();
        let score: Vec<Option<f64>> = (0..n).map(|i| Some(i as f64 / 1000.0)).collect();
        let class: Vec<Option<&str>> = (0..n).map(|i| Some(["A", "B", "C"][i % 3])).collect();
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(StringArray::from(entity)) as ArrayRef,
                StdArc::new(Float64Array::from(score)) as ArrayRef,
                StdArc::new(StringArray::from(class)) as ArrayRef,
            ],
            64,
        );

        let columns = vec![
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
            col("entity", IngestedType::String, SemanticType::Id),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();

        assert_eq!(val(&full, "SCORE__kumo.numerical__COUNT__False"), n as u64);
        assert_eq!(val(&full, "entity__kumo.ID__COUNT__False"), n as u64);
        assert_eq!(
            val(&full, "CLASS__kumo.categorical__NUM_UNIQUE__False"),
            3u64
        );
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(mfull.contains_key("SCORE__kumo.numerical__HISTOGRAM__False"));
    }

    #[test]
    fn invalid_schema_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("numeric.parquet");
        let schema = Schema::new(vec![Field::new("SCORE", DataType::Float64, true)]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])) as ArrayRef],
            1024,
        );
        let columns = vec![col(
            "NOT_THERE",
            IngestedType::Float64,
            SemanticType::Numerical,
        )];
        let err = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("NOT_THERE") && msg.contains("schema error"));

        let path = dir.path().join("string.parquet");
        let schema = Schema::new(vec![Field::new("SCORE", DataType::Utf8, true)]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(StringArray::from(vec![Some("x"), Some("y")])) as ArrayRef],
            1024,
        );
        let columns = vec![col("SCORE", IngestedType::Float64, SemanticType::Numerical)];
        let err = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap_err();
        assert!(format!("{err:#}").contains("type mismatch"));
    }

    #[test]
    fn narrow_int_columns_produce_stats() {
        use arrow_array::{Int32Array, UInt8Array};
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("cnt", DataType::Int32, true),
            Field::new("b", DataType::UInt8, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(Int32Array::from(vec![Some(10), Some(20), Some(30)])) as ArrayRef,
                StdArc::new(UInt8Array::from(vec![Some(1u8), Some(2), Some(3)])) as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("cnt", IngestedType::Int32, SemanticType::Numerical),
            col("b", IngestedType::UInt8, SemanticType::Numerical),
        ];
        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        assert_eq!(val(&full, "cnt__kumo.numerical__COUNT__False"), 3u64);
        assert_eq!(val(&full, "b__kumo.numerical__COUNT__False"), 3u64);
        let mfull: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(mfull.contains_key("cnt__kumo.numerical__HISTOGRAM__False"));
        assert!(mfull.contains_key("b__kumo.numerical__HISTOGRAM__False"));
        assert_eq!(
            val(&full, "b__kumo.numerical__HISTOGRAM__False"),
            json!([[1, 33.33333333333333], [2, 66.66666666666666]])
        );
    }

    #[test]
    fn integer_histogram_preserves_values_above_f64_precision() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let values = vec![i64::MAX - 2, i64::MAX - 1, i64::MAX];
        write_parquet(
            &path,
            Schema::new(vec![Field::new("v", DataType::Int64, true)]),
            vec![StdArc::new(Int64Array::from(values)) as ArrayRef],
            1,
        );

        let (full, _) = compute_parquet_col_stats(
            &[path.as_path()],
            &[col("v", IngestedType::Int64, SemanticType::Numerical)],
            "v1",
        )
        .unwrap();
        assert_eq!(
            val(&full, "v__kumo.numerical__MIN__False"),
            i64::MAX - 2
        );
        assert_eq!(val(&full, "v__kumo.numerical__MAX__False"), i64::MAX);
        assert_eq!(
            val(&full, "v__kumo.numerical__HISTOGRAM__False"),
            json!([
                [i64::MAX - 2, 33.33333333333333],
                [i64::MAX - 1, 66.66666666666666]
            ])
        );
    }

    #[test]
    fn integer_histogram_handles_full_i64_range() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        write_parquet(
            &path,
            Schema::new(vec![Field::new("v", DataType::Int64, true)]),
            vec![StdArc::new(Int64Array::from(vec![i64::MIN, i64::MAX])) as ArrayRef],
            1,
        );

        let (full, _) = compute_parquet_col_stats(
            &[path.as_path()],
            &[col("v", IngestedType::Int64, SemanticType::Numerical)],
            "v1",
        )
        .unwrap();
        let histogram = val(&full, "v__kumo.numerical__HISTOGRAM__False");
        let histogram = histogram.as_array().unwrap();
        assert_eq!(histogram.len(), NUMERICAL_HISTOGRAM_BINS);
        assert_eq!(histogram.first().unwrap()[0], i64::MIN);
        assert_eq!(histogram.first().unwrap()[1], 50.0);
        assert_eq!(histogram.last().unwrap()[1], 50.0);
    }

    #[test]
    fn constant_integer_omits_histogram_like_legacy_kumo() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        write_parquet(
            &path,
            Schema::new(vec![Field::new("v", DataType::Int64, true)]),
            vec![StdArc::new(Int64Array::from(vec![7, 7])) as ArrayRef],
            1,
        );

        let (full, _) = compute_parquet_col_stats(
            &[path.as_path()],
            &[col("v", IngestedType::Int64, SemanticType::Numerical)],
            "v1",
        )
        .unwrap();
        let stats: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(!stats.contains_key("v__kumo.numerical__HISTOGRAM__False"));
    }

    #[test]
    fn constant_float_omits_histogram_like_legacy_kumo() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        write_parquet(
            &path,
            Schema::new(vec![Field::new("v", DataType::Float64, true)]),
            vec![StdArc::new(Float64Array::from(vec![7.0, 7.0])) as ArrayRef],
            1,
        );

        let (full, _) = compute_parquet_col_stats(
            &[path.as_path()],
            &[col("v", IngestedType::Float64, SemanticType::Numerical)],
            "v1",
        )
        .unwrap();
        let stats: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(!stats.contains_key("v__kumo.numerical__HISTOGRAM__False"));
    }

    #[test]
    fn uint64_outside_i64_range_errors() {
        use arrow_array::UInt64Array;

        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![Field::new("v", DataType::UInt64, true)]);
        let overflow = i64::MAX as u64 + 1;
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(UInt64Array::from(vec![Some(0), Some(overflow)])) as ArrayRef],
            1024,
        );
        let columns = vec![col("v", IngestedType::UInt64, SemanticType::Numerical)];

        let err = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("type mismatch"), "msg={msg}");
        assert!(msg.contains("column 'v'"), "msg={msg}");
        assert!(msg.contains(&overflow.to_string()), "msg={msg}");
    }

    #[test]
    fn extreme_numerical_histogram_emits_finite_coordinates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("extreme", DataType::Float64, true),
            Field::new("subnormal", DataType::Float64, true),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(Float64Array::from(vec![-f64::MAX, 0.0, f64::MAX])) as ArrayRef,
                StdArc::new(Float64Array::from(vec![
                    f64::from_bits(1),
                    f64::from_bits(2),
                    f64::from_bits(2),
                ])) as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("extreme", IngestedType::Float64, SemanticType::Numerical),
            col("subnormal", IngestedType::Float64, SemanticType::Numerical),
        ];

        let (full, _min) = compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
        let histogram = val(&full, "extreme__kumo.numerical__HISTOGRAM__False");
        let histogram = histogram.as_array().unwrap();
        assert_eq!(histogram.len(), NUMERICAL_HISTOGRAM_BINS);
        assert!(histogram
            .iter()
            .all(|bin| bin[0].as_f64().is_some_and(f64::is_finite)));
        let percentage_sum: f64 = histogram.iter().map(|bin| bin[1].as_f64().unwrap()).sum();
        assert!((percentage_sum - 100.0).abs() < 1e-9);

        let histogram = val(&full, "subnormal__kumo.numerical__HISTOGRAM__False");
        let histogram = histogram.as_array().unwrap();
        assert!(histogram[0][1].as_f64().unwrap() > 0.0);
        assert!(histogram[19][1].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn arrow_subsecond_timestamps_use_canonical_ceil_rounding() {
        use arrow_array::{
            TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        };

        let cases: Vec<(&str, DataType, ArrayRef)> = vec![
            (
                "ms",
                DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None),
                StdArc::new(TimestampMillisecondArray::from(vec![1_001, -1, -1_001])),
            ),
            (
                "us",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
                StdArc::new(TimestampMicrosecondArray::from(vec![
                    1_000_001, -1, -1_000_001,
                ])),
            ),
            (
                "ns",
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                StdArc::new(TimestampNanosecondArray::from(vec![
                    1_000_000_001,
                    -1,
                    -1_000_000_001,
                ])),
            ),
        ];

        let dir = tempdir().unwrap();
        for (label, data_type, values) in cases {
            let path = dir.path().join(format!("predictions_{label}.parquet"));
            write_parquet(
                &path,
                Schema::new(vec![Field::new("ts", data_type, true)]),
                vec![values],
                1024,
            );
            let columns = vec![ts_col("ts", None)];
            let (full, _min) =
                compute_parquet_col_stats(&[path.as_path()], &columns, "v1").unwrap();
            assert_eq!(
                val(&full, "ts__kumo.timestamp__MIN__False"),
                "kumo_timestamp_1969-12-31 23:59:59",
                "MIN wrong for {label}"
            );
            assert_eq!(
                val(&full, "ts__kumo.timestamp__MAX__False"),
                "kumo_timestamp_1970-01-01 00:00:02",
                "MAX wrong for {label}"
            );
            assert_eq!(
                val(&full, "ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"),
                json!([
                    ["kumo_timestamp_1969-12-16 00:00:00", 1],
                    ["kumo_timestamp_1970-01-01 00:00:00", 2]
                ]),
                "histogram wrong for {label}"
            );
        }
    }
}
