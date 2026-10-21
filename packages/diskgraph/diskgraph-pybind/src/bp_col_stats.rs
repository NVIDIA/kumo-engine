// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compute Kumo-format statistics over completed prediction parquet files.
//!
//! Columns are resolved by name. Scalar and category statistics use the first
//! pass; exact numerical and timestamp distributions require a second pass.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow::compute::filter_record_batch;
use arrow::row::{OwnedRow, RowConverter, SortField};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::{DataType, Schema};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;
use serde_json::{json, Map as JsonMap, Value as JsonValue};

#[cfg(test)]
use diskgraph_core::accumulators::MAX_CATEGORIES;
use diskgraph_core::accumulators::{
    kumo_entry, kumo_key, CategoryCountStrategy, WelfordAccumulator,
};
use diskgraph_ingest::col_stats::{FileColStats, TableCategoryCounts};
use diskgraph_ingest::graph_config::{ColumnSpec, IngestedType, SemanticType, TimestampUnit};
use diskgraph_ingest::PublicIngestError;
use diskgraph_rowpack::{ColumnValueObserver, RowAccess, RowAccessHints, RowAccessStats};

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

/// Options beyond the ingest-compatible statistics; the default changes nothing.
#[derive(Clone, Debug)]
pub struct ColStatsOptions {
    /// Emit `PERCENTILES` (q01..q99, approximate) for every numerical column.
    pub percentiles: bool,
    /// Emit `SUMMARY_STATS_BY_CLASS` for a `SCORE`/`CLASS` column pair.
    pub summary_stats_by_class: bool,
    /// Emit `HISTOGRAM` / `HISTOGRAM_HALF_MONTH` (a second read).
    pub histograms: bool,
    /// Keep at most this many `CATEGORY_COUNTS` entries in the full output,
    /// and record `CATEGORY_COUNTS_CAPPED`: whether counting hit the
    /// category cap, so `NUM_UNIQUE` is an estimate.
    pub category_counts_limit: Option<usize>,
    /// Compute every statistic over only the first row of each run of equal
    /// values of these columns, in path and row order.
    pub first_row_per: Vec<String>,
}

impl Default for ColStatsOptions {
    fn default() -> Self {
        Self {
            percentiles: false,
            summary_stats_by_class: true,
            histograms: true,
            category_counts_limit: None,
            first_row_per: Vec::new(),
        }
    }
}

/// Run boundaries for `ColStatsOptions::first_row_per`.
struct RunKeys {
    names: Vec<String>,
    /// Per work item, the key row that precedes its row group.
    previous: Vec<Option<OwnedRow>>,
}

fn key_rows(batch: &RecordBatch, names: &[String]) -> Result<arrow::row::Rows> {
    let arrays: Vec<ArrayRef> = names
        .iter()
        .map(|name| Ok(batch.column(batch.schema().index_of(name)?).clone()))
        .collect::<Result<_>>()?;
    let converter = RowConverter::new(
        arrays
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect(),
    )?;
    Ok(converter.convert_columns(&arrays)?)
}

/// Keep the rows whose key differs from the row before; `carry` holds the
/// last key seen.
fn first_rows_of_runs(
    batch: RecordBatch,
    names: &[String],
    carry: &mut Option<OwnedRow>,
) -> Result<RecordBatch> {
    if batch.num_rows() == 0 {
        return Ok(batch);
    }
    let rows = key_rows(&batch, names)?;
    let mask: BooleanArray = (0..rows.num_rows())
        .map(|idx| {
            Some(if idx == 0 {
                carry.as_ref().is_none_or(|last| last.row() != rows.row(0))
            } else {
                rows.row(idx) != rows.row(idx - 1)
            })
        })
        .collect();
    *carry = Some(rows.row(rows.num_rows() - 1).owned());
    Ok(filter_record_batch(&batch, &mask)?)
}

fn last_key_row(path: &Path, row_group: usize, names: &[String]) -> Result<Option<OwnedRow>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let rows = builder.metadata().row_group(row_group).num_rows() as usize;
    if rows == 0 {
        return Ok(None);
    }
    let indices = resolve_named_indices(names, builder.schema())?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), indices);
    let reader = builder
        .with_row_groups(vec![row_group])
        .with_projection(projection)
        .with_row_selection(RowSelection::from(vec![
            RowSelector::skip(rows - 1),
            RowSelector::select(1),
        ]))
        .build()?;
    let mut last = None;
    for batch in reader {
        let batch = batch?;
        if batch.num_rows() > 0 {
            let rows = key_rows(&batch, names)?;
            last = Some(rows.row(rows.num_rows() - 1).owned());
        }
    }
    Ok(last)
}

fn resolve_named_indices(names: &[String], schema: &Schema) -> Result<Vec<usize>> {
    names
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .with_context(|| format!("schema error: missing parquet column '{name}'"))
        })
        .collect()
}

fn run_keys(
    parquet_paths: &[&Path],
    work: &[(usize, usize)],
    options: &ColStatsOptions,
) -> Result<Option<RunKeys>> {
    if options.first_row_per.is_empty() {
        return Ok(None);
    }
    let names = options.first_row_per.clone();
    let lasts = work
        .par_iter()
        .map(|&(pi, rg)| last_key_row(parquet_paths[pi], rg, &names))
        .collect::<Result<Vec<_>>>()?;
    let mut previous = Vec::with_capacity(lasts.len());
    let mut carry: Option<OwnedRow> = None;
    for last in lasts {
        previous.push(carry.clone());
        if last.is_some() {
            carry = last;
        }
    }
    Ok(Some(RunKeys { names, previous }))
}

/// Projected parquet columns: the specs' plus any run keys.
fn with_key_indices(
    mut indices: Vec<usize>,
    run_keys: Option<&RunKeys>,
    schema: &Schema,
) -> Result<Vec<usize>> {
    if let Some(keys) = run_keys {
        indices.extend(resolve_named_indices(&keys.names, schema)?);
    }
    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

fn validate_stats_dtype(col: &ColumnSpec) -> Result<()> {
    if col.semantic_type == Some(SemanticType::Timestamp) {
        return Ok(());
    }
    match col.ingested_dtype {
        IngestedType::Int64
        | IngestedType::Int32
        | IngestedType::Int16
        | IngestedType::Int8
        | IngestedType::UInt64
        | IngestedType::UInt32
        | IngestedType::UInt16
        | IngestedType::UInt8
        | IngestedType::Float64
        | IngestedType::Float32
        | IngestedType::Bool
        | IngestedType::String
        | IngestedType::Bytes
        | IngestedType::Timestamp
        | IngestedType::Float32Array
        | IngestedType::Float64Array
        | IngestedType::Float16Array
        | IngestedType::Int64Array
        | IngestedType::Int32Array
        | IngestedType::Int16Array
        | IngestedType::Int8Array
        | IngestedType::StringArray => Ok(()),
        ref other => bail!("unsupported parquet statistics type: {other:?}"),
    }
}

/// Map logical spec positions to physical parquet columns. Physical value
/// compatibility is decided by RowAccess, as it is during ingestion.
fn resolve_parquet_indices(columns: &[ColumnSpec], schema: &Schema) -> Result<Vec<usize>> {
    columns
        .iter()
        .map(|col| {
            let idx = schema
                .index_of(&col.name)
                .with_context(|| format!("schema error: missing parquet column '{}'", col.name))?;
            let arrow = schema.field(idx).data_type();
            if matches!(arrow, DataType::Null) {
                bail!(
                    "schema error: unsupported parquet type Null for column '{}'",
                    col.name
                );
            }
            Ok(idx)
        })
        .collect()
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

fn integer_timestamp(
    access: &RowAccess<'_>,
    col: &ColumnSpec,
) -> Result<Option<(TimestampUnit, i64)>> {
    let is_integer = matches!(
        access.array(0)?.data_type(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
    );
    if !is_integer || col.timestamp_format.is_some() {
        return Ok(None);
    }
    Ok(access.get_int64(0)?.and_then(|raw| {
        col.timestamp_unit
            .or_else(|| TimestampUnit::from_epoch_magnitude(raw))
            .or(Some(TimestampUnit::Seconds))
            .zip(Some(raw))
    }))
}

fn timestamp_seconds(access: &RowAccess<'_>, col: &ColumnSpec) -> Result<Option<i64>> {
    if let Some((unit, raw)) = integer_timestamp(access, col)? {
        return Ok(Some(unit.scale_to_secs(raw)));
    }
    access.get_timestamp(0)
}

fn histogram_timestamp_seconds(
    col: &ColumnSpec,
    access: &RowAccess<'_>,
    raw_timestamp: Option<(TimestampUnit, i64)>,
) -> Result<Option<i64>> {
    let Some(seconds) = timestamp_seconds(access, col)? else {
        return Ok(None);
    };
    let raw_unit = raw_timestamp.or(integer_timestamp(access, col)?);
    Ok(raw_unit.map_or(Some(seconds), |(unit, raw)| {
        timestamp_seconds_in_signed_ns_domain(raw, unit).map(|_| seconds)
    }))
}

fn for_each_timestamp_second(
    array: &arrow_array::ArrayRef,
    col: &ColumnSpec,
    mut observe: impl FnMut(i64),
) -> Result<()> {
    let timestamp_data = match array.data_type() {
        DataType::Timestamp(unit, _) => Some((
            match unit {
                arrow_schema::TimeUnit::Second => TimestampUnit::Seconds,
                arrow_schema::TimeUnit::Millisecond => TimestampUnit::Millis,
                arrow_schema::TimeUnit::Microsecond => TimestampUnit::Micros,
                arrow_schema::TimeUnit::Nanosecond => TimestampUnit::Nanos,
            },
            array.to_data(),
        )),
        _ => None,
    };
    let arrays = [array.clone()];
    let mut access = RowAccess::new(vec![row_access_hints(col)], RowAccessStats::new(None));
    access.set_cols(&arrays);
    for row in 0..array.len() {
        access.set_row(row);
        let raw_timestamp = timestamp_data.as_ref().map(|(unit, data)| {
            (
                *unit,
                data.buffers()[0].typed_data::<i64>()[data.offset() + row],
            )
        });
        if let Some(seconds) =
            histogram_timestamp_seconds(col, &access, raw_timestamp).unwrap_or(None)
        {
            observe(seconds);
        }
    }
    Ok(())
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

fn row_access_hints(col: &ColumnSpec) -> RowAccessHints {
    RowAccessHints {
        timestamp_format: col.timestamp_format.clone(),
        timestamp_unit: col.timestamp_unit.map(|unit| match unit {
            TimestampUnit::Seconds => arrow_schema::TimeUnit::Second,
            TimestampUnit::Millis => arrow_schema::TimeUnit::Millisecond,
            TimestampUnit::Micros => arrow_schema::TimeUnit::Microsecond,
            TimestampUnit::Nanos => arrow_schema::TimeUnit::Nanosecond,
        }),
        decimal_scale: col.scale,
        multicategorical_delimiter: col.multicategorical_delimiter.clone(),
        column_name: Some(col.name.clone()),
        ..Default::default()
    }
}

fn for_each_normalized<T>(
    array: &arrow_array::ArrayRef,
    col: &ColumnSpec,
    mut read: impl FnMut(&RowAccess<'_>) -> Result<Option<T>>,
    mut observe: impl FnMut(T),
) -> Result<()> {
    let arrays = [array.clone()];
    let mut access = RowAccess::new(vec![row_access_hints(col)], RowAccessStats::new(None));
    access.set_cols(&arrays);
    for row in 0..array.len() {
        access.set_row(row);
        if let Some(value) = read(&access).unwrap_or(None) {
            observe(value);
        }
    }
    Ok(())
}

fn integer_value(access: &RowAccess<'_>, idx: usize, col: &ColumnSpec) -> Result<Option<i64>> {
    match col.ingested_dtype {
        IngestedType::Int64 => access.get_int64(idx),
        IngestedType::Int32 | IngestedType::Int16 | IngestedType::Int8 => {
            access.get_int32(idx).map(|value| value.map(i64::from))
        }
        IngestedType::UInt64 => access
            .get_uint64(idx)?
            .map(i64::try_from)
            .transpose()
            .map_err(Into::into),
        IngestedType::UInt32 | IngestedType::UInt16 | IngestedType::UInt8 => {
            access.get_uint32(idx).map(|value| value.map(i64::from))
        }
        ref other => bail!("expected integer column, got {other:?}"),
    }
}

fn numerical_value(access: &RowAccess<'_>, idx: usize, col: &ColumnSpec) -> Result<Option<f64>> {
    match col.ingested_dtype {
        IngestedType::Float64 => access.get_float64(idx),
        IngestedType::Float32 => access.get_float32(idx).map(|value| value.map(f64::from)),
        ref dtype if is_integer_dtype(dtype) => {
            integer_value(access, idx, col).map(|value| value.map(|value| value as f64))
        }
        ref other => bail!("expected numerical column, got {other:?}"),
    }
}

fn bytes_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(value, "{byte:02x}").expect("write to string");
    }
    value
}

fn categorical_value(
    access: &mut RowAccess<'_>,
    idx: usize,
    col: &ColumnSpec,
) -> Result<Option<String>> {
    match col.ingested_dtype {
        ref dtype if is_integer_dtype(dtype) => {
            integer_value(access, idx, col).map(|value| value.map(|value| value.to_string()))
        }
        IngestedType::Float64 | IngestedType::Float32 => {
            numerical_value(access, idx, col).map(|value| value.map(|value| value.to_string()))
        }
        IngestedType::Bool => access
            .get_bool(idx)
            .map(|value| value.map(|value| i64::from(value).to_string())),
        IngestedType::String => access.with_string(idx, |value| Ok(value.to_owned())),
        IngestedType::Bytes => access.with_bytes(idx, |value| Ok(bytes_hex(value))),
        _ => Ok(None),
    }
}

fn fixed_len(col: &ColumnSpec) -> Option<usize> {
    col.fixed_len
        .or((col.semantic_type == Some(SemanticType::Latlong)).then_some(2))
}

fn float_list_value(
    access: &mut RowAccess<'_>,
    idx: usize,
    col: &ColumnSpec,
) -> Result<Option<diskgraph_rowpack::FloatListView>> {
    let value = access.float_list_view(idx)?;
    if value.as_ref().is_some_and(|value| {
        fixed_len(col).is_some_and(|len| value.len() != len || value.has_nulls())
    }) {
        bail!("fixed float array length mismatch");
    }
    Ok(value)
}

fn int_list_value(
    access: &mut RowAccess<'_>,
    idx: usize,
    col: &ColumnSpec,
) -> Result<Option<diskgraph_rowpack::IntListView>> {
    let value = access.int_list_view(idx)?;
    if let Some(value) = value.as_ref() {
        if fixed_len(col).is_some_and(|len| value.len() != len || value.has_nulls()) {
            bail!("fixed integer array length mismatch");
        }
    }
    Ok(value)
}

#[inline]
fn access_or_parse_failure<T, O: ColumnValueObserver>(
    value: Result<Option<T>>,
    col_idx: usize,
    observer: &mut O,
) -> Option<T> {
    match value {
        Ok(value) => value,
        Err(_) => {
            observer.observe_parse_failure(col_idx);
            None
        }
    }
}

fn integer_or_parse_failure<O: ColumnValueObserver>(
    access: &RowAccess<'_>,
    array: &arrow_array::ArrayRef,
    col: &ColumnSpec,
    col_idx: usize,
    observer: &mut O,
) -> Result<Option<i64>> {
    match integer_value(access, 0, col) {
        Ok(value) => Ok(value),
        Err(_)
            if col.ingested_dtype == IngestedType::Int64
                && matches!(
                    array.data_type(),
                    DataType::Decimal128(_, 0) | DataType::Decimal256(_, 0)
                ) =>
        {
            Err(PublicIngestError::int64_decimal_out_of_range(None, &col.name).into())
        }
        Err(_) => {
            observer.observe_parse_failure(col_idx);
            Ok(None)
        }
    }
}

fn bin_numeric_col(
    array: &arrow_array::ArrayRef,
    col: &ColumnSpec,
    spec: FloatingHistogramSpec,
    counts: &mut [u64; NUMERICAL_HISTOGRAM_BINS],
) -> Result<()> {
    for_each_normalized(
        array,
        col,
        |access| numerical_value(access, 0, col),
        |value| {
            if value.is_finite() {
                counts[histogram_bin(value, spec)] += 1;
            }
        },
    )
}

fn integer_histogram_bin(value: i64, spec: IntegerHistogramSpec) -> usize {
    let offset = (i128::from(value) - i128::from(spec.min)) as u128;
    usize::try_from(offset / spec.bin_width)
        .unwrap_or(usize::MAX)
        .min(spec.num_bins - 1)
}

fn bin_integer_col(
    array: &arrow_array::ArrayRef,
    col: &ColumnSpec,
    spec: IntegerHistogramSpec,
    counts: &mut [u64],
) -> Result<()> {
    for_each_normalized(
        array,
        col,
        |access| integer_value(access, 0, col),
        |value| counts[integer_histogram_bin(value, spec)] += 1,
    )
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
    run: Option<(&RunKeys, usize)>,
    out: &mut DistributionPartial,
) -> Result<()> {
    let projected_indices: Vec<usize> = floating_specs
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
    for batch_res in row_group_batches(path, row_group, &projected_indices, run)? {
        let batch = batch_res?;
        let schema = batch.schema();
        for (hist_idx, spec) in floating_specs.iter().enumerate() {
            let array = batch.column(schema.index_of(&columns[spec.spec_idx].name)?);
            let counts = &mut out.floating[hist_idx];
            bin_numeric_col(array, &columns[spec.spec_idx], *spec, counts)?;
        }
        for (hist_idx, spec) in integer_specs.iter().enumerate() {
            let array = batch.column(schema.index_of(&columns[spec.spec_idx].name)?);
            let counts = &mut out.integer[hist_idx];
            bin_integer_col(array, &columns[spec.spec_idx], *spec, counts)?;
        }
        for (hist_idx, spec) in timestamp_specs.iter().enumerate() {
            let array = batch.column(schema.index_of(&columns[spec.spec_idx].name)?);
            for_each_timestamp_second(array, &columns[spec.spec_idx], |seconds| {
                out.timestamps[hist_idx].observe(seconds)
            })?;
        }
    }
    Ok(())
}

/// Feed one name-resolved Arrow column into a stats observer through the same
/// checked normalization layer used by ingestion.
fn drive_from_arrow_column<O: ColumnValueObserver>(
    col_idx: usize,
    array: &arrow_array::ArrayRef,
    col: &ColumnSpec,
    observer: &mut O,
    mut integer_range: Option<&mut IntegerRange>,
) -> Result<()> {
    let arrays = [array.clone()];
    let mut access = RowAccess::new(vec![row_access_hints(col)], RowAccessStats::new(None));
    access.set_cols(&arrays);
    for row in 0..array.len() {
        access.set_row(row);
        let present = if col.semantic_type == Some(SemanticType::Timestamp) {
            access_or_parse_failure(timestamp_seconds(&access, col), col_idx, observer)
                .map(|value| observer.observe_timestamp(col_idx, value))
                .is_some()
        } else {
            match col.ingested_dtype {
                IngestedType::Int64
                | IngestedType::Int32
                | IngestedType::Int16
                | IngestedType::Int8
                | IngestedType::UInt64
                | IngestedType::UInt32
                | IngestedType::UInt16
                | IngestedType::UInt8 => {
                    if let Some(value) = integer_or_parse_failure(
                        &access, array, col, col_idx, observer,
                    )? {
                        if let Some(range) = integer_range.as_deref_mut() {
                            range.observe(value);
                        }
                        observer.observe_i64(col_idx, value);
                        true
                    } else {
                        false
                    }
                }
                IngestedType::Float64 | IngestedType::Float32 => {
                    access_or_parse_failure(numerical_value(&access, 0, col), col_idx, observer)
                        .map(|value| observer.observe_f64(col_idx, value))
                        .is_some()
                }
                IngestedType::Bool => {
                    access_or_parse_failure(access.get_bool(0), col_idx, observer)
                        .map(|value| observer.observe_i64(col_idx, i64::from(value)))
                        .is_some()
                }
                IngestedType::String => {
                    let value = access.with_string(0, |value| {
                        observer.observe_string(col_idx, value);
                        Ok(())
                    });
                    access_or_parse_failure(value, col_idx, observer).is_some()
                }
                IngestedType::Bytes => {
                    let value = access.with_bytes(0, |bytes| {
                        let value = bytes_hex(bytes);
                        observer.observe_string(col_idx, &value);
                        Ok(())
                    });
                    access_or_parse_failure(value, col_idx, observer).is_some()
                }
                IngestedType::Timestamp => {
                    access_or_parse_failure(timestamp_seconds(&access, col), col_idx, observer)
                        .map(|value| observer.observe_timestamp(col_idx, value))
                        .is_some()
                }
                IngestedType::Float32Array
                | IngestedType::Float64Array
                | IngestedType::Float16Array => access_or_parse_failure(
                    float_list_value(&mut access, 0, col),
                    col_idx,
                    observer,
                )
                .map(|value| observer.observe_float_list(col_idx, &value))
                .is_some(),
                IngestedType::Int64Array
                | IngestedType::Int32Array
                | IngestedType::Int16Array
                | IngestedType::Int8Array => {
                    if let Some(value) = access_or_parse_failure(
                        int_list_value(&mut access, 0, col),
                        col_idx,
                        observer,
                    ) {
                        let bits = match col.ingested_dtype {
                            IngestedType::Int8Array => 8,
                            IngestedType::Int16Array => 16,
                            IngestedType::Int32Array => 32,
                            IngestedType::Int64Array => 64,
                            _ => unreachable!(),
                        };
                        value
                            .require_fits_in_signed(bits, &col.name, access.row())
                            .context("type mismatch in integer array")?;
                        observer.observe_int64_list(col_idx, &value);
                        true
                    } else {
                        false
                    }
                }
                IngestedType::StringArray => {
                    access_or_parse_failure(access.string_list_view(0), col_idx, observer)
                        .map(|value| observer.observe_string_list(col_idx, &value))
                        .is_some()
                }
                ref other => bail!("unsupported parquet statistics type: {other:?}"),
            }
        };
        if !present {
            observer.observe_null(col_idx);
        }
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
    score_col: &ColumnSpec,
    class_col: &ColumnSpec,
    out: &mut ByClassStats,
) -> Result<()> {
    let arrays = [
        batch.column(score_idx).clone(),
        batch.column(class_idx).clone(),
    ];
    let mut access = RowAccess::new(
        vec![RowAccessHints::default(), RowAccessHints::default()],
        RowAccessStats::new(None),
    );
    access.set_cols(&arrays);
    for row in 0..batch.num_rows() {
        access.set_row(row);
        if let Some(score) = numerical_value(&access, 0, score_col).unwrap_or(None) {
            if let Some(class) = categorical_value(&mut access, 1, class_col).unwrap_or(None) {
                out.entry(class).or_default().observe(score);
            }
        }
    }
    Ok(())
}

type RowGroupStats = (FileColStats, ByClassStats, Vec<IntegerRange>);

/// The row group's batches, reduced to run starts when `first_row_per` is set.
fn row_group_batches<'a>(
    path: &Path,
    row_group: usize,
    parquet_indices: &[usize],
    run: Option<(&'a RunKeys, usize)>,
) -> Result<impl Iterator<Item = Result<RecordBatch>> + 'a> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let indices = with_key_indices(
        parquet_indices.to_vec(),
        run.map(|(keys, _)| keys),
        builder.schema(),
    )?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), indices);
    let reader = builder
        .with_row_groups(vec![row_group])
        .with_projection(projection)
        .build()?;
    let mut carry = run.and_then(|(keys, work_idx)| keys.previous[work_idx].clone());
    Ok(reader.map(move |batch| {
        let batch = batch?;
        match run {
            Some((keys, _)) => first_rows_of_runs(batch, &keys.names, &mut carry),
            None => Ok(batch),
        }
    }))
}

#[allow(clippy::too_many_arguments)]
fn stats_for_row_group(
    path: &Path,
    row_group: usize,
    columns: &[ColumnSpec],
    parquet_indices: &[usize],
    integer_spec_indices: &[usize],
    table_cat: &Arc<TableCategoryCounts>,
    score_class: Option<(usize, usize)>,
    run: Option<(&RunKeys, usize)>,
) -> Result<RowGroupStats> {
    let mut fcs = FileColStats::new_with_categories(columns, Arc::clone(table_cat));
    let mut by_class = ByClassStats::new();
    let mut integer_ranges = vec![IntegerRange::default(); integer_spec_indices.len()];
    for batch_res in row_group_batches(path, row_group, parquet_indices, run)? {
        let batch = batch_res?;
        let schema = batch.schema();
        for (i, col) in columns.iter().enumerate() {
            let range = integer_spec_indices
                .binary_search(&i)
                .ok()
                .map(|idx| &mut integer_ranges[idx]);
            let result = drive_from_arrow_column(
                i,
                batch.column(schema.index_of(&col.name)?),
                col,
                &mut fcs,
                range,
            );
            if let Err(error) = result {
                if error.is::<PublicIngestError>() {
                    return Err(error);
                }
                return Err(error.context(format!(
                    "compute statistics for column '{}'",
                    col.name
                )));
            }
        }
        if let Some((score, class)) = score_class {
            accumulate_by_class(
                &batch,
                schema.index_of(&columns[score].name)?,
                schema.index_of(&columns[class].name)?,
                &columns[score],
                &columns[class],
                &mut by_class,
            )?;
        }
    }
    fcs.flush_categories();
    Ok((fcs, by_class, integer_ranges))
}

/// Accumulate per-column statistics for [`compute_parquet_col_stats_with_options`].
fn accumulate_parquet_col_stats<'a>(
    parquet_paths: &'a [&'a Path],
    columns: &'a [ColumnSpec],
    max_categories: u64,
    options: &ColStatsOptions,
) -> Result<(FileColStats, Arc<TableCategoryCounts>, Accumulated<'a>)> {
    if max_categories == 0 {
        bail!("category count limit must be greater than zero");
    }
    let stypes: Vec<_> = columns.iter().map(|c| c.semantic_type).collect();
    for col in columns {
        validate_stats_dtype(col)?;
    }

    // Hybrid counting bounds memory for high-cardinality columns.
    let table_cat = Arc::new(TableCategoryCounts::new_with_max_categories(
        CategoryCountStrategy::Hybrid,
        &stypes,
        /*row_subsample=*/ false,
        max_categories,
    ));

    let mut work: Vec<(usize, usize)> = Vec::new();
    let mut parquet_indices = Vec::with_capacity(parquet_paths.len());
    for (pi, path) in parquet_paths.iter().enumerate() {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        parquet_indices.push(resolve_parquet_indices(columns, builder.schema())?);
        let num_row_groups = builder.metadata().num_row_groups();
        for rg in 0..num_row_groups {
            work.push((pi, rg));
        }
    }

    let integer_spec_indices: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter_map(|(idx, col)| {
            (col.semantic_type == Some(SemanticType::Numerical)
                && is_integer_dtype(&col.ingested_dtype))
            .then_some(idx)
        })
        .collect();

    let score_class = find_score_class(columns).filter(|_| options.summary_stats_by_class);
    let run_keys = run_keys(parquet_paths, &work, options)?;
    let empty = || {
        (
            FileColStats::new_with_categories(columns, Arc::clone(&table_cat)),
            ByClassStats::new(),
            vec![IntegerRange::default(); integer_spec_indices.len()],
        )
    };

    let (acc, by_class, integer_ranges) = if work.is_empty() {
        let (mut fcs, by_class, integer_ranges) = empty();
        fcs.flush_categories();
        (fcs, by_class, integer_ranges)
    } else {
        work.par_iter()
            .enumerate()
            .map(|(work_idx, &(pi, rg))| {
                stats_for_row_group(
                    parquet_paths[pi],
                    rg,
                    columns,
                    &parquet_indices[pi],
                    &integer_spec_indices,
                    &table_cat,
                    score_class,
                    run_keys.as_ref().map(|keys| (keys, work_idx)),
                )
            })
            .try_reduce(
                empty,
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
    Ok((
        acc,
        table_cat,
        Accumulated {
            parquet_paths,
            columns,
            work,
            parquet_indices,
            integer_spec_indices,
            integer_ranges,
            by_class,
            score_class,
            run_keys,
        },
    ))
}

/// Everything besides the column stats that the finishing passes need.
struct Accumulated<'a> {
    parquet_paths: &'a [&'a Path],
    columns: &'a [ColumnSpec],
    work: Vec<(usize, usize)>,
    parquet_indices: Vec<Vec<usize>>,
    integer_spec_indices: Vec<usize>,
    integer_ranges: Vec<IntegerRange>,
    by_class: ByClassStats,
    score_class: Option<(usize, usize)>,
    run_keys: Option<RunKeys>,
}

/// [`compute_parquet_col_stats_with_options`] with default options and a given
/// retained-vocabulary limit.
#[cfg(test)]
fn compute_parquet_col_stats_with_max_categories(
    parquet_paths: &[&Path],
    columns: &[ColumnSpec],
    data_version: &str,
    max_categories: u64,
) -> Result<(String, String)> {
    compute_parquet_col_stats_with_options(
        parquet_paths,
        columns,
        data_version,
        max_categories,
        &ColStatsOptions::default(),
    )
}

/// Compute full and min Kumo-format statistics for prediction parquet files,
/// with the optional statistics in `options`.
pub fn compute_parquet_col_stats_with_options(
    parquet_paths: &[&Path],
    columns: &[ColumnSpec],
    data_version: &str,
    max_categories: u64,
    options: &ColStatsOptions,
) -> Result<(String, String)> {
    let (mut acc, table_cat, run) =
        accumulate_parquet_col_stats(parquet_paths, columns, max_categories, options)?;
    let percentiles: Vec<(usize, Vec<f64>)> = if options.percentiles {
        let levels: Vec<f64> = (1..100).map(|q| q as f64 / 100.0).collect();
        (0..columns.len())
            .filter_map(|idx| Some((idx, acc.numerical_quantiles(idx, &levels)?)))
            .collect()
    } else {
        Vec::new()
    };
    let capped: Vec<(usize, bool)> = if options.category_counts_limit.is_some() {
        table_cat
            .columns
            .iter()
            .enumerate()
            .filter_map(|(idx, counts)| Some((idx, counts.as_ref()?.is_capped())))
            .collect()
    } else {
        Vec::new()
    };
    let names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
    let types: Vec<IngestedType> = columns.iter().map(|c| c.ingested_dtype.clone()).collect();
    let mut kumo = acc.into_kumo_stats(
        "prediction",
        &names,
        &types,
        Some(table_cat),
        None,
        data_version,
    )?;
    let mut full = std::mem::take(&mut kumo.stats);
    add_distribution_stats(&run, &mut full, data_version, options.histograms)?;
    // Category arrays are filled in, and trimmed for min, when serialized.
    let mut min = full.clone();
    for (idx, capped) in capped {
        let name = &columns[idx].name;
        let prefix = format!("{name}__");
        let stype = full.keys().find_map(|key| {
            key.strip_prefix(&prefix)?
                .strip_suffix("__CATEGORY_COUNTS__False")
                .map(str::to_string)
        });
        if let Some(stype) = stype {
            let count = full
                .get(&kumo_key(name, &stype, "COUNT", false))
                .and_then(|entry| entry.get("value")?.as_u64())
                .unwrap_or_default();
            full.insert(
                kumo_key(name, &stype, "CATEGORY_COUNTS_CAPPED", false),
                kumo_entry(json!(capped), false, data_version, count),
            );
        }
    }
    for (idx, values) in percentiles {
        let name = &columns[idx].name;
        let count = full
            .get(&kumo_key(name, "kumo.numerical", "COUNT", false))
            .and_then(|entry| entry.get("value")?.as_u64())
            .unwrap_or_default();
        full.insert(
            kumo_key(name, "kumo.numerical", "PERCENTILES", true),
            kumo_entry(json!(values), true, data_version, count),
        );
    }
    add_summary_stats_by_class(&run, &mut full, &mut min, data_version);
    Ok((
        kumo.to_json_string(&full, options.category_counts_limit.unwrap_or(usize::MAX))?,
        kumo.to_json_string(&min, MIN_COL_STATS_CATEGORY_LIMIT)?,
    ))
}

fn add_summary_stats_by_class(
    run: &Accumulated<'_>,
    full: &mut JsonMap<String, JsonValue>,
    min: &mut JsonMap<String, JsonValue>,
    data_version: &str,
) {
    // The cache fallback reads this stat from the min variant.
    if let Some((key, entry)) =
        build_summary_stats_by_class(run.columns, run.score_class, &run.by_class, data_version)
    {
        full.insert(key.clone(), entry.clone());
        min.insert(key, entry);
    }
}

/// Exact integer endpoints and the histograms, from a second pass.
fn add_distribution_stats(
    run: &Accumulated<'_>,
    full: &mut JsonMap<String, JsonValue>,
    data_version: &str,
    histograms: bool,
) -> Result<()> {
    let Accumulated {
        parquet_paths,
        columns,
        work,
        parquet_indices,
        integer_spec_indices,
        integer_ranges,
        run_keys,
        ..
    } = run;
    let (parquet_paths, columns) = (*parquet_paths, *columns);

    // FileColStats uses floating accumulators for every numerical dtype. Keep
    // its mean/variance/quantiles, but restore exact integer endpoints before
    // serializing transformed prediction columns above f64's exact range.
    for (&spec_idx, range) in integer_spec_indices.iter().zip(integer_ranges.iter()) {
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

    if !histograms {
        return Ok(());
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
        .zip(integer_ranges.iter())
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
            .enumerate()
            .try_fold(
                || {
                    DistributionPartial::new(
                        floating_specs.len(),
                        &integer_specs,
                        timestamp_specs.len(),
                    )
                },
                |mut out, (work_idx, &(pi, rg))| {
                    accumulate_distributions_for_row_group(
                        parquet_paths[pi],
                        rg,
                        columns,
                        &parquet_indices[pi],
                        &floating_specs,
                        &integer_specs,
                        &timestamp_specs,
                        run_keys.as_ref().map(|keys| (keys, work_idx)),
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
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Arc as StdArc;

    use arrow_array::builder::{Float64Builder, ListBuilder};
    use arrow_array::{
        ArrayRef, Date32Array, Decimal128Array, Decimal256Array, Float64Array, Int32Array,
        Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray, TimestampSecondArray,
    };
    use arrow::datatypes::i256;
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
            .set_max_row_group_row_count(Some(max_rows_per_rg))
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

    #[test]
    fn configurable_small_category_cap_keeps_capping_internal() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("capped.parquet");
        write_parquet(
            &path,
            Schema::new(vec![Field::new("status", DataType::Utf8, false)]),
            vec![StdArc::new(StringArray::from(vec![
                "red", "blue", "green", "red", "blue", "red",
            ])) as ArrayRef],
            100,
        );
        let columns = vec![col(
            "status",
            IngestedType::String,
            SemanticType::Categorical,
        )];

        let (full, _) =
            compute_parquet_col_stats_with_max_categories(&[path.as_path()], &columns, "v1", 2)
                .unwrap();
        assert_eq!(
            val(&full, "status__kumo.categorical__CATEGORY_COUNTS__False"),
            json!([["red", 3], ["blue", 2]])
        );
        let full: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(full
            .keys()
            .all(|key| !key.contains("CATEGORY_COUNTS_CAPPED")));
        assert!(!full.contains_key("status__kumo.categorical__NUM_UNIQUE__True"));
        let num_unique = &full["status__kumo.categorical__NUM_UNIQUE__False"];
        assert_eq!(num_unique["value"], json!(3));
        assert_eq!(num_unique["approx"], json!(false));
        assert!(!full.contains_key("status__kumo.categorical__UNIQUE_FRACTION__True"));
        assert_eq!(
            full["status__kumo.categorical__UNIQUE_FRACTION__False"]["approx"],
            json!(false)
        );
    }

    /// The pre-streaming implementation: copy the counts into JSON values,
    /// then deep-clone and truncate for min.
    fn legacy_col_stats(
        paths: &[&Path],
        columns: &[ColumnSpec],
        max_categories: u64,
    ) -> (String, String) {
        use diskgraph_ingest::col_stats::minify_kumo_stats;
        let (mut acc, table_cat, run) = accumulate_parquet_col_stats(
            paths,
            columns,
            max_categories,
            &ColStatsOptions::default(),
        )
        .unwrap();
        let names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
        let types: Vec<IngestedType> = columns.iter().map(|c| c.ingested_dtype.clone()).collect();
        let mut full = acc.to_kumo_json(&names, &types, Some(&table_cat), None, "v1");
        add_distribution_stats(&run, &mut full, "v1", true).unwrap();
        let mut min = minify_kumo_stats(&full, MIN_COL_STATS_CATEGORY_LIMIT);
        add_summary_stats_by_class(&run, &mut full, &mut min, "v1");
        (
            serde_json::to_string(&full).unwrap(),
            serde_json::to_string(&min).unwrap(),
        )
    }

    #[test]
    fn streamed_output_matches_legacy_output() {
        // Under a binding cap, which values are admitted depends on row-group
        // scheduling, so that case reads one row group to stay deterministic.
        for (files, rows_per_group, cap) in [(2, 250, MAX_CATEGORIES), (1, 10_000, 25)] {
            check_streamed_matches_legacy(files, rows_per_group, cap);
        }
    }

    fn check_streamed_matches_legacy(files: u64, rows_per_group: usize, cap: u64) {
        let dir = tempdir().unwrap();
        let schema = Schema::new(vec![
            Field::new("item", DataType::Utf8, true),
            Field::new("store", DataType::Int64, true),
            Field::new("qty", DataType::Int64, true),
            Field::new("price", DataType::Float64, true),
            Field::new("SCORE", DataType::Float64, true),
            Field::new("CLASS", DataType::Int64, true),
        ]);
        let mut paths = Vec::new();
        for file in 0..files {
            let rows = 0..1_500u64;
            let key = |r: u64| (r * 7919 + file * 31) % 211;
            let path = dir.path().join(format!("p{file}.parquet"));
            write_parquet(
                &path,
                schema.clone(),
                vec![
                    StdArc::new(StringArray::from_iter(rows.clone().map(|r| {
                        (r % 17 != 0).then(|| format!("item-{}", key(r) % (1 + key(r) / 3)))
                    }))) as ArrayRef,
                    StdArc::new(Int64Array::from_iter_values(
                        rows.clone().map(|r| (key(r) % 40) as i64 - 20),
                    )) as ArrayRef,
                    StdArc::new(Int64Array::from_iter_values(
                        rows.clone().map(|r| key(r) as i64),
                    )) as ArrayRef,
                    StdArc::new(Float64Array::from_iter_values(
                        rows.clone().map(|r| key(r) as f64 / 3.0),
                    )) as ArrayRef,
                    StdArc::new(Float64Array::from_iter_values(
                        rows.clone().map(|r| (r % 100) as f64 / 100.0),
                    )) as ArrayRef,
                    StdArc::new(Int64Array::from_iter_values(rows.map(|r| (r % 3) as i64)))
                        as ArrayRef,
                ],
                rows_per_group,
            );
            paths.push(path);
        }
        let paths: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let columns = vec![
            col("item", IngestedType::String, SemanticType::Categorical),
            col("store", IngestedType::Int64, SemanticType::Id),
            col("qty", IngestedType::Int64, SemanticType::Numerical),
            col("price", IngestedType::Float64, SemanticType::Numerical),
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::Int64, SemanticType::Categorical),
        ];
        let streamed =
            compute_parquet_col_stats_with_max_categories(&paths, &columns, "v1", cap).unwrap();
        assert_eq!(
            streamed,
            legacy_col_stats(&paths, &columns, cap),
            "cap {cap}"
        );
        assert!(streamed.0.contains("SUMMARY_STATS_BY_CLASS"));
        assert!(streamed.0.contains("HISTOGRAM"));
    }

    #[test]
    fn configurable_category_cap_rejects_zero() {
        let error = compute_parquet_col_stats_with_max_categories(&[], &[], "v1", 0).unwrap_err();
        assert!(error
            .to_string()
            .contains("category count limit must be greater than zero"));
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
        let (full, min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();

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

        // entity is an `Id` column: its category counts are collected (capped at
        // MAX_CATEGORIES), so with n well under the cap the map is complete and
        // NUM_UNIQUE is exact rather than an HLL estimate.
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

    #[test]
    fn warehouse_physical_types_produce_distributions() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("warehouse.parquet");
        let schema = Schema::new(vec![
            Field::new("rank", DataType::Int32, false),
            Field::new("revenue", DataType::Decimal128(28, 2), false),
            Field::new("ratio", DataType::Float64, false),
            Field::new(
                "month",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("Etc/UTC".into())),
                false,
            ),
            Field::new("day", DataType::Date32, false),
        ]);
        write_parquet(
            &path,
            schema,
            vec![
                StdArc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                StdArc::new(
                    Decimal128Array::from(vec![12345, 23456])
                        .with_precision_and_scale(28, 2)
                        .unwrap(),
                ) as ArrayRef,
                StdArc::new(Float64Array::from(vec![16_777_217.0, 16_777_219.0])) as ArrayRef,
                StdArc::new(
                    TimestampMicrosecondArray::from(vec![
                        1_659_304_801_000_000,
                        1_661_983_201_000_000,
                    ])
                    .with_timezone("Etc/UTC"),
                ) as ArrayRef,
                StdArc::new(Date32Array::from(vec![19_204, 19_235])) as ArrayRef,
            ],
            1,
        );
        let columns = vec![
            col("rank", IngestedType::Int32, SemanticType::Numerical),
            col("revenue", IngestedType::Float64, SemanticType::Numerical),
            col("ratio", IngestedType::Float32, SemanticType::Numerical),
            ts_col("month", None),
            ts_col("day", None),
        ];
        let (full, _) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
        for (name, stype) in [
            ("rank", "kumo.numerical"),
            ("revenue", "kumo.numerical"),
            ("ratio", "kumo.numerical"),
            ("month", "kumo.timestamp"),
            ("day", "kumo.timestamp"),
        ] {
            assert_eq!(val(&full, &format!("{name}__{stype}__COUNT__False")), 2);
        }
        assert_eq!(val(&full, "revenue__kumo.numerical__MAX__False"), 234.56);
        assert_eq!(
            val(&full, "ratio__kumo.numerical__MAX__False"),
            16_777_220.0
        );
        let stats: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(stats.contains_key("revenue__kumo.numerical__HISTOGRAM__False"));
        assert!(stats.contains_key("day__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"));
    }

    #[test]
    fn implicit_latlong_shape_is_enforced() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("latlong.parquet");
        let mut floats = ListBuilder::new(Float64Builder::new());
        floats.values().append_value(1.5);
        floats.values().append_value(2.5);
        floats.append(true);
        floats.values().append_value(3.5);
        floats.append(true);
        floats.values().append_value(4.5);
        floats.values().append_null();
        floats.append(true);
        floats.append(false);
        let array = StdArc::new(floats.finish()) as ArrayRef;
        let schema = Schema::new(vec![Field::new(
            "location",
            array.data_type().clone(),
            true,
        )]);
        write_parquet(&path, schema, vec![array], 2);
        // Lat/long columns are fixed pairs even when the public adapter does
        // not carry a separate fixed-length field.
        let columns = [col(
            "location",
            IngestedType::Float32Array,
            SemanticType::Latlong,
        )];
        let (full, _) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
        assert_eq!(
            val(&full, "location__kumo.latlong__SEQUENCE_MAX_LENGTH__False"),
            2
        );
        assert_eq!(val(&full, "location__kumo.latlong__NUM_NA__False"), 3);
    }

    /// Float32 SCORE and numeric CLASS both normalize through shared access.
    #[test]
    fn float32_score_by_class() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        let schema = Schema::new(vec![
            Field::new("SCORE", DataType::Float32, true),
            Field::new("CLASS", DataType::Int64, true),
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
                StdArc::new(Int64Array::from(vec![Some(2), Some(1), Some(0)])) as ArrayRef,
            ],
            1024,
        );
        let columns = vec![
            col("SCORE", IngestedType::Float32, SemanticType::Numerical),
            col("CLASS", IngestedType::Bool, SemanticType::Categorical),
        ];
        let (full, _min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
        // Numerical scalar stats populated (not empty-Welford defaults).
        let mx = val(&full, "SCORE__kumo.numerical__MAX__False")
            .as_f64()
            .unwrap();
        assert!((mx - 5.0).abs() < 1e-6, "max={mx}");
        let bc = val(
            &full,
            "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False",
        );
        assert!((bc["1"]["mean"].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((bc["0"]["min"].as_f64().unwrap() - 5.0).abs() < 1e-6);
        assert!(bc.get("2").is_none());
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
        let (full, _min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
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
            compute_parquet_col_stats_with_options(
                &[p1.as_path(), p2.as_path()],
                &columns,
                "v1",
                MAX_CATEGORIES,
                &ColStatsOptions::default(),
            ).unwrap();
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
        let err = compute_parquet_col_stats_with_options(
            &[p1.as_path(), p2.as_path()],
            &[col("score", IngestedType::Float64, SemanticType::Numerical)],
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
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
        let (full, _min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
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
            compute_parquet_col_stats_with_options(
                &[path.as_path()],
                &[ts_col("ts", None)],
                "v1",
                MAX_CATEGORIES,
                &ColStatsOptions::default(),
            ).unwrap();
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
        use arrow_array::TimestampNanosecondArray;

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
            compute_parquet_col_stats_with_options(
                &[path.as_path()],
                &[ts_col("ts", None)],
                "v1",
                MAX_CATEGORIES,
                &ColStatsOptions::default(),
            ).unwrap();
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
        let (full, _min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
        let stats: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(!stats.contains_key("ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"));
        assert!(!stats.contains_key("null_ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"));
        assert!(stats.contains_key("ts__kumo.timestamp__MIN__False"));
        assert!(stats.contains_key("ts__kumo.timestamp__MAX__False"));
        assert_ne!(
            stats["ts__kumo.timestamp__MIN__False"]["value"],
            "kumo_timestamp_NaT"
        );
        assert_ne!(
            stats["ts__kumo.timestamp__MAX__False"]["value"],
            "kumo_timestamp_NaT"
        );
        assert_eq!(stats["ts__kumo.timestamp__NUM_NA__False"]["value"], 1);
        assert_eq!(
            stats["score__kumo.numerical__HISTOGRAM__False"]["value"]
                .as_array()
                .unwrap()
                .len(),
            NUMERICAL_HISTOGRAM_BINS
        );
    }

    /// Int64 epoch-ms timestamps prefer an authoritative unit and retain the
    /// legacy magnitude fallback for callers that have not started sending it.
    #[test]
    fn int64_epoch_ms_timestamp_scales_to_correct_year() {
        let dir = tempdir().unwrap();
        // Two equal-instant rows so MIN == MAX and the assertion is exact.
        let raw_ms = vec![1_659_304_801_388_i64, 1_659_304_801_388_i64];
        let expected = "kumo_timestamp_2022-07-31 22:00:02";

        let path = dir.path().join("explicit_unit.parquet");
        let schema = Schema::new(vec![Field::new("ts", DataType::Int64, true)]);
        write_parquet(
            &path,
            schema,
            vec![StdArc::new(Int64Array::from(raw_ms)) as ArrayRef],
            64,
        );
        let mut timestamp = ts_col("ts", Some(TimestampUnit::Millis));
        timestamp.ingested_dtype = IngestedType::Int64;
        let columns = vec![timestamp];
        let (full, _min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
        assert_eq!(val(&full, "ts__kumo.timestamp__MIN__False"), expected);
        assert_eq!(val(&full, "ts__kumo.timestamp__MAX__False"), expected);
        assert_eq!(
            val(&full, "ts__kumo.timestamp__HISTOGRAM_HALF_MONTH__False"),
            json!([["kumo_timestamp_2022-07-16 00:00:00", 2]])
        );

        let mut timestamp = ts_col("ts", None);
        timestamp.ingested_dtype = IngestedType::Int64;
        let (full, _min) =
            compute_parquet_col_stats_with_options(
                &[path.as_path()],
                &[timestamp],
                "v1",
                MAX_CATEGORIES,
                &ColStatsOptions::default(),
            ).unwrap();
        assert_eq!(val(&full, "ts__kumo.timestamp__MIN__False"), expected);
    }

    /// Int64 epoch-seconds uses its explicit unit without over-scaling.
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
        let columns = vec![ts_col("ts", Some(TimestampUnit::Seconds))];
        let (full, _min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
        assert_eq!(
            val(&full, "ts__kumo.timestamp__MIN__False"),
            "kumo_timestamp_2022-07-31 22:00:01"
        );
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
        let err = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap_err();
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
        let (full, _) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
        assert_eq!(
            val(&full, "SCORE__kumo.numerical__MIN__False"),
            JsonValue::Null
        );
        assert_eq!(
            val(&full, "SCORE__kumo.numerical__MAX__False"),
            JsonValue::Null
        );
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
        let (full, _min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
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

        let (full, _) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &[col("v", IngestedType::Int64, SemanticType::Numerical)],
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        )
        .unwrap();
        assert_eq!(val(&full, "v__kumo.numerical__MIN__False"), i64::MAX - 2);
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

        let (full, _) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &[col("v", IngestedType::Int64, SemanticType::Numerical)],
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
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
    fn scale_zero_decimal_int64_boundaries_are_preserved() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("decimal128.parquet");
        let values = Decimal128Array::from(vec![
            Some(i128::from(i64::MIN)),
            None,
            Some(i128::from(i64::MAX)),
        ])
        .with_precision_and_scale(38, 0)
        .unwrap();
        write_parquet(
            &path,
            Schema::new(vec![Field::new("wide_int", DataType::Decimal128(38, 0), true)]),
            vec![StdArc::new(values) as ArrayRef],
            1,
        );

        let (full, _) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &[col( "wide_int", IngestedType::Int64, SemanticType::Numerical, )],
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        )
        .unwrap();

        assert_eq!(
            val(&full, "wide_int__kumo.numerical__MIN__False"),
            i64::MIN
        );
        assert_eq!(
            val(&full, "wide_int__kumo.numerical__MAX__False"),
            i64::MAX
        );
        assert_eq!(
            val(&full, "wide_int__kumo.numerical__NUM_NA__False"),
            1
        );
    }

    #[test]
    fn malformed_string_int64_remains_parse_failure() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("string_int64.parquet");
        write_parquet(
            &path,
            Schema::new(vec![Field::new("integer", DataType::Utf8, true)]),
            vec![StdArc::new(StringArray::from(vec![
                Some("7"),
                Some("not-an-integer"),
                None,
            ])) as ArrayRef],
            1,
        );

        let (full, _) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &[col( "integer", IngestedType::Int64, SemanticType::Numerical, )],
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        )
        .unwrap();

        assert_eq!(val(&full, "integer__kumo.numerical__MIN__False"), 7);
        assert_eq!(val(&full, "integer__kumo.numerical__MAX__False"), 7);
        assert_eq!(val(&full, "integer__kumo.numerical__NUM_NA__False"), 2);
    }

    #[test]
    fn scale_zero_decimal_overflow_is_public_validation_error() {
        let dir = tempdir().unwrap();
        let cases: Vec<(&str, DataType, ArrayRef)> = vec![
            (
                "decimal128_above",
                DataType::Decimal128(38, 0),
                StdArc::new(
                    Decimal128Array::from(vec![
                        Some(i128::from(i64::MAX)),
                        None,
                        Some(i128::from(i64::MAX) + 1),
                    ])
                    .with_precision_and_scale(38, 0)
                    .unwrap(),
                ),
            ),
            (
                "decimal128_below",
                DataType::Decimal128(38, 0),
                StdArc::new(
                    Decimal128Array::from(vec![i128::from(i64::MIN) - 1])
                        .with_precision_and_scale(38, 0)
                        .unwrap(),
                ),
            ),
            (
                "decimal256_above",
                DataType::Decimal256(76, 0),
                StdArc::new(
                    Decimal256Array::from(vec![i256::from_i128(i128::from(i64::MAX) + 1)])
                        .with_precision_and_scale(76, 0)
                        .unwrap(),
                ),
            ),
            (
                "decimal256_below",
                DataType::Decimal256(76, 0),
                StdArc::new(
                    Decimal256Array::from(vec![i256::from_i128(i128::from(i64::MIN) - 1)])
                        .with_precision_and_scale(76, 0)
                        .unwrap(),
                ),
            ),
        ];

        for (name, data_type, values) in cases {
            let path = dir.path().join(format!("{name}.parquet"));
            write_parquet(
                &path,
                Schema::new(vec![Field::new("wide_int", data_type, true)]),
                vec![values],
                1,
            );
            let error = compute_parquet_col_stats_with_options(
                &[path.as_path()],
                &[col( "wide_int", IngestedType::Int64, SemanticType::Numerical, )],
                "v1",
                MAX_CATEGORIES,
                &ColStatsOptions::default(),
            )
            .unwrap_err();

            assert!(
                error.downcast_ref::<PublicIngestError>().is_some(),
                "{name}: unexpected error type: {error:#}"
            );
            assert!(
                error.to_string().contains(
                    "column 'wide_int' contains a scale-zero decimal value outside Kumo's signed \
                     64-bit integer range"
                ),
                "{name}: unexpected error: {error:#}"
            );
        }
    }

    #[test]
    fn constant_numerical_columns_omit_histograms_like_legacy_kumo() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("predictions.parquet");
        write_parquet(
            &path,
            Schema::new(vec![
                Field::new("integer", DataType::Int64, true),
                Field::new("float", DataType::Float64, true),
            ]),
            vec![
                StdArc::new(Int64Array::from(vec![7, 7])) as ArrayRef,
                StdArc::new(Float64Array::from(vec![7.0, 7.0])) as ArrayRef,
            ],
            1,
        );

        let (full, _) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &[ col("integer", IngestedType::Int64, SemanticType::Numerical), col("float", IngestedType::Float64, SemanticType::Numerical), ],
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        )
        .unwrap();
        let stats: JsonMap<String, JsonValue> = serde_json::from_str(&full).unwrap();
        assert!(["integer", "float"]
            .into_iter()
            .all(|name| !stats.contains_key(&format!("{name}__kumo.numerical__HISTOGRAM__False"))));
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

        let (full, _min) = compute_parquet_col_stats_with_options(
            &[path.as_path()],
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
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
                compute_parquet_col_stats_with_options(
                    &[path.as_path()],
                    &columns,
                    "v1",
                    MAX_CATEGORIES,
                    &ColStatsOptions::default(),
                ).unwrap();
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

    fn with_options(paths: &[&Path], columns: &[ColumnSpec], options: ColStatsOptions) -> String {
        compute_parquet_col_stats_with_options(paths, columns, "v1", MAX_CATEGORIES, &options)
            .unwrap()
            .0
    }

    fn write_scores(path: &Path, scores: Vec<f64>, rows_per_group: usize) {
        write_parquet(
            path,
            Schema::new(vec![Field::new("SCORE", DataType::Float64, true)]),
            vec![StdArc::new(Float64Array::from(scores)) as ArrayRef],
            rows_per_group,
        );
    }

    #[test]
    fn unset_options_change_nothing_and_options_are_additive() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("scores.parquet");
        let n = 3000usize;
        write_parquet(
            &path,
            Schema::new(vec![
                Field::new("SCORE", DataType::Float64, false),
                Field::new("CLASS", DataType::Utf8, false),
            ]),
            vec![
                StdArc::new(Float64Array::from(
                    (0..n).map(|i| (i % 97) as f64 / 97.0).collect::<Vec<_>>(),
                )) as ArrayRef,
                StdArc::new(StringArray::from(
                    (0..n)
                        .map(|i| format!("c{}", (i * i) % 50))
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ],
            256,
        );
        let columns = vec![
            col("SCORE", IngestedType::Float64, SemanticType::Numerical),
            col("CLASS", IngestedType::String, SemanticType::Categorical),
        ];
        let paths = [path.as_path()];
        let parse = |json: &str| serde_json::from_str::<JsonMap<String, JsonValue>>(json).unwrap();

        let (default, default_min) = compute_parquet_col_stats_with_options(
            &paths,
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        ).unwrap();
        let unset = compute_parquet_col_stats_with_options(
            &paths,
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions::default(),
        )
        .unwrap();
        assert_eq!((default.clone(), default_min.clone()), unset);

        let (full, min) = compute_parquet_col_stats_with_options(
            &paths,
            &columns,
            "v1",
            MAX_CATEGORIES,
            &ColStatsOptions {
                percentiles: true,
                summary_stats_by_class: false,
                histograms: false,
                category_counts_limit: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
        let (mut full, mut default) = (parse(&full), parse(&default));
        let percentiles = full
            .remove("SCORE__kumo.numerical__PERCENTILES__True")
            .unwrap();
        assert_eq!(percentiles["value"].as_array().unwrap().len(), 99);
        assert_eq!(percentiles["num_entries"], json!(n));
        for key in [
            "SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False",
            "SCORE__kumo.numerical__HISTOGRAM__False",
        ] {
            assert!(default.remove(key).is_some());
        }
        assert_eq!(
            full.remove("CLASS__kumo.categorical__CATEGORY_COUNTS_CAPPED__False").unwrap()["value"],
            json!(false)
        );
        let counts_key = "CLASS__kumo.categorical__CATEGORY_COUNTS__False";
        let all = default.remove(counts_key).unwrap()["value"].clone();
        let top = full.remove(counts_key).unwrap()["value"].clone();
        assert_eq!(top.as_array().unwrap()[..], all.as_array().unwrap()[..5]);
        assert_eq!(full, default);
        let mut default_min = parse(&default_min);
        default_min.remove("SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False");
        default_min.remove("SCORE__kumo.numerical__HISTOGRAM__False");
        assert_eq!(parse(&min), default_min);
    }

    #[test]
    fn capped_category_counts_are_flagged() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("classes.parquet");
        write_parquet(
            &path,
            Schema::new(vec![Field::new("CLASS", DataType::Utf8, false)]),
            vec![StdArc::new(StringArray::from(vec!["a", "b", "c", "a"])) as ArrayRef],
            100,
        );
        let columns = vec![col("CLASS", IngestedType::String, SemanticType::Categorical)];
        let options = ColStatsOptions {
            category_counts_limit: Some(1),
            ..Default::default()
        };
        for (cap, capped) in [(2, true), (3, false)] {
            let (full, _) =
                compute_parquet_col_stats_with_options(&[path.as_path()], &columns, "v1", cap, &options)
                    .unwrap();
            assert_eq!(
                val(&full, "CLASS__kumo.categorical__CATEGORY_COUNTS_CAPPED__False"),
                json!(capped)
            );
            assert_eq!(
                val(&full, "CLASS__kumo.categorical__CATEGORY_COUNTS__False"),
                json!([["a", 2]])
            );
        }
    }

    #[test]
    fn percentiles_track_exact_ranks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("scores.parquet");
        let mut scores: Vec<f64> = (1..=20_000).map(|i| (i as f64 / 2000.0).exp()).collect();
        scores.extend((1..=5_000).map(|i| -(i as f64) / 10.0));
        scores.push(f64::NAN);
        write_scores(&path, scores.clone(), 1000);
        let columns = vec![col("SCORE", IngestedType::Float64, SemanticType::Numerical)];

        let full = with_options(
            &[path.as_path()],
            &columns,
            ColStatsOptions {
                percentiles: true,
                ..Default::default()
            },
        );
        let values = val(&full, "SCORE__kumo.numerical__PERCENTILES__True");
        let mut finite: Vec<f64> = scores.into_iter().filter(|v| v.is_finite()).collect();
        finite.sort_by(f64::total_cmp);
        for (idx, value) in values.as_array().unwrap().iter().enumerate() {
            let rank = finite.partition_point(|v| *v < value.as_f64().unwrap()) as f64
                / finite.len() as f64;
            assert!(
                (rank - (idx + 1) as f64 / 100.0).abs() < 0.005,
                "q{:02}",
                idx + 1
            );
        }
    }

    #[test]
    fn first_row_per_keeps_one_row_per_run_across_row_groups_and_files() {
        let dir = tempdir().unwrap();
        let schema = Schema::new(vec![
            Field::new("ENTITY", DataType::Int64, false),
            Field::new("ANCHOR", DataType::Int64, false),
            Field::new("SCORE", DataType::Float64, false),
        ]);
        // Runs of 3 rows (entity, anchor), scores descending in each run; the
        // same entity repeats with a new anchor, and runs straddle row groups
        // (7 rows each) and the file boundary.
        let runs = 40i64;
        let entity: Vec<i64> = (0..runs * 3).map(|i| i / 6).collect();
        let anchor: Vec<i64> = (0..runs * 3).map(|i| (i / 3) % 2).collect();
        let score: Vec<f64> = (0..runs * 3)
            .map(|i| (i / 3) as f64 + 1.0 - (i % 3) as f64 * 0.25)
            .collect();
        let split = 61usize;
        let mut paths = Vec::new();
        for (idx, range) in [0..split, split..(runs * 3) as usize]
            .into_iter()
            .enumerate()
        {
            let path = dir.path().join(format!("part-{idx}.parquet"));
            write_parquet(
                &path,
                schema.clone(),
                vec![
                    StdArc::new(Int64Array::from(entity[range.clone()].to_vec())) as ArrayRef,
                    StdArc::new(Int64Array::from(anchor[range.clone()].to_vec())) as ArrayRef,
                    StdArc::new(Float64Array::from(score[range].to_vec())) as ArrayRef,
                ],
                7,
            );
            paths.push(path);
        }
        let paths: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let columns = vec![col("SCORE", IngestedType::Float64, SemanticType::Numerical)];

        let full = with_options(
            &paths,
            &columns,
            ColStatsOptions {
                percentiles: true,
                first_row_per: vec!["ENTITY".into(), "ANCHOR".into()],
                ..Default::default()
            },
        );
        assert_eq!(
            val(&full, "SCORE__kumo.numerical__COUNT__False"),
            json!(runs)
        );
        assert_eq!(val(&full, "SCORE__kumo.numerical__MIN__False"), json!(1.0));
        assert_eq!(
            val(&full, "SCORE__kumo.numerical__MEAN__False"),
            json!((runs + 1) as f64 / 2.0)
        );
        assert_eq!(
            val(&full, "SCORE__kumo.numerical__PERCENTILES__True")
                .as_array()
                .unwrap()[49],
            json!(20.5)
        );
        let histogram = val(&full, "SCORE__kumo.numerical__HISTOGRAM__False");
        let total: f64 = histogram
            .as_array()
            .unwrap()
            .iter()
            .map(|bin| bin[1].as_f64().unwrap())
            .sum();
        assert!((total - 100.0).abs() < 1e-9);

        let entity_only = with_options(
            &paths,
            &columns,
            ColStatsOptions {
                first_row_per: vec!["ENTITY".into()],
                ..Default::default()
            },
        );
        assert_eq!(
            val(&entity_only, "SCORE__kumo.numerical__COUNT__False"),
            json!(runs / 2)
        );
    }
}
