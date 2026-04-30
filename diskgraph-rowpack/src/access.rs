use crate::parser::*;
use crate::RowAccessStats;
use crate::MAX_INPUT_STRING_BYTES;
use crate::{
    apply_decimal_scale_i128, apply_decimal_scale_i256, decimal_i128_to_f64, decimal_i256_to_f64,
    format_decimal_i128, format_decimal_i256,
};
use anyhow::{anyhow, Result};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use num_traits::ToPrimitive;
use std::sync::Arc;

/// How to format binary data when converting to string via `with_string()`.
#[derive(Debug, Clone, Copy, Default)]
pub enum BinaryStringHint {
    /// Interpret bytes as UTF-8 (default, current behavior).
    #[default]
    Utf8,
    /// Format 16 bytes as UUID hex string (e.g., "c0993ca4-2804-4256-...").
    Uuid,
    /// Format 32 bytes as hex string.
    Uuid256,
    /// Format 16 bytes as IP address string.
    Ip,
}

#[derive(Debug, Clone, Default)]
pub struct RowAccessHints {
    pub timestamp_format: Option<String>,
    pub timestamp_unit: Option<arrow_schema::TimeUnit>,
    pub decimal_scale: Option<i32>,
    pub binary_string_hint: BinaryStringHint,
    /// Delimiter to use when reading a parquet `Utf8` scalar as a list.
    /// When set, scalar-to-list conversion in `string_list_view` /
    /// `int64_list_view` splits on this byte (with GE-parity drop-empty
    /// semantics) instead of JSON-parsing. Carrying this on the access layer
    /// is what lets a `multicategorical` column declared with a delimiter
    /// flow into the `IngestedType::StringArray` / `Int64Array` writer path
    /// without changing the source parquet shape.
    pub multicategorical_delimiter: Option<String>,
    /// Column name, used only for error messages on parse failure.
    pub column_name: Option<String>,
}

pub struct RowAccess<'p> {
    row: usize,
    cols_ptr: *const ArrayRef,
    cols_len: usize,
    _stats: RowAccessStats<'p>,
    hints: Vec<RowAccessHints>,
    fast_parsers: Vec<Option<TimestampFormatParser>>,
    /// Per-column pre-tokenized `ListArray<Utf8>` for multicategorical
    /// columns whose source is a `Utf8`/`LargeUtf8` scalar with a declared
    /// delimiter. Built once per batch in `set_cols` so per-row
    /// `string_list_view` calls don't have to re-split, re-allocate a
    /// `Vec<String>` per row, or memcpy each token into its own heap
    /// buffer. The whole batch's tokens land in one Arrow value buffer
    /// (one allocation), with two offsets arrays (list + value) on top.
    /// Indexed by column index; `None` for columns that don't need it.
    /// kumo-ai/kumo#28478.
    tokenized_lists: Vec<Option<ArrayRef>>,
}

pub enum FloatListValues {
    F16(ArrayRef),
    F32(ArrayRef),
    F64(ArrayRef),
    /// Parsed from a string like "[1.0, 2.0, 3.0]".
    Parsed(Vec<f32>),
}

pub struct FloatListView {
    pub start: usize,
    pub end: usize,
    pub values: FloatListValues,
}

pub enum Int64ListValues {
    Native(ArrayRef),
    /// Parsed from a string like "[1, 2, 3]".
    Parsed(Vec<i64>),
}

pub struct Int64ListView {
    pub start: usize,
    pub end: usize,
    pub values: Int64ListValues,
}

impl Int64ListView {
    #[inline]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Iterate over the non-null elements of this list. Null elements are
    /// dropped; the callback is only invoked for validity-bit=true positions.
    /// For the `Parsed` variant (which has no per-element null information)
    /// every value is emitted.
    #[inline]
    pub fn for_each_i64(&self, mut f: impl FnMut(i64)) {
        match &self.values {
            Int64ListValues::Native(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .unwrap();
                for i in self.start..self.end {
                    if !arr.is_null(i) {
                        f(arr.value(i));
                    }
                }
            }
            Int64ListValues::Parsed(vec) => {
                for &v in &vec[self.start..self.end] {
                    f(v);
                }
            }
        }
    }
}

pub enum StringListValues {
    Utf8(ArrayRef),
    LargeUtf8(ArrayRef),
    /// Parsed from a string like `["a","b","c"]`.
    Parsed(Vec<String>),
}

pub struct StringListView {
    pub start: usize,
    pub end: usize,
    pub values: StringListValues,
}

impl StringListView {
    #[inline]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Iterate over the non-null elements of this list. Null elements are
    /// dropped; the callback is only invoked for validity-bit=true positions.
    /// A legitimate empty string is preserved — only the null bit filters.
    /// For the `Parsed` variant (which has no per-element null information)
    /// every value is emitted.
    #[inline]
    pub fn for_each_str(&self, mut f: impl FnMut(&str)) {
        match &self.values {
            StringListValues::Utf8(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .unwrap();
                for i in self.start..self.end {
                    if !arr.is_null(i) {
                        f(arr.value(i));
                    }
                }
            }
            StringListValues::LargeUtf8(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .unwrap();
                for i in self.start..self.end {
                    if !arr.is_null(i) {
                        f(arr.value(i));
                    }
                }
            }
            StringListValues::Parsed(vec) => {
                for v in &vec[self.start..self.end] {
                    f(v.as_str());
                }
            }
        }
    }
}

impl FloatListView {
    /// Number of elements in this list.
    #[inline]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns true if this list has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Iterate over all values as f64, calling `f` for each.
    /// Null elements are passed as 0.0 (matching the encoding behaviour).
    #[inline]
    pub fn for_each_f64(&self, mut f: impl FnMut(f64)) {
        match &self.values {
            FloatListValues::F16(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float16Array>()
                    .unwrap();
                for i in self.start..self.end {
                    let v = if arr.is_null(i) {
                        0.0
                    } else {
                        half::f16::to_f64(arr.value(i))
                    };
                    f(v);
                }
            }
            FloatListValues::F32(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .unwrap();
                for i in self.start..self.end {
                    let v = if arr.is_null(i) {
                        0.0
                    } else {
                        arr.value(i) as f64
                    };
                    f(v);
                }
            }
            FloatListValues::F64(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .unwrap();
                for i in self.start..self.end {
                    let v = if arr.is_null(i) { 0.0 } else { arr.value(i) };
                    f(v);
                }
            }
            FloatListValues::Parsed(vec) => {
                for &v in &vec[self.start..self.end] {
                    f(v as f64);
                }
            }
        }
    }

    /// Iterate over non-null values as f64, calling `f` for each present value.
    /// This is for stats paths where null list elements should be dropped,
    /// without changing RowPack's encoding-time null-as-zero behaviour.
    #[inline]
    pub fn for_each_non_null_f64(&self, mut f: impl FnMut(f64)) {
        match &self.values {
            FloatListValues::F16(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float16Array>()
                    .unwrap();
                for i in self.start..self.end {
                    if !arr.is_null(i) {
                        f(half::f16::to_f64(arr.value(i)));
                    }
                }
            }
            FloatListValues::F32(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .unwrap();
                for i in self.start..self.end {
                    if !arr.is_null(i) {
                        f(arr.value(i) as f64);
                    }
                }
            }
            FloatListValues::F64(arr) => {
                let arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .unwrap();
                for i in self.start..self.end {
                    if !arr.is_null(i) {
                        f(arr.value(i));
                    }
                }
            }
            FloatListValues::Parsed(vec) => {
                for &v in &vec[self.start..self.end] {
                    f(v as f64);
                }
            }
        }
    }
}

/// Parse a JSON-style string like `"[1.0, 2.0, 3.0]"` or `"1.0, 2.0, 3.0"`
/// into a `Vec<f32>`. Uses `fast_float` for ~5x faster parsing than stdlib.
///
/// Returns `Ok(vec![])` for empty/whitespace-only strings or `"[]"`.
/// Returns `Err` for malformed strings (hard error).
/// Parse a JSON-encoded string like "[1, -2, 300]" into a Vec<i64>.
/// Accepts bracket-wrapped or bare comma-separated integers.
/// Empty string / empty brackets → empty vec.
/// Non-integer tokens → hard error.
pub fn parse_string_as_int64_array(s: &str) -> Result<Vec<i64>> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let inner = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(s)
        .trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|tok| {
            let tok = tok.trim();
            tok.parse::<i64>()
                .map_err(|_| anyhow!("failed to parse int64 from {:?}", tok))
        })
        .collect()
}

/// Split `s` on `delim` and drop empty tokens (leading, trailing, middle).
/// Matches GraphEngine PySpark behavior:
/// `pyspark.sql.functions.split(col, sep)` then `filter(elem != '')`
/// (`kumo/data/data_processing_backend/pyspark_processing_backend.py:267`).
/// Examples: `""` → `[]`, `"a,"` → `["a"]`, `"a,,b"` → `["a","b"]`,
/// `",a,"` → `["a"]`.
pub fn split_multicat_string(s: &str, delim: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(delim)
        .filter(|tok| !tok.is_empty())
        .map(|tok| tok.to_string())
        .collect()
}

/// Build a `ListArray<Utf8>` over an entire batch by tokenizing every row
/// of a parquet `Utf8`/`LargeUtf8` scalar column on the declared delimiter
/// (drop-empty, GE-parity). All non-empty tokens across all rows go into a
/// single shared value buffer; the per-row token boundaries land in one
/// list-offsets array and the per-token byte boundaries in one value-offsets
/// array. Net allocations for the whole batch: one growing-bytes buffer +
/// two offsets buffers + one validity buffer, all amortized across N rows
/// regardless of token count. Replaces the per-row `Vec<String>` /
/// `String::to_string()` allocator round-trips that dominated the original
/// post-#28478 ingest path. kumo-ai/kumo#28478.
fn build_multicat_list_array(arr: &dyn Array, delim: &str) -> Result<ArrayRef> {
    use arrow_array::builder::{ListBuilder, StringBuilder};
    let mut builder = ListBuilder::new(StringBuilder::new());
    if let Some(s) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        for row in 0..s.len() {
            if s.is_null(row) {
                builder.append(false);
                continue;
            }
            let cell = s.value(row);
            if !cell.is_empty() {
                for tok in cell.split(delim).filter(|t| !t.is_empty()) {
                    builder.values().append_value(tok);
                }
            }
            builder.append(true);
        }
        return Ok(Arc::new(builder.finish()));
    }
    if let Some(s) = arr.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
        for row in 0..s.len() {
            if s.is_null(row) {
                builder.append(false);
                continue;
            }
            let cell = s.value(row);
            if !cell.is_empty() {
                for tok in cell.split(delim).filter(|t| !t.is_empty()) {
                    builder.values().append_value(tok);
                }
            }
            builder.append(true);
        }
        return Ok(Arc::new(builder.finish()));
    }
    Err(anyhow!(
        "build_multicat_list_array: expected Utf8/LargeUtf8 scalar source"
    ))
}

/// Same split as `split_multicat_string`, but parses each token as i64.
/// Returns a structured error with row + token + column context on the first
/// parse failure; the caller is expected to surface it directly to the user.
pub fn split_multicat_string_as_int64(
    s: &str,
    delim: &str,
    column: Option<&str>,
    row: usize,
) -> Result<Vec<i64>> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(delim)
        .filter(|tok| !tok.is_empty())
        .map(|tok| {
            tok.parse::<i64>().map_err(|_| {
                anyhow!(
                    "ingest type mismatch: column={} row={} declared=Int64Array \
                     but parquet token {:?} could not be parsed as i64 \
                     (sample row text: {:?})",
                    column.unwrap_or("?"),
                    row,
                    tok,
                    s,
                )
            })
        })
        .collect()
}

/// Parse a JSON-encoded string like `["a","b","c"]` into a Vec<String>.
/// Uses serde_json for correctness (handles escapes, nested quotes, etc.).
/// Empty array → empty vec. Non-array JSON → hard error.
pub fn parse_string_as_string_array(s: &str) -> Result<Vec<String>> {
    let s = s.trim();
    if s.is_empty() || s == "[]" {
        return Ok(Vec::new());
    }
    let value: serde_json::Value = serde_json::from_str(s)
        .map_err(|e| anyhow!("failed to parse string array from {:?}: {}", s, e))?;
    let arr = value
        .as_array()
        .ok_or_else(|| anyhow!("expected JSON array, got {:?}", s))?;
    arr.iter()
        .map(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow!("expected string element, got {:?}", v))
        })
        .collect()
}

pub fn parse_string_as_float32_array(s: &str) -> Result<Vec<f32>> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    // Strip optional brackets
    let inner = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(s)
        .trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|tok| {
            let tok = tok.trim();
            fast_float::parse::<f32, _>(tok)
                .map_err(|_| anyhow!("failed to parse float from {:?}", tok))
        })
        .collect()
}

impl<'p> RowAccess<'p> {
    pub fn new(hints: Vec<RowAccessHints>, stats: RowAccessStats<'p>) -> Self {
        let fast_parsers = hints
            .iter()
            .map(|hint| {
                hint.timestamp_format
                    .as_deref()
                    .and_then(TimestampFormatParser::compile)
            })
            .collect();
        let n = hints.len();
        Self {
            row: 0,
            cols_ptr: std::ptr::null(),
            cols_len: 0,
            _stats: stats,
            hints,
            fast_parsers,
            tokenized_lists: vec![None; n],
        }
    }

    pub fn set_cols(&mut self, cols: &[ArrayRef]) {
        self.cols_ptr = cols.as_ptr();
        self.cols_len = cols.len();
        // Reset and rebuild the per-column pre-tokenized list cache.
        // Any column whose hint declares a `multicategorical_delimiter` and
        // whose source is a parquet `Utf8`/`LargeUtf8` scalar gets its
        // entire batch's tokens flattened into a single `ListArray<Utf8>`
        // up-front. Subsequent `string_list_view(idx)` calls then read row
        // ranges out of the cached list — one allocation amortized across
        // the whole batch instead of `Vec<String>` per row. kumo-ai/kumo#28478.
        for slot in self.tokenized_lists.iter_mut() {
            *slot = None;
        }
        for (idx, hint) in self.hints.iter().enumerate() {
            let Some(delim) = hint.multicategorical_delimiter.as_deref() else {
                continue;
            };
            let Some(arr) = cols.get(idx) else { continue };
            let dt = arr.data_type();
            if !matches!(dt, DataType::Utf8 | DataType::LargeUtf8) {
                continue;
            }
            // On build failure, leave the slot `None` and let the per-row
            // path in `string_list_view` hit the same failure with row
            // context. Avoids surfacing batch-level errors at `set_cols`
            // time where the caller may not be ready for them.
            self.tokenized_lists[idx] = build_multicat_list_array(arr.as_ref(), delim).ok();
        }
    }

    pub fn set_row(&mut self, row: usize) {
        self.row = row;
    }

    pub fn row(&self) -> usize {
        self.row
    }

    pub fn array(&self, idx: usize) -> Result<&ArrayRef> {
        if self.cols_ptr.is_null() {
            return Err(anyhow!("row access columns not set"));
        }
        let cols = unsafe { std::slice::from_raw_parts(self.cols_ptr, self.cols_len) };
        cols.get(idx)
            .ok_or_else(|| anyhow!("missing column {}", idx))
    }

    fn hint(&self, idx: usize) -> Result<&RowAccessHints> {
        self.hints
            .get(idx)
            .ok_or_else(|| anyhow!("missing column hint {}", idx))
    }

    pub fn get_int64(&self, idx: usize) -> Result<Option<i64>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string cast"))?;
                parse_i64(arr.value(self.row))
                    .ok_or_else(|| anyhow!("int64 parse failed"))
                    .map(Some)
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .ok_or_else(|| anyhow!("large string cast"))?;
                parse_i64(arr.value(self.row))
                    .ok_or_else(|| anyhow!("int64 parse failed"))
                    .map(Some)
            }
            DataType::Binary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or_else(|| anyhow!("binary cast"))?;
                parse_i64(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("int64 parse failed"))
                    .map(Some)
            }
            DataType::LargeBinary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| anyhow!("large binary cast"))?;
                parse_i64(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("int64 parse failed"))
                    .map(Some)
            }
            DataType::FixedSizeBinary(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                    .ok_or_else(|| anyhow!("fixed binary cast"))?;
                parse_i64(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("int64 parse failed"))
                    .map(Some)
            }
            DataType::Decimal128(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?;
                let value = arr.value(self.row);
                let scaled = apply_decimal_scale_i128(value, i32::from(*scale));
                Ok(Some(scaled as i64))
            }
            DataType::Decimal256(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?;
                let value = arr.value(self.row);
                let scaled = apply_decimal_scale_i256(value, i32::from(*scale));
                scaled
                    .to_i64()
                    .ok_or_else(|| anyhow!("decimal256 overflow"))
                    .map(Some)
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                Ok(Some(arr.value(self.row) as i64))
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?;
                Ok(Some(if arr.value(self.row) { 1 } else { 0 }))
            }
            _ => Err(anyhow!("int64 type mismatch")),
        }
    }

    pub fn get_int32(&self, idx: usize) -> Result<Option<i32>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string cast"))?;
                parse_i32(arr.value(self.row))
                    .ok_or_else(|| anyhow!("int32 parse failed"))
                    .map(Some)
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .ok_or_else(|| anyhow!("large string cast"))?;
                parse_i32(arr.value(self.row))
                    .ok_or_else(|| anyhow!("int32 parse failed"))
                    .map(Some)
            }
            DataType::Binary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or_else(|| anyhow!("binary cast"))?;
                parse_i32(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("int32 parse failed"))
                    .map(Some)
            }
            DataType::LargeBinary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| anyhow!("large binary cast"))?;
                parse_i32(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("int32 parse failed"))
                    .map(Some)
            }
            DataType::FixedSizeBinary(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                    .ok_or_else(|| anyhow!("fixed binary cast"))?;
                parse_i32(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("int32 parse failed"))
                    .map(Some)
            }
            DataType::Decimal128(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?;
                let value = arr.value(self.row);
                let scaled = apply_decimal_scale_i128(value, i32::from(*scale));
                Ok(Some(scaled as i32))
            }
            DataType::Decimal256(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?;
                let value = arr.value(self.row);
                let scaled = apply_decimal_scale_i256(value, i32::from(*scale));
                scaled
                    .to_i32()
                    .ok_or_else(|| anyhow!("decimal256 overflow"))
                    .map(Some)
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                Ok(Some(arr.value(self.row) as i32))
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?;
                Ok(Some(if arr.value(self.row) { 1 } else { 0 }))
            }
            _ => Err(anyhow!("int32 type mismatch")),
        }
    }

    pub fn get_uint64(&self, idx: usize) -> Result<Option<u64>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string cast"))?;
                parse_u64(arr.value(self.row))
                    .ok_or_else(|| anyhow!("uint64 parse failed"))
                    .map(Some)
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .ok_or_else(|| anyhow!("large string cast"))?;
                parse_u64(arr.value(self.row))
                    .ok_or_else(|| anyhow!("uint64 parse failed"))
                    .map(Some)
            }
            DataType::Binary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or_else(|| anyhow!("binary cast"))?;
                parse_u64(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("uint64 parse failed"))
                    .map(Some)
            }
            DataType::LargeBinary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| anyhow!("large binary cast"))?;
                parse_u64(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("uint64 parse failed"))
                    .map(Some)
            }
            DataType::FixedSizeBinary(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                    .ok_or_else(|| anyhow!("fixed binary cast"))?;
                parse_u64(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("uint64 parse failed"))
                    .map(Some)
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                Ok(Some(arr.value(self.row) as u64))
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?;
                Ok(Some(if arr.value(self.row) { 1 } else { 0 }))
            }
            DataType::Decimal128(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?;
                let value = arr.value(self.row);
                let scaled = apply_decimal_scale_i128(value, i32::from(*scale));
                Ok(Some(scaled as u64))
            }
            DataType::Decimal256(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?;
                let value = arr.value(self.row);
                let scaled = apply_decimal_scale_i256(value, i32::from(*scale));
                scaled
                    .to_u64()
                    .ok_or_else(|| anyhow!("decimal256 overflow"))
                    .map(Some)
            }
            _ => Err(anyhow!("uint64 type mismatch")),
        }
    }

    pub fn get_uint32(&self, idx: usize) -> Result<Option<u32>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string cast"))?;
                parse_u32(arr.value(self.row))
                    .ok_or_else(|| anyhow!("uint32 parse failed"))
                    .map(Some)
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .ok_or_else(|| anyhow!("large string cast"))?;
                parse_u32(arr.value(self.row))
                    .ok_or_else(|| anyhow!("uint32 parse failed"))
                    .map(Some)
            }
            DataType::Binary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or_else(|| anyhow!("binary cast"))?;
                parse_u32(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("uint32 parse failed"))
                    .map(Some)
            }
            DataType::LargeBinary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| anyhow!("large binary cast"))?;
                parse_u32(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("uint32 parse failed"))
                    .map(Some)
            }
            DataType::FixedSizeBinary(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                    .ok_or_else(|| anyhow!("fixed binary cast"))?;
                parse_u32(std::str::from_utf8(arr.value(self.row))?)
                    .ok_or_else(|| anyhow!("uint32 parse failed"))
                    .map(Some)
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                Ok(Some(arr.value(self.row) as u32))
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?;
                Ok(Some(if arr.value(self.row) { 1 } else { 0 }))
            }
            DataType::Decimal128(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?;
                let value = arr.value(self.row);
                let scaled = apply_decimal_scale_i128(value, i32::from(*scale));
                Ok(Some(scaled as u32))
            }
            DataType::Decimal256(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?;
                let value = arr.value(self.row);
                let scaled = apply_decimal_scale_i256(value, i32::from(*scale));
                scaled
                    .to_u32()
                    .ok_or_else(|| anyhow!("decimal256 overflow"))
                    .map(Some)
            }
            _ => Err(anyhow!("uint32 type mismatch")),
        }
    }

    pub fn get_float64(&self, idx: usize) -> Result<Option<f64>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                Ok(Some(arr.value(self.row) as f64))
            }
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_) => {
                let text = match array.data_type() {
                    DataType::Utf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?
                        .value(self.row),
                    DataType::LargeUtf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .ok_or_else(|| anyhow!("large string cast"))?
                        .value(self.row),
                    DataType::Binary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .ok_or_else(|| anyhow!("binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::LargeBinary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::LargeBinaryArray>()
                            .ok_or_else(|| anyhow!("large binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::FixedSizeBinary(_) => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                            .ok_or_else(|| anyhow!("fixed binary cast"))?
                            .value(self.row),
                    )?,
                    _ => return Err(anyhow!("float64 type mismatch")),
                };
                parse_f64(text)
                    .ok_or_else(|| anyhow!("float64 parse failed"))
                    .map(Some)
            }
            DataType::Decimal128(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?;
                let value = arr.value(self.row);
                let scaled = decimal_i128_to_f64(value, i32::from(*scale));
                Ok(Some(scaled))
            }
            DataType::Decimal256(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?;
                let value = arr.value(self.row);
                decimal_i256_to_f64(value, i32::from(*scale))
                    .ok_or_else(|| anyhow!("decimal256 overflow"))
                    .map(Some)
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?;
                Ok(Some(if arr.value(self.row) { 1.0 } else { 0.0 }))
            }
            _ => Err(anyhow!("float64 type mismatch")),
        }
    }

    pub fn get_float32(&self, idx: usize) -> Result<Option<f32>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                Ok(Some(arr.value(self.row) as f32))
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?;
                Ok(Some(if arr.value(self.row) { 1.0 } else { 0.0 }))
            }
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_) => {
                let text = match array.data_type() {
                    DataType::Utf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?
                        .value(self.row),
                    DataType::LargeUtf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .ok_or_else(|| anyhow!("large string cast"))?
                        .value(self.row),
                    DataType::Binary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .ok_or_else(|| anyhow!("binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::LargeBinary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::LargeBinaryArray>()
                            .ok_or_else(|| anyhow!("large binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::FixedSizeBinary(_) => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                            .ok_or_else(|| anyhow!("fixed binary cast"))?
                            .value(self.row),
                    )?,
                    _ => return Err(anyhow!("float32 type mismatch")),
                };
                parse_f32(text)
                    .ok_or_else(|| anyhow!("float32 parse failed"))
                    .map(Some)
            }
            DataType::Decimal128(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?;
                let value = arr.value(self.row);
                let scaled = decimal_i128_to_f64(value, i32::from(*scale));
                Ok(Some(scaled as f32))
            }
            DataType::Decimal256(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?;
                let value = arr.value(self.row);
                decimal_i256_to_f64(value, i32::from(*scale))
                    .map(|v| v as f32)
                    .ok_or_else(|| anyhow!("decimal256 overflow"))
                    .map(Some)
            }
            _ => Err(anyhow!("float32 type mismatch")),
        }
    }

    pub fn get_bool(&self, idx: usize) -> Result<Option<bool>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                Ok(Some(arr.value(self.row) != 0))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                Ok(Some(arr.value(self.row) != 0))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                Ok(Some(arr.value(self.row) != 0))
            }
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                Ok(Some(arr.value(self.row) != 0))
            }
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                Ok(Some(arr.value(self.row) != 0))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                Ok(Some(arr.value(self.row) != 0))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                Ok(Some(arr.value(self.row) != 0))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                Ok(Some(arr.value(self.row) != 0))
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                let value = arr.value(self.row);
                Ok(Some(!value.is_nan() && value != 0.0))
            }
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                let value = arr.value(self.row);
                Ok(Some(!value.is_nan() && value != 0.0))
            }
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_) => {
                let text = match array.data_type() {
                    DataType::Utf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?
                        .value(self.row),
                    DataType::LargeUtf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .ok_or_else(|| anyhow!("large string cast"))?
                        .value(self.row),
                    DataType::Binary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .ok_or_else(|| anyhow!("binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::LargeBinary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::LargeBinaryArray>()
                            .ok_or_else(|| anyhow!("large binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::FixedSizeBinary(_) => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                            .ok_or_else(|| anyhow!("fixed binary cast"))?
                            .value(self.row),
                    )?,
                    _ => return Err(anyhow!("bool type mismatch")),
                };
                let value = parse_bool(text).ok_or_else(|| anyhow!("bool parse failed"))?;
                Ok(Some(value))
            }
            _ => Err(anyhow!("bool type mismatch")),
        }
    }

    pub fn with_string<T>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&str) -> Result<T>,
    ) -> Result<Option<T>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        let trimmed = self._stats.trimmed_strings;
        let value = match array.data_type() {
            DataType::Utf8 => array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string cast"))?
                .value(self.row),
            DataType::LargeUtf8 => array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| anyhow!("large string cast"))?
                .value(self.row),
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => {
                let bytes = match array.data_type() {
                    DataType::Binary => array
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .ok_or_else(|| anyhow!("binary cast"))?
                        .value(self.row),
                    DataType::LargeBinary => array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeBinaryArray>()
                        .ok_or_else(|| anyhow!("large binary cast"))?
                        .value(self.row),
                    _ => array
                        .as_any()
                        .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                        .ok_or_else(|| anyhow!("fixed binary cast"))?
                        .value(self.row),
                };
                match self
                    .hints
                    .get(idx)
                    .map(|h| h.binary_string_hint)
                    .unwrap_or_default()
                {
                    BinaryStringHint::Uuid if bytes.len() == 16 => {
                        let s = format!(
                            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                            bytes[0], bytes[1], bytes[2], bytes[3],
                            bytes[4], bytes[5], bytes[6], bytes[7],
                            bytes[8], bytes[9], bytes[10], bytes[11],
                            bytes[12], bytes[13], bytes[14], bytes[15]
                        );
                        return Ok(Some(f(&s)?));
                    }
                    BinaryStringHint::Uuid256 if bytes.len() == 32 => {
                        use std::fmt::Write;
                        let mut s = String::with_capacity(64);
                        for b in bytes {
                            write!(s, "{:02x}", b).unwrap();
                        }
                        return Ok(Some(f(&s)?));
                    }
                    BinaryStringHint::Ip if bytes.len() == 4 => {
                        let v4 = std::net::Ipv4Addr::new(bytes[0], bytes[1], bytes[2], bytes[3]);
                        let s = v4.to_string();
                        return Ok(Some(f(&s)?));
                    }
                    BinaryStringHint::Ip if bytes.len() == 16 => {
                        let v6 = std::net::Ipv6Addr::from(<[u8; 16]>::try_from(bytes).unwrap());
                        let s = match v6.to_ipv4_mapped() {
                            Some(v4) => v4.to_string(),
                            None => v6.to_string(),
                        };
                        return Ok(Some(f(&s)?));
                    }
                    _ => std::str::from_utf8(bytes)?,
                }
            }
            DataType::Int64 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::Int32 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::Int16 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::Int8 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::UInt64 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::UInt32 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::UInt16 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::UInt8 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::Float64 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::Float32 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::Boolean => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .ok_or_else(|| anyhow!("bool cast"))?
                    .value(self.row);
                return Ok(Some(f(if value { "true" } else { "false" })?));
            }
            DataType::Date32 => {
                let value = array
                    .as_any()
                    .downcast_ref::<arrow_array::Date32Array>()
                    .ok_or_else(|| anyhow!("date cast"))?
                    .value(self.row);
                return Ok(Some(with_stack_str(value, trimmed, f)?));
            }
            DataType::Timestamp(_, _) => {
                let secs = timestamp_secs_from_array(array, self.row)?;
                return Ok(Some(with_stack_str(secs, trimmed, f)?));
            }
            DataType::Decimal128(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?;
                let value = arr.value(self.row);
                let text = format_decimal_i128(value, i32::from(*scale));
                return Ok(Some(with_stack_str(text, trimmed, f)?));
            }
            DataType::Decimal256(_, scale) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?;
                let value = arr.value(self.row);
                let text = format_decimal_i256(value, i32::from(*scale));
                return Ok(Some(with_stack_str(text, trimmed, f)?));
            }
            _ => return Err(anyhow!("string type mismatch")),
        };
        let value = trim_borrowed_str(value, MAX_INPUT_STRING_BYTES, trimmed);
        Ok(Some(f(value)?))
    }

    pub fn with_bytes<T>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<Option<T>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        let trimmed = self._stats.trimmed_strings;
        let data = match array.data_type() {
            DataType::Utf8 => array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string cast"))?
                .value(self.row)
                .as_bytes(),
            DataType::LargeUtf8 => array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| anyhow!("large string cast"))?
                .value(self.row)
                .as_bytes(),
            DataType::Binary => array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| anyhow!("binary cast"))?
                .value(self.row),
            DataType::LargeBinary => array
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .ok_or_else(|| anyhow!("large binary cast"))?
                .value(self.row),
            DataType::FixedSizeBinary(_) => array
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                .ok_or_else(|| anyhow!("fixed binary cast"))?
                .value(self.row),
            _ => return Err(anyhow!("bytes type mismatch")),
        };
        let data = trim_borrowed_bytes(data, MAX_INPUT_STRING_BYTES, trimmed);
        Ok(Some(f(data)?))
    }

    pub fn get_uuid(&self, idx: usize) -> Result<Option<[u8; 16]>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string cast"))?;
                parse_uuid_string(arr.value(self.row).as_bytes())
                    .ok_or_else(|| anyhow!("rowpack uuid parse failed"))
                    .map(Some)
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .ok_or_else(|| anyhow!("large string cast"))?;
                parse_uuid_string(arr.value(self.row).as_bytes())
                    .ok_or_else(|| anyhow!("rowpack uuid parse failed"))
                    .map(Some)
            }
            DataType::Binary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or_else(|| anyhow!("binary cast"))?;
                let data = arr.value(self.row);
                if data.len() != 16 {
                    return Err(anyhow!("rowpack uuid binary length mismatch"));
                }
                let mut out = [0u8; 16];
                out.copy_from_slice(data);
                Ok(Some(out))
            }
            DataType::LargeBinary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| anyhow!("large binary cast"))?;
                let data = arr.value(self.row);
                if data.len() != 16 {
                    return Err(anyhow!("rowpack uuid binary length mismatch"));
                }
                let mut out = [0u8; 16];
                out.copy_from_slice(data);
                Ok(Some(out))
            }
            DataType::FixedSizeBinary(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                    .ok_or_else(|| anyhow!("fixed binary cast"))?;
                let data = arr.value(self.row);
                if data.len() != 16 {
                    return Err(anyhow!("rowpack uuid binary length mismatch"));
                }
                let mut out = [0u8; 16];
                out.copy_from_slice(data);
                Ok(Some(out))
            }
            _ => Err(anyhow!("uuid type mismatch")),
        }
    }

    pub fn get_uuid256(&self, idx: usize) -> Result<Option<[u8; 32]>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string cast"))?;
                parse_uuid256_string(arr.value(self.row).as_bytes())
                    .ok_or_else(|| anyhow!("rowpack uuid256 parse failed"))
                    .map(Some)
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .ok_or_else(|| anyhow!("large string cast"))?;
                parse_uuid256_string(arr.value(self.row).as_bytes())
                    .ok_or_else(|| anyhow!("rowpack uuid256 parse failed"))
                    .map(Some)
            }
            DataType::Binary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or_else(|| anyhow!("binary cast"))?;
                let data = arr.value(self.row);
                if data.len() != 32 {
                    return Err(anyhow!("rowpack uuid256 binary length mismatch"));
                }
                let mut out = [0u8; 32];
                out.copy_from_slice(data);
                Ok(Some(out))
            }
            DataType::LargeBinary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| anyhow!("large binary cast"))?;
                let data = arr.value(self.row);
                if data.len() != 32 {
                    return Err(anyhow!("rowpack uuid256 binary length mismatch"));
                }
                let mut out = [0u8; 32];
                out.copy_from_slice(data);
                Ok(Some(out))
            }
            DataType::FixedSizeBinary(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                    .ok_or_else(|| anyhow!("fixed binary cast"))?;
                let data = arr.value(self.row);
                if data.len() != 32 {
                    return Err(anyhow!("rowpack uuid256 binary length mismatch"));
                }
                let mut out = [0u8; 32];
                out.copy_from_slice(data);
                Ok(Some(out))
            }
            _ => Err(anyhow!("uuid256 type mismatch")),
        }
    }

    pub fn get_ip(&self, idx: usize) -> Result<Option<[u8; 16]>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string cast"))?;
                parse_ip_string(arr.value(self.row).as_bytes())
                    .ok_or_else(|| anyhow!("rowpack ip parse failed"))
                    .map(Some)
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .ok_or_else(|| anyhow!("large string cast"))?;
                parse_ip_string(arr.value(self.row).as_bytes())
                    .ok_or_else(|| anyhow!("rowpack ip parse failed"))
                    .map(Some)
            }
            DataType::Binary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .ok_or_else(|| anyhow!("binary cast"))?;
                parse_ip_bytes(arr.value(self.row))
                    .ok_or_else(|| anyhow!("rowpack ip binary length mismatch"))
                    .map(Some)
            }
            DataType::LargeBinary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| anyhow!("large binary cast"))?;
                parse_ip_bytes(arr.value(self.row))
                    .ok_or_else(|| anyhow!("rowpack ip binary length mismatch"))
                    .map(Some)
            }
            DataType::FixedSizeBinary(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                    .ok_or_else(|| anyhow!("fixed binary cast"))?;
                parse_ip_bytes(arr.value(self.row))
                    .ok_or_else(|| anyhow!("rowpack ip binary length mismatch"))
                    .map(Some)
            }
            _ => Err(anyhow!("ip type mismatch")),
        }
    }

    pub fn get_date(&self, idx: usize) -> Result<Option<i32>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        let hint = self.hint(idx)?;
        let format = hint.timestamp_format.as_deref();
        let unit = hint.timestamp_unit;
        let fast = self.fast_parsers.get(idx).and_then(|v| v.as_ref());
        match array.data_type() {
            DataType::Date32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Date32Array>()
                    .ok_or_else(|| anyhow!("date cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_) => {
                let text = match array.data_type() {
                    DataType::Utf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?
                        .value(self.row),
                    DataType::LargeUtf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .ok_or_else(|| anyhow!("large string cast"))?
                        .value(self.row),
                    DataType::Binary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .ok_or_else(|| anyhow!("binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::LargeBinary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::LargeBinaryArray>()
                            .ok_or_else(|| anyhow!("large binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::FixedSizeBinary(_) => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                            .ok_or_else(|| anyhow!("fixed binary cast"))?
                            .value(self.row),
                    )?,
                    _ => return Err(anyhow!("date type mismatch")),
                };
                if let Some(parser) = fast {
                    if let Some(days) = parser.parse_date(text) {
                        return Ok(Some(days));
                    }
                }
                if let Some(days) = parse_date_text(text, format) {
                    return Ok(Some(days));
                }
                if let Some(value) = parse_i64(text) {
                    let secs = timestamp_from_int(value, unit);
                    return Ok(Some(days_from_secs(secs)));
                }
                Err(anyhow!("date parse failed"))
            }
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => {
                let value = match array.data_type() {
                    DataType::Int64 => array
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .ok_or_else(|| anyhow!("int64 cast"))?
                        .value(self.row),
                    DataType::Int32 => array
                        .as_any()
                        .downcast_ref::<arrow_array::Int32Array>()
                        .ok_or_else(|| anyhow!("int32 cast"))?
                        .value(self.row) as i64,
                    DataType::Int16 => array
                        .as_any()
                        .downcast_ref::<arrow_array::Int16Array>()
                        .ok_or_else(|| anyhow!("int16 cast"))?
                        .value(self.row) as i64,
                    DataType::Int8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::Int8Array>()
                        .ok_or_else(|| anyhow!("int8 cast"))?
                        .value(self.row) as i64,
                    DataType::UInt64 => array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt64Array>()
                        .ok_or_else(|| anyhow!("uint64 cast"))?
                        .value(self.row) as i64,
                    DataType::UInt32 => array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt32Array>()
                        .ok_or_else(|| anyhow!("uint32 cast"))?
                        .value(self.row) as i64,
                    DataType::UInt16 => array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt16Array>()
                        .ok_or_else(|| anyhow!("uint16 cast"))?
                        .value(self.row) as i64,
                    DataType::UInt8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt8Array>()
                        .ok_or_else(|| anyhow!("uint8 cast"))?
                        .value(self.row) as i64,
                    _ => return Err(anyhow!("date type mismatch")),
                };
                let secs = timestamp_from_int(value, unit);
                Ok(Some(days_from_secs(secs)))
            }
            DataType::Timestamp(_, _) => {
                let secs = timestamp_secs_from_array(array, self.row)?;
                Ok(Some(days_from_secs(secs)))
            }
            _ => Err(anyhow!("date type mismatch")),
        }
    }

    pub fn get_timestamp(&self, idx: usize) -> Result<Option<i64>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        let hint = self.hint(idx)?;
        let format = hint.timestamp_format.as_deref();
        let unit = hint.timestamp_unit;
        let fast = self.fast_parsers.get(idx).and_then(|v| v.as_ref());
        match array.data_type() {
            DataType::Timestamp(_, _) => {
                let secs = timestamp_secs_from_array(array, self.row)?;
                Ok(Some(secs))
            }
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_) => {
                let text = match array.data_type() {
                    DataType::Utf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?
                        .value(self.row),
                    DataType::LargeUtf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .ok_or_else(|| anyhow!("large string cast"))?
                        .value(self.row),
                    DataType::Binary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .ok_or_else(|| anyhow!("binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::LargeBinary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::LargeBinaryArray>()
                            .ok_or_else(|| anyhow!("large binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::FixedSizeBinary(_) => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                            .ok_or_else(|| anyhow!("fixed binary cast"))?
                            .value(self.row),
                    )?,
                    _ => return Err(anyhow!("timestamp type mismatch")),
                };
                if let Some(parser) = fast {
                    if let Some(secs) = parser.parse_timestamp(text) {
                        return Ok(Some(secs));
                    }
                }
                if let Some(secs) = parse_timestamp_text(text, format) {
                    return Ok(Some(secs));
                }
                if let Some(value) = parse_i64(text) {
                    return Ok(Some(timestamp_from_int(value, unit)));
                }
                Err(anyhow!("timestamp parse failed"))
            }
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => {
                let value = match array.data_type() {
                    DataType::Int64 => array
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .ok_or_else(|| anyhow!("int64 cast"))?
                        .value(self.row),
                    DataType::Int32 => array
                        .as_any()
                        .downcast_ref::<arrow_array::Int32Array>()
                        .ok_or_else(|| anyhow!("int32 cast"))?
                        .value(self.row) as i64,
                    DataType::Int16 => array
                        .as_any()
                        .downcast_ref::<arrow_array::Int16Array>()
                        .ok_or_else(|| anyhow!("int16 cast"))?
                        .value(self.row) as i64,
                    DataType::Int8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::Int8Array>()
                        .ok_or_else(|| anyhow!("int8 cast"))?
                        .value(self.row) as i64,
                    DataType::UInt64 => array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt64Array>()
                        .ok_or_else(|| anyhow!("uint64 cast"))?
                        .value(self.row) as i64,
                    DataType::UInt32 => array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt32Array>()
                        .ok_or_else(|| anyhow!("uint32 cast"))?
                        .value(self.row) as i64,
                    DataType::UInt16 => array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt16Array>()
                        .ok_or_else(|| anyhow!("uint16 cast"))?
                        .value(self.row) as i64,
                    DataType::UInt8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt8Array>()
                        .ok_or_else(|| anyhow!("uint8 cast"))?
                        .value(self.row) as i64,
                    _ => return Err(anyhow!("timestamp type mismatch")),
                };
                // If a format hint is set (e.g., "%Y%m%d"), convert the integer to
                // a string and parse with the format. This handles date-encoded
                // integers like YYYYMMDD (20240301) that are NOT epoch values.
                if let Some(fmt) = format {
                    let text = value.to_string();
                    if let Some(secs) = parse_timestamp_text(&text, Some(fmt)) {
                        return Ok(Some(secs));
                    }
                }
                Ok(Some(timestamp_from_int(value, unit)))
            }
            DataType::Date32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Date32Array>()
                    .ok_or_else(|| anyhow!("date cast"))?;
                let days = arr.value(self.row);
                // Reject dates outside safe range.
                // Lower: pandas Timestamp.min (~1677-09-21 = day -106752)
                // Upper: i32 seconds overflow (~2038-01-19 = day 24855)
                // Sort keys use i32 epoch seconds; days beyond 24855 overflow.
                // Matches Spark's cast_spark_df which nulls out-of-range dates.
                if !(-106752..=24855).contains(&days) {
                    return Err(anyhow!("date out of range: day {days}"));
                }
                // Convert days since epoch to seconds since epoch
                let secs = days as i64 * 86400;
                Ok(Some(secs))
            }
            _ => Err(anyhow!("timestamp type mismatch")),
        }
    }

    pub fn get_decimal128(&self, idx: usize) -> Result<Option<i128>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        let scale = self.hint(idx)?.decimal_scale.unwrap_or(0);
        match array.data_type() {
            DataType::Decimal128(_, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .ok_or_else(|| anyhow!("decimal128 cast"))?;
                Ok(Some(arr.value(self.row)))
            }
            // Integer types → Decimal128: multiply by 10^scale
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                let multiplier = 10_i128.pow(scale as u32);
                Ok(Some(i128::from(arr.value(self.row)) * multiplier))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                let multiplier = 10_i128.pow(scale as u32);
                Ok(Some(i128::from(arr.value(self.row)) * multiplier))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                let multiplier = 10_i128.pow(scale as u32);
                Ok(Some(i128::from(arr.value(self.row)) * multiplier))
            }
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                let multiplier = 10_i128.pow(scale as u32);
                Ok(Some(i128::from(arr.value(self.row)) * multiplier))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                let multiplier = 10_i128.pow(scale as u32);
                Ok(Some(i128::from(arr.value(self.row)) * multiplier))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                let multiplier = 10_i128.pow(scale as u32);
                Ok(Some(i128::from(arr.value(self.row)) * multiplier))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                let multiplier = 10_i128.pow(scale as u32);
                Ok(Some(i128::from(arr.value(self.row)) * multiplier))
            }
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                let multiplier = 10_i128.pow(scale as u32);
                Ok(Some(i128::from(arr.value(self.row)) * multiplier))
            }
            // Float types → Decimal128: multiply by 10^scale and round
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                let multiplier = 10_f64.powi(scale);
                let scaled = (f64::from(arr.value(self.row)) * multiplier).round();
                Ok(Some(scaled as i128))
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                let multiplier = 10_f64.powi(scale);
                let scaled = (arr.value(self.row) * multiplier).round();
                Ok(Some(scaled as i128))
            }
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_) => {
                let text = match array.data_type() {
                    DataType::Utf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?
                        .value(self.row),
                    DataType::LargeUtf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .ok_or_else(|| anyhow!("large string cast"))?
                        .value(self.row),
                    DataType::Binary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .ok_or_else(|| anyhow!("binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::LargeBinary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::LargeBinaryArray>()
                            .ok_or_else(|| anyhow!("large binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::FixedSizeBinary(_) => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                            .ok_or_else(|| anyhow!("fixed binary cast"))?
                            .value(self.row),
                    )?,
                    _ => return Err(anyhow!("decimal128 type mismatch")),
                };
                parse_decimal_i128(text, scale)
                    .ok_or_else(|| anyhow!("decimal128 parse failed"))
                    .map(Some)
            }
            _ => Err(anyhow!("decimal128 type mismatch")),
        }
    }

    pub fn get_decimal256(&self, idx: usize) -> Result<Option<[u8; 32]>> {
        let array = self.array(idx)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        let scale = self.hint(idx)?.decimal_scale.unwrap_or(0);
        match array.data_type() {
            DataType::Decimal256(_, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal256Array>()
                    .ok_or_else(|| anyhow!("decimal256 cast"))?;
                Ok(Some(arr.value(self.row).to_le_bytes()))
            }
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_) => {
                let text = match array.data_type() {
                    DataType::Utf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| anyhow!("string cast"))?
                        .value(self.row),
                    DataType::LargeUtf8 => array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .ok_or_else(|| anyhow!("large string cast"))?
                        .value(self.row),
                    DataType::Binary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .ok_or_else(|| anyhow!("binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::LargeBinary => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::LargeBinaryArray>()
                            .ok_or_else(|| anyhow!("large binary cast"))?
                            .value(self.row),
                    )?,
                    DataType::FixedSizeBinary(_) => std::str::from_utf8(
                        array
                            .as_any()
                            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                            .ok_or_else(|| anyhow!("fixed binary cast"))?
                            .value(self.row),
                    )?,
                    _ => return Err(anyhow!("decimal256 type mismatch")),
                };
                parse_decimal_i256(text, scale)
                    .map(|v| v.to_le_bytes())
                    .ok_or_else(|| anyhow!("decimal256 parse failed"))
                    .map(Some)
            }
            // Integer types → Decimal256: multiply by 10^scale
            DataType::Int8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int8Array>()
                    .ok_or_else(|| anyhow!("int8 cast"))?;
                let multiplier = 10_i64.pow(scale as u32);
                let value = i64::from(arr.value(self.row)) * multiplier;
                Ok(Some(arrow_buffer::i256::from(value).to_le_bytes()))
            }
            DataType::Int16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int16Array>()
                    .ok_or_else(|| anyhow!("int16 cast"))?;
                let multiplier = 10_i64.pow(scale as u32);
                let value = i64::from(arr.value(self.row)) * multiplier;
                Ok(Some(arrow_buffer::i256::from(value).to_le_bytes()))
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .ok_or_else(|| anyhow!("int32 cast"))?;
                let multiplier = 10_i64.pow(scale as u32);
                let value = i64::from(arr.value(self.row)) * multiplier;
                Ok(Some(arrow_buffer::i256::from(value).to_le_bytes()))
            }
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 cast"))?;
                let multiplier = 10_i64.pow(scale as u32);
                let value = arr.value(self.row).saturating_mul(multiplier);
                Ok(Some(arrow_buffer::i256::from(value).to_le_bytes()))
            }
            DataType::UInt8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt8Array>()
                    .ok_or_else(|| anyhow!("uint8 cast"))?;
                let multiplier = 10_i64.pow(scale as u32);
                let value = i64::from(arr.value(self.row)) * multiplier;
                Ok(Some(arrow_buffer::i256::from(value).to_le_bytes()))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt16Array>()
                    .ok_or_else(|| anyhow!("uint16 cast"))?;
                let multiplier = 10_i64.pow(scale as u32);
                let value = i64::from(arr.value(self.row)) * multiplier;
                Ok(Some(arrow_buffer::i256::from(value).to_le_bytes()))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| anyhow!("uint32 cast"))?;
                let multiplier = 10_i64.pow(scale as u32);
                let value = i64::from(arr.value(self.row)) * multiplier;
                Ok(Some(arrow_buffer::i256::from(value).to_le_bytes()))
            }
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| anyhow!("uint64 cast"))?;
                // Note: large UInt64 values may overflow i64, use saturating conversion
                let multiplier = 10_u64.pow(scale as u32);
                let value = arr.value(self.row).saturating_mul(multiplier) as i64;
                Ok(Some(arrow_buffer::i256::from(value).to_le_bytes()))
            }
            // Float types → Decimal256: multiply by 10^scale and round
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 cast"))?;
                let multiplier = 10_f64.powi(scale);
                let scaled = (f64::from(arr.value(self.row)) * multiplier).round() as i64;
                Ok(Some(arrow_buffer::i256::from(scaled).to_le_bytes()))
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 cast"))?;
                let multiplier = 10_f64.powi(scale);
                let scaled = (arr.value(self.row) * multiplier).round() as i64;
                Ok(Some(arrow_buffer::i256::from(scaled).to_le_bytes()))
            }
            _ => Err(anyhow!("decimal256 type mismatch")),
        }
    }

    pub fn with_float32_array<T>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&[f32]) -> Result<T>,
    ) -> Result<Option<T>> {
        let array = self.array(idx)?.clone();
        if array.is_null(self.row) {
            return Ok(None);
        }
        // String column → parse "[1.0, 2.0, ...]" into Vec<f32>
        if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
            let parsed = parse_string_as_float32_array(str_arr.value(self.row))?;
            return Ok(Some(f(&parsed)?));
        }
        if let Some(str_arr) = array
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
        {
            let parsed = parse_string_as_float32_array(str_arr.value(self.row))?;
            return Ok(Some(f(&parsed)?));
        }
        let (start, end, values) = if let Some(list) =
            array.as_any().downcast_ref::<arrow_array::ListArray>()
        {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else if let Some(list) = array.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else {
            return Err(anyhow!("list cast"));
        };
        match values.data_type() {
            DataType::Float32 => {
                let values_arr = values
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| anyhow!("float32 list cast"))?;
                if values_arr.null_count() == 0 {
                    return Ok(Some(f(&values_arr.values()[start..end])?));
                }
                let mut vec = Vec::with_capacity(end.saturating_sub(start));
                for idx in start..end {
                    if values_arr.is_null(idx) {
                        vec.push(0.0);
                    } else {
                        vec.push(values_arr.value(idx));
                    }
                }
                Ok(Some(f(&vec)?))
            }
            DataType::Float64 => {
                let values_arr = values
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| anyhow!("float64 list cast"))?;
                let mut vec = Vec::with_capacity(end.saturating_sub(start));
                for idx in start..end {
                    if values_arr.is_null(idx) {
                        vec.push(0.0);
                    } else {
                        vec.push(values_arr.value(idx) as f32);
                    }
                }
                Ok(Some(f(&vec)?))
            }
            DataType::Float16 => {
                let values_arr = values
                    .as_any()
                    .downcast_ref::<arrow_array::Float16Array>()
                    .ok_or_else(|| anyhow!("float16 list cast"))?;
                let mut vec = Vec::with_capacity(end.saturating_sub(start));
                for idx in start..end {
                    if values_arr.is_null(idx) {
                        vec.push(0.0);
                    } else {
                        vec.push(values_arr.value(idx).to_f32());
                    }
                }
                Ok(Some(f(&vec)?))
            }
            _ => Err(anyhow!("float32 array type mismatch")),
        }
    }

    pub fn float_list_view(&mut self, idx: usize) -> Result<Option<FloatListView>> {
        let array = self.array(idx)?.clone();
        if array.is_null(self.row) {
            return Ok(None);
        }
        // String column → parse "[1.0, 2.0, ...]" into Parsed(Vec<f32>)
        if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
            let parsed = parse_string_as_float32_array(str_arr.value(self.row))?;
            let len = parsed.len();
            return Ok(Some(FloatListView {
                start: 0,
                end: len,
                values: FloatListValues::Parsed(parsed),
            }));
        }
        if let Some(str_arr) = array
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
        {
            let parsed = parse_string_as_float32_array(str_arr.value(self.row))?;
            let len = parsed.len();
            return Ok(Some(FloatListView {
                start: 0,
                end: len,
                values: FloatListValues::Parsed(parsed),
            }));
        }
        let (start, end, values) = if let Some(list) =
            array.as_any().downcast_ref::<arrow_array::ListArray>()
        {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else if let Some(list) = array.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else {
            return Err(anyhow!("list cast"));
        };
        let values = match values.data_type() {
            DataType::Float16 => FloatListValues::F16(values.clone()),
            DataType::Float32 => FloatListValues::F32(values.clone()),
            DataType::Float64 => FloatListValues::F64(values.clone()),
            _ => return Err(anyhow!("float array type mismatch")),
        };
        Ok(Some(FloatListView { start, end, values }))
    }

    pub fn int64_list_view(&mut self, idx: usize) -> Result<Option<Int64ListView>> {
        let array = self.array(idx)?.clone();
        if array.is_null(self.row) {
            return Ok(None);
        }
        // Snapshot the multicat-delim hint and column name up-front so we can
        // use them inside the scalar-string branch without holding a borrow on
        // `self.hints`.
        let multicat_delim = self.hint(idx)?.multicategorical_delimiter.clone();
        let col_name = self.hint(idx)?.column_name.clone();
        let row = self.row;
        let col_label = col_name.as_deref().unwrap_or("?");
        if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
            let raw = str_arr.value(self.row);
            let parsed = match multicat_delim.as_deref() {
                Some(delim) => split_multicat_string_as_int64(
                    raw,
                    delim,
                    col_name.as_deref(),
                    row,
                )?,
                None => parse_string_as_int64_array(raw).map_err(|e| {
                    anyhow!(
                        "ingest type mismatch: column={} row={} declared=Int64Array \
                         but parquet scalar string {:?} could not be parsed as a \
                         JSON-style int list ({}). If the source uses a custom delimiter, \
                         declare `multicategorical_delimiter` on the column and \
                         `semantic_type=multicategorical`.",
                        col_label, row, raw, e,
                    )
                })?,
            };
            let end = parsed.len();
            return Ok(Some(Int64ListView {
                start: 0,
                end,
                values: Int64ListValues::Parsed(parsed),
            }));
        }
        if let Some(str_arr) = array
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
        {
            let raw = str_arr.value(self.row);
            let parsed = match multicat_delim.as_deref() {
                Some(delim) => split_multicat_string_as_int64(
                    raw,
                    delim,
                    col_name.as_deref(),
                    row,
                )?,
                None => parse_string_as_int64_array(raw).map_err(|e| {
                    anyhow!(
                        "ingest type mismatch: column={} row={} declared=Int64Array \
                         but parquet scalar string {:?} could not be parsed as a \
                         JSON-style int list ({})",
                        col_label, row, raw, e,
                    )
                })?,
            };
            let end = parsed.len();
            return Ok(Some(Int64ListView {
                start: 0,
                end,
                values: Int64ListValues::Parsed(parsed),
            }));
        }
        let (start, end, values) = if let Some(list) =
            array.as_any().downcast_ref::<arrow_array::ListArray>()
        {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else if let Some(list) = array.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else {
            return Err(anyhow!(
                "ingest type mismatch: column={} row={} declared=Int64Array \
                 but parquet column is neither a list nor a string scalar \
                 (got {:?}). Either change the declared dtype to match the \
                 source, or convert the source column to List<Int64>.",
                col_label,
                row,
                array.data_type(),
            ));
        };
        if values
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .is_none()
        {
            return Err(anyhow!(
                "ingest type mismatch: column={} row={} declared=Int64Array \
                 but parquet list element type is {:?} (need Int64). \
                 If the elements are stringified ints, change the declared \
                 dtype to StringArray + add a multicategorical_delimiter, or \
                 cast the source list elements to Int64 upstream.",
                col_label,
                row,
                values.data_type(),
            ));
        }
        Ok(Some(Int64ListView {
            start,
            end,
            values: Int64ListValues::Native(values.clone()),
        }))
    }

    pub fn string_list_view(&mut self, idx: usize) -> Result<Option<StringListView>> {
        // Fast path: pre-tokenized batch list built in `set_cols` for
        // (multicat-delim, Utf8 source) columns. Read the row's token
        // range directly out of the cached `ListArray<Utf8>`; no per-row
        // splitting or allocation. kumo-ai/kumo#28478.
        if let Some(list_ref) = self.tokenized_lists.get(idx).and_then(|s| s.clone()) {
            let list = list_ref
                .as_any()
                .downcast_ref::<arrow_array::ListArray>()
                .ok_or_else(|| anyhow!("tokenized list cache: not a ListArray"))?;
            if list.is_null(self.row) {
                return Ok(None);
            }
            let offsets = list.value_offsets();
            let start = offsets[self.row] as usize;
            let end = offsets[self.row + 1] as usize;
            return Ok(Some(StringListView {
                start,
                end,
                values: StringListValues::Utf8(list.values().clone()),
            }));
        }
        let array = self.array(idx)?.clone();
        if array.is_null(self.row) {
            return Ok(None);
        }
        let multicat_delim = self.hint(idx)?.multicategorical_delimiter.clone();
        let col_name = self.hint(idx)?.column_name.clone();
        let col_label = col_name.as_deref().unwrap_or("?");
        let row = self.row;
        if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
            let parsed = match multicat_delim.as_deref() {
                Some(delim) => split_multicat_string(str_arr.value(self.row), delim),
                None => parse_string_as_string_array(str_arr.value(self.row))?,
            };
            let end = parsed.len();
            return Ok(Some(StringListView {
                start: 0,
                end,
                values: StringListValues::Parsed(parsed),
            }));
        }
        if let Some(str_arr) = array
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
        {
            let parsed = match multicat_delim.as_deref() {
                Some(delim) => split_multicat_string(str_arr.value(self.row), delim),
                None => parse_string_as_string_array(str_arr.value(self.row))?,
            };
            let end = parsed.len();
            return Ok(Some(StringListView {
                start: 0,
                end,
                values: StringListValues::Parsed(parsed),
            }));
        }
        let (start, end, values) = if let Some(list) =
            array.as_any().downcast_ref::<arrow_array::ListArray>()
        {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else if let Some(list) = array.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else {
            return Err(anyhow!(
                "ingest type mismatch: column={} row={} declared=StringArray \
                 but parquet column is neither a list nor a string scalar \
                 (got {:?}). Either change the declared dtype to match the \
                 source, or convert the source column to List<Utf8>.",
                col_label,
                row,
                array.data_type(),
            ));
        };
        if values
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .is_some()
        {
            return Ok(Some(StringListView {
                start,
                end,
                values: StringListValues::Utf8(values.clone()),
            }));
        }
        if values
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
            .is_some()
        {
            return Ok(Some(StringListView {
                start,
                end,
                values: StringListValues::LargeUtf8(values.clone()),
            }));
        }
        Err(anyhow!(
            "ingest type mismatch: column={} row={} declared=StringArray \
             but parquet list element type is {:?} (need Utf8 or LargeUtf8). \
             Cast the source list elements to a string type upstream.",
            col_label,
            row,
            values.data_type(),
        ))
    }

    pub fn with_float64_array<T>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&[f64]) -> Result<T>,
    ) -> Result<Option<T>> {
        let array = self.array(idx)?.clone();
        if array.is_null(self.row) {
            return Ok(None);
        }
        // String column → parse "[1.0, 2.0, ...]" into Vec<f64>
        if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
            let parsed = parse_string_as_float32_array(str_arr.value(self.row))?;
            let f64_vec: Vec<f64> = parsed.iter().map(|&v| v as f64).collect();
            return Ok(Some(f(&f64_vec)?));
        }
        if let Some(str_arr) = array
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
        {
            let parsed = parse_string_as_float32_array(str_arr.value(self.row))?;
            let f64_vec: Vec<f64> = parsed.iter().map(|&v| v as f64).collect();
            return Ok(Some(f(&f64_vec)?));
        }
        let (start, end, values) = if let Some(list) =
            array.as_any().downcast_ref::<arrow_array::ListArray>()
        {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else if let Some(list) = array.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let offsets = list.value_offsets();
            (
                offsets[self.row] as usize,
                offsets[self.row + 1] as usize,
                list.values(),
            )
        } else {
            return Err(anyhow!("list cast"));
        };
        let values_arr = values
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .ok_or_else(|| anyhow!("float64 list cast"))?;
        if values_arr.null_count() == 0 {
            return Ok(Some(f(&values_arr.values()[start..end])?));
        }
        let mut vec = Vec::with_capacity(end.saturating_sub(start));
        for idx in start..end {
            if values_arr.is_null(idx) {
                vec.push(0.0);
            } else {
                vec.push(values_arr.value(idx));
            }
        }
        Ok(Some(f(&vec)?))
    }

    /// Read a `List<Int64>` or `LargeList<Int64>` column for the current row.
    ///
    /// Returns `None` if the row-level value is null, `Some(f(slice))` otherwise.
    /// Null elements inside the list are dropped from the slice handed to `f`;
    /// the slice length reflects only non-null elements. A legitimate zero is
    /// preserved — only the validity bit filters.
    pub fn with_int64_array<T>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&[i64]) -> Result<T>,
    ) -> Result<Option<T>> {
        let Some(view) = self.int64_list_view(idx)? else {
            return Ok(None);
        };
        match &view.values {
            Int64ListValues::Native(values) => {
                let values_arr = values
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .ok_or_else(|| anyhow!("int64 list element cast"))?;
                if values_arr.null_count() == 0 {
                    return Ok(Some(f(&values_arr.values()[view.start..view.end])?));
                }
                let mut vec = Vec::with_capacity(view.len());
                view.for_each_i64(|v| vec.push(v));
                Ok(Some(f(&vec)?))
            }
            Int64ListValues::Parsed(values) => Ok(Some(f(&values[view.start..view.end])?)),
        }
    }

    /// Read a `List<Utf8>` or `LargeList<LargeUtf8>` column for the current row.
    ///
    /// Returns `None` if the row-level value is null.
    /// Calls `f` with a slice of `&str` tokens. Null elements inside the list
    /// are dropped from the slice handed to `f`; the slice length reflects
    /// only non-null elements. A legitimate empty string is preserved —
    /// only the validity bit filters.
    pub fn with_string_array<T>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&[&str]) -> Result<T>,
    ) -> Result<Option<T>> {
        let Some(view) = self.string_list_view(idx)? else {
            return Ok(None);
        };
        match &view.values {
            StringListValues::Utf8(values) => {
                let values_arr = values
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| anyhow!("string list element cast"))?;
                let mut refs: Vec<&str> = Vec::with_capacity(view.len());
                for i in view.start..view.end {
                    if !values_arr.is_null(i) {
                        refs.push(values_arr.value(i));
                    }
                }
                Ok(Some(f(&refs)?))
            }
            StringListValues::LargeUtf8(values) => {
                let values_arr = values
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .ok_or_else(|| anyhow!("string list element cast"))?;
                let mut refs: Vec<&str> = Vec::with_capacity(view.len());
                for i in view.start..view.end {
                    if !values_arr.is_null(i) {
                        refs.push(values_arr.value(i));
                    }
                }
                Ok(Some(f(&refs)?))
            }
            StringListValues::Parsed(values) => {
                let refs: Vec<&str> = values[view.start..view.end]
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                Ok(Some(f(&refs)?))
            }
        }
    }
}
