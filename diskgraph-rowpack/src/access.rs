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
}

pub struct RowAccess<'p> {
    row: usize,
    cols_ptr: *const ArrayRef,
    cols_len: usize,
    _stats: RowAccessStats<'p>,
    hints: Vec<RowAccessHints>,
    fast_parsers: Vec<Option<TimestampFormatParser>>,
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
}

/// Parse a JSON-style string like `"[1.0, 2.0, 3.0]"` or `"1.0, 2.0, 3.0"`
/// into a `Vec<f32>`. Uses `fast_float` for ~5x faster parsing than stdlib.
///
/// Returns `Ok(vec![])` for empty/whitespace-only strings or `"[]"`.
/// Returns `Err` for malformed strings (hard error).
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
        Self {
            row: 0,
            cols_ptr: std::ptr::null(),
            cols_len: 0,
            _stats: stats,
            hints,
            fast_parsers,
        }
    }

    pub fn set_cols(&mut self, cols: &[ArrayRef]) {
        self.cols_ptr = cols.as_ptr();
        self.cols_len = cols.len();
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
                if days < -106752 || days > 24855 {
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
        if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
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
        if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
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
        if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
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
}
