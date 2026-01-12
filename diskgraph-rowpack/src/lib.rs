use anyhow::{anyhow, Result};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use base64::Engine;
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use gxhash::{HashMap, HashMapExt};
use half::f16;
use num_bigint::BigInt;
use num_traits::Signed;
use rmp::encode as mp;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use uuid_simd::{format_simple, parse_hyphenated, parse_simple, AsOut, AsciiCase};

const ROWPACK_VERSION: u8 = 3;
const MAX_FIELD_BYTES: usize = 64 * 1024 * 1024;
const MAX_ARRAY_LEN: usize = 1_000_000;
const ALPHABET_MAX_CHARS: usize = 64;
const ALPHABET_MIN_COVERAGE: f32 = 0.95;
const ALPHABET_INVALID: u8 = u8::MAX;
type FastHashMap<K, V> = HashMap<K, V>;

#[derive(Clone)]
struct AlphabetMap {
    map: [u8; 256],
    bits: u8,
    packer: AlphabetPacker,
}

#[derive(Clone, Copy)]
enum AlphabetPacker {
    Bits1,
    Bits2,
    Bits3,
    Bits4,
    Bits5,
    Bits6,
    Generic { bits: u8 },
}

struct BpeTrie {
    next: Vec<i32>,
    value: Vec<Option<u16>>,
    dict_len: usize,
}

impl BpeTrie {
    fn from_dict(dict: &[Vec<u8>]) -> Self {
        let mut next = vec![-1; 256];
        let mut value = vec![None];
        for (idx, entry) in dict.iter().enumerate() {
            let mut node = 0usize;
            for &b in entry {
                let slot = node * 256 + b as usize;
                let next_idx = next[slot];
                if next_idx < 0 {
                    next.extend(std::iter::repeat(-1).take(256));
                    value.push(None);
                    let new_idx = (value.len() - 1) as i32;
                    next[slot] = new_idx;
                    node = new_idx as usize;
                } else {
                    node = next_idx as usize;
                }
            }
            value[node] = Some(256u16 + idx as u16);
        }
        Self {
            next,
            value,
            dict_len: dict.len(),
        }
    }

    #[inline]
    fn next_index(&self, node: usize, byte: u8) -> i32 {
        self.next[node * 256 + byte as usize]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowPackConfig {
    #[serde(default)]
    pub sample_rows: Option<usize>,
    #[serde(default)]
    pub max_dict_bytes_per_col: Option<usize>,
    #[serde(default)]
    pub max_dict_entries: Option<usize>,
    #[serde(default)]
    pub bpe_max_dict_bytes: Option<usize>,
    #[serde(default)]
    pub bpe_max_dict_entries: Option<usize>,
    #[serde(default)]
    pub bpe_max_samples: Option<usize>,
    #[serde(default)]
    pub bpe_min_gain_ratio: Option<f32>,
    #[serde(default)]
    pub bpe_min_avg_len: Option<usize>,
    #[serde(default)]
    pub string_dict_coverage: Option<f32>,
    #[serde(default)]
    pub string_long_min_len: Option<usize>,
    #[serde(default)]
    pub string_long_min_ratio: Option<f32>,
}

impl Default for RowPackConfig {
    fn default() -> Self {
        Self {
            sample_rows: Some(10_000),
            max_dict_bytes_per_col: Some(1_000_000),
            max_dict_entries: Some(50_000),
            bpe_max_dict_bytes: Some(1_000_000),
            bpe_max_dict_entries: Some(256),
            bpe_max_samples: Some(5_000),
            bpe_min_gain_ratio: Some(0.95),
            bpe_min_avg_len: Some(12),
            string_dict_coverage: Some(0.7),
            string_long_min_len: Some(128),
            string_long_min_ratio: Some(0.85),
        }
    }
}

impl RowPackConfig {
    pub fn merge(outer: Option<&RowPackConfig>, inner: Option<&RowPackConfig>) -> RowPackConfig {
        let base = outer.cloned().unwrap_or_default();
        if let Some(inner) = inner {
            RowPackConfig {
                sample_rows: inner.sample_rows.or(base.sample_rows),
                max_dict_bytes_per_col: inner
                    .max_dict_bytes_per_col
                    .or(base.max_dict_bytes_per_col),
                max_dict_entries: inner.max_dict_entries.or(base.max_dict_entries),
                bpe_max_dict_bytes: inner.bpe_max_dict_bytes.or(base.bpe_max_dict_bytes),
                bpe_max_dict_entries: inner.bpe_max_dict_entries.or(base.bpe_max_dict_entries),
                bpe_max_samples: inner.bpe_max_samples.or(base.bpe_max_samples),
                bpe_min_gain_ratio: inner.bpe_min_gain_ratio.or(base.bpe_min_gain_ratio),
                bpe_min_avg_len: inner.bpe_min_avg_len.or(base.bpe_min_avg_len),
                string_dict_coverage: inner.string_dict_coverage.or(base.string_dict_coverage),
                string_long_min_len: inner.string_long_min_len.or(base.string_long_min_len),
                string_long_min_ratio: inner.string_long_min_ratio.or(base.string_long_min_ratio),
            }
        } else {
            base
        }
    }

    pub fn sample_rows(&self) -> usize {
        self.sample_rows.unwrap_or(10_000)
    }

    fn max_dict_bytes(&self) -> usize {
        self.max_dict_bytes_per_col.unwrap_or(1_000_000)
    }

    fn max_dict_entries(&self) -> usize {
        self.max_dict_entries.unwrap_or(50_000)
    }

    fn bpe_max_dict_bytes(&self) -> usize {
        self.bpe_max_dict_bytes.unwrap_or(256_000)
    }

    fn bpe_max_dict_entries(&self) -> usize {
        self.bpe_max_dict_entries.unwrap_or(512)
    }

    fn bpe_max_samples(&self) -> usize {
        self.bpe_max_samples.unwrap_or(2_000)
    }

    fn bpe_min_gain_ratio(&self) -> f32 {
        self.bpe_min_gain_ratio.unwrap_or(0.95)
    }

    fn bpe_min_avg_len(&self) -> usize {
        self.bpe_min_avg_len.unwrap_or(12)
    }

    fn string_dict_coverage(&self) -> f32 {
        self.string_dict_coverage.unwrap_or(0.7)
    }

    fn string_long_min_len(&self) -> usize {
        self.string_long_min_len.unwrap_or(128)
    }

    fn string_long_min_ratio(&self) -> f32 {
        self.string_long_min_ratio.unwrap_or(0.85)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IngestedType {
    Int8,
    Int16,
    Int64,
    Int32,
    UInt8,
    UInt16,
    UInt64,
    UInt32,
    Float64,
    Float32,
    Bool,
    String,
    Bytes,
    Uuid,
    Uuid256,
    Ip,
    Date,
    Timestamp,
    Decimal128,
    Decimal256,
    Float16Array,
    Float32Array,
    Float64Array,
}

#[derive(Debug, Clone)]
pub struct RowPackColumn {
    pub name: String,
    pub logical_type: IngestedType,
    pub fixed_len: Option<usize>,
    pub precision: Option<u32>,
    pub scale: Option<i32>,
}

const MAX_INPUT_STRING_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct RowAccessHints {
    pub timestamp_format: Option<String>,
    pub timestamp_unit: Option<arrow_schema::TimeUnit>,
    pub decimal_scale: Option<i32>,
}

pub struct RowAccess<'p> {
    row: usize,
    cols_ptr: *const ArrayRef,
    cols_len: usize,
    _stats: RowAccessStats<'p>,
    hints: Vec<RowAccessHints>,
    fast_parsers: Vec<Option<TimestampFormatParser>>,
}

impl<'p> RowAccess<'p> {
    pub fn new(
        hints: Vec<RowAccessHints>,
        stats: RowAccessStats<'p>,
    ) -> Self {
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
                let (secs, _) = timestamp_parts_from_array(array, self.row)?;
                return Ok(Some(with_stack_str(secs, trimmed, f)?));
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
                    let (secs, _) = timestamp_from_int(value, unit);
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
                let (secs, _) = timestamp_from_int(value, unit);
                Ok(Some(days_from_secs(secs)))
            }
            _ => Err(anyhow!("date type mismatch")),
        }
    }

    pub fn get_timestamp(&self, idx: usize) -> Result<Option<(i64, u32)>> {
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
                let (secs, nanos) = timestamp_parts_from_array(array, self.row)?;
                Ok(Some((secs, nanos)))
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
                    if let Some((secs, nanos)) = parser.parse_timestamp(text) {
                        return Ok(Some((secs, nanos)));
                    }
                }
                if let Some((secs, nanos)) = parse_timestamp_text(text, format) {
                    return Ok(Some((secs, nanos)));
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
                Ok(Some(timestamp_from_int(value, unit)))
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

    pub fn with_float64_array<T>(
        &mut self,
        idx: usize,
        f: impl FnOnce(&[f64]) -> Result<T>,
    ) -> Result<Option<T>> {
        let array = self.array(idx)?.clone();
        if array.is_null(self.row) {
            return Ok(None);
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

fn parse_bool(value: &str) -> Option<bool> {
    let mut text = value.trim();
    if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        text = &text[1..text.len() - 1];
    }
    let text = text.trim();
    let lower = text.to_ascii_lowercase();
    match lower.as_str() {
        "" | "false" | "f" | "no" | "n" | "off" | "disabled" | "disable" | "unchecked" => {
            return Some(false);
        }
        "true" | "t" | "yes" | "y" | "on" | "enabled" | "enable" | "checked" | "x" => {
            return Some(true);
        }
        _ => {}
    }
    if let Ok(int_value) = lower.parse::<i64>() {
        return Some(int_value != 0);
    }
    if let Ok(float_value) = lower.parse::<f64>() {
        return Some(!float_value.is_nan() && float_value != 0.0);
    }
    None
}

fn parse_i64(value: &str) -> Option<i64> {
    value.trim().parse::<i64>().ok()
}

fn parse_i32(value: &str) -> Option<i32> {
    value.trim().parse::<i32>().ok()
}

fn parse_u64(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

fn parse_u32(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok()
}

fn parse_f64(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok()
}

fn parse_f32(value: &str) -> Option<f32> {
    value.trim().parse::<f32>().ok()
}

fn trim_str_bytes(value: &str, max: usize) -> &str {
    if value.len() <= max {
        return value;
    }
    let mut end = max;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn trim_borrowed_str<'a>(
    value: &'a str,
    max: usize,
    counter: Option<&std::sync::atomic::AtomicU64>,
) -> &'a str {
    if value.len() <= max {
        return value;
    }
    if let Some(counter) = counter {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    trim_str_bytes(value, max)
}

fn trim_borrowed_bytes<'a>(
    value: &'a [u8],
    max: usize,
    counter: Option<&std::sync::atomic::AtomicU64>,
) -> &'a [u8] {
    if value.len() <= max {
        return value;
    }
    if let Some(counter) = counter {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    &value[..max]
}

struct StackStr {
    buf: [u8; 64],
    len: usize,
}

impl StackStr {
    fn new() -> Self {
        Self {
            buf: [0u8; 64],
            len: 0,
        }
    }

    fn as_str(&self) -> Result<&str> {
        std::str::from_utf8(&self.buf[..self.len]).map_err(|_| anyhow!("utf8 format failed"))
    }
}

impl std::fmt::Write for StackStr {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let bytes = s.as_bytes();
        if self.len + bytes.len() > self.buf.len() {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }
}

fn with_stack_str<Out, V: std::fmt::Display>(
    value: V,
    counter: Option<&std::sync::atomic::AtomicU64>,
    f: impl FnOnce(&str) -> Result<Out>,
) -> Result<Out> {
    use std::fmt::Write as _;
    let mut buf = StackStr::new();
    write!(&mut buf, "{}", value).map_err(|_| anyhow!("string format overflow"))?;
    let s = buf.as_str()?;
    let s = trim_borrowed_str(s, MAX_INPUT_STRING_BYTES, counter);
    f(s)
}

fn parse_uuid_string(bytes: &[u8]) -> Option<[u8; 16]> {
    let mut out = [0u8; 16];
    if bytes.len() == 32 {
        parse_simple(bytes, out.as_out()).ok()?;
        return Some(out);
    }
    if bytes.len() == 36 {
        parse_hyphenated(bytes, out.as_out()).ok()?;
        return Some(out);
    }
    None
}

fn parse_uuid256_string(bytes: &[u8]) -> Option<[u8; 32]> {
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    // Safety: `out` is 32 bytes, so the two 16-byte regions are in-bounds and non-overlapping.
    let out_ptr = out.as_mut_ptr();
    let hi = unsafe { &mut *(out_ptr as *mut [u8; 16]) };
    let lo = unsafe { &mut *(out_ptr.add(16) as *mut [u8; 16]) };
    parse_simple(&bytes[..32], hi.as_out()).ok()?;
    parse_simple(&bytes[32..], lo.as_out()).ok()?;
    Some(out)
}

fn parse_ip_string(bytes: &[u8]) -> Option<[u8; 16]> {
    let s = std::str::from_utf8(bytes).ok()?;
    let ip: std::net::IpAddr = s.parse().ok()?;
    Some(ip_to_ipv6_bytes(ip))
}

fn parse_ip_bytes(bytes: &[u8]) -> Option<[u8; 16]> {
    match bytes.len() {
        4 => Some(ip_to_ipv6_bytes(std::net::IpAddr::V4(
            std::net::Ipv4Addr::new(bytes[0], bytes[1], bytes[2], bytes[3]),
        ))),
        16 => {
            let mut out = [0u8; 16];
            out.copy_from_slice(bytes);
            Some(out)
        }
        _ => None,
    }
}

fn ip_to_ipv6_bytes(ip: std::net::IpAddr) -> [u8; 16] {
    match ip {
        std::net::IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
        std::net::IpAddr::V6(v6) => v6.octets(),
    }
}

fn parse_timestamp_text(value: &str, format: Option<&str>) -> Option<(i64, u32)> {
    let ts = if let Some(fmt) = format {
        let has_tz = fmt.contains("%z") || fmt.contains("%:z");
        if has_tz {
            chrono::DateTime::parse_from_str(value, fmt)
                .ok()
                .map(|v| v.naive_utc())?
        } else {
            NaiveDateTime::parse_from_str(value, fmt).ok().or_else(|| {
                NaiveDate::parse_from_str(value, fmt)
                    .ok()
                    .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
            })?
        }
    } else {
        chrono::DateTime::parse_from_rfc3339(value)
            .ok()
            .map(|v| v.naive_utc())?
    };
    let ts = ts.and_utc();
    Some((ts.timestamp(), ts.timestamp_subsec_nanos()))
}

fn parse_date_text(value: &str, format: Option<&str>) -> Option<i32> {
    let dt = if let Some(fmt) = format {
        NaiveDate::parse_from_str(value, fmt).ok().or_else(|| {
            NaiveDateTime::parse_from_str(value, fmt)
                .ok()
                .map(|v| v.date())
        })
    } else {
        NaiveDate::parse_from_str(value, "%Y-%m-%d")
            .ok()
            .or_else(|| {
                NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
                    .ok()
                    .map(|v| v.date())
            })
    }?;
    let ts = dt.and_hms_opt(0, 0, 0)?.and_utc().timestamp();
    Some(days_from_secs(ts))
}

#[derive(Clone, Debug)]
enum TimestampFormatKind {
    Ymd,
    YmdHmSpace,
    YmdHmsSpace,
    YmdHmsSpaceFrac,
    YmdHmsT,
    YmdHmsTFrac,
    YmdHmsSpaceOffset,
    YmdHmsSpaceOffsetColon,
    YmdHmsSpaceOffsetSpaced,
    YmdHmsSpaceOffsetColonSpaced,
    YmdHmsTOffset,
    YmdHmsTOffsetColon,
}

#[derive(Clone, Debug)]
struct TimestampFormatParser {
    kind: TimestampFormatKind,
}

impl TimestampFormatParser {
    fn compile(format: &str) -> Option<Self> {
        let kind = match format {
            "%Y-%m-%d" => TimestampFormatKind::Ymd,
            "%Y-%m-%d %H:%M" => TimestampFormatKind::YmdHmSpace,
            "%Y-%m-%d %H:%M:%S" => TimestampFormatKind::YmdHmsSpace,
            "%Y-%m-%d %H:%M:%S%.f" => TimestampFormatKind::YmdHmsSpaceFrac,
            "%Y-%m-%dT%H:%M:%S" => TimestampFormatKind::YmdHmsT,
            "%Y-%m-%dT%H:%M:%S%.f" => TimestampFormatKind::YmdHmsTFrac,
            "%Y-%m-%d %H:%M:%S%z" => TimestampFormatKind::YmdHmsSpaceOffset,
            "%Y-%m-%d %H:%M:%S %z" => TimestampFormatKind::YmdHmsSpaceOffsetSpaced,
            "%Y-%m-%d %H:%M:%S%:z" => TimestampFormatKind::YmdHmsSpaceOffsetColon,
            "%Y-%m-%d %H:%M:%S %:z" => TimestampFormatKind::YmdHmsSpaceOffsetColonSpaced,
            "%Y-%m-%dT%H:%M:%S%z" => TimestampFormatKind::YmdHmsTOffset,
            "%Y-%m-%dT%H:%M:%S%:z" => TimestampFormatKind::YmdHmsTOffsetColon,
            _ => return None,
        };
        Some(Self { kind })
    }

    fn parse_timestamp(&self, value: &str) -> Option<(i64, u32)> {
        match self.kind {
            TimestampFormatKind::Ymd => {
                return parse_ymd_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmSpace => {
                return parse_ymd_hm_space_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpace => {
                return parse_ymd_hms_space_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceFrac => {
                return parse_ymd_hms_space_frac_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsT => {
                return parse_ymd_hms_space_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsTFrac => {
                return parse_ymd_hms_space_frac_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceOffset => {
                return parse_ymd_hms_offset_nocolon_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceOffsetColon => {
                return parse_ymd_hms_offset_colon_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceOffsetSpaced => {
                return parse_ymd_hms_offset_nocolon_spaced_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceOffsetColonSpaced => {
                return parse_ymd_hms_offset_colon_spaced_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsTOffset => {
                return parse_ymd_hms_offset_nocolon_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsTOffsetColon => {
                return parse_ymd_hms_offset_colon_fast(value.as_bytes());
            }
        }
    }

    fn parse_date(&self, value: &str) -> Option<i32> {
        let (secs, _nanos) = self.parse_timestamp(value)?;
        Some(days_from_secs(secs))
    }
}

fn parse_fraction(bytes: &[u8], pos: usize) -> Option<(u32, usize, usize)> {
    let mut idx = pos;
    let mut val: u32 = 0;
    let mut digits = 0usize;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        if digits < 9 {
            val = val
                .saturating_mul(10)
                .saturating_add((bytes[idx] - b'0') as u32);
        }
        digits += 1;
        idx += 1;
    }
    if digits == 0 {
        return None;
    }
    Some((val, digits.min(9), idx))
}

fn fraction_to_nanos(value: u32, digits: usize) -> u32 {
    const POW10: [u32; 10] = [
        1_000_000_000,
        100_000_000,
        10_000_000,
        1_000_000,
        100_000,
        10_000,
        1_000,
        100,
        10,
        1,
    ];
    if digits == 0 {
        0
    } else if digits >= 9 {
        value
    } else {
        value.saturating_mul(POW10[digits])
    }
}

// "%Y-%m-%d"
fn parse_ymd_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 10 {
        return None;
    }
    let a = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const u64) };
    let mut tail = [0u8; 8];
    tail[..2].copy_from_slice(&bytes[8..10]);
    let b = u64::from_le_bytes(tail);

    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK_A: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00]);
    const MASK_B: u64 = u64::from_le_bytes([0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);

    let x_a = (a & MASK_A) | (!MASK_A & ZERO);
    let x_b = (b & MASK_B) | (!MASK_B & ZERO);
    if !all_digits(x_a) || !all_digits(x_b) {
        return None;
    }

    let da = x_a.wrapping_sub(ZERO);
    let db = x_b.wrapping_sub(ZERO);
    let year =
        byte_u32(da, 0) * 1000 + byte_u32(da, 8) * 100 + byte_u32(da, 16) * 10 + byte_u32(da, 24);
    let month = byte_u32(da, 40) * 10 + byte_u32(da, 48);
    let day = byte_u32(db, 0) * 10 + byte_u32(db, 8);

    let secs = timestamp_from_parts_fast(year as i32, month, day, 0, 0, 0);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M"
fn parse_ymd_hm_space_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 16 {
        return None;
    }
    let a = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const u64) };
    let b = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(8) as *const u64) };

    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK_A: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00]);
    const MASK_B: u64 = u64::from_le_bytes([0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0xFF]);

    let x_a = (a & MASK_A) | (!MASK_A & ZERO);
    let x_b = (b & MASK_B) | (!MASK_B & ZERO);
    if !all_digits(x_a) || !all_digits(x_b) {
        return None;
    }

    let da = x_a.wrapping_sub(ZERO);
    let db = x_b.wrapping_sub(ZERO);
    let year =
        byte_u32(da, 0) * 1000 + byte_u32(da, 8) * 100 + byte_u32(da, 16) * 10 + byte_u32(da, 24);
    let month = byte_u32(da, 40) * 10 + byte_u32(da, 48);
    let day = byte_u32(db, 0) * 10 + byte_u32(db, 8);
    let hour = byte_u32(db, 24) * 10 + byte_u32(db, 32);
    let minute = byte_u32(db, 48) * 10 + byte_u32(db, 56);

    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, 0);
    Some((secs, 0))
}

#[inline]
fn parse_ymd_hms_digits(bytes: &[u8]) -> Option<(u32, u32, u32, u32, u32, u32)> {
    if bytes.len() < 19 {
        return None;
    }
    let a = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const u64) };
    let b = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(8) as *const u64) };
    let mut tail = [0u8; 8];
    tail[..3].copy_from_slice(&bytes[16..19]);
    let c = u64::from_le_bytes(tail);

    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK_A: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00]);
    const MASK_B: u64 = u64::from_le_bytes([0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0xFF]);
    const MASK_C: u64 = u64::from_le_bytes([0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00]);

    let x_a = (a & MASK_A) | (!MASK_A & ZERO);
    let x_b = (b & MASK_B) | (!MASK_B & ZERO);
    let x_c = (c & MASK_C) | (!MASK_C & ZERO);
    if !all_digits(x_a) || !all_digits(x_b) || !all_digits(x_c) {
        return None;
    }

    let da = x_a.wrapping_sub(ZERO);
    let db = x_b.wrapping_sub(ZERO);
    let dc = x_c.wrapping_sub(ZERO);

    let year =
        byte_u32(da, 0) * 1000 + byte_u32(da, 8) * 100 + byte_u32(da, 16) * 10 + byte_u32(da, 24);
    let month = byte_u32(da, 40) * 10 + byte_u32(da, 48);
    let day = byte_u32(db, 0) * 10 + byte_u32(db, 8);
    let hour = byte_u32(db, 24) * 10 + byte_u32(db, 32);
    let minute = byte_u32(db, 48) * 10 + byte_u32(db, 56);
    let second = byte_u32(dc, 8) * 10 + byte_u32(dc, 16);

    Some((year, month, day, hour, minute, second))
}

// "%Y-%m-%d %H:%M:%S" / "%Y-%m-%dT%H:%M:%S"
fn parse_ymd_hms_space_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 19 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M:%S%.f" / "%Y-%m-%dT%H:%M:%S%.f"
fn parse_ymd_hms_space_frac_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() < 21 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let (fraction, digits, next) = parse_fraction(bytes, 20)?;
    if next != bytes.len() {
        return None;
    }
    let nanos = fraction_to_nanos(fraction, digits);
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second);
    Some((secs, nanos))
}

#[inline]
fn parse_offset_hhmm(bytes: &[u8], pos: usize) -> Option<i64> {
    let sign = match *bytes.get(pos)? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let mut tail = [0u8; 8];
    tail[..4].copy_from_slice(bytes.get(pos + 1..pos + 5)?);
    let word = u64::from_le_bytes(tail);
    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]);
    let x = (word & MASK) | (!MASK & ZERO);
    if !all_digits(x) {
        return None;
    }
    let d = x.wrapping_sub(ZERO);
    let hour = byte_u32(d, 0) * 10 + byte_u32(d, 8);
    let minute = byte_u32(d, 16) * 10 + byte_u32(d, 24);
    Some(sign * (hour as i64 * 3_600 + minute as i64 * 60))
}

#[inline]
fn parse_offset_hhmm_colon(bytes: &[u8], pos: usize) -> Option<i64> {
    let sign = match *bytes.get(pos)? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let mut tail = [0u8; 8];
    tail[..5].copy_from_slice(bytes.get(pos + 1..pos + 6)?);
    let word = u64::from_le_bytes(tail);
    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK: u64 = u64::from_le_bytes([0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00]);
    let x = (word & MASK) | (!MASK & ZERO);
    if !all_digits(x) {
        return None;
    }
    let d = x.wrapping_sub(ZERO);
    let hour = byte_u32(d, 0) * 10 + byte_u32(d, 8);
    let minute = byte_u32(d, 24) * 10 + byte_u32(d, 32);
    Some(sign * (hour as i64 * 3_600 + minute as i64 * 60))
}

// "%Y-%m-%d %H:%M:%S%z" / "%Y-%m-%dT%H:%M:%S%z"
fn parse_ymd_hms_offset_nocolon_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 24 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm(bytes, 19)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M:%S%:z" / "%Y-%m-%dT%H:%M:%S%:z"
fn parse_ymd_hms_offset_colon_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 25 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm_colon(bytes, 19)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M:%S %z"
fn parse_ymd_hms_offset_nocolon_spaced_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 25 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm(bytes, 20)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M:%S %:z"
fn parse_ymd_hms_offset_colon_spaced_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 26 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm_colon(bytes, 20)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some((secs, 0))
}

#[inline]
fn byte_u32(value: u64, shift: u32) -> u32 {
    ((value >> shift) & 0xFF) as u32
}

#[inline]
fn all_digits(value: u64) -> bool {
    const ZERO: u64 = 0x3030_3030_3030_3030;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let add = value.wrapping_add(0x4646_4646_4646_4646);
    let sub = value.wrapping_sub(ZERO);
    ((add | sub) & HIGH) == 0
}

fn timestamp_from_parts_fast(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> i64 {
    let days = days_from_civil_fast(year, month, day);
    days * 86_400 + hour as i64 * 3_600 + minute as i64 * 60 + second as i64
}

fn days_from_civil_fast(year: i32, month: u32, day: u32) -> i64 {
    let mut y = year as i64;
    let m = month as i64;
    let d = day as i64;
    y -= (m <= 2) as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = m + 9 - (m > 2) as i64 * 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn timestamp_from_int(value: i64, unit: Option<arrow_schema::TimeUnit>) -> (i64, u32) {
    let unit = unit.unwrap_or(arrow_schema::TimeUnit::Second);
    match unit {
        arrow_schema::TimeUnit::Second => (value, 0),
        arrow_schema::TimeUnit::Millisecond => {
            let secs = value.div_euclid(1_000);
            let millis = value.rem_euclid(1_000) as u32;
            (secs, millis * 1_000_000)
        }
        arrow_schema::TimeUnit::Microsecond => {
            let secs = value.div_euclid(1_000_000);
            let micros = value.rem_euclid(1_000_000) as u32;
            (secs, micros * 1_000)
        }
        arrow_schema::TimeUnit::Nanosecond => {
            let secs = value.div_euclid(1_000_000_000);
            let nanos = value.rem_euclid(1_000_000_000) as u32;
            (secs, nanos)
        }
    }
}

fn days_from_secs(secs: i64) -> i32 {
    secs.div_euclid(86_400) as i32
}

fn parse_decimal_i128(value: &str, scale: i32) -> Option<i128> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let digits = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    let mut parts = digits.split('.');
    let whole = parts.next().unwrap_or("");
    let frac = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return None;
    }
    let scale = scale.max(0) as usize;
    if frac.len() > scale {
        return None;
    }
    let mut buf = String::with_capacity(whole.len() + scale);
    buf.push_str(whole);
    buf.push_str(frac);
    for _ in frac.len()..scale {
        buf.push('0');
    }
    let parsed = buf.parse::<i128>().ok()?;
    if negative {
        Some(-parsed)
    } else {
        Some(parsed)
    }
}

fn parse_decimal_i256(value: &str, scale: i32) -> Option<arrow_buffer::i256> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let digits = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    let mut parts = digits.split('.');
    let whole = parts.next().unwrap_or("");
    let frac = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return None;
    }
    let scale = scale.max(0) as usize;
    if frac.len() > scale {
        return None;
    }
    let mut buf = String::with_capacity(whole.len() + scale);
    buf.push_str(whole);
    buf.push_str(frac);
    for _ in frac.len()..scale {
        buf.push('0');
    }
    let mut parsed = arrow_buffer::i256::from_string(&buf)?;
    if negative {
        parsed = parsed.checked_neg()?;
    }
    Some(parsed)
}

fn timestamp_parts_from_array(array: &ArrayRef, row: usize) -> Result<(i64, u32)> {
    let unit = match array.data_type() {
        DataType::Timestamp(unit, _) => *unit,
        _ => return Err(anyhow!("timestamp type mismatch")),
    };
    match unit {
        arrow_schema::TimeUnit::Second => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampSecondArray>()
                .ok_or_else(|| anyhow!("timestamp second cast"))?;
            Ok((arr.value(row), 0))
        }
        arrow_schema::TimeUnit::Millisecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                .ok_or_else(|| anyhow!("timestamp millisecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000);
            let millis = value.rem_euclid(1_000) as u32;
            Ok((secs, millis * 1_000_000))
        }
        arrow_schema::TimeUnit::Microsecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                .ok_or_else(|| anyhow!("timestamp microsecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000_000);
            let micros = value.rem_euclid(1_000_000) as u32;
            Ok((secs, micros * 1_000))
        }
        arrow_schema::TimeUnit::Nanosecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                .ok_or_else(|| anyhow!("timestamp nanosecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000_000_000);
            let nanos = value.rem_euclid(1_000_000_000) as u32;
            Ok((secs, nanos))
        }
    }
}


pub struct RowAccessStats<'a> {
    pub trimmed_strings: Option<&'a std::sync::atomic::AtomicU64>,
}

impl<'a> RowAccessStats<'a> {
    pub fn new(trimmed_strings: Option<&'a std::sync::atomic::AtomicU64>) -> Self {
        Self { trimmed_strings }
    }
}

impl<'a> Default for RowAccessStats<'a> {
    fn default() -> Self {
        Self {
            trimmed_strings: None,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowPackContext {
    pub version: u8,
    pub columns: Vec<ColumnContext>,
    pub estimated_row_bytes: u32,
}

impl RowPackContext {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let bytes = bincode::serialize(self)?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let ctx: RowPackContext = bincode::deserialize(bytes)?;
        if ctx.version != ROWPACK_VERSION {
            return Err(anyhow!("unsupported rowpack version {}", ctx.version));
        }
        Ok(ctx)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnContext {
    pub name: String,
    pub logical_type: IngestedType,
    pub encoding: ColumnEncoding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnEncoding {
    Int {
        signed: bool,
        varint: bool,
        bits: u8,
    },
    Float {
        storage: FloatStorage,
    },
    Bool,
    String {
        encoding: StringEncoding,
    },
    Bytes {
        fixed_len: Option<usize>,
    },
    Uuid,
    Uuid256,
    Ip,
    Date {
        varint: bool,
        bits: u8,
    },
    Timestamp,
    Decimal {
        precision: u32,
        scale: i32,
        bytes: u8,
        storage: DecimalStorage,
    },
    FloatArray {
        storage: FloatStorage,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecimalStorage {
    Raw,
    Varint,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FloatStorage {
    F16,
    F32,
    F64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StringEncoding {
    Raw,
    Dict { values: Vec<Vec<u8>> },
    Alphabet { alphabet: Vec<u8> },
    Bpe { dict: Vec<Vec<u8>> },
    Lz4,
}

#[derive(Debug, Clone)]
pub enum ValueRef<'a> {
    Null,
    Int64(i64),
    Int32(i32),
    UInt64(u64),
    UInt32(u32),
    Float64(f64),
    Float32(f32),
    Bool(bool),
    String(&'a str),
    Bytes(&'a [u8]),
    Uuid([u8; 16]),
    Uuid256([u8; 32]),
    Ip([u8; 16]),
    Date(i32),
    Timestamp { secs: i64, nanos: u32 },
    Decimal128(i128),
    Decimal256([u8; 32]),
    Float32Array(&'a [f32]),
    Float64Array(&'a [f64]),
}

struct BitWriter {
    buf: Vec<u8>,
    cur: u8,
    used: u8,
}

impl BitWriter {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            cur: 0,
            used: 0,
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            cur: 0,
            used: 0,
        }
    }

    fn write_bits(&mut self, mut value: u8, mut bits: u8) {
        while bits > 0 {
            let avail = 8 - self.used;
            let take = bits.min(avail);
            let mask = (1u16 << take) - 1;
            let chunk = (value & mask as u8) << self.used;
            self.cur |= chunk;
            self.used += take;
            value >>= take;
            bits -= take;
            if self.used == 8 {
                self.buf.push(self.cur);
                self.cur = 0;
                self.used = 0;
            }
        }
    }

    #[allow(dead_code)]
    fn finish(mut self) -> Vec<u8> {
        if self.used > 0 {
            self.buf.push(self.cur);
        }
        self.buf
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.cur = 0;
        self.used = 0;
    }

    fn append_to(&mut self, out: &mut Vec<u8>) {
        if self.used > 0 {
            self.buf.push(self.cur);
            self.cur = 0;
            self.used = 0;
        }
        out.extend_from_slice(&self.buf);
        self.buf.clear();
    }
}

struct BitWriterRef<'a> {
    buf: &'a mut Vec<u8>,
    cur: u8,
    used: u8,
}

impl<'a> BitWriterRef<'a> {
    fn new(buf: &'a mut Vec<u8>) -> Self {
        Self {
            buf,
            cur: 0,
            used: 0,
        }
    }

    fn write_bits(&mut self, mut value: u8, mut bits: u8) {
        while bits > 0 {
            let avail = 8 - self.used;
            let take = bits.min(avail);
            let mask = (1u16 << take) - 1;
            let chunk = (value & mask as u8) << self.used;
            self.cur |= chunk;
            self.used += take;
            value >>= take;
            bits -= take;
            if self.used == 8 {
                self.buf.push(self.cur);
                self.cur = 0;
                self.used = 0;
            }
        }
    }

    fn finish(&mut self) {
        if self.used > 0 {
            self.buf.push(self.cur);
            self.cur = 0;
            self.used = 0;
        }
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    idx: usize,
    cur: u8,
    used: u8,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            idx: 0,
            cur: 0,
            used: 8,
        }
    }

    fn read_bits(&mut self, bits: u8) -> Result<u8> {
        let mut out = 0u8;
        let mut offset = 0u8;
        let mut remaining = bits;
        while remaining > 0 {
            if self.used == 8 {
                if self.idx >= self.buf.len() {
                    return Err(anyhow!("rowpack bitstream truncated"));
                }
                self.cur = self.buf[self.idx];
                self.idx += 1;
                self.used = 0;
            }
            let avail = 8 - self.used;
            let take = remaining.min(avail);
            let mask = (1u16 << take) - 1;
            let chunk = (self.cur >> self.used) & mask as u8;
            out |= chunk << offset;
            self.used += take;
            remaining -= take;
            offset += take;
        }
        Ok(out)
    }

    fn read_bits_u32(&mut self, bits: u8) -> Result<u32> {
        if bits > 32 {
            return Err(anyhow!("rowpack bit width too large"));
        }
        let mut out = 0u32;
        let mut offset = 0u8;
        let mut remaining = bits;
        while remaining > 0 {
            if self.used == 8 {
                if self.idx >= self.buf.len() {
                    return Err(anyhow!("rowpack bitstream truncated"));
                }
                self.cur = self.buf[self.idx];
                self.idx += 1;
                self.used = 0;
            }
            let avail = 8 - self.used;
            let take = remaining.min(avail);
            let mask = (1u16 << take) - 1;
            let chunk = (self.cur >> self.used) & mask as u8;
            out |= (chunk as u32) << offset;
            self.used += take;
            remaining -= take;
            offset += take;
        }
        Ok(out)
    }

    fn align_to_byte(&mut self) {
        if self.used != 0 && self.used < 8 {
            self.used = 8;
        }
    }

    fn remaining_bytes(&self) -> usize {
        if self.used == 8 {
            self.buf.len().saturating_sub(self.idx)
        } else {
            self.buf.len().saturating_sub(self.idx).saturating_sub(1)
        }
    }
}

fn write_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn write_varint_u128(mut value: u128, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint(input: &mut &[u8]) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    let mut idx = 0usize;
    while idx < input.len() {
        let byte = input[idx];
        idx += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if (byte & 0x80) == 0 {
            *input = &input[idx..];
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err(anyhow!("varint overflow"));
        }
    }
    Err(anyhow!("varint truncated"))
}

fn read_varint_u128(input: &mut &[u8]) -> Result<u128> {
    let mut value = 0u128;
    let mut shift = 0u32;
    let mut idx = 0usize;
    while idx < input.len() {
        let byte = input[idx];
        idx += 1;
        value |= ((byte & 0x7f) as u128) << shift;
        if (byte & 0x80) == 0 {
            *input = &input[idx..];
            return Ok(value);
        }
        shift += 7;
        if shift > 127 {
            return Err(anyhow!("varint overflow"));
        }
    }
    Err(anyhow!("varint truncated"))
}

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

fn zigzag_encode_i128(value: i128) -> u128 {
    ((value << 1) ^ (value >> 127)) as u128
}

fn zigzag_decode_i128(value: u128) -> i128 {
    ((value >> 1) as i128) ^ (-((value & 1) as i128))
}

fn write_msgpack_timestamp(out: &mut Vec<u8>, secs: i64, nanos: u32) -> Result<()> {
    let secs_u = secs as i128;
    if secs_u < 0 {
        mp::write_ext_meta(out, 12, -1)?;
        out.extend_from_slice(&(nanos as u32).to_be_bytes());
        out.extend_from_slice(&(secs as i64).to_be_bytes());
        return Ok(());
    }
    let secs_u = secs_u as u64;
    if nanos == 0 && secs_u <= u32::MAX as u64 {
        mp::write_ext_meta(out, 4, -1)?;
        out.extend_from_slice(&(secs_u as u32).to_be_bytes());
        return Ok(());
    }
    if secs_u < (1u64 << 34) {
        let value = ((nanos as u64) << 34) | secs_u;
        mp::write_ext_meta(out, 8, -1)?;
        out.extend_from_slice(&value.to_be_bytes());
        return Ok(());
    }
    mp::write_ext_meta(out, 12, -1)?;
    out.extend_from_slice(&(nanos as u32).to_be_bytes());
    out.extend_from_slice(&(secs as i64).to_be_bytes());
    Ok(())
}

fn format_timestamp_default(secs: i64, nanos: u32) -> String {
    let dt = DateTime::<Utc>::from_timestamp(secs, nanos)
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
    dt.format("%Y-%m-%d %H:%M:%S%.f").to_string()
}

#[derive(Debug, Clone)]
pub struct RowPackContextBuilder {
    config: RowPackConfig,
    columns: Vec<IngestedType>,
    names: Vec<String>,
    fixed_lens: Vec<Option<usize>>,
    decimal_specs: Vec<Option<DecimalSpec>>,
    samples_seen: usize,
    stats: Vec<ColumnStats>,
}

#[derive(Debug, Clone, Copy)]
struct DecimalSpec {
    precision: u32,
    scale: i32,
}

impl RowPackContextBuilder {
    pub fn new(columns: Vec<RowPackColumn>, config: RowPackConfig) -> Self {
        let mut stats = Vec::with_capacity(columns.len());
        let mut names = Vec::with_capacity(columns.len());
        let mut types = Vec::with_capacity(columns.len());
        let mut fixed_lens = Vec::with_capacity(columns.len());
        let mut decimal_specs = Vec::with_capacity(columns.len());
        for col in columns {
            names.push(col.name);
            types.push(col.logical_type);
            fixed_lens.push(col.fixed_len);
            let spec = match col.logical_type {
                IngestedType::Decimal128 => match (col.precision, col.scale) {
                    (Some(precision), Some(scale)) => Some(DecimalSpec { precision, scale }),
                    _ => None,
                },
                IngestedType::Decimal256 => match (col.precision, col.scale) {
                    (Some(precision), Some(scale)) => Some(DecimalSpec { precision, scale }),
                    _ => None,
                },
                _ => None,
            };
            decimal_specs.push(spec);
            stats.push(ColumnStats::new());
        }
        Self {
            config,
            columns: types,
            names,
            fixed_lens,
            decimal_specs,
            samples_seen: 0,
            stats,
        }
    }

    pub fn observe_row<'a>(&mut self, values: &[ValueRef<'a>]) -> Result<()> {
        if self.samples_seen >= self.config.sample_rows() {
            return Ok(());
        }
        if values.len() != self.columns.len() {
            return Err(anyhow!("rowpack sample column mismatch"));
        }
        for (idx, value) in values.iter().enumerate() {
            self.stats[idx].observe(value, &self.config)?;
        }
        self.samples_seen += 1;
        Ok(())
    }

    pub fn observe_row_from_access(&mut self, access: &mut RowAccess<'_>) -> Result<()> {
        if self.samples_seen >= self.config.sample_rows() {
            return Ok(());
        }
        for (idx, logical) in self.columns.iter().enumerate() {
            let stats = self
                .stats
                .get_mut(idx)
                .ok_or_else(|| anyhow!("rowpack sample column mismatch"))?;
            match logical {
                IngestedType::Bool => match access.get_bool(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(_) => stats.observe_bool(),
                },
                IngestedType::Int64 => match access.get_int64(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(v) => stats.observe_int64(v),
                },
                IngestedType::Int16 | IngestedType::Int8 | IngestedType::Int32 => {
                    match access.get_int32(idx).ok().flatten() {
                        None => stats.observe_null(),
                        Some(v) => stats.observe_int32(v),
                    }
                }
                IngestedType::UInt64 => match access.get_uint64(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(v) => stats.observe_uint64(v),
                },
                IngestedType::UInt16 | IngestedType::UInt8 | IngestedType::UInt32 => {
                    match access.get_uint32(idx).ok().flatten() {
                        None => stats.observe_null(),
                        Some(v) => stats.observe_uint32(v),
                    }
                }
                IngestedType::Float64 => match access.get_float64(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(v) => stats.observe_float64(v),
                },
                IngestedType::Float32 => match access.get_float32(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(v) => stats.observe_float32(v),
                },
                IngestedType::String => {
                    match access.with_string(idx, |v| {
                        stats.observe_string(v, &self.config);
                        Ok(())
                    })? {
                        None => stats.observe_null(),
                        Some(_) => {}
                    }
                }
                IngestedType::Bytes => match access.with_bytes(idx, |v| Ok(v.len()))? {
                    None => stats.observe_null(),
                    Some(len) => stats.observe_bytes_len(len),
                },
                IngestedType::Uuid => match access.get_uuid(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(_) => stats.observe_uuid(),
                },
                IngestedType::Uuid256 => match access.get_uuid256(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(_) => stats.observe_uuid256(),
                },
                IngestedType::Ip => match access.get_ip(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(_) => stats.observe_ip(),
                },
                IngestedType::Date => match access.get_date(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(_) => stats.count = stats.count.saturating_add(1),
                },
                IngestedType::Timestamp => match access.get_timestamp(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some((secs, _)) => stats.observe_timestamp(secs),
                },
                IngestedType::Decimal128 => match access.get_decimal128(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(_) => stats.count = stats.count.saturating_add(1),
                },
                IngestedType::Decimal256 => match access.get_decimal256(idx).ok().flatten() {
                    None => stats.observe_null(),
                    Some(_) => stats.count = stats.count.saturating_add(1),
                },
                IngestedType::Float32Array | IngestedType::Float16Array => {
                    match access.with_float32_array(idx, |values| {
                        stats.observe_array_f32(values);
                        Ok(())
                    })? {
                        None => stats.observe_null(),
                        Some(_) => {}
                    }
                }
                IngestedType::Float64Array => {
                    match access.with_float64_array(idx, |values| {
                        stats.observe_array_f64(values);
                        Ok(())
                    })? {
                        None => stats.observe_null(),
                        Some(_) => {}
                    }
                }
            }
        }
        self.samples_seen += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<RowPackContext> {
        let mut columns = Vec::with_capacity(self.columns.len());
        let mut estimate = 0usize;
        let mut bools = 0usize;
        for (idx, logical_type) in self.columns.iter().enumerate() {
            let encoding = match (logical_type, self.decimal_specs[idx]) {
                (IngestedType::Decimal128, Some(spec)) => {
                    if spec.precision == 0 {
                        return Err(anyhow!("rowpack decimal precision missing"));
                    }
                    let bytes = decimal_raw_bytes(spec.precision)?;
                    ColumnEncoding::Decimal {
                        precision: spec.precision,
                        scale: spec.scale,
                        bytes,
                        storage: DecimalStorage::Varint,
                    }
                }
                (IngestedType::Decimal256, Some(spec)) => {
                    if spec.precision == 0 {
                        return Err(anyhow!("rowpack decimal precision missing"));
                    }
                    let bytes = decimal_raw_bytes(spec.precision)?;
                    ColumnEncoding::Decimal {
                        precision: spec.precision,
                        scale: spec.scale,
                        bytes,
                        storage: DecimalStorage::Raw,
                    }
                }
                (IngestedType::Decimal128, None) | (IngestedType::Decimal256, None) => {
                    return Err(anyhow!("rowpack decimal precision missing"));
                }
                _ => self.stats[idx].choose_encoding(
                    *logical_type,
                    &self.config,
                    self.fixed_lens[idx],
                )?,
            };
            estimate += estimate_column_bytes(&encoding, &self.stats[idx]);
            if matches!(encoding, ColumnEncoding::Bool) {
                bools += 1;
            }
            columns.push(ColumnContext {
                name: self.names[idx].clone(),
                logical_type: *logical_type,
                encoding,
            });
        }
        let tag_bytes = (self.columns.len() * 2 + 7) / 8;
        let bool_bytes = (bools + 7) / 8;
        estimate += tag_bytes + bool_bytes;
        Ok(RowPackContext {
            version: ROWPACK_VERSION,
            columns,
            estimated_row_bytes: estimate.min(u32::MAX as usize) as u32,
        })
    }
}

fn estimate_column_bytes(encoding: &ColumnEncoding, stats: &ColumnStats) -> usize {
    match encoding {
        ColumnEncoding::Bool => 0,
        ColumnEncoding::Int { varint, bits, .. } => {
            if *varint {
                varint_bytes(*bits)
            } else {
                ((*bits + 7) / 8) as usize
            }
        }
        ColumnEncoding::Date { varint, bits } => {
            if *varint {
                varint_bytes(*bits)
            } else {
                ((*bits + 7) / 8) as usize
            }
        }
        ColumnEncoding::Timestamp => {
            let secs_bits = if stats.ts_seen {
                int_bits(stats.ts_min, stats.ts_max)
            } else {
                64
            };
            varint_bytes(secs_bits) + 5
        }
        ColumnEncoding::Float { storage } => match storage {
            FloatStorage::F16 => 2,
            FloatStorage::F32 => 4,
            FloatStorage::F64 => 8,
        },
        ColumnEncoding::Bytes { fixed_len } => {
            let avg = avg_len(stats.bytes_total, stats.bytes_count);
            let len = fixed_len.unwrap_or(avg);
            if fixed_len.is_some() {
                len
            } else {
                varint_bytes_u64(len as u64) + len
            }
        }
        ColumnEncoding::Uuid => 16,
        ColumnEncoding::Uuid256 => 32,
        ColumnEncoding::Ip => 16,
        ColumnEncoding::String { encoding } => match encoding {
            StringEncoding::Dict { .. } => varint_bytes_u64(1024),
            StringEncoding::Alphabet { alphabet } => {
                let avg = avg_len(stats.string_bytes, stats.string_total);
                let bits = alphabet_bits(alphabet.len());
                let packed = ((avg * bits as usize) + 7) / 8;
                varint_bytes_u64(avg as u64) + packed
            }
            StringEncoding::Bpe { .. } => {
                let avg = avg_len(stats.string_bytes, stats.string_total);
                let bits = bpe_token_bits(match encoding {
                    StringEncoding::Bpe { dict } => dict.len(),
                    _ => 0,
                }) as usize;
                let packed = ((avg * bits) + 7) / 8;
                varint_bytes_u64(avg as u64) + packed
            }
            StringEncoding::Lz4 => {
                let avg = avg_len(stats.string_bytes, stats.string_total);
                varint_bytes_u64(avg as u64) + varint_bytes_u64(avg as u64) + avg
            }
            StringEncoding::Raw => {
                let avg = avg_len(stats.string_bytes, stats.string_total);
                varint_bytes_u64(avg as u64) + avg
            }
        },
        ColumnEncoding::Decimal { bytes, .. } => *bytes as usize,
        ColumnEncoding::FloatArray { storage } => {
            let avg_len = avg_len(stats.array_len_total, stats.array_count);
            let elem = match storage {
                FloatStorage::F16 => 2,
                FloatStorage::F32 => 4,
                FloatStorage::F64 => 8,
            };
            varint_bytes_u64(avg_len as u64) + avg_len * elem
        }
    }
}

fn varint_bytes(bits: u8) -> usize {
    ((bits as usize) + 6) / 7
}

fn varint_bytes_u64(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    ((bits.max(1)) + 6) / 7
}

fn varint_bytes_u128(value: u128) -> usize {
    let bits = 128 - value.leading_zeros() as usize;
    ((bits.max(1)) + 6) / 7
}

fn int_bits(min: i64, max: i64) -> u8 {
    let max_abs = min.abs().max(max.abs()) as u64;
    let bits = 64 - max_abs.leading_zeros() as u8;
    bits.max(1)
}

fn avg_len(total: usize, count: usize) -> usize {
    if count == 0 {
        0
    } else {
        total / count
    }
}

fn decimal_raw_bytes(precision: u32) -> Result<u8> {
    if precision == 0 {
        return Err(anyhow!("rowpack decimal precision missing"));
    }
    let bits = (precision as f64 * 3.321928094887362_f64).ceil() as u32;
    let bytes = ((bits + 7) / 8).max(1);
    u8::try_from(bytes).map_err(|_| anyhow!("rowpack decimal precision too large"))
}

#[derive(Debug, Clone)]
struct ColumnStats {
    count: usize,
    nulls: usize,
    int_min: i64,
    int_max: i64,
    uint_max: u64,
    float_min: f64,
    float_max: f64,
    float_has_nan: bool,
    float_has_inf: bool,
    bool_seen: bool,
    string_counts: FastHashMap<Vec<u8>, usize>,
    string_max_len: usize,
    string_samples: Vec<Vec<u8>>,
    string_sample_bytes: usize,
    alphabet_bytes: [bool; 256],
    alphabet_len: usize,
    alphabet_ok: bool,
    alphabet_hits: usize,
    string_bytes: usize,
    string_total: usize,
    bytes_total: usize,
    bytes_count: usize,
    string_long_bytes: usize,
    string_long_raw: usize,
    string_long_compressed: usize,
    array_len_total: usize,
    array_count: usize,
    ts_min: i64,
    ts_max: i64,
    ts_seen: bool,
}

impl ColumnStats {
    fn new() -> Self {
        Self {
            count: 0,
            nulls: 0,
            int_min: i64::MAX,
            int_max: i64::MIN,
            uint_max: 0,
            float_min: f64::INFINITY,
            float_max: f64::NEG_INFINITY,
            float_has_nan: false,
            float_has_inf: false,
            bool_seen: false,
            string_counts: FastHashMap::new(),
            string_max_len: 0,
            string_samples: Vec::new(),
            string_sample_bytes: 0,
            alphabet_bytes: [false; 256],
            alphabet_len: 0,
            alphabet_ok: true,
            alphabet_hits: 0,
            string_bytes: 0,
            string_total: 0,
            bytes_total: 0,
            bytes_count: 0,
            string_long_bytes: 0,
            string_long_raw: 0,
            string_long_compressed: 0,
            array_len_total: 0,
            array_count: 0,
            ts_min: i64::MAX,
            ts_max: i64::MIN,
            ts_seen: false,
        }
    }

    fn observe<'a>(&mut self, value: &ValueRef<'a>, config: &RowPackConfig) -> Result<()> {
        self.count += 1;
        match value {
            ValueRef::Null => {
                self.nulls += 1;
            }
            ValueRef::Int64(v) => {
                self.int_min = self.int_min.min(*v);
                self.int_max = self.int_max.max(*v);
            }
            ValueRef::Int32(v) => {
                let v = *v as i64;
                self.int_min = self.int_min.min(v);
                self.int_max = self.int_max.max(v);
            }
            ValueRef::UInt64(v) => {
                self.uint_max = self.uint_max.max(*v);
            }
            ValueRef::UInt32(v) => {
                self.uint_max = self.uint_max.max(*v as u64);
            }
            ValueRef::Float64(v) => {
                if v.is_nan() {
                    self.float_has_nan = true;
                } else if v.is_infinite() {
                    self.float_has_inf = true;
                } else {
                    self.float_min = self.float_min.min(*v);
                    self.float_max = self.float_max.max(*v);
                }
            }
            ValueRef::Float32(v) => {
                let v = *v as f64;
                if v.is_nan() {
                    self.float_has_nan = true;
                } else if v.is_infinite() {
                    self.float_has_inf = true;
                } else {
                    self.float_min = self.float_min.min(v);
                    self.float_max = self.float_max.max(v);
                }
            }
            ValueRef::Bool(_) => {
                self.bool_seen = true;
            }
            ValueRef::String(v) => {
                self.string_total += 1;
                let bytes = v.as_bytes();
                self.string_max_len = self.string_max_len.max(bytes.len());
                self.string_bytes += bytes.len();
                if self.string_counts.len() < 1_000_000 {
                    *self.string_counts.entry(bytes.to_vec()).or_insert(0) += 1;
                }
                if self.string_samples.len() < config.bpe_max_samples() {
                    if self.string_sample_bytes + bytes.len() <= config.bpe_max_dict_bytes() * 4 {
                        self.string_samples.push(bytes.to_vec());
                        self.string_sample_bytes += bytes.len();
                    }
                }
                if self.alphabet_ok {
                    if bytes.len() > config.string_long_min_len() {
                        self.alphabet_ok = false;
                    } else {
                        let mut ok = true;
                        for b in bytes {
                            let idx = *b as usize;
                            if !self.alphabet_bytes[idx] {
                                if self.alphabet_len + 1 > ALPHABET_MAX_CHARS {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok {
                            for b in bytes {
                                let idx = *b as usize;
                                if !self.alphabet_bytes[idx] {
                                    self.alphabet_bytes[idx] = true;
                                    self.alphabet_len += 1;
                                }
                            }
                            self.alphabet_hits += 1;
                        } else {
                            self.alphabet_ok = false;
                        }
                    }
                }
                if bytes.len() >= config.string_long_min_len() {
                    let raw_len = bytes.len();
                    let raw_total = varint_bytes_u64(raw_len as u64) + raw_len;
                    self.string_long_bytes += raw_len;
                    self.string_long_raw += raw_total;
                    let max = lz4_flex::block::get_maximum_output_size(raw_len);
                    let mut buf = vec![0u8; max];
                    let written =
                        lz4_flex::block::compress_into(bytes, &mut buf).unwrap_or(raw_len);
                    let compressed_total = varint_bytes_u64(raw_len as u64)
                        + varint_bytes_u64(written as u64)
                        + written;
                    self.string_long_compressed += compressed_total;
                }
            }
            ValueRef::Bytes(v) => {
                self.bytes_total += v.len();
                self.bytes_count += 1;
            }
            ValueRef::Uuid(_) => {
                self.bytes_total += 16;
                self.bytes_count += 1;
            }
            ValueRef::Uuid256(_) => {
                self.bytes_total += 32;
                self.bytes_count += 1;
            }
            ValueRef::Ip(_) => {
                self.bytes_total += 16;
                self.bytes_count += 1;
            }
            ValueRef::Date(_) => {}
            ValueRef::Timestamp { secs, .. } => {
                self.ts_min = self.ts_min.min(*secs);
                self.ts_max = self.ts_max.max(*secs);
                self.ts_seen = true;
            }
            ValueRef::Decimal128(_) => {}
            ValueRef::Decimal256(_) => {}
            ValueRef::Float32Array(values) => {
                self.array_len_total += values.len();
                self.array_count += 1;
                for v in *values {
                    let v = *v as f64;
                    if v.is_nan() {
                        self.float_has_nan = true;
                    } else if v.is_infinite() {
                        self.float_has_inf = true;
                    } else {
                        self.float_min = self.float_min.min(v);
                        self.float_max = self.float_max.max(v);
                    }
                }
            }
            ValueRef::Float64Array(values) => {
                self.array_len_total += values.len();
                self.array_count += 1;
                for v in *values {
                    if v.is_nan() {
                        self.float_has_nan = true;
                    } else if v.is_infinite() {
                        self.float_has_inf = true;
                    } else {
                        self.float_min = self.float_min.min(*v);
                        self.float_max = self.float_max.max(*v);
                    }
                }
            }
        }
        Ok(())
    }

    fn observe_null(&mut self) {
        self.count += 1;
        self.nulls += 1;
    }

    fn observe_int64(&mut self, v: i64) {
        self.count += 1;
        self.int_min = self.int_min.min(v);
        self.int_max = self.int_max.max(v);
    }

    fn observe_int32(&mut self, v: i32) {
        self.observe_int64(v as i64);
    }

    fn observe_uint64(&mut self, v: u64) {
        self.count += 1;
        self.uint_max = self.uint_max.max(v);
    }

    fn observe_uint32(&mut self, v: u32) {
        self.observe_uint64(v as u64);
    }

    fn observe_float64(&mut self, v: f64) {
        self.count += 1;
        self.observe_float_stats(v);
    }

    fn observe_float32(&mut self, v: f32) {
        self.observe_float64(v as f64);
    }

    fn observe_bool(&mut self) {
        self.count += 1;
        self.bool_seen = true;
    }

    fn observe_string(&mut self, value: &str, config: &RowPackConfig) {
        self.count += 1;
        self.string_total += 1;
        let bytes = value.as_bytes();
        self.string_max_len = self.string_max_len.max(bytes.len());
        self.string_bytes += bytes.len();
        if self.string_counts.len() < 1_000_000 {
            *self.string_counts.entry(bytes.to_vec()).or_insert(0) += 1;
        }
        if self.string_samples.len() < config.bpe_max_samples() {
            if self.string_sample_bytes + bytes.len() <= config.bpe_max_dict_bytes() * 4 {
                self.string_samples.push(bytes.to_vec());
                self.string_sample_bytes += bytes.len();
            }
        }
        if self.alphabet_ok {
            if bytes.len() > config.string_long_min_len() {
                self.alphabet_ok = false;
            } else {
                let mut ok = true;
                for b in bytes {
                    let idx = *b as usize;
                    if !self.alphabet_bytes[idx] {
                        if self.alphabet_len + 1 > ALPHABET_MAX_CHARS {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    for b in bytes {
                        let idx = *b as usize;
                        if !self.alphabet_bytes[idx] {
                            self.alphabet_bytes[idx] = true;
                            self.alphabet_len += 1;
                        }
                    }
                    self.alphabet_hits += 1;
                } else {
                    self.alphabet_ok = false;
                }
            }
        }
        if bytes.len() >= config.string_long_min_len() {
            let raw_len = bytes.len();
            let raw_total = varint_bytes_u64(raw_len as u64) + raw_len;
            self.string_long_bytes += raw_len;
            self.string_long_raw += raw_total;
            let max = lz4_flex::block::get_maximum_output_size(raw_len);
            let mut buf = vec![0u8; max];
            let written = lz4_flex::block::compress_into(bytes, &mut buf).unwrap_or(raw_len);
            let compressed_total =
                varint_bytes_u64(raw_len as u64) + varint_bytes_u64(written as u64) + written;
            self.string_long_compressed += compressed_total;
        }
    }

    fn observe_bytes_len(&mut self, len: usize) {
        self.count += 1;
        self.bytes_total += len;
        self.bytes_count += 1;
    }

    fn observe_uuid(&mut self) {
        self.observe_bytes_len(16);
    }

    fn observe_uuid256(&mut self) {
        self.observe_bytes_len(32);
    }

    fn observe_ip(&mut self) {
        self.observe_bytes_len(16);
    }

    fn observe_timestamp(&mut self, secs: i64) {
        self.count += 1;
        self.ts_min = self.ts_min.min(secs);
        self.ts_max = self.ts_max.max(secs);
        self.ts_seen = true;
    }

    fn observe_array_f32(&mut self, values: &[f32]) {
        self.count += 1;
        self.array_len_total += values.len();
        self.array_count += 1;
        for v in values {
            self.observe_float_stats(*v as f64);
        }
    }

    fn observe_array_f64(&mut self, values: &[f64]) {
        self.count += 1;
        self.array_len_total += values.len();
        self.array_count += 1;
        for v in values {
            self.observe_float_stats(*v);
        }
    }

    fn observe_float_stats(&mut self, v: f64) {
        if v.is_nan() {
            self.float_has_nan = true;
        } else if v.is_infinite() {
            self.float_has_inf = true;
        } else {
            self.float_min = self.float_min.min(v);
            self.float_max = self.float_max.max(v);
        }
    }

    fn choose_encoding(
        &self,
        logical: IngestedType,
        config: &RowPackConfig,
        fixed_len: Option<usize>,
    ) -> Result<ColumnEncoding> {
        match logical {
            IngestedType::Int64 => Ok(ColumnEncoding::Int {
                signed: true,
                varint: int_use_varint(self.int_min, self.int_max),
                bits: 64,
            }),
            IngestedType::Int32 => Ok(ColumnEncoding::Int {
                signed: true,
                varint: int_use_varint(self.int_min, self.int_max),
                bits: 32,
            }),
            IngestedType::Int16 => Ok(ColumnEncoding::Int {
                signed: true,
                varint: int_use_varint(self.int_min, self.int_max),
                bits: 16,
            }),
            IngestedType::Int8 => Ok(ColumnEncoding::Int {
                signed: true,
                varint: false,
                bits: 8,
            }),
            IngestedType::UInt64 => Ok(ColumnEncoding::Int {
                signed: false,
                varint: uint_use_varint(self.uint_max),
                bits: 64,
            }),
            IngestedType::UInt32 => Ok(ColumnEncoding::Int {
                signed: false,
                varint: uint_use_varint(self.uint_max),
                bits: 32,
            }),
            IngestedType::UInt16 => Ok(ColumnEncoding::Int {
                signed: false,
                varint: uint_use_varint(self.uint_max),
                bits: 16,
            }),
            IngestedType::UInt8 => Ok(ColumnEncoding::Int {
                signed: false,
                varint: false,
                bits: 8,
            }),
            IngestedType::Float64 => Ok(ColumnEncoding::Float {
                storage: choose_float_storage(true),
            }),
            IngestedType::Float32 => Ok(ColumnEncoding::Float {
                storage: choose_float_storage(false),
            }),
            IngestedType::Bool => Ok(ColumnEncoding::Bool),
            IngestedType::String => Ok(ColumnEncoding::String {
                encoding: choose_string_encoding(self, config)?,
            }),
            IngestedType::Bytes => Ok(ColumnEncoding::Bytes { fixed_len }),
            IngestedType::Uuid => Ok(ColumnEncoding::Uuid),
            IngestedType::Uuid256 => Ok(ColumnEncoding::Uuid256),
            IngestedType::Ip => Ok(ColumnEncoding::Ip),
            IngestedType::Date => Ok(ColumnEncoding::Date {
                varint: int_use_varint(self.int_min, self.int_max),
                bits: 32,
            }),
            IngestedType::Timestamp => Ok(ColumnEncoding::Timestamp),
            IngestedType::Decimal128 | IngestedType::Decimal256 => {
                Err(anyhow!("rowpack decimal encoding missing"))
            }
            IngestedType::Float32Array => Ok(ColumnEncoding::FloatArray {
                storage: choose_float_storage(false),
            }),
            IngestedType::Float64Array => Ok(ColumnEncoding::FloatArray {
                storage: choose_float_storage(true),
            }),
            IngestedType::Float16Array => Ok(ColumnEncoding::FloatArray {
                storage: FloatStorage::F16,
            }),
        }
    }
}

fn int_use_varint(min: i64, max: i64) -> bool {
    let max_abs = min.abs().max(max.abs()) as u64;
    let bits = 64 - max_abs.leading_zeros() as u8;
    bits <= 32
}

fn uint_use_varint(max: u64) -> bool {
    let bits = 64 - max.leading_zeros() as u8;
    bits <= 32
}

fn choose_float_storage(is_f64: bool) -> FloatStorage {
    if is_f64 {
        FloatStorage::F64
    } else {
        FloatStorage::F32
    }
}

fn choose_string_encoding(stats: &ColumnStats, config: &RowPackConfig) -> Result<StringEncoding> {
    let total = stats.string_total.max(1) as f32;
    let avg_len = if stats.string_total == 0 {
        0
    } else {
        stats.string_bytes / stats.string_total
    };
    let bpe_allowed = avg_len >= config.bpe_min_avg_len() && avg_len < config.string_long_min_len();
    if stats.alphabet_ok && stats.alphabet_len > 1 {
        let coverage_ratio = stats.alphabet_hits as f32 / total;
        if coverage_ratio >= ALPHABET_MIN_COVERAGE {
            let mut alphabet = Vec::with_capacity(stats.alphabet_len);
            for (idx, seen) in stats.alphabet_bytes.iter().enumerate() {
                if *seen {
                    alphabet.push(idx as u8);
                }
            }
            let mut best = StringEncoding::Alphabet { alphabet };
            if bpe_allowed && !stats.string_samples.is_empty() {
                let bpe_dict = train_bpe_dict(
                    &stats.string_samples,
                    config.bpe_max_dict_entries(),
                    config.bpe_max_dict_bytes(),
                );
                if !bpe_dict.is_empty() {
                    let bpe_bytes = estimate_bpe_bytes(&stats.string_samples, &bpe_dict);
                    let alpha_bytes =
                        estimate_alphabet_bytes(&stats.string_samples, stats.alphabet_len);
                    if bpe_bytes < (alpha_bytes as f64 * config.bpe_min_gain_ratio() as f64) as u64
                    {
                        best = StringEncoding::Bpe { dict: bpe_dict };
                    }
                }
            }
            return Ok(best);
        }
    }
    if bpe_allowed && !stats.string_samples.is_empty() {
        let bpe_dict = train_bpe_dict(
            &stats.string_samples,
            config.bpe_max_dict_entries(),
            config.bpe_max_dict_bytes(),
        );
        if !bpe_dict.is_empty() {
            let bpe_bytes = estimate_bpe_bytes(&stats.string_samples, &bpe_dict);
            let raw_bytes = estimate_raw_bytes(&stats.string_samples);
            if bpe_bytes < (raw_bytes as f64 * config.bpe_min_gain_ratio() as f64) as u64 {
                return Ok(StringEncoding::Bpe { dict: bpe_dict });
            }
        }
    }
    if !stats.string_counts.is_empty() {
        let mut entries: Vec<(Vec<u8>, usize)> = stats
            .string_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        let mut values = Vec::new();
        let mut bytes = 0usize;
        let mut coverage = 0usize;
        let max_bytes = config.max_dict_bytes();
        let max_entries = config.max_dict_entries();
        for (value, count) in entries {
            let next_bytes = bytes + value.len();
            if next_bytes > max_bytes || values.len() >= max_entries {
                break;
            }
            bytes = next_bytes;
            coverage += count;
            values.push(value);
        }
        let coverage_ratio = coverage as f32 / total;
        let avg_hits = coverage as f32 / values.len().max(1) as f32;
        if coverage_ratio >= config.string_dict_coverage() && avg_hits >= 2.0 && !values.is_empty()
        {
            return Ok(StringEncoding::Dict { values });
        }
    }
    if stats.string_long_bytes > 0 {
        let ratio = stats.string_long_compressed as f32 / stats.string_long_raw as f32;
        if ratio <= config.string_long_min_ratio() {
            return Ok(StringEncoding::Lz4);
        }
    }
    Ok(StringEncoding::Raw)
}

fn train_bpe_dict(samples: &[Vec<u8>], max_entries: usize, max_bytes: usize) -> Vec<Vec<u8>> {
    if max_entries == 0 || max_bytes == 0 || samples.is_empty() {
        return Vec::new();
    }
    let mut sequences: Vec<Vec<u16>> = samples
        .iter()
        .map(|s| s.iter().map(|b| *b as u16).collect())
        .collect();
    let mut symbols: Vec<Vec<u8>> = (0u16..=255u16).map(|b| vec![b as u8]).collect();
    let mut dict_entries = Vec::new();
    let mut dict_bytes = 0usize;
    for _ in 0..max_entries {
        let mut pair_counts: FastHashMap<(u16, u16), usize> = FastHashMap::new();
        for seq in &sequences {
            if seq.len() < 2 {
                continue;
            }
            for window in seq.windows(2) {
                let key = (window[0], window[1]);
                *pair_counts.entry(key).or_insert(0) += 1;
            }
        }
        let mut best_pair = None;
        let mut best_count = 0usize;
        for (pair, count) in pair_counts {
            if count > best_count {
                best_pair = Some(pair);
                best_count = count;
            }
        }
        let Some((left, right)) = best_pair else {
            break;
        };
        if best_count < 2 {
            break;
        }
        let mut merged = Vec::new();
        merged.extend_from_slice(&symbols[left as usize]);
        merged.extend_from_slice(&symbols[right as usize]);
        if dict_bytes + merged.len() > max_bytes {
            break;
        }
        let new_symbol = symbols.len() as u16;
        symbols.push(merged.clone());
        dict_entries.push(merged);
        dict_bytes += dict_entries.last().map(|v| v.len()).unwrap_or(0);
        for seq in sequences.iter_mut() {
            if seq.len() < 2 {
                continue;
            }
            let mut out = Vec::with_capacity(seq.len());
            let mut i = 0usize;
            while i < seq.len() {
                if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
                    out.push(new_symbol);
                    i += 2;
                } else {
                    out.push(seq[i]);
                    i += 1;
                }
            }
            *seq = out;
        }
    }
    let mut dict_counts = vec![0usize; dict_entries.len()];
    if !dict_entries.is_empty() {
        let trie = BpeTrie::from_dict(&dict_entries);
        let mut tokens = Vec::new();
        for sample in samples {
            tokenize_bpe(sample, &trie, &mut tokens);
            for token in tokens.iter().copied() {
                if token >= 256 {
                    let idx = (token - 256) as usize;
                    if let Some(count) = dict_counts.get_mut(idx) {
                        *count += 1;
                    }
                }
            }
        }
    }
    let mut entries: Vec<(Vec<u8>, usize)> = dict_entries
        .into_iter()
        .zip(dict_counts.into_iter())
        .collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.into_iter().map(|(bytes, _)| bytes).collect()
}

pub struct RowPackEncoder {
    ctx: RowPackContext,
    dict_maps: Vec<Option<FastHashMap<Vec<u8>, u32>>>,
    alphabet_maps: Vec<Option<AlphabetMap>>,
    bpe_tries: Vec<Option<BpeTrie>>,
}

#[derive(Debug, Clone, Copy)]
pub struct RowPackRowLens {
    pub tag_bytes: usize,
    pub bool_bytes: usize,
    pub data_bytes: usize,
}

pub struct RowPackScratch {
    tags: BitWriter,
    bools: BitWriter,
    data: Vec<u8>,
    bpe_tokens: Vec<u32>,
    lz4_buf: Vec<u8>,
}

impl RowPackScratch {
    pub fn new(ctx: &RowPackContext) -> Self {
        let tag_bytes = (ctx.columns.len() * 2 + 7) / 8;
        let bool_columns = ctx
            .columns
            .iter()
            .filter(|c| matches!(c.encoding, ColumnEncoding::Bool))
            .count();
        let bool_bytes = (bool_columns + 7) / 8;
        Self {
            tags: BitWriter::with_capacity(tag_bytes),
            bools: BitWriter::with_capacity(bool_bytes),
            data: Vec::with_capacity(ctx.estimated_row_bytes as usize),
            bpe_tokens: Vec::new(),
            lz4_buf: Vec::new(),
        }
    }

    fn reset(&mut self, ctx: &RowPackContext) {
        let tag_bytes = (ctx.columns.len() * 2 + 7) / 8;
        if self.tags.buf.capacity() < tag_bytes {
            self.tags.buf.reserve(tag_bytes - self.tags.buf.capacity());
        }
        let bool_columns = ctx
            .columns
            .iter()
            .filter(|c| matches!(c.encoding, ColumnEncoding::Bool))
            .count();
        let bool_bytes = (bool_columns + 7) / 8;
        if self.bools.buf.capacity() < bool_bytes {
            self.bools
                .buf
                .reserve(bool_bytes - self.bools.buf.capacity());
        }
        if self.data.capacity() < ctx.estimated_row_bytes as usize {
            self.data
                .reserve(ctx.estimated_row_bytes as usize - self.data.capacity());
        }
        self.tags.reset();
        self.bools.reset();
        self.data.clear();
        self.bpe_tokens.clear();
        self.lz4_buf.clear();
    }
}

impl RowPackEncoder {
    pub fn new(ctx: RowPackContext) -> Result<Self> {
        let mut dict_maps = Vec::with_capacity(ctx.columns.len());
        let mut alphabet_maps = Vec::with_capacity(ctx.columns.len());
        let mut bpe_tries = Vec::with_capacity(ctx.columns.len());
        for col in &ctx.columns {
            match &col.encoding {
                ColumnEncoding::String {
                    encoding: StringEncoding::Dict { values },
                } => {
                    let mut map = FastHashMap::with_capacity(values.len());
                    for (idx, value) in values.iter().enumerate() {
                        map.insert(value.clone(), idx as u32);
                    }
                    dict_maps.push(Some(map));
                }
                _ => dict_maps.push(None),
            }
            match &col.encoding {
                ColumnEncoding::String {
                    encoding: StringEncoding::Alphabet { alphabet },
                } => {
                    let mut map = [ALPHABET_INVALID; 256];
                    for (idx, b) in alphabet.iter().enumerate() {
                        map[*b as usize] = idx as u8;
                    }
                    let bits = alphabet_bits(alphabet.len());
                    let packer = match bits {
                        1 => AlphabetPacker::Bits1,
                        2 => AlphabetPacker::Bits2,
                        3 => AlphabetPacker::Bits3,
                        4 => AlphabetPacker::Bits4,
                        5 => AlphabetPacker::Bits5,
                        6 => AlphabetPacker::Bits6,
                        _ => AlphabetPacker::Generic { bits },
                    };
                    alphabet_maps.push(Some(AlphabetMap { map, bits, packer }));
                }
                _ => alphabet_maps.push(None),
            }
            match &col.encoding {
                ColumnEncoding::String {
                    encoding: StringEncoding::Bpe { dict },
                } => {
                    bpe_tries.push(Some(BpeTrie::from_dict(dict)));
                }
                _ => bpe_tries.push(None),
            }
        }
        Ok(Self {
            ctx,
            dict_maps,
            alphabet_maps,
            bpe_tries,
        })
    }

    pub fn context(&self) -> &RowPackContext {
        &self.ctx
    }

    pub fn encode_row_into<'a>(
        &self,
        values: &[ValueRef<'a>],
        out: &mut Vec<u8>,
        scratch: &mut RowPackScratch,
    ) -> Result<()> {
        if values.len() != self.ctx.columns.len() {
            return Err(anyhow!("rowpack column mismatch"));
        }
        scratch.reset(&self.ctx);
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let value = &values[idx];
            let tag = match value {
                ValueRef::Null => 0u8,
                _ if matches!(col.encoding, ColumnEncoding::Bool) => {
                    let v = match value {
                        ValueRef::Bool(v) => *v,
                        _ => return Err(anyhow!("rowpack bool mismatch")),
                    };
                    scratch.bools.write_bits(if v { 1 } else { 0 }, 1);
                    1u8
                }
                _ => match (&col.encoding, value) {
                    (ColumnEncoding::String { .. }, ValueRef::String(v)) => encode_string(
                        &mut scratch.data,
                        v,
                        col,
                        self.dict_maps[idx].as_ref(),
                        self.alphabet_maps[idx].as_ref(),
                        self.bpe_tries[idx].as_ref(),
                        Some(&mut scratch.bpe_tokens),
                        Some(&mut scratch.lz4_buf),
                    )?,
                    _ => encode_value_with_tag(
                        &mut scratch.data,
                        col,
                        value,
                        self.dict_maps[idx].as_ref(),
                        self.alphabet_maps[idx].as_ref(),
                        self.bpe_tries[idx].as_ref(),
                        Some(&mut scratch.bpe_tokens),
                        Some(&mut scratch.lz4_buf),
                    )?,
                },
            };
            scratch.tags.write_bits(tag, 2);
        }
        out.clear();
        out.reserve(scratch.tags.buf.len() + scratch.bools.buf.len() + scratch.data.len() + 8);
        scratch.tags.append_to(out);
        scratch.bools.append_to(out);
        out.extend_from_slice(&scratch.data);
        Ok(())
    }

    pub fn encode_row_from_access(
        &self,
        access: &mut RowAccess<'_>,
        out: &mut Vec<u8>,
        scratch: &mut RowPackScratch,
    ) -> Result<()> {
        scratch.reset(&self.ctx);
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = match col.logical_type {
                IngestedType::Bool => match access.get_bool(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        scratch.bools.write_bits(if v { 1 } else { 0 }, 1);
                        1u8
                    }
                },
                IngestedType::Int64 => match access.get_int64(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int64 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Int16 => match access.get_int32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int16 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Int8 => match access.get_int32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int8 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Int32 => match access.get_int32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int32 encoding mismatch"));
                        }
                    }
                },
                IngestedType::UInt64 => match access.get_uint64(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint64 encoding mismatch"));
                        }
                    }
                },
                IngestedType::UInt16 => match access.get_uint32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint16 encoding mismatch"));
                        }
                    }
                },
                IngestedType::UInt8 => match access.get_uint32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint8 encoding mismatch"));
                        }
                    }
                },
                IngestedType::UInt32 => match access.get_uint32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint32 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float64 => match access.get_float64(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Float { storage } = &col.encoding {
                            encode_float_with_tag(&mut scratch.data, col.logical_type, *storage, v)?
                        } else {
                            return Err(anyhow!("rowpack float64 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float32 => match access.get_float32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Float { storage } = &col.encoding {
                            encode_float_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                v as f64,
                            )?
                        } else {
                            return Err(anyhow!("rowpack float32 encoding mismatch"));
                        }
                    }
                },
                IngestedType::String => {
                    match access.with_string(idx, |v| {
                        encode_string(
                            &mut scratch.data,
                            v,
                            col,
                            self.dict_maps[idx].as_ref(),
                            self.alphabet_maps[idx].as_ref(),
                            self.bpe_tries[idx].as_ref(),
                            Some(&mut scratch.bpe_tokens),
                            Some(&mut scratch.lz4_buf),
                        )
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
                IngestedType::Bytes => {
                    match access.with_bytes(idx, |v| {
                        if let ColumnEncoding::Bytes { fixed_len } = &col.encoding {
                            if let Some(len) = fixed_len {
                                if *len == v.len() {
                                    scratch.data.extend_from_slice(v);
                                    Ok(1u8)
                                } else {
                                    write_varint(v.len() as u64, &mut scratch.data);
                                    scratch.data.extend_from_slice(v);
                                    Ok(2u8)
                                }
                            } else {
                                write_varint(v.len() as u64, &mut scratch.data);
                                scratch.data.extend_from_slice(v);
                                Ok(1u8)
                            }
                        } else {
                            Err(anyhow!("rowpack bytes encoding mismatch"))
                        }
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
                IngestedType::Uuid => match access.get_uuid(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if !matches!(col.encoding, ColumnEncoding::Uuid) {
                            return Err(anyhow!("rowpack uuid encoding mismatch"));
                        }
                        scratch.data.extend_from_slice(&v);
                        1u8
                    }
                },
                IngestedType::Uuid256 => match access.get_uuid256(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if !matches!(col.encoding, ColumnEncoding::Uuid256) {
                            return Err(anyhow!("rowpack uuid256 encoding mismatch"));
                        }
                        scratch.data.extend_from_slice(&v);
                        1u8
                    }
                },
                IngestedType::Ip => match access.get_ip(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if !matches!(col.encoding, ColumnEncoding::Ip) {
                            return Err(anyhow!("rowpack ip encoding mismatch"));
                        }
                        scratch.data.extend_from_slice(&v);
                        1u8
                    }
                },
                IngestedType::Date => match access.get_date(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Date { varint, bits } = &col.encoding {
                            encode_int(&mut scratch.data, true, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack date encoding mismatch"));
                        }
                    }
                },
                IngestedType::Timestamp => match access.get_timestamp(idx).ok().flatten() {
                    None => 0u8,
                    Some((secs, nanos)) => {
                        write_varint(zigzag_encode(secs), &mut scratch.data);
                        write_varint(nanos as u64, &mut scratch.data);
                        1u8
                    }
                },
                IngestedType::Decimal128 => match access.get_decimal128(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Decimal {
                            bytes,
                            scale,
                            storage,
                            ..
                        } = &col.encoding
                        {
                            if *bytes == 0 || *bytes > 16 {
                                return Err(anyhow!("rowpack decimal size mismatch"));
                            }
                            encode_decimal_i128_with_tag(
                                &mut scratch.data,
                                v,
                                *scale,
                                storage,
                                *bytes,
                            )?
                        } else {
                            return Err(anyhow!("rowpack decimal128 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Decimal256 => match access.get_decimal256(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Decimal {
                            bytes,
                            scale,
                            storage,
                            ..
                        } = &col.encoding
                        {
                            if *bytes == 0 || *bytes > 32 {
                                return Err(anyhow!("rowpack decimal size mismatch"));
                            }
                            encode_decimal_i256_with_tag(
                                &mut scratch.data,
                                &v,
                                *scale,
                                storage,
                                *bytes,
                            )?
                        } else {
                            return Err(anyhow!("rowpack decimal256 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float32Array => {
                    match access.with_float32_array(idx, |values| {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_array_f32_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                values,
                            )
                        } else {
                            Err(anyhow!("rowpack float32 array encoding mismatch"))
                        }
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
                IngestedType::Float16Array => {
                    match access.with_float32_array(idx, |values| {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_array_f32_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                values,
                            )
                        } else {
                            Err(anyhow!("rowpack float16 array encoding mismatch"))
                        }
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
                IngestedType::Float64Array => {
                    match access.with_float64_array(idx, |values| {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_array_f64_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                values,
                            )
                        } else {
                            Err(anyhow!("rowpack float64 array encoding mismatch"))
                        }
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
            };
            scratch.tags.write_bits(tag, 2);
        }
        out.clear();
        out.reserve(scratch.tags.buf.len() + scratch.bools.buf.len() + scratch.data.len() + 8);
        scratch.tags.append_to(out);
        scratch.bools.append_to(out);
        out.extend_from_slice(&scratch.data);
        Ok(())
    }

    pub fn encode_row_from_access_into_buffers(
        &self,
        access: &mut RowAccess<'_>,
        tags: &mut Vec<u8>,
        bools: &mut Vec<u8>,
        data: &mut Vec<u8>,
        key_bufs: &[Vec<u8>],
    ) -> Result<RowPackRowLens> {
        debug_assert_eq!(key_bufs.len(), self.ctx.columns.len());
        let tag_start = tags.len();
        let bool_start = bools.len();
        let data_start = data.len();
        let mut tag_writer = BitWriterRef::new(tags);
        let mut bool_writer = BitWriterRef::new(bools);

        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = match col.logical_type {
                IngestedType::Bool => match access.get_bool(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        bool_writer.write_bits(if v { 1 } else { 0 }, 1);
                        1u8
                    }
                },
                IngestedType::Int64 => match access.get_int64(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_int(data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int64 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Int16 => match access.get_int32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_int(data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int16 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Int8 => match access.get_int32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_int(data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int8 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Int32 => match access.get_int32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_int(data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int32 encoding mismatch"));
                        }
                    }
                },
                IngestedType::UInt64 => match access.get_uint64(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_uint(data, *signed, *varint, *bits, v)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint64 encoding mismatch"));
                        }
                    }
                },
                IngestedType::UInt16 => match access.get_uint32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_uint(data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint16 encoding mismatch"));
                        }
                    }
                },
                IngestedType::UInt8 => match access.get_uint32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_uint(data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint8 encoding mismatch"));
                        }
                    }
                },
                IngestedType::UInt32 => match access.get_uint32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } = &col.encoding
                        {
                            encode_uint(data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint32 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float64 => match access.get_float64(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Float { storage } = &col.encoding {
                            encode_float_with_tag(data, col.logical_type, *storage, v)?
                        } else {
                            return Err(anyhow!("rowpack float64 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float32 => match access.get_float32(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Float { storage } = &col.encoding {
                            encode_float_with_tag(data, col.logical_type, *storage, v as f64)?
                        } else {
                            return Err(anyhow!("rowpack float32 encoding mismatch"));
                        }
                    }
                },
                IngestedType::String => {
                    match access.with_string(idx, |v| {
                        encode_string(
                            data,
                            v,
                            col,
                            self.dict_maps[idx].as_ref(),
                            self.alphabet_maps[idx].as_ref(),
                            self.bpe_tries[idx].as_ref(),
                            None,
                            None,
                        )
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
                IngestedType::Bytes => {
                    match access.with_bytes(idx, |v| {
                        if let ColumnEncoding::Bytes { fixed_len } = &col.encoding {
                            if let Some(len) = fixed_len {
                                if *len == v.len() {
                                    data.extend_from_slice(v);
                                    Ok(1u8)
                                } else {
                                    write_varint(v.len() as u64, data);
                                    data.extend_from_slice(v);
                                    Ok(2u8)
                                }
                            } else {
                                write_varint(v.len() as u64, data);
                                data.extend_from_slice(v);
                                Ok(1u8)
                            }
                        } else {
                            Err(anyhow!("rowpack bytes encoding mismatch"))
                        }
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
                IngestedType::Uuid => {
                    let buf = &key_bufs[idx];
                    if buf.len() == 16 {
                        data.extend_from_slice(buf);
                        1u8
                    } else {
                        match access.get_uuid(idx).ok().flatten() {
                            None => 0u8,
                            Some(v) => {
                                data.extend_from_slice(&v);
                                1u8
                            }
                        }
                    }
                }
                IngestedType::Uuid256 => {
                    let buf = &key_bufs[idx];
                    if buf.len() == 32 {
                        data.extend_from_slice(buf);
                        1u8
                    } else {
                        match access.get_uuid256(idx).ok().flatten() {
                            None => 0u8,
                            Some(v) => {
                                data.extend_from_slice(&v);
                                1u8
                            }
                        }
                    }
                }
                IngestedType::Ip => {
                    let buf = &key_bufs[idx];
                    if buf.len() == 16 {
                        data.extend_from_slice(buf);
                        1u8
                    } else {
                        match access.get_ip(idx).ok().flatten() {
                            None => 0u8,
                            Some(v) => {
                                data.extend_from_slice(&v);
                                1u8
                            }
                        }
                    }
                }
                IngestedType::Date => match access.get_date(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Date { varint, bits } = &col.encoding {
                            encode_int(data, true, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack date encoding mismatch"));
                        }
                    }
                },
                IngestedType::Timestamp => match access.get_timestamp(idx).ok().flatten() {
                    None => 0u8,
                    Some((secs, nanos)) => {
                        write_varint(zigzag_encode(secs), data);
                        write_varint(nanos as u64, data);
                        1u8
                    }
                },
                IngestedType::Decimal128 => match access.get_decimal128(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Decimal {
                            bytes,
                            scale,
                            storage,
                            ..
                        } = &col.encoding
                        {
                            if *bytes == 0 || *bytes > 16 {
                                return Err(anyhow!("rowpack decimal size mismatch"));
                            }
                            encode_decimal_i128_with_tag(data, v, *scale, storage, *bytes)?
                        } else {
                            return Err(anyhow!("rowpack decimal128 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Decimal256 => match access.get_decimal256(idx).ok().flatten() {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Decimal {
                            bytes,
                            scale,
                            storage,
                            ..
                        } = &col.encoding
                        {
                            if *bytes == 0 || *bytes > 32 {
                                return Err(anyhow!("rowpack decimal size mismatch"));
                            }
                            encode_decimal_i256_with_tag(data, &v, *scale, storage, *bytes)?
                        } else {
                            return Err(anyhow!("rowpack decimal256 encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float32Array => {
                    match access.with_float32_array(idx, |values| {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_array_f32_with_tag(
                                data,
                                col.logical_type,
                                *storage,
                                values,
                            )
                        } else {
                            Err(anyhow!("rowpack float32 array encoding mismatch"))
                        }
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
                IngestedType::Float16Array => {
                    match access.with_float32_array(idx, |values| {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_array_f32_with_tag(
                                data,
                                col.logical_type,
                                *storage,
                                values,
                            )
                        } else {
                            Err(anyhow!("rowpack float16 array encoding mismatch"))
                        }
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
                IngestedType::Float64Array => {
                    match access.with_float64_array(idx, |values| {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_array_f64_with_tag(
                                data,
                                col.logical_type,
                                *storage,
                                values,
                            )
                        } else {
                            Err(anyhow!("rowpack float64 array encoding mismatch"))
                        }
                    })? {
                        None => 0u8,
                        Some(tag) => tag,
                    }
                }
            };
            tag_writer.write_bits(tag, 2);
        }
        tag_writer.finish();
        bool_writer.finish();
        Ok(RowPackRowLens {
            tag_bytes: tags.len() - tag_start,
            bool_bytes: bools.len() - bool_start,
            data_bytes: data.len() - data_start,
        })
    }
}

fn encode_string(
    out: &mut Vec<u8>,
    value: &str,
    col: &ColumnContext,
    _dict: Option<&FastHashMap<Vec<u8>, u32>>,
    _alphabet: Option<&AlphabetMap>,
    _bpe: Option<&BpeTrie>,
    _bpe_tokens: Option<&mut Vec<u32>>,
    _lz4_buf: Option<&mut Vec<u8>>,
) -> Result<u8> {
    let ColumnEncoding::String { encoding } = &col.encoding else {
        return Err(anyhow!("rowpack string encoding mismatch"));
    };
    match encoding {
        StringEncoding::Raw => encode_string_raw(out, value),
        StringEncoding::Dict { .. } => encode_string_dict(out, value, _dict),
        StringEncoding::Alphabet { .. } => encode_string_alphabet(out, value, _alphabet),
        StringEncoding::Bpe { .. } => encode_string_bpe(out, value, _bpe, _bpe_tokens),
        StringEncoding::Lz4 => encode_string_lz4(out, value, _lz4_buf),
    }
}

fn encode_string_raw(out: &mut Vec<u8>, value: &str) -> Result<u8> {
    let bytes = value.as_bytes();
    write_varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
    Ok(1)
}

fn encode_string_dict(
    out: &mut Vec<u8>,
    value: &str,
    dict: Option<&FastHashMap<Vec<u8>, u32>>,
) -> Result<u8> {
    if let Some(map) = dict {
        if let Some(idx) = map.get(value.as_bytes()) {
            write_varint(*idx as u64, out);
            Ok(1)
        } else {
            let bytes = value.as_bytes();
            write_varint(bytes.len() as u64, out);
            out.extend_from_slice(bytes);
            Ok(2)
        }
    } else {
        Err(anyhow!("rowpack missing dict"))
    }
}

fn encode_string_alphabet(
    out: &mut Vec<u8>,
    value: &str,
    alphabet: Option<&AlphabetMap>,
) -> Result<u8> {
    let bytes = value.as_bytes();
    let map = alphabet.ok_or_else(|| anyhow!("rowpack missing alphabet map"))?;
    write_varint(bytes.len() as u64, out);
    let packed_len = ((bytes.len() * map.bits as usize) + 7) / 8;
    out.reserve(packed_len);
    let payload_start = out.len();
    let packed_ok = match map.packer {
        AlphabetPacker::Bits1 => pack_bits_1(out, bytes, &map.map, payload_start),
        AlphabetPacker::Bits2 => pack_bits_2(out, bytes, &map.map, payload_start),
        AlphabetPacker::Bits3 => pack_bits_3(out, bytes, &map.map, payload_start),
        AlphabetPacker::Bits4 => pack_bits_4(out, bytes, &map.map, payload_start),
        AlphabetPacker::Bits5 => pack_bits_5(out, bytes, &map.map, payload_start),
        AlphabetPacker::Bits6 => pack_bits_6(out, bytes, &map.map, payload_start),
        AlphabetPacker::Generic { bits } => {
            pack_bits_generic(out, bytes, &map.map, bits, payload_start)
        }
    };
    if packed_ok {
        Ok(1)
    } else {
        out.extend_from_slice(bytes);
        Ok(2)
    }
}

fn encode_string_bpe(
    out: &mut Vec<u8>,
    value: &str,
    bpe: Option<&BpeTrie>,
    mut bpe_tokens: Option<&mut Vec<u32>>,
) -> Result<u8> {
    let bytes = value.as_bytes();
    let trie = bpe.ok_or_else(|| anyhow!("rowpack missing bpe trie"))?;
    let mut local_tokens = Vec::new();
    let tokens = match bpe_tokens.as_deref_mut() {
        Some(buf) => {
            buf.clear();
            buf
        }
        None => &mut local_tokens,
    };
    tokenize_bpe(bytes, trie, tokens);
    write_varint(tokens.len() as u64, out);
    let bits = bpe_token_bits(trie.dict_len);
    write_tokens_packed(out, tokens, bits);
    Ok(1)
}

fn encode_string_lz4(
    out: &mut Vec<u8>,
    value: &str,
    mut lz4_buf: Option<&mut Vec<u8>>,
) -> Result<u8> {
    let bytes = value.as_bytes();
    let raw_len = bytes.len();
    let mut local_buf = Vec::new();
    let buf = match lz4_buf.as_deref_mut() {
        Some(buf) => {
            buf.clear();
            buf
        }
        None => &mut local_buf,
    };
    let max = lz4_flex::block::get_maximum_output_size(raw_len);
    if buf.capacity() < max {
        buf.reserve(max - buf.capacity());
    }
    buf.resize(max, 0);
    let written = lz4_flex::block::compress_into(bytes, buf)?;
    let compressed = &buf[..written];
    let raw_total = varint_bytes_u64(raw_len as u64) + raw_len;
    let compressed_total = varint_bytes_u64(raw_len as u64)
        + varint_bytes_u64(compressed.len() as u64)
        + compressed.len();
    if compressed_total < raw_total {
        write_varint(raw_len as u64, out);
        write_varint(compressed.len() as u64, out);
        out.extend_from_slice(compressed);
        Ok(1)
    } else {
        write_varint(raw_len as u64, out);
        out.extend_from_slice(bytes);
        Ok(2)
    }
}

fn encode_value_with_tag<'a>(
    out: &mut Vec<u8>,
    col: &ColumnContext,
    value: &ValueRef<'a>,
    _dict: Option<&FastHashMap<Vec<u8>, u32>>,
    _alphabet: Option<&AlphabetMap>,
    _bpe: Option<&BpeTrie>,
    _bpe_tokens: Option<&mut Vec<u32>>,
    _lz4_buf: Option<&mut Vec<u8>>,
) -> Result<u8> {
    match (&col.encoding, value) {
        (
            ColumnEncoding::Int {
                signed,
                varint,
                bits,
            },
            ValueRef::Int64(v),
        ) => {
            encode_int(out, *signed, *varint, *bits, *v as i128)?;
            Ok(1)
        }
        (
            ColumnEncoding::Int {
                signed,
                varint,
                bits,
            },
            ValueRef::Int32(v),
        ) => {
            encode_int(out, *signed, *varint, *bits, *v as i128)?;
            Ok(1)
        }
        (
            ColumnEncoding::Int {
                signed,
                varint,
                bits,
            },
            ValueRef::UInt64(v),
        ) => {
            encode_uint(out, *signed, *varint, *bits, *v)?;
            Ok(1)
        }
        (
            ColumnEncoding::Int {
                signed,
                varint,
                bits,
            },
            ValueRef::UInt32(v),
        ) => {
            encode_uint(out, *signed, *varint, *bits, *v as u64)?;
            Ok(1)
        }
        (ColumnEncoding::Float { storage }, ValueRef::Float64(v)) => {
            let tag = encode_float_with_tag(out, col.logical_type, *storage, *v)?;
            Ok(tag)
        }
        (ColumnEncoding::Float { storage }, ValueRef::Float32(v)) => {
            let tag = encode_float_with_tag(out, col.logical_type, *storage, *v as f64)?;
            Ok(tag)
        }
        (ColumnEncoding::Bytes { fixed_len }, ValueRef::Bytes(v)) => {
            if let Some(len) = fixed_len {
                if *len == v.len() {
                    out.extend_from_slice(v);
                    return Ok(1);
                }
                write_varint(v.len() as u64, out);
                out.extend_from_slice(v);
                Ok(2)
            } else {
                write_varint(v.len() as u64, out);
                out.extend_from_slice(v);
                Ok(1)
            }
        }
        (ColumnEncoding::Uuid, ValueRef::Uuid(v)) => {
            out.extend_from_slice(v);
            Ok(1)
        }
        (ColumnEncoding::Uuid256, ValueRef::Uuid256(v)) => {
            out.extend_from_slice(v);
            Ok(1)
        }
        (ColumnEncoding::Ip, ValueRef::Ip(v)) => {
            out.extend_from_slice(v);
            Ok(1)
        }
        (ColumnEncoding::Date { varint, bits }, ValueRef::Date(v)) => {
            encode_int(out, true, *varint, *bits, *v as i128)?;
            Ok(1)
        }
        (ColumnEncoding::Timestamp, ValueRef::Timestamp { secs, nanos }) => {
            write_varint(zigzag_encode(*secs), out);
            write_varint(*nanos as u64, out);
            Ok(1)
        }
        (
            ColumnEncoding::Decimal {
                bytes,
                storage,
                scale,
                ..
            },
            ValueRef::Decimal128(v),
        ) => {
            if *bytes == 0 || *bytes > 16 {
                return Err(anyhow!("rowpack decimal size mismatch"));
            }
            encode_decimal_i128_with_tag(out, *v, *scale, storage, *bytes)
        }
        (
            ColumnEncoding::Decimal {
                bytes,
                storage,
                scale,
                ..
            },
            ValueRef::Decimal256(v),
        ) => {
            if *bytes == 0 || *bytes > 32 {
                return Err(anyhow!("rowpack decimal size mismatch"));
            }
            encode_decimal_i256_with_tag(out, v, *scale, storage, *bytes)
        }
        (ColumnEncoding::String { .. }, ValueRef::String(_)) => {
            Err(anyhow!("rowpack string encoding mismatch"))
        }
        (ColumnEncoding::FloatArray { storage }, ValueRef::Float32Array(values)) => {
            let tag = encode_float_array_f32_with_tag(out, col.logical_type, *storage, *values)?;
            Ok(tag)
        }
        (ColumnEncoding::FloatArray { storage }, ValueRef::Float64Array(values)) => {
            let tag = encode_float_array_f64_with_tag(out, col.logical_type, *storage, *values)?;
            Ok(tag)
        }
        _ => Err(anyhow!("rowpack value type mismatch")),
    }
}

fn encode_float_with_tag(
    out: &mut Vec<u8>,
    logical: IngestedType,
    storage: FloatStorage,
    v: f64,
) -> Result<u8> {
    let _ = logical;
    encode_float(out, storage, v)?;
    Ok(1)
}

fn encode_decimal_i128_with_tag(
    out: &mut Vec<u8>,
    value: i128,
    scale: i32,
    storage: &DecimalStorage,
    raw_bytes: u8,
) -> Result<u8> {
    match storage {
        DecimalStorage::Raw => {
            let bytes = value.to_le_bytes();
            out.extend_from_slice(&bytes[..raw_bytes as usize]);
            Ok(1)
        }
        DecimalStorage::Varint => {
            let _ = scale;
            let zigzag = zigzag_encode_i128(value);
            let varint_len = varint_bytes_u128(zigzag);
            if varint_len < raw_bytes as usize {
                write_varint_u128(zigzag, out);
                Ok(1)
            } else {
                let bytes = value.to_le_bytes();
                out.extend_from_slice(&bytes[..raw_bytes as usize]);
                Ok(2)
            }
        }
    }
}

fn encode_decimal_i256_with_tag(
    out: &mut Vec<u8>,
    value: &[u8; 32],
    scale: i32,
    storage: &DecimalStorage,
    raw_bytes: u8,
) -> Result<u8> {
    match storage {
        DecimalStorage::Raw => {
            out.extend_from_slice(&value[..raw_bytes as usize]);
            Ok(1)
        }
        DecimalStorage::Varint => {
            let _ = scale;
            out.extend_from_slice(&value[..raw_bytes as usize]);
            Ok(2)
        }
    }
}

fn encode_float_array_f32_with_tag(
    out: &mut Vec<u8>,
    logical: IngestedType,
    storage: FloatStorage,
    values: &[f32],
) -> Result<u8> {
    let _ = logical;
    write_varint(values.len() as u64, out);
    for v in values {
        encode_float(out, storage, *v as f64)?;
    }
    Ok(1)
}

fn encode_float_array_f64_with_tag(
    out: &mut Vec<u8>,
    logical: IngestedType,
    storage: FloatStorage,
    values: &[f64],
) -> Result<u8> {
    let _ = logical;
    write_varint(values.len() as u64, out);
    for v in values {
        encode_float(out, storage, *v)?;
    }
    Ok(1)
}

fn encode_int(out: &mut Vec<u8>, signed: bool, varint: bool, bits: u8, v: i128) -> Result<()> {
    if signed {
        if varint {
            write_varint(zigzag_encode(v as i64), out);
        } else if bits <= 8 {
            out.extend_from_slice(&(v as i8).to_le_bytes());
        } else if bits <= 16 {
            out.extend_from_slice(&(v as i16).to_le_bytes());
        } else if bits <= 32 {
            out.extend_from_slice(&(v as i32).to_le_bytes());
        } else {
            out.extend_from_slice(&(v as i64).to_le_bytes());
        }
    } else {
        encode_uint(out, false, varint, bits, v as u64)?;
    }
    Ok(())
}

fn encode_uint(out: &mut Vec<u8>, _signed: bool, varint: bool, bits: u8, v: u64) -> Result<()> {
    if varint {
        write_varint(v, out);
    } else if bits <= 8 {
        out.extend_from_slice(&(v as u8).to_le_bytes());
    } else if bits <= 16 {
        out.extend_from_slice(&(v as u16).to_le_bytes());
    } else if bits <= 32 {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    } else {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

fn encode_float(out: &mut Vec<u8>, storage: FloatStorage, v: f64) -> Result<()> {
    match storage {
        FloatStorage::F16 => {
            let half = f16::from_f32(v as f32);
            out.extend_from_slice(&half.to_bits().to_le_bytes());
        }
        FloatStorage::F32 => {
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
        FloatStorage::F64 => {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(())
}

fn alphabet_bits(len: usize) -> u8 {
    if len <= 1 {
        1
    } else {
        (usize::BITS - (len - 1).leading_zeros()) as u8
    }
}

fn bpe_token_bits(dict_len: usize) -> u8 {
    let space = 256usize.saturating_add(dict_len);
    alphabet_bits(space.max(1))
}

#[inline]
fn alphabet_index(map: &[u8; 256], b: u8) -> Option<u8> {
    let idx = map[b as usize];
    if idx == ALPHABET_INVALID {
        None
    } else {
        Some(idx)
    }
}

fn pack_bits_generic(
    out: &mut Vec<u8>,
    bytes: &[u8],
    map: &[u8; 256],
    bits: u8,
    payload_start: usize,
) -> bool {
    let mut cur: u64 = 0;
    let mut used: u8 = 0;
    for b in bytes {
        let idx = match alphabet_index(map, *b) {
            Some(idx) => idx as u64,
            None => {
                out.truncate(payload_start);
                return false;
            }
        };
        cur |= idx << used;
        used = used.saturating_add(bits);
        while used >= 8 {
            out.push(cur as u8);
            cur >>= 8;
            used -= 8;
        }
    }
    if used > 0 {
        out.push(cur as u8);
    }
    true
}

fn pack_bits_1(out: &mut Vec<u8>, bytes: &[u8], map: &[u8; 256], payload_start: usize) -> bool {
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let mut cur: u8 = 0;
        for bit in 0..8 {
            let idx = match alphabet_index(map, bytes[i + bit]) {
                Some(idx) => idx,
                None => {
                    out.truncate(payload_start);
                    return false;
                }
            };
            cur |= (idx & 1) << bit;
        }
        out.push(cur);
        i += 8;
    }
    if i < bytes.len() {
        return pack_bits_generic(out, &bytes[i..], map, 1, payload_start);
    }
    true
}

fn pack_bits_2(out: &mut Vec<u8>, bytes: &[u8], map: &[u8; 256], payload_start: usize) -> bool {
    let mut i = 0usize;
    while i + 4 <= bytes.len() {
        let b0 = match alphabet_index(map, bytes[i]) {
            Some(idx) => idx,
            None => {
                out.truncate(payload_start);
                return false;
            }
        };
        let b1 = match alphabet_index(map, bytes[i + 1]) {
            Some(idx) => idx,
            None => {
                out.truncate(payload_start);
                return false;
            }
        };
        let b2 = match alphabet_index(map, bytes[i + 2]) {
            Some(idx) => idx,
            None => {
                out.truncate(payload_start);
                return false;
            }
        };
        let b3 = match alphabet_index(map, bytes[i + 3]) {
            Some(idx) => idx,
            None => {
                out.truncate(payload_start);
                return false;
            }
        };
        let cur = (b0 & 0x3) | ((b1 & 0x3) << 2) | ((b2 & 0x3) << 4) | ((b3 & 0x3) << 6);
        out.push(cur);
        i += 4;
    }
    if i < bytes.len() {
        return pack_bits_generic(out, &bytes[i..], map, 2, payload_start);
    }
    true
}

fn pack_bits_3(out: &mut Vec<u8>, bytes: &[u8], map: &[u8; 256], payload_start: usize) -> bool {
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let mut cur: u32 = 0;
        for offset in 0..8 {
            let idx = match alphabet_index(map, bytes[i + offset]) {
                Some(idx) => idx as u32,
                None => {
                    out.truncate(payload_start);
                    return false;
                }
            };
            cur |= (idx & 0x7) << (offset * 3);
        }
        out.push(cur as u8);
        out.push((cur >> 8) as u8);
        out.push((cur >> 16) as u8);
        i += 8;
    }
    if i < bytes.len() {
        return pack_bits_generic(out, &bytes[i..], map, 3, payload_start);
    }
    true
}

fn pack_bits_4(out: &mut Vec<u8>, bytes: &[u8], map: &[u8; 256], payload_start: usize) -> bool {
    let mut i = 0usize;
    while i + 2 <= bytes.len() {
        let b0 = match alphabet_index(map, bytes[i]) {
            Some(idx) => idx,
            None => {
                out.truncate(payload_start);
                return false;
            }
        };
        let b1 = match alphabet_index(map, bytes[i + 1]) {
            Some(idx) => idx,
            None => {
                out.truncate(payload_start);
                return false;
            }
        };
        let cur = (b0 & 0x0f) | ((b1 & 0x0f) << 4);
        out.push(cur);
        i += 2;
    }
    if i < bytes.len() {
        return pack_bits_generic(out, &bytes[i..], map, 4, payload_start);
    }
    true
}

fn pack_bits_5(out: &mut Vec<u8>, bytes: &[u8], map: &[u8; 256], payload_start: usize) -> bool {
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let mut cur: u64 = 0;
        for offset in 0..8 {
            let idx = match alphabet_index(map, bytes[i + offset]) {
                Some(idx) => idx as u64,
                None => {
                    out.truncate(payload_start);
                    return false;
                }
            };
            cur |= (idx & 0x1f) << (offset * 5);
        }
        out.push(cur as u8);
        out.push((cur >> 8) as u8);
        out.push((cur >> 16) as u8);
        out.push((cur >> 24) as u8);
        out.push((cur >> 32) as u8);
        i += 8;
    }
    if i < bytes.len() {
        return pack_bits_generic(out, &bytes[i..], map, 5, payload_start);
    }
    true
}

fn pack_bits_6(out: &mut Vec<u8>, bytes: &[u8], map: &[u8; 256], payload_start: usize) -> bool {
    let mut i = 0usize;
    while i + 4 <= bytes.len() {
        let mut cur: u32 = 0;
        for offset in 0..4 {
            let idx = match alphabet_index(map, bytes[i + offset]) {
                Some(idx) => idx as u32,
                None => {
                    out.truncate(payload_start);
                    return false;
                }
            };
            cur |= (idx & 0x3f) << (offset * 6);
        }
        out.push(cur as u8);
        out.push((cur >> 8) as u8);
        out.push((cur >> 16) as u8);
        i += 4;
    }
    if i < bytes.len() {
        return pack_bits_generic(out, &bytes[i..], map, 6, payload_start);
    }
    true
}

fn write_tokens_packed(out: &mut Vec<u8>, tokens: &[u32], bits: u8) {
    if bits == 0 {
        return;
    }
    let mut cur: u64 = 0;
    let mut used: u8 = 0;
    for token in tokens.iter().copied() {
        cur |= (token as u64) << used;
        used = used.saturating_add(bits);
        while used >= 8 {
            out.push(cur as u8);
            cur >>= 8;
            used -= 8;
        }
    }
    if used > 0 {
        out.push(cur as u8);
    }
}

fn tokenize_bpe(bytes: &[u8], trie: &BpeTrie, tokens: &mut Vec<u32>) {
    tokens.clear();
    let mut i = 0usize;
    while i < bytes.len() {
        let mut node = 0usize;
        let mut j = i;
        let mut last = None;
        while j < bytes.len() {
            let next_idx = trie.next_index(node, bytes[j]);
            if next_idx < 0 {
                break;
            }
            node = next_idx as usize;
            if let Some(token) = trie.value[node] {
                last = Some((token, j + 1));
            }
            j += 1;
        }
        if let Some((token, end)) = last {
            tokens.push(token as u32);
            i = end;
        } else {
            tokens.push(bytes[i] as u32);
            i += 1;
        }
    }
}

fn estimate_raw_bytes(samples: &[Vec<u8>]) -> u64 {
    let mut total = 0u64;
    for sample in samples {
        total += varint_bytes_u64(sample.len() as u64) as u64;
        total += sample.len() as u64;
    }
    total
}

fn estimate_alphabet_bytes(samples: &[Vec<u8>], alphabet_len: usize) -> u64 {
    let bits = alphabet_bits(alphabet_len) as usize;
    let mut total = 0u64;
    for sample in samples {
        total += varint_bytes_u64(sample.len() as u64) as u64;
        let packed = ((sample.len() * bits) + 7) / 8;
        total += packed as u64;
    }
    total
}

fn estimate_bpe_bytes(samples: &[Vec<u8>], dict: &[Vec<u8>]) -> u64 {
    if dict.is_empty() {
        return u64::MAX;
    }
    let trie = BpeTrie::from_dict(dict);
    let bits = bpe_token_bits(dict.len()) as usize;
    let mut tokens = Vec::new();
    let mut total = 0u64;
    for sample in samples {
        tokenize_bpe(sample, &trie, &mut tokens);
        total += varint_bytes_u64(tokens.len() as u64) as u64;
        let packed = ((tokens.len() * bits) + 7) / 8;
        total += packed as u64;
    }
    total
}

pub struct RowPackDecoder {
    ctx: RowPackContext,
    dict_values: Vec<Option<Vec<Vec<u8>>>>,
}

#[derive(Default)]
pub struct RowPackDecodeScratch {
    string: String,
    bytes: Vec<u8>,
    float32: Vec<f32>,
    float64: Vec<f64>,
}

impl RowPackDecoder {
    pub fn new(ctx: RowPackContext) -> Result<Self> {
        let mut dict_values = Vec::with_capacity(ctx.columns.len());
        for col in &ctx.columns {
            match &col.encoding {
                ColumnEncoding::String {
                    encoding: StringEncoding::Dict { values },
                } => dict_values.push(Some(values.clone())),
                _ => dict_values.push(None),
            }
        }
        Ok(Self { ctx, dict_values })
    }

    pub fn context(&self) -> &RowPackContext {
        &self.ctx
    }

    pub fn decode_msgpack(&self, row: &[u8]) -> Result<Vec<u8>> {
        let mut reader = BitReader::new(row);
        let col_count = self.ctx.columns.len();
        let mut tags = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            tags.push(reader.read_bits(2)?);
        }
        reader.align_to_byte();
        let mut bools = Vec::new();
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            if matches!(col.encoding, ColumnEncoding::Bool) && tags[idx] != 0 {
                bools.push(reader.read_bits(1)?);
            }
        }
        reader.align_to_byte();
        let mut data = &row[row.len() - reader.remaining_bytes()..];
        let mut out = Vec::new();
        mp::write_map_len(&mut out, col_count as u32)?;
        let mut bool_idx = 0usize;
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            mp::write_str(&mut out, &col.name)?;
            let tag = tags[idx];
            if tag > 2 {
                return Err(anyhow!("rowpack invalid tag {}", tag));
            }
            if tag == 0 {
                mp::write_nil(&mut out)?;
                continue;
            }
            if matches!(col.encoding, ColumnEncoding::Bool) {
                let v = bools.get(bool_idx).copied().unwrap_or(0) != 0;
                bool_idx += 1;
                mp::write_bool(&mut out, v)?;
                continue;
            }
            decode_value_msgpack(
                &mut out,
                col,
                tag,
                &mut data,
                self.dict_values[idx].as_ref(),
            )?;
        }
        Ok(out)
    }

    pub fn decode_json(&self, row: &[u8]) -> Result<JsonValue> {
        let mut reader = BitReader::new(row);
        let col_count = self.ctx.columns.len();
        let mut tags = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            tags.push(reader.read_bits(2)?);
        }
        reader.align_to_byte();
        let mut bools = Vec::new();
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            if matches!(col.encoding, ColumnEncoding::Bool) && tags[idx] != 0 {
                bools.push(reader.read_bits(1)?);
            }
        }
        reader.align_to_byte();
        let mut data = &row[row.len() - reader.remaining_bytes()..];
        let mut map = serde_json::Map::new();
        let mut bool_idx = 0usize;
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = tags[idx];
            if tag > 2 {
                return Err(anyhow!("rowpack invalid tag {}", tag));
            }
            let value = if tag == 0 {
                JsonValue::Null
            } else if matches!(col.encoding, ColumnEncoding::Bool) {
                let v = bools.get(bool_idx).copied().unwrap_or(0) != 0;
                bool_idx += 1;
                JsonValue::Bool(v)
            } else {
                decode_value_json(col, tag, &mut data, self.dict_values[idx].as_ref())?
            };
            map.insert(col.name.clone(), value);
        }
        Ok(JsonValue::Object(map))
    }

    pub fn decode_column_value<'a>(
        &self,
        row: &'a [u8],
        col_idx: usize,
        scratch: &'a mut RowPackDecodeScratch,
    ) -> Result<ValueRef<'a>> {
        let mut reader = BitReader::new(row);
        let col_count = self.ctx.columns.len();
        if col_idx >= col_count {
            return Err(anyhow!("rowpack column index out of range"));
        }
        let mut tags = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            tags.push(reader.read_bits(2)?);
        }
        reader.align_to_byte();
        let mut bools = Vec::new();
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            if matches!(col.encoding, ColumnEncoding::Bool) && tags[idx] != 0 {
                bools.push(reader.read_bits(1)?);
            }
        }
        reader.align_to_byte();
        let mut data = &row[row.len() - reader.remaining_bytes()..];
        let mut bool_idx = 0usize;
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = tags[idx];
            if tag > 2 {
                return Err(anyhow!("rowpack invalid tag {}", tag));
            }
            if tag == 0 {
                if idx == col_idx {
                    return Ok(ValueRef::Null);
                }
                continue;
            }
            if matches!(col.encoding, ColumnEncoding::Bool) {
                let v = bools.get(bool_idx).copied().unwrap_or(0) != 0;
                bool_idx += 1;
                if idx == col_idx {
                    return Ok(ValueRef::Bool(v));
                }
                continue;
            }
            if idx == col_idx {
                return decode_value_ref(
                    col,
                    tag,
                    &mut data,
                    self.dict_values[idx].as_ref(),
                    scratch,
                );
            }
            decode_value_skip(col, tag, &mut data, self.dict_values[idx].as_ref())?;
        }
        Err(anyhow!("rowpack column index out of range"))
    }
}

fn decode_value_msgpack(
    out: &mut Vec<u8>,
    col: &ColumnContext,
    tag: u8,
    data: &mut &[u8],
    dict: Option<&Vec<Vec<u8>>>,
) -> Result<()> {
    match &col.encoding {
        ColumnEncoding::Int {
            signed,
            varint,
            bits,
        } => {
            if *signed {
                let v = decode_int(data, *varint, *bits)?;
                mp::write_sint(out, v)?;
            } else {
                let v = decode_uint(data, *varint, *bits)?;
                mp::write_uint(out, v)?;
            }
        }
        ColumnEncoding::Float { storage } => {
            let v = decode_float(data, *storage)?;
            match col.logical_type {
                IngestedType::Float32 => mp::write_f32(out, v as f32)?,
                _ => mp::write_f64(out, v)?,
            }
        }
        ColumnEncoding::Bytes { fixed_len } => {
            let bytes = if tag == 2 {
                decode_bytes(data, None)?
            } else {
                decode_bytes(data, *fixed_len)?
            };
            mp::write_bin(out, bytes)?;
        }
        ColumnEncoding::Uuid => {
            let bytes = decode_fixed_bytes::<16>(data)?;
            let mut buf = [0u8; 32];
            let encoded = format_simple(&bytes, buf.as_out(), AsciiCase::Lower);
            let text = std::str::from_utf8(encoded)
                .map_err(|_| anyhow!("rowpack uuid format invalid utf8"))?;
            mp::write_str(out, &text)?;
        }
        ColumnEncoding::Uuid256 => {
            let bytes = decode_fixed_bytes::<32>(data)?;
            let mut buf = [0u8; 64];
            // Safety: `bytes` and `buf` are fixed-size arrays, so the splits are in-bounds.
            let bytes_ptr = bytes.as_ptr();
            let hi = unsafe { &*(bytes_ptr as *const [u8; 16]) };
            let lo = unsafe { &*(bytes_ptr.add(16) as *const [u8; 16]) };
            let out_ptr = buf.as_mut_ptr();
            let hi_out = unsafe { &mut *(out_ptr as *mut [u8; 32]) };
            let lo_out = unsafe { &mut *(out_ptr.add(32) as *mut [u8; 32]) };
            let _ = format_simple(hi, hi_out.as_out(), AsciiCase::Lower);
            let _ = format_simple(lo, lo_out.as_out(), AsciiCase::Lower);
            let text = std::str::from_utf8(&buf)
                .map_err(|_| anyhow!("rowpack uuid256 format invalid utf8"))?;
            mp::write_str(out, &text)?;
        }
        ColumnEncoding::Ip => {
            let bytes = decode_fixed_bytes::<16>(data)?;
            let text = format_ip(bytes);
            mp::write_str(out, &text)?;
        }
        ColumnEncoding::Date { varint, bits } => {
            let v = decode_int(data, *varint, *bits)?;
            mp::write_sint(out, v)?;
        }
        ColumnEncoding::Timestamp => {
            let secs = zigzag_decode(read_varint(data)?);
            let nanos = read_varint(data)? as u32;
            write_msgpack_timestamp(out, secs, nanos)?;
        }
        ColumnEncoding::Decimal {
            scale,
            bytes,
            storage,
            ..
        } => {
            if tag == 2 || matches!(storage, DecimalStorage::Raw) {
                let raw = decode_decimal_bytes(data, *bytes as usize)?;
                let text = format_decimal_from_le(raw, *scale);
                mp::write_str(out, &text)?;
            } else {
                match storage {
                    DecimalStorage::Varint => {
                        let v = zigzag_decode_i128(read_varint_u128(data)?);
                        if *scale == 0 && v >= i64::MIN as i128 && v <= i64::MAX as i128 {
                            mp::write_sint(out, v as i64)?;
                        } else {
                            let text = format_decimal_i128(v, *scale);
                            mp::write_str(out, &text)?;
                        }
                    }
                    DecimalStorage::Raw => {}
                }
            }
        }
        ColumnEncoding::String { encoding } => {
            let value = if tag == 2 {
                decode_string_fallback(data)?
            } else {
                decode_string(data, encoding, dict)?
            };
            mp::write_str(out, &value)?;
        }
        ColumnEncoding::FloatArray { storage } => {
            let len = read_varint(data)? as usize;
            let elem_bytes = match storage {
                FloatStorage::F16 => 2,
                FloatStorage::F32 => 4,
                FloatStorage::F64 => 8,
            };
            let max_len = data.len() / elem_bytes;
            if len > MAX_ARRAY_LEN || len > max_len || len > u32::MAX as usize {
                return Err(anyhow!("rowpack array length overflow"));
            }
            mp::write_array_len(out, len as u32)?;
            for _ in 0..len {
                let v = decode_float(data, *storage)?;
                match col.logical_type {
                    IngestedType::Float16Array | IngestedType::Float32Array => {
                        mp::write_f32(out, v as f32)?
                    }
                    _ => mp::write_f64(out, v)?,
                }
            }
        }
        ColumnEncoding::Bool => unreachable!(),
    }
    Ok(())
}

fn decode_value_json(
    col: &ColumnContext,
    tag: u8,
    data: &mut &[u8],
    dict: Option<&Vec<Vec<u8>>>,
) -> Result<JsonValue> {
    let value = match &col.encoding {
        ColumnEncoding::Int {
            signed,
            varint,
            bits,
        } => {
            if *signed {
                JsonValue::from(decode_int(data, *varint, *bits)?)
            } else {
                JsonValue::from(decode_uint(data, *varint, *bits)?)
            }
        }
        ColumnEncoding::Float { storage } => {
            let v = decode_float(data, *storage)?;
            match col.logical_type {
                IngestedType::Float32 => JsonValue::from(v as f32),
                _ => JsonValue::from(v),
            }
        }
        ColumnEncoding::Bytes { fixed_len } => {
            let bytes = if tag == 2 {
                decode_bytes(data, None)?
            } else {
                decode_bytes(data, *fixed_len)?
            };
            JsonValue::from(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        ColumnEncoding::Uuid => {
            let bytes = decode_fixed_bytes::<16>(data)?;
            let mut buf = [0u8; 32];
            let encoded = format_simple(&bytes, buf.as_out(), AsciiCase::Lower);
            let text = std::str::from_utf8(encoded)
                .map_err(|_| anyhow!("rowpack uuid format invalid utf8"))?;
            JsonValue::from(text)
        }
        ColumnEncoding::Uuid256 => {
            let bytes = decode_fixed_bytes::<32>(data)?;
            let mut buf = [0u8; 64];
            // Safety: `bytes` and `buf` are fixed-size arrays, so the splits are in-bounds.
            let bytes_ptr = bytes.as_ptr();
            let hi = unsafe { &*(bytes_ptr as *const [u8; 16]) };
            let lo = unsafe { &*(bytes_ptr.add(16) as *const [u8; 16]) };
            let out_ptr = buf.as_mut_ptr();
            let hi_out = unsafe { &mut *(out_ptr as *mut [u8; 32]) };
            let lo_out = unsafe { &mut *(out_ptr.add(32) as *mut [u8; 32]) };
            let _ = format_simple(hi, hi_out.as_out(), AsciiCase::Lower);
            let _ = format_simple(lo, lo_out.as_out(), AsciiCase::Lower);
            let text = std::str::from_utf8(&buf)
                .map_err(|_| anyhow!("rowpack uuid256 format invalid utf8"))?;
            JsonValue::from(text)
        }
        ColumnEncoding::Ip => {
            let bytes = decode_fixed_bytes::<16>(data)?;
            JsonValue::from(format_ip(bytes))
        }
        ColumnEncoding::Date { varint, bits } => JsonValue::from(decode_int(data, *varint, *bits)?),
        ColumnEncoding::Timestamp => {
            let secs = zigzag_decode(read_varint(data)?);
            let nanos = read_varint(data)? as u32;
            let formatted = format_timestamp_default(secs, nanos);
            JsonValue::from(formatted)
        }
        ColumnEncoding::Decimal {
            scale,
            bytes,
            storage,
            ..
        } => {
            if tag == 2 || matches!(storage, DecimalStorage::Raw) {
                let raw = decode_decimal_bytes(data, *bytes as usize)?;
                JsonValue::from(format_decimal_from_le(raw, *scale))
            } else {
                match storage {
                    DecimalStorage::Varint => {
                        let v = zigzag_decode_i128(read_varint_u128(data)?);
                        if *scale == 0 && v >= i64::MIN as i128 && v <= i64::MAX as i128 {
                            JsonValue::from(v as i64)
                        } else {
                            JsonValue::from(format_decimal_i128(v, *scale))
                        }
                    }
                    DecimalStorage::Raw => JsonValue::Null,
                }
            }
        }
        ColumnEncoding::String { encoding } => {
            let value = if tag == 2 {
                decode_string_fallback(data)?
            } else {
                decode_string(data, encoding, dict)?
            };
            JsonValue::from(value)
        }
        ColumnEncoding::FloatArray { storage } => {
            let len = read_varint(data)? as usize;
            let elem_bytes = match storage {
                FloatStorage::F16 => 2,
                FloatStorage::F32 => 4,
                FloatStorage::F64 => 8,
            };
            let max_len = data.len() / elem_bytes;
            if len > MAX_ARRAY_LEN || len > max_len {
                return Err(anyhow!("rowpack array length overflow"));
            }
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let v = decode_float(data, *storage)?;
                match col.logical_type {
                    IngestedType::Float16Array | IngestedType::Float32Array => {
                        values.push(JsonValue::from(v as f32))
                    }
                    _ => values.push(JsonValue::from(v)),
                }
            }
            JsonValue::from(values)
        }
        ColumnEncoding::Bool => JsonValue::Null,
    };
    Ok(value)
}

fn decode_value_ref<'a>(
    col: &ColumnContext,
    tag: u8,
    data: &mut &'a [u8],
    dict: Option<&Vec<Vec<u8>>>,
    scratch: &'a mut RowPackDecodeScratch,
) -> Result<ValueRef<'a>> {
    match &col.encoding {
        ColumnEncoding::Int {
            signed,
            varint,
            bits,
        } => {
            if *signed {
                Ok(ValueRef::Int64(decode_int(data, *varint, *bits)?))
            } else {
                Ok(ValueRef::UInt64(decode_uint(data, *varint, *bits)?))
            }
        }
        ColumnEncoding::Float { storage } => {
            let v = decode_float(data, *storage)?;
            match col.logical_type {
                IngestedType::Float32 => Ok(ValueRef::Float32(v as f32)),
                _ => Ok(ValueRef::Float64(v)),
            }
        }
        ColumnEncoding::Bytes { fixed_len } => {
            let bytes = if tag == 2 {
                decode_bytes(data, None)?
            } else {
                decode_bytes(data, *fixed_len)?
            };
            Ok(ValueRef::Bytes(bytes))
        }
        ColumnEncoding::Uuid => {
            let bytes = decode_fixed_bytes::<16>(data)?;
            Ok(ValueRef::Uuid(bytes))
        }
        ColumnEncoding::Uuid256 => {
            let bytes = decode_fixed_bytes::<32>(data)?;
            Ok(ValueRef::Uuid256(bytes))
        }
        ColumnEncoding::Ip => {
            let bytes = decode_fixed_bytes::<16>(data)?;
            Ok(ValueRef::Ip(bytes))
        }
        ColumnEncoding::Date { varint, bits } => {
            Ok(ValueRef::Date(decode_int(data, *varint, *bits)? as i32))
        }
        ColumnEncoding::Timestamp => {
            let secs = zigzag_decode(read_varint(data)?);
            let nanos = read_varint(data)? as u32;
            Ok(ValueRef::Timestamp { secs, nanos })
        }
        ColumnEncoding::Decimal {
            scale,
            bytes,
            storage,
            ..
        } => {
            let text = if tag == 2 || matches!(storage, DecimalStorage::Raw) {
                let raw = decode_decimal_bytes(data, *bytes as usize)?;
                format_decimal_from_le(raw, *scale)
            } else {
                match storage {
                    DecimalStorage::Varint => {
                        let v = zigzag_decode_i128(read_varint_u128(data)?);
                        if *scale == 0 && v >= i64::MIN as i128 && v <= i64::MAX as i128 {
                            (v as i64).to_string()
                        } else {
                            format_decimal_i128(v, *scale)
                        }
                    }
                    DecimalStorage::Raw => unreachable!(),
                }
            };
            scratch.string.clear();
            scratch.string.push_str(&text);
            Ok(ValueRef::String(&scratch.string))
        }
        ColumnEncoding::String { encoding } => {
            scratch.string.clear();
            if tag == 2 {
                decode_string_fallback_into(data, &mut scratch.string)?;
            } else {
                decode_string_into(data, encoding, dict, scratch)?;
            }
            Ok(ValueRef::String(&scratch.string))
        }
        ColumnEncoding::FloatArray { storage } => {
            let len = read_varint(data)? as usize;
            let elem_bytes = match storage {
                FloatStorage::F16 => 2,
                FloatStorage::F32 => 4,
                FloatStorage::F64 => 8,
            };
            let max_len = data.len() / elem_bytes;
            if len > MAX_ARRAY_LEN || len > max_len {
                return Err(anyhow!("rowpack array length overflow"));
            }
            match col.logical_type {
                IngestedType::Float16Array | IngestedType::Float32Array => {
                    scratch.float32.clear();
                    for _ in 0..len {
                        let v = decode_float(data, *storage)?;
                        scratch.float32.push(v as f32);
                    }
                    Ok(ValueRef::Float32Array(&scratch.float32))
                }
                IngestedType::Float64Array => {
                    scratch.float64.clear();
                    for _ in 0..len {
                        let v = decode_float(data, *storage)?;
                        scratch.float64.push(v);
                    }
                    Ok(ValueRef::Float64Array(&scratch.float64))
                }
                _ => Err(anyhow!("rowpack invalid array type")),
            }
        }
        ColumnEncoding::Bool => Err(anyhow!("rowpack bool unexpected")),
    }
}

fn decode_value_skip(
    col: &ColumnContext,
    tag: u8,
    data: &mut &[u8],
    dict: Option<&Vec<Vec<u8>>>,
) -> Result<()> {
    match &col.encoding {
        ColumnEncoding::Int {
            signed,
            varint,
            bits,
        } => {
            if *signed {
                let _ = decode_int(data, *varint, *bits)?;
            } else {
                let _ = decode_uint(data, *varint, *bits)?;
            }
        }
        ColumnEncoding::Float { storage } => {
            let _ = decode_float(data, *storage)?;
        }
        ColumnEncoding::Bytes { fixed_len } => {
            let _ = if tag == 2 {
                decode_bytes(data, None)?
            } else {
                decode_bytes(data, *fixed_len)?
            };
        }
        ColumnEncoding::Uuid => {
            let _ = decode_fixed_bytes::<16>(data)?;
        }
        ColumnEncoding::Uuid256 => {
            let _ = decode_fixed_bytes::<32>(data)?;
        }
        ColumnEncoding::Ip => {
            let _ = decode_fixed_bytes::<16>(data)?;
        }
        ColumnEncoding::Date { varint, bits } => {
            let _ = decode_int(data, *varint, *bits)?;
        }
        ColumnEncoding::Timestamp => {
            let _ = read_varint(data)?;
            let _ = read_varint(data)?;
        }
        ColumnEncoding::Decimal {
            scale,
            bytes,
            storage,
            ..
        } => {
            if tag == 2 || matches!(storage, DecimalStorage::Raw) {
                let _ = decode_decimal_bytes(data, *bytes as usize)?;
            } else {
                match storage {
                    DecimalStorage::Varint => {
                        let _ = read_varint_u128(data)?;
                    }
                    DecimalStorage::Raw => {}
                }
            }
            let _ = scale;
        }
        ColumnEncoding::String { encoding } => match encoding {
            StringEncoding::Raw => {
                let len = read_varint(data)? as usize;
                if len > MAX_FIELD_BYTES {
                    return Err(anyhow!("rowpack string too large"));
                }
                if data.len() < len {
                    return Err(anyhow!("rowpack string truncated"));
                }
                *data = &data[len..];
            }
            StringEncoding::Dict { .. } => {
                let idx = read_varint(data)? as usize;
                if let Some(values) = dict {
                    if idx < values.len() {
                        return Ok(());
                    }
                }
                let len = read_varint(data)? as usize;
                if len > MAX_FIELD_BYTES {
                    return Err(anyhow!("rowpack dict fallback too large"));
                }
                if data.len() < len {
                    return Err(anyhow!("rowpack dict fallback truncated"));
                }
                *data = &data[len..];
            }
            StringEncoding::Alphabet { alphabet } => {
                let len = read_varint(data)? as usize;
                if len > MAX_FIELD_BYTES {
                    return Err(anyhow!("rowpack string too large"));
                }
                let bits = alphabet_bits(alphabet.len());
                let total_bits = len.saturating_mul(bits as usize);
                let byte_len = (total_bits + 7) / 8;
                if data.len() < byte_len {
                    return Err(anyhow!("rowpack string truncated"));
                }
                *data = &data[byte_len..];
            }
            StringEncoding::Bpe { dict } => {
                let token_count = read_varint(data)? as usize;
                if token_count > MAX_ARRAY_LEN {
                    return Err(anyhow!("rowpack bpe token count too large"));
                }
                let bits = bpe_token_bits(dict.len());
                let total_bits = token_count.saturating_mul(bits as usize);
                let byte_len = (total_bits + 7) / 8;
                if data.len() < byte_len {
                    return Err(anyhow!("rowpack bpe truncated"));
                }
                *data = &data[byte_len..];
            }
            StringEncoding::Lz4 => {
                let raw_len = read_varint(data)? as usize;
                if raw_len > MAX_FIELD_BYTES {
                    return Err(anyhow!("rowpack lz4 too large"));
                }
                let compressed_len = read_varint(data)? as usize;
                if data.len() < compressed_len {
                    return Err(anyhow!("rowpack lz4 truncated"));
                }
                *data = &data[compressed_len..];
            }
        },
        ColumnEncoding::FloatArray { storage } => {
            let len = read_varint(data)? as usize;
            let elem_bytes = match storage {
                FloatStorage::F16 => 2,
                FloatStorage::F32 => 4,
                FloatStorage::F64 => 8,
            };
            let total = len.saturating_mul(elem_bytes);
            if data.len() < total {
                return Err(anyhow!("rowpack array truncated"));
            }
            *data = &data[total..];
        }
        ColumnEncoding::Bool => {}
    }
    Ok(())
}

fn decode_int(data: &mut &[u8], varint: bool, bits: u8) -> Result<i64> {
    if varint {
        let v = read_varint(data)?;
        Ok(zigzag_decode(v))
    } else if bits <= 8 {
        if data.len() < 1 {
            return Err(anyhow!("rowpack int truncated"));
        }
        let (head, tail) = data.split_at(1);
        *data = tail;
        Ok(i8::from_le_bytes(head.try_into().unwrap()) as i64)
    } else if bits <= 16 {
        if data.len() < 2 {
            return Err(anyhow!("rowpack int truncated"));
        }
        let (head, tail) = data.split_at(2);
        *data = tail;
        Ok(i16::from_le_bytes(head.try_into().unwrap()) as i64)
    } else if bits <= 32 {
        if data.len() < 4 {
            return Err(anyhow!("rowpack int truncated"));
        }
        let (head, tail) = data.split_at(4);
        *data = tail;
        Ok(i32::from_le_bytes(head.try_into().unwrap()) as i64)
    } else {
        if data.len() < 8 {
            return Err(anyhow!("rowpack int truncated"));
        }
        let (head, tail) = data.split_at(8);
        *data = tail;
        Ok(i64::from_le_bytes(head.try_into().unwrap()))
    }
}

fn decode_uint(data: &mut &[u8], varint: bool, bits: u8) -> Result<u64> {
    if varint {
        read_varint(data)
    } else if bits <= 8 {
        if data.len() < 1 {
            return Err(anyhow!("rowpack uint truncated"));
        }
        let (head, tail) = data.split_at(1);
        *data = tail;
        Ok(u8::from_le_bytes(head.try_into().unwrap()) as u64)
    } else if bits <= 16 {
        if data.len() < 2 {
            return Err(anyhow!("rowpack uint truncated"));
        }
        let (head, tail) = data.split_at(2);
        *data = tail;
        Ok(u16::from_le_bytes(head.try_into().unwrap()) as u64)
    } else if bits <= 32 {
        if data.len() < 4 {
            return Err(anyhow!("rowpack uint truncated"));
        }
        let (head, tail) = data.split_at(4);
        *data = tail;
        Ok(u32::from_le_bytes(head.try_into().unwrap()) as u64)
    } else {
        if data.len() < 8 {
            return Err(anyhow!("rowpack uint truncated"));
        }
        let (head, tail) = data.split_at(8);
        *data = tail;
        Ok(u64::from_le_bytes(head.try_into().unwrap()))
    }
}

fn decode_float(data: &mut &[u8], storage: FloatStorage) -> Result<f64> {
    match storage {
        FloatStorage::F16 => {
            if data.len() < 2 {
                return Err(anyhow!("rowpack f16 truncated"));
            }
            let (head, tail) = data.split_at(2);
            *data = tail;
            let bits = u16::from_le_bytes(head.try_into().unwrap());
            Ok(f16::from_bits(bits).to_f32() as f64)
        }
        FloatStorage::F32 => {
            if data.len() < 4 {
                return Err(anyhow!("rowpack f32 truncated"));
            }
            let (head, tail) = data.split_at(4);
            *data = tail;
            Ok(f32::from_le_bytes(head.try_into().unwrap()) as f64)
        }
        FloatStorage::F64 => {
            if data.len() < 8 {
                return Err(anyhow!("rowpack f64 truncated"));
            }
            let (head, tail) = data.split_at(8);
            *data = tail;
            Ok(f64::from_le_bytes(head.try_into().unwrap()))
        }
    }
}

fn decode_bytes<'a>(data: &mut &'a [u8], fixed_len: Option<usize>) -> Result<&'a [u8]> {
    let len = if let Some(len) = fixed_len {
        len
    } else {
        read_varint(data)? as usize
    };
    if len > MAX_FIELD_BYTES {
        return Err(anyhow!("rowpack bytes too large"));
    }
    if data.len() < len {
        return Err(anyhow!("rowpack bytes truncated"));
    }
    let (head, tail) = data.split_at(len);
    *data = tail;
    Ok(head)
}

fn decode_fixed_bytes<const N: usize>(data: &mut &[u8]) -> Result<[u8; N]> {
    if data.len() < N {
        return Err(anyhow!("rowpack bytes truncated"));
    }
    let (head, tail) = data.split_at(N);
    *data = tail;
    let mut out = [0u8; N];
    out.copy_from_slice(head);
    Ok(out)
}

fn format_ip(bytes: [u8; 16]) -> String {
    let addr = std::net::Ipv6Addr::from(bytes);
    if let Some(v4) = addr.to_ipv4() {
        v4.to_string()
    } else {
        addr.to_string()
    }
}

fn decode_decimal_bytes<'a>(data: &mut &'a [u8], bytes: usize) -> Result<&'a [u8]> {
    if data.len() < bytes {
        return Err(anyhow!("rowpack decimal truncated"));
    }
    let (head, tail) = data.split_at(bytes);
    *data = tail;
    Ok(head)
}

fn format_decimal_from_le(bytes: &[u8], scale: i32) -> String {
    let value = BigInt::from_signed_bytes_le(bytes);
    let negative = value.is_negative();
    let mut digits = value.abs().to_str_radix(10);
    if scale <= 0 {
        let zeros = (-scale) as usize;
        if zeros > 0 {
            digits.push_str(&"0".repeat(zeros));
        }
        return if negative {
            format!("-{}", digits)
        } else {
            digits
        };
    }
    let scale = scale as usize;
    if digits.len() <= scale {
        let pad = scale - digits.len();
        let mut out = String::with_capacity(scale + 2);
        if negative {
            out.push('-');
        }
        out.push_str("0.");
        out.push_str(&"0".repeat(pad));
        out.push_str(&digits);
        return out;
    }
    let split = digits.len() - scale;
    let mut out = String::with_capacity(digits.len() + 1);
    if negative {
        out.push('-');
    }
    out.push_str(&digits[..split]);
    out.push('.');
    out.push_str(&digits[split..]);
    out
}

fn format_decimal_i128(value: i128, scale: i32) -> String {
    let negative = value.is_negative();
    let mut digits = value.unsigned_abs().to_string();
    if scale <= 0 {
        let zeros = (-scale) as usize;
        if zeros > 0 {
            digits.push_str(&"0".repeat(zeros));
        }
        return if negative {
            format!("-{}", digits)
        } else {
            digits
        };
    }
    let scale = scale as usize;
    if digits.len() <= scale {
        let pad = scale - digits.len();
        let mut out = String::with_capacity(scale + 2);
        if negative {
            out.push('-');
        }
        out.push_str("0.");
        out.push_str(&"0".repeat(pad));
        out.push_str(&digits);
        return out;
    }
    let split = digits.len() - scale;
    let mut out = String::with_capacity(digits.len() + 1);
    if negative {
        out.push('-');
    }
    out.push_str(&digits[..split]);
    out.push('.');
    out.push_str(&digits[split..]);
    out
}

fn decode_string(
    data: &mut &[u8],
    encoding: &StringEncoding,
    dict: Option<&Vec<Vec<u8>>>,
) -> Result<String> {
    match encoding {
        StringEncoding::Raw => {
            let len = read_varint(data)? as usize;
            if len > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack string too large"));
            }
            if data.len() < len {
                return Err(anyhow!("rowpack string truncated"));
            }
            let (head, tail) = data.split_at(len);
            *data = tail;
            Ok(String::from_utf8_lossy(head).to_string())
        }
        StringEncoding::Dict { .. } => {
            let idx = read_varint(data)? as usize;
            if let Some(values) = dict {
                if let Some(bytes) = values.get(idx) {
                    Ok(String::from_utf8_lossy(bytes).to_string())
                } else {
                    let len = read_varint(data)? as usize;
                    if len > MAX_FIELD_BYTES {
                        return Err(anyhow!("rowpack dict fallback too large"));
                    }
                    if data.len() < len {
                        return Err(anyhow!("rowpack dict fallback truncated"));
                    }
                    let (head, tail) = data.split_at(len);
                    *data = tail;
                    Ok(String::from_utf8_lossy(head).to_string())
                }
            } else {
                Err(anyhow!("rowpack missing dict"))
            }
        }
        StringEncoding::Alphabet { alphabet } => {
            let len = read_varint(data)? as usize;
            if len > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack string too large"));
            }
            let bits = alphabet_bits(alphabet.len());
            let total_bits = len.saturating_mul(bits as usize);
            let byte_len = (total_bits + 7) / 8;
            if data.len() < byte_len {
                return Err(anyhow!("rowpack string truncated"));
            }
            let (head, tail) = data.split_at(byte_len);
            *data = tail;
            let mut reader = BitReader::new(head);
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                let idx = reader.read_bits(bits)? as usize;
                let byte = alphabet
                    .get(idx)
                    .ok_or_else(|| anyhow!("rowpack alphabet index out of range"))?;
                out.push(*byte);
            }
            Ok(String::from_utf8_lossy(&out).to_string())
        }
        StringEncoding::Bpe { dict } => {
            let token_count = read_varint(data)? as usize;
            if token_count > MAX_ARRAY_LEN {
                return Err(anyhow!("rowpack bpe token count too large"));
            }
            let bits = bpe_token_bits(dict.len());
            let total_bits = token_count.saturating_mul(bits as usize);
            let byte_len = (total_bits + 7) / 8;
            if data.len() < byte_len {
                return Err(anyhow!("rowpack bpe truncated"));
            }
            let (head, tail) = data.split_at(byte_len);
            *data = tail;
            let mut reader = BitReader::new(head);
            let mut out = Vec::new();
            for _ in 0..token_count {
                let token = reader.read_bits_u32(bits)? as usize;
                if token < 256 {
                    out.push(token as u8);
                } else {
                    let idx = token - 256;
                    let bytes = dict
                        .get(idx)
                        .ok_or_else(|| anyhow!("rowpack bpe token out of range"))?;
                    out.extend_from_slice(bytes);
                }
                if out.len() > MAX_FIELD_BYTES {
                    return Err(anyhow!("rowpack string too large"));
                }
            }
            Ok(String::from_utf8_lossy(&out).to_string())
        }
        StringEncoding::Lz4 => {
            let raw_len = read_varint(data)? as usize;
            if raw_len > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack lz4 too large"));
            }
            let compressed_len = read_varint(data)? as usize;
            if data.len() < compressed_len {
                return Err(anyhow!("rowpack lz4 truncated"));
            }
            let (head, tail) = data.split_at(compressed_len);
            *data = tail;
            let decoded = lz4_flex::block::decompress(head, raw_len)
                .map_err(|_| anyhow!("rowpack lz4 decode failed"))?;
            if decoded.len() != raw_len || decoded.len() > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack lz4 too large"));
            }
            Ok(String::from_utf8_lossy(&decoded).to_string())
        }
    }
}

fn decode_string_into(
    data: &mut &[u8],
    encoding: &StringEncoding,
    dict: Option<&Vec<Vec<u8>>>,
    scratch: &mut RowPackDecodeScratch,
) -> Result<()> {
    scratch.string.clear();
    match encoding {
        StringEncoding::Raw => {
            let len = read_varint(data)? as usize;
            if len > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack string too large"));
            }
            if data.len() < len {
                return Err(anyhow!("rowpack string truncated"));
            }
            let (head, tail) = data.split_at(len);
            *data = tail;
            push_utf8_lossy(&mut scratch.string, head);
            Ok(())
        }
        StringEncoding::Dict { .. } => {
            let idx = read_varint(data)? as usize;
            if let Some(values) = dict {
                if let Some(bytes) = values.get(idx) {
                    push_utf8_lossy(&mut scratch.string, bytes);
                    Ok(())
                } else {
                    decode_string_fallback_into(data, &mut scratch.string)
                }
            } else {
                Err(anyhow!("rowpack missing dict"))
            }
        }
        StringEncoding::Alphabet { alphabet } => {
            let len = read_varint(data)? as usize;
            if len > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack string too large"));
            }
            let bits = alphabet_bits(alphabet.len());
            let total_bits = len.saturating_mul(bits as usize);
            let byte_len = (total_bits + 7) / 8;
            if data.len() < byte_len {
                return Err(anyhow!("rowpack string truncated"));
            }
            let (head, tail) = data.split_at(byte_len);
            *data = tail;
            let mut reader = BitReader::new(head);
            scratch.bytes.clear();
            scratch.bytes.reserve(len);
            for _ in 0..len {
                let idx = reader.read_bits(bits)? as usize;
                let byte = alphabet
                    .get(idx)
                    .ok_or_else(|| anyhow!("rowpack alphabet index out of range"))?;
                scratch.bytes.push(*byte);
            }
            push_utf8_lossy(&mut scratch.string, &scratch.bytes);
            Ok(())
        }
        StringEncoding::Bpe { dict } => {
            let token_count = read_varint(data)? as usize;
            if token_count > MAX_ARRAY_LEN {
                return Err(anyhow!("rowpack bpe token count too large"));
            }
            let bits = bpe_token_bits(dict.len());
            let total_bits = token_count.saturating_mul(bits as usize);
            let byte_len = (total_bits + 7) / 8;
            if data.len() < byte_len {
                return Err(anyhow!("rowpack bpe truncated"));
            }
            let (head, tail) = data.split_at(byte_len);
            *data = tail;
            let mut reader = BitReader::new(head);
            scratch.bytes.clear();
            for _ in 0..token_count {
                let token = reader.read_bits_u32(bits)? as usize;
                if token < 256 {
                    scratch.bytes.push(token as u8);
                } else {
                    let idx = token - 256;
                    let bytes = dict
                        .get(idx)
                        .ok_or_else(|| anyhow!("rowpack bpe token out of range"))?;
                    scratch.bytes.extend_from_slice(bytes);
                }
                if scratch.bytes.len() > MAX_FIELD_BYTES {
                    return Err(anyhow!("rowpack string too large"));
                }
            }
            push_utf8_lossy(&mut scratch.string, &scratch.bytes);
            Ok(())
        }
        StringEncoding::Lz4 => {
            let raw_len = read_varint(data)? as usize;
            if raw_len > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack lz4 too large"));
            }
            let compressed_len = read_varint(data)? as usize;
            if data.len() < compressed_len {
                return Err(anyhow!("rowpack lz4 truncated"));
            }
            let (head, tail) = data.split_at(compressed_len);
            *data = tail;
            let decoded = lz4_flex::block::decompress(head, raw_len)
                .map_err(|_| anyhow!("rowpack lz4 decode failed"))?;
            if decoded.len() != raw_len || decoded.len() > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack lz4 too large"));
            }
            push_utf8_lossy(&mut scratch.string, &decoded);
            Ok(())
        }
    }
}

fn decode_string_fallback(data: &mut &[u8]) -> Result<String> {
    let len = read_varint(data)? as usize;
    if len > MAX_FIELD_BYTES {
        return Err(anyhow!("rowpack string fallback too large"));
    }
    if data.len() < len {
        return Err(anyhow!("rowpack string fallback truncated"));
    }
    let (head, tail) = data.split_at(len);
    *data = tail;
    Ok(String::from_utf8_lossy(head).to_string())
}

fn decode_string_fallback_into(data: &mut &[u8], out: &mut String) -> Result<()> {
    let len = read_varint(data)? as usize;
    if len > MAX_FIELD_BYTES {
        return Err(anyhow!("rowpack string fallback too large"));
    }
    if data.len() < len {
        return Err(anyhow!("rowpack string fallback truncated"));
    }
    let (head, tail) = data.split_at(len);
    *data = tail;
    out.clear();
    push_utf8_lossy(out, head);
    Ok(())
}

fn push_utf8_lossy(out: &mut String, bytes: &[u8]) {
    if let Ok(text) = std::str::from_utf8(bytes) {
        out.push_str(text);
    } else {
        out.push_str(&String::from_utf8_lossy(bytes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, Float64Array, Int32Array};
    use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime};
    use std::sync::Arc;

    fn encode_row(encoder: &RowPackEncoder, row: &[ValueRef<'_>]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut scratch = RowPackScratch::new(encoder.context());
        encoder
            .encode_row_into(row, &mut out, &mut scratch)
            .unwrap();
        out
    }

    fn build_context() -> RowPackContext {
        RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![
                ColumnContext {
                    name: "id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "name".to_string(),
                    logical_type: IngestedType::String,
                    encoding: ColumnEncoding::String {
                        encoding: StringEncoding::Dict {
                            values: vec![b"alice".to_vec(), b"bob".to_vec()],
                        },
                    },
                },
                ColumnContext {
                    name: "active".to_string(),
                    logical_type: IngestedType::Bool,
                    encoding: ColumnEncoding::Bool,
                },
                ColumnContext {
                    name: "score".to_string(),
                    logical_type: IngestedType::Float32,
                    encoding: ColumnEncoding::Float {
                        storage: FloatStorage::F16,
                    },
                },
                ColumnContext {
                    name: "payload".to_string(),
                    logical_type: IngestedType::Bytes,
                    encoding: ColumnEncoding::Bytes { fixed_len: None },
                },
                ColumnContext {
                    name: "created_at".to_string(),
                    logical_type: IngestedType::Timestamp,
                    encoding: ColumnEncoding::Timestamp,
                },
                ColumnContext {
                    name: "embed".to_string(),
                    logical_type: IngestedType::Float32Array,
                    encoding: ColumnEncoding::FloatArray {
                        storage: FloatStorage::F32,
                    },
                },
            ],
            estimated_row_bytes: 0,
        }
    }

    fn decimal256_bytes(value: &BigInt) -> [u8; 32] {
        let bytes = value.to_signed_bytes_le();
        let fill = if value.is_negative() { 0xff } else { 0x00 };
        let mut out = [fill; 32];
        if !bytes.is_empty() {
            out[..bytes.len()].copy_from_slice(&bytes);
        }
        out
    }

    #[test]
    fn bool_from_numeric_ints() {
        let cols: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![0, 1, -1]))];
        let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: None,
        }];
        let mut access = RowAccess::new(hints, RowAccessStats::new(None));
        access.set_cols(&cols);

        access.set_row(0);
        assert_eq!(access.get_bool(0).unwrap(), Some(false));
        access.set_row(1);
        assert_eq!(access.get_bool(0).unwrap(), Some(true));
        access.set_row(2);
        assert_eq!(access.get_bool(0).unwrap(), Some(true));
    }

    #[test]
    fn bool_from_numeric_floats() {
        let cols: Vec<ArrayRef> =
            vec![Arc::new(Float64Array::from(vec![0.0, 1.0, -0.5, f64::NAN]))];
        let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: None,
        }];
        let mut access = RowAccess::new(hints, RowAccessStats::new(None));
        access.set_cols(&cols);

        access.set_row(0);
        assert_eq!(access.get_bool(0).unwrap(), Some(false));
        access.set_row(1);
        assert_eq!(access.get_bool(0).unwrap(), Some(true));
        access.set_row(2);
        assert_eq!(access.get_bool(0).unwrap(), Some(true));
        access.set_row(3);
        assert_eq!(access.get_bool(0).unwrap(), Some(false));
    }

    #[test]
    fn bool_from_float32_nan() {
        let cols: Vec<ArrayRef> = vec![Arc::new(Float32Array::from(vec![f32::NAN]))];
        let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: None,
        }];
        let mut access = RowAccess::new(hints, RowAccessStats::new(None));
        access.set_cols(&cols);
        access.set_row(0);
        assert_eq!(access.get_bool(0).unwrap(), Some(false));
    }

    #[test]
    fn parse_bool_string_variants() {
        for value in ["TRUE", "t", "Yes", "Y", "on", "enabled", "checked", "x"] {
            assert_eq!(parse_bool(value), Some(true));
        }
        for value in ["FALSE", "f", "No", "n", "off", "disabled", "unchecked", ""] {
            assert_eq!(parse_bool(value), Some(false));
        }
        for value in ["1", "-2", "0.5"] {
            assert_eq!(parse_bool(value), Some(true));
        }
        for value in ["0", "0.0", "NaN"] {
            assert_eq!(parse_bool(value), Some(false));
        }
        assert_eq!(parse_bool("\"yes\""), Some(true));
        assert_eq!(parse_bool("\"0\""), Some(false));
    }

    #[test]
    fn parse_precompiled_timestamp_formats_match_chrono() {
        let cases: Vec<(&str, Vec<&str>)> = vec![
            ("%Y-%m-%d", vec!["1970-01-01", "2020-02-29"]),
            (
                "%Y-%m-%d %H:%M",
                vec!["1970-01-01 00:00", "2020-02-29 23:59"],
            ),
            (
                "%Y-%m-%d %H:%M:%S",
                vec!["1970-01-01 00:00:01", "2020-02-29 23:59:59"],
            ),
            (
                "%Y-%m-%d %H:%M:%S%.f",
                vec![
                    "1970-01-01 00:00:00.5",
                    "2020-02-29 23:59:59.123456789",
                    "2020-02-29 23:59:59.000000001",
                ],
            ),
            (
                "%Y-%m-%dT%H:%M:%S",
                vec!["1970-01-01T00:00:02", "2020-02-29T23:59:59"],
            ),
            (
                "%Y-%m-%dT%H:%M:%S%.f",
                vec!["1970-01-01T00:00:03.250", "2020-02-29T23:59:59.999999999"],
            ),
            (
                "%Y-%m-%d %H:%M:%S%z",
                vec![
                    "1970-01-01 00:00:00+0000",
                    "1970-01-01 01:00:00+0100",
                    "2020-02-29 23:59:59-0530",
                ],
            ),
            (
                "%Y-%m-%d %H:%M:%S %z",
                vec!["1970-01-01 00:00:00 +0000", "1970-01-01 01:00:00 +0100"],
            ),
            (
                "%Y-%m-%d %H:%M:%S%:z",
                vec!["1970-01-01 00:00:00+00:00", "1970-01-01 01:00:00+01:00"],
            ),
            (
                "%Y-%m-%d %H:%M:%S %:z",
                vec!["1970-01-01 00:00:00 +00:00", "1970-01-01 01:00:00 +01:00"],
            ),
            (
                "%Y-%m-%dT%H:%M:%S%z",
                vec!["1970-01-01T00:00:00+0000", "1970-01-01T01:00:00+0100"],
            ),
            (
                "%Y-%m-%dT%H:%M:%S%:z",
                vec!["1970-01-01T00:00:00+00:00", "1970-01-01T01:00:00+01:00"],
            ),
        ];

        for (fmt, values) in cases {
            let parser = TimestampFormatParser::compile(fmt).unwrap();
            for value in values {
                let (fast_secs, fast_nanos) = parser.parse_timestamp(value).unwrap();
                let (chrono_secs, chrono_nanos) = if fmt.contains("%z") || fmt.contains("%:z") {
                    let dt: DateTime<FixedOffset> = DateTime::parse_from_str(value, fmt).unwrap();
                    let utc = dt.naive_utc().and_utc();
                    (utc.timestamp(), utc.timestamp_subsec_nanos())
                } else if fmt == "%Y-%m-%d" {
                    let date = NaiveDate::parse_from_str(value, fmt).unwrap();
                    let dt = date.and_hms_opt(0, 0, 0).unwrap();
                    let utc = dt.and_utc();
                    (utc.timestamp(), utc.timestamp_subsec_nanos())
                } else {
                    let dt = NaiveDateTime::parse_from_str(value, fmt).unwrap();
                    let utc = dt.and_utc();
                    (utc.timestamp(), utc.timestamp_subsec_nanos())
                };
                assert_eq!((fast_secs, fast_nanos), (chrono_secs, chrono_nanos));
            }
        }
    }

    #[test]
    fn roundtrip_basic() {
        let ctx = build_context();
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![
            ValueRef::Int64(42),
            ValueRef::String("alice"),
            ValueRef::Bool(true),
            ValueRef::Float32(3.5),
            ValueRef::Bytes(b"hello"),
            ValueRef::Timestamp {
                secs: 1_700_000_000,
                nanos: 123_000_000,
            },
            ValueRef::Float32Array(&[0.1, 0.2, 0.3]),
        ];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let msg = decoder.decode_msgpack(&encoded).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(msg.len() > 0);
        assert!(json.is_object());
    }

    #[test]
    fn timestamp_nanos_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "ts".to_string(),
                logical_type: IngestedType::Timestamp,
                encoding: ColumnEncoding::Timestamp,
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Timestamp {
            secs: 1_700_000_000,
            nanos: 123_000_000,
        }];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains(".123"));
    }

    #[test]
    fn decimal128_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "amount".to_string(),
                logical_type: IngestedType::Decimal128,
                encoding: ColumnEncoding::Decimal {
                    precision: 10,
                    scale: 2,
                    bytes: decimal_raw_bytes(10).unwrap(),
                    storage: DecimalStorage::Varint,
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Decimal128(-12345i128)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("amount"), Some(&JsonValue::from("-123.45")));
    }

    #[test]
    fn decimal256_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "amount".to_string(),
                logical_type: IngestedType::Decimal256,
                encoding: ColumnEncoding::Decimal {
                    precision: 50,
                    scale: 4,
                    bytes: decimal_raw_bytes(50).unwrap(),
                    storage: DecimalStorage::Raw,
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let big = BigInt::parse_bytes(b"123456789012345678901234567890", 10).unwrap();
        let row = vec![ValueRef::Decimal256(decimal256_bytes(&big))];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert!(map.get("amount").and_then(|v| v.as_str()).is_some());
    }

    #[test]
    fn decimal_raw_encoding_when_out_of_range() {
        let config = RowPackConfig::default();
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "amount".to_string(),
                logical_type: IngestedType::Decimal128,
                fixed_len: None,
                precision: Some(10),
                scale: Some(2),
            }],
            config,
        );
        builder
            .observe_row(&[ValueRef::Decimal128(i128::MAX)])
            .unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::Decimal {
                storage: DecimalStorage::Varint,
                ..
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Decimal128(i128::MAX)];
        let encoded = encode_row(&encoder, &row);
        assert_eq!(encoded[0] & 0x03, 2);
    }

    #[test]
    fn decimal_int_encoding_scale_zero() {
        let config = RowPackConfig::default();
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "amount".to_string(),
                logical_type: IngestedType::Decimal128,
                fixed_len: None,
                precision: Some(10),
                scale: Some(0),
            }],
            config,
        );
        builder
            .observe_row(&[ValueRef::Decimal128(12345i128)])
            .unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::Decimal {
                storage: DecimalStorage::Varint,
                ..
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn decimal_scale_nonzero_int_roundtrip() {
        let config = RowPackConfig::default();
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "amount".to_string(),
                logical_type: IngestedType::Decimal128,
                fixed_len: None,
                precision: Some(10),
                scale: Some(3),
            }],
            config,
        );
        builder
            .observe_row(&[ValueRef::Decimal128(123456i128)])
            .unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::Decimal {
                storage: DecimalStorage::Varint,
                ..
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Decimal128(123456i128)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("amount"), Some(&JsonValue::from("123.456")));
    }

    #[test]
    fn decimal_int_fallback_when_out_of_range() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "amount".to_string(),
                logical_type: IngestedType::Decimal128,
                encoding: ColumnEncoding::Decimal {
                    precision: 20,
                    scale: 0,
                    bytes: decimal_raw_bytes(20).unwrap(),
                    storage: DecimalStorage::Varint,
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let big = i128::MAX;
        let row = vec![ValueRef::Decimal128(big)];
        let encoded = encode_row(&encoder, &row);
        assert_eq!(encoded[0] & 0x03, 2);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert!(map.get("amount").unwrap().as_str().is_some());
    }

    #[test]
    fn decimal_varint_thresholds_all_precisions() {
        fn pow10_i128(exp: u32) -> i128 {
            let mut out = 1i128;
            for _ in 0..exp {
                out = out.saturating_mul(10);
            }
            out
        }

        for precision in 1..=38u32 {
            let bytes = decimal_raw_bytes(precision).unwrap();
            let ctx = RowPackContext {
                version: ROWPACK_VERSION,
                columns: vec![ColumnContext {
                    name: "amount".to_string(),
                    logical_type: IngestedType::Decimal128,
                    encoding: ColumnEncoding::Decimal {
                        precision,
                        scale: 0,
                        bytes,
                        storage: DecimalStorage::Varint,
                    },
                }],
                estimated_row_bytes: 0,
            };
            let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
            let small = vec![ValueRef::Decimal128(1)];
            let encoded_small = encode_row(&encoder, &small);
            let expected_small_tag = if bytes == 1 { 2 } else { 1 };
            assert_eq!(encoded_small[0] & 0x03, expected_small_tag);

            let big = pow10_i128(precision) - 1;
            let encoded_big = encode_row(&encoder, &[ValueRef::Decimal128(big)]);
            assert_eq!(encoded_big[0] & 0x03, 2);
        }
    }

    #[test]
    fn dict_fallback() {
        let ctx = build_context();
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![
            ValueRef::Int64(1),
            ValueRef::String("carol"),
            ValueRef::Bool(false),
            ValueRef::Float32(1.0),
            ValueRef::Bytes(b"data"),
            ValueRef::Timestamp { secs: 0, nanos: 0 },
            ValueRef::Float32Array(&[]),
        ];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("carol"));
    }

    #[test]
    fn alphabet_encoding_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Alphabet {
                        alphabet: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-".to_vec(),
                    },
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("USER-0001")];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("code").and_then(|v| v.as_str()), Some("USER-0001"));
    }

    #[test]
    fn alphabet_encoding_fallback() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Alphabet {
                        alphabet: b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-".to_vec(),
                    },
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("USER-0001?")];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("code").and_then(|v| v.as_str()), Some("USER-0001?"));
    }

    #[test]
    fn alphabet_encoding_selected_for_short_strings() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        for idx in 0..100 {
            let value = format!("USER-{:04}", idx);
            builder.observe_row(&[ValueRef::String(&value)]).unwrap();
        }
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::Alphabet { .. },
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn alphabet_encoding_selected_at_long_threshold() {
        let config = RowPackConfig::default();
        let max_len = config.string_long_min_len.unwrap_or(64);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            config,
        );
        let value = "ABCD".repeat((max_len + 3) / 4)[..max_len].to_string();
        for _ in 0..50 {
            builder.observe_row(&[ValueRef::String(&value)]).unwrap();
        }
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::Alphabet { .. },
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn bpe_encoding_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Bpe {
                        dict: vec![b"TOKEN123".to_vec()],
                    },
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("TOKEN123TOKEN123")];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(
            map.get("code").and_then(|v| v.as_str()),
            Some("TOKEN123TOKEN123")
        );
    }

    #[test]
    fn bpe_selected_over_alphabet() {
        let mut config = RowPackConfig::default();
        config.bpe_min_gain_ratio = Some(1.0);
        config.bpe_max_dict_entries = Some(64);
        config.bpe_max_dict_bytes = Some(1024);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            config,
        );
        for _ in 0..50 {
            builder
                .observe_row(&[ValueRef::String("TOKEN123TOKEN123TOKEN123")])
                .unwrap();
        }
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::Bpe { .. },
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn bpe_decode_rejects_token_out_of_range() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Bpe {
                        dict: vec![b"TOKEN".to_vec()],
                    },
                },
            }],
            estimated_row_bytes: 0,
        };
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut tags = BitWriter::with_capacity(1);
        tags.write_bits(1, 2);
        let mut row = tags.finish();
        write_varint(1, &mut row);
        let tokens = vec![256 + 10];
        let bits = bpe_token_bits(1);
        write_tokens_packed(&mut row, &tokens, bits);
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("bpe token out of range"));
    }

    #[test]
    fn bpe_decode_rejects_excess_tokens() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Bpe {
                        dict: vec![b"TOKEN".to_vec()],
                    },
                },
            }],
            estimated_row_bytes: 0,
        };
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut tags = BitWriter::with_capacity(1);
        tags.write_bits(1, 2);
        let mut row = tags.finish();
        write_varint((MAX_ARRAY_LEN as u64) + 1, &mut row);
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("bpe token count too large"));
    }

    fn encode_string_size(encoding: StringEncoding, value: &str) -> usize {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String { encoding },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx).unwrap();
        let row = vec![ValueRef::String(value)];
        let encoded = encode_row(&encoder, &row);
        encoded.len()
    }

    #[test]
    fn alphabet_beats_bpe_on_structured_strings() {
        let samples = vec![
            "2024-01-05T03:24:15Z",
            "2024-02-11T14:09:26Z",
            "2024-03-19T22:58:01Z",
            "2024-04-07T08:03:44Z",
            "2024-05-23T16:47:10Z",
            "2024-06-30T01:11:59Z",
            "2024-07-02T12:35:22Z",
            "2024-08-14T19:00:05Z",
            "2024-09-28T05:42:33Z",
            "2024-10-31T23:59:59Z",
        ];
        let mut alpha_config = RowPackConfig::default();
        alpha_config.max_dict_entries = Some(0);
        alpha_config.bpe_max_dict_entries = Some(0);
        alpha_config.bpe_max_dict_bytes = Some(0);
        let alphabet_encoding = build_string_encoding(&samples, alpha_config);
        let StringEncoding::Alphabet { alphabet } = alphabet_encoding else {
            panic!("expected alphabet encoding");
        };
        let bpe_config = RowPackConfig::default();
        let bpe_dict = train_bpe_dict(
            &samples
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>(),
            bpe_config.bpe_max_dict_entries(),
            bpe_config.bpe_max_dict_bytes(),
        );
        for value in samples {
            let alpha_size = encode_string_size(
                StringEncoding::Alphabet {
                    alphabet: alphabet.clone(),
                },
                value,
            );
            let bpe_size = encode_string_size(
                StringEncoding::Bpe {
                    dict: bpe_dict.clone(),
                },
                value,
            );
            assert!(
                alpha_size < bpe_size,
                "alphabet size {} not smaller than bpe size {} for {}",
                alpha_size,
                bpe_size,
                value
            );
            assert!(
                alpha_size < value.len(),
                "alphabet size {} not smaller than raw size {} for {}",
                alpha_size,
                value.len(),
                value
            );
        }
    }

    #[test]
    fn bpe_beats_alphabet_on_english_sentences() {
        let samples = vec![
            "We left early for the concert and found a quiet seat near the aisle before the crowd arrived.",
            "The pilot announced a smooth flight, and the cabin lights dimmed as everyone settled in.",
            "She organized the pantry by date and labeled each jar so cooking would be quicker later.",
            "A soft breeze moved through the trees while the hikers checked the map and shared water.",
            "The mechanic explained the repair clearly, then showed the worn part to avoid confusion.",
            "We wrote a short update for the team and included a timeline for the next review.",
            "His sketchbook held dozens of city scenes, each capturing traffic, shadows, and street noise.",
            "The festival opened at noon, and vendors arranged colorful tents along the main walkway.",
            "She practiced the presentation aloud, adjusting pacing so each section felt balanced.",
            "The museum tour ended with a restored mural that had been hidden behind the wall.",
        ];
        let mut alpha_config = RowPackConfig::default();
        alpha_config.max_dict_entries = Some(0);
        alpha_config.bpe_max_dict_entries = Some(0);
        alpha_config.bpe_max_dict_bytes = Some(0);
        let alphabet_encoding = build_string_encoding(&samples, alpha_config);
        let StringEncoding::Alphabet { alphabet } = alphabet_encoding else {
            panic!("expected alphabet encoding");
        };
        let bpe_dict = train_bpe_dict(
            &samples
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>(),
            256,
            32 * 1024,
        );
        assert!(!bpe_dict.is_empty(), "expected bpe dict to be non-empty");
        let mut total_alpha = 0usize;
        let mut total_bpe = 0usize;
        for value in samples {
            let alpha_size = encode_string_size(
                StringEncoding::Alphabet {
                    alphabet: alphabet.clone(),
                },
                value,
            );
            let bpe_size = encode_string_size(
                StringEncoding::Bpe {
                    dict: bpe_dict.clone(),
                },
                value,
            );
            total_alpha += alpha_size;
            total_bpe += bpe_size;
            assert!(
                bpe_size < value.len(),
                "bpe size {} not smaller than raw size {} for {}",
                bpe_size,
                value.len(),
                value
            );
        }
        assert!(
            total_bpe < total_alpha,
            "bpe total {} not smaller than alphabet total {}",
            total_bpe,
            total_alpha
        );
    }

    fn build_string_encoding(samples: &[&str], config: RowPackConfig) -> StringEncoding {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "code".to_string(),
                logical_type: IngestedType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            config,
        );
        for value in samples {
            builder.observe_row(&[ValueRef::String(value)]).unwrap();
        }
        let ctx = builder.finish().unwrap();
        let ColumnEncoding::String { encoding } = ctx.columns[0].encoding.clone() else {
            panic!("expected string encoding");
        };
        encoding
    }

    #[test]
    fn float_array_float16_encoding() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "embed".to_string(),
                logical_type: IngestedType::Float16Array,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        builder
            .observe_row(&[ValueRef::Float32Array(&[0.1, 0.2, 0.3])])
            .unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::FloatArray { storage } => {
                assert_eq!(*storage, FloatStorage::F16);
            }
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Float32Array(&[0.1, 0.2, 0.3])];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("0.1"));
    }

    #[test]
    fn bytes_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "blob".to_string(),
                logical_type: IngestedType::Bytes,
                encoding: ColumnEncoding::Bytes { fixed_len: None },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Bytes(b"xyz")];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("eHl6"));
    }

    #[test]
    fn bytes_fixed_len_success() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "blob".to_string(),
                logical_type: IngestedType::Bytes,
                encoding: ColumnEncoding::Bytes { fixed_len: Some(4) },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Bytes(b"ABCD")];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("QUJDRA=="));
    }

    #[test]
    fn bytes_fixed_len_fallback() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "blob".to_string(),
                logical_type: IngestedType::Bytes,
                encoding: ColumnEncoding::Bytes { fixed_len: Some(4) },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Bytes(b"ABCDE")];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("QUJDREU="));
    }

    #[test]
    fn uuid_json_hex() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "id".to_string(),
                logical_type: IngestedType::Uuid,
                encoding: ColumnEncoding::Uuid,
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let value = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let row = vec![ValueRef::Uuid(value)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json
            .to_string()
            .contains("00112233445566778899aabbccddeeff"));
    }

    #[test]
    fn uuid_value_ref_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "id".to_string(),
                logical_type: IngestedType::Uuid,
                encoding: ColumnEncoding::Uuid,
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let value = [
            0xde, 0xad, 0xbe, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x01, 0x23,
            0x45, 0x67,
        ];
        let row = vec![ValueRef::Uuid(value)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut scratch = RowPackDecodeScratch::default();
        let decoded = decoder
            .decode_column_value(&encoded, 0, &mut scratch)
            .unwrap();
        match decoded {
            ValueRef::Uuid(bytes) => assert_eq!(bytes, value),
            other => panic!("unexpected value ref: {:?}", other),
        }
    }

    #[test]
    fn uuid256_json_hex() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "id".to_string(),
                logical_type: IngestedType::Uuid256,
                encoding: ColumnEncoding::Uuid256,
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let mut value = [0u8; 32];
        for (idx, byte) in value.iter_mut().enumerate() {
            *byte = idx as u8;
        }
        let row = vec![ValueRef::Uuid256(value)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json
            .to_string()
            .contains("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"));
    }

    #[test]
    fn uuid256_value_ref_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "id".to_string(),
                logical_type: IngestedType::Uuid256,
                encoding: ColumnEncoding::Uuid256,
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let mut value = [0u8; 32];
        for (idx, byte) in value.iter_mut().enumerate() {
            *byte = 0xffu8.wrapping_sub(idx as u8);
        }
        let row = vec![ValueRef::Uuid256(value)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut scratch = RowPackDecodeScratch::default();
        let decoded = decoder
            .decode_column_value(&encoded, 0, &mut scratch)
            .unwrap();
        match decoded {
            ValueRef::Uuid256(bytes) => assert_eq!(bytes, value),
            other => panic!("unexpected value ref: {:?}", other),
        }
    }

    #[test]
    fn uuid_string_parse_error() {
        use std::sync::Arc;

        let arr = arrow_array::StringArray::from(vec!["not-a-uuid"]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: None,
        }];
        let stats = RowAccessStats::default();
        let mut access = RowAccess::new(hints, stats);
        access.set_cols(&cols);
        access.set_row(0);
        assert!(access.get_uuid(0).is_err());
    }

    #[test]
    fn uuid256_string_parse_error() {
        use std::sync::Arc;

        let text = "z".repeat(64);
        let arr = arrow_array::StringArray::from(vec![text.as_str()]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: None,
        }];
        let stats = RowAccessStats::default();
        let mut access = RowAccess::new(hints, stats);
        access.set_cols(&cols);
        access.set_row(0);
        assert!(access.get_uuid256(0).is_err());
    }

    #[test]
    fn ip_string_parse_error() {
        use std::sync::Arc;

        let arr = arrow_array::StringArray::from(vec!["999.999.999.999"]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: None,
        }];
        let stats = RowAccessStats::default();
        let mut access = RowAccess::new(hints, stats);
        access.set_cols(&cols);
        access.set_row(0);
        assert!(access.get_ip(0).is_err());
    }

    #[test]
    fn ip_json_string() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "ip".to_string(),
                logical_type: IngestedType::Ip,
                encoding: ColumnEncoding::Ip,
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let ip = std::net::Ipv4Addr::new(127, 0, 0, 1)
            .to_ipv6_mapped()
            .octets();
        let row = vec![ValueRef::Ip(ip)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("127.0.0.1"));
    }

    #[test]
    fn ip_value_ref_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "ip".to_string(),
                logical_type: IngestedType::Ip,
                encoding: ColumnEncoding::Ip,
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let ip = std::net::Ipv6Addr::LOCALHOST.octets();
        let row = vec![ValueRef::Ip(ip)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut scratch = RowPackDecodeScratch::default();
        let decoded = decoder
            .decode_column_value(&encoded, 0, &mut scratch)
            .unwrap();
        match decoded {
            ValueRef::Ip(bytes) => assert_eq!(bytes, ip),
            other => panic!("unexpected value ref: {:?}", other),
        }
    }

    #[test]
    fn small_int_encodings() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![
                ColumnContext {
                    name: "i8".to_string(),
                    logical_type: IngestedType::Int8,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: false,
                        bits: 8,
                    },
                },
                ColumnContext {
                    name: "u8".to_string(),
                    logical_type: IngestedType::UInt8,
                    encoding: ColumnEncoding::Int {
                        signed: false,
                        varint: false,
                        bits: 8,
                    },
                },
                ColumnContext {
                    name: "i16".to_string(),
                    logical_type: IngestedType::Int16,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: false,
                        bits: 16,
                    },
                },
                ColumnContext {
                    name: "u16".to_string(),
                    logical_type: IngestedType::UInt16,
                    encoding: ColumnEncoding::Int {
                        signed: false,
                        varint: false,
                        bits: 16,
                    },
                },
            ],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![
            ValueRef::Int32(-7),
            ValueRef::UInt32(200),
            ValueRef::Int32(32000),
            ValueRef::UInt32(65000),
        ];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("-7"));
        assert!(json.to_string().contains("200"));
        assert!(json.to_string().contains("32000"));
        assert!(json.to_string().contains("65000"));
    }

    #[test]
    fn varint_int32_negative() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "i32".to_string(),
                logical_type: IngestedType::Int32,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: true,
                    bits: 32,
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Int32(-12345)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("-12345"));
    }

    #[test]
    fn lz4_encoding_selected() {
        let mut config = RowPackConfig::default();
        config.string_long_min_len = Some(8);
        config.string_long_min_ratio = Some(0.9);
        config.max_dict_entries = Some(0);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "text".to_string(),
                logical_type: IngestedType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            config,
        );
        let value = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        builder.observe_row(&[ValueRef::String(&value)]).unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::Lz4,
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String(&value)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("aaaaaaaa"));
    }

    #[test]
    fn lz4_falls_back_to_raw() {
        let mut config = RowPackConfig::default();
        config.string_long_min_len = Some(8);
        config.string_long_min_ratio = Some(0.5);
        config.max_dict_entries = Some(0);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "text".to_string(),
                logical_type: IngestedType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            config,
        );
        let value = "x3f9A2kLmN8pQrStUvWxYz0123456789-longer-string".to_string();
        builder.observe_row(&[ValueRef::String(&value)]).unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::Raw,
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String(&value)];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("x3f9A2k"));
    }

    #[test]
    fn dict_fallback_string() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "name".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Dict {
                        values: vec![b"alice".to_vec()],
                    },
                },
            }],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("carol")];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("carol"));
    }

    #[test]
    fn null_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![
                ColumnContext {
                    name: "id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "note".to_string(),
                    logical_type: IngestedType::String,
                    encoding: ColumnEncoding::String {
                        encoding: StringEncoding::Raw,
                    },
                },
            ],
            estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Null, ValueRef::Null];
        let encoded = encode_row(&encoder, &row);
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("null"));
    }

    #[test]
    fn decode_rejects_oversize_string() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "name".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Raw,
                },
            }],
            estimated_row_bytes: 0,
        };
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut tags = BitWriter::new();
        tags.write_bits(1, 2);
        let mut row = tags.finish();
        row.push(0x80);
        row.push(0x80);
        row.push(0x80);
        row.push(0x80);
        row.push(0x10);
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("too large"));
    }

    #[test]
    fn decode_rejects_oversize_array() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "embed".to_string(),
                logical_type: IngestedType::Float32Array,
                encoding: ColumnEncoding::FloatArray {
                    storage: FloatStorage::F32,
                },
            }],
            estimated_row_bytes: 0,
        };
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut tags = BitWriter::new();
        tags.write_bits(1, 2);
        let mut row = tags.finish();
        row.push(0xff);
        row.push(0xff);
        row.push(0xff);
        row.push(0xff);
        row.push(0x0f);
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("array length"));
    }

    #[test]
    fn decode_rejects_truncated_bool_bitstream() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "active".to_string(),
                logical_type: IngestedType::Bool,
                encoding: ColumnEncoding::Bool,
            }],
            estimated_row_bytes: 0,
        };
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut tags = BitWriter::new();
        tags.write_bits(1, 2);
        let row = tags.finish();
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("truncated"));
    }

    #[test]
    fn decode_rejects_invalid_tag() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "id".to_string(),
                logical_type: IngestedType::Int64,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: true,
                    bits: 64,
                },
            }],
            estimated_row_bytes: 0,
        };
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut tags = BitWriter::new();
        tags.write_bits(3, 2);
        let row = tags.finish();
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("invalid tag"));
    }

    #[test]
    fn decode_rejects_corrupt_lz4() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "text".to_string(),
                logical_type: IngestedType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Lz4,
                },
            }],
            estimated_row_bytes: 0,
        };
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut tags = BitWriter::new();
        tags.write_bits(1, 2);
        let mut row = tags.finish();
        row.push(0x05);
        row.push(0x05);
        row.extend_from_slice(&[0x00, 0x01, 0x02, 0x03, 0x04]);
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("lz4"));
    }
}
